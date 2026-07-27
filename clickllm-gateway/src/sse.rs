//! Server-Sent Events framing for streaming completions.
//!
//! The gateway must observe a token stream **without buffering it**: bytes are
//! forwarded to the client as they arrive, and the same bytes are inspected in
//! flight so we can meter tokens and score a shadow response. That means an
//! incremental parser that tolerates frames split across arbitrary chunk
//! boundaries — a network read has no obligation to align with an SSE frame.

use bytes::{Bytes, BytesMut};

/// One decoded SSE data payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// A `data:` payload that is not the terminator.
    Data(String),
    /// The `data: [DONE]` sentinel that ends an OpenAI-style stream.
    Done,
}

/// Incremental SSE decoder.
///
/// Feed it whatever bytes arrive; it yields whole events and retains any partial
/// tail for the next call.
#[derive(Debug, Default)]
pub struct Decoder {
    buf: BytesMut,
    /// Set after an oversized partial frame is discarded. The next complete frame
    /// is dropped rather than emitted, because its leading bytes went with it and
    /// parsing the remainder would surface a corrupt payload as if it were real.
    desynced: bool,
    dropped_frames: u64,
}

/// Cap on a single retained frame. A server that never sends a frame terminator
/// must not be able to grow this buffer without bound — a proxy that OOMs under a
/// hostile upstream is worse than one that gives up on the stream.
///
/// This is **enforced**, not merely observed: [`Decoder::push`] discards the
/// partial frame once the cap is passed. Detection alone would leave the
/// invariant documented but untrue.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

/// Bytes retained across a discard so a terminator straddling the cut is still
/// found. The longest terminator is `\n\r\n`, so two trailing bytes suffice.
const RESYNC_TAIL: usize = 2;

impl Decoder {
    /// New decoder with an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes currently held pending a frame terminator.
    pub fn pending(&self) -> usize {
        self.buf.len()
    }

    /// Number of frames discarded because they exceeded [`MAX_FRAME_BYTES`].
    ///
    /// Non-zero means this stream lost metering data — the client's bytes are
    /// unaffected, since the proxy forwards them independently of decoding.
    pub fn dropped_frames(&self) -> u64 {
        self.dropped_frames
    }

    /// True while resynchronising after a discard.
    pub fn desynced(&self) -> bool {
        self.desynced
    }

    /// Push received bytes and drain every complete event.
    ///
    /// Frames are separated by a blank line (`\n\n`, or `\r\n\r\n`). Comment lines
    /// (`:`) and non-`data:` fields are skipped — we care only about payloads.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<Event> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();

        while let Some(end) = find_frame_end(&self.buf) {
            let frame = self.buf.split_to(end.frame_len);
            if self.desynced {
                // The head of this frame was discarded with the oversized one.
                // Emitting the tail would present a truncated payload as whole.
                self.desynced = false;
                continue;
            }
            let body = frame.get(..end.content_len).unwrap_or_default();
            if let Some(ev) = parse_frame(body) {
                out.push(ev);
            }
        }

        // Enforce the cap. Everything left here is a partial frame with no
        // terminator; past the cap it is not a legitimate SSE frame, so drop it
        // and keep only enough tail to spot a terminator spanning the cut.
        if self.buf.len() > MAX_FRAME_BYTES {
            let keep = self.buf.len().saturating_sub(RESYNC_TAIL);
            let _ = self.buf.split_to(keep);
            self.desynced = true;
            self.dropped_frames = self.dropped_frames.saturating_add(1);
            tracing::warn!(
                cap = MAX_FRAME_BYTES,
                dropped_frames = self.dropped_frames,
                "discarded an oversized SSE frame; metering will miss it"
            );
        }
        out
    }
}

struct FrameEnd {
    /// Bytes to consume, including the terminator.
    frame_len: usize,
    /// Bytes of content, excluding the terminator.
    content_len: usize,
}

fn find_frame_end(buf: &[u8]) -> Option<FrameEnd> {
    for i in 0..buf.len() {
        if buf.get(i) != Some(&b'\n') {
            continue;
        }
        // "\n\n"
        if buf.get(i + 1) == Some(&b'\n') {
            return Some(FrameEnd {
                frame_len: i + 2,
                content_len: i,
            });
        }
        // "\n\r\n"
        if buf.get(i + 1) == Some(&b'\r') && buf.get(i + 2) == Some(&b'\n') {
            return Some(FrameEnd {
                frame_len: i + 3,
                content_len: i,
            });
        }
    }
    None
}

fn parse_frame(body: &[u8]) -> Option<Event> {
    let text = std::str::from_utf8(body).ok()?;
    let mut data = String::new();
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        // A multi-line data field is concatenated with newlines, per the spec.
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data.is_empty() {
        return None;
    }
    if data.trim() == "[DONE]" {
        return Some(Event::Done);
    }
    Some(Event::Data(data))
}

/// Render an event back onto the wire.
///
/// Used only for synthesised frames — a passthrough forwards the original bytes
/// untouched so we can never corrupt a stream by re-encoding it.
pub fn encode(event: &Event) -> Bytes {
    match event {
        Event::Data(d) => Bytes::from(format!("data: {d}\n\n")),
        Event::Done => Bytes::from_static(b"data: [DONE]\n\n"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_simple_stream() {
        let mut d = Decoder::new();
        let evs = d.push(b"data: {\"a\":1}\n\ndata: [DONE]\n\n");
        assert_eq!(evs, vec![Event::Data("{\"a\":1}".into()), Event::Done]);
    }

    #[test]
    fn frames_split_across_chunk_boundaries_survive() {
        // The failure this guards: a network read landing mid-frame must not
        // drop or duplicate the event.
        let whole = b"data: {\"tok\":\"hello\"}\n\ndata: [DONE]\n\n";
        for split in 1..whole.len() {
            let mut d = Decoder::new();
            let mut evs = d.push(&whole[..split]);
            evs.extend(d.push(&whole[split..]));
            assert_eq!(
                evs,
                vec![Event::Data("{\"tok\":\"hello\"}".into()), Event::Done],
                "split at {split} lost or duplicated an event"
            );
        }
    }

    #[test]
    fn byte_at_a_time_delivery_still_decodes() {
        let whole = b"data: one\n\ndata: two\n\ndata: [DONE]\n\n";
        let mut d = Decoder::new();
        let mut evs = Vec::new();
        for b in whole {
            evs.extend(d.push(&[*b]));
        }
        assert_eq!(
            evs,
            vec![
                Event::Data("one".into()),
                Event::Data("two".into()),
                Event::Done
            ]
        );
        assert_eq!(
            d.pending(),
            0,
            "nothing should be retained at end of stream"
        );
    }

    #[test]
    fn crlf_terminators_are_accepted() {
        let mut d = Decoder::new();
        assert_eq!(
            d.push(b"data: x\r\n\r\n"),
            vec![Event::Data("x".into())],
            "SSE permits CRLF; upstreams do use it"
        );
    }

    #[test]
    fn comments_and_other_fields_are_ignored() {
        let mut d = Decoder::new();
        let evs = d.push(b": keep-alive\n\nevent: ping\nid: 7\n\ndata: real\n\n");
        assert_eq!(evs, vec![Event::Data("real".into())]);
    }

    #[test]
    fn multi_line_data_fields_concatenate() {
        let mut d = Decoder::new();
        assert_eq!(
            d.push(b"data: line1\ndata: line2\n\n"),
            vec![Event::Data("line1\nline2".into())]
        );
    }

    #[test]
    fn optional_space_after_the_colon_is_stripped_once() {
        let mut d = Decoder::new();
        // "data:  x" has one leading space left after stripping the optional one.
        assert_eq!(d.push(b"data:x\n\n"), vec![Event::Data("x".into())]);
        assert_eq!(d.push(b"data:  x\n\n"), vec![Event::Data(" x".into())]);
    }

    #[test]
    fn a_partial_trailing_frame_is_retained_not_emitted() {
        let mut d = Decoder::new();
        assert!(d.push(b"data: incomplete").is_empty());
        assert!(d.pending() > 0);
        assert_eq!(d.push(b"\n\n"), vec![Event::Data("incomplete".into())]);
    }

    #[test]
    fn an_upstream_that_never_terminates_a_frame_cannot_grow_the_buffer() {
        // The invariant the module documents. Detection alone left it documented
        // but untrue — which is exactly what code review caught.
        let mut d = Decoder::new();
        let chunk = vec![b'x'; 64 * 1024];
        for _ in 0..200 {
            assert!(d.push(&chunk).is_empty());
            assert!(
                d.pending() <= MAX_FRAME_BYTES,
                "buffer reached {} bytes, past the {MAX_FRAME_BYTES} cap",
                d.pending()
            );
        }
        // 12.5 MiB pushed at a 1 MiB cap; memory stayed bounded throughout.
        assert!(d.dropped_frames() > 0);
    }

    #[test]
    fn the_decoder_resynchronises_after_discarding_an_oversized_frame() {
        let mut d = Decoder::new();
        d.push(&vec![b'x'; MAX_FRAME_BYTES + 1]);
        assert!(d.desynced());

        // The remainder of the bad frame is dropped, not emitted as if whole.
        assert!(d.push(b"tail-of-the-bad-frame\n\n").is_empty());
        assert!(
            !d.desynced(),
            "should have resynchronised at the terminator"
        );

        // ...and the next real frame decodes normally.
        assert_eq!(d.push(b"data: good\n\n"), vec![Event::Data("good".into())]);
    }

    #[test]
    fn a_discard_never_emits_a_truncated_payload_as_if_it_were_whole() {
        let mut d = Decoder::new();
        let mut big = b"data: ".to_vec();
        big.extend(std::iter::repeat_n(b'x', MAX_FRAME_BYTES));
        d.push(&big);
        // The tail of the oversized frame must not surface as an Event.
        let evs = d.push(b"trailing-garbage\n\ndata: clean\n\n");
        assert_eq!(evs, vec![Event::Data("clean".into())]);
    }

    #[test]
    fn invalid_utf8_drops_the_frame_rather_than_panicking() {
        let mut d = Decoder::new();
        let mut bad = b"data: ".to_vec();
        bad.extend_from_slice(&[0xff, 0xfe]);
        bad.extend_from_slice(b"\n\n");
        assert!(d.push(&bad).is_empty());
        // ...and the decoder keeps working afterwards.
        assert_eq!(d.push(b"data: ok\n\n"), vec![Event::Data("ok".into())]);
    }

    #[test]
    fn encode_round_trips_through_the_decoder() {
        for ev in [Event::Data("{\"x\":1}".into()), Event::Done] {
            let mut d = Decoder::new();
            assert_eq!(d.push(&encode(&ev)), vec![ev.clone()]);
        }
    }
}
