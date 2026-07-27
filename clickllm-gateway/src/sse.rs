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
    /// Sticky once a retained partial frame has ever exceeded [`MAX_FRAME_BYTES`].
    overflowed: bool,
}

/// Cap on a single retained frame. A server that never sends a frame terminator
/// must not be able to grow this buffer without bound — a proxy that OOMs under a
/// hostile upstream is worse than one that gives up on the stream. Enforced in
/// [`Decoder::push`], which drops the buffered partial frame once it is exceeded.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

impl Decoder {
    /// New decoder with an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes currently held pending a frame terminator.
    pub fn pending(&self) -> usize {
        self.buf.len()
    }

    /// True once a retained partial frame has ever exceeded [`MAX_FRAME_BYTES`].
    ///
    /// Sticky for the life of the decoder: the offending bytes are dropped as soon
    /// as the cap is hit (see [`Decoder::push`]), so `pending()` alone can't be
    /// used to detect this after the fact.
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Push received bytes and drain every complete event.
    ///
    /// Frames are separated by a blank line (`\n\n`, or `\r\n\r\n`). Comment lines
    /// (`:`) and non-`data:` fields are skipped — we care only about payloads.
    ///
    /// If no terminator shows up before the retained partial frame exceeds
    /// [`MAX_FRAME_BYTES`], the buffered bytes are dropped so the decoder's memory
    /// use stays capped regardless of how long the misbehaving upstream keeps
    /// withholding a terminator.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<Event> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();

        while let Some(end) = find_frame_end(&self.buf) {
            let frame = self.buf.split_to(end.frame_len);
            let body = frame.get(..end.content_len).unwrap_or_default();
            if let Some(ev) = parse_frame(body) {
                out.push(ev);
            }
        }

        if self.buf.len() > MAX_FRAME_BYTES {
            self.overflowed = true;
            self.buf.clear();
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
    fn overflow_is_detectable_so_a_hostile_upstream_cannot_grow_us_without_bound() {
        let mut d = Decoder::new();
        assert!(!d.overflowed());
        d.push(&vec![b'x'; MAX_FRAME_BYTES + 1]);
        assert!(
            d.overflowed(),
            "a frame with no terminator must be detectable"
        );
    }

    #[test]
    fn overflow_is_enforced_the_buffer_is_capped_not_just_flagged() {
        let mut d = Decoder::new();
        d.push(&vec![b'x'; MAX_FRAME_BYTES + 1]);
        assert!(
            d.pending() <= MAX_FRAME_BYTES,
            "the offending frame must be dropped, not merely detected, once it \
             exceeds the cap"
        );

        // A hostile upstream that keeps withholding a terminator forever must not
        // be able to grow the buffer chunk after chunk either.
        for _ in 0..8 {
            d.push(&vec![b'x'; MAX_FRAME_BYTES]);
            assert!(
                d.pending() <= MAX_FRAME_BYTES,
                "buffer grew unbounded across repeated pushes from a stalled upstream"
            );
        }
        assert!(d.overflowed());
    }

    #[test]
    fn a_terminator_after_an_overflow_resumes_normal_decoding() {
        let mut d = Decoder::new();
        d.push(&vec![b'x'; MAX_FRAME_BYTES + 1]);
        assert!(d.overflowed());
        // The dropped bytes leave no dangling frame behind; decoding resumes as
        // normal for whatever the upstream sends next.
        assert_eq!(d.push(b"data: ok\n\n"), vec![Event::Data("ok".into())]);
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
