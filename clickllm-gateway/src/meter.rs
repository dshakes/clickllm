//! Token and cost metering.
//!
//! Every request that crosses the gateway is metered, because the cost ledger is
//! what makes stage ① useful on day one — before anything has been proved or
//! migrated. Two constraints shape this module:
//!
//! 1. **Metering must not buffer.** A streaming response is forwarded as it
//!    arrives; usage is extracted from frames in flight.
//! 2. **A missing count is reported as missing, never as zero.** Some upstreams
//!    omit `usage` on streamed responses unless asked. Silently recording zero
//!    would understate spend, which is the one direction a cost tool must never
//!    err in.

use serde::{Deserialize, Serialize};

use crate::sse::Event;

/// Token counts for one exchange.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the request.
    pub prompt_tokens: u64,
    /// Tokens generated.
    pub completion_tokens: u64,
}

impl Usage {
    /// Total tokens billed.
    pub fn total(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    /// Cost in micro-dollars (1e-6 USD), given per-million-token rates.
    ///
    /// Integer micro-dollars rather than floats: a ledger that drifts by
    /// accumulated rounding is not a ledger. Rates are per million tokens.
    pub fn cost_micros(&self, prompt_per_mtok_micros: u64, completion_per_mtok_micros: u64) -> u64 {
        let p = self
            .prompt_tokens
            .saturating_mul(prompt_per_mtok_micros)
            .saturating_div(1_000_000);
        let c = self
            .completion_tokens
            .saturating_mul(completion_per_mtok_micros)
            .saturating_div(1_000_000);
        p.saturating_add(c)
    }
}

/// What the meter could determine about a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Metered {
    /// The upstream reported usage.
    Reported(Usage),
    /// The upstream reported nothing. Carries the number of streamed content
    /// deltas seen, which is a *lower bound* on completion tokens — never
    /// presented as the real count.
    Unreported {
        /// Content deltas observed in the stream.
        deltas: u64,
    },
}

impl Metered {
    /// Usage if the upstream actually reported it.
    pub fn usage(&self) -> Option<Usage> {
        match self {
            Self::Reported(u) => Some(*u),
            Self::Unreported { .. } => None,
        }
    }

    /// True when the figure is authoritative rather than inferred.
    pub fn is_reported(&self) -> bool {
        matches!(self, Self::Reported(_))
    }
}

/// Accumulates usage across a streamed or unary response.
#[derive(Debug, Default)]
pub struct Meter {
    usage: Option<Usage>,
    deltas: u64,
}

impl Meter {
    /// New, empty meter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one decoded SSE event. Cheap and allocation-light: this runs on
    /// the hot path for every frame of every streamed response.
    pub fn observe(&mut self, event: &Event) {
        let Event::Data(payload) = event else {
            return;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            // A frame we cannot parse is not a reason to fail the request; the
            // client still gets its bytes. We just cannot meter it.
            return;
        };
        if let Some(u) = usage_from(&v) {
            self.usage = Some(u);
        }
        if has_content_delta(&v) {
            self.deltas = self.deltas.saturating_add(1);
        }
    }

    /// Observe a complete non-streaming response body.
    pub fn observe_body(&mut self, body: &[u8]) {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(body)
            && let Some(u) = usage_from(&v)
        {
            self.usage = Some(u);
        }
    }

    /// Final verdict, without consuming the meter.
    ///
    /// The streaming path accumulates behind a lock and reads the result from a
    /// drop hook, so it needs a borrow rather than a move.
    pub fn snapshot(&self) -> Metered {
        match self.usage {
            Some(u) => Metered::Reported(u),
            None => Metered::Unreported {
                deltas: self.deltas,
            },
        }
    }

    /// Final verdict.
    pub fn finish(self) -> Metered {
        self.snapshot()
    }
}

/// Extract `usage` from an OpenAI-shaped payload.
///
/// A `usage` object present but null — which some upstreams emit on every
/// streamed frame until the last — must not be read as zeros.
fn usage_from(v: &serde_json::Value) -> Option<Usage> {
    let u = v.get("usage")?;
    if u.is_null() {
        return None;
    }
    let prompt = u.get("prompt_tokens")?.as_u64()?;
    let completion = u.get("completion_tokens")?.as_u64()?;
    Some(Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
    })
}

/// True when a streamed chunk carries generated content.
fn has_content_delta(v: &serde_json::Value) -> bool {
    v.get("choices")
        .and_then(|c| c.as_array())
        .is_some_and(|choices| {
            choices.iter().any(|c| {
                c.get("delta")
                    .and_then(|d| d.get("content"))
                    .is_some_and(|c| c.as_str().is_some_and(|s| !s.is_empty()))
            })
        })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn ev(s: &str) -> Event {
        Event::Data(s.to_owned())
    }

    #[test]
    fn reads_usage_from_a_streamed_final_frame() {
        let mut m = Meter::new();
        m.observe(&ev(r#"{"choices":[{"delta":{"content":"hi"}}]}"#));
        m.observe(&ev(
            r#"{"usage":{"prompt_tokens":12,"completion_tokens":30}}"#,
        ));
        let got = m.finish();
        assert!(got.is_reported());
        assert_eq!(
            got.usage().unwrap(),
            Usage {
                prompt_tokens: 12,
                completion_tokens: 30
            }
        );
    }

    #[test]
    fn missing_usage_is_reported_as_missing_never_as_zero() {
        // Understating spend is the one direction a cost ledger must not err in.
        let mut m = Meter::new();
        for _ in 0..5 {
            m.observe(&ev(r#"{"choices":[{"delta":{"content":"x"}}]}"#));
        }
        match m.finish() {
            Metered::Unreported { deltas } => assert_eq!(deltas, 5),
            Metered::Reported(u) => panic!("fabricated a usage figure: {u:?}"),
        }
    }

    #[test]
    fn a_null_usage_field_does_not_read_as_zeros() {
        let mut m = Meter::new();
        m.observe(&ev(
            r#"{"usage":null,"choices":[{"delta":{"content":"a"}}]}"#,
        ));
        assert!(!m.finish().is_reported());
    }

    #[test]
    fn a_partial_usage_object_is_rejected_rather_than_half_counted() {
        let mut m = Meter::new();
        m.observe(&ev(r#"{"usage":{"prompt_tokens":10}}"#));
        assert!(!m.finish().is_reported(), "completion_tokens was absent");
    }

    #[test]
    fn unparseable_frames_do_not_break_metering() {
        let mut m = Meter::new();
        m.observe(&ev("not json at all"));
        m.observe(&ev(
            r#"{"usage":{"prompt_tokens":1,"completion_tokens":2}}"#,
        ));
        assert_eq!(m.finish().usage().unwrap().total(), 3);
    }

    #[test]
    fn the_done_sentinel_is_not_counted_as_content() {
        let mut m = Meter::new();
        m.observe(&Event::Done);
        match m.finish() {
            Metered::Unreported { deltas } => assert_eq!(deltas, 0),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn empty_content_deltas_are_not_counted() {
        let mut m = Meter::new();
        m.observe(&ev(r#"{"choices":[{"delta":{"content":""}}]}"#));
        m.observe(&ev(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#));
        match m.finish() {
            Metered::Unreported { deltas } => assert_eq!(deltas, 0),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn later_usage_wins_so_a_final_frame_corrects_an_earlier_estimate() {
        let mut m = Meter::new();
        m.observe(&ev(
            r#"{"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
        ));
        m.observe(&ev(
            r#"{"usage":{"prompt_tokens":1,"completion_tokens":99}}"#,
        ));
        assert_eq!(m.finish().usage().unwrap().completion_tokens, 99);
    }

    #[test]
    fn unary_bodies_are_metered_too() {
        let mut m = Meter::new();
        m.observe_body(br#"{"usage":{"prompt_tokens":7,"completion_tokens":8}}"#);
        assert_eq!(m.finish().usage().unwrap().total(), 15);
    }

    #[test]
    fn cost_is_integer_micros_and_does_not_drift() {
        let u = Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 500_000,
        };
        // $3.00/Mtok prompt, $15.00/Mtok completion
        assert_eq!(u.cost_micros(3_000_000, 15_000_000), 3_000_000 + 7_500_000);
        // Summing many small requests must equal one big one — no float drift.
        let small = Usage {
            prompt_tokens: 1_000,
            completion_tokens: 0,
        };
        let summed: u64 = (0..1000).map(|_| small.cost_micros(3_000_000, 0)).sum();
        assert_eq!(summed, u.cost_micros(3_000_000, 0));
    }

    #[test]
    fn cost_and_total_saturate_rather_than_wrapping() {
        let u = Usage {
            prompt_tokens: u64::MAX,
            completion_tokens: u64::MAX,
        };
        assert_eq!(u.total(), u64::MAX);
        assert!(
            u.cost_micros(u64::MAX, u64::MAX) > 0,
            "must not wrap to a small number"
        );
    }
}
