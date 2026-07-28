//! Live telemetry — what the engine is actually doing, right now.
//!
//! The gateway already knows what *it* did: requests, latency, tokens, which
//! backend served them. That is half the picture and the less useful half. The
//! number that decides whether a deployment is healthy lives inside the engine:
//!
//! > **KV cache utilisation.** At 60% you have headroom. At 95% the scheduler is
//! > preempting requests to make room, and the p99 you are seeing is not a model
//! > problem or a network problem — it is that number.
//!
//! Nothing in the gateway can compute it. It has to be read from the engine, so
//! this module scrapes the Prometheus endpoint every serving engine already
//! exposes and parses the handful of series that matter.
//!
//! ## Metric names are verified, not remembered
//!
//! Every name below was checked against vLLM's published metrics reference
//! (see [`SOURCE`]). That is not ceremony: this repo shipped
//! `--guided-decoding-backend` from memory, a flag that had been renamed, and the
//! configs it generated could not start. Metric names drift the same way —
//! `vllm:num_requests_swapped` and `vllm:cpu_cache_usage_perc` are already gone
//! in V1, and a dashboard still asking for them shows a confident blank.
//!
//! ## Absent is not zero
//!
//! Every field is an `Option`. An engine that does not publish a series yields
//! `None`, which renders as `—`, never as `0`.
//!
//! This is the same rule [`crate::meter::Metered::Unreported`] follows for token
//! counts, and it matters more here: a KV utilisation of `0%` reads as *lots of
//! headroom*, which is the exact opposite of *we could not tell*. A monitoring
//! surface that cannot distinguish those two is worse than no monitoring
//! surface, because someone will trust it during an incident.

use std::time::Duration;

use serde::Serialize;

/// Where the metric names were verified.
pub const SOURCE: &str = "https://docs.vllm.ai/en/stable/design/metrics.html (checked 2026-07-27)";

/// How long to wait on an engine's metrics endpoint.
///
/// Short on purpose: this is a dashboard poll, and a slow scrape must degrade to
/// "unavailable" rather than holding a connection while an operator waits.
const SCRAPE_TIMEOUT: Duration = Duration::from_millis(1500);

/// One reading from a serving engine.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Snapshot {
    /// Fraction of KV blocks in use, `0.0..=1.0`. The number that explains most
    /// unexplained latency.
    pub kv_cache_used: Option<f64>,
    /// Requests currently decoding.
    pub running: Option<f64>,
    /// Requests admitted but not yet running. Sustained non-zero means the
    /// engine is the bottleneck, not the client.
    pub waiting: Option<f64>,
    /// Mean time to first token, milliseconds.
    pub ttft_ms: Option<f64>,
    /// Mean inter-token latency, milliseconds.
    pub itl_ms: Option<f64>,
    /// Prompt tokens processed since start.
    pub prompt_tokens: Option<f64>,
    /// Generated tokens since start.
    pub generation_tokens: Option<f64>,
    /// Prefix cache hit rate, `0.0..=1.0`, when both counters are published.
    pub prefix_hit_rate: Option<f64>,
    /// Why this scrape failed, when it did. Present means every field above is
    /// `None` because we could not look — not because the engine is idle.
    pub error: Option<String>,
}

impl Snapshot {
    /// Whether anything at all was read.
    pub fn any(&self) -> bool {
        self.kv_cache_used.is_some() || self.running.is_some() || self.ttft_ms.is_some()
    }

    /// Whether the KV cache is close enough to full that the scheduler is
    /// likely preempting.
    ///
    /// `None` when unknown — deliberately not `false`, so a caller cannot read
    /// "we did not measure" as "we are fine".
    pub fn kv_pressure(&self) -> Option<bool> {
        self.kv_cache_used.map(|v| v >= 0.90)
    }
}

/// Parsed Prometheus exposition text.
///
/// A tiny parser rather than a dependency: we need eight series out of a few
/// hundred lines, and the format is `name{labels} value`.
struct Series<'a>(&'a str);

impl<'a> Series<'a> {
    /// Sum of every sample whose name matches `metric`, across label sets.
    ///
    /// Summed rather than first-match because these endpoints are labelled by
    /// `model_name`, and a gateway pointed at a multi-model server would
    /// otherwise silently report one model's numbers as the whole fleet's.
    fn sum(&self, metric: &str) -> Option<f64> {
        let mut total = None;
        for line in self.0.lines() {
            let line = line.trim();
            if line.starts_with('#') {
                continue;
            }
            let Some(rest) = line.strip_prefix(metric) else {
                continue;
            };
            // Guard against `foo_total` matching a request for `foo`.
            let rest = match rest.chars().next() {
                None | Some(' ') => rest,
                Some('{') => match rest.find('}') {
                    Some(i) => rest.get(i + 1..).unwrap_or_default(),
                    None => continue,
                },
                _ => continue,
            };
            if let Some(v) = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok())
            {
                *total.get_or_insert(0.0) += v;
            }
        }
        total
    }

    /// Mean of a histogram, in seconds, from its `_sum` and `_count`.
    ///
    /// `None` when the count is zero: dividing by it would produce a confident
    /// `0ms` for an engine that has served nothing.
    fn histogram_mean(&self, metric: &str) -> Option<f64> {
        let sum = self.sum(&format!("{metric}_sum"))?;
        let count = self.sum(&format!("{metric}_count"))?;
        (count > 0.0).then_some(sum / count)
    }
}

/// Parse an engine's Prometheus text into a snapshot.
///
/// Split from the fetch so the parsing is testable against real exposition text
/// without a server.
pub fn parse(text: &str) -> Snapshot {
    let s = Series(text);
    let (queries, hits) = (
        s.sum("vllm:prefix_cache_queries"),
        s.sum("vllm:prefix_cache_hits"),
    );
    Snapshot {
        kv_cache_used: s.sum("vllm:kv_cache_usage_perc"),
        running: s.sum("vllm:num_requests_running"),
        waiting: s.sum("vllm:num_requests_waiting"),
        ttft_ms: s
            .histogram_mean("vllm:time_to_first_token_seconds")
            .map(|v| v * 1000.0),
        itl_ms: s
            .histogram_mean("vllm:inter_token_latency_seconds")
            .map(|v| v * 1000.0),
        prompt_tokens: s.sum("vllm:prompt_tokens_total"),
        generation_tokens: s.sum("vllm:generation_tokens_total"),
        // Rate rather than the raw counters: vLLM removed its hit-rate gauge and
        // expects it computed, so computing it here keeps the caller honest.
        prefix_hit_rate: match (queries, hits) {
            (Some(q), Some(h)) if q > 0.0 => Some(h / q),
            _ => None,
        },
        error: None,
    }
}

/// Scrape `base_url`'s `/metrics` endpoint.
///
/// Never returns an error: a monitoring call that fails should render as
/// "unavailable" in the surface that asked, not propagate and take the page
/// down with it. The reason lands in [`Snapshot::error`].
pub async fn scrape(client: &reqwest::Client, base_url: &str) -> Snapshot {
    let url = format!("{}/metrics", base_url.trim_end_matches('/'));
    let fail = |reason: String| Snapshot {
        error: Some(reason),
        ..Default::default()
    };

    match client.get(&url).timeout(SCRAPE_TIMEOUT).send().await {
        Ok(resp) if resp.status().is_success() => match resp.text().await {
            Ok(body) => {
                let snap = parse(&body);
                if snap.any() {
                    snap
                } else {
                    // A 200 that parsed to nothing is its own failure mode: the
                    // endpoint exists but publishes none of the series we know.
                    fail(format!(
                        "{url} responded but published none of the expected series; \
                         the engine may not be vLLM, or the names have drifted"
                    ))
                }
            }
            Err(e) => fail(format!("could not read {url}: {e}")),
        },
        Ok(resp) => fail(format!("{url} returned HTTP {}", resp.status())),
        Err(e) if e.is_timeout() => fail(format!("{url} did not answer within {SCRAPE_TIMEOUT:?}")),
        Err(e) => fail(format!("could not reach {url}: {e}")),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// Shaped like real vLLM exposition output, including the noise.
    const SAMPLE: &str = r#"
# HELP vllm:kv_cache_usage_perc Fraction of used KV cache blocks
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="Qwen/Qwen3-32B"} 0.73
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen3-32B"} 12.0
vllm:num_requests_waiting{model_name="Qwen/Qwen3-32B"} 3.0
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{model_name="Qwen/Qwen3-32B"} 42.0
vllm:time_to_first_token_seconds_count{model_name="Qwen/Qwen3-32B"} 200.0
vllm:inter_token_latency_seconds_sum{model_name="Qwen/Qwen3-32B"} 18.0
vllm:inter_token_latency_seconds_count{model_name="Qwen/Qwen3-32B"} 1200.0
vllm:prompt_tokens_total{model_name="Qwen/Qwen3-32B"} 91000.0
vllm:generation_tokens_total{model_name="Qwen/Qwen3-32B"} 15000.0
vllm:prefix_cache_queries{model_name="Qwen/Qwen3-32B"} 1000.0
vllm:prefix_cache_hits{model_name="Qwen/Qwen3-32B"} 620.0
"#;

    #[test]
    fn a_real_looking_scrape_parses_every_series_we_claim() {
        let s = parse(SAMPLE);
        assert_eq!(s.kv_cache_used, Some(0.73));
        assert_eq!(s.running, Some(12.0));
        assert_eq!(s.waiting, Some(3.0));
        assert_eq!(s.ttft_ms, Some(210.0), "42s over 200 requests");
        assert_eq!(s.itl_ms, Some(15.0), "18s over 1200 tokens");
        assert_eq!(s.prompt_tokens, Some(91_000.0));
        assert_eq!(s.prefix_hit_rate, Some(0.62));
        assert!(s.error.is_none() && s.any());
    }

    #[test]
    fn an_absent_series_is_none_and_never_zero() {
        // The invariant this module exists for. `0%` KV usage reads as "lots of
        // headroom", which is the opposite of "we could not tell".
        let s = parse("vllm:num_requests_running{model_name=\"m\"} 4.0\n");
        assert_eq!(s.running, Some(4.0));
        assert_eq!(s.kv_cache_used, None, "must not fabricate a utilisation");
        assert_eq!(s.ttft_ms, None);
        assert_eq!(s.kv_pressure(), None, "unknown is not 'fine'");
    }

    #[test]
    fn a_multi_model_endpoint_reports_the_fleet_not_one_label_set() {
        let text = "vllm:num_requests_running{model_name=\"a\"} 4.0\n\
                    vllm:num_requests_running{model_name=\"b\"} 6.0\n";
        assert_eq!(parse(text).running, Some(10.0));
    }

    #[test]
    fn a_histogram_with_no_observations_is_unknown_rather_than_zero_latency() {
        let text = "vllm:time_to_first_token_seconds_sum{m=\"a\"} 0.0\n\
                    vllm:time_to_first_token_seconds_count{m=\"a\"} 0.0\n";
        assert_eq!(parse(text).ttft_ms, None, "0/0 must not render as 0ms");
    }

    #[test]
    fn a_prefix_name_does_not_match_a_longer_metric() {
        // `vllm:prompt_tokens_total` must not be picked up by a request for
        // `vllm:prompt_tokens`, or counters leak into each other.
        let text = "vllm:prompt_tokens_total{m=\"a\"} 5.0\n";
        assert_eq!(Series(text).sum("vllm:prompt_tokens"), None);
        assert_eq!(Series(text).sum("vllm:prompt_tokens_total"), Some(5.0));
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let text = "# HELP vllm:num_requests_running 999\n\n\
                    vllm:num_requests_running{m=\"a\"} 2.0\n";
        assert_eq!(parse(text).running, Some(2.0));
    }

    #[test]
    fn an_unlabelled_series_still_parses() {
        assert_eq!(parse("vllm:num_requests_running 7\n").running, Some(7.0));
    }

    #[test]
    fn kv_pressure_is_only_true_near_the_ceiling() {
        let at =
            |v: f64| parse(&format!("vllm:kv_cache_usage_perc{{m=\"a\"}} {v}\n")).kv_pressure();
        assert_eq!(at(0.60), Some(false));
        assert_eq!(at(0.89), Some(false));
        assert_eq!(at(0.95), Some(true));
    }

    #[test]
    fn garbage_parses_to_nothing_rather_than_to_wrong_numbers() {
        for text in [
            "",
            "not prometheus at all",
            "vllm:num_requests_running abc\n",
        ] {
            let s = parse(text);
            assert!(!s.any(), "{text:?} produced {s:?}");
        }
    }
}
