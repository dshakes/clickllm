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
    ///
    /// Grades the engine choice: a plan that picked SGLang for RadixAttention,
    /// or turned on prefix caching, predicted this would be high. If it is not,
    /// the prediction was wrong and the config should change.
    pub prefix_hit_rate: Option<f64>,
    /// Mean time a request waits before it starts running, milliseconds.
    ///
    /// Sustained non-zero queue time with KV headroom means the concurrency cap
    /// is the limit, not the hardware — a knob, not a purchase.
    pub queue_ms: Option<f64>,
    /// Mean prefill time, milliseconds.
    pub prefill_ms: Option<f64>,
    /// Mean decode time, milliseconds.
    ///
    /// The prefill/decode split is the diagnostic that decides which half of the
    /// stack to tune, and it is invisible from outside the engine.
    pub decode_ms: Option<f64>,
    /// Mean end-to-end request latency, milliseconds.
    pub e2e_ms: Option<f64>,
    /// Requests that finished because the model stopped, by itself.
    pub finished_stop: Option<f64>,
    /// Requests cut off at the token limit. A rising count means outputs are
    /// being truncated — usually `max_tokens` set too low, and it looks like a
    /// quality problem rather than a configuration one.
    pub finished_length: Option<f64>,
    /// Requests the client abandoned. Rising means people are giving up waiting.
    pub finished_abort: Option<f64>,
    /// Fraction of drafted tokens the target model accepted, `0.0..=1.0`.
    ///
    /// Grades the speculative-decoding decision directly. Below roughly 0.6 the
    /// draft pass is costing more than it returns, whatever the plan assumed.
    pub draft_acceptance: Option<f64>,
    /// Mean gap between reuses of a KV block, seconds. Short means prefix reuse
    /// is working; long means blocks are evicted before they are hit again.
    pub kv_reuse_gap_s: Option<f64>,
    /// LoRA adapters currently loaded, when multi-adapter serving is in use.
    pub lora_adapters: Option<f64>,
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

    /// Requests that finished, by every reason we can read.
    pub fn finished_total(&self) -> Option<f64> {
        let parts = [
            self.finished_stop,
            self.finished_length,
            self.finished_abort,
        ];
        parts
            .iter()
            .any(Option::is_some)
            .then(|| parts.iter().flatten().sum())
    }

    /// Fraction of completions cut off at the token limit.
    ///
    /// A high value looks like a quality problem — truncated answers — and is
    /// almost always `max_tokens` set too low. Worth surfacing because nothing
    /// outside the engine can distinguish the two.
    pub fn truncation_rate(&self) -> Option<f64> {
        match (self.finished_length, self.finished_total()) {
            (Some(l), Some(t)) if t > 0.0 => Some(l / t),
            _ => None,
        }
    }

    /// Fraction of requests the client abandoned.
    pub fn abandon_rate(&self) -> Option<f64> {
        match (self.finished_abort, self.finished_total()) {
            (Some(a), Some(t)) if t > 0.0 => Some(a / t),
            _ => None,
        }
    }

    /// Share of request time spent in prefill, when both halves are published.
    ///
    /// The single most useful ratio for deciding *which* half of the stack to
    /// tune, and it cannot be computed from outside the engine.
    pub fn prefill_share(&self) -> Option<f64> {
        match (self.prefill_ms, self.decode_ms) {
            (Some(p), Some(d)) if p + d > 0.0 => Some(p / (p + d)),
            _ => None,
        }
    }

    /// Check the plan's predictions against what is actually happening.
    ///
    /// This is the half of the loop that is usually missing. A planner emits a
    /// config on the strength of assumptions — this workload shares prefixes,
    /// this concurrency leaves compute idle for a drafter — and then nobody ever
    /// checks. Every finding below is a *prediction the deployment falsified*.
    ///
    /// `expect_prefix_reuse` and `expect_speculation` are what the plan assumed.
    /// Findings are returned in the order they should be acted on.
    pub fn contradictions(
        &self,
        expect_prefix_reuse: bool,
        expect_speculation: bool,
    ) -> Vec<String> {
        let mut out = Vec::new();

        if expect_speculation
            && let Some(rate) = self.draft_acceptance
            && rate < 0.6
        {
            out.push(format!(
                "speculative decoding was enabled on the assumption of idle \
                 compute, but only {:.0}% of drafted tokens are being accepted. \
                 Below ~60% the draft pass costs more than it returns — turn it \
                 off or change the drafter.",
                rate * 100.0
            ));
        }
        if expect_prefix_reuse
            && let Some(hit) = self.prefix_hit_rate
            && hit < 0.2
        {
            out.push(format!(
                "prefix reuse was configured for, but the hit rate is {:.0}%. \
                 Either the prompts share less than assumed, or blocks are being \
                 evicted before they are reused — the bookkeeping is being paid \
                 for nothing.",
                hit * 100.0
            ));
        }
        if let Some(t) = self.truncation_rate()
            && t > 0.05
        {
            out.push(format!(
                "{:.0}% of completions are stopping at the token limit rather \
                 than finishing. That reads as a quality problem and is almost \
                 always max_tokens set too low.",
                t * 100.0
            ));
        }
        if let Some(a) = self.abandon_rate()
            && a > 0.02
        {
            out.push(format!(
                "{:.0}% of requests were abandoned by the client. People are \
                 giving up before the answer arrives, which no quality metric \
                 will show you.",
                a * 100.0
            ));
        }
        if let Some(q) = self.queue_ms
            && q > 50.0
            && self.kv_cache_used.is_some_and(|kv| kv < 0.75)
        {
            out.push(format!(
                "requests wait {:.0}ms before starting while the KV cache sits at \
                 {:.0}%. The concurrency cap is the limit, not the hardware — \
                 that is a knob, not a purchase.",
                q,
                self.kv_cache_used.unwrap_or(0.0) * 100.0
            ));
        }
        out
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

    /// Sum of samples whose `label` equals `value`.
    ///
    /// `vllm:request_success_total` is only useful broken out by
    /// `finished_reason`: `length` means outputs are being truncated, `abort`
    /// means clients gave up waiting. Collapsed into one total, both look like
    /// success.
    fn sum_where(&self, metric: &str, label: &str, value: &str) -> Option<f64> {
        let needle = format!("{label}=\"{value}\"");
        let mut total = None;
        for line in self.0.lines() {
            let line = line.trim();
            if line.starts_with('#') || !line.starts_with(metric) || !line.contains(&needle) {
                continue;
            }
            if let Some(v) = line
                .split_whitespace()
                .next_back()
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
    let ms = |s: &Series<'_>, m: &str| s.histogram_mean(m).map(|v| v * 1000.0);
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
        queue_ms: ms(&s, "vllm:request_queue_time_seconds"),
        prefill_ms: ms(&s, "vllm:request_prefill_time_seconds"),
        decode_ms: ms(&s, "vllm:request_decode_time_seconds"),
        e2e_ms: ms(&s, "vllm:e2e_request_latency_seconds"),
        finished_stop: s.sum_where("vllm:request_success_total", "finished_reason", "stop"),
        finished_length: s.sum_where("vllm:request_success_total", "finished_reason", "length"),
        finished_abort: s.sum_where("vllm:request_success_total", "finished_reason", "abort"),
        // Legacy in V1 and under review upstream, so its absence is expected
        // rather than a fault. Read when present; never required.
        draft_acceptance: s.sum("vllm:spec_decode_draft_acceptance_rate"),
        kv_reuse_gap_s: s.histogram_mean("vllm:kv_block_reuse_gap_seconds"),
        lora_adapters: s.sum("vllm:lora_requests_info"),
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
vllm:request_queue_time_seconds_sum{model_name="Qwen/Qwen3-32B"} 8.0
vllm:request_queue_time_seconds_count{model_name="Qwen/Qwen3-32B"} 200.0
vllm:request_prefill_time_seconds_sum{model_name="Qwen/Qwen3-32B"} 30.0
vllm:request_prefill_time_seconds_count{model_name="Qwen/Qwen3-32B"} 200.0
vllm:request_decode_time_seconds_sum{model_name="Qwen/Qwen3-32B"} 90.0
vllm:request_decode_time_seconds_count{model_name="Qwen/Qwen3-32B"} 200.0
vllm:e2e_request_latency_seconds_sum{model_name="Qwen/Qwen3-32B"} 128.0
vllm:e2e_request_latency_seconds_count{model_name="Qwen/Qwen3-32B"} 200.0
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="Qwen/Qwen3-32B"} 170.0
vllm:request_success_total{finished_reason="length",model_name="Qwen/Qwen3-32B"} 24.0
vllm:request_success_total{finished_reason="abort",model_name="Qwen/Qwen3-32B"} 6.0
vllm:spec_decode_draft_acceptance_rate{model_name="Qwen/Qwen3-32B"} 0.41
vllm:kv_block_reuse_gap_seconds_sum{model_name="Qwen/Qwen3-32B"} 240.0
vllm:kv_block_reuse_gap_seconds_count{model_name="Qwen/Qwen3-32B"} 120.0
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
    fn the_full_verified_series_set_is_parsed_not_a_subset() {
        // The complete list from the metrics reference, minus the two removed in
        // V1. A field added to Snapshot without a parse rule fails here.
        let s = parse(SAMPLE);
        assert_eq!(s.queue_ms, Some(40.0));
        assert_eq!(s.prefill_ms, Some(150.0));
        assert_eq!(s.decode_ms, Some(450.0));
        assert_eq!(s.e2e_ms, Some(640.0));
        assert_eq!(s.finished_stop, Some(170.0));
        assert_eq!(s.finished_length, Some(24.0));
        assert_eq!(s.finished_abort, Some(6.0));
        assert_eq!(s.draft_acceptance, Some(0.41));
        assert_eq!(s.kv_reuse_gap_s, Some(2.0));
    }

    #[test]
    fn finish_reasons_are_split_because_collapsed_they_all_look_like_success() {
        let s = parse(SAMPLE);
        assert_eq!(s.finished_total(), Some(200.0));
        assert_eq!(s.truncation_rate(), Some(0.12));
        assert_eq!(s.abandon_rate(), Some(0.03));
    }

    #[test]
    fn the_prefill_decode_split_is_computed_from_both_halves() {
        // 150ms prefill against 450ms decode: a decode-bound workload, which
        // says tune bandwidth rather than prefill chunking.
        assert_eq!(parse(SAMPLE).prefill_share(), Some(0.25));
        // One half alone cannot produce a share.
        let half = "vllm:request_prefill_time_seconds_sum{m=\"a\"} 1.0\n\
                    vllm:request_prefill_time_seconds_count{m=\"a\"} 1.0\n";
        assert_eq!(parse(half).prefill_share(), None);
    }

    #[test]
    fn telemetry_contradicts_a_plan_that_predicted_wrongly() {
        // The half of the loop that is usually missing: the plan assumed spare
        // compute for a drafter and shared prefixes. The deployment says
        // otherwise, and saying so is the point.
        let s = parse(SAMPLE);
        let found = s.contradictions(true, true);
        let joined = found.join(" | ");
        assert!(joined.contains("41% of drafted tokens"), "{joined}");
        assert!(joined.contains("12% of completions"), "{joined}");
        assert!(joined.contains("3% of requests were abandoned"), "{joined}");
        // 62% prefix hits is healthy, so that prediction is NOT contradicted.
        assert!(!joined.contains("prefix reuse was configured"), "{joined}");
    }

    #[test]
    fn a_plan_that_predicted_correctly_is_not_second_guessed() {
        // Only falsified predictions are reported. A config that never enabled
        // speculation must not be told its acceptance rate is low.
        let s = parse(SAMPLE);
        let found = s.contradictions(false, false).join(" | ");
        assert!(!found.contains("drafted tokens"), "{found}");
        assert!(!found.contains("prefix reuse"), "{found}");
    }

    #[test]
    fn queue_pressure_is_only_reported_when_it_is_actually_the_cap() {
        // Waiting with a full KV cache is a hardware limit; waiting with an
        // empty one is a concurrency setting. Only the second is actionable,
        // and conflating them sends people shopping for GPUs they do not need.
        let with = |kv: f64| {
            parse(&format!(
                "vllm:kv_cache_usage_perc{{m=\"a\"}} {kv}\n\
                 vllm:request_queue_time_seconds_sum{{m=\"a\"}} 40.0\n\
                 vllm:request_queue_time_seconds_count{{m=\"a\"}} 100.0\n"
            ))
            .contradictions(false, false)
            .join(" ")
        };
        assert!(with(0.30).contains("knob, not a purchase"));
        assert!(!with(0.97).contains("knob, not a purchase"));
    }

    #[test]
    fn contradictions_say_nothing_when_nothing_was_measured() {
        // Absent telemetry must not read as a passing grade.
        assert!(Snapshot::default().contradictions(true, true).is_empty());
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
