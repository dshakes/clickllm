//! The `Runtime` trait — M2, and the load-bearing abstraction of the crate.
//!
//! Development happens on Apple Metal; production is CUDA. They share nothing but
//! the OpenAI wire format. Every engine therefore sits behind this trait, and
//! **no engine-specific type may escape it** (ADR-0002). Violating that is the
//! single most likely way to break portability under deadline pressure, so it is
//! checked by `tests/no_engine_leak.rs`.
//!
//! [`Runtime::render`] emits **native** configuration — a real `vllm serve`
//! invocation, a real `InferencePool` — that runs with clickllm uninstalled
//! (NFR-4). We never wrap an engine at runtime: that is BentoML's failure mode,
//! where wrappers lag upstream by a release and every flag change costs a rebuild.

pub mod llmd;
pub mod sglang;
pub mod vllm;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::spec::{Hardware, ModelSpec, Workload};

/// Whether a runtime can serve a model on given hardware.
///
/// `Degraded` exists because "yes, but slowly" is more useful than a runtime
/// error on someone's cluster — but it must always carry the reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "verdict")]
pub enum Feasibility {
    /// Runs well.
    Yes,
    /// Runs, with a caveat the user must see.
    Degraded {
        /// What is compromised, in one sentence.
        reason: String,
    },
    /// Cannot run.
    No {
        /// Why not, in one sentence.
        reason: String,
    },
}

impl Feasibility {
    /// True for [`Feasibility::Yes`] and [`Feasibility::Degraded`].
    pub fn is_usable(&self) -> bool {
        !matches!(self, Self::No { .. })
    }

    /// The caveat or refusal reason, if any.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Yes => None,
            Self::Degraded { reason } | Self::No { reason } => Some(reason),
        }
    }
}

/// Speculative decoding configuration.
///
/// EAGLE-3's headline 2–3× is a *single-stream* figure; realistic serving sees
/// ~1.3–1.8×, and acceptance degrades past batch ~32 where the draft cost exceeds
/// the win. A tuner that always enables this makes users slower, so
/// [`SpecDecode::for_workload`] returns `None` above the cliff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecDecode {
    /// Method name as the engine spells it (`eagle3`, `ngram`).
    pub method: String,
    /// Draft tokens proposed per step.
    pub num_speculative_tokens: u32,
}

/// Concurrency past which speculative decoding stops paying for itself.
pub const SPEC_DECODE_CONCURRENCY_CLIFF: u32 = 32;

impl SpecDecode {
    /// Choose a configuration for this workload, or `None` when it would hurt.
    pub fn for_workload(wl: &Workload) -> Option<Self> {
        if wl.concurrency > SPEC_DECODE_CONCURRENCY_CLIFF {
            tracing::info!(
                concurrency = wl.concurrency,
                cliff = SPEC_DECODE_CONCURRENCY_CLIFF,
                "speculative decoding disabled: acceptance falls off above the cliff \
                 and the draft pass would cost more than it saves"
            );
            return None;
        }
        // Draft length shrinks as batching rises: more sequences means lower
        // per-sequence acceptance, so long drafts waste verification.
        let n = if wl.concurrency <= 4 { 5 } else { 3 };
        Some(Self {
            method: "eagle3".into(),
            num_speculative_tokens: n,
        })
    }
}

/// A fully-resolved deployment configuration. Every field is *derived*, never
/// asked for (ADR-0004): the user names a model and a target, nothing else.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimePlan {
    /// Model this plan serves.
    pub model: ModelSpec,
    /// Quantisation label chosen by the memory solve.
    pub quant: String,
    /// Bits per weight implied by `quant`.
    pub bits_per_weight: f64,
    /// Context to configure — the observed p95, not the model's maximum.
    pub max_model_len: u32,
    /// Concurrent sequences to admit.
    pub max_num_seqs: u32,
    /// Tensor-parallel degree; matches device count.
    pub tensor_parallel: u32,
    /// Speculative decoding, when it helps.
    pub spec_decode: Option<SpecDecode>,
    /// Prefix/radix caching, when traffic actually shares prefixes.
    pub prefix_caching: bool,
    /// Fraction of device memory the engine may reserve.
    pub memory_utilization: f64,
    /// Why each choice was made, for `--explain`. A recommendation a user cannot
    /// audit will not be trusted with production.
    pub rationale: Vec<String>,
}

/// Where a rendered configuration is meant to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    /// A supervised process on this machine.
    LocalProcess,
    /// A container image plus compose file.
    Container,
    /// Kubernetes manifests.
    Kubernetes,
    /// A systemd unit, for a plain VM or a Mac mini under launchd's cousin.
    ///
    /// The answer to "I have an EC2 box" — which is not a platform, it is a
    /// machine, and what it needs is a supervisor that restarts the process.
    Systemd,
}

impl Target {
    /// Every target, for callers that render all of them.
    pub const ALL: [Target; 4] = [
        Target::LocalProcess,
        Target::Container,
        Target::Kubernetes,
        Target::Systemd,
    ];
}

/// One generated file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    /// Suggested path, relative to the output directory.
    pub path: String,
    /// File contents.
    pub contents: String,
}

/// A running, OpenAI-compatible endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Endpoint {
    /// Base URL, e.g. `http://127.0.0.1:8000/v1`.
    pub url: String,
    /// Model name clients should send.
    pub served_model: String,
}

/// An inference engine clickllm can plan for, configure, and launch.
pub trait Runtime: Send + Sync {
    /// Stable identifier used in manifests and logs.
    fn name(&self) -> &'static str;

    /// Can this engine serve `model` on `hw`?
    fn supports(&self, hw: &Hardware, model: &ModelSpec) -> Feasibility;

    /// Derive a full configuration. Called only when [`Runtime::supports`] is usable.
    ///
    /// # Errors
    /// [`crate::Error::RuntimePlan`] when no viable configuration exists.
    fn plan(&self, hw: &Hardware, model: &ModelSpec, wl: &Workload) -> Result<RuntimePlan>;

    /// Emit native configuration for `target`.
    ///
    /// # Errors
    /// [`crate::Error::RuntimePlan`] when the engine cannot express the plan for
    /// that target.
    fn render(&self, plan: &RuntimePlan, target: Target) -> Result<Vec<Artifact>>;
}

/// Provenance header stamped into every generated artifact.
///
/// The artifact is a receipt (ADR-0004): it records what was chosen and why, and
/// stays runnable if clickllm is uninstalled.
pub(crate) fn provenance(runtime: &str, plan: &RuntimePlan, comment: &str) -> String {
    let mut s = format!(
        "{comment} Generated by clickllm {} — runtime: {runtime}, model: {}, quant: {}\n\
         {comment} This file is a receipt, not a form. It runs without clickllm installed.\n",
        crate::VERSION,
        plan.model.id,
        plan.quant,
    );
    for line in &plan.rationale {
        s.push_str(&format!("{comment}   · {line}\n"));
    }
    s
}


/// A systemd unit for a plain VM.
///
/// `Restart=always` with a backoff, because the thing this buys over `nohup` is
/// that the process comes back. `TimeoutStartSec` is generous: loading tens of
/// gigabytes of weights from disk is not a hung service, and the default would
/// kill a healthy start.
pub(crate) fn systemd_unit(name: &str, argv: &[String], plan: &RuntimePlan) -> String {
    // systemd requires ExecStart='s first token to be an absolute path — a bare
    // command name like `vllm` or `python3` is rejected at load with "Executable
    // path is not absolute". `/usr/bin/env` is itself absolute and resolves the
    // real executable from $PATH, same trick as a `#!/usr/bin/env` shebang.
    let exec = std::iter::once("/usr/bin/env".to_string())
        .chain(argv.iter().cloned())
        .map(|a| {
            // systemd splits ExecStart on whitespace, so `--speculative-config
            // {"method": "eagle3"}` becomes four arguments and the unit fails to
            // start. Quoting is triggered by whitespace *or* any character that
            // is special to systemd's own quoted-argument parser (`"`, `'`,
            // `\`) — a lone quote elsewhere in the argument would otherwise be
            // read as an unbalanced quote or silently stripped.
            if a.contains(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == '\\') {
                format!("\"{}\"", a.replace('\\', "\\\\").replace('"', "\\\""))
            } else {
                a
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "# clickllm — {}, {} @ {} context. Runs with clickllm uninstalled.\n\
         [Unit]\nDescription=clickllm inference — {}\nAfter=network-online.target\n\
         Wants=network-online.target\n\n[Service]\nType=simple\n\
         ExecStart={exec}\nRestart=always\nRestartSec=10\n\
         # Weights can be tens of GiB; the default 90s would kill a healthy start.\n\
         TimeoutStartSec=1800\n\
         # An OOM here should restart the service, not take the box with it.\n\
         OOMPolicy=stop\nLimitNOFILE=65536\n\n\
         [Install]\nWantedBy=multi-user.target\n",
        plan.model.id, plan.quant, plan.max_model_len, name,
    )
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    #[test]
    fn feasibility_usability_and_reasons() {
        assert!(Feasibility::Yes.is_usable());
        assert!(Feasibility::Yes.reason().is_none());

        let d = Feasibility::Degraded {
            reason: "slow".into(),
        };
        assert!(d.is_usable());
        assert_eq!(d.reason(), Some("slow"));

        let n = Feasibility::No {
            reason: "no room".into(),
        };
        assert!(!n.is_usable());
        assert_eq!(n.reason(), Some("no room"));
    }

    #[test]
    fn spec_decode_is_disabled_above_the_concurrency_cliff() {
        // The negative case is the acceptance test for the whole tuner: a tuner
        // that only ever adds optimisations has not been tested.
        let wl = Workload {
            concurrency: SPEC_DECODE_CONCURRENCY_CLIFF + 1,
            ..Workload::default()
        };
        assert!(SpecDecode::for_workload(&wl).is_none());
    }

    #[test]
    fn spec_decode_is_enabled_at_and_below_the_cliff() {
        for c in [1, 4, 8, SPEC_DECODE_CONCURRENCY_CLIFF] {
            let wl = Workload {
                concurrency: c,
                ..Workload::default()
            };
            assert!(
                SpecDecode::for_workload(&wl).is_some(),
                "concurrency {c} should keep spec-decode"
            );
        }
    }

    #[test]
    fn draft_length_shrinks_as_concurrency_rises() {
        let low = SpecDecode::for_workload(&Workload {
            concurrency: 2,
            ..Workload::default()
        })
        .unwrap();
        let high = SpecDecode::for_workload(&Workload {
            concurrency: 16,
            ..Workload::default()
        })
        .unwrap();
        assert!(high.num_speculative_tokens < low.num_speculative_tokens);
    }

    #[test]
    fn provenance_names_the_tool_and_carries_the_rationale() {
        let plan = crate::runtime::vllm::tests::sample_plan();
        let p = provenance("vllm", &plan, "#");
        assert!(p.contains("clickllm"));
        assert!(p.contains(&plan.model.id));
        assert!(
            p.lines().all(|l| l.starts_with('#')),
            "must be fully commented"
        );
        for line in &plan.rationale {
            assert!(p.contains(line.as_str()), "rationale {line:?} must survive");
        }
    }

    #[test]
    fn systemd_exec_start_is_an_absolute_path() {
        // systemd rejects a unit at load time with "Executable path is not
        // absolute" unless ExecStart='s first token starts with `/`.
        let plan = crate::runtime::vllm::tests::sample_plan();
        let u = systemd_unit("model", &["vllm".into(), "serve".into()], &plan);
        let exec = u
            .lines()
            .find(|l| l.starts_with("ExecStart="))
            .expect("no ExecStart");
        let first_token = exec
            .trim_start_matches("ExecStart=")
            .split_whitespace()
            .next()
            .unwrap();
        assert!(
            first_token.starts_with('/'),
            "ExecStart must start with an absolute path: {exec}"
        );
        assert_eq!(first_token, "/usr/bin/env");
    }

    #[test]
    fn systemd_quotes_arguments_with_bare_quotes_or_backslashes() {
        // Whitespace is not the only character systemd's ExecStart parser treats
        // specially. A lone `'` without surrounding whitespace produces an
        // unbalanced quote and the unit fails to load; a lone `"` gets silently
        // stripped, corrupting the argument the model runtime receives.
        let plan = crate::runtime::vllm::tests::sample_plan();
        let u = systemd_unit(
            "model",
            &[
                "vllm".into(),
                "--user-model=user's-model".into(),
                "--flag=\"quoted\"".into(),
            ],
            &plan,
        );
        let exec = u
            .lines()
            .find(|l| l.starts_with("ExecStart="))
            .expect("no ExecStart");
        assert!(
            exec.contains("\"--user-model=user's-model\""),
            "argument with a bare single quote must be wrapped in quotes: {exec}"
        );
        assert!(
            exec.contains(r#""--flag=\"quoted\"""#),
            "argument with bare double quotes must be wrapped and escaped: {exec}"
        );
    }

}
