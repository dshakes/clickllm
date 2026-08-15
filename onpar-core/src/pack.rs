//! Inference in a box — a portable, tuned artifact and what happens when it lands.
//!
//! A box carries a model reference, the plan that was tuned for it, the hardware
//! it was tuned *on*, and the measurements that ratified that tuning. It ships as
//! an OCI artifact so it moves through registries people already run.
//!
//! The interesting part is not packaging, it is **arrival**. A box packed on four
//! H100s and pulled onto a single 24 GB L40S must re-quantise and re-fit rather
//! than OOM on the first request. So a box is a *tuned starting point plus the
//! evidence behind it*, never a frozen command line — applying stale settings on
//! different hardware is the "wrong auto-tune" failure of ADR-0004, arriving by
//! post.
//!
//! Two refusals are deliberate:
//!
//! - **An unpinned model reference cannot be packed.** A box that says "latest"
//!   cannot promise the same weights tomorrow, which makes its bench evidence a
//!   claim about a model that may no longer exist.
//! - **Unsupported hardware is reported, never approximated.** "No runtime can
//!   serve this here" is a usable answer; a confident wrong one is not.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::model_ref::ModelRef;
use crate::runtime::{Feasibility, Runtime, RuntimePlan};
use crate::spec::{Accelerator, Hardware, ModelSpec, Workload};

/// Manifest schema version. Bumped when a field changes meaning, so an old box
/// is rejected with a version message rather than misread.
pub const SCHEMA: u32 = 1;

/// The hardware a box was tuned on, reduced to what actually decides a re-solve.
///
/// Deliberately coarse. A different GPU *model* with the same memory, bandwidth
/// and device count needs no re-solve; a different *amount of memory* always does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostClass {
    /// Accelerator family.
    pub accelerator: Accelerator,
    /// Human-readable name, for the provenance line.
    pub name: String,
    /// Memory usable for inference.
    pub usable_bytes: u64,
    /// Peak bandwidth, when known.
    pub bandwidth_gbps: Option<f64>,
    /// Device count.
    pub devices: u32,
}

impl HostClass {
    /// Describe a machine.
    pub fn of(hw: &Hardware) -> Self {
        Self {
            accelerator: hw.accelerator,
            name: hw.name.clone(),
            usable_bytes: hw.usable_bytes,
            bandwidth_gbps: hw.bandwidth_gbps,
            devices: hw.devices,
        }
    }

    /// Whether a plan tuned for `self` can be applied unchanged on `other`.
    ///
    /// Requires the same accelerator family and device count, and **at least as
    /// much** memory. More memory is fine — the plan simply leaves headroom
    /// unused, which is wasteful but never wrong. Less memory never is.
    pub fn accepts(&self, other: &Self) -> bool {
        self.accelerator == other.accelerator
            && self.devices == other.devices
            && other.usable_bytes >= self.usable_bytes
    }

    /// One line naming the difference, for the arrival report.
    pub fn diff(&self, other: &Self) -> Vec<String> {
        let mut out = Vec::new();
        if self.accelerator != other.accelerator {
            out.push(format!(
                "accelerator {:?} -> {:?}",
                self.accelerator, other.accelerator
            ));
        }
        if self.devices != other.devices {
            out.push(format!("devices {} -> {}", self.devices, other.devices));
        }
        if self.usable_bytes != other.usable_bytes {
            out.push(format!(
                "usable memory {:.0} GiB -> {:.0} GiB",
                self.usable_bytes as f64 / crate::spec::GIB as f64,
                other.usable_bytes as f64 / crate::spec::GIB as f64
            ));
        }
        out
    }
}

/// A measurement that ratified a plan. Absent means the tuning was never
/// benchmarked, and callers are told rather than left to assume it was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bench {
    /// Measured decode throughput.
    pub tokens_per_sec: f64,
    /// Measured time to first token, milliseconds.
    pub ttft_ms: Option<f64>,
    /// Host class the measurement was taken on.
    pub on: HostClass,
    /// Optimisations that were tried and reverted because they did not help.
    pub reverted: Vec<String>,
}

/// Everything a box carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version.
    pub schema: u32,
    /// Pinned reference to the weights. Always pinned — see module docs.
    pub model_ref: String,
    /// Model geometry, so a re-solve needs no catalogue lookup.
    pub model: ModelSpec,
    /// Digest of the weights this box was built against.
    pub weights_digest: String,
    /// The tuned plan.
    pub plan: RuntimePlan,
    /// Runtime the plan targets.
    pub runtime: String,
    /// Hardware it was tuned on.
    pub packed_on: HostClass,
    /// Workload it was tuned for.
    pub workload: Workload,
    /// Evidence, if the tuning was measured.
    pub bench: Option<Bench>,
    /// Version of onpar that packed it.
    pub tool_version: String,
}

impl Manifest {
    /// Build a manifest, refusing anything that cannot be reproduced.
    ///
    /// # Errors
    /// [`Error::RuntimePlan`] if the model reference is not pinned.
    pub fn new(
        model_ref: &ModelRef,
        weights_digest: &str,
        plan: RuntimePlan,
        runtime: &str,
        packed_on: HostClass,
        workload: Workload,
        bench: Option<Bench>,
    ) -> Result<Self> {
        if !model_ref.is_pinned() {
            return Err(Error::RuntimePlan {
                runtime: "pack",
                reason: format!(
                    "{model_ref} is not pinned to a revision; a box that says \"latest\" \
                     cannot promise the same weights tomorrow, which would make its \
                     benchmark evidence a claim about a model that may no longer exist"
                ),
            });
        }
        Ok(Self {
            schema: SCHEMA,
            model_ref: model_ref.to_string(),
            model: plan.model.clone(),
            weights_digest: weights_digest.to_owned(),
            plan,
            runtime: runtime.to_owned(),
            packed_on,
            workload,
            bench,
            tool_version: crate::VERSION.to_owned(),
        })
    }

    /// Human-readable provenance, written into the box as `README.md`.
    pub fn provenance(&self) -> String {
        let mut s = format!(
            "# {}\n\n\
             Packed by onpar {} for `{}` on {}.\n\n\
             - model: `{}`\n- quantisation: `{}`\n- context: {}\n- concurrency: {}\n\
             - weights: `{}`\n",
            self.model.id,
            self.tool_version,
            self.runtime,
            self.packed_on.name,
            self.model_ref,
            self.plan.quant,
            self.plan.max_model_len,
            self.plan.max_num_seqs,
            self.weights_digest,
        );
        match &self.bench {
            Some(b) => {
                s.push_str(&format!(
                    "\n## Measured on {}\n\n- {:.0} tok/s{}\n",
                    b.on.name,
                    b.tokens_per_sec,
                    b.ttft_ms
                        .map(|t| format!(", {t:.0} ms to first token"))
                        .unwrap_or_default()
                ));
                for r in &b.reverted {
                    s.push_str(&format!("- reverted: {r}\n"));
                }
            }
            None => s.push_str(
                "\n## Not benchmarked\n\nThis plan was derived but never measured. \
                 Its throughput figures are estimates.\n",
            ),
        }
        s.push_str("\n## Why these settings\n\n");
        for line in &self.plan.rationale {
            s.push_str(&format!("- {line}\n"));
        }
        s.push_str(
            "\nThis box re-profiles the machine it lands on and re-solves if the \
             hardware differs. It is a tuned starting point plus the evidence behind \
             it, not a frozen command line.\n",
        );
        s
    }
}

/// What happened when a box was opened on a machine.
#[derive(Debug, Clone, PartialEq)]
pub enum Arrival {
    /// The host matches what the box was tuned for. Use it as shipped.
    AsPacked {
        /// The plan, unchanged.
        plan: RuntimePlan,
    },
    /// The host differs. The plan was re-derived here.
    Resolved {
        /// The re-derived plan.
        plan: RuntimePlan,
        /// What about the hardware differed.
        host_changes: Vec<String>,
        /// What about the plan changed as a result.
        plan_changes: Vec<String>,
        /// True when the box carried a measurement taken on different hardware,
        /// so its throughput figures no longer apply here.
        bench_invalidated: bool,
    },
    /// Nothing here can serve this model.
    Unsupported {
        /// Why, in one sentence.
        reason: String,
    },
}

impl Arrival {
    /// The plan to run, if there is one.
    pub fn plan(&self) -> Option<&RuntimePlan> {
        match self {
            Self::AsPacked { plan } | Self::Resolved { plan, .. } => Some(plan),
            Self::Unsupported { .. } => None,
        }
    }

    /// A line for the run log. Silence about a re-solve is a bug: a user who
    /// benchmarked this box elsewhere needs to know its numbers moved.
    pub fn report(&self) -> String {
        match self {
            Self::AsPacked { .. } => {
                "host matches the packed target; using the shipped plan".into()
            }
            Self::Resolved {
                host_changes,
                plan_changes,
                bench_invalidated,
                ..
            } => {
                let mut s = format!("re-solved on arrival ({})", host_changes.join(", "));
                for c in plan_changes {
                    s.push_str(&format!("\n  {c}"));
                }
                if *bench_invalidated {
                    s.push_str(
                        "\n  the benchmark in this box was measured on different hardware \
                         and no longer applies",
                    );
                }
                s
            }
            Self::Unsupported { reason } => format!("cannot run here: {reason}"),
        }
    }
}

/// Smallest context worth serving. Below this the deployment is not useful and
/// "cannot run here" is the more honest answer than a 512-token window.
pub const MIN_DEGRADED_CONTEXT: u32 = 2048;

/// Try the packed workload, then progressively smaller ones.
///
/// A box pulled onto a smaller machine should give something up rather than
/// nothing — that is the difference between "re-quantised, context 32k -> 16k"
/// and "cannot run here". What it gave up is always reported: a deployment that
/// silently serves a quarter of the context it was built for is worse than one
/// that refuses, because nobody finds out until a long request truncates.
fn resolve_here(
    rt: &dyn Runtime,
    host: &Hardware,
    model: &ModelSpec,
    wanted: &Workload,
) -> Option<(RuntimePlan, Vec<String>)> {
    if let Ok(p) = rt.plan(host, model, wanted) {
        return Some((p, Vec::new()));
    }

    // Context first: it is usually the more elastic of the two, and halving it
    // frees KV linearly.
    let mut ctx = wanted.p95_context;
    while ctx > MIN_DEGRADED_CONTEXT {
        // Halving is the intent: KV scales linearly in context, so each step
        // frees a predictable amount.
        #[allow(clippy::integer_division)]
        let halved = ctx / 2;
        ctx = halved.max(MIN_DEGRADED_CONTEXT);
        let attempt = Workload {
            p95_context: ctx,
            ..*wanted
        };
        if let Ok(p) = rt.plan(host, model, &attempt) {
            return Some((
                p,
                vec![format!(
                    "context reduced {} -> {} to fit this machine",
                    wanted.p95_context, ctx
                )],
            ));
        }
    }

    // Then concurrency, at the smallest useful context.
    let mut conc = wanted.concurrency;
    while conc > 1 {
        #[allow(clippy::integer_division)]
        let halved = conc / 2;
        conc = halved.max(1);
        let attempt = Workload {
            p95_context: MIN_DEGRADED_CONTEXT,
            concurrency: conc,
            ..*wanted
        };
        if let Ok(p) = rt.plan(host, model, &attempt) {
            return Some((
                p,
                vec![
                    format!(
                        "context reduced {} -> {}",
                        wanted.p95_context, MIN_DEGRADED_CONTEXT
                    ),
                    format!(
                        "concurrency reduced {} -> {} to fit this machine",
                        wanted.concurrency, conc
                    ),
                ],
            ));
        }
    }
    None
}

/// Open a box on `host`, re-solving if the hardware differs.
///
/// `runtimes` are tried in order; the first that supports the model wins.
pub fn on_arrival(m: &Manifest, host: &Hardware, runtimes: &[&dyn Runtime]) -> Arrival {
    let here = HostClass::of(host);

    if m.packed_on.accepts(&here) {
        return Arrival::AsPacked {
            plan: m.plan.clone(),
        };
    }

    let host_changes = m.packed_on.diff(&here);
    let mut refusals = Vec::new();

    for rt in runtimes {
        match rt.supports(host, &m.model) {
            Feasibility::No { reason } => {
                refusals.push(format!("{}: {reason}", rt.name()));
                continue;
            }
            _ => match resolve_here(*rt, host, &m.model, &m.workload) {
                Some((plan, mut concessions)) => {
                    let mut plan_changes = plan_diff(&m.plan, &plan);
                    plan_changes.append(&mut concessions);
                    return Arrival::Resolved {
                        bench_invalidated: m.bench.is_some(),
                        plan,
                        host_changes,
                        plan_changes,
                    };
                }
                None => refusals.push(format!(
                    "{}: does not fit even at reduced context and concurrency",
                    rt.name()
                )),
            },
        }
    }

    Arrival::Unsupported {
        reason: if refusals.is_empty() {
            "no runtimes were offered".into()
        } else {
            refusals.join("; ")
        },
    }
}

/// Name what changed between two plans, in user terms.
fn plan_diff(before: &RuntimePlan, after: &RuntimePlan) -> Vec<String> {
    let mut out = Vec::new();
    if before.quant != after.quant {
        out.push(format!("re-quantised {} -> {}", before.quant, after.quant));
    }
    if before.max_model_len != after.max_model_len {
        out.push(format!(
            "context {} -> {}",
            before.max_model_len, after.max_model_len
        ));
    }
    if before.max_num_seqs != after.max_num_seqs {
        out.push(format!(
            "concurrency {} -> {}",
            before.max_num_seqs, after.max_num_seqs
        ));
    }
    if before.tensor_parallel != after.tensor_parallel {
        out.push(format!(
            "tensor parallel {} -> {}",
            before.tensor_parallel, after.tensor_parallel
        ));
    }
    match (&before.spec_decode, &after.spec_decode) {
        (Some(_), None) => out.push("speculative decoding disabled here".into()),
        (None, Some(s)) => out.push(format!("speculative decoding enabled ({})", s.method)),
        _ => {}
    }
    if before.prefix_caching != after.prefix_caching {
        out.push(format!(
            "prefix caching {}",
            if after.prefix_caching { "on" } else { "off" }
        ));
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::runtime::vllm::Vllm;
    use crate::runtime::vllm::tests::{h100, model};
    use crate::spec::GIB;

    fn hw(devices: u32, gib: u64) -> Hardware {
        Hardware {
            devices,
            usable_bytes: gib * GIB,
            ..h100()
        }
    }

    fn manifest_on(host: &Hardware, bench: bool) -> Manifest {
        let wl = Workload {
            p95_context: 16384,
            concurrency: 4,
            prefix_share: 0.0,
        };
        let plan = Vllm::new().plan(host, &model(), &wl).unwrap();
        let cls = HostClass::of(host);
        Manifest::new(
            &"hf:org/m@abc123#q4".parse().unwrap(),
            "deadbeef",
            plan,
            "vllm",
            cls.clone(),
            wl,
            bench.then(|| Bench {
                tokens_per_sec: 74.0,
                ttft_ms: Some(180.0),
                on: cls,
                reverted: vec!["speculative decoding: cost 8% here".into()],
            }),
        )
        .unwrap()
    }

    #[test]
    fn an_unpinned_reference_cannot_be_packed() {
        // A box that says "latest" cannot promise the same weights tomorrow.
        let host = hw(1, 72);
        let plan = Vllm::new()
            .plan(&host, &model(), &Workload::default())
            .unwrap();
        let err = Manifest::new(
            &"hf:org/m#q4".parse().unwrap(), // no @revision
            "d",
            plan,
            "vllm",
            HostClass::of(&host),
            Workload::default(),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("not pinned"), "{err}");
    }

    #[test]
    fn identical_hardware_uses_the_shipped_plan() {
        let host = hw(1, 72);
        let m = manifest_on(&host, true);
        let a = on_arrival(&m, &host, &[&Vllm::new()]);
        assert!(matches!(a, Arrival::AsPacked { .. }));
        assert_eq!(a.plan().unwrap().quant, m.plan.quant);
        assert!(a.report().contains("matches"));
    }

    #[test]
    fn more_memory_than_packed_for_is_still_as_packed() {
        // The plan leaves headroom unused, which is wasteful but never wrong.
        let m = manifest_on(&hw(1, 72), false);
        let a = on_arrival(&m, &hw(1, 140), &[&Vllm::new()]);
        assert!(matches!(a, Arrival::AsPacked { .. }));
    }

    #[test]
    fn landing_on_a_smaller_box_re_solves_rather_than_ooming() {
        // The headline behaviour: packed for 72 GiB, pulled onto 24 GiB.
        let m = manifest_on(&hw(1, 72), true);
        let a = on_arrival(&m, &hw(1, 22), &[&Vllm::new()]);

        let Arrival::Resolved {
            plan,
            host_changes,
            plan_changes,
            bench_invalidated,
        } = &a
        else {
            panic!("expected a re-solve, got {a:?}");
        };
        assert!(host_changes.iter().any(|c| c.contains("usable memory")));
        assert!(!plan_changes.is_empty(), "something must have changed");
        assert!(plan.quant != m.plan.quant || plan.max_model_len < m.plan.max_model_len);
        assert!(
            *bench_invalidated,
            "a benchmark from other hardware must be flagged"
        );
        assert!(a.report().contains("re-solved"));
    }

    #[test]
    fn a_re_solve_without_a_bench_does_not_claim_one_was_invalidated() {
        let m = manifest_on(&hw(1, 72), false);
        let a = on_arrival(&m, &hw(1, 22), &[&Vllm::new()]);
        let Arrival::Resolved {
            bench_invalidated, ..
        } = a
        else {
            panic!("expected a re-solve");
        };
        assert!(!bench_invalidated);
    }

    #[test]
    fn a_box_that_does_not_fit_gives_something_up_rather_than_nothing() {
        // ADR-0005's headline: packed for 72 GiB, pulled onto 22 GiB, and it
        // still serves — with the concession stated.
        let m = manifest_on(&hw(1, 72), false);
        let a = on_arrival(&m, &hw(1, 22), &[&Vllm::new()]);
        let Arrival::Resolved {
            plan, plan_changes, ..
        } = &a
        else {
            panic!("expected a degraded re-solve, got {a:?}");
        };
        assert!(plan.max_model_len < m.plan.max_model_len);
        assert!(
            plan_changes.iter().any(|c| c.contains("reduced")),
            "a silent downgrade is worse than a refusal: {plan_changes:?}"
        );
        assert!(a.report().contains("reduced"));
    }

    #[test]
    fn degradation_stops_at_a_useful_floor_rather_than_serving_a_toy_window() {
        // A machine far too small must refuse, not offer a 512-token context.
        let m = manifest_on(&hw(1, 72), false);
        let a = on_arrival(&m, &hw(1, 2), &[&Vllm::new()]);
        assert!(matches!(a, Arrival::Unsupported { .. }), "got {a:?}");
        let Arrival::Unsupported { reason } = &a else {
            unreachable!()
        };
        // Either refusal is fine; both are actionable. What matters is that it
        // says why rather than quietly serving a 512-token window.
        assert!(
            reason.contains("weights") || reason.contains("reduced context"),
            "refusal must be actionable: {reason}"
        );
        assert!(a.plan().is_none());
    }

    #[test]
    fn a_different_device_count_forces_a_re_solve() {
        let m = manifest_on(&hw(4, 288), false);
        let a = on_arrival(&m, &hw(1, 288), &[&Vllm::new()]);
        let Arrival::Resolved {
            host_changes,
            plan_changes,
            ..
        } = &a
        else {
            panic!("expected a re-solve, got {a:?}");
        };
        assert!(host_changes.iter().any(|c| c.contains("devices")));
        assert!(plan_changes.iter().any(|c| c.contains("tensor parallel")));
    }

    #[test]
    fn hardware_that_cannot_run_it_says_so_rather_than_approximating() {
        let apple = Hardware {
            accelerator: Accelerator::Apple,
            name: "M4 Max".into(),
            usable_bytes: 96 * GIB,
            bandwidth_gbps: Some(546.0),
            devices: 1,
        };
        let m = manifest_on(&hw(1, 72), false);
        let a = on_arrival(&m, &apple, &[&Vllm::new()]);
        let Arrival::Unsupported { reason } = &a else {
            panic!("vLLM cannot run on Metal; expected Unsupported, got {a:?}");
        };
        assert!(reason.contains("CUDA"), "{reason}");
        assert!(a.plan().is_none());
        assert!(a.report().contains("cannot run here"));
    }

    #[test]
    fn no_runtimes_offered_is_reported_not_silently_empty() {
        let m = manifest_on(&hw(1, 72), false);
        let a = on_arrival(&m, &hw(1, 22), &[]);
        assert!(matches!(a, Arrival::Unsupported { .. }));
    }

    #[test]
    fn provenance_states_whether_it_was_measured() {
        let measured = manifest_on(&hw(1, 72), true).provenance();
        assert!(measured.contains("Measured on"));
        assert!(
            measured.contains("reverted"),
            "reverted optimisations must be recorded"
        );

        let estimated = manifest_on(&hw(1, 72), false).provenance();
        assert!(estimated.contains("Not benchmarked"));
        assert!(estimated.contains("estimates"));
    }

    #[test]
    fn provenance_carries_the_rationale_so_the_box_explains_itself() {
        let m = manifest_on(&hw(1, 72), false);
        let p = m.provenance();
        for line in &m.plan.rationale {
            assert!(p.contains(line.as_str()), "lost rationale: {line}");
        }
        assert!(p.contains(&m.weights_digest));
        assert!(p.contains("re-profiles"));
    }

    #[test]
    fn a_manifest_round_trips_through_json() {
        let m = manifest_on(&hw(1, 72), true);
        let s = serde_json::to_string(&m).unwrap();
        let back: Manifest = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.schema, SCHEMA);
    }

    #[test]
    fn accepts_is_directional() {
        let small = HostClass::of(&hw(1, 24));
        let big = HostClass::of(&hw(1, 80));
        assert!(small.accepts(&big), "a plan for 24 GiB is safe on 80");
        assert!(!big.accepts(&small), "a plan for 80 GiB is not safe on 24");
    }
}
