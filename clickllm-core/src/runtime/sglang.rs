//! SGLang — the engine to reach for when requests share a prefix.
//!
//! vLLM is the better-supported default and this is not a replacement for it.
//! SGLang earns its place on one property: **RadixAttention reuses a shared
//! prefix's KV across requests instead of recomputing it per request.** For an
//! agent fleet on a fixed system prompt, or few-shot templates, or RAG with a
//! stable preamble, that is a structural win no vLLM flag reproduces — and it
//! grows with fleet size rather than washing out.
//!
//! ## Its flags are not vLLM's flags with different names
//!
//! Every argument below was verified against the published server-arguments
//! page (see [`SOURCE`]), and three of them differ from vLLM in ways a rename
//! would get wrong:
//!
//! | Intent | vLLM | SGLang |
//! |---|---|---|
//! | context | `--max-model-len` | `--context-length` |
//! | concurrency cap | `--max-num-seqs` | `--max-running-requests` |
//! | memory | `--gpu-memory-utilization` | `--mem-fraction-static` |
//! | chunked prefill | `--enable-chunked-prefill` (bool) | `--chunked-prefill-size` (int, `-1` off) |
//! | prefix reuse | `--enable-prefix-caching` (opt **in**) | `--disable-radix-cache` (opt **out**) |
//!
//! That last row is the trap. Radix caching is **on by default** here, so a
//! plan that wants prefix reuse emits *nothing*, and a plan that does not want
//! it emits an explicit disable. A layer that mapped `--enable-prefix-caching`
//! onto the nearest-looking SGLang flag would turn the optimisation into its
//! exact inverse, in a config that starts cleanly and merely runs slower.
//!
//! ## What is deliberately not emitted
//!
//! The published argument page was truncated before the speculative-decoding
//! family and the grammar backend. `--grammar-backend` is the *likely* name —
//! it is referenced indirectly elsewhere in those docs — and likely is not a
//! standard this file is willing to meet, because the output here is a command
//! a human runs against their own hardware. So [`RuntimePlan::spec_decode`] is
//! reported as unexpressed rather than guessed at.

use serde::{Deserialize, Serialize};

use super::vllm::k8s_name_for;
use super::{Artifact, Feasibility, Runtime, RuntimePlan, Target, provenance};
use crate::error::{Error, Result};
use crate::spec::{Accelerator, Hardware, ModelSpec, Workload};

/// Where these flag names were verified.
pub const SOURCE: &str =
    "https://docs.sglang.io/advanced_features/server_arguments.html (checked 2026-07-27)";

/// SGLang, driven through `sglang.launch_server`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sglang;

impl Runtime for Sglang {
    fn name(&self) -> &'static str {
        "sglang"
    }

    fn supports(&self, hw: &Hardware, model: &ModelSpec) -> Feasibility {
        match hw.accelerator {
            Accelerator::Apple => Feasibility::No {
                reason: "SGLang is CUDA/ROCm-only; Apple Silicon has no Metal backend".into(),
            },
            Accelerator::Cpu => Feasibility::No {
                // Deliberately harsher than vLLM's "degraded": SGLang's whole
                // reason for being chosen is RadixAttention throughput, and a
                // CPU deployment delivers none of it. "Slow" would be a
                // misleading way to describe getting nothing you came for.
                reason: "SGLang on CPU gives up the RadixAttention throughput that is \
                         the only reason to prefer it — use llama.cpp instead"
                    .into(),
            },
            Accelerator::Nvidia | Accelerator::Amd => {
                let smallest = 4.0;
                if model.weight_bytes(smallest) > hw.usable_bytes {
                    return Feasibility::No {
                        reason: format!(
                            "weights need {:.0} GiB at the most aggressive quantisation, \
                             {:.0} GiB usable",
                            model.weight_bytes(smallest) as f64 / crate::spec::GIB as f64,
                            hw.usable_bytes as f64 / crate::spec::GIB as f64,
                        ),
                    };
                }
                Feasibility::Yes
            }
        }
    }

    fn plan(&self, hw: &Hardware, model: &ModelSpec, wl: &Workload) -> Result<RuntimePlan> {
        // The sizing arithmetic is the engine-independent part, so it is not
        // re-derived here: a second copy would drift from vLLM's and produce two
        // different answers to "does this fit", which is worse than either.
        let mut plan = super::vllm::Vllm.plan(hw, model, wl).map_err(|e| match e {
            Error::RuntimePlan { reason, .. } => Error::RuntimePlan {
                runtime: "sglang",
                reason,
            },
            other => other,
        })?;

        if let Feasibility::No { reason } = self.supports(hw, model) {
            return Err(Error::RuntimePlan {
                runtime: "sglang",
                reason,
            });
        }

        plan.rationale.push(
            "sglang selected for RadixAttention: shared prefixes are served from cached \
             KV rather than recomputed per request"
                .into(),
        );
        if plan.spec_decode.is_some() {
            // Do not silently drop it — say it was dropped and why.
            plan.spec_decode = None;
            plan.rationale.push(
                "speculative decoding omitted: SGLang's flags for it are not on the \
                 verified argument page, and a guessed flag name is worse than an \
                 admitted gap"
                    .into(),
            );
        }
        Ok(plan)
    }

    fn render(&self, plan: &RuntimePlan, target: Target) -> Result<Vec<Artifact>> {
        let args = launch_args(plan);
        let prov = provenance("sglang", plan, "#");
        Ok(match target {
            Target::LocalProcess => vec![Artifact {
                path: "serve.sh".into(),
                contents: format!(
                    "#!/usr/bin/env bash\nset -euo pipefail\n\n{prov}\nexec {}\n",
                    args.join(" \\\n  ")
                ),
            }],
            Target::Container => vec![Artifact {
                path: "compose.yaml".into(),
                contents: format!(
                    "{prov}services:\n  sglang:\n    image: lmsysorg/sglang:latest\n    \
                     command: [{}]\n    ports: [\"30000:30000\"]\n    \
                     deploy:\n      resources:\n        reservations:\n          \
                     devices:\n            - capabilities: [gpu]\n              count: {}\n",
                    quoted(&args, 1),
                    plan.tensor_parallel,
                ),
            }],
            Target::Kubernetes => vec![Artifact {
                path: "deployment.yaml".into(),
                contents: format!(
                    "{prov}apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: {name}\n\
                     spec:\n  replicas: 1\n  selector:\n    matchLabels:\n      app: {name}\n  \
                     template:\n    metadata:\n      labels:\n        app: {name}\n    spec:\n      \
                     containers:\n        - name: sglang\n          \
                     image: lmsysorg/sglang:latest\n          args: [{}]\n          \
                     ports:\n            - containerPort: 30000\n          \
                     resources:\n            limits:\n              nvidia.com/gpu: {}\n",
                    quoted(&args, 1),
                    plan.tensor_parallel,
                    name = k8s_name_for(&plan.model.id),
                ),
            }],
        })
    }
}

/// Quote an argument vector for a YAML inline list, skipping `skip` leading entries.
fn quoted(args: &[String], skip: usize) -> String {
    args.iter()
        .skip(skip)
        .map(|a| format!("\"{a}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The native `sglang.launch_server` argument vector.
///
/// Native on purpose (NFR-4): this runs with clickllm uninstalled. Nothing here
/// wraps the engine, and nothing here invents a flag.
fn launch_args(plan: &RuntimePlan) -> Vec<String> {
    let mut a = vec![
        "python3".into(),
        "-m".into(),
        "sglang.launch_server".into(),
        "--model-path".into(),
        plan.model.id.clone(),
        "--quantization".into(),
        plan.quant.clone(),
        // NOT --max-model-len.
        "--context-length".into(),
        plan.max_model_len.to_string(),
        // NOT --max-num-seqs.
        "--max-running-requests".into(),
        plan.max_num_seqs.to_string(),
        // NOT --gpu-memory-utilization. Covers weights *and* the KV pool here,
        // where vLLM splits them — same number, different meaning.
        "--mem-fraction-static".into(),
        format!("{:.2}", plan.memory_utilization),
    ];
    if plan.tensor_parallel > 1 {
        a.push("--tp-size".into());
        a.push(plan.tensor_parallel.to_string());
    }
    // THE INVERSION. Radix caching is on by default, so wanting prefix reuse
    // emits nothing at all, and only *not* wanting it produces a flag.
    if !plan.prefix_caching {
        a.push("--disable-radix-cache".into());
    }
    a
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::spec::{Accelerator, Hardware, Workload};

    fn gpu(gib: u64, devices: u32) -> Hardware {
        Hardware {
            accelerator: Accelerator::Nvidia,
            name: "H100".into(),
            usable_bytes: gib * crate::spec::GIB,
            bandwidth_gbps: Some(3350.0),
            devices,
        }
    }

    // Reuses vLLM's test model so the two engines are compared on identical
    // input; a divergent fixture would hide a real disagreement about sizing.
    use super::super::vllm::tests::model;

    fn workload(concurrency: u32) -> Workload {
        Workload {
            concurrency,
            p95_context: 32_768,
            ..Workload::default()
        }
    }

    fn args_for(concurrency: u32, prefix: bool) -> Vec<String> {
        let mut plan = Sglang
            .plan(&gpu(80, 1), &model(), &workload(concurrency))
            .unwrap();
        plan.prefix_caching = prefix;
        launch_args(&plan)
    }

    #[test]
    fn wanting_prefix_reuse_emits_nothing_rather_than_the_opposite() {
        // The trap this file exists to avoid. Radix caching is on by default, so
        // "enable prefix reuse" must produce no flag — certainly not a disable.
        let on = args_for(4, true);
        assert!(!on.iter().any(|a| a.contains("radix")), "{on:?}");
        assert!(!on.iter().any(|a| a.contains("prefix-caching")), "{on:?}");

        let off = args_for(4, false);
        assert!(
            off.contains(&"--disable-radix-cache".to_string()),
            "{off:?}"
        );
    }

    #[test]
    fn the_flags_are_sglangs_own_not_vllms_renamed() {
        let a = args_for(4, true);
        for sglang in [
            "--context-length",
            "--max-running-requests",
            "--mem-fraction-static",
            "--model-path",
        ] {
            assert!(a.contains(&sglang.to_string()), "missing {sglang}: {a:?}");
        }
        for vllm_only in [
            "--max-model-len",
            "--max-num-seqs",
            "--gpu-memory-utilization",
            "--enable-prefix-caching",
        ] {
            assert!(
                !a.contains(&vllm_only.to_string()),
                "leaked {vllm_only}: {a:?}"
            );
        }
    }

    #[test]
    fn tensor_parallel_uses_tp_size_and_only_when_sharding() {
        let single = Sglang.plan(&gpu(80, 1), &model(), &workload(4)).unwrap();
        assert!(!launch_args(&single).contains(&"--tp-size".to_string()));

        let multi = Sglang.plan(&gpu(80, 4), &model(), &workload(4)).unwrap();
        let a = launch_args(&multi);
        assert!(a.contains(&"--tp-size".to_string()), "{a:?}");
        assert!(!a.contains(&"--tensor-parallel-size".to_string()), "{a:?}");
    }

    #[test]
    fn speculative_decoding_is_dropped_and_says_so_rather_than_guessing() {
        // Low concurrency, so the shared planner would have enabled it.
        let plan = Sglang.plan(&gpu(80, 1), &model(), &workload(2)).unwrap();
        assert!(plan.spec_decode.is_none());
        assert!(
            plan.rationale
                .iter()
                .any(|r| r.contains("not on the verified")),
            "{:?}",
            plan.rationale
        );
        let a = launch_args(&plan);
        assert!(!a.iter().any(|x| x.contains("spec")), "{a:?}");
    }

    #[test]
    fn apple_and_cpu_are_refused_with_different_reasons() {
        for (acc, needle) in [
            (Accelerator::Apple, "Metal"),
            (Accelerator::Cpu, "RadixAttention throughput"),
        ] {
            let hw = Hardware {
                accelerator: acc,
                ..gpu(64, 1)
            };
            let f = Sglang.supports(&hw, &model());
            assert!(!f.is_usable(), "{acc:?} should be refused");
            assert!(f.reason().unwrap().contains(needle), "{:?}", f.reason());
            assert!(Sglang.plan(&hw, &model(), &workload(4)).is_err());
        }
    }

    #[test]
    fn a_model_too_big_for_the_card_is_refused_with_the_numbers() {
        let f = Sglang.supports(&gpu(8, 1), &model());
        let reason = f.reason().unwrap();
        assert!(!f.is_usable());
        assert!(reason.contains("GiB usable"), "{reason}");
    }

    #[test]
    fn every_target_renders_a_runnable_artifact_with_provenance() {
        let plan = Sglang.plan(&gpu(160, 2), &model(), &workload(8)).unwrap();
        for (target, path, needle) in [
            (Target::LocalProcess, "serve.sh", "sglang.launch_server"),
            (Target::Container, "compose.yaml", "lmsysorg/sglang"),
            (Target::Kubernetes, "deployment.yaml", "kind: Deployment"),
        ] {
            let arts = Sglang.render(&plan, target).unwrap();
            assert_eq!(arts.len(), 1);
            assert_eq!(arts[0].path, path);
            assert!(arts[0].contents.contains(needle), "{}", arts[0].contents);
            // NFR-4: it must say what was chosen and that it runs standalone.
            assert!(arts[0].contents.contains("sglang"), "{}", arts[0].contents);
            assert!(arts[0].contents.contains('#'), "no provenance header");
            // And it must never carry a vLLM flag.
            assert!(
                !arts[0].contents.contains("--max-model-len"),
                "vLLM flag leaked into {path}"
            );
        }
    }

    #[test]
    fn the_kubernetes_name_is_a_valid_dns_label() {
        let plan = Sglang.plan(&gpu(80, 1), &model(), &workload(4)).unwrap();
        let y = &Sglang.render(&plan, Target::Kubernetes).unwrap()[0].contents;
        assert!(y.contains("name: qwen3-32b"), "{y}");
    }

    #[test]
    fn sizing_matches_vllm_because_the_arithmetic_is_shared() {
        // Two engines disagreeing about whether a model fits would be worse than
        // either being wrong, so the solve is not duplicated.
        let (hw, m, wl) = (gpu(80, 1), model(), workload(4));
        let a = Sglang.plan(&hw, &m, &wl).unwrap();
        let b = super::super::vllm::Vllm.plan(&hw, &m, &wl).unwrap();
        assert_eq!(a.quant, b.quant);
        assert_eq!(a.max_model_len, b.max_model_len);
        assert_eq!(a.max_num_seqs, b.max_num_seqs);
    }
}
