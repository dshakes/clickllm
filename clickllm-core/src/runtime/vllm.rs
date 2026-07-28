//! vLLM backend — CUDA/ROCm, the broadest-support engine.
//!
//! Renders a real `vllm serve` invocation, a compose file, or a Kubernetes
//! Deployment. Output runs with clickllm uninstalled (NFR-4).

use crate::error::{Error, Result};
use crate::runtime::{Artifact, Feasibility, Runtime, RuntimePlan, SpecDecode, Target, provenance};
use crate::spec::{Accelerator, Hardware, ModelSpec, Workload};

/// Non-weight, non-KV memory: activations, workspace, allocator fragmentation.
const OVERHEAD_FRACTION: f64 = 0.08;
/// Fixed runtime floor on top of the proportional overhead.
const OVERHEAD_FLOOR_BYTES: u64 = 1_610_612_736; // 1.5 GiB

/// Quantisations vLLM accepts, highest precision first. Capped at 8-bit: decode is
/// bandwidth-bound, so 16-bit halves tokens/sec for inference-quality gains that
/// are negligible.
const QUANTS: &[(&str, f64)] = &[("fp8", 8.0), ("awq", 4.5), ("gptq", 4.0)];

/// The vLLM runtime.
#[derive(Debug, Default, Clone, Copy)]
pub struct Vllm;

impl Vllm {
    /// Construct.
    pub fn new() -> Self {
        Self
    }

    /// Total bytes a configuration needs.
    fn required_bytes(model: &ModelSpec, bits: f64, ctx: u32, seqs: u32) -> u64 {
        let w = model.weight_bytes(bits);
        // Saturating throughout: an overflowed requirement must read as "too big"
        // and refuse, never wrap to a small number and appear to fit.
        let kv = model
            .kv_bytes_per_token(2)
            .saturating_mul(u64::from(ctx))
            .saturating_mul(u64::from(seqs));
        // A fraction of a byte count is a floor by construction; the product cannot
        // exceed the u64 it came from.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let proportional = (w as f64 * OVERHEAD_FRACTION) as u64;
        w.saturating_add(kv)
            .saturating_add(proportional)
            .saturating_add(OVERHEAD_FLOOR_BYTES)
    }
}

impl Runtime for Vllm {
    fn name(&self) -> &'static str {
        "vllm"
    }

    fn supports(&self, hw: &Hardware, model: &ModelSpec) -> Feasibility {
        match hw.accelerator {
            Accelerator::Apple => Feasibility::No {
                reason: "vLLM is CUDA/ROCm-only; Apple Silicon has no Metal backend \
                         in upstream vLLM"
                    .into(),
            },
            Accelerator::Cpu => Feasibility::Degraded {
                reason: "no accelerator — vLLM on CPU will be an order of magnitude \
                         slower than a GPU deployment"
                    .into(),
            },
            Accelerator::Nvidia | Accelerator::Amd => {
                // Weights alone must fit; KV can be traded down, weights cannot.
                let smallest = QUANTS.last().map_or(4.0, |q| q.1);
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
        let span =
            tracing::info_span!("vllm_plan", model = %model.id, concurrency = wl.concurrency);
        let _e = span.enter();

        if let Feasibility::No { reason } = self.supports(hw, model) {
            return Err(Error::RuntimePlan {
                runtime: "vllm",
                reason,
            });
        }

        let ctx = wl.p95_context.min(model.max_context);
        let mut rationale = Vec::new();
        if ctx < model.max_context {
            rationale.push(format!(
                "max_model_len {ctx} from observed p95, not the model's {} maximum — \
                 sizing for unused context wastes KV memory",
                model.max_context
            ));
        }

        // Highest precision that fits at the observed shape.
        let mut chosen: Option<(&str, f64)> = None;
        for (label, bits) in QUANTS {
            if Self::required_bytes(model, *bits, ctx, wl.concurrency) <= hw.usable_bytes {
                chosen = Some((label, *bits));
                break;
            }
        }
        let (quant, bits) = chosen.ok_or_else(|| Error::RuntimePlan {
            runtime: "vllm",
            reason: format!(
                "no quantisation fits {} at {ctx} context × {} concurrent on {}",
                model.id,
                wl.concurrency,
                hw.describe()
            ),
        })?;
        rationale.push(format!(
            "quant {quant} — highest precision that fits at {ctx} ctx × {} concurrent",
            wl.concurrency
        ));

        let spec_decode = SpecDecode::for_workload(wl);
        match &spec_decode {
            Some(sd) => rationale.push(format!(
                "speculative decoding {} with {} draft tokens — expect ~1.3–1.8× at this \
                 concurrency, not the 2–3× single-stream figure",
                sd.method, sd.num_speculative_tokens
            )),
            None => rationale.push(format!(
                "speculative decoding disabled — at concurrency {} acceptance falls off \
                 and the draft pass would make this slower",
                wl.concurrency
            )),
        }

        // Prefix caching costs memory; only worth it when traffic actually shares prefixes.
        let prefix_caching = wl.prefix_share >= 0.20;
        rationale.push(if prefix_caching {
            format!(
                "prefix caching on — {:.0}% of observed requests share a prefix",
                wl.prefix_share * 100.0
            )
        } else {
            format!(
                "prefix caching off — only {:.0}% of requests share a prefix, so the \
                 cache would cost memory it does not earn back",
                wl.prefix_share * 100.0
            )
        });

        if hw.devices > 1 {
            rationale.push(format!(
                "tensor parallel {} to match device count",
                hw.devices
            ));
        }

        Ok(RuntimePlan {
            model: model.clone(),
            quant: quant.to_owned(),
            bits_per_weight: bits,
            max_model_len: ctx,
            max_num_seqs: wl.concurrency.max(1),
            tensor_parallel: hw.devices.max(1),
            spec_decode,
            prefix_caching,
            memory_utilization: 0.90,
            rationale,
        })
    }

    fn render(&self, plan: &RuntimePlan, target: Target) -> Result<Vec<Artifact>> {
        let args = serve_args(plan);
        match target {
            Target::LocalProcess => Ok(vec![Artifact {
                path: "serve.sh".into(),
                contents: format!(
                    "#!/usr/bin/env bash\nset -euo pipefail\n\n{}\nexec {}\n",
                    provenance("vllm", plan, "#"),
                    args.join(" \\\n  ")
                ),
            }]),
            Target::Container => Ok(vec![Artifact {
                path: "compose.yaml".into(),
                contents: format!(
                    "{}services:\n  vllm:\n    image: vllm/vllm-openai:latest\n    \
                     command: [{}]\n    ports: [\"8000:8000\"]\n    \
                     deploy:\n      resources:\n        reservations:\n          \
                     devices:\n            - capabilities: [gpu]\n              \
                     count: {}\n",
                    provenance("vllm", plan, "#"),
                    args.iter()
                        .skip(1)
                        .map(|a| format!("\"{a}\""))
                        .collect::<Vec<_>>()
                        .join(", "),
                    plan.tensor_parallel,
                ),
            }]),
            Target::Kubernetes => Ok(vec![Artifact {
                path: "deployment.yaml".into(),
                contents: format!(
                    "{}apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: {}\n\
                     spec:\n  replicas: 1\n  selector:\n    matchLabels:\n      app: {}\n  \
                     template:\n    metadata:\n      labels:\n        app: {}\n    spec:\n      \
                     containers:\n        - name: vllm\n          \
                     image: vllm/vllm-openai:latest\n          args: [{}]\n          \
                     ports:\n            - containerPort: 8000\n          \
                     resources:\n            limits:\n              nvidia.com/gpu: {}\n",
                    provenance("vllm", plan, "#"),
                    k8s_name_for(&plan.model.id),
                    k8s_name_for(&plan.model.id),
                    k8s_name_for(&plan.model.id),
                    args.iter()
                        .skip(1)
                        .map(|a| format!("\"{a}\""))
                        .collect::<Vec<_>>()
                        .join(", "),
                    plan.tensor_parallel,
                ),
            }]),
            Target::Systemd => Ok(vec![Artifact {
                path: "clickllm-vllm.service".into(),
                contents: super::systemd_unit(&plan.model.id, &args, plan),
            }]),
        }
    }
}

/// Kubernetes names are DNS-1123 labels: lowercase alphanumeric and `-` only.
pub(crate) fn k8s_name_for(id: &str) -> String {
    let s: String = id
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let s = s.trim_matches('-').to_owned();
    if s.is_empty() { "model".into() } else { s }
}

/// The native `vllm serve` argument vector.
fn serve_args(plan: &RuntimePlan) -> Vec<String> {
    let mut a = vec![
        "vllm".into(),
        "serve".into(),
        plan.model.id.clone(),
        "--quantization".into(),
        plan.quant.clone(),
        "--max-model-len".into(),
        plan.max_model_len.to_string(),
        "--max-num-seqs".into(),
        plan.max_num_seqs.to_string(),
        "--gpu-memory-utilization".into(),
        format!("{:.2}", plan.memory_utilization),
    ];
    if plan.tensor_parallel > 1 {
        a.push("--tensor-parallel-size".into());
        a.push(plan.tensor_parallel.to_string());
    }
    if plan.prefix_caching {
        a.push("--enable-prefix-caching".into());
    }
    if let Some(sd) = &plan.spec_decode {
        a.push("--speculative-config".into());
        a.push(format!(
            r#"{{"method": "{}", "num_speculative_tokens": {}}}"#,
            sd.method, sd.num_speculative_tokens
        ));
    }
    a
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
pub(crate) mod tests {
    use super::*;
    use crate::spec::{GIB, KvScheme};

    pub(crate) fn model() -> ModelSpec {
        ModelSpec {
            id: "qwen3-32b".into(),
            params_b: 32.8,
            active_b: 32.8,
            layers: 64,
            kv_heads: 8,
            head_dim: 128,
            kv_scheme: KvScheme::Gqa,
            kv_lora_rank: None,
            max_context: 131_072,
            licence: "apache-2.0".into(),
        }
    }

    pub(crate) fn h100() -> Hardware {
        Hardware {
            accelerator: Accelerator::Nvidia,
            name: "NVIDIA H100 80GB".into(),
            usable_bytes: 72 * GIB,
            bandwidth_gbps: Some(3350.0),
            devices: 1,
        }
    }

    pub(crate) fn sample_plan() -> RuntimePlan {
        Vllm::new()
            .plan(&h100(), &model(), &Workload::default())
            .unwrap()
    }

    #[test]
    fn refuses_apple_silicon_with_an_actionable_reason() {
        let hw = Hardware {
            accelerator: Accelerator::Apple,
            name: "M4 Max".into(),
            usable_bytes: 96 * GIB,
            bandwidth_gbps: Some(546.0),
            devices: 1,
        };
        let f = Vllm::new().supports(&hw, &model());
        assert!(!f.is_usable());
        assert!(f.reason().unwrap().contains("CUDA"), "{f:?}");
        // And planning must fail rather than silently produce CUDA config.
        assert!(
            Vllm::new()
                .plan(&hw, &model(), &Workload::default())
                .is_err()
        );
    }

    #[test]
    fn refuses_when_weights_alone_cannot_fit() {
        let tiny = Hardware {
            usable_bytes: 4 * GIB,
            ..h100()
        };
        let f = Vllm::new().supports(&tiny, &model());
        assert!(!f.is_usable());
        assert!(f.reason().unwrap().contains("weights"), "{f:?}");
    }

    #[test]
    fn cpu_is_degraded_not_refused_and_says_why() {
        let cpu = Hardware {
            accelerator: Accelerator::Cpu,
            name: "x86".into(),
            devices: 1,
            ..h100()
        };
        let f = Vllm::new().supports(&cpu, &model());
        assert!(f.is_usable());
        assert!(matches!(f, Feasibility::Degraded { .. }));
    }

    #[test]
    fn context_is_capped_to_observed_p95_not_the_model_maximum() {
        let wl = Workload {
            p95_context: 18_000,
            concurrency: 4,
            prefix_share: 0.0,
        };
        let p = Vllm::new().plan(&h100(), &model(), &wl).unwrap();
        assert_eq!(p.max_model_len, 18_000);
        assert!(p.rationale.iter().any(|r| r.contains("p95")));
    }

    #[test]
    fn context_never_exceeds_what_the_weights_support() {
        let wl = Workload {
            p95_context: 999_999,
            ..Workload::default()
        };
        let p = Vllm::new().plan(&h100(), &model(), &wl).unwrap();
        assert_eq!(p.max_model_len, model().max_context);
    }

    #[test]
    fn spec_decode_is_dropped_above_the_cliff_and_explained() {
        let wl = Workload {
            concurrency: 64,
            p95_context: 2048,
            prefix_share: 0.0,
        };
        let p = Vllm::new().plan(&h100(), &model(), &wl).unwrap();
        assert!(p.spec_decode.is_none());
        assert!(
            p.rationale.iter().any(|r| r.contains("disabled")),
            "a silently dropped optimisation is a bug: {:?}",
            p.rationale
        );
        assert!(!serve_args(&p).iter().any(|a| a == "--speculative-config"));
    }

    #[test]
    fn prefix_caching_follows_measured_prefix_sharing() {
        let shared = Workload {
            prefix_share: 0.8,
            ..Workload::default()
        };
        let unshared = Workload {
            prefix_share: 0.01,
            ..Workload::default()
        };
        assert!(
            Vllm::new()
                .plan(&h100(), &model(), &shared)
                .unwrap()
                .prefix_caching
        );
        assert!(
            !Vllm::new()
                .plan(&h100(), &model(), &unshared)
                .unwrap()
                .prefix_caching
        );
    }

    #[test]
    fn tensor_parallel_matches_device_count() {
        let quad = Hardware {
            devices: 4,
            usable_bytes: 288 * GIB,
            ..h100()
        };
        let p = Vllm::new()
            .plan(&quad, &model(), &Workload::default())
            .unwrap();
        assert_eq!(p.tensor_parallel, 4);
        assert!(serve_args(&p).contains(&"--tensor-parallel-size".to_owned()));
    }

    #[test]
    fn every_plan_explains_itself() {
        let p = sample_plan();
        assert!(!p.rationale.is_empty());
        assert!(p.rationale.iter().all(|r| !r.trim().is_empty()));
    }

    #[test]
    fn renders_every_target_with_provenance() {
        let p = sample_plan();
        for target in Target::ALL {
            let arts = Vllm::new().render(&p, target).unwrap();
            assert_eq!(arts.len(), 1, "{target:?}");
            let a = &arts[0];
            assert!(!a.path.is_empty());
            assert!(
                a.contents.contains("clickllm"),
                "{target:?} lost provenance"
            );
            assert!(
                a.contents.contains(&p.model.id),
                "{target:?} lost the model"
            );
        }
    }

    #[test]
    fn local_script_is_a_runnable_native_invocation() {
        let p = sample_plan();
        let arts = Vllm::new().render(&p, Target::LocalProcess).unwrap();
        let c = &arts[0].contents;
        assert!(c.starts_with("#!/usr/bin/env bash"));
        assert!(c.contains("set -euo pipefail"));
        assert!(
            c.contains("vllm \\\n  serve") || c.contains("vllm serve"),
            "{c}"
        );
        // NFR-4: no clickllm invocation anywhere in the output.
        assert!(
            !c.lines().any(|l| l.trim_start().starts_with("clickllm")),
            "generated config must not shell back into clickllm"
        );
    }

    #[test]
    fn kubernetes_names_are_dns_1123_labels() {
        assert_eq!(k8s_name_for("Qwen/Qwen3-32B"), "qwen-qwen3-32b");
        assert_eq!(k8s_name_for("--weird--"), "weird");
        assert_eq!(k8s_name_for("!!!"), "model");
        for name in [k8s_name_for("Qwen/Qwen3-32B"), k8s_name_for("a_b.c")] {
            assert!(
                name.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "{name} is not a valid label"
            );
            assert!(!name.starts_with('-') && !name.ends_with('-'));
        }
    }

    #[test]
    fn serve_args_carry_every_derived_knob() {
        let p = sample_plan();
        let a = serve_args(&p);
        for flag in [
            "--quantization",
            "--max-model-len",
            "--max-num-seqs",
            "--gpu-memory-utilization",
        ] {
            assert!(a.iter().any(|x| x == flag), "{flag} missing from {a:?}");
        }
    }

    #[test]
    fn the_systemd_unit_survives_a_slow_start_and_restarts() {
        let p = sample_plan();
        let u = &Vllm::new().render(&p, Target::Systemd).unwrap()[0].contents;
        assert!(u.contains("Restart=always"), "{u}");
        // Weights are tens of GiB; systemd's default 90s would kill a healthy start.
        assert!(u.contains("TimeoutStartSec=1800"), "{u}");
        assert!(u.contains("clickllm"), "provenance header missing");
        // systemd splits ExecStart on whitespace, so an argument containing
        // spaces — `--speculative-config {"method": "eagle3"}` — must arrive
        // quoted or the unit starts with four broken arguments and dies.
        let exec = u
            .lines()
            .find(|l| l.starts_with("ExecStart="))
            .expect("no ExecStart");
        assert!(exec.contains("vllm"), "{exec}");
        if exec.contains("eagle3") {
            assert!(
                exec.contains("\"{\\\"method\\\""),
                "the nested JSON argument must be quoted and escaped: {exec}"
            );
        }
    }

    #[test]
    fn every_target_renders_something_runnable() {
        let p = sample_plan();
        for t in Target::ALL {
            let arts = Vllm::new()
                .render(&p, t)
                .expect("vllm supports every target");
            assert_eq!(arts.len(), 1, "{t:?}");
            assert!(!arts[0].contents.is_empty(), "{t:?} rendered nothing");
            assert!(
                arts[0].contents.contains("clickllm"),
                "{t:?} lacks provenance"
            );
        }
    }
}
