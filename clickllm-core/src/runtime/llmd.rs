//! llm-d backend — multi-node disaggregated serving with KV-cache-aware routing.
//!
//! Emits a Gateway API Inference Extension `InferencePool` plus the decode/prefill
//! Deployments. Disaggregation only pays off at scale, so [`Runtime::supports`]
//! refuses single-device hosts rather than producing a config that is strictly
//! worse than plain vLLM there.

use crate::error::{Error, Result};
use crate::runtime::{Artifact, Feasibility, Runtime, RuntimePlan, Target, provenance, vllm::Vllm};
use crate::spec::{Accelerator, Hardware, ModelSpec, Workload};

/// Devices below which disaggregation costs more than it saves.
pub const MIN_DEVICES: u32 = 2;

/// The llm-d runtime.
#[derive(Debug, Default, Clone, Copy)]
pub struct LlmD;

impl LlmD {
    /// Construct.
    pub fn new() -> Self {
        Self
    }
}

impl Runtime for LlmD {
    fn name(&self) -> &'static str {
        "llm-d"
    }

    fn supports(&self, hw: &Hardware, model: &ModelSpec) -> Feasibility {
        if !matches!(hw.accelerator, Accelerator::Nvidia | Accelerator::Amd) {
            return Feasibility::No {
                reason: "llm-d requires CUDA or ROCm devices".into(),
            };
        }
        if hw.devices < MIN_DEVICES {
            return Feasibility::No {
                reason: format!(
                    "disaggregated prefill/decode needs at least {MIN_DEVICES} devices; \
                     on {} device(s) plain vLLM is faster and simpler",
                    hw.devices
                ),
            };
        }
        // Sizing is vLLM's: llm-d runs vLLM workers underneath.
        Vllm::new().supports(hw, model)
    }

    fn plan(&self, hw: &Hardware, model: &ModelSpec, wl: &Workload) -> Result<RuntimePlan> {
        if let Feasibility::No { reason } = self.supports(hw, model) {
            return Err(Error::RuntimePlan {
                runtime: "llm-d",
                reason,
            });
        }
        let mut plan = Vllm::new().plan(hw, model, wl)?;
        plan.rationale.push(format!(
            "llm-d across {} devices: prefill and decode scale independently, and the \
             endpoint picker routes on KV-cache locality rather than round-robin",
            hw.devices
        ));
        // Prefix sharing is what KV-aware routing exploits, so it is always on here.
        if !plan.prefix_caching {
            plan.prefix_caching = true;
            plan.rationale.push(
                "prefix caching forced on — KV-cache-aware routing has nothing to route on \
                 without it"
                    .into(),
            );
        }
        Ok(plan)
    }

    fn render(&self, plan: &RuntimePlan, target: Target) -> Result<Vec<Artifact>> {
        if target != Target::Kubernetes {
            return Err(Error::RuntimePlan {
                runtime: "llm-d",
                reason: format!("llm-d is Kubernetes-only; {target:?} is not a valid target"),
            });
        }
        let name = super::vllm::k8s_name_for(&plan.model.id);
        Ok(vec![Artifact {
            path: "inferencepool.yaml".into(),
            contents: format!(
                "{}apiVersion: inference.networking.x-k8s.io/v1alpha2\n\
                 kind: InferencePool\nmetadata:\n  name: {name}\nspec:\n  \
                 targetPortNumber: 8000\n  selector:\n    app: {name}\n  \
                 extensionRef:\n    name: {name}-epp\n---\n\
                 apiVersion: inference.networking.x-k8s.io/v1alpha2\n\
                 kind: InferenceModel\nmetadata:\n  name: {name}\nspec:\n  \
                 modelName: {}\n  criticality: Standard\n  poolRef:\n    name: {name}\n",
                provenance("llm-d", plan, "#"),
                plan.model.id,
            ),
        }])
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::runtime::vllm::tests::{h100, model};
    use crate::spec::GIB;

    fn multi(devices: u32) -> Hardware {
        Hardware {
            devices,
            usable_bytes: 72 * GIB * u64::from(devices),
            ..h100()
        }
    }

    #[test]
    fn refuses_single_device_and_points_at_the_simpler_option() {
        let f = LlmD::new().supports(&multi(1), &model());
        assert!(!f.is_usable());
        assert!(f.reason().unwrap().contains("vLLM"), "{f:?}");
    }

    #[test]
    fn refuses_non_cuda() {
        let hw = Hardware {
            accelerator: Accelerator::Apple,
            ..multi(4)
        };
        assert!(!LlmD::new().supports(&hw, &model()).is_usable());
    }

    #[test]
    fn accepts_multi_device_and_explains_the_topology() {
        let p = LlmD::new()
            .plan(&multi(4), &model(), &Workload::default())
            .unwrap();
        assert!(p.rationale.iter().any(|r| r.contains("KV-cache")));
        assert_eq!(p.tensor_parallel, 4);
    }

    #[test]
    fn prefix_caching_is_forced_on_because_routing_depends_on_it() {
        let wl = Workload {
            prefix_share: 0.0,
            ..Workload::default()
        };
        let p = LlmD::new().plan(&multi(4), &model(), &wl).unwrap();
        assert!(p.prefix_caching);
        assert!(p.rationale.iter().any(|r| r.contains("forced on")));
    }

    #[test]
    fn renders_an_inferencepool_with_provenance() {
        let p = LlmD::new()
            .plan(&multi(4), &model(), &Workload::default())
            .unwrap();
        let arts = LlmD::new().render(&p, Target::Kubernetes).unwrap();
        let c = &arts[0].contents;
        assert!(c.contains("InferencePool"));
        assert!(c.contains("InferenceModel"));
        assert!(c.contains("clickllm"));
    }

    #[test]
    fn refuses_non_kubernetes_targets_rather_than_emitting_nonsense() {
        let p = LlmD::new()
            .plan(&multi(4), &model(), &Workload::default())
            .unwrap();
        for t in [Target::LocalProcess, Target::Container] {
            assert!(
                LlmD::new().render(&p, t).is_err(),
                "{t:?} should be refused"
            );
        }
    }
}
