//! Hardware, model, and workload descriptions shared by every runtime.
//!
//! These mirror the Python control-plane types. The sizing arithmetic lives on
//! [`ModelSpec`] because getting it wrong is silent: a GQA model sized with
//! attention-head counts overestimates KV by up to 8×, and an MLA model sized with
//! the GQA formula overestimates by ~50×.

use serde::{Deserialize, Serialize};

/// Bytes in a gibibyte.
pub const GIB: u64 = 1024 * 1024 * 1024;

/// Accelerator family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Accelerator {
    /// Apple Silicon, unified memory, Metal.
    Apple,
    /// NVIDIA, CUDA.
    Nvidia,
    /// AMD, ROCm.
    Amd,
    /// No accelerator.
    Cpu,
}

/// A deployment target's compute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hardware {
    /// Accelerator family.
    pub accelerator: Accelerator,
    /// Human-readable name (`M4 Max`, `NVIDIA H100 80GB HBM3`).
    pub name: String,
    /// Memory usable for inference, after the engine's own reservation.
    pub usable_bytes: u64,
    /// Peak memory bandwidth, when known. Decode is bandwidth-bound, so this
    /// drives the throughput roofline; `None` means we do not estimate.
    pub bandwidth_gbps: Option<f64>,
    /// Number of accelerator devices. `> 1` implies tensor parallelism.
    pub devices: u32,
}

impl Hardware {
    /// Short description used in error messages.
    pub fn describe(&self) -> String {
        let d = if self.devices > 1 {
            format!("{}× ", self.devices)
        } else {
            String::new()
        };
        format!(
            "{d}{} ({:.0} GiB usable)",
            self.name,
            self.usable_bytes as f64 / GIB as f64
        )
    }
}

/// How KV cache is stored. The formula differs enough between these that using
/// the wrong one is an order-of-magnitude sizing error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvScheme {
    /// Multi-head attention: one KV head per attention head.
    Mha,
    /// Grouped-query attention: `kv_heads` ≪ attention heads.
    Gqa,
    /// Multi-head latent attention (DeepSeek family): K and V compressed into one
    /// low-rank latent of `kv_lora_rank` per layer.
    Mla,
}

/// Everything a runtime needs to know about a model to size and configure it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Catalogue id.
    pub id: String,
    /// Total parameters in billions — **all** experts, for MoE.
    pub params_b: f64,
    /// Parameters active per token. Equals `params_b` for dense models.
    pub active_b: f64,
    /// Transformer blocks.
    pub layers: u32,
    /// KV heads (not attention heads).
    pub kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// KV storage scheme.
    pub kv_scheme: KvScheme,
    /// Compressed KV dimension per layer. Required when `kv_scheme` is [`KvScheme::Mla`].
    pub kv_lora_rank: Option<u32>,
    /// Maximum context the weights support.
    pub max_context: u32,
    /// Published licence id.
    pub licence: String,
}

impl ModelSpec {
    /// True when only a subset of parameters is active per token.
    pub fn is_moe(&self) -> bool {
        self.active_b < self.params_b * 0.95
    }

    /// Resident weight bytes at `bits_per_weight`.
    ///
    /// MoE sizes on **total** parameters: sparsity reduces compute, not residency.
    /// This is the single most common sizing error in the wild.
    pub fn weight_bytes(&self, bits_per_weight: f64) -> u64 {
        bytes_from(self.params_b, bits_per_weight)
    }

    /// KV cache bytes per token of context, per sequence.
    pub fn kv_bytes_per_token(&self, kv_dtype_bytes: u32) -> u64 {
        let per_layer = match (self.kv_scheme, self.kv_lora_rank) {
            // MLA stores one compressed latent instead of 2 × kv_heads × head_dim.
            (KvScheme::Mla, Some(rank)) => u64::from(rank),
            // An MLA model without a rank would silently size ~50× too high, so
            // fall back to the conservative GQA formula rather than guess low.
            _ => 2 * u64::from(self.kv_heads) * u64::from(self.head_dim),
        };
        per_layer * u64::from(self.layers) * u64::from(kv_dtype_bytes)
    }

    /// Bytes read per decoded token — the throughput roofline's numerator.
    /// Scales with **active** parameters, which is why sparse MoE decodes fast.
    pub fn read_bytes_per_token(&self, bits_per_weight: f64) -> u64 {
        bytes_from(self.active_b, bits_per_weight)
    }
}

/// Billions of parameters at `bits` each, in bytes.
///
/// Floors deliberately. NaN and negative inputs collapse to 0 rather than
/// wrapping to a huge number — an under-estimate surfaces as "does not fit",
/// which is recoverable; a wrapped over-estimate would look like plenty of room.
fn bytes_from(params_b: f64, bits: f64) -> u64 {
    let bytes = params_b * 1e9 * bits / 8.0;
    if !bytes.is_finite() || bytes <= 0.0 {
        return 0;
    }
    // f64 represents every integer below 2^53 exactly, and the clamp above makes
    // the conversion total, so the truncation here is the intended floor.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        bytes.min(u64::MAX as f64) as u64
    }
}

/// The traffic shape a deployment must serve. Sourced from observed traffic
/// (stage ②) rather than from a model's advertised maximum — nobody actually
/// sends 1M tokens, and sizing for it wastes the machine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Workload {
    /// 95th-percentile prompt+completion length actually observed.
    pub p95_context: u32,
    /// Concurrent requests to plan for.
    pub concurrency: u32,
    /// Fraction of requests sharing a prompt prefix, in `0.0..=1.0`. Drives whether
    /// prefix/radix caching is worth enabling.
    pub prefix_share: f64,
}

impl Default for Workload {
    /// A single-stream, no-prefix-sharing workload — the conservative assumption
    /// when no traffic has been observed yet.
    fn default() -> Self {
        Self {
            p95_context: 8192,
            concurrency: 1,
            prefix_share: 0.0,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn dense() -> ModelSpec {
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

    fn mla() -> ModelSpec {
        ModelSpec {
            id: "deepseek-v3".into(),
            params_b: 671.0,
            active_b: 37.0,
            layers: 61,
            kv_heads: 128,
            head_dim: 128,
            kv_scheme: KvScheme::Mla,
            kv_lora_rank: Some(512),
            max_context: 131_072,
            licence: "mit".into(),
        }
    }

    #[test]
    fn degenerate_sizes_floor_to_zero_rather_than_wrapping() {
        // A wrapped huge value would read as "plenty of room"; zero reads as
        // "does not fit", which is the recoverable direction.
        assert_eq!(bytes_from(f64::NAN, 4.0), 0);
        assert_eq!(bytes_from(-1.0, 4.0), 0);
        assert_eq!(bytes_from(f64::INFINITY, 4.0), 0);
        assert_eq!(bytes_from(0.0, 4.0), 0);
    }

    #[test]
    fn moe_is_detected_from_the_active_fraction() {
        assert!(!dense().is_moe());
        assert!(mla().is_moe());
    }

    #[test]
    fn moe_weights_size_on_total_not_active_params() {
        let m = mla();
        let w = m.weight_bytes(4.5);
        assert!(
            w > 300 * GIB,
            "671B at q4 must exceed 300 GiB; sizing on 37B active would be the bug"
        );
        assert!(w > m.read_bytes_per_token(4.5) * 10);
    }

    #[test]
    fn gqa_kv_matches_the_hand_computation() {
        // 64 layers × 2 × 8 kv_heads × 128 dim × 2 bytes
        assert_eq!(dense().kv_bytes_per_token(2), 64 * 2 * 8 * 128 * 2);
    }

    #[test]
    fn mla_kv_is_far_smaller_than_gqa_would_suggest() {
        let m = mla();
        let actual = m.kv_bytes_per_token(2);
        let as_if_gqa = 2 * u64::from(m.kv_heads) * u64::from(m.head_dim) * u64::from(m.layers) * 2;
        assert!(
            actual * 10 < as_if_gqa,
            "MLA {actual} must be far below the GQA figure {as_if_gqa}"
        );
    }

    #[test]
    fn mla_without_a_rank_falls_back_conservatively_rather_than_guessing_low() {
        let mut m = mla();
        m.kv_lora_rank = None;
        assert!(
            m.kv_bytes_per_token(2) > mla().kv_bytes_per_token(2),
            "a missing rank must over-estimate, never under-estimate"
        );
    }

    #[test]
    fn sparse_models_read_far_fewer_bytes_per_token_than_a_dense_model_of_equal_total_size() {
        let sparse = mla(); // 671B total, 37B active
        let dense_equivalent = ModelSpec {
            active_b: sparse.params_b, // same weights, no sparsity
            ..sparse.clone()
        };
        assert!(
            sparse.read_bytes_per_token(4.5) * 10 < dense_equivalent.read_bytes_per_token(4.5),
            "37B of 671B active must read roughly 18x less per token"
        );
        // ...while both still occupy the same memory. Sparsity buys speed, not space.
        assert_eq!(sparse.weight_bytes(4.5), dense_equivalent.weight_bytes(4.5));
    }

    #[test]
    fn kv_scales_linearly_in_dtype_width() {
        let m = dense();
        assert_eq!(m.kv_bytes_per_token(1) * 2, m.kv_bytes_per_token(2));
    }

    #[test]
    fn hardware_describe_mentions_device_count_only_when_plural() {
        let one = Hardware {
            accelerator: Accelerator::Nvidia,
            name: "H100".into(),
            usable_bytes: 80 * GIB,
            bandwidth_gbps: Some(3350.0),
            devices: 1,
        };
        assert!(!one.describe().contains('×'));
        let many = Hardware { devices: 4, ..one };
        assert!(many.describe().contains("4×"), "{}", many.describe());
    }

    #[test]
    fn default_workload_is_the_conservative_one() {
        let w = Workload::default();
        assert_eq!(w.concurrency, 1);
        assert_eq!(w.prefix_share, 0.0);
    }
}
