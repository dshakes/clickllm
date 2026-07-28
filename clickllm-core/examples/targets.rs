//! Render every deployment target for one plan, so they can be eyeballed.
#![allow(clippy::expect_used, clippy::print_stdout)]

use clickllm_core::runtime::vllm::Vllm;
use clickllm_core::runtime::{Runtime, Target};
use clickllm_core::spec::{Accelerator, Hardware, KvScheme, ModelSpec, Workload};

fn main() {
    let hw = Hardware {
        accelerator: Accelerator::Nvidia,
        name: "H100".into(),
        usable_bytes: 80 * clickllm_core::spec::GIB,
        bandwidth_gbps: Some(3350.0),
        devices: 2,
    };
    let model = ModelSpec {
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
    };
    let wl = Workload {
        concurrency: 4,
        p95_context: 8192,
        ..Workload::default()
    };
    let plan = Vllm::new().plan(&hw, &model, &wl).expect("plan");

    for t in Target::ALL {
        for a in Vllm::new().render(&plan, t).expect("render") {
            println!("\n{:=^76}", format!("  {t:?} → {}  ", a.path));
            println!("{}", a.contents);
        }
    }
}
