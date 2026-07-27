//! # clickllm-core
//!
//! The datapath and weights core: model acquisition, runtime planning, and native
//! configuration generation.
//!
//! ## Design rules that are load-bearing
//!
//! - **No engine-specific type escapes [`runtime::Runtime`].** The moment one does,
//!   portability is gone and development on Apple Silicon stops working, because
//!   vLLM/SGLang/llm-d are CUDA-only. See ADR-0002.
//! - **[`render`](runtime::Runtime::render) emits native config**, never a wrapper.
//!   Output must run with clickllm uninstalled. See ADR-0002 and NFR-4.
//! - **Licences are gated before bytes move.** See [`licence`].
//! - **Nothing here panics.** `unwrap`, `expect`, `panic!`, and slice indexing are
//!   denied at the lint level; fallible paths return [`Error`].
//!
//! ## Layout
//!
//! | Module | Milestone | Role |
//! |---|---|---|
//! | [`model_ref`] | M1 | parse and canonicalise references to weights |
//! | [`licence`] | M1 | refuse or oblige, before download |
//! | [`runtime`] | M2 | the `Runtime` trait and its backends |
//!
//! ## Observability
//!
//! Every fallible operation runs inside a [`tracing`] span carrying the identifiers
//! that make a failure diagnosable (model, runtime, path). The crate emits events
//! and never installs a subscriber — that is the binary's choice.

#![doc(html_root_url = "https://docs.rs/clickllm-core")]

pub mod error;
pub mod licence;
pub mod model_ref;
pub mod runtime;
pub mod spec;

pub use error::{Error, Result};
pub use licence::{Intent, Policy, Verdict};
pub use model_ref::{ModelRef, Source};
pub use runtime::{Endpoint, Feasibility, Runtime, RuntimePlan, Target};
pub use spec::{Accelerator, Hardware, KvScheme, ModelSpec, Workload};

/// Crate version, for manifests and provenance headers.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    #[test]
    fn version_is_populated() {
        assert!(!super::VERSION.is_empty());
    }
}
