//! # clickllm-gateway
//!
//! The OpenAI-compatible datapath: routing, shadow mirroring, quality-gated
//! canary, and token metering.
//!
//! ## What makes this a *migration* router, not a load balancer
//!
//! Once traffic is fully cut over, balancing is handed to GAIE/llm-d and we leave
//! the datapath entirely. Until then this layer does three things no balancer
//! does: it mirrors traffic to a candidate that is **scored but never served**,
//! it splits by percentage **deterministically** so a retry lands the same way,
//! and it applies **per-cluster policy** so the tasks where the open model loses
//! keep going to the incumbent.
//!
//! ## Constraints
//!
//! - **Sub-15ms added p95** (NFR-1). Nothing expensive happens here; judgment
//!   lives in the control plane.
//! - **Never buffer a stream.** Bytes reach the client as they arrive and are
//!   inspected in flight. See [`sse`] and [`meter`].
//! - **Never fabricate a usage figure.** A missing count is reported missing, not
//!   as zero — understating spend is the one direction a cost ledger must not err.

#![doc(html_root_url = "https://docs.rs/clickllm-gateway")]

pub mod meter;
pub mod proxy;
pub mod router;
pub mod sse;

pub use meter::{Meter, Metered, Usage};
pub use proxy::{AppState, Record, app};
pub use router::{Backend, Decision, Phase, Route, Router};
pub use sse::{Decoder, Event};

/// Crate version, for request logs and provenance.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
