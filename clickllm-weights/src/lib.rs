//! # clickllm-weights
//!
//! Getting model weights onto a machine, correctly and once.
//!
//! Weights are the largest and slowest thing this tool touches, so the module is
//! shaped around three failure modes that matter more than throughput:
//!
//! - **A half-downloaded file that looks complete.** Partial fetches live beside
//!   their target and only become an entry after their digest verifies; the
//!   rename is the commit point.
//! - **Paying twice for the same bytes.** Entries are keyed by canonical
//!   reference, so `Q4_K_M` and `q4-k-m` are one entry.
//! - **Evicting weights out from under a running engine.** In-use entries are
//!   pinned and never evicted, even when that means staying over budget.
//!
//! Licence gating happens in [`clickllm_core::licence`] and runs *before* any
//! bytes move — the point of the gate is to be cheap to obey.

#![doc(html_root_url = "https://docs.rs/clickllm-weights")]

pub mod cache;
pub mod error;
pub mod fetch;

pub use cache::{Cache, Entry};
pub use error::{Error, Result};
pub use fetch::{Download, Progress, fetch_file};
