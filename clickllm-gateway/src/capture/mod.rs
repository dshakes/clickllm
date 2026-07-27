//! Capture — recording traffic so it can become an eval set.
//!
//! The whole module exists under one constraint: **unredacted prompt text must
//! never touch disk.** Redaction runs on the write path, and a redaction failure
//! drops the capture rather than storing it. See [`mod@redact`].

//! [`mod@store`] holds the other half: redaction is applied *inside* the write
//! path, so there is no way to append a record that skipped it.

pub mod redact;
pub mod store;

pub use redact::{Kind, Refused, Report, redact};
pub use store::{Capture, CaptureStore, StoreError};
