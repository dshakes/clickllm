//! Capture — recording traffic so it can become an eval set.
//!
//! The whole module exists under one constraint: **unredacted prompt text must
//! never touch disk.** Redaction runs on the write path, and a redaction failure
//! drops the capture rather than storing it. See [`mod@redact`].

pub mod redact;

pub use redact::{Kind, Refused, Report, redact};
