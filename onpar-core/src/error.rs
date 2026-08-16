//! One error type for the crate.
//!
//! Every variant carries the offending value. An error a user cannot act on is a
//! bug report we will receive instead of a fix they could have made themselves.

use std::path::PathBuf;

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything `onpar-core` can fail with.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A model reference could not be parsed.
    #[error("bad model reference {input:?}: {reason}")]
    ModelRef {
        /// What the caller passed.
        input: String,
        /// Why it was rejected.
        reason: String,
    },

    /// Licence policy refused this model before any bytes moved.
    #[error("licence {licence:?} for {model} is not permitted: {reason}")]
    LicenceDenied {
        /// Model the policy was evaluated for.
        model: String,
        /// SPDX id or family name as published.
        licence: String,
        /// Which rule refused it.
        reason: String,
    },

    /// A licence needs explicit acknowledgement that was not given.
    #[error("licence {licence:?} for {model} requires acknowledgement: {obligation}")]
    LicenceNotAcknowledged {
        /// Model the policy was evaluated for.
        model: String,
        /// SPDX id or family name as published.
        licence: String,
        /// What the user is agreeing to.
        obligation: String,
    },

    /// Downloaded bytes did not match the expected digest.
    #[error("checksum mismatch for {path}: expected {expected}, got {actual}")]
    Checksum {
        /// File that failed verification.
        path: PathBuf,
        /// Digest the manifest promised.
        expected: String,
        /// Digest the bytes actually produced.
        actual: String,
    },

    /// No runtime can serve this model on this hardware.
    #[error("no runtime supports {model} on {hardware}: {reason}")]
    Unsupported {
        /// Model that could not be placed.
        model: String,
        /// Hardware description it was evaluated against.
        hardware: String,
        /// Why every candidate was rejected.
        reason: String,
    },

    /// A runtime rejected the plan it was handed.
    #[error("{runtime} cannot execute this plan: {reason}")]
    RuntimePlan {
        /// Runtime that refused.
        runtime: &'static str,
        /// Why.
        reason: String,
    },

    /// Filesystem failure, with the path that caused it.
    #[error("{op} failed for {path}")]
    Io {
        /// What was being attempted.
        op: &'static str,
        /// Path involved.
        path: PathBuf,
        /// Underlying cause.
        #[source]
        source: std::io::Error,
    },
}

impl Error {
    /// Wrap an [`std::io::Error`] with the path and operation that produced it.
    pub fn io(op: &'static str, path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            op,
            path: path.into(),
            source,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_offending_value() {
        let e = Error::ModelRef {
            input: "nonsense".into(),
            reason: "no slash".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("nonsense"), "{msg}");
        assert!(msg.contains("no slash"), "{msg}");
    }

    #[test]
    fn io_errors_keep_path_and_source() {
        let e = Error::io(
            "read",
            "/tmp/x",
            std::io::Error::new(std::io::ErrorKind::NotFound, "gone"),
        );
        assert!(e.to_string().contains("/tmp/x"));
        assert!(
            std::error::Error::source(&e).is_some(),
            "source must survive"
        );
    }

    #[test]
    fn checksum_error_shows_both_digests() {
        let msg = Error::Checksum {
            path: "/w/model.gguf".into(),
            expected: "aaa".into(),
            actual: "bbb".into(),
        }
        .to_string();
        assert!(msg.contains("aaa") && msg.contains("bbb"), "{msg}");
    }
}
