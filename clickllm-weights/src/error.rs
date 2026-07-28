//! Errors from weight acquisition. Every variant names what failed and where.

use std::path::PathBuf;

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything weight acquisition can fail with.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Downloaded bytes did not match the expected digest.
    #[error("checksum mismatch for {file}: expected {expected}, got {actual}")]
    Checksum {
        /// File that failed verification.
        file: String,
        /// Digest the manifest promised.
        expected: String,
        /// Digest the bytes actually produced.
        actual: String,
    },

    /// The remote could not be reached, or answered with an error.
    #[error("fetching {url} failed: {reason}")]
    Fetch {
        /// What was being fetched.
        url: String,
        /// Why it failed.
        reason: String,
    },

    /// The server does not support resuming, and a partial file exists.
    #[error("{url} does not support range requests; delete the .part file to restart")]
    NoResume {
        /// The URL that refused a Range request.
        url: String,
    },

    /// An entry is loaded by a running engine and must not be touched.
    #[error("cache entry {key} is in use and cannot be removed")]
    Pinned {
        /// The pinned cache key.
        key: String,
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
