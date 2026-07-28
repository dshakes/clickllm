//! Resumable, verified download.
//!
//! A 60 GB download that fails at 58 GB and starts over is not a slow tool, it
//! is an unusable one. So:
//!
//! **Resume by default.** If a `.part` file exists we ask for the remainder with
//! a `Range` header and append. A server that ignores the range would silently
//! produce a corrupt file — appending a fresh copy of the whole body to an
//! existing partial — so a missing `206 Partial Content` is an explicit error,
//! never a hopeful retry.
//!
//! **Hash in flight.** Bytes are digested as they stream past, so verification
//! costs one pass rather than a second read of 60 GB. On resume the existing
//! partial is digested first, which costs a read but is the only way the running
//! hash can be correct.
//!
//! **Verify before committing.** A file whose digest does not match never
//! becomes a cache entry. Corrupt weights produce garbage output that looks like
//! a bad model rather than a bad download, which is a genuinely awful thing to
//! debug.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::error::{Error, Result};

/// How far along a download is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// Bytes already on disk, including anything resumed.
    pub done: u64,
    /// Total expected, when the server says.
    pub total: Option<u64>,
    /// Bytes that were already present when this attempt started.
    pub resumed_from: u64,
}

impl Progress {
    /// Completion in `0.0..=1.0`, when the total is known.
    pub fn fraction(&self) -> Option<f64> {
        match self.total {
            Some(t) if t > 0 => Some((self.done as f64 / t as f64).min(1.0)),
            _ => None,
        }
    }
}

/// One file to acquire.
#[derive(Debug, Clone)]
pub struct Download {
    /// Where to get it.
    pub url: String,
    /// Where it goes, including any `.part` staging directory.
    pub dest: PathBuf,
    /// Expected SHA-256, lowercase hex. `None` means the source published none —
    /// the download still succeeds, but nothing can be verified, and callers are
    /// told rather than left to assume it was checked.
    pub expected_sha256: Option<String>,
}

/// Outcome of a completed download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
    /// Where the bytes landed.
    pub path: PathBuf,
    /// Digest of what was written.
    pub digest: String,
    /// Total bytes now on disk.
    pub bytes: u64,
    /// Bytes that were already present and did not need re-fetching.
    pub resumed_from: u64,
    /// Whether the digest was checked against a published value. False means
    /// the source published none — not that verification failed.
    pub verified: bool,
}

/// Download `d`, resuming if a partial exists, and verify it.
///
/// `on_progress` is called as bytes arrive; keep it cheap, it runs on the
/// transfer path.
///
/// # Errors
/// [`Error::Fetch`] if the remote fails, [`Error::NoResume`] if a partial exists
/// and the server ignores `Range`, [`Error::Checksum`] if the bytes are wrong,
/// [`Error::Io`] on filesystem failure.
pub async fn fetch_file(
    client: &reqwest::Client,
    d: &Download,
    mut on_progress: impl FnMut(Progress),
) -> Result<Fetched> {
    if let Some(parent) = d.dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| Error::io("create download dir", parent, e))?;
    }

    let existing = tokio::fs::metadata(&d.dest)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    let mut hasher = Sha256::new();

    if existing > 0 {
        // The running hash must cover the resumed bytes or the final digest is
        // meaningless. Costs one read of what we already have.
        tracing::info!(bytes = existing, path = %d.dest.display(), "resuming");
        hash_existing(&d.dest, &mut hasher).await?;
    }

    let mut req = client.get(&d.url);
    if existing > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={existing}-"));
    }
    let resp = req.send().await.map_err(|e| Error::Fetch {
        url: d.url.clone(),
        reason: e.to_string(),
    })?;

    // A server that ignores Range answers 200 with the WHOLE body. Appending
    // that to a partial file silently produces a corrupt result, so refuse.
    if existing > 0 && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(Error::NoResume { url: d.url.clone() });
    }
    if !resp.status().is_success() {
        return Err(Error::Fetch {
            url: d.url.clone(),
            reason: format!("HTTP {}", resp.status()),
        });
    }

    let total = resp.content_length().map(|c| c + existing);
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&d.dest)
        .await
        .map_err(|e| Error::io("open download", &d.dest, e))?;

    let mut done = existing;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| Error::Fetch {
            url: d.url.clone(),
            reason: e.to_string(),
        })?;
        hasher.update(&bytes);
        file.write_all(&bytes)
            .await
            .map_err(|e| Error::io("write download", &d.dest, e))?;
        done += bytes.len() as u64;
        on_progress(Progress {
            done,
            total,
            resumed_from: existing,
        });
    }
    file.flush()
        .await
        .map_err(|e| Error::io("flush download", &d.dest, e))?;

    let digest = hex::encode(hasher.finalize());
    if let Some(want) = &d.expected_sha256
        && !want.eq_ignore_ascii_case(&digest)
    {
        // Leave the bad file where it is: it is evidence, and deleting it would
        // make the next attempt look like a fresh download of a broken source.
        return Err(Error::Checksum {
            file: d.dest.display().to_string(),
            expected: want.clone(),
            actual: digest,
        });
    }

    Ok(Fetched {
        path: d.dest.clone(),
        digest,
        bytes: done,
        resumed_from: existing,
        verified: d.expected_sha256.is_some(),
    })
}

async fn hash_existing(path: &Path, hasher: &mut Sha256) -> Result<()> {
    use tokio::io::AsyncReadExt;

    let mut f = tokio::fs::File::open(path)
        .await
        .map_err(|e| Error::io("read partial", path, e))?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .await
            .map_err(|e| Error::io("read partial", path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(buf.get(..n).unwrap_or_default());
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn progress_fraction_is_absent_when_the_total_is_unknown() {
        assert_eq!(
            Progress {
                done: 5,
                total: None,
                resumed_from: 0
            }
            .fraction(),
            None
        );
        let p = Progress {
            done: 50,
            total: Some(100),
            resumed_from: 0,
        };
        assert!((p.fraction().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn progress_never_exceeds_one_even_if_the_server_undercounts() {
        let p = Progress {
            done: 150,
            total: Some(100),
            resumed_from: 0,
        };
        assert_eq!(p.fraction(), Some(1.0));
    }
}
