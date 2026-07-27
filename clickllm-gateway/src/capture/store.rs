//! The capture log — redacted, then encrypted, then written. In that order.
//!
//! ## Threat model, stated plainly
//!
//! Encryption here protects **data at rest**: a stolen laptop, a copied backup,
//! a directory synced somewhere it should not have been. That is a real and
//! common exposure for a file full of production prompts.
//!
//! It does **not** protect against an attacker who can read the running process
//! or the key file, because the key lives beside the data. clickllm is a
//! local-first tool with no key server, so there is nowhere else to put it. The
//! honest summary: this raises the cost of a cold disk from *trivial* to
//! *needs the key file*, and does nothing against a warm host. Anyone who needs
//! more should point `CLICKLLM_CAPTURE_KEY` at a key their own KMS issues.
//!
//! Claiming more than that would be worse than claiming nothing, because it
//! would let someone put this on a shared box and believe they were covered.
//!
//! ## Ordering
//!
//! Redaction runs **before** encryption, not instead of it. Encrypting raw
//! prompts would mean anyone who ever legitimately reads the log — an engineer
//! debugging a cluster, an exported eval set — sees the unredacted originals.
//! The plaintext inside the envelope is already redacted.
//!
//! A redaction failure drops the record. That is the whole contract, and it is
//! enforced here rather than left to callers.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use chacha20poly1305::aead::{Aead, KeyInit, OsRng, rand_core::RngCore};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use serde::{Deserialize, Serialize};

use super::redact::{self, Report};

/// One captured exchange, already redacted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capture {
    /// Correlates with the gateway's request record.
    pub request_id: String,
    /// Model the client asked for.
    pub model: String,
    /// Backend that served it.
    pub backend: String,
    /// Redacted request messages, as sent.
    pub messages: serde_json::Value,
    /// Redacted response text.
    pub response: String,
    /// Prompt tokens, when the upstream reported them.
    pub prompt_tokens: Option<u64>,
    /// Completion tokens, when the upstream reported them.
    pub completion_tokens: Option<u64>,
    /// Wall-clock duration.
    pub duration_ms: u64,
    /// What redaction removed. Non-empty is the normal case for real traffic.
    pub redacted: Report,
}

/// Why a capture was not stored.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StoreError {
    /// Redaction could not complete, so nothing was written.
    #[error("capture dropped: {0}")]
    RedactionRefused(#[from] redact::Refused),

    /// Encryption or decryption failed.
    #[error("{op} failed — the log may be truncated or the key wrong")]
    Crypto {
        /// Which direction failed.
        op: &'static str,
    },

    /// Filesystem failure, with the path that caused it.
    #[error("{op} failed for {path}")]
    Io {
        /// What was attempted.
        op: &'static str,
        /// Path involved.
        path: PathBuf,
        /// Underlying cause.
        #[source]
        source: std::io::Error,
    },
}

impl StoreError {
    fn io(op: &'static str, path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            op,
            path: path.into(),
            source,
        }
    }
}

type Result<T> = std::result::Result<T, StoreError>;

/// Length-prefixed record framing: 4-byte length, 24-byte nonce, then ciphertext.
const LEN_BYTES: usize = 4;
const NONCE_BYTES: usize = 24;

/// An append-only, encrypted capture log.
pub struct CaptureStore {
    path: PathBuf,
    cipher: XChaCha20Poly1305,
    /// Held across the whole of one record's write.
    ///
    /// `O_APPEND` makes a single `write` atomic, but a record is a length, a
    /// nonce and a ciphertext — three writes, which concurrent requests
    /// interleave into a log that decrypts as garbage. The frame is assembled in
    /// memory and written once, under this lock.
    ///
    /// The lock is process-wide, which is the right scope: one gateway owns its
    /// log. Two processes pointed at the same file would still corrupt it, and
    /// that would need `flock`.
    writer: parking_lot::Mutex<Option<std::fs::File>>,
}

impl CaptureStore {
    /// Open (creating if needed) a log at `path`, using `key`.
    ///
    /// # Errors
    /// [`StoreError::Io`] if the parent directory cannot be created.
    pub fn open(path: impl Into<PathBuf>, key: &[u8; 32]) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| StoreError::io("create capture dir", parent, e))?;
        }
        Ok(Self {
            path,
            cipher: XChaCha20Poly1305::new(Key::from_slice(key)),
            writer: parking_lot::Mutex::new(None),
        })
    }

    /// Load a key from `path`, generating one if absent.
    ///
    /// The file is created with owner-only permissions. That is the entire
    /// access control — see the module's threat model.
    ///
    /// # Errors
    /// [`StoreError::Io`] on any filesystem failure.
    pub fn load_or_create_key(path: &Path) -> Result<[u8; 32]> {
        if let Ok(existing) = std::fs::read(path)
            && existing.len() == 32
        {
            let mut k = [0u8; 32];
            k.copy_from_slice(&existing);
            return Ok(k);
        }

        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| StoreError::io("create key dir", parent, e))?;
        }
        std::fs::write(path, key).map_err(|e| StoreError::io("write key", path, e))?;
        restrict(path)?;
        tracing::info!(path = %path.display(), "generated a capture key");
        Ok(key)
    }

    /// Redact, encrypt, and append.
    ///
    /// Redaction happens here rather than in the caller so it cannot be skipped:
    /// there is no path into this log that bypasses it.
    ///
    /// # Errors
    /// [`StoreError::RedactionRefused`] drops the record without writing;
    /// [`StoreError::Crypto`] or [`StoreError::Io`] on failure to persist.
    pub fn append(&self, mut c: Capture) -> Result<Report> {
        let (messages, mut report) = redact_json(&c.messages)?;
        let (response, response_report) = redact::redact(&c.response)?;
        for (k, v) in response_report.counts {
            *report.counts.entry(k).or_insert(0) += v;
        }
        c.messages = messages;
        c.response = response;
        c.redacted = report.clone();

        let plaintext = serde_json::to_vec(&c).map_err(|_| StoreError::Crypto { op: "serialise" })?;

        let mut nonce_bytes = [0u8; NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = XNonce::from_slice(&nonce_bytes);
        let ct = self
            .cipher
            .encrypt(nonce, plaintext.as_ref())
            .map_err(|_| StoreError::Crypto { op: "encrypt" })?;

        // One buffer, one write. See the note on `writer`.
        let len = u32::try_from(ct.len()).map_err(|_| StoreError::Crypto { op: "frame" })?;
        let mut frame = Vec::with_capacity(LEN_BYTES + NONCE_BYTES + ct.len());
        frame.extend_from_slice(&len.to_le_bytes());
        frame.extend_from_slice(&nonce_bytes);
        frame.extend_from_slice(&ct);

        let mut guard = self.writer.lock();
        if guard.is_none() {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .map_err(|e| StoreError::io("open capture log", &self.path, e))?;
            restrict(&self.path)?;
            *guard = Some(f);
        }
        let Some(f) = guard.as_mut() else {
            return Err(StoreError::Crypto { op: "open" });
        };
        f.write_all(&frame)
            .and_then(|()| f.flush())
            .map_err(|e| StoreError::io("append capture", &self.path, e))?;

        Ok(report)
    }

    /// Read every record back.
    ///
    /// A truncated final record — a crash mid-append — is skipped rather than
    /// failing the whole read. Losing the last capture is acceptable; losing the
    /// corpus because of it is not.
    ///
    /// # Errors
    /// [`StoreError::Io`] if the log cannot be read, [`StoreError::Crypto`] if a
    /// complete record fails to decrypt (wrong key, or tampering).
    pub fn read_all(&self) -> Result<Vec<Capture>> {
        let raw = match std::fs::read(&self.path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(StoreError::io("read capture log", &self.path, e)),
        };

        let mut out = Vec::new();
        let mut i = 0usize;
        loop {
            let header_end = i.saturating_add(LEN_BYTES);
            let start = header_end.saturating_add(NONCE_BYTES);
            let (Some(len_bytes), Some(nonce_bytes)) =
                (raw.get(i..header_end), raw.get(header_end..start))
            else {
                break;
            };
            let mut le = [0u8; LEN_BYTES];
            le.copy_from_slice(len_bytes);
            let len = u32::from_le_bytes(le) as usize;
            let end = start.saturating_add(len);
            let Some(ct) = raw.get(start..end) else {
                tracing::warn!("capture log ends mid-record; skipping the partial tail");
                break;
            };
            let pt = self
                .cipher
                .decrypt(XNonce::from_slice(nonce_bytes), ct)
                .map_err(|_| StoreError::Crypto { op: "decrypt" })?;
            if let Ok(c) = serde_json::from_slice::<Capture>(&pt) {
                out.push(c);
            }
            i = end;
        }
        Ok(out)
    }

    /// Number of records currently stored.
    ///
    /// # Errors
    /// As [`CaptureStore::read_all`].
    pub fn len(&self) -> Result<usize> {
        Ok(self.read_all()?.len())
    }

    /// Whether the log holds nothing.
    ///
    /// # Errors
    /// As [`CaptureStore::read_all`].
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// Assistant text carried by one streamed event.
///
/// Returns `None` for anything that is not generated content — the `[DONE]`
/// sentinel, role-only preamble frames, tool-call deltas. Concatenating those
/// would put protocol noise into the eval set.
pub fn delta_text(event: &crate::sse::Event) -> Option<String> {
    let crate::sse::Event::Data(payload) = event else {
        return None;
    };
    let v: serde_json::Value = serde_json::from_str(payload).ok()?;
    let s = v
        .get("choices")?
        .get(0)?
        .get("delta")?
        .get("content")?
        .as_str()?;
    (!s.is_empty()).then(|| s.to_owned())
}

/// Assistant text in a complete, non-streamed response body.
///
/// An unparseable or error body yields an empty string rather than a guess: a
/// fabricated response would poison the eval set far more quietly than a blank.
pub fn body_text(body: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("choices")?
                .get(0)?
                .get("message")?
                .get("content")?
                .as_str()
                .map(ToOwned::to_owned)
        })
        .unwrap_or_default()
}

/// Redact every string in a JSON value, in place, preserving structure.
fn redact_json(v: &serde_json::Value) -> Result<(serde_json::Value, Report)> {
    let mut report = Report::default();
    let out = walk(v, &mut report)?;
    Ok((out, report))
}

fn walk(v: &serde_json::Value, report: &mut Report) -> Result<serde_json::Value> {
    use serde_json::Value;
    Ok(match v {
        Value::String(s) => {
            let (clean, r) = redact::redact(s)?;
            for (k, n) in r.counts {
                *report.counts.entry(k).or_insert(0) += n;
            }
            Value::String(clean)
        }
        Value::Array(a) => Value::Array(
            a.iter()
                .map(|x| walk(x, report))
                .collect::<Result<Vec<_>>>()?,
        ),
        Value::Object(o) => {
            let mut m = serde_json::Map::new();
            for (k, val) in o {
                m.insert(k.clone(), walk(val, report)?);
            }
            Value::Object(m)
        }
        other => other.clone(),
    })
}

/// Owner-only permissions. Best effort on platforms without POSIX modes.
fn restrict(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(path, perms)
            .map_err(|e| StoreError::io("restrict permissions", path, e))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use serde_json::json;

    fn key() -> [u8; 32] {
        [7u8; 32]
    }

    fn capture(msg: &str, resp: &str) -> Capture {
        Capture {
            request_id: "r1".into(),
            model: "gpt-5".into(),
            backend: "incumbent".into(),
            messages: json!([{"role": "user", "content": msg}]),
            response: resp.into(),
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            duration_ms: 42,
            redacted: Report::default(),
        }
    }

    #[test]
    fn a_capture_round_trips() {
        let d = tempfile::tempdir().unwrap();
        let s = CaptureStore::open(d.path().join("log"), &key()).unwrap();
        s.append(capture("hello", "world")).unwrap();
        let back = s.read_all().unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].response, "world");
        assert_eq!(back[0].duration_ms, 42);
    }

    #[test]
    fn nothing_reaches_disk_unredacted() {
        // The contract. Read the raw file and assert the secret is not in it.
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("log");
        let s = CaptureStore::open(&path, &key()).unwrap();
        s.append(capture(
            "email ada@example.com about card 4539578763621486",
            "reply to ada@example.com",
        ))
        .unwrap();

        let raw = std::fs::read(&path).unwrap();
        let as_text = String::from_utf8_lossy(&raw);
        assert!(!as_text.contains("ada@example.com"));
        assert!(!as_text.contains("4539578763621486"));

        let back = s.read_all().unwrap();
        assert!(back[0].response.contains("<EMAIL>"));
        assert!(!back[0].redacted.is_empty(), "removals must be recorded");
    }

    #[test]
    fn redaction_reaches_nested_json_not_just_the_top_level() {
        let d = tempfile::tempdir().unwrap();
        let s = CaptureStore::open(d.path().join("log"), &key()).unwrap();
        let mut c = capture("hi", "ok");
        c.messages = json!({
            "messages": [{"role": "user", "content": {"deep": ["mail ada@example.com"]}}]
        });
        s.append(c).unwrap();
        let back = s.read_all().unwrap();
        let text = back[0].messages.to_string();
        assert!(!text.contains("ada@example.com"), "{text}");
        assert!(text.contains("<EMAIL>"));
    }

    #[test]
    fn json_structure_survives_redaction() {
        // Clustering keys on shape; flattening or dropping keys would corrupt it.
        let d = tempfile::tempdir().unwrap();
        let s = CaptureStore::open(d.path().join("log"), &key()).unwrap();
        let mut c = capture("hi", "ok");
        c.messages = json!([{"role": "system", "content": "be brief"},
                            {"role": "user", "content": "mail a@b.com"}]);
        s.append(c).unwrap();
        let m = &s.read_all().unwrap()[0].messages;
        assert_eq!(m.as_array().unwrap().len(), 2);
        assert_eq!(m[0]["role"], "system");
        assert_eq!(m[1]["content"], "mail <EMAIL>");
    }

    #[test]
    fn an_unredactable_capture_is_dropped_not_stored() {
        // The fail-closed contract, enforced here rather than left to callers.
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("log");
        let s = CaptureStore::open(&path, &key()).unwrap();
        let huge = "a".repeat(redact::MAX_SCAN_BYTES + 1);
        let err = s.append(capture("hi", &huge)).unwrap_err();
        assert!(matches!(err, StoreError::RedactionRefused(_)));
        assert!(
            s.read_all().unwrap().is_empty(),
            "a refused capture must leave nothing behind"
        );
    }

    #[test]
    fn the_log_is_unreadable_with_the_wrong_key() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("log");
        CaptureStore::open(&path, &key())
            .unwrap()
            .append(capture("secret plan", "ok"))
            .unwrap();

        let wrong = CaptureStore::open(&path, &[9u8; 32]).unwrap();
        assert!(matches!(
            wrong.read_all(),
            Err(StoreError::Crypto { op: "decrypt" })
        ));
    }

    #[test]
    fn tampering_is_detected_rather_than_silently_accepted() {
        // AEAD: flipping a ciphertext byte must fail the tag, not decode garbage.
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("log");
        let s = CaptureStore::open(&path, &key()).unwrap();
        s.append(capture("hello", "world")).unwrap();

        let mut raw = std::fs::read(&path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xff;
        std::fs::write(&path, raw).unwrap();

        assert!(matches!(s.read_all(), Err(StoreError::Crypto { .. })));
    }

    #[test]
    fn a_crash_mid_append_costs_one_record_not_the_corpus() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("log");
        let s = CaptureStore::open(&path, &key()).unwrap();
        s.append(capture("one", "1")).unwrap();
        s.append(capture("two", "2")).unwrap();

        // Simulate a partial write of a third record.
        let mut raw = std::fs::read(&path).unwrap();
        raw.extend_from_slice(&999u32.to_le_bytes());
        raw.extend_from_slice(&[0u8; NONCE_BYTES]);
        raw.extend_from_slice(b"truncated");
        std::fs::write(&path, raw).unwrap();

        let back = s.read_all().unwrap();
        assert_eq!(back.len(), 2, "the two complete records must survive");
    }

    #[test]
    fn many_appends_all_read_back_in_order() {
        let d = tempfile::tempdir().unwrap();
        let s = CaptureStore::open(d.path().join("log"), &key()).unwrap();
        for i in 0..50 {
            s.append(capture(&format!("q{i}"), &format!("a{i}"))).unwrap();
        }
        let back = s.read_all().unwrap();
        assert_eq!(back.len(), 50);
        assert_eq!(back[0].response, "a0");
        assert_eq!(back[49].response, "a49");
    }

    #[test]
    fn every_record_gets_its_own_nonce() {
        // Reusing a nonce with the same key breaks the cipher outright.
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("log");
        let s = CaptureStore::open(&path, &key()).unwrap();
        s.append(capture("same", "same")).unwrap();
        s.append(capture("same", "same")).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let n1 = &raw[LEN_BYTES..LEN_BYTES + NONCE_BYTES];
        let mut le = [0u8; LEN_BYTES];
        le.copy_from_slice(&raw[..LEN_BYTES]);
        let second = LEN_BYTES + NONCE_BYTES + u32::from_le_bytes(le) as usize;
        let n2 = &raw[second + LEN_BYTES..second + LEN_BYTES + NONCE_BYTES];
        assert_ne!(n1, n2, "identical plaintexts must not share a nonce");
    }

    #[test]
    fn a_key_is_generated_once_and_reused() {
        let d = tempfile::tempdir().unwrap();
        let kp = d.path().join("keys").join("capture.key");
        let a = CaptureStore::load_or_create_key(&kp).unwrap();
        let b = CaptureStore::load_or_create_key(&kp).unwrap();
        assert_eq!(a, b, "a regenerated key would orphan the existing log");
        assert_ne!(a, [0u8; 32]);
    }

    #[cfg(unix)]
    #[test]
    fn the_key_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempfile::tempdir().unwrap();
        let kp = d.path().join("capture.key");
        CaptureStore::load_or_create_key(&kp).unwrap();
        let mode = std::fs::metadata(&kp).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "the key is the entire access control");
    }

    #[test]
    fn an_absent_log_reads_as_empty_rather_than_failing() {
        let d = tempfile::tempdir().unwrap();
        let s = CaptureStore::open(d.path().join("never-written"), &key()).unwrap();
        assert!(s.is_empty().unwrap());
    }
}
