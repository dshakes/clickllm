//! The local weight store.
//!
//! Weights are the largest thing this tool touches — tens to hundreds of
//! gigabytes — so the store has two jobs beyond holding bytes:
//!
//! **Never download the same thing twice.** Entries are keyed by the canonical
//! [`ModelRef`] cache key, so `Q4_K_M` and `q4-k-m` land on the same entry
//! rather than costing 60 GB twice.
//!
//! **Never fill the disk.** A budget with LRU eviction, and — the part that is
//! easy to get wrong — **an entry currently in use is never evicted**. Deleting
//! weights out from under a running engine would look like model corruption and
//! be diagnosed as anything but a cache bug.
//!
//! Partial downloads live beside their target as `.part` files so a resumed
//! fetch can find them, and are never mistaken for complete entries: an entry is
//! only real once its digest has been verified and it has been renamed into
//! place. That rename is the commit point.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use clickllm_core::ModelRef;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Metadata written beside every completed entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// Canonical reference these bytes satisfy.
    pub reference: String,
    /// Cache key derived from that reference.
    pub key: String,
    /// Total bytes on disk.
    pub bytes: u64,
    /// SHA-256 of the payload, lowercase hex.
    pub digest: String,
    /// Unix seconds when this entry was last opened. Drives LRU.
    pub last_used: u64,
    /// Files that make up the entry, relative to its directory.
    pub files: Vec<String>,
}

impl Entry {
    fn touch(&mut self) {
        self.last_used = now();
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A content-addressed store for model weights.
#[derive(Debug)]
pub struct Cache {
    root: PathBuf,
    /// Bytes the store may occupy before eviction begins. `None` = unbounded,
    /// which is a legitimate choice on a dedicated inference box.
    budget_bytes: Option<u64>,
    /// Keys currently loaded by a running engine. Never evicted.
    pinned: BTreeMap<String, u32>,
}

/// Written into each entry directory. Named so it cannot collide with a
/// weight file.
const META: &str = ".clickllm-entry.json";

impl Cache {
    /// Open (creating if needed) a store rooted at `root`.
    ///
    /// # Errors
    /// [`Error::Io`] if the directory cannot be created.
    pub fn open(root: impl Into<PathBuf>, budget_bytes: Option<u64>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|e| Error::io("create cache dir", &root, e))?;
        Ok(Self {
            root,
            budget_bytes,
            pinned: BTreeMap::new(),
        })
    }

    /// Where an entry's files live.
    pub fn entry_dir(&self, r: &ModelRef) -> PathBuf {
        self.root.join(r.cache_key())
    }

    /// Path a partial download writes to. Beside the target, not inside it, so
    /// an interrupted fetch can never be mistaken for a complete entry.
    pub fn part_path(&self, r: &ModelRef, filename: &str) -> PathBuf {
        self.root
            .join(format!("{}.part", r.cache_key()))
            .join(filename)
    }

    /// The completed entry for `r`, if there is one.
    pub fn get(&self, r: &ModelRef) -> Option<Entry> {
        let meta = self.entry_dir(r).join(META);
        let raw = std::fs::read_to_string(meta).ok()?;
        serde_json::from_str(&raw).ok()
    }

    /// True when `r` is present and complete.
    pub fn contains(&self, r: &ModelRef) -> bool {
        self.get(r).is_some()
    }

    /// Mark an entry as used now, so LRU reflects reality.
    ///
    /// # Errors
    /// [`Error::Io`] if the metadata cannot be rewritten.
    pub fn touch(&self, r: &ModelRef) -> Result<()> {
        let Some(mut e) = self.get(r) else {
            return Ok(());
        };
        e.touch();
        self.write_meta(r, &e)
    }

    fn write_meta(&self, r: &ModelRef, e: &Entry) -> Result<()> {
        let dir = self.entry_dir(r);
        std::fs::create_dir_all(&dir).map_err(|x| Error::io("create entry dir", &dir, x))?;
        let path = dir.join(META);
        let json = serde_json::to_string_pretty(e).unwrap_or_default();
        std::fs::write(&path, json).map_err(|x| Error::io("write entry metadata", &path, x))
    }

    /// Commit a verified download into the store.
    ///
    /// This is the only way an entry becomes real. Callers must have verified
    /// the digest first — [`crate::fetch`] does, and refuses to call this
    /// otherwise.
    ///
    /// # Errors
    /// [`Error::Io`] on any filesystem failure.
    pub fn commit(&mut self, r: &ModelRef, digest: &str, files: Vec<String>) -> Result<Entry> {
        let staging = self.root.join(format!("{}.part", r.cache_key()));
        let dir = self.entry_dir(r);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| Error::io("replace entry", &dir, e))?;
        }
        // Rename is the commit point: an entry appears complete or not at all.
        std::fs::rename(&staging, &dir).map_err(|e| Error::io("commit entry", &staging, e))?;

        let bytes = dir_size(&dir);
        let entry = Entry {
            reference: r.to_string(),
            key: r.cache_key(),
            bytes,
            digest: digest.to_owned(),
            last_used: now(),
            files,
        };
        self.write_meta(r, &entry)?;
        tracing::info!(key = %entry.key, bytes, "cached");
        Ok(entry)
    }

    /// Every completed entry, oldest-used first.
    pub fn entries(&self) -> Vec<Entry> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&self.root) else {
            return out;
        };
        for item in rd.flatten() {
            let meta = item.path().join(META);
            if let Ok(raw) = std::fs::read_to_string(meta)
                && let Ok(e) = serde_json::from_str::<Entry>(&raw)
            {
                out.push(e);
            }
        }
        out.sort_by_key(|e| (e.last_used, e.key.clone()));
        out
    }

    /// Total bytes held by completed entries.
    pub fn used_bytes(&self) -> u64 {
        self.entries().iter().map(|e| e.bytes).sum()
    }

    /// Prevent `key` from being evicted while an engine has it loaded.
    pub fn pin(&mut self, key: &str) {
        *self.pinned.entry(key.to_owned()).or_insert(0) += 1;
    }

    /// Release a pin. Balanced with [`Cache::pin`]; extra releases are ignored.
    pub fn unpin(&mut self, key: &str) {
        if let Some(n) = self.pinned.get_mut(key) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.pinned.remove(key);
            }
        }
    }

    /// Whether `key` is currently protected from eviction.
    pub fn is_pinned(&self, key: &str) -> bool {
        self.pinned.contains_key(key)
    }

    /// Evict least-recently-used entries until the store fits its budget.
    ///
    /// Pinned entries are skipped. If the budget cannot be met without evicting
    /// something pinned, the store stays over budget and says so — deleting
    /// weights from under a running engine would look like model corruption and
    /// be debugged as anything but a cache bug.
    ///
    /// Returns the keys evicted.
    ///
    /// # Errors
    /// [`Error::Io`] if an eviction fails partway.
    pub fn enforce_budget(&mut self) -> Result<Vec<String>> {
        let Some(budget) = self.budget_bytes else {
            return Ok(Vec::new());
        };
        let mut used = self.used_bytes();
        let mut evicted = Vec::new();
        if used <= budget {
            return Ok(evicted);
        }

        for e in self.entries() {
            if used <= budget {
                break;
            }
            if self.is_pinned(&e.key) {
                tracing::debug!(key = %e.key, "over budget but pinned; not evicting");
                continue;
            }
            let dir = self.root.join(&e.key);
            std::fs::remove_dir_all(&dir).map_err(|x| Error::io("evict entry", &dir, x))?;
            used = used.saturating_sub(e.bytes);
            tracing::info!(key = %e.key, bytes = e.bytes, "evicted");
            evicted.push(e.key);
        }

        if used > budget {
            tracing::warn!(
                used,
                budget,
                "still over budget: everything remaining is in use"
            );
        }
        Ok(evicted)
    }

    /// Remove a completed entry.
    ///
    /// # Errors
    /// [`Error::Pinned`] if it is in use, [`Error::Io`] on a filesystem failure.
    pub fn remove(&mut self, r: &ModelRef) -> Result<bool> {
        let key = r.cache_key();
        if self.is_pinned(&key) {
            return Err(Error::Pinned { key });
        }
        let dir = self.entry_dir(r);
        if !dir.exists() {
            return Ok(false);
        }
        std::fs::remove_dir_all(&dir).map_err(|e| Error::io("remove entry", &dir, e))?;
        Ok(true)
    }
}

fn dir_size(dir: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    rd.flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => dir_size(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn r(s: &str) -> ModelRef {
        s.parse().unwrap()
    }

    /// Stage bytes as a fetch would, then commit them.
    fn stage(c: &mut Cache, m: &ModelRef, bytes: usize) -> Entry {
        let part = c.root.join(format!("{}.part", m.cache_key()));
        std::fs::create_dir_all(&part).unwrap();
        std::fs::write(part.join("model.bin"), vec![0u8; bytes]).unwrap();
        c.commit(m, "deadbeef", vec!["model.bin".into()]).unwrap()
    }

    #[test]
    fn quant_spelling_variants_share_one_entry() {
        // The whole point of canonicalisation: not paying for 60 GB twice.
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), None).unwrap();
        stage(&mut c, &r("hf:org/m#Q4_K_M"), 1024);
        assert!(c.contains(&r("hf:org/m#q4-k-m")));
        assert!(c.contains(&r("hf:org/m#q4_k_m")));
        assert_eq!(c.entries().len(), 1);
    }

    #[test]
    fn different_quants_are_different_entries() {
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), None).unwrap();
        stage(&mut c, &r("hf:org/m#q4"), 1024);
        assert!(!c.contains(&r("hf:org/m#q8")));
    }

    #[test]
    fn a_partial_download_is_never_mistaken_for_a_complete_entry() {
        let d = tempfile::tempdir().unwrap();
        let c = Cache::open(d.path(), None).unwrap();
        let m = r("hf:org/m#q4");
        let part = d.path().join(format!("{}.part", m.cache_key()));
        std::fs::create_dir_all(&part).unwrap();
        std::fs::write(part.join("model.bin"), b"half").unwrap();

        assert!(
            !c.contains(&m),
            "an interrupted fetch must not look complete"
        );
        assert!(c.entries().is_empty());
    }

    #[test]
    fn commit_is_atomic_from_the_readers_point_of_view() {
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), None).unwrap();
        let m = r("hf:org/m#q4");
        let e = stage(&mut c, &m, 4096);
        assert_eq!(e.bytes, 4096);
        assert!(c.entry_dir(&m).join("model.bin").exists());
        assert!(!d.path().join(format!("{}.part", m.cache_key())).exists());
    }

    #[test]
    fn lru_evicts_the_oldest_first() {
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), Some(2500)).unwrap();
        let old = r("hf:org/old#q4");
        let new = r("hf:org/new#q4");
        stage(&mut c, &old, 1500);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        stage(&mut c, &new, 1500);

        let evicted = c.enforce_budget().unwrap();
        assert_eq!(evicted.len(), 1);
        assert!(!c.contains(&old), "oldest should have gone");
        assert!(c.contains(&new));
    }

    #[test]
    fn an_entry_in_use_is_never_evicted() {
        // Deleting weights from under a running engine looks like model
        // corruption and gets debugged as anything but a cache bug.
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), Some(1000)).unwrap();
        let loaded = r("hf:org/loaded#q4");
        stage(&mut c, &loaded, 5000);
        c.pin(&loaded.cache_key());

        let evicted = c.enforce_budget().unwrap();
        assert!(evicted.is_empty(), "a pinned entry must survive its budget");
        assert!(c.contains(&loaded));
        assert!(
            c.used_bytes() > 1000,
            "the store stays over budget rather than corrupting a run"
        );
    }

    #[test]
    fn unpinning_lets_it_be_evicted_again() {
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), Some(1000)).unwrap();
        let m = r("hf:org/m#q4");
        stage(&mut c, &m, 5000);
        c.pin(&m.cache_key());
        assert!(c.enforce_budget().unwrap().is_empty());
        c.unpin(&m.cache_key());
        assert_eq!(c.enforce_budget().unwrap().len(), 1);
    }

    #[test]
    fn pins_are_counted_so_two_engines_can_share_an_entry() {
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), None).unwrap();
        c.pin("k");
        c.pin("k");
        c.unpin("k");
        assert!(c.is_pinned("k"), "one engine still has it loaded");
        c.unpin("k");
        assert!(!c.is_pinned("k"));
        c.unpin("k"); // extra release is harmless
        assert!(!c.is_pinned("k"));
    }

    #[test]
    fn no_budget_means_no_eviction() {
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), None).unwrap();
        stage(&mut c, &r("hf:org/a#q4"), 10_000);
        stage(&mut c, &r("hf:org/b#q4"), 10_000);
        assert!(c.enforce_budget().unwrap().is_empty());
        assert_eq!(c.entries().len(), 2);
    }

    #[test]
    fn removing_a_pinned_entry_is_refused_rather_than_done() {
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), None).unwrap();
        let m = r("hf:org/m#q4");
        stage(&mut c, &m, 100);
        c.pin(&m.cache_key());
        assert!(matches!(c.remove(&m), Err(Error::Pinned { .. })));
        c.unpin(&m.cache_key());
        assert!(c.remove(&m).unwrap());
        assert!(!c.contains(&m));
    }

    #[test]
    fn touch_updates_recency_so_lru_reflects_actual_use() {
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), Some(2500)).unwrap();
        let a = r("hf:org/a#q4");
        let b = r("hf:org/b#q4");
        stage(&mut c, &a, 1500);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        stage(&mut c, &b, 1500);
        // `a` is older, but using it makes it the newer one.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        c.touch(&a).unwrap();

        c.enforce_budget().unwrap();
        assert!(c.contains(&a), "recently used entry should survive");
        assert!(!c.contains(&b));
    }

    #[test]
    fn used_bytes_reflects_what_is_actually_on_disk() {
        let d = tempfile::tempdir().unwrap();
        let mut c = Cache::open(d.path(), None).unwrap();
        stage(&mut c, &r("hf:org/a#q4"), 2048);
        // Metadata adds a little; the payload must be accounted for at minimum.
        assert!(c.used_bytes() >= 2048);
    }
}
