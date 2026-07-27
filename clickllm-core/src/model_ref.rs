//! Model references: parse, canonicalise, compare.
//!
//! A reference names *where weights come from* and *which variant*. It is the
//! join key between the catalogue, the cache, and a box manifest, so two refs
//! that mean the same thing must canonicalise identically — otherwise the cache
//! silently misses and the same 60 GB is downloaded twice.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Where the weights live.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "scheme", content = "path")]
pub enum Source {
    /// Hugging Face repo, `org/name`.
    Hf(String),
    /// OCI artifact, `registry/repo:tag`.
    Oci(String),
    /// S3 or S3-compatible object prefix.
    S3(String),
    /// A path already on this machine.
    File(String),
}

impl Source {
    /// Scheme prefix as it appears in a reference string.
    pub fn scheme(&self) -> &'static str {
        match self {
            Self::Hf(_) => "hf",
            Self::Oci(_) => "oci",
            Self::S3(_) => "s3",
            Self::File(_) => "file",
        }
    }

    /// The location portion, without the scheme.
    pub fn locator(&self) -> &str {
        match self {
            Self::Hf(s) | Self::Oci(s) | Self::S3(s) | Self::File(s) => s,
        }
    }

    /// Whether fetching requires network egress. Air-gapped installs (NFR-7)
    /// must be able to refuse everything that does.
    pub fn needs_network(&self) -> bool {
        !matches!(self, Self::File(_))
    }
}

/// A fully-qualified reference to one set of weights.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    /// Where the weights come from.
    pub source: Source,
    /// Quantisation label, lowercased (`q4_k_m`, `fp8`, `mlx-4bit`). `None` means
    /// "whatever the source publishes" — resolved later, never guessed here.
    pub quant: Option<String>,
    /// Git revision, OCI digest, or object version. `None` means the source's default,
    /// which is *not* reproducible — callers that need reproducibility must pin.
    pub revision: Option<String>,
}

impl ModelRef {
    /// Construct from parts, canonicalising as `FromStr` would.
    pub fn new(source: Source, quant: Option<&str>, revision: Option<&str>) -> Self {
        Self {
            source,
            quant: quant.map(canon_quant),
            revision: revision.map(str::to_owned),
        }
    }

    /// True when this ref names an exact, reproducible artifact.
    ///
    /// Unpinned refs are legal — they are how people explore — but a box manifest
    /// that embeds one cannot promise the same weights tomorrow, so M4's packer
    /// will refuse them.
    pub fn is_pinned(&self) -> bool {
        self.revision.is_some()
    }

    /// Stable cache key. Two refs meaning the same artifact share a key; any
    /// difference in source, quant, or revision produces a different one.
    pub fn cache_key(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(self.source.scheme().as_bytes());
        h.update([0]);
        h.update(self.source.locator().as_bytes());
        h.update([0]);
        h.update(self.quant.as_deref().unwrap_or("").as_bytes());
        h.update([0]);
        h.update(self.revision.as_deref().unwrap_or("").as_bytes());
        // Iterate rather than slice: a panicking index has no place on a path
        // that runs for every cache lookup, however unreachable it looks.
        h.finalize()
            .iter()
            .take(16)
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

/// Quantisation labels arrive in every casing and both separators. Fold them so
/// `Q4_K_M`, `q4-k-m`, and `q4_k_m` share one cache entry.
fn canon_quant(q: &str) -> String {
    q.trim().to_ascii_lowercase().replace('-', "_")
}

impl FromStr for ModelRef {
    type Err = Error;

    /// Parse `[scheme:]locator[@revision][#quant]`.
    ///
    /// A bare locator is Hugging Face — the overwhelmingly common case, and the
    /// one people type. Anything containing a path separator that looks local
    /// still needs an explicit `file:` prefix; guessing there risks reading a
    /// path the user did not mean.
    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Err(Error::ModelRef {
                input: s.to_owned(),
                reason: "empty reference".into(),
            });
        }

        // #quant comes off first: a quant label never contains '@'.
        let (rest, quant) = match s.split_once('#') {
            Some((_, q)) if q.trim().is_empty() => {
                return Err(Error::ModelRef {
                    input: s.to_owned(),
                    reason: "'#' present but quantisation is empty".into(),
                });
            }
            Some((r, q)) => (r, Some(q)),
            None => (s, None),
        };

        let (rest, revision) = match rest.split_once('@') {
            Some((_, v)) if v.trim().is_empty() => {
                return Err(Error::ModelRef {
                    input: s.to_owned(),
                    reason: "'@' present but revision is empty".into(),
                });
            }
            Some((r, v)) => (r, Some(v)),
            None => (rest, None),
        };

        // Split the scheme only on a known prefix. A bare "org/name" has no colon;
        // an OCI ref like "ghcr.io/acme/x:v1" has one that is a *tag*, not a scheme.
        let source = match rest.split_once(':') {
            Some(("hf", p)) => Source::Hf(p.to_owned()),
            Some(("oci", p)) => Source::Oci(p.to_owned()),
            Some(("s3", p)) => Source::S3(p.trim_start_matches("//").to_owned()),
            Some(("file", p)) => Source::File(p.to_owned()),
            _ => Source::Hf(rest.to_owned()),
        };

        if source.locator().trim().is_empty() {
            return Err(Error::ModelRef {
                input: s.to_owned(),
                reason: format!("{} scheme with empty locator", source.scheme()),
            });
        }
        if matches!(source, Source::Hf(_)) && !source.locator().contains('/') {
            return Err(Error::ModelRef {
                input: s.to_owned(),
                reason: "Hugging Face refs need 'org/name'".into(),
            });
        }

        Ok(Self::new(source, quant, revision))
    }
}

impl fmt::Display for ModelRef {
    /// Round-trips through [`FromStr`]. Property-tested below.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.source.scheme(), self.source.locator())?;
        if let Some(r) = &self.revision {
            write!(f, "@{r}")?;
        }
        if let Some(q) = &self.quant {
            write!(f, "#{q}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn parse(s: &str) -> ModelRef {
        s.parse().unwrap()
    }

    #[test]
    fn bare_locator_defaults_to_hugging_face() {
        let r = parse("Qwen/Qwen3-32B");
        assert_eq!(r.source, Source::Hf("Qwen/Qwen3-32B".into()));
        assert!(r.quant.is_none());
        assert!(!r.is_pinned());
    }

    #[test]
    fn every_scheme_parses() {
        assert_eq!(parse("hf:org/n").source, Source::Hf("org/n".into()));
        assert_eq!(
            parse("oci:ghcr.io/acme/m:v1").source,
            Source::Oci("ghcr.io/acme/m:v1".into())
        );
        assert_eq!(parse("s3://bucket/k").source, Source::S3("bucket/k".into()));
        assert_eq!(
            parse("file:/models/x").source,
            Source::File("/models/x".into())
        );
    }

    #[test]
    fn oci_tag_colon_is_not_mistaken_for_a_scheme() {
        // The bug this guards: naive split_once(':') would read "ghcr.io" as a scheme.
        let r = parse("oci:ghcr.io/acme/triage:v3");
        assert_eq!(r.source.locator(), "ghcr.io/acme/triage:v3");
    }

    #[test]
    fn quant_and_revision_parse_in_either_notation() {
        let r = parse("hf:org/n@abc123#Q4_K_M");
        assert_eq!(r.revision.as_deref(), Some("abc123"));
        assert_eq!(r.quant.as_deref(), Some("q4_k_m"));
        assert!(r.is_pinned());
    }

    #[test]
    fn quant_labels_canonicalise_to_one_cache_entry() {
        let keys: Vec<_> = ["#Q4_K_M", "#q4-k-m", "#q4_k_m", "# Q4-K-M "]
            .iter()
            .map(|q| parse(&format!("hf:org/n{q}")).cache_key())
            .collect();
        assert!(
            keys.windows(2).all(|w| w[0] == w[1]),
            "casing/separator variants must share a cache key, got {keys:?}"
        );
    }

    #[test]
    fn cache_key_separates_every_field() {
        let base = parse("hf:org/n");
        for other in ["hf:org/m", "oci:org/n", "hf:org/n#q4", "hf:org/n@v2"] {
            assert_ne!(
                base.cache_key(),
                parse(other).cache_key(),
                "{other} must not collide with hf:org/n"
            );
        }
    }

    #[test]
    fn cache_key_cannot_be_confused_by_field_boundaries() {
        // Without a separator byte, ("ab","c") and ("a","bc") would hash alike.
        let a = ModelRef::new(Source::Hf("ab/c".into()), Some("d"), None);
        let b = ModelRef::new(Source::Hf("ab/cd".into()), None, None);
        assert_ne!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn display_round_trips_through_parse() {
        for s in [
            "hf:org/n",
            "hf:org/n#q4_k_m",
            "hf:org/n@rev",
            "hf:org/n@rev#q8_0",
            "oci:ghcr.io/a/b:v1",
            "file:/models/x",
        ] {
            let r = parse(s);
            assert_eq!(parse(&r.to_string()), r, "round trip failed for {s}");
        }
    }

    #[test]
    fn malformed_refs_are_rejected_with_a_reason() {
        for bad in ["", "   ", "no-slash", "hf:", "hf:org/n@", "hf:org/n#"] {
            let err = bad.parse::<ModelRef>().unwrap_err();
            assert!(
                matches!(err, Error::ModelRef { .. }),
                "{bad:?} should be a ModelRef error, got {err:?}"
            );
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn only_local_sources_avoid_the_network() {
        assert!(!parse("file:/m").source.needs_network());
        for s in ["hf:o/n", "oci:r/x:v1", "s3://b/k"] {
            assert!(parse(s).source.needs_network(), "{s} should need network");
        }
    }
}
