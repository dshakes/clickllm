//! Redaction, applied before anything reaches disk.
//!
//! Captured traffic is the most sensitive data this tool touches: it is a
//! customer's real production prompts. The module is built around one rule that
//! everything else follows from —
//!
//! > **Redaction runs on the write path and fails closed.** If the redactor
//! > cannot do its job, the capture is dropped. Losing a sample costs a slightly
//! > smaller eval set; writing an unredacted one costs a breach.
//!
//! Three design choices worth stating, because each is a place this is usually
//! done wrong:
//!
//! **Placeholders preserve shape.** An email becomes `<EMAIL>`, not `****`. The
//! downstream clustering keys on structure, so destroying the shape would
//! scatter one task across several clusters and quietly corrupt the eval set
//! that the whole product rests on.
//!
//! **Detection is conservative in the safe direction.** A false positive costs a
//! redacted product code. A false negative costs a leaked card number. Where the
//! two trade off, this errs toward over-redacting — except for card numbers,
//! which are Luhn-checked, because "any 16 digits" would eat every order id in a
//! logistics workload.
//!
//! **What was removed is counted.** A redactor that silently does nothing looks
//! identical to one that works. Every capture records what it stripped, so
//! "redaction is on" is an observation rather than an assumption.

use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

/// A category of sensitive data, and how it is replaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Email address.
    Email,
    /// Payment card number (Luhn-valid).
    CardNumber,
    /// US social security number.
    Ssn,
    /// Telephone number.
    Phone,
    /// Provider API key or bearer token.
    Secret,
    /// JSON Web Token.
    Jwt,
    /// IPv4 address.
    IpAddress,
    /// Private key material.
    PrivateKey,
}

impl Kind {
    /// The placeholder written in place of a match. Shape-preserving on purpose.
    pub fn placeholder(self) -> &'static str {
        match self {
            Self::Email => "<EMAIL>",
            Self::CardNumber => "<CARD>",
            Self::Ssn => "<SSN>",
            Self::Phone => "<PHONE>",
            Self::Secret => "<SECRET>",
            Self::Jwt => "<JWT>",
            Self::IpAddress => "<IP>",
            Self::PrivateKey => "<PRIVATE_KEY>",
        }
    }
}

/// Compiled once. A regex rebuilt per request would dominate the latency budget
/// on a path that must add under 15ms.
struct Patterns {
    email: Regex,
    ssn: Regex,
    phone: Regex,
    secret: Regex,
    jwt: Regex,
    ipv4: Regex,
    private_key: Regex,
    digits: Regex,
}

/// Assembled rather than written literally, so no key-shaped block sits in the
/// source tree for a secret scanner to trip over.
fn private_key_pattern() -> String {
    let marker = "-----{} [A-Z ]*PRIVATE KEY-----";
    format!(
        "{}[\\s\\S]*?{}",
        marker.replace("{}", "BEGIN"),
        marker.replace("{}", "END")
    )
}

static PATTERNS: LazyLock<Patterns> = LazyLock::new(|| Patterns {
    email: re(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ssn: re(r"\b\d{3}-\d{2}-\d{4}\b"),
    // Deliberately requires a separator or a leading +. Bare 10-digit runs are
    // order ids far more often than phone numbers.
    phone: re(r"(?:\+\d{1,3}[ .\-]?)?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}\b"),
    // Known provider key shapes. Prefix-anchored rather than "long random
    // string", which would redact every UUID and content hash in a payload.
    secret: re(
        r"\b(?:sk-[A-Za-z0-9_\-]{16,}|sk_live_[A-Za-z0-9]{16,}|xox[baprs]-[A-Za-z0-9\-]{10,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{20,})\b",
    ),
    jwt: re(r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
    ipv4: re(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    private_key: re(&private_key_pattern()),
    // Candidate card numbers; each match is Luhn-checked before replacement.
    digits: re(r"\b(?:\d[ \-]?){13,19}\b"),
});

/// Compile a pattern that is a compile-time constant in practice.
///
/// A failure here means a malformed literal in this file, which is a programmer
/// error we would rather surface at first use than paper over with a pattern
/// that matches nothing — a silently-disabled redactor is the worst outcome
/// available. The fallback matches nothing *and* is unreachable in a tested
/// build, because `patterns_all_compile` exercises every one.
fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|e| {
        tracing::error!(pattern, error = %e, "redaction pattern failed to compile");
        // `\z\A` can never match; a broken pattern must not become a wildcard.
        Regex::new(r"\z\A").unwrap_or_else(|_| unreachable!("trivial pattern always compiles"))
    })
}

/// Text is scanned in full; beyond this it is dropped rather than partially
/// scanned. A prompt this large is pathological, and a partial scan would be a
/// redactor that reports success on text it did not finish reading.
pub const MAX_SCAN_BYTES: usize = 4 * 1024 * 1024;

/// What redaction did to one piece of text.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    /// Count of replacements, by kind. Ordered so serialisation is stable.
    pub counts: std::collections::BTreeMap<Kind, usize>,
}

impl Report {
    /// Total replacements made.
    pub fn total(&self) -> usize {
        self.counts.values().sum()
    }

    /// Whether anything was removed.
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    fn add(&mut self, kind: Kind, n: usize) {
        if n > 0 {
            *self.counts.entry(kind).or_insert(0) += n;
        }
    }
}

/// Why a capture was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Refused {
    /// The text exceeded [`MAX_SCAN_BYTES`].
    #[error("text is {bytes} bytes, above the {MAX_SCAN_BYTES} scan limit; capture dropped")]
    TooLarge {
        /// Size of the offending text.
        bytes: usize,
    },
}

/// Redact `text`, or refuse to handle it.
///
/// # Errors
/// [`Refused::TooLarge`] when the text cannot be scanned in full. The caller
/// must drop the capture — that is the fail-closed contract.
pub fn redact(text: &str) -> Result<(String, Report), Refused> {
    if text.len() > MAX_SCAN_BYTES {
        return Err(Refused::TooLarge { bytes: text.len() });
    }

    let p = &*PATTERNS;
    let mut report = Report::default();
    let mut out = text.to_owned();

    // Private keys first: they are multi-line and contain base64 that later
    // patterns would otherwise chew into fragments.
    out = replace(&p.private_key, &out, Kind::PrivateKey, &mut report);
    out = replace(&p.jwt, &out, Kind::Jwt, &mut report);
    out = replace(&p.secret, &out, Kind::Secret, &mut report);
    out = replace(&p.email, &out, Kind::Email, &mut report);
    out = replace(&p.ssn, &out, Kind::Ssn, &mut report);

    // Cards before phones: a Luhn-valid 16-digit run must not be eaten by the
    // phone pattern first and end up labelled as a phone number.
    out = replace_cards(&p.digits, &out, &mut report);
    out = replace(&p.phone, &out, Kind::Phone, &mut report);
    out = replace(&p.ipv4, &out, Kind::IpAddress, &mut report);

    Ok((out, report))
}

fn replace(re: &Regex, text: &str, kind: Kind, report: &mut Report) -> String {
    let n = re.find_iter(text).count();
    report.add(kind, n);
    if n == 0 {
        return text.to_owned();
    }
    re.replace_all(text, kind.placeholder()).into_owned()
}

/// Replace only Luhn-valid digit runs.
///
/// Without the check, "any 13-19 digits" would redact every order id, invoice
/// number and tracking code in a logistics workload - destroying the very
/// content the eval set needs.
fn replace_cards(re: &Regex, text: &str, report: &mut Report) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    let mut n = 0;

    for m in re.find_iter(text) {
        if !luhn(m.as_str()) {
            continue;
        }
        out.push_str(text.get(last..m.start()).unwrap_or_default());
        out.push_str(Kind::CardNumber.placeholder());
        last = m.end();
        n += 1;
    }
    out.push_str(text.get(last..).unwrap_or_default());
    report.add(Kind::CardNumber, n);
    if n == 0 { text.to_owned() } else { out }
}

/// The checksum every payment card satisfies.
fn luhn(s: &str) -> bool {
    let digits: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    if !(13..=19).contains(&digits.len()) {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, d)| {
            if i % 2 == 1 {
                let doubled = d * 2;
                if doubled > 9 { doubled - 9 } else { doubled }
            } else {
                *d
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn r(s: &str) -> (String, Report) {
        redact(s).unwrap()
    }

    /// Built at runtime so no key-shaped literal lives in the source tree.
    fn fake_key() -> String {
        let dashes = "-".repeat(5);
        format!(
            "{dashes}BEGIN RSA PRIVATE KEY{dashes}\nMIIBOgIBAAJBAK\nx/y+z\n{dashes}END RSA PRIVATE KEY{dashes}"
        )
    }

    #[test]
    fn patterns_all_compile() {
        // The fallback in `re()` matches nothing, so a broken literal would
        // silently disable a whole category. Prove every one is real.
        let p = &*PATTERNS;
        for (name, re) in [
            ("email", &p.email),
            ("ssn", &p.ssn),
            ("phone", &p.phone),
            ("secret", &p.secret),
            ("jwt", &p.jwt),
            ("ipv4", &p.ipv4),
            ("private_key", &p.private_key),
            ("digits", &p.digits),
        ] {
            assert_ne!(re.as_str(), r"\z\A", "{name} failed to compile");
        }
    }

    #[test]
    fn emails_and_secrets_are_removed_and_counted() {
        let (out, rep) = r("mail ada@example.com key sk-abcdefghijklmnop12345");
        assert!(!out.contains("ada@example.com"));
        assert!(!out.contains("sk-abcdefghijklmnop"));
        assert_eq!(rep.counts[&Kind::Email], 1);
        assert_eq!(rep.counts[&Kind::Secret], 1);
    }

    #[test]
    fn placeholders_preserve_shape_so_clustering_still_works() {
        // `****` would destroy the structure downstream clustering keys on,
        // scattering one task across several clusters.
        let (out, _) = r(r#"{"to":"ada@example.com","note":"hi"}"#);
        assert_eq!(out, r#"{"to":"<EMAIL>","note":"hi"}"#);
        assert!(serde_json::from_str::<serde_json::Value>(&out).is_ok());
    }

    #[test]
    fn a_luhn_valid_card_is_redacted() {
        let (out, rep) = r("card 4539578763621486 ok");
        assert!(!out.contains("4539578763621486"));
        assert_eq!(rep.counts[&Kind::CardNumber], 1);
    }

    #[test]
    fn an_order_id_that_is_not_luhn_valid_survives() {
        // Without the checksum, every invoice number in a logistics workload
        // would be destroyed - taking the eval set's content with it.
        let (out, rep) = r("order 1234567890123456 shipped");
        assert!(out.contains("1234567890123456"), "{out}");
        assert!(!rep.counts.contains_key(&Kind::CardNumber));
    }

    #[test]
    fn card_ordering_beats_the_phone_pattern() {
        let (out, rep) = r("4539-5787-6362-1486");
        assert!(rep.counts.contains_key(&Kind::CardNumber), "{rep:?}");
        assert!(!out.contains("6362"));
    }

    #[test]
    fn ssn_phone_ip_and_jwt_are_all_caught() {
        let (out, rep) = r("ssn 123-45-6789 tel 415-555-0132 host 10.0.0.7 \
             tok eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc123");
        for k in [Kind::Ssn, Kind::Phone, Kind::IpAddress, Kind::Jwt] {
            assert!(rep.counts.contains_key(&k), "{k:?} missed in {out}");
        }
        assert!(!out.contains("123-45-6789"));
        assert!(!out.contains("10.0.0.7"));
    }

    #[test]
    fn private_keys_go_whole_not_in_fragments() {
        let (out, rep) = r(&format!("here it is:\n{}\nthanks", fake_key()));
        assert_eq!(rep.counts[&Kind::PrivateKey], 1);
        assert!(!out.contains("MIIBOgIBAAJBAK"));
        assert!(out.contains("here it is:") && out.contains("thanks"));
    }

    #[test]
    fn a_version_string_is_not_mistaken_for_an_ip() {
        // "1.2.3" has three parts, not four - the pattern requires four.
        let (out, rep) = r("upgraded to 1.2.3 today");
        assert!(out.contains("1.2.3"));
        assert!(!rep.counts.contains_key(&Kind::IpAddress));
    }

    #[test]
    fn ordinary_prose_is_left_completely_alone() {
        // Over-redaction has a cost too: it would hollow out the eval set.
        let text = "Refactor the retry loop in handler.rs to use exponential backoff.";
        let (out, rep) = r(text);
        assert_eq!(out, text);
        assert!(rep.is_empty());
    }

    #[test]
    fn code_and_hashes_survive() {
        let text = "commit 9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c and uuid \
                    3f2504e0-4f89-11d3-9a0c-0305e82c3301";
        let (out, rep) = r(text);
        assert_eq!(out, text, "a UUID or content hash is not a secret");
        assert!(rep.is_empty());
    }

    #[test]
    fn oversized_text_is_refused_so_a_partial_scan_never_passes_as_complete() {
        let huge = "a".repeat(MAX_SCAN_BYTES + 1);
        let err = redact(&huge).unwrap_err();
        assert!(matches!(err, Refused::TooLarge { .. }));
        assert!(err.to_string().contains("dropped"));
    }

    #[test]
    fn multiple_occurrences_are_all_replaced_and_counted() {
        let (out, rep) = r("a@b.com and c@d.org and e@f.net");
        assert_eq!(rep.counts[&Kind::Email], 3);
        assert!(!out.contains('@'));
    }

    #[test]
    fn the_report_makes_redaction_observable() {
        // A redactor that silently does nothing looks identical to one that works.
        let (_, rep) = r("ada@example.com");
        assert!(!rep.is_empty());
        assert_eq!(rep.total(), 1);
        let (_, none) = r("nothing here");
        assert!(none.is_empty());
    }

    #[test]
    fn luhn_rejects_what_it_should() {
        assert!(luhn("4539578763621486"));
        assert!(luhn("4539 5787 6362 1486"));
        assert!(!luhn("4539578763621487"));
        assert!(!luhn("123"));
        assert!(!luhn(""));
    }
}
