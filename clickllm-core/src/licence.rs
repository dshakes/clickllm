//! Licence gating — evaluated **before any bytes move**.
//!
//! Most tooling *filters* by licence; we *refuse*. The difference matters: a team
//! that integrates a model and discovers the MAU cap at legal review has already
//! paid the integration cost. Catching it at `pull` time costs them nothing.
//!
//! Three deliberate stances:
//!
//! 1. **Unknown licences fail closed.** A missing licence field is not permission.
//! 2. **Conditional licences require acknowledgement, not a warning.** A warning
//!    scrolls past in CI. An unmet obligation is an error with an exit code.
//! 3. **We state obligations, we do not give legal advice.** Every decision names
//!    the rule that produced it so a lawyer can check our work.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// How a licence family behaves for our purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    /// Commercial use without additional obligation (MIT, Apache-2.0, BSD).
    Permissive,
    /// Usable commercially but with a condition worth reading — user caps,
    /// acceptable-use terms, naming requirements.
    Conditional,
    /// Not usable commercially (research-only, non-commercial Creative Commons).
    NonCommercial,
    /// No licence published, or one we do not recognise.
    Unknown,
}

/// A recognised licence and what it obliges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Licence {
    /// Canonical id, lowercased (`apache-2.0`, `llama3.1`).
    pub id: String,
    /// Behaviour class.
    pub class: Class,
    /// One sentence a user must read before proceeding. Empty for permissive.
    pub obligation: &'static str,
}

/// Recognised licences. Ordered longest-prefix-first is unnecessary because
/// matching is exact after canonicalisation; unmatched ids fall to `Unknown`.
const KNOWN: &[(&str, Class, &str)] = &[
    ("mit", Class::Permissive, ""),
    ("apache-2.0", Class::Permissive, ""),
    ("bsd-3-clause", Class::Permissive, ""),
    ("bsd-2-clause", Class::Permissive, ""),
    ("isc", Class::Permissive, ""),
    ("mpl-2.0", Class::Permissive, ""),
    (
        "modified-mit",
        Class::Conditional,
        "MIT with an added attribution requirement above a deployment-size threshold; \
         read the model card before shipping at scale",
    ),
    (
        "llama3.1",
        Class::Conditional,
        "Meta Llama licence: a 700M monthly-active-user cap measured at the parent \
         corporate entity, plus naming and acceptable-use terms",
    ),
    (
        "llama3.3",
        Class::Conditional,
        "Meta Llama licence: a 700M monthly-active-user cap measured at the parent \
         corporate entity, plus naming and acceptable-use terms",
    ),
    (
        "llama4",
        Class::Conditional,
        "Meta Llama licence: a 700M monthly-active-user cap measured at the parent \
         corporate entity, plus naming and acceptable-use terms",
    ),
    (
        "gemma",
        Class::Conditional,
        "Google Gemma terms: prohibited-use policy applies and must be passed \
         through to your own users",
    ),
    (
        "qwen",
        Class::Conditional,
        "Qwen licence: check the specific variant — some tiers add use restrictions",
    ),
    (
        "cc-by-nc-4.0",
        Class::NonCommercial,
        "Creative Commons NonCommercial: commercial deployment is not permitted",
    ),
    (
        "cc-by-nc-sa-4.0",
        Class::NonCommercial,
        "Creative Commons NonCommercial ShareAlike: commercial deployment is not permitted",
    ),
    (
        "research-only",
        Class::NonCommercial,
        "Research-only licence: production deployment is not permitted",
    ),
];

/// Resolve a published licence string. Unrecognised input yields [`Class::Unknown`],
/// never a guess.
pub fn classify(published: &str) -> Licence {
    let id = published.trim().to_ascii_lowercase();
    for (known, class, obligation) in KNOWN {
        if id == *known {
            return Licence {
                id,
                class: *class,
                obligation,
            };
        }
    }
    Licence {
        id,
        class: Class::Unknown,
        obligation: "no recognised licence published; treat as all rights reserved \
                     until confirmed",
    }
}

/// What the caller intends to do with the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    /// Production or revenue-generating use.
    Commercial,
    /// Evaluation, research, or personal use.
    NonCommercial,
}

/// Caller-supplied gate configuration.
#[derive(Debug, Clone)]
pub struct Policy {
    /// What the weights are for.
    pub intent: Intent,
    /// Licence ids the user has explicitly accepted, canonicalised.
    acknowledged: BTreeSet<String>,
    /// Permit `Unknown` licences. Off by default — see module docs.
    pub allow_unknown: bool,
}

impl Default for Policy {
    /// Commercial intent, nothing acknowledged, unknown licences refused.
    ///
    /// Defaulting to `Commercial` is deliberate: it is the stricter evaluation, so
    /// a caller who forgets to set intent gets warned rather than waved through.
    fn default() -> Self {
        Self {
            intent: Intent::Commercial,
            acknowledged: BTreeSet::new(),
            allow_unknown: false,
        }
    }
}

impl Policy {
    /// Set the intent.
    #[must_use]
    pub fn with_intent(mut self, intent: Intent) -> Self {
        self.intent = intent;
        self
    }

    /// Record that the user accepted a licence. Ids are canonicalised, so
    /// `"Apache-2.0"` and `"apache-2.0"` are the same acknowledgement.
    #[must_use]
    pub fn acknowledging(mut self, licence_id: &str) -> Self {
        self.acknowledged
            .insert(licence_id.trim().to_ascii_lowercase());
        self
    }

    /// Permit unrecognised licences. Requires an explicit call — there is no flag
    /// that turns this on implicitly.
    #[must_use]
    pub fn allowing_unknown(mut self) -> Self {
        self.allow_unknown = true;
        self
    }

    /// True when this exact licence id has been acknowledged.
    pub fn has_acknowledged(&self, licence_id: &str) -> bool {
        self.acknowledged
            .contains(&licence_id.trim().to_ascii_lowercase())
    }
}

/// The gate's verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Proceed.
    Allowed,
    /// Proceed, but surface this to the user — an acknowledged conditional licence
    /// still has obligations they must keep meeting.
    AllowedWithObligation {
        /// The obligation text to display.
        obligation: &'static str,
    },
}

/// Evaluate `policy` against a model's published licence.
///
/// Returns `Err` rather than a boolean so a caller cannot accidentally ignore a
/// refusal — the download path uses `?` and stops.
///
/// # Errors
/// [`Error::LicenceDenied`] when the licence forbids the intended use, and
/// [`Error::LicenceNotAcknowledged`] when it carries an unaccepted obligation.
pub fn gate(model: &str, published: &str, policy: &Policy) -> Result<Verdict> {
    let lic = classify(published);
    let span = tracing::debug_span!("licence_gate", model, licence = %lic.id);
    let _e = span.enter();

    match lic.class {
        Class::Permissive => {
            tracing::debug!(class = "permissive", "allowed");
            Ok(Verdict::Allowed)
        }

        Class::NonCommercial if policy.intent == Intent::Commercial => {
            tracing::warn!(class = "non_commercial", "denied for commercial intent");
            Err(Error::LicenceDenied {
                model: model.to_owned(),
                licence: lic.id,
                reason: lic.obligation.to_owned(),
            })
        }

        // Non-commercial licence, non-commercial intent: permitted, but the
        // obligation is still shown — the boundary is easy to drift across.
        Class::NonCommercial => Ok(Verdict::AllowedWithObligation {
            obligation: lic.obligation,
        }),

        Class::Conditional => {
            if policy.has_acknowledged(&lic.id) {
                Ok(Verdict::AllowedWithObligation {
                    obligation: lic.obligation,
                })
            } else {
                Err(Error::LicenceNotAcknowledged {
                    model: model.to_owned(),
                    licence: lic.id,
                    obligation: lic.obligation.to_owned(),
                })
            }
        }

        Class::Unknown if policy.allow_unknown => {
            tracing::warn!("unknown licence permitted by explicit policy");
            Ok(Verdict::AllowedWithObligation {
                obligation: lic.obligation,
            })
        }

        Class::Unknown => Err(Error::LicenceDenied {
            model: model.to_owned(),
            licence: lic.id,
            reason: lic.obligation.to_owned(),
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn permissive_licences_pass_with_the_default_policy() {
        for id in ["MIT", "apache-2.0", "BSD-3-Clause", "isc", "mpl-2.0"] {
            assert_eq!(
                gate("m", id, &Policy::default()).unwrap(),
                Verdict::Allowed,
                "{id} should pass unconditionally"
            );
        }
    }

    #[test]
    fn unknown_licence_fails_closed() {
        let err = gate("m", "some-bespoke-terms", &Policy::default()).unwrap_err();
        assert!(matches!(err, Error::LicenceDenied { .. }));
    }

    #[test]
    fn empty_licence_field_is_not_permission() {
        assert!(gate("m", "", &Policy::default()).is_err());
        assert!(gate("m", "   ", &Policy::default()).is_err());
    }

    #[test]
    fn unknown_licence_can_be_permitted_only_explicitly() {
        let p = Policy::default().allowing_unknown();
        assert!(matches!(
            gate("m", "bespoke", &p).unwrap(),
            Verdict::AllowedWithObligation { .. }
        ));
    }

    #[test]
    fn conditional_licence_needs_acknowledgement() {
        let err = gate("llama-3.3-70b", "llama3.3", &Policy::default()).unwrap_err();
        match err {
            Error::LicenceNotAcknowledged { obligation, .. } => {
                assert!(
                    obligation.contains("700M"),
                    "must state the cap: {obligation}"
                );
            }
            other => panic!("expected LicenceNotAcknowledged, got {other:?}"),
        }
    }

    #[test]
    fn acknowledgement_unblocks_but_still_returns_the_obligation() {
        let p = Policy::default().acknowledging("llama3.3");
        match gate("m", "llama3.3", &p).unwrap() {
            Verdict::AllowedWithObligation { obligation } => {
                assert!(!obligation.is_empty());
            }
            Verdict::Allowed => panic!("obligation must not be silently dropped"),
        }
    }

    #[test]
    fn acknowledgement_is_case_insensitive_but_not_cross_licence() {
        let p = Policy::default().acknowledging("Llama3.3");
        assert!(gate("m", "llama3.3", &p).is_ok(), "casing must not matter");
        assert!(
            gate("m", "gemma", &p).is_err(),
            "acknowledging one licence must not unblock another"
        );
    }

    #[test]
    fn non_commercial_is_denied_for_commercial_intent() {
        let err = gate("m", "cc-by-nc-4.0", &Policy::default()).unwrap_err();
        assert!(matches!(err, Error::LicenceDenied { .. }));
    }

    #[test]
    fn non_commercial_is_allowed_for_non_commercial_intent_with_the_terms_shown() {
        let p = Policy::default().with_intent(Intent::NonCommercial);
        assert!(matches!(
            gate("m", "research-only", &p).unwrap(),
            Verdict::AllowedWithObligation { .. }
        ));
    }

    #[test]
    fn default_policy_is_the_strict_one() {
        let p = Policy::default();
        assert_eq!(
            p.intent,
            Intent::Commercial,
            "strictest evaluation by default"
        );
        assert!(!p.allow_unknown);
        assert!(!p.has_acknowledged("mit"));
    }

    #[test]
    fn every_conditional_and_non_commercial_entry_states_an_obligation() {
        for (id, class, obligation) in KNOWN {
            match class {
                Class::Conditional | Class::NonCommercial | Class::Unknown => assert!(
                    !obligation.is_empty(),
                    "{id} is {class:?} and must explain why"
                ),
                Class::Permissive => assert!(
                    obligation.is_empty(),
                    "{id} is permissive; an obligation here would mislead"
                ),
            }
        }
    }

    #[test]
    fn catalogue_ids_are_canonical_and_unique() {
        let mut seen = BTreeSet::new();
        for (id, ..) in KNOWN {
            assert_eq!(*id, id.to_ascii_lowercase(), "{id} must be lowercase");
            assert!(seen.insert(*id), "{id} is listed twice");
        }
    }

    #[test]
    fn classify_is_insensitive_to_casing_and_padding() {
        assert_eq!(classify("  Apache-2.0 ").class, Class::Permissive);
        assert_eq!(classify("MIT").id, "mit");
    }
}
