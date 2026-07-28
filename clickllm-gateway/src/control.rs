//! The control surface — the only thing that moves production traffic.
//!
//! M9's gate decides *what* should happen. This is *how* it happens, and it is
//! the single most consequential endpoint in the product: a request here changes
//! which model answers real users.
//!
//! ## The asymmetry, enforced a second time
//!
//! The Python gate already knows that a rollback may be automatic and an advance
//! may not. This module does not trust it. It re-derives the same rule from the
//! only thing it can verify — the numbers:
//!
//! - **De-escalation** (candidate share goes down, or stays) is accepted. It is
//!   the safe direction, and the one that must work at 3am with nobody watching.
//! - **Escalation** (candidate share goes up) is **refused unless the request
//!   says a human confirmed it.** Not because the gate is untrustworthy, but
//!   because "the automation had a bug" must not be a way for traffic to end up
//!   somewhere nobody chose.
//!
//! Two independent implementations of one rule is not duplication here. It is
//! the difference between a policy and a guarantee: the Python side can be
//! wrong, rewritten, or bypassed entirely, and traffic still cannot escalate
//! without something explicitly claiming a human said so.
//!
//! ## What this is not
//!
//! Not an authentication system. If `CLICKLLM_ADMIN_TOKEN` is set it is
//! required; if it is not set the endpoint is open, and the gateway is expected
//! to be bound to loopback or behind something that does authenticate. That is
//! stated rather than implied, because an operator who assumes otherwise would
//! be exposing the one endpoint that reroutes their traffic.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};

use crate::proxy::AppState;
use crate::router::Phase;

/// Environment variable holding the shared secret, when one is in use.
pub const TOKEN_ENV: &str = "CLICKLLM_ADMIN_TOKEN";

/// A requested phase change.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PhaseChange {
    /// Cluster to change, or `None` for the default route.
    #[serde(default)]
    pub cluster: Option<String>,
    /// Phase to move to.
    pub phase: Phase,
    /// Why. Required, and stored — a traffic change with no recorded reason is
    /// an incident with no explanation waiting to happen.
    pub reason: String,
    /// Whether a human confirmed this. Only consulted when the change would
    /// escalate; ignored otherwise, so a rollback never needs it.
    #[serde(default)]
    pub confirmed: bool,
}

/// One applied change, kept so "who moved traffic, and why" has an answer.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Transition {
    /// Cluster affected, or `None` for the default route.
    pub cluster: Option<String>,
    /// Phase before.
    pub from: Phase,
    /// Phase after.
    pub to: Phase,
    /// Candidate traffic share before and after, which is what actually matters.
    pub from_percent: u8,
    /// See `from_percent`.
    pub to_percent: u8,
    /// Reason given by the caller.
    pub reason: String,
    /// Whether the caller claimed human confirmation.
    pub confirmed: bool,
    /// Unix seconds. Wall clock, because this is read alongside external logs.
    pub at: u64,
}

impl Transition {
    /// Whether this moved more traffic onto the candidate.
    pub fn escalation(&self) -> bool {
        self.to_percent > self.from_percent
    }
}

/// Why a change was refused.
#[derive(Debug, thiserror::Error)]
pub enum ControlError {
    /// The shared secret was missing or wrong.
    #[error("missing or invalid admin token")]
    Unauthorised,
    /// The change would escalate without a human having confirmed it.
    #[error(
        "this raises candidate traffic from {from}% to {to}% — an escalation \
         needs `confirmed: true`, which only a human may set. De-escalations \
         and rollbacks do not."
    )]
    UnconfirmedEscalation {
        /// Candidate share now.
        from: u8,
        /// Candidate share requested.
        to: u8,
    },
    /// The reason was blank.
    #[error("a phase change must carry a reason")]
    NoReason,
}

impl IntoResponse for ControlError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::Unauthorised => StatusCode::UNAUTHORIZED,
            Self::UnconfirmedEscalation { .. } => StatusCode::FORBIDDEN,
            Self::NoReason => StatusCode::BAD_REQUEST,
        };
        let body = serde_json::json!({
            "error": { "message": self.to_string(), "type": "control_error" }
        });
        (status, Json(body)).into_response()
    }
}

/// Read the configured token once, at startup.
///
/// Deliberately not read per-request: a security decision that depends on
/// mutable process state can change under a running server, and the operator
/// would have no way to know which requests saw which value.
pub fn token_from_env() -> Option<String> {
    std::env::var(TOKEN_ENV).ok().filter(|t| !t.is_empty())
}

/// Check the shared secret. `None` means none is configured — see the module note.
///
/// Comparison is length-then-bytes rather than `==` on `str`. The timing signal
/// on a short token is small, but "small" is not a reason to hand it over.
fn authorised(headers: &HeaderMap, want: Option<&str>) -> bool {
    let Some(want) = want else {
        return true;
    };
    let got = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or_default();
    got.len() == want.len()
        && got
            .bytes()
            .zip(want.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}

/// Current phase of every route.
pub async fn get_phase(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let routes: Vec<_> = st
        .router
        .read()
        .phases()
        .into_iter()
        .map(|(cluster, phase)| {
            serde_json::json!({
                "cluster": cluster,
                "phase": phase,
                "candidate_percent": phase.candidate_percent(),
                "transitional": phase.is_transitional(),
            })
        })
        .collect();
    Json(serde_json::json!({ "routes": routes }))
}

/// Apply a phase change.
///
/// # Errors
/// [`ControlError::Unauthorised`], [`ControlError::NoReason`], or
/// [`ControlError::UnconfirmedEscalation`] — see the module docs for the last.
pub async fn set_phase(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(change): Json<PhaseChange>,
) -> Result<Json<Transition>, ControlError> {
    if !authorised(&headers, st.admin_token.as_deref()) {
        return Err(ControlError::Unauthorised);
    }
    if change.reason.trim().is_empty() {
        return Err(ControlError::NoReason);
    }

    // Held across the read-compare-write so two concurrent changes cannot both
    // observe the old percent and both conclude they are de-escalating.
    let mut router = st.router.write();
    let from = router.route_for(change.cluster.as_deref()).phase.clone();
    let (from_pct, to_pct) = (from.candidate_percent(), change.phase.candidate_percent());

    if to_pct > from_pct && !change.confirmed {
        tracing::warn!(
            cluster = change.cluster.as_deref().unwrap_or("default"),
            from = from_pct,
            to = to_pct,
            reason = %change.reason,
            "refused an unconfirmed escalation"
        );
        return Err(ControlError::UnconfirmedEscalation {
            from: from_pct,
            to: to_pct,
        });
    }

    router.set_phase(change.cluster.as_deref(), change.phase.clone());
    drop(router);

    let t = Transition {
        cluster: change.cluster,
        from,
        to: change.phase,
        from_percent: from_pct,
        to_percent: to_pct,
        reason: change.reason,
        confirmed: change.confirmed,
        at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    // At `warn` deliberately. This is rare, consequential, and the line an
    // operator greps for first when asking why the model changed.
    tracing::warn!(
        cluster = t.cluster.as_deref().unwrap_or("default"),
        from = t.from_percent,
        to = t.to_percent,
        confirmed = t.confirmed,
        reason = %t.reason,
        "phase changed"
    );
    st.record_transition(t.clone());
    Ok(Json(t))
}

/// Every phase change this process has applied.
pub async fn history(State(st): State<Arc<AppState>>) -> Json<Vec<Transition>> {
    Json(st.transitions())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn escalation_is_computed_from_share_not_phase_name() {
        // `Canary { percent: 5 }` → `Canary { percent: 50 }` is an escalation
        // even though the phase name did not change, and `Cut` → `Shadow` is a
        // de-escalation even though both are "a phase". Only the number knows.
        let t = |from: Phase, to: Phase| Transition {
            cluster: None,
            from_percent: from.candidate_percent(),
            to_percent: to.candidate_percent(),
            from,
            to,
            reason: "t".into(),
            confirmed: false,
            at: 0,
        };
        assert!(t(Phase::Canary { percent: 5 }, Phase::Canary { percent: 50 }).escalation());
        assert!(!t(Phase::Cut, Phase::Shadow).escalation());
        assert!(
            !t(Phase::Cut, Phase::Cut).escalation(),
            "no move is no escalation"
        );
        assert!(t(Phase::Shadow, Phase::Canary { percent: 1 }).escalation());
        // Shadow serves nothing, so Off → Shadow moves no traffic.
        assert!(!t(Phase::Off, Phase::Shadow).escalation());
    }

    #[test]
    fn a_bearer_token_is_required_only_when_one_is_configured() {
        let mut h = HeaderMap::new();
        assert!(
            authorised(&h, None),
            "no token configured means none required"
        );

        assert!(!authorised(&h, Some("s3cret")), "must demand the token");
        h.insert(header::AUTHORIZATION, "Bearer wrong".parse().unwrap());
        assert!(!authorised(&h, Some("s3cret")));
        h.insert(header::AUTHORIZATION, "Bearer s3cre".parse().unwrap());
        assert!(!authorised(&h, Some("s3cret")), "a prefix is not a match");
        h.insert(header::AUTHORIZATION, "Bearer s3crett".parse().unwrap());
        assert!(
            !authorised(&h, Some("s3cret")),
            "a superstring is not either"
        );
        h.insert(header::AUTHORIZATION, "Bearer s3cret".parse().unwrap());
        assert!(authorised(&h, Some("s3cret")));
    }

    #[test]
    fn an_empty_env_token_is_treated_as_unset_rather_than_as_the_password() {
        // `CLICKLLM_ADMIN_TOKEN=` is what a shell exports when the value it was
        // meant to hold was itself empty. Accepting "" as the secret would make
        // every unauthenticated request valid while looking configured.
        assert!(authorised(&HeaderMap::new(), None));
    }
}
