//! The control surface, over real TCP against a running gateway.
//!
//! The claim under test is not "the handler returns 200". It is that a phase
//! change on a *live* process actually reroutes the *next* request — which is
//! the whole point of M9's rollback, and which a handler-level test would pass
//! even if the router were cloned and the change thrown away.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::net::SocketAddr;
use std::sync::Arc;

use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Router as AxumRouter, http::StatusCode};
use onpar_gateway::proxy::{AppState, app};
use onpar_gateway::router::{Backend, Phase, Route, Router};

async fn spawn(router: AxumRouter) -> SocketAddr {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(l, router).await;
    });
    addr
}

/// Two upstreams that name themselves, so the response says which one answered.
fn naming(who: &'static str) -> AxumRouter {
    AxumRouter::new().route(
        "/chat/completions",
        post(move || async move {
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                format!(
                    r#"{{"choices":[{{"message":{{"content":"{who}"}}}}],
                        "usage":{{"prompt_tokens":1,"completion_tokens":1}}}}"#
                ),
            )
                .into_response()
        }),
    )
}

async fn gateway(phase: Phase) -> (SocketAddr, Arc<AppState>) {
    let inc = spawn(naming("incumbent")).await;
    let cand = spawn(naming("candidate")).await;
    let state = Arc::new(AppState::new(
        Router::new(Route {
            phase,
            incumbent: Backend {
                name: "incumbent".into(),
                base_url: format!("http://{inc}"),
                model: None,
            },
            candidate: Backend {
                name: "candidate".into(),
                base_url: format!("http://{cand}"),
                model: None,
            },
            failover: false,
        }),
        reqwest::Client::new(),
    ));
    let addr = spawn(app(Arc::clone(&state))).await;
    (addr, state)
}

/// Ask the gateway a question and report which backend answered.
async fn who_answered(addr: SocketAddr) -> String {
    reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "m", "messages": []}))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap()["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned()
}

async fn change(addr: SocketAddr, body: serde_json::Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("http://{addr}/control/phase"))
        .json(&body)
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn a_rollback_reroutes_the_very_next_request() {
    // The claim M9 rests on: no restart, no redeploy, next request is safe.
    let (addr, _st) = gateway(Phase::Cut).await;
    assert_eq!(who_answered(addr).await, "candidate");

    let r = change(
        addr,
        serde_json::json!({"phase": {"phase": "off"}, "reason": "eval regression"}),
    )
    .await;
    assert_eq!(r.status(), 200);

    assert_eq!(
        who_answered(addr).await,
        "incumbent",
        "a rollback that needs a restart is not a rollback"
    );
}

#[tokio::test]
async fn a_rollback_needs_no_human_confirmation() {
    // The safe direction must work at 3am with nobody watching.
    let (addr, _st) = gateway(Phase::Canary { percent: 100 }).await;
    let r = change(
        addr,
        serde_json::json!({"phase": {"phase": "shadow"}, "reason": "p95 doubled"}),
    )
    .await;
    assert_eq!(r.status(), 200, "de-escalation must never be blocked");
    assert_eq!(who_answered(addr).await, "incumbent");
}

#[tokio::test]
async fn an_unconfirmed_escalation_is_refused_and_changes_nothing() {
    let (addr, _st) = gateway(Phase::Shadow).await;
    let r = change(
        addr,
        serde_json::json!({
            "phase": {"phase": "canary", "percent": 50},
            "reason": "gate says the evidence supports it"
        }),
    )
    .await;
    assert_eq!(r.status(), 403);
    let body: serde_json::Value = r.json().await.unwrap();
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(msg.contains("confirmed: true"), "{msg}");
    assert!(msg.contains("0% to 50%"), "{msg}");

    assert_eq!(
        who_answered(addr).await,
        "incumbent",
        "a refused change must not have partially applied"
    );
}

#[tokio::test]
async fn a_confirmed_escalation_is_allowed() {
    let (addr, _st) = gateway(Phase::Shadow).await;
    let r = change(
        addr,
        serde_json::json!({
            "phase": {"phase": "cut"},
            "reason": "reviewed the matrix",
            "confirmed": true
        }),
    )
    .await;
    assert_eq!(r.status(), 200);
    assert_eq!(who_answered(addr).await, "candidate");
}

#[tokio::test]
async fn a_change_without_a_reason_is_refused() {
    // A traffic change with no recorded reason is an incident with no explanation.
    let (addr, _st) = gateway(Phase::Cut).await;
    for reason in ["", "   "] {
        let r = change(
            addr,
            serde_json::json!({"phase": {"phase": "off"}, "reason": reason}),
        )
        .await;
        assert_eq!(r.status(), 400, "reason {reason:?}");
    }
    assert_eq!(who_answered(addr).await, "candidate");
}

#[tokio::test]
async fn pinning_one_cluster_leaves_every_other_where_it_was() {
    // The regret set: one cluster goes back to the incumbent, the rest keep
    // migrating. Moving everything would be a far worse bug than moving nothing.
    let (addr, _st) = gateway(Phase::Cut).await;
    let r = change(
        addr,
        serde_json::json!({
            "cluster": "long-ctx-reasoning",
            "phase": {"phase": "off"},
            "reason": "regressed at 62%"
        }),
    )
    .await;
    assert_eq!(r.status(), 200);

    // Default route untouched.
    assert_eq!(who_answered(addr).await, "candidate");

    // ...and the pinned cluster is on the incumbent.
    let answered = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("x-onpar-cluster", "long-ctx-reasoning")
        .json(&serde_json::json!({"model": "m", "messages": []}))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap()["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(answered, "incumbent");
}

#[tokio::test]
async fn every_applied_change_is_recorded_with_its_reason() {
    let (addr, _st) = gateway(Phase::Cut).await;
    change(
        addr,
        serde_json::json!({"phase": {"phase": "off"}, "reason": "classify regressed"}),
    )
    .await;
    // Refused changes must not appear — the log is what happened, not what was asked.
    change(
        addr,
        serde_json::json!({"phase": {"phase": "cut"}, "reason": "nope"}),
    )
    .await;

    let hist: serde_json::Value = reqwest::get(format!("http://{addr}/control/history"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let rows = hist.as_array().unwrap();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["reason"], "classify regressed");
    assert_eq!(rows[0]["from_percent"], 100);
    assert_eq!(rows[0]["to_percent"], 0);
    assert_eq!(rows[0]["confirmed"], false);
    assert!(rows[0]["at"].as_u64().unwrap() > 1_700_000_000);
}

#[tokio::test]
async fn the_current_phase_is_readable() {
    let (addr, _st) = gateway(Phase::Canary { percent: 25 }).await;
    let v: serde_json::Value = reqwest::get(format!("http://{addr}/control/phase"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let routes = v["routes"].as_array().unwrap();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0]["cluster"], serde_json::Value::Null);
    assert_eq!(routes[0]["candidate_percent"], 25);
    assert_eq!(routes[0]["transitional"], true);
}

#[tokio::test]
async fn traffic_keeps_flowing_while_the_phase_changes_underneath_it() {
    // The lock must not be a stall. Requests and changes interleave for real.
    let (addr, _st) = gateway(Phase::Cut).await;
    let mut tasks = Vec::new();
    for _ in 0..30 {
        tasks.push(tokio::spawn(async move { who_answered(addr).await }));
    }
    for i in 0..6 {
        let phase = if i % 2 == 0 { "off" } else { "cut" };
        change(
            addr,
            serde_json::json!({
                "phase": {"phase": phase}, "reason": "flapping", "confirmed": true
            }),
        )
        .await;
    }
    for t in tasks {
        let who = t.await.unwrap();
        assert!(who == "incumbent" || who == "candidate", "{who}");
    }
}
