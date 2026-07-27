//! End-to-end tests over real TCP.
//!
//! No mocked transport: a real upstream server binds a real port, the gateway
//! binds another, and a real HTTP client talks to it. That is the only way to
//! prove the streaming path actually streams — a test that calls the handler
//! directly would pass even if the response were buffered end-to-end.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Router as AxumRouter, http::StatusCode};
use clickllm_gateway::proxy::{AppState, app};
use clickllm_gateway::router::{Backend, Phase, Route, Router};
use futures_util::StreamExt;

/// Spawn a server on an ephemeral port and return its address.
async fn spawn(router: AxumRouter) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    addr
}

/// An upstream that emits SSE frames with a delay between them, so a buffering
/// proxy is distinguishable from a streaming one by timing alone.
async fn slow_stream() -> impl IntoResponse {
    let frames = vec![
        r#"data: {"choices":[{"delta":{"content":"one"}}]}"#.to_string(),
        r#"data: {"choices":[{"delta":{"content":"two"}}]}"#.to_string(),
        r#"data: {"choices":[{"delta":{"content":"three"}}],"usage":null}"#.to_string(),
        r#"data: {"usage":{"prompt_tokens":11,"completion_tokens":3}}"#.to_string(),
        "data: [DONE]".to_string(),
    ];
    let s = futures_util::stream::iter(frames).then(|f| async move {
        tokio::time::sleep(Duration::from_millis(60)).await;
        Ok::<_, std::io::Error>(axum::body::Bytes::from(format!("{f}\n\n")))
    });
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        axum::body::Body::from_stream(s),
    )
}

async fn unary() -> impl IntoResponse {
    axum::Json(serde_json::json!({
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7}
    }))
}

fn state_pointing_at(addr: SocketAddr, phase: Phase) -> Arc<AppState> {
    let backend = |n: &str| Backend {
        name: n.to_owned(),
        base_url: format!("http://{addr}/v1"),
        model: None,
    };
    let router = Router::new(Route {
        incumbent: backend("incumbent"),
        candidate: backend("candidate"),
        phase,
    });
    Arc::new(AppState::new(router, reqwest::Client::new()))
}

#[tokio::test]
async fn streaming_response_reaches_the_client_incrementally() {
    let up = spawn(AxumRouter::new().route("/v1/chat/completions", post(slow_stream))).await;
    let st = state_pointing_at(up, Phase::Off);
    let gw = spawn(app(Arc::clone(&st))).await;

    let started = Instant::now();
    let resp = reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "gpt-5", "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("x-clickllm-backend").unwrap(),
        "incumbent"
    );

    // The first byte must arrive long before the last frame is produced.
    // 5 frames x 60ms means a buffering proxy cannot answer before ~300ms.
    let mut stream = resp.bytes_stream();
    let first = stream.next().await.unwrap().unwrap();
    let ttfb = started.elapsed();
    assert!(!first.is_empty());
    assert!(
        ttfb < Duration::from_millis(250),
        "first byte took {ttfb:?} — the response was buffered, not streamed"
    );

    let mut seen = first.len();
    while let Some(chunk) = stream.next().await {
        seen += chunk.unwrap().len();
    }
    assert!(seen > 100, "body was truncated: {seen} bytes");
}

#[tokio::test]
async fn streamed_usage_is_metered_without_buffering() {
    let up = spawn(AxumRouter::new().route("/v1/chat/completions", post(slow_stream))).await;
    let st = state_pointing_at(up, Phase::Off);
    let gw = spawn(app(Arc::clone(&st))).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "gpt-5", "stream": true}))
        .send()
        .await
        .unwrap();
    // Drain fully so the record is finalised.
    let _ = resp.bytes().await.unwrap();

    // The record is written when the response body drops; give it a moment.
    for _ in 0..50 {
        if !st.records().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let records = st.records();
    assert_eq!(records.len(), 1, "expected exactly one record");
    let r = &records[0];
    assert!(r.streaming);
    assert_eq!(r.status, 200);
    assert!(
        r.metered.is_reported(),
        "usage was in the stream and should have been metered: {:?}",
        r.metered
    );
    let u = r.metered.usage().unwrap();
    assert_eq!(u.prompt_tokens, 11);
    assert_eq!(u.completion_tokens, 3);
}

#[tokio::test]
async fn unary_requests_are_metered_too() {
    let up = spawn(AxumRouter::new().route("/v1/chat/completions", post(unary))).await;
    let st = state_pointing_at(up, Phase::Off);
    let gw = spawn(app(Arc::clone(&st))).await;

    let body: serde_json::Value = reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "gpt-5"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "hello");

    let records = st.records();
    assert_eq!(records.len(), 1);
    assert!(!records[0].streaming);
    assert_eq!(records[0].metered.usage().unwrap().total(), 12);
}

#[tokio::test]
async fn shadow_phase_never_returns_the_candidate_to_the_client() {
    // The invariant the entire safety story rests on, proven over real HTTP.
    let up = spawn(AxumRouter::new().route("/v1/chat/completions", post(unary))).await;
    let st = state_pointing_at(up, Phase::Shadow);
    let gw = spawn(app(Arc::clone(&st))).await;

    for i in 0..25 {
        let resp = reqwest::Client::new()
            .post(format!("http://{gw}/v1/chat/completions"))
            .header("x-request-id", format!("req-{i}"))
            .json(&serde_json::json!({"model": "gpt-5"}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get("x-clickllm-backend").unwrap(),
            "incumbent"
        );
    }
    assert!(
        st.records().iter().all(|r| r.backend == "incumbent"),
        "shadow served the candidate"
    );
    assert!(
        st.records().iter().all(|r| r.mirrored_to.is_some()),
        "shadow must mirror every request"
    );
}

#[tokio::test]
async fn an_unreachable_upstream_returns_502_in_the_openai_error_shape() {
    // Port 1 is reserved and nothing listens there.
    let backend = Backend {
        name: "dead".into(),
        base_url: "http://127.0.0.1:1/v1".into(),
        model: None,
    };
    let st = Arc::new(AppState::new(
        Router::new(Route {
            incumbent: backend.clone(),
            candidate: backend,
            phase: Phase::Off,
        }),
        reqwest::Client::new(),
    ));
    let gw = spawn(app(st)).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "gpt-5"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    let body: serde_json::Value = resp.json().await.unwrap();
    // Existing OpenAI SDKs parse this shape; anything else makes them throw.
    assert!(body["error"]["message"].is_string());
    assert_eq!(body["error"]["type"], "upstream_error");
}

#[tokio::test]
async fn a_malformed_body_is_a_400_not_a_panic() {
    let st = state_pointing_at("127.0.0.1:1".parse().unwrap(), Phase::Off);
    let gw = spawn(app(st)).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body("not json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn healthz_and_metrics_endpoints_answer() {
    let st = state_pointing_at("127.0.0.1:1".parse().unwrap(), Phase::Off);
    let gw = spawn(app(st)).await;

    let c = reqwest::Client::new();
    assert_eq!(
        c.get(format!("http://{gw}/healthz"))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    let m: serde_json::Value = c
        .get(format!("http://{gw}/metrics/requests"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(m.is_array(), "metrics must be a JSON array");
}

#[tokio::test]
async fn the_console_is_served_and_reflects_real_recorded_traffic() {
    let up = spawn(AxumRouter::new().route("/v1/chat/completions", post(unary))).await;
    let st = state_pointing_at(up, Phase::Off);
    let gw = spawn(app(Arc::clone(&st))).await;
    let c = reqwest::Client::new();

    let page = c.get(format!("http://{gw}/")).send().await.unwrap();
    assert_eq!(page.status(), StatusCode::OK);
    let ct = page.headers()[axum::http::header::CONTENT_TYPE]
        .to_str()
        .unwrap()
        .to_owned();
    assert!(ct.starts_with("text/html"), "got {ct}");
    let html = page.text().await.unwrap();
    // Self-contained: no external fetches, so it works air-gapped.
    assert!(
        !html.contains("http://") && !html.contains("https://"),
        "console must not load remote assets"
    );
    assert!(
        html.contains("metrics/requests"),
        "console must read the real endpoint"
    );

    // The endpoint it reads must carry what the console renders.
    c.post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "gpt-5"}))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    let rows: serde_json::Value = c
        .get(format!("http://{gw}/metrics/requests"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let row = &rows[0];
    for field in [
        "backend",
        "model",
        "reason",
        "status",
        "duration_ms",
        "metered",
    ] {
        assert!(!row[field].is_null(), "console needs {field}");
    }
    // The honesty contract: metered is a tagged union, so "no usage reported"
    // can never be rendered as zero.
    assert!(row["metered"]["kind"].is_string());
}
