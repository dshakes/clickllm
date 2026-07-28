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

/// A misbehaving upstream: one SSE frame with no terminator, well past
/// `MAX_FRAME_BYTES`, followed by a normal well-formed frame. Exercises the
/// decoder's cap end to end — the gateway must not grow its metering buffer
/// without bound, and must keep streaming raw bytes to the client regardless.
async fn stalled_giant_frame() -> impl IntoResponse {
    let giant = axum::body::Bytes::from(vec![b'x'; clickllm_gateway::sse::MAX_FRAME_BYTES + 1]);
    let rest = axum::body::Bytes::from_static(
        b"\n\ndata: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n",
    );
    let s = futures_util::stream::iter(vec![giant, rest]).map(Ok::<_, std::io::Error>);
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
        failover: false,
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
async fn a_stalled_frame_past_the_cap_does_not_break_the_stream() {
    let up =
        spawn(AxumRouter::new().route("/v1/chat/completions", post(stalled_giant_frame))).await;
    let st = state_pointing_at(up, Phase::Off);
    let gw = spawn(app(Arc::clone(&st))).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "gpt-5", "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Raw bytes pass through untouched even though the internal frame decoder
    // gave up on (and dropped) the oversized frame — the cap bounds the
    // gateway's own buffer, it does not truncate what the client receives.
    let body = resp.bytes().await.unwrap();
    assert!(
        body.len() > clickllm_gateway::sse::MAX_FRAME_BYTES,
        "the client must still receive the full byte stream: got {} bytes",
        body.len()
    );

    for _ in 0..50 {
        if !st.records().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let records = st.records();
    assert_eq!(records.len(), 1, "expected exactly one record");
    assert_eq!(records[0].status, 200);
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
            failover: false,
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

// --------------------------------------------------------------------------- //
// Mirroring must actually dispatch — not merely be recorded and displayed.
// --------------------------------------------------------------------------- //

use std::sync::atomic::{AtomicUsize, Ordering};

/// An upstream that counts the requests it genuinely received.
fn counting_upstream(hits: Arc<AtomicUsize>) -> AxumRouter {
    AxumRouter::new().route(
        "/v1/chat/completions",
        post(move || {
            let hits = Arc::clone(&hits);
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                axum::Json(serde_json::json!({
                    "choices": [{"message": {"content": "candidate"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4}
                }))
            }
        }),
    )
}

async fn wait_for(f: impl Fn() -> bool) -> bool {
    for _ in 0..100 {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    false
}

#[tokio::test]
async fn shadow_actually_sends_the_request_to_the_candidate_backend() {
    // The defect this guards: computing a mirror decision, recording it, and
    // showing it in the console while never dispatching anything. Shadow mode
    // would look healthy and gather no evidence at all.
    let inc_hits = Arc::new(AtomicUsize::new(0));
    let cand_hits = Arc::new(AtomicUsize::new(0));
    let incumbent = spawn(counting_upstream(Arc::clone(&inc_hits))).await;
    let candidate = spawn(counting_upstream(Arc::clone(&cand_hits))).await;

    let st = Arc::new(AppState::new(
        Router::new(Route {
            incumbent: Backend {
                name: "incumbent".into(),
                base_url: format!("http://{incumbent}/v1"),
                model: None,
            },
            candidate: Backend {
                name: "candidate".into(),
                base_url: format!("http://{candidate}/v1"),
                model: None,
            },
            phase: Phase::Shadow,
            failover: false,
        }),
        reqwest::Client::new(),
    ));
    let gw = spawn(app(Arc::clone(&st))).await;

    const N: usize = 10;
    for i in 0..N {
        let resp = reqwest::Client::new()
            .post(format!("http://{gw}/v1/chat/completions"))
            .header("x-request-id", format!("r-{i}"))
            .json(&serde_json::json!({"model": "gpt-5"}))
            .send()
            .await
            .unwrap();
        // The candidate's body must never reach the client.
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "candidate");
    }

    assert_eq!(
        inc_hits.load(Ordering::SeqCst),
        N,
        "incumbent must serve every request"
    );
    assert!(
        wait_for(|| cand_hits.load(Ordering::SeqCst) == N).await,
        "candidate received {} of {N} mirrored requests — mirroring is not dispatching",
        cand_hits.load(Ordering::SeqCst)
    );

    assert!(
        wait_for(|| st.mirrors().len() == N).await,
        "mirror outcomes not recorded"
    );
    let m = &st.mirrors()[0];
    assert_eq!(m.backend, "candidate");
    assert_eq!(m.status, Some(200));
    assert_eq!(
        m.metered.usage().unwrap().total(),
        7,
        "candidate response must be metered"
    );
    assert!(m.error.is_none());
}

#[tokio::test]
async fn a_dead_candidate_is_recorded_rather_than_silently_dropped() {
    // A candidate that is down must be visible: shadow mode looking healthy while
    // gathering no evidence is the failure this prevents.
    let inc_hits = Arc::new(AtomicUsize::new(0));
    let incumbent = spawn(counting_upstream(Arc::clone(&inc_hits))).await;

    let st = Arc::new(AppState::new(
        Router::new(Route {
            incumbent: Backend {
                name: "incumbent".into(),
                base_url: format!("http://{incumbent}/v1"),
                model: None,
            },
            candidate: Backend {
                name: "candidate".into(),
                base_url: "http://127.0.0.1:1/v1".into(), // reserved; nothing listens
                model: None,
            },
            phase: Phase::Shadow,
            failover: false,
        }),
        reqwest::Client::new(),
    ));
    let gw = spawn(app(Arc::clone(&st))).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "gpt-5"}))
        .send()
        .await
        .unwrap();
    // The client is unaffected by the candidate being down.
    assert_eq!(resp.status(), StatusCode::OK);

    assert!(
        wait_for(|| !st.mirrors().is_empty()).await,
        "a failed mirror must still be recorded"
    );
    assert!(
        st.mirrors()[0].error.is_some(),
        "the failure must be visible"
    );
    assert!(st.mirrors()[0].status.is_none());
}

#[tokio::test]
async fn a_slow_candidate_does_not_delay_the_client() {
    async fn slow_candidate() -> impl IntoResponse {
        tokio::time::sleep(Duration::from_millis(800)).await;
        axum::Json(serde_json::json!({"usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
    }
    let incumbent = spawn(AxumRouter::new().route("/v1/chat/completions", post(unary))).await;
    let candidate =
        spawn(AxumRouter::new().route("/v1/chat/completions", post(slow_candidate))).await;

    let st = Arc::new(AppState::new(
        Router::new(Route {
            incumbent: Backend {
                name: "incumbent".into(),
                base_url: format!("http://{incumbent}/v1"),
                model: None,
            },
            candidate: Backend {
                name: "candidate".into(),
                base_url: format!("http://{candidate}/v1"),
                model: None,
            },
            phase: Phase::Shadow,
            failover: false,
        }),
        reqwest::Client::new(),
    ));
    let gw = spawn(app(Arc::clone(&st))).await;

    let started = Instant::now();
    reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "gpt-5"}))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let took = started.elapsed();
    assert!(
        took < Duration::from_millis(400),
        "client waited {took:?} on an 800ms candidate — the mirror is on the hot path"
    );
}

#[tokio::test]
async fn the_console_can_read_the_mirror_endpoint_it_renders() {
    let st = state_pointing_at("127.0.0.1:1".parse().unwrap(), Phase::Off);
    let gw = spawn(app(Arc::clone(&st))).await;
    let c = reqwest::Client::new();

    let html = c
        .get(format!("http://{gw}/"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        html.contains("metrics/mirrors"),
        "console must read the mirror endpoint"
    );

    let m: serde_json::Value = c
        .get(format!("http://{gw}/metrics/mirrors"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(m.is_array());
}

// --------------------------------------------------------------------------- //
// Split routing and failover, over real TCP.
// --------------------------------------------------------------------------- //

fn split_state(local: SocketAddr, cloud: &str, failover: bool) -> Arc<AppState> {
    Arc::new(AppState::new(
        Router::new(Route {
            incumbent: Backend {
                name: "cloud".into(),
                base_url: cloud.to_owned(),
                model: None,
            },
            candidate: Backend {
                name: "local".into(),
                base_url: format!("http://{local}/v1"),
                model: None,
            },
            phase: Phase::Split { to_candidate: true },
            failover,
        }),
        reqwest::Client::new(),
    ))
}

#[tokio::test]
async fn a_split_serves_local_and_never_mirrors_to_the_cloud() {
    let local_hits = Arc::new(AtomicUsize::new(0));
    let cloud_hits = Arc::new(AtomicUsize::new(0));
    let local = spawn(counting_upstream(Arc::clone(&local_hits))).await;
    let cloud = spawn(counting_upstream(Arc::clone(&cloud_hits))).await;

    let st = split_state(local, &format!("http://{cloud}/v1"), false);
    let gw = spawn(app(Arc::clone(&st))).await;

    for i in 0..8 {
        let r = reqwest::Client::new()
            .post(format!("http://{gw}/v1/chat/completions"))
            .header("x-request-id", format!("r{i}"))
            .json(&serde_json::json!({"model": "any"}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.headers().get("x-clickllm-backend").unwrap(), "local");
    }

    assert_eq!(local_hits.load(Ordering::SeqCst), 8);
    // A settled split gathers no evidence — the decision is already made, and
    // mirroring to a paid API forever would be a permanent, invisible bill.
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(
        cloud_hits.load(Ordering::SeqCst),
        0,
        "split must not mirror"
    );
    assert!(st.mirrors().is_empty());
}

#[tokio::test]
async fn failover_reaches_the_cloud_when_local_is_down() {
    // A local model is one machine. When it is restarting, "the request failed"
    // is a worse answer than "it cost a bit more this time".
    let cloud_hits = Arc::new(AtomicUsize::new(0));
    let cloud = spawn(counting_upstream(Arc::clone(&cloud_hits))).await;

    // Port 1 is reserved; nothing listens there.
    let st = split_state(
        "127.0.0.1:1".parse().unwrap(),
        &format!("http://{cloud}/v1"),
        true,
    );
    let gw = spawn(app(Arc::clone(&st))).await;

    let r = reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "any"}))
        .send()
        .await
        .unwrap();

    assert_eq!(
        r.status(),
        StatusCode::OK,
        "failover should have rescued this"
    );
    assert_eq!(r.headers().get("x-clickllm-backend").unwrap(), "cloud");
    assert_eq!(cloud_hits.load(Ordering::SeqCst), 1);

    let rec = &st.records()[0];
    assert!(
        rec.failed_over,
        "a degraded deployment must be visible, not merely working"
    );
    assert_eq!(rec.backend, "cloud");
}

#[tokio::test]
async fn without_failover_a_dead_primary_is_an_error_not_a_surprise_bill() {
    let cloud_hits = Arc::new(AtomicUsize::new(0));
    let cloud = spawn(counting_upstream(Arc::clone(&cloud_hits))).await;
    let st = split_state(
        "127.0.0.1:1".parse().unwrap(),
        &format!("http://{cloud}/v1"),
        false,
    );
    let gw = spawn(app(Arc::clone(&st))).await;

    let r = reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "any"}))
        .send()
        .await
        .unwrap();

    assert_eq!(r.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        cloud_hits.load(Ordering::SeqCst),
        0,
        "traffic must not reach a paid API without failover being asked for"
    );
}

#[tokio::test]
async fn a_healthy_primary_is_not_marked_as_failed_over() {
    let hits = Arc::new(AtomicUsize::new(0));
    let local = spawn(counting_upstream(Arc::clone(&hits))).await;
    let st = split_state(local, "http://127.0.0.1:1/v1", true);
    let gw = spawn(app(Arc::clone(&st))).await;

    reqwest::Client::new()
        .post(format!("http://{gw}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "any"}))
        .send()
        .await
        .unwrap();

    let rec = &st.records()[0];
    assert!(!rec.failed_over);
    assert_eq!(rec.backend, "local");
}
