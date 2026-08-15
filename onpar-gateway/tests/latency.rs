//! NFR-1, measured rather than asserted.
//!
//! The PRD says proxy overhead must stay **under 15 ms p95 added latency**, and
//! the README sells "no GC pauses against a p95 budget". Until this file, no
//! check anywhere compared those claims to a running gateway — the number lived
//! in a spec with nothing behind it, which is the shape of most of the defects
//! this repo has found.
//!
//! What is measured is *added* latency: the same upstream, the same client, the
//! same machine, with and without the gateway in between. An absolute number
//! would measure the fixture's upstream more than the proxy.
//!
//! Two things this deliberately does:
//!
//! * **Interleaves the two arms** rather than running them in blocks. A CPU
//!   frequency change, a GC pause in the test host, or a noisy neighbour that
//!   lands entirely inside one block reads as added latency and is not.
//! * **Runs with capture on**, because that is the configuration whose overhead
//!   anyone cares about — redaction, encryption and the append are the work.
//!   The no-capture arm is measured too, so the cost of recording is separable
//!   from the cost of proxying.
//!
//! This is a lower bound on real-world overhead, and says so where it prints:
//! a loopback upstream that answers instantly has no network jitter to hide
//! behind, so the proxy's own cost is all that is left. A real deployment adds
//! a network hop the gateway does not control.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    // Both are the arithmetic of a percentile over a bounded sample: the rank
    // is a small non-negative integer by construction, and integer division is
    // what picking a middle index means.
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::integer_division
)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Router as AxumRouter, http::StatusCode};
use onpar_gateway::capture::CaptureStore;
use onpar_gateway::proxy::{AppState, app};
use onpar_gateway::router::{Backend, Phase, Route, Router};

/// The budget, from `docs/20-prd.md` NFR-1.
const BUDGET_MS: f64 = 15.0;

/// Requests per arm. Enough for a p95 to mean something without making the
/// suite slow: 200 puts the 95th percentile at the 10th-worst sample, so one
/// unlucky scheduling hiccup cannot define it.
const SAMPLES: usize = 200;

async fn spawn(router: AxumRouter) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    addr
}

/// A completion with enough text that redaction has real work to do — a
/// one-word response would measure an empty scan.
async fn upstream() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        r#"{"choices":[{"message":{"role":"assistant","content":
            "Thanks for the note. I have emailed ada@example.com and copied
             bob@example.org, and the invoice reference is INV-4417 for the
             amount discussed. Call 555-0142 if anything looks wrong."}}],
            "usage":{"prompt_tokens":64,"completion_tokens":48}}"#,
    )
}

fn backend(name: &str, at: SocketAddr) -> Backend {
    Backend {
        name: name.into(),
        base_url: format!("http://{at}"),
        model: None,
    }
}

fn body() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-5",
        "messages": [{
            "role": "user",
            "content": "Reply to this and confirm: invoice INV-4417, contact ada@example.com",
        }],
    })
}

/// The p95 of a sorted-in-place sample, in milliseconds.
fn p95(mut xs: Vec<Duration>) -> f64 {
    xs.sort_unstable();
    // Nearest-rank: the smallest value at or above the 95th percentile. With
    // 200 samples that is index 189 — the 10th worst — rather than the maximum,
    // which is what an off-by-one here would silently report.
    let i = ((xs.len() as f64) * 0.95).ceil() as usize - 1;
    xs[i.min(xs.len() - 1)].as_secs_f64() * 1000.0
}

fn median(mut xs: Vec<Duration>) -> f64 {
    xs.sort_unstable();
    xs[xs.len() / 2].as_secs_f64() * 1000.0
}

/// Send one request and return how long it took.
///
/// The path differs by arm: the gateway strips the `/v1` prefix before
/// forwarding, so the upstream serves `/chat/completions` and only the proxied
/// arm uses `/v1/...`. Sending both to the same path 404s one of them, and a
/// 404 is much faster than a completion — which would have shown up as the
/// gateway *adding* latency rather than as a broken measurement.
async fn once(client: &reqwest::Client, at: SocketAddr, path: &str) -> Duration {
    let start = Instant::now();
    let resp = client
        .post(format!("http://{at}{path}"))
        .json(&body())
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 200);
    // Read the body: a proxy that returned headers fast and the body slowly
    // would look free if the timer stopped at the status line.
    let _ = resp.bytes().await.expect("body");
    start.elapsed()
}

/// What the upstream serves directly.
const UP: &str = "/chat/completions";
/// What a client sends to the gateway.
const VIA: &str = "/v1/chat/completions";

/// Measure both arms, interleaved.
async fn measure(direct: SocketAddr, proxied: SocketAddr) -> (Vec<Duration>, Vec<Duration>) {
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(4)
        .build()
        .expect("client");

    // Warm both paths: the first request through each pays connection setup and
    // first-touch allocation, which is a real cost but not a per-request one.
    for _ in 0..10 {
        once(&client, direct, UP).await;
        once(&client, proxied, VIA).await;
    }

    let mut d = Vec::with_capacity(SAMPLES);
    let mut p = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES {
        // Alternate which arm goes first, so a periodic disturbance cannot
        // land preferentially on one of them.
        if i % 2 == 0 {
            d.push(once(&client, direct, UP).await);
            p.push(once(&client, proxied, VIA).await);
        } else {
            p.push(once(&client, proxied, VIA).await);
            d.push(once(&client, direct, UP).await);
        }
    }
    (d, p)
}

async fn gateway_over(up: SocketAddr, capture: Option<Arc<CaptureStore>>) -> SocketAddr {
    let mut state = AppState::new(
        Router::new(Route {
            phase: Phase::Off,
            incumbent: backend("incumbent", up),
            candidate: backend("candidate", up),
            failover: false,
        }),
        reqwest::Client::new(),
    );
    if let Some(store) = capture {
        state = state.with_capture(store);
    }
    spawn(app(Arc::new(state))).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn added_latency_stays_inside_the_nfr1_budget() {
    let dir = tempfile::tempdir().expect("tempdir");
    let up = spawn(AxumRouter::new().route("/chat/completions", post(upstream))).await;

    let store = CaptureStore::open(dir.path().join("captures.log"), &[7u8; 32]).expect("store");
    store.ready().expect("writable");
    let recording = gateway_over(up, Some(Arc::new(store))).await;
    let plain = gateway_over(up, None).await;

    let (direct, proxied) = measure(up, recording).await;
    let (_, proxied_plain) = measure(up, plain).await;

    let (d95, p95_rec) = (p95(direct.clone()), p95(proxied));
    let p95_plain = p95(proxied_plain);
    let added_recording = p95_rec - d95;
    let added_plain = p95_plain - d95;

    println!(
        "\nNFR-1, measured over {SAMPLES} interleaved requests per arm:\n\
         \x20 upstream direct        p50 {:.2} ms   p95 {:.2} ms\n\
         \x20 through gateway        p95 {:.2} ms   added {:+.2} ms   (capture on)\n\
         \x20 through gateway        p95 {:.2} ms   added {:+.2} ms   (--no-capture)\n\
         \x20 budget                 {BUDGET_MS:.1} ms added p95\n\
         \x20 note: loopback upstream, so this is a lower bound — a real\n\
         \x20       deployment adds a network hop the gateway does not control.\n",
        median(direct.clone()),
        d95,
        p95_rec,
        added_recording,
        p95_plain,
        added_plain,
    );

    assert!(
        added_recording < BUDGET_MS,
        "NFR-1: {added_recording:.2} ms added p95 with capture on, budget is {BUDGET_MS:.1} ms"
    );
    assert!(
        added_plain < BUDGET_MS,
        "NFR-1: {added_plain:.2} ms added p95 as a plain proxy, budget is {BUDGET_MS:.1} ms"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_measurement_can_detect_overhead_that_is_really_there() {
    // The control for the test above. A latency check that passes because it is
    // measuring nothing is the most comfortable kind of green there is, and this
    // repo has shipped that shape before — so: inject a known delay upstream and
    // confirm the harness reports it, to within a wide tolerance.
    //
    // Without this, `added_latency_stays_inside_the_nfr1_budget` is consistent
    // with a proxy that is fast and with a stopwatch that is broken.
    const INJECTED_MS: u64 = 25;

    async fn slow() -> impl IntoResponse {
        tokio::time::sleep(Duration::from_millis(INJECTED_MS)).await;
        upstream().await
    }

    let fast = spawn(AxumRouter::new().route("/chat/completions", post(upstream))).await;
    let slow_up = spawn(AxumRouter::new().route("/chat/completions", post(slow))).await;

    let client = reqwest::Client::new();
    for _ in 0..5 {
        once(&client, fast, UP).await;
        once(&client, slow_up, UP).await;
    }
    let mut f = Vec::new();
    let mut s = Vec::new();
    for _ in 0..40 {
        f.push(once(&client, fast, UP).await);
        s.push(once(&client, slow_up, UP).await);
    }

    let seen = p95(s) - p95(f);
    println!("\ncontrol: injected {INJECTED_MS} ms upstream, harness measured {seen:.2} ms\n");
    assert!(
        seen > (INJECTED_MS as f64) * 0.5,
        "the harness measured {seen:.2} ms of an injected {INJECTED_MS} ms — it is not \
         measuring latency, so the NFR-1 result above means nothing"
    );
}
