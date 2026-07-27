//! The HTTP datapath.
//!
//! One route that matters: `POST /v1/chat/completions`. It decides a backend,
//! forwards the request, and returns the upstream response — **streaming bytes
//! through as they arrive**, never buffering, while metering usage in flight.
//!
//! The tee is the interesting part. A naive proxy collects the body, parses it,
//! then forwards it: correct, and it destroys time-to-first-token. Here each
//! chunk is handed to the client and to the [`Meter`] in the same pass, so the
//! client sees the upstream's own latency and we still get a cost figure.

use std::sync::Arc;
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use futures_util::StreamExt;
use parking_lot::Mutex;

use crate::meter::{Meter, Metered};
use crate::router::{Decision, Router};
use crate::sse::Decoder;

/// Shared server state.
pub struct AppState {
    /// Routing policy.
    pub router: Router,
    /// Upstream HTTP client. Cloning is cheap and shares the connection pool —
    /// building one per request would dominate the latency budget.
    pub client: reqwest::Client,
    /// Completed request records. Bounded: the datapath must not become a leak.
    records: Mutex<Vec<Record>>,
    /// Cap on retained records before the oldest are dropped.
    pub max_records: usize,
}

/// What one request cost and where it went.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Record {
    /// Backend that served the response.
    pub backend: String,
    /// Backend the request was mirrored to, if any.
    pub mirrored_to: Option<String>,
    /// Why the router chose this path.
    pub reason: &'static str,
    /// Model the client asked for.
    pub model: String,
    /// Whether the client asked for a stream.
    pub streaming: bool,
    /// Upstream status code.
    pub status: u16,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Token accounting, or an explicit statement that upstream reported none.
    pub metered: Metered,
}

impl AppState {
    /// New state with a default record cap.
    pub fn new(router: Router, client: reqwest::Client) -> Self {
        Self {
            router,
            client,
            records: Mutex::new(Vec::new()),
            max_records: 10_000,
        }
    }

    /// Record a completed request, dropping the oldest once the cap is reached.
    pub fn record(&self, r: Record) {
        let mut g = self.records.lock();
        if g.len() >= self.max_records {
            g.remove(0);
        }
        tracing::info!(
            backend = %r.backend,
            model = %r.model,
            status = r.status,
            duration_ms = r.duration_ms,
            reported = r.metered.is_reported(),
            tokens = r.metered.usage().map_or(0, |u| u.total()),
            "request complete"
        );
        g.push(r);
    }

    /// Snapshot of retained records.
    pub fn records(&self) -> Vec<Record> {
        self.records.lock().clone()
    }
}

/// The local console, embedded in the binary so there is no asset to deploy and
/// nothing to fetch from a network. It renders only what this gateway actually
/// recorded — request volume, routing decisions, and reported token usage.
const CONSOLE_HTML: &str = include_str!("console.html");

/// Build the router. Kept separate from `serve` so tests can drive it directly.
pub fn app(state: Arc<AppState>) -> AxumRouter {
    AxumRouter::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/healthz", get(|| async { "ok" }))
        .route("/metrics/requests", get(records))
        .route("/", get(console))
        .with_state(state)
}

/// Serve the console.
async fn console() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        CONSOLE_HTML,
    )
}

async fn records(State(st): State<Arc<AppState>>) -> Json<Vec<Record>> {
    Json(st.records())
}

/// Errors the datapath can return to a client.
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    /// The request body was not the JSON we need to route on.
    #[error("invalid request body: {0}")]
    BadRequest(String),
    /// The upstream could not be reached.
    #[error("upstream {backend} unreachable: {source}")]
    Upstream {
        /// Backend that failed.
        backend: String,
        /// Underlying transport error.
        #[source]
        source: reqwest::Error,
    },
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, kind) = match &self {
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            Self::Upstream { .. } => (StatusCode::BAD_GATEWAY, "upstream_error"),
        };
        // OpenAI-shaped error, so existing client SDKs surface it properly rather
        // than throwing on an unexpected schema.
        let body = serde_json::json!({
            "error": { "message": self.to_string(), "type": kind }
        });
        (status, Json(body)).into_response()
    }
}

async fn chat_completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    let started = Instant::now();

    let parsed: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| ProxyError::BadRequest(format!("expected JSON: {e}")))?;
    let model = parsed
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_owned();
    let streaming = parsed
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    // The task cluster comes from the control plane, which has already clustered
    // this workload. A header keeps the datapath from having to classify inline —
    // classification is judgment, and judgment does not belong on the hot path.
    let cluster = headers
        .get("x-clickllm-cluster")
        .and_then(|v| v.to_str().ok());
    let key = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&model);

    let decision = st.router.decide(cluster, key);
    let span = tracing::info_span!(
        "chat_completions",
        backend = %decision.serve.name,
        model = %model,
        streaming
    );
    let _e = span.enter();

    let upstream = send(&st, &decision, &parsed, &headers, &body).await?;
    let status = upstream.status();

    let mut out_headers = HeaderMap::new();
    if let Some(ct) = upstream.headers().get(header::CONTENT_TYPE) {
        out_headers.insert(header::CONTENT_TYPE, ct.clone());
    }
    // Streaming responses must not be buffered by an intermediary.
    if streaming {
        out_headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        out_headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    }
    out_headers.insert(
        "x-clickllm-backend",
        HeaderValue::from_str(&decision.serve.name)
            .unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );

    let base = Record {
        backend: decision.serve.name.clone(),
        mirrored_to: decision.mirror.as_ref().map(|m| m.name.clone()),
        reason: decision.reason,
        model,
        streaming,
        status: status.as_u16(),
        duration_ms: 0,
        metered: Metered::Unreported { deltas: 0 },
    };

    if !streaming {
        let bytes = upstream.bytes().await.map_err(|e| ProxyError::Upstream {
            backend: decision.serve.name.clone(),
            source: e,
        })?;
        let mut meter = Meter::new();
        meter.observe_body(&bytes);
        st.record(Record {
            duration_ms: elapsed_ms(started),
            metered: meter.finish(),
            ..base
        });
        return Ok((status, out_headers, bytes).into_response());
    }

    // Streaming: tee each chunk to the client and to the meter in one pass.
    let st2 = Arc::clone(&st);
    // Shared so the drop hook can read what the stream closure accumulated.
    // Without this the record is written from the pre-metering snapshot and every
    // streamed request reports no usage — caught by tests/streaming.rs.
    let meter = Arc::new(Mutex::new(Meter::new()));
    let meter_for_stream = Arc::clone(&meter);
    let mut decoder = Decoder::new();
    let stream = upstream.bytes_stream().map(move |chunk| match chunk {
        Ok(bytes) => {
            let events = decoder.push(&bytes);
            if !events.is_empty() {
                let mut m = meter_for_stream.lock();
                for ev in &events {
                    m.observe(ev);
                }
            }
            // The decoder enforces its own cap and logs what it discarded; the
            // client's bytes are forwarded regardless, so an oversized frame
            // costs metering accuracy and nothing else.
            Ok(bytes)
        }
        Err(e) => {
            tracing::warn!(error = %e, "upstream stream error");
            Err(std::io::Error::other(e))
        }
    });

    // The meter is moved into the stream closure, so the record is finalised by a
    // completion hook rather than here. Simplest correct form: wrap the stream so
    // the final chunk triggers the write.
    let counted = FinishOnDrop {
        state: st2,
        record: Mutex::new(Some(base)),
        meter,
        started,
    };
    let body = Body::from_stream(stream.chain(futures_util::stream::once(async move {
        drop(counted);
        Ok(Bytes::new())
    })));

    Ok((status, out_headers, body).into_response())
}

/// Writes the request record when the response stream finishes or is dropped.
///
/// A client that disconnects mid-stream still produced cost upstream, so the
/// record must be written on drop rather than only on clean completion.
struct FinishOnDrop {
    state: Arc<AppState>,
    record: Mutex<Option<Record>>,
    meter: Arc<Mutex<Meter>>,
    started: Instant,
}

impl Drop for FinishOnDrop {
    fn drop(&mut self) {
        if let Some(mut r) = self.record.lock().take() {
            r.duration_ms = elapsed_ms(self.started);
            r.metered = self.meter.lock().snapshot();
            self.state.record(r);
        }
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

async fn send(
    st: &AppState,
    decision: &Decision,
    parsed: &serde_json::Value,
    headers: &HeaderMap,
    body: &Bytes,
) -> Result<reqwest::Response, ProxyError> {
    // Rewrite the model only when the backend names a different one.
    let payload = match &decision.serve.model {
        Some(m) => {
            let mut v = parsed.clone();
            if let Some(obj) = v.as_object_mut() {
                obj.insert("model".into(), serde_json::Value::String(m.clone()));
            }
            Bytes::from(serde_json::to_vec(&v).unwrap_or_else(|_| body.to_vec()))
        }
        None => body.clone(),
    };

    let mut req = st
        .client
        .post(format!(
            "{}/chat/completions",
            decision.serve.base_url.trim_end_matches('/')
        ))
        .header(header::CONTENT_TYPE, "application/json")
        .body(payload);

    // Forward authorisation untouched: credentials belong to the caller and this
    // process never stores or inspects them.
    if let Some(auth) = headers.get(header::AUTHORIZATION) {
        req = req.header(header::AUTHORIZATION, auth.clone());
    }

    req.send().await.map_err(|e| ProxyError::Upstream {
        backend: decision.serve.name.clone(),
        source: e,
    })
}
