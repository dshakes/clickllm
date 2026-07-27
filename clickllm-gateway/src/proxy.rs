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

use crate::capture::store::{self, Capture, CaptureStore};
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
    /// Outcomes of mirrored (shadow) requests.
    mirrors: Mutex<Vec<MirrorRecord>>,
    /// Cap on retained records before the oldest are dropped.
    pub max_records: usize,
    /// Where captured traffic is persisted. `None` — the default — means nothing
    /// is written. Recording a customer's production prompts is a decision
    /// someone makes, never something that happens because they started a proxy.
    capture: Option<Arc<CaptureStore>>,
}

/// What one request cost and where it went.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Record {
    /// Backend that served the response.
    pub backend: String,
    /// True when the primary was unreachable and the fallback answered. Surfaced
    /// so a quietly-degraded deployment is visible rather than merely working.
    pub failed_over: bool,
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

/// What a mirrored request did. Written independently of the client's response,
/// because the mirror must never delay or influence it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MirrorRecord {
    /// Candidate backend the request was mirrored to.
    pub backend: String,
    /// Model the client asked for.
    pub model: String,
    /// Status the candidate returned, or `None` if it could not be reached.
    pub status: Option<u16>,
    /// How long the candidate took.
    pub duration_ms: u64,
    /// Token accounting for the candidate's response.
    pub metered: Metered,
    /// Transport failure, if any. A candidate that is down must be visible —
    /// silently dropping mirror errors would make shadow mode look healthy while
    /// gathering no evidence at all.
    pub error: Option<String>,
}

impl AppState {
    /// New state with a default record cap.
    pub fn new(router: Router, client: reqwest::Client) -> Self {
        Self {
            router,
            client,
            records: Mutex::new(Vec::new()),
            mirrors: Mutex::new(Vec::new()),
            max_records: 10_000,
            capture: None,
        }
    }

    /// Persist redacted traffic to `store`.
    ///
    /// Off unless called. See [`crate::capture::store`] for what encryption here
    /// does and does not protect against.
    #[must_use]
    pub fn with_capture(mut self, store: Arc<CaptureStore>) -> Self {
        self.capture = Some(store);
        self
    }

    /// Whether traffic is being persisted.
    pub fn capturing(&self) -> bool {
        self.capture.is_some()
    }

    /// Write one exchange, off the request path.
    ///
    /// Spawned rather than awaited for the same reason the mirror is: capture is
    /// evidence-gathering, and a slow disk must never become a slow response. A
    /// failed write is logged and dropped — refusing to serve because we could
    /// not record would trade an outage for a missing sample.
    fn capture(&self, c: Capture) {
        let Some(store) = self.capture.clone() else {
            return;
        };
        tokio::spawn(async move {
            match tokio::task::spawn_blocking(move || store.append(c)).await {
                Ok(Ok(report)) => tracing::debug!(redacted = report.total(), "capture stored"),
                Ok(Err(e)) => tracing::warn!(error = %e, "capture not stored"),
                Err(e) => tracing::warn!(error = %e, "capture task failed"),
            }
        });
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

    /// Record a completed mirror, dropping the oldest once the cap is reached.
    pub fn record_mirror(&self, m: MirrorRecord) {
        let mut g = self.mirrors.lock();
        if g.len() >= self.max_records {
            g.remove(0);
        }
        tracing::info!(
            backend = %m.backend,
            status = m.status.unwrap_or(0),
            duration_ms = m.duration_ms,
            error = m.error.as_deref().unwrap_or(""),
            "mirror complete"
        );
        g.push(m);
    }

    /// Snapshot of retained mirror outcomes.
    pub fn mirrors(&self) -> Vec<MirrorRecord> {
        self.mirrors.lock().clone()
    }
}

/// Dispatch a copy of the request to the candidate backend.
///
/// Spawned, never awaited by the request handler: the client's latency must not
/// depend on the candidate, and the candidate's response must never be served.
/// This is the mechanism shadow mode *is* — computing a mirror decision and
/// recording it without sending anything would make the safety story a display.
fn dispatch_mirror(
    st: Arc<AppState>,
    backend: crate::router::Backend,
    parsed: serde_json::Value,
    headers: HeaderMap,
    body: Bytes,
    model: String,
) {
    tokio::spawn(async move {
        let started = Instant::now();
        let decision = Decision {
            serve: backend.clone(),
            mirror: None,
            // A mirror is evidence-gathering, not service. If the candidate is
            // unreachable that IS the finding; falling back would hide it.
            fallback: None,
            reason: "mirror",
        };
        let mut rec = MirrorRecord {
            backend: backend.name.clone(),
            model,
            status: None,
            duration_ms: 0,
            metered: Metered::Unreported { deltas: 0 },
            error: None,
        };

        match send(&st, &decision.serve, &parsed, &headers, &body).await {
            Ok(resp) => {
                rec.status = Some(resp.status().as_u16());
                let mut meter = Meter::new();
                let mut decoder = Decoder::new();
                let mut stream = resp.bytes_stream();
                // Drain the candidate fully: a partially-read response gives a
                // partial token count, which would understate the comparison.
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(b) => {
                            for ev in decoder.push(&b) {
                                meter.observe(&ev);
                            }
                            meter.observe_body(&b);
                        }
                        Err(e) => {
                            rec.error = Some(e.to_string());
                            break;
                        }
                    }
                }
                rec.metered = meter.finish();
            }
            Err(e) => rec.error = Some(e.to_string()),
        }
        rec.duration_ms = elapsed_ms(started);
        st.record_mirror(rec);
    });
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
        .route("/metrics/mirrors", get(mirrors))
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

async fn mirrors(State(st): State<Arc<AppState>>) -> Json<Vec<MirrorRecord>> {
    Json(st.mirrors())
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

    // Dispatch the mirror BEFORE awaiting the served backend, so shadow traffic is
    // not serialised behind the client's response.
    if let Some(candidate) = decision.mirror.clone() {
        dispatch_mirror(
            Arc::clone(&st),
            candidate,
            parsed.clone(),
            headers.clone(),
            body.clone(),
            model.clone(),
        );
    }

    let mut served_by = decision.serve.name.clone();
    let mut failed_over = false;
    let upstream = match send(&st, &decision.serve, &parsed, &headers, &body).await {
        Ok(r) => r,
        Err(primary_err) => match &decision.fallback {
            // Only a transport failure falls over. A 4xx is bad at both backends,
            // and retrying it just bills the request twice.
            Some(alt) => {
                tracing::warn!(
                    primary = %decision.serve.name,
                    fallback = %alt.name,
                    error = %primary_err,
                    "primary unreachable, falling over"
                );
                failed_over = true;
                served_by = alt.name.clone();
                send(&st, alt, &parsed, &headers, &body).await?
            }
            None => return Err(primary_err),
        },
    };
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
        HeaderValue::from_str(&served_by).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );

    // Everything capture needs, gathered before `base` takes ownership of `model`.
    // `None` when capture is off, so a proxy that is not recording pays nothing
    // for the machinery — not a clone, not an allocation.
    let pending_capture = st.capturing().then(|| Capture {
        request_id: headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned(),
        model: model.clone(),
        backend: served_by.clone(),
        messages: parsed
            .get("messages")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        response: String::new(),
        prompt_tokens: None,
        completion_tokens: None,
        duration_ms: 0,
        redacted: crate::capture::Report::default(),
    });

    let base = Record {
        backend: served_by.clone(),
        failed_over,
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
        let metered = meter.finish();
        if let Some(mut c) = pending_capture {
            c.response = store::body_text(&bytes);
            c.duration_ms = elapsed_ms(started);
            (c.prompt_tokens, c.completion_tokens) = tokens(&metered);
            st.capture(c);
        }
        st.record(Record {
            duration_ms: elapsed_ms(started),
            metered,
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
    // Only allocated when capture is on: reassembling the response costs memory
    // proportional to its length, which a proxy that is not recording should not
    // pay. `None` here means the deltas are observed for metering and discarded.
    let transcript = pending_capture
        .is_some()
        .then(|| Arc::new(Mutex::new(String::new())));
    let transcript_for_stream = transcript.clone();
    let mut decoder = Decoder::new();
    let stream = upstream.bytes_stream().map(move |chunk| match chunk {
        Ok(bytes) => {
            let events = decoder.push(&bytes);
            if !events.is_empty() {
                let mut m = meter_for_stream.lock();
                for ev in &events {
                    m.observe(ev);
                }
                if let Some(t) = &transcript_for_stream {
                    let mut t = t.lock();
                    for ev in &events {
                        if let Some(s) = store::delta_text(ev) {
                            t.push_str(&s);
                        }
                    }
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
        capture: Mutex::new(pending_capture),
        transcript,
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
/// record must be written on drop rather than only on clean completion. The same
/// applies to the capture: a truncated answer is still evidence of what the
/// backend produced, and silently discarding it would bias the eval set toward
/// requests that happened to finish.
struct FinishOnDrop {
    state: Arc<AppState>,
    record: Mutex<Option<Record>>,
    meter: Arc<Mutex<Meter>>,
    started: Instant,
    capture: Mutex<Option<Capture>>,
    transcript: Option<Arc<Mutex<String>>>,
}

impl Drop for FinishOnDrop {
    fn drop(&mut self) {
        let metered = self.meter.lock().snapshot();
        if let Some(mut r) = self.record.lock().take() {
            r.duration_ms = elapsed_ms(self.started);
            r.metered = metered;
            self.state.record(r);
        }
        if let Some(mut c) = self.capture.lock().take() {
            if let Some(t) = &self.transcript {
                c.response = t.lock().clone();
            }
            c.duration_ms = elapsed_ms(self.started);
            (c.prompt_tokens, c.completion_tokens) = tokens(&metered);
            self.state.capture(c);
        }
    }
}

/// Split reported usage into the two counts a capture records.
///
/// `None` when the upstream reported nothing — the same rule the meter follows.
/// A zero here would read as "this request was free", which is the one direction
/// a cost figure must never err in.
fn tokens(m: &Metered) -> (Option<u64>, Option<u64>) {
    m.usage().map_or((None, None), |u| {
        (Some(u.prompt_tokens), Some(u.completion_tokens))
    })
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

async fn send(
    st: &AppState,
    backend: &crate::router::Backend,
    parsed: &serde_json::Value,
    headers: &HeaderMap,
    body: &Bytes,
) -> Result<reqwest::Response, ProxyError> {
    // Rewrite the model only when the backend names a different one.
    let payload = match &backend.model {
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
            backend.base_url.trim_end_matches('/')
        ))
        .header(header::CONTENT_TYPE, "application/json")
        .body(payload);

    // Forward authorisation untouched: credentials belong to the caller and this
    // process never stores or inspects them.
    if let Some(auth) = headers.get(header::AUTHORIZATION) {
        req = req.header(header::AUTHORIZATION, auth.clone());
    }

    req.send().await.map_err(|e| ProxyError::Upstream {
        backend: backend.name.clone(),
        source: e,
    })
}
