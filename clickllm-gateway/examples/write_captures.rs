//! Writes a capture log with the real store, for the Python seam tests.
//!
//! A fixture assembled in Python would only prove the binding agrees with a
//! format Python invented. These bytes come from the code that runs in the
//! gateway.

#![allow(clippy::expect_used)]

use clickllm_gateway::capture::store::{Capture, CaptureStore};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: write_captures <path>");
    let s = CaptureStore::open(&path, &[7u8; 32]).expect("open");

    s.append(Capture {
        request_id: "req-a".into(),
        model: "gpt-5".into(),
        backend: "incumbent".into(),
        messages: serde_json::json!([
            {"role": "user", "content": "email ada@example.com the invoice"}
        ]),
        response: "Sent to ada@example.com".into(),
        prompt_tokens: Some(11),
        completion_tokens: Some(4),
        duration_ms: 42,
        tools: serde_json::Value::Null,
        tool_calls: Vec::new(),
        response_format: None,
        redacted: clickllm_gateway::capture::Report::default(),
    })
    .expect("append a");

    // Unreported usage, so the seam is exercised on the case that matters:
    // `None` must survive as `None` and never arrive in Python as 0.
    s.append(Capture {
        request_id: "req-b".into(),
        model: "gpt-5".into(),
        backend: "candidate".into(),
        messages: serde_json::json!([{"role": "user", "content": "hello"}]),
        response: "hi".into(),
        prompt_tokens: None,
        completion_tokens: None,
        duration_ms: 7,
        tools: serde_json::Value::Null,
        tool_calls: Vec::new(),
        response_format: None,
        redacted: clickllm_gateway::capture::Report::default(),
    })
    .expect("append b");
}
