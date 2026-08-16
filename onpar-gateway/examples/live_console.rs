//! Run the gateway against two backends so the console can be looked at.
//!
//! Not a test fixture — a way to see the live telemetry panel behave, including
//! the case that matters most: a backend that publishes no metrics at all must
//! render as "no telemetry" rather than as a healthy engine at 0%.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::sync::Arc;

use onpar_gateway::proxy::{AppState, app};
use onpar_gateway::router::{Backend, Phase, Route, Router};

#[tokio::main]
async fn main() {
    let backend = |name: &str, port: u16| Backend {
        name: name.into(),
        base_url: format!("http://127.0.0.1:{port}"),
        model: None,
    };
    let state = Arc::new(AppState::new(
        Router::new(Route {
            phase: Phase::Shadow,
            incumbent: backend("candidate-local", 9101),
            candidate: backend("silent-engine", 9102),
            failover: false,
        }),
        reqwest::Client::new(),
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:9100")
        .await
        .expect("bind 9100");
    println!("console on http://127.0.0.1:9100/");
    axum::serve(listener, app(state)).await.expect("serve");
}
