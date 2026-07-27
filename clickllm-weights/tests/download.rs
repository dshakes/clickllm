//! Resume and verification, over real TCP against a real range-supporting server.
//!
//! A mocked transport cannot prove resume works: the whole mechanism is a
//! `Range` header, a `206`, and appending to a file that already exists on disk.
//! These tests interrupt a transfer for real and finish it for real.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use clickllm_weights::fetch::{Download, fetch_file};
use clickllm_weights::{Cache, Error};
use sha2::{Digest, Sha256};

#[derive(Clone)]
struct Blob {
    bytes: Arc<Vec<u8>>,
    /// Requests served, so a test can prove a resume fetched less.
    served: Arc<AtomicUsize>,
    /// When false the server ignores Range and returns 200 with the whole body,
    /// which is exactly the behaviour that would silently corrupt a resume.
    honour_range: bool,
}

async fn serve_blob(State(b): State<Blob>, headers: HeaderMap) -> Response {
    b.served.fetch_add(1, Ordering::SeqCst);
    let total = b.bytes.len();

    let start = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("bytes="))
        .and_then(|v| v.split('-').next())
        .and_then(|v| v.parse::<usize>().ok());

    match start {
        Some(s) if b.honour_range && s < total => (
            StatusCode::PARTIAL_CONTENT,
            [
                (
                    header::CONTENT_RANGE,
                    format!("bytes {s}-{}/{total}", total - 1),
                ),
                (header::ACCEPT_RANGES, "bytes".into()),
            ],
            Body::from(b.bytes[s..].to_vec()),
        )
            .into_response(),
        // No range asked for, or a server that refuses to honour it.
        _ => (StatusCode::OK, Body::from(b.bytes.as_ref().clone())).into_response(),
    }
}

async fn spawn(blob: Blob) -> SocketAddr {
    let app = Router::new()
        .route("/model.bin", get(serve_blob))
        .with_state(blob);
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(l, app).await;
    });
    addr
}

fn payload(n: usize) -> Vec<u8> {
    // Non-uniform, so a truncated or duplicated region changes the digest.
    // The modulus keeps every value inside u8 by construction.
    (0..n).map(|i| u8::try_from(i % 251).unwrap_or(0)).collect()
}

fn sha(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

#[tokio::test]
async fn a_whole_file_downloads_and_verifies() {
    let data = payload(64 * 1024);
    let served = Arc::new(AtomicUsize::new(0));
    let addr = spawn(Blob {
        bytes: Arc::new(data.clone()),
        served: Arc::clone(&served),
        honour_range: true,
    })
    .await;

    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.bin");
    let got = fetch_file(
        &reqwest::Client::new(),
        &Download {
            url: format!("http://{addr}/model.bin"),
            dest: dest.clone(),
            expected_sha256: Some(sha(&data)),
        },
        |_| {},
    )
    .await
    .unwrap();

    assert_eq!(got.bytes, data.len() as u64);
    assert_eq!(got.resumed_from, 0);
    assert!(got.verified);
    assert_eq!(std::fs::read(&dest).unwrap(), data);
}

#[tokio::test]
async fn an_interrupted_download_resumes_and_still_hashes_correctly() {
    // The failure this guards: resuming without hashing the bytes already on
    // disk produces a digest of only the tail, which then "fails" verification
    // on a perfectly good file.
    let data = payload(200 * 1024);
    let served = Arc::new(AtomicUsize::new(0));
    let addr = spawn(Blob {
        bytes: Arc::new(data.clone()),
        served: Arc::clone(&served),
        honour_range: true,
    })
    .await;

    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.bin");

    // Simulate a transfer that died 3/4 of the way through. Integer division is
    // the intent — we want a byte offset, not a fraction.
    #[allow(clippy::integer_division)]
    let cut = data.len() * 3 / 4;
    std::fs::write(&dest, &data[..cut]).unwrap();

    let got = fetch_file(
        &reqwest::Client::new(),
        &Download {
            url: format!("http://{addr}/model.bin"),
            dest: dest.clone(),
            expected_sha256: Some(sha(&data)),
        },
        |_| {},
    )
    .await
    .unwrap();

    assert_eq!(
        got.resumed_from, cut as u64,
        "should have resumed, not restarted"
    );
    assert_eq!(got.bytes, data.len() as u64);
    assert_eq!(
        got.digest,
        sha(&data),
        "digest must cover the resumed bytes too"
    );
    assert_eq!(
        std::fs::read(&dest).unwrap(),
        data,
        "file must be byte-identical"
    );
}

#[tokio::test]
async fn a_server_that_ignores_range_is_refused_rather_than_corrupting_the_file() {
    // Appending a fresh copy of the whole body to a partial file produces a
    // corrupt result that verifies as "bad model" rather than "bad download".
    let data = payload(32 * 1024);
    let addr = spawn(Blob {
        bytes: Arc::new(data.clone()),
        served: Arc::new(AtomicUsize::new(0)),
        honour_range: false,
    })
    .await;

    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.bin");
    std::fs::write(&dest, &data[..1000]).unwrap();

    let err = fetch_file(
        &reqwest::Client::new(),
        &Download {
            url: format!("http://{addr}/model.bin"),
            dest: dest.clone(),
            expected_sha256: Some(sha(&data)),
        },
        |_| {},
    )
    .await
    .unwrap_err();

    assert!(matches!(err, Error::NoResume { .. }), "got {err:?}");
    assert_eq!(
        std::fs::read(&dest).unwrap().len(),
        1000,
        "the partial must be left untouched, not appended to"
    );
}

#[tokio::test]
async fn a_wrong_digest_fails_and_never_becomes_a_cache_entry() {
    let data = payload(16 * 1024);
    let addr = spawn(Blob {
        bytes: Arc::new(data.clone()),
        served: Arc::new(AtomicUsize::new(0)),
        honour_range: true,
    })
    .await;

    let dir = tempfile::tempdir().unwrap();
    let cache = Cache::open(dir.path().join("store"), None).unwrap();
    let r: clickllm_core::ModelRef = "hf:org/m#q4".parse().unwrap();
    let dest = cache.part_path(&r, "model.bin");

    let err = fetch_file(
        &reqwest::Client::new(),
        &Download {
            url: format!("http://{addr}/model.bin"),
            dest,
            expected_sha256: Some("0".repeat(64)),
        },
        |_| {},
    )
    .await
    .unwrap_err();

    assert!(matches!(err, Error::Checksum { .. }));
    assert!(!cache.contains(&r), "corrupt bytes must never be committed");
}

#[tokio::test]
async fn a_verified_download_commits_into_the_cache_once() {
    let data = payload(48 * 1024);
    let addr = spawn(Blob {
        bytes: Arc::new(data.clone()),
        served: Arc::new(AtomicUsize::new(0)),
        honour_range: true,
    })
    .await;

    let dir = tempfile::tempdir().unwrap();
    let mut cache = Cache::open(dir.path().join("store"), None).unwrap();
    let r: clickllm_core::ModelRef = "hf:org/m#Q4_K_M".parse().unwrap();

    let got = fetch_file(
        &reqwest::Client::new(),
        &Download {
            url: format!("http://{addr}/model.bin"),
            dest: cache.part_path(&r, "model.bin"),
            expected_sha256: Some(sha(&data)),
        },
        |_| {},
    )
    .await
    .unwrap();

    assert!(!cache.contains(&r), "not an entry until committed");
    cache
        .commit(&r, &got.digest, vec!["model.bin".into()])
        .unwrap();
    assert!(cache.contains(&r));

    // Canonicalisation means a differently-spelled quant is already satisfied.
    let same: clickllm_core::ModelRef = "hf:org/m#q4-k-m".parse().unwrap();
    assert!(
        cache.contains(&same),
        "must not pay for the same bytes twice"
    );
}

#[tokio::test]
async fn progress_is_reported_and_reaches_the_total() {
    let data = payload(96 * 1024);
    let addr = spawn(Blob {
        bytes: Arc::new(data.clone()),
        served: Arc::new(AtomicUsize::new(0)),
        honour_range: true,
    })
    .await;

    let dir = tempfile::tempdir().unwrap();
    let mut seen: Vec<u64> = Vec::new();
    let mut last_fraction = 0.0_f64;

    fetch_file(
        &reqwest::Client::new(),
        &Download {
            url: format!("http://{addr}/model.bin"),
            dest: dir.path().join("model.bin"),
            expected_sha256: None,
        },
        |p| {
            seen.push(p.done);
            if let Some(f) = p.fraction() {
                assert!(f >= last_fraction, "progress must not go backwards");
                last_fraction = f;
            }
        },
    )
    .await
    .unwrap();

    assert!(!seen.is_empty(), "progress must be reported");
    assert_eq!(*seen.last().unwrap(), data.len() as u64);
}

#[tokio::test]
async fn an_unpublished_digest_downloads_but_says_it_was_not_verified() {
    // Silence here would let a caller assume a check happened that did not.
    let data = payload(8 * 1024);
    let addr = spawn(Blob {
        bytes: Arc::new(data.clone()),
        served: Arc::new(AtomicUsize::new(0)),
        honour_range: true,
    })
    .await;

    let dir = tempfile::tempdir().unwrap();
    let got = fetch_file(
        &reqwest::Client::new(),
        &Download {
            url: format!("http://{addr}/model.bin"),
            dest: dir.path().join("model.bin"),
            expected_sha256: None,
        },
        |_| {},
    )
    .await
    .unwrap();

    assert!(!got.verified);
    assert_eq!(got.digest, sha(&data), "we still record what we got");
}

#[tokio::test]
async fn an_unreachable_source_reports_the_url() {
    let dir = tempfile::tempdir().unwrap();
    let err = fetch_file(
        &reqwest::Client::new(),
        &Download {
            url: "http://127.0.0.1:1/model.bin".into(), // reserved; nothing listens
            dest: dir.path().join("model.bin"),
            expected_sha256: None,
        },
        |_| {},
    )
    .await
    .unwrap_err();

    assert!(matches!(err, Error::Fetch { .. }));
    assert!(err.to_string().contains("127.0.0.1:1"));
}
