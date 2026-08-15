//! The binary's refusals, exercised by running the binary.
//!
//! `ready()` and `parse()` have unit tests, and they both passed while the
//! *call site* was missing — which is the same shape as the bug they were
//! written for: a guard that exists and is not on the path. So these run the
//! real process and read its exit code.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::Command;

/// The binary under test, as cargo built it beside this test.
fn bin() -> std::path::PathBuf {
    let mut p = std::env::current_exe().expect("test exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("onpar-gateway")
}

/// Run the binary and wait for it to exit — but only for a while.
///
/// Every case here asserts a *refusal*, so the failure mode under test is the
/// binary deciding to serve instead. `output()` would then block until the heat
/// death of the suite, and a test that hangs is worse than one that fails: CI
/// reports a timeout, not a diagnosis. So: poll, and on the deadline kill it and
/// report the thing that actually happened.
fn run(args: &[&str]) -> (i32, String) {
    use std::process::Stdio;
    let mut child = Command::new(bin())
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn gateway");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match child.try_wait().expect("wait") {
            Some(status) => {
                let out = child.wait_with_output().expect("output");
                let text = String::from_utf8_lossy(&out.stdout).to_string()
                    + &String::from_utf8_lossy(&out.stderr);
                return (status.code().unwrap_or(-1), text);
            }
            None if std::time::Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("still running after 10s — it served instead of refusing: {args:?}");
            }
            None => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    }
}

#[test]
fn it_refuses_to_serve_when_the_capture_log_cannot_be_written() {
    // The whole reason to be in the request path is the recording. A gateway
    // that proxies happily while recording nothing looks identical to one that
    // is working — and the append error is logged from a spawned task, so
    // nothing surfaces it.
    let dir = tempfile::tempdir().expect("tempdir");
    let occupied = dir.path().join("log-is-a-directory");
    std::fs::create_dir(&occupied).expect("mkdir");

    let (code, out) = run(&[
        "--upstream",
        "http://127.0.0.1:1",
        "--capture",
        occupied.to_str().expect("utf8"),
    ]);
    assert_eq!(code, 2, "should exit rather than serve: {out}");
    assert!(out.contains("not writable"), "{out}");
    assert!(!out.contains("listening"), "it bound a port anyway: {out}");
}

#[test]
fn it_refuses_to_move_traffic_at_startup() {
    // `control.rs` refuses an unconfirmed candidate-share increase — reason,
    // admin token, transition record. A launch flag had none of that.
    let (code, out) = run(&[
        "--upstream",
        "http://a",
        "--candidate",
        "http://b",
        "--percent",
        "100",
        "--no-capture",
    ]);
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("control surface"), "{out}");
    assert!(!out.contains("listening"), "{out}");
}

#[test]
fn it_refuses_an_upstream_it_would_have_to_invent() {
    let (code, out) = run(&["--no-capture"]);
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("--upstream is required"), "{out}");
}

#[test]
fn help_is_help_and_not_an_error() {
    // The control: a binary that exited 2 on everything would satisfy every
    // assertion above.
    let (code, out) = run(&["--help"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("--upstream"), "{out}");
}
