//! `onpar-gateway` — the binary that runs the datapath.
//!
//! Everything here is wiring. The proxy, router, capture store, meter and SSE
//! inspector are the library's; this binds a port, assembles them from flags,
//! and says clearly what it is about to do. Nothing that costs latency belongs
//! in this file — see the crate docs for the <15ms budget (NFR-1).
//!
//! ## Two refusals, before any traffic moves
//!
//! **No capture without a key.** If the key cannot be created or read, the
//! process exits rather than serving with capture silently disabled. A gateway
//! that proxies happily while recording nothing looks identical to one that is
//! working, and the whole reason to be in the path is the recording.
//!
//! **No upstream you did not name.** There is no default endpoint and no
//! environment variable consulted for one. Egress is exactly where you point it
//! (NFR-2), and a tool that guesses an upstream is a tool that will eventually
//! guess wrong in the direction of somebody else's server.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use onpar_gateway::capture::store::CaptureStore;
use onpar_gateway::{AppState, Backend, Phase, Route, Router, app};

/// What the flags parsed into. Deliberately small — the library owns the model.
#[cfg_attr(test, derive(Debug))]
struct Args {
    port: u16,
    incumbent: String,
    incumbent_model: Option<String>,
    candidate: Option<String>,
    candidate_model: Option<String>,
    capture: Option<PathBuf>,
    key: Option<PathBuf>,
    percent: u8,
}

const USAGE: &str = "\
onpar-gateway — record traffic on its way to the model you already use

USAGE:
    onpar-gateway --upstream <URL> [OPTIONS]

REQUIRED:
    --upstream <URL>        Where requests go today, e.g. https://api.openai.com/v1

OPTIONS:
    --port <N>              Listen port (default 8787)
    --upstream-model <NAME> Send this model name upstream instead of the client's
    --candidate <URL>       Open model to mirror to; scored, never served
    --candidate-model <NAME>
    --percent <0-100>       Serve this share from the candidate (default 0 = shadow)
    --capture <PATH>        Capture log (default ./onpar-captures.log)
    --key <PATH>            Encryption key (default alongside the capture log)
    --no-capture            Run as a plain proxy, recording nothing
    -h, --help

Capture is on by default. Redaction runs inside the write path, so unredacted
prompt text never reaches disk; a redaction failure drops the capture rather
than storing it.
";

fn parse() -> Result<Args, String> {
    parse_from(std::env::args().skip(1))
}

fn parse_from(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut a = Args {
        port: 8787,
        incumbent: String::new(),
        incumbent_model: None,
        candidate: None,
        candidate_model: None,
        capture: Some(PathBuf::from("onpar-captures.log")),
        key: None,
        percent: 0,
    };
    let mut it = args;
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--upstream" => a.incumbent = value()?,
            "--upstream-model" => a.incumbent_model = Some(value()?),
            "--candidate" => a.candidate = Some(value()?),
            "--candidate-model" => a.candidate_model = Some(value()?),
            "--capture" => a.capture = Some(PathBuf::from(value()?)),
            "--key" => a.key = Some(PathBuf::from(value()?)),
            "--no-capture" => a.capture = None,
            "--port" => a.port = value()?.parse().map_err(|_| "--port must be a number")?,
            "--percent" => {
                // Parsed as u8 directly: `as u8` on a u16 would silently turn
                // 256 into 0 — "serve none of it" from a flag that asked for
                // all of it, which is the wrong direction to be quiet about.
                a.percent = value()?.parse().map_err(|_| "--percent must be 0-100")?;
                if a.percent > 100 {
                    return Err("--percent must be 0-100".into());
                }
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if a.incumbent.is_empty() {
        return Err("--upstream is required; there is no default endpoint".into());
    }
    if a.percent > 0 && a.candidate.is_none() {
        return Err("--percent needs a --candidate to send that share to".into());
    }
    // Startup may not move traffic. `control.rs` refuses an unconfirmed
    // candidate-share increase — it records a reason, checks an admin token and
    // writes a transition — and `--percent 100` at launch reached `Phase::Cut`
    // with none of that. A launch flag is set by whatever composes the command,
    // which for a container spec or a systemd unit is config drift, and
    // "nothing authorises a cutover except shadow mode" has to hold against
    // config too or it holds against nobody.
    if a.percent > 0 {
        return Err(
            "--percent cannot move traffic at startup: a candidate is scored in shadow \
             first, and escalation goes through the control surface, which records a \
             reason and refuses an unconfirmed increase. Start in shadow (omit \
             --percent) and escalate there."
                .into(),
        );
    }
    Ok(a)
}

/// Which phase a candidate and a percentage imply.
///
/// `Off` with no candidate, not `Shadow`: shadow *mirrors*, and the mirror
/// target would be the incumbent — doubling every upstream call to record what
/// was already recorded, and paying for it twice on a metered API.
fn phase_for(candidate: Option<&str>, percent: u8) -> Phase {
    match (candidate, percent) {
        (None, _) => Phase::Off,
        (Some(_), 0) => Phase::Shadow,
        (Some(_), 100) => Phase::Cut,
        (Some(_), p) => Phase::Canary { percent: p },
    }
}

fn backend(name: &str, url: &str, model: Option<String>) -> Backend {
    Backend {
        name: name.into(),
        base_url: url.into(),
        model,
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber_init();
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    // Shadow unless asked otherwise: a candidate is scored before it is served,
    // and defaulting to any served share would move production traffic on a
    // flag nobody thought about.
    // `Off` with no candidate, not `Shadow`: shadow *mirrors*, and mirroring to
    // the incumbent-as-candidate would double every upstream call to record
    // something already recorded. `Off` is the honest description of "observing
    // only" and it is also the cheap one.
    let phase = phase_for(args.candidate.as_deref(), args.percent);
    let candidate = args.candidate.as_deref().map_or_else(
        || backend("incumbent", &args.incumbent, args.incumbent_model.clone()),
        |url| backend("candidate", url, args.candidate_model.clone()),
    );

    let route = Route {
        incumbent: backend("incumbent", &args.incumbent, args.incumbent_model.clone()),
        candidate,
        phase,
        failover: false,
    };

    // Described before the route is consumed by the router.
    let phase_desc = describe(&route.phase);
    let mut state = AppState::new(Router::new(route), reqwest::Client::new());

    if let Some(log) = args.capture.clone() {
        let key_path = args.key.unwrap_or_else(|| log.with_extension("key"));
        // Fail closed. A gateway that proxies while recording nothing is
        // indistinguishable from one that is working, and recording is the
        // entire reason to be in the request path.
        let key = match CaptureStore::load_or_create_key(&key_path) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("error: cannot create or read the capture key at {key_path:?}: {e}");
                eprintln!("       pass --no-capture to run as a plain proxy instead.");
                std::process::exit(2);
            }
        };
        // `open` builds a cipher and touches no filesystem; `append` opens
        // lazily, from a spawned task whose errors are logged and dropped. So
        // `open` succeeding proved nothing — a directory as the log path
        // started, served, and recorded silently nothing. `ready()` performs
        // the same open `append` will, now, before anything binds.
        let store = match CaptureStore::open(&log, &key) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot open the capture log at {log:?}: {e}");
                std::process::exit(2);
            }
        };
        if let Err(e) = store.ready() {
            eprintln!("error: the capture log at {log:?} is not writable: {e}");
            eprintln!("       pass --no-capture to run as a plain proxy instead.");
            std::process::exit(2);
        }
        state = state.with_capture(Arc::new(store));
        println!(
            "  capture   {} (key: {})",
            log.display(),
            key_path.display()
        );
    } else {
        println!("  capture   off (--no-capture)");
    }

    println!("  upstream  {}", args.incumbent);
    if let Some(c) = args.candidate.as_deref() {
        println!("  candidate {c}  [{phase_desc}]");
    }
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    // The base_url must include `/v1`, because that is the only route served
    // (proxy.rs registers `/v1/chat/completions`). Printing the bare address
    // under "point your base_url here and nothing else changes" sent every
    // request to a 404: found by running the chain end to end and getting four
    // 404s from an invocation that followed the banner exactly.
    println!(
        "  listening http://{addr}\n  point your base_url at http://{addr}/v1 and nothing else changes.\n"
    );

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot bind {addr}: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = axum::serve(listener, app(Arc::new(state))).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn describe(p: &Phase) -> String {
    match *p {
        Phase::Off => "off — incumbent only, nothing mirrored".into(),
        Phase::Shadow => "shadow — mirrored and scored, never served".into(),
        Phase::Canary { percent } => format!("canary {percent}% — a human moved it here"),
        Phase::Cut => "cut over — candidate only".into(),
        Phase::Split { to_candidate } => {
            let who = if to_candidate {
                "candidate"
            } else {
                "incumbent"
            };
            format!("split — permanently on the {who}, not a migration step")
        }
    }
}

/// Logging to stderr, off by default so the datapath is quiet unless asked.
fn tracing_subscriber_init() {
    // Local stderr only. No exporter, no collector, no network: NFR-2 is about
    // data leaving the machine, and an operator reading their own stderr is not
    // egress. This function used to be empty on that reasoning, which silenced
    // every warn!/error! in the gateway — including `capture not stored`,
    // `capture task failed`, and `redaction pattern failed to compile`. The last
    // is an NFR-3 event: the fail-closed guarantee reporting that it fired, to
    // nobody. See ADR-0019.
    //
    // Default WARN, so a normal run is quiet and only trouble prints. ONPAR_LOG
    // raises or lowers it (`ONPAR_LOG=debug`, `ONPAR_LOG=onpar_gateway=trace`).
    //
    // try_init, not init: a second call must not panic. The binary calls this
    // once, but tests and embedders may install their own subscriber first, and
    // a logging setup that can abort the process is worse than no logging.
    let filter = tracing_subscriber::EnvFilter::try_from_env("ONPAR_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    // The subscriber is what makes every warn!/error! in this crate reach a
    // human. It was an empty function for a long time, so the assertion that
    // matters is not "init does not panic" — an empty body passes that — but
    // that a dispatcher is actually installed afterwards.
    #[test]
    fn tracing_init_installs_a_dispatcher() {
        tracing_subscriber_init();
        assert!(
            tracing::dispatcher::has_been_set(),
            "no tracing dispatcher after init: every warn!/error! in this crate \
             is discarded, including the capture and redaction failures"
        );
    }

    // try_init, not init: a second call must be a no-op rather than a panic, so
    // an embedder that installs its own subscriber first does not abort us.
    #[test]
    fn tracing_init_is_idempotent() {
        tracing_subscriber_init();
        tracing_subscriber_init();
    }

    fn parse(args: &[&str]) -> Result<Args, String> {
        parse_from(args.iter().map(|s| (*s).to_owned()))
    }

    #[test]
    fn an_upstream_is_required_and_never_guessed() {
        // No default endpoint and no environment variable consulted for one.
        // A tool that guesses an upstream eventually guesses wrong, in the
        // direction of somebody else's server (NFR-2).
        let e = parse(&["--port", "9000"]).expect_err("should refuse");
        assert!(e.contains("--upstream is required"), "{e}");
    }

    #[test]
    fn a_percent_over_one_hundred_is_refused_rather_than_truncated() {
        // Parsed as u8 directly. `as u8` on a u16 turns 256 into 0 — "serve
        // none of it" from a flag that asked for all of it, which is the wrong
        // direction to be quiet about.
        for bad in ["101", "256", "1000"] {
            let e = parse(&[
                "--upstream",
                "http://x",
                "--candidate",
                "http://y",
                "--percent",
                bad,
            ])
            .expect_err("should refuse");
            assert!(e.contains("0-100"), "{bad}: {e}");
        }
    }

    #[test]
    fn a_share_with_nowhere_to_send_it_is_refused() {
        let e = parse(&["--upstream", "http://x", "--percent", "25"]).expect_err("should refuse");
        assert!(e.contains("--candidate"), "{e}");
    }

    #[test]
    fn capture_is_on_unless_declined() {
        // The whole reason to be in the request path is the recording, so it is
        // the default and turning it off has to be deliberate.
        assert!(
            parse(&["--upstream", "http://x"])
                .unwrap()
                .capture
                .is_some()
        );
        assert!(
            parse(&["--upstream", "http://x", "--no-capture"])
                .unwrap()
                .capture
                .is_none()
        );
    }

    #[test]
    fn an_unknown_flag_is_refused_rather_than_ignored() {
        // A silently ignored flag is a setting the operator believes is applied.
        let e = parse(&["--upstream", "http://x", "--capture-everything"]).expect_err("refuse");
        assert!(e.contains("unknown flag"), "{e}");
    }

    #[test]
    fn no_candidate_means_off_not_shadow() {
        // Shadow *mirrors*. With no candidate the mirror target would be the
        // incumbent, doubling every upstream call to record what was already
        // recorded — and paying for it twice on a metered API.
        assert_eq!(phase_for(None, 0), Phase::Off);
        assert_eq!(phase_for(Some("http://y"), 0), Phase::Shadow);
        assert_eq!(
            phase_for(Some("http://y"), 25),
            Phase::Canary { percent: 25 }
        );
        assert_eq!(phase_for(Some("http://y"), 100), Phase::Cut);
    }

    #[test]
    fn startup_cannot_move_traffic() {
        // `control.rs` refuses an unconfirmed candidate-share increase: it
        // records a reason, checks an admin token, writes a transition. A
        // launch flag had none of that and reached `Phase::Cut` directly.
        //
        // A flag is set by whatever composes the command — a container spec, a
        // systemd unit, a helm chart — so "nothing authorises a cutover except
        // shadow mode" has to hold against config drift or it holds against
        // nobody.
        for pct in ["1", "5", "50", "100"] {
            let e = parse(&[
                "--upstream",
                "http://x",
                "--candidate",
                "http://y",
                "--percent",
                pct,
            ])
            .expect_err("startup escalation must be refused");
            assert!(e.contains("control surface"), "{pct}: {e}");
        }
    }

    #[test]
    fn shadow_is_still_allowed_at_startup() {
        // The control that keeps the refusal honest: scoring a candidate
        // without serving it is the *point*, and a guard that blocked it would
        // make the gateway useless for the thing it exists to do.
        let a = parse(&["--upstream", "http://x", "--candidate", "http://y"]).unwrap();
        assert_eq!(phase_for(a.candidate.as_deref(), a.percent), Phase::Shadow);
    }

    #[test]
    fn every_phase_describes_itself() {
        // The startup banner is how an operator confirms what they just asked
        // for. A phase with no description would print an empty bracket.
        for p in [
            Phase::Off,
            Phase::Shadow,
            Phase::Canary { percent: 5 },
            Phase::Cut,
            Phase::Split { to_candidate: true },
        ] {
            assert!(!describe(&p).is_empty(), "{p:?}");
        }
    }
}
