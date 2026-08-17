# ADR-0019 — Local stderr logging is not egress, and silence is not a security property

**Status:** accepted · 2026-08-16

## Context

`onpar-gateway` calls `tracing_subscriber_init()` on startup. The function had an
empty body, with this justification in the code:

> Deliberately minimal: no subscriber crate, no config file, no exporter.
> Zero telemetry and zero egress by default (NFR-2) is easier to guarantee
> when there is nothing here that could send anything anywhere.

The instinct is right and the conclusion does not follow. **NFR-2 governs data
leaving the machine** — no background sync, no exporter, no collector, export only
as an explicit command. An operator reading their own process's stderr is not
egress; nothing crosses a trust boundary, and nothing is written that the operator
could not already see by other means.

What the empty body bought was not a stronger NFR-2. It was silence. Nine call
sites in `onpar-gateway` were discarded at runtime:

| Site | Message |
|---|---|
| `capture/redact.rs:123` | `error!("redaction pattern failed to compile")` |
| `proxy.rs:145` | `warn!("capture not stored")` |
| `proxy.rs:146` | `warn!("capture task failed")` |
| `proxy.rs:477`, `:620` | upstream and streaming errors |
| `capture/store.rs:316` | `warn!("capture log ends mid-record")` |
| `control.rs:201`, `sse.rs:101` | control-plane and SSE failures |

The first is the serious one. A redaction pattern failing to compile is an
**NFR-3 event** — the fail-closed guarantee reporting that it fired. It fired to
nobody. The middle two mean captured evidence was dropped and the operator had no
way to know: `onpar prove` would later score a model on a corpus quietly missing
whatever failed to store, and nothing anywhere would say so.

This is the failure shape this repo keeps rediscovering, in its runtime rather
than its CI: **a green signal over something that didn't happen.** A gateway that
looks healthy while silently dropping the evidence it exists to collect is worse
than one that crashes, because the report it eventually produces still looks
authoritative.

`onpar-core`'s convention that "every fallible operation runs inside a `tracing`
span carrying model/runtime/path" also produced nothing, for the same reason.

## Decision

**Install a local stderr subscriber. Keep every other part of NFR-2 exactly as it
was.**

```rust
let filter = tracing_subscriber::EnvFilter::try_from_env("ONPAR_LOG")
    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
let _ = tracing_subscriber::fmt()
    .with_env_filter(filter)
    .with_writer(std::io::stderr)
    .try_init();
```

Specifically:

- **stderr only.** No exporter, no collector, no OTLP, no file, no network. There
  is nothing in this construction that can send anything anywhere.
- **Default `warn`.** A normal run stays quiet; only trouble prints. `ONPAR_LOG`
  raises or lowers it (`ONPAR_LOG=debug`, `ONPAR_LOG=onpar_gateway=trace`).
- **`try_init`, not `init`.** A second call is a no-op rather than a panic, so an
  embedder that installed its own subscriber first is not aborted by ours. A
  logging setup that can kill the process is worse than no logging.
- `tracing-subscriber` moves from `[dev-dependencies]` to `[dependencies]`; it was
  a dev-dep, so the shipped binary could not have used it even had the body been
  written.

## Consequences

- Capture and redaction failures are now visible to the operator at the moment
  they happen, rather than inferred later from a corpus that is quietly short.
- The gateway gains a runtime dependency it did not have. It is compiled in
  regardless of whether any log line is ever emitted.
- Default `warn` means a healthy run's output is unchanged, so this does not
  become noise in the common case.
- **NFR-2 is unchanged and still auditable**: the test for it should assert no
  network sink is configured, not that no logging exists.
- Prompt text must never reach a log line. Nothing currently logs request bodies,
  and nothing should start: the redaction boundary is about persistence *and*
  display. A future `debug!` that dumps a request would violate NFR-3 by a route
  this ADR opens, and that is the risk it introduces.

## What would falsify this

- If a log line is found to contain unredacted prompt text, the default level or
  the offending site is wrong — not this decision, but its implementation.
- If operators report the default `warn` output is noisy in normal running, the
  level or the specific sites need revisiting.
- If a future requirement genuinely needs remote log shipping, that is a separate
  decision and a separate ADR; it does not follow from this one.

## Alternatives rejected

- **Keep the empty body.** Rejected: it does not strengthen NFR-2, and it costs
  the operator every signal on the evidence path. Silence is not a security
  property.
- **Delete the nine `warn!`/`error!` sites and the `tracing` dependency.** This
  was the honest alternative to doing nothing — dead observability code that
  implies failures are reported when they are not is worse than none. Rejected
  because the sites are the right sites; what was missing was the subscriber.
- **A file-based or structured JSON log.** Rejected as more machinery than the
  problem needs, and a file is closer to "data at rest the operator did not ask
  for" than stderr is.
- **`init()` instead of `try_init()`.** Rejected: it panics on a second call, so
  a library consumer with their own subscriber would abort the process.
