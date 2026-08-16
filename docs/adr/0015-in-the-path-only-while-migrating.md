# ADR-0015 — In the request path only while migrating

**Status:** accepted · **Date:** 2026-08-10

## Context

The README's *What it is not* table says:

> **Not a router or a proxy.** Nothing sits in your request path. onpar sizes,
> configures, launches and measures; where the traffic goes afterwards is your
> call and your infrastructure. A proxy that must be up for your app to work is
> a liability you did not have before.

`onpar-gateway` is 3,461 lines of Rust that sits in the request path. It has
a proxy, a router, a token meter, an SSE inspector and a capture store, with 26
tests including real-TCP streaming ones. Roadmap Phase 1 ships it as "a drop-in
proxy". The crate's own docstring argues the distinction:

> The migration router is not a load balancer. Once traffic is fully cut over,
> balancing is handed to GAIE/llm-d and we leave the datapath.

So the code has already reconciled this and the customer-facing promise has not.
Today that is survivable, because nothing runs the gateway — there is no
`[[bin]]` and no CLI command. The moment `onpar observe` ships, the README is
false.

This ADR exists because the honest fix is not to soften the README until it
stops contradicting the code. It is to state the boundary that makes both true,
and to say what would prove it wrong.

## Decision

**onpar sits in the request path only while a migration is in flight, and its
exit is a feature rather than an afterthought.**

Three things follow, and they are what separate this from the load balancer
CLAUDE.md's invariant 1 forbids:

1. **It does something no balancer does.** It mirrors traffic to a candidate
   that is *scored but never served*; it splits by percentage
   *deterministically*, so a retry lands the same way; and it applies
   *per-cluster* policy, so the tasks where the open model loses keep going to
   the incumbent. Those exist to answer "is the candidate good enough", not to
   spread load.

2. **The terminal state is leaving.** `Phase::Cut` is 100% candidate — the point
   at which there is nothing left to compare and no reason to be in the path.
   Balancing is handed to GAIE/llm-d. A migration router that never reaches
   `Cut` has failed at its job, not succeeded at being infrastructure.

3. **It is opt-in and removable.** `onpar fit`, `prove`, `where` and the
   receipt path never touch the datapath. A user who only wants to know what
   fits never starts it, and one who finishes a migration stops it.

The README row is corrected to say this rather than to claim the datapath is
never entered.

## Consequences

**The claim now carries an obligation.** "We leave" is only true if leaving is
implemented and exercised. `Phase::Cut` must be reachable end to end and there
must be a test that a cut-over deployment runs with the gateway stopped —
otherwise this ADR is a sentence rather than a boundary.

**NFR-1 becomes load-bearing rather than aspirational.** A thing in the request
path is a thing that can take your service down. The <15ms added p95 has to be
*measured* against a real upstream over real TCP, and a build that cannot hold
it is a finding to report, not a number to relax.

**NFR-3 is the reason the capture store is shaped the way it is.** Redaction
runs inside the write path, so there is no code path that appends a record which
skipped it. That is structural rather than conventional, and it must stay that
way: the day redaction becomes a step a caller can forget, this decision stops
being defensible.

**CLAUDE.md invariant 1 is unchanged.** "No production load balancer" still
holds and this does not weaken it. The difference is not size or quality, it is
purpose and duration: a balancer's job is to stay, and this one's job is to
finish.

## What would falsify this

A user running the gateway permanently *because it became their load balancer* —
not because a migration is still in progress. If that becomes the common
deployment, the boundary was rhetorical and invariant 1 was breached in
practice while being honoured in wording.

The observable version: a `Phase::Cut` deployment that keeps the gateway in the
path with no candidate under evaluation. Worth instrumenting once there are real
users, and worth stating now so nobody has to reconstruct the intent later.

## Alternatives rejected

**Leave the README as it is and never ship `observe`.** Phase 1 is an on-ramp
the roadmap describes as "useful even if you never switch" — the cost dashboard
alone. Dropping it because a table row is awkward is the wrong trade.

**Soften the row to "not *primarily* a proxy".** Weasel wording. A reader
deciding whether to put this in front of production traffic deserves a straight
answer, and "primarily" is not one.

**Capture from logs rather than from the path.** Considered and genuinely
attractive, since it avoids the datapath entirely. It fails on redaction: log
formats vary per provider and per SDK, so redaction becomes best-effort parsing
of somebody else's shape — the opposite of a guarantee that unredacted text
never touches disk.
