//! Routing policy: which backend serves a request, and whether it is mirrored.
//!
//! The migration router is not a load balancer. Once traffic is fully cut over,
//! balancing is handed to GAIE/llm-d and we leave the datapath. What this does
//! that a balancer does not:
//!
//! - splits by **percentage of traffic** during a canary, deterministically, so
//!   the same request always lands the same way for a given rollout state;
//! - applies **per-cluster policy**, so the task clusters where the open model
//!   loses keep going to the incumbent (the regret set) while everything else
//!   moves;
//! - **mirrors** to a shadow backend that is scored but never served.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A place a request can be sent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Backend {
    /// Stable name used in logs, metrics and reports.
    pub name: String,
    /// Base URL, e.g. `https://api.openai.com/v1`.
    pub base_url: String,
    /// Model name to send upstream, when it differs from what the client asked for.
    pub model: Option<String>,
}

/// What the router decided for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    /// Backend whose response is returned to the client.
    pub serve: Backend,
    /// Backend to mirror to. Its response is scored and discarded.
    pub mirror: Option<Backend>,
    /// Backend to try if `serve` cannot be reached. Never used for a 4xx — a
    /// bad request is bad at both backends, and retrying it just bills twice.
    pub fallback: Option<Backend>,
    /// Why this route was chosen, for the request log.
    pub reason: &'static str,
}

/// Rollout state for one route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "phase")]
pub enum Phase {
    /// Everything to the incumbent. No candidate traffic at all.
    Off,
    /// Everything to the incumbent, mirrored to the candidate. The candidate's
    /// output is scored and thrown away — this is the only safe way to gather
    /// evidence, and it is what must precede any real traffic.
    Shadow,
    /// A percentage of traffic served by the candidate.
    Canary {
        /// Percent of requests to the candidate, 0–100.
        percent: u8,
    },
    /// Everything to the candidate.
    Cut,
    /// **Steady state, not a migration step.** This cluster is permanently
    /// assigned to one backend and stays there.
    ///
    /// The common shape is local-plus-cloud: a small open model handles
    /// classification, extraction and summarising, while a frontier model keeps
    /// the long-context reasoning it is genuinely better at. That is not a
    /// half-finished migration — it is the destination, and modelling it as
    /// `Cut` on some clusters and `Off` on others would say "unfinished" about
    /// a deliberate design.
    ///
    /// Split never mirrors: mirroring exists to gather evidence for a decision
    /// already made here.
    Split {
        /// Send this cluster to the candidate (typically local) rather than the
        /// incumbent (typically the frontier model).
        to_candidate: bool,
    },
}

impl Phase {
    /// Percentage of live traffic the candidate serves.
    pub fn candidate_percent(&self) -> u8 {
        match self {
            Self::Off | Self::Shadow => 0,
            Self::Canary { percent } => (*percent).min(100),
            Self::Cut => 100,
            Self::Split { to_candidate } => {
                if *to_candidate {
                    100
                } else {
                    0
                }
            }
        }
    }

    /// Whether this phase is part of a rollout, as opposed to a settled design.
    ///
    /// A dashboard should show progress for the first and none for the second —
    /// reporting "60% migrated" forever, when 60% is the intended end state, is
    /// how a finished project looks permanently unfinished.
    pub fn is_transitional(&self) -> bool {
        matches!(self, Self::Shadow | Self::Canary { .. })
    }
}

/// Policy for a single task cluster, or the default when no cluster matches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Route {
    /// Incumbent — the closed model being replaced.
    pub incumbent: Backend,
    /// Candidate — the open model replacing it.
    pub candidate: Backend,
    /// Where this route is in its rollout.
    pub phase: Phase,
    /// Fall back to the other backend when the chosen one cannot be reached.
    ///
    /// The reason this exists: a local model is a single machine. When it is
    /// down, saturated, or mid-restart, "the request fails" is a worse answer
    /// than "it cost a bit more this time". Off by default — silently sending
    /// traffic to a paid API is not something to do without being asked.
    #[serde(default)]
    pub failover: bool,
}

/// The full routing table.
#[derive(Debug, Clone)]
pub struct Router {
    default: Route,
    /// Per-cluster overrides. `BTreeMap` so iteration and any future serialisation
    /// are deterministic — a routing table that reorders itself is unauditable.
    by_cluster: BTreeMap<String, Route>,
}

impl Router {
    /// Router with a single default route and no cluster overrides.
    pub fn new(default: Route) -> Self {
        Self {
            default,
            by_cluster: BTreeMap::new(),
        }
    }

    /// Pin one task cluster to its own policy.
    ///
    /// This is how the regret set stays on the incumbent while everything else
    /// migrates: give those clusters `Phase::Off`.
    #[must_use]
    pub fn with_cluster(mut self, cluster: &str, route: Route) -> Self {
        self.by_cluster.insert(cluster.to_owned(), route);
        self
    }

    /// The route that applies to `cluster`.
    pub fn route_for(&self, cluster: Option<&str>) -> &Route {
        cluster
            .and_then(|c| self.by_cluster.get(c))
            .unwrap_or(&self.default)
    }

    /// Decide where one request goes.
    ///
    /// `key` is any stable per-request identifier (a request id, or a session id
    /// when session stickiness matters). Bucketing is deterministic in `key`, so
    /// a retry of the same request lands on the same backend — otherwise a
    /// canary comparison is measuring two different things.
    pub fn decide(&self, cluster: Option<&str>, key: &str) -> Decision {
        let route = self.route_for(cluster);

        // `serve` is chosen first; the fallback is always the *other* backend,
        // so failover works in both directions without extra configuration.
        let (serve, mirror, reason) = match route.phase {
            Phase::Off => (route.incumbent.clone(), None, "phase off — incumbent only"),
            Phase::Shadow => (
                route.incumbent.clone(),
                Some(route.candidate.clone()),
                "shadow — candidate scored, never served",
            ),
            Phase::Cut => (route.candidate.clone(), None, "cut over — candidate only"),
            Phase::Split { to_candidate } => {
                if to_candidate {
                    (
                        route.candidate.clone(),
                        None,
                        "split — this cluster is assigned to the candidate",
                    )
                } else {
                    (
                        route.incumbent.clone(),
                        None,
                        "split — this cluster is assigned to the incumbent",
                    )
                }
            }
            Phase::Canary { percent } => {
                if bucket(key) < u32::from(percent.min(100)) {
                    (
                        route.candidate.clone(),
                        None,
                        "canary — in the candidate bucket",
                    )
                } else {
                    (
                        route.incumbent.clone(),
                        Some(route.candidate.clone()),
                        "canary — incumbent, still mirroring for comparison",
                    )
                }
            }
        };

        let fallback = if route.failover {
            let other = if serve.name == route.candidate.name {
                route.incumbent.clone()
            } else {
                route.candidate.clone()
            };
            Some(other)
        } else {
            None
        };

        Decision {
            serve,
            mirror,
            fallback,
            reason,
        }
    }
}

/// Map a key uniformly onto `0..100`.
///
/// FNV-1a: small, fast, and — unlike `DefaultHasher` — stable across processes
/// and releases. A bucket function that changes between versions would silently
/// reshuffle every in-flight canary.
fn bucket(key: &str) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Fold to reduce the influence of any single byte position.
    let folded = (h >> 32) ^ (h & 0xffff_ffff);
    u32::try_from(folded % 100).unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn be(n: &str) -> Backend {
        Backend {
            name: n.to_owned(),
            base_url: format!("https://{n}.example/v1"),
            model: None,
        }
    }

    fn route(phase: Phase) -> Route {
        Route {
            incumbent: be("gpt-5"),
            candidate: be("glm-5.2"),
            phase,
            failover: false,
        }
    }

    fn route_with_failover(phase: Phase) -> Route {
        Route {
            failover: true,
            ..route(phase)
        }
    }

    #[test]
    fn off_serves_the_incumbent_and_mirrors_nothing() {
        let d = Router::new(route(Phase::Off)).decide(None, "k");
        assert_eq!(d.serve.name, "gpt-5");
        assert!(d.mirror.is_none());
    }

    #[test]
    fn shadow_never_serves_the_candidate() {
        // The invariant the whole safety story rests on.
        let r = Router::new(route(Phase::Shadow));
        for i in 0..1000 {
            let d = r.decide(None, &format!("req-{i}"));
            assert_eq!(d.serve.name, "gpt-5", "shadow served the candidate");
            assert_eq!(d.mirror.as_ref().unwrap().name, "glm-5.2");
        }
    }

    #[test]
    fn cut_serves_only_the_candidate() {
        let d = Router::new(route(Phase::Cut)).decide(None, "k");
        assert_eq!(d.serve.name, "glm-5.2");
        assert!(d.mirror.is_none());
    }

    #[test]
    fn canary_splits_close_to_the_requested_percentage() {
        for pct in [1_u8, 5, 25, 50, 75, 99] {
            let r = Router::new(route(Phase::Canary { percent: pct }));
            let n = 20_000;
            let hits = (0..n)
                .filter(|i| r.decide(None, &format!("req-{i}")).serve.name == "glm-5.2")
                .count();
            let actual = (hits as f64 / f64::from(n)) * 100.0;
            assert!(
                (actual - f64::from(pct)).abs() < 2.0,
                "asked for {pct}%, got {actual:.1}%"
            );
        }
    }

    #[test]
    fn bucketing_is_deterministic_so_a_retry_lands_the_same_way() {
        let r = Router::new(route(Phase::Canary { percent: 50 }));
        for i in 0..500 {
            let k = format!("req-{i}");
            let first = r.decide(None, &k);
            for _ in 0..5 {
                assert_eq!(r.decide(None, &k), first, "routing flapped for {k}");
            }
        }
    }

    #[test]
    fn canary_still_mirrors_the_requests_it_does_not_serve() {
        // Otherwise the comparison sample shrinks exactly as the rollout grows.
        let r = Router::new(route(Phase::Canary { percent: 10 }));
        let mut mirrored = 0;
        for i in 0..1000 {
            let d = r.decide(None, &format!("req-{i}"));
            if d.serve.name == "gpt-5" {
                assert!(d.mirror.is_some());
                mirrored += 1;
            }
        }
        assert!(
            mirrored > 800,
            "expected most traffic still on the incumbent"
        );
    }

    #[test]
    fn percentages_above_100_are_clamped_not_wrapped() {
        let r = Router::new(route(Phase::Canary { percent: 250 }));
        assert_eq!(r.route_for(None).phase.candidate_percent(), 100);
        assert_eq!(r.decide(None, "k").serve.name, "glm-5.2");
    }

    #[test]
    fn zero_percent_canary_sends_nothing_to_the_candidate() {
        let r = Router::new(route(Phase::Canary { percent: 0 }));
        for i in 0..2000 {
            assert_eq!(r.decide(None, &format!("r{i}")).serve.name, "gpt-5");
        }
    }

    #[test]
    fn a_regret_cluster_stays_on_the_incumbent_while_the_rest_cuts_over() {
        // The hybrid-policy outcome: move what is proven, keep what is not.
        let r = Router::new(route(Phase::Cut)).with_cluster("long-ctx-refactor", route(Phase::Off));
        assert_eq!(r.decide(Some("codegen"), "k").serve.name, "glm-5.2");
        assert_eq!(r.decide(Some("long-ctx-refactor"), "k").serve.name, "gpt-5");
        assert_eq!(r.decide(None, "k").serve.name, "glm-5.2");
    }

    #[test]
    fn unknown_clusters_fall_back_to_the_default_route() {
        let r = Router::new(route(Phase::Shadow)).with_cluster("x", route(Phase::Cut));
        assert_eq!(r.decide(Some("never-seen"), "k").serve.name, "gpt-5");
    }

    #[test]
    fn bucket_is_uniform_enough_and_within_range() {
        let mut counts = [0_u32; 10];
        for i in 0..100_000 {
            let b = bucket(&format!("key-{i}"));
            assert!(b < 100);
            // Integer division is the point here: bucket into deciles.
            #[allow(clippy::integer_division)]
            let decile = (b / 10) as usize;
            counts[decile] += 1;
        }
        for (i, c) in counts.iter().enumerate() {
            assert!(
                (8_000..12_000).contains(c),
                "decile {i} had {c} of 100000 — distribution is skewed"
            );
        }
    }

    #[test]
    fn split_is_a_destination_not_a_rollout_step() {
        // Local-plus-cloud is a design, not a half-finished migration. A
        // dashboard reporting "60% migrated" forever, when 60% is the intended
        // end state, makes a finished project look permanently unfinished.
        for to_candidate in [true, false] {
            let p = Phase::Split { to_candidate };
            assert!(!p.is_transitional());
        }
        assert!(Phase::Shadow.is_transitional());
        assert!(Phase::Canary { percent: 5 }.is_transitional());
        assert!(!Phase::Cut.is_transitional());
    }

    #[test]
    fn split_assigns_a_cluster_permanently_and_never_mirrors() {
        let local = Router::new(route(Phase::Split { to_candidate: true }));
        let cloud = Router::new(route(Phase::Split {
            to_candidate: false,
        }));
        for i in 0..200 {
            let k = format!("r{i}");
            let d = local.decide(None, &k);
            assert_eq!(d.serve.name, "glm-5.2");
            assert!(
                d.mirror.is_none(),
                "split gathers no evidence; the decision is made"
            );
            assert_eq!(cloud.decide(None, &k).serve.name, "gpt-5");
        }
    }

    #[test]
    fn a_local_plus_cloud_split_routes_each_cluster_to_its_own_backend() {
        // The shape people actually want: cheap local for the easy work, the
        // frontier model for what it is genuinely better at.
        let r = Router::new(route(Phase::Split { to_candidate: true })).with_cluster(
            "long-ctx-reasoning",
            route(Phase::Split {
                to_candidate: false,
            }),
        );
        assert_eq!(r.decide(Some("classify"), "k").serve.name, "glm-5.2");
        assert_eq!(r.decide(Some("summarise"), "k").serve.name, "glm-5.2");
        assert_eq!(
            r.decide(Some("long-ctx-reasoning"), "k").serve.name,
            "gpt-5"
        );
    }

    #[test]
    fn failover_is_off_unless_asked_for() {
        // Silently sending traffic to a paid API is not a default.
        assert!(
            Router::new(route(Phase::Cut))
                .decide(None, "k")
                .fallback
                .is_none()
        );
    }

    #[test]
    fn failover_names_the_other_backend_in_both_directions() {
        let to_local = Router::new(route_with_failover(Phase::Split { to_candidate: true }));
        let d = to_local.decide(None, "k");
        assert_eq!(d.serve.name, "glm-5.2");
        assert_eq!(d.fallback.as_ref().unwrap().name, "gpt-5");

        let to_cloud = Router::new(route_with_failover(Phase::Split {
            to_candidate: false,
        }));
        let d2 = to_cloud.decide(None, "k");
        assert_eq!(d2.serve.name, "gpt-5");
        assert_eq!(d2.fallback.as_ref().unwrap().name, "glm-5.2");
    }

    #[test]
    fn split_percentages_read_as_all_or_nothing() {
        assert_eq!(Phase::Split { to_candidate: true }.candidate_percent(), 100);
        assert_eq!(
            Phase::Split {
                to_candidate: false
            }
            .candidate_percent(),
            0
        );
    }

    #[test]
    fn every_decision_carries_a_reason() {
        let phases = [
            Phase::Off,
            Phase::Shadow,
            Phase::Cut,
            Phase::Canary { percent: 50 },
            Phase::Split { to_candidate: true },
            Phase::Split {
                to_candidate: false,
            },
        ];
        for p in phases {
            let d = Router::new(route(p)).decide(None, "k");
            assert!(!d.reason.is_empty());
        }
    }
}
