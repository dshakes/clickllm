//! Host stats — the accelerator as the *operating system* sees it.
//!
//! [`crate::telemetry`] reads what the engine reports about itself. This reads
//! what the machine reports about the engine, and the two disagree in a way that
//! is diagnostic rather than annoying:
//!
//! > The engine says the KV cache is 60% full. The host says the card is at 98%
//! > memory. The difference is *someone else's process* — a stray notebook, a
//! > second model, a training job — and no amount of engine telemetry will ever
//! > show it to you.
//!
//! That is the whole reason this file exists. Everything else it reports is
//! nice to have.
//!
//! ## Why this is deliberately thin
//!
//! Reading GPU state means shelling out, and every vendor spells it
//! differently. The temptation is to normalise hard — one `Gpu` struct, every
//! field populated, one number per vendor. That produces confident wrong
//! answers on the platforms that were not tested.
//!
//! So: **NVIDIA only, via `nvidia-smi`, and everything else reports why not.**
//!
//! - **AMD** would be `rocm-smi`, with different columns. Not implemented rather
//!   than guessed at.
//! - **Apple** has no per-process GPU memory API without `powermetrics`, which
//!   needs root. A tool that asked for `sudo` to draw a dashboard would deserve
//!   the refusal it got.
//! - **TPU** exposes utilisation through a separate metrics agent, not a local
//!   binary, so it is not a host-stats problem at all.
//!
//! Each of those returns [`Support::Unavailable`] carrying the reason, because
//! "we cannot see this here" and "this is idle" must never render alike.

use std::time::Duration;

use serde::Serialize;

/// How long `nvidia-smi` gets before we give up on it.
///
/// It can block for seconds when the driver is wedged — which is exactly when
/// someone is staring at the dashboard, so it must not hang the page.
const PROBE_TIMEOUT: Duration = Duration::from_millis(2000);

/// One accelerator, as the host sees it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Device {
    /// Index as the driver reports it.
    pub index: u32,
    /// Product name.
    pub name: String,
    /// Memory in use, MiB — **all processes**, not just ours. That is the point.
    pub memory_used_mib: u64,
    /// Total memory, MiB.
    pub memory_total_mib: u64,
    /// Utilisation percent, 0–100.
    pub utilisation_pct: u32,
    /// Degrees Celsius.
    pub temperature_c: Option<u32>,
    /// Watts drawn, when the card reports it.
    pub power_w: Option<f64>,
}

impl Device {
    /// Memory in use as a fraction, `0.0..=1.0`.
    pub fn memory_used(&self) -> Option<f64> {
        (self.memory_total_mib > 0)
            .then(|| self.memory_used_mib as f64 / self.memory_total_mib as f64)
    }
}

/// Host CPU and RAM, read from the kernel rather than a vendor tool.
///
/// Modest on purpose. The interesting failure is not "the box is busy" — it is
/// **the box is busy and the GPU is idle**, which means the accelerator is
/// waiting on the host: tokenisation, the sampler, or a data loader. That is a
/// real and commonly misdiagnosed shape, and it needs both halves to see.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct System {
    /// Logical CPUs.
    pub cpus: Option<usize>,
    /// 1-minute load average, where the platform exposes one.
    pub load_1m: Option<f64>,
    /// Load average per CPU — the figure that is comparable across machines.
    pub load_per_cpu: Option<f64>,
    /// Total RAM, MiB.
    pub memory_total_mib: Option<u64>,
    /// Available RAM, MiB.
    pub memory_available_mib: Option<u64>,
}

impl System {
    /// Fraction of RAM in use, when both figures are known.
    pub fn memory_used(&self) -> Option<f64> {
        let (t, a) = (self.memory_total_mib?, self.memory_available_mib?);
        (t > 0).then(|| (t.saturating_sub(a)) as f64 / t as f64)
    }

    /// Whether the host looks saturated enough to be starving the accelerator.
    ///
    /// `None` when load is unreadable — not `false`, which would read as "the
    /// host is fine" on a platform we simply could not measure.
    pub fn cpu_bound(&self) -> Option<bool> {
        self.load_per_cpu.map(|l| l >= 0.90)
    }
}

/// Read CPU count, load average and memory from the kernel.
///
/// Every field is independently optional: `/proc/meminfo` exists on Linux and
/// not macOS, load average is the reverse of that on Windows. A partial answer
/// is more useful than an all-or-nothing one.
pub fn system() -> System {
    let cpus = std::thread::available_parallelism().ok().map(Into::into);
    let load_1m = read_loadavg();
    let (memory_total_mib, memory_available_mib) = read_meminfo();
    System {
        cpus,
        load_1m,
        load_per_cpu: match (load_1m, cpus) {
            (Some(l), Some(c)) if c > 0 => Some(l / c as f64),
            _ => None,
        },
        memory_total_mib,
        memory_available_mib,
    }
}

/// 1-minute load average. Linux exposes it as a file; elsewhere we decline.
fn read_loadavg() -> Option<f64> {
    std::fs::read_to_string("/proc/loadavg")
        .ok()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Total and available RAM in MiB, from `/proc/meminfo`.
///
/// `MemAvailable` rather than `MemFree`: free memory on a busy Linux box is
/// near zero because the page cache holds the rest, and reporting that as
/// pressure would be alarming and wrong.
fn read_meminfo() -> (Option<u64>, Option<u64>) {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return (None, None);
    };
    let field = |key: &str| {
        text.lines()
            .find(|l| l.starts_with(key))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()
            // KiB → MiB. Truncation is the intent: a fractional MiB in a
            // capacity readout is noise, not precision.
            .map(|kib| {
                #[allow(clippy::integer_division)]
                {
                    kib / 1024
                }
            })
    };
    (field("MemTotal:"), field("MemAvailable:"))
}

/// Whether host stats could be read at all.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum Support {
    /// Devices were read.
    Available {
        /// One entry per accelerator.
        devices: Vec<Device>,
    },
    /// Nothing could be read, and why.
    ///
    /// Never collapsed into an empty device list: "no GPUs found" and "we have
    /// no way to look on this platform" are different facts, and a dashboard
    /// that renders both as a blank panel teaches people to ignore it.
    Unavailable {
        /// One sentence, aimed at someone deciding whether to worry.
        reason: String,
    },
}

impl Support {
    /// Devices, or an empty slice when unavailable.
    pub fn devices(&self) -> &[Device] {
        match self {
            Self::Available { devices } => devices,
            Self::Unavailable { .. } => &[],
        }
    }

    /// The most-used device's memory fraction, when any were read.
    pub fn peak_memory_used(&self) -> Option<f64> {
        self.devices()
            .iter()
            .filter_map(Device::memory_used)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Compare host memory against what the engine claims of its KV cache.
    ///
    /// The one thing engine telemetry structurally cannot tell you. If the card
    /// is nearly full while the engine's own cache is not, the remainder belongs
    /// to something else on the box.
    ///
    /// Whether the host is saturated while the accelerator is not.
    ///
    /// The misdiagnosis this exists to prevent: a GPU sitting at 20% while the
    /// CPU is pinned means the accelerator is *waiting on the host* —
    /// tokenisation, sampling, a data loader — and no engine flag fixes it.
    /// People buy bigger GPUs for this.
    pub fn starved_by_host(&self, sys: &System) -> Option<String> {
        let busy = sys.cpu_bound()?;
        let gpu = self
            .devices()
            .iter()
            .map(|d| d.utilisation_pct)
            .max()
            .map(f64::from)?;
        (busy && gpu < 50.0).then(|| {
            format!(
                "the host is at {:.2} load per CPU while the busiest accelerator is \
                 only {gpu:.0}% utilised. The GPU is waiting on the host — \
                 tokenisation, sampling or data loading — and a bigger card will \
                 not fix that.",
                sys.load_per_cpu.unwrap_or(0.0)
            )
        })
    }

    /// `None` when either side is unknown — an unknown must not be reported as
    /// "nothing else is running".
    pub fn foreign_memory(&self, engine_kv_used: Option<f64>) -> Option<String> {
        let (host, kv) = (self.peak_memory_used()?, engine_kv_used?);
        (host > 0.85 && host - kv > 0.30).then(|| {
            format!(
                "the card is at {:.0}% memory but the engine's KV cache is only at \
                 {:.0}%. Roughly {:.0}% of the device is held by another process — \
                 engine telemetry alone cannot show you this.",
                host * 100.0,
                kv * 100.0,
                (host - kv) * 100.0
            )
        })
    }
}

/// Read accelerator state from the host.
///
/// Never fails: an unreadable host is a [`Support::Unavailable`] with the
/// reason, because a monitoring call that propagates an error takes down the
/// page that was asking whether things were healthy.
pub fn probe() -> Support {
    if !cfg!(any(target_os = "linux", target_os = "windows")) && !cfg!(target_os = "macos") {
        return Support::Unavailable {
            reason: "unrecognised platform".into(),
        };
    }
    if cfg!(target_os = "macos") {
        return Support::Unavailable {
            reason: "Apple silicon exposes no per-process GPU memory without \
                     powermetrics, which requires root. A dashboard is not worth \
                     a sudo prompt — the engine's own KV figures still apply."
                .into(),
        };
    }
    nvidia_smi()
}

/// Query `nvidia-smi` in its machine-readable mode.
fn nvidia_smi() -> Support {
    let query = "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw";
    let out = match run(
        "nvidia-smi",
        &[
            &format!("--query-gpu={query}"),
            "--format=csv,noheader,nounits",
        ],
    ) {
        Ok(o) => o,
        Err(e) => {
            return Support::Unavailable {
                reason: format!(
                    "{e}. On a machine with no NVIDIA driver this is expected, not a \
                     fault — AMD would need rocm-smi, which is not implemented rather \
                     than guessed at."
                ),
            };
        }
    };

    let devices: Vec<Device> = out.lines().filter_map(parse_row).collect();
    if devices.is_empty() {
        return Support::Unavailable {
            reason: "nvidia-smi ran but reported no devices".into(),
        };
    }
    Support::Available { devices }
}

/// One CSV row from `nvidia-smi`. Unparseable rows are dropped rather than
/// defaulted — a device reported as 0 MiB used would read as idle.
fn parse_row(line: &str) -> Option<Device> {
    let f: Vec<&str> = line.split(',').map(str::trim).collect();
    // `[N/A]` is what nvidia-smi prints for a field the card does not expose,
    // and it must stay `None` rather than becoming a zero. `parse` handles that
    // for us — the point is that the failure is not swallowed into a default.
    Some(Device {
        index: f.first()?.parse().ok()?,
        name: (*f.get(1)?).to_owned(),
        memory_used_mib: f.get(2)?.parse().ok()?,
        memory_total_mib: f.get(3)?.parse().ok()?,
        utilisation_pct: f.get(4)?.parse().ok()?,
        temperature_c: f.get(5).and_then(|s| s.parse::<u32>().ok()),
        power_w: f.get(6).and_then(|s| s.parse::<f64>().ok()),
    })
}

/// Run a command with a deadline, returning stdout.
fn run(bin: &str, args: &[&str]) -> Result<String, String> {
    use std::process::{Command, Stdio};

    let mut child = Command::new(bin)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("{bin} could not be run: {e}"))?;

    // Poll rather than block: a wedged driver makes nvidia-smi hang for
    // seconds, and that is precisely when someone is watching the dashboard.
    let deadline = std::time::Instant::now() + PROBE_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break,
            Ok(Some(status)) => return Err(format!("{bin} exited with {status}")),
            Ok(None) if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                return Err(format!("{bin} did not answer within {PROBE_TIMEOUT:?}"));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(e) => return Err(format!("{bin} failed: {e}")),
        }
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("{bin} output unreadable: {e}"))?;
    String::from_utf8(out.stdout).map_err(|e| format!("{bin} produced invalid UTF-8: {e}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// Real `nvidia-smi --format=csv,noheader,nounits` output shape.
    const ROWS: &str = "0, NVIDIA H100 80GB HBM3, 74210, 81559, 97, 61, 412.55\n\
                        1, NVIDIA H100 80GB HBM3, 1204, 81559, 3, 38, 78.20";

    #[test]
    fn real_shaped_output_parses_every_column() {
        let d: Vec<Device> = ROWS.lines().filter_map(parse_row).collect();
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].index, 0);
        assert_eq!(d[0].name, "NVIDIA H100 80GB HBM3");
        assert_eq!(d[0].memory_used_mib, 74210);
        assert_eq!(d[0].utilisation_pct, 97);
        assert_eq!(d[0].temperature_c, Some(61));
        assert_eq!(d[0].power_w, Some(412.55));
        assert!((d[0].memory_used().unwrap() - 0.9099).abs() < 0.001);
    }

    #[test]
    fn a_field_the_card_does_not_expose_stays_none_rather_than_zero() {
        // nvidia-smi prints [N/A] for these. A power draw of 0.0 W would read
        // as an idle card rather than an unreported one.
        let d = parse_row("0, L4, 100, 24000, 5, [N/A], [N/A]").unwrap();
        assert_eq!(d.temperature_c, None);
        assert_eq!(d.power_w, None);
        assert_eq!(d.memory_used_mib, 100, "required fields still parse");
    }

    #[test]
    fn an_unparseable_row_is_dropped_not_defaulted() {
        // A device silently reported as 0 MiB used would read as idle, which is
        // the worst possible failure for a capacity dashboard.
        let mixed = format!("{ROWS}\ngarbage, not, a, row");
        let d: Vec<Device> = mixed.lines().filter_map(parse_row).collect();
        assert_eq!(d.len(), 2);
        assert!(parse_row("").is_none());
        assert!(parse_row("0, only-two-fields").is_none());
    }

    #[test]
    fn unavailable_is_not_an_empty_device_list() {
        // "no GPUs found" and "we cannot look here" are different facts, and a
        // panel that renders both as blank teaches people to ignore it.
        let u = Support::Unavailable {
            reason: "no driver".into(),
        };
        assert!(u.devices().is_empty());
        assert_eq!(u.peak_memory_used(), None);
        assert_eq!(u.foreign_memory(Some(0.1)), None);
    }

    #[test]
    fn foreign_memory_is_the_thing_engine_telemetry_cannot_see() {
        // The card is nearly full; the engine's own cache is not. The remainder
        // belongs to something else on the box.
        let busy = Support::Available {
            devices: ROWS.lines().filter_map(parse_row).collect(),
        };
        let found = busy.foreign_memory(Some(0.30)).unwrap();
        assert!(found.contains("91% memory"), "{found}");
        assert!(found.contains("30%"), "{found}");
        assert!(found.contains("another process"), "{found}");
    }

    #[test]
    fn a_card_full_because_of_our_own_kv_cache_is_not_reported_as_foreign() {
        let busy = Support::Available {
            devices: ROWS.lines().filter_map(parse_row).collect(),
        };
        // Engine says 88%, host says 91% — that is us, not a squatter.
        assert_eq!(busy.foreign_memory(Some(0.88)), None);
    }

    #[test]
    fn an_unknown_on_either_side_reports_nothing_rather_than_guessing() {
        let busy = Support::Available {
            devices: ROWS.lines().filter_map(parse_row).collect(),
        };
        assert_eq!(busy.foreign_memory(None), None, "unknown KV is not 'clear'");
        let empty = Support::Available { devices: vec![] };
        assert_eq!(empty.foreign_memory(Some(0.1)), None);
    }

    #[test]
    fn peak_is_the_worst_device_not_the_first() {
        let busy = Support::Available {
            devices: ROWS.lines().filter_map(parse_row).collect(),
        };
        // Device 1 is nearly idle; the answer must still be device 0's 91%.
        assert!((busy.peak_memory_used().unwrap() - 0.9099).abs() < 0.001);
    }

    #[test]
    fn probing_this_machine_never_panics_and_always_explains_itself() {
        // Runs on whatever CI is. Either it reads devices or it says why not —
        // there is no third outcome, and neither is an error.
        match probe() {
            Support::Available { devices } => assert!(!devices.is_empty()),
            Support::Unavailable { reason } => assert!(
                reason.len() > 20,
                "an unavailable reason must be a sentence: {reason}"
            ),
        }
    }

    #[test]
    fn system_stats_are_partial_rather_than_all_or_nothing() {
        // /proc/meminfo exists on Linux and not macOS; load average is the
        // reverse on Windows. A partial answer beats an empty one.
        let s = system();
        assert!(s.cpus.is_some(), "CPU count is available on every platform");
        if let Some(l) = s.load_1m {
            assert!(l >= 0.0);
            assert!(s.load_per_cpu.is_some(), "per-CPU load follows from both");
        }
        if s.memory_total_mib.is_some() {
            assert!(s.memory_used().is_some());
        }
    }

    #[test]
    fn an_unmeasurable_host_is_unknown_rather_than_healthy() {
        // `false` here would read as "the host is fine" on a platform we simply
        // could not measure — the same mistake as reporting 0% KV usage.
        let blank = System {
            cpus: Some(8),
            load_1m: None,
            load_per_cpu: None,
            memory_total_mib: None,
            memory_available_mib: None,
        };
        assert_eq!(blank.cpu_bound(), None);
        assert_eq!(blank.memory_used(), None);
    }

    #[test]
    fn memory_used_is_computed_from_available_not_free() {
        // MemFree on a busy Linux box is near zero because the page cache holds
        // the rest; reporting that as pressure would be alarming and wrong.
        let s = System {
            cpus: Some(8),
            load_1m: None,
            load_per_cpu: None,
            memory_total_mib: Some(64_000),
            memory_available_mib: Some(48_000),
        };
        assert!((s.memory_used().unwrap() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn a_busy_host_with_an_idle_gpu_is_named_as_host_starvation() {
        // The misdiagnosis this prevents: people buy bigger cards for this.
        let idle_gpu = Support::Available {
            devices: vec![parse_row("0, L4, 100, 24000, 12, 40, 60.0").unwrap()],
        };
        let pinned = System {
            cpus: Some(8),
            load_1m: Some(9.6),
            load_per_cpu: Some(1.2),
            memory_total_mib: Some(64_000),
            memory_available_mib: Some(8_000),
        };
        let found = idle_gpu.starved_by_host(&pinned).unwrap();
        assert!(found.contains("waiting on the host"), "{found}");
        assert!(found.contains("bigger card will not fix"), "{found}");
    }

    #[test]
    fn a_busy_host_with_a_busy_gpu_is_not_starvation() {
        let busy_gpu = Support::Available {
            devices: ROWS.lines().filter_map(parse_row).collect(),
        };
        let pinned = System {
            cpus: Some(8),
            load_1m: Some(9.6),
            load_per_cpu: Some(1.2),
            memory_total_mib: None,
            memory_available_mib: None,
        };
        // 97% GPU utilisation: the host being busy is expected, not a finding.
        assert_eq!(busy_gpu.starved_by_host(&pinned), None);
    }

    #[test]
    fn starvation_is_silent_when_either_side_is_unknown() {
        let gpu = Support::Available {
            devices: ROWS.lines().filter_map(parse_row).collect(),
        };
        let unknown = System {
            cpus: Some(8),
            load_1m: None,
            load_per_cpu: None,
            memory_total_mib: None,
            memory_available_mib: None,
        };
        assert_eq!(gpu.starved_by_host(&unknown), None);
        assert_eq!(
            Support::Unavailable { reason: "x".into() }.starved_by_host(&unknown),
            None
        );
    }

    #[test]
    fn a_missing_binary_is_a_reason_not_a_crash() {
        let e = run("definitely-not-a-real-binary-xyz", &["--help"]).unwrap_err();
        assert!(e.contains("could not be run"), "{e}");
    }
}
