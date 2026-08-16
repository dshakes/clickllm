//! The Rust↔Python seam.
//!
//! ## What crosses, and why it is this narrow
//!
//! onpar is split on a deliberate line: **Rust holds the datapath, Python
//! holds the judgment.** Routing, metering and capture must be fast and must not
//! stall; clustering, grading and equivalence must be readable and easy to change.
//! Neither half wants to become the other.
//!
//! So only two things cross, and both for the same reason — they would otherwise
//! exist twice:
//!
//! - **The capture format.** It is length-framed XChaCha20-Poly1305. Python
//!   re-implementing that would be a second implementation of a security-critical
//!   format, and the two would drift. The one that already exists is exposed.
//! - **The redactor.** Its patterns and its Luhn check are policy. Two copies of
//!   a policy means one of them is quietly out of date, and the stale one is the
//!   one that leaks.
//!
//! What deliberately does *not* cross:
//!
//! - **Sizing and fit.** Pure Python, zero dependencies, so `uvx onpar fit`
//!   works on any machine with an interpreter. Moving it here would trade that
//!   for a compiled wheel and buy nothing.
//! - **Routing control.** A running gateway holds its router in memory; writing
//!   a file from Python would change nothing. Phase changes go over HTTP to the
//!   live process, which is also the only form that has an audit trail.
//! - **Grading and clustering.** Judgment. It changes weekly and belongs where
//!   it is cheap to change.
//!
//! ## Version skew
//!
//! A Python package paired with a mismatched extension is the failure mode this
//! seam invites: it imports fine and then misreads a format. `__version__` is
//! exported so the Python side can refuse loudly instead.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use onpar_gateway::capture::store::{Capture, CaptureStore};
use onpar_gateway::capture::{Kind, redact as redact_text};

pyo3::create_exception!(
    _onpar_core,
    CaptureError,
    pyo3::exceptions::PyException,
    "A capture log could not be read — usually the wrong key, or a tampered file."
);

/// Version of the compiled extension.
///
/// The Python side compares this against its own and refuses a mismatch rather
/// than reading a format it may not understand.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Redact `text`, returning the cleaned string and a count of what was removed.
///
/// The counts are the point as much as the string: a redactor that silently does
/// nothing is indistinguishable from one that works.
///
/// Raises `ValueError` if the text is too large to scan in full — the fail-closed
/// contract, surfaced rather than papered over with a partial result.
#[pyfunction]
#[pyo3(name = "redact")]
fn redact_py(py: Python<'_>, text: &str) -> PyResult<(String, Py<PyDict>)> {
    let (clean, report) = redact_text(text).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let counts = PyDict::new(py);
    for (kind, n) in report.counts {
        counts.set_item(kind_name(kind), n)?;
    }
    Ok((clean, counts.into()))
}

/// Stable snake_case names, matching the JSON the Rust side serialises.
fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Email => "email",
        Kind::CardNumber => "card_number",
        Kind::Ssn => "ssn",
        Kind::Phone => "phone",
        Kind::Secret => "secret",
        Kind::Jwt => "jwt",
        Kind::IpAddress => "ip_address",
        Kind::PrivateKey => "private_key",
    }
}

/// Read and decrypt every capture in the log at `path`.
///
/// Returns a list of dicts — plain data, so the Python side is free to reshape
/// it without a binding change every time a field moves.
///
/// An absent log is an empty list, not an error: "nothing captured yet" is the
/// normal state of a fresh install, and making the caller distinguish it from a
/// real failure would only invite a bare `except`.
///
/// Raises `CaptureError` on a wrong key or a tampered file, `ValueError` if the
/// key is not 32 bytes, `IOError` if the log cannot be read.
#[pyfunction]
#[pyo3(signature = (path, key))]
fn read_captures(py: Python<'_>, path: &str, key: &[u8]) -> PyResult<Py<PyList>> {
    let key = key32(key)?;
    // Decryption of a large log is real work and touches no Python objects.
    // Holding the GIL through it would stall every other thread in the process.
    let captures = py.detach(|| -> Result<Vec<Capture>, String> {
        let store = CaptureStore::open(path, &key).map_err(|e| e.to_string())?;
        store.read_all().map_err(|e| e.to_string())
    });

    let captures = match captures {
        Ok(c) => c,
        // A read failure is almost always the wrong key; say so, since the
        // underlying "decrypt failed" is technically true and practically useless.
        Err(e) if e.contains("decrypt") => {
            return Err(CaptureError::new_err(format!(
                "{e} — check the key matches the one the gateway wrote with"
            )));
        }
        Err(e) => return Err(PyIOError::new_err(e)),
    };

    let out = PyList::empty(py);
    for c in captures {
        out.append(capture_to_dict(py, &c)?)?;
    }
    Ok(out.into())
}

fn capture_to_dict(py: Python<'_>, c: &Capture) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("request_id", &c.request_id)?;
    d.set_item("model", &c.model)?;
    d.set_item("backend", &c.backend)?;
    d.set_item("messages", json_to_py(py, &c.messages)?)?;
    d.set_item("response", &c.response)?;
    // None, not 0. A missing token count means the upstream reported none, and a
    // zero here would read as "this request was free".
    d.set_item("prompt_tokens", c.prompt_tokens)?;
    d.set_item("completion_tokens", c.completion_tokens)?;
    // `latency_ms`, not `duration_ms`. The Rust struct and the Python
    // dataclass named the same measurement differently, and `Capture(**row)`
    // raises on the mismatch — so the bridge is where the two names meet, and
    // the seam test asserts the keys here are exactly the fields there.
    d.set_item("latency_ms", c.duration_ms)?;
    d.set_item("tools", json_to_py(py, &c.tools)?)?;
    d.set_item("tool_calls", c.tool_calls.clone())?;
    d.set_item("response_format", c.response_format.clone())?;
    let redacted = PyDict::new(py);
    for (kind, n) in &c.redacted.counts {
        redacted.set_item(kind_name(*kind), n)?;
    }
    d.set_item("redacted", redacted)?;
    Ok(d.into())
}

/// Convert `serde_json` to native Python types.
///
/// Handing back a JSON string instead would make every caller parse it, and the
/// message tree is the part downstream clustering reads most.
fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use serde_json::Value;
    Ok(match v {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        Value::Number(n) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => i.into_pyobject(py)?.into_any().unbind(),
            (None, Some(f)) => f.into_pyobject(py)?.into_any().unbind(),
            _ => py.None(),
        },
        Value::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        Value::Array(a) => {
            let list = PyList::empty(py);
            for item in a {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        Value::Object(o) => {
            let d = PyDict::new(py);
            for (k, val) in o {
                d.set_item(k, json_to_py(py, val)?)?;
            }
            d.into_any().unbind()
        }
    })
}

/// Load the capture key at `path`, generating one if it does not exist.
///
/// The generated file is owner-only. That is the whole of the access control —
/// see `onpar_gateway::capture::store` for what this protects against and,
/// more importantly, what it does not.
#[pyfunction]
fn load_or_create_key(py: Python<'_>, path: &str) -> PyResult<Py<PyBytes>> {
    let key = CaptureStore::load_or_create_key(std::path::Path::new(path))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &key).into())
}

fn key32(key: &[u8]) -> PyResult<[u8; 32]> {
    <[u8; 32]>::try_from(key)
        .map_err(|_| PyValueError::new_err(format!("key must be 32 bytes, got {}", key.len())))
}

/// The `onpar_core` extension module.
#[pymodule]
fn _onpar_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("CaptureError", m.py().get_type::<CaptureError>())?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(redact_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_captures, m)?)?;
    m.add_function(wrap_pyfunction!(load_or_create_key, m)?)?;
    Ok(())
}
