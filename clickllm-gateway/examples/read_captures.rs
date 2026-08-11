//! Read a capture log back, for verifying the write path by hand.
//!
//! Pairs with `write_captures.rs`. That one exists so the Python seam tests
//! read bytes the gateway actually produced; this one exists so a human can
//! confirm what reached disk — specifically that redaction ran, which is the
//! one guarantee in this crate that is worth checking with your own eyes.

#![allow(clippy::expect_used, clippy::print_stdout)]

use clickllm_gateway::capture::store::CaptureStore;

fn main() {
    let mut args = std::env::args().skip(1);
    let log = args.next().expect("usage: read_captures <log> <key>");
    let key_path = args.next().expect("usage: read_captures <log> <key>");
    let key = CaptureStore::load_or_create_key(std::path::Path::new(&key_path)).expect("key");
    let store = CaptureStore::open(&log, &key).expect("open");
    for c in store.read_all().expect("read") {
        println!("request  {}", c.request_id);
        println!("  model    {}", c.model);
        println!("  messages {}", c.messages);
        println!("  response {}", c.response);
        println!("  redacted {:?}", c.redacted);
    }
}
