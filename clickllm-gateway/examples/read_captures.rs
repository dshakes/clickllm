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
    // Read, never create. `load_or_create_key` would turn a typo in the key
    // path into a brand-new key file, and then report an empty log — surprising
    // filesystem mutation from a tool whose entire purpose is to look.
    let raw = std::fs::read(&key_path).expect("read key file");
    let key: [u8; 32] = raw
        .as_slice()
        .try_into()
        .expect("key file must be 32 bytes");
    let store = CaptureStore::open(&log, &key).expect("open");
    for c in store.read_all().expect("read") {
        println!("request  {}", c.request_id);
        println!("  model    {}", c.model);
        println!("  messages {}", c.messages);
        println!("  response {}", c.response);
        println!("  redacted {:?}", c.redacted);
    }
}
