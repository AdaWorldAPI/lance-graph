//! Dump the base `Network` header at the front of a serialized recognizer
//! component (`eng.lstm`) — the Rust side of the network base-header byte-parity
//! leaf, sibling to `recoder_dump`. Also prints the [`FacetCascade`] the node
//! sinks onto (the ruff→OGAR harvest → V3 SoA target).
//!
//! ```sh
//! # Extract the lstm component (starts with the network, lstmrecognizer.cpp:135):
//! combine_tessdata -u $(dpkg -L tesseract-ocr-eng | grep eng.traineddata) /tmp/eng.
//! # C++ oracle (network_spec_oracle.cpp): links libtesseract, calls the REAL
//! # Network::CreateFromFile on the same bytes and dumps the loaded top node's
//! # type / ni / no / num_weights / name + spec() (the known-answer self-check).
//! #   ./network_spec_oracle /tmp/eng.lstm > /tmp/oracle_network.txt
//! # Rust side (parses only the base header — the shared prefix of every layer):
//! cargo run -p lance-graph-contract --example network_dump -- /tmp/eng.lstm > /tmp/rust_network.txt
//! # The "header:" line is byte-identical between the two => the base header
//! # parse is byte-parity green.
//! ```

#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::process::ExitCode;

use lance_graph_contract::network::NetworkHeader;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: network_dump <path/to/eng.lstm>");
        return ExitCode::FAILURE;
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("error reading {path}: {err}");
            return ExitCode::FAILURE;
        }
    };
    match NetworkHeader::from_le_bytes(&bytes) {
        Ok((header, consumed)) => {
            // The byte-parity line (diffed against the oracle's loaded top node).
            println!("header: {}", header.dump());
            // The V3 SoA sink: the 16-byte FacetCascade (classid + 6×8:8), hex.
            let f = header.to_facet();
            let hex: String = f.to_bytes().iter().map(|b| format!("{b:02x}")).collect();
            println!("facet:  classid={:#010x} bytes={hex}", f.facet_classid);
            println!("consumed: {consumed} bytes (base header; subclass payload follows)");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error parsing header: {err:?}");
            ExitCode::FAILURE
        }
    }
}
