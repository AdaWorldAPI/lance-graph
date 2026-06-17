//! Dump a `.unicharset`'s id→unichar table — the Rust side of the byte-parity
//! probe `PROBE-OGAR-ADAPTER-UNICHARSET`.
//!
//! ```sh
//! # on a box with libtesseract + libleptonica installed:
//! combine_tessdata -u $(dpkg -L tesseract-ocr-eng | grep eng.traineddata) /tmp/eng.
//! # C++ oracle (links -lleptonica only to satisfy the linker; never calls it):
//! #   g++ oracle.cpp -ltesseract -lleptonica -o oracle && ./oracle /tmp/eng.unicharset > /tmp/oracle.tsv
//! # Rust side:
//! cargo run -p lance-graph-contract --example unicharset_dump -- /tmp/eng.unicharset > /tmp/rust.tsv
//! diff /tmp/oracle.tsv /tmp/rust.tsv   # byte-identical => CONJECTURE -> FINDING
//! ```

#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::path::Path;
use std::process::ExitCode;

use lance_graph_contract::unicharset::UniCharSet;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: unicharset_dump <path/to/eng.unicharset>");
        return ExitCode::FAILURE;
    };
    match UniCharSet::load_from_file(Path::new(&path)) {
        Ok(unicharset) => {
            print!("{}", unicharset.dump());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
