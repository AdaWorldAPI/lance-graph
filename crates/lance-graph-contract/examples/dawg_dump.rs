//! Dump a `SquishedDawg`'s header + edges — the Rust side of the dawg
//! byte-parity leaf, sibling to `recoder_dump`.
//!
//! ```sh
//! # on a box with libtesseract + libleptonica installed:
//! combine_tessdata -u $(dpkg -L tesseract-ocr-eng | grep eng.traineddata) /tmp/eng.
//! # Rust side:
//! cargo run -p lance-graph-contract --example dawg_dump -- /tmp/eng.lstm-punc-dawg
//! ```

#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::path::Path;
use std::process::ExitCode;

use lance_graph_contract::dawg::SquishedDawg;

fn main() -> ExitCode {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/eng.lstm-punc-dawg".to_string());
    match SquishedDawg::load_from_file(Path::new(&path)) {
        Ok(dawg) => {
            println!("hdr\t{}\t{}", dawg.unicharset_size(), dawg.num_edges());
            for i in 0..dawg.num_edges() {
                println!(
                    "e\t{}\t{}\t{}\t{}",
                    i,
                    dawg.edge_letter(i),
                    dawg.next_node(i),
                    u8::from(dawg.end_of_word(i))
                );
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error reading {path}: {err}");
            ExitCode::FAILURE
        }
    }
}
