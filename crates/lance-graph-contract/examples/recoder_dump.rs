//! Dump a `.lstm-recoder`'s encoder table (`encode`) or decode round-trip
//! (`decode`) — the Rust side of the recoder byte-parity leaf, sibling to
//! `unicharset_dump`.
//!
//! ```sh
//! # on a box with libtesseract + libleptonica installed:
//! combine_tessdata -u $(dpkg -L tesseract-ocr-eng | grep eng.traineddata) /tmp/eng.
//! # C++ oracle (recoder_oracle.cpp): loads the SAME component via TFile and dumps
//! # EncodeUnichar / DecodeUnichar / code_range. It also prints, per id, the
//! # UNICHARSET bijection + an Encode.Decode round-trip so the NEW UnicharCompress
//! # object layout self-validates against the 5.5.0-header / 5.3.4-lib ABI skew.
//! #   ./recoder_oracle /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder encode > /tmp/oracle_recoder_encode.tsv
//! #   ./recoder_oracle /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder decode > /tmp/oracle_recoder_decode.tsv
//! # Rust side:
//! cargo run -p lance-graph-contract --example recoder_dump -- /tmp/eng.lstm-recoder encode > /tmp/rust_recoder_encode.tsv
//! cargo run -p lance-graph-contract --example recoder_dump -- /tmp/eng.lstm-recoder decode > /tmp/rust_recoder_decode.tsv
//! diff /tmp/oracle_recoder_encode.tsv /tmp/rust_recoder_encode.tsv \
//!   && diff /tmp/oracle_recoder_decode.tsv /tmp/rust_recoder_decode.tsv
//! # both byte-identical => the recoder load-side is byte-parity green
//!
//! # The `beam` mode dumps the SetupDecoder trie maps (is_valid_start_ /
//! # final_codes_ / next_codes_ — the RecodeBeamSearch surface, Leaf 7a):
//! #   ./recoder_oracle /tmp/eng.lstm-unicharset /tmp/eng.lstm-recoder beam > /tmp/oracle_recoder_beam.tsv
//! cargo run -p lance-graph-contract --example recoder_dump -- /tmp/eng.lstm-recoder beam > /tmp/rust_recoder_beam.tsv
//! diff /tmp/oracle_recoder_beam.tsv /tmp/rust_recoder_beam.tsv   # byte-identical => 7a green
//! ```

#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use std::path::Path;
use std::process::ExitCode;

use lance_graph_contract::unicharcompress::UnicharCompress;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: recoder_dump <path/to/eng.lstm-recoder> [encode|decode]");
        return ExitCode::FAILURE;
    };
    let mode = std::env::args().nth(2).unwrap_or_default();
    match UnicharCompress::load_from_file(Path::new(&path)) {
        Ok(recoder) => {
            match mode.as_str() {
                "decode" => print!("{}", recoder.dump_decode()),
                "beam" => print!("{}", recoder.dump_beam()),
                _ => print!("{}", recoder.dump_encode()),
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
