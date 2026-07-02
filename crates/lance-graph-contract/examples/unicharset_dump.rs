//! Dump a `.unicharset`'s id→unichar table (default) or a per-id column:
//! `properties` (category bits), `script` (script ids), `other_case` (case-pair
//! ids), `direction` (bidi codes), `mirror` (mirror ids) — the Rust side of the
//! byte-parity probe `PROBE-OGAR-ADAPTER-UNICHARSET`.
//!
//! ```sh
//! # on a box with libtesseract + libleptonica installed:
//! combine_tessdata -u $(dpkg -L tesseract-ocr-eng | grep eng.traineddata) /tmp/eng.
//! # C++ oracle (links -lleptonica only to satisfy the linker; never calls it).
//! # An oracle that prints BOTH modes self-validates the build: the bijection
//! # half is a proven 112/112 reference (E-CPP-PARITY-1), so a 0-diff there
//! # confirms the object layout before the property half (E-CPP-PARITY-3) is
//! # trusted — important when the source header and installed lib versions skew.
//! #   g++ oracle.cpp -ltesseract -lleptonica -o oracle
//! #   ./oracle /tmp/eng.unicharset bijection  > /tmp/oracle.tsv
//! #   ./oracle /tmp/eng.unicharset properties > /tmp/oracle_props.tsv
//! # Rust side:
//! cargo run -p lance-graph-contract --example unicharset_dump -- /tmp/eng.unicharset            > /tmp/rust.tsv
//! cargo run -p lance-graph-contract --example unicharset_dump -- /tmp/eng.unicharset properties > /tmp/rust_props.tsv
//! diff /tmp/oracle.tsv /tmp/rust.tsv && diff /tmp/oracle_props.tsv /tmp/rust_props.tsv
//! # both byte-identical => CONJECTURE -> FINDING
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
        eprintln!(
            "usage: unicharset_dump <path/to/eng.unicharset> [properties|script|other_case|direction|mirror]"
        );
        return ExitCode::FAILURE;
    };
    let mode = std::env::args().nth(2).unwrap_or_default();
    match UniCharSet::load_from_file(Path::new(&path)) {
        Ok(unicharset) => {
            match mode.as_str() {
                "properties" => print!("{}", unicharset.dump_properties()),
                "script" => print!("{}", unicharset.dump_script()),
                "other_case" => print!("{}", unicharset.dump_other_case()),
                "direction" => print!("{}", unicharset.dump_direction()),
                "mirror" => print!("{}", unicharset.dump_mirror()),
                _ => print!("{}", unicharset.dump()),
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
