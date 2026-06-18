//! Dump the `UNICHAR` byte-parity TSV — the Rust side of the `unichar` half of
//! the Tesseract transcode probe (`PROBE-OGAR-ADAPTER-UNICHARSET` sibling).
//!
//! Emits the exact shape a small libtesseract oracle prints: all 256
//! `utf8_step` values, then `utf8_to_utf32` over a curated hex corpus.
//!
//! ```sh
//! # C++ oracle (links -lleptonica only to satisfy the linker; never calls it):
//! #   g++ -std=c++17 -I<tess>/include oracle.cpp -ltesseract -lleptonica -o oracle
//! #   ./oracle cases.hex > /tmp/oracle.tsv
//! cargo run -p lance-graph-contract --example unichar_dump > /tmp/rust.tsv
//! diff /tmp/oracle.tsv /tmp/rust.tsv   # byte-identical => parity holds
//! ```

#![allow(
    clippy::print_stdout,
    reason = "a dump CLI example writes to stdout by design"
)]

use lance_graph_contract::unichar::{utf8_step, utf8_to_utf32};

/// Curated corpus (hex-encoded bytes), identical to the oracle's input: valid
/// 1/2/3/4-byte chars, multi-char strings, lone illegal leads, and the overlong
/// NUL quirk. Hex keeps the exact bytes independent of source encoding.
const CASES: &[&str] = &[
    "41",           // A
    "c3a9",         // é  U+00E9
    "e4b8ad",       // 中 U+4E2D
    "f09f9880",     // 😀 U+1F600
    "414243",       // ABC
    "48c3a9",       // Hé
    "e4b8ade69687", // 中文
    "80",           // lone continuation -> ILLEGAL
    "bf",           // lone continuation -> ILLEGAL
    "f8",           // 5-byte form       -> ILLEGAL
    "ff",           // 0xFF              -> ILLEGAL
    "c080",         // overlong NUL      -> [0]
];

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("valid hex pair"))
        .collect()
}

fn main() {
    // Section 1: exhaustive utf8_step over all 256 lead bytes.
    for b in 0u16..256 {
        println!("STEP\t{b}\t{}", utf8_step(b as u8));
    }
    // Section 2: utf8_to_utf32 over the corpus.
    for &hex in CASES {
        let bytes = hex_to_bytes(hex);
        let decoded = match utf8_to_utf32(&bytes) {
            None => "ILLEGAL".to_string(),
            Some(codepoints) => codepoints
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        };
        println!("U32\t{hex}\t{decoded}");
    }
}
