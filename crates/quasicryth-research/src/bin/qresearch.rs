//! `qresearch` — command-line interface for the quasicryth-research
//! compressor pipeline.
//!
//! Subcommands:
//! - `compress [-v flat|cow] <input> <output>` — read `input`, compress,
//!   write to `output`. Default variant: flat.
//! - `decompress <input> <output>` — read compressed `input`, decompress,
//!   write to `output`.
//! - `round-trip [-v flat|cow] <input>` — compress `input` to memory,
//!   decompress, verify identity, print stats.
//!
//! NOTE: this binary is **research-grade**. It is NOT byte-compatible
//! with the upstream `quasicryth` v5.6 `.qm56` format. See the crate
//! README for the format spec and the full list of simplifications.

use std::fs;
use std::process::ExitCode;

use quasicryth_research::pipeline::{compress, decompress, Variant};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();

    match args_ref.as_slice() {
        ["compress", "-v", "flat", input, output] | ["compress", input, output] => {
            run_compress(input, output, Variant::Flat)
        }
        ["compress", "-v", "cow", input, output] => run_compress(input, output, Variant::CowRadix),
        ["decompress", input, output] => run_decompress(input, output),
        ["round-trip", "-v", "flat", input] | ["round-trip", input] => {
            run_round_trip(input, Variant::Flat)
        }
        ["round-trip", "-v", "cow", input] => run_round_trip(input, Variant::CowRadix),
        ["--help"] | ["-h"] | [] => {
            print_usage();
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("error: unrecognized arguments");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    eprintln!(
        "qresearch — quasicryth-research CLI

USAGE:
  qresearch compress   [-v flat|cow] <input> <output>
  qresearch decompress <input> <output>
  qresearch round-trip [-v flat|cow] <input>

Default variant: flat.
Research-grade only — NOT byte-compatible with the upstream qm56 format."
    );
}

fn run_compress(input: &str, output: &str, variant: Variant) -> ExitCode {
    let data = match fs::read(input) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error reading {input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let compressed = match compress(&data, variant) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("compress failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = fs::write(output, &compressed) {
        eprintln!("error writing {output}: {e}");
        return ExitCode::FAILURE;
    }
    let ratio = 100.0 * compressed.len() as f64 / data.len() as f64;
    println!(
        "compressed {} → {} ({} → {} bytes, {:.2}%, variant={:?})",
        input,
        output,
        data.len(),
        compressed.len(),
        ratio,
        variant
    );
    ExitCode::SUCCESS
}

fn run_decompress(input: &str, output: &str) -> ExitCode {
    let data = match fs::read(input) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error reading {input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let decompressed = match decompress(&data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("decompress failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = fs::write(output, &decompressed) {
        eprintln!("error writing {output}: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "decompressed {} → {} ({} → {} bytes)",
        input,
        output,
        data.len(),
        decompressed.len()
    );
    ExitCode::SUCCESS
}

fn run_round_trip(input: &str, variant: Variant) -> ExitCode {
    let data = match fs::read(input) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error reading {input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let compressed = match compress(&data, variant) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("compress failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let decompressed = match decompress(&compressed) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("decompress failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if decompressed == data {
        let ratio = 100.0 * compressed.len() as f64 / data.len() as f64;
        println!(
            "round-trip OK: {} bytes → {} compressed ({:.2}%) → identical decompressed, variant={:?}",
            data.len(),
            compressed.len(),
            ratio,
            variant
        );
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "round-trip MISMATCH: input {} bytes, decoded {} bytes (variant={:?})",
            data.len(),
            decompressed.len(),
            variant
        );
        ExitCode::FAILURE
    }
}
