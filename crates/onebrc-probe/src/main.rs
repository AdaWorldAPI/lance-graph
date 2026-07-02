//! `onebrc-probe` CLI — generate a deterministic corpus, or run a lane
//! against one, printing throughput + a correctness spot-check.
//!
//! ```text
//! onebrc-probe gen <path> <rows> <seed>
//! onebrc-probe run <path> <lane:a|b|c|d> [workers]
//! ```
//!
//! Lane `b` requires `--features lane-b`; lane `d` requires
//! `--features lane-d` (see `README.md` §3/§4).

use onebrc_probe::{gen::gen, lane_a_scalar, lane_c_threads};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gen") => cmd_gen(&args[2..]),
        Some("run") => cmd_run(&args[2..]),
        _ => {
            eprintln!(
                "usage:\n  onebrc-probe gen <path> <rows> <seed>\n  onebrc-probe run <path> <lane:a|b|c|d> [workers]"
            );
            std::process::exit(2);
        }
    }
}

fn cmd_gen(args: &[String]) {
    let path = PathBuf::from(args.first().expect("usage: gen <path> <rows> <seed>"));
    let rows: u64 = args
        .get(1)
        .expect("usage: gen <path> <rows> <seed>")
        .parse()
        .expect("rows must be a u64");
    let seed: u64 = args
        .get(2)
        .expect("usage: gen <path> <rows> <seed>")
        .parse()
        .expect("seed must be a u64");

    let result = gen(&path, rows, seed).expect("corpus generation failed");
    // The archival recipe line: input + recipe + hash travel together, per
    // the workspace's archival convention (see README §2, gen.rs doc).
    println!(
        "rows={} seed={} sha256={}",
        result.rows, result.seed, result.sha256_hex
    );
}

fn cmd_run(args: &[String]) {
    let path = PathBuf::from(
        args.first()
            .expect("usage: run <path> <lane:a|b|c|d> [workers]"),
    );
    let lane = args.get(1).map(String::as_str).unwrap_or("a");
    let workers: usize = args
        .get(2)
        .map(|s| s.parse().expect("workers must be a usize"))
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });

    // NOTE (mmap note): plain `std::fs::read`, NOT mmap. automataIA/1brc-rs
    // treats `memmap2::Mmap` as "the only path to break the 2-second
    // barrier" for the full 1B-row file (avoiding ~13 GB of explicit
    // allocation). This probe deliberately trades that peak-throughput
    // headroom for staying at zero external dependencies for lanes A/C
    // (see Cargo.toml + README §1). Revisit if/when a Lane B or later adds
    // an mmap-capable dependency.
    let data = fs::read(&path).expect("read corpus file");
    let rows = data.iter().filter(|&&b| b == b'\n').count();

    let start = Instant::now();
    let map = match lane {
        "a" => lane_a_scalar(&data),
        "c" => lane_c_threads(&data, workers),
        "b" => {
            #[cfg(feature = "lane-b")]
            {
                onebrc_probe::lane_b_simd(&data)
            }
            #[cfg(not(feature = "lane-b"))]
            {
                eprintln!("lane b requires --features lane-b");
                std::process::exit(1);
            }
        }
        "d" => {
            #[cfg(feature = "lane-d")]
            {
                onebrc_probe::lane_d_ractor(&data, workers)
            }
            #[cfg(not(feature = "lane-d"))]
            {
                eprintln!("lane d requires --features lane-d");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown lane '{other}' (expected 'a', 'b', 'c', or 'd')");
            std::process::exit(2);
        }
    };
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput_mrows_s = (rows as f64 / 1_000_000.0) / elapsed.as_secs_f64();

    println!(
        "lane={lane} rows={rows} workers={workers} elapsed_ms={elapsed_ms:.3} throughput_mrows_s={throughput_mrows_s:.3}"
    );

    // Correctness spot-check surface — first/last 3 stations by name (map
    // is a BTreeMap, so iteration order is the sorted station-name order).
    println!("-- first 3 stations --");
    for (name, stats) in map.iter().take(3) {
        println!("  {name}: {stats:?}");
    }
    println!("-- last 3 stations --");
    for (name, stats) in map.iter().rev().take(3) {
        println!("  {name}: {stats:?}");
    }
}
