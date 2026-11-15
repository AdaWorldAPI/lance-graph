//! `onebrc-probe` CLI — generate a deterministic corpus, or run a lane
//! against one, printing throughput + a correctness spot-check.
//!
//! ```text
//! onebrc-probe gen <path> <rows> <seed>
//! onebrc-probe run <path> <lane:a|b|c|d|e|f|r|s> [workers] [batches]
//! ```
//!
//! Lane `b` requires `--features lane-b`; lane `d` and lane `e` require
//! `--features lane-d` / `--features lane-e` respectively (see `README.md`
//! §3/§4). `batches` is lane-`e`-only (optional, default `workers * 16`).

use onebrc_probe::{lane_a_scalar, lane_c_threads};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gen") => cmd_gen(&args[2..]),
        Some("run") => cmd_run(&args[2..]),
        Some("wall") => cmd_wall(&args[2..]),
        Some("dist") => cmd_dist(&args[2..]),
        _ => {
            eprintln!(
                "usage:\n  onebrc-probe gen <path> <rows> <seed> [stations]\n  onebrc-probe run <path> <lane:a|b|c|d|e|f|r|s|k|kt|km|kw|kmw|kh|kmwh|kmwc> [workers] [batches]\n  onebrc-probe wall <path> <k|kt|km|kw|kmw|kh|kmwh|kmwc|s> [workers]\n  onebrc-probe dist <path> [workers] [hist|c9]"
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

    // Optional 4th argument: number of distinct stations (default 400).
    let stations: usize = args
        .get(3)
        .map(|s| s.parse().expect("stations must be a usize"))
        .unwrap_or(onebrc_probe::gen::STATION_COUNT);
    let result = onebrc_probe::gen::gen_with_stations(&path, rows, seed, stations)
        .expect("corpus generation failed");
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
            .expect("usage: run <path> <lane:a|b|c|d|e|f|r|s> [workers] [batches]"),
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
    // Lane-`e`-only: number of kanban-journaled batches the corpus splits
    // into (default `workers * 16` — fine-grained batching prices per-card
    // scheduling; see `lane_e.rs` module doc).
    let batches: usize = args
        .get(3)
        .map(|s| s.parse().expect("batches must be a usize"))
        .unwrap_or(workers * 16);
    // Lane-`g`-only: the SAME 4th positional arg is the shard-owner count
    // (Morton-tile-range mailboxes; default 4). Lanes e and g never share
    // an invocation, so overloading the position is unambiguous.
    let shards: usize = args
        .get(3)
        .map(|s| s.parse().expect("shards must be a usize"))
        .unwrap_or(4);

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
        "e" => {
            #[cfg(feature = "lane-e")]
            {
                onebrc_probe::lane_e_kanban(&data, workers, batches)
            }
            #[cfg(not(feature = "lane-e"))]
            {
                eprintln!("lane e requires --features lane-e");
                std::process::exit(1);
            }
        }
        // Lanes F/R are std-only (no feature gate): the substrate-native
        // Morton-tile SoA lane and its plain-radix control (lane_f.rs).
        "f" => onebrc_probe::lane_f_morton(&data, workers),
        "r" => onebrc_probe::lane_r_radix(&data, workers),
        "s" => onebrc_probe::lane_s_swar(&data, workers),
        "k" | "kt" | "km" | "kw" | "kmw" | "kh" | "kmwh" | "kmwc" => {
            let (p, h, g) = lane_k_variant(lane);
            onebrc_probe::lane_k_codebook_agg(&data, workers, p, h, g)
        }
        "t" => onebrc_probe::lane_t_trie(&data, workers),
        "t8" => onebrc_probe::lane_t_byte(&data, workers),
        "g" => {
            #[cfg(feature = "lane-g")]
            {
                // 4th positional arg = shards for lane g (default 4).
                onebrc_probe::lane_g_kanban_soa(&data, workers, shards)
            }
            #[cfg(not(feature = "lane-g"))]
            {
                eprintln!("lane g requires --features lane-g");
                std::process::exit(1);
            }
        }
        "i" => {
            #[cfg(feature = "lane-i")]
            {
                // Fixed topology: all 65536 mailboxes upfront, 64K-row
                // CAM batches (see lane_i.rs; breakdown on stderr).
                onebrc_probe::lane_i_batch_pipeline(&data, workers)
            }
            #[cfg(not(feature = "lane-i"))]
            {
                eprintln!("lane i requires --features lane-i");
                std::process::exit(1);
            }
        }
        "j" => {
            #[cfg(feature = "lane-j")]
            {
                // Positional knobs: [workers] [grid] [lanes] [registry 0|1]
                // (defaults: gridlake 4096, 1 lane pair, no registry).
                let grid: usize = args
                    .get(3)
                    .map(|s| s.parse().expect("grid must be a usize"))
                    .unwrap_or(4096);
                let lanes: usize = args
                    .get(4)
                    .map(|s| s.parse().expect("lanes must be a usize"))
                    .unwrap_or(1);
                let registry = args.get(5).map(String::as_str) == Some("1");
                onebrc_probe::lane_j_grid_pipeline_with(
                    &data,
                    workers,
                    grid,
                    lanes,
                    registry,
                    1 << 16,
                )
            }
            #[cfg(not(feature = "lane-j"))]
            {
                eprintln!("lane j requires --features lane-j");
                std::process::exit(1);
            }
        }
        "h" => {
            #[cfg(feature = "lane-h")]
            {
                // 4th positional arg = NOMINAL owner granularity for lane h
                // (lazy activation makes live owners track occupancy).
                onebrc_probe::lane_h_orchestrated(&data, workers, shards)
            }
            #[cfg(not(feature = "lane-h"))]
            {
                eprintln!("lane h requires --features lane-h");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!(
                "unknown lane '{other}' (expected 'a', 'b', 'c', 'd', 'e', 'f', 'r', or 's')"
            );
            std::process::exit(2);
        }
    };
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput_mrows_s = (rows as f64 / 1_000_000.0) / elapsed.as_secs_f64();

    // `batches=<B>` is appended ONLY for lane e (see README §4) — the other
    // lanes don't have a batch concept, so the printed line stays identical
    // for them.
    if lane == "e" {
        println!(
            "lane={lane} rows={rows} workers={workers} batches={batches} elapsed_ms={elapsed_ms:.3} throughput_mrows_s={throughput_mrows_s:.3}"
        );
    } else if lane == "g" {
        println!(
            "lane={lane} rows={rows} workers={workers} shards={shards} elapsed_ms={elapsed_ms:.3} throughput_mrows_s={throughput_mrows_s:.3}"
        );
    } else {
        println!(
            "lane={lane} rows={rows} workers={workers} elapsed_ms={elapsed_ms:.3} throughput_mrows_s={throughput_mrows_s:.3}"
        );
    }

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

/// `wall <path> <k|kt|km|kw|kmw|kh|kmwh|kmwc|s> [workers]`: end-to-end time from opening the file
/// to the finished result, reported in nanoseconds per row. Lanes `k`/`kt`
/// memory-map the file; lane `s` reads it with `fs::read` inside the timer.
/// Run each lane twice and keep the second: the first run warms the page
/// cache, which neither lane controls.
#[cfg(unix)]
fn cmd_wall(args: &[String]) {
    let usage = "usage: wall <path> <k|kt|km|kw|kmw|kh|kmwh|kmwc|s> [workers]";
    let path = PathBuf::from(args.first().expect(usage));
    let lane = args.get(1).map(String::as_str).expect(usage);
    let workers: usize = args
        .get(2)
        .map(|s| s.parse().expect("workers must be a usize"))
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    let start = Instant::now();
    let (map, rows) = match lane {
        "k" | "kt" | "km" | "kw" | "kmw" | "kh" | "kmwh" | "kmwc" => {
            let m = onebrc_probe::lane_k::Mapped::open(&path).expect("map corpus file");
            let (p, h, g) = lane_k_variant(lane);
            let map = onebrc_probe::lane_k_codebook_agg(m.as_slice(), workers, p, h, g);
            let rows: u64 = map.values().map(|s| s.count as u64).sum();
            (map, rows)
        }
        "s" => {
            let data = fs::read(&path).expect("read corpus file");
            let map = onebrc_probe::lane_s_swar(&data, workers);
            let rows: u64 = map.values().map(|s| s.count as u64).sum();
            (map, rows)
        }
        other => {
            eprintln!("unknown wall lane '{other}' (expected k, kt, km, kw, kmw, kh, kmwh or s)");
            std::process::exit(2);
        }
    };
    let ns_per_row = start.elapsed().as_nanos() as f64 / rows.max(1) as f64;
    println!(
        "wall lane={lane} rows={rows} workers={workers} stations={} ns_per_row={ns_per_row:.3}",
        map.len()
    );
}

#[cfg(not(unix))]
fn cmd_wall(_: &[String]) {
    eprintln!("wall needs a Unix mmap");
    std::process::exit(1);
}

/// Lane K's variants: `k` parse, `kt` temperature table, `km` merykitty
/// parse; a `w` hashes names 8 bytes at a time; a trailing `h` aggregates
/// into a per-station histogram.
fn lane_k_variant(
    lane: &str,
) -> (
    onebrc_probe::TempPath,
    onebrc_probe::NameHash,
    onebrc_probe::Agg,
) {
    use onebrc_probe::{Agg, NameHash, TempPath};
    match lane {
        "k" => (TempPath::Parse, NameHash::Fnv, Agg::Running),
        "kt" => (TempPath::Table, NameHash::Fnv, Agg::Running),
        "km" => (TempPath::Swar, NameHash::Fnv, Agg::Running),
        "kw" => (TempPath::Parse, NameHash::Word, Agg::Running),
        "kmw" => (TempPath::Swar, NameHash::Word, Agg::Running),
        "kh" => (TempPath::Parse, NameHash::Fnv, Agg::Histogram),
        "kmwh" => (TempPath::Swar, NameHash::Word, Agg::Histogram),
        #[cfg(feature = "codec9")]
        "kmwc" => (TempPath::Swar, NameHash::Word, Agg::Codec9),
        #[cfg(not(feature = "codec9"))]
        "kmwc" => {
            eprintln!("lane kmwc requires --features codec9");
            std::process::exit(1);
        }
        other => unreachable!("lane K variant {other}"),
    }
}

/// `dist <path> [workers]`: lane K's histogram path over an mmapped file,
/// printing min, p10, median, p90 and max (tenths) for the first and last
/// 3 stations, plus the open-to-result time in ns per row.
#[cfg(unix)]
fn cmd_dist(args: &[String]) {
    let path = PathBuf::from(args.first().expect("usage: dist <path> [workers]"));
    let workers: usize = args
        .get(1)
        .map(|s| s.parse().expect("workers must be a usize"))
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    let start = Instant::now();
    let m = onebrc_probe::lane_k::Mapped::open(&path).expect("map corpus file");
    let agg = match args.get(2).map(String::as_str) {
        None | Some("hist") => onebrc_probe::Agg::Histogram,
        #[cfg(feature = "codec9")]
        Some("c9") => onebrc_probe::Agg::Codec9,
        Some(other) => {
            eprintln!("unknown table '{other}' (expected hist, or c9 with --features codec9)");
            std::process::exit(2);
        }
    };
    let dist = onebrc_probe::lane_k_distributions_agg(
        m.as_slice(),
        workers,
        onebrc_probe::TempPath::Swar,
        onebrc_probe::NameHash::Word,
        agg,
    );
    let rows: u64 = dist.values().map(|d| d.count()).sum();
    let ns_per_row = start.elapsed().as_nanos() as f64 / rows.max(1) as f64;
    println!(
        "dist rows={rows} workers={workers} stations={} ns_per_row={ns_per_row:.3}",
        dist.len()
    );
    let line = |name: &str, d: &onebrc_probe::Distribution| {
        let q = |p| d.percentile(p).unwrap();
        println!(
            "  {name}: min {} p10 {} median {} p90 {} max {}",
            q(0),
            q(10),
            q(50),
            q(90),
            q(100)
        );
    };
    for (name, d) in dist.iter().take(3) {
        line(name, d);
    }
    for (name, d) in dist.iter().rev().take(3) {
        line(name, d);
    }
}

#[cfg(not(unix))]
fn cmd_dist(_: &[String]) {
    eprintln!("dist needs a Unix mmap");
    std::process::exit(1);
}
