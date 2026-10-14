//! D-WAL-2 / D-WAL-3 — cost and soak of `LanceCycleWriter` (L) against the
//! lancedb 0.40 MemWAL write path (M). Plan: `.claude/plans/wal-shootout-v1.md`.
//!
//! ```text
//! # cost table (D-WAL-2)
//! cargo run --release -p lance-graph --features lancedb-sdk --example wal_shootout -- latency <scratch-dir>
//! # soak, one process per arm (D-WAL-3)
//! cargo run --release -p lance-graph --features lancedb-sdk --example wal_shootout -- \
//!     soak <L|M> <dir> <minutes> <rows-per-cycle> <interval-ms> <csv>
//! ```

#[path = "../tests/wal_common/mod.rs"]
mod wal_common;

use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use lance_graph::graph::cycle_sink::LanceCycleWriter;
use wal_common::*;

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

fn free_bytes(dir: &Path) -> u64 {
    // statvfs through `df` keeps this dependency-free.
    let out = std::process::Command::new("df")
        .args(["-B1", "--output=avail"])
        .arg(dir)
        .output()
        .ok();
    out.and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().nth(1)?.trim().parse().ok())
        .unwrap_or(u64::MAX)
}

/// One arm's writer.
enum Arm {
    L(LanceCycleWriter, String),
    M(lancedb::Table, String),
}

impl Arm {
    async fn open(kind: &str, dir: &str) -> Self {
        match kind {
            "L" => {
                let p = format!("{dir}/cycles.lance");
                Arm::L(LanceCycleWriter::open(p.clone()).await.expect("open L"), p)
            }
            k if k.starts_with('M') => {
                let flush: u64 = k[1..].parse().unwrap_or(100);
                Arm::M(m_create(dir, flush).await, dir.to_string())
            }
            other => panic!("arm {other}"),
        }
    }

    async fn commit(&mut self, cycle: u64, n: usize) {
        match self {
            Arm::L(w, _) => l_commit(w, cycle, n).await,
            Arm::M(t, _) => m_put(t, &frozen(cycle, 0, n)).await,
        }
    }

    /// Rows readable now, and the scan time. L: a fresh open at the latest
    /// version. M: the LSM read of this process (memtable + generations).
    async fn scan(&self) -> (usize, Duration) {
        let t0 = Instant::now();
        let rows = match self {
            Arm::L(_, p) => l_rows_by_cycle(p).await.unwrap_or_default().values().sum(),
            Arm::M(t, _) => m_rows_by_cycle(t, true).await.values().sum(),
        };
        (rows, t0.elapsed())
    }

    /// Lance versions of the L store (M: versions of the base table, which the
    /// MemWAL path does not advance until compaction).
    async fn versions(&self) -> u64 {
        match self {
            Arm::L(w, _) => w.head().0,
            Arm::M(t, _) => t.version().await.unwrap_or(0),
        }
    }
}

async fn latency(root: &str) {
    println!(
        "tier: avx512f={} | host I/O bound: numbers are per cycle (µs) and per stored row (ns)",
        cfg!(target_feature = "avx512f")
    );
    println!("arm  N      cycles  p50_us    p99_us    ns/row   rows      MiB     files  versions  scan_ms");
    for (n, cycles) in [(64usize, 200u64), (1024, 200), (8192, 40)] {
        for kind in ["L", "M10", "M100"] {
            let dir = format!("{root}/lat-{kind}-{n}");
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let mut arm = Arm::open(kind, &dir).await;
            let mut lat = Vec::with_capacity(cycles as usize);
            for c in 1..=cycles {
                let t0 = Instant::now();
                arm.commit(c, n).await;
                lat.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (rows, scan) = arm.scan().await;
            assert_eq!(rows, rows_per_cycle(n) * cycles as usize, "{kind} rows");
            let (bytes, files) = disk_usage(Path::new(&dir));
            let p50 = pct(&lat, 0.5);
            println!(
                "{kind:<4} {n:<6} {cycles:<7} {p50:<9.0} {:<9.0} {:<8.1} {rows:<9} {:<7.1} {files:<6} {:<9} {:.1}",
                pct(&lat, 0.99),
                p50 * 1e3 / rows_per_cycle(n) as f64,
                bytes as f64 / (1 << 20) as f64,
                arm.versions().await,
                scan.as_secs_f64() * 1e3,
            );
            drop(arm);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

async fn soak(kind: &str, dir: &str, minutes: u64, n: usize, interval_ms: u64, csv: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv)
        .expect("csv");
    writeln!(
        out,
        "t_s,arm,cycles,commit_p50_us,commit_p99_us,commit_max_us,disk_mib,files,versions,scan_rows,scan_ms,rss_mib"
    )
    .unwrap();
    let mut arm = Arm::open(kind, dir).await;
    let start = Instant::now();
    let end = start + Duration::from_secs(minutes * 60);
    let mut next_sample = start + Duration::from_secs(30);
    let mut window: Vec<f64> = Vec::new();
    let mut cycle = 0u64;
    while Instant::now() < end {
        let tick = Instant::now();
        cycle += 1;
        arm.commit(cycle, n).await;
        window.push(tick.elapsed().as_secs_f64() * 1e6);
        if Instant::now() >= next_sample {
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (bytes, files) = disk_usage(Path::new(dir));
            let (rows, scan) = arm.scan().await;
            writeln!(
                out,
                "{:.0},{kind},{cycle},{:.0},{:.0},{:.0},{:.1},{files},{},{rows},{:.1},{:.1}",
                start.elapsed().as_secs_f64(),
                pct(&window, 0.5),
                pct(&window, 0.99),
                window.last().copied().unwrap_or(f64::NAN),
                bytes as f64 / (1 << 20) as f64,
                arm.versions().await,
                scan.as_secs_f64() * 1e3,
                rss_kib() as f64 / 1024.0,
            )
            .unwrap();
            out.flush().unwrap();
            window.clear();
            next_sample += Duration::from_secs(30);
            if free_bytes(Path::new(dir)) < 1 << 30 {
                writeln!(out, "# stopped early: under 1 GiB free").unwrap();
                break;
            }
        }
        let spent = tick.elapsed();
        if spent < Duration::from_millis(interval_ms) {
            tokio::time::sleep(Duration::from_millis(interval_ms) - spent).await;
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("latency") => latency(args.get(2).expect("scratch dir")).await,
        Some("soak") => {
            soak(
                &args[2],
                &args[3],
                args[4].parse().expect("minutes"),
                args[5].parse().expect("rows"),
                args[6].parse().expect("interval ms"),
                &args[7],
            )
            .await
        }
        _ => eprintln!("usage: wal_shootout latency <dir> | soak <L|M> <dir> <min> <rows> <interval_ms> <csv>"),
    }
}
