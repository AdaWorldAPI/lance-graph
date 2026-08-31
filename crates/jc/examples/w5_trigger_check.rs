//! W5 trigger check — is the PowerSig scalability wave due yet?
//!
//! W5 is DEFERRED BY DESIGN. Its entry criterion was written down in advance
//! precisely so it would be a trigger rather than a judgment call: *the first
//! real stream whose Goursat solve exceeds 1 GiB or 10 s*. This example makes
//! that criterion executable — it measures the shipped solver's actual cost
//! curve and prints the exact path length at which each half fires.
//!
//! Run: `cargo run --release --manifest-path crates/jc/Cargo.toml \
//!       --features hambly-lyons --example w5_trigger_check`

use sigker::signature_kernel_pde;
use std::time::Instant;

fn path(n: usize) -> Vec<Vec<f64>> {
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            vec![
                t + 0.05 * (260.0 * t).cos(),
                0.5 * t + 0.05 * (260.0 * t).sin(),
            ]
        })
        .collect()
}

fn main() {
    const GIB: f64 = (1u64 << 30) as f64;
    const TIME_TRIGGER_S: f64 = 10.0;

    println!(
        "{:>8} {:>12} {:>12} {:>14}",
        "len", "grid MiB", "secs", "ns/cell"
    );
    let mut samples: Vec<(f64, f64)> = Vec::new(); // (len, secs)
    for &n in &[256usize, 512, 1024, 2048, 4096] {
        let (x, y) = (path(n), path(n));
        let t0 = Instant::now();
        let _ = signature_kernel_pde(&x, &y);
        let secs = t0.elapsed().as_secs_f64();
        // The solver allocates one f64 cell per (i, j).
        let cells = (n + 1) as f64 * (n + 1) as f64;
        let mib = cells * 8.0 / (1024.0 * 1024.0);
        println!(
            "{:>8} {mib:>12.2} {secs:>12.4} {:>14.2}",
            n + 1,
            secs / cells * 1e9
        );
        samples.push(((n + 1) as f64, secs));
    }

    // Memory trigger is exact: cells * 8 bytes = 1 GiB.
    let mem_len = (GIB / 8.0).sqrt();

    // Time trigger from the measured ns/cell at the largest sample (the
    // regime that matters), since cost is O(len^2).
    let (last_len, last_secs) = *samples.last().unwrap();
    let per_cell = last_secs / (last_len * last_len);
    let time_len = (TIME_TRIGGER_S / per_cell).sqrt();

    println!("\nTriggers for W5 (PowerSig tile-local Neumann/power-series solves):");
    println!("  memory  1 GiB grid  at path length ~{mem_len:.0}");
    println!("  time    10 s solve  at path length ~{time_len:.0}  (release, this machine)");
    let fires_first = if mem_len < time_len { "memory" } else { "time" };
    println!(
        "  → the {fires_first} half fires first, at length ~{:.0}",
        mem_len.min(time_len)
    );

    // The longest path this workspace's own certification legs construct.
    let longest_in_tree = 4609usize; // W2 triangle at 1536 pts/segment, 3 segments
    println!(
        "\n  longest path constructed anywhere in-tree today: {longest_in_tree} \
         (the W2 depth-infinity converse leg)"
    );
    let fired = longest_in_tree as f64 >= mem_len.min(time_len);
    println!("  TRIGGER FIRED: {fired}");
    if !fired {
        println!(
            "  → W5 stays deferred. It needs a real stream ~{:.0}x longer than \
             anything in-tree; measure-then-pin says this wave waits for the \
             workload, not the paper.",
            mem_len.min(time_len) / longest_in_tree as f64
        );
    }
}
