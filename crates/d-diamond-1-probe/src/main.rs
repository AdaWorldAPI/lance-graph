//! D-DIAMOND-1 — run the four arms at N = 1M and print the report.
//! `cargo run --release --manifest-path crates/d-diamond-1-probe/Cargo.toml`

use d_diamond_1_probe::*;
use lance_graph_contract::facet::SemanticPrefix;
use lance_graph_contract::ordered_lane::digest_of;
use lance_graph_mask_risc::words_for;
use std::sync::{Arc, RwLock};

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    println!("D-DIAMOND-1 probe — N = {n}, seed = {SEED:#x}, p_dep = 0.7");
    println!(
        "rustc: {}",
        option_env!("RUSTC_VERSION").unwrap_or("(see rustc --version)")
    );
    let mut r = SplitMix64(SEED ^ 0xD1A_0001);
    let world = build_world(n, SEED, 0.7);
    println!(
        "seal (sort + attest) of {n} keys: {:.1} ms",
        world.seal_ns / 1e6
    );

    // ── P1 ──
    println!("\n== P1 — point universe: 8-tile LCP, ns/pair (min of 7 over 64K pairs), is_ancestor at depth(a)=3 ==");
    println!(
        "{:<30} {:>10} {:>12} {:>12} {:>10}",
        "pair class", "tzcnt+swap", "peek u16", "peek bytes", "anc frac"
    );
    for row in run_p1(world.lane.keys(), 65_536, 3, &mut r) {
        println!(
            "{:<30} {:>10.2} {:>12.2} {:>12.2} {:>10.3}",
            row.class.label(),
            row.tzcnt_ns,
            row.peek_u16_ns,
            row.peek_bytes_ns,
            row.ancestor_frac
        );
    }

    // ── P2 ──
    println!("\n== P2 — field universe: witnessed bound + TOUCHED-ONLY write (dst sized to words_for(hi), not words_for(N)) vs MatchU64 sweep (ns, min of 7) ==");
    println!(
        "{:>9} {:>5} {:>9} {:>8} | {:>10} {:>10} {:>10} | {:>10} {:>8}",
        "N", "depth", "kept", "kept%", "bound", "touch-wr", "bound+wr", "ref-sweep", "speedup"
    );
    let sizes = [128usize, 256, 512, 1_000, 4_000, 16_000, 64_000, 256_000, n];
    let depths = [1u8, 2, 3, 4, 5, 6, 7];
    let mut crossover: Vec<(u8, Option<usize>)> = depths.iter().map(|&d| (d, None)).collect();
    for &sz in &sizes {
        let lane = subsample(&world.lane, sz);
        let w = lane.witness();
        let prefixes = pick_prefixes(&lane, &w, &depths, &mut r);
        for row in run_p2(&lane, &w, &prefixes, None) {
            println!(
                "{:>9} {:>5} {:>9} {:>7.2}% | {:>10.0} {:>10.0} {:>10.0} | {:>10.0} {:>8.2}x",
                row.n,
                row.depth,
                row.kept,
                100.0 * row.kept as f64 / row.n as f64,
                row.bound_ns,
                row.touched_write_ns,
                row.bound_fold_total(),
                row.reference_sweep_ns,
                row.speedup()
            );
            if row.speedup() > 1.0 {
                if let Some(c) = crossover.iter_mut().find(|(d, _)| *d == row.depth) {
                    if c.1.is_none() {
                        c.1 = Some(row.n);
                    }
                }
            }
        }
    }
    println!(
        "crossover N (smallest N where bound+write < sweep), per depth: {:?}",
        crossover
    );

    println!("\n-- P2 cache regime at N = {n} and N = 64K: L2-resident vs L2-evicted (64 MiB stream before each round; L3 = 260 MiB is NOT evicted) --");
    let mut scratch = vec![0u64; 8 * 1024 * 1024];
    for &sz in &[64_000usize, n] {
        let lane = subsample(&world.lane, sz);
        let w = lane.witness();
        let prefixes = pick_prefixes(&lane, &w, &[2u8, 4, 6], &mut r);
        let resident = run_p2(&lane, &w, &prefixes, None);
        let evicted = run_p2(&lane, &w, &prefixes, Some(&mut scratch));
        for (a, b) in resident.iter().zip(&evicted) {
            println!(
                "N={:>8} d={} kept={:>7} | resident: bound {:>7.0} touch-wr {:>7.0} ref-sweep {:>8.0} | evicted: bound {:>7.0} touch-wr {:>7.0} ref-sweep {:>8.0}",
                a.n, a.depth, a.kept, a.bound_ns, a.touched_write_ns, a.reference_sweep_ns,
                b.bound_ns, b.touched_write_ns, b.reference_sweep_ns
            );
        }
    }

    println!("\n-- P2 touched-write flatness: FIXED ABSOLUTE range [500, 600) — same lo/hi regardless of the lane's n_rows — must NOT grow as n_rows grows (proves cost depends on (lo,hi), never on N) --");
    println!("(n_rows is passed here ONLY to size the OLD, buggy whole-lane buffer for comparison — touched_write's own signature never takes n_rows)");
    const FLAT_LO: u32 = 500;
    const FLAT_HI: u32 = 600;
    for &sz in &[1_000u32, 16_000, 256_000, n as u32] {
        let touched_ns = time_ns(9, 200, || {
            let d = touched_write(std::hint::black_box(FLAT_LO), std::hint::black_box(FLAT_HI));
            std::hint::black_box(&d);
        });
        let old_whole_lane_ns = time_ns(9, 200, || {
            let mut dst = vec![0u64; words_for(sz as usize)];
            ndarray::simd::mask_set_range(
                std::hint::black_box(&mut dst),
                FLAT_LO as usize,
                FLAT_HI as usize,
            );
            std::hint::black_box(&dst);
        });
        println!(
            "N={:>9} range=[{:>3},{:>3}) touched_write_ns={:>8.2} | old_whole_lane_sized_ns={:>8.2}",
            sz, FLAT_LO, FLAT_HI, touched_ns, old_whole_lane_ns
        );
    }

    // ── P3 ──
    println!("\n== P3 — fold intersection over ONE ordinal via a Morton-interleaved joint key ==");
    println!("(equal depths only — a joint key packs 2 tiles·16 bits per depth, capped at JOINT_MAX_DEPTH=4 to fit a u128)");
    match tenant_attest_over_ontology_ordinal(&world) {
        Err(e) => println!("finding: the tenant lane cannot be attested over the ontology ordinal → no second witnessed bound: {e}"),
        Ok(_) => println!("UNEXPECTED: tenant lane attested as ordered over the ontology ordinal"),
    }
    let (joint, joint_build_ns) = JointIndex::build(&world);
    println!(
        "joint index build (sort {n} interleaved keys): {:.1} ms",
        joint_build_ns / 1e6
    );
    println!(
        "{:>4} {:>9} {:>9} {:>9} | {:>12} {:>22} {:>8}",
        "d", "kept A", "kept B", "kept ∩", "fold ns", "ref (2 sweeps+AND) ns", "speedup"
    );
    for d in [2u8, 3, 4] {
        match run_p3(&world, &joint, d, &mut r) {
            Some(row) => println!(
                "{:>4} {:>9} {:>9} {:>9} | {:>12.0} {:>22.0} {:>8.2}x",
                row.depth,
                row.kept_a,
                row.kept_b,
                row.kept_and,
                row.fold_ns,
                row.reference_two_sweeps_ns,
                row.reference_two_sweeps_ns / row.fold_ns
            ),
            None => println!("{d:>4}  (no F4-passing pair found)"),
        }
    }

    // ── P4 ──
    println!("\n== P4 — sealed reader vs open writer ==");
    let published: Published = Arc::new(RwLock::new(world.lane.clone()));
    let pinned = published.read().unwrap().clone(); // the reader's at(version) snapshot
    let w = pinned.witness();
    let digest_before = digest_of(pinned.keys());
    let pairs = make_pairs(pinned.keys(), PairClass::Unrelated, 65_536, &mut r);
    let prefixes: Vec<SemanticPrefix> = pick_prefixes(&pinned, &w, &[2u8, 3, 4, 5, 6], &mut r)
        .into_iter()
        .map(|(p, _)| p)
        .collect();
    const READ_MS: u64 = 4000;
    let (peek0, bound0) = reader_measure(&pinned, &w, &pairs, &prefixes, READ_MS);
    println!(
        "no writer   ({:>3} rounds): sealed peek ns/pair min {:.2} med {:.2} p90 {:.2} | sealed bound ns/prefix min {:.0} med {:.0} p90 {:.0}",
        peek0.rounds, peek0.min, peek0.median, peek0.p90, bound0.min, bound0.median, bound0.p90
    );
    let ((peek1, bound1), stats) =
        with_open_writer(published.clone(), 20_000, SEED ^ 0xBEEF, || {
            reader_measure(&pinned, &w, &pairs, &prefixes, READ_MS)
        });
    println!(
        "open writer ({:>3} rounds): sealed peek ns/pair min {:.2} med {:.2} p90 {:.2} | sealed bound ns/prefix min {:.0} med {:.0} p90 {:.0}",
        peek1.rounds, peek1.min, peek1.median, peek1.p90, bound1.min, bound1.median, bound1.p90
    );
    println!(
        "ratios writer/no-writer: peek median {:.3}x p90 {:.3}x | bound median {:.3}x p90 {:.3}x",
        peek1.median / peek0.median,
        peek1.p90 / peek0.p90,
        bound1.median / bound0.median,
        bound1.p90 / bound0.p90
    );
    let digest_after = digest_of(pinned.keys());
    println!(
        "pinned image unchanged: {} (digest {:#x})",
        digest_before == digest_after && pinned.validate(&w).is_ok(),
        digest_after
    );
    let med = |v: &Vec<f64>| {
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if s.is_empty() {
            0.0
        } else {
            s[s.len() / 2]
        }
    };
    println!(
        "writer: {} seals, {} appended (batches of 20K, arrival order random); median sort {:.1} ms, attest {:.1} ms, publish {:.0} ns; published version now {}",
        stats.seals, stats.appended, med(&stats.sort_ns) / 1e6, med(&stats.attest_ns) / 1e6, med(&stats.publish_ns), published.read().unwrap().version()
    );
    // A reader that takes a fresh snapshot per query pays the read-lock + Arc clone:
    let snap_ns = time_ns(7, 1000, || {
        let s = published.read().unwrap().clone();
        std::hint::black_box(s.version());
    });
    println!(
        "fresh-snapshot cost (read lock + Arc clone): {:.0} ns",
        snap_ns
    );
}
