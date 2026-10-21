//! Translation and query cost, and what one shared index saves over k
//! thoughts. Shared and rebuilt costs are printed on the same basis: per thought
//! (`*_per_thought_*`) and as a total over 64 thoughts (`*_total_k64`). Run: `cd crates/mork-pathmap-probe && cargo run --release --example cost`
//! (add `--features leapfrog` for MORK's leapfrog join).
//!
//! Timings are one run per cell on whatever machine runs this; read them as orders of magnitude.

use std::time::Instant;

use mork_pathmap_probe::{
    Backend, JoinInput, RelationView, Semantics, bind, boundary::mork_join, reference,
};

// No allocation counter: MORK turns on PathMap's `jemalloc` feature, which
// installs a `#[global_allocator]` inside the library, so a second one here
// does not link. That is itself a finding (see the plan's entropy ledger).

/// Deterministic rows: R(x, y) with y in a small domain so the join fans out.
fn rows(n: usize, seed: u32, domain: u32) -> Vec<[u32; 2]> {
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for i in 0..n as u32 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push([i, s % domain]);
    }
    v
}

fn main() {
    println!("mork join: {}", mork_join());
    println!(
        "backend,n_rows,answers,build_us,build_ns_per_row,query_us,shared_per_thought_us_k8,shared_per_thought_us_k64,rebuilt_per_thought_us,shared_total_us_k64,rebuilt_total_us_k64"
    );
    for &n in &[1_000usize, 10_000, 50_000] {
        let domain = (n as u32 / 4).max(1);
        let r = rows(n, 7, domain);
        // S(y, z): one z per y, so |answers| ~ |R|.
        let s: Vec<[u32; 2]> = (0..domain).map(|y| [y, y + 1_000_000]).collect();
        let inp = JoinInput {
            r: RelationView::from_rows(&r, 1),
            s: RelationView::from_rows(&s, 1),
        };
        for backend in [Backend::PathMapZipper, Backend::Mork] {
            let t = Instant::now();
            let mut idx = bind(backend, &inp, Semantics::Set).unwrap();
            let build = t.elapsed();
            let t = Instant::now();
            let got = idx.answers(&inp).unwrap();
            let query = t.elapsed();
            // Every row the engine returns is a derivation the oracle counts.
            for row in got.rows.keys() {
                assert!(
                    reference::matching(&inp, row.map(Some)) > 0,
                    "{backend:?} n={n}: {row:?}"
                );
            }
            let us = |d: std::time::Duration| d.as_secs_f64() * 1e6;
            // One index built once and queried by k thoughts, per thought.
            let shared_per_thought = |k: f64| us(build) / k + us(query);
            let rebuilt_per_thought = us(build) + us(query);
            println!(
                "{backend:?},{n},{},{:.3},{:.0},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
                got.rows.len(),
                us(build),
                build.as_nanos() as f64 / (r.len() + s.len()) as f64,
                us(query),
                shared_per_thought(8.0),
                shared_per_thought(64.0),
                rebuilt_per_thought,
                us(build) + 64.0 * us(query),
                64.0 * rebuilt_per_thought,
            );
        }
    }
}
