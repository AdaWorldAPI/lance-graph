//! Cost breakdown of `SealedFacetLane::seal`, so "is the seal worth it" is
//! answered with numbers instead of intuition.
//!
//! Three write-side parts, timed separately:
//!   sort            — O(N log N), and separately what it costs on a lane
//!                     that is ALREADY in order (a write-in-order producer)
//!   first_inversion — O(N) verification that the order actually holds
//!   digest_of       — O(N * 16) FNV over every stored byte
//!
//! Plus the two read-side costs: `validate` (O(1)) and `verify` (O(n)).
use d_diamond_1_probe::*;
use lance_graph_contract::facet::FacetCascade;
use lance_graph_contract::ordered_lane::{digest_of, first_inversion, SealedFacetLane};
use std::time::Instant;

fn ms(mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_nanos() as f64 / 1e6);
    }
    best
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let world = build_world(n, SEED, 0.7);
    let sorted: Vec<FacetCascade> = world.lane.keys().to_vec();
    let mut shuffled = sorted.clone();
    let mut r = SplitMix64(SEED ^ 0x5EA1);
    for i in (1..shuffled.len()).rev() {
        shuffled.swap(i, (r.next_u64() % (i as u64 + 1)) as usize);
    }

    // The clone is in every arm below; subtract it so the numbers are the
    // operation, not the allocation.
    let clone_ms = ms(|| {
        let k = sorted.clone();
        std::hint::black_box(&k);
    });

    println!("N = {n}   (clone baseline {clone_ms:.2} ms, subtracted from the two sort rows)");
    println!(
        "sort, shuffled input       {:>9.2} ms",
        ms(|| {
            let mut k = shuffled.clone();
            k.sort_unstable_by(FacetCascade::cmp_numeric_projection);
            std::hint::black_box(&k);
        }) - clone_ms
    );
    println!(
        "sort, ALREADY in order     {:>9.2} ms   <- what a write-in-order producer pays",
        ms(|| {
            let mut k = sorted.clone();
            k.sort_unstable_by(FacetCascade::cmp_numeric_projection);
            std::hint::black_box(&k);
        }) - clone_ms
    );
    println!(
        "first_inversion (verify)   {:>9.2} ms",
        ms(|| {
            std::hint::black_box(first_inversion(&sorted));
        })
    );
    println!(
        "digest_of                  {:>9.2} ms",
        ms(|| {
            std::hint::black_box(digest_of(&sorted));
        })
    );

    let lane = SealedFacetLane::attest_sorted(sorted.clone(), 1).unwrap();
    let w = lane.witness();
    println!(
        "validate (read side, O(1)) {:>9.5} ms",
        ms(|| {
            lane.validate(&w).unwrap();
        })
    );
    println!(
        "verify   (read side, O(n)) {:>9.2} ms",
        ms(|| {
            lane.verify().unwrap();
        })
    );
}
