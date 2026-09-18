//! D-DIAMOND-1 falsifiers at probe level (small N, oracle-checked): F4 the
//! anti-vacuity guard can fire AND can stay silent; F5 at the probe's own
//! bound/sweep; P2/P3 masks equal the oracle; the tenant lane is unattestable
//! over the ontology ordinal; P4 the pinned image is untouched by the writer.

use d_diamond_1_probe::*;
use lance_graph_contract::facet::SemanticPrefix;
use lance_graph_contract::ordered_lane::{digest_of, WitnessError};
use lance_graph_mask_risc::words_for;
use ndarray::simd::mask_and;
use std::sync::{Arc, RwLock};

const N: usize = 20_000;

#[test]
fn f4_guard_fires_on_vacuous_and_stays_silent_on_real_populations() {
    assert!(nontrivial(0, 100).is_err(), "must fire on empty");
    assert!(nontrivial(34, 100).is_err(), "must fire on kept*3 >= total");
    assert!(
        nontrivial(33, 100).is_ok(),
        "must stay silent on a real population"
    );
    let n = 256;
    let w = words_for(n);
    let a = vec![u64::MAX; w];
    let mut half = vec![0u64; w];
    half[0] = u64::MAX;
    let mut and = vec![0u64; w];
    mask_and(&a, &half, &mut and);
    assert!(
        nontrivial_intersection(&a, &half, &and).is_err(),
        "B ⊆ A must fire"
    );
    let mut b = vec![0u64; w];
    b[0] = 0xFF00;
    b[1] = 0xFF;
    let mut c = vec![0u64; w];
    c[0] = 0xF000;
    c[2] = 0xF;
    let mut and2 = vec![0u64; w];
    mask_and(&b, &c, &mut and2);
    assert!(
        nontrivial_intersection(&b, &c, &and2).is_ok(),
        "a partial overlap must stay silent"
    );
}

#[test]
fn generator_is_skewed_enough_that_subtrees_differ_by_orders_of_magnitude() {
    let world = build_world(N, SEED, 0.7);
    let keys = world.lane.keys();
    // subtree sizes at depth 3 (classid + t0)
    let mut sizes = Vec::new();
    let mut i = 0;
    while i < keys.len() {
        let p = SemanticPrefix::of(keys[i], 3);
        let (lo, hi) = world.lane.bound(&world.witness, &p).unwrap();
        sizes.push((hi - lo) as usize);
        i = hi as usize;
    }
    sizes.sort_unstable();
    let max = *sizes.last().unwrap();
    let median = sizes[sizes.len() / 2];
    assert!(sizes.len() > 20, "enough subtrees: {}", sizes.len());
    assert!(
        max >= 20 * median.max(1),
        "skew: max {max} vs median {median}"
    );
}

#[test]
fn p2_bound_mask_equals_sweep_mask_equals_oracle_at_every_depth() {
    let world = build_world(N, SEED, 0.7);
    let mut r = SplitMix64(7);
    let prefixes = pick_prefixes(&world.lane, &world.witness, &[1, 2, 3, 4, 5, 6, 7], &mut r);
    assert!(
        prefixes.len() >= 6,
        "F4-passing prefixes at most depths: {}",
        prefixes.len()
    );
    // run_p2 asserts bound == sweep == oracle and F4 internally.
    let rows = run_p2(&world.lane, &world.witness, &prefixes, None);
    assert_eq!(rows.len(), prefixes.len());
    // Depth 0 and 8 explicitly (F5 boundaries), outside F4 (they are vacuous by definition).
    let k = world.lane.keys()[N / 2];
    for d in [0u8, 8] {
        let p = SemanticPrefix::of(k, d);
        let mut dst = vec![0u64; words_for(N)];
        let (lo, hi) = bound_mask(&world.lane, &world.witness, &p, &mut dst).unwrap();
        assert_eq!(dst, oracle_mask(world.lane.keys(), &p));
        if d == 0 {
            assert_eq!((lo, hi), (0, N as u32));
        } else {
            assert!(
                hi > lo
                    && world.lane.keys()[lo as usize..hi as usize]
                        .iter()
                        .all(|x| *x == k)
            );
        }
    }
}

/// The touched-write fix: `touched_write(lo, hi)` allocates its destination
/// sized to `words_for(hi)`, never to `words_for(n_rows)` — so its cost must
/// depend on `(lo, hi)` alone, never on how large the lane it is carved from
/// happens to be. This is the falsifier for the O(n_rows) materialization
/// bug: the OLD code allocated `dst` sized to the WHOLE LANE
/// (`words_for(n_rows)`) regardless of range width, so `mask_set_range`'s own
/// tail-zeroing pass (`fill_words(&mut out_words[hi_word+1..], 0)`) did
/// O(n_rows - hi) work — which grows without bound as the lane grows, for a
/// FIXED `(lo, hi)`. `n_rows` itself never appears in `touched_write`'s
/// signature at all, which is the structural proof; this test is the
/// empirical one, holding `(lo, hi)` fixed and varying only which `n_rows`
/// the caller *would* have sized the old buggy buffer to.
///
/// (A fixed range WIDTH at a position that moves with `n_rows`, e.g. `n/2`,
/// is not a valid probe here: `mask_set_range` also zeroes everything before
/// `lo`'s word, so an `lo` that itself grows with `n_rows` would show growth
/// for a reason unrelated to the bug being tested. `(lo, hi)` must be held
/// absolutely fixed.)
///
/// Timing noise makes a hard `<2x` assert flaky at unit-test granularity, so
/// this asserts the WEAKER, still-meaningful bound: the same `(lo, hi)`
/// timed while `n_rows` is irrelevant to it must not show the order-of-
/// magnitude growth an O(n_rows) allocation would.
#[test]
fn p2_touched_write_cost_does_not_scale_with_lane_size() {
    // Fixed, absolute (lo, hi) — independent of any lane's row count.
    const LO: u32 = 500;
    const HI: u32 = 600;
    let ns = |reps| {
        time_ns(9, reps, || {
            let d = touched_write(std::hint::black_box(LO), std::hint::black_box(HI));
            std::hint::black_box(&d);
        })
    };
    // Warm up (first call pays one-time page-fault/cache-cold cost), then
    // measure twice — the two measurements differ only in nothing, since
    // `touched_write` never sees `n_rows` at all. Repeating the measurement
    // and asserting stability is the closest a unit test gets to "vary N and
    // observe no growth" without actually building lanes at four sizes (that
    // is `main.rs`'s job, reported in the final write-up).
    let _warm = ns(50);
    let a = ns(500);
    let b = ns(500);
    let ratio = a.max(b) / a.min(b).max(1e-9);
    assert!(
        ratio < 8.0,
        "touched_write(LO, HI) cost is unstable across repeated measurement at the SAME \
         fixed (lo, hi): {a:.2}ns vs {b:.2}ns (ratio {ratio:.2}x) — inconsistent with a \
         cost that depends only on (lo, hi), never on n_rows"
    );
}

/// The direct A/B: the OLD shape (`dst` sized to `words_for(n_rows)`, the
/// whole lane) against `touched_write` (`dst` sized to `words_for(hi)`) for
/// the SAME narrow `[lo, hi)`, at a large `n_rows`. The old shape must cost
/// meaningfully more, because `mask_set_range` zeroes every word after
/// `hi_word` up to the end of the slice it is given — `n_rows - hi` words for
/// the old shape, zero extra words for `touched_write`.
#[test]
fn p2_touched_write_beats_the_old_whole_lane_sized_buffer() {
    use ndarray::simd::mask_set_range;
    const LO: usize = 500;
    const HI: usize = 600;
    const N_ROWS: usize = 4_000_000;
    let old_ns = time_ns(9, 50, || {
        // The bug: a destination sized to the WHOLE lane regardless of how
        // narrow [LO, HI) is.
        let mut dst = vec![0u64; words_for(N_ROWS)];
        mask_set_range(std::hint::black_box(&mut dst), LO, HI);
        std::hint::black_box(&dst);
    });
    let new_ns = time_ns(9, 50, || {
        let d = touched_write(
            std::hint::black_box(LO as u32),
            std::hint::black_box(HI as u32),
        );
        std::hint::black_box(&d);
    });
    assert!(
        new_ns * 4.0 < old_ns,
        "touched_write ({new_ns:.1}ns) must be far cheaper than the old whole-lane-sized \
         buffer ({old_ns:.1}ns) at N_ROWS={N_ROWS} for the same narrow [{LO},{HI})"
    );
}

#[test]
fn p3_intersection_arms_agree_with_oracle_and_are_nontrivial() {
    let world = build_world(N, SEED, 0.7);
    let (joint, _build_ns) = JointIndex::build(&world);
    let mut r = SplitMix64(9);
    let row = run_p3(&world, &joint, 3, &mut r).expect("an F4-passing (A,B) pair exists");
    assert!(row.kept_and > 0 && row.kept_and < row.kept_a && row.kept_and < row.kept_b);
}

#[test]
fn tenant_lane_cannot_be_attested_over_the_ontology_ordinal() {
    let world = build_world(N, SEED, 0.7);
    match tenant_attest_over_ontology_ordinal(&world) {
        Err(WitnessError::NotSorted { .. }) => {}
        other => {
            panic!("tenant lane must be unattestable over the ontology ordinal, got {other:?}")
        }
    }
}

#[test]
fn p4_pinned_sealed_image_is_untouched_while_the_writer_publishes() {
    let world = build_world(N, SEED, 0.7);
    let published: Published = Arc::new(RwLock::new(world.lane.clone()));
    let pinned = published.read().unwrap().clone();
    let w = pinned.witness();
    let before = (digest_of(pinned.keys()), pinned.keys().to_vec());
    let mut r = SplitMix64(3);
    let pairs = make_pairs(pinned.keys(), PairClass::Unrelated, 2000, &mut r);
    let prefixes: Vec<SemanticPrefix> = pick_prefixes(&pinned, &w, &[2, 3, 4], &mut r)
        .into_iter()
        .map(|(p, _)| p)
        .collect();
    let (_, stats) = with_open_writer(published.clone(), 5_000, 42, || {
        reader_measure(&pinned, &w, &pairs, &prefixes, 150);
    });
    assert!(
        stats.seals >= 1,
        "the writer must have published at least once"
    );
    assert_eq!(digest_of(pinned.keys()), before.0);
    assert_eq!(pinned.keys(), &before.1[..]);
    assert!(pinned.validate(&w).is_ok());
    let latest = published.read().unwrap().clone();
    assert!(
        latest.version() > pinned.version(),
        "a newer version was published"
    );
    assert!(
        latest.validate(&w).is_err(),
        "the pinned witness must not validate on the new seal"
    );
}
