//! `COUNT(DISTINCT key)` without an intermediate population mask.
//!
//! Two terminals answer it, and which one is legal is a property of the KEY
//! LANE'S LAYOUT, not of the query:
//!
//! - `Terminal::CountKeyRunsU32` — a key-clustered lane (equal keys
//!   contiguous). Two words of state, tile by tile. Proven here against the
//!   oracle and an independent seen-set, across tilings.
//! - `Terminal::ScatterCountU32` — HELD. Its accumulator is one bit per key
//!   of the universe; the pigeonhole falsifier below shows that is the
//!   minimum any exact fold can carry under ARBITRARY row order. That bounds
//!   the wrong traversal, it does not license it: a lane out of key order is
//!   REFUSED (`ExecError::LaneNotOrdered`), never folded through a seen-set
//!   as a fallback. The terminal remains as the falsifier's instrument.

use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::reference::reference_execute_into;
use lance_graph_mask_risc::{
    scratch_words_for, words_for, ExecError, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred,
    Program, Terminal, Value,
};
use std::collections::BTreeSet;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

const S0: Operand = Operand::Scratch(0);

/// `WHERE status == 1` then the distinct fold over `key`.
fn program(terminal: fn(Operand) -> Terminal) -> Program {
    Program::new(
        vec![MaskOp::Pred {
            pred: Pred::EqU32 { lane: 0, v: 1 },
            under: None,
            dst: 0,
        }],
        terminal(S0),
    )
}

fn seen_set(status: &[u32], key: &[u32]) -> usize {
    status
        .iter()
        .zip(key)
        .filter(|(s, _)| **s == 1)
        .map(|(_, k)| *k)
        .collect::<BTreeSet<_>>()
        .len()
}

fn run(program: &Program, planes: &Planes<'_>, tile_words: usize) -> usize {
    let slots = program.scratch_slots as usize;
    let mut buf = vec![0u64; scratch_words_for(tile_words, slots).expect("sized")];
    let mut scratch = Scratch::over(&mut buf, tile_words, slots).expect("carves");
    match execute_into(program, planes, &Foreign::NONE, &mut scratch, Out::None).expect("runs") {
        Value::Count(c) => c,
        other => panic!("expected a count, got {other:?}"),
    }
}

fn oracle(program: &Program, planes: &Planes<'_>) -> usize {
    match reference_execute_into(program, planes, &Foreign::NONE, Out::None).expect("oracle runs") {
        Value::Count(c) => c,
        other => panic!("expected a count, got {other:?}"),
    }
}

#[test]
fn on_a_lane_in_key_order_the_run_fold_is_the_distinct_count_under_every_tiling() {
    let mut seed = 0xD15u64;
    for &n in &[1usize, 63, 64, 65, 500, 4096] {
        let universe = 97u32;
        let mut key: Vec<u32> = (0..n)
            .map(|_| (lcg(&mut seed) % u64::from(universe)) as u32)
            .collect();
        key.sort_unstable(); // key order: the T0 address projection, given
        let status: Vec<u32> = (0..n).map(|_| (lcg(&mut seed) % 3) as u32).collect();
        let lanes = [LaneRef::U32(&status), LaneRef::U32(&key)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let p = program(|m| Terminal::CountKeyRunsU32 { mask: m, lane: 1 });
        let want = seen_set(&status, &key);
        assert_eq!(oracle(&p, &planes), want, "n={n}: oracle vs seen-set");
        for tile_words in [1usize, 2, 8, words_for(n).max(1)] {
            assert_eq!(
                run(&p, &planes, tile_words),
                want,
                "n={n} tile_words={tile_words}"
            );
        }
    }
}

#[test]
fn a_run_split_across_a_tile_edge_is_counted_once_and_an_unhit_run_never() {
    // 130 rows: key 5 spans rows 60..70 (crosses the first 64-row tile
    // edge), selected at rows 61 and 68 only; key 6 is never selected.
    let n = 130usize;
    let key: Vec<u32> = (0..n)
        .map(|i| match i {
            0..=59 => 1,
            60..=69 => 5,
            70..=99 => 6,
            _ => 9,
        })
        .collect();
    let mut status = vec![0u32; n];
    status[61] = 1;
    status[68] = 1;
    status[100] = 1;
    let lanes = [LaneRef::U32(&status), LaneRef::U32(&key)];
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &lanes,
    };
    let p = program(|m| Terminal::CountKeyRunsU32 { mask: m, lane: 1 });
    assert_eq!(
        run(&p, &planes, 1),
        2,
        "keys 5 and 9; key 5 once, key 6 never"
    );
    assert_eq!(oracle(&p, &planes), 2);
}

#[test]
fn a_lane_out_of_key_order_is_refused_by_executor_and_oracle_alike() {
    let mut seed = 0xBADu64;
    let n = 2048usize;
    let universe = 64u32;
    let key: Vec<u32> = (0..n)
        .map(|_| (lcg(&mut seed) % u64::from(universe)) as u32)
        .collect();
    let status: Vec<u32> = (0..n).map(|_| (lcg(&mut seed) % 2) as u32).collect();
    assert!(
        key.windows(2).any(|w| w[1] < w[0]),
        "fixture must be out of order"
    );
    let lanes = [LaneRef::U32(&status), LaneRef::U32(&key)];
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &lanes,
    };
    let runs = program(|m| Terminal::CountKeyRunsU32 { mask: m, lane: 1 });
    let slots = runs.scratch_slots as usize;
    let mut buf = vec![0u64; scratch_words_for(8, slots).expect("sized")];
    let mut scratch = Scratch::over(&mut buf, 8, slots).expect("carves");
    assert_eq!(
        execute_into(&runs, &planes, &Foreign::NONE, &mut scratch, Out::None),
        Err(ExecError::LaneNotOrdered { lane: 1 })
    );
    assert_eq!(
        reference_execute_into(&runs, &planes, &Foreign::NONE, Out::None),
        Err(ExecError::LaneNotOrdered { lane: 1 })
    );
    // The smallest witness of why: 1,2,1 has three runs and two keys.
    let k = [1u32, 2, 1];
    let s1 = [1u32; 3];
    let l = [LaneRef::U32(&s1), LaneRef::U32(&k)];
    let p3 = Planes {
        n_rows: 3,
        masks: &[],
        lanes: &l,
    };
    let mut b3 = vec![0u64; scratch_words_for(1, slots).expect("sized")];
    let mut s3 = Scratch::over(&mut b3, 1, slots).expect("carves");
    assert_eq!(
        execute_into(&runs, &p3, &Foreign::NONE, &mut s3, Out::None),
        Err(ExecError::LaneNotOrdered { lane: 1 })
    );
    assert_eq!(seen_set(&s1, &k), 2);
}

#[test]
fn clustered_but_unsorted_is_refused_by_design() {
    // 3 3 1 1: a run count WOULD be exact here, but the certificate the fold
    // can check in O(1) is order, and 1 < 3 breaks it.
    let k = [3u32, 3, 1, 1];
    let s = [1u32; 4];
    let l = [LaneRef::U32(&s), LaneRef::U32(&k)];
    let planes = Planes {
        n_rows: 4,
        masks: &[],
        lanes: &l,
    };
    let p = program(|m| Terminal::CountKeyRunsU32 { mask: m, lane: 1 });
    let slots = p.scratch_slots as usize;
    let mut buf = vec![0u64; scratch_words_for(1, slots).expect("sized")];
    let mut scratch = Scratch::over(&mut buf, 1, slots).expect("carves");
    assert_eq!(
        execute_into(&p, &planes, &Foreign::NONE, &mut scratch, Out::None),
        Err(ExecError::LaneNotOrdered { lane: 1 })
    );
}

/// The pigeonhole falsifier for "an exact distinct count over an
/// UNCLUSTERED key lane can be folded with less than one bit per key".
///
/// Take a universe of `K` keys. For every pair of DISTINCT key sets
/// `A ≠ B`, build the prefix streams "one selected row per key of A" and
/// "… of B". One of the two sets is not contained in the other — say
/// `A ⊄ B` — and the suffix "one selected row per key of B" then yields
/// `|A ∪ B| > |B|` after prefix A but exactly `|B|` after prefix B. The
/// exact answers differ, so any fold that is exact on every stream must be
/// in DIFFERENT states after the two prefixes; its state distinguishes all
/// `2^K` key sets, hence carries at least `K` bits — one per key of the
/// universe, exactly `ScatterCountU32`'s accumulator. Row order was
/// arbitrary throughout, so no order-free fold escapes the bound; only a
/// lane whose ORDER already encodes the key set (clustering) does, and that
/// is `CountKeyRunsU32`'s precondition, not a smaller state.
#[test]
fn pigeonhole_every_pair_of_key_sets_diverges_under_some_suffix() {
    const K: u32 = 6;
    let subsets = 1u32 << K;
    let keys_of = |set: u32| -> Vec<u32> { (0..K).filter(|k| set >> k & 1 == 1).collect() };
    let exact = |prefix: u32, suffix: u32| -> usize { (prefix | suffix).count_ones() as usize };
    // Run the real terminal on the real streams, not the closure alone: the
    // fold under test must reproduce the divergence — and it does, because
    // its state IS the key set.
    let count_with = |prefix: u32, suffix: u32| -> usize {
        let mut key = keys_of(prefix);
        key.extend(keys_of(suffix));
        let status = vec![1u32; key.len()];
        let lanes = [LaneRef::U32(&status), LaneRef::U32(&key)];
        let planes = Planes {
            n_rows: key.len(),
            masks: &[],
            lanes: &lanes,
        };
        let p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::EqU32 { lane: 0, v: 1 },
                under: None,
                dst: 0,
            }],
            Terminal::ScatterCountU32 {
                mask: S0,
                lane: 1,
                out_rows: K,
            },
        );
        let mut sink = vec![0u64; 1];
        let mut buf = vec![0u64; scratch_words_for(1, 1).expect("sized")];
        let mut scratch = Scratch::over(&mut buf, 1, 1).expect("carves");
        match execute_into(
            &p,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::Mask(&mut sink),
        )
        .expect("runs")
        {
            Value::Count(c) => c,
            other => panic!("{other:?}"),
        }
    };
    let mut pairs = 0usize;
    for a in 0..subsets {
        for b in (a + 1)..subsets {
            // The suffix is the WHOLE of whichever set the other does not
            // contain — that is the stream on which the two prefixes must
            // answer differently.
            let suffix = if a & !b != 0 { b } else { a };
            let (ea, eb) = (exact(a, suffix), exact(b, suffix));
            assert_ne!(
                ea, eb,
                "A={a:06b} B={b:06b} suffix={suffix:06b}: exact counts must differ"
            );
            assert_eq!(count_with(a, suffix), ea);
            assert_eq!(count_with(b, suffix), eb);
            pairs += 1;
        }
    }
    // Every one of the 2^K·(2^K−1)/2 pairs is separated ⇒ ≥ 2^K states ⇒
    // ≥ K bits of state. That is the accumulator width, to the bit.
    assert_eq!(pairs, (subsets as usize) * (subsets as usize - 1) / 2);
    assert!(words_for(K as usize) * 64 >= K as usize);
}
