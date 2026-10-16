//! Dog-food probe: ruff's MeTTa "lowerable / interpreter-only" verdict,
//! recomputed as presence-bit algebra through quack and mask-risc.
//!
//! `ruff_metta_spo` classifies every rule head of a MeTTa corpus: a head is
//! interpreter-only when its own body uses a native (or a variable head), or
//! when it calls an interpreter-only head; the second rule runs to a fixpoint
//! over the call graph. ruff computes that fixpoint in Rust over maps. Here
//! the same fixpoint runs over our own substrate, and ruff's verdict is the
//! oracle.
//!
//! # Layout
//!
//! - Every rule head gets one coordinate: its index in sorted order. Names are
//!   labels, read once at ingest and never seen by the algebra.
//! - The call graph is ONE presence plane `E` over `n · S` addresses: the bit
//!   at `h · S + g` says head `h` calls head `g`. `S` is `n` rounded up to whole
//!   words, so every row starts on a word.
//! - The taint plane `T` (one bit per head) is tiled into every row segment,
//!   a word copy, giving `T_rep` with bit `h · S + g` = `T[g]`.
//! - One step: head `h` becomes tainted when `And(Plane(E), Plane(T_rep))`
//!   has `Any` bit in row `h`'s extent. The callee bit and the taint bit of
//!   callee `g` sit at the same coordinate, so the step reads nothing at a
//!   position taken from another lane's values.
//! - Iterate until `T` stops changing.
//!
//! # Seed (stated, because it is borrowed)
//!
//! The direct taint (a head's own body uses a native) is not recomputed: it is
//! read from ruff's dump (`direct = 1`). What this probe tests is the CLOSURE.
//! The disable arm below proves the closure is load-bearing.
//!
//! # Input
//!
//! A directory written by
//! `cargo run --manifest-path crates/ruff_metta_spo/Cargo.toml --example metta_census -- --dump <dir> <corpus>`
//! in the ruff repository (`heads.tsv`, `calls.tsv`).
//!
//! ```text
//! cargo run -p lance-graph-quack --example ruff_taint_probe -- <dump-dir>...
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{words_for, Foreign, Out, Planes, Program, Value};
use lance_graph_quack::{lower, Agg, Filter, Mask, Query};

struct Dump {
    /// Sorted head labels; the index is the coordinate.
    labels: Vec<String>,
    direct: Vec<bool>,
    lowerable: Vec<bool>,
    /// `(caller, callee)` coordinates.
    calls: Vec<(usize, usize)>,
}

fn read_dump(dir: &Path) -> Dump {
    let heads = std::fs::read_to_string(dir.join("heads.tsv")).expect("heads.tsv");
    let mut rows: Vec<(String, bool, bool)> = heads
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 3, "heads.tsv line {l:?}");
            (f[0].to_string(), f[1] == "1", f[2] == "1")
        })
        .collect();
    rows.sort();
    let labels: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    let coord: BTreeMap<&str, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), i))
        .collect();
    let calls_text = std::fs::read_to_string(dir.join("calls.tsv")).expect("calls.tsv");
    let calls = calls_text
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 2, "calls.tsv line {l:?}");
            (coord[f[0]], coord[f[1]])
        })
        .collect();
    Dump {
        direct: rows.iter().map(|r| r.1).collect(),
        lowerable: rows.iter().map(|r| r.2).collect(),
        labels,
        calls,
    }
}

fn set(words: &mut [u64], i: usize) {
    words[i / 64] |= 1 << (i % 64);
}

fn get(words: &[u64], i: usize) -> bool {
    words[i / 64] >> (i % 64) & 1 == 1
}

/// The closure over `iterations` steps (`None` = to the fixpoint). Returns the
/// taint plane and the number of steps that changed it.
fn closure(d: &Dump, seed: &[bool], iterations: Option<usize>) -> (Vec<u64>, usize) {
    let n = d.labels.len();
    let w = words_for(n);
    let stride = w * 64;
    let n_rows = n * stride;

    // E: the call presence plane. Built once, at ingest.
    let mut e = vec![0u64; words_for(n_rows)];
    for &(h, g) in &d.calls {
        set(&mut e, h * stride + g);
    }

    let mut t = vec![0u64; w];
    for (i, &s) in seed.iter().enumerate() {
        if s {
            set(&mut t, i);
        }
    }

    // And(E, T_rep), Any — lowered once, run per row extent.
    let program: Program = lower(&Query {
        filter: Filter::And(vec![Filter::Plane(Mask(0)), Filter::Plane(Mask(1))]),
        agg: Agg::Any,
    })
    .expect("lowers");

    let mut changed_steps = 0;
    let mut t_rep = vec![0u64; words_for(n_rows)];
    loop {
        if iterations.is_some_and(|k| changed_steps >= k) {
            break;
        }
        // Tile T into every row segment: a positional word copy.
        for h in 0..n {
            t_rep[h * w..(h + 1) * w].copy_from_slice(&t);
        }
        let masks: [&[u64]; 2] = [&e, &t_rep];
        let planes = Planes {
            n_rows,
            masks: &masks,
            lanes: &[],
        };
        let mut next = t.clone();
        for h in 0..n {
            if get(&t, h) {
                continue;
            }
            let mut s = Scratch::for_program(&program, n_rows).expect("scratch");
            let v = execute_extent(
                &program,
                &planes,
                &Foreign::NONE,
                &mut s,
                Out::None,
                h * stride..(h + 1) * stride,
            )
            .expect("runs");
            if matches!(v, Value::Bool(true)) {
                set(&mut next, h);
            }
        }
        if next == t {
            break;
        }
        t = next;
        changed_steps += 1;
    }
    (t, changed_steps)
}

fn mismatches(d: &Dump, t: &[u64]) -> usize {
    (0..d.labels.len())
        .filter(|&h| get(t, h) == d.lowerable[h])
        .count()
}

fn main() {
    let tier = if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else {
        "not v4"
    };
    println!("tier: {tier}");
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    assert!(!dirs.is_empty(), "pass one or more dump directories");
    for dir in dirs {
        let d = read_dump(Path::new(&dir));
        let n = d.labels.len();
        let by_calls = (0..n).filter(|&h| !d.direct[h] && !d.lowerable[h]).count();

        // 1. To the fixpoint: must equal ruff's verdict on every head.
        let (t, steps) = closure(&d, &d.direct, None);
        let tainted = (0..n).filter(|&h| get(&t, h)).count();
        let miss = mismatches(&d, &t);
        println!(
            "{dir}: heads {n}, call edges {}, seed {}, tainted at fixpoint {tainted} after {steps} steps, mismatches vs ruff {miss}",
            d.calls.len(),
            d.direct.iter().filter(|&&b| b).count()
        );
        assert_eq!(miss, 0, "closure disagrees with ruff");

        // 2. Disable: no closure. The heads tainted only through a call must
        //    now disagree — exactly those, and there must be some.
        let (t0, _) = closure(&d, &d.direct, Some(0));
        let miss0 = mismatches(&d, &t0);
        println!(
            "  disable (0 steps): mismatches {miss0}, heads tainted only through calls {by_calls}"
        );
        assert!(by_calls > 0, "fixture cannot tell closure from seed");
        assert_eq!(miss0, by_calls);

        // 3. Silence: an empty seed taints nothing.
        let (te, _) = closure(&d, &vec![false; n], None);
        assert!((0..n).all(|h| !get(&te, h)), "closure invented taint");
        println!("  silence (empty seed): nothing tainted");
    }
}
