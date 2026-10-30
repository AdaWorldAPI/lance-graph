//! D-LITHO G4: the dose identity, on the real executor.
//!
//! A per-position count `c(i) < 2^K` (a bag multiplicity, an evidence count)
//! is held as K Boolean planes ("rods"): plane `k` has bit `i` set iff bit `k`
//! of `c(i)` is set. The planes stay Boolean; nothing about a DR cell
//! changes. Under a selection `R`, the total develops as
//!
//! `Σ_{i ∈ R} c(i) = Σ_k 2^k · popcount(R ∧ plane_k)`,
//!
//! one `And` + `Count` program per rod. This is the counted semiring over
//! Boolean planes; the oracle below sums `c(i)` per position.

use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::{
    scratch_words_for, Foreign, MaskOp, Operand, Out, Planes, Program, Terminal, Value,
};

/// One DR64 focus.
const N: usize = 4096;
const WORDS: usize = N / 64;
/// Counts up to 2^K - 1.
const K: usize = 6;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// The rods of `counts`: plane `k` holds bit `k` of every count.
fn rods(counts: &[u32]) -> Vec<Vec<u64>> {
    let mut planes = vec![vec![0u64; WORDS]; K];
    for (i, &c) in counts.iter().enumerate() {
        for (k, plane) in planes.iter_mut().enumerate() {
            if c >> k & 1 == 1 {
                plane[i / 64] |= 1 << (i % 64);
            }
        }
    }
    planes
}

/// `popcount(R ∧ plane_k)` on the executor. Plane 0 is R, plane `1 + k` is rod k.
fn count_under(planes: &Planes<'_>, rod: u16) -> u64 {
    let program = Program::new(
        vec![MaskOp::And {
            a: Operand::Plane(0),
            b: Operand::Plane(1 + rod),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let slots = program.scratch_slots as usize;
    let mut buf = vec![0u64; scratch_words_for(N, slots).expect("sized")];
    let mut scratch = Scratch::over(&mut buf, N, slots).expect("carves");
    match execute_into(&program, planes, &Foreign::NONE, &mut scratch, Out::None).expect("runs") {
        Value::Count(c) => c as u64,
        other => panic!("expected a count, got {other:?}"),
    }
}

/// The developed total: Σ_k 2^k · popcount(R ∧ plane_k).
fn develop(planes: &Planes<'_>, rods_used: &[usize]) -> u64 {
    rods_used
        .iter()
        .map(|&k| (1u64 << k) * count_under(planes, k as u16))
        .sum()
}

fn fixture(seed: u64, density: u64) -> (Vec<u32>, Vec<u64>) {
    let mut s = seed;
    let counts: Vec<u32> = (0..N)
        .map(|_| (xorshift(&mut s) % (1 << K)) as u32)
        .collect();
    let mut sel = vec![0u64; WORDS];
    for i in 0..N {
        if xorshift(&mut s) % 100 < density {
            sel[i / 64] |= 1 << (i % 64);
        }
    }
    (counts, sel)
}

fn oracle(counts: &[u32], sel: &[u64]) -> u64 {
    (0..N)
        .filter(|&i| sel[i / 64] >> (i % 64) & 1 == 1)
        .map(|i| u64::from(counts[i]))
        .sum()
}

/// FAILS IF: the developed total differs from the per-position sum, at any
/// selection density including empty and full.
#[test]
fn the_dose_stack_develops_to_the_per_position_sum() {
    for (seed, density) in [(1, 0), (2, 3), (3, 50), (4, 97), (5, 100)] {
        let (counts, sel) = fixture(seed, density);
        let rods = rods(&counts);
        let masks: Vec<&[u64]> = std::iter::once(sel.as_slice())
            .chain(rods.iter().map(Vec::as_slice))
            .collect();
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &[],
        };
        let all: Vec<usize> = (0..K).collect();
        assert_eq!(
            develop(&planes, &all),
            oracle(&counts, &sel),
            "seed {seed}, density {density}%"
        );
    }
}

/// FAILS IF: leaving one rod out of the develop step still matches the
/// oracle. Every rod must be load-bearing on a non-trivial fixture.
#[test]
fn every_rod_is_load_bearing() {
    let (counts, sel) = fixture(7, 50);
    let rods = rods(&counts);
    let masks: Vec<&[u64]> = std::iter::once(sel.as_slice())
        .chain(rods.iter().map(Vec::as_slice))
        .collect();
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &[],
    };
    let want = oracle(&counts, &sel);
    for drop in 0..K {
        let kept: Vec<usize> = (0..K).filter(|&k| k != drop).collect();
        assert_ne!(develop(&planes, &kept), want, "rod {drop} was not needed");
    }
}
