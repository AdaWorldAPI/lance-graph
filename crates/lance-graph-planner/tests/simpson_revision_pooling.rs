//! Probe P-pool (coresearch 2026-10-17, entry
//! `2026-10-17-coresearch-simpson-epistemic.md`): NAL revision of stratum
//! beliefs equals the union recount, so it reproduces Simpson's reversal.
//!
//! A stratum belief "exposure → outcome" carries `f = hits / n` and
//! `c = n / (n + 1)`, so its evidence weight `c / (1 - c)` is `n`. Revision
//! pools `f·w`, the positive evidence, which is the union recount. Nothing in
//! the revision arithmetic looks at strata; the probe pins that the reversal
//! happens, at f32 and through the packed u8 register, and pins the two twins
//! that bound any guard built on it.

use causal_edge::isa::truth;
use causal_edge::CausalEdge64;

/// One stratum cell: `n` units, `hits` of them with the outcome.
#[derive(Clone, Copy)]
struct Cell {
    n: u32,
    hits: u32,
}

/// One stratum: the exposed and the unexposed cell.
#[derive(Clone, Copy)]
struct Stratum {
    exposed: Cell,
    unexposed: Cell,
}

fn f_of(c: Cell) -> f32 {
    c.hits as f32 / c.n as f32
}

fn c_of(c: Cell) -> f32 {
    c.n as f32 / (c.n as f32 + 1.0)
}

/// Direction by exact integer cross-multiplication, as `certification::compare`
/// does: `true` when the outcome is more frequent among the exposed.
fn raises(e: Cell, u: Cell) -> bool {
    u64::from(e.hits) * u64::from(u.n) > u64::from(u.hits) * u64::from(e.n)
}

/// Revise the stratum beliefs of one arm at f32.
fn revise_f32(cells: &[Cell]) -> (f32, f32) {
    let mut acc = (f_of(cells[0]), c_of(cells[0]));
    for &c in &cells[1..] {
        acc = truth::revision(acc.0, acc.1, f_of(c), c_of(c)).expect("evidence");
    }
    acc
}

fn pack(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Revise the stratum beliefs of one arm through the packed CE64 register.
fn revise_u8(cells: &[Cell]) -> u8 {
    let edge = |c: Cell| {
        let mut e = CausalEdge64(0);
        e.set_frequency_u8(pack(f_of(c)));
        e.set_confidence_u8(pack(c_of(c)));
        e
    };
    let mut acc = edge(cells[0]);
    for &c in &cells[1..] {
        acc = acc.revision(edge(c));
    }
    acc.frequency_u8()
}

fn union(cells: &[Cell]) -> Cell {
    cells.iter().fold(Cell { n: 0, hits: 0 }, |a, c| Cell {
        n: a.n + c.n,
        hits: a.hits + c.hits,
    })
}

struct Outcome {
    every_stratum_raises: Vec<bool>,
    union_raises: bool,
    revised_f32: (f32, f32),
    revised_u8: (u8, u8),
}

fn run(strata: &[Stratum]) -> Outcome {
    let e: Vec<Cell> = strata.iter().map(|s| s.exposed).collect();
    let u: Vec<Cell> = strata.iter().map(|s| s.unexposed).collect();
    Outcome {
        every_stratum_raises: strata
            .iter()
            .map(|s| raises(s.exposed, s.unexposed))
            .collect(),
        union_raises: raises(union(&e), union(&u)),
        revised_f32: (revise_f32(&e).0, revise_f32(&u).0),
        revised_u8: (revise_u8(&e), revise_u8(&u)),
    }
}

/// The `certification` Simpson fixture: S0 exposed 6/8, unexposed 2/2;
/// S1 exposed 0/2, unexposed 1/8.
fn simpson() -> [Stratum; 2] {
    [
        Stratum {
            exposed: Cell { n: 8, hits: 6 },
            unexposed: Cell { n: 2, hits: 2 },
        },
        Stratum {
            exposed: Cell { n: 2, hits: 0 },
            unexposed: Cell { n: 8, hits: 1 },
        },
    ]
}

#[test]
fn revising_stratum_beliefs_is_the_union_recount_and_reverses() {
    let o = run(&simpson());
    // Exposure lowers the outcome in every stratum...
    assert_eq!(o.every_stratum_raises, [false, false]);
    // ...and raises it in the union recount (6/10 vs 3/10).
    assert!(o.union_raises);
    // f32 revision equals the union recount exactly.
    assert!((o.revised_f32.0 - 0.6).abs() < 1e-6, "{:?}", o.revised_f32);
    assert!((o.revised_f32.1 - 0.3).abs() < 1e-6, "{:?}", o.revised_f32);
    // The packed register reverses too.
    assert!(o.revised_u8.0 > o.revised_u8.1, "{:?}", o.revised_u8);
}

/// Silence twin: both strata raise and exposure is spread evenly across
/// them, so pooling must agree with the strata.
#[test]
fn balanced_strata_that_agree_do_not_reverse() {
    let o = run(&[
        Stratum {
            exposed: Cell { n: 5, hits: 4 },
            unexposed: Cell { n: 5, hits: 2 },
        },
        Stratum {
            exposed: Cell { n: 5, hits: 3 },
            unexposed: Cell { n: 5, hits: 1 },
        },
    ]);
    assert_eq!(o.every_stratum_raises, [true, true]);
    assert!(o.union_raises);
    assert!(o.revised_f32.0 > o.revised_f32.1, "{:?}", o.revised_f32);
    assert!(o.revised_u8.0 > o.revised_u8.1, "{:?}", o.revised_u8);
}

/// Counter-case for "pool only at Supports": every stratum raises, but the
/// exposed sit mostly in the low-outcome stratum, so the pool lowers. Agreement
/// alone does not license pooling; exposure balance is needed too.
#[test]
fn agreeing_but_unbalanced_strata_still_reverse() {
    let o = run(&[
        Stratum {
            exposed: Cell { n: 90, hits: 18 },
            unexposed: Cell { n: 10, hits: 1 },
        },
        Stratum {
            exposed: Cell { n: 10, hits: 9 },
            unexposed: Cell { n: 90, hits: 72 },
        },
    ]);
    assert_eq!(o.every_stratum_raises, [true, true]);
    assert!(!o.union_raises);
    assert!(o.revised_f32.0 < o.revised_f32.1, "{:?}", o.revised_f32);
    assert!(o.revised_u8.0 < o.revised_u8.1, "{:?}", o.revised_u8);
}
