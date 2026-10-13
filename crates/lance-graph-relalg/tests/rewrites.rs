//! D-RA-1: every rewrite is checked three ways — the rewritten query, the
//! query as written, and a row-at-a-time oracle over the same lanes must give
//! the same answer — and the rewrites that change cost are pinned by the
//! passes they save.

use lance_graph_mask_risc::{words_for, ExecError, LaneRef, Planes};
use lance_graph_r2il::FoldError;
use lance_graph_relalg::{
    normalize, run, run_as_written, Answer, Atom, Cmp, Measure, Query, Rel, RelError,
};

const N: usize = 1_000;

// Lane indexes.
const X: u16 = 0; // i32, (i % 50) - 20: negatives, zero, positives
const Y: u16 = 1; // i32, (i * 7) % 31
const V: u16 = 2; // i32, the summed value
const K: u16 = 3; // u32, i % 5

struct Fixture {
    x: Vec<i32>,
    y: Vec<i32>,
    v: Vec<i32>,
    k: Vec<u32>,
    p0: Vec<u64>,
    p1: Vec<u64>,
}

fn plane(f: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; words_for(N)];
    for r in (0..N).filter(|&r| f(r)) {
        w[r / 64] |= 1 << (r % 64);
    }
    w
}

fn fixture() -> Fixture {
    Fixture {
        x: (0..N).map(|i| (i % 50) as i32 - 20).collect(),
        y: (0..N).map(|i| ((i * 7) % 31) as i32).collect(),
        v: (0..N).map(|i| (i % 13) as i32 - 6).collect(),
        k: (0..N).map(|i| (i % 5) as u32).collect(),
        p0: plane(|r| r % 3 == 0),
        p1: plane(|r| r % 7 < 3),
    }
}

/// Run `f` with the fixture's planes.
fn with_planes<T>(fx: &Fixture, f: impl FnOnce(&Planes<'_>) -> T) -> T {
    let lanes = [
        LaneRef::I32(&fx.x),
        LaneRef::I32(&fx.y),
        LaneRef::I32(&fx.v),
        LaneRef::U32(&fx.k),
    ];
    let masks: [&[u64]; 2] = [&fx.p0, &fx.p1];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &lanes,
    };
    f(&planes)
}

// ── the row oracle ─────────────────────────────────────────────────────────

fn atom_holds(fx: &Fixture, a: Atom, r: usize) -> bool {
    match a {
        Atom::I32 { lane, op, t } => {
            let x = match lane {
                X => fx.x[r],
                Y => fx.y[r],
                V => fx.v[r],
                _ => unreachable!("fixture has no I32 lane {lane}"),
            };
            match op {
                Cmp::Lt => x < t,
                Cmp::Le => x <= t,
                Cmp::Gt => x > t,
                Cmp::Ge => x >= t,
                Cmp::Eq => x == t,
                Cmp::Ne => x != t,
            }
        }
        Atom::U32 { lane, eq, v } => {
            assert_eq!(lane, K);
            (fx.k[r] == v) == eq
        }
    }
}

fn holds(fx: &Fixture, rel: &Rel, r: usize) -> bool {
    match rel {
        Rel::All => true,
        Rel::Empty => false,
        Rel::Plane(p) => {
            let w = if *p == 0 { &fx.p0 } else { &fx.p1 };
            w[r / 64] >> (r % 64) & 1 == 1
        }
        Rel::Atom(a) => atom_holds(fx, *a, r),
        Rel::And(cs) => cs.iter().all(|c| holds(fx, c, r)),
        Rel::Or(cs) => cs.iter().any(|c| holds(fx, c, r)),
        Rel::Not(c) => !holds(fx, c, r),
    }
}

fn oracle(fx: &Fixture, q: &Query) -> (Answer, Vec<u64>) {
    let rows: Vec<usize> = (0..N).filter(|&r| holds(fx, &q.rel, r)).collect();
    let mask = plane(|r| holds(fx, &q.rel, r));
    let vals = |l: u16| -> Vec<i32> {
        let lane = if l == X {
            &fx.x
        } else if l == Y {
            &fx.y
        } else {
            &fx.v
        };
        rows.iter().map(|&r| lane[r]).collect()
    };
    let a = match q.measure {
        Measure::Count => Answer::Count(rows.len() as u64),
        Measure::Sum(l) => Answer::Sum(vals(l).iter().map(|&v| i64::from(v)).sum()),
        Measure::Min(l) => Answer::Min(vals(l).into_iter().min()),
        Measure::Max(l) => Answer::Max(vals(l).into_iter().max()),
        Measure::Keep => Answer::Kept,
    };
    (a, mask)
}

/// One run's result and the mask it kept.
type Ran = (Result<lance_graph_relalg::Run, RelError>, Vec<u64>);

/// Rewritten and as-written answers, each with its kept mask.
fn both(fx: &Fixture, q: &Query) -> (Ran, Ran) {
    with_planes(fx, |planes| {
        let mut k1 = vec![0u64; words_for(N)];
        let mut k2 = vec![0u64; words_for(N)];
        let r1 = run(q, planes, Some(&mut k1));
        let r2 = run_as_written(q, planes, Some(&mut k2));
        ((r1, k1), (r2, k2))
    })
}

/// The one refusal the as-written lowering may give: one fold needs more
/// scratch slots than the dialect has (every negation takes a fresh slot).
fn out_of_slots(e: &RelError) -> bool {
    matches!(
        e,
        RelError::Fold(FoldError::Exec(ExecError::ScratchBufferTooSmall { .. }))
    )
}

// ── a deterministic generator of small descriptions ────────────────────────

struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[(self.next() % xs.len() as u64) as usize]
    }

    fn atom(&mut self) -> Rel {
        if self.next().is_multiple_of(4) {
            let v = (self.next() % 6) as u32;
            return if self.next().is_multiple_of(2) {
                Rel::eq_u32(K, v)
            } else {
                Rel::ne_u32(K, v)
            };
        }
        let lane = self.pick(&[X, X, Y]);
        let op = self.pick(&[Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq, Cmp::Ne]);
        let t = self.pick(&[-21, -20, -5, 0, 1, 7, 15, 29, 30, i32::MIN, i32::MAX]);
        Rel::i32(lane, op, t)
    }

    /// A tree naming at most `budget` atoms (so the as-written lowering fits
    /// the dialect's scratch slots).
    fn rel(&mut self, depth: u32, budget: &mut u32) -> Rel {
        if depth == 0 || *budget <= 1 || self.next().is_multiple_of(3) {
            debug_assert!(*budget > 0);
            // Every leaf costs a scratch slot, planes included.
            *budget = budget.saturating_sub(1);
            return match self.next() % 8 {
                0 => Rel::Plane((self.next() % 2) as u16),
                1 => Rel::Not(Box::new(Rel::Plane((self.next() % 2) as u16))),
                _ => self.atom(),
            };
        }
        match self.next() % 3 {
            0 => Rel::Not(Box::new(self.rel(depth - 1, budget))),
            k => {
                // Stop adding children once the leaf budget is spent.
                let n = 2 + (self.next() % 2) as usize;
                let mut cs = Vec::new();
                while cs.len() < n && *budget > 0 {
                    cs.push(self.rel(depth - 1, budget));
                }
                if k == 1 {
                    Rel::And(cs)
                } else {
                    Rel::Or(cs)
                }
            }
        }
    }
}

const MEASURES: [Measure; 5] = [
    Measure::Count,
    Measure::Sum(V),
    Measure::Min(X),
    Measure::Max(Y),
    Measure::Keep,
];

/// FAILS IF: any rewrite changes which rows a description selects — the
/// rewritten answer, the as-written answer and the row oracle must agree on
/// every measure, kept masks included — or a rewrite is not idempotent.
#[test]
fn rewritten_equals_as_written_equals_row_oracle() {
    let fx = fixture();
    let mut g = Gen(0xD0_0D);
    let mut changed = 0;
    let mut refused = 0;
    for i in 0..400 {
        let mut budget = 3;
        let rel = g.rel(3, &mut budget);
        let norm = normalize(&rel);
        assert_eq!(normalize(&norm), norm, "not idempotent: {rel:?}");
        if norm != rel {
            changed += 1;
        }
        for m in MEASURES {
            let q = Query {
                rel: rel.clone(),
                measure: m,
            };
            let (want, want_mask) = oracle(&fx, &q);
            let ((r1, k1), (r2, k2)) = both(&fx, &q);
            let r1 = r1.unwrap_or_else(|e| panic!("#{i} rewritten {rel:?}: {e:?}"));
            assert_eq!(r1.answer, want, "#{i} rewritten {rel:?} -> {norm:?} {m:?}");
            if m == Measure::Keep {
                assert_eq!(k1, want_mask, "#{i} rewritten keep {rel:?}");
            }
            match r2 {
                Ok(r2) => {
                    assert_eq!(r2.answer, want, "#{i} as written {rel:?} {m:?}");
                    if m == Measure::Keep {
                        assert_eq!(k2, want_mask, "#{i} as written keep {rel:?}");
                    }
                }
                Err(e) if out_of_slots(&e) => refused += 1,
                Err(e) => panic!("#{i} as written {rel:?}: {e:?}"),
            }
        }
    }
    // Anti-vacuity: the generator must reach the rewrites, or this test only
    // proves the identity.
    eprintln!("rewritten {changed}/400 descriptions; as written refused {refused} runs");
    assert!(
        changed * 3 > 400,
        "only {changed}/400 descriptions were rewritten"
    );
    // The as-written arm must still be the comparison most of the time.
    assert!(
        refused * 10 < 400 * MEASURES.len(),
        "as written refused {refused} runs"
    );
}

/// Count both ways against the oracle. The as-written run is `None` when it
/// ran out of scratch slots.
fn count(fx: &Fixture, rel: Rel) -> (lance_graph_relalg::Run, Option<lance_graph_relalg::Run>) {
    let q = Query {
        rel,
        measure: Measure::Count,
    };
    let (want, _) = oracle(fx, &q);
    let ((r1, _), (r2, _)) = both(fx, &q);
    let r1 = r1.expect("rewritten");
    assert_eq!(r1.answer, want);
    let r2 = match r2 {
        Ok(r2) => {
            assert_eq!(r2.answer, want);
            Some(r2)
        }
        Err(e) if out_of_slots(&e) => None,
        Err(e) => panic!("as written: {e:?}"),
    };
    (r1, r2)
}

/// Both runs, the as-written one required.
fn count_both(fx: &Fixture, rel: Rel) -> (lance_graph_relalg::Run, lance_graph_relalg::Run) {
    let (r1, r2) = count(fx, rel);
    (r1, r2.expect("as written ran out of slots"))
}

/// FAILS IF: comparisons on one lane are not fused into one interval.
#[test]
fn comparisons_on_one_lane_fuse_into_one_interval() {
    let rel = Rel::And(vec![
        Rel::i32(X, Cmp::Lt, 30),
        Rel::i32(X, Cmp::Le, 19),
        Rel::i32(X, Cmp::Ge, -5),
        Rel::i32(X, Cmp::Gt, -10),
    ]);
    assert_eq!(
        normalize(&rel),
        Rel::And(vec![Rel::i32(X, Cmp::Lt, 20), Rel::i32(X, Cmp::Gt, -6)])
    );
    let (r1, r2) = count_both(&fixture(), rel);
    eprintln!(
        "passes: rewritten {} vs as written {}",
        r1.passes, r2.passes
    );
    assert!(r1.passes < r2.passes, "{} vs {}", r1.passes, r2.passes);
}

/// FAILS IF: fusion merges comparisons on different lanes.
#[test]
fn comparisons_on_different_lanes_stay_apart() {
    let rel = Rel::And(vec![Rel::i32(X, Cmp::Lt, 5), Rel::i32(Y, Cmp::Lt, 3)]);
    assert_eq!(normalize(&rel).atoms(), 2);
    count(&fixture(), rel);
}

/// FAILS IF: an excluded value at an interval end is not folded into the
/// bound, or a single surviving value is not an equality.
#[test]
fn exclusions_at_the_ends_tighten_the_interval() {
    let base = vec![
        Rel::i32(X, Cmp::Ge, 0),
        Rel::i32(X, Cmp::Le, 3),
        Rel::i32(X, Cmp::Ne, 0),
        Rel::i32(X, Cmp::Ne, 3),
    ];
    assert_eq!(
        normalize(&Rel::And(base.clone())),
        Rel::And(vec![Rel::i32(X, Cmp::Lt, 3), Rel::i32(X, Cmp::Gt, 0)])
    );
    let mut one = base;
    one.push(Rel::i32(X, Cmp::Ne, 1));
    assert_eq!(normalize(&Rel::And(one.clone())), Rel::i32(X, Cmp::Eq, 2));
    count(&fixture(), Rel::And(one));
}

/// FAILS IF: a contradiction is executed instead of answered.
#[test]
fn a_contradiction_is_answered_without_a_program() {
    let fx = fixture();
    for rel in [
        Rel::And(vec![Rel::i32(X, Cmp::Lt, 5), Rel::i32(X, Cmp::Gt, 10)]),
        Rel::And(vec![Rel::eq_u32(K, 1), Rel::eq_u32(K, 2)]),
        Rel::And(vec![Rel::eq_u32(K, 1), Rel::ne_u32(K, 1)]),
        Rel::And(vec![Rel::Plane(0), Rel::Not(Box::new(Rel::Plane(0)))]),
        Rel::i32(X, Cmp::Lt, i32::MIN),
        Rel::i32(X, Cmp::Gt, i32::MAX),
    ] {
        assert_eq!(normalize(&rel), Rel::Empty, "{rel:?}");
        for m in [
            Measure::Count,
            Measure::Sum(V),
            Measure::Min(X),
            Measure::Max(X),
        ] {
            let q = Query {
                rel: rel.clone(),
                measure: m,
            };
            let r = with_planes(&fx, |p| run(&q, p, None)).expect("run");
            assert_eq!(r.answer, oracle(&fx, &q).0, "{rel:?} {m:?}");
            assert_eq!(r.programs, 0, "{rel:?} {m:?} ran a program");
        }
    }
}

/// FAILS IF: a tautology's count is executed instead of answered.
#[test]
fn a_tautology_is_counted_without_a_program() {
    let fx = fixture();
    for rel in [
        Rel::Or(vec![Rel::i32(X, Cmp::Lt, 5), Rel::i32(X, Cmp::Ge, 5)]),
        Rel::Or(vec![Rel::Plane(1), Rel::Not(Box::new(Rel::Plane(1)))]),
        Rel::i32(X, Cmp::Le, i32::MAX),
        Rel::i32(X, Cmp::Ge, i32::MIN),
    ] {
        assert_eq!(normalize(&rel), Rel::All, "{rel:?}");
        let (r, _) = count(&fx, rel);
        assert_eq!(r.answer, Answer::Count(N as u64));
        assert_eq!(r.programs, 0);
    }
}

/// FAILS IF: overlapping or adjacent intervals in a union are not merged.
#[test]
fn a_union_of_intervals_merges() {
    assert_eq!(
        normalize(&Rel::Or(vec![
            Rel::i32(X, Cmp::Lt, 5),
            Rel::i32(X, Cmp::Lt, 9)
        ])),
        Rel::i32(X, Cmp::Lt, 9)
    );
    let eqs = Rel::Or(vec![
        Rel::i32(X, Cmp::Eq, 3),
        Rel::i32(X, Cmp::Eq, 5),
        Rel::i32(X, Cmp::Eq, 4),
    ]);
    assert_eq!(
        normalize(&eqs),
        Rel::And(vec![Rel::i32(X, Cmp::Lt, 6), Rel::i32(X, Cmp::Gt, 2)])
    );
    let (r1, r2) = count_both(&fixture(), eqs);
    eprintln!(
        "passes: rewritten {} vs as written {}",
        r1.passes, r2.passes
    );
    assert!(r1.passes < r2.passes, "{} vs {}", r1.passes, r2.passes);
    // A gap stays a gap.
    let gap = Rel::Or(vec![Rel::i32(X, Cmp::Eq, 3), Rel::i32(X, Cmp::Eq, 5)]);
    assert_eq!(normalize(&gap).atoms(), 2);
    count(&fixture(), gap);
}

/// FAILS IF: `x ∧ (x ∨ y)` is not reduced to `x`.
#[test]
fn absorption() {
    let rel = Rel::And(vec![
        Rel::Plane(0),
        Rel::Or(vec![Rel::Plane(0), Rel::i32(Y, Cmp::Lt, 3)]),
    ]);
    assert_eq!(normalize(&rel), Rel::Plane(0));
    count(&fixture(), rel);
}

/// FAILS IF: a negation is not pushed to the leaves.
#[test]
fn negation_reaches_the_leaves() {
    let rel = Rel::Not(Box::new(Rel::And(vec![
        Rel::i32(X, Cmp::Lt, 5),
        Rel::Plane(0),
    ])));
    assert_eq!(
        normalize(&rel),
        Rel::Or(vec![
            Rel::i32(X, Cmp::Gt, 4),
            Rel::Not(Box::new(Rel::Plane(0)))
        ])
    );
    count(&fixture(), rel);
}

/// FAILS IF: the rewrite stops shrinking a wide conjunction — as written it
/// needs more scratch slots than one fold has and is refused; rewritten it is
/// one interval and runs.
#[test]
fn a_wide_conjunction_runs_only_after_the_rewrite() {
    let rel = Rel::And((0..10).map(|i| Rel::i32(X, Cmp::Lt, 20 - i)).collect());
    let fx = fixture();
    let q = Query {
        rel,
        measure: Measure::Count,
    };
    let ((r1, _), (r2, _)) = both(&fx, &q);
    assert!(r2.as_ref().is_err_and(out_of_slots), "as written: {r2:?}");
    assert_eq!(r1.expect("rewritten").answer, oracle(&fx, &q).0);
}
