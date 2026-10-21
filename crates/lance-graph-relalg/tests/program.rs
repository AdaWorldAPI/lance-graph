//! D-RU-2 (U2): a [`SetProgram`] run with shared lets must give the same
//! answers as running every query with its lets inlined, and as a
//! row-at-a-time oracle — and must read fewer lanes when something is
//! shared.

use lance_graph_contract::set_algebra::{Cmp, SetExpr};
use lance_graph_contract::set_program::{share, ProgramError as ShapeError, SetProgram};
use lance_graph_mask_risc::{words_for, LaneRef, Planes};
use lance_graph_relalg::{run_program, run_recompute, Answer, Measure, ProgramError};

const N: usize = 3_000;
const VERSION: u64 = 7;
const RESIDENT: u32 = 2;

// Lane indexes.
const X: u16 = 0; // i32, (i % 50) - 20
const Y: u16 = 1; // i32, (i * 7) % 31
const V: u16 = 2; // i32, the summed value
const K: u16 = 3; // u32, i % 5

struct Fixture {
    x: Vec<i32>,
    y: Vec<i32>,
    v: Vec<i32>,
    k: Vec<u32>,
    p: [Vec<u64>; 2],
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
        p: [plane(|r| r % 3 == 0), plane(|r| r % 7 < 3)],
    }
}

fn with_planes<T>(fx: &Fixture, f: impl FnOnce(&Planes<'_>) -> T) -> T {
    let lanes = [
        LaneRef::I32(&fx.x),
        LaneRef::I32(&fx.y),
        LaneRef::I32(&fx.v),
        LaneRef::U32(&fx.k),
    ];
    let masks: [&[u64]; 2] = [&fx.p[0], &fx.p[1]];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &lanes,
    };
    f(&planes)
}

// ── the row oracle, over resident-only descriptions ─────────────────────────

fn holds(fx: &Fixture, e: &SetExpr, r: usize) -> bool {
    use lance_graph_contract::set_algebra::Res;
    match e {
        SetExpr::All => true,
        SetExpr::Empty => false,
        SetExpr::Sym(i) => fx.p[*i as usize][r / 64] >> (r % 64) & 1 == 1,
        SetExpr::Res(Res::I32 { lane, op, t }) => {
            let x = [&fx.x, &fx.y, &fx.v][*lane as usize][r];
            match op {
                Cmp::Lt => x < *t,
                Cmp::Le => x <= *t,
                Cmp::Gt => x > *t,
                Cmp::Ge => x >= *t,
                Cmp::Eq => x == *t,
                Cmp::Ne => x != *t,
            }
        }
        SetExpr::Res(Res::U32 { lane, eq, v }) => {
            assert_eq!(*lane, K);
            (fx.k[r] == *v) == *eq
        }
        SetExpr::And(cs) => cs.iter().all(|c| holds(fx, c, r)),
        SetExpr::Or(cs) => cs.iter().any(|c| holds(fx, c, r)),
        SetExpr::Not(c) => !holds(fx, c, r),
    }
}

fn oracle(fx: &Fixture, e: &SetExpr, m: Measure) -> (Answer, Vec<u64>) {
    let rows: Vec<usize> = (0..N).filter(|&r| holds(fx, e, r)).collect();
    let mask = plane(|r| holds(fx, e, r));
    let vals = |l: u16| -> Vec<i32> {
        let lane = [&fx.x, &fx.y, &fx.v][l as usize];
        rows.iter().map(|&r| lane[r]).collect()
    };
    let a = match m {
        Measure::Count => Answer::Count(rows.len() as u64),
        Measure::Sum(l) => Answer::Sum(vals(l).iter().map(|&v| i64::from(v)).sum()),
        Measure::Min(l) => Answer::Min(vals(l).into_iter().min()),
        Measure::Max(l) => Answer::Max(vals(l).into_iter().max()),
        Measure::Keep => Answer::Kept,
    };
    (a, mask)
}

// ── building blocks ─────────────────────────────────────────────────────────

fn x(op: Cmp, t: i32) -> SetExpr {
    SetExpr::i32(X, op, t)
}
fn y(op: Cmp, t: i32) -> SetExpr {
    SetExpr::i32(Y, op, t)
}
fn k(v: u32) -> SetExpr {
    SetExpr::Res(lance_graph_contract::set_algebra::Res::U32 {
        lane: K,
        eq: true,
        v,
    })
}
fn and(cs: Vec<SetExpr>) -> SetExpr {
    SetExpr::And(cs)
}
fn or(cs: Vec<SetExpr>) -> SetExpr {
    SetExpr::Or(cs)
}

/// Four queries that all carry the conjunction `x >= -5 ∧ y < 20 ∧ k == 2`.
fn shared_queries() -> Vec<SetExpr> {
    let core = vec![x(Cmp::Ge, -5), y(Cmp::Lt, 20), k(2)];
    let with = |extra: SetExpr| {
        let mut cs = core.clone();
        cs.push(extra);
        and(cs)
    };
    vec![
        with(SetExpr::Sym(0)),
        with(SetExpr::Sym(1)),
        with(x(Cmp::Lt, 25)),
        or(vec![and(core.clone()), y(Cmp::Eq, 3)]),
    ]
}

fn measures(n: usize) -> Vec<Measure> {
    let cycle = [
        Measure::Count,
        Measure::Sum(V),
        Measure::Min(X),
        Measure::Max(Y),
        Measure::Keep,
    ];
    (0..n).map(|i| cycle[i % cycle.len()]).collect()
}

/// Program, recompute and oracle must agree on every answer and kept mask.
fn assert_agree(
    fx: &Fixture,
    p: &SetProgram,
    ms: &[Measure],
) -> (
    lance_graph_relalg::ProgramCost,
    lance_graph_relalg::ProgramCost,
) {
    let keeps_n = ms.iter().filter(|m| matches!(m, Measure::Keep)).count();
    let mut k1 = vec![vec![0u64; words_for(N)]; keeps_n];
    let mut k2 = vec![vec![0u64; words_for(N)]; keeps_n];
    let (a, b) = with_planes(fx, |pl| {
        let mut s1: Vec<&mut [u64]> = k1.iter_mut().map(Vec::as_mut_slice).collect();
        let mut s2: Vec<&mut [u64]> = k2.iter_mut().map(Vec::as_mut_slice).collect();
        (
            run_program(p, VERSION, pl, ms, &mut s1).expect("program runs"),
            run_recompute(p, VERSION, pl, ms, &mut s2).expect("recompute runs"),
        )
    });
    assert_eq!(a.answers, b.answers, "program vs recompute");
    let mut keep_i = 0;
    for (i, (q, &m)) in p.queries.iter().zip(ms).enumerate() {
        let (want, mask) = oracle(fx, &p.inline(q), m);
        assert_eq!(a.answers[i], want, "query {i} vs oracle");
        if m == Measure::Keep {
            assert_eq!(k1[keep_i], mask, "query {i} program kept mask");
            assert_eq!(k2[keep_i], mask, "query {i} recompute kept mask");
            keep_i += 1;
        }
    }
    (a.cost, b.cost)
}

// ── tests ───────────────────────────────────────────────────────────────────

/// FAILS IF: a let is evaluated against the wrong planes, a query reads a
/// let's plane at the wrong index, or the sinks are paired with the wrong
/// queries.
#[test]
fn shared_lets_give_the_same_answers_as_recompute_and_the_oracle() {
    let fx = fixture();
    let qs = shared_queries();
    let p = share(VERSION, RESIDENT, &qs);
    assert!(!p.lets.is_empty(), "the shared conjunction must be lifted");
    for rot in 0..5 {
        let mut ms = measures(qs.len() + rot);
        ms.drain(..rot);
        assert_agree(&fx, &p, &ms);
    }
}

/// FAILS IF: sharing does not cut lane reads, or the program does not
/// account for the planes it materializes.
#[test]
fn sharing_reads_fewer_lanes_and_pays_one_plane_per_let() {
    let fx = fixture();
    let qs = shared_queries();
    let p = share(VERSION, RESIDENT, &qs);
    let (prog, rec) = assert_agree(&fx, &p, &measures(qs.len()));
    // The core's three comparisons are read once instead of four times.
    assert_eq!(rec.lane_reads, 3 * 4 + 1 + 1, "recompute: {rec:?}");
    assert_eq!(prog.lane_reads, 3 + 1 + 1, "program: {prog:?}");
    assert!(prog.passes < rec.passes, "{prog:?} vs {rec:?}");
    assert_eq!(prog.planes_materialized, p.lets.len());
    assert_eq!(prog.plane_bytes, p.lets.len() * words_for(N) * 8);
    assert_eq!(rec.planes_materialized, 0);
    assert_eq!(rec.plane_bytes, 0);
    assert_eq!(prog.programs, rec.programs + p.lets.len());
    // Trees, not DAGs: scratch stays small.
    assert!(prog.max_scratch_slots <= 3, "{prog:?}");
    assert!(rec.max_scratch_slots <= 3, "{rec:?}");
}

/// A let that reads another let, plus a second, unrelated let.
fn nested_queries() -> Vec<SetExpr> {
    let ab = vec![x(Cmp::Ge, -5), y(Cmp::Lt, 20)];
    let cat = |extra: Vec<SetExpr>| {
        let mut cs = ab.clone();
        cs.extend(extra);
        and(cs)
    };
    let alt = or(vec![k(4), y(Cmp::Gt, 25)]);
    vec![
        cat(vec![k(2), SetExpr::Sym(0)]),
        cat(vec![k(2), SetExpr::Sym(1)]),
        cat(vec![x(Cmp::Lt, 10)]),
        cat(vec![y(Cmp::Gt, 4)]),
        and(vec![alt.clone(), SetExpr::Sym(0)]),
        and(vec![alt, x(Cmp::Lt, 0)]),
    ]
}

/// Whether some let reads an earlier let.
fn has_nested_let(p: &SetProgram) -> bool {
    fn reads_let(e: &SetExpr, resident: u32) -> bool {
        match e {
            SetExpr::Sym(i) => *i >= resident,
            SetExpr::And(cs) | SetExpr::Or(cs) => cs.iter().any(|c| reads_let(c, resident)),
            SetExpr::Not(c) => reads_let(c, resident),
            _ => false,
        }
    }
    p.lets.iter().any(|l| reads_let(l, p.resident))
}

/// FAILS IF: a let is evaluated with another let's description, or cannot
/// read the plane of a let computed before it.
#[test]
fn nested_and_distinct_lets_agree_with_recompute_and_the_oracle() {
    let fx = fixture();
    let qs = nested_queries();
    let p = share(VERSION, RESIDENT, &qs);
    // Anti-vacuity: the fixture must yield several lets, one reading another.
    assert!(p.lets.len() >= 3, "lets: {:?}", p.lets);
    assert!(has_nested_let(&p), "no nested let: {:?}", p.lets);
    let (prog, rec) = assert_agree(&fx, &p, &measures(qs.len()));
    assert!(prog.lane_reads < rec.lane_reads, "{prog:?} vs {rec:?}");
    assert_eq!(prog.planes_materialized, p.lets.len());
}

/// FAILS IF: an unshared program changes cost, or materializes anything.
#[test]
fn nothing_shared_costs_exactly_what_recompute_costs() {
    let fx = fixture();
    let qs = vec![
        and(vec![x(Cmp::Ge, 0), SetExpr::Sym(0)]),
        or(vec![y(Cmp::Lt, 5), k(1)]),
        SetExpr::Not(Box::new(SetExpr::Sym(1))),
    ];
    let p = share(VERSION, RESIDENT, &qs);
    assert!(p.lets.is_empty());
    let (prog, rec) = assert_agree(&fx, &p, &measures(qs.len()));
    assert_eq!(prog, rec);
}

/// FAILS IF: an answer computed on one version is served for another.
#[test]
fn a_program_for_another_version_is_refused() {
    let fx = fixture();
    let p = share(VERSION, RESIDENT, &shared_queries());
    let ms = measures(p.queries.len());
    with_planes(&fx, |pl| {
        let mut sink = vec![0u64; words_for(N)];
        let mut sinks: Vec<&mut [u64]> = vec![&mut sink];
        let want = ProgramError::VersionMismatch {
            program: VERSION,
            table: VERSION + 1,
        };
        assert_eq!(
            run_program(&p, VERSION + 1, pl, &ms, &mut sinks).unwrap_err(),
            want
        );
        assert_eq!(
            run_recompute(&p, VERSION + 1, pl, &ms, &mut sinks).unwrap_err(),
            want
        );
    });
}

/// FAILS IF: a malformed program, the wrong resident planes, or too few
/// measures run anyway.
#[test]
fn malformed_programs_and_mismatched_inputs_are_refused() {
    let fx = fixture();
    with_planes(&fx, |pl| {
        let mut none: Vec<&mut [u64]> = Vec::new();
        let forward = SetProgram {
            version: VERSION,
            resident: RESIDENT,
            lets: vec![SetExpr::Sym(RESIDENT + 1), x(Cmp::Lt, 0)],
            queries: vec![SetExpr::Sym(RESIDENT)],
        };
        assert_eq!(
            run_program(&forward, VERSION, pl, &[Measure::Count], &mut none).unwrap_err(),
            ProgramError::Shape(ShapeError::ForwardReference {
                at: 0,
                sym: RESIDENT + 1
            })
        );

        let three = SetProgram {
            version: VERSION,
            resident: 3,
            lets: vec![],
            queries: vec![SetExpr::Sym(2)],
        };
        assert_eq!(
            run_program(&three, VERSION, pl, &[Measure::Count], &mut none).unwrap_err(),
            ProgramError::ResidentPlanes {
                expected: 3,
                got: 2
            }
        );

        let p = share(VERSION, RESIDENT, &shared_queries());
        assert_eq!(
            run_program(&p, VERSION, pl, &[Measure::Count], &mut none).unwrap_err(),
            ProgramError::Measures {
                queries: p.queries.len(),
                measures: 1
            }
        );
    });
}

/// FAILS IF: any of many generated query sets disagrees between the three.
/// The pool is small so repeated junctions are common.
#[test]
fn generated_query_sets_agree() {
    let fx = fixture();
    let pool = [
        x(Cmp::Ge, -5),
        x(Cmp::Lt, 10),
        y(Cmp::Lt, 20),
        y(Cmp::Gt, 4),
        k(2),
        k(4),
        SetExpr::Sym(0),
        SetExpr::Sym(1),
    ];
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |m: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % m as u64) as usize
    };
    let mut shared_any = 0;
    for _ in 0..60 {
        let mut qs = Vec::new();
        for _ in 0..(2 + next(5)) {
            let pick = |next: &mut dyn FnMut(usize) -> usize| -> Vec<SetExpr> {
                (0..(2 + next(3)))
                    .map(|_| pool[next(pool.len())].clone())
                    .collect()
            };
            let inner = pick(&mut next);
            let q = match next(4) {
                0 => and(inner),
                1 => or(inner),
                2 => and(vec![or(inner), pool[next(pool.len())].clone()]),
                _ => SetExpr::Not(Box::new(and(inner))),
            };
            qs.push(q);
        }
        let p = share(VERSION, RESIDENT, &qs);
        if !p.lets.is_empty() {
            shared_any += 1;
        }
        let (prog, rec) = assert_agree(&fx, &p, &measures(qs.len()));
        assert!(prog.lane_reads <= rec.lane_reads, "{prog:?} vs {rec:?}");
    }
    // Anti-vacuity: the generator must actually exercise sharing.
    assert!(shared_any > 10, "only {shared_any}/60 sets shared anything");
}
