//! D-DR256-WHERE P2 (lab only, reported, not gated): one bound selection R
//! reused across selected properties, through the real paths.
//!
//! Universe 65,536 rows. R = live ∧ sel = 1 ∧ region <> 'x', bound through
//! Quack's own `Draft::bind` and a consumer `Binder`. Property p is an i32
//! lane `val_p` gated by a resident presence plane `pres_p`. The answer per
//! property is Σ val_p over R ∧ pres_p. Every arm is checked against a
//! per-row oracle.
//!
//! Arms for the P property reads (each program lowered and compiled ONCE,
//! outside the timed loop; lowering is timed separately):
//! - QA  independent programs: criteria ∧ pres_p, `lower`. R is recomputed P times.
//! - QAf the same with `lower_fused`.
//! - QB  R kept once (`Agg::Rows` → `Keep`), then Plane(R) ∧ pres_p, `lower`.
//! - QBf the same with `lower_fused`.
//! - RL  relalg `SetProgram { lets: [R], queries: [R ∧ pres_p] }` via
//!       `run_program` (the let's plane is allocated per call), and
//!       `run_recompute` (lets inlined) as its reference.
//! - NB  the P1 shared block schedule over the same R (ndarray only), as the
//!       reference for "what skipping dead 256-row blocks would buy".
//!
//! Stages timed separately: bind, lower, selection creation (Keep into an
//! 8 KiB caller buffer = the only materialization), COUNT without a mask,
//! occupancy (lab scalar), property reads.
//!
//! Controls (asserted): stale codebook, scope (live plane) change, selection
//! change and property-version change. Quack has no frame (D-NFR S5): the
//! first three run silently stale, and the probe asserts that they do, so
//! the gap is measured, not assumed. relalg refuses a version mismatch.

use lance_graph_contract::set_algebra::{Cmp as SCmp, Res, SetExpr};
use lance_graph_contract::set_program::SetProgram;
use lance_graph_mask_risc::{
    execute_compiled, execute_into, words_for, CompiledProgram, Foreign, LaneRef, Out, Planes,
    Scratch, Value,
};
use lance_graph_quack::bind::{table, Binder, BoundField, FieldKind, FieldRef, TableId};
use lance_graph_quack::{lower, lower_fused, Agg, Cmp, Col, Filter, Mask, Query};
use lance_graph_relalg::{run_program, run_recompute, Answer, Measure, ProgramError};
use ndarray::simd::{mask_and, masked_sum_i32, popcount_batch_u64};
use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 65_536;
const WORDS: usize = N / 64;
const BLOCK_WORDS: usize = 4;
const BLOCKS: usize = WORDS / BLOCK_WORDS;
const PMAX: usize = 64;
const REGION: FieldRef = FieldRef::new("t", "region");

fn median_ns(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[reps / 2]
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The consumer's catalog + codebook. `code_x` is the code 'x' resolves to;
/// re-observing the codebook changes it, which is the stale-binding control.
struct Catalog {
    code_x: u32,
}
impl Binder for Catalog {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == "t").then_some(TableId(0))
    }
    fn live(&self, _t: TableId) -> Mask {
        Mask(0)
    }
    fn field(&self, _t: TableId, name: &str) -> Option<BoundField> {
        match name {
            "sel" => Some(BoundField { col: Col(0), kind: FieldKind::I32, validity: None }),
            "region" => Some(BoundField { col: Col(1), kind: FieldKind::Code, validity: None }),
            _ => None,
        }
    }
    fn code(&self, _t: TableId, _f: Col, literal: &str) -> Option<u32> {
        (literal == "x").then_some(self.code_x)
    }
}

/// The resident world: lanes and planes the programs read in place.
struct World {
    live: Vec<u64>,
    sel: Vec<i32>,
    region: Vec<u32>,
    vals: Vec<Vec<i32>>,
    pres: Vec<Vec<u64>>,
}

fn bit(m: &[u64], i: usize) -> bool {
    m[i >> 6] >> (i & 63) & 1 == 1
}

fn oracle_r(w: &World, code_x: u32) -> Vec<u64> {
    let mut m = vec![0u64; WORDS];
    for i in 0..N {
        if bit(&w.live, i) && w.sel[i] == 1 && w.region[i] != code_x {
            m[i >> 6] |= 1 << (i & 63);
        }
    }
    m
}

fn oracle_sums(w: &World, r: &[u64], p: usize) -> Vec<i64> {
    (0..p)
        .map(|q| {
            (0..N)
                .filter(|&i| bit(r, i) && bit(&w.pres[q], i))
                .map(|i| w.vals[q][i] as i64)
                .sum()
        })
        .collect()
}

fn occupancy(m: &[u64]) -> (u64, u32, u32) {
    let (mut pop, mut words, mut blocks) = (0u64, 0u32, 0u32);
    for b in m.chunks_exact(BLOCK_WORDS) {
        let mut any = 0u64;
        for &x in b {
            pop += x.count_ones() as u64;
            words += (x != 0) as u32;
            any |= x;
        }
        blocks += (any != 0) as u32;
    }
    (pop, words, blocks)
}

fn fixtures() -> Vec<(&'static str, BTreeSet<usize>)> {
    let mut rng = Rng(0xD256_0002);
    let uniform = |rng: &mut Rng, k: usize| {
        let mut u = BTreeSet::new();
        while u.len() < k {
            u.insert(rng.below(N));
        }
        u
    };
    let mut clustered = BTreeSet::new();
    while clustered.len() < 4096 {
        let w = rng.below(WORDS);
        clustered.extend(w * 64..w * 64 + 64);
    }
    vec![
        ("empty", BTreeSet::new()),
        ("contiguous 256", (0..256).collect()),
        ("uniform 256", uniform(&mut rng, 256)),
        ("clustered 4096", clustered),
        ("dense 4096 (64 words)", (0..4096).collect()),
        ("scattered 4096 (1/16)", (0..N).step_by(16).collect()),
        ("uniform 32768", uniform(&mut rng, 32_768)),
        ("full 65536", (0..N).collect()),
    ]
}

/// One fixture's world: `sel` marks the fixture rows; `region` is 'x' (code
/// 5) on ~1 % of rows, so the second predicate removes a few survivors.
fn world(rows: &BTreeSet<usize>, rng: &mut Rng) -> World {
    let mut sel = vec![0i32; N];
    for &r in rows {
        sel[r] = 1;
    }
    let region = (0..N).map(|_| if rng.below(100) == 0 { 5 } else { 7 }).collect();
    let pres = (0..PMAX)
        .map(|p| {
            let keep = PMAX - p;
            (0..WORDS)
                .map(|_| {
                    let mut w = 0u64;
                    for b in 0..64 {
                        if rng.below(PMAX + 1) < keep {
                            w |= 1 << b;
                        }
                    }
                    w
                })
                .collect()
        })
        .collect();
    let vals = (0..PMAX)
        .map(|_| (0..N).map(|_| (rng.next() % 2001) as i32 - 1000).collect())
        .collect();
    World { live: vec![u64::MAX; WORDS], sel, region, vals, pres }
}

/// masks = [live, pres_0..pres_{P-1}, R]; lanes = [sel, region, val_0..].
fn planes_of<'a>(n: usize, masks: &'a [&'a [u64]], lanes: &'a [LaneRef<'a>]) -> Planes<'a> {
    Planes { n_rows: n, masks, lanes }
}

fn run_sum(c: &CompiledProgram, planes: &Planes<'_>, s: &mut Scratch<'_>) -> i64 {
    match execute_compiled(&c.compiled(), planes, &Foreign::NONE, s, Out::None, 0..N).unwrap() {
        Value::SumI64(v) => v,
        v => panic!("expected a sum, got {v:?}"),
    }
}

fn main() {
    if std::env::args().any(|a| a == "--controls") {
        // Correctness controls only; no benchmark arm runs.
        controls(&mut Rng(0x0D0D_F00D));
        return;
    }
    let reps = 101;
    println!("# D-DR256-WHERE P2 — Quack path. ns, median of {reps}. All arms oracle-checked.");
    let mut rng = Rng(0x0D0D_F00D);
    let cat = Catalog { code_x: 5 };
    let draft = table("t").where_eq("sel", 1).where_(REGION.ne("x"));

    // ── binding and lowering, independent of data ─────────────────────────
    let t_bind = median_ns(reps, || {
        black_box(draft.bind(black_box(&cat)).unwrap());
    });
    let criteria_q = draft.bind(&cat).unwrap();
    let crit = criteria_q.filter.clone();
    let keep_q = Query { filter: crit.clone(), agg: Agg::Rows };
    let t_lower = median_ns(reps, || {
        black_box(lower(black_box(&keep_q)).unwrap());
    });
    let t_lower_f = median_ns(reps, || {
        black_box(lower_fused(black_box(&keep_q)).unwrap());
    });
    println!("bind (Draft::bind, 2 predicates + live) {t_bind:.0} ns; lower {t_lower:.0} ns; lower_fused {t_lower_f:.0} ns");
    println!("bound filter: {crit:?}");
    println!();
    println!("fixture | P | |R| | words | blocks | keep R | count-only | occupancy | QA | QAf | QB | QBf | RL share | RL recompute | NB blocks | QB/QA | NB/QB");

    for (name, rows) in fixtures() {
        let w = world(&rows, &mut rng);
        let r_oracle = oracle_r(&w, 5);
        let (pop, aw, ab) = occupancy(&r_oracle);

        // Selection creation: Keep into a caller-owned 8 KiB buffer.
        let lanes_base: Vec<LaneRef<'_>> = [LaneRef::I32(&w.sel), LaneRef::U32(&w.region)]
            .into_iter()
            .chain(w.vals.iter().map(|v| LaneRef::I32(v)))
            .collect();
        let keep_prog = CompiledProgram::new(lower(&keep_q).unwrap());
        let count_prog = CompiledProgram::new(
            lower(&Query { filter: crit.clone(), agg: Agg::Count }).unwrap(),
        );
        let base_masks: Vec<&[u64]> = std::iter::once(&w.live[..])
            .chain(w.pres.iter().map(|p| &p[..]))
            .collect();
        let pl0 = planes_of(N, &base_masks, &lanes_base);
        let mut r = vec![0u64; WORDS];
        let mut sk = Scratch::for_program(keep_prog.program(), N).unwrap();
        execute_compiled(&keep_prog.compiled(), &pl0, &Foreign::NONE, &mut sk, Out::Mask(&mut r), 0..N)
            .unwrap();
        assert_eq!(r, r_oracle, "{name}: Keep(R) != oracle");
        let t_keep = median_ns(reps, || {
            execute_compiled(&keep_prog.compiled(), &pl0, &Foreign::NONE, &mut sk, Out::Mask(&mut r), 0..N)
                .unwrap();
        });
        let mut sc = Scratch::for_program(count_prog.program(), N).unwrap();
        let cnt = execute_compiled(&count_prog.compiled(), &pl0, &Foreign::NONE, &mut sc, Out::None, 0..N)
            .unwrap();
        assert_eq!(cnt, Value::Count(rows.len() - rows.iter().filter(|&&i| w.region[i] == 5).count()));
        let t_count = median_ns(reps, || {
            black_box(
                execute_compiled(&count_prog.compiled(), &pl0, &Foreign::NONE, &mut sc, Out::None, 0..N)
                    .unwrap(),
            );
        });
        assert_eq!(popcount_batch_u64(&r), pop);
        let t_occ = median_ns(reps, || {
            black_box(occupancy(black_box(&r)));
        });

        // Planes with R appended as a resident plane (index 1 + PMAX).
        let r_idx = (1 + PMAX) as u16;
        let masks_r: Vec<&[u64]> = base_masks.iter().copied().chain(std::iter::once(&r[..])).collect();
        let pl = planes_of(N, &masks_r, &lanes_base);
        let active: Vec<usize> = (0..BLOCKS)
            .filter(|&b| r[b * BLOCK_WORDS..(b + 1) * BLOCK_WORDS].iter().any(|&x| x != 0))
            .collect();

        for &p in &[1usize, 2, 4, 8, 16, 32, 64] {
            let want = oracle_sums(&w, &r, p);
            let prog = |f: Filter, q: usize, fused: bool| {
                let query = Query { filter: f, agg: Agg::SumI32(Col(2 + q as u16)) };
                CompiledProgram::new(if fused { lower_fused(&query) } else { lower(&query) }.unwrap())
            };
            let indep = |q: usize| Filter::and([crit.clone(), Filter::plane(Mask(1 + q as u16))]);
            let reuse = |q: usize| Filter::and([Filter::plane(Mask(r_idx)), Filter::plane(Mask(1 + q as u16))]);
            let qa: Vec<_> = (0..p).map(|q| prog(indep(q), q, false)).collect();
            let qaf: Vec<_> = (0..p).map(|q| prog(indep(q), q, true)).collect();
            let qb: Vec<_> = (0..p).map(|q| prog(reuse(q), q, false)).collect();
            let qbf: Vec<_> = (0..p).map(|q| prog(reuse(q), q, true)).collect();
            let slots = qa.iter().chain(&qaf).chain(&qb).chain(&qbf).map(|c| c.program().scratch_slots).max().unwrap_or(1) as usize;
            let mut s = Scratch::new(lance_graph_mask_risc::tile_words_for(N), slots.max(1));
            let arm = |progs: &[CompiledProgram], s: &mut Scratch<'_>| -> Vec<i64> {
                progs.iter().map(|c| run_sum(c, &pl, s)).collect()
            };
            for (label, progs) in [("QA", &qa), ("QAf", &qaf), ("QB", &qb), ("QBf", &qbf)] {
                assert_eq!(arm(progs, &mut s), want, "{name} P={p}: {label}");
            }
            let t_qa = median_ns(reps, || { black_box(arm(&qa, &mut s)); });
            let t_qaf = median_ns(reps, || { black_box(arm(&qaf, &mut s)); });
            let t_qb = median_ns(reps, || { black_box(arm(&qb, &mut s)); });
            let t_qbf = median_ns(reps, || { black_box(arm(&qbf, &mut s)); });

            // relalg: R as a let over the resident planes [live, pres_0..].
            let r_set = SetExpr::And(vec![
                SetExpr::Sym(0),
                SetExpr::Res(Res::I32 { lane: 0, op: SCmp::Eq, t: 1 }),
                SetExpr::Res(Res::U32 { lane: 1, eq: false, v: 5 }),
            ]);
            let resident = (1 + PMAX) as u32;
            let sp = SetProgram {
                version: 7,
                resident,
                lets: vec![r_set],
                queries: (0..p)
                    .map(|q| SetExpr::And(vec![SetExpr::Sym(resident), SetExpr::Sym(1 + q as u32)]))
                    .collect(),
            };
            let measures: Vec<Measure> = (0..p).map(|q| Measure::Sum(2 + q as u16)).collect();
            let rl = |share: bool| -> Vec<i64> {
                let run = if share {
                    run_program(&sp, 7, &pl0, &measures, &mut [])
                } else {
                    run_recompute(&sp, 7, &pl0, &measures, &mut [])
                }
                .unwrap();
                run.answers
                    .iter()
                    .map(|a| match a {
                        Answer::Sum(v) => *v,
                        a => panic!("expected a sum, got {a:?}"),
                    })
                    .collect()
            };
            assert_eq!(rl(true), want, "{name} P={p}: relalg share");
            assert_eq!(rl(false), want, "{name} P={p}: relalg recompute");
            let t_rl = median_ns(reps, || { black_box(rl(true)); });
            let t_rlr = median_ns(reps, || { black_box(rl(false)); });

            // P1 block schedule over the same R.
            let mut tb = [0u64; BLOCK_WORDS];
            let mut nb = || -> Vec<i64> {
                (0..p)
                    .map(|q| {
                        let mut acc = 0i64;
                        for &b in &active {
                            let o = b * BLOCK_WORDS;
                            mask_and(&r[o..o + BLOCK_WORDS], &w.pres[q][o..o + BLOCK_WORDS], &mut tb);
                            acc += masked_sum_i32(&w.vals[q][b * 256..b * 256 + 256], &tb);
                        }
                        acc
                    })
                    .collect()
            };
            assert_eq!(nb(), want, "{name} P={p}: NB");
            let t_nb = median_ns(reps, || { black_box(nb()); });

            println!(
                "{name} | {p} | {pop} | {aw} | {ab} | {t_keep:.0} | {t_count:.0} | {t_occ:.0} | {t_qa:.0} | {t_qaf:.0} | {t_qb:.0} | {t_qbf:.0} | {t_rl:.0} | {t_rlr:.0} | {t_nb:.0} | {:.2} | {:.2}",
                t_qb / t_qa,
                t_nb / t_qb
            );
        }
    }

    controls(&mut rng);
}

/// Stale binding, scope change, selection change, property version.
fn controls(rng: &mut Rng) {
    println!();
    println!("# controls (scattered 4096)");
    let rows: BTreeSet<usize> = (0..N).step_by(16).collect();
    let mut w = world(&rows, rng);
    let draft = table("t").where_eq("sel", 1).where_(REGION.ne("x"));
    let keep_q = |cat: &Catalog| Query { filter: draft.bind(cat).unwrap().filter, agg: Agg::Rows };
    let keep = |q: &Query, w: &World| -> Vec<u64> {
        let lanes: Vec<LaneRef<'_>> = [LaneRef::I32(&w.sel), LaneRef::U32(&w.region)]
            .into_iter()
            .chain(w.vals.iter().map(|v| LaneRef::I32(v)))
            .collect();
        let masks: Vec<&[u64]> = std::iter::once(&w.live[..]).chain(w.pres.iter().map(|p| &p[..])).collect();
        let prog = lower(q).unwrap();
        let mut s = Scratch::for_program(&prog, N).unwrap();
        let mut out = vec![0u64; words_for(N)];
        execute_into(&prog, &planes_of(N, &masks, &lanes), &Foreign::NONE, &mut s, Out::Mask(&mut out)).unwrap();
        out
    };

    // 1. Stale codebook: the binder re-observes 'x' as code 9; region rows that
    //    were 'x' (5) are re-coded. The query bound under the old codebook still
    //    runs and answers with the old code. Nothing refuses (S5 × S6 missing).
    let q_old = keep_q(&Catalog { code_x: 5 });
    for v in w.region.iter_mut() {
        if *v == 5 {
            *v = 9;
        }
    }
    let stale = keep(&q_old, &w);
    let fresh = keep(&keep_q(&Catalog { code_x: 9 }), &w);
    assert_eq!(fresh, oracle_r(&w, 9));
    assert_ne!(stale, fresh, "stale-codebook control must differ, or it tests nothing");
    println!(
        "stale codebook: old binding answers |R|={} vs fresh {} — SILENT (no frame on a bound Query)",
        popcount_batch_u64(&stale),
        popcount_batch_u64(&fresh)
    );

    // 2. Scope change: rows leave the live plane. A kept R from before is reused
    //    as a plane; Quack cannot tell, so property reads include dead rows.
    let r_before = keep(&keep_q(&Catalog { code_x: 9 }), &w);
    for wd in w.live.iter_mut().step_by(2) {
        *wd = 0;
    }
    let r_after = keep(&keep_q(&Catalog { code_x: 9 }), &w);
    assert_ne!(r_before, r_after, "scope control must change R");
    println!(
        "scope change: reused R has {} rows, fresh {} — SILENT (a kept plane has no witness)",
        popcount_batch_u64(&r_before),
        popcount_batch_u64(&r_after)
    );

    // 3. Selection change: sel flips on 64 rows; same silence.
    // Odd words stay live after the scope control cleared the even ones.
    for i in (65..N).step_by(1024) {
        w.sel[i] = 1;
    }
    let r_sel = keep(&keep_q(&Catalog { code_x: 9 }), &w);
    assert_ne!(r_after, r_sel, "selection control must change R");
    println!("selection change: reused R {} vs fresh {} — SILENT", popcount_batch_u64(&r_after), popcount_batch_u64(&r_sel));

    // 4a. Predicate-dependent property mutation. R2 also requires
    //     val_0 >= 0, so val_0 is a criteria input of R2. Mutating val_0 on
    //     selected rows must change the fresh R2; the reused R2 is then stale,
    //     and so is any property read through it. Quack does not refuse.
    let crit2 = Filter::and([
        keep_q(&Catalog { code_x: 9 }).filter,
        Filter::Cmp(Col(2), Cmp::GeI32(0)),
    ]);
    let q2 = Query { filter: crit2, agg: Agg::Rows };
    let r2_before = keep(&q2, &w);
    for i in 0..N {
        if bit(&r_sel, i) && i % 2 == 0 {
            w.vals[0][i] = -1 - w.vals[0][i].abs().min(1000);
        }
    }
    let r2_after = keep(&q2, &w);
    assert_ne!(r2_before, r2_after, "predicate-dependent mutation must change R2");
    let read = |r: &[u64], w: &World| {
        let mut m = vec![0u64; WORDS];
        mask_and(r, &w.pres[1], &mut m);
        masked_sum_i32(&w.vals[1], &m)
    };
    assert_ne!(read(&r2_before, &w), read(&r2_after, &w), "a read through the stale R2 must differ");
    println!(
        "predicate-dependent property mutation: reused R2 {} vs fresh {}; val_1 read differs — SILENT (no dependency witness)",
        popcount_batch_u64(&r2_before),
        popcount_batch_u64(&r2_after)
    );

    // 4b. Property mutation outside R's criteria: val_0 changes again, R (which
    //     does not read val_0) stays valid, and a value read must come from the
    //     current lane. Nothing caches values, so it does.
    for v in w.vals[0].iter_mut().step_by(3) {
        *v = v.wrapping_add(11);
    }
    let want = oracle_sums(&w, &r_sel, 1)[0];
    let mut m = vec![0u64; WORDS];
    mask_and(&r_sel, &w.pres[0], &mut m);
    assert_eq!(masked_sum_i32(&w.vals[0], &m), want);
    println!("property version change: reads through reused R equal the fresh oracle (no value cache exists)");

    // 5. relalg refuses a version mismatch (value comparison, not seal identity).
    let lanes: Vec<LaneRef<'_>> = [LaneRef::I32(&w.sel), LaneRef::U32(&w.region)].into_iter().collect();
    let masks: Vec<&[u64]> = vec![&w.live[..]];
    let sp = SetProgram {
        version: 7,
        resident: 1,
        lets: vec![SetExpr::And(vec![SetExpr::Sym(0), SetExpr::Res(Res::I32 { lane: 0, op: SCmp::Eq, t: 1 })])],
        queries: vec![SetExpr::Sym(1)],
    };
    let e = run_program(&sp, 8, &planes_of(N, &masks, &lanes), &[Measure::Count], &mut []).unwrap_err();
    assert_eq!(e, ProgramError::VersionMismatch { program: 7, table: 8 });
    println!("relalg SetProgram under version 8: refused {e:?}");
}
