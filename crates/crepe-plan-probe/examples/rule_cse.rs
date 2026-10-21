//! D-RU-3 (U3): cross-rule CSE over 256 rules, against today's per-rule
//! pipeline. Run from this crate's directory:
//!
//! ```text
//! cargo run --release --example rule_cse    # the benchmark
//! cargo test --example rule_cse             # the correctness fixtures
//! ```
//!
//! # The rules
//!
//! Rule `k` (one per `k` in `0..RULES`):
//!
//! ```text
//! Out(k, x, z) <- R_g(x, c), S_g(c, z), T(c, k)        with g = group(k)
//! ```
//!
//! A *shared-core rate* `p` decides `group`: the first `p · RULES` rules use
//! group 0, every other rule has a core of its own. The rules are a star on
//! `c`; `T(c, k)` is a side restricted to `k`, so no two rules have the same
//! third side. Written as one parametrized rule, the same program is
//!
//! ```text
//! Out(k, x, z) <- G(k, g), RG(g, x, c), SG(g, c, z), T(c, k)
//! ```
//!
//! which is what `crepe!` and `ascent!` evaluate as oracles.
//!
//! # Arms (plan §12)
//!
//! * **A** — [`pipeline::evaluate`]: one table and one fold per rule.
//! * **U** — [`cse::evaluate_shared`] with [`Layout::OneTable`] and lets: one
//!   table for all rules, over the union of all centers. Kept as the measured
//!   counter-example: when rules share nothing it is slower than A.
//! * **D0** — [`Layout::Components`] without lets: one table per group of
//!   rules that share a side,
//!   still one fold per rule.
//! * **D** — the same with lets: each part two or more rules share is ANDed
//!   once into a plane.
//!
//! Every arm runs with binding propagation and dedup on and semijoin off,
//! because the shared path cannot apply a per-rule semijoin.
//!
//! Two answers are timed. `Exists` is mask-only: it isolates the folds.
//! `Bindings(Bag)` hands out each rule's kept slots over the prepared tables;
//! no row is built. The correctness checks, outside every timed region,
//! compare answers by folds: a row is read through its head template over
//! the lanes in place, never copied.

use std::collections::BTreeSet;
use std::time::Duration;

use crepe::crepe;
use crepe_plan_probe::cse::{Layout, SharedReport, evaluate_shared};
use crepe_plan_probe::logic::{Atom, RelId, Rule, Term, VarId};
use crepe_plan_probe::native::{PreparedStar, Store};
use crepe_plan_probe::optimize::{Options, ResultKind};
use crepe_plan_probe::pipeline::{Answers, Outcome, Report, demand, evaluate};
use mork_pathmap_probe::Semantics;

/// Rewrites on, semijoin off: the arms differ only in sharing.
const OPTS: Options = Options {
    propagate: true,
    dedup: true,
    semijoin: false,
};
const REPS: usize = 5;

/// One core: `R_g` (x, c) rows and `S_g` (c, z) rows.
type Core = (Vec<[u32; 2]>, Vec<[u32; 2]>);

/// A generated workload: relations and rules.
pub(crate) struct Workload {
    /// `rel[2g]` is `R_g` (x, c); `rel[2g + 1]` is `S_g` (c, z).
    pub cores: Vec<Core>,
    /// `T(c, k)`.
    pub t: Vec<[u32; 2]>,
    /// `group[k]`: the core rule `k` uses.
    pub group: Vec<usize>,
    pub rules: Vec<Rule>,
}

impl Workload {
    fn t_rel(&self) -> RelId {
        RelId(2 * self.cores.len() as u16)
    }
    fn out_rel(&self) -> RelId {
        RelId(2 * self.cores.len() as u16 + 1)
    }

    pub fn store(&self) -> Store {
        let mut relations = Vec::new();
        for (g, (r, s)) in self.cores.iter().enumerate() {
            relations.push((RelId(2 * g as u16), r.as_slice(), 1));
            relations.push((RelId(2 * g as u16 + 1), s.as_slice(), 1));
        }
        relations.push((self.t_rel(), self.t.as_slice(), 1));
        Store::from_rows(relations)
    }
}

/// SplitMix64, so fixtures are deterministic.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `rules` rules over `centers` centers, `rows` rows per core relation;
/// `shared` of the rules use core 0.
pub(crate) fn workload(rules: usize, shared: usize, centers: u32, rows: usize) -> Workload {
    let mut group = Vec::with_capacity(rules);
    let mut next = 1;
    for k in 0..rules {
        if k < shared {
            group.push(0);
        } else {
            group.push(next);
            next += 1;
        }
    }
    let ngroups = next.max(1);
    let rel = |g: usize, s: u64| -> Vec<[u32; 2]> {
        let mut v: Vec<[u32; 2]> = (0..rows as u64)
            .map(|i| {
                let h = mix(s ^ ((g as u64) << 32) ^ i);
                [(h % 50) as u32, ((h >> 20) % u64::from(centers)) as u32]
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let cores = (0..ngroups)
        .map(|g| {
            let r = rel(g, 1);
            let s = rel(g, 2).into_iter().map(|[z, c]| [c, z]).collect();
            (r, s)
        })
        .collect();
    let mut t = Vec::new();
    for c in 0..centers {
        for k in 0..rules as u32 {
            if mix((u64::from(c) << 32) | u64::from(k)).is_multiple_of(10) {
                t.push([c, k]);
            }
        }
    }
    let mut w = Workload {
        cores,
        t,
        group,
        rules: Vec::new(),
    };
    let (x, c, z) = (VarId(0), VarId(1), VarId(2));
    let t_rel = w.t_rel();
    let out = w.out_rel();
    w.rules = (0..rules)
        .map(|k| {
            let g = w.group[k] as u16;
            Rule {
                head: Atom {
                    rel: out,
                    terms: vec![Term::Const(k as u64), Term::Var(x), Term::Var(z)],
                },
                body: vec![
                    Atom {
                        rel: RelId(2 * g),
                        terms: vec![Term::Var(x), Term::Var(c)],
                    },
                    Atom {
                        rel: RelId(2 * g + 1),
                        terms: vec![Term::Var(c), Term::Var(z)],
                    },
                    Atom {
                        rel: t_rel,
                        terms: vec![Term::Var(c), Term::Const(k as u64)],
                    },
                ],
            }
        })
        .collect();
    w
}

crepe! {
    @input
    struct G(u32, u32);
    @input
    struct RG(u32, u32, u32);
    @input
    struct SG(u32, u32, u32);
    @input
    struct T(u32, u32);
    @output
    struct Out(u32, u32, u32);
    Out(k, x, z) <- G(k, g), RG(g, x, c), SG(g, c, z), T(c, k);
}

ascent::ascent! {
    struct CseAscent;
    relation g(u32, u32);
    relation rg(u32, u32, u32);
    relation sg(u32, u32, u32);
    relation t(u32, u32);
    relation out(u32, u32, u32);
    out(k, x, z) <-- g(k, gg), rg(gg, x, c), sg(gg, c, z), t(c, k);
}

type Tuples = BTreeSet<(u32, u32, u32)>;

pub(crate) fn crepe_answer(w: &Workload) -> Tuples {
    let mut rt = Crepe::new();
    rt.extend(
        w.group
            .iter()
            .enumerate()
            .map(|(k, &g)| G(k as u32, g as u32)),
    );
    for (g, (r, s)) in w.cores.iter().enumerate() {
        rt.extend(r.iter().map(|&[x, c]| RG(g as u32, x, c)));
        rt.extend(s.iter().map(|&[c, z]| SG(g as u32, c, z)));
    }
    rt.extend(w.t.iter().map(|&[c, k]| T(c, k)));
    let (out,) = rt.run();
    out.into_iter().map(|Out(k, x, z)| (k, x, z)).collect()
}

pub(crate) fn ascent_answer(w: &Workload) -> Tuples {
    let mut prog = CseAscent {
        g: w.group
            .iter()
            .enumerate()
            .map(|(k, &g)| (k as u32, g as u32))
            .collect(),
        t: w.t.iter().map(|&[c, k]| (c, k)).collect(),
        ..CseAscent::default()
    };
    for (g, (r, s)) in w.cores.iter().enumerate() {
        prog.rg.extend(r.iter().map(|&[x, c]| (g as u32, x, c)));
        prog.sg.extend(s.iter().map(|&[c, z]| (g as u32, c, z)));
    }
    prog.run();
    prog.out.into_iter().collect()
}

fn answers(o: &Outcome) -> &Answers {
    match o {
        Outcome::Bindings(a) => a,
        other => panic!("expected bindings, got {other:?}"),
    }
}

/// Two answers agree: same total, and every row one of them reads (through
/// its head template, over the lanes in place) is answered the same number
/// of times by the other. Folds; no row is built.
pub(crate) fn same(a: &Outcome, ta: &[PreparedStar], b: &Outcome, tb: &[PreparedStar]) -> bool {
    let (a, b) = (answers(a), answers(b));
    if a.total(ta) != b.total(tb) {
        return false;
    }
    let mut ok = true;
    a.for_each(ta, |get, _| {
        let me = |c: usize| Some(get(c));
        ok &= a.matching(ta, me) == b.matching(tb, me);
    });
    ok
}

/// The answer agrees with an oracle's set: every oracle tuple is answered,
/// and every answered row is an oracle tuple.
pub(crate) fn same_set(o: &Outcome, t: &[PreparedStar], set: &Tuples) -> bool {
    let a = answers(o);
    let mut ok = set.iter().all(|&(x, y, z)| {
        let row = [x, y, z];
        a.matching(t, |c| row.get(c).map(|&v| u64::from(v))) > 0
    });
    a.for_each(t, |get, _| {
        ok &= set.contains(&(get(0) as u32, get(1) as u32, get(2) as u32));
    });
    ok
}

/// Against rows written out by a test: same total, each row answered its
/// multiplicity.
#[cfg(test)]
pub(crate) fn same_bag(o: &Outcome, t: &[PreparedStar], want: &[(Vec<u64>, u64)]) -> bool {
    let a = answers(o);
    a.total(t) == want.iter().map(|(_, m)| m).sum::<u64>()
        && want
            .iter()
            .all(|(row, m)| a.matching(t, |c| row.get(c).copied()) == *m)
}

fn arm_a(w: &Workload, kind: ResultKind) -> (Outcome, Report) {
    let d = demand(w.out_rel(), vec![None; 3]);
    evaluate(&w.rules, &d, kind, OPTS, &w.store()).expect("arm A")
}

fn arm_d(w: &Workload, kind: ResultKind, layout: Layout, sharing: bool) -> (Outcome, SharedReport) {
    let d = demand(w.out_rel(), vec![None; 3]);
    evaluate_shared(&w.rules, &d, kind, OPTS, &w.store(), layout, sharing).expect("arm D")
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let rules = env("BENCH_RULES", 256);
    let centers = env("BENCH_CENTERS", 4_000) as u32;
    let rows = env("BENCH_ROWS", 8_000);
    let reps = env("BENCH_REPS", REPS);
    let oracle = env("BENCH_ORACLE", 1) == 1;
    println!(
        "rules={rules} centers={centers} rows/core relation={rows} reps={reps} avx512f={}",
        cfg!(target_feature = "avx512f")
    );
    for pct in [0usize, 50, 90] {
        let shared = rules * pct / 100;
        let w = workload(rules, shared, centers, rows);
        let bag = ResultKind::Bindings(Semantics::Bag);

        // Correctness, outside every timed region.
        let (a, ra) = arm_a(&w, bag);
        let (d0, r0) = arm_d(&w, bag, Layout::Components, false);
        let (d, rd) = arm_d(&w, bag, Layout::Components, true);
        assert!(same(&a, &ra.tables, &d0, &r0.tables), "D0 vs A");
        assert!(same(&a, &ra.tables, &d, &rd.tables), "D vs A");
        let checked = if oracle {
            let set = crepe_answer(&w);
            assert!(same_set(&a, &ra.tables, &set), "A vs crepe");
            assert_eq!(set, ascent_answer(&w), "crepe vs ascent");
            format!("crepe = ascent = A = D0 = D ({} oracle tuples)", set.len())
        } else {
            "A = D0 = D (oracles skipped)".to_string()
        };
        println!("\nshared core {pct}% ({shared} of {rules} rules): {checked}");
        println!(
            "{:<4} {:>9} {:>10} {:>9} {:>11} {:>8} {:>7} {:>6} {:>10} {:>11}",
            "arm",
            "kind",
            "prepare",
            "execute",
            "rows prep.",
            "folds",
            "ops",
            "lets",
            "planes rd",
            "plane bytes"
        );
        for (kname, kind) in [("exists", ResultKind::Exists), ("bindings", bag)] {
            let mut ta = (Vec::new(), Vec::new());
            let mut last_a = None;
            for _ in 0..reps {
                let (_, r) = arm_a(&w, kind);
                ta.0.push(r.costs.prepare);
                ta.1.push(r.costs.execute);
                last_a = Some(r);
            }
            let ra = last_a.expect("reps >= 1");
            println!(
                "{:<4} {:>9} {:>8.2}ms {:>7.2}ms {:>11} {:>8} {:>7} {:>6} {:>10} {:>11}",
                "A",
                kname,
                ms(median(ta.0)),
                ms(median(ta.1)),
                ra.rows_prepared,
                ra.programs_run,
                ra.programs.iter().map(|p| p.ops.len()).sum::<usize>(),
                "-",
                "-",
                0
            );
            for (arm, layout, sharing) in [
                ("U", Layout::OneTable, true),
                ("D0", Layout::Components, false),
                ("D", Layout::Components, true),
            ] {
                let mut t = (Vec::new(), Vec::new());
                let mut split = (Vec::new(), Vec::new(), Vec::new());
                let mut last = None;
                for _ in 0..reps {
                    let (_, r) = arm_d(&w, kind, layout, sharing);
                    t.0.push(r.prepare);
                    t.1.push(r.execute);
                    split.0.push(r.share);
                    split.1.push(r.fold);
                    split.2.push(r.terminal);
                    last = Some(r);
                }
                let r = last.expect("reps >= 1");
                println!(
                    "{:<4} {:>9} {:>8.2}ms {:>7.2}ms {:>11} {:>8} {:>7} {:>6} {:>10} {:>11}",
                    arm,
                    kname,
                    ms(median(t.0)),
                    ms(median(t.1)),
                    r.rows_prepared,
                    r.cost.programs,
                    r.cost.passes,
                    r.lets,
                    r.cost.plane_reads,
                    r.cost.plane_bytes
                );
                println!(
                    "     execute = share {:.2}ms + folds {:.2}ms + terminal {:.2}ms",
                    ms(median(split.0)),
                    ms(median(split.1)),
                    ms(median(split.2))
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crepe_plan_probe::cse::{CseRefusal, execute_shared, prepare_shared};
    use crepe_plan_probe::logic::RelId as Rid;
    use crepe_plan_probe::pipeline::{PipelineRefusal, plan};
    use crepe_plan_probe::reference::eval;
    use mork_pathmap_probe::join::JoinRefusal;

    const RULES: usize = 24;
    const CENTERS: u32 = 40;
    const ROWS: usize = 80;

    fn reference_bag(w: &Workload) -> Vec<(Vec<u64>, u64)> {
        let store = w.store();
        let rows = |r: Rid| -> Vec<Vec<u64>> {
            let v = store.view(r);
            (0..v.len())
                .map(|i| v.cols.iter().map(|c| u64::from(c[i])).collect())
                .collect()
        };
        eval(&w.rules, w.out_rel(), &[None, None, None], &rows)
            .into_iter()
            .collect()
    }

    /// FAILS IF: the shared table, the lets or the per-rule tuple read change
    /// any answer or multiplicity, at any shared-core rate.
    #[test]
    fn every_arm_matches_the_reference_crepe_and_ascent() {
        for shared in [0, RULES / 2, RULES * 9 / 10, RULES] {
            let w = workload(RULES, shared, CENTERS, ROWS);
            let bag = ResultKind::Bindings(Semantics::Bag);
            let (a, ra) = arm_a(&w, bag);
            let (d0, r0) = arm_d(&w, bag, Layout::Components, false);
            let (d, rd) = arm_d(&w, bag, Layout::Components, true);
            let want = reference_bag(&w);
            assert!(want.len() > 50, "fixture too thin: {} rows", want.len());
            assert!(
                want.iter().any(|(_, m)| *m > 1),
                "fixture has no multiplicity above 1"
            );
            assert!(
                same_bag(&a, &ra.tables, &want),
                "A vs reference, shared={shared}"
            );
            assert!(
                same(&d0, &r0.tables, &a, &ra.tables),
                "D0 vs A, shared={shared}"
            );
            assert!(
                same(&d, &rd.tables, &a, &ra.tables),
                "D vs A, shared={shared}"
            );
            let set = crepe_answer(&w);
            assert_eq!(set.len(), want.len(), "crepe, shared={shared}");
            assert!(same_set(&a, &ra.tables, &set), "crepe, shared={shared}");
            assert_eq!(set, ascent_answer(&w), "ascent, shared={shared}");
            // `Out(k, x, z)` has two free columns (x, z): a set count would
            // need a seen-set over value pairs, so every arm refuses it.
            let d = demand(w.out_rel(), vec![None; 3]);
            let set_count = ResultKind::Count(Semantics::Set);
            assert!(matches!(
                evaluate(&w.rules, &d, set_count, OPTS, &w.store()),
                Err(PipelineRefusal::Set { .. })
            ));
            for layout in [Layout::OneTable, Layout::Components] {
                for sharing in [true, false] {
                    assert!(matches!(
                        evaluate_shared(&w.rules, &d, set_count, OPTS, &w.store(), layout, sharing),
                        Err(CseRefusal::Pipeline(PipelineRefusal::Set { .. }))
                    ));
                }
            }
            for kind in [ResultKind::Exists, ResultKind::Count(Semantics::Bag)] {
                let (want, _) = arm_a(&w, kind);
                for layout in [Layout::OneTable, Layout::Components] {
                    assert_eq!(arm_d(&w, kind, layout, true).0, want, "{kind:?}");
                    assert_eq!(arm_d(&w, kind, layout, false).0, want, "{kind:?}");
                }
            }
        }
    }

    /// FAILS IF: a demand that propagation does not push into the rules is
    /// not applied to the shared path's answers (the post filter), or one
    /// it does push changes them.
    #[test]
    fn a_bound_demand_filters_the_shared_answers() {
        let w = workload(RULES, RULES / 2, CENTERS, ROWS);
        let bag = ResultKind::Bindings(Semantics::Bag);
        let x = w.cores[0].0[0][0];
        for (bound, propagate) in [
            (vec![None, Some(u64::from(x)), None], false),
            (vec![None, Some(u64::from(x)), None], true),
            (vec![Some(3), None, None], false),
        ] {
            let opts = Options { propagate, ..OPTS };
            let d = demand(w.out_rel(), bound.clone());
            let (a, ra) = evaluate(&w.rules, &d, bag, opts, &w.store()).expect("A");
            assert!(
                answers(&a).total(&ra.tables) > 0,
                "demand {bound:?} selects nothing"
            );
            let store = w.store();
            let rows = |r: Rid| -> Vec<Vec<u64>> {
                let v = store.view(r);
                (0..v.len())
                    .map(|i| v.cols.iter().map(|c| u64::from(c[i])).collect())
                    .collect()
            };
            let want: Vec<(Vec<u64>, u64)> = eval(&w.rules, w.out_rel(), &bound, &rows)
                .into_iter()
                .collect();
            assert!(same_bag(&a, &ra.tables, &want), "A vs reference, {bound:?}");
            for sharing in [false, true] {
                let (dd, rdd) = evaluate_shared(
                    &w.rules,
                    &d,
                    bag,
                    opts,
                    &w.store(),
                    Layout::Components,
                    sharing,
                )
                .expect("D");
                assert!(
                    same(&dd, &rdd.tables, &a, &ra.tables),
                    "sharing={sharing}, {bound:?}, propagate={propagate}"
                );
            }
        }
    }

    /// FAILS IF: a shared core is not lifted, is read once per rule anyway,
    /// or the shared table lays the core out once per rule.
    #[test]
    fn a_shared_core_is_laid_out_once_and_anded_once() {
        let shared = RULES * 9 / 10;
        let w = workload(RULES, shared, CENTERS, ROWS);
        let (_, a) = arm_a(&w, ResultKind::Exists);
        let (_, d0) = arm_d(&w, ResultKind::Exists, Layout::Components, false);
        let (_, d) = arm_d(&w, ResultKind::Exists, Layout::Components, true);
        // One fold per rule in A and D0; D adds one per let.
        assert_eq!(a.programs_run, RULES);
        assert_eq!(d0.cost.programs, RULES);
        assert_eq!(d.lets, 1, "the shared core R_0 ∧ S_0 is one let");
        assert_eq!(d.cost.programs, RULES + 1);
        // D0 reads three planes per rule; D reads the core's two once, then
        // two per shared rule (let, T) and three per unshared rule.
        assert_eq!(d0.cost.plane_reads, 3 * RULES);
        assert_eq!(d.cost.plane_reads, 2 + 2 * shared + 3 * (RULES - shared));
        assert_eq!(d.cost.planes_materialized, 1);
        // Distinct sides: one core pair per group, one T side per rule.
        let groups = 1 + (RULES - shared);
        assert_eq!(d0.sides, 2 * groups + RULES);
        // One table for the rules on core 0, one per other rule.
        assert_eq!(d0.groups, groups);
        // A lays the shared core out once per rule that uses it.
        let core_rows = w.cores[0].0.len() + w.cores[0].1.len();
        assert!(
            a.rows_prepared >= d0.rows_prepared + (shared - 1) * core_rows,
            "A {} vs shared {}",
            a.rows_prepared,
            d0.rows_prepared
        );
    }

    /// FAILS IF: lets are formed, or cost changes, when no two rules share
    /// a part.
    #[test]
    fn nothing_is_lifted_when_no_core_is_shared() {
        let w = workload(RULES, 0, CENTERS, ROWS);
        let (_, d0) = arm_d(&w, ResultKind::Exists, Layout::Components, false);
        let (_, d) = arm_d(&w, ResultKind::Exists, Layout::Components, true);
        assert_eq!(d.lets, 0);
        assert_eq!(d.cost, d0.cost);
        // Every rule keeps a table of its own, laid out from exactly the rows
        // the per-rule path lays out; one table would widen all of them.
        let (_, a) = arm_a(&w, ResultKind::Exists);
        assert_eq!(d0.groups, RULES);
        assert_eq!(d0.rows_prepared, a.rows_prepared);
        let (_, one) = arm_d(&w, ResultKind::Exists, Layout::OneTable, false);
        assert_eq!(one.groups, 1);
        // In one table every rule's fold runs at the union width.
        assert!(
            one.domain * RULES > d0.domain,
            "union {} x {RULES} rules vs per-rule widths {}",
            one.domain,
            d0.domain
        );
    }

    /// FAILS IF: tables prepared for one plan are run with another plan of
    /// the same shape (same rule count, same generations), or with a plan
    /// holding one star more or one fewer.
    #[test]
    fn tables_prepared_for_another_plan_are_refused() {
        let kind = ResultKind::Exists;
        let w1 = workload(RULES, RULES / 2, CENTERS, ROWS);
        let w2 = workload(RULES, RULES / 2 - 1, CENTERS, ROWS);
        let d1 = demand(w1.out_rel(), vec![None; 3]);
        let d2 = demand(w2.out_rel(), vec![None; 3]);
        let p1 = plan(&w1.rules, &d1, kind, OPTS).expect("plan 1");
        let p2 = plan(&w2.rules, &d2, kind, OPTS).expect("plan 2");
        assert_eq!(p1.stars.len(), p2.stars.len(), "same shape");
        let shared = prepare_shared(&p1.stars, &w1.store(), Layout::OneTable);
        let gens: Vec<Vec<u64>> = shared
            .groups
            .iter()
            .map(|g| g.prepared.keys.iter().map(|_| 1).collect())
            .collect();
        assert!(execute_shared(&p1, &shared, kind, &gens, true).is_ok());
        assert_eq!(
            execute_shared(&p2, &shared, kind, &gens, true).err(),
            Some(CseRefusal::NoCoordinateWitness)
        );
        // A star no group covers.
        let mut long = p1.clone();
        long.stars.push(long.stars[0].clone());
        long.optimized.rules.push(long.optimized.rules[0].clone());
        assert_eq!(
            execute_shared(&long, &shared, kind, &gens, true).err(),
            Some(CseRefusal::NoCoordinateWitness)
        );
        let mut short = p1.clone();
        short.stars.pop();
        short.optimized.rules.pop();
        assert_eq!(
            execute_shared(&short, &shared, kind, &gens, true).err(),
            Some(CseRefusal::NoCoordinateWitness)
        );
    }

    /// FAILS IF: the shared path runs against a side whose generation moved.
    #[test]
    fn a_stale_side_is_refused() {
        let w = workload(RULES, RULES / 2, CENTERS, ROWS);
        let store = w.store();
        let d = demand(w.out_rel(), vec![None; 3]);
        let kind = ResultKind::Exists;
        let planned = plan(&w.rules, &d, kind, OPTS).expect("plan");
        let shared = prepare_shared(&planned.stars, &store, Layout::Components);
        let mut generations: Vec<Vec<u64>> = shared
            .groups
            .iter()
            .map(|g| g.prepared.keys.iter().map(|_| 1).collect())
            .collect();
        assert!(execute_shared(&planned, &shared, kind, &generations, true).is_ok());
        generations[0][0] = 2;
        assert!(matches!(
            execute_shared(&planned, &shared, kind, &generations, true),
            Err(CseRefusal::Table(JoinRefusal::Stale { .. }))
        ));
        generations[0][0] = 1;
        generations[0].pop();
        assert!(matches!(
            execute_shared(&planned, &shared, kind, &generations, true),
            Err(CseRefusal::Generations { .. })
        ));
    }
}
