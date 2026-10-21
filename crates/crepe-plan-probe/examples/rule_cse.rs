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
//! * **D0** — [`cse::evaluate_shared`] without lets: one table for all rules,
//!   still one fold per rule.
//! * **D** — the same with lets: each part two or more rules share is ANDed
//!   once into a plane.
//!
//! Every arm runs with binding propagation and dedup on and semijoin off,
//! because the shared path cannot apply a per-rule semijoin.
//!
//! Two answers are timed. `Exists` is mask-only: it isolates the folds.
//! `Bindings(Bag)` is the full answer with multiplicities; it adds reading
//! the tuples, which is the same work in every arm.

use std::collections::BTreeSet;
use std::time::Duration;

use crepe::crepe;
use crepe_plan_probe::cse::{SharedReport, evaluate_shared};
use crepe_plan_probe::logic::{Atom, RelId, Rule, Term, VarId};
use crepe_plan_probe::native::Store;
use crepe_plan_probe::optimize::{Options, ResultKind};
use crepe_plan_probe::pipeline::{Outcome, Report, demand, evaluate};
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

    pub fn store(&self) -> Store<'_> {
        let mut relations = Vec::new();
        for (g, (r, s)) in self.cores.iter().enumerate() {
            relations.push((RelId(2 * g as u16), r.as_slice(), 1));
            relations.push((RelId(2 * g as u16 + 1), s.as_slice(), 1));
        }
        relations.push((self.t_rel(), self.t.as_slice(), 1));
        Store { relations }
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

pub(crate) fn as_set(o: &Outcome) -> Tuples {
    match o {
        Outcome::Bindings(b) => b
            .iter()
            .map(|(r, _)| (r[0] as u32, r[1] as u32, r[2] as u32))
            .collect(),
        other => panic!("expected bindings, got {other:?}"),
    }
}

fn arm_a(w: &Workload, kind: ResultKind) -> (Outcome, Report) {
    let d = demand(w.out_rel(), vec![None; 3]);
    evaluate(&w.rules, &d, kind, OPTS, &w.store()).expect("arm A")
}

fn arm_d(w: &Workload, kind: ResultKind, sharing: bool) -> (Outcome, SharedReport) {
    let d = demand(w.out_rel(), vec![None; 3]);
    evaluate_shared(&w.rules, &d, kind, OPTS, &w.store(), sharing).expect("arm D")
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
        let (a, _) = arm_a(&w, bag);
        let (d0, _) = arm_d(&w, bag, false);
        let (d, _) = arm_d(&w, bag, true);
        assert_eq!(a, d0, "D0 vs A");
        assert_eq!(a, d, "D vs A");
        let checked = if oracle {
            let set = as_set(&a);
            assert_eq!(set, crepe_answer(&w), "A vs crepe");
            assert_eq!(set, ascent_answer(&w), "A vs ascent");
            format!("crepe = ascent = A = D0 = D ({} tuples)", set.len())
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
            for (arm, sharing) in [("D0", false), ("D", true)] {
                let mut t = (Vec::new(), Vec::new());
                let mut split = (Vec::new(), Vec::new(), Vec::new());
                let mut last = None;
                for _ in 0..reps {
                    let (_, r) = arm_d(&w, kind, sharing);
                    t.0.push(r.prepare);
                    t.1.push(r.execute);
                    split.0.push(r.share);
                    split.1.push(r.fold);
                    split.2.push(r.tuples);
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
                    "     execute = share {:.2}ms + folds {:.2}ms + tuples {:.2}ms",
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
    use crepe_plan_probe::pipeline::plan;
    use crepe_plan_probe::reference::eval;
    use mork_pathmap_probe::join::JoinRefusal;

    const RULES: usize = 24;
    const CENTERS: u32 = 40;
    const ROWS: usize = 80;

    fn reference_bag(w: &Workload) -> Vec<(Vec<u64>, u64)> {
        let store = w.store();
        let rows = |r: Rid| -> Vec<Vec<u64>> {
            store
                .view(r)
                .0
                .iter()
                .map(|p| p.iter().map(|&v| u64::from(v)).collect())
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
            let (a, _) = arm_a(&w, bag);
            let (d0, _) = arm_d(&w, bag, false);
            let (d, _) = arm_d(&w, bag, true);
            let Outcome::Bindings(ab) = &a else {
                panic!("bindings")
            };
            assert!(ab.len() > 50, "fixture too thin: {} tuples", ab.len());
            assert!(
                ab.iter().any(|(_, m)| *m > 1),
                "fixture has no multiplicity above 1"
            );
            assert_eq!(*ab, reference_bag(&w), "A vs reference, shared={shared}");
            assert_eq!(d0, a, "D0 vs A, shared={shared}");
            assert_eq!(d, a, "D vs A, shared={shared}");
            let set = as_set(&a);
            assert_eq!(set, crepe_answer(&w), "crepe, shared={shared}");
            assert_eq!(set, ascent_answer(&w), "ascent, shared={shared}");
            for kind in [
                ResultKind::Exists,
                ResultKind::Count(Semantics::Set),
                ResultKind::Count(Semantics::Bag),
            ] {
                let (want, _) = arm_a(&w, kind);
                assert_eq!(arm_d(&w, kind, true).0, want, "{kind:?}");
                assert_eq!(arm_d(&w, kind, false).0, want, "{kind:?}");
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
            let (a, _) = evaluate(&w.rules, &d, bag, opts, &w.store()).expect("A");
            let Outcome::Bindings(ab) = &a else {
                panic!("bindings")
            };
            assert!(!ab.is_empty(), "demand {bound:?} selects nothing");
            let store = w.store();
            let rows = |r: Rid| -> Vec<Vec<u64>> {
                store
                    .view(r)
                    .0
                    .iter()
                    .map(|p| p.iter().map(|&v| u64::from(v)).collect())
                    .collect()
            };
            let want: Vec<(Vec<u64>, u64)> = eval(&w.rules, w.out_rel(), &bound, &rows)
                .into_iter()
                .collect();
            assert_eq!(*ab, want, "A vs reference, {bound:?}");
            for sharing in [false, true] {
                let (dd, _) =
                    evaluate_shared(&w.rules, &d, bag, opts, &w.store(), sharing).expect("D");
                assert_eq!(dd, a, "sharing={sharing}, {bound:?}, propagate={propagate}");
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
        let (_, d0) = arm_d(&w, ResultKind::Exists, false);
        let (_, d) = arm_d(&w, ResultKind::Exists, true);
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
        let (_, d0) = arm_d(&w, ResultKind::Exists, false);
        let (_, d) = arm_d(&w, ResultKind::Exists, true);
        assert_eq!(d.lets, 0);
        assert_eq!(d.cost, d0.cost);
    }

    /// FAILS IF: the shared path runs against a side whose generation moved.
    #[test]
    fn a_stale_side_is_refused() {
        let w = workload(RULES, RULES / 2, CENTERS, ROWS);
        let store = w.store();
        let d = demand(w.out_rel(), vec![None; 3]);
        let kind = ResultKind::Exists;
        let planned = plan(&w.rules, &d, kind, OPTS).expect("plan");
        let shared = prepare_shared(&planned.stars, &store);
        let mut generations: Vec<u64> = shared.prepared.keys.iter().map(|_| 1).collect();
        assert!(execute_shared(&planned, &shared, kind, &generations, true).is_ok());
        generations[0] = 2;
        assert!(matches!(
            execute_shared(&planned, &shared, kind, &generations, true),
            Err(CseRefusal::Table(JoinRefusal::Stale { .. }))
        ));
        generations.pop();
        assert!(matches!(
            execute_shared(&planned, &shared, kind, &generations, true),
            Err(CseRefusal::Generations { .. })
        ));
    }
}
