//! Cost split of the `two_way` join, native against the MORK oracle, so the
//! oracle's reference costs are not one lump. Run in release from this crate's
//! directory:
//!
//! ```text
//! cargo run --release --example mork_split [n]
//! ```
//!
//! `T(x, y, z) <- R(x, y), S(y, z)` with `R(i, i % (n/4))` for `i` in `0..n`
//! and `S(y, 500_000 + y)` for `y` in `0..n/4` (default `n = 10_000`). Every
//! `R` row meets exactly one `S` row, so there are `n` answers.
//!
//! Native: `pipeline::evaluate` with `Count(BAG)` (the terminal) and with
//! `Bindings(BAG)` (answer production); `Report::costs` separates the
//! preparation (laying the relations out at one coordinate) from the
//! execution. MORK: `OracleCost` (fact encoding and build, unification only,
//! `exec` plus decode). MORK stores sets and its numbers are reference costs
//! for an oracle, not gates. Before any timing the native answer set is
//! asserted equal to the MORK set.

use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use crepe_plan_probe::logic::{Labels, RelId, Rule, resolve};
use crepe_plan_probe::mork_api::to_oracle;
use crepe_plan_probe::native::Store;
use crepe_plan_probe::optimize::{Demand, Options, ResultKind};
use crepe_plan_probe::pipeline::{Outcome, evaluate};
use crepe_plan_probe::programs::two_way;
use mork_pathmap_probe::Semantics;
use mork_pathmap_probe::boundary::{OracleCost, OracleFacts, mork_rules};

const RUNS: usize = 5;
const BAG: Semantics = Semantics::Bag;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// Microseconds, and nanoseconds per input row.
fn show(label: &str, d: Duration, rows: usize) -> String {
    format!(
        "  {label:<28} {:>12.1} us {:>10.1} ns/row",
        us(d),
        d.as_secs_f64() * 1e9 / rows as f64
    )
}

struct Native {
    wall: Duration,
    optimize: Duration,
    map_and_admit: Duration,
    prepare: Duration,
    execute: Duration,
}

fn run_native(rules: &[Rule], demand: &Demand, kind: ResultKind, store: &Store) -> Native {
    let mut runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let (out, rep) =
            evaluate(rules, demand, kind, Options::ALL, store).expect("native answers");
        let wall = t.elapsed();
        black_box((&out, &rep.costs));
        runs.push((wall, rep.costs));
    }
    let col = |f: &dyn Fn(&(Duration, crepe_plan_probe::pipeline::Costs)) -> Duration| {
        median(runs.iter().map(f).collect())
    };
    Native {
        wall: col(&|r| r.0),
        optimize: col(&|r| r.1.optimize),
        map_and_admit: col(&|r| r.1.map_and_admit),
        prepare: col(&|r| r.1.prepare),
        execute: col(&|r| r.1.execute),
    }
}

fn print_native(name: &str, n: &Native, rows: usize) {
    println!("{name}");
    println!("{}", show("wall (evaluate)", n.wall, rows));
    println!("{}", show("  optimize", n.optimize, rows));
    println!("{}", show("  map and admit", n.map_and_admit, rows));
    println!("{}", show("  prepare (layout)", n.prepare, rows));
    println!("{}", show("  execute", n.execute, rows));
}

fn main() {
    let n: u32 = std::env::args()
        .nth(1)
        .map(|a| a.parse().expect("n is an integer"))
        .unwrap_or(10_000);
    let key_domain = n / 4;
    assert!(key_domain > 0, "n must be at least 4");

    let plan = two_way::crepe_plan();
    let labels = Labels::of(&plan);
    let rules: Vec<Rule> = resolve(&plan, &labels)
        .into_iter()
        .map(|r| r.expect("admitted rule"))
        .collect();
    let (rel_r, rel_s): (RelId, RelId) = (labels.id("R").unwrap(), labels.id("S").unwrap());
    let r: Vec<[u32; 2]> = (0..n).map(|i| [i, i % key_domain]).collect();
    let s: Vec<[u32; 2]> = (0..key_domain).map(|y| [y, 500_000 + y]).collect();
    let store = Store::from_rows(vec![(rel_r, &r, 1), (rel_s, &s, 1)]);
    let input_rows = r.len() + s.len();
    let demand = Demand {
        goal: labels.id("T").unwrap(),
        bound: vec![None, None, None],
    };

    // The MORK oracle.
    let oracle_rules: Vec<_> = rules
        .iter()
        .filter(|rule| rule.head.rel == demand.goal)
        .map(|rule| to_oracle(rule).expect("admitted"))
        .collect();
    let facts = [
        OracleFacts {
            rel: rel_r.0,
            arity: 2,
            view: store.view(rel_r),
        },
        OracleFacts {
            rel: rel_s.0,
            arity: 2,
            view: store.view(rel_s),
        },
    ];
    let mut costs: Vec<OracleCost> = Vec::with_capacity(RUNS);
    let mut mork_set = BTreeSet::new();
    for _ in 0..RUNS {
        let (set, cost) = mork_rules(&facts, &oracle_rules).expect("MORK answers");
        mork_set = set.rows;
        costs.push(cost);
    }
    let matches = costs[0].matches;
    assert!(
        costs.iter().all(|c| c.matches == matches),
        "the match count is deterministic"
    );

    // Parity before any timing: the native answer set equals MORK's.
    let (out, rep) = evaluate(
        &rules,
        &demand,
        ResultKind::Bindings(BAG),
        Options::ALL,
        &store,
    )
    .expect("native answers");
    let Outcome::Bindings(answers) = out else {
        panic!("a bindings answer");
    };
    let mut native_set: BTreeSet<Vec<u32>> = BTreeSet::new();
    // Comparison only (not timed): rows are built here to compare sets with MORK.
    answers.for_each(&rep.tables, |get, _weight| {
        native_set.insert(
            (0..3)
                .map(|c| u32::try_from(get(c)).expect("value fits u32"))
                .collect(),
        );
    });
    assert!(!native_set.is_empty(), "vacuous fixture");
    assert_eq!(native_set, mork_set, "native answers equal MORK's set");

    println!(
        "mork_split: release build, n = {n}, runs = {RUNS} (median), input rows = {input_rows}, answers = {}",
        native_set.len()
    );
    println!("MORK numbers are oracle reference costs, not gates.");
    println!("units: microseconds, and nanoseconds per input row\n");

    let count = run_native(&rules, &demand, ResultKind::Count(BAG), &store);
    print_native("native, Count(BAG) (terminal)", &count, input_rows);
    let bindings = run_native(&rules, &demand, ResultKind::Bindings(BAG), &store);
    print_native(
        "native, Bindings(BAG) (deferred handle; enumeration below)",
        &bindings,
        input_rows,
    );
    // The bindings outcome is a deferred handle over the prepared tables; its
    // `execute` cost does not include reading the answers. This times one
    // full enumeration: every answer's three columns and weight are read and
    // folded into a checksum, nothing is allocated per answer.
    let mut enum_runs = Vec::with_capacity(RUNS);
    let mut visited = 0u64;
    for _ in 0..RUNS {
        let t = Instant::now();
        let mut sum = 0u64;
        let mut seen = 0u64;
        answers.for_each(&rep.tables, |get, weight| {
            sum = sum.wrapping_add(get(0) ^ get(1).rotate_left(21) ^ get(2).rotate_left(42));
            sum = sum.wrapping_add(weight);
            seen += 1;
        });
        enum_runs.push(t.elapsed());
        black_box(sum);
        visited = seen;
    }
    assert_eq!(
        visited as usize,
        native_set.len(),
        "one visit per answer under this fixture"
    );
    println!(
        "{}",
        show("  enumerate all answers", median(enum_runs), input_rows)
    );

    let med = |f: &dyn Fn(&OracleCost) -> Duration| median(costs.iter().map(f).collect());
    println!("MORK oracle (set semantics, matches = {matches})");
    println!(
        "{}",
        show("encode + build", med(&|c| c.encode_build), input_rows)
    );
    println!(
        "{}",
        show(
            "match only (query_multi)",
            med(&|c| c.match_only),
            input_rows
        )
    );
    println!(
        "{}",
        show("exec + decode", med(&|c| c.exec_decode), input_rows)
    );
}
