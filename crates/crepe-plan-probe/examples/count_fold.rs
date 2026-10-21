//! The native count terminal folded against the enumerating one (gate G11 of
//! the terminal-fold spec). Run in release from this crate's directory:
//!
//! ```text
//! cargo run --release --example count_fold
//! ```
//!
//! `T(x, y, z) <- R(x, y), S(y, z)` over 50,000 centers `y`; every center has
//! 4 `R` rows and 4 `S` rows, one of each a duplicate, so bag and set answers
//! differ. Each answer is checked against the enumerating terminal before it
//! is timed. Timings are medians of 11 runs after 2 warm-ups. Only the
//! unfiltered Count(Bag) ratio is gated (kill below 10x); it is the best case
//! (a factor is a slice length), so the filtered and Set ratios are reported
//! beside it and no general speed-up is claimed from the gated one.

use std::hint::black_box;
use std::time::{Duration, Instant};

use crepe_plan_probe::logic::{Labels, resolve};
use crepe_plan_probe::native::{Store, TerminalPath, prepare};
use crepe_plan_probe::optimize::{Demand, Options, ResultKind};
use crepe_plan_probe::pipeline::{Outcome, execute_with, plan};
use crepe_plan_probe::programs::two_way;
use mork_pathmap_probe::Semantics;

const CENTERS: u32 = 50_000;
const REPS: usize = 11;
const WARMUP: usize = 2;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let p = two_way::crepe_plan();
    let labels = Labels::of(&p);
    let rules: Vec<_> = resolve(&p, &labels)
        .into_iter()
        .map(|r| r.expect("admitted"))
        .collect();
    let mut r = Vec::new();
    let mut s = Vec::new();
    for y in 0..CENTERS {
        for k in [0, 1, 2, 2] {
            r.push([y * 4 + k, y]);
            s.push([y, y % 97 + k]);
        }
    }
    let st = Store {
        relations: vec![
            (labels.id("R").unwrap(), &r, 1),
            (labels.id("S").unwrap(), &s, 1),
        ],
    };
    let goal = labels.id("T").unwrap();
    let cases = [
        (
            "Count(Bag)",
            ResultKind::Count(Semantics::Bag),
            vec![None, None, None],
            Options::ALL,
            true,
        ),
        (
            "Count(Bag), filtered",
            ResultKind::Count(Semantics::Bag),
            vec![None, None, Some(5)],
            Options::NONE,
            false,
        ),
        (
            "Count(Set)",
            ResultKind::Count(Semantics::Set),
            vec![None, None, None],
            Options::ALL,
            false,
        ),
        (
            "Count(Set), filtered",
            ResultKind::Count(Semantics::Set),
            vec![None, None, Some(5)],
            Options::NONE,
            false,
        ),
    ];
    println!(
        "{:<22} | {:>10} | {:>12} {:>12} | {:>8} | gate",
        "query", "answer", "folded ms", "enumer. ms", "ratio"
    );
    let mut killed = false;
    for (name, kind, bound, opts, gated) in cases {
        let d = Demand { goal, bound };
        let run = |fold: bool| {
            let planned = plan(&rules, &d, kind, opts).expect("plans");
            let prepared = planned.stars.iter().map(|x| prepare(x, &st)).collect();
            let gens: Vec<Vec<u64>> = planned
                .stars
                .iter()
                .map(|x| x.sides.iter().map(|y| st.view(y.key.rel).1).collect())
                .collect();
            // Only the run half is timed; plan and preparation are equal for
            // both terminals.
            let t = Instant::now();
            let (o, rep) = execute_with(planned, prepared, Duration::ZERO, kind, &gens, fold)
                .expect("answers");
            (t.elapsed(), o, rep.terminal)
        };
        let (_, folded, ft) = run(true);
        let (_, enumerated, et) = run(false);
        assert_eq!(folded, enumerated, "{name}: folded answer differs");
        assert_eq!(ft, TerminalPath::Folded, "{name}");
        assert_eq!(et, TerminalPath::Enumerated, "{name}");
        let time = |fold: bool| {
            for _ in 0..WARMUP {
                black_box(run(fold));
            }
            median((0..REPS).map(|_| black_box(run(fold)).0).collect())
        };
        let (tf, te) = (time(true), time(false));
        let ratio = te.as_secs_f64() / tf.as_secs_f64();
        let verdict = if gated {
            if ratio >= 10.0 {
                "PASS (>= 10x)"
            } else {
                killed = true;
                "KILL (< 10x)"
            }
        } else {
            "reported"
        };
        let answer = match folded {
            Outcome::Count(n) => n,
            other => panic!("{other:?}"),
        };
        println!(
            "{name:<22} | {answer:>10} | {:>12.3} {:>12.3} | {ratio:>7.1}x | {verdict}",
            ms(tf),
            ms(te)
        );
    }
    assert!(!killed, "G11: the gated Count(Bag) ratio is below 10x");
}
