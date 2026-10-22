//! In-fold cost of E1 (encode, and lowering + execution on R2IL), release
//! build, median of 5.
//!
//! The egglog phase is timed in `Compiled::timings.egglog` but NOT printed:
//! it is a cold, out-of-fold compile whose cost is above the workspace's
//! microsecond budget, and whether cold-compile costs may be reported at all
//! is an open operator decision (lance-graph `CLAUDE.md`, "no millisecond
//! results"). Until it is ruled, this example does not publish that number.
//!
//!   cargo run --release --example e1_cold_cost
//!
//! Agreement with the oracle is asserted FIRST; a number for a wrong answer is
//! not worth printing.

use std::time::{Duration, Instant};

use egglog_r2il_probe::{Arith, compile, e1_problem, lower_and_run, oracle};

const RUNS: usize = 5;
const C: i64 = 123_456;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn micros(d: Duration) -> u128 {
    d.as_micros()
}

fn main() {
    let p = e1_problem();

    // Agreement first.
    let probe = compile(&p, Arith::Exact, 8);
    assert!(probe.error.is_none(), "egglog error: {:?}", probe.error);
    let got = lower_and_run(&probe, &[("c", C)]).expect("lowers and runs");
    let want = oracle::solve(&p, C).expect("oracle answers");
    assert_eq!(
        i128::from(got),
        want,
        "lowered value disagrees with the oracle"
    );

    let mut encode = Vec::with_capacity(RUNS);
    let mut lower = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let c = compile(&p, Arith::Exact, 8);
        assert!(c.error.is_none(), "egglog error: {:?}", c.error);
        encode.push(c.timings.encode);
        let t = Instant::now();
        let v = lower_and_run(&c, &[("c", C)]).expect("lowers and runs");
        lower.push(t.elapsed());
        assert_eq!(i128::from(v), want);
    }

    println!("in-fold costs; reported, not gates (egglog phase withheld, see module doc)");
    println!("E1 under Exact, fuel 8, c = {C}, median of {RUNS}, microseconds:");
    println!(
        "  encode                              {:>10}",
        micros(median(encode))
    );
    println!(
        "  lower + execute                     {:>10}",
        micros(median(lower))
    );
    println!("  extracted: {}", probe.extracted);
}
