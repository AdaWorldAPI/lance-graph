//! Per-call timing of the four queries on the rev1 fixture, in nanoseconds.
//! Cold costs are reported in microseconds, not gated.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use lance_graph_codegraph::Codegraph;

const ITERS: u32 = 10_000;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/rev1")
}

fn per_call_ns(mut f: impl FnMut()) -> f64 {
    for _ in 0..ITERS {
        f();
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    t.elapsed().as_nanos() as f64 / f64::from(ITERS)
}

fn main() {
    let t = Instant::now();
    let mut g = Codegraph::open(&dir()).expect("rev1 opens");
    println!(
        "open: {} us (cold, reported, not gated)",
        t.elapsed().as_micros()
    );

    let double = g.pos("cg_lib::util::double").expect("double");
    let run = g.pos("cg_app::run").expect("run");

    let t = Instant::now();
    black_box(g.impact(double).expect("impact"));
    println!(
        "first closure build (impact): {} us (cold, reported, not gated)",
        t.elapsed().as_micros()
    );

    let ns = per_call_ns(|| {
        black_box(g.callers(double).unwrap());
    });
    println!("callers:    {ns:.1} ns/call");
    let ns = per_call_ns(|| {
        black_box(g.callees(run).unwrap());
    });
    println!("callees:    {ns:.1} ns/call");
    let ns = per_call_ns(|| {
        black_box(g.references(double));
    });
    println!("references: {ns:.1} ns/call");
    let ns = per_call_ns(|| {
        black_box(g.impact(double).unwrap());
    });
    println!("impact:     {ns:.1} ns/call");
}
