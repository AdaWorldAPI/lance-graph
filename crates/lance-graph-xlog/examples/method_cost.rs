//! Cost of a registered method over a cube relation's lent lanes:
//! `cargo run --release -p lance-graph-xlog --example method_cost`.
//! `prepare` is once per binding, `run` once per question. Every count is
//! checked against an edge walk that shares no code with the cube.

#[path = "../tests/support/method.rs"]
#[allow(dead_code)]
mod support;

use std::time::Instant;

use lance_graph_dt64::MethodId;
use lance_graph_xlog::cube::{Cube, CubeLanes};
use support::*;

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let mut s = 0x7EA1_u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let ids: Vec<u32> = (0..256).collect();
    let t = table(&ids);
    let (src, dst) = (17u32, 200u32);
    let w = world(vec![method(1, &t, src, dst)]);
    let r = w.store.method_ref(CLASS, MethodId(1)).unwrap();
    println!("rows  count  prepare_ns  run_ns  run_ns_per_row");
    for rows in [1024usize, 16384, 262144] {
        let e: Vec<(u32, u32)> = (0..rows)
            .map(|_| ((next() % 256) as u32, (next() % 256) as u32))
            .collect();
        let cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
        let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
        let want = oracle(&e, src, dst);

        let mut prepare = Vec::new();
        for _ in 0..31 {
            let t0 = Instant::now();
            let p = std::hint::black_box(Prepared::prepare(&w, &source, &t, &r).unwrap());
            prepare.push(t0.elapsed().as_nanos());
            drop(p);
        }
        let mut p = Prepared::prepare(&w, &source, &t, &r).unwrap();
        let mut run = Vec::new();
        for _ in 0..201 {
            let t0 = Instant::now();
            let got = std::hint::black_box(p.run());
            run.push(t0.elapsed().as_nanos());
            assert_eq!(got, want, "rows {rows}");
        }
        let run_ns = median(run);
        println!(
            "{rows}  {want}  {}  {run_ns}  {:.3}",
            median(prepare),
            run_ns as f64 / rows as f64
        );
    }
}
