//! D-DRISA-1 bench: the same methods over the same canonical SoA, three ways.
//!
//! - A `direct`: planes assembled by the caller from the SoA and a selection
//!   mask it already holds; no register checks.
//! - B `bind each`: every query binds a register file (routes the selection's
//!   global ids through the DR maps, checks frame/source/layout/fields),
//!   compiles the method and runs it.
//! - C `bind once`: one register file, methods compiled once; each query
//!   revalidates (frame, scope, source, layout digests, value witness) and
//!   runs.
//!
//! Per query: wall time (µs), heap allocations and global → local lookups.
//! A counting allocator wraps the system one; it counts calls, not bytes.
//!
//! Run: cargo run --release --manifest-path crates/drisa-register-probe/Cargo.toml --example drisa_bench

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use drisa_register_probe::{
    build_soa, build_table, lookups, run_direct, Checks, Fixture, MethodOp, RegisterFile,
};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_dt64::dr::{Scope, Width};
use lance_graph_dt64::ValueDemand;
use lance_graph_mask_risc::Value;

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

const FIELDS: usize = 64;
const DEMANDED: [u8; 3] = [5, 17, 42];
const SCOPE: Scope = Scope { id: 7 };
const OPS: [MethodOp; 4] = [
    MethodOp::Count,
    MethodOp::Exists,
    MethodOp::CountWhereGt { field: 17, t: -5 },
    MethodOp::SumWhereGt {
        field: 17,
        t: -5,
        sum: 42,
    },
];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    println!("D-DRISA-1: same methods, same SoA. Per query (one of 4 methods, round robin), median of 9 trials.");
    println!(
        "{:<22} {:>4} {:>11} {:>11} {:>11} {:>9} {:>9} {:>9}",
        "fixture", "Q", "A direct", "B bind/q", "C bind 1x", "allocs B", "allocs C", "lkup B/q"
    );
    for (name, n, width) in [
        ("DR64, 1 tile", 64usize, Width::W64),
        ("DR256, 1 tile", 256, Width::W256),
        ("DR256, 256 tiles", 65_536, Width::W256),
    ] {
        let fixture = Fixture::new(n, FIELDS, 31);
        let (table, tiles) = build_table(&fixture, width, 1, 77);
        let soa = build_soa(&fixture, &table, &tiles, width);
        let selection: Vec<u64> = fixture.ids().into_iter().step_by(10).collect();
        let auth = WideFieldMask::from_positions(&DEMANDED);
        let bind = || {
            RegisterFile::bind(
                &table,
                &table.seal(),
                &tiles,
                width,
                SCOPE,
                FIELDS,
                &auth,
                &DEMANDED,
                ValueDemand::Values,
                &selection,
                &soa,
                Checks::ALL,
            )
            .unwrap()
        };
        let once = bind();
        let compiled: Vec<_> = OPS.iter().map(|&op| once.compile(op).unwrap()).collect();
        let mut scratch: Vec<_> = compiled.iter().map(|c| once.scratch(c).unwrap()).collect();
        let sel_words = once.selection().to_vec();
        // Parity once per fixture: all three arms give the same answers.
        for (i, c) in compiled.iter().enumerate() {
            let a = run_direct(c.program(), &soa, &sel_words, &DEMANDED, &mut scratch[i]).unwrap();
            let c_ = once.run(&table, &soa, SCOPE, c, &mut scratch[i]).unwrap();
            let fresh = bind();
            let cb = fresh.compile(OPS[i]).unwrap();
            let mut sb = fresh.scratch(&cb).unwrap();
            let b = fresh.run(&table, &soa, SCOPE, &cb, &mut sb).unwrap();
            assert!(a == c_ && b == c_, "{name} {:?}", OPS[i]);
            assert!(!matches!(c_, Value::Count(0)), "non-vacuous");
        }
        for q in [1usize, 2, 8, 16, 64, 256] {
            let (mut ta, mut tb, mut tc) = (vec![], vec![], vec![]);
            let (mut ab, mut ac, mut lb) = (0u64, 0u64, 0u64);
            for _ in 0..9 {
                let t = Instant::now();
                for k in 0..q {
                    let i = k % OPS.len();
                    black_box(
                        run_direct(
                            compiled[i].program(),
                            &soa,
                            &sel_words,
                            &DEMANDED,
                            &mut scratch[i],
                        )
                        .unwrap(),
                    );
                }
                ta.push(t.elapsed().as_nanos() as f64 / q as f64 / 1000.0);

                let l0 = lookups(&table, &tiles);
                let a0 = ALLOCS.load(Ordering::Relaxed);
                let t = Instant::now();
                for k in 0..q {
                    let i = k % OPS.len();
                    let f = bind();
                    let c = f.compile(OPS[i]).unwrap();
                    let mut s = f.scratch(&c).unwrap();
                    black_box(f.run(&table, &soa, SCOPE, &c, &mut s).unwrap());
                }
                let dt = t.elapsed().as_nanos() as f64 / q as f64 / 1000.0;
                ab = (ALLOCS.load(Ordering::Relaxed) - a0) / q as u64;
                tb.push(dt);
                lb = (lookups(&table, &tiles) - l0) / q as u64;

                let a0 = ALLOCS.load(Ordering::Relaxed);
                let t = Instant::now();
                for k in 0..q {
                    let i = k % OPS.len();
                    black_box(
                        once.run(&table, &soa, SCOPE, &compiled[i], &mut scratch[i])
                            .unwrap(),
                    );
                }
                let dt = t.elapsed().as_nanos() as f64 / q as f64 / 1000.0;
                ac = (ALLOCS.load(Ordering::Relaxed) - a0) / q as u64;
                tc.push(dt);
            }
            println!(
                "{name:<22} {q:>4} {:>9.3}µs {:>9.3}µs {:>9.3}µs {ab:>9} {ac:>9} {lb:>9}",
                median(ta),
                median(tb),
                median(tc)
            );
        }
        println!(
            "{name:<22} register file owns {} B (selection register); SoA holds {} B of values",
            once.owned_bytes(),
            soa.n_rows() * FIELDS * 4
        );
    }
}
