//! D-DYNBAG-1 cost probe: ℕ-compose (`matmul_i8_to_i32` over DR64² tiles)
//! against the Boolean `relcomp::compose` on the DR64 register, same data.
//! Hot paths only, reported in ns per call (median of 201 runs).
//!
//!   cargo run --release --example ncompose_cost --manifest-path crates/xlog-relcomp-probe/Cargo.toml
use std::hint::black_box;
use std::time::Instant;

use dr_math_binding_probe::relcomp::{Work, compose};
use dr_math_binding_probe::{Axis, AxisKind};
use lance_graph_dt64::dr::{LocalMap, Width};
use ndarray::Array2;
use ndarray::simd::{amx_available, matmul_i8_to_i32};
use xlog_relcomp_probe::ncompose::{PairQuery, pair_counts};
use xlog_relcomp_probe::{AddressMap, register};

const AX: [Axis; 3] = [
    Axis {
        class: 10,
        kind: AxisKind::Identity,
    },
    Axis {
        class: 11,
        kind: AxisKind::Property,
    },
    Axis {
        class: 12,
        kind: AxisKind::Community,
    },
];

type Tuples = Vec<(u32, u32)>;

fn median_ns(mut f: impl FnMut()) -> u128 {
    let mut t: Vec<u128> = (0..201)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_nanos()
        })
        .collect();
    t.sort_unstable();
    t[100]
}

fn local(ids: impl Iterator<Item = u32>) -> LocalMap {
    let mut v: Vec<u64> = ids.map(u64::from).collect();
    v.sort_unstable();
    v.dedup();
    LocalMap::prepare(Width::W64, &v).unwrap()
}

fn main() {
    println!("amx_available: {}", amx_available());
    let full: Vec<(u32, u32)> = (0..64).flat_map(|a| (0..64).map(move |b| (a, b))).collect();
    let cases: [(&str, Tuples, Tuples); 2] = [
        (
            "sparse",
            vec![(1, 2), (5, 2), (1, 3), (9, 7)],
            vec![(2, 4), (2, 6), (3, 4), (7, 8)],
        ),
        ("dense 64x64", full.clone(), full),
    ];
    let q = PairQuery {
        left: xlog_core::RelId(0),
        right: xlog_core::RelId(1),
    };
    for (name, r, s) in cases {
        // Boolean: the register composition.
        let xs = AddressMap::new(AX[0], r.iter().map(|t| t.0)).unwrap();
        let ys =
            AddressMap::new(AX[1], r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))).unwrap();
        let zs = AddressMap::new(AX[2], s.iter().map(|t| t.1)).unwrap();
        let (rr, ss) = (
            register(&r, &xs, &ys).unwrap(),
            register(&s, &ys, &zs).unwrap(),
        );
        let mut work = Work::default();
        let b = median_ns(|| {
            black_box(compose(black_box(&rr), black_box(&ss), &mut work));
        });
        // ℕ: the product alone, on prepared i8 tiles, output reused.
        let (lx, ly, lz) = (
            local(r.iter().map(|t| t.0)),
            local(r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))),
            local(s.iter().map(|t| t.1)),
        );
        let mut a = Array2::<i8>::zeros((64, 64));
        let mut bm = Array2::<i8>::zeros((64, 64));
        for &(x, y) in &r {
            a[(
                lx.local(x.into()).unwrap() as usize,
                ly.local(y.into()).unwrap() as usize,
            )] = 1;
        }
        for &(y, z) in &s {
            bm[(
                ly.local(y.into()).unwrap() as usize,
                lz.local(z.into()).unwrap() as usize,
            )] = 1;
        }
        let mut out = Array2::<i32>::zeros((64, 64));
        let n = median_ns(|| {
            matmul_i8_to_i32(black_box(a.view()), black_box(bm.view()), out.view_mut()).unwrap();
        });
        // ℕ end to end: tiles built from the tuples, then the product.
        let e = median_ns(|| {
            black_box(pair_counts(&q, &r, &s, &lx, &ly, &lz).unwrap());
        });
        println!(
            "{name}: compose (B) {b} ns | matmul (N, tiles prepared) {n} ns | pair_counts (N, from tuples) {e} ns"
        );
    }
}
