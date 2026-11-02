//! D-DRCA-1 stage costs: preparation, coordinate mapping, rotation,
//! projection, folds, cross-table composition and allocations, each measured
//! on its own over a dense 64 × 64 register.
//!
//! `cargo run --release --example hhtl16_stages`
//!
//! Reports ns per operation (median of 21 timed runs). Preparation is the
//! cold, once-per-frame global → local step; every other row is a warm read
//! over borrowed rows and must not touch the global map.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::time::Instant;

use dr_math_binding_probe::hhtl16::{
    LocalMap, address, cell, cell_rotated, column, cooccurrence, cooccurrence_count, coordinates,
    rotate_yz,
};
use dr_math_binding_probe::relcomp::{Chain, RelExpr, Rows};
use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Scope, bind_dr64,
};

struct Counting;
thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}
fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}
// SAFETY: forwards to `System` unchanged; only counts calls.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

const RUNS: usize = 21;

/// Median ns per op of `f`, which performs `ops` operations per call, and the
/// allocations of one call.
fn measure(name: &str, ops: u64, mut f: impl FnMut() -> u64) {
    let a0 = allocs();
    black_box(f());
    let per_call_allocs = allocs() - a0;
    let mut t: Vec<f64> = (0..RUNS)
        .map(|_| {
            let s = Instant::now();
            black_box(f());
            s.elapsed().as_nanos() as f64 / ops as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    println!(
        "{name:<34} {ops:>6} ops  {:>9.3} ns/op  {per_call_allocs:>3} allocs/call",
        t[RUNS / 2]
    );
}

fn fill(seed: u64) -> Rows {
    let mut x = seed | 1;
    let mut r = [0u64; 64];
    for w in r.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *w = x;
    }
    r
}

fn main() {
    let globals: Vec<u64> = (0..64u64)
        .map(|i| 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i + 1))
        .collect();
    let props: Vec<u64> = (0..64u64)
        .map(|i| 0xC2B2_AE3D_27D4_EB4Fu64.wrapping_mul(i + 7))
        .collect();

    measure("prepare: LocalMap (64 ids)", 1, || {
        LocalMap::prepare(&globals).unwrap().len() as u64
    });
    let emp = LocalMap::prepare(&globals).unwrap();
    let prop = LocalMap::prepare(&props).unwrap();
    measure("prepare: global->local lookup", 64, || {
        globals
            .iter()
            .map(|&g| u64::from(emp.local(g).unwrap()))
            .sum()
    });

    let r = fill(0x51);
    let fk = fill(0x77);
    let st = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let (focus, scope) = (Focus { revision: 1 }, Scope { id: 1 });
    let ax = |class, kind| Axis { class, kind };
    let ip = MethodSignature::presence(1, 1, ax(1, AxisKind::Identity), ax(1, AxisKind::Property));
    let fk_sig =
        MethodSignature::presence(1, 2, ax(1, AxisKind::Identity), ax(1, AxisKind::Identity));
    let bind = |m: MethodSignature, rows| -> BoundDr64<'_> {
        let at = Guards {
            frame: st.seal(),
            focus,
            scope,
        };
        bind_dr64(
            &st,
            m,
            m,
            &at,
            (focus, scope),
            Operand {
                rows,
                validity: None,
            },
        )
        .unwrap()
    };
    measure("bind: bind_dr64 (frame+focus+scope)", 1, || {
        bind(ip, &r).rows[0]
    });
    let b = bind(ip, &r);

    let lookups_before = emp.lookups() + prop.lookups();

    measure("map: address+coordinates (4096)", 4096, || {
        let mut acc = 0u64;
        for x in 0..64u8 {
            for y in 0..64u8 {
                let (cx, cy) = coordinates(address(x, y).unwrap()).unwrap();
                acc += u64::from(cx ^ cy);
            }
        }
        acc
    });
    measure("read: cell, stored orientation", 4096, || {
        let mut acc = 0u64;
        for x in 0..64u8 {
            for y in 0..64u8 {
                acc += u64::from(cell(b.rows, address(x, y).unwrap()).unwrap());
            }
        }
        acc
    });
    measure("read: cell through YZ rotation", 4096, || {
        let mut acc = 0u64;
        for x in 0..64u8 {
            for y in 0..64u8 {
                acc += u64::from(cell_rotated(b.rows, rotate_yz(address(x, y).unwrap())).unwrap());
            }
        }
        acc
    });
    measure("project: identity row (64)", 64, || {
        (0..64u8)
            .map(|x| b.rows[usize::from(x)])
            .fold(0, |a, w| a ^ w)
    });
    measure("project: property column (64)", 64, || {
        (0..64u8)
            .map(|y| column(b.rows, y).unwrap())
            .fold(0, |a, w| a ^ w)
    });
    measure("fold: P x P Boolean (R^T o R)", 1, || {
        cooccurrence(&b).unwrap()[0]
    });
    measure("fold: P x P count, one pair", 1, || {
        u64::from(cooccurrence_count(b.rows, 3, 5).unwrap())
    });
    let b_fk = bind(fk_sig, &fk);
    let b_ip = bind(ip, &r);
    let chain = Chain::new(&st, vec![b_fk, b_ip]).unwrap();
    let order = RelExpr::comp(RelExpr::Leaf(0), RelExpr::Leaf(1));
    measure("fold: cross-table FK o R (Chain)", 1, || {
        chain.eval(&order).unwrap().0[0]
    });

    println!(
        "global->local lookups during warm reads: {}",
        emp.lookups() + prop.lookups() - lookups_before
    );
    println!(
        "rows set in the dense register: {}",
        r.iter().map(|w| w.count_ones()).sum::<u32>()
    );
}
