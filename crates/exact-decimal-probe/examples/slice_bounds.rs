//! D-AED-1 measurement: per-slice N, S, Q by a slice-id lane (one grouped
//! pass) against sealed-slice boundaries (one extent pass per slice).
//! Run: `cargo run --release --example slice_bounds`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use exact_decimal_probe::{slice_counts_sums, slice_sums, slice_sums_by_bounds};
use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{
    words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Terminal,
};

/// The grouped plain sum under one key: what grouping alone costs, with no
/// square, to tell it from the Σx² fold.
fn group_sum_one_key(coef: &[i32], plane: &[u64], zeros: &[u32], bounds: &[usize]) -> Vec<i64> {
    let ops = vec![
        MaskOp::Pred {
            pred: Pred::NeI32 { lane: 0, v: 0 },
            under: None,
            dst: 0,
        },
        MaskOp::Or {
            a: Operand::Scratch(0),
            b: Operand::Plane(0),
            dst: 0,
        },
    ];
    let program = Program::new(
        ops,
        Terminal::GroupSumI32 {
            mask: Operand::Scratch(0),
            key: 1,
            val: 0,
        },
    );
    let masks = [plane];
    let lanes = [LaneRef::I32(coef), LaneRef::U32(zeros)];
    let planes = Planes {
        n_rows: coef.len(),
        masks: &masks,
        lanes: &lanes,
    };
    let mut scratch = Scratch::for_program(&program, coef.len()).unwrap();
    let mut out = vec![0i64; bounds.len() - 1];
    for (k, w) in bounds.windows(2).enumerate() {
        execute_extent(
            &program,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::I64(std::slice::from_mut(&mut out[k])),
            w[0]..w[1],
        )
        .unwrap();
    }
    out
}

struct Counting;
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

/// Best of `reps` runs: ns per 64 lanes, and bytes allocated by one run.
fn measure(n: usize, reps: usize, mut f: impl FnMut()) -> (f64, usize) {
    let before = BYTES.load(Ordering::Relaxed);
    f();
    let bytes = BYTES.load(Ordering::Relaxed) - before;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_nanos() as f64);
    }
    (best / (n as f64 / 64.0), bytes)
}

fn main() {
    let n = 1 << 20;
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut coef = Vec::with_capacity(n);
    let mut plane = vec![0u64; words_for(n)];
    for i in 0..n {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let r = (x >> 33) as i64;
        if r % 5 != 0 {
            coef.push((r % 200_001 - 100_000) as i32);
            plane[i / 64] |= 1 << (i % 64);
        } else {
            coef.push(0);
        }
    }
    let zeros = vec![0u32; n];
    println!("rows {n}; ns per 64 lanes (best of 50); bytes allocated per call");
    for slices in [1usize, 16, 256] {
        // Unaligned boundaries: every slice is cut 7 rows past a word.
        let mut bounds: Vec<usize> = (0..slices).map(|k| (k * n / slices + 7).min(n)).collect();
        bounds[0] = 0;
        bounds.push(n);
        let mut slice_of = Vec::with_capacity(n);
        for (k, w) in bounds.windows(2).enumerate() {
            slice_of.extend(std::iter::repeat_n(k as u32, w[1] - w[0]));
        }
        let want = slice_sums(&coef, &plane, &slice_of, slices).unwrap();
        assert_eq!(
            slice_sums_by_bounds(&coef, &plane, &bounds, &zeros).unwrap(),
            want
        );
        let sums: Vec<i64> = want.iter().map(|s| s.sum).collect();
        assert_eq!(group_sum_one_key(&coef, &plane, &zeros, &bounds), sums);

        let rows = [
            (
                "grouped, slice-id lane built per call",
                measure(n, 50, || {
                    let mut ids = Vec::with_capacity(n);
                    for (k, w) in bounds.windows(2).enumerate() {
                        ids.extend(std::iter::repeat_n(k as u32, w[1] - w[0]));
                    }
                    black_box(slice_sums(&coef, &plane, &ids, slices).unwrap());
                }),
            ),
            (
                "grouped, slice-id lane held",
                measure(n, 50, || {
                    black_box(slice_sums(&coef, &plane, &slice_of, slices).unwrap());
                }),
            ),
            (
                "bounds, N S Q (zero lane held)",
                measure(n, 50, || {
                    black_box(slice_sums_by_bounds(&coef, &plane, &bounds, &zeros).unwrap());
                }),
            ),
            (
                "bounds, grouped S only (zero lane held)",
                measure(n, 50, || {
                    black_box(group_sum_one_key(&coef, &plane, &zeros, &bounds));
                }),
            ),
            (
                "bounds, N S only (no lane)",
                measure(n, 50, || {
                    black_box(slice_counts_sums(&coef, &plane, &bounds).unwrap());
                }),
            ),
        ];
        for (name, (ns, bytes)) in rows {
            println!("slices {slices:>3}  {name:<40} {ns:>7.2} ns/64  {bytes:>9} B");
        }
    }
}
