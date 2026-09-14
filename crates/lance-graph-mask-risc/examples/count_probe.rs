//! D-MRX-6 — the first 64k probe: `COUNT(alpha & ((A & B) | C))` over 65,536
//! rows, four arms that must agree bit-for-bit:
//!
//! * `reference`   — the row-at-a-time oracle (no SIMD facade);
//! * `handwritten` — the facade words called directly, three passes;
//! * `interpreted` — the same three ops as a `Program` through `execute`;
//! * `fused`       — the `BoolExpr` fused to ternlogs through `execute`;
//!
//! plus the F-X1 pair — a gated predicate against the two-op `pred + and`
//! spelling on a sparse gate. They must agree on the COUNT (gated); their
//! ns/exec is the only observable difference between the two routings, and
//! is printed, never asserted.
//!
//! The counting allocator must read ZERO bytes per `execute` after warm-up
//! (law L1), and the timings say what the interpreter costs over the hand
//! spelling. Timings are printed, never asserted — pin them from a run, not
//! from a guess. The gate compares COUNTS, not masks.
//!
//! ```text
//! cargo run --release -p lance-graph-mask-risc --example count_probe
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute, Scratch};
use lance_graph_mask_risc::fuse::{fuse_program, BoolExpr};
use lance_graph_mask_risc::reference::reference_execute;
use lance_graph_mask_risc::{LaneRef, MaskOp, Operand, Planes, Pred, Program, Terminal, Value};
use ndarray::simd::{mask_and, mask_or, popcount_batch_u64};

struct Counting;
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: a pure pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

const N: usize = 65_536;
const WORDS: usize = N / 64;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn plane(seed: &mut u64, ands: u32) -> Vec<u64> {
    (0..WORDS)
        .map(|_| {
            let mut w = lcg(seed);
            for _ in 0..ands {
                w &= lcg(seed);
            }
            w
        })
        .collect()
}

fn time<F: FnMut() -> usize>(mut f: F) -> (f64, usize, usize) {
    let mut reps = 1usize;
    loop {
        let before = BYTES.load(Ordering::Relaxed);
        let t = Instant::now();
        let mut last = 0;
        for _ in 0..reps {
            last = f();
        }
        let e = t.elapsed();
        let bytes = BYTES.load(Ordering::Relaxed) - before;
        if e.as_millis() >= 200 {
            return (e.as_nanos() as f64 / reps as f64, last, bytes / reps);
        }
        reps *= 2;
    }
}

fn main() {
    let mut seed = 0x5EED;
    let alpha = plane(&mut seed, 0);
    let a = plane(&mut seed, 1);
    let b = plane(&mut seed, 1);
    let c = plane(&mut seed, 2);
    // A SPARSE gate (≈1 word in 8 non-zero) and one value lane: the gated
    // predicate's whole claim is that it skips words the gate zeroes, and a
    // dense gate cannot show that.
    let sparse: Vec<u64> = (0..WORDS)
        .map(|i| if i % 8 == 0 { lcg(&mut seed) } else { 0 })
        .collect();
    let lane: Vec<i32> = (0..N)
        .map(|_| (lcg(&mut seed) % 2000) as i32 - 1000)
        .collect();
    let masks: [&[u64]; 5] = [&alpha, &a, &b, &c, &sparse];
    let lanes = [LaneRef::I32(&lane)];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &lanes,
    };
    let (pa, pb, pc, palpha) = (
        Operand::Plane(1),
        Operand::Plane(2),
        Operand::Plane(3),
        Operand::Plane(0),
    );

    let interpreted = Program::new(
        vec![
            MaskOp::And {
                a: pa,
                b: pb,
                dst: 0,
            },
            MaskOp::Or {
                a: Operand::Scratch(0),
                b: pc,
                dst: 0,
            },
            MaskOp::And {
                a: palpha,
                b: Operand::Scratch(0),
                dst: 0,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let expr = BoolExpr::And(
        Box::new(BoolExpr::Leaf(palpha)),
        Box::new(BoolExpr::Or(
            Box::new(BoolExpr::And(
                Box::new(BoolExpr::Leaf(pa)),
                Box::new(BoolExpr::Leaf(pb)),
            )),
            Box::new(BoolExpr::Leaf(pc)),
        )),
    );
    let fused = match fuse_program(&expr, 0, |m| Terminal::Count { mask: m }) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("fuse failed: {e:?}");
            std::process::exit(1);
        }
    };
    println!(
        "interpreted: {} mask passes; fused: {} mask passes ({} ternlog)",
        interpreted.op_histogram().mask_passes(),
        fused.op_histogram().mask_passes(),
        fused.op_histogram().ternlog
    );

    let reference = match reference_execute(&interpreted, &planes, None) {
        Ok(Value::Count(c)) => c,
        other => {
            eprintln!("reference failed: {other:?}");
            std::process::exit(1);
        }
    };

    let mut t0 = vec![0u64; WORDS];
    let mut t1 = vec![0u64; WORDS];
    let mut si = Scratch::for_program(&interpreted, N);
    let mut sf = Scratch::for_program(&fused, N);
    let count_of = |v: Result<Value, _>| match v {
        Ok(Value::Count(c)) => c,
        _ => usize::MAX,
    };
    // warm-up (first touch of the scratch pages is not the executor's cost)
    let _ = execute(&interpreted, &planes, &mut si, None);
    let _ = execute(&fused, &planes, &mut sf, None);

    let (ns_hand, hand, b_hand) = time(|| {
        mask_and(&a, &b, &mut t0);
        mask_or(&t0, &c, &mut t1);
        mask_and(&alpha, &t1, &mut t0);
        popcount_batch_u64(&t0) as usize
    });
    let (ns_int, int, b_int) = time(|| count_of(execute(&interpreted, &planes, &mut si, None)));
    let (ns_fused, fus, b_fus) = time(|| count_of(execute(&fused, &planes, &mut sf, None)));

    // ---- F-X1: the gated predicate's routing is a COST property ----
    //
    // `Pred { under: Some(g) }` must reach `*_to_mask_under` (one pass that
    // skips words where `g` is zero), NOT `*_to_mask` followed by `mask_and`
    // (two full passes). The two spellings are semantically identical, so no
    // differential can separate them — the only observable difference is
    // time on a sparse gate. Printed, never asserted: a timing assertion is
    // not a gate, and this is the honest instrument for a cost claim.
    let gate = Operand::Plane(4);
    let gated = Program::new(
        vec![MaskOp::Pred {
            pred: Pred::GtI32 { lane: 0, t: 0 },
            under: Some(gate),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let two_op = Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::GtI32 { lane: 0, t: 0 },
                under: None,
                dst: 0,
            },
            MaskOp::And {
                a: Operand::Scratch(0),
                b: gate,
                dst: 0,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let mut sg = Scratch::for_program(&gated, N);
    let mut st = Scratch::for_program(&two_op, N);
    let _ = execute(&gated, &planes, &mut sg, None);
    let _ = execute(&two_op, &planes, &mut st, None);
    let (ns_gated, c_gated, b_gated) = time(|| count_of(execute(&gated, &planes, &mut sg, None)));
    let (ns_two, c_two, b_two) = time(|| count_of(execute(&two_op, &planes, &mut st, None)));

    println!("arm          count    ns/exec   heap B/exec");
    println!("reference    {reference:>6}");
    println!("handwritten  {hand:>6}  {ns_hand:>9.0}  {b_hand:>6}");
    println!("interpreted  {int:>6}  {ns_int:>9.0}  {b_int:>6}");
    println!("fused        {fus:>6}  {ns_fused:>9.0}  {b_fus:>6}");
    println!("-- F-X1, sparse gate (1 word in 8) — same answer, different cost --");
    println!("gated        {c_gated:>6}  {ns_gated:>9.0}  {b_gated:>6}");
    println!("pred+and     {c_two:>6}  {ns_two:>9.0}  {b_two:>6}");

    let ok = hand == reference
        && int == reference
        && fus == reference
        && b_int == 0
        && b_fus == 0
        && c_gated == c_two
        && b_gated == 0
        && b_two == 0;
    println!("gate: {}", if ok { "ok" } else { "FAILED" });
    if !ok {
        std::process::exit(1);
    }
}
