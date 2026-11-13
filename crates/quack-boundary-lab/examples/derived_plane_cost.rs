//! D-QUACK-DERIVED-PLANE-PROBE, P4: what the derived-plane checks cost on top
//! of the binding check they follow. Lab only.
//!
//! Every arm's answer is compared with a row loop before it is timed. The
//! warm arms must allocate nothing; the run aborts if one does.
//!
//! `cargo run --release -p quack-boundary-lab --example derived_plane_cost`

use lance_graph_mask_risc::exec::{scratch_words_for, tile_words_for};
use lance_graph_mask_risc::{Foreign, MaskOp, Operand, Out, Program, Scratch, Terminal, Value};
use quack_boundary_lab::derived::{admitted_planes, check_planes, DerivedPlane, Recipe};
use quack_boundary_lab::{unbound, Prepared, RowOrder, Source, World, READER, REGION, ROWS};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
// SAFETY: forwards to the system allocator unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn counted<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    let (a, b) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    let v = f();
    (v, ALLOCS.load(Relaxed) - a, BYTES.load(Relaxed) - b)
}

/// Median of 21 samples of `k` calls, in nanoseconds per call. Aborts if a
/// warm call allocates.
fn warm<T>(arm: &str, k: u32, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let (_, allocs, bytes) = counted(|| black_box(f()));
    assert_eq!((allocs, bytes), (0, 0), "{arm} allocates when warm");
    let mut samples: Vec<f64> = (0..21)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..k {
                black_box(f());
            }
            t.elapsed().as_nanos() as f64 / f64::from(k)
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{arm}\t{:.1}\t0", samples[10]);
    samples[10]
}

fn main() {
    println!(
        "# rows={ROWS} avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    let w = World::new();
    let s = Source::new(RowOrder::ById);
    let b = w
        .bind_fields(&s, &[REGION], READER, s.space())
        .expect("region binds");

    // Cold: three bit-planes of one bound lane.
    let t = Instant::now();
    let (planes, allocs, bytes) =
        counted(|| [2u8, 1, 0].map(|k| DerivedPlane::bit(&w, &w.view, &b, &s, REGION, k).unwrap()));
    let cold = t.elapsed();
    let held: usize = planes.iter().map(DerivedPlane::bytes).sum();
    println!(
        "# cold: 3 planes in {:.1} us, {allocs} allocations, {bytes} B allocated, {held} B held, lane read {} B",
        cold.as_secs_f64() * 1e6,
        ROWS * 4
    );

    let expects = [2u8, 1, 0].map(|k| Recipe::Bit { field: REGION, k });
    let refs = [&planes[0], &planes[1], &planes[2]];
    let subset = 0b0010_0110u8;
    let program = Program::new(
        vec![MaskOp::Ternlog {
            imm: subset,
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            c: Operand::Plane(2),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let want = (0..ROWS)
        .filter(|&r| (subset >> s.row(r).region) & 1 == 1)
        .count();
    let scratch_bytes =
        scratch_words_for(tile_words_for(ROWS), program.scratch_slots as usize).unwrap_or(0) * 8;
    println!("# scratch {scratch_bytes} B; bytes written per warm query: 0");
    let mut scratch = Scratch::for_program(&program, ROWS).unwrap();
    let p = Prepared::over(program.clone(), &s, &b);
    assert_eq!(
        admitted_planes(
            &p,
            &expects,
            &refs,
            &w,
            &w.view,
            &b,
            &s,
            &mut scratch,
            Out::None
        ),
        Ok(Value::Count(want))
    );

    println!("arm\tmedian_ns\twarm_allocs");
    let bind = warm("binding_check", 20_000, || {
        b.check(&w.store, &w.view, s.frame_store(), &s)
    });
    let plane = warm("plane_checks_3", 20_000, || {
        check_planes(&expects, &refs, &b, &s)
    });
    let masks: [&[u64]; 3] = [planes[0].words(), planes[1].words(), planes[2].words()];
    let lanes = s.raw_lanes();
    let exec = warm("execute_unchecked", 2_000, || {
        unbound(
            &program,
            &lanes,
            &masks,
            &Foreign::NONE,
            &mut scratch,
            Out::None,
        )
    });
    let all = warm("admitted_planes_total", 2_000, || {
        admitted_planes(
            &p,
            &expects,
            &refs,
            &w,
            &w.view,
            &b,
            &s,
            &mut scratch,
            Out::None,
        )
    });
    println!(
        "# plane checks / binding check {:.2}; checks / execution {:.3}; total - execution {:.1} ns",
        plane / bind,
        (bind + plane) / exec,
        all - exec
    );
}
