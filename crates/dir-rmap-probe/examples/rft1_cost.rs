//! D-RFT-1 follow-up: where arm A's time goes (measurement only).
//!
//! `rft1_bench` reports three different comparisons, which this file keeps
//! apart: arm A (`Composer::compose`: r2il procedures replayed through
//! mask-risc, support first), arm C (direct `ndarray::simd` over every
//! coordinate) and arm C+ (arm A's support-first plan written directly with
//! `ndarray::simd`). Here arm A is decomposed:
//!
//! 1. preparation (`Composer::new`) against execution;
//! 2. the number of `execute_into` invocations per pass, by kind;
//! 3. the per-invocation cost of each prepared program on its own planes;
//! 4. the four-word mask cost: one prepared `count_and(3)` invocation against
//!    the same AND, AND, popcount written with `ndarray::simd`;
//! 5. support-first pruning: coordinates evaluated against all coordinates;
//! 6. allocations per pass, and those of the per-page `plane_count`, which
//!    compiles a program inside `compose`;
//! 7. how much of arm A a batched multi-coordinate fold could remove, as an
//!    upper bound from (2) and (3). No such terminal exists; this is not a
//!    design, and adding one needs its own admission decision.
//!
//! Every timing is the median of 101 runs, in ns; per-invocation costs are a
//! median over batches of 1000 invocations. Nothing here changes `compose`.
//!
//! ```text
//! cargo run --release --example rft1_cost
//! ```

use dir_rmap_probe::World;
use dir_rmap_probe::compose::{Carriers, Composer, Law, Members, add_bag};
use dir_rmap_probe::fixture::{Fixture, Spec};
use dir_rmap_probe::focus::SLOTS;
use dir_rmap_probe::prep::{Prepared, Refusal, ones};
use dir_rmap_probe::qr::{Page, Qr};
use lance_graph_mask_risc::{
    Foreign, Out, Planes, Program, Scratch, Value, execute_into, words_for,
};
use lance_graph_r2il::{FoldDialect, Val, procs};
use ndarray::simd::{mask_and_assign, mask_or_assign, popcount_batch_u64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards unchanged to `System`; the counter is an
// atomic and does not affect the allocation.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from `System.alloc` with `l`.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs_of<T>(mut f: impl FnMut() -> T) -> usize {
    let a0 = ALLOCS.load(Ordering::Relaxed);
    black_box(f());
    ALLOCS.load(Ordering::Relaxed) - a0
}

fn median_ns<T>(reps: usize, mut f: impl FnMut() -> T) -> u128 {
    let mut t = Vec::with_capacity(reps);
    for _ in 0..reps {
        let s = Instant::now();
        black_box(f());
        t.push(s.elapsed().as_nanos());
    }
    t.sort_unstable();
    t[t.len() / 2]
}

/// Median ns per call of `f`, over batches of 1000 calls.
fn per_call_ns(mut f: impl FnMut()) -> f64 {
    const BATCH: u32 = 1000;
    median_ns(101, || {
        for _ in 0..BATCH {
            f();
        }
    }) as f64
        / f64::from(BATCH)
}

fn build(spec: Spec) -> Result<(World, Prepared, Qr), Refusal> {
    let w = World::new(Fixture::generate(spec));
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness())?;
    Ok((w, b, qr))
}

/// A prepared program over `k` planes of `n_rows`, built the way
/// `compose.rs`'s `Replay::compile` builds it.
fn prepare(
    k: usize,
    n_rows: usize,
    keep: bool,
    body: impl FnOnce(&mut FoldDialect<'_>, &mut Vec<Val>, &[u16]),
) -> (Program, Scratch<'static>) {
    let zero = vec![0u64; words_for(n_rows)];
    let mut sink = vec![0u64; words_for(n_rows)];
    let masks = vec![zero.as_slice(); k];
    let planes = Planes {
        n_rows,
        masks: &masks,
        lanes: &[],
    };
    let ix: Vec<u16> = (0..k).map(|p| p as u16).collect();
    let mut d = if keep {
        FoldDialect::with_keep(&planes, &Foreign::NONE, &mut sink)
    } else {
        FoldDialect::new(&planes, &Foreign::NONE)
    };
    let mut stack = Vec::new();
    body(&mut d, &mut stack, &ix);
    let program = d.last_program().expect("folds").clone();
    let scratch = Scratch::for_program(&program, n_rows).expect("scratch");
    (program, scratch)
}

fn count_and(k: usize, n_rows: usize) -> (Program, Scratch<'static>) {
    prepare(k, n_rows, false, |d, s, ix| {
        procs::count_and(d, s, ix).expect("compiles");
        s.pop();
    })
}

fn run(p: &Program, s: &mut Scratch<'static>, masks: &[&[u64]], n_rows: usize, out: Out<'_>) {
    let planes = Planes {
        n_rows,
        masks,
        lanes: &[],
    };
    match execute_into(p, &planes, &Foreign::NONE, s, out) {
        Ok(Value::Count(_) | Value::Mask(_)) => {}
        other => panic!("unexpected {other:?}"),
    }
}

fn main() {
    let (w, b, qr) = match build(Spec::iam(8000)) {
        Ok(x) => {
            println!("fixture: Spec::iam(8000)");
            x
        }
        Err(Refusal::TooLarge) => {
            println!("fixture: Spec::iam(8000) exceeds 256 slots; Spec::small()");
            build(Spec::small()).expect("Spec::small fits 256 slots")
        }
        Err(e) => panic!("bind refused: {e:?}"),
    };
    let users = b.users;
    let pages: Vec<Page> = (0..qr.pages())
        .map(|p| qr.page(&b, p).expect("page"))
        .collect();
    let counts: Vec<usize> = (0..pages.len())
        .map(|p| users.saturating_sub(p * SLOTS).min(SLOTS))
        .collect();
    let members: Vec<Members<'_>> = pages
        .iter()
        .zip(&counts)
        .map(|(pg, &n)| Members::of(&qr, pg, n))
        .collect();
    let carriers = Carriers::of(&qr);
    let current = w.witness();

    // Arm A, one prepared composer.
    let mut composer = Composer::new();
    let mut arm_a = || {
        let (mut bag, mut distinct) = (0u64, 0u64);
        for m in &members {
            let out = composer
                .compose(m, &carriers, &current, Law::Count)
                .expect("compose");
            bag = add_bag(bag, out.bag).expect("bag");
            distinct = add_bag(distinct, out.distinct).expect("distinct");
        }
        (bag, distinct)
    };

    // (2), (5): the invocation census, from the same support-first plan in
    // direct SIMD (which arm C+ shows is answer-equal to arm A).
    let (mut present_calls, mut union_calls, mut path_calls) = (0usize, 0usize, 0usize);
    let (mut bag, mut distinct) = (0u64, 0u64);
    let mut key_counts: Vec<usize> = Vec::new();
    for (m, &n) in members.iter().zip(&counts) {
        for i in 0..n {
            present_calls += 1;
            let mut present = m.member[i];
            mask_and_assign(&mut present, &m.valid[i]);
            let keys: Vec<usize> = ones(&present).collect();
            if keys.is_empty() {
                continue;
            }
            union_calls += 1;
            key_counts.push(keys.len());
            let mut support = [0u64];
            for &s in &keys {
                mask_or_assign(&mut support, std::slice::from_ref(&qr.role[s]));
            }
            distinct += popcount_batch_u64(&support);
            for r in ones(&support) {
                path_calls += 1;
                let mut buf = present;
                mask_and_assign(&mut buf, &carriers.role_slots[r]);
                bag += popcount_batch_u64(&buf);
            }
        }
    }
    assert_eq!(arm_a(), (bag, distinct), "census plan vs arm A");
    let fan_calls = 64 * pages.len();
    let plane_count_calls = pages.len();
    let calls = present_calls + union_calls + path_calls + fan_calls + plane_count_calls;
    let all_coords = users * 64;
    println!(
        "users {users}, pages {}, bag {bag}, distinct {distinct}",
        pages.len()
    );
    println!(
        "(2) execute_into per pass: {calls} = present {present_calls} + union {union_calls} \
         + path {path_calls} + fan {fan_calls} + plane_count {plane_count_calls}"
    );
    println!(
        "(5) support first: {path_calls} path folds of {all_coords} coordinates ({:.1}%)",
        100.0 * path_calls as f64 / all_coords as f64
    );

    // (1) preparation vs execution.
    let prep_ns = median_ns(101, Composer::new);
    let exec_ns = median_ns(101, &mut arm_a);
    println!("(1) Composer::new {prep_ns} ns; arm A execution {exec_ns} ns per pass");

    // (3) per-invocation cost of each prepared program, on real-shaped planes.
    let m0 = &members[0];
    let (i0, r0) = (0usize, 0usize);
    let (pa, mut sa) = count_and(3, SLOTS);
    let path_ns = per_call_ns(|| {
        run(
            &pa,
            &mut sa,
            &[&m0.member[i0], &m0.valid[i0], &carriers.role_slots[r0]],
            SLOTS,
            Out::None,
        )
    });
    let (pi, mut si) = prepare(2, SLOTS, true, |d, s, ix| {
        procs::intersect(d, s, ix).expect("compiles");
        procs::keep(d, s).expect("compiles");
    });
    let mut present = [0u64; 4];
    let present_ns = per_call_ns(|| {
        run(
            &pi,
            &mut si,
            &[&m0.member[i0], &m0.valid[i0]],
            SLOTS,
            Out::Mask(&mut present),
        )
    });
    key_counts.sort_unstable();
    let k = key_counts.get(key_counts.len() / 2).copied().unwrap_or(1);
    let (pu, mut su) = prepare(k, 64, true, |d, s, ix| {
        procs::union(d, s, ix).expect("compiles");
        procs::keep(d, s).expect("compiles");
    });
    let words: Vec<&[u64]> = (0..k).map(|s| std::slice::from_ref(&qr.role[s])).collect();
    let mut support = [0u64];
    let union_ns = per_call_ns(|| run(&pu, &mut su, &words, 64, Out::Mask(&mut support)));
    let (pf, mut sf) = count_and(1, SLOTS);
    let fan_ns = per_call_ns(|| run(&pf, &mut sf, &[&carriers.role_slots[r0]], SLOTS, Out::None));
    let plane_count_compile_ns =
        median_ns(101, || count_and(1, (SLOTS * 64).min(users.max(1) * 64)));
    println!(
        "(3) ns per invocation: path count_and(3) {path_ns:.1}, present keep(intersect 2) \
         {present_ns:.1}, support keep(union {k}, median keys) {union_ns:.1}, fan count_and(1) \
         {fan_ns:.1}; plane_count compile+scratch {plane_count_compile_ns} ns per page"
    );
    let predicted = path_ns * path_calls as f64
        + present_ns * present_calls as f64
        + union_ns * union_calls as f64
        + fan_ns * fan_calls as f64
        + plane_count_compile_ns as f64 * plane_count_calls as f64;
    println!(
        "    predicted from (2)x(3): {predicted:.0} ns vs measured {exec_ns} ns ({:.0}%); \
         path folds {:.0}%, plane_count compile {:.0}% of the prediction",
        100.0 * predicted / exec_ns as f64,
        100.0 * path_ns * path_calls as f64 / predicted,
        100.0 * plane_count_compile_ns as f64 * plane_count_calls as f64 / predicted
    );

    // (4) the four-word cost: the same AND, AND, popcount in direct SIMD.
    let simd_ns = per_call_ns(|| {
        let mut buf = m0.member[i0];
        mask_and_assign(&mut buf, &m0.valid[i0]);
        mask_and_assign(&mut buf, &carriers.role_slots[r0]);
        black_box(popcount_batch_u64(&buf));
    });
    println!(
        "(4) four-word count_and(3): mask-risc {path_ns:.1} ns vs ndarray::simd {simd_ns:.1} ns; \
         fixed per-invocation overhead about {:.1} ns",
        path_ns - simd_ns
    );

    // (6) allocations.
    let pass_allocs = allocs_of(&mut arm_a);
    let exec_allocs = allocs_of(|| {
        run(
            &pa,
            &mut sa,
            &[&m0.member[i0], &m0.valid[i0], &carriers.role_slots[r0]],
            SLOTS,
            Out::None,
        )
    });
    let compile_allocs = allocs_of(|| count_and(1, (SLOTS * 64).min(users.max(1) * 64)));
    println!(
        "(6) allocations: {pass_allocs} per pass; one prepared execute_into {exec_allocs}; \
         one plane_count compile {compile_allocs} (x{plane_count_calls} pages = {})",
        compile_allocs * plane_count_calls
    );

    // (7) upper bound for a batched multi-coordinate fold: it would replace
    // the path invocations' fixed overhead, not their four-word work.
    let removable = (path_ns - simd_ns).max(0.0) * path_calls as f64;
    println!(
        "(7) a batched fold over the path coordinates could remove at most about {removable:.0} ns \
         ({:.0}% of arm A); NOT built, needs its own admission decision",
        100.0 * removable / exec_ns as f64
    );
}
