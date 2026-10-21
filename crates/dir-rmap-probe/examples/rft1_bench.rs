//! D-RFT-1 benchmark: the admitted query (every user's role multiplicities
//! through direct security groups and role carriers) three ways, over all
//! pages of one fixture.
//!
//! * A: `Composer::compose` (the typed composition: r2il procedures,
//!   compiled once, replayed through mask-risc; support first, so only
//!   coordinates that can hold a path are counted);
//! * C: every coordinate's AND + popcount with `ndarray::simd` primitives on
//!   the 4-word rows, no support pass (the brute-force SIMD reference);
//! * C+: arm A's plan (support first) written directly with
//!   `ndarray::simd`: the gap A − C+ is mask-risc's per-execution cost;
//! * B: LAB-ONLY tuple baseline: every path materialized as a
//!   `(user, role, group)` tuple, sorted, runs counted. It exists only to
//!   show what leaving the fold costs; it is not a candidate.
//!
//! Bind, page preparation, program preparation (`Composer::new`) and
//! execution are timed separately (median of 101 runs, in ns). All three arms must agree on bag and distinct before any timing.
//! Allocations are counted by a wrapping global allocator.
//!
//! ```text
//! cargo run --release --example rft1_bench
//! ```

use dir_rmap_probe::World;
use dir_rmap_probe::compose::{Carriers, Composer, Law, Members, add_bag};
use dir_rmap_probe::fixture::{Fixture, Spec};
use dir_rmap_probe::focus::SLOTS;
use dir_rmap_probe::prep::{Prepared, Refusal, ones};
use dir_rmap_probe::qr::{Page, Qr};
use ndarray::simd::{mask_and_assign, mask_or_assign, popcount_batch_u64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards unchanged to `System`; the counters are
// atomics and do not affect the allocation.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
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

fn counters() -> (usize, usize) {
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

/// Median wall time in ns of `reps` runs of `f`.
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

/// Allocations and bytes of one run of `f`.
fn allocs_of<T>(mut f: impl FnMut() -> T) -> (usize, usize) {
    let (a0, b0) = counters();
    black_box(f());
    let (a1, b1) = counters();
    (a1 - a0, b1 - b0)
}

fn build(spec: Spec) -> Result<(World, Prepared, Qr), Refusal> {
    let w = World::new(Fixture::generate(spec));
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness())?;
    Ok((w, b, qr))
}

fn main() {
    let (w, b, qr) = match build(Spec::iam(8000)) {
        Ok(x) => {
            println!("fixture: Spec::iam(8000)");
            x
        }
        Err(Refusal::TooLarge) => {
            println!(
                "fixture: Spec::iam(8000) exceeds 256 security slots (Refusal::TooLarge); \
                 falling back to Spec::small()"
            );
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

    // Arm A: one prepared composer, reused.
    let mut composer = Composer::new();
    let mut arm_a = || {
        let (mut bag, mut distinct) = (0u64, 0u64);
        for m in &members {
            let out = composer
                .compose(m, &carriers, &current, Law::Count)
                .expect("compose");
            bag = add_bag(bag, out.bag).expect("no overflow");
            distinct = add_bag(distinct, out.distinct).expect("no overflow");
        }
        (bag, distinct)
    };
    // Arm C: direct ndarray::simd on the 4-word rows.
    let arm_c = || {
        let (mut bag, mut distinct) = (0u64, 0u64);
        for (m, &n) in members.iter().zip(&counts) {
            for i in 0..n {
                for r in 0..64 {
                    let mut buf = m.member[i];
                    mask_and_assign(&mut buf, &m.valid[i]);
                    mask_and_assign(&mut buf, &carriers.role_slots[r]);
                    let c = popcount_batch_u64(&buf);
                    bag += c;
                    distinct += u64::from(c > 0);
                }
            }
        }
        (bag, distinct)
    };
    // Arm C+: support first, direct ndarray::simd.
    let arm_cp = || {
        let (mut bag, mut distinct) = (0u64, 0u64);
        for (m, &n) in members.iter().zip(&counts) {
            for i in 0..n {
                let mut present = m.member[i];
                mask_and_assign(&mut present, &m.valid[i]);
                let mut support = [0u64];
                for s in ones(&present) {
                    mask_or_assign(&mut support, std::slice::from_ref(&qr.role[s]));
                }
                distinct += popcount_batch_u64(&support);
                for r in ones(&support) {
                    let mut buf = present;
                    mask_and_assign(&mut buf, &carriers.role_slots[r]);
                    bag += popcount_batch_u64(&buf);
                }
            }
        }
        (bag, distinct)
    };
    // Arm B: LAB-ONLY tuple baseline. A heap row per path, then sort.
    let arm_b = || {
        let mut paths: Vec<(u32, u8, u16)> = Vec::new();
        for (p, (m, &n)) in members.iter().zip(&counts).enumerate() {
            for i in 0..n {
                let mut present = m.member[i];
                mask_and_assign(&mut present, &m.valid[i]);
                for s in ones(&present) {
                    for r in 0..64u8 {
                        if qr.role[s] >> r & 1 == 1 {
                            paths.push(((p * SLOTS + i) as u32, r, s as u16));
                        }
                    }
                }
            }
        }
        paths.sort_unstable();
        let bag = paths.len() as u64;
        let mut distinct = 0u64;
        let mut last = None;
        for &(u, r, _) in &paths {
            if last != Some((u, r)) {
                distinct += 1;
                last = Some((u, r));
            }
        }
        (bag, distinct)
    };

    let (a, c, t) = (arm_a(), arm_c(), arm_b());
    assert_eq!(a, c, "arm A vs arm C: (bag, distinct)");
    assert_eq!(a, arm_cp(), "arm A vs arm C+: (bag, distinct)");
    assert_eq!(a, t, "arm A vs arm B: (bag, distinct)");
    println!(
        "users {users}, pages {}, bag {}, distinct {} (all three arms agree)",
        pages.len(),
        a.0,
        a.1
    );

    const REPS: usize = 101;
    let bind_ns = median_ns(REPS, || Qr::bind(&b, &current).expect("bind"));
    let prep_ns = median_ns(REPS, || {
        (0..qr.pages())
            .map(|p| qr.page(&b, p).expect("page"))
            .collect::<Vec<_>>()
    });
    let compile_ns = median_ns(REPS, Composer::new);
    println!("bind     {bind_ns} ns");
    println!("pages    {prep_ns} ns ({} pages)", pages.len());
    println!("programs {compile_ns} ns (Composer::new)");

    let ta = median_ns(REPS, &mut arm_a);
    let tc = median_ns(REPS, arm_c);
    let tcp = median_ns(REPS, arm_cp);
    let tb = median_ns(REPS, arm_b);
    let (aa, ab) = allocs_of(&mut arm_a);
    let (ca, cb) = allocs_of(arm_c);
    let (cpa, cpb) = allocs_of(arm_cp);
    let (ba, bb) = allocs_of(arm_b);
    println!("A compose (mask-risc)      {ta} ns  allocs {aa} ({ab} B)");
    println!("C ndarray::simd direct     {tc} ns  allocs {ca} ({cb} B)");
    println!("C+ support-first simd      {tcp} ns  allocs {cpa} ({cpb} B)");
    println!("B tuple baseline (lab-only) {tb} ns  allocs {ba} ({bb} B)");
    println!(
        "ratios: B/A {:.2}  A/C {:.2}  A/C+ {:.2}",
        tb as f64 / ta as f64,
        ta as f64 / tc as f64,
        ta as f64 / tcp as f64
    );
}
