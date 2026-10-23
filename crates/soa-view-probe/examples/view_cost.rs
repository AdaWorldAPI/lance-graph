//! D-NFR-6: does an operation through a borrowed `NodeRowPacket` cost
//! anything beyond the ndarray kernel it names? Reported, not gated.
//!
//! Arm A: the ndarray kernel on the raw LE bytes. Arm B: the same kernel
//! through `FacetFold` on the packet. Both write into caller scratch, so
//! neither should allocate; a counting allocator checks that.
//!
//! `cargo run --release --example view_cost`
//! Assembly: `cargo rustc --release --example view_cost -- --emit=asm`, then
//! compare the `probe_a_*` / `probe_b_*` bodies.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lance_graph_contract::NodeRowPacket;
use lance_graph_contract::canonical_node::NODE_ROW_STRIDE;
use lance_graph_contract::soa_envelope::SoaEnvelope;
use soa_view_probe::fixture::{CLASS_A, rows};
use soa_view_probe::{FacetFold, FacetSlot};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
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

const AT: usize = 32; // value facet 0

#[inline(never)]
fn probe_a_count(bytes: &[u8], n: usize, sel: &mut [u64]) -> u64 {
    ndarray::simd::eq_u32_strided_to_mask(bytes, AT, NODE_ROW_STRIDE, n, CLASS_A, sel);
    ndarray::simd::popcount_batch_u64(sel)
}

#[inline(never)]
fn probe_b_count(view: &NodeRowPacket<'_>, sel: &mut [u64]) -> u64 {
    view.classid_mask(FacetSlot::value(0).unwrap(), CLASS_A, sel);
    ndarray::simd::popcount_batch_u64(sel)
}

#[inline(never)]
fn probe_a_sum(bytes: &[u8], n: usize, sel: &mut [u64]) -> i64 {
    ndarray::simd::eq_u32_strided_to_mask(bytes, AT, NODE_ROW_STRIDE, n, CLASS_A, sel);
    ndarray::simd::masked_strided_group_sum(bytes, AT + 4, NODE_ROW_STRIDE, n, 12, 1, sel).unwrap()
}

#[inline(never)]
fn probe_b_sum(view: &NodeRowPacket<'_>, sel: &mut [u64]) -> i64 {
    let slot = FacetSlot::value(0).unwrap();
    view.classid_mask(slot, CLASS_A, sel);
    view.payload_byte_sum(slot, sel)
}

#[inline(never)]
fn probe_a_match(bytes: &[u8], n: usize, p: &[u8; 16], c: &[u8; 16], sel: &mut [u64]) -> u64 {
    ndarray::simd::ternary_match_strided16_to_mask(bytes, AT, NODE_ROW_STRIDE, n, p, c, sel);
    ndarray::simd::popcount_batch_u64(sel)
}

#[inline(never)]
fn probe_b_match(view: &NodeRowPacket<'_>, p: &[u8; 16], c: &[u8; 16], sel: &mut [u64]) -> u64 {
    view.facet_match_mask(FacetSlot::value(0).unwrap(), p, c, sel);
    ndarray::simd::popcount_batch_u64(sel)
}

fn median_ns(reps: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            for _ in 0..iters {
                f();
            }
            s.elapsed().as_nanos() as f64 / iters as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[reps / 2]
}

fn main() {
    let mut pattern = [0u8; 16];
    pattern[..4].copy_from_slice(&CLASS_A.to_le_bytes());
    let mut care = [0u8; 16];
    care[..4].fill(0xFF);
    care[4] = 0x80;

    println!("rows   op      A direct ns   B packet ns   allocs A/B");
    for n in [64usize, 4096, 65536] {
        let owner = rows(n);
        let view = NodeRowPacket::new(&owner, 1);
        let bytes = view.as_le_bytes();
        let mut sel = vec![0u64; n.div_ceil(64)];
        let iters = (1 << 22) / n.max(64);

        assert_eq!(
            probe_a_count(bytes, n, &mut sel),
            probe_b_count(&view, &mut sel)
        );
        assert_eq!(
            probe_a_sum(bytes, n, &mut sel),
            probe_b_sum(&view, &mut sel)
        );
        assert_eq!(
            probe_a_match(bytes, n, &pattern, &care, &mut sel),
            probe_b_match(&view, &pattern, &care, &mut sel)
        );

        let row = |name: &str, a: &mut dyn FnMut(), b: &mut dyn FnMut()| {
            let a0 = ALLOCS.load(Ordering::Relaxed);
            let ta = median_ns(21, iters, &mut *a);
            let a1 = ALLOCS.load(Ordering::Relaxed);
            let tb = median_ns(21, iters, &mut *b);
            let b1 = ALLOCS.load(Ordering::Relaxed);
            println!(
                "{n:>6} {name:<7} {ta:>12.1} {tb:>13.1}   {}/{}",
                a1 - a0,
                b1 - a1
            );
        };
        let s = std::cell::RefCell::new(&mut sel);
        row(
            "count",
            &mut || {
                black_box(probe_a_count(black_box(bytes), n, &mut s.borrow_mut()));
            },
            &mut || {
                black_box(probe_b_count(black_box(&view), &mut s.borrow_mut()));
            },
        );
        row(
            "sum",
            &mut || {
                black_box(probe_a_sum(black_box(bytes), n, &mut s.borrow_mut()));
            },
            &mut || {
                black_box(probe_b_sum(black_box(&view), &mut s.borrow_mut()));
            },
        );
        row(
            "match",
            &mut || {
                black_box(probe_a_match(
                    black_box(bytes),
                    n,
                    &pattern,
                    &care,
                    &mut s.borrow_mut(),
                ));
            },
            &mut || {
                black_box(probe_b_match(
                    black_box(&view),
                    &pattern,
                    &care,
                    &mut s.borrow_mut(),
                ));
            },
        );
    }
}
