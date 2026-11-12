//! Cost of the compatibility buffer (D-DR-SELECT-K-SV-3), term by term.
//! `cargo run --release --example selvec_cost`. Medians of 201, ns, TSV.
//!
//! - `native_borrow`: handing out the borrowed native list (no copy).
//! - `create`, `destroy`: an empty owned buffer of K entries.
//! - `materialize`: `selection_from_positions` (create + copy K*4 bytes).
//! - `read`: one pass over the buffer's K entries.
//! - `revalidate`: the range check a consumer runs after an external write.

use std::hint::black_box;
use std::time::Instant;

use duckdb_selvec_compat::{
    create_selection_vector, destroy_selection_vector, positions, selection_from_positions,
};

const N: u32 = 1 << 20;

fn median_ns<T>(mut f: impl FnMut() -> T) -> u128 {
    for _ in 0..20 {
        black_box(f());
    }
    let mut t: Vec<u128> = (0..201)
        .map(|_| {
            let s = Instant::now();
            black_box(f());
            s.elapsed().as_nanos()
        })
        .collect();
    t.sort_unstable();
    t[100]
}

fn main() {
    println!("k\tnative_borrow_ns\tcreate_destroy_ns\tmaterialize_destroy_ns\tread_ns\trevalidate_ns\tbytes_copied\tallocations");
    for k in [64usize, 256, 65_536] {
        let native: Vec<u32> = (0..k as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % N)
            .collect();
        let borrow = median_ns(|| black_box(native.as_slice()).len());
        let create = median_ns(|| {
            let s = create_selection_vector(k as u64);
            // SAFETY: `s` is live and destroyed once.
            unsafe { destroy_selection_vector(s) }
        });
        let mut receipt = None;
        let materialize = median_ns(|| {
            let (s, r) = selection_from_positions(black_box(&native));
            receipt = Some(r);
            // SAFETY: `s` is live and destroyed once.
            unsafe { destroy_selection_vector(s) }
        });
        let (sel, _) = selection_from_positions(&native);
        // SAFETY: `sel` is live and not written during the measurements.
        let pos = unsafe { positions(sel, k as u64) }.unwrap();
        let read = median_ns(|| black_box(pos).iter().map(|&p| u64::from(p)).sum::<u64>());
        let revalidate = median_ns(|| black_box(pos).iter().position(|&p| p >= N));
        // SAFETY: `pos` is not used after this point.
        unsafe { destroy_selection_vector(sel) };
        let r = receipt.unwrap();
        println!(
            "{k}\t{borrow}\t{create}\t{materialize}\t{read}\t{revalidate}\t{}\t{}",
            r.bytes, r.allocations
        );
    }
}
