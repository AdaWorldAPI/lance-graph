//! D-DRAV-1 AV1 cost split: set preparation, binding and map-only membership, reported
//! separately and combined, against a value gather from the same map. Reported, not gated.
//! Warm medians of 2001 calls; the set is built from a list of row ordinals (no value read).
//!   cargo run --release --example av1_membership
use drf_focus_probe::membership::*;
use drf_focus_probe::*;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const NROWS: usize = 65536;

fn median_ns(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[reps / 2]
}

fn main() {
    let id = SourceId {
        dataset: 1,
        branch: 1,
        version: 1,
        index_seal: 7,
    };
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut ord = Box::new([0u16; SLOTS]);
    ord.iter_mut()
        .for_each(|o| *o = (next() % NROWS as u64) as u16);
    let map = Arc::new(SealedMap::new(MapId(1), 7, ord));
    let rows_data: Vec<Row> = (0..NROWS).map(|_| [0u8; ROW]).collect();
    let soa = Soa {
        rows: &rows_data,
        id,
    };
    let facet = Facet::new(16).unwrap();
    let gather = median_ns(2001, || {
        let mgr = FocusManager::new();
        black_box(mgr.load(&soa, &map, Scope(1), facet).unwrap());
    });
    let bound = BoundMap::bind(map.clone(), id, Scope(1), NROWS).unwrap();
    let bind = median_ns(2001, || {
        black_box(BoundMap::bind(map.clone(), id, Scope(1), NROWS).unwrap());
    });
    println!("value gather (FocusManager::load, same map): {gather:.0} ns");
    println!("bind (seal + 4096 range checks): {bind:.0} ns");
    println!(
        "{:>7} {:>9} {:>9} {:>9} {:>6}",
        "set_k", "prep_ns", "exec_ns", "total_ns", "hits"
    );
    for k in [64usize, 1024, 16384] {
        let list: Vec<u16> = (0..k).map(|_| (next() % NROWS as u64) as u16).collect();
        let prep = median_ns(2001, || {
            black_box(RowSet::new(id, Scope(1), NROWS, list.iter().copied()).unwrap());
        });
        let set = RowSet::new(id, Scope(1), NROWS, list.iter().copied()).unwrap();
        let exec = median_ns(2001, || {
            black_box(membership(black_box(&bound), black_box(&set)).unwrap());
        });
        let hits: u32 = membership(&bound, &set)
            .unwrap()
            .iter()
            .map(|w| w.count_ones())
            .sum();
        println!(
            "{:>7} {:>9.0} {:>9.0} {:>9.0} {:>6}",
            k,
            prep,
            exec,
            bind + prep + exec,
            hits
        );
    }
}
