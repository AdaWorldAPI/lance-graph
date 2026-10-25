//! D-DRF W8 + focus churn: eviction under a resident-byte budget while
//! readers still hold foci.
use drf_focus_probe::*;
use std::sync::Arc;

const NROWS: usize = 1000;
const OFF: usize = 16;

fn rows() -> Vec<Row> {
    (0..NROWS)
        .map(|i| {
            let mut r = [0u8; ROW];
            r[OFF..OFF + 4].copy_from_slice(&((i as i32) * 3 - 700).to_le_bytes());
            r
        })
        .collect()
}
fn soa(r: &[Row]) -> Soa<'_> {
    Soa {
        rows: r,
        id: SourceId {
            dataset: 1,
            branch: 1,
            version: 1,
            index_seal: 9,
        },
    }
}
fn map(id: u64) -> Arc<SealedMap> {
    let mut o = Box::new([0u16; SLOTS]);
    for (s, v) in o.iter_mut().enumerate() {
        *v = ((s * (id as usize + 1)) % NROWS) as u16;
    }
    Arc::new(SealedMap::new(MapId(id), 9, o))
}
fn facet() -> Facet {
    Facet::new(OFF).unwrap()
}
const ALL: [u64; TILE_WORDS] = [u64::MAX; TILE_WORDS];

#[test]
fn w8_a_pinned_focus_survives_eviction_and_may_exceed_the_budget() {
    let r = rows();
    let s = soa(&r);
    let (ma, mb, mc) = (map(1), map(2), map(3));
    let mgr = FocusManager::new();
    let a = mgr.load(&s, &ma, Scope(1), facet()).unwrap();
    let want = direct_sum_where(&s, &ma, facet(), &ALL, 0).unwrap();
    drop(mgr.load(&s, &mb, Scope(1), facet()).unwrap());
    drop(mgr.load(&s, &mc, Scope(1), facet()).unwrap());
    assert_eq!(mgr.resident_bytes(), 3 * FOCUS_BYTES);

    let rep = mgr.evict_to_budget(0);
    assert_eq!(rep.evicted, 2, "only the two unpinned foci go");
    assert_eq!(rep.resident_bytes, FOCUS_BYTES);
    assert_eq!(rep.pinned_bytes, FOCUS_BYTES);
    assert!(
        rep.over_budget,
        "a live reader may hold the manager over budget"
    );
    assert_eq!(mgr.len(), 1);
    // The pinned view is untouched and still correct.
    assert_eq!(a.sum_where(&ALL), want);

    drop(a);
    let rep = mgr.evict_to_budget(0);
    assert_eq!(
        (rep.evicted, rep.resident_bytes, rep.over_budget),
        (1, 0, false)
    );
    assert!(mgr.is_empty());
}

#[test]
fn w8_an_evicted_view_rebuilds_deterministically() {
    let r = rows();
    let s = soa(&r);
    let m = map(4);
    let mgr = FocusManager::new();
    let first = mgr.load(&s, &m, Scope(1), facet()).unwrap();
    let col: Vec<i32> = first.column().to_vec();
    let sum = first.sum_where(&ALL);
    drop(first);
    assert_eq!(mgr.evict_to_budget(0).evicted, 1);
    let again = mgr.load(&s, &m, Scope(1), facet()).unwrap();
    assert_eq!(mgr.builds(), 2, "evicted, so it was gathered again");
    assert_eq!(again.column().to_vec(), col);
    assert_eq!(again.sum_where(&ALL), sum);
}

#[test]
fn w8_eviction_takes_the_least_recently_loaded_focus_first() {
    let r = rows();
    let s = soa(&r);
    let (ma, mb, mc) = (map(1), map(2), map(3));
    let mgr = FocusManager::new();
    for m in [&ma, &mb, &mc] {
        drop(mgr.load(&s, m, Scope(1), facet()).unwrap());
    }
    // A is loaded again (a hit): B is now the least recently used.
    drop(mgr.load(&s, &ma, Scope(1), facet()).unwrap());
    assert_eq!(mgr.builds(), 3);
    let rep = mgr.evict_to_budget(2 * FOCUS_BYTES);
    assert_eq!(
        (rep.evicted, rep.resident_bytes, rep.over_budget),
        (1, 2 * FOCUS_BYTES, false)
    );
    // An exact fit evicts nothing.
    assert_eq!(mgr.evict_to_budget(2 * FOCUS_BYTES).evicted, 0);
    // A and C are hits, B was the one evicted.
    drop(mgr.load(&s, &ma, Scope(1), facet()).unwrap());
    drop(mgr.load(&s, &mc, Scope(1), facet()).unwrap());
    assert_eq!(mgr.builds(), 3);
    drop(mgr.load(&s, &mb, Scope(1), facet()).unwrap());
    assert_eq!(mgr.builds(), 4);
}

/// Cyclic switching over `k` foci with an eviction pass after every load.
fn cyclic_builds(k: u64, budget_foci: usize, rounds: usize) -> usize {
    let r = rows();
    let s = soa(&r);
    let maps: Vec<_> = (1..=k).map(map).collect();
    let mgr = FocusManager::new();
    for _ in 0..rounds {
        for m in &maps {
            drop(mgr.load(&s, m, Scope(1), facet()).unwrap());
            mgr.evict_to_budget(budget_foci * FOCUS_BYTES);
        }
    }
    mgr.builds()
}

#[test]
fn churn_cyclic_switching_rebuilds_every_load_once_the_budget_is_one_short() {
    // ABCABC... over 3 foci for 10 rounds: 30 loads.
    assert_eq!(cyclic_builds(3, 3, 10), 3, "fits: one gather per focus");
    // One focus short: LRU always evicts exactly the focus needed next.
    assert_eq!(cyclic_builds(3, 2, 10), 30, "every load is a rebuild");
    assert_eq!(cyclic_builds(3, 1, 10), 30);
}
