//! D-DRF-P1 falsifiers. Negative race/stale fixtures first (spec §9).
use drf_focus_probe::*;
use std::sync::Arc;

const NROWS: usize = 1000;
const OFF: usize = 16;

fn rows() -> Vec<Row> {
    (0..NROWS)
        .map(|i| {
            let mut r = [0u8; ROW];
            let v = (i as i32).wrapping_mul(7919) - 3000; // mixed signs
            r[OFF..OFF + 4].copy_from_slice(&v.to_le_bytes());
            r
        })
        .collect()
}
fn facet_of(i: usize) -> i64 {
    ((i as i32).wrapping_mul(7919) - 3000) as i64
}
fn sid(dataset: u64, branch: u64, version: u64, seal: u64) -> SourceId {
    SourceId {
        dataset,
        branch,
        version,
        index_seal: seal,
    }
}
fn map_from(id: u64, seal: u64, mut f: impl FnMut(usize) -> u16) -> Arc<SealedMap> {
    let mut o = Box::new([0u16; SLOTS]);
    for (s, v) in o.iter_mut().enumerate() {
        *v = f(s);
    }
    Arc::new(SealedMap::new(MapId(id), seal, o))
}
fn xs(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
fn facet() -> Facet {
    Facet::new(OFF).unwrap()
}
fn oracle(map: &SealedMap, mask: &[u64; 64]) -> i64 {
    (0..SLOTS)
        .filter(|s| mask[s / 64] >> (s % 64) & 1 == 1)
        .map(|s| facet_of(map.ord[s] as usize))
        .sum()
}

#[test]
fn g4_stale_map_is_refused_and_matching_seal_accepted() {
    let r = rows();
    let m = map_from(1, 10, |s| (s % NROWS) as u16);
    let bad = Soa {
        rows: &r,
        id: sid(1, 1, 1, 11),
    };
    let good = Soa {
        rows: &r,
        id: sid(1, 1, 1, 10),
    };
    let mgr = FocusManager::new();
    assert_eq!(
        mgr.load(&bad, &m, Scope(1), facet()).err(),
        Some(Refusal::StaleMap {
            map_seal: 10,
            source_seal: 11
        })
    );
    assert!(mgr.is_empty(), "a refusal must not stay resident");
    assert!(mgr.load(&good, &m, Scope(1), facet()).is_ok());
    let mask = [u64::MAX; 64];
    assert!(direct_sum_where(&bad, &m, facet(), &mask, 0).is_err());
}

#[test]
fn ordinal_out_of_range_is_refused_not_zeroed_not_prefetched() {
    let r = rows();
    let m = map_from(2, 5, |s| if s == 77 { NROWS as u16 } else { 0 });
    let soa = Soa {
        rows: &r,
        id: sid(1, 1, 1, 5),
    };
    let want = Refusal::OrdinalOutOfRange {
        slot: 77,
        ord: NROWS as u16,
    };
    assert_eq!(
        FocusManager::new().load(&soa, &m, Scope(1), facet()).err(),
        Some(want.clone())
    );
    for la in [0, 4, 64] {
        assert_eq!(
            direct_sum_where(&soa, &m, facet(), &[u64::MAX; 64], la),
            Err(want.clone())
        );
    }
}

#[test]
fn facet_out_of_row_is_refused() {
    assert!(Facet::new(ROW - 4).is_ok());
    assert_eq!(Facet::new(ROW - 3), Err(Refusal::FacetOutOfRow));
    assert_eq!(Facet::new(ROW), Err(Refusal::FacetOutOfRow));
}

#[test]
fn g1_geometry() {
    assert_eq!(std::mem::size_of::<Row>(), 512);
    assert_eq!(std::mem::size_of::<[u16; SLOTS]>(), 8192);
    assert_eq!(std::mem::size_of::<[u64; TILE_WORDS]>(), 512);
    assert_eq!(TILE_WORDS * 64, SLOTS);
}

#[test]
fn g3_other_dataset_or_branch_never_reuses() {
    let r = rows();
    let m = map_from(3, 9, |s| (s % NROWS) as u16);
    let mgr = FocusManager::new();
    let a = mgr
        .load(
            &Soa {
                rows: &r,
                id: sid(1, 1, 1, 9),
            },
            &m,
            Scope(1),
            facet(),
        )
        .unwrap();
    let b = mgr
        .load(
            &Soa {
                rows: &r,
                id: sid(2, 1, 1, 9),
            },
            &m,
            Scope(1),
            facet(),
        )
        .unwrap();
    let c = mgr
        .load(
            &Soa {
                rows: &r,
                id: sid(1, 2, 1, 9),
            },
            &m,
            Scope(1),
            facet(),
        )
        .unwrap();
    assert_eq!(mgr.builds(), 3);
    assert!(!Arc::ptr_eq(&a, &b) && !Arc::ptr_eq(&a, &c) && !Arc::ptr_eq(&b, &c));
    // Positive twin: identical full witness reuses.
    let a2 = mgr
        .load(
            &Soa {
                rows: &r,
                id: sid(1, 1, 1, 9),
            },
            &m,
            Scope(1),
            facet(),
        )
        .unwrap();
    assert!(Arc::ptr_eq(&a, &a2));
    assert_eq!(mgr.builds(), 3);
}

#[test]
fn g6_scope_is_part_of_the_key() {
    let r = rows();
    let m = map_from(4, 9, |s| (s % NROWS) as u16);
    let soa = Soa {
        rows: &r,
        id: sid(1, 1, 1, 9),
    };
    let mgr = FocusManager::new();
    let a = mgr.load(&soa, &m, Scope(1), facet()).unwrap();
    let b = mgr.load(&soa, &m, Scope(2), facet()).unwrap();
    assert_eq!(mgr.builds(), 2);
    assert!(!Arc::ptr_eq(&a, &b));
    assert_ne!(a.key(), b.key());
}

#[test]
fn l1_concurrent_loads_build_once() {
    let r = rows();
    let m = map_from(5, 9, |s| (s * 3 % NROWS) as u16);
    let soa = Soa {
        rows: &r,
        id: sid(1, 1, 1, 9),
    };
    let mgr = FocusManager::new();
    let got: Vec<Arc<Focus>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..8)
            .map(|_| sc.spawn(|| mgr.load(&soa, &m, Scope(1), facet()).unwrap()))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(mgr.builds(), 1);
    assert!(got.iter().all(|f| Arc::ptr_eq(f, &got[0])));
}

#[test]
fn l2_retire_keeps_holders_valid_and_b_is_independent() {
    let r = rows();
    let soa = Soa {
        rows: &r,
        id: sid(1, 1, 1, 9),
    };
    let ma = map_from(6, 9, |s| (s % NROWS) as u16);
    let mb = map_from(7, 9, |s| ((s * 5 + 1) % NROWS) as u16);
    let mask = [0x5555_5555_5555_5555u64; 64];
    let mgr = FocusManager::new();
    let a = mgr.load(&soa, &ma, Scope(1), facet()).unwrap();
    let before = a.sum_where(&mask);
    assert_eq!(before, oracle(&ma, &mask));
    mgr.retire(a.key());
    assert!(mgr.is_empty());
    let b = mgr.load(&soa, &mb, Scope(1), facet()).unwrap();
    assert_eq!(a.sum_where(&mask), before);
    assert_eq!(b.sum_where(&mask), oracle(&mb, &mask));
    assert_ne!(
        before,
        b.sum_where(&mask),
        "fixture must distinguish A from B"
    );
    // Reloading A after retire is a fresh build, not a resurrection.
    let a2 = mgr.load(&soa, &ma, Scope(1), facet()).unwrap();
    assert!(!Arc::ptr_eq(&a, &a2));
    assert_eq!(mgr.builds(), 3);
}

#[test]
fn l3_evict_unused_reclaims_only_unheld() {
    let r = rows();
    let soa = Soa {
        rows: &r,
        id: sid(1, 1, 1, 9),
    };
    let ma = map_from(8, 9, |s| (s % NROWS) as u16);
    let mb = map_from(9, 9, |s| ((s + 1) % NROWS) as u16);
    let mgr = FocusManager::new();
    let held = mgr.load(&soa, &ma, Scope(1), facet()).unwrap();
    let spec = mgr.load(&soa, &mb, Scope(1), facet()).unwrap();
    let weak = Arc::downgrade(&spec);
    drop(spec);
    assert_eq!(mgr.len(), 2);
    assert!(mgr.evict_unused() >= 1);
    assert_eq!(mgr.len(), 1);
    assert!(
        weak.upgrade().is_none(),
        "speculative focus must be reclaimed"
    );
    // The held one survives and is still the cached instance.
    let again = mgr.load(&soa, &ma, Scope(1), facet()).unwrap();
    assert!(Arc::ptr_eq(&held, &again));
    assert_eq!(mgr.evict_unused(), 0);
}

#[test]
fn p1_direct_equals_oracle_equals_focus_for_all_lookaheads() {
    let r = rows();
    let soa = Soa {
        rows: &r,
        id: sid(1, 1, 1, 9),
    };
    let mut st = 0x9E37_79B9_7F4A_7C15u64;
    let patterns: Vec<(&str, Arc<SealedMap>)> = vec![
        (
            "random",
            map_from(10, 9, |_| (xs(&mut st) % NROWS as u64) as u16),
        ),
        ("clustered", map_from(11, 9, |s| (s / 8 % NROWS) as u16)),
        ("repeated", map_from(12, 9, |_| 123)),
        (
            "descending",
            map_from(13, 9, |s| (NROWS - 1 - s % NROWS) as u16),
        ),
        ("duplicates", map_from(14, 9, |s| [5u16, 5, 900, 5][s % 4])),
    ];
    let masks: [[u64; 64]; 3] = [[u64::MAX; 64], [0xA5A5_0F0F_FFFF_0001; 64], {
        let mut m = [0u64; 64];
        for w in m.iter_mut() {
            *w = xs(&mut st);
        }
        m
    }];
    let mgr = FocusManager::new();
    for (name, map) in &patterns {
        let focus = mgr.load(&soa, map, Scope(1), facet()).unwrap();
        for mask in &masks {
            let want = oracle(map, mask);
            assert_eq!(focus.sum_where(mask), want, "{name} focus");
            for la in [0usize, 4, 8, 16, 32, 64] {
                assert_eq!(
                    direct_sum_where(&soa, map, facet(), mask, la).unwrap(),
                    want,
                    "{name} lookahead {la}"
                );
            }
        }
    }
    // Anti-vacuity: the fixture's sums are not all equal.
    let s: Vec<i64> = patterns.iter().map(|(_, m)| oracle(m, &masks[0])).collect();
    assert!(s.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn p2_mispredicted_focus_and_64_fast_switches_leave_no_lease_and_no_corruption() {
    let r = rows();
    let soa = Soa {
        rows: &r,
        id: sid(1, 1, 1, 9),
    };
    let mgr = FocusManager::new();
    let full = [u64::MAX; 64];
    // Predict B, use C: B is loaded and never read.
    let b = map_from(100, 9, |s| (s % NROWS) as u16);
    let c = map_from(101, 9, |s| ((s * 7) % NROWS) as u16);
    drop(mgr.load(&soa, &b, Scope(1), facet()).unwrap());
    let fc = mgr.load(&soa, &c, Scope(1), facet()).unwrap();
    assert_eq!(fc.sum_where(&full), oracle(&c, &full));
    // 64 fast switches; only the last one stays held.
    let mut held = None;
    for k in 0..64u64 {
        let m = map_from(200 + k, 9, |s| ((s + k as usize) % NROWS) as u16);
        let f = mgr.load(&soa, &m, Scope(1), facet()).unwrap();
        assert_eq!(f.sum_where(&full), oracle(&m, &full), "switch {k}");
        held = Some((f, m));
    }
    assert_eq!(mgr.len(), 66);
    // Every unheld entry is reclaimed: B and 63 switches. C and the last stay.
    assert_eq!(mgr.evict_unused(), 64);
    assert_eq!(mgr.len(), 2);
    let (last, lm) = held.unwrap();
    assert_eq!(last.sum_where(&full), oracle(&lm, &full));
    assert_eq!(fc.sum_where(&full), oracle(&c, &full));
    drop((last, fc));
    assert_eq!(mgr.evict_unused(), 2);
    assert!(mgr.is_empty());
}
