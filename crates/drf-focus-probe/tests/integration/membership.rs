//! D-DRAV-1 AV1: map-only membership against an independent scalar oracle.
use drf_focus_probe::membership::*;
use drf_focus_probe::*;
use std::sync::Arc;

const N: usize = 65536;

fn src(dataset: u64) -> SourceId {
    SourceId {
        dataset,
        branch: 1,
        version: 3,
        index_seal: 7,
    }
}

fn xs(mut s: u64) -> impl FnMut() -> u64 {
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn map_from(ord: Box<[u16; SLOTS]>) -> Arc<SealedMap> {
    Arc::new(SealedMap::new(MapId(1), 7, ord))
}

fn random_map(seed: u64, nrows: usize) -> Box<[u16; SLOTS]> {
    let mut r = xs(seed);
    let mut o = Box::new([0u16; SLOTS]);
    o.iter_mut().for_each(|v| *v = (r() % nrows as u64) as u16);
    o
}

/// Independent oracle: linear search over the row list, bit order slot i -> word i/64, bit i%64.
fn oracle(ord: &[u16; SLOTS], rows: &[u16]) -> SlotMask {
    let mut m = [0u64; TILE_WORDS];
    for i in 0..SLOTS {
        if rows.iter().any(|&r| r == ord[i]) {
            m[i / 64] += 1u64 << (i % 64);
        }
    }
    m
}

fn pop(m: &SlotMask) -> u32 {
    m.iter().map(|w| w.count_ones()).sum()
}

fn run(ord: &[u16; SLOTS], rows: &[u16], nrows: usize) -> SlotMask {
    let b = BoundMap::bind(map_from(Box::new(*ord)), src(1), Scope(1), nrows).unwrap();
    let s = RowSet::new(src(1), Scope(1), nrows, rows.iter().copied()).unwrap();
    membership(&b, &s).unwrap()
}

#[test]
fn av1_random_map_matches_oracle_with_hits_and_misses() {
    for seed in 1..=8u64 {
        let ord = random_map(seed, 2048);
        let mut r = xs(seed * 31);
        let rows: Vec<u16> = (0..300).map(|_| (r() % 2048) as u16).collect();
        let m = run(&ord, &rows, 2048);
        assert_eq!(m, oracle(&ord, &rows), "seed {seed}");
        let p = pop(&m);
        assert!(p > 0 && (p as usize) < SLOTS, "anti-vacuity: {p} hits");
    }
}

#[test]
fn av1_duplicate_ordinals_all_match() {
    // 1000 slots point at row 7, the rest at row 9.
    let mut ord = Box::new([9u16; SLOTS]);
    for (i, o) in ord.iter_mut().enumerate() {
        if i % 4 == 1 || i < 24 {
            *o = 7;
        }
    }
    let dup = ord.iter().filter(|&&o| o == 7).count() as u32;
    assert!(dup > 1000);
    let m = run(&ord, &[7], 16);
    assert_eq!(pop(&m), dup, "every slot that points at row 7 matches");
    assert_eq!(m, oracle(&ord, &[7]));
    // A repeated row in the set changes nothing.
    assert_eq!(run(&ord, &[7, 7, 7], 16), m);
}

#[test]
fn av1_empty_and_full_sets() {
    let ord = random_map(5, 300);
    assert_eq!(pop(&run(&ord, &[], 300)), 0);
    let all: Vec<u16> = (0..300).collect();
    assert_eq!(run(&ord, &all, 300), [u64::MAX; TILE_WORDS]);
}

#[test]
fn av1_shuffled_map_permutes_the_mask() {
    let ord = random_map(11, 1000);
    let rows: Vec<u16> = (0..1000).filter(|r| r % 3 == 0).collect();
    let base = run(&ord, &rows, 1000);
    // Seeded permutation p: shuffled[i] = ord[p[i]], so bit i of the new mask = bit p[i] of the old.
    let mut p: Vec<usize> = (0..SLOTS).collect();
    let mut r = xs(99);
    for i in (1..SLOTS).rev() {
        p.swap(i, (r() % (i as u64 + 1)) as usize);
    }
    let mut sh = Box::new([0u16; SLOTS]);
    for i in 0..SLOTS {
        sh[i] = ord[p[i]];
    }
    let m = run(&sh, &rows, 1000);
    for i in 0..SLOTS {
        let got = m[i / 64] >> (i % 64) & 1;
        let want = base[p[i] / 64] >> (p[i] % 64) & 1;
        assert_eq!(got, want, "slot {i}");
    }
    assert_ne!(m, base, "anti-vacuity: the shuffle moved bits");
}

#[test]
fn av1_cross_dataset_ordinal_collision_refuses() {
    let ord = random_map(3, 500);
    let b = BoundMap::bind(map_from(ord), src(1), Scope(1), 500).unwrap();
    // Same ordinals, other dataset: the row 42 here is a different row.
    let other = RowSet::new(src(2), Scope(1), 500, [42u16, 43]).unwrap();
    assert!(matches!(
        membership(&b, &other),
        Err(Refusal::DomainMismatch { .. })
    ));
    let mut v = src(1);
    v.version = 4;
    let newer = RowSet::new(v, Scope(1), 500, [42u16]).unwrap();
    assert!(matches!(
        membership(&b, &newer),
        Err(Refusal::DomainMismatch { .. })
    ));
    let scoped = RowSet::new(src(1), Scope(2), 500, [42u16]).unwrap();
    assert!(matches!(
        membership(&b, &scoped),
        Err(Refusal::ScopeMismatch { .. })
    ));
    // Control: same domain and scope is answered.
    let same = RowSet::new(src(1), Scope(1), 500, [42u16]).unwrap();
    assert!(membership(&b, &same).is_ok());
}

#[test]
fn av1_binding_refuses_stale_seal_and_out_of_range() {
    let mut s = src(1);
    s.index_seal = 8;
    assert!(matches!(
        BoundMap::bind(map_from(random_map(1, 100)), s, Scope(1), 100),
        Err(Refusal::StaleMap { .. })
    ));
    assert!(matches!(
        BoundMap::bind(map_from(random_map(1, 100)), src(1), Scope(1), 99),
        Err(Refusal::OrdinalOutOfRange { .. })
    ));
    assert!(matches!(
        RowSet::new(src(1), Scope(1), 100, [100u16]),
        Err(Refusal::OrdinalOutOfRange { .. })
    ));
    // Control: full u16 domain binds.
    assert!(BoundMap::bind(map_from(random_map(1, N)), src(1), Scope(1), N).is_ok());
}
