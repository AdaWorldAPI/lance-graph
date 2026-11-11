//! The sealed order index: persist, reopen, attest. Every refusal has a test
//! that fires and a silent twin on a non-trivial input. Expected sequences are
//! literals derived by hand from the fixture; nothing here orders rows.

use dr256_ternlog_lab::order_store::{
    layout, KeyLane, SealedOrder, StoreRefused, TIE_KEY_THEN_FRAME_POSITION,
};
use dr256_ternlog_lab::*;
use lance_graph_dt64::dr::Scope;
use lance_graph_mask_risc::LaneRef;

const READER: Scope = Scope { id: 7 };
const AGE: [i32; 12] = [30, 20, 30, 10, 20, 30, 10, 40, 20, 10, 40, 30];
// ids are not the positions: id[i] = 1000 + 7 * i
const BY_AGE_THEN_POSITION: [u32; 12] = [3, 6, 9, 1, 4, 8, 0, 2, 5, 11, 7, 10];
const KEY: KeyLane = KeyLane::Field(person::AGE);
const PRESENCE: u64 = 5;
const INDEX_POP: u64 = 99;

struct Fx {
    id: Vec<u32>,
    age: Vec<i32>,
}

impl Fx {
    fn new() -> Fx {
        Fx::with_age(AGE.to_vec())
    }
    fn with_age(age: Vec<i32>) -> Fx {
        Fx {
            id: (0..12u32).map(|i| 1000 + 7 * i).collect(),
            age,
        }
    }
    fn table(&self, rev: u64) -> Table<'_> {
        Table::new(
            PERSON,
            1,
            12,
            rev,
            READER,
            &[
                (person::ID, LaneRef::U32(&self.id)),
                (person::AGE, LaneRef::I32(&self.age)),
            ],
        )
    }
    fn index(&self, t: &Table<'_>) -> OrderIndex {
        OrderIndex::by_i32(t, person::AGE, INDEX_POP).unwrap()
    }
    fn sealed(&self) -> SealedOrder {
        let t = self.table(1);
        SealedOrder::seal(self.index(&t), &t, KEY, PRESENCE).unwrap()
    }
}

// Keys at each index position, by hand: age 10 at rows 3,6,9; 20 at 1,4,8;
// 30 at 0,2,5,11; 40 at 7,10.
const KEYS_BY_AGE: [i32; 12] = [10, 10, 10, 20, 20, 20, 30, 30, 30, 30, 40, 40];
// With AGE[3] = 50: 10 at 6,9; 20 at 1,4,8; 30 at 0,2,5,11; 40 at 7,10; 50 at 3.
const BY_AGE_THEN_POSITION_MOVED: [u32; 12] = [6, 9, 1, 4, 8, 0, 2, 5, 11, 7, 10, 3];

fn patch_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn patch_u64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
fn get_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn order_off(i: usize) -> usize {
    layout::HEADER_BYTES + 4 * i
}
fn key_off(i: usize) -> usize {
    layout::HEADER_BYTES + 4 * 12 + 4 * i
}
fn swap_u32(b: &mut [u8], a: usize, c: usize) {
    let (x, y) = (get_u32(b, a), get_u32(b, c));
    patch_u32(b, a, y);
    patch_u32(b, c, x);
}
fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("dr256-order-{}-{name}.ordix", std::process::id()))
}

#[test]
fn a_sealed_order_survives_persist_and_reopen_and_reproduces_the_sequence() {
    let fx = Fx::new();
    let want_rows: Vec<u32> = BY_AGE_THEN_POSITION.to_vec();

    // The index as the real constructor builds it, sealed, written, read back.
    let sealed = fx.sealed();
    assert!(sealed.matches_index(&fx.index(&fx.table(1))));
    let bytes = sealed.to_bytes();
    assert_eq!(bytes.len(), 88 + 8 * 12);
    assert_eq!(
        get_u32(&bytes, layout::OFF_TIE_BREAK),
        TIE_KEY_THEN_FRAME_POSITION
    );
    let path = tmp("t1");
    sealed.write_to(&path).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 88 + 8 * 12);

    let reopened = SealedOrder::read_from(&path).unwrap();
    let fresh = fx.table(1);
    assert_eq!(reopened.attest(&fresh, KEY, PRESENCE), Ok(()));
    assert_eq!(reopened.rows(), &BY_AGE_THEN_POSITION[..]);
    assert_eq!(reopened.rows(), &want_rows[..]);
    assert_eq!(reopened.keys(), &KEYS_BY_AGE[..]);
    // Lossless: the reopened seal equals the original index (permutation,
    // keys, population, origin).
    assert!(reopened.matches_index(&fx.index(&fresh)));
    let ids: Vec<u32> = reopened.rows().iter().map(|&r| fx.id[r as usize]).collect();
    let want_ids: Vec<u32> = want_rows.iter().map(|&r| 1000 + 7 * r).collect();
    assert_eq!(ids, want_ids);
    assert_eq!(
        SealedOrder::read_from(&path).unwrap().rows(),
        reopened.rows()
    );
    assert_eq!(reopened.to_bytes(), bytes);
    assert_eq!(reopened.len(), 12);
    assert_eq!(reopened.key(), KEY);
    assert_eq!(reopened.presence(), PRESENCE);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn a_reopened_index_feeds_the_existing_range_count() {
    let world = World::new();
    let fx = Fx::new();
    let table = fx.table(1);
    let mask = select(&world.view, PERSON, &["person_id", "age"]).unwrap();
    let reopened = SealedOrder::from_bytes(&fx.sealed().to_bytes()).unwrap();
    let original = fx.index(&table);
    assert!(reopened.matches_index(&original));
    let index = reopened.into_index(&table, KEY, PRESENCE).unwrap();
    // Same answers as the original through the existing functions.
    assert_eq!(index.positions(), original.positions());
    for (lo, hi) in [(10, 10), (20, 30), (25, 40), (41, 60), (0, 100)] {
        assert_eq!(index.between(lo, hi), original.between(lo, hi));
    }
    assert_eq!(index.gt(20), original.gt(20));

    let visible = |r: usize| !r.is_multiple_of(3);
    let mut prepared = prepare_order(&world, &table, &mask, &index, visible).unwrap();
    let got = prepared
        .count(&world, &table, &index, index.between(20, 30))
        .unwrap();
    let in_band = |r: usize| (20..=30).contains(&AGE[r]);
    let want = (0..12).filter(|&r| visible(r) && in_band(r)).count() as u64;
    let all = (0..12).filter(|&r| in_band(r)).count() as u64;
    assert_eq!(got, want);
    assert!(want > 0 && want < all, "visibility removes rows");
    for (at, &row) in BY_AGE_THEN_POSITION.iter().enumerate() {
        assert_eq!(index.row(at), Some(row));
    }
    let mut again = prepare_order(&world, &table, &mask, &original, visible).unwrap();
    assert_eq!(
        again
            .count(&world, &table, &original, original.between(20, 30))
            .unwrap(),
        want
    );
}

#[test]
fn content_changed_with_the_mapping_unchanged_is_refused() {
    let fx = Fx::new();
    let mut moved = AGE.to_vec();
    moved[3] = 50;
    let after_fx = Fx::with_age(moved.clone());
    let after = after_fx.table(2);
    assert_ne!(
        BY_AGE_THEN_POSITION_MOVED, BY_AGE_THEN_POSITION,
        "the old order would be wrong"
    );
    let rebuilt = after_fx.index(&after);
    for (at, &row) in BY_AGE_THEN_POSITION_MOVED.iter().enumerate() {
        assert_eq!(
            rebuilt.row(at),
            Some(row),
            "the literal matches the real path"
        );
    }

    let err = StoreRefused::SourceChanged {
        sealed: 1,
        found: 2,
    };
    assert_eq!(fx.sealed().attest(&after, KEY, PRESENCE), Err(err.clone()));
    assert_eq!(
        fx.sealed().into_index(&after, KEY, PRESENCE).err(),
        Some(err)
    );
    assert_eq!(fx.sealed().attest(&fx.table(1), KEY, PRESENCE), Ok(()));
}

#[test]
fn advancing_the_source_refuses_and_a_rebuilt_seal_answers() {
    let fx = Fx::new();
    let sealed = fx.sealed();
    let mut table = fx.table(1);
    table.advance_source();
    assert_eq!(
        sealed.attest(&table, KEY, PRESENCE),
        Err(StoreRefused::SourceChanged {
            sealed: 1,
            found: 2
        })
    );
    let rebuilt = SealedOrder::seal(fx.index(&table), &table, KEY, PRESENCE).unwrap();
    assert_eq!(rebuilt.attest(&table, KEY, PRESENCE), Ok(()));
    assert_eq!(rebuilt.source_revision(), 2);
}

#[test]
fn another_table_is_refused() {
    let fx = Fx::new();
    let sealed = fx.sealed();
    let friend = Table::new(FRIEND_OF, 1, 12, 1, READER, &[(0, LaneRef::U32(&fx.id))]);
    assert_eq!(
        sealed.attest(&friend, KEY, PRESENCE),
        Err(StoreRefused::AnotherTable)
    );
    let other_pop = Table::new(
        PERSON,
        2,
        12,
        1,
        READER,
        &[
            (person::ID, LaneRef::U32(&fx.id)),
            (person::AGE, LaneRef::I32(&fx.age)),
        ],
    );
    assert_eq!(
        sealed.attest(&other_pop, KEY, PRESENCE),
        Err(StoreRefused::AnotherTable)
    );
    let short = Table::new(
        PERSON,
        1,
        11,
        1,
        READER,
        &[
            (person::ID, LaneRef::U32(&fx.id[..11])),
            (person::AGE, LaneRef::I32(&fx.age[..11])),
        ],
    );
    assert_eq!(
        sealed.attest(&short, KEY, PRESENCE),
        Err(StoreRefused::AnotherTable)
    );
    assert_eq!(sealed.attest(&fx.table(1), KEY, PRESENCE), Ok(()));
}

#[test]
fn a_moved_mapping_is_refused() {
    let fx = Fx::new();
    let table = fx.table(1);
    let good = fx.sealed().to_bytes();
    let silent = SealedOrder::from_bytes(&good).unwrap();
    assert_eq!(silent.attest(&table, KEY, PRESENCE), Ok(()));
    for off in [layout::OFF_COORDINATES, layout::OFF_RELATION_MAP] {
        let mut b = good.clone();
        patch_u64(&mut b, off, 2);
        let s = SealedOrder::from_bytes(&b).unwrap();
        assert_eq!(
            s.attest(&table, KEY, PRESENCE),
            Err(StoreRefused::MappingChanged)
        );
    }
}

#[test]
fn another_key_and_another_presence_version_are_refused() {
    let fx = Fx::new();
    let table = fx.table(1);
    let s = fx.sealed();
    assert_eq!(
        s.attest(&table, KeyLane::Field(person::ID), PRESENCE),
        Err(StoreRefused::AnotherKey)
    );
    assert_eq!(
        s.attest(&table, KeyLane::Computed(u64::from(person::AGE)), PRESENCE),
        Err(StoreRefused::AnotherKey)
    );
    assert_eq!(
        s.attest(&table, KEY, 6),
        Err(StoreRefused::PresenceChanged {
            sealed: 5,
            found: 6
        })
    );
    assert_eq!(s.attest(&table, KEY, PRESENCE), Ok(()));
}

#[test]
fn the_key_then_frame_position_tie_break_is_part_of_the_file() {
    let good = Fx::new().sealed().to_bytes();
    assert!(SealedOrder::from_bytes(&good).is_ok());
    let mut b = good.clone();
    // Index positions 0 and 1 hold rows 3 and 6, both key 10.
    swap_u32(&mut b, order_off(0), order_off(1));
    assert_eq!(
        SealedOrder::from_bytes(&b).err(),
        Some(StoreRefused::OutOfOrder { at: 1 })
    );
}

#[test]
fn keys_out_of_order_are_refused() {
    let mut b = Fx::new().sealed().to_bytes();
    // Keys are 10,10,10,20,...; swapping positions 2 and 3 gives 10,10,20,10:
    // the first descending pair is at position 3.
    swap_u32(&mut b, key_off(2), key_off(3));
    assert_eq!(
        SealedOrder::from_bytes(&b).err(),
        Some(StoreRefused::OutOfOrder { at: 3 })
    );
}

#[test]
fn a_repeated_or_out_of_range_row_is_refused() {
    let good = Fx::new().sealed().to_bytes();
    // Row 7 sits at position 10; writing it at 11 repeats it.
    let mut repeat = good.clone();
    patch_u32(&mut repeat, order_off(11), 7);
    assert_eq!(
        SealedOrder::from_bytes(&repeat).err(),
        Some(StoreRefused::NotAPermutation { at: 11 })
    );
    let mut range = good.clone();
    patch_u32(&mut range, order_off(0), 12);
    assert_eq!(
        SealedOrder::from_bytes(&range).err(),
        Some(StoreRefused::NotAPermutation { at: 0 })
    );
    assert!(SealedOrder::from_bytes(&good).is_ok());
}

#[test]
fn a_damaged_header_is_refused() {
    let good = Fx::new().sealed().to_bytes();
    let refused = |b: &[u8]| SealedOrder::from_bytes(b).err();
    let mut b = good.clone();
    b[0] ^= 0xFF;
    assert_eq!(refused(&b), Some(StoreRefused::BadMagic));
    let mut b = good.clone();
    patch_u32(&mut b, layout::OFF_VERSION, 2);
    assert_eq!(
        refused(&b),
        Some(StoreRefused::UnknownLayout { version: 2 })
    );
    let mut b = good.clone();
    patch_u32(&mut b, layout::OFF_TIE_BREAK, 2);
    assert_eq!(refused(&b), Some(StoreRefused::UnknownTieBreak { rule: 2 }));
    let mut b = good.clone();
    patch_u32(&mut b, layout::OFF_KEY_KIND, 7);
    assert_eq!(refused(&b), Some(StoreRefused::UnknownKeyKind { kind: 7 }));
    let mut b = good.clone();
    patch_u64(&mut b, layout::OFF_KEY_ID, 256);
    assert_eq!(refused(&b), Some(StoreRefused::UnknownKeyKind { kind: 0 }));
    let mut b = good.clone();
    patch_u64(&mut b, layout::OFF_INDEX_POPULATION, 1);
    assert_eq!(refused(&b), Some(StoreRefused::OwnRowOrderClaimed));
    assert!(SealedOrder::from_bytes(&good).is_ok());
}

#[test]
fn a_wrong_length_is_refused() {
    let good = Fx::new().sealed().to_bytes();
    assert_eq!(good.len(), 184);
    let refused = |b: &[u8]| SealedOrder::from_bytes(b).err();
    assert_eq!(
        refused(&good[..183]),
        Some(StoreRefused::Length {
            expected: 184,
            found: 183
        })
    );
    let mut longer = good.clone();
    longer.push(0);
    assert_eq!(
        refused(&longer),
        Some(StoreRefused::Length {
            expected: 184,
            found: 185
        })
    );
    assert_eq!(
        refused(&good[..10]),
        Some(StoreRefused::Length {
            expected: 88,
            found: 10
        })
    );
    let mut forged = good.clone();
    patch_u64(&mut forged, layout::OFF_N_ROWS, u64::MAX);
    assert_eq!(
        refused(&forged),
        Some(StoreRefused::Length {
            expected: u64::MAX,
            found: 184
        })
    );
    assert!(SealedOrder::from_bytes(&good).is_ok());
}

#[test]
fn seal_checks_the_cached_keys_against_the_lane_once() {
    let fx = Fx::new();
    let table = fx.table(1);
    let negated = |t: &Table<'_>| OrderIndex::by_key(t, INDEX_POP, |r| -AGE[r]).unwrap();
    assert_eq!(
        SealedOrder::seal(negated(&table), &table, KEY, PRESENCE).err(),
        Some(StoreRefused::KeyMismatch { at: 0 })
    );
    assert!(SealedOrder::seal(negated(&table), &table, KeyLane::Computed(1), PRESENCE).is_ok());
    assert!(SealedOrder::seal(fx.index(&table), &table, KEY, PRESENCE).is_ok());
}

#[test]
fn seal_refuses_a_missing_key_lane_and_another_tables_index() {
    let fx = Fx::new();
    let table = fx.table(1);
    for field in [person::ID, person::NAME] {
        assert_eq!(
            SealedOrder::seal(fx.index(&table), &table, KeyLane::Field(field), PRESENCE).err(),
            Some(StoreRefused::KeyLaneMissing { field })
        );
    }
    let later = fx.table(2);
    assert_eq!(
        SealedOrder::seal(fx.index(&table), &later, KEY, PRESENCE).err(),
        Some(StoreRefused::IndexOfAnotherTable)
    );
}

#[test]
fn an_index_over_more_rows_than_the_table_states_is_refused() {
    let fx = Fx::new();
    // The lane holds 12 rows but the table states 10.
    let table = Table::new(
        PERSON,
        1,
        10,
        1,
        READER,
        &[(person::AGE, LaneRef::I32(&fx.age))],
    );
    let index = OrderIndex::by_i32(&table, person::AGE, INDEX_POP).unwrap();
    assert_eq!(
        SealedOrder::seal(index, &table, KEY, PRESENCE).err(),
        Some(StoreRefused::IndexOfAnotherTable)
    );
    assert!(SealedOrder::seal(fx.index(&fx.table(1)), &fx.table(1), KEY, PRESENCE).is_ok());
}

#[test]
fn a_missing_file_is_an_io_refusal() {
    let path = tmp("missing");
    assert_eq!(
        SealedOrder::read_from(&path).err(),
        Some(StoreRefused::Io(std::io::ErrorKind::NotFound))
    );
}

#[test]
fn the_module_never_sorts() {
    let src = include_str!("../src/order_store.rs");
    assert!(!src.contains(".sort"));
}
