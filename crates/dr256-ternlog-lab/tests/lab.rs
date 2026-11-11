//! Every fold is compared with a row-at-a-time count that touches neither
//! DT64 nor mask-RISC. Each refusal has a test that fires and a twin that
//! stays silent on a non-trivial input.

use dr256_ternlog_lab::*;
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::{LaneRefusal, Refusal};
use lance_graph_mask_risc::LaneRef;

const READER: Scope = Scope { id: 7 };

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
/// 3 of 4 rows are present.
fn present(i: usize) -> bool {
    mix(i as u64) % 4 != 0
}
/// 7 of 8 rows are in the reader's scope.
fn in_scope(i: usize) -> bool {
    mix(i as u64 ^ 0xA5A5) % 8 != 0
}

struct People {
    id: Vec<u32>,
    age: Vec<i32>,
}
impl People {
    fn new(n: usize) -> People {
        People {
            id: (0..n as u32).collect(),
            age: (0..n as i32).map(|i| 20 + (i % 60)).collect(),
        }
    }
    fn table(&self) -> Table<'_> {
        Table::new(
            PERSON,
            1,
            self.age.len(),
            1,
            READER,
            &[
                (person::ID, LaneRef::U32(&self.id)),
                (person::AGE, LaneRef::I32(&self.age)),
            ],
        )
    }
    fn mask(world: &World) -> WideFieldMask {
        select(&world.view, PERSON, &["person_id", "age"]).unwrap()
    }
}

struct Edges {
    src: Vec<u32>,
    dst: Vec<u32>,
}
impl Edges {
    /// `m` edges over ids `0..domain`, uneven degrees.
    fn hashed(m: usize, domain: u64) -> Edges {
        Edges {
            src: (0..m as u64)
                .map(|j| (mix(2 * j) % domain) as u32)
                .collect(),
            dst: (0..m as u64)
                .map(|j| (mix(2 * j + 1) % domain) as u32)
                .collect(),
        }
    }
    fn table(&self) -> Table<'_> {
        Table::new(
            FRIEND_OF,
            2,
            self.src.len(),
            1,
            READER,
            &[
                (friend::SRC, LaneRef::U32(&self.src)),
                (friend::DST, LaneRef::U32(&self.dst)),
            ],
        )
    }
}

#[test]
fn filter_count_equals_the_row_count_across_slab_boundaries() {
    let world = World::new();
    // 100: one short slab. 65,536: exactly one. 70,000: one full, one short.
    // 200,000: four slabs.
    for n in [100usize, 65_536, 70_000, 200_000] {
        let people = People::new(n);
        let table = people.table();
        let prepared =
            prepare_nodes(&world, &table, &People::mask(&world), present, in_scope).unwrap();
        assert_eq!(prepared.slabs(), n.div_ceil(SLAB_ROWS));
        let mut fold = Fold::count_gt(&prepared, person::AGE, 50).unwrap();
        let got = fold.count(&world, &table, &prepared).unwrap();
        let want = (0..n)
            .filter(|&i| present(i) && in_scope(i) && people.age[i] > 50)
            .count() as u64;
        assert_eq!(got, want, "n={n}");
        // Each of the three planes removes rows.
        let age_only = (0..n).filter(|&i| people.age[i] > 50).count() as u64;
        let no_scope = (0..n).filter(|&i| present(i) && people.age[i] > 50).count() as u64;
        assert!(want < no_scope && no_scope < age_only && want > 0, "n={n}");
        // Reuse: the same prepared planes answer again, and another threshold.
        assert_eq!(fold.count(&world, &table, &prepared).unwrap(), want);
        let mut other = Fold::count_gt(&prepared, person::AGE, 70).unwrap();
        let want70 = (0..n)
            .filter(|&i| present(i) && in_scope(i) && people.age[i] > 70)
            .count() as u64;
        assert_eq!(other.count(&world, &table, &prepared).unwrap(), want70);
        assert!(want70 < want);
    }
}

#[test]
fn one_hop_count_reads_the_fk_once_and_equals_the_row_count() {
    let world = World::new();
    let n = 70_000usize;
    let m = 150_000usize;
    let edges = Edges::hashed(m, n as u64);
    let table = edges.table();
    let visible = |id: u32| Some(present(id as usize) && in_scope(id as usize));
    let edge_live = |e: usize| mix(e as u64 ^ 0x77) % 16 != 0;
    let prepared = prepare_edges(&world, &table, edge_live, visible).unwrap();
    assert_eq!(prepared.slabs(), 3);
    assert_eq!(
        prepared.lookups,
        2 * m as u64,
        "two fk reads per edge, once"
    );
    let mut fold = Fold::count_and3().unwrap();
    let want = (0..m)
        .filter(|&e| {
            edge_live(e) && visible(edges.src[e]).unwrap() && visible(edges.dst[e]).unwrap()
        })
        .count() as u64;
    assert_eq!(fold.count(&world, &table, &prepared).unwrap(), want);
    assert_eq!(fold.count(&world, &table, &prepared).unwrap(), want);
    assert_eq!(prepared.lookups, 2 * m as u64, "a query adds no fk read");
    let all = (0..m).filter(|&e| edge_live(e)).count() as u64;
    assert!(want > 0 && want < all);
}

#[test]
fn rank_map_equals_a_naive_rank_and_keeps_order() {
    let globals = 3 * SLAB_ROWS + 1234;
    let map = RankMap::new(globals, present).unwrap();
    let mut next = 0u64;
    let mut absent = 0;
    for g in 0..globals {
        if present(g) {
            assert_eq!(map.local(g as u64), Some(next), "g={g}");
            next += 1;
        } else {
            assert_eq!(map.local(g as u64), None);
            absent += 1;
        }
    }
    assert_eq!(map.len(), next);
    assert!(absent > globals / 5);
    assert_eq!(map.local(globals as u64), None, "past the domain");
    assert_eq!(map.local(u64::MAX), None);
    // 256 slabs is the whole domain; one id more is refused.
    assert!(RankMap::new(MAX_SLABS * SLAB_ROWS + 1, |_| false).is_none());
}

#[test]
fn one_hop_over_compacted_nodes_translates_global_ids_through_the_rank_map() {
    let world = World::new();
    // Ids 0..globals; only the present ones have a node row. Node rows are
    // compacted in id order, so a node's row is its rank.
    let globals = 2 * SLAB_ROWS + 5_000;
    let map = RankMap::new(globals, present).unwrap();
    let local_scope: Vec<bool> = (0..map.len() as usize).map(in_scope).collect();
    let m = 90_000usize;
    let edges = Edges::hashed(m, globals as u64);
    let table = edges.table();
    let visible = |id: u32| {
        Some(
            map.local(u64::from(id))
                .is_some_and(|l| local_scope[l as usize]),
        )
    };
    let prepared = prepare_edges(&world, &table, |_| true, visible).unwrap();
    let mut fold = Fold::count_and3().unwrap();
    // Oracle without the map: rank by counting.
    let rank: Vec<Option<usize>> = {
        let mut next = 0;
        (0..globals)
            .map(|g| {
                present(g).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect()
    };
    let ok = |id: u32| rank[id as usize].is_some_and(in_scope);
    let want = (0..m)
        .filter(|&e| ok(edges.src[e]) && ok(edges.dst[e]))
        .count() as u64;
    assert_eq!(fold.count(&world, &table, &prepared).unwrap(), want);
    assert!(want > 0 && want < m as u64);
}

#[test]
fn a_stale_frame_is_refused_and_a_fresh_preparation_answers() {
    let world = World::new();
    let people = People::new(70_000);
    let mut table = people.table();
    let mask = People::mask(&world);
    let prepared = prepare_nodes(&world, &table, &mask, present, in_scope).unwrap();
    let mut fold = Fold::count_gt(&prepared, person::AGE, 50).unwrap();
    let want = fold.count(&world, &table, &prepared).unwrap();
    assert!(want > 0);

    table.advance_source();
    assert!(matches!(
        fold.count(&world, &table, &prepared),
        Err(Refused::Lane(LaneRefusal::Frame(_)))
    ));
    let fresh = prepare_nodes(&world, &table, &mask, present, in_scope).unwrap();
    assert_eq!(fold.count(&world, &table, &fresh).unwrap(), want);
}

#[test]
fn a_plane_from_another_row_order_is_refused() {
    let world = World::new();
    let people = People::new(2 * SLAB_ROWS);
    let table = people.table();
    let mask = People::mask(&world);
    let mut prepared = prepare_nodes(&world, &table, &mask, present, in_scope).unwrap();
    let mut fold = Fold::count_gt(&prepared, person::AGE, 50).unwrap();
    assert!(fold.count(&world, &table, &prepared).unwrap() > 0);

    // Slab 1's scope plane has the same length as slab 0's and is in
    // another row order.
    let foreign = prepared.plane(1, 1).clone();
    assert_eq!(foreign.words.len(), prepared.plane(0, 1).words.len());
    prepared.set_plane(0, 1, foreign);
    assert_eq!(
        fold.count(&world, &table, &prepared),
        Err(Refused::Misaligned { slab: 0, plane: 1 })
    );
}

#[test]
fn one_mask_selects_storage_compute_and_display() {
    let world = World::new();
    let people = People::new(1_000);
    let mask = select(&world.view, PERSON, &["age", "person_id"]).unwrap();
    // Storage: exactly these columns, in basis order, whatever order was asked.
    assert_eq!(
        columns(&world.view, PERSON, &mask),
        vec![(person::ID, "person_id"), (person::AGE, "age")]
    );
    assert!(select(&world.view, PERSON, &["age", "salary"]).is_none());

    // Compute: the selected fields bind; `name` was not selected.
    let table = people.table();
    let prepared = prepare_nodes(&world, &table, &mask, |_| true, |_| true).unwrap();
    assert!(Fold::count_gt(&prepared, person::AGE, 50).is_ok());
    assert!(matches!(
        Fold::count_gt(&prepared, person::NAME, 50),
        Err(Refused::FieldNotSelected {
            field: person::NAME
        })
    ));

    // A selected field storage did not read cannot be bound.
    let only_id = Table::new(
        PERSON,
        1,
        1_000,
        1,
        READER,
        &[(person::ID, LaneRef::U32(&people.id))],
    );
    assert!(matches!(
        prepare_nodes(&world, &only_id, &mask, |_| true, |_| true),
        Err(Refused::Lane(LaneRefusal::Unbound {
            field: person::AGE,
            ..
        }))
    ));

    // Display: the same fields, same order.
    assert_eq!(
        display_row(&world.view, &table, &mask, 41).unwrap(),
        vec![("person_id".to_string(), 41), ("age".to_string(), 61)]
    );
    let age_only = select(&world.view, PERSON, &["age"]).unwrap();
    assert_eq!(
        display_row(&world.view, &table, &age_only, 41).unwrap(),
        vec![("age".to_string(), 61)]
    );
}

#[test]
fn one_reservation_holds_one_slab_and_no_more() {
    assert!(
        admits_reservation(256).is_ok(),
        "256 tiles of 256 is one slab"
    );
    assert!(matches!(
        admits_reservation(257),
        Err(Refusal::ReservationTooWide {
            axis: "identity",
            ..
        })
    ));
}

#[test]
fn an_fk_outside_the_node_domain_is_refused() {
    let world = World::new();
    let edges = Edges::hashed(1_000, 500);
    let table = edges.table();
    let inside = |id: u32| (id < 500).then_some(true);
    assert!(prepare_edges(&world, &table, |_| true, inside).is_ok());
    let narrower = |id: u32| (id < 400).then_some(true);
    assert!(matches!(
        prepare_edges(&world, &table, |_| true, narrower),
        Err(Refused::ForeignKeyOutOfRange { .. })
    ));
}

#[test]
fn a_million_rows_stack_as_sixteen_slabs() {
    let world = World::new();
    let n = 1_000_000usize;
    let people = People::new(n);
    let table = people.table();
    let prepared = prepare_nodes(&world, &table, &People::mask(&world), present, in_scope).unwrap();
    assert_eq!(prepared.slabs(), 16);
    // 15 full slabs of 1,024 words and one of 265 (16,960 rows), three planes each.
    assert_eq!(prepared.plane_bytes(), 3 * 8 * (15 * SLAB_WORDS + 265));
    let mut fold = Fold::count_gt(&prepared, person::AGE, 50).unwrap();
    let want = (0..n)
        .filter(|&i| present(i) && in_scope(i) && people.age[i] > 50)
        .count() as u64;
    assert_eq!(fold.count(&world, &table, &prepared).unwrap(), want);
}
