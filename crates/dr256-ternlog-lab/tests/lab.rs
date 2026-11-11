//! Every fold is compared with a row-at-a-time count that touches neither
//! DT64 nor mask-RISC. Each refusal has a test that fires and a twin that
//! stays silent on a non-trivial input.

use dr256_ternlog_lab::*;
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::{CoordSpace, LaneRefusal, Refusal};
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
    !mix(i as u64).is_multiple_of(4)
}
/// 7 of 8 rows are in the reader's scope.
fn in_scope(i: usize) -> bool {
    !mix(i as u64 ^ 0xA5A5).is_multiple_of(8)
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
    let edge_live = |e: usize| !mix(e as u64 ^ 0x77).is_multiple_of(16);
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

#[test]
fn a_key_range_over_an_order_index_counts_without_a_selection_mask() {
    let world = World::new();
    for n in [100usize, 70_000, 200_000] {
        let people = People::new(n);
        let table = people.table();
        let mask = People::mask(&world);
        let index = OrderIndex::by_i32(&table, person::AGE, 99).unwrap();
        // The index is a permutation of the rows, ascending in the key.
        let mut seen = vec![false; n];
        for at in 0..n {
            let r = index.row(at).unwrap() as usize;
            assert!(!seen[r]);
            seen[r] = true;
            if at > 0 {
                let prev = index.row(at - 1).unwrap() as usize;
                assert!(people.age[prev] <= people.age[r]);
            }
        }
        let visible = |r: usize| present(r) && in_scope(r);
        let mut prepared = prepare_order(&world, &table, &mask, &index, visible).unwrap();

        let want = (0..n).filter(|&i| visible(i) && people.age[i] > 50).count() as u64;
        let got = prepared
            .count(&world, &table, &index, index.gt(50))
            .unwrap();
        assert_eq!(got, want, "n={n}");
        assert!(
            want > 0 && want < index.gt(50).len() as u64,
            "the plane removes rows"
        );

        let band = (0..n)
            .filter(|&i| visible(i) && (30..=40).contains(&people.age[i]))
            .count() as u64;
        assert_eq!(
            prepared
                .count(&world, &table, &index, index.between(30, 40))
                .unwrap(),
            band
        );
        assert!(band > 0 && band < want);
        assert_eq!(
            prepared
                .count(&world, &table, &index, index.gt(1_000))
                .unwrap(),
            0
        );
        let everything = (0..n).filter(|&i| visible(i)).count() as u64;
        assert_eq!(
            prepared.count(&world, &table, &index, 0..n).unwrap(),
            everything
        );
    }
}

#[test]
fn an_order_index_refuses_a_stale_frame_and_a_plane_in_table_order() {
    let world = World::new();
    let people = People::new(70_000);
    let mut table = people.table();
    let mask = People::mask(&world);
    let index = OrderIndex::by_i32(&table, person::AGE, 99).unwrap();
    // An index may not claim the table's own row order.
    assert!(OrderIndex::by_i32(&table, person::AGE, 1).is_none());
    assert!(
        OrderIndex::by_i32(&table, person::ID, 99).is_none(),
        "not an i32 key"
    );

    let mut prepared = prepare_order(&world, &table, &mask, &index, present).unwrap();
    assert!(
        prepared
            .count(&world, &table, &index, index.gt(50))
            .unwrap()
            > 0
    );

    // A plane in table order has the same length and another row order.
    let in_table_order = prepare_nodes(&world, &table, &mask, present, in_scope).unwrap();
    let foreign = in_table_order.plane(0, 0).clone();
    assert_eq!(foreign.words.len(), prepared.plane(0).words.len());
    let own = prepared.plane(0).clone();
    prepared.set_plane(0, foreign);
    assert_eq!(
        prepared.count(&world, &table, &index, index.gt(20)),
        Err(Refused::Misaligned { slab: 0, plane: 0 })
    );
    prepared.set_plane(0, own);

    table.advance_source();
    assert!(matches!(
        prepared.count(&world, &table, &index, index.gt(50)),
        Err(Refused::Lane(LaneRefusal::Frame(_)))
    ));
}

#[test]
fn a_hop_with_a_filter_on_the_source_is_one_edge_range_of_an_order_index() {
    let world = World::new();
    let n = 70_000usize;
    let m = 180_000usize;
    let people = People::new(n);
    let nodes = people.table();
    let edges = Edges::hashed(m, n as u64);
    let edge_table = edges.table();

    // Nodes in age order; edges in the order of their source's position.
    // No adjacency structure is built: `csr` is an ordinary OrderIndex, and
    // "CSR" is only the mental model for what its key makes contiguous.
    let by_age = OrderIndex::by_i32(&nodes, person::AGE, 99).unwrap();
    let at = by_age.positions();
    let csr = OrderIndex::by_key(&edge_table, 98, |e| at[edges.src[e] as usize] as i32).unwrap();

    let node_visible = |r: usize| present(r) && in_scope(r);
    let edge_live = |e: usize| !mix(e as u64 ^ 0x77).is_multiple_of(16);
    let edge_mask = WideFieldMask::from_positions(&[friend::SRC, friend::DST]);
    let mut prepared = prepare_order(&world, &edge_table, &edge_mask, &csr, |e| {
        edge_live(e) && node_visible(edges.src[e] as usize) && node_visible(edges.dst[e] as usize)
    })
    .unwrap();

    // `MATCH (a)-[]->(b) WHERE a.age > t RETURN count(*)`: the node range
    // of the age index maps to one range of the edge index.
    for t in [20, 50, 70, 1_000] {
        let node_range = by_age.gt(t);
        let edge_range = if node_range.is_empty() {
            0..0
        } else {
            csr.between(node_range.start as i32, node_range.end as i32 - 1)
        };
        let want = (0..m)
            .filter(|&e| {
                edge_live(e)
                    && people.age[edges.src[e] as usize] > t
                    && node_visible(edges.src[e] as usize)
                    && node_visible(edges.dst[e] as usize)
            })
            .count() as u64;
        assert_eq!(
            prepared
                .count(&world, &edge_table, &csr, edge_range)
                .unwrap(),
            want,
            "t={t}"
        );
    }
    // The filter removes edges, and the unfiltered hop is the whole index.
    let all = prepared.count(&world, &edge_table, &csr, 0..m).unwrap();
    let some = prepared
        .count(
            &world,
            &edge_table,
            &csr,
            csr.between(by_age.gt(50).start as i32, n as i32 - 1),
        )
        .unwrap();
    assert!(some > 0 && some < all);
}

/// A fold keeps lane indices resolved in one field selection. Run against a
/// preparation that selects other fields, lane 0 would be another column.
#[test]
fn a_fold_refuses_a_preparation_with_another_field_selection() {
    let world = World::new();
    let people = People::new(1_000);
    let table = people.table();
    let age_only = select(&world.view, PERSON, &["age"]).unwrap();
    let both = People::mask(&world);
    assert!(age_only != both);
    let narrow = prepare_nodes(&world, &table, &age_only, present, in_scope).unwrap();
    let wide = prepare_nodes(&world, &table, &both, present, in_scope).unwrap();
    let mut fold = Fold::count_gt(&narrow, person::AGE, 50).unwrap();
    let want = (0..1_000)
        .filter(|&i| present(i) && in_scope(i) && people.age[i] > 50)
        .count() as u64;
    assert!(want > 0);
    assert_eq!(fold.count(&world, &table, &narrow).unwrap(), want);
    assert!(matches!(
        fold.count(&world, &table, &wide),
        Err(Refused::SelectionChanged)
    ));
    // A fold that reads no lane has no selection to disagree with.
    let mut and3 = Fold::count_and3().unwrap();
    assert!(and3.count(&world, &table, &wide).is_ok());
}

/// An index holds one table's keys. Prepared against a replacement table it
/// would answer from the old keys.
#[test]
fn an_order_index_refuses_a_table_it_was_not_built_over() {
    let world = World::new();
    let mask = People::mask(&world);
    let id = [0u32, 1];
    let old_age = [0i32, 100];
    let new_age = [100i32, 0];
    let table = |age: &'static [i32; 2], rev: u64| -> Table<'static> {
        // Leaked so both tables can outlive this closure; two tiny arrays.
        let id: &'static [u32; 2] = Box::leak(Box::new(id));
        Table::new(
            PERSON,
            1,
            2,
            rev,
            READER,
            &[
                (person::ID, LaneRef::U32(id)),
                (person::AGE, LaneRef::I32(age)),
            ],
        )
    };
    let old: &'static [i32; 2] = Box::leak(Box::new(old_age));
    let new: &'static [i32; 2] = Box::leak(Box::new(new_age));
    let before = table(old, 1);
    let after = table(new, 2);
    let index = OrderIndex::by_i32(&before, person::AGE, 99).unwrap();
    // On its own table: only row 0 visible, its age is 0, nothing is > 50.
    let mut own = prepare_order(&world, &before, &mask, &index, |r| r == 0).unwrap();
    assert_eq!(own.count(&world, &before, &index, index.gt(50)).unwrap(), 0);
    // The replacement's row 0 has age 100. The old keys would still say 0.
    assert!(matches!(
        prepare_order(&world, &after, &mask, &index, |r| r == 0),
        Err(Refused::IndexOfAnotherTable)
    ));
    let fresh = OrderIndex::by_i32(&after, person::AGE, 99).unwrap();
    let mut right = prepare_order(&world, &after, &mask, &fresh, |r| r == 0).unwrap();
    assert_eq!(
        right.count(&world, &after, &fresh, fresh.gt(50)).unwrap(),
        1
    );
}

#[test]
fn a_plane_reads_as_256_selections_of_256_in_place() {
    let world = World::new();
    let people = People::new(70_000);
    let table = people.table();
    let prepared = prepare_nodes(&world, &table, &People::mask(&world), present, in_scope).unwrap();
    let preds: [fn(usize) -> bool; 2] = [present, in_scope];
    for s in 0..2usize {
        for (p, pred) in preds.iter().enumerate() {
            let plane = prepared.plane(s, p);
            assert_eq!(plane.masks(), table.slab_rows(s).div_ceil(256));
            assert_eq!(plane.masks(), [256, 18][s]);
            for y in 0..plane.masks() {
                let sel = plane.mask(y).unwrap();
                assert_eq!(sel.row() as usize, y);
                assert_eq!(sel.space(), table.space(s));
                assert!(std::ptr::eq(
                    sel.words().as_ptr(),
                    plane.words[y * MASK_WORDS..].as_ptr()
                ));
                for x in 0..=255u8 {
                    let i = y * 256 + x as usize;
                    let want = i < table.slab_rows(s) && pred(s * SLAB_ROWS + i);
                    assert_eq!(sel.has(x), want, "s={s} p={p} y={y} x={x}");
                }
            }
            let sum: u64 = plane.selections().map(|m| u64::from(m.count())).sum();
            assert_eq!(sum, plane.count());
            assert!(sum > 0 && sum < table.slab_rows(s) as u64);
        }
    }
}

#[test]
fn a_selection_past_the_plane_is_absent_and_the_last_one_is_short() {
    let space = CoordSpace {
        population: 7,
        coordinates: 1,
    };
    let two = Plane::from_fn(space, 512, present);
    assert_eq!(two.masks(), 2);
    assert_eq!(two.mask(1).unwrap().words().len(), 4);
    assert!(two.mask(2).is_none());
    assert_eq!(two.selections().count(), 2);

    let short = Plane::from_fn(space, 4_464, present);
    assert_eq!(short.words.len(), 70);
    assert_eq!(short.masks(), 18);
    let last = short.mask(17).unwrap();
    assert_eq!(last.words().len(), 2);
    let want = (4_352..4_464).filter(|&i| present(i)).count() as u32;
    assert!(want > 0);
    assert_eq!(last.count(), want);
    assert!(!last.has(200));
    assert!(short.mask(18).is_none());

    let full = Plane {
        space,
        words: vec![u64::MAX; 1_028],
    };
    assert_eq!(full.mask(255).unwrap().count(), 256);
    assert!(full.mask(256).is_none());
}

#[test]
fn a_selection_equals_the_wide_field_mask_of_the_same_positions() {
    let space = CoordSpace {
        population: 7,
        coordinates: 1,
    };
    let rule = |y: usize, x: usize| match y {
        0 => x < 64 && x.is_multiple_of(3),
        1 => x.is_multiple_of(5),
        2 => x >= 200,
        _ => false,
    };
    let plane = Plane::from_fn(space, 1_024, |i| rule(i / 256, i % 256));
    let positions = |y: usize| -> Vec<u8> {
        (0..256usize)
            .filter(|&x| rule(y, x))
            .map(|x| x as u8)
            .collect()
    };
    let masks: Vec<WideFieldMask> = (0..4)
        .map(|y| WideFieldMask::from_positions(&positions(y)))
        .collect();
    for (y, mask) in masks.iter().enumerate() {
        let sel = plane.mask(y).unwrap();
        assert!(sel.same_bits(mask), "row {y}");
        assert_eq!(sel.count(), mask.count(), "row {y}");
    }
    let r0 = plane.mask(0).unwrap();
    assert_eq!(masks[0].words().len(), 1);
    assert_eq!(r0.words().len(), 4);
    assert_eq!(plane.mask(1).unwrap().words(), masks[1].words());
    let r3 = plane.mask(3).unwrap();
    assert!(r3.is_empty());
    assert!(r3.same_bits(&WideFieldMask::EMPTY));

    assert!(!r0.same_bits(&masks[0].clone().with(100)));
    let p1 = positions(1);
    assert!(!plane
        .mask(1)
        .unwrap()
        .same_bits(&WideFieldMask::from_positions(&p1[1..])));
    assert!(!plane.mask(2).unwrap().same_bits(&WideFieldMask::EMPTY));
    assert!(!r3.same_bits(&WideFieldMask::EMPTY.with(0)));
}

#[test]
fn selections_carry_the_coordinate_space_of_their_plane() {
    let world = World::new();
    let people = People::new(2 * SLAB_ROWS);
    let table = people.table();
    let mask = People::mask(&world);
    let prepared = prepare_nodes(&world, &table, &mask, present, in_scope).unwrap();
    let index = OrderIndex::by_i32(&table, person::AGE, 99).unwrap();
    let order = prepare_order(&world, &table, &mask, &index, present).unwrap();

    let a = prepared.plane(0, 0).mask(3).unwrap();
    let b = prepared.plane(1, 0).mask(3).unwrap();
    assert_eq!(a.row(), b.row());
    assert_eq!(a.space(), table.space(0));
    assert_eq!(b.space(), table.space(1));
    assert_ne!(a.space(), b.space());
    assert_eq!(prepared.plane(0, 1).mask(3).unwrap().space(), a.space());

    let o = order.plane(0).mask(3).unwrap();
    assert_ne!(o.space(), a.space());
    assert_eq!(o.space().population, 99 << 8);
}
