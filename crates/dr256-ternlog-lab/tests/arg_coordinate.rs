//! Where the extreme is: `Agg::ArgMin` / `Agg::ArgMax` over a slab stack.
//!
//! The answer is `(space, row, value)`. A row number names a position only
//! inside its coordinate space, so the space travels with it. Ties resolve
//! to the lowest slab, then the lowest row.
//!
//! The table, world, planes and folds are built the way `weather_fold.rs`
//! builds them (helpers copied, not imported). The value lane is synthetic.

use dr256_ternlog_lab::*;
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::{CoordSpace, LaneRefusal};
use lance_graph_mask_risc::LaneRef;

const READER: Scope = Scope { id: 7 };
const VALUE: u8 = person::AGE;
/// Three full slabs and a partial fourth.
const N: usize = 3 * SLAB_ROWS + 1_000;

type Coord = Option<(CoordSpace, usize, i32)>;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
fn region(i: usize) -> bool {
    !mix(i as u64 ^ 0x11).is_multiple_of(3)
}
fn valid(i: usize) -> bool {
    !mix(i as u64 ^ 0x22).is_multiple_of(50)
}
fn third(i: usize) -> bool {
    !mix(i as u64 ^ 0x44).is_multiple_of(5)
}

struct Data {
    key: Vec<u32>,
    value: Vec<i32>,
}

impl Data {
    /// Values in -20..=20: 41 distinct values over ~200,000 rows, so every
    /// extreme is tied many times, across slabs.
    fn tied(n: usize) -> Data {
        Data {
            key: (0..n as u64)
                .map(|i| (mix(i ^ 0x33) % 413) as u32)
                .collect(),
            value: (0..n as u64).map(|i| (mix(i) % 41) as i32 - 20).collect(),
        }
    }

    fn table(&self, population: u64) -> Table<'_> {
        Table::new(
            PERSON,
            population,
            self.value.len(),
            1,
            READER,
            &[
                (person::ID, LaneRef::U32(&self.key)),
                (VALUE, LaneRef::I32(&self.value)),
            ],
        )
    }
}

/// Bind the age field, take the region and validity planes, then write the
/// third plane (`prepare_nodes` leaves it all ones).
fn prepare(
    world: &World,
    table: &Table<'_>,
    region: &dyn Fn(usize) -> bool,
    valid: &dyn Fn(usize) -> bool,
    third: &dyn Fn(usize) -> bool,
) -> Prepared {
    let fields = select(&world.view, PERSON, &["age"]).unwrap();
    let mut prepared = prepare_nodes(world, table, &fields, |i| region(i), |i| valid(i)).unwrap();
    for s in 0..table.slabs() {
        let base = s * SLAB_ROWS;
        let plane = Plane::from_fn(table.space(s), table.slab_rows(s), |i| third(base + i));
        prepared.set_plane(s, 2, plane);
    }
    prepared
}

fn and3(world: &World, table: &Table<'_>, prepared: &Prepared, agg: Agg) -> Answer {
    Fold::and3(prepared, agg)
        .unwrap()
        .answer(world, table, prepared)
        .unwrap()
}

fn arg_min(a: Answer) -> Coord {
    match a {
        Answer::ArgMin(c) => c,
        other => panic!("not an ArgMin answer: {other:?}"),
    }
}

fn arg_max(a: Answer) -> Coord {
    match a {
        Answer::ArgMax(c) => c,
        other => panic!("not an ArgMax answer: {other:?}"),
    }
}

/// The scalar oracle: slabs ascending, rows ascending, a candidate is kept
/// only when strictly better. `keep(global row, value)` is the selection.
fn oracle(table: &Table<'_>, value: &[i32], max: bool, keep: &dyn Fn(usize, i32) -> bool) -> Coord {
    let mut best: Coord = None;
    for slab in 0..table.slabs() {
        for row in 0..table.slab_rows(slab) {
            let g = slab * SLAB_ROWS + row;
            let v = value[g];
            if !keep(g, v) {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, _, held)) => {
                    if max {
                        v > held
                    } else {
                        v < held
                    }
                }
            };
            if better {
                best = Some((table.space(slab), row, v));
            }
        }
    }
    best
}

/// FAILS IF: ArgMin or ArgMax disagrees with a scalar loop that visits slabs
/// then rows in ascending order and keeps only strictly better candidates, for
/// `and3` (all three planes) and for `between` (two planes and a range).
#[test]
fn arg_equals_a_scalar_oracle() {
    let data = Data::tied(N);
    let world = World::new();
    let table = data.table(1);
    assert_eq!(table.slabs(), 4);
    let prepared = prepare(&world, &table, &region, &valid, &third);

    let keep3 = |g: usize, _v: i32| region(g) && valid(g) && third(g);
    for (agg, max) in [(Agg::ArgMin(VALUE), false), (Agg::ArgMax(VALUE), true)] {
        let want = oracle(&table, &data.value, max, &keep3);
        let (_, _, extreme) = want.expect("the selection is not empty");
        let ties = (0..N)
            .filter(|&g| keep3(g, 0) && data.value[g] == extreme)
            .count();
        assert!(ties > 1, "the extreme must be tied for the rule to matter");
        let got = and3(&world, &table, &prepared, agg);
        let got = if max { arg_max(got) } else { arg_min(got) };
        assert_eq!(got, want, "{agg:?} over and3");
    }

    // `between` reads planes 0 and 1 and the range; plane 2 plays no part.
    let (lo, hi) = (-10, 10);
    let keep_between = |g: usize, v: i32| region(g) && valid(g) && lo < v && v < hi;
    for (agg, max) in [(Agg::ArgMin(VALUE), false), (Agg::ArgMax(VALUE), true)] {
        let want = oracle(&table, &data.value, max, &keep_between);
        assert!(want.is_some());
        let got = Fold::between(&prepared, VALUE, lo, hi, agg)
            .unwrap()
            .answer(&world, &table, &prepared)
            .unwrap();
        let got = if max { arg_max(got) } else { arg_min(got) };
        assert_eq!(got, want, "{agg:?} over between({lo}, {hi})");
    }
}

/// FAILS IF: the value inside ArgMin differs from `Answer::Min`, or the value
/// inside ArgMax differs from `Answer::Max`, for the same planes.
#[test]
fn the_value_agrees_with_min_and_max() {
    let data = Data::tied(N);
    let world = World::new();
    let table = data.table(1);
    let prepared = prepare(&world, &table, &region, &valid, &third);

    let lowest = arg_min(and3(&world, &table, &prepared, Agg::ArgMin(VALUE)));
    let highest = arg_max(and3(&world, &table, &prepared, Agg::ArgMax(VALUE)));
    assert!(lowest.is_some() && highest.is_some());
    assert_eq!(
        and3(&world, &table, &prepared, Agg::Min(VALUE)),
        Answer::Min(lowest.map(|(_, _, v)| v))
    );
    assert_eq!(
        and3(&world, &table, &prepared, Agg::Max(VALUE)),
        Answer::Max(highest.map(|(_, _, v)| v))
    );
}

/// FAILS IF: among equal extreme values the winner is not the lowest slab and
/// then the lowest row. One unique-extreme value is planted at four places.
#[test]
fn lowest_slab_then_lowest_row_wins() {
    const V: i32 = -100;
    let mut data = Data::tied(N);
    let planted = [(2, 77), (1, 900), (1, 40), (3, 5)];
    for (slab, row) in planted {
        data.value[slab * SLAB_ROWS + row] = V;
    }
    let at = |slab: usize, row: usize| slab * SLAB_ROWS + row;
    let world = World::new();
    let table = data.table(1);

    // The planes are rebuilt for each selection, as a plane change is.
    let run = |cleared: &[usize]| {
        let valid_but = |g: usize| !cleared.contains(&g);
        let prepared = prepare(&world, &table, &|_| true, &valid_but, &|_| true);
        assert_eq!(
            and3(&world, &table, &prepared, Agg::Min(VALUE)),
            Answer::Min(Some(V)),
            "the planted value is the minimum while any copy is selected"
        );
        arg_min(and3(&world, &table, &prepared, Agg::ArgMin(VALUE)))
    };

    assert_eq!(run(&[]), Some((table.space(1), 40, V)));
    assert_eq!(run(&[at(1, 40)]), Some((table.space(1), 900, V)));
    assert_eq!(run(&[at(1, 40), at(1, 900)]), Some((table.space(2), 77, V)));
    assert_eq!(
        run(&[at(1, 40), at(1, 900), at(2, 77)]),
        Some((table.space(3), 5, V))
    );
}

/// FAILS IF: an empty selection returns a coordinate instead of `None`.
#[test]
fn empty_selection_is_none() {
    let data = Data::tied(N);
    let world = World::new();
    let table = data.table(1);
    let prepared = prepare(&world, &table, &|_| false, &valid, &third);
    assert_eq!(prepared.plane(0, 0).count(), 0);
    assert_eq!(
        and3(&world, &table, &prepared, Agg::ArgMin(VALUE)),
        Answer::ArgMin(None)
    );
    assert_eq!(
        and3(&world, &table, &prepared, Agg::ArgMax(VALUE)),
        Answer::ArgMax(None)
    );
}

/// FAILS IF: the answer drops the coordinate space, so that row 123 of slab 0
/// and row 123 of slab 2 compare equal.
#[test]
fn the_same_row_number_in_two_slabs_is_two_different_coordinates() {
    const TOP: i32 = 1_000;
    let mut data = Data::tied(N);
    data.value[123] = TOP;
    data.value[2 * SLAB_ROWS + 123] = TOP;
    let world = World::new();
    let table = data.table(1);

    let limited_to = |slab: usize| {
        let in_slab = move |g: usize| g / SLAB_ROWS == slab;
        let prepared = prepare(&world, &table, &in_slab, &|_| true, &|_| true);
        arg_max(and3(&world, &table, &prepared, Agg::ArgMax(VALUE)))
    };
    let first = limited_to(0).expect("slab 0 selects rows");
    let second = limited_to(2).expect("slab 2 selects rows");

    assert_eq!(first.1, 123);
    assert_eq!(second.1, 123);
    assert_eq!(first.2, second.2);
    assert_ne!(first.0, second.0);
    assert_eq!(first.0, table.space(0));
    assert_eq!(second.0, table.space(2));
    // A row number is a coordinate only together with its space.
    assert_ne!(first, second);
}

/// FAILS IF: after the source moves, a fold under the old frame still returns
/// a coordinate instead of the refusal the other aggregates get.
///
/// `Table::advance_source` bumps `FrameRevisions::source` only. `Table::space`
/// reads `coordinates`, which it does not touch, so the `CoordSpace` of a slab
/// is unchanged by the move: the frame check, not the space, is what refuses.
#[test]
fn a_stale_frame_returns_no_coordinate() {
    let data = Data::tied(N);
    let world = World::new();
    let mut table = data.table(1);
    let prepared = prepare(&world, &table, &region, &valid, &third);
    let mut fold = Fold::and3(&prepared, Agg::ArgMax(VALUE)).unwrap();
    let before = arg_max(fold.answer(&world, &table, &prepared).unwrap());
    let (space_before, _, _) = before.expect("the selection is not empty");

    table.advance_source();
    assert!(
        matches!(
            fold.answer(&world, &table, &prepared),
            Err(Refused::Lane(LaneRefusal::Frame(_)))
        ),
        "ArgMax read a frame the source has moved past"
    );

    // Re-prepared under the new frame it answers again.
    let fresh = prepare(&world, &table, &region, &valid, &third);
    let after = arg_max(
        Fold::and3(&fresh, Agg::ArgMax(VALUE))
            .unwrap()
            .answer(&world, &table, &fresh)
            .unwrap(),
    );
    let (space_after, _, _) = after.expect("the selection is not empty");
    // The coordinates revision did not move, so the space is the same one.
    assert_eq!(space_after.coordinates, space_before.coordinates);
    assert_eq!(after, before);
}

/// FAILS IF: two tables with different populations but identical values and
/// planes produce equal answers, i.e. the population is left out of the space.
#[test]
fn two_tables_do_not_share_coordinates() {
    let data = Data::tied(N);
    let world = World::new();
    let one = data.table(1);
    let two = data.table(2);
    let p1 = prepare(&world, &one, &region, &valid, &third);
    let p2 = prepare(&world, &two, &region, &valid, &third);

    let a = arg_max(and3(&world, &one, &p1, Agg::ArgMax(VALUE))).unwrap();
    let b = arg_max(and3(&world, &two, &p2, Agg::ArgMax(VALUE))).unwrap();

    assert_eq!(a.1, b.1);
    assert_eq!(a.2, b.2);
    assert_ne!(a.0.population, b.0.population);
    assert_ne!(Some(a), Some(b));
}
