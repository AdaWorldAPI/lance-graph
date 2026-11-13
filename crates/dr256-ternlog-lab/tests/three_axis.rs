//! Falsifiers that need no new lab API: the AND3 count on edge populations,
//! and whether three axis masks stand for a three-axis relation.
//! Pass conditions: `.claude/board/entries/2026-11-13-weather-dr256-ternlog.md`
//! H1 and H2.

use dr256_ternlog_lab::*;
use lance_graph_dt64::dr::Scope;
use lance_graph_mask_risc::{ExecError, LaneRef};

const READER: Scope = Scope { id: 7 };
/// One ERA5 0.25 degree timestep: 721 x 1440 cells.
const GRID: usize = 721 * 1440;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct Rows {
    id: Vec<u32>,
}
impl Rows {
    fn new(n: usize) -> Rows {
        Rows {
            id: (0..n as u32).collect(),
        }
    }
    fn table(&self) -> Table<'_> {
        Table::new(
            PERSON,
            1,
            self.id.len(),
            1,
            READER,
            &[(person::ID, LaneRef::U32(&self.id))],
        )
    }
}

/// Count `a & b & c` through the slab fold and through a row loop.
fn and3(
    n: usize,
    a: impl Fn(usize) -> bool,
    b: impl Fn(usize) -> bool,
    c: impl Fn(usize) -> bool,
) -> (u64, u64) {
    let world = World::new();
    let rows = Rows::new(n);
    let table = rows.table();
    let fields = select(&world.view, PERSON, &["person_id"]).unwrap();
    let mut prepared = prepare_nodes(&world, &table, &fields, &a, &b).unwrap();
    for s in 0..prepared.slabs() {
        let base = s * SLAB_ROWS;
        let plane = Plane::from_fn(table.space(s), table.slab_rows(s), |i| c(base + i));
        prepared.set_plane(s, 2, plane);
    }
    let got = Fold::count_and3()
        .unwrap()
        .count(&world, &table, &prepared)
        .unwrap();
    let want = (0..n).filter(|&i| a(i) && b(i) && c(i)).count() as u64;
    (got, want)
}

#[test]
fn a_weather_timestep_is_sixteen_slabs_with_a_short_last_one() {
    let rows = Rows::new(GRID);
    let table = rows.table();
    assert_eq!(GRID, 1_038_240);
    assert_eq!(table.slabs(), 16);
    assert_eq!(16 * SLAB_ROWS - GRID, 10_336, "inactive positions");
    assert_eq!(table.slab_rows(15), 55_200);
    assert_ne!(table.slab_rows(15) % 64, 0, "the last word is short too");
}

#[test]
fn and3_equals_the_row_loop_on_empty_full_sparse_dense_and_ragged() {
    let yes = |_: usize| true;
    let no = |_: usize| false;
    let sparse = |i: usize| mix(i as u64).is_multiple_of(997);
    let dense = |i: usize| !mix(i as u64 ^ 0x51).is_multiple_of(997);
    let half = |i: usize| mix(i as u64 ^ 0xC3) & 1 == 0;

    assert_eq!(and3(GRID, yes, yes, no), (0, 0), "empty");
    assert_eq!(
        and3(GRID, yes, yes, yes),
        (GRID as u64, GRID as u64),
        "full"
    );
    for n in [1, 63, 64, 65, SLAB_ROWS - 1, SLAB_ROWS, SLAB_ROWS + 1, GRID] {
        let (got, want) = and3(n, sparse, yes, half);
        assert_eq!(got, want, "sparse, n = {n}");
        let (got, want) = and3(n, dense, half, yes);
        assert_eq!(got, want, "dense, n = {n}");
    }
    let (got, want) = and3(GRID, sparse, dense, half);
    assert!(want > 0 && want < GRID as u64 / 1000);
    assert_eq!(got, want);
}

#[test]
fn a_plane_with_a_bit_past_the_slabs_rows_is_refused() {
    let world = World::new();
    let rows = Rows::new(GRID);
    let table = rows.table();
    let fields = select(&world.view, PERSON, &["person_id"]).unwrap();
    let mut prepared = prepare_nodes(&world, &table, &fields, |_| true, |_| true).unwrap();
    let mut fold = Fold::count_and3().unwrap();
    assert_eq!(fold.count(&world, &table, &prepared).unwrap(), GRID as u64);

    let last = prepared.slabs() - 1;
    let mut dirty = prepared.plane(last, 2).clone();
    let dead = table.slab_rows(last); // first position past the grid
    dirty.words[dead / 64] |= 1 << (dead % 64);
    prepared.set_plane(last, 2, dirty);
    assert!(
        matches!(
            fold.count(&world, &table, &prepared),
            Err(Refused::Exec(ExecError::PlaneTail(2)))
        ),
        "a dead position must not be counted"
    );
}

// ---------------------------------------------------------- three axes
//
// The product X x Y x Z of three 256-position axes, stacked as 256 slabs:
// slab = z, position in slab = x * 256 + y. Nothing of size 256^3 bytes is
// built; a plane over the product is 2 MiB.

const N: usize = 256;

fn xyz(row: usize) -> (usize, usize, usize) {
    (row / N % N, row % N, row / (N * N))
}

/// AND3 over three planes of the whole product, each from a per-row rule.
fn product_count(
    a: impl Fn(usize) -> bool,
    b: impl Fn(usize) -> bool,
    c: impl Fn(usize) -> bool,
) -> (u64, usize) {
    let world = World::new();
    let rows = Rows::new(N * N * N);
    let table = rows.table();
    assert_eq!(table.slabs(), MAX_SLABS);
    let fields = select(&world.view, PERSON, &["person_id"]).unwrap();
    let mut prepared = prepare_nodes(&world, &table, &fields, &a, &b).unwrap();
    for s in 0..prepared.slabs() {
        let base = s * SLAB_ROWS;
        prepared.set_plane(
            s,
            2,
            Plane::from_fn(table.space(s), SLAB_ROWS, |i| c(base + i)),
        );
    }
    let count = Fold::count_and3()
        .unwrap()
        .count(&world, &table, &prepared)
        .unwrap();
    (count, prepared.plane_bytes())
}

#[test]
fn a_factorized_selection_counts_the_product_of_its_popcounts() {
    let on = |seed: u64| move |v: usize| mix(v as u64 ^ seed).is_multiple_of(16);
    let (ax, by, cz) = (on(1), on(2), on(3));
    let pop = |f: &dyn Fn(usize) -> bool| (0..N).filter(|&v| f(v)).count() as u64;
    let want = pop(&ax) * pop(&by) * pop(&cz);
    assert!(want > 0 && want < (N * N * N) as u64 / 100);
    let (got, bytes) = product_count(|r| ax(xyz(r).0), |r| by(xyz(r).1), |r| cz(xyz(r).2));
    assert_eq!(got, want);
    assert_eq!(
        bytes,
        3 * 2 * 1024 * 1024,
        "three planes of the product: 6 MiB"
    );
}

#[test]
fn three_axis_masks_do_not_stand_for_a_three_axis_relation() {
    let in_r = |r: usize| {
        let (x, y, z) = xyz(r);
        (x + y + z) % N == 0
    };
    // Independent oracle: for each (x, y) exactly one z.
    let mut oracle = 0u64;
    let (mut px, mut py, mut pz) = ([false; N], [false; N], [false; N]);
    for x in 0..N {
        for y in 0..N {
            let z = (2 * N - x - y) % N;
            assert_eq!((x + y + z) % N, 0);
            oracle += 1;
            (px[x], py[y], pz[z]) = (true, true, true);
        }
    }
    assert_eq!(oracle, 65_536);
    assert!(
        px.iter().chain(&py).chain(&pz).all(|&p| p),
        "every axis projection is full"
    );

    // The tightest three axis masks, folded by the real kernel.
    let (from_axes, _) = product_count(|r| px[xyz(r).0], |r| py[xyz(r).1], |r| pz[xyz(r).2]);
    assert_eq!(from_axes, 16_777_216);
    assert_ne!(
        from_axes, oracle,
        "REFUTED: axis masks answer 256x too many"
    );

    // The relation as one explicit plane over the product.
    let (explicit, bytes) = product_count(in_r, |_| true, |_| true);
    assert_eq!(explicit, oracle);
    assert_eq!(bytes / 3, 2 * 1024 * 1024, "the relation plane is 2 MiB");
}
