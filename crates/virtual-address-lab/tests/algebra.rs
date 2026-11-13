//! H1 to H4: the recipe algebra, parity with a reference that moves data,
//! cancellation before execution, and demand-driven reads.
//! Pass conditions: `.claude/plans/d-virtual-address-algebra-1.md` section 4.

use virtual_address_lab::grid::{Grid, World};
use virtual_address_lab::physical;
use virtual_address_lab::recipe::{Access, Recipe};
use virtual_address_lab::route::{plan_sum_where_eq, Stop};
use virtual_address_lab::view::{Trace, VirtualTable};

/// Pointwise composition, independent of `Recipe::then`: resolve through
/// `next`, then resolve THAT cell through `first`.
fn pointwise(first: Recipe, next: Recipe, n: usize, r: usize, c: usize) -> (usize, usize) {
    let (r1, c1) = next.resolve(n, n, r, c).unwrap();
    first.resolve(n, n, r1, c1).unwrap()
}

/// H1. FAILS IF: `then` differs from pointwise composition anywhere, the set
/// is not closed, an inverse is missing, or a required identity fails.
#[test]
fn the_eight_recipes_compose_as_their_addresses_do() {
    for &g in &Recipe::ALL {
        for &h in &Recipe::ALL {
            let gh = g.then(h);
            assert!(Recipe::ALL.contains(&gh));
            for n in 1..=8 {
                for r in 0..n {
                    for c in 0..n {
                        assert_eq!(
                            gh.resolve(n, n, r, c).unwrap(),
                            pointwise(g, h, n, r, c),
                            "{} then {} at n={n} ({r},{c})",
                            g.name(),
                            h.name()
                        );
                    }
                }
            }
        }
        assert_eq!(g.then(g.inverse()), Recipe::IDENTITY);
        assert_eq!(g.inverse().then(g), Recipe::IDENTITY);
    }
    let id = Recipe::IDENTITY;
    assert_eq!(Recipe::ROTATE90.then(Recipe::ROTATE270), id);
    assert_eq!(Recipe::ROTATE270.then(Recipe::ROTATE90), id);
    assert_eq!(Recipe::ROTATE180.then(Recipe::ROTATE180), id);
    assert_eq!(Recipe::MIRROR_X.then(Recipe::MIRROR_X), id);
    assert_eq!(Recipe::MIRROR_Y.then(Recipe::MIRROR_Y), id);
    assert_eq!(Recipe::ROTATE90.then(Recipe::ROTATE90), Recipe::ROTATE180);
    // The group is not commutative: the table is not symmetric.
    assert_ne!(
        Recipe::ROTATE90.then(Recipe::MIRROR_X),
        Recipe::MIRROR_X.then(Recipe::ROTATE90)
    );
    // Eight distinct address maps, not eight names for fewer.
    for (i, &g) in Recipe::ALL.iter().enumerate() {
        for &h in &Recipe::ALL[..i] {
            assert!((0..9).any(|k| g.resolve(3, 3, k / 3, k % 3) != h.resolve(3, 3, k / 3, k % 3)));
        }
    }
}

fn flat(m: &physical::Matrix) -> Vec<i32> {
    m.iter().flatten().copied().collect()
}

/// H2. FAILS IF: a recipe, or a chain of two, reads another value than the
/// reference that builds the view by moving data.
#[test]
fn a_resolved_recipe_reads_what_the_moved_matrix_holds() {
    let grid = Grid::three();
    let world = World::new(3);
    let binding = world.bind(&grid).unwrap();
    let trace = Trace::default();
    let base = VirtualTable::bound(&world, &binding, &grid, Recipe::IDENTITY, &trace).unwrap();
    assert_eq!(
        base.through(Recipe::ROTATE90).materialize(),
        vec![3, 6, 9, 2, 5, 8, 1, 4, 7]
    );

    // A 5 x 5 tile with every cell distinct, all 8 and all 64 chains.
    let grid = Grid::new(5, 5, |l, p| (l * 100 + p) as i32);
    let world = World::new(5);
    let binding = world.bind(&grid).unwrap();
    let base = VirtualTable::bound(&world, &binding, &grid, Recipe::IDENTITY, &trace).unwrap();
    let scrap = Trace::default();
    let m = physical::of(&grid, &scrap);
    let mut distinct = std::collections::BTreeSet::new();
    for &g in &Recipe::ALL {
        let want = flat(&physical::apply(&m, g, &scrap));
        assert_eq!(base.through(g).materialize(), want, "{}", g.name());
        distinct.insert(want);
        for &h in &Recipe::ALL {
            assert_eq!(
                base.through(g).through(h).materialize(),
                flat(&physical::apply_chain(&m, &[g, h], &scrap)),
                "{} then {}",
                g.name(),
                h.name()
            );
        }
    }
    assert_eq!(distinct.len(), 8, "the eight views are eight arrangements");
}

fn entity_grid() -> Grid {
    // 32 lanes, each recognisable: lane l, position p holds l * 10_000 + p.
    Grid::new(32, 4_000, |l, p| (l * 10_000 + p) as i32)
}

/// H3, the primary falsifier. FAILS IF: a chain that normalises to the
/// identity runs another program than the unchained view, or its route
/// transforms, gathers or copies anything. The reference that does not
/// normalise is shown to move data for the same chain.
#[test]
fn a_chain_that_cancels_executes_nothing_but_the_fold() {
    let grid = entity_grid();
    let world = World::new(32);
    let binding = world.bind(&grid).unwrap();
    let address = grid.buffer_address();

    let chains: [&[Recipe]; 4] = [
        &[Recipe::ROTATE90, Recipe::ROTATE270],
        &[Recipe::ROTATE180, Recipe::ROTATE180],
        &[Recipe::MIRROR_X, Recipe::MIRROR_X],
        &[
            Recipe::ROTATE90,
            Recipe::ROTATE90,
            Recipe::ROTATE90,
            Recipe::ROTATE90,
        ],
    ];
    // Lane 7 holds 70_000 + p; "where lane 7 = 70_123" selects one row.
    let (sum_col, where_col, v) = (5usize, 7usize, 70_123);
    let want: i64 = (0..grid.len())
        .filter(|&p| grid.cell(where_col, p) == v)
        .map(|p| i64::from(grid.cell(sum_col, p)))
        .sum();
    assert_eq!(want, 50_123);

    let direct_trace = Trace::default();
    let entity =
        VirtualTable::bound(&world, &binding, &grid, Recipe::TRANSPOSE, &direct_trace).unwrap();
    let mut direct = plan_sum_where_eq(&entity, sum_col, where_col, v).unwrap();
    assert_eq!(direct.sum(&entity), Ok(want));

    for chain in chains {
        let normal = Recipe::normalize(chain);
        assert_eq!(normal, Recipe::IDENTITY, "{chain:?}");
        let trace = Trace::default();
        let table = VirtualTable::bound(&world, &binding, &grid, Recipe::TRANSPOSE, &trace)
            .unwrap()
            .through(normal);
        let mut plan = plan_sum_where_eq(&table, sum_col, where_col, v).unwrap();
        assert_eq!(plan.program, direct.program, "the same program");
        assert!(!plan.reversed);
        assert_eq!(plan.sum(&table), Ok(want));
        // resolutions = 2 (two columns), reads = gathers = transforms = copied = 0, one fold.
        assert_eq!(trace.snapshot(), [2, 0, 0, 0, 0, 1], "{chain:?}");

        // The reference that applies every step: same answer, data moved.
        let moved = Trace::default();
        let m = physical::apply(&physical::of(&grid, &moved), Recipe::TRANSPOSE, &moved);
        let out = physical::apply_chain(&m, chain, &moved);
        let got: i64 = out
            .iter()
            .filter(|row| row[where_col] == v)
            .map(|row| i64::from(row[sum_col]))
            .sum();
        assert_eq!(got, want);
        assert!(moved.transforms.get() as usize >= chain.len());
        assert!(moved.copied.get() as usize >= chain.len() * 32 * 4_000);
    }
    assert_eq!(grid.buffer_address(), address);
}

/// H4. FAILS IF: forming a row object reads a value, or a demanded property
/// costs more than one resolution and one read, or returns another lane's
/// value.
#[test]
fn a_row_object_reads_only_the_property_that_is_demanded() {
    let grid = entity_grid();
    let world = World::new(32);
    let binding = world.bind(&grid).unwrap();
    let trace = Trace::default();
    let table = VirtualTable::bound(&world, &binding, &grid, Recipe::TRANSPOSE, &trace).unwrap();
    assert_eq!(table.dims(), (4_000, 32));

    let rows: Vec<_> = (0..4_000).map(|r| table.row(r).unwrap()).collect();
    assert_eq!(rows.len(), 4_000);
    assert_eq!(rows[0].properties(), 32);
    assert_eq!(
        trace.snapshot(),
        [0; 6],
        "4,000 objects of 32 properties, nothing read"
    );

    assert_eq!(rows[1_234].get(17), Some(171_234));
    assert_eq!(trace.snapshot(), [1, 1, 0, 0, 0, 0]);
    assert_eq!(rows[3_999].get(0), Some(3_999));
    assert_eq!(rows[0].get(31), Some(310_000));
    assert_eq!(trace.snapshot(), [3, 3, 0, 0, 0, 0]);
    assert_eq!(rows[0].get(32), None);
    assert!(table.row(4_000).is_none());

    // The brief's Rotate90 over the same lanes: row r is entity N-1-r.
    let rotated = VirtualTable::bound(&world, &binding, &grid, Recipe::ROTATE90, &trace).unwrap();
    assert_eq!(rotated.row(0).unwrap().get(17), Some(173_999));
    assert_eq!(rotated.row(3_999).unwrap().get(17), Some(170_000));
}

/// Recorded. FAILS IF: a lane-crossing column turns out to be a bound
/// operand after all (the plan's first forced-materialisation point would be
/// wrong), or the cell-by-cell route disagrees with the oracle.
#[test]
fn a_column_that_crosses_lanes_is_no_bound_operand() {
    let grid = Grid::new(64, 64, |l, p| ((l * 7 + p * 3) % 11) as i32);
    let world = World::new(64);
    let binding = world.bind(&grid).unwrap();
    let trace = Trace::default();
    // The lane matrix itself: view column c is position c of every lane.
    let table = VirtualTable::bound(&world, &binding, &grid, Recipe::IDENTITY, &trace).unwrap();
    assert_eq!(
        table.column(9).unwrap().access,
        Access::AcrossLanes { pos: 9 }
    );
    assert_eq!(
        plan_sum_where_eq(&table, 9, 20, 4).err(),
        Some(Stop::CrossLane)
    );

    let want: i64 = (0..64)
        .filter(|&l| grid.cell(l, 20) == 4)
        .map(|l| i64::from(grid.cell(l, 9)))
        .sum();
    assert!(want > 0);
    let before = trace.reads.get();
    assert_eq!(
        virtual_address_lab::route::sum_where_eq_cell_by_cell(&table, 9, 20, 4),
        Some(want)
    );
    assert!(
        trace.reads.get() - before >= 64,
        "one read per lane at least"
    );
    assert_eq!(trace.copied.get(), 0);

    // The same demand as ONE program over the source's slab: unadmitted.
    let mut across = virtual_address_lab::route::AcrossPlan::sum_where_eq(&grid, 4).unwrap();
    assert_eq!(across.run(&grid, 9, 20), Ok(want));
}
