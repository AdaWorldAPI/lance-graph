//! Plan section 15: the virtual cube. A method (selection name, orientation)
//! is a constant; mapping and selection stay DR's. Fixture: the lab `Grid`,
//! whose frame and selections are its own and say nothing about the mailbox.

use lance_graph_dt64::LaneRefusal;
use virtual_address_lab::grid::{Grid, World};
use virtual_address_lab::method::{Cell, Method};
use virtual_address_lab::physical::{apply, Moved};
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::view::{Ledger, Refused};

const FIELDS: usize = 3;
const ROWS: usize = 64;
const KEY: u8 = 1;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Distinct in every cell: the field in the thousands, a scrambled row below.
fn value(lane: usize, pos: usize) -> i32 {
    (lane * 1_000) as i32 + (mix(pos as u64 + 7 * lane as u64) % 1_000) as i32
}

/// The method, composed at compile time.
const BY_ORDER: Method = Method::new(0, Recipe::TRANSPOSE);
const BY_ORDER_ROUND_TRIP: Method = BY_ORDER.then(Recipe::ROTATE90).then(Recipe::ROTATE270);

/// FAILS IF: a method holds more than a name and a recipe, composing it
/// needs a source, or inverse recipes do not cancel in it.
#[test]
fn a_method_is_a_constant() {
    assert_eq!(std::mem::size_of::<Method>(), 6);
    assert_eq!(BY_ORDER_ROUND_TRIP, BY_ORDER);
    assert_ne!(BY_ORDER.then(Recipe::ROTATE90), BY_ORDER);
    // The selection is not touched by orientation.
    for g in Recipe::ALL {
        assert_eq!(BY_ORDER.then(g).selection, 0);
    }
    // Equal methods have equal address functions over the whole tile; no
    // source exists in this test at all.
    let (mut cells, mut moved) = (0, 0);
    for r in 0..ROWS {
        for c in 0..FIELDS {
            let a = BY_ORDER.cell(FIELDS, ROWS, r, c);
            assert_eq!(BY_ORDER_ROUND_TRIP.cell(FIELDS, ROWS, r, c), a);
            assert_eq!(
                a,
                Some(Cell {
                    field: c,
                    ordinal: r
                })
            );
            let turned = Method::new(0, Recipe::ANTI_TRANSPOSE).cell(FIELDS, ROWS, r, c);
            assert_eq!(
                turned,
                Some(Cell {
                    field: FIELDS - 1 - c,
                    ordinal: ROWS - 1 - r
                })
            );
            moved += usize::from(turned != a);
            cells += 1;
        }
    }
    assert_eq!(cells, ROWS * FIELDS);
    // The middle field of three is the only one a flip leaves in place, and
    // no ordinal of 64 is.
    assert_eq!(moved, cells);
    assert_eq!(BY_ORDER.cell(FIELDS, ROWS, ROWS, 0), None);
}

/// The selected tile gathered into an OWNED matrix (the thing the method
/// never builds): `m[field][ordinal] = grid[field][positions[ordinal]]`.
fn gathered(grid: &Grid, positions: &[u32]) -> Vec<Vec<i32>> {
    (0..FIELDS)
        .map(|f| {
            positions
                .iter()
                .map(|&p| grid.cell(f, p as usize))
                .collect()
        })
        .collect()
}

/// FAILS IF: any cell of any orientation of any selection differs from the
/// gathered-and-moved matrix, a repeated position is not read twice, or the
/// positions and lanes lent are not the source's own.
#[test]
fn every_cell_of_the_cube_is_the_moved_matrix_cell() {
    let mut grid = Grid::new(FIELDS, ROWS, value);
    let world = World::new(FIELDS);
    let order = grid
        .issue_order(KEY)
        .expect("the source produces the order");
    let filter = grid.issue_filter(0, 200, 700);
    let bag = grid.issue_list(vec![5, 5, 63, 0, 5, 17]);
    let binding = world.bind(&grid).unwrap();
    let ledger = Ledger::default();
    let lane0 = grid.raw_lane(0).as_ptr();

    let mut cells = 0u64;
    for sel in [order, filter, bag] {
        let positions = grid.raw_positions(sel).to_vec();
        let n = positions.len();
        assert!(n > 1);
        let tile = gathered(&grid, &positions);
        for recipe in Recipe::ALL {
            let method = Method::new(sel, recipe);
            let mut moved = Moved::default();
            let built = apply(&tile, recipe, &mut moved);
            assert_eq!(
                (built.len(), built[0].len()),
                recipe.view_dims(FIELDS, n),
                "{}",
                recipe.name()
            );
            for (r, line) in built.iter().enumerate() {
                for (c, &want) in line.iter().enumerate() {
                    assert_eq!(
                        method.read(&world, &binding, &grid, r, c, &ledger),
                        Ok(want),
                        "sel {sel} {} cell ({r}, {c})",
                        recipe.name()
                    );
                    cells += 1;
                }
            }
            let (rows, cols) = recipe.view_dims(FIELDS, n);
            assert_eq!(
                method.read(&world, &binding, &grid, rows, 0, &ledger),
                Err(Refused::OutsideTheView)
            );
            assert_eq!(
                method.read(&world, &binding, &grid, 0, cols, &ledger),
                Err(Refused::OutsideTheView)
            );
        }
    }
    assert_eq!(ledger.reads.get(), cells);
    let selected = grid.raw_positions(filter).len() as u64;
    assert!(selected > 1 && selected < ROWS as u64, "a real subset");
    assert_eq!(cells, 8 * FIELDS as u64 * (ROWS as u64 + selected + 6));

    // The order is not the row order, and the bag repeats.
    let ordered = grid.raw_positions(order);
    assert!(ordered.iter().enumerate().any(|(i, &p)| p as usize != i));
    let m = Method::new(bag, Recipe::TRANSPOSE);
    let first = m.read(&world, &binding, &grid, 0, 2, &ledger).unwrap();
    assert_eq!(m.read(&world, &binding, &grid, 1, 2, &ledger), Ok(first));
    assert_eq!(first, grid.cell(2, 5));
    // Reversed traversal of the same bag: ordinal 0 is its last position.
    let back = Method::new(bag, Recipe::ROTATE90);
    assert_eq!(
        back.read(&world, &binding, &grid, 0, 0, &ledger),
        Ok(grid.cell(0, 17))
    );

    // Nothing moved and nothing was copied: the witness lends the source's
    // own positions and lane.
    let (lane, lent) = binding
        .selected(
            &world.store,
            &world.view,
            grid.frame_store(),
            &grid,
            0,
            order,
        )
        .unwrap();
    assert!(std::ptr::eq(
        lent.as_ptr(),
        grid.raw_positions(order).as_ptr()
    ));
    let lance_graph_mask_risc::LaneRef::I32(values) = lane else {
        panic!("an i32 lane")
    };
    assert!(std::ptr::eq(values.as_ptr(), lane0));
}

/// FAILS IF: the method, not the binding, decides which lane a field is.
#[test]
fn mapping_stays_the_bindings() {
    let mut plain = Grid::new(FIELDS, ROWS, value);
    let mut renumbered = Grid::new(FIELDS, ROWS, value).with_lane_ids(vec![40, 7, 19]);
    let world = World::new(FIELDS);
    let (a, b) = (
        plain.issue_order(KEY).unwrap(),
        renumbered.issue_order(KEY).unwrap(),
    );
    assert_eq!(a, b);
    let (bind_a, bind_b) = (
        world.bind(&plain).unwrap(),
        world.bind(&renumbered).unwrap(),
    );
    let ledger = Ledger::default();
    for recipe in [Recipe::TRANSPOSE, Recipe::ANTI_TRANSPOSE, Recipe::MIRROR_X] {
        let method = Method::new(a, recipe);
        let (rows, cols) = recipe.view_dims(FIELDS, ROWS);
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(
                    method.read(&world, &bind_b, &renumbered, r, c, &ledger),
                    method.read(&world, &bind_a, &plain, r, c, &ledger),
                );
            }
        }
    }
}

/// FAILS IF: a method reads through a binding sealed before the source moved
/// its frame. The refusal is the binding's; the method adds none.
#[test]
fn admission_is_the_bindings_not_the_methods() {
    let mut grid = Grid::new(FIELDS, ROWS, value);
    let world = World::new(FIELDS);
    let sel = grid.issue_order(KEY).unwrap();
    let binding = world.bind(&grid).unwrap();
    let ledger = Ledger::default();
    let method = Method::new(sel, Recipe::TRANSPOSE);
    assert!(method.read(&world, &binding, &grid, 3, 1, &ledger).is_ok());

    grid.advance_source();
    let refused = method.read(&world, &binding, &grid, 3, 1, &ledger);
    assert!(
        matches!(refused, Err(Refused::Lane(LaneRefusal::Frame(_)))),
        "{refused:?}"
    );
    // The method is the same constant before and after; only admission moved.
    assert_eq!(method, Method::new(sel, Recipe::TRANSPOSE));
    assert_eq!(ledger.reads.get(), 1);

    // An unissued selection name is refused by the binding too.
    let fresh = world.bind(&grid).unwrap();
    assert!(matches!(
        Method::new(99, Recipe::TRANSPOSE).read(&world, &fresh, &grid, 0, 0, &ledger),
        Err(Refused::Lane(LaneRefusal::SelectionUnbound { sel: 99 }))
    ));
}

/// Review finding. FAILS IF: a cell whose field does not fit the binding's
/// field width is narrowed onto another field and read.
#[test]
fn a_cell_field_past_the_basis_width_is_refused() {
    let mut grid = Grid::new(FIELDS, ROWS, value);
    let world = World::new(FIELDS);
    let sel = grid.issue_order(KEY).unwrap();
    let binding = world.bind(&grid).unwrap();
    let ledger = Ledger::default();
    let method = Method::new(sel, Recipe::TRANSPOSE);
    let cell = |field| Cell { field, ordinal: 0 };
    assert!(method
        .read_cell(&world, &binding, &grid, cell(0), &ledger)
        .is_ok());
    for field in [256, 257, 512, usize::MAX] {
        assert_eq!(
            method.read_cell(&world, &binding, &grid, cell(field), &ledger),
            Err(Refused::OutsideTheView),
            "field {field}"
        );
    }
    assert_eq!(ledger.reads.get(), 1);
}
