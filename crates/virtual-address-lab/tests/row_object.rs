//! P2: a row of a virtual table is a descriptor; a value is read when, and
//! only when, a property is demanded.

use lance_graph_dt64::dr::FrameRefusal;
use lance_graph_dt64::LaneRefusal;
use virtual_address_lab::grid::{Grid, World};
use virtual_address_lab::physical::{apply, Moved};
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::view::{Ledger, Refused, RowObject};

const LANES: usize = 32;
const LEN: usize = 300;

fn grid() -> Grid {
    Grid::new(LANES, LEN, |l, p| (l * 1_000 + p) as i32)
}

/// FAILS IF: forming rows reads, a demanded property reads more than one
/// value, or a value differs from the view the moving oracle builds.
#[test]
fn only_demanded_properties_are_read() {
    let grid = grid();
    let world = World::new(LANES);
    let binding = world.bind(&grid).unwrap();
    let ledger = Ledger::default();

    // The entity view: row = position, property = lane.
    let rows: Vec<RowObject> = (0..LEN)
        .map(|row| RowObject {
            recipe: Recipe::TRANSPOSE,
            row,
        })
        .collect();
    assert_eq!(ledger.reads.get(), 0, "forming 300 rows read nothing");
    assert!(std::mem::size_of::<RowObject>() <= 16);

    let mut moved = Moved::default();
    let built = apply(&grid.matrix(), Recipe::TRANSPOSE, &mut moved);
    assert_eq!(moved.values, (LANES * LEN) as u64);

    let row = rows[123];
    assert_eq!(
        row.get(&world, &binding, &grid, 5, &ledger),
        Ok(built[123][5])
    );
    assert_eq!(
        row.get(&world, &binding, &grid, 31, &ledger),
        Ok(built[123][31])
    );
    assert_eq!(built[123][5], 5_123);
    // Two of the row's 32 properties, two values.
    assert_eq!(ledger.reads.get(), 2);
    assert_eq!(ledger.resolutions.get(), 2);
    assert_eq!(
        row.get(&world, &binding, &grid, LANES, &ledger),
        Err(Refused::OutsideTheView)
    );
    assert_eq!(ledger.reads.get(), 2);
}

/// FAILS IF: any recipe's row object disagrees with the built view at any
/// cell (a non-square tile, so an exchanged axis shows).
#[test]
fn every_recipe_reads_what_the_moving_oracle_builds() {
    let grid = Grid::new(4, 7, |l, p| (l * 50 + p * 3) as i32);
    let world = World::new(4);
    let binding = world.bind(&grid).unwrap();
    let ledger = Ledger::default();
    let mut cells = 0u64;
    for recipe in Recipe::ALL {
        let mut moved = Moved::default();
        let built = apply(&grid.matrix(), recipe, &mut moved);
        let (rows, cols) = recipe.view_dims(4, 7);
        assert_eq!(
            (built.len(), built[0].len()),
            (rows, cols),
            "{}",
            recipe.name()
        );
        for (row, built_row) in built.iter().enumerate() {
            for (c, want) in built_row.iter().enumerate() {
                let object = RowObject { recipe, row };
                assert_eq!(object.get(&world, &binding, &grid, c, &ledger), Ok(*want));
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 8 * 28);
    assert_eq!(ledger.reads.get(), cells);
}

/// FAILS IF: a row object formed under an old frame still reads after the
/// source moved.
#[test]
fn a_row_object_does_not_outlive_the_frame() {
    let mut grid = grid();
    let world = World::new(LANES);
    let binding = world.bind(&grid).unwrap();
    let ledger = Ledger::default();
    let row = RowObject {
        recipe: Recipe::TRANSPOSE,
        row: 7,
    };
    assert_eq!(row.get(&world, &binding, &grid, 3, &ledger), Ok(3_007));
    grid.advance_source();
    assert_eq!(
        row.get(&world, &binding, &grid, 3, &ledger),
        Err(Refused::Lane(LaneRefusal::Frame(
            FrameRefusal::SourceChanged
        )))
    );
    assert_eq!(ledger.reads.get(), 1);
}
