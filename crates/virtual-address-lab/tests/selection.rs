//! P3, P4 and the negative controls: a recipe in front of the EXISTING
//! selection path (`SelectionSource` -> `LaneBinding::selected` ->
//! `fold_bound_selection` -> `execute_sel`).

use lance_graph_dt64::dr::FrameRefusal;
use lance_graph_dt64::{CoordSpace, LaneRefusal};
use lance_graph_mask_risc::Value;
use lance_graph_report::bound_sel::{BoundFold, BoundSelRefusal};
use virtual_address_lab::grid::{Grid, World};
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::view::{field_through, fold_property, Ledger, Refused, RowObject};

const LANES: usize = 8;
const LEN: usize = 1_000;
/// The lane the order is keyed on.
const KEY: u8 = 2;
const ENTITY: [Recipe; 1] = [Recipe::TRANSPOSE];

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn value(lane: usize, pos: usize) -> i32 {
    (mix((lane * 100_003 + pos) as u64) % 2_001) as i32 - 1_000
}

fn grid() -> Grid {
    Grid::new(LANES, LEN, value)
}

/// Row at a time, over a position list, with no witness.
fn scalar_sum(grid: &Grid, field: usize, rows: &[u32]) -> i64 {
    rows.iter()
        .map(|&r| i64::from(grid.cell(field, r as usize)))
        .sum()
}

fn scalar_count_gt(grid: &Grid, field: usize, rows: &[u32], t: i32) -> u64 {
    rows.iter()
        .filter(|&&r| grid.cell(field, r as usize) > t)
        .count() as u64
}

/// P3. FAILS IF: the source-issued order does not reach the existing fold,
/// the fold disagrees with the scalar oracle, the positions are a copy, or
/// the fixture's order is trivially the row order.
#[test]
fn a_source_issued_order_reaches_the_existing_fold() {
    let mut grid = grid();
    let world = World::new(LANES);
    let sel = grid
        .issue_order(KEY)
        .expect("the source produces the order");
    let binding = world.bind(&grid).unwrap();

    // The witness lends the source's own positions: same address, no copy.
    let (_, positions) = binding
        .selected(&world.store, &world.view, grid.frame_store(), &grid, 5, sel)
        .unwrap();
    assert!(std::ptr::eq(
        positions.as_ptr(),
        grid.raw_positions(sel).as_ptr()
    ));
    assert_eq!(positions.len(), LEN);
    // It is the key order, and not the row order.
    let keys: Vec<i32> = positions
        .iter()
        .map(|&r| grid.cell(usize::from(KEY), r as usize))
        .collect();
    assert!(keys.windows(2).all(|w| w[0] <= w[1]));
    assert!(positions.iter().enumerate().any(|(i, &r)| r as usize != i));

    for property in 0..LANES {
        let rows = grid.raw_positions(sel).to_vec();
        assert_eq!(
            fold_property(
                &world,
                &binding,
                &grid,
                &ENTITY,
                property,
                sel,
                BoundFold::SumI32
            ),
            Ok(Value::I64(scalar_sum(&grid, property, &rows)))
        );
        assert_eq!(
            fold_property(
                &world,
                &binding,
                &grid,
                &ENTITY,
                property,
                sel,
                BoundFold::CountGtI32 { t: 250 }
            ),
            Ok(Value::Count(scalar_count_gt(&grid, property, &rows, 250)))
        );
    }
}

/// P1 at the fold. FAILS IF: a chain that normalises to the entity view
/// folds differently from it, or a recipe that names another field folds the
/// same (the recipe would then not be consulted).
#[test]
fn the_chain_is_normalised_before_the_fold_is_called() {
    let mut grid = grid();
    let world = World::new(LANES);
    let sel = grid.issue_filter(usize::from(KEY), -200, 300);
    let binding = world.bind(&grid).unwrap();
    let rows = grid.raw_positions(sel).to_vec();
    assert!(rows.len() > 100 && rows.len() < LEN - 100);

    let direct = fold_property(&world, &binding, &grid, &ENTITY, 1, sel, BoundFold::SumI32);
    assert_eq!(direct, Ok(Value::I64(scalar_sum(&grid, 1, &rows))));
    let long = [Recipe::TRANSPOSE, Recipe::ROTATE90, Recipe::ROTATE270];
    assert_eq!(field_through(&long, &grid, 1), Ok(1));
    assert_eq!(
        fold_property(&world, &binding, &grid, &long, 1, sel, BoundFold::SumI32),
        direct
    );
    // Rotate90 is the entity view with its rows reversed: the same field, and
    // a sum does not see row order.
    assert_eq!(field_through(&[Recipe::ROTATE90], &grid, 1), Ok(1));
    // AntiTranspose reverses the property axis: another field, another sum.
    assert_eq!(field_through(&[Recipe::ANTI_TRANSPOSE], &grid, 1), Ok(6));
    let other = fold_property(
        &world,
        &binding,
        &grid,
        &[Recipe::ANTI_TRANSPOSE],
        1,
        sel,
        BoundFold::SumI32,
    );
    assert_eq!(other, Ok(Value::I64(scalar_sum(&grid, 6, &rows))));
    assert_ne!(other, direct);
}

/// The missing operand, reported. FAILS IF: a property that is one position
/// of every lane is folded anyway.
#[test]
fn a_property_that_crosses_fields_is_reported_not_run() {
    let mut grid = grid();
    let world = World::new(LANES);
    let sel = grid.issue_filter(0, 0, 1_000);
    let binding = world.bind(&grid).unwrap();
    for recipe in [
        Recipe::IDENTITY,
        Recipe::ROTATE180,
        Recipe::MIRROR_X,
        Recipe::MIRROR_Y,
    ] {
        assert_eq!(
            fold_property(
                &world,
                &binding,
                &grid,
                &[recipe],
                3,
                sel,
                BoundFold::SumI32
            ),
            Err(Refused::PropertyCrossesFields { recipe })
        );
    }
    assert_eq!(
        field_through(&ENTITY, &grid, LANES),
        Err(Refused::OutsideTheView)
    );
}

/// P4, control 4. FAILS IF: reordering every row reproduces a field-axis
/// change, or a selection can reach a lane the fold was not given.
#[test]
fn a_row_order_is_not_a_field_axis_transformation() {
    let mut grid = grid();
    let world = World::new(LANES);
    let reversed = grid.issue_list((0..LEN as u32).rev().collect());
    let forward = grid.issue_list((0..LEN as u32).collect());
    let binding = world.bind(&grid).unwrap();
    let ledger = Ledger::default();

    // What a full row reversal DOES equal: Rotate90, row by row.
    let rows = grid.raw_positions(reversed).to_vec();
    for (i, &source_row) in rows.iter().enumerate().step_by(97) {
        for property in 0..LANES {
            let turned = RowObject {
                recipe: Recipe::ROTATE90,
                row: i,
            };
            let entity = RowObject {
                recipe: Recipe::TRANSPOSE,
                row: source_row as usize,
            };
            assert_eq!(
                turned.get(&world, &binding, &grid, property, &ledger),
                entity.get(&world, &binding, &grid, property, &ledger)
            );
        }
    }
    // What it does not: the field axis. Both row orders fold field 1 ...
    let a = fold_property(
        &world,
        &binding,
        &grid,
        &ENTITY,
        1,
        reversed,
        BoundFold::SumI32,
    );
    let b = fold_property(
        &world,
        &binding,
        &grid,
        &ENTITY,
        1,
        forward,
        BoundFold::SumI32,
    );
    assert_eq!(a, b);
    // ... and neither is the reversed property axis.
    let flipped = fold_property(
        &world,
        &binding,
        &grid,
        &[Recipe::ANTI_TRANSPOSE],
        1,
        forward,
        BoundFold::SumI32,
    );
    assert_ne!(flipped, a);

    // A selection reads only the lane of the field it is folded with:
    // rewriting lane 6 leaves field 1's fold alone and moves the flipped one.
    grid.rewrite_lane(6, |p| p as i32);
    assert_eq!(
        fold_property(
            &world,
            &binding,
            &grid,
            &ENTITY,
            1,
            reversed,
            BoundFold::SumI32
        ),
        a
    );
    assert_eq!(
        fold_property(
            &world,
            &binding,
            &grid,
            &[Recipe::ANTI_TRANSPOSE],
            1,
            forward,
            BoundFold::SumI32
        ),
        Ok(Value::I64((0..LEN as i64).sum()))
    );
}

/// Control 2. FAILS IF: applying the source's order to one of two aligned
/// lanes and not the other gives the aligned answer. (Row at a time: no
/// admitted fold takes two lanes over a position list.)
#[test]
fn an_order_applied_to_one_of_two_aligned_lanes_changes_the_result() {
    let mut grid = grid();
    let sel = grid.issue_order(KEY).unwrap();
    let rows = grid.raw_positions(sel);
    let pair = |a: usize, b: usize| i64::from(grid.cell(0, a)) * i64::from(grid.cell(1, b));
    let aligned: i64 = rows.iter().map(|&r| pair(r as usize, r as usize)).sum();
    let in_row_order: i64 = (0..LEN).map(|r| pair(r, r)).sum();
    let one_lane_only: i64 = rows
        .iter()
        .enumerate()
        .map(|(i, &r)| pair(r as usize, i))
        .sum();
    assert_eq!(
        aligned, in_row_order,
        "a bijection applied to both lanes keeps the sum"
    );
    assert_ne!(one_lane_only, aligned);
}

/// Control 3. FAILS IF: a list with a duplicate is folded as a set.
#[test]
fn a_selection_with_duplicates_is_a_bag() {
    let mut grid = grid();
    let world = World::new(LANES);
    let bag = grid.issue_list(vec![7, 7, 7, 400]);
    let binding = world.bind(&grid).unwrap();
    let (once, far) = (i64::from(grid.cell(3, 7)), i64::from(grid.cell(3, 400)));
    assert_ne!(once, 0);
    assert_eq!(
        fold_property(&world, &binding, &grid, &ENTITY, 3, bag, BoundFold::SumI32),
        Ok(Value::I64(3 * once + far))
    );
    // As membership (a mask) it would be `once + far`: not a bijection of rows.
    assert_ne!(3 * once + far, once + far);
}

/// Control 5. FAILS IF: positions in another row order are folded.
#[test]
fn a_foreign_coordinate_space_is_refused() {
    let mut grid = grid();
    let world = World::new(LANES);
    let foreign = CoordSpace {
        population: 9,
        coordinates: 1,
    };
    let sel = grid.issue_list_in(vec![1, 2, 3], foreign);
    let own = grid.issue_list(vec![1, 2, 3]);
    let binding = world.bind(&grid).unwrap();
    let got = fold_property(&world, &binding, &grid, &ENTITY, 0, sel, BoundFold::SumI32);
    assert!(
        matches!(
            got,
            Err(Refused::Fold(BoundSelRefusal::Binding(
                LaneRefusal::SelectionSpace { .. }
            )))
        ),
        "{got:?}"
    );
    // The same positions in the source's own row order fold.
    assert!(fold_property(&world, &binding, &grid, &ENTITY, 0, own, BoundFold::SumI32).is_ok());
    // A binding that expects the foreign order is refused at the lanes.
    assert!(matches!(
        world.bind_as(&grid, foreign),
        Err(LaneRefusal::CoordinateSpace { .. })
    ));
}

/// Control 6. FAILS IF: a binding or a selection made before the source
/// moved is still admitted.
#[test]
fn an_old_frame_and_an_old_selection_are_refused() {
    let mut grid = grid();
    let world = World::new(LANES);
    let sel = grid.issue_order(KEY).unwrap();
    let old = world.bind(&grid).unwrap();
    assert!(fold_property(&world, &old, &grid, &ENTITY, 0, sel, BoundFold::SumI32).is_ok());

    grid.advance_source();
    assert_eq!(
        fold_property(&world, &old, &grid, &ENTITY, 0, sel, BoundFold::SumI32),
        Err(Refused::Fold(BoundSelRefusal::Binding(LaneRefusal::Frame(
            FrameRefusal::SourceChanged
        ))))
    );
    // A fresh binding does not revive a selection issued at the old revision.
    let fresh = world.bind(&grid).unwrap();
    let got = fold_property(&world, &fresh, &grid, &ENTITY, 0, sel, BoundFold::SumI32);
    assert!(
        matches!(
            got,
            Err(Refused::Fold(BoundSelRefusal::Binding(
                LaneRefusal::SelectionUnbound { .. }
            )))
        ),
        "{got:?}"
    );
    let again = grid.issue_order(KEY).unwrap();
    let fresh = world.bind(&grid).unwrap();
    assert!(fold_property(&world, &fresh, &grid, &ENTITY, 0, again, BoundFold::SumI32).is_ok());
}

/// Control 7, RECORDED AS OPEN. Lane bytes rewritten under an unmoved frame
/// are folded: neither the binding nor a recipe digest says who produced
/// them. FAILS IF: that changes (then this boundary moved and the plan's
/// table must be corrected).
#[test]
fn a_recipe_digest_does_not_witness_the_producer_of_the_bytes() {
    let mut grid = grid();
    let world = World::new(LANES);
    let sel = grid.issue_list((0..LEN as u32).collect());
    let binding = world.bind(&grid).unwrap();
    let digest = Recipe::normalize(&ENTITY).digest();
    let before = fold_property(&world, &binding, &grid, &ENTITY, 4, sel, BoundFold::SumI32);

    grid.rewrite_lane(4, |_| 1);
    let after = fold_property(&world, &binding, &grid, &ENTITY, 4, sel, BoundFold::SumI32);
    assert_eq!(after, Ok(Value::I64(LEN as i64)));
    assert_ne!(after, before);
    assert_eq!(Recipe::normalize(&ENTITY).digest(), digest);
}
