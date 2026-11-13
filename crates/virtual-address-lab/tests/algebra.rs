//! P0, P1: the eight recipes against two references that do not share their
//! bits: cell-by-cell composition, and the oracle that moves data.

use virtual_address_lab::grid::Grid;
use virtual_address_lab::physical::{apply, apply_chain, Moved};
use virtual_address_lab::recipe::Recipe;

const N: usize = 5;

fn tile() -> Grid {
    Grid::new(N, N, |l, p| (l * 100 + p * 7 + 1) as i32)
}

/// The view of `grid` through `recipe`, read cell by cell.
fn virtual_view(grid: &Grid, recipe: Recipe) -> Vec<Vec<i32>> {
    let (rows, cols) = recipe.view_dims(grid.lanes(), grid.len());
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let (lane, pos) = recipe.resolve(grid.lanes(), grid.len(), r, c).unwrap();
                    grid.cell(lane, pos)
                })
                .collect()
        })
        .collect()
}

/// P0. FAILS IF: any of the 64 products disagrees with composing the two
/// address maps cell by cell, or with building one view from the other.
#[test]
fn every_product_matches_two_independent_references() {
    let grid = tile();
    let mut moved = Moved::default();
    for g in Recipe::ALL {
        for h in Recipe::ALL {
            let product = g.then(h);
            for r in 0..N {
                for c in 0..N {
                    let (r1, c1) = h.resolve(N, N, r, c).unwrap();
                    assert_eq!(
                        product.resolve(N, N, r, c),
                        g.resolve(N, N, r1, c1),
                        "{} then {}",
                        g.name(),
                        h.name()
                    );
                }
            }
            assert_eq!(
                virtual_view(&grid, product),
                apply_chain(&grid.matrix(), &[g, h], &mut moved),
                "{} then {}",
                g.name(),
                h.name()
            );
        }
    }
    // 12 rearrangements for the eight recipes (0+2+2+2+1+1+1+3), each used
    // 8 times as `g` and 8 times as `h`.
    assert_eq!(moved.transposes, 16 * 12, "the oracle did move data");
}

/// P0. FAILS IF: the eight are not a group of eight distinct maps, or the
/// product is commutative (it must not be).
#[test]
fn the_eight_form_a_group_that_is_not_commutative() {
    let grid = tile();
    let views: Vec<_> = Recipe::ALL
        .iter()
        .map(|&g| virtual_view(&grid, g))
        .collect();
    for i in 0..8 {
        for j in 0..i {
            assert_ne!(views[i], views[j], "two recipes read the same view");
        }
    }
    for g in Recipe::ALL {
        assert_eq!(g.then(g.inverse()), Recipe::IDENTITY);
        assert_eq!(g.inverse().then(g), Recipe::IDENTITY);
        assert_eq!(g.then(Recipe::IDENTITY), g);
    }
    assert_ne!(
        Recipe::ROTATE90.then(Recipe::MIRROR_X),
        Recipe::MIRROR_X.then(Recipe::ROTATE90)
    );
    let quarter = Recipe::ROTATE90;
    assert_eq!(quarter.then(quarter), Recipe::ROTATE180);
    assert_eq!(quarter.then(quarter).then(quarter), Recipe::ROTATE270);
}

/// The brief's fixture. FAILS IF: Rotate90 of 1..9 is not `3 6 9 / 2 5 8 /
/// 1 4 7`, or a reversal of one index alone produces it.
#[test]
fn rotate90_is_the_quarter_turn_and_no_single_reversal() {
    let grid = Grid::three();
    let want = vec![vec![3, 6, 9], vec![2, 5, 8], vec![1, 4, 7]];
    assert_eq!(virtual_view(&grid, Recipe::ROTATE90), want);
    let mut moved = Moved::default();
    assert_eq!(apply(&grid.matrix(), Recipe::ROTATE90, &mut moved), want);
    // Reversing lanes, reversing positions, or both, never gives it.
    for reversal in [Recipe::MIRROR_X, Recipe::MIRROR_Y, Recipe::ROTATE180] {
        assert_ne!(virtual_view(&grid, reversal), want, "{}", reversal.name());
    }
}

/// P1. FAILS IF: a chain that undoes itself does not normalise to Identity,
/// or normalising read or moved anything.
#[test]
fn a_chain_that_undoes_itself_normalises_to_identity() {
    let grid = tile();
    let chain = [Recipe::ROTATE90, Recipe::ROTATE270];
    assert_eq!(Recipe::normalize(&chain), Recipe::IDENTITY);
    assert_eq!(
        Recipe::normalize(&[Recipe::ROTATE180, Recipe::ROTATE180]),
        Recipe::IDENTITY
    );
    // Cell by cell, without the product: applying Rotate180 twice is the cell.
    for r in 0..N {
        for c in 0..N {
            let (r1, c1) = Recipe::ROTATE180.resolve(N, N, r, c).unwrap();
            assert_eq!(Recipe::ROTATE180.resolve(N, N, r1, c1), Some((r, c)));
        }
    }
    let long = [
        Recipe::ROTATE90,
        Recipe::MIRROR_X,
        Recipe::ROTATE270,
        Recipe::MIRROR_Y,
    ];
    assert_eq!(Recipe::normalize(&long), Recipe::IDENTITY);
    // The moving reference needs four rearrangements for the same view.
    let mut moved = Moved::default();
    assert_eq!(
        apply_chain(&grid.matrix(), &chain, &mut moved),
        grid.matrix()
    );
    assert_eq!(moved.transposes, 4);
    assert_eq!(moved.values, 4 * (N * N) as u64);
}
