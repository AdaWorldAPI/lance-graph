//! The reference that MOVES data: each recipe as a sequence of physical
//! matrix operations, written from the recipes' names and independent of
//! [`crate::recipe::Recipe`]'s bits. It is the scalar oracle of the address
//! algebra and the deliberately expensive baseline of the measurements.

use crate::grid::Grid;
use crate::recipe::Recipe;
use crate::view::Trace;

/// One owned copy of a view, a row per `Vec`.
pub type Matrix = Vec<Vec<i32>>;

/// The lane matrix: one lane per row. A copy of the source.
pub fn of(grid: &Grid, trace: &Trace) -> Matrix {
    Trace::bump(&trace.copied, (grid.lanes() * grid.len()) as u64);
    (0..grid.lanes())
        .map(|l| grid.raw_lane(l).to_vec())
        .collect()
}

fn moved(m: &Matrix, trace: &Trace) {
    Trace::bump(&trace.transforms, 1);
    Trace::bump(&trace.copied, m.iter().map(|r| r.len() as u64).sum());
}

pub fn transpose(m: &Matrix, trace: &Trace) -> Matrix {
    let cols = m.first().map_or(0, Vec::len);
    let out: Matrix = (0..cols)
        .map(|c| m.iter().map(|row| row[c]).collect())
        .collect();
    moved(&out, trace);
    out
}

pub fn reverse_rows(m: &Matrix, trace: &Trace) -> Matrix {
    let out: Matrix = m.iter().rev().cloned().collect();
    moved(&out, trace);
    out
}

pub fn reverse_cols(m: &Matrix, trace: &Trace) -> Matrix {
    let out: Matrix = m
        .iter()
        .map(|row| row.iter().rev().copied().collect())
        .collect();
    moved(&out, trace);
    out
}

/// The view of `m` through `recipe`, built.
pub fn apply(m: &Matrix, recipe: Recipe, trace: &Trace) -> Matrix {
    if recipe == Recipe::IDENTITY {
        m.clone()
    } else if recipe == Recipe::ROTATE90 {
        reverse_rows(&transpose(m, trace), trace)
    } else if recipe == Recipe::ROTATE180 {
        reverse_cols(&reverse_rows(m, trace), trace)
    } else if recipe == Recipe::ROTATE270 {
        transpose(&reverse_rows(m, trace), trace)
    } else if recipe == Recipe::MIRROR_X {
        reverse_rows(m, trace)
    } else if recipe == Recipe::MIRROR_Y {
        reverse_cols(m, trace)
    } else if recipe == Recipe::TRANSPOSE {
        transpose(m, trace)
    } else {
        reverse_cols(&reverse_rows(&transpose(m, trace), trace), trace)
    }
}

/// A chain of views, each one built from the last.
pub fn apply_chain(m: &Matrix, chain: &[Recipe], trace: &Trace) -> Matrix {
    chain
        .iter()
        .fold(m.clone(), |acc, &g| apply(&acc, g, trace))
}
