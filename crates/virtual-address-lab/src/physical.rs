//! The reference that MOVES data: each recipe as a sequence of physical
//! matrix operations, written from the recipes' names and independent of
//! [`crate::recipe::Recipe`]'s bits. Oracle and baseline only.

use crate::recipe::Recipe;

/// One owned copy of a view, a row per `Vec`.
pub type Matrix = Vec<Vec<i32>>;

/// What building views cost.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Moved {
    /// Physical rearrangements.
    pub transposes: u64,
    /// Values written into a new buffer.
    pub values: u64,
}

fn count(m: &Matrix, moved: &mut Moved) {
    moved.transposes += 1;
    moved.values += m.iter().map(|r| r.len() as u64).sum::<u64>();
}

pub fn transpose(m: &Matrix, moved: &mut Moved) -> Matrix {
    let cols = m.first().map_or(0, Vec::len);
    let out: Matrix = (0..cols)
        .map(|c| m.iter().map(|row| row[c]).collect())
        .collect();
    count(&out, moved);
    out
}

pub fn reverse_rows(m: &Matrix, moved: &mut Moved) -> Matrix {
    let out: Matrix = m.iter().rev().cloned().collect();
    count(&out, moved);
    out
}

pub fn reverse_cols(m: &Matrix, moved: &mut Moved) -> Matrix {
    let out: Matrix = m
        .iter()
        .map(|row| row.iter().rev().copied().collect())
        .collect();
    count(&out, moved);
    out
}

/// The view of `m` through `recipe`, built.
pub fn apply(m: &Matrix, recipe: Recipe, moved: &mut Moved) -> Matrix {
    if recipe == Recipe::IDENTITY {
        m.clone()
    } else if recipe == Recipe::ROTATE90 {
        let t = transpose(m, moved);
        reverse_rows(&t, moved)
    } else if recipe == Recipe::ROTATE180 {
        let r = reverse_rows(m, moved);
        reverse_cols(&r, moved)
    } else if recipe == Recipe::ROTATE270 {
        let r = reverse_rows(m, moved);
        transpose(&r, moved)
    } else if recipe == Recipe::MIRROR_X {
        reverse_rows(m, moved)
    } else if recipe == Recipe::MIRROR_Y {
        reverse_cols(m, moved)
    } else if recipe == Recipe::TRANSPOSE {
        transpose(m, moved)
    } else {
        let t = transpose(m, moved);
        let r = reverse_rows(&t, moved);
        reverse_cols(&r, moved)
    }
}

/// A chain of views, each one built from the last.
pub fn apply_chain(m: &Matrix, chain: &[Recipe], moved: &mut Moved) -> Matrix {
    let mut out = m.clone();
    for &g in chain {
        out = apply(&out, g, moved);
    }
    out
}
