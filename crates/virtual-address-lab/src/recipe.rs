//! The address recipe: one of the eight symmetries of a square tile.
//!
//! A tile is `M[lane][pos]`: `lanes` source-owned lanes of `len` positions.
//! A recipe says which physical cell view cell `(r, c)` reads. It holds three
//! bits and moves nothing.
//!
//! ```text
//! (a, b)  = if swap { (c, r) } else { (r, c) }
//! lane    = if flip_lane { lanes - 1 - a } else { a }
//! pos     = if flip_pos  { len   - 1 - b } else { b }
//! ```
//!
//! Convention: `ROTATE90` is `view(r, c) = M[c][N-1-r]`, the counter-clockwise
//! quarter turn of the lane matrix drawn one lane per row. The entity view
//! (row `r` an object, property `c` the lane `c`) is `TRANSPOSE`.

/// One symmetry of the square, as an address map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Recipe {
    swap: bool,
    flip_lane: bool,
    flip_pos: bool,
}

const fn r(swap: bool, flip_lane: bool, flip_pos: bool) -> Recipe {
    Recipe {
        swap,
        flip_lane,
        flip_pos,
    }
}

impl Recipe {
    pub const IDENTITY: Recipe = r(false, false, false);
    pub const ROTATE90: Recipe = r(true, false, true);
    pub const ROTATE180: Recipe = r(false, true, true);
    pub const ROTATE270: Recipe = r(true, true, false);
    /// Lane order reversed.
    pub const MIRROR_X: Recipe = r(false, true, false);
    /// Position order reversed.
    pub const MIRROR_Y: Recipe = r(false, false, true);
    pub const TRANSPOSE: Recipe = r(true, false, false);
    pub const ANTI_TRANSPOSE: Recipe = r(true, true, true);

    /// All eight, in the order of [`Recipe::index`].
    pub const ALL: [Recipe; 8] = [
        Recipe::IDENTITY,
        Recipe::ROTATE90,
        Recipe::ROTATE180,
        Recipe::ROTATE270,
        Recipe::MIRROR_X,
        Recipe::MIRROR_Y,
        Recipe::TRANSPOSE,
        Recipe::ANTI_TRANSPOSE,
    ];

    pub const NAMES: [&'static str; 8] = [
        "Identity",
        "Rotate90",
        "Rotate180",
        "Rotate270",
        "MirrorX",
        "MirrorY",
        "Transpose",
        "AntiTranspose",
    ];

    /// Position in [`Recipe::ALL`].
    pub fn index(self) -> usize {
        Recipe::ALL
            .iter()
            .position(|&g| g == self)
            .expect("ALL holds every recipe")
    }

    pub fn name(self) -> &'static str {
        Recipe::NAMES[self.index()]
    }

    /// Whether view rows run along positions (so a view COLUMN is one lane).
    pub fn swaps(self) -> bool {
        self.swap
    }

    /// `(rows, cols)` of the view over a tile of `lanes` x `len`.
    pub fn view_dims(self, lanes: usize, len: usize) -> (usize, usize) {
        if self.swap {
            (len, lanes)
        } else {
            (lanes, len)
        }
    }

    /// The physical `(lane, pos)` view cell `(row, col)` reads, or `None`
    /// outside the view.
    pub fn resolve(
        self,
        lanes: usize,
        len: usize,
        row: usize,
        col: usize,
    ) -> Option<(usize, usize)> {
        let (a, b) = if self.swap { (col, row) } else { (row, col) };
        if a >= lanes || b >= len {
            return None;
        }
        Some((
            if self.flip_lane { lanes - 1 - a } else { a },
            if self.flip_pos { len - 1 - b } else { b },
        ))
    }

    /// The recipe of "view through `self`, then view THAT through `next`":
    /// `(self.then(next))(r, c) = self(next(r, c))`.
    pub fn then(self, next: Recipe) -> Recipe {
        if self.swap {
            // `self` exchanges the components `next` produced, so each flip of
            // `self` meets the OTHER flip of `next`.
            r(
                !next.swap,
                self.flip_lane ^ next.flip_pos,
                self.flip_pos ^ next.flip_lane,
            )
        } else {
            r(
                next.swap,
                self.flip_lane ^ next.flip_lane,
                self.flip_pos ^ next.flip_pos,
            )
        }
    }

    /// The recipe that undoes this one.
    pub fn inverse(self) -> Recipe {
        *Recipe::ALL
            .iter()
            .find(|g| self.then(**g) == Recipe::IDENTITY)
            .expect("every symmetry has an inverse")
    }

    /// A chain of views, first applied first, as one recipe. Reads nothing.
    pub fn normalize(chain: &[Recipe]) -> Recipe {
        chain.iter().fold(Recipe::IDENTITY, |acc, &g| acc.then(g))
    }

    /// A stable number naming the recipe, for a method guard.
    pub fn digest(self) -> u32 {
        0x5EC1_9E00 | self.index() as u32
    }

    /// How view column `col` lies in the tile.
    pub fn column(self, lanes: usize, len: usize, col: usize) -> Option<Column> {
        let (rows, cols) = self.view_dims(lanes, len);
        if col >= cols || rows == 0 {
            return None;
        }
        let (lane0, pos0) = self.resolve(lanes, len, 0, col)?;
        Some(if self.swap {
            Column {
                access: Access::Lane(lane0),
                reversed: self.flip_pos,
            }
        } else {
            Column {
                access: Access::AcrossLanes { pos: pos0 },
                reversed: self.flip_lane,
            }
        })
    }
}

/// Where a view column's cells are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// One physical lane, every position.
    Lane(usize),
    /// One position of every lane.
    AcrossLanes { pos: usize },
}

/// A view column as a physical run: which cells, and whether view row order
/// is the reverse of physical order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Column {
    pub access: Access,
    pub reversed: bool,
}
