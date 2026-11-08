//! A virtual CSR over the 256 x 256 identity cube.
//!
//! The adjacency lives in the same local coordinates as the lanes: identity
//! `i` is row `slot(i)` of row tile `tile(i)`. Each row tile lists its
//! nonempty column tiles, and each (row tile, column tile) pair is one
//! 256 x 256 bitmask block, four words per row. There is no row-pointer or
//! column-index array: row `i`'s neighbors are the set bits of its row in the
//! blocks of its tile, read in place. [`ExplicitCsr`] builds the materialized
//! form from the same edges; it exists only as the differential oracle.

use std::collections::BTreeMap;

use lance_graph_dt64::CoordSpace;

use crate::cube::{slot, tile, TILE};

/// Words per block row.
const ROW_WORDS: usize = TILE / 64;

/// One 256 x 256 adjacency block: column tile `tc` of its row tile.
pub struct Block {
    pub tc: usize,
    words: Box<[u64]>,
}

impl Block {
    /// The four words of block row `r`; bit `c` is the edge to column `c`.
    pub fn row_words(&self, r: usize) -> &[u64] {
        &self.words[r * ROW_WORDS..(r + 1) * ROW_WORDS]
    }

    /// Whether any of block rows `rows` has an edge.
    pub fn any_in_rows(&self, rows: std::ops::Range<usize>) -> bool {
        self.words[rows.start * ROW_WORDS..rows.end * ROW_WORDS]
            .iter()
            .any(|&w| w != 0)
    }
}

/// The virtual CSR: a tile directory of bitmask blocks, in one row order.
pub struct VirtualCsr {
    pub space: CoordSpace,
    pub n: usize,
    dir: Vec<Vec<Block>>,
}

impl VirtualCsr {
    /// Build from `(row, col)` edges over `n` identities in `space`.
    pub fn from_edges(
        space: CoordSpace,
        n: usize,
        edges: impl IntoIterator<Item = (usize, usize)>,
    ) -> VirtualCsr {
        let tiles = n.div_ceil(TILE);
        let mut blocks: BTreeMap<(usize, usize), Box<[u64]>> = BTreeMap::new();
        for (i, j) in edges {
            assert!(i < n && j < n, "edge ({i}, {j}) outside n = {n}");
            let words = blocks
                .entry((tile(i), tile(j)))
                .or_insert_with(|| vec![0u64; TILE * ROW_WORDS].into_boxed_slice());
            let c = slot(j);
            words[slot(i) * ROW_WORDS + c / 64] |= 1 << (c % 64);
        }
        let mut dir: Vec<Vec<Block>> = (0..tiles).map(|_| Vec::new()).collect();
        for ((tr, tc), words) in blocks {
            dir[tr].push(Block { tc, words });
        }
        VirtualCsr { space, n, dir }
    }

    /// The blocks of row tile `tr`, by ascending column tile.
    pub fn row_tile(&self, tr: usize) -> &[Block] {
        &self.dir[tr]
    }

    /// Row tiles.
    pub fn row_tiles(&self) -> usize {
        self.dir.len()
    }

    /// Row `i`'s neighbors, ascending, read from the blocks in place.
    pub fn row(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let r = slot(i);
        self.dir[tile(i)].iter().flat_map(move |b| {
            b.row_words(r)
                .iter()
                .enumerate()
                .flat_map(move |(w, &word)| {
                    let mut bits = word;
                    std::iter::from_fn(move || {
                        (bits != 0).then(|| {
                            let c = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            b.tc * TILE + w * 64 + c
                        })
                    })
                })
        })
    }

    /// Stored edges.
    pub fn nnz(&self) -> usize {
        self.dir
            .iter()
            .flatten()
            .map(|b| {
                b.words
                    .iter()
                    .map(|w| w.count_ones() as usize)
                    .sum::<usize>()
            })
            .sum()
    }

    /// Nonempty blocks.
    pub fn blocks(&self) -> usize {
        self.dir.iter().map(Vec::len).sum()
    }

    /// Bytes of block storage.
    pub fn bytes(&self) -> usize {
        self.blocks() * TILE * ROW_WORDS * 8
    }
}

/// The materialized CSR of the same edges: the differential oracle.
pub struct ExplicitCsr {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
}

impl ExplicitCsr {
    pub fn from_edges(n: usize, edges: impl IntoIterator<Item = (usize, usize)>) -> ExplicitCsr {
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, j) in edges {
            rows[i].push(j);
        }
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        row_ptr.push(0);
        for mut r in rows {
            r.sort_unstable();
            r.dedup();
            col_idx.extend(r);
            row_ptr.push(col_idx.len());
        }
        ExplicitCsr { row_ptr, col_idx }
    }

    /// Row `i`'s neighbors, ascending.
    pub fn row(&self, i: usize) -> &[usize] {
        &self.col_idx[self.row_ptr[i]..self.row_ptr[i + 1]]
    }
}
