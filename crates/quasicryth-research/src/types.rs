//! Algebraic types — direct transcode of `qtc.h` types.
//!
//! All structures here mirror the C reference; ownership is converted to
//! idiomatic Rust (`Vec` instead of `malloc`/`free`). No `unsafe`.

/// One tile in a quasicrystal word-tiling.
///
/// L-tile consumes 2 words (bigram); S-tile consumes 1 word (unigram).
/// `wpos` is the word position at which this tile begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// Word position where this tile begins (0-based).
    pub wpos: u32,
    /// Number of source words consumed — 1 for S, 2 for L.
    pub nwords: u8,
    /// `true` for an L-tile (large; bigram), `false` for S (small; unigram).
    pub is_l: bool,
}

impl Tile {
    /// Construct an S-tile (1 word) at the given word-position.
    #[inline]
    #[must_use]
    pub const fn small(wpos: u32) -> Self {
        Self {
            wpos,
            nwords: 1,
            is_l: false,
        }
    }

    /// Construct an L-tile (2 words) at the given word-position.
    #[inline]
    #[must_use]
    pub const fn large(wpos: u32) -> Self {
        Self {
            wpos,
            nwords: 2,
            is_l: true,
        }
    }
}

/// One entry at one hierarchy level — represents a super-tile spanning
/// `[start, end)` in the level below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HLevel {
    /// Start index (inclusive) into the level below.
    pub start: u32,
    /// End index (exclusive) into the level below.
    pub end: u32,
    /// `true` if this super-tile carries the L identity.
    pub is_l: bool,
}

/// Parent-pointer for a child at one level → its super-tile at the next level.
///
/// `parent_idx = None` if the child has no parent at the next level
/// (boundary case at the trailing edge of the sequence).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParentMap {
    /// Index of the parent super-tile at the next level, or `None` if this
    /// child has no parent (boundary case at the trailing edge of the sequence).
    pub parent_idx: Option<u32>,
    /// Position within parent: 0 = first child (always; the only one for super-S),
    /// 1 = second child (only present for super-L, which has both an L and an S below).
    pub pos: i8,
}

/// Full substitution hierarchy: `levels[0]` is the tile level,
/// `levels[k]` for `k > 0` is the result of `k` deflations.
#[derive(Debug, Clone)]
pub struct Hierarchy {
    /// Per-level slices. `levels[k]` has `level_count[k]` entries.
    pub levels: Vec<Vec<HLevel>>,
    /// `parent_maps[k]`: for each entry at level `k`, its parent at level `k+1`.
    pub parent_maps: Vec<Vec<ParentMap>>,
}

impl Hierarchy {
    pub(crate) fn empty() -> Self {
        Self {
            levels: Vec::new(),
            parent_maps: Vec::new(),
        }
    }

    /// Number of hierarchy levels including the tile level (level 0).
    #[inline]
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Count of entries at level `k`. Returns 0 if `k` is out of range.
    #[inline]
    #[must_use]
    pub fn level_count(&self, k: usize) -> usize {
        self.levels.get(k).map_or(0, Vec::len)
    }
}

/// Deep-position detection result.
///
/// `can[k][i] = true` means tile `i` at level 0 is a legal entry point for an
/// `n`-gram lookup at hierarchy level `k`, where `n = HIER_WORD_LENS[k]`.
/// `skip[k][i]` is the number of additional level-0 tiles consumed by the
/// match (so the next position to consider is `i + 1 + skip[k][i]`).
#[derive(Debug, Clone)]
pub struct DeepPositions {
    /// `can[k][i] = true` iff tile `i` is a legal entry point for an n-gram
    /// lookup at hierarchy level `k`.
    pub can: Vec<Vec<bool>>,
    /// `skip[k][i]` = number of additional level-0 tiles consumed by the
    /// match at tile `i`, level `k`. Next candidate is `i + 1 + skip[k][i]`.
    pub skip: Vec<Vec<u32>>,
    /// Maximum hierarchy level reachable by this detection result.
    pub max_k: usize,
}

/// A tiling descriptor: a one-D irrational `alpha` in `(0,1)` and a phase
/// shift `phase` in `[0,1)`. Both inputs to the cut-and-project rule.
#[derive(Debug, Clone, Copy)]
pub struct TilingDesc {
    /// Irrational slope α ∈ (0,1) for the cut-and-project rule.
    pub alpha: f64,
    /// Phase shift θ ∈ [0,1).
    pub phase: f64,
    /// Human-readable label mirroring the upstream C reference.
    pub name: &'static str,
}
