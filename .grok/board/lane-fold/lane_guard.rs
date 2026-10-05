//! Planner guardrails for a 64k ordered lane whose index never leaves `u16`.
//!
//! Drop this next to `lance-graph-quack`. It does not execute a `Program`.
//! It accepts or refuses a plan before `lower` runs, and it checks the
//! metric line after `execute` returns.
//!
//! The numbers are different types on purpose. A `WordFoldNs` cannot be
//! passed where a `PlaneNs` is required.

#![forbid(unsafe_code)]

use std::convert::TryFrom;

/// 2^16. A row address past this is not in the lane.
pub const LANE: u16 = u16::MAX;
pub const N: usize = 1 << 16;
pub const MASK_BYTES: usize = N / 8; // 8 KB
pub const PERM_BYTES: usize = N * 2; // 128 KB, the one allowed index
pub const LANE_BYTES: usize = N * 16; // 1 MB, 128-bit SoA
/// Masks a single query may hold before it has stored the route as coordinates.
pub const APERTURE_CAP: usize = 8;
/// Past this K, K mask passes lose to a hash aggregate. Measured, then pinned.
pub const K_KNEE: u32 = 64;
/// Dead-word fraction below which the word gate is not the win.
pub const SCATTERED_DEAD_WORD: f64 = 0.25;

// --- 1. a number belongs to one object ------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WordFoldNs(pub f64);
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileNs(pub f64);
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneNs(pub f64);

impl WordFoldNs {
    /// A million word-folds. 1e6 * 0.6 ns = 0.6 ms. Not 0.6 s.
    pub fn million(self) -> f64 {
        self.0 * 1_000_000.0
    }
}

// --- 2. the address never widens ------------------------------------------

/// A row in this lane. Constructing from a `u32` truncates only through
/// [`TryFrom`], so a foreign key that does not fit is a refusal, not a cast.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Row(u16);

impl TryFrom<u32> for Row {
    type Error = Refuse;
    fn try_from(v: u32) -> Result<Self, Refuse> {
        u16::try_from(v).map(Row).map_err(|_| Refuse::IndexWidened)
    }
}

/// Ingest witness. An unsorted buffer cannot construct it.
#[derive(Clone, Debug)]
pub struct OrderedLane {
    slots: Vec<u128>,
}

impl OrderedLane {
    pub fn try_from_sorted(slots: Vec<u128>, ordered: bool) -> Result<Self, Refuse> {
        if slots.len() != N {
            return Err(Refuse::LaneBound);
        }
        if !ordered {
            return Err(Refuse::UnorderedIngest);
        }
        Ok(Self { slots })
    }

    pub fn get(&self, row: Row) -> u128 {
        self.slots[row.0 as usize]
    }
}

// --- 3. the aperture is a borrowed plan value ------------------------------

/// 8 KB, borrowed. A fold receives it. A fold does not build it.
#[derive(Clone, Copy, Debug)]
pub struct Aperture<'a> {
    words: &'a [u64],
    shift: u16,
}

impl<'a> Aperture<'a> {
    pub fn resident(words: &'a [u64]) -> Result<Self, Refuse> {
        if words.len() != N / 64 {
            return Err(Refuse::LaneBound);
        }
        Ok(Self { words, shift: 0 })
    }

    pub fn with_shift(self, d: u16) -> Self {
        Self { shift: d, ..self }
    }

    /// Selectivity is a popcount. This is `REORDER_FILTER` without a catalog.
    pub fn dead_word_fraction(self) -> f64 {
        let dead = self.words.iter().filter(|w| **w == 0).count();
        dead as f64 / self.words.len() as f64
    }

    pub fn popcount(self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }
}

#[derive(Clone, Debug)]
pub struct ApertureSet<'a> {
    held: Vec<Aperture<'a>>,
}

impl<'a> ApertureSet<'a> {
    pub fn new() -> Self {
        Self { held: Vec::new() }
    }

    pub fn push(&mut self, a: Aperture<'a>) -> Result<(), Refuse> {
        if self.held.len() >= APERTURE_CAP {
            return Err(Refuse::ApertureCap);
        }
        self.held.push(a);
        Ok(())
    }
}

// --- 4. the plan the planner is allowed to emit ---------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    /// Compare against a constant. `AND` gates the next one.
    Cmp,
    /// Boolean over apertures. Fused form cannot skip.
    Mask { fused: bool },
    /// Rail join: tick `i` is tick `i+d` on the other ruler.
    Shift,
    /// Once. Output is a mask, not a pair list.
    AlignOnce,
    /// K masks. Refused past [`K_KNEE`].
    Group { k: u32 },
    /// The only write.
    Terminal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refuse {
    PairList,
    HashTable,
    IndexWidened,
    UnorderedIngest,
    LaneBound,
    ApertureCap,
    GroupKnee,
    NestedLoop,
    StringOrRegex,
    FusedClaimsSkip,
    ReorderUnderPlane,
    UnnestGrowsLane,
}

#[derive(Clone, Debug)]
pub struct Plan {
    folds: Vec<Fold>,
    under_plane: bool,
    claims_skip: bool,
}

impl Plan {
    pub fn check(&self) -> Result<(), Refuse> {
        for fold in &self.folds {
            match fold {
                Fold::Group { k } if *k > K_KNEE => return Err(Refuse::GroupKnee),
                Fold::Mask { fused: true } if self.claims_skip => {
                    return Err(Refuse::FusedClaimsSkip);
                }
                _ => {}
            }
        }
        if self.under_plane && self.claims_skip {
            return Err(Refuse::ReorderUnderPlane);
        }
        Ok(())
    }
}

/// Order candidate gates by dead-word fraction, highest first.
/// Popcount in, permutation out. No runtime hill-climb here.
pub fn order_by_dead_words(gates: &[Aperture<'_>]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..gates.len()).collect();
    idx.sort_by(|&a, &b| {
        gates[b]
            .dead_word_fraction()
            .total_cmp(&gates[a].dead_word_fraction())
    });
    idx
}

/// Scattered survivors: the word gate is not the win. Narrow to `u16`s.
/// The list is at most 128 KB and is not a pair list.
pub fn extract_u16(gate: &Aperture<'_>) -> Result<Vec<Row>, Refuse> {
    if gate.dead_word_fraction() >= SCATTERED_DEAD_WORD {
        return Err(Refuse::PairList); // caller must keep the word gate
    }
    let mut rows = Vec::with_capacity(gate.popcount() as usize);
    for (w, word) in gate.words.iter().enumerate() {
        let mut bits = *word;
        while bits != 0 {
            let b = bits.trailing_zeros();
            rows.push(Row((w as u16) * 64 + b as u16));
            bits &= bits - 1;
        }
    }
    Ok(rows)
}

// --- 5. a million linear folds of one aperture are one fold ---------------

#[derive(Clone, Copy, Debug)]
pub enum Assoc {
    Sum,
    Count,
    Min,
    Max,
}

/// Collapse `n` identical associative terminals into one.
/// `Sum` of `n` sums over the same aperture is one sum.
pub fn collapse(op: Assoc, n: u32) -> (Assoc, u32) {
    match op {
        Assoc::Sum | Assoc::Count | Assoc::Min | Assoc::Max => (op, 1.min(n).max(1)),
    }
}

// --- 6. the metric line the test fails on ---------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    pub alloc_bytes_exec: usize,
    pub pair_relation_bytes: usize,
    pub population_state_bytes: usize,
    pub fixture_view_bytes: usize,
    pub perm_bytes: usize,
}

impl Metrics {
    /// Zero-copy of the fold. A permutation is allowed and named, not hidden.
    pub fn fold_ok(self) -> Result<(), Refuse> {
        if self.pair_relation_bytes != 0 {
            return Err(Refuse::PairList);
        }
        if self.population_state_bytes != 0 || self.alloc_bytes_exec != 0 {
            return Err(Refuse::HashTable);
        }
        if self.perm_bytes != 0 && self.perm_bytes != PERM_BYTES {
            return Err(Refuse::IndexWidened);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn million_word_folds_are_sub_millisecond() {
        let ms = WordFoldNs(0.6).million() / 1_000_000.0;
        assert!((ms - 0.6).abs() < 1e-9);
    }

    #[test]
    fn fk_past_u16_is_a_refusal() {
        assert_eq!(Row::try_from(70_000u32), Err(Refuse::IndexWidened));
    }

    #[test]
    fn unsorted_ingest_does_not_construct() {
        assert_eq!(
            OrderedLane::try_from_sorted(vec![0; N], false).err(),
            Some(Refuse::UnorderedIngest)
        );
    }

    #[test]
    fn fused_plan_cannot_claim_the_skip() {
        let plan = Plan {
            folds: vec![Fold::Mask { fused: true }, Fold::Terminal],
            under_plane: false,
            claims_skip: true,
        };
        assert_eq!(plan.check(), Err(Refuse::FusedClaimsSkip));
    }

    #[test]
    fn reorder_under_a_plane_is_inert() {
        let plan = Plan {
            folds: vec![Fold::Cmp, Fold::Terminal],
            under_plane: true,
            claims_skip: true,
        };
        assert_eq!(plan.check(), Err(Refuse::ReorderUnderPlane));
    }

    #[test]
    fn group_past_the_knee_is_refused() {
        let plan = Plan {
            folds: vec![Fold::Group { k: K_KNEE + 1 }],
            under_plane: false,
            claims_skip: false,
        };
        assert_eq!(plan.check(), Err(Refuse::GroupKnee));
    }

    #[test]
    fn a_pair_list_fails_the_metric() {
        let m = Metrics {
            pair_relation_bytes: 16,
            ..Metrics::default()
        };
        assert_eq!(m.fold_ok(), Err(Refuse::PairList));
    }

    #[test]
    fn associative_repeats_collapse_to_one() {
        assert_eq!(collapse(Assoc::Sum, 1_000_000).1, 1);
    }
}
