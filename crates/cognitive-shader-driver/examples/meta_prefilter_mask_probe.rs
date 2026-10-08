//! PROBE-PREFILTER-MASK
//! (`.claude/board/entries/2026-10-08-coresearch-ce64-moore-masking-wiring.md`).
//!
//! `BindSpace::meta_prefilter` walks rows one at a time and returns a dense
//! `Vec<u32>` of row indices: a selection vector, which mask-risc's A1 does
//! not admit. This probe asks whether the same filter can be computed as a
//! row MASK using only `ndarray::simd` masking ops, with no field extraction.
//!
//! The lowering: every `MetaFilter` clause is a predicate on one packed
//! sub-field of the `MetaWord` (`u32`). A threshold `x >= k` on an n-bit field
//! is a disjoint union of at most n + 1 ternary patterns on the packed word:
//! `x == k`, plus, for each bit i where `k` has a 0, "agrees with k above i,
//! has a 1 at i". `x <= k` is the mirror. The style bitset is one equality
//! pattern per admitted style. So the whole filter is OR (within a clause)
//! and AND (across clauses) of `ternary_match_u32_to_mask_under` calls, each
//! gated by the clauses already applied.
//!
//! Claim: the mask's set rows equal `meta_prefilter`'s output exactly.

use cognitive_shader_driver::bindspace::BindSpace;
use lance_graph_contract::cognitive_shader::{ColumnWindow, MetaFilter, MetaWord};
use ndarray::simd::{mask_or_assign, mask_set_range, ternary_match_u32_to_mask_under};

/// One packed sub-field of `MetaWord`.
#[derive(Clone, Copy)]
struct Field {
    shift: u32,
    width: u32,
}

const THINKING: Field = Field { shift: 0, width: 6 };
const AWARENESS: Field = Field { shift: 6, width: 4 };
const NARS_F: Field = Field {
    shift: 10,
    width: 8,
};
const NARS_C: Field = Field {
    shift: 18,
    width: 8,
};
const FREE_E: Field = Field {
    shift: 26,
    width: 6,
};

impl Field {
    fn care_from(self, bit: u32) -> u32 {
        // Field bits at or above `bit`, in packed position.
        let field = ((1u32 << self.width) - 1) << self.shift;
        field & !((1u32 << (self.shift + bit)) - 1)
    }
    fn at(self, v: u32) -> u32 {
        v << self.shift
    }
}

/// Ternary patterns whose union is `x >= k` (`at_least`) or `x <= k`.
fn threshold_patterns(f: Field, k: u32, at_least: bool) -> Vec<(u32, u32)> {
    let mut out = vec![(f.at(k), f.care_from(0))]; // x == k
    for i in 0..f.width {
        let ki = (k >> i) & 1;
        // ">=" branches where k has 0 (x takes 1); "<=" where k has 1 (x takes 0).
        if (at_least && ki == 0) || (!at_least && ki == 1) {
            let above = (k >> (i + 1)) << (i + 1);
            let bit = if at_least { 1u32 << i } else { 0 };
            out.push((f.at(above | bit), f.care_from(i)));
        }
    }
    out
}

/// Narrow `running` to rows matching any of `patterns`.
fn apply_clause(values: &[u32], patterns: &[(u32, u32)], running: &mut Vec<u64>) {
    let mut acc = vec![0u64; running.len()];
    let mut tmp = vec![0u64; running.len()];
    for &(pattern, care) in patterns {
        ternary_match_u32_to_mask_under(values, pattern, care, running, &mut tmp);
        mask_or_assign(&mut acc, &tmp);
    }
    *running = acc;
}

/// `MetaFilter` lowered to a row mask over `values[start..end)`.
pub fn meta_filter_mask(values: &[u32], win: ColumnWindow, f: &MetaFilter) -> Vec<u64> {
    let n = values.len();
    let mut running = vec![0u64; n.div_ceil(64)];
    let (start, end) = ((win.start as usize).min(n), (win.end as usize).min(n));
    if start < end {
        mask_set_range(&mut running, start, end);
    }
    if f.thinking_mask != 0 {
        let styles: Vec<(u32, u32)> = (0..64u32)
            .filter(|t| f.thinking_mask & (1u64 << t) != 0)
            .map(|t| (THINKING.at(t), THINKING.care_from(0)))
            .collect();
        apply_clause(values, &styles, &mut running);
    }
    let clauses = [
        (AWARENESS, f.awareness_min as u32, true),
        (NARS_F, f.nars_f_min as u32, true),
        (NARS_C, f.nars_c_min as u32, true),
        (FREE_E, f.free_e_max as u32, false),
    ];
    for (field, k, at_least) in clauses {
        let full = (1u32 << field.width) - 1;
        // `MetaFilter` bounds are `u8`; a field can be narrower. A lower bound
        // above the field's maximum admits nothing, and must not reach the
        // patterns, whose care mask would drop its high bits.
        if at_least && k > full {
            running.iter_mut().for_each(|w| *w = 0);
            break;
        }
        let vacuous = if at_least { k == 0 } else { k >= full };
        if !vacuous {
            apply_clause(
                values,
                &threshold_patterns(field, k, at_least),
                &mut running,
            );
        }
    }
    running
}

fn set_rows(mask: &[u64]) -> Vec<u32> {
    let mut out = Vec::new();
    for (wi, &w) in mask.iter().enumerate() {
        let mut w = w;
        while w != 0 {
            out.push((wi * 64) as u32 + w.trailing_zeros());
            w &= w - 1;
        }
    }
    out
}

/// SplitMix64, deterministic.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn space(rows: usize, seed: u64) -> BindSpace {
    let mut bs = BindSpace::zeros(rows);
    let mut r = Rng(seed);
    for row in 0..rows {
        bs.meta.set(row, MetaWord(r.next() as u32));
    }
    bs
}

fn filters(seed: u64, count: usize) -> Vec<MetaFilter> {
    let mut r = Rng(seed);
    (0..count)
        .map(|_| MetaFilter {
            thinking_mask: if r.next() % 3 == 0 { 0 } else { r.next() },
            // Full `u8` domain: the field is 4 bits, so bounds >= 16 must
            // reject every row (Codex review, PR #1405).
            awareness_min: if r.next() % 4 == 0 {
                r.next() as u8
            } else {
                (r.next() % 16) as u8
            },
            nars_f_min: (r.next() % 256) as u8,
            nars_c_min: (r.next() % 256) as u8,
            free_e_max: if r.next() % 4 == 0 {
                r.next() as u8
            } else {
                (r.next() % 64) as u8
            },
        })
        .collect()
}

fn main() {
    let rows = 4099; // not a multiple of 64: exercises the tail word
    let bs = space(rows, 7);
    let mut checked = 0usize;
    let mut selective = 0usize;
    for (fi, f) in filters(11, 400).iter().enumerate() {
        let win = ColumnWindow::new((fi % 37) as u32, (rows - fi % 53) as u32);
        let want = bs.meta_prefilter(win, f);
        let got = set_rows(&meta_filter_mask(&bs.meta.0, win, f));
        assert_eq!(got, want, "filter {fi}: {f:?}");
        checked += 1;
        if want.len() * 3 < rows {
            selective += 1;
        }
    }
    println!("PROBE-PREFILTER-MASK: {checked} filters x {rows} rows identical; {selective} admit < 1/3 of rows");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FAILS IF: the mask lowering selects any row set different from
    /// `meta_prefilter`, on random words and random filters.
    #[test]
    fn mask_lowering_equals_meta_prefilter() {
        let rows = 1031;
        let bs = space(rows, 3);
        let mut selective = 0;
        for (fi, f) in filters(5, 200).iter().enumerate() {
            let win = ColumnWindow::new((fi % 7) as u32, (rows - fi % 11) as u32);
            let want = bs.meta_prefilter(win, f);
            assert_eq!(set_rows(&meta_filter_mask(&bs.meta.0, win, f)), want);
            if want.len() * 3 < rows {
                selective += 1;
            }
        }
        // Anti-vacuity: most filters must actually exclude most rows.
        assert!(selective > 100, "only {selective} selective filters");
    }

    /// A lower bound above a narrow field's maximum admits nothing, exactly
    /// as `MetaFilter::accepts` does (awareness is 4 bits; the bound is `u8`).
    #[test]
    fn an_out_of_range_lower_bound_admits_nothing() {
        let bs = space(500, 13);
        let win = ColumnWindow::new(0, 500);
        for awareness_min in [16u8, 17, 200, 255] {
            let f = MetaFilter {
                awareness_min,
                ..MetaFilter::ALL
            };
            assert!(bs.meta_prefilter(win, &f).is_empty());
            assert!(set_rows(&meta_filter_mask(&bs.meta.0, win, &f)).is_empty());
        }
        // Boundary: 15 is the field maximum and still admits rows.
        let f = MetaFilter {
            awareness_min: 15,
            ..MetaFilter::ALL
        };
        let want = bs.meta_prefilter(win, &f);
        assert!(!want.is_empty());
        assert_eq!(set_rows(&meta_filter_mask(&bs.meta.0, win, &f)), want);
    }

    /// Silent arm: `MetaFilter::ALL` sets every in-range row and no other.
    #[test]
    fn the_all_filter_sets_exactly_the_window() {
        let bs = space(300, 9);
        let win = ColumnWindow::new(5, 290);
        let got = set_rows(&meta_filter_mask(&bs.meta.0, win, &MetaFilter::ALL));
        assert_eq!(got, (5u32..290).collect::<Vec<_>>());
    }

    /// Each threshold decomposition, checked exhaustively over its field.
    #[test]
    fn threshold_patterns_cover_exactly_the_threshold() {
        for field in [AWARENESS, FREE_E, NARS_C] {
            let full = 1u32 << field.width;
            for k in 0..full {
                for at_least in [true, false] {
                    let pats = threshold_patterns(field, k, at_least);
                    for x in 0..full {
                        let w = field.at(x);
                        let hits = pats.iter().filter(|&&(p, c)| (w ^ p) & c == 0).count();
                        let want = if at_least { x >= k } else { x <= k };
                        assert_eq!(hits, want as usize, "k={k} x={x} at_least={at_least}");
                    }
                }
            }
        }
    }
}
