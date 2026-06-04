//! Substitution hierarchy + deep-position detection — transcoded from `fib.c`.
//!
//! Two operations:
//!
//! 1. [`build_hierarchy`] applies the inverse Fibonacci substitution (deflation)
//!    iteratively, merging each `(L, S)` pair at level `k` into a super-L at
//!    level `k+1`, and each isolated `L` into a super-S. Parent pointers are
//!    recorded so the level-0 tile can be walked back up to its eventual
//!    super-tile at any reachable level.
//!
//! 2. [`detect_deep_positions`] traces each level-0 L upward through the
//!    hierarchy. A tile qualifies as an entry point for an n-gram lookup at
//!    level `k` if (a) it is the leftmost child at every level from 0 to k,
//!    and (b) the spanning super-tile covers exactly the expected
//!    `HIER_WORD_LENS[k]` words.
//!
//! Plus a small context byte ([`hier_context`]) for arithmetic-coding models
//! that condition on the local hierarchy structure (transcoded as-is from the
//! reference).

use crate::constants::{HIER_WORD_LENS, MAX_HIER, N_LEVELS};
use crate::types::{DeepPositions, HLevel, Hierarchy, ParentMap, Tile};

/// Build the substitution hierarchy from a flat tile sequence.
///
/// Direct transcode of `build_hierarchy` in `fib.c`. Iterates the deflation
/// rule `(L, S) → super-L`, `L → super-S` up to `max_levels` times (with an
/// internal cap at [`MAX_HIER`]).
///
/// Returns the hierarchy with `n_levels()` tracking the number of populated
/// levels including the tile level (level 0). On Fibonacci tilings,
/// `n_levels()` grows as `⌊log_φ(n_tiles)⌋`; on collapsing periodic tilings
/// (e.g. Period-5), the level count saturates within a few hops (paper Thm 3).
#[must_use]
pub fn build_hierarchy(tiles: &[Tile], max_levels: usize) -> Hierarchy {
    let mut hier = Hierarchy::empty();
    let cap = max_levels.min(MAX_HIER);

    // Level 0: copy tile-level structure.
    let level0: Vec<HLevel> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| HLevel {
            start: i as u32,
            end: (i + 1) as u32,
            is_l: t.is_l,
        })
        .collect();
    hier.levels.push(level0);

    // Iteratively deflate.
    for lvl in 0..cap {
        let prev = &hier.levels[lvl];
        if prev.len() < 2 {
            break;
        }

        let mut pmap = vec![
            ParentMap {
                parent_idx: None,
                pos: -1
            };
            prev.len()
        ];
        let mut cur = Vec::with_capacity(prev.len());
        let mut i = 0;

        while i < prev.len() {
            if i + 1 < prev.len() && prev[i].is_l && !prev[i + 1].is_l {
                // (L, S) → super-L spanning both.
                let ci = cur.len() as u32;
                cur.push(HLevel {
                    start: prev[i].start,
                    end: prev[i + 1].end,
                    is_l: true,
                });
                pmap[i] = ParentMap {
                    parent_idx: Some(ci),
                    pos: 0,
                };
                pmap[i + 1] = ParentMap {
                    parent_idx: Some(ci),
                    pos: 1,
                };
                i += 2;
            } else {
                // Isolated L → super-S, or an S that has no L companion.
                let ci = cur.len() as u32;
                cur.push(HLevel {
                    start: prev[i].start,
                    end: prev[i].end,
                    is_l: false,
                });
                pmap[i] = ParentMap {
                    parent_idx: Some(ci),
                    pos: 0,
                };
                i += 1;
            }
        }

        hier.parent_maps.push(pmap);
        hier.levels.push(cur);
    }

    hier
}

/// 3-bit hierarchy context — direct transcode of `get_hier_ctx` in `fib.c`.
///
/// Used by the C reference as an arithmetic-coding context conditioner
/// (8 specialised sub-models per tile type). Mixes the tile's own L-flag with
/// up to three ancestors' positions and L-flags via a deterministic hash.
#[must_use]
pub fn hier_context(tile_idx: u32, hier: &Hierarchy) -> u8 {
    let mut h: u8 = u8::from(hier.levels[0][tile_idx as usize].is_l);
    let mut idx: u32 = tile_idx;
    let max_climb = hier.n_levels().saturating_sub(1).min(3);

    for k in 0..max_climb {
        let pm_level = match hier.parent_maps.get(k) {
            Some(p) if (idx as usize) < hier.levels[k].len() => p,
            _ => break,
        };
        let Some(parent_idx) = pm_level[idx as usize].parent_idx else {
            break;
        };
        let pos = pm_level[idx as usize].pos as u8;
        let parent_l = u8::from(hier.levels[k + 1][parent_idx as usize].is_l);
        h = (h
            .wrapping_mul(5)
            .wrapping_add(pos.wrapping_mul(3))
            .wrapping_add(parent_l))
            & 0x07;
        idx = parent_idx;
    }
    h
}

/// Detect deep n-gram entry points across the hierarchy.
///
/// Direct transcode of `detect_deep_positions` in `fib.c`. A tile `ti` at
/// level 0 is a legal n-gram entry at hierarchy level `k+1` if:
///
/// - it is an L,
/// - it is the position-0 child of its ancestor at every level from 0 to k,
/// - the ancestor at level `k+1` is itself a super-L,
/// - the ancestor at level `k+1` spans exactly `HIER_WORD_LENS[k+1]` words.
///
/// Implementation note: at `k == 0` the C reference adds a guard that the
/// next level-0 tile must also be an L (matching the (L, S) → super-L pattern
/// that should have already been formed; the guard prevents level-2 lookups
/// at positions that haven't reached the right shape yet).
///
/// Returns per-level boolean masks and the skip counts.
#[must_use]
pub fn detect_deep_positions(tiles: &[Tile], hier: &Hierarchy) -> DeepPositions {
    let max_k = hier.n_levels().saturating_sub(1).min(MAX_HIER);
    let n = tiles.len();

    // Allocate `max_k + 1` slots so callers can index `can[k]` for `k = 0..=max_k`.
    // `can[0]` and `skip[0]` are unused (deep positions start at level 1)
    // but kept for index parity with the C reference.
    let mut can: Vec<Vec<bool>> = (0..=max_k).map(|_| vec![false; n]).collect();
    let mut skip: Vec<Vec<u32>> = (0..=max_k).map(|_| vec![0u32; n]).collect();

    for ti in 0..n {
        if !tiles[ti].is_l {
            continue;
        }
        let mut idx_k: u32 = ti as u32;

        for k in 0..max_k {
            let pm_level = match hier.parent_maps.get(k) {
                Some(p) if (idx_k as usize) < hier.levels[k].len() => p,
                _ => break,
            };
            let Some(parent_idx) = pm_level[idx_k as usize].parent_idx else {
                break;
            };
            if pm_level[idx_k as usize].pos != 0 {
                break;
            }
            let parent = &hier.levels[k + 1][parent_idx as usize];
            if !parent.is_l {
                break;
            }

            // Word coverage of the super-tile (sum of nwords across its
            // descendants at level 0).
            let st = parent.start as usize;
            let en = parent.end as usize;
            let nw_cov: u32 = tiles[st..en].iter().map(|t| u32::from(t.nwords)).sum();

            let expected = if k + 1 < N_LEVELS + 1 {
                Some(HIER_WORD_LENS[k + 1])
            } else {
                None
            };

            if let Some(want) = expected {
                if nw_cov as usize == want {
                    if k == 0 {
                        // C reference: skip recording at k==0 when the next
                        // level-0 tile is also an L (the L-S form hasn't
                        // closed at this position yet).
                        if ti + 1 >= n || tiles[ti + 1].is_l {
                            idx_k = parent_idx;
                            continue;
                        }
                    }
                    can[k + 1][ti] = true;
                    skip[k + 1][ti] = (en - st - 1) as u32;
                }
            }
            idx_k = parent_idx;
        }
    }

    DeepPositions { can, skip, max_k }
}

/// Count usable deep entry points at each hierarchy level.
///
/// Returns a vector of length `max_k + 1` where `counts[k]` is the number of
/// level-0 tiles that qualify as n-gram entries at hierarchy level `k`.
/// Level 0 is always 0 (entries start at level 1).
#[must_use]
pub fn deep_counts(dp: &DeepPositions) -> Vec<usize> {
    dp.can
        .iter()
        .map(|level| level.iter().filter(|&&b| b).count())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::PHI;
    use crate::tiling::{period5_tiling, qc_word_tiling};

    #[test]
    fn golden_hierarchy_grows_deep() {
        // For N tiles, hierarchy depth ≈ ⌊log_φ N⌋.
        let tiles = qc_word_tiling(10_000, 0.0);
        let hier = build_hierarchy(&tiles, MAX_HIER);
        // log_φ(10_000) ≈ 19.1, but the deflation halves count per level so we
        // expect depth ≈ log_φ(n_tiles) where n_tiles ≈ 0.62 * 10_000 = 6_200.
        // log_φ(6_200) ≈ 18.1 — capped at MAX_HIER = 10.
        assert!(hier.n_levels() >= 8, "depth = {}", hier.n_levels());
    }

    #[test]
    fn period5_hierarchy_count_decays() {
        // Periodic tilings still build a hierarchy structurally, but their
        // tile counts at higher levels become trivial (one or both types
        // vanish). At the very least the count should drop dramatically.
        let tiles = period5_tiling(1_000);
        let hier = build_hierarchy(&tiles, MAX_HIER);

        // Verify per-level counts decrease.
        let counts: Vec<usize> = (0..hier.n_levels()).map(|k| hier.level_count(k)).collect();
        for w in counts.windows(2) {
            assert!(w[1] <= w[0], "non-monotone {counts:?}");
        }
    }

    #[test]
    fn golden_l_density_at_level_0() {
        // freq(L) on the golden tiling = φ/(φ+1) ≈ 0.618 (Perron-Frobenius).
        let tiles = qc_word_tiling(100_000, 0.0);
        let n_l = tiles.iter().filter(|t| t.is_l).count() as f64;
        let n_total = tiles.len() as f64;
        let density = n_l / n_total;
        let expected = PHI / (PHI + 1.0);
        assert!(
            (density - expected).abs() < 0.005,
            "density = {density}, expected = {expected}"
        );
    }

    #[test]
    fn hierarchy_context_is_in_range() {
        let tiles = qc_word_tiling(1_000, 0.0);
        let hier = build_hierarchy(&tiles, MAX_HIER);
        for i in 0..(tiles.len() as u32).min(100) {
            let h = hier_context(i, &hier);
            assert!(h < 8, "context = {h}");
        }
    }

    #[test]
    fn deep_positions_nonempty_at_low_levels_on_golden() {
        let tiles = qc_word_tiling(10_000, 0.0);
        let hier = build_hierarchy(&tiles, MAX_HIER);
        let dp = detect_deep_positions(&tiles, &hier);
        let counts = deep_counts(&dp);

        // Level 1 (trigram entries) should be plentiful.
        assert!(counts[1] > 100, "level-1 counts = {}", counts[1]);
        // Higher levels should have decreasing but still nonzero counts.
        let later: Vec<usize> = counts.iter().skip(1).take(4).copied().collect();
        for w in later.windows(2) {
            assert!(w[1] <= w[0], "non-monotone deep counts {later:?}");
        }
    }

    #[test]
    fn aperiodic_hierarchy_advantage_emerges() {
        // Paper Thm 1: at corpus scales beyond the periodic-collapse depth,
        // Fibonacci provides deep n-gram positions that Period-5 cannot.
        let n_words = 10_000u32;
        let golden = qc_word_tiling(n_words, 0.0);
        let period5 = period5_tiling(n_words);

        let g_hier = build_hierarchy(&golden, MAX_HIER);
        let p_hier = build_hierarchy(&period5, MAX_HIER);

        let g_dp = detect_deep_positions(&golden, &g_hier);
        let p_dp = detect_deep_positions(&period5, &p_hier);

        let g_counts = deep_counts(&g_dp);
        let p_counts = deep_counts(&p_dp);

        // Paper Cor 4: Period-5 collapses around k* ≈ log(5)/log(φ) ≈ 3.3.
        // Beyond that point Period-5 should be unable to produce deep entries.
        let g_deep: usize = g_counts.iter().skip(4).sum();
        let p_deep: usize = p_counts.iter().skip(4).sum();
        assert!(
            g_deep > p_deep,
            "golden deep ≥ 5 = {g_deep}, period5 deep ≥ 5 = {p_deep}"
        );
    }
}
