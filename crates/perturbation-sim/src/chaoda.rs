//! **CAKES + CHAODA over HHTL-keyed basins** — the similarity (attraction) /
//! anomaly (repulsion) pair from the CLAM family, applied to grid resilience.
//!
//! The picture (cancer-detection / foveation framing): three layers cooperate —
//!
//! - **HHTL** = the semantic *family basin* (the GUID-key cascade, [`crate::hhtl`])
//!   — "**where** in the tree".
//! - **CAKES** ([`cakes_neighbors`]) = *attraction*: the `k` nearest basins by
//!   resilience-feature distance — "**who are my relatives / who looks similar**".
//! - **CHAODA** ([`chaoda_scores`]) = *repulsion*: a per-basin anomaly score = how
//!   far a basin sits from its family — "**who looks wrong / why am I different**".
//!   The high-anomaly basin is the **fail-first compartment**.
//!
//! `CAKES pulls in the similar + CHAODA pushes out the unusual = meaningful
//! intelligence`. The pair is **domain-agnostic** — the same two functions score
//! papillary muscles, terrain tiles, invoices, or grid basins; only the feature
//! adapter ([`resilience_basin_features`]) is grid-specific.
//!
//! Zero-dep, deterministic. This is a CHAODA-**lite** (a single kNN-distance
//! scorer), **not** ndarray's full `ClamTree::anomaly_scores` graph ensemble; the
//! `ndarray-clam` bridge (gated behind the `ndarray-simd` feature, since there is
//! no local ndarray sibling here) is the production path. The [`CHAODA_FLAG`]
//! threshold (0.75) mirrors ndarray's CLAM anomaly flag so the lite and full
//! scorers agree on the binary decision.

use crate::buffer::inertia_buffer_column;
use crate::graph::Grid;
use crate::hhtl::{basin_lambda2, hhtl_keys, HhtlKey};
use std::collections::BTreeMap;

/// The CHAODA anomaly flag threshold: a normalized score `≥ 0.75` is "anomalous"
/// (matches ndarray CLAM's `ClamTree::anomaly_scores` flag, so the lite scorer here
/// and the full ensemble agree on the binary call).
pub const CHAODA_FLAG: f64 = 0.75;

/// Euclidean distance between two feature rows (the metric both CAKES and CHAODA
/// read; rows must be equal length).
fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// **CAKES (attraction).** The `k` nearest rows to `query` by feature distance —
/// "who are my relatives". Returns `(index, distance)` ascending, excluding `query`
/// itself; `k` is clamped to the available population.
pub fn cakes_neighbors(rows: &[Vec<f64>], query: usize, k: usize) -> Vec<(usize, f64)> {
    if query >= rows.len() {
        return Vec::new();
    }
    let mut d: Vec<(usize, f64)> = rows
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != query)
        .map(|(i, r)| (i, dist(&rows[query], r)))
        .collect();
    d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    d.truncate(k.min(d.len()));
    d
}

/// **CHAODA (repulsion).** Per-row anomaly score = the mean distance to its `k`
/// nearest neighbors, min-max **normalized to `[0,1]`** across all rows. High = far
/// from its family = "why am I different" = the fail-first compartment. A row that
/// is feature-isolated scores near `1.0`; tight family members score near `0.0`.
///
/// This is the kNN-distance CHAODA-lite (one scorer), not the full ClamTree
/// ensemble. Degenerate (`< 2` rows, or all rows identical) yields all-zeros, never
/// `NaN`.
pub fn chaoda_scores(rows: &[Vec<f64>], k: usize) -> Vec<f64> {
    let n = rows.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let raw: Vec<f64> = (0..n)
        .map(|i| {
            let nbrs = cakes_neighbors(rows, i, k);
            if nbrs.is_empty() {
                0.0
            } else {
                nbrs.iter().map(|(_, d)| *d).sum::<f64>() / nbrs.len() as f64
            }
        })
        .collect();
    let lo = raw.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    raw.iter()
        .map(|&r| if span > 0.0 { (r - lo) / span } else { 0.0 })
        .collect()
}

/// CHAODA anomaly ranking: `(row index, score)` sorted by score descending — the
/// fail-first compartment is the head of the list.
pub fn anomaly_ranking(rows: &[Vec<f64>], k: usize) -> Vec<(usize, f64)> {
    let scores = chaoda_scores(rows, k);
    let mut r: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
    r.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    r
}

/// Build per-basin resilience feature rows from a grid + per-bus inertia. Each row
/// is `[λ₂_norm, size_norm, inertia_norm]` — the **topology / scale / buffer** axes
/// the CAKES/CHAODA pair reasons over (the three orthogonal resilience axes). Rows
/// are min-max normalized per axis across basins so the Euclidean metric weights
/// them comparably. Returns the basin keys (ordered) aligned with the rows.
///
/// `inertia_h` is per-bus inertia (length `grid.n`); when real `H` is unavailable a
/// deterministic proxy is fine — the *structure* (basins as families, the outlier as
/// fail-first) holds regardless (buffer ⊥ topology by the key/value split).
pub fn resilience_basin_features(
    grid: &Grid,
    inertia_h: &[f64],
    df_band: f64,
) -> (Vec<HhtlKey>, Vec<Vec<f64>>) {
    let keys = hhtl_keys(grid);
    let l2 = basin_lambda2(grid, &keys);
    let buf = inertia_buffer_column(inertia_h, df_band);

    // Aggregate per basin: (node count, summed normalized buffer).
    let mut agg: BTreeMap<(u16, u16, u16), (usize, f64)> = BTreeMap::new();
    for (n, key) in keys.iter().enumerate() {
        let e = agg.entry((key.heel, key.hip, key.twig)).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += *buf.get(n).unwrap_or(&0.0) as f64;
    }

    // Per-basin raw triples in deterministic key order.
    let mut basin_keys = Vec::with_capacity(agg.len());
    let mut raw: Vec<[f64; 3]> = Vec::with_capacity(agg.len());
    for (&(heel, hip, twig), &(count, buf_sum)) in &agg {
        let key = HhtlKey { heel, hip, twig };
        let lam = l2.get(&key).copied().unwrap_or(0.0);
        let mean_buf = if count > 0 {
            buf_sum / count as f64
        } else {
            0.0
        };
        basin_keys.push(key);
        raw.push([lam, count as f64, mean_buf]);
    }

    // Min-max normalize each of the 3 axes across basins.
    let mut rows: Vec<Vec<f64>> = raw.iter().map(|r| r.to_vec()).collect();
    for axis in 0..3 {
        let lo = raw.iter().map(|r| r[axis]).fold(f64::INFINITY, f64::min);
        let hi = raw
            .iter()
            .map(|r| r[axis])
            .fold(f64::NEG_INFINITY, f64::max);
        let span = hi - lo;
        if span > 0.0 {
            for (row, r) in rows.iter_mut().zip(&raw) {
                row[axis] = (r[axis] - lo) / span;
            }
        } else {
            for row in rows.iter_mut() {
                row[axis] = 0.0;
            }
        }
    }
    (basin_keys, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four tight family members + one far outlier. CHAODA must flag the outlier
    /// (score 1.0, ≥ flag) and keep the family low (the repulsion picture).
    #[test]
    fn chaoda_flags_the_isolated_outlier() {
        let rows = vec![
            vec![0.10, 0.10, 0.10],
            vec![0.12, 0.09, 0.11],
            vec![0.09, 0.11, 0.10],
            vec![0.11, 0.10, 0.09],
            vec![0.95, 0.95, 0.95], // the anomaly — far from the family
        ];
        let scores = chaoda_scores(&rows, 2);
        let rank = anomaly_ranking(&rows, 2);
        assert_eq!(rank[0].0, 4, "the outlier is the top anomaly");
        assert!((scores[4] - 1.0).abs() < 1e-9, "outlier normalizes to 1.0");
        assert!(scores[4] >= CHAODA_FLAG, "outlier crosses the CHAODA flag");
        for &i in &[0usize, 1, 2, 3] {
            assert!(scores[i] < CHAODA_FLAG, "family member {i} stays unflagged");
        }
    }

    /// CAKES (attraction): a query inside the family pulls in its family members,
    /// not the outlier — "who are my relatives".
    #[test]
    fn cakes_pulls_in_the_similar_not_the_outlier() {
        let rows = vec![
            vec![0.10, 0.10, 0.10],
            vec![0.12, 0.09, 0.11],
            vec![0.09, 0.11, 0.10],
            vec![0.95, 0.95, 0.95], // outlier
        ];
        let nbrs = cakes_neighbors(&rows, 0, 2);
        let idx: Vec<usize> = nbrs.iter().map(|(i, _)| *i).collect();
        assert!(
            idx.contains(&1) && idx.contains(&2),
            "relatives are the family"
        );
        assert!(!idx.contains(&3), "the outlier is NOT a relative");
    }

    /// Degenerate inputs never NaN / never panic.
    #[test]
    fn chaoda_degenerate_is_safe() {
        assert!(chaoda_scores(&[], 3).is_empty());
        assert_eq!(chaoda_scores(&[vec![1.0, 2.0]], 3), vec![0.0]);
        // All-identical rows → no span → all zeros.
        let same = vec![vec![0.5, 0.5], vec![0.5, 0.5], vec![0.5, 0.5]];
        assert!(chaoda_scores(&same, 2).iter().all(|&s| s == 0.0));
    }

    /// The resilience adapter: one row per distinct basin, every feature in [0,1],
    /// deterministic, and CHAODA runs over it. Two clean blocks joined by a weak
    /// bridge → 2 basins; a uniform-inertia proxy keeps the buffer axis flat so the
    /// split is carried by topology, as expected.
    #[test]
    fn resilience_adapter_builds_normalized_basin_rows() {
        // Two 4-cliques + a weak bridge (the hhtl two-block topology).
        let mut e = Vec::new();
        for (a, b) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
            e.push(crate::graph::Edge::new(a, b, 1.0, 1.0));
        }
        for (a, b) in [(4, 5), (4, 6), (5, 7), (6, 7)] {
            e.push(crate::graph::Edge::new(a, b, 1.0, 1.0));
        }
        e.push(crate::graph::Edge::new(3, 4, 0.01, 1.0));
        let grid = Grid::new(8, e);
        let h = vec![4.0; grid.n]; // uniform proxy inertia
        let (keys, rows) = resilience_basin_features(&grid, &h, 0.2);
        assert_eq!(keys.len(), rows.len(), "rows align with basins");
        assert!(keys.len() >= 2, "the weak bridge gives at least two basins");
        for r in &rows {
            assert_eq!(r.len(), 3, "[λ₂, size, inertia]");
            assert!(r.iter().all(|&x| (0.0..=1.0).contains(&x)), "normalized");
        }
        // Deterministic: same grid ⇒ same rows.
        let (_, rows2) = resilience_basin_features(&grid, &h, 0.2);
        assert_eq!(rows, rows2);
        // CHAODA runs and returns one score per basin in range.
        let scores = chaoda_scores(&rows, 1);
        assert_eq!(scores.len(), keys.len());
        assert!(scores.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }
}
