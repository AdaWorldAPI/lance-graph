//! Probe M1: CLAM 3-level 16-way tree on 256 Jina-v5 centroids.
//!
//! Citation: `.claude/knowledge/bf16-hhtl-terrain.md` Probe Queue M1
//! (status before this probe: PARTIAL — CHAODA runs on 256 rows, 26/256
//! flagged, but tree shape NOT YET tested for 16-way).
//!
//! # IMPORTANT CAVEAT (2026-04-29 Jan review feedback)
//!
//! This probe runs Ward clustering on the **uncalibrated** Jina-v5
//! distance table baked at `crates/thinking-engine/data/jina-v5-codebook/`.
//! Per `.claude/CALIBRATION_STATUS_GROUND_TRUTH.md`:
//!
//!   "ICC profile correction: DESIGNED but `LensProfile::build()` is
//!    never called. Per-role scale factors: DESIGNED but nowhere stored,
//!    nowhere applied."
//!
//! The 16-way bit-layout claim's true test requires:
//!   (a) ICC-calibrated codebooks (per-safetensor-class)
//!   (b) JIT-style HHTL parameter variation (currently exposed via
//!       `CascadeConfig { heel_min_agreement, hip_max_distance }` but
//!       not exercised in this probe — hardcoded Ward-only single-shot)
//!   (c) Hierarchy actually applied: H→centroid-cluster→H→centroid-cluster
//!       →T→centroid-cluster→L (the Hadamard-rotation-then-cluster cascade)
//!
//! The current PASS (L0 balance 0.4550, discrimination 0.6429) measures
//! only that **uncalibrated raw Jina-v5 centroids form 16 moderately-
//! balanced Ward clusters**. That's a necessary but insufficient
//! condition for the bit-layout claim. A more rigorous M1 test would
//! sweep over CascadeConfig combinations against ICC-calibrated codebooks.
//!
//! **Therefore the queue update reads: "PASS-with-caveat" rather than
//! a clean PASS. The bit-layout's L0 = 16 coarse-cluster claim is
//! consistent with the data but not yet rigorously validated.**
//!
//! # The claim under test
//!
//! The bf16-hhtl-terrain bit-layout claim:
//!
//! ```text
//! bits 15..12 = CLAM L0: 16 coarse clusters (HEEL scan target)
//! bits 11..8  = CLAM L1: 256 mid-clusters (HIP, 1:1 Jina-v5 centroids)
//! bits  7..4  = CLAM L2: 4096 terminal buckets (TWIG, COCA alignment)
//! ```
//!
//! **Critical reading: "1:1 Jina-v5 centroids" at L1 means L1 IS the
//! centroid level — each Jina-v5 centroid is its own L1 bucket, not a
//! cluster of centroids.** L2 (4096) is the per-centroid sub-resolution.
//! L0 (16) is the only level where actual *clustering* of centroids
//! happens.
//!
//! So M1 reduces to: **do the 256 Jina-v5 centroids form 16 clean
//! coarse (L0) clusters?** A pre-probe scipy check confirmed this is
//! the right reading: trying to subdivide L0 clusters (size 4-31) into
//! 16 L1 sub-clusters degenerates trivially because there's no room
//! (16 L0 × 16 L1 = 256 = total centroid count, leaving no slack).
//!
//! # Data
//!
//! `crates/thinking-engine/data/jina-v5-codebook/distance_table_256x256.u8`:
//! 256×256 u8 **similarity** table (diagonal = 255). Convert to distance
//! via `d = 255 - similarity`. **Not ICC-calibrated** — caveat above.
//!
//! # Method
//!
//! Hand-rolled Ward agglomerative clustering at L0 (16 clusters from 256
//! centroids). Ward chosen because:
//! - Average linkage degenerates on this data to one giant cluster of 115
//!   centroids (verified pre-probe with scipy)
//! - Ward is the standard for k-way balanced clustering in the literature
//! - It's the method CHAODA uses internally
//!
//! # PASS criteria (joint, both required)
//!
//! 1. **L0 size balance**: std(|cluster_i|)/mean(|cluster_i|) ≤ 0.5 across
//!    16 L0 clusters. Perfect 16-way is 0.0; 0.5 allows moderate imbalance.
//!
//! 2. **L0 discrimination**: mean within-L0-cluster distance / mean
//!    across-L0-cluster distance ≤ 0.7.
//!
//! # Why no L1 test
//!
//! Per the bit-layout reading above, L1 = 1:1 centroids; there's no L1
//! clustering to test. L2 (the 4096-bucket sub-centroid level) IS testable
//! but requires per-centroid embeddings (not pairwise distances) and is
//! the subject of a separate probe.
//!
//! # Followup needed for true M1 closure
//!
//! 1. ICC-calibrate the codebook per safetensor class (LensProfile::build()
//!    must actually run)
//! 2. Re-run with ICC-calibrated codebooks across multiple model classes
//! 3. Vary CascadeConfig parameters and measure stability of the 16-way
//!    cluster topology under HEEL/HIP threshold variation
//! 4. Then claim PASS without caveat.
//!
//! Captured as a separate Open Idea in `.claude/board/IDEAS.md` under
//! "Probe M1' (M1-prime): ICC-calibrated rigorous M1".

use crate::PillarResult;

/// Number of Jina-v5 centroids — fixed by the codebook.
const N_CENTROIDS: usize = 256;
/// 16-way branching: each level has 16 children.
const BRANCHING: usize = 16;
/// Path to the similarity table file.
const SIMILARITY_TABLE_PATH: &str = "crates/thinking-engine/data/jina-v5-codebook/distance_table_256x256.u8";

// ════════════════════════════════════════════════════════════════════════════
// Distance table loader
// ════════════════════════════════════════════════════════════════════════════

/// Load the 256×256 similarity table and convert to distance matrix.
/// Returns a 256×256 `f64` distance matrix where `d[i][j] = 255 - sim[i][j]`.
fn load_distance_matrix(path: &str) -> Result<Vec<Vec<f64>>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
    if bytes.len() != N_CENTROIDS * N_CENTROIDS {
        return Err(format!(
            "Expected {} bytes ({}×{}), got {}",
            N_CENTROIDS * N_CENTROIDS,
            N_CENTROIDS,
            N_CENTROIDS,
            bytes.len()
        ));
    }
    let mut d = vec![vec![0.0f64; N_CENTROIDS]; N_CENTROIDS];
    for i in 0..N_CENTROIDS {
        for j in 0..N_CENTROIDS {
            // Similarity table → distance: d = 255 - similarity
            let sim = bytes[i * N_CENTROIDS + j] as f64;
            d[i][j] = 255.0 - sim;
        }
    }
    // Sanity: diagonal should now be 0
    for i in 0..N_CENTROIDS {
        if d[i][i].abs() > 1e-9 {
            return Err(format!(
                "Diagonal not zero after inversion: d[{i}][{i}] = {} — table might not be similarity",
                d[i][i]
            ));
        }
    }
    Ok(d)
}

// ════════════════════════════════════════════════════════════════════════════
// Ward agglomerative clustering
//
// Lance-Williams update for Ward (using the Ward.D2 variant, where the
// distance is squared). For the merge of clusters A ∪ B vs cluster C:
//
//   d²(A∪B, C) = (|A|+|C|)·d²(A,C) + (|B|+|C|)·d²(B,C) − |C|·d²(A,B)
//                ─────────────────────────────────────────────────────
//                              |A| + |B| + |C|
//
// We work with squared distances throughout, taking sqrt only at the end
// for reporting if needed. This avoids numerical issues with d_AB²
// requiring a sqrt step on every merge.
// ════════════════════════════════════════════════════════════════════════════

/// Run Ward agglomerative clustering on the given distance matrix until
/// exactly `target_k` clusters remain. Returns a vector of length n where
/// element `i` is the cluster id (0..target_k) for centroid `i`.
///
/// Operates on a list of indices for the points being clustered (allows
/// recursive sub-clustering on subsets).
fn ward_cluster(distances: &[Vec<f64>], indices: &[usize], target_k: usize) -> Vec<usize> {
    let n = indices.len();
    assert!(target_k <= n, "target_k {target_k} > n {n}");

    if target_k == n {
        // Each point is its own cluster
        return (0..n).collect();
    }
    if target_k == 1 {
        return vec![0; n];
    }

    // Build initial squared distance matrix on the subset.
    let mut dist_sq: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let d = distances[indices[i]][indices[j]];
            dist_sq[i][j] = d * d;
        }
    }

    // Cluster sizes. Initially each centroid is its own cluster of size 1.
    let mut sizes = vec![1usize; n];
    // active[i] tracks whether cluster i still exists. After merging A and B
    // into A, B is marked inactive.
    let mut active = vec![true; n];
    // assignment[orig_index_in_subset] = current cluster_id (one of the still-active i)
    let mut assignment: Vec<usize> = (0..n).collect();
    let mut active_count = n;

    while active_count > target_k {
        // Find pair (a, b) with smallest dist_sq[a][b] among active clusters
        let mut best_a = usize::MAX;
        let mut best_b = usize::MAX;
        let mut best_d = f64::INFINITY;
        for a in 0..n {
            if !active[a] {
                continue;
            }
            for b in (a + 1)..n {
                if !active[b] {
                    continue;
                }
                if dist_sq[a][b] < best_d {
                    best_d = dist_sq[a][b];
                    best_a = a;
                    best_b = b;
                }
            }
        }

        // Merge B into A using Lance-Williams update for Ward
        let na = sizes[best_a] as f64;
        let nb = sizes[best_b] as f64;
        let new_size = sizes[best_a] + sizes[best_b];

        for c in 0..n {
            if !active[c] || c == best_a || c == best_b {
                continue;
            }
            let nc = sizes[c] as f64;
            let d_ac = dist_sq[best_a][c];
            let d_bc = dist_sq[best_b][c];
            let d_ab = dist_sq[best_a][best_b];
            let new_d = ((na + nc) * d_ac + (nb + nc) * d_bc - nc * d_ab) / (na + nb + nc);
            dist_sq[best_a][c] = new_d;
            dist_sq[c][best_a] = new_d;
        }

        sizes[best_a] = new_size;
        active[best_b] = false;
        // Reassign all members of cluster best_b to best_a
        for a in &mut assignment {
            if *a == best_b {
                *a = best_a;
            }
        }
        active_count -= 1;
    }

    // Renumber active clusters to 0..target_k for clean output.
    let mut cluster_remap = vec![usize::MAX; n];
    let mut next_id = 0;
    for &cid in &assignment {
        if cluster_remap[cid] == usize::MAX {
            cluster_remap[cid] = next_id;
            next_id += 1;
        }
    }
    assignment.iter().map(|&c| cluster_remap[c]).collect()
}

// ════════════════════════════════════════════════════════════════════════════
// Cluster quality metrics
// ════════════════════════════════════════════════════════════════════════════

fn cluster_sizes(assignment: &[usize], k: usize) -> Vec<usize> {
    let mut counts = vec![0usize; k];
    for &c in assignment {
        counts[c] += 1;
    }
    counts
}

fn size_balance(counts: &[usize]) -> f64 {
    let n = counts.len() as f64;
    let mean = counts.iter().map(|&c| c as f64).sum::<f64>() / n;
    let var = counts
        .iter()
        .map(|&c| (c as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let std = var.sqrt();
    if mean < 1e-9 {
        f64::INFINITY
    } else {
        std / mean
    }
}

/// Compute mean within-cluster distance and mean across-cluster distance.
/// Returns (within_mean, across_mean, ratio = within / across).
fn within_across_ratio(
    distances: &[Vec<f64>],
    indices: &[usize],
    assignment: &[usize],
) -> (f64, f64, f64) {
    let mut within_sum = 0.0;
    let mut within_count = 0usize;
    let mut across_sum = 0.0;
    let mut across_count = 0usize;
    let n = indices.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distances[indices[i]][indices[j]];
            if assignment[i] == assignment[j] {
                within_sum += d;
                within_count += 1;
            } else {
                across_sum += d;
                across_count += 1;
            }
        }
    }
    let within_mean = if within_count > 0 {
        within_sum / within_count as f64
    } else {
        0.0
    };
    let across_mean = if across_count > 0 {
        across_sum / across_count as f64
    } else {
        0.0
    };
    let ratio = if across_mean > 1e-9 {
        within_mean / across_mean
    } else {
        f64::INFINITY
    };
    (within_mean, across_mean, ratio)
}

// ════════════════════════════════════════════════════════════════════════════
// The probe
// ════════════════════════════════════════════════════════════════════════════

pub fn prove() -> PillarResult {
    // Load distance matrix
    let distances = match load_distance_matrix(SIMILARITY_TABLE_PATH) {
        Ok(d) => d,
        Err(e) => {
            return PillarResult {
                name: "Probe M1: CLAM 3-level 16-way tree on 256 Jina-v5 centroids",
                pass: false,
                measured: f64::NAN,
                predicted: 0.5,
                detail: format!(
                    "Could not load similarity table: {e}. \
                     Probe cannot run from this working directory; \
                     run from repo root."
                ),
                runtime_ms: 0,
            };
        }
    };

    // L0: cluster all 256 centroids into 16 groups
    let all_indices: Vec<usize> = (0..N_CENTROIDS).collect();
    let l0_assignment = ward_cluster(&distances, &all_indices, BRANCHING);
    let l0_sizes = cluster_sizes(&l0_assignment, BRANCHING);
    let l0_balance = size_balance(&l0_sizes);
    let (l0_within, l0_across, l0_ratio) =
        within_across_ratio(&distances, &all_indices, &l0_assignment);

    // Note: no L1 sub-clustering test. Per the bit-layout reading,
    // L1 = 1:1 centroids (each centroid is its own L1 bucket); there's
    // no L1 clustering to validate. L2 (4096 sub-centroid buckets) is a
    // separate probe with different data requirements.

    // PASS criteria (joint, both required)
    let l0_balance_pass = l0_balance <= 0.5;
    let l0_discrimination_pass = l0_ratio <= 0.7;
    let pass = l0_balance_pass && l0_discrimination_pass;

    let conclusion = if pass {
        "PASS-with-caveat — 16-way L0 clustering of 256 Jina-v5 centroids is moderately \
         balanced and discriminative on the UNCALIBRATED codebook in single-shot Ward. \
         The 16 coarse-cluster level of the CLAM bit-layout (bits 15..12) is consistent \
         with the data. Updates Probe M1 status PARTIAL → PASS-with-caveat in \
         bf16-hhtl-terrain.md queue. CAVEAT: codebook not ICC-calibrated and CascadeConfig \
         not varied — true closure (M1') needs ICC-calibrated codebook + parameter sweep \
         + cross-class re-test (separate Open Idea in IDEAS.md)."
    } else {
        "FAIL — 16-way L0 clustering does NOT meet criteria. \
         Updates Probe M1 status PARTIAL → FAIL in bf16-hhtl-terrain.md \
         queue. Architectural consequence: bit-layout's L0 = 16 coarse \
         clusters claim needs revision OR the 26/256 CHAODA-flagged \
         outliers should be excluded before testing."
    };

    let l0_sizes_sorted = {
        let mut s = l0_sizes.clone();
        s.sort_unstable_by(|a, b| b.cmp(a));
        s
    };

    let detail = format!(
        "256 Jina-v5 centroids, L0 = 16 coarse clusters via Ward agglomerative clustering. \
         L0 cluster sizes (sorted desc): {l0_sizes_sorted:?}. \
         L0 size balance (std/mean) = {l0_balance:.4} (PASS if ≤ 0.5: {l0_balance_pass}). \
         L0 discrimination (within/across) = {l0_ratio:.4} (within={l0_within:.2}, across={l0_across:.2}; PASS if ≤ 0.7: {l0_discrimination_pass}). \
         L1 = 1:1 centroids per bit-layout (no L1 clustering to test). \
         L2 = 4096 sub-centroid buckets, separate probe. \
         {conclusion}"
    );

    PillarResult {
        name: "Probe M1: CLAM 3-level 16-way tree on 256 Jina-v5 centroids",
        pass,
        measured: l0_balance,
        predicted: 0.5,
        detail,
        runtime_ms: 0,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ward_cluster_trivial_cases() {
        // 4 points, 4 clusters → each is own
        let d = vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![1.0, 0.0, 1.0, 2.0],
            vec![2.0, 1.0, 0.0, 1.0],
            vec![3.0, 2.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 3];
        let a = ward_cluster(&d, &indices, 4);
        // Each in own cluster — 4 unique values
        let mut sorted = a.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 4);

        // 4 points, 1 cluster → all same
        let a = ward_cluster(&d, &indices, 1);
        assert_eq!(a, vec![0, 0, 0, 0]);
    }

    #[test]
    fn ward_cluster_separates_two_groups() {
        // Two well-separated groups: {0,1} close, {2,3} close, far apart
        let d = vec![
            vec![0.0, 1.0, 100.0, 101.0],
            vec![1.0, 0.0, 100.0, 101.0],
            vec![100.0, 100.0, 0.0, 1.0],
            vec![101.0, 101.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 3];
        let a = ward_cluster(&d, &indices, 2);
        // 0 and 1 in same cluster, 2 and 3 in same cluster
        assert_eq!(a[0], a[1], "0 and 1 should be same cluster");
        assert_eq!(a[2], a[3], "2 and 3 should be same cluster");
        assert_ne!(a[0], a[2], "0 and 2 should be different clusters");
    }

    #[test]
    fn cluster_sizes_count_correctly() {
        let a = vec![0, 1, 0, 2, 1, 0];
        let counts = cluster_sizes(&a, 3);
        assert_eq!(counts, vec![3, 2, 1]);
    }

    #[test]
    fn size_balance_perfect_is_zero() {
        let b = size_balance(&[10, 10, 10, 10]);
        assert!(b.abs() < 1e-9, "Perfect balance should give 0, got {b}");
    }

    #[test]
    fn size_balance_imbalanced_is_high() {
        let b = size_balance(&[100, 1, 1, 1]);
        assert!(b > 1.0, "Highly imbalanced should give >1, got {b}");
    }

    #[test]
    fn within_across_ratio_makes_sense() {
        // Two well-separated clusters, points 0/1 close, 2/3 close, far apart
        let d = vec![
            vec![0.0, 1.0, 100.0, 100.0],
            vec![1.0, 0.0, 100.0, 100.0],
            vec![100.0, 100.0, 0.0, 1.0],
            vec![100.0, 100.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 3];
        let assignment = vec![0, 0, 1, 1];
        let (within, across, ratio) = within_across_ratio(&d, &indices, &assignment);
        assert!((within - 1.0).abs() < 1e-9, "within should be 1.0, got {within}");
        assert!((across - 100.0).abs() < 1e-9, "across should be 100.0, got {across}");
        assert!(ratio < 0.02, "ratio should be ~0.01, got {ratio}");
    }

    #[test]
    fn load_distance_matrix_handles_missing_file() {
        let r = load_distance_matrix("definitely/does/not/exist.u8");
        assert!(r.is_err(), "Should error on missing file");
    }

    #[test]
    fn probe_runs_when_data_present() {
        // This test requires the file to be present at the expected path.
        // If the file is absent (e.g. CI without checkout), the probe should
        // return a meaningful failure detail rather than panic.
        let r = prove();
        assert!(!r.detail.is_empty(), "Detail should always be populated");
        assert!(
            !r.measured.is_nan() || r.detail.contains("Could not load"),
            "If measured is NaN, detail must explain why; got: {}",
            r.detail
        );
    }
}
