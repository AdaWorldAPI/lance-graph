//! Per-role variance audit for NeuronPrint 6D validation.
//!
//! Computes inter-role vs intra-role variance to determine whether
//! the 6 tensor roles (Q/K/V/Gate/Up/Down) are distinguishable in
//! Base17 space. If inter/intra ratio >> 1, the 6D NeuronPrint
//! decomposition captures meaningful structure.

use crate::projection::Base17;

/// Which role a tensor belongs to (mirrors hydrate.rs TensorRole).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    Q = 0,
    K = 1,
    V = 2,
    Gate = 3,
    Up = 4,
    Down = 5,
}

impl Role {
    pub const ALL: [Role; 6] = [Role::Q, Role::K, Role::V, Role::Gate, Role::Up, Role::Down];

    pub fn label(&self) -> &'static str {
        match self {
            Role::Q => "Q",
            Role::K => "K",
            Role::V => "V",
            Role::Gate => "Gate",
            Role::Up => "Up",
            Role::Down => "Down",
        }
    }

    /// Parse from tensor name (HuggingFace or GGUF convention).
    pub fn from_name(name: &str) -> Option<Self> {
        let n = name.to_lowercase();
        if n.contains("q_proj") || n.contains("attn_q") || n.contains(".wq.") { Some(Role::Q) }
        else if n.contains("k_proj") || n.contains("attn_k") || n.contains(".wk.") { Some(Role::K) }
        else if n.contains("v_proj") || n.contains("attn_v") || n.contains(".wv.") { Some(Role::V) }
        else if n.contains("gate_proj") || n.contains("ffn_gate") || n.contains(".w1.") { Some(Role::Gate) }
        else if n.contains("up_proj") || n.contains("ffn_up") || n.contains(".w3.") { Some(Role::Up) }
        else if n.contains("down_proj") || n.contains("ffn_down") || n.contains(".w2.") { Some(Role::Down) }
        else { None }
    }
}

/// Per-role statistics.
#[derive(Clone, Debug)]
pub struct RoleStats {
    pub role: Role,
    /// Number of Base17 vectors in this role.
    pub count: usize,
    /// Centroid (element-wise mean).
    pub centroid: Base17,
    /// Mean intra-role L1 distance to centroid.
    pub mean_intra_l1: f64,
    /// Variance of intra-role L1 distances.
    pub var_intra_l1: f64,
    /// Mean absolute value per dimension (activation magnitude).
    pub mean_magnitude: f64,
}

/// Complete variance audit report.
#[derive(Clone, Debug)]
pub struct RoleVarianceReport {
    pub model_name: String,
    pub per_role: Vec<RoleStats>,
    /// Mean inter-role centroid L1 distance.
    pub mean_inter_l1: f64,
    /// Mean intra-role L1 distance (averaged across roles).
    pub mean_intra_l1: f64,
    /// Inter/intra ratio: > 1 means roles are distinguishable.
    pub ratio: f64,
    /// Per-role-pair centroid distances (6×6 matrix, row-major).
    pub centroid_distances: Vec<(Role, Role, u32)>,
}

impl RoleVarianceReport {
    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Role Variance Audit: {}\n\
             Inter-role mean L1: {:.1}\n\
             Intra-role mean L1: {:.1}\n\
             Ratio (inter/intra): {:.3} {}\n\n",
            self.model_name,
            self.mean_inter_l1,
            self.mean_intra_l1,
            self.ratio,
            if self.ratio > 2.0 { "DISTINCT" }
            else if self.ratio > 1.0 { "SEPARABLE" }
            else { "MUSHY" },
        );
        for rs in &self.per_role {
            s.push_str(&format!(
                "  {:<5}: n={:>5}, mean_intra_L1={:>8.1}, var={:>10.1}, magnitude={:.3}\n",
                rs.role.label(), rs.count, rs.mean_intra_l1, rs.var_intra_l1, rs.mean_magnitude
            ));
        }
        s.push_str("\nCentroid distance matrix (top pairs):\n");
        let mut sorted_dists = self.centroid_distances.clone();
        sorted_dists.sort_by_key(|&(_, _, d)| d);
        for &(r1, r2, d) in sorted_dists.iter().take(10) {
            s.push_str(&format!("  {} ↔ {} : {}\n", r1.label(), r2.label(), d));
        }
        s
    }
}

/// Compute per-role variance audit from labeled Base17 vectors.
///
/// Input: Vec of (role, Base17) pairs from a single model.
/// Typically produced by reading bgz7 files and parsing tensor names.
pub fn compute_variance(
    model_name: &str,
    labeled: &[(Role, Base17)],
) -> RoleVarianceReport {
    // Group by role
    let mut groups: [Vec<&Base17>; 6] = Default::default();
    for (role, b17) in labeled {
        groups[*role as usize].push(b17);
    }

    // Compute per-role stats
    let mut per_role = Vec::new();
    for role in Role::ALL {
        let vecs = &groups[role as usize];
        if vecs.is_empty() {
            continue;
        }

        let centroid = compute_centroid(vecs);
        let distances: Vec<u32> = vecs.iter().map(|v| v.l1(&centroid)).collect();
        let mean_d = distances.iter().map(|&d| d as f64).sum::<f64>() / distances.len() as f64;
        let var_d = distances.iter().map(|&d| {
            let diff = d as f64 - mean_d;
            diff * diff
        }).sum::<f64>() / distances.len() as f64;

        let mean_mag = vecs.iter().flat_map(|v| v.dims.iter())
            .map(|&d| (d as f64).abs()).sum::<f64>() / (vecs.len() * 17) as f64;

        per_role.push(RoleStats {
            role,
            count: vecs.len(),
            centroid,
            mean_intra_l1: mean_d,
            var_intra_l1: var_d,
            mean_magnitude: mean_mag,
        });
    }

    // Compute inter-role centroid distances
    let mut centroid_distances = Vec::new();
    let mut inter_sum = 0.0f64;
    let mut inter_count = 0;
    for i in 0..per_role.len() {
        for j in (i + 1)..per_role.len() {
            let d = per_role[i].centroid.l1(&per_role[j].centroid);
            centroid_distances.push((per_role[i].role, per_role[j].role, d));
            inter_sum += d as f64;
            inter_count += 1;
        }
    }

    let mean_inter = if inter_count > 0 { inter_sum / inter_count as f64 } else { 0.0 };
    let mean_intra = if per_role.is_empty() {
        0.0
    } else {
        per_role.iter().map(|r| r.mean_intra_l1).sum::<f64>() / per_role.len() as f64
    };
    let ratio = if mean_intra > 0.0 { mean_inter / mean_intra } else { 0.0 };

    RoleVarianceReport {
        model_name: model_name.to_string(),
        per_role,
        mean_inter_l1: mean_inter,
        mean_intra_l1: mean_intra,
        ratio,
        centroid_distances,
    }
}

/// Compute Base17 centroid.
fn compute_centroid(vecs: &[&Base17]) -> Base17 {
    let n = vecs.len() as i64;
    if n == 0 {
        return Base17::zero();
    }
    let mut sums = [0i64; 17];
    for v in vecs {
        for (d, sum) in sums.iter_mut().enumerate() {
            *sum += v.dims[d] as i64;
        }
    }
    let mut dims = [0i16; 17];
    for (d, dim) in dims.iter_mut().enumerate() {
        *dim = (sums[d] / n).clamp(-32768, 32767) as i16;
    }
    Base17 { dims }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_role_vectors(role: Role, base: i16, n: usize) -> Vec<(Role, Base17)> {
        (0..n).map(|i| {
            let mut dims = [0i16; 17];
            for d in 0..17 {
                dims[d] = base + ((i * 7 + d * 3) % 100) as i16;
            }
            (role, Base17 { dims })
        }).collect()
    }

    #[test]
    fn distinct_roles() {
        let mut labeled = Vec::new();
        // Each role has a distinct base value → centroids should be far apart
        labeled.extend(make_role_vectors(Role::Q, 100, 50));
        labeled.extend(make_role_vectors(Role::K, 500, 50));
        labeled.extend(make_role_vectors(Role::V, 1000, 50));
        labeled.extend(make_role_vectors(Role::Gate, 2000, 50));
        labeled.extend(make_role_vectors(Role::Up, 3000, 50));
        labeled.extend(make_role_vectors(Role::Down, 4000, 50));

        let report = compute_variance("test_model", &labeled);
        assert!(report.ratio > 1.0,
            "distinct roles should have ratio > 1: {}", report.ratio);
        assert_eq!(report.per_role.len(), 6);
    }

    #[test]
    fn identical_roles_mushy() {
        let mut labeled = Vec::new();
        // All roles have the same base → centroids are identical → ratio ≈ 0
        for role in Role::ALL {
            labeled.extend(make_role_vectors(role, 500, 50));
        }

        let report = compute_variance("mushy_model", &labeled);
        // When all roles have the same distribution, inter ≈ intra
        assert!(report.ratio < 1.5, "identical distributions should have low ratio: {}", report.ratio);
    }

    #[test]
    fn empty_input() {
        let report = compute_variance("empty", &[]);
        assert_eq!(report.per_role.len(), 0);
        assert_eq!(report.ratio, 0.0);
    }

    #[test]
    fn gate_dominates() {
        let mut labeled = Vec::new();
        // Gate has much higher magnitude (matching FfnGate finding)
        labeled.extend(make_role_vectors(Role::Q, 100, 50));
        labeled.extend(make_role_vectors(Role::K, 110, 50));
        labeled.extend(make_role_vectors(Role::V, 120, 50));
        labeled.extend(make_role_vectors(Role::Gate, 5000, 50)); // 50× magnitude
        labeled.extend(make_role_vectors(Role::Up, 130, 50));
        labeled.extend(make_role_vectors(Role::Down, 140, 50));

        let report = compute_variance("gate_dominant", &labeled);
        let gate_stats = report.per_role.iter().find(|r| r.role == Role::Gate).unwrap();
        let q_stats = report.per_role.iter().find(|r| r.role == Role::Q).unwrap();
        assert!(gate_stats.mean_magnitude > q_stats.mean_magnitude * 10.0,
            "Gate should have much higher magnitude");
    }
}
