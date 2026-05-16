//! Splat op fleet (D-CSV-12) — scalar implementations.
//! Per cognitive-substrate-convergence-v1.md §11 D-CSV-12.
//!
//! Sprint-12 scope: free functions taking `&mut [SplatField]`. Sprint-13+
//! migrates these to methods on the `Think` carrier once the splat field
//! is wired into the struct (see knowledge/splat-shader-rayon-struct-
//! method-vision.md for the destination shape).

/// A single Gaussian splat in the field. Mirrors the ndarray
/// `SplatFieldStream::SplatField` (W-F6 sibling). Local def to avoid
/// the ndarray dep cycle.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct SplatField {
    pub mean: u32,
    pub variance: f32,
    pub energy: f32,
    pub generation: u32,
}

/// Splat a Gaussian centered at `mean` with `variance` into the field,
/// adding `energy` quantum. Existing splats with the same mean merge
/// (energy adds, variance blends weighted by energy).
pub fn splat_gaussian(
    field: &mut Vec<SplatField>,
    mean: u32,
    variance: f32,
    energy: f32,
    generation: u32,
) {
    for s in field.iter_mut() {
        if s.mean == mean {
            let total_e = s.energy + energy;
            if total_e > 0.0 {
                s.variance = (s.variance * s.energy + variance * energy) / total_e;
            }
            s.energy = total_e;
            s.generation = generation;
            return;
        }
    }
    field.push(SplatField { mean, variance, energy, generation });
}

/// Score the "hole closure" potential of the field: how much of the
/// total energy is concentrated in <=k peaks. Returns a [0.0, 1.0] ratio.
/// High ratio = field has converged; low ratio = scattered.
pub fn score_hole_closure(field: &[SplatField], k: usize) -> f32 {
    if field.is_empty() {
        return 0.0;
    }
    let mut energies: Vec<f32> = field.iter().map(|s| s.energy).collect();
    energies.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = energies.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let top_k: f32 = energies.iter().take(k).sum();
    (top_k / total).clamp(0.0, 1.0)
}

/// Replay-coherence: how closely the current field matches a prior generation's
/// shape. Returns cosine-similarity-style metric in [-1, +1] of energy vectors
/// (aligned by mean). Used to detect "this thought is repeating".
pub fn replay_coherence(current: &[SplatField], prior: &[SplatField]) -> f32 {
    use std::collections::HashMap;
    let mut p_map: HashMap<u32, f32> = HashMap::new();
    for s in prior {
        *p_map.entry(s.mean).or_insert(0.0) += s.energy;
    }
    let mut dot = 0.0_f32;
    let mut c_mag = 0.0_f32;
    let p_mag: f32 = p_map.values().map(|e| e * e).sum();
    for s in current {
        c_mag += s.energy * s.energy;
        if let Some(&pe) = p_map.get(&s.mean) {
            dot += s.energy * pe;
        }
    }
    let denom = (c_mag.sqrt() * p_mag.sqrt()).max(f32::EPSILON);
    (dot / denom).clamp(-1.0, 1.0)
}

/// Decide whether the field state qualifies as an "epiphany emission":
/// hole-closure above threshold AND replay-coherence above similarity_floor.
/// Returns `Some(top_splat)` if epiphany detected, `None` otherwise.
pub fn emit_if_epiphany(
    field: &[SplatField],
    prior: &[SplatField],
    closure_threshold: f32,
    similarity_floor: f32,
) -> Option<SplatField> {
    let closure = score_hole_closure(field, 3);
    let coherence = replay_coherence(field, prior);
    if closure >= closure_threshold && coherence >= similarity_floor {
        field
            .iter()
            .max_by(|a, b| {
                a.energy
                    .partial_cmp(&b.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── splat_gaussian ──────────────────────────────────────────────────────────

    #[test]
    fn splat_gaussian_new_entry() {
        let mut field: Vec<SplatField> = Vec::new();
        splat_gaussian(&mut field, 10, 1.0, 5.0, 1);
        assert_eq!(field.len(), 1);
        assert_eq!(field[0].mean, 10);
        assert_eq!(field[0].energy, 5.0);
        assert_eq!(field[0].generation, 1);
    }

    #[test]
    fn splat_gaussian_merge_same_mean() {
        let mut field: Vec<SplatField> = Vec::new();
        splat_gaussian(&mut field, 10, 1.0, 3.0, 1);
        splat_gaussian(&mut field, 10, 1.0, 7.0, 2);
        assert_eq!(field.len(), 1, "same mean should merge");
        assert!((field[0].energy - 10.0).abs() < 1e-6);
        assert_eq!(field[0].generation, 2);
    }

    #[test]
    fn splat_gaussian_weighted_variance_blend() {
        let mut field: Vec<SplatField> = Vec::new();
        // First splat: variance=2.0, energy=4.0
        splat_gaussian(&mut field, 5, 2.0, 4.0, 1);
        // Second splat: variance=6.0, energy=4.0
        // Blended variance = (2.0*4.0 + 6.0*4.0) / (4.0+4.0) = 32/8 = 4.0
        splat_gaussian(&mut field, 5, 6.0, 4.0, 2);
        let expected_var = 4.0_f32;
        assert!(
            (field[0].variance - expected_var).abs() < 1e-5,
            "variance blend failed: got {}",
            field[0].variance
        );
    }

    #[test]
    fn splat_gaussian_generation_update() {
        let mut field: Vec<SplatField> = Vec::new();
        splat_gaussian(&mut field, 7, 1.0, 1.0, 3);
        assert_eq!(field[0].generation, 3);
        splat_gaussian(&mut field, 7, 1.0, 1.0, 9);
        assert_eq!(field[0].generation, 9, "generation must update on merge");
    }

    // ── score_hole_closure ──────────────────────────────────────────────────────

    #[test]
    fn score_hole_closure_empty_field() {
        assert_eq!(score_hole_closure(&[], 3), 0.0);
    }

    #[test]
    fn score_hole_closure_single_splat() {
        let field = vec![SplatField { mean: 1, variance: 1.0, energy: 10.0, generation: 1 }];
        // k=1: top-1/total = 1.0
        assert!((score_hole_closure(&field, 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_hole_closure_evenly_distributed() {
        // 4 equal splats, k=1 → 0.25
        let field: Vec<SplatField> = (0..4)
            .map(|i| SplatField { mean: i, variance: 1.0, energy: 1.0, generation: 1 })
            .collect();
        let score = score_hole_closure(&field, 1);
        assert!((score - 0.25).abs() < 1e-6, "got {score}");
    }

    #[test]
    fn score_hole_closure_concentrated() {
        // 1 dominant splat energy=8, 3 tiny=1 each → top-1 = 8/11 ≈ 0.727
        let mut field = vec![SplatField { mean: 0, variance: 1.0, energy: 8.0, generation: 1 }];
        for i in 1..4_u32 {
            field.push(SplatField { mean: i, variance: 1.0, energy: 1.0, generation: 1 });
        }
        let score = score_hole_closure(&field, 1);
        let expected = 8.0_f32 / 11.0;
        assert!((score - expected).abs() < 1e-5, "got {score}");
    }

    // ── replay_coherence ────────────────────────────────────────────────────────

    #[test]
    fn replay_coherence_empty_prior() {
        let current = vec![SplatField { mean: 1, variance: 1.0, energy: 5.0, generation: 1 }];
        // dot=0, p_mag=0 → denom = EPSILON → result clamps near 0
        let r = replay_coherence(&current, &[]);
        assert!(r.abs() < 1e-3, "expected ~0 with empty prior, got {r}");
    }

    #[test]
    fn replay_coherence_identical_fields() {
        let field = vec![
            SplatField { mean: 1, variance: 1.0, energy: 3.0, generation: 1 },
            SplatField { mean: 2, variance: 1.0, energy: 4.0, generation: 1 },
        ];
        let r = replay_coherence(&field, &field);
        assert!((r - 1.0).abs() < 1e-5, "identical fields should give 1.0, got {r}");
    }

    #[test]
    fn replay_coherence_orthogonal_fields() {
        // Non-overlapping means → dot=0 → coherence=0
        let current = vec![SplatField { mean: 1, variance: 1.0, energy: 5.0, generation: 1 }];
        let prior = vec![SplatField { mean: 99, variance: 1.0, energy: 5.0, generation: 1 }];
        let r = replay_coherence(&current, &prior);
        assert!(r.abs() < 1e-5, "orthogonal means → 0, got {r}");
    }

    #[test]
    fn replay_coherence_partial_overlap() {
        // current: mean=1 (e=3), mean=2 (e=4). prior: mean=1 (e=3) only.
        // dot = 3*3=9; c_mag=9+16=25; p_mag=9; denom=5*3=15; result=9/15=0.6
        let current = vec![
            SplatField { mean: 1, variance: 1.0, energy: 3.0, generation: 1 },
            SplatField { mean: 2, variance: 1.0, energy: 4.0, generation: 1 },
        ];
        let prior = vec![SplatField { mean: 1, variance: 1.0, energy: 3.0, generation: 1 }];
        let r = replay_coherence(&current, &prior);
        assert!((r - 0.6).abs() < 1e-5, "expected 0.6, got {r}");
    }

    // ── emit_if_epiphany ────────────────────────────────────────────────────────

    #[test]
    fn emit_if_epiphany_below_threshold_none() {
        // Scattered field: hole closure will be low. Prior identical → coherence high.
        // But closure < threshold → None.
        let field: Vec<SplatField> = (0..10)
            .map(|i| SplatField { mean: i, variance: 1.0, energy: 1.0, generation: 1 })
            .collect();
        // score_hole_closure(k=3) = 3/10 = 0.3 < 0.8 threshold
        let result = emit_if_epiphany(&field, &field, 0.8, 0.0);
        assert!(result.is_none(), "scattered field should not emit epiphany");
    }

    #[test]
    fn emit_if_epiphany_both_pass_returns_top() {
        // Concentrated field: 1 dominant, closure high; identical prior, coherence=1.
        let field = vec![
            SplatField { mean: 1, variance: 0.5, energy: 90.0, generation: 2 },
            SplatField { mean: 2, variance: 1.0, energy: 5.0, generation: 2 },
            SplatField { mean: 3, variance: 1.0, energy: 5.0, generation: 2 },
        ];
        // closure(k=3) = 100/100 = 1.0; coherence(identical)=1.0
        let result = emit_if_epiphany(&field, &field, 0.5, 0.5);
        assert!(result.is_some());
        let top = result.unwrap();
        assert_eq!(top.mean, 1, "top splat should be the highest-energy one");
        assert!((top.energy - 90.0).abs() < 1e-6);
    }

    #[test]
    fn emit_if_epiphany_only_closure_passes_none() {
        // Concentrated field but orthogonal prior → coherence=0 < floor
        let field = vec![
            SplatField { mean: 1, variance: 0.5, energy: 90.0, generation: 2 },
            SplatField { mean: 2, variance: 1.0, energy: 5.0, generation: 2 },
        ];
        let prior = vec![SplatField { mean: 99, variance: 1.0, energy: 50.0, generation: 1 }];
        // closure ~= 1.0 > 0.5 ✓ but coherence = 0.0 < 0.5 ✗
        let result = emit_if_epiphany(&field, &prior, 0.5, 0.5);
        assert!(result.is_none(), "only closure passes → no epiphany");
    }

    #[test]
    fn emit_if_epiphany_only_coherence_passes_none() {
        // Scattered field (closure low) but identical prior (coherence=1)
        let field: Vec<SplatField> = (0..20)
            .map(|i| SplatField { mean: i, variance: 1.0, energy: 1.0, generation: 1 })
            .collect();
        // closure(k=3)=3/20=0.15 < 0.5 ✗; coherence=1.0 > 0.0 ✓
        let result = emit_if_epiphany(&field, &field, 0.5, 0.0);
        assert!(result.is_none(), "only coherence passes → no epiphany");
    }
}
