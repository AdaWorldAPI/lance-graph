//! `Think` — the doctrinal cognitive carrier (CLAUDE.md "Thinking is a struct").
//!
//! Sprint-13 minimum-viable scope (D-CSV-14): holds the splat field + cycle
//! counter. The four splat methods (`splat_gaussian`, `score_hole_closure`,
//! `replay_coherence`, `emit_if_epiphany`) live here per the on-Think migration
//! of D-CSV-14 (cognitive-substrate-convergence-v2.md line 383).
//!
//! Future sprints (14+) will accrete `trajectory: Vsa16kF32`,
//! `awareness: ParamTruths`, `free_energy: FreeEnergy`,
//! `resolution: Resolution`, `episodic: &EpisodicMemory`,
//! `graph: &TripletGraph`, `global_context: &Vsa16kF32`,
//! `codec: &CamPqCodec` per CLAUDE.md doctrine "Thinking is a struct"
//! and "AriGraph, episodic memory, SPO, CAM-PQ are thinking tissue".
//!
//! Doctrine cite: CLAUDE.md "The Click" §"Thinking is a struct" + §"Litmus
//! tests": "Free function = reject. Method = accept."

use crate::splat_ops::SplatField;

/// The doctrinal cognitive carrier (sprint-13 splat-scoped minimum).
///
/// State invariant: `cycle` increments monotonically on every method that
/// mutates `splat_field`. Read-only methods (`score_hole_closure`,
/// `replay_coherence`, `emit_if_epiphany`) do NOT touch `cycle`.
///
/// Sprint-14+ extensions land additively as new fields on this same carrier
/// without renaming. The struct is named `Think` (not `ThinkSplat`) precisely
/// so that it can grow into the full CLAUDE.md 8-field doctrinal shape.
#[derive(Clone, Debug, Default)]
pub struct Think {
    /// Gaussian splat field — additive perturbation surface. Each entry
    /// carries (mean, variance, energy, generation). See `splat_ops::SplatField`.
    ///
    /// Added per OQ-CSV-9: `splat_field` lives directly on `Think` (not on a
    /// sibling `Splat` carrier, which would re-create the free-function-on-state
    /// anti-pattern the doctrine is migrating away from).
    pub splat_field: Vec<SplatField>,

    /// Monotonic cycle counter. Replaces the `generation: u32` parameter that
    /// the free fns took. Each call to `splat_gaussian` advances `cycle` before
    /// writing `SplatField::generation = self.cycle`.
    ///
    /// Derives `SplatField::generation` per OQ-CSV-10: single source of truth;
    /// `cycle` already advances on every Think step, so no separate
    /// `splat_generation` field is needed.
    pub cycle: u32,
}

impl Think {
    /// Construct a fresh carrier with an empty splat field at cycle 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct from a pre-populated field (used by free-fn deprecation shims
    /// and by test fixtures replaying historical splat states).
    pub fn from_field(splat_field: Vec<SplatField>, cycle: u32) -> Self {
        Self { splat_field, cycle }
    }

    /// Advance the cycle counter. Invoked by all mutating splat methods.
    ///
    /// Public so test fixtures can rewind/replay without going through a full
    /// splat. Uses `wrapping_add` — at 1 M cycles/sec, wrap-around is ~1.2 hours;
    /// real Think instances checkpoint long before that horizon.
    pub fn advance_cycle(&mut self) {
        self.cycle = self.cycle.wrapping_add(1);
    }

    // ── mutating methods ──────────────────────────────────────────────────────

    /// Splat a Gaussian centered at `mean` with `variance` into the field,
    /// adding `energy` quantum. Merges with existing same-mean entries (energy
    /// adds, variance blends weighted by energy). Advances `self.cycle`; the
    /// resulting `SplatField::generation` is derived from the post-advance
    /// `self.cycle` (OQ-CSV-10).
    ///
    /// Sprint-13 D-CSV-14 migration: replaces free `splat_ops::splat_gaussian`
    /// which took `(&mut Vec<SplatField>, mean, variance, energy, generation)`.
    /// The generation parameter is now read from `self.cycle` after advance.
    pub fn splat_gaussian(&mut self, mean: u32, variance: f32, energy: f32) {
        self.advance_cycle();
        let generation = self.cycle;
        for s in self.splat_field.iter_mut() {
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
        self.splat_field.push(SplatField {
            mean,
            variance,
            energy,
            generation,
        });
    }

    // ── read-only methods ─────────────────────────────────────────────────────

    /// Score the "hole closure" potential of the current field: how much of the
    /// total energy is concentrated in `<=k` peaks. Returns a `[0.0, 1.0]`
    /// ratio. High ratio = field has converged; low = scattered.
    ///
    /// Read-only — does NOT advance `self.cycle`.
    ///
    /// Sprint-13 D-CSV-14 migration: replaces free `splat_ops::score_hole_closure`
    /// which took `(field: &[SplatField], k: usize)`.
    pub fn score_hole_closure(&self, k: usize) -> f32 {
        if self.splat_field.is_empty() {
            return 0.0;
        }
        let mut energies: Vec<f32> = self.splat_field.iter().map(|s| s.energy).collect();
        energies.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let total: f32 = energies.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let top_k: f32 = energies.iter().take(k).sum();
        (top_k / total).clamp(0.0, 1.0)
    }

    /// Replay-coherence between this Think's current field and a prior
    /// generation's field. Returns cosine-similarity-style metric in `[-1, +1]`
    /// of energy vectors aligned by mean.
    ///
    /// Used to detect "this thought is repeating". Read-only — does NOT advance
    /// `self.cycle`.
    ///
    /// Sprint-13 D-CSV-14 migration: replaces free `splat_ops::replay_coherence`
    /// which took `(current: &[SplatField], prior: &[SplatField])`.
    pub fn replay_coherence(&self, prior: &[SplatField]) -> f32 {
        use std::collections::HashMap;
        let mut p_map: HashMap<u32, f32> = HashMap::new();
        for s in prior {
            *p_map.entry(s.mean).or_insert(0.0) += s.energy;
        }
        let mut dot = 0.0_f32;
        let mut c_mag = 0.0_f32;
        let p_mag: f32 = p_map.values().map(|e| e * e).sum();
        for s in &self.splat_field {
            c_mag += s.energy * s.energy;
            if let Some(&pe) = p_map.get(&s.mean) {
                dot += s.energy * pe;
            }
        }
        let denom = (c_mag.sqrt() * p_mag.sqrt()).max(f32::EPSILON);
        (dot / denom).clamp(-1.0, 1.0)
    }

    /// Decide whether the field state qualifies as an "epiphany emission":
    /// hole-closure above `closure_threshold` AND replay-coherence above
    /// `similarity_floor` vs the supplied `prior` field.
    ///
    /// Returns `Some(top_splat)` (the highest-energy splat) if the epiphany
    /// fires, `None` otherwise. Read-only — does NOT advance `self.cycle`.
    ///
    /// Sprint-13 D-CSV-14 migration: replaces free `splat_ops::emit_if_epiphany`
    /// which took `(field, prior, closure_threshold, similarity_floor)`.
    pub fn emit_if_epiphany(
        &self,
        prior: &[SplatField],
        closure_threshold: f32,
        similarity_floor: f32,
    ) -> Option<SplatField> {
        let closure = self.score_hole_closure(3);
        let coherence = self.replay_coherence(prior);
        if closure >= closure_threshold && coherence >= similarity_floor {
            self.splat_field
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
}

#[cfg(test)]
mod think_splat_tests {
    use super::*;

    // ── 16 migrated tests (from splat_ops.rs free-fn tests) ──────────────────

    // ── splat_gaussian ────────────────────────────────────────────────────────

    #[test]
    fn think_splat_gaussian_new_entry() {
        // Seed cycle=0 so first splat lands at generation=1.
        let mut think = Think::new();
        think.splat_gaussian(10, 1.0, 5.0);
        assert_eq!(think.splat_field.len(), 1);
        assert_eq!(think.splat_field[0].mean, 10);
        assert_eq!(think.splat_field[0].energy, 5.0);
        assert_eq!(think.splat_field[0].generation, 1);
    }

    #[test]
    fn think_splat_gaussian_merge_same_mean() {
        let mut think = Think::new();
        think.splat_gaussian(10, 1.0, 3.0); // cycle→1, gen=1
        think.splat_gaussian(10, 1.0, 7.0); // cycle→2, merge, gen=2
        assert_eq!(think.splat_field.len(), 1, "same mean should merge");
        assert!((think.splat_field[0].energy - 10.0).abs() < 1e-6);
        assert_eq!(think.splat_field[0].generation, 2);
    }

    #[test]
    fn think_splat_gaussian_weighted_variance_blend() {
        let mut think = Think::new();
        // First splat: variance=2.0, energy=4.0
        think.splat_gaussian(5, 2.0, 4.0);
        // Second splat: variance=6.0, energy=4.0
        // Blended variance = (2.0*4.0 + 6.0*4.0) / (4.0+4.0) = 32/8 = 4.0
        think.splat_gaussian(5, 6.0, 4.0);
        let expected_var = 4.0_f32;
        assert!(
            (think.splat_field[0].variance - expected_var).abs() < 1e-5,
            "variance blend failed: got {}",
            think.splat_field[0].variance
        );
    }

    #[test]
    fn think_splat_gaussian_generation_update() {
        // Seed at cycle=2 so first splat lands at generation=3 (mirrors free-fn test).
        let mut think = Think::from_field(Vec::new(), 2);
        think.splat_gaussian(7, 1.0, 1.0); // cycle→3, gen=3
        assert_eq!(think.splat_field[0].generation, 3);
        // Advance to cycle=8 so merge lands at generation=9.
        let mut think2 = Think::from_field(think.splat_field.clone(), 8);
        think2.splat_gaussian(7, 1.0, 1.0); // cycle→9, gen=9
        assert_eq!(
            think2.splat_field[0].generation, 9,
            "generation must update on merge"
        );
    }

    // ── score_hole_closure ────────────────────────────────────────────────────

    #[test]
    fn think_score_hole_closure_empty_field() {
        let think = Think::new();
        assert_eq!(think.score_hole_closure(3), 0.0);
    }

    #[test]
    fn think_score_hole_closure_single_splat() {
        let field = vec![SplatField {
            mean: 1,
            variance: 1.0,
            energy: 10.0,
            generation: 1,
        }];
        let think = Think::from_field(field, 0);
        // k=1: top-1/total = 1.0
        assert!((think.score_hole_closure(1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn think_score_hole_closure_evenly_distributed() {
        // 4 equal splats, k=1 → 0.25
        let field: Vec<SplatField> = (0..4)
            .map(|i| SplatField {
                mean: i,
                variance: 1.0,
                energy: 1.0,
                generation: 1,
            })
            .collect();
        let think = Think::from_field(field, 0);
        let score = think.score_hole_closure(1);
        assert!((score - 0.25).abs() < 1e-6, "got {score}");
    }

    #[test]
    fn think_score_hole_closure_concentrated() {
        // 1 dominant splat energy=8, 3 tiny=1 each → top-1 = 8/11 ≈ 0.727
        let mut field = vec![SplatField {
            mean: 0,
            variance: 1.0,
            energy: 8.0,
            generation: 1,
        }];
        for i in 1..4_u32 {
            field.push(SplatField {
                mean: i,
                variance: 1.0,
                energy: 1.0,
                generation: 1,
            });
        }
        let think = Think::from_field(field, 0);
        let score = think.score_hole_closure(1);
        let expected = 8.0_f32 / 11.0;
        assert!((score - expected).abs() < 1e-5, "got {score}");
    }

    // ── replay_coherence ──────────────────────────────────────────────────────

    #[test]
    fn think_replay_coherence_empty_prior() {
        let field = vec![SplatField {
            mean: 1,
            variance: 1.0,
            energy: 5.0,
            generation: 1,
        }];
        let think = Think::from_field(field, 0);
        // dot=0, p_mag=0 → denom = EPSILON → result clamps near 0
        let r = think.replay_coherence(&[]);
        assert!(r.abs() < 1e-3, "expected ~0 with empty prior, got {r}");
    }

    #[test]
    fn think_replay_coherence_identical_fields() {
        let field = vec![
            SplatField {
                mean: 1,
                variance: 1.0,
                energy: 3.0,
                generation: 1,
            },
            SplatField {
                mean: 2,
                variance: 1.0,
                energy: 4.0,
                generation: 1,
            },
        ];
        let think = Think::from_field(field.clone(), 0);
        let r = think.replay_coherence(&field);
        assert!(
            (r - 1.0).abs() < 1e-5,
            "identical fields should give 1.0, got {r}"
        );
    }

    #[test]
    fn think_replay_coherence_orthogonal_fields() {
        // Non-overlapping means → dot=0 → coherence=0
        let current = vec![SplatField {
            mean: 1,
            variance: 1.0,
            energy: 5.0,
            generation: 1,
        }];
        let prior = vec![SplatField {
            mean: 99,
            variance: 1.0,
            energy: 5.0,
            generation: 1,
        }];
        let think = Think::from_field(current, 0);
        let r = think.replay_coherence(&prior);
        assert!(r.abs() < 1e-5, "orthogonal means → 0, got {r}");
    }

    #[test]
    fn think_replay_coherence_partial_overlap() {
        // current: mean=1 (e=3), mean=2 (e=4). prior: mean=1 (e=3) only.
        // dot = 3*3=9; c_mag=9+16=25; p_mag=9; denom=5*3=15; result=9/15=0.6
        let current = vec![
            SplatField {
                mean: 1,
                variance: 1.0,
                energy: 3.0,
                generation: 1,
            },
            SplatField {
                mean: 2,
                variance: 1.0,
                energy: 4.0,
                generation: 1,
            },
        ];
        let prior = vec![SplatField {
            mean: 1,
            variance: 1.0,
            energy: 3.0,
            generation: 1,
        }];
        let think = Think::from_field(current, 0);
        let r = think.replay_coherence(&prior);
        assert!((r - 0.6).abs() < 1e-5, "expected 0.6, got {r}");
    }

    // ── emit_if_epiphany ──────────────────────────────────────────────────────

    #[test]
    fn think_emit_if_epiphany_below_threshold_none() {
        // Scattered field: hole closure will be low. Prior identical → coherence high.
        // But closure < threshold → None.
        let field: Vec<SplatField> = (0..10)
            .map(|i| SplatField {
                mean: i,
                variance: 1.0,
                energy: 1.0,
                generation: 1,
            })
            .collect();
        let think = Think::from_field(field.clone(), 0);
        // score_hole_closure(k=3) = 3/10 = 0.3 < 0.8 threshold
        let result = think.emit_if_epiphany(&field, 0.8, 0.0);
        assert!(result.is_none(), "scattered field should not emit epiphany");
    }

    #[test]
    fn think_emit_if_epiphany_both_pass_returns_top() {
        // Concentrated field: 1 dominant, closure high; identical prior, coherence=1.
        let field = vec![
            SplatField {
                mean: 1,
                variance: 0.5,
                energy: 90.0,
                generation: 2,
            },
            SplatField {
                mean: 2,
                variance: 1.0,
                energy: 5.0,
                generation: 2,
            },
            SplatField {
                mean: 3,
                variance: 1.0,
                energy: 5.0,
                generation: 2,
            },
        ];
        let think = Think::from_field(field.clone(), 0);
        // closure(k=3) = 100/100 = 1.0; coherence(identical)=1.0
        let result = think.emit_if_epiphany(&field, 0.5, 0.5);
        assert!(result.is_some());
        let top = result.unwrap();
        assert_eq!(top.mean, 1, "top splat should be the highest-energy one");
        assert!((top.energy - 90.0).abs() < 1e-6);
    }

    #[test]
    fn think_emit_if_epiphany_only_closure_passes_none() {
        // Concentrated field but orthogonal prior → coherence=0 < floor
        let field = vec![
            SplatField {
                mean: 1,
                variance: 0.5,
                energy: 90.0,
                generation: 2,
            },
            SplatField {
                mean: 2,
                variance: 1.0,
                energy: 5.0,
                generation: 2,
            },
        ];
        let prior = vec![SplatField {
            mean: 99,
            variance: 1.0,
            energy: 50.0,
            generation: 1,
        }];
        let think = Think::from_field(field, 0);
        // closure ~= 1.0 > 0.5 ✓ but coherence = 0.0 < 0.5 ✗
        let result = think.emit_if_epiphany(&prior, 0.5, 0.5);
        assert!(result.is_none(), "only closure passes → no epiphany");
    }

    #[test]
    fn think_emit_if_epiphany_only_coherence_passes_none() {
        // Scattered field (closure low) but identical prior (coherence=1)
        let field: Vec<SplatField> = (0..20)
            .map(|i| SplatField {
                mean: i,
                variance: 1.0,
                energy: 1.0,
                generation: 1,
            })
            .collect();
        let think = Think::from_field(field.clone(), 0);
        // closure(k=3)=3/20=0.15 < 0.5 ✗; coherence=1.0 > 0.0 ✓
        let result = think.emit_if_epiphany(&field, 0.5, 0.0);
        assert!(result.is_none(), "only coherence passes → no epiphany");
    }

    // ── 4 cycle-integration tests (OQ-CSV-10: generation derives from self.cycle) ──

    #[test]
    fn think_splat_gaussian_advances_cycle() {
        let mut think = Think::new();
        assert_eq!(think.cycle, 0);
        think.splat_gaussian(1, 1.0, 5.0);
        assert_eq!(think.cycle, 1, "first splat advances cycle to 1");
        assert_eq!(think.splat_field[0].generation, 1);
        think.splat_gaussian(2, 1.0, 5.0);
        assert_eq!(think.cycle, 2, "second splat advances cycle to 2");
        assert_eq!(think.splat_field[1].generation, 2);
    }

    #[test]
    fn think_score_hole_closure_does_not_advance_cycle() {
        let mut think = Think::new();
        think.splat_gaussian(1, 1.0, 5.0); // cycle → 1
        let pre_cycle = think.cycle;
        let _ = think.score_hole_closure(3);
        assert_eq!(
            think.cycle, pre_cycle,
            "read-only method must not advance cycle"
        );
    }

    #[test]
    fn think_replay_coherence_does_not_advance_cycle() {
        let mut think = Think::new();
        think.splat_gaussian(1, 1.0, 5.0);
        let prior = think.splat_field.clone();
        let pre_cycle = think.cycle;
        let _ = think.replay_coherence(&prior);
        assert_eq!(think.cycle, pre_cycle);
    }

    #[test]
    fn think_emit_if_epiphany_does_not_advance_cycle() {
        let mut think = Think::new();
        think.splat_gaussian(1, 1.0, 90.0);
        think.splat_gaussian(2, 1.0, 5.0);
        let prior = think.splat_field.clone();
        let pre_cycle = think.cycle;
        let _ = think.emit_if_epiphany(&prior, 0.5, 0.5);
        assert_eq!(think.cycle, pre_cycle, "epiphany detection is read-only");
    }
}
