//! Splat op fleet (D-CSV-12) — scalar implementations.
//! Per cognitive-substrate-convergence-v1.md §11 D-CSV-12.
//!
//! Sprint-12 scope: free functions taking `&mut [SplatField]`. Sprint-13
//! D-CSV-14 migrated these to methods on the `Think` carrier (see
//! `crate::think::Think`). The free functions below are **deprecated** and
//! remain for one release cycle per the deprecation strategy in spec
//! `.claude/specs/pr-sprint-13-think-methods.md` §4. They will be removed
//! in sprint-15+ once all callers have migrated.
//!
//! Canonical surface: `Think::splat_gaussian`, `Think::score_hole_closure`,
//! `Think::replay_coherence`, `Think::emit_if_epiphany`.

use crate::think::Think;

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
///
/// # Deprecation
///
/// Use `Think::splat_gaussian` instead. Construct a `Think` via
/// `Think::from_field(std::mem::take(field), generation.saturating_sub(1))`
/// and call `.splat_gaussian(mean, variance, energy)`. This shim performs
/// that delegation and discards the supplied `generation` (the method
/// derives generation from `Think::cycle`).
#[deprecated(
    since = "0.2.0",
    note = "Migrated to Think::splat_gaussian per D-CSV-14 / \
            CLAUDE.md 'Thinking is a struct' doctrine. The generation \
            parameter is now read from Think::cycle. Callers should \
            construct a Think and use the method directly; this shim \
            forwards to the method and discards the supplied generation."
)]
pub fn splat_gaussian(
    field: &mut Vec<SplatField>,
    mean: u32,
    variance: f32,
    energy: f32,
    generation: u32,
) {
    let mut think = Think::from_field(std::mem::take(field), generation.saturating_sub(1));
    think.splat_gaussian(mean, variance, energy);
    *field = think.splat_field;
}

/// Score the "hole closure" potential of the field: how much of the
/// total energy is concentrated in <=k peaks. Returns a [0.0, 1.0] ratio.
/// High ratio = field has converged; low ratio = scattered.
///
/// # Deprecation
///
/// Use `Think::score_hole_closure` instead. Construct a `Think` via
/// `Think::from_field(field.to_vec(), 0)` and call `.score_hole_closure(k)`.
#[deprecated(
    since = "0.2.0",
    note = "Migrated to Think::score_hole_closure per D-CSV-14 / \
            CLAUDE.md 'Thinking is a struct' doctrine. \
            Use Think::from_field(field.to_vec(), 0).score_hole_closure(k)."
)]
pub fn score_hole_closure(field: &[SplatField], k: usize) -> f32 {
    Think::from_field(field.to_vec(), 0).score_hole_closure(k)
}

/// Replay-coherence: how closely the current field matches a prior generation's
/// shape. Returns cosine-similarity-style metric in [-1, +1] of energy vectors
/// (aligned by mean). Used to detect "this thought is repeating".
///
/// # Deprecation
///
/// Use `Think::replay_coherence` instead. Construct a `Think` via
/// `Think::from_field(current.to_vec(), 0)` and call `.replay_coherence(prior)`.
#[deprecated(
    since = "0.2.0",
    note = "Migrated to Think::replay_coherence per D-CSV-14 / \
            CLAUDE.md 'Thinking is a struct' doctrine. \
            Use Think::from_field(current.to_vec(), 0).replay_coherence(prior)."
)]
pub fn replay_coherence(current: &[SplatField], prior: &[SplatField]) -> f32 {
    Think::from_field(current.to_vec(), 0).replay_coherence(prior)
}

/// Decide whether the field state qualifies as an "epiphany emission":
/// hole-closure above threshold AND replay-coherence above similarity_floor.
/// Returns `Some(top_splat)` if epiphany detected, `None` otherwise.
///
/// # Deprecation
///
/// Use `Think::emit_if_epiphany` instead. Construct a `Think` via
/// `Think::from_field(field.to_vec(), 0)` and call
/// `.emit_if_epiphany(prior, closure_threshold, similarity_floor)`.
#[deprecated(
    since = "0.2.0",
    note = "Migrated to Think::emit_if_epiphany per D-CSV-14 / \
            CLAUDE.md 'Thinking is a struct' doctrine. \
            Use Think::from_field(field.to_vec(), 0).emit_if_epiphany(prior, ct, sf)."
)]
pub fn emit_if_epiphany(
    field: &[SplatField],
    prior: &[SplatField],
    closure_threshold: f32,
    similarity_floor: f32,
) -> Option<SplatField> {
    Think::from_field(field.to_vec(), 0).emit_if_epiphany(
        prior,
        closure_threshold,
        similarity_floor,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 4 deprecation-shim forwarding tests (§5.2) ────────────────────────────
    // Each asserts the deprecated free fn produces the same result as calling
    // the equivalent Think method directly. Uses #[allow(deprecated)] since
    // the tests intentionally exercise the deprecated surface.

    #[test]
    #[allow(deprecated)]
    fn deprecated_splat_gaussian_forwards_to_method() {
        // Build field f and its clone f_method, apply each path, compare.
        let mut field_shim: Vec<SplatField> = vec![SplatField {
            mean: 5,
            variance: 2.0,
            energy: 4.0,
            generation: 0,
        }];
        // generation=3 → shim seeds Think at cycle=2, advance→3
        splat_gaussian(&mut field_shim, 5, 6.0, 4.0, 3);

        // Method path: seed at cycle=2, call splat_gaussian
        let field_method_init = vec![SplatField {
            mean: 5,
            variance: 2.0,
            energy: 4.0,
            generation: 0,
        }];
        let mut think = Think::from_field(field_method_init, 2);
        think.splat_gaussian(5, 6.0, 4.0);

        assert_eq!(field_shim.len(), think.splat_field.len());
        assert_eq!(field_shim[0].mean, think.splat_field[0].mean);
        assert!(
            (field_shim[0].energy - think.splat_field[0].energy).abs() < 1e-6,
            "shim energy {} != method energy {}",
            field_shim[0].energy,
            think.splat_field[0].energy
        );
        assert!(
            (field_shim[0].variance - think.splat_field[0].variance).abs() < 1e-5,
            "shim variance {} != method variance {}",
            field_shim[0].variance,
            think.splat_field[0].variance
        );
        assert_eq!(field_shim[0].generation, think.splat_field[0].generation);
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_score_hole_closure_forwards_to_method() {
        let field = vec![
            SplatField {
                mean: 1,
                variance: 1.0,
                energy: 8.0,
                generation: 1,
            },
            SplatField {
                mean: 2,
                variance: 1.0,
                energy: 1.0,
                generation: 1,
            },
            SplatField {
                mean: 3,
                variance: 1.0,
                energy: 1.0,
                generation: 1,
            },
        ];
        let k = 2;
        let shim_result = score_hole_closure(&field, k);
        let method_result = Think::from_field(field.to_vec(), 0).score_hole_closure(k);
        assert!(
            (shim_result - method_result).abs() < 1e-6,
            "shim={shim_result} method={method_result}"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_replay_coherence_forwards_to_method() {
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
        let shim_result = replay_coherence(&current, &prior);
        let method_result = Think::from_field(current.to_vec(), 0).replay_coherence(&prior);
        assert!(
            (shim_result - method_result).abs() < 1e-6,
            "shim={shim_result} method={method_result}"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_emit_if_epiphany_forwards_to_method() {
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
        let prior = field.clone();
        let ct = 0.5;
        let sf = 0.5;
        let shim_result = emit_if_epiphany(&field, &prior, ct, sf);
        let method_result = Think::from_field(field.to_vec(), 0).emit_if_epiphany(&prior, ct, sf);
        assert_eq!(
            shim_result, method_result,
            "deprecated shim and method must agree: shim={shim_result:?} method={method_result:?}"
        );
    }
}
