//! META-AGENT: add `pub mod markov_bundle;` to lib.rs.
//!
//! Slice coordinates are imported from `lance_graph_contract::grammar::role_keys`
//! so that this module and the rest of the workspace agree on the [start:stop)
//! boundaries of every grammatical role inside the 16384-dim VSA carrier.
//! The previously hard-coded equal-partition layout (`16384 / 5 = 3277` per
//! role) was incompatible with the domain-specific widths in role_keys (e.g.
//! SUBJECT owns [0..2000), TEMPORAL owns [9000..9200)) — see PR #279 review,
//! CRITICAL #1.

use crate::trajectory::Trajectory;

use lance_graph_contract::grammar::role_keys::{
    CONTEXT_SLICE, INSTRUMENT_SLICE, KAUSAL_SLICE, LOKAL_SLICE, MODAL_SLICE,
    MODIFIER_SLICE, OBJECT_SLICE, PREDICATE_SLICE, RoleKeySlice, SUBJECT_SLICE,
    TEMPORAL_SLICE, VSA_DIMS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Kernel {
    Uniform,
    #[default]
    MexicanHat,
    Gaussian,
}

impl Kernel {
    pub fn weight(&self, delta: i32, radius: u32) -> f32 {
        let d = delta.abs() as f32 / radius.max(1) as f32;
        match self {
            Self::Uniform => 1.0,
            Self::MexicanHat => (1.0 - d * d) * (-(d * d) / 2.0).exp(),
            Self::Gaussian => (-(d * d) / 2.0).exp(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammaticalRole {
    Subject,
    Predicate,
    Object,
    Modifier,
    Context,
    Temporal,
    Kausal,
    Modal,
    Lokal,
    Instrument,
}

impl GrammaticalRole {
    /// Canonical [start:stop) slice of the 16384-dim VSA carrier that owns
    /// this role, sourced from `lance_graph_contract::grammar::role_keys`.
    /// This is the single source of truth — the constants below are simple
    /// re-exports of the contract crate's `RoleKeySlice` descriptors so that
    /// every consumer (markov bundler, role-key catalogue, slice-aware
    /// codecs) agrees on the same boundaries.
    pub fn slice(&self) -> RoleKeySlice {
        match self {
            Self::Subject => SUBJECT_SLICE,
            Self::Predicate => PREDICATE_SLICE,
            Self::Object => OBJECT_SLICE,
            Self::Modifier => MODIFIER_SLICE,
            Self::Context => CONTEXT_SLICE,
            // TEKAMOLO sub-slices (NOT inside Context — they live in their
            // own [9000..9650) post-context band per role_keys.rs layout).
            Self::Temporal => TEMPORAL_SLICE,
            Self::Kausal => KAUSAL_SLICE,
            Self::Modal => MODAL_SLICE,
            Self::Lokal => LOKAL_SLICE,
            Self::Instrument => INSTRUMENT_SLICE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenWithRole {
    pub content_fp: Vec<f32>,
    pub role: GrammaticalRole,
}

#[derive(Debug, Clone)]
pub struct WindowedSentence {
    pub tokens: Vec<TokenWithRole>,
}

pub struct MarkovBundler {
    pub radius: u32,
    pub kernel: Kernel,
    pub dims: usize,
    buffer: std::collections::VecDeque<WindowedSentence>,
}

impl MarkovBundler {
    pub fn new(radius: u32, kernel: Kernel) -> Self {
        Self {
            radius,
            kernel,
            // Width of the canonical VSA carrier — kept in lock-step with
            // `lance_graph_contract::grammar::role_keys::VSA_DIMS` (16_384).
            dims: VSA_DIMS,
            buffer: std::collections::VecDeque::with_capacity((2 * radius + 1) as usize),
        }
    }

    pub fn push(&mut self, sentence: WindowedSentence) -> Option<Trajectory> {
        let cap = (2 * self.radius + 1) as usize;
        if self.buffer.len() == cap {
            self.buffer.pop_front();
        }
        self.buffer.push_back(sentence);
        if self.buffer.len() < cap {
            return None;
        }
        Some(self.bundle_current())
    }

    fn bundle_current(&self) -> Trajectory {
        let mut acc = vec![0.0f32; self.dims];
        let focal = self.radius as i32;
        for (i, sent) in self.buffer.iter().enumerate() {
            let delta = (i as i32) - focal;
            let weight = self.kernel.weight(delta, self.radius);
            for tok in &sent.tokens {
                let slice = tok.role.slice();
                // Use the canonical role_keys width (NOT an equal partition).
                let len = slice.len().min(tok.content_fp.len());
                for k in 0..len {
                    acc[slice.start + k] += weight * tok.content_fp[k];
                }
            }
        }
        // REMOVED: post-bundle acc.rotate_right(k) — corrupted role-slice alignment.
        // Plan called for per-sentence pre-bundle vsa_permute; that's a follow-up.
        // Until then, no permutation = aligned bundle.

        // Bundle normalization (HIGH item from PR #279 review): divide by the
        // sum of |kernel weights| so cosine comparisons across kernel choices
        // are invariant to kernel-shape magnitude. Without this, MexicanHat
        // bundles have systematically smaller norms than Uniform bundles
        // simply because the kernel weights peak at 1 and decay.
        let radius_i = self.radius as i32;
        let total_abs_weight: f32 = (-radius_i..=radius_i)
            .map(|d| self.kernel.weight(d, self.radius).abs())
            .sum();
        if total_abs_weight > 1e-9 {
            for v in acc.iter_mut() {
                *v /= total_abs_weight;
            }
        }

        Trajectory {
            fingerprint: acc,
            radius: self.radius,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tok(role: GrammaticalRole, len: usize) -> TokenWithRole {
        TokenWithRole {
            content_fp: vec![1.0; len],
            role,
        }
    }
    #[test]
    fn first_pushes_return_none_until_window_full() {
        let mut b = MarkovBundler::new(5, Kernel::MexicanHat);
        for _ in 0..10 {
            assert!(b
                .push(WindowedSentence {
                    tokens: vec![tok(GrammaticalRole::Subject, 4)]
                })
                .is_none());
        }
        assert!(b
            .push(WindowedSentence {
                tokens: vec![tok(GrammaticalRole::Subject, 4)]
            })
            .is_some());
    }
    #[test]
    fn kernel_uniform_constant() {
        assert_eq!(Kernel::Uniform.weight(0, 5), 1.0);
        assert_eq!(Kernel::Uniform.weight(3, 5), 1.0);
    }
    #[test]
    fn kernel_mexican_symmetric() {
        assert!(
            (Kernel::MexicanHat.weight(-2, 5) - Kernel::MexicanHat.weight(2, 5)).abs() < 1e-6
        );
    }
    #[test]
    fn role_slices_disjoint() {
        // SPO core slices are contiguous: SUBJECT.stop == PREDICATE.start by
        // construction in `role_keys.rs` (0..2000, 2000..4000, ...).
        let s = GrammaticalRole::Subject.slice();
        let p = GrammaticalRole::Predicate.slice();
        assert_eq!(s.stop, p.start);
    }

    #[test]
    fn role_slice_widths_match_role_keys_canonical() {
        // Spot-check that `GrammaticalRole::slice` returns the role_keys-canonical
        // widths (NOT the old equal-partition 16384/5 = 3277 layout).
        assert_eq!(GrammaticalRole::Subject.slice().len(),    2000);
        assert_eq!(GrammaticalRole::Predicate.slice().len(),  2000);
        assert_eq!(GrammaticalRole::Object.slice().len(),     2000);
        assert_eq!(GrammaticalRole::Modifier.slice().len(),   1500);
        assert_eq!(GrammaticalRole::Context.slice().len(),    1500);
        assert_eq!(GrammaticalRole::Temporal.slice().len(),    200);
        assert_eq!(GrammaticalRole::Kausal.slice().len(),      200);
        assert_eq!(GrammaticalRole::Modal.slice().len(),       100);
        assert_eq!(GrammaticalRole::Lokal.slice().len(),       150);
        assert_eq!(GrammaticalRole::Instrument.slice().len(),  100);
    }

    /// Helper: fill a bundler's window so a single push triggers `bundle_current`.
    fn fill_and_bundle(
        kernel: Kernel,
        radius: u32,
        sent: WindowedSentence,
    ) -> Trajectory {
        let mut b = MarkovBundler::new(radius, kernel);
        let cap = (2 * radius + 1) as usize;
        let mut last: Option<Trajectory> = None;
        for _ in 0..cap {
            last = b.push(sent.clone());
        }
        last.expect("bundler should emit a trajectory once window is full")
    }

    /// Helper: push a sequence of distinct sentences so per-position
    /// kernel weights actually shape the bundle. Returns the trajectory
    /// emitted on the final push (window saturated).
    fn bundle_sequence(
        kernel: Kernel,
        radius: u32,
        sentences: Vec<WindowedSentence>,
    ) -> Trajectory {
        let mut b = MarkovBundler::new(radius, kernel);
        let cap = (2 * radius + 1) as usize;
        assert_eq!(sentences.len(), cap, "sequence must fill exactly one window");
        let mut last: Option<Trajectory> = None;
        for s in sentences {
            last = b.push(s);
        }
        last.expect("bundler should emit on the saturating push")
    }

    /// REGRESSION (PR #279 CRITICAL #2): the removed `rotate_right` shifted
    /// SUBJECT-slice content into the PREDICATE slice (or worse, the
    /// CONTEXT band). After the fix, a SUBJECT-only window must keep all
    /// non-zero content inside `[0, 3277)` and have ~zero everywhere else.
    #[test]
    fn bundle_does_not_rotate_subject_dims_outside_subject_slice() {
        // SUBJECT-only window: every sentence has a single SUBJECT token
        // whose content_fp is all 1.0 across the SUBJECT slice.
        let subject_len = GrammaticalRole::Subject.slice().1
            - GrammaticalRole::Subject.slice().0;
        let sent = WindowedSentence {
            tokens: vec![TokenWithRole {
                content_fp: vec![1.0; subject_len],
                role: GrammaticalRole::Subject,
            }],
        };
        let traj = fill_and_bundle(Kernel::Uniform, 5, sent);

        let (s_start, s_stop) = GrammaticalRole::Subject.slice();
        // SUBJECT slice should be non-zero (positive after normalization).
        let subject_sum: f32 =
            traj.fingerprint[s_start..s_stop].iter().sum();
        assert!(
            subject_sum > 1.0,
            "expected non-trivial SUBJECT content, got sum={subject_sum}"
        );
        // Outside the SUBJECT slice every dim must be ~0 (no rotation).
        let outside_max: f32 = traj.fingerprint[s_stop..]
            .iter()
            .fold(0.0f32, |acc, v| acc.max(v.abs()));
        assert!(
            outside_max < 1e-6,
            "rotation leaked SUBJECT content past slice boundary: \
             max |outside| = {outside_max}"
        );
    }

    /// MexicanHat and Uniform kernels must produce materially different
    /// bundles on the same window — otherwise the kernel selector is
    /// ineffective at runtime. Uses an asymmetric heterogeneous window
    /// (one outlier position carries content; others are blank) so that
    /// per-position kernel weights reshape the accumulated bundle in a
    /// way symmetric kernels can't equalize.
    #[test]
    fn mexican_hat_bundle_differs_from_uniform_bundle() {
        let subject_len = GrammaticalRole::Subject.slice().1
            - GrammaticalRole::Subject.slice().0;
        let radius = 5u32;
        let cap = (2 * radius + 1) as usize;
        // Single outlier at position 1 (delta = -4). Uniform weights this
        // identically to focal; MexicanHat strongly attenuates it
        // (w(-4, 5) ≈ 0.26 vs w(0, 5) = 1.0). Normalization divides each
        // by its own Σ|w|, so the per-dim values differ across the
        // SUBJECT slice.
        let outlier_pos = 1usize;
        let sentences: Vec<WindowedSentence> = (0..cap)
            .map(|i| WindowedSentence {
                tokens: vec![TokenWithRole {
                    content_fp: vec![
                        if i == outlier_pos { 1.0 } else { 0.0 };
                        subject_len
                    ],
                    role: GrammaticalRole::Subject,
                }],
            })
            .collect();
        let uni = bundle_sequence(Kernel::Uniform, radius, sentences.clone());
        let mex = bundle_sequence(Kernel::MexicanHat, radius, sentences);
        assert_eq!(uni.fingerprint.len(), mex.fingerprint.len());
        let l2: f32 = uni
            .fingerprint
            .iter()
            .zip(mex.fingerprint.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        assert!(
            l2 > 1e-3,
            "MexicanHat bundle should differ from Uniform bundle, l2={l2}"
        );
    }

    /// Bundle normalization (HIGH from PR #279) makes the L2 norm
    /// invariant to kernel-shape magnitude. We assert all three kernels
    /// land in a loose [0.5, 1.5] band on a controlled SUBJECT-only window.
    #[test]
    fn bundle_l2_norm_invariant_to_kernel() {
        let subject_len = GrammaticalRole::Subject.slice().1
            - GrammaticalRole::Subject.slice().0;
        let sent = WindowedSentence {
            tokens: vec![TokenWithRole {
                content_fp: vec![1.0; subject_len],
                role: GrammaticalRole::Subject,
            }],
        };
        for k in [Kernel::Uniform, Kernel::MexicanHat, Kernel::Gaussian] {
            let traj = fill_and_bundle(k, 5, sent.clone());
            // Per-dim mean of |v| × sqrt(N_subj) ≈ L2 norm; we test L2 directly.
            let l2: f32 = traj
                .fingerprint
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            // Each SUBJECT dim sums to (Σ_i w_i) / (Σ_i |w_i|). For Uniform
            // and Gaussian (all-positive weights) this is exactly 1.0 per dim,
            // so L2 = sqrt(subject_len) ≈ 57.2. For Mexican-hat the negative
            // brim cancels part of the positive core, dropping the per-dim
            // value but keeping it within the same order of magnitude.
            // We loose-bound on L2 / sqrt(subject_len) ∈ [0.5, 1.5].
            let scale = (subject_len as f32).sqrt();
            let norm_l2 = l2 / scale;
            assert!(
                (0.5..=1.5).contains(&norm_l2),
                "kernel {k:?}: normalized L2 {norm_l2} (raw {l2}) out of [0.5, 1.5]"
            );
        }
    }
}
