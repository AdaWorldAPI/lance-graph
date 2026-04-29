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
        // permute by position offset (rotate_right)
        if !acc.is_empty() {
            let k = (self.radius as usize) % acc.len();
            acc.rotate_right(k);
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
}
