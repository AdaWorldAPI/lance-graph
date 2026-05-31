//! The **arcuate fasciculus** — the Broca↔Wernicke cable made to carry signal
//! (closes the conduction-aphasia gap, `E-ARCUATE-CONDUCTION`).
//!
//! ## What it does
//!
//! Owns both ends of the cable and slides the projection into the evidence ring:
//! - **Broca / producer** — a `MarkovBundler`. Each fed sentence advances its
//!   window; once saturated it emits a `Trajectory` (the role-superposed
//!   projection wave).
//! - **Wernicke / evidence ring** — a contract `ContextChain` (the ±5 replay
//!   surface). Each emitted projection is sign-binarized and **slid** into the
//!   ring's newest slot.
//!
//! The diagnosis it fixes: `disambiguator_glue` is the cable, and the contract
//! `ContextChain` gives fill + coherence + replay primitives — but **no
//! streaming advance**, and `MarkovBundler::push` had no caller. So the cable
//! existed but carried no signal (production + comprehension intact, *repetition*
//! failing — textbook conduction aphasia). `Arcuate` owns the producer + the
//! ring-slide, so the projection now flows from Broca into Wernicke's window.
//!
//! ## Scope (deliberate, anti-spaghetti)
//!
//! This is a SEPARATE seam — it is **not** wired into `pipeline.rs`'s live
//! 512-bit `ContextWindow`. How the two coexist is a distinct decision; fusing
//! them here would be the spaghetti the design explicitly avoids. The connector
//! is offline-testable on its own.
//!
//! ## Firewall
//!
//! The only thing crossing into the contract is a `Binary16K` fingerprint (the
//! sign-binarized projection) — never a COCA rank. The contract takes no
//! `deepnsm` dependency; `deepnsm` injects through the existing fingerprint seam.
//! Double-windowing note: the bundler does ±radius bundling and the chain is ±5,
//! so the ring holds windowed-projection fingerprints — adequate to carry signal;
//! whether per-sentence (radius-0) fingerprints are preferable is `OQ-ARC-WINDOW`.

use crate::disambiguator_glue::sign_binarize_to_binary16k;
use crate::markov_bundle::{Kernel, MarkovBundler, WindowedSentence};
use crate::trajectory::Trajectory;

use lance_graph_contract::crystal::fingerprint::CrystalFingerprint;
use lance_graph_contract::grammar::context_chain::{
    ContextChain, DisambiguateOpts, DisambiguationResult,
};

/// The arcuate connector: a `MarkovBundler` producer feeding a ±5
/// `ContextChain` evidence ring.
pub struct Arcuate {
    bundler: MarkovBundler,
    chain: ContextChain,
}

impl Arcuate {
    /// New connector with bundler `radius` + `kernel`. The ring is the
    /// contract's fixed ±5 (`CHAIN_LEN = 11`), independent of `radius`.
    #[must_use]
    pub fn new(radius: u32, kernel: Kernel) -> Self {
        Self {
            bundler: MarkovBundler::new(radius, kernel),
            chain: ContextChain::new(),
        }
    }

    /// Feed one sentence's windowed tokens (Broca). When the bundler window
    /// saturates it emits a `Trajectory` (the projection); that projection is
    /// sign-binarized and slid into the ±5 ring (the cable carrying signal).
    /// Returns the emitted projection, or `None` while the window still fills.
    pub fn feed(&mut self, sentence: WindowedSentence) -> Option<Trajectory> {
        let traj = self.bundler.push(sentence)?;
        let bits = sign_binarize_to_binary16k(&traj.fingerprint);
        self.slide_in(CrystalFingerprint::Binary16K(bits));
        Some(traj)
    }

    /// Slide the ±5 ring forward by one: drop the oldest slot, append the
    /// newest (so the newest sits at the last index and the focal at index 5
    /// trails it by five). The contract offers no streaming advance, so the
    /// connector owns the ring via the chain's public `fingerprints`.
    fn slide_in(&mut self, fp: CrystalFingerprint) {
        self.chain.fingerprints.remove(0);
        self.chain.fingerprints.push(Some(fp));
    }

    /// The ±5 evidence ring (Wernicke's replay surface).
    #[must_use]
    pub fn chain(&self) -> &ContextChain {
        &self.chain
    }

    /// Disambiguate at the focal position against `candidates` (the ±5 replay).
    /// Delegates to the contract chain; the populated ring is the evidence.
    pub fn disambiguate<I>(&self, candidates: I) -> DisambiguationResult
    where
        I: IntoIterator<Item = CrystalFingerprint>,
    {
        self.chain.disambiguate_with(
            ContextChain::focal_index(),
            candidates,
            DisambiguateOpts::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markov_bundle::{GrammaticalRole, TokenWithRole};

    /// One SUBJECT-band sentence whose content distinguishes it by `value`.
    fn sentence(value: f32) -> WindowedSentence {
        let len = GrammaticalRole::Subject.slice().len();
        WindowedSentence {
            tokens: vec![TokenWithRole {
                content_fp: vec![value; len],
                role: GrammaticalRole::Subject,
            }],
        }
    }

    fn fp(value: f32) -> CrystalFingerprint {
        CrystalFingerprint::Binary16K(sign_binarize_to_binary16k(&vec![value; 16_384]))
    }

    #[test]
    fn window_fills_before_first_projection() {
        // radius 2 → bundler needs 2*2+1 = 5 pushes before the first emit.
        let mut arc = Arcuate::new(2, Kernel::Uniform);
        for _ in 0..4 {
            assert!(arc.feed(sentence(1.0)).is_none());
        }
        assert_eq!(arc.chain().filled(), 0, "ring untouched while window fills");
    }

    #[test]
    fn projection_slides_into_ring_when_window_saturates() {
        let mut arc = Arcuate::new(2, Kernel::Uniform);
        let mut emitted = None;
        for _ in 0..5 {
            emitted = arc.feed(sentence(1.0));
        }
        assert!(emitted.is_some(), "5th feed (radius 2) emits a projection");
        assert_eq!(arc.chain().filled(), 1, "one projection slid into the ring");
        // Newest occupies the last slot; ring length stays at the contract ±5.
        assert_eq!(arc.chain().fingerprints.len(), 11);
        assert!(arc.chain().fingerprints[10].is_some(), "newest at the tail");
    }

    #[test]
    fn ring_saturates_and_keeps_length_after_many_feeds() {
        let mut arc = Arcuate::new(2, Kernel::Uniform);
        // 5 to warm up + 11 emits to fill all slots = 16 feeds; vary content so
        // the ring is not degenerate.
        for i in 0..16 {
            arc.feed(sentence(1.0 + i as f32));
        }
        assert_eq!(arc.chain().fingerprints.len(), 11);
        assert!(arc.chain().is_saturated(), "ring full after ≥15 emits");
        assert!(arc.chain().focal().is_some(), "focal slot carries signal");
    }

    #[test]
    fn disambiguate_over_populated_ring_ranks_candidates() {
        let mut arc = Arcuate::new(2, Kernel::Uniform);
        for i in 0..16 {
            arc.feed(sentence(1.0 + i as f32));
        }
        let result = arc.disambiguate([fp(1.0), fp(-1.0)]);
        assert_eq!(result.candidate_count, 2, "both candidates evaluated");
        assert!(result.winner_index < 2, "a real winner over the ±5 evidence");
    }
}
