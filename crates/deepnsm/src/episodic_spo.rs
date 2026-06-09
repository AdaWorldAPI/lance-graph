//! Episodic SPO frame — the auditable sentence-level witness.
//!
//! An `EpisodicSpoFrame` is one row in the reading surface for a single
//! triple within a single sentence. It is the **truth** / **witness**:
//! inspectable, tombs-toneable, AriGraph-committable.
//!
//! The `Cam64` fast-index stored inside each frame is NOT the meaning —
//! it is the reading-locality key. See `cam64` module for the distinction.
//!
//! ## Column layout
//!
//! The column names match the spec verbatim so the SoA projection is
//! mechanical. All fields are `Copy` — the frame is meant to be stacked
//! in `Vec<EpisodicSpoFrame>` and swept SIMD-style.
//!
//! ## Crystallisation lifecycle
//!
//! ```text
//! EpisodicSpoFrame (emitted per sentence)
//!   → repeated SPO detail → story basin candidate
//!   → basin coherent enough → tombstone witness (SPO + Lance columnar)
//!   → new facts classified as: reinforcement / novelty / wisdom / contradiction / epiphany
//! ```

use crate::cam64::Cam64;
use crate::morphology::MorphFlags;
use crate::pos::PoS;

/// Sentinel: "no role / unresolved" for 12-bit vocabulary ranks.
///
/// Re-exported from `crate::spo` so there is a single canonical `0xFFF`
/// definition — removes the drift risk of two independent constants. Existing
/// `use crate::episodic_spo::NO_ROLE` imports keep working unchanged.
pub use crate::spo::NO_ROLE;

// ── Role classification enums ─────────────────────────────────────────────

/// Syntactic dependency role of the head term in this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum DependencyRole {
    #[default]
    Unknown = 0,
    Subject = 1,
    Predicate = 2,
    Object = 3,
    Modifier = 4,
    Complement = 5,
    Adjunct = 6,
    Specifier = 7,
}

/// Clause-level structural role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ClauseRole {
    #[default]
    Main = 0,
    Relative = 1,
    Subordinate = 2,
    Infinitival = 3,
    Participial = 4,
    Coordinate = 5,
}

/// Discourse-level role of the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum DiscourseRole {
    /// Topic of the current discourse segment.
    #[default]
    Topic = 0,
    /// Comment / new information about the topic.
    Comment = 1,
    /// Opening of a new discourse segment / basin.
    Opener = 2,
    /// Closing / summary of an existing basin.
    Closer = 3,
    /// Bridge: connects prior and new discourse segments.
    Bridge = 4,
    /// Background / presupposition.
    Background = 5,
}

// ── EpisodicSpoFrame ─────────────────────────────────────────────────────

/// One auditable episodic SPO row — the reading surface for a single triple.
///
/// All fields are `Copy`. Stack many in `Vec<EpisodicSpoFrame>` for the SoA sweep.
#[derive(Clone, Copy, Debug)]
pub struct EpisodicSpoFrame {
    // ── Position ────────────────────────────────────────────────────────────
    pub doc_id: u32,
    pub sentence_id: u32,
    pub token_span_start: u16, // inclusive byte/token offset of the head term
    pub token_span_end: u16,   // exclusive

    // ── Lexical ─────────────────────────────────────────────────────────────
    /// Vocabulary rank of the head term (lemma_id in the NSM / COCA vocabulary).
    pub term_id: u16,
    pub pos_tag: PoS,
    pub morph_flags: MorphFlags,

    // ── Syntactic ───────────────────────────────────────────────────────────
    pub dependency_role: DependencyRole,
    pub clause_role: ClauseRole,
    pub discourse_role: DiscourseRole,

    // ── SPO candidates ──────────────────────────────────────────────────────
    /// Vocabulary rank of the resolved subject (NO_ROLE if absent).
    pub subject_candidate_id: u16,
    /// Vocabulary rank of the resolved predicate.
    pub predicate_candidate_id: u16,
    /// Vocabulary rank of the resolved object (NO_ROLE if intransitive).
    pub object_candidate_id: u16,
    /// Resolved coreference target (NO_ROLE if not a pronoun or unresolvable).
    pub refers_to_candidate_id: u16,

    // ── Window position ─────────────────────────────────────────────────────
    /// Offset from the current sentence: 0 for current, -1 for prior, etc.
    /// Always 0 at emit time; downstream can patch window-relative offsets.
    pub sentence_window_offset: i8,

    // ── NSM semantic masks ──────────────────────────────────────────────────
    /// Bitmask of NSM semantic primes active in this frame (63 primes → 64 bits).
    pub nsm_prime_mask: u64,
    /// Bitmask of NSM semantic molecules (composite concepts).
    pub nsm_molecule_mask: u64,

    // ── CAM locality ────────────────────────────────────────────────────────
    /// CAM-PQ 6-subspace code for the subject head (6 centroid indices, one per subspace).
    /// Used for fast palette-distance lookups against the codebook.
    pub cam_code: [u8; 6],

    // ── Episodic quality ────────────────────────────────────────────────────
    pub confidence: f32,
    pub frequency: f32,
    pub novelty: f32,
    pub wisdom: f32,
    pub staunen: f32, // aesthetic/cognitive surprise (German: astonishment)
    pub entropy: f32,
    pub free_energy_delta: f32,

    // ── Reading-state locality code (NOT the truth) ──────────────────────
    /// Fast 64-bit reading-locality index. Basin-matching, prefetch, coreference heuristics.
    /// The `subject/predicate/object_candidate_id` fields above are the truth.
    pub cam64: Cam64,
}

impl EpisodicSpoFrame {
    /// Is the subject a pronoun resolved to a prior entity?
    #[inline]
    pub fn is_coreference(&self) -> bool {
        self.refers_to_candidate_id != NO_ROLE
    }

    /// Is this triple intransitive (no object)?
    #[inline]
    pub fn is_intransitive(&self) -> bool {
        self.object_candidate_id == NO_ROLE
    }

    /// Is this triple negated?
    #[inline]
    pub fn is_negated(&self) -> bool {
        self.morph_flags.is_negated()
    }

    /// Is this triple a past-tense event (story-arc candidate)?
    #[inline]
    pub fn is_episodic_event(&self) -> bool {
        self.morph_flags.is_past()
    }

    /// Classification relative to an existing basin.
    ///
    /// V1 heuristic over novelty, entropy, confidence, and wisdom:
    /// - high novelty + low entropy → `Epiphany`
    /// - high novelty (high entropy) → `NoveltyDelta`
    /// - low confidence → `Contradiction`
    /// - high wisdom → `WisdomDelta`
    /// - otherwise → `Reinforcement`
    ///
    /// `BasinClassification::Branch` is intentionally never produced here: a
    /// per-frame heuristic cannot see the parallel narrative line that defines a
    /// branch. It is reserved for the cross-frame basin tracker, which assigns it
    /// when a frame opens a divergent story arc.
    pub fn basin_classification(&self) -> BasinClassification {
        let high_novelty = self.novelty > 0.7;
        let low_entropy = self.entropy < 0.3;

        if high_novelty && low_entropy {
            BasinClassification::Epiphany
        } else if high_novelty {
            BasinClassification::NoveltyDelta
        } else if self.confidence < 0.2 {
            BasinClassification::Contradiction
        } else if self.wisdom > 0.6 {
            BasinClassification::WisdomDelta
        } else {
            BasinClassification::Reinforcement
        }
    }
}

/// How a new episodic frame relates to an existing story basin.
///
/// From the spec:
/// - `Reinforcement` — detail repeats or strengthens the basin
/// - `NoveltyDelta` — surprising new detail that extends the basin
/// - `WisdomDelta` — detail reduces entropy / makes the story simpler/more explanatory
/// - `Contradiction` — detail conflicts with basin
/// - `Branch` — opens a parallel narrative line
/// - `Epiphany` — high novelty + coherence gain (surprising but entropy-reducing)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasinClassification {
    Reinforcement,
    NoveltyDelta,
    WisdomDelta,
    Contradiction,
    Branch,
    Epiphany,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cam64::Cam64;
    use crate::morphology::MorphFlags;
    use crate::pos::PoS;

    fn blank_frame() -> EpisodicSpoFrame {
        EpisodicSpoFrame {
            doc_id: 1,
            sentence_id: 0,
            token_span_start: 0,
            token_span_end: 5,
            term_id: 42,
            pos_tag: PoS::Noun,
            morph_flags: MorphFlags::default(),
            dependency_role: DependencyRole::Subject,
            clause_role: ClauseRole::Main,
            discourse_role: DiscourseRole::Topic,
            subject_candidate_id: 10,
            predicate_candidate_id: 20,
            object_candidate_id: 30,
            refers_to_candidate_id: NO_ROLE,
            sentence_window_offset: 0,
            nsm_prime_mask: 0,
            nsm_molecule_mask: 0,
            cam_code: [0; 6],
            confidence: 0.9,
            frequency: 0.5,
            novelty: 0.1,
            wisdom: 0.3,
            staunen: 0.0,
            entropy: 0.5,
            free_energy_delta: 0.0,
            cam64: Cam64::default(),
        }
    }

    #[test]
    fn coreference_detection() {
        let mut f = blank_frame();
        assert!(!f.is_coreference());
        f.refers_to_candidate_id = 5;
        assert!(f.is_coreference());
    }

    #[test]
    fn intransitive_detection() {
        let mut f = blank_frame();
        assert!(!f.is_intransitive());
        f.object_candidate_id = NO_ROLE;
        assert!(f.is_intransitive());
    }

    #[test]
    fn negated_via_morph() {
        let mut f = blank_frame();
        assert!(!f.is_negated());
        f.morph_flags = f.morph_flags.set(MorphFlags::NEGATED);
        assert!(f.is_negated());
    }

    #[test]
    fn episodic_event_via_past() {
        let mut f = blank_frame();
        assert!(!f.is_episodic_event());
        f.morph_flags = f.morph_flags.set(MorphFlags::PAST);
        assert!(f.is_episodic_event());
    }

    #[test]
    fn epiphany_high_novelty_low_entropy() {
        let mut f = blank_frame();
        f.novelty = 0.9;
        f.entropy = 0.1;
        assert_eq!(f.basin_classification(), BasinClassification::Epiphany);
    }

    #[test]
    fn reinforcement_low_novelty() {
        let mut f = blank_frame();
        f.novelty = 0.1;
        f.entropy = 0.5;
        f.confidence = 0.9;
        assert_eq!(f.basin_classification(), BasinClassification::Reinforcement);
    }

    #[test]
    fn contradiction_low_confidence() {
        let mut f = blank_frame();
        f.novelty = 0.1;
        f.confidence = 0.1;
        assert_eq!(f.basin_classification(), BasinClassification::Contradiction);
    }

    #[test]
    fn size_is_reasonable() {
        // Frame should be small enough to stack efficiently.
        // Current layout: ~80 bytes is acceptable; alert if it balloons.
        let size = core::mem::size_of::<EpisodicSpoFrame>();
        assert!(
            size <= 128,
            "EpisodicSpoFrame grew to {size} bytes — check alignment/padding"
        );
    }
}
