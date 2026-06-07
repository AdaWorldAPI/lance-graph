//! Reading state machine — sentence-by-sentence AriGraph reader.
//!
//! ## Left-corner framing
//!
//! `step()` is a left-corner transition: it fuses top-down expectation
//! (what the current reading frame predicts) with bottom-up evidence
//! (what the parser saw in this sentence) to emit episodic SPO frames
//! and advance the state.
//!
//! A pure bottom-up parser says "I saw these words, build structure."
//! A pure top-down parser says "I expect this form, fill it."
//! A left-corner parser (Manning & Carpenter 1997) does the useful hybrid:
//!
//! ```text
//! ReadingState_t                  (top-down expectation)
//!   + SentenceFeatures_t          (bottom-up evidence)
//!   + LeftCornerTrigger           (first strong frame signal)
//!   + ±5 sentence window          (entity / coreference stack)
//! → Vec<EpisodicSpoFrame>         (auditable witnesses)
//! → ReadingState_t+1              (updated expectation)
//! ```
//!
//! The `Cam64` in each emitted frame encodes the reading-state locality,
//! not semantic truth. The SPO fields in `EpisodicSpoFrame` are the truth.
//!
//! ## Coreference (v1)
//!
//! When the caller marks `TripleFeatures::subject_is_pronoun = true`, the
//! state machine resolves the subject against the `SentenceWindow` entity
//! stack (most-recent-first, recency heuristic). Resolution sets
//! `refers_to_candidate_id` and the coreference bit in `Cam64`.
//! Full antecedent-ranking (gender / number / semantic-type agreement)
//! is a v2 concern.

use crate::cam64::Cam64;
use crate::episodic_spo::{
    ClauseRole, DependencyRole, DiscourseRole, EpisodicSpoFrame, NO_ROLE,
};
use crate::morphology::MorphFlags;
use crate::parser::SentenceStructure;
use crate::pos::PoS;
use crate::window::{SentenceWindow, WindowEntry};

// ── Left-corner trigger ───────────────────────────────────────────────────

/// The first strong signal of a semantic frame at the start of a sentence.
///
/// Used to set the top-down expectation in `ReadingState` before the full
/// SPO parse completes. Maps vocabulary rank or morphological pattern to
/// a frame type, enabling O(1) frame pre-selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LeftCornerTrigger {
    /// No special trigger — plain declarative SVO frame.
    #[default]
    Declarative,
    /// Causal connector ("because", "therefore", "so") → causal explanation frame.
    Causal,
    /// Temporal marker ("after", "before", "when", "then") → temporal ordering frame.
    Temporal,
    /// Relative pronoun ("who", "which", "that") → relative-clause + coreference frame.
    Relative,
    /// Personal pronoun subject ("he", "she", "it", "they") → anaphora lookup.
    Anaphora,
    /// First-person subject ("I", "we") → agent-perspective frame.
    FirstPerson,
    /// Domain-specific trigger from caller (e.g. "invoice" → business-document frame).
    Domain(u8), // caller-supplied domain tag (0-255)
}

impl LeftCornerTrigger {
    /// Lane-7 basin byte for this trigger (feeds into Cam64 basin lane).
    pub fn basin_byte(self) -> u8 {
        match self {
            Self::Declarative  => 0x00,
            Self::Causal       => 0x01,
            Self::Temporal     => 0x02,
            Self::Relative     => 0x04,
            Self::Anaphora     => 0x08,
            Self::FirstPerson  => 0x10,
            Self::Domain(tag)  => 0x80 | (tag & 0x7F),
        }
    }
}

// ── Per-triple caller-supplied features ──────────────────────────────────

/// Features supplied by the caller for one triple within a sentence.
///
/// The parser produces `SentenceStructure` (grammar signals). The caller
/// supplies `TripleFeatures` (semantic / lexical annotations unavailable
/// to the FSM parser). Both together feed `step()`.
#[derive(Clone, Debug, Default)]
pub struct TripleFeatures {
    /// Byte offsets (token-span) of the head term in the original text.
    pub token_span_start: u16,
    pub token_span_end: u16,
    /// PoS of the head term (if known; defaults to Noun).
    pub pos_tag: Option<PoS>,
    /// Is the subject head a personal pronoun? (enables coreference lookup)
    pub subject_is_pronoun: bool,
    /// Is the object head a personal pronoun?
    pub object_is_pronoun: bool,
    /// NSM semantic prime mask (63 primes → 64 bits, bit 0-62 = prime index).
    pub nsm_prime_mask: u64,
    /// NSM semantic molecule mask.
    pub nsm_molecule_mask: u64,
    /// CAM-PQ 6-subspace code for the subject head.
    pub cam_code: [u8; 6],
    /// Left-corner trigger for this sentence (first strong frame signal).
    /// Only the first triple's trigger is used to set the frame expectation.
    pub left_corner_trigger: LeftCornerTrigger,
    /// Episodic quality annotations (caller-supplied, 0.0..1.0).
    pub confidence: f32,
    pub frequency: f32,
    pub novelty: f32,
    pub wisdom: f32,
    pub staunen: f32,
    pub entropy: f32,
    pub free_energy_delta: f32,
}

/// All caller-supplied features for a sentence.
#[derive(Clone, Debug, Default)]
pub struct SentenceFeatures {
    /// One entry per triple in `SentenceStructure::triples`.
    /// If shorter than the triple list, missing entries use `TripleFeatures::default()`.
    pub per_triple: Vec<TripleFeatures>,
}

impl SentenceFeatures {
    pub fn get(&self, idx: usize) -> &TripleFeatures {
        self.per_triple.get(idx).unwrap_or(&DEFAULT_TRIPLE_FEATURES)
    }
}

static DEFAULT_TRIPLE_FEATURES: TripleFeatures = TripleFeatures {
    token_span_start: 0,
    token_span_end: 0,
    pos_tag: None,
    subject_is_pronoun: false,
    object_is_pronoun: false,
    nsm_prime_mask: 0,
    nsm_molecule_mask: 0,
    cam_code: [0u8; 6],
    left_corner_trigger: LeftCornerTrigger::Declarative,
    confidence: 1.0,
    frequency: 0.5,
    novelty: 0.0,
    wisdom: 0.0,
    staunen: 0.0,
    entropy: 0.5,
    free_energy_delta: 0.0,
};

// ── ReadingState ──────────────────────────────────────────────────────────

/// The complete reading state at sentence boundary `t`.
///
/// `step()` takes `ReadingState_t` + `SentenceFeatures_t` and returns
/// `(frames_t, ReadingState_t+1)`. Pure function; `self` is not mutated.
#[derive(Clone, Debug)]
pub struct ReadingState {
    pub doc_id:      u32,
    pub sentence_id: u32,

    // ── Top-down expectation (left-corner "I expected something") ────────
    /// Vocabulary-rank bucket of the expected subject (NO_ROLE = no expectation).
    pub expected_subject_bucket: u16,
    /// Vocabulary-rank bucket of the expected predicate.
    pub expected_predicate_bucket: u16,
    /// Active frame type set by the left-corner trigger.
    pub active_trigger: LeftCornerTrigger,

    // ── Bottom-up evidence (last resolved triple) ────────────────────────
    pub active_subject:   u16,
    pub active_predicate: u16,
    pub active_object:    u16,

    // ── Entity / coreference stack ───────────────────────────────────────
    entity_stack: [u16; 8],
    entity_stack_len: usize,

    // ── Current Cam64 locality code ──────────────────────────────────────
    pub cam64: Cam64,

    // ── ±5 sentence window ────────────────────────────────────────────────
    pub window: SentenceWindow,
}

impl ReadingState {
    /// Create an initial reading state for a new document.
    pub fn new(doc_id: u32) -> Self {
        Self {
            doc_id,
            sentence_id: 0,
            expected_subject_bucket: NO_ROLE,
            expected_predicate_bucket: NO_ROLE,
            active_trigger: LeftCornerTrigger::Declarative,
            active_subject: NO_ROLE,
            active_predicate: NO_ROLE,
            active_object: NO_ROLE,
            entity_stack: [NO_ROLE; 8],
            entity_stack_len: 0,
            cam64: Cam64::default(),
            window: SentenceWindow::new(),
        }
    }

    /// Advance the reading state by one sentence.
    ///
    /// Returns the episodic SPO frames emitted for this sentence and the
    /// next reading state. Pure: `self` is consumed, `next` is returned.
    ///
    /// `features` provides per-triple annotations unavailable to the FSM
    /// parser (pronoun flags, NSM masks, CAM codes, quality markers).
    pub fn step(
        self,
        sentence: &SentenceStructure,
        features: &SentenceFeatures,
    ) -> (Vec<EpisodicSpoFrame>, ReadingState) {
        let mut frames = Vec::with_capacity(sentence.triples.len().max(1));
        let mut next = self;
        next.sentence_id += 1;

        if sentence.is_empty() {
            return (frames, next);
        }

        // Left-corner trigger from the first triple's features sets the frame.
        let first_feat = features.get(0);
        next.active_trigger = first_feat.left_corner_trigger;

        // Update top-down expectation based on the trigger.
        // Causal/temporal triggers shift the expected predicate toward the
        // connective vocabulary bucket; anaphora trigger signals that we
        // need entity stack lookup for the subject.
        match next.active_trigger {
            LeftCornerTrigger::Causal | LeftCornerTrigger::Temporal => {
                // Expectation: predicate will be a temporal/causal verb.
                // We don't know the exact rank yet, so keep as NO_ROLE
                // but leave the trigger set so cam64 lane 7 carries the signal.
            }
            LeftCornerTrigger::Anaphora | LeftCornerTrigger::Relative => {
                // Expectation: subject is a pronoun, resolve from entity stack.
                // This is set per-triple by features.subject_is_pronoun; the
                // trigger here is the sentence-level pre-selection.
            }
            _ => {}
        }

        let mut window_entry = WindowEntry {
            sentence_id: next.sentence_id,
            ..WindowEntry::default()
        };

        for (triple_idx, triple) in sentence.triples.iter().enumerate() {
            let feat = features.get(triple_idx);
            let morph = MorphFlags::from_sentence_structure(sentence, triple_idx);
            let has_temporal = sentence.is_temporal(triple_idx);

            // ── Coreference resolution ───────────────────────────────────
            let refers_to = if feat.subject_is_pronoun {
                // Left-corner anaphora: resolve from window entity stack.
                let candidate = next.window.resolve_pronoun(triple.subject());
                candidate
            } else {
                NO_ROLE
            };
            let coref_resolved = refers_to != NO_ROLE;

            // ── Effective subject after coreference ──────────────────────
            let effective_subject = if coref_resolved {
                refers_to
            } else {
                triple.subject()
            };

            // ── Build Cam64 locality code ────────────────────────────────
            // Use the effective triple (post-coref) for the entity bucket.
            // The basin lane incorporates the left-corner trigger.
            let stack_depth = next.entity_stack_len.min(127) as u8;
            let base_cam64 = Cam64::from_triple(
                triple,
                morph,
                stack_depth,
                coref_resolved,
                has_temporal,
                feat.novelty > 0.7,
            );
            // Overlay basin lane with the left-corner trigger signal.
            let cam64 = base_cam64.with_lane(7,
                base_cam64.basin_state() | next.active_trigger.basin_byte());

            // ── Discourse role ──────────────────────────────────────────
            let discourse_role = if triple_idx == 0 {
                match next.active_trigger {
                    LeftCornerTrigger::Causal   => DiscourseRole::Comment,
                    LeftCornerTrigger::Temporal => DiscourseRole::Bridge,
                    LeftCornerTrigger::Anaphora |
                    LeftCornerTrigger::Relative => DiscourseRole::Background,
                    _                           => DiscourseRole::Topic,
                }
            } else {
                DiscourseRole::Comment
            };

            // ── Clause role ─────────────────────────────────────────────
            let clause_role = if morph.is_relative_clause() {
                ClauseRole::Relative
            } else if morph.is_subordinate() {
                ClauseRole::Subordinate
            } else if morph.is_infinitive() {
                ClauseRole::Infinitival
            } else {
                ClauseRole::Main
            };

            // ── Emit frame ───────────────────────────────────────────────
            let frame = EpisodicSpoFrame {
                doc_id:           next.doc_id,
                sentence_id:      next.sentence_id,
                token_span_start: feat.token_span_start,
                token_span_end:   feat.token_span_end,
                term_id:          effective_subject,
                pos_tag:          feat.pos_tag.unwrap_or(PoS::Noun),
                morph_flags:      morph,
                dependency_role:  DependencyRole::Subject,
                clause_role,
                discourse_role,
                subject_candidate_id:   effective_subject,
                predicate_candidate_id: triple.predicate(),
                object_candidate_id:    triple.object(),
                refers_to_candidate_id: refers_to,
                sentence_window_offset: 0,
                nsm_prime_mask:    feat.nsm_prime_mask,
                nsm_molecule_mask: feat.nsm_molecule_mask,
                cam_code:          feat.cam_code,
                confidence:        feat.confidence,
                frequency:         feat.frequency,
                novelty:           feat.novelty,
                wisdom:            feat.wisdom,
                staunen:           feat.staunen,
                entropy:           feat.entropy,
                free_energy_delta: feat.free_energy_delta,
                cam64,
            };

            frames.push(frame);

            // ── Update entity stack ──────────────────────────────────────
            // Push non-pronoun heads so future sentences can resolve coreference.
            if !feat.subject_is_pronoun && triple.subject() != NO_ROLE {
                next.push_entity(triple.subject());
                window_entry.push_head(triple.subject());
            }
            if !feat.object_is_pronoun && triple.object() != NO_ROLE {
                next.push_entity(triple.object());
                window_entry.push_head(triple.object());
            }

            // Track the primary (first) triple as the active bottom-up evidence.
            if triple_idx == 0 {
                next.active_subject   = effective_subject;
                next.active_predicate = triple.predicate();
                next.active_object    = triple.object();
                next.cam64            = cam64;

                // Update top-down expectation buckets from what we just saw.
                next.expected_subject_bucket   = (effective_subject >> 5) & 0x7F;
                next.expected_predicate_bucket = (triple.predicate() >> 5) & 0x7F;
            }
        }

        window_entry.primary_spo_packed = if !sentence.triples.is_empty() {
            sentence.triples[0].as_u64()
        } else {
            0
        };
        next.window.push(window_entry);

        (frames, next)
    }

    // ── Entity stack helpers ─────────────────────────────────────────────

    /// Push an entity into the coreference stack (LIFO, bounded at 8).
    /// Evicts the oldest entry when full.
    pub fn push_entity(&mut self, rank: u16) {
        if rank == NO_ROLE {
            return;
        }
        if self.entity_stack_len < 8 {
            self.entity_stack[self.entity_stack_len] = rank;
            self.entity_stack_len += 1;
        } else {
            // Rotate left, drop oldest (index 0), push newest at the end.
            self.entity_stack.rotate_left(1);
            self.entity_stack[7] = rank;
        }
    }

    /// Iterate entity stack from most recent to oldest.
    pub fn entities_recent_first(&self) -> impl Iterator<Item = u16> + '_ {
        self.entity_stack[..self.entity_stack_len].iter().rev().copied()
    }

    /// Number of entities currently in the stack.
    pub fn entity_count(&self) -> usize {
        self.entity_stack_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spo::SpoTriple;

    fn sentence_one_triple(s: u16, p: u16, o: u16) -> SentenceStructure {
        SentenceStructure {
            triples: vec![SpoTriple::new(s, p, o)],
            modifiers: vec![],
            negations: vec![],
            temporals: vec![],
        }
    }

    fn plain_features() -> SentenceFeatures {
        SentenceFeatures {
            per_triple: vec![TripleFeatures {
                confidence: 0.9,
                frequency: 0.5,
                ..Default::default()
            }],
        }
    }

    #[test]
    fn step_increments_sentence_id() {
        let rs = ReadingState::new(1);
        let s = sentence_one_triple(10, 20, 30);
        let (_, rs2) = rs.step(&s, &plain_features());
        assert_eq!(rs2.sentence_id, 1);
        let (_, rs3) = rs2.step(&s, &plain_features());
        assert_eq!(rs3.sentence_id, 2);
    }

    #[test]
    fn active_spo_updated_from_first_triple() {
        let rs = ReadingState::new(0);
        let s = sentence_one_triple(100, 200, 300);
        let (frames, rs2) = rs.step(&s, &plain_features());
        assert_eq!(frames.len(), 1);
        assert_eq!(rs2.active_subject, 100);
        assert_eq!(rs2.active_predicate, 200);
        assert_eq!(rs2.active_object, 300);
    }

    #[test]
    fn entity_stack_grows_with_non_pronoun_heads() {
        let rs = ReadingState::new(0);
        let s = sentence_one_triple(50, 60, 70);
        let (_, rs2) = rs.step(&s, &plain_features());
        // Subject (50) and object (70) both pushed
        assert_eq!(rs2.entity_count(), 2);
    }

    #[test]
    fn pronoun_resolves_to_prior_entity() {
        let rs = ReadingState::new(0);
        // First sentence: introduce entity 50
        let s1 = sentence_one_triple(50, 60, 70);
        let (_, rs2) = rs.step(&s1, &plain_features());

        // Second sentence: subject is a pronoun (rank=5, some pronoun rank)
        let s2 = sentence_one_triple(5, 80, 90);
        let feat = SentenceFeatures {
            per_triple: vec![TripleFeatures {
                subject_is_pronoun: true,
                confidence: 0.8,
                ..Default::default()
            }],
        };
        let (frames, _) = rs2.step(&s2, &feat);
        assert_eq!(frames.len(), 1);
        // Most recent entity was 70 (object of first sentence), then 50.
        // resolve_pronoun returns most recent → 70.
        assert_eq!(frames[0].refers_to_candidate_id, 70);
        // Effective subject = resolved referent.
        assert_eq!(frames[0].subject_candidate_id, 70);
    }

    #[test]
    fn pronoun_no_prior_entity_stays_no_role() {
        let rs = ReadingState::new(0);
        let s = sentence_one_triple(5, 20, 30);
        let feat = SentenceFeatures {
            per_triple: vec![TripleFeatures {
                subject_is_pronoun: true,
                ..Default::default()
            }],
        };
        let (frames, _) = rs.step(&s, &feat);
        assert_eq!(frames[0].refers_to_candidate_id, NO_ROLE);
    }

    #[test]
    fn empty_sentence_emits_no_frames() {
        let rs = ReadingState::new(0);
        let s = SentenceStructure {
            triples: vec![],
            modifiers: vec![],
            negations: vec![],
            temporals: vec![],
        };
        let (frames, rs2) = rs.step(&s, &SentenceFeatures::default());
        assert!(frames.is_empty());
        assert_eq!(rs2.sentence_id, 1);
    }

    #[test]
    fn left_corner_trigger_recorded_in_cam64_basin_lane() {
        let rs = ReadingState::new(0);
        let s = sentence_one_triple(10, 20, 30);
        let feat = SentenceFeatures {
            per_triple: vec![TripleFeatures {
                left_corner_trigger: LeftCornerTrigger::Causal,
                ..Default::default()
            }],
        };
        let (frames, _) = rs.step(&s, &feat);
        // Basin lane must have the causal trigger bit set.
        let basin = frames[0].cam64.basin_state();
        assert_ne!(basin & LeftCornerTrigger::Causal.basin_byte(), 0);
    }

    #[test]
    fn entity_stack_evicts_oldest_at_capacity() {
        let mut rs = ReadingState::new(0);
        for i in 0..9u16 {
            rs.push_entity(i * 10);
        }
        // Stack is bounded at 8; oldest entry (0) should be gone.
        let entities: Vec<u16> = rs.entities_recent_first().collect();
        assert_eq!(entities.len(), 8);
        assert!(!entities.contains(&0), "oldest entity should have been evicted");
        assert!(entities.contains(&80), "newest entity should be present");
    }

    #[test]
    fn window_populated_after_step() {
        let rs = ReadingState::new(0);
        let s = sentence_one_triple(10, 20, 30);
        let (_, rs2) = rs.step(&s, &plain_features());
        assert_eq!(rs2.window.len(), 1);
        let entry = rs2.window.most_recent().unwrap();
        assert!(entry.contains(10)); // subject pushed
        assert!(entry.contains(30)); // object pushed
    }

    #[test]
    fn temporal_sentence_sets_past_morph_in_frame() {
        let rs = ReadingState::new(0);
        let s = SentenceStructure {
            triples: vec![SpoTriple::new(1, 2, 3)],
            modifiers: vec![],
            negations: vec![],
            temporals: vec![(0, 99)], // triple 0 is temporal
        };
        let (frames, _) = rs.step(&s, &plain_features());
        assert!(frames[0].morph_flags.is_past());
    }
}
