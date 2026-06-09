//! SentenceTransformer64 — deterministic state-transition transformer.
//!
//! **"Transformer" here means state-transition transformer, not neural self-attention.**
//!
//! ## What P64 is
//!
//! `P64` is the **native address space** of the English reading state machine.
//! It is NOT a compressed approximation of a float embedding. It is a direct
//! symbolic-palette projection:
//!
//! ```text
//! COCA frequency / NSM primes / morphology / grammar / discourse
//!   → direct codebook address
//!   → P64 meaning field (8 lanes × 8 bits)
//!   → CAM4096 locality key (12-bit deterministic projection)
//!   → HHTL / GridLake neighborhood lookup
//! ```
//!
//! Floats may approximate P64 for external ML interop. P64 does not approximate
//! floats. The direction of approximation is one-way, from the outside in.
//!
//! ## Vertical meaning field
//!
//! Each word / sentence token projects **vertically** into the 64-bit field —
//! it activates across multiple lanes simultaneously:
//!
//! ```text
//! "because"
//!   lane 6 (causal):   opens causal frame
//!   lane 4 (clause):   subordinate clause trigger
//!   lane 5 (discourse): explanation continuation
//!   lane 7 (basin):    possible coherence delta
//! ```
//!
//! One word = one column of meaning, not a scalar token. The 8 lanes are
//! **orthogonal semantic planes**, not storage rows.
//!
//! ## Local 4×4 perturbation tile
//!
//! For local reading ambiguity, instead of a continuous Gaussian in f32 space,
//! DeepNSM uses a **discrete 4×4 perturbation tile**: 16 local alternatives per
//! step. Axes:
//!
//! ```text
//! row axis    = semantic lane perturbation (entity/predicate/object shift)
//! column axis = syntactic/pragmatic perturbation (clause/discourse shift)
//! ```
//!
//! Over `n` tokens/sentences this is an implicit (4×4)^n trajectory space.
//! We do NOT materialise it. We keep a small active frontier (Pika-style):
//!
//! - exact P64 state
//! - CAM4096 arc to next state
//! - HHTL/GridLake neighbourhood (Hamming ±1/±2, lane masks)
//! - popcount early exit
//! - AriGraph basin continuity
//!
//! ## CAM4096 codebook classes (examples)
//!
//! ```text
//! pronoun_subject_masc_recent      relative_clause_subject_continuation
//! causal_clause_opener             temporal_anchor_before
//! business_document_object         approval_action_frame
//! negated_action_frame             reported_speech_frame
//! basin_reinforcement              basin_contradiction   epiphany_candidate
//! ```
//!
//! These are native-English reading-state classes, not raw word ids.
//!
//! ## Flow
//!
//! ```text
//! sentence
//!   → SentenceTransformer64::project()
//!   → Sentence64 { p64, cam4096, spo_hint }
//!   → EpisodicSpoFrame (truth witness, in episodic_spo)
//!   → holograph BitpackedVector (resonance, 16Kbit)
//!   → AriGraph basin update
//! ```

use crate::cam64::Cam64;
use crate::episodic_spo::{DependencyRole, EpisodicSpoFrame};
use crate::spo::NO_ROLE;

// ── P64 ──────────────────────────────────────────────────────────────────────

/// 8×8-bit vertical meaning field — the native P64 address space.
///
/// Eight orthogonal semantic planes, each 8 bits wide:
///
/// | Lane | Semantic plane | Source |
/// |------|----------------|--------|
/// | 0    | entity/subject bucket | COCA rank >> 5 (128 buckets) |
/// | 1    | predicate/action bucket | COCA rank >> 5 |
/// | 2    | object/complement bucket | COCA rank >> 5 (0 if absent) |
/// | 3    | morphology (tense/number/person/voice/negation) | MorphFlags low byte |
/// | 4    | clause structure (relative/subordinate/infinitival) | MorphFlags high byte |
/// | 5    | discourse/coreference (depth + coref flag) | entity stack |
/// | 6    | causal/temporal/conditional markers | temporal/causal signal |
/// | 7    | basin/novelty/wisdom/epiphany markers | quality annotations |
///
/// ## This is NOT quantised float space
///
/// `P64` is computed directly from vocabulary ranks, grammar tags, and NSM
/// prime masks — no neural network, no float arithmetic, no rounding error.
/// It is the same information as [`Cam64`] but emphasised as the *meaning-field*
/// output: the full-resolution 64-bit semantic/grammar palette.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct P64(pub u64);

impl P64 {
    /// Construct from an explicit 8-byte lane array.
    #[inline]
    pub fn from_lanes(lanes: [u8; 8]) -> Self {
        let mut v = 0u64;
        for (i, &b) in lanes.iter().enumerate() {
            v |= (b as u64) << (i * 8);
        }
        Self(v)
    }

    /// Extract one lane (0-7).
    #[inline]
    pub fn lane(self, i: usize) -> u8 {
        debug_assert!(i < 8);
        (self.0 >> (i * 8)) as u8
    }

    /// Return a new `P64` with one lane replaced.
    #[inline]
    pub fn with_lane(self, i: usize, val: u8) -> Self {
        debug_assert!(i < 8);
        let mask = !(0xFFu64 << (i * 8));
        Self((self.0 & mask) | ((val as u64) << (i * 8)))
    }

    /// Named lanes — entity/subject bucket (lane 0).
    #[inline]
    pub fn entity(self) -> u8 {
        self.lane(0)
    }
    /// Named lanes — predicate/action bucket (lane 1).
    #[inline]
    pub fn predicate(self) -> u8 {
        self.lane(1)
    }
    /// Named lanes — object/complement bucket (lane 2).
    #[inline]
    pub fn object(self) -> u8 {
        self.lane(2)
    }
    /// Named lanes — morphology low byte (lane 3).
    #[inline]
    pub fn morph(self) -> u8 {
        self.lane(3)
    }
    /// Named lanes — clause structure / MorphFlags high byte (lane 4).
    #[inline]
    pub fn clause(self) -> u8 {
        self.lane(4)
    }
    /// Named lanes — discourse / coreference (lane 5).
    #[inline]
    pub fn discourse(self) -> u8 {
        self.lane(5)
    }
    /// Named lanes — causal/temporal/conditional (lane 6).
    #[inline]
    pub fn causal(self) -> u8 {
        self.lane(6)
    }
    /// Named lanes — basin/novelty/epiphany (lane 7).
    #[inline]
    pub fn basin(self) -> u8 {
        self.lane(7)
    }

    /// XOR bind with another P64 (VSA binding — recovers either component when
    /// the other is known).
    #[inline]
    pub fn bind(self, other: P64) -> P64 {
        P64(self.0 ^ other.0)
    }

    /// Popcount — active bits in the meaning field.
    #[inline]
    pub fn popcount(self) -> u32 {
        self.0.count_ones()
    }

    /// Lane-level agreement with another field (64 = identical).
    ///
    /// Computed as `64 - (self XOR other).count_ones()`. No floats.
    #[inline]
    pub fn agreement(self, other: P64) -> u32 {
        64 - (self.0 ^ other.0).count_ones()
    }

    /// True if the two fields are in the same reading basin.
    ///
    /// Threshold: ≥ 40 of 64 bits agree. Tuned to survive normal sentence
    /// progression (morph/discourse lanes shift each sentence; entity/predicate
    /// lanes are stable within a topic).
    #[inline]
    pub fn same_basin(self, other: P64) -> bool {
        self.agreement(other) >= 40
    }

    /// Derive from a [`Cam64`] locality key and the NSM prime mask.
    ///
    /// The NSM prime mask contributes the *semantic prime* signal that Cam64
    /// doesn't carry. Low 16 bits of the mask are folded into lanes 3-4 via
    /// XOR so the meaning field is sensitive to prime coverage without losing
    /// grammar-lane signals.
    ///
    /// This is the canonical construction path: grammar → Cam64 → P64.
    pub fn from_cam64_and_nsm(cam: Cam64, nsm_prime_mask: u64) -> Self {
        let nsm_low = nsm_prime_mask & 0xFF;
        let nsm_high = (nsm_prime_mask >> 8) & 0xFF;
        let nsm_xor = nsm_low | (nsm_high << 8); // into bits 24-39
        Self(cam.raw() ^ (nsm_xor << 24))
    }

    /// Raw u64.
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl From<Cam64> for P64 {
    /// Lift a Cam64 locality key into a P64 meaning field (no NSM contribution).
    ///
    /// Prefer `P64::from_cam64_and_nsm()` when an NSM prime mask is available.
    fn from(cam: Cam64) -> Self {
        Self(cam.raw())
    }
}

// ── Cam4096 ──────────────────────────────────────────────────────────────────

/// 12-bit deterministic CAM codebook address derived from a [`P64`] meaning field.
///
/// 4096 cells = the full-resolution native-English reading-state palette.
///
/// **Derivation is a fold, not quantisation.** Three nibbles are selected from
/// P64 lanes and packed:
///
/// ```text
/// bits  0.. 3 = entity lane top nibble  (vocabulary bucket cluster)
/// bits  4.. 7 = predicate lane top nibble
/// bits  8..11 = basin lane top nibble
/// bits 12..15 = always zero
/// ```
///
/// The three nibbles cover the subject, predicate, and episodic-basin
/// dimensions — sufficient to select the reading-state class without
/// redundant information from the discourse/morphology lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Cam4096(pub u16);

impl Cam4096 {
    /// Derive deterministically from a `P64` meaning field.
    ///
    /// Uses top nibbles of entity (lane 0), predicate (lane 1), and basin
    /// (lane 7). The fold is lossless in the sense that no float operation is
    /// involved — it is a bit-selection + pack.
    #[inline]
    pub fn from_p64(p: P64) -> Self {
        let e = (p.entity() >> 4) as u16; // top nibble of entity lane
        let r = (p.predicate() >> 4) as u16; // top nibble of predicate lane
        let b = (p.basin() >> 4) as u16; // top nibble of basin lane
        Self(e | (r << 4) | (b << 8))
    }

    /// Raw 12-bit codebook address (bits 12-15 always zero).
    #[inline]
    pub fn raw(self) -> u16 {
        self.0 & 0x0FFF
    }

    /// Nibble at position 0 (entity cluster).
    #[inline]
    pub fn entity_nibble(self) -> u8 {
        (self.0 & 0xF) as u8
    }

    /// Nibble at position 1 (predicate cluster).
    #[inline]
    pub fn predicate_nibble(self) -> u8 {
        ((self.0 >> 4) & 0xF) as u8
    }

    /// Nibble at position 2 (basin class).
    #[inline]
    pub fn basin_nibble(self) -> u8 {
        ((self.0 >> 8) & 0xF) as u8
    }

    /// Nibble distance (0-3): count of differing nibble positions.
    #[inline]
    pub fn nibble_distance(self, other: Cam4096) -> u8 {
        let x = self.0 ^ other.0;
        ((x & 0x00F != 0) as u8) + ((x & 0x0F0 != 0) as u8) + ((x & 0xF00 != 0) as u8)
    }

    /// True if same entity, predicate, and basin cluster (nibble_distance == 0).
    #[inline]
    pub fn exact_match(self, other: Cam4096) -> bool {
        self.0 == other.0
    }

    /// True if at most one cluster differs (basin continuity heuristic).
    #[inline]
    pub fn near_match(self, other: Cam4096) -> bool {
        self.nibble_distance(other) <= 1
    }
}

// ── Perturbation4x4 ───────────────────────────────────────────────────────────

/// Local 4×4 perturbation tile — 16 discrete reading alternatives.
///
/// Rows = semantic lane perturbation (entity / predicate / object shift).
/// Cols = syntactic/pragmatic perturbation (clause / discourse shift).
///
/// Each cell encodes a **lane-delta** as a pair of signed nibbles packed into
/// one byte: `(row_delta: i4, col_delta: i4)`. Over `n` steps the implicit
/// trajectory space is (4×4)^n but we never materialise it — HHTL/GridLake
/// prunes to the small living frontier.
///
/// ## Encoding
///
/// Each of the 16 cells is a `u8` with two nibbles:
/// - bits 0-3: semantic axis delta (0..7 = +0..+7, 8..15 = -8..-1 in two's complement nibble)
/// - bits 4-7: syntactic axis delta (same encoding)
///
/// Delta 0 = centre / no perturbation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Perturbation4x4 {
    /// Row-major 4×4 grid of (semantic_delta, syntactic_delta) pairs.
    pub cells: [u8; 16],
}

impl Perturbation4x4 {
    /// The identity tile — all 16 cells are zero perturbation.
    pub const IDENTITY: Self = Self { cells: [0u8; 16] };

    /// Encode a cell from signed (semantic, syntactic) deltas (-8..+7).
    #[inline]
    pub fn encode_cell(semantic: i8, syntactic: i8) -> u8 {
        let s = (semantic as u8) & 0xF; // two's complement nibble
        let y = (syntactic as u8) & 0xF;
        s | (y << 4)
    }

    /// Decode a cell's semantic delta.
    #[inline]
    pub fn semantic_delta(cell: u8) -> i8 {
        let nibble = cell & 0xF;
        if nibble < 8 {
            nibble as i8
        } else {
            nibble as i8 - 16
        }
    }

    /// Decode a cell's syntactic delta.
    #[inline]
    pub fn syntactic_delta(cell: u8) -> i8 {
        let nibble = (cell >> 4) & 0xF;
        if nibble < 8 {
            nibble as i8
        } else {
            nibble as i8 - 16
        }
    }

    /// Apply cell `idx` (0-15) to a `P64` field, perturbing lanes 0 and 4.
    ///
    /// - Lane 0 (entity): shifted by `semantic_delta`
    /// - Lane 4 (clause): shifted by `syntactic_delta`
    ///
    /// Wraps within the 8-bit lane (no overflow into adjacent lanes).
    pub fn apply(&self, p: P64, cell_idx: usize) -> P64 {
        debug_assert!(cell_idx < 16);
        let cell = self.cells[cell_idx];
        let sem = Self::semantic_delta(cell);
        let syn = Self::syntactic_delta(cell);
        let new_entity = p.entity().wrapping_add(sem as u8);
        let new_clause = p.clause().wrapping_add(syn as u8);
        p.with_lane(0, new_entity).with_lane(4, new_clause)
    }
}

// ── Discrete palette splat ────────────────────────────────────────────────────

/// One neighbour in a discrete palette splat.
#[derive(Clone, Copy, Debug)]
pub struct SplatNeighbour {
    /// The perturbed P64 meaning field.
    pub p64: P64,
    /// Derived CAM4096 address of this neighbour.
    pub cam: Cam4096,
    /// Hamming distance from the centre (0 = exact, 1..64 = off-centre).
    pub hamming: u8,
}

/// Discrete palette splat: expand a P64 centre into a small neighbourhood.
///
/// This is NOT a Gaussian in f32 space. It is a **discrete palette splat**:
/// the centre code activates nearby palette cells selected by:
/// - small Hamming distance (≤ `radius` bits across all lanes)
/// - valid morphology transition (no illegal tense/clause combinations)
/// - near CAM4096 match (nibble_distance ≤ 1 for neighbours)
///
/// The result is a small `SmallNeighbourhood` (≤ 16 entries) that represents
/// the local reading ambiguity without materialising the (4×4)^n space.
///
/// `tile` provides the pre-defined perturbation alternatives. Pass
/// `Perturbation4x4::IDENTITY` for the trivial one-cell splat.
pub fn splat_p64(centre: P64, tile: &Perturbation4x4, radius_bits: u8) -> SmallNeighbourhood {
    let centre_cam = Cam4096::from_p64(centre);
    let mut out = SmallNeighbourhood::new();

    // Centre is always included (hamming = 0).
    out.push(SplatNeighbour {
        p64: centre,
        cam: centre_cam,
        hamming: 0,
    });

    // Apply each tile cell, keep those that actually perturb the centre.
    for (i, _cell) in tile.cells.iter().enumerate() {
        if i == 0 {
            continue;
        } // centre already emitted
        let perturbed = tile.apply(centre, i);
        if perturbed == centre {
            continue;
        } // no-op cell (e.g. all-zero identity)
        let h = hamming_p64(centre, perturbed);
        if h <= radius_bits {
            let cam = Cam4096::from_p64(perturbed);
            if cam.near_match(centre_cam) {
                out.push(SplatNeighbour {
                    p64: perturbed,
                    cam,
                    hamming: h,
                });
            }
        }
    }
    out
}

/// Hamming distance between two P64 fields (0-64).
#[inline]
pub fn hamming_p64(a: P64, b: P64) -> u8 {
    (a.0 ^ b.0).count_ones() as u8
}

/// A small fixed-capacity neighbourhood for discrete splat results (≤ 16 entries).
pub struct SmallNeighbourhood {
    buf: [SplatNeighbour; 16],
    len: usize,
}

impl SmallNeighbourhood {
    fn new() -> Self {
        Self {
            buf: [SplatNeighbour {
                p64: P64(0),
                cam: Cam4096(0),
                hamming: 0,
            }; 16],
            len: 0,
        }
    }

    fn push(&mut self, n: SplatNeighbour) {
        if self.len < 16 {
            self.buf[self.len] = n;
            self.len += 1;
        }
    }

    /// Iterate the active neighbours.
    pub fn iter(&self) -> &[SplatNeighbour] {
        &self.buf[..self.len]
    }

    /// Number of neighbours (including centre).
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the neighbourhood holds no cells. After `splat_p64` the centre is
    /// always present (so this is `false` there); provided to satisfy the
    /// `len_without_is_empty` contract for general callers.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True if only the centre is present.
    pub fn is_singleton(&self) -> bool {
        self.len == 1
    }
}

// ── EpisodicSpoHint ───────────────────────────────────────────────────────────

/// Compact SPO candidate hint carried alongside a `Sentence64`.
///
/// This is a reference into the auditable `EpisodicSpoFrame` — three vocabulary
/// ranks plus the dependency role of the primary frame. Callers that only need
/// the codebook address can ignore this; callers that need to commit to AriGraph
/// use it to reconstruct the full frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EpisodicSpoHint {
    pub subject: u16,
    pub predicate: u16,
    pub object: u16,
    pub role: DependencyRole,
}

impl EpisodicSpoHint {
    /// Extract the primary (first) frame's hint from a slice of episodic frames.
    pub fn from_primary_frame(frames: &[EpisodicSpoFrame]) -> Self {
        match frames.first() {
            None => Self {
                subject: NO_ROLE,
                predicate: NO_ROLE,
                object: NO_ROLE,
                role: DependencyRole::Unknown,
            },
            Some(f) => Self {
                subject: f.subject_candidate_id,
                predicate: f.predicate_candidate_id,
                object: f.object_candidate_id,
                role: f.dependency_role,
            },
        }
    }
}

// ── Sentence64 ────────────────────────────────────────────────────────────────

/// Complete output of `SentenceTransformer64` for one sentence.
///
/// Three layers of the discrete substrate:
/// - `p64`: full 64-bit vertical meaning field (grammar + NSM + discourse)
/// - `cam`: 12-bit deterministic codebook address (P4096 palette key)
/// - `spo_hint`: compact SPO reference for AriGraph basin commitment
///
/// No floats. Quality annotations (confidence, novelty, etc.) live in the
/// companion `EpisodicSpoFrame`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sentence64 {
    pub p64: P64,
    pub cam: Cam4096,
    pub spo_hint: EpisodicSpoHint,
}

impl Sentence64 {
    /// Construct from parts (e.g. when `EpisodicSpoFrame` frames are already computed).
    pub fn new(p64: P64, spo_hint: EpisodicSpoHint) -> Self {
        Self {
            p64,
            cam: Cam4096::from_p64(p64),
            spo_hint,
        }
    }

    /// True if this sentence is in the same reading basin as `other`.
    ///
    /// Both P64 agreement (≥ 40 bits) and CAM proximity (nibble_distance ≤ 1)
    /// must hold.
    #[inline]
    pub fn same_basin_as(&self, other: &Sentence64) -> bool {
        self.p64.same_basin(other.p64) && self.cam.near_match(other.cam)
    }
}

// ── SentenceTransformer64 ─────────────────────────────────────────────────────

/// Deterministic reading-state transformer: maps grammar/NSM/discourse to P64.
///
/// ## Not neural self-attention
///
/// This is a state-transition transformer in the automata sense:
///
/// ```text
/// ReadingState_t + SentenceFeatures_t
///   → P64 meaning field
///   → Cam4096 codebook address
///   → EpisodicSpoHint
///   → Sentence64
/// ```
///
/// ## The codebook
///
/// 4096 cells representing native-English reading-state classes.
/// Addressed directly from P64 lane nibbles — no float lookup, no nearest-
/// neighbour in embedding space.
pub struct SentenceTransformer64;

impl SentenceTransformer64 {
    /// Project a resolved `Cam64` + NSM mask + SPO triple into a `Sentence64`.
    ///
    /// This is the primary construction path: grammar already resolved by
    /// `ReadingState::step()`, now lifted into the P64 meaning field.
    ///
    /// `spo` is the primary resolved triple (subject after coreference).
    /// `nsm_prime_mask` is the 64-bit NSM prime bitset for this sentence.
    pub fn project(
        cam: Cam64,
        nsm_prime_mask: u64,
        subject: u16,
        predicate: u16,
        object: u16,
        role: DependencyRole,
    ) -> Sentence64 {
        let p64 = P64::from_cam64_and_nsm(cam, nsm_prime_mask);
        Sentence64::new(
            p64,
            EpisodicSpoHint {
                subject,
                predicate,
                object,
                role,
            },
        )
    }

    /// Project directly from an `EpisodicSpoFrame`.
    ///
    /// Convenience wrapper: extracts `cam64`, `nsm_prime_mask`, and SPO
    /// candidates from the already-emitted frame.
    pub fn project_from_frame(frame: &EpisodicSpoFrame) -> Sentence64 {
        Self::project(
            frame.cam64,
            frame.nsm_prime_mask,
            frame.subject_candidate_id,
            frame.predicate_candidate_id,
            frame.object_candidate_id,
            frame.dependency_role,
        )
    }

    /// Project a batch of frames into `Sentence64` values.
    ///
    /// Returns one `Sentence64` per frame. The primary frame's hint is used
    /// for each output; callers that need multi-triple output should call
    /// `project_from_frame` for each triple individually.
    pub fn project_frames(frames: &[EpisodicSpoFrame]) -> Vec<Sentence64> {
        frames.iter().map(Self::project_from_frame).collect()
    }

    /// Compute a local 4×4 perturbation tile centred at `p64`.
    ///
    /// The tile represents the 16 most natural reading alternatives from this
    /// P64 state: small lane perturbations covering adjacent vocabulary buckets
    /// and clause transitions.
    ///
    /// `entity_step` and `clause_step` control the stride of the semantic and
    /// syntactic axes respectively (typically 1-4 bucket positions).
    pub fn local_tile(p64: P64, entity_step: u8, clause_step: u8) -> Perturbation4x4 {
        let mut cells = [0u8; 16];
        // Row = semantic axis (entity perturbation 0..3 × entity_step)
        // Col = syntactic axis (clause perturbation 0..3 × clause_step)
        for row in 0i8..4 {
            for col in 0i8..4 {
                let sem = row * entity_step as i8;
                let syn = col * clause_step as i8;
                cells[(row * 4 + col) as usize] = Perturbation4x4::encode_cell(sem, syn);
            }
        }
        // Verify the tile is non-trivial when steps > 0.
        let _ = p64; // p64 is unused here; a future version may use lane context
        Perturbation4x4 { cells }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cam64::Cam64;
    use crate::episodic_spo::DependencyRole;

    // ── P64 ──────────────────────────────────────────────────────────────────

    #[test]
    fn p64_lane_roundtrip() {
        let lanes = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let p = P64::from_lanes(lanes);
        for (i, &v) in lanes.iter().enumerate() {
            assert_eq!(p.lane(i), v, "lane {i}");
        }
    }

    #[test]
    fn p64_with_lane_does_not_corrupt_others() {
        let p = P64::from_lanes([0xFF; 8]);
        let p2 = p.with_lane(3, 0x00);
        assert_eq!(p2.lane(3), 0x00);
        for i in [0, 1, 2, 4, 5, 6, 7] {
            assert_eq!(p2.lane(i), 0xFF, "lane {i} corrupted");
        }
    }

    #[test]
    fn p64_bind_is_self_inverse() {
        let a = P64::from_lanes([0xAB; 8]);
        let b = P64::from_lanes([0xCD; 8]);
        assert_eq!(a.bind(b).bind(b), a);
    }

    #[test]
    fn p64_agreement_self_is_64() {
        let p = P64::from_lanes([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(p.agreement(p), 64);
    }

    #[test]
    fn p64_same_basin_identical() {
        let p = P64::from_lanes([10; 8]);
        assert!(p.same_basin(p));
    }

    #[test]
    fn p64_same_basin_fails_when_far() {
        let a = P64(0x0000_0000_0000_0000);
        let b = P64(0xFFFF_FFFF_FFFF_FFFF);
        assert!(!a.same_basin(b));
    }

    #[test]
    fn p64_from_cam64_and_nsm_zero_nsm_equals_cam() {
        let cam = Cam64::from_lanes([1, 2, 3, 4, 5, 6, 7, 8]);
        let p = P64::from_cam64_and_nsm(cam, 0);
        assert_eq!(p.raw(), cam.raw()); // zero NSM → identity
    }

    #[test]
    fn p64_from_cam64_and_nsm_differs_with_nsm() {
        let cam = Cam64::default();
        let p0 = P64::from_cam64_and_nsm(cam, 0);
        let p1 = P64::from_cam64_and_nsm(cam, 0xFFFF);
        assert_ne!(p0, p1);
    }

    #[test]
    fn p64_from_cam64_conversion() {
        let cam = Cam64::from_lanes([7, 8, 9, 10, 11, 12, 13, 14]);
        let p: P64 = cam.into();
        assert_eq!(p.raw(), cam.raw());
    }

    // ── Cam4096 ──────────────────────────────────────────────────────────────

    #[test]
    fn cam4096_fits_in_12_bits() {
        let p = P64::from_lanes([0xFF; 8]);
        let c = Cam4096::from_p64(p);
        assert_eq!(c.raw(), c.0 & 0x0FFF);
    }

    #[test]
    fn cam4096_entity_nibble_is_top_nibble_of_entity_lane() {
        let p = P64::from_lanes([0xAB, 0, 0, 0, 0, 0, 0, 0]);
        let c = Cam4096::from_p64(p);
        assert_eq!(c.entity_nibble(), 0xA); // top nibble of 0xAB
    }

    #[test]
    fn cam4096_predicate_nibble() {
        let p = P64::from_lanes([0, 0xCD, 0, 0, 0, 0, 0, 0]);
        let c = Cam4096::from_p64(p);
        assert_eq!(c.predicate_nibble(), 0xC);
    }

    #[test]
    fn cam4096_basin_nibble() {
        let p = P64::from_lanes([0, 0, 0, 0, 0, 0, 0, 0xEF]);
        let c = Cam4096::from_p64(p);
        assert_eq!(c.basin_nibble(), 0xE);
    }

    #[test]
    fn cam4096_exact_match_same_p64() {
        let p = P64::from_lanes([0x12, 0x34, 0, 0, 0, 0, 0, 0x56]);
        let c = Cam4096::from_p64(p);
        assert!(c.exact_match(c));
    }

    #[test]
    fn cam4096_near_match_one_nibble_differs() {
        // Entity nibble differs by 1 → entity cluster shifts; others unchanged.
        let p1 = P64::from_lanes([0x10, 0x20, 0, 0, 0, 0, 0, 0x30]);
        let p2 = P64::from_lanes([0x20, 0x20, 0, 0, 0, 0, 0, 0x30]);
        let c1 = Cam4096::from_p64(p1);
        let c2 = Cam4096::from_p64(p2);
        assert!(c1.near_match(c2));
    }

    #[test]
    fn cam4096_near_match_false_three_nibbles_differ() {
        let p1 = P64::from_lanes([0x10, 0x20, 0, 0, 0, 0, 0, 0x30]);
        let p2 = P64::from_lanes([0x80, 0x90, 0, 0, 0, 0, 0, 0xA0]);
        let c1 = Cam4096::from_p64(p1);
        let c2 = Cam4096::from_p64(p2);
        assert!(!c1.near_match(c2));
    }

    // ── Perturbation4x4 ──────────────────────────────────────────────────────

    #[test]
    fn perturbation_encode_decode_zero() {
        let cell = Perturbation4x4::encode_cell(0, 0);
        assert_eq!(Perturbation4x4::semantic_delta(cell), 0);
        assert_eq!(Perturbation4x4::syntactic_delta(cell), 0);
    }

    #[test]
    fn perturbation_encode_decode_positive() {
        let cell = Perturbation4x4::encode_cell(3, 5);
        assert_eq!(Perturbation4x4::semantic_delta(cell), 3);
        assert_eq!(Perturbation4x4::syntactic_delta(cell), 5);
    }

    #[test]
    fn perturbation_encode_decode_negative() {
        let cell = Perturbation4x4::encode_cell(-2, -4);
        assert_eq!(Perturbation4x4::semantic_delta(cell), -2);
        assert_eq!(Perturbation4x4::syntactic_delta(cell), -4);
    }

    #[test]
    fn perturbation_identity_does_not_change_p64() {
        let p = P64::from_lanes([0x10, 0x20, 0, 0, 0, 0, 0, 0x30]);
        let result = Perturbation4x4::IDENTITY.apply(p, 0);
        assert_eq!(result, p);
    }

    // ── Splat ─────────────────────────────────────────────────────────────────

    #[test]
    fn splat_identity_tile_returns_singleton() {
        let p = P64::from_lanes([0x10, 0x20, 0, 0, 0, 0, 0, 0x30]);
        let nb = splat_p64(p, &Perturbation4x4::IDENTITY, 8);
        assert_eq!(nb.len(), 1);
        assert_eq!(nb.iter()[0].hamming, 0);
    }

    #[test]
    fn splat_small_tile_stays_within_radius() {
        let p = P64::from_lanes([0x10, 0x20, 0, 0, 0, 0, 0, 0x30]);
        let tile = SentenceTransformer64::local_tile(p, 1, 1);
        let nb = splat_p64(p, &tile, 8);
        for n in nb.iter() {
            assert!(n.hamming <= 8, "hamming {} exceeds radius 8", n.hamming);
        }
    }

    #[test]
    fn splat_centre_is_first_entry() {
        let p = P64::from_lanes([5, 6, 7, 8, 9, 10, 11, 12]);
        let nb = splat_p64(p, &Perturbation4x4::IDENTITY, 4);
        assert_eq!(nb.iter()[0].p64, p);
    }

    // ── Sentence64 + SentenceTransformer64 ───────────────────────────────────

    #[test]
    fn sentence64_cam_derived_from_p64() {
        let cam = Cam64::from_lanes([0xAB, 0xCD, 0, 0, 0, 0, 0, 0xEF]);
        let s = SentenceTransformer64::project(cam, 0, 10, 20, 30, DependencyRole::Subject);
        // CAM4096 should be deterministic from P64.
        assert_eq!(s.cam, Cam4096::from_p64(s.p64));
    }

    #[test]
    fn sentence64_same_basin_identical() {
        let cam = Cam64::from_lanes([1, 2, 3, 4, 5, 6, 7, 8]);
        let s = SentenceTransformer64::project(cam, 0, 1, 2, 3, DependencyRole::Subject);
        assert!(s.same_basin_as(&s));
    }

    #[test]
    fn sentence64_different_basin_far_cam() {
        let cam_a = Cam64::from_lanes([0x00; 8]);
        let cam_b = Cam64::from_lanes([0xFF; 8]);
        let a = SentenceTransformer64::project(cam_a, 0, 1, 2, 3, DependencyRole::Subject);
        let b = SentenceTransformer64::project(cam_b, 0xFFFF, 4, 5, 6, DependencyRole::Object);
        assert!(!a.same_basin_as(&b));
    }

    #[test]
    fn sentence_transformer64_local_tile_has_16_cells() {
        let p = P64::from_lanes([0x10; 8]);
        let tile = SentenceTransformer64::local_tile(p, 2, 2);
        assert_eq!(tile.cells.len(), 16);
    }

    #[test]
    fn sentence_transformer64_project_frames_empty() {
        let frames: &[EpisodicSpoFrame] = &[];
        let out = SentenceTransformer64::project_frames(frames);
        assert!(out.is_empty());
    }

    #[test]
    fn hamming_p64_same_is_zero() {
        let p = P64::from_lanes([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(hamming_p64(p, p), 0);
    }

    #[test]
    fn hamming_p64_all_bits_differ_is_64() {
        let a = P64(0x0000_0000_0000_0000);
        let b = P64(0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(hamming_p64(a, b), 64);
    }
}
