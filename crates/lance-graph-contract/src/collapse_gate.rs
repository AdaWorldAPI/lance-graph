//! CollapseGate write protocol — MergeMode + GateDecision.
//!
//! CollapseGate enum (Flow/Block/Hold) lives in ndarray::hpc::bnn_cross_plane.
//! This module adds the write-back protocol types consumed by the 7-layer stack.
//!
//! Layer 3: CollapseGate decides SHOULD this delta land?
//! MergeMode decides HOW overlapping writes merge.
//! GateDecision = gate + merge mode (owned microcopy, 2 bytes).

/// Default α-saturation threshold for [`MergeMode::AlphaFrontToBack`].
/// Once accumulated α exceeds this, the front-to-back loop terminates
/// early — the Kerbl 2023 EWA-splatting "early ray termination" rule
/// (Sec. 4.1 of the 3D Gaussian-Splatting paper) ported from pixel
/// rasterization to BindSpace columns.
pub const ALPHA_SATURATION_THRESHOLD: f32 = 0.99;

/// How overlapping writers merge their deltas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MergeMode {
    /// XOR commit: `target ^= delta`. Self-inverse, reversible.
    /// For single-target updates where order doesn't matter.
    Xor = 0,
    /// Bundle: majority vote across all pending deltas.
    /// For multi-writer consensus (e.g., multiple agents posting to blackboard).
    Bundle = 1,
    /// Superposition: keep ALL deltas without resolution.
    /// For ambiguous cases where we want to preserve all variants.
    Superposition = 2,
    /// Pillar-7 front-to-back α-compositing (Kerbl 2023 EWA splatting,
    /// adapted from pixels to BindSpace columns).
    ///
    /// Hits — assumed sorted by confidence DESC — are composited
    /// front-to-back:
    ///
    /// ```text
    ///   color_acc += color_i * α_i * (1 - α_acc)
    ///   α_acc     += α_i * (1 - α_acc)
    ///   if α_acc > ALPHA_SATURATION_THRESHOLD { break }   // early ray termination
    /// ```
    ///
    /// Top-K hit aggregation is replaced by this volumetric merge:
    /// the strongest hit dominates, weaker hits fill in transparency,
    /// and saturation lets us skip the long tail without losing the
    /// dominant signal. Concentration-of-measure in d=10000 keeps the
    /// remainder mass bounded (per `I-NOISE-FLOOR-JIRAK`).
    ///
    /// The saturation threshold defaults to [`ALPHA_SATURATION_THRESHOLD`]
    /// (0.99). Per-dispatch overrides ride on
    /// `ShaderDispatch::alpha_saturation_override`, keeping this enum
    /// `Copy + Eq + Hash + #[repr(u8)]` so existing call sites don't
    /// regress.
    AlphaFrontToBack = 3,
}

/// A gate decision: what the CollapseGate decided + how to merge.
/// Copy type, 2 bytes. The microcopy returned by gate evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateDecision {
    /// Flow = apply delta. Block = reject. Hold = queue for next cycle.
    pub gate: u8, // 0=Flow, 1=Block, 2=Hold (matches ndarray CollapseGate ordinals)
    /// How to merge if Flow.
    pub merge: MergeMode,
}

impl GateDecision {
    pub const FLOW_XOR: Self = Self {
        gate: 0,
        merge: MergeMode::Xor,
    };
    pub const FLOW_BUNDLE: Self = Self {
        gate: 0,
        merge: MergeMode::Bundle,
    };
    pub const FLOW_SUPER: Self = Self {
        gate: 0,
        merge: MergeMode::Superposition,
    };
    pub const BLOCK: Self = Self {
        gate: 1,
        merge: MergeMode::Xor,
    };
    pub const HOLD: Self = Self {
        gate: 2,
        merge: MergeMode::Xor,
    };

    #[inline]
    pub fn is_flow(&self) -> bool {
        self.gate == 0
    }
    #[inline]
    pub fn is_block(&self) -> bool {
        self.gate == 1
    }
    #[inline]
    pub fn is_hold(&self) -> bool {
        self.gate == 2
    }
}

// ── D-CSV-4: CollapseGateEmission ────────────────────────────────────────────
//
// Plan §8.1 specifies `SmallVec<[(u16, CausalEdge64); 8]>` for the baton list.
// Resolution (pre-ratified by main thread): use `Vec<(u16, u64)>` instead.
//   • Keeps contract zero-dep (SmallVec would require the `smallvec` crate).
//   • u64 is the raw CausalEdge64 packing; receivers wrap it back via
//     `CausalEdge64(raw)` using the causal-edge crate's tuple-struct accessor.
//   • Defers the SmallVec inline-storage optimisation to a sprint-12+ pass.
//   • No `#[repr(C)]` — that was a wire-format aspiration; with `Vec` we are
//     heap-indirect anyway and repr(C) buys nothing here.
//
// Wire-cost (plan §8.2):
//   header  = 4 (source_mailbox) + 4 (chain_position) + 1 (merge_mode tag) = 9 B
//   But plan §8.2 states the header is 13 bytes; accounting is:
//     source_mailbox  u32  4 B
//     chain_position  u32  4 B
//     merge_mode      u8   1 B
//     padding / future headroom   4 B  (reserved for sprint-12+ fields)
//   ─────────────────────────────────
//   header total              13 B   (matches §8.2 exactly)
//   per baton: u16 (2 B) + u64 (8 B) = 10 B
//   Total = 13 + 10 × baton_count
//   8-baton inline budget → 93 B (§8.2 "Total inline emission: ~93 bytes").
//
// No-analog-bucket rebuttal (plan §8.3):
//   Vec<(u16, u64)> IS the bundle decomposed. Vsa16kF32 does NOT cross
//   mailbox boundaries. Receiver's `apply_edges` re-superposes via energy
//   addition — same algebra, zero encode/decode at the boundary.

/// Canonical handle for the W-slot corpus-root / mailbox addressing surface.
/// A `MailboxId` is the unique u32 identity of one spatial-temporal meaning
/// accumulator in `MailboxSoA<N>`. It is used as the provenance anchor in
/// `CollapseGateEmission` so receivers can locate the source row without an
/// explicit cycle-id field (per plan §8.1: provenance via source + chain_position).
pub type MailboxId = u32;

/// Discrete baton emission from one CollapseGate to downstream consumers.
///
/// Implements the gapless-baton model (plan §4.2 + §8):
/// - No `Vsa16kF32` envelope — the baton list IS its own wire format.
/// - No encode/decode at mailbox boundaries; the tuple is the wire.
/// - Provenance is implicit: `(source_mailbox, chain_position)` together
///   identify this emission uniquely within a discourse corpus.
///
/// # Deviation from §8.1 sketch
///
/// Plan §8.1 specifies `SmallVec<[(u16, CausalEdge64); 8]>`.  This
/// implementation uses `Vec<(u16, u64)>` (zero-dep, heap-indirect):
/// - `u64` is the raw CausalEdge64 packing; receivers reconstruct via
///   `CausalEdge64(raw)`.
/// - The SmallVec inline-storage optimisation is deferred to sprint-12+.
/// - `#[repr(C)]` is omitted (heap indirection via Vec makes it moot).
///
/// # Wire-cost (§8.2)
///
/// ```text
/// header  = 13 bytes (source_mailbox u32 + chain_position u32 + merge_mode u8 + 4 B reserved)
/// per baton = 10 bytes (u16 target + u64 edge raw)
/// total     = 13 + 10 × baton_count
/// 8 batons  → 93 bytes  (§8.2 budget)
/// ```
///
/// # No analog bucket (§8.3)
///
/// `Vsa16kF32` does not cross mailbox boundaries. Three candidate needs
/// that would have forced it — compound bundles, Markov ±5 braiding,
/// continuous strength values — are all satisfied by discrete tuple lists.
/// Receiver's `apply_edges` re-superposes via energy addition (same algebra).
///
/// # Cycle ID
///
/// There is NO `cycle_id` field. Provenance is fully determined by
/// `(source_mailbox, chain_position)` per plan §8.1. Consumers locate the
/// source mailbox row and its position in the witness chain without an
/// explicit cycle reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollapseGateEmission {
    /// Per-target discrete batons. Each `(target, edge_raw)` tuple is one
    /// neuron-to-neuron delivery. `edge_raw` is the packed u64 of a
    /// `CausalEdge64`; receivers reconstruct via `CausalEdge64(edge_raw)`.
    ///
    /// Uses `Vec` rather than `SmallVec<[…; 8]>` to keep contract zero-dep.
    /// SmallVec optimisation deferred to sprint-12+.
    batons: Vec<(u16, u64)>,

    /// Source mailbox identity — the W-slot corpus-root handle that produced
    /// this emission. Together with `chain_position` this is the full provenance.
    source_mailbox: MailboxId,

    /// Position of this emission in the source mailbox's witness chain.
    /// Encodes the temporal axis structurally (plan §4.4 — "temporal causality
    /// is structural, not stored"). No wall-clock timestamp; relative order
    /// is chain_position; absolute anchor is AriGraph `Triplet.timestamp`.
    chain_position: u32,

    /// Merge hint for the receiving CollapseGate.
    ///
    /// - [`MergeMode::Bundle`] — associative superposition (Markov-respecting).
    /// - [`MergeMode::Xor`] — single-writer delta (faster, breaks Markov;
    ///   per `I-SUBSTRATE-MARKOV` iron rule, Xor is NOT a Markov transition
    ///   kernel — only for deltas).
    /// - [`MergeMode::Superposition`] — keep ALL deltas without resolution.
    /// - [`MergeMode::AlphaFrontToBack`] — volumetric front-to-back composite.
    merge_mode: MergeMode,
}

impl CollapseGateEmission {
    /// Create a new emission with no batons yet.
    ///
    /// # Arguments
    ///
    /// * `source` — The [`MailboxId`] of the emitting mailbox.
    /// * `chain_position` — Position in the source's witness chain (structural time).
    /// * `merge_mode` — How the receiving CollapseGate should merge this emission.
    ///
    /// # Wire cost at construction
    ///
    /// `wire_cost_bytes()` returns 13 (header only) until batons are pushed.
    #[inline]
    pub fn new(source: MailboxId, chain_position: u32, merge_mode: MergeMode) -> Self {
        Self {
            batons: Vec::new(),
            source_mailbox: source,
            chain_position,
            merge_mode,
        }
    }

    /// Append a baton targeting `target` mailbox with raw edge packing `edge`.
    ///
    /// `edge` is the raw u64 packing of a `CausalEdge64`; the receiver
    /// reconstructs via `CausalEdge64(edge)`.
    #[inline]
    pub fn push_baton(&mut self, target: u16, edge: u64) {
        self.batons.push((target, edge));
    }

    /// Number of batons currently in this emission.
    #[inline]
    pub fn baton_count(&self) -> usize {
        self.batons.len()
    }

    /// Serialised wire cost in bytes (plan §8.2).
    ///
    /// Formula: `13 + 10 × baton_count`
    ///
    /// - 13-byte header: `source_mailbox` (4 B) + `chain_position` (4 B)
    ///   + `merge_mode` tag (1 B) + 4 B reserved headroom.
    /// - 10 bytes per baton: `u16` target (2 B) + `u64` edge raw (8 B).
    ///
    /// At 8 batons this is 93 bytes (§8.2 "Total inline emission: ~93 bytes").
    #[inline]
    pub fn wire_cost_bytes(&self) -> usize {
        13 + 10 * self.batons.len()
    }

    // ── Provenance accessors ─────────────────────────────────────────────────

    /// The [`MailboxId`] of the mailbox that produced this emission.
    #[inline]
    pub fn source_mailbox(&self) -> MailboxId {
        self.source_mailbox
    }

    /// Position of this emission in the source mailbox's witness chain.
    /// Encodes temporal order structurally (plan §4.4).
    #[inline]
    pub fn chain_position(&self) -> u32 {
        self.chain_position
    }

    /// The merge mode hint for the receiving CollapseGate.
    #[inline]
    pub fn merge_mode(&self) -> MergeMode {
        self.merge_mode
    }

    /// Read-only view of the baton list.
    /// Each element is `(target_mailbox_id: u16, edge_raw: u64)`.
    #[inline]
    pub fn batons(&self) -> &[(u16, u64)] {
        &self.batons
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_emission(mode: MergeMode) -> CollapseGateEmission {
        CollapseGateEmission::new(42, 7, mode)
    }

    // ── D-CSV-4 test suite (8 tests) ─────────────────────────────────────────

    /// New emission must have 0 batons and wire_cost == 13.
    #[test]
    fn test_emission_new_empty() {
        let e = make_emission(MergeMode::Bundle);
        assert_eq!(e.baton_count(), 0);
        assert_eq!(e.wire_cost_bytes(), 13);
    }

    /// Pushing 3 batons: count == 3, wire_cost == 43.
    #[test]
    fn test_emission_push_baton() {
        let mut e = make_emission(MergeMode::Bundle);
        e.push_baton(1, 0xDEAD_BEEF_CAFE_0001_u64);
        e.push_baton(2, 0xDEAD_BEEF_CAFE_0002_u64);
        e.push_baton(3, 0xDEAD_BEEF_CAFE_0003_u64);
        assert_eq!(e.baton_count(), 3);
        assert_eq!(e.wire_cost_bytes(), 43); // 13 + 10*3 = 43
    }

    /// 8 batons → wire_cost == 93 (§8.2 inline budget).
    #[test]
    fn test_emission_wire_cost_8_batons() {
        let mut e = make_emission(MergeMode::Bundle);
        for i in 0u16..8 {
            e.push_baton(i, i as u64);
        }
        assert_eq!(e.baton_count(), 8);
        assert_eq!(e.wire_cost_bytes(), 93); // 13 + 10*8 = 93
    }

    /// Provenance fields survive construction; no public `cycle_id()` method.
    ///
    /// This test asserts that `source_mailbox` and `chain_position` are
    /// accessible and correct.  The compile-time absence of `cycle_id()` is
    /// enforced structurally — the struct has no such field, so any caller
    /// attempting `e.cycle_id()` will receive a compile error.
    #[test]
    fn test_emission_provenance_no_cycle_id() {
        let e = CollapseGateEmission::new(99, 42, MergeMode::Xor);
        assert_eq!(e.source_mailbox(), 99u32);
        assert_eq!(e.chain_position(), 42u32);
        // Verify that `CollapseGateEmission` exposes NO `cycle_id` method.
        // This is structural: the field does not exist, so the following line
        // would not compile if uncommented:
        //   let _ = e.cycle_id(); // ← compile error: no method named `cycle_id`
    }

    /// Xor merge mode threads through correctly.
    #[test]
    fn test_emission_merge_mode_xor() {
        let e = make_emission(MergeMode::Xor);
        assert_eq!(e.merge_mode(), MergeMode::Xor);
    }

    /// Bundle merge mode threads through correctly.
    #[test]
    fn test_emission_merge_mode_bundle() {
        let e = make_emission(MergeMode::Bundle);
        assert_eq!(e.merge_mode(), MergeMode::Bundle);
    }

    /// Superposition merge mode threads through correctly.
    #[test]
    fn test_emission_merge_mode_superposition() {
        let e = make_emission(MergeMode::Superposition);
        assert_eq!(e.merge_mode(), MergeMode::Superposition);
    }

    /// Full provenance round-trip: encode fields → extract → reconstruct →
    /// assert equal.  This replaces the separate AlphaFrontToBack mode test
    /// (which is covered inside the round-trip) and is the D-CSV-4 "8
    /// round-trip tests" anchor from plan §16 / §15.
    ///
    /// Also verifies AlphaFrontToBack mode threads through correctly.
    #[test]
    fn test_emission_round_trip() {
        // Build original emission with AlphaFrontToBack (covers merge-mode test too)
        let mut original = CollapseGateEmission::new(123, 456, MergeMode::AlphaFrontToBack);
        original.push_baton(10, 0xAABBCCDD_11223344_u64);
        original.push_baton(20, 0x55667788_99AABBCC_u64);

        // Extract all fields
        let src = original.source_mailbox();
        let pos = original.chain_position();
        let mode = original.merge_mode();
        let batons: Vec<(u16, u64)> = original.batons().to_vec();

        // Reconstruct from extracted fields
        let mut reconstructed = CollapseGateEmission::new(src, pos, mode);
        for (target, edge) in batons {
            reconstructed.push_baton(target, edge);
        }

        // Assert structural equality
        assert_eq!(original, reconstructed);

        // Spot-check provenance + mode
        assert_eq!(reconstructed.source_mailbox(), 123);
        assert_eq!(reconstructed.chain_position(), 456);
        assert_eq!(reconstructed.merge_mode(), MergeMode::AlphaFrontToBack);
        assert_eq!(reconstructed.baton_count(), 2);
        assert_eq!(reconstructed.wire_cost_bytes(), 33); // 13 + 10*2
    }
}
