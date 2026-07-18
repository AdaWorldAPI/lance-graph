//! # `soa_view` — the transparent, zero-copy read view over the ONE SoA.
//!
//! **R1 "one SoA never transformed":** the per-mailbox SoA is never serialized or
//! copied; it lives from mailbox spawn to tombstone and is mutated only by
//! cognitive operations. This module is the **zero-dep borrow vocabulary** that
//! lets three holders read the SAME bytes:
//!
//! - `cognitive-shader-driver`'s `MailboxSoA<N>` — the in-RAM hot owner (implements
//!   [`MailboxSoaOwner`]; ractor drives it),
//! - `surreal_container` — the transparent kv-lance-backed VIEW (implements the
//!   read-only [`MailboxSoaView`] over the same Lance columns; no Arrow re-encode),
//! - `lance-graph-planner` — a CONSUMER (plans over the columns directly).
//!
//! The contract owns **no** SoA storage — only this lens. It cannot name
//! `MailboxSoA<N>` (another crate) without a dependency, so the lens is a trait the
//! owner/view implement — the same dependency-inversion pattern as
//! [`crate::plan::PlannerContract`] and [`crate::orchestration::OrchestrationBridge`].

use crate::collapse_gate::MailboxId;
use crate::kanban::{KanbanColumn, KanbanMove, RubiconTransitionError};

/// Which dense identity plane a value-side read selects — the orthogonal
/// perspective axes of a node's content (`E-TENANT-ANGLE-RANK-IS-CAM-PQ-ADC`).
/// Each is a `WORDS_PER_FP`-u64 fingerprint in the value slab; reading one is a
/// **value decode** (the costed tier), never the zero-decode key path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityPlane {
    /// The content identity fingerprint plane.
    Content,
    /// The topic identity fingerprint plane.
    Topic,
    /// The angle (perspective) identity fingerprint plane.
    Angle,
}

/// Which lane of the **autopoiesis triangle** a style read selects — the three
/// per-row policy lanes appended after Kanban in the canonical `NodeRow` value slab
/// (`ValueTenant::{FrozenStyle, LearnedStyle, ExploreStyle}`, offsets 152/164/176).
/// Each lane is 12 palette256 atoms (one per [`StyleFamily`](crate::style_family::StyleFamily)
/// ordinal 0..11, OR per compiled-template step 0..11 — ClassView-selected). Reading
/// one is a value-slab read (the costed tier), mirroring [`IdentityPlane`].
///
/// The triangle is the ancestry-pipeline wiring surface: **dispatch** reads
/// [`Frozen`](StyleLane::Frozen) (the checkpoint policy the can't-stop-thinking
/// cycle runs off), the P64 perturbation ladder writes [`Explore`](StyleLane::Explore),
/// and the L4 learning seam writes [`Learned`](StyleLane::Learned) via NARS revision —
/// each write an **owned SoA update** on the mailbox's own lane (`MailboxSoaOwner`,
/// ractor sole-mutator), never a by-convention `&mut`. Triangle plan
/// `.claude/plans/triangle-tenants-gestalt-separation-v1.md` §1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StyleLane {
    /// The CHECKPOINT policy lane (`ValueTenant::FrozenStyle`) — what the row does now.
    Frozen,
    /// The NARS-revision-updated policy lane (`ValueTenant::LearnedStyle`) — the L4 write target.
    Learned,
    /// The deterministic address-derived exploration variant (`ValueTenant::ExploreStyle`).
    Explore,
}

/// A transparent, read-only view over one mailbox's SoA columns.
///
/// Implementors return **borrows** (`&[T]`) or `Copy` scalars — never clones of the
/// backing store. A `surreal_container` view and the in-RAM `MailboxSoA` are both
/// valid implementors over the *same* bytes; that two-implementor symmetry is what
/// "transparent view" means here (R1).
pub trait MailboxSoaView {
    /// Identity of the mailbox this view reads.
    fn mailbox_id(&self) -> MailboxId;
    /// Number of populated rows in the SoA.
    fn n_rows(&self) -> usize;
    /// 6-bit witness-table slot (0..=63) the mailbox occupies.
    fn w_slot(&self) -> u8;
    /// Monotonic cognitive cycle stamp.
    fn current_cycle(&self) -> u32;
    /// The Rubicon phase the mailbox is currently in (kanban column).
    fn phase(&self) -> KanbanColumn;

    // ── zero-copy column borrows (the SIMD / surreal-projection surface) ──

    /// Per-row spatial-temporal energy accumulator.
    fn energy(&self) -> &[f32];
    /// Per-row packed `CausalEdge64` as raw `u64` (reconstruct via `CausalEdge64(raw)`;
    /// kept raw so the contract stays zero-dep — `causal-edge` is not a contract dep).
    fn edges_raw(&self) -> &[u64];
    /// Per-row packed `MetaWord` as raw `u32`.
    fn meta_raw(&self) -> &[u32];
    /// Per-row entity-type id.
    fn entity_type(&self) -> &[u16];

    /// Per-row **class discriminator** — the Cognitive-RISC `class_id` / `shape_id`
    /// (a.k.a. the OGIT `EntityTypeId`). Aliases
    /// [`entity_type`](MailboxSoaView::entity_type) today: the existing `u16` slot IS
    /// the class hook, so no new column is added (honors R1 "one SoA never
    /// transformed"). Only the `u16` discriminator lives on the SoA; the machinery it
    /// keys — label inheritance, column projection, jinja templates — resolves ONE
    /// LAYER UP via the OGIT ontology cache (`lance-graph-ontology`), never in the SoA
    /// / kv-lance columns. This is the Cognitive-RISC N1 freeze-time hook.
    #[inline]
    fn class_id(&self) -> &[u16] {
        self.entity_type()
    }

    /// The `class_id` of a single row.
    #[inline]
    fn class_id_at(&self, row: usize) -> u16 {
        self.entity_type()[row]
    }

    /// Resolve a canonical [`NodeGuid::local_key`](crate::canonical_node::NodeGuid::local_key)
    /// (bytes 10..16 = family++identity, the basin-local discriminator) to a row
    /// index in this view — the **key→row baton** a `Backend::MailboxSoa` graph
    /// router needs to land a Cypher `MATCH`/edge-slot deref on the GUID-keyed
    /// substrate (`cypher-kanban-ast-unification-v1` Inc 0; the baton-handoff-auditor's
    /// CATCH-CRITICAL — the View previously exposed only `n_rows`, with no way to go
    /// from the canon address back to a row).
    ///
    /// **Default = `None` (zero-fallback, deferred binding).** A view that has NOT
    /// materialized a per-row key index returns `None` for every key — the same
    /// deferred-accessor discipline as `qualia` / `episodic_witness` below: the
    /// resolver contract is declared here so the router signature can name it, and an
    /// owner that stores keys (the in-RAM `MailboxSoA`, once it carries a `local_key`
    /// column) overrides this. Until then a consumer that gets `None` falls back to
    /// the positional `(mailbox_id, row)` address, never a wrong row.
    #[inline]
    fn row_for_local_key(&self, _local_key: u64) -> Option<usize> {
        None
    }

    /// The HHTL routing path ([`NiblePath`](crate::hhtl::NiblePath)) of `row`'s
    /// GUID key — the `classid·HEEL·HIP·TWIG` cascade lowered to a nibble path.
    /// This is the **radix-trie / CLAM cluster address** of the node
    /// (`panCAKES ≡ radix trie ≡ HHTL`): containment = `is_ancestor_of`,
    /// CAKES nearest = `common_prefix_depth`, both pure key arithmetic, **zero
    /// value decode**.
    ///
    /// **Default = `None` (zero-fallback, deferred binding)** — same discipline as
    /// [`row_for_local_key`](MailboxSoaView::row_for_local_key): a view that has
    /// not materialized a per-row key/HHTL column returns `None`, and a CLAM/CAKES
    /// scan over it yields nothing (the consumer falls back to a coarser facet).
    /// An owner that carries the GUID key per row overrides this (the canon
    /// `NodeRow` already holds `key(16)`, so the override exposes what is there).
    #[inline]
    fn hhtl_path_at(&self, _row: usize) -> Option<crate::hhtl::NiblePath> {
        None
    }

    /// The 16-byte [`EdgeBlock`](crate::canonical_node::EdgeBlock) of `row` — the
    /// node's **explicit typed edges** (12 in-family + 4 out-of-family one-byte
    /// slots), bytes 16..32 of the canonical `NodeRow`. This is the edge region,
    /// **NOT the value slab** (32..512), so reading it is **zero value decode**.
    ///
    /// How the 16 bytes are *interpreted* is the class's
    /// [`EdgeCodecFlavor`](crate::canonical_node::EdgeCodecFlavor)
    /// (`CoarseOnly` = 12-family/4-external adjacency, `Pq32x4` = 32×4 turbovec
    /// residue) — resolved `classid → ClassView`, never guessed by the query
    /// (`E-ADJACENCY-IS-KEY-AND-EDGECODEC`).
    ///
    /// **Default = `None` (zero-fallback, deferred binding)** — a view that has not
    /// materialized the edge region returns `None`; an owner that carries the
    /// canonical `NodeRow` (which holds `edges(16)`) overrides this.
    #[inline]
    fn edge_block_at(&self, _row: usize) -> Option<crate::canonical_node::EdgeBlock> {
        None
    }

    /// `row`'s dense identity-plane fingerprint (`WORDS_PER_FP` u64) for the
    /// selected [`IdentityPlane`] — content / topic / angle. This is the
    /// **value-side** read behind the costed distance/sweep tier
    /// (`E-TENANT-ANGLE-RANK-IS-CAM-PQ-ADC`): a Hamming/CAM rank "from an angle"
    /// reads this plane, so unlike the key facets it is **NOT zero value decode**.
    ///
    /// **Default = `None` (zero-fallback, deferred binding)** — a view that has not
    /// materialized the planes returns `None`; the in-RAM `MailboxSoA` owner
    /// (which carries content/topic/angle planes, W1b) overrides this.
    #[inline]
    fn identity_plane_at(&self, _row: usize, _plane: IdentityPlane) -> Option<&[u64]> {
        None
    }

    /// `row`'s 12-atom **autopoiesis-triangle** lane for the selected
    /// [`StyleLane`] (Frozen / Learned / Explore) — the value-slab policy bytes at
    /// `ValueTenant::{FrozenStyle, LearnedStyle, ExploreStyle}` (offsets
    /// 152/164/176 in the canonical `NodeRow`). Index `f` of the returned `[u8; 12]`
    /// is the palette256 atom for [`StyleFamily`](crate::style_family::StyleFamily)
    /// ordinal `f` (atom `0` = null default — an un-populated lane reads all-null,
    /// never a wrong policy).
    ///
    /// **Default = `None` (zero-fallback, deferred binding)** — same discipline as
    /// [`identity_plane_at`](MailboxSoaView::identity_plane_at): a view that has not
    /// materialized the triangle lanes returns `None`, and a consumer falls back to
    /// the default style. The in-RAM `MailboxSoA` owner (which carries the triangle
    /// columns, P4) overrides this; the canonical `NodeRow` already exposes the bytes
    /// via [`NodeRow::style_lane`](crate::canonical_node::NodeRow::style_lane).
    #[inline]
    fn style_lane_at(&self, _row: usize, _lane: StyleLane) -> Option<[u8; 12]> {
        None
    }

    /// `row`'s per-family triangle read `(frozen, learned, explore)` — the three
    /// palette256 atoms at [`StyleFamily`](crate::style_family::StyleFamily) ordinal
    /// `family`, one glance across the three lanes (mirrors
    /// [`NodeRow::triangle_for`](crate::canonical_node::NodeRow::triangle_for)).
    ///
    /// Default composes [`style_lane_at`](MailboxSoaView::style_lane_at) over the
    /// three lanes: `None` if the view has not materialized them, or if `family >= 12`
    /// (out of range → no triangle, never an aliased slot — the #717 `triangle_for`
    /// guard). An owner with a cheaper single-row read may override.
    #[inline]
    fn triangle_at(&self, row: usize, family: u8) -> Option<(u8, u8, u8)> {
        if family >= 12 {
            return None;
        }
        let f = self.style_lane_at(row, StyleLane::Frozen)?;
        let l = self.style_lane_at(row, StyleLane::Learned)?;
        let e = self.style_lane_at(row, StyleLane::Explore)?;
        let i = family as usize;
        Some((f[i], l[i], e[i]))
    }

    // NOTE (follow-up): the qualia column (`QualiaI4_16D`) accessor is intentionally omitted —
    // add `fn qualia(&self) -> &[crate::qualia::QualiaI4_16D]` when the first consumer
    // (planner strategy selection) needs it; keep the read surface minimal until then.

    // NOTE (follow-up, P2 of the three-Markovs / EW64 reactive-seam ordering):
    // the EpisodicWitness64 column accessor is intentionally omitted for now —
    // add `fn episodic_witness(&self) -> &[EpisodicWitness64]` (same deferred-
    // accessor pattern as `qualia` above) when the first consumer needs it.
    //
    // WHAT EpisodicWitness64 IS: it is **AriGraph living in the mailbox SoA view**.
    // AriGraph is a Markov chain in the cold path (`lance-graph::graph::arigraph`:
    // `episodic` / `witness_corpus` / `triplet_graph`); this column is that same
    // episodic graph **promoted to the hot path** as a per-row SoA column — the
    // `CausalEdge64` W-slot → witness arc (the deterministic "Markov #1" chain;
    // see `witness_table.rs`: "the chain of W-references across edges forms a
    // Markov-style belief-update arc through episodic-reference vectors"). EW64 is
    // the *particle* (discrete, addressable, exact witness pointer); the windowed
    // projection `arigraph::markov_soa` is the *wave*. Both ARE AriGraph.
    //
    // STATUS: `EpisodicWitness64` is NOT YET a code symbol (a queued design — see
    // EPIPHANIES `E-EW64-IS-PREDICTIVE-PREFETCH`; the shipped seeds are the 6-bit
    // W-slot `causal-edge::CausalEdge64` + `WitnessTable<64>`/`WitnessEntry` +
    // `arigraph::{episodic,witness_corpus}`). Like every column the contract holds
    // it stays AGNOSTIC: the witness arc carries SPO from ANY source — the
    // *language* layer (DeepNSM/COCA) stays strictly upstream and never reaches in.

    // ── per-row scalar read (mirrors `MailboxSoA::energy_at`) ──

    /// Energy at `row`. Default indexes [`energy`](MailboxSoaView::energy); override
    /// if the implementor can read a single row more cheaply.
    #[inline]
    fn energy_at(&self, row: usize) -> f32 {
        self.energy()[row]
    }
}

/// The mutation airgap for the SoA **owner** only (the ractor-driven hot path).
///
/// A read-only view (e.g. `surreal_container`) deliberately does **not** implement
/// this — that is what makes "the view is read-only" a structural guarantee rather
/// than a convention. Only the in-RAM `MailboxSoA` owner advances phases.
pub trait MailboxSoaOwner: MailboxSoaView {
    /// Drive one Rubicon phase transition to `to`; return the emitted move.
    ///
    /// The only mutation surface at the contract level: cognitive operations advance
    /// the lifecycle column. The SoA columns themselves are mutated by the owner's
    /// own (crate-private) cognitive ops, never serialized through here (R1).
    fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove;

    /// Checked phase advance: enforce the Rubicon lifecycle DAG
    /// ([`KanbanColumn::can_transition_to`]) before mutating.
    ///
    /// Returns the emitted [`KanbanMove`] on a legal edge, or
    /// [`RubiconTransitionError`] on an illegal one (no mutation occurs). The ractor
    /// lifecycle driver should prefer this over the unchecked
    /// [`advance_phase`](MailboxSoaOwner::advance_phase) so an illegal transition is a
    /// typed error, not silent corruption.
    fn try_advance_phase(
        &mut self,
        to: KanbanColumn,
    ) -> Result<KanbanMove, RubiconTransitionError> {
        let from = self.phase();
        if from.can_transition_to(to) {
            Ok(self.advance_phase(to))
        } else {
            Err(RubiconTransitionError { from, to })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal in-memory implementor proving the trait is satisfiable and the
    /// `&[T]` borrows compile + read zero-copy — without any consumer crate.
    struct FakeSoa {
        id: MailboxId,
        phase: KanbanColumn,
        energy: Vec<f32>,
        edges: Vec<u64>,
        meta: Vec<u32>,
        etype: Vec<u16>,
        cycle: u32,
    }

    impl MailboxSoaView for FakeSoa {
        fn mailbox_id(&self) -> MailboxId {
            self.id
        }
        fn n_rows(&self) -> usize {
            self.energy.len()
        }
        fn w_slot(&self) -> u8 {
            (self.id & 0x3F) as u8
        }
        fn current_cycle(&self) -> u32 {
            self.cycle
        }
        fn phase(&self) -> KanbanColumn {
            self.phase
        }
        fn energy(&self) -> &[f32] {
            &self.energy
        }
        fn edges_raw(&self) -> &[u64] {
            &self.edges
        }
        fn meta_raw(&self) -> &[u32] {
            &self.meta
        }
        fn entity_type(&self) -> &[u16] {
            &self.etype
        }
    }

    impl MailboxSoaOwner for FakeSoa {
        fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove {
            let from = self.phase;
            self.phase = to;
            KanbanMove {
                mailbox: self.id,
                from,
                to,
                witness_chain_position: self.cycle,
                libet_offset_us: if to == KanbanColumn::CognitiveWork {
                    -550_000
                } else {
                    0
                },
                exec: crate::kanban::ExecTarget::Native,
            }
        }
    }

    fn sample() -> FakeSoa {
        FakeSoa {
            id: 7,
            phase: KanbanColumn::Planning,
            energy: vec![0.1, 0.2, 0.3],
            edges: vec![0, 1, 2],
            meta: vec![10, 11, 12],
            etype: vec![100, 101, 102],
            cycle: 1,
        }
    }

    #[test]
    fn view_reads_columns_zero_copy() {
        let soa = sample();
        // Borrow points INTO the backing store (zero-copy): identical pointer.
        assert_eq!(soa.energy().as_ptr(), soa.energy.as_ptr());
        assert_eq!(soa.n_rows(), 3);
        assert_eq!(soa.edges_raw(), &[0, 1, 2]);
        assert_eq!(soa.meta_raw(), &[10, 11, 12]);
        assert_eq!(soa.entity_type(), &[100, 101, 102]);
        // class_id is the Cognitive-RISC N1 hook aliasing the entity_type slot.
        assert_eq!(soa.class_id(), &[100, 101, 102]);
        assert_eq!(soa.class_id_at(0), 100);
        assert_eq!(soa.energy_at(1), 0.2);
        assert_eq!(soa.phase(), KanbanColumn::Planning);
        assert_eq!(soa.w_slot(), 7);
    }

    #[test]
    fn row_for_local_key_defaults_to_none_until_a_key_index_is_materialized() {
        // Deferred-binding default: a view with no per-row key column resolves
        // every local_key to None — the consumer falls back to the positional
        // (mailbox_id, row) address, never a wrong row. The Backend::MailboxSoa
        // router can name this baton; an owner that stores keys overrides it.
        let soa = sample();
        assert_eq!(soa.row_for_local_key(0), None);
        assert_eq!(soa.row_for_local_key(u64::MAX), None);
    }

    #[test]
    fn owner_advances_phase_and_sets_libet_anchor() {
        let mut soa = sample();
        let m = soa.advance_phase(KanbanColumn::CognitiveWork);
        assert_eq!(m.from, KanbanColumn::Planning);
        assert_eq!(m.to, KanbanColumn::CognitiveWork);
        assert_eq!(m.libet_offset_us, -550_000);
        assert_eq!(soa.phase(), KanbanColumn::CognitiveWork);
    }

    #[test]
    fn try_advance_phase_enforces_lifecycle() {
        let mut soa = sample(); // Planning
                                // Illegal skip Planning -> Evaluation is rejected, no mutation.
        let err = soa.try_advance_phase(KanbanColumn::Evaluation).unwrap_err();
        assert_eq!(err.from, KanbanColumn::Planning);
        assert_eq!(err.to, KanbanColumn::Evaluation);
        assert_eq!(
            soa.phase(),
            KanbanColumn::Planning,
            "rejected transition must not mutate"
        );
        // Legal Planning -> CognitiveWork succeeds.
        let m = soa.try_advance_phase(KanbanColumn::CognitiveWork).unwrap();
        assert_eq!(m.to, KanbanColumn::CognitiveWork);
        assert_eq!(soa.phase(), KanbanColumn::CognitiveWork);
    }

    #[test]
    fn style_lane_defaults_to_none_until_the_owner_materializes_the_triangle() {
        // Deferred binding: a view with no triangle columns resolves every style
        // lane (and the composed per-family read) to None — the consumer falls back
        // to the default style, never a wrong policy. The in-RAM MailboxSoA owner
        // (P4) overrides this.
        let soa = sample();
        assert_eq!(soa.style_lane_at(0, StyleLane::Frozen), None);
        assert_eq!(soa.style_lane_at(0, StyleLane::Learned), None);
        assert_eq!(soa.style_lane_at(0, StyleLane::Explore), None);
        assert_eq!(soa.triangle_at(0, 0), None);
    }

    /// An owner that HAS materialized the triangle lanes — proves the override +
    /// `triangle_at` composition + the `family >= 12` guard (the #717 aliasing guard).
    struct StyledSoa {
        frozen: [u8; 12],
        learned: [u8; 12],
        explore: [u8; 12],
    }

    impl MailboxSoaView for StyledSoa {
        fn mailbox_id(&self) -> MailboxId {
            1
        }
        fn n_rows(&self) -> usize {
            1
        }
        fn w_slot(&self) -> u8 {
            1
        }
        fn current_cycle(&self) -> u32 {
            0
        }
        fn phase(&self) -> KanbanColumn {
            KanbanColumn::CognitiveWork
        }
        fn energy(&self) -> &[f32] {
            &[]
        }
        fn edges_raw(&self) -> &[u64] {
            &[]
        }
        fn meta_raw(&self) -> &[u32] {
            &[]
        }
        fn entity_type(&self) -> &[u16] {
            &[]
        }
        fn style_lane_at(&self, _row: usize, lane: StyleLane) -> Option<[u8; 12]> {
            Some(match lane {
                StyleLane::Frozen => self.frozen,
                StyleLane::Learned => self.learned,
                StyleLane::Explore => self.explore,
            })
        }
    }

    #[test]
    fn triangle_at_composes_the_three_lanes_and_guards_family_range() {
        let soa = StyledSoa {
            frozen: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            learned: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
            explore: [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
        };
        // Per-family read picks index `family` from each lane.
        assert_eq!(soa.triangle_at(0, 0), Some((10, 30, 50)));
        assert_eq!(soa.triangle_at(0, 11), Some((21, 41, 61)));
        // family == 12 is out of range → None, never the aliased slot 12 (#717 guard).
        assert_eq!(soa.triangle_at(0, 12), None);
        assert_eq!(soa.triangle_at(0, u8::MAX), None);
        // The whole lane reads through.
        assert_eq!(soa.style_lane_at(0, StyleLane::Learned).unwrap()[5], 35);
    }
}
