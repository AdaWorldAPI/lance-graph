## 2026-08-19 — E-TWO-WITNESS-SHAPES-CONTEST-ONE-LANDING-ZONE-1

**Status:** FINDING (CE64/EW64/dismech investigation + V3 migration-plan
audit, operator-directed). Plan: `.claude/plans/ew64-witness-unification-v1.md`.

**The finding:** the EpisodicWitness64→V3 migration has no plan because its
landing zone is already occupied by a different shape with experimental
status. Tenant 14 `CausalWitness` (`.claude/v3/soa_layout/tenants.md`, 16 B
facet at row [204,220), G24N4 = 24 signed-i4 ±8-window context pointers,
"EXPERIMENTAL — not in the operator-locked §3 catalogue") and the queued
`EpisodicWitness64` (4-slot recency MRU of `EdgeRef{family,local}`,
`episodic_edges.rs` Phase A shipped #446–#448) are two witness shapes that
never reference each other (grep-verified both directions), while tenant 2
`MaterializedEdges` (4 out-of-family CausalEdge64) is a third adjacent edge
carrier. Four W1 scaffolds each self-document the missing seam
(`witness_table.rs` "scaffold-only", `soa_view.rs:257-277` deferred
accessor, `markov_soa.rs` "truly-correct home is still inside the
EW64-in-SoA seam"). Candidate resolution (NOT banked — operator decision):
the two are two RUNGS of one witness ladder — positional stream context vs
episodic reference — and EW64's sub-byte packing lands as a `U64 × 1` LANE
(Qualia/Kanban precedent), never a §3 byte-axis carving. Companion
correction: the EW64-prefetch spec's "three open decisions" are TWO —
`RawEdge(i8)` mantissa-only is SHIPPED (`counterfactual.rs:456/472-479`);
only the `impl EpisodicEdge for CausalEdge64` bridge location remains open.

Refs: `.claude/specs/episodic-witness64-ce64-prefetch.md`, tenants.md §
tenant 14, MODULE-TABLE rows 48/167/227/243, E-EW64-IS-PREDICTIVE-PREFETCH.

