# agent-W5 scratchpad — sprint-log-10

## W5 run: 2026-05-14

**Worker:** W5 (arigraph-spo-g)
**Output:** `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` (23 KB)

### Mandatory reads completed

1. `.claude/board/sprint-log-10/MANIFEST.md` — fleet table read; W5 row confirmed
2. `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §6 + §7 — PR-CE64-MB-4 entry read
3. `.claude/plans/ogit-g-context-bundle-v1.md` — D-OGIT-G-1 SPO-G u32 slot spec read in full
4. `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §1-§9 — SPOW tetrahedron + Gaussian splat read
5. `.claude/board/LATEST_STATE.md` — AriGraph inventory confirmed (4696 LOC, 7 modules)
6. `.claude/board/AGENT_LOG.md` — not found (new board log file, no prior sprint-10 entries)
7. `crates/lance-graph/src/graph/arigraph/` — mod structure confirmed (mod.rs, triplet_graph.rs, episodic.rs, retrieval.rs, sensorium.rs, orchestrator.rs, xai_client.rs, language.rs, spo_bridge.rs)
8. `grep -rn "named_graph|ogit_context_id|SPO_G|witness_ref"` — zero hits (no partial SPO-G implementation)

### Key findings from source inspection

- Current `Triplet` struct: `{subject, object, relation, truth: TruthValue, timestamp: u64}` — no G field
- `spo_bridge::promote_to_spo` takes `(triplet, gate, spo)` — no G parameter
- `lance_cache.rs` SCHEMA_VERSION currently 2 — confirmed at line 62
- `lance_cache_invalidate_*` + `schema_version_pinned` test pattern confirmed for migration guidance
- `contract::hash::fnv1a` canonical per PR #307 — recommended for `witness_ref` derivation

### Spec scope delivered

1. SPO-G quad: Triplet extended with `g: u32`, `pearl_rung: u8`, `witness_ref: u64`
2. Ghost-edge persistence: `ghost.rs` NEW with `GhostReason`, `GhostStore`, `GhostReactivationEvent`, `nars_revise_ghosts`
3. Witness shapes: `witness.rs` NEW with `SpoWitness64` (Copy, 8B) + `SpoWitnessChain<N=32>` + `WitnessChainStore`
4. Schema migration: SCHEMA_VERSION 2→3, backward-compat via default g=0
5. Test plan: 7 tests covering round-trip, G-filter, witness packing, ghost persistence, reactivation, NARS decay, chain truncation
6. Risk matrix: HIGH (Lance schema bump, promote_to_spo API break), MED (chain sizing, decay rate), LOW (rung encoding)

### Open questions surfaced

- OQ-W5-1: Lance persistence for ghost edges (recommend defer to PR-CE64-MB-4b)
- OQ-W5-2: promote_to_spo API evolution (recommend new function, not builder)
- OQ-W5-3: witness_ref derivation via contract::hash::fnv1a (confirm dependency already exists)

### Delta vs parent plan

Parent plan §6 had: "SPO-G quad mode + ghost-edge persistence + SpoWitnessChain" as 3-line summary.
Spec adds: exact `Triplet` field layout (confirmed from source), `SpoWitness64` bit packing (mirrors CausalEdge64 layout), `GhostStore<'a>` lifetime-parameterized design, `nars_revise_ghosts` batch API, SCHEMA_VERSION bump target, `WitnessChainStore` HashMap design, context-separation law enforcement (G vs witness_ref).

**Status: DONE**
