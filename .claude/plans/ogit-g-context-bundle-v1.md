# OGIT-G Context Bundle — Tier-1 sub-plan (v1)

> **APPEND-ONLY governance.** This is a sprint-2 plan-doc covering Patterns A+B+C of the unified OGIT architecture: SPO-G u32 slot + ContextBundle typed surface + GenericBridge dispatching per-G ConsumerPointer.

## Motivation

The workspace has an OGIT registry (`lance-graph-ontology`, shipped via PR #355) delivering O(1) namespace lookup (2554× faster than SPARQL-proxy). What's missing:

1. **The u32 G doesn't yet appear in SPO triples.** SPO is `(S, P, O)` today; for oxigraph-shape SPO-G quads, G becomes the 4th tuple position. Lance MVCC versioning provides the temporal axis — `(G, version)` becomes `(G, lance_version)`.
2. **G doesn't resolve to a typed bundle.** It resolves to a u32 + `MappingRow` cascade-cols (PR #355 D-CASCADE-V1-7). Pattern B needs an explicit `ContextBundle` type with named slots.
3. **No GenericBridge yet.** PR #29 (`SmbMembraneGate`) and PR #98 (`MedCareMembraneGate`) hand-rolled per-consumer newtype gates because of the orphan rule. With `ConsumerPointer`-as-data + one canonical bridge, the orphan rule problem dissolves.

## Three deliverables

### D-OGIT-G-1: SPO-G u32 slot in the quad store

Extend `lance-graph::graph::spo` and `arigraph::triplet_graph` from `(S, P, O)` to `(S, P, O, G)`. Lance MVCC versioning provides the temporal axis.

- Schema: add `g: u32, version: u32` columns to `MappingRow` + SPO Lance dataset
- Writer: `SpoBridge::promote_to_spo` (shipped PR #355 D-ONTO-V5-9) grows `g: u32` parameter (default 0 backwards-compat)
- Reader: queries scope by G or scan all G
- Lance time-travel: `read_as_of_version(g, v)` for partial-state queries

**Effort:** medium (~300 LOC + schema migration + tests).

### D-OGIT-G-2: ContextBundle as typed OGIT surface

New type in `crates/lance-graph-ontology`:

```rust
pub struct ContextBundle {
    pub g: u32,
    pub version: u32,
    pub domain_name: SmolStr,
    pub inherits_from: Option<u32>,  // parent G (DOLCE = root, G=0)

    // OWL overlay
    pub ontology: Option<Arc<OntologySlot>>,
    pub codebook: Option<Arc<CodebookSlot>>,
    pub schema: Option<Arc<SchemaSlot>>,
    pub labels: Option<Arc<LabelsSlot>>,
    pub vocabulary: Option<Arc<VocabularySlot>>,

    // Operational behavior
    pub consumer_pointer: Option<Arc<ConsumerPointer>>,

    // Per Pattern G (best-practice thinking inheritance)
    pub thinking_styles: SmallVec<[u8; 8]>,
    pub thinking_adjacency: Option<Arc<AdjacencyStore<u8>>>,
    pub qualia_codebook: Option<Arc<QualiaCodebook>>,

    // Per-G specialization
    pub mul_threshold_profile: Option<MulThresholdProfile>,
    pub trust_texture_set: SmallVec<[u8; 4]>,
    pub flow_state_set: SmallVec<[u8; 4]>,
}
```

Most slots `Option<Arc<…>>` — populated by hydrator (Pattern D, separate sub-plans). Inert G has `consumer_pointer = None`; queryable but not executable.

**Effort:** small (~200 LOC type defs + a few sample hydrators for DOLCE root, Healthcare, Gotham).

### D-OGIT-G-3: GenericBridge dispatching per-G ConsumerPointer

New in `crates/lance-graph-callcenter/src/generic_bridge.rs`:

```rust
pub struct GenericBridge {
    registry: Arc<OntologyRegistry>,
}

impl MembraneGate for GenericBridge {
    fn should_emit(&self, commit: &Commit, ctx: &RequestContext) -> bool {
        let g = ctx.g;
        let bundle = self.registry.resolve(g);
        let pointer = bundle.and_then(|b| b.consumer_pointer.clone());
        // dispatch via pointer.rbac_policy, pointer.action_capabilities, etc.
        ...
    }
}
```

Ergonomic wrappers for backwards-compat with PR #29 and PR #98:
```rust
pub struct SmbMembraneGate(GenericBridge);
pub struct MedCareMembraneGate(GenericBridge);
```

The 33 medcare regulatory tests reroute through `MedCareMembraneGate(GenericBridge::for_g(MEDCARE_G))` and stay green.

**Effort:** medium (~200 LOC + tests).

## Open design questions

1. **ConsumerPointer schema location.** `lance-graph-contract::consumer::ConsumerPointer` (zero-deps canonical) vs `lance-graph-ontology::ConsumerPointer` (co-located with registry). Recommend contract — every consumer pulls it.
2. **Action capabilities expressed in ConsumerPointer.** Meta-3 HIGH #1 (medcare sprint) flagged that gate routes Read/Write only; actions go through policy.evaluate directly. `ConsumerPointer.action_capabilities` slot closes this — declares which actions need gate-side Escalate (BtM, finalize/retract, anonymize).
3. **Inheritance semantics.** G=Healthcare inherits from G=DOLCE — set-union for SmallVec slots (`thinking_styles`, `trust_texture_set`), override for scalar slots (`mul_threshold_profile`).
4. **Versioning rollover.** When Healthcare:v1 → v2, both stay registered. Consumers target via `(G, version)` const constants.
5. **Inert bundle storage.** Cold-only in Lance dataset (DOLCE, FMA never-active default); hot index in memory for active G.

## Acceptance criteria

- 3 deliverables independently shippable (D-OGIT-G-1, D-OGIT-G-2, D-OGIT-G-3)
- Backwards-compat: existing SPO consumers (without G) keep working; PR #29 + PR #98 newtype gates keep working as ergonomic aliases
- Tests: SPO-G round-trip, ContextBundle resolution, GenericBridge dispatch matching legacy newtype behavior

## Dependencies

- Builds on PR #355 (lance-graph-ontology, OntologyRegistry, NamespaceRegistry::seed_defaults)
- Cross-references:
  - `unified-ogit-architecture-v1.md` Tier 1 (W1's master)
  - TECH_DEBT entries TD-OGIT-G-SLOT-1, TD-CONTEXT-BUNDLE-2, TD-GENERIC-BRIDGE-3 (W5)
  - Sibling sub-plans: `compile-time-consumer-binding-v1.md` (W11; manifest.yaml + ractor) builds on D-OGIT-G-2; `anatomy-realtime-v1.md` (W12; FMA hydrator) consumes D-OGIT-G-2

## Sequencing within Tier 1

1. D-OGIT-G-2 first (just types; no migration risk)
2. D-OGIT-G-1 second (SPO-G schema; migration; needs ContextBundle for G→consumer routing)
3. D-OGIT-G-3 third (uses both)
