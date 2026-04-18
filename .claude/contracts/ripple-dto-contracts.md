# Contract Proposal: Ripple DTOs

## Purpose

This document proposes the first stable contract for the treaty layer between anatomy, resonance, bus execution, and metacognitive storage.

The goal is not to make the final perfect schema.
The goal is to prevent premature collapse into ad hoc structs.

## Contract principles

1. `StreamDto` is ingress, not truth.
2. `ResonanceDto` is superposition, not a cache of uncertain facts.
3. `BusDto` is explicit and accountable enough to compile into execution.
4. `ThoughtStruct` is the durable thought object the system can revise, observe, and consult.
5. All contracts should preserve contradiction and temporal persistence.

---

## `StreamDto`

### Role
Temporal arrival object.
Preserves sequence and hints without forcing early commitment.

### Proposed fields

```rust
pub struct StreamDto {
    pub source_kind: SourceKind,
    pub turn_idx: u64,
    pub event_idx: u64,
    pub raw_text: Option<String>,
    pub spo_seeds: Vec<SpoSeed>,
    pub anchor_hints: Vec<AnchorHint>,
    pub family_hints: Vec<FamilyHint>,
    pub recency_score: f32,
    pub confidence: f32,
}
```

### Invariants

- May contain ambiguity.
- Must preserve order.
- Must not claim explicit traversal policy.
- Must not become the universal substrate.

---

## `ResonanceDto`

### Role
Active superpositional field before commitment.

### Proposed fields

```rust
pub struct ResonanceDto {
    pub topic_field: FieldSummary,
    pub angle_field: FieldSummary,
    pub hypothesis_field: FieldSummary,
    pub searchable_field_ref: Option<FieldRef>,
    pub coherence: f32,
    pub contradiction: f32,
    pub pressure: f32,
    pub drift: f32,
    pub style_mix: StyleMix,
    pub family_candidates: Vec<FamilyCandidate>,
    pub branch_candidates: Vec<BranchCandidate>,
}
```

### Invariants

- May contain multiple competing candidates.
- Must preserve contradiction as structure.
- Must not compile directly into text.
- Should support merge and decay rules.

---

## `BusDto`

### Role
Explicit thought packet compiled enough for `p64` and CognitiveShader.

### Proposed fields

```rust
pub struct BusDto {
    pub topic_anchor: TopicAnchor,
    pub angle_mask: AngleMask,
    pub edge_hypotheses: Vec<EdgeHypothesis>,
    pub style_ordinal: StyleOrdinal,
    pub layer_mask: u8,
    pub combine_mode: CombineMode,
    pub contra_mode: ContraMode,
    pub density_target: u8,
    pub support_pressure: f32,
    pub contradiction_pressure: f32,
    pub novelty: f32,
}
```

### Invariants

- Must be compilable into `CausalEdge64` and p64 style settings.
- Must be accountable enough to log and revise.
- Must not hide contradiction inside generic confidence.

---

## `ThoughtStruct`

### Role
Durable, revisable thought object.
This is what the rest of the system should consult.

### Proposed fields

```rust
pub struct ThoughtStruct {
    pub thought_id: ThoughtId,
    pub topic_anchor: TopicAnchor,
    pub angle_mask: AngleMask,
    pub style_ordinal: StyleOrdinal,
    pub frontier_hits: Vec<FrontierHit>,
    pub support_pressure: f32,
    pub contradiction_pressure: f32,
    pub novelty: f32,
    pub episodic_refs: Vec<EpisodicRef>,
    pub provenance: Vec<ThoughtProvenance>,
    pub revision_epoch: u64,
}
```

### Invariants

- Must be storable on blackboard.
- Must support revision rather than replacement-only semantics.
- Must be distinct from raw text and from temporary field state.

---

## Lifecycle suggestion

1. `StreamDto` enters.
2. Multiple `StreamDto`s feed or perturb `ResonanceDto`.
3. `ResonanceDto` collapses into `BusDto` when traversal becomes accountable.
4. `BusDto` executes through `p64 + CognitiveShader`.
5. Resulting explicit state stabilizes as `ThoughtStruct`.
6. `ThoughtStruct` may later seed new resonance.

## Open questions

- Which fields should be enums versus scalar pressures?
- Should `ThoughtStruct` precede `BusDto` in some cases as an intent scaffold?
- What should merge and decay rules be for `ResonanceDto`?
- Which parts of `BusDto` map 1:1 onto current p64 style settings?
