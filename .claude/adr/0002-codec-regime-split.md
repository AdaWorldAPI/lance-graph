# ADR 0002 — I1 Codec Regime Split (proposed)

> **Status:** **Proposed** (drafted 2026-04-24)
> **Supersedes:** None
> **Superseded by:** None
>
> **Scope:** Locks the invariant that governs every codec choice across
> the BindSpace SoA, the Lance persistence schema, AriGraph (episodic +
> triplet_graph), archetype / persona catalogues, and role-key storage.
>
> **Quantitative gate:** jc pillar 5 (+ pending pillar 5b on Pearl 2³
> mask accuracy — see `TECH_DEBT.md` 2026-04-24 jc Pillar 5b entry).

---

## Context

Across multiple sessions the same question kept surfacing in different
disguises: "should we compress field X?" The field varied — 3 SPO planes
in `cognitive_nodes.lance`, episodic fingerprints in `arigraph`, persona
resonance fingerprints, role-key slices — but the answer always hinged
on the **same distinction**: is this field identity-bearing, or is it
similarity-searchable?

The contract crate **already encodes the answer** in
`crates/lance-graph-contract/src/cam.rs`:

```rust
pub enum CodecRoute {
    CamPq,        // argmax regime — compression OK
    Passthrough,  // index regime — lossless required
    Skip,         // too small / not a codec target
}
```

with shipped prose:

> *"Identity lookup must be exact — no codec can survive Invariant I1."*

What this ADR does is **lift that codec-routing enum into a workspace-
wide invariant** and specify how every new structure must be classified.

## Decision

### The invariant (I1 Codec Regime Split)

Every field added to the BindSpace SoA, the Lance persistence schema,
the AriGraph crate, the archetype / persona catalogues, or any role-key
storage MUST be classified into exactly one of three regimes:

| Regime | Codec | When it applies |
|---|---|---|
| **Index** | `Passthrough` (lossless) | Field is used for exact identity lookup, hash-keyed retrieval, independent-component addressability (e.g. Pearl 2³), or VSA bind/unbind role. Bit-level / byte-level round-trip MUST be exact. |
| **Argmax** | `CamPq` (lossy OK) | Field is used for nearest-neighbor similarity, cascade filtering, resonance dispatch. Small error budget acceptable; only relative order matters. |
| **Skip** | `Passthrough` trivially | Field too small to benefit from compression (norms, biases, packed truth values). |

### Classification rules

The following cases are normative and must not be rediscovered per-PR:

| Structure | Regime | Reason |
|---|---|---|
| Pearl 2³ S/P/O planes (`cognitive_nodes.lance`) | **Index** | Mask evaluation requires independent per-role addressability; collapsing to a shared codebook violates Berry-Esseen IID assumptions (see Jirak pillar 5) |
| `integrated_16k` cascade L1 | **Argmax** | Fast HHTL filter — CAM-PQ legitimate as first-tier scent |
| AriGraph `Triplet.{subject, object, relation}` | **Index** | Strings are ground-truth identity; HashMap-keyed lookup |
| AriGraph `Episode.fingerprint` | **Argmax** | Hamming-similarity retrieval; CAM-PQ-eligible as cascade filter |
| `PersonaCard.entry.id` (ExpertId) | **Index** | Enum/ID is the identity; dispatch is exact |
| Per-persona resonance codebook | **Argmax** | Implicit-routing similarity match |
| Role keys (`grammar/role_keys.rs`) | **Index** | Bipolar bind/unbind identity — per I-VSA-IDENTITIES |
| NARS truth (f, c) | **Skip** | 32 bits total; no codec payoff |

### Quantitative gate

Every proposed codec change must either:

1. Keep `cargo run --manifest-path crates/jc/Cargo.toml --release --example prove_it` green (the five-pillar proof), OR
2. Cite which pillar it extends, and add the corresponding arm to the proof binary.

Pillar 5 (Jirak Berry-Esseen) is the current quantitative anchor: weak-
dependent data (25 % shared-codebook prefix + 10 % overlapping role
slices) showed sup-error 0.013287 at d=16384, N=5000 — vs IID baseline
0.011671. That 14 % inflation IS the cost of violating Index regime.

Pending extension (Pillar 5b, TECH_DEBT 2026-04-24): direct Pearl 2³
mask-misclassification rate, three-plane vs CAM-PQ-bundled. Required
before ADR-0002 acceptance.

## Consequences

### What this permits

- CAM-PQ compression on argmax-regime overlays: `integrated_16k`,
  episodic fingerprints, resonance codebooks.
- Stack-side VSA binding of metadata (role, card_id) into role-key
  slices — stays in Index regime because the mapping is deterministic
  and lossless.
- Cascade search paths that use CAM-PQ as first-tier filter + exact
  match on lossless fields as commit tier.

### What this forbids

- Replacing the three S/P/O planes in `cognitive_nodes.lance` with
  CAM-PQ codes. They are Index regime (Pearl 2³ addressability).
- Replacing `AriGraph::Triplet` strings with compressed codes.
- Replacing `PersonaCard.entry.id` with a CAM-PQ code.
- Running `MergeMode::Xor` on state-transition paths (violates
  I-SUBSTRATE-MARKOV; see CLAUDE.md). XOR merge is legitimate only
  for single-writer deltas.

### Migration

Current code is already compliant — no migration required. This ADR
codifies the existing CodecRoute invariant and extends it to AriGraph
+ archetype surfaces that were previously unclassified.

Future codec decisions consult this ADR first, not session discussion.

## Acceptance criteria

This ADR moves from **Proposed** to **Accepted** when:

1. Pillar 5b extension ships (direct Pearl 2³ mask-accuracy measurement).
2. Pillar 5b numbers are cited in this ADR.
3. `@truth-architect` + `@integration-lead` sign off.

Until then, the classification table above is the operating rule but
the ADR lacks its quantitative anchor.

## References

- `crates/lance-graph-contract/src/cam.rs` `CodecRoute` + `route_tensor`
- `crates/jc/src/jirak.rs` pillar 5 (current)
- `crates/jc/examples/prove_it.rs` five-pillar harness
- CLAUDE.md I-VSA-IDENTITIES, I-NOISE-FLOOR-JIRAK, I-SUBSTRATE-MARKOV
- `.claude/board/EPIPHANIES.md` 2026-04-24 "I1 Codec Regime Split"
- `.claude/board/TECH_DEBT.md` 2026-04-24 "jc Pillar 5b"
- ADR 0001 Archetype Transcode + Stack Lock (parent ADR for stack decisions)
