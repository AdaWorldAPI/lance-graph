# Palette256 / 3DSB / PhiSpiral Attention Integration Plan — lance-graph

## Purpose

Capture the updated integration shape:

```text
super-exact but economic one-dimensional Palette256 cosine attention lane
        +
3DSB / spatial-splat-block 4D representation
        +
PhiSpiral256 / original Poincare leaf location
        +
OGIT / OWL / DOLCE domain inheritance
        +
BGZ17 recoverable sparse traces
```

The goal is not to make attention larger.

The goal is to replace broad dense attention search with a compact, content-addressable, ontology-inherited, recoverable semantic-spatial attention packet.

## Capstone links

This integration plan sits on top of the following capstone/headstone documents:

- [3DGS Cesium / BindSpace4 Headstone Exploration](./3DGS-Cesium-BindSpace4-headstone-exploration.md)
- [BGZ17 / Poincare SparseEdgeField Tectonic Map Plan](./BGZ17-Poincare-SparseEdgeField-tectonic-map.md)
- [PhiSpiral256 SoA Cross-System Integration Plan](./PhiSpiral256-SoA-cross-system-integration-plan.md)
- [3DGS PR-X12 Crosspollination Capstone](./3DGS-PRX12-crosspollination-capstone.md)
- [Cognitive Substrate Convergence v1](./cognitive-substrate-convergence-v1.md)

Cross-repo capstone:

```text
AdaWorldAPI/ndarray/.claude/plans/PR-X12-tensor-container-expansion-capstone.md
```

## Core thesis

```text
Palette256 replaces broad attention search.
OGIT / OWL / DOLCE inheritance keeps the semantic range economic.
3DSB carries local spatial/block structure.
Poincare restores the original relative leaf geometry.
PhiSpiral256 pins the orthogonal local residue.
BGZ17 makes the residue recoverable.
```

In BF16 language:

```text
DOLCE / OWL / OGIT domain inheritance
  = exponent / semantic range

Palette256 cosine lane
  = one-dimensional attention candidate

3DSB + Poincare + PhiSpiral256
  = semantic-spatial mantissa

BGZ17
  = recoverable sparse sampling tail
```

## Why this exists

Dense attention materializes too much:

```text
Q × K^T
  -> broad pairwise similarity
  -> large intermediate score matrix
  -> context selected after expensive comparison
```

The AdaWorld route should be:

```text
semantic domain inheritance
  -> CAM/codebook lookup
  -> Palette256 cosine candidate
  -> Poincare-local leaf geometry
  -> PhiSpiral residual location
  -> BGZ recovery schedule
  -> exact hydrate only if certified
```

The attention path becomes content-addressable and recoverable instead of materialized.

## Term separation

Do not blur these lanes:

```text
Palette256
  one-dimensional cosine / candidate-ranking attention lane

3DSB
  spatial-splat-block / 4D local field carrier

Poincare
  local relative chart around the semantic or spatial anchor

PhiSpiral256
  one-byte orthogonal residual location inside the Poincare-local chart

BGZ17
  golden offset/stride recoverable sparse sampling skeleton

CAM_PQ
  meaning / semantic basin anchor

PolarQuant
  magnitude / similarity lane

OGIT / OWL / DOLCE
  domain inheritance / semantic range

CausalEdge64
  causal semantic trajectory candidate between thoughts, blocks, or schemas
```

## Conceptual packet

Do not treat this as a JSON object. Treat it as an address packet.

```rust
#[repr(C)]
pub struct SemanticSpatialAttentionLeaf {
    /// DOLCE / OWL / OGIT inherited semantic compartment.
    pub domain_id: u32,

    /// CAM_PQ / local meaning anchor.
    pub cam_pq_id: u16,

    /// Super-exact but economic one-dimensional cosine attention candidate.
    pub palette256_id: u8,

    /// Original Poincare leaf-local orthogonal location.
    pub phi_spiral_id: u8,

    /// 4-bit magnitude + BGZ offset/stride family in packed form.
    pub mag4_bgz4: u8,

    /// Optional confidence/truth texture lane.
    pub confidence_q: u8,
}
```

A harder-packed hot-path variant can collapse the tail into existing SoA lanes or route keys. The important point is the lane separation.

## Attention replacement flow

```text
Input thought / tensor block / scene block / sparse edge field
        ->
OGIT / OWL / DOLCE inherited domain lookup
        ->
CAM_PQ meaning anchor
        ->
Palette256 cosine attention candidate
        ->
3DSB local field/block carrier
        ->
Poincare local chart around the anchor
        ->
PhiSpiral256 residual location
        ->
BGZ17 offset/stride recovery family
        ->
CausalEdge64 trajectory candidate or certified hydrate decision
```

This replaces broad dense attention with a domino cascade of stable indexes.

## 1D attention lane

Palette256 is the economic attention spine:

```text
query context
        ->
cosine against 256 ranked/codebook candidates
        ->
palette256_id
        ->
small candidate set
```

This lane is deliberately one-dimensional from the routing perspective:

```text
attention candidate = u8
```

The lost geometry is not shoved into Palette256. It is recovered by 3DSB + Poincare + PhiSpiral256.

## 3DSB + spiral as 4D representation

3DSB is the local block carrier:

```text
3D spatial / splat / edge / tensor block
  + semantic/provenance/cert lane
  = 4D local carrier
```

PhiSpiral/Poincare gives the leaf its local place:

```text
semantic/spatial anchor
        ->
Poincare-local chart
        ->
orthogonal residual
        ->
PhiSpiral256 address
```

This means a leaf can carry:

```text
what it attends to
where the unexplained detail lives
how strong it is
how to recover it
```

without materializing the full context.

## Domain inheritance keeps it economic

Without ontology inheritance, every local code has to carry too much.

With DOLCE / OWL / OGIT inheritance:

```text
upper semantic compartment
  -> inherited schema / class / property family
  -> CAM_PQ meaning anchor
  -> local attention and residue packet
```

The ontology range acts like an exponent. The local packet acts like a mantissa.

```text
semantic exponent:
  domain_id / schema_label / CAM_PQ

leaf mantissa:
  Palette256 + PhiSpiral256 + mag4 + BGZ offset/stride
```

## Relationship to SparseEdgeField

SparseEdgeField stores structural seams, not dense maps.

```text
Palette256
  says which attention candidate matters

SparseEdgeField
  says where seams / boundaries / missing links live

PhiSpiral256
  gives each seam a local orthogonal place

BGZ17
  remembers how to recover sparse traces
```

The same packet can route:

```text
semantic context
spatial edge
3DGS block
GGUF tensor block
mailbox thought transition
```

## Relationship to GGUF / PR-X12

GGUF attention headers can carry this as implicit context headers.

PR-X12 can encode the packet as a hierarchical block grammar:

```text
block header
  -> domain / CAM_PQ semantic exponent
  -> Palette256 attention lane
  -> PhiSpiral/BGZ leaf mantissa
  -> optional exact tail
```

This supports decode-during-GEMM or decode-during-query:

```text
read compact header
  -> choose candidate / skip / refine
  -> hydrate exact payload only inside a certified window
```

## Relationship to mailbox / BindSpace cascade

No JSON objects. No materialized context.

```text
SemanticSpatialAttentionLeaf
        ->
CausalEdge64 candidate
        ->
MailboxSoA energy integration
        ->
CollapseGateEmission
        ->
next thought-local shard
```

Palette256 replaces broad search. CausalEdge64 carries the trajectory. MailboxSoA integrates the baton.

## Calibration and tests

Compare:

```text
A. dense attention baseline
B. Palette256 only
C. Palette256 + ontology inheritance
D. Palette256 + PhiSpiral leaf location
E. Palette256 + 3DSB + PhiSpiral + BGZ17
F. full hybrid with SparseEdgeField and CausalEdge64 cascade
```

Metrics:

```text
attention recall@1 / recall@k
candidate fanout reduction
context hydration reduction
wrong-high-confidence rate
semantic compartment recall
orthogonal leaf-location recall
SparseEdgeField seam reconstruction error
bytes per context packet
ns per route
cache footprint
```

## Acceptance criteria

```text
Palette256 beats broad lookup on candidate fanout.
Ontology inheritance improves semantic compartment recall.
PhiSpiral improves orthogonal leaf-location recall over Palette256-only.
BGZ17 improves sparse trace recovery over random sampling.
3DSB preserves enough local structure to reconstruct seams or splat blocks.
Full hybrid reduces exact context hydration without losing correctness.
```

## Anti-patterns

Do not turn the packet into JSON.

Do not make Palette256 carry geometry.

Do not make PhiSpiral256 carry meaning.

Do not make OGIT / OWL / DOLCE a runtime object hydrate on the hot path.

Do not decode full tensors, scenes, or context windows unless the certificate requires it.

Do not let neural decoders sit inside the hot loop. Neural systems may train codebooks; deterministic tables and kernels should run the route.

## Wall sentence

```text
Palette256 is the cheap attention spine; OGIT/OWL/DOLCE gives the semantic range; 3DSB and Poincare restore local structure; PhiSpiral256 marks the leaf residue; BGZ17 remembers how to recover it.
```
