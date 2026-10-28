# EAW-2 — CausalEdge64 epistemic/activation DR population projection plan

**2026-10-27 — Docs only; proposed lab architecture, not implemented, admitted, executed or benchmarked.** Main-based and independent of other EAW branch tips. [EAW-0 scratchpad #1820](https://github.com/AdaWorldAPI/lance-graph/pull/1820), EAW-1 logical coordinate/presence stage; [DR64³ gap inventory](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/board/entries/2026-10-26-dr64-cubed-amendment.md); [EXO-146](https://linear.app/exo-red/issue/EXO-146); [EXO-76](https://linear.app/exo-red/issue/EXO-76); [EXO-135](https://linear.app/exo-red/issue/EXO-135); [EXO-51/54](https://linear.app/exo-red/issue/EXO-54).

## Question and thesis

Can existing **per-edge** `CausalEdge64` evidence be decoded under a declared class/rail/generation and projected back onto already addressed DR64 relationship positions as a *derived, witnessed epistemic population*? Multiple edge records may share one pairwise projection. The derived selection is NOT a new relationship, a causal upgrade, a new column/ABI, or the canonical source-of-truth for cognition.

Shared base address: `(Identity, Property/Edge)`. Two distinct views:
- `(Identity, Property/Edge, Predicate)` for ON/WHEN/DO typed mappings and K3 eligibility.
- `(Identity, Property/Edge, CausalEdge64 evidence instance)` for per-edge epistemic/activation readings. The evidence instance is not automatically a third 64-wide *physical* axis; no 64-bit state explosion is permitted.

A versioned, actual source/focus witness is required to project evidence onto rule coordinates. Do not reuse the CE64 palette S/P/O byte fields as a globally unique ID or assume two codebooks share meanings.

## Current, source-checked CausalEdge64 v2 layout and operator corrections

| Bits | Actual storage | Safe reading boundary |
|---|---|---|
| 0..23 | Three Palette256 S/P/O codes | Only via a declared, generation-checked palette codebook, not arbitrary global IDs. |
| 24..39 | 8-bit NARS f/c | Quantized measurements, not automatically source-fact truth or independent corroboration. |
| 40..42 | `CausalMask` Pearl/SPO selection | Which planes a query projects. **Projection flag is not a causal certificate.** |
| 43..45 | Orientation | Source/interpreter declaration required. |
| 46..49 | signed i4 semantic activation pressure OR legacy inference mantissa | Overlapping alternative interpretation; operator C40/ISA-LEGACY test-first migration **OPEN** ([EXO-146](https://linear.app/exo-red/issue/EXO-146)). Never read both as simultaneous facts. |
| 50..52 | WIRE-50 operator `Option<Maturation3>` pyramid semantics; presence OUTSIDE | Current code still also offers plasticity or novelty/wisdom readings; feature/read declarations must prohibit accidental mixing. Code adoption OPEN EXO-146. Raw 0 ≠ Wisdom. |
| 53..58 | W (6 bits), live thought layer 0 givens, 1..63 hypotheses per operator decision (#1597/#1628) | NOT a pointer, durable source ID, or independent corpus evidence root. After sealing, recycled slots invalidate W as a replay pointer. Historical `layout.rs` corpus-root doc is stale; do not import it. |
| 59..63 | **ONE `EpistemicState5` field** | `raw5 = topology2 | (certification3 << 2)`. 4 topologies × 6 certifications = 24 meaningful codes; certification 6/7 (8 codes) RESERVED. |

Canonical source:
- [CausalEdge64 layout](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/causal-edge/src/layout.rs), [edge.rs](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/causal-edge/src/edge.rs);
- [EpistemicState5 class/rail/generation declaration and facts_population](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/epistemic_state5.rs);
- operator six decisions [#1640](https://github.com/AdaWorldAPI/lance-graph/pull/1640), live Thought W [#1628](https://github.com/AdaWorldAPI/lance-graph/pull/1628), WIRE/ISA-LEGACY [EXO-146](https://linear.app/exo-red/issue/EXO-146).

**Epistemic code values:** Topology2 Direct / IndirectKnown / IndirectUnknown / Unknown. Certification3 Open / Associated / Related / Supports / CausalCandidate / Causes. Epi5 is a product with 24 meaningful, eight reserved states. A claim `IndirectUnknown × Causes` can be lawful after an independent causal intervention certificate even though the mechanism remains unresolved. Certification cannot be deduced from Pearl/SPO flags, activation or confidence alone.

## A necessary TWO-DOMAIN mapping

`facts_population(required: Facts) -> u32` selects a subset of **32 Epi5 raw codes**, *not* the DR64² source-object population mask over up to 4096 addressed positions. The read path is conceptually:

```text
real source-qualified relationship addresses + edge instance list
    -> validate edge provenance/class/rail/Epi5 generation
    -> decode raw5; refuse 24..31 and undeclared/legacy alias
    -> membership against the u32 Epi5 code selection
    -> project allowed edge *instances* onto the same sealed relationship addresses
    -> derive witnessed DR64 presence-of-eligible-evidence population
```

For more than one edge record at one coordinate, preserve the complete support path (multiple roots, contradictory or unknown evidence). A projected OR of `Associated` from edge1 and `Direct` from edge2 is NOT automatically one edge that is `Direct×Associated`. Per-edge conjunction must happen *before* row-level existential projection unless a source-attested multi-evidence law says otherwise. Distinct input records sharing an evidence root cannot be counted as independent observations.

No automatic rank promotion: `Associated` → `Related` → `Supports` → `Causes` requires separate certification obligations (EXO-135, EXO-127). NARS revision only where source root independence and carrier closure are admitted (EXO-132/138).

## Activation versus epistemic awareness

Let `Ep_v(r)` mean “at least one source-declared Epi5 edge for addressed relationship r satisfies exact required facts,” `Act_v(r)` mean “a separately declared activation/maturation law makes r eligible for the requested inquiry,” and `Presence_v(r)` be source attested. They are separate. A downstream mask fold is a transient **reasoning-eligibility** decision, not actor cognition, truth promotion or permission to mutate.

An activation change must not implicitly modify the raw 59..63 Epi5 code. An epistemic certification change must not rewrite 46..52. An Epi5 change must be source-/law-authorized and replay-witnessed, not a magic side effect of `with_activation`.

## Preregistered falsifiers (NOT RUN)

| ID | Fixture | Required |
|---|---|---|
| E2-EPI-01 | Enumerate all 32 raw5 codes, every declared class/rail, reserved legacy provenance | 24 meaningful, 8 reserved; reject undeclared, mismatched generation and untrusted v1 provenance. |
| E2-EPI-02 | `IndirectUnknown × Causes` versus `Direct × Associated`; apply `facts_population` independently | Membership exact; cross-factor masks share semantics without a false ordinal comparison. |
| E2-EPI-03 | Two edge instances at same relationship address, one `Direct×Open`, other `Unknown×Associated` | No fictional `Direct×Associated` row from cross-instance conjuncts. |
| E2-EPI-04 | Same packed 46..49 bits under activation vs legacy inference declaration | Only declared reading allowed; no concurrent interpretations of the same field. |
| E2-EPI-05 | WIRE-50 `0` with presence externally 0 and 1; old plasticity reading comparison | No claim of Wisdom or actual absence from 0; unadmitted legacy reading explicitly refused. |
| E2-EPI-06 | Live W=7 thought is landed/sealed, slot7 reused for unrelated thought | Durable replay never resolves using stale W pointer; external version + thought id/witness needed. |
| E2-EPI-07 | Toggle activation and eligibility under pinned Epi5 code, then toggle Epi5 grade with activation held | Distinct masks change correctly; no hidden certification promotion. |
| E2-EPI-08 | Change source scope/schema/class/focus seal with identical 64-bit edge payload | Rebind or refuse; no cross-source palette identity alias. |
| E2-EPI-09 | Two edges from same evidence root, compared with two truly independent roots | No fake double-count NARS confidence or causal support. |
| E2-EPI-10 | Projection from Epi5 u32 selector to DR64 row mask with 4,096 rows, multiple edge instances and 2 distinct local maps | Full mask and witness parity, no code-index-as-row or 64-bit-payload-as-coordinate bug. |

Use source-independent row/edge oracle and source-declared classes; no self-reference through `facts_population` as the only checker. The EAW-2 lab is BLOCKED from making a canonical Maturation3 implementation claim until EXO-146's real-owner field migration is landed and independently validated.

## Risks, economic question and interface boundary

Measure cost of decoding raw5 per edge, grouped raw-code indexes vs direct scan, projection/intermediate masks, repeated queries and invalidation. Existing compiled `COMPILED_FACTS_V1` is a derived shortcut, not authority; a wider [u64;32] eligibility table must not be treated as source law. No inference compiler, new epistemic storage, CE64 layout mutation, activation loop, W reinterpretation, or effectful business dispatch.

**Success condition:** one source-bound relationship address can be inspected through several *separately declared* and independently testable evidence/activation lenses without ever becoming a new relationship or a certified causal claim as a side effect. Stage EAW-3 owns the later co-addressed TERNLOGQ fold.
