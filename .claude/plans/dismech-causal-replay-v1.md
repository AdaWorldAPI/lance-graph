# dismech-causal-replay-v1 — causality replay over the DisMech palette

> **Status: PROPOSED — PLAN/BOARD ONLY.** No production implementation in
> this PR. Sibling and FIRST DOMAIN INSTANTIATION of
> `rubicon-loco-rung-cognitive-fabric-v1` (2026-08-29): that plan verified
> the fabric (loco carrier, callable atoms, masks, counterfactual,
> `temporal.rs` replay); this plan binds ONE domain to it — recorded
> disease-mechanism causal chains under the `ogar-dismech` predicate
> palette. **Every `F-RLR-*` STOP gate is inherited unchanged**; nothing
> here re-litigates a fabric verdict.
>
> **Operator direction (2026-08-31, verbatim anchors):** the mechanics are
> *"causality replay shaped"* (the keystone); candidates evaluate by
> *"Mengenlehre"*; *"Shannon proprioception + EWA Sandwich"* carry
> uncertainty; *"rung 0-9 inference frontier scheduling"*;
> *"ogar-loco Orchestration as config / ogar-r2il thinking as config"*;
> *"lance-graph p64 stockfish-rs ALU 64x64 4096 shape later,
> cognitive-shader-driver as ALU crunching"*.

## §0 State audit — what EXISTS (verified 2026-08-31, sibling repos checked per F-RLR-11)

| thing | state | where |
|---|---|---|
| causal predicate palette | **MINTED, loco-addressable** — 19 predicates (`causes`..`variant_of`), `FnIndex` `0x90..=0xA2`, concept `0x0333` | OGAR `ogar-dismech` (mirrors `ogar-ro`'s `RelationVocabulary` pattern) |
| mechanism modules | **ADDRESSED** — 124 cross-disease templates, 585 nodes, `conforms_to` address inheritance, append-only registry at `0x0333` | consumer-side transcode (`module_registry`) |
| corpus transcode + replay gate | **SHIPPED** — resolver reproduces the upstream builder field-for-field; parity diffs against the 1,870 pathographs the upstream resolver produced (replay-equivalence IS its gate) | consumer-side transcode (per OGAR's own doc: the public `dismech-rs` posture — zero OGAR dep, like `ogar-obo`) |
| baked slab | **NONE EXISTS.** The bake bin emits loose TSVs to a local dir; no release tag, no pin-table row anywhere | — (consumer leg, §4 W-C) |
| `lance-graph-ontology` cache | **UNRELATED** — OGIT-TTL tenant naming registry; zero dismech references | this repo |
| set algebra (∩ ∖ ⊆) | **EXISTS + CALLABLE** — `contract::revision::EvidenceMask` (#1075) | this repo |
| counterfactual / intervention | **EXISTS** — `contract::counterfactual`; R2IL V4 probe rows | this repo |
| replay substrate | **EXISTS** — `planner/src/temporal.rs` (version-range reads, per-reader rung, replayable) | this repo |
| EWA covariance | **EXISTS, not loco-addressable** — `jc::ewa_sandwich` (Pillar 9/9b certify `J·W·Σ·Wᵀ·Jᵀ` push-forward) | this repo |
| Shannon entropy | **EXISTS, SCATTERED** — ≥6 uncoordinated `entropy()` surfaces (rubicon §C) | this repo |
| loco call pattern to copy | `lance-graph-ogar/src/recipe_vocab.rs` — `op_of` / `recipe_of` / `ladder_program` | this repo |
| chain-step eval kernel | `CausalEdge64` + `NarsTables` ("precomputed NARS as lookup tables") | causal-edge / planner cache |

**The gap is exactly one thing: nothing REPLAYS a recorded causal chain
against an evidence set.** Every ingredient exists; the composition does not.

## §1 The object

**Replay, never generate.** The domain knowledge is trajectory-shaped
(curated mechanism chains); differential evaluation is *which recorded
trajectories are consistent with this evidence*, i.e.:

```text
candidates = { recorded chains }                       (sets of chain ids)
evidence arrives → EvidenceMask ∩ on support,
                   ∖ on refute (the skip-word semantics as ordinals)
frontier pick   → expected entropy reduction over candidates (§W4)
each expansion  → REPLAY the chain: loco calls under the dismech
                   vocabulary, one CausalEdge64 + NarsTruth revision per
                   step, trace on temporal.rs
counterfactual  → the SAME replay with one edge cut (Pearl rung 3)
grade           → replay equivalence: same inputs ⟹ byte-identical trace
```

The trace of an evaluation is itself a loco program — thinking as config
makes causality replay **self-hosting**: yesterday's evaluation replays
today byte-for-byte, and template-equivalence machinery grades it.

## §2 Design constraints (inherited + house)

1. **No new carrier** (`F-RLR-2`, automatic STOP): steps are loco
   `(FnIndex, value)` calls under the already-minted dismech vocabulary;
   traces ride `temporal.rs`; per-step state is `CausalEdge64` +
   `NarsTruth`. A `ReplayFrame` struct that owns what a lane already holds
   is the zero-copy-warden's MATERIALIZES verdict — methods on carriers,
   never a new layer.
2. **No second set algebra:** Mengenlehre = `EvidenceMask` ops. If a
   needed op (symmetric difference, popcount-weighted split) is missing,
   it is added THERE, with the field-isolation discipline.
3. **No 7th entropy.** W4 opens with a consolidation DECISION — which of
   the ≥6 existing `entropy()` surfaces is canonical for the candidate
   readout — recorded on the board before any call site is written.
4. **Constitutional (rubicon, carried verbatim):** ambiguity/entropy/
   parallax never terminate cognition — only the Rubicon boundary owns
   stop/commit/veto. **Shannon reduction ≠ evidential increase**; the
   frontier picks WHERE to look next, never WHAT is true. Prefetch ≠
   belief.
5. **Fixtures in this repo are SYNTHETIC.** The real corpus and its bake
   live consumer-side (§4 W-C pointer); this repo's gates run on
   generated chains whose ground truth is constructed, exactly the
   `dismech_parity` discipline at engine scale.
6. **Predicate ordinals, not strings**, on every hot path; the palette
   binds at the membrane (`op_of`-shaped). A `typed → stringify → parse`
   step anywhere inside is `F-RLR-7`'s STOP.
7. **Rung semantics:** `r` = the `RungLevel` discriminant 0..=9 (rubicon
   §I pin). Horizon reads via `EpistemicMode::for_rung`; nothing here
   privileges a rung.

## §3 Waves

### W0 — the two NNUE numbers + palette round-trip (measure, no code) — `D-DCR-0`

Stockfish-depth claims are arithmetic over two measured quantities this
workspace has never measured for THIS composition:

- **evals/ms**: one chain step = `NarsTables` lookup + `CausalEdge64`
  revision, benched over synthetic chains (existing kernels, bench only).
- **effective branching**: candidate-set shrink per evidence item under
  `EvidenceMask` ops on generated corpora of 10³/10⁴/10⁵ chains.
- **palette round-trip**: dismech `FnIndex` `0x90..=0xA2` compose/
  decompose through `ogar_loco` (cite the OGAR-side tests; add none here
  unless a gap is measured).

**Gate (two-sided):** numbers recorded with the harness committed;
the deferred ALU wave (§5) states its BUY threshold in terms of them.
KILL: if a single full scan of 10⁵ chains costs less than one frontier
decision, the scheduler is decoration at that scale — record it and
descope W5 accordingly.

### W1 — replay core — `D-DCR-1`

Replay ONE synthetic chain end-to-end: loco calls under the dismech
vocabulary → per-step `CausalEdge64` + `NarsTruth` revision → trace on
`temporal.rs` → typed receipt on the #879 sealed-cycle path.

**Gates:** (a) determinism — two replays of identical inputs produce
byte-identical traces; (b) can-fire — a one-edge perturbation of the
chain produces a trace that DIFFERS at exactly the perturbed step index;
(c) silence — a perturbation of a non-replayed sibling chain changes
nothing. Disable-verified per the falsifiability rule.

### W2 — Mengenlehre candidate evaluation — `D-DCR-2`

Candidate sets as masks; support evidence intersects, refuting evidence
subtracts (the skip-word semantics — `biomarker/disputed/modifier/
protective/refuted/unknown` — as refute-class ordinals, never strings).

**Gates:** anti-vacuity (`kept * 3 < total` on the synthetic corpus);
two-sided discrimination (a discriminating evidence item must split the
set; a redundant one must NOT shrink it further); the refute path has
its own can-fire + stay-silent pair.

### W3 — counterfactual replay (Pearl rung 3) — `D-DCR-3`

The SAME W1 replay with one edge cut, through `contract::counterfactual`
— never a second replay path.

**Gates (two-sided):** cutting a load-bearing edge flips the chain's
consistency verdict; cutting a redundant (parallel-path) edge does not.
Both fixture halves constructed so the property is reachable — the
fixture's SHAPE is part of the coverage.

### W4 — Σ transport + the Shannon readout — `D-DCR-4`

Per-edge evidence counts → `NarsTruth` (`w/(w+1)`) → a small Σ; pushed
along the replayed path via `jc::ewa_sandwich`'s certified form (Pillar
9/9b are the standing gates — the mint rule from the math-atoms
discussion applies: no loco address for the atom while its pillar is
red). Candidate-set entropy = the proprioception readout; value of the
next observation = expected entropy drop.

**Gates:** Σ growth monotone under fan-in; a discriminating observation
must score strictly above a redundant one (two-sided); the entropy
CONSOLIDATION decision (constraint 3) recorded before the first call.

### W5 — frontier scheduling — `D-DCR-5` — **HELD**

Priority = expected-information-gain / rung-cost, horizons via
`EpistemicMode::for_rung`. **HELD behind the operator rung 5–9 table
ruling** (persona-vs-rung-ladder O-items) AND W0's KILL check. Interface
note only; the scheduler never decides truth, causality, band promotion
or revision acceptance (§K carried).

### W-C — consumer leg (pointer, not this repo's work)

The corpus bake (TSVs → a tagged, gated artifact) and the live-evidence
binding happen consumer-side under that repo's own pin doctrine. This
plan's engine consumes classid-addressed rows and synthetic fixtures
only. §0's "NONE EXISTS" row is the reason this pointer exists.

### Deferred — p64 64×64 ALU + shader-driver dispatch

64×64 = 4096 = one node's bit budget: the ALU tile is exactly one node.
NNUE's real trick is the incrementally-updated accumulator over a delta
— evidence landing IS the delta. **BUY only on W0's numbers**; until
then this row exists so nobody builds it early. Dispatch belongs to
`cognitive-shader-driver` (the loop that can't-not-think while F is
above floor) — never a bespoke driver.

## §4 What this plan deliberately does NOT do

No new carrier, mask type, entropy surface, replay substrate, or
scheduler-that-decides-truth. No renames. No consumer corpus data in
this repo. No R2IL transcode of anything (`ogar-r2il`'s own ruling:
executed, never pre-converted). No second Rubicon.

## §5 Model allocation

Opus: this plan, wave specs, review, W4's consolidation decision.
Sonnet: W0 harness, W1–W3 implementation against written specs, one
source in / one shape out. Haiku: only the guarded-executor card.

## §6 D-id table

| id | wave | status |
|---|---|---|
| D-DCR-0 | W0 measurements | Queued |
| D-DCR-1 | replay core | Queued (after D-DCR-0) |
| D-DCR-2 | Mengenlehre evaluation | Queued |
| D-DCR-3 | counterfactual replay | Queued |
| D-DCR-4 | Σ transport + Shannon readout | Queued (entropy decision first) |
| D-DCR-5 | frontier scheduling | **HELD** (operator rung table + W0 KILL) |
| D-DCR-6 | consumer-leg pointer honoured (no corpus data lands here) | standing gate |
