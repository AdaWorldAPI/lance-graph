# Shader-field lineage audit — D-ARW-0 follow-up

**Date:** 2026-08-30  
**Status:** SOURCE AUDIT / KNOWLEDGE NOTE, not an implementation plan  
**Parent:** merged PR #1078, `alpha-reason-witness-cognitive-fabric-v1.md` + shader-field lineage addendum

This note closes the first concrete archaeology pass requested by D-ARW-0 after
#1078 merged. It deliberately separates current executable source, historical
source, operator-recovered intent, and hypotheses. It does **not** authorize a
new DTO, address space, rung tenant, provenance bit, scheduler, or controller.

## 0. Evidence labels

- **[S] SOURCE FACT** — current executable source establishes the claim.
- **[HS] HISTORICAL SOURCE FACT** — committed historical source/merged PR establishes the old state or documented intent.
- **[OR] OPERATOR-RECOVERED INTENT** — original design intent recalled by the operator; not promoted to source fact unless independently recovered.
- **[BW] BROKEN WIRE** — both relevant ends exist, but the load-bearing identity/data does not cross the seam.
- **[MD] MODERN DESCENDANT** — later mechanism plausibly carries part of the old function; ancestry does not imply identity.
- **[H] HYPOTHESIS** — useful only behind a falsifier.

## 1. Current executable chain

### 1.1 Thinking-engine field

**[S]** `crates/thinking-engine/src/dto.rs` still names the four-stage field:

```text
Φ StreamDto        sensor/intake indices
Ψ PerturbationDto  dense f32 energy + top_k[8]
B BusDto           committed headline/top_k/provenance
Γ ThoughtStruct    stabilized thought
```

`PerturbationDto` is explicitly documented there as the mechanically renamed
successor of the old `ResonanceDto` and as a Morton-tile / inverse-pyramid
perturbation field. `from_energy_f32()` ranks the whole energy vector and keeps
the best eight; it does not establish a Morton `(x,y)` address map.

**[S]** `crates/thinking-engine/src/engine.rs` executes the runtime energy update
as a dense matrix-vector operation over the codebook-sized field. The source
calls the distance table precomputed and says it is the runtime thinking
kernel. The default codebook size is 4096.

**Important correction:** this is **not by itself evidence against the old
"without materialization" doctrine**. Historical PR #755 explicitly separated
an immaterialized PHASE plane from a materialized-or-analytic shared metric LUT.
The right audit question is therefore *what is being materialized*, not "is any
LUT present?".

### 1.2 Perturbation → shader seam

**[S][BW]** `crates/cognitive-shader-driver/src/engine_bridge.rs` documents and
implements two adjacent pipelines rather than one continuous ALU:

```text
thinking-engine: Φ StreamDto → Ψ PerturbationDto → B BusDto → Γ ThoughtStruct
shader-driver:   Φ ShaderDispatch → Ψ ShaderResonance → B ShaderBus → Γ ShaderCrystal
```

At the bridge:

- `StreamDto.codebook_indices` can seed BindSpace content fingerprints.
- `PerturbationDto.energy` does **not** cross into P64.
- `PerturbationDto.top_k` is filtered, then collapsed to `min(index)..max(index)`
  as a `ColumnWindow`.
- `{7,42,900}` therefore means "scan rows 7..901", not the sparse mask
  `{7,42,900}`.

This is the already-recorded #1051 broken wire, now re-confirmed against current
main rather than inherited from the PR prose.

### 1.3 P64 is live in the driver

**[S]** `crates/cognitive-shader-driver/Cargo.toml` directly depends on
`p64-bridge`, `bgz17`, `causal-edge`, and `ndarray`; its package description
states that it wires the contract DTOs, P64 `CognitiveShader`, and optional
thinking-engine.

**[S]** `crates/cognitive-shader-driver/src/driver.rs` constructs a live
`p64_bridge::cognitive_shader::CognitiveShader` every dispatch from a snapshot
of the eight predicate planes and the resident `PaletteSemiring`.

The hot path is therefore real:

```text
BindSpace row
   ↓ read row CausalEdge64.s_idx()
query palette index
   ↓
P64 8-plane mask
   ↓ 4×4 refinement
bgz17 metric lookup
   ↓
CascadeHit { target, distance, predicates }
```

### 1.4 What P64 actually means today

**[S]** `crates/p64-bridge/src/lib.rs` establishes:

- `edge_to_block`: `S/4 → row`, `O/4 → col`.
- eight predicate mask planes are built from CE64 causal/inference readings.
- `CognitiveShader::cascade(query, radius, layer_mask)` uses
  `query/4` as the block row, masks active block columns, expands each active
  block to four `target` archetype indices, and evaluates the precomputed
  semiring distance.
- the current P64 comments are explicit: mask = **WHERE**, metric = **HOW FAR**,
  composition table = **WHAT path composition means**.

This is already a useful form of the lithographic rule: topology selects the
population before the metric/algebra answers the question.

## 2. New decisive finding: the P64 target is dropped before CE64 emission

This is the smallest important finding from this pass.

**[S]** P64 returns `CascadeHit { target, distance, predicates }`, where
`target` is the concrete target archetype index `0..255`.

**[S]** In `ShaderDriver::run`, the source BindSpace row supplies
`query = backing.edge(row).s_idx()`. For every P64 `CascadeHit`, the driver then
creates:

```text
ShaderHit {
    row,               // ORIGINAL BindSpace source row
    distance,
    predicates,
    resonance,
    ...
}
```

The P64 `hit.target` is not copied into `ShaderHit`.

**[S]** Later CE64 emission derives:

```text
S = h.row % 256
P = 0
O = (h.row / 4) % 256
```

and packs that with resonance-derived `(f,c)` and the predicate bits.

Therefore:

**[BW] The target archetype selected by P64 does not survive into the emitted
CausalEdge64 identity.** The emitted S/O are reconstructed from the source
BindSpace row number instead.

This is stronger than the older "4096 == 4096 is not identity" warning because
it is a current producer→carrier→consumer identity loss visible inside one hot
path.

It is not yet labelled a bug: a synthetic "resonance-about-this-row" edge might
have intentionally discarded the target. But if the emitted edge is supposed to
be a replayable statement about the P64 relation that fired, target identity is
load-bearing and the present path cannot replay it.

### F-ARW-TARGET-1 — first falsifier

Construct one source BindSpace row whose P64 cascade returns two different
`target` archetypes A and B under the same predicate layer and equal distance.
Observe the emitted CE64s.

- If the two emitted edges remain distinguishable by relation identity, this
  finding is false and **NO BUY**.
- If they collapse to identical S/P/O identity (allowing only incidental
  resonance/cycle differences), target identity is proven lost.

**STOP:** do not mint a new carrier merely because this test fires. First audit
whether an existing byte/field in the already-frozen `ShaderHit` ABI is reserved
and legally usable, whether the CE64 can be emitted directly while the
`CascadeHit.target` is still in scope, or whether target identity belongs in an
existing witness/receipt rather than `ShaderHit`. Representation follows the
producer/consumer proof.

## 3. CE64 "vertical mantissa" lineage: what is recovered vs executable

**[S]** Current CE64 still spends the first 24 bits on S/P/O palette bytes.
`p64-bridge` consumes S and O as the coordinates of its 64×64 topology field.
This is a real surviving structural coupling.

**[HS]** merged #1051 records the historical interpretation that the original
4096 COCA surface, SPO `2³` readings, and CE64's 3×8 S/P/O bytes were intended
to form one amortized field, while also recording that the actual
`codebook_id ↔ (row,col)` map no longer exists in current code.

**[OR]** sharpened operator recall, 2026-08-30:

> CE64 was intended as the **vertical mantissa**, with SPO `2³` giving eight
> amortized readings while the Morton cascade supplied horizontal/local working
> geometry and cache economy.

Treat this wording as recovered intent until a pre-#1051 source spells out the
same "vertical mantissa" relation.

**[S][BW]** the live target-drop above is presently more decisive than debating
the historical wording: whatever the old mantissa meant, the current P64 target
cannot be reconstructed from the emitted CE64 path as written.

## 4. Morton cascade: real, but not the missing 4096 address proof

**[S]** current `AdaWorldAPI/OGAR/docs/DISCOVERY-MAP.md` records:

```text
64 → 256 → 1024 → 4096 → 16k → 64k → 256k
```

as an **immaterialized Morton enumeration**, one extra nibble per level, and
states that the cascade is a coordinate transform rather than a stored grid.

**[HS]** PR #755 tied MortonShift, inverse-pyramid perturbation, regenerated
phase and a shared metric LUT into:

```text
perturb(addr, L) = M[addr @ coarse] · P(phase(addr, L)) at loc(addr)
```

with two critical anti-collapse rules:

1. "without materialization" was scoped to the regenerated PHASE plane, not
   every shared metric artifact;
2. phase axis and palette-LUT axis are different coordinates.

**[BW]** none of that establishes the missing old mapping
`codebook_id[0..4095] ↔ (row,col)[0..63]²`. #1051 was right to leave that gate
open. Row-major, Morton deinterleave, and codebook permutation remain distinct
historical hypotheses.

**Current consequence:** do not restore the 4096 lexicon mapping simply because
4096 recurs. P64 currently has a defensible role as a ≤4096 **active relation
working surface** even if the old COCA address map is never recovered.

## 5. Stockfish: two oracle roles, neither is Morton proof

**[HS]** PR #679 explicitly grades the Stockfish-NNUE correspondence as a
synthesis/reference, not canon. The grounded parts are the resident accumulator,
deterministic feature addressing, SoA/incremental update discipline and the
byte-exact external oracle. Morton transfer remains probe-gated.

**[OR]** operator intent, re-stated 2026-08-30: `stockfish-rs` was meant to do
both jobs:

1. systems teacher — learn how a mature engine keeps a tiny hot geometry
   incremental and cache-efficient;
2. expert-iteration / reinforcement oracle — deep search teaches which choices
   are good, with held-out replay judging learned policy.

So Stockfish can teach *economy of cognition* without being evidence that chess
itself uses Morton.

## 6. EWA is present as a kernel/carrier, not proven live in the current shader pass

**[S]** `lance-graph-contract::sigma_propagation` contains the production
2×2 `Spd2` and `ewa_sandwich(M, Σ) = M·Σ·Mᵀ` kernel. Its module docs still name
shader-driver propagation as a planned consumer.

**[S]** `BindSpace::FingerprintColumns` has a resident `sigma: Box<[u8]>`, one
Σ-codebook index per row, with `sigma_at`/`write_sigma` accessors.

**[S, bounded]** the current `ShaderDriver::run` source examined in this audit
contains no `sigma` / `ewa_sandwich` consumption in its hot P64→CE64 path. This
is a bounded source fact about that function, **not a repo-wide absence claim**.

**[HS]** PR #289 is the direct historical source tying EWA-Sandwich to the
"cant-stop-thinking" loop:

```text
Frame[n+1] = J(edge_state) · Frame[n] · J(edge_state)ᵀ
```

with PSD-preservation measured by the jc pillar. PR #755 later kept geometry,
phase and metric as separate axes rather than collapsing them.

**Verdict:** EWA belongs in the lineage, but "the live P64 field currently
propagates EWA Σ" is **not established** by this audit.

## 7. There are two different Alpha mechanisms today

Do not unify these by name.

### 7.1 Shader AlphaComposite

**[S]** `cognitive-shader-driver::driver` has a live
`alpha_front_to_back_composite()` path selected by
`MergeMode::AlphaFrontToBack`. It is order-dependent front-to-back compositing
of hit qualia with saturation/early termination.

The current implementation pre-materializes a small
`Vec<(row, [f32;17])>` for hit qualia before compositing. That is a local
materialization exception, not the mask-native zero-copy ideal.

### 7.2 MedCare AlphaOverlay

**[S, cross-repo audit from #1078]** MedCare's `medcare-nodesoa::alpha` is a
thin-provisioned, same-address, discardable attention overlay over canonical
`NodeRow`, with `AlphaMask` algebra and per-claim `AlphaStamp`.

These are semantically different:

```text
AlphaComposite = how ranked hits are blended
AlphaOverlay   = where attention/residual landed on canonical addresses
```

**[H]** they may be complementary descendants of the old graphics/shader
intuition. No ancestry or shared carrier is assumed.

## 8. Lithography is real historical lineage

**[HS, cross-repo]** `AdaWorldAPI/ada-docs`, commit
`9a7fe16d29f288013af640b0683de2d9470d511f` (2026-01-22), added
`architecture/SHARED_LITHOGRAPHY.md` under the title
"Shared Lithography — One Brain, Two Hemispheres".

That document describes bighorn + agi-chat as two perspectives over one shared
lithography substrate, with a constantly-changing activation mask and XOR deltas
rather than duplicated full state.

This does not prove the current P64 implementation was mechanically descended
from that file, but it proves **"lithography" + shared substrate + activation
mask** was established project vocabulary before the current lance-graph-java
mask membrane.

**[MD]** current lance-graph-java independently makes the modern mechanical
version precise: the facade names `where()/hop()/out()/has()` intent, native
mask algebra is the execution currency, row materialization is an explicit
terminal, and Java does not become the row-wise executor.

The useful doctrine is therefore:

> **Lithography rule:** describe the exposure (`where` / mask) and the operation
> (`method` / atom); execute over resident substrate. Do not materialize a row
> population merely so a homunculus can iterate it.

This is a mnemonic, not a new ABI rule. The existing mask-native constitution
remains authoritative.

## 9. Current braid, without romanticizing it

Current main is not one clean descendant. It is a braided fossil record:

```text
thinking-engine
  dense field + metric MatVec
          │ top_k only
          ▼
engine_bridge
  sparse candidates collapse to ColumnWindow       [BROKEN FIELD→MASK WIRE]
          │
          ▼
ShaderDriver
  BindSpace source row → CE64.s_idx query
          │
          ▼
P64 mask → target archetypes → bgz17 metric
          │
          │ target dropped                           [NEW BROKEN IDENTITY WIRE]
          ▼
ShaderHit(source row, distance, predicates, resonance)
          │
          ├─ optional AlphaFrontToBack composite
          │
          ▼
CE64 reconstructed from source row
          │
          ▼
ShaderBus / sink / persistence
```

Alongside it:

- OGAR retains the immaterialized Morton cascade.
- EWA kernel + per-row Σ index exist, but this audit does not find them in the
  live `ShaderDriver::run` P64 pass.
- MedCare has a separate same-address Alpha attention overlay.
- loco/R2IL is the modern reusable-program direction from #1078 and should be
  preferred over adding more handwritten policy to this old driver.

## 10. Falsifier-first next actions

### BUY-0 — run F-ARW-TARGET-1 first

Why first: it is local, current, deterministic, and does not depend on recovering
a lost historical codebook mapping.

If target identity is intentionally disposable, document the reason and stop.
If it is load-bearing, preserve it using the smallest already-legal carrier or
local execution shape. Do not widen an ABI before proving no existing route can
carry it.

### BUY-1 — only after target identity

Re-run the older #1051 field-mask experiment on a **modern active set**, not the
obsolete 4096-token lexicon:

```text
CONTROL     active ids → current min/max ColumnWindow
EXPERIMENT  same active ids → exact mask → P64
SABOTAGE    permute active-id↔field placement
```

The experiment must hold work budget and evidence constant. If EXPERIMENT ≈
SABOTAGE on the locality-sensitive metric, the spatial/Morton interpretation is
decorative and gets NO BUY.

### BUY-2 — EWA only after field identity

If BUY-1 establishes a meaningful local field, test one shared typed 2×2 tensor
operator as geometry only. EWA/Σ may influence propagation/attention; it must not
mint evidence, trust, provenance, ReasoningBand or Rubicon authority.

## 11. Hard STOPs carried forward

- no `top_k == Morton cascade` claim without producer evidence;
- no `AlphaOverlay == PerturbationDto` claim without lineage + consumer proof;
- no `AlphaComposite == AlphaOverlay` collapse;
- no CE64 "vertical mantissa" promotion from recalled wording alone;
- no row-number-as-SPO identity if F-ARW-TARGET-1 shows target loss;
- no new carrier before existing reserved/local/receipt routes are exhausted;
- no EWA Σ → trust/evidence scalar;
- no Stockfish → Morton inference;
- no materialized lexicon merely to make 4096 aesthetically match P64;
- no bypass of Revision/Rubicon for persistent epistemic change.

## 12. D-ARW-0 verdict

**BUY the archaeology, not the restoration.**

The old field architecture is not imaginary: Morton cascade, perturbation
field, P64 mask, CE64 S/P/O coupling, EWA cant-stop-thinking math, lithography
vocabulary, Stockfish reference discipline and the shader driver all have real
source anchors.

But the current executable seam is visibly torn in two places:

1. dense perturbation energy becomes a min/max row window instead of a mask;
2. P64 target identity is discarded before CE64 emission.

The second is the smallest falsifier-first next action. Fix or explain that
before reconstructing any larger cognitive field.