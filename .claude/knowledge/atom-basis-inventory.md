# Atom Basis Inventory — D-ATOM-0 (the LOCKED 33-dim ThinkingStyleVector)

> **READ BY:** D-ATOM-1 (`contract::atoms` → `ThinkingStyleI4_32D`), D-ATOM-2, D-ATOM-3; `truth-architect`.
> **Status:** D-ATOM-0 resolution. Source = **`E-AGICHAT-DIMENSION-CONTRACT`** — agichat's **LOCKED** 33-dim TSV (`CANONICAL_DIMENSION_ALLOCATION.md`, "Status: LOCKED"). **Supersedes the earlier qualia draft AND the callcenter-32 draft — both were the wrong source.** This is the rung-ladder the atoms live in.
> **Date:** 2026-05-27.

## Smallest → largest (do not confuse the layers)

- **atom** = the **smallest unit = one pole** (e.g. `deduce`, `induce`, `R5`, `Φ`, `exploration`). 64 atoms = 32 lanes × ±. An atom is **not** a group and **not** the vector.
- **thinking style** = **one i4-32D vector** — a weighting across all the atoms (2^128 possible). Kant / Schopenhauer = specific vectors. **This is the molecule.**
- **persona** = a composition of styles + thresholds + purpose + β.

`ThinkingStyleI4_32D` is the *type that holds a style* (a molecule); its **individual lanes/poles are the atoms**. The groups below (Pearl/Rung/Σ/Ops/Presence/Meta) are **allocation families** — neither atoms nor molecules, just how the lanes are budgeted.

**Atoms are bare-metal by design — not human-legible.** A single i4 pole means nothing to a human; that's fine. Atoms exist to be *composed into object-oriented metacognition*: a **style is an object** (a `StyleRecipe` carrying behavior over its atom-weighting), a **persona is an object** (a `PersonaRecipe` composing styles + β/thresholds). The metacognition you reason about is Kant-the-object / the-OSINT-persona-object — never "pole #37". This is the workspace's "thinking is a struct / the object speaks for itself" doctrine; the **objects are the cognition**.

**Execution stack: `atoms → cognitive-shader-driver → SIMD`.** Atoms are NOT SIMD. Atoms **dispatch through `cognitive-shader-driver`** (the encode/decode engine that sweeps the SoA columns); the **shader-driver** is the layer that uses SIMD (ndarray i4-32). So **D-ATOM-1 defines the atoms and routes them into the shader-driver — no SIMD dot in the atom layer** (that's the driver's job, largely already built); **D-ATOM-2 builds the OO layer (the actual metacognition)** as objects dispatched through the same driver.

## The basis is LOCKED, not derived

`ThinkingStyleI4_32D` = **i4 × 33** (32 + 1 spare), riding the shipped ndarray i4-32 unpack (`E-I4-META-1`, `8de1dcf8`). The allocation IS the contract — `CANONICAL_DIMENSION_ALLOCATION.md` rejects arbitrary dim moves. Do not re-derive; record and classify.

| group | n | dims | kind |
|---|---|---|---|
| **Pearl** | 3 | SEE (association) / DO (intervention) / IMAGINE (counterfactual) | ordinal causal ladder |
| **Rung** | 9 | R1–R9 (meaning-depth) | ordinal depth ladder 🪜 |
| **Sigma** | 5 | Ω / Δ / Φ / Θ / Λ (σ-tier chain) | ordinal tier sequence |
| **Operations** | 8 | abduct / deduce / induce / synthesize / preflight / escalate / transcend / model_other | operations (one inference ± pair inside) |
| **Presence** | 4 | authentic / performance / protective / absent | modes |
| **Meta** | 4 | confidence_threshold / preflight_depth / exploration / verbosity | scalar knobs |

= **33.** Qualia (`QualiaI4_16D`, 16D packed from the 18D PCS) is a **separate vector**, NOT part of the TSV — that's why qualia was the wrong source.

**Business is NOT an atom — it is a sidecar inherited from OGIT.** No business/FIBU dims in the TSV. Business context rides in via the front-door OGIT class resolution (`E-OGIT-STAKES-LINCHPIN`): the request's OWL/DOLCE class → O(1) `lance_graph_ontology::MappingRow` → `Marking::Financial` → the bookkeeping savant + `RuleBundle`/SKR04 (`E-FIBU-GOBD-BY-CONSTRUCTION`). It is inherited per-request on the existing inherit-set (marking→stakes, thinking_style→savant, …), not hand-authored and not a pole.

## Dichotomy-bounded? (first pass — confirm)

Most TSV dims are **not ± dichotomies** — they are ordinal ladders / scalar knobs / distinct ops:

- **Pearl (3), Rung (9), Sigma (5) = 17 dims → ordinal LEVELS.** A magnitude along a ladder, readable as −pole = low end ↔ +pole = high end (association↔counterfactual; shallow↔deep R1↔R9; Ω↔Λ). Bipolar *as endpoints*, not opposite operations.
- **Operations (8):** mostly distinct operations (selectors). The one clean ± pair is **deduce ↔ induce** (top-down ↔ bottom-up); **abduct** is the third inference mode — this is exactly the "abduction–induction / deduction–induction" hint, and it lives *inside* the 8 Ops, not as a separate triad. synthesize / preflight / escalate / transcend / model_other = distinct unipolar ops.
- **Presence (4):** **authentic ↔ performance** is a ± pair; protective / absent = modes (or present↔absent).
- **Meta (4):** scalar knobs. **exploration = the explore↔exploit / temperature axis** (✅ bipolar). confidence_threshold / preflight_depth / verbosity = magnitudes.

**Genuine ± dichotomies are few:** deduce↔induce, authentic↔performance, exploration. The bulk are **ordinal levels or scalar knobs** → i4 as signed-position or unsigned-magnitude, not opposite-poles.

## Resolved by this source

- **Encoding:** i4 × 33 on the shipped i4-32 unpack. Ordinal ladders → magnitude lanes; the few ± pairs → signed lanes.
- **abduction:** it is 1 of the 8 Operations (with deduce/induce) — not a separate triad to pair off.
- **"not NARS / orthogonal":** the 6 groups (Pearl/Rung/Sigma/Ops/Presence/Meta) are the orthogonal dimension-groups; NARS-inference = ~3 of the 8 Ops. Confirmed.

## OPEN

1. **The 33rd/spare dims:** `STYLE_ENCODING.md` says "3 Pearl + 9 Rung + 5 Σ + 8 Op + **8 spare**" (= 33); the contract body names the last 8 as **4 Presence + 4 Meta**. Confirm which is canonical (spare vs Presence+Meta).
2. Confirm the ± reads (deduce↔induce, authentic↔performance, exploration) vs treating Ops/Presence as pure selectors.
3. Per-lane i4 sign convention for the ordinal ladders (0=R1 ascending, or centered).
