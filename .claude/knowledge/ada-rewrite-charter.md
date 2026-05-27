# Ada Rewrite Charter — the once-and-for-all decisions

> **READ BY:** everyone. This is the settled-decision record for the rewrite. If a later doc contradicts this, this wins unless explicitly superseded by a dated entry.
> **Date:** 2026-05-27.

## D0 — We rewrite. ladybug-rs has no relation and never will.

`ladybug-rs` ran 10,000 × 10,000-D VSA and produced no meaningful output — the "empty cathedral" (E-AGICHAT-DIMENSION-CONTRACT). It is **NOT a dependency, NOT a port target, NOT consumed**. Its docs (the 34-tactics × reasoning-ladder map, the SPOQ audit) are **specification references only** — they tell us *what each capability must do and which science grounds it*, never *what code to copy*. Same for ada-consciousness and neo4j-rs. **Iron rule: restore the contract on our i4 SoA floor; never port the carrier.** "Use the ladybug implementations" = use them as the spec to write our own working recipes.

## D1 — The substrate (settled)

- **atoms → cognitive-shader-driver → SIMD.** Atoms are cognitive units (bare-metal, not human-legible); they dispatch through `cognitive-shader-driver`, which owns the ndarray i4 SIMD. No SIMD in the atom layer.
- **Three layers:** atom (one lane/pole) → thinking-style (one i4 vector = molecule) → persona (composition of styles + thresholds + β). The OO style/persona objects are the metacognition; atoms are the bytes.
- **The lattice is SPOQ.** SPO 2³ = the *causal* slice (8 projections, Counterfactual=`SPO`=0b111, Intervention=`_PO`). **Q (Qualia) is the 4th role** — affective overlay, orthogonal to causality. Causal reasoning rides 2³; qualia rides Q.
- **Business = OGIT-inherited sidecar**, not an atom (front-door `MappingRow` → `Marking`).
- **Markers gate implicitly** (CPU-style clock-gating): entropy = CollapseGate `SD` (FLOW<0.15/HOLD/BLOCK>0.35), free-energy = the rest-floor, rung R1–R9 = pipeline depth, temperature (Staunen↔Wisdom) = speculation width, dissonance = counterfactual-fork gate. Markers are free byproducts; expensive units stay dark by default.

## D2 — The hardware partition (where work lives)

| bucket | = CPU | holds |
|---|---|---|
| **datapath** | vector ALU | uniform branch-free SIMD: atom lanes, FreeEnergy, resonance (cosine sweep) — in `cognitive-shader-driver` |
| **control** | microcode | branchy decisions: quorum/InnerCouncil, counterfactual fork, persona dispatch, OGIT lookup — in planner + `contract::escalation` |
| **gate** | clock/power-gating | the markers above decide *whether* control/datapath fire — in `elevation/` + CollapseGate SD |

## D3 — The 34 tactics are the recipe targets

The 34 LLM tactics (ladybug spec) reduce to **three mechanisms** = the partition:
1. **Parallel-Independence** (breaks Tier-2 `P=p^n` error) → datapath/redundancy. #1,2,5,20,26,30.
2. **Truth-Aware-Inference** (NARS truth per step, revision, abduction, CollapseGate HOLD, Brier) → control. #3,7,10,11,17,21,28.
3. **Structural-Divergence** (12 styles can't converge; counterfactual XOR world; Granger; reversible fusion) → gate+control. #4,6,9,13,23,31,34.

A **recipe** = a named composition over our substrate that realizes a tactic. We write them as working, tested code (`contract::recipes` = the catalogue spine; per-recipe evaluators land as substrate readiness allows). Recipes compose *our* primitives (atoms, SPO 2³ masks, NARS truth, CollapseGate SD, markers) — never call ladybug.

## D4 — Build order

1. ✅ Atom catalogue (locked 33-TSV), markers, CollapseGate SD, NARS, escalation — exist.
2. **▶ Recipe catalogue** (`contract::recipes`, the 34 as working data + registry + tests) — THIS deliverable.
3. Per-recipe evaluators, tier by tier (Hard-tier truth/parallel recipes first — their substrate is most built).
4. cognitive-shader-driver carrier wiring (atom pack/unpack), then datapath recipes.

**Cross-ref:** `agi-stack-cross-repo.md` (the spec sources), `spo-2cubed-list-coverage.md` (2³ coverage + the 34 table), `atom-basis-inventory.md` (33-TSV), `EPIPHANIES.md` E-AGICHAT-DIMENSION-CONTRACT (the empty-cathedral lineage).
