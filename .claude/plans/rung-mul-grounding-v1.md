# rung-mul-grounding-v1 — the MUL fine-tuned into the ladder as an experience curve over the SPO 2³ NARS decomposition

**Status:** PROPOSAL (follow-on to `rung-ladder-grounding-v1`)
**Date:** 2026-05-26
**Confidence:** HIGH on the structure (it is the Dunning-Kruger curve mechanized — every strategy is the move that becomes *necessary* at one evidence level); MED on the per-projection `SpoHead` refactor (real change, D-RUNG-MUL-1); CONJECTURE on wisdom-marker calibration readout until D-RUNG-MUL-4 lands a test.
**Predecessors:** `rung-ladder-grounding-v1` (b0ef6fa), `E-AGICHAT-DIMENSION-CONTRACT` (afabefd), `cognitive-substrate-convergence-v1` (CausalEdge64 v2 §6: causal mask = Pearl 2³ IS the rung axis), `E-I4-META-1`.
**Grounded types (verified this session):** `nars_engine.rs::{SpoHead, all_projections, StyleVector}`; `FreeEnergy::compose` (cognitive-shader-driver/src/driver.rs:264, contract `grammar/free_energy.rs`); `mul/trust.rs::{TrustQualia, TrustTexture}`; `elevation/mod.rs::ElevationLevel` (L0:Point..L5:Async); `proprioception.rs` wonder axis (idx 9); planner `free_will_modifier`; holograph `NarsBudget{priority,durability,quality}`.

---

## 0. The correction this plan is built on

The SPO 2³ is **not a distance-cube** (popcount/Hamming-reach). It is the **powerset of {S,P,O}** — the 8 evidential projections you decompose a causal claim into and test separately through NARS. The essence is the *decomposition for causality testing*, not a metric. `nars_engine.rs` today computes `all_projections() -> [u32;8]` as **distances** (`causal_distance` sums active planes) and a `SpoHead` carries **one** truth — that is the cube-metric reading, and de-grounding it is D-RUNG-MUL-1.

```
___   ∅            base rate / prior — "nothing" (the null)
S__   marginal     does S occur at all?
_P_   marginal     does P occur at all?
__O   marginal     does O occur at all?
SP_   pair         S–P holds, O marginalized            (cause↔mechanism)
S_O   pair         S–O holds, P marginalized            (cause↔effect — maybe spurious)
_PO   pair         P–O holds, S marginalized            (mechanism↔effect)
SPO   full joint   the complete claim — does it cohere?
```

Causation = the **screening-off pattern** across the 8, not the joint alone: if `S_O` is strong but conditioning on P (`SP_` ∧ `_PO`) screens it off, the direct S→O is spurious/mediated. Every projection compared against `___` for lift over base rate. That is front-door/back-door logic expressed as NARS truth over the lattice.

## 1. The experience curve (the organizing skeleton)

Order every strategy by the evidence level at which it becomes **necessary**. The result is the Dunning-Kruger curve with a mechanical trigger at every point:

| # | Trigger (evidence state) | Necessary strategy | DK position | Confidence ceiling |
|---|---|---|---|---|
| 0 | **NaN** — no field, cannot even form a prior | cautious exploration → fanout → **request Lab** | pre-Mount-Stupid (unknown unknowns) | hard-floored ≈ 0 |
| 1 | sparse, NARS `c≈0`, expectation→0.5 | **gaussian splat over data field → `FreeEnergy::compose(likelihood, KL)`** | foot of curve | = the free-energy itself (high F ⇒ low conf) |
| 2 | one projection holds (e.g. `S_O`) | tempting to assert | **Mount Stupid** | DK-gate *penalizes* (high f, low c) |
| 3 | decompose 2³, run screening-off | analytical / counterfactual work | **Valley of Despair** (`S_O` screened off by P) | Boole/Fréchet — ≤ weakest link |
| 4 | decomposition coheres across cycles | exploit, expectation-gated | Slope of Enlightenment | graded, earned |
| 5 | bundle accumulated truth → **wisdom marker** | hydrate as prior *before the fact* | Plateau (φ-1 permanent-humility ceiling) | calibrated; will not re-inflate |

**Two curves over one axis.** Work (competence) climbs monotonically with decomposition; *confidence* is DK-shaped (spikes at Mount Stupid, craters in the Valley, recovers calibrated). The gap between them is the readout:

> **wisdom = the calibration gap closing — `|confidence − competence| → 0`.**

A hydrated wisdom marker starts the *next* situation already calibrated instead of at Mount Stupid, so the curve is a **spiral**: each pass seeds the prior (the KL anchor) for the next NaN/sparse encounter.

## 2. The work metric (exploit half)

Work is **decomposition + screening-off coverage**, expectation-gated, never popcount:

```
work(head) = Σ_{m ∈ 2³}  tested(m) · screening_weight(m) · expectation_m
             gated by:    budget.quality ≥ q_min            (AIKR resource gate)
```

- `tested(m)` = was projection `m`'s NARS truth actually derived (1) or skipped (0).
- `screening_weight(m)` — conditional-independence tests (the ones that distinguish causation from correlation) weigh most; raw marginals least; `___` is the base-rate reference.
- `expectation_m = c_m·(f_m − 0.5) + 0.5` — **confidence-gated, not frequency-gated.** Frequency alone is the Mount-Stupid signal (high f, low c = overconfidence on thin evidence). Confidence does the humility work; the deduction path already shrinks it (`deduced.c() < deduced.f()`), so deeper chains inherit more humility by construction.

Asserting `SPO` directly = zero work. Deriving `S_O` and stopping = correlation, little work. Discovering `S_O` is screened off = the real causal work *and* the humble result.

## 3. The humility / wisdom checks

- **Boole/Fréchet bound (hard invariant):** `conf(SPO) ≤ min` over the conjunction's decomposed parts. You cannot be more certain of the conjunction than of its weakest link. Checkable per-projection; this is the rigorous form of "confidence graded down the ladder."
- **Calibration gap (the wisdom readout):** `wisdom = 1 − |confidence − competence|`, where competence = `work` (§2) normalized. This is what a wisdom marker measures and persists.

## 4. The two sparse-data escalation routes

- **NaN route (no field at all).** NaN gates the **Exploratory** style at high `exploration_rate` / low `speed_bias` ("cautious exploration" — a *region* of `thinking/style.rs` param space, **not a named style**) → fan out → raise `ElevationLevel` (L_n→L_{n+1}) → **request a Lab** (sandbox to generate the missing evidence). Note: NARS "no evidence" is properly `c=0`/expectation=0.5, **not** NaN — so this introduces a *deliberate NaN sentinel* meaning "no field to splat over," distinct from `c=0` ("evidence absent but prior exists"). c=0 → ordinary escalation; NaN → cautious-exploration + Lab.
- **Splat route (sparse field exists).** No direct NARS evidence ⇒ the *only* sanctioned confidence is gaussian splat over the data field → `FreeEnergy::compose(likelihood, KL)`. The free energy itself caps confidence (high F ⇒ low conf), which is exactly what enforces **no data ⇏ inflated overconfidence**. The hydrated wisdom marker is the prior in the KL term.

The drive that *chooses* to escalate into untested projections is the explore complement of work: **`wonder` (proprioception idx 9) × `free_will_modifier` × trust**. Work without wonder never escalates; wonder without the Boole-bound just hallucinates new claims.

## 5. Deliverables

| D-id | title | crate | ~LOC | risk |
|---|---|---|---|---|
| D-RUNG-MUL-1 | per-projection NARS truth — `SpoHead` carries the 2³ as 8 `(f,c)` (not 1); `all_projections` returns truths, not just distances | lance-graph-planner (nars_engine) | 220 | MED |
| D-RUNG-MUL-2 | NaN sentinel gate → Exploratory(high exploration_rate, low speed) + `ElevationLevel`↑ + Lab-request signal; distinct from `c=0` path | planner (mul + elevation) | 160 | MED |
| D-RUNG-MUL-3 | wisdom marker — VSA-**identity** bundle (≤32 per I-VSA-IDENTITIES; truths in content store, not in the bundle) + `hydrate` before cycle as the KL prior | contract + planner | 180 | MED |
| D-RUNG-MUL-4 | screening-off `work` metric + Boole-bound + calibration-gap (`wisdom = 1−|conf−competence|`) readout | planner (mul) | 150 | MED |
| D-RUNG-MUL-5 | splat→`FreeEnergy::compose` path as the sole confidence source under sparse data; F caps confidence | planner + cognitive-shader-driver | 120 | MED |

Tests: spurious-correlation case (`S_O` strong, screened off by P → work credits the screening-off, confidence on direct S→O drops); Boole-bound violation rejected; NaN→cautious-exploration vs `c=0`→ordinary-escalation distinction; sparse-field confidence never exceeds `FreeEnergy`-derived ceiling; calibration gap shrinks across simulated experience.

## 6. Reconciliation (don't fork)

- **`elevation/homeostasis.rs` is already MUL-L6.** The experience curve's escalation (steps 0,1,3) folds in beside it and beside `evaluate_rung_shift` from rung-ladder-grounding-v1 — the rung ladder is the coarse integer escalation; the MUL curve is its graded trigger source (DK-position → which rung-shift fires). Co-finetune; do not add a parallel module.
- **Pearl 2³ already IS the rung axis** per CausalEdge64 v2 §6 (`[40:42] causal mask, counterfactual at 0b111 SPO`). The 8 projections of this plan are that mask's powerset reading — extend the existing field's semantics, don't invent a new one.
- **No wisdom qualia type exists today** (`wisdom` is only aphorism strings in `high_heel.rs`) — D-RUNG-MUL-3 is net-new and should land as a marker over existing `QualiaColumn`/proprioception axes, not a new struct (AGI-as-SoA: new capability = new column, not new layer).

## 7. Invariants honored

- **Confidence-gated, never frequency-gated** (frequency alone = Mount Stupid). - **Boole/Fréchet bound** on conjunction confidence. - **no data ⇏ overconfidence** — only `FreeEnergy` (splat route) or floored NaN (Lab route) may produce a signal. - **I-VSA-IDENTITIES**: wisdom markers bundle ≤32 identities; truths live in the content store. - **AIKR**: `budget.quality` gate caps fanout (no syntactic explosion). - **AGI-as-SoA**: markers = column reads/writes, not a new service. - decomposition (powerset), not distance-cube (popcount).

## 8. Cross-refs

`rung-ladder-grounding-v1` (the coarse integer ladder this grades); CausalEdge64 v2 §6 (Pearl 2³ = rung axis); `nars_engine.rs::{SpoHead, all_projections, StyleVector}`; `FreeEnergy::compose`; `mul/{trust,homeostasis,gate}.rs`; `elevation/mod.rs::ElevationLevel`; `proprioception.rs` wonder axis; `NarsBudget`. Iron rules: `I-VSA-IDENTITIES` (bundle identities ≤32, content in store), `I-NOISE-FLOOR-JIRAK` (significance on bit-exact lanes), data-flow (no `&mut` during compute). Lab surface: `lab-vs-canonical-surface.md` (the Lab request routes through the canonical bridge, not a new endpoint).
