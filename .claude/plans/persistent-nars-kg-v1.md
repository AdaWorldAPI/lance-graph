# Persistent NARS Knowledge Graph — Integration Plan v1

> **Status:** ACTIVE (2026-07-19). Author: main-thread synthesis over three
> parallel specialist mappers (truth-architect, trajectory-cartographer,
> general-purpose recipes/atoms). **Probe-first**: two gates (M-RUNG-1,
> P-PERSIST-1) precede their dependent wiring.
> **Operator directive:** jan@exo.red "escape hatches" message, 2026-07-19.

## 0. The operator directive (the "escape hatches")

1. Multipass reading with NARS confidence + episodic-witness vs Leiden
   communities/basins; **capitalized-within-sentence + histogram = a naming
   signal**.
2. The **12×12=144 verb table** as verb meaning families / semantic basins
   (part_of / is_a).
3. The 4096 may need a **12×12=144 verbs × SPO-position** map for ambiguity
   resolution.
4. **NARS reasoning MUST be stored** — not from zero every pass, even when a
   branched ambiguity counterfactual hits a conclusive dead end.
5. The **34 (or 36) NARS strategies** are documented recipes — program once as
   higher-order thinking atoms, invoke on escalation.
6. **Build the knowledge graph — that's what the SoA is for.**
7. Check **`lance-graph-arm-discovery`**.
8. The **DeepNSM grammar heuristics** — parse yourself, add the missing
   heuristic templates.
9. When stuck on edge cases, use the **MUL (Dunning-Kruger overconfidence vs
   trust)** as the signal.
10. **SPO 2³ = NARS reasoning rung decomposition + causality trajectory
    candidates.**
11. The possible/impossible resolution of candidates = **grammatical game
    theory with causality edges**.

## 1. The one-line finding

**Everything is designed; almost nothing is wired into a loop.** Every
reasoning atom (34 recipes, 7 truth functions, the 2³ mask, MUL, the verb
table) and every persistence lane (the 512-byte node's tenants) already
EXISTS. The program is **wiring + persistence + two measurements**, not
building substrate. Three claims over-reach the code and are marked CONJECTURE
below.

## 2. EXISTS / MISSING / SMALLEST-WIRE (the map — file:line sourced)

### A. Reasoning atoms
| Piece | Status | Where |
|---|---|---|
| 34 NARS recipes as atoms | **EXISTS/IMPL** — `RECIPES:[Recipe;34]` + 34 `Tactic` kernels + palette `AtomId` | `contract::recipes.rs:87`, `recipe_kernels.rs:923`, `cognitive_palette.rs:80` |
| F→34→F selector + loop | **EXISTS** but **observe-only** (provenance, "never alters gate") | `contract::materialize.rs:77,176`; `driver.rs:633` |
| 7 NARS truth functions | **EXISTS** inline f32 (2 tabled: deduction 128KB, revision ≤32MB) | `ndarray/src/hpc/nars.rs:322-448`; `causal-edge/tables.rs:41` |
| 2³ `CausalMask` | **EXISTS** as a *projection selector* (SO=R1, PO=R2, SPO=R3, SP=Simpson's) | `causal-edge/pearl.rs:28-49`; consumers `edge.rs:597,626` |
| "game theory" gate + scorer | **EXISTS** — `syllogism.rs::figure()` (possible/impossible) + `TruthPropagatingSemiring` (⊗=ded, ⊕=rev) | `causal-edge/syllogism.rs:138`; `planner/physical/accumulate.rs:143` |
| MUL stuck→escalate signal | **EXISTS** but **TWO divergent type-sets** (contract vs planner) | `contract::mul.rs`; `planner/mul/*` |
| escalation ↔ MUL | **EXISTS** (`verdict_from`) but **test-only caller** | `planner/mul/escalation.rs:21` |

### B. Persistence lanes (the 512-byte node)
| Operator field | Byte-locked carrier | Status |
|---|---|---|
| SPO identity | `SpoFacet` 6×(8:8) rails 0-2 | EXISTS (lane) |
| NARS confidence | `ValueTenant::Meta` → `MetaWord.nars_f`/`nars_c` bit-slots | EXISTS (lane), **unpopulated** |
| Episodic-witness | `SpoFacet` rails 3-5 + `WitnessTable<64>` W-slot | EXISTS (lane), **zeroed** |
| Leiden basin | key `family` (bytes 10-13) + `part_of:is_a` cascade tail | EXISTS |
| zero-copy write | `NodeRowPacket` (a `SoaEnvelope`) → Lance | EXISTS, **not called** |
| Cross-pass read | — | **MISSING** (`infer_deductions(&self)` writes nothing back — `triplet_graph.rs:830`) |

### C. The genuinely-MISSING structures
- **verbs × SPO-position** map (#3) — the verb table is verbs×**tense**→TEKAMOLO
  (`verb_table.rs:186`, 12 `VerbFamily`×12 `Tense`), NOT ×{S,P,O}. Two divergent
  144-tables exist (`sigma_rosetta.rs` disagrees; ladder-doc O7).
- **verb families as part_of/is_a basins** (#2) — `VerbFamily` is a flat 12-enum.
- **escalation → recipe dispatch edge** (#5,#9) — Half A (MUL council →
  `CollapseHint`) and Half B (`select_tactic` → `materialize`) are unconnected.
- **2³ → rung-candidate generator** (#10) — no iterator over the mask lattice.

## 3. The two probe gates (measurement before wiring)

### M-RUNG-1 (gates all reasoning wiring — W-B → W-D)
**Claim:** projecting one edge through `{SO=R1, PO=R2, SPO=R3}` yields
*measurably different, individually useful* rung candidates.
**File:** `crates/lance-graph/examples/probe_rung_fan.rs` (sibling of
`scorpion_frog_counterfactual.rs`).
**Pass:** ≥1 corpus pair where R2 (`PO`, confounder-projected) reverses the R1
(`SO`) `causal_distance` ranking (Simpson's-style — `pearl.rs:102` already
screens `simpsons_paradox_risk`).
**Fail:** all three rungs share the top-k ordering → the mask is decorative
here → drop the rung fan, keep pairwise `syllogism` only.
**Cost:** ~60 LOC; reuses `NarsTables`, `causal_distance`, `intervene_on`.

### P-PERSIST-1 (gates persistence claims — W-C)
**Benchmark:** run `text_stream_to_soa` **twice on the same text**. Pass 2 must
show: (a) re-derived deductions = 0 (only genuinely-new-text deductions — vs
95,410 recomputed today); (b) a re-observed fact's confidence strictly
increases (`TruthValue::revision`); (c) pass-1 dead-end counterfactual branches
are NOT re-expanded (skip count = #(−6-marked edges)); (d) the reasoning set is
read from the store in < recompute wall-time.

## 4. Waves

| Wave | Deliverable | Gate | Escape hatch |
|---|---|---|---|
| **W-A** ✅ | deepnsm **naming heuristic**: preserve case (`split_words`), OOV-restricted named-entity detection for mid-sentence capitals (recovers `Napoleon`/`Snowball`; ordinary capitalized inflections `Dogs`→`dog` resolve normally, nothing dropped — Codex P2 r3610093782), `parser::named_entities()`; example `names` readout | — | #1, #8 |
| **W-A.1** | **histogram override** (the "+ histogram" half): a corpus-level consumer flags a surface that is capitalized-mid-sentence AND never appears lowercase as a named entity even when it resolved to a rank — catches inflection-collision names (`Jean`→`jeans`, `Boxer`→`box`) that W-A cannot separate at the token level | W-A | #1 |
| **W-B** | Probe **M-RUNG-1**: `CausalEdge64::rung_candidates() → [_;3]` + Simpson's-reversal test | — | #10 |
| **W-C** ✅ (in-process) | **self-correction logic**: re-reading revises in place (`revise_with_evidence`) instead of `new()` from zero — confidence accumulates `0.500→0.667→0.750`, 0 new triples after pass 1 (`examples/self_correcting_kg.rs`, `E-SELF-CORRECTING-KG-1`). The P-PERSIST-1 in-process cut. | — | #4, #6 |
| **W-C.2** | **durable persistence**: materialize committed `Triplet`→`NodeRow` (Meta tenant `nars_f`/`nars_c` + `family`=Leiden basin) via `NodeRowPacket`→Lance, hydrate on entry — so the self-correction survives across process runs, not just passes | W-C | #4, #6 |
| **W-D** | **escalation→recipe edge**: chain `verdict_from(&MulAssessment)` → `select_tactic`/`materialize`; wire the rung fan | M-RUNG-1 pass + two-MUL reconcile | #5, #9, #10, #11 |
| **W-E** | **dead-end counterfactual memory**: add `ContextTag::Contradiction`; `revise_with_evidence` retains both poles stamped −6; read-on-entry skips explored dead ends | W-C | #4 |
| **W-F** | **verbs × SPO-position** map (`[[SlotPrior;3];12]`) + `part_of/is_a` over `VerbFamily`; wire `verb_table` into the live `grammar::disambiguator` | reconcile the two 144-tables (O7) | #2, #3 |
| **W-G** | named entities → **graph nodes**: connect `parser::named_entities()` to the string-keyed `TripletGraph` + a `NodeRow` (register identity, never a 12-bit rank) | W-A, W-C | #1, #6 |

## 5. Frankenstein guards (composition risks — from truth-architect §Frankenstein)

- **Two MUL type-sets do NOT compose** (contract `Plateau`/4-`TrustTexture`/
  `Transition`/`GateDecision` vs planner `PlateauOfMastery`/5-`TrustTexture`/
  `Apathy`/`MulGateDecision`). Bridge by **scalar signals**
  (`InnerCouncil::from_signals(f32×4)`), **never add a third MUL**.
- **7 truth functions live in 4 places** (ndarray + NarsTables + syllogism.rs +
  edge.rs). Any wire calls **one**, never a fifth copy.
- **The 2³ mask cannot carry rung content** (64-bit budget). A "trajectory
  candidate" is an edge+mask plus out-of-band args (palette indices / ContextTag
  G-slot), not richer edge bits.
- **"36" is a decoy** — `ThinkingStyle` (36 persona adjectives) is OFF the
  reasoning ladder ("carried, not acted on"). The recipes are **34**. Do not
  decompose the 2³ into the 36.
- **Named entities use the register** (surface string / HashMap), never forced
  into the 12-bit SPO rank (I-VSA-IDENTITIES Test 0).
- **Persistence target = `NodeRow` SoA (`NodeRowPacket`→Lance) + `temporal.rs`
  stream**, NOT the dev-only in-memory `SpoStore` (`store.rs:4-11`).

## 6. arm-discovery (#7) — what it is

`lance-graph-arm-discovery` = the **proposer / discovery leg** (a float-free
Aerial+ association-rule miner → NARS-truth SPO candidates via
`arm_to_truth_u8`). It **produces** candidate facts; it does NOT persist/read
reasoning — a **separate** concern from W-C. Its unbuilt Stage C
(`streaming-arm-nars-discovery-v1.md §2.3:219-257`) is the closest existing
*spec* for the read-prior-then-revise loop; reuse its Stage-C shape in W-C, but
do not treat the crate as the loop.

## 7. Cross-refs
- `EPIPHANIES.md` E-REASONING-CORE-GROUND-TRUTH-1 (the 2³/NARS/counterfactual
  spine), E-CODEBOOK-OOV-SURFACE-FIDELITY-1 (the naming-collapse finding W-A
  fixes), E-MARKOV-TEMPORAL-STREAM-1 (episodic = Lance versions).
- `.claude/v3/soa_layout/{le-contract.md, tenants.md}` (tenant offsets).
- `.claude/v3/knowledge/persona-vs-rung-ladder.md` (rung 2 = 144 verbs, rung 3 =
  34 recipes; the 34-vs-36 demarcation).
