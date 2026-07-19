# The 34 NARS reasoning recipes — when they fire, where they wire

> **Updated:** 2026-07-19. **Source of truth:** `lance_graph_contract::recipes`
> (`RECIPES: [Recipe; 34]`) + `recipe_kernels` (the 34 `Tactic` kernels) +
> `materialize` (the F→34→F loop). Every **value** below is measured by
> `cargo run -p lance-graph-contract --example recipe_dispatch_map` — regenerate
> there to verify; **never hand-edit the numbers**. The tables are then
> hand-formatted for readability (the identical coherent/contradicted rows the
> generator prints separately are merged into one `coh|contra` row, and the
> generator's empty `requires()` output is annotated `*(empty)*`), so a
> byte-diff against raw stdout will differ in layout, never in the values.
>
> **Ladder position (operator-ruled):** these 34 are **rung 3** of the
> rung-content ladder — THE runbooks, executable inference recipes
> (`.claude/v3/knowledge/persona-vs-rung-ladder.md`). Anti-conflation: they are
> NOT the adjective-36 in `contract::thinking` (persona-modeling storyline,
> off-ladder, unwired) and NOT ada_10k's "36 NARS thinking styles as VSA
> dimensions" (`.claude/knowledge/34-tactics-vs-ada.md`, external Ada substrate
> mapping). Three different spaces share the word "style"; only THIS one is the
> reasoning runbooks.

## §1 The catalogue (id · code · name · tier · mechanism · bucket · SPO-2³ · checklist · substrate)

Spec source: ladybug `34_TACTICS_x_REASONING_LADDER` + Sun et al. (2025) —
spec only, never a dependency (ada-rewrite-charter D0). `reads` is the kernel's
**declared input checklist** (`Tactic::requires()` over the eight
`ThoughtField`s: sd, F=free_energy, diss, temp, conf, rung, cand=candidates,
belief=beliefs) — reliability = checklist coverage.

| id | code | name | tier | mechanism | bucket | 2³ | reads | substrate (charter D3) |
|---:|------|------|------|-----------|--------|----|-------|------------------------|
| 1 | RTE | Recursive Thought Expansion | Hard | ParallelIndependence | Control | — | F+rung | rung depth × Expand/Compress; Berry-Esseen stop |
| 2 | HTD | Hierarchical Thought Decomposition | Hard | ParallelIndependence | Control | — | cand | CLAM bipolar split / Decompose op |
| 3 | SMAD | Structured Multi-Agent Debate | XHard | TruthAwareInference | Control | — | cand | a2a_blackboard + InnerCouncil (NARS-revised vote) |
| 4 | RCR | Reverse Causality Reasoning | XHard | StructuralDivergence | Control | **Covered** | cand | SPO 2³ backward S_O + Abduction + Granger |
| 5 | TCP | Thought Chain Pruning | Hard | ParallelIndependence | Gate | — | sd+cand | CollapseGate SD BLOCK prunes branch |
| 6 | TR | Thought Randomization | XHard | StructuralDivergence | Gate | — | temp+cand | temperature (Staunen) perturb above noise floor |
| 7 | ASC | Adversarial Self-Critique | XHard | TruthAwareInference | Control | Partial | conf | InnerCouncil split / 5 challenge types |
| 8 | CAS | Conditional Abstraction Scaling | Cross | Infrastructure | Gate | — | rung | HDR cascade INT1/4/8/32 × Abstract↔Concretize |
| 9 | IRS | Iterative Roleplay Synthesis | XHard | StructuralDivergence | Control | — | temp+cand | persona FieldModulation (distinct kernels) |
| 10 | MCP | Meta-Cognition Prompting | Hard | TruthAwareInference | Control | — | F+conf | MUL DK + Brier calibration; Meta lane |
| 11 | CR | Contradiction Resolution | Hard | TruthAwareInference | Control | Partial | belief | NARS opposing-truth detect; Contradiction preserved |
| 12 | TCA | Temporal Context Augmentation | Cross | Infrastructure | Datapath | — | cand | Granger temporal lane / Markov ±5 / 24 temporal verbs |
| 13 | CDT | Convergent & Divergent Thinking | XHard | StructuralDivergence | Gate | — | temp+cand | explore↔exploit temperature; style oscillation |
| 14 | MCT | Multimodal Chain-of-Thought | Cross | Infrastructure | Datapath | — | cand | GrammarTriangle: NSM+Causality+Qualia → one fp |
| 15 | LSI | Latent Space Introspection | Cross | Infrastructure | Control | — | sd+cand | CRP distribution / Mexican-hat over fp clusters |
| 16 | PSO | Prompt Scaffold Optimization | Cross | Infrastructure | Control | — | cand | ThinkingTemplate slots + TD-learned discovery |
| 17 | CDI | Cognitive Dissonance Induction | Cross | TruthAwareInference | Control | Partial | diss+belief | Festinger dissonance = opposing NARS truth; HOLD |
| 18 | CWS | Context Window Simulation | Cross | Infrastructure | Control | — | conf+cand+belief | persistent BindSpace / WitnessCorpus / episodic |
| 19 | ARE | Algorithmic Reverse Engineering | Cross | Infrastructure | Datapath | — | *(empty)* | ABBA unbind: A⊗B⊗B=A (exact algebraic inverse) |
| 20 | TCF | Thought Cascade Filtering | Hard | ParallelIndependence | Gate | — | cand | N search strategies + agreement rate; SD select |
| 21 | SSR | Self-Skepticism Reinforcement | Hard | TruthAwareInference | Control | Partial | F+conf | challenge schedule × MUL uncertainty |
| 22 | ETD | Emergent Task Decomposition | Cross | Infrastructure | Control | — | cand | CLAM cluster geometry determines subtasks |
| 23 | AMP | Adaptive Meta-Prompting | XHard | StructuralDivergence | Control | — | F+rung | TD-learning on ThinkingStyle Q-values (W32-39) |
| 24 | ZCF | Zero-Shot Concept Fusion | Cross | Infrastructure | Datapath | — | *(empty)* | VSA bind(A,B): valid in both spaces, recoverable |
| 25 | HPM | Hyperdimensional Pattern Matching | Cross | Infrastructure | Datapath | — | cand | fingerprint cosine/Hamming sweep (SIMD) |
| 26 | CUR | Cascading Uncertainty Reduction | Hard | ParallelIndependence | Gate | — | cand | FreeEnergy / CRP percentiles; coarse-to-fine prune |
| 27 | MPC | Multi-Perspective Compression | Cross | Infrastructure | Datapath | — | cand | bundle = majority-vote-per-bit consensus + delta |
| 28 | SSAM | Self-Supervised Analogical Mapping | XHard | StructuralDivergence | Datapath | Partial | sd | NARS analogy A→B,C≈A⊢C→B; bind+similarity |
| 29 | IDR | Intent-Driven Reframing | Cross | Infrastructure | Control | — | cand | GrammarTriangle CausalityFlow agent/action/patient |
| 30 | SPP | Shadow Parallel Processing | Hard | ParallelIndependence | Control | Partial | cand | independent paths + agreement (ECC/RAID); CF fork |
| 31 | ICR | Iterative Counterfactual Reasoning | XHard | StructuralDivergence | Control | **Covered** | *(empty)* | world⊗factual⊗counterfactual; SPO=0b111; −6 mantissa |
| 32 | SDD | Semantic Distortion Detection | Cross | Infrastructure | Datapath | — | cand | Berry-Esseen noise floor + reciprocal validation |
| 33 | DTMF | Dynamic Task Meta-Framing | Cross | Infrastructure | Control | — | sd+temp | template switch on CollapseGate BLOCK |
| 34 | HKF | Hyperdimensional Knowledge Fusion | XHard | StructuralDivergence | Datapath | — | *(empty)* | cross-domain bind(A,rel,B); reversible fusion |

Mechanism tally (test-pinned): ParallelIndependence 6 (#1,2,5,20,26,30) ·
TruthAwareInference 6 (#3,7,10,11,17,21) · StructuralDivergence 8
(#4,6,9,13,23,28,31,34) · Infrastructure 14. Only #4 RCR and #31 ICR fully
cover the SPO-2³ causal lattice. *Checklist gap (measured): #19, #24, #31, #34
declare EMPTY `requires()` masks — `recipe_kernels.rs` says a real tactic
"should never" read nothing; these four demo bodies need their masks declared.*

## §2 When they fire — the three gates, in order

A recipe fires when it survives **all three** layers:

**(a) The selector** — `materialize::select_tactic(&ThoughtCtx) -> u8`.
`free_energy` (surprise) is the PRIMARY, causal axis — it alone picks the
wanted mechanism (the materialization criterion; `awareness_is_causal` is the
falsifier and it holds, measured):

| axis | bands | picks |
|------|-------|-------|
| `free_energy` | ≥0.66 leap / 0.33..0.66 inference / <0.33 routine | wanted **Mechanism**: StructuralDivergence / TruthAwareInference / ParallelIndependence (+5 on match; Infrastructure is never wanted) |
| `sd` (CollapseGate) | <0.15 FLOW / ≤0.35 HOLD / >0.35 BLOCK | wanted **Bucket**: Datapath / Control / Gate (+2 on match) |
| `rung` | ≥7 / 4..6 / <4 | wanted **Tier**: ExtremelyHard / Hard / CrossTier (+1 on match) |
| `dissonance` | ≥0.5 | +1 to every TruthAwareInference recipe (tie-weight, never an override) |

Highest score wins; ties go to the **lowest id**. Inside `materialize`, F is
recomputed each step as `0.4·(1−confidence) + 0.3·dissonance + 0.3·sd` — the
selector's F axis is independent only for standalone calls.

**(b) The kernel gate** — `Tactic::gate` (ONE default, no per-tactic
overrides, measured): **Gate-bucket** recipes (#5, 6, 8, 13, 20, 26) fire only
when the gate is NOT in FLOW (there is dispersion to act on); Control/Datapath
recipes always fire. A fired kernel folds its `delta_conf` into
`ctx.confidence`; a gated-off kernel returns `Outcome::skipped()`.

**(c) The loop** — `materialize(ctx, max_steps)`: each step recomputes F;
**rest** = gate in FLOW ∧ F < 0.2 (`HOMEOSTASIS_FLOOR`). A step that *fired*
settles the state (`sd ×= 0.85`, `dissonance ×= 0.6`,
`confidence += 0.35·(1−confidence)` — active inference: attending resolves
uncertainty), guaranteeing FLOW in ~log steps. A *blocked* step changed
nothing, so the loop halts rather than spin. "Can't NOT think while surprise
exists" is these three lines.

## §3 The measured dispatch map (54 awareness cells)

```text
gate   rung  dissonance     F<.33 routine   .33-.66 inference   F≥.66 leap
FLOW   R1-3  coh|contra     #1  RTE         #17 CDI             #28 SSAM
FLOW   R4-6  coh|contra     #1  RTE         #10 MCP             #28 SSAM
FLOW   R7-9  coh|contra     #1  RTE         #3  SMAD            #28 SSAM
HOLD   R1-3  coh|contra     #1  RTE         #17 CDI             #4  RCR
HOLD   R4-6  coh|contra     #1  RTE         #10 MCP             #4  RCR
HOLD   R7-9  coh|contra     #1  RTE         #3  SMAD            #4  RCR
BLOCK  R1-3  coh|contra     #5  TCP         #17 CDI             #6  TR
BLOCK  R4-6  coh|contra     #5  TCP         #10 MCP             #6  TR
BLOCK  R7-9  coh|contra     #5  TCP         #3  SMAD            #6  TR
```

**Measured findings (E-RECIPE-SELECTOR-REACHABILITY-1):**

1. **8 of 34 are selector-reachable**: #1 RTE, #3 SMAD, #4 RCR, #5 TCP, #6 TR,
   #10 MCP, #17 CDI, #28 SSAM. The other 26 NEVER win `select_tactic`.
2. **All 14 Infrastructure recipes: unreachable** (mechanism match = +5, the
   max non-match score = bucket 2 + tier 1 + reconcile 1 = 4; Infrastructure is
   never the wanted mechanism).
3. **In-band shadowing by the lowest-id tie**: PI loses #2, 20, 26, 30; TAI
   loses #7, 11, 21; SD loses #9, 13, 23, **#31 ICR**, 34. Sharpest case:
   **ICR — one of only two causal-lattice-Covered recipes — is permanently
   shadowed by #4 RCR** (same band, bucket, tier; higher id). The
   counterfactual runbook can only fire via the style fan or a direct
   `kernel(31)` call.
4. **The dissonance axis is inert in dispatch** (all 27 coherent/contradicted
   cell-pairs identical): the +1 boosts all six TAI recipes uniformly in-band
   and cannot cross the +5 gap between bands. Contradiction currently changes
   *nothing* about which tactic fires — only `materialize`'s settle path decays
   it. (The #515 regression test guards that dissonance must not OVERRIDE F; no
   test guards that it does anything at all.)
5. Selector and kernel-gate are mutually consistent: Gate-bucket winners (#5,
   #6) win only in BLOCK cells, where their ≠FLOW gate is satisfied — the
   selector never dispatches a recipe that would immediately skip.

Consequence for wiring: **the surprise selector is a narrow front-door (8
recipes); the style→mechanism fan is the wide door (whole mechanism class).**
Any production escalation edge (MUL → `select_tactic`) inherits the 8-recipe
surface unless it composes the style fan — a design decision, not an accident,
to make explicitly when W-D production wiring lands.

## §4 Where they wire — today (live consumers)

| wire | file | what it does |
|------|------|--------------|
| **The engine** | `crates/lance-graph-contract/src/materialize.rs` | `select_tactic` + `kernel(id).run` in the closed F→34→F loop; `Trace` = which tactic fired and why (provenance) |
| **Shader driver (provenance)** | `crates/cognitive-shader-driver/src/driver.rs:980` | `materialize_provenance(..)` runs `materialize(&mut ctx, 64)` ALONGSIDE every dispatch cycle; the `Trace` lands in the crystal as `MaterializeProvenance` |
| **Planner / kanban (the wide door)** | `crates/lance-graph-planner/src/strategy/style_strategy.rs` | `ThinkingStyle → cluster() → Mechanism → recipes_for(style)` fans the WHOLE mechanism class through `kernel(recipe.id).run` — the only live route to the 26 selector-unreachable recipes (`ExecTarget::Elixir`, the interpreted layer) |
| **Registry** | `crates/lance-graph-contract/src/recipe_kernels.rs:916` | `kernel(id) -> Option<&'static dyn Tactic>`, `all_kernels() -> [_; 34]` — direct dispatch for anyone holding an id |
| **Kanban view updates (example-tier)** | `crates/lance-graph/examples/graph_self_reasoning.rs` | the operator-priority wire: the thinking atoms (prefetch/deduction/syllogism/hypothesis/counterfactual/antithesis/synthesis/extrapolation/inference) as shipped graph ops, every unit of work a legal `KanbanMove` via `KanbanColumn::advance_on_gate` (Planning −550µs Σ-commit → CognitiveWork → Evaluation → Commit), `materialize` consuming `ThoughtCtx` read FROM the live graph, gestalt texture before/after (Chalmers emulation frame) |
| Exhibits | `contract/examples/cognitive_cycle.rs` · `contract/examples/recipe_dispatch_map.rs` (this doc's generator) · `lance-graph/examples/reasoning_loop.rs` (W-D: the loop reading a temporal stream) | runnable proof of each layer |

**Substrate split (rung-3 ladder row):** catalogue spine + kernels live HERE
(contract, zero-dep); the SIMD substrate primitives live in
`ndarray::hpc::styles` (`fn(Base17, NarsTruth) → result`) — **29 of 34
shipped**; missing exactly `RCR, TR, ASC, CAS, CR` (#4, 6, 7, 8, 11).

## §5 Where they wire — open edges (planned, gated)

| edge | status | gate |
|------|--------|------|
| **Kanban view updates, production** — the planner-side `Outcome`→`Candidate`/`KanbanMove` adapter | **deferred** (`style_strategy.rs:29`); example-tier wire shipped in `graph_self_reasoning.rs` | v3-kanban-executor-engineer territory (D-MBX-A6) |
| **MUL escalation → recipe** (`verdict_from(&MulAssessment)` → `select_tactic`/`materialize`) — the W-D production edge | designed, not wired | M-RUNG-1 pass + two-MUL reconcile (`.claude/plans/persistent-nars-kg-v1.md` W-D row) |
| **Rung fan → rung axis** (W-B `rung_candidates()` feeding `ctx.rung`) | probe not run | M-RUNG-1 |
| **StyleFamily(12) → recipe selection** — the TRUE rung-4→rung-3 `default_runbook` semantics | absent | persona-vs-rung-ladder **O2** (today's `default_runbook()` points at the persona vocabulary instead) |
| **Rung↔content wiring** (RungLevel knows Pearl depth, not content occupants) | absent | persona-vs-rung-ladder **O1** |
| Dissonance made causal in dispatch (finding 4 above), or documented as settle-only | undecided | next selector revision (with the #515 guard kept) |

## §7 All 34 carved out AND wired — the `(mechanism × depth)` address space + kanban foveation

The §3 finding (surprise selector reaches only 8) is not a gap in coverage — it
is a gap in ONE door. The 34 are fully addressable as a **bijective
`(mechanism × depth)` space**, and the union of the two live doors reaches every
one (measured, `foveated_awareness.rs`, KILL-gated on 34/34):

- **The carve (partition, proven from `RECIPES`):** each mechanism is a *column*,
  its recipes ordered by tier depth (CrossTier → Hard → ExtremelyHard) then id.
  `6 (PI) + 6 (TAI) + 8 (SD) + 14 (Infra) = 34`, no recipe in two columns. So
  `(mechanism, position)` is a complete address for the catalogue.
- **Door A — the style→mechanism fan (the wide door): reaches all 34.** The five
  style clusters cover all four mechanisms (`cluster_mechanism`: Analytical/Direct
  → TruthAware, Creative/Exploratory → Divergence, Empathic → Parallel, **Meta →
  Infrastructure** — the only door to the 14 Infrastructure recipes). `recipes_for`
  yields a whole column, so iterating the four covering styles yields all 34.
- **Door B — the surprise selector (the narrow front door): 8** (§3).
- **Union: 34/34.** Every recipe is carved out (a unique address) and wired (a
  live dispatch path). The one caveat kept explicit: the selector alone shadows 26;
  breadth needs the style fan.

**KanbanStep = self-driven foveated rendering (the model, runnable).** A board of
cards (mailboxes) renders at two resolutions: the ONE focal card (the *fovea*)
renders cognition at full detail — it climbs the rung ladder and dispatches the
carved recipe at `(its style's mechanism, its rung)` — while peripheral cards
render coarse (no dispatch). **Free energy is the saccade:** each cycle the fovea
jumps to the highest-`free_energy` card (attention spent where surprise is), and a
resolved card defoveates to Commit. Measured: the saccade sequence is
non-increasing in surprise; every card reaches rest+Commit; each focal climb is
Maslow-monotone. Each foveation is a legal `KanbanMove` (Planning −550µs Σ-commit
→ CognitiveWork → Evaluation → Commit) — the kanban-view update.

**RungLevel 0-9 = the Maslow pyramid of cognition.** The ten `RungLevel` names
ARE the pyramid; a foveated card tries the cheapest rung first and ESCALATES only
while surprise remains (unmet need drives the climb), resting when a rung resolves
it. Pearl anchors land where expected:

| rung | RungLevel | cognitive need | Pearl |
|---:|---|---|---|
| 0 | Surface | take in the raw signal | — |
| 1 | Shallow | match to something known | — |
| 2 | Contextual | relate to neighbors | **L1** association |
| 3 | Analogical | map onto another domain | — |
| 4 | Abstract | generalize past the instance | — |
| 5 | Structural | model the mechanism | **L2** do/intervene |
| 6 | Counterfactual | imagine it otherwise | **L3** counterfactual |
| 7 | Meta | reason about the reasoning | — |
| 8 | Recursive | reason about that, in turn | — |
| 9 | Transcendent | the whole reasons about itself | apex |

Honest frame: foveation / saccade / Maslow are **resource-allocation models made
mechanical** (a finite dispatch budget spent by surprise), not consciousness
claims. Board: `E-FOVEATED-AWARENESS-1`.

## §6 Regenerate

```sh
cargo run -p lance-graph-contract --example recipe_dispatch_map   # §1 checklist col, §3 map + reachability
cargo run -p lance-graph-contract --example foveated_awareness    # §7 carve, 34/34 wiring, foveation, Maslow
cargo test  -p lance-graph-contract recipes                       # catalogue pins (34, order, tallies)
```
