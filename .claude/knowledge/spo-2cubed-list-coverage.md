# SPO 2³ — the official list + cross-repo list-coverage inventory

> **READ BY:** truth-architect; anyone checking whether a taxonomy reduces to the causal lattice.
> **Status:** rubric established (this file). Inventory pass = in progress (Opus agent).
> **Date:** 2026-05-27.

## The official 2³ Rung ladder list (the coverage rubric)

The SPO 2³ is the **powerset of {S, P, O}** — 8 evidential projections of a causal claim, **NOT** a distance-cube. Grounded in `rung-mul-grounding-v1` + CausalEdge64 v2 §6 (causal mask `[40:42]`, `nars_engine::MASK_SPO = 0b111`). Counterfactual is the top element (it is "one of those").

| mask | proj | level | meaning |
|---|---|---|---|
| `0b000` | `___` | 0 | base rate (Association / prior) |
| `0b100` | `S__` | 1 | marginal: does S occur? |
| `0b010` | `_P_` | 1 | marginal: does P occur? |
| `0b001` | `__O` | 1 | marginal: does O occur? |
| `0b110` | `SP_` | 2 | pair: cause↔mechanism (O marginalized) |
| `0b101` | `S_O` | 2 | pair: cause↔effect (P marginalized — maybe spurious) |
| `0b011` | `_PO` | 2 | pair: mechanism↔effect = **Intervention** |
| `0b111` | `SPO` | 3 | joint = **Counterfactual** (top of the ladder) |

Causation = the **screening-off pattern** across the 8 (front-door/back-door as NARS truth over the lattice), not the joint alone. The 4 levels (0/1/2/3) = the Pearl ladder = the rung axis.

## Coverage test ("is list X covered by SPO 2³?")

A list is **covered** iff its members map onto the 8 projections / 4 levels (or a sub-lattice) — i.e. it *is* the causal decomposition wearing other names. **Partial** = some members map, others are orthogonal. **Not covered** = orthogonal axis (qualia, presence, persona, operations, tensor/ML types).

## REFERENCE SET — the 34 LLM reasoning tactics (MUST be included)

Source: Stakelum, "Beyond Chain-of-Thought: 34 next-gen LLM tactics." These are the mandated reference styles the catalogue must cover. Bucket = hardware-partition (datapath/control/gate). 2³ = causal-lattice coverage.

| # | tactic | bucket | 2³? | maps to (workspace) |
|---|---|---|---|---|
| 4 | RCR Reverse-Causality | control | **Covered** | backward `S_O` / Abduction |
| 31 | ICR Iterative-Counterfactual | control | **Covered** | `SPO`=0b111 Counterfactual (top) |
| 11 | ICR Contradiction-Resolution | control | Partial | Revision + Contradiction (committed, not resolved) |
| 7 | ASC Adversarial-Self-Critique | control | Partial | InnerCouncil split / dissonance |
| 21 | SSR Self-Skepticism | control | Partial | InnerCouncil split |
| 17 | CDI Cognitive-Dissonance-Induction | control | Partial | dissonance gate |
| 30 | SPP Shadow-Parallel | control | Partial | the counterfactual majority/minority fork |
| 2 | HTD Hierarchical-Decomposition | control | Not | Decompose op |
| 22 | ETD Emergent-Task-Decomposition | control | Not | Decompose op (dynamic) |
| 19 | ARE Algorithmic-Reverse-Eng | control | Not | Decompose / Abduct |
| 1 | RTE Recursive-Thought-Expansion | gate | Not | rung depth × Expand/Compress |
| 8 | CAS Conditional-Abstraction-Scaling | gate | Not | Abstract↔Concretize × rung |
| 3 | SMAD Simulated-Multi-Agent-Debate | control | Not | `a2a_blackboard` / InnerCouncil |
| 5 | TCP Thought-Chain-Pruning | gate | Not | entropy(SD) prune |
| 20 | TCF Thought-Cascade-Filtering | gate | Not | entropy(SD) select |
| 6 | TRR Thought-Randomization | gate | Not | temperature (Staunen) |
| 13 | CDT Convergent-Divergent | gate | Not | explore↔exploit temperature |
| 26 | CUR Cascading-Uncertainty-Reduction | gate | Not | FreeEnergy / confidence gate |
| 10 | MCP Meta-Cognition | control | Not | Meta lane |
| 15 | LSI Latent-Space-Introspection | control | Not | Meta / self-state-awareness |
| 23 | AMP Adaptive-Meta-Prompting | control | Not | Meta dispatch |
| 33 | DTM Dynamic-Task-Meta-Framing | control | Not | Meta dispatch |
| 16 | PSO Prompt-Scaffold-Optimization | control | Not | Hierarchize/scaffold |
| 12 | TCA Temporal-Context-Augmentation | datapath | Not | temporal lane / Markov ±5 |
| 18 | CWS Context-Window-Simulation | control | Not | episodic memory / WitnessCorpus |
| 28 | SSAM Self-Supervised-Analogical-Mapping | control | Not | Analogy op |
| 24 | ZCF Zero-Shot-Concept-Fusion | control | Not | Synthesize / Weave |
| 34 | HKF Hyperdimensional-Knowledge-Fusion | control | Not | Synthesize (cross-domain) |
| 25 | HPM Hyperdimensional-Pattern-Matching | datapath | Not | resonance (cosine sweep) |
| 27 | MPC Multi-Perspective-Compression | control | Not | Compress × perspective |
| 9 | IRS Iterative-Roleplay-Synthesis | control | Not | persona / model_other |
| 29 | IDR Intent-Driven-Reframing | control | Not | disambiguation / clarification |
| 32 | SDD Semantic-Distortion-Detection | control | Not | clarification op |
| 14 | M-CoT Multimodal-CoT | datapath | Not | multimodal sensorium |

**Tally:** 2³ Covered = 2 (RCR, ICR-counterfactual) · Partial = 5 · Not = 27. So **the 2³ lattice covers only the causal-reasoning tactics**; the other 27 are orthogonal axes (operations / meta / gating / memory) — confirming 2³ is the *causal* spine, not the whole style space. Buckets: ~6 datapath-touching, ~21 control, ~7 gate.

## Inventory — workspace lists (Opus inventory pass, 2026-05-27)

**31 enumerated cognitive lists** inventoried across contract / planner / cognitive-shader-driver / thinking-engine / holograph + 3 markdown taxonomies (bighorn-36 ops, agi-chat Quad-Triangle / styleBank / 9-RI). **2³ tally: Covered 4 · Partial 6 · Not 21** — confirming 2³ is the *causal spine only*.

**Covered (4 — these ARE the lattice):**
- `CausalMask` — the 8 projections verbatim.
- `nars_engine` `MASK_*` / `ALL_MASKS` — the powerset masks.
- `PearlLevel` — the 3 rungs (Association/Intervention/Counterfactual).
- `CANONICAL_ATOMS[0..3]` — the Pearl lanes of the 33-TSV.

**Partial (6 — causal members ride the lattice, rest orthogonal):**
- the 4 NARS inference enums (Intervention=`_PO`, Counterfactual=`SPO` ride it; Deduction/Induction/Abduction/Revision/Synthesis/Analogy orthogonal).
- `InferenceOp` (10, Pearl-tagged).
- `CANONICAL_ATOMS` Operation group + holograph `VerbCategory` (its Causal subcategory).

**Not covered (21 — orthogonal axes):** every style / cluster / σ-tier / rung-ladder / layer / qualia / ghost-echo / presence / MUL / expert-capability list. (These are the operation/meta/gating/memory/affect axes — not causal.)

**Gaps — reference tactics with NO enumerated workspace home yet:** #18 CWS (context-window/memory), #14 M-CoT (multimodal), #29 IDR / #32 SDD (intent/clarification), #16 PSO / #23 AMP / #33 DTM (meta-prompting scaffold), #12 TCA (temporal), #22 ETD / #19 ARE (dynamic decomposition), #5 TCP / #20 TCF (dedicated thought-pruning). Strongest orthogonal homes for the rest: bighorn-36 ops (#2/#8/#27), InnerCouncil Archetype+split (#3/#7/#21/#30), MUL layer (#26/#10/#15).

> **Cross-repo note (per `agi-stack-cross-repo.md`):** ladybug-rs implements **all 34** as structural primitives (mapped to the Sun et al. reasoning ladder); ada-consciousness implements 21. So the "gaps" above are gaps in *lance-graph's enumerated lists*, not in the AGI stack — the homes exist upstream. Also: the real lattice is **SPOQ** (S/P/O + Q-qualia, 4 roles); SPO 2³ = the causal slice, Q = the orthogonal affective overlay.
