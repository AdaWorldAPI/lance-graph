# Research pressure and forward momentum — 2026-08-23

Status: documentation / research synthesis only. No new bit allocation, no widened ABI, no opcode allocation, no tenant allocation, and no claim that any cited mechanism is already implemented in lance-graph.

This note records a small set of recent papers that materially pressure the current causal / epistemic / behavioral architecture. The purpose is not literature collection. Each source is reduced to: **what the paper actually establishes, what architectural pressure follows, where the pressure lands in the current substrate, what not to copy, and the smallest falsifier that would justify further work.**

The standing default is conservative:

> **compose with the existing 64-bit / 6×(8:8) substrate first; mint a new representation only after a falsifier proves that the existing one aliases information that must remain distinct.**

## Snapshot of the current internal boundary

Relevant recent work in this repository:

- #989 established that the cognitive micro-IR is a real compiler-like front end and that the missing piece was an interpreter/trace, not a missing neural mechanism.
- #990 framed the self-learning / self-programming endgame around precision before closure, owner-local traces, counterfactual replay, Revision and MUL-gated promotion.
- #991 established `IntermediateUnknown` as a **constraint-addressed pothole**, not a generic uncertainty label: the hole is an address into inherited structure.
- #992 is the concrete follow-up to #989: `PROBE-LOCO-INTERPRETER-1` now has an actual minimal interpreter, input-dependent execution traces and deterministic replay on real algorithms. This materially lowers the cost of testing mechanism identity and trace-conditioned abstractions.
- #993 proposes the next domain-general falsifier ladder: Sudoku → chess → crossword, with anonymous internal quality basins over alpha-traced reasoning and strict separation between object truth, epistemic state and metacognitive cartography.

The research below should therefore be read against an architecture that already has: causal edges, explicit intermediate-known/intermediate-unknown distinctions, ontology ancestry, historical/version boundaries, counterfactual and Revision machinery, provenance/witness concepts, an emerging executable OGAR-loco trace, and an alpha-layer concept for reasoning visibility.

---

# 1. Bipartite Graphical Causal Models: intervention identity is mechanism-specific

## Source

Joris M. Mooij, **“Causal Reasoning with Bipartite Graphical Causal Models.”** Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence (UAI 2026), PMLR 337:4607–4633. Conference 17–21 Aug 2026. Extended arXiv version posted 20 Aug 2026.

- Proceedings: https://proceedings.mlr.press/v337/mooij26a.html
- arXiv: https://arxiv.org/abs/2608.19831

**Evidence grade:** peer-reviewed conference paper + extended author version.

## What it actually establishes

The paper gives a precise counterexample to the assumption that a perfect intervention is always fully identified by a target variable and imposed value. In systems at equilibrium with cyclic feedback, two interventions can both enforce the same `X = x` and nevertheless have different consequences because they replace **different governing equations/mechanisms**.

The proposed hard-intervention notation is correspondingly richer:

```text
do(f_j : X_v = ξ_v)
```

The intervention identifies:

1. which equation/mechanism is replaced;
2. which variable is targeted;
3. which value is imposed.

The bipartite graph makes variables and equations first-class, separate nodes.

## Why this pressures lance-graph

The current causal substrate is intentionally dense. That density becomes dangerous if two interventions that are semantically distinct collapse to the same representation because they share endpoint/value/topological location.

The paper therefore creates one sharp question:

> **Does lance-graph have enough existing identity to distinguish “same variable, same value, different governing mechanism” during counterfactual execution?**

This does **not** yet justify new CausalEdge64 bits.

The most promising existing carrier is executable operation/mechanism identity from OGAR-loco. #992 matters here: the substrate now has real `FnIndex`-bearing execution traces rather than only a static recipe vocabulary. Mechanism identity may therefore be composable as a sidecar/reference into counterfactual receipts instead of being packed into the edge.

## Smallest falsifier: `PROBE-BGCM-MECHANISM-IDENTITY-1`

Construct a tiny cyclic/equilibrium fixture with:

- one target variable `X`;
- one imposed value `x`;
- two distinct mechanisms/equations `f1` and `f2` that can each be replaced to impose `X=x`;
- downstream state chosen so replacing `f1` and replacing `f2` produce different consequences.

Required result:

```text
same target X
same imposed value x
mechanism f1 != mechanism f2
        ↓
counterfactual histories remain distinguishable
```

**KILL** the current counterfactual identity contract if the two interventions alias before or during replay.

If the probe survives by carrying existing `FnIndex` / mechanism identity in a versioned receipt, stop. Do not allocate a bit.

If the probe fails because there is genuinely nowhere lossless to bind mechanism identity, the failure itself is the evidence needed for a new reference contract.

## Do not copy

Do not mirror the paper’s full BGCM object model merely because it is elegant. The architectural requirement is narrower: preserve intervention mechanism identity where it changes consequences.

## Forward momentum

**Priority: P0.** This is the clearest recent source of a possible silent causal alias.

---

# 2. Property-driven causal abstraction: compress by reason, not by surface state

## Source

Jule Schmidt, Maximilian Weininger, Clemens Dubslaff, David Parker, Nils Jansen, **“Property-driven Causal Abstractions for Markov Decision Processes.”** arXiv v1 29 Jul 2026; updated 7 Aug 2026. FMCAD 2026 paper with reproducibility artifact.

- arXiv: https://arxiv.org/abs/2607.26787
- Artifact: https://zenodo.org/records/21825827

**Evidence grade:** accepted formal-methods work + reproducibility artifact.

## What it actually establishes

The paper addresses state-space explosion in factored MDPs. Its useful move is not simply “cluster similar states.” It identifies states that share the **same causal reasons for fulfilling or violating a chosen property**, and uses those reason-equivalence classes as abstractions.

The reported reduced models can support near-optimal policies and can generalize to related larger MDPs.

## Why this pressures lance-graph

#991 and #993 already require a way to decide when two reasoning situations are “the same kind of pothole” or “the same kind of reasoning moment.” A naïve learner could cluster by:

- candidate count;
- same rung;
- same MUL bits;
- similar alpha shape;
- similar trace length;
- similar vector/Q-basin.

Those are easy but potentially wrong.

The paper suggests a stronger criterion:

> **two reasoning states may be abstractable when the causal reason-set for the next relevant property is the same, even if their surface graph states differ.**

Conversely, two states with identical candidate counts should remain distinct when the reasons supporting/refuting their next action differ.

This is a possible bridge between #991’s Sudoku epiphany and #993’s domain-general internal basins.

## Smallest falsifier: `PROBE-REASON-EQUIVALENCE-ABSTRACTION-1`

Use Sudoku first because the oracle is exact.

Prepare four states:

```text
S1, S2: same candidate count, different causal reason-set
S3, S4: different board surfaces, same causal reason-set for the next inference
```

Test two abstraction schemes:

```text
A. surface/state similarity
B. causal reason-equivalence
```

Measure whether an OGAR-loco strategy learned or selected on the abstraction:

- remains correct;
- preserves the next warranted elimination/commit;
- reduces touched states / trace length;
- transfers across held-out puzzles.

**KILL** the reason-equivalence hypothesis if B gives no correctness/transfer advantage over a simple surface baseline, or if it collapses states whose different reasons require different actions.

## Landing point

Start as a **derived trace-side equivalence relation**. Do not make it storage identity. Do not make a new Q-basin type. Do not change `IntermediateUnknown`.

The reason set can initially be reconstructed from existing receipts: which constraints, witnesses, ancestry relations and counterfactual eliminations actually supported the transition.

## Forward momentum

**Priority: P1 after the Sudoku trace exists.** If it works, this becomes a principled candidate for how domain-general reasoning basins are formed.

---

# 3. Sequential chance constraints: local safety does not compose into trajectory safety

## Source

Minkyoung Kim, Beakcheol Jang, **“Chance-constrained selection of sequential intervention strategies from counterfactual estimates.”** arXiv, 13 Aug 2026. Code released with the paper.

- arXiv: https://arxiv.org/abs/2608.13209
- Code: https://github.com/mfriendly/counterfactual-chance-selection

**Evidence grade:** recent preprint + released implementation + experiments including environments with exact counterfactual ground truth.

## What it actually establishes

For sequential intervention strategies under cumulative resource constraints, two strategies can have the same expected total cost while having very different probabilities of exceeding the budget.

The key architectural point is:

> **tail risk is a property of the whole strategy and does not generally decompose into independent stage-local risk checks.**

The paper therefore evaluates candidate strategies as wholes under a chance constraint rather than assuming that acceptable local means imply acceptable trajectory behavior.

## Why this pressures lance-graph

A compact causal/reasoning substrate is vulnerable to a seductive mistake:

```text
edge looks safe
+ edge looks safe
+ edge looks safe
= trajectory is safe
```

That implication is invalid whenever dependencies/correlations across the path matter.

The warning applies broadly to future use of:

- per-edge confidence;
- per-step MUL trust;
- local eigenvalue/score;
- local provenance completeness;
- per-action resource estimate;
- per-rung admissibility.

None of those alone proves whole-counterfactual-path safety or bounded resource use.

## Smallest falsifier: `PROBE-WHOLE-TRAJECTORY-RISK-1`

Build two short multi-step counterfactual strategies:

```text
A = a1 → a2 → a3
B = b1 → b2 → b3
```

Constrain them so:

- each corresponding step has the same local expected risk/cost;
- the cumulative dependency structure differs;
- one strategy has materially higher probability of a bad terminal/budget event.

Then ask the current selection logic to rank A vs B **without revealing the realized final outcome**.

**KILL** any claim that local edge/step safety is sufficient if the system cannot distinguish the strategies.

## Landing point

Whole-trajectory risk belongs in a **counterfactual execution receipt / strategy evaluation sidecar**, not in CausalEdge64.

If a future MUL policy consumes it, MUL should read the whole-strategy evidence rather than attempting to reconstruct a path guarantee from local confidence bits.

## Forward momentum

**Priority: P1/P2.** This should become a standing guard before self-programming promotion or expensive external action is allowed to treat local trust as a global guarantee.

---

# 4. Onto-Explain: provenance says how; ontology says why

## Source

Wassim Jaziri, **“Ontologies explain what and provenance explains how: onto-explain for execution-consistent KG explanations.”** Expert Systems with Applications, article 133876, available online 7 Aug 2026.

- DOI / publisher: https://www.sciencedirect.com/science/article/abs/pii/S0957417426027843

**Evidence grade:** peer-reviewed journal article, in press / journal pre-proof.

## What it actually establishes

The paper’s useful distinction is simple and strong:

```text
execution provenance  → how the result was produced
ontology justification → why the result is semantically admissible
```

It argues that either alone is insufficient for a faithful explanation and combines provenance tracing, ontology lifting, attribution/ranking and contrastive reasoning in one execution-consistent pipeline.

The reported evaluation includes high execution faithfulness and scaling tests up to 100M triples.

## Why this mostly confirms lance-graph

This is not a reason to build another explainer subsystem. It confirms an architectural separation already emerging across the substrate:

- execution receipts / provenance;
- HHTL / ontology ancestry and constraints;
- counterfactual/contrastive reasoning;
- alpha as visibility rather than proof.

#992 makes the distinction newly actionable because there is now an actual execution trace to compose with semantic justification.

## Smallest falsifier: `PROBE-HOW-WHY-CONSISTENCY-1`

For one bounded reasoning episode, produce two independently inspectable products:

```text
HOW:
  exact OGAR-loco execution receipt / provenance chain

WHY:
  ontology/HHTL constraints that license the derived conclusion
```

Then cross-check them.

**KILL** explanation faithfulness if either of these occurs:

1. the ontology produces a plausible “why” that was not supported by the actual execution path;
2. the execution trace is exact but yields a conclusion inconsistent with the ontology constraints;
3. alpha focus is mistaken for either provenance or semantic proof.

## Landing point

Compose existing representations first. The desired artifact is a paired receipt, not a new causal edge variant.

A useful future explanation object should be able to answer separately:

```text
What did the machine actually do?
What facts/constraints make the conclusion warranted?
What alternative failed, and why?
```

## Forward momentum

**Priority: P2.** Strong confirmation and an excellent audit invariant, but less likely than BGCM to expose a missing primitive.

---

# 5. RippleMem: anchor retrieval plus bounded associative completion

## Source

Jingbo Ji, Lingyi Li, Xilong Cheng, Yuhao Zhou, Wenji Zhang, Yuting Tan, Yunxiao Qin, **“RippleMem: From Isolated Retrieval to Associative Recollection for Long-Term Agent Memory.”** arXiv, 13 Aug 2026.

- arXiv: https://arxiv.org/abs/2608.13334

**Evidence grade:** recent preprint with LoCoMo / LongMemEval-S experiments.

## What it actually establishes

The paper argues that the memory problem is often not storage but **recovering distributed supporting evidence**. Its retrieval pattern is:

```text
query
  ↓
retrieve one or more relevant anchors
  ↓
expand from those anchors over semantic/structural associations
  ↓
recover evidence that flat retrieval missed
```

The reported experiments show gains over isolated retrieval and substantially lower graph-construction cost than compared graph-memory approaches.

## Why this is mostly confirmation

The pattern is already close to AriGraph / episodic graph composition and therefore should not trigger a new memory subsystem.

The useful part is the experimentally crisp contrast:

```text
anchor-only
vs
anchor + bounded associative expansion
```

That is worth measuring against the existing substrate.

## Smallest falsifier: `PROBE-EPISODIC-ASSOCIATIVE-COMPLETION-1`

Create queries whose decisive evidence is deliberately split across episodes so that the first relevant anchor alone is insufficient.

Compare:

1. anchor-only retrieval;
2. anchor + bounded graph expansion;
3. current AriGraph composition if it differs from (2).

Measure:

- evidence recall;
- false-support rate;
- touched rows/edges;
- expansion depth;
- answer/reasoning correctness;
- whether historical/version boundaries remain intact.

**KILL** the need for any new work if existing AriGraph already provides the same gain at equal or lower touched-state cost.

## Forward momentum

**Priority: P3 / opportunistic.** Test the retrieval pattern, do not copy the architecture.

---

# Cross-paper synthesis

The five papers do not point toward five new subsystems. They pressure four existing boundaries.

## A. Identity pressure

BGCM says causal intervention identity may require **mechanism identity**, not just target/value.

The existing design response should be:

```text
counterfactual target/value
        +
versioned mechanism / FnIndex identity
        ↓
exact intervention receipt
```

before any new bit is considered.

## B. Abstraction pressure

Property-driven causal abstraction says reusable reasoning states may be grouped by **why they satisfy/violate a property**, not by superficial state similarity.

This is especially relevant to:

```text
IntermediateUnknown
→ Sudoku trace
→ repeated reasoning episode
→ anonymous Q-basin / learned macro
```

A future basin should earn its identity by preserving causal/reason structure, not merely by clustering telemetry.

## C. Composition pressure

The chance-constrained work says local good-looking steps do not prove a good whole trajectory.

That means future counterfactual / self-programming policy must preserve a distinction between:

```text
local edge/step evidence
whole-plan evidence
```

MUL should govern commitment using the latter when the decision concerns the latter.

## D. Explanation pressure

Onto-Explain sharpens the boundary between:

```text
HOW  = execution provenance
WHY  = semantic/ontology justification
LOOKED-HERE = alpha attention
```

These three should remain separate and composable.

RippleMem adds a smaller retrieval lesson: a relevant anchor may be only the start of evidence acquisition, not the evidence itself.

---

# Forward momentum: proposed sequence

The following sequence is intentionally ordered by **information value**, not conceptual glamour.

## P0 — `PROBE-BGCM-MECHANISM-IDENTITY-1`

Why first: it tests a potentially silent alias in causal intervention semantics.

Pass condition: same target/value + different mechanism survives as different counterfactual histories using existing reference/receipt machinery.

Failure meaning: evidence for a missing intervention-identity contract.

No new bits before failure.

## P1 — `PROBE-INTERMEDIATE-UNKNOWN-SUDOKU-1`

Already motivated by #991. Use Sudoku as the sealed-world oracle for:

- parent/HHTL constraint inheritance;
- candidate-space narrowing without premature closure;
- counterfactual branch;
- contradiction;
- Revision;
- historical replay;
- alpha visibility;
- exact brute-force comparison.

This becomes the first natural source of real reasoning traces for the next probes.

## P1b — `PROBE-REASON-EQUIVALENCE-ABSTRACTION-1`

Run only after Sudoku traces exist.

Ask whether the causal reason-set predicts reusable strategy better than raw state similarity.

If yes, use it as a candidate abstraction layer for behavioral macro learning and later anonymous quality basins.

## P2 — `PROBE-WHOLE-TRAJECTORY-RISK-1`

Before any serious autonomous self-programming or external action, demonstrate that the system can distinguish whole-plan risk from local step quality.

Do not allow a local confidence field to silently become a trajectory guarantee.

## P2b — `PROBE-HOW-WHY-CONSISTENCY-1`

Once execution receipts and Sudoku/counterfactual episodes exist, produce paired HOW/WHY explanations and prove that neither alpha attention nor ontology plausibility can counterfeit execution truth.

## P3 — `PROBE-EPISODIC-ASSOCIATIVE-COMPLETION-1`

Run as a benchmark against existing AriGraph behavior. Keep only if it provides measured retrieval gain.

## P4 — #993 curriculum

Only after the sealed-world pieces survive:

```text
Sudoku
  ↓ exact constraint reasoning
Chess
  ↓ adversarial counterfactual planning + Elo curve
Crossword
  ↓ semantic ambiguity + reciprocal Revision
```

The universal claim is not “one solver handles three games.” The claim is stronger and therefore easier to falsify:

> **the same domain-neutral cognitive grammar, epistemic state machinery, counterfactual/Revision loop and learned behavioral abstractions remain useful when the surface domain changes.**

For chess, publish not only Elo but also:

- experience / games;
- wall-clock;
- nodes/branches considered;
- brute-force share;
- touched bytes;
- Revision count;
- learned macro support;
- Elo.

The interesting curve is `Elo ↑` while unnecessary search/touched-state cost falls.

For cross-domain quality basins, keep the strongest #993 fence:

> anonymous `Q*` basins first; human names only after recurrence, permutation, seed and domain-leakage controls.

---

# What not to do

1. **Do not reopen bits 59..63.** The research above gives no reason to repurpose them as reward, curiosity, free energy, trajectory risk or explanation state.
2. **Do not make alpha proof.** Alpha is a metacognitive map of where reasoning went. Provenance says what executed; ontology says what is warranted.
3. **Do not create a BGCM clone.** Preserve mechanism identity only where a falsifier proves endpoint/value identity is insufficient.
4. **Do not equate clustering with abstraction.** A useful reasoning basin must preserve a reason/behavioral invariant and survive controls.
5. **Do not infer whole-plan safety from local confidence.** Store/evaluate whole-path evidence where whole-path claims are made.
6. **Do not create another memory system for RippleMem.** Benchmark anchor+expansion against AriGraph first.
7. **Do not let frequency become success.** Repeated traces are macro candidates; outcome, precision gain, causal validity and held-out transfer decide usefulness.
8. **Do not let future knowledge repair historical ignorance.** Every probe remains version-bounded and no-hindsight.

---

# Research ledger

| Source | Evidence | Pressure | Current verdict | Next action |
|---|---|---|---|---|
| Mooij, BGCM, UAI 2026 | peer-reviewed | intervention needs mechanism identity | **NOVEL GAP candidate** | P0 mechanism-identity falsifier |
| Schmidt et al., causal MDP abstraction | accepted + artifact | abstract by causal reasons | **NOVEL GAP candidate** | test on Sudoku traces |
| Kim & Jang, sequential chance constraints | preprint + code | whole trajectory ≠ sum of local safety | **NOVEL GAP / guard** | whole-path risk falsifier |
| Jaziri, Onto-Explain | peer-reviewed journal | provenance HOW + ontology WHY | **CONFIRMS** | paired execution/semantic audit |
| Ji et al., RippleMem | preprint | anchor → bounded associative completion | **CONFIRMS / benchmark** | compare against AriGraph |

---

# The narrow architectural thesis

These papers do not tell lance-graph to become larger. They tell it where **precision must survive compression**.

A dense substrate can remain dense if it preserves the few distinctions that actually change reasoning:

```text
same target/value but different mechanism
same surface state but different causal reason
same local risk but different whole-path tail
same explanation text but different execution provenance
same anchor relevance but different distributed evidence
```

That is the forward direction:

> **keep the bytes boring; spend complexity on identities, constraints, receipts and falsifiers that prevent different epistemic situations from aliasing.**

If the existing topology and sidecars can carry those distinctions, keep them. If a falsifier proves they cannot, then the failure has earned the next primitive.
