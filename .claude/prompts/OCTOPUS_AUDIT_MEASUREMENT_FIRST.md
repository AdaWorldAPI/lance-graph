# OCTOPUS — measurement-first related-work + contract audit (scoped prompt)

> **Filed 2026-08-26, operator-issued. NOT YET RUN.** This is the audit spec, not
> its result. A session executing it must return the §17 report BEFORE any code.
> Plan carrying the doctrine it audits:
> `.claude/plans/alpha-interventional-faithfulness-v1.md`.
> Companions: `entropy-closure-causal-ground-v1` (#1057),
> `mul-calibration-not-verdict-v1` (#1055).

## Standing constraints

- No code until the ontology classification (§1) and the F-OCT falsifiers (§15)
  are complete.
- Do not invent a new graph population. Do not rename established types until
  current code and existing plans are measured.
- **If a current plan already captures a point, cite/update it rather than
  opening duplicate architecture.**

## 0. Working thesis (hypothesis, not a naming mandate)

> Octopus is a domain-conditioned causal Graph-CoT over resident causal
> topology, whose cognitive traversal is epistemically typed and
> interventionally testable.

```text
Graph of Thoughts  graph OF reasoning        thoughts are vertices; edges = generation/dependency
Graph-CoT          reasoning OVER a graph    graph is substrate / KG / world representation
Causal-CoT         construct/complete/verify a causal graph during reasoning
Octopus            domain-conditioned reasoning OVER resident causal topology
                   + domain semantic lenses + alpha/attention receipt
                   + CE64 epistemic topology + Counterfactual/Revision
                   + interventional falsifiability
```

**Candidate law:** *The world graph remains canonical. The thought graph is a
receipt or projection, never a second world.*

## 1. Classify the four graph ontologies FIRST

| | nodes | edges | purpose |
|---|---|---|---|
| **A. World DAG** | domain/world entities | claimed causal relations | reason ABOUT the world |
| **B. Reasoning DAG** | thought steps / alpha events / log acts | generation / inheritance | diagnose THIS run |
| **C. Attribution graph** | tokens / features / logits / circuits | mechanistic influence | verify internals of one step |
| **D. Scoring graph** | expert-annotated entities/events | gold causal relations | GRADE reasoning, never generate |

Expectation to confirm or refute **with explicit evidence per mapping**: CE64
graph = World DAG; alpha = cognitive receipt / attention projection; any
diagnostic influence topology stays a borrowed Reasoning-DAG view; no
attribution graph is introduced; CausalPhys-style graphs are SCORING, not
storage.

## 2. Related work to verify (primary sources where possible)

**A. Graph of Thoughts** — what does an edge MEAN? (likely generation
dependency, not causality). Do not call Octopus a GoT unless the code genuinely
treats thought dependencies as canonical reasoning topology. Steal: non-linear
reasoning search.

**B. Graph-CoT** — does walking an existing graph match Octopus more closely?
Steal: CoT as query/traversal over an existing substrate.

**C. Causal-CoT** (EXTRACT / COMPLETE / VERIFY) — measure Stage I skeleton,
Stage II mediator/confounder completion, Stage III do-calculus verification.
**Do NOT import its generated DAG as resident state.** Architectural
translation to test:

```text
EXTRACT    claimed alpha/surprise state from the existing receipt
COMPLETE   retrieve from existing resident graph/views — invent no canonical mediators
CONSTRAIN  target = the predicted next cognitive transition
VERIFY     intervene on the claimed alpha state
CONTROL    irrelevant alpha intervention stays silent; shuffled topology must lose
```

**D. "Causal Graphs Meet Thoughts"** — likely the closest external precedent to
the resident-world doctrine (CoT as a query plan over an existing causal KG;
completion by retrieval, not invention). Measure that claim.

**E. Causal-Audit target-aware growth** — Exact/CloseHit/Bridge/None, Top-K
frontier. A search/VIEW constraint only, never a new ontology.

**F. PNS-CoT / sufficiency-necessity** — constructive/interventional pruning
only. Import no optimizer unless a local falsifier first demonstrates
decorative cognition.

## 3. CausalPhys mapping (only if justified)

ACC / Entity Faithfulness / Relation Awareness / Description Correctness.
Lesson: **right answer ≠ right causal relations**; models name entities far
better than edges. Map onto Octopus, then test whether typed execution receipts
let it go further. Candidate metrics — Entity, Relation, Interventional,
Epistemic-Topology, Reasoning-Band/Permission, Counterfactual Necessity.
**For each metric, identify a measurable current carrier. Do not invent
metrics because the names sound good.**

## 4. Alpha must earn the word "thinking"

recorded reasoning ≠ causally load-bearing reasoning. The standard is stricter
than whole-step textual redaction:

```text
ALPHA CLAIM      "this alpha/surprise state participates in cognition"
  ⇒ INTERVENTION perturb ONLY that claimed state
  ⇒ PREDICTED    next saccade / revision / attention move changes as predicted
  while IRRELEVANT ALPHA PERTURBATION remains SILENT
```

Hold spine/world/domain state fixed; measure the next transition, not the final
answer. Same behaviour on relevant and irrelevant ⇒ instrumentation. No effect
from the claimed state ⇒ decorative.

## 5. Diagnostic completeness

**replayability ≠ diagnostic completeness.** Translate the stage ladder onto
existing cognitive stages and LABEL the translation as ours: attention landing
→ saccade selection → digestion/revision → unresolved surprise. Stage-ablate
over EXISTING receipts; add no resident memory field. Question: can a failure be
localized to the earliest stage that lost or corrupted the needed state?

## 6. Adaptive Influence Graphs

They DO instantiate a per-failed-trace graph; they do NOT provide our
shuffled-edge null. Steal: diagnostic topology as a projection of a recorded
run. Discipline: flat receipt vs derived projection vs same projection with
randomized edges preserving counts — topology earns itself only by beating both
controls, inheriting the semantic-locality null-result discipline.
Law: **one receipt, many views** — not *every diagnosis deserves another
canonical graph*.

## 7. GFCM

Tail/variance dependence can hide from mean/covariance tests. Placement:
tail-sensitive CI test → evidence/test receipt → causal discovery → possibly
changes which world edge is admitted. **No "tail causal" bit inside CE64.**

## 8. Cognitive axes to preserve unless code disproves

MUL/DK (felt vs demonstrated; NOT a universal gate, NOT a consumer verdict, NOT
Sandbox, NOT Flow) · Flow/Homeostasis (regime fit, regulatory regions, NOT a
decision agent) · **Sandbox := Counterfactual + Revision, period** · Resonance
(proposes HOW to think) · Alpha (WHERE attention conducts) · ΔF (did the
adaptation earn itself).

## 9. Entropy + Settlement

`SettlementCell` = closure density × evidence competence → Crystal / **Glass**
(dense closure on thin evidence, the humility quadrant) / GroundedUnresolved /
Fog. Entropy is the THIRD refining signal (concentration of surviving
hypotheses). No `glass_gap()`, no new confidence scalar.

## 10. CE64 59..60 / 61..63 — re-verify before any normative text

59..60 `CausalTopology`: Direct / IndirectKnownIntermediates /
IndirectUnknownIntermediates / Unknown ⇒ known / projected / hole. No new bits.
D-ACR-7 guards binding: declared per class/rail, projection fallible,
unasserted provenance REFUSES, v1 temporal-taint must not masquerade as ground.
61..63 `ReasoningBand` is an epistemic PERMISSION band, **not** `confidence =
0.83`; scalar confidence stays in TruthValue `(f,c)`. A Surface/Association
fill must not close a causal hole.

## 11. Sudoku walker

DETECT → BOUND → PROPOSE → FILTER → GATE → TEST (counterfactual REMOVAL must
degrade coherence) → ACCEPT / KEEP UNKNOWN / ASK FOR MEANS. Narrowing 60,000 to
7 is persisted epistemic structure. Unknown never forces guessing.

## 12. Humility

No Boolean, enum, agent or homunculus. Ladder over existing carriers: I KNOW /
I DO NOT KNOW / I KNOW THAT I DO NOT KNOW / I LACK THE MEANS / I NEED TO ASK.
No new epistemic state machine unless an existing carrier fails a falsifier.

## 13. What "domain-aware" must mean

Not "the prompt contains domain vocabulary". The same resident world read
through different domain ClassView lenses — domain changes admissible reading,
constraints, interpretation, possibly allowable traversal; it does NOT change
world identity, canonical storage, or edge encoding. Test whether
"Domain-Aware Causal CoT" is precise enough vs "Domain-Conditioned Causal
Graph-CoT". Do not rename Octopus internally unless asked.

## 14. The strongest claim, and its two orthogonal proofs

> Octopus is a domain-aware causal reasoning trace only if its internal state is
> interventionally load-bearing AND its traversal is faithful to the causal
> topology it claims to reason over.

Keep the failure modes separate: graph-faithful but computationally decorative ·
load-bearing but causally wrong · right answer wrong relation · right entity
wrong edge.

## 15. Required falsifiers

| id | falsifier |
|---|---|
| F-OCT-1 | relevant alpha perturbation changes the predicted next transition |
| F-OCT-2 | matched irrelevant alpha perturbation does NOT change it |
| F-OCT-3 | correct entities + incorrect causal edges must FAIL even with a correct final answer |
| F-OCT-4 | a real diagnostic/influence topology beats a randomized-edge view preserving node/event counts |
| F-OCT-5 | masking each stage (landing / saccade / revision / surprise) exposes whether localization depends on it |
| F-OCT-6 | two basins, identical low entropy, Direct-dominant vs Unknown-dominant, settle and route differently |
| F-OCT-7 | an Association/Surface-band mediator may not close a causal-topology hole |
| F-OCT-8 | an accepted mediator's removal measurably degrades the claimed explanation |
| F-OCT-9 | a search that narrows without resolving preserves the unresolved state rather than fabricating certainty |
| F-OCT-10 | any proposed new graph structure must show why it cannot be a derived view; if it cannot, REJECT the population |

## 16. What NOT to do

No second canonical thought graph · no Causal-CoT DAG as resident state · no
AIG per-trace graph as canonical storage · no GFCM/tail evidence in CE64 · no
ReasoningBand→scalar · no TruthValue merged with topology · no Flow as decision
agent · no MUL as universal gate · no Sandbox enum/state machine · no Humility
enum/bool · no universal GateGround · replay is not proof of cognition · answer
accuracy is not proof of causal reasoning · entity mention is not relation
faithfulness · low entropy is not mastery · no new DTO merely to make the
architecture easier to describe.

## 17. Required output (measurement report, before implementation)

1. exact current Octopus topology in code · 2. exact role of alpha in the
next-transition path · 3. every place alpha is observational vs causally
load-bearing · 4. mapping onto World/Reasoning/Attribution/Scoring · 5. GoT vs
Graph-CoT vs Causal-CoT vs Octopus · 6. CausalPhys metrics → native carriers ·
7. intervention mapping onto alpha · 8. stage-ablation mapping onto existing
receipts · 9. projection/null-test mapping · 10. current CE64 59..60 / 61..63
semantics · 11. Settlement + entropy + TruthValue + MUL/DK division of labour ·
12. Sudoku-walker fit to current code · 13. what already exists · 14. what is
only hypothesis · 15. smallest plan/board correction · 16. falsifier table with
carrier, intervention, expected result, failure meaning · 17. naming
recommendation.
