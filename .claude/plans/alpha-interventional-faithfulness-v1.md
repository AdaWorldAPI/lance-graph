# Plan: a recorded alpha is instrumentation until it is interventionally load-bearing (`alpha-interventional-faithfulness-v1`)

> **Status:** PROPOSAL (measured targets, unbuilt) — 2026-08-26. PLAN/BOARD ONLY.
> **Companion to:** `entropy-closure-causal-ground-v1` (#1057),
> `mul-calibration-not-verdict-v1` (#1055).
> **Level distinction:** #1057 is humility about **the world**. This is humility
> about **introspection**.

> **⚠ Literature status: OPERATOR-SUPPLIED, UNVERIFIED IN SESSION.** Every paper
> cited below came from an operator literature survey. No citation was fetched
> or read in this session and several IDs post-date any corpus available here.
> They are recorded as *provenance for the idea*, never as measured evidence.
> Any claim resting on a paper's numbers must be re-checked against the paper
> before it is promoted past PROPOSAL.

---

## 0. The thesis

> **A recorded alpha is instrumentation until an intervention on it moves the
> next transition, and an intervention off it does not.**

Or in the strictest form: *a domain-aware causal reasoning trace only if its
internal state is interventionally load-bearing and its traversal is faithful to
the causal graph it claims to reason over. Anything less is instrumentation with
good typography.*

**The governing law (ratified 2026-08-26), which binds everything below:**

> **The world graph remains canonical. The thought graph is a receipt or a
> projection, never a second world.**

Two axes, never collapsed:

```text
1. INTERVENTIONAL FAITHFULNESS
   Does changing the claimed internal state
   change the next cognitive transition?

2. CAUSAL-GRAPH FAITHFULNESS
   Does the trace preserve the actual causal relations,
   not merely the right entities or the final answer?
```

## 1. The probe, and its measured target

```text
ALPHA CLAIM        "this state is part of cognition"
      ↓ implies
INTERVENTION       change that state — ONLY that state
      ↓ must cause
TRANSITION CHANGE  the next saccade / revision / attention move changes,
                   in the predicted direction
      ↓ while
IRRELEVANT ALPHA   perturbing a state NOT claimed must remain SILENT
```

**This is the workspace's own can-fire + can-stay-silent twin** (`CLAUDE.md`
§ falsifiability rule) applied to introspective state rather than to a guard.
A claimed-alpha that cannot move the transition is the watchdog that cannot
bark; one that moves it from everywhere is the guard that fires on everything.

### The target is already shipped and already declares its own null

`contract::dispatch_mode` elects a `Domain` → `DispatchMode` per dispatch, from
exactly the states an alpha claim would name:

| `Domain` | measured condition | `DispatchMode` |
|---|---|---|
| `Confused` | required marker NaN / no candidates | `FieldGather` |
| `Chaotic` | gate BLOCK (`sd > SD_BLOCK`) + high surprise | `Stabilize` |
| `Complex` | a contradiction is bound (`dissonance > 0`) | `Sweep` (contradiction PRESERVED) |
| `Clear` | gate FLOW + low surprise + no contradiction | `Saccade` (`select_tactic`) |
| `Complicated` | everything else | `Sweep` |

with `DkPosition::MountStupid` **vetoing** the `Clear` election. Three
properties make this the right first target:

1. **The transition is typed and deterministic** — `DispatchMode` is the "next
   saccade" the probe predicts, not a vibe.
2. **The election is computed per dispatch and never stored** — so an
   intervention cannot be masked by cached state.
3. **The silent arm has a PRE-DECLARED expected answer**: the module states it
   reads *"only LOGICAL markers + optional MUL — never qualia (qualia is
   additive stakes, not logic)"*. So perturbing qualia MUST leave the election
   unchanged. The null is not invented for the probe; the code already
   committed to it, which means the probe can falsify a documented claim.

## 1b. The receipt geometry — and the two axes that must not fold together

> **The 8×10 is the receipt. Do not graph it. And do not confuse rung with
> semantic grain.**

Distilled shape (ratified 2026-08-26):

```text
logical geometry   domain × reflective rung
content            stable ontology / world state          ← the picture
alpha              active participation mask              ← which of it is cognition NOW
saccade            a transition of that participation
receipt            sparse typed record of the transition
views              admission / budget / influence / ladder projections
causal falsifier   intervene on alpha, test the next transition
world causality    remains CE64
epistemic permit   remains CE64 61..63
Sandbox            remains Counterfactual + Revision
```

The Photoshop reading is exact and worth keeping: **the ontology is the picture;
alpha is which pixels participate in the composite.** Content is stable; the
mask is the claim; the saccade paints and erases.

### M1 — the rung axis is shipped; the domain axis is not, and its name is taken

- **`RungLevel` is exactly ten variants, 0..9** (`cognitive_shader.rs:159-168`:
  Surface, Shallow, Contextual, Analogical, Abstract, Structural,
  Counterfactual, Meta, Recursive, Transcendent). The reflective-elevation half
  of the lattice is real, ordinal, and already carries the one-way visibility
  doctrine from the merged rung work.
- **There is no shipped 8-valued ontology axis**, and — measured — **`Domain` is
  already taken**: `dispatch_mode::Domain` is the *Cynefin* sense-making domain
  (`Clear / Complicated / Complex / Chaotic / Confused`, five variants) in the
  very module this probe targets. Using "domain" for the ontology/lens axis
  would be the fifth same-word collision this arc has caught (two
  `GateDecision`s, `MulGateDecision`, five `TrustTexture`s). **The ontology axis
  needs its own name before it is written down anywhere normative** — and it is
  hypothesis, not inventory.

### M2 — 80 is a logical address, never 80 resident objects

`α[lens, rung]` is a useful *address*. It does not imply eighty independently
allocated resident channels. A sparse/SoA receipt —
`(receipt_id, lens, rung, mask, payload, surprise)` — satisfies the lattice
exactly, and the ABI views are **queries over those columns**. Materialising
the cells as a population would collapse the receipt back into an invented
graph, i.e. §0's law violated from the inside.

### The four-coordinate admission test (stronger than the source's three)

The source's *Quantity* axis means **semantic grain** (*infection probability*
vs *infection risk* vs *case count*). The rung means **reflective elevation**
(surface … meta … recursive). **These are orthogonal, and mapping quantity onto
rung folds two axes into one.** Octopus therefore admits on four coordinates —
the fourth is a dimension the source does not have:

| coordinate | asks |
|---|---|
| **Entity** | same lens / ontology region? |
| **Grain** | same semantic quantity / abstraction grain? |
| **State** | compatible polarity? (conflict = hard kill) |
| **Rung** | is this cognitive level permitted to consume / project this evidence? |

`r2 finding ≠ r7 differential` is a **rung** mismatch, not a quantity one; two
r2 findings can still differ in grain. Both tests must be able to fire
independently, or the fold has happened silently (see F-OCT-11).

### The cross-rung intervention law (bank this)

One-way visibility — higher rungs read the composite of lower, lower cannot see
up — is not a rendering convention if it is causal. That is testable, and it is
a **stronger HOT falsifier than textual ablation**:

```text
higher rung claims dependence on the lower composite
        ↓ intervene BELOW
higher state must respond in the predicted way
```

with the directionality dual:

```text
lower intervention  → higher changes    proves UPWARD causal dependence
higher intervention → lower unchanged   preserves ONE-WAY visibility
```

A higher alpha that claims dependence and does not move when the composite
beneath it is rewritten is decorative — structured elevation with no causal
job. Intervening higher may change what is *reported* about lower; that is
readout, and it does not falsify the lower claim.

### Two corrections to the naive probe

**(a) Contribution ≠ necessity.** `do(Exact) MUST change the next saccade` is
too strong: when several alpha claims *jointly* determine a transition, a
genuinely causal cell may not be individually necessary, and calling that
redundancy "decorative" is a false positive. The relevant intervention must
therefore either

- **(A)** change the deterministic next transition, **or**
- **(B)** measurably move its score / rank / probability in the *predicted
  direction*,

while the matched irrelevant cell does neither. This yields the clean triple —
**alpha intervention → contribution · counterfactual removal → necessity ·
successful insertion → sufficiency** — with necessity asked separately by the
walker's removal test (F-OCT-8) rather than smuggled into the contribution
test.

**(b) Silence must be scoped, or the test becomes the homunculus.** A cell
classed `None` **for target T** must have no causal effect **on T, with the
orthogonal modulators frozen** — it may still legitimately touch
Flow/Homeostasis, `ThinkingStyle`, resonance, or a later revision without
participating in *this* saccade. So the arm reads *"None for T → no effect on
T"*, never *"no effect anywhere in cognition"*. Freezing the orthogonal paths is
part of the protocol, not an afterthought.

### ABI restraint

These operations are Rust-level probes. The path to a stable membrane symbol is

```text
Rust cognitive implementation → measured stable primitive
   → an actual cross-language consumer? → ABI
```

never *interesting architecture → ABI symbol*. A falsifier surface promoted
early becomes permanent membrane API for an experiment that may not survive its
own test.

## 2. Two proofs, not one (the replay/localize split)

> **replayable ≠ diagnostically complete**

Alpha owes two separate proofs:

```text
can REPLAY thought          (a trace exists and re-runs)
        AND
can LOCALIZE where thought went wrong   (the earliest lost stage is nameable)
```

Operator-supplied provenance (D²ACCI / Mi-Memory): an evidence-preservation
ladder — `ingestion_gap → retrieval_gap → filtered → generation_error` — where
a fix that cannot name the earliest lost stage is not a diagnosis. **The steal
is the LADDER, not the memory stack.** The workspace translation is
architecture-side and owes its own naming: attention landing / saccade /
revision / unresolved surprise. **No resident field is added.**

## 3. The evaluation stack

Operator-supplied provenance (CausalPhys): rationales are scored against a
reference causal graph — Entity Faithfulness, Relation Awareness, Description
Correctness — and the reported gap is the point: **models name the right
entities far more reliably than they state the right edges.** The graph-level
sibling of a decorative trace.

```text
"the model knows the nouns"  ≠  "the model knows what causes what"
```

The Octopus stack, with the last three the departure from the cited work:

| metric | question |
|---|---|
| Entity Faithfulness | did attention land on the right things? |
| Relation Faithfulness | did cognition traverse the right causal edges? |
| **Interventional Faithfulness** | did the claimed internal state actually drive the next transition? |
| **Epistemic Faithfulness** | were the 59–60 topology and 61–63 permission readings appropriate? |
| **Counterfactual Necessity** | does removing the proposed mediator damage the explanation as predicted? |

### Why this can be computed natively, without an LLM judge

The cited evaluation must ask a judge whether prose reflects a graph element.
If the trace is a receipt — visited `NodeGuid`, traversed `CausalEdge64`,
`ReasoningBand`, `TruthValue`, the alpha claim, the counterfactual/revision
outcome — Relation Faithfulness is set arithmetic over the receipt:

```text
|required causal edges ∩ actually traversed/admitted edges| / |required edges|
```

and it refines with the field we already have, distinguishing cases textual RA
cannot express: *correct relation, wrong epistemic status* · *correct relation,
associative band only* · *correct relation at causal band* · *correct mediator,
failed counterfactual removal*.

**Level ladder, kept distinct:**

```text
cited work   "does your CoT contain the causal graph?"
#1057        "do you know whether the graph you traverse is grounded?"
this plan    "is your claimed internal state load-bearing,
              and is your traversal faithful to what it claims?"
```

## 3b. Four construction ontologies — methods do NOT transfer across rows

Four different objects get called "a causal CoT graph". Mixing them is exactly
how a second canonical graph population gets stood up by accident.

| Ontology | Nodes | Edges | Built from | Role |
|---|---|---|---|---|
| **World DAG** | domain variables | claimed cause→effect | query / corpus / KG | reason *about* the world |
| **Reasoning DAG** | CoT steps / log acts | generation or inheritance | a recorded trace | diagnose *this run* |
| **Attribution graph** | tokens, features, logits | circuit influence | internals of one step | verify computation |
| **Scoring graph** | expert objects/events | gold causal edges | annotators, not the model | grade a trace, never build one |

**A construction method does not transfer between rows without changing what
the graph IS.** In this workspace the World DAG is the resident substrate
(AriGraph / SPO / `CausalEdge64`); a Reasoning DAG may exist only as a
*receipt or projection* over it; an Attribution graph is the wrong layer for
this falsifier (real, but orders of magnitude too expensive); a Scoring graph
grades and never constructs.

### Where Octopus sits among the neighbours

```text
Graph-of-Thoughts   thought ─used-by→ thought      = graph OF reasoning
Graph-CoT           reasoning ─walks→ existing KG  = reasoning OVER a graph
Causal-CoT          construct / complete / verify a causal graph while reasoning
Octopus             reason OVER resident causal topology through domain lenses,
                    while making the traversal epistemically typed and
                    interventionally testable
```

The ontological difference is in what an edge *means*: a Graph-of-Thoughts edge
says *thought B depended on thought A*; an Octopus edge says *the system claims
A causally relates to B under a specific CE64 reading*. Those are different
objects wearing the same arrow.

**Never say "Octopus is a Graph of Thoughts."** That silently moves the
canonical graph from the world to the thoughts — precisely the second
population the law forbids.

### Steal / leave

| Steal | Leave |
|---|---|
| **Target-aware growth constraint** (Exact / CloseHit / Bridge / None + Top-K frontier) as a **view filter** over existing receipts | a resident DAG grown by unconstrained completion, as a new population |
| **Hard shape constraints** (DAG, one root, one sink, critic grounded against the log) as a **projection checker** | the per-trace graph object as canonical storage |
| **"CoT as a query plan over a graph you already have"** — completion by retrieval, not invention | harvesting a fresh domain KG per domain |
| **Edge-level intervention** (relevant perturbation moves the next saccade; irrelevant stays silent) | do-calculus theatre over invented mediators |
| **Relation-Awareness as a readout** — a graph can name the right nodes and still miss the edges | scoring graphs used as construction |

The reusable contribution of the construct/complete/verify pipeline is its
*shape*, not the graph it emits. Mapped onto alpha:

```text
EXTRACT     claimed alpha / surprise states from the receipt
COMPLETE    ONLY via existing views — no invented mediators
CONSTRAIN   target = the next predicted transition; prune what does not align
VERIFY      do(alpha) changes that transition; do(irrelevant alpha) does not
CONTROL     same topology, shuffled edges — must lose
```

That is the pipeline with its construction stages **demoted to views** and its
verification stage **kept as the falsifier**. Anything that materialises a
second graph population is a regression against the architecture #1055–#1057
just finished.

**The control is ours.** Per the operator survey, no paper in the surveyed set
runs a randomized-edge / shuffled-topology null. That absence is why F-AIF-5
exists and why "graph traversal improved attribution" is not evidence that the
topology carried the information.


## 3c. Admission vs budget — the three-axis filter is the load-bearing steal

> **Placement (ratified 2026-08-26): the NATIVE home of this material is the
> Sudoku walker, `entropy-closure-causal-ground-v1` §4b-i..iv / D-ECG-7** — its
> `A → ? → C` hole is the literal analogue of the source's `X → ? → Y_b`, and
> the ADMIT/BUDGET/PERMIT/TEST pipeline is specified there. What stays *here* is
> the second, alpha-specific application below: re-targeting the same admission
> grammar at the **next predicted transition** makes the two probe arms
> derivable rather than hand-assigned. Read the walker plan first; this section
> does not restate its rules.

*(Numbers operator-supplied, UNVERIFIED in session — see the header caveat.
They are used here as a lower bound and a shape, never as our measurement.)*

Target-aware growth is **two stacked filters, not one**, and only the first
carries the result:

```text
candidates
   │
   ▼
[1] FGVA  three-axis alignment   ← ADMISSION.  hard class; None discarded,
   │                               never enters frontier or visited-set
   ▼
[2] TAP   Top-K rank             ← BUDGET.     soft, among survivors only
   │
   ▼
frontier
```

Reported ablation (WIQA), read by **both** columns:

| component | variant | Acc | Path Reach |
|---|---|---|---|
| TAP | target-aware rank | 67.92 | 96.23 |
| TAP | **random prune** (same K) | 64.60 | **97.17** |
| FGVA | 3-axis alignment | 67.92 | 96.23 |
| FGVA | **simple alignment** | 55.66 | **14.62** |

**Random pruning keeps reach and costs ~3pp.** Ranking does not win by covering
more — it wins by which slots get filled. **Dropping the three axes destroys the
graph** (reach 96 → 15). Admission is load-bearing; budget is secondary.

### The three axes and the class rule

| axis | values | what it tests |
|---|---|---|
| Entity | Exact / Partial / None | the same object, or merely an overlapping topic |
| Quantity | Exact / Subset / Agg / None | the same *grain* |
| State | Match / Conflict / None | polarity compatible with the move |

```text
Exact     ← entity Exact   ∧ quantity Exact          ∧ state Match
CloseHit  ← entity Exact   ∧ quantity ∈ {Subset,Agg} ∧ ¬state Conflict
Bridge    ← entity Partial ∧ ¬state Conflict          (the only expandable class)
None      ← otherwise, INCLUDING any state Conflict   (hard drop)
```

**State-Conflict is a hard kill even when the entity matches** — the right
object with the wrong polarity is not a near-miss, it is noise. That is the
silent-irrelevant arm arriving by a different route.

### Mapped onto alpha: the target is the TRANSITION, not a world node

The steal is the admission grammar as a **typed view filter over existing
receipts**. The target is not a world variable; it is the **next predicted
transition** (the `DispatchMode` election of §1).

```text
ENTITY    is this the state the transition is about, or a neighbouring topic?
QUANTITY  the same grain as the claimed surprise, or an aggregate/subset proxy?
STATE     polarity compatible with the predicted move, or in conflict?
```

| class | meaning for alpha | required next test |
|---|---|---|
| **Exact** | the claimed alpha IS the transition's object, grain and polarity | `do(alpha)` **must** change the next saccade — **F-OCT-1** |
| CloseHit | right object, wrong grain | not load-bearing until re-grained; bridge only if the bridge is itself interventional |
| Bridge | partial overlap, expandable as a **view**, never a resident node | may appear in a diagnostic projection; not a new field |
| **None** | irrelevant, or polarity-conflicted | `do(alpha)` **must remain silent** — **F-OCT-2** |

**This is the section's real contribution to the probe:** the two arms of
F-OCT-1/F-OCT-2 become **derivable from a typed admission test** instead of
hand-assigned by whoever runs the experiment. That removes the degree of
freedom where "irrelevant" is chosen after seeing the result — the class
predicts the arm, and a misprediction is itself a finding.

### Two operational rules

1. **FGVA before TAP.** Top-K without the three-axis class reproduces "simple
   alignment", and the topology does not reach the target. Admit first, budget
   second.
2. **The published control is weak, so ours stands.** Random-prune randomizes
   *which survivors take the K slots*, not whether the edge set means anything —
   and it moved only ~3pp. That is **not** enough to bank a topology as earned.
   **F-OCT-4 keeps its stronger form** (same receipts, shuffled edges/class
   labels, the predicted-transition effect must collapse) and now carries a
   concrete floor: *a claimed topology must beat a K-slot shuffle by more than
   ~3pp*, since less than that is within reach of pure budget randomization.

### Steal / leave

| Steal | Leave |
|---|---|
| three-axis admission (entity / quantity / state) as a typed view over receipts | a second LLM judge as the ranker |
| `None` = hard drop, state-conflict included | CloseHit→target bridging as a new edge type |
| `Exact` = terminal for THIS diagnostic hop, not a stored node | the BFS graph population and its depth-D materialization |
| the ~3pp random-prune figure as a **lower bound** for any topology claim | their benchmark score as a target |

### Limits to carry forward

- **Both filters are LLM judges, so neither tests load-bearingness.** A node can
  look `Exact` and still be causally decorative — the redaction critique applies
  to *the admission test itself*, which is exactly why the interventional arm
  sits on top of the class rather than replacing it.
- **Pruning is irreversible** (visited-set injected as an avoid-list); a good
  candidate ranked just below K is gone. Reported reach says that is rare on a
  short-horizon benchmark — not evidence for a long-tail domain.
- **Parametric bound**: if the target is outside the judge's world knowledge,
  real mediators get marked `None` and fluent irrelevance ranks in. Same
  world-knowledge bottleneck the construct/complete/verify line hit.

Constrained completion is the steal; the graph it produces is not.

## 4. Three banking rules

1. **Falsifier, not a memory subsystem.** The ladder is a stage-ablation test
   over receipts that already exist. Do not stand up a store to hold it.
2. **Topology must earn itself.** A diagnostic graph may be a useful *derived
   view*; that graph traversal improves attribution does NOT confer canonical
   status. Any such view must beat a **shuffled-edge null** — a control the
   cited work does not supply, so it is ours to run. No second canonical graph
   population. (Weighed against the semantic-locality negative result already
   on the board.)
3. **Tails stay in the test receipt.** A tail/quantile-sensitive independence
   test can change the *evidence supporting* an edge; it never changes the
   *meaning* of `CausalEdge64`. Evidence-side, not encoding-side.

**The constructive dual, deliberately NOT imported:** a
sufficiency/necessity-based CoT optimizer prunes low-necessity steps and adds
missing sufficient ones — that is how you *construct* load-bearing traces. This
plan is how you *falsify* that a recorded trace already is one. Keep the
falsifier; do not import the optimizer. Mixing them makes the test grade its
own generator.

## 5. Deliverables

| D-id | Deliverable | Gate |
|---|---|---|
| D-AIF-1 | receipt schema over EXISTING artifacts (visited node, traversed edge, band, truth, alpha claim, CF/revision outcome) — a read, not a new store | F-AIF-1 |
| D-AIF-2 | the alpha intervention probe against `DispatchMode` election: perturb-claimed → predicted change; perturb-qualia → silence | **F-AIF-1, F-AIF-2** |
| D-AIF-3 | stage ladder (landing / saccade / revision / unresolved surprise) as an ablation test; earliest-lost-stage nameable | F-AIF-3 |
| D-AIF-4 | native Relation Faithfulness from the receipt + its CE64 refinements | F-AIF-4 |
| D-AIF-5 | shuffled-edge null harness for any derived diagnostic view | F-AIF-5 |
| D-AIF-6 | naming decision for external prose (§7) | none — editorial |

## 6. Falsifiers

| id | falsifier | fails when |
|---|---|---|
| **F-AIF-1** | perturbing a CLAIMED alpha state (surprise / dissonance / gate / DK) changes the elected `DispatchMode` in the predicted direction | the claim is decorative — the state is instrumentation |
| **F-AIF-2** | perturbing qualia (declared unread) leaves the election UNCHANGED, on non-trivial inputs | the module's own documented boundary is false, or the probe is coupling through a side channel |
| F-AIF-3 | on a seeded failure, the ladder names the earliest lost stage; a replay that cannot localize FAILS the test even when it reproduces exactly | replay is being sold as diagnosis |
| F-AIF-4 | Relation Faithfulness discriminates: a trace with correct entities but wrong/absent edges scores high on entity, low on relation (both cases in corpus) | the metric is measuring nouns |
| F-AIF-5 | a derived diagnostic view beats its shuffled-edge control by a stated margin | the topology carried no information — the gain was the reader, not the graph |

F-AIF-1/F-AIF-2 are the load-bearing pair, and they are the can-fire/can-stay-
silent twin: **both must be run, and F-AIF-2 must use non-trivial inputs** — an
empty-input silence proves nothing.

## 6b. Falsifier namespace reconciliation — F-OCT is canonical

The operator-issued audit spec (`.claude/prompts/OCTOPUS_AUDIT_MEASUREMENT_FIRST.md`,
filed 2026-08-26, **not yet run**) carries a broader falsifier set, F-OCT-1..10.
**That is the canonical namespace.** This plan's F-AIF ids are aliases into it,
not a competing set — per the audit's own standing rule, *if a current plan
already captures a point, cite/update it rather than opening duplicate
architecture*.

| this plan | F-OCT | note |
|---|---|---|
| F-AIF-1 | **F-OCT-1** | relevant alpha perturbation moves the next transition — in the **(A) changes it / (B) moves its score in the predicted direction** form (§1b(a)) |
| F-AIF-2 | **F-OCT-2** | matched irrelevant perturbation stays silent — **scoped to target T with orthogonal modulators frozen** (§1b(b)) |
| F-AIF-3 | **F-OCT-5** | stage masking exposes localization dependence |
| F-AIF-4 | **F-OCT-3** | right entities + wrong edges must fail despite a right answer |
| F-AIF-5 | **F-OCT-4** | topology must beat a shuffled-edge null preserving counts |

F-OCT ids with no F-AIF alias are carried by their own plans and are NOT
re-opened here: **F-OCT-6** (twin-basin settlement) = `F-ECG-1` in
`entropy-closure-causal-ground-v1`; **F-OCT-7** (band permission) = `F-ECG-6`;
**F-OCT-8** (counterfactual necessity) = `F-ECG-7`; **F-OCT-9** (unknown may
remain unknown) is the #1057 §4b `KEEP UNKNOWN` outcome; **F-OCT-10** (no second
world) is §0's law plus §3b's ontology rule.

**New ids opened by §1b, owned here:**

| id | falsifier | fails when |
|---|---|---|
| **F-OCT-11** | rung and grain fire independently: two same-rung candidates differing only in grain must classify differently, and two same-grain candidates differing only in rung must classify differently | the two axes were folded into one |
| **F-OCT-12** | intervening on the lower composite moves a higher rung that claimed dependence on it | the higher alpha is decorative elevation |
| **F-OCT-13** | intervening on a higher rung leaves the lower unchanged (one-way visibility holds causally, not merely by convention) | the hierarchy leaks upward — visibility was a rendering choice |

So the whole F-OCT set already has a home across three plans, with no duplicate
population and no orphan. A session running the audit reports against F-OCT ids
and updates the owning plan.

## 7. Naming (editorial, D-AIF-6)

`Octopus` has **zero occurrences in this workspace** (measured: `grep -rn -i
octopus` over `*.rs` / `*.md` / `*.toml` → 0 hits), so it is free as the
internal architecture name. It is crowded externally per the operator survey —
including at least one collision that is close *in kind* (multi-capability
reasoning orchestration), which is worse than a distant one.

Recommendation: keep **Octopus** internal; use a descriptor externally.

**The genus is resolved (2026-08-26): Graph-CoT, not CoT, and never GoT.**
Plain "Chain-of-Thought" names a linear textual rationale and invites
benchmarking on rationale quality, which is not what this does.
"Graph-of-Thoughts" is worse than imprecise — it relocates the canonical graph
from the world to the thoughts, violating §0's law. **Graph-CoT** — reasoning
*over* a graph that already exists — is the genus matching the doctrine that
completion is retrieval, not invention.

Working external form:

> **Octopus: a domain-conditioned causal Graph-CoT whose reasoning receipt is
> itself tested for interventional faithfulness.**

The trailing clause is the part no neighbour claims, and it is what D-AIF-2
must actually earn.

## 8. Out of scope

- No new resident field, no new store, no second canonical graph population.
- No import of the constructive optimizer (§4).
- No change to `CausalEdge64` encoding from evidence-side tests (§4 rule 3).
- Sanctioned non-answers (`KEEP UNKNOWN` / `ASK FOR MEANS`, #1057 §4b) remain
  outcomes, and a metric that penalises them as failures is measuring the wrong
  thing — noted so the evaluation stack does not silently re-import
  answer-completion as the target.
