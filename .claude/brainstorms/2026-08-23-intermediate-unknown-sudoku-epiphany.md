# IntermediateUnknown as a constraint-addressed reasoning state

> **Status: BRAINSTORM / architecture synthesis.**
>
> This note does not allocate new bits, widen CausalEdge64, redefine the ratified 59..63 contract, or mint a new reasoning-band value. It records an architectural epiphany: the information density of `IntermediateUnknown` comes from topology and inherited constraints, not from storing a description of the missing thing inside the two MUL bits.
>
> Read after PRs #988, #989 and #990.

---

## 0. One sentence

`IntermediateUnknown` is dense because it can mean:

> **there is a missing intermediary here; its location is known; its parent/HHTL ancestry constrains what may occupy it; the ontology and surrounding causal graph provide the Sudoku; reasoning is allowed to search, falsify and revise before any stronger reading is admitted.**

The two bits do not contain the answer.

They point at a graph position whose surrounding structure carries the answer constraints.

---

## 1. The four-state MUL reading

The useful distinction is not a generic confidence ladder.

It is a structural distinction between four different computational situations:

```text
DIRECT
A ─────────────→ C

INDIRECT / INTERMEDIATE KNOWN
A → B → C
    ↑
    intermediary identified

INDIRECT / INTERMEDIATE UNKNOWN
A → ? → C
    ↑
    missing intermediary is located but unresolved

UNKNOWN
A  ?  C
```

Conceptually:

| state | what is known | what remains to do |
|---|---|---|
| Direct | relation is available without a missing intermediary | consume / validate / continue |
| IndirectIntermediateKnown | an indirect path exists and the intermediary is identified | traverse / validate the known path |
| IndirectIntermediateUnknown | the indirect path structure is known, but the intermediary is missing | solve the constrained gap |
| Unknown | even the useful path structure is not established | acquire enough structure before claiming a localizable gap |

The important property is that `IndirectIntermediateUnknown` preserves **structured ignorance**.

It is neither `null`, nor `0`, nor low confidence, nor a guessed mediator.

---

## 2. The epiphany: the hole is an address into constraints

A Sudoku square contains no explicit digit, yet row, column and box may constrain it to one value.

The same idea applies here.

```text
A → ? → C
     │
     └── the hole itself is tiny
```

But the graph around that hole may already provide:

```text
parent / HHTL ancestry
ontology scope
inherited properties
relation class
source constraints
target constraints
temporal horizon
observed witnesses
provenance
causal neighborhood
sibling mechanisms
current attention focus
current reasoning rung
historical version
```

The information is not duplicated into the two-bit state.

Instead:

```text
IntermediateUnknown
        ↓
read my position
        ↓
read my parent / inherited structure
        ↓
read the ontology and local causal constraints
        ↓
construct candidate space
```

This is the core information-density claim.

The state is small because the semantics live in the topology.

---

## 3. Why "look at my parent" is not hand-waving

The intended operation is not:

```text
IntermediateUnknown
  → invent a mediator
```

It is:

```text
IntermediateUnknown at graph position p
  → resolve parent / inherited path of p
  → collect inherited admissibility constraints
  → intersect with local ontology scope
  → intersect with source/target causal constraints
  → intersect with temporal / provenance constraints
  → derive candidate set
```

Informally:

```text
Candidates(?) =
    inherited(parent(?))
  ∩ ontology_scope(?)
  ∩ relation_constraints(A, ?, C)
  ∩ temporal_constraints(v)
  ∩ witness_constraints(v)
```

No scalar confidence is required to make this useful.

The candidate set may move through:

```text
Universe
  → ontology subtree
  → inherited class family
  → candidate mechanisms
  → {x, y, z}
  → {x, z}
  → {z}
```

Each narrowing step is epistemic progress even before the hole is closed.

---

## 4. HHTL supplies inherited structure, not proof

The role of HHTL in this picture is constraint inheritance / navigation.

It can make a local gap cheap to contextualize because the missing node or relation can inherit the admissible shape of its parent path.

But:

> **HHTL remains a pruning/navigation mechanism, not a proof system.**

The flow is therefore:

```text
HHTL ancestry
  → candidate restriction
  → cheaper search
  → candidate hypotheses
  → causal / temporal / witness / exact validation
```

not:

```text
HHTL ancestry
  → truth
```

The distinction matters because the same inherited structure that makes the Sudoku solvable could also make a wrong parent assumption look deceptively elegant.

Revision must retain permission to falsify the parent.

---

## 5. Horizontal traversal: solve the Sudoku at the same reasoning level

When the hole is genuinely a missing piece of evidence or structure, the first useful move may be horizontal.

Examples:

```text
IntermediateUnknown
  → inspect ontology children / siblings
  → inspect inherited parent constraints
  → inspect alternative causal paths
  → inspect episodic witnesses
  → inspect source/target neighborhoods
  → retrieve external evidence
  → execute deterministic sandbox probe
```

Horizontal traversal asks:

> **Can the missing intermediary be constrained or identified without changing the level of reasoning?**

This prevents a pathological policy in which every unknown causes automatic ascent into increasingly abstract reasoning.

---

## 6. Vertical traversal: question the reasoning that produced the hole

A candidate set can also collapse to nothing:

```text
Candidates(?) = ∅
```

That is not necessarily failure.

It may mean the premise above the hole is wrong.

Then the useful move is vertical:

```text
IntermediateUnknown
  → no admissible candidate survives
  → inspect the assumption that requires the intermediary
  → higher-order reasoning
  → counterfactual branch
  → Revision
  → repair / reject parent hypothesis
```

Vertical traversal therefore means:

> **do not merely fill the blank; question the rules that made the blank necessary.**

This is the difference between constrained search and metacognition.

---

## 7. Counterfactual is the Sudoku eliminator

Once the candidate space is small enough, Counterfactual can test each candidate against the rest of the graph.

```text
A → ? → C

? = X  → does the surrounding causal structure remain coherent?
? = Y  → does the expected downstream path survive?
? = Z  → does temporal / observational evidence contradict it?
```

The useful outcomes are not only winner selection.

### Outcome A: one candidate survives

```text
{X, Y, Z}
  → {X, Z}
  → {Z}
```

The intermediary can become known, subject to the relevant validation gates.

### Outcome B: several candidates survive

```text
{X, Y}
```

The state remains `IntermediateUnknown`, but with reduced uncertainty.

This is still progress.

### Outcome C: no candidate survives

```text
{X, Y, Z}
  → ∅
```

Revision should inspect the parent hypothesis rather than stuffing an arbitrary value into the gap.

### Outcome D: the path itself is not warranted

The correct transition may be from a localized intermediate unknown back to a less committed unknown state.

A healthy system must permit epistemic retreat.

---

## 8. Bits 59..60 and 61..63 remain orthogonal

The epiphany does **not** require storing search progress, reward, mediator identity, or candidate cardinality inside CausalEdge64.

The two MUL bits answer one structural question.

The reasoning-band bits 61..63 answer another.

Conceptually:

```text
59..60
  "what kind of direct/indirect/unknown situation is this?"

61..63
  "under the relevant ClassView, how is this edge currently read by reasoning?"
```

A successful Sudoku completion may eventually justify a change in the 61..63 reading.

For example, if the relevant ClassView has a valid `related` interpretation, a completed and validated path might make that reading admissible.

But this note does **not** claim that `related` is globally encoded at a particular 61..63 value.

The order must be:

```text
localized gap
  → inherited constraints
  → candidate generation
  → counterfactual elimination
  → witness / provenance / temporal validation
  → Revision
  → only then, if licensed by the current ClassView, reasoning-band transition
```

Never:

```text
plausible candidate
  → flip 61..63
```

---

## 9. Precision is the reward, not a reward field

The informational gain can be large while the two MUL bits remain unchanged.

Example:

```text
IntermediateUnknown
  candidate space = unknown huge set
```

then:

```text
IntermediateUnknown
  candidate space = ontology subtree P
```

then:

```text
IntermediateUnknown
  candidate space = {X, Y, Z}
```

then:

```text
IntermediateUnknown
  candidate space = {X, Z}
```

All four states may still use the same two-bit reading.

Yet the system knows vastly more.

This is why the bits should not be overloaded with reward magnitude.

The progress is recoverable from the graph, candidate restrictions, witnesses and trace.

The architecture becomes denser precisely because a tiny state can remain stable while surrounding structure becomes more informative.

---

## 10. Entropy interpretation

The useful entropy story is not "drive uncertainty to zero as quickly as possible".

That would permit:

```text
?
  → guess X
  → certainty
```

which is maximally cheap and epistemically terrible.

The intended sequence is:

```text
high ambiguity
  → structured ambiguity
  → localized ambiguity
  → constrained candidate set
  → discriminating evidence
  → warranted reduction
```

`IntermediateUnknown` is valuable because it lets the substrate hold a low-description-length state for a high-dimensional unresolved problem without collapsing the problem prematurely.

In other words:

> **the substrate earns entropy reduction by converting vague uncertainty into topologically constrained uncertainty before commitment.**

---

## 11. MUL: do not confuse located ignorance with low competence

A system that says:

```text
A → ? → C
```

and correctly identifies the inherited/ontological constraints on `?` may be more trustworthy than a system that emits a high-confidence candidate immediately.

MUL should therefore treat explicit `IntermediateUnknown` as a potentially healthy epistemic state.

The suspicious event is not "unknown remains unknown".

It is:

```text
IntermediateUnknown
  → strong commitment
while
new witness = none
counterfactual discrimination = none
parent validation = none
provenance gain = none
```

That is premature closure.

The desired behavior is:

```text
trust insufficient
  → keep hole explicit
  → choose next discriminating action
  → acquire evidence
  → Revision
  → update warranted trust
```

---

## 12. Rubicon / -550 .. -220 .. 0 mapping

This state has a natural place in the commitment timing model.

```text
-550
multiple candidate continuations can be active

      ↓

-220
Revision checkpoint

questions:
- is the intermediate still unknown?
- did horizontal search reduce the candidate space?
- did vertical reasoning invalidate the parent?
- did counterfactual eliminate candidates?
- did a witness arrive?
- is the current ClassView reading now warranted?

      ↓

0
commit / externalize only if the remaining state is sufficiently warranted
```

The timing is an engineering model for staged commitment, not a claim to solve human free will.

The architectural point is that an `IntermediateUnknown` can survive the checkpoint without being treated as failure.

---

## 13. Alpha channel: make the Sudoku visible without contaminating the object

The object graph should not be rewritten merely because the system is currently thinking about one of its holes.

Instead the ephemeral alpha layer can show:

```text
object graph:
A → ? → C

alpha / thought residue:
- parent inspected
- ontology branch P expanded
- candidates X,Y,Z considered
- Y rejected by temporal witness
- X rejected counterfactually
- Z remains
- Revision pending
```

A higher rung can then inspect the shape of this traversal.

That gives a concrete meaning to "thinking about thinking":

```text
rung n:
  solve the missing intermediary

rung n+1:
  inspect where rung n looked,
  what it ignored,
  where it narrowed,
  and whether the narrowing was warranted
```

The alpha channel is therefore metacognitive cartography, not ontology truth.

Counterfactual branches can remain visible as faint alternative trajectories without becoming asserted world-state.

---

## 14. Owner-locality and historical horizon

Every execution or trace that acts on an `IntermediateUnknown` must remain bound to the thought that owns it.

At minimum future receipts need enough identity to prevent accidental mixing across:

```text
mailbox owner
historical version
branch / counterfactual world
rung
focus / ontology scope
```

The no-hindsight rule is essential.

If a mediator is discovered at version `v20`, replaying reasoning at `v10` must not silently make the mediator available.

A correct historical replay should still see:

```text
v10: IntermediateUnknown
```

if that is what was knowable then.

This is what allows later learning to distinguish bad reasoning from good reasoning under genuinely missing information.

---

## 15. Transition table

A useful conceptual transition table is:

| current | event | next | meaning |
|---|---|---|---|
| Unknown | path structure localized | IntermediateUnknown | the hole now has coordinates |
| IntermediateUnknown | candidate space narrowed | IntermediateUnknown | more precision, same structural state |
| IntermediateUnknown | intermediary validated | IntermediateKnown | missing node/path component identified |
| IntermediateKnown | relation collapses to validated direct representation | Direct | indirection no longer needed in the relevant reading |
| IntermediateUnknown | parent assumption falsified | Unknown or revised structure | epistemic retreat / restructure |
| IntermediateUnknown | evidence insufficient | IntermediateUnknown | stay unresolved, choose another epistemic action |
| any | hindsight-only evidence appears | unchanged under historical replay | temporal firewall holds |

This table should be treated as architecture intent until code truth confirms the exact current transition surfaces.

---

## 16. What a future interpreter trace should record

PR #989 identifies the missing interpreter/trace as the practical blocker for learned behavioral macros.

For this epiphany, a future trace should make the Sudoku observable without bloating CausalEdge64.

Candidate receipt shape:

```text
EpistemicStepReceipt {
    owner,
    historical_version,
    branch,
    rung,
    focus,

    mul_before,
    reasoning_band_before,

    operation,
    operand_scope,

    candidate_count_before?,
    candidate_count_after?,
    witness_refs,
    falsified_parent: bool,

    mul_after,
    reasoning_band_after,
    outcome,
}
```

Question marks are deliberate: candidate cardinality may be derivable or too expensive to persist and should not be minted without measurement.

The critical thing is that the trace can answer:

> **what action made this hole more precise?**

---

## 17. Learning target

Once real traces exist, the learner should not ask only:

```text
which sequence led to an answer?
```

It should ask:

```text
given this shape of IntermediateUnknown,
which sequence most often increased warranted precision?
```

Examples of potentially learnable behaviors:

```text
ontology parent → sibling scan → causal path check

witness lookup → temporal replay → Revision

counterfactual X/Y/Z → eliminate → sandbox probe

parent falsifier → vertical escalation → rebuild candidate frame
```

A learned macro would therefore be a reusable research behavior, not a replacement for causal truth.

The primitive expansion must remain reversible and versioned.

---

## 18. Falsifiers

### F1 — STRUCTURED-UNKNOWN-NOT-USEFUL

Construct cases where `IntermediateUnknown` is correctly localized.

Compare against a generic `Unknown` baseline.

**KILL** the epiphany if parent/HHTL/ontology context does not reduce search work or improve localization relative to the generic baseline.

### F2 — PARENT-DOES-NOT-CONSTRAIN

For a known missing-link fixture, remove or scramble the parent/HHTL inheritance.

**KILL** the "look at my parent" claim if candidate generation is unchanged.

### F3 — SUDOKU-OVERFITS

Shuffle sibling/ontology constraints while preserving superficial counts.

**KILL** if the same candidates survive at similar rates.

### F4 — COUNTERFACTUAL-DOES-NOT-DISCRIMINATE

Create at least two candidates that satisfy local type constraints but differ in downstream causal consequences.

**KILL** if counterfactual evaluation cannot separate them.

### F5 — PARENT-FALSIFICATION-IMPOSSIBLE

Create a fixture where no valid mediator exists because the parent causal assumption is wrong.

**KILL** the metacognitive claim if the system can only keep searching horizontally and cannot surface the parent as the thing requiring Revision.

### F6 — PREMATURE-BAND-WRITE

Ensure no 61..63 transition occurs merely because a candidate is plausible.

**KILL** if reasoning-band state advances before the required validation/witness/Revision boundary.

### F7 — HINDSIGHT-LEAK

Mediator absent at `v10`, discovered at `v20`.

Replay `v10`.

**KILL** if the replay sees the `v20` mediator or upgrades the historical state.

### F8 — CROSS-MAILBOX-CONTAMINATION

Two mailboxes reason over similar gaps with different evidence.

**KILL** if one mailbox's trace, rung or evidence changes the other's admission/transition semantics without an explicit shared object contract.

### F9 — ALPHA-BECOMES-PROOF

Remove or alter the ephemeral attention overlay while preserving object-state evidence.

**KILL** the separation if truth/causal validation depends on what was merely visualized as attention.

### F10 — PRECISION-WITHOUT-CLOSURE-NOT-MEASURABLE

Use a fixture where candidate space narrows significantly but no unique mediator is found.

**KILL** the proposed learning target if the trace cannot distinguish this useful narrowing from complete non-progress.

---

## 19. Smallest implementation probe after #989

Do not start by adding a learner.

The smallest useful experiment is:

```text
PROBE-INTERMEDIATE-UNKNOWN-SUDOKU-1
```

Fixture:

```text
A → ? → C
```

with:

- a known parent/HHTL path;
- a bounded ontology subtree;
- at least three admissible local candidates;
- one candidate eliminated by inherited structure;
- one eliminated only by counterfactual / downstream evidence;
- one surviving candidate;
- historical versions before and after the decisive witness.

Record:

```text
candidate set after each operation
historical version
owner
rung
MUL state
reasoning-band state
witness/provenance references
Revision decision
```

Success requires:

1. the gap remains explicit until evidence warrants closure;
2. parent/HHTL structure reduces the candidate space;
3. counterfactual work eliminates at least one otherwise admissible candidate;
4. historical replay preserves the old unknown;
5. no reasoning-band transition precedes its validation gate;
6. the final state can become more precise without adding a byte to CausalEdge64.

---

## 20. Why this matters for the whole substrate

The architecture has spent a long time making tiny physical states carry precise semantics.

The important achievement is not that two bits can encode four labels.

It is that one of those labels can safely delegate almost all of its meaning to already-existing structure:

```text
2-bit state
  + graph position
  + inherited topology
  + ontology
  + causal context
  + historical version
  + witnesses
  + reasoning machinery
```

The result is not a compressed prose description of ignorance.

It is an executable address into a constrained research problem.

That is the epiphany:

> **`IntermediateUnknown` is not merely a label for missing knowledge. It is a compact invitation to perform bounded epistemic work at a known location in the graph.**

And because the graph can answer "where is the hole?", "what does its parent imply?", "which candidates survive?", "what did the historical self know?", and "which reasoning move reduced the ambiguity?", the system can eventually learn how to solve such holes without confusing learned procedure with truth.

The bits stay boring.

The topology does the thinking.
