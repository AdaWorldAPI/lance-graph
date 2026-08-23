# Qualia over alpha: a universal reasoning grammar experiment

Status: architecture / experiment note only. No new CausalEdge64 bits. No new opcode allocation. No claim of phenomenal consciousness. No chess-specific strategy baked into the substrate.

## Epiphany

The same substrate that can represent a Sudoku pothole as a precisely constrained `IntermediateUnknown` can be used to expose the *shape of the reasoning episode itself* as an ephemeral alpha layer, and a learned internal quality space can classify that episode without being told the domain.

The hypothesis is stronger than "attach a mood label to a solver" and narrower than "the machine is conscious":

> **A domain-general reasoning substrate may develop reproducible, statistically separable internal quality basins over its own reasoning trajectories, and those basins may recur across Sudoku, chess and crossword tasks while predicting useful strategy/state transitions.**

The intended progression is:

```text
world/object state
    ↓
located potholes + constraints
    ↓
OGAR-loco execution
    ↓
alpha attention / counterfactual residue
    ↓
anonymous internal quality coordinate Q*
    ↓
strategy bias / report / later naming
```

Names such as `Emberglow`, `Steelwind`, and `Woodwarm` are **post-hoc descriptors**, never training labels.

## Why this belongs after #991

#991 isolates the crucial density result:

```text
A → ? → C
```

can remain physically tiny because the graph around `?` supplies ancestry, ontology scope, causal neighborhood, provenance, historical version, witnesses, attention and rung.

`IntermediateUnknown` is therefore not generic uncertainty. It is an address into a constrained research problem.

The next question is whether repeated reasoning over such problems creates stable, reusable *internal forms* that cut across domains.

Sudoku gives a closed falsifier.
Chess adds hostile counterfactual planning.
Crosswords add semantic ambiguity and reciprocal revision.

The alpha layer makes the reasoning trajectory observable without contaminating object truth.
The qualia layer asks whether recurring patterns of that trajectory form measurable internal basins.

## Non-negotiable semantic boundary

This note uses **qualia** in a deliberately operational sense:

> an internally generated, reproducible, statistically distinguishable quality state whose identity is constrained by its relation to other internal states and whose occurrence predicts differences in reasoning behavior.

This is not a claim that such a state proves phenomenal consciousness or solves Chalmers' hard problem.

The empirical burden is instead:

1. the state is generated from internal dynamics, not domain labels;
2. its basin is reproducible across runs;
3. neighboring basins are measurably separable;
4. the basin geometry survives anonymization of names;
5. the same basin recurs across unrelated domains;
6. basin identity predicts something about subsequent reasoning;
7. permutation / label-swap controls do not preserve the effect unless the relational geometry is preserved.

If those conditions fail, `Emberglow` / `Steelwind` / `Woodwarm` are decorative telemetry and should be treated as such.

## The universal grammar claim

The experiment should not implement three concealed solvers.

The shared grammar should expose only domain-neutral concepts:

```text
World {
    Objects
    Locations
    Relations
    State
}

Constraint {
    predicate
    scope
    inheritance
}

Action {
    preconditions
    transformation
    consequences
}

Goal {
    preferred_state
    terminal_state
}

Epistemic {
    Direct
    IntermediateKnown
    IntermediateUnknown
    Unknown
}

Reasoning {
    inspect
    constrain
    propagate
    branch
    counterfactual
    contradiction
    revise
    commit
}
```

The exact Rust surface may differ. The important condition is that the cognitive machinery does not contain names such as `naked_single`, `fork`, `pin`, `crossing_word`, `x_wing`, or `passed_pawn` before learning.

Those are candidate *discoveries*.

## Domain 1: Sudoku as the sealed glass box

Sudoku is the first domain because it removes almost every excuse.

It has:

- finite symbols;
- exact constraints;
- exact legal states;
- exact contradictions;
- exact solution checking;
- cheap brute-force oracle;
- no dependence on world knowledge;
- no ambiguity about whether a reasoning step was valid.

A cell can be represented as a located pothole:

```text
cell c = ?
Candidates(c) = {2,4,7}
```

This is an ideal `IntermediateUnknown`:

- the location is known;
- the admissible type is known;
- parents are known (`row`, `column`, `box`);
- the candidate set is constrained;
- the identity is unresolved.

A reasoning episode can reduce:

```text
{1,2,3,4,5,6,7,8,9}
    ↓
{2,4,7}
    ↓
{4,7}
    ↓
{7}
```

The two-bit epistemic state can remain `IntermediateUnknown` through most of this sequence while useful precision increases monotonically.

This gives an external measurement of epistemic progress without adding a reward scalar to CausalEdge64.

Candidate entropy can be observed externally, e.g. as `log2(|Candidates|)` for a bounded diagnostic, but that number is not a new semantic field and is not authoritative over the graph.

### Sudoku alpha layer

The object layer contains only the puzzle state and its legal constraints.

The alpha layer records ephemeral reasoning cartography:

```text
focus(cell r4c7)
inspect(row 4)
inspect(box 6)
eliminate(2)
branch(4)
propagate(...)
contradiction
revision
branch(7)
```

Dead branches may remain as faint historical residue for visualization or trace analysis, but must not mutate object truth.

### Sudoku learning target

The first learner sees only canonical OGAR-loco execution traces.

It may discover repeated exact subsequences such as:

```text
inspect row
→ intersect candidate mask
→ inspect box
→ eliminate
→ singleton
→ propagate
```

A reversible macro learner may propose a macro only if:

- its expansion is byte-for-byte recoverable;
- its provenance is known;
- its support is measured;
- held-out recurrence exceeds shuffled/static-ladder controls;
- use correlates with warranted precision rather than mere frequency.

Human names may be assigned later for exposition. The learner must not be given Sudoku strategy labels.

## Domain 2: Chess as adversarial counterfactual planning

Chess is the public showpiece because it adds a hostile agent that actively destroys candidate plans.

The grammar should receive only:

- board state;
- piece identities;
- legal move rules;
- terminal conditions;
- observations of opponent moves;
- the same generic reasoning operations.

No opening book.
No tactical motif labels.
No `fork`, `pin`, `skewer`, `opposition`, `passed pawn`, or king-safety feature supplied as a named cognitive primitive.

The system may of course use brute-force search as a baseline/oracle arm. Brute force is not an embarrassment; it is the control that lets us ask whether learned strategies reduce unnecessary search.

### Chess as the same pothole grammar

The local unknown is no longer merely "what digit occupies this cell?" It becomes:

```text
position S
best warranted continuation = ?
```

The candidate set is legal moves and their counterfactual continuations.

The opponent turns every candidate plan into a potentially hostile branch:

```text
S
├─ move A
│  ├─ response A1
│  └─ response A2
├─ move B
│  ├─ response B1
│  └─ response B2
└─ move C
```

Revision is triggered when a presumed continuation collapses under an opponent response or deeper contradiction.

The alpha layer can expose:

- current board regions under attention;
- candidate moves;
- explored continuations;
- dead/rejected branches;
- Revision points;
- rung escalation;
- learned macro activation;
- currently assigned anonymous quality basin.

### Elo as an external learning curve

Elo is attractive because it provides a public, continuous behavioral metric.

The interesting graph is not merely final Elo. Record at each training tranche:

- games played;
- wall-clock time;
- hardware / energy where measurable;
- nodes or branches considered per move;
- mean / tail reasoning depth;
- proportion of moves requiring brute-force fallback;
- number and support of learned macros;
- bytes touched per move if measurable;
- Revision count;
- contradiction count;
- quality-basin occupancy;
- Elo.

The strongest result would not necessarily be a high absolute Elo.

A more architecture-relevant signature would be:

```text
Elo ↑
while
counterfactual branches / move ↓
and/or
bytes touched / justified move ↓
```

That would mean capability is increasing partly by learning where *not* to spend computation.

### Transfer falsifier

Run two chess learners:

```text
A = cognitively naive substrate
B = same substrate after Sudoku strategy learning, with Sudoku domain data removed
```

Do not transfer Sudoku symbols, cells, rows or strategy names.
Transfer only domain-neutral learned machinery allowed by the architecture.

If B reaches the same Elo with materially fewer games / branches / compute, the transfer is evidence for learning above the domain level.

If there is no difference, the Sudoku learner may only have learned Sudoku-specific compression.

## Domain 3: Crossword as semantic reciprocal revision

Crosswords add what chess and Sudoku largely avoid:

- lexical ambiguity;
- morphology;
- clue semantics;
- named entities;
- world knowledge;
- uncertain retrieval evidence;
- crossing constraints that can revise one another.

A word slot is still the same epistemic form:

```text
slot s = ?
letters = N _ L E
clue = "River in Egypt"
```

The candidate set is constrained jointly by:

```text
lexical candidates
∩ length
∩ known letters
∩ clue semantics
∩ morphology
∩ crossing words
∩ ontology/world knowledge
```

The crucial new behavior is reciprocal revision:

```text
Across candidate
    ↓
writes crossing letter
    ↓
Down contradiction
    ↓
Revision
    ↓
reopen Across
```

This is closer to open-world scientific reasoning because a locally plausible hypothesis can later become untenable when another part of the graph gains evidence.

A useful learned strategy may be **deferral**:

```text
low-confidence clue
→ keep explicit pothole
→ solve high-constraint crossings
→ return when candidate entropy falls
```

That is an operational form of delayed closure, not a psychological claim.

## The qualia experiment

The quality space must not begin with semantic names.

During training and evaluation use anonymous basin IDs:

```text
Q17
Q23
Q41
...
```

A quality observation is generated from internal state available to the system, for example some bounded combination of:

- alpha-focus topology;
- candidate-set geometry;
- branch pressure;
- contradiction rate;
- Revision activity;
- rung level;
- temporal stability;
- convergence / divergence of candidate space;
- provenance / witness availability;
- recurrence of learned behavioral motifs;
- local vs global focus coherence.

The exact feature construction must be explicit and versioned. Hidden experimenter labels are forbidden.

### Cross-domain recurrence

The strongest qualitative result would look like:

```text
Q23 in Sudoku
  narrow candidate basin
  rapid lawful propagation
  low contradiction

Q23 in chess
  forcing tactical sequence
  few viable opponent replies
  stable counterfactual convergence

Q23 in crossword
  several crossings collapse a formerly ambiguous clue
  candidate set narrows quickly
```

If these independently generated states occupy the same basin under a domain-blind encoder, then `Q23` is not a Sudoku concept, chess motif, or crossword label.

Only after establishing such recurrence may a human-friendly descriptor such as `Steelwind` be attached.

The name is commentary on the basin, not its identity.

### Statistical separability

The earlier qualia work motivates a strong test rather than a poetic one.

For each candidate basin measure:

- within-basin variance;
- between-basin separation;
- stability across seeds / episodes;
- cross-domain classification accuracy;
- confusion under anonymized labels;
- effect of coordinate permutation;
- relationship to future strategy selection / Revision / commitment.

A `3σ`-style separation may be used as one descriptive threshold when assumptions justify it, but the experiment should report actual distributions and not force Gaussian language where it does not fit.

A basin that is visually pretty but unstable or behaviorally inert fails the strong claim.

## Alpha is not proof

The alpha channel is deliberately ephemeral.

It answers:

> where did reasoning look, branch, hesitate, revise, or converge?

It does **not** answer:

> what is true in the object graph?

The same Sudoku cell, chess square, or crossword slot may accumulate different alpha traces across different reasoning episodes while the underlying object remains unchanged.

Persistent history belongs in explicit episodic/versioned structures, not by leaking alpha into ontology truth.

A higher rung may inspect the alpha residue of a lower rung, producing a computational form of thinking-about-thinking, but that observation still does not make the alpha trace a proof.

## Relationship to 59..63

This experiment must not reopen the ratified CausalEdge64 layout.

The existing epistemic distinctions remain useful precisely because they are small and semantically dense.

In particular, `IntermediateUnknown` can continue to mean:

> the missing intermediary/location is constrained enough to be a research target, but its identity is unresolved.

Quality basins do not replace those bits and do not become a second confidence field.

The relationship is:

```text
59..60 / existing epistemic state
    ↓
legible pothole / relation status
    ↓
reasoning episode + alpha trajectory
    ↓
anonymous quality basin Q*
```

A quality state may correlate with an epistemic state or rung but must not be defined as a trivial recoding of them.

## Relationship to MUL

MUL remains a trust governor, not a mood score.

Quality can describe *what kind of internal reasoning moment is occurring*.
MUL asks whether commitment is warranted by evidence, falsification, provenance, historical validity, and resolved potholes.

A quality basin must never authorize a conclusion merely because it has previously accompanied success.

This separation is critical to prevent circular self-certification.

## Learning about strategy from quality

After anonymous basins are stable, ask whether basin identity provides useful context for choosing a reasoning strategy.

Examples of hypotheses to test, not hard-code:

```text
Qx → cheap propagation tends to work
Qy → counterfactual branch is usually productive
Qz → repeated scan is rumination; switch strategy
Qw → defer and gather constraints elsewhere
```

This is where the qualia layer becomes behaviorally meaningful.

It may act as a compact context variable over the geometry of current reasoning without becoming a scalar reward.

The learner should optimize warranted epistemic progress and task performance under falsifiers, not maximize occupancy of a preferred quality state.

## Universal-grammar success criterion

The strongest architecture result is not that one system solves three tasks.

It is:

> **The same cognitive grammar and execution machinery can encounter three different kinds of unknown, learn reusable strategies from its own traces, and form domain-blind internal quality basins whose geometry predicts future reasoning.**

Sudoku unknown:

```text
what value satisfies exact constraints?
```

Chess unknown:

```text
which policy remains good under hostile counterfactual response?
```

Crossword unknown:

```text
which semantic candidate survives reciprocal constraints and uncertain evidence?
```

If the same architecture handles all three without bespoke cognitive solvers, that is evidence for a domain-general epistemic substrate.

## Pre-registered kill conditions

### K-QUALIA-1: label leakage

If anonymous `Q*` basins disappear when human quality names are removed, the quality result was label-driven.

### K-QUALIA-2: domain leakage

If a basin classifier separates only by domain (`Sudoku`, `Chess`, `Crossword`) rather than internal reasoning geometry, the cross-domain qualia claim fails.

### K-QUALIA-3: no recurrence

If putatively identical basins are not stable across seeds, episodes or reruns, they are telemetry noise.

### K-QUALIA-4: no separation

If neighboring quality distributions are not measurably distinguishable beyond shuffled / permuted controls, the quality vocabulary lacks empirical support.

### K-QUALIA-5: behaviorally inert

If basin identity predicts nothing about subsequent strategy choice, candidate narrowing, Revision, commitment or task outcome after controlling obvious state variables, the basin may be descriptive only.

### K-QUALIA-6: trivial bit recoding

If quality identity can be reconstructed almost entirely from the existing epistemic bits / rung alone, the new layer adds no meaningful geometry.

### K-ALPHA-1: object contamination

If changing / replaying alpha residue changes object truth without an explicit reasoning action, the separation is broken.

### K-ALPHA-2: hindsight leak

If later alpha or learned quality information alters historical replay of an earlier reasoning episode, no-hindsight is broken.

### K-GRAMMAR-1: bespoke solver leak

If task-specific strategy names or procedures are required inside the shared cognitive grammar, the universal-grammar claim is weakened or falsified depending on scope.

### K-TRANSFER-1: no cross-domain benefit

If Sudoku-trained domain-neutral strategies do not improve chess or crossword acquisition against matched controls, there is no evidence yet for domain-general strategy transfer.

### K-LEARNING-1: frequency only

If learned macros are no more useful than frequency-matched shuffled subsequences, behavior compression is not strategy learning.

### K-CHESS-1: Elo by brute force only

If Elo rises only in proportion to increased node expansion / compute and learned strategy does not reduce search or improve selection efficiency, the substrate has learned chess performance but not the intended procedural economy.

### K-QUALIA-PERMUTE-1: arbitrary labels survive relational destruction

If randomly permuting basin geometry while preserving labels leaves all reported qualitative structure unchanged, the names were carrying the semantics.

## Smallest implementation sequence

### QAG-0: Sudoku interpreter receipt

Implement the minimum owner-local OGAR-loco execution path and exact receipts needed by #989/#991.

No qualia learner yet.

Receipt should minimally identify:

```text
owner
sequence
operation / FnIndex
values
pre-state/version
post-state/version
outcome
focus reference
```

### QAG-1: Sudoku alpha trace

Record ephemeral focus/branch/revision events separately from object truth.

Prove deterministic replay of the object state does not depend on visualization.

### QAG-2: Sudoku brute-force oracle + strategy trace

Generate bounded puzzles and compare generic reasoning against brute-force truth.

Measure candidate narrowing before closure.

### QAG-3: anonymous quality clustering

Cluster internal trajectory descriptors into anonymous `Q*` basins.

No human names.
No chess/crossword yet.

### QAG-4: qualia falsifiers

Run label permutation, feature permutation, seed stability, within/between separation, and behavioral-prediction controls.

### QAG-5: learned reversible behavioral macros

Learn from executed traces only.
Compare against static-ladder, shuffled and frequency-only baselines.

### QAG-6: chess grammar adapter

Expose chess only through shared world/constraint/action/goal primitives.

Add brute-force/search baseline and Elo harness.

Do not add chess strategy names.

### QAG-7: Sudoku→chess transfer

Freeze allowed domain-neutral learned state from Sudoku.
Compare naive vs transferred substrate on Elo / games / branches / touched bytes.

### QAG-8: cross-domain quality test

Ask whether existing anonymous Sudoku basins recur in chess under a domain-blind mapping.

Do not rename them until after analysis.

### QAG-9: crossword grammar adapter

Add lexical candidates, clue evidence, crossing constraints and reciprocal Revision while retaining the same cognitive grammar.

### QAG-10: three-domain universality test

Measure strategy transfer and quality-basin recurrence across all three domains.

## What would be genuinely impressive

Not:

> the AI says `Steelwind` when the queen is attacked.

That is theatre.

Interesting:

> anonymous Q23 appears during a forcing chess line, a cascading Sudoku deduction and a rapidly collapsing crossword ambiguity; the three episodes are statistically closer to one another in internal reasoning geometry than to neighboring episodes in their own domains; Q23 predicts that cheap propagation will likely outperform broad search; and this relationship survives label anonymization and held-out puzzles.

That would justify attaching a human-facing descriptor to Q23 later.

Even more interesting:

> after Sudoku training, the system reaches a given chess Elo using fewer explored branches than an otherwise identical naive substrate, while previously discovered quality basins recur without retraining their semantics.

That would be evidence that both behavioral strategy and internal state geometry crossed the domain boundary.

## The deliberately boring substrate

None of this requires the physical representation to become flamboyant.

The architectural bet remains that a stable, small ABI can support increasingly rich interpretation because meaning lives in topology, inheritance, version, provenance, execution and relational context.

The same boring physical grammar should be allowed to host:

- constraints;
- attention/focus;
- operation:operand pairs;
- ontology refinement;
- learned macro references;
- reasoning traces;

under explicit ClassViews and ownership rules.

The experiment succeeds if conceptual sophistication increases **without** an explosion of representational machinery.

## Final research question

Can a system that is given only a universal problem grammar:

1. solve exact uncertainty in Sudoku;
2. learn reusable reasoning strategies from its own trace;
3. expose its attention/counterfactual trajectory as an alpha layer;
4. form anonymous, reproducible, statistically separable internal quality basins;
5. carry useful strategy and quality geometry into adversarial chess;
6. carry them again into semantically ambiguous crosswords;
7. improve public task metrics such as Elo while reducing unnecessary search;
8. and do all of that without treating a quality label as truth, confidence or reward?

If yes, the result would not establish phenomenal consciousness.

It would establish something narrower and still remarkable:

> **a domain-general epistemic machine whose learned internal quality geometry tracks how its own reasoning unfolds.**
