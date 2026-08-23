# Self-learning self-programming endgame — precision before commitment

> **Status: BRAINSTORM / architecture synthesis.** This document does not mint a type,
> allocate a bit, redefine the 59..63 reading, or authorize runtime self-modification.
> It states the intended endgame and the falsifiable control loop that later work must earn.
>
> Read together with #988 (learning hypotheses) and #989 (behavioral micro-IR fathoming).
> #989 asks whether the cognitive bytecode can be executed and traced. This document asks:
> **if that trace exists, what is it ultimately for?**

---

## 0. Endgame

The target is not “add BPE”, “add RL”, or “add a neural policy”.

The target is a substrate that can eventually:

1. observe its own reasoning and execution;
2. represent what was known, unknown, inherited, observed, mechanical, epistemic, and historical;
3. keep uncertainty precise without forcing premature closure;
4. discover recurring useful behavior over its own canonical execution trace;
5. propose reversible new programs or macro-programs;
6. test those proposals under historical replay, counterfactual alternatives, provenance and causal constraints;
7. acquire missing evidence through tools or sandboxed experiments when trust is insufficient;
8. revise rather than bluff;
9. promote only changes whose improvement is warranted;
10. preserve the previous self, the rejected branches, the evidence and the rollback path.

In compact form:

```text
experience
   ↓
episodic / causal / reasoning trace
   ↓
canonical behavioral IR
   ↓
pattern discovery / candidate program
   ↓
historical + counterfactual replay
   ↓
MUL trust / provenance / causal validation
   ↓
insufficient trust? ── yes ──→ acquire evidence / sandbox / Revision
   │
   no
   ↓
bounded promotion
   ↓
new behavior becomes available
   ↓
observe again
```

This is the intended meaning of **self-learning self-programming AGI-aspiring substrate**.
It is not a claim that the present repository is AGI.

---

## 1. The central epistemic rule: precision is progress even when the answer remains unknown

The substrate must not reward premature commitment merely because closure is cheap.

A useful research state is often:

```text
A → ? → C
```

where the missing link is not vague ignorance. HHTL / inherited ontology structure may already constrain:

```text
? ∈ ontology subtree P
? inherits properties q,r,s
? must satisfy the local causal / mereological / taxonomic relation
? is unresolved at historical version v
```

That means an **indirect intermediate unknown** is a located missing link in an otherwise constrained structure.

The useful transition is not only:

```text
unknown → known
```

It may be:

```text
vague unknown
  → ontology located
  → relation kind identified
  → mechanical vs epistemic vs observed status separated
  → candidate set narrowed
  → one missing intermediary isolated
  → discriminating test identified
  → witness acquired
  → Revision
  → warranted conclusion
```

Every intermediate step can be genuine epistemic progress.

A researcher thrives on exactly this kind of precision: **the puzzle gets sharper before it gets solved.**

---

## 2. Bits 59..63 are not reward storage

Do not repurpose the existing CausalEdge64 59..63 contract as reward, curiosity, free-energy magnitude,
or generic confidence storage.

Their value is indirect: the existing truth / reasoning-band reading can participate in making the
current inquiry state **precisely legible**.

The broader state is composed from existing carriers, not forced into five bits:

```text
InquiryState ≈
    causal/truth reading
  + reasoning band / rung
  + attention focus
  + ontology / inherited path
  + intermediate known / unknown
  + mechanical / epistemic / observed distinction
  + witness / provenance
  + historical Lance version
  + attempted behavior
  + unresolved pothole
```

The architectural reward is the precision itself, not a reward scalar stored in 59..63.

Do not reopen the ratified 59..63 layout merely because this synthesis gives it a richer role.

---

## 3. Horizontal and vertical traversal

A located gap gives the system at least two distinct ways to continue without bluffing.

### Horizontal traversal — find missing evidence or structure

Examples:

```text
missing mediator
  → ontology neighbor
  → inherited ancestor / descendant
  → sibling mechanism
  → episodic witness
  → alternate causal path
  → external source
  → sandboxed experiment
```

Horizontal movement asks:

> Can the gap be reduced without increasing the level of abstraction?

### Vertical traversal — think about the reasoning itself

Examples:

```text
direct grounding
  → Revision
  → counterfactual
  → causal abstraction
  → higher-order recipe
  → metacognitive inspection
```

Vertical movement asks:

> Is the evidence present but the current mode of reasoning insufficient?

The failure mode to avoid is automatic vertical escalation:

```text
I do not know
  → deeper rung
  → deeper rung
  → deeper rung
  → sophisticated unsupported answer
```

MUL should help distinguish **missing evidence** from **insufficient reasoning over available evidence**.

---

## 4. MUL: trust as the governor of commitment

MUL is not merely confidence.

The intended role is to detect when the system is prepared to act with more certainty than its evidence warrants.

A Dunning-Kruger-like failure mode is operationally:

```text
commitment / confidence rises
while
new evidence = 0
falsification = 0
pothole reduction = 0
provenance improvement = 0
justified inference gain = 0
```

The correct response is not simply “lower confidence”.

It is:

```text
trust deficit
  → keep the gap explicit
  → identify discriminating evidence
  → run a test / search / sandbox experiment
  → consume the witness
  → Revision
  → recompute warranted trust
```

The future promotion rule for self-programming should therefore resemble:

```text
candidate change
  + evidence
  + historical replay
  + counterfactual advantage
  + regression tests
  + provenance
  + unresolved potholes
  → MUL trust gate
```

Insufficient trust means **more epistemic work**, not optimistic promotion.

---

## 5. Resilient flow and delayed gratification

Architectural hypothesis, not established psychology:

A useful model of premature overconfidence is **premature closure after resilient flow breaks**.

```text
difficulty rises
  → uncertainty becomes aversive
  → inquiry flow breaks
  → closure becomes immediately gratifying
  → answer crystallizes before evidence warrants it
```

The target state is neither premature closure nor endless rumination.

| state | computational behavior |
|---|---|
| premature closure | exits uncertainty before trust warrants it |
| resilient flow | remains engaged while useful evidence can still be acquired |
| rumination | continues after marginal epistemic value has collapsed |

The delayed gratification is **closure**.

The machine should be allowed to treat:

```text
“I do not know yet”
```

as a productive state when it has a bounded next epistemic action.

This is one reason precise potholes matter: an unresolved state can remain navigable rather than becoming a generic failure.

---

## 6. Rubicon / Heckhausen + Libet timing as an engineering commitment model

Do not claim that the repository has solved human free will.

The useful engineering interpretation of the `-550 .. -220 .. 0` timing is a staged commitment window:

```text
-550
candidate activity exists
multiple trajectories remain alive

     ↓

-220
Revision / metacognitive checkpoint

questions:
- is trust sufficient?
- is the gap mechanical, epistemic, observed, or mixed?
- should we search horizontally?
- should we escalate vertically?
- should we acquire external evidence?
- should we preserve multiple counterfactual branches?

     ↓

0
commit / externalize / execute
```

The important future invariant is:

> commitment should be causally accountable to evidential or inferential work performed before the crossing.

A suspicious crossing is:

```text
alternatives: many → one
trust: low → high
new evidence: none
new falsification: none
```

A healthy crossing is:

```text
alternatives: many
  → discriminating work
  → falsified branches
  → reduced potholes
  → increased warranted trust
  → one committed trajectory
```

Rubicon visibility is therefore useful only if it can eventually be bound to the same mailbox / branch / rung / version / historical horizon.

---

## 7. Counterfactual without regret

Self-learning must not confuse outcome hindsight with decision quality.

The question is not:

```text
Given everything known now, was the old action optimal?
```

It is:

```text
Given only what was knowable at historical version v,
was that action warranted?
```

That allows the learner to distinguish:

- bad reasoning;
- reasonable reasoning under uncertainty;
- good reasoning followed by bad luck;
- a later fact that was impossible to know at decision time.

This is why historical version binding is part of the epistemic firewall for self-improvement.

Counterfactual branches should remain useful even when not chosen:

```text
                 S@v
              /   |   \
             A    B    C
             |    |    |
           outA outB pothole
              \   |   /
               compare
                  ↓
               Revision
```

Rejected branches are not garbage. They are training evidence about the shape of the decision space.

---

## 8. Potholes as the natural unit of self-learning

A pothole may be a better training boundary than an arbitrary reward window.

One episode can eventually be shaped as:

```text
PotholeContext {
    historical_version,
    mailbox_owner,
    rung,
    focus,
    ontology_scope,
    inherited_constraints,
    missing_relation_or_field,
    mechanical_epistemic_observed_status,
}

        ↓

executed behavior sequence

        ↓

Outcome {
    unresolved | narrowed | resolved | contradicted,
    witnesses_acquired,
    trust_change,
    time_to_resolution,
    causal validity,
}
```

The key question for learning becomes:

> For this precise shape of not-knowing, which behavior tends to produce warranted epistemic progress?

That is stronger than:

> Which action produces reward fastest?

---

## 9. Behavioral micro-IR: learning composition without mutating primitive meaning

#989 found the cognitive bytecode front-end and the missing interpreter/trace boundary.

If actual owner-local execution produces canonical atoms of the form:

```text
(FnIndex : Value)
```

then the first learning problem can be compiler-like:

```text
executed atoms
  → repeated subsequences
  → reversible behavioral macros
  → contextual usefulness measurement
```

BPE / Sequitur / Re-Pair are candidate discovery mechanisms, not architecture mandates.

The crucial distinction remains:

- **retired:** BPE as the mechanism behind the ontology / centroid codebook itself;
- **open:** reversible sequence induction over actual semantic execution traces.

A learned macro must never silently mutate primitive recipe semantics.

Preferred shape:

```text
macro id
  → exact primitive expansion
  → scope
  → version
  → provenance / representation hash
  → observed support
  → outcome receipts
```

Learning discovers composition.
Causal / epistemic / provenance gates remain authoritative.

---

## 10. Learning should optimize epistemic progress, not mere closure

A useful objective is not necessarily one scalar.

Candidate observable progress events include:

```text
gap localized
ontology scope narrowed
candidate eliminated
hypothesis falsified
witness acquired
mechanical vs epistemic ambiguity separated
intermediate unknown identified
historical contradiction found
confidence justified
pothole resolved
```

A frequent macro that merely reaches an answer faster is not automatically good.

A useful macro should improve some combination of:

- warranted precision;
- pothole reduction;
- causal validity;
- evidence quality;
- historical robustness;
- time / compute cost;
- avoidance of premature commitment.

Do not collapse these into one magnitude prematurely.

---

## 11. Sandbox use is an epistemic behavior and can itself be learned

When trust is insufficient, the system should eventually learn not merely a better answer but a better **evidence-acquisition procedure**.

Examples:

```text
unknown API behavior
  → compile / run minimal sandbox probe

unknown causal relation
  → historical / counterfactual comparison

unknown ontology mapping
  → exact reasoner / source lookup

unknown performance claim
  → benchmark

unknown machine-code behavior
  → R2IL / symbolic analysis
```

A sandbox does not create trust by magic.
It creates a witness.

```text
uncertainty
  → competing hypotheses
  → discriminating test
  → sandbox execution
  → witness
  → provenance / causal validation
  → Revision
  → MUL trust update
```

Tool use can therefore become part of the learned procedural vocabulary.

---

## 12. The alpha channel: thinking made visible without becoming world-state

The alpha-channel attention / eye-tracker idea is valuable because it keeps two facts separate:

```text
world / ontology state
```

and

```text
where this reasoning process looked while trying to understand it
```

The latter is metacognitive cartography, not ontology truth.

With 10 rung levels, a higher layer can inspect attention and execution residue below it:

```text
thinking
  ↓
attention / execution residue
  ↓
thinking about that thinking
  ↓
higher-order residue
```

That is a concrete form of “thinking about thinking” if the observation is bound to the same owner, branch, version and historical horizon.

Counterfactual branches can be rendered as a computational “multiverse”:

```text
chosen trajectory        strong alpha
rejected trajectories    faint alpha
unresolved branch        open / flashing pothole
Revision point           visible crossing
```

This is visualization / introspection, not proof by visualization.

---

## 13. Friston / free-energy connection: promising structure, not yet a claim

There is an architectural rhyme worth testing:

```text
expected model
  ↓
observation / witness
  ↓
mismatch
  ↓
located epistemic gap
  ↓
choose:
  - revise model
  - seek information
  - alter action
```

This resembles an operational active-inference loop.

But do not declare a Free Energy Principle implementation until a concrete quantity, prediction model and falsifier exist.

The potentially useful difference in this substrate is that mismatch can remain **semantically typed**:

```text
mechanical
vs epistemic
vs observed
vs causal
vs temporal
vs ontology mismatch
vs missing mediator
```

That structured discrepancy may be more useful than prematurely compressing all surprise into one scalar.

---

## 14. Self-programming must be transactional

Never define self-programming as unrestricted mutation of the currently executing self.

Preferred model:

```text
current self
  ↓
proposes candidate self'
  ↓
exact expansion / diff / provenance
  ↓
historical replay
  ↓
counterfactual evaluation
  ↓
regression tests
  ↓
sandboxed bounded execution
  ↓
MUL trust gate
  ↓
versioned promotion or rejection
```

Every candidate should preserve enough information for:

- parent version;
- exact change;
- reason / pothole that motivated it;
- evidence used;
- tests run;
- historical horizon;
- counterfactual comparison;
- promotion decision;
- rollback.

Self-improvement should be **proposal → proof → promotion**, not self-editing by enthusiasm.

---

## 15. External program IR and internal cognitive IR may eventually share generic machinery

Do not claim R2IL and cognitive reasoning are semantically the same.

But there may eventually be a shared pattern-processing layer over:

```text
typed operation
+ typed operand
+ dependencies
+ control transition
+ provenance
+ version
+ reversible expansion
```

Potentially shared generic algorithms:

- sequence canonicalization;
- block hashing;
- recurring subgraph discovery;
- trace comparison;
- reversible macro expansion;
- profile-guided specialization;
- guarded execution;
- deoptimization back to primitives.

Domain semantics remain separate:

- R2IL knows registers, memory, control flow;
- cognitive IR knows ontology scope, epistemic gaps, causal truth, witnesses, rungs and ownership.

The interesting target is not “one IR for everything”.
It is **one generic behavioral-pattern layer over domain-specific IRs** if code truth supports it.

---

## 16. The first self-learning loop after #989

Do not jump directly to self-programming.

The first loop should be deliberately boring:

```text
1. execute cognitive bytecode owner-locally
2. record exact trace receipts
3. preserve historical version and episode boundary
4. identify pothole open / narrow / close events where trustworthy
5. learn reversible repeated subsequences offline
6. freeze dictionary
7. replay exact expansion
8. measure held-out recurrence
9. measure outcome correlation
10. run shuffled / random / static-ladder controls
```

Kill the hypothesis if:

- execution simply reproduces `ladder_program()`;
- learned macros collapse out of sample;
- shuffled traces perform similarly;
- frequency fails to correlate with epistemic progress;
- macros require future evidence to look useful;
- expansion is not exact;
- learned behavior violates ownership, provenance or historical replay.

Only surviving behavior becomes a candidate for contextual routing.

---

## 17. The eventual closed loop

If the preceding stages survive, the long-term loop is:

```text
precise state of inquiry
      ↓
known / unknown / inherited / observed / mechanical / epistemic
      ↓
owner-local execution
      ↓
behavior trace
      ↓
learned procedural candidate
      ↓
MUL asks: is this warranted here?
      │
      ├── no → horizontal search / vertical reasoning / sandbox evidence
      │          ↓
      │        Revision
      │
      └── yes
             ↓
      counterfactual + historical replay
             ↓
      bounded promotion
             ↓
      future thought may use the new program
```

The system then learns not only answers, but **how it tends to turn particular kinds of not-knowing into better-grounded knowing**.

That is the endgame.

---

## 18. Non-negotiable fences

Until falsified by stronger evidence:

- no new CE64 bits for reward / curiosity / confidence;
- no widening `6×2×8bit` merely for learning;
- preserve the ratified 59..63 reading;
- pointers / references before inline learned magnitudes;
- primitive recipe semantics remain stable;
- learned macros must be exactly reversible;
- owner-local cognition must not be driven by unrelated fleet census;
- `kanban_actor` is visibility unless a separate resource-control contract explicitly says otherwise;
- no hindsight leakage;
- historical replay must use what was knowable then;
- a sandbox produces evidence, not automatic trust;
- uncertainty is not failure;
- precision without closure is valid epistemic progress;
- frequency is not success;
- confidence without accountable informational work is suspect;
- visualization is not proof;
- external neuroscience / psychology analogies remain hypotheses until operationalized.

---

## 19. What #990 should cause next

This PR should not itself implement the endgame.

It should make the sequence explicit so later agents do not rediscover it as disconnected ideas:

```text
#988  learning hypotheses separated
  ↓
#989  cognitive bytecode / missing interpreter and trace
  ↓
next   owner-local interpreter + exact receipts
  ↓
then   reversible behavioral-macro falsifier
  ↓
then   pothole-conditioned routing + evidence acquisition
  ↓
then   MUL-governed candidate-program evaluation
  ↓
then   transactional self-programming
```

The architecture should earn every arrow.

But it is now allowed to know where the arrows are trying to go.
