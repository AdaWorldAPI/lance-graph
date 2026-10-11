# Maslov Pyramid of Cognition — from Astonishment to generative Wisdom, and back

> READ BY: truth-architect, contradiction-cartographer, trajectory-cartographer,
> v3-envelope-auditor, and any session touching CausalEdge64 bits 50..52,
> novelty / Staunen / wisdom scheduling, `pearl::hydrate` feedback, or anchor
> ablation.
> STATUS: WORKING-MODEL. The probe is EXPERIMENTAL and test-local
> (`crates/lance-graph-planner/tests/maturation_ladder.rs`, D-MATURATION3-0).
> Nothing below is wired in production.

## Name and scope

The operator named this model after Maslow's hierarchy. It is a proposed
computational model, not a claim that Maslow defined these stages.

It is **not** Wang's NAL-1..9. NAL levels describe what a reasoner is able to
do. The pyramid describes how far **one discovery** has matured.

## The eight stages (bits 50..52, 3 bits, direction 7 → 0)

| code | stage | earned by an executed result |
|---|---|---|
| 7 | Astonishment | an observation contradicts an expectation (one root) |
| 6 | Anomaly | the contradiction has ≥ 2 independent roots |
| 5 | DogEar | hydration localizes the gap (ProposedOnly) |
| 4 | Candidates | hydration finds one binding (Partial) |
| 3 | Mechanism | hydration finds both bindings (Hydrated) |
| 2 | Falsification | a discriminating test passed the survivor AND ruled out a rival |
| 1 | Integration | used as a premise, it resolved ≥ 1 other relation |
| 0 | Wisdom | from Integration, ablating it in an isolated copy reopens ≥ 1 relation |

Rules:

- Stages may be skipped when an executed result justifies it.
- A forward result never moves a more mature state backward.
- Repeating a completed result (another hydration pass) never advances it.
- Only a contradiction reopens a stage.
- Wisdom is not "zero uncertainty". It is measured reuse: an ablation must
  show that other relations depend on the anchor.

## The recursion

An anchor at 0 generates expectations through the relations derived from it.
An observation that contradicted nothing before can now contradict one of
those expectations. That contradiction lands on **the contradicted
relation**, which starts its own pyramid at 7. **The anchor stays at 0.**

Only counterevidence against the anchor's **own** relation reopens the anchor.

So wisdom does not end surprise. It makes new surprises recognizable. This is
recursive scaffolding, not circular confirmation.

## Separation of contracts

| bits | contract | role here |
|---|---|---|
| 50..52 | maturation (this model) | where the discovery stands |
| 53..58 | witness / breadcrumb | handle to the transition record |
| 59..63 | `EpistemicState5` | what is licensed; the input to ablation |
| 24..39 | F/C | statistical evidence under its declared reading |
| — | thought / recipe | which operation ran |

Maturation never writes F/C, W or 59..63. Certification never rises because a
rival was ruled out.

## Generative reuse is not independent evidence

A resolved relation may become a premise for further inference. It carries
the **union of its parents' observation roots** and never adds one of its
own. This is the same mask algebra as the existing
`lance_graph_contract::revision` (`independent_roots`, `difference` against
the ancestry). Two guards keep reuse from becoming self-support:

1. Only resolved relations may serve as premises, so neither a direct cycle
   nor an indirect one (A → B → C → A) can ground itself.
2. A relation's own descendants are never its premises. Without this, a
   relation reopened by counterevidence would be re-grounded by what was
   derived from it, and circular support would survive the loss of its
   external ground.

Going round a grounded cycle adds nothing.

## Status

| claim | status |
|---|---|
| `pearl::hydrate` reads only the sealed chain; resolved relations never feed back | VERIFIED-IN-CODE |
| test-local feedback makes A → B → C resolvable without a new observation | EXPERIMENTAL (probe) |
| anchor ablation measures dependence (impact 2 on the chain) | EXPERIMENTAL (probe) |
| an anchor turns a neutral observation into new Astonishment; the anchor stays at 0 | EXPERIMENTAL (probe) |
| `mature`: ~2–3 ns median, 0 allocations (release, one container) | MEASURED |
| bits 50..52 carry the code in production | BLOCKED — they are plasticity (`learn`, `MooreNars16`, `CausalEdgeV3`) |
| production feedback | BLOCKED — needs D2 provenance ids |
| bit migration and where plasticity lives | CONTRACT DECISION (operator) |
