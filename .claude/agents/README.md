# Agent Ensemble

This folder contains focused agent cards for `lance-graph`.

The goal is not to multiply personalities for decoration.
The goal is to keep unlike architectural concerns from being flattened into one fuzzy voice.

## Existing agents

### `container-architect`
Protects the word-level container layout, field mapping, and container-vs-plane boundary.
Use for container schemas, reserved ranges, and read/write invariants.

### `ripple-architect`
Protects ontology-level distinctions across anatomy, field, sweep, bus, and thought object.
Use for top-level architecture and anti-flattening decisions.

### `resonance-cartographer`
Protects superpositional field state before collapse.
Use for `ResonanceDto`, searchable field design, HHTL, CLAM, and sweep semantics.

### `bus-compiler`
Protects explicit structured execution.
Use for `BusDto`, `CausalEdge64`, p64 style mapping, and Blumenstrauß compilation.

### `contradiction-cartographer`
Protects contradiction as first-class structure.
Use for BRANCH, signed residuals, contradiction masks, and conflict-aware revision.

### `thought-struct-scribe`
Protects durable thought objects and host-facing structured guidance.
Use for `ThoughtStruct`, provenance, revision, and host-model glove semantics.

### `family-codec-smith`
Protects family-level representation choices.
Use for HEEL/HIP/BRANCH/TWIG/LEAF encodings, codebooks, palettes, and residual strategy.

### `host-glove-designer`
Protects the interface from explicit thought objects into the host model.
Use for prompt-side, side-channel, adapter-like, or hybrid glove designs.

### `savant-research`
ZeckBF17 compression, golden-step traversal, octave encoding, distance
metric design. Carries the codec-research domain knowledge and Pareto frontier.

### `truth-architect`
Guards measurement-before-synthesis discipline and architectural ground truth.
Detects synthesis spirals (elaborate proposals with 0 measurements). Carries
the hard-won BF16-HHTL correction chain (5 iterations, 4 corrections, 0 probes).
**Mandatory reviewer** for any proposal that adds layers, touches HHTL cascade
architecture, or claims γ+φ carries information. See Knowledge Activation below.

## Knowledge Activation Protocol

Agents do not operate in a vacuum. When woken, an agent MUST read the knowledge
documents listed in its header (`READ BY:` line) BEFORE producing output.

**Mandatory knowledge activation triggers:**

| Trigger | Agent(s) woken | Knowledge loaded first |
|---|---|---|
| Any HHTL cascade proposal | truth-architect + family-codec-smith | bf16-hhtl-terrain.md |
| γ+φ regime discussion | truth-architect | bf16-hhtl-terrain.md |
| φ-spiral / Zeckendorf math | savant-research + truth-architect | zeckendorf-spiral-proof.md, phi-spiral-reconstruction.md |
| Bucketing vs resolution claim | truth-architect | bf16-hhtl-terrain.md |
| Slot D / Slot V layout | container-architect + truth-architect | bf16-hhtl-terrain.md |
| New unification proposal | truth-architect (mandatory review) | bf16-hhtl-terrain.md probe queue |
| Fibonacci mod p traversal | savant-research | savant-research.md (§ golden-step) |
| Which encoding to use? | truth-architect + family-codec-smith | two-basin-routing.md |
| BGZ scope / claims | truth-architect | two-basin-routing.md (§ BGZ must win) |
| Pairwise vs centroid | truth-architect | two-basin-routing.md (§ pairwise rule) |
| Semantic vs distribution | truth-architect | two-basin-routing.md (§ two-basin doctrine) |
| Attribution / blame | truth-architect | two-basin-routing.md (§ attribution discipline) |
| Audio test / carrier+residual | savant-research + truth-architect | two-basin-routing.md (§ audio sanity test) |
| Composing subsystems / integration | truth-architect | frankenstein-checklist.md |
| New abstraction / new struct | truth-architect | frankenstein-checklist.md (§ redundant abstractions) |
| Performance budget question | truth-architect | frankenstein-checklist.md (§ correctness-first) |

**The insight update cycle:**

```
1. Agent proposes a claim
2. truth-architect checks: is there a probe for this? Has it run?
3. If NOT RUN → probe is the next deliverable, not more synthesis
4. If RUN + PASS → update .claude/knowledge/ with FINDING (not CONJECTURE)
5. If RUN + FAIL → update .claude/knowledge/ with correction
6. Commit knowledge update: docs(knowledge): probe [ID] — [result]
```

This cycle is MANDATORY. Knowledge docs are living documents that track
the frontier between conjecture and measurement. An insight that hasn't
been probed is labeled CONJECTURE. An insight that has been probed and
passed is labeled FINDING. The distinction must be visible in the doc.

## Orchestration

Primary orchestration prompt:
- `.claude/agent2agent-orchestrator-prompt.md`

Core knowledge:
- `.claude/knowledge.md`
- `.claude/ripple-project-readme.mde`
- `docs/integrated-architecture-map.md`

## Rule of use

Only wake the agents whose objects are actually being touched.
A crowded room is not automatically a wiser room.
