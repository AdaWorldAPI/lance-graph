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
