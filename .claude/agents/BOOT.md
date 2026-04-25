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
Use for `BusDto`, `CausalEdge64`, p64 style mapping, and CognitiveShader compilation.

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

## Meta-Agents (scopes and when to use)

Meta-agents coordinate across specialists. They do NOT replace
specialist judgment — they route, prime, and compose.

| Meta-agent | Scope |
|---|---|
| **`workspace-primer`** | Reads `LATEST_STATE.md` + `PR_ARC_INVENTORY.md`, reports current contract inventory, queued work, active branch. **First thing to wake on any new session.** |
| **`integration-lead`** | Composes specialist outputs into a coherent shipping plan. Use when ≥ 3 specialists have weighed in and their outputs need reconciliation before a PR. |
| **`adk-coordinator`** | Coordinates ADK tooling across crates. Use when orchestration crosses crate boundaries. |
| **`adk-behavior-monitor`** | Watches for drift between spec and implementation on multi-crate PRs. |
| **`truth-architect`** (elevated meta) | Mandatory reviewer on any proposal touching HHTL cascade, γ+φ, or claims-without-probes. |

Meta-agents run on Opus. Specialists run on Sonnet for single-source
grindwork (draft-file-from-spec); Opus for multi-source accumulation.
Never Haiku. See `../CLAUDE.md § Model Policy`.

## Session-Start Knowledge Bootload (per agent)

Every subagent spawned on this workspace loads these in order:

1. **Tier-0 (MANDATORY, all agents):**
   - `.claude/board/LATEST_STATE.md` — current contract inventory,
     what's shipped, what's queued, what's explicitly deferred.
   - `.claude/board/PR_ARC_INVENTORY.md` — per-PR Added / Locked /
     Deferred / Docs / Confidence. APPEND-ONLY.

2. **Tier-1 (domain-triggered):** load matching docs per the
   Knowledge Activation table below.

3. **Tier-2 (on-demand):** `docs/*.md` loaded when referenced.

Bootload cost: ~3 Read calls per agent. Fixed-cost insurance against
hallucinating structure that already exists.

## Knowledge Activation Protocol (updated 2026-04-19)

Agents do not operate in a vacuum. When woken, an agent MUST read the knowledge
documents listed in its trigger row BEFORE producing output.

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
| **Grammar / DeepNSM / TEKAMOLO** | integration-lead + truth-architect | grammar-landscape.md, grammar-tiered-routing.md, linguistic-epiphanies-2026-04-19.md |
| **Crystal / CrystalFingerprint / sandwich** | container-architect + family-codec-smith | crystal-quantum-blueprints.md, cross-repo-harvest-2026-04-19.md |
| **NARS inference / thinking styles** | truth-architect + bus-compiler | linguistic-epiphanies-2026-04-19.md (Chomsky isomorphism), integration-plan-grammar-crystal-arigraph.md |
| **Coreference / Markov ±5** | truth-architect + resonance-cartographer | grammar-landscape.md §5 §6, integration-plan-grammar-crystal-arigraph.md |
| **Cross-linguistic bundling** | integration-lead | grammar-landscape.md §4 case inventories, linguistic-epiphanies-2026-04-19.md E22 Rosetta |
| **AriGraph / episodic memory** | ripple-architect + integration-lead | LATEST_STATE.md (§AriGraph Inventory), PR_ARC_INVENTORY.md #208 |
| **Story arc / ONNX emergence** | ripple-architect + contradiction-cartographer | crystal-quantum-blueprints.md (§Quantum mode), endgame-holographic-agi.md |
| **Argmax / codec / PolarQuant** | truth-architect + family-codec-smith | fractal-codec-argmax-regime.md (ORTHOGONAL to grammar work) |
| **Cross-repo harvest** | savant-research | cross-repo-harvest-2026-04-19.md, linguistic-epiphanies-2026-04-19.md |
| **REST / gRPC / Wire DTO / endpoint / serve / shader-lab / external API** | integration-lead + truth-architect | **lab-vs-canonical-surface.md (MANDATORY — prevents System-1 "add another REST endpoint" hallucination)** |
| **OrchestrationBridge / UnifiedStep / StepDomain / BridgeSlot** | integration-lead | lab-vs-canonical-surface.md, cam-pq-unified-pipeline.md |
| **codec research / calibrate / probe / tensors op** | integration-lead + truth-architect | lab-vs-canonical-surface.md (research is just a consumer, not the canonical) |

**The insight update cycle:**

```
1. Agent proposes a claim
2. truth-architect checks: is there a probe for this? Has it run?
3. If NOT RUN → probe is the next deliverable, not more synthesis
4. If RUN + PASS → update .claude/knowledge/ with FINDING (not CONJECTURE)
5. If RUN + FAIL → update .claude/knowledge/ with correction
6. Commit knowledge update: docs(knowledge): probe [ID] — [result]
```

This cycle is MANDATORY.

## Agent Tool Discipline (all agents)

1. **Write large files via `tee -a path > /dev/null <<'EOF' ... EOF`.**
   Never use the `Write` tool for files over ~100 lines — the streaming
   response idle-timeouts. Chunk into 50-100 line `tee -a` calls.

2. **Never run `cargo clippy --fix`.** All clippy fixes must be manual
   `Edit` calls after reading context. Unused imports/variables/fields
   are **wiring signals** for future code, not cleanup targets.

3. **Never remove `use`, `let`, `fn`, `pub`, struct fields, or `mut`
   qualifiers** even if clippy flags them as unused/dead. These are
   intentional placeholders for in-progress wiring.

4. **Always add `--timeout 300000` (or higher) to `cargo` commands.**
   Compilation can take minutes on this workspace.

---

## Handover Protocol (agent A → agent B)

When an agent completes work that feeds another agent's scope, the
handover goes through a **handover blackboard entry** — not verbal
chat, not main-thread summary.

### The handover entry (file under `.claude/handovers/`)

Path: `.claude/handovers/YYYY-MM-DD-HHMM-<from>-to-<to>.md`

Required sections:

```markdown
# Handover: <from-agent> → <to-agent>
**Date:** YYYY-MM-DD HH:MM
**Scope:** one sentence — what's being handed off

## What I did
Short factual list. Files touched, tests run, decisions made.
Include commit SHAs where applicable.

## What I know (FINDING)
Claims I hold as measured / verified. Include the probe or test
that verifies each.

## What I suspect (CONJECTURE)
Claims I hold as plausible-but-unverified. The receiving agent
should probe or escalate to truth-architect before building on these.

## What I couldn't do
Blockers, out-of-scope items, ambiguities I punted on. Explicit.

## What you need to know
Minimum context transfer. Point at knowledge docs + source files,
do not repeat their content.

## Open questions
Questions the receiving agent must answer before proceeding. Each
bullet has an expected answer shape (yes/no, number, choice of N).
```

### Handover rules

1. **Every multi-agent workflow writes at least one handover.**
   Two-agent chain = 1 handover. N agents = N−1 handovers.
2. **Handovers are APPEND-ONLY.** Same governance as
   `PR_ARC_INVENTORY.md` — don't rewrite a previous agent's
   handover; append a follow-up if correction is needed.
3. **Receiving agent's FIRST action is to read the handover.**
   Then load the knowledge docs it cites. Then start work.
4. **Handovers name their successor.** If agent B needs agent C
   next, B's handover to C is its final deliverable.
5. **Main thread is the router, not a handover surface.** Main
   orchestrates which agents run; agents write handovers that other
   agents consume. Keeps main-thread context small.
6. **`truth-architect` review** is a valid handover target from any
   agent before a PR lands. Default final link on HHTL / codec /
   claims-without-probes work.

## Orchestration

Primary orchestration prompt:
- `.claude/agent2agent-orchestrator-prompt.md`

Core knowledge (workspace-wide):
- `.claude/board/LATEST_STATE.md` (Tier-0, mandatory)
- `.claude/board/PR_ARC_INVENTORY.md` (Tier-0, mandatory)
- `.claude/knowledge/*.md` (Tier-1, trigger-activated — see table above)
- `docs/integrated-architecture-map.md`

## Rule of use

Only wake the agents whose objects are actually being touched.
A crowded room is not automatically a wiser room.
