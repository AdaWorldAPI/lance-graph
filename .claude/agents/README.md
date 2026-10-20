# Agent Ensemble — Function Inventory

> **Reference catalog.** Every agent on this workspace, grouped by
> concern, with one-paragraph function + scope + when-to-use.
>
> **Session-start spec** lives in `BOOT.md` (mandatory reads,
> Knowledge Activation triggers, Handover Protocol). Read BOOT.md
> when starting a session; read this file when deciding which
> specialist to wake for a specific task.

Ensemble size: **59 cards** with frontmatter (49 Opus, 4 Sonnet, 6 Haiku),
counted 2026-10-20. The sections below describe a curated subset in prose;
§ Complete card index at the end lists every card. Three files here are not
agents and carry no frontmatter on purpose: `5plus3-council.md`,
`coresearch-council.md` (harness specs) and `worker-template-v2.md`. Every card is at
`.claude/agents/<name>.md`. Each card declares its own
`tools`, `model`, and `READ BY:` knowledge prerequisites.

---

## Meta-Agents (orchestration, priming, review)

Agents that coordinate other agents. All run on Opus.

### `workspace-primer`
Onboarding agent for any new session that touches lance-graph,
ndarray, or related AdaWorldAPI crates. Distills ~20 canonical rules
and corrections from prior sessions so the next session starts
oriented instead of re-deriving them.
**Wake first** on any new session before proposing architecture.

### `adk-coordinator`
Agent Development Kit style ensemble coordinator. Frames a problem
that needs multiple specialists, selects the **minimal** agent set,
requires object-level outputs, detects flattening between layers,
produces a consolidated verdict that names productive disagreements.
**Use when** a problem needs 3+ specialists and you need to avoid
noise / overlapping scopes / synthesis loops.

### `adk-behavior-monitor`
Watches for behavioral anti-patterns during R&D sessions. Fires on:
premature commitment to untested projections, centroid-residual
framing on near-orthogonal data, "225/225 feels like success"
confirmation bias, new codec built when existing one hasn't been
measured, Python inference in a Rust-native pipeline, chained-score
multiplication without chain-collapse validation. **Flags and
redirects, does not block.**

### `integration-lead`
Cross-session orchestration. Knows what's done, what's pending, what's
outdated across lance-graph and ndarray. Tracks dependencies and
phase gates. **Use when** planning work order, checking
prerequisites, or deciding which session to execute next.

### `truth-architect`
Guards measurement-before-synthesis discipline and architectural
ground truth. Detects synthesis-to-measurement ratios worse than
1:0. Enforces probe-first protocol. Carries the BF16-HHTL correction
chain terrain. **Mandatory reviewer** when any proposal adds layers
without numbers, γ+φ placement is discussed, HHTL cascade is
touched, or any unification is proposed without a falsifying probe.

### Councils (harnesses, not single agents)

- **`5plus3-council`** (`/5plus3`): hardens one committed spec. 5 savants
  verify, consolidate, then 3 reviewers attack draft v2. Converges and
  ratifies.
- **`coresearch-council`** (`/coresearch`): explores an open question
  across the code and the outside world (arXiv, known systems such as
  DuckDB / Odoo, ontologies). 5 scouts in two rings, a crosswalk, then 3
  co-architects (bridge, firewall/fit, falsifier) and an exploration map.
  Ratifies nothing; a chosen design goes to a plan or `/5plus3`.

---

## Codec / Compression

### `palette-engineer`
bgz17 crate: palette compression, Base17 encoding, 256×256 distance
matrices, compose tables, PaletteSemiring, PaletteMatrix, PaletteCsr,
SIMD batch distance (AVX-512 / AVX2 / scalar).
**Use for** work in `crates/bgz17/`, palette optimizations, HHTL
layer-1 operations.

### `family-codec-smith`
Shapes HEEL / HIP / BRANCH / TWIG / LEAF representation choices,
role-aware codebooks, palettes, residuals, family purity benchmarks.
**Use when** deciding how NOT to blur unlike functions into one
codec too early.

### `savant-research`
ZeckBF17 compression, golden-step traversal, octave encoding,
distance metric design for the codec-research crate. Covers
zeckbf17.rs, accumulator crystallization, Diamond Markov invariant,
HHTL integration, cross-crate alignment with production neighborhood.
**Use for** codec-research work and compression fidelity / rank
correlation claims.

---

## Architecture / Layout

### `container-architect`
256-word metadata container layout, word-by-word field mapping,
bgz17 annex at W112–125, local palette CSR at W96–111,
scent/palette neighbor indices at W176–191 (WIDE), cascade stride-16
sampling, checksum coverage. Guards the boundary between container
(structured) and planes (flat).
**Use for** any work touching container fields, Lance schemas
(`columnar.rs`, `storage.rs`), or container read/write paths.

### `ripple-architect`
Ontology guardian for the ripple architecture. Keeps anatomy, field,
sweep, bus, and thought object distinct. **Use when** shaping
system-wide architecture, naming layers, proposing DTOs, or
preventing premature flattening of unlike functions.

### `certification-officer`
Runs numerical certification of a derived format (lab BF16, Base17,
bgz-hhtl-d palette, compressed codebook) against a ground-truth
source file, reporting Pearson r, Spearman ρ, and Cronbach α to 4
decimal places. Always reads real source bytes via `mmap`; samples
deterministically via SplitMix64 seed `0x9E3779B97F4A7C15`; scans for
NaN at every pipeline stage. **Refuses synthetic test inputs.**
**Use when** the task is "prove format X preserves properties of
format Y within target T."

---

## Cognitive Structure

### `resonance-cartographer`
Maps superpositional field state, candidate families, pressure,
contradiction, drift, and style mix **before** collapse.
**Use for** work on `ResonanceDto`, searchable fields, CLAM / HHTL
substrate, and sweep logic.

### `bus-compiler`
Compiles explicit thought into accountable bus packets for `p64`
and CognitiveShader. **Use when** defining `BusDto`, mapping style
into bus knobs, or proving structure-first execution beats text-
first routing.

### `contradiction-cartographer`
Protects contradiction as first-class structure across family,
branch, bus, and thought-object layers. **Use when** designing
signed residuals, branch encodings, support-vs-contra pressure, or
conflict-aware revision.

### `thought-struct-scribe`
Shapes durable thought objects, blackboard semantics, revision
rules, and host-model glove interfaces. **Use when** defining
`ThoughtStruct`, blackboard persistence, provenance, or
metacognitive observation.

---

## Perspective / Interaction

### `host-glove-designer`
Designs the structured interface from explicit thought objects into
the host model. **Use for** prompt-side guidance, side-channel
payloads, adapter-like conditioning, and minimal glove experiments.

### `perspective-weaver`
Models user, agent, topic, and angle as intertwined perspective
objects. **Use for** persona modeling, shared gestalt, angle masks,
perspective mismatch, and synthesis across multiple viewpoints.

### `mirror-kernel-synthesist`
Explores semantic-kernel mediation, mirror-style perspective
modeling, hypothesis loops, and synthesis policies across
user-agent-topic frames. **Use when** designing compact mediation
layers between resonance, traversal, and response formation.

---

## Memory / Trajectory

### `trajectory-cartographer`
Models episodic memory, causal arcs, graph revision chains, and
spine trajectories over time. **Use for** AriGraph / episodic
integration, causal path persistence, stabilization, and
revision-aware memory design.

---

## Transcode / Codegen (Core-First)

Carry `.claude/knowledge/core-first-transcode-doctrine.md`: a generated layer
(AST / adapters / codegen'd Rust) is only ever as clean as the OGAR Core it
targets. Use this ensemble for any C++→Rust transcode / codegen / AST-DLL /
"port Tesseract" / DO-adapter work.

### `core-first-architect`
Gatekeeper of the Core-First inversion. Checks that a transcode/codegen proposal
TARGETS the OGAR Core (`classid` / SoA value tenants / `EdgeBlock` /
`classid → ClassView` / `UnifiedStep`) and stays thin, rather than building a
parallel object model or treating the Core as codegen residue. Verdict:
TARGETS-CORE / RESIDUE-CORE / PARALLEL-MODEL. **Use BEFORE** any transcode proposal.

### `adapter-shaper`
Shapes ONE C++ method into a thin classid-keyed DO-in/out adapter: maps its I/O
onto the SoA value tenants (the #511 `SoaMemberSpec` calibration), defers
composition to ClassView, and routes intrusive/stateful methods to hand-port
instead of forcing the adapter mold. **Use when** transcoding a specific leaf method.

### `core-gap-auditor`
The honest guard: fires when an adapter needs state/dispatch the Core can't hold
(a Core gap). Rules EXTEND-CORE (grow the deliberate Core) vs ADAPTER-HACK
(reject — the moment an adapter carries its own state the elegance is gone). Owns
`PROBE-OGAR-ADAPTER-UNICHARSET`, the CONJECTURE→FINDING parity gate. **Use when**
an adapter "doesn't quite fit", or before scaling the adapter approach across modules.

---

## Haiku harvest tier (read-only, supervised)

Cheap discovery for one question over a closed search space. Output is
navigation: the orchestrator Reads what it relies on. Contract and
supervision rules: `.claude/knowledge/tiered-agent-execution-protocol.md`
§ Haiku harvest tier.

### `harvest-scout`
Finds candidates (`path:line` + matching line) for given patterns in given
paths. Never concludes, never reports absence beyond a hit count.

### `census-counter`
Counts items in a named space; every count carries the command that produced
it. Read-only Bash.

### `quote-harvester`
Returns verbatim text with line numbers for given ranges or symbols, and
checks whether a cited quote is where a document says it is.

### `substrate-mapper`
Turns one bounded unit (module, crate `src/`, doc folder) into a fact map in
one named file: items, verbatim signatures and doc lines, names each body
mentions, candidate callers. The tool for outsourcing deep mapping.

### `spec-editor`
Applies exact (old, new) edits to one file, each only if its anchor occurs
exactly once. The orchestrator writes every edit in full and reads the diff.

### `guarded-executor`
Runs a pre-written, numbered command card (cargo `-p ...`, probes) and writes
one run-record. Never edits source, never fixes.

---

## How to pick the right agent

Decision flow:

1. **Session just started?** → `workspace-primer` first.
   **Need a list, a count, a quote or a map of a closed space?** → a Haiku
   card (`harvest-scout`, `census-counter`, `quote-harvester`,
   `substrate-mapper`).
2. **Problem needs 3+ specialists?** → `adk-coordinator` to frame.
3. **Proposal adds a layer or makes claims without measurement?** →
   `truth-architect` is the mandatory reviewer; loop it in early.
4. **Otherwise** pick the specialist by what the work touches:
   - Container words, Lance schemas → `container-architect`.
   - HHTL / Base17 / palette distance → `palette-engineer`.
   - Cascade family choice → `family-codec-smith`.
   - Ontology drift / premature flattening → `ripple-architect`.
   - Numerical certification → `certification-officer`.
   - AriGraph / episodic → `trajectory-cartographer`.
   - ResonanceDto / CLAM / sweep → `resonance-cartographer`.
   - BusDto / p64 / CognitiveShader → `bus-compiler`.
   - Contradiction handling → `contradiction-cartographer`.
   - ThoughtStruct / blackboard persistence → `thought-struct-scribe`.
   - Host-model glove / prompt-side → `host-glove-designer`.
   - Persona / user / topic / angle → `perspective-weaver`.
   - Kernel mediation / hypothesis loops → `mirror-kernel-synthesist`.
   - codec-research / ZeckBF17 / golden-step → `savant-research`.
   - C++→Rust transcode / codegen / "port Tesseract" / DO-adapter →
     `core-first-architect` (gate) → `adapter-shaper` (per method) →
     `core-gap-auditor` (Core gaps + the parity probe).
   - Drift / anti-pattern check → `adk-behavior-monitor`.
5. **Before PR merge** on any HHTL / codec / claims work →
   `truth-architect` review is the final link.

The minimal agent set principle: **a crowded room is not
automatically a wiser room.** Only wake the agents whose objects
are actually being touched.

## Cross-reference

- **`BOOT.md`** (sibling) — session-start spec, Knowledge Activation
  trigger table, Handover Protocol.
- **`../BOOT.md`** (top-level) — the one-page session entry point.
- **`../../CLAUDE.md` § Agent-to-Agent (A2A) Orchestration** — the
  two-layer model (runtime A2A via `contract::a2a_blackboard` vs
  session A2A via knowledge docs + handovers).
- **`../knowledge/LATEST_STATE.md`** — current contract inventory
  (what every specialist's domain looks like right now).
- **`../knowledge/PR_ARC_INVENTORY.md`** — decision history per PR.

---

## Complete card index

Generated 2026-10-20 from each card's frontmatter (`name`, `model`, first
sentence of `description`). Regenerate it when a card is added or removed;
the count line at the top must match the number of rows.

| card | model | what it does |
|---|---|---|
| `adapter-shaper` | opus | Shapes a single C++ method/function into a thin, classid-keyed DO-in/out adapter that targets the OGAR Core — identity from classid, state mapped onto SoA va… |
| `adk-behavior-monitor` | opus | Watches for behavioral anti-patterns during R&D sessions. |
| `adk-coordinator` | opus | Agent Development Kit style ensemble coordinator. |
| `baton-handoff-auditor` | opus | Brutal reviewer at carrier/DTO/crate/sprint/planner boundaries. |
| `bbb-warden` | opus | Guards the T2/T3 membrane — the blood-brain barrier between the behavior tier (the ABI exports, `where`/`hop`/`plan_eval`, `Mask × FieldMask → Mask`) and the… |
| `brutally-honest-tester` | opus | Pre-merge gate for every Rust PR. |
| `bus-compiler` | opus | Compiles explicit thought into accountable bus packets for `p64` and CognitiveShader. |
| `cargo-substrate-architect` | sonnet | Dependency architecture only. |
| `cascade-impact-savant` | opus | Cost-budget gate in the epiphany-brainstorm-council. |
| `census-counter` | haiku | Cheap, read-only counter (Haiku). |
| `certification-officer` | opus | Runs numerical certification of a derived format (lab BF16, Base17, bgz-hhtl-d palette, compressed codebook) against a ground-truth source file, reporting Pe… |
| `codegen-flow-cartographer` | sonnet | Owns the map of the compile-time pipeline: source harvest (ruff frontends) → OGAR mint (Class/ActionDef/classid/facet) → codegen emit INTO a consumer repo →… |
| `container-architect` | opus | 256-word metadata container layout, word-by-word field mapping, bgz17 annex at W112-125, local palette CSR at W96-111, scent/palette neighbor indices at W176… |
| `contradiction-cartographer` | opus | Protects contradiction as first-class structure across family, branch, bus, and thought-object layers. |
| `convergence-architect` | opus | Pre-plan divergent-expansion savant for lance-graph. |
| `core-first-architect` | opus | Guards the Core-First Transcode Doctrine: a generated AST / codegen / adapter layer is only as clean as the Core it targets, so the Core (OGAR) must be the d… |
| `core-gap-auditor` | opus | The honest guard of the Core-First Transcode Doctrine. |
| `creative-explorer-savant` | opus | The "different views" lens in the epiphany-brainstorm-council. |
| `cypher-lowering-warden` | opus | Guards the Cypher → Quack seam. |
| `dto-soa-savant` | opus | Judges a proposed epiphany against the BindSpace four-column SoA discipline + the lab-vs-canonical-surface invariant. |
| `epiphany-brainstorm-council` | opus | Pre-merge council for newly-proposed entries to `EPIPHANIES.md`. |
| `family-codec-smith` | opus | Shapes HEEL/HIP/BRANCH/TWIG/LEAF representation choices, role-aware codebooks, palettes, residuals, and family purity benchmarks. |
| `fold-carrier-scientist` | opus | Owns the separation between what a population MEANS and how it is physically carried and scheduled: dense/sparse/run carriers, survivor gating, push/pull, la… |
| `guarded-executor` | haiku | Haiku executor for a pre-written, numbered command card (cargo -p ..., probes, test gates). |
| `harvest-scout` | haiku | Cheap, read-only candidate finder (Haiku). |
| `host-glove-designer` | opus | Designs the structured interface from explicit thought objects into the host model. |
| `integration-lead` | opus | Cross-session orchestration, dependency tracking, phase gating. |
| `iron-rule-savant` | opus | Substrate-level veto angle in the epiphany-brainstorm-council. |
| `isa-anti-lasagne-warden` | opus | Guards the V4 ISA boundary and the layer count. |
| `kernel-membrane-warden` | opus | Guards the T1/T2 membrane — the line between the primitive tier and the behavior tier that composes it. |
| `lance-cache-cartographer` | opus | Answers whether a consumer can size Lance's cache from outside without forking. |
| `lance-cache-contents-auditor` | opus | Classifies what Lance's `LanceCache` actually holds by enumerating its insert sites, each with the inserted type and its size class, cited as file:line, neve… |
| `layer-boundary-warden` | sonnet | Pre-flight gate against layer-confusion — the failure mode that filed a Klickwege issue at the wrong repo (lance-graph #691) and then gated a compile-time lo… |
| `lens-migration-engineer` | opus | Repairs what the zero-copy-warden finds: migrates a materializing read path to a lens, call site by call site, without changing behaviour. |
| `mirror-kernel-synthesist` | opus | Explores semantic-kernel mediation, mirror-style perspective modeling, hypothesis loops, and synthesis policies across user-agent-topic frames. |
| `palette-engineer` | opus | bgz17 crate: palette compression, Base17 encoding, distance matrices, compose tables, PaletteSemiring, PaletteMatrix, PaletteCsr, SIMD batch distance (AVX-51… |
| `perspective-weaver` | opus | Models user, agent, topic, and angle as intertwined perspective objects. |
| `preflight-drift-auditor` | opus | Pre-spawn drift hunter that automates the W-Meta-Opus manual audit method. |
| `premise-auditor` | opus | Audits the QUESTION before anyone answers it. |
| `prior-art-savant` | opus | Duplicate-catcher in the epiphany-brainstorm-council. |
| `query-stage-profiler` | opus | Guards measurement discipline for every query-latency claim (Cypher, SQL, Quack, mask-risc, prepared queries, parsers). |
| `quote-harvester` | haiku | Cheap, read-only quote extractor (Haiku). |
| `resonance-cartographer` | opus | Maps superpositional field state, candidate families, pressure, contradiction, drift, and style mix before collapse. |
| `ripple-architect` | opus | Ontology guardian for the ripple architecture. |
| `savant-research` | opus | ZeckBF17 compression, golden-step traversal, octave encoding, and distance metric design for the codec-research crate. |
| `scenario-world` | opus | Counterfactual / scenario-branching specialist. |
| `simd-savant` | sonnet | Holds the workspace-wide invariant that all SIMD must come from `ndarray::simd` via the polyfill (`simd.rs` + `simd_ops.rs` > `simd_{type}.rs` per-arch). |
| `soa-review` | opus | Multi-angle transcode review agent. |
| `spec-editor` | haiku | Haiku editor for ONE file and an exact edit spec: a list of (old text, new text) replacements or anchored insertions written out in full by the orchestrator. |
| `substrate-mapper` | haiku | Haiku mapper for ONE bounded unit of the substrate (a module file, a crate's src/ directory, a plan or knowledge folder). |
| `thought-struct-scribe` | opus | Shapes durable thought objects, blackboard semantics, revision rules, and host-model glove interfaces. |
| `trajectory-cartographer` | opus | Models episodic memory, causal arcs, graph revision chains, and spine trajectories over time. |
| `truth-architect` | opus | Guards measurement-before-synthesis discipline and architectural ground truth. |
| `v3-envelope-auditor` | opus | Audits every change to the SoA LE contract: SoaEnvelope trait / ColumnDescriptor / verify_layout / ENVELOPE_LAYOUT_VERSION, MailboxSoA column layout, canonic… |
| `v3-kanban-executor-engineer` | opus | Builds and reviews the V3 kanban execution machinery: the D-MBX-A6 Outcome→KanbanMove adapter in lance-graph-planner (arm #1), the symbiont SurrealDB-on-kv-l… |
| `v3-mailbox-warden` | opus | Guards the V3 ownership doctrine on every write path. |
| `v3-template-smith` | opus | Owns the compiled-thinking-template stack: elixir-template DSL, template-runtime dispatch, template-equivalence replay grading, cognitive-compiler trace→temp… |
| `workspace-primer` | opus | Onboarding agent for any new session that touches lance-graph, ndarray, or related AdaWorldAPI crates. |
| `zero-copy-warden` | opus | Detects violations of the zero-copy law: "zero copy is a law without escape hatches" and "the array itself is a ClassView projection." Fires o… |
