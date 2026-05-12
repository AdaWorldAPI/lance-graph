# Agent Ensemble — Function Inventory

> **Reference catalog.** Every agent on this workspace, grouped by
> concern, with one-paragraph function + scope + when-to-use.
>
> **Session-start spec** lives in `BOOT.md` (mandatory reads,
> Knowledge Activation triggers, Handover Protocol). Read BOOT.md
> when starting a session; read this file when deciding which
> specialist to wake for a specific task.

Ensemble size: **19 specialists + 5 meta-agents**. Every card is at
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

## How to pick the right agent

Decision flow:

1. **Session just started?** → `workspace-primer` first.
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
