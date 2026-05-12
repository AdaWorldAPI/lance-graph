# Tier-0 Pattern Recognition — Map Patterns A–O to Shipped Code

> **READ THIS BEFORE READING ANYTHING ELSE in a unified-OGIT session.**
>
> Authored 2026-05-12 by Worker Agent W2 of the OGIT-architecture sprint
> as the **anti-Discovery-Loop** doc. The unified-OGIT synthesis
> crystallized 15 architectural patterns (A–O). Brutally honest fact:
> **~80 % of those patterns are already shipped in this workspace.**
> The only genuinely new work is the OGIT-G overlay wiring + the
> per-G manifest pattern.
>
> Every prior multi-agent sprint that did not consult this map
> spent 6–10 turns rediscovering substrate before producing a single
> commit. This file exists so the next session pays a one-read tax
> instead of an N-turn rediscovery tax.
>
> **Companion docs:**
> - `.claude/patterns.md` — five traversal patterns (P-1..P-5)
> - `.claude/knowledge/soa-dto-fma-map.md` — region map R0–R8
> - `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` — open entropy rows
> - `unified-ogit-architecture-v1.md` (W1, sibling deliverable) —
>   master plan-doc with full Pattern A–O definitions
>
> **Rule of engagement:** if you are about to propose "build Pattern X"
> for any X in {A..O}, first locate it on the table below. If a file
> path appears in the right column, you are extending — not building.

---

## TL;DR — Which patterns are already shipped, which need wiring

| Pattern | Status | Ships in |
|---|---|---|
| A — OGIT Triangle (T/A/P + W as four contexts) | **Shipped (consumer side)** | `crates/thinking-engine/src/world_model.rs`, contract `SelfState/UserState/FieldState/ContextState` |
| B — Context Bundle (8-slot per-G manifest) | **Half-shipped** | 8 predicate planes in `crates/p64-bridge/src/lib.rs`; per-G manifest spec missing |
| C — Per-G Tokenizer Registry | **Shipped** | `crates/thinking-engine/src/tokenizer_registry.rs` |
| D — Persona Runtime | **Shipped** | `crates/thinking-engine/src/persona.rs` |
| E — Sensor Bridge | **Shipped** | `crates/thinking-engine/src/sensor.rs`, `crates/thinking-engine/src/osint_bridge.rs` |
| F — Actor Message Shape (ractor-equivalent) | **Shipped (proof-of-shape)** | `crates/cognitive-shader-driver/src/grpc.rs` tonic service trait |
| G — OGIT Overlay (per-G shader bind) | **NEW WORK** | nothing — this is the actual deliverable |
| H — Switchable Cognitive Vessel | **Shipped** | `crates/p64-bridge/src/lib.rs` (CognitiveShader + 12 STYLES + 8 planes) |
| I — Implicit-Cognition Flush Gate | **Shipped** | `crates/lance-graph-contract/src/cycle_accumulator.rs` |
| J — Switchable Reranker Lens | **Shipped** | `crates/thinking-engine/src/reranker_lens.rs` |
| K — Crystallization-Side Learning | **Shipped** | `crates/thinking-engine/src/contrastive_learner.rs`, `crates/lance-graph/src/cam_pq/jitson_kernel.rs` |
| L — MUL Counterfactual Branching | **Shipped** | `crates/thinking-engine/src/branching.rs`, `crates/thinking-engine/src/ghosts.rs`, `crates/lance-graph/src/graph/arigraph/` |
| M — Wave-Side BNN World Model | **Shipped** | `crates/thinking-engine/src/world_model.rs`, `superposition.rs`, `jina_lens.rs`, `bge_m3_lens.rs` |
| N — Fingerprint-as-Codebook-Address | **Shipped** | `crates/thinking-engine/src/prime_fingerprint.rs` |
| O — Phenomenological Memory | **Shipped** | `crates/thinking-engine/src/qualia.rs`, `awareness_dto.rs` |

**Net new work for OGIT-G:** Pattern G overlay wiring + Pattern B's
manifest format. Everything else is **extension**, not construction.

---

## Pattern-by-pattern recognition

### Pattern A — OGIT Triangle (T/A/P + W)

The four cognitive surfaces — Topic, Angle, Planner, World — are not
new abstractions; they are the four **BindSpace columns** already
specified in CLAUDE.md AGI-as-glove doctrine + shipped as the
`WorldModelDto` four-section split.

**Already ships:** `crates/thinking-engine/src/world_model.rs` (13 KB)
defines `SelfState / UserState / FieldState / ContextState`. Each is
a typed struct of fields the agent actually reads. The OGIT triangle
labels (Topic / Angle / Planner / World) map onto these four sections
1:1 — `ContextState` IS Topic, `UserState` IS Angle, `SelfState` +
`FieldState` IS the Planner/World pair.

**Anti-rediscovery:** do NOT propose `struct OgitTriangle { topic,
angle, planner, world }`. Read `WorldModelDto` first.

### Pattern B — Context Bundle (8 slots) + per-G Manifest

The 8-slot structure already ships as the 8 predicate planes in
`p64-bridge::CognitiveShader::planes: [[u64; 64]; 8]`. Each plane
(CAUSES / ENABLES / SUPPORTS / CONTRADICTS / REFINES / ABSTRACTS /
GROUNDS / BECOMES) IS a slot of the bundle.

**Already ships:** `crates/p64-bridge/src/lib.rs:113-122` —
predicate-layer indices + bit positions.

**What's missing:** the per-G *manifest* — a small YAML or struct
that says, for graph G, which tokenizer / persona / role catalogue /
codebook / lens binds into those 8 slots. This is the **only**
genuinely new artifact on this axis.

### Pattern C — Per-G Tokenizer Registry

**Already ships:** `crates/thinking-engine/src/tokenizer_registry.rs`
(16 KB). Routes between Qwen3 BPE (Jina v5 + Reranker v3), Qwen2 BPE
(Qwopus, Reader-LM), XLM-R SentencePiece (Jina v3, BGE-M3 legacy),
and OLMo BPE (ModernBERT). The per-G overlay just selects which
registry entry binds for graph G.

**Anti-rediscovery:** do NOT write a "TokenizerRouter" trait — the
registry already routes by model family.

### Pattern D — Persona Runtime

**Already ships:** `crates/thinking-engine/src/persona.rs` (18 KB).
Per-G personas are catalogue entries here, not new types.

### Pattern E — Sensor Bridge

**Already ships:** `crates/thinking-engine/src/sensor.rs` (13 KB) for
generic input handling, plus the Gotham-equivalent consumer-side
bridge in `crates/thinking-engine/src/osint_bridge.rs` (5.5 KB).
Per-G overlay swaps which sensor schema binds; the bridge itself is
done.

### Pattern F — Actor Message Shape (ractor-equivalent)

**Already ships (proof-of-shape):**
`crates/cognitive-shader-driver/src/grpc.rs` — the tonic
`CognitiveShaderService` trait IS the actor handler shape we would
otherwise reach for ractor to provide. `async fn dispatch(req) ->
Response` is the message-handler signature; the `Arc<Mutex<ShaderDriver>>`
field is the actor's state cell.

**Rephrase before designing a new actor framework:** any
ractor-shaped proposal needs to justify what it adds over this
trait. The runtime A2A blackboard
(`lance_graph_contract::a2a_blackboard`) is the cycle-orchestration
substrate; tonic services are the I/O surface. There is no Pattern-F
gap to fill — only a question of which lab-vs-canonical edge a new
endpoint sits on (see `.claude/knowledge/lab-vs-canonical-surface.md`).

### Pattern G — OGIT Overlay (THE NEW WORK)

This is the one pattern with no prior implementation. The OGIT-G
overlay binds a specific graph G into the `CognitiveShader::planes`
+ `STYLES` + persona registry simultaneously. The wiring shape:

1. Read the per-G manifest (Pattern B's missing piece).
2. Resolve tokenizer / persona / lens / role catalogue / codebook.
3. Construct `CognitiveShader<'a>` with G's planes + G's semiring.
4. Hand to `ShaderDriver::dispatch` via `UnifiedStep`.

**No existing crate does step 1 or step 4 end-to-end for an
arbitrary G.** Everything inside steps 2–3 is shipped substrate.

### Pattern H — Switchable Cognitive Vessel

**Already ships:** `crates/p64-bridge/src/lib.rs` is the canonical
shipping of Pattern H. Concretely:

```rust
pub struct CognitiveShader<'a> {
    pub planes: [[u64; 64]; 8],         // 8 predicate planes (Pattern B slot)
    pub semiring: &'a PaletteSemiring,   // bgz17 O(1) distance + compose
    pub k: usize,
}
```

Plus the **12-entry STYLES codebook** at `lib.rs:161-180`:

- *Convergent cluster:* analytical / convergent / systematic
- *Divergent cluster:* creative / divergent / exploratory
- *Attention cluster:* focused / diffuse / peripheral
- *Speed cluster:* intuitive / deliberate / metacognitive

Each style is a `StyleParams { layer_mask, combine, contra,
density_target, name }` — the same shape Pattern H needs as a vessel
switch.

The doc comment at `lib.rs:237-238` is the giveaway:

> *"No POPCNT. No Hamming. Distance is PRECOMPUTED in the codebook.
> The mask gates access. The table provides the answer. O(1)."*

That sentence IS the Pattern H thesis. The HHTL cascade lives at
`cognitive_shader::cascade(query, radius, layer_mask)`:

- **HEEL** = `layer_mask` gates which Z planes are active
- **HIP** = `planes[z][block_row]` gates which X-Y cols are active
- **TWIG** = 4x4 block expansion to archetype indices
- **LEAF** = `semiring.distance(query, target)` O(1) lookup

The transitive-chain reasoning lives at `cognitive_shader::deduce_path(query,
cause_layer, effect_layer, max_hops)` — Pattern L's chain-of-thought
at substrate level via `semiring.compose`.

**Anti-rediscovery:** if a future session says "we need a switchable
cognitive vessel with 8 predicate planes and 12 styles" — point them
at `p64-bridge/src/lib.rs` and stop the meeting.

### Pattern I — Implicit-Cognition Flush Gate

**Already ships:** `crates/lance-graph-contract/src/cycle_accumulator.rs`
implements topology I-4: the per-cadence gate that sits between the
20–200 ns BindSpace ops and the 2–200 ms outbound sink. The doc
comment explicitly distinguishes it from `CollapseGate` (the per-row
write-airgap). Both are gates; they govern different boundaries.

**Anti-rediscovery:** do NOT propose a "thinking accumulator" or
"reflection batch buffer" — `CycleAccumulator` IS it. See
`.claude/board/SINGLE_BINARY_TOPOLOGY.md` Per-cadence gate section.

### Pattern J — Switchable Reranker Lens

**Already ships:** `crates/thinking-engine/src/reranker_lens.rs`
(7 KB). The lens is switchable per-call; the registry of available
rerankers (Qwen3-based v3 ground-truth, legacy v2 lens, etc.) is
documented in CLAUDE.md Model Registry section.

### Pattern K — Crystallization-Side Learning

**Already ships:**

- `crates/thinking-engine/src/contrastive_learner.rs` (9 KB) —
  crystallization-side learning loop
- `crates/lance-graph/src/cam_pq/jitson_kernel.rs` — JIT muscle that
  recompiles when the learner promotes a kernel from "researched"
  to "hot path"

**Anti-rediscovery:** the question is never "build learning into the
crystal" — it is "wire the learner's promotion signal to the JIT
kernel cache." That wiring is open; the substrate is done.

### Pattern L — MUL Counterfactual Branching

**Already ships:**

- `crates/thinking-engine/src/branching.rs` (10 KB) — branch
  bookkeeping
- `crates/thinking-engine/src/ghosts.rs` (11 KB) — counterfactual
  ghost branches (the MUL ambiguity carriers)
- `crates/thinking-engine/src/domino.rs` (23 KB) — cascading
  inference chains across branches
- `crates/lance-graph/src/graph/arigraph/` — AriGraph string-keyed
  SPO store providing the position index for branch-local facts

`p64-bridge::CognitiveShader::deduce_path` at substrate level
composes via `semiring.compose` per hop.

### Pattern M — Wave-Side BNN World Model

**Already ships:**

- `crates/thinking-engine/src/world_model.rs` (13 KB) — the BNN-like
  world model itself (`WorldModelDto` four-section structure)
- `crates/thinking-engine/src/superposition.rs` (9.6 KB) — wave-mode
  quantum-like superposition
- `crates/thinking-engine/src/jina_lens.rs` and `bge_m3_lens.rs` —
  wave-mode lenses INTO the LM embedding spaces (the wave side
  observes via these lenses; particle side reads the same data via
  fingerprint comparisons)

Pattern M's "wave" vs "particle" dual is the same split as
`thinking-engine::layered.rs` vs `thinking-engine::cognitive_stack.rs`:
streaming-perception lenses on one side, L1–L4 commit pipeline on
the other.

### Pattern N — Fingerprint-as-Codebook-Address

**Already ships:** `crates/thinking-engine/src/prime_fingerprint.rs`
(15 KB). The fingerprint IS the address into the codebook; this is
the substrate Pattern N declares. Reference doc:
`.claude/knowledge/primzahl-encoding-research.md`. Companion
substrate at `crates/bgz17/src/palette_semiring.rs` (O(1) distance +
compose algebra).

### Pattern O — Phenomenological Memory

**Already ships:**

- `crates/thinking-engine/src/qualia.rs` (39 KB) — 17D qualia, 10
  family centroids, music-calibrated dims (octave / fifth / third /
  tritone → arousal / valence / warmth / tension), audio bridges,
  220-pair Jina v3 cross-validation
- `crates/thinking-engine/src/awareness_dto.rs` (12 KB) — the
  meta-awareness layer DTO that carries qualia between cycles

This is the most surprising "already done" — 39 KB of calibrated
17-dim phenomenological substrate. Any session proposing "give the
agent feelings" must read `qualia.rs` first.

---

## Supporting substrate (reads to do before any OGIT-G work)

| File | Size | Why it matters |
|---|---|---|
| `crates/thinking-engine/src/cognitive_stack.rs` | 13 KB | L1–L4 cognitive substrate the OGIT overlay plugs into |
| `crates/thinking-engine/src/cognitive_trace.rs` | 6 KB | Audit trail of cognitive steps; every G's session writes here |
| `crates/thinking-engine/src/l4.rs`, `l4_bridge.rs` | — | L4 spatial-BLAS surface |
| `crates/thinking-engine/src/meaning_axes.rs` | 9 KB | Semantic dimensions (HdrResonance, Archetype, Viscosity) |
| `crates/thinking-engine/src/cronbach.rs` | 12 KB | Psychometric reliability of any per-G codebook (use BEFORE shipping a manifest) |
| `crates/causal-edge/src/edge.rs` | — | CausalEdge64 = 7+1 Bach channels = the 8 predicate layers |
| `crates/bgz17/src/palette_semiring.rs` | — | O(1) distance + compose algebra Pattern H rests on |
| `crates/lance-graph/src/cam_pq/jitson_kernel.rs` | — | JIT muscle for Pattern K |
| `crates/lance-graph/src/graph/arigraph/` | dir | String-keyed SPO position index (Pattern L) |
| `crates/lance-graph-contract/src/cycle_accumulator.rs` | — | Pattern I per-cadence flush gate |

---

## Ledger row reframes (these are the rediscovery traps)

The `ARCHITECTURE_ENTROPY_LEDGER.md` flags several rows as Spaghetti
or High-entropy. Read flat, they invite "fix by consolidating." Read
through the Pattern A–O lens, several are **intentional layering**,
not entropy. Future sessions: do not chase these without checking
the reframe first.

### THINK-1 — *not* a 4-copy drift

**Ledger:** `contract::thinking::ThinkingStyle`(36) +
`planner::thinking::style::ThinkingStyle`(12) +
`thinking-engine::cognitive_stack::ThinkingStyle`(12) +
`thinking-engine::superposition::ThinkingStyle`(5) +
`cognitive-shader-driver::engine_bridge::UNIFIED_STYLES`(12-const) +
RL bandit. Flagged Spaghetti / entropy 5.

**Reframe:** the 12-style codebook lives canonically in
`p64-bridge::STYLES` (the substrate codebook of 4 clusters × 3
styles). The 36-entry `contract::thinking::ThinkingStyle` is the
**composed surface** (12 base styles × 3 modulation levels — same
relationship as a base palette and its tinted derivatives). The
5-entry superposition variant and the bandit selector are
*projections* of the 12-base codebook onto more constrained
selectors. This is Pattern H's vessel mechanism + Pattern K's
learning loop, working as designed. The fix is `From` adapters and
a doc explaining the layering — not collapsing the layers.

### HEEL-1 — *not* three different orderings, one canonical cascade

**Ledger:** `contract::cam::CamByte::{Heel, Branch, TwigA, TwigB,
Leaf, Gamma}` vs `lance-graph::graph::neighborhood::SearchCascade::
{heel, hip, twig, leaf_rerank}` vs `bgz17/router.rs` +
`bgz-tensor/hhtl_d.rs`. Flagged High / entropy 4.

**Reframe:** the canonical HHTL cascade is the four-step
`cognitive_shader::cascade` documented in `p64-bridge/src/lib.rs:288-293`:
HEEL→HIP→TWIG→LEAF. The three "orderings" in the ledger are
different **views** of the same cascade at different points in the
pipeline (CamByte = packed addressing, SearchCascade = neighborhood
walk, bgz17/router = bf16 basin routing). All four bottom out in
`semiring.distance` O(1). Fix is to consolidate the **labels** so
the four views announce themselves as the same cascade — not to
re-implement.

### ADJ-THINK-1 — *not* "aspirational, never written"

**Ledger:** "I5 doctrine: 36 thinking styles are nodes in a CSR/CSC
`AdjacencyStore` at tau-prefix `0x0D`. `tau()` addresses are computed;
**nothing writes those rows.**" Flagged Aspirational / entropy 4.

**Reframe:** the `[[u64; 64]; 8]` `planes` field of
`p64-bridge::CognitiveShader` **IS** the adjacency store. Each
predicate plane is a 64×64 boolean adjacency matrix. The 36 styles
are addressed via the `layer_mask` field of `StyleParams` — that
mask IS the tau-prefix the doctrine asked for. What's missing is the
`tau()` **write API** that exposes "write style S's adjacency into
plane z" to the planner; reads work fine. One method, not a new
crate.

### CRYSTAL-1 — *not* a name collision

**Ledger:** `contract::crystal::sentence::SentenceCrystal` +
`holograph::crystal_dejavu::SentenceCrystal` +
`holograph::sentence_crystal::SemanticCrystal`. Flagged High /
entropy 4.

**Reframe:** these are **two legitimate codebooks at different
Pattern-N layers**. `contract::SentenceCrystal` is the contract-side
sentence-grain crystal (the read-only carrier the orchestrator
hands to consumers). `holograph::SemanticCrystal` is the holographic
internal crystal (excluded crate, internal-grain). Both are valid
codebook entries at different layers of the same Pattern N
hierarchy (fingerprint → crystal → codebook address). The
collision is *nominal*, not semantic. Holograph being excluded
means the consumer never sees both at once. Fix is rename inside
holograph, not unification.

### CAM-DIST-1 — *not* stalled

**Ledger:** "UDF registered at `cam_pq/udf.rs:241,257,326`. Called
from `query.rs:470` only when `with_cam_codebook(...)` is opted
into. `datafusion_planner/mod.rs` does NOT register; default Cypher
path can't reference `cam_distance`." Flagged Stalled / entropy 3.

**Reframe:** this is a **one-line registration fix** in
`DataFusionPlanner::new` — register `cam_distance` unconditionally
the way other UDFs already are. The substrate (the UDF body, the
codebook, the distance table) is shipped. "Stalled" overstates it;
"awaiting one-line wire-up" is accurate. Anyone reaching for the
`cam-pq-production-wiring-v1` plan should grep `DataFusionPlanner::new`
first and confirm the gap is genuinely one line.

---

## Pre-work checklist for future sessions (extends `.claude/patterns.md`)

Before proposing **any** Pattern A–O work, run this checklist in
order. Stop at the first hit.

1. **Pattern-recognition (this file):** is the pattern in the table
   at the top? If yes, the file path in the right column is your
   starting read — not your starting blank file.
2. **CRATE-FIRST (P-1):** `grep -E '^\s+"crates/' Cargo.toml` — if
   the pattern's home crate is a workspace member, it is already
   shipped to consumers.
3. **REGION-FIRST (P-2):** name the R0–R8 region from
   `.claude/knowledge/soa-dto-fma-map.md`. If R5 or R6, the
   substrate is almost certainly thinking-engine or contract.
4. **ENTROPY-FIRST (P-3):** check `ARCHITECTURE_ENTROPY_LEDGER.md`
   for a related row. **Then check the reframes above.** Several
   rows that look like entropy are intentional Pattern A–O layering.
5. **CLUSTER-AWARE (P-5):** if you are touching THINK-1, the cluster
   includes COMPASS-1, TRUST-1, FLOW-1, MUL-ASSESS-1, ADJ-THINK-1 —
   move them together or not at all. Section B of the ledger lists
   the clusters.
6. **CONSULT-DON'T-GUESS (CLAUDE.md Stance):** if a knowledge doc
   has a `READ BY:` header naming your subagent type, load it before
   writing.

The 30-turn rediscovery tax this checklist prevents:

> "Let's build a thinking codebook." → 6 turns of design →
> someone greps and finds p64-bridge::STYLES → rewrite as adapter →
> 4 turns of unwinding the new code → land 1-line adapter.
>
> *vs.*
>
> "Let's build a thinking codebook." → read tier-0-pattern-recognition.md
> Pattern H section → write 1-line adapter.

The five turns of "checklist read" are the cheapest five turns in
the session.

---

## Cross-references

- **W1's master:** `.claude/knowledge/unified-ogit-architecture-v1.md`
  — full Pattern A–O definitions, dependency arrows between
  patterns, and the OGIT-G wiring spec
- **Region map:** `.claude/knowledge/soa-dto-fma-map.md` (R0–R8)
- **Open entropy:** `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md`
  (especially rows THINK-1 / HEEL-1 / ADJ-THINK-1 / CRYSTAL-1 /
  CAM-DIST-1 — see Reframes section above before acting)
- **Topology invariants:** `.claude/board/SINGLE_BINARY_TOPOLOGY.md`
  (especially I-4 Pattern-I gate distinction)
- **Lab-vs-canonical doctrine:** `.claude/knowledge/lab-vs-canonical-surface.md`
  (mandatory before Pattern F endpoint work)
- **VSA-substrate iron rules:** CLAUDE.md sections I-SUBSTRATE-MARKOV /
  I-NOISE-FLOOR-JIRAK / I-VSA-IDENTITIES (mandatory before any
  Pattern M / N change)

---

## Honest scope statement

This doc maps Pattern names → file paths. It does **not**:

- Define what the patterns mean (W1's job)
- Specify the OGIT-G overlay wiring (W1 + W3+ collaborative)
- Author the per-G manifest format (later sprint deliverable)
- Resolve the entropy ledger rows (separate cleanup work)

It does:

- Stop the next session from rebuilding what already exists
- Reframe ledger rows that look like entropy but are Pattern layering
- List the substrate reads required before touching any pattern
- Surface the **one** genuinely new bit (Pattern G + Pattern B's
  manifest)

If a future session reads this and still proposes a parallel
implementation of a shipped pattern, the rediscovery is on them —
not on the workspace.
