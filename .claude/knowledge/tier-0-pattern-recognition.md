# Tier-0 Pattern Recognition — read this BEFORE proposing architectural work

> **MANDATORY READ** for any session proposing Pattern A-O work.
> The Anti-Pattern this doc is designed to prevent: **"Designing What's
> Already Built"** (architectural-scale generalization of the
> Discovery-Loop anti-pattern in `.claude/patterns.md`).
>
> Every pattern letter A-O below uses the **canonical assignment**
> from `.claude/plans/unified-ogit-architecture-v1.md` (W1's master
> synthesis). If you read code/PRs/board files and see a different
> letter mapping, the canonical W1 master and `.claude/patterns.md`
> Pattern Recognition Framework are authoritative.

---

## CRITICAL CORRECTION (2026-05-07, post-PR #358)

**Earlier draft of this file used a non-canonical pattern-letter scheme
that mislabeled Pattern A as "OGIT Triangle / Shipped".** That was
wrong: canonical Pattern A is the **SPO-G u32 OGIT slot in the quad
store, currently DESIGN PHASE** (tracked by TD-OGIT-G-SLOT-1 in
`.claude/board/TECH_DEBT.md`).

If this file ever tells you "Pattern X is shipped" in a way that
contradicts `.claude/plans/unified-ogit-architecture-v1.md` or
`.claude/patterns.md`, the master plan-doc + patterns.md win. **Do
not skip TD-tracked work because of a label collision.**

The file→shipped-substrate insights from the earlier draft are
preserved below under each canonical pattern as supporting evidence
where they apply, and as a separate "Substrate inventory" appendix
where they describe shipped pieces that do NOT correspond to any
canonical A-O pattern letter.

---

## TL;DR — canonical Pattern A-O status (the load-bearing table)

| # | Canonical pattern (from W1 master) | Status | Where to find / What's needed |
|---|---|---|---|
| **A** | **SPO-G with u32 OGIT slot** | 🚧 **DESIGN PHASE** | TD-OGIT-G-SLOT-1; sub-plan in `.claude/plans/ogit-g-context-bundle-v1.md` D-OGIT-G-1 |
| **B** | **Context Bundle per G (typed surface)** | 🚧 **DESIGN PHASE** | TD-CONTEXT-BUNDLE-2; D-OGIT-G-2 in same sub-plan. *Adjacent shipped primitives:* the 8 predicate planes in `p64-bridge::CognitiveShader` are a partial slot-shape (not the typed bundle) |
| **C** | **Generic Bridge dispatching ConsumerPointer** | 🚧 **DESIGN PHASE** | TD-GENERIC-BRIDGE-3; D-OGIT-G-3 in same sub-plan. *PR #29 SmbMembraneGate + PR #98 MedCareMembraneGate are the per-consumer newtypes this replaces* |
| **D** | **Meta-Structure Hydration (OWL/JanusGraph/Foundry → bundle)** | 🚧 **DESIGN PHASE** | per-ontology hydrators TBD; first concrete: FMA OWL hydrator in `.claude/plans/anatomy-realtime-v1.md` PR-ANATOMY-1 |
| **E** | **Compile-Time Consumer Binding (`/modules/<name>/manifest.yaml`)** | 🚧 **DESIGN PHASE** | TD-MANIFEST-MODULES-4; sub-plan `.claude/plans/compile-time-consumer-binding-v1.md` D-MANIFEST-MODULES |
| **F** | **ractor/BEAM Supervisor in Zone 2/3** | 🚧 **DESIGN PHASE — actor shape PROVEN** | TD-RACTOR-SUPERVISOR-5. The handler-arm shape is mechanical from `crates/cognitive-shader-driver/src/grpc.rs` tonic service trait |
| **G** | **Best-Practice Thinking Inheritance per OGIT-G** | 🚧 **DESIGN PHASE** | The 12-style base codebook in `crates/p64-bridge/src/lib.rs::STYLES` is the substrate. The per-G inheritance chain (DOLCE root → Healthcare/Gotham/SMB/CRM extensions) is what's missing |
| **H** | **Switchable Cognitive Vessel** | ✅ **SHIPPED** | `crates/p64-bridge/src/lib.rs::cognitive_shader::CognitiveShader` (8 predicate planes + bgz17 PaletteSemiring + HHTL cascade + `deduce_path`). Doc says: *"No POPCNT. No Hamming. Distance is PRECOMPUTED in the codebook. The mask gates access. The table provides the answer. O(1)."* |
| **I** | **Implicit Cognition (continuous, not request-driven)** | ✅ **SHIPPED** | `crates/lance-graph-contract/src/cycle_accumulator.rs` (PR #337) — the per-cadence flush gate that lets L1 fire continuously and L2 pull on threshold |
| **J** | **INT4-32D Thinking Atoms** | 🚧 **DESIGN PHASE** | TD-INT4-32D-ATOMS-6. *Adjacent shipped primitives:* `crates/thinking-engine/src/reranker_lens.rs` is the lens shape; INT4-32D fingerprint format itself is new |
| **K** | **Circular Compilation (YAML→JIT→YAML→static)** | ⏳ **ASPIRATIONAL** | TD-CIRCULAR-COMPILATION-7. *Precedent muscle:* `crates/lance-graph/src/cam_pq/jitson_kernel.rs` is a YAML/JSON-driven JIT pattern at kernel scale; extending to actor scale is real new infra |
| **L** | **SPO-Chain Narrative (skip Markov)** | ◐ **PARTIALLY SHIPPED** | `crates/lance-graph/src/graph/arigraph/triplet_graph.rs` (string-keyed SPO position index) + `lance-graph-planner::nars::truth::TruthValue` (NARS algebra) exist. *What's new:* MUL marker propagation through pronoun resolution + counterfactual NARS synthesis loop |
| **M** | **Wave-Particle Bimodal Cognition** | ◐ **SHIPPED (primitives)** | Wave: `crates/bgz17`, `crates/holograph::resonance`, `crates/thinking-engine/src/qualia.rs`, `crates/thinking-engine/src/world_model.rs` (the four-section BindSpace world model). Particle: `crates/lance-graph/src/graph/arigraph/`, `crates/lance-graph/src/graph/spo/`, NARS. *What's new:* the per-G blend dial that selects how much wave vs particle for a given task |
| **N** | **Fingerprint-as-Codebook-Address** | ✅ **SHIPPED** | `crates/thinking-engine/src/prime_fingerprint.rs`, `crates/thinking-engine/src/qualia.rs::FAMILY_CENTROIDS` (10 named family centroids), `crates/p64-bridge/src/lib.rs::STYLES` (12-entry thinking-style codebook), `crates/lance-graph/src/cam_pq/` (256-entry palette), `crates/bgz17/` palette codebook. The substrate-level cognitive operation: fingerprint → codebook → O(1) recognition |
| **O** | **Phenomenological Memory Layers** | ✅ **SHIPPED** | `crates/thinking-engine/src/qualia.rs` (39 KB, 17D + 10 family centroids + music calibration via Octave 2:1 / Fifth 3:2 / Third 5:4 / Tritone √2:1 + cross-validation against Jina v3) + `crates/thinking-engine/src/awareness_dto.rs` (meta-awareness DTO) + `crates/causal-edge::CausalEdge64` (Bach 7+1 = canonical 8 logical relations) |

**Net new work for OGIT-G:** Patterns A + B + C + D + E + F + G are
all design phase. Patterns H + I + N + O are fully shipped. Patterns
J + L + M have shipped primitives that need the per-G dial / new
glue. Pattern K is aspirational.

**Read this whenever you see "let's build Pattern X" in a plan-doc.**
If X ∈ {H, I, N, O}, the answer is "no — it's shipped; what you
mean is extending or wiring around it."
If X ∈ {A, B, C, D, E, F, G, J}, the TD-X row in
`.claude/board/TECH_DEBT.md` is the canonical work item.
If X ∈ {L, M}, partial work; check the substrate references first.
If X = K, defer per "aspirational" status unless the JITson kernel
pattern has matured enough to extend to actor scope.

---

## Pattern-by-pattern recognition (canonical letters)

### Pattern A — SPO-G with u32 OGIT slot (DESIGN PHASE)

The quad shape `(S, P, O, G)` where G is a u32 OGIT index. Replaces
oxigraph's IRI-based named-graph slot with O(1) integer indexing.

**Status:** NOT SHIPPED. The current SPO surfaces are 3-tuple:
- `crates/lance-graph/src/graph/spo/` — fingerprint-keyed cold store
- `crates/lance-graph/src/graph/arigraph/triplet_graph.rs` — string-keyed warm cache
- `SpoBridge::promote_to_spo` writer (PR #355 D-ONTO-V5-9) — bridges them

The G slot is missing. `MappingRow` cascade columns from PR #355
contain `ontology_context_id: u32` (CONTEXT-ID-1 ledger row) — this
is the SAME `u32` that becomes the canonical G, but the SPO writer
doesn't yet thread it.

**TD reference:** TD-OGIT-G-SLOT-1 in `.claude/board/TECH_DEBT.md`.
**Sub-plan:** `.claude/plans/ogit-g-context-bundle-v1.md` D-OGIT-G-1.

**Adjacent shipped infrastructure:** Lance MVCC versioning (`Dataset::checkout_latest().version()`)
provides the temporal axis once the G slot lands. `OntologyRegistry`
in `crates/lance-graph-ontology` (PR #355) is the lookup table.

### Pattern B — Context Bundle per G (DESIGN PHASE)

A typed `ContextBundle` struct living in OGIT, addressable by G,
with named slots: `ontology`, `codebook`, `schema`, `labels`,
`vocabulary`, `consumer_pointer`, `thinking_styles`,
`thinking_adjacency`, `qualia_codebook`, `mul_threshold_profile`,
`trust_texture_set`, `flow_state_set`.

**Status:** NOT SHIPPED as a typed surface. Adjacent primitives that
will populate the slots:
- `crates/p64-bridge/src/lib.rs::cognitive_shader::CognitiveShader.planes: [[u64; 64]; 8]` — partial Pattern B slot for the 8-predicate-layer adjacency (one of many bundle slots)
- `crates/thinking-engine/src/qualia.rs::FAMILY_CENTROIDS` — qualia codebook slot content
- `crates/p64-bridge/src/lib.rs::STYLES` — thinking-style codebook slot content
- `crates/lance-graph-contract::mul::MulThresholdProfile` (PR #355) — mul_threshold_profile slot content
- `crates/causal-edge::CausalEdge64` — Pattern B's causal-edge slot type

**What's missing:** the typed `ContextBundle` struct itself + the
`OntologyRegistry::resolve(g: u32) -> Option<&ContextBundle>` API.

**TD reference:** TD-CONTEXT-BUNDLE-2.
**Sub-plan:** `ogit-g-context-bundle-v1.md` D-OGIT-G-2.

### Pattern C — Generic Bridge dispatching ConsumerPointer (DESIGN PHASE)

One canonical `MembraneGate` impl that reads per-G `ConsumerPointer`
from OGIT and dispatches accordingly. Replaces the per-consumer
newtype gates (PR #29 `SmbMembraneGate`, PR #98 `MedCareMembraneGate`).

**Status:** NOT SHIPPED. The per-consumer newtypes work; the generic
dispatcher is what reduces per-new-consumer cost from ~800 LOC
(per `.claude/board/MEDCARE_POLICY_GAP.md`) to ~30 LOC.

**TD reference:** TD-GENERIC-BRIDGE-3.
**Sub-plan:** `ogit-g-context-bundle-v1.md` D-OGIT-G-3.

### Pattern D — Meta-Structure Hydration (DESIGN PHASE)

A trait + implementations: `OwlHydrator`, `JanusGraphHydrator`,
`FoundryObjectHydrator`, `OxigraphRdfHydrator`, etc. Each takes a
source format (TTL / property-graph schema / Foundry export / RDF
quads) and emits a `ContextBundle` per Pattern B.

**Status:** NOT SHIPPED. First concrete deliverable: FMA OWL hydrator
in `.claude/plans/anatomy-realtime-v1.md` PR-ANATOMY-1.

**Adjacent shipped infrastructure:** the existing `OntologyRegistry`
namespace seeding (PR #355 NamespaceRegistry::seed_defaults with 14
mappings: WorkOrder=1, Healthcare=2, Network=3, SMB=0, Medical/{ICD10CM..CHEBI}=10..19)
is hand-coded today; Pattern D hydrators replace this with TTL-driven seeding.

### Pattern E — Compile-Time Consumer Binding (DESIGN PHASE)

`/modules/<name>/manifest.yaml` declares each consumer's G-binding,
ConsumerPointer fields, RBAC, action capabilities. The build script
in `lance-graph-contract` reads all `/modules/*/manifest.yaml` at
compile time and generates Rust glue.

**Status:** NOT SHIPPED. The PostNuke (anno-2000) module pattern is
the precedent.

**TD reference:** TD-MANIFEST-MODULES-4.
**Sub-plan:** `compile-time-consumer-binding-v1.md` D-MANIFEST-MODULES.

### Pattern F — ractor/BEAM Supervisor in Zone 2/3 (DESIGN PHASE — actor shape PROVEN)

Per-consumer ractor actors, supervised in `lance-graph-callcenter`,
with typed `Msg` enums. Each `impl Consumer` exposes its actor type;
the supervisor enumerates them at compile time and spawns the tree.

**Status:** SUPERVISOR NOT SHIPPED. Actor message shape PROVEN by
`crates/cognitive-shader-driver/src/grpc.rs` (~345 LOC tonic service
trait — `dispatch / ingest / health / qualia / styles / tensors /
calibrate / probe`). The translation to ractor handlers is mechanical
(tonic async fn → ractor sync handle, same args, same return types).

**TD reference:** TD-RACTOR-SUPERVISOR-5.
**Sub-plan:** `compile-time-consumer-binding-v1.md` D-RACTOR-SUPERVISOR.

### Pattern G — Best-Practice Thinking Inheritance per OGIT-G (DESIGN PHASE)

Each G slot's bundle declares `thinking_styles: SmallVec<[u8; 8]>` +
`thinking_adjacency: AdjacencyStore<u8>`. Healthcare inherits from
DOLCE + adds clinical-specific (Differential, EvidenceBased,
RiskStratified). Gotham inherits from DOLCE + adds investigation-
specific (LinkAnalytic, AttributionTracing, TimelineReconstructive).

**Status:** NOT SHIPPED as inheritance chain. Adjacent shipped
infrastructure:
- `crates/p64-bridge/src/lib.rs::STYLES` — 12-entry base thinking-style codebook (the substrate)
- `crates/lance-graph-contract::thinking::ThinkingStyle` — 36-entry composed surface
- `crates/thinking-engine/src/cognitive_stack.rs` — runtime selection

The base codebook + composed surface co-exist by design (recognition
that THINK-1 ledger row's "Spaghetti-5 drift" framing was wrong;
they're intentional layering — see W6's reframe in
`.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md`).

**What's missing:** the per-G inheritance chain that lets a consumer
crate inherit DOLCE + extend with domain-specific styles.

### Pattern H — Switchable Cognitive Vessel ✅ SHIPPED

`cognitive-shader-driver` is the L1+L2 vessel; OGIT G is the dispatch
parameter; per-G program selection.

**Already ships:** `crates/p64-bridge/src/lib.rs::cognitive_shader::CognitiveShader`:

```rust
pub struct CognitiveShader<'a> {
    pub planes: [[u64; 64]; 8],         // 8 predicate planes (topology)
    pub semiring: &'a PaletteSemiring,  // bgz17 O(1) distance + compose algebra
    pub k: usize,
}
```

8 predicate planes (CAUSES / ENABLES / SUPPORTS / CONTRADICTS /
REFINES / ABSTRACTS / GROUNDS / BECOMES) = the canonical 8 logical
relations = Bach's 7+1 = `causal_edge::CausalEdge64`'s 7+1 channels.

The HHTL cascade lives at `CognitiveShader::cascade()`:
- HEEL: `layer_mask` gates Z (which kinds of relations matter)
- HIP: `planes[z][block_row]` gates X-Y (topological neighborhood)
- TWIG: 4-archetype-per-block expansion (4×4 refinement)
- LEAF: `semiring.distance(query, target)` (O(1) bgz17 metric lookup)

The doc comment is the thesis: *"No POPCNT. No Hamming. Distance is
PRECOMPUTED in the codebook. The mask gates access. The table
provides the answer. O(1)."*

**HEEL-1 ledger row reframe:** the "3 different orderings" framing
was wrong; the cascade is implemented in this single canonical place.

`CognitiveShader::deduce_path(query, cause_layer, effect_layer, max_hops)`
is Pattern L's chain-of-thought primitive at the substrate.

**What G-overlay needs to add:** thread G through `cascade()` so that
which `planes` and which `semiring` get loaded is per-G data from
ContextBundle (Pattern B).

### Pattern I — Implicit Cognition ✅ SHIPPED

The CycleAccumulator gate from PR #337 lets L1 fire continuously;
L2 pulls on threshold. This is the "can't stop thinking" pattern —
the system thinks in the background, flushes on cadence or pull.

**Already ships:** `crates/lance-graph-contract/src/cycle_accumulator.rs`
(per topology I-4 distinct-naming rule from `SINGLE_BINARY_TOPOLOGY.md`).

### Pattern J — INT4-32D Thinking Atoms (DESIGN PHASE)

32 dimensions × 4 bits = 16-byte cognitive-style fingerprints. Used
for K-NN proximity search over the thinking-style codebook when OGIT
doesn't have best-practice patterns for a new domain yet (e.g.,
hubspo-rs landing without G=CRM thinking_adjacency).

**Status:** NOT SHIPPED. The reranker-lens shape is precedent.

**Adjacent shipped infrastructure:**
`crates/thinking-engine/src/reranker_lens.rs` — switchable reranker
backend. The K-NN proximity step would slot in as one reranker mode.

**TD reference:** TD-INT4-32D-ATOMS-6.

### Pattern K — Circular Compilation ⏳ ASPIRATIONAL

YAML manifest at compile time → static glue. New pattern at runtime
→ JIT-loaded → write back to YAML → next build statically compiles.
Self-extending architecture.

**Status:** ASPIRATIONAL. Real new infra needed (cranelift integration
+ dynamic actor loading).

**Precedent muscle in workspace:** `crates/lance-graph/src/cam_pq/jitson_kernel.rs`
is a YAML/JSON-driven JIT at kernel scale. Extending to actor scale
is the genuine new work.

**TD reference:** TD-CIRCULAR-COMPILATION-7.

### Pattern L — SPO-Chain Narrative (PARTIALLY SHIPPED)

Skip Markov bundling for narrative comprehension. Parse to SPO
triples; AriGraph indexes by (page, sentence, word, role); pronoun
resolution via prior SPO context; MUL markers for ambiguity; NARS
counterfactual synthesis.

**Already ships:**
- `crates/lance-graph/src/graph/arigraph/triplet_graph.rs` — string-keyed SPO + position index
- `crates/lance-graph-planner::nars::truth::TruthValue` — NARS algebra (revise / deduce / induce / abduce)
- `crates/lance-graph-contract::mul::MulAssessment` + `MulThresholdProfile` (PR #355)
- `crates/p64-bridge/src/lib.rs::cognitive_shader::CognitiveShader::deduce_path` — chain-of-thought primitive
- `crates/thinking-engine/src/branching.rs` — branch-based counterfactual synthesis
- `crates/thinking-engine/src/ghosts.rs` — counterfactual "ghost" branches (Pattern L's MUL ambiguity)

**What's new:** the MUL marker propagation through pronoun resolution
+ Lance MVCC time-travel for "read book as of chapter 5" partial-state
queries.

### Pattern M — Wave-Particle Bimodal Cognition (PRIMITIVES SHIPPED)

Wave: bgz17/resonance/qualia distributed continuous (BNN-like, plastic).
Particle: SPO-G/AriGraph/NARS discrete queryable.

**Already ships (wave side):**
- `crates/bgz17/` — palette + distance matrix + semiring (the wave algebra)
- `crates/holograph/src/resonance.rs` — wave-mode similarity
- `crates/thinking-engine/src/qualia.rs` — 17D phenomenology (wave)
- `crates/thinking-engine/src/world_model.rs` (13 KB) — `SelfState / UserState / FieldState / ContextState` four-section BindSpace world model. (This was the "OGIT Triangle / four contexts" insight from the earlier draft of this doc — preserved here as substrate, not as a separate Pattern A.)
- `crates/thinking-engine/src/superposition.rs` — interference-field machinery cited by `qualia.rs::from_superposition`

**Already ships (particle side):**
- `crates/lance-graph/src/graph/arigraph/`
- `crates/lance-graph/src/graph/spo/`
- `crates/lance-graph-planner::nars::*`

**What's new:** the per-G blend dial that selects wave vs particle
weight per task per consumer.

### Pattern N — Fingerprint-as-Codebook-Address ✅ SHIPPED

The substrate-level cognitive operation: `fingerprint(content) →
codebook lookup → O(1) recognition`. Recognition (codebook hit) is
most of cognition; crystallization (codebook miss = new entry) is rare.

**Already ships across multiple codebooks:**
- `crates/thinking-engine/src/prime_fingerprint.rs` (15 KB) — content-addressable prime-coded fingerprints
- `crates/thinking-engine/src/qualia.rs::FAMILY_CENTROIDS` — 10 named family centroids (emberglow, woodwarm, steelwind, oceandrift, frostbite, sunburst, nightshade, thornrose, velvetdusk, stormbreak)
- `crates/p64-bridge/src/lib.rs::STYLES` — 12-entry thinking-style codebook (analytical, convergent, systematic, creative, divergent, exploratory, focused, diffuse, peripheral, intuitive, deliberate, metacognitive)
- `crates/lance-graph/src/cam_pq/` — 256-entry CAM-PQ palette
- `crates/bgz17/` — palette codebook + 256×256 distance table
- `crates/deepnsm/src/codebook.rs` — vocabulary codebook
- `crates/causal-edge::CausalEdge64` (8 channels = Bach 7+1) — per-edge fingerprint

**Substrate clarification:** Vsa16kF32 is NOT the canonical substrate.
It's a special-purpose Markov-accumulation mode (one program among
many that the cognitive-shader-driver vessel runs). The actual
substrate is the codebook collection per G. See `.claude/patterns.md`
"Substrate clarification" section + W6's VSA-1 ledger reframe.

### Pattern O — Phenomenological Memory Layers ✅ SHIPPED

Multiple parallel memory traces per experience: SPO + Qualia17D +
CausalEdge64 + Resonance + Epiphany + meta-awareness.

**Already ships:** `crates/thinking-engine/src/qualia.rs` (39 KB):
- 17D qualia substrate calibrated from musical interval ratios
- Octave (2:1) → arousal, Fifth (3:2) → valence, Third (5:4) → warmth, Tritone (√2:1) → tension
- Universal invariance: a fifth sounds like a fifth in any register
- Cross-validated against 220 Jina v3 calibrated pairs in Upstash
- 10 family centroids + emotional blend (second-order codebook)
- Multiple `from_*` constructors: `from_convergence`, `from_engine`, `from_superposition`, `from_band_energies` — each is a different memory layer projecting INTO the same 17D space
- Bidirectional bridge: Qualia17D ↔ musical mode ↔ band energies ↔ voice channels
- `verify_grid_completeness()` — formal proof that all 17 dims are covered

Plus:
- `crates/thinking-engine/src/awareness_dto.rs` (12 KB) — meta-awareness DTO (the recursive layer)
- `crates/causal-edge::CausalEdge64` — Bach 7+1 = canonical 8 logical relations
- `crates/holograph/src/epiphany.rs` + `.claude/board/EPIPHANIES.md` — high-salience meta-cognition crystallization

---

## Substrate inventory (shipped pieces that don't map 1:1 to canonical Pattern A-O)

These are real shipped substrate files that the earlier draft of this
doc misattributed as "Pattern A through Pattern G". They're preserved
here as supporting infrastructure for the canonical patterns above:

| Substrate | File | Supports canonical pattern(s) |
|---|---|---|
| WorldModel four-section split (T/A/P + W as four contexts) | `crates/thinking-engine/src/world_model.rs` (13 KB; SelfState/UserState/FieldState/ContextState) | Pattern M wave-side; future Pattern B `world_state` slot |
| Per-G Tokenizer Registry | `crates/thinking-engine/src/tokenizer_registry.rs` (16 KB) | Pattern B `vocabulary` slot infrastructure |
| Persona Runtime | `crates/thinking-engine/src/persona.rs` (18 KB) | Pattern C `consumer_pointer.persona` field; per-G persona selection |
| Sensor Bridge | `crates/thinking-engine/src/sensor.rs` (13 KB) | Pattern E ingestion-side; consumer-actor input handlers |
| Reranker Lens | `crates/thinking-engine/src/reranker_lens.rs` (7 KB) | Pattern J substrate (proximity reranker is one switchable backend) |
| Crystallization-Side Learning | `crates/thinking-engine/src/contrastive_learner.rs` (9 KB) | Pattern K crystallization step (codebook entry generation) |
| MUL Counterfactual Branching | `crates/thinking-engine/src/branching.rs` (10 KB) + `ghosts.rs` (11 KB) | Pattern L MUL marker propagation |
| Cronbach reliability | `crates/thinking-engine/src/cronbach.rs` (11 KB) | Pattern N codebook quality metric |
| Meaning axes | `crates/thinking-engine/src/meaning_axes.rs` (9 KB) | Pattern G semantic dimensions |
| Composite/dual/signed engines | `crates/thinking-engine/src/{composite,dual,signed,bf16,f32}_engine.rs` | Pattern H multiple-substrate vessel modes |

---

## Pre-work checklist (extends `.claude/patterns.md` P-1..P-5)

Before proposing **any** Pattern A-O work:

```
[ ] Read this Tier-0 doc to identify canonical pattern letter (NOT a
    creative re-letter; use the W1 master assignment)
[ ] If canonical pattern is SHIPPED (H, I, N, O fully shipped; M
    primitives shipped; L partially shipped) → propose extension or
    wiring, NOT construction
[ ] If canonical pattern is DESIGN PHASE (A, B, C, D, E, F, G, J)
    → cite the TD-X row in `.claude/board/TECH_DEBT.md` AND the
    sub-plan in `.claude/plans/` BEFORE writing code
[ ] If canonical pattern is ASPIRATIONAL (K) → defer unless the
    JITson kernel pattern has matured to actor scope
[ ] Check `.claude/patterns.md` Pattern Recognition Framework
    (W3-appended section) for any state changes since this Tier-0 doc
[ ] Check `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` and
    `_RESOLVED.md` for ledger row state changes
[ ] If your work spans multiple patterns, cite all letters + the
    cluster in `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` Section B
[ ] Cite `file:line` for every "currently broken" or "not shipped"
    claim in your proposal
```

If any checklist box reveals existing work, do NOT propose new work.
Either:
- Update the existing row's status (state change) per ledger Update Protocol
- Cite the existing as canonical and propose only the missing edge
- Stop and ask the user

---

## Cross-references

- **`.claude/plans/unified-ogit-architecture-v1.md`** (W1) — master synthesis (authoritative for Pattern letter assignments)
- **`.claude/patterns.md`** (W3-appended Pattern Recognition Framework) — second authoritative source on pattern letters
- **`.claude/board/TECH_DEBT.md`** (W5) — TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11 (all canonical Pattern A-O work items)
- **`.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md`** (OPEN) — W6's reframes of THINK-1, HEEL-1, ADJ-THINK-1, CRYSTAL-1, CAM-DIST-1, VSA-1
- **`.claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md`** — RECOGNITION-1 meta-row (W7-rev2)
- **`.claude/plans/ogit-g-context-bundle-v1.md`** (W10-rev2) — Tier-1 sub-plan: Patterns A + B + C
- **`.claude/plans/compile-time-consumer-binding-v1.md`** (W11) — Tier-2 sub-plan: Patterns E + F
- **`.claude/plans/anatomy-realtime-v1.md`** (W12) — proof-of-vision: exercises every pattern A-O end-to-end
- **`.claude/board/EPIPHANIES.md`** — 17 dated epiphanies E-OGIT-1 through E-RECOGNITION-OVER-DESIGN-17
- **`.claude/board/sprint-log-2/sprint-summary.md`** — sprint-2 closure summary
- **`.claude/board/SINGLE_BINARY_TOPOLOGY.md`** — three-layer architecture invariants (I-1 single binary, I-2 tokio outbound only, I-3 BBB compile-time, I-4 per-row vs per-cadence gates distinct)

## Maintenance

This file is the load-bearing entry point for future architectural
work. Maintain ruthlessly:

1. **Pattern letter assignments are immutable** — they come from the
   W1 master synthesis. Future workers must not invent new letter schemes.
2. **Status changes** (DESIGN PHASE → SHIPPED) get a dated note when
   a TD-X is closed. Move the canonical line in the TL;DR table; do
   not rewrite history.
3. **New shipped substrate files** that don't map to a canonical
   letter go in the "Substrate inventory" appendix.
4. **If you find a contradiction with W1 master or W3 patterns.md,
   they win.** Open a follow-up to fix this Tier-0 doc.
