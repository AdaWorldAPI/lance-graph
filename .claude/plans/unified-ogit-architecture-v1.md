# Unified OGIT Architecture — v1 (Master Synthesis)

> **APPEND-ONLY governance.** This document is the master plan-doc synthesizing
> 15 architectural patterns (A-O) crystallized from a 16-turn architectural
> conversation. It is the authoritative tiered roadmap for the OGIT cognitive
> substrate. **Do not retro-edit.** New findings ship as `unified-ogit-architecture-v2.md`
> (or v1-ERRATUM.md for tactical corrections). Sub-plans referenced here may
> evolve independently with their own append-only discipline.
>
> **Authoritative companion docs (this sprint):**
> - `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` — reframed by W6 (see "Ledger Reframes" below)
> - `.claude/board/EPIPHANIES.md` — append-batch by W4 covering this sprint's
>   recognitions (Patterns H/N/O are shipped substrate, not aspirational)
> - `.claude/board/TECH_DEBT.md` — append-batch by W5 covering wiring gaps that
>   remain after Tier-0 recognition
> - `.claude/plans/ogit-g-context-bundle-v1.md` — sub-plan by W10 (Tier 1 detail)
> - `.claude/plans/compile-time-consumer-binding-v1.md` — sub-plan by W11 (Tier 1 detail)
> - `.claude/plans/anatomy-realtime-v1.md` — north-star demo by W12

---

## 0. Why This Document Exists

The architectural conversation that produced this synthesis spanned 16 turns and
crossed three previously-disjoint discussions:

1. **Surface-level "drift" cleanup** — the entropy ledger was flagging multiple
   rows as duplication (THINK-1, HEEL-1, ADJ-THINK-1, CRYSTAL-1) when in fact
   each was a deliberate codebook-layer separation.
2. **Membrane proliferation** — `SmbMembraneGate`, `MedCareMembraneGate`, and
   pending newtypes for each future consumer (Q2, callcenter, Gotham, …) were
   on track to become an N×M maintenance liability.
3. **Cognitive shader unification** — `crates/p64-bridge::CognitiveShader` and
   `crates/thinking-engine/` had accreted shipped substrate for predicate
   planes, qualia, fingerprints, and HHTL cascade, but **no plan-doc had named
   it as a unified phenomenological architecture** yet.

The synthesis: **everything dispatches on a single u32 OGIT slot `G`** (Pattern A),
and **G selects a ContextBundle** (Pattern B) that drives a **generic bridge**
(Pattern C). The membrane proliferation collapses to one trait with N data
rows. The "drift" rows are not drift — they are intentional Pattern N layering.
The shader substrate is already a switchable vessel (Pattern H) — it just
needs the G-overlay wiring (Tier 1) to expose what it already does.

This document is the master roadmap. Read it first; then descend into the three
sub-plans (W10/W11/W12) for implementation detail; then update the ledger,
epiphanies, and tech-debt boards.

---

## 1. The 15 Patterns

For each pattern below: **definition**, **status**, **code refs**, and where it
fits in the tiered plan.

### Pattern A — SPO-G with u32 OGIT slot

**Definition.** Every triple in the unified store carries a fourth slot `G:
u32` that names the ontology / context graph. `G` is **not** an IRI — it is
a compact handle that resolves via a static codebook to a `ContextBundle`. The
shape is **oxigraph-like** (SPO + named graph), but the named graph index is a
u32 not an IRI, giving O(1) bundle lookup with cache-line-friendly layout.

**Status.** Partially shipped. The SPO-shaped storage exists; the G slot is
implicit in current named-graph fields but not yet typed as `u32` with a
codebook lookup.

**Code refs.** Storage shape in `crates/lance-graph/`, ontology codebook
patterns in `crates/graph-ontology/`. Named-graph plumbing in
`crates/lance-graph-rdf/` (oxigraph integration).

**Tier.** Tier 1.

### Pattern B — Context Bundle per G

**Definition.** Each `G` resolves to a record of **twelve** fields:

```text
ContextBundle {
    ontology:            OntologySnapshot,         // TBox / TTL hydrated form
    codebook:            FingerprintCodebook,      // Pattern N lookup
    schema:              ArrowSchema,              // physical column layout
    labels:              LabelRegistry,            // human-readable surface
    vocabulary:          VocabRegistry,            // domain terms
    consumer_pointer:    ConsumerPointer,          // Pattern C dispatch
    thinking_styles:     ThinkingStyleCodebook,    // per-G codebook of styles
    thinking_adjacency:  AdjacencyPlanes,          // Pattern G inheritance
    qualia_codebook:     QualiaCodebook,           // Pattern O dimensions
    trust_texture_set:   TrustTextureSet,          // per-G trust gradients
    flow_state_set:      FlowStateSet,             // per-G flow profiles
    mul_threshold_profile: MULThresholdProfile,    // Pattern L ambiguity gating
}
```

**Status.** Open. Some fields exist as standalone types across crates
(`thinking-engine::qualia::FAMILY_CENTROIDS`, `p64-bridge::STYLES`,
`graph-ontology::*`) — Tier 1 binds them under one struct.

**Code refs.** Field-level shipped pieces named per-row above; the bundle
type itself is new.

**Tier.** Tier 1 (definition); Tier 3 (per-G specialization fills the
codebook/adjacency/qualia fields).

### Pattern C — Generic Bridge dispatching per-G ConsumerPointer

**Definition.** A single `GenericBridge` trait replaces the hand-written
`SmbMembraneGate`, `MedCareMembraneGate`, and all future per-consumer
membrane newtypes. The bridge takes `(G, action, payload)` and dispatches via
`ContextBundle.consumer_pointer` (a fn ptr table or vtable). One trait, N data
rows; not N traits with M methods each.

**Status.** Open. Current code has `SmbMembraneGate` and `MedCareMembraneGate`
as separate newtypes; this pattern collapses them.

**Code refs.** Current per-consumer membranes in `crates/smb-office-rs/` and
`crates/medcare-rs/`. Target: a single `crates/generic-bridge/` (or method
on `crates/p64-bridge/`) plus per-consumer data tables.

**Tier.** Tier 1.

### Pattern D — Meta-Structure Hydration

**Definition.** Adding a new ontology (FMA, SNOMED, DOLCE, Foundry types,
JanusGraph schemas) does **not** require a new Rust crate. It requires:
(a) the TTL/OWL/JSON source, and (b) a tiny "hydrator entry" — typically
~20 lines of code — that registers a parser keyed by source format. The
hydrator emits a `ContextBundle` keyed by the new `G`. FMA hydration is the
proof case for this (75 K classes, OWL source, hydrated into the bundle without
touching the bridge or shader code).

**Status.** Partially shipped. RDF/oxigraph hydration shape exists in
`crates/lance-graph-rdf/`; FMA hydration is the north-star validation
(see `anatomy-realtime-v1`).

**Code refs.** `crates/lance-graph-rdf/`, existing OWL/RDF plumbing.
Target hydrator entry table: per-format dispatch in the new Tier-1 module.

**Tier.** Tier 1 (mechanism); proof in `anatomy-realtime-v1` (W12).

### Pattern E — Compile-Time Consumer Binding

**Definition.** **Cargo dep presence** at workspace level determines which
`G` bundles are **active** (hot, mmap-resident, hydrated) versus **inert**
(present in codebook, not loaded). A PostNuke-style
`/modules/<name>/manifest.yaml` per consumer crate names its `G` values, its
schema version, and its dependencies. A build-script reads the manifest set
and emits a static glue file (the dispatcher table) at compile time. Versioned
`(G, version)` tuples handle schema evolution: bundles are immutable once
sealed; a schema change is a new version row, not a mutation.

**Status.** Open. The Cargo workspace and per-crate structure exist; the
manifest convention and build-script are new.

**Code refs.** Workspace `Cargo.toml`, per-consumer crates. Target:
`build.rs` in a new `crates/ogit-bind/` or top-of-workspace build hook.

**Tier.** Tier 1. Sub-plan `compile-time-consumer-binding-v1.md` (W11).

### Pattern F — ractor/BEAM Supervisor in Zone 2/3

**Definition.** Per-consumer **ractor actors** in `lance-graph-callcenter`
provide BEAM-style supervision for Zone-2 (hot, low-latency) and Zone-3
(durable, replayable) operations. The actor's message shape is **proved by
the existing gRPC service trait** in
`crates/cognitive-shader-driver/src/grpc.rs` (dispatch / ingest / health /
qualia / styles / tensors / calibrate / probe). One actor per `(consumer, G)`
pair; restart strategy follows OTP supervisor semantics; messages flow over
mpsc inside the binary or over gRPC across binaries.

**Status.** Partially shipped. gRPC service trait shape is shipped and
production-shaped; the ractor port and supervisor tree are open.

**Code refs.** `crates/cognitive-shader-driver/src/grpc.rs` (proof of message
shape), `crates/lance-graph-callcenter/` (target home for supervisors).

**Tier.** Tier 2.

### Pattern G — Best-Practice Thinking Style Inheritance per G

**Definition.** DOLCE provides the **root** thinking-style context (the upper
ontology's canonical decomposition: endurant / perdurant / abstract / quality
…). Domain `G`s (Healthcare, Gotham, SMB, CRM) **inherit and extend**: each
G's `ThinkingStyleCodebook` overlays domain-specific styles on the DOLCE base.
New domain `G`s without curated best-practices fall back to Pattern J (INT4-32D
nearest-neighbour over the union codebook).

**Status.** Partially shipped. `p64-bridge::STYLES` is the 12-entry root
codebook; `contract::thinking::ThinkingStyle` is the 36-entry composed
surface. The inheritance / overlay mechanism per G is open.

**Code refs.** `crates/p64-bridge/src/lib.rs` (`STYLES`),
`contract::thinking::ThinkingStyle`.

**Tier.** Tier 3.

### Pattern H — Switchable Cognitive Vessel

**Definition.** `cognitive-shader-driver` **is** the vessel. The OGIT `G`
slot is the dispatch parameter that switches the vessel's "personality" —
predicate weights, semiring palette, HHTL cascade thresholds, persona
projection — without swapping the binary. One vessel, N modes selected by `G`.

**Status.** **Already shipped.** `crates/p64-bridge::CognitiveShader` ships
the 8 predicate planes (CAUSES / ENABLES / SUPPORTS / CONTRADICTS / REFINES /
ABSTRACTS / GROUNDS / BECOMES), the `bgz17::PaletteSemiring`, the HHTL cascade
(`HEEL=layer_mask`, `HIP=mask row`, `TWIG=block expansion 4×4`,
`LEAF=semiring.distance O(1)`), and the 12-entry `STYLES` codebook. What's
missing is the `G`-selectable mode binding (Tier 1) and exposure via tau-write
API (Pattern ADJ-THINK-1 in the ledger).

**Code refs.** `crates/p64-bridge/src/lib.rs`, `crates/cognitive-shader-driver/`.

**Tier.** Tier 0 (recognize); Tier 1 (G-overlay wiring).

### Pattern I — Implicit Cognition

**Definition.** Continuous background L1 cycles run independent of inbound
requests. A `CycleAccumulator` integrates over cycles and decides when to
**flush** changes to persistent state. This is non-request-driven cognition —
the substrate "thinks" between calls, accruing meaning-axis updates,
re-fingerprinting, and consolidating qualia traces.

**Status.** Open. The L1 / L4 layers exist in `thinking-engine` but the
accumulator + flush policy is unspecified.

**Code refs.** `crates/thinking-engine/src/l4.rs`, candidate
`thinking-engine::CycleAccumulator` (new).

**Tier.** Tier 2 (lives under the ractor supervisor as a background actor).

### Pattern J — INT4-32D Thinking Atoms

**Definition.** A thinking-style atom is **32 nibbles in 16 bytes**: an
INT4-quantized 32-dim cognitive-proximity vector. K-NN over the per-G
thinking-style codebook handles the bootstrap case where a new domain has
no curated best-practices — the atom retrieves the nearest known style and
the bridge proceeds with that style as a working hypothesis (recorded under
Pattern L's MUL marker for revisit).

**Status.** Open. Substrate exists for fingerprinting and codebook lookup;
the 32D INT4 quantization layer is new.

**Code refs.** Fingerprint substrate in
`crates/thinking-engine/src/prime_fingerprint.rs`; codebook patterns in
`crates/p64-bridge/src/lib.rs` (`STYLES`).

**Tier.** Tier 3.

### Pattern K — Circular Compilation

**Definition.** Two-phase loop. **Build time:** YAML manifests
(Pattern E) compile into static glue. **Runtime:** new patterns observed
through cycles JIT into runnable code via `ractor` + `cranelift`, then write
their crystallized form back into a YAML manifest. **Next build:** the new
manifest compiles statically. The substrate learns over time and lays new
muscle into the static codebase. Existing `jitson_kernel` in `cam_pq` is the
JIT muscle.

**Status.** Open (compositionally — every piece exists; the loop closure is new).

**Code refs.** `crates/cam_pq/src/jitson_kernel.rs` (JIT primitive),
build-script from Pattern E.

**Tier.** Tier 4.

### Pattern L — SPO-Chain Narrative Comprehension

**Definition.** Comprehension of long-form text **skips** Markov
n-gram bundling and instead parses to SPO triples directly. The AriGraph index
keys triples by `(page, sentence, word, role-position)` for locality.
Pronoun / coreference resolution reads from prior-context triples. Ambiguous
parses get a **MUL marker** for later resolution. NARS-style counterfactual
synthesis explores branches. **Books are first-class OGIT `G` bundles.**

**Status.** Open. AriGraph plumbing exists in part; the role-position index
and MUL marker semantics are new.

**Code refs.** AriGraph wire-up in `crates/lance-graph-rdf/` and surrounding
crates; NARS / counterfactual primitives in `crates/thinking-engine/`.

**Tier.** Tier 3.

### Pattern M — Wave-Particle Bimodal Cognition

**Definition.** Two parallel cognitive modes blend per-G:
- **Wave** (distributed, continuous): `bgz17` semiring, resonance, qualia
  superposition, palette-space arithmetic.
- **Particle** (discrete, symbolic): SPO-G triples, AriGraph index, NARS
  inference, codebook lookups.

Each `G` carries blend weights (e.g., poetry-G favours wave; legal-G favours
particle). Brain-plasticity analogy: weights drift with experience.

**Status.** Partially shipped. Both halves exist in isolation (`bgz17`
semiring on the wave side, SPO-G / NARS on the particle side); per-G blend
weights and the blend operator are new.

**Code refs.** `crates/bgz17/`, `crates/thinking-engine/src/superposition.rs`
(wave); `crates/lance-graph-rdf/`, NARS pieces in `crates/thinking-engine/`
(particle).

**Tier.** Tier 3.

### Pattern N — Fingerprint-as-Codebook-Address

**Definition.** The **universal cognitive operation**:
`fingerprint(content) → codebook lookup → O(1) recognition`. Every recognition
in the system — style, qualia family, persona, palette colour, pattern — uses
this shape. Different layers have different codebooks at different
granularities; the operation is the same.

**Status.** **Already shipped** across at least five places:
- `thinking-engine::prime_fingerprint`
- `qualia::FAMILY_CENTROIDS` (10 named families)
- `p64-bridge::STYLES` (12-entry)
- `cam_pq` codebook
- `bgz17` palette.

**Code refs.** Listed above. This is **why** the entropy-ledger rows
THINK-1 / CRYSTAL-1 are not drift: they are different codebooks at different
Pattern-N layers.

**Tier.** Tier 0 (recognize).

### Pattern O — Phenomenological Memory Layers

**Definition.** Memory is layered:
- **SPO** (relational)
- **Qualia17D** (subjective-quality vector, 17 dims)
- **CausalEdge64** (7+1 Bach channels — six counterpoint voices + bass + meta)
- **Resonance** (wave-mode trace)
- **Epiphany** (recognized novelty marker)
- **Meta-awareness** (the substrate observing its own state).

Music calibration is load-bearing: octave → arousal, fifth → valence, third →
warmth, tritone → tension. The 7+1 Bach counterpoint structure isomorphic to
CausalEdge64's 7+1 channels is not a metaphor — it is the literal layout.

**Status.** **Already shipped.** `crates/thinking-engine/src/qualia.rs` has
17 dims, 10 family centroids, and music-calibrated dim semantics.
`causal-edge::CausalEdge64` ships 7+1 channels. The integration into a single
"phenomenological memory" view is the open part.

**Code refs.** `crates/thinking-engine/src/qualia.rs`,
`crates/causal-edge/` (`CausalEdge64`), `crates/thinking-engine/src/awareness_dto.rs`.

**Tier.** Tier 0 (recognize); Tier 3 (integrate per G).

---

## 2. Already-Shipped Substrate (Tier-0 Recognition Map)

This sprint's biggest finding: **we have more than we thought**. The map below
is the file → pattern crosswalk so future work doesn't rebuild what exists.

| File / module | Pattern | Notes |
|---|---|---|
| `crates/thinking-engine/src/qualia.rs` | O, N | 17 dims, 10 family centroids, music-calibrated semantics |
| `crates/thinking-engine/src/prime_fingerprint.rs` | N | Fingerprint primitive |
| `crates/thinking-engine/src/world_model.rs` | I, L | World-state container |
| `crates/thinking-engine/src/superposition.rs` | M (wave) | Wave-mode superposition |
| `crates/thinking-engine/src/awareness_dto.rs` | O | Meta-awareness DTO |
| `crates/thinking-engine/src/branching.rs` | L | Counterfactual / branching primitive |
| `crates/thinking-engine/src/domino.rs` | I | Cycle propagation |
| `crates/thinking-engine/src/ghosts.rs` | L | Pronoun / coreference candidates |
| `crates/thinking-engine/src/persona.rs` | G, N | Persona codebook |
| `crates/thinking-engine/src/reranker_lens.rs` | G | Per-context reranking |
| `crates/thinking-engine/src/contrastive_learner.rs` | J, K | Bootstrap learner |
| `crates/thinking-engine/src/cronbach.rs` | (stats) | Reliability |
| `crates/thinking-engine/src/meaning_axes.rs` | O | Meaning-axis dimensions |
| `crates/thinking-engine/src/l4.rs` | I | L4 cycle layer |
| `crates/thinking-engine/src/signed_engine.rs` | M | Signed / polarity wave |
| `crates/p64-bridge/src/lib.rs` (`CognitiveShader`) | H, A | 8 predicate planes + bgz17 + HHTL + STYLES |
| `crates/cognitive-shader-driver/src/grpc.rs` | F | Proof of ractor message shape |
| `crates/causal-edge/` (`CausalEdge64`) | O | 7+1 Bach channels |
| `crates/bgz17/` (`PaletteSemiring`) | H, M | O(1) distance + compose |
| `crates/cam_pq/src/jitson_kernel.rs` | K | JIT muscle |

**Rule.** Before writing a new module, check this map. If a row covers your
need, extend it; do not duplicate.

---

## 3. Ledger Reframes

Five rows currently flagged Spaghetti / Aspirational in
`ARCHITECTURE_ENTROPY_LEDGER.md` are **not drift**. W6 reframes them in the
ledger; the rationale is recorded here so future agents understand the
distinction:

- **THINK-1** (entropy 5 → reframe). **Not** a 4-copy drift. It is a
  **12-base codebook** (`p64-bridge::STYLES`) plus a **36-entry composed
  surface** (`contract::thinking::ThinkingStyle`). Intentional Pattern-N
  layering. Action: rename / document, do not collapse.
- **HEEL-1** (entropy 4 → reframe). **Not** "3 orderings of HHTL". The HHTL
  cascade is a single canonical impl in
  `p64-bridge::cognitive_shader::cascade` — the four letters name layers
  in one pipeline: `HEEL = layer_mask`, `HIP = mask row`,
  `TWIG = block expansion 4×4`, `LEAF = semiring.distance O(1)`. Action:
  document the letter→layer map; close the row.
- **ADJ-THINK-1** (entropy 4 → reframe). **Not** Aspirational. The
  `[u64; 64]; 8` planes inside `p64-bridge::CognitiveShader` **are** the
  adjacency store. Missing piece: a `tau()` write API. Action: keep open
  but reclassify from Aspirational to Implementation-Gap.
- **CRYSTAL-1** (entropy 4 → reframe). **Not** a name collision. Two
  **legitimate codebooks at different Pattern-N layers**. Action:
  disambiguate via path prefix, do not merge.
- **CAM-DIST-1** (entropy 3 → reframe). One-line registration fix.
  Substrate is shipped. Action: ship the registration in Tier 1.

See `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` (W6's append) for the
canonical reframe text.

---

## 4. Tiered Plan

### Tier 0 — Recognition (no code; documentation pass)

**Goal.** Write down what already exists so the next four tiers don't rebuild it.

**Deliverables.**
- This document (Section 2 map).
- W2's Tier-0 doc: deeper file-by-file walk, including LOC counts and
  test-coverage notes (`.claude/plans/<W2's filename>.md`).
- W4's epiphany batch on EPIPHANIES.md acknowledging Patterns H/N/O are
  shipped, not aspirational.
- W6's ledger reframe rows (above).

**Acceptance.** Every shipped row in Section 2 has an open issue or planned
sub-task referencing it for Tier-1+. No new pattern proposals can ship without
checking this map first.

**Exit criterion.** Tier 1 cannot start until Section 2 is validated against
the current `main` (i.e., the files named still exist and contain the named
primitives).

### Tier 1 — G-Overlay Wiring (Patterns A + B + C + E)

**Goal.** Bind everything to a single `u32 G` slot with a static codebook.

**Deliverables.**
1. **Pattern A.** Add `g: u32` slot to the SPO row type in `lance-graph` and
   update the storage codec. Backward-compatible via default-G = 0 (DOLCE).
2. **Pattern B.** Define `ContextBundle` (12 fields, exactly as in Section 1).
   Field types reference existing crate types where they exist (qualia codebook,
   thinking-style codebook, ontology snapshot); placeholder types where they
   don't.
3. **Pattern C.** Define `GenericBridge` trait. Port `SmbMembraneGate` and
   `MedCareMembraneGate` to data rows over the new trait. Delete the two
   newtypes.
4. **Pattern E.** Define `manifest.yaml` schema. Write `build.rs` that reads
   all per-crate manifests and emits the static dispatcher. `(G, version)`
   tuples in manifests; immutable bundles.

**Sub-plans.**
- `.claude/plans/ogit-g-context-bundle-v1.md` (W10) — covers (1) + (2) in
  detail.
- `.claude/plans/compile-time-consumer-binding-v1.md` (W11) — covers (4) in
  detail.

**Acceptance.**
- Round-trip test: insert SPO-G, query by G, dispatch through GenericBridge,
  observe correct per-G behaviour for both smb-office-rs and medcare-rs.
- The two old membrane gates are deleted, not deprecated.
- A new ontology (small synthetic, ~10 classes) can be added via TTL + 20-line
  hydrator entry without touching any other code.

**Exit criterion.** All Tier-1 acceptance tests green; `LATEST_STATE.md`
updated.

### Tier 2 — Supervised Consumer Mesh (Pattern F)

**Goal.** Port the gRPC service trait shape to ractor actors and stand up the
first two consumers under supervision.

**Deliverables.**
1. **ractor port.** A `ConsumerActor<G>` whose `Message` enum mirrors the
   gRPC service: `Dispatch`, `Ingest`, `Health`, `Qualia`, `Styles`, `Tensors`,
   `Calibrate`, `Probe`.
2. **Supervisor tree.** A `ConsumerSupervisor` per-G with OTP restart strategy.
   Lives in `crates/lance-graph-callcenter/`.
3. **First two consumers wired.** `medcare-rs` and `q2` bind via
   `ConsumerPointer` (their manifests resolved by Pattern E). Each gets a
   supervised actor instance.
4. **Pattern I hook.** Background `CycleAccumulator` actor (one per `G`) under
   the same supervisor.

**Acceptance.**
- Kill -9 a consumer actor; supervisor restarts within its budget.
- Both medcare and q2 actors handle a real ingest + dispatch round-trip via
  the GenericBridge.
- CycleAccumulator runs idle (no inbound requests) and produces measurable
  cycle metrics over 60 s.

**Exit criterion.** Two-actor mesh stable for 24 h under synthetic load.

### Tier 3 — Per-G Specialization (Patterns G + J + L + M + O)

**Goal.** Populate the per-G fields of `ContextBundle` with real codebooks
and adjacency tables. Wire the phenomenological memory integration.

**Deliverables.**
1. **Pattern G inheritance.** DOLCE root codebook → Healthcare / Gotham / SMB /
   CRM overlays. Each overlay names the styles it adds, replaces, or removes.
2. **Pattern J bootstrap.** INT4-32D quantizer over the union of overlays. K-NN
   lookup wired as the fallback path in GenericBridge for unknown styles.
3. **Pattern L narrative.** AriGraph role-position index `(page, sentence,
   word, role-position)`. MUL marker semantics. Books-as-G adapter (one G per
   work).
4. **Pattern M blend.** Per-G `(wave_weight, particle_weight)` field on
   `ContextBundle`. The blend operator in the dispatcher.
5. **Pattern O integration.** Unified `PhenomenologicalView` reading SPO +
   Qualia17D + CausalEdge64 + Resonance + Epiphany + Meta-awareness. Per-G
   qualia-codebook lookup wired.

**Acceptance.**
- A new domain G (e.g., "anatomy" for the north-star demo) can specify only
  the deltas from DOLCE and inherit the rest.
- INT4-32D bootstrap produces sensible style retrieval for a held-out test set.
- A narrative passage parses to SPO-G with correct MUL marking on ambiguous
  pronouns; counterfactual synthesis runs without panicking.
- Wave-mode and particle-mode produce comparable but not identical retrievals
  on the same query; the blend is monotone in its weight.

**Exit criterion.** Anatomy-G fully populated; north-star demo Tier-3 prereqs
green.

### Tier 4 — Self-Extension (Pattern K)

**Goal.** Close the circular-compilation loop.

**Deliverables.**
1. **JIT path.** New patterns observed via cycles get compiled by
   `cam_pq::jitson_kernel` into runnable form, executed under ractor
   supervision.
2. **Crystallize-back.** Successful JIT'd patterns serialize to a YAML
   manifest entry (the same shape Pattern E expects).
3. **Next-build absorption.** The build-script picks up new manifest entries
   on the next `cargo build`; the JIT path retires the pattern in favour of
   the statically-compiled version.
4. **Provenance.** Every entry carries `(observed_at, jit_first_run,
   crystallized_at, build_absorbed_at)` for audit.

**Acceptance.**
- A synthetic pattern injected at runtime survives a `cargo build` cycle and
  becomes static.
- Provenance trail is queryable.
- No infinite-loop hazard: a malformed JIT'd pattern is quarantined, not
  crystallized.

**Exit criterion.** Three independent patterns make a full
runtime → crystallize → build → static round-trip.

---

## 5. Proof of Vision — `anatomy-realtime-v1`

The north-star demo is `anatomy-realtime-v1` (sub-plan owned by W12 at
`.claude/plans/anatomy-realtime-v1.md`). Summary, because it exercises every
pillar of this synthesis:

- **Hydrate FMA** (75 K-class Foundational Model of Anatomy, OWL source) via
  the OWL hydrator entry (Pattern D).
- **Ingest a medical scan** (DICOM) via a separate hydrator (also Pattern D
  — proving the hydrator entry pattern generalizes beyond TTL).
- **Render in Q2 cockpit** with realtime FMA-class overlay on the splatted
  scan geometry.
- **Exercises:** SplatShaderBlas (Pillar series), Pillar-6 EWA-Sandwich,
  Pillar-7 α-saturation, OGIT-G (Pattern A), GenericBridge (Pattern C) for a
  brand-new domain (anatomy was never built into the system), and RBAC via
  `medcare-rs` (Pattern C dispatch).

If `anatomy-realtime-v1` works, every claim in this document is validated by
running code. It is the integration test for the synthesis.

---

## 6. Cross-References

- **Ledger:** `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` (W6's reframe
  rows for THINK-1, HEEL-1, ADJ-THINK-1, CRYSTAL-1, CAM-DIST-1).
- **Epiphanies:** `.claude/board/EPIPHANIES.md` (W4's append covering
  Pattern H = shipped, Pattern N = shipped at 5 sites, Pattern O = shipped
  with music calibration).
- **Tech debt:** `.claude/board/TECH_DEBT.md` (W5's append covering wiring
  gaps that remain after Tier-0 recognition: the `tau()` write API,
  CycleAccumulator, blend operator, MUL marker semantics).
- **Sub-plans (this sprint):**
  - `.claude/plans/ogit-g-context-bundle-v1.md` (W10) — Tier 1 (1) + (2).
  - `.claude/plans/compile-time-consumer-binding-v1.md` (W11) — Tier 1 (4).
  - `.claude/plans/anatomy-realtime-v1.md` (W12) — proof of vision.
- **Adjacent prior plans:**
  - `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` —
    earlier soa-merge framing; this doc supersedes its dispatcher layer.
  - `.claude/plans/callcenter-membrane-v1.md` — earlier per-consumer
    membrane plan; Pattern C (GenericBridge) supersedes the newtype-per-consumer
    direction.
  - `.claude/plans/jc-pillars-runtime-wiring-v1.md` — pillar substrate;
    anatomy-realtime-v1 exercises pillars 6 and 7.
  - `.claude/plans/lance-graph-rdf-fma-snomed-v1.md` — FMA / SNOMED RDF
    ingest; anatomy-realtime-v1 uses its hydrator path.

---

## 7. Honest Self-Assessment

What this document **does**:
- Names 15 patterns crisply and maps each to shipped code.
- Reframes five entropy-ledger rows from "drift" to "intentional layering."
- Provides a 5-tier plan (0 → 4) with acceptance criteria per tier.
- Hands off three sub-plans to dedicated workers (W10/W11/W12).

What this document **does not** do:
- It does not specify the `ContextBundle` field types at byte-level — that's
  W10's sub-plan.
- It does not specify the manifest YAML schema in detail — that's W11.
- It does not give a CI-runnable test plan — that's per-tier and per-sub-plan.
- It does not yet name the qualia / thinking-style overlay tables for
  Anatomy-G specifically — Tier 3 deliverable, scoped in W12.

What remains **honestly open** (not just "future work"):
- The `tau()` write API for the adjacency planes (ADJ-THINK-1).
- The CycleAccumulator's flush policy (Pattern I).
- The wave/particle blend operator's exact form (Pattern M).
- The MUL marker resolution policy (Pattern L).
- The quarantine policy for malformed JIT'd patterns (Pattern K).

What is **not yet started** but framed:
- Pattern D's hydrator entry table (TTL exists, FMA exists, table is new).
- Pattern E's build-script (workspace exists, manifests are new).
- Pattern F's ractor port (gRPC trait shape exists, actor port is new).
- Pattern K's full loop closure (all primitives exist, the loop is new).

What is **already shipped and just needs naming + binding**:
- Patterns H, N, O substrate (extensively, across 5+ sites).
- Pattern G's 12-entry STYLES root codebook.
- Pattern F's message shape (via the gRPC trait).
- Pattern A's storage shape (oxigraph-style SPO + named graph).

This document is the contract for the next four tiers of work. If a reader
reaches Tier 4 without disagreement on the patterns named here, the synthesis
was correct. If a pattern is missing or mis-framed, append a v1-ERRATUM.md
rather than retro-editing.

---

*End of unified-ogit-architecture-v1.md.*
