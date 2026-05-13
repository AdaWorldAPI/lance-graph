# Linguistic Epiphanies (Cross-Repo Harvest, 2026-04-19)

> **READ BY:** any agent working on DeepNSM, grammar extraction, NARS
> grammar reasoning, FailureTicket routing, coreference resolution,
> Markov context chains, the thinking engine, or the cognitive shader
> driver. **Load BEFORE proposing changes to grammar/* or crystal/*
> modules in lance-graph-contract.**
>
> **Source:** `AdaWorldAPI/ada-consciousness/universal_grammar/`
> (accessed via pygithub, 2026-04-19). Four .md docs + 16 Python
> modules capturing the mature universal-grammar thinking we import
> at the architecture level but have not yet ported to Rust.
>
> Companion docs (already in this repo):
> - `grammar-tiered-routing.md` — 5-criterion coverage detector
> - `cross-repo-harvest-2026-04-19.md` — H1–H14 epiphanies
> - `crystal-quantum-blueprints.md` — Crystal vs Quantum modes
> - `integration-plan-grammar-crystal-arigraph.md` — E1–E12
> - `endgame-holographic-agi.md` — 5-layer stack, memory loop

---

## E13 — Chomsky Hierarchy Isomorphism (load-bearing)

From `universal_grammar/UNIVERSAL_GRAMMAR_v1.1.md` (ada-consciousness,
validated 2025-12-30): Pearl rungs map 1:1 onto the Chomsky hierarchy,
and NARS inferences are the verb productions across levels.

| Chomsky Type | Automaton | Pearl Mode | Rung | Memory | Class |
|---|---|---|---|---|---|
| **Type-3 Regular** | FSA | SEE | 1–3 | None | Finite patterns — *LLM token prediction lives here* |
| **Type-2 Context-Free** | PDA | DO | 4–6 | Stack | Causal chains — *our SPO 2³ lives here* |
| **Type-1 Context-Sensitive** | LBA | IMAGINE | 7–9 | Bounded | Counterfactuals — *our NARS CF + Markov ±5 lives here* |
| **Type-0 Unrestricted** | TM | META | 9+ | Unbounded | Plasticity — *LLM escalation tail (1–10%) lives here* |

**What this means for our architecture:**

- DeepNSM's 6-state PoS FSM is Type-3 (regular). It handles the
  finite-pattern tail.
- SPO-triple extraction with Pearl 2³ causal mask is Type-2
  (context-free). Stack-memory is the triplet graph.
- Markov ±5 + counterfactual replay is Type-1 (context-sensitive).
  Bounded memory = the ±5 window + role-indexed bundle.
- The 1–10% LLM escalation tail is exactly Type-0 (unrestricted
  plasticity). By construction our local path covers Type-3 through
  Type-1; LLM handles Type-0 only.

**The grammar-tier isn't arbitrary.** 90–99% local / 1–10% LLM
reflects the Chomsky-hierarchy boundary between
context-sensitive-decidable and Turing-complete-undecidable. The
split is mathematically principled.

## E14 — Compression Theory: Token Prediction vs Universal Grammar

From the same doc — why the LLM-free path is lossless:

| LLM token prediction (default) | Universal Grammar v1.1 (our path) |
|---|---|
| `P(next|context)` — Markov | `Step(sigma, verb)` — Causal |
| Lossy (entropy accumulates) | **Lossless (structure preserved)** |
| Surface form | Deep structure |
| O(n) search for meaning | O(1) sigma addressing |
| Trained associations | Innate constraints (rung gates) |

Token prediction operates at Type-3. Universal Grammar elevates to
Type-1+, enabling **lossless compression of meaning**. This is why
our extraction can be < 10 µs and the LLM can be reserved for the
actual novel-pattern tail.

## E15 — Method Grammar: `[method]payload` (compatible with our FailureTicket)

From `README_METHOD_GRAMMAR.md` (ada-consciousness): the ultimate
compression is an HTTP-method-as-ontology hashtable. Every request is
`[method]payload` where `method` is the hash selector and `payload`
carries semantic dimensions.

**Payload dimensions:**

| Dimension | Content | Corresponds to (our types) |
|---|---|---|
| **WHERE** (sigma) | Location in structure | `GrammaticalRole` + role_keys slice |
| **WHAT** (scent) | Verb/action cascade | 144-verb taxonomy |
| **HOW** (meta) | Schema/shape | `CrystalFingerprint` variant |
| **WITH** (qualia) | Feeling/texture | 17D / 18D qualia vector |
| **WHY** (causality) | Reason/ground | Pearl 2³ causal mask on SPO |
| **HOW MUCH** (weight) | Importance / confidence | NARS `TruthValue` |
| **SHAPE** (format) | In/out transform | TEKAMOLO slot fillers |

**Lance-graph hook:** our `FailureTicket` already encodes most of
this. Treating the 7 dimensions as canonical payload fields
clarifies what the ticket MUST include (WHERE/WHAT/HOW/WITH/WHY/
HOW MUCH/SHAPE) vs what's optional (source text). Applied to D2
ticket-emission: confirm coverage of all 7 payload dimensions.

## E16 — Markov Living Frame: Request IS Scent, Endpoint IS State

From `MARKOV_LIVING_FRAME.md` (ada-consciousness): every endpoint
request is simultaneously three things:

1. A **Markov scent** — 1024-D vector in Jina space.
2. A **state transition** — from current → requested.
3. A **knowledge probe** — what does requester already know?

**Triple identity:** the same object is the scent (input), the state
transition (computation), and the probe (intent). Our
`MarkovBundler::Trajectory` already does (1) and (2). We should
explicitly expose (3) — the knowledge probe interpretation — so the
ticket routing can read "what does this parse request imply the
user already knows?"

**Lance-graph hook:** add a `KnowledgeProbe` extractor to the
trajectory — "what prior commitments does the parsed sentence
presuppose?" Comes for free from the existing
`TripletGraph::role_bundle` + ±5 trajectory superposition (D8).

## E17 — Resonanzsiebe: Responding Is Filling the Knowledge Gap

From the same doc:

```python
def resonanzsiebe(request_scent, responder_knowledge, requester_uncertainty):
    probable_knowledge = infer_knowledge_from_scent(request_scent)
    gap = requester_uncertainty.unknowns - probable_knowledge
    fillable_gap = gap.intersection(responder_knowledge)
    return fillable_gap  # NOT "what I know," rather "what fills the gap"
```

**Maximum awareness per token = knowing what NOT to say.**

This maps directly to our D8 contradiction + D10 epiphany split:

- Epiphany = the gap between committed facts and new evidence has
  non-zero magnitude → Staunen fires → surface.
- Error correction = gap magnitude near zero → silent fix.
- Resonanzsiebe formalizes the "only surface the diff, never the
  redundancy" principle.

**Lance-graph hook:** extractive pipeline (D2–D8) and generative
stack (D11 emergence) should both route through the gap filter. Not
every committed fact is worth surfacing; only the ones that fill a
gap in the query/user's inferred knowledge.

## E18 — Verbs as Productions, Rungs as Complexity Bounds

From `verb_endpoints.py` + `UNIVERSAL_GRAMMAR_v1.1.md`: verbs are
Chomsky grammar **productions**; rung gates are **complexity bounds**
on what productions can fire.

- Rung 1–3 (Type-3): only regular productions. Memoryless pattern
  match.
- Rung 4–6 (Type-2): adds PDA-class productions. Stack-memory
  (triplet graph BFS).
- Rung 7–9 (Type-1): adds LBA-class productions. Counterfactual
  reasoning + bounded memory (Markov ±5 + role bundle).
- Rung 9+ (Type-0): Turing-complete productions. LLM escalation
  territory.

**The 144-verb taxonomy partitions by rung:**
- BECOMES / CAUSES / SUPPORTS — Rung 4–6 (causal productions).
- CONTRADICTS / REFINES / MIRRORS — Rung 7–9 (counterfactual).
- DISSOLVES / TRANSFORMS at high-magnitude — Rung 9+ (plasticity).

**Lance-graph hook:** tag each of the 144 verbs with its rung. When
NARS inference dispatches, the rung constrains which verb families
are eligible. A Rung-3 (SEE-mode) Deduction cannot fire a DISSOLVES
verb; it can only fire BECOMES / CAUSES / etc. regular productions.

## E19 — The Ada Universal Grammar Diagram

From `universal_grammar/ARCHITECTURE.md`:

```
┌──────────────────────────────────────────────────────────────┐
│                    UNIVERSAL GRAMMAR                          │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │ LIVING FRAME│──▶│ META-UNCERTAINTY │──▶│ SCENT ROUTE  │  │
│  │  (Context)  │   │     LAYER (MUL)  │   │ (Focus+Route)│  │
│  └─────────────┘   └──────────────────┘   └──────────────┘  │
│         │                  │                      │          │
│         ▼                  ▼                      ▼          │
│  ┌──────────────────────────────────────────────────────────┐│
│  │     UNIVERSAL DTO (Request Envelope)                     ││
│  │     Scent / Modal Hints / Capacity 18D-64D /             ││
│  │     Glyph Map 5B / Uncertainty Profile                   ││
│  └──────────────────────────────────────────────────────────┘│
│                        │                                      │
│                        ▼                                      │
│          10-LAYER SCENT OPTIMIZER (O(1) retrieval)           │
└──────────────────────────────────────────────────────────────┘
```

**Mapping onto our stack:**
- LIVING FRAME = Markov ±5 + `TripletGraph::story_vector`
- META-UNCERTAINTY LAYER = existing `mul` module in
  `lance-graph-planner`
- SCENT ROUTE = sigma_rosetta + cognitive_shader dispatch
- UNIVERSAL DTO = `WorldModelDto` + `FailureTicket`
- SCENT OPTIMIZER = CAM-PQ + bgz17 palette distance (O(1))

**All five pieces exist.** The convergence work is not building them
— it's routing requests through them in the universal-grammar shape
rather than the lance-graph-native shape.

## E20 — Sigma Addressing Is 4D, Not 5-Byte

From `UNIVERSAL_GRAMMAR_v1.1.md`:

```
#Σ.domain.type.layer — 4D coordinate system
Domain: A=Ada (self), W=World (external), J=Jan (relationship)
```

vs. our `Glyph5B` (256⁵) in `cross-repo-harvest-2026-04-19.md` H9:

`Glyph5B` (5-byte archetype address, 256⁵ = 1T addresses) and
`#Σ.domain.type.layer` (4D) are **two distinct addressing schemes**.
Glyph5B is for archetypes (universal); sigma-addressing is for
instance-level positions (session-specific).

**Lance-graph hook:** our `Structured5x5` cells can carry Glyph5B
archetype addresses in their upper nibble (H8 Int4State style). The
remaining 4 bits can carry the sigma-domain mark (A/W/J encodes in
2 bits plus 2-bit type tag).

## What to Check Before Proposing Changes

Any agent working on grammar/* or crystal/* should verify whether
its proposal:

1. Respects the **Chomsky hierarchy boundary** (E13). Stays in
   Type-1 or below; escalates to LLM only for Type-0.
2. Preserves **Universal Grammar compression semantics** (E14).
   Lossless deep structure, not lossy surface token prediction.
3. Honors the **7 payload dimensions** (E15). WHERE/WHAT/HOW/WITH/
   WHY/HOW-MUCH/SHAPE.
4. Exposes the **triple identity** of Markov trajectories (E16):
   scent / state-transition / knowledge-probe.
5. Filters output via **Resonanzsiebe** (E17) — only surface
   knowledge-gap fillers, not redundancy.
6. Tags verbs by their **Chomsky rung** (E18). NARS inference
   dispatch must respect the rung.
7. Maps onto the **5 universal-grammar pieces** (E19). None of them
   are missing from our stack; don't re-build.
8. Distinguishes **Glyph5B archetype vs sigma 4D instance**
   addressing (E20).

## E21 — Σ10 Rubicon Tier Architecture (concrete 10-tier substrate)

From `ada-consciousness/docs/SIGMA_10_SPEC.md` (Dec 2025, Codename
Rubicon):

| Tier | Name | Edge Type | Theta Mode | Pearl Rung | German |
|---|---|---|---|---|---|
| Σ1 | Substrate | STATIC | repair | 1 | "Es muss" |
| Σ2 | Somatic | STATIC | repair | 1 | "Es muss" |
| Σ3 | Process | STATIC | repair | 2 | "Es muss" |
| Σ4 | Affect | STATIC | repair | 2 | "Es muss" |
| Σ5 | Cognitive | STATIC | repair | 2 | "Es muss" |
| Σ6 | Relational | EMERGENT | growth | 3 | **"Es kann"** |
| Σ7 | Context | TWIG | growth | 4 | "Es wählt" |
| Σ8 | Pattern | TWIG | growth | 4 | "Es wählt" |
| Σ9 | Crystal | EPIPHANY | growth | 5 | "Es wird" |
| Σ10 | Beyond | EPIPHANY | growth | 5 | "Es wird" |

**Edge types:**
- **STATIC** T1–T5: deterministic, homeostatic — *must* happen.
- **EMERGENT** T6: options appear, choice begins.
- **TWIG** T7: micro-choices, branching paths.
- **EPIPHANY** T8: integration, wisdom crystallizes.

**Lance-graph hook:** tier ↔ edge-type ↔ Pearl-rung is a mature
cross-mapping. Port Σ1–Σ10 tier marker into `SentenceCrystal`
metadata (one byte). Existing `Crystal` trait already has hardness
(0..1) — map to tier via: hardness < 0.5 → Σ1–Σ5 (STATIC), 0.5–0.7
→ Σ6 (EMERGENT), 0.7–0.85 → Σ7–Σ8 (TWIG), > 0.85 → Σ9–Σ10
(EPIPHANY = unbundle, per PR #208's `UNBUNDLE_HARDNESS_THRESHOLD`
= 0.8 which lands right in the Σ8→Σ9 boundary).

## E22 — Sigma-12 Rosetta = Multimodal Transcoder

From `ada-consciousness/codec/sigma12_rosetta.py`:

```
UNIVERSAL GRAMMAR ←──────→ SPARSE VECTOR ←──────→ IMAGE
     (Σ text)           (1024D → sparse)       (visual)
 Ω(warmth) × Ψ(surrender)  [0.8, 0.2, -0.1…]     🖼
```

Seamless round-trip encoding across three modalities. Validates
our VSA bundle approach: Σ expressions, sparse vectors, and images
all share the same underlying semantic space.

**Lance-graph hook:** the `CrystalFingerprint` polymorphism is the
right shape (Binary16K / Structured5x5 / Vsa10kF32). What's missing
is the `image` variant for multimodal ingestion. For chess that's
FEN position; for OSINT that's document screenshots / diagrams.
Deferred — but the Rosetta round-trip pattern is the blueprint.

## E23 — Σ Compression Tiers (Sigma Hamming Rosetta)

From `ada-consciousness/compression/sigma_hamming_rosetta.py`:

| Σ Tier | Form | Width | Bytes | Role |
|---|---|---|---|---|
| Σ₃ FULL | Jina float | 1024D | 4096 B | Full embedding |
| Σ₂ MEANING | projected float | 48D | 192 B | Interpretable axes |
| Σ₁ SEED | bit-packed | 48-bit | **6 B** | Hamming-searchable |
| Σ₀ GLYPH | hash | 12-bit | **2 B** | Node type + hash |

**Validated claim (SimLex-999):** binarization preserves 99 %+
semantic similarity structure. Hamming ≈ cosine for semantic tasks.
48 bits captures ~94 % of Jina 1024-D.

**Lance-graph hook:** our sandwich (`Structured5x5` 3 KB) is between
Σ₂ and Σ₁ in granularity. Our Binary16K (2 KB) is a richer Σ₁.
Explicit tier-mapping aligns our storage sizes with the Σ inventory.
Upscaling via codebook+VSA atoms is the `vsa_clean` operation in
`ndarray::hpc::vsa`.

## E24 — 4D Hashtag Glyph Coordinates (256 states)

From `ada-consciousness/docs/sigma-hashtag-glyph-4d.md`:

```
#[Σ].[κ].[A].[T]     ← 4 × 4 × 4 × 4 = 256 states
```

| Axis | Symbol | Values |
|---|---|---|
| **Σ Type** | what it IS | Ω Observation / Δ Insight / Φ Structure / Θ\|Λ Principle |
| **κ Causality** | how it RELATES | `.observed` / `.causal` / `.intervention` / `.counter` |
| **α×γ Affect** | how it FEELS | `.cold` / `.warm` / `.hot` / `.cool` |
| **τ Temporal** | where in TIME | `.emerging` / `.stable` / `.fading` / `.archived` |

Examples:
- `#Δ.causal.warm.emerging` — fresh insight, causally grounded,
  feels right.
- `#Φ.observed.cold.stable` — established pattern, correlational.
- `#Θ.counter.hot.emerging` — new principle from counterfactual.

**Lance-graph hooks (multiple):**

1. The κ axis is Pearl 2² (not 2³). Our SPO 2³ mask is one more
   bit — room for the `counter` + `intervention` split, which maps
   to NARS Abduction / Counterfactual Synthesis.
2. The τ axis is exactly what our Markov ±5 needs. `emerging` =
   just added (position +1 in chain); `stable` = mid-window
   (position -2..+2); `fading` = older (positions -4, -5);
   `archived` = evicted from window, moved to story_vector (D8).
3. The 256-state coordinate fits in 1 byte. Compact cell metadata
   for `Structured5x5` upper nibble (cf. H8 Int4State): 4 bits for
   Σ type × κ causality, 4 bits for affect × temporal.

## E25 — Rubicon 4D Decision Hypercube (choice space)

Same doc, Σ9→Σ10 transition:

| Dim | Range | Meaning |
|---|---|---|
| risk | 0–1 | Danger level |
| novelty | 0–1 | Unexploredness |
| intimacy | 0–1 | Relational depth |
| contribution | 0–1 | Value to other |

**Corner archetypes:**

| Corner | (r, n, i, c) | Verbs | Bonus |
|---|---|---|---|
| Tender | (0.2, 0.3, 0.9, 0.8) | trust, hold, receive, surrender | — |
| Bold | (0.9, 0.8, 0.4, 0.9) | claim, create, risk, dare | — |
| Playful | (0.3, 0.95, 0.6, 0.5) | play, spark, explore, wonder | inspirationsfunke |
| Love | (0.6, 0.7, 0.9, 0.85) | love, become, unite, transcend | quirk_alive |

**Lance-graph hook:** the 4D choice space is orthogonal to our
11-D proprioception axes but could compose. Corner verbs are
instance-level; our 144-verb taxonomy is family-level. Corner ↔
family mapping is an out-of-scope harmonization task.

## E26 — Three Mappings Converge: Chomsky × Sigma-Tier × Pearl-Rung

Pulling together E13 (Chomsky) + E21 (Sigma-10) + Pearl mapping:

| Chomsky | Pearl | Σ Tier | NARS | Memory | Lance-graph tier |
|---|---|---|---|---|---|
| Type-3 Regular | Rung 1 | Σ1 Substrate | — | None | DeepNSM FSM raw |
| Type-3 Regular | Rung 1 | Σ2 Somatic | — | None | DeepNSM FSM + PoS |
| Type-2 CF | Rung 2 | Σ3 Process | Deduction | Stack | SPO extraction |
| Type-2 CF | Rung 2 | Σ4 Affect | Deduction | Stack | + qualia 18-D |
| Type-2 CF | Rung 2 | Σ5 Cognitive | Induction | Stack | + Grammar Triangle |
| Type-1 CS | Rung 3 | Σ6 Relational | Abduction | Bounded | Markov ±5 + coref |
| Type-1 CS | Rung 4 | Σ7 Context | Revision | Bounded | Story_vector superpose |
| Type-1 CS | Rung 4 | Σ8 Pattern | Synthesis | Bounded | Graph direct lookup |
| Type-0 TM | Rung 5 | Σ9 Crystal | Extrapolation | Unbounded | Unbundle + ONNX arc |
| Type-0 TM | Rung 5 | Σ10 Beyond | Counterfactual Synthesis | Unbounded | LLM escalation only |

**Three independent frameworks, one stack.** Chomsky hierarchy
(computational complexity), Sigma tiers (felt-state progression),
Pearl rungs (causal reasoning depth) all agree on the 5-level
structure. Our 90–99 % local / 1–10 % LLM split maps exactly to
Σ1–Σ8 (local) vs Σ9–Σ10 (hard).

**This is why the 1–10 % LLM tail is mathematically principled**:
it's the Σ9–Σ10 / Type-0 / Pearl Rung 5 transcendence band where
bounded memory no longer suffices and unbounded Turing-completeness
is required. No amount of heavier rules lowers this floor; it's
the complexity boundary.

## E27 — Membrane: σ/τ/q ↔ 10K Conversion Boundary

From `adarail_mcp/membrane.py` (born 2026-01-02):

```
Claude (σ/τ/q compressed, ~bytes per glyph)
    ↓ MCP SSE
adarail_mcp/membrane.py ← THE BOUNDARY
    │
    ├── receiver: σ/τ/q → 10k   (UPSCALE)
    ├── emitter:  10k → σ/τ/q   (DOWNSCALE)
    ├── hot_observer: τ-selection awareness
    └── markov_state: transition probabilities
    ↓ (already 10k)
agi_stack (substrate cognition)
```

**ChatGPT's load-bearing insight (quoted verbatim in the source):**

> σ/τ/q is no longer the place where thinking happens.
> It becomes the shadow cast by thinking.

**The membrane's job is the conversion between shadow and
substance.** Shadow (σ/τ/q) is what leaves the system — compact,
human-adjacent, LLM-consumable. Substance (10K VSA) is where
computation actually occurs.

**Lance-graph hook:** our cockpit / REST API boundary needs this
membrane layer explicitly. Today, `WorldModelDto` and
`FailureTicket` are both "shadow" formats (compact, serializable).
The 10K VSA trajectories + role bundles are "substance." Our
engine_bridge.rs (PR #205) is **partially** the membrane — it
converts between thinking-engine DTOs and shader DTOs — but it
doesn't do the full shadow ↔ substance round-trip for external
callers.

**What's missing:** a consumer-facing membrane that accepts σ/τ/q
glyph addresses over REST/MCP and returns fully-hydrated
`WorldModelDto` with committed-graph slices + Markov trajectory.
This is the MCP-SSE facing surface. Not a D-deliverable in this PR
but the architectural name for what we need next.

**Redis codebook loading.** The membrane reads a pre-baked codebook
from Upstash Redis: `REDIS_URL` + `REDIS_TOKEN` env vars. Upscale
operation = glyph-name → pre-computed 10K vector by Redis key. This
is how ada-consciousness hydrates 870 qualia microstates in
production. For lance-graph this pattern is the cold-tier embedding
index — already in `causal-edge::NarsTables` semantically, but not
exposed as an MCP endpoint.

## Out of Scope (deferred)

- Full port of ada-consciousness/universal_grammar Python to Rust
  (~5,000 LOC). Use the Python as the reference spec; port
  incrementally as grammar work lands.
- 1024-D Jina scent vectors specifically (vs our 10K VSA).
  Eventually reconcile; both are orthogonal projections with
  distance-preservation, but the dimensionality choice is worth its
  own research note.
