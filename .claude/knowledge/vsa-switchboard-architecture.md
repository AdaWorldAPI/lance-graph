# VSA Switchboard Architecture — The Corrected Framing

> **READ BY:** Every session touching VSA, role keys, persona,
> callcenter, archetype, thinking styles, AriGraph episodic,
> or any domain that does role-indexed superposition.
>
> **Supersedes:** The 2026-04-21 Frankenstein framing where
> `Vsa10k = [u64; 157]` bitpacked + XOR was treated as the
> "grammar VSA format." That was a domain-specific conflation
> of the general switchboard carrier with a grammar role catalogue.
>
> **Created:** 2026-04-21 cleanup session
> **Status:** FINDING (corrects multiple EPIPHANIES entries)

---

## The Three Layers

The architecture separates into THREE clean layers. Session 2026-04-21
pre-cleanup conflated layers 1 and 2; this doc fixes the separation.

### Layer 1 — Switchboard Carrier (general, in `crystal/`)

One set of types + one algebra, used by ALL domains:

| Type | Layout | Used when |
|---|---|---|
| `Vsa16kF32` | `Box<[f32; 16_384]>` = 64 KB | Hot-path compute (precision matters) |
| `Vsa16kBF16` | `Box<[bf16; 16_384]>` = 32 KB | AMX-accelerated bundling (Intel Sapphire Rapids+) |
| `Vsa16kF16` | `Box<[f16; 16_384]>` = 32 KB | Apple M-series / ARMv8.2+ compact compute |
| `Vsa16kI8` | `Box<[i8; 16_384]>` = 16 KB | Persistence / quantized storage |
| `Binary16K` | `Box<[u64; 256]>` = 2 KB | Hamming comparison / compact fingerprint |

**Algebra (element-wise over ℝ or GF(2), per Shaw 2501.05368 Kan extension theorem):**

- `vsa_bind(a, b) = a[i] * b[i]` (element-wise multiply, self-inverse for ±1 bipolar)
- `vsa_bundle(vecs) = Σ v[i]` (element-wise add — LOSSLESS accumulation within f32 dynamic range)
- `vsa_superpose(vecs, weights)` — weighted add
- `vsa_cosine(a, b) = dot(a, b) / (||a|| · ||b||)` — similarity

**Binary variant algebra (for Binary16K only):**

- XOR for comparison (Hamming popcount)
- Sign-binarize from F32 variant for compact similarity
- **Never** used for lossless bundling — XOR saturates at ~5-7 items

**Transitions (documented passthrough):**
- Vsa16kF32 ↔ Binary16K: 1:1 bit-to-dim lossy passthrough (sign-binarize)
- Vsa16kF32 ↔ Vsa16kI8: quantization, lossless up to i8 precision
- Vsa16kF32 ↔ Vsa16kBF16: precision reduction (f32 mantissa → bf16 7-bit)

### Layer 2 — Domain Role Catalogues (per-domain)

Each domain ships its OWN module with role keys + slice layout.
All role keys are `Vsa16kF32` values (bipolar ±1 in their slice,
zero elsewhere). Role keys are IDENTITIES; they point into the
switchboard carrier's slice address space.

| Domain module | Roles |
|---|---|
| `grammar/role_keys.rs` | SUBJECT, PREDICATE, OBJECT, MODIFIER, CONTEXT, TEMPORAL, KAUSAL, MODAL, LOKAL, INSTRUMENT, BENEFICIARY, GOAL, SOURCE, 15 Finnish cases, 12 tenses, 7 NARS inferences |
| `persona/role_keys.rs` (future) | MODAL, AFFECTIVE, TONE, REGISTER, STANCE, GENDER, AGE_GROUP, FORMALITY |
| `callcenter/role_keys.rs` (future) | INTENT, SENTIMENT, AGENT_ACTION, URGENCY, ESCALATION_TRIGGER |
| `archetype/role_keys.rs` (future) | ARCHETYPE_FAMILY_12, VOICE_CHANNEL_16, QUALIA_17 |

Slice boundaries per domain — each owns disjoint `[start:end)`
regions of the 16,384-dim space. Multiple domains can share the
SAME carrier if they use disjoint slices; or domains can have
their OWN carrier instances if they never need to superpose
with each other.

### Layer 3 — Content Stores (per-domain, NOT VSA)

Actual content lives in structured stores, O(1) retrieval by
identity:

| Store | Content | Key |
|---|---|---|
| `thinking_styles/*.yaml` | Style config (NARS priority, morphology tables, TEKAMOLO priority) | style name enum |
| `persona/*.yaml` (future) | Persona slots, prompts, behavior rules | persona name |
| `callcenter/intents.yaml` (future) | Intent definition, resolution flow, escalation rules | intent name |
| `TripletGraph` | SPO facts with NARS truth + Pearl 2³ mask | fingerprint + feature index |
| `EpisodicMemory` | Episode snapshots with ±5 context | episode_index + fingerprint |

**The bridge:** Domain role catalogue provides the IDENTITY fingerprint;
content store provides the actual definition; retrieval is either:

- **Register lookup:** name → content (HashMap / YAML / enum). O(1),
  exact match. Use when the item has a known name.
- **VSA resonance:** fingerprint → nearest-known-identity via cosine.
  Use when the item is inferred from context (signal profile, tone,
  narrative arc) rather than named explicitly.


---

## The Four Tests Before Reaching for VSA

Apply in order. First failure short-circuits — use something else.

### Test 0 — Register laziness check (new, added 2026-04-21)

> Does this thing have a register — a name, ID, enum variant,
> HashMap key — that uniquely identifies it?

If YES: use the register. `HashMap<&str, PersonaDef>`, `enum
ThinkingStyle`, `TripletGraph::nodes_matching(id)` — these are the
right tools. VSA bundle + cosine is NOT a substitute for a well-
defined lookup table.

Examples of register-laziness anti-patterns:

- "Find persona Alice" → HashMap, not VSA
- "Session is in analytical mode" → enum variant, not VSA resonance
- "Character is labeled Napoleon" → graph node by ID, not VSA search

VSA earns its complexity ONLY when the answer requires resonance
across multiple concurrent items or partial-match reasoning from
uncertain input.

### Test 1 — Bundle size N check

> What is N, the number of items in the bundle?

If N > √d / 4 (for 16K dim, that's > 32 items): stop. Superposition
SNR drops below usable threshold. Use a different tool (direct
graph lookup, indexed search, codebook nearest-neighbor).

Safe regime: N ≤ 32 at 16K dim. Typical VSA workloads stay well
below this (Markov ±5 has N=11, sentence SPO has N=3-8).

### Test 2 — Role orthogonality check

> Are the role keys mutually orthogonal in the carrier's slice space?

If slices overlap or role keys are correlated, unbind doesn't recover
content cleanly — superposition is broken regardless of N.

Disjoint slice addressing in `grammar/role_keys.rs` guarantees this
for grammar. New domain catalogues must maintain the property.

### Test 3 — Cleanup codebook check

> Is there a known codebook to match against after unbind?

Without cleanup, the unbind returns noisy approximation. Raw bundle
inspection is unreliable. Domain codebooks:

- Grammar: 24K word content fingerprints (COCA vocab)
- Persona: the persona registry's identity fingerprints
- Callcenter: the intent catalogue's identity fingerprints

No codebook → no VSA. Period.

---

## Identity vs Content: The Register Loss Problem

**Refined iron rule (I-VSA-IDENTITIES):**

VSA operates on IDENTITIES, not on CONTENT. When content is stored
in a format with a lossy register (CAM-PQ codebook indices,
bitpacked quantization, i8 scalar quantization, sign-binarized
fingerprints), VSA must not operate on that register directly.

**The register loss problem:** XOR-bundling 5 CAM-PQ codes makes
the bit patterns of the codebook indices XOR together. You cannot
recover WHICH centroids contributed. The register — the mapping
from bits back to codebook entries — is destroyed.

**The right pattern:**

```
┌─────────────────────────────────────────────────────────────┐
│ RESONANCE LAYER (VSA Vsa16kF32)                             │
│  - Domain identity fingerprint (FP32, 16K dims, bipolar)    │
│  - Bundled for multi-identity context (superposition OK)    │
│  - Retrieved by cosine similarity (resonance)               │
│  - Never carries the ACTUAL content                         │
└────────────────────┬────────────────────────────────────────┘
                     │ winning fingerprint IS the lookup key
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ CONTENT LAYER (YAML / TripletGraph / sled)                  │
│  - Actual definition (slots, rules, prompts, behavior)      │
│  - O(1) hash lookup by identity / name / ID                 │
│  - Editable, hierarchical, human-authored                   │
│  - Never bundled, never superposed                          │
└─────────────────────────────────────────────────────────────┘
```

**Sometimes Vsa16kF32 IS just laziness to define a register.**

If you're tempted to reach for VSA but the item has a natural name
or a small enum, that's Test 0 failing. Skip VSA, use the register.

---

## CAM vs CAM-PQ vs Vsa16kF32 — When to Use Which

**All three are compression / similarity formats. They serve
DIFFERENT operations.**

### CAM (Content-Addressable Memory, un-quantized)

- **What:** Direct content-keyed lookup. Key is the full fingerprint;
  value is the content. Like a HashMap keyed by 256-word fingerprint.
- **Size per entry:** 2 KB (Binary16K key) + content
- **When to use:** When you know the exact fingerprint and want O(1)
  content retrieval. Rigid-designator lookups (Napoleon's fingerprint
  → Napoleon's triple). Direct fact storage in AriGraph.
- **Algebra:** exact match (`key == query`) or Hamming similarity
  (`popcount(key ^ query)` for ordered retrieval)
- **NOT for:** Superposition. CAM entries are atomic.

### CAM-PQ (Content-Addressable Memory with Product Quantization)

- **What:** Compressed content-keyed lookup. Each fingerprint
  subdivided into M subvectors, each quantized to one of K codebook
  centroids. Storage: M indices (bitpacked).
- **Size per entry:** ~6-64 bytes for index codes
- **When to use:** Vector search at scale. Millions of fingerprints
  where the 2 KB Binary16K would be too heavy. Approximate nearest-
  neighbor via precomputed distance tables.
- **Algebra:** `table[subspace_m][q_idx][k_idx]` lookup per subvector,
  sum per entry, k-NN rank.
- **NOT for:** VSA bundling. Register loss when XOR-superposed. Use
  ONLY for nearest-neighbor lookup at compressed scale.

### Vsa16kF32 (Real-valued VSA for Lossless Role Bundling)

- **What:** 64 KB f32 vector. Multiply-add algebra. Bipolar ±1 content
  and role keys. Superposition preserves all N contributions (lossless
  within f32 dynamic range, N ≤ √d/4 ≈ 32).
- **Size per entry:** 64 KB per bundle
- **When to use:** Role-indexed bundling with small N (Markov ±5,
  sentence SPO, persona slots, callcenter turn state). When you need
  partial-match reasoning from context, not exact lookup.
- **Algebra:** `vsa_bind` (element-wise multiply), `vsa_bundle`
  (element-wise add), `vsa_cosine` (similarity).
- **NOT for:** Exact lookups (use CAM), vector search at scale (use
  CAM-PQ), SQL queries, graph traversal, arbitrary similarity over
  non-role-bound vectors.

### Decision Matrix by Operation

| Operation | Use |
|---|---|
| Exact match by fingerprint | CAM |
| Rigid-designator lookup (Napoleon, GPT-4, known entity) | CAM or register |
| Vector search at scale (1M+ documents) | CAM-PQ |
| Approximate nearest-neighbor in compressed space | CAM-PQ |
| Role-indexed bundling (grammar/persona/callcenter) | Vsa16kF32 |
| Multi-item superposition for context | Vsa16kF32 |
| Partial-match reasoning (which persona fits this vibe?) | Vsa16kF32 resonance against codebook |
| Cosine similarity after VSA unbind | Vsa16kF32 |
| "Find all subjects ever" | SPO graph traversal (NOT VSA) |
| "Is this fact consistent with committed beliefs?" | Graph + local Markov coherence (NOT VSA alone) |

### Combining: The decompress-first rule

If content is stored in CAM-PQ for scale, and you need VSA operations
on the content:

1. Nearest-neighbor search in CAM-PQ → top-K candidates (cheap)
2. Decompress top-K to Vsa16kF32 (moderate)
3. VSA operations on decompressed vectors (precise)
4. Re-quantize to CAM-PQ only if persisting new content

Never run VSA on CAM-PQ codes directly. Never bundle CAM-PQ codes.


---

## The Archetype ↔ AriGraph ↔ Persona ↔ ThinkingStyle Unification

All four of these are **role catalogues** in the Layer-2 sense. They
share the same architectural pattern:

| Catalogue | Registers (Test 0) | Identity fingerprints (Vsa16kF32) | Content store |
|---|---|---|---|
| **Archetype** | 12 archetype families × 12 voice channels (existing: palette archetypes, VoiceArchetype in ndarray::hpc::audio) | 144 identity fingerprints, Vsa16kF32, bipolar in disjoint slices | Existing archetype tables in ndarray + lance-graph bgz17 palette |
| **AriGraph** | SPO triples keyed by (subject_fp, predicate_fp, object_fp) | Entity identity fingerprints (one per rigid designator) | `TripletGraph` nodes + edges with NARS truth + Pearl 2³ mask |
| **Persona** | Named personas from YAML registry (Alice, Bob, …) | Persona identity fingerprint per entry | `persona/*.yaml` (future) with slots, prompts, behavior rules |
| **ThinkingStyle** | 12-style enum (Analytical, Exploratory, …) + 36 variants | Style identity fingerprint per variant | `thinking_styles/*.yaml` (12 starter configs, D7 follow-up) |

### The pattern: Identity in VSA, content in YAML/graph

1. **Register defined at load time.** HashMap `&str → Vsa16kF32`
   (identity fingerprint) + HashMap `&str → ContentDef` (YAML
   content). Both keyed by name.

2. **Runtime dispatch via resonance OR explicit name.**
   - Explicit (Test 0 passes): `personas.get("Alice")` — O(1) content
   - Resonance (Test 0 fails): compute context fingerprint from signal
     profile; cosine-rank against identity codebook; winner's name →
     content lookup
   
3. **Committed facts bundle the identity, not the content.**
   AriGraph stores (Napoleon_fp, decrees_fp, it_fp) as the edge. The
   actual biographical content of "Napoleon" lives separately in
   graph properties or a YAML profile — NOT bundled into the Vsa
   trajectory.

### Cross-catalogue bundling is legal when slices are disjoint

A single `Vsa16kF32` bundle can carry role bindings from MULTIPLE
catalogues if they allocate disjoint slice ranges:

```
[0..2048)     Grammar SUBJECT
[2048..4096)  Grammar PREDICATE
...
[8192..9216)  Persona MODAL
[9216..10240) Persona AFFECTIVE
...
[12288..13312) Callcenter INTENT
[13312..14336) Callcenter SENTIMENT
...
```

**This is powerful.** A single trajectory can carry:
- Grammar roles (who did what)
- Persona roles (what voice / tone / stance)
- Callcenter roles (intent / urgency / action)

...all in one 64 KB vector, all unbindable per role catalogue,
all losslessly superposed (within N ≤ 32 per catalogue).

### Cross-catalogue bundling fails if slices overlap

If callcenter INTENT slice collides with grammar OBJECT slice,
unbinding OBJECT returns a mixture of grammar object content
and callcenter intent content. Noise dominates signal.

**Governance rule:** every new domain catalogue must declare its
`[start:end)` slice range in a SHARED `role_catalogue_registry.md`
knowledge doc that tracks allocations. No overlap ever.

---

## ONNX 16kbit Learning — Architectural Placement

The `3x16kbit Plane accumulator` (shipped in ndarray `plane.rs`,
alpha-aware Hamming, RL support per `CROSS_REPO_AUDIT_2026_04_01.md`)
is the **write-path** accumulator for AriGraph edge learning. It
is NOT a VSA carrier, NOT an edge storage format, NOT a role
fingerprint.

| Question | Answer |
|---|---|
| When used? | During encounter (saturating i8 accumulation) — the node/edge is being UPDATED based on new evidence |
| What does it store? | 48 KB per edge of i8 accumulator state |
| Is it persisted? | No — it's the write-path only. Persisted edges are bgz17 palette-compressed (3 bytes/edge) |
| Can VSA bundle these? | NO. These are accumulators in a specific algebraic register (saturating i8), not superposable fingerprints |
| Relation to ONNX? | The "ONNX 16kbit learning" was the D9 plan idea for exporting the accumulator state as a learnable graph that predicts state transitions. Deferred. |

**For D9 (ONNX arc export, deferred):**
- The story-arc predictor learns from accumulated (state, arc_pressure,
  arc_derivative) tuples
- Input format: Vsa16kF32 trajectory fingerprints (identity layer)
- NOT the 3x16kbit plane state (write-path layer)
- ONNX model consumes fingerprints, predicts next arc transition

---

## Callcenter BBB / Supabase Transcode — Intent Preservation

**Status:** SPECULATIVE — not yet tracked as a deliverable. The
discussion references callcenter intent classification, persona-agent
routing, and Supabase persistence. None of these exist in code yet.

**Preserved intent (for future planning sessions):**

1. **Callcenter agents as persona-registry consumers.** Each agent
   role (greeting, triage, resolution, escalation) is a persona in
   the shared persona registry. Identity fingerprint routes signal
   to agent; YAML content defines the agent's behavior.

2. **Intent classification via VSA resonance.** Caller turn →
   extracted role bundle (INTENT + SENTIMENT + URGENCY slots) →
   cosine-rank against intent codebook → top-K candidates → dispatch
   to resolution flow.

3. **Supabase as content layer (NOT VSA layer).** Supabase stores:
   - Persona YAML (agent definitions)
   - Intent definitions (resolution flows)
   - Call transcripts (episodic archive)
   - Cumulative awareness (per-agent per-customer NARS truth)
   
   Supabase does NOT store VSA bundles — those are ephemeral compute
   state, regenerated per call.

4. **"BBB" references** in prior conversations referred to specific
   callcenter use cases (Better Business Bureau complaint handling
   as a benchmark scenario). These are SPECULATIVE benchmarks, not
   shipped features. They illustrate the pattern; they're not on the
   roadmap.

**Don't ship without explicit user green-light.** The architecture
SUPPORTS callcenter applications; the current scope doesn't include
them. Preserve the framing for when callcenter becomes a tracked
deliverable.

---

## What Cleanup Replaced

**Pre-cleanup misalignment (this session 2026-04-21 D5 shipped files):**

1. `crates/deepnsm/src/content_fp.rs` — used `Vsa10k = [u64; 157]`
2. `crates/deepnsm/src/markov_bundle.rs` — XOR-bundled bitpacked
   vectors, "braiding" via bit rotation
3. `crates/deepnsm/src/trajectory.rs` — `recovery_margin` as Hamming
   within slice on bitpacked vectors
4. `crates/lance-graph-contract/src/grammar/role_keys.rs` additions:
   `RoleKey::bind/unbind/recovery_margin` on `[u64; 157]`, `vsa_xor`,
   `Vsa10k` type alias

**All of these are structurally wrong.** They built GF(2) algebra on
a bitpacked binary substrate when the stack uses ℝ algebra on
`Vsa10kF32 = Box<[f32; 10_000]>` (or future `Vsa16kF32 = Box<[f32;
16_384]>`).

**The corrected path (not yet implemented):**

1. Revert the four above to their pre-session state
2. Re-implement on `Vsa16kF32` carrier (after Vsa10k→Vsa16k coordinated
   PR lands in contract + ndarray)
3. Role keys become `Box<[f32; 16_384]>` bipolar values living in
   `grammar/role_keys.rs` (catalogue only, no methods — methods live
   on the carrier in `crystal/fingerprint.rs`)
4. `bind` → `vsa_bind` (multiply); `unbind` → same (self-inverse);
   `recovery_margin` → cosine over slice; `vsa_xor` → DELETED,
   use `vsa_bundle` (add)

This cleanup doc PRESERVES the architectural insights (Five Lenses,
Think struct, tissue-not-storage, shader-cant-resist, grammar-of-
awareness) while correcting the concrete format choice.

---

## Cross-References

- `EPIPHANIES.md` — see CORRECTION-OF entries dated 2026-04-21 cleanup
- `.claude/board/TECH_DEBT.md` — VSA substrate rename debt, D5 revert debt
- `CLAUDE.md § The Click (P-1)` — meta-architecture (correct), updated to reflect FP32 multiply/add
- `CLAUDE.md § I-CAMPQ-VS-VSA iron rule` — refined to "VSA operates on identities"
- `.claude/plans/categorical-algebraic-inference-v1.md` — Five Lenses plan (correct)
- `.claude/knowledge/paper-landscape-grammar-parsing.md` — 14-paper landscape (correct)
- `.claude/knowledge/session-2026-04-21-categorical-click.md` — session handover (needs correction annotation)

