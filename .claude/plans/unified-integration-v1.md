# Plan — Unified Integration: PersonaHub × ONNX × Archetype × MM-CoT × RoleDB

> **Version:** v1
> **Author:** main-thread session 2026-04-23
> **Status:** Active — brainstorm phase complete; deliverables defined; no code shipped yet
> **Confidence:** CONJECTURE — all integration points grounded in repo evidence and upstream docs;
> no implementations shipped yet beyond DM-0 / DM-1 skeleton in callcenter-membrane-v1.md.
>
> **READ BY:** Any session touching persona modeling, thinking style selection, the ONNX oracle,
> Archetype ECS integration, MM-CoT stage split, or the RoleDB DataFusion UDFs.
>
> **Related plan:** `.claude/plans/callcenter-membrane-v1.md` §§ 15–17 (architecture ground truth)

---

## § 0 — One-Sentence Goal

Integrate four upstream systems (PersonaHub, ONNX classifier, Archetype ECS, MM-CoT)
into the lance-graph cognitive substrate WITHOUT adding new architectural layers —
each maps onto existing contract types with zero new traits.

---

## § 1 — The Integration Surface Map

```
PersonaHub (Tencent) ──► PersonaSignature (56-bit) ──► template_id ──► YAML runbook
Chronos (Amazon) ────► SUPERSEDED by ONNX classifier (see DU-1)
ONNX classifier ─────► OnnxPersonaClassifier ──► style_oracle in Think struct
Archetype (VangelisTech) ──► sits ABOVE callcenter ──► Entity=PersonaCard, World=Blackboard
MM-CoT (Amazon) ─────► rationale_phase: bool in CognitiveEventRow (zero new types)
RoleDB ──────────────► DataFusion VSA UDFs (5 functions) ──► internal_dataset queryable
```

---

## § 2 — Deliverables

### DU-0 — PersonaHub Compression (offline extraction)

**Scope:** Compress PersonaHub 370M personas to 56-bit signatures.

**Output artifacts:**
- Flat binary: `personas.bin` — 370M × 7 bytes ≈ 2.6 GB
- Deduplicated signature table: `sigs_dedup.bin` — ~1–5M unique signatures × 7 bytes
- YAML template library: `templates/*.yaml` — ~10K runbook templates
- Index: `sig_to_template.u16.bin` — `signature → template_id` lookup

**Algorithm:**
1. Stream PersonaHub HF parquet (`tencent-ailab/persona-hub`, ~3 parquet shards)
2. For each persona JSON: parse `system_prompt` → detect active atoms (32-bit bitset)
3. Score atom activation intensity (0–15 per atom) → `palette_weight: u8` coarse encoding
4. Cluster similar signatures → assign `template_id` from nearest YAML template centroid
5. Pack `(atom_bitset: u32, palette_weight: u8, template_id: u16)` → 7-byte record

**Atom detection** — NLP heuristics (no LLM needed):
- `deduction`: "therefore", "it follows", "conclude"
- `counterfactual`: "if instead", "had been", "would have"
- `induction`: "in general", "pattern", "typically"
- … (32 keyword/phrase patterns, one per atom)

**ONNX connection:** deduplicated signatures + F-labeled Lance rows = training corpus for DU-1.

**Status:** Queued  
**Effort:** ~2 days (streaming extraction; no training)  
**Owner:** offline tool; can run on HF dataset in Colab or local machine

---

### DU-1 — ONNX Persona Classifier @ L4/L5

**Scope:** Replace `StyleSelector::Auto` (static qualia→style rule) with a learned
ONNX classifier that predicts the full `(ExternalRole, ThinkingStyle)` product.

**Why ONNX over Chronos:**
- Output: 288 logits (full persona product) vs Chronos 1D scalar (style only)
- Task: classification vs time-series forecasting
- Training: from Lance E-DEPLOY-1 corpus (already labeled by F outcome)
- Infra: `ort` crate justified by existing Jina v5 ONNX on disk
- Precision: full `role × thinking` product vs style axis only

**Input/Output:**
```
Input:
  recent_fingerprints: Tensor[N, 16384]  // N cycle fingerprints [u64;256] as f32 bits
                                          // L4/L5 speed-lane format — fingerprint, NOT Vsa10k
                                          // (L4/L5 is motion/learning/fast dispatch; stays at 2KB/row)
  current_meta: Tensor[4]                // MetaWord: thinking, awareness, nars_f, nars_c unpacked

Output:
  logits: Tensor[288]  // log-softmax over (ExternalRole × ThinkingStyle) pairs
  argmax → PersonaId { role: ExternalRole, style: ThinkingStyle }
```

**Training:**
```
Source: Lance internal_cold dataset
Schema: { fingerprint: [u64;256], meta: MetaWord, f_outcome: f32, role: u8, style: u8 }
Labels: f_outcome < 0.2 → positive (committed persona = correct prediction)
        f_outcome > 0.8 → negative (failed cycle = wrong persona for context)
Min corpus: ~10K labeled cycles before training is meaningful
Target model size: < 2MB ONNX (fits in-process, hot-reloadable)
```

**Integration:** `Think.style_oracle: Option<&OnnxPersonaClassifier>`
- `None` → falls back to `StyleSelector::Auto` (existing static rule)
- `Some(oracle)` → oracle prediction replaces static rule; `StyleSelector::Auto` still runs
  as confidence calibration baseline

**Layer:** L4/L5 (internal). Never exposed externally. ONNX file is internal asset.

**Status:** Queued  
**Effort:** ~3 days (training script + ort integration + Think struct addition)  
**Blocking:** needs ~10K labeled cycles from Lance internal_cold (DM-2 must ship first)

---

### DU-2 — Archetype ECS Integration Layer

**Upstream:** `https://github.com/VangelisTech/archetype` — Rust ECS simulation engine,
DataFrame-first, LanceDB-backed, tick-based.

**Mapping:**

| Archetype concept | Callcenter / lance-graph concept | Notes |
|---|---|---|
| `Entity` | `PersonaCard = (ExternalRole, ThinkingStyle)` | One entity per canonical persona |
| `Component` | `FacultyDescriptor` field | Each component = one faculty axis |
| `Processor` | `OrchestrationBridge::route()` | Processes steps each tick |
| `World` | `Blackboard` | Shared state across all entities per round |
| `Tick` | Blackboard round + CollapseGate fire | One tick = one committed turn |
| `System` | `FacultyDescriptor.tools: &[ToolAbility]` | Systems declared per entity type |
| `DataFrame` | Arrow RecordBatch from `project()` | Archetype is DataFrame-first |
| `LanceDB` | Lance external_dataset | Archetype persists to Lance natively |

**Position in stack:**

```
q2 (GUI: Gotham/Neo4j/Quarto) ──► Archetype ECS (simulation layer)
                                          │
                                          ▼
                                 callcenter (headless wire)
                                          │
                                          ▼
                            lance-graph cognitive substrate
```

Archetype sits ABOVE the callcenter. It is the simulation orchestrator that
drives tick cycles. The callcenter remains headless wire — it does not know
about Archetype. Archetype drives it by calling `ExternalMembrane::ingest()`
per tick.

**Why not merge Archetype into lance-graph:**
- Archetype is ECS simulation; callcenter is cognitive wire
- Archetype belongs in q2's simulation layer; callcenter belongs at wire level
- Keeping them separate preserves the q2=GUI / callcenter=headless boundary

**Integration deliverable (per ADR 0001 Decision 1 — TRANSCODE, not bridge):**
native Rust crate `lance-graph-archetype` that assimilates the Archetype
ECS contracts against Lance + DataFusion + Arrow. Python upstream is
design spec, not runtime dependency.

- `Component` trait maps to Arrow field
- `Processor` trait = stateless `RecordBatch → RecordBatch` transform
- `World` = Lance versioned dataset + tick counter + `CommandBroker`
- `ArchetypeWorld.tick()` = drain commands → run processors → append snapshot
- `ArchetypeWorld.fork()` = Lance version branch (free time-travel / counterfactual)
- `project()` emits Arrow `RecordBatch` consumed by `ExternalMembrane`

**Not "bridge":** the crate does not FFI into Python Archetype at
runtime. It transcodes the contracts natively. The upstream Python
package is the DESIGN reference, not a live dependency. See
`.claude/adr/0001-archetype-transcode-stack.md` Decision 1.

**Status:** Queued (naming + scope updated post-ADR 0001)
**Effort:** ~3 days (native crate; no changes to callcenter or lance-graph core)

---

### DU-3 — RoleDB DataFusion VSA UDFs

**Scope:** Register 5 VSA UDFs in DataFusion so the internal dataset is queryable
as a "DuckDB over roles."

**UDF registry:**

```rust
// In lance-graph core or lance-graph-callcenter:
fn register_vsa_udfs(ctx: &SessionContext) {
    ctx.register_udf(unbind_udf());      // (u8, [u64;256]) → f32
    ctx.register_udf(bundle_udf());      // ([u64;256]...) → [u64;256]
    ctx.register_udf(hamming_dist_udf()); // ([u64;256], [u64;256]) → u32
    ctx.register_udf(braid_at_udf());    // (i32, [u64;256]) → [u64;256]
    ctx.register_udf(top_k_udf());       // ([u64;256], u32) → list[u16]
}
```

**Example query (dispatch scoring):**
```sql
SELECT expert_id, role, style,
       unbind(target_role, fingerprint) AS dispatch_score
FROM internal_dataset
WHERE round >= (SELECT max(round) - 5 FROM internal_dataset)  -- ±5 Markov window
ORDER BY dispatch_score DESC
LIMIT 10;
```

**BBB compliance:** UDFs operate on internal_dataset only. Results are scalar
(f32 dispatch scores, u32 Hamming distances). VSA types never cross to external_dataset.

**Status:** Queued  
**Effort:** ~2 days (DataFusion ScalarUDF boilerplate; VSA ops already exist in contract)

---

### DU-4 — MM-CoT Stage Split (minimal, zero new types)

**Upstream:** `https://github.com/amazon-science/mm-cot` — two-stage CoT
(rationale generation → answer generation) with optional visual features.

**Mapping to existing architecture:**
- MM-CoT Stage 1 (rationale) = `FacultyDescriptor.inbound_style` (thinking mode)
- MM-CoT Stage 2 (answer) = `FacultyDescriptor.outbound_style` (emission mode)
- `FacultyDescriptor.is_asymmetric()` returns `true` iff stages differ = MM-CoT condition
- Visual features = future `FacultyRole::Vision` variant (not in scope for DU-4 v1)

**Code change:** one field added to `CognitiveEventRow`:
```rust
pub rationale_phase: bool,  // true = Stage 1 rationale, false = Stage 2 answer
```

This field surfaces in the projected RecordBatch so external subscribers can filter
to either stage. No new trait, no new struct.

**Status:** Shipped (2026-04-23) — `rationale_phase: bool` added to `CognitiveEventRow`
in `external_intent.rs`; `project()` in `lance_membrane.rs` populates `rationale_phase: false`
(Phase A stub). Commit `a05979e`.

---

### DU-5 — Board Hygiene + STATUS_BOARD Update

**Scope:** Register DU-0 through DU-4 in STATUS_BOARD.md, update INTEGRATION_PLANS.md,
update LATEST_STATE.md contract inventory with new plan sections.

**Status:** Shipped (2026-04-23) — `unified-integration-v1` entry prepended to
`INTEGRATION_PLANS.md`; DU-* rows appended to `STATUS_BOARD.md`. Commit `a05979e`.

---

## § 3 — Sequencing

```
DM-2 (LanceMembrane impl) ──► DU-1 needs labeled corpus (Lance internal_cold)
DU-0 (PersonaHub compress) → can run in parallel with DM-2 (offline, no deps)
DU-3 (RoleDB UDFs)         → can run in parallel with DM-2 (DataFusion, no deps)
DU-4 (MM-CoT field)        → can run immediately (trivial, no deps)
DU-5 (Board hygiene)       → run last

DU-1 (ONNX oracle)  ──► needs DM-2 (corpus) + DU-0 (atom vocabulary for feature alignment)
DU-2 (Archetype)    ──► needs DM-2 (ExternalMembrane impl to adapt against)
```

**Recommended order:**
1. DU-4 (trivial, unlock immediately)
2. DU-3 (UDFs, unblocks RoleDB queries)
3. DU-0 (offline extraction, runs async)
4. DM-2 (LanceMembrane impl — unlocks DU-1, DU-2)
5. DU-1 (ONNX oracle, depends on DM-2 corpus)
6. DU-2 (Archetype transcode crate, depends on DM-2)
7. DU-5 (board hygiene, last)

---

## § 4 — Invariants (must not be violated)

1. **BBB invariant:** `DU-1` ONNX oracle output is `PersonaId { role, style }` — scalar only.
   The ONNX model lives internal. No VSA tensor crosses to external_dataset.

2. **I-SUBSTRATE-MARKOV:** `bundle_udf` in DU-3 MUST use `MergeMode::Bundle`, not `MergeMode::Xor`.
   XOR breaks the CK guarantee for state-transition paths.

3. **q2=GUI / callcenter=headless:** DU-2 `lance-graph-archetype` (the native
   transcode crate per ADR 0001) does NOT put ECS logic inside
   `lance-graph-callcenter`. The transcode crate sits alongside, calling INTO
   the callcenter via `ExternalMembrane::ingest()`. Callcenter has zero
   knowledge of Archetype.

4. **Jirak-derived thresholds (I-NOISE-FLOOR-JIRAK):** `unbind_udf` dispatch scores are
   probabilistic, not exact. Any significance threshold for dispatch scoring must cite
   Jirak 2016, not classical Berry-Esseen.

5. **`role × thinking = persona` is the identity:** PersonaCard does not gain new fields
   beyond `(role: ExternalRole, style: ThinkingStyle)`. PersonaSignature (56-bit) is a
   compressed fingerprint, not the identity.

---

## § 5 — Open Questions

- **DU-0 atom vocabulary stability:** The 32 named atoms are provisional. Do they need
  to be aligned with any external taxonomy (e.g., FrameNet, ACT-R cognitive operators)?
  Current stance: internal vocabulary, evolves with F-descent evidence.

- **DU-1 minimum corpus size:** 10K labeled cycles is a rough estimate. Actual minimum
  depends on class balance across 288 persona classes. Some personas may be rare in
  practice (e.g., `(OpenClaw, Koan)`) — rare class handling may be needed.

- **DU-2 Archetype API stability:** `VangelisTech/archetype` is an active Python repo,
  which is the DESIGN spec (not a runtime dependency). The `lance-graph-archetype`
  transcode crate defines its OWN Rust interface based on the conceptual contracts
  (`Component`, `Processor`, `World`, `CommandBroker`) rather than mirroring Python
  signatures. Upstream API changes inform future transcodes; they don't break running
  code. See ADR 0001 Decision 1.

- **DU-3 fingerprint column type in DataFusion:** `[u64; 256]` is not a native Arrow type.
  Representation options: `FixedSizeBinary(2048)`, `FixedSizeList<u64>(256)`, or custom
  extension type. Choose before writing UDFs — type determines UDF signature.

---

## § 6 — SoAReview Sweep Findings (2026-04-24)

**Added 2026-04-24** by the first run of the new `@soa-review` agent
(`.claude/agents/soa-review.md`). Four parallel Opus-level angle
reviews: callcenter / archetype / persona / grammar-Markov column layout.

### Cross-cutting verdict

The biggest workspace-level DRIFT surfaced by the sweep:

> **`FingerprintColumns.cycle` in `bindspace.rs` is `Box<[u64]>`
> (Binary16K), not `Box<[f32; 16_384]>` (Vsa16kF32) as CLAUDE.md §The
> Click mandates.**

This single DRIFT blocks DU-3 (RoleDB VSA UDFs cannot bind-unbind
cleanly on a binary carrier), blocks the semantic-kernel (Markov +
CAM-PQ) integration, and leaves `MarkovBundler` / `Trajectory` /
`vsa_permute` as doc-referenced-but-unimplemented types. It is the
substrate precondition for most remaining deliverables.

### Angle-by-angle verdict

| Angle | Verdict | One-sentence reason |
|---|---|---|
| Callcenter | **PARTIAL** | BBB spine LIVE post-commit `564aac4`; `dialect` + `scent` + `subscribe()` remain ghosts; three `vsa_udfs.rs` ops broken. |
| Archetype | **LOCKED-MAPPING-INCOMPLETE** | ADR-0001 locks the transcode; `lance-graph-archetype` crate not yet created; AsyncProcessor / CommandBroker / Component Rust types undefined. |
| Persona | **DRIFTING-BUT-MANAGEABLE** | Contract-side `PersonaCard` BBB-clean; `thinking-engine::persona::PersonaProfile` carries 12 f32 soul priors as struct content; `persona/role_keys.rs` catalogue missing. |
| Grammar-Markov | **SCATTERED-NOT-UNIFIED** | FingerprintColumns drift above; `global_context` absent; son/father/grandfather permutation-offset retrieval has no method and no named epiphany entry. |

### Three-role-taxonomy check

| Role taxonomy | Catalogue | Status |
|---|---|---|
| Grammatical (SUBJECT/PRED/OBJ/TEKAMOLO/Finnish cases/tense) | `contract::grammar::role_keys` | **LIVE** |
| User / Agent / Persona (ExternalRole + ExpertId) | `contract::persona::role_keys` | **MISSING** (file does not exist) |
| Thinking-style (36 ThinkingStyles + faculty asymmetric styles) | `contract::thinking_styles::role_keys` | **MISSING** (file does not exist) |

The three taxonomies must coexist via disjoint slice allocations on
the same Vsa16kF32 carrier. Today only one of three is shipped. The
missing two are prerequisites for the semantic-kernel (= Markov +
CAM-PQ) to superpose all three role bindings into one trajectory row.

### Ranked expansion list (smallest wire first)

1. **Vsa16kF32 substrate migration** — change `FingerprintColumns.cycle`
   from `Box<[u64]>` (Binary16K) to `Box<[f32; 16_384]>` (Vsa16kF32).
   Blocks every downstream role-key bind/unbind. Touches `bindspace.rs`
   + `vsa_udfs.rs` + `crystal/fingerprint.rs`. Estimated ~200 LOC +
   rebuild test suite.
2. **`persona/role_keys.rs`** — disjoint slice catalogue for the
   User/Agent/Persona role taxonomy. Mirror `grammar/role_keys.rs`
   shape. ~160 LOC. Unblocks persona DRIFT.
3. **`thinking_styles/role_keys.rs`** — same pattern for the
   36-style taxonomy. ~120 LOC. Unblocks semantic-kernel bundle
   completeness.
4. **Slice allocation map** — `contract/src/vsa_layout.rs` listing
   every `[start:end)` across all three taxonomies; `const_assert`
   on non-overlap. ~80 LOC. Compile-time proof of the three-role
   invariant.
5. **`MarkovBundler` + `Trajectory` + `vsa_permute`** —
   implementation of the doc-referenced types on the new Vsa16kF32
   carrier. Blocks the semantic-kernel commit tier. ~300 LOC in
   deepnsm.
6. **`global_context: Box<[f32; 16_384]>`** field on `BindSpace` +
   `accumulate_fact()` method. Per CLAUDE.md `Think { global_context:
   &Vsa10k }`. ~40 LOC.
7. **`persona/content_store.rs` + YAML migration** — move 12 f32 soul
   priors out of `PersonaProfile` (DRIFT) into YAML under ExpertId.
   ~150 LOC + YAML files.
8. **`dialect` + `scent` ghost-column wiring** — the smallest remaining
   callcenter wiring. See TECH_DEBT 2026-04-24 ghost-columns entry.
9. **`lance-graph-archetype` crate scaffold** — DU-2 Queued; Agent B
   work. Creates the crate; defines AsyncProcessor / CommandBroker /
   Component traits per ADR-0001.
10. **`subscribe()` live implementation** — DM-4 LanceVersionWatcher +
    DM-6 DrainTask in lance_membrane.

Items 1-4 are the substrate-level unification. Items 5-7 are the
semantic-kernel itself. Items 8-10 are the surface-level ghosts that
become tractable once the substrate is in place.

### Semantic-kernel definition (new framing, 2026-04-24)

```
Markov + CAM-PQ = semantic kernel
```

Markov (VSA bundle on Vsa16kF32) produces a lossless commit tier
(Index regime). CAM-PQ compressed code produces an argmax search
tier. Cascade: CAM-PQ ADC narrows candidates → exact Vsa16kF32 unbind
on survivors. All three role taxonomies (grammar × persona × thinking)
superpose losslessly into one trajectory row. Content (200-500 grammar
templates, soul priors, style definitions) stays in content stores;
never enters the VSA register.

### Reusability inside / outside BBB

SoA + DTO enforce the discipline: `BindSpace` four columns on the
inside; `CognitiveEventRow` Arrow-scalar on the outside. Same algebra,
different codec regime per side. Markov and Supabase-shape patterns
reuse across the gate precisely because the compile-time deny-list
on `ExternalMembrane::Commit` stops VSA / RoleKey / NarsTruth from
leaking through.

Cross-ref: `.claude/agents/soa-review.md` (the agent card with full
process + reference run notes); EPIPHANIES 2026-04-24 "I1 Codec
Regime Split" + "SoAReview sweep"; TECH_DEBT 2026-04-24 ghost-columns
+ pillar 5b + AriGraph episodic CAM-PQ cascade.
