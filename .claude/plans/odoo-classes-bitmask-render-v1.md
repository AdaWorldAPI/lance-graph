# odoo-classes-bitmask-render-v1 — the bounded-weekend fix classes.md prescribes, with per-agent A2A split

> **READ BY:** every agent spawned for any D-CLS-* row, the council if convened on this plan, the spec owner before unblocking pre-conditions.
> **Status:** PLAN (pre-council). Authored 2026-05-30 after the post-#439 rebase, the 4-savant council recalibration, and the on-disk inventory that found 4-way `DolceCategory` duplication + zero `class_id` / `presence_mask` / template engine.
> **Anchors:** `cognitive-risc-classes.md` v0.2 (the spec this plan implements verbatim), `cognitive-risc-core.md` v0.1 (invariants 1, 2, 3, 4, 9, 10), `wikidata-hhtl-load.md` (OWL→HHTL facet table), `faiss-homology-cam-pq.md` (facet u64 = SoA facet column, SIMD batch-AND).
> **Doctrine line, verbatim** (classes.md:56-57): *"The fix is bounded (a weekend, not a subsystem): discriminator + parent-pointer + parent-walking resolution against the existing cache. Full machinery (shape-compiler-to-grid, behavior/traits, SIMD kernels) is explicitly DEFERRED."*

---

## 0. Non-goals (explicit deferrals — DO NOT scope-creep)

- Full shape-compiler-to-grid. Out of scope (classes.md:57).
- SIMD kernels over the facet column. Out of scope.
- Behavior/trait composition (the recipe layer). Already partially exists (`recipe_kernels.rs` from #439); not extended here.
- Wikidata 115M-entity loader. That's a different plan (wikidata-hhtl-load.md).
- Chess bring-up as a behaviour test. The chess slice IS the N4 falsifier this plan honours by NOT freezing the SoA schema; chess implementation lives in `D-CHESS-BRINGUP-1` (queued, separate branch).
- Hot-path mailbox-SoA wiring of the new column. This plan adds `class_id` to the `OdooEntity` **const** (the class catalogue) — NOT to the runtime row. Runtime wiring is a follow-up plan once class IDs are stable.
- The `discovery_origin` byte. Different axis (provenance); held under OD-1/2/3 escalation.

## 1. Hard constraints (violating any aborts the plan)

- **C1** — WAL is OPEN. All work here is markdown + new contract types + new const data. NO byte-grammar freeze.
- **C2** — Bitmask is PRESENCE, NEVER semantics (classes.md:50). A bit means "field N is populated"; it does NOT mean "field N behaves differently here." Any code that branches on a mask bit's *meaning* (rather than its presence) violates this and must be rejected at review.
- **C3** — Shape inherits, behaviour composes (classes.md:37). `class_id` carries field/label/template inheritance. It does NOT carry method dispatch. Methods stay on `OdooMethod` recipes.
- **C4** — Universal column ISA is frozen; per-class grids are views (classes.md:38). Adding `class_id` to `OdooEntity` is adding a discriminator, not minting a new column type per shape-family.
- **C5** — Class taxonomy is DISCOVERED, not hand-assigned (classes.md:41-44). Shape-families come from Aerial+ on structural signature, not human selection.
- **C6** — All deliverables are additive. No rename of existing types. No deletion. The 4-way `DolceCategory` consolidation collapses by adding ONE canonical type + thin re-exports; legacy aliases stay for one release cycle.
- **C7** — Per CLAUDE.md mandatory board-hygiene rule: every PR landing a D-CLS-* row updates STATUS_BOARD + AGENT_LOG in the SAME commit.
- **C8** (NEW, council F7 from R1) — No `class_id` field, `ClassId` table, or per-class registry may occupy a byte position reserved for `discovery_origin` widening (per core.md:55-62 N2). The classes-spec hook lands WITHOUT burning byte real estate the proposer-id widening still needs. Specifically: `ClassId(u16)` lives in its own column on the SoA, NOT packed into the same byte as `discovery_origin`.

## 2. Spec-owner pre-conditions (blocking; user must answer before code)

These are decisions only the spec owner / project lead can make. **No agent should start D-CLS-1+ until all FIVE are answered.**

> **2026-05-31 honesty pass** (8-savant council fact-finding, B1 + R2 + R3 catches): the v1-as-authored §2 presented 3 of 4 "default leans" as spec-citations when only one was real. The table below states honestly, per OD, whether the lean is from the spec or from the author. Author-leans are still useful as a starting point for ratify/reject; they are NOT to be cited as "the spec says X."

| OD | Question | Lean source | Lean (for ratify/reject) |
|---|---|---|---|
| **OD-DOLCE-VARIANT-SET** (NEW, council F2) | Which canonical variant SET wins? The 4 sites have **different variant counts AND name conflicts**: `contract::cognition::entity` = 6 variants; `ontology::hydrators::dolce_odoo` = 4 with `AbstractEntity`; `arm-discovery::aerial::ontology` = 4 with `Abstract`; `callcenter::super_domain::DolceMarker` = 5 with `Unknown`. Picking a canonical CRATE alone (OD-DOLCE-CANONICAL below) is not enough — `AbstractEntity ≠ AbstractObject ≠ Abstract` so re-export fails to compile. Spec owner must declare the canonical variant set. | **NO SPEC LEAN** — author suggestion only | Author suggestion: 5-variant set `{Endurant, Perdurant, Quality, Abstract, Unknown}` (DOLCE-Lite-Plus four + the real-world-necessary Unknown from callcenter). Other naming (`AbstractObject`/`AbstractEntity`) becomes deprecated aliases pointing at `Abstract`. Reject if you prefer DUL naming (`Object`/`Event` instead of `Endurant`/`Perdurant`). |
| **OD-DOLCE-CANONICAL** | Which crate owns the canonical `DolceCategory` definition? | **NO SPEC LEAN** — author taste | Author suggestion: `lance-graph-contract::cognition::entity::DolceCategory` (zero-dep, already in cognition pipeline). Spec names no crate; treat this as author preference. |
| **OD-CLASSID-WIDTH** | `class_id: u8` (≤256 classes), `u16` (≤65,535), or `ShapeHash(u128)` (CAM-style)? | **NO SPEC LEAN** — author cross-fielded from N2 | Author suggestion: `ClassId(u16)`. **Honesty correction (council B1):** spec line 64's `u16` reference is about **proposer-id width (N2)**, not class_id (N1). The spec is **silent** on class_id width. R2's reviewer recommended `u8` ("≤40 shape-families today; bounded-weekend defers thousands-class scale to a follow-up"). Spec owner: pick u8 or u16 on merits, NOT by misciting the spec. |
| **OD-CLASSID-VS-ENTITYKIND** | Does `class_id` *replace* `OdooEntity.kind: OdooEntityKind` ({Model, Transient, Abstract})? Or coexist? | **NO SPEC LEAN** — `OdooEntityKind` not in spec | **Honesty correction (council B1):** `OdooEntityKind` appears **nowhere** in cognitive-risc-classes.md. The question is author-invented; the spec has no opinion. Author suggestion: coexist (orthogonal axes — ORM-base vs semantic-shape). Reject if you'd rather collapse them. |
| **OD-TEMPLATE-ENGINE** | F3 from classes.md. Askama (compile-time typed) vs minijinja (runtime). | **REAL spec lean** (weakly worded) | Spec line 72 verbatim: *"templates likely compile per-class (askama) with OGIT resolving the late-bound labels at render."* "Likely" is the spec's word — not "must." Author concurs with askama; spec owner can flip to minijinja with a stated reason. |

Until these **five** are answered, every agent below has `Status: Blocked-on-OD`. F1-F7 council corrections (above + below) apply to plan structure regardless.

## 3. The pipeline (visual + textual)

```text
   ┌─ Wave 0 ───────────────────────────────────────────────────────────────┐
   │  Spec-owner ratifies OD-DOLCE-CANONICAL, OD-CLASSID-WIDTH,             │
   │  OD-CLASSID-VS-ENTITYKIND, OD-TEMPLATE-ENGINE.                         │
   │  Output: updated .claude/board/ISSUES.md status flips Open→Resolved.   │
   └─────────────────────────────────────┬──────────────────────────────────┘
                                         ▼
   ┌─ Wave 1 (3 parallel Sonnet agents) ────────────────────────────────────┐
   │  1A  Consolidate DolceCategory                    (contract crate)     │
   │  1B  Structural-signature audit of 66 entities    (read-only, table)   │
   │  1C  Add askama to one consumer crate Cargo.toml  (1-file edit)        │
   └─────────────────────────────────────┬──────────────────────────────────┘
                                         ▼
   ┌─ Wave 2 (1 Opus agent — judgment) ─────────────────────────────────────┐
   │  2A  Aerial+ structural-hash → shape-family candidates                 │
   │      Output: ~10-15 named candidate classes for human review.          │
   │  → spec owner reviews + names → CANONICAL_CLASS_TABLE static const     │
   └─────────────────────────────────────┬──────────────────────────────────┘
                                         ▼
   ┌─ Wave 3 (5 parallel Sonnet agents) ────────────────────────────────────┐
   │  3A  ClassId newtype + tests                      (contract)           │
   │  3B  class_id field on OdooEntity + 66 lanes      (ontology)           │
   │  3C  FieldPositionTable per class                  (ontology)           │
   │  3D  FieldMask(u64) + per-class width audit       (contract+ontology) │
   │  3E  ClassId → (DolceCategory, FieldPositionTable) (ontology)         │
   └─────────────────────────────────────┬──────────────────────────────────┘
                                         ▼
   ┌─ Wave 4 (3 parallel Sonnet agents) ────────────────────────────────────┐
   │  4A  Per-class askama templates (~10-15 .html.j2)  (new crate)         │
   │  4B  render(entity_const, mask) function          (new crate)          │
   │  4C  Unit tests: mask round-trip + render-skip    (new crate)          │
   └─────────────────────────────────────┬──────────────────────────────────┘
                                         ▼
   ┌─ Wave 5 (1 Opus agent — integration synthesis) ────────────────────────┐
   │  5A  End-to-end: 66 entities × class templates → 66 render outputs.    │
   │       Verify presence-discipline (C2): no bit branches on semantics.   │
   │       Flag for chess falsifier (D-CHESS-BRINGUP-1, separate branch).   │
   └────────────────────────────────────────────────────────────────────────┘
```

Total: **9 deliverables (D-CLS-1..D-CLS-9), 10 agent runs across 5 waves, ~1,800 LOC across 1 new crate + 3 modified crates.** No code on the hot path; only the const catalogue + render path.

---

## 4. Deliverables (D-CLS-1 through D-CLS-9 — meticulous)

### D-CLS-1 — Canonical `DolceCategory` in `lance-graph-contract`

**What.** Pick `lance-graph-contract::cognition::entity::DolceCategory` as canonical (per OD-DOLCE-CANONICAL). Add `Unknown` variant if not present. Add re-export aliases from the other 3 sites pointing at the canonical:
- `lance-graph-ontology::hydrators::dolce_odoo::DolceCategory` → `pub use lance_graph_contract::cognition::entity::DolceCategory;` + deprecate the local enum with `#[deprecated(note = "use lance_graph_contract::cognition::entity::DolceCategory")]`
- `lance-graph-arm-discovery::aerial::ontology::DolceCategory` → same pattern (but: arm-discovery is **zero-dep / excluded from workspace**, so the re-export uses a local newtype that mirrors the canonical and a TryFrom; do NOT add `lance-graph-contract` as a dep — that violates the zero-dep stance of #436).
- `lance-graph-callcenter::super_domain::DolceMarker` → keep `DolceMarker` as-is (it's the *unknown-tolerant* projection); add `From<DolceCategory> for DolceMarker` to the canonical.

**Why.** classes.md known-debt: "stop hand-rolling the inheritance the cache was built to serve." Four parallel definitions IS the hand-rolling. C6 says additive — re-exports + deprecations, no rename.

**Files.**
- `crates/lance-graph-contract/src/cognition/entity.rs` — add `Unknown` variant if absent; add doc-comment noting canonical status.
- `crates/lance-graph-ontology/src/hydrators/dolce_odoo.rs` — re-export + deprecation.
- `crates/lance-graph-arm-discovery/src/aerial/ontology.rs` — local newtype + TryFrom (cannot add contract dep).
- `crates/lance-graph-callcenter/src/super_domain.rs` — add `From<DolceCategory> for DolceMarker`.
- **TESTS:** `crates/lance-graph-contract/src/cognition/entity.rs` (existing test module) — assert all 4 DOLCE canonical variants round-trip; assert callcenter `From` mapping preserves Endurant/Perdurant/Quality/Abstract and maps `Unknown` to `DolceMarker::Unknown`.

**Depends.** OD-DOLCE-CANONICAL ratified.
**Blocks.** D-CLS-2, D-CLS-5 (every downstream uses the canonical).
**LOC.** ~80 added, ~0 deleted (per C6 additive).
**Blast radius.** Contract crate (n8n-rs, crewai-rust consume it — additive only, no API break).
**Agent.** Wave 1A, Sonnet.

---

### D-CLS-2 — Structural-signature audit of the 66 `OdooEntity` consts

**What.** Read each of `crates/lance-graph-ontology/src/odoo_blueprint/l{1..15}.rs`. For each `pub const NAME: OdooEntity` produce a row:

```text
(const_name, model_name, kind, field_count, method_kind_histogram, decorator_kind_histogram, has_state_machine, depends_signature_hash, emits_signature_hash, structural_hash)
```

where `structural_hash` = stable hash over `(sorted_field_kinds, sorted_method_kinds, sorted_decorator_kinds, has_state_machine_bool, depends_signature_hash, emits_signature_hash)`. Use BLAKE3-128 truncated to u64 for the structural hash; this is the input Aerial+ groups on in D-CLS-3.

Output: `.claude/knowledge/odoo-66-structural-signatures.psv` — pipe-separated table, alphabetical by const_name.

**Why.** classes.md:43 — "Group the 20k by structural signature (which fields, compute-method shape, depends_on/emits pattern) — computable, via group-by-on-structural-hash or Aerial+." This is the structural-hash precomputation step. Aerial+ (D-CLS-3) groups by it.

**Files.**
- READ: all `crates/lance-graph-ontology/src/odoo_blueprint/l{1..15}.rs` (66 consts).
- WRITE: `.claude/knowledge/odoo-66-structural-signatures.psv` (NEW, ~70 lines).
- WRITE: `.claude/knowledge/odoo-66-structural-signatures-method.md` (NEW, ~80 LOC) — explains hash algorithm + reproducibility recipe.
- **TESTS:** none (read-only emission); but the method doc includes a manual reproduction command using `cargo test -p lance-graph-ontology --test signature_dump -- --nocapture` IF the agent decides to add a test-time emitter (optional, additive).

**Depends.** None — read-only.
**Blocks.** D-CLS-3.
**LOC.** ~150 in the PSV + ~80 in the method doc.
**Blast radius.** None (knowledge docs only).
**Agent.** Wave 1B, Sonnet.

---

### D-CLS-3 — Aerial+ → shape-family candidates (running existing #436 crate)

**What.** Use `lance-graph-arm-discovery::aerial::AerialProposer` over the structural-signature table from D-CLS-2. Each row = one "transaction"; each (field_kind, method_kind, decorator_kind) bit = one "item". Mine rules at `max_antecedent = 1` and `max_antecedent = 2` with the Jirak-bound floor (D-ARM-7 when it lands; until then, use the conservative `support ≥ 3, confidence ≥ 0.85` default per ARM-plan §3 — be explicit this is a placeholder).

Output: ranked list of structural-similarity clusters. Each cluster = candidate shape-family. Expected ~10-15 clusters from 66 entities per classes.md:42 ("~dozens of shape-families").

Emit: `.claude/knowledge/odoo-66-shape-family-candidates.md` with per-cluster:
- Cluster ID (provisional, e.g. `SF-CANDIDATE-001`)
- Member const names (typically 3-8 OdooEntity consts per cluster)
- Defining-signature digest (which field/method/decorator bits ALL members share)
- One-line semantic guess for the spec owner to ratify or rename

**Why.** classes.md:43-44 — "The shapes sharing a signature ARE a class; name the groups, don't hand-assign them." Aerial+ is the named tool for this in the spec.

**Files.**
- READ: `.claude/knowledge/odoo-66-structural-signatures.psv` (from D-CLS-2).
- READ: `crates/lance-graph-arm-discovery/src/aerial/` (existing #436 crate, no modification).
- WRITE: `crates/lance-graph-arm-discovery/examples/odoo_66_class_discovery.rs` (NEW example binary, ~150 LOC) — loads the PSV, runs AerialProposer, emits clusters as markdown.
- WRITE: `.claude/knowledge/odoo-66-shape-family-candidates.md` (NEW, ~200 LOC).
- **TESTS:** the example is itself a test — Sonnet runs `cargo run --example odoo_66_class_discovery` and pastes output into the markdown.

**Depends.** D-CLS-2.
**Blocks.** Wave 3 (spec owner must NAME the clusters before D-CLS-3a output becomes the CANONICAL_CLASS_TABLE).
**LOC.** ~350.
**Blast radius.** New example in `lance-graph-arm-discovery`; standalone, excluded from workspace, no consumer impact.
**Agent.** Wave 2A, **Opus** (running Aerial+ is mechanical; INTERPRETING the clusters into shape-family names requires judgment + cross-reference to the 66 entities' L-doc prose — accumulation).

**SPEC-OWNER GATE.** After D-CLS-3, the spec owner reviews the ~10-15 candidate clusters and either ratifies the auto-generated names or renames them. The ratified names become the `CANONICAL_CLASS_TABLE` (D-CLS-9). Until ratified, Wave 3 cannot start.

---

### D-CLS-4 — Add askama to the new render crate

**What.** Create a new workspace crate `lance-graph-ontology-render` (excluded from default workspace, like bgz17 — so the heavy lance/datafusion deps don't pull it). Add `askama = "0.12"` (or current). Skeleton crate with `src/lib.rs` + `templates/` directory + `Cargo.toml`. NO templates yet (D-CLS-7).

**Why.** OD-TEMPLATE-ENGINE = askama per spec lean. F3 in classes.md:72. Standalone crate keeps the template engine out of the contract crate (which is zero-dep) and out of `lance-graph-ontology` (which already has heavy deps; adding askama would bloat compile time for every consumer).

**Files.**
- `crates/lance-graph-ontology-render/Cargo.toml` (NEW, ~25 LOC).
- `crates/lance-graph-ontology-render/src/lib.rs` (NEW, ~20 LOC — skeleton with one smoke test).
- `crates/lance-graph-ontology-render/templates/.gitkeep` (NEW).
- `Cargo.toml` (workspace root) — add to `exclude = [...]` list (NOT to members, per the standalone pattern).
- **TESTS:** one smoke test in `lib.rs`: assert askama hello-world compiles.

**Depends.** OD-TEMPLATE-ENGINE ratified.
**Blocks.** D-CLS-7, D-CLS-8.
**LOC.** ~70.
**Blast radius.** None (new excluded crate).
**Agent.** Wave 1C, Sonnet.

---

### D-CLS-5 — `ClassId(u16)` newtype in `lance-graph-contract` (coordinates with PR #437's existing `MailboxSoaView::class_id()`)

**What.** Add `pub struct ClassId(pub u16);` to `lance-graph-contract::cognition::entity` (next to `OgitUriRef`), **with `#[repr(transparent)]`** so it's transmute-safe over `u16`. Implement `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`, `Default` (= 0 = "unclassified"). Add `pub const UNCLASSIFIED: ClassId = ClassId(0);` so `Default::default()` is meaningful. Add a doc-comment quoting classes.md:63 N1 verbatim.

**Honesty correction (council F3 from R3):** PR #437 (merged) already shipped `MailboxSoaView::class_id() -> &[u16]` at `crates/lance-graph-contract/src/soa_view.rs:46-58`, aliasing `entity_type`. This deliverable does NOT parallel-add; it WRAPS that existing slot. Update `MailboxSoaView::class_id()` signature to return `&[ClassId]` via the transmute-safe `#[repr(transparent)]` reinterpretation. Per F3, this brings the typed newtype and the existing `u16` slice borrow into ONE answer to "where is class_id," not two divergent ones.

**Honesty correction (council F1 from B1):** OD-CLASSID-WIDTH default lean is the author's, NOT the spec's. If spec owner picks `u8`, the entire deliverable trivially scales down (`pub struct ClassId(pub u8);`) and PR #437's `&[u16]` slice becomes `&[u8]` — a wider blast radius requiring a `soa_view.rs` API revision, not an additive `#[repr(transparent)]` reinterpretation. Flag for spec owner: u16 has the cheaper migration here because PR #437 already chose u16.

**Why.** Cognitive-RISC N1 (classes.md:63). The single hook the entire triangle hangs off. Spec is emphatic: "Add it even before full inheritance exists — it's the hook."

**Files.**
- `crates/lance-graph-contract/src/cognition/entity.rs` — add `ClassId` struct + `UNCLASSIFIED` const + doc + `#[repr(transparent)]`.
- `crates/lance-graph-contract/src/soa_view.rs` — update `MailboxSoaView::class_id()` return type to `&[ClassId]` via transmute (safe per `#[repr(transparent)]`). **THIS FILE WAS OMITTED from v1 §6 ownership matrix; council F3 caught it.**
- **TESTS:** assert `ClassId::UNCLASSIFIED == ClassId(0)`; `Default::default() == ClassId::UNCLASSIFIED`; `ClassId(u16::MAX)` constructs; transmute round-trip `&[u16] -> &[ClassId] -> &[u16]` is bitwise-identical.

**Depends.** OD-CLASSID-WIDTH ratified, OD-DOLCE-VARIANT-SET ratified (so the canonical class enum is decidable too).
**Blocks.** D-CLS-6, D-CLS-9, D-CLS-10.
**LOC.** ~60 (was 40; +20 for the soa_view.rs transmute migration).
**Blast radius.** Contract crate (consumers see new symbol + soa_view return-type tightening; existing call sites that bound the result to `&[u16]` need a one-liner `.iter().map(|c| c.0)` adapter — flagged in code review).
**Agent.** Wave 3A, Sonnet.

---

### D-CLS-6 — Add `class_id: ClassId` field to `OdooEntity` + back-fill all 66 consts

**What.** Add a `pub class_id: ClassId` field to `OdooEntity` struct at `crates/lance-graph-ontology/src/odoo_blueprint/mod.rs:115`. Default initialization to `ClassId::UNCLASSIFIED` so the 66 existing consts compile UNCHANGED initially (additive per C6).

THEN: walk all 66 consts, set `class_id` to the ratified `CANONICAL_CLASS_TABLE` value (from the spec owner's review of D-CLS-3 output). Each lane file gets the same surgical edit per its 2-6 consts.

**Why.** The hook lands on the entity catalogue. `OdooEntity` is the const-data class catalogue; this gives every catalogue entry its class discriminator. Per C4: this is adding a discriminator field, NOT minting a new column type per family — same struct, new field, all entities use it.

**Files.**
- `crates/lance-graph-ontology/src/odoo_blueprint/mod.rs` — `OdooEntity` struct gets `class_id` field; constructor docs updated.
- `crates/lance-graph-ontology/src/odoo_blueprint/l1.rs` through `l15.rs` — each of 66 const literals gets `class_id: SF_<NAME>,` (where `SF_<NAME>` references the CANONICAL_CLASS_TABLE shape-family).
- `crates/lance-graph-ontology/src/odoo_blueprint/CLASS_TABLE.rs` (NEW, ~80 LOC) — `pub const SF_<NAME>: ClassId = ClassId(N);` for each shape-family + a `static SHAPE_FAMILIES: &[(ClassId, &str)]` index.
- **TESTS:** `crates/lance-graph-ontology/src/odoo_blueprint/mod.rs` test module — assert ALL 66 consts have a non-default `class_id`; assert every class_id used appears in `SHAPE_FAMILIES`; assert no two shape-families share a `ClassId`.

**Depends.** D-CLS-5, D-CLS-3 (spec-owner ratification of the cluster names).
**Blocks.** D-CLS-7, D-CLS-8, D-CLS-9.
**LOC.** ~30 in mod.rs + ~80 in CLASS_TABLE + ~150 across 15 lane files (the `class_id: SF_*` line per const).
**Blast radius.** `lance-graph-ontology` only. No consumer crate sees a breaking change (new field, default initialization where unset).
**Agent.** Wave 3B, Sonnet (mechanical edit across 15 lane files + 1 new table file; clear input, clear output).

---

### D-CLS-7 — Per-class field-position table + `FieldMask(u64)`

**What.** For each shape-family in `CANONICAL_CLASS_TABLE`, define a `FieldPositionTable` — an append-only ordered list of `OdooField` names that members of this class can carry. The position in the list IS the bit position in the presence mask. New fields APPEND; retired fields are kept with a `Retired` marker (per C3-style discipline + classes.md:65 N3).

```rust
pub struct FieldPositionTable {
    pub class: ClassId,
    pub positions: &'static [FieldPosition],
}

pub struct FieldPosition {
    pub name: &'static str,           // field name as on OdooField
    pub status: FieldPositionStatus,  // Active | Retired
    pub since_version: u16,           // append-only audit
}

pub enum FieldPositionStatus { Active, Retired }

pub struct FieldMask(pub u64);  // bit N is set iff position N is populated
```

For each shape-family (the ~10-15 from CANONICAL_CLASS_TABLE), generate the position table by union of `field.name` across all `OdooEntity` consts assigned to that ClassId.

Audit: per-class field count ≤ 64. If any class exceeds, FLAG to spec owner (decision: split the class, OR widen to `FieldMask128(u128)`, OR move some fields to a sub-class). DO NOT auto-resolve.

**Why.** Classes.md:65 N3 — stable bitmask bit-positions per class, append-only. Mask width per spec: "dozens of bits" (line 51) — u64 is generous. Audit catches the union-disease early.

**Files.**
- `crates/lance-graph-contract/src/cognition/field_mask.rs` (NEW, ~80 LOC) — `FieldMask`, `FieldPosition`, `FieldPositionStatus`, `FieldPositionTable` types + their tests (mask AND/OR/coverage helpers; reject set-on-Retired).
- `crates/lance-graph-contract/src/cognition/mod.rs` — `pub mod field_mask;`.
- `crates/lance-graph-ontology/src/odoo_blueprint/CLASS_TABLE.rs` — extend with `pub const SF_<NAME>_POSITIONS: &'static [FieldPosition] = &[...];` per class + `pub const SF_<NAME>_TABLE: FieldPositionTable = FieldPositionTable { class: SF_<NAME>, positions: SF_<NAME>_POSITIONS };`.
- `crates/lance-graph-ontology/src/odoo_blueprint/class_audit.rs` (NEW, ~120 LOC) — emits the audit report at test-time; assertion test that all class field counts ≤ 64 (FAIL with explicit message if not, naming the offending class + field count).
- **TESTS:** `crates/lance-graph-contract/src/cognition/field_mask.rs` test module (mask round-trip; Retired-position reject; OR/AND); `crates/lance-graph-ontology/src/odoo_blueprint/class_audit.rs` test module (per-class width gate; per-class field-name uniqueness).

**Depends.** D-CLS-5 (ClassId), D-CLS-6 (CLASS_TABLE).
**Blocks.** D-CLS-8 (render needs mask + position table).
**LOC.** ~80 + ~120 + ~50 in CLASS_TABLE extensions ≈ 250.
**Blast radius.** Contract crate (additive new type); ontology crate (new const data + new audit test).
**Agent.** Wave 3C + 3D split — 3C does the contract-side types + tests, 3D does the ontology-side per-class tables + audit. Both Sonnet, both mechanical (input is the CLASS_TABLE from D-CLS-6 + the OdooField data from the existing 66 consts).

---

### D-CLS-8 — `render(entity_const, mask) -> String` + per-class askama templates

> **Council-flagged-as-deferrable (F5 from B2 + R4).** B2's scope-creep audit: this deliverable IS the "shape-compiler-to-grid" the spec defers (classes.md:57). The author rebranded "compile per-class grids from the universal column ISA" (classes.md:38) as "compile per-class templates from FieldPositionTable" and called it render — same machinery, same dispatch-on-class_id, same per-shape-family projection. R4 + R1's N4 catch: the 66 snapshots in D-CLS-9 freeze positions on Odoo data before chess (the N4 falsifier). **Spec owner decides whether this ships in this plan or pushes to a follow-up plan. NOT auto-promoted to In-Progress when OD gates close; requires separate ratification.**

**What.** In the new `lance-graph-ontology-render` crate:

1. Per shape-family in `CANONICAL_CLASS_TABLE`, write ONE askama template `templates/<shape_family>.html.j2` (or `.txt.j2` — see note). The template references only the fields in that class's `FieldPositionTable`; each field is wrapped in `{% if mask.has(POS_N) %} ... {% endif %}` (presence-only gate).

2. A render function:
```rust
pub fn render(entity: &OdooEntity, mask: FieldMask) -> Result<String, RenderErr>
```
that dispatches on `entity.class_id` to the right askama template and renders with the entity's data + the mask.

3. **CRITICAL — C2 enforcement at render time.** The render function refuses to read a field's *value* unless the corresponding bit in the mask is set. Even if the field has a default. This makes presence the actual gate, not just a hint.

Note on template extension: per F3 spec is silent on output format. Default to `.txt.j2` — the templates are *projections of the entity into a presentation format*, not necessarily HTML. HTML templates can be added per-class on top of the txt baseline. Templates are tiny (one per class, dozens of lines each).

**Why.** classes.md:46-51 — "Render = class template, fields gated by bitmask, off-bits skipped." This is the render path.

**Files.**
- `crates/lance-graph-ontology-render/src/lib.rs` (extend from D-CLS-4 skeleton) — add `render()`, `RenderErr`, dispatch logic.
- `crates/lance-graph-ontology-render/templates/<shape_family>.txt.j2` × N (where N = #classes from D-CLS-3) — typically 10-15 templates, ~30 LOC each ≈ 300-450 LOC total.
- `crates/lance-graph-ontology-render/src/templates.rs` (NEW, ~80 LOC) — askama derive structs (one per template, one per ClassId), each carrying the slice of `OdooField` borrows + the mask.
- **TESTS:** per-class smoke test (render returns Ok for a known good entity + mask; refuses to access a field when the bit is unset; renders the same output for the same (entity, mask) pair bit-for-bit identically).

**Depends.** D-CLS-4 (crate skeleton), D-CLS-6 (CLASS_TABLE), D-CLS-7 (FieldMask + per-class tables).
**Blocks.** D-CLS-9 (integration test).
**LOC.** ~80 + ~80 + ~350 (templates) ≈ 510. Templates dominate.
**Blast radius.** New crate; no consumer impact (consumers opt-in by depending on it).
**Agent.** Wave 4A + 4B split — 4A authors the ~10-15 templates (mechanical, one per shape-family); 4B writes the dispatch + render + template-derive structs. Wave 4C writes the per-class unit tests. All Sonnet.

---

### D-CLS-9 — Integration test: 66 entities × ratified classes × render

> **Council-flagged-as-deferrable (F5 from R4 + R1).** R4's doctrine catch: shipping 66 golden snapshots over the Odoo-only universe before chess has touched the column-ISA *is* the N4 freeze classes.md:73 F4 warns against — just spelled as test fixtures instead of WAL bytes. The "we're only touching the const catalogue" alibi (§0 line 17) does not survive: positions live in code, snapshots lock the rendered shape. **Same gate as D-CLS-8: ratify shipping-here vs follow-up before promoting to In-Progress.** R1's amendment: if it does ship, mark the snapshot test ADVISORY (not a blocking gate) and keep only the round-trip + C2 mutant test as the floor.

**What.** Single end-to-end test:
1. Iterate the 66 OdooEntity consts.
2. For each, look up its `class_id` → `FieldPositionTable`.
3. Compute the presence mask by walking the entity's `fields` slice and setting bit N iff `field.name == positions[N].name`.
4. Call `render(entity, mask)`.
5. Assert Ok + non-empty.
6. Snapshot the output (insta or hand-coded golden files) for each entity.

Then a **second test** asserting C2 (presence-not-semantics discipline): build a mutant mask that flips a bit OFF, render, assert the field's content does NOT appear in the output. Then flip the bit ON without populating the field — assert the render fails (or emits a clearly-empty value, depending on the canonical decision).

Then a **third (advisory) audit**: emit a markdown report listing per-shape-family the average mask density (fraction of positions populated across members) — this is the metric wikidata-hhtl-load.md:84 calls "THE optimization metric". Land it as a CI-visible artifact, not a gate.

**Why.** Wave-5 closure. Verifies the bounded-weekend objective: every OdooEntity can be classified, masked, and rendered through its class template; presence discipline holds.

**Files.**
- `crates/lance-graph-ontology-render/tests/integration_66_entities.rs` (NEW, ~250 LOC).
- `crates/lance-graph-ontology-render/tests/snapshots/` (NEW directory; 66 golden snapshot files, one per entity).
- `crates/lance-graph-ontology-render/src/audit.rs` (NEW, ~80 LOC) — mask-density metric + markdown reporter.
- `.claude/knowledge/odoo-66-mask-density-report.md` (NEW, generated by the audit test, ~80 LOC) — per-class density numbers + interpretation per wikidata-hhtl-load.md:84.
- **CI HOOK:** `cargo test -p lance-graph-ontology-render` is the gate; if any of the 66 entities fails to render, the test fails.

**Depends.** D-CLS-1 through D-CLS-8 all green.
**Blocks.** Nothing (this is the closure).
**LOC.** ~250 + ~80 + 66 snapshots × ~30 LOC each ≈ 2,310. Bulk is snapshots.
**Blast radius.** Render crate only.
**Agent.** Wave 5A, **Opus** (integration synthesis + the C2-discipline mutant-test design needs judgment about what "clearly-empty" means; this is not mechanical).

---

## 5. Per-agent split (the A2A wave map)

| Wave | Agent | Model | D-CLS | Deliverable | LOC | Parallel? |
|---:|---|---|---|---|---:|---|
| 1 | **1A** | Sonnet | D-CLS-1 | DolceCategory consolidation | 80 | Yes |
| 1 | **1B** | Sonnet | D-CLS-2 | Structural-signature audit of 66 | 230 | Yes |
| 1 | **1C** | Sonnet | D-CLS-4 | Render crate skeleton + askama dep | 70 | Yes |
| 2 | **2A** | **Opus** | D-CLS-3 | Aerial+ structural-hash → shape-family candidates | 350 | No (sequential after Wave 1) |
| — | (spec owner) | — | — | Ratifies cluster names → CANONICAL_CLASS_TABLE | — | Blocking gate |
| 3 | **3A** | Sonnet | D-CLS-5 | ClassId(u16) newtype in contract | 40 | Yes |
| 3 | **3B** | Sonnet | D-CLS-6 | class_id field + back-fill 66 consts | 260 | Yes |
| 3 | **3C** | Sonnet | D-CLS-7 (contract side) | FieldMask + FieldPositionTable types | 80 | Yes |
| 3 | **3D** | Sonnet | D-CLS-7 (ontology side) | Per-class position tables + audit | 170 | Yes (after 3A+3B start, can run concurrently from same baton) |
| 3 | **3E** | Sonnet | D-CLS-7 (lookup glue) | ClassId → (DolceCategory, FieldPositionTable) helper | 50 | Yes |
| 4 | **4A** | Sonnet | D-CLS-8 (templates) | ~10-15 askama templates per shape-family | 350 | Yes |
| 4 | **4B** | Sonnet | D-CLS-8 (dispatch) | render() + template-derive structs | 160 | Yes |
| 4 | **4C** | Sonnet | D-CLS-8 (tests) | Per-class unit tests | 80 | Yes |
| 5 | **5A** | **Opus** | D-CLS-9 | Integration test + mask-density audit | 2,310 | No (final synthesis) |

**Totals:** 11 agent runs (10 working agents + 1 spec-owner gate). 2 Opus, 9 Sonnet. ~4,230 LOC across 1 new crate + 3 modified crates. Bulk LOC is generated content (66 snapshot files in D-CLS-9 + 15 askama templates in D-CLS-8).

### Inter-wave dependency DAG (the baton chain)

```text
                   [OD pre-conditions ratified]
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
    1A (DOLCE)            1B (signatures)        1C (render-crate)
       │                      │                      │
       └──────────┬───────────┴──────────────────────┘
                  ▼
            2A (Aerial+ on signatures)
                  │
                  ▼
         [spec owner names ~10-15 clusters → CANONICAL_CLASS_TABLE]
                  │
       ┌──────────┼──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼          ▼
       3A         3B         3C         3D         3E
    ClassId   class_id    FieldMask  pos tables  lookup glue
       │          │          │          │          │
       └──────────┴──────────┴──────────┴──────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
       4A (templates)      4B (render)         4C (tests)
       │                      │                      │
       └──────────┬───────────┴──────────────────────┘
                  ▼
            5A (integration + audit)
                  │
                  ▼
            [READY FOR REVIEW / MERGE]
```

### A2A coordination protocol (Layer-2 blackboard)

Per CLAUDE.md `Layer-2 — Session A2A`, `.claude/board/AGENT_LOG.md` is the blackboard. Every agent's prompt MUST include this preamble:

```text
You are agent <ID> of multi-agent plan `odoo-classes-bitmask-render-v1`.
MANDATORY READS (in order, before any output):
1. .claude/board/AGENT_LOG.md (top 50 lines — what just shipped)
2. .claude/plans/odoo-classes-bitmask-render-v1.md §<your D-CLS row>
3. .claude/specs/cognitive-risc-classes.md (§"Jinja = classes + presence bitmask")
4. [your D-CLS dependency outputs — the specific files prior agents wrote]

Your D-CLS: <D-CLS-X>
Your files (exact paths): [from §4]
Your tests gate: [from §4]
HARD CONSTRAINTS: C1-C7 from plan §1. Honour all 7 or your output is rejected.

AFTER committing your work:
1. Prepend an AGENT_LOG entry in this format:
   ## [<Wave><Agent ID> / Sonnet|Opus] D-CLS-X — <one-liner>
   **Branch:** <branch> | **Files:** <list>
   **Cargo:** <result>
   **Outcome:** DONE/BLOCKED + one-paragraph what + what's next
2. Do NOT modify any other D-CLS-X row's files (cross-lane discipline).
3. If you discover a hard constraint violation in your input, BLOCK and surface
   it to AGENT_LOG instead of attempting a fix.
```

### Per-wave gate (cannot start next wave until prior wave passes)

- **After Wave 1:** all three agents committed; `cargo check -p lance-graph-contract -p lance-graph-ontology` green; D-CLS-2 PSV present in knowledge dir.
- **After Wave 2:** spec owner has reviewed `.claude/knowledge/odoo-66-shape-family-candidates.md` and added a `## RATIFIED:` section naming the ~10-15 shape-families. AGENT_LOG entry from spec owner (manual; not an agent task).
- **After Wave 3:** all 5 agents committed; `cargo test -p lance-graph-contract` green (incl. ClassId, FieldMask tests); `cargo test -p lance-graph-ontology` green (incl. class-audit); per-class field-width audit reports zero violations.
- **After Wave 4:** all 3 agents committed; `cargo test -p lance-graph-ontology-render` green (per-class smoke tests).
- **After Wave 5:** integration test green; mask-density audit emitted to `.claude/knowledge/`; PR ready for review.

---

## 6. File-level ownership matrix (no two agents touch the same file)

| File | Agent | Operation |
|---|---|---|
| `crates/lance-graph-contract/src/cognition/entity.rs` | 1A + 3A | 1A adds `DolceCategory::Unknown` (if absent); **3A** adds `ClassId(u16)` (different region of same file) |
| `crates/lance-graph-ontology/src/hydrators/dolce_odoo.rs` | 1A | re-export + deprecation |
| `crates/lance-graph-arm-discovery/src/aerial/ontology.rs` | 1A | local newtype + TryFrom |
| `crates/lance-graph-callcenter/src/super_domain.rs` | 1A | `From<DolceCategory>` |
| `.claude/knowledge/odoo-66-structural-signatures.psv` | 1B | NEW |
| `.claude/knowledge/odoo-66-structural-signatures-method.md` | 1B | NEW |
| `crates/lance-graph-ontology-render/Cargo.toml` | 1C | NEW |
| `crates/lance-graph-ontology-render/src/lib.rs` | 1C (skeleton) + 4B (extend) | sequenced |
| `crates/lance-graph-ontology-render/templates/.gitkeep` | 1C | NEW |
| `Cargo.toml` (workspace root) | 1C | edit `exclude` list |
| `crates/lance-graph-arm-discovery/examples/odoo_66_class_discovery.rs` | 2A | NEW |
| `.claude/knowledge/odoo-66-shape-family-candidates.md` | 2A | NEW |
| `crates/lance-graph-ontology/src/odoo_blueprint/mod.rs` | 3B | add `class_id` field |
| `crates/lance-graph-ontology/src/odoo_blueprint/l{1..15}.rs` | 3B | back-fill 66 consts |
| `crates/lance-graph-ontology/src/odoo_blueprint/CLASS_TABLE.rs` | 3B (skeleton) + 3D (extend) | sequenced |
| `crates/lance-graph-contract/src/cognition/field_mask.rs` | 3C | NEW |
| `crates/lance-graph-contract/src/cognition/mod.rs` | 3C | add `pub mod field_mask;` |
| `crates/lance-graph-ontology/src/odoo_blueprint/class_audit.rs` | 3D | NEW |
| `crates/lance-graph-ontology/src/odoo_blueprint/class_lookup.rs` | 3E | NEW (~50 LOC) |
| `crates/lance-graph-ontology-render/templates/<shape_family>.txt.j2` × ~12 | 4A | NEW |
| `crates/lance-graph-ontology-render/src/templates.rs` | 4B | NEW |
| `crates/lance-graph-ontology-render/src/lib.rs` (extend) | 4B | dispatch + render + RenderErr |
| `crates/lance-graph-ontology-render/tests/per_class_smoke_<shape_family>.rs` × ~12 | 4C | NEW |
| `crates/lance-graph-ontology-render/tests/integration_66_entities.rs` | 5A | NEW |
| `crates/lance-graph-ontology-render/tests/snapshots/*.snap` × 66 | 5A | NEW (generated) |
| `crates/lance-graph-ontology-render/src/audit.rs` | 5A | NEW |
| `.claude/knowledge/odoo-66-mask-density-report.md` | 5A | NEW (test-generated) |
| `.claude/board/AGENT_LOG.md` | every agent | PREPEND only |
| `.claude/board/STATUS_BOARD.md` | every agent at PR-merge | row status flip Queued→In-progress→Shipped |

**Two files (`entity.rs`, `lib.rs`, `CLASS_TABLE.rs`) have multi-agent edits in different waves** — sequenced, never parallel on the same file. Per-line conflicts impossible by construction (1A's region is `DolceCategory`; 3A's region is `ClassId` — different parts of same file).

---

## 7. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Wave-2 Aerial+ returns >20 clusters → too fragmented to manage | Med | Med | Spec owner can merge clusters; the structural-hash is repeatable, so re-runs with different `max_antecedent` parameters explore the cluster-coarseness axis |
| Wave-2 returns <5 clusters → too coarse, distinct entities collide | Low | Med | Same axis: re-run with stricter `min_confidence` to force differentiation |
| A class's field count exceeds 64 → FieldMask(u64) overflow | Med | High | D-CLS-7 audit catches this AT TEST TIME and fails the build; explicit decision required (split class, widen to u128, or sub-classes) |
| The spec owner doesn't ratify the Wave-2 cluster names | Block | Block | Wave 3 blocked indefinitely; no auto-fallback. Plan explicitly names this as a manual gate (not "automatically continue with provisional names") |
| `askama` template engine choice rejected (OD-TEMPLATE-ENGINE flipped to minijinja or a third option) | Low | High | Re-do D-CLS-4 + D-CLS-8 dispatch + templates. Everything else (D-CLS-1, 2, 3, 5, 6, 7, 9) is engine-agnostic and unaffected |
| Aerial+ structural-hash is noisy on small N=66 input | Med | Low | Spec doc warned this at #438: "not measurable on 10" (wikidata-hhtl-load.md:85). N=66 may be borderline. Fall back to deterministic group-by on structural-hash if Aerial+ is unstable |
| D-CLS-1 deprecation triggers downstream warnings in n8n-rs / crewai-rust | Med | Low | Deprecations are warnings, not errors; downstream has one release cycle to migrate. Document the migration path in the deprecation message itself |
| Cross-file edits (entity.rs, lib.rs, CLASS_TABLE.rs) collide between 1A+3A or 1C+4B | Low | Med | Sequencing per the DAG eliminates this; agents in the same wave touch DIFFERENT files |
| C2 violation lands undetected (a bit is read as semantics) | Med | High | D-CLS-9's second test (mutant mask) is the canonical detector. Make the test mandatory for green CI |
| Workspace `cargo check` regresses due to D-CLS-4's new excluded crate not being checked in CI | Low | Low | Add `cargo test --manifest-path crates/lance-graph-ontology-render/Cargo.toml` to CI matrix (standalone crate pattern per bgz17/deepnsm) |
| The 4-way DOLCE consolidation breaks arm-discovery's zero-dep stance | Low | High | D-CLS-1 explicitly does NOT add `lance-graph-contract` as an arm-discovery dep; uses a local newtype + TryFrom. Sonnet must follow this precisely |

---

## 8. What this plan does NOT decide (escalations remain)

- OD-1/2/3 (`discovery_origin` byte width + tier set + Conjecture/Derived) — still SPEC-OWNER from yesterday. This plan does not touch `discovery_origin`.
- F1/F2/F4 from classes.md v0.2 — F1 settled, F2 default-federate, F4 universal column ISA design — F4 is the load-bearing unsolved piece *for the SoA*, but this plan only adds `class_id` to the const catalogue, not to the SoA row. F4 is downstream.
- The chess bring-up (D-CHESS-BRINGUP-1, queued). Per R1+R2 from yesterday's council it deserves its own branch + freeze-decision authority. This plan does not implement chess; it just doesn't *prevent* chess (per C1 — no WAL freeze).
- The hot-path mailbox-SoA `class_id` + `field_mask` columns. Different plan. This plan touches only the const catalogue + render path.

---

## 9. Cross-refs

- `cognitive-risc-classes.md` v0.2 §"Jinja = classes + presence bitmask" + §"NON-DEFERRABLE freeze-time moves" — the spec.
- `cognitive-risc-core.md` v0.1 §"discovery_origin" — parallel byte-grammar work, NOT in this plan's scope.
- `wikidata-hhtl-load.md` §"Facets: OWL/DOLCE as the template" — the OWL→facet mapping table that informs D-CLS-3's signature design.
- `faiss-homology-cam-pq.md` §"Why CAM_PQ is cheaper than float PQ" — confirms `facet u64 IS the SoA facet column → SIMD batch-AND`; informs the FieldMask u64 choice in D-CLS-7.
- `.claude/knowledge/discovery-origin-provenance-reconciliation-v1.md` — the reconciliation work that surfaced the 4-way DOLCE duplication this plan fixes in D-CLS-1.
- `.claude/plans/post-438-integration-options-v1.md` §1 Option G — chess bring-up (the N4 falsifier this plan honours by not freezing the SoA schema).
- `.claude/board/EPIPHANIES.md` `E-DISCOVERY-ORIGIN-HOME-IS-ARIGRAPH-BRIDGE` — the structurally adjacent insight; this plan's `FieldMask`/`class_id` IS the same shape but at the catalogue layer.
- `.claude/board/EPIPHANIES.md` `E-TEMPLATE-IS-CHECKLIST-IS-DATOMS` + `E-RELIABILITY-IS-CHECKLIST-COVERAGE` (parallel session, #439) — the `ThoughtMask` shape this plan generalises to `FieldMask`.
- `crates/lance-graph-arm-discovery/src/aerial/` (#436) — the existing crate D-CLS-3 calls; no modification.
- `crates/lance-graph-contract/src/recipe_kernels.rs:137` (#439) — the existing `ThoughtMask(u8)` this plan's `FieldMask(u64)` is the structural sibling of.
- 4 `DolceCategory` definitions catalogued in the prior session's bitmask-state inventory; D-CLS-1 fixes the duplication.

End of plan.
