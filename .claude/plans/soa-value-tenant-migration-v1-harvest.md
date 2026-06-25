# SoA Value-Tenant Migration — Phase-1 Harvest Result (the filled §4 inventory)

> **Status:** HARVEST (2026-06-25, cont.⁴³). The Phase-1 deliverable the brief
> (`soa-value-tenant-migration-v1.md`) commissioned: the §4 inventory, filled by
> READ (not grep-to-conclude), plus the two findings the read surfaced and the
> operator's contained-facet closure. **NOT the migration** — §5 is still written
> for real only after the two 5+3 panels (§6) sign off on this.
>
> **Provenance / honesty (per the brief's read-discipline):**
> - **Confirmed-by-read (main thread):** `lance-graph-contract/src/canonical_node.rs`
>   lines 1–1091 (the entire `ValueTenant`/`VALUE_TENANTS`/`ValueSchema`/`ReadMode`
>   definitional surface — the remaining 1092–1866 is `KanbanTenant::to_bytes` +
>   the test module; the `const _` assert `ValueSchema::Full.field_mask().count()
>   == VALUE_TENANTS.len()` *proves* there is no 11th tenant hiding in the tests);
>   `class_view.rs` (full); `cascade_key.rs` (full); `AGENT_LOG.md` head (cont.³⁸–⁴²);
>   `INTEGRATION_PLANS.md` head.
> - **Subagent-mapped (confirmed-by-read by the agent):** in-workspace + ndarray
>   producer/consumer map (Opus general-purpose, `tasks/a820786a55a562a0f`);
>   cross-repo consumer locator (Explore).
> - **Not read this session (named, not hidden):** `place_buffer.rs`/`columns.rs`
>   (taken from AGENT_LOG cont.⁴⁰ + the in-workspace agent's full read);
>   `MedCare-rs` + `OGAR` (a casing-miss in the cross-repo sweep — see §6);
>   the dense body of `LATEST_STATE.md` (grep-located only, board_status below).

---

## 1. Two headline findings (the read changed the question)

### Finding A — there are TWO disjoint "SoA" worlds `[G]`

The migration's object is not one structure but the *seam between two*:

- **(A) the canonical `NodeRow.value` 480-byte slab** — the 10 `ValueTenant`s,
  carved by `VALUE_TENANTS` (`canonical_node.rs:606`), addressed via
  `ValueTenant::value_offset()`.
- **(B) a parallel `MailboxSoA<N>`** (`cognitive-shader-driver/src/mailbox_soa.rs`)
  — separate `[T; N]` / `Vec<T>` columns (energy / qualia / meta / entity_type /
  edges) that implement the SAME `MailboxSoaView`/`MailboxSoaOwner` traits
  (`contract::soa_view.rs`) but **do NOT touch the slab or `ValueTenant`**.

The two worlds share exactly **one** semantic column: `MailboxSoaView::class_id()`
aliases `entity_type()` (`soa_view.rs:75`), which is `ValueTenant::EntityType`.
`SymbiontBoard` (`symbiont/src/kanban_loop.rs:137`) straddles both — it *carries*
`Vec<NodeRow>` (the slab) but exposes tenants through **parallel mirror `Vec`s**
(`self.energy/edges/meta/entity`), reading the slab head only for
`edge_block_at`/`hhtl_path_at`. **The near-term migration is reconciling A and B**,
not homogenizing — and the choice of which becomes canonical is the load-bearing
design decision the §6 panels must arbitrate.

**Live-producer reality (decisive for migration_class):** only **4 of 10** slab
tenants have a live (non-test) slab producer — **Energy, EntityType, Kanban,
Fingerprint**. The other **6 — Meta, MaterializedEdges, HelixResidue,
TurbovecResidue, Plasticity** — exist only as schema-membership tests (`ocr.rs`)
or as parallel-`MailboxSoA` mirrors; `MaterializedEdges`/`HelixResidue`/
`TurbovecResidue` are explicitly **deferred**. A producer-less tenant migrates
very differently from an actively-written one.

### Finding B — homogeneity does NOT close over the slab; it closes as ONE contained facet `[H]`

The §8.5 KILL gate asked: are the value facets homogeneous, or irreducibly
heterogeneous? Applying §3's gates to the *actual* 10 tenants: **heterogeneous.**
They are identity (Fingerprint), scalars (Energy / Plasticity / EntityType — the
§3 named conflation trap), a bitfield (Meta), a fixed 16×i4 vector (Qualia),
already-PQ codes (Turbovec), a structured cursor (Kanban), and deferred edges.
**Only HelixResidue (the structure/place axis) matches the facet shape.** So §8
reduces — per its own §8.5 fallback, *not* a failure — to "**classid is a schema
pointer**", which is **shipped and real**: `ReadMode { value_schema, edge_codec }`
resolved by `classid_read_mode()` through `BUILTIN_READ_MODES`
(`canonical_node.rs:891`), with the cleanest end-to-end exemplar at
`ocr.rs:105 to_node_row` (reads `classid_read_mode(classid).value_schema`, then
`schema.has(tenant)`-gates every write).

**But the closure exists — as one CONTAINED special case (operator, 2026-06-25):**

```text
contained place⊕search facet = 16 B, layout-preserving, facet_classid-selected codec
  facet_classid (4 B)  ← the schema pointer (which ClassView reads these 12 bytes)
  helix-place   (6 B)  ← 48-bit Signed360 place⊕residue   = ValueTenant::HelixResidue
  cam-pq        (6 B)  ← 48-bit canonical CAM-PQ 6×256 code  (the 6 B CAM-PQ, NOT turbovec)
```

This is the §8.1 facet (`facet_classid(4) | 12`) made concrete with a **place⊕search
codec** instead of a `6×(8:8 part_of:is_a)` tile — both are 12-byte payloads under
a 4-byte `facet_classid`; the difference is the codec, and **`facet_classid`
discriminates the codec** (the §3 / §8.1 provision, so this is a `classid →
ClassView` *reading*, never a new `ValueSchema` variant → no #500 violation,
layout-preserving). It fuses the three codes that genuinely belong together —
**identity (helix place, the frozen ruler, ICC→1.0) ⊥ search (CAM-PQ, the §3
"scalars → PQ-code facet" target) ⊥ schema (facet_classid)** — into one
self-describing 16-byte unit, the same width as the key and the EdgeBlock.

It is **I-VSA-IDENTITIES-clean by construction**: helix and CAM-PQ occupy
*disjoint* byte ranges ([0:6] and [6:12]) — concatenated, never XOR-bundled — so
the iron rule "CAM-PQ is for search, kept separate from the identity/bundling
layer" is enforced by the carve itself.

**The one precise design point (do not gloss):** the facet wants the **6 B
canonical CAM-PQ** (`canonical_node` codec atlas "CAM-PQ (6B, varies)";
OGAR-path-aligned 6×256), which is **NOT** today's `ValueTenant::TurbovecResidue`
(16 B turbovec 32×4-bit). So the contained facet is *not* "HelixResidue(6) +
TurbovecResidue(16)" (= 22 B); it consolidates **HelixResidue + a 6 B CAM-PQ
reading + classid → 16 B**, and the migration must decide whether the 16 B
turbovec stays out-of-facet (Full-preset only) or is replaced by the 6 B CAM-PQ in
the Compressed/cold-reference domain. Grade **`[H]`**, gated **F-1** (4⁴-vs-flat
codebook fidelity) **+ F-code** (lossless containment) — the shape is canon-sound;
the fidelity is unproven.

---

## 2. The filled §4 inventory (10 rows)

Slab offset = full-row `row_offset` − 32. `layout` is **preserving** for all
(every preset/facet carves within the reserved 480 B slab; `const _`-asserted at
`canonical_node.rs:795`, `is_layout_preserving()==true`). `producers`/`consumers`
cite the in-workspace map (Agent-confirmed-by-read) + the cross-repo sweep.

| # | tenant | def_site | slab off / B | axis | codec | producers (slab) | consumers | migration_class | conflation_risk | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **Meta** | `canonical_node.rs:554` | 0 / 8 | truth/awareness | `MetaWord` u64 bitfield | none (parallel-SoA `mailbox_soa.rs:583 set_meta`) | `mailbox_soa.rs:577 meta_at` | **KEEP** (flat bitfield, not hierarchical) | low | none |
| 1 | **Qualia** | `:556` | 8 / 8 | angle | `QualiaI4_16D` 16×i4 | none (test `:1205`; parallel `mailbox_soa.rs:571 set_qualia`) | slab `:1125 NodeRow::qualia()`→`:1149 mul_phase_step`; `sigma-tier-router:365` | **DEFER** (i4-16D — bigger substrate-validation test, §2a) | low | substrate-validation |
| 2 | **MaterializedEdges** | `:558` | 16 / 32 | composition | 4× `CausalEdge64` | none — **deferred** (`mailbox_scan.rs:39`) | none | **KEEP / deferred** (reconcile vs `MailboxSoA.edges`) | n/a | none |
| 3 | **Fingerprint** | `:560` | 48 / 32 | identity | `Fingerprint<256>` 256-bit print | **live:** `symbiont/domino.rs:69 write_lanes` | `domino.rs:61 read_lanes` (AMX GEMM) | **KEEP** (the identity ruler — homogenizing identity is nonsensical, thesis §3) | none | none |
| 4 | **HelixResidue** | `:564` | 80 / 6 | **structure** | helix `Signed360` 48-bit place⊕residue | none (prototype `place_buffer.rs:49 helix_place`) | none live; `ocr.rs` Compressed-schema tests | **homogenize-to-facet** → the contained 16 B facet (Finding B) | low (place⊕residue already orthogonal, cont.⁴⁰) | **F-1 + F-code** |
| 5 | **TurbovecResidue** | `:566` | 86 / 16 | codec/search | turbovec PQ 32×4-bit (128-bit) | none — deferred | none live; `ocr.rs` tests | **PQ-already** — but the contained facet wants the **6 B CAM-PQ**, not this 16 B turbovec (§1 design point) | none (already a PQ code) | F-1 (if activated) |
| 6 | **Energy** | `:568` | 102 / 4 | dynamics | `f32` accumulator (scalar) | **live:** `symbiont/bridge.rs:38 set_energy`, `domino.rs:77`, `ocr.rs:113` | `bridge.rs:46`, `domino.rs:81`, `nan_projection.rs` | **KEEP** (scalar — forcing into 8:8 is the §3 split-error in reverse) | **HIGH if homogenized** | none (explicitly NOT homogenize) |
| 7 | **Plasticity** | `:570` | 106 / 4 | dynamics | u32 counter + stamp | none (parallel `mailbox_soa.rs:336`) | `ocr.rs:262` test; `mailbox_soa.rs:517` | **KEEP** (scalar pair) | high-if-homogenized | none |
| 8 | **EntityType** | `:572` | 110 / 2 | schema/identity | u16 OGIT discriminator | **live:** `ocr.rs:110` | `ocr.rs:183`; **`soa_view.rs:75 class_id()` alias** → `mailbox_scan.rs:78 match_nodes_by_class` (Cypher `MATCH (n:Label)`) | **KEEP** (the class discriminator = the ClassView key; the A↔B reconciliation anchor) | none | none |
| 9 | **Kanban** | `:578` | 112 / 8 | lifecycle | `KanbanTenant` phase+exec+cycle | **live:** `canonical_node.rs:1114 set_kanban` (owner-only) | `:1103 kanban()`→`:1150 mul_phase_step`; `tenant_counter.rs:16` | **KEEP** (structured cursor, owner-only-write exemplar) | none | none |

**Net:** **8 KEEP, 2 DEFER (Qualia i4-16D + the future thinking-style i4-32D), 1 homogenize-to-facet (HelixResidue, into the contained 16 B
place⊕search facet with a 6 B CAM-PQ).** This IS the honest §8.5 outcome:
homogeneity does not close over the slab; it closes as the operator's one
contained facet for the cold place⊕search (Compressed / FMA-anatomy) workload,
while the hot heterogeneous tenants stay `KEEP` (the Cognitive/Full presets) and the
i4-quantized cognitive vectors (Qualia, thinking-style) **DEFER** (§2a).

### 2a. Deferred for future substrate validation (operator, 2026-06-25)

**Qualia (i4-16D, tenant #1) and the future thinking-style (i4-32D) DEFER — not
`KEEP`, not homogenize.** Both are i4-quantized (signed-4-bit, 16 levels) cognitive
vectors; whether i4 *faithfully* encodes a 16-D qualia / 32-D thinking-style
geometry is a **bigger substrate-validation test** than the contained facet's
F-1/F-code — it is the thinking-engine calibration question (Cronbach α / Spearman
ρ / ICC vs the f32 ground truth, with Jirak-not-classical significance bounds per
`I-NOISE-FLOOR-JIRAK`, since the channels are weakly dependent). So they are
**parked** for a dedicated substrate-validation pass, not migrated this round. The
thinking-style **i4-32D** is the operator-named *companion future tenant* to Qualia
(NOT one of the 10 today — `Meta` carries the MetaWord bits; the i4-32D style vector
is carried alongside Qualia when validated). This keeps the migration's first cut to
what is provable now (the contained facet + the A↔B reconciliation) and quarantines
the heavier measurement question.

`board_status` (all 10): the canon is recorded in `LATEST_STATE.md` Contract
Inventory at the **type level** (`canonical_node` / `NodeRow` / `ValueSchema` /
`ReadMode` — ~20 grep hits), **not per-tenant**. This filled table is the first
per-tenant enumeration → a board-hygiene addition (recorded in the INTEGRATION_PLANS
prepend, not a drift).

---

## 3. The carriers — the "classid → ClassView reading" mechanism is SHIPPED `[G]`

The migration's §5.1 ("ClassView reading over an existing preset, no new variant")
is not aspirational — it is built and tested:

- **`ValueSchema`** (`canonical_node.rs:715`): `Bootstrap`(∅) / `Cognitive`(7
  tenants) / `Compressed`(4) / `Full`(10). `is_layout_preserving()==true`,
  `const _`-asserted. *(Doc-drift noted: `Cognitive`'s doc-comment lists 6 tenants
  but `field_mask` adds `Kanban` = 7 — `:719` vs `:734`.)*
- **`ReadMode` + `classid_read_mode()`** (`:815`,`:912`): the `{value_schema,
  edge_codec}` pair, resolved through `BUILTIN_READ_MODES` (DEFAULT/OSINT/FMA/
  PROJECT/ERP), zero-fallback to `ReadMode::DEFAULT`. **Live debt:**
  `ReadMode::DEFAULT.value_schema = Full` is a **TEMPORARY 2026-06-15 POC**
  (`:826`) paired with `ClassView::value_schema`'s `Full` default (`class_view.rs:395`)
  and the revert test `value_schema_default_is_full_temporary_poc` — the migration
  must flip both to `Bootstrap` together (one revert, two sites + test).
- **`ClassView`** (`class_view.rs:305`): the resolver-above-the-SoA; owns
  `value_schema()` + `edge_codec_flavor()` (selection-only, "never a stride
  change") + `compute_dag()`. **The contained facet (Finding B) lands here as a
  new `ClassView` reading method in the same mold** — additive, layout-preserving.

---

## 4. The perturbation-sim prototypes — relationship clarified `[G]`

The brief §2.3/§2.4 named `cascade_key`/`place_buffer`/`INERTIA_SLOT` as "prototype
tenants." The read corrects this: **they are a SEPARATE, zero-dep SoA that is NOT
wired into `NodeRow.value`** (`columns.rs` header, verbatim: "*nothing here
serializes or touches the operator-locked canonical_node spine*").

- **`cascade_key.rs` V3** `CascadeKeyV3` (the `(part_of:is_a)` 8:8 tile,
  `:226`,`tier_v3:238`) produces **HEEL/HIP/TWIG = KEY bytes** (GUID offsets
  4..10), **not value tenants**. The `v3_two_hierarchies_are_independent` test
  (`:534`) proves part_of ⊥ is_a. It is the prototype the §8.1 tile-codec
  generalizes — but it lives in the *key cascade*, not the value slab.
- **`place_buffer.rs`**: `helix_place` (LOCATION, deterministic √u golden-spiral,
  never reads the grid, ICC 1.00) ⊥ `BufferResidue` (8×BF16 impulse-permeability,
  ICC 0.51). Maps *conceptually* onto `ValueTenant::HelixResidue` (the 6 B place
  octets of the contained facet) + a future buffer tenant — but never imports
  `lance_graph_contract`.
- **`columns.rs` `SoaMemberSpec`** is a **calibration spec, not a runtime
  encoder**; `INERTIA_SLOT=5` + `INERTIA_PROMOTION` = the §0-guardrail
  `RatifiedReuse` verdict (operator sign-off 2026-06-16: reuses the helix-residue
  slot, invents no tenant).

The only live `NodeRow.value` ↔ perturbation bridge is `symbiont/bridge.rs` +
`domino.rs`, which write only `Energy` (+ `Fingerprint` for the AMX tile).

---

## 5. Cross-repo consumer blast radius `[G]` (read-side)

- **q2 — CLEAR.** `osint-bake` / `cockpit-server` call `NodeGuid::new_v2(...)`
  (`osint-bake/src/lib.rs:606,745`). **The brief's §2.5 `new_v2` blocker is
  CLOSED:** `new_v2` now exists (`canonical_node.rs:244`, `#[cfg(feature =
  "guid-v2-tail")]`, the D-GV2-1 7-arg v2 tail) AND q2 enables `guid-v2-tail` at
  the workspace root → correctly gated, not an `I-LEGACY-API-FEATURE-GATED`
  violation. The brief's "a 7-group API that does NOT exist" framing is stale.
- **woa-rs / openproject-nexgen-rs / redmine-rs — CLEAR.** Pull via
  `ogar_vocab::canonical_concept_id`, no `*Bridge` construction outside the
  allow-list, no codebook copy. woa-rs owns its 16 B layout dep-free
  (`src/erp/canon.rs`).
- **odoo-rs — zero hits.**
- **BBB-barrier audit: PASS** across all swept consumers.

---

## 6. Completeness note — named gaps (not hidden)

1. **`MedCare-rs` + `OGAR` not swept — casing miss.** The cross-repo agent
   searched `/home/user/medcare-rs` and `/home/user/ogar`; the clones are
   **`/home/user/MedCare-rs`** and **`/home/user/OGAR`** (case-sensitive FS). Its
   "disk-walled" verdict for those two is **wrong** — both are available. `OGAR`
   is where "a minted class's read-mode is layered in one level up"
   (`canonical_node.rs:888`) — i.e. the producer side of `classid → ReadMode`
   specialization. `MedCare-rs` is a named BBB consumer. **Top follow-up: a
   corrective sweep of `/home/user/{OGAR,MedCare-rs}`** before §5 is written.
2. **6 producer-less tenants** (Meta / MaterializedEdges / HelixResidue /
   TurbovecResidue / Plasticity) — a real finding, not a coverage gap: their
   live writes (where any) happen in the parallel `MailboxSoA`, not the slab.
3. **`place_buffer.rs` / `columns.rs`** taken from AGENT_LOG cont.⁴⁰ + the
   in-workspace agent's full read, not a main-thread read this session.
4. **`(located, not read)`** by the in-workspace agent: `crates/learning/*`,
   `lance-graph-cognitive/src/grammar/*`, `surreal_container/*` (the kv-lance
   `MailboxSoaView` impl, DESIGN-LOCKED/deferred) — likely parallel-SoA
   consumers, lower confidence.
5. **Open design question (un-answerable from source):** does the migration
   intend `MailboxSoA.edges` to *become* the slab `MaterializedEdges` tenant?
   This is the crux of the A↔B reconciliation (Finding A).
6. **`LATEST_STATE.md` body** is grep-located only (the file exceeds the
   single-read token cap); board_status established at the canon-type level.

---

## 7. Hand-off to Phase 2 (the two independent 5+3 panels)

The filled inventory (§2) + the two findings (§1) + the carriers (§3) are the
input the §6 sign-off panels harden. The seams to scrutinize hardest:

1. **The contained facet (Finding B).** `truth-architect` / `container-architect`:
   is the 6 B-CAM-PQ-vs-16 B-turbovec width decision sound, and does F-1/F-code
   actually gate it? `iron-rule-savant`: confirm helix∥CAM-PQ disjoint-range
   concatenation respects I-VSA-IDENTITIES (no superposition).
2. **The A↔B reconciliation (Finding A).** `dto-soa-savant`: does collapsing the
   parallel `MailboxSoA` into the slab (or vice-versa) land as a column, not a new
   layer? `baton-handoff-auditor`: the `class_id()≡entity_type()` alias is the
   only shared column — does the cross-crate roundtrip survive?
3. **The POC default.** `overclaim-auditor`: the `value_schema = Full` POC
   (two sites + test) is unreverted — any inventory row that assumes `Bootstrap`
   semantics is premature.
4. **`firewall-warden` / `dilution-collapse-sentinel`:** does folding HelixResidue
   into the contained facet *dilute* a sharp tenant or *collapse* a distinct one?
   (The harvest's read: no — it *consolidates* the place⊕search pair the operator
   identified, and the hot heterogeneous tenants stay KEEP.)

**Reconciliation rule (brief §6):** run the two panels independently; convergence
is signal, divergence names the seam. No tenant migrates on one panel's say-so.

---

## Cross-references
- `soa-value-tenant-migration-v1.md` (the brief: §2 where-to-read, §3 gates, §4
  schema, §6 5+3).
- `substrate-unification-thesis.md` §8.1 (homogeneous facet), §8.5
  (homogeneity-non-closure — the KILL gate this harvest answers).
- `canonical_node.rs` (`ValueTenant:552` / `VALUE_TENANTS:606` / `ValueSchema:715`
  / `ReadMode:815` / `classid_read_mode:912`; `new_v2:244` gated).
- `class_view.rs` (`ClassView:305`, `value_schema:395` POC default).
- `perturbation-sim/src/{cascade_key.rs, place_buffer.rs, columns.rs}` (key-side
  prototypes, separate SoA).
- Iron rules: `I-VSA-IDENTITIES` (the contained facet's disjoint-range carve),
  `I-LEGACY-API-FEATURE-GATED` (the q2 `new_v2` gate, now closed),
  `I-SUBSTRATE-MARKOV`.
- AGENT_LOG cont.⁴⁰ (location⊥permeability split), cont.⁴² (§8 thesis, PR #610).
