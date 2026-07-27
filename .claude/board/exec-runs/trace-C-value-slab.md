# Trace C — value-slab[480] census (belief-tenant candidacy)

Depth: full read of `canonical_node.rs` (3123 lines, two passes across
offsets 1-800/800-1600/1600-2400/2400-2800), `awareness_facet.rs` (207
lines, full), `.claude/v3/soa_layout/le-contract.md` (379 lines, full),
plus targeted greps into `ocr.rs`, `nan_projection.rs`, `soa_view.rs`,
`nars.rs`, `causal_audit.rs`, `settlement.rs`, `graph_render.rs`,
`soa_graph.rs`, `crates/lance-graph/src/graph/mailbox_scan.rs`.

## Carve map

`NodeRow` (`canonical_node.rs:726-730`): `key(16) | edges(16) | value(480)`,
`#[repr(C, align(64))]`, size asserted 512 (`canonical_node.rs:733-735`).
`VALUE_SLAB_ROW_OFFSET = 32`, `VALUE_SLAB_LEN = 480` (`:814-817`).

`VALUE_TENANTS: &[ColumnDescriptor]` (`:917-1016`) — the ONLY authoritative
carve of the 480-byte slab. Contiguous, discriminant-ordered, compile-time
asserted (`:1018-1044`: discriminant==index, no gaps, fits 512, fits 480).
Current occupied span: value-slab bytes `[0, 172)` of 480 — i.e. row bytes
`[32, 204)`. Everything from row byte 204 to 512 (276 B) is **unclaimed and
un-described by any `ColumnDescriptor`** — reserved by omission, not by an
explicit "reserved" entry.

| `ValueTenant` (discriminant) | row_offset | bytes | kind | Accessor on `NodeRow`? | Notes |
|---|---|---|---|---|---|
| `Meta` (0) | 32 | 8 (`U64`, `:919-923`) | U64 | **None** — no `.meta()`/`.set_meta()` on `NodeRow`. `MetaWord` (the doc-comment's referent, `:829`) is `pub struct MetaWord(pub u32)` in `cognitive_shader.rs:44`, i.e. 4 bytes, not 8 — a known width mismatch, see le-contract.md discrepancy #4 below. |
| `Qualia` (1) | 40 | 8 (`U64`, `:924-929`) | U64 | **`row.qualia()`** (`:1647-1653`), decodes as `QualiaI4_16D(u64)`. No setter on `NodeRow` (qualia is written via `set_kanban`-style code in `ocr.rs`/callers directly into `row.value`, not through a `NodeRow::set_qualia`). |
| `MaterializedEdges` (2) | 48 | 32 (`U64`×4, `:930-935`) | U64 | **None.** Doc: "the 4 out-of-family edges materialised as full `CausalEdge64`" (`:833`) — no accessor found anywhere in the crate. |
| `Fingerprint` (3) | 80 | 32 (`U8`×32, `:936-941`) | U8 | **None.** 32-byte identity print — no `.fingerprint()` reader/writer on `NodeRow`. |
| `HelixResidue` (4) | 112 | 6 (`U8`×6, `:942-949`) | U8 | **None** on `NodeRow`. Referenced only via `.value_offset()`/`.has()` in `ocr.rs` tests. |
| `TurbovecResidue` (5) | 118 | 16 (`U8`×16, `:950-955`) | U8 | **None** on `NodeRow`. |
| `Energy` (6) | 134 | 4 (`F32`, `:956-961`) | F32 | **None** on `NodeRow` directly, but `nan_projection.rs:53,95` reads/writes it via `ValueTenant::Energy.value_offset()` + raw slice indexing (a *consumer-side* accessor, not a `NodeRow` method). `ocr.rs:112-113,198` likewise. |
| `Plasticity` (7) | 138 | 4 (`U32`, `:962-967`) | U32 | **None** on `NodeRow`. le-contract.md discrepancy #4 flags a persisted-vs-hot width mismatch here too (4 B/1 B). |
| `EntityType` (8) | 142 | 2 (`U16`, `:968-973`) | U16 | **None** on `NodeRow`; read/written via `value_offset()` in `ocr.rs:109,194,232`. |
| `Kanban` (9) | 144 | 8 (`U64`, `:974-981`) | U64 | **`row.kanban()` / `row.set_kanban()`** (`:1622-1641`) — the only tenant with a full typed read+write pair including a field-isolation test (`:1745-1771`). |
| `FrozenStyle` (10) | 152 | 12 (`U8`×12, `:987-992`) | U8 | **`row.style_lane(FrozenStyle)` / `.set_style_lane(...)` / `.triangle_for(f)`** (`:1676-1738`) — shared generic accessor across the three triangle tenants. |
| `LearnedStyle` (11) | 164 | 12 | U8 | same generic accessor. |
| `ExploreStyle` (12) | 176 | 12 | U8 | same generic accessor. |
| `Tekamolo` (13) | 188 | 16 (`U8`×16, `:1010-1015`) | U8 | **None** on `NodeRow`. Doc references `TekamoloFacet` in `crate::tekamolo_facet` (`:878-889`) and an extractor in `examples/insight_reason_wired.rs`, but no `row.tekamolo()` method exists in `canonical_node.rs` itself. |

`ValueTenant::value_offset()` / `::byte_len()` (`:893-909`) are the only
generic, tenant-agnostic accessors; every specific-typed accessor
(`kanban`, `qualia`, `style_lane`/`triangle_for`) is hand-written on top of
them. 5 of 14 tenants (`Meta`, `MaterializedEdges`, `Fingerprint`,
`HelixResidue`, `TurbovecResidue`, `Plasticity`, `EntityType`, `Tekamolo` —
8, not 5; corrected count: **8 of 14**) have no `NodeRow` method at all —
callers reach for `ValueTenant::X.value_offset()` + manual byte slicing.

`ValueSchema` presets (`:1058-1150`, byte budgets certified at
`:2277-2295`): `Bootstrap` (0 B), `Cognitive` (66 B: Meta+Qualia+
Fingerprint+Energy+Plasticity+EntityType+Kanban — NOT the triangle, NOT
Tekamolo), `Compressed` (56 B: Fingerprint+HelixResidue+TurbovecResidue+
EntityType), `Full` (172 B — every tenant, `:1094-1119`). `ReadMode::DEFAULT`
currently resolves every un-minted classid to `ValueSchema::Full` as a
**temporary POC** (`:1233-1249`, explicitly flagged for reversion to
`Bootstrap` later — "one revert, two sites").

## Tenant-addition rules

1. **Append-only, discriminant-ordered, no gaps** (`:1018-1044`, a
   `const _: ()` compile assertion). A new tenant is a new
   `ValueTenant` variant with the NEXT discriminant, appended to the END
   of `VALUE_TENANTS` at `prev_end` (the current end, byte 204 in the
   row / 172 in the slab) — never inserted mid-list, never reusing a
   retired discriminant ("RESERVE, DON'T RECLAIM", stated explicitly at
   `:821-825` and repeated in every tenant's addition comment, e.g.
   `:1005-1009` for Tekamolo: "Appended after the autopoiesis triangle
   (additive, reserve-don't-reclaim)").
2. **Must fit the 480-byte slab** — `Full.tenant_bytes() <= VALUE_SLAB_LEN`
   is compile-asserted (`:1154`); currently 172/480 used, 308 B headroom
   in the *slab*, but note the row-level unclaimed span quoted above is
   the SAME 308 B (480−172), since every existing tenant is also in
   `Full`.
3. **No `ENVELOPE_LAYOUT_VERSION` bump required** — every doc-comment for
   a new tenant explicitly certifies "NODE_ROW_STRIDE unchanged" (e.g.
   `:1007-1008`, `:854`, `:985-986`). This is because `NODE_ROW_STRIDE`
   (512) and `NODE_ROW_COLUMNS` (`:765-784`, the 3-column key/edges/value
   envelope-level table) never change — the value slab is carved
   *inside* one already-declared 480-byte `ColumnDescriptor`
   (`row_offset: 32, elems_per_row: 480`, `:778-783`), so a new
   `ValueTenant` is invisible to `SoaEnvelope::verify_layout()` /
   `ENVELOPE_LAYOUT_VERSION` entirely — that gate only watches the
   3-column envelope, not the tenant sub-carve.
4. **A preset (`ValueSchema`) decides who materialises the new tenant** —
   adding a `ValueTenant` variant does not itself grant it to any class;
   a `ValueSchema` variant's `field_mask()` (`:1076-1120`) must be
   updated to `.has()` it, and `Full` must cover every tenant
   (compile-asserted `:1155`: `Full.field_mask().count() == VALUE_TENANTS.len()`)
   — so a brand-new tenant is FORCED into `Full`'s mask by the compiler,
   but is opt-in for `Cognitive`/`Compressed`/`Bootstrap`.
5. **ClassView registration is the per-class selection mechanism**, not
   coded in `canonical_node.rs` itself — `ValueSchema` is "selected via
   `ClassView::value_schema`" (`:1049`), and `classid_read_mode()`
   (`:1430-1435`) resolves a `classid → ReadMode{tail_variant,
   value_schema, edge_codec}` through the `BUILTIN_READ_MODES`
   `LazyLock<HashMap>` registry (`:1382-1423`). Un-registered classids
   fall through to `ReadMode::DEFAULT` (the zero-fallback ladder, same
   pattern as `NodeGuid`'s classid/family fallback).
6. **A field-isolation matrix test is expected** (I-LEGACY-API-FEATURE-
   GATED, cited directly in the triangle test docstring at `:2408-2414`):
   flip each tenant's bytes, assert only its own byte range changed, key
   and edges untouched. The `FrozenStyle`/`LearnedStyle`/`ExploreStyle`
   test (`:2409-2519`) and the `Kanban` test (`:1745-1771`) are the two
   worked examples a new tenant's test should mirror.
7. **le-contract.md §3b "two-level LE contract"** adds an external,
   process-level gate beyond the byte-carve mechanics above: *"a
   consumer that starts reading a tenant lane owes a jc-pillar
   certification run (ICC, Spearman ρ, Cronbach α... real bytes,
   deterministic sampling, 4-decimal reporting) before its reading is
   trusted in any downstream claim"* (le-contract.md:303-311). This is a
   reading-validation gate, separate from the byte-layout gate — layout
   tests prove bytes don't move; jc pillars prove a reading preserves
   semantics.
8. **Slot-purity rule (le-contract.md §2, line 40-48):** "Labels and
   positions come from the ClassView — NEVER from a slot in the
   payload." A candidate tenant layout containing a name string, a
   label, or an ordinal/position field inside the 12-byte (or any)
   payload is explicitly called out as "a LAYOUT-BREAK-class defect —
   reject at review" (le-contract.md:48).

## EdgeBlock reality

`EdgeBlock` (`canonical_node.rs:645-650`): `in_family: [u8;12]`,
`out_family: [u8;4]`, `#[repr(C, align(16))]`, size-asserted 16
(`:734`). Doc: "Canonical, not mandatory: the 16 bytes are ALWAYS
reserved (zeroed when unused). A class never shrinks this block."

Production (non-test) readers/writers found:

- **`crates/lance-graph/src/graph/mailbox_scan.rs`** — `edge_slots_coarse<V: MailboxSoaView>()`
  (`:160-` region, doc at `:160-189`) is a real function, not test-only,
  that reads a node's `EdgeBlock` under `EdgeCodecFlavor::CoarseOnly` as
  "16 family-node adapter slots": 12 `in_family` (intra-basin) + 4
  `out_family` (cross-family interface) byte refs, each non-zero byte a
  basin-local edge reference, zero byte = empty slot. The module
  doc (`mailbox_scan.rs:27-38`) is explicit that **row resolution is
  deliberately deferred**: *"EdgeBlock slot-byte → neighbor-row
  resolution — needs the basin-local-index convention... This module
  lands the edge structure, never fakes the row resolution."* So today
  this reads which slots are populated and their raw byte refs, but does
  NOT resolve a slot byte to an actual neighboring `NodeRow`.
- **`crates/lance-graph-contract/src/soa_graph.rs`** — `project_snapshot()`
  (doc `:1-38`) reads the SAME 16-byte `EdgeBlock` under `CoarseOnly` as
  "16 family-node adapter slots (operator model, 2026-06-20)" to build a
  `graph_render::GraphSnapshot`/`RenderEdge` view for the Palantir-Gotham
  cockpit surface — a second, independent production consumer of the raw
  edge bytes, also stopping at "family" resolution (`family & 0xFF`),
  not per-instance node resolution.
- **`EdgeCodecFlavor`** (`:669-704`) names three read interpretations
  (`CoarseOnly` = default/canon, `CoarseResidue`, `Pq32x4`) of the SAME
  16 bytes (plus, for `CoarseResidue`, a residue borrowed from the value
  slab). Both production readers above explicitly refuse `CoarseResidue`/
  `Pq32x4` as adjacency (`E-ADJACENCY-IS-KEY-AND-EDGECODEC` boundary,
  `mailbox_scan.rs:29-31,185-189`) rather than coercing them.

Net: the 12+4 slots ARE used for real relations today, but only as far as
"which basin-local byte-ref slots are populated, split in-family vs
out-of-family" — no code in this repo resolves an edge-block byte to a
concrete target `NodeRow`/key. That resolution step is named but explicitly
NOT built (`row_for_local_key`-style deferred binding pattern appears
identically in `soa_view.rs:109-127` for the analogous key→row problem).

## Palette-lane precedent

`ValueTenant::{FrozenStyle, LearnedStyle, ExploreStyle}` — the
"autopoiesis triangle" — are the concrete, shipped precedent for "12
palette256 atoms, one content-blind register, reading is ClassView-selected":

- **Descriptor:** each is `ColumnDescriptor{kind: ColumnKind::U8,
  elems_per_row: 12, row_offset: 152|164|176}` (`:987-1004`), contiguous,
  `byte_len() == 12` (debug/release-guarded, see below).
- **Read accessor:** `NodeRow::style_lane(&self, tenant: ValueTenant) -> [u8; 12]`
  (`:1684-1698`). Guards `tenant.byte_len() != 12` by returning the
  all-zero lane rather than panicking or reading out of bounds — "a
  non-12-byte tenant would otherwise read past its lane into the next
  tenant's bytes — release-safe by construction, not a debug-only guard"
  (`:1687-1690`).
- **Write accessor:** `NodeRow::set_style_lane(&mut self, tenant, atoms: [u8;12])`
  (`:1705-1715`) — same 12-byte guard, no-op otherwise; bumps
  `crate::tenant_counter::tenant_update(tenant)` on write.
- **Per-family composite read:** `NodeRow::triangle_for(&self, family_ordinal: u8) -> (u8,u8,u8)`
  (`:1722-1738`) — reads the SAME byte index `i` out of all three 12-byte
  lanes at once, guarded `family_ordinal >= 12` → null triangle (the
  "never a wrong policy" zero-fallback). This is the shape a
  cross-tenant composite read (e.g. reading truth+rung+contradiction
  together) would mirror if they lived in sibling 12-byte lanes.
- **Field-isolation proof:** `thinking_style_triangle_tenant_carve_field_isolation_matrix`
  (`:2408-2519`) is the canonical worked example — (1) offsets exactly
  match `152-VALUE_SLAB_ROW_OFFSET` etc., (2) `Full` carries it,
  `Cognitive`/`Compressed` do not, (3) XOR-flip-and-check isolation
  matrix over all three lanes plus key/edges untouched, (4) typed
  accessor round-trip + atom-0 zero-fallback + out-of-range-ordinal
  null-triangle, (5) writing the triangle never disturbs `Kanban`/key/edges.
- **The `SpoFacet` reading (`awareness_facet.rs`) is the SAME precedent
  applied to a DIFFERENT existing 12-byte register** — its module doc is
  explicit: *"This is a reading, not a layout... nothing here reserves,
  moves, or stores a byte"* (`awareness_facet.rs:8-11`). It relabels the
  bytes already held by `MailboxSoaView::style_rails_at` (rail convention
  `rail k = (bytes[2k], bytes[2k+1])`) OR a `FacetCascade` payload, into
  6 `(u8,u8)` pairs = 3 semantic-SPO + 3 episodic-witness centroids
  (`SpoFacet::{subject,predicate,object,ew_subject,ew_predicate,ew_object}`,
  `from_register`/`to_register` round-trip proven loss-free by test).
  Per le-contract.md's "WHICH class reads its register as a `SpoFacet`
  is an OGAR mint... never a property of these bytes" — i.e. a belief
  tenant could analogously be "a reading of" the FrozenStyle/other
  12-byte lane rather than a brand-new tenant, IF a classview declared
  it so; this is a documented pattern, not something I am proposing here.

## Belief-state homes (EXISTS/NONE)

- **Truth (NARS frequency/confidence):**
  **EXISTS, but not as a `NodeRow` tenant.** `MetaWord` (`cognitive_shader.rs:38-76`)
  packs `nars_f(u8)` and `nars_c(u8)` — literally the NARS truth-value
  pair — into a `u32`: `thinking(6) + awareness(4) + nars_f(8) + nars_c(8)
  + free_e(6)`. `ValueTenant::Meta`'s doc-comment (`:829`) names `MetaWord`
  as its content, so the INTENDED home is the `Meta` tenant at value-slab
  offset 0 (row 32). But `Meta`'s `ColumnDescriptor` is `U64`/8 bytes
  while `MetaWord` is 4 bytes — this exact gap is called out as an open,
  unresolved discrepancy in le-contract.md §5 "Honest discrepancies" #4:
  *"Persisted-vs-hot width mismatches (Meta 8 B/4 B, Plasticity 4 B/1 B)
  — see tenants.md §7; parity test required before any 1:1 sync"*
  (le-contract.md:373-374). No `NodeRow::meta()`/`set_meta()` accessor
  exists to resolve which 4 (or 8) bytes actually carry `MetaWord` today.
  `graph_render::RenderEdge`/`InferredConnection` (`graph_render.rs:36-39,53-56`)
  separately carry `frequency: f32`/`confidence: f32` as NARS truth, but
  those are heap-based rendering DTOs, not row tenants.
- **Rung (reasoning depth):**
  **EXISTS as a standalone enum, NO row-tenant home found.**
  `RungLevel` (`cognitive_shader.rs:157-169`, 0..9, `Surface`..`Transcendent`)
  is a full `u8`-backed enum with `from_u8`/`as_u8`/`elevate`/`de_elevate`/
  `pearl_level`/`causal_mask_bits` methods (`:171-249`) — clearly designed
  to be stored as one byte somewhere, but no `ValueTenant` variant, no
  `ColumnDescriptor`, and no `NodeRow` field references `RungLevel`
  anywhere in `canonical_node.rs`. Grep across the crate found zero
  wiring of `RungLevel` into the value slab.
- **Contradiction depth:**
  **NONE EXISTS as a byte/row tenant.** `graph_render::Contradiction`
  (`graph_render.rs:61-70`) is the only `Contradiction`-named type in the
  crate: `{triplet_a: usize, triplet_b: usize, description: String}` — a
  heap-allocated (`String`, `usize`) rendering DTO living in
  `GraphSnapshot::contradictions: Vec<Contradiction>`, structurally
  incompatible with a fixed-byte value tenant (no `Copy`, unbounded
  size). CLAUDE.md's "Contradiction depth from Staunen × Wisdom qualia"
  (top-of-file Click section) suggests it should derive from
  `QualiaI4_16D` (the `Qualia` tenant, which DOES have a row home and a
  `row.qualia()` accessor) rather than needing its own tenant, but no
  code computes or stores a contradiction-depth scalar from qualia in
  this crate.
- **Premise/derivation refs:**
  **NONE EXISTS as a row tenant; a rich standalone vocabulary exists
  elsewhere, unwired to `NodeRow`.** `causal_audit.rs` defines
  `SupportBasis` (`:184-210`, 9 variants incl. `DerivationalTrace = 7`,
  "A derivation admitted it: recipe, rule, or rail, with a traceable
  path") and `CausalLocus::Derivational` ("the provenance of a
  conclusion... a belief from those premises", `causal_audit.rs:75-76`),
  plus an opaque evidence-source id type (`:239-` region). None of these
  types are referenced by `canonical_node.rs`, `VALUE_TENANTS`, or any
  `ValueTenant` variant — they exist as a free-standing audit/provenance
  vocabulary, not wired to the 480-byte slab.
- **Evidential base:**
  **NONE EXISTS as a row tenant.** `SupportBasis` (above) is a bitmask-
  friendly `#[repr(u8)]` enum (`.bit() -> u16`, `:213-218`) — structurally
  the closest existing candidate shape (an evidence-basis bitmask would
  fit in 2 bytes) — but again, no `ValueTenant` variant or
  `ColumnDescriptor` references it, and `settlement.rs`'s "evidentially
  earned" / "evidential grounding (1-U)" (`settlement.rs:101,118`) is a
  separate f32-based settlement-verdict type, also unwired to the row.

Summary: of the five belief-state facets asked about, only **Truth**
(via `MetaWord`'s `nars_f`/`nars_c`, doc-declared to live in the `Meta`
tenant, width-mismatched and un-accessored) has anything resembling a
declared row home. **Rung** has a shaped standalone type with zero row
wiring. **Contradiction depth**, **premise/derivation refs**, and
**evidential base** have no row-tenant home at all — the closest
material is heap-based rendering DTOs (`Contradiction`) or a free-
standing provenance vocabulary (`SupportBasis`/`CausalLocus`) that has
never been carved into `VALUE_TENANTS`.

## UNDETERMINED

- Whether the `Meta` tenant's actual byte 32..40 currently holds a raw
  `MetaWord` (upper 4 bytes unused/reserved) or something else — no
  `NodeRow` accessor exists to check this empirically; le-contract.md
  flags the width mismatch but does not say which 4 of the 8 bytes are
  "real."
- Whether `RungLevel` is intended to eventually live inside `Meta`'s
  spare bits, inside a new tenant, or inside a facet reading of an
  existing 12-byte lane (à la `SpoFacet`) — no doc-comment in
  `canonical_node.rs` or `le-contract.md` states an intended target.
- Whether the deferred "EdgeBlock slot-byte → neighbor-row resolution"
  (mailbox_scan.rs) is planned to use `local_key`-style basin indexing
  or a different scheme — the module doc names the open question but
  does not resolve it, and no plan file was read in this trace to check
  for a resolution (out of scope per the read list given).
