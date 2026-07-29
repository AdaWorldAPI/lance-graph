# Tenant Lanes — the value-slab catalogue (code ground truth 2026-07-02, refreshed 2026-07-28)

> READ BY: v3-envelope-auditor (mandatory), anyone reading/writing a tenant
> lane, adding a ValueSchema, or wiring a consumer. Byte atom + payload
> catalogue: `le-contract.md`. Who writes what: `consumer-map.md`.
> Ground truth source: mapping-fleet pass over
> `crates/lance-graph-contract/src/{canonical_node,soa_envelope,facet,kanban,soa_view}.rs`
> + `crates/cognitive-shader-driver/src/mailbox_soa.rs`.

> **⊘ 2026-07-28 refresh note — this catalogue is GENERATED FROM
> `canonical_node.rs::VALUE_TENANTS` (+ the `ValueTenant` enum), which is
> the single source of truth.** The doc had drifted **four tenants
> behind** between the 2026-07-02 freeze and this refresh: §2 stopped at
> tenant 9 (`Kanban [144,152)`) while code had already shipped tenants
> 10–14 (`FrozenStyle`/`LearnedStyle`/`ExploreStyle`/`Tekamolo`/
> `CausalWitness`, running the value slab to `[204,220)`). **Byte offsets
> in this doc are DERIVED and go stale on every tenant landing — cite
> `ValueTenant::X.value_offset()` symbolically in code and docs, never a
> literal offset.** This is the exact failure mode that made the
> BoardAggregates reservation go stale three times in a row
> (152 → 188 → 204) as tenants kept landing underneath it.

## Status: FINDING (byte-accurate, file:line-cited; two flagged seams at the end)

---

## §1 The persisted row and its slab

`NodeRow` = 512 B: `key(16) | edges(16) | value(480)`.
`VALUE_SLAB_ROW_OFFSET = 32`, `VALUE_SLAB_LEN = 480`
(canonical_node.rs:718-720). **`ENVELOPE_LAYOUT_VERSION = 2`**
(soa_envelope.rs:54) — v2 = HelixResidue right-sized 48 B → 6 B, shifting
every downstream tenant offset. Every tenant below carries its OWN LE
contract nested in the envelope's (le-contract.md §3b).

## §2 The 15 value tenants (`ValueTenant`, canonical_node.rs:828-909; `VALUE_TENANTS`, canonical_node.rs:935-1054)

Discriminant = FieldMask bit = VALUE_TENANTS index (compile-asserted).
Offsets are FULL-ROW; subtract 32 for slab-relative.

| # | Tenant | Kind × n | Width | Row range | Carries |
|---|---|---|---|---|---|
| 0 | Meta | U64 × 1 | 8 B | [32,40) | MetaWord (thinking/awareness/NARS/free-energy bits) |
| 1 | Qualia | U64 × 1 | 8 B | [40,48) | QualiaI4_16D — 16 signed-4-bit channels |
| 2 | MaterializedEdges | U64 × 4 | 32 B | [48,80) | 4 out-of-family CausalEdge64 |
| 3 | Fingerprint | U8 × 32 | 32 B | [80,112) | 32 B identity print (not the 16 Kbit plane) |
| 4 | HelixResidue | U8 × 6 | 6 B | [112,118) | 48-bit helix place (2× 24-bit equal-area hemisphere, Signed360) |
| 5 | TurbovecResidue | U8 × 16 | 16 B | [118,134) | PQ32x4 residue (16 B turbovec — NOT the 6 B canonical CAM-PQ) |
| 6 | Energy | F32 × 1 | 4 B | [134,138) | spatio-temporal accumulator |
| 7 | Plasticity | U32 × 1 | 4 B | [138,142) | persisted plasticity |
| 8 | EntityType | U16 × 1 | 2 B | [142,144) | OGIT class ordinal (1-based registry index) |
| 9 | Kanban | U64 × 1 | 8 B | [144,152) | `phase(u8) \| exec(u8) \| reserved(u16) \| cycle(u32)` (KanbanTenant::to/from_bytes, canonical_node.rs:1385-1409) |
| 10 | FrozenStyle | U8 × 12 | 12 B | [152,164) | Autopoiesis-triangle FROZEN lane — 12 palette256 atoms, one per StyleFamily ordinal (or compiled-template step); the CHECKPOINT policy the can't-stop-thinking dispatch runs off. Atom 0 = null default (zero-fallback) |
| 11 | LearnedStyle | U8 × 12 | 12 B | [164,176) | Autopoiesis-triangle LEARNED lane — same 12-slot palette256 shape; the NARS-revision-updated policy the L4 learning seam writes (owner `&mut`); promotes to `frozen[f]` only after winning the held-out arm |
| 12 | ExploreStyle | U8 × 12 | 12 B | [176,188) | Autopoiesis-triangle EXPLORE lane — same 12-slot palette256 shape; exploration variant from the P64 perturbation ladder (StreamDto → PerturbationDto), deterministic address-derived jitter (D-QUANTGATE coprime walk, never RNG) |
| 13 | Tekamolo | U8 × 16 | 16 B | [188,204) | TEKAMOLO facet lane — 16 B content-blind V3 4+12 facet (`classid(4) + 6×(u8:u8)`), read G4D3 as `temporal · kausal · modal · lokal` (when/why/how/where circumstance-frame). All-zero = unaddressed |
| 14 | CausalWitness | U8 × 16 | 16 B | [204,220) | CausalWitness facet lane — 16 B content-blind V3 4+12 facet read as **G24N4** (24 signed i4 loci, a lane shape name, never a `CascadeShape` variant); each nibble is a context pointer (signed ±8 window offset), not a strength. Slots 16..24 reserved-zero. **Status: EXPERIMENTAL — not in the operator-locked §3 catalogue** (per its own doc-comment) |

`ValueSchema::Full`'s `field_mask()` (canonical_node.rs:1132-1157, as read
2026-07-28) lists all 15 tenants 0–14 (Meta … `CausalWitness`), totalling
220 B of 480 — **260 B headroom, RESERVE-DON'T-RECLAIM** (compile-asserted
≤ 480, canonical_node.rs:1197).

> **⊘ TRANSIENT-READ CORRECTION (2026-07-28, same day).** An earlier
> revision of this section flagged a live defect — "`Full` lists 0–13 while
> `VALUE_TENANTS` carries 15; the assert at canonical_node.rs:1198 requires
> these to match (14 vs 15 as read)." **That mismatch never existed as a
> committed state.** It was an in-flight intermediate: the `CausalWitness`
> mint was landing concurrently, and this doc was generated between the
> descriptor row and the `Full` field-mask update. The assert is exactly
> what makes such a window uncompilable, so it can never be observed in any
> commit — reading it from a working tree mid-edit is the only way to see
> it. Recorded because it is this document's own lesson turned on itself:
> **a value read from a moving surface is transient, and writing it down as
> standing fact is the same class of error as recording a derived byte
> offset as if it were primary.** Verify against a committed tree, never a
> mid-edit one.

## §3 ValueSchema presets (canonical_node.rs:894-970)

| Preset | Tenants | Use |
|---|---|---|
| Bootstrap = 0 (default) | none (FieldMask::EMPTY) | zero-fallback ladder |
| Cognitive = 1 | Meta, Qualia, Fingerprint, Energy, Plasticity, EntityType, Kanban (7) | thinking rows |
| Compressed = 2 | Fingerprint, HelixResidue, TurbovecResidue, EntityType (4) | baked/search rows (q2 bakes) |
| Full = 3 | all 15 tenants 0–14 (Meta … `CausalWitness`) | superset; count compile-asserted `== VALUE_TENANTS.len()` |

## §4 The classid → tenant resolution (ReadMode registry)

`classid_read_mode(classid) -> ReadMode { tail_variant, value_schema,
edge_codec }` via `BUILTIN_READ_MODES` (canonical_node.rs:1041-1227):

- `TailVariant`: V1 `family(u24)·identity(u24)` (default) / V2
  `leaf·family·identity (3×u16)` (feature `guid-v2-tail`) / V3
  cascade-key `(part_of:is_a)` 8:8 tile (feature `guid-v3-tail`).
- OSINT/FMA/PROJECT/ERP = {V1, Cognitive|Compressed}; `*_V3` classids
  (OSINT_V3/FMA_V3/CPIC_V3) = {V3, …}. `DEFAULT = {V1, Full, CoarseOnly}`
  — documented TEMPORARY.
- Unmapped classids fall through to DEFAULT; legacy `_LEGACY` alias keys
  keep pre-flip forms resolving forever (retirement = corpus-proof-gated,
  W6).

This registry IS the "classview selects the reading" mechanism at its
current maturity: today it selects tail + schema + codec; the full
64k-ClassView focus-lens (le-contract §3, E-V3-CLASSVIEW-FOCUS-LENS)
lands post-P4.

## §5 The facet lane (Phase-2 target)

`FacetCascade { facet_classid: u32, tiers: [FacetTier; 6] }` = the coded
4+12 atom (facet.rs:31-100, `size_of == 16` const-asserted). Byte shapes:
`CascadeShape::{G6D2, G4D3, G3D4}` = the L1–L4 / L5 / L6 readings;
`hi_chain()/lo_chain()` (facet.rs:207-223) = the L7/L8 2×48-bit lane
("separate lane … never dragged into ClassView shape selection").
Phase 2 of `soa-value-tenant-migration-v2` re-reads HelixResidue +
canonical 6 B CAM-PQ as the ONE contained 16 B facet
`facet_classid(4) | helix-place(6) | cam-pq(6)` — a ClassView READING
over existing presets, no enum variant, no layout bump.

## §6 In-RAM mirror: MailboxSoA columns (mailbox_soa.rs:58-207)

Per-mailbox hot columns: `energy[f32]`, `plasticity_counter[u8]`,
`last_active_cycle[u32]`, `last_write_cycle[u32]`, `edges[CausalEdge64]`,
`qualia[QualiaI4_16D]`, `meta[MetaWord=u32]`, `entity_type[u16]`,
`temporal[u64]`, `expert[u16]`, `sigma[u8]`, heap planes
content/topic/angle (256×u64 each), + `phase: KanbanColumn` (owner-only).
Zero-copy per-column views via repr(transparent) reinterprets
(`edges_raw()`, `meta_raw()`). `DefaultMailboxSoA = MailboxSoA<1024>`.

## §7 Seams every session must know (flagged, not resolved)

1. **SoaEnvelope trait has NO production implementor** — MailboxSoA
   implements `MailboxSoaView/MailboxSoaOwner` (soa_view.rs); NodeRow
   reads via the VALUE_TENANTS table + hand accessors. Two parallel
   column-geometry systems share ColumnDescriptor/ColumnKind but not the
   trait. Wiring `mailbox_owner()` provenance into the PRODUCTION path =
   INTEGRATION-PLAN W1; collapsing the two systems is an entropy
   milestone (ENTROPY-MILESTONES.md).
2. **Width mismatches, persisted vs hot:** Meta 8 B (slab) vs 4 B
   (MetaWord u32); Plasticity 4 B (slab U32) vs 1 B (saturating u8
   counter). No slab↔MailboxSoA parity test exists (only
   MailboxSoA↔BindSpace, mailbox_soa.rs:1144). Any 1:1 sync needs an
   explicit shim + parity test first (envelope-auditor gate).
3. **MailboxId ≠ NiblePath in code.** `MailboxId = u32`
   (collapse_gate.rs:121); `NiblePath{path:u64, depth:u8}` (hhtl.rs:56).
   The three-tier doc's "MailboxId IS the NiblePath" is DOC-ONLY — no
   conversion, no shared trait. Needs an operator/architecture ruling
   before any code assumes the identity.
4. **jc-pillar gate applies per lane** (le-contract §3b): a consumer
   starting to read any tenant above owes an ICC/Spearman/Cronbach
   certification run before its reading backs any downstream claim.

Cross-ref: `le-contract.md`, `routing.md` §4, `consumer-map.md`,
board E-V3-TWO-LEVEL-LE-JC-GATE, `.claude/plans/soa-value-tenant-migration-v2.md`.
