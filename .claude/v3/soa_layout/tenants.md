# Tenant Lanes — the value-slab catalogue (code ground truth 2026-07-02)

> READ BY: v3-envelope-auditor (mandatory), anyone reading/writing a tenant
> lane, adding a ValueSchema, or wiring a consumer. Byte atom + payload
> catalogue: `le-contract.md`. Who writes what: `consumer-map.md`.
> Ground truth source: mapping-fleet pass over
> `crates/lance-graph-contract/src/{canonical_node,soa_envelope,facet,kanban,soa_view}.rs`
> + `crates/cognitive-shader-driver/src/mailbox_soa.rs`.

## Status: FINDING (byte-accurate, file:line-cited; two flagged seams at the end)

---

## §1 The persisted row and its slab

`NodeRow` = 512 B: `key(16) | edges(16) | value(480)`.
`VALUE_SLAB_ROW_OFFSET = 32`, `VALUE_SLAB_LEN = 480`
(canonical_node.rs:718-720). **`ENVELOPE_LAYOUT_VERSION = 2`**
(soa_envelope.rs:54) — v2 = HelixResidue right-sized 48 B → 6 B, shifting
every downstream tenant offset. Every tenant below carries its OWN LE
contract nested in the envelope's (le-contract.md §3b).

## §2 The 10 value tenants (`ValueTenant`, canonical_node.rs:729-849)

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

`ValueSchema::Full` uses 152 B of 480 — **328 B headroom,
RESERVE-DON'T-RECLAIM** (compile-asserted ≤ 480, canonical_node.rs:974).

## §3 ValueSchema presets (canonical_node.rs:894-970)

| Preset | Tenants | Use |
|---|---|---|
| Bootstrap = 0 (default) | none (FieldMask::EMPTY) | zero-fallback ladder |
| Cognitive = 1 | Meta, Qualia, Fingerprint, Energy, Plasticity, EntityType, Kanban (7) | thinking rows |
| Compressed = 2 | Fingerprint, HelixResidue, TurbovecResidue, EntityType (4) | baked/search rows (q2 bakes) |
| Full = 3 | all 10 | superset |

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
