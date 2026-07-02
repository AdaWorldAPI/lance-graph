# The V3 LE Contract — the 4+12 facet atom

> READ BY: v3-envelope-auditor (mandatory), v3-mailbox-warden, anyone
> touching soa_envelope.rs / canonical_node.rs / tenant value schemas /
> bake pipelines. Routing semantics: `routing.md`. Lane meanings:
> `tenants.md`. Who writes what: `consumer-map.md`.

## Status: OPERATOR-LOCKED (2026-07-02, verbatim spec) — board entry E-V3-FACET-4-PLUS-12

---

## §1 The atom: 4 + 12 bytes (96-bit payload)

Every V3 unit is a **16-byte facet**: a 4-byte address prefix followed by a
96-bit payload. Little-endian throughout; nothing above the
`from_le_bytes` boundary cares about byte order.

```
byte 0        byte 1      bytes 2..3        bytes 4..15
┌───────────┬───────────┬────────────────┬──────────────────────────────┐
│ class/    │ appid     │ classview      │ payload — 96 bits, one of    │
│ domain    │ (2 nibble)│ (u16, 4 nibble)│ the sanctioned layouts (§3)  │
│ (2 nibble)│           │                │                              │
└───────────┴───────────┴────────────────┴──────────────────────────────┘
└────────── composed classid u32 ────────┘
   canon hi u16 = domain:appid              custom lo u16 = classview
```

- The 4-byte prefix IS the composed classid (canon-high, #628): canon
  hi u16 = `domain byte ++ appid byte` (e.g. `0x07:01` = OSINT:q2);
  custom lo u16 = the **ClassView selector**. Today the classview u16
  hosts the `0x1000` V3-adoption monitor + the OGAR §2 app render
  prefixes + interim kind slots; post-P4 it is the full 64k
  ClassView/template catalogue.
- Compose/split ONLY via the contract helpers (`render_classid`,
  `compose_classid`, `classid_canon`, `classid_canon_compat`) — the
  byte spelling above is documentation, never license for bit math
  (guardrails §1 rule 4).

## §2 The slot-purity rule (operator: "never waste a slot")

**Labels and positions come from the ClassView — NEVER from a slot in the
payload.** No name strings, no display labels, no layout positions, no
ordinal columns in any facet. The classview u16 resolves to the ClassView,
and the ClassView carries labels, field positions, and render order
(register-file model: the SoA is dumb bytes; the class makes them
meaningful). A proposed facet layout containing a label/position slot is a
LAYOUT-BREAK-class defect — reject at review.

## §3 The sanctioned 96-bit payload layouts (operator-locked catalogue)

Every 12-byte payload is exactly one of these; the classview selects which.

| # | Layout | Shape | Semantics | Notes |
|---|---|---|---|---|
| L1 | rails | 6 × (8:8) | `part_of : is_a` | the V3 mereology:taxonomy key rails; one-byte refs per slot |
| L2 | rails | 6 × (8:8) | `memberof : members` | membership plane |
| L3 | rails | 6 × (8:8) | `mereology : taxonomy` | generic mereology:taxonomy plane (as dictated) |
| L4 | palette pairs | 6 × (8:8) | `palette256²` — **CAM_PQ "digital" new style** | each byte pair indexes the 256×256 palette distance/compose tables (bgz17 lineage); similarity = ONE table read |
| L5 | triplets | 4 × (8:8:8) | SPO-style triplets | four 3-byte triples |
| L6 | quads | 3 × (8:8:8:8) | odoo-shaped relations | **[H] semantics open** — operator marked "odoo ?"; do not implement semantics before a ruling |
| L7 | absolute location | 2 × 48-bit | `hhtl ++ helix` | hhtl(48) = HEEL\|HIP\|TWIG (3×u16); helix(48) = helix place code. Together = absolute location, **two hemispheres** — see q2 FMA for usage |
| L8 | location, old style | 2 × 48-bit | `helix ++ CAM_PQ` — **CAM_PQ "analog" old style** | the harvest facet `helix-place(6) \| cam-pq(6)` — the 6-byte canonical CAM-PQ, NOT the 16-byte turbovec residue |

Byte accounting: every layout is exactly 12 bytes — 6×2, 4×3, 3×4, 2×6.
The 16-byte facet stride never changes; layouts differ only in how the
96 payload bits subdivide.

### The (8:8) pair is polymorphic — the classview selects the reading

Beyond L1–L4, a `6×(8:8)` plane admits these operator-sanctioned readings
(2026-07-02 extension):

- **`area : location` in stacked exactness** — the six pairs stack as a
  precision ladder: each pair refines the location within its area;
  stacked levels = progressive exactness.
- **Second GUID (relationships):** when a node carries a second GUID
  dedicated to relationships, its rail plane is ENCOURAGED to carry six
  relations as **`basin : relationtype`** pairs (one-byte basin ref +
  one-byte relation type).
- **Static-basin variant:** if the basins are **12 static**, the pair
  upgrades to **`relationtype : relationtype_orthogonal`** — the basin is
  implied by position/static table, freeing both bytes for two orthogonal
  relation types.

The reading is ALWAYS selected by the classview (slot purity §2) — never
by inspecting payload bytes, never by convention-in-code.

### The classview is a FOCUS LENS (operator, 2026-07-02)

The layout choice is not a storage convention — **the classview is the
focus lens that the DATA SHAPE wants**. A class whose data is relational
focuses rails; positional data focuses the location layouts; similarity
data focuses palette pairs. The lens follows the data, and the classview
carries that focus; code never second-guesses it.

### Let go of the cramped 64-bit register (operator, 2026-07-02)

The prior approach — cramming awareness into a 64-bit packed edge register
and *"hoping for a 3-bit mantissa to mean the whole awareness"*
(the CausalEdge64 inference-mantissa lineage, cf. the
I-LEGACY-API-FEATURE-GATED 5-instance catalogue those bits generated) —
is **let go**. Awareness/relation semantics get real width in the 96-bit
facet payloads, lens-selected per class. Consequences:

- Do NOT extend CausalEdge64 bit fields to carry new awareness semantics;
  new semantics land as facet layouts (L1–L8 + sanctioned readings).
- **[H] open:** CausalEdge64's residual role (wire/protocol/EdgeBlock
  compatibility) vs full retirement — needs a scoping ruling; the
  `edges[16B]` block in the CANON node is untouched by this (it is
  one-byte SLOT refs, not packed mantissas).

### CAM_PQ grounding (digital vs analog)

CAM_PQ's codebook is the **DeepNSM 4096-word English-native-speaker
codebook** (`crates/deepnsm`, COCA vocabulary): similarity between codes
A and B is a **distance-table lookup, never a float** — the 4096² u8
distance matrix (and the 256×256 palette tables at L4 granularity) IS the
similarity function. Two styles coexist:

- **"digital" new style (L4):** discrete palette-code byte pairs; distance
  composes through the 256×256 LUTs (palette semiring).
- **"analog" old style (L8):** 48-bit helix place codes + 6-byte CAM-PQ
  codes; continuous-flavored place geometry, table-compared.

### Open reconciliation items ([H] — flag, don't resolve locally)

- **L7 helix(48) vs the CANON key tail `family(u24)++identity(u24)`:**
  both are the trailing 48 bits of a 16-byte unit. Whether the CANON node
  key is literally an L7 facet (payload = absolute location) or a sibling
  layout is a ground-truth question for the envelope reconciliation —
  do not unify silently in code.
- **L6 semantics** await an operator ruling (the "odoo ?" mark).

## §3b Two-level LE contract + the jc-pillar validation gate (operator, 2026-07-02)

**"32-bit class, 96-bit data"** — and the contract is TWO-LEVEL:

- **Every TENANT has its own LE contract** (the facet layout it carries,
  per §3), and
- **the SoA envelope has ITS LE contract** (the register-file descriptor:
  ColumnDescriptor offsets/widths, `verify_layout()`,
  `ENVELOPE_LAYOUT_VERSION`).

The nesting is the point: the tenant contract is scoped INSIDE the
envelope contract, and both inherit the **single, compile-time-inherited
SoA write ownership** (owner-borrows structurally,
`mailbox_owner()` nominally). A tenant lane can never acquire a writer
the envelope doesn't know; an envelope can never write a lane whose
tenant contract it doesn't carry. Scopedness is the guarantee, not a
convention.

**Consumer validation gate (jc pillars):** every payload layout is so
DISTINCT that consumer readings are validated LATER against the `jc`
crate pillars in lance-graph — the certification math: **ICC, Spearman ρ,
Cronbach α** (and siblings). A consumer that starts reading a tenant lane
owes a jc-pillar certification run (the certification-officer pattern:
real bytes, deterministic sampling, 4-decimal reporting) before its
reading is trusted in any downstream claim. This is the statistical
mirror of the field-isolation matrix: layout tests prove bytes don't
move; jc pillars prove the READING preserves the semantics.

## §4 Relation to the CANON node and the envelope

- The CANON node (`CLAUDE.md` § Minimal SoA node, locked 2026-06-13)
  stays authoritative for the 512-byte row: `key(16) | edges(16) |
  value(480)`. The facet atom is the unit INSIDE value lanes (and the key
  itself is 16 bytes with the same 4+12 rhythm — see §3 open items).
- The 480-byte value slab holds facets per the tenant schema:
  `classid_read_mode(c).value_schema` selects which lanes/layouts a class
  carries (`tenants.md` catalogues them).
- `SoaEnvelope` is the register-file descriptor (ColumnDescriptor per
  column; `verify_layout()` = ABI conformance; `ENVELOPE_LAYOUT_VERSION`
  gates any byte movement). `SoaEnvelope::mailbox_owner()` stamps
  ownership; it lives on the ENVELOPE, never in a facet slot (§2 applies
  to ownership too).
- RESERVE-DON'T-RECLAIM: zeroed facets/lanes are dormant, never
  compacted; layouts are selected by classview, never by stride change.

## §5 Code ground truth (mapping fleet, 2026-07-02)

**The facet atom is CODED.** `facet.rs`: `FacetTier{lo,hi}` (2 B) +
`FacetCascade{facet_classid: u32, tiers: [FacetTier;6]}` — 4+12=16 B,
`size_of == 16` const-asserted (facet.rs:31-100); LE wire order
`facet_classid@[0..4)` then 6×[lo,hi]. Slot purity is enforced by
construction: the cascade is content-blind; only the ClassView interprets
the 8:8.

**Byte shapes vs the L1–L8 semantic catalogue:** the code implements
exactly 3 shapes + 1 lane (facet.rs:394-546, 207-223):

| Code | Shape | Covers |
|---|---|---|
| `CascadeShape::G6D2` | 6 groups × 2 levels (shift `>>1`) | L1–L4 — differentiated ONLY by ClassView (slot purity, as §2 demands) |
| `CascadeShape::G4D3` | 4 × 3 (divide) | L5 triplets |
| `CascadeShape::G3D4` | 3 × 4 (shift `>>2`) | L6 quads / canonical GUID tiers |
| `hi_chain()/lo_chain()` | 2 × 48-bit | L7/L8 — "a SEPARATE lane … never dragged into ClassView shape selection" (module doc) |

**Envelope:** `ENVELOPE_LAYOUT_VERSION = 2` (soa_envelope.rs:54; v2 =
HelixResidue 48 B→6 B right-sizing). `ColumnKind` (8 width-only LE kinds),
`ColumnDescriptor{name_id: u16, kind, elems_per_row: u16, row_offset: u32}`,
`verify_layout()` with 6 error paths incl. wasm-overflow checks.
`SoaEnvelope::mailbox_owner()` default 0 (soa_envelope.rs:170-197).
Tenant catalogue + ValueSchema presets + ReadMode registry: `tenants.md`.

**Honest discrepancies (carried, not hidden):**

1. **`SoaEnvelope` has NO production implementor** — only a test-only
   TestEnvelope. `MailboxSoA<N>` implements the sibling
   `MailboxSoaView/MailboxSoaOwner` traits; `NodeRow` reads via the
   `VALUE_TENANTS` table. Two parallel column-geometry systems share
   ColumnDescriptor/ColumnKind by convention, not by trait
   (ENTROPY-MILESTONES M7; W1 wiring decides the survivor).
2. **L7 open item sharpened:** `hi_chain/lo_chain` interleave one byte per
   tier ACROSS 6 tiers; the V1 key tail is two CONTIGUOUS u24s
   (family/identity). Structurally different bit-groupings — the code does
   NOT silently unify them; the reconciliation stays [H] as flagged in §3.
3. **MailboxId ≠ NiblePath in code** — `MailboxId = u32`
   (collapse_gate.rs:121), `NiblePath{path: u64, depth: u8}` (hhtl.rs:56);
   no conversion, no shared trait, no code comment linking them. The
   three-tier doc's "MailboxId IS the NiblePath" is a doc-only claim
   awaiting a ruling or a wiring PR.
4. Persisted-vs-hot width mismatches (Meta 8 B/4 B, Plasticity 4 B/1 B) —
   see `tenants.md` §7; parity test required before any 1:1 sync.

Cross-ref: board `E-V3-FACET-4-PLUS-12` (canonical ruling text),
`routing.md` §1 (prefix routing), `tenants.md`, primer §2,
`.claude/plans/soa-value-tenant-migration-v2.md` (the 16 B facet lineage:
`facet_classid(4) | helix-place(6) | cam-pq(6)` = L8 with its prefix).
