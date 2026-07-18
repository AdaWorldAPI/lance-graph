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
| L6 | quads | 3 × (8:8:8:8) | **SPOG** — three subject:predicate:object:graph quads (the Odoo factoring) | **RULED** (operator, 2026-07-06, odoo-transpile briefing §A3: "3×(8:8:8:8) SPOG (ODOO)"; consumed by odoo-rs — the former "odoo ?" mark is resolved). Byte shape = `CascadeShape::G3D4`; the attribute→byte-position assignor is the named remaining code (no owner yet, odoo-rs council 2026-07-07) |
| L7 | absolute location | 2 × 48-bit | `hhtl ++ helix` | hhtl(48) = HEEL\|HIP\|TWIG (3×u16); helix(48) = helix place code. Together = absolute location, **two hemispheres** — see q2 FMA for usage |
| L8 | location, old style | 2 × 48-bit | `helix ++ CAM_PQ` — **CAM_PQ "analog" old style** | the harvest facet `helix-place(6) \| cam-pq(6)` — the 6-byte canonical CAM-PQ, NOT the 16-byte turbovec residue |

Byte accounting: every layout is exactly 12 bytes — 6×2, 4×3, 3×4, 2×6.
The 16-byte facet stride never changes; layouts differ only in how the
96 payload bits subdivide.

### §3a Grace-period wide carvings — strongly discouraged (the V1 migration waiting room) (operator, 2026-07-13)

L1–L8 are the **axis-grouped byte-tile** catalogue: every tier is a *byte*
(8:8 pairs / 8:8:8 triplets / 8:8:8:8 quads), so the classview projects
real rails and `group_of` is a pure shift. A class that has **not yet
decomposed into byte-axis tiles** may sit *temporarily* in one of these
**wide contiguous carvings** — still exactly 96 bits, still one
content-blind register, **but not a tail and not first-class**:

| # | Layout | Shape | 96-bit? |
|---|---|---|---|
| G1 | wide-mixed | 3 × 16-bit + 2 × 24-bit | 48 + 48 ✓ |
| G2 | wide-triple | 4 × 24-bit (contiguous — **NOT** the axis-grouped `4×(8:8:8)`) | 96 ✓ |
| G3 | wide-quad | 3 × 32-bit (contiguous — **NOT** the axis-grouped `3×(8:8:8:8)`) | 96 ✓ |

**Strongly discouraged if god-object-related or lacking proper bucket
rollover; migrate to cosine-replacement palette256 (L4).** The conditions
are the diagnosis, not decoration:

- **god-object-related** → a wide field is a symptom of a class carrying too
  many concerns; you owe a **decomposition** (split the class / focus the
  lens), never a wider field.
- **lacking proper bucket rollover** → a wide contiguous field with no HHTL
  cascade spill has nothing to overflow *into*; it saturates silently. Give
  it rollover, or narrow it.
- **the exit** → the real destination is **L4 `6×(8:8)` palette256²**
  (each byte pair indexes the 256×256 palette LUT — cosine-replacement),
  the axis-grouped shape the wide field is standing in for.

These carvings exist **only** to give an un-migrated class a legal V3 home
during migration instead of crashing — the V1 `family:identity` u24 fragment
is the degenerate G1/G2 case. They are **not** a revival of the V1 *tail*
model (there is no path/tail split; this is one content-blind register read
coarsely), and **`CascadeShape` gains no variants for them** — that enum
stays byte-axis-only, and a wide carving is precisely what it refuses to
bless. New classes MUST NOT be born into G1–G3; the waiting room is not a
destination.

Code home: **`lance_graph_contract::legacy_outliers`** (`legacy_outliers.rs`)
— bluntly named so it announces its own status. `LegacyOutlier::{WideMixed
(G1), WideTriple (G2), WideQuad (G3)}` with LE read/write over the 12-byte
payload; deliberately a *separate* module from `facet::CascadeShape`, never a
variant of it.

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

### The canonical cosine/centroid replacement is ANALYTIC (operator, 2026-07-16)

L4's "cosine-replacement palette256" reads through the **analytic Fisher-z
codec** (`bgz-tensor::fisher_z::{FamilyGamma, FisherZTable}` — cosine →
`atanh` → 8-byte per-family affine → normalized i8; hydrate via `tanh`;
certified ρ≥0.999, E-PALETTE-NNUE-COSINE-GREEN-1). **A materialized k×k
table is a CACHE of the formula, never the canon**: ranking compares raw
i8 (monotone), distance is `|Δi8|·(z_range/254)`, and hydration is one
affine + `tanh` — so an L4 ClassView may declare an analytic codebook
(the 8-byte gamma) with zero table materialization. Boundary: this
replaces the distance/rank READ; the semiring COMPOSE keeps its table
(z-addition does not compose cosines). The location-side sibling, one
rung up: **helix is to Fisher-2z what the cosine-replacement is to
Fisher-z** — helix pins `hyperbolic_depth = 2·fisher_z` as a tested
identity (`crates/helix/src/fisher_z.rs:110-121`) and runs its residue
pipeline on that doubled scale (`Signed360` = the hemisphere-doubled
register). Canonical ruling text: EPIPHANIES
`E-FISHERZ-CANONICAL-COSINE-REPLACEMENT-1`.

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

### Honourable mention — the bgz-tensor Hadamard-residual ladder (index + residual; out-of-row) (operator, 2026-07-16)

Next to the three flavours of 256 (CAM-PQ = 6×256² compressed to per-query
6×256 ADC rows; bgz17/L4 = the explicit materialized 256²; the V3 facet's
`6×(8:8)` rails = codec-agnostic 6×256² ADDRESS, classview-switched), a
fourth **operational mode** deserves explicit mention — shipped in
`crates/bgz-tensor/src/adaptive_codec.rs`: **index + residual**.
`AdaptiveRow { centroid_idx: u16, scale_bf16, scale2_bf16, … }` keeps the
palette/CLAM centroid as the coarse deterministic PLACE and stores a
**Hadamard-rotated residual** in a three-tier LFD split (corrected
2026-07-16, codex P2 on #700 — `classify_rows_by_lfd`): the **hardest
~top-10% LFD rows escape to `RowPrecision::Passthrough`** (the exact
original vector is stored; NO centroid/residual representation — the
codec's own refuse-to-force-the-mold tier), the **next ~20%** get an i8
residual, and the **bottom ~70%** the i4+i2 cascade. Index PLACES,
residual CORRECTS — the same place/magnitude decomposition as the
perturbation pyramid, with the magnitude side stored as a graded ladder;
consumers must NOT assume every row has an index+residual representation
(the Passthrough tier does not).

Why it earns the mention here: it is the **continuous-field exit** that
flat L4 cannot provide. A bare 256-level index terraces a continuous field
(first consumer instance, geo arc 2026-07-16: 256 levels over ~1500 m of
relief ≈ 6 m elevation terraces), while categorical/narrow surfaces
(`Signed360` normals, harmonized colour) stay flat L4.

**Demarcation refined (WI-3 measured, 2026-07-18, `E-BGZ-TENSOR-LANE-REVIEW-1`):**
for a **monotone BOUNDED** continuous field (elevation is the canonical
case), the exit is **analytic, not materialized** — helix `RollingFloor`
(`quantize`→`bucket_center`) reconstructs the ~1500 m elevation field at
**1 byte / RMSE ≈ 1.7 m = 0.11 % of range** (reproducing the 5.86 m =
1500/256 terrace figure exactly), and a composable 2-byte stacked floor
reaches ≈ 0.023 m terraces (PROBE-HELIX-CONTINUOUS-FIELD, helix
`continuous_field.rs`). So the honoured **materialized** bgz-tensor ladder
is NOT the owner of the *monotone-bounded* exit — it demarcates to its
distinct lane: **unbounded / clustered / multi-modal** value distributions
(the heavy-tailed weight-row reconstruction the CLAM/centroid machinery is
built for) that do NOT fit a single bounded `RollingFloor` range. Caveat:
the shipped multi-byte residue codecs (`ResidueEdge`/`Signed360`) are
sphere/place codecs, so "analytic" here means `RollingFloor` palette256 +
stacked floors, not `ResidueEdge`.

Placement
discipline: this is **NOT a ninth 12-byte layout** — the residual ladder is
**out-of-row** (same status as `Signed360` in the 96-bit carving); the
sanctioned in-row refinement budget remains the turbovec 6×4-bit nibble
lane. L4 byte pairs stay the index; the ladder rides beside the row when a
class's classview focuses a continuous field.

Formal anchor: **Hambly–Lyons 2010** (Annals of Mathematics 171,
"Uniqueness for the signature of a path of bounded variation and the
reduced path group"): a bounded-variation path is determined by its
**signature** — the graded cascade of its iterated integrals — up to
tree-like equivalence. That is the theorem-shaped version of the ladder's
promise (a graded residual cascade determines the continuous field up to
negligible equivalence) and of the replayable-trajectory framing (store
the graded cascade, recover the path). **The theorem side is already
in-workspace, not an external citation** (corrected 2026-07-16, operator
pointer): **jc Pillar 11** (`crates/jc/src/hambly_lyons.rs`, feature
`hambly-lyons` → the `sigker` sibling; forward probe = tree-equivalent
excursion collapses to the identity signature via Chen's identity,
converse probe = triangle's non-zero level-2 signed area; deliberately
uses `sigker::signature_truncated`, not the known-buggy
`signature_kernel_pde`) and **ndarray `src/hpc/pillar/signature.rs`**
(B7: signature transform + Hambly–Lyons sig-kernel, Gram-matrix
positive-semidefiniteness check, certification probe over 1 000 Lévy
paths). What stays **[S — analogy-grade; consult [FORMAL-SCAFFOLD] before
promotion]** is only the **ladder-levels → signature-levels mapping** for
the residual cascade; its probe can be built directly on the existing
Pillar-11 harnesses rather than from scratch.

**Provenance caveat (operator, 2026-07-16):** bgz-tensor predates
turbovec/PolarQuant and helix — the ladder's mechanisms (Hadamard
rotation, the bespoke i4+i2 cascade, the Passthrough escape) were chosen
before today's lane inventory existed. This mention records what SHIPS,
not what we would build now; an **engineering follow-up review is
queued** as `TD-BGZ-TENSOR-PRE-LANE-REVIEW` (reconcile vs turbovec
Lloyd-Max+NativeLut, PolarQuant rotation findings, helix
residue/`CurveRuler`; outcome = consume-the-lane / demarcate / retire).

Cross-ref: ndarray `pr-x12-h268-morton-wgpu-synergies.md` §8 (the three
flavours + this mention), `E-PALETTE-RESIDUAL-LADDER-1` (EPIPHANIES),
`E-H268-REPLAYABLE-TILE-1`, `TD-BGZ-TENSOR-PRE-LANE-REVIEW` (TECH_DEBT).

### Open reconciliation items ([H] — flag, don't resolve locally)

- **L7 helix(48) vs the CANON key tail `family(u24)++identity(u24)`:**
  both are the trailing 48 bits of a 16-byte unit. Whether the CANON node
  key is literally an L7 facet (payload = absolute location) or a sibling
  layout is a ground-truth question for the envelope reconciliation —
  do not unify silently in code.
- ~~**L6 semantics** await an operator ruling (the "odoo ?" mark).~~ **RULED 2026-07-06**: L6 = 3×SPOG quads (subject:predicate:object:graph), the Odoo factoring — see the regraded table row above. Open residue: the attribute→byte-position assignor (name→slot projection rule per §2 slot-purity) does not exist yet for any layout.

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
