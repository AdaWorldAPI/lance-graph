# Plan — 32-tenant 512-byte SoA: honest full-width awareness → jc-measured collapse (M20)

> **Status:** DRAFT (envelope-auditor gate pending) · **Branch:** `claude/happy-hamilton-0azlw4` · **Date:** 2026-07-18
> **Advances:** ENTROPY-MILESTONES **M20** (64-bit awareness cram → 96-bit facet payloads)
> **Operator directive (verbatim):** *"keep the causaledge64 and episodicwitness64 code for reference until we actually can find a way to REALLY encode awareness in 64 bit sized. The priority is to get all 32 tenants together in 512 bytes sized SoA to reach the actual smartness, and then use JC crate to find out how to find redundancies. For now we want to overcome conjecture first."*

---

## §0 The conjecture, stated so it can be falsified

`CausalEdge64` (v2 layout, `cognitive-substrate-convergence-v1.md` §6) is the
**awareness after 2³ AND streaming cycles**, packed into 64 bits:

| bits | field | awareness role |
|---|---|---|
| 0..24 | S·P·O palette (3×u8) | the triple the awareness is *about* |
| 24..40 | NARS freq·conf (2×u8) | how strongly believed |
| 40..43 | **Causal mask 3b** | **the Pearl 2³ rung axis** (L-5) — the "2³" |
| 43..46 | direction triad 3b | sign per S/P/O |
| 46..50 | inference mantissa 4b s | direction × NARS rule |
| 50..53 | plasticity 3b | hot/cold per plane |
| 53..59 | W-slot 6b | witness-corpus root |
| 59..61 | truth-band lens 2b | committed-vs-ambiguous |

The **"streaming cycles"** half is *not stored in the edge* — L-2 dropped the
temporal field precisely because "temporal causality is structural" (carried by
`MailboxSoA::cycle` order + witness-chain emission-cycle position). `EpisodicEdges64`
(EW64) carries the episodic side: 4×u16 `(family:4b, local:12b-COCA)` edge addresses
into the episode basin.

**The conjecture:** that these ~64 bits *are* the awareness — that 2³ marginal
patterns over one 3×u8 SPO triple + 16-bit truth + a 6-bit corpus handle actually
captures "the actual smartness." This has **never been measured**. The 3/4-bit
inference mantissa "hoping to mean the whole awareness" is exactly what the
operator ruled we **let go** (le-contract §3 "Let go of the cramped 64-bit
register", 2026-07-02).

**We do NOT delete CausalEdge64 / EpisodicEdges64.** They are the reference —
the compressed hypothesis we are trying to *earn*. The plan below builds the
honest full-width representation *beside* them, measures it, and only then asks
whether 64 bits was enough.

---

## §0.5 OPERATOR DIRECTIVE (2026-07-18) — base single-sourced, OGAR mints per-app, two override points

> *"the most important part is making sure that the base layout is only imported
> in one place and ogar responsible for minting per app additional — that way
> activation can be overridden at any time in 2 places and cheaply tested and
> versioned and shrunken later."*

This **supersedes the "add 19 named `ValueTenant` variants" reading of §2.** The
512-byte envelope is **FIXED** — "shrink" is NOT byte-shrinking; it is the LATER
reorganization of the jc-proven tenants into a clean layout *within* the fixed 512,
**without hardcoding** (§0.6). 19 hand-hardcoded named base variants would make that
reorganization a hand-edit of the `VALUE_TENANTS` array + a blast radius across
every hand-copied consumer — the opposite of reorganizable-without-hardcoding. The
correct shape (which is *also* the already-coded `tenants.md §5` "facet lane"
model — *"a ClassView READING over existing presets, no enum variant, no layout
bump"*):

| Layer | Holds | Override point | Reorganize (the "shrink", §0.6) |
|---|---|---|---|
| **Base** — `lance-graph-contract`, ONE import site | the tenant layout descriptor (offsets/widths in the fixed 512) + the `FacetCascade` shape. Content-blind. **Mint-sourced / codegen'd, NOT hand-hardcoded.** | **Place 1** — the layout descriptor + `ENVELOPE_LAYOUT_VERSION` | re-mint → **regenerate** the descriptor (a versioned reorg + field-isolation matrix), never a hand-edit |
| **OGAR** — `ogar-vocab`, per-app mint | mints classid → ClassView whose `value_schema` **projects** the readings onto the layout | **Place 2** — the mint (versioned per app) | **re-mint** which readings each proven tenant carries |

So the A1–A7 + sibling facets of §2 are **OGAR-minted READINGS over one reserved
region, NOT base enum variants.** The base defines *where the bytes are* and
*what shape a facet is*, once; OGAR decides *which readings each app activates*.

**This mechanism already SHIPPED for the triangle (#729 P4, 2026-07-18) — reuse
it, don't reinvent.** `MailboxSoaView::style_rails_at(row, lane) -> Option<[(u8,u8);
6]>` (`soa_view.rs`) reads a 12-byte content-blind register as six `(u8:u8)`
rails — *"one register, two ClassView-selected readings — identical bytes,
different interpretation"* (the operator's "one register, two readings" ruling:
*"226 ARE the frozen; anything else needs 6×2×8bit … for an Orchestration for v3
substrate replayability"*). The **per-class doctrine** (#729 `3248bd9`): *"the
reading is ClassView-selected PER ROW/CLASS, never per lane."* The awareness
facets are **additional `style_rails_at`-shaped 6×(8:8) ClassView readings**,
OGAR-minted per app — the same substrate, one rung wider. `style_lane_at` (12×u8
→ 226-atom FROZEN palette256, `cognitive_palette.rs`) is the FROZEN sibling
reading; awareness facets are the "anything else" orchestration half.
Consequences that gate everything below:

1. **Single-import enforcement (Place 1 integrity).** The base type has one root —
   **`lance-graph-contract`** — consumed by exactly the OGAR-aligned cluster:
   **`lance-graph-ogar` + `OGAR` + `ogar-vocab`** (the mint/bridge; OGAR/ogar-vocab
   carrying the encoder is CORRECT, they are the mint side, not a violation).
   Everyone else *imports* the type, never copies. Real violation to close =
   ENTROPY **M21**: the non-sanctioned consumer hand-copies (`woa-rs/erp/canon.rs`,
   q2 cpic + fma) → one `canon-node-bytes` all import. **`symbiont` is NOT a fix
   target** — it is surrealdb-based and **deprecated in favor of OGAR**; its
   `bridge.rs:30 NODE_ROW_STRIDE = 512` copy retires *with* symbiont (dropping the
   INTEGRATION-PLAN W2c SurrealDB arm from the live path), not by wiring it to the
   base. Closing M21 is the prerequisite — a shrink is only "2 places" if the base
   is genuinely one root.
2. **The per-app mint moves to OGAR (Place 2 integrity).** The awareness
   `value_schema` projection is an **OGAR mint** (classid → ClassView), not a new
   hardcoded `BUILTIN_READ_MODES` preset baked in `canonical_node.rs`. The code's
   own intent already says so (`:888` "read-mode is layered in one level up").
3. **The auditor's LAYOUT-GATED verdict is re-scoped.** It audited the bake-19
   path. The new delta is far smaller: reserve ONE facet region (or a small fixed
   count of generic 12-B facet slots) + the ClassView reading contract — NOT 19
   named `ValueTenant` variants + 19 `Full`-mask entries. This needs a **re-audit
   of the region-reservation delta** before any `canonical_node.rs` edit; the 5
   blocking items shrink accordingly (the `Full.field_mask().count() ==
   VALUE_TENANTS.len()` lockstep becomes "one region descriptor," not 19).

### §0.6 What "shrink" means — 512 fixed, reorganize-the-proven, without hardcoding

**The 512-byte node is a fixed envelope; it never resizes.** "Shrink" (operator,
2026-07-18) = *"reorganizing the tenants that are proven to work, at a later time,
without hardcoding."* The arc:

1. **Spread wide** (this plan): fill the fixed 512's reserved headroom with
   candidate awareness readings — additive, RESERVE-DON'T-RECLAIM, no version bump.
2. **Measure** (§3): jc tells which candidates are proven (non-redundant,
   load-bearing) and which are redundant.
3. **Reorganize** (the "shrink", later): consolidate the *proven* tenants into a
   clean contiguous layout within the fixed 512, freeing the redundant slots — a
   deliberate, versioned reorganization (`ENVELOPE_LAYOUT_VERSION` bump +
   field-isolation matrix, I-LEGACY-API-FEATURE-GATED).

**"Without hardcoding" is the load-bearing constraint on step 3.** The tenant
layout descriptor must be **mint-sourced / codegen'd from OGAR**, not a
hand-authored `VALUE_TENANTS` array — so reorganizing is *re-mint → regenerate the
descriptor → recompile*, never hand-moving offsets in source. Compile-time
guarantees (contiguity / Full / ≤480 asserts, zero-copy) still hold because the
generated descriptor is compile-time-known; it is just **sourced from the mint,
not typed by hand**. That is why the reorg stays a **2-place** change (regenerate
the one base descriptor + re-mint the OGAR readings) instead of an N-place hand-edit
across every consumer — the M21 single-import + OGAR-produces-the-layout are the
two *enabling* invariants, not nice-to-haves.

> **Implication for the base today:** `VALUE_TENANTS` is currently a hand-authored
> `const` array (`canonical_node.rs:904`). Making it **OGAR-mint-sourced /
> codegen'd** is the prerequisite for reorganize-without-hardcoding — a broader V3
> direction this plan's awareness lanes must ride, not fight: do NOT add
> hand-hardcoded awareness variants that would then have to be hand-reorganized.

## §1 Why "spread wide, then measure" (the operator-blessed method)

This is not a new pattern — it is the workspace's own collapse discipline:

- `INTEGRATION_PLANS.md:75` — *"redundancy landed old+new and MEASURED with the
  jc battery before any [collapse]."*
- `CALIBRATION_STATUS_GROUND_TRUTH.md:165` — *"Cronbach α on 3 baked lenses:
  71.5% disagreement (superposition NOT redundant)"* — the precedent: three
  representations were kept **because jc measured them non-redundant.**
- le-contract §3b — *"every payload layout is so DISTINCT that consumer readings
  are validated LATER against the jc crate pillars — ICC, Spearman ρ, Cronbach α."*

So the arc is: **land the full-width (deliberately redundant) 32-tenant SoA →
run the jc battery across the awareness lanes → keep what is non-redundant,
collapse what is → the surviving width is the *measured* awareness width.** If it
collapses back toward 64 independent bits, the conjecture is *vindicated* and
CausalEdge64 becomes the proven canonical. If not, we have the real number.
Either way the conjecture is overcome by measurement, not assertion.

---

## §2 The honest full-width spread — 13 → 32 readings (derivation, not invention)

> **Read through §0.5.** The A1–A7 + sibling table below is the **catalogue of
> OGAR-minted ClassView READINGS** over the base's one reserved facet region —
> NOT 19 named base `ValueTenant` variants. "13 → 32" counts the *readings a
> fully-activated app projects*, not 32 base enum entries. The byte arithmetic
> below sizes the **reserved region**; the per-reading activation is the OGAR
> mint (Place 2).

Current `VALUE_TENANTS` (canonical_node.rs:904-993) = **13 lanes**, offsets
[32,188), 156 B of the 480-byte slab. `ValueSchema::Full` ends at 188;
**324 B of headroom, RESERVE-DON'T-RECLAIM.** BoardAggregates (W2a) reserves 188.

The new awareness lanes are **derived field-by-field from the two reference
types** — each awareness dimension CausalEdge64/EW64 crams gets *real width* as an
L4 `6×(8:8)` palette256² facet (12 B, the cosine-replacement/Fisher-z address),
plus deliberately-redundant sibling representations so jc has ≥2 readings of each
construct to correlate.

| # | Tenant (new) | 12 B facet reading (L4 unless noted) | Reference field it widens | Redundant sibling |
|---|---|---|---|---|
| A1 | `SpoFacet` | 6×(8:8): 3 SPO + 3 episodicwitness palette256² pairs | CE64 `S·P·O` (3×u8) + EW64 triple | vs CE64 24-bit SPO |
| A2 | `PearlRungFacet` | 6×(8:8): the 2³=8 intervention marginals over {S,P,O} | CE64 causal mask 3b | vs 3-bit mask |
| A3 | `NarsTruthFacet` | 6×(8:8): freq·conf per S/P/O plane (basin:strength) | CE64 freq·conf 16b | vs 16-bit truth |
| A4 | `FreeEnergyFacet` | 6×(8:8): likelihood / KL / F descent per cycle-window | active-inference (not in CE64) | vs Energy tenant (f32) |
| A5 | `StreamCycleFacet` | 6×(8:8): ±5 / ±50 / ±500 window awareness | "streaming cycles" (structural) | vs Kanban.cycle u32 |
| A6 | `DirectionInferenceFacet` | 6×(8:8): direction triad × inference mantissa full-width | CE64 dir 3b + mantissa 4b | vs 7-bit pack |
| A7 | `WitnessLensFacet` | 6×(8:8): W-slot corpus root + truth-band lens | CE64 W-slot 6b + lens 2b | vs 8-bit pack |

7 awareness facets × 12 B = 84 B. **13 + BoardAggregates(1) + 7 = 21 lanes.** The
remaining ~11 lanes to reach 32 are the **redundant siblings** made first-class
(each awareness construct carried in a *second* representation — e.g. Fisher-z i8
lane, raw-COCA-12bit lane — so the jc battery correlates representation-A vs
representation-B of the *same* construct). The exact sibling count is **not
pre-committed here** — it is bounded by the slab and *chosen to give jc a clean
k-item scale per construct*, not by a target number.

**Layout arithmetic (fits, layout-preserving, RESERVE-DON'T-RECLAIM):**
`188 (Full end) + 8 (BoardAggregates) + 7×12 (A1–A7) + ~11×12 (siblings) ≈ 412 B`
→ ends ≤ 412, slab-relative ≤ 380 ≤ 480. **No `NODE_ROW_STRIDE` change, no
`ENVELOPE_LAYOUT_VERSION` bump** (all additive, existing offsets frozen).

**Slot purity holds (le-contract §2):** every lane is a content-blind 12-B
palette256² register; the *reading* (which construct, which redundant sibling) is
the ClassView's, never a slot. No label/name/ordinal in any payload.

---

## §3 The mechanical collapse gate — the jc battery (`jc::reliability`)

All four measures are implemented and callable (`crates/jc/src/reliability.rs`:
Pearson `r`, Spearman `ρ`, Cronbach `α`, ICC `Icc2_1`/`Icc3_1`). The M20 gate:

1. **Per-construct Cronbach α** — for each awareness construct (A1–A7), treat its
   ≥2 representations as a k-item scale over N nodes. **α high (lanes cohere) ⇒
   redundant ⇒ collapse to one.** α low (71.5%-disagreement precedent) ⇒ the
   representations measure *different* facets of the construct ⇒ **keep both.**
2. **Pairwise ICC / Spearman** across *all* awareness lanes — find lane pairs that
   agree (absolute-agreement `Icc2_1`) → redundancy edges in a lane-graph.
3. **Collapse** the high-agreement clusters to one canonical lane each; the
   surviving lane count × its bit-width = the **measured awareness width.**
4. **Significance:** per `I-NOISE-FLOOR-JIRAK`, any "N σ above noise" claim uses
   `crate::jirak` (weak dependence), never IID Berry-Esseen.

This IS M20's mechanical gate ("a parity test, never a claim"): the collapse from
N redundant lanes to the canonical awareness width is *proven by the jc numbers*,
not asserted. Real bytes, deterministic sampling, 4-decimal reporting
(certification-officer pattern; each lane owes a jc-cert run before its reading
backs a downstream claim — tenants.md §7.4).

**This EXTENDS D-TRI-2, it does not fork it.** D-TRI-2 already is *"jc battery
(ICC, Pearson/Spearman, Cronbach α) over real shader cycles; collapse only on
measured identity"* — scoped to the 12-family vs 12-step reading. §3 is the SAME
gate widened to the A1–A7 + sibling awareness lanes. The D-TRI-1 triangle tenants
(Frozen/Learned/Explore, 3× palette256, merged #717, auditor LAYOUT-CLEAN) are the
shipped precedent shape for A1–A7 and the proof the additive-lane discipline
holds. This plan rides the **same batched OGAR mint** as BoardAggregates + D-TRI-1
classid + chess `0x06` (dtri1-classid-mint-spec-v1) — never a solo mint.

---

## §4 Sequencing + gates (nothing solo, nothing silent)

0. **Single-import prerequisite (Place 1 integrity — §0.5.1).** Base root =
   `lance-graph-contract`, consumed by `lance-graph-ogar` + `OGAR` + `ogar-vocab`.
   Close ENTROPY **M21** (the woa-rs/q2 consumer hand-copies → one
   `canon-node-bytes` all import). `symbiont` is deprecated (surrealdb → OGAR) and
   is NOT a fix target — its copy retires with symbiont. A shrink is only "2
   places" if the base is genuinely one root. Gates the plan, not just a lane.
1. **Region re-audit FIRST.** The delta is now *reserve ONE generic facet region*
   (or a small fixed count of generic 12-B slots) + the `FacetCascade` shape — NOT
   19 named `ValueTenant` variants. `v3-envelope-auditor` re-reviews THIS smaller
   delta (RESERVE-DON'T-RECLAIM, version-stability, region contiguity) *before any
   canonical_node.rs edit*. The prior LAYOUT-GATED verdict audited the superseded
   bake-19 path.
2. **OGAR mints the per-app readings, never solo (Place 2).** The awareness
   ClassView `value_schema` projections are minted in `ogar-vocab` (classid →
   ClassView), riding the batched mint with BoardAggregates + chess `0x06`
   (dtri1-classid-mint-spec-v1) — NOT a new hardcoded `BUILTIN_READ_MODES` preset.
3. **Reading at a time, jc-cert per reading.** A1 (`SpoFacet`) first — the user's
   established base design (3 SPO + 3 episodicwitness), least speculative. Each
   reading lands as an OGAR-minted ClassView projection + a jc-cert fixture over
   real node data. Reorganize later (the "shrink", §0.6) = re-mint → regenerate the
   layout descriptor for the proven tenants (versioned, within the fixed 512),
   never a hand-edit.
4. **CausalEdge64 / EpisodicEdges64 stay.** No shrink, no delete, until §3's
   measured width either vindicates or replaces the 64-bit hypothesis (M20
   residual-role ruling, still [H]).

---

## §5 Cross-session convergence

- **chess+reasoning session owns D-MTS-6b** (driver-integrated awareness fixture,
  `E-COMMA-AWARENESS-MEASURED-1`: k*=1, 2 truth bits/edge vs baseline 16). That
  fixture is the *awareness signal generator*; **this plan is the tenant
  substrate it writes into.** The jc battery (§3) is where the two meet — the
  fixture produces awareness, the 32 lanes hold it wide, jc measures the width.
- **This (graphrag/episodic) session** supplies A1's episodic side: `EpisodicBasins`
  / `episodic_search` / `EpisodeTheses` (merged #722/#725/#727) are the episodic-
  witness readings that populate `SpoFacet`'s 3 episodicwitness pairs — once
  recarriered off strings onto codebook indices (the separate numeric-codebook
  migration, subordinate to this plan per the operator's priority).

---

## §6 What this plan does NOT do (honest boundary)

- It does **not** pre-commit the exact sibling-lane count — that is jc-derived.
- It does **not** touch CausalEdge64/EW64 bit-fields (M20 review gate: "no NEW
  CausalEdge64 bit-field semantics").
- It does **not** claim the 64-bit conjecture is wrong — it builds the apparatus
  to *test* it.
- It does **not** edit canonical_node.rs before the envelope-auditor verdict.

Cross-ref: ENTROPY-MILESTONES M20 · le-contract §2/§3/§3b · tenants.md §7.4 ·
cognitive-substrate-convergence-v1 §6 (CE64 v2) · dtri1-classid-mint-spec-v1
(BoardAggregates + batched mint) · jc/src/reliability.rs.
