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

## §2 The honest full-width spread — 13 → 32 tenants (derivation, not invention)

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

1. **Envelope-auditor gate FIRST.** The A1–A7 + sibling lanes are a layout delta →
   `v3-envelope-auditor` reviews the field-isolation matrix, RESERVE-DON'T-RECLAIM,
   version-stability *before any canonical_node.rs edit*. (This turn.)
2. **Batched OGAR mint, never solo.** The awareness-facet classids ride the next
   batched mint with BoardAggregates + chess `0x06` (dtri1-classid-mint-spec-v1),
   not a separate train.
3. **Lane at a time, jc-cert per lane.** A1 (`SpoFacet`) first — it is the user's
   own established base design (3 SPO + 3 episodicwitness) and the least
   speculative. Each subsequent lane lands with its own facet-layout + a jc-cert
   fixture reading real node data.
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
