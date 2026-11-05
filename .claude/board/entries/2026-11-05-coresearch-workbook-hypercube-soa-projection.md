# /coresearch exploration map — the SharePoint workbook as a hypercube projection over the 64K SoA (2026-11-05)

**Status:** EXPLORATION MAP. It ratifies, implements and adopts nothing on its own authority.
The brief, crosswalk and raw scout notes are banked in the session scratchpad and are not part of
the record; every claim below carries its file:line in the text or cites the crosswalk row it
comes from.

- **Council:** premise gate (before and after); 5 scouts (code cartographer, internal prior art,
  literature, systems, concepts and ontology); 2 added scouts, declared (ternlog plus `simd_stride`
  cartographer, open-plans auditor); 3 co-architects (bridge, firewall and fit, falsifier).
- **Consumer:** report-rs PR #33 (`11d2bec`), Linear EXO-50.
- **Sources:**
  - lance-graph `f05c9d1`
  - ndarray `b3d7919`
  - report-rs `feat/workbook-schema-reconciliation`
  - papers and systems, graded in each row

## 0. Question and anchors

**Question.** How does one workbook revision become a projection over the 64K SoA, with no side
structure? The codebook lives inside the SoA, every cell is one u64 word, and the pairwise
relations chain cheaply (X ⊗ Y = Z).

**Anchors (operator, 2026-11-05):**
- Text never enters the hot path.
- DR64 never admits text.
- Everything is a hypercube projection over the SoA.
- A codebook that bypasses the 64K SoA is a violation.
- u64 is never mixed with f64 or u32.
- Every AB, BC and CA pair chains like X ⊗ Y = Z (ternlog chaining amortization; `simd_stride.rs`).
- Every open plan is checked.
- **PR #33 is a materialization violation until it is resolved.**

**Premise gate (before):** PREMISE-WRONG on the first framing.
- "C = values" is not an axis. Values are the codomain of the cell function, and magnitudes have
  no position.
- Joining projections of one functional ternary relation yields spurious tuples.
- "Content address = row address" conflates a hash (`ContentId`, FNV-1a, `content_store.rs:51-63`)
  with a dense minted position.

The anchor stands; what changed is the choice of X, Y and Z:

| Symbol | Meaning |
|---|---|
| A | Identity positions (≤ 64K) |
| B | Property positions (≤ 256) |
| C | Codebook positions: the 64K SoA rows that text identities are minted into (text only) |
| AB | The cell function (a, b) → tagged u64 word |
| CA_b | The converse of AB_b |

**⊗ = contraction over a shared index with one index pinned.** It is never a Kronecker product,
and never a rebuild of the ternary relation.

## 1. The answer in one paragraph

The workbook needs one materialized foundation: AB, as one plain u64 lane per property indexed by
identity position. Each word is a fixed tag prefix plus a payload.
- The tag holds CellKind and the raw markers `-`, `offen` and `Impl.` as distinct tags.
- The payload holds either a C position (text) or a scaled integer (magnitude; the scale is per-B
  DT meaning, and overflow is refused).

Text exists only at intake, where it is minted ONCE into a C position by an in-SoA open-addressed
probe whose slot is the row (precedent: `onebrc-probe` GridCodebook, `lane_j.rs:58-91`), and at
egress, where it is hydrated last.

Everything else is a fold:
- CA_b is `MatchU64` over lane b, giving an 8 KB mask on demand. It is never a stored index.
- Chains of pinned contractions fuse into ternlog passes: at most 3 distinct leaves per pass,
  measured ≈ 0.4 µs per 64K for up to 6 resident planes, with 0 allocations.
- XVERWEIS = pin b1 on the key's code → first/only position → **aligned** read of lane b2 at the
  same A coordinate. This is the "read where it sits" discipline of `simd_stride.rs:1-23`: the
  load is the read. It is never a via-lane gather.
- The generation diff is `ne_u64_to_mask` per lane, OR-ed into a WideFieldMask per position. It is
  exact only if the mint is stable across generations.

The chain is exact because every CA is taken with b pinned. One shared mint keeps cross-column
equality a word compare (Abadi SIGMOD06 §6), and pinning removes the spurious-tuple risk that a
global code space carries (lossless join needs C→B otherwise).

The side codebook (CamLabels + MemKv), the f64 and mixed-width lanes, the `StoredCell` lanes, the
`Dirty::between` host loop and every plan that prescribes them are superseded.

## 2. Answers to the re-asked questions

- **Q1′ mint and word.**
  - Mint at intake only: hash → probed 64K slot, append-only and never renumbered. It is
    witnessed by a generation stamp carried with the rows.
  - Occupancy folds into the tag word (tag 0 = empty), so no bool lane.
  - Name bytes live in an intake/egress content lane that Quack never reads.
  - Word: a frozen tag/payload bit budget declared once in the contract (X3). **Not yet frozen:
    gate G2.**
- **Q2′a / Q2′b.**
  - The one foundation is the AB lanes (X4; the factorized-DB shape A → one value per B).
  - CA is a fold, not a second foundation (X5). A prepared inverse column is a forbidden index;
    the permutation is the only admitted index (lane-fold capstone).
- **Q3′a.** Mask AND/ternlog over aligned A words; measured figures in X6. Fusion drops the gated
  word skip, which is a cost to watch.
- **Q3′b.**
  - Reading AB at a found position is an aligned load, not a gather.
  - mxm (`relcomp`, `matmul_i8_to_i32`) enters only for bag multiplicity or batched key columns.
    Park it.
  - The 256×256 compose LUT is lossy and was rejected before. Skip it.
- **Q4′ XVERWEIS.** Pin, then a first/only position terminal, then the aligned read. #NV is an
  empty mask.
  - First-match vs refuse-duplicate is a DT policy. Excel defaults to first match in sheet order;
    SH7 refuses (`rpt-sh7-mapping-hydration-xlsx-v1.md:43`).
  - Two parts are MISSING:
    - a first/only-set-position terminal (mask-risc `Terminal`, `ir.rs:289-380`);
    - a sheet-order lane. It must be read as a mask-ranked min (a fold), never as `x[y[i]]`.
- **Q5′ diff.** Per-lane word inequality, on the identity side of the two-algebra rule (legal
  even for magnitudes, because it tests bit equality and never merges by XOR). Raw formula text
  takes part only as a C position.
- **Q6 decimals.** Scaled integer in the payload; per-B (scale, max-digits) in DT. The ceiling is
  below 18 digits once the tag bits are taken, so it must be declared, and overflow is refused at
  intake, never wrapped or sent to an f64 fallback.

## 3. Verdicts

| # | idea (crosswalk row) | bridge | firewall | verdict | probe / reason |
|---|---|---|---|---|---|
| X1 | in-SoA mint, slot = row | OPPORTUNITY | CONFLICT → seam MINT-AT-INTAKE (probe only at ingest; names in an intake content lane, not `Vec<Vec<u8>>`; occupancy in tag) | **PROBE** | PR-X1/X17 |
| X2 | retire side codebook | OPPORTUNITY | PASS | **ADOPT-NOW** (supersession only) | §5 |
| X3 | tagged u64 cell word | OPPORTUNITY | CONFLICT (minor) → frozen fixed-prefix bit budget | **PROBE** | PR-X3/X11; gate G2 |
| X4 | AB = u64 lane per B | OPPORTUNITY | PASS (lanes.rs today = TRAP instance) | **ADOPT-NOW** (shape), after X3 | PR-X4/X9 |
| X5 | CA_b as a fold | OPPORTUNITY | PASS (prepared index = forbidden; Roaring arrays = PARK) | **ADOPT-NOW** | PR-X5 confirms |
| X6 | ⊗ = ternlog-fused contraction | OPPORTUNITY (exists) | PASS | **ADOPT-NOW** | PR-X6 confirms |
| X7 | one shared mint, b always pinned | OPPORTUNITY (ruling) | PASS (refuse unpinned BC⋈CA at plan time) | **ADOPT-NOW** as ruling, operator to confirm | PR-X7 proves pinning load-bearing |
| X8 | XVERWEIS pin → position → aligned read | WORTH-EXPLORING | CONFLICT → key resolved at the boundary; `EqU32Via` forbidden on this path | **PROBE** (pre-registered RED) | PR-X8 |
| X9 | read where it sits (fold to a mov) | OPPORTUNITY (rule) | PASS; AoS strided scan = TRAP | **ADOPT-NOW** (discipline) | — |
| X10 | diff = `ne_u64_to_mask` → WideFieldMask | OPPORTUNITY | PASS | **PROBE** | PR-X10 vs `Dirty::between` as oracle only |
| X11 | scaled decimals, per-B scale | OPPORTUNITY | CONFLICT → declared ceiling, refuse overflow | **PROBE** | with PR-X3 |
| X12 | mxm / relcomp | DROP single / EXPLORE batched | PASS as PARK (width mix outside DR64) | **PARK** | only if PR-X8's batched gate fails |
| X13 | 256×256 compose LUT | DROP | TRAP | **SKIP** | rejected prior (agreement 0.025–0.043) |
| X14 | DR globals from mint | WORTH-EXPLORING | CONFLICT → globals = C position only for text-keyed identity; RouteLog append-only first | **PREMISE-WRONG (gate after)** → the property belongs elsewhere: a separate identity mint (same mechanism, own tenant/classid) writing into the DR MapTable, numeric keys routed there too | PR-X14 (`MapTable.advance`, dr.rs:593-601) still expected RED |
| X15 | 128-bit facet lane vs u64 word | WORTH-EXPLORING | CONFLICT → dedicated u64 DR64 lanes, facet lane a separate tenant | **PROBE** + operator gate G3 | PR-X15 |
| X16 | deepnsm-v2 16-bit versioned codebook | WORTH-EXPLORING | CONFLICT → same contract, separate classid-scoped codebook | **PARK** | align after X1 lands |
| X17 | codebook witness with the rows | OPPORTUNITY | PASS, required | **ADOPT-NOW** | in PR-X1/X17 |
| X18 | stable code space is ours | OPPORTUNITY | PASS | **ADOPT-NOW** (constraint) | PR-X16/X18 can-fire check |

## 4. Pre-registered probes

Ground rules for every probe:
- Synthetic K-code fixtures only.
- Every timing asserts against an oracle first.
- Every probe has a disable run that must go red.
- A result in ms is a failed gate.

Run them in this order:

1. **PR-X1/X17 — mint stability and collisions** (M).
   - Input: 64K texts at load factor 1.0, plus 4K adversarial collisions in the low 16 bits.
   - Assert: lookups equal a test-only BTreeMap oracle; C positions survive a generation with
     10 % edited, 5 % deleted, 5 % appended and the rows reordered; a stale frame gives
     `StaleCodebook`.
   - Kill: any renumbering or wrong hit; mean lookup above 50 ns; max probe above 256 at LF 1.0.
     A kill here means a 2× slot space is required.
   - Disable: remove name verification, or mint by sorted position.
2. **PR-X3/X11 — tagged word round-trip** (S).
   - Input: every CellKind × every raw marker × payload extremes.
   - Assert: `MatchU64` selects exactly the oracle set (can-fire and stay-silent cases); a value
     one digit past the ceiling is REFUSED; `1.10` and `1.1` at scale 2 give equal words.
   - Kill: aliasing between kinds, or an accepted overflow.
3. **PR-X7 — pinned exactness** (M).
   - Input: `offen` in two columns for different identity sets.
   - Assert: the pinned query equals the oracle; the unpinned composition shows at least one
     spurious A; with C→B holding, the two agree.
4. **PR-X5 / PR-X6 — fold costs** (S).
   - Kill: a CA fold above 2 µs per 64K; 3 leaves above 1 µs; 5–6 leaves above 2 µs; any
     allocation proportional to the population.
5. **PR-X8 — XVERWEIS semantics** (M).
   - Assert against a hand-written XLOOKUP reference (search_mode 1 and −1, if_not_found), with
     duplicates at sheet rows 7 and 3, an absent key, and an edit that reorders them.
   - **Expected RED today**, for lack of a sheet-order lane and a position terminal.
   - Gates: a single lookup at most 1 µs; a batched 64K key column below 1 ms (otherwise the X12
     probe is triggered).
6. **PR-X10 — generation diff** (S).
   - Assert: the p15 fixtures re-encoded give exactly `Dirty::between`'s (a, b) set. That struct
     is used as an oracle only, never as a shape (A10). Self-diff gives an all-zero mask.
   - Disable: an unstable mint must flag every coded cell.
7. **PR-X14 / PR-X15** (M / S).
   - X14: the frame history must survive 3 generations; **expected RED** (`advance` overwrites).
   - X15: run the same folds on a u64 lane and on the u64 half of a facet; kill the carrier that
     needs a mixed-width read or costs more than 20 % extra.

## 5. Supersession needed (open plans checked)

These conflict with the anchors and need a dated supersession note. They are not deleted.
- **report-rs:**
  - `docs/plans/rpt-xl-delta-02-corrections-v1.md`: whole plan; it still reads "proposed".
  - `rpt-xl-delta-02-review-v1.md`: R2 (AbiBatch as the lane store) and "What survives" row 2
    (codebook ordinals as globals).
  - `rpt-xl-delta-01-v1.md`: §3, lines 66-67 and 75-76.
  - `decision-log-v1.md`: D-23 and D-29 (lines 29 and 35), superseded by a new D-30.
  - `rpt-sh7-mapping-hydration-xlsx-v1.md`: line 33 (Codebook = MemKv + CamLabels) and line 92
    (persist the codebook).
- **Kept from these plans:** the corrections plan's C5/C8 falsifiers (as acceptance tests); the
  review's R5/R7/R11; SH7's refuse-duplicate policy and its witness pattern; the rpt-2 F_STATUS
  tag vocabulary; the DT64 `crm_company` basis binding.
- **lance-graph rows that bound this design:**
  - STATUS_BOARD D-QMB-0..5, D-ASM-11, D-RA-4 (lines 639-644, 679-681 and 722) carry
    "⛔ VIOLATION — GATHER". This forbids `EqU32Via` on the XVERWEIS path.
  - Sanctioned aligned reads: D-HOLO-H14d (`d-holo-h14d-logic-mask-read-v1.md:181-190`).
  - Bag-only mxm: D-DYNBAG-1.
  - `d-dynbag-4-dr-versioned-mapping-v1.md`: frame history.
  - `duckdb-dr-quack-machine-room-reuse-v1.md:62`: the permutation is the only pointer index.
  - `cross-glove-business-parity-v1.md:262-268`: per-field canonical scale.
  - `deepnsm-v2-lexical-address-v1.md`: the sibling versioned codebook.

## 6. Three operator gates before any code

- **G1 GATHER-DECISION** (still OPEN on the #1590 ledger). Is the mint probe an intake-only
  operation, outside the hot path, so that it is not a hot-path gather? The design assumes yes.
- **G2 Bit budget.** Freeze the tag/payload split (for example 8 tag bits and 56 payload bits)
  and the decimal ceiling it implies.
- **G3 Carrier width.** Are DR64 cell lanes dedicated u64 lanes, separate from the 128-bit facet
  lane (capstone: "Lane is 64k ordered 128-bit SoA")?

## 7. Not searched

- Kùzu design documents (the host was blocked).
- ClickHouse.
- The primary papers on triadic FCA and on Leapfrog Triejoin (seen only second-hand).
- lance-graph board entries, EPIPHANIES, ISSUES and TECH_DEBT beyond the items named.
- About 330 plans that matched no keyword.
- report-rs `docs/requirements/`.
- Linear EXO-50.
- The "first set position" ABSENT verdict rests on a grep over named files (mask-risc `Terminal`
  enum; ndarray simd, masking and bitwise public functions). It is a closed search, not a proof
  that no such primitive exists anywhere.

## 8. Premise gate (after) — PREMISE-SPLIT, folded in

The gate passed X2, X4–X7, X9, X11–X13, X15 and X17. It split or rejected the following:

- **X14 PREMISE-WRONG.**
  - An A-global answers "which entity" and is written by the DR. A C position answers "which
    text" and is written by the mint. Making one equal to the other is a category error.
  - The property belongs in a separate identity mint (the same mechanism, its own tenant) that
    writes into the MapTable. Numeric keys need their own route into it.
  - Re-asked: where do A-globals come from, for text keys and for numeric keys, independent of the
    C mint?
- **X1 / G1 SPLIT.** "Mint" names two operations:
  - **mint-write** (claim a slot), once, at intake;
  - **mint-resolve** (read a slot for a query key), on every text-keyed XVERWEIS.
  
  Both address memory from data, so by signature both are gathers. The question is whether a
  gather outside the hot path is sanctioned. G1 therefore becomes:
  - **G1a:** is the intake mint-write sanctioned?
  - **G1b:** is a per-query resolve a sanctioned, named boundary gather, or must every key arrive
    already resolved to a C position?
- **X3 / PR-X7 SPLIT.** PR-X3 files `offen` as a tag (status); PR-X7 uses it as a shared code
  (value). Each raw marker is one or the other, never both. **G4:** is `offen` (and are `-` and
  `Impl.`) status or value? PR-X7's fixture follows the answer.
- **A vs C SPLIT.** Both are called "64K SoA rows". **G5:** are identity rows and codebook rows
  separate tenants (proposed) or one address space?
- **BC MISSING.** Building "which codes column b admits" writes mask bits at value-derived
  positions. That is a scatter, not an aligned fold, and the map was silent on it. Either name a
  sanctioned intake scatter, or drop BC from the A4 chain and record why.
  - Note: the XVERWEIS and diff paths use only AB and CA_b.
- **X8 sheet order SPLIT.** A per-A rank is either an ordinary value lane read by a masked min
  (DT meaning), or the permutation (the one admitted index). Decide which.
- **X10 SOUND, with a condition.** An append-only mint fills across generations. Add the case
  "codebook fills across generations" to PR-X1: 64K slots at load factor 1.0 is the kill line,
  and a refusal past the knee is a refusal, never a hidden hash table.

**Operator gates, revised:** G1a, G1b, G2 (bit budget), G3 (carrier width), G4 (markers: status
or value), G5 (A and C tenants), plus the BC scatter question.
