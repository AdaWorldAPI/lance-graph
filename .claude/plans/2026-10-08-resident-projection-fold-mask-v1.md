# resident-projection-fold-mask-v1 — mask × fold over projections of resident bytes

> **Status:** PROPOSAL (D-RPF-0..8). No code authorized. Every phase is a probe
> or a measurement first; a phase that needs a new primitive STOPS and names the
> gap instead of building it one layer up.
>
> **Read first:** `.grok/board/CLAUDE_LANE_FOLD_CAPSTONE.md` (settled numbers,
> refusal list — this plan does not reopen any of it);
> `entries/2026-10-08-ce64-isa-register-contract.md` (strict decoder, field
> contracts); `entries/2026-10-08-hhtl-nars-moore-value-tenants.md` (tenants
> 18..20); `entries/2026-10-08-coresearch-ce64-moore-masking-wiring.md`
> (prefilter-as-mask, Moore-as-Morton-shift).
>
> **Origin:** session reconnaissance on optical / lithographic primitives
> (chat only, no repo artifact). What survived the reconnaissance as algebra,
> not vocabulary: aperture = indicator product; detection = masked sum;
> hotspot dedup = evaluate once per proven-equivalent class; Abbe/SOCS =
> identical operators collapse because the sum is linear; the latent →
> threshold → irreversible staging. Everything below is grounded in shipped
> code; the optics is not cited as evidence for anything.

## Overview

One resident byte range — a `NodeRow` (512 B: `key 16 | edges 16 | value 480`)
held by its owner — has three readings, and this plan keeps them apart:

| reading | question | carrier | executes in |
|---|---|---|---|
| **mask** | which rows are admissible | `u64` plane, LSB-first | `mask-risc` `Pred` / `MaskOp` over a `LaneRef::Strided` projection |
| **fold** | what single answer leaves | a `Terminal` | `mask-risc` executor, tiled, one terminal per homomorphism class |
| **register** | what one CE64 instruction computes | `CausalEdge64` | `causal-edge` `isa::execute` / named register methods |

The rules that bind all three are already law:

- mask-risc **A1**: the plan describes, the executor borrows, ndarray computes,
  the caller owns memory. No selection vector, no per-row object.
- mask-risc **A2**: masks choose admissibility; magnitude is a separate
  reduction over survivors. A mask bit is never a weight.
- **Tiled execution** (2026-09-21): no intermediate population when the next
  fold can consume the projection directly. A caller-owned bitmap is still
  materialisation.
- **Capstone**: quack lowers and must not evaluate; one terminal per
  homomorphism class; a lossy decomposition is a refusal, never a fold; do not
  route lane folds through cognitive layers, the shader driver, or ontology
  bridges.
- **CE64 ISA**: reasoning chooses the operation; `CausalEdge64` defines what
  it means. Unsupported codes refuse (`IsaFault::Unsupported`).

The work is wiring between the three readings without a materialised hop
between them. Where a hop has no primitive, the phase stops and files the gap.

## Checklist

- [ ] **D-RPF-0** — Resident CE64 field predicate in place
- [ ] **D-RPF-1** — Law-table → pattern-set compiler (recipe eligibility as a mask)
- [ ] **D-RPF-2** — Class census: how many distinct eligibility and read-set classes a real population has
- [ ] **D-RPF-3** — Read-set dedup for CE64 instructions (hotspot dedup, exact)
- [ ] **D-RPF-4** — Population revision as one fold: decision + gap filing
- [ ] **D-RPF-5** — Moore tenant lanes under a declared `SlabReading`
- [ ] **D-RPF-6** — Tile skip on a zero gate (capstone point 3) measured on resident rows
- [ ] **D-RPF-7** — Rank spectrum of the 256×256 palette distance LUTs (measurement only)
- [ ] **D-RPF-8** — Boundary declaration: what mask-risc never executes

## Details

### D-RPF-0 — Resident CE64 field predicate in place

**Claim to test.** A Pearl3 / Epi5 / mantissa predicate over the CE64 words in
`ValueTenant::MaterializedEdges` (U64 × 4) needs no extracted lane:
`Pred::MatchFacet16Strided` over a `StridedRef { stride: 512 }` whose 16-byte
window holds edge `k`, with `care` set only on the field's bits, gives the same
mask as the CE64 accessors row by row. A threshold on an n-bit field is a union
of at most n + 1 ternary patterns (the PROBE-PREFILTER-MASK result), combined
with `Or`.

**Offsets.** Taken from `ValueTenant::MaterializedEdges.value_offset()`, never
a literal (tenants.md §refresh note).

**Falsifier.**
- can fire: a pattern whose `care` names the wrong bit range must disagree with
  the accessor on a fixture that varies that field;
- can stay silent: a field the predicate does not name (S/P/O, W) varied over
  every row must not change the mask;
- anti-vacuity: the fixture's predicates admit fewer than 1/3 of rows.

**OPEN (decision, not code).** For edges `k = 3` (and any window placed so the
edge sits in one half), the 16-byte window spans bytes of a neighbouring tenant
with `care = 0`. Whether a `care = 0` overlap counts as *reading* that tenant —
for RBAC projection (`WideFieldMask`) and for the zero-copy law — must be
decided before this lands. The alternative is an 8-byte strided ternary match,
which does not exist in the IR today (`MatchU64` is contiguous only). If the
overlap is refused, D-RPF-0 files that primitive as the gap.

### D-RPF-1 — Recipe eligibility as a mask

**Grounding (VERIFIED-IN-CODE).** `affordance_measurement_probe.rs`: eligibility
is `STATE[raw5] & PEARL[pearl3]`; every other CE64 field is swept and proven
irrelevant. Only 10 of 32 Epi5 codes are valid; Pearl3 is read by one recipe.

**Claim to test.** For each recipe bit `r`, the set of admitted
`(raw5, pearl3)` pairs is enumerable at build time from the law tables, so
"rows where recipe `r` is eligible" is an `Or` of at most `10 × 8` ternary
patterns over the D-RPF-0 projection — no per-row `u64` eligibility word is
materialised.

**Deliverable.** A pure compiler from the law's compiled tables to
`Vec<(pattern, care)>` per recipe (a build-time function, not a runtime
object), and the differential: compiled mask == per-row `eligible` bit, for
every recipe, on the probe's fixture.

**Falsifier.**
- can fire: dropping one admitted code from a recipe's pattern set must lose
  exactly the rows carrying that code;
- can stay silent: `OBSERVE_FOLD` (no requires, no forbids) must compile to
  "every row whose Epi5 is valid", and a v1-provenance row must refuse
  (the `band_reading` rule), not match.

### D-RPF-2 — Class census

**Question.** How much does the hotspot idea buy on real data? Count distinct
classes, do not assume them.

**Measure, on a real resident population (not a synthetic uniform one):**
1. distinct `(raw5, pearl3)` — the eligibility classes (bounded above by 80);
2. distinct read-set keys per CE64 instruction (D-RPF-3's key);
3. the multiplicity histogram for each.

**Output.** `classes / rows` per instruction. If the ratio is near 1, D-RPF-3
is dropped with the number recorded — the exact dedup has nothing to save.

**Constraint.** Counting distinct keys must not use a population-sized seen-set
(`Terminal::ScatterCountU32` is HELD). Either the lane is key-ordered
(`Terminal::CountKeyRunsU32`) or the census runs as an offline measurement
outside mask-risc and says so.

### D-RPF-3 — Read-set dedup for CE64 instructions (exact hotspot dedup)

**Grounding (VERIFIED-IN-CODE).** `causal_edge::isa::contracts::{FORWARD,
LEARN, SYLLOGIZE, REVISION}` declare reads / computes / passes / constants per
instruction, measured by perturbation in `ce64_isa_contract.rs`. The
MooreNars16 probe measured that Direction3, Witness6 and Epi5 are never read by
any instruction — only carried or dropped.

**Claim to test.** Two operands that are equal on an instruction's declared
read set produce results equal on every computed field. The class key is the
projection onto the read set **plus the resolution context of every handle the
result carries**:

| handle | context the key must include |
|---|---|
| W slot (6 bits) | the cohort / mailbox whose `WitnessTable` resolves it |
| Epi5 code | `(classid, rail, generation)` — the code's meaning is declared per class (`layout.rs`) |
| S/P/O palette index | the classid / ClassView whose codebook the index addresses |

Then: evaluate once per class, attach the multiplicity, and fold. A
pass-through field is copied from its declared source, never deduplicated.

**Falsifier.**
- can fire: removing the cohort from the key must change at least one result
  on a fixture with two cohorts sharing a W index;
- can stay silent: dedup-then-fold equals per-row evaluation for Count, Sum,
  and Avg-as-`(sum, count)`;
- idempotent folds (Any / Or) ignore multiplicity; additive folds must not.

**Out of scope.** Execution of the deduplicated instruction over survivors is
not a mask-risc op (see D-RPF-8).

### D-RPF-4 — Population revision as one fold

**Observation.** NARS revision of `n` independent truths is the weighted mean
`f = Σ wᵢfᵢ / Σ wᵢ`, `c = W / (W + 1)` with `W = Σ wᵢ`, `wᵢ = cᵢ / (1 − cᵢ)`.
Over the reals that is a two-moment fold `(Σw, Σw·f)` divided once — the same
shape as the capstone's `AVG` as `(sum, count)`. The pairwise
`CausalEdge64::revision` chain re-quantises to u8 after every step, so the two
are NOT expected to agree bit for bit.

**Decisions required before any code (OPEN):**
1. **Which is normative** for a population: the one-shot fold or the pairwise
   register chain. Until decided, neither may be substituted for the other.
2. **The c = 255 defect** (ISA entry, OPEN): `evidence_weight(1.0)` is
   `f32::MAX`, two of them overflow to `inf` and the result collapses to 0.
   A fold over a population hits this whenever any member has c = 255. The
   smallest fix (cap c before `evidence_weight`) must land first.
3. **Admission is the mask.** Stamp disjointness / `GadamerRevision` decides
   which rows may enter; A2 puts that in the mask, never in the weight. The
   self-revision defect (PROBE-STAMP-GATE) is exactly a missing admission mask.
4. **Witness scope.** One W per mailbox (`MailboxSoA::apply_edges`), so a fold
   over one mailbox's rows has a uniform witness; across mailboxes it does not
   and must refuse.

**The gap (named, not built).** `wᵢ` is a nonlinear function of a resident
byte. No terminal reads a byte through a 256-entry LUT into a weighted sum in
place (`MaskedStridedGroupSum` sums raw fields; `GroupPowerSumsI32` needs an
`I32` lane). Materialising a weight lane is forbidden by the tiled law. The
primitive belongs in `ndarray::simd` (T1) and is filed there; mask-risc only
names it once it exists.

### D-RPF-5 — Moore tenant lanes under a declared reading

**Grounding.** Tenants 18..20 (`Nars16x8`, `MoorePalettePairs`, `MooreNars16`)
are ratified and pinned, with no production reader; no `SlabReading` variant
declares them, so `ResolvedReading` cannot gate a consumer on them; the
production Simpson detector cannot refuse a Moore direction.

**Steps.**
1. Contract decision: a `SlabReading` variant for the Moore tenants (additive;
   `from_tag` refuses unknown tags already).
2. Mask over the eight `MooreNars16` lanes as strided `u16` fields
   (`stride 512`, offset from `value_offset() + 2·slot`): Epi5 and Pearl3 are
   inside each lane, so D-RPF-0's pattern technique applies per slot.
3. The Moore schedule stays the measured `grid & shift(grid, −d)` over
   `mask_shift_morton` (PROBE-MOORE-PLANES). The palette fold stays a value-plane
   LUT fold (A2).

**Falsifier.** A consumer that requires a sign triple must refuse a Moore lane
through the declared reading, as the probe-local consumer already does — and
the production Simpson path must be gated until it takes a tagged operand.

### D-RPF-6 — Tile skip on a zero gate, on resident rows

Capstone point 3, measured on the resident-row predicates of D-RPF-0/1.
A skipped 256-word tile of an all-zero gate is identical to the unskipped one by
construction, so the filter is conservative without a proof obligation beyond
the existing differential. Print dead-tile fraction beside dead-word fraction.
Points 4 (`Range` without `_under`) and 9 (scattered survivors) are in the same
order of work and are not re-derived here.

### D-RPF-7 — Rank spectrum of the palette distance LUTs

`causal_distance` sums three 256×256 `u16` LUT reads — a bilinear form per
plane. Measure each table's eigenvalue (or SVD) spectrum. This is a measurement
only: a truncated decomposition is lossy and, per the capstone, can never be a
fold. It is reported as a possible documented approximation, nothing more. The
compose LUTs (`u8 → u8`) are not linear and are out of scope.

### D-RPF-8 — Boundary: what mask-risc never executes

To be written down, then enforced by review:

- mask-risc never executes a CE64 instruction. Deduction chains are not
  associative/commutative folds; `forward` dispatches on an operand; neither
  fits a terminal. A CE64 kernel over survivors is a separate bulk kernel
  (in `causal-edge`, over `ndarray::simd` where it vectorises), reading the
  mask, writing through the owner (write-on-behalf), never a per-row object.
- No VSA / superposition carrier enters this path (capstone: no cognitive
  layers). A VSA tree query is at most a comparison arm outside the crate.
- No new opcode, field, or tenant is introduced by this plan; D-RPF-4 and
  D-RPF-0 may each file one T1 primitive, backend-first.

## Order

D-RPF-0 → D-RPF-1 → D-RPF-2 (decides whether D-RPF-3 is worth doing) →
D-RPF-3. D-RPF-4 waits on its four decisions. D-RPF-5 waits on the
`SlabReading` decision. D-RPF-6 and D-RPF-7 are independent measurements.

## Not searched / not verified

- `ndarray` was not checked out in the session that wrote this plan; primitive
  names (`mask_shift_morton`, `ternary_match_*_to_mask_under`,
  `masked_strided_group_sum`, `ternary_match_strided16_to_mask`) are taken from
  mask-risc's own docs and the probes, not from ndarray's source.
- No real resident population was measured; D-RPF-2 is the first measurement.
