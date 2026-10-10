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

- [x] **D-RPF-0** — Resident CE64 field predicate in place (`lance-graph-mask-risc/tests/resident_ce64_predicate.rs`; entry `2026-10-08-d-rpf-0-resident-ce64-predicate.md`)
- [ ] **D-RPF-1** — Law-table → pattern-set compiler (recipe eligibility as a mask)
- [ ] **D-RPF-2** — Class census: how many distinct eligibility and read-set classes a real population has
- [ ] **D-RPF-3** — Read-set dedup for CE64 instructions (hotspot dedup, exact)
- [ ] **D-RPF-4** — Population revision as one fold: decision + gap filing
- [ ] **D-RPF-5** — Moore tenant lanes under a declared `SlabReading`
- [ ] **D-RPF-6** — Tile skip on a zero gate (capstone point 3) measured on resident rows
- [ ] **D-RPF-7** — Rank spectrum of the 256×256 palette distance LUTs (measurement only)
- [ ] **D-RPF-8** — Boundary declaration: what mask-risc never executes
- [x] **D-RPF-9** — Fold-Join deforestation probe (measured 2026-10-08; rewrite contract proposed, not implemented)

## Three stages of deforestation (added 2026-10-08, D-RPF-9)

Each stage removes one kind of intermediate. They are distinct and must not be
conflated:

| stage | what is never built | state on `main` | phases |
|---|---|---|---|
| **1. Projection** | an extracted CE64 lane | predicates read `MaterializedEdges` in place through `MatchFacet16Strided`; equal to the accessors (D-RPF-0 test-pinned, D-RPF-9 probe) | D-RPF-0, D-RPF-1, D-RPF-5 |
| **2. Mask** | a population-sized bitmap between a predicate and a terminal that consumes it | already the tiled law: one 2 KB tile per slot, 0 allocations; remaining work is R2 (strided pattern merge), R3 (gate selection), G1 (strided `_under`), G3 (tile skip) | D-RPF-6, D-RPF-9 |
| **3. Fold-Join** | a pair relation, cross product, join result or intermediate aggregate when aligned resident addresses and the fold's law suffice | shipped for ≤ 6 resident planes (`fused_ternlog` / `tern2` / `tern3`, no slot); G2 (two-window strided match) is the open gap for CE64 pairs | D-RPF-9 |

Stage 3 is an intersection at the SAME address. A hop that resolves a
different address (`Gather`, `ScatterOrU32`, multi-hop graph joins) is not a
Fold-Join and keeps its own lowering. Two signed-register lanes are only
combinable when both bind under the same `RegisterLaw`; a mismatch is refused
by `bind_signed_register` (`RegisterLawMismatch`, #1410) before any mask
exists, and `RelativeOffset`, `AxisPosition` and `Support` are never
interchangeable.

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

**Windows stay inside the tenant.** `MaterializedEdges` is 32 B, so two
16-byte windows cover it exactly: edges 0 and 1 in the first, edges 2 and 3 in
the second. Edge `k` is matched in window `k / 2`, half `k % 2`, with `care`
zero on the other edge's eight bytes. No window reaches a neighbouring tenant,
so no RBAC question arises and no 8-byte strided primitive is needed. (An
earlier draft placed one window per edge and flagged an overlap for `k = 3`;
that placement was unnecessary — Codex review on #1411.)

**Status (2026-10-08).** Measured and test-pinned by #1413
(`lance-graph-mask-risc/tests/resident_ce64_predicate.rs`), including the
threshold-as-union-of-patterns half. D-RPF-9's probe reaches the same equality
independently, from the quack side (`fold_join_probe`), and adds that the two
16 B windows sit on separate 64 B lines (row bytes 48..80).

### D-RPF-1 — Recipe eligibility as a mask

**Grounding (VERIFIED-IN-CODE).** `affordance_measurement_probe.rs`: eligibility
is `STATE[raw5] & PEARL[pearl3]`; every other CE64 field is swept and proven
irrelevant. Pearl3 is read by one recipe. The contract's
`EpistemicState5` has **24 meaningful codes** (`MEANINGFUL_STATES`, raw 0..23;
24..31 reserved and refused by `decode`). (The probe's own doc still says "ten
valid codes"; that predates D-EPI-MIG-0 and is stale — Codex review on #1411.)

**Claim to test.** For each recipe bit `r`, the set of admitted
`(raw5, pearl3)` pairs is enumerable at build time from the law tables, so
"rows where recipe `r` is eligible" is a mask over the D-RPF-0 projection,
with no per-row `u64` eligibility word materialised.

⊘ **Corrected 2026-10-08 (synergy pass, see § Synergies below).** This
claim first read "an `Or` of at most `24 × 8 = 192` ternary patterns". That is
the product form, and it is the wrong shape: `measure_state` is
`t.state[raw5] & t.pearl[pearl3]` (`affordance_law.rs:267-270`), so recipe
`r`'s mask is SEPARABLE, `Or(S_r) ∧ Or(P_r)`, at most `24 + 8` patterns before
minimisation. The difference is load-bearing because D-RPF-9 measured that a
strided predicate costs one full pass over the rows (~1 ms per 64K rows,
512-byte stride): 192 passes is ~0.2 s, 32 passes ~32 ms, and a minimised
cover fewer still. The compiler emits a minimised ternary cover of each set
(the D-RPF-0 threshold union, ≤ `width + 1` patterns, is one such cover), and
the reserved codes 24..31 are excluded from `S_r`, never treated as
don't-cares: `decode` refuses them, and a refusal is not "ineligible".

**Admission comes first, and the patterns cannot supply it.**
`measure_declared` projects the code through
`Epi5Declarations::project_state5(class, rail, generation, raw5, provenance)`
before any table is consulted, and refuses on an undeclared class/rail, a
generation mismatch, or untrusted provenance. None of those facts is in the
CE64 bits, so a pattern over bits 59..63 alone would admit rows that
`measure_declared` refuses. The compiled patterns therefore run `under` an
admission plane:

- the row's classid (key bytes 0..4) equals the declared class — an
  `EqU32Strided` predicate on the key;
- rail and generation are per-population constants of that declaration, fixed
  when the program is compiled, never per row;
- provenance is not stored in the edge. It is either a caller-supplied
  resident plane (trusted rows) or a precondition that the population was
  validated as a whole. If neither exists, the program refuses instead of
  running the patterns.

**Deliverable.** A pure compiler from the law's compiled tables to
`Vec<(pattern, care)>` per recipe (a build-time function, not a runtime
object), and the differential: compiled mask == per-row `eligible` bit, for
every recipe, on the probe's fixture.

**Falsifier.**
- can fire: dropping one admitted code from a recipe's pattern set must lose
  exactly the rows carrying that code;
- can stay silent: `OBSERVE_FOLD` (no requires, no forbids) must compile to
  all 24 meaningful codes across all eight Pearl projections, and nothing
  else;
- admission: a v1-provenance row, or a row of an undeclared class, with bits
  that WOULD match must be excluded by the admission plane, and removing that
  plane must make it match (the disable run).

### D-RPF-2 — Class census

**Question.** How much does the hotspot idea buy on real data? Count distinct
classes, do not assume them.

**Measure, on a real resident population (not a synthetic uniform one):**
1. distinct `(raw5, pearl3)` — the eligibility classes (bounded above by 192, per declared class; this bounds the CLASS count only — the pass count of D-RPF-1 is the separable cover size, ≤ 32);
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

**Cost evidence (D-SCF-CARE-PAIR-0, MEASURED 2026-10-09).** This is the same
key shape on a counted relation: a projection plus every link's resolution
context. Reuse pays once a relation is queried N times:

| instance | search/query | lookup | validate (fingerprint / key) | break-even N |
|---|---|---|---|---|
| H1 | 63.6 µs | 0.018 µs | 802 µs / 0.6 µs | 36 (24 with a versioned key) |
| H2 | 162.0 µs | 0.016 µs | 818 µs / 0.7 µs | 15 (9 with a versioned key) |

Almost all of the validation cost is content fingerprinting. A generation
counter in the key, which is this plan's `(classid, rail, generation)` row,
removes it. Falsifier template: one isolating fixture per key field, with
forced reuse past the refusal producing a wrong answer (H3 context, H4
no-repeat with equal lengths, H5 no-repeat alone).
Source: `.claude/board/entries/2026-10-09-d-scf-care-pair-0-crossing-counterfactuals.md`.

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

### D-RPF-9 — Fold-Join deforestation probe

**Measured.** `crates/lance-graph-quack/examples/fold_join_probe.rs`; numbers
and the full inventory in
`entries/2026-10-08-fold-join-deforestation-probe.md`. Summary:

- Resident ∧ resident → fold is already deforested (no slot, ~0.4 µs / 64K).
- Predicate → fold writes no population mask (tiled); for i32 lanes the two-
  predicate chain runs at roughly the per-row scalar loop's speed.
- Gating pays only at a high dead-word fraction (crossover ≈ 0.5 on this host);
  quack `lower` always gates and `lower_fused` never does.
- Strided predicates cost one pass over the rows each; merging two field
  predicates in one 16 B window into one ternary pattern halved the time
  (2.47 → 1.17 ms) with the same answer.

**Proposed rewrite contract (not implemented; each its own focused PR):**
R2 strided pattern merge (`(p1 & c1) | (p2 & c2)`, `c1 | c2` only when
`(p1 ^ p2) & c1 & c2 == 0`, else empty — never an OR of unmasked patterns);
R3 gate selection from a known dead-word fraction; gaps G1 strided `_under`,
G2 two-window strided match, G3 tile skip — backend-first in `ndarray::simd`.

## Synergies with the probes merged 2026-10-08 (added after the rebase)

Read from `entries/2026-10-08-{fold-join-deforestation-probe,
mexhat-bucket-cascade-probe, mexhat-df-choice, algebraic-recipe-probe,
phasor-trig-probe}.md`. Status of each line: WORKING-MODEL unless marked; none
is implemented, and none adds a primitive.

| probe finding | D-RPF phase | consequence |
|---|---|---|
| D-RPF-9 f4: a strided predicate is one full row pass; merging two field predicates of one 16 B window into one pattern halves the time (R2) | D-RPF-1 | Pass count, not pattern correctness, is the cost. The compiler must (a) use the separable form (VERIFIED-IN-CODE, correction above), (b) minimise each cover, (c) apply R2 to every `Match ∧ Match` on the same window — Epi5 (bits 59..63) and Pearl3 (40..42) sit in the same edge word, disjoint cares, so R2's conflict check always passes for them. |
| D-RPF-9 f5 + G1: gating a strided predicate skips nothing without `_under` | D-RPF-1 admission plane | The admission plane is correct but buys no time until G1 lands. Order it by cost: the classid `EqU32Strided` reads key bytes 0..4, in the SAME 64 B line as window 0 (row bytes 48..64); a fused row visit (G2's shape, extended to the key) would read both at once. |
| D-RPF-9 G2 two-window strided match | D-RPF-0, D-RPF-1 | Edges 2,3 (window 1) sit in line 1. A recipe reading edges in both windows pays two lines per row; G2 is the primitive that makes it one visit. |
| D-RPF-9 f3 / R3 + D-MHB-2 Q: choose fuse-or-gate per QUERY from an EXACT count, not a statistic | D-RPF-6 | Tile skip and gate selection use the gate's exact dead-word / dead-tile popcount, already printed by `fold_join_probe`. `RollingFloor` stays PARKED here, as in D-MHB-1. The crossover is a per-host pin. |
| D-MHB-1 R-MHB-1: a masked sum over a static integer weight template equals `Σ_b 2^b · (Count(P∧Pos_b) − Count(P∧Neg_b))`, exact, no value lane | D-RPF-4 | HYPOTHESIS: the T1 gap "LUT-weighted strided sum" has an exact formulation without a new primitive. Quantise `w = W(c)` to an integer LUT over the 8-bit confidence byte; each weight bit `b` is a SET of `c` values, i.e. a strided pattern cover on the C byte (bits 32..39). Then `Σw = Σ_b 2^b · Count(Adm ∧ c ∈ C_b)`, and `Σw·f` bit-slices `f` (bits 24..31) too, each `f_j ∧ c ∈ C_b` an R2-merged pattern in one edge word. Exact for the quantised semantics; the quantisation is the declared reading (A2: magnitude is a separate reduction). Cost is `bits(W) × 8 × |cover|` passes, so it is an ORACLE-GRADE route and a correctness unblock, not the fast path — the single-pass kernel stays the gap. The `c = 255` revision defect (`inf/inf`) must be a refused or clamped LUT entry, stated, not inherited. D-MHB-1 arm B (per-query weight lane, 26–150× slower) is the shape this avoids. |
| D-ART-1 f5: a static array indexed by enum discriminants is 1.2 ns against SipHash 27 ns, and cannot collide | D-RPF-3 | The read-set dedup key is small and enumerable per instruction (`isa::contracts` reads); key lookup is a static array over the packed projection, never a `HashMap`. |
| D-ART-1 §3: pushing a fold below a transform is wrong when the group key reads a transformed column | D-RPF-3 | Same guard shape as D-RPF-3's handle-context rule: the dedup key must include every input the result's handles resolve through, or equal keys yield unequal results. |
| D-ART-1 §3: moments are not sufficient; Population × {quantile, vertex, collide} → `Materialize` | D-RPF-4, capstone | Confirms that only folds with a merge law ride the resident path; a revision terminal is `(Σw, Σw·f)`, which has one, and nothing order-statistic shaped joins it. |
| D-PHT-1 u32-turns phase + LUT | D-RPF-7 | Weak. Both are "table instead of transcendental"; D-RPF-7's palette LUT rank question is unaffected. Recorded only so it is not re-derived. |

**What the probes do NOT license.** None of them executes a CE64 instruction
inside mask-risc (D-RPF-8 stands); the bit-sliced revision route reads raw
bits under a declared quantisation and never calls `forward` or `syllogize`.

**Measurement owed before D-RPF-1 ships:** the minimised cover size per
recipe for the real law tables, and the resulting pass count × 1 ms/64K, so
the D-RPF-1 PR reports its own cost rather than citing this table.

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
