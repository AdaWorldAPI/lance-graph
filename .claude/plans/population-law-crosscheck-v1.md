# population-law-crosscheck-v1 — which execution law survives two independent witnesses

> **Status:** ANALYSIS + source verification, plus one MEASURED test-only probe (D-PLX-1,
> §N, `crates/lance-graph-quack/tests/result_operand_probe.rs`). D-PLX-0..1.
> **Revised 2026-10-03 after review:** §E, §G, §H, §L and §M carry dated ⊘ notes; §N is new.
> **Inputs:** merged #1311 (`2f2b67c`, `crates/lance-graph-quack/tests/gremlin_parity.rs`,
> `.claude/plans/frontend-parity-witness-v1.md`) and an independent population-fold
> experiment over a real 20,845-row source file (reported to this session, not run here;
> its counts were re-derived from the CSV as an oracle, §B).
> **Board:** entry `entries/2026-10-03-population-law-crosscheck.md`.

Evidence grades used throughout: **MEASURED** (a test or run in this tree),
**STRUCTURAL** (equality of lowered values), **VERIFIED-IN-CODE** (read at a cited
location), **ANALYZED** (reasoning over verified code), **UNPROVEN**.

## A. Current main and what survives of #1311

`main` = `2f2b67c` (merge of #1311, on top of #1310). This branch was reset to it.

| #1311 claim | grade after the review caveats |
|---|---|
| Gremlin and SQL lower to the same `Query` for the supported shapes | **STRUCTURAL** — holds |
| Those queries' results equal DuckDB (`join_sum_country`, `join_group_count_country`) | **MEASURED** — DuckDB is an independent oracle here |
| Bulk-oracle agreement on the other shapes | **MEASURED, but not independent where adapter and oracle share a modelling choice** |
| Anchor rule: functional hop keeps the anchor; one fan-out re-anchors on the path population | **MEASURED** for the shapes tested (forward/reverse fk, one edge-table hop) |
| Bag vs set stay apart after a fan-out | **MEASURED** |
| "Second fan-out ⇒ barrier" | **REVISED** (§D): the cause is carrying a fan-in aggregate, not hop count |
| G4 "sum of a foreign value needs a primitive" | **MIS-SPECIFIED WITNESS** (caveat 0.1): the test used a `u32` field, so even the local sum cannot lower |
| `values()` handling | **DEFECT, shared by adapter and oracle** (caveat 0.2): `values(f)` leaves the cursor on the vertex, so `V(Line).values(Amount).emit()` returns vertices in both. No #1311 test uses that shape, so no MEASURED result depends on it. Agreement between adapter and oracle on it would prove nothing |

## B. The experiment's claims, checked against code

| claim | verdict | where |
|---|---|---|
| Counts O=20,845, I=20,842, S=18,559; the three n=2 keys; the PoS-per-spelling histogram | **re-derived**, identical (host Python oracle over `crates/deepnsm/word_frequency/academic_20k.csv`) | — |
| `GroupReduce{Count, Pair{hi,lo,stride}}` exists and builds no composite key column | **consistent** | `mask-risc/src/ir.rs` `GroupKey::Pair` ("Fused: no composite key lane is ever materialised"); ndarray `GroupKeyAddr::Pair`, `simd_masking_ops.rs:1558` |
| Sink sizes: 296,944 slots; Count 2.38 MB; moments 9.5 MB; temp key 1.19 MB | **consistent arithmetic**: 18,559·16 slots × 8 B / × 32 B (`PowerSums` is 32 B, const-asserted) / × 4 B (`u32`) | ndarray `simd_masking_ops.rs:2008-2025` |
| "GroupMomentsI32" | **no such symbol on main**. Moments exist only as the ndarray facade `masked_group_power_sums_i32{,_via,_pair}` → `[PowerSums]`. mask-risc has no moments fold or terminal, so the probe called ndarray directly | ndarray `:2059, :2094, :2133`; `grep` of mask-risc/quack/report `src` = 0 hits |
| n=2 recovers both observations from `(n, Σx, Σx²)` | **true** (e.g. `wastewater/n`: s=1587, 2q−s²=109², roots 848 and 739) | arithmetic |
| A Count sink can become a lane only by unsafe reinterpretation | **consistent**: `Out::I64(&mut [i64])` is the sink; `LaneRef` has `I32, U32, U64, Strided` and no `I64`; mask-risc and Quack are `#![forbid(unsafe_code)]` | `value.rs:44-55`, `ir.rs:16-35` |
| A `PowerSums` sink cannot be a lane at all | **consistent**: no `LaneRef` views records; `Strided` takes `&[u8]`, and `&[PowerSums]` → `&[u8]` needs `unsafe` | as above |
| Today's grouping cannot refold a completed Pair result by a projection of its key | **consistent**: every key form reads a lane (`Resident`, `Via`, `Pair`); none is a function of the row coordinate | ndarray `simd_masking_ops.rs:1558-1567` |

## C. The common execution law

1. **Execute over the population whose rows already carry the required multiplicity.**
   An anchor row is one unit of the answer, and multiplicity is never re-created by
   copying.
2. **Reach other data only through functional reads from the anchor row.** A read
   chain stays one pass as long as every step names at most one row.
3. **Switch anchors rather than build a target population**, when the new
   population's rows can reach every carried predicate through functional reads.
4. **A phase boundary is real exactly when the next computation needs, per row, a
   value that is a fold over several rows of an earlier computation** (a fan-in
   aggregate), and no resident population already carries that value or the
   ordering that would let it be folded in one pass.

Hop count, functional vs many-to-many, and graph vs relational do not appear in the
rule. They only decide which of 2, 3 and 4 applies.

## D. Anchor / population-switch law

- **The anchor stays** across any chain of functional reads. Today the IR reads one
  fk deep (`EqU32Via`, `GroupKey::Via`), so a chain of two is gap 1.
- **Re-anchoring is sound** when every new anchor row reaches every carried predicate
  through functional reads: from a parent to the children that reference it, or from
  a vertex to the edges whose `src`/`dst` reference it. Measured in #1311.
- **Repeated re-anchoring handles further fan-outs** when each next population holds
  an fk into the current anchor (A → children R1 → R1's own children R2). The carried
  predicates then sit two fks deep. That is **gap 1 (functional composition), not a
  barrier.** So #1311's "second fan-out ⇒ barrier" is wrong as a rule.
- **What breaks it** is a step whose new rows are reached by many old rows: A → R1 → R2
  where R1 and R2 are edge tables meeting at B. Each R2 row's multiplicity is the
  number of qualifying R1 rows that end at its `src`: a fold over R1 grouped by B,
  read by R2 through an fk. In set semantics it is a mask over B produced from R1 and
  read by R2's `Gather`, which is the produced-plane shape #1310 rules out. Either way
  a produced result becomes an operand (G2). The only exception is a resident
  population whose rows already are the composed paths, and that is rule 1 again.

## E. I → S root cause (the 1.19 MB key column)

- The question ("how many (spelling, PoS) identities per spelling") is a **presence
  fold over I's cells.** It is not a re-roll of I's counts: merging counts along the
  spelling axis gives observations per spelling (O→S, available directly), while the
  question counts *occupied cells*. Re-rolling a mergeable state is a homomorphism;
  counting present cells is not. **So I must be complete before S is computed: the
  dependency is real.**
- The **key column is not**: the spelling of cell `j` is a pure function of `j`, given
  the producing key (for `Pair{stride}`, `j / stride`). It had to be materialised
  because (a) the sink keeps no record of the key that produced it, and (b) no group
  key form computes a key from the coordinate (`GroupKeyAddr` reads lanes only).
  `Pred::Range` is the precedent for an index-derived predicate that reads no lane;
  there is no index-derived key.
- `lance-graph-report`'s `CellSpace` (`result.rs:35-210`) does keep `dims`, `domain`
  and `strides` with the produced cells, and re-rolls along a partial coordinate
  without moving the payload (`merged`). It only merges fold states, and it runs in
  host code.
- Layout note: with the source stored in (spelling, PoS) order, the presence fold
  becomes a run count in one pass (the `CountKeyRunsU32` law: an ordered resident
  projection replaces a seen-set). So this specific dependency is layout-avoidable;
  a predicate over completed sums (HAVING) is not.

**Classification: CURRENT API ARTIFACT** for the key column; the phase dependency itself
is **SEMANTICALLY NECESSARY** without an ordered projection.

> **⊘ Corrected 2026-10-03 (review thread on #1312).** The presence fold does NOT need a
> completed I. PoS has a bounded 16-value domain, so one pass over O grouped by spelling
> can OR a 16-bit presence bitmap per group (bit `pos`) and popcount it when finishing.
> A generic per-group distinct state is the same counterexample. The current IR lacks
> that fold state (`GroupFold` has `Count / MinI32 / MaxI32 / SumSymI32` only), so I→S
> is an **AGGREGATE-STATE LIMITATION, not a phase dependency.** The key-column verdict
> (API artifact) stands. Consequence: I→S is no longer a witness for the value seam; it
> witnesses only the key seam, and only when one insists on refolding I itself.

## F. Result-as-operand root cause

| result | physical form | typed form | readable without copy | operand / group key / predicate input today |
|---|---|---|---|---|
| `Keep` / filter | `Out::Mask(&mut [u64])` over the same rows | mask | yes | **yes, as a resident plane of the same table**: Quack's `GroupPlan` binds it as `planes.masks[filter_plane]` for phase 2 (`quack/src/lib.rs`, `GroupPlan` doc). Cross-table it must not be a `Foreign` plane (`Filter::Semijoin` precondition; #1310) |
| `ScatterOrU32` | `Out::Mask` over the target rows | mask | yes | only as the demanded result (`ir.rs`, survival condition) |
| `Count` / `GroupReduce` / `GroupSum*` | `Out::I64(&mut [i64])`, K slots | `i64` with seed-marked empties | yes, as a slice | **no**: no `LaneRef::I64`, no `i64` predicate. Consumed only by host code: `GroupHavingPlan::finish` loops over groups in Rust; `normalize_group_sink` maps seeds |
| `PowerSums` | `[PowerSums]`, 32 B records | `u64, i64, u128` fields | yes, as a slice | **no** lane view of a record field without `unsafe`; not reachable through mask-risc at all on main |
| report `CellSpace` | `values: Vec<Box<[i64]>>` + dims/strides | fold states with key metadata | yes | re-roll and present only, in host code; not an input to a mask-risc program |

The smallest demonstrated missing seam is: **make a completed K-sized result consumable
by a later pass.** Whether that means typed lane exposure, a checked narrow view, the
existing foreign/Via plumbing, or a small result descriptor is not established by
source reading alone. (For reference: ndarray has `gt_u64_to_mask` and friends but no
`i64` compares, and no lane views record fields such as `PowerSums`.) §N measures one
consumer.

**Where the boundary actually bites.** Quack rules HAVING deliberately: *"HAVING is not
a new primitive and not a population operation… the O(K) finalization over those sinks…
No O(N) mask crosses a program boundary"* (`quack/src/lib.rs`, `GroupHaving` doc). So
a phase B whose output stays in the K-sized result domain (HAVING, AVG's `avg_finish`, a scalar
statistical finish over sufficient statistics) is **already handled by design, as
K-sized host finalization**. (It sits in tension with the crate's "never evaluate a
`Program`" rule, but the crate states it as a choice, not a gap.)

The capability is missing only when **a produced K-sized result domain result must be read per row
by a later N-row pass**: an observation row reading its identity's folded value, or
an edge row reading the per-vertex count from the previous hop (§D). No host
finalization can serve that without either a host loop over N rows (the duplicate
evaluator) or a population-sized copy.

## G. G1 vs G2 — PARTIALLY SHARED

- **Shared:** a completed keyed result occupies a K-sized group (result) domain and may
  need to become an operand of a later pass. Both gaps sit behind that phase order. What
  mechanism exposes it is open (§F).
- **G1 (distinct): key metadata.** The next fold needs a key that is a function of the
  coordinate (the inverse of the producing `GroupKey`). This needs the producing key to
  travel with the result, and a coordinate-derived key form in the walker.
- **G2 (distinct): value typing.** The next program needs to read `i64` / record values
  as lanes and compare them.
> **⊘ Refined 2026-10-03 by the probe (§N): VALUE SEAM vs KEY SEAM.** The split is
> better named by what metadata each needs. **Value seam:** a later row already carries
> the key (an fk) and only needs the produced values addressable by it. §N shows that
> needs nothing beyond a lane of a supported type, *provided the result's coordinate
> space equals the fk's target space* (true for a `GroupKey::Lane` fold over that fk).
> **Key seam:** a later fold needs a component of the result's own coordinates (e.g.
> `Pair.hi`), which needs the producing key's geometry. Both sit behind the same phase
> order (the result must be complete); they need different, non-overlapping metadata.

- G2 that crosses back into an N-row pass is independently needed by a per-row read
  of an I value from F (experiment) and by the A → R1 → R2 traversal (frontend side,
  §D). G2 confined to the K-sized result domain (HAVING, I→S's present-cell count) is served today
  by host finalization. G1 is needed only by refolds along a key component.

## H. Multi-terminal vs multi-phase

| shape | workloads | needs |
|---|---|---|
| **multi-terminal** (one input, independent outputs) | AVG (`lower_avg`: SUM + COUNT, two programs today); HAVING phase A (one `GroupReduce` per aggregate); Count + moments over the same Pair key; the test-only `FoldDialect` `SUM − SUM` | more terminals per pass at most; no dependency |
| **multi-phase** (a produced result is a later input) | HAVING phase B; I→S presence fold (without ordered layout); F reading a value of I; A → R1 → R2 over edge tables; DISTINCT without an ordered projection | §F's operand capability |

Adding terminals to one `Program` removes none of the multi-phase dependencies: each
phase-B input is a *completed* fold.

> **⊘ Corrected 2026-10-03.** Remove "I→S presence fold" from the multi-phase row: it is
> a missing fold state (§E note). HAVING phase B stays multi-phase but is K-sized result domain
> finalization by design (§F), not the N-row consumer. The multi-phase case that
> remains is a completed result read per row by a later pass (F reading I; an edge row
> reading the per-vertex count of the previous hop; §N's probe).

## I. The five #1311 gaps, reclassified

| # | gap | old | new | evidence | smallest missing operation |
|---|---|---|---|---|---|
| 1 | composed functional read (two fks deep) | general gap | **CONFIRMED GENERAL GAP** (also absorbs re-anchored second fan-outs, §D) | IR and ndarray `*_via` are depth 1 (`eq_u32_via_to_mask`, `masked_group_*_via`); a composed fk lane is a population-sized copy | a depth-k composed read `table[fk_k[…fk_1[i]]]`, tile-local (#1308 option (a)) |
| 2 | barrier + workspace after a fan-out | general gap | **SPLIT:** fan-out onto a population that holds an fk to the anchor = **POPULATION RE-ANCHOR** (+ gap 1); a fan-in aggregate carried to the next population = **PHASE-BOUNDARY GAP** (= G2) | §D | §F |
| 3 | ordered compare through an fk | general gap | **SIMPLE MISSING OPERATOR** | only `eq_u32_via_to_mask` exists; ordered `*_to_mask` kernels exist for `i32`/`u8`/`u64` | an ordered via predicate (ndarray kernel + `Pred` variant) |
| 4 | sum of a foreign value | general gap | **MIS-SPECIFIED WITNESS**; by source a **SIMPLE MISSING OPERATOR**, not measured | witness used `u32` `Country`; the local `i32` sum exists (`Terminal::MaskedSumI32`) and no terminal or ndarray kernel sums `table[fk[i]]` (Terminal enum read in full; ndarray `masked_sum_*` list) | `masked_sum_i32_via`. Its composition route (Count by fk, then a dot product) would itself need G2 |
| 5 | ordered compare on `u32` | general gap | **SIMPLE MISSING OPERATOR** | ndarray has `gt/lt/ge/le` for `i32`, `u8`, `u64` but not `u32`; `Pred` has none for `u32` | `{gt,lt,ge,le}_u32_to_mask` + `Pred` variants |

## J. Existing production precedent

1. **Quack `GroupPlan` (forest lowering):** a produced mask re-bound as a resident plane of
   the same table for phase 2. This is the one mask-risc-executed result dependency, and
   it is mask-typed only.
2. **Quack `GroupHavingPlan::finish`:** produced `i64` sinks → host loop → K-bit masks,
   ruled as O(K) finalization by design. The closest precedent for a typed result
   consumed by later computation. It never feeds an N-row pass.
3. **lance-graph-report `CellSpace`:** a produced result that keeps its key metadata and
   re-rolls by coordinate projection without moving data (host code, mergeable states
   only).

`FoldDialect` (test-only) and R2IL were not used as precedent.

## K. Materialization consequence

- **Fusion:** unchanged. Pointwise predicates fuse into one pass.
- **Functional composition:** the only way to reach other data without a new
  population. Its depth limit (1 today) is gap 1, and it also caps multi-fan-out
  re-anchoring.
- **Population re-anchor:** replaces target-population materialisation for any number
  of fan-outs onto populations that reference the current anchor.
- **Phase boundary:** only for fan-in aggregates consumed per row later, and avoidable
  for distinctness when an ordered resident projection exists.
- **Workspace:** a phase-B input is K-sized (the group universe), private to the
  computation, and never a `Foreign` or alpha plane.
- **Demanded output:** unchanged. A scattered or folded result is legal as the answer;
  as an operand it needs the phase rule.

## L. Smallest next step — MORE PROBE NEEDED

The two witnesses agree on where the boundary is (§C rule 4), but not yet on what the
operand capability must be: a wider lane kind, or a typed result handle that also
carries provenance or key metadata. So no primitive is chosen yet.

**The single falsifier** (test-only, in `lance-graph-quack/tests/`, over the DuckDB
fixture): *"lines whose partner has exactly v posted lines"*.

1. Phase A: `GroupReduce{Lane(partner_id), Count}` over posted `line` rows → K = 64 sink.
2. A checked narrowing copy of the sink to `u32` (K-sized, test-only, stated as such).
3. Phase B: a `line` query with `EqU32Via(partner_id, <that lane>, v)` → `Count`,
   compared with a host oracle for every v that occurs, plus one v that does not.

- **Passes with only the narrowing** → existing per-row reads suffice once the values are
  in a supported lane type; no result-handle architecture is indicated for that
  consumer. Which mechanism should remove the copy stays open.
- **Needs more** (seed-marked empties need the producing fold's identity, or the
  caller cannot express the phase order) → a typed result handle is justified, sized by
  exactly what was missing.

The same shape is the experiment's F→I read and the frontend side's A → R1 → R2 weight
read, which is why it is the one probe that tests both witnesses.

> **⊘ Resolved 2026-10-03: run as specified; OUTCOME A (§N).** It passed with only the
> narrowing. A review note on #1312 is correct that the copy discards the `i64` type and
> any provenance, so the probe cannot distinguish an `i64` lane from a result handle;
> it shows instead that *this consumer needs neither*: once the values are in a
> supported lane, existing machinery composes. Scalar width is classified separately
> (§N, H).

## N. Measured: the result-as-operand probe (D-PLX-1, 2026-10-03)

File: `crates/lance-graph-quack/tests/result_operand_probe.rs` (test-only, 4 tests),
over the DuckDB fixture on `main` `2f2b67c`.

**Path.**

| step | what | machinery |
|---|---|---|
| phase 1 | `COUNT(*) GROUP BY partner_id` over posted lines | `Agg::GroupReduce{Local(partner_id), Count}` → `Terminal::GroupReduce` → `Out::I64`, K = 64 |
| knife | checked `i64 → u32` copy of the sink, panics if a value does not fit | test-only, K-sized |
| phase 2a | lines whose partner has exactly `v` posted lines, for every `v` that occurs and one that does not | `Filter::EqU32Via{fk: partner_id, key: ForeignLane(0), v}` → `Pred::EqU32Via` → `eq_u32_via_to_mask` |
| phase 2b | histogram of lines by their partner's count, ONE program | `GroupAddr::Via{fk: partner_id, key: ForeignLane(0)}` + `Count` → `masked_group_count_u32_via` |
| oracle | plain loops over the fixture vectors | no Quack, no mask-risc |

**Result: PASS.**
- N = 4,096 lines, K = 64 partners, 26 distinct partner counts.
- Both phase-2 consumers match the oracle on every bucket.
- An absent count matches 0 lines.
- The per-`v` totals add up to every posted line.

**Load-bearing, each turned the test red:**
- the phase-2 predicate with the via read removed;
- the phase-2 key read locally instead of through the lane;
- the oracle's count for one partner altered.

**Also asserted in-suite:**
- a rotated lane (each partner given its neighbour's count) is caught;
- a corrupted single count is caught;
- the same lane reached through a different local column (`cost_center`) gives a different histogram.

**Allocation / materialization.**
- Phase 2 execution allocated **0 bytes** (counting allocator).
- Storage beyond the inputs: the sink (512 B = K × 8), the narrowed lane (256 B = K × 4) and the histogram sink (496 B).
- All of it is O(K); nothing is O(N).
- Removing the copy leaves only the sink the fold already produces.
- No host per-line lookup exists in the path. Phase 2b is one `execute_into`; phase 2a loops in host code over the 26 values only, as verification.

**Value vs key seam.**
- The consumer needed no key metadata. Its only requirement is that slot `j` of the
  result means partner row `j`, which holds by construction for a `GroupKey::Lane`
  fold over the same fk.
- A result keyed by `Via` or `Pair` would not satisfy that. Reading it per row needs
  the producing key, which is the key seam.

**CellSpace cross-check** (`lance-graph-report/src/result.rs`).
- It preserves:
  - per-dimension `CoordSpec` and domain;
  - the physical layout (`Dense{strides}`, or `Sparse{coords, index}`);
  - the fold states;
  - one `i64` value column per state.
- That covers the key seam for re-rolls by merging, in host code.
- It does not offer its values as a lane to a mask-risc program, so it does not cover
  the value seam.

**Scalar width: B.** The existing `u32` lane was sufficient after a checked conversion.
A first-class wider type would remove only the copy, and the copy exists because the two
sides use different widths:
- the Count fold writes `&mut [i64]` (ndarray `masked_group_count_u32`);
- the readers take `table: &[u32]` (`eq_u32_via_to_mask`, `masked_group_count_u32_via`).

The width can be met on either side:
- a `u32` (or `u64`) Count sink, since a count is non-negative and bounded by the row count;
- `i64` tables in the `*_via` readers.

Which side is a design choice not settled here. An `i64` lane is not implied.

**Verdict: OUTCOME A — a typed lane is enough for this consumer.** No result-handle
architecture is indicated. Measured: once the values sit in a supported lane type, the
existing foreign/Via plumbing composes. Not measured: which mechanism should remove the
copy (typed lane exposure, a checked narrow view, a different sink width, or a small
result descriptor).

**One candidate for removing the copy (not implemented, not chosen).**
- **Where:** `lance-graph-mask-risc` (`value.rs` `Out`, `exec.rs` `Terminal::GroupReduce`) and the ndarray count kernel.
- **What:** let `GroupFold::Count` write a `u32` sink (`Out::U32`), with the row bound checked the way `MASKED_SUM_I32_MAX_ROWS` is.
- **Result:** the sink is then directly a `LaneRef::U32` for the next phase.
- **Test:** this probe with the narrowing removed.
- **Negative test:** a Count over more than `u32::MAX` rows is refused, not wrapped.

Provenance (a produced lane passed through `Foreign`) needs the same ruling #1310 gives
planes: computation-private workspace, never resident authority. That is a rule, not a
type.

**Not done:** the optional three-population witness (A → R1 → R2 via fks). The
fixture has no table that references `line`, and it was not worth building one here.

## M. Downstream handover (domain-neutral)

```text
EXECUTION FACTS SURVIVING BOTH INDEPENDENT WITNESSES
1. Execute over the population whose rows already carry the answer's
   multiplicity; one row = one unit of the answer.
2. Reach other data only by functional reads (each step names at most one
   row). Today's depth is 1; depth >1 is an open general gap.
3. On fan-out, switch to the population whose rows are the paths, carrying
   predicates as functional reads. This repeats for any number of fan-outs
   onto populations that reference the current one (cost: read depth).
4. A phase boundary is real only when a later computation needs, per row,
   a fold over several rows of an earlier computation, and no resident
   population (or ordered projection) already carries it.
5. A completed keyed result occupies a K-sized group domain and may need
   to become an operand of a later pass. It is computation-private
   workspace, never a new identity or an authoritative population.
6. Re-rolling a mergeable fold along its key needs no phase boundary (fold
   the source with the coarser key, or merge cells). Presence over a bounded
   domain needs no phase boundary either: it is a per-group OR state.
7. A completed K-slot result read per row by a later N-row pass composes
   with existing reads once its values sit in a supported lane type
   (measured: 0 bytes allocated in the later pass; extra state O(K)). This
   requires slot j of the result to mean target row j of the reading
   reference.
8. A key computed from a result's own coordinate needs no stored column;
   materialising one is an API artifact.
9. Open, small: ordered compares on u32, ordered compares through a
   reference, and summing a value through a reference.
10. Value reuse needs no key metadata; refolding a result by a component of
    its own key does (key geometry). Open: the producer/reader width mismatch
    (integer sink vs 32-bit reader); no result handle is indicated.
```
