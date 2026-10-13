# d-cmr-concept-chain-normalization-v1 — preserve the equation, not the intermediate result

**Status:** PROPOSAL (D-CMR-0..9). Only D-CMR-0 (this document and its board
rows) is in the PR that adds it. No code, no dependency.
**Revision 2026-10-13:** the normalization gets one new IR, a set-theory IR
(§3). This reverses point 3 of the concept-chain rule in the reset entry
("no new universal IR"); §3.1 is the insufficiency proof that justifies it.
**Revision 2, 2026-10-13 (aligned with the eight-point ruling):** the set IR
is the shared *semantic domain* of normalization, not a third frontend
syntax. Quack's `Filter`/`Query` stays the home of Quack compositions and NAL
stays on R2IL. What is shared is the rule catalogue: identities, their
preconditions and their cost effects, plus the set values they compute over.
**DECISION:** the IR describes and never executes. Thinking in abstract
relations before materializing is the point; what #1524 removed was a layer
that executed early. §3.1 justifies the IR; it is not a review gate.
**Date:** 2026-10-13. **Repo:** lance-graph (Quack, mask-risc, core NAL module).
**Parents:** D-CMR-RESET (`entries/2026-10-12-session-reset-inventory.md`,
root cause and the eight-point concept-chain rule), #1524, #1532.

## 1. Definition

A method in a pure chain contributes an unresolved relation, never a computed
intermediate. The chain stays unresolved until normalization has seen every
relation in it. Normalization applies proven identities; only the irreducible
remainder is lowered, once, into a `Compiled` program.

The test case is `(a+b)−a`. It does not mean "accumulate `a+b` in a wider
register". It means `a+b` never has to exist: the identity leaves `b`. A wider
accumulator only repairs a step that did not need to happen.

The goal is not faster or more precise intermediate steps. It is to find which
steps survive once the chain is simplified.

## 2. Inventory: normalization that exists today (VERIFIED-IN-CODE, `c157c0a`)

| mechanism | where | what it does | limit |
|---|---|---|---|
| `fold_ranges` | quack `lib.rs:1918`, called by `lower_with` (`:1956`), so by every Quack lowering | `Range ∧ Range → Range` via `max(lo)`, `min(hi)` | same `col`, siblings of one `AND`; no `OR`/`NOT` crossing |
| survivor skip, `gate_walk` + `hoist_gate_subset` | quack `:2026`, `:2010` | evaluates the rest of an `AND` only under an established gate | one gate per `AND` path |
| `sql_where` | quack `:549` | rewrites a filter into SQL three-valued `WHERE` for listed nullable columns | a semantic guard source, not a simplification; NULL literals, `COALESCE`, nullable arithmetic out of scope |
| `fuse` / `emit_fused` | mask-risc `fuse.rs:194`, quack `:2314` | any Boolean subtree over ≤ 3 leaves → one ternlog table | `lower_fused` only |
| lowering recognition | mask-risc `ir.rs:1349` | `Range` (`Count`/`Any` as `hi − lo`), `Ternlog`, `TernlogKeep`, `Tern2`, `Tern3`, else `Tiled` | `Range` folds only `Count` and `Any` |
| `Compiled` / `execute_compiled` | mask-risc `ir.rs:1388`, `exec.rs:1469` | recognize once, execute many times | — |
| `execute_extent` | mask-risc `exec.rs:1393` | runs a program over a row range | **Quack never emits an extent** (`lower_with` returns only a `Program`; no "extent" in `quack/src/lib.rs`). NAL uses it by hand via `bound_aperture` (`ordered_lane.rs:344`): 127 ns as extent vs 195 µs as an ANDed mask at 1,048,576 rows |
| aggregate laws | quack `lower_avg` (`:1522`, SUM+COUNT); `Terminal::merge_group_sink`; `PowerSums::checked_merge` | combine partial aggregates by each fold's own law | — |
| arithmetic semantics of folds | ndarray `masked_sum_i32` (exact `i64` over `i32` rows); keyed sums `wrapping_add` | fixes which cancellations are exact | no value-arithmetic IR anywhere (NAL §4 C, D-SQLP nullable arithmetic: OPEN) |

So `(a+b)−a` has no home today at the value level. It appears first at the
aggregate level: `Sum(A) − Sum(A∧B) = Sum(A∧¬B)`.

## 3. The set-theory IR

### 3.1 Why a new IR (insufficiency proof, VERIFIED-IN-CODE at `c157c0a`)

Everything set-like in the tree is either a predicate or a materialized bitset:
- Quack's `Filter` is predicates, lowered straight to masks. `fold_ranges`
  yields at most one interval per column, never merges ranges under `Or`, and
  its doc names `col` as the only order identity.
- `EvidenceMask` (`revision.rs:31`, `union`/`intersection`/`difference`) and
  mask-risc masks are set algebra after materialization.
- `WideFieldMask` (`intersect`/`difference`) is sets of fields, not of rows.
- No interval-set type exists anywhere in `crates/`.

Four things therefore have no home:
1. **The universe of a set:** which proven ordering (coordinate basis) of
   which population it lives in.
2. **An intersection result as a value** before materialization: empty, one
   point, one interval, or several disjoint intervals.
3. **Mappings between universes:** a hop is the preimage `f⁻¹(S)`, not a
   stacked key read.
4. **Measures as functions of a set:** `Count(S)`, `Sum_x(S)`, `Exists(S)`,
   computed from the set's description where that description is known.

The IR exists only to hold these four. It is not a second method algebra and
not a second executor: execution stays in mask-risc. This is enforced by a
test, not by convention: the IR module holds no row data and calls no
mask-risc execution function; its only output is a normalized description
handed to lowering.

### 3.2 The IR

```
U_b                    universe over basis b (a proven ordering of one population)
S ::= ∅
    | [lo,hi)_b                  interval; a point is [p, p+1)
    | Runs_b{[l1,h1), …}         sorted, disjoint intervals (one scanline)
    | Rect_{b1×b2}               product of intervals over a composite coordinate
    | Res_b(p)                   residual: an opaque predicate over basis b
    | S ∩ S | S ∪ S | U_b ∖ S
    | f⁻¹(S')                    preimage under a proven mapping b → b'
M ::= Count(S) | Sum_x(S) | Exists(S)
```

- **Normal form:** `Runs_b ∩ Res_b`. Interval sets are closed under `∩`, `∪`
  and `∖` at a cost linear in the number of intervals, with no row read; `∅`
  is a proof that the answer is 0. The residual stays opaque.
- **Edges, not content.** A set is stored by its boundaries. Interval-only
  operations scale with the number of edges, never with rows. Evaluating a
  residual and lowering a rectangle (one run per `hi`) scale with the rows or
  coordinate extent they visit.
- **Two dimensions.** A condition on `hi` and one on `lo` of a composite
  coordinate (`hi·stride + lo`, the `_pair` family) form a rectangle. It
  becomes 1-D runs only at lowering: one run per `hi` in row-major order, or a
  few aligned runs in Morton order (`mask_shift_morton`, the HHTL prefixes).
- **Limits the IR enforces:** two-valued only (three-valued logic is resolved
  by `sql_where` into the TRUE set first; complements only over two-valued
  sets); sets, not bags (preimage gives a set; the image of a many-to-one
  mapping is refused and stays a physical aggregation); intersection only
  within one universe, otherwise through a proven `f`, otherwise residual.
- **Home:** the algebra lives in `lance-graph-contract`, next to `ClassView`,
  and needs no dependency (the contract's `Cargo.toml` stays dependency-free).
  Lowering lives in core, which depends on mask-risc, Quack and R2IL (the
  other session's step (b)); it is never in the contract.
- **Frontends keep their representations.** A normalization pass over Quack's
  `Filter` (and later over NAL's R2IL program) computes set values in this
  domain with the shared rules: ranges on a proven basis become `Runs`,
  everything else stays residual. The frontend tree is rewritten to the
  irreducible remainder; neither frontend implements intervals of its own.

### 3.3 The lithography reading

A chip is designed as set algebra over polygon edges, entirely before any wafer
is exposed. A transistor exists where poly ∩ active is non-empty; a via where
metal1 ∩ via ∩ metal2 is. Layout tools compute layer booleans on edges with a
scanline, at a cost in edges, never in area. Rasterization happens once, in the
mask writer, at the end; exposure processes only the final mask.

| lithography | IR |
|---|---|
| wafer and its coordinate frame | `U_b` |
| one mask layer | a set `S`, stored as edges |
| device = layer intersection | `∩`, possibly empty, one point or many |
| positive / negative resist | `S` / `U_b ∖ S` |
| etch / deposit | `∖` / `∪` |
| overlay alignment between layers | intersection only within one basis or through a proven `f` |
| layer booleans and design-rule checks | normalization |
| optical proximity correction tables | the rule catalogue (§3.5) |
| rasterization in the mask writer | lowering to mask words, once |
| exposure and etch | execution in mask-risc |
| minimum feature size, writer grid | one row; 64-row words (an unaligned edge costs an edge merge, `execute_extent` doc) |

The withdrawn method layer exposed and etched each layer before the design was
complete, then tried to combine the etched wafers.

### 3.4 Measurement levels (the laser-filter reading)

A measure is light through a stack of masks onto a detector: light passes only
where every mask is open, so the stack forms the intersection and the detector
integrates it. Three levels, cheapest first; lowering picks the first that
applies:

| level | when | how | in the tree |
|---|---|---|---|
| 1. geometry, no light | no residual | `Count(Runs) = Σ(hᵢ − lᵢ)`; `Exists = Runs ≠ ∅` | `Lowering::Range` (one interval, `Count`/`Any`) |
| 2. optics, no etch | residual present, caller needs a measure | residual predicates fused into one filter, one pass per run as extent, nothing written | **resident planes only:** `mask_ternlog_popcount`, `Lowering::Ternlog` → `Count`, `execute_extent`. A residual *comparison* is a `Pred`, which `fused_ternlog` declines (`ir.rs` doc), so it takes `Lowering::Tiled` and writes scratch. Predicate fusion is missing (part of D-CMR-2) |
| 3. etch | caller needs the set itself | materialize the mask for the surviving runs only | `Keep` / `Out::Mask` |

`Sum_x(S)` is a graded filter: the existing masked sum fold over the runs as
extents (level 2). A prefix lane would move it to level 1; it stays a separate,
measured candidate.

### 3.5 Rule catalogue (the Tafelwerk) over the set IR

A rule is an identity, not a stored result: pattern, semantic preconditions,
rewrite, proof (independent oracle plus a negative control), cost effect,
scope. The key is the pattern **and** its preconditions. Rules are applied
until none fires or a step budget is reached, and none may increase the size of
the expression. Heuristics order rules; they never replace a proof.

The mechanism has three known parts, used in this order and only as far as
measured need goes: **constraint propagation** (narrow bounds, detect `∅`, the
interval part of `fold_ranges` generalized), **term rewriting** with the
catalogue, and **cost-based selection** among valid forms. Equality saturation
(egg, egglog) keeps competing forms instead of committing to one; it is a
research reference (D-CMR-8), justified only if a bounded local rewriter
measurably commits to worse forms. Rules found automatically (Ruler) are
candidates only, admitted after validation under the type, overflow and NULL
conditions of §4.

## 4. Identity matrix with semantic conditions

| identity | legal when | illegal when |
|---|---|---|
| `(a+b)−a → b` | exact integers; wrapping two's complement (a group mod 2⁶⁴) | checked arithmetic (removes an overflow the original would report); IEEE stepwise rounding (unless an exact-expression contract is declared) |
| `Sum(A) − Sum(A∧B) → Sum(A∧¬B)` | exact `i64` sums of `i32` within the row bound; wrapping sums | the `_sym` family (`SYM_EMPTY_I64` marks "no row", not a value); three-valued `B` |
| `⋂[lo,hi) → [max lo, min hi)` | same proven coordinate basis | different orderings; a bound past `n_rows` |
| `Count([lo,hi)) → max(0, hi−lo)` | valid interval on the population's own row order | bag after fan-out |
| `Count(¬A) → N − Count(A)` | two-valued `A`, fixed universe `N` | `A` may be UNKNOWN; `N` not the same population |
| `Sum([lo,hi)) → P[hi] − P[lo]` | a prefix lane exists, cannot overflow, matches the snapshot | the population changed since `P` was built |
| `A∧A → A`, absorption, De Morgan, ≤ 6 planes → truth tables | two-valued planes | three-valued leaves without `sql_where` |
| `hi·stride+lo` as one coordinate | `lo < stride` (enforced: ndarray `group_of` drops `lo ≥ stride`), no overflow | otherwise |
| NAL truth laws | the W4 law representation, parity in codes | before W4 |

**Numeric contract today:** mask-risc sums are exact (`i32` into `i64`, row
count bounded so the total cannot overflow, `ir.rs:296`, `:461`); there is no
floating point on the surface. The IR therefore assumes exact integers and
refuses any other operand. A per-method declaration of numeric rules is added
when the first floating-point or quantized column arrives, not before.

## 5. Symbolic preservation is not numerical precision

- **Exact semantic preservation:** the rewritten form gives the same result
  under the original contract. Allowed silently.
- **Explicit semantic refinement:** a newly declared exact-expression contract
  delays rounding or quantization to a named boundary. Allowed only where
  declared, never inferred from "the result looks more accurate".

## 6. Implementation path

| D-id | step | gate / falsifier |
|---|---|---|
| D-CMR-1 | **Set IR core** in the contract: `Runs` with `∩ ∪ ∖` and complement, `Res`, measures; pure, no lowering | algebra laws (closure, distributivity, De Morgan, `∅`) checked against materialized bitsets as oracle on random and edge-case runs |
| D-CMR-2 | **Lowering** to mask-risc: runs → extents, residual → predicates within each run, the three measurement levels. Level 2 for residual comparisons needs predicate fusion in mask-risc (`fused_ternlog` declines `Pred` today) | == row oracle; a `Count` with no residual reads zero rows (instrumented); disabling level 1 or 2 keeps answers |
| D-CMR-3 | **Quack normalization pass:** over `Filter`, computing set values (ranges on a proven basis → `Runs`, other leaves → `Res`, `sql_where` first). Extent hoisting falls out of this (Experiment C) | answers identical to today's `lower` path on the quack suite; a range on another basis stays residual (negative control) |
| D-CMR-4 | **Two dimensions:** `Rect` over composite coordinates, linearized row-major or in Morton order | rectangle `∩` == bitset oracle; run count within the stated bound |
| D-CMR-5 | **Preimage** `f⁻¹(S)` under a proven mapping (the hop) | == row oracle; a many-to-one image is refused |
| D-CMR-6 | **Rule catalogue and fixpoint driver**: complement, relational difference `Sum(A) − Sum(A∧B) → Sum(A∧¬B)` | Experiments A, D, E; each guard removal fails |
| D-CMR-7 | **NAL normalization pass:** aperture → `Runs` over the R2IL program, R2IL route unchanged (after NAL lands in core) | NAL W2 tests unchanged |
| D-CMR-8 | **Equality-saturation probe** (`egg`, `egglog` as references only) | only if rules compete; compared against D-CMR-6; no dependency without evidence |
| D-CMR-9 | **Benchmark** (section 8) | answers asserted before timing |

Compile-time normalization is separate from runtime binding: a parameter
known only per call (W3's range) is bound per call with a small boundary
calculation, never a population scan, a materialized mask or a re-lowering.

Prefix lanes stay a separate, measured candidate (ordering, overflow, NULL,
update and maintenance laws), never a precondition for `SUM`.

## 7. Experiments, each with an independent oracle and a negative control

- **A. Cancellation.** At the aggregate level: `Sum(A) − Sum(A∧B)` against
  `Sum(A∧¬B)` and an exact `i128` oracle. An eager variant that deliberately
  accumulates into `i32` is the counterexample that shows the loss; it is not
  a model of today's fold, which accumulates exactly into `i64` within the row
  bound. Under checked arithmetic the rewrite must refuse.
- **B. Range intersection.** k compatible ranges intersect before any row is
  read. Baseline: `fold_ranges` and `Lowering::Range` (tests at quack
  `lib.rs:5204–5248`).
- **C. Mixed.** A sorted-coordinate range plus an unsorted value predicate:
  the range becomes `Runs` and the extent, the predicate stays `Res` and is
  measured at level 2 (D-CMR-2, D-CMR-3).
- **D. Composition.** Two identities on one expression, the second enabled by
  the first (intersection, then `Count = hi − lo`).
- **E. Invalid rewrites.** NULL, overflow, incompatible coordinates, IEEE
  rounding, impure operations: each refuses or keeps the original.

Every rule: disabling the optimization leaves the answer; removing the
required semantic guard produces a failing counterexample.

## 8. Benchmark plan

Three arms on the same data: eager materialization (one mask per link, AND,
fold), the normalized chain, and the plain row loop (the performance floor:
the withdrawn layer measured 430–480 µs against 283 µs). Report rows scanned,
bytes materialized, preparation cost, per-call boundary cost and latency;
median of 101, tier named from the build's own report, answers asserted first.

## 9. Gap ledger

| item | state |
|---|---|
| range intersection within one `AND` | SHIPPED (`fold_ranges`) |
| `Count`/`Any` over a range without row reads | SHIPPED (`Lowering::Range`) |
| ≤ 6-plane Boolean → truth tables | SHIPPED (`Ternlog`/`Tern2`/`Tern3`) |
| compile once | SHIPPED (`Compiled`) |
| range as execution extent | SHIPPED in mask-risc and NAL; **missing in Quack** (D-CMR-3) |
| fused filter + count without writing a mask (level 2) | SHIPPED for resident planes only; **missing for residual comparisons** (`fused_ternlog` declines `Pred`, so they run tiled) — D-CMR-2 |
| sets with a universe, interval sets as values | OPEN (D-CMR-1) |
| multi-run extents, two-dimensional sets, preimage | OPEN (D-CMR-2, -4, -5) |
| chains composed before lowering | OPEN (D-CMR-3, -7) |
| repeated rule application, complement, relational difference | OPEN (D-CMR-6) |
| prefix aggregates | candidate |
| value-arithmetic IR (`a+b` as an expression) | OPEN, shared with NAL §4 C and D-SQLP |
| equality saturation | research (D-CMR-8) |

## 10. Ownership

- D-CMR-1 (pure algebra in the contract) can start now.
- D-CMR-2 lives in core and needs core's dependency on mask-risc, Quack and R2IL (step (b)); no contract dependency is needed.
- D-CMR-3..6 touch quack; D-CMR-7 waits until NAL lands in core.
- Owner to be assigned.
