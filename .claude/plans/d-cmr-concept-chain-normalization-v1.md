# d-cmr-concept-chain-normalization-v1 — preserve the equation, not the intermediate result

**Status:** PROPOSAL (D-CMR-0..7). Only D-CMR-0 (this document and its board
rows) is in the PR that adds it. No code, no dependency, no new IR.
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

## 3. Rule catalogue (the Tafelwerk), in existing representations

A rule is an identity, not a stored result. Each entry has six fields:
pattern, semantic preconditions, rewrite, proof (independent oracle plus a
negative control), cost effect, scope. The lookup key is the pattern **and**
its preconditions: `Count(¬A)` under three-valued logic is a different entry
from `Count(¬A)` under Boolean logic.

- **Home.** Quack-bound chains: passes over `Filter`/`Query` in the quack
  crate, next to `fold_ranges`. Global NAL methods: rules applied while the
  core NAL module emits its R2IL procedure. No new IR. A `Rule` type exists
  only if the fixpoint driver (D-CMR-3) cannot be written over plain passes;
  the insufficiency is recorded first.
- **Application.** Repeat until no rule fires or a step budget is reached.
  Every rule must not increase expression size, so the loop terminates.
- **Heuristics** order the rules; they never replace a proof.

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

## 5. Symbolic preservation is not numerical precision

- **Exact semantic preservation:** the rewritten form gives the same result
  under the original contract. Allowed silently.
- **Explicit semantic refinement:** a newly declared exact-expression contract
  delays rounding or quantization to a named boundary. Allowed only where
  declared, never inferred from "the result looks more accurate".

## 6. Minimal implementation path (no second algebra)

| D-id | step | gate / falsifier |
|---|---|---|
| D-CMR-1 | **Extent hoisting.** A top-level range conjunct that `fold_ranges` produced becomes the `execute_extent` extent; the rest of the filter runs as the residual (Experiment C) | answers == unhoisted == row oracle; disabling hoisting keeps answers; a range on a different coordinate basis must not hoist (negative control) |
| D-CMR-2 | **Lazy binding.** A bound method yields a `Filter` fragment plus aggregate, composed into one `Query` before lowering; no per-link execution | chain of k methods lowers once; no scratch allocated per link |
| D-CMR-3 | **Fixpoint driver** over the existing passes, with a step budget and size non-increase | Experiment D: a second rule fires only because the first did |
| D-CMR-4 | **Complement rule**, guarded on two-valued leaves | removing the guard fails on a nullable column |
| D-CMR-5 | **Relational difference rule** `Sum(A) − Sum(A∧B) → Sum(A∧¬B)` (Experiment A at the aggregate level) | equal to the exact `i128` oracle; the `_sym` and checked variants refuse |
| D-CMR-6 | **Equality-saturation probe** (`egg`, `egglog`, as references only) | runs only if two rules compete and order changes the result; compared against D-CMR-3 with extraction cost taken from the real lowerings; no dependency without that evidence |
| D-CMR-7 | **Benchmark** (section 8) | answers asserted before timing |

Prefix lanes stay a separate, measured candidate (ordering, overflow, NULL,
update and maintenance laws), never a precondition for `SUM`.

## 7. Experiments, each with an independent oracle and a negative control

- **A. Cancellation.** At the aggregate level: `Sum(A) − Sum(A∧B)` against
  `Sum(A∧¬B)` and an exact `i128` oracle; an eager variant accumulating into
  `i32` shows the loss. Under checked arithmetic the rewrite must refuse.
- **B. Range intersection.** k compatible ranges intersect before any row is
  read. Baseline: `fold_ranges` and `Lowering::Range` (tests at quack
  `lib.rs:5204–5248`).
- **C. Mixed.** A sorted-coordinate range plus an unsorted value predicate:
  the range becomes the extent, the predicate stays residual (D-CMR-1).
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
| range as execution extent | SHIPPED in mask-risc and NAL; **missing in Quack** (D-CMR-1) |
| chains composed before lowering | OPEN (D-CMR-2) |
| repeated rule application | OPEN (D-CMR-3) |
| complement, relational difference | OPEN (D-CMR-4, -5) |
| prefix aggregates | candidate |
| value-arithmetic IR (`a+b` as an expression) | OPEN, shared with NAL §4 C and D-SQLP |
| equality saturation | research (D-CMR-6) |

## 10. Ownership

D-CMR-1..5 touch quack and the core NAL module. The other session is moving
R2IL and NAL into core; D-CMR-1 (quack only) can start independently,
D-CMR-2..5 after NAL lands in core. Owner to be assigned.
