# 2026-11-13 — coresearch: what the bag as a thinking object should be

**Kind:** exploration map (co-research council). Ratifies nothing.
**Follows:** the 5+3 review EXO-262 (no bag object; ℕ on coordinates not
implemented; ℕ→B untested at act level; realization is a match arm; binding
per call).
**Plans explored:** `.claude/plans/d-dynbag-holographic-bags-v1.md`,
`.claude/plans/dr-widefieldmask-dt-classview-v1.md` §1c,
`.claude/plans/dr-select-k-v1.md`, `.claude/plans/d-vas-implementation-v1.md` R9–R14.
**Run:** premise gate (PREMISE-SPLIT, applied), 5 scouts (code, internal prior
art, literature, systems, concepts), crosswalk of 16 ideas, 3 co-architects.
Tree: origin/main `ea43c9876`.

## 1. The question, after the premise gate

"Bag object" bundled three things with different lifetimes and writers. They
were explored separately:

- **Q-C carrier:** how ℕ on coordinates is carried, and whether the B reading
  is derived from it or kept as its own plane.
- **Q-B binding:** a resolve-once handle tied to a sealed frame, one
  realization per semiring, typed refusal per (operation, semiring).
- **Q-I intake** (the operator question from EXO-262): is a position list with
  repeats the intake form of the carrier, or the permanent bag form?
- **Exit option kept on the map:** the bag is a description resolved to
  carriers, and no stored ℕ object exists.

## 2. What the exploration points to

None of the sources searched (§6 lists those not searched) supports a
**stored** bag object. Whether the bag is **addressable** is a separate
question: the plan says methods take a bag by address (§1c, pictures eight
and nine), and this map keeps that. The shape the evidence converges on:

> A bag is a **description**: a semiring tag, an intake form and a frame
> version. It is bound once, in the one binding store, and resolved to
> carriers that already exist. ℕ is computed from count sinks on demand. B is
> the image of ℕ under the support map `s(n) = (n > 0)`, or a cached plane
> whose equality with that image is a tested invariant. The position list and
> the `deg × deg` product are two intake forms of the same thing. Its address
> is the binding store's handle; that handle is the object methods are called
> on. Today nothing issues or holds such a handle (X13, D3), so the bag is not
> yet addressable.

Evidence for each part:

- **Carrier.** `lance-graph-xlog` already computes a bag count without storing
  ℕ: two `GroupReduce Count` sinks and a checked i64 product
  (`crates/lance-graph-xlog/src/lib.rs:240-297`), with its own admission for
  `Population::Bag` (`:384-403`). Factorized databases and Kùzu do the same
  (arXiv 1307.0441 §3; Kùzu CIDR 2023 §3). Bit-sliced count planes, the
  literature's stored carrier, were measured inside and lost as storage or for
  arithmetic (`2026-11-03-d-dram-p3-5-kofn.md`: 1.7–4.0× slower than byte
  counters; `2026-11-13-weather-bit-planes.md`: sums about 5× slower).
- **Law.** ℕ is the initial commutative semiring and `s` is the only
  homomorphism ℕ→B. It commutes with select, union, join and project (Green,
  Karvounarakis, Tannen, PODS 2007, Prop. 3.5). It does **not** commute with
  monus difference or any non-monotone test on a count (arXiv 1105.2255).
- **Legality.** COUNT and SUM are defined only under ℕ; MIN and MAX under
  both, because a monoid is a B-semimodule only if idempotent (arXiv
  1101.1110 §2.2). A typed refusal of SUM under B is a theorem.
- **Binding.** `Prepared` plus `Guards{frame, focus, scope}` is already a
  resolve-once handle that refuses when stale
  (`crates/lance-graph-r2il/src/binding.rs:166-178, 564-608`). Nothing holds
  it: `act_in` builds one per call
  (`crates/lance-graph-mailbox-lanes/src/act.rs:434, 469`; ~3.7 µs, 45
  allocations, d-vas R13).
- **Intake.** In every system read, a sequence with repeats is absorbed by the
  structure's own addition (SDQL arXiv 2103.06376 §2; DBSP arXiv 2203.16684
  §4). None treats the repeated list as the permanent form. This does not
  conflict with D-DR-SELECT-K-SV, which forbids lossy conversion to a mask,
  not counting.

**Inside finding (X7):** two bag realizations exist and share only the
`Population::Bag` tag. R12 counts a position list per occurrence
(`crates/lance-graph-r2il/src/lib.rs:550-586`); xlog multiplies two sink
counts. They have different admission paths, different refusal enums and no
common test. The 5+3 review did not see the xlog one.

## 3. Verdicts

| # | idea | verdict | why |
|---|---|---|---|
| X1 | support map `s` as the one tested law | **ADOPT-NOW** (test only) | firewall PASS; probe P-A |
| X2 | failure rows: monus, `count = k`, HAVING, anti-join; `except` is two operations | **ADOPT-NOW** (plan table + witness test) | firewall PASS; probe P-B |
| X3 | (operation × `Population`) legality table from the semimodule rule, checked at prepare | **PROBE** | must replace the per-opcode arms, not sit beside them; probe P-C first |
| X4 | opaque runtime semiring object (GraphBLAS style) | **SKIP** | firewall CONFLICT: dynamic dispatch and a kernel cache are a new executor; `Population` stays the closed tag (folded into X3) |
| X5 | stored-zero trap (present but zero) | **PARK** | matters only if a count is ever kept; probe P-F is ready |
| X6 | ℕ computed from sinks, never stored | **operator decision D1** | already shipped for the 2-atom chain; MISSING-3, -4, -6 stay open |
| X7 | R12 list bag and xlog `deg × deg` bag are one thing with two intake forms | **PROBE** | probe P-D; unification must remove a realization, not add a third |
| X8 | bit-sliced count lane | **SKIP** | measured kill inside; a kept copy is a second stored projection. Residual: transient counts above 255, PARK |
| X9 | plain i32 count lane kept and bound | **PARK**, needs its own decision | firewall CONFLICT: a count is on the same rung as its inputs, so a kept lane is a second owner. Allowed only frame-scoped and disposable; probe P-E gives the break-even |
| X10 | the list is an intake form | **evidence for D2** | accumulate only through the existing `GroupFold::Count` sink |
| X11 | `unique` fact on a bound selection | **PARK** | the fact passes; reuse count may choose a realization, never license a kept lane |
| X12 | set = distinct(bag), applied late | **PROBE** | covered by P-A; also tests building the `member` mask late |
| X13 | hold the `Prepared` instead of rebuilding it | **PROBE**, blocked on an owner | firewall CONFLICT: only the one binding store may hold it; the Quack Binding Store is planned, unbuilt; probe P-G |
| X14 | B plane kept as a cache of `s` | **ALREADY-HAVE** | today B is computed and ℕ is not stored; keep at most one of the two |
| X15 | 2-D count tile by integer matmul | **SKIP** | out of scope; measured ~21 µs per 64² tile |
| X16 | formal concept analysis | **SKIP** | Boolean only |

## 4. Probes (pre-registered; none was run)

- **P-A, commuting square at act level** (X1, X12). In
  `crates/lance-graph-mailbox-lanes/tests/population.rs`. Repeat-free lists:
  bag COUNT `==` distinct COUNT. Random lists with repeats:
  `(bag > 0) == (distinct > 0)`. Anti-vacuity: a pinned number of cases with
  bag > distinct, and at least one zero case. Disable: the bag path ignores
  the guard. ~90 lines.
- **P-B, inequality witness** (X2). A holds row r twice, B once:
  `s(A ∸ B)[r] == 1`, `(s(A) ∖ s(B))[r] == 0`. Silent twin: disjoint A and B
  agree. ~40 lines.
- **P-C, legality enumeration** (X3). Every (opcode × `Population`) pair is
  admitted or refused, never a silent Boolean answer; admitted count pinned
  `== N`; refusal arrives before any lane is read. ~100 lines.
- **P-D, the two bags agree** (X7). A 2-atom chain small enough to enumerate;
  per key, the R12 list count `==` `xlog::bag_count`. At least one key with
  `deg × deg ≥ 4` and one with a zero side. ~80 lines.
- **P-E, lane against list break-even** (X9, X11). n = 5000, list lengths 64,
  1024, 16384 with 30% repeats; accumulate once then k folds, against k list
  folds, k ∈ {1, 2, 4, 16, 64}; ns per operation, target-cpu named. Kill for a
  kept lane: break-even above k = 16 at length 1024, or under 2× at k = 64.
  ~120 lines.
- **P-F, stored zero** (X5). Present-and-zero, absent, and present-and-3 rows:
  structural support and valued support must differ exactly at the zero row.
  ~35 lines.
- **P-G, stale handle** (X13). One `Prepared` run twice on one frame: equal
  results, second run's allocations pinned. Advance frame, focus and scope
  separately: each returns its own refusal and writes nothing. Kill: a stale
  run answers, or holding saves under 10× of the rebuild. ~90 lines.

## 5. Open for the operator

- **D1.** Is "ℕ is computed from sinks, never stored" the default carrier
  (X6)? The evidence supports yes.
- **D2.** What is the position list? (a) an intake form of the bag; (b) the
  permanent bag form; (c) only an operand under D-DR-SELECT-K-SV, with bag
  identity belonging to the binding handle. No source read supports (b). (a)
  and (c) differ only in whether a count is ever accumulated from the list.
- **D3.** Who owns a held `Prepared` (X13)? The firewall allows only the one
  binding store; that store is unbuilt.
- **D4.** Should P-A and P-B (tests only) go ahead now?

## 6. Not searched

- Roaring; FeatureBase sum-by-popcount from a primary source.
- GraphBLAS build-from-tuples with a duplicate-combining operator (the
  closest outside precedent for D2; known from memory only).
- Whether a DuckDB SelectionVector may hold repeats, and its perfect-hash
  aggregate: not in the two inside DuckDB plans, and the web harvest was
  skipped because DuckDB is already mapped.
- Bag-set semantics (Chaudhuri and Vardi).
- Inside: `crates/lance-graph-xlog/src/cube.rs` second `bag_count`, and
  `bag_count` in `mistaber-kripke-probe` and `mork-pathmap-probe`.
- No outside source gives nanosecond costs.

The zero-copy rung test that grades X9 is itself marked CONJECTURE in
`.claude/knowledge/zero-copy-lens-law.md`.

## 7. Closeout

`coresearch | STATUS: open | OUTCOME: the evidence points to a bag as a bound
description over existing carriers, not a stored object; 2 ideas adopt-now as
tests, 4 probes, 4 operator questions | OPEN: D1–D4`
