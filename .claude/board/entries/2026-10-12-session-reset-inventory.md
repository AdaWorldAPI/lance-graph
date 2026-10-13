# 2026-10-12 — Session reset: method layer moved to backup (D-CMR-RESET)

**Status:** DONE (removal); the rebuild is OPEN.

**Finding:** the class-method layer built on 2026-10-12 was a second algebra
beside the existing folds, not a binding over them, and it was slower than a
scalar row loop: `billable_hours` measured 430–480 µs per call against 283 µs
for the row loop (1,048,576 rows, AVX-512, release). Its hops were
foreign-key reads; the graph has none. Everything here is moved to backup and
rebuilt.

**Root cause (2026-10-13, VERIFIED-IN-CODE on the backup branch):** each
method was evaluated on its own, so a chain never existed as one expression
that could be simplified before execution.
1. *Materialized per call:* `run_quack` (old `method.rs:788`) built a fresh
   population-sized `Scratch::for_program(program, n_rows)` on every call and
   ran the tiled path; `Compiled` and `Lowering::Range` were never used.
2. *Gather stacking instead of differences:* hops were per-row key reads
   (`Relation { fk }`, `ForeignLane`, `Cond::Reaches/Via`), stacked further by
   the chained gather (#358, #1522); cost grew with rows × hops. The tree already
   intersects ranges symbolically (`fold_ranges`, quack `lib.rs:1918`) and
   answers `Count` as `hi − lo` (`Lowering::Range`).
3. *Optimized in the wrong place:* the care went into identity structures
   (`MethodKey`, the FNV body digest, `SchemaSeal`, linear `digests` search,
   `edges.retain`), not into the data path. Their own cost was not measured.

## Where the old state lives

| repo | branch | what |
|---|---|---|
| lance-graph | `backup/session-011BMJ-2026-10-12` | `main` at `8ae5bce4`, with the whole method layer |
| lance-graph | `backup/session-011BMJ-2026-10-12-chain2` | #1522 (`GatherChain`, `Filter::Chain`, two-hop glove) |
| lance-graph | `backup/session-011BMJ-2026-10-12-sagfix` | #1519 (SAG reprojection fixes) |
| lance-graph | `backup/session-011BMJ-2026-10-12-capstone` | #1523 (SAG docs capstone) |
| ndarray | `backup/session-011BMJ-2026-10-12` | `master` at `e726f98`, with `mask_gather_chain_u32` (#358) |

## Removed

| item | where | why |
|---|---|---|
| `method.rs`, `call.rs`, `decl.rs`, `gremlin.rs`, `awareness.rs` | `crates/lance-graph-glove-parity/src` | own `MethodBody` algebra, lowered per call through `Filter::cmp` over value lanes with fresh scratch (`run_quack`), no `Compiled`, no `lower_fused`, no resident planes |
| `Relation { fk }`, `ForeignLane` hops, `Cond::Reaches/Via`, edge-table anchor, `Fold::Distinct` | `method.rs` | foreign-key model; a hop is a coordinate range, not a key read |
| SAG graph (`Vec<Edge>`, `retain`, `HashSet`) | `awareness.rs` | O(edges) per operation; built on the removed method layer |
| 10 test files, 2 method fixtures, 4 examples | `crates/lance-graph-glove-parity/{tests,examples}` | test the removed code |
| mask-risc `GatherChain`, Quack `Filter::Chain` | #1522 (not merged) | per-row chained key reads |
| `mask_gather_chain_u32[_under]`, `ChainHop`, `ChainTest` | ndarray (#359 removes) | same |
| `lance-graph-r2il` dependency, OGAR checkout in `glove-parity.yml` | crate `Cargo.toml`, workflow | only the method layer needed them; both restored to their state before this session |

## Kept, and why

| item | why |
|---|---|
| `basis.rs`, `tests/authority.rs`, `tests/canonical_front.rs` | another session's work (D-XGP-1/2); untouched, tests green |
| r2il edits: every fold reads a resident plane as its mask (`8d5a36fb`); MIN/MAX and ordered-i32 procedures (`964a95c5`); declarative bodies (`0dbb6b52`, `b61f62fa`) | woven into r2il's own procedure surface; reviewed in the rebuild, not reverted blind |
| contract: compute DAG reads all 256 field positions (`6a83f99f`) | a correctness fix in `class_view.rs`, independent of the method layer |
| build: native target-cpu, debug 0, x86_64 pinned to v3 for wheels (`5cfc8453`, `12cdfde9`) | build configuration, not a primitive |
| NAL recipe audit, CE64 plateau, nal-polyfill plan, planner revision/receipt probes | research and oracle tests, no query path |

## The rebuild (D-CMR)

A class method is a binding `classid × WideFieldMask → an existing fold`: one
`Compiled` mask-risc program over resident planes and coordinate ranges
(`Lowering::Range`, ternlog fusion, the fold terminals, `CountKeyRunsU32`,
Quack `lower_fused`, `fold_ranges`, `prefix_*`). A call resolves the binding
and runs the compiled program; it does not lower, allocate or interpret. Chained
hops are coordinate-range intersections inside the same program. A fold that
does not exist is reported as a gap in the fold catalogue, not built in the
method layer.

**Concept chains (DECISION 2026-10-13; SCOPE: the D-CMR rebuild and every
composition of pure methods; BASIS: the root cause above; REVISIT WHEN a rule
needs information no existing IR can carry).** Methods in a pure chain are
unresolved relations, not executions. The whole chain is normalized before
lowering; only the irreducible remainder becomes a `Compiled` program.
1. A chained method returns a relation, never an intermediate population.
2. Normalization uses proven identities only: intersections, complements,
   differences, coordinate transformations, Boolean truth tables (ternlog),
   applicable law tables.
3. Existing IRs and folds only. Quack's `Filter`/`Query` tree holds
   Quack-bound chains; global NAL methods stay R2IL-bound and are not routed
   through Quack. No second method algebra, no new universal IR.
   ⊘ 2026-10-13: the "no new universal IR" half is superseded by one
   set-theory IR (`plans/d-cmr-concept-chain-normalization-v1.md` §3, with
   its insufficiency proof); Quack and NAL become frontends lowering into it.
   "No second method algebra" stands.
4. `SUM` first uses the existing fold over the minimized extent. A prefix lane
   (`P[hi] − P[lo]`) is a separate, measured candidate with its own ordering,
   overflow, NULL, update and maintenance laws.
5. A coordinate is valid only under a proven ordering or index mapping.
   Constraints on one coordinate basis intersect symbolically; different bases
   need a checked transformation or an index; otherwise the predicate is a
   residual over the already-reduced extent. Two fields never share an order
   by assumption.
6. Rules compile once; parameter bounds bind per call as a small boundary
   computation, with no population scan and no materialized mask.
7. Every rewrite is checked against the unrewritten form and an independent
   row oracle. Disabling an optimization leaves the answer unchanged; removing
   a required semantic guard (three-valued logic, bag vs set, overflow) must
   produce a failing counterexample.
8. A missing law is a rule-catalogue gap; a missing fold is a physical-catalogue
   gap. Neither is built as an executor in the method layer.

**Test cases to port into the rebuild** (on `backup/session-011BMJ-2026-10-12`,
`8ae5bce`, under `crates/lance-graph-glove-parity/tests/`): `polyfill.rs` (15
tests, NARS revision as data; W3 of #42), `ordered.rs` (3, MIN/MAX),
`distinct.rs` (6, distinct hop targets), `relation.rs` (8, functional
relations), `awareness.rs` (19, the H1 candidate). They call the removed
`MethodBody` API, so each case is re-expressed against a `Compiled` program,
and its row-loop answer is re-read and re-run before it counts as an oracle.

OPEN: #1523 anchors `ProposedMethod` on two removed types: the rewrite shape
on `glove-parity::awareness::Candidate`, the action shape on
`glove-parity::method::EffectClass::Propose` (not the unrelated Cohen's d
`EffectClass` in `lance-graph-cognitive`). The rewrite anchor moves to the
rebuilt binding. The action anchor is the action gate: a `Pending`
`ActionInvocation`. Today two methods advance it: `commit_via` (no admin
bypass) and `commit` (admin bypass). Which one a proposal must pass is OPEN
(capstone §2.1 item 7, which also lists
what the gate does not carry yet).
