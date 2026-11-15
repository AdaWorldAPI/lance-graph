# 2026-11-15 — D-XLOG-COV-0: XLOG coverage reconciliation (lab only)

> ⊘ **Superseded on the ladder counts by D-XLOG-COV-1**
> (`2026-11-15-d-xlog-cov-1-coverage-parity.md`). The committed census is now
> `census-656def9.tsv`; `census-c9bd9dc.tsv` is in git history. The 20
> "unverified" rules below were limits of this probe's oracle, plus two cube
> defects it could not yet see. All of them now have verdicts. Everything
> else in this entry stands.

**Kind:** lab probe `crates/xlog-coverage-probe` (own workspace, not a member),
two safety reproductions, board correction. No production code changed, no
execution authority added, nothing admitted.
**Grade:** MEASURED against an independent oracle; every rung below has its own
evidence. Disable controls: 7 of 7 red.
**Sources:** lance-graph base `668d6bf`; XLOG fork `AdaWorldAPI/xlog` `main` at
`c9bd9dcebc6da3ff90747d32fad0f2090d75e4af` (`xlog-cube`, `xlog-logic`,
`xlog-ir`, `xlog-core`); corpus `xlog-integration/tests/*.rs` (47 files).

## Three numbers that were being read as one

| figure | what it counts | where | status |
|---|---|---|---|
| 7/13 | RIR node KINDS with an admission rule in `lance-graph-xlog` | `d-xlog-native-v1.md` Amendment E, census 2026-11-01 | historical, kept |
| 12/230 | rules admitted, as first stated on the D-XLOG-NATIVE-7 row | STATUS_BOARD | stale; replaced by a reference to this entry |
| 140/230 | rules NOT refused by `xlog-cube` over EMPTY base relations (first-refusal census, `coverage.rs`) | `xlog-wiring-to-hot-path-v1.md` | a refusal classification, not correctness |

None of them is a correctness count. The ladder below is.

## The ladder (230 rules)

Run: `cargo run --release --manifest-path crates/xlog-coverage-probe/Cargo.toml --example reconcile -- --out census.tsv <xlog>/crates/xlog-integration/tests/*.rs`.
Committed output: `crates/xlog-coverage-probe/census-c9bd9dc.tsv` (one row per rule).

| rung | count | evidence |
|---|---|---|
| source-observed | 396 literals | every `r#"…"#` literal in the corpus |
| parsed and compiled | 188 programs, 230 rules | `Compiler::compile` succeeded (208 literals did not compile: not XLOG programs, or not this dialect) |
| refused (empty bases) | 90 | `execute_plan` refused the SCC; reproduces `coverage.rs` exactly |
| refused-on-data | 49 | ran over empty bases, refused over non-empty: all 49 in recursive SCCs |
| lowered-unverified | 20 | ran on data, no usable oracle answer (reasons below) |
| oracle-disagrees | 0 | — |
| oracle-agrees | 71 | equal to the naive oracle on all 7 seeds, non-empty on at least one |
| native-tested | 32 | oracle-agrees and not a host loop (the other 39 run in `xlog-cube`'s host loop over owned rows) |
| production-admitted | 0 | nothing in production calls `xlog-cube` |

So of the 140 "admitted", 49 are refused as soon as there is data, 20 cannot
be judged, and 71 are correct on the fixtures; 32 of those run on cube masks.

**Why 49 hide from the census.** In `xlog-cube`'s `recursive_scc` an empty
seed delta ends the loop before any recursive body is lowered, so a body the
cube cannot lower is never tried. On data: 28 refuse with "computed or
out-of-range output column", 21 with "`tri`/`cyc` is not a binary relation"
(a ternary member of a recursive SCC).

**The 20 unverified:** 7 heads the compiler generates (`__xlog_query_0`, no AST
rule); 10 rules in programs with a non-binary base relation (`role_perm`, `gateway`,
`input`, `node`); 2 over the naive oracle's 200,000-row bound; 1 derived
predicate that also has ground facts (the cube ignores such facts).

## The oracle

`src/naive.rs`: a naive bottom-up Datalog evaluator over the parsed AST, set
semantics, stratified negation, comparisons over variables and integers, least
fixpoint per component. It reads no RIR and no plan; only the component ORDER
comes from the compiler. Unsupported terms and literals are refused by name.
XLOG's reference evaluator runs alongside as a cross-check: it answered 17 of
the 71 (all agree) and refused the rest (`InsufficientPositiveAtoms` for
one-atom rules, which is why the first run of this probe judged only 17).

Seeds: 6 random (1..=12 rows over 0..16 plus the program's integer constants),
1 dense (the complete graph over 0..6). Without the dense seed 36 agreements
were over empty answers only; an empty agreement is never counted (stop
condition). With it, 13 of the 36 agree on real rows and 21 turn out to be
refused on data.

Semantics checked: shared and repeated variables, constant selection and head
construction, projection, comparison filters, negation, joins, cycles and
recursion, empty and non-empty results. Set vs bag: these programs declare set
semantics, and both sides are compared as sets; multiplicity of the cube's
answer is not checked.

## Disable controls (committed first, each anchor asserted, reverted)

| control | red |
|---|---|
| D1 comparison always agrees | 2 tests |
| D2 naive ignores integer constants | 1 |
| D3 empty agreement not demoted | 1 |
| D4 refused-on-data rung removed | 1 |
| D5 the cube's answer loses one row | 6 (every agreement test) |
| D6 naive ignores negation | 1 |
| D7 naive ignores comparisons | 1 |

## Where the mismatches cluster

Three missing operations account for 144 of the 159 compiled rules that do
not agree (90 refused + 49 refused-on-data + 20 unverified):

1. **An n-ary relation** (binary only today): 33 refused + 21 on data + 10
   unverified = 64.
2. **Column construction and projection** (computed or reordered output
   columns, projection over projection / group-by / join): 12 + 28 on data +
   5 + 3 + 2 = 50.
3. **Predicates and join shapes beyond `col == const` and one shared key**:
   17 filter + 7 `Join body` + 6 join key = 30.

The other 15: 5 behaviour words (`0xFFFF_F100/1`, W7) and 10 unverified for
reasons outside the cube (generated heads, oracle bounds, facts on a derived
predicate). Whether the same three are the gaps for MeTTa, MORK and DT64 is not
measured here; it is the next falsifier, not a finding.

## Safety findings, reproduced and not repaired

`crates/xlog-coverage-probe/tests/safety.rs`, public `cube` API only.

**A. `Cube::refresh_closure` accepts a closure of a stale base.** The
precondition (the closure is of the current base lanes) is documented and not
checked. Fixture: base `{1→2, 2→3}`, closure taken; `3→4` added and folded;
`4→5` logged. The control (closure of the current base) equals a fresh
`closure_wide`; the stale closure returns `Ok` and loses pairs.
Proposal, not built: a per-relation fold generation (none exists in `cube.rs`;
`refold` would bump it), a stamp on each closure, and a typed
`CubeRefusal::StaleBase { expected, found }`. The mechanism exists in-tree:
NSL's `FocusLease` records a `SourceRef` once and refuses another with
`NslError::StaleSeal`. Falsifier: the test flips to "refused"; the control
stays green.

**B. `Cube::commit` adopts a wrong frame in a release build.** The check is a
`debug_assert!`. Debug panics (pinned). Release adopts the stale frame,
applies the batch, and a later `materialize_rows` panics decoding the
committed coordinate (pinned with `catch_unwind`).
Proposal, not built: `commit` returns `Result`, refusing with
`CubeRefusal::FrameMismatch` before touching the cube, in every build.
Falsifier: the two cfg arms become one test expecting the refusal and an
unchanged cube.

## Reuse, with the limits

| candidate | limit |
|---|---|
| cube `column_counts` + `heaviest` for W4 | W4's `PrefixDegreeStats` holds `avg_degree` = rows ÷ distinct keys of ONE column (Σc(p) ÷ #{p : c(p)>0}) and `max_degree` = `heaviest`. `join_size ÷ support` is a per-join average multiplicity and is not W4's metric. The insert-path producer is not built. |
| RelAlg `normalize` for the 17 filter refusals | moves only refusals whose normal form is a conjunction of `col == const`; it adds no comparison kernel. Not counted. |
| `is_alpha_acyclic` (GYO) | a planning classifier; it licenses no execution of the 13 join refusals, and the Gather ban stands. |
| NAL compile-once `Prepared` | a cost fix; the census is a cold oracle arm with no hot path. |
| ONNX packed masks | shared packing only. |
| DR `Frame` vs NSL `index_seal` | not one identity (per-tile `MapTable` revisions vs `(dataset_id, version, index_seal)` recorded once per lease). The mechanism transfers (defect A); the identities do not. |

## Undecided, with the smallest falsifiers

**W5 (`RirNode::Fixpoint` dormant).** (a) Lower recursive SCCs to `Fixpoint`
with one semi-naive owner; (b) remove the node. Smallest falsifier for (a): the
49 refused-on-data rules; a recursive lowering that is right turns them into
oracle-agrees in this census with no new disagreement. For (b): P-A4a plus this
census unchanged after removal.

**W7 (F100 execution authority).** (a) Keep the 5 behaviour-word rules refused
by address; (b) execute F100 through `BindStore` with a relational word.
Smallest falsifier for (b): P-I1 on one word, then these 5 rules reach
oracle-agrees here. Neither is decided by this entry.

## What this does not show

- Correctness beyond the seven seeds and the corpus.
- Bag counts.
- Anything about `xlog-runtime` (CUDA).
- That the three clusters generalize beyond XLOG.
