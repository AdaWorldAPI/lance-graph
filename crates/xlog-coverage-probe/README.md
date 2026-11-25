# xlog-coverage-probe

A lab probe that answers one question about XLOG's fork on the lance-graph
cube: for each compiled rule, how much evidence is there that the cube's
answer is right?

XLOG's census (`xlog-cube/examples/coverage.rs`) counts the rules the cube
admits, with every base relation empty, so "admitted" says nothing about
correctness. This probe runs the same compiled plan on the cube with
non-empty base relations and compares the cube's output with an oracle that
works from the parsed AST, never from the compiled plan the cube runs.

## The oracles

- `naive` (`src/naive.rs`): this crate's own bottom-up Datalog evaluator. Set
  semantics; positive atoms, negated atoms (stratified only), comparisons
  between variables and integers; each component iterated to its least
  fixpoint. Anything else is refused by name. It decides the rung.
- `xlog-ref`: XLOG's reference evaluator
  (`xlog_logic::hypergraph::evaluate_scc_fixpoint`), run on the same seeds as
  a cross-check column. It decides the rung only when the naive oracle has no
  answer (in the committed census, never).

## Datafrog cross-check (D-XLOG-KNOWLEDGE-G1)

A third evaluator, `src/datafrog_arm.rs`, runs the same parsed rules over the
same seed rows through the Datafrog engine (AdaWorldAPI fork): a semi-naive
fixpoint over `Variable`s, each body join lowered to keyed `from_join` /
`from_antijoin` statements. It shares no code with the naive oracle or with
XLOG's reference evaluator.

It covers a subset: positive and negated atoms over named variables and `_`
(an anonymous variable in a negated atom anti-joins the relation projected
onto the named columns), integer constants in positive atoms (a semijoin
against a one-row relation), comparisons between bound variables and integer
constants (a filtered join against a one-row relation), head terms that are
variables bound by a positive atom or integer constants, negation only of
predicates outside the component, every value and constant fitting `u32`. A
rule with no positive atom starts from one empty row, so only a ground body
makes it hold. Constants in negated atoms, aggregates, `is` expressions,
floats, repeated variables inside one atom and the rest are
`datafrog: not applicable: <why>`.

The census gains two columns, `datafrog` (`agree`, `disagree`,
`not-applicable`, empty when the rule never reached the oracle stage) and
`datafrog_reason`. It never decides a rung: rungs are the same as without it.

## Reference column: naive oracle vs Datafrog without the cube

The `datafrog` column only exists for rules the cube ran, so it cannot speak
for a refused rule. A second comparison runs the naive oracle and Datafrog
over the same seeds for EVERY compiled rule, with no cube involved, and
records whether the two independent evaluators agree on what the rule should
produce. Three columns: `reference` (`agree`, `disagree`, `unavailable`),
`reference_nonempty` (seeds on which the naive answer was non-empty; an
agreement with 0 is two empty answers) and `reference_reason`. A seed on
which the naive oracle hits its bound is skipped. It never decides a rung.

## Positive witnesses (D-XLOG-KNOWLEDGE-G1c)

`witness(src, rows)` runs the naive oracle and Datafrog over base rows the
caller supplies, with no seeds and no cube. `tests/witness.rs` uses it for the
programs whose refused rules agree only over empty answers in the census (their
bases are unary or ternary, so the seeds never fill them). Every head gets a
hand-derived non-empty expected set, and both evaluators must equal it exactly.
The census itself is unchanged.

## The seeds

Six random seeds (1..=12 rows per base relation over values 0..16 plus the
program's own integer constants) and two dense seeds: `DENSE_SEED`, where
every base relation is the complete graph over 0..6, self-loops included, and
`DENSE_SMALL_SEED`, the complete graph over 0..4. Random sparse rows almost
never close a triangle, cycle or clique; the dense seeds always do. A wide
rule can exceed the naive oracle's bound on the large dense seed (a 7-clique
over 6 values is 6^7 rows). That seed is then skipped for that head; it is
not judged. The small dense seed still gives such a rule a non-empty answer,
and it is the only seed that does for `clique7` and `clique8` in the corpus.
Base relations the program states facts for use those facts.

What the oracles read, per rule, not per program:

- A one-column `u32` base relation is seeded like a binary one (its stated
  facts, otherwise `0..6`, `0..4` or 1..=12 random values per seed) and is
  handed to the cube with `put_set`. Since xlog#9 the fork routes rules over
  such sets to cube set expressions (`route` column: `set` or `reach`).
- A base relation the cube cannot hold (neither binary nor one-column `u32`)
  gets no rows in the cube, so the cube refuses every rule that reads it. The
  naive oracle reads its stated integer facts. Rules that do not read it are
  judged as usual.
- A component that fails, or reads an unavailable predicate, makes only the
  components that depend on it unavailable.
- A query `?- p(X, Y).` is judged through the rule the compiler generates for
  it: `__xlog_query_i(<the query atom's distinct variables, in order>) :-
  <the query atom>`, with head `(1)` when there is no variable.
- Facts stated for a derived predicate seed it in the naive oracle and are
  handed to the cube with `put_relation` (two columns) or `put_set` (one
  column), as any host would.

## The ladder

Lowest to highest (`Rung`):

- `observed`: the literal did not compile (program level).
- `compiled`: compiled, but `execute_plan` returned `Err` for the whole plan.
- `refused`: the cube refused the rule's SCC over empty bases (admission).
- `refused-on-data`: admission over empty bases reported it as ran; over
  non-empty bases the cube refuses it. In a recursive SCC an empty seed delta
  ends the loop before any recursive body is lowered.
- `lowered-unverified`: the cube ran it and no oracle gave a usable answer.
  The reason says why. An agreement over answers empty on every seed is
  reported here (`agreed only on empty answers`), never as agreement.
- `oracle-disagrees`: the oracle's answer differs from the cube's on at least
  one seed.
- `oracle-agrees`: the oracle's answer equals the cube's on every seed, and
  is non-empty on at least one (`nonempty_seeds`).

A rule also carries `host_loop`: its SCC is recursive and the fork did not
route it, so the cube evaluates it with a host loop over owned rows, not on
cube masks. A `reach`-routed SCC is recursive but evaluated inside the cube
(`Cube::reach`), so its `host_loop` is false. "Native
tested" is `oracle-agrees` and not `host_loop`. Production-admitted is 0 by
construction: nothing in production calls `xlog-cube`.

## What it is, and is not

A lab probe and an oracle arm: cold, correctness only, never a timing gate.
There is no timing anywhere in it. It does not make the cube a production
path: nothing in production calls `xlog-cube`.

## Run

```
cargo test --release --manifest-path crates/xlog-coverage-probe/Cargo.toml
cargo run --release --manifest-path crates/xlog-coverage-probe/Cargo.toml --example reconcile -- --out /tmp/x.tsv /path/to/xlog/crates/xlog-integration/tests/*.rs
```

The probe depends on XLOG's fork (`AdaWorldAPI/xlog`, branch `main`) and
patches `lance-graph-xlog`, `lance-graph-dt64` and `lance-graph-contract` to
this checkout, so it tests the cube in this tree.

The TSV has a header row, then one row per rule, plus one row per program with
no rule (empty head; rung `observed` or `compiled`). Columns: file, literal
index, head, rung, host_loop, oracle, xlog_ref, nonempty_seeds, reason,
datafrog, datafrog_reason, reference, reference_nonempty, reference_reason,
route (empty, `set` or `reach`).
The census is committed as `census-28b41f5.tsv` (2026-11-25, EXO-400). It was taken at XLOG
`28b41f5` (xlog#12: a non-recursive derived binary head is a recipe for its readers, EXO-396)
against the native Cube at lance-graph `11e8e01`. Against `census-1435111.tsv` (now in git
history), exactly 5 rows changed and nothing else moved: 114 oracle-agrees (was 109),
67 refused (was 72), 49 refused-on-data, 0 oracle-disagrees. The five are `fof` (two files),
`mutual` and the two `sg_result` rules, each from refused ("computed or out-of-range output
column") to oracle-agrees, with Datafrog agreeing on all five. Set routes stay at 29.

The previous census, `census-1435111.tsv` (2026-11-24, EXO-394), was taken at XLOG
`1435111` (xlog#11: stratified negation over one-column sets) against the native Cube at
lance-graph `2075f7b` (#2171: `SetExpr::Column`). Against `census-097c351.tsv` (now in git
history), exactly 16 rows changed and nothing else moved: 109 oracle-agrees (was 102),
72 refused (was 79), 49 refused-on-data, 0 oracle-disagrees. Seven rules moved from refused
to oracle-agrees, all with Datafrog agreeing: `isolated` ×2, `no_in` ×2, `leaf_part`, and
the two `__xlog_query_0` rules that read `no_in`. Nine rules that already agreed moved onto
the set route (`has_in` ×2, `part` ×2, `has_children`, `dummy_e1`/`e2`/`e3`); set routes went
from 13 to 29. `keep` and `dry` stay refused (negation over a join-shaped body).

The previous census, `census-097c351.tsv` (2026-11-24, EXO-390), was re-taken at XLOG
`097c351` (xlog#10, arrow 58) against the native Cube at lance-graph `855d051`. It is
byte-identical to the previous census `census-007bb19.tsv` (now in git history) once the
revision label is removed: same rungs, reasons, Datafrog, reference and route columns on all
438 rows. The XLOG test corpus and xlog-cube are unchanged since `007bb19`, and the
adapter consumes none of the Cube additions made since then (`body_answer`, `closure_wide`,
`base_closure`, tiled construction, the delta lifecycle; see
`.claude/plans/xlog-reactor-swap-inventory-v1.md`), so the result stays the same. The rest
of this paragraph describes the `007bb19` census, which still holds. It was taken over the XLOG fork at
`007bb19` (AdaWorldAPI/xlog#9: one-column heads routed to cube sets), with
the test corpus unchanged since `c9bd9dc`. Every compiled rule has a verdict:
102 oracle-agrees, 79 refused, 49 refused-on-data, 0 oracle-disagrees,
0 lowered-unverified; 58 of the 102 are native-tested. Against the previous
census (`census-656def9.tsv`, in git history) exactly 15 rows changed, and
they are the 15 rows with a `route` (13 `set`, 2 `reach`), each from refused
to oracle-agrees; the other 423 rows are byte-identical in the earlier 14
columns. Datafrog agrees on all 102 oracle-agrees rules and disagrees on none
(integer constants in positive atoms are lowered as semijoins). With its join
deliberately broken at `656def9`, Datafrog disagreed on 69 of the then 87
while every rung stayed the same.

The reference column (naive oracle vs Datafrog, no cube) gives an expected
answer for refused rules too. Over the 230 compiled rules: 102 oracle-agrees
and 49 refused-on-data agree; of the 79 refused, 26 agree on non-empty
answers, 9 agree only on empty answers, and 44 are unavailable (the naive
oracle has no floats, `is` expressions or aggregates, five ILP `learnable`
rules have no ordinary AST rule, and five read a base with non-integer
facts). 0 disagreements. With Datafrog's join deliberately broken (the right
row read from the wrong end), the reference column reported 114
disagreements at `656def9`, 43 of them on refused rules, while rungs and
reasons stayed byte-identical. A census takes about 15 minutes on this
container.

## Known limits

- Components are evaluated in the compiler's stratum order: only the ORDER
  comes from the plan, every row from the AST. A wrong order that both sides
  share would not show.
- The naive oracle refuses negation of a predicate inside its own component,
  unsafe rules, and every term other than variables, `_` and integers. Bounds:
  200,000 rows per relation, 2,000,000 partial bindings per firing, 10,000
  rounds (policy pins).
- XLOG's reference evaluator is exhaustive over a value domain; a rule whose
  bound `values ^ variables` exceeds 5,000,000 gets no `xlog-ref` answer.
- Bag counts are not checked: both sides are compared as sets, which is the
  declared semantics of these programs.
- A rule reading a base relation that is neither binary `u32` nor integer
  facts (strings, floats) has no naive oracle.
- XLOG's reference evaluator (the cross-check column) reads binary bases only
  (one-column bases are marked unavailable for it),
  and gives no answer for a derived predicate with facts.
- Derived integer columns are narrowed to `u32` before a later SCC reads them
  in the `xlog-ref` arm (its matcher does not equate `U32(1)` with `I64(1)`).
