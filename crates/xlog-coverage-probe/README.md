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

## The seeds

Six random seeds (1..=12 rows per base relation over values 0..16 plus the
program's own integer constants) and one dense seed (`DENSE_SEED`: every base
relation is the complete graph over 0..6, self-loops included). Random sparse
rows almost never close a triangle, cycle or clique; the dense seed always
does. Base relations the program states facts for use those facts.

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

A rule also carries `host_loop`: its SCC is recursive, and the cube evaluates
a recursive SCC with a host loop over owned rows, not on cube masks. "Native
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
index, head, rung, host_loop, oracle, xlog_ref, nonempty_seeds, reason. The
census over XLOG `c9bd9dc` is committed as `census-c9bd9dc.tsv`.

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
- Base relations must be binary `u32`; a program with any other base relation
  has no oracle.
- A derived predicate that also has facts has no oracle (the cube ignores such
  facts).
- Heads the compiler generates (`__xlog_query_0`) have no AST rule and stay
  `lowered-unverified`.
- Derived integer columns are narrowed to `u32` before a later SCC reads them
  in the `xlog-ref` arm (its matcher does not equate `U32(1)` with `I64(1)`).
