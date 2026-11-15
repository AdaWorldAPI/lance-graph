# xlog-coverage-probe

A lab probe that answers one question about XLOG's fork on the lance-graph
cube: for each compiled rule, how much evidence is there that the cube's
answer is right?

XLOG's census (`xlog-cube/examples/coverage.rs`) counts the rules the cube
admits, with every base relation empty, so "admitted" says nothing about
correctness. This probe runs the same compiled plan on the cube with
non-empty base relations and compares the cube's output with XLOG's own CPU
reference fixpoint evaluator (`xlog_logic::hypergraph::evaluate_scc_fixpoint`).
The oracle works from the parsed AST, not from the compiled plan the cube runs.

## The ladder

Lowest to highest (`Rung`):

- `observed`: the literal did not compile (program level).
- `compiled`: compiled, but `execute_plan` returned `Err` for the whole plan.
- `refused`: the cube refused the rule's SCC.
- `lowered-unverified`: the cube ran it and no oracle answer was available.
  The reason says why.
- `oracle-disagrees`: the oracle's answer differs from the cube's on at least
  one seed.
- `oracle-agrees`: the oracle's answer equals the cube's on every seed.

A rule also carries `host_loop`: its SCC is recursive, and the cube evaluates
a recursive SCC with a host loop over owned rows, not on cube masks. "Native
tested" is `oracle-agrees` and not `host_loop`.

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
no rule (empty head; rung `observed` or `compiled`).

## Known limits

- The oracle is XLOG's reference evaluator: set semantics, and gated by its
  eligibility check. It refuses a rule with fewer than two positive body
  atoms (`InsufficientPositiveAtoms`), negation, aggregation, and `is`
  expressions. One refused rule leaves its whole SCC, and every later SCC,
  without an oracle (`lowered-unverified`). Single-atom rules are common, so
  many programs stay `lowered-unverified`.
- The oracle is exhaustive over a value domain. A rule whose bound
  `values ^ variables` exceeds 5,000,000 is reported unavailable.
- Bag counts are not checked: both sides are compared as sets.
- Base relations must be binary `u32`. Random rows hold 1..=12 rows over
  values below 16, so every program fits one 256-id tile. A program with a
  non-binary base relation, or a base fact that is not a non-negative integer
  pair, has no oracle.
- Where the program states ground facts for a base relation, those facts are
  used instead of random rows. A derived predicate that also has facts has no
  oracle (the cube ignores such facts).
- The oracle reads the parsed AST, so a program the compiler rewrites (such as
  generated query rules) can have heads with no AST rule; those are
  `lowered-unverified`.
- Derived integer columns are narrowed to `u32` before a later SCC reads them
  (the evaluator emits a head integer constant as `I64`, and its matcher does
  not equate `U32(1)` with `I64(1)`).
