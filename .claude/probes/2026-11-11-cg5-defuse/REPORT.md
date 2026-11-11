# CG-5 probe: def-use rows separate a scalar threshold from a per-element quotient

Slice: CG-5 (Linear EXO-237), after CG-8 (EXO-241). Ruling: D-CODEGRAPH-BOARD-RULING-2 allowed one
bounded, read-only probe on the ndarray PR 422 case. Status: MEASURED. Nothing here is a production
path; no row format, opcode, lowering or rewrite is added.

## Question

`foundry-parity-lab` has four `/` sites. Two build one threshold before any row is read
(`(q + per_staff - 1) / per_staff`); two divide once per recipe ingredient (`v / d`). The effects rows
of CG-8 find all four and say nothing about which kind each is. Does a def-use derivation tell them
apart, with a proof, and does it move when the code moves?

The expected answer was posted on EXO-237 before any derivation existed:

| site | expected |
|---|---|
| `src/method.rs:58` (`can_cook_guard`) | scalar threshold |
| `src/arm_b.rs:245` (`cook_ops`) | scalar threshold |
| `src/arm_a.rs:75` (`can_cook`) | per-element quotient |
| `src/oracle.rs:62` (`can_cook`) | per-element quotient |

## Method

1. `charon cargo --no-dedup-serialized-ast -- --lib` on the crate (Charon 0.1.297).
2. `extract_defuse.py` reads the LLBC and writes rows for every function holding a `Div` or `Rem`:
   `ARG`, `DEF` (one per defining statement, with the locals it reads), `ELEM` (result of an iterator
   `next` call inside a loop), `INDEX` (value read through an index projection), `SITE` (the operator,
   its file, line, loop depth and two operands). The rows are flow-insensitive. 783 rows, 16 sites
   (5 `Div` statements at 4 source lines, 11 `Rem`).
3. `crates/cg-defuse-probe` answers each site three ways:
   - **egglog** (`derive`): the rows as terms, eight rules, proofs mode, stepped to a fixpoint.
     A class is reported only with the proof egglog checked.
   - **plain fixpoint** (`fixpoint`): the same two closures as Rust loops. Shares the rows only.
   - **source scan** (`source_class`): reads the source line and the open `for` headers. Shares
     neither rows nor rules.
   Loop nesting of the statement (`nesting_class`) is the cheap comparator.

Rules, per function: a site is per-element when an operand depends on an element source through any
chain of definitions. A local is scalar when it is an argument, or is not an element source, has
exactly one definition, and that definition reads only scalar locals. A site is a scalar threshold
when both operands are scalar or literals. Everything else is unknown.

## Result

All three arms give the expected table. From `cargo run --release --example report`:

| site | egglog | plain fixpoint | source scan | loop nesting |
|---|---|---|---|---|
| `method.rs:58` | scalar | scalar | scalar | scalar |
| `arm_b.rs:245` (two LLBC copies) | scalar | scalar | scalar | **per-element (wrong)** |
| `arm_a.rs:75` | per-element | per-element | per-element | per-element |
| `oracle.rs:62` | per-element | per-element | per-element | per-element |

- Loop nesting is wrong on one of four real sites. `arm_b.rs:245` stands after the source loop, and
  Charon copies the tail of `cook_ops` into the loop's exit paths, so the statement appears twice
  inside a `Loop` node.
- Of the 11 `Rem` sites in `fixture.rs`, 8 are scalar and 3 are unknown: their divisor is a global
  place, which the rules do not classify. Unknown carries no proof.
- egglog and the plain fixpoint agree on all 63 sites of the four fixtures.
- The proof for `method.rs:58` is 62,654 bytes; for `arm_a.rs:75`, 2,838 bytes.

## Mutation control

`mutate.py` writes three variants of `src/method.rs`; each is run through Charon again.

| mutant | change | egglog | source scan | loop nesting |
|---|---|---|---|---|
| m1 | division moved into the loop, divisor is the element `d` | per-element | per-element | per-element |
| m2 | division removed | no site (16 → 15 sites) | no operator on the line | — |
| m3 | division moved into the loop, operands unchanged | scalar | scalar | **per-element (wrong)** |

m1 flips the class, m2 removes the row, and m3 shows the classes follow the operands and not the
position of the statement.

## Cost (reported, not a gate)

One cold derivation over 783 rows, release build: egglog 1.65 to 2.10 s (1,654,062 to 2,101,388 µs,
26 steps), including the proofs of all sites. The plain fixpoint over the same rows: 912 to 1,190 µs.
The derivation runs once per program and never per query.

## What this does and does not show

- It shows that operand provenance from LLBC decides this case where loop nesting does not, and that
  egglog can carry the closure with a checked proof per site.
- The plain fixpoint gets the same classes about 1,800 times faster. What egglog adds is the proof,
  not the answer. Whether a proof per site is worth two seconds per crate is open.
- The source scan also gets all seven cases right. It is a line scan with `for` headers; it would
  miss a divisor that reaches the line through a helper or a renamed binding. No such case is in the
  crate, so that limit is not measured here.
- "Per element" means per item of the loop the operand comes from. In this crate that loop is over
  recipe ingredients, not store rows. Dependence on the row argument is a different question and
  was not asked.
- The rows are flow-insensitive and intraprocedural. A call result is scalar when its arguments are.
  A value written through a `&mut` passed to a call counts as a second definition.
- One crate, four division sites, three mutants.

## Open

- Should def-use rows become a store row kind? That is the format decision of CG-3 (EXO-235) and is
  not made here; the rows live in this probe's fixtures only.
- Interprocedural provenance (is the divisor `fixture.rs:105`'s `per_staff`?) needs field-level
  def-use across the `recipe` call. Not attempted.

## Reproduce

```
cd crates/cg-defuse-probe && cargo test --release                # 18 tests
cargo run --release --example report -- tests/data/m0.defuse.tsv src/method.rs:58
CHARON=<charon binary> .claude/probes/2026-11-11-cg5-defuse/regenerate.sh   # rewrites tests/data
python3 .claude/probes/2026-11-11-cg5-defuse/disable_runs.py     # 24 guards, receipt: 24 red
```

`regenerate.sh` was run after the fixtures were committed and changed no byte. The crate has its own
workspace root and no CI workflow.
