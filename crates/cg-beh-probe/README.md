# cg-beh-probe: ON × WHEN × DO evidence (D-CG-BEH-2, EXO-155)

A lab crate. It reads what a `BindStore` has registered for a method, keeps ON, WHEN and DO apart,
and says which obligations a change makes stale. It registers nothing in a production store, adds
no polyfill word and executes no method.

```sh
cargo test --release --manifest-path crates/cg-beh-probe/Cargo.toml
CG_BEH_SOURCE_REV=$(git rev-parse HEAD) cargo run --release \
  --manifest-path crates/cg-beh-probe/Cargo.toml --bin facts -- .claude/probes/2026-11-11-cg-beh-2
python3 -I .claude/probes/2026-11-11-cg-beh-2/disable_runs.py   # clean tree only
```

## Evidence grades

| grade | obtained from | says | does not say |
|---|---|---|---|
| `SOURCE_STRUCTURE` | source text: the `match` arm of `FoldDialect::call` a line is in | where code is | what the code computes |
| `DT64_REGISTRATION` | `method_ref`, `admit`, `resolve`, `resolve_algebra` on a store | what is registered | that it runs, or runs correctly |
| `EXECUTED_PARITY` | an executed comparison against an oracle | that an answer matched | (this crate produces none) |

A grade is attached to every hop of an evidence path and is never raised by combining hops.

## `cg-beh.facts.v1`

Tab separated, one fact per row, the row kind first. Classids are `0x%08X`. Written by
`Snapshot::to_tsv`; this file is the only description of the format.

| kind | columns after the kind |
|---|---|
| `H` | `cg-beh.facts.v1`, source revision, label |
| `CLASS` | classid, schema revision, field count, basis digest |
| `BASIS` | classid, position, predicate |
| `METHOD` | classid, method id, signature revision, binding digest, issuer, population, truth, demand, source name |
| `ON` | classid, method id, position, predicate |
| `READ` | classid, method id, position, predicate |
| `WHEN` | classid, method id, logic, expression, witness |
| `WHEN_NONE` | classid, method id |
| `DEP` | classid, method id, ordinal, kind, number, DT64 reading, consumer position, consumer predicate |
| `DO` | classid, method id, ordinal, lens, word, word predicate, arity, target position, target predicate, arguments |
| `DO_NONE` | classid, method id |

What each value is:

- **Source revision.** Required. A snapshot without one is refused.
- **Basis digest.** This crate's FNV-1a 64 of the ordered predicates `resolve` returned. The store's
  own `BasisDigest` is private; this value detects a difference between two snapshots and nothing else.
- **Binding digest.** `Stamp::binding()` of the reference the store issued. It covers the classid,
  the basis, the method id, the signature revision, ON, reads, the ordered DO steps with each lens
  revision and arity, and population, truth and demand. It does not cover the guard or the name.
- **Issuer.** Always `ADMITTED_BY_ISSUER`: the row exists only because the store that issued the
  reference admitted it. A store has no printable identity, so none is written.
- **Predicate.** The `predicate_iri` at that position of the basis `resolve` accepted at the bound
  revision. Labels are not written.
- **Witness.** The `Debug` form of `Guard::witness()`, the guard's normal form under its logic. Two
  guards with equal witnesses fire on the same rows.
- **`DEP`.** One row per leaf of the guard. The DT64 reading is always `UNKNOWN`: `SetExpr::Sym` is an
  opaque numbered set and a residual names a lane, and registration does not check either against
  the basis. The consumer columns hold a field only when the consumer declared that guard numbers
  are field positions and the position is in the method's registered reads; otherwise `UNKNOWN`.
- **`DO` arguments.** `F<position>` or `C<constant>`, in order.
- **`WHEN_NONE`, `DO_NONE`.** Written explicitly. An absent row is never the way "none" is stated.

## `cg-beh.dispatch.v1`

`H`, format name, digest of the source text read. Then one `ARM` row per pattern name: name,
polyfill word or `NOT_A_WORD`, first line, last line. The digest is the witness: a query made
against other text is refused.

## Queries

- `query::diff(before, after)`: per method `Unchanged`, `Added`, `Removed`, or `Changed` with the
  components that moved (`Basis`, `Signature`, `On`, `Reads`, `When`, `WhenSpelling`, `Do`, `Demand`),
  the evidence, and the stale obligations. Facts that contradict the binding digest are refused.
- `query::impl_change(snapshot, dispatch, source, lens, line)`: the arm, its words, the registered
  DO steps on them. No registered fact changes; the open obligation is executed parity.
- `query::word_impact(...)`: the same for one word. A word with no arm is refused.

## Reading these files from another tool

A reader needs no access to this crate. Join `DO` rows to `dispatch.tsv` on the word column. Keep the
grade of each side: the join of a `SOURCE_STRUCTURE` row and a `DT64_REGISTRATION` row is not a
statement about behaviour.
