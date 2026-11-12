# dr-select-k-lab: one versioned selection, four views (D-DR-SELECT-K-1)

A lab crate. Plan: `.claude/plans/dr-select-k-v1.md`, section 4.

## What the lab asks

For one versioned selection of K rows, bound once, which existing physical
representation is cheaper for a given operation and a given reuse count?
The views are the index order, sparse local positions, a K-bit mask and the
implicit ALL. The operations are a Boolean count, a sum over a lane and an
ordered top-T. The lab does not ask whether cold storage or a resident
structure is faster. "One view wins everywhere" is a valid result.

## What it is not

There is no strategy selector, cache, mapper, executor or writer. `u128`
stands in for `NodeGuid`; nothing here is a contract type. The source is an
in-memory stand-in for storage, so no Lance cost is measured here: producing
the selection and staging its rows are counted, but they are not Lance calls.

## The pieces

| piece | role |
|---|---|
| `Source` | owns N rows at one version and one order index (key, then guid); produces a selection with `select_filter` or `select_ordered`, and stages rows |
| `Bound` | the one binding: local rows `0..K`, the order witness, the staged K-local lanes |
| `Residual` / `View` | a further predicate over the K local rows, held as ALL, ascending positions or a K-bit mask |
| `ops` | `residual_a_ge` (convert), `PreparedFolds` (count, sum, ordered top-T), `admissible`, and a resident baseline over all N rows |
| `oracle` | an independent row-at-a-time answer, used for every cell |
| `Counters` | storage accesses, binds, operand rows read by folds, conversions, bytes copied |
| `fixture` | deterministic rows, clustered or random, with a duplication factor |

## Admissibility

`count_b_gt` and `sum_a` do not depend on order and are admissible for every
view. Ordered top-T needs a witnessed key order: it is admissible only when
the binding was made from the index order. A mask or ascending positions keep
the local order only, so over a filter binding the answer is refused and the
report prints the row with `admissible = no` and no timings. Nothing in the
crate sorts at query time.

## How to run

```sh
cargo test --manifest-path crates/dr-select-k-lab/Cargo.toml
cargo run --release --manifest-path crates/dr-select-k-lab/Cargo.toml --example report
```

The report prints one TSV table on stdout, a `#` comment line first and then
the header. Progress goes to stderr. Every cell is compared with the oracle
before it is timed, and a mismatch stops the run with status 1.

## Report columns

| column | meaning |
|---|---|
| `pattern` | `clustered` or `random` placement of a key range in the source |
| `k` | rows selected; the key range is `0..=k-1` with one row per key |
| `control` | `control75` for the 75 percent negative control, `-` otherwise |
| `producer` | `filter`, `ordered`, or `none` for the resident baseline |
| `residual` | `all` (no residual), or `t_a=99` and `t_a=50` for `a >= t_a` |
| `view` | `all`, `positions`, `mask`, or `resident_n` for the baseline |
| `op` | `count_b_gt` (t = 500), `sum_a`, `top_by_key` (limit 10) |
| `admissible` | `yes` or `no`; a `no` row has no timing fields |
| `m` | rows the residual keeps (K for `all`; selected rows for the baseline) |
| `produce_ns` | the selection call; for the baseline, building the N-bit plane |
| `bind_ns` | `Bound::bind`, including staging |
| `prepare_ns` | `PreparedFolds::new` |
| `convert_ns` | building the residual in its stored form; 0 for `all` |
| `fold_ns` | one call of the operation |
| `total_r1_ns` | produce + bind + prepare + convert + 1 fold |
| `total_r100_ns` | produce + bind + prepare + convert + 100 folds |
| `bytes_copied` | bytes copied by one produce, bind and convert sequence |
| `retained_bytes` | residual bytes, plus 4 per row of the binding, plus 28 per row of staged lanes |
| `storage_after_bind` | how far the storage counter moved across the timed folds; must be 0 |

All times are nanoseconds, the median of 41 runs after 3 warm-up runs.
Produce, bind, prepare and convert are paid once per version and are reported,
not gated. The fold is the hot term.
