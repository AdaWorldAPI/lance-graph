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
| `Source` | owns N rows at one version and two order indexes (key then source row, key then guid); produces a selection with `select_filter` or `select_ordered(range, order)`, and stages rows |
| `Bound` | the one binding: local rows `0..K`, the order witness, the staged K-local lanes |
| `Residual` / `View` | a further predicate over the K local rows, held as ALL, ascending positions or a K-bit mask |
| `ops` | `residual_a_ge` (convert), `PreparedFolds` (count, sum, ordered top-T), `admissible`, and a resident baseline over all N rows |
| `oracle` | an independent row-at-a-time answer, used for every cell |
| `Counters` | storage accesses, binds, operand rows read by folds, conversions, bytes copied |
| `fixture` | deterministic rows, clustered or random, with a duplication factor |

## Admissibility

`count_b_gt` and `sum_a` do not depend on order and are admissible for every
view. An ordered top-T names its order contract, that is, which tie rule it
means, and it is admitted only when the binding witnesses exactly that
contract. Two contracts exist:

- `KeyThenRow`: key ascending, ties by source row position. This is the
  contract the existing `SealedOrder` uses, and it holds for one source
  version only.
- `KeyThenGuid`: key ascending, ties by guid. A separately declared contract,
  stable across a reordering of the source, with its own index.

A request for one contract over a binding made from the other is refused, and
so is a request for the source order, which declares no key order. A mask or
ascending positions keep the local order only, so no view can supply an order
the binding does not have. Over a filter binding the answer is refused for
both contracts, and the report prints the row with `admissible = no`, no
timings and an empty `fold_path`. Nothing in the crate sorts at query time.

## Execution paths are not symmetric

The mask and ALL views fold through mask-RISC. The positions view folds
through a plain scalar loop inside this lab, which exists only to measure the
access pattern. It is not a mask-RISC path and not a proposal for a second
executor. Whether an admitted indexed or sparse consumer exists in mask-RISC,
Quack or ndarray is a separate question that the lab does not answer.

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
| `producer` | `filter`, `ordered` (`KeyThenRow`), `ordered_guid` (`KeyThenGuid`, run at K = 65,536 only), or `none` for the resident baseline |
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
| `bytes_copied` | bytes copied by one produce, bind and convert sequence: 4 per row for the position vector a selection returns, 28 per row for the staged lanes, plus the converted residual |
| `retained_bytes` | residual bytes, plus 4 per row of the binding, plus 28 per row of staged lanes |
| `storage_after_bind` | how far the storage counter moved across the timed folds; must be 0 |
| `fold_path` | the execution path of the fold: `mask-risc` (`all` and `mask` views of `count_b_gt` and `sum_a`, and the `resident_n` baseline), `scalar-loop` (the `positions` view of `count_b_gt` and `sum_a`), `scalar-walk` (`top_by_key`, every view); empty on a `no` row |

All times are nanoseconds, the median of 41 runs after 3 warm-up runs.
Produce, bind, prepare and convert are paid once per version and are reported,
not gated. The fold is the hot term.
