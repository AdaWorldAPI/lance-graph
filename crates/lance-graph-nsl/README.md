# lance-graph-nsl

NSL neighbor selection, checked against NSL itself. This is W0 + W1 of the NSL
research plan in `AdaWorldAPI/neural-structured-learning`
(`research/2026-10-25/`, H1 `01a` fixtures, H7 `07a` oracle manifest).

## What it does

Given a TSV graph and labeled/unlabeled examples, it chooses each seed's
neighbors exactly as NSL's `pack_nbrs` does. NSL then copies each neighbor's
features into every seed that selects it. This crate instead stores every
example once (`FeatureTable`), interns node IDs into a sorted label codebook
(`Graph`: an ID's code is its rank, so NSL's ID tie-break is an integer
compare) and keeps neighbors as row references (`Adjacency`, a CSR sorted in
NSL order). `loss::graph_loss` is NSL's graph loss (L2, L1, cosine, and KL / JS after
softmax; `SUM_BY_NONZERO_WEIGHTS`) in f32. `compat_lines` writes the records
NSL would have produced, so the two can be compared line by line.

| NSL behaviour | here |
|---|---|
| directed edges, missing weight = 1.0, duplicates keep the max, first of equal weights wins | `Graph::from_tsv` (stable sort) |
| only neighbors with an example are selected | `Adjacency::build` |
| order: (weight, ID) descending on the parsed f64; `0.0 == -0.0` | `partial_cmp`, not `total_cmp` |
| emitted weight is the f64 cast to f32 | `compat_lines` |
| a seed's example wins over an unlabeled one with the same ID | `FeatureTable::new` |
| `max_nbrs = 0` selects nothing, `None` is unlimited | `Adjacency::neighbors` |

Refused rather than approximated:

| input | why |
|---|---|
| NaN weight (G12b) | NSL's order then depends on which edge was read first |
| weight text such as `1_0` | Python's `float` reads 10.0; Rust's parser does not |
| a duplicate example ID in one table | NSL keeps whichever was read last |
| a read whose `SourceRef` differs from the loaded one (G09) | `FocusLease` refuses before any feature is read |

## The oracle

`data/fixtures.txt` is read by both sides. `oracle/nsl_pack_oracle.py` runs
NSL's unmodified `pack_nbrs` end to end (TFRecord in, TSV in, TFRecord out),
loading only `tools/graph_utils.py` and `tools/pack_nbrs.py`. The package
`__init__` imports Keras and is not executed. The script writes
`data/nsl_golden.txt`, whose header records the SHA-256 of both source files.

```
python -I oracle/nsl_pack_oracle.py <neural-structured-learning checkout>
python -I oracle/nsl_loss_oracle.py <neural-structured-learning checkout>
```

The loss oracle runs NSL's unmodified `lib/distances.pairwise_distance_wrapper`
(and `configs/`), freezing the seeded inputs together with TensorFlow's result.

Two runs give the same golden file.

### Manifest

| field | value |
|---|---|
| experiment_id | NSL-DR-001 (B03, neighbor retrieval) |
| nsl | `056c4ac5ca0ca2da9efad0ebd5d3620eda7a2cbb` |
| lance_graph base | `e731a49be38dced21a955144d319aac1cd4cd0d7` |
| tensorflow | 2.21.0 (CPU), Python 3.13.17 |
| graph_utils.py sha256 | `ffae995c76e5cfdf070ebfd872112b4cd3d4dacb0ffdb99cea09da96f45cbb8c` |
| pack_nbrs.py sha256 | `ff9ae806f04581736f4c54ad31a16dc4fc1f885a7b7907484cab921767630a7c` |
| oracle script sha256 | `52abffbc09fc21db8d82b255e275864b66b4f2bbd0e66e6d980621113b81d9fd` |
| fixtures.txt sha256 | `252a2fa38f0b5f2efdbea0e72901f951ac0d580403af20ecee6a7aa5dae8cae3` |
| nsl_golden.txt sha256 | `221cb9640c66815241cb77482e443df009fb98f32eaab94d393adf8eea58e087` |
| loss oracle script sha256 | `6ded468a096a8cdd0016e9aafe96d8519e0009b4df5266df4b03d3e4be90a180` |
| nsl_loss_golden.txt sha256 | `6d71cb9ad0504cf4680f66b7c01b422f32d1c49d35b28c60d3e202cb212578ca` |
| distances.py / configs.py sha256 | in the loss golden header |
| rustc | 1.99.0, x86_64-unknown-linux-gnu, Intel Xeon @ 2.80GHz (avx512f) |
| tolerance | exact bits (f32 weight and feature hex) |
| randomness | none in the fixtures; the cost probe uses a fixed xorshift seed |

## Fixtures

G01–G12 are the H1 table. Four were added; each separates a correct
implementation from a plausible wrong one.

| fixture | input | observable |
|---|---|---|
| G01 / G01b | two weights / one edge without a weight | C then B / the missing weight is 1.0 |
| G02 / G02b / G02c | A→B .4 and .9 / 0.0 then -0.0 / -0.0 then 0.0 | one B at .9 / first of equal weights kept (+0, then −0 bits) |
| G03 / G03b | equal weights / equal as f32, different as f64 | reverse ID tie-break / f64 decides |
| G04 | A→missing .99 | missing filtered |
| G05 | B both seed and unlabeled | seed features copied |
| G06, G07 | no out-edges; directed only | `NL_num_nbrs = 0`, no `NL_nbr_0_*` |
| G07u | G07 symmetrized | B gains A |
| G08 | C selected by three seeds | C stored once; 30 copied bytes vs 22 referenced |
| G09 | changed `index_seal` | refused, loader not called |
| G10 | 8 concurrent readers | one load, one shared table |
| G11a–c | cap 0, 1, none | 0, 1, 3 neighbors |
| G12a | −0.5, +inf, −inf, 0.0 | the order Python gives |
| G12b | NaN | refused (NSL's own order recorded) |
| G12c | 0.0 and −0.0 | tie; ID decides (C first) |

## Results

- 18 of 19 packing fixtures match the golden byte for byte; G12b is refused by design.
- `graph_loss` agrees with TensorFlow within 0–3 ULP on the 10 L2 / L1 /
  cosine cases. On the 6 KL / JS cases the gate is TensorFlow's own f32
  cancellation budget (eps32 times the summed term magnitudes). TF's KL is a
  difference of near-equal f32 terms, so at small divergences it sits up to
  22,000 ULP from the f64 value. The reference lands at 0.10 of that budget
  or less.
- 25 tests plus 11 ignored probes. 30 disables: 29 are red on a named test.
  One is not load-bearing: the KL / JS arm of the nonzero count duplicates
  the `Axis::Last` arm, which these cases already take.
- Cost probe, release, 20,001 edges, 1,000 seeds: `neighbors(seed, 16)` 1 ns
  per seed. Cold builds, reported, not gates: TSV parse ≈ 3,700 µs, freeze
  ≈ 1,000 µs.
- Palette512 / SM9, palette256 needle + Fisher-z LUT, TFLM-shaped softmax
  table and residual-coding evaluation (edge weights, loss distance, stored
  features)
  and the production recommendation: `docs/research/nsl/production-path-evaluation.md`.

## Not covered

- Feature columns other than `id`, `tag`, `x`; NSL copies any feature.
- The graph loss and `build_graph` (LSH candidates), which H1 keeps separate.
- A comparison against a plain HashMap + SoA control on the same workload, and
  memory or latency at Cora scale (B01).
