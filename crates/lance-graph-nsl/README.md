# lance-graph-nsl

NSL neighbor selection, checked against NSL itself. This is W0 + W1 of the NSL
research plan in `AdaWorldAPI/neural-structured-learning`
(`research/2026-10-25/`, H1 `01a` fixtures, H7 `07a` oracle manifest).

## What it does

Given a TSV graph and labeled/unlabeled examples, it chooses each seed's
neighbors exactly as NSL's `pack_nbrs` does. NSL then copies each neighbor's
features into every seed that selects it. This crate instead stores every
example once (`FeatureTable`) and keeps neighbors as row references
(`Adjacency`, a CSR sorted in NSL order). `compat_lines` writes the records
NSL would have produced, so the two can be compared line by line.

| NSL behaviour | here |
|---|---|
| directed edges, missing weight = 1.0, duplicates keep the max | `Graph::from_tsv`, `add_edge` |
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
```

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
| fixtures.txt sha256 | `ac3e57a2d8b6de47a4a79de853c2e00c30d18a3cf99e9cf86edcb5e1e5867b0e` |
| nsl_golden.txt sha256 | `16f05fd0269dc849be81781ec2783422ede590633e0659713311f352dbfbf8a0` |
| rustc | 1.99.0, x86_64-unknown-linux-gnu, Intel Xeon @ 2.80GHz (avx512f) |
| tolerance | exact bits (f32 weight and feature hex) |
| randomness | none in the fixtures; the cost probe uses a fixed xorshift seed |

## Fixtures

G01–G12 are the H1 table. Four were added; each separates a correct
implementation from a plausible wrong one.

| fixture | input | observable |
|---|---|---|
| G01 / G01b | two weights / one edge without a weight | C then B / the missing weight is 1.0 |
| G02 | A→B .4 and .9 | one B at .9 |
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

- 16 of 17 fixtures match the golden byte for byte; G12b is refused by design.
- 12 tests plus an ignored cost probe. Each of 11 disables turns a named test red:
  dedup by min, reversed tie, seal check removed, lease bypassed, sort on f32,
  `total_cmp`, NaN accepted, unlabeled features winning, cap ignored, no
  symmetrization, default weight 0.
- Cost probe, release, 20,001 edges, 1,000 seeds: `neighbors(seed, 16)` 1–2 ns
  per seed. Cold builds, reported, not gates: TSV parse ≈ 7,600 µs, freeze
  ≈ 2,000 µs.

## Not covered

- Feature columns other than `id`, `tag`, `x`; NSL copies any feature.
- The graph loss and `build_graph` (LSH candidates), which H1 keeps separate.
- A comparison against a plain HashMap + SoA control on the same workload, and
  memory or latency at Cora scale (B01).
