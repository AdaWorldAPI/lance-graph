# D-TINKER-JANUS-CARRIER-MATRIX

> Companion to `D-TINKER-JANUS-FOLD-0.md`. Sufficiency verdicts are from
> `.claude/tools/carrier_sufficiency.py --min` (exhaustive over small graphs:
> a carrier is sufficient iff no two inputs share it with different answers).
> WALK semantics unless stated.

## Carriers

| Carrier | Identity | Multiplicity | Path | Quack today | TinkerPop analogue |
|---|---|---|---|---|---|
| Support (S) | terminal node | no | no | node mask (`Rows`/`Keep`, `CountDistinctOrderedU32`) | `withBulk(false)` |
| WeightedSupport (K) | terminal node | per-node walk count | no | `GroupReduce{Local(dst), Count}` | traverser `bulk` |
| Factorized (E/W) | per-hop edges | per-edge walk count | per-hop, not joined | edge-table filter (hop 1); W past hop 1 is a gap | — |
| FullBindings (B) | every variable | yes | yes | none (refused as `Bindings`) | path / labels in the merge key |

## Which question each answers (after two hops)

| Query | S | K | Factorized | Full |
|---|---|---|---|---|
| EXISTS | ✓ | ✓ | ✓ | ✓ |
| COUNT DISTINCT terminal | ✓ | ✓ | ✓ | ✓ |
| COUNT(*) | ✗ (DAG: 3 ≠ 4) | ✓ | ✓ | ✓ |
| GROUP BY terminal COUNT(*) | ✗ | ✓ | ✓ | ✓ |
| GROUP BY middle COUNT(*) | ✗ | ✗ | ✓ (W) | ✓ |
| earlier variable (`n0`) | ✗ | ✗ | ✗ | ✓ |
| path identity | ✗ | ✗ | ✗ | ✓ |
| TRAIL-sensitive | ✗ | ✗ | ✗ (needs TW) | ✓ |

## Techniques

| Technique | Layer | New V4 op? |
|---|---|---|
| Traverser bulking | semantic carrier (K) | no |
| Lazy barrier | BUNDLE (merge placement); mechanism itself rejected (F6) | no |
| Path retraction | semantic plan (liveness) | no |
| Incident→adjacent | semantic rewrite, legal under support demand | no |
| Has pushdown | semantic normalisation + BUNDLE access path | no |
| Prepared binding (GValue + pinning) | BIND slot + BUNDLE pin set | no |
| Bound adjacency slice | BIND metadata (ordering witness) + BUNDLE (slice) | no |
| Directional projection | V3 layout + BIND metadata | no |
| Vertex-centric index | V3 layout (extra ordered projection) + BUNDLE scoring | no |
