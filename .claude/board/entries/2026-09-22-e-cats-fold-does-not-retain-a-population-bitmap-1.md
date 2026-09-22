## 2026-09-22 — E-CATS-FOLD-DOES-NOT-RETAIN-A-POPULATION-BITMAP-1 — CATS aggregate lowers to one tiled grouped terminal; bitmap realization is a requested boundary sink

## 2026-09-22 — SAP/CATS fold-first dependency stack (PR #1257, unmerged)

- Substrate: ndarray #318 `d0376505`; lance-graph #1256 `4d27032e`.
  #1246 and #1247 are already ancestors. SAP CI pins the ndarray commit.
- `CatsQuery` lowers one `Agg::GroupSumI32` through Quack. Bounded tile
  scratch remains; no population bitmap is retained by binding or aggregation.
  `select_into` requests a bitmap only for the terminal BAPI assignment sink.
- Removed the unused all-ones alpha allocation. Positive hours are checked
  at binding so every subset/group fits I64 despite the kernel's wrapping sum.
- A rotation changes coordinate-to-ordinal references; it does not transpose
  carrier storage. The existing BAPI ordinal permutation is an edge read map.
- SAP tests pass across 0/1/63/64/65/131/4097 inputs, against independent
  arithmetic and the substrate reference interpreter; repeat resets, bounded
  scratch, zero hot allocations and original-assignment boundary selection.
  Strict SAP clippy passes. No nanosecond performance claim is made.
- Inventory: isolated `lance-graph-sap` consumer contains source descriptors,
  `CatsSchema: ClassView`, column binding to existing `LaneRef`, cold codebooks,
  Quack orchestration and explicit C#/hash/BAPI sinks. No SAP execution engine.

