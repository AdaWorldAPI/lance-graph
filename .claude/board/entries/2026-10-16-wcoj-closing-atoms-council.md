# 2026-10-16 — WCOJ closing atoms: 5+3 council record

**Status:** RATIFIED SPEC (`.claude/plans/wcoj-closing-atoms-v1.md`); not
authorized to build. Follows the P-7 insufficiency in
`2026-10-16-wcoj-probes-1-3.md`.

## Outcome

- **Part A, `Pred::EqU32ViaLane { fk, key, lane }`**: `EqU32Via` with the
  constant replaced by the row's own lane. Comparison family, three-valued,
  O(1) per row. Closes a cyclic body over functional relations.
- **Part B, `MaskOp::RunsMeet { .. }`**: for each row of R-edges, do two sorted
  runs share a value. EXISTS family, beside `Gather{under}`, Semijoin's NULL
  rule. Closes a cyclic body over n:m relations; elements touched ≤ N^1.5 [H].
- The two answer different questions over different head populations (node
  rows vs R-edge rows); they coincide only when R is functional.

## What changed between rounds

| round | finding | consequence |
|---|---|---|
| premise | PREMISE-SPLIT: Part B is an EXISTS over a set, not a comparison | split into Q-A / Q-B; Part B gets Semijoin's NULL rule and a gated execution |
| S1 | no prior ruling; planner `AdjacencyBatch::intersect` and `WcoJoin` are stale parallels | not reused |
| S2 | `if let Filter::Semijoin` (`quack/lib.rs:584`) would misroute a new EXISTS leaf under `NOT` | becomes `matches!` over both |
| S3 | F2/F3a/F3b fixtures do not exist yet; the ndarray "no gather" doc is overstated | G0 writes them; sibling doc corrected |
| S4 | exhaustive `ExecError` match in lgj-abi; Pred `_under` absent in the Via family | Part B is a `MaskOp`; new error variant paired with lgj-abi |
| S5 | re-anchoring does not remove Part B; k-way is not AND of 2-way | decisions 10, 11 |
| R1, R2 | G5 NULL-under-`NOT` polarity was inverted | fixed: a NULL key is KEPT under `NOT` |
| R1 | visited-only order check made Ok/Err depend on the gate | complete check at validate |
| R1 | quack gates `RunsMeet` on the first resident plane, not all earlier conjuncts | stated; no "rejected rows do no work" claim |
| R1 | G3 compared R-edge rows with head rows `a`; G2 could not fail; G4 metric changed | edge-level oracle + projection; G2 vs the disjunction; G4 on elements touched |
| R3 | a refusal inside the tile loop leaves a partly written sink | order invariant checked before any tile; `ForeignLaneNotOrdered` |

## OPEN

- Authorization of D-ASM-11 (still gated on D-ASM-2).
- Part B's insufficiency is argued, not measured; G0 measures it.
- The sealed CSR witness that removes the per-execution O(|S| + |T|) check.
- A bounds-masked u32 gather backend for Part A (TECH_DEBT).
