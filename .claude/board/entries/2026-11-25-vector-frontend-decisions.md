# Vector as a frontend of the native fabric — operator decisions on the open items

**Status:** RECORDED. Operator decisions, 2026-11-25.

DECISION: Vector is another frontend to the native algebraic fabric, beside Quack, and it must not compromise the fabric's invariants. No parallel execution or indexing architecture enters lance-graph for VRL compatibility. The method decides what is evaluated, and the row type decides how stored state is read; neither VRL nor Quack owns stored semantics. Compatibility is a property of lowering, not of storage or execution.

| # | Item | Decision | Instruction |
|---|---|---|---|
| 1 | `==`/`!=` over optional fields (Q1) | Modify | Keep native presence semantics. Refuse ambiguous optional-field comparisons until presence-aware lowering is available. First investigate whether the deviations can be expressed through existing presence-aware methods. No `VrlCompat` execution mode. Keep the deviations as a permanent differential corpus. |
| 2 | Normative VRL (Q2) | Approve | Vector's pinned VRL (7738f74) is normative. OpenObserve (0.31) stays a differential witness. |
| 3 | Append scaling (`ISS-INTAKE-APPEND-LEAVES-THE-FOLD`) | Modify | DT64 tile bounds are acceptable only as a derived description of the DR tiles, never a second key registry. Skipping tiles is not enough: remove the global tile scan. Investigate ordered tile boundaries with binary range resolution, and one sorted batch merged monotonically against tile ranges. Benchmark both against growing existing table size. |
| 4 | Vector fork commits | Access needed | Preserve the local commits now: done, as patches in `.claude/handovers/vector-fork/`. Push once access exists. Full binary CI is mandatory. |
| 5 | Floats | Approve | No silent float conversion. Exact decimals stay exact from the earliest ingestion boundary; an existing `f64` rendered as a string does not recover precision. |
| 6 | Lance persistence | Approve | Close the DR-frame-to-Lance witness gap first. |
| 7 | Dedup after restart | Modify | Rebuild identity membership from the canonical DR lanes (the four id lanes). No independent authoritative dedup registry. Shares one correctness contract with #6. Test failure at each boundary: before append, after append but before commit, after commit but before acknowledgement, after acknowledgement. Measure the startup cost, and avoid rebuilding a heap set that duplicates DR. |
| 8 | Enrichment binding | Approve | Measure first, including allocations and materialization. Bind once (table identity, field binding, condition structure, algebraic method) and supply per event (lookup value, operands, presence, input lanes). |
| 9 | Null versus missing | Approve | Add explicit null, missing, zero, false and empty-string cases. |
| 10 | Workspace `fmt` (`shakmaty`) | Approve | A separate maintenance PR. |
| 11 | One `vrl` identity | Approve | Keep one resolved `vrl`, and keep the pins in sync. |

Execution order:
1. Preserve the Vector commits.
2. The presence differential: explicit nulls and the deviations.
3. Append scaling with no second registry.
4. The DR-to-Lance witness.
5. Persistence, restart and replay together.
6. Enrichment binding.
7. Full Vector CI and the end-to-end differential.
