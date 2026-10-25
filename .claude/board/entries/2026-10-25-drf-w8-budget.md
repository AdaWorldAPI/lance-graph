# D-DRF-W8 — eviction under a resident budget while readers hold foci (lab)

**Status:** TEST-PINNED (lab, `cargo test`; no CI job runs this crate).
**Code:** `crates/drf-focus-probe/src/lib.rs` `FocusManager::{evict_to_budget, resident_bytes}`,
`BudgetReport`, `FOCUS_BYTES`. **Tests:** `tests/integration/budget.rs`.
Plan: `.claude/plans/d-dr64-focus-cache-falsifiers-v1.md` §4 W8, §1 (focus churn).

- **Budget unit:** `FOCUS_BYTES` = the gathered i32 column, 16 KiB. The 8 KiB sealed map is
  caller-owned and shared through an `Arc`, so it is not counted.
- **W8 PASS:** with A held by a reader, B and C idle, `evict_to_budget(0)` evicts B and C only
  and reports `over_budget` (resident = pinned = 16 KiB). A still folds to the direct-read
  oracle. Once A is dropped, the next pass evicts it and the manager is under budget.
- **Rebuild PASS:** an evicted focus re-gathers on its next `load` (builds 1 → 2) with an
  identical column and fold.
- **Order PASS:** least recently *loaded* first (the tick is refreshed on a hit). An exact fit
  evicts nothing.
- **Churn, measured:** cyclic switching over 3 foci for 10 rounds (30 loads), one eviction pass
  per load. Budget 3 foci → 3 builds; budget 2 or 1 → 30 builds. One focus short of the working
  set, LRU always evicts exactly the focus needed next, so every load is a full gather.
- **Disables, each red:** evict pinned foci; no recency refresh on a hit; newest first; strict
  `<` fit.

OPEN: the eviction policy is unmeasured beyond these cases. LRU is the worst policy under a
cyclic working set one larger than the budget (a newest-first disable changed the 30-build
result). Whether real thought workloads switch cyclically is not known. A budget-aware caller
could also fall back to direct reads, which the P2 break-even says cost about the same as a
cold build. Neither is decided here.
