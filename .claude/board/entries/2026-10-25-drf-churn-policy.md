# D-DRF-CHURN — LRU vs MRU victim order under focus churn (lab)

**Status:** MEASURED (lab, `cargo test`; no CI job runs this crate).
**Code:** `crates/drf-focus-probe/src/lib.rs` `Evict`, `FocusManager::evict_to_budget_by`
(`evict_to_budget` stays LRU). **Tests:** `tests/integration/budget.rs` `churn_*`.
Follows `2026-10-25-drf-w8-budget.md` (its OPEN point on the policy).

One eviction pass after every load, 3 foci. Builds counted; each expected count was worked out
by hand before the run, and all matched.

| workload | budget | LRU | MRU |
|---|---|---|---|
| cyclic ABC x10 (30 loads) | 3 | 3 | 3 |
| cyclic ABC x10 | 2 | 30 | 12 |
| cyclic ABC x10 | 1 | 30 | 21 |
| phased AAAA BBBB CCCC x2 (24 loads) | 2 | 6 | 10 |

- Neither order wins both workloads. MRU keeps two of three cyclic foci hot; under phased use it
  evicts the focus just loaded, so every load in the third phase rebuilds.
- **Disable, red:** making `Evict::Mru` sort like LRU turns both `churn_*` comparison tests red.

OPEN: which access pattern thought workloads have is still unmeasured, so no default change.
A scan-resistant policy (one that adapts between the two) or a direct-read fallback are the
next candidates; neither is tried here.
