# D-DRF-P1 — focus lease + gather along the u16 map (lab probe)

**Status:** MEASURED (lab). **Crate:** `crates/drf-focus-probe` (workspace-excluded).

- Lease begins at `FocusManager::load`; concurrent loads of one key gather once (L1, OnceLock).
- Key = full `SourceId` (dataset, branch, version, index_seal) + `MapId` + `Scope` (G3, G6).
- Stale map (seal mismatch) refused (G4); out-of-range ordinal refused before any read or prefetch.
- Gathered i32 facet column is the D-DRF-1 §5 derived cache; fold via `ndarray::simd::masked_sum_i32`.
- P1: direct (lookahead 0/4/8/16/32/64) == oracle == focus over 5 map shapes × 3 masks.
- 10/10 green on native AVX-512 and x86-64-v3. Disable runs: L1 (bypass OnceLock) and G4 (drop seal check) each red.

OPEN: no timing taken (P2 cost comparison not run); dirty flush (§6) not probed.
