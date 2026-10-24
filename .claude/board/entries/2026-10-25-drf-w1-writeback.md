# D-DRF W1–W3 — staged writeback coalesced per canonical row (lab)

**Status:** TEST-PINNED (lab). **Code:** `crates/drf-focus-probe/src/lib.rs` (`Focus::writeback`,
`Writeback::{stage, dirty_rows, flush}`, `RowImage`). **Tests:** `w1_*`, `w2_*`, `w3_*` in
`tests/integration/focus.rs`; 14/14 green on native AVX-512 and x86-64-v3.

- **W1 PASS:** 4096 slots over 37 distinct rows, 64 thoughts → exactly 37 row images
  (37 × 512 = 18 944 B), in ordinal order, bytes outside the written facet equal to the
  sealed source. A1 (dirty fraction) for this workload = 37 / 4096 slots = 0.009.
- **W2 PASS:** two slots aliasing one row write different facets; the single image keeps
  both, identical for both staging orders.
- **W3 PASS (refusal arm):** two different values for one facet of one row →
  `Refusal::Conflict`; equal values coalesce. Two facets whose byte ranges overlap
  (offsets 16 and 17) also refuse; facets that only touch both survive (codex P2 on #1765).
  Stale source at flush → `StaleMap`.
- **Disables, each red:** one image per patch (W1/W2/W3), last-payload-wins row replacement
  (W2), conflict check removed (W3), overlap clause removed (W3 overlap).

OPEN: W3's "declared stream position" ordering (deterministic noncommuting apply) is not
built; this probe only refuses. W4–W8 (owned handoff to a CycleWriter, WAL, durability,
eviction under pressure) not run. Images are owned copies by design: they are the frozen
batch a writer takes ownership of (W4), not a read path.
