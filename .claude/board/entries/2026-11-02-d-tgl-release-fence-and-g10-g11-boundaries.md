# D-TGL: the generation fence holds in release; G10 × G11 boundaries located (2026-11-02)

**Status:** MEASURED (release run) + SPEC (G10 × G11 lab, plan [`g10-g11-atomic-publication-lab-v1.md`](../plans/g10-g11-atomic-publication-lab-v1.md), D-TGL-PUB).
**Source:** operator review of #1924 (2026-11-02). Production Thought-Git land stays unapproved; EXO-148 stays locked.

## Git hygiene
#1923 was merged at `83ac4b3a8`. The two commits after that point (`ab9fb36f6`, `4025292ea`) are in #1924, which contains current `main` (`2e19a5adc`, merged in, no history rewrite, no force push). Nothing was lost.

## Release profile (MEASURED)
- `cargo test --release --lib` in `crates/lance-graph-codegraph`: 7 passed, including `lifecycle::f_land_*` and `f_discard_*`.
- **Release disable (fence removed):** F-LAND fails on its own assertion: `a stale preparation must be refused, got Ok(RefreshStats { edges_added: 6, symbols_appended: 1, .. })`.
  - With `debug_assert` compiled out, the stale preparation is **accepted silently**.
  - The generation fence is the only guard, and it refuses correctly on its own.
  - The debug-build catch (`artifact.rs:186`) was a second net, not the guard.

## G10 × G11 boundaries (read in code)
- **DT64 `route`:**
  - It refuses only `UnknownClass` / `OutsideReservation`.
  - The class registry is insert-only and the route log push-only, so a checked route intent never goes stale.
  - **Gap:** no read-only `check_route` exists.
  - **Re-applying `route` ticks `RouteVersion` again (not idempotent).**
- **Codegraph:** re-applying a source revision is a no-op (digest skip). DT64 recovery therefore needs a watermark; Codegraph recovery does not.
- **Writer (`persist_sink.rs`):**
  - The refusals and outcomes are `Fenced` / `HashConflict` / `InvalidArtifact` / `Io` / `Ambiguous` → `Reconciled`.
  - `scan_sealed` returns landings without payload, so land records must be read from the image.
  - `recover_and_apply` is idempotent by watermark, not by phase.
- **Five crash boundaries:** B1 before commit; B2 refused; B3 ambiguous; B4 committed but not published; B5 process loss. Plan §1.4.
- **The smallest common falsifier set is L1–L6.** Each has its own disable. The decisive one is L4: without the watermark, a second recovery ticks `RouteVersion` while the Codegraph stays unchanged.

OPEN: `check_route`, the lab `WalSink`, and the L1–L6 test are three separate reviewable steps (plan §4). None is implemented.
