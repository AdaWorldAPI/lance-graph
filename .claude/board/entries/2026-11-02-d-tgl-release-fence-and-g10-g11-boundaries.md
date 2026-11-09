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

## Step 1 done: `BindStore::check_route` (2026-11-02)
- **Code:**
  - `BindStore::check_route(&self, position, class) -> Result<(), Refusal>` applies `route`'s two refusals and writes nothing.
  - `route` runs the same check (`route_identity`, then `RouteLog::check`) before it pushes, so the two cannot disagree.
- **Tests** (`crates/lance-graph-dt64/tests/bind_store.rs`):
  - agreement with `route` on 5 cases: 2 accepted, 1 `UnknownClass`, 2 `OutsideReservation` at the 255/256 boundary and beyond;
  - writes nothing: version, `routed_at` and `claimed` unchanged; the twin `route` does move them;
  - a checked intent stays valid as the store grows: other classes bound, other routes and retracts, then applied without a refusal.
- **Disables (each red on the agreement test):** always accept; skip the reservation; skip the class lookup.
  - "Writes nothing" holds by the type (`&self`), so it has no disable. Its twin shows the comparison can fail.
- **Gates:** dt64 suites green (25 + 29); clippy and fmt clean. dt64 already runs in `rust-test.yml`.

## Step 2 done: the lab `WalSink` (2026-11-02)
- **Code:** `LabWal` in `crates/lance-graph-codegraph/src/lab_wal.rs`, `#[cfg(test)]` only. It is an in-memory `WalSink` that follows the production contract:
  - `Fenced` against the head, with nothing written;
  - one version per commit;
  - reconciliation of a re-submitted `(cycle, batch_hash)`;
  - `HashConflict` for another batch of a durable cycle;
  - a payload-free `scan_sealed`.
- **Lab seams:**
  - injected faults: I/O before publish, and `Ambiguous` after publish (B3);
  - `image(cycle)`, the durable payload read that recovery must use.
- **Dependencies:**
  - `lance-graph-planner`, `lance-graph-contract` and `tokio` (rt) are dev-dependencies only; `cargo tree -e normal` shows no planner.
  - The crate's own workspace patches the ndarray git coordinate to the sibling checkout, as the root workspace does. `cargo tree -i ndarray` shows one package; `-d` shows no duplicate.
  - CI's `test` job already checks out ndarray as a sibling.
- **Tests:**
  - stale base fenced, nothing written;
  - a commit publishes one version, and the payload lives only in the image;
  - a re-submit reconciles, a different batch conflicts;
  - a fault before publish and one after publish leave different durable states.
- **Disables (each red on its own test):** drop the fence; leak the payload through `scan`; skip the reconcile lookup; make `Ambiguous` publish nothing.
- **Gates:** codegraph suites green (12 lib tests); clippy and fmt clean.

## Step 3 done: L1–L6 (2026-11-02, MEASURED in the lab)
- **Code:** `crates/lance-graph-codegraph/src/graph/publication.rs` (`#[cfg(test)]`).
  - A branch revises one Codegraph source and carries DT64 route intents. It lands in the G10 order:
    1. prepare (`prepare_refresh` + `check_route`);
    2. durable commit through `LabWal`, with the land record (source + route intents, no positions, no `RouteVersion`) as payload;
    3. an infallible publication (`commit_refresh`, `route`, watermark).
  - Recovery reads records from the durable image above the watermark, in canonical cycle order. The record codec refuses truncated or padded bytes.
- **Results (all green):**
  - **L1/L5:** prepare and discard leave the Codegraph snapshot, `RouteVersion`, `routed_at(P)` and `claimed` unchanged.
  - **L2:** `Fenced` publishes nothing; the branch lands after re-preparing against the new head.
  - **L3:** after a crash between the durable commit and publication, recovery restores exactly the confirmed state: clean-open equality, `RouteVersion` +1, `P → C1`.
  - **L4:** a second recovery changes nothing.
  - **L6:** a rebuild from the base with reversed physical order equals the live state.
- **Disables (each red at the intended assertion):**
  - **L1/L5 eager `route`:** L1, L2 and L3 red.
  - **L2 publish before the durable commit:** "nothing published" red.
  - **L3 read records from `scan_sealed`:** "land record in the durable image" red.
  - **L4 drop the watermark:** the Codegraph snapshot stays equal (digest skip) while `RouteVersion` goes 1 → 2. This is the predicted non-idempotence of `route`, now measured.
  - **L6 drop the canonical sort:** the rebuild replays cycle 2 first, then the watermark skips cycle 1. The record is lost, not merely reordered (`RouteVersion` 1, `P → C2` only). **Physical-order replay plus a watermark loses data silently.**
- **Gates:** codegraph suites green (17 lib tests); clippy and fmt clean.
- **Limits:**
  - The lab `prepare` takes `&mut` only so the L1 disable can call `route`. A production prepare should take `&self`.
  - The `WalSink` is in memory; no 512-byte ABI and no `LanceCycleWriter`.
  - EXO-148 stays locked.
  - Lab results satisfy no production gate.
