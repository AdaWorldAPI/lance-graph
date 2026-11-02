# P4 operator decisions: GitHub-analogue Thought-Git lifecycle; F-LAND / F-DISCARD lab result (2026-11-02)

**Status:** lab DONE (Codegraph, crate-private, no Thought-Git wiring); contract written; production gates OPEN.
**Source:** [P4 review](2026-11-02-p4-thought-git-codegraph-dt64-review.md); plan [`thought-git-lifecycle-contract-v1.md`](../plans/thought-git-lifecycle-contract-v1.md) (D-TGL).

## Decisions (operator, 2026-11-02)
- **North Star:** Thought-Git maps Git/GitHub processes onto DR³ and DT64.
  - W0 is main; W1..63 are generation-qualified reusable slots; `ThoughtCommitId` is slot-independent.
  - Lifecycle: `branch → change → diff → PR → checks → review → rebase/squash → merge/land → receipt/replay`. A refusal leads to rework or discard.
  - Shared by Codegraph, DT64, XLOG and later participants.
- **F-LAND / F-DISCARD:** approved as isolated Codegraph lab tests.
- **DRed exception:** conditional. Replace-and-recompute is allowed only for Codegraph facts that are fully and deterministically reconstructible from immutably identified source artifacts. It does not apply to derived XLOG facts, incremental conclusions or DT64 routing.
- **G8:** merge source revisions, never raw edge ADDs. The rules are in plan §4.
- **New production gates:**
  - G9: lifecycle as a contract;
  - G10: atomic publication (prepared changes, production fence, controlled W0/DT64 publication, replay or reconciliation after a crash);
  - G11: branch-local routing;
  - G12: identities kept apart.

## Lab result (MEASURED)
- `refresh_crate` is now `prepare_refresh` (`&self`; returns an owned `PreparedRefresh`, or `None` for an unchanged file) followed by `commit_refresh`.
  - `commit_refresh` first checks the graph generation the change was prepared against, and refuses `RefreshRefusal::Stale` before changing anything.
  - The generation is bumped by every commit and by `refold`. It is local and in memory: not an identity, not persisted.
- **F-LAND** (`graph.rs` `mod lifecycle`):
  - Setup: two branches over one W0 (source `m` gains `m::x`; source `n` gains `n::y`). Both preparations start their new symbol at the same free position (asserted).
  - After A lands, B's old preparation is refused `Stale` and the full-state snapshot is unchanged.
  - B is then recomputed against the current W0 and lands at a distinct position.
  - Every symbol's callers, call-site count, callees and impact equal a clean open of both revised sources. `m::b` has 5 call sites from two sources.
- **F-DISCARD:** dropping a prepared branch leaves the snapshot unchanged and issues no map-table version. The snapshot covers interner, positions, `last_owner`, lanes, digests, table version and history, cube frame and logs, masks, cached closures and generation. A twin that commits the same change shows the snapshot does move.
- **Disable runs (red):** dropping the generation check, or not bumping the generation, turns F-LAND red.
  - **Refusal finding:** in both disables the stale change is caught only by the interner's `debug_assert` "interner changed under the stage" (`artifact.rs:186`). A release build would apply the colliding positions silently.
  - The fence is therefore the load-bearing guard, not a convenience.
- **Gates:** codegraph 7 lib tests + all suites green; clippy and fmt clean.

## Limits stated
- The Codegraph has no `BindStore`, so F-DISCARD's "global `RouteVersion` unchanged" holds by construction. DT64 routing staging is open (G10/G11: no prepare/validate route API on `BindStore`).
- `PreparedRefresh` / `commit_refresh` are crate-private. Nothing publishes, persists or routes.
- Lab results do not satisfy G9–G12 or EXO-148.

## Correction (codex P2 on #1924, fixed in #1925)
- **Defect:** a read between `prepare_refresh` and `commit_refresh` (`impact` and `closure` take `&self`) could fill a closure cache the preparation had not seen. That cache was built from the pre-commit edges. `commit_refresh` kept it, because the prepared plan said "no cache", and the generation fence does not move on a read. `impact` then answered from the old graph: `m::x` was missing as a caller of `m::b`.
- **Fix:** when the batch changes edges, `commit_refresh` drops every closure cache it has no prepared refresh for.
- **Test:** `lifecycle::a_closure_cached_between_prepare_and_commit_is_not_served_stale`, which compares the result with a clean open.
- **Disable:** dropping the new arm turns it red in debug and release.
- **Why the lab tests missed it:** they never read between prepare and commit. Interleaved reads are part of the lifecycle (a PR is read while it waits), so the fixture now covers that case.
