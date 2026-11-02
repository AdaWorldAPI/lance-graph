# Lab spec: G10 × G11 — prepared land, fenced durable commit, crash recovery, branch-local routing (2026-11-02)

**D-id:** D-TGL-PUB ([STATUS_BOARD](../board/STATUS_BOARD.md)).
**Status:** SPEC, lab only. No production integration, no workflow engine, no extra semantic store, no automatic merge.
**Parent:** [`thought-git-lifecycle-contract-v1.md`](thought-git-lifecycle-contract-v1.md) §5 G10/G11.
**Fence:** EXO-148 stays locked for real durable thought landings. This lab lands a Codegraph source revision and DT64 route intents through a test `WalSink`; it mints no `ThoughtCommitId` or `DurableRef`.

## 1. Code boundaries (read 2026-11-02)

### 1.1 DT64 routing
- `BindStore::route(position, class)` (dt64 `lib.rs`) refuses in only two cases:
  - `UnknownClass`, when the class is not registered;
  - `OutsideReservation`, when `position >= extent.identity.reserved()` (`RouteLog::route`, `dr.rs:255`).
- Otherwise it pushes an event and ticks the shared `VersionClock`.
- `BindStore::classes` is insert-only (`lib.rs:622`, no remove) and `RouteLog` is push-only. A route intent that passes the check therefore stays valid forever. A prepared route cannot go stale.
- **Gap (the one read-only addition needed):** `BindStore` has no check that applies `route`'s two refusals without pushing.
  - Reaching `extent` today requires `resolve(view, …)`, which needs a `ClassView`.
  - Smallest fix: `BindStore::check_route(&self, position, class) -> Result<(), Refusal>`, the same two conditions with no write. It is a pure function of the insert-only registry.
- **Not idempotent:** applying the same `route(p, c)` twice ticks `RouteVersion` twice and appends a second event. Codegraph re-application is idempotent through the digest skip; route re-application is not. Recovery idempotence for DT64 therefore needs the watermark of §1.3.
- **G12 caveat:** the Codegraph `MapTable` has its own clock (`MapTable::new`). The lab keeps the two clocks separate. Joining them through `with_clock` would make Codegraph map growth tick DT64 `RouteVersion`.

### 1.2 Codegraph
- `prepare_refresh(&self)` returns a `PreparedRefresh`, or `None` for an unchanged digest.
- `commit_refresh` refuses `Stale` before any change; it has no error path after that check (#1924).
- Re-applying the same source revision is a no-op (digest skip). Multiplicities follow G8: a source's contribution is replaced, never added twice.

### 1.3 Durable writer (`lance-graph-planner/src/persist_sink.rs`)
- `WalSink::commit_cycle(DetachedCycleBatch)` checks against `batch.frame.base_version`.
  - `Fenced { current_head }` (nothing written);
  - `HashConflict` (fail closed);
  - `InvalidArtifact` (permanent);
  - `Io` (nothing published);
  - `Ambiguous` (outcome unknown; reconcile by re-submitting the same frozen batch).
  - Success is `Committed { version, cycle, batch_hash }`, or `Reconciled { current_head, … }` on a re-submit.
- `scan_sealed(after_cycle)` returns landings **without payload**; durable payloads live only in the coalesced image. Recovery must read land records from the image, never from `scan_sealed`'s slots.
- `recover_and_apply(owner, sealed, applied_through)` replays in canonical `(cycle, stream_position)` order above a durable watermark. **The watermark, not phase equality, is the idempotence key.** On error it returns the partial watermark with the error.
- Production payload ABI: `LanceCycleWriter` requires 512-byte artifacts (`InvalidArtifact`). The lab uses a test `WalSink` and must not claim that ABI.

### 1.4 Crash and replay boundaries

| boundary | durable state | in-memory W0 / DT64 | recovery |
|---|---|---|---|
| B1: crash before `commit_cycle` | unchanged | unchanged | none; the branch is discardable |
| B2: `Fenced`, `HashConflict`, `InvalidArtifact`, `Io` | unchanged | unchanged | regenerate against the current head (Fenced/Io), or stop (HashConflict/InvalidArtifact) |
| B3: `Ambiguous` | unknown | unchanged | re-submit the same batch → `Reconciled` or `Committed` |
| B4: committed, crash before publication | land record durable | unchanged | read records above the watermark from the image; re-prepare; apply; advance the watermark |
| B5: process loss during publication | land record durable | lost | rebuild W0 from the durable base, then replay all records once in canonical order |

The in-memory publication is one infallible step (Codegraph `commit_refresh` + the checked routes + the watermark). Inside one process there is therefore no half-published state. B5 is the process-loss case, and it rebuilds instead of patching.

## 2. The land record (payload of one landing)

`{ source: (crate, file digest, rows content), routes: [(position, class)], base: DatasetVersion }`
- It contains no positions: W0 assigns them at apply (G11, F-LAND).
- It contains no `RouteVersion`: the version is issued at apply.
- **Lab encoding:** a fixed little-endian layout inside the test `WalSink` payload.

## 3. Falsifiers (the smallest common set)

**Fixture:**
- **Codegraph:** the F-LAND sources `m` / `n`.
- **DT64:** a `BindStore` with one registered derived class `C'` and a free position `p` in its reservation.
- **Writer:** a test `WalSink` with a head and an injectable `Fenced`.
- **Branch:** revises source `m` and carries the route intent `(p, C')`.

| # | case | must | disable that must turn it red |
|---|---|---|---|
| L1 | prepare: Codegraph `prepare_refresh` + `check_route` | W0 snapshot, `BindStore::current_version()` and `routed_at(·, p)` unchanged | let prepare call `route` instead of `check_route` |
| L2 | `commit_cycle` returns `Fenced` (base behind) | nothing published: W0 snapshot, `RouteVersion`, `routed_at` unchanged; the branch can re-prepare against the new head | publish before the durable commit (the order G10 forbids) |
| L3 | commit succeeds, crash injected before publication | recovery from the durable record restores exactly the confirmed state: snapshot equals a clean open of the landed sources (call-site counts included); `routed_at(current, p) == Some(C')`; `RouteVersion` is exactly one tick past the base | recover from `scan_sealed` slots (no payload) instead of the image |
| L4 | second recovery with the persisted watermark | no change: multiplicities, snapshot and `RouteVersion` identical to after L3 | drop the watermark: `RouteVersion` ticks again (route is not idempotent) while the Codegraph stays unchanged (digest skip), proving the watermark is load-bearing for DT64 |
| L5 | branch route before land | `BindStore` shows no trace of `(p, C')` (version, `routed_at`, `claimed`) until L3 publishes; after a discard, still none | apply branch routes eagerly at prepare |
| L6 | B5: rebuild from the durable base + replay of all records | equals L3's state; one replay per record | replay records in physical order instead of canonical `(cycle, stream_position)` |

**Anti-vacuity:**
- In L3 the snapshot and `RouteVersion` actually move, compared with L1.
- In L4 the watermark-less twin really ticks.

## 4. What has to exist before the lab runs (each its own reviewable step)

1. **DONE (2026-11-02):** `BindStore::check_route` (read-only; the two refusals of `route`, no write; `route` runs the same check). Its own falsifiers:
   - it agrees with `route` on accept and refuse for every case;
   - it leaves `current_version()` unchanged.
2. **DONE (2026-11-02):** a lab-only test `WalSink` (`LabWal`, `crates/lance-graph-codegraph/src/lab_wal.rs`) beside the Codegraph test (dev-dependency on `lance-graph-planner`; test `WalSink` impls exist in ten places, for example `lance-graph-planner/examples/blw_fusion.rs` `MemWal`). No production edge from Codegraph to the planner.
3. The lab test itself, in-crate, because `prepare_refresh` / `commit_refresh` are crate-private.

Out of scope: `LanceCycleWriter` (needs the 512-byte ABI and real Lance), `ThoughtCommitId`/`DurableRef` (EXO-148), MergeQueue, review gates.
