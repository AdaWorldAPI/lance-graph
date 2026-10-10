# 2026-10-10 — split-tunnel persistence measured in a consumer: sparse append on write, resident fold on read

**Status:** MEASURED (one consumer: HubSPO-rs `hubspo-store::log`, PRs #23–#25) · WORKING-MODEL for other carriers.
**Knowledge doc:** `.claude/knowledge/split-tunnel-sparse-append-resident-fold.md` (the pattern, its rules and how to adopt it).

First measured instance of two OPEN working models:
- newest-first over append-only, from `2026-09-23-stable-row-ids-hold-tombstones-exact-seal-still-owns-cohort.md`;
- the persistence half of the alpha split tunnel / merge-on-read `SparseDelta`, from `2026-10-05-quack-storage-portability.md`.

It does not use or change the `Alpha*` types: `AlphaOverlay` carries no payload, so this is the persistence half beside it.

**Pattern:**
- **Write:** sparse `(version, owner, op, row, lane, present, value)` records, one Lance version per batch. Never updated in place, never compacted.
- **Recovery:** Lance is read once, at replay.
- **Read:** every read is a Quack / mask-risc fold over the same records held resident. Newest wins by append order, with no sort.

**Measured** (release, 2026-10-10; 1.2M log records, 1,000 versions, 1,001 fragments):

| read | median |
|---|---|
| history as a resident fold | **0.72 ms** |
| replay, once at startup | 532 ms |
| history by scanning Lance per read | 336 ms |
| history by address index into in-place updates (P-HIST-1) | 46 ms |
| budget | < 20 ms |

- Sparse appends write 1,000 versions in 9.2–12.4 s, against ~300 s for in-place updates.
- 29 B per resident record.

**DECISION (2026-10-10):**
- Logs of this shape are never compacted: append order is version order.
- Derived reads (history, diffs) write nothing.
- The only new persistent state is a learned outcome, recorded after as one merged, auditable record.
- BASIS: `2026-09-24-cost-scales-with-dirty-rows-replay-budget-before-materialization.md` (one write ≈ 10⁶ folds).

**OPEN:**
- the same pattern over 512-byte `NodeRow` images or V3 facet payloads;
- the `changed_coordinates` representation (still open, unchanged);
- resident sets larger than memory (a checkpoint becomes the recovery read).
