# Split-tunnel persistence: sparse append on write, resident fold on read

**READ BY:** any session designing a write path, a history / audit / "what
changed" read, a replay or recovery path, or anything that reads a Lance
dataset on a hot path. Also `v3-kanban-executor-engineer`,
`fold-carrier-scientist`, `query-stage-profiler`, and anyone touching
`contract::alpha` / `alpha_tunnel` who wants the persistence half.

**Status:**
- MEASURED in one consumer (HubSPO-rs `hubspo-store::log`, PRs #23–#25, 2026-10-10).
- VERIFIED-IN-CODE there.
- WORKING-MODEL for every other consumer and carrier shape (see § Scope).

**BASIS:**
- `board/entries/2026-09-24-cost-scales-with-dirty-rows-replay-budget-before-materialization.md`:
  one fold ≈ 1.7 ns, one write ≈ 1.7 ms floor, so one write buys about 10⁶ folds.
- `board/entries/2026-09-24-three-clocks-never-meet-implicitly.md`: a Lance scan is
  recovery-clock cost (334 ms at 1,001 fragments).
- `board/entries/2026-09-23-stable-row-ids-hold-tombstones-exact-seal-still-owns-cohort.md`
  § "newest-first over append-only" and
  `board/entries/2026-10-05-quack-storage-portability.md` § "OPEN — SparseDelta". This doc
  is the first measured instance of both working models.

**REVISIT WHEN** a consumer's resident records no longer fit memory, or a read
needs data from before the last replay that the resident set no longer holds.

## The pattern in one picture

```
            WRITE path (persistence clock, ms)          READ path (live clock, ns)

batch ──► check ──► encode sparse records ──► Lance append   (one version per batch)
                          │                                   never updated in place
                          │                                   never compacted
                          └──────────► resident records ◄──── fold (Quack / mask-risc)
                                              ▲                 history, "what changed",
                                              │                 newest-wins merge
                         Lance ──► replay ────┘  (read ONCE: startup / recovery)
```

The **split tunnel** here is the separation of the two paths, which is the
same separation `alpha_tunnel.rs` makes in memory. Reads never go through
storage, and writes never go through the read structures. The writer appends
sparse records to Lance; readers fold over the same records held in memory.
`AlphaOverlay` carries no payload, so it cannot be the persistence half
(`2026-10-05-quack-storage-portability.md` § "OPEN — SparseDelta": "Alpha does not supply that signal (it carries no payload)"). This pattern is that
persistence half. It is **not** a use of the `Alpha*` types.

## The rules

1. **Write only what changed, as sparse records.** One record per changed
   coordinate: `(version, owner, op, row, lane, present, value)` in the reference
   implementation, with `op ∈ {BORN, SET, DELETE}`. An insert writes BORN plus one
   SET per lane. An update writes one SET. A delete writes one DELETE. Never a
   full-row image of unchanged lanes.
2. **One batch = one Lance version, from exactly one writer.** The single
   writer comes from **ownership**: one writer per mailbox (`mailbox_owner`,
   write on behalf). The version checks do not enforce it.
   - The reference implementation checks the dataset's latest version before
     the append and after it (VERIFIED-IN-CODE: `log.rs` `commit`).
   - The check before the append refuses a *stale* writer (one that is behind
     before it writes).
   - It cannot refuse a *racing* writer. Two writers that both pass the check
     both append, because Lance rebases an `Append` instead of failing it
     (`spog-alpha-channel-v1.md` F6: a read-then-append guard is a TOCTOU and
     cannot "refuse, not renumber").
   - The check after the append only *detects* that race, once the second batch
     is already durable. Replay then refuses the log, which stays refused until
     reconciled.
   - The durable fix is in-band idempotency, `(cycle, batch_hash)` in the same
     commit and reconciled first (F6). The reference implementation does not
     have it yet (OPEN).
3. **Never update in place, never compact.**
   - DECISION 2026-10-10, BASIS: append order is version order.
   - So "newest wins" is applied by replaying in append order, with no sort.
     A log out of order is refused at replay, as a batch whose version is not
     the next one.
   - Compaction would add Lance versions and break the version ⇔ batch mapping.
   - It is also unnecessary: the read path never scans Lance.
4. **Lance is read exactly once per process: at replay.**
   - Replay decodes every record, applies them through the same `apply` a live
     writer uses, and only on success installs them as the resident set.
   - A refused replay leaves the resident set unchanged, so a corrupt record can
     never reach a reader.
5. **Every committed batch's records also stay resident.**
   - `commit` appends them to the resident set after the Lance append succeeds.
   - A writer whose resident set is not at the table's version refuses to
     commit; otherwise reads would silently miss earlier batches.
6. **Reads fold over the resident records.**
   - Example: the history of `(row, lane)` is one Quack program,
     `row = r ∧ ((lane = c ∧ op = SET) ∨ op = DELETE)`, run on mask-risc over
     the resident `U32` lanes. Only matches are decoded.
   - No index, no column projection, no Lance filter expression (so no
     DataFusion), no compaction.
7. **Derived reads write nothing.** History, diffs and "what changed since v"
   are recomputed by folding (rule 6), never stored.
   - DECISION 2026-10-10: the only new persistent state is a *learned* outcome,
     recorded after the fact as one merged, auditable record.
   - BASIS: rule-of-thumb 10⁶ folds per write.

## Measured (reference implementation)

HubSPO-rs `crates/hubspo-store/examples/probe_history.rs`.
- Run: release, one container, 2026-10-10.
- Data: 100k records, 1,000 versions × 1,000 updates, 1,001 Lance fragments,
  1,199,991 log records.

| read | median (range) | runs |
|---|---|---|
| history, resident fold (this pattern) | **0.72 ms** (0.64–1.99) | 101 |
| replay from Lance (once, at startup) | 532 ms (447–611) | 5 |
| history by scanning Lance on every read (#24) | 336 ms (320 ms of it the scan) | 7 |
| history by address index into in-place updates (P-HIST-1) | 46 ms best | — |
| budget (P-HIST-1) | < 20 ms | — |

- **The read** costs about 0.6 ns per resident record, consistent with the
  10⁶-folds-per-write rule.
- **The scan** is the whole cost of a Lance-read history: the program itself
  is 3–15 ms of it.
- **Removing the sort** in favour of append order was speed-neutral: run-to-run
  noise (up to 46 ms) exceeded every difference between variants.
- **Writes** (P-SPARSE-1, same consumer): 1,000 versions of sparse appends take
  9.2–12.4 s, against ~300 s for in-place updates. The appends leave no deletion
  vectors to read.
- **Memory:** 29 B per resident record in the reference shape (1.2M records ≈ 35 MB).

**Disable runs, all red** (`tests/log.rs`, each guard removed alone):
- resident-version check in `commit`;
- records installed before replay validates;
- `commit` not appending to the resident set;
- the `row`, `op = DELETE` and `op = SET` terms of the history program;
- the I32 sign bit.

A replayed log gives histories identical to the one that wrote them.

## Scope — what is and is not shown

- **Shown:** one consumer, one carrier shape (u32 lanes, I32 as bit pattern),
  one writer per log, single-process recovery.
- **Not shown, and not provided by this pattern:** protection against two
  concurrent writers (rule 2). Ownership must guarantee one writer. A race is
  only detected afterwards, and in-band reconciliation is OPEN.
- **WORKING-MODEL, not shown:**
  - 512-byte `NodeRow` images or V3 facet payloads as the record;
  - field-bitmap `changed_coordinates` (the open `SparseDelta` representation
    question stays open);
  - multi-mailbox logs;
  - a resident set larger than memory (then a checkpoint, a base written once
    plus the tail since it, becomes the recovery read, still never a hot-path
    read).
- **Not a claim:** that Lance scans are slow in general. They are recovery
  cost, measured here; the pattern just keeps them off the read path.

## How to adopt it

1. Define the sparse record for your carrier (rule 1). Keep `version` and
   `owner` on every record.
2. Wrap the writer as `check → append one version → apply → extend resident`,
   in that order (rules 2, 5).
3. Write `replay` as the only reader of Lance (rule 4). Install the resident
   set only after every batch applied.
4. Express every read as a Quack program over the resident lanes (rule 6).
   If a read seems to need Lance, it is a recovery read, not a hot-path read.
5. Measure with a probe that times the fold and the replay separately, and
   pin the guards with disable runs.

Reference implementation: HubSPO-rs `crates/hubspo-store/src/log.rs` (`Log::commit`,
`Log::replay`, `Log::history`) and `tests/log.rs`.
