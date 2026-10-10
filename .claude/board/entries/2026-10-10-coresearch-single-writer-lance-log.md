# 2026-10-10 — coresearch: refuse, not renumber, for a single-owner append-only Lance log

**Status:** OPEN. Exploration map only; the council ratifies nothing. No probe has run.
**Source:** HubSPO-rs `ISS-LOG-RACING-WRITER` (codex P1 on lance-graph #1466),
`plans/spog-alpha-channel-v1.md` F6, and rule 2 of
`knowledge/split-tunnel-sparse-append-resident-fold.md`.
**Code read:** lance and lance-table **13.0.0** (registry copy),
`HubSPO-rs/crates/hubspo-store/src/log.rs`, and
`crates/lance-graph/src/graph/cycle_sink.rs`.
**Harness:** `agents/coresearch-council.md`.

## The question, as split by the premise gate

The first question ranked five fixes as if they answered one question. They
answer two, on different clocks, and "version" meant two different numbers:

- **V-L** is the Lance version. Lance assigns it.
- **V-T** is the table version. The owner declares it on every record (`log.rs:300`).

The two questions:

- **Q-A (commit time, store side).** Can unpatched Lance 13 publish an append
  only while the head still equals the writer's expected version, and refuse
  otherwise?
- **Q-B (replay time, reader side).** Ownership has failed and two durable
  batches of one owner declare the same V-T. What lets replay name the foreign
  one, and is the outcome an accepted history or a named refusal?

A single writer instance per mailbox is ownership's job (one writer per
mailbox). It is not on this menu.

## What the council found

1. **The literature splits cleanly along the two clocks.**
   - A store-side refusal is always a check inside the commit path: an
     expected version, a compare-and-swap, an append position, a fencing token
     the store compares, or OCC with a non-empty read set.
   - Anything carried in-band without such a check is a witness that readers
     use. It can detect and reconcile, never refuse.
   - Sources (all SECTION-READ): Herlihy, TOPLAS 1991 (read-then-write has
     consensus number 1, which is the formal TOCTOU; CAS has ∞); Kung &
     Robinson, TODS 1981 (a blind append has an empty read set, so it always
     validates); Kleppmann 2016 and Chubby, OSDI 2006 (a token helps only where
     the store compares it).
2. **Every lakehouse format renumbers appends.** Delta, Iceberg and Lance each
   refuse a second writer at the storage primitive (put- or rename-if-absent on
   the version slot) and then rebase in their retry loop. Refuse-not-renumber
   belongs to the retry policy over a refusing primitive.
   - Lance 13 is the same (VERIFIED-IN-CODE):
     - Append vs Append is `Ok` (`conflict_resolver.rs:1598-1610`).
     - Strict mode exists only for Overwrite with zero retries (`io/commit.rs:1600-1603`).
     - Every attempt rebases first.
     - `CommitConfig` still reads `// TODO: add isolation_level`.
   - D-LNC-5b measured the same on lance 11 and 12: 28 commits gave 28 versions, none refused.
   - Upstream issue lance-format/lance#8699 (open, maintainer-assigned) asks for a strict expected-head commit mode.
3. **A public extension point gives Q-A a candidate answer without patching
   Lance (X1, CLAIMED until probed).** `WriteParams.commit_handler` is honoured
   on the URI path, including the head load at write time (`insert.rs:418-447`).
   - Lance passes the handler the already-rebased target version (`io/commit.rs:1693, :1716`).
   - A handler that returns `CommitError::OtherError` is verified and returned
     with no rebase and no retry (`:1867-1912`).
   - The proposed wrapper refuses any target other than `before + 1` and
     otherwise delegates to the inner put-if-absent handler. Together they form
     a CAS on the owner's expected slot.
   - Conditions, each VERIFIED-IN-CODE:
     - **Delegate every trait method.** The trait defaults
       `is_version_not_found_definitive = false` and
       `propagate_commit_error_after_success = true` would otherwise turn a
       refusal into `commit_status_unknown`, and would break Lance's own
       lost-ack recognition (PR #7722).
     - **Refuse with `OtherError`, never `CommitConflict`.**
     - **Build the inner handler from an allowlist.** Unknown URL schemes fall
       back to `UnsafeCommitHandler` (lance-table `io/commit.rs:1261`), where
       both racers report success and one batch is silently lost.
     - **`s3+ddb` cannot take a custom handler on the URI path** (`write.rs:2653-2657`).
     - **The trait is marked `// TODO: pub(crate)`** (lance-table `io/commit.rs:875`),
       so this is a temporary seam until #8699 lands.
4. **The persisted transaction can name a rebased racer at no cost per record
   (Y1, CLAIMED).**
   - A rebased Append keeps its original `read_version`
     (`conflict_resolver.rs:2250-2260`), and `read_transaction_by_version` is public.
   - Pre-registered limit: a racer that loaded the head after the winner landed
     (shape R2) is not rebased, so `read_version` cannot flag it. The V-T column
     catches it.
   - The discriminator lives in old manifests, so `cleanup_old_versions` with
     any `before_*` policy erases it.
5. **Hidden hazards in today's log, found while reading `log.rs` (VERIFIED-IN-CODE).**
   - **Two raced batches merge into one decoded run.** They declare the same
     V-T, so they fail as `Corrupt("born out of order")` (`log.rs:474-480, :555-561`)
     rather than as a race. The log is still refused; the reason is mis-named.
   - **`LogError::Diverged` means opposite things.** Before the append
     (`:286-298`) it means nothing was written. After the append (`:323-327`) it
     means the batch is durable and the log needs reconciling.
   - **Any non-batch Lance version on the log dataset wedges it permanently.**
     Examples are a CreateIndex (`log.rs:42` lists "an index on `row`" as a
     roadmap item), UpdateConfig and Restore. Every later commit is refused, and
     replay refuses the whole log.
   - **"Never compacted" does not cover cleanup.** An explicit
     `cleanup_old_versions` deletes old manifests and transactions.
6. **Corrections to prior art.** Each lands as an append-only storno in its
   owning file, never as an in-place edit.
   - (a) F6 calls `sealed_version = base_version + 1` "a verified identity". The
     shipped writer checks only before the append and accepts the returned
     version (`cycle_sink.rs:775-786, :878-885`), so the identity holds only
     under one writer.
   - (b) `find_frame` returns the first row of `kind = 0 AND cycle = N` with no
     ordered scan (`cycle_sink.rs:558-596`). Two frames at one cycle are never
     detected.
   - (c) D-LNC-3's pre-registered fence falsifier expects "lance retry left
     enabled must NOT produce a commit". That is false for a plain Append on
     lance 13, and true only under X1.
   - (d) `temporal/08-memwal-vs-batchwriter.md` claims MemWAL `writer_epoch`
     "gives the same guarantee WITH replay". It fences WAL appends, not dataset
     commits (`mem_wal/manifest.rs`, `wal.rs`).
   - (e) HubSPO-rs: the post-append check is green under its disable run
     (`LATEST_STATE.md:135-139`), so the detection half has no test.
7. **The workspace holds three incompatible offset readings.**
   - (i) offset = the Lance-assigned version: HubSPO-rs H-ADR-15, and the
     persistence plan's "accept the actual returned version".
   - (ii) the producer declares the version, and V-L == V-T: `log.rs`, and
     split-tunnel rule 3.
   - (iii) the semantic generation is not the dataset version: the 2026-10-05
     quack-storage entry, for `cycle_sink`.
   - Under X1, (i) and (ii) become the same number for this log, by
     construction. That holds only while the dataset never receives a non-batch
     version.

## Exploration map

| id | idea | verdict | why |
|---|---|---|---|
| H0 | Race harness: a test commit handler parks writer B inside `commit()` while A lands, and reproduces ISS-LOG-RACING-WRITER deterministically | PROBE (first; gates every race arm) | Makes the race schedulable, so the untested detection half (6e) gets its can-fire test |
| X1 | `PinnedSlot` commit-handler wrapper, refusing any target ≠ `before + 1`; optionally `with_max_retries(1)` through `CommitBuilder` (X6) | PROBE (P-SLOT-1) | Firewall verdict: CONFLICT behind a named seam, a single `hubspo-store` module that owns the lance-table / lance-io / object_store imports, the inner-handler allowlist and full delegation. Retired when #8699 lands |
| Y1(1) | Zero-cost replay pre-check: runs of equal V-T must read 1, 2, 3, …, else a named `LogError::Raced` before decode | ADOPT-NOW candidate | Fixes the mis-named refusal (5a); no reads, no dependencies |
| — | Split `LogError::Diverged` into "nothing written" vs "durable, reconcile" | ADOPT-NOW candidate | Fixes 5b; naming only |
| Y1(2) | `read_version` names a rebased racer at replay, read only where V-T already flags | PROBE | A naming aid for R1-shaped races; dies as the sole discriminator by pre-registration |
| Y5 | Replay outcome: named refusal (MV) vs skip-on-fold (LWW) | PROBE, then operator decision | Pre-registered: under V-L == V-T no accepting rule exists without delete or compaction; LWW needs the relaxed reading "append order is version order" |
| Z1 | Offset readings (i) and (ii) coincide under X1 | PROBE (after X1), then board amendment of H-ADR-15 | — |
| W1c | Re-run D-LNC-3's falsifier on lance 13: plain Append, X1, zero retries | PROBE (after X1) | Its expectation is false for a plain Append |
| W1a–e | Prior-art corrections | ADOPT-NOW candidate (board only) | Append-only stornos in the owning repo |
| X2 | Upstream strict expected-head mode (#8699); predecessor-conditioned commit defined in lance-table but unwired | PARK (watch item + compile canary) | A1: upstream is authoritative. Commenting on #8699 is a public write and the operator's call |
| X7 | MemWAL shard epoch claim as a handover fence | PARK | Its check sits outside the commit (TOCTOU); each claim adds a non-batch version; writer leases are deliberately absent (persistence plan §2) |
| Y3 | Once-per-batch header instead of per-record `version` / `owner` (≈ 29 → 17 B per record) | PARK | Out of scope; would make the log's identity depend on old manifests surviving cleanup |
| X3 | Tag-create successor claim | SKIP (fallback only if X1 dies) | A string id; one permanent object per batch; an orphan claim blocks the slot |
| X4 | `merge_insert` keyed on V-T | SKIP | Reads Lance on the write path, uses internal DataFusion, and turns every batch into an Update |
| X5 | UpdateConfig "writer epoch" key | SKIP | No old-value compare; compatible with Append |
| X8 | Writer lease in code | SKIP | Deliberately absent (persistence plan §2); a deployment concern |
| Y2 | Per-batch writer-epoch witness | SKIP today | Needs an epoch authority (X7 or a lease, both parked) and a new identity |
| Y4 | Per-batch content fingerprint plus an expected-behind rule at replay | SKIP for this log | Under X1 the trigger never fires; the likeliest duplicate (a caller retry after a lost ack) lands at a different V-T and belongs to the producer's idempotency key |

## Probes (pre-registered; full cards in the session scratchpad)

1. **H0 — race harness, no X1.**
   - Expect: 5 versions; markers 100 and 200 durable once each; B gets exactly
     `Diverged { expected: 4, found: 5 }`; replay refuses.
   - KILL (harness vacuous): versions ≠ 5, marker 200 ≠ 1, handler entries ≠ 2, or replay `Ok`.
   - Disable: remove `log.rs:323-328`.
2. **P-SLOT-1 — X1, over races R1 and R2, a recover arm, and a create race at version 0.**
   - KILL if any of:
     - a version > 4 exists, or marker 200 is durable;
     - the error is not the named refusal, or is `commit_status_unknown`;
     - the handler retries after refusing;
     - any manifest references the refused fragments;
     - replay is not `Ok`;
     - latency rises by Δp50 > max(5 %, 2 × noise) over 5 × 1,000 commits, release profile, interleaved.
   - Silence twin: 1,000 honest batches from version 0, with handler entries == 1000 exactly.
   - Three disables, each expected red: drop the compare; inherit `definitive = false`; return `CommitConflict`.
3. **Y1, Y5, Z1, W1c** follow the cards. The pre-registered expectations are:
   - Y1 is blind to R2.
   - Y5 fires: no accepting rule under V-L == V-T.
   - W1c: a plain Append commits renumbered, and X1 does not.

## Open

- Which identity A3 freezes for this log: "Lance version == table version", or
  only "append order is version order". This decides Y5. **OPEN.**
- Whether "never compacted" extends to "never cleaned" for log datasets. Y1
  depends on it, and so does any per-version discriminator. **OPEN.**
- Where an index on `row` would live, since a CreateIndex version on the log
  dataset wedges it. **OPEN.**
- Object-store variants of X1 (S3 conditional put, `s3+ddb`) are not probed here.

## Not searched

- PostgreSQL advisory locks; DynamoDB conditional writes as a standalone item;
  the Hudi timeline instant spec; the Zab paper; Kafka SIGMOD 2021.
- arXiv 2608.00501 (machine-checked dual-write recovery) is a LEAD only:
  SECTION-READ, theorems not checked.
- The lance-namespace and DynamoDB external-manifest stores beyond the
  `put_if_predecessor` closed search.
