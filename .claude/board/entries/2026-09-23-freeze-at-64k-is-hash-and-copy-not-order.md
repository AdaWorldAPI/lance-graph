# 2026-09-23 — freeze at 64k is hash and copy, not order

**Status:** MEASURED (release, 4-core Xeon @ 2.8 GHz, median of 7) · VERIFIED-IN-CODE (contended-row fraction = 0 on every in-tree path) · OPEN (off-loop freeze design)
**D-ids:** none new — measurement input for the hot-window design (`.claude/plans/measure-64k-axes-v4.md`) and D-LNC-5b's seal discussion.

## Measured — `persist_sink::tests::freeze_step_timing_at_64k` (`#[ignore]`, run with `--release -- --ignored`)

65,536 casts, one owner per row (the 1:1 main model), 512-byte payloads (witness ABI). Each step
timed alone on inputs built outside the timed region; the timed hash is asserted equal to
`freeze`'s own `batch_hash`.

| arrival | sort (`order_cycle_stably`) | fold (`row → payload.clone()`) | hash (`content_hash`, FNV-1a) | whole `freeze` |
|---|---|---|---|---|
| bit-reversed (O-arm's scrambled arrival) | 6.3 ms | 32.5 ms | **61.0 ms** | **102.7 ms** |
| in-order (what a slot-indexed stack gives) | 0.4 ms | 12.6 ms | **43.6 ms** | 58.0 ms |

- The hash is ~60 % of freeze: a byte-at-a-time FNV over all 32 MiB of payload. It is the
  commit's idempotency key only — no thought reads it.
- The fold is ~30 %: 32 MiB of `payload.clone()`. The declared-but-unbuilt descriptor
  (`batch_writer.rs`: `(mailbox, dirty row-range, cycle)`) removes it.
- The sort is negligible; arrival order also drives fold/hash locality (scrambled ≈ 1.8×).
- The fixture has **zero contended rows** (1:1): all 58–103 ms is spent on rows no other owner
  touched. Only contended rows need the seal's cross-owner order.

## Consequence (WORKING-MODEL, not built)

What the seal *decides* — order, fold result, cohort — costs a few ms. Hash and copy can leave
the thought loop: hash to the single background writer (unchanged algorithm, so the durable
`batch_hash` identity and reconciliation stay byte-identical), copy replaced by row-range
descriptors. Replacing FNV with a faster hash is a separate decision: it changes the durable
idempotency key existing stores reconcile against.

## Scope

Synthetic payloads, one host, in-process only (no Lance write). v3's recorded seal of
11.6–20 ms used a different payload shape; the two are not directly comparable.

## Contended-row fraction — 0 by construction (VERIFIED-IN-CODE)

A slot's row is `row_of(owner)` (`cycle_driver.rs:387`, `collect_casts`), a function of the
owner alone. Every in-tree caller passes `u64::from` (production `run_cycle` callers, the
supervisor tests, `measure_wal_curve`); `blw_tenant` casts many rows but from ONE owner. So two
distinct owners can never target one row: **cross-owner contention is 0 on every path, not
merely on this fixture.** What remains is same-owner repeats within a cycle, and their order is
the owner's own cast order — per-owner information `temporal.rs` already carries (`cast_seq`).

The injectivity is the ownership model itself, not an accident of the callers. DECISION
(2026-09-23), BASIS: an SoA row has exactly one owner, ever, and a `NodeGuid` IS its owner —
the one-writer-per-mailbox rule (`SoaEnvelope::mailbox_owner`, E-CE64-MB-4). A row shared by
several owners is not a design option to keep open.

Consequence (WORKING-MODEL): the seal's cross-owner sort has nothing to decide, and freeze
reduces to hash + copy, both movable off the loop.
