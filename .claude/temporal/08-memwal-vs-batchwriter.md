# 08 — MemWAL vs BatchWriter: **VERDICT FOLD**

Keep the descriptor/cast layer; put durability on MemWAL. The seam is
`WalSink` itself (`persist_sink.rs:534-578`).

## ⊘ CORRECTION TO THIS SESSION'S EARLIER REPORTING

**The "P is a DESCRIPTOR — (mailbox, dirty row-range, cycle) — never owned
delta bytes" rule is DOC-ONLY.** I cited it as a real architectural constraint
and as the pattern alpha should conform to. Measured:

- `SweepSlot::payload: Vec<u8>` (`persist_sink.rs:190`)
- `BatchWriter<Vec<u8>>` (`cycle_driver.rs:358`)
- `as_le_bytes` is **never called on any write path**
- `cycle_sink.rs:70-78` documents a copy boundary

So the advertised zero-copy-descriptor model is prose, and the incompatibility
I described between it and MemWAL's row staging **does not exist**.

**Second correction.** I said "there IS a real production implementation,
`impl WalSink for LanceCycleWriter`." True as CODE. But **`LanceCycleWriter::open`
has ZERO callers repo-wide**, so it is never instantiated. The durable path
exists and is not entered.

## What each thing actually is

| | BatchWriter | LanceCycleWriter | MemWAL (lance 11) |
|---|---|---|---|
| size | 233 lines | 2,208 lines | ~13.5K-line LSM subsystem |
| I/O | **none** (`BTreeMap` + `Vec`, `batch_writer.rs:95-181`) | real Lance MVCC | LSM staging under `_mem_wal/` |
| own doc says | *"ephemeral staging, not a durable WAL"* (`:22-25`) | — | pre-version staging |

**MemWAL's real counterpart is `LanceCycleWriter`, not `BatchWriter`.** That
reframing is what makes the verdict FOLD rather than REPLACE.

## Wiring, measured

- `deinterlace` — **CONFIRMED unwired.** All seven callers sit below
  `temporal.rs:471`'s `#[cfg(test)]`; the only others are examples.
- `cast()` "zero production call sites" — **REFUTED literally** (three callers:
  `owner_adapter.rs:100`, `mailbox_soa.rs:764`, `cycle_driver.rs:407`) but
  **CONFIRMED in substance**: every terminal caller of `run_cycle` is a test,
  and `LanceCycleWriter::open` has zero callers repo-wide.
- Ledger: `TECH_DEBT.md:973-990`, `TD-DOC-COMMENTS-CLAIM-UNWIRED-BEHAVIOUR`,
  still open.

## The three real incompatibilities — all ordering/horizon

1. **No MemWAL analog of `stream_position`.** Fixable: sort before `put`, which
   `order_cycle_stably` already does.
2. **The sealed read horizon** (`persist_sink.rs:36-43`) is the INVERSE of
   MemWAL's memtable scanner. Every thought in an open cycle reads only the
   sealed predecessor; MemWAL's scanner is built to expose unflushed state.
   **This is the property that would be silently LOST**, and silence is the
   danger: nothing would fail, thoughts would just start seeing in-flight
   siblings as though they were history.
3. **Generation-local row ids** (already recorded in `01-delta-api-lance11.md`).

## What must be KEPT — MemWAL knows only "batches with a PK"

`on_behalf_of` pairing · `intent_moves` · the ≤1-move-per-owner invariant
(`cycle_driver.rs:373-382`) · the artifact gate · the delegation cache.

## What a fold RETIRES, and what it gains free

**Retires:** the hand-rolled `(cycle, batch_hash)` reconciliation and fencing
(`cycle_sink.rs:44-58`). MemWAL's `writer_epoch` fencing (`wal.rs:41-58`) gives
the same guarantee **with replay**, which the in-house version explicitly
lacks.

**Gains free:** backpressure (`StoreFull`, `batch_store.rs:150`). The current
staging layer has literally none — and the persistence plan's own §"Capacity +
backpressure" names that as a required concrete-sink property that was never
built.

## The ONE measurement that would overturn FOLD

Can `ShardWriter::put` seal N landing rows plus the frame row **atomically**?

Kill mid-flush on a 5,000-row cycle. Observing 0 or 5,000 ⇒ FOLD stands.
Anything in between ⇒ KEEP, because the single-atomic-commit-per-cycle
guarantee is the whole point of `commit_cycle` and MemWAL would not preserve
it.

## Incidental defect found

`BatchWriter`'s `board` is **never cleared by any method** — only
`pending_payloads` drains (`batch_writer.rs:179-181`). Unbounded growth. Not
load-bearing today because nothing production-side drives `run_cycle`, which is
exactly why it has gone unnoticed.
