# Schema wardrobe P0: who owns a resident write (audit)

**D-id:** D-SCHEMA-WARDROBE-P0. **Status:** AUDIT + falsifier suite, in PR.
No ingestion subsystem, writer, actor, SoA owner, schema registry or
persistence API is added. **Linear (existing):** EXO-253.

**Question.** Before a foreign schema is admitted, which existing type owns a
resident write, and is the chain `MailboxSoA → cast → BatchWriter → one
background writer → Lance version → paired step` actually wired?

**Method.** The ten sources named in the brief were read; call sites were
counted over every `*.rs` under `crates/` (census output in
`.claude/probes/2026-11-12-schema-wardrobe-p0/`), and every site cited below
was read at its line.

## 1. The chain, edge by edge

| edge | symbol (file) | state |
|---|---|---|
| resident owner | `MailboxSoA<N>` in `crates/cognitive-shader-driver/src/mailbox_soa.rs`; implements `MailboxSoaView` and `MailboxSoaOwner` | SHIPPED |
| read-only surface | `MailboxSoaView` in `crates/lance-graph-contract/src/soa_view.rs` (19 methods, all `&self`) | SHIPPED |
| lifecycle mutation | `MailboxSoaOwner::{advance_phase, try_advance_phase}` (`&mut self`); only two production-type implementors besides fakes: `MailboxSoA`, `SymbiontBoard` | SHIPPED |
| owner-stamped cast | `MailboxSoA::cast_on_behalf` (feature `with-planner`) → `BatchWriter::cast(self.mailbox_id(), …)` | SHIPPED, **no caller outside tests** |
| intent recorder | `BatchWriter<P>` in `crates/lance-graph-planner/src/batch_writer.rs` | SHIPPED; constructed only in tests, examples and `#[cfg(test)]` modules |
| bootstrap rebinding | `owner_adapter::emit_bootstrap_intent` | SHIPPED; callers are three planner examples only |
| drain → slots | `collect_casts` in `crates/lance-graph-supervisor/src/cycle_driver.rs` (feature `cycle-driver`) | SHIPPED; callers are tests, examples, probes |
| one durable write | `seal_cycle` → `persist_cycle` → `WalSink::commit_cycle` | SHIPPED |
| single background writer | `cycle_writer` / `CycleWriterHandle::try_submit` in `crates/lance-graph-supervisor/src/cycle_writer.rs` | SHIPPED; **one non-test submitter**: `MapLog` in `crates/lance-graph-dt64-lance/src/lib.rs` |
| Lance sink | `LanceCycleWriter` (`impl WalSink`) in `crates/lance-graph/src/graph/cycle_sink.rs` | SHIPPED |
| paired step after the seal | `apply_sealed_transitions` over a `MailboxFleet` | SHIPPED; fleet implementors: `HashMap<MailboxId, O>` and one test fleet |
| a production loop that joins all of the above for a `MailboxSoA` | none found | **MISSING** |

The chain exists as parts and is exercised end to end only in tests and
examples. The one production-shaped user of the background writer is the DR
map log, and it does not go through `MailboxSoA` or `BatchWriter`: it builds
`SweepSlot`s itself with `owner: self.on_behalf`, a mailbox id it was handed.

## 2. Delegation census (cast sites outside tests and examples)

| site | owner evidence |
|---|---|
| `MailboxSoA::cast_on_behalf` (`mailbox_soa.rs`) | the live owner: `&self` on the owned SoA supplies the id |
| `owner_adapter::emit_bootstrap_intent` | caller-supplied `owner: MailboxId`; rebinding is refused unless the move is the bootstrap sentinel |
| `restage_held` (`cycle_driver.rs`) | re-stages an id recorded by an earlier cast; no new evidence |
| `MapLog` slot construction (`lance-graph-dt64-lance/src/lib.rs`) | caller-supplied `on_behalf: MailboxId`; bypasses `BatchWriter` |

`BatchWriter::cast(on_behalf: MailboxId, …)` is public and takes a plain
integer. **Nothing proves the caller owns that mailbox.** What is checked
downstream: a paired move naming another mailbox than its landing is refused
(`PersistError::OwnerMismatch` in `persist_cycle`), and
`apply_sealed_transitions` re-checks it. What is not checked: a payload-only
cast under a foreign id. The suite pins this as a finding
(`finding_public_cast_accepts_any_mailbox_id`).

**Missing enforcement seam (reported, not built):** a cast that can only be
made through the owner, for example by making the id-taking `cast` crate-
private to the planner and leaving `cast_on_behalf`-shaped entry points as the
public surface. That is a contract change in `lance-graph-planner` and needs
its own decision; no owner registry is proposed.

## 3. Ownership bypasses found

1. **Forgeable `on_behalf`** (§2). Gap in enforcement; no production caller
   exploits it.
2. **`MapLog` writes without a `MailboxSoA`.** It owns a DR `MapTable`, not a
   mailbox SoA, and stamps a handed-in id. It is single-writer by construction
   and restore refuses foreign landings, so it is consistent with the
   one-writer rule, but it is a second way to reach the sink that does not
   pass the owner-stamped cast.
3. No type other than the owner hands out mutable SoA columns: the view trait
   has no `&mut` method and no implementor of `MailboxSoaOwner` exists for a
   read adapter (`SurrealMailboxView` implements the view only).

## 4. Copy ledger

| stage | what happens | copies of payload bytes |
|---|---|---|
| borrowed SoA read through `MailboxSoaView` | slices of the owner's columns | 0 (pointer equality pinned by the suite) |
| cast | payload moved into `BatchWriter` | 0, but with `P = Vec<u8>` the payload is owned bytes the caller built |
| drain (`drain_pending_payloads`, `collect_casts`) | payload moved into `SweepSlot` | 0 |
| submit (`try_submit`) | the `Vec<SweepSlot>` is moved | 0 |
| seal: retry copy (`seal_cycle`, `casts.clone()`) | held until commit success | 1 |
| seal: freeze (`DetachedCycleBatch::freeze`, `payload.clone()` per row) | coalesced image | 1 |
| Lance append in `LanceCycleWriter` | Arrow arrays built from the image | at least 1 (not measured here) |

The documented descriptor route (`BatchWriter<RowSpanDescriptor>`: the batch
names a span, the sink reads the SoA) exists only in three planner examples
and three supervisor tests. `collect_casts` is typed
`BatchWriter<Vec<u8>>`, so **the wired seal path carries owned payload bytes,
not descriptors**. "End-to-end zero-copy" is not true of this chain today.

## 5. Stale documentation

- `.claude/v3/knowledge/write-on-behalf.md` status line and its 2026-08-01
  note name `cycle_driver.rs`'s `cognitive_pass` as the production caller of
  `emit_bootstrap_intent`. No `cognitive_pass` function exists in
  `cycle_driver.rs`, and `emit_bootstrap_intent` has no caller outside
  examples.
- The same file's rule text says "the ractor owner mailbox"; its own stale
  note records that the actor was deleted 2026-08-05.
- `.claude/v3/knowledge/d-mbx-a6-owner-consume-and-persistence.md` describes
  the sink as gated on a Lance 7 MemWAL surface and "zero production callers"
  of `BatchWriter`. The sink that shipped is `LanceCycleWriter` (one append per
  cycle); the "zero production callers" half is still true.
- `.claude/v3/knowledge/mailbox-kanban-model.md` step 5 says Lance writes "from
  the in-place backing store (zero-copy)". The wired path copies twice before
  the append (§4).
- `soa_mailbox` / `SoAMailbox`: no such symbol. The module is
  `mailbox_soa.rs`, the type `MailboxSoA`.

These are flagged, not edited, in this PR.

## 6. Falsifier suite

`crates/lance-graph-supervisor/tests/ownership_preservation.rs` (real
`MailboxSoA`, real `BatchWriter`, `collect_casts`, `seal_cycle`,
`apply_sealed_transitions`, `cycle_writer`; an in-file `WalSink`) and
compile-fail doctests in `crates/cognitive-shader-driver/src/ownership_laws.rs`.

| control | where | kind |
|---|---|---|
| owner stamps its own cast | `the_owner_stamps_its_own_cast` | runtime |
| foreign code supplies another mailbox id | `finding_public_cast_accepts_any_mailbox_id` | runtime, pins the GAP |
| forged cross-owner move | `a_forged_cross_owner_move_is_refused_before_it_changes_a_mailbox` | runtime |
| a cast is not a commit | `a_cast_is_not_a_commit` | runtime |
| failed seal advances nothing, casts returned | `a_failed_seal_advances_nothing_and_returns_the_casts` | runtime |
| full queue drops nothing | `a_full_queue_hands_the_casts_back` | runtime |
| second writer on a stale head | `a_second_writer_on_a_stale_head_is_fenced` | runtime |
| a view is the owner's columns, no second population | `a_view_is_the_owners_own_columns` | runtime |
| two mailboxes share nothing | `two_mailboxes_do_not_share_state` | runtime |
| two mutable borrows of one mailbox | `ownership_laws` doctest | compile-fail |
| a view promoted to an owner | `ownership_laws` doctest | compile-fail |
| a cast after the mailbox was moved away | `ownership_laws` doctest | compile-fail |

Already covered by shipped tests, cited instead of repeated
(`crates/lance-graph-dt64/tests/bind_store.rs`, `tests/dr.rs`,
`crates/lance-graph-report/tests/bound_sel.rs`): unknown class is not a
default (`an_unregistered_class_is_unknown_not_empty`), unknown behaviour is
not empty (`an_unknown_behaviour_lens_fails_closed_instead_of_inheriting_its_concept`),
revision change with reused ordinals
(`schema_revision_and_field_basis_are_checked`,
`a_same_count_reorder_across_revisions_replays_only_through_history`), equal
positions in different frames (`two_stores_at_equal_revisions_do_not_alias`,
`positions_of_one_frame_are_never_read_against_another_sources_lanes`), source
change without witness (`every_frame_revision_is_a_separate_refusal`).

**Not testable today:** "a consumer writes directly to Lance instead of
delegating". `LanceCycleWriter::open` is public and Rust cannot forbid a crate
from calling it; only review and the one-writer fence
(`a_second_writer_on_a_stale_head_is_fenced`, and the sink's own tests) apply.

## 7. Wardrobe admission

No new fixture was added. The cold class-admission path a wardrobe needs
already ships and is tested: `BindStore::register` over a `ClassView` basis
with a schema revision and an explicit behaviour declaration
(`bind_store.rs`), and the consumer-class binding probe
(`crates/dt64-hotplug-probe`). A wardrobe would call those; it holds no SoA,
no writer and no Lance handle.

## 8. Resident codegraph: BLOCKED

No owner is established for a resident codegraph population. The codegraph
crate publishes through its own lab `WalSink` (`lab_wal.rs`,
`graph/publication.rs`), not through a `MailboxSoA`. Moving it needs a
decision on which mailbox owns the population and how its rows map onto
`MailboxSoA` columns. Its current ingestion and persistence path is unchanged.

## 9. Production STOP conditions

Stop any foreign-schema ingestion work that would: add an implementor of
`MailboxSoaOwner` for an adapter; call `BatchWriter::cast` with an id not
obtained from a live owner; construct a second `CycleWriter` or sink for the
same dataset; build `SweepSlot`s outside `collect_casts` for a mailbox-owned
population; treat a cast or a returned ticket as durability; apply a paired
move before its ticket resolves `Ok`; or describe the path as zero-copy.

## 10. Smallest follow-up (proposed, not built)

1. Decide the enforcement seam for `on_behalf` (§2).
2. Decide whether `collect_casts` should carry descriptors (§4) before any
   resident ingestion is sized against it.
3. Correct the four stale documents (§5) in a documentation PR.
4. Name the owner for a resident codegraph population (§8).

R4, R7 and the reader `Scope` issuer are untouched.
