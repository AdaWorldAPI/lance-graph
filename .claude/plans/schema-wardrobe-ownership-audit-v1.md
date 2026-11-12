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
| bootstrap rebinding | `owner_adapter::emit_bootstrap_intent`, called by `cognitive_pass` in `cycle_driver.rs` with the id of an owner resolved from the fleet | SHIPPED; `cognitive_pass` is reached only through `run_cognitive_work*`, whose callers are supervisor tests; three planner examples call the adapter directly |
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

## 1a. The resident write itself (added after a second read of `mailbox_soa.rs`, `soa_envelope.rs`, `kanban_actor.rs`)

The first pass of this audit followed the cast and missed the write. These
were read afterwards and change what "the lawful owner mutation path" means.

| fact | where | state |
|---|---|---|
| the one cycle-aware row mutator: `MailboxSoA::write_row(row, cycle, &WriteCell) -> WriteOutcome`; a write for another cycle is `Stale` or `Future` and mutates nothing | `mailbox_soa.rs` | SHIPPED |
| `WriteCell` borrows its planes (`Option<&[u64]>`); the bytes are copied once, into the owner's column | `mailbox_soa.rs` | SHIPPED |
| callers of `write_row` outside tests: `BackingStoreWrite::write_row` (`backing.rs`) and the codebook-index ingest in `engine_bridge.rs` | `cognitive-shader-driver/src` | SHIPPED |
| the mailbox arm of `backing.rs` is behind feature `mailbox-thoughtspace`, which is **off by default**: the default build writes the singleton `BindSpace`, whose write is cycle-blind by its own doc comment | `backing.rs`, `Cargo.toml` | mailbox arm NOT the default |
| neither caller casts on behalf after writing: a resident write is not reported to a `BatchWriter` | `backing.rs`, `engine_bridge.rs` | join MISSING |
| public cycle-blind mutators beside `write_row`: `apply_edges`, `set_edge`, `set_meta`, `set_content`, `set_topic`, `set_angle`, `set_qualia`, `set_entity_type`, `set_temporal`, `set_expert`, `set_sigma`, `reset_row`, the style-lane setters; all need `&mut MailboxSoA` | `mailbox_soa.rs` | SHIPPED; they bypass the cycle gate, not the owner |
| the documented owner stamp `SoaEnvelope::mailbox_owner()` defaults to `0` and has no caller outside its own tests; its one production implementor `NodeRowPacket` does not override it; `MailboxSoA` does not implement `SoaEnvelope` | `lance-graph-contract/src/soa_envelope.rs`, `canonical_node.rs` | UNWIRED |
| `kanban_actor.rs` holds a read-only `PhaseCensus`, `mul_target` and a step-string parser; the actor and its RPC drivers are deleted | `lance-graph-supervisor/src/kanban_actor.rs` | read-only, no write path |

Consequences:

- A foreign-schema ingestion into a resident mailbox has exactly one lawful
  entry: `write_row` under `&mut MailboxSoA` at the owner's current cycle. It
  exists and is tested. What does not exist is the step after it (the cast
  that reports the write).
- "`MailboxSoA` is the canonical resident owner" is true of the type and not
  yet of the default build, which still routes writes to the singleton.
- There are two owner stamps: `MailboxSoaView::mailbox_id()` (used by
  `cast_on_behalf`) and `SoaEnvelope::mailbox_owner()` (documented as the
  stamp, used by nothing).

## 2. Delegation census (cast sites outside tests and examples)

| site | owner evidence |
|---|---|
| `MailboxSoA::cast_on_behalf` (`mailbox_soa.rs`) | the live owner: `&self` on the owned SoA supplies the id |
| `owner_adapter::emit_bootstrap_intent` | caller-supplied `owner: MailboxId`; rebinding is refused unless the move is the bootstrap sentinel. Its one library caller, `cognitive_pass`, passes `owner.mailbox_id()` of an owner it resolved from the fleet |
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

- `.claude/v3/knowledge/write-on-behalf.md` calls `cycle_driver.rs`'s
  `cognitive_pass` "the existing production caller" of
  `emit_bootstrap_intent`. The function exists and does call it, but
  `cognitive_pass` is private and its public drivers (`run_cognitive_work*`)
  are called only from supervisor tests. "Library code with no production
  driver" is the accurate wording.
- The same file's rule text says "the ractor owner mailbox"; its own stale
  note records that the actor was deleted 2026-08-05.
- `.claude/v3/knowledge/d-mbx-a6-owner-consume-and-persistence.md` describes
  the sink as gated on a Lance 7 MemWAL surface and "zero production callers"
  of `BatchWriter`. The sink that shipped is `LanceCycleWriter` (one append per
  cycle); the "zero production callers" half is still true.
- `.claude/v3/knowledge/mailbox-kanban-model.md` step 5 says Lance writes "from
  the in-place backing store (zero-copy)". The wired path copies twice before
  the append (§4).
- `write-on-behalf.md` and `mailbox-kanban-model.md` give the sanctioned
  shape as `cast(on_behalf = envelope.mailbox_owner(), …)`. No code has that
  shape; `mailbox_owner()` is unwired (§1a).
- The module doc of `mailbox_soa.rs` says Lance writes LE bytes "from this
  in-place store (`SoaEnvelope` describes the geometry)". `MailboxSoA` does not
  implement `SoaEnvelope`, and the wired seal path copies (§4).
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

**Results.** 9 of 9 runtime tests and 7 of 7 doctests pass (3 laws, each a
compiling control beside its compile-fail twin, so a typo cannot pass as a
refusal). Disable runs against production code, each patch asserted to apply
once: the owner stamp replaced by 0 (3 tests red), the cross-owner check in
`persist_cycle` removed (1 red), a failed seal returning no casts (1 red), a
full queue dropping its casts (1 red): 4 of 4 red.

**Limits of the suite.** The fence in
`a_second_writer_on_a_stale_head_is_fenced` and the failure injection are in
the in-file sink; they check that `seal_cycle` carries a sink's refusal
through unchanged, not that `LanceCycleWriter` fences (its own tests in
`cycle_sink.rs` do). `a_cast_is_not_a_commit` cannot fail through the sink,
because a cast holds no sink; it pins that the owner's phase does not move.
`MailboxSoA`'s columns and `mailbox_id` are public fields: any holder of
`&mut MailboxSoA` can write them directly, so the lifecycle trait is not the
only mutation path for the owner itself.

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
population; write a mailbox row through anything but `write_row` at the owner's current cycle; treat a cast or a returned ticket as durability; apply a paired
move before its ticket resolves `Ok`; or describe the path as zero-copy.

## 10. Smallest follow-up (proposed, not built)

1. Decide the enforcement seam for `on_behalf` (§2).
1a. Decide which owner stamp is the contract (`mailbox_id` or `mailbox_owner`) and retire or wire the other (§1a).
1b. Decide whether a resident `write_row` must be followed by a cast, and who makes it (§1a).
2. Decide whether `collect_casts` should carry descriptors (§4) before any
   resident ingestion is sized against it.
3. Correct the four stale documents (§5) in a documentation PR.
4. Name the owner for a resident codegraph population (§8).

R4, R7 and the reader `Scope` issuer are untouched.
