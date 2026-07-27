# Trace A — Write Path (BeliefArena → V3 substrate migration prep)

Scope: `soa_envelope.rs`, `collapse_gate.rs` (contract), `kanban.rs`,
`mailbox_soa.rs`, `kanban_actor.rs`, `kanban_loop.rs` — all read in full
(whole-file `Read`, no offset/limit truncation except `mailbox_soa.rs` which
required two `Read` calls, offset 1 and offset 1152, both full-content, per
primer §1 "two/three Reads covering the entire region" allowance). Additional
files opened for a *named, load-bearing hop* only (not primary comprehension
targets, per task scope): `lance-graph-contract/src/soa_view.rs` lines
280–349 (the `MailboxSoaOwner`/`MailboxSoaView` trait definitions that every
target file calls but none of them defines), and `lance-graph-contract/src/mul.rs`
lines 55–160 (to resolve a `GateDecision` name collision — see Violations).
Everything below is file:line-grounded; anything not locatable is reported as
MISSING or UNDETERMINED, never designed.

Batch writer file, corrected: the task listed `batch_writer.rs` under
`lance-graph-contract/src/`; it actually lives at
`crates/lance-graph-planner/src/batch_writer.rs` (confirmed via Glob — no
`batch_writer*` exists under `lance-graph-contract`). Read at the corrected path.

## Ownership chain

- **`MailboxSoA<const N: usize>`** (`crates/cognitive-shader-driver/src/mailbox_soa.rs:58-230`)
  is the owning SoA. It holds `mailbox_id: MailboxId` (:61), the value-tenant
  columns (`energy`, `plasticity_counter`, `edges`, `qualia`, `meta`,
  `entity_type`, `temporal`, `expert`, `sigma`, `content`/`topic`/`angle`,
  the three style lanes) and `pub(crate) phase: KanbanColumn` (:229).
- **Owner-only mutation surface:** `phase` is `pub(crate)`, not `pub`
  (:225-229, doc: *"mutated only via `MailboxSoaOwner::advance_phase` /
  `try_advance_phase`"*). `MailboxSoA<N>` implements `MailboxSoaOwner`
  (:949-974) with exactly one method, `advance_phase` (:953-973); the
  checked wrapper `try_advance_phase` is a **default trait method**, not
  overridden here — defined once at `soa_view.rs:309-319`:
  ```
  fn try_advance_phase(&mut self, to) -> Result<KanbanMove, RubiconTransitionError> {
      let from = self.phase();
      if from.can_transition_to(to) { Ok(self.advance_phase(to)) }
      else { Err(RubiconTransitionError { from, to }) }
  }
  ```
  So EVERY owner in this trace (`MailboxSoA`, `kanban_actor.rs`'s `TestBoard`,
  `kanban_loop.rs`'s `SymbiontBoard`) gets DAG-checking for free from one
  place; only the unchecked primitive `advance_phase` is per-type.
- **Write-on-behalf entry points:**
  - `MailboxSoA::write_row` (:417-463) — the "ONE deinterlacing mutator",
    cycle-gated (`WriteOutcome::{Accepted,Stale,Future}`), applies a
    `WriteCell<'_>` field-by-field via existing `set_*` setters.
  - `MailboxSoA::cast_on_behalf` (:757-764, feature `with-planner`) — calls
    `writer.cast(self.mailbox_id(), moves, payload)` where `writer: &mut
    BatchWriter<P>`. The owner (`&self`) reads its OWN `mailbox_id()`, so a
    call site cannot pass a different owner than the SoA it is casting for
    (comment :730-744: "a mispair unrepresentable" — enforced by the method
    signature, not a runtime check).
  - `set_style_lane` / `set_style_atom` / `promote_family` (:786-839) — owner
    `&mut self` writes to the three P4 style lanes; guarded by `row >=
    self.populated` / `family >= 12` no-ops, never by a Kanban check (these
    are NOT phase transitions).
- **Read-only side:** `MailboxSoaView` (`soa_view.rs:66` onward, read via
  `mailbox_soa.rs:852-947` impl) has NO mutating methods; `phase()` (:876-878)
  is a plain getter. `soa_view.rs:288-292` states the structural guarantee
  explicitly: *"A read-only view (e.g. `surreal_container`) deliberately does
  NOT implement `MailboxSoaOwner` — that is what makes 'the view is read-only'
  a structural guarantee rather than a convention."*
- **`SoaEnvelope` is a DIFFERENT trait than `MailboxSoaView`/`MailboxSoaOwner`,
  and `MailboxSoA` does NOT implement it.** Grep across the whole repo
  (`impl.*SoaEnvelope for`) finds exactly three implementors:
  `TestEnvelope` (`soa_envelope.rs:303`, test-only), `Owned`
  (`soa_envelope.rs:357`, test-only), and `NodeRowPacket<'a>`
  (`canonical_node.rs:1479`, the ONLY non-test implementor). `mailbox_owner()`
  (`soa_envelope.rs:189-197`, defaulted to `0`) and `verify_layout()`
  (`soa_envelope.rs:222-288`, the "Lance read boundary" gate) exist on
  `SoaEnvelope`, not on `MailboxSoaOwner`/`MailboxSoaView`. **`MailboxSoA<N>`
  never calls or implements `verify_layout`/`as_le_bytes`/`mailbox_owner()`.**
  This is a genuine split in the codebase between two ownership-flavored
  contracts that the primer (§6, §11) discusses as if they were one seam;
  see Violations/Conflicts below.

## Kanban governance

- **`KanbanColumn`** (`kanban.rs:33-50`): `Planning(0) → CognitiveWork(1) →
  Evaluation(2) → {Commit(3), Plan(4), Prune(5)}`, `Plan → Planning`. DAG
  encoded in `next_phases()` (:91-99); `can_transition_to` (:103-105) is the
  single legality check every mutation path funnels through (via
  `try_advance_phase`, above).
- **Where a `KanbanMove` is constructed** (the transition record, not the
  mutation itself):
  - `MailboxSoA::advance_phase` (:953-973) — stamps `witness_chain_position =
    self.current_cycle` (:967), `exec: ExecTarget::Native` always (:971,
    "No Libet stamp: the window is DERIVED").
  - `kanban_actor.rs` `TestBoard::advance_phase` (:429-442, test-only) —
    same shape, also bumps its own `cycle`.
  - `kanban_loop.rs` `SymbiontBoard::advance_phase` (:180-190) — same shape.
  - `lance_graph_contract::scheduler::NextPhaseScheduler::on_version` (NOT
    read in full — outside this trace's file list, referenced at
    `kanban_actor.rs:29,296,696` and `kanban_loop.rs:50,114,198`) is the
    **proposer**: "propose, don't dispose" — it reads a `MailboxSoaView` and
    a `DatasetVersion`, returns `Option<KanbanMove>` with `to` set, never
    mutates. The owner disposes via `try_advance_phase` (`kanban_actor.rs:117,
    134; kanban_loop.rs:120`) or the checked `Advance` actor message
    (`kanban_actor.rs:114-119`).
- **Legal transitions are applied through exactly one narrow surface**: the
  `MailboxSoaOwner` trait (`&mut self`). In `kanban_actor.rs` this is wrapped
  in a `ractor::Actor::handle` (:107-153) so the actor's serialized mailbox
  gives single-writer-at-a-time; in `kanban_loop.rs` there is no actor at all
  — `SymbiontBoard::step` (:114-121) calls `try_advance_phase` directly,
  documented as intentional (module doc :19-27: *"there is no ractor message
  actor here and no tokio... `step()` drives the loop by direct owned
  mutation, never by sending a message"*).
- **Where a new Lance version is produced: NOT FOUND in any of the six
  traced files.** None of `soa_envelope.rs`, `collapse_gate.rs`, `kanban.rs`,
  `mailbox_soa.rs`, `kanban_actor.rs`, `kanban_loop.rs` contains a call into
  `lance`/`lancedb` (no `Dataset`, no `commit`, no `write_stream`, no
  `.lance` path literal). The KanbanColumn::Commit variant (:43-44, doc:
  *"Terminal — calcify: commit to Lance SPO-G + AriGraph pointer"*) NAMES the
  intended effect in a doc comment; **no code in these six files performs
  it.** See "the belief-write seam" below — this is the FIRST MISSING HOP.

## `cast()` production call sites

Grep for `BatchWriter` construction/use across the entire `lance-graph` repo
(not just the six target files):

```
lance-graph-planner/tests/w1_probes.rs           — 4 sites, all in a #[test]-only file (tests/)
lance-graph-planner/src/batch_writer.rs           — struct def + its own #[cfg(test)] mod tests
cognitive-shader-driver/src/mailbox_soa.rs:759    — the `cast_on_behalf` method SIGNATURE
  (feature = "with-planner"; NOT itself test-gated)
cognitive-shader-driver/src/mailbox_soa.rs:1776+  — 3 sites, all inside
  `#[cfg(all(test, feature = "with-planner"))] mod w4a_cast_pairing_tests`
```

**Finding: `BatchWriter::cast` / `BatchWriter::new` have ZERO production
(non-test) call sites in this repo.** The only non-test code that even
mentions `BatchWriter` is the `cast_on_behalf` method signature
(`mailbox_soa.rs:757-764`), which takes `&mut BatchWriter<P>` as a parameter
— but nothing in the traced files, and nothing found by a repo-wide grep,
ever constructs a `BatchWriter`, holds one as a field, or drains one
(`drain_pending_payloads`, `batch_writer.rs:139`) outside its own test module.
There is no production owner of a live `BatchWriter` instance anywhere in
this codebase as it stands.

**Payload descriptor shape, confirmed from the only real (non-generic) test
payload used at the call site closest to production**
(`mailbox_soa.rs:1780-1786`, `w4a_cast_pairing_tests`):
```rust
struct DirtyRange { first_row: u32, rows: u32, cycle: u32 }
```
This matches the primer's DO-NOT-INVENT rule 17 verbatim
(`batch_writer.rs:18-20`: *"`P` is a DESCRIPTOR — (mailbox, dirty row-range,
cycle) — never owned delta bytes"*). **CONFIRMED, not refuted**: `P` is
payload-generic (`BatchWriter<P>`, `batch_writer.rs:55`) and the writer
"never inspects `P`" (`batch_writer.rs:1,4-5,65`); the one production-typed
payload wired at all (`BusDto`, gated `with-engine`,
`mailbox_soa.rs:1836-1855`) carries `codebook_index / energy / top_k /
cycle_count / converged` — cognitive-provenance scalars, no owned SoA bytes.
Deltas genuinely stay in the SoA backing store; nothing here ever copies a
row into the payload.

## The belief-write seam (sequence, with the first missing hop marked)

Purely descriptive — no design. If a belief row's truth byte were updated
TODAY through the traced substrate, the existing call chain is:

1. Something computes a new value and calls a `set_*` method directly on a
   `MailboxSoA` column (e.g. `set_style_atom`, `mailbox_soa.rs:800-810`, or
   the generic cycle-gated `write_row`, `mailbox_soa.rs:417-463) — **in
   place, `&mut self`, no allocation.** [Owner-only mutation, confirmed.]
2. If the write should also advance the Rubicon lifecycle: the owner calls
   its own `try_advance_phase(to)` (`soa_view.rs:309-319`, DAG-checked) or is
   driven to via `ractor` (`kanban_actor.rs:114-119`) or a direct `step()`
   (`kanban_loop.rs:114-121`). This emits a `KanbanMove`
   (`advance_phase`, one of the three impls above). [Confirmed, in-process,
   synchronous.]
3. **MISSING HOP.** The `KanbanMove` (and/or the write itself) needs to be
   turned into "a coherent LE in-place layout at cycle N" that Lance can
   version (`soa_envelope.rs:16-19`). This requires an `impl SoaEnvelope`.
   `MailboxSoA<N>` does not implement `SoaEnvelope` (confirmed above — zero
   hits). The only `SoaEnvelope` implementor that is not test scaffolding is
   `NodeRowPacket<'a>` (`canonical_node.rs:1479`), a *different* type built
   from `&[NodeRow]` (`symbiont/bridge.rs:129`: `NodeRowPacket::new(&rows, 0)`
   — itself only exercised in a demo function, `bridge.rs:120`
   `run_scale_demo`, and in `#[cfg(test)]` at `bridge.rs:143+`). **There is
   no code path from a `MailboxSoA` row to a `NodeRowPacket`, and no code
   path from either to an actual Lance `Dataset` write** (confirmed by
   repo-wide grep for `Dataset::write`/`dataset.write`/`lance::Dataset`/
   `commit(` under `lance-graph-planner` — zero hits, and none in the six
   traced files).
4. Even if step 3 existed, the ahead-firing `BatchWriter::cast` machinery
   that the module doc (`batch_writer.rs:1-31`) describes as the intended
   AHEAD-of-storage intent-recording layer has **no live instance anywhere
   in production code** (previous section) — so today there is nothing that
   would even receive the descriptor and trigger step 3's hypothetical flush.

**Named first missing hop:** *MailboxSoA (or any owner) → SoaEnvelope
implementation → Lance write/version.* Neither the trait impl nor the flush
sink exists for `MailboxSoA` in this repo. `NodeRowPacket` has the trait impl
but no wiring back to a live `MailboxSoA`/mailbox-owned row, and no wiring
forward to an actual Lance dataset write. Both halves of the chain are
missing; nothing here should be filled in with an invented carrier.

## Violations (against §11's discriminator, on these six files + the two
consulted hop files only)

- **None of the six target files themselves contains a materializing
  allocation on the reasoning/write path** (no `Vec<Belief>`, no CSR build,
  no serialize call). `write_row`, `set_*`, `advance_phase`,
  `try_advance_phase` all operate on already-resident `[T; N]` / `Box<[u64]>`
  arrays via direct indexing — consistent with §11's "borrowed
  interpretation of existing bytes."
- **`MailboxSoA::new`** (`mailbox_soa.rs:292-335`) allocates three
  `Box<[u64]>` heap planes (`content`/`topic`/`angle`, :322-324) and the
  fixed-size stack arrays — this is CONSTRUCTION of the owned backing store
  itself, not a materialized *copy* of already-resident state, so it is not
  a §11 violation by the discriminator's own test ("kernel-local... never an
  owned alternate representation of substrate state" — this IS the primary
  representation, not an alternate one).
  `NodeRowPacket::new(&rows, 0)` (`symbiont/bridge.rs:129`, built from a
  `Vec<NodeRow>` that is itself a heap `Vec`, not an SoA) is a closer call —
  it packs an existing `Vec<NodeRow>` into LE bytes, which is at minimum a
  **copy of already-resident row state into a second representation**
  (`NodeRowPacket` doc not read in this trace; flagged here for a future
  targeted read, not adjudicated).
- **Naming collision, not a substrate violation:** `kanban.rs:24` imports
  `crate::mul::GateDecision` (an enum: `Flow | Hold{reason:String} |
  Block{reason:String}`, `mul.rs:144-150`) and dispatches on it in
  `advance_on_gate` (`kanban.rs:136-143`). `collapse_gate.rs` ALSO defines a
  type named `GateDecision` (`collapse_gate.rs:58-100`, a `Copy` struct of
  `{gate: u8, merge: MergeMode}`) that is entirely unrelated and unused by
  `kanban.rs`. Two distinct types share one name across two modules in the
  same crate — a readability/discoverability hazard, not a data violation.
- **`BatchWriter`'s `board: BTreeMap<CastId,...>` and
  `pending_payloads: Vec<(CastId, P)>`** (`batch_writer.rs:58-66`) ARE
  heap-owned collections that accumulate state across casts. The module doc
  explicitly frames this as *"ephemeral staging, not a durable WAL"*
  (`batch_writer.rs:11-16`) and rules out a confirmation ledger by name. This
  is consistent with the operator's `E-ACK-ELIMINATED-1` ruling as
  documented, but note it is still a **heap-resident intermediate structure
  that exists purely because there is no live sink to drain it into
  (§ cast() production call sites, above)** — worth flagging as
  architecturally load-bearing but currently unexercised outside tests.

## UNDETERMINED items (with why)

- **Whether `NodeRowPacket` is meant to be the eventual `SoaEnvelope` for
  `MailboxSoA`, or a wholly separate lineage (the `canonical_node.rs` 512-byte
  key/edges/value row vs. the `MailboxSoA` per-column SoA)** — UNDETERMINED.
  `canonical_node.rs` itself was outside this trace's file list; only the
  `impl SoaEnvelope for NodeRowPacket<'a>` line and its one non-test call site
  were located via grep, not read in full. Answering this needs a full read
  of `canonical_node.rs` around `NodeRowPacket`.
  scheduler.on_version` — UNDETERMINED whether it ever mutates or only reads.
  It was referenced (not defined) in all three consumer files
  (`kanban_actor.rs`, `kanban_loop.rs`) as the "propose" half of "propose,
  don't dispose"; its full body lives in `lance_graph_contract::scheduler`,
  a file not on this trace's list. Everything reported above about it is
  taken from doc comments and call-site usage in the traced files, not from
  reading its body.
- **Whether any Lance write sink exists ANYWHERE in the workspace** (not just
  the six traced files or their immediate neighbors) — UNDETERMINED at
  workspace scope. Repo-wide greps were run only for a bounded set of
  literal strings (`Dataset::write`, `dataset.write`, `lance::Dataset`,
  `write_stream`, `commit(`) under `lance-graph-planner`; a wider sweep
  across `lance-graph` core, `lance-graph-catalog`, and any Lance-adjacent
  crate was not performed and could surface a sink this trace did not see.
