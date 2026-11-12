# Schema wardrobe P0: who owns a resident write (audit)

**D-id:** D-SCHEMA-WARDROBE-P0. **Status:** AUDIT + falsifier suite, in PR.
No ingestion subsystem, writer, actor, SoA owner, schema registry or
persistence API is added. **Linear (existing):** EXO-253.

**Question.** Before a foreign schema is admitted, which existing type owns a
resident write, and is the chain `MailboxSoA → cast → BatchWriter → one
background writer → Lance version → paired step` actually wired?

**Method and evidence record.** The first two versions of this audit were
written from excerpts, search hits and summaries of the sources. That was
wrong, and several statements were wrong because of it (corrections are listed
in §11). This version rests on these files, each read from first to last line
of its non-test code:

| file | lines read |
|---|---|
| `crates/cognitive-shader-driver/src/mailbox_soa.rs` | all, tests included |
| `crates/cognitive-shader-driver/src/backing.rs` | all, tests included |
| `crates/cognitive-shader-driver/src/engine_bridge.rs` | all, tests included |
| `crates/lance-graph-contract/src/soa_view.rs` | all non-test |
| `crates/lance-graph-contract/src/soa_envelope.rs` | all, tests included |
| `crates/lance-graph-planner/src/batch_writer.rs` | all non-test |
| `crates/lance-graph-planner/src/owner_adapter.rs` | all non-test |
| `crates/lance-graph-planner/src/persist_sink.rs` | all non-test |
| `crates/lance-graph-supervisor/src/cycle_driver.rs` | all non-test |
| `crates/lance-graph-supervisor/src/cycle_writer.rs` | all non-test |
| `crates/lance-graph-supervisor/src/kanban_actor.rs` | all |
| `crates/lance-graph/src/graph/cycle_sink.rs` | all non-test |
| `crates/lance-graph-dt64-lance/src/lib.rs` | all, tests included |
| `crates/symbiont/src/kanban_loop.rs` | the board, its `step` and both trait impls |
| `.claude/v3/knowledge/write-on-behalf.md`, `mailbox-kanban-model.md`, `d-mbx-a6-owner-consume-and-persistence.md` | all |

Not read in full, and named where a statement depends on them:
`canonical_node.rs` (`NodeRowPacket`), `crates/onebrc-probe`, the codegraph
crate, the test modules of the files marked "non-test". A statement of the form
"no caller outside tests" is a search over every `*.rs` under `crates/` for the
symbol's name, with each hit in `src/` then read; it is as strong as that
search and no stronger.

## 1. The chain, edge by edge

| edge | symbol (file) | state |
|---|---|---|
| resident owner | `MailboxSoA<N>` in `crates/cognitive-shader-driver/src/mailbox_soa.rs`; implements `MailboxSoaView` and `MailboxSoaOwner` | SHIPPED |
| read-only surface | `MailboxSoaView` in `crates/lance-graph-contract/src/soa_view.rs` (19 methods, all `&self`) | SHIPPED |
| lifecycle mutation | `MailboxSoaOwner::{advance_phase, try_advance_phase}` (`&mut self`). Implementors outside test modules: `MailboxSoA`; `SymbiontBoard` (`crates/symbiont`, excluded from the workspace); a search also lists `ProbeBoard` in `crates/onebrc-probe` (excluded, not read) | SHIPPED |
| owner-stamped cast | `MailboxSoA::cast_on_behalf` (feature `with-planner`) → `BatchWriter::cast(self.mailbox_id(), …)` | SHIPPED, **no caller outside tests** |
| intent recorder | `BatchWriter<P>` in `crates/lance-graph-planner/src/batch_writer.rs` | SHIPPED; constructed only in tests, examples and `#[cfg(test)]` modules |
| bootstrap rebinding | `owner_adapter::emit_bootstrap_intent`, called by `cognitive_pass` in `cycle_driver.rs` with the id of an owner resolved from the fleet | SHIPPED; `cognitive_pass` is reached only through `run_cognitive_work*`, whose callers are supervisor tests; three planner examples call the adapter directly |
| drain → slots | `collect_casts` in `crates/lance-graph-supervisor/src/cycle_driver.rs` (feature `cycle-driver`) | SHIPPED; callers are tests, examples, probes |
| one durable write | `seal_cycle` → `persist_cycle` → `WalSink::commit_cycle` | SHIPPED |
| single background writer | `cycle_writer` / `CycleWriterHandle::try_submit` in `crates/lance-graph-supervisor/src/cycle_writer.rs` | SHIPPED; **one non-test submitter**: `MapLog` in `crates/lance-graph-dt64-lance/src/lib.rs` |
| Lance sink | `LanceCycleWriter` (`impl WalSink`) in `crates/lance-graph/src/graph/cycle_sink.rs` | SHIPPED |
| paired step after the seal | `apply_sealed_transitions` over a `MailboxFleet` (normal path); `recover_fleet` → `persist_sink::recover_and_apply` (restart, committed history only) | SHIPPED; fleet implementors: `HashMap<MailboxId, O>` and one test fleet |
| a cycle with no artifact | `persist_cycle` returns `CommitOutcome::NoChange` without calling the sink; `seal_cycle` still returns the paired moves and `apply_sealed_transitions` applies them | SHIPPED by ruling (2026-08-09, `persist_sink.rs` module doc): **a step without a version** |
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
| `write_row` has two callers in `src/`, and neither has a caller of its own in `src/`: `BackingStoreWrite::write_row` (`pub(crate)`, `#[allow(dead_code)]`, by its doc "exercised only by the W2 differential harness", and the same doc says `driver.run()` is `&self` and writes no column) and `ingest_codebook_indices_soa` (`pub`, re-exported from `lib.rs`, called by one example and one in-file test) | `backing.rs`, `engine_bridge.rs` | LAB: no production write into a `MailboxSoA` row |
| the mailbox arm of `backing.rs` is behind feature `mailbox-thoughtspace`, **off by default**; the default arm writes the singleton `BindSpace`, cycle-blind by its own doc comment. `ingest_codebook_indices_soa` is not feature-gated | `backing.rs`, `engine_bridge.rs` | mailbox arm NOT the default |
| `engine_bridge.rs` still writes the singleton directly: `ingest_codebook_indices`, `dispatch_busdto`, `write_qualia_observed` and `engine_bridge::persist_cycle` (a different function from `persist_sink::persist_cycle`) take `&mut BindSpace` | `engine_bridge.rs` | SHIPPED; a second, ownerless write surface |
| neither `write_row` caller casts on behalf after writing: a resident write is not reported to a `BatchWriter` | `backing.rs`, `engine_bridge.rs` | join MISSING |
| `write_row` does not check `populated`; a row `>= N` returns `Stale` without incrementing `stale_write_count` | `mailbox_soa.rs` | SHIPPED |
| public cycle-blind mutators beside `write_row`: `apply_edges`, `set_edge`, `set_meta`, `set_content`, `set_topic`, `set_angle`, `set_qualia`, `set_entity_type`, `set_temporal`, `set_expert`, `set_sigma`, `reset_row`, the style-lane setters; all need `&mut MailboxSoA` | `mailbox_soa.rs` | SHIPPED; they bypass the cycle gate, not the owner |
| the documented owner stamp `SoaEnvelope::mailbox_owner()` defaults to `0` and has no caller outside its own tests; its one production implementor `NodeRowPacket` does not override it; `MailboxSoA` does not implement `SoaEnvelope` | `lance-graph-contract/src/soa_envelope.rs`, `canonical_node.rs` | UNWIRED |
| `kanban_actor.rs` holds a read-only `PhaseCensus`, `mul_target` and a step-string parser; the actor and its RPC drivers are deleted | `lance-graph-supervisor/src/kanban_actor.rs` | read-only, no write path |

Consequences:

- A foreign-schema ingestion into a resident mailbox has exactly one lawful
  entry: `write_row` under `&mut MailboxSoA` at the owner's current cycle. It
  exists and is tested, and one ingest function already uses it
  (`ingest_codebook_indices_soa`). No production code calls either. What does
  not exist at all is the step after it (the cast that reports the write).
- "`MailboxSoA` is the canonical resident owner" is true of the type and not
  yet of the default build, which still routes writes to the singleton.
- There are two owner stamps: `MailboxSoaView::mailbox_id()` (used by
  `cast_on_behalf`) and `SoaEnvelope::mailbox_owner()` (documented as the
  stamp, used by nothing).

## 1b. What the type system holds, and what it does not

Read from the field and method visibilities of `MailboxSoA`:

| property | holds by | does not hold |
|---|---|---|
| one mutable borrower at a time; a view cannot mutate | the borrow checker; `MailboxSoaView` has no `&mut` method (compile-fail doctests) | |
| outside its crate, the phase moves only through `MailboxSoaOwner` | `phase`, `populated`, `stale_write_count` are `pub(crate)` | inside `cognitive-shader-driver` they are plain fields |
| the phase moves only after a seal | nothing | `try_advance_phase` is a public trait method: any holder of `&mut` owner can step it with no cast, no seal and no writer. `SymbiontBoard::step` does exactly that from a version tick, and `mailbox_soa.rs`'s own test `test_in_ram_driving_loop_walks_rubicon_to_commit` drives the whole arc through `VersionScheduler::on_version`. The sealed-cycle driver is a convention of `cycle_driver.rs`, not a type-level rule |
| a row write is gated by the owner's cycle | `write_row` compares against `current_cycle` | `current_cycle` is a `pub` field and `tick()` is public, so the gate's clock is in the hands of every `&mut` holder; all per-column setters, `apply_edges`, `reset_row` and `set_populated` skip the gate entirely |
| the owner id is fixed | nothing | `mailbox_id` is a `pub` field: a `&mut` holder can restamp the mailbox, and `cast_on_behalf` then stamps the new id |
| a cast is paired with a write | nothing | `cast_on_behalf` takes `&self` and any payload; it does not look at `last_write_cycle` or any row |

`kanban_actor.rs` and `soa_view.rs` both state that phase progression is
enacted only by the sealed-cycle path. For code that goes through
`cycle_driver.rs` that is true; as a statement about what can happen it is not.

## 1c. Outer and inner: which side each finding is on

Added after reading `lance-graph-java` (`docs/abi.md` in full,
`docs/architecture.md`, `docs/panama.md`, `docs/execution-boundary.md`,
`docs/valhalla-lab.md`, `.claude/plans/mask-membrane-valhalla-integration-v1.md`,
its `CLAUDE.md`). Its sources were read afterwards; §1d records which, and what
they change. That repo states the boundary this audit had blurred:

> Shape may cross. Meaning may cross. Operations may cross. Ownership does not
> cross. — lance-graph owns the only canonical copy.

What crosses a membrane is identity, descriptor, layout contract, operation and
result identity. Its write side is one sentence: stage semantic intent, cross
as descriptors, the sole native writer lands and publishes; no batch or commit
verb exists on the outer surface.

Applied here:

| | outer (wardrobe, foreign schema, consumer) | inner (owner, staging, sole writer, Lance) |
|---|---|---|
| holds | a class binding: `ClassView`, `WideFieldMask`, schema revision, behaviour declaration; a reference to source-owned input | the rows, `&mut` on them, the cast, the cycle, the sink |
| may know | the row geometry (512 × 32 × (4 + 12)); it is a contract fact, and knowing it is not owning it | everything |
| never holds | a row buffer of its own, `&mut MailboxSoA`, a `BatchWriter`, a `SweepSlot`, a sink | |

Consequences for this audit's own findings:

- **The 512-byte artifact rule, the row-keyed image (§3.4), the public fields
  and the direct `try_advance_phase` (§1b) are inner findings.** A foreign
  schema cannot reach any of them: no `&mut` owner and no payload crosses. They
  are hygiene of the inner crate, and earlier versions overstated them by
  describing what "any holder of `&mut`" can do as if an adapter could be that
  holder.
- **The one finding that sits ON the boundary is the owner identity.**
  `BatchWriter::cast(on_behalf: MailboxId, …)` accepts whatever integer it is
  given. A lance-graph-java handle is an integer on the wire too (`u64`,
  generation in the high half, slot in the low half); the difference is that
  it is minted by the registry and resolved on every call against generation,
  kind and, for a mask, its parent, and a fabricated value is refused with a
  status (§1d). The Rust-side cast resolves nothing. That is the shape of the
  missing seam in §2, stated as an observation, not as a chosen design.
- **A wardrobe does not produce rows, in any width.** It resolves fields over
  input that stays source-owned; the owner writes its own rows from them. So
  "which lane takes non-512-byte source rows" was never a wardrobe question.
- A layout is a schema over the same content (`abi.md` §18: a second layout is
  a constructor, not a resource kind). A foreign schema is admitted the same
  way: as a reading, with no second population.

## 1d. What the lance-graph-java sources show

**Read in full (first to last line, tests included where the file has them):**
Rust `native/lgj-abi/src/` — `lib.rs`, `abi.rs`, `registry.rs`,
`class_view_provider.rs`, `rowstore.rs`, `fixture.rs`, `plan_lower.rs`,
`kernels.rs`. Java `java/src/main/java/com/adaworldapi/lancegraph/` — all 32
files, and `internal/ffm/` — all 6 (`Abi`, `Downcalls`, `Engine`, `Layouts`,
`PlanOp`, `Status`). Consumers, main sources only — `bricks` (`Bricks`,
`BricksQuery`, `BricksSession`, `Orders`, `Role`), `graph` (`Graph`, `Edge`),
`trades` (`Trade`, `World`). `valhalla-lab/docs/three-truths.md`.

**Read in part:** `native/lgj-abi/src/exports.rs` — every exported function and
helper, and the first part of its test module; the rest of that test module was
not read.

**Not read:** `exports/tests/*.rs`, `native/lgj-abi/tests/*.rs`, the Java test
suites, the consumers' tests, `bench/`, `interop/tinkerpop/`, the
`valhalla-lab` sources and reproducers, `README.md`. Nothing below rests on
them.

What the sources establish, each stated for lance-graph-java only:

1. **The outer side holds a vocabulary, not rows.** `Trade`, `Orders` and
   `Edge` have private constructors that throw; no instance of any of them can
   exist. Each is a list of constants binding a domain name to a lane:
   `Orders.REGION`, `Trade.VENUE` and `Pattern.CLASS` are three names over the
   same lane of the same resource (`Orders.java`, `Trade.java`,
   `Pattern.java`). Three schemas, one population. `Pattern.java` says the
   file "is shaped exactly as a code generator would emit it"; `Field.java`
   separates a generator surface (constructor, `lane()`) from a consumer
   surface (`name()`, the typed comparisons). This is the admitted shape of a
   schema on the outer side, and it carries no bytes.
2. **Operations cross as descriptors and are validated whole before any
   write.** A `Predicate` wraps a `PlanOp` with no accessor for opcode, lane or
   operand (`Predicate.java`); `Engine.marshal` writes the chain as
   `LgjOpDesc` records; `validate_plan` in `exports.rs` checks every op before
   the destination is touched.
3. **The meaning of a row's bytes is resolved natively, once, at the edge of a
   population.** `lgj_reduce_facet_sum_resolved` resolves
   `classid → ClassView → cascade_shape` for the selected rows through
   `kernels::resolve_population_carving` and refuses a population that spans
   differing readings, contains a classid with no answer, or is empty
   (`UNRESOLVED_CARVING`). `lgj_hop` narrows the caller's facet mask by
   `class_view_provider::edge_participation`; the caller cannot widen it. The
   sibling `lgj_reduce_facet_sum` applies a caller-named reading unchecked, and
   `RowStore.facetSumAs` says so in its own doc.
4. **Authorization is composed into the selection before execution.**
   `BricksQuery.authorize` adds the role's predicate to the same lazy chain;
   every terminal refuses an unauthorized chain before any native call
   (`requireAuthorized`). No row leaves that package's public surface.
5. **Identity is checked, not trusted.** `registry::resolve`,
   `resolve_kind` and `resolve_mask_with_parent` gate every export. The
   register sweeps additionally require the mask's parent to be the same
   resource as the store (`Arc::ptr_eq` in `lgj_reduce_facet_sum` and its
   resolved sibling). `every_stale_handle_shape_is_a_status_not_a_crash` in
   `exports.rs` covers closed, double-closed and fabricated handles.
6. **No export writes a row.** Every pattern and row-store lane is described
   `READABLE` and never `WRITABLE` (`lgj_lane_describe`); the only writable
   lane is a mask's word lane (`lgj_mask_describe`), and the only Java code
   that writes through it is the named import (`RowStore.importRows`). The
   stores are generated from a seed. So the write side this audit quoted in
   §1c (stage intent, cross as descriptors, sole native writer lands) is
   **specified in that repo's documents and absent from its code**. Nothing in
   lance-graph-java can be cited as a working example of an outer producer
   handing work to the inner owner.

Limits of that evidence, so it is not over-read:

- lance-graph-java's own membrane is not closed everywhere. `RowStore` reads
  its cached lane windows against a non-volatile Java flag, and `LaneProbe`
  ships as a no-op; `Engine.close` is reachable outside the owning object.
  The repo records both as open.
- The outer side does hold buffers proportional to rows in two named places:
  the result segment of `RowStore.facetMatches` (one `u32` per row, written by
  native code) and the array returned by `Mask.materializeRows`. Neither is a
  copy of a row. The §1c table's "never holds a row buffer of its own" means
  canonical rows.
- The `ClassView` answers there come from a fixture provider unless the
  `ogar-classview` feature is on (`class_view_provider.rs`).

What this changes for the wardrobe question:

- A foreign schema, admitted, looks like item 1 plus item 3: names over
  existing lanes, and `ClassView` answers for its classids. Item 6 means the
  read half of that has a working precedent and the write half has none.
- §7 already said the cold admission path exists in lance-graph
  (`BindStore::register`). That stands. What does not exist in either
  repository is a caller that goes from an admitted class to
  `write_row` + `cast` under a checked owner (§1a, §2).

## 1e. The Quack surface against the same boundaries

Operator requirement (2026-11-12): the Quack polyfill needs
non-materialisation boundaries like lance-graph-java's. This section records
where `lance-graph-quack` stands against them. Nothing was changed in that
crate.

**Read in full:** `crates/lance-graph-quack/src/lib.rs` and `src/bind.rs`, all
non-test code. **Not read:** both files' test modules, the crate's `tests/` and
`examples/`, `lance-graph-mask-risc` (the executor, where `execute_into`,
`execute_sel`, `Out`, `Planes` and `materialize_rows` live) and
`lance-graph-cypher-quack`. Statements about the executor below are what
Quack's own doc comments say about it, and are marked so.

The structural difference comes first. lance-graph-java has a membrane: the
rows and the mask words live behind handles, and the outer side gets names.
Quack has none by design: its module doc says it "may build a `Program` and
must never evaluate one", and the consumer executes on planes, scratch and
output buffers the consumer owns. So in Quack every boundary is either a
property of what can be lowered, or a documented obligation of the caller.

| boundary | lance-graph-java | `lance-graph-quack` as read |
|---|---|---|
| nothing proportional to rows leaves except through a named materialiser | `Mask.materializeRows` is the only one; the default terminal is a count | the materialiser is named (`materialize_rows`, per the `Agg::Rows` doc) and is the caller's to call. But `bind::Draft` defaults to `Want::Rows`, so a draft with no terminal lowers to `Terminal::Keep` and a population-sized mask in the caller's buffer |
| a population never travels between two programs | one fused program per terminal; `lgj_plan_group_sum_i32` exists to remove the selection | `lower_group_by` returns a `GroupPlan` whose phase 1 keeps a filter that the caller binds as a plane for K phase-2 programs. `lower_group_by_semantic` / `lower_group_by_auto` fold it into one program for `SUM`/`COUNT`/`MIN`/`MAX`; other aggregates still take the two-phase forest |
| the forbidden shapes cannot be expressed | no export takes a row list; import is one named method | `Filter::Semijoin` requires its foreign plane to be resident and never "a mask another `Program` produced"; `Agg::ScatterOrU32` forbids handing its mask to a second program. Both are stated in doc comments. `ForeignPlane(pub u16)` is an index, so neither is checked |
| a row-id list is never the internal currency | forbidden outside `importRows` / `materializeRows` | `lower_sel` admits a borrowed position list as an operand for two folds (`SUM(i32)`, `COUNT(i32 > t)`), by recorded decision; every other query is refused (`NotASelectionFold`) |
| identity is checked, not trusted | handles: generation, kind, parent on every call | `Col`, `Mask`, `ForeignPlane`, `ForeignLane` are public `u16` indices. A `Query` carries no table identity: `bind` resolves a `TableId` and drops it. `PrefixLowering::Bound` says in its own doc that "a `Planes` value carries no order identity" and that the order precondition is "a documented obligation with carried evidence, not an enforced invariant" |
| stale meaning is refused | epoch on descriptors; `PARENT_CLOSED` | `TypedQuery::admit` re-checks a code-space witness and refuses `StaleCodeSpace`; `Draft::bind` returns a bare `Query` with no witness, and `admit` hands out `&Query`, which is `Clone` |
| no text or physics past the boundary | no FFM type on a public signature (`ApiSurfaceTest`) | no text past `bind` (the module doc names `tests/string_fence.rs`); lane and plane indices are the public vocabulary by design |
| answers sized by the question | `GroupTotals`, `long[groups]` | `Out::I64` of `groups` slots; `GroupHavingPlan::finish` and `normalize_group_sink` work in O(K) and return K-bit masks |

Three observations, none of them a design:

1. The two repositories agree on the rules. Quack states most of them in its
   own comments ("No O(N) kept-mask crosses a program boundary" on
   `lower_group_by_semantic`). The difference is that lance-graph-java
   enforces them by what its surface can express, and Quack mostly documents
   them, because it has no execution boundary to enforce them at.
2. The missing identity is the same one as in §1c. A lowered `Program` is
   bound to nothing: not to a table, not to a row order, not to a plane set.
   `BatchWriter::cast` has the same shape on the write side.
3. `bind::Registrar` (`create_table … WIDTH bytes`) is a declared seam for a
   table catalog with no production implementation; its doc says so. A
   wardrobe that declares a foreign schema would meet this seam, and it is not
   built.

What a like-for-like boundary would need is a decision, listed in §10.

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

The delegation cache does not close this. `BatchWriter::resolve_owner` is a
separate method that `cast` never calls; its only callers are in
`crates/lance-graph-planner/tests/w1_probes.rs`. The knowledge documents
describe a check "at cast time"; no code performs one.

This is not a new discovery. `batch_writer.rs` says so in its own module doc
(STATUS note, 2026-07-27: `cast()` "has **zero production call sites**", the
durability read side is unwired) and names `collect_casts` as the actual
consumer (CORRECTED note, 2026-08-04), together with a rule the audit must
carry: **at most one move per owner per cycle is sealed**; further moves are
held and re-cast (`collect_casts`, `restage_held`).

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
4. **The durable payload is keyed by row, not by owner.**
   `DetachedCycleBatch::freeze` folds payloads into `image: BTreeMap<row,
   payload>`, and `LanceCycleWriter::build_batch` writes each image row with
   `owner = 0`. The owner is recorded only on the landing rows, which carry no
   payload. Two mailboxes whose `row_of` returns the same row in one cycle
   therefore leave one payload (the later stream position) and two landings;
   nothing refuses it. Pinned as
   `finding_two_owners_on_one_row_leave_one_image_row`. `write-on-behalf.md`
   says "every persisted row is attributable to a mailbox"; for image rows
   that holds only through a join on `(cycle, row)` and only while rows do not
   collide across owners.
5. **A step can happen without a seal** (§1b): by a direct
   `try_advance_phase`, and inside the sealed path itself for an intent-only
   cycle (§1, `NoChange`). Pinned as
   `finding_an_intent_only_cycle_steps_without_a_version`.
6. **The singleton is still writable.** `engine_bridge.rs` and the default arm
   of `backing.rs` write `BindSpace` with no owner and no cycle gate (§1a).

## 4. Copy ledger

| stage | what happens | copies of payload bytes |
|---|---|---|
| borrowed SoA read through `MailboxSoaView` | slices of the owner's columns | 0 (pointer equality pinned by the suite) |
| cast | payload moved into `BatchWriter` | 0, but with `P = Vec<u8>` the payload is owned bytes the caller built |
| drain (`drain_pending_payloads`, `collect_casts`) | payload moved into `SweepSlot` | 0 |
| submit (`try_submit`) | the `Vec<SweepSlot>` is moved | 0 |
| seal: retry copy (`seal_cycle`, `casts.clone()`) | held until commit success | 1 |
| seal: freeze (`DetachedCycleBatch::freeze`, `payload.clone()` per row) | coalesced image | 1 |
| Lance append (`LanceCycleWriter::build_batch`) | image payloads appended to a `FixedSizeBinaryBuilder`; the module doc states this copy boundary itself | 1 |
| read back (`scan_image`) | `payload.value(i).to_vec()` | 1 per read |

At commit time one artifact payload therefore exists four times: in the retry
clone, in `landings`, in `image`, and in the Arrow builder. The landing rows do
not write it (their payload column is null).

`LanceCycleWriter` accepts an artifact payload only if it is exactly 512 bytes
(`EPISODIC_WITNESS_BYTES`); any other length is `CommitError::InvalidArtifact`,
permanent. This is an INNER fact: the canonical node row stride
(`key 16 | edges 16 | value 480`) as the sole writer's ABI. It constrains the
owner that casts, and nobody else.

⊘ An earlier version of this paragraph concluded "a wardrobe that hands
arbitrary source rows to this sink cannot work; it must produce 512-byte node
rows or use a different lane". That sentence put the wardrobe on the wrong side
of the boundary and is withdrawn; see §1c.

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

- `mailbox-kanban-model.md` says the delegation cache is checked "at cast
  time" and lists "the ahead-firing batch writer itself" as still needed. The
  writer exists; the check does not (§2).
- `mailbox-kanban-model.md` gives the trigger as the version tick
  (`on_version → try_advance_phase`); `d-mbx-a6-owner-consume-and-persistence.md`
  and the module doc of `cycle_driver.rs` say a version is not permission to
  step and only the paired move is. The two documents contradict each other;
  `batch_writer.rs` records the same correction in its 2026-08-04 note.
- `d-mbx-a6-owner-consume-and-persistence.md` names "the ractor single-owner
  guarantee" as the version-ordering guarantee and pins Lance 7 / DataFusion
  53. The actor is deleted; the workspace is on a later Lance.
- `cycle_driver.rs`, honesty ledger: "Durability: fake … until the concrete
  `LanceShardSink` lands". The concrete sink is `LanceCycleWriter`, and
  `persist_sink.rs` names it.
- `persist_sink.rs`, `WalSink` trait doc: "`&self` (shared) and no owner
  borrow", two paragraphs above `commit_cycle(&mut self, …)`.
- `batch_writer.rs` names `persist_sink::recover_and_apply` as what applies the
  paired move. That is the restart path; the normal path is
  `apply_sealed_transitions`.
- `batch_writer.rs`: "the sink reads them through `NodeRowPacket::as_le_bytes`
  at flush time". `collect_casts` is typed `BatchWriter<Vec<u8>>` and the sink
  receives owned bytes (§4).
- `mailbox_soa.rs`, comment on `cast_on_behalf`: a mispaired owner is
  "unrepresentable". It is unrepresentable through that method; `mailbox_id`
  is a public field and the id-taking `cast` is public (§1b, §2).
- `write-on-behalf.md` cites `cognitive_pass` by a line number that no longer
  matches.

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
| an intent-only cycle steps with no version | `finding_an_intent_only_cycle_steps_without_a_version` | runtime, pins current behaviour |
| two owners on one row leave one payload | `finding_two_owners_on_one_row_leave_one_image_row` | runtime, pins the GAP (with a distinct-rows twin) |
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

**Results.** 11 of 11 runtime tests and 7 of 7 doctests pass (3 laws, each a
compiling control beside its compile-fail twin, so a typo cannot pass as a
refusal). Disable runs against production code, each patch asserted to apply
once: the owner stamp replaced by 0 (3 tests red), the cross-owner check in
`persist_cycle` removed (1 red), a failed seal returning no casts (1 red), a
full queue dropping its casts (1 red): 4 of 4 red. The two `finding_` tests added in the third version have no disable run: each pins current behaviour and fails when that behaviour changes; the row test carries a distinct-rows control so it cannot pass on an image that merges everything.

**Overlap with existing tests.** Two controls restate properties
`mailbox_soa.rs` already tests in its own module, found only when the file was
read in full: `the_owner_stamps_its_own_cast`
(`pairing_takes_the_owner_from_the_carrier_never_the_call_site`) and the
pointer identity in `a_view_is_the_owners_own_columns`
(`test_edges_raw_meta_raw_reinterpret_round_trips`). They are kept because
this suite runs them against the supervisor path, and are not new coverage.

**Limits of the suite.** The fence in
`a_second_writer_on_a_stale_head_is_fenced` and the failure injection are in
the in-file sink; they check that `seal_cycle` carries a sink's refusal
through unchanged, not that `LanceCycleWriter` fences (its own tests in
`cycle_sink.rs` do). `a_cast_is_not_a_commit` cannot fail through the sink,
because a cast holds no sink; it pins that the owner's phase does not move.
Every test maps every owner to row 0 (`row_of = |_| 0`), which is the
collision §3.4 describes; the suite did not notice it until the freeze code was
read. No test here exercises `write_row`, `LanceCycleWriter`, or `MapLog`.
What the type system does and does not hold for the owner itself is §1b.

**Not testable today:** "a consumer writes directly to Lance instead of
delegating". Inside one process a second `LanceCycleWriter::open` on the same
store is refused (`WriterClaim` over a process-wide registry; the sink's test
`a_second_live_writer_on_the_same_path_is_refused`), keyed on a lexical path
identity that its doc says symlinks and scheme rewrites defeat. Across
processes, and for a crate that calls `lance::Dataset::write` itself, nothing
in the type system applies; the fence (`CommitError::Fenced`) reports a moved
head after the fact.

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
move before its ticket resolves `Ok`; map two owners to one image row in one
cycle; give the wardrobe or any adapter a row buffer, a writer or a sink of its
own (§1c);
write the singleton `BindSpace` for a population that has a mailbox owner; or
describe the path as zero-copy.

## 10. Smallest follow-up (proposed, not built)

1. Decide the enforcement seam for `on_behalf` (§2).
1a. Decide which owner stamp is the contract (`mailbox_id` or `mailbox_owner`) and retire or wire the other (§1a).
1b. Decide whether a resident `write_row` must be followed by a cast, and who makes it (§1a).
2. Decide whether `collect_casts` should carry descriptors (§4) before any
   resident ingestion is sized against it.
3. Correct the stale documents and comments (§5) in a documentation PR.
4. Name the owner for a resident codegraph population (§8).

The plain-id cast (item 1) is the write-boundary admission issue and stays on
this list. The three items below are NOT part of the wardrobe admission gate.

### 10a. Internal hygiene track (separated 2026-11-12; evidence kept, all unresolved)

These concern the inner owner's own correctness and API, independent of any
outer producer. Nothing here is to be changed as a side effect of admission
work.

| item | kind | evidence | status |
|---|---|---|---|
| The cycle image is keyed by row and written with owner 0; decide `(owner, row)` or prove `row_of` injective across owners | internal persistence correctness | §3.4 | unresolved |
| `NoChange` on an intent-only cycle, a failed seal, and a direct `try_advance_phase` are three different transitions; decide which of them "no seal, no step" binds | internal lifecycle governance | §1b, §3.5 | unresolved; no change before the three are specified separately |
| `mailbox_id` and `current_cycle` are public fields | internal owner API hygiene | §1b | unresolved |

### 10b. Quack execution boundary (decided 2026-11-12; work moved to PR #2045)

The four questions raised in §1e were decided as follows. None changes a
production API.

| question | decision |
|---|---|
| `Draft` defaults to rows | Retained. A demanded `Keep` mask, a tile-local mask, a population mask handed to another program, materialised row indices and SQL tuples are five different things; only the third and later are the materialisation expense |
| two-phase `GroupPlan` | Kept, behind the existing semantic-first `lower_group_by_auto`. A forest that carries a population mask between programs is reported as intermediate materialisation |
| source identity in `Program` | Not added. The consumer-side binding is audited instead |
| resident-plane preconditions as comments | To be checked at the admission boundary, not in the physical IR; where no ratified authority exists the case stays BLOCKED |

The lab that follows these decisions is `crates/quack-boundary-lab`
(plan `quack-boundary-lab-v1.md`, PR #2045). It is a separate proof and adds
nothing to this audit's findings.

R4, R7 and the reader `Scope` issuer are untouched.

## 11. Corrections to earlier versions of this audit

Each was caused by writing from an excerpt, a search hit or a summary.

| earlier statement | what the source says |
|---|---|
| `cognitive_pass` does not exist | it exists, private, in `cycle_driver.rs`; the search output had been truncated (fixed before the first push) |
| the resident write was not mentioned at all | `write_row` is the cycle-gated row mutator (§1a, added in the second version) |
| "callers of `write_row` outside tests … SHIPPED" | both callers are themselves uncalled in `src/`; one is `dead_code` by its own annotation (§1a) |
| "`MailboxSoA`'s columns and `mailbox_id` are public fields" as the whole story | `phase`, `populated` and `stale_write_count` are `pub(crate)`; `current_cycle` is public, which matters more (§1b) |
| the forgeable `on_behalf` and the missing production caller presented as findings of this audit | `batch_writer.rs` records both in its module doc (§2) |
| "failed seal advances nothing" read as "no seal, no step" | an intent-only cycle steps under `NoChange`; a direct `try_advance_phase` needs no seal (§1b, §3.5) |
| Lance append "at least 1 (not measured)" | exactly one, stated in `cycle_sink.rs`; plus the 512-byte artifact rule, which the audit had missed (§4) |
| nothing on the image key | the image is keyed by row and written with owner 0 (§3.4) |
| nothing on the delegation cache | `resolve_owner` is never called by `cast` (§2) |
| "a wardrobe cannot hand `LanceCycleWriter` arbitrary source rows; it must produce 512-byte rows" | a wardrobe hands the writer nothing. The 512-byte rule is the inner writer's row ABI; the wardrobe is outer and carries a binding, not bytes (§1c) |
| inner-crate facts (public fields, direct phase step, row-keyed image) presented as bypasses open to a foreign adapter | no `&mut` owner crosses the boundary; these are inner hygiene. The boundary finding is the plain-integer owner id (§1c) |
| "in the lance-graph-java membrane the equivalent value is never a plain integer" | a handle is a `u64` on the wire. It is provenance because the registry mints it and resolves it on every call, not because of its type (§1c, §1d) |
| the outer/inner section written from lance-graph-java's documents only | its Rust and Java sources read afterwards; the write side quoted from the documents has no code behind it there (§1d item 6) |
