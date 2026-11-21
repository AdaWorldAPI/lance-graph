# The register contract: its contradictions, how each is resolved, and what follows

> **READ BY:**
> - any session that reads lanes through DT64 (`LaneBinding`, `borrow_all*`,
>   a `LaneSource` / `PresenceSource` / `SelectionSource`);
> - any session that writes a schema, revision or basis check;
> - any session that answers a missing value, a deleted row or an outer-join side;
> - any session that joins over a cube or over a consumer's tables;
> - any session that ports a consumer store onto the 512-byte SoA rows;
> - `layer-boundary-warden`, `premise-auditor`, `prior-art-savant`,
>   `v3-envelope-auditor`, `cypher-lowering-warden`.
>
> **Status:** CURRENT-CONTRACT where a row says DECIDED and cites the decision.
> PROPOSED where a row names a resolution that waits on an owner decision.
> OPEN where neither exists. Every claim carries its grade: MEASURED (with its
> receipt), VERIFIED-IN-CODE (with its path), QUOTED (a plan's or knowledge
> doc's words, with path:line), or the owner's words with date (provenance, not
> validation, per `FIRST-HAND-SOURCE-LAW.md`).
>
> **Code (2026-11-21):** the polyfill these rows constrain is
> `lance_graph_contract::pg_row` plus `crates/lance-graph-pg-row`, §4.
>
> **Written:** 2026-11-21, from the zammad-rs session's measurements
> (zammad-rs #95, #96, #97) and lance-graph #2146–#2151, read at origin/main
> `87d7b41`. Plan with the D-ids: `.claude/plans/d-regc-register-contract-reconciliation-v1.md`.

## 0. The contract in five lines (what nothing below changes)

1. A row is `identity[index].key : identity[index].value`: aligned lanes read
   at one ordinal, each with its presence bit. It is never a stored pair
   (EXO-332; `classid-on-do-call-shape.md` §0).
2. Before the last step there is one ordinal register: the lanes lent by one
   `borrow_all*` call, read at the same position. The join is the last step
   (EXO-332, owner 2026-11-18).
3. A read is lent in place and encoded only at the wire. The only
   materialization is the returned cells (QUOTED
   `plans/d-qsp-classview-quack-postgresql-parity-v1.md:30-39`).
4. The substrate is the 512-byte SoA row with an owner (`canonical_node.rs`,
   `SoaEnvelope::mailbox_owner`). A consumer's own `Vec` lanes are an interim
   store, not a second substrate (owner 2026-11-20: "SoA 512 with ownership
   guarantee and potential borrow_all is the only Substrat").
5. Absent is one code: presence bit 0 and value 0, i.e. −0, at 8 and 9 bits
   alike. At 9 bits the presence bit is the sign, so presence 0 with a nonzero
   magnitude is a negative value, never absent (VERIFIED-IN-CODE
   `lance-graph-contract/src/lane_codec.rs:55-63`; `lance-graph-dt64/src/lane.rs:119`).

## 1. The contradictions

Each entry is written the same way: the sources that disagree, quoted; the
resolution this document proposes or records; its implications; the falsifier;
and who decides.

### C1. Schema store: Decision J against immutable classids — PROPOSED, owner decision needed

**Sources that disagree:**
- QUOTED `plans/d-dt64-7-behaviour-addressing-registration-v1.md:158-163`,
  Decision J (owner, 2026-11-07): "A class's ordered `predicate_iri` basis is
  immutable within one class `SchemaRev`. Reordering, replacing, removing or
  appending a field takes a new revision."
- QUOTED `plans/d-dt64-6-routing-bindstore-behavior-v1.md:31`: "A classid's
  behaviour is permanently immutable. It is versioned under a fixed concept
  through a derived DTO classid."
- QUOTED `d-dt64-6:397`: EXO-199, "the classid stays the class; no class
  versioning".
- QUOTED `d-dt64-6:158`, OPEN: is the immutable unit a `(classid, binding
  version)` entry, or one classid per revision?
- Owner, 2026-11-20: "classid per se are immutable / I don't want any Schema
  store because we enforce classview".

**Code today (VERIFIED-IN-CODE, `lance-graph-dt64`):**
- `SchemaRev` (`lib.rs:128`) and `ClassSchema.schema_rev` (`:267-268`).
- `BindStore::register` binds "at `schema.schema_rev` (Decision J)" and keeps a
  basis digest (`:829-843`).
- `LaneBinding::check` re-checks that digest and every bound predicate on every
  read (`lane.rs` around `:400-440`, `StoreBasisMismatch`).
- `SchemaRev` is named in 10 crates' source (`lance-graph-mailbox-lanes` 14
  files, `-r2il` 8, `-dt64` 8, `-xlog` 7, `-report` 5, and lab crates).

**Measured weight:** schema and basis sub-checks are about a third of each
`LaneBinding::check`, 49,140 of 146,453 over zammad-rs's 512-call wide workload.
This is derived from `check` at `87d7b41`; only the source callbacks are counted
from zammad-rs (MEASURED, zammad-rs#96 `tests/register_census.rs`).

**Proposed resolution:**
- One classid has one field basis forever. A new shape is a new classid, the
  "derived DTO classid" of d-dt64-6:31. That answers d-dt64-6:158 with "one
  classid per shape".
- The `ClassView` is the schema and its only enforcement. `SchemaRev` becomes a
  constant of the classid, not a store.
- The basis is resolved through the `ClassView` at registration and at
  bind, never again per read. After bind, a read uses the basis recorded in
  the binding, not the view.

What a read must still refuse, and how. Review on #2152 pointed out that
`register` alone cannot cover either case:
- **A different store.** A store may hold a different basis for the same
  classid. Today `LaneBinding::check` refuses this with the O(1) compare
  `bound != self.basis` (`lane.rs:427-430`): the store's registered digest
  against the binding's. No hash is recomputed. Before that compare,
  `resolve_counted` asks the view for the fields and checks revision and
  count through `check_basis` (`lib.rs:1128-1140`). The proposal keeps the
  digest compare and drops the trip through the view.
- **A view mutated after bind.** Owner, 2026-11-21: a `ClassView` changes
  only while ruff's class proposer stages classes that do not exist yet. That
  staging is Explore, outside the store (`classid-on-do-call-shape.md` §113:
  "There is no lifecycle state besides versioned, replayable immutability").
  Once registered, a class is read through the BindStore as `xlog::Cube ×
  DT[ClassView] × DR[WideFieldMask]`. In the normal scenario a change is a
  new versioned reference, never a mutation in place. A read compares the
  reference it was bound under (O(1)) and is replayed by that reference: the
  kept binding (`MapTable::keep / kept / kept_under`,
  `LaneBinding::keep_in`, #2150) and the route log (`RouteVersion`).
  So in-place mutation is not a read-path case. If it ever happens, solving
  it belongs to the replayability table, not to a per-read basis check.
  Until that table owns it, the cheap rule stands: no read consults the
  view, so a mutation changes no answer.

**Replay key** (owner, 2026-11-21: data is replayable in relation to version
× DR reference × BindStore version). A read is replayable only when all
three are named:
- **data version:** the source's sealed generation, `FrameRevisions::source`
  (for Lance, the dataset version the lanes come from);
- **DR reference:** the row order and relation map,
  `FrameRevisions::{relation_map, coordinates}`, plus the route version
  (`RouteVersion`), under which mappings are kept (`MapTable::keep / kept /
  kept_under`, #2150);
- **BindStore version:** the registered class reference (classid, revision,
  basis digest) the binding was made at.

The three move independently, and each derived artifact depends on some of
them:
- A resolved key position depends on the DR reference.
- A kept WHERE mask or an order index depends on the data version too.
- A binding depends on the BindStore version.

An artifact is reused only while every component it depends on is
unchanged. Otherwise it is recomputed or replayed under the old triple. The
frame check, the row-order check (`CoordinateSpace` / `SelectionSpace`) and
the digest compare are the three refusals that enforce this today.

**Implications:**
- **D-REGC-2:** retire the per-read re-resolution of the basis through the
  view. Keep the per-read digest compare against the store, and
  `register`'s refusal of a changed basis.
- A registered class changes only by a new versioned reference. An in-place
  change of a classid's fields is a contract violation. The replayability
  table owns it, and reads never see it.
- What becomes of Decision J's "append-only across revisions" (`class_view.rs:1058-1059`)
  is part of the owner decision. Across classids, it becomes "a derived classid
  extends its parent's prefix".

**Falsifiers:**
- A `ClassView` that returns a different basis for an already-registered
  classid is refused at `register` and at bind.
- A read through a store holding a different basis for the classid is
  refused.
- A view mutated after bind leaves every read's answer unchanged.

**Decides:** the owner (DECISION NEEDED, lance-graph · Decisions).

### C2. Is a binding needed for a read through one class lens? — PROPOSED

**Sources:**
- Owner, 2026-11-20: "a class doesn't need binding if the caller has one class
  as a reading lens".
- QUOTED `plans/d-lanesock-dr-dt64-lane-socket-v1.md:179`: "Validation stays per
  call. There is no check-once or lease."
- The 2026-11-21 survey found no plan that defines or refuses a lens-only read.
  The search space was the DT64, lane-socket, class-governance v2/v3, quack-seam
  and xlog plans read for it, so this is not a global negative. The only "lens"
  in those plans is the behaviour lens `0xF###`.

**What `bind_lanes` does (VERIFIED-IN-CODE, `lance-graph-dt64/src/lane.rs:261-300`):**
1. route: which class is at a DR position;
2. basis: resolve and check the class's fields;
3. mapping: each field's `FieldRef` → the source's lane via `lane_of` by predicate;
4. witness: the frame is the source's current seal, the scope matches, and the
   row order matches.

**Proposed resolution:**
- With one class held as the lens, 1–3 are fixed by the lens itself. Route is
  not consulted; basis per C1; mapping is the source's own lookup by the lens's
  predicates.
- Per read, only 4 remains: frame, scope and row order.
- A read API of the form "lens + frame + scope + order → lanes (+ presence,
  + selection)" does not add a registry. It is `borrow_all*` without steps 1–3.

**Implications:** D-REGC-1, a dt64 read with no `LaneBinding`, gated on C1.
Consumers with one class per table (zammad-rs: `ClassMap` holds one class per
table) drop per-bind route and basis work. #2150's kept mappings remain the form
for routed reads (a DR position that can hold several classes).

**Falsifier:** the lens read refuses exactly what `borrow_all_present` refuses
for frame, scope and order (foreign frame, stale frame, another row order,
another scope). It is equal on every lane, and it counts no route or basis
sub-check.

**Decides:** the lance-graph DT64 owner, after C1.

### C3. Missing, empty, deleted: one absent code, or several? — RECORDED (owner) + PROPOSED (delete)

**Sources:**
- VERIFIED-IN-CODE `lane_codec.rs:55-63`: absent = presence 0 and value 0, named
  as NaN, `$null` and UNKNOWN.
- Owner, 2026-11-20/21: "In our Substrat NULL = NaN"; "0 presence 0 value= -0 =
  NaN"; "we follow the Powershell logic: 0 = number, $null = NaN = '' = NULL
  (SQL/SAP)"; "The only thing that is usually is ''=$Null and it says no value or
  missing or deleted"; "We never use float"; another presence bit is "useless
  until we really need it".
- MEASURED zammad-rs#95: a deleted row keeps presence 1 and its old values.
  Reading it returned 547 stale rows where PostgreSQL returned NULL (deleted
  parent, left join).
- OPEN: EXO-337 (may a selection name a non-live position) and EXO-294
  (decision B, absent = −0).

**Resolution, recorded** (DECISION EXO-372): one absent code, −0, for `$null`, NaN, UNKNOWN, `''`,
missing and deleted. `0` is a number (presence 1, value 0). One presence bit, no
second one until a measured case needs it.

**Resolution, proposed for deletes:**
- A delete writes −0 on every lane of the row (clear presence AND value) and
  clears the live bit. A dead row then reads as absent, like an untouched slot.
- Clearing presence alone is wrong. Under a 9-bit reading, a nonzero magnitude
  with presence 0 is a negative value.
- Until this lands, reads fail closed: a selection that names a non-live
  position is refused. VERIFIED-IN-CODE zammad-rs `Refusal::DeadPosition`, which
  is checked on the live plane, not on presence, because a live row may hold −0
  cells.

**Implications:**
- **D-REGC-3:** a delete writes −0; the live bit stays authoritative. Until it
  lands, `lance-graph-pg-row` reads liveness through its own `LiveSource`
  socket, intersects every `WHERE` with it and refuses a dead position (§4).
- **D-REGC-10, egress to PostgreSQL** (in PR: `pg_row::AbsentAs`): an absent text cell renders as `''` where
  the column's default is `''`, and as NULL otherwise. MEASURED in zammad-rs's
  wide capture: 1958 `''` cells in PostgreSQL answers, from 16 nullable text
  columns that never also hold NULL.
- The PostgreSQL harness keeps `''` and NULL distinct. That is what makes the
  egress rule falsifiable.

**Falsifiers:**
- A delete followed by a read returns −0, never an old value.
- A column holding both `''` and NULL in PostgreSQL breaks the egress rule, and
  the harness reports it.

**Decides:** the owner. The deletion half is EXO-337 / EXO-294; the absent code
is recorded from the owner's words.

### C4. A key with no live row: no position, or a dead position? — DECIDED in a consumer, PROPOSED for DR

**Sources:**
- MEASURED zammad-rs#95: a missing parent (key 0) was answered by reading
  Identity's dead position 0. It was right only by layout coincidence.
- report-rs#175 pins the same rule as `Unresolved`
  (`a_known_identity_without_a_binding_is_unresolved`).
- VERIFIED-IN-CODE `MapTable::locate` refuses `Ambiguous` / `Unmapped`. DR maps
  identities and does not know liveness; liveness is the owner's live plane.

**Resolution:**
- A key with no live row resolves to no position.
- The missing side of an outer join is −0, read at no position.
- DECIDED and test-pinned in zammad-rs (`position_of` returns `None` for a
  non-live key or one past the last row).
- PROPOSED for every DR consumer: locate answers "unresolved" and the read
  refuses a dead position.

**Implications:** D-REGC-4, the selection carries its own presence plane, so
that an outer-join side with no position renders −0. A selection today is
`&[u32]` positions only, and a reserved −0 position is the alternative. This
needs a decision. Until then `lance-graph-pg-row::read_bag` takes the plane
per side from the caller and does not read the position at a 0 bit (§4).

**Falsifier:** zammad-rs#95's deleted-parent arm. It was red on the old code (547
stale rows) and is green with the rule, in both layouts.

### C5. Coordinates are bijective; relations are bags — CURRENT-CONTRACT

**Evidence:**
- MEASURED lance-graph#2149 (cube, one tile): `global(local(g)) == g` for all 38
  ids, and the reverse for every coordinate. Outbound costs 0 map lookups;
  inbound costs 1 per cell.
- MEASURED zammad-rs#96 `tests/cube_bijectivity.rs` (consumer DR placement):
  - the round trip, no shared positions, and no position for dead or absent keys;
  - after every one of 51 feeds, in both layouts;
  - on the size-1025 capture across 5–6 tiles, with feeds that reach a new tile,
    delete across tiles and revive a key.
- MEASURED report-rs#175 `a_repeated_binding_is_a_bag_not_a_set`; zammad-rs and
  #2149 `join_size` equal PostgreSQL's `count(*)` with duplicates.

**Contract:**
- The coordinate maps are bijective onto live rows.
- A relation is a bag and must never be made a set: PostgreSQL join semantics.
- The coordinate space includes row order, not only the frame. A logged edge or a
  refold moves rows without moving the frame (#2136, #2146).

**Still unmeasured** (D-REGC-8):
- the cube across several tiles (`MapTable::locate`);
- the cube's bijection after `add_edge` and `refold`;
- the cost of `locate` per key in a multi-tile join.

### C6. Joins: per-row key reads in a consumer against the aligned pass — CURRENT-CONTRACT (direction) + OPEN (admission)

**Sources:**
- QUOTED `knowledge/classid-on-do-call-shape.md:35-38`: "A join is an aligned
  pass over lanes that share coordinates… Key sets, translated positions and
  reads through a key are what a consumer falls back to".
- QUOTED `:149-155`: zammad-rs's form is "parity evidence for the answers, not
  the join form to copy".
- QUOTED `:313`: the fix is at intake.
- MEASURED zammad-rs#96:
  - joins place one key per kept tuple per dimension from an edge-lane value
    (546 resolutions over 57 wide calls), all after the last fold;
  - the join's tables have no class binding;
  - the placed positions are built into `JoinStats::tuples` and not used for the
    rendered cells.
- OPEN: EXO-140 (`EqU32Via` / `Semijoin` / `Gather` admission) and EXO-327
  (filter on a join side).

**Resolution:**
- The join is an aligned pass over classes placed on shared coordinates at intake.
- Its count is a fold (`Σ left·right`, `Cube::join_size`).
- Its rows are a virtual tuple bag: one selection per side, aligned by index,
  row `i = (laneA[selA[i]], laneB[selB[i]])`, read through one pointer table and
  never built.
- A value-driven per-row key read is a fallback, gated by ADMISSION=B (QUOTED
  `plans/ascent-byods-dr-fold-polyfill-v1.md:66`).

**Implications:**
- **D-REGC-9:** consumer intake places related classes on shared coordinates.
- D-REGC-4 supplies the outer-join side.
- D-REGC-5 issues the selections.
- Off the execution path (QUOTED):
  - egglog / DialEgg certify rewrites offline (`plans/d-nam-native-algebra-dto-capstone-v1.md:29`);
  - "No runtime Ascent, MORK or datafrog fallback" (`ascent-byods:113`);
  - datafrog is "a Datalog fixpoint comparator, not the production substrate"
    (`plans/xlog-knowledge-spine-v1.md:50`).

### C7. Who issues a selection? — OPEN (EXO-329), PROPOSED

**Sources:**
- EXO-329: DR owns positions, DT owns labels, and consumers hold neither.
- VERIFIED-IN-CODE `SelectionSource` (`dt64/src/lane.rs:110`): the source is the
  only issuer of a position list.
- `CubeLanes` has no `SelectionSource` (`xlog/src/cube/lend.rs`, per #2149 and
  #2147's notes).
- MEASURED #2149: inbound costs one `locate` per cell, not per distinct id.

**Proposed resolution:**
- DR issues selections, under its frame and the relation's row order.
- Keys are resolved once per distinct key per frame *and row order*
  (`CoordSpace`), at binding, never per cell or per row. A logged edge or a
  refold moves rows without moving the frame (C5). So a resolved position is
  reused only while the selection's `CoordSpace` is unchanged; after a
  row-order change it is resolved again. The socket already refuses a
  selection issued in another row order (`SelectionSpace`).

**Implications:** D-REGC-5 (`CubeLanes: SelectionSource`, issued by DR) and
D-REGC-7 (resolve once per distinct key).

### C8. The read shape over 512-byte rows: `LaneRef::Strided` — VERIFIED-IN-CODE (exists), OPEN (coverage)

**Sources:**
- QUOTED `plans/waben-fold-execution-loop-v1.md:1266`, at `lg@25988f3c`:
  "`LaneRef::{I32,U32,U64}` — no strided variant". That was true at that commit.
- VERIFIED-IN-CODE at `87d7b41` `lance-graph-mask-risc/src/ir.rs:27-50`:
  `LaneRef::Strided(StridedRef { bytes, first_offset, stride, records })`, "a
  `NodeRow` column stays inside its 512-byte rows". Quack reads it in place
  (`lance-graph-quack/src/lib.rs:256-268`, `:964-975`).
- Owner, 2026-11-20: "Quack is supposed to use SoA 512, as long as the Data can
  be mapped which is enforced by DT".

**Which predicates read a strided lane** (VERIFIED-IN-CODE, `mask-risc/src/ir.rs`
`Pred` / `Terminal`, `exec.rs:671-730`):

| Reads `Strided` | Lane-only (`I32` / `U32` / `U64`), no strided form |
|---|---|
| `EqU32Strided`, `NeU32Strided`, `MatchFacetStrided`, `MatchFacet16Strided`; terminal `MaskedStridedGroupSum` | `Gt/Lt/Ge/LeI32`, `Eq/NeI32`, `Eq/NeU32`, `MatchU32/U64`, `Range`, `EqU32Via`, `Gather`; terminals `MaskedSum/Min/Max/ArgMin/ArgMaxI32`, `GroupSum*`, `GroupReduce`, the scatter terminals |

`Count`, `Any`, `All` and `Keep` work on masks, so they are layout-free.

**Resolution:**
- DT's `ClassView` names each field's offset and carving.
- The source lends `LaneRef::Strided` views over the owner's 512-byte rows.
- The lane-only predicates a consumer's workload uses get strided forms, or that
  workload stays on lent `u32` lanes. Either way the gap is counted, not assumed.

**Implications:**
- **D-REGC-6:** strided forms for the comparisons, `Range`, code-set membership
  and the masked aggregates a recorded workload uses.
- The first count: zammad-rs's 512 calls, by predicate.
- Where the presence bit lives for a strided lane is also open: a plane per
  owner, or bits in the value region.

### C9. A consumer's own store against the one substrate — OPEN

**Sources:**
- Owner, 2026-11-20 (C8, §0 item 4).
- VERIFIED-IN-CODE zammad-rs `quack-shadow/src/lib.rs:74-300`: `Vec<u32>` lanes
  and `Vec<u64>` validity per column, owned by `Catalog`, with no mailbox owner.
- Everything zammad-rs measured (the census, #95's PostgreSQL gate) is therefore
  measured on an interim store.

**Resolution:**
- The census and the PostgreSQL gate are the oracle for moving the store onto
  owner-held 512-byte rows lent as `Strided`.
- The same counts and the same PostgreSQL answers must hold before and after.

**Implications:** D-REGC-11, in the consumer. It is blocked on D-REGC-6 for the
predicates the workload uses.

## 2. What is settled, at a glance

| | State | Where |
|---|---|---|
| Row as `identity[index].key : .value`, join last | CURRENT-CONTRACT | EXO-332 |
| Absent = −0 for `$null` / NaN / `''` / missing / deleted | RECORDED (owner words) | C3 |
| Missing key → no position; dead position refused at read | TEST-PINNED in zammad-rs; PROPOSED for DR | C4 |
| Coordinates bijective, relations bags | MEASURED | C5 |
| Join = aligned pass over shared coordinates; per-row key reads are a fallback | CURRENT-CONTRACT (direction) | C6 |
| egglog / DialEgg / datafrog / Ascent off the execution path | QUOTED, consistent | C6 |
| `LaneRef::Strided` exists; most predicates lack a strided form | VERIFIED-IN-CODE | C8 |
| Schema store / Decision J | **owner decision needed** | C1 |
| Lens-only read (read by reference, no view after bind) | VERIFIED-IN-CODE, in PR: `LaneBinding::check_bound` + `*_bound` borrows, `tests/bound_read.rs`; additive, `check(view)` unchanged | C1, C2 |
| Delete writes −0 | PROPOSED (EXO-337 / EXO-294) | C3 |
| Selection issuer, outer-join side in a selection | OPEN (EXO-329), PROPOSED | C4, C7 |

## 3. How a future session uses this document

1. Before adding a check, a revision, a binding step or a store, find its row
   here. If it is PROPOSED or OPEN, do not build against the proposed side
   without the decision.
2. Before answering a missing, empty or deleted value, follow C3 and C4.
3. Before writing a join, follow C6. If the classes do not share coordinates, the
   fix is at intake, not a new join form.
4. Before porting a store onto 512-byte rows, count the predicates the workload
   uses against the C8 table.
5. When a decision lands, change the row's state here in the same PR. Do not add
   a second document.

## 4. The polyfill API (code, 2026-11-21)

**Contract, zero-dep:** `lance_graph_contract::pg_row`.
- `PgType` holds the admitted types: boolean, smallint, integer, bigint, text,
  varchar(n), timestamp(p) without time zone, date. `PgType::parse` refuses
  every other type: numeric, float, time zones, arrays, json. `gdesc` prints
  the `\gdesc` name.
- `LaneEncoding` is how a `u32` lane holds a value: `Bool`, `Unsigned`,
  `Signed`, `EpochSeconds`, `EpochDays`, `TextCode`.
- `AbsentAs` spells −0 as either `Null` or `EmptyText` (C3, D-REGC-10).
- `PgColumn::admit` is checked once per column. It refuses an encoding that
  cannot hold the type, and `''` for a non-text column.
- `is_absent` is presence 0 and value 0 (C3).
- `write_cell` is the one spelling function. Its unit tests pin −0, `''`,
  leap years and centuries.

**Reader, for consumers:** `crates/lance-graph-pg-row`. Each read uses one
DT64 check per table.
- `Reader::open` and `Reader::open_selected` wrap `borrow_all_present` and
  `borrow_all_selected_present`, then take the live plane through
  `LiveSource`. They admit U32 and I32 lanes. Strided and 64-bit lanes are
  refused (C8).
- `Reader::keep_where` and `count_where` take a Quack `Filter` over the
  reader's columns (`Col(k)`):
  1. Each column's validity (presence ∨ value≠0) is computed as a mask-risc
     fold.
  2. `Filter::sql_where` makes the filter three-valued.
  3. The filter is ANDed with the live plane, lowered by Quack and executed
     by mask-risc.

  A plane, a foreign lane or a semijoin in the filter is refused.
- `Reader::rows_at` reads an issued selection: a bag, in list order.
  `Reader::rows_in` reads a kept mask, ascending. Both refuse a dead or
  out-of-range position.
- `read_bag` is the virtual tuple bag (C6). It takes one position list per
  side, aligned by index, and an optional present plane per side. At a 0 bit
  that side reads −0 without reading a position.

**Gate:** `tests/pg_gate.rs` runs every case against a model oracle built
from logical rows, and against PostgreSQL 16.15 when a server answers. It
fails under `PG_ROW_REQUIRE_PG=1` if no server answers. MEASURED 2026-11-21,
4 of 4 cases equal:

| Case | Rows | What it covers |
|---|---|---|
| `SELECT *` | 276 live of 300 | 24 deleted rows keep their values |
| K3 `WHERE` | 139 | includes rows where the filter is UNKNOWN |
| issued bag | 82 | with repeats |
| `LEFT JOIN` | 276 | 130 missing sides |

The fixture holds 145 `''` notes, 29 NULL titles, 64 present `''` titles, and
negative integers. Disable runs, each red:
- `disable-pg-row-presence`: 408 rows differ;
- `disable-pg-row-live`: WHERE keeps 15 deleted rows;
- `disable-pg-row-bag-present`: 260 join rows differ.

**Not covered:**
- The validity fold's "presence 0, value ≠ 0" arm has no fixture row. The
  fixture writes canonical −0 only.
- `ORDER BY`.
- Aggregates other than `COUNT(*)`. They stay Quack's `lower_sum` /
  `lower_group_by`.
- The wire protocol.
- Who issues join positions (C7) and where the outer side's plane comes from
  (C4). The gate's test stands in for both.

The consumer facade over all of this (data as config, class as config):
`consumer-surface-data-and-class-as-config.md`.

Row order (ordinal records against Morton cells of a DR tile), and why a
reader refuses a mixed order instead of converting it: `morton-vs-ordinal-rows.md`.
