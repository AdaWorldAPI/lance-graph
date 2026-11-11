# D-DRDT-CLASS-GOVERNANCE-1 — Snapshot versioning census (DR/DT64)

Status: census, docs only. Read at main 238b6d1e4 (2026-11-11).

Question: what is versioned, replayable and immutable today in the DR/DT64 stores, and what is not.

Paths are under `crates/` unless stated. "dt64" is `lance-graph-dt64`; "dt64-lance" is `lance-graph-dt64-lance`. Line numbers were read in the source, not taken from a summary.

## 1. What each piece does

### Route log: `RouteVersion`, `VersionClock`, `RouteLog`

- `RouteVersion(u64)` is a position in an append-only log. `RouteVersion(0)` is the empty log (`dr.rs:153-156`).
- `VersionClock` is one shared `Arc<AtomicU64>` counter. It only increases. `tick()` issues the next version (`dr.rs:166-188`).
- `RouteLog` keeps `BTreeMap<u16, Vec<(RouteVersion, Option<u32>)>>`: per position, the events in version order. `None` is a retract (`dr.rs:209-215`).
- `route` checks the position against the identity reservation, then pushes an event and ticks the clock (`dr.rs:255-263`, `285-292`). `retract` pushes `None` (`dr.rs:281-283`).
- `routed_at(version, position)` returns the class at that version by binary search over the events (`dr.rs:296-314`). A version past the clock is `FutureVersion` (`dr.rs:301-307`).
- `claimed(class, extent, version)` counts the identity tiles holding a position routed to the class at that version (`dr.rs:319-335`).
- `BindStore` owns a `RouteLog` (`lib.rs:777`). Its `route`, `retract`, `routed_at` and `claimed` delegate to it (`lib.rs:1018-1070`, `1052-1058`).
- `RouteLog::clone` forks the clock (`dr.rs:217-226`). `BindStore::clone` clones the log and mints a new identity (`lib.rs:782-791`).

### `SchemaRev` and `SigRev`

- `SchemaRev(u32)` and `SigRev(u32)` are plain numbers (`lib.rs:117-122`).
- A class has one `ClassSchema` in the store, with one `schema_rev` (`lib.rs:254-258`, `273-285`).
- `resolve` refuses any other revision: `SchemaRevisionMismatch` (`lib.rs:1227-1233`).
- Registering a different entry for an already bound classid is `BehaviourImmutable` (`lib.rs:904-906`). So the store cannot hold rev 1 and rev 2 of one classid. An override is a new derived classid plus a re-route (`lib.rs:32-40` module doc, `824-841`).
- `SigRev` is on `MethodBinding.sig_rev` and in the `MethodRef`. `admit` refuses a mismatch (`lib.rs:1192-1199`).

### `ClassView::fields_at` and `lens_fields_at`

- Both are on the `ClassView` trait in `lance-graph-contract/src/class_view.rs` (`1105-1108`, `1115-1118`).
- Both default to `None`. The doc says the source that owns a class's history answers them (`class_view.rs:1089-1104`).
- dt64 asks for the historical basis first, then falls back to the current basis (`lib.rs:1268-1287`).
- At registration the fallback basis is the caller's claim for that revision. It becomes the anchor and is not proved (`lib.rs:1259-1262`, `331-342`).
- So history of the field basis lives in the source, not in dt64. Whether any implementation of `ClassView` in the workspace keeps it: UNKNOWN (not searched).

### `FrameStore`, `Frame`, `FrameRevisions`

- `FrameStore` holds an identity, the current `Arc<FrameRevisions>`, a clock and the version of the current seal (`dr.rs:378-385`). It keeps no list of past seals.
- `FrameRevisions { source, relation_map, coordinates }` (`dr.rs:354-361`).
- `Frame` carries `Weak` pointers to the store and to the seal, a copy of the revisions it was sealed at, and a `version` (`dr.rs:485-491`).
- `issued(&frame)` returns the frame's version if this store issued it, even when the frame is no longer current (`dr.rs:441-446`).
- `check(&frame)` is `Ok` only for the current seal. An older frame is `SourceChanged`, `MapChanged`, `CoordinatesChanged` or `FrameResealed` (`dr.rs:462-479`).
- `advance` ticks the clock and moves to new revisions (`dr.rs:426-430`).

### `MapTable` (global to local maps)

- `MapTable` holds a `FrameStore` and a `BTreeMap<TileId, TileLog>` (`dr.rs:871-874`).
- `TileLog` keeps generations plus events `(version, Some((generation, len)))` or `None` (`dr.rs:713-757`). `at(version)` returns the layout in force then.
- `map(frame, tile)` reads at the frame's version through `issued` (`dr.rs:1057-1060`). A frame of another table is `Stale(ForeignFrame)`.
- `diagnose_at(version, tile)` reads at a bare number and returns a `MapDiagnostic`, which cannot be passed to an executor (`dr.rs:1062-1088`).
- Qualifier: an extension (new map keeps every old position and appends) replaces the stored generation with its superset: `*slot = m` (`dr.rs:727-733`). The doc says nothing is updated in place (`dr.rs:859-863`). Old versions still answer, because each event stores a length, and the old prefix is identical.
- `claim` and `advance_at` let a caller take a version first and apply the advance later (`dr.rs:981-1014`). `advance_at` refuses a version at or before the last applied one (`NotAfterLast`) and one the clock has not issued (`NotIssued`).
- `FrameRevisions.coordinates` is an FNV-1a over the current tiles. The doc says it cannot key history (`dr.rs:944-962`, `864-869`).

### `CodeTable`

- Per `CodeSpaceId`, a `Vec<CodeGeneration>` each stamped with `at: RouteVersion` (`dr.rs:1229-1234`, `1247-1250`).
- `advance` advances the bound `FrameStore` once and pushes a generation per listed space (`dr.rs:1271-1308`).
- Reads are keyed by a `Frame` and pick the generation in force at the frame's version (`dr.rs:1337-1350`).

### `LaneBinding::check` and `bind_lanes`

- `LaneBinding` stores class, rev, route, frame, scope, selection, fields, lane addresses and a `BasisDigest` (`lane.rs:153-165`).
- `bind_lanes` calls `frames.check(req.frame)` (`lane.rs:179`). So the frame must be the current seal.
- It reads the class at `req.route.version` (`lane.rs:181`) and then requires that the position still routes to that class at the store's current version (`lane.rs:186`, `228-240`).
- `check` repeats: frame current (`lane.rs:315`), route current (`lane.rs:316`), basis digest equal (`lane.rs:324-327`), each lane address equal to what the source maps now (`lane.rs:328-346`).
- It does not re-hash the basis. A same-count reorder in the view after the bind is caught only if the source maps lanes by predicate (`lane.rs:303-306`, `317-323`).

### Digests

- `BasisDigest` is a 128-bit FNV-1a over a length-prefixed list of `predicate_iri`s in position order. Domain tag `dt64.basis.v1` (`lib.rs:287-329`). It is `pub(crate)`.
- `binding_digest` is FNV-1a 128 over class, basis digest, method id, sig rev, ON, reads, effects, population, truth and demand. Domain tag `dt64.binding.v1`. The guard is not in it (`lib.rs:400-468`).
- `ClassEntry` stores the basis digest and one binding digest per method (`lib.rs:273-285`).
- The doc says both digests detect change and do not authenticate (`lib.rs:300-302`, `408-410`).

### What `dt64-lance` persists

- `MapLog` owns a `MapTable` and submits each advance as one cycle through `LanceCycleWriter`. Cycle id equals route version (`dt64-lance/src/lib.rs:14-22`, `279-308`).
- Per advance it writes a marker cast, then one 512-byte cast per 64 global ids of each changed tile. The marker is `kind` at byte 0, then little-endian `version`, `source`, `relation_map` at bytes 8..32 (`lib.rs:400-445`). A removal is one zeroed cast at part 7 (`lib.rs:428-431`).
- The cycle store schema is 13 columns: `kind`, `cycle`, `base_version`, `batch_hash`, `stream_position`, `owner`, `row`, five nullable `move_*` columns, and a nullable `payload` `FixedSizeBinary(512)` (`lance-graph/src/graph/cycle_sink.rs:127-147`).
- This crate writes no dataset itself (`dt64-lance/src/lib.rs:14-17`). The writer does.
- `restore` scans all landings, checks every landing's owner, reads each cycle's image and replays: `scan_sealed(None)` then `scan_image(cycle)` per cycle, then `replay` (`dt64-lance/src/lib.rs:513-547`; `cycle_sink.rs:1064-1150`, `1200-1235`). It rebuilds the table to the latest sealed cycle. It does not open an old Lance dataset version (`ds.scan()` on the current handle).
- `replay` calls `MapTable::advance_at` at each stored version and ticks the shared clock up to it (`dt64-lance/src/lib.rs:587-589`, `618-626`).
- Not carried: frames. Frames sealed before a restore are foreign to the restored table. Issuing frames at a past version after restore is "a DR owner decision this crate does not make (OPEN)" (`dt64-lance/src/lib.rs:37-41`).
- Failure modes are named: gap versions, `Poisoned`, `OutcomeUnknown`, `Refused` (`dt64-lance/src/lib.rs:84-134`).

## 2. Table

| thing | versioned? | persisted? | replayable at an old version? | evidence |
|---|---|---|---|---|
| `RouteVersion` / `VersionClock` | yes, one counter | counter: no. `restore` ticks a given clock up to the stored versions | n/a | `dr.rs:156`, `167`; `dt64-lance/src/lib.rs:587-589` |
| `RouteLog` (route, retract) | yes, event per position | no. `dt64-lance` does not use it | yes in process: `routed_at`, `claimed` | `dr.rs:209-335`; grep for `RouteLog` in `dt64-lance`: only a doc comment at `lib.rs:210` |
| `SchemaRev` | one value per classid in a store; no history in the store | no | no. Another rev is refused | `lib.rs:254-258`, `904-906`, `1227-1233` |
| `SigRev` | one value per method | no | no | `lib.rs:215-218`, `1192-1199` |
| `ClassView::fields_at` / `lens_fields_at` | in the source, if it implements them. Default `None` | UNKNOWN | UNKNOWN per implementation | `class_view.rs:1105-1118` |
| `BindStore` classes and methods | add-only. Never changed or removed | no | no snapshot. Nothing is dated | `lib.rs:26-30`, `772-780`; no `serde` in dt64 |
| `BindStore` algebra lenses | add-only (`AlgebraImmutable`) | no | no | `lib.rs:917-939` |
| `FrameStore` / `FrameRevisions` | current seal plus its version only | no | `issued` gives an old frame's version; `check` refuses it | `dr.rs:379-385`, `441-479` |
| `Frame` | carries its version | no. Foreign after restore | reads history through `MapTable::map` and `CodeTable` | `dr.rs:485-503`; `dt64-lance/src/lib.rs:37-41` |
| `MapTable` | yes, tile events by version | yes via `MapLog` | yes: `map(frame)`, `diagnose_at(version)` | `dr.rs:871-1100`; `dt64-lance/src/lib.rs` |
| `CodeTable` | yes, generation per `RouteVersion` | no (not referenced in `dt64-lance`) | yes for a frame of the bound store, in process | `dr.rs:1229-1350` |
| `LaneBinding` | stores its route version and frame | no | no. `check` and `bind_lanes` need the current frame and current route | `lane.rs:179`, `186`, `307-349` |
| `BasisDigest`, binding digest | stored per class, per method | no | no | `lib.rs:273-285`, `304`, `411` |
| `MethodRef` / `Stamp` | `sig_rev` and binding digest inside | no | not outside the issuing store instance | `lib.rs:353-398`, `1161-1212` |

## 3. Not replayable today

Each item below was checked in the source.

1. **`Stamp.issuer` is a `Weak<()>`** (`lib.rs:371-374`). `check_issuer` compares `Weak::as_ptr` to `Arc::as_ptr` of the store's identity (`lib.rs:1161-1167`). A `MethodRef` therefore cannot be validated in another process, after a restart, or by a clone of the store (`lib.rs:767-770`, `782-791`: a clone is another store). `Stamp::binding()` (`lib.rs:380`) is readable, but `admit` never trusts it over the stored digest (`lib.rs:1207-1210`).
2. **`BindStore` is not persisted.** The crate docs say "no persistence path" (`lib.rs:26-30`). dt64 has no `serde` hit. `dt64-lance` references no `BindStore`, `SchemaRev`, `SigRev` or `MethodBinding` (grep: only a `RouteLog` word in a doc comment). Classes, methods, lenses, the route log and both digests live only in process memory. Whether another crate persists a `BindStore` or a view of it: UNKNOWN. `lance-graph-codegraph/src/graph/publication.rs` holds one in a lab struct next to a lab `WalSink` and persists route intents, not the store (`publication.rs:1-9`, `157-163`).
3. **Class definitions have no version axis in the store.** Because rev change is refused (`lib.rs:904-906`, `1227-1233`), "the class at rev N on date D" cannot be asked of `BindStore`. It can only be asked of the source's `ClassView` history, and only if that source implements `fields_at`.
4. **A lane binding is valid only against the present.** `bind_lanes` and `check` refuse a stale frame (`lane.rs:179`, `315`; `dr.rs:462-479`) and a route that has moved (`lane.rs:186`, `228-240`). A binding made at an old route version can be re-made only if the position still routes to the same class now.
5. **`BindStore` routes on its own clock.** `BindStore` holds a default `RouteLog` (`lib.rs:777`). grep for `with_clock` and `.clock(` in `dt64/src/lib.rs` returns nothing. So `BindStore::route` versions are not on the same clock as a `MapTable` unless the clock is shared another way. UNKNOWN whether any caller shares it. The shared clock is available on `RouteLog::with_clock` and `MapTable::with_clock` (`dr.rs:235`, `885`).
6. **Frames do not survive a restore** (`dt64-lance/src/lib.rs:37-41`). Past layouts stay readable only by number, through `diagnose_at`, which cannot feed an executor (`dr.rs:1062-1088`).
7. **The restored table's clock is caller-supplied.** `restore` ticks the given clock up to the stored versions (`dt64-lance/src/lib.rs:587-589`, `618-620`). Nothing in the store records which route log or `BindStore` that clock belonged to.
8. **Gaps are real.** A claimed version whose cycle fails to seal is a hole in the table and the store (`dt64-lance/src/lib.rs:18-22`). A reader asking `diagnose_at` for a gap version gets the layout of the last event before it (`dr.rs:748-756`).
9. **Both digests are non-cryptographic.** FNV-1a 128. The doc says a source built to collide is not excluded (`lib.rs:296-302`).
10. **The basis anchor is a claim.** At registration with a view that has no history, the current basis is taken as the declared revision's and cannot be proved (`lib.rs:58-64`, `1259-1262`).
11. **`LaneBinding::check` does not re-hash the basis** (`lane.rs:303-306`).

## 4. Open items named in the files read

- EXO-46 (counted global-to-local membrane for production) is named at `dr.rs:569`.
- EXO-55 (several thoughts per cycle, one version) is named at `thought-git-probe/src/receipt.rs:23`.
- EXO-225 and EXO-140 do not appear in the files read for this census. Their content is not stated here.

## 5. Not checked

- Whether any `ClassView` implementation keeps `fields_at` history.
- Whether any caller joins a `BindStore` route log and a `MapTable` on one clock.
- Whether another crate persists `BindStore` content.
- The commit hash in the Status line was given in the brief. The working tree is not a git repository here, so it was not checked.
