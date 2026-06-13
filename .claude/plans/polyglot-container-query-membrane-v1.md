# Polyglot Container Query Membrane — SurrealQL AST + DataFusion UDF + Cypher over one HHTL address space (v1)

> **ERRATA (2026-06-13, post-#490):** Ratified as **research-only / superseded in spirit**. The "membrane / strategy-registry" framing was superseded in discussion by the self-describing-key convergence (class-in-key makes the cold path already a graph; no membrane needed) and by the GUID-canon PR #482 that crystallised the operator's canon. The verified file:line surface of lance-graph + the surrealdb fork's AST/RecordId/kv-lance remains useful as a grounded inventory but is not a committed direction. Full diff resolution: `soa-migration-diff-resolution-2026-06-13.md`.

> **Status:** RESEARCH MAP + INTEGRATION PLAN. Grounded 2026-06-09 by two parallel
> repo sweeps (lance-graph + the surrealdb fork) with main-thread spot-verification
> of every load-bearing claim (one agent claim caught false and corrected, §2.4).
> **Branch:** `claude/nice-edison-g4rhhl`.
> **Companions:** `identity-architecture-exists-vs-needs-v1.md` (the address layer),
> `bindspace-singleton-to-mailbox-soa-v1.md` (the mailbox layer, PR #418),
> `.claude/handovers/2026-05-28-1200-...md` §2 (the SurrealDB-as-VIEW ruling),
> `.claude/surreal/` POC corpus (12 tasks; framing partially superseded, see ruling).

## 1. Thesis — the Christmas tree

The ontology is a Christmas tree: the **registry bijection** (`entity_type ↔
NiblePath`, landed in D-IDENTITY-2) is the always-resident skeleton, and **rows
are decorations that are NOT stored in the tree** — they *materialize at read
time at their HHTL address* from whichever tier owns them (hot mailbox snapshot,
Lance cold store, consumer store via `EntityKey`). Three query dialects —
**Neo4j/Cypher**, **SurrealQL (DML + DDL, via the fork's typed AST)**, and
**DataFusion SQL/UDF** — resolve classes against the SAME registry catalog and
addresses against the SAME order-preserving key codec. Consequence: *parsing a
mailbox is indistinguishable from scanning a cold table*. The mailbox is just
another tier behind the same 5-method read contract — "a normal cold path."

Standing ruling respected throughout (handover 2026-05-28 §2, `E-RUBICON-RACTOR`):
**LanceDB is the leading store; SurrealDB is a view/frontend, never the store.**
An AST adapter is a query surface — it strengthens that ruling rather than
bending it. The embedded SurrealDB-on-kv-lance engine remains the OPTIONAL leg
(kanban view), explicitly off the critical path.

## 2. Grounded inventory — what EXISTS (verified file:line)

### 2.1 lance-graph: container, mailbox, addresses, frontends

| Surface | Where | Status |
|---|---|---|
| `Container = [u64; 256]` (2 KB); `CogRecord { meta, content }` (4 KB); `ContentGeometry` (Bitpacked16K / DenseF32 / TripleSPO / EdgePacked) | contract `container.rs:14-67` | **[G]** |
| `MailboxSoaView` — zero-copy column borrows: `energy() -> &[f32]`, `edges_raw() -> &[u64]`, `meta_raw() -> &[u32]`, `entity_type() -> &[u16]`, + `phase() -> KanbanColumn`, `current_cycle()` | contract `soa_view.rs:40-70` | **[G]** trait |
| `MailboxSoaOwner::try_advance_phase` (Rubicon gate = the snapshot/transaction boundary) | contract `soa_view.rs` | **[G]** trait |
| `SoaEnvelope` — LE geometry: `as_le_bytes()`, `row_le(row)`, `column_le(row,col)`, `verify_layout()`, `cycle()` | contract `soa_envelope.rs:143-252` | **[G]** trait / **[H] ZERO real impls** (§2.4) |
| `MailboxSoA<const N: usize>` concrete columns (energy/edges/qualia/meta/entity_type…) | cognitive-shader-driver `mailbox_soa.rs:43` | **[G]** struct; implements NEITHER trait yet |
| `NiblePath` algebra: `child = (parent.path << 4) \| nibble`, low-aligned, `MAX_DEPTH=16`; `is_ancestor_of` = one shift-compare | contract `hhtl.rs:47-101, 176-183` | **[G]** |
| Registry bijection: `path_by_type`/`type_by_path`, `register_class_path` (conflict-rejecting), `niblepath_of`, `entity_type_of`, `rows_with_entity_type`; dedup-by-URI mint | ontology `registry.rs:64-72, 343-418, 565-579` | **[G]** (D-IDENTITY-2) |
| `NodeGuid` UUIDv8 octets: `namespace u8 \| entity_type u16 \| kind u8 \| niblepath_prefix u16 (≤4 nibbles, routing cache ONLY) \| depth \| shape_hash 22b \| local u24 \| layout_version` | contract `identity.rs:68-81, 159-206` | **[G]** (D-IDENTITY-1) |
| Polyglot strategy registry: `cypher_parse` / `gql_parse` / `gremlin_parse` / `sparql_parse` / `arena_ir` all registered as boxed strategies; selector scores by dialect | planner `strategy/mod.rs:50-60`, `strategy/*.rs` | **[G]** — the frontend slot shape |
| DataFusion UDF registration + custom physical ops (`CamPqScanOp`, `CollapseOp`) | core `datafusion_planner/udf.rs`; planner `physical/` | **[G]** |
| `cypher_bridge.rs` in shader-driver (Cypher already reaches the cognitive side) | cognitive-shader-driver | **[G]** |
| Hot==cold intent stated: mailbox columns "read identically whether in RAM … or via Lance snapshot" | `docs/SUBSTRATE-ENDGAME-RUNTIME-VIEW.md` §1.1 | **[G]** doc intent — this plan wires it |

### 2.2 surrealdb fork (3.1.0-alpha): AST, RecordId, key encoding, kv-lance

| Surface | Where | Status |
|---|---|---|
| **Typed AST crate** `surrealdb-ast` — `TopLevelExpr`, `Expr`, `Select`, `Create`/`Update`/`Delete`/`Insert`, `DefineNamespace/Database/Table/Field/Index/…`, `RecordId`, `RecordIdKey`, `RecordIdKeyRange`; visitor infra (`visit/`, `mac.rs`) | fork `surrealdb/ast/src/lib.rs:37-160` | **[G]** — public, programmatically constructible |
| Dedicated parser crate (recursive descent) | fork `surrealdb/parser/src/parse/mod.rs:66-69,736` | **[G]** |
| `RecordIdKey::{Number, String, Uuid, **Array**, Object, **Range(Box<RecordIdKeyRange>)**}` | fork `types/src/value/record_id/key.rs:20-55 (Array at :28)` | **[G]** |
| `RecordIdKeyRange { start: Bound<RecordIdKey>, end: Bound<RecordIdKey> }` | fork `types/.../range.rs:17-22` | **[G]** |
| **Order-preserving KV key encoding** via `storekey` — layout `/*{ns_id}*{db_id}*{tb}*{record_id_key}`; lexicographic byte order == logical order, arrays included | fork `core/src/key/record.rs:5-26` | **[G]** — the load-bearing property (codec-level); composed-key proof test is OURS to write (P0) |
| **Record-range → KV byte-range scan**: `range_start_key`/`range_end_key` → `txn.stream_keys_vals(beg..end, …)` | fork `core/src/exec/operators/scan/pipeline.rs:211-238 (:223), 367-412`; `scan/record_id.rs:305-331` | **[G]** — subtree-as-range has a native execution path |
| **kv-lance backend: FULLY IMPLEMENTED in-tree** — `get` :646, `keys` :824, `keysr` :836, `scan` :848, `scanr`, writes, savepoints; MVCC via Lance versions, `Timeline` time-travel, background optimizer; ~6k lines of tests; SDK `Surreal::new::<Lance>(path)` | fork `core/src/kvs/lance/mod.rs`; `src/engine/local/mod.rs:329`; features `core/Cargo.toml:27`, sdk `Cargo.toml:27` | **[G]** — supersedes `surreal_container` BLOCKED(C/D) |
| `Transactable` contract; read-only subset = `get / keys / keysr / scan / scanr` (+ `kind/closed/writeable`) | fork `core/src/kvs/api.rs:76+` | **[G]** — the tier contract M2 targets |
| **C16b DDL builders** (`op-codegen-bridge`): `new_for_ddl()` + `with_*` setters on `catalog::{Table,Field,Index}Definition`, render via `ToSql` **without a database**; downstream consumer `op-surreal-ast` (openproject-nexgen-rs); C16c adds `From<op_surreal_ast::*> for catalog::*` | fork `.claude/op-codegen-bridge/README.md`; `core/src/catalog/{table.rs, schema/field.rs, schema/index.rs}` | **[G]** active initiative — the DDL exchange format M3 reuses |
| SDK `.query()` accepts **strings only** (`Vec<Cow<str>>`) | fork `src/method/query.rs:28-32` | **[G]** constraint → OQ-PG1 |

### 2.3 lance-graph surreal prior art (and its standing correction)

- `crates/surreal_container/` — BLOCKED skeleton (12 task stubs). Its BLOCKED(C/D)
  reasons are **stale**: the fork dep coordinates now exist locally and `kv-lance`
  is in-tree (§2.2). Remains **optional** per the ruling (D-MBX-6 note: "NOT on
  the critical path").
- `.claude/surreal/` 12-task POC + `RECONCILIATION` + `cognitive-substrate.md` —
  partially superseded framing ("SurrealDB = Zone-2 cold store" → "LanceDB
  leading, SurrealDB a view"); supersedure annotation still pending
  (`E-SURREAL-POC-UNANNOTATED-SUPERSEDURE`).

### 2.4 Sweep-error correction (recorded for provenance)

The lance-graph sweep agent claimed `MailboxSoA<N>` implements `SoaEnvelope`.
**Spot-grep proves it does not** — the only `impl SoaEnvelope` in the workspace is
the in-test `TestEnvelope` (`soa_envelope.rs:266`), and `mailbox_soa.rs` contains
no trait impls for the type at all. The identity plan's gap **N3 ("SoaEnvelope:
zero impls") stands LIVE** and appears below as D-PG-2. (Method: never let an
agent claim into a plan without a main-thread grep.)

## 3. The mapping — five moves

### M1 — One address codec under all three dialects [CONJECTURE until D-PG-1 proof]

Define the **sortable HHTL address**: `addr64 = path << (4 · (16 − depth))`
(left-align the low-aligned `NiblePath` into the u64). Then for any branch `p`
at depth `d`, **every descendant at every deeper depth falls in ONE contiguous
range** `[ p·16^(16−d) , (p+1)·16^(16−d) )` — and `is_ancestor_of` (the
hhtl.rs:176 shift-compare) is exactly range-containment under this codec.

Per-dialect, the SAME range is:

| Dialect | The subtree read |
|---|---|
| SurrealQL | `SELECT * FROM node:[⟨addr_lo⟩]..[⟨addr_hi⟩]` → `RecordIdKeyRange` → `stream_keys_vals(beg..end)` (native, §2.2) |
| DataFusion SQL/UDF | `WHERE addr64 >= lo AND addr64 < hi` on the stored column → partition/row-group pruning; helper UDFs `hhtl_subtree(addr, depth)`, `guid_class(guid)` in `udf.rs` |
| Cypher | label/class scan via registry (`entity_type → NiblePath → range`); blasgraph HHTL basin walk uses the same prefix arithmetic it already has |

RecordId form for rows: `node:[addr64, local]` (`RecordIdKey::Array` — storekey
keeps array order). **Honest scope:** storekey's element-wise order preservation
is [G] at codec level, but the COMPOSED key (`u64` + `u32` array under
surrealdb's value ordering, negative/width edge cases) gets a property test
before anything builds on it — that test IS deliverable D-PG-1, the falsifiable
brick. **NodeGuid caveat:** the GUID carries only a 4-nibble routing prefix
(identity.rs octets 4-6); scans deeper than 4 resolve the FULL path through the
registry bijection (`niblepath_of`) — the tree is resident, so this is one
HashMap hop, not I/O.

### M2 — The mailbox is a tier, not an engine [DESIGN]

1. Implement `SoaEnvelope` for `MailboxSoA<N>` (D-PG-2 = identity-plan N3): the
   columns already exist; the impl is descriptor table + LE byte views.
2. A **read-only adapter** implementing the 5-method `Transactable` read subset
   (`get/keys/keysr/scan/scanr`) over a **phase-pinned snapshot**: Rubicon
   `try_advance_phase` is the transaction boundary; `MailboxSoaView`'s read
   airgap satisfies the no-`&mut`-during-computation data-flow rule by
   construction.
3. The query layer above (any dialect) cannot tell hot from cold. Acceptance is
   a **differential test**: same range query against (a) the mailbox tier and
   (b) the same rows persisted in Lance — byte-identical results (D-PG-3).

This is `SUBSTRATE-ENDGAME` §1.1's "read identically in RAM or via Lance
snapshot" sentence turned into a contract with a test.

### M3 — DDL declares the tree; the registry IS the catalog [DESIGN]

- `DEFINE TABLE <class> SCHEMAFULL` ⇄ registry append (dedup-by-URI mint —
  same URI never re-mints, D-IDENTITY-2) + `register_class_path` (hang the
  ornament hook on the tree).
- `DEFINE FIELD <prop> ON <class>` ⇄ `MappingRow` property → `FieldMask` bit
  (parent-OR-delta inheritance already exists).
- `DEFINE INDEX` ⇄ bijection entry / Lance index declaration.
- **Exchange format = the C16b builders**: registry → `TableDefinition`
  (`new_for_ddl().with_…`) → `to_sql()` → SurrealQL text → fork parser →
  typed AST → registry. Round-trip = the schema's `roundtrip_eq` analogue.
- Cypher `CREATE (:Label)` and DataFusion `CREATE EXTERNAL TABLE` resolve to
  the SAME mint — three DDL dialects, one catalog, zero duplicate ids (the
  dedup mint guarantees it).

### M4 — SurrealQL as frontend #5 [DESIGN]

`strategy/surrealql_parse.rs`: typed `surrealdb-ast` statements → `ArenaIR`,
registered exactly like `sparql_parse` (one `Box::new` line in
`strategy/mod.rs:57-60`, one selector scoring rule). Subset order: SELECT
point-get → SELECT record-range (M1) → graph-step (`->edge->` onto
episodic/causal edges) → DDL (M3). Cypher is already frontend #1; GQL/Gremlin/
SPARQL prove the slot shape; DataFusion UDFs make the same primitives available
to plain SQL. **No new REST endpoint, no new service** — this is a parser
strategy feeding the existing IR, per `lab-vs-canonical-surface.md`.

### M5 — Embedded SurrealDB view [OPTIONAL — per the ruling]

Unblock `surreal_container` (deps are now real, §2.2/§2.3) ONLY for the Rubicon
kanban view over leading LanceDB (the #418 framing). Execution seam is open
(OQ-PG1: SDK `.query()` is string-only → render via `ToSql`, or call core
`Datastore` directly). Explicitly not on the critical path; D-PG-1..5 do not
depend on it.

## 4. Deliverables

| D-id | What | Reuses [G] | Adds [H] | Gate |
|---|---|---|---|---|
| **D-PG-1** | `addr64` codec + **order-preservation property test** (random NiblePaths: byte-order of encoded keys ⇔ `is_ancestor_of` containment; composed `[addr64, local]` array form under storekey) | NiblePath, storekey | the codec fn + proptest | none — **first brick** |
| **D-PG-2** | `SoaEnvelope` impl for `MailboxSoA<N>` (= identity N3, confirmed live §2.4) + LE parity vs Lance bytes | trait + struct + columns | the impl + `verify_layout` test | none |
| **D-PG-3** | Read-only mailbox `Transactable` adapter (5 methods over phase-pinned envelope snapshot) + **hot==cold differential test** | Transactable contract, Rubicon, D-PG-2 | the adapter | D-PG-1,2 |
| **D-PG-4** | `SurrealqlParse` strategy → ArenaIR (SELECT point/range subset) + selector rule + 3 golden queries | surrealdb-ast, ArenaIR, strategy registry | the strategy | D-PG-1 |
| **D-PG-5** | DDL ⇄ registry bridge (Define walker → mint/MappingRow; reverse render via C16b builders) | C16b, dedup mint, bijection | the walker + round-trip test | C16c upstream (`From` impls) |
| **D-PG-6** *(opt)* | `surreal_container` unblock → kanban view over LanceDB | kv-lance in-tree | dep wiring + view | ruling-compliant; OQ-PG1 |

Phases: **P0** = D-PG-1 alone (falsifiable; everything stands on it). **P1** =
D-PG-2+3 (the "mailbox is a normal cold path" claim becomes a passing test).
**P2** = D-PG-4 (three dialects live). **P3** = D-PG-5 (the tree is declared in
DDL). P4 = D-PG-6 if/when wanted.

## 5. Iron-rule + ruling compliance

- **I-VSA-IDENTITIES:** pure register pattern — addresses point to content;
  nothing is bundled or superposed. Test 0 (natural ids exist) passes by
  construction.
- **LanceDB-leading ruling:** SurrealQL here is a *dialect*; the engine leg is
  optional and view-only. No persistence moves to SurrealDB.
- **I-LEGACY-API-FEATURE-GATED:** the codec carries `layout_version` (NodeGuid
  octet 13 already reserves it); any future addr64 layout change versions
  through it.
- **Data-flow rule (no `&mut` during computation):** all query reads are `&self`
  on phase-pinned snapshots; mutation stays behind Rubicon + commit_event.
- **lab-vs-canonical-surface:** no new endpoints; frontends are parser
  strategies into the existing IR/bridge.

## 6. Open questions

- **OQ-PG1** — embedded-view execution seam: SDK `.query()` is string-only;
  choose `ToSql` render vs direct core `Datastore` call when D-PG-6 activates.
- **OQ-PG2** — store `addr64` denormalized as a row column (cheap pruning,
  16 B/row total with local) vs derive from registry at plan time (no
  duplication; one hop). Decide at D-PG-3 with a measurement, per Rule 7.
- **OQ-PG3** — `Bound` semantics: normalize SurrealQL inclusive/exclusive
  bounds to the half-open `[lo, hi)` convention of the subtree formula at
  D-PG-1 (property test covers both bound kinds).

## 7. Cross-refs

Identity plan (addresses; N3→D-PG-2), #418 mailbox plan + D-MBX-6, handover
2026-05-28 §2 (ruling), `.claude/surreal/` POC (fold-in pending), fork
`op-codegen-bridge` C16b/C16c, `SUBSTRATE-ENDGAME-RUNTIME-VIEW.md` §1.1,
`docs/CLUSTER_ASYMMETRY.md` (surreal-cluster as Raft provider — unrelated leg).

## 8. Addendum 2026-06-09 — left-prefix parsing + deterministic foveated tree construction (user direction)

**User framing, confirmed against `identity.rs:68-81`:** the GUID reads as
`classid-HHHH-HHHH-TTTT-LLLLLLIDENTI` — octets 0-3 class (`namespace|entity_type|kind`),
4-7 tree address (NiblePath prefix + depth), 8-9 shape, 10-15 local identity.
**The left half is plain order-preserving bytes** ⇒ Neo4j/Cypher label + subtree
patterns compile to byte-prefix/range predicates on a `FixedSizeBinary(16)` GUID
column — Lance zone-maps/scalar indexes serve them directly; quantized-vector
indexes (RaBitQ-style 1-bit codes / CAM-PQ / Binary16K Hamming) serve the
similarity leg on the SAME row. Structural predicate + similarity predicate =
two indexes, one container. Caveats (both already in §3 M1): namespace sorts
first (cross-namespace template scans = ≤256 ranges or one registry hop);
GUID carries ≤4 path nibbles (deeper scans resolve via registry / addr64).

**M6 — deterministic foveated tree construction [CONJECTURE until D-PG-7 test].**
NiblePath assignments need not be purely editorial (E-OGAR named curation the
only real cost): they can be COMPUTED by a deterministic hierarchical
partitioner over class fingerprints/co-occurrence — "deterministic Louvain" in
the user's phrase, with three mandatory properties:

1. **Deterministic** — canonical input ordering + stable tie-breaks (Leiden-with-
   canonical-order, or the already-in-tree deterministic divisive splitter:
   **ndarray CLAM**, 46 tests — preferred starting point over Louvain proper,
   which is node-order dependent in its classic form).
2. **16-way capacity-bounded** — one nibble per level; a basin subdivides only
   when it exceeds capacity/density θ ⇒ **foveation falls out**: depth grows
   where data is dense, stays shallow where sparse.
3. **Append-stable (the identity-stability requirement)** — clustering runs ONCE
   as bootstrap; thereafter new classes insert greedily under the nearest
   existing basin and **minted paths never move** (protobuf-field-number
   discipline, same as the entity_type mint). Full re-clustering on rebuild
   would re-address every GUID — forbidden. Tree layout changes version through
   `layout_version` (octet 13) per I-LEGACY-API-FEATURE-GATED.

Query-time twin: the ndarray cascade (Belichtungsmesser bands, early exit) and
bgz-tensor's HHTL cache (95% of pairs skipped) are the SAME foveation principle
applied at read time — build-time depth where dense, query-time attention where
relevant. One principle, both sides of the store.

**D-PG-7 (Queued):** deterministic tree-builder bootstrap — partition class
fingerprints (CLAM-style pole-split, 16-way, capacity θ) → NiblePath
assignments → `register_class_path` batch; property tests: (a) two runs over
permuted input yield byte-identical trees, (b) append of M new classes leaves
all prior paths unchanged, (c) depth distribution tracks density (foveation
witness). Gated on D-PG-1 (the codec the paths feed).
