# D-QSP-R — PostgreSQL parity roadmap over inverted tuples (v1)

**Status:** PROPOSED / DOCS ONLY, 2026-11-11. An umbrella roadmap, not a new
architecture. It orders work that already has owners: D-QSP
(`d-qsp-classview-quack-postgresql-parity-v1.md`), D-DYNBAG
(`d-dynbag-holographic-bags-v1.md`), EXO-114 (order contract), the DR / DT
binding work (`dr-widefieldmask-dt-classview-v1.md`, S2–S8) and DECISION
EXO-255. No code, ABI, contract or codebook change is authorized here.

## 1. The contract

> A SQL tuple is a contextual projection over bound identities and
> properties. Execution may preserve that projection virtually.
> Materialization happens when an operator or consumer requires it.

PostgreSQL defines the observable SQL behaviour. Inverted tuples
(`Tuple(i) = (A[i], B[i], C[i])` over one shared index), class schemas
(`ClassView`) and registered algebra (Quack / relalg / mask-RISC / `ndarray`)
provide the implementation. **Semantic parity comes first;** a performance
comparison follows each admitted capability, never precedes it.

**Scope boundary.** Mapping versioning (EXO-255) does not provide
PostgreSQL transaction or snapshot semantics. The roadmap starts with
read-only queries over controlled fixtures. Concurrent reads, isolation and
writes are separate parity workstreams (D-QSP §"Contract": "transaction
isolation/MVCC, DDL … are separate future strata").

## 2. Stages, mapped onto existing IDs

| Stage | Deliverable | Independent gate | Existing owner |
|---|---|---|---|
| 1. Class / schema binding | SQL columns resolve to class properties; context selects the projection; the mapping has an explicit version | same schema, values, missing-property behaviour and supported types as PostgreSQL | D-QSP-1; D-DYNBAG X4 / MISSING-2 (no Quack `Binder` backed by `BindStore` / `ClassView`); `MapTable` (D-DR-VMAP, #1923) |
| 2. Virtual scan and projection | read selected properties through the mapping without building an intermediate X × Y population | results equal; copied bytes and allocations measured separately | D-QSP-1; EXO-220 (source-owned planes) |
| 3. Selection | supported predicates lowered into registered algebra | parity for NULL, three-valued logic, boundaries, empty inputs | D-QSP-0 (slices 1–2 shipped, PG 16 parity measured: #1668, #1675, #1682) |
| 4. Bag operations | joins, projection, aggregation, DISTINCT with explicit multiplicity | duplicate-heavy fixtures, unmatched rows, NULL keys, many-to-many joins | D-QSP-2; D-DYNBAG §6–7 (ℕ-compose; cross-tile open); `LEFT` / `FULL` currently refused |
| 5. Ordering | explicit order contract; existing implicit order where it suffices; a prepared index where needed | full ordered sequence, ties, NULL placement, `LIMIT` / `OFFSET` | EXO-114; S3; D-DYNBAG §9.3 `ORDER BY` row |
| 6. Reuse and change | reusable mappings distinguished from content-dependent preparations | change values, keys, mappings and schema independently; verify correct reuse or rebuild | EXO-255; S3, S7, S8 |
| 7. Recursion and wider SQL | supported recursive and further constructs admitted one by one | termination, duplicate handling and results equal PostgreSQL | D-QSP-4; D-RFT-3 (datafrog); xlog `Cube::closure` |
| 8. Cross-runtime comparison | the same admitted workload through further implementations | equivalent semantics and workloads before any cost comparison | D-QSP-5; duckdb plan §A.9 |

Each stage keeps D-QSP's independent verdicts: semantic correctness, binding
legality and provenance, performance, memory, cost of mutation. A pass in
one never waives a fail in another.

## 3. First deliverable: the parity harness (D-QSP-H)

The pieces exist per slice (`lance-graph-quack/examples/qsp_agg_pg_parity.rs`,
`qsp_null_operand_pg_parity.rs`, `k3_where_pg_parity.rs`;
`lance-graph-contract/examples/k3_pg_parity.rs`;
`lance-graph-r2il/examples/null_aggs_pg_parity.rs`): each drives `psql`
against a real server, prints `SKIP` without one, and keeps data in column
lanes. What is missing is one harness that every stage reuses:

- identical fixtures and queries against real PostgreSQL and the
  implementation;
- without `ORDER BY`: compare **multisets**, duplicates preserved;
- with `ORDER BY`: compare sequences, allowing only the ties the query
  leaves unspecified;
- compare types, NULLs and errors, not only displayed values;
- record every unsupported feature as a typed refusal, never a silent skip;
- mutation tests: an unchanged mapping with changed values must expose the
  current values; changed keys must invalidate dependent preparations
  (EXO-255).

## 4. First milestone (D-QSP-M1)

Two classes, a contextual property projection, `WHERE`, one
duplicate-preserving join and an ordered result, checked against PostgreSQL
through the harness, **with evidence that no intermediate X × Y tuples were
materialized**.

A counting allocator alone does not prove this: a pre-allocated buffer can
hold X × Y tuples without any new allocation. The evidence therefore has
three parts, and must tell three cases apart — **virtual binding**,
**bounded scratch**, **materialized X × Y population**:

1. **Execution path:** the lowered program is inspected; no operator on the
   path writes one record per (x, y) pair.
2. **Written intermediate rows / bytes:** every operator reports how many
   intermediate rows and bytes it wrote. Virtual binding writes none; bounded
   scratch writes at most a stated bound that does not grow with |X| × |Y|;
   a materialized population writes at least the number of pairs.
3. **Allocations:** a counting `#[global_allocator]` in a dedicated test
   binary (as D-DR-VMAP's F5 does), bounded by the result and the prepared
   artifacts.

The gate scales the fixture (|X| and |Y| each ×4) and checks that (2) and
(3) do not grow with the product. A negative control that does build the
pairs must be reported as "materialized" by the same instrument.

## 5. Comparison targets and their roles

| Target | Role |
|---|---|
| Real PostgreSQL | semantic oracle and reference execution |
| PgLance (`lance-format/pglance`) | **unverified comparison candidate.** Before any run: pin the commit used and check its Lance version, pushdown and execution path at that commit. The duckdb plan §A.9 records an earlier reading (Lance 1.0, row-at-a-time FDW, no pushdown); that is a lead, not a verified property of the commit in use. A runtime difference is never labelled tuple-conversion overhead as a whole; that share is measured separately (for example per-cell conversion time against scan time) |
| lance-graph native | the virtual-projection and algebra implementation |
| DuckDB 2.0 API (polyfill) | an explicit compatibility surface, like the SQL row polyfill: DuckDB-dialect SQL and its relational API answered by the same inverted tuples and registered algebra. Real DuckDB 2.0.0 (LTS, 2026-10-21) is the oracle for that surface; PostgreSQL stays the oracle for standard SQL semantics. Scope only what DuckDB calls stable: its own release post says C API v2 is not yet marked stable and Rust bindings are only "in the works", so no ABI-level polyfill until they are. DuckDB's Quack remote protocol is a separate target, §5a |
| lance-graph-java (Valhalla / Panama) | later experiment separating representation cost, JVM execution and native-call overhead |

For Java, a JVM implementation and a Panama-to-native path are measured
separately, so a faster native kernel is never read as a benefit of the Java
object representation. DuckDB stays the analytical performance comparator
(D-QSP). The evidence roles are never combined into one speed-up (duckdb
plan §A.9).

## 5a. DuckDB's Quack remote protocol as a compatibility target

**Two different things share the name.** *DuckDB-Quack* is an external
client-server protocol owned by DuckDB. *Our Quack* (`lance-graph-quack`) is
the execution substrate: binding and lowering into relalg / mask-RISC /
`ndarray`. An adapter may speak the first so that its clients reach the
second; the protocol never becomes the engine, and the engine never takes
the protocol's types. PostgreSQL stays the SQL parity oracle; real DuckDB
2.0 is the oracle only for the DuckDB dialect and for the protocol.

**What the protocol is (read 2026-11-11 from DuckDB's own pages; no official
wire specification exists):**

- Transport: HTTP(S) `POST /quack`, body MIME type `application/duckdb`,
  default port 9494; version negotiated in the connection messages
  (`min_supported_quack_version` / `max_supported_quack_version`,
  `quack_version`). Source: `duckdb/duckdb-quack`
  `src/include/quack_message.json`; DuckDB 2.0 docs, Quack reference.
- Messages: connection request / response (with `auth_string`), prepare
  request (SQL text, `sql_query`) / response (results as data chunks), fetch,
  send-data, cancel, heartbeat, acknowledgement, success, error, disconnect.
  No plan, catalog-lookup, scan or transaction message type is named.
- Execution side: the server. "The query executes remotely and the server
  streams the result back"; `ATTACH` "binds the remote's table definitions on
  the client", and filters on attached tables run remotely.
- Result encoding: DuckDB's own binary serialization of data chunks, not
  Arrow ("This is why we use our own serialization in Quack").
- Status: the extension repo says "pre-release … currently experimental";
  DuckDB's blog planned the first production release with 2.0. License MIT.

**Can its clients reach our Quack through an adapter? Yes in principle,
because the client ships SQL text and the server executes.** An adapter has
four parts, each with its own gate:

1. **Wire layer (zone 3):** HTTP endpoint, message (de)serialization in
   DuckDB's binary format, pinned to one DuckDB version, since the format is
   coupled to DuckDB internals and unspecified. Gate: a real DuckDB 2.0 client
   completes connect / prepare / fetch / disconnect against the adapter.
2. **SQL intake:** DuckDB-dialect SQL text, both what users type and what the
   client itself generates (for `ATTACH`'s table definitions and for pushed
   filters), lowered to our Quack. Unsupported statements return the
   protocol's error response with a typed reason, never a silent fallback.
3. **Catalog view:** whatever the client queries to bind table definitions
   is answered from `ClassView` schemas and the versioned mapping.
4. **Result encoding:** our results encoded as DuckDB data chunks at the
   boundary only (boundary materialization, §1). Gate: results equal the same
   query on a real DuckDB 2.0 server over the same data, and equal PostgreSQL
   where the query is standard SQL.

Scope as everywhere: read-only first; writes and forwarded transactions are
refused with a typed error until a separate workstream admits them.

**What is unknown, and the probe that settles it (D-QSP-Q0):** which SQL the
client sends during `ATTACH` and when it pushes filters, and the exact byte
layout of the messages and data chunks. Probe: run a DuckDB 2.0 client
against a DuckDB 2.0 `quack_serve` server, record every request and response
(the extension's log types, or a reverse proxy in between), and list the
statements and chunk layouts. Until Q0 runs, the adapter's size is not known
and no part of it is scheduled.

## 6. Not verified

Nothing was built or run for this roadmap. The stage-to-owner mapping cites
plans and board rows read on main. PgLance is listed as an unverified
candidate; nothing about it has been checked at a pinned commit.
