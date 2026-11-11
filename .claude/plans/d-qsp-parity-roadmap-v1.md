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
materialized**. The evidence uses a counting `#[global_allocator]` in a
dedicated test binary, as D-DR-VMAP's F5 does: allocations stay bounded by
the result and the prepared artifacts, not by the cross product. This tests
the proposition directly.

## 5. Comparison targets and their roles

| Target | Role |
|---|---|
| Real PostgreSQL | semantic oracle and reference execution |
| PgLance (`lance-format/pglance`) | further comparison arm, after inspecting its SQL coverage and execution path. Recorded so far (duckdb plan §A.9): Lance 1.0, row-at-a-time FDW, no pushdown, so it measures tuple-conversion overhead, not resident folds |
| lance-graph native | the virtual-projection and algebra implementation |
| lance-graph-java (Valhalla / Panama) | later experiment separating representation cost, JVM execution and native-call overhead |

For Java, a JVM implementation and a Panama-to-native path are measured
separately, so a faster native kernel is never read as a benefit of the Java
object representation. DuckDB stays the analytical performance comparator
(D-QSP). The evidence roles are never combined into one speed-up (duckdb
plan §A.9).

## 6. Not verified

Nothing was built or run for this roadmap. The stage-to-owner mapping cites
plans and board rows read on main; PgLance's coverage is as recorded in the
duckdb plan, not re-inspected.
