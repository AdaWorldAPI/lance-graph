# D-REGC-12: the PostgreSQL row polyfill for consumers (2026-11-21)

**Status:** MEASURED (test-pinned), in PR #2152.

**What:**
- `lance_graph_contract::pg_row` holds the admitted types, the lane
  encodings, the absent spelling (`Null` or `''`) and the one cell writer.
- `crates/lance-graph-pg-row` is the reader. Each table gets one DT64 check.
  WHERE goes through Quack `sql_where` and mask-risc. Rows are read at
  issued positions or under a kept mask. Joins are a virtual tuple bag.

**Measured** (`cargo test -p lance-graph-pg-row`, with `PGHOST=/var/run/postgresql`
and `PG_ROW_REQUIRE_PG=1`, PostgreSQL 16.15). The model oracle and
PostgreSQL agree on all 4 cases:

| Case | Rows |
|---|---|
| `SELECT *` | 276 live of 300 |
| K3 `WHERE` | 139 |
| issued bag | 82, with repeats |
| `LEFT JOIN` | 276, 130 missing sides |

**Disable runs**, each red:
- presence: 408 rows differ;
- live: 15 deleted rows kept;
- bag-present: 260 rows differ.

**Open:**
- the presence 0, value ≠ 0 arm has no fixture row;
- `ORDER BY`;
- the issuer of join positions (EXO-329);
- the outer side's plane (D-REGC-4);
- zammad-rs adoption.

**Fruit:** move zammad-rs `quack-shadow`'s `read_selection` / `absent_row`
rendering onto `lance-graph-pg-row::Reader`, keeping #95's gate as the oracle.
