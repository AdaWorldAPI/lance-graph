# D-REGC-14: data as config, class as config (2026-11-21)

**Status:** MEASURED (test-pinned), in PR. Knowledge:
`.claude/knowledge/consumer-surface-data-and-class-as-config.md`.

**What it is.** `lance-graph-intake::Table::from_rows` takes column names and
PostgreSQL-text cells. It:
- detects each column's type, lane encoding, absent spelling and the key;
- merges hand `ShapeConfig`, where present values win and each column gets a
  verdict with conservation;
- builds and registers the class and routes it;
- places keys in W256 DR tiles;
- binds once.

Reads go through `lance-graph-pg-row` by reference.

**Measured** (`cargo test -p lance-graph-intake`,
`PGHOST=/var/run/postgresql PGDATABASE=vrow PG_ROW_REQUIRE_PG=1`):
- 24 tests pass.
- The model oracle and PostgreSQL 16.15 agree on `SELECT *` (276), K3 WHERE
  (139), a key bag (82) and LEFT JOIN (276).
- Disables, each red:
  - `disable-intake-signed`: Quack refuses an `i32` compare on a `u32` lane;
  - `disable-intake-live`: deleted rows come back.

**Found while verifying the draft:**
- the first draft lent signed columns as `u32` lanes, so WHERE failed;
- `observe` re-detected a batch on its own, so an all-NULL batch read as a
  changed shape. It is red on the old code and fixed.

**Open:** the caveats table, §6 of the knowledge doc. Main items:
- 64-bit lanes;
- a delete that does not advance the data version (EXO-337);
- the fallback join form (D-REGC-9);
- the owner's `Vec`s instead of SoA rows (D-REGC-11).

**Fruit:** zammad-rs `quack-shadow` onto `Table`, with #95's gate as the
oracle.
