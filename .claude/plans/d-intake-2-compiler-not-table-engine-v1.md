# D-INTAKE-2: intake as a compiler into a versioned, codebook-addressed execution surface (v1)

**Status:** 2A, 2B, 2C built and in PR. 2D measured: two failed gates, now 2G
and 2H. 2E and 2F queued. Results: `knowledge/consumer-surface-data-and-class-as-config.md` §9.

**Follows:**
- D-REGC-14 (#2154, `lance-graph-intake`), whose consumer API is kept;
- `.claude/knowledge/consumer-surface-data-and-class-as-config.md`.

**Basis:** owner, 2026-11-21:
- "`lance-graph-intake` should be a compiler from external data into a
  versioned, codebook-addressed execution surface, not another table engine";
- "The ORM is allowed to think in rows and objects. The engine should not
  have to";
- PostgreSQL is the semantic oracle, DuckDB the execution benchmark.

## 1. The shape

```text
ORM / PowerShell / PostgreSQL tuples, text, schema proposals, NULL policies
  │  admission + append (cold)
  ▼
DT codebooks (CodeTable) · DR identity placement (MapTable) · class binding (BindStore)
  │  append immutable data revisions
  ▼
XLOG cube / versioned register: codes · lanes · masks · counts · coordinates
  │  fold · refold · keep · replay
  ▼
amortized selection: DR-owned positions + a witnessed SelectionRef
  │
  ▼
virtual tuple bag: left positions × right positions × presence × multiplicity
  (no joined tuple is materialized; values are decoded only when the consumer asks)
```

**Ownership.**
- **DT** owns string → code translation.
- **DR** owns identity → position translation.
- **XLOG and Quack** operate on admitted codes and ordinal positions only.
- **The facade** owns nothing that duplicates any of these.

## 2. What #2154 does today, read against that shape

VERIFIED-IN-CODE at `6e0a15c`:

| Concern | #2154 | Target |
|---|---|---|
| string → code | a local `Codes { Vec<String>, BTreeMap<String, u32> }` per column, in first-seen order | DT `CodeTable<u32>` per `CodeSpaceId { class, property }`, advanced through `MapTable::advance_codes` (`dr.rs:1190`), so a code change advances the lanes' frame |
| identity → position | already DR: `MapTable::locate` over W256 tiles | unchanged |
| selections | `Owner::selections`, built and cleared per call | a `SelectionRef` kept under the replay key with `MapTable::keep` / `kept` (#2150), reused while its version is witnessed |
| data revisions | one build, then `delete_key` flips live bits without advancing a version | `append` and `delete` advance the data version (`FrameRevisions::source`); an old snapshot's selection stays readable under its own frame |
| decode | at the end of each read (`write_cell`) | unchanged: decode only on demand |

## 3. Slices

| D-id | Deliverable | Gate (red first, then green) |
|---|---|---|
| D-INTAKE-2A | **Text codes in DT.** The facade's `Codes` is replaced by one `CodeTable<u32>` per table, with one `CodeSpaceId` per text column, advanced with `MapTable::advance_codes`. The `CodeText` view for `write_cell` reads `CodeTable::key` at the bound frame. No string is held outside DT. | the PostgreSQL gate is unchanged; a code read under a frame from before an append returns that frame's mapping; a test that the facade holds no `String` per row |
| D-INTAKE-2B | **Append, not rebuild.** `Table::append(rows)` admits new rows as codes and positions: new keys extend the last tile or add a tile (`LocalMap::extends`), new strings extend each space's generation, and the data version advances once. `delete_key` advances it too. | after `append`, the gate equals PostgreSQL on old ∪ new rows; append cost is counted in rows touched, which must equal the rows appended, never the table's size; a binding made before the append is refused after it (frame), then rebound |
| D-INTAKE-2C | **Amortized selection.** `Table::select_ref(filter) -> SelectionRef` computes a WHERE once, keeps its positions with `MapTable::keep` under the current version, and returns a handle. `rows_of(&SelectionRef)` borrows the kept positions. A newer version refuses the ref with `NotWitnessed`; an old snapshot's ref still reads under its own frame. | warm `rows_of` does no predicate work (counted: 0 mask-risc executions); after an append the old ref is refused for the newest state and still reads its snapshot; replay is bit-equal |
| D-INTAKE-2D | **DuckDB benchmark.** The same data and predicate run through DuckDB 2.0.1 (Python, `duckdb.sql` over Arrow or a VALUES table) and through `Table`. Results are reported in ns, separately: cold selection, warm replay, validation overhead (one `check_bound` plus digest), decode per cell. | cold selection ns per selected row within 2× DuckDB at 10^5 and 10^6 rows (target, not a claim); warm replay against DuckDB re-running the query, labelled amortization; validation overhead at 1, 64 and 4096 selected rows; no millisecond result (lance-graph P0) |
| D-INTAKE-2E | **Multiplicity and ordering gates.** Covers bag multiplicity through append (repeated keys in a relation), NULL policies after append, and snapshot validity. | PostgreSQL multiset equality per snapshot; ORDER BY stays out until EXO-114 |
| D-INTAKE-2F | **zammad-rs onto `Table`.** `quack-shadow` reads through the facade. | zammad-rs#95's PostgreSQL gate unchanged; the census (#96) shows one check per table read |
| D-INTAKE-2G | Sink-based cell writer in `pg-row` (`write_rows(positions, &mut impl CellSink)`, one reused buffer, L1 chunks) | `rows_of(4096)` in µs, with every cell equal to `write_cell` (today a failed gate: 272–357 ns per row) |
| D-INTAKE-2H | Validity planes kept once per data version (`MapTable::keep`) | `count_where` P2 at 10^6 rows in µs (today a failed gate); 0 validity folds on a warm WHERE (counted) |

## 4. Rules this plan keeps

1. **Text never on the hot path.** Lexical ordering needs an explicit
   codebook-side order index (EXO-114). Code order is never SQL order.
2. **Absence is decided at admission.** NULL versus `''` is the declared
   `AbsentAs` plus presence, never re-interpreted from strings at read time
   (EXO-372).
3. **Bijectivity is not the whole correctness story.** Append ordering,
   publication, bag multiplicity and replay each get their own gate.
4. **Bench cold and warm separately.** A cached replay against DuckDB
   recomputing a query shows amortization, not a faster selection vector.
5. **The consumer API of #2154 stays.** `from_rows` becomes a convenience
   over `append` on an empty table.

## 5. Open

- Whether `SelectionRef` needs `relation_version` separately, or whether the
  relation's `CoordSpace` already carries it. Today it does for a cube
  relation (`CubeLanes::space`).
- How a selection over two tables (a join) is witnessed: two refs, or one
  ref per bag. This depends on EXO-329.
- How data versions map onto Lance versions (durable append). Out of scope
  here.
