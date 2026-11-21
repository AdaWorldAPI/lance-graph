# The consumer surface: data as config, class as config

> **READ BY:**
> - any session that gives a consumer (zammad-rs, report-rs, the codegraph,
>   OpenObserve, odoo-rs, openproject-nexgen-rs) a PostgreSQL-shaped,
>   Quack-shaped or cube-shaped read;
> - any session that is about to write a column type, a class view, a key
>   map or a binding by hand;
> - any session that extends `lance-graph-intake`, `lance-graph-pg-row`,
>   `lance_graph_contract::pg_row` or the by-reference read in DT64.
>
> **Status:** VERIFIED-IN-CODE and MEASURED where a receipt is named.
> OPEN rows wait on the decision named in them.
>
> **Written:** 2026-11-21, right after the implementation, so the reasons are
> recorded before they fade. Companions:
> - `register-contract-reconciliation.md`: the contract and its
>   contradictions;
> - `morton-vs-ordinal-rows.md`: which coordinate a row is.

## 0. What a consumer sees

```rust
let tickets = Table::from_rows("ticket", classid, &columns, &rows, &ShapeConfig::default())?;
tickets.select_where(&filter)?;              // PgRows, as PostgreSQL prints them
tickets.rows_for_keys(&[7, 3, 7])?;          // a bag, list order, repeats kept
tickets.left_join("owner_id", &agents)?;     // missing side NULL / ''
tickets.delete_key(9)?;
tickets.observe(&new_rows)?;                 // Ok, or ShapeChanged
```

The consumer hands in column names and cells as PostgreSQL text (or NULL). It
never names any of these:
- a lane, a presence plane or a live plane;
- a DR tile, a position or a coordinate space;
- a `ClassView`, a `BindStore`, a route or a `LaneBinding`;
- a mask-risc program.

Inside, the table *is* the register contract. Every row is read as
`identity[index].key : identity[index].value` at an ordinal position, through
one check per table, and keys and positions are a bijection over the DR
tiles. The consumer does not have to think about any of that. That is the
point of the facade.

## 1. The layers, and what each owns

| Layer | Crate / module | Owns | Never does |
|---|---|---|---|
| Consumer facade | `lance-graph-intake` (`Table`, `ShapeConfig`) | detection, hand-config merge, verdicts; the source owner (lanes, presence, live, code tables, `MapTable`, selections); registration and the one bind | evaluate a predicate; spell a cell |
| PostgreSQL row polyfill | `lance-graph-pg-row` (`Reader`, `read_bag`) | one by-reference check per table; WHERE through Quack; rows at positions or under a mask; the virtual tuple bag | own data; keep a class view |
| Cell contract | `lance_graph_contract::pg_row` | the admitted types, `LaneEncoding`, `AbsentAs`, `is_absent` (−0), `write_cell` | read a lane |
| Read by reference | `lance-graph-dt64` `LaneBinding::check_bound`, `borrow_all{,_selected}_present_bound` | frame, route, store revision and digest (O(1)), lane mapping, scope, row order | consult a `ClassView` after bind |
| Predicates | `lance-graph-quack` (`Filter`, `sql_where`, `lower`) → `lance-graph-mask-risc` (`execute_into`) | three-valued WHERE, `Count`, `Keep` | loop over rows in the consumer |
| Placement | `lance-graph-dt64::dr` (`MapTable`, `LocalMap`, `locate`) | key → (tile ordinal, local), bijective, versioned | hold a value |

## 2. What was built, with receipts

All of it landed or is in PR on 2026-11-21. The PostgreSQL oracle is 16.15
throughout.

- **#2152, the cell contract and the reader.**
  - What landed: `pg_row` (5 unit tests: −0, `''`, leap years, centuries,
    type refusals), and `lance-graph-pg-row` with `tests/pg_gate.rs`.
  - Gate: equal to PostgreSQL on `SELECT *` 276, K3 WHERE 139, issued bag
    82, LEFT JOIN 276 with 130 missing sides.
  - Disables, each red: presence (408 rows differ), live (15 deleted rows
    kept), bag-present (260 rows differ).
- **#2153, the read by reference** (`tests/bound_read.rs`, 4 tests).
  - Lends the same pointers as the view read.
  - A view changed after bind is invisible to it.
  - Refuses a different-basis store, a stale frame, a foreign frame store,
    a moved route, a remapped lane, and a selection in another row order.
  - Disable `disable-bound-digest` goes red.
- **#2153, the cube over three tiles** (`xlog/tests/cube_tiled_bijection.rs`).
  - 600 ids: bijective, and unchanged through edges and a refold.
  - One lookup per listed tile per key: 1800 for 600 keys.
- **This PR, `lance-graph-intake`** (24 tests).
  - **Detection:** one test per rule, an all-NULL column, and the witness of
    a value past `u32`.
  - **Hand config:** `Differs`, `Agrees`, unknown keys, an unparseable
    forced cell, conservation.
  - **Placement refusals.**
  - **`observe`:**
    - a changed shape is reported;
    - an all-NULL batch fits, and small values fit an int8 column (red on
      the first draft, which re-detected the batch alone).
  - **Deletes.**
  - **Gate:** model oracle and PostgreSQL agree on 276 / 139 / 82 / 276.
  - **Disables, each red:**
    - `disable-intake-signed`: a signed column lent as `u32` makes Quack
      refuse the WHERE (`LaneKind { expected: I32, found: U32 }`).
    - `disable-intake-live`: delete becomes a no-op, and two tests fail.

## 3. Data as config: the detection rules

Over a column's non-NULL cells, in this order (`detect.rs`):

1. **Empty:** no non-NULL cell gives text, NULL-absent, support 0.
2. **Bool:** every cell is `t` / `f` gives `boolean`.
3. **Integer:** every cell is a *canonical* `i64` (`"007"`, `"+5"`, `"-0"`
   are not), so the read prints back exactly what came in.
   - All fit `i32`: `integer`, a signed `i32` lane.
   - Else all fit `0..=u32::MAX`: `bigint`, an unsigned `u32` lane.
   - Else text, with the first row that did not fit as its witness.
4. **Timestamp:** `YYYY-MM-DD HH:MM:SS`, whole seconds, epoch within `u32`,
   gives `timestamp without time zone`.
5. **Date:** `YYYY-MM-DD`, days within `u32`, gives `date`.
6. **Text:** anything else gives `text`, as codes in first-seen order. Code 0
   is reserved for absent.

**Absent spelling.** A text column with `''` cells and no NULL is
`AbsentAs::EmptyText`: `''` is absent and is printed as `''`. Every other
column is `AbsentAs::Null`. This is DECISION EXO-372 in practice: `''`,
`$null`, NaN, NULL, missing and deleted are one code, −0.

**Key.** The first column named `id` that is an integer, never NULL and
distinct. Else the first integer column with those properties. Else
`ShapeConfig::key`, or `NoKey`.

## 4. Class as config, and the escape hatch

The class is built from the detected fields:
- one `FieldRef` per column, `intake:<table>/<column>`;
- `ClassView` view class = `classid >> 16`;
- `BindStore::register` at `SchemaRev(1)` with `Extent::DR64_CUBE`;
- routed on DR position 1;
- bound once, scope 1, in the source's own row order.

The **classid is the caller's.** The intake mints no concept: identity comes
from OGAR or the consumer's codebook (HubSPO rule 2, OGAR consumer preflight).

The **escape hatch** is `ShapeConfig`:
- per column `ty` / `encoding` / `absent`, plus the `key`;
- a present hand value wins over detection;
- every column gets a verdict: `Agrees`, `Differs { detected }` or
  `DetectedOnly`;
- the verdicts must add up to the columns (asserted in `Shape::build`);
- a config key that names no column is refused.

So a hand value is never silently wrong: it is either confirmed or shown to
differ, with the detection beside it. This is the same contract as the
codegraph intake arm (zammad-rs #94).

**A new shape is a new class.** `observe` checks new rows cell by cell against
the merged shape. Only a cell that does not fit gives `ShapeChanged { was,
now }`. The caller then builds a new `Table` under a new classid, as a new
versioned reference. It never mutates a registered class (C1, and the owner,
2026-11-21).

## 5. Synergies

- **Codegraph intake arm (zammad-rs `config_intake`).** It works the same way
  one level up: detect config from the source, let hand values win, and give
  every key a verdict with conservation. Rails schema facts (ruff schema
  stratum) can feed `ShapeConfig` directly:
  - `null: false, default: ""` gives `AbsentAs::EmptyText`;
  - a column type gives `ty`.

  The ruff schema stratum does not emit the default yet
  (`ruff/.claude/knowledge/rails-harvest-to-register.md` §4).
- **Read by reference.** Because the facade binds once and reads with no
  view, a registered class cannot drift under a read. That is the code form
  of "a `ClassView` changes only while a proposer stages a class that does
  not exist yet".
- **Replay key** (data version × DR reference × BindStore version). The
  facade's three refusals are frame, row order and digest, and each one
  enforces one component of it (`register-contract-reconciliation.md` C1).
- **Cube placement.** `Table` places keys exactly as `Cube::new_tiled` does:
  `ordinal * 256 + local` via `MapTable::locate`, so the same bijection holds
  (D-REGC-8).
- **XVERWEIS (report-rs EXO-369)** is `read_bag` with two sides. A report-rs
  source can sit behind the same `Reader`.
- **zammad-rs `quack-shadow`.** Its hand-written rendering and absent rule are
  what `Table` now does generically. Moving onto it is D-REGC-12's fruit, and
  #95's PostgreSQL gate is the oracle.
- **Morton vs ordinal.** Every table here is `Ordinal`: rows are records in
  storage order. A Morton-ordered projection is a different descriptor
  (D-REGC-13).

## 6. Caveats, and the solution for each

| Caveat | Why it is so | Solution | State |
|---|---|---|---|
| A value outside `i32` / `0..=u32::MAX` becomes text | a lane holds 32 bits; `pg-row` refuses `U64` lanes | lend `LaneRef::U64` and admit it in `pg-row` (`Int8` / `Signed64`), or split into two lanes | OPEN |
| Timestamps: whole seconds, no zone, up to 2106 | `EpochSeconds` is `u32` | a second lane for sub-seconds, or a `U64` epoch in µs | OPEN |
| Non-canonical integers (`007`) become text | so the read prints back what came in | the right behaviour; a consumer that wants numbers forces `ty` in config, and the cell is refused | BY DESIGN |
| An `EmptyText` column cannot tell a present `''` from an absent one | DECISION EXO-372: they are one code | none needed | BY DESIGN |
| `left_join` decodes the fk lane row by row (keys are resolved once per distinct key) | the classes do not share coordinates, so this is the fallback join form (C6) | place related classes on shared coordinates at intake, so the join becomes an aligned pass (D-REGC-9) | OPEN |
| `delete_key` clears the live bit without advancing the data version | the live plane is the owner's; the frame is the key map's | advance `FrameRevisions::source` on delete, so kept masks and selections are refused after it (replay key), or a delete writes −0 (D-REGC-3) | OPEN (EXO-337) |
| Duplicate column names are not refused | they collide on `intake:<t>/<c>`, and `lane_of` takes the first | refuse at `from_rows` | OPEN, small |
| Rows live in the owner's `Vec`s, not in 512-byte SoA rows | the interim store, as in zammad-rs (C9) | lend `LaneRef::Strided` over owner-held rows once the predicates the workload uses have strided forms (D-REGC-6, D-REGC-11) | OPEN |
| Text codes are in first-seen order | a code is an identity, not an order | text `ORDER BY` and ranges need an order index (EXO-114) | OPEN |
| `rows_for_keys` takes `&mut self` | the owner issues one selection per call | fine for one writer per mailbox; a shared reader needs selections keyed by call | ACCEPTED |
| One table, one class, one route position | the facade's scope | several tables per `BindStore` and route log share one store | OPEN |
| The classid comes from the caller | the intake never mints | OGAR / the consumer's codebook | BY DESIGN |
| The gate fixture is synthetic | deterministic, no real data committed | run consumer captures (zammad-rs `shadow-rows.pg.json`) through `Table` | OPEN |

## 7. Open ends, in one place

**Decisions:**
- EXO-370: one classid per shape. The read by reference already follows it.
- EXO-337 / EXO-294: whether a delete writes −0 (D-REGC-3).
- D-REGC-4: where an outer side's missing-row plane comes from. Today
  `Table` builds it as the source owner.
- EXO-329: the selection issuer for cross-source selections (D-REGC-5, -7).
  Within a `Table` the owner issues its own.
- D-REGC-13: name the row order in the DR descriptor.

**Measurements:**
- the basis share of the census (D-REGC-2);
- the predicate count over a consumer workload (D-REGC-6);
- chunked reads: allocations, bytes copied, ns per chunk at 256–4096
  positions (D-QSP-VROW).

**Features:**
- `ORDER BY`;
- aggregates through the facade (Quack has them: `lower_sum`,
  `lower_group_by`);
- 64-bit lanes;
- shared coordinates at intake (D-REGC-9);
- strided lanes (D-REGC-6);
- duplicate column names.

**Consumers:**
- zammad-rs onto `Table` / `Reader` (fruit of D-REGC-12);
- report-rs XVERWEIS on `read_bag`;
- ruff schema defaults into `ShapeConfig`.

## 8. Rules for the next session

1. **A consumer gets `Table`, not lanes.** If you write a `LaneSource`, a
   `ClassView` or a key map for a consumer by hand, check first whether
   `from_rows` plus a `ShapeConfig` entry expresses it.
2. **Detection proposes, config decides, verdicts report.** Never drop the
   verdict. Never let a hand value win silently.
3. **A shape change is a new classid**, never an edit to a registered class.
4. **Every new read path keeps the two oracles:** the model built from
   logical rows (always) and PostgreSQL (when reachable; with
   `PG_ROW_REQUIRE_PG=1` a SKIP fails). It also keeps one disable run per
   claim.
5. **When a caveat in §6 is fixed, change its row in the same PR.** Do not add
   a second document.
