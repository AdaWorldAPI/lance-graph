# quack-multitable-bind-v1 — one bind path for joins, shared by MeTTa bodies and SAP

> Status: PROPOSAL (2026-10-16). Nothing below is built. D-QMB-0..5.
> Feeds D-ASM-1 / D-ASM-11 (MeTTa bodies) and SAP CATS joins
> (`sap-glove-quack-v1.md`). Read against `origin/main` + PR #1587.

## 0. Why

`Draft` binds one table (`lance-graph-quack/src/bind.rs:192-199`). A predicate
naming another table is refused (`BindError::WrongTable`, `bind.rs:295-301`).
The executor already runs joins inside one program, through three leaves that
read a caller-supplied `Foreign { planes, lanes }`:

- `Filter::EqU32Via { fk, key: ForeignLane, v }` — a foreign value compared
  through the address lane, no foreign plane (`lib.rs:348-365`);
- `Filter::Semijoin { fk, foreign: ForeignPlane }` — existence through a
  RESIDENT foreign plane, lowered to `MaskOp::Gather` (`lib.rs:366-392`);
- `Agg::GroupSumViaI32` — a group key read through the address lane.

So execution is not the gap. Two things are:

1. **No name reaches those leaves.** A caller hand-picks `ForeignLane(0)`,
   `ForeignPlane(1)` and must assemble `Foreign` in the same order. No binder
   says which field of which table sits in which slot.
2. **No binder says which lane is an address.** `Binder::field` returns a
   `BoundField { col, kind, validity }`; nothing marks `lines.partner_id` as
   holding row coordinates of `partner`.

Both consumers hit the same gap (entry `2026-10-16-wcoj-probes-1-3.md`
§ "Bind DTO synergies"): a MeTTa body `(R a b) (S b c)` and an SAP
`CATS ⋈ assignment` both need one head table plus foreign reads.

## 1. Frozen decisions

1. Bind once, execute numeric: the bound result holds no name (`bind.rs:15-20`,
   `tests/string_fence.rs`).
2. The binder never mints (`Binder::code`, `bind.rs:99-102`).
3. An address lane holds row coordinates of another table; it is not a foreign
   key in the relational sense and the plan does not add one (rule: hops are
   coordinate ranges). The existing IR field name `fk` stays as it is.
4. Chained hops fold into one program; nothing is materialized per hop
   (`Semijoin` already forbids a plane another `Program` produced, `lib.rs:193-198`).
5. One trait: `Binder` grows by DEFAULT methods only, so the four implementers
   (`lance-graph-sap::CatsBinder`, `lance-graph-dir-sim`, `lance-graph-report`'s
   test, `lance-graph-glove-parity`) compile unchanged.
6. Codebooks stay per binder. CATS codes are batch-local; atom addresses are
   stable. Share the trait, never the dictionary.

## 2. Shape

### 2.1 The binder declares addresses

```rust
trait Binder {
    // … existing four methods …
    /// The table whose rows `field` addresses, if it is an address lane.
    fn addresses(&self, _table: TableId, _field: Col) -> Option<TableId> { None }
}
```

Default `None`: nothing changes for a binder that declares no address lane.

### 2.2 The draft names the hop

```rust
table("lines")
    .via("partner_id")                 // the address lane, by name
    .where_eq("country", "DE")         // a field of the ADDRESSED table
    .count()
```

`via(field)` switches the predicate scope to the addressed table until the
next `via` or `back()`. Nested `via` chains (`lines → partner → region`) are
D-QMB-4, not v1.

### 2.3 The bound result carries its slot layout

```rust
pub struct Bound {
    pub query: Query,
    /// Slot i of `Foreign::lanes` must be this (table, column).
    pub lanes: Vec<(TableId, Col)>,
    /// Slot i of `Foreign::planes` must be this (table, plane).
    pub planes: Vec<(TableId, Mask)>,
}
```

Slots are assigned in first-use order and deduplicated, so the same foreign
field read twice takes one slot. The caller (or a later catalog) builds
`Foreign` from this list; a mismatch is a caller bug the executor already
rejects by lane kind (`reference.rs:383-388`). `Draft::bind` keeps returning
`Query` for one-table drafts; the new entry point is `Draft::bind_joined`.

### 2.4 What a hop binds to

For `via(a)` to table `T` with predicates `p₁ … pₙ`:

- each `T.f = literal` binds to `EqU32Via { fk: a, key: slot(T, f), v }`;
- the hop always adds `Semijoin { fk: a, foreign: slot(T, live(T)) }`.

The second leaf is load-bearing. `eq_u32_via_to_mask` drops an out-of-range
address (`ndarray/src/simd_masking_ops.rs:3156-3158`) but does not read the
foreign table's live plane, so a deleted target row still matches. And
`NOT EqU32Via` keeps every dangling address. Conjoining the live-plane
semijoin makes a dangling or deleted target a known FALSE.

Three-valued rules follow `Filter::sql_where` (`lib.rs:506-535`):
`EqU32Via` is UNKNOWN when its address lane is nullable; `Semijoin` is never
UNKNOWN (NULL address = FALSE).

### 2.5 What v1 does not bind

- `<>` on a foreign field: `NeU32Via` does not exist. A `<>` through a hop is
  `NOT EqU32Via ∧ Semijoin(live)`; v1 refuses it with a new
  `BindError::Unsupported` until a test pins that rewrite.
- `I32` foreign fields: there is no `EqI32Via`. Refused.
- Optional foreign fields with their own validity plane: refused until the
  foreign validity plane is a `ForeignPlane` slot conjoined like `live`.
- Comparing two lanes (a cyclic body's closing atom) and n:m runs: these need
  the predicates under council in `council-p7` (Part A / Part B). The bind
  path takes them when they exist; it does not invent them.

## 3. Work items

- [ ] **D-QMB-0** tests first (`lance-graph-quack/src/bind.rs` tests):
  - a two-table draft binds to exactly the query an expert writes
    (`EqU32Via` + live `Semijoin`) and the expected slot layout;
  - binder call count: one `table`, `live`, `field`, `code`, `addresses` per
    distinct name, as the one-table test pins (`bind.rs:536-557`);
  - executed answer == a nested-loop oracle over the same rows, including a
    dangling address and a deleted target row (both must drop);
  - disable runs: drop the live `Semijoin` → the deleted-target row is kept
    (red); skip slot dedup → two slots for one field (red).
- [ ] **D-QMB-1** `Binder::addresses` (default) + `Draft::via` + `Bound` +
  `bind_joined`; `BindError::{NotAnAddress, Unsupported}`.
- [ ] **D-QMB-2** SAP first consumer: CATS hours through the employee
  assignment where the assignment is functional (one row per employee):
  `CatsBinder::addresses`. Oracle: the SAP fixture's own expected rows.
- [ ] **D-QMB-3** MeTTa first consumer: a two-atom acyclic body
  `(R a b) (S b c)` binds through `via`; answer == datafrog SET
  (`lance-graph-metta/examples/wcoj_probe.rs` oracle).
- [ ] **D-QMB-4** nested `via` chains (`a → b → c`): one program; needs a
  read through two address lanes. Check first whether `GatherChain`'s
  withdrawal (#1521/#1522) already settled this shape.
- [ ] **D-QMB-5** cyclic bodies and n:m relations: after the `council-p7`
  predicates land; a cyclic body binds its closing atom to Part A or Part B.

## 4. Non-goals

- A table catalog or storage owner (`Registrar` stays unimplemented, `bind.rs:403-412`).
- Aggregates across an n:m join (fan-out counting is D-ASM-3).
- Join ordering, a planner, DataFusion (in grace).
- Any change to the executor beyond what D-QMB-5 inherits from the council.

## 5. Kill conditions

- The bound query differs from the expert query on any D-QMB-0 fixture.
- A deleted or dangling target is kept by `=` (the live semijoin is missing).
- An existing `Binder` implementer needs an edit to compile.
