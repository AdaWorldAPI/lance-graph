# D-MAPPING-POLYFILL-1: a PSCustomObject-style field mapping over DT64 / DR / Quack (2026-11-16)

**Status:** VERIFIED-IN-CODE and TEST-PINNED for every REUSE row; one WIRE
(`MapTable::frames`); lab evidence, nothing production-admitted. Consumer:
report-rs (read-only). Base: lance-graph `c73e3d8`.

## The mental model, and what it maps to

```text
Source:    DirectoryUsers     -> one source owner: its MapTable + its lanes (LaneSource)
Selection: AuthorizedUsers    -> a position list the source holds (SelectionSource)
Fields:    Identity    <- dir:objectId      -> identity key: MapTable::locate
           Name        <- dir:displayName   -> lane: LaneBinding::selected
           PrimaryMail <- dir:mail          -> lane: LaneBinding::selected
           Groups      <- dir:memberOf      -> relation: dir-sim View::groups_transitive (not a lane)
```

Aliases are display metadata. They resolve to a field position through the
class basis (`FieldRef::predicate_iri`) once, at bind, and take no part in
identity, routing or authority. No new runtime DSL.

## A. Evidence matrix

| Operation | Existing implementation | Oracle in this PR | Admission | Missing edge |
|---|---|---|---|---|
| Direct property projection | `lance-graph-dt64/src/lane.rs`: `BindStore::bind_lanes`, `LaneBinding::selected` / `lane` / `borrow_all` (D-LANESOCK-1, #1969/#1970) | `dt64/tests/mapping_polyfill.rs` falsifiers 1, 7, 9, 10 | lab; report-rs adapter `AbiLanes` is `#[cfg(test)]` only | a production reader-scope witness (see "Authorization") |
| Identity-keyed lookup (XVERWEIS by id) | `dr.rs` `MapTable::locate` (Ambiguous / Unmapped), `LocalMap::prepare` (Duplicate) | falsifier 3, `identity_key_lookup_reads_the_return_field_under_one_frame` | lab | **was:** the table's frame store was private, so locate and the lane read had two witnesses. **WIRED here:** `MapTable::frames()` |
| Value-keyed lookup (XVERWEIS by value) | Quack `Filter::cmp(col, Cmp::EqU32(key))` + `Agg::Count`, then `Agg::Rows` (#1963 §9) | `quack/tests/xverweis_masked_read.rs` | lab; no `MethodRef` | a same-program unique terminal and an admitted `MethodRef` (report-rs #33 ledger 8). Composed from two programs today |
| Text key to code | `dr.rs` `CodeTable` (per-property bijection; Duplicate refused; frame-keyed history) | not re-tested here | lab | canonical text-to-code authority (#33 ledger 6, EXO-207) |
| Membership (user -> groups) | `lance-graph-dir-sim/src/view.rs` `groups_transitive`, `security_identifiers`, `is_member`; separate user/group ordinals | falsifier 5 (separate coordinates; relation refused as a lane) | lab | none for reading; membership is a SET (`snapshot.rs` sort + dedup) |
| Bag / count | selection list is a bag (`lane.rs` SelectionSource: never sorts or dedups); `Cube::bag_count`; `r2il binding::run_bag` (#2066/#2069) | falsifier 6 | lab | none |
| Join size without tuples | `Cube::join_size` | `xlog/tests/cube_counts.rs` (#2097) | lab | none |
| Versioned read | `FrameStore` check / `issued`; `MapTable::map` answers old frames at their own version; dir-sim `log.rs` sparse batches + `replay` | falsifier 2; identity test (old frame reads its own map) | lab | lanes admit the current frame only; history of VALUES is dir-sim's log (C below) |
| Field authorization | `lance-graph-rbac/tests/scoped_lane_binding.rs` (EXO-243): `requested ∩ ScopedDecision::visible_fields()` before `bind_lanes` | not re-tested here | test-only | row half is a sketch; `Scope` equality is a consistency check, never a grant |

## B. Consumer contract (report-rs), version 1

What report-rs supplies (it owns none of the substrate):

1. **A source owner** implementing `LaneSource` + `SelectionSource`
   (`lance-graph-dt64::lane`). It owns its lanes, its selections and, for
   identity keys, its `MapTable`; `frames()` returns `MapTable::frames()`, so
   one store witnesses keys and lanes. It never copies, reorders or allocates a
   lane in `borrow`.
2. **A recipe**: ordered `(alias, predicate_iri)` pairs. report-rs resolves an
   alias to a field position by `predicate_iri` against the ClassView it
   registered; an alias it cannot resolve is refused, never defaulted.
3. **A request**: `LaneRequest { route, rev, fields: &WideFieldMask, frame,
   scope, selection }`. `fields` is the authorized field set (requested ∩
   visible). `frame` is the source's current seal.

What it gets back:

- `Ok(LaneBinding)` from `BindStore::bind_lanes` (cold, allocates the binding
  once); then per field `LaneBinding::selected(.., field, sel) ->
  (S::Lane<'_>, &[u32])` (hot: borrowed lane + the source's own positions, no
  allocation). The terminal (CSV, grid, PSCustomObject) is report-rs's and is
  built only after this point.
- For an identity key: `MapTable::locate(frame, tiles, id) -> (tile, local)`;
  the row is `tile * 256 + local` in the source's row order (the source's
  obligation, stated in its own docs).
- For a value key: the two-program recipe of A, row 3. Not a library call.

Refusals (all returned before a lane or a position is borrowed):
`LaneRefusal::{Frame(ForeignFrame | stale), RouteChanged, StoreBasisMismatch,
PredicateChanged, FieldNotInBasis, Unbound, MappingChanged, Vanished,
NotBound, CoordinateSpace, Scope, SelectionUnbound, SelectionSpace,
SelectionScope, SelectionVanished, Bind(..)}`, plus
`LocateRefusal::{Ambiguous, Unmapped, Read}` and `MapRefusal::Duplicate`. A
value read has three states: value (including an explicit zero), missing, and
deleted; an absent field is `Unbound`, never empty.

Provenance report-rs must keep with each result: the frame (its version and
revisions), the route version, the schema revision, the bound field set, the
selection id, and the scope. Lifetime: a binding is valid while its frame is
current; any source advance refuses it, and report-rs rebinds.

## C. Read-only probe

`crates/lance-graph-dt64/tests/mapping_polyfill.rs` (10 tests): 512 users
(2048 in the growth arm), 64-field basis, a five-identity selection, two
projected fields, a restricted field, a membership field with no lane.
`crates/lance-graph-quack/tests/xverweis_masked_read.rs` (5 tests): unique /
absent / duplicate keys over a 512 and 2048 row mail lane.

## D. Trace, with measured counts

Identity | Select-Object Name, PrimaryMail, five identities:

1. `field_of(view, alias)` -> position by `predicate_iri` (test-side, once).
2. `BindStore::bind_lanes` -> `source_owns`, `frames.check`, route, basis,
   one `lane_of` per field: **2 `lane_of` calls**; allocates the binding.
3. Per field `LaneBinding::selected` -> `check` (again `lane_of` per bound
   field), `selection_of`, `borrow`, `borrow_positions`: **1 borrow per field,
   0 allocations, 0 map lookups**.
4. Terminal reads `lane[row]` per selected position: **10 reads** (5 x 2).

Measured identical at 512 and 2048 users (lane_of, borrows, map lookups,
allocations, reads). No join, no XLOG, no tuple. Identity lookup: **one
lookup per listed tile** (2 for 512 users), no lane scan. Value lookup:
**2 programs** for a unique key, **1** for absent or duplicate, **0
allocations** in the prepared read; each program is a column fold, so its
cost grows linearly with rows (no index is claimed).

Disable runs, each red before the guard was restored:

| Guard disabled | Test that failed |
|---|---|
| `frames.check` in `LaneBinding::check` | reordered rows, stale-frame, identity lookup (3) |
| `LocateRefusal::Ambiguous` | duplicate keys |
| `PredicateChanged` compare | reordered basis |
| scope check in `admit_lane` | foreign/scope/stale |
| `source_owns` at bind | foreign/scope/stale (after isolating it: the first fixture let `frames.check` refuse first, so the guard was not reached) |
| an allocation in `selected` | five identities; growth |
| `Count` capped at 1 in mask-risc | duplicate value key |

## E. Proposal

- **REUSE** the lane socket, `MapTable::locate`, `CodeTable`, dir-sim
  membership readers, the bag selection, Quack `Count`/`Rows`.
- **WIRE** (this PR): `MapTable::frames(&self) -> &FrameStore`, read-only.
- **PROBE** (next, separate): a same-program unique terminal for value-keyed
  XVERWEIS, so the count and the read cannot see different lanes; today they
  are one frame apart only by convention of running both under one binding.
- **REFUSE**: reading a relation (memberOf) as a lane; first-match on a
  duplicate; any gather across coordinate spaces.

## F. Handoff (for the report-rs session)

- Entry points: `lance_graph_dt64::{BindStore::bind_lanes, LaneBinding::{selected,
  lane, borrow_all, check}, LaneSource, SelectionSource, LaneRequest}`,
  `lance_graph_dt64::dr::{MapTable::{locate, frames, seal, advance},
  LocalMap::prepare, CodeTable}`, `lance_graph_quack::{lower, Query, Filter,
  Cmp::EqU32, Agg::{Count, Rows}}`, `lance_graph_mask_risc::{execute,
  execute_into, Scratch, Out::Mask}`.
- Fixture recipe: copy `Source` from `mapping_polyfill.rs`: one MapTable with
  W256 tiles of identity ids, one lane per field with presence and tombstone
  bits, `frames()` returning `table.frames()`.
- Limits: current frame only for lanes; `Scope` is not authorization;
  membership is a set; value lookup is two programs with no `MethodRef`;
  `AbiLanes` stays test-only; nothing in report-rs changes in this PR.
