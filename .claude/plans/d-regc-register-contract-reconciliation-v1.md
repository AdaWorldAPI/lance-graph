# D-REGC: reconciling the register contract (v1, 2026-11-21)

**Status:** PROPOSED / DOCS ONLY. Nothing in this plan is built. Each row that
changes code waits on its decision.
**Knowledge doc:** `.claude/knowledge/register-contract-reconciliation.md`. It
holds the contradictions C1–C9, their sources, and the falsifier for each.
**Evidence:**
- zammad-rs #95: the PostgreSQL gate through `borrow_all_selected_present`.
- zammad-rs #96 / #97: the ordinal register census and the consumer DR
  bijectivity.
- lance-graph #2146–#2151.
- Coordination: EXO-332, EXO-368, EXO-253, EXO-329, EXO-337, EXO-294, EXO-140,
  EXO-327.

## Why one plan

Each row below was found by measuring one consumer against PostgreSQL. Each
touches a contract that several plans state differently. This plan keeps the
fixes in one order, so that no slice builds on the side of a contradiction that
the owner has not chosen.

## Order and gates

| D-id | Deliverable | Gate (red first, then green) | Needs |
|---|---|---|---|
| D-REGC-0 | Knowledge doc and this plan; ⊘ notes on d-dt64-7 §1f and waben:1266; pointer from classid-on-do | docs only | — |
| D-REGC-1 | DT64 read through one class lens: frame + scope + order checks only, no `LaneBinding`, no route or basis step per read | lens read equals `borrow_all_present` on every lane and refuses the same frame/scope/order cases; counts 0 route and 0 basis sub-checks | C1 decision, then DT64 owner |
| D-REGC-2 | Basis resolved through the view at `register` and bind only; per read, one O(1) digest compare against the store (knowledge doc C1) | a changed basis refused at `register` and bind; a different store refused per read; a view mutated after bind changes no answer; per-read sub-checks fall by the re-resolution share. The census counts schema and basis sub-checks together (49,140 of 146,453), so the basis share alone is measured first | C1 decision |
| D-REGC-3 | A delete writes −0 on every lane of the row and clears the live bit | delete then read returns −0, never the old value; 9-bit lane: no nonzero magnitude left under presence 0 | EXO-337 / EXO-294 |
| D-REGC-4 | The selection carries a presence plane (or a reserved −0 position) for an outer-join side with no row | zammad-rs #95's deleted-parent arm, without the consumer rendering the missing side itself | decision on the form |
| D-REGC-5 | `CubeLanes: SelectionSource`; DR issues selections under its frame and row order | a selection issued before `add_edge` / `refold` is refused after it | EXO-329 |
| D-REGC-6 | Strided forms of the lane-only predicates and masked aggregates a recorded workload uses | per-predicate count over zammad-rs's 512 calls first; then each strided form equals its lane form on the same rows | — |
| D-REGC-7 | Keys resolved once per distinct key per frame and row order (`CoordSpace`), never per cell or per row; resolved again after a row-order change | inbound `locate` count == distinct ids, not cells (from #2149's per-cell count) | D-REGC-5 |
| D-REGC-8 | Measure the cube across several tiles, and bijection after `add_edge` / `refold` | round trip and injectivity on ≥2 tiles; `locate` count per key in a multi-tile join | — |
| D-REGC-9 | Consumer intake places related classes on shared coordinates; the join is an aligned pass | zammad-rs join calls show 0 per-row key resolutions and equal PostgreSQL | EXO-140; D-REGC-4 |
| D-REGC-10 | PostgreSQL egress renders an absent text cell as `''` where the column defaults to `''`, else NULL | zammad-rs wide capture: 1958 `''` cells equal; a column mixing `''` and NULL is reported, not merged | C3 recorded |
| D-REGC-11 | zammad-rs's store moved onto owner-held 512-byte rows lent as `Strided` | census counts and PostgreSQL answers identical before and after | D-REGC-6 |
| D-REGC-12 | `lance_graph_contract::pg_row` + `crates/lance-graph-pg-row`: the PostgreSQL row polyfill for consumers (knowledge doc §4) | `tests/pg_gate.rs` = model oracle = PostgreSQL 16.15 on 4 cases; 3 disables red | — (In PR) |
| D-REGC-13 | The DR projection descriptor names its row order (`Ordinal` / `Morton { width }`); order-dependent predicates refused against the other order (`knowledge/morton-vs-ordinal-rows.md` §5) | one prefix predicate admitted as `Range` only under Morton; both answers equal a value-sweep oracle | DT64 / DR owner decision |
| D-REGC-14 | `lance-graph-intake`: data as config, class as config; one facade over pg-row, DT64 and DR (`knowledge/consumer-surface-data-and-class-as-config.md`) | `tests/intake.rs` (20) + unit (4); model oracle = PostgreSQL 16.15 on 4 cases; disables `disable-intake-signed`, `disable-intake-live` red | — (In PR) |

## Decisions this plan needs

1. **C1, schema store.** Decision J (d-dt64-7 §1f) against EXO-199 and the
   owner's 2026-11-20 statement. The proposed answer is one classid per shape,
   with the `ClassView` as the only schema. This is filed in lance-graph ·
   Decisions as DECISION NEEDED.
2. **C3, deletes.** Does a delete write −0 (D-REGC-3), or does every read keep
   refusing dead positions? This is EXO-337 / EXO-294.
3. **C4, the outer-join side.** A presence plane on the selection, or a
   reserved −0 position.
4. **C7, the selection issuer.** EXO-329.

## Not in scope

- New join operators. The join stays an aligned pass, and admission of
  `EqU32Via` / `Semijoin` / `Gather` stays EXO-140.
- egglog / DialEgg / datafrog / Ascent on the execution path. They stay offline
  and oracle only (knowledge doc C6).
- A second presence bit. The owner, 2026-11-21: "useless until we really need
  it".
