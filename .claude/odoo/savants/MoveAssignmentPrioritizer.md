# Savant: MoveAssignmentPrioritizer  (id 25 · family None · lane L7)

**Tuple:** kind=NextBestAction · inference=Induction · semiring=NarsTruth · style=Exploratory
**Feeds Reasoner impl:** NextBestActionReasoner   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:31`). Induction -> `QueryStrategy::CamWide`.

## What it decides (AXIS-B core)
`action_assign` (picking level, L7 R4, `stock_picking.py:L1195-1208`) sorts confirmed moves by
`(-priority, not date_deadline, date_deadline, date, id)` before reserving stock against them. That
sort is a closed-form policy (AXIS-A). The ambiguity the savant owns: when **stock is scarce and not
every confirmed move can be satisfied**, which moves should claim the limited quants *first* — a
judgement that the static priority/deadline lexicographic sort approximates but does not fully
capture (e.g. two equal-priority deadline-bearing moves where one is nearly satisfiable from on-hand
and the other is not, or where one move's customer is more time-critical than its raw deadline says).
The savant induces a confidence-weighted ordering over the contended moves from their urgency
signals. Output is a NARS-weighted move ordering, never a reservation write.

## Deterministic guard (AXIS-A — stays in woa-rs)
The `action_assign` lexicographic sort `(-int(priority), not bool(date_deadline), date_deadline,
date, id)` (`stock_picking.py:L1195-1208`, L7:308-312, L7:457-461) and the entire inner
`_action_assign` reservation loop with its in-memory sibling accounting (the deterministic part once
the quant set is fixed — `assigned_moves_ids` / `partially_available_moves_ids` passed into
`_get_available_move_lines_out`, `stock_move.py:L1901-2043`, L7:251-316). The quant *selection* that
loop calls is RemovalStrategySelector (R3), not this savant. `_trigger_assign`'s domain/filter is
AXIS-A; only its `reference_ids` secondary sort is a soft heuristic (R9, WEAK — deterministic
fallback acceptable, L7:730-731). The savant fires only on the residual "which contended move wins
the scarce stock" core (L7 R4 Axis-2 "STRONG", L7:314-324).

## Slot 1 — Evidence (Arrow EvidenceRef)
Primary table = the set of confirmed/waiting/partially_available moves contending for stock in one
`action_assign` batch. `EvidenceRef { table: "stock_move.assign_candidates", schema_fingerprint, rows }`,
one row per contending move:

| column | dtype | signal |
|---|---|---|
| `move_id` | `Int64` | identity of the contending move |
| `picking_id` | `Int64`/nullable | parent picking (batch partition) |
| `product_id` | `Int64` | product the move needs |
| `priority` | `Int64` (`0\|1`) | Urgent flag — the primary sort axis (`-priority`); `1` = Urgent reserved first |
| `date_deadline` | `Timestamp`/nullable | commitment deadline; deadline-bearing moves precede deadline-less (`not date_deadline`) |
| `date` | `Timestamp` | scheduled date — tertiary sort axis |
| `product_uom_qty` | `Float64` | demanded qty (the need) |
| `reserved_qty` | `Float64` | already-reserved (`missing = product_uom_qty - reserved`); near-complete moves are cheap wins |
| `free_qty_at_src` | `Float64` | available qty at the move's source location; how satisfiable this move is right now |
| `state` | `Utf8` (`confirmed\|waiting\|partially_available`) | only these contend; `partially_available` already holds some stock |
| `procure_method` | `Utf8` (`make_to_stock\|make_to_order`) | MTO moves skip MTS reservation (Branch C); discriminates eligibility |
| `move_type` | `Utf8` (`direct\|one`) | shipping policy — `one` (all-at-once) raises the cost of a partial win |

All columns are present in community `stock.move` / `stock.picking` — **NOT NEEDS-INPUT**.

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `priority`, `date_deadline`, `date`, `id` (the sort axes) <- `action_assign` move sort -- `L7-STOCK.md:308-312` (R4; `stock_picking.py:L1195-1208`) and `L7-STOCK.md:457-461` (R7).
- `product_uom_qty`, `reserved_qty`, `missing = product_uom_qty - reserved_availability[move]` <- `_action_assign` per-move logic -- `L7-STOCK.md:265-270` (R4; `stock_move.py:L1901-2043`).
- `free_qty_at_src` <- `_get_available_quantity` move wrapper / quant level -- `L7-STOCK.md:116-137` (R2; `stock_quant.py:L793-832`, `stock_move.py:L1846-1850`).
- `state` eligibility (`['confirmed','waiting','partially_available']`) <- `_action_assign` entry -- `L7-STOCK.md:257` (R4).
- `procure_method` (MTO skip / MTS branch) <- `_action_assign` Branch B/C -- `L7-STOCK.md:279-292` (R4).
- `move_type` (`direct` vs `one`) shipping policy <- picking `_get_relevant_state_among_moves` / `_compute_state` -- `L7-STOCK.md:451,489` (R7; `stock_move.py:L1320-1360`).
- the assignment-priority NARS tuple is named in the lane -- `L7-STOCK.md:318-324` (R4 Axis-2) and `L7-STOCK.md:729` (NARS summary: Induction · NarsTruth · Exploratory · STRONG).

## Slot 3 — Property-level alignment
N/A — family None (needs alignment axiom); no property IRIs defined.

`stock.move` is UNRESOLVED in `odoo_alignment` (L7 FLAG-1, `L7-STOCK.md:567-581`); proposed
`gs1:LogisticEvent` is *class*-level only, "needs alignment authoring", interim `Other(7)` — no
property IRI exists. Do not invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the contending-move rows (slot 1) for one `action_assign` batch under scarce stock.

-> Conclusion C = `AssignOrder([move_id, ...])` — the order in which moves should claim the limited
quants — emitted with NARS `(frequency, confidence)` where:
- **frequency** of "satisfy this move first" rises with `priority == 1` (Urgent), an earlier
  `date_deadline`, a small `missing` qty relative to `free_qty_at_src` (a near-complete cheap win),
  and `move_type == 'one'` only when fully satisfiable (else an all-or-nothing move shouldn't hog
  partial stock).
- **frequency** falls for deadline-less moves, MTO moves (which skip MTS reservation anyway), and
  moves whose `free_qty_at_src` cannot make a dent.
- **confidence** is high and well-supported (all signals present community fields); it reflects how
  cleanly the urgency signals separate the contenders. Capped by the phi-1 humility ceiling.

Discriminating features (ranked, assignment axis): priority (Urgent) >> deadline proximity >
satisfiability-from-on-hand (`missing` vs `free_qty_at_src`) > shipping policy (`one` vs `direct`) >
scheduled `date`. Induction generalises the best-first ordering from the urgency signals (CamWide).

## Parity / GoBD notes
Suggestion-only (Iron Rule 7): the savant proposes an assignment order; woa-rs reserves through
`_action_assign` behind the AXIS-A guard, never an un-guarded write. The in-loop sibling accounting
that prevents double-reservation stays in woa-rs (L7:708 gotcha #6). No GoBD ledger surface
(reservation is pre-valuation). **Not NEEDS-INPUT** — all discriminating signals are present
community `stock.move` / `stock.picking` fields.
