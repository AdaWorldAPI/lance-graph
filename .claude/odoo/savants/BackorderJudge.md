# Savant: BackorderJudge  (id 26 · family None · lane L7)

**Tuple:** kind=NextBestAction · inference=Abduction · semiring=NarsTruth · style=Exploratory
**Feeds Reasoner impl:** NextBestActionReasoner   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:31`). **Abduction** -> `QueryStrategy::DnTreeFull` (this savant abduces
> *why* the fulfilment is partial, then judges backorder-vs-cancel — see L7 R7 Axis-2).

## What it decides (AXIS-B core)
On a partially-fulfilled picking, `_create_backorder` / `_check_backorder` (L7 R7,
`stock_picking.py:L1533-1602`) decide whether to split the unfulfilled remainder into a backorder
(deliver later) or cancel it. The mechanism is gated by `picking_type.create_backorder` =
`always` | `never` | `ask`; only the `ask` case is undecided (L7:482, L7:486-489). The ambiguity the
savant owns: in the `ask` case, **should the remainder become a backorder or be cancelled?** — which
hinges on *why* the picking is partial (transient stock-out that will resolve vs a structural
shortage / discontinued line vs a customer who won't accept a split delivery). The savant abduces the
most likely cause of the partial fulfilment and judges the remainder accordingly. Output is a
NARS-weighted backorder-vs-cancel judgement, never an un-guarded picking write.

## Deterministic guard (AXIS-A — stays in woa-rs)
The `always` / `never` arms of `picking_type.create_backorder` (closed-form), the
`_check_backorder` predicate (a move needs backorder if `product_uom_qty > 0 AND not picked` OR
`_get_picked_quantity() < product_uom_qty` at Product-Unit precision, `stock_picking.py:L1533-1546`,
L7:482), the mechanical `_create_backorder` split (copy picking, move unfulfilled moves, reset
`user_id=False`/`picked=False`, re-assign if `at_confirm`, `stock_picking.py:L1577-1602`, L7:472-480),
and the move-level `_create_backorder` / `_split` arithmetic at Product-Unit precision
(`stock_move.py:L2174-2260`, L7:88-90). The savant fires **only** on the `create_backorder == 'ask'`
residual (L7:482, R7 Axis-2 "MODERATE", L7:484-496).

## Slot 1 — Evidence (Arrow EvidenceRef)
Primary table = the partially-fulfilled picking and its short moves.
`EvidenceRef { table: "stock_picking.backorder_candidate", schema_fingerprint, rows }`, one row per
unfulfilled move on the picking (header fields repeated per row or carried in namespace):

| column | dtype | signal |
|---|---|---|
| `picking_id` | `Int64` | identity of the partial picking |
| `move_id` | `Int64` | the short move |
| `product_id` | `Int64` | product the move concerns |
| `product_uom_qty` | `Float64` | demanded qty |
| `picked_qty` | `Float64` | actually-fulfilled qty; `shortfall = product_uom_qty - picked_qty` |
| `move_type` | `Utf8` (`direct\|one`) | shipping policy — `one` (all-at-once) leans cancel-or-full, `direct` accepts a split |
| `create_backorder` | `Utf8` (`ask\|always\|never`) | the gate; savant fires **only** on `ask` |
| `free_qty_at_src` | `Float64` | current available stock at source; high => transient stock-out (backorder will fill), zero/structural => cancel |
| `procure_method` | `Utf8` (`make_to_stock\|make_to_order`) | MTO shortfall implies upstream failure (different cause than MTS stock-out) |
| `product_active` | `Boolean` | a discontinued/archived product leans cancel (no future fill) |
| `reservation_method` | `Utf8` (`at_confirm\|manual\|...`) | `at_confirm` => backorder auto-re-assigns (cheap to keep open) |

All columns are present in community `stock.picking` / `stock.move` / `product.product` — **NOT
NEEDS-INPUT**.

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `create_backorder` gate (`ask`/`always`/`never`) + the `ask`-only ambiguity <- `_check_backorder` / `button_validate` -- `L7-STOCK.md:482` and `L7-STOCK.md:486-489` (R7; `stock_picking.py:L1533-1546`, L1398-1458).
- `product_uom_qty`, `picked_qty`, `shortfall` at Product-Unit precision <- `_check_backorder` / move `_create_backorder` -- `L7-STOCK.md:482` and `L7-STOCK.md:88-90` (R1/R7; `stock_move.py:L2174-2190`, `stock_picking.py:L1533-1546`).
- `move_type` (`direct` vs `one`) "is a partial a valid delivery?" judgement <- R7 Axis-2 -- `L7-STOCK.md:489` and `L7-STOCK.md:710` (gotcha #7; `_compute_state:L858-862`).
- `reservation_method == 'at_confirm'` => backorder auto-re-assign <- `_create_backorder` post-create -- `L7-STOCK.md:480` and `L7-STOCK.md:720` (gotcha #12; `stock_picking.py:L1577-1602`).
- `free_qty_at_src` (transient vs structural shortage) <- `_get_available_quantity` -- `L7-STOCK.md:116-137` (R2; `stock_quant.py:L793-832`).
- the backorder NARS tuple is named in the lane -- `L7-STOCK.md:491-496` (R7 Axis-2) and `L7-STOCK.md:730` (NARS summary: Abduction · NarsTruth · Exploratory · MODERATE).

## Slot 3 — Property-level alignment
N/A — family None (needs alignment axiom); no property IRIs defined.

`stock.picking` is UNRESOLVED in `odoo_alignment` (L7 FLAG-2, `L7-STOCK.md:583-594`); proposed
`ubl:DespatchAdvice` is *class*-level only, "needs alignment authoring", interim `Other(8)` — no
property IRI exists. Do not invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the short-move rows of one partial picking (slot 1) with `create_backorder == 'ask'`.

-> Conclusion C = `BackorderRemainder(picking_id)` vs `CancelRemainder(picking_id)` — emitted with
NARS `(frequency, confidence)` where:
- **frequency** of *backorder* rises when the cause abduces to a **transient** stock-out (high
  `free_qty_at_src` building up, `move_type == 'direct'` so a split is acceptable, `product_active`,
  `reservation_method == 'at_confirm'` making re-assignment automatic).
- **frequency** of *cancel* rises when the cause abduces to a **structural** shortage (`free_qty_at_src`
  near zero with no inbound, an archived `product_active == false`, or `move_type == 'one'` where a
  partial delivery is unacceptable so the remainder is better cancelled than left dangling).
- **confidence** is well-supported (all signals present community fields); it tracks how decisively
  the evidence points to one cause. Capped by the phi-1 humility ceiling.

Discriminating features (ranked, backorder axis): abduced shortage cause — transient vs structural
(`free_qty_at_src` + inbound) >> shipping policy (`one` vs `direct`) > product still active >
auto-re-assign cheapness (`at_confirm`). Abduction (DnTreeFull) picks the most economical explanation
of the partial fulfilment, which determines the judgement.

## Parity / GoBD notes
Suggestion-only (Iron Rule 7): the savant judges backorder-vs-cancel for the `ask` case; woa-rs
executes the split/cancel behind the AXIS-A guard, never an un-guarded write. The mechanical split
(Product-Unit precision, `user_id=False` reset, auto-re-assign) stays in woa-rs. No GoBD ledger
surface (fulfilment is pre-valuation; the delivery's GL impact is the separate fresh SVL build,
L13:84-91). **Not NEEDS-INPUT** — all discriminating signals are present community fields.
