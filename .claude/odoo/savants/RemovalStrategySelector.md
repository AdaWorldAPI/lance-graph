# Savant: RemovalStrategySelector  (id 24 · family None · lane L7)

**Tuple:** kind=NextBestAction · inference=Induction · semiring=**XorBundle** · style=Exploratory
**Feeds Reasoner impl:** NextBestActionReasoner   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:31`). Induction -> `QueryStrategy::CamWide`.
> **Semiring note:** this savant is `XorBundle`, **not** `NarsTruth` — the multi-quant set-binding
> fuses by XOR (each unit of demand is satisfied by *exactly one* quant, so the selected quants form
> a disjoint cover). The `NextBestActionReasoner` impl **branches on `savant.semiring`**: the
> `XorBundle` arm fuses candidate-quant evidence as a set cover; the `NarsTruth` arm (the other six
> in this group) fuses confidence-weighted. (Roster confirms `semiring: XorBundle`,
> `savants.rs:85`; L7 R3 confirms XorBundle rationale, `L7-STOCK.md:215-217`.)

## What it decides (AXIS-B core)
`_gather` / `_get_removal_strategy` (L7 R3, `stock_quant.py:L617-791`) selects which quants to bind
to a reservation. The strategy *order* (FIFO `in_date ASC` / LIFO `in_date DESC` / FEFO by expiry /
closest / least_packages) and the strategy *source* (product category then location-hierarchy
fallback, default `fifo`) are deterministic once the strategy is fixed (AXIS-A). The ambiguity the
savant owns: given the demand qty and the candidate quants (each carrying `in_date`, `lot_id`,
`removal_date`, `package_id`, `location`, available qty), **which ordered subset of quants forms the
best disjoint cover of the demand** under the active strategy — and when the cover is necessarily
**partial** (insufficient stock), where to stop. Output is an ordered `(quant_id, qty_to_reserve)`
set summing to `min(demand, total_available)`, XOR-fused, never an un-guarded reservation write.

## Deterministic guard (AXIS-A — stays in woa-rs)
`_get_removal_strategy` priority (category -> location-hierarchy walk -> default `fifo`,
`stock_quant.py:L617-628`); `_get_gather_domain` strict/non-strict domain construction
(`stock_quant.py:L750-769`); the SQL `order` per strategy and the final `.sorted(lambda q: not
q.lot_id)` lot-float-to-top tiebreak (`stock_quant.py:L771-791`, L7:180-197); and the whole
`_get_reserve_quantity` arithmetic once quants are gathered (DOWN-then-HALF-UP UoM conversion, serial
integer guard, per-quant `min(max_quantity, remaining_demand)`, `stock_quant.py:L834-914`, L7 R5
L7:332-368). The least_packages A* itself is deterministic given the package list (L7 FLAG-6,
L7:626-628). The savant fires for the heuristic *multi-criterion selection / when-to-stop* core
(L7 R3 Axis-2, "STRONG", L7:201-243).

## Slot 1 — Evidence (Arrow EvidenceRef)
Primary table = the pre-filtered candidate-quant set for one reservation request.
`EvidenceRef { table: "stock_quant.reservation_candidates", schema_fingerprint, rows }`, one row per
available quant (pre-filtered to `quantity - reserved_quantity > 0`), plus the demand carried in the
namespace/budget. This mirrors the L7 R3 evidence block (`L7-STOCK.md:222-243`):

| column | dtype | signal |
|---|---|---|
| `quant_id` | `Int64` | identity of the candidate quant (the unit selected into the cover) |
| `product_id` | `Int64` | product key (all rows share it) |
| `location_id` | `Int64` | location of the quant; drives `closest` strategy + child-of domain |
| `location_complete_name` | `Utf8` | hierarchical location name; the `closest` sort key (`sorted(complete_name, -id)`) |
| `lot_id` | `Int64`/nullable | lot/serial; lot-bearing quants float to top (specificity tiebreak); FEFO needs it for expiry |
| `in_date` | `Timestamp` | incoming date — the FIFO (`ASC`) / LIFO (`DESC`) ordering axis |
| `removal_date` | `Timestamp`/nullable | expiry-derived date — the FEFO ordering axis (`with_expiration` domain) |
| `quantity` | `Float64` | gross qty at the quant |
| `reserved_quantity` | `Float64` | already-reserved; `available = quantity - reserved_quantity` |
| `available_qty` | `Float64` | the bindable qty (the cover draws from this) |
| `package_id` | `Int64`/nullable | package grouping; drives `least_packages` A* and full-packaging rounding |
| `strategy_hint` | `Utf8` (`fifo\|fefo\|lifo\|closest\|least_packages`) | the AXIS-A-resolved active strategy (one value per batch) |
| `strict` | `Boolean` | exact lot/package/owner match required? (domain mode) |
| `packaging_uom_id` | `Int64`/nullable | full-packaging reserve method => floor to whole-package multiples |

Demand: `demand_qty: f64` (carried via the reservation request, not a per-row column). All evidence
columns are present in community `stock.quant` — **NOT NEEDS-INPUT**.

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `in_date` (FIFO/LIFO axis), the strategy order, the lot-float tiebreak <- `_gather` / `_get_removal_strategy_order` -- `L7-STOCK.md:161-197` (R3; `stock_quant.py:L617-791`); FIFO=`in_date ASC,id ASC`, LIFO=`in_date DESC,id DESC` (`L7-STOCK.md:168-169`).
- `removal_date` (FEFO axis) + `with_expiration` domain <- `_get_gather_domain` -- `L7-STOCK.md:170,178` (R3; `stock_quant.py:L750-769`); FEFO is domain-filtered, not a named order, and community `_get_removal_strategy_order` raises on unknown (`L7-STOCK.md:622-624` FLAG-5).
- `location_complete_name` (closest sort) <- `_gather` closest path -- `L7-STOCK.md:172,192-194` (R3; `stock_quant.py:L771-791`).
- `quantity`, `reserved_quantity`, `available_qty` <- `_get_available_quantity` -- `L7-STOCK.md:125-137` (R2; `stock_quant.py:L793-832`); `available = quantity - reserved_quantity`.
- `package_id` (least_packages A*) <- `_run_least_packages_removal_strategy_astar` -- `L7-STOCK.md:199` (R3; `stock_quant.py:L630-738`).
- `strict`, `packaging_uom_id`, the per-quant `min(max_quantity, remaining_demand)` draw + DOWN/HALF-UP guard <- `_get_reserve_quantity` -- `L7-STOCK.md:340-362` (R5; `stock_quant.py:L834-914`).
- the full evidence shape is given verbatim in the lane -- `L7-STOCK.md:222-243` (R3 "Evidence the Reasoner receives").

## Slot 3 — Property-level alignment
N/A — family None (needs alignment axiom); no property IRIs defined.

`stock.quant` is UNRESOLVED in `odoo_alignment` (L7 FLAG-3, `L7-STOCK.md:596-607`); the proposed
`gs1:QuantityElement` pivot is *class*-level only and explicitly "needs alignment authoring" with an
interim `Other(9)` tag — no property IRI exists. Do not invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the candidate-quant rows (slot 1) for one reservation request with demand `D` under
`strategy_hint`.

-> Conclusion C = `QuantCover([(quant_id, qty_to_reserve), ...])` — the ordered disjoint cover whose
qtys sum to `min(D, sum(available_qty))` — fused by **XorBundle** (each demand unit attributed to
exactly one quant, so the selected set is a partition, not a weighted blend). NARS
`(frequency, confidence)` where:
- **frequency** of selecting a given quant rises when it ranks early under the active strategy axis
  (earliest `in_date` for FIFO, nearest `removal_date` for FEFO, nearest `location_complete_name` for
  closest, fewest packages for least_packages) and carries a `lot_id` (specificity float-to-top).
- **confidence** is high and well-supported here (all signals are present community fields); it dips
  only when the cover is **partial** (`sum(available_qty) < D`) — the Abduction sub-path then
  explains the shortfall ("insufficient stock at location X", "all lots reserved", per
  `L7-STOCK.md:241`). Capped by the phi-1 humility ceiling.

Discriminating features (ranked, removal/assignment axis): active strategy order key (FIFO `in_date` /
FEFO `removal_date` / LIFO / closest) >> lot-specificity float > package consolidation
(least_packages) > strict-domain exactness. Induction generalises the strategy-consistent cover over
the candidate set (CamWide); the XorBundle semiring enforces the disjoint-cover fusion.

## Parity / GoBD notes
Suggestion-only (Iron Rule 7): the savant proposes a quant cover; woa-rs applies it through
`_update_reserved_quantity` behind the AXIS-A guard, never an un-guarded write. The reserved-qty
floor invariant `reserved_quantity = max(0, reserved + delta)` (never negative, L7:411, L7:706) stays
in woa-rs. No GoBD ledger surface (reservation is pre-valuation); the SVL/GL valuation that *would*
touch GoBD is a separate fresh build (L13:84-91). **Not NEEDS-INPUT** — all discriminating signals
are present community `stock.quant` fields.
