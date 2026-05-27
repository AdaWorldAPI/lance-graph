# Savant: RouteTiebreaker  (id 14 · family None · lane L13)

**Tuple:** kind=NextBestAction · inference=Abduction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** NextBestActionReasoner   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:31`). **Abduction** -> `QueryStrategy::DnTreeFull` (this savant abduces
> *why* one route is the better explanation of the demand, vs ProcurementRuleSelector's Induction).

> **STATUS: NEEDS-INPUT** — same root cause as ProcurementRuleSelector(11): supplier lead/cost/
> capacity are not community `stock` fields; the `purchase` addon is absent (L13:5-6, open Q#5/#7).
> The schema below is what the impl consumes once woa-rs supplies a purchase feed; until then the tie
> is irreducible from `stock` data alone.

## What it decides (AXIS-B core)
Distinct from ProcurementRuleSelector (which ranks rules *within* a chain walk), RouteTiebreaker
addresses the narrower artefact in `stock_rule.py:L586-611` (L13 R16): when product-specific routes
have been floated to the front but **two routes still share the same sequence**, the surviving order
is pure Python `sorted` stability — arbitrary. The savant abduces which route best *explains* a
reliable fulfilment of the demand, weighing **supplier lead-time, cost, and capacity** across the
competing routes. Output is an abductive preference (the route that is the most economical
explanation of "demand met on time"), NARS-weighted, never a route write.

## Deterministic guard (AXIS-A — stays in woa-rs)
Product-specific-routes-sort-first plus the `(route_sequence, sequence)` ordering
(`stock_rule.py:L586-611`, L13:80-82 R16). Any strict ordering resolves in woa-rs. The savant fires
**only** on the residual equal-sequence route tie (R16 explicitly: "equal-sequence order is arbitrary
(Python sorted stability)").

## Slot 1 — Evidence (Arrow EvidenceRef)
Primary table = the tied **route** set (not individual rules), each route joined to its terminal
supplier/manufacture leg. `EvidenceRef { table: "stock_route.tied_candidates", schema_fingerprint, rows }`,
one row per route tying on sequence:

| column | dtype | signal |
|---|---|---|
| `route_id` | `Int64` | identity of the candidate route |
| `route_sequence` | `Int64` | the tie key — rows in one evidence batch share this value |
| `n_rules` | `Int64` | number of legs in the route; a longer chain accrues more cumulative `delay` |
| `cum_delay_days` | `Int64` | sum of `rule.delay` across the route's legs — the only lead-time proxy community `stock` provides |
| `terminal_action` | `Utf8` (`buy\|manufacture\|pull`) | what the last leg does; `buy` rows need the supplier feed, `manufacture` rows need an mrp feed (also absent, L13:96) |
| `supplier_id` | `Int64`/nullable | **REQUIRES purchase feed (absent)** — vendor on the terminal `buy` leg |
| `supplier_lead_days` | `Float64`/nullable | **REQUIRES purchase feed (absent)** — realised lead time; the primary discriminator |
| `supplier_capacity` | `Float64`/nullable | **REQUIRES purchase feed (absent)** — can this supplier cover the demand qty? capacity discriminator |
| `route_cost` | `Float64`/nullable | **REQUIRES purchase feed (absent)** — landed cost through the route; secondary discriminator |

The three nullable supplier columns are the NEEDS-INPUT core; `cum_delay_days` is the only
non-null lead signal and is too coarse (static) to break the tie confidently.

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `route_sequence` / equal-sequence-is-arbitrary <- `stock.route` ordering in route selection -- `L13-STOCK-VALUATION-PROCUREMENT.md:80-82` (R16; `stock_rule.py:L586-611`).
- `cum_delay_days` <- sum of `stock.rule.delay` across legs (the `_run_push` `new_date = move.date + rule.delay` and the `_get_stock_move_values` `date_planned - rule.delay`) -- `L13-STOCK-VALUATION-PROCUREMENT.md:48-49` (R6; `stock_rule.py:L222-285`) and `L13-STOCK-VALUATION-PROCUREMENT.md:45-46` (R5; `stock_rule.py:L324-408`).
- `terminal_action` <- `stock.rule.action` (`buy`/`manufacture`/`pull`) per R3 priority cascade -- `L13-STOCK-VALUATION-PROCUREMENT.md:38-39` (R3; `stock_rule.py:L564-638`).
- `supplier_id`, `supplier_lead_days`, `supplier_capacity`, `route_cost` <- **NO community `stock` field**; `purchase`/`purchase_stock` absent (`L13-STOCK-VALUATION-PROCUREMENT.md:5-6`); `mrp` `_run_manufacture` route action also absent (`L13-STOCK-VALUATION-PROCUREMENT.md:96`); `purchase_delay` set by the purchase addon, handle-missing-gracefully (`L13-STOCK-VALUATION-PROCUREMENT.md:105` open Q#7). => woa-rs purchase feed required; **NEEDS-INPUT**.

## Slot 3 — Property-level alignment
N/A — family None (needs alignment axiom); no property IRIs defined.

`stock.rule -> fibo:Agreement` is a proposed *class*-level pivot only
(`L13-STOCK-VALUATION-PROCUREMENT.md:22`); no property IRI exists for lead/cost/capacity. Do not
invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the tied route rows (slot 1) sharing one `route_sequence`.

-> Conclusion C = `PreferRoute(route_id)` — the route that most economically explains "demand
fulfilled on time" — emitted with NARS `(frequency, confidence)` where:
- **frequency** rises for the route with the shortest `supplier_lead_days`, sufficient
  `supplier_capacity` for the demand qty, and the lowest `route_cost`; falls for long-lead /
  capacity-short / costly routes.
- **confidence** is **structurally capped low** while the supplier feed is absent — with only
  `cum_delay_days` (static, often equal across tied routes) the abduction has no discriminating
  evidence and defers to the AXIS-A arbitrary order. Capped by the phi-1 humility ceiling.

Discriminating features (ranked): supplier lead-time >> supplier capacity (can it even cover the
qty?) > route cost > cumulative static `delay`. Abduction (DnTreeFull) selects the single most
economical route-as-explanation consistent with E.

## Parity / GoBD notes
Suggestion-only (Iron Rule 7): re-ranks tied routes, never writes a route or procurement. No GoBD
surface (pre-ledger). **NEEDS-INPUT** must be surfaced: until the purchase feed exists, the reasoner
returns a low-confidence echo of the AXIS-A stable-sort order.
