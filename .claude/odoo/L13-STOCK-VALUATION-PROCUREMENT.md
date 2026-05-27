RICHNESS-LANE-OK

# Lane L13 — Stock↔Accounting Valuation Bridge + Procurement

## Critical Enterprise gap (read first)
`stock_account` is **absent** from this community clone: no `stock_valuation_layer.py`, no `_run_fifo`/`_run_average`/`_create_account_move_vals` (those live in `stock_account`/`purchase_stock`). The valuation **engine** must be built fresh in woa-rs; this lane specs the **interface boundary** (SVL shape, FIFO/AVCO/standard formulas, anglo-saxon vs continental GL) from the field scaffolding that IS present, plus the fully-present procurement/reorder/lot logic. (L7 already covered moves/picking/quant/reservation.)

## Sources read
- stock/models/stock_rule.py : L1-747 : full
- stock/models/stock_orderpoint.py : L1-817 : full
- stock/models/stock_lot.py : L1-431 : full
- stock/models/product.py : L1-1389 : full
- stock/models/res_company.py : L1-215 : full
- product/models/product_product.py : standard_price + onchange : targeted
- product/models/product_category.py : L1-69 : full
- account/models/company.py : anglo_saxon_accounting + price_diff acct : targeted
- stock/models/stock_move.py : L1546-1679 (procure routing), L2101-2160 : targeted
- stock/models/__init__.py : confirms stock_valuation_layer absent

## Ontology rows
| odoo class | owl pivot | family | DOLCE |
|---|---|---|---|
| `stock.warehouse.orderpoint` | fibo:Obligation | None — Layer-2 axiom needed | Perdurant |
| `stock.rule` | fibo:Agreement | None | Abstract |
| `stock.lot` | schema:ProductModel + GS1 | None | Endurant |
| `product.product` (standard_price) | schema:Product / fibo:Asset | None (→ propose 0x63 ProductCatalog) | Quality |
| `res.company` (anglo_saxon) | fibo:LegalEntity config | 0x62 SMBAccounting | Abstract |
| `stock.move` (procure_method) | fibo:Transfer | None | Perdurant |

## Rules extracted (16; 11 AXIS-A, 5 AXIS-B/HYBRID)

### R1 — standard_price field [AXIS-A]
- product_product.py:L62-68 — Float, company_dependent (ir.property → woa-rs: `product_cost(product_id,company_id)` table), groups base.group_user, negative-guard onchange. Decimal (RFC-009).

### R2 — anglo_saxon vs continental [AXIS-A]
- account/company.py:L146, L298-311 — AngloSaxon: COGS at invoice (interim via expense_account_id, price_difference_account_id absorbs std-vs-bill delta). Continental (default, GoBD-correct DE): COGS at delivery. woa-rs default Continental.

### R3 — _get_rule (location-hierarchy walk + route priority) [HYBRID → SAVANT]
- stock_rule.py:L564-638 — build location ancestor chain; `_search_rule_for_warehouses` grouped by (dest, warehouse, route) ORDER BY (route_seq, seq); route priority: values.route_ids → packaging → product|categ routes → warehouse routes; walk chain, first match wins; transit-location edge adds customers loc. Walk+priority = AXIS-A; equal-sequence tiebreak = heuristic.
- `SAVANT: name=ProcurementRuleSelector family=None reasoning=NextBestAction inference=Induction semiring=NarsTruth style=Analytical — route priority among equal-sequence rules should weigh lead time, stock availability, supplier reliability, not deterministic tiebreak.`

### R4 — _run_pull [AXIS-A]
- stock_rule.py:L287-316 — validate location_src; sort positive-qty first; mts_else_mto→make_to_stock at pull time; build move vals; group by company; create (sudo, with_company) + _action_confirm.

### R5 — _get_stock_move_values [AXIS-A]
- stock_rule.py:L324-408 — date = date_planned - rule.delay; partner = rule.partner_address_id or values; location_dest only if location_dest_from_rule else picking-type default; to_refund if qty<0; serialize procurement_values (ids/isoformat).

### R6 — _run_push [AXIS-A]
- stock_rule.py:L222-285 — transparent (modify dest in-place, recurse on change, loop guard) vs manual (copy move, procure_method=make_to_order, new_date = move.date + rule.delay).

### R7 — procure_method routing in _action_confirm [AXIS-A]
- stock_move.py:L1546-1678 — buckets: waiting (move_orig_ids or make_to_order), create_proc (triggers rule.run), mts_else_mto: `qty_to_procure = max(product_qty - free_qty, 0)`, `qty_from_stock = min(product_qty, free_qty)`; free_qty from source location.

### R8 — _get_qty_to_order (min/max reorder) [AXIS-A]
- stock_orderpoint.py:L461-476 — trigger if `qty_forecast < product_min_qty` (float_compare w/ UoM rounding); `qty_to_order = max(min,max) - (virtual_available(to=lead_horizon) + qty_in_progress)`; round UP to replenishment_uom multiple. qty_to_order_manual overrides (cleared when trigger=auto).

### R9 — _compute_deadline_date [HYBRID → SAVANT]
- stock_orderpoint.py:L123-178 — fast path qty_on_hand<min ⇒ today; else simulate daily net flow within horizon, first day below min ⇒ deadline = day - lead_days; lead_days = rule delays + horizon_time. Simulation AXIS-A; optimal timing under demand/supplier uncertainty heuristic.
- `SAVANT: name=ReorderTimingAdvisor family=None reasoning=NextBestAction inference=Induction semiring=NarsTruth style=Analytical — horizon_days/lead_days should be evidence-weighted (demand variability, seasonality, supplier reliability), not static config.`

### R10 — get_horizon_days [AXIS-A]
- stock_orderpoint.py:L811-817; res_company.py:L44-47 — context.global_horizon_days else company.horizon_days (Float, default 365).

### R11 — _procure_orderpoint_confirm (scheduler batch) [AXIS-A]
- stock_orderpoint.py:L707-794 — batch 1000; date = lead_horizon noon company-tz→UTC, minus global_horizon_days; rule.run in savepoint; ProcurementException → activity on product template; per-batch commit (new cursor).

### R12 — _run_scheduler_tasks [AXIS-A]
- stock_rule.py:L691-723 — compute qty_to_order → deadline → procure_confirm → assign confirmed moves (reservation_date<=today OR picking_type.reservation_method=at_confirm) batch 1000 → quant merge.

### R13 — lot/serial uniqueness [AXIS-A]
- stock_lot.py:L103-126 — unique (product_id, company_id, name); company_id NULL lots checked cross-company. DB partial unique index + app cross-company check.

### R14 — generate_lot_names / _get_next_serial [AXIS-A]
- stock_lot.py:L72-101 — find last numeric segment (regex \d+), increment count, zfill to original width; preserve prefix/suffix.

### R15 — _get_orderpoint_action (replenishment report) [AXIS-B → SAVANT]
- stock_orderpoint.py:L492-628 — virtual_available across replenishment locations; for forecast<0 compute lead days, re-read at horizon, subtract qty_in_progress; create/update manual orderpoints. Which shortfalls are actionable vs noise = judgment.
- `SAVANT: name=ReplenishmentReportAdvisor family=None reasoning=NextBestAction inference=Induction semiring=NarsTruth style=Analytical — distinguish real shortfalls from demand noise via demand-pattern inference.`

### R16 — equal-sequence route tiebreak [AXIS-B → SAVANT]
- stock_rule.py:L586-611 — product-specific routes sort first; equal-sequence order is arbitrary (Python sorted stability).
- `SAVANT: name=RouteTiebreaker family=None reasoning=NextBestAction inference=Abduction semiring=NarsTruth style=Analytical — equal-sequence tiebreak should weigh supplier lead/cost/capacity.`

## SVL interface contract (build fresh in woa-rs)
```
stock.move.done → create SVL { product, company, quantity(signed), unit_cost, value=qty*unit_cost (round@currency), remaining_qty/value (FIFO), move_id }
              → GL: receipt Dr stock_input Cr inventory_valuation ; delivery Dr inventory_valuation Cr stock_output
Cost methods (define enum): standard (=standard_price) | average (running avg = (old_qty*old_price+in_qty*in_price)/total, Decimal HALF_UP) | fifo (oldest layer; vacuum on later bill price)
Anglo-saxon: delivery Dr COGS_interim Cr inventory_valuation ; bill reverses + actual + price_diff→price_difference_account_id
Continental (DE default): delivery Dr stock_output Cr inventory_valuation ; no invoice deferral
```

## Enterprise gaps
- `stock_account` (SVL, FIFO/AVCO/standard engine, GL bridge, property_cost_method): absent → fresh.
- `stock_landed_costs`: absent.
- `mrp` (_run_manufacture route action): absent.

## Open questions
1. AVCO rounding mode (Decimal HALF_UP @ currency precision — confirm from stock_account).
2. FIFO vacuum layer-depletion algorithm (sort by create_date) — source from stock_account.
3. qty_to_order_computed store=true vs recompute-on-demand.
4. mts_else_mto cross-move consumption context (session-scoped, pass as mutable struct).
5. _quantity_in_progress extension trait (purchase_stock overrides to count open RFQs).
6. lot_sequence_id.next_by_id → DB sequence per product.
7. purchase_delay key set by purchase addon — handle missing gracefully.

## Depth-proof footer
```
Read: stock/models/stock_rule.py lines=747 depth=full
Read: stock/models/stock_orderpoint.py lines=817 depth=full
Read: stock/models/stock_lot.py lines=431 depth=full
Read: stock/models/product.py lines=1389 depth=full
Read: stock/models/res_company.py lines=215 depth=full
Read: product/models/product_category.py lines=69 depth=full
Read: stock/models/stock_move.py lines=2682 depth=targeted (L1546-1679, L2101-2160)
Read: stock/models/__init__.py lines=30 depth=full (confirms SVL absent)
```
