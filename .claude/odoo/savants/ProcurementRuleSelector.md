# Savant: ProcurementRuleSelector  (id 11 · family None · lane L13)

**Tuple:** kind=NextBestAction · inference=Induction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** NextBestActionReasoner   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:31`). Induction -> `QueryStrategy::CamWide`.

> **STATUS: NEEDS-INPUT** — the discriminating signals (supplier reliability, historical
> lead-time, cost, capacity) are NOT community `stock` fields. Community `stock.rule` carries only a
> static `delay` (Integer days). The supplier/cost feed lives in the `purchase` addon, which is
> **absent** from this clone (L13:5-6, open Q#5/#7). Slot 1 below is the schema the impl will consume
> **once woa-rs supplies a purchase feed**; until then the reasoner can only echo the AXIS-A
> deterministic order (equal-sequence => arbitrary) and must report low confidence.

## What it decides (AXIS-B core)
`_get_rule` (L13 R3, `stock_rule.py:L564-638`) builds the location-ancestor chain and selects rules
`ORDER BY (route_sequence, sequence)` with a fixed priority cascade (values.route_ids -> packaging ->
product|categ routes -> warehouse routes), **first match wins**. The walk + the priority cascade are
deterministic (AXIS-A). The residual ambiguity the savant owns: **when two or more candidate rules
tie on `(route_sequence, sequence)`**, which rule is the best next action to procure through? Odoo
breaks the tie by Python `sorted` stability (arbitrary insertion order). The savant re-ranks the
tied set by **expected fulfilment value** — weighing supplier lead-time, current stock availability
at the source location, and supplier reliability — instead of an arbitrary stable-sort artefact.
Output is a NARS-weighted preference over the tied rules, never an un-guarded route write.

## Deterministic guard (AXIS-A — stays in woa-rs)
The whole of `_get_rule`: location-ancestor chain construction, `_search_rule_for_warehouses`
grouped by `(dest, warehouse, route)`, the route-priority cascade, and first-match-wins
(`stock_rule.py:L564-638`, L13:38-39 R3). The savant is invoked **only** for the residual
equal-`(route_sequence, sequence)` tie — everything with a strict ordering is resolved in woa-rs
before delegation. The `rule.delay` subtraction in `_get_stock_move_values` (R5,
`stock_rule.py:L324-408`) is also AXIS-A.

## Slot 1 — Evidence (Arrow EvidenceRef)
Primary table = the tied candidate-rule set joined to its source-location stock and (REQUIRED, not
yet available) a supplier feed. `EvidenceRef { table: "stock_rule.tied_candidates", schema_fingerprint, rows }`,
one row per rule that ties on `(route_sequence, sequence)`:

| column | dtype | signal |
|---|---|---|
| `rule_id` | `Int64` | identity of the candidate rule |
| `route_id` | `Int64` | the route this rule belongs to (the tie partition) |
| `route_sequence` | `Int64` | tie key part 1 — rows in one evidence batch share this value |
| `sequence` | `Int64` | tie key part 2 — rows in one evidence batch share this value (=> the tie) |
| `action` | `Utf8` (`pull\|push\|pull_push\|buy\|manufacture`) | the procurement action; `buy` rows are the ones that need the supplier feed |
| `delay` | `Int64` (days) | **the ONLY lead-time signal community `stock` provides** (static rule delay); insufficient alone — flagged NEEDS-INPUT |
| `location_src_id` | `Int64` | source location whose on-hand/forecast availability discriminates the tie |
| `free_qty_at_src` | `Float64` | **REQUIRES woa-rs feed** — available qty at `location_src_id` (R7 free_qty, `stock_move.py:L1546-1678`); a rule sourcing from a stocked location beats one needing replenishment |
| `supplier_id` | `Int64`/nullable | **REQUIRES purchase feed (absent)** — the vendor behind a `buy` rule |
| `hist_lead_time_days` | `Float64`/nullable | **REQUIRES purchase feed (absent)** — realised historical lead time per supplier; the true lead-time signal `delay` only approximates |
| `supplier_reliability` | `Float64`/nullable | **REQUIRES purchase feed (absent)** — on-time-delivery frequency per supplier; the dominant reliability discriminator |
| `unit_cost` | `Float64`/nullable | **REQUIRES purchase feed (absent)** — cost per unit through this rule; secondary discriminator |

The four nullable supplier/availability columns are the NEEDS-INPUT core: without them the reasoner
sees only `delay` (static, identical across most rules) and cannot discriminate.

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `route_sequence`, `sequence` (the tie keys) <- `stock.rule` ordering in `_search_rule_for_warehouses` -- `L13-STOCK-VALUATION-PROCUREMENT.md:38-39` (R3; `stock_rule.py:L564-638`) and `L13-STOCK-VALUATION-PROCUREMENT.md:80-82` (R16; `stock_rule.py:L586-611`, "equal-sequence order is arbitrary, Python sorted stability").
- `delay` (the only lead-time field) <- `stock.rule.delay` used as `date = date_planned - rule.delay` in `_get_stock_move_values` -- `L13-STOCK-VALUATION-PROCUREMENT.md:45-46` (R5; `stock_rule.py:L324-408`).
- `action` / `procure_method` routing buckets <- `_action_confirm` -- `L13-STOCK-VALUATION-PROCUREMENT.md:51-52` (R7; `stock_move.py:L1546-1678`).
- `free_qty_at_src` <- the `free_qty` source-location availability in the mts_else_mto bucket: `qty_to_procure = max(product_qty - free_qty, 0)` -- `L13-STOCK-VALUATION-PROCUREMENT.md:51-52` (R7; `stock_move.py:L1546-1678`). woa-rs must surface this as a column.
- `supplier_id`, `hist_lead_time_days`, `supplier_reliability`, `unit_cost` <- **NO community `stock` field**. L13 confirms `purchase`/`purchase_stock` is absent (`L13-STOCK-VALUATION-PROCUREMENT.md:5-6`); `_quantity_in_progress` is overridden by `purchase_stock` to count open RFQs (`L13-STOCK-VALUATION-PROCUREMENT.md:103` open Q#5); `purchase_delay` is set by the purchase addon and must be "handled gracefully" when missing (`L13-STOCK-VALUATION-PROCUREMENT.md:105` open Q#7). => woa-rs purchase feed required; **NEEDS-INPUT**.

## Slot 3 — Property-level alignment
N/A — family None (needs alignment axiom); no property IRIs defined.

`stock.rule -> fibo:Agreement` and `stock.move -> fibo:Transfer` are proposed *class*-level pivots
only (`L13-STOCK-VALUATION-PROCUREMENT.md:21-28`); no `owl:equivalentProperty` / property IRI exists
in the repo for the lead-time / reliability / cost attributes this decision weighs. Do not invent
IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the tied candidate-rule rows (slot 1) sharing one `(route_sequence, sequence)` value.

-> Conclusion C = `PreferRule(rule_id)` — the rule with the highest expected procurement value —
emitted with NARS `(frequency, confidence)` where:
- **frequency** rises for a rule whose `supplier_reliability` is high, `hist_lead_time_days` is low,
  `free_qty_at_src` already covers the demand (no replenishment needed), and `unit_cost` is low.
- **frequency** falls for a rule sourcing from an empty location, a low-reliability / long-lead
  supplier, or a high cost.
- **confidence** is **structurally capped low** while the supplier feed is absent: with only static
  `delay` visible, the evidence cannot discriminate, so confidence sits near the prior and the
  reasoner effectively defers to the AXIS-A arbitrary order. Confidence rises only once woa-rs
  supplies `supplier_reliability` / `hist_lead_time_days` / `free_qty_at_src`. Capped by the phi-1
  humility ceiling.

Discriminating features (ranked, per the procurement family): supplier reliability >> historical
lead-time > source-location availability > unit cost > static `delay`. Induction generalises the
best-value action from the realised supplier/availability history (CamWide).

## Parity / GoBD notes
Suggestion-only (Iron Rule 7): the savant re-ranks tied rules; it never creates, reorders, or writes
a `stock.rule` or a procurement move. The deterministic walk + first-match-wins stays authoritative
in woa-rs; the savant's preference is applied only behind that guard. No GoBD Festschreibung surface
here (procurement routing is pre-ledger). **NEEDS-INPUT** must be surfaced to the caller: until the
purchase feed exists, the reasoner returns a low-confidence echo of the AXIS-A order rather than a
confident re-rank.
