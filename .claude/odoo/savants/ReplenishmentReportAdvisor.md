# Savant: ReplenishmentReportAdvisor  (id 13 · family None · lane L13)

**Tuple:** kind=NextBestAction · inference=Induction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** NextBestActionReasoner   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:31`). Induction -> `QueryStrategy::CamWide`.

> **STATUS: NEEDS-INPUT** — distinguishing a *real* shortfall from *demand noise* needs a
> demand-pattern history, which is NOT a community `stock` field (same root cause as
> ReorderTimingAdvisor(12): only static `horizon_days` default 365 + `lead_days`). Demand-noise
> inference requires a woa-rs **movement-history feed**. Slot 1 is the schema the impl consumes once
> that feed exists; until then the reasoner can only echo the raw report rows.

## What it decides (AXIS-B core)
`_get_orderpoint_action` (L13 R15, `stock_orderpoint.py:L492-628`) builds the replenishment report:
`virtual_available` across replenishment locations, and for any `forecast < 0` it computes lead days,
re-reads at the horizon, subtracts `qty_in_progress`, and creates/updates manual orderpoints. The
arithmetic is deterministic (AXIS-A). The ambiguity the savant owns: **which negative-forecast rows
are genuine, actionable shortfalls versus transient demand noise** (a one-off spike, a return about
to land, a forecast wobble that self-corrects). The savant induces an actionability score per report
line from the product's demand pattern, so woa-rs surfaces only real shortfalls and suppresses noise.
Output is a NARS-weighted per-line actionability judgement, never an orderpoint write.

## Deterministic guard (AXIS-A — stays in woa-rs)
The whole report computation in `_get_orderpoint_action` (`stock_orderpoint.py:L492-628`, L13:76-77
R15): `virtual_available` across replenishment locations, lead-day computation for `forecast < 0`,
re-read at horizon, `qty_in_progress` subtraction, manual-orderpoint create/update. The batch
scheduler `_procure_orderpoint_confirm` (R11, `stock_orderpoint.py:L707-794`) and `_run_scheduler_tasks`
(R12, `stock_rule.py:L691-723`) are also AXIS-A. The savant fires only on the residual
"is this negative forecast signal or noise?" per line.

## Slot 1 — Evidence (Arrow EvidenceRef)
Primary table = the replenishment-report lines joined to a (REQUIRED, not yet available)
demand-history series. `EvidenceRef { table: "stock_orderpoint.replenishment_lines", schema_fingerprint, rows }`,
one row per report line (a product/location with a computed shortfall):

| column | dtype | signal |
|---|---|---|
| `orderpoint_id` | `Int64` | identity of the (possibly auto-created) orderpoint line |
| `product_id` | `Int64` | product the line concerns |
| `location_id` | `Int64` | replenishment location partition |
| `qty_forecast` | `Float64` | `virtual_available` at the horizon; **negative => a candidate shortfall** |
| `qty_in_progress` | `Float64` | already-inbound qty subtracted from the shortfall (R15) |
| `product_min_qty` | `Float64` | reorder floor; context for whether the shortfall is structural |
| `lead_days` | `Float64` | lead window over which the shortfall is measured (static) |
| `horizon_days` | `Float64` | **static config (default 365)** — the only horizon signal; insufficient — flagged NEEDS-INPUT |
| `demand_mean_per_day` | `Float64`/nullable | **REQUIRES woa-rs movement-history feed** — baseline draw; a shortfall far below mean draw is more likely real |
| `demand_stddev_per_day` | `Float64`/nullable | **REQUIRES movement-history feed** — the noise band; a forecast dip *within* one stddev is likely noise, **the dominant discriminator** |
| `recent_spike_flag` | `Boolean`/nullable | **REQUIRES movement-history feed** — was the negative forecast driven by a single anomalous order? (noise, not trend) |
| `pending_return_qty` | `Float64`/nullable | **REQUIRES movement feed** — inbound returns not yet in `qty_in_progress` that would erase the shortfall |

The four nullable demand columns are the NEEDS-INPUT core: without the demand series the reasoner
cannot tell a real shortfall from a wobble inside the noise band.

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `qty_forecast` (virtual_available), `qty_in_progress`, the lead-day re-read for `forecast < 0` <- `_get_orderpoint_action` -- `L13-STOCK-VALUATION-PROCUREMENT.md:76-77` (R15; `stock_orderpoint.py:L492-628`).
- `product_min_qty` trigger context <- `_get_qty_to_order` -- `L13-STOCK-VALUATION-PROCUREMENT.md:54-55` (R8; `stock_orderpoint.py:L461-476`).
- `lead_days`, `horizon_days` (static) <- `get_horizon_days` / `_compute_deadline_date` -- `L13-STOCK-VALUATION-PROCUREMENT.md:61-62` (R10; `stock_orderpoint.py:L811-817`) and `L13-STOCK-VALUATION-PROCUREMENT.md:57-58` (R9).
- `demand_mean_per_day`, `demand_stddev_per_day`, `recent_spike_flag`, `pending_return_qty` <- **NO community `stock` field**. R15 itself flags it: "Which shortfalls are actionable vs noise = judgment" and the SAVANT note says "distinguish real shortfalls from demand noise via demand-pattern inference" (`L13-STOCK-VALUATION-PROCUREMENT.md:77-78`). The demand series must be derived from the `stock.move` ledger by woa-rs. => **NEEDS-INPUT**.

## Slot 3 — Property-level alignment
N/A — family None (needs alignment axiom); no property IRIs defined.

`stock.warehouse.orderpoint -> fibo:Obligation` is a proposed *class*-level pivot only
(`L13-STOCK-VALUATION-PROCUREMENT.md:23`); no property IRI exists for demand-pattern attributes. Do
not invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the replenishment-report rows + their demand-history series (slot 1).

-> Conclusion C = `ActionableShortfall(orderpoint_id)` — the report lines that represent genuine,
act-now shortfalls — emitted with NARS `(frequency, confidence)` where:
- **frequency** rises for a line whose negative `qty_forecast` exceeds the demand noise band
  (magnitude >> `demand_stddev_per_day`), with `demand_mean_per_day` confirming sustained draw, no
  `recent_spike_flag`, and no `pending_return_qty` to erase it.
- **frequency** falls toward "noise, suppress" for a small dip within one stddev, a one-off
  `recent_spike_flag`, or an inbound return covering the gap.
- **confidence** is **structurally capped low** while the movement-history feed is absent — with only
  the raw forecast and static horizon the reasoner cannot estimate the noise band and defers to
  surfacing the raw report. Confidence rises with the demand-series length/quality. Capped by the
  phi-1 humility ceiling.

Discriminating features (ranked, demand-noise axis): shortfall magnitude vs demand stddev (noise
band) >> sustained mean draw > absence of a recent spike > absence of pending returns. Induction
generalises actionability from the realised demand series (CamWide).

## Parity / GoBD notes
Suggestion-only (Iron Rule 7): the savant scores report lines; it never creates/updates an
orderpoint or fires a procurement. The deterministic report stays authoritative in woa-rs; the
actionability score only filters what a human sees. No GoBD surface (pre-ledger). **NEEDS-INPUT**
must be surfaced: until the movement-history feed exists, the reasoner returns a low-confidence
pass-through of the raw report rows.
