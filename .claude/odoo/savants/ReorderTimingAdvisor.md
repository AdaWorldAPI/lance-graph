# Savant: ReorderTimingAdvisor  (id 12 · family None · lane L13)

**Tuple:** kind=NextBestAction · inference=Induction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** NextBestActionReasoner   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:31`). Induction -> `QueryStrategy::CamWide`.

> **STATUS: NEEDS-INPUT** — the discriminating signal is **demand variability / seasonality over
> time**, which is NOT a community `stock` field. Community `stock` exposes only the static
> `horizon_days` (Float, default 365, `res_company.py:L44-47`) and per-rule `lead_days`. Inferring
> demand noise requires a woa-rs **movement-history feed** (a time series of past `stock.move`
> consumption). Slot 1 below is the schema the impl consumes once that feed exists; until then the
> reasoner can only echo the static-horizon AXIS-A timing.

## What it decides (AXIS-B core)
`_compute_deadline_date` (L13 R9, `stock_orderpoint.py:L123-178`) computes the reorder deadline:
fast path `qty_on_hand < min => today`; else simulate daily net flow within the horizon, take the
first day below `min`, and subtract `lead_days`. The simulation is deterministic given fixed inputs
(AXIS-A). The ambiguity the savant owns: **the horizon and lead-time used in that simulation are
static config** (`horizon_days` default 365, fixed `lead_days`), so the deadline ignores that real
demand is variable and seasonal. The savant induces an **evidence-weighted reorder timing** — pulling
the deadline earlier when recent demand is volatile or trending up, later when demand is stable —
instead of trusting a flat horizon. Output is a NARS-weighted timing suggestion, never an
orderpoint write.

## Deterministic guard (AXIS-A — stays in woa-rs)
The whole net-flow simulation in `_compute_deadline_date` (`stock_orderpoint.py:L123-178`, L13:57-58
R9): fast-path `qty_on_hand < min => today`, daily simulation, first-day-below-min, `deadline = day -
lead_days`. Also `_get_qty_to_order` (R8, `stock_orderpoint.py:L461-476`: trigger if `qty_forecast <
product_min_qty`, `qty_to_order = max(min,max) - (virtual_available + qty_in_progress)`, round up to
UoM multiple) and `get_horizon_days` (R10, `stock_orderpoint.py:L811-817`). The savant fires only on
the residual "is the *static* horizon/lead the right timing under *this* product's demand pattern?"

## Slot 1 — Evidence (Arrow EvidenceRef)
Primary table = the orderpoint joined to a (REQUIRED, not yet available) movement-history series for
the product. `EvidenceRef { table: "stock_orderpoint.timing_evidence", schema_fingerprint, rows }`,
one row per orderpoint (with the demand series carried as the history feed):

| column | dtype | signal |
|---|---|---|
| `orderpoint_id` | `Int64` | identity of the reorder point |
| `product_id` | `Int64` | product the orderpoint governs |
| `location_id` | `Int64` | warehouse/location partition |
| `product_min_qty` | `Float64` | the reorder trigger floor (static) |
| `product_max_qty` | `Float64` | the reorder ceiling (static) |
| `qty_forecast` | `Float64` | current `virtual_available` at the horizon; the trigger axis |
| `qty_on_hand` | `Float64` | current on-hand (fast-path discriminator: `< min => today`) |
| `horizon_days` | `Float64` | **static config (default 365)** — the only horizon signal community `stock` provides; insufficient — flagged NEEDS-INPUT |
| `lead_days` | `Float64` | static rule delays + horizon_time; the only lead signal |
| `demand_mean_per_day` | `Float64`/nullable | **REQUIRES woa-rs movement-history feed** — mean daily consumption from past `stock.move`; sets the baseline draw rate |
| `demand_stddev_per_day` | `Float64`/nullable | **REQUIRES movement-history feed** — demand variability; the dominant discriminator (high noise => reorder earlier) |
| `demand_trend` | `Float64`/nullable | **REQUIRES movement-history feed** — slope of recent consumption (up-trend => earlier deadline) |
| `seasonality_index` | `Float64`/nullable | **REQUIRES movement-history feed** — seasonal multiplier for the period the deadline falls in |

The four nullable demand columns are the NEEDS-INPUT core: without a movement-history time series the
reasoner sees only static `horizon_days` / `lead_days` and cannot weigh demand noise.

## Slot 2 — Odoo field → signal map                 (cite L-doc file:lines)
- `product_min_qty`, `product_max_qty`, `qty_forecast`, `qty_to_order` math <- `_get_qty_to_order` -- `L13-STOCK-VALUATION-PROCUREMENT.md:54-55` (R8; `stock_orderpoint.py:L461-476`).
- `qty_on_hand` fast-path + the daily net-flow simulation + `lead_days` <- `_compute_deadline_date` -- `L13-STOCK-VALUATION-PROCUREMENT.md:57-58` (R9; `stock_orderpoint.py:L123-178`).
- `horizon_days` (static default 365) <- `get_horizon_days` reading `company.horizon_days` -- `L13-STOCK-VALUATION-PROCUREMENT.md:61-62` (R10; `stock_orderpoint.py:L811-817`, `res_company.py:L44-47`).
- `demand_mean_per_day`, `demand_stddev_per_day`, `demand_trend`, `seasonality_index` <- **NO community `stock` field**. R9 itself flags it: "optimal timing under demand/supplier uncertainty heuristic" and the SAVANT note says "horizon_days/lead_days should be evidence-weighted (demand variability, seasonality, supplier reliability), not static config" (`L13-STOCK-VALUATION-PROCUREMENT.md:58-59`). The only history Odoo stores is the `stock.move` ledger itself; deriving a per-product demand series from it is the woa-rs feed. => **NEEDS-INPUT**.

## Slot 3 — Property-level alignment
N/A — family None (needs alignment axiom); no property IRIs defined.

`stock.warehouse.orderpoint -> fibo:Obligation` is a proposed *class*-level pivot only
(`L13-STOCK-VALUATION-PROCUREMENT.md:23`); no property IRI exists for demand-variability /
seasonality. Do not invent IRIs.

## Slot 4 — AXIS-B decision in evidence terms
Let E = the orderpoint row + its demand-history series (slot 1) for one `(product_id, location_id)`.

-> Conclusion C = `ReorderDeadline(orderpoint_id, suggested_date)` — the timing that best balances
stockout risk against early-order cost — emitted with NARS `(frequency, confidence)` where:
- **frequency** of "reorder earlier than the static deadline" rises with high
  `demand_stddev_per_day`, a positive `demand_trend`, and a `seasonality_index > 1` over the lead
  window; falls toward the static deadline when demand is stable, flat, and out of season.
- **confidence** is **structurally capped low** while the movement-history feed is absent — with only
  static `horizon_days` / `lead_days` the reasoner cannot estimate variability and defers to the
  AXIS-A simulated deadline. Confidence rises with the length/quality of the supplied demand series.
  Capped by the phi-1 humility ceiling.

Discriminating features (ranked, per the procurement family — demand-noise axis): demand variability
(stddev) >> demand trend > seasonality > static horizon/lead. Induction generalises the timing from
the realised consumption series (CamWide).

## Parity / GoBD notes
Suggestion-only (Iron Rule 7): the savant proposes a timing; it never writes `qty_to_order_manual`,
the orderpoint, or a procurement. The static-horizon simulation stays authoritative in woa-rs; the
suggestion is applied only behind it. No GoBD surface (pre-ledger replenishment). **NEEDS-INPUT**
must be surfaced: until the movement-history feed exists, the reasoner returns a low-confidence echo
of the static-horizon deadline.
