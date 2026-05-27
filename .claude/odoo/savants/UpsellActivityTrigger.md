# Savant: UpsellActivityTrigger  (id 22 · family 0x81 · lane L6)

**Tuple:** kind=NextBestAction · inference=Induction · semiring=NarsTruth · style=Exploratory
**Feeds Reasoner impl:** `NextBestActionReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::NextBestAction` -> "induce the action with the highest expected value"
> (`examples/savant_dispatch.rs:32`). Induction -> `QueryStrategy::CamWide` via
> `InferenceType::Induction::default_strategy()`. Style Exploratory inherited from 0x81
> SmbFoundryInvoice (a document-basin next-action recommendation has Exploratory character per L6 S-5,
> not the family's Direct default).

## What it decides (AXIS-B core)
When a sale order line has `invoice_status == 'upselling'` (delivered more than ordered on an
order-policy line), odoo auto-creates a TODO activity for the salesperson (L6 S-5,
`sale_order_line.py:L1964-1975`). The detection (`qty_delivered > qty_ordered`) is deterministic; the
AXIS-B core is **whether this upselling signal is worth acting on, and what the next best action is** --
induce, from lines-like-this, whether to (a) schedule a salesperson TODO, (b) auto-create an upsell
order line, or (c) suppress (noise: tiny over-delivery, a line that will be reconciled by a refund).
Output is a recommended next-best-action with NARS `(frequency, confidence)`; woa-rs schedules the
activity (or not) behind its guard -- it never bills the extra quantity automatically.

## Deterministic guard (AXIS-A -- stays in woa-rs)
The `'upselling'` status itself is deterministic: line `_compute_invoice_status` step 4 --
`state == 'sale' AND invoice_policy == 'order' AND qty >= 0.0 AND float_compare(qty_delivered,
product_uom_qty) == 1` (delivered MORE than ordered) -> `'upselling'`
(`L6-SALE-PURCHASE.md:351-360`, S-5; `sale_order_line.py:L1066-1095`), rolled up to the order
(`L6-SALE-PURCHASE.md:364-378`, S-5). The activity-creation hook fires on the status transition
(`_create_upsell_activity`, skipped if `mail_activity_automation_skip` in context,
`L6-SALE-PURCHASE.md:380-381`). The `float_compare` precision comes from
`decimal.precision.precision_get('Product Unit')` (not hardcoded, `L6-SALE-PURCHASE.md:362, 842`). The
detection and the precision gate stay in woa-rs; the savant decides what to *do* about a detected
upsell.

## Slot 1 -- Evidence (Arrow EvidenceRef)
Primary table the upsell line context, `EvidenceRef { table: "sale_order_line.upsell_context", schema_fingerprint, rows }`
(one row = a line flagged `'upselling'`, or its order rollup):

| column | dtype | signal |
|---|---|---|
| `order_line_id` | `Int64` | the line the action is scoped to |
| `order_id` | `Int64` | parent order (activity is scheduled on the order / its salesperson) |
| `product_id` | `Int64` | what was over-delivered (induction axis: upsell history for this product) |
| `invoice_status` | `Utf8` (`upselling\|to invoice\|invoiced\|no`) | the AXIS-A flag (must be `upselling` to delegate) |
| `product_uom_qty` | `Float64` | ordered quantity (baseline) |
| `qty_delivered` | `Float64` | delivered quantity; `qty_delivered - product_uom_qty` is the over-delivery magnitude |
| `qty_invoiced` | `Float64` | already-invoiced quantity (an over-delivery already billed is not an open upsell) |
| `invoice_policy` | `Utf8` (`order\|delivery`) | guard echo (only `order` policy lines reach `upselling`) |
| `user_id` | `Int64` | salesperson the TODO targets (consistency / workload signal) |
| `customer_rank` | `Int64` | partner maturity (an established customer is a better upsell target) |

The over-delivery magnitude `over_delivered = qty_delivered - product_uom_qty` is derived in woa-rs
(AXIS-A) and passed as a column so the reasoner can weight noise vs real upsell.

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- line `_compute_invoice_status` `'upselling'` rule (`qty_delivered > product_uom_qty` on order-policy lines) <- `L6-SALE-PURCHASE.md:351-360` (S-5; `sale_order_line.py:L1066-1095`).
- INVOICE_STATUS value space (`upselling`/`invoiced`/`to invoice`/`no`) <- `L6-SALE-PURCHASE.md:341-349` (S-5; `sale_order_line.py:L19-24`).
- order-level rollup (ALL `invoiced` or `upselling` -> `upselling`) <- `L6-SALE-PURCHASE.md:364-378` (S-5; `sale_order.py:L618-664`).
- upsell-activity auto-creation hook on status transition (`_create_upsell_activity`, `mail_activity_automation_skip` mute) <- `L6-SALE-PURCHASE.md:380-381` (S-5 side effect; `sale_order_line.py:L1964-1975`).
- `qty_to_invoice` / `qty_invoiced` derivation (refunds decrease invoiced; `round=False`) <- `L6-SALE-PURCHASE.md:268-309` (S-4; `sale_order_line.py:L972-1064`).
- `invoice_policy` order-vs-delivery semantics <- `L6-SALE-PURCHASE.md:314-315` (S-4).
- `Product Unit` precision for `float_compare` <- `L6-SALE-PURCHASE.md:362, 842` (S-5 / porter gotcha 13).
- delegation tuple `(NextBestAction, Induction, NarsTruth, Exploratory)` <- `L6-SALE-PURCHASE.md:387` (S-5 Axis-2).

## Slot 3 -- Property-level alignment
N/A -- class-level pivots only; no `owl:equivalentProperty` defined. (Confirmed: `odoo_alignment.rs`
holds only class-level `owl:equivalentClass` rows; zero property IRIs in the repo.) `sale.order` and
`sale.order.line` are **unmapped** today (`resolve_odoo` -> `None`; L6 proposes `ubl:Order` /
`ubl:OrderLine` -> 0x81 SmbFoundryInvoice but no alignment row exists yet,
`L6-SALE-PURCHASE.md:87-94, 761-766`). The created `account.move` (if the action ever bills) ->
`fibo:Transaction` (class-level, `odoo_alignment.rs:221-229`). No property crosses the seam at
decision time. **N/A -- decision is over sale-order quantities; the sale.order Layer-2 axiom is
lance-graph follow-on (not sourceable from L6).**

## Slot 4 -- AXIS-B decision in evidence terms
Let E = the upsell line context rows (slot 1) with derived `over_delivered`, restricted to
`invoice_status == 'upselling'`.

-> Conclusion C = `RecommendUpsellAction(order_line_id, { schedule_todo | create_upsell_line | suppress })`
emitted with NARS `(frequency, confidence)` where:
- **frequency** of recommending an action (schedule a salesperson TODO) rises with: a material
  `over_delivered` magnitude relative to `product_uom_qty` (not a rounding sliver), a product with a
  history of accepted upsells (the inductive "lines like this" signal), an established customer
  (`customer_rank`), and `qty_invoiced` still at the ordered level (the extra is genuinely un-billed).
- **frequency** collapses toward `suppress` for tiny over-deliveries, lines where the over-delivery is
  already invoiced, or products that historically get refunded rather than upsold.
- **confidence** is the NARS evidence weight from how many comparable upsell outcomes have been
  observed for this product/segment; a novel product keeps confidence low even when frequency is high.
  Capped by phi-1.

Discriminating features (ranked): `over_delivered` magnitude vs ordered qty >> product upsell history
> `customer_rank` > `qty_invoiced` gap > `user_id` workload. Induction is literally "over-deliveries
like this one have historically converted to a successful upsell," which is why the action (not the
detection) is delegated (L6 S-5 rationale).

## Parity / GoBD notes
Upselling is a sales-CRM next-action, entirely outside the GoBD ledger -- scheduling a TODO touches no
posted entry and carries no Festschreibung weight. Suggestion-only per Iron Rule 7: the savant
recommends scheduling an activity (or creating an upsell line); it must **never** auto-bill the
over-delivered quantity -- billing requires a confirmed order-line change and a human, since
over-delivery that is silently invoiced is a customer-dispute and revenue-recognition risk. The
`mail_activity_automation_skip` context mute (S-5) is honoured as a hard suppression. woa-rs has no
line-level `qty_delivered`/`qty_invoiced` today (L6 S-4 GAP: MAJOR) -- so this savant is dormant until
woa-rs grows a `vorgang_line` model with delivery/invoice quantity tracking; flagged, not blocking the
contract.
