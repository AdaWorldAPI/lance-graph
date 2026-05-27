RICHNESS-LANE-OK

# L6 — Sales + Purchase Order Flow → Invoice Creation (Vorgang Lifecycle)

**Lane:** Sales + Purchase order flow → invoice creation (quote → order → invoice in odoo terms;
Vorgang lifecycle: Angebot → Auftrag → Rechnung)

**Written:** 2026-05-26

---

## 1. Scope + Odoo Files Read

| File | Lines | Depth |
|---|---|---|
| `/home/user/odoo/addons/sale/models/sale_order.py` | 2301 | full |
| `/home/user/odoo/addons/sale/models/sale_order_line.py` | 1819 | full |
| `/home/user/odoo/addons/purchase/models/purchase_order.py` | 1418 | full |

**Also read (calibration):**
- `/home/user/woa-rs/src/models/work_order.rs` (full, for woa-rs Vorgang shape)
- `/home/user/lance-graph/crates/lance-graph-callcenter/src/odoo_alignment.rs` (full, for existing alignment table)
- `/home/user/woa-rs/.claude/board/odoo-richness/BRIEFING.md` (full, output contract)

---

## 2. Per-Rule Sections

---

### Rule S-1: Sale Order State Machine

**Odoo source:** `sale_order.py:L26-31, L70-76, L1058-1065, L1156-1196, L1231-1233, L1318-1333`

#### Axis-1 Rich-AST Spec

**State values (SALE_ORDER_STATE constant, L26-31):**
```python
SALE_ORDER_STATE = [
    ('draft', "Quotation"),
    ('sent', "Quotation Sent"),
    ('sale', "Sales Order"),
    ('cancel', "Cancelled"),
]
```
Field declaration (L70-76): `state = fields.Selection(SALE_ORDER_STATE, default='draft', readonly=True, copy=False, index=True, tracking=3, group_expand=True)`

**Note: there is NO `'done'` state in community odoo sale module.** The briefing mentioned
`done` — that state existed in older versions. Community odoo 17 only has `draft/sent/sale/cancel`.
The `locked` boolean (L77-81) replaces what was previously a separate `done` state in older versions.

**Transitions:**

| From | To | Method | Guard |
|---|---|---|---|
| `draft` | `sent` | `action_quotation_sent()` L1156-1164 | state must be `draft`; raises UserError otherwise |
| `draft`/`sent` | `sent` (implicit) | `message_post()` L1714-1718 | context flag `mark_so_as_sent=True` |
| `draft`/`sent` | `sale` | `action_confirm()` L1166-1196 | calls `_confirmation_error_message()` L1203-1216; guards: state ∈ {`draft`,`sent`}; all non-display, non-downpayment lines must have `product_id` |
| `cancel`/`sent` | `draft` | `action_draft()` L1058-1065 | only processes orders in `cancel` or `sent`; also clears `signature`, `signed_by`, `signed_on` |
| `sale`/`draft`/`sent` | `cancel` | `action_cancel()` L1324-1328 | raises UserError if any order is `locked`; delegates to `_action_cancel()` |
| — | `locked` | `action_lock()` L1318-1319 | sets `locked=True` on the record |

**`action_confirm()` full control flow (L1166-1196):**
1. For each order: call `_confirmation_error_message()`. If non-falsy, raise UserError.
2. Call `self.order_line._validate_analytic_distribution()` (validates analytics before confirming).
3. Call `self.write(self._prepare_confirmation_values())` which sets `{'state': 'sale', 'date_order': fields.Datetime.now()}` — date_order is OVERWRITTEN to confirmation time.
4. Strip `default_name` and `default_user_id` from context (avoids propagation to linked record creation).
5. Call `self.with_context(context)._action_confirm()` — extensibility hook (empty in base sale module; overridden in sale_stock to create pickings etc.).
6. For each SO that `_should_be_locked()` returns True: call `action_lock()`. Lock guard: checks `sale.group_auto_done_setting` feature flag (L1198-1201).
7. If `send_email` in context: call `_send_order_confirmation_mail()`.

**`_action_cancel()` full flow (L1330-1333):**
1. Find all draft invoices linked to the order lines: `self.invoice_ids.filtered(lambda inv: inv.state == 'draft')`.
2. Cancel those draft invoices: `inv.button_cancel()`.
3. Write `{'state': 'cancel'}` on all orders.

**Deletion guard (L1032-1038):** `_unlink_except_draft_or_cancel()` — raises UserError if state not in `('draft', 'cancel')`.

**Write guard (L1040-1043):** `write()` raises UserError if `pricelist_id` is being changed on a confirmed (`sale`) order.

**SQL constraint (L41-44):** `_date_order_conditional_required` — DB-level CHECK: `(state = 'sale' AND date_order IS NOT NULL) OR state != 'sale'`. Confirmed orders must have `date_order`.

#### Axis Classification
**DETERMINISTIC** — pure state machine, all transitions are guarded/unguarded writes. No scoring or ranking. Port directly to Rust enum + transition table.

#### Ontology Mapping
`odoo:sale.order` — **UNRESOLVED** (not in ODOO_ALIGNMENTS table, `resolve_odoo("odoo:sale.order")` returns `None`).

**FLAG: Missing alignment row — proposal:**
- OWL pivot: `ubl:Order` (UBL 2.1 Order document, the canonical commercial order standard)
- OGIT family: `SmbFoundryInvoice` (0x81) — the document/transaction basin; a sale order is a pre-invoice commercial document in the same document lifecycle as `account.move`
- DOLCE: `Perdurant` (it is a process/event document with temporal extent, analogous to `.move`)
- Proposed row: `odoo:sale.order → ubl:Order → OGIT SmbFoundryInvoice (0x81) → DOLCE Perdurant`
- This aligns with woa-rs Vorgang which is the closest analog (see §woa-rs calibration)

#### K-Step
This is the **core ERP Vorgang lifecycle**, not directly a K3/K7/K8 step. It is the prerequisite document flow that gates K3 (double-entry posting happens when the invoice created from this order is posted) and K7 (USt triggers when invoice lines carry tax_ids).

#### woa-rs Target Module
`src/models/work_order.rs` — the existing `Model` with `doc_type` and `status` fields IS the woa-rs Vorgang. The odoo state machine maps as:
- odoo `draft` → woa-rs `status = 'draft'`
- odoo `sent` → woa-rs `status = 'open'` (Angebot verschickt)
- odoo `sale` → woa-rs `status = 'in_progress'` (Auftrag bestätigt)
- odoo `cancel` → woa-rs `status = 'cancelled'`
- odoo `locked` flag → woa-rs GoBD lock (`is_wo_gobd_locked()` in `src/gobd.rs`)

The Rust state machine logic belongs in a new `src/erp/sale_order_fsm.rs` or as methods on `work_order::Model`.

---

### Rule S-2: Amount Computation (_compute_amounts / _amount_all)

**Odoo source:** `sale_order.py:L512-528` (`_compute_amounts`); `purchase_order.py:L29-44` (`_amount_all`)

#### Axis-1 Rich-AST Spec

**`_compute_amounts` (sale, L512-528):**

`@api.depends('order_line.price_subtotal', 'currency_id', 'company_id', 'payment_term_id')`

Full control flow for each order:
1. `order_lines = order._get_priced_lines()` — returns `order_line.filtered(lambda x: not x.display_type)` (L509-510). Excludes section/note lines.
2. Build base_lines: `[line._prepare_base_line_for_taxes_computation() for line in order_lines]`
3. Append EPD (Early Payment Discount) lines: `base_lines += order._add_base_lines_for_early_payment_discount()` — only adds lines if `payment_term_id.early_discount AND early_pay_discount_computation == 'mixed' AND discount_percentage` (L530-567). These lines represent the discounted amount for tax computation.
4. `AccountTax._add_tax_details_in_base_lines(base_lines, order.company_id)` — computes tax breakdown per line using the company's tax engine.
5. `AccountTax._round_base_lines_tax_details(base_lines, order.company_id)` — applies rounding per `company.tax_calculation_rounding_method` (round_per_line vs round_globally).
6. `tax_totals = AccountTax._get_tax_totals_summary(base_lines, currency, company)` — returns dict with `base_amount_currency`, `tax_amount_currency`, `total_amount_currency`.
7. Assign: `order.amount_untaxed = tax_totals['base_amount_currency']`, `order.amount_tax = tax_totals['tax_amount_currency']`, `order.amount_total = tax_totals['total_amount_currency']`.

**`_amount_all` (purchase, L29-44):**
Identical pipeline to sale's `_compute_amounts` but:
- No EPD lines
- Adds `order.amount_total_cc = tax_totals['total_amount']` (company-currency total, not order-currency)
- `@api.depends('order_line.price_subtotal', 'company_id', 'currency_id')`

**Rounding critical notes:**
- All currency rounding goes through `company_id.currency_id` or `order.currency_id` as appropriate.
- The `company.tax_calculation_rounding_method` field (`tax_calculation_rounding_method`, L308-310 in sale_order) controls whether rounding is per-line or global. This is a **company-level setting** — porter must respect it.
- The actual arithmetic is inside `account.tax._add_tax_details_in_base_line` (lane L3 territory); from sale_order's perspective it is a black-box call that takes `price_unit * qty * (1-discount/100)` and returns tax-split amounts.
- EPD (Early Payment Discount mixed mode): creates virtual negative base lines for the discount amount, then a balancing positive line. This affects `amount_tax` and `amount_total` on the SO even before invoicing.

**`_prepare_base_line_for_taxes_computation` (sale_order_line, L816-837):**
Called per line; builds the dict:
```python
{
    'tax_ids': self.tax_ids,
    'quantity': self.product_uom_qty,
    'partner_id': self.order_id.partner_id,
    'currency_id': self.order_id.currency_id or company.currency_id,
    'rate': self.order_id.currency_rate,
    'name': self.name,
    # if _is_global_discount(): 'special_type': 'global_discount'
    # elif is_downpayment: 'special_type': 'down_payment'
}
```
This is then passed to `account.tax._prepare_base_line_for_taxes_computation(self, **base_values)` — the ORM model record (`self`) is the `record` argument so the tax engine can read `price_unit` and `discount` from it directly.

**`_compute_amount` (sale_order_line, L843-853):**
`@api.depends('product_uom_qty', 'discount', 'price_unit', 'tax_ids')`
Per line:
1. `base_line = line._prepare_base_line_for_taxes_computation()`
2. `AccountTax._add_tax_details_in_base_line(base_line, company)`
3. `AccountTax._round_base_lines_tax_details([base_line], company)`
4. `line.price_subtotal = base_line['tax_details']['total_excluded_currency']`
5. `line.price_total = base_line['tax_details']['total_included_currency']`
6. `line.price_tax = line.price_total - line.price_subtotal`

Note: `price_unit` already accounts for fiscal position tax mapping (see `_compute_price_unit`/`_reset_price_unit`). The line amount formula in simplified form: `price_unit * qty * (1 - discount/100)` before tax engine, but this is computed inside the tax engine not in sale_order_line.

#### Axis Classification
**DETERMINISTIC** — all arithmetic is deterministic given inputs. The tax engine is a black box from this lane's perspective (belongs to lane L3). Currency rounding is deterministic (company currency round). EPD mixed logic is a conditional computation, deterministic once payment_term_id is set.

#### Ontology Mapping
`odoo:sale.order` — see Rule S-1 (UNRESOLVED, propose ubl:Order → SmbFoundryInvoice 0x81 → Perdurant).

#### K-Step
Core ERP amount arithmetic. Feeds K3 (posting amounts), K7 (USt computation trigger), K8 (BWA line amounts). Not a standalone K-step but prerequisite to all.

#### woa-rs Target Module
woa-rs stores `netto_summe`, `brutto_summe` on `work_order::Model` (inferred from `home.rs` references to `rechnung_nr`/`bezahlt` and `partner_commission.rs::netto_summe_der_endkundenrechnung`). The odoo pipeline (prepare_base_line → add_tax_details → round → totals) should land in `src/erp/tax_compute.rs` or as part of the ERP crate.

**GAP:** woa-rs Vorgang has no concept of `qty_to_invoice`, partial invoicing amounts, or EPD. These are richer than what woa-rs currently tracks. The `amount_untaxed / amount_tax / amount_total` trio on the SO maps well but the EPD mixed-mode computation is entirely absent.

---

### Rule S-3: Line Amount — price_unit, discount, tax_ids

**Odoo source:** `sale_order_line.py:L162-188, L541-568, L586-633, L783-814, L843-853`

#### Axis-1 Rich-AST Spec

**`_compute_tax_ids` (L541-568):**
`@api.depends('product_id', 'company_id')`

Full control flow:
1. Group lines by company.
2. For each company group, for each line `with_company(company)`:
   a. If `product_type == 'combo'`: `line.tax_ids = False`, continue.
   b. If `product_id`: `taxes = product_id.taxes_id._filter_taxes_by_company(company)`.
   c. If no product_id or no taxes: `line.tax_ids = False`, continue.
   d. Cache key: `(fiscal_position.id, company.id, tuple(taxes.ids)) + _get_custom_compute_tax_cache_key()`.
   e. `result = fiscal_position.map_tax(taxes)` (maps taxes through fiscal position rules).
   f. `line.tax_ids = result`.

**`_compute_price_unit` (L586-617):**
`@api.depends('product_id', 'product_uom_id', 'product_uom_qty')`

Guard conditions that SKIP recomputation (price stays as-is):
- Line has no `order_id` (orphan line)
- `is_downpayment` is True
- `_is_global_discount()` is True (extra_tax_data starts with 'global_discount,')
- `not force_recompute AND has_manual_price(line)` — where `has_manual_price` checks `currency.compare_amounts(technical_price_unit, price_unit) != 0`. If user manually edited price away from pricelist price, it sticks.
- `qty_invoiced > 0` — price frozen once any quantity has been invoiced
- `product_id.expense_policy == 'cost' AND is_expense` — expense cost lines

If none of the guards match: call `line._reset_price_unit()`.

**`_reset_price_unit` (L619-633):**
1. `line = self.with_company(self.company_id)`
2. `price = line._get_display_price()` — gets pricelist-based price (before discount, for display)
3. `product_taxes = line.product_id.taxes_id._filter_taxes_by_company(line.company_id)`
4. `price_unit = line.product_id._get_tax_included_unit_price_from_price(price, product_taxes, fiscal_position=line.order_id.fiscal_position_id)` — adjusts price if company has price-include taxes
5. `line.update({'price_unit': price_unit, 'technical_price_unit': price_unit})`

**`technical_price_unit` (L182):** shadow field that tracks the "system-computed" price so user edits to `price_unit` can be detected. If `technical_price_unit != price_unit`, the price was manually edited.

**`_compute_discount` (L783-814):**
`@api.depends('product_id', 'product_uom_id', 'product_uom_qty')`

Control flow:
1. `discount_enabled = product.pricelist.item._is_discount_feature_enabled()` — requires the discount feature to be active.
2. If no product or display_type: `discount = 0.0`.
3. If no pricelist or discount feature not enabled or no product_uom_id: skip (no change).
4. If `combo_item_id`: `discount = line._get_linked_line().discount` (inherit parent combo's discount).
5. `line.discount = 0.0` (reset).
6. If `not pricelist_item_id._show_discount()`: continue (pricelist didn't specify a discount).
7. `pricelist_price = line._get_pricelist_price()`
8. `base_price = line._get_pricelist_price_before_discount()`
9. If `base_price != 0`: `discount = (base_price - pricelist_price) / base_price * 100`
10. Show discount only if: `(discount > 0 and base_price > 0) OR (discount < 0 and base_price < 0)` (negative discounts = surcharge, hidden unless price is also negative)
11. `line.discount = discount`

#### Axis Classification
**DETERMINISTIC** — pure pricelist/fiscal-position arithmetic. No scoring. The `has_manual_price` check and the `qty_invoiced > 0` freeze are deterministic guards.

**Axis-2 tag for pricelist selection (`_compute_pricelist_item_id`):** This resolves which pricelist rule to use. If we later need to recommend a "best pricelist" or "next-action on pricing" for a Vorgang, that is `ReasoningKind::NextBestAction, InferenceType::Synthesis, SemiringChoice::NarsTruth, ThinkingStyle cluster: Exploratory` (inherited from SmbFoundryInvoice document family → the commercial document negotiation angle is Exploratory). But pricelist rule selection within the odoo model itself (given that `pricelist_id` is already set on the order) is deterministic.

#### Ontology Mapping
`odoo:sale.order.line` — **UNRESOLVED** in ODOO_ALIGNMENTS.

**FLAG: Missing alignment row — proposal:**
- OWL pivot: `ubl:OrderLine` (UBL 2.1 OrderLine, line item within a commercial order)
- OGIT family: `SmbFoundryInvoice` (0x81) — line is part of the invoice/order document basin
- DOLCE: `Perdurant` (`.line` suffix → Perdurant per DOLCE suffix classifier rule in BRIEFING)
- Proposed row: `odoo:sale.order.line → ubl:OrderLine → OGIT SmbFoundryInvoice (0x81) → DOLCE Perdurant`

#### K-Step
Price/tax computation feeds K7 (USt: `tax_ids` on the line is what gets mapped through fiscal position and applied). Discount computation feeds K3 (affects posting amounts).

---

### Rule S-4: qty_to_invoice / qty_invoiced — Partial Invoicing Tracking

**Odoo source:** `sale_order_line.py:L238-251, L972-1065`

#### Axis-1 Rich-AST Spec

**`_compute_qty_invoiced` (L972-1006):**
`@api.depends('invoice_lines.move_id.state', 'invoice_lines.quantity')`

Delegates to `_prepare_qty_invoiced()`:
```python
for line in self:
    for invoice_line in line._get_invoice_lines():
        if invoice_line.move_id.state != 'cancel' or invoice_line.move_id.payment_state == 'invoicing_legacy':
            invoice_qty = invoice_line.product_uom_id._compute_quantity(invoice_line.quantity, line.product_uom_id, round=False)
            if invoice_line.move_id.move_type == 'out_invoice':
                invoiced_qties[line] += invoice_qty
            elif invoice_line.move_id.move_type == 'out_refund':
                invoiced_qties[line] -= invoice_qty
```

Key notes:
- Cancelled invoices are NOT counted (state == 'cancel'), UNLESS `payment_state == 'invoicing_legacy'` (backward compat).
- Refunds (`out_refund`) DECREASE the invoiced qty — so the SO line knows it can be re-invoiced.
- UoM conversion is done with `round=False` (raw qty before rounding, to avoid double-rounding).

**`_compute_qty_to_invoice` (L1036-1064):**
`@api.depends('qty_invoiced', 'qty_delivered', 'product_uom_qty', 'state')`

Full control flow:
```python
combo_lines = set()
for line in self:
    if line.state == 'sale' and not line.display_type:
        if line.product_id.type == 'combo':
            combo_lines.add(line)
        elif line.product_id.invoice_policy == 'order':
            line.qty_to_invoice = line.product_uom_qty - line.qty_invoiced
        else:  # 'delivery'
            line.qty_to_invoice = line.qty_delivered - line.qty_invoiced
        if line.combo_item_id and line.linked_line_id:
            combo_lines.add(line.linked_line_id)
    else:
        line.qty_to_invoice = 0
# Combo lines: only invoiceable if at least one combo item line has qty_to_invoice > 0
for combo_line in combo_lines:
    if any(line.combo_item_id and line.qty_to_invoice for line in combo_line.linked_line_ids):
        combo_line.qty_to_invoice = combo_line.product_uom_qty - combo_line.qty_invoiced
    else:
        combo_line.qty_to_invoice = 0
```

**Invoice policy `'order'`:** invoice based on ordered quantity (pre-delivery billing; typical for services).
**Invoice policy `'delivery'`:** invoice based on delivered quantity (post-delivery billing; typical for physical goods).

**`_force_lines_to_invoice_policy_order` (sale_order.py L1797-1807):** override that forces all lines to act as `invoice_policy='order'` — used for automatic invoice creation after payment (so full SO is invoiced regardless of delivery status).

#### Axis Classification
**DETERMINISTIC** — pure arithmetic on stored quantities. The invoice policy is a product configuration field, not a heuristic. The combo line logic is deterministic (any/all check on linked lines).

#### Ontology Mapping
`odoo:sale.order.line` — see Rule S-3 (UNRESOLVED, propose ubl:OrderLine → SmbFoundryInvoice 0x81 → Perdurant).

#### K-Step
This is the partial-invoicing tracking that enables K3 (only post the invoiced portion). Directly gates which lines appear in the created invoice.

#### woa-rs Target Module
woa-rs Vorgang has NO equivalent to `qty_to_invoice` / `qty_invoiced`. The Python WoA model is simpler: one Vorgang row = one invoice-level document. There is no line-level partial invoicing state. This is a significant ERP enrichment gap.

**GAP (Major):** woa-rs Vorgang does not track qty_invoiced/qty_to_invoice per line. If ERP functionality is needed, a new `vorgang_line` table and `qty_invoiced` column would be required, analogous to `sale.order.line`. Until then, this logic has no home in woa-rs.

---

### Rule S-5: invoice_status Derivation (Line + Order)

**Odoo source:** `sale_order_line.py:L1066-1095` (line); `sale_order.py:L618-664` (order)

#### Axis-1 Rich-AST Spec

**INVOICE_STATUS values (L19-24):**
```python
INVOICE_STATUS = [
    ('upselling', 'Upselling Opportunity'),
    ('invoiced', 'Fully Invoiced'),
    ('to invoice', 'To Invoice'),
    ('no', 'Nothing to Invoice')
]
```

**Line `_compute_invoice_status` (L1066-1095):**
`@api.depends('state', 'product_uom_qty', 'qty_delivered', 'qty_to_invoice', 'qty_invoiced')`

Ordered decision tree (first matching condition wins):
1. `state != 'sale'` → `'no'`
2. `is_downpayment AND untaxed_amount_to_invoice == 0` → `'invoiced'`
3. `not float_is_zero(qty_to_invoice, precision)` → `'to invoice'`
4. `state == 'sale' AND invoice_policy == 'order' AND qty >= 0.0 AND float_compare(qty_delivered, product_uom_qty) == 1` (delivered MORE than ordered) → `'upselling'`
5. `float_compare(qty_invoiced, product_uom_qty) >= 0` (invoiced >= ordered) → `'invoiced'`
6. else → `'no'`

**Precision:** uses `decimal.precision.precision_get('Product Unit')` — this is the DB-stored precision for product quantities, NOT a hardcoded decimal place.

**Order `_compute_invoice_status` (L618-664):**
`@api.depends('state', 'order_line.invoice_status')`

Full control flow:
1. `confirmed_orders = self.filtered(lambda so: so.state == 'sale')`
2. `(self - confirmed_orders).invoice_status = 'no'` — non-confirmed orders always 'no'.
3. Batch query: `_read_group` on `sale.order.line` where `is_downpayment=False AND display_type=False AND order_id in confirmed_orders.ids`, grouped by `['order_id', 'invoice_status']`.
4. For each confirmed order, from the `line_invoice_status` list:
   - If empty: `'no'`
   - If any `'to invoice'`:
     - If also any `'no'` present: check if the ONLY invoiceable lines are those that `_can_be_invoiced_alone()` returns False (i.e., discount/delivery/promo lines). If so → `'no'`. Otherwise → `'to invoice'`.
     - If no `'no'` in list: → `'to invoice'`
   - Elif ALL are `'invoiced'`: → `'invoiced'`
   - Elif ALL are `'invoiced'` or `'upselling'`: → `'upselling'`
   - else: → `'no'`

**Upselling side effect (L1964-1975):** When `invoice_status` transitions to `'upselling'`, the system automatically creates a TODO activity for the salesperson. This is done via `_compute_field_value` override that calls `_create_upsell_activity()`. The activity creation is skipped if `mail_activity_automation_skip` is in context.

**`_can_be_invoiced_alone` (L1097-1105):** Returns True unless the product is the company's `sale_discount_product_id` (global discount product). Discount-only invoiceable states don't count as "really invoiceable".

#### Axis Classification
**DETERMINISTIC** — pure aggregation/comparison logic on stored quantities. The batch `_read_group` query is a performance optimization, not a heuristic.

**Axis-2 note on upselling activity creation:** The creation of the TODO activity when `invoice_status → 'upselling'` is a **next-best-action trigger**. In woa-rs terms: `(ReasoningKind::NextBestAction, InferenceType::Induction, SemiringChoice::NarsTruth, ThinkingStyle cluster: Exploratory)`. Inherited from SmbFoundryInvoice family → a document-basin next-action recommendation has Exploratory character. But the detection logic itself (compare qty_delivered > qty_ordered) is deterministic; only the "what to do about it" (activity scheduling, sales suggestions) is Axis-2.

#### Ontology Mapping
Both `odoo:sale.order` and `odoo:sale.order.line` — UNRESOLVED. Proposals in S-1 and S-3.

#### K-Step
Determines when invoicing is triggered → feeds K3 (posting) and K8 (invoice reports).

---

### Rule S-6: _prepare_invoice — Order→Invoice Field Mapping

**Odoo source:** `sale_order.py:L1411-1450`

#### Axis-1 Rich-AST Spec

**`_prepare_invoice` (L1411-1450):**
Called per SO in `_create_invoices`. Returns a dict of `account.move` creation values.

Full field mapping:
```python
{
    'ref': self.client_order_ref or self.name,        # customer reference or SO name
    'move_type': 'out_invoice',                        # sales invoice (not bill)
    'narration': self.note,                            # T&C text → invoice narration
    'currency_id': self.currency_id.id,
    'campaign_id': self.campaign_id.id,                # UTM tracking
    'medium_id': self.medium_id.id,
    'source_id': self.source_id.id,
    'team_id': self.team_id.id,                        # sales team
    'partner_id': self.partner_invoice_id.id,          # INVOICE address (not shipping!)
    'partner_shipping_id': self.partner_shipping_id.id,
    'fiscal_position_id': (self.fiscal_position_id or
        self.fiscal_position_id._get_fiscal_position(self.partner_invoice_id)).id,
    'invoice_origin': self.name,                       # SO number in invoice origin field
    'invoice_payment_term_id': self.payment_term_id.id,
    'preferred_payment_method_line_id': self.preferred_payment_method_line_id.id,
    'invoice_user_id': self.user_id.id,               # salesperson → invoice user
    'payment_reference': self.reference,               # payment ref (Verwendungszweck)
    'transaction_ids': [Command.set(txs_to_be_linked.ids)],  # link payment transactions
    'company_id': self.company_id.id,
    'invoice_line_ids': [],                            # populated next by _create_invoices
    'user_id': self.user_id.id,
}
# Conditional: if self.journal_id: values['journal_id'] = self.journal_id.id
```

**Transaction linking (L1419-1424):** Filters payment transactions to those in `pending`/`authorized` state OR `done` + `not payment_id.is_reconciled`. These unreconciled transactions are linked to the new invoice for later reconciliation.

**Sudo context (L1547-1548):** Invoice creation happens via `self.env['account.move'].sudo()` with context `default_move_type='out_invoice'`. A salesperson can create an invoice from SO without billing access rights, but cannot create invoices from scratch.

**`_prepare_invoice` for purchase (L925-946):**
Purchase variant creates `in_invoice` (vendor bill), maps:
```python
{
    'move_type': move_type,          # context.get('default_move_type', 'in_invoice')
    'narration': self.note,
    'currency_id': self.currency_id.id,
    'partner_id': partner_invoice.id,  # vendor invoice address
    'fiscal_position_id': ...,
    'partner_bank_id': partner_bank_id.id,  # vendor bank (purchase only, not in sale)
    'invoice_origin': self.name,
    'invoice_payment_term_id': self.payment_term_id.id,
    'invoice_line_ids': [],
    'company_id': self.company_id.id,
}
```
Purchase does NOT copy `team_id`, `campaign_id`, `payment_reference`, `preferred_payment_method_line_id`, or transaction_ids to the bill.

#### Axis Classification
**DETERMINISTIC** — static field mapping. No heuristics.

#### Ontology Mapping
`odoo:sale.order` → see S-1; `odoo:account.move` is already resolved: `odoo:account.move → fibo:Transaction → SmbFoundryInvoice (0x81) → DOLCE Perdurant` (in ODOO_ALIGNMENTS L132-137).

#### K-Step
K3 — the created `account.move` is the posting document. When posted, double-entry bookkeeping entries are created. This method is the bridge between the order world and the accounting world.

#### woa-rs Target Module
In woa-rs, the Vorgang IS the invoice (`doc_type = 'invoice'`). The SO→invoice transition in woa-rs would be: update `doc_type` from `'order'` to `'invoice'`, set `rechnung_nr`, etc. There is no separate `account.move` model in woa-rs — the Vorgang row represents both the order and the invoice. This is a fundamental architectural difference.

**GAP:** odoo separates `sale.order` (commercial document) and `account.move` (accounting document). woa-rs collapses both into `workorders`. The porter needs to decide whether to keep the collapsed model (current woa-rs architecture, simpler) or introduce a separate posting document. Given the Rust port is behaviorally-preserving (Iron Rule 7), the collapsed model is correct for now.

---

### Rule S-7: _create_invoices — Invoice Creation Algorithm

**Odoo source:** `sale_order.py:L1499-1692`

#### Axis-1 Rich-AST Spec

**`_get_invoiceable_lines(final=False)` (L1499-1542):**

Iterates `self.order_line` in order. Tracks `section_line_ids` (current section), `subsection_line_ids` (current subsection).

Line classification:
- `display_type == 'line_section'`: reset `section_line_ids = [line.id]`, reset subsection, skip.
- `display_type == 'line_subsection'`: set `subsection_line_ids = [line.id]`, skip.
- `display_type != 'line_note' AND float_is_zero(qty_to_invoice, precision)`: skip (nothing to invoice).
- `qty_to_invoice > 0 OR (qty_to_invoice < 0 AND final) OR display_type == 'line_note'`:
  - If `is_downpayment`: add to `down_payment_line_ids` (put at end).
  - If under subsection: collect subsection_line_ids + section_line_ids first (lazy collection of headers).
  - If under section only: collect section_line_ids first.
  - Add line to `invoiceable_line_ids`.

Returns: `self.env['sale.order.line'].browse(invoiceable_line_ids + down_payment_line_ids)` — down payments always at the END.

**`_create_invoices(grouped=False, final=False, date=None)` (L1550-1692):**

Phase 1 — Build invoice_vals_list:
1. Access check: requires `account.move` create access OR write access on SO. If neither: return empty.
2. For each SO (with partner language + company context):
   a. `invoice_vals = order._prepare_invoice()`
   b. `invoiceable_lines = order._get_invoiceable_lines(final)`
   c. If ALL lines are display_type (no real lines): `continue` (skip this SO).
   d. Build `invoice_line_vals`:
      - Track `down_payment_section_added` (create a section header for down payments once).
      - For each line: if `is_downpayment` and section not yet added → insert down payment section via `_prepare_down_payment_section_line(sequence=...)`.
      - For down payment lines in final invoice: `optional_values['quantity'] = -1.0` AND `optional_values['extra_tax_data'] = reversed extra_tax_data`. This NEGATES the down payment on the final invoice.
      - `for vals in line._prepare_invoice_lines_vals_list(**optional_values): invoice_line_vals.append(Command.create(vals))`
   e. `invoice_vals['invoice_line_ids'] += invoice_line_vals`
3. If `invoice_vals_list` is empty and `raise_if_nothing_to_invoice` context: raise UserError.

Phase 2 — Grouping:
- If `grouped=True`: one invoice per SO (no grouping).
- If `grouped=False`: group by `_get_invoice_grouping_keys()` = `['company_id', 'partner_id', 'partner_shipping_id', 'currency_id', 'fiscal_position_id']`. Multiple SOs with same group keys → merged into one invoice. Merged refs = comma-separated (truncated to 2000 chars). payment_reference only kept if unique across the group.

Phase 3 — Resequencing:
- If `len(invoice_vals_list) < len(self)` (grouping happened): resequence lines via `_get_invoice_line_sequence(new=seq, old=old_seq)`. Prevents duplicate sequence numbers when combining from multiple SOs.

Phase 4 — Create invoices:
- `moves = self._create_account_invoices(invoice_vals_list, final)` (creates in sudo).

Phase 5 — Refund conversion:
- If `final=True`: find moves with `amount_total < 0` after tax calculation → call `action_switch_move_type()` to convert to `out_refund`. Sets reversed entry relationship.

Phase 6 — Message posting:
- Post origin link message on each invoice.

**Purchase `action_create_invoice` (purchase_order.py L760-833):**

Simpler than sale's `_create_invoices`:
1. No `final` parameter — purchase always invoices all quantities.
2. No `grouped` parameter by name, but still groups by `(company_id, partner_id, currency_id)`.
3. Section lines: only included if they precede a non-display line (`pending_section` pattern — section buffered, flushed when next real line appears).
4. Negative total → `action_switch_move_type()` (same as sale).
5. Optional: if `attachment_ids` provided, extend the created invoice with OCR attachment data and link message.

#### Axis Classification
**DETERMINISTIC** — the grouping, sequencing, line inclusion, down-payment negation are all deterministic algorithms.

#### Ontology Mapping
All UNRESOLVED classes (sale.order, sale.order.line) per proposals above. The created `account.move` is resolved.

#### K-Step
K3 (double-entry creation is next step after invoice is posted), K11 (Festschreibung happens when invoice is posted/locked).

---

### Rule S-8: Purchase Order State Machine (button_confirm + approval)

**Odoo source:** `purchase_order.py:L105-111, L625-668, L615-619, L1252-1261`

#### Axis-1 Rich-AST Spec

**State values (L105-111):**
```python
[
    ('draft', 'RFQ'),
    ('sent', 'RFQ Sent'),
    ('to approve', 'To Approve'),
    ('purchase', 'Purchase Order'),
    ('cancel', 'Cancelled')
]
```
Field: `state = fields.Selection(..., default='draft', readonly=True, index=True, copy=False, tracking=True)`

**`button_confirm` (L625-639):**
For each order:
1. Skip if state not in `['draft', 'sent']`.
2. `_confirmation_error_message()` (L657-668): guard — any non-display, non-downpayment line missing product → error.
3. `order.order_line._validate_analytic_distribution()`.
4. `order._add_supplier_to_product()` (L682-708): adds vendor to product supplierinfo if not already there, max 10 suppliers per product. This is a SIDE EFFECT of confirming.
5. If `_approval_allowed()`:
   - Call `button_approve()` → write `{'state': 'purchase', 'date_approve': now()}`. If `lock_confirmed_po == 'lock'`: also write `{'locked': True}`.
6. Else: write `{'state': 'to approve'}` (needs a second approval).

**`_approval_allowed` (L1252-1261):**
Returns True if:
- `company.po_double_validation == 'one_step'` (no second approval needed), OR
- `company.po_double_validation == 'two_step' AND amount_total < po_double_validation_amount` (below threshold), OR
- User has `purchase.group_purchase_manager` role.

**`button_cancel` (L641-649):** Raises UserError if:
- Any order is locked.
- Any order has non-cancelled/non-draft invoices (must cancel bills first).
Then writes `{'state': 'cancel'}`.

**Deletion guard (L408-412):** `_unlink_if_cancelled` — can only delete cancelled POs.

#### Axis Classification
**DETERMINISTIC** for the state machine itself. **HEURISTIC** for the double-validation amount threshold: the `po_double_validation_amount` is a configured threshold, not computed. The `_approval_allowed` check is deterministic given the company config.

**Note on `_add_supplier_to_product`:** This is an interesting side-effect. Odoo automatically enriches product supplier info on PO confirmation. This is DETERMINISTIC business logic (not heuristic) but represents an ERP enrichment entirely absent from woa-rs.

#### Ontology Mapping
`odoo:purchase.order` — **UNRESOLVED** in ODOO_ALIGNMENTS.

**FLAG: Missing alignment row — proposal:**
- OWL pivot: `ubl:Order` (same UBL Order standard, just on the buy side — UBL uses same Order for both purchase and sales orders from the buyer's perspective; the type is contextual)
- OGIT family: `SmbFoundryInvoice` (0x81) — purchase order is a commercial document in the same document basin
- DOLCE: `Perdurant`
- Proposed row: `odoo:purchase.order → ubl:Order → OGIT SmbFoundryInvoice (0x81) → DOLCE Perdurant`

Note: distinguishing sale vs purchase UBL: UBL 2.1 has `ubl:Order` (purchase order from buyer) and there is no separate "sale order" in UBL — the seller's perspective is the `ubl:OrderResponse`. If we need distinct OWL pivots: `ubl:Order` for purchase.order and `ubl:OrderResponse` for sale.order. Either way, same OGIT family.

#### K-Step
Same as sale: core ERP Vorgang lifecycle, prerequisite to K3 (bill posting).

#### woa-rs Target Module
woa-rs has no purchase order model. Purchase functionality is entirely absent from WoA Python source as well. Not a current target.

---

### Rule S-9: invoice_status on Purchase Order

**Odoo source:** `purchase_order.py:L46-68`

#### Axis-1 Rich-AST Spec

**`_get_invoiced` (L46-68):**
`@api.depends('state', 'order_line.qty_to_invoice')`

```python
for order in self:
    if order.state != 'purchase':
        order.invoice_status = 'no'
        continue
    precision = decimal.precision.precision_get('Product Unit')
    any_to_invoice = any(
        not float_is_zero(line.qty_to_invoice, precision_digits=precision)
        for line in order.order_line.filtered(lambda l: not l.display_type)
    )
    all_invoiced_with_invoices = (
        all(float_is_zero(line.qty_to_invoice, ...) for non-display lines)
        and order.invoice_ids  # must have at least one invoice!
    )
    if any_to_invoice: order.invoice_status = 'to invoice'
    elif all_invoiced_with_invoices: order.invoice_status = 'invoiced'
    else: order.invoice_status = 'no'
```

**Key difference from sale:** Purchase only has 3 statuses (`'no'`, `'to invoice'`, `'invoiced'`) — NO `'upselling'` status. The `all_invoiced_with_invoices` check requires `invoice_ids` to exist (can't be invoiced with zero invoices even if qty_to_invoice is 0 — prevents marking new POs as 'invoiced').

#### Axis Classification
**DETERMINISTIC** — same as sale.

---

### Rule S-10: _prepare_invoice_line — Line→InvoiceLine Mapping

**Odoo source:** `sale_order_line.py:L1488-1536`

#### Axis-1 Rich-AST Spec

**`_prepare_invoice_line` (L1491-1536):**
Standard line (not combo):
```python
{
    'display_type': self.display_type or 'product',
    'sequence': self.sequence,
    'name': account.move.line._get_journal_items_full_name(self.name, self.product_id.display_name),
    'product_id': self.product_id.id,
    'product_uom_id': self.product_uom_id.id,
    'quantity': self.qty_to_invoice,          # KEY: invoices ONLY qty_to_invoice, not full qty
    'discount': self.discount,
    'price_unit': self.price_unit,
    'tax_ids': [Command.set(self.tax_ids.ids)],
    'sale_line_ids': [Command.link(self.id)],  # back-link to SO line (many2many)
    'is_downpayment': self.is_downpayment,
    'extra_tax_data': self.extra_tax_data,
    'collapse_prices': self.collapse_prices,
    'collapse_composition': self.collapse_composition,
}
```

Down payment lines: if `is_downpayment` and existing `downpayment_lines` exist → carry over `account_id` from previous invoice line.

Display type lines: `account_id = False`.

Combo product lines (L1499-1512): create a `line_section` display line instead of a product line, with `name = f'{product.name} x {qty_to_invoice}'`.

Optional values override: any `**optional_values` passed in `_create_invoices` (e.g., `quantity=-1.0` for down payment negation in final invoice) are applied last, overriding defaults.

**`sale_line_ids` many2many link:** This is the critical back-link. The `sale_order_line_invoice_rel` junction table links SO lines to invoice lines. This is how `qty_invoiced` is computed (via `invoice_lines.move_id.state` / `.quantity`).

#### Axis Classification
**DETERMINISTIC** — static field mapping with one conditional (down payment account).

#### K-Step
K3 — the invoice line `price_unit × quantity × (1-discount/100) × tax` becomes the posting amount.

---

### Rule S-11: untaxed_amount_to_invoice / amount_to_invoice

**Odoo source:** `sale_order_line.py:L1143-1200`

#### Axis-1 Rich-AST Spec

**`_compute_untaxed_amount_to_invoice` (L1143-1189):**
`@api.depends('state', 'product_id', 'untaxed_amount_invoiced', 'qty_delivered', 'product_uom_qty', 'price_unit')`

Only computes for `state == 'sale'`. Otherwise `amount_to_invoice = 0.0`.

```python
uom_qty_to_consider = (qty_delivered if invoice_policy == 'delivery' else product_uom_qty)
price_reduce = price_unit * (1 - (discount or 0.0) / 100.0)
price_subtotal = price_reduce * uom_qty_to_consider

# If tax is price-inclusive: strip included tax from price_subtotal
if any(tax.price_include for tax in self.tax_ids):
    price_subtotal = self.tax_ids.compute_all(
        price_reduce, currency=currency, quantity=uom_qty_to_consider,
        product=product, partner=partner_shipping_id
    )['total_excluded']

# Check for invoice lines with different discount (re-invoicing case)
inv_lines = line._get_invoice_lines()
if any(l.discount != line.discount for l in inv_lines):
    # Manual re-calculation to handle discount drift
    amount = sum(
        tax_ids.compute_all(converted_price * qty)['total_excluded']  # if price_include
        OR converted_price * qty  # otherwise
        for l in inv_lines
    )
    amount_to_invoice = max(price_subtotal - amount, 0)
else:
    amount_to_invoice = price_subtotal - line.untaxed_amount_invoiced
```

**Key: `max(…, 0)`** — the remaining amount to invoice is FLOORED at 0. Can't have a negative amount to invoice (that would be a credit situation handled differently).

**`_compute_amount_to_invoice` (L1191-1200):** Tax-inclusive version (uses `price_total`):
```python
if product_uom_qty:
    uom_qty = qty_delivered if delivery_policy else product_uom_qty
    qty_to_invoice = uom_qty - qty_invoiced_posted
    unit_price_total = price_total / product_uom_qty  # tax-included unit price
    amount_to_invoice = unit_price_total * qty_to_invoice
else: 0.0
```

Uses `qty_invoiced_posted` (only posted invoices, not draft) to compute the remaining tax-included amount.

#### Axis Classification
**DETERMINISTIC** — arithmetic with max(0) floor.

---

## 3. Enterprise / Unresolved Flags

### Enterprise Boundary Flags

1. **`account_asset` (K12 Anlagen):** Not in community clone. No odoo source for asset depreciation computation. The sale of assets via `sale.order` is visible in community (asset products can be ordered), but the depreciation schedule is Enterprise-only. This lane does not touch K12.

2. **`sale.order` → `project.project` link (sale_project):** The `_action_confirm()` hook in base sale is empty. `sale_project` extension fills it to create project/tasks on confirmation. This is part of community odoo (not Enterprise) but NOT in the scoped files. The Vorgang lifecycle in woa-rs does have `project_id` FK — so this hook is relevant but not in scope here.

3. **`account_analytic` distribution validation:** `order_line._validate_analytic_distribution()` is called on confirmation. The analytic module is community but not scoped in this lane. The validation may raise for missing analytic accounts. Porter should note this as a guard step.

### ODOO_ALIGNMENTS Unresolved Classes (from this lane)

All four classes touched in this lane are UNRESOLVED in `ODOO_ALIGNMENTS`:

| odoo class | Current status | Proposed OWL pivot | Proposed OGIT family | Proposed DOLCE |
|---|---|---|---|---|
| `odoo:sale.order` | UNRESOLVED | `ubl:Order` | `SmbFoundryInvoice` (0x81) | Perdurant |
| `odoo:sale.order.line` | UNRESOLVED | `ubl:OrderLine` | `SmbFoundryInvoice` (0x81) | Perdurant |
| `odoo:purchase.order` | UNRESOLVED | `ubl:Order` | `SmbFoundryInvoice` (0x81) | Perdurant |
| `odoo:purchase.order.line` | UNRESOLVED | `ubl:OrderLine` | `SmbFoundryInvoice` (0x81) | Perdurant |

**Note on UBL vs other pivot candidates:** UBL (Universal Business Language) is the canonical B2B XML standard for orders and invoices, widely used in e-invoicing. It is the natural OWL pivot for commercial order documents. Alternatives considered:
- `schema:Order` (schema.org): less formal, more e-commerce oriented
- `fibo:CommercialAgreement`: too abstract
- `ccts:OrderDocument` (UN/CEFACT): technically more precise but less commonly linked in OWL ontologies

Recommendation: use `ubl:Order` / `ubl:OrderLine` as they are the closest standard analogs and align with the X-Rechnung/ZUGFeRD work in woa-rs (those use UBL structure).

---

## 4. woa-rs Calibration Summary

### What woa-rs Vorgang currently has (from `src/models/work_order.rs`)

- `doc_type`: `'workorder'|'offer'|'order'|'invoice'|'craft_invoice'|'credit'|'collective_invoice'`
- `status`: `'draft'|'open'|'in_progress'|'completed'|'invoiced'|'paid'|'cancelled'`
- `angebot_nr`, `auftrags_nr`, `workorder_nr`, `rechnung_nr`, `gutschrift_nr` — separate number fields for each lifecycle stage
- `sammelrechnung_id` — self-referential FK for collective invoicing
- `zahlungsart`: `'rechnung'|'vorkasse'|'anzahlung'`
- `anzahlung_prozent`, `anzahlung_betrag` — down payment fields (partial analog to odoo's down payment lines)
- `bezahlt`, `bezahlt_am` — payment tracking
- `mahnstufe`, `letzte_mahnung` — dunning (no odoo analog in sale module)
- `kleinunternehmer_snapshot`, `zahlungsziel_tage_snapshot` — GoBD freeze fields

### Richness Gaps (odoo richer than woa-rs)

| Odoo feature | woa-rs status | Gap severity |
|---|---|---|
| `qty_to_invoice` / `qty_invoiced` per line | Absent (no line model at all) | MAJOR |
| Partial invoicing state machine | Absent | MAJOR |
| `invoice_status` enum (no/to invoice/invoiced/upselling) | Absent (status covers different states) | MAJOR |
| Multi-line SO with sections/subsections | Absent | MODERATE |
| Invoice grouping by partner/currency | Absent | MODERATE |
| Fiscal position → tax mapping | Absent | MODERATE |
| EPD (Early Payment Discount) mixed mode | Absent | LOW (unusual edge case) |
| Upselling activity automation | Absent | LOW (UI feature) |
| Down payment line negation on final invoice | Absent | LOW (no line model) |

### What woa-rs can absorb without a line model

The following odoo logic CAN be ported to woa-rs even without a `vorgang_line` table:

1. **State machine transitions** (S-1): map to `status` field transitions with guard logic in `src/erp/sale_order_fsm.rs`.
2. **Amount computation** (S-2, simplified): `brutto_summe = netto_summe * (1 + mwst_satz)` — woa-rs already tracks these. The odoo tax engine pipeline (prepare_base_line → add_tax_details → round) is the richer version; for woa-rs the `tax_rate::Model` provides the rate.
3. **Invoice creation field mapping** (S-6): the transition from Vorgang `doc_type='order'` to `doc_type='invoice'` with `rechnung_nr` set is the woa-rs equivalent of `_prepare_invoice`.
4. **GoBD lock** (S-1 cancel guard): already implemented in `src/gobd.rs`.

---

## 5. Porter's Checklist — Non-Obvious Gotchas

1. **No `'done'` state.** Community odoo 17 `sale.order` has only `draft/sent/sale/cancel`. The `locked` boolean is separate from state. Do not implement a `done` state.

2. **`date_order` is OVERWRITTEN on confirm.** `_prepare_confirmation_values()` sets `date_order = now()`. The original creation date is in `create_date` (read-only). In woa-rs, `datum` is the user-set date; a separate `confirmed_at` timestamp may be needed if this distinction matters.

3. **Price is frozen once invoiced.** `_compute_price_unit` skips recompute if `qty_invoiced > 0`. In woa-rs, once a Vorgang is in `invoiced` or `paid` status, price edits should be blocked (GoBD lock already enforces this via `src/gobd.rs`).

4. **Fiscal position maps taxes.** The `_compute_tax_ids` pipeline calls `fiscal_position.map_tax(taxes)`. A customer in a different country/VAT regime gets different taxes applied. woa-rs currently has no fiscal position concept — all tax is flat-rate via `tax_rate::Model`. This is a simplification that is acceptable for Stefan's domestic use case.

5. **`qty_invoiced` counts refunds negatively.** An `out_refund` against a line DECREASES `qty_invoiced`, making the line invoiceable again. In woa-rs, a credit note (`doc_type='credit'`) logically reverses an invoice but there is no qty tracking to unlock re-invoicing.

6. **Down payment negation on final invoice.** When `final=True`, down payment lines get `quantity=-1.0` and reversed `extra_tax_data`. This avoids double-counting: the customer already paid the down payment, so the final invoice shows it as a deduction. woa-rs has `anzahlung_betrag` but no line-level negation mechanism.

7. **Invoice grouping by shipping address.** `_get_invoice_grouping_keys()` includes `partner_shipping_id`. Multiple SOs with different shipping addresses → separate invoices even if same billing partner. woa-rs has no shipping address concept.

8. **Sudo for invoice creation.** `account.move` is created in sudo so salespersons can create invoices without full accounting access. In woa-rs, permission checks are handled at route level; this is not a direct concern but the pattern (salesperson can create invoice from SO) should be preserved.

9. **Combo products.** When `product.type == 'combo'`, `tax_ids = False` and `qty_to_invoice` is computed differently (via linked item lines). woa-rs has no combo product concept; this is safe to ignore.

10. **Purchase `_approval_allowed`** double-validation threshold: `po_double_validation_amount` is compared using `company_currency._convert` to the order's currency. Multi-currency comparison is deterministic but requires currency tables. woa-rs is single-currency (EUR).

11. **`technical_price_unit` shadow field.** This is not a displayed field but is essential for detecting manual price edits. When porting, any price-edit lock mechanism needs to compare "pricelist-computed" vs "actual" price, requiring this dual-field pattern or equivalent.

12. **Section/subsection carry-forward.** `_get_invoiceable_lines()` includes section header lines only when there is at least one invoiceable line under them. Empty sections are dropped. This prevents orphan section headers on partial invoices.

13. **Precision from `decimal.precision`** not hardcoded. `precision_get('Product Unit')` returns a DB-stored decimal precision setting. Porter should use a configurable precision, not a compile-time constant.

14. **`_can_be_invoiced_alone()` discount product check.** A discount-only SO (all lines are global discount lines) gets `invoice_status = 'no'` even if technically there are `to invoice` lines. This prevents creating invoices that contain nothing but discounts.

---

## 6. Depth Proof

Read: `/home/user/odoo/addons/sale/models/sale_order.py` lines=2301 depth=full
Read: `/home/user/odoo/addons/sale/models/sale_order_line.py` lines=1819 depth=full
Read: `/home/user/odoo/addons/purchase/models/purchase_order.py` lines=1418 depth=full
Read: `/home/user/woa-rs/src/models/work_order.rs` lines=~160 depth=full
Read: `/home/user/lance-graph/crates/lance-graph-callcenter/src/odoo_alignment.rs` lines=~260 depth=full
Read: `/home/user/woa-rs/.claude/board/odoo-richness/BRIEFING.md` lines=124 depth=full
