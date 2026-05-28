//! Lane L6 (SALE-PURCHASE) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L6-SALE-PURCHASE.md`.
//!
//! Entities covered (4):
//!   - `sale.order`          (Rules S-1 … S-7: state machine, amounts, invoice creation)
//!   - `sale.order.line`     (Rules S-3 … S-5, S-10, S-11: price/discount/qty tracking)
//!   - `purchase.order`      (Rules S-8, S-9: state machine, invoice_status)
//!   - `purchase.order.line` (Rule S-9 sub-model: qty_to_invoice / invoice_status)
//!
//! **L8 overlap note:** `product.pricelist` and `product.pricelist.item` are
//! referenced in Rules S-3/_compute_discount (L6) and are primary entities in
//! L8-PRODUCT-UOM-PRICELIST.  L6 only touches pricelist fields indirectly via
//! `pricelist_id` on `sale.order` and computed helpers on the line; the
//! canonical projection of those two models belongs in L8.  They are NOT
//! included here to avoid double-coverage.
//!
//! **Odoo source files read by the L6 author:**
//!   `addons/sale/models/sale_order.py`        (2301 lines, full)
//!   `addons/sale/models/sale_order_line.py`   (1819 lines, full)
//!   `addons/purchase/models/purchase_order.py` (1418 lines, full)

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooState, OdooStateMachine, OdooStateSemantic,
    OdooTransition,
};

// ─── sale.order ──────────────────────────────────────────────────────────────

/// State machine for `sale.order`.
///
/// Community Odoo 17: states are `draft / sent / sale / cancel`.
/// There is NO `done` state (that existed in older versions).
/// The `locked` boolean (separate field) replaces the old `done` state.
/// Source: `sale_order.py:L26-31`.
const SALE_ORDER_STATE_MACHINE: OdooStateMachine = OdooStateMachine {
    state_field: "state",
    states: &[
        OdooState { name: "draft", semantic: OdooStateSemantic::Draft },
        OdooState { name: "sent", semantic: OdooStateSemantic::Active },
        OdooState { name: "sale", semantic: OdooStateSemantic::InProgress },
        OdooState { name: "cancel", semantic: OdooStateSemantic::Cancelled },
    ],
    transitions: &[
        OdooTransition {
            from: "draft",
            to: "sent",
            trigger: "action_quotation_sent",
            guards: &[],
            // Guard: state must be draft; raises UserError otherwise (L1156-1164).
        },
        OdooTransition {
            from: "draft",
            to: "sale",
            trigger: "action_confirm",
            guards: &["_confirmation_error_message"],
            // Full flow: validate analytics → write state+date_order → _action_confirm
            // → optional auto-lock → optional confirmation email (L1166-1196).
        },
        OdooTransition {
            from: "sent",
            to: "sale",
            trigger: "action_confirm",
            guards: &["_confirmation_error_message"],
        },
        OdooTransition {
            from: "cancel",
            to: "draft",
            trigger: "action_draft",
            guards: &[],
            // Also clears signature/signed_by/signed_on (L1058-1065).
        },
        OdooTransition {
            from: "sent",
            to: "draft",
            trigger: "action_draft",
            guards: &[],
        },
        OdooTransition {
            from: "sale",
            to: "cancel",
            trigger: "action_cancel",
            guards: &[],
            // Guard: raises UserError if locked; delegates to _action_cancel which
            // cancels all draft invoices first (L1324-1333).
        },
        OdooTransition {
            from: "draft",
            to: "cancel",
            trigger: "action_cancel",
            guards: &[],
        },
        OdooTransition {
            from: "sent",
            to: "cancel",
            trigger: "action_cancel",
            guards: &[],
        },
    ],
};

pub const SALE_ORDER: OdooEntity = OdooEntity {
    model_name: "sale.order",
    description: "Commercial sale quotation / order; Vorgang lifecycle (draft → sent → \
                  sale → cancel); generates `account.move` via `_create_invoices`. \
                  OWL pivot proposed: ubl:Order → SmbFoundryInvoice (0x81) → DOLCE Perdurant.",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: Some("_compute_name"),
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "state",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // SALE_ORDER_STATE: draft/sent/sale/cancel. No 'done' in Odoo 17 community.
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "locked",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Replaces the old 'done' state; set via action_lock (L1318-1319).
            // GoBD-equivalent: prevents further edits after confirmation.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "date_order",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // OVERWRITTEN to now() on action_confirm. Original creation date in create_date.
            // DB constraint: must be non-null when state='sale' (L41-44).
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "partner_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "partner_invoice_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: true,
            computed: Some("_compute_partner_invoice_id"),
            depends: &["partner_id"],
            // Invoice address (not shipping address) used in _prepare_invoice.
            semantic_role: OdooSemanticRole::Address,
        },
        OdooField {
            name: "partner_shipping_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: true,
            computed: Some("_compute_partner_shipping_id"),
            depends: &["partner_id"],
            // Shipping address included in invoice grouping key.
            semantic_role: OdooSemanticRole::Address,
        },
        OdooField {
            name: "pricelist_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.pricelist"),
            required: false,
            computed: None,
            depends: &[],
            // Write guard: cannot change on confirmed (state='sale') order (L1040-1043).
            // Primary model for product.pricelist is L8 (PRODUCT-UOM-PRICELIST).
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "currency_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.currency"),
            required: false,
            computed: Some("_compute_currency_id"),
            depends: &["pricelist_id.currency_id", "company_id.currency_id"],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "order_line",
            kind: OdooFieldKind::One2many,
            target: Some("sale.order.line"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "invoice_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("account.move"),
            required: false,
            computed: Some("_compute_invoice_ids"),
            depends: &["order_line.invoice_lines.move_id"],
            semantic_role: OdooSemanticRole::Document,
        },
        OdooField {
            name: "invoice_status",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: Some("_compute_invoice_status"),
            depends: &["state", "order_line.invoice_status"],
            // Values: 'upselling' | 'invoiced' | 'to invoice' | 'no'.
            // Upselling triggers a TODO activity for the salesperson (L1964-1975).
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "amount_untaxed",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_compute_amounts"),
            depends: &[
                "order_line.price_subtotal",
                "currency_id",
                "company_id",
                "payment_term_id",
            ],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "amount_tax",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_compute_amounts"),
            depends: &[
                "order_line.price_subtotal",
                "currency_id",
                "company_id",
                "payment_term_id",
            ],
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "amount_total",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_compute_amounts"),
            depends: &[
                "order_line.price_subtotal",
                "currency_id",
                "company_id",
                "payment_term_id",
            ],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "payment_term_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.payment.term"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "fiscal_position_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.fiscal.position"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "user_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.users"),
            required: false,
            computed: None,
            depends: &[],
            // Salesperson — copied as invoice_user_id in _prepare_invoice.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "team_id",
            kind: OdooFieldKind::Many2one,
            target: Some("crm.team"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "client_order_ref",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Customer purchase order reference. Copied as 'ref' in _prepare_invoice.
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "note",
            kind: OdooFieldKind::Html,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Terms & Conditions text — copied as 'narration' in _prepare_invoice.
            semantic_role: OdooSemanticRole::Document,
        },
    ],
    methods: &[
        OdooMethod {
            name: "action_quotation_sent",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["sent"],
        },
        OdooMethod {
            name: "action_confirm",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Boolean,
            triggers: &["sale"],
            // Full flow: guard check → validate analytics → write state/date_order →
            // _action_confirm hook → optional auto-lock → optional confirmation email.
        },
        OdooMethod {
            name: "action_draft",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["draft"],
        },
        OdooMethod {
            name: "action_cancel",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["cancel"],
        },
        OdooMethod {
            name: "action_lock",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Sets locked=True. Not a state transition; boolean flag only.
        },
        OdooMethod {
            name: "_action_cancel",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Cancels draft invoices first, then writes state='cancel'.
        },
        OdooMethod {
            name: "_compute_amounts",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Pipeline: _get_priced_lines → _prepare_base_line_for_taxes_computation →
            // EPD lines → AccountTax._add_tax_details → _round → _get_tax_totals_summary.
        },
        OdooMethod {
            name: "_compute_invoice_status",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Batch _read_group on order_line, aggregates per-line invoice_status.
        },
        OdooMethod {
            name: "_prepare_invoice",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
            // Field mapping SO → account.move (move_type='out_invoice').
        },
        OdooMethod {
            name: "_get_invoiceable_lines",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
            // Collects lines with qty_to_invoice != 0; carries section headers lazily.
        },
        OdooMethod {
            name: "_create_invoices",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
            // 6-phase algorithm: build vals → group by partner/currency → resequence →
            // create in sudo → convert negative totals to out_refund → post origin msg.
        },
        OdooMethod {
            name: "_confirmation_error_message",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        OdooMethod {
            name: "_unlink_except_draft_or_cancel",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Deletion guard: only draft/cancel orders can be deleted (L1032-1038).
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &[
                "order_line.price_subtotal",
                "currency_id",
                "company_id",
                "payment_term_id",
            ],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["state", "order_line.invoice_status"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["state"],
        },
    ],
    state_machine: Some(&SALE_ORDER_STATE_MACHINE),
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "(state = 'sale' AND date_order IS NOT NULL) OR state != 'sale' — \
                        confirmed orders must have date_order (L41-44)",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "pricelist_id cannot be changed on a confirmed (state='sale') order",
            source_method: Some("write"),
            // Enforced in write() override, not a dedicated @api.constrains method.
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "cannot delete order unless state in (draft, cancel)",
            source_method: Some("_unlink_except_draft_or_cancel"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L6-SALE-PURCHASE.md",
        l_doc_lines: (31, 543),
        odoo_source: &[],
        confidence: OdooConfidence::Curated,
    },
};

// ─── sale.order.line ─────────────────────────────────────────────────────────

pub const SALE_ORDER_LINE: OdooEntity = OdooEntity {
    model_name: "sale.order.line",
    description: "One line on a sale order; tracks price_unit/discount/tax_ids, \
                  partial invoicing state (qty_invoiced / qty_to_invoice), and \
                  invoice_status per line.  OWL pivot proposed: ubl:OrderLine → \
                  SmbFoundryInvoice (0x81) → DOLCE Perdurant.",
    fields: &[
        OdooField {
            name: "order_id",
            kind: OdooFieldKind::Many2one,
            target: Some("sale.order"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "product_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.product"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "product_uom_qty",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Ordered quantity (before delivery); basis for invoice_policy='order'.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "product_uom_id",
            kind: OdooFieldKind::Many2one,
            target: Some("uom.uom"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "price_unit",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: Some("_compute_price_unit"),
            depends: &["product_id", "product_uom_id", "product_uom_qty"],
            // Frozen once qty_invoiced > 0. Pricelist-computed unless manually edited.
            // Manual edit detected via technical_price_unit shadow field.
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "technical_price_unit",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Shadow field: tracks system-computed price; diff from price_unit → manual edit.
            // Not displayed to users.
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "discount",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_discount"),
            depends: &["product_id", "product_uom_id", "product_uom_qty"],
            // Percentage discount (0–100). Formula: (base_price - pricelist_price) / base_price * 100.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "tax_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("account.tax"),
            required: false,
            computed: Some("_compute_tax_ids"),
            depends: &["product_id", "company_id"],
            // Mapped through fiscal_position; combo products → tax_ids=False.
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "price_subtotal",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_compute_amount"),
            depends: &["product_uom_qty", "discount", "price_unit", "tax_ids"],
            // tax-excluded subtotal; via AccountTax pipeline.
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "price_total",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_compute_amount"),
            depends: &["product_uom_qty", "discount", "price_unit", "tax_ids"],
            // tax-included total.
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "price_tax",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_amount"),
            depends: &["product_uom_qty", "discount", "price_unit", "tax_ids"],
            // price_total - price_subtotal.
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "qty_delivered",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_qty_delivered"),
            depends: &[],
            // Basis for invoice_policy='delivery' invoicing.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "qty_invoiced",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_qty_invoiced"),
            depends: &["invoice_lines.move_id.state", "invoice_lines.quantity"],
            // Refunds (out_refund) DECREASE this; cancelled invoices excluded.
            // UoM-converted with round=False to avoid double-rounding.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "qty_to_invoice",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_qty_to_invoice"),
            depends: &[
                "qty_invoiced",
                "qty_delivered",
                "product_uom_qty",
                "state",
            ],
            // invoice_policy='order': product_uom_qty - qty_invoiced.
            // invoice_policy='delivery': qty_delivered - qty_invoiced.
            // 0 when state != 'sale'.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "invoice_status",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: Some("_compute_invoice_status"),
            depends: &[
                "state",
                "product_uom_qty",
                "qty_delivered",
                "qty_to_invoice",
                "qty_invoiced",
            ],
            // Decision tree: no → invoiced → to invoice → upselling → invoiced → no.
            // 'upselling': qty_delivered > product_uom_qty when invoice_policy='order'.
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "invoice_lines",
            kind: OdooFieldKind::Many2many,
            target: Some("account.move.line"),
            required: false,
            computed: None,
            depends: &[],
            // Back-link via sale_order_line_invoice_rel junction; used by qty_invoiced.
            semantic_role: OdooSemanticRole::Document,
        },
        OdooField {
            name: "is_downpayment",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Down payment lines sorted to end in _get_invoiceable_lines;
            // negated (quantity=-1) on final invoice.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "display_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // 'line_section' | 'line_subsection' | 'line_note' | False.
            // Non-False lines excluded from price/qty computation.
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "sequence",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "untaxed_amount_to_invoice",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_compute_untaxed_amount_to_invoice"),
            depends: &[
                "state",
                "product_id",
                "untaxed_amount_invoiced",
                "qty_delivered",
                "product_uom_qty",
                "price_unit",
            ],
            // Floored at 0 (max(..., 0)). Handles discount-drift in reinvoicing.
            semantic_role: OdooSemanticRole::Money,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_tax_ids",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Groups by company; maps taxes through fiscal_position; combo → False.
        },
        OdooMethod {
            name: "_compute_price_unit",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Skip conditions: orphan line, downpayment, global discount, manual edit,
            // qty_invoiced>0, expense cost line.
        },
        OdooMethod {
            name: "_reset_price_unit",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Gets pricelist price → strips tax-include → updates price_unit + technical_price_unit.
        },
        OdooMethod {
            name: "_compute_discount",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // (base_price - pricelist_price) / base_price * 100. Shows only positive discounts.
        },
        OdooMethod {
            name: "_compute_amount",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Per-line: _prepare_base_line → _add_tax_details → _round → set subtotal/total.
        },
        OdooMethod {
            name: "_compute_qty_invoiced",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_qty_to_invoice",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_invoice_status",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_untaxed_amount_to_invoice",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_prepare_invoice_line",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
            // Maps SO line → account.move.line vals; quantity = qty_to_invoice (not full qty).
        },
        OdooMethod {
            name: "_prepare_base_line_for_taxes_computation",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_validate_analytic_distribution",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Called in action_confirm before state write.
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_id", "company_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_id", "product_uom_id", "product_uom_qty"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_uom_qty", "discount", "price_unit", "tax_ids"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["invoice_lines.move_id.state", "invoice_lines.quantity"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &[
                "qty_invoiced",
                "qty_delivered",
                "product_uom_qty",
                "state",
            ],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &[
                "state",
                "product_uom_qty",
                "qty_delivered",
                "qty_to_invoice",
                "qty_invoiced",
            ],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "analytic distribution must be valid before order confirmation",
        source_method: Some("_validate_analytic_distribution"),
    }],
    provenance: OdooProvenance {
        l_doc: "L6-SALE-PURCHASE.md",
        l_doc_lines: (186, 743),
        odoo_source: &[],
        confidence: OdooConfidence::Curated,
    },
};

// ─── purchase.order ───────────────────────────────────────────────────────────

/// State machine for `purchase.order`.
///
/// States: `draft / sent / to approve / purchase / cancel`.
/// The `to approve` state is intermediate (two-step PO validation).
/// Source: `purchase_order.py:L105-111`.
const PURCHASE_ORDER_STATE_MACHINE: OdooStateMachine = OdooStateMachine {
    state_field: "state",
    states: &[
        OdooState { name: "draft", semantic: OdooStateSemantic::Draft },
        OdooState { name: "sent", semantic: OdooStateSemantic::Active },
        OdooState { name: "to approve", semantic: OdooStateSemantic::Active },
        OdooState { name: "purchase", semantic: OdooStateSemantic::InProgress },
        OdooState { name: "cancel", semantic: OdooStateSemantic::Cancelled },
    ],
    transitions: &[
        OdooTransition {
            from: "draft",
            to: "to approve",
            trigger: "button_confirm",
            guards: &["_confirmation_error_message", "_approval_allowed"],
            // Two-step path: company.po_double_validation='two_step' AND
            // amount >= po_double_validation_amount AND not purchase.group_purchase_manager.
        },
        OdooTransition {
            from: "sent",
            to: "to approve",
            trigger: "button_confirm",
            guards: &["_confirmation_error_message", "_approval_allowed"],
        },
        OdooTransition {
            from: "draft",
            to: "purchase",
            trigger: "button_confirm",
            guards: &["_confirmation_error_message", "_approval_allowed"],
            // One-step path: _approval_allowed() returns True.
            // Side effect: _add_supplier_to_product() (max 10 suppliers per product).
        },
        OdooTransition {
            from: "sent",
            to: "purchase",
            trigger: "button_confirm",
            guards: &["_confirmation_error_message", "_approval_allowed"],
        },
        OdooTransition {
            from: "to approve",
            to: "purchase",
            trigger: "button_approve",
            guards: &[],
            // Writes state='purchase' + date_approve=now(). Optional lock.
        },
        OdooTransition {
            from: "draft",
            to: "cancel",
            trigger: "button_cancel",
            guards: &[],
            // Guard: raises UserError if locked or has non-cancelled/non-draft invoices.
        },
        OdooTransition {
            from: "sent",
            to: "cancel",
            trigger: "button_cancel",
            guards: &[],
        },
        OdooTransition {
            from: "to approve",
            to: "cancel",
            trigger: "button_cancel",
            guards: &[],
        },
        OdooTransition {
            from: "purchase",
            to: "cancel",
            trigger: "button_cancel",
            guards: &[],
        },
        OdooTransition {
            from: "cancel",
            to: "draft",
            trigger: "button_draft",
            guards: &[],
        },
    ],
};

pub const PURCHASE_ORDER: OdooEntity = OdooEntity {
    model_name: "purchase.order",
    description: "Vendor purchase order (RFQ → PO); optional two-step approval via \
                  po_double_validation amount threshold; creates `account.move` \
                  (in_invoice / vendor bill) via action_create_invoice. \
                  OWL pivot proposed: ubl:Order → SmbFoundryInvoice (0x81) → DOLCE Perdurant.",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: Some("_compute_name"),
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "state",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // draft | sent | to approve | purchase | cancel
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "partner_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "currency_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.currency"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "date_order",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "date_approve",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Set to now() by button_approve.
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "order_line",
            kind: OdooFieldKind::One2many,
            target: Some("purchase.order.line"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "invoice_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("account.move"),
            required: false,
            computed: Some("_compute_invoice"),
            depends: &[],
            semantic_role: OdooSemanticRole::Document,
        },
        OdooField {
            name: "invoice_status",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: Some("_get_invoiced"),
            depends: &["state", "order_line.qty_to_invoice"],
            // Only 3 values: 'no' | 'to invoice' | 'invoiced'.
            // NO 'upselling' (purchase has no qty_delivered > qty_ordered concept).
            // 'invoiced' requires invoice_ids to be non-empty.
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "amount_untaxed",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_amount_all"),
            depends: &["order_line.price_subtotal", "company_id", "currency_id"],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "amount_tax",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_amount_all"),
            depends: &["order_line.price_subtotal", "company_id", "currency_id"],
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "amount_total",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_amount_all"),
            depends: &["order_line.price_subtotal", "company_id", "currency_id"],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "amount_total_cc",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_amount_all"),
            depends: &["order_line.price_subtotal", "company_id", "currency_id"],
            // Company-currency total (not order currency); purchase-only field.
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "fiscal_position_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.fiscal.position"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "payment_term_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.payment.term"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "locked",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Set when lock_confirmed_po='lock' on button_approve. Blocks cancel.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "note",
            kind: OdooFieldKind::Html,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Copied as 'narration' in _prepare_invoice (purchase variant).
            semantic_role: OdooSemanticRole::Document,
        },
    ],
    methods: &[
        OdooMethod {
            name: "button_confirm",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Boolean,
            triggers: &["purchase", "to approve"],
            // Guard → validate analytics → _add_supplier_to_product → _approval_allowed path.
        },
        OdooMethod {
            name: "button_approve",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["purchase"],
        },
        OdooMethod {
            name: "button_cancel",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["cancel"],
            // Guard: locked OR has non-draft/non-cancel invoices → UserError.
        },
        OdooMethod {
            name: "button_draft",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["draft"],
        },
        OdooMethod {
            name: "_amount_all",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Same pipeline as sale _compute_amounts but no EPD lines; adds amount_total_cc.
        },
        OdooMethod {
            name: "_get_invoiced",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "action_create_invoice",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
            // Creates in_invoice; groups by (company_id, partner_id, currency_id).
            // No 'final' parameter; negative total → out_refund switch.
        },
        OdooMethod {
            name: "_prepare_invoice",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
            // Produces in_invoice vals; includes partner_bank_id (vendor bank).
            // Does NOT copy team_id, campaign_id, payment_reference vs sale variant.
        },
        OdooMethod {
            name: "_approval_allowed",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
            // one_step config OR amount < threshold OR user has purchase_manager group.
        },
        OdooMethod {
            name: "_add_supplier_to_product",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
            // Side effect on confirm: adds vendor to product.supplierinfo (max 10).
        },
        OdooMethod {
            name: "_confirmation_error_message",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["order_line.price_subtotal", "company_id", "currency_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["state", "order_line.qty_to_invoice"],
        },
    ],
    state_machine: Some(&PURCHASE_ORDER_STATE_MACHINE),
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "cannot delete PO unless state='cancel'",
            source_method: Some("_unlink_if_cancelled"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "cannot cancel if locked or has non-draft/non-cancel invoices",
            source_method: Some("button_cancel"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L6-SALE-PURCHASE.md",
        l_doc_lines: (546, 643),
        odoo_source: &[],
        confidence: OdooConfidence::Curated,
    },
};

// ─── purchase.order.line ─────────────────────────────────────────────────────

pub const PURCHASE_ORDER_LINE: OdooEntity = OdooEntity {
    model_name: "purchase.order.line",
    description: "One line on a purchase order; tracks qty_to_invoice / qty_received; \
                  per-line invoice_status feeds the order-level `_get_invoiced` computation. \
                  OWL pivot proposed: ubl:OrderLine → SmbFoundryInvoice (0x81) → DOLCE Perdurant.",
    fields: &[
        OdooField {
            name: "order_id",
            kind: OdooFieldKind::Many2one,
            target: Some("purchase.order"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "product_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.product"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "product_qty",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "product_uom",
            kind: OdooFieldKind::Many2one,
            target: Some("uom.uom"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "price_unit",
            kind: OdooFieldKind::Float,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "taxes_id",
            kind: OdooFieldKind::Many2many,
            target: Some("account.tax"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "price_subtotal",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_compute_amount"),
            depends: &["product_qty", "price_unit", "taxes_id"],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "qty_received",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_qty_received"),
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "qty_invoiced",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_qty_invoiced"),
            depends: &["invoice_lines.move_id.state", "invoice_lines.quantity"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "qty_to_invoice",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_qty_to_invoice"),
            depends: &["qty_invoiced", "qty_received", "product_qty", "order_id.state"],
            // Used by order-level _get_invoiced (L46-68).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "invoice_lines",
            kind: OdooFieldKind::Many2many,
            target: Some("account.move.line"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Document,
        },
        OdooField {
            name: "display_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Excluded from invoice_status calculation (section/note lines).
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "date_planned",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Planned delivery date for this line.
            semantic_role: OdooSemanticRole::Date,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_amount",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_qty_invoiced",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_qty_to_invoice",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_qty_received",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_qty", "price_unit", "taxes_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["invoice_lines.move_id.state", "invoice_lines.quantity"],
        },
    ],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L6-SALE-PURCHASE.md",
        l_doc_lines: (611, 643),
        odoo_source: &[],
        confidence: OdooConfidence::Curated,
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// Entities documented in lane L6 (sale order + purchase order + upsell
/// + pricelist application).
///
/// 4 entities:
///   [0] `sale.order`          — K3 gate, Vorgang lifecycle, state machine, invoice creation
///   [1] `sale.order.line`     — price/discount/tax, partial invoicing, upselling detection
///   [2] `purchase.order`      — RFQ/PO lifecycle, two-step approval, vendor bill creation
///   [3] `purchase.order.line` — qty tracking for purchase invoice_status
///
/// Entities NOT included (skipped / deferred):
///   - `product.pricelist`      — Primary coverage in L8 (PRODUCT-UOM-PRICELIST). L6 only
///                                references `pricelist_id` as a field; no new fields projected.
///   - `product.pricelist.item` — Same; L8 is the canonical lane. See module-level note.
pub const ENTITIES: &[OdooEntity] = &[
    SALE_ORDER,
    SALE_ORDER_LINE,
    PURCHASE_ORDER,
    PURCHASE_ORDER_LINE,
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooStateSemantic};

    #[test]
    fn entities_slice_has_4_entries() {
        assert_eq!(ENTITIES.len(), 4);
    }

    #[test]
    fn all_entities_cite_l6_doc() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.l_doc,
                "L6-SALE-PURCHASE.md",
                "entity {} must cite L6 doc",
                entity.model_name
            );
        }
    }

    #[test]
    fn all_entities_are_curated() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.confidence,
                OdooConfidence::Curated,
                "entity {} must be Curated",
                entity.model_name
            );
        }
    }

    #[test]
    fn sale_order_state_machine_no_done_state() {
        let sm = SALE_ORDER
            .state_machine
            .expect("sale.order must have a state machine");
        assert_eq!(sm.state_field, "state");
        // Community Odoo 17: no 'done' state — locked boolean replaces it.
        assert!(
            sm.states.iter().all(|s| s.name != "done"),
            "sale.order must NOT have a 'done' state in Odoo 17 community"
        );
        let names: Vec<&str> = sm.states.iter().map(|s| s.name).collect();
        assert!(names.contains(&"draft"));
        assert!(names.contains(&"sent"));
        assert!(names.contains(&"sale"));
        assert!(names.contains(&"cancel"));
    }

    #[test]
    fn sale_order_action_confirm_transitions_to_sale() {
        let sm = SALE_ORDER
            .state_machine
            .expect("sale.order must have a state machine");
        let confirm_txn = sm
            .transitions
            .iter()
            .find(|t| t.trigger == "action_confirm" && t.to == "sale");
        assert!(
            confirm_txn.is_some(),
            "action_confirm must have a transition to 'sale'"
        );
    }

    #[test]
    fn purchase_order_has_to_approve_state() {
        let sm = PURCHASE_ORDER
            .state_machine
            .expect("purchase.order must have a state machine");
        let to_approve = sm.states.iter().find(|s| s.name == "to approve");
        assert!(to_approve.is_some(), "'to approve' state must be present");
        // 'to approve' is a waiting/active state (not cancelled or terminal).
        assert_eq!(
            to_approve.unwrap().semantic,
            OdooStateSemantic::Active
        );
    }

    #[test]
    fn purchase_order_invoice_status_has_no_upselling() {
        // purchase.order only has 'no'/'to invoice'/'invoiced' — no upselling.
        let f = PURCHASE_ORDER
            .fields
            .iter()
            .find(|f| f.name == "invoice_status")
            .expect("invoice_status field must be present on purchase.order");
        // Verify it is computed via _get_invoiced.
        assert_eq!(f.computed, Some("_get_invoiced"));
    }

    #[test]
    fn sale_order_line_qty_to_invoice_depends_on_state() {
        let f = SALE_ORDER_LINE
            .fields
            .iter()
            .find(|f| f.name == "qty_to_invoice")
            .expect("qty_to_invoice must be present on sale.order.line");
        assert!(
            f.depends.contains(&"state"),
            "qty_to_invoice must depend on state"
        );
        assert_eq!(f.computed, Some("_compute_qty_to_invoice"));
    }

    #[test]
    fn sale_order_has_sql_constraint_on_date_order() {
        let sql_constraints: Vec<&str> = SALE_ORDER
            .constraints
            .iter()
            .filter(|c| c.kind == crate::odoo_blueprint::OdooConstraintKind::Sql)
            .map(|c| c.condition)
            .collect();
        assert!(
            !sql_constraints.is_empty(),
            "sale.order must have at least one SQL constraint (date_order check)"
        );
    }
}
