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
//! referenced in L6 (Rule S-3 `_compute_discount`) but their canonical projection
//! lives in L8-PRODUCT-UOM-PRICELIST.  They are NOT duplicated here.
//!
//! Odoo source files read by the L6 author (depth=full):
//!   `addons/sale/models/sale_order.py`         (2301 lines)
//!   `addons/sale/models/sale_order_line.py`    (1819 lines)
//!   `addons/purchase/models/purchase_order.py` (1418 lines)

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooState, OdooStateMachine, OdooStateSemantic,
    OdooTransition,
};

// ─── sale.order ──────────────────────────────────────────────────────────────

// Community Odoo 17: states are draft/sent/sale/cancel — NO 'done' state.
// The `locked` boolean (separate field) replaces the old 'done' state.
// Source: sale_order.py L26-31, L70-76.
const SALE_ORDER_SM: OdooStateMachine = OdooStateMachine {
    state_field: "state",
    states: &[
        OdooState { name: "draft",  semantic: OdooStateSemantic::Draft       },
        OdooState { name: "sent",   semantic: OdooStateSemantic::Active       },
        OdooState { name: "sale",   semantic: OdooStateSemantic::InProgress   },
        OdooState { name: "cancel", semantic: OdooStateSemantic::Cancelled    },
    ],
    transitions: &[
        OdooTransition {
            from: "draft", to: "sent", trigger: "action_quotation_sent", guards: &[],
        },
        OdooTransition {
            from: "draft", to: "sale", trigger: "action_confirm",
            guards: &["_confirmation_error_message"],
            // Full flow L1166-1196: analytics validate → write state+date_order →
            // _action_confirm hook → optional auto-lock → optional confirm email.
        },
        OdooTransition {
            from: "sent", to: "sale", trigger: "action_confirm",
            guards: &["_confirmation_error_message"],
        },
        OdooTransition {
            from: "cancel", to: "draft", trigger: "action_draft", guards: &[],
            // Also clears signature/signed_by/signed_on (L1058-1065).
        },
        OdooTransition {
            from: "sent", to: "draft", trigger: "action_draft", guards: &[],
        },
        OdooTransition {
            from: "sale", to: "cancel", trigger: "action_cancel", guards: &[],
            // Guard: raises UserError if locked; _action_cancel cancels draft invoices
            // first, then writes state='cancel' (L1324-1333).
        },
        OdooTransition {
            from: "draft", to: "cancel", trigger: "action_cancel", guards: &[],
        },
        OdooTransition {
            from: "sent", to: "cancel", trigger: "action_cancel", guards: &[],
        },
    ],
};

pub const SALE_ORDER: OdooEntity = OdooEntity {
    model_name: "sale.order",
    description: "Commercial sale quotation/order (Vorgang lifecycle: draft→sent→sale→cancel); \
                  generates account.move via _create_invoices. \
                  Proposed OWL pivot: ubl:Order → SmbFoundryInvoice (0x81) → DOLCE Perdurant.",
    fields: &[
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None, required: false,
            computed: Some("_compute_name"), depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "state", kind: OdooFieldKind::Selection, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
        OdooField { name: "locked", kind: OdooFieldKind::Boolean, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy,
            // Replaces old 'done' state; set by action_lock.
        },
        OdooField { name: "date_order", kind: OdooFieldKind::Datetime, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Date,
            // OVERWRITTEN to now() on action_confirm (L1166). DB CHECK: non-null when state='sale'.
        },
        OdooField { name: "partner_id", kind: OdooFieldKind::Many2one, target: Some("res.partner"),
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "partner_invoice_id", kind: OdooFieldKind::Many2one, target: Some("res.partner"),
            required: true, computed: Some("_compute_partner_invoice_id"), depends: &["partner_id"],
            semantic_role: OdooSemanticRole::Address },
        OdooField { name: "partner_shipping_id", kind: OdooFieldKind::Many2one, target: Some("res.partner"),
            required: true, computed: Some("_compute_partner_shipping_id"), depends: &["partner_id"],
            semantic_role: OdooSemanticRole::Address },
        OdooField { name: "pricelist_id", kind: OdooFieldKind::Many2one, target: Some("product.pricelist"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy,
            // Cannot change on confirmed order (write() guard L1040-1043).
            // Primary model: L8-PRODUCT-UOM-PRICELIST.
        },
        OdooField { name: "currency_id", kind: OdooFieldKind::Many2one, target: Some("res.currency"),
            required: false, computed: Some("_compute_currency_id"),
            depends: &["pricelist_id.currency_id", "company_id.currency_id"],
            semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "order_line", kind: OdooFieldKind::One2many, target: Some("sale.order.line"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "invoice_ids", kind: OdooFieldKind::Many2many, target: Some("account.move"),
            required: false, computed: Some("_compute_invoice_ids"),
            depends: &["order_line.invoice_lines.move_id"],
            semantic_role: OdooSemanticRole::Document },
        OdooField { name: "invoice_status", kind: OdooFieldKind::Selection, target: None,
            required: false, computed: Some("_compute_invoice_status"),
            depends: &["state", "order_line.invoice_status"],
            // Values: 'upselling' | 'invoiced' | 'to invoice' | 'no'.
            // upselling → creates TODO activity for salesperson (L1964-1975).
            semantic_role: OdooSemanticRole::Status },
        OdooField { name: "amount_untaxed", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_compute_amounts"),
            depends: &["order_line.price_subtotal", "currency_id", "company_id", "payment_term_id"],
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "amount_tax", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_compute_amounts"),
            depends: &["order_line.price_subtotal", "currency_id", "company_id", "payment_term_id"],
            semantic_role: OdooSemanticRole::Tax },
        OdooField { name: "amount_total", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_compute_amounts"),
            depends: &["order_line.price_subtotal", "currency_id", "company_id", "payment_term_id"],
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "payment_term_id", kind: OdooFieldKind::Many2one,
            target: Some("account.payment.term"), required: false, computed: None, depends: &[],
            semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "fiscal_position_id", kind: OdooFieldKind::Many2one,
            target: Some("account.fiscal.position"), required: false, computed: None, depends: &[],
            semantic_role: OdooSemanticRole::Tax },
        OdooField { name: "user_id", kind: OdooFieldKind::Many2one, target: Some("res.users"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "team_id", kind: OdooFieldKind::Many2one, target: Some("crm.team"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "company_id", kind: OdooFieldKind::Many2one, target: Some("res.company"),
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "client_order_ref", kind: OdooFieldKind::Char, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "note", kind: OdooFieldKind::Html, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Document },
    ],
    methods: &[
        OdooMethod { name: "action_quotation_sent", kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit, triggers: &["sent"] },
        OdooMethod { name: "action_confirm", kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Boolean, triggers: &["sale"] },
        OdooMethod { name: "action_draft", kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit, triggers: &["draft"] },
        OdooMethod { name: "action_cancel", kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit, triggers: &["cancel"] },
        OdooMethod { name: "action_lock", kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_action_cancel", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_amounts", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_invoice_status", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_prepare_invoice", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_get_invoiceable_lines", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset, triggers: &[] },
        OdooMethod { name: "_create_invoices", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset, triggers: &[] },
        OdooMethod { name: "_confirmation_error_message", kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Boolean, triggers: &[] },
        OdooMethod { name: "_unlink_except_draft_or_cancel", kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["order_line.price_subtotal", "currency_id", "company_id", "payment_term_id"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["state", "order_line.invoice_status"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiConstrains, targets: &["state"] },
    ],
    state_machine: Some(&SALE_ORDER_SM),
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "(state='sale' AND date_order IS NOT NULL) OR state!='sale' (L41-44)",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "pricelist_id cannot change on confirmed (state='sale') order",
            source_method: Some("write"),
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
    description: "One line on a sale order; tracks price_unit/discount/tax_ids and \
                  partial invoicing state (qty_invoiced/qty_to_invoice/invoice_status). \
                  Proposed OWL pivot: ubl:OrderLine → SmbFoundryInvoice (0x81) → DOLCE Perdurant.",
    fields: &[
        OdooField { name: "order_id", kind: OdooFieldKind::Many2one, target: Some("sale.order"),
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "product_id", kind: OdooFieldKind::Many2one, target: Some("product.product"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None,
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "product_uom_qty", kind: OdooFieldKind::Float, target: None,
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "product_uom_id", kind: OdooFieldKind::Many2one, target: Some("uom.uom"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "price_unit", kind: OdooFieldKind::Float, target: None,
            required: true, computed: Some("_compute_price_unit"),
            depends: &["product_id", "product_uom_id", "product_uom_qty"],
            // Frozen once qty_invoiced>0. Manual edit detected via technical_price_unit.
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "technical_price_unit", kind: OdooFieldKind::Float, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Other,
            // Shadow field: system-computed price; diff from price_unit → manual edit.
        },
        OdooField { name: "discount", kind: OdooFieldKind::Float, target: None,
            required: false, computed: Some("_compute_discount"),
            depends: &["product_id", "product_uom_id", "product_uom_qty"],
            // (base_price - pricelist_price) / base_price * 100; shows only if > 0.
            semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "tax_ids", kind: OdooFieldKind::Many2many, target: Some("account.tax"),
            required: false, computed: Some("_compute_tax_ids"),
            depends: &["product_id", "company_id"],
            // Mapped through fiscal_position; combo products → tax_ids=False.
            semantic_role: OdooSemanticRole::Tax },
        OdooField { name: "price_subtotal", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_compute_amount"),
            depends: &["product_uom_qty", "discount", "price_unit", "tax_ids"],
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "price_total", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_compute_amount"),
            depends: &["product_uom_qty", "discount", "price_unit", "tax_ids"],
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "price_tax", kind: OdooFieldKind::Float, target: None,
            required: false, computed: Some("_compute_amount"),
            depends: &["product_uom_qty", "discount", "price_unit", "tax_ids"],
            semantic_role: OdooSemanticRole::Tax },
        OdooField { name: "qty_delivered", kind: OdooFieldKind::Float, target: None,
            required: false, computed: Some("_compute_qty_delivered"), depends: &[],
            semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "qty_invoiced", kind: OdooFieldKind::Float, target: None,
            required: false, computed: Some("_compute_qty_invoiced"),
            depends: &["invoice_lines.move_id.state", "invoice_lines.quantity"],
            // out_refund DECREASES this; cancelled invoices excluded; round=False.
            semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "qty_to_invoice", kind: OdooFieldKind::Float, target: None,
            required: false, computed: Some("_compute_qty_to_invoice"),
            depends: &["qty_invoiced", "qty_delivered", "product_uom_qty", "state"],
            // invoice_policy='order': product_uom_qty - qty_invoiced.
            // invoice_policy='delivery': qty_delivered - qty_invoiced. 0 when state!='sale'.
            semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "invoice_status", kind: OdooFieldKind::Selection, target: None,
            required: false, computed: Some("_compute_invoice_status"),
            depends: &["state", "product_uom_qty", "qty_delivered", "qty_to_invoice", "qty_invoiced"],
            // Decision tree: no → invoiced → to invoice → upselling → invoiced → no.
            semantic_role: OdooSemanticRole::Status },
        OdooField { name: "invoice_lines", kind: OdooFieldKind::Many2many,
            target: Some("account.move.line"), required: false, computed: None, depends: &[],
            semantic_role: OdooSemanticRole::Document },
        OdooField { name: "is_downpayment", kind: OdooFieldKind::Boolean, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "display_type", kind: OdooFieldKind::Selection, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Other },
        OdooField { name: "untaxed_amount_to_invoice", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_compute_untaxed_amount_to_invoice"),
            depends: &["state", "product_id", "untaxed_amount_invoiced",
                       "qty_delivered", "product_uom_qty", "price_unit"],
            // Floored at max(..., 0). Handles discount drift in re-invoicing.
            semantic_role: OdooSemanticRole::Money },
    ],
    methods: &[
        OdooMethod { name: "_compute_tax_ids", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_price_unit", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_reset_price_unit", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_discount", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_amount", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_qty_invoiced", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_qty_to_invoice", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_invoice_status", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_untaxed_amount_to_invoice", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_prepare_invoice_line", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_prepare_base_line_for_taxes_computation", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_validate_analytic_distribution", kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_id", "company_id"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_id", "product_uom_id", "product_uom_qty"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_uom_qty", "discount", "price_unit", "tax_ids"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["invoice_lines.move_id.state", "invoice_lines.quantity"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["qty_invoiced", "qty_delivered", "product_uom_qty", "state"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["state", "product_uom_qty", "qty_delivered", "qty_to_invoice", "qty_invoiced"] },
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

// States: draft / sent / to approve / purchase / cancel.
// 'to approve' is intermediate when two-step PO validation is active.
// Source: purchase_order.py L105-111.
const PURCHASE_ORDER_SM: OdooStateMachine = OdooStateMachine {
    state_field: "state",
    states: &[
        OdooState { name: "draft",      semantic: OdooStateSemantic::Draft      },
        OdooState { name: "sent",       semantic: OdooStateSemantic::Active      },
        OdooState { name: "to approve", semantic: OdooStateSemantic::Active      },
        OdooState { name: "purchase",   semantic: OdooStateSemantic::InProgress  },
        OdooState { name: "cancel",     semantic: OdooStateSemantic::Cancelled   },
    ],
    transitions: &[
        OdooTransition {
            from: "draft", to: "to approve", trigger: "button_confirm",
            guards: &["_confirmation_error_message", "_approval_allowed"],
            // Two-step path: po_double_validation='two_step' AND amount>=threshold
            // AND user NOT in purchase.group_purchase_manager.
        },
        OdooTransition {
            from: "sent", to: "to approve", trigger: "button_confirm",
            guards: &["_confirmation_error_message", "_approval_allowed"],
        },
        OdooTransition {
            from: "draft", to: "purchase", trigger: "button_confirm",
            guards: &["_confirmation_error_message", "_approval_allowed"],
            // One-step path. Side effect: _add_supplier_to_product() (max 10 vendors).
        },
        OdooTransition {
            from: "sent", to: "purchase", trigger: "button_confirm",
            guards: &["_confirmation_error_message", "_approval_allowed"],
        },
        OdooTransition {
            from: "to approve", to: "purchase", trigger: "button_approve",
            guards: &[],
            // Writes state='purchase' + date_approve=now(); optional lock.
        },
        OdooTransition {
            from: "draft",      to: "cancel", trigger: "button_cancel", guards: &[],
        },
        OdooTransition {
            from: "sent",       to: "cancel", trigger: "button_cancel", guards: &[],
        },
        OdooTransition {
            from: "to approve", to: "cancel", trigger: "button_cancel", guards: &[],
        },
        OdooTransition {
            from: "purchase",   to: "cancel", trigger: "button_cancel", guards: &[],
            // Guard: raises UserError if locked or has non-draft/non-cancel invoices.
        },
        OdooTransition {
            from: "cancel", to: "draft", trigger: "button_draft", guards: &[],
        },
    ],
};

pub const PURCHASE_ORDER: OdooEntity = OdooEntity {
    model_name: "purchase.order",
    description: "Vendor purchase order (RFQ→PO); optional two-step approval via \
                  po_double_validation amount threshold; creates account.move (in_invoice) \
                  via action_create_invoice. \
                  Proposed OWL pivot: ubl:Order → SmbFoundryInvoice (0x81) → DOLCE Perdurant.",
    fields: &[
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None,
            required: false, computed: Some("_compute_name"), depends: &[],
            semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "state", kind: OdooFieldKind::Selection, target: None,
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
        OdooField { name: "partner_id", kind: OdooFieldKind::Many2one, target: Some("res.partner"),
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "currency_id", kind: OdooFieldKind::Many2one, target: Some("res.currency"),
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "date_order", kind: OdooFieldKind::Datetime, target: None,
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Date },
        OdooField { name: "date_approve", kind: OdooFieldKind::Datetime, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Date,
            // Set to now() by button_approve.
        },
        OdooField { name: "order_line", kind: OdooFieldKind::One2many,
            target: Some("purchase.order.line"), required: false, computed: None, depends: &[],
            semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "invoice_ids", kind: OdooFieldKind::Many2many,
            target: Some("account.move"), required: false, computed: Some("_compute_invoice"),
            depends: &[], semantic_role: OdooSemanticRole::Document },
        OdooField { name: "invoice_status", kind: OdooFieldKind::Selection, target: None,
            required: false, computed: Some("_get_invoiced"),
            depends: &["state", "order_line.qty_to_invoice"],
            // Values: 'no' | 'to invoice' | 'invoiced'. NO 'upselling' (purchase only).
            // 'invoiced' requires invoice_ids non-empty (L46-68).
            semantic_role: OdooSemanticRole::Status },
        OdooField { name: "amount_untaxed", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_amount_all"),
            depends: &["order_line.price_subtotal", "company_id", "currency_id"],
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "amount_tax", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_amount_all"),
            depends: &["order_line.price_subtotal", "company_id", "currency_id"],
            semantic_role: OdooSemanticRole::Tax },
        OdooField { name: "amount_total", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_amount_all"),
            depends: &["order_line.price_subtotal", "company_id", "currency_id"],
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "amount_total_cc", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_amount_all"),
            depends: &["order_line.price_subtotal", "company_id", "currency_id"],
            // Company-currency total; purchase-only (not in sale variant).
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "fiscal_position_id", kind: OdooFieldKind::Many2one,
            target: Some("account.fiscal.position"), required: false, computed: None, depends: &[],
            semantic_role: OdooSemanticRole::Tax },
        OdooField { name: "payment_term_id", kind: OdooFieldKind::Many2one,
            target: Some("account.payment.term"), required: false, computed: None, depends: &[],
            semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "company_id", kind: OdooFieldKind::Many2one, target: Some("res.company"),
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "locked", kind: OdooFieldKind::Boolean, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "note", kind: OdooFieldKind::Html, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Document },
    ],
    methods: &[
        OdooMethod { name: "button_confirm", kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Boolean, triggers: &["purchase", "to approve"] },
        OdooMethod { name: "button_approve", kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit, triggers: &["purchase"] },
        OdooMethod { name: "button_cancel", kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit, triggers: &["cancel"] },
        OdooMethod { name: "button_draft", kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit, triggers: &["draft"] },
        OdooMethod { name: "_amount_all", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_get_invoiced", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "action_create_invoice", kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action, triggers: &[] },
        OdooMethod { name: "_prepare_invoice", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_approval_allowed", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Boolean, triggers: &[] },
        OdooMethod { name: "_add_supplier_to_product", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_confirmation_error_message", kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Boolean, triggers: &[] },
    ],
    decorators: &[
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["order_line.price_subtotal", "company_id", "currency_id"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["state", "order_line.qty_to_invoice"] },
    ],
    state_machine: Some(&PURCHASE_ORDER_SM),
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
    description: "One line on a purchase order; qty_to_invoice feeds the order-level \
                  _get_invoiced computation (L46-68). \
                  Proposed OWL pivot: ubl:OrderLine → SmbFoundryInvoice (0x81) → DOLCE Perdurant.",
    fields: &[
        OdooField { name: "order_id", kind: OdooFieldKind::Many2one, target: Some("purchase.order"),
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "product_id", kind: OdooFieldKind::Many2one, target: Some("product.product"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None,
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "product_qty", kind: OdooFieldKind::Float, target: None,
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "product_uom", kind: OdooFieldKind::Many2one, target: Some("uom.uom"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "price_unit", kind: OdooFieldKind::Float, target: None,
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Money },
        OdooField { name: "taxes_id", kind: OdooFieldKind::Many2many, target: Some("account.tax"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Tax },
        OdooField { name: "price_subtotal", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_compute_amount"),
            depends: &["product_qty", "price_unit", "taxes_id"],
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "qty_received", kind: OdooFieldKind::Float, target: None,
            required: false, computed: Some("_compute_qty_received"), depends: &[],
            semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "qty_invoiced", kind: OdooFieldKind::Float, target: None,
            required: false, computed: Some("_compute_qty_invoiced"),
            depends: &["invoice_lines.move_id.state", "invoice_lines.quantity"],
            semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "qty_to_invoice", kind: OdooFieldKind::Float, target: None,
            required: false, computed: Some("_compute_qty_to_invoice"),
            depends: &["qty_invoiced", "qty_received", "product_qty", "order_id.state"],
            semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "invoice_lines", kind: OdooFieldKind::Many2many,
            target: Some("account.move.line"), required: false, computed: None, depends: &[],
            semantic_role: OdooSemanticRole::Document },
        OdooField { name: "display_type", kind: OdooFieldKind::Selection, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Other },
        OdooField { name: "date_planned", kind: OdooFieldKind::Datetime, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Date },
    ],
    methods: &[
        OdooMethod { name: "_compute_amount", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_qty_invoiced", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_qty_to_invoice", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_qty_received", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["product_qty", "price_unit", "taxes_id"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["invoice_lines.move_id.state", "invoice_lines.quantity"] },
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
///   [3] `purchase.order.line` — qty_to_invoice for purchase invoice_status
///
/// Skipped / deferred:
///   - `product.pricelist`      — primary coverage in L8-PRODUCT-UOM-PRICELIST (see module note)
///   - `product.pricelist.item` — same; L8 is canonical
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
    use crate::odoo_blueprint::{OdooConfidence, OdooConstraintKind, OdooStateSemantic};

    #[test]
    fn entities_slice_has_4_entries() {
        assert_eq!(ENTITIES.len(), 4);
    }

    #[test]
    fn all_entities_cite_l6_doc() {
        for entity in ENTITIES {
            assert_eq!(entity.provenance.l_doc, "L6-SALE-PURCHASE.md",
                "entity {} must cite L6 doc", entity.model_name);
        }
    }

    #[test]
    fn all_entities_are_curated() {
        for entity in ENTITIES {
            assert_eq!(entity.provenance.confidence, OdooConfidence::Curated,
                "entity {} must be Curated", entity.model_name);
        }
    }

    #[test]
    fn sale_order_state_machine_no_done_state() {
        let sm = SALE_ORDER.state_machine.expect("sale.order must have a state machine");
        assert_eq!(sm.state_field, "state");
        // Community Odoo 17: no 'done' state — locked boolean replaces it.
        assert!(sm.states.iter().all(|s| s.name != "done"),
            "sale.order must NOT have a 'done' state in Odoo 17 community");
        let names: Vec<&str> = sm.states.iter().map(|s| s.name).collect();
        assert!(names.contains(&"draft") && names.contains(&"sent")
            && names.contains(&"sale") && names.contains(&"cancel"));
    }

    #[test]
    fn sale_order_action_confirm_transitions_to_sale() {
        let sm = SALE_ORDER.state_machine.expect("sale.order must have a state machine");
        assert!(sm.transitions.iter().any(|t| t.trigger == "action_confirm" && t.to == "sale"),
            "action_confirm must have a transition to 'sale'");
    }

    #[test]
    fn purchase_order_has_to_approve_state() {
        let sm = PURCHASE_ORDER.state_machine.expect("purchase.order must have a state machine");
        let to_approve = sm.states.iter().find(|s| s.name == "to approve");
        assert!(to_approve.is_some(), "'to approve' state must be present");
        assert_eq!(to_approve.unwrap().semantic, OdooStateSemantic::Active);
    }

    #[test]
    fn purchase_order_invoice_status_computed_by_get_invoiced() {
        let f = PURCHASE_ORDER.fields.iter().find(|f| f.name == "invoice_status")
            .expect("invoice_status must be present on purchase.order");
        assert_eq!(f.computed, Some("_get_invoiced"));
    }

    #[test]
    fn sale_order_line_qty_to_invoice_depends_on_state() {
        let f = SALE_ORDER_LINE.fields.iter().find(|f| f.name == "qty_to_invoice")
            .expect("qty_to_invoice must be present");
        assert!(f.depends.contains(&"state"), "qty_to_invoice must depend on state");
        assert_eq!(f.computed, Some("_compute_qty_to_invoice"));
    }

    #[test]
    fn sale_order_has_sql_constraint_on_date_order() {
        assert!(SALE_ORDER.constraints.iter()
            .any(|c| c.kind == OdooConstraintKind::Sql),
            "sale.order must have at least one SQL constraint (date_order check)");
    }
}
