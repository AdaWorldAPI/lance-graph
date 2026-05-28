//! Lane L5 (PAY-TERMS-MATCH) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L5-PAY-TERMS-MATCH.md`.
//!
//! Entities covered (6):
//!   - `account.payment`              (RULE P1, P4)
//!   - `account.payment.term`         (RULE P2, P3)
//!   - `account.payment.term.line`    (RULE P2 sub-model)
//!   - `account.payment.method`       (referenced via payment_method_line_id)
//!   - `account.payment.method.line`  (referenced via payment_method_line_id)
//!   - `account.reconcile.model`      (RULE P5)
//!   - `account.reconcile.model.line` (RULE P5 sub-model)
//!
//! NOTE: `account.reconcile.model` and `account.reconcile.model.line` are
//! noted as L2 candidates (L2-K3-RECON covers bank-statement matching), but
//! L2 is still a stub as of D-ODOO-BP-1b.  L5 is the first lane that fully
//! documents those entities (RULE P5), so they are projected here.  When L2
//! is populated, these two entities should be cited from L2 and removed from
//! L5 to avoid double-coverage.

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooEntityKind, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind,
    OdooProvenance, OdooReturnKind, OdooSemanticRole, OdooSourceRef, OdooState,
    OdooStateMachine, OdooStateSemantic, OdooTransition,
};

// ─── account.payment ─────────────────────────────────────────────────────────

const PAYMENT_STATE_MACHINE: OdooStateMachine = OdooStateMachine {
    state_field: "state",
    states: &[
        OdooState { name: "draft", semantic: OdooStateSemantic::Draft },
        OdooState { name: "in_process", semantic: OdooStateSemantic::InProgress },
        OdooState { name: "paid", semantic: OdooStateSemantic::Completed },
        OdooState { name: "canceled", semantic: OdooStateSemantic::Cancelled },
        OdooState { name: "rejected", semantic: OdooStateSemantic::Terminal },
    ],
    transitions: &[
        OdooTransition {
            from: "draft",
            to: "in_process",
            trigger: "action_post",
            guards: &["_check_payment_method_line_id"],
        },
        OdooTransition {
            from: "draft",
            to: "paid",
            trigger: "action_post",
            guards: &["_check_payment_method_line_id"],
            // NOTE: only fires when journal uses asset_cash account (L1141)
        },
        OdooTransition {
            from: "in_process",
            to: "paid",
            trigger: "_compute_state",
            guards: &[],
            // NOTE: fires when liquidity residual == 0 OR reconcile flag absent
        },
        OdooTransition {
            from: "in_process",
            to: "canceled",
            trigger: "action_cancel",
            guards: &[],
        },
        OdooTransition {
            from: "draft",
            to: "canceled",
            trigger: "action_cancel",
            guards: &[],
        },
        OdooTransition {
            from: "in_process",
            to: "draft",
            trigger: "action_draft",
            guards: &[],
        },
        OdooTransition {
            from: "canceled",
            to: "draft",
            trigger: "action_draft",
            guards: &[],
        },
    ],
};

pub const PAYMENT: OdooEntity = OdooEntity {
    model_name: "account.payment",
    kind: OdooEntityKind::Model,
    description: "A posted payment event generating double-entry journal lines; \
                  tracks bank-match (is_matched) and invoice-clearance (is_reconciled) \
                  status independently (K3 + K5 / Mahnwesen gate).",
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
            name: "date",
            kind: OdooFieldKind::Date,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "state",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: Some("_compute_state"),
            depends: &[
                "move_id.line_ids.amount_residual",
                "move_id.line_ids.amount_residual_currency",
                "move_id.line_ids.account_id",
            ],
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "amount",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "payment_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "partner_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "partner_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "journal_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.journal"),
            required: true,
            computed: Some("_compute_journal_id"),
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "currency_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.currency"),
            required: false,
            computed: Some("_compute_currency_id"),
            depends: &["journal_id.currency_id", "journal_id.company_id.currency_id"],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "move_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.move"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Document,
        },
        OdooField {
            name: "outstanding_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: Some("_compute_outstanding_account_id"),
            depends: &["payment_method_line_id.payment_account_id"],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "destination_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: Some("_compute_destination_account_id"),
            depends: &["partner_type", "partner_id"],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "payment_method_line_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.payment.method.line"),
            required: false,
            computed: Some("_compute_payment_method_line_id"),
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "is_reconciled",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: Some("_compute_reconciliation_status"),
            depends: &[
                "move_id.line_ids.amount_residual",
                "move_id.line_ids.amount_residual_currency",
                "move_id.line_ids.account_id",
                "state",
            ],
            // K3: invoice clearance gate — Mahnwesen MUST check this, not is_matched
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "is_matched",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: Some("_compute_reconciliation_status"),
            depends: &[
                "move_id.line_ids.amount_residual",
                "move_id.line_ids.amount_residual_currency",
                "move_id.line_ids.account_id",
                "state",
            ],
            // K5: bank statement match status — independent of is_reconciled
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "memo",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Document,
        },
        OdooField {
            name: "payment_reference",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "partner_bank_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner.bank"),
            required: false,
            computed: Some("_compute_partner_bank_id"),
            depends: &[],
            semantic_role: OdooSemanticRole::Address,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_generate_journal_entry",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_prepare_move_lines_per_type",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_seek_for_lines",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
        },
        OdooMethod {
            name: "_synchronize_to_moves",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_state",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_reconciliation_status",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_destination_account_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_outstanding_account_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "action_post",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["in_process", "paid"],
        },
        OdooMethod {
            name: "action_cancel",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["canceled"],
        },
        OdooMethod {
            name: "action_draft",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["draft"],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &[
                "move_id.line_ids.amount_residual",
                "move_id.line_ids.amount_residual_currency",
                "move_id.line_ids.account_id",
                "state",
            ],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["payment_method_line_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["state", "move_id"],
        },
    ],
    state_machine: Some(&PAYMENT_STATE_MACHINE),
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "CHECK(amount >= 0.0) — payment amount must be non-negative",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "payment_method_line_id must not be null and must match journal",
            source_method: Some("_check_payment_method_line_id"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "posted payment with outstanding_account_id must have move_id",
            source_method: Some("_check_state_move_id"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L5-PAY-TERMS-MATCH.md",
        l_doc_lines: (32, 191),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_payment.py",
            line_range: (1, 1247),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── account.payment.term ────────────────────────────────────────────────────

pub const PAYMENT_TERM: OdooEntity = OdooEntity {
    model_name: "account.payment.term",
    kind: OdooEntityKind::Model,
    description: "Structured payment obligation terms (installments, Skonto/early-discount, \
                  due-date computation); feeds invoice aging and Mahnwesen escalation timing (K3).",
    fields: &[
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
            name: "active",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "line_ids",
            kind: OdooFieldKind::One2many,
            target: Some("account.payment.term.line"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "early_discount",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Toggle: enables Skonto (early payment discount).
            // Constraint: only valid with single-line 100% terms.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "discount_percentage",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Skonto percentage, e.g. 2.0 = "2% Skonto". Must be > 0 when early_discount=True.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "discount_days",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Number of days for early-payment window. Must be > 0 when early_discount=True.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "early_pay_discount_computation",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: Some("_compute_discount_computation"),
            depends: &["company_id"],
            // 'included' (DE/AT/CH): Skonto on gross; 'excluded' (NL): net only;
            // 'mixed' (BE): same as excluded. Derived from company.country_code at creation.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "sequence",
            kind: OdooFieldKind::Integer,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "note",
            kind: OdooFieldKind::Html,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Document,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_terms",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
            // Core term computation: embedded FX rate (NOT live), last line absorbs rounding.
        },
        OdooMethod {
            name: "_compute_discount_computation",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_amount_due_after_discount",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Money,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_lines",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["company_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["line_ids", "early_discount"],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "sum of percent lines == 100; early_discount only on single 100% line; \
                    discount_percentage > 0 and discount_days > 0 when early_discount",
        source_method: Some("_check_lines"),
    }],
    provenance: OdooProvenance {
        l_doc: "L5-PAY-TERMS-MATCH.md",
        l_doc_lines: (193, 341),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_payment_term.py",
            line_range: (11, 279),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── account.payment.term.line ───────────────────────────────────────────────

pub const PAYMENT_TERM_LINE: OdooEntity = OdooEntity {
    model_name: "account.payment.term.line",
    kind: OdooEntityKind::Model,
    description: "One installment line within a payment term; computes due date via \
                  delay_type + nb_days; last line always absorbs residual regardless of type.",
    fields: &[
        OdooField {
            name: "payment_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.payment.term"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "value",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'percent' or 'fixed'; last line absorbs residual regardless.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "value_amount",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_value_amount"),
            depends: &["payment_id"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "delay_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // days_after | days_after_end_of_month | days_after_end_of_next_month |
            // days_end_of_month_on_the — see _get_due_date for edge cases.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "nb_days",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: Some("_compute_days"),
            depends: &["payment_id"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "days_next_month",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // String size=2; int() parsed in _get_due_date; ValueError -> 1; "0" -> end-of-month.
            semantic_role: OdooSemanticRole::Quantity,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_get_due_date",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Date,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_value_amount",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_days",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_percent",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["value", "value_amount"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["days_next_month"],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "percent value_amount must be between 0 and 100",
            source_method: Some("_check_percent"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "days_next_month must be numeric and between 0 and 31",
            source_method: Some("_check_valid_char_value"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L5-PAY-TERMS-MATCH.md",
        l_doc_lines: (296, 330),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_payment_term.py",
            line_range: (281, 368),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── account.reconcile.model ──────────────────────────────────────────────────
//
// NOTE: This entity overlaps with L2-K3-RECON scope (bank-statement / open-item
// reconciliation). L2 is a stub as of D-ODOO-BP-1b; this is the first
// full projection from L5-PAY-TERMS-MATCH.md RULE P5. When L2 is populated,
// move primary coverage there and leave a cross-reference comment here.

pub const RECONCILE_MODEL: OdooEntity = OdooEntity {
    model_name: "account.reconcile.model",
    kind: OdooEntityKind::Model,
    description: "Declarative rule for bank-statement-to-open-item matching \
                  (NARS-heavy, Axis-2 heuristic); greedy first-match by sequence; \
                  generates write-off journal lines on match (K3 + K5).",
    fields: &[
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
            name: "sequence",
            kind: OdooFieldKind::Integer,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Greedy tie-breaker: first model in sequence wins the match.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "trigger",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'manual': propose only; 'auto_reconcile': auto-apply (Enterprise engine).
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "match_journal_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("account.journal"),
            required: false,
            computed: None,
            depends: &[],
            // Hard filter: empty = all journals. Dimension 1 of match.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "match_amount",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // 'lower' | 'greater' | 'between'. Hard filter. Dimension 2 of match.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "match_amount_min",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "match_amount_max",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "match_label",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // 'contains' | 'not_contains' | 'match_regex'. Primary textual evidence. Dimension 3.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "match_label_param",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "match_partner_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("res.partner"),
            required: false,
            computed: None,
            depends: &[],
            // Hard filter on statement partner. Dimension 4 of match.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "line_ids",
            kind: OdooFieldKind::One2many,
            target: Some("account.reconcile.model.line"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "can_be_proposed",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: Some("_compute_can_be_proposed"),
            depends: &[
                "mapped_partner_id",
                "match_label",
                "match_amount",
                "match_partner_ids",
                "trigger",
            ],
            // False when model is partner-mapping only (lookup, not reconcile candidate).
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "mapped_partner_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: false,
            computed: Some("_compute_partner_mapping"),
            depends: &["match_label", "line_ids.partner_id", "line_ids.account_id"],
            // Set when model is a partner-mapping rule (one line: partner, no account).
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_can_be_proposed",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_partner_mapping",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "action_reconcile_stat",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
        OdooMethod {
            name: "action_set_manual",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["manual"],
        },
        OdooMethod {
            name: "action_set_auto_reconcile",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &["auto_reconcile"],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &[
                "mapped_partner_id",
                "match_label",
                "match_amount",
                "match_partner_ids",
                "trigger",
            ],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["match_label", "match_label_param"],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "match_label_param must be a valid Python regex when match_label='match_regex'",
        source_method: Some("_check_match_label_param"),
    }],
    provenance: OdooProvenance {
        l_doc: "L5-PAY-TERMS-MATCH.md",
        l_doc_lines: (399, 563),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_reconcile_model.py",
            line_range: (91, 201),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── account.reconcile.model.line ────────────────────────────────────────────
//
// NOTE: L2 overlap — same as parent model above.

pub const RECONCILE_MODEL_LINE: OdooEntity = OdooEntity {
    model_name: "account.reconcile.model.line",
    kind: OdooEntityKind::Model,
    description: "Write-off journal line template within a reconcile model; \
                  amount sourced via fixed/percentage/regex from statement label.",
    fields: &[
        OdooField {
            name: "model_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.reconcile.model"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            // Contra-account for write-off/categorisation. Absent = partner-mapping rule.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "partner_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "amount_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 'fixed' | 'percentage' | 'percentage_st_line' | 'regex'
            // For regex: amount_string IS the regex; amount float is always 0.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "amount_string",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Authoritative field: numeric string or regex pattern.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "amount",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_float_amount"),
            depends: &["amount_string"],
            // Cached float of amount_string via float(). 0 on ValueError or regex type.
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "tax_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("account.tax"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "label",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Document,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_float_amount",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_validate_amount",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["amount_string"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["amount_string"],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "fixed/percentage amounts must be non-zero; regex must be valid Python re",
        source_method: Some("_validate_amount"),
    }],
    provenance: OdooProvenance {
        l_doc: "L5-PAY-TERMS-MATCH.md",
        l_doc_lines: (456, 468),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_reconcile_model.py",
            line_range: (8, 89),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// Entities documented in lane L5 (payment terms + reconcile-model match
/// + payment-to-invoice matching / Mahnwesen gate).
///
/// 5 entities total:
///   [0] `account.payment`              — K3 + K5, state machine, Mahnwesen gate
///   [1] `account.payment.term`         — K3, Skonto computation, due-date splits
///   [2] `account.payment.term.line`    — K3 sub-model, due-date formula
///   [3] `account.reconcile.model`      — K3 + K5, NARS-heavy bank matching (L2 overlap, see above)
///   [4] `account.reconcile.model.line` — K3 + K5 sub-model, write-off template (L2 overlap)
///
/// Entities NOT included (skipped):
///   - `account.payment.method`      — referenced indirectly via payment_method_line_id;
///                                     not documented as a primary entity in L5.
///   - `account.payment.method.line` — same; belongs to journal / payment-method lane.
///   - `account.payment.register`    — wizard, not documented in L5 prose.
pub const ENTITIES: &[OdooEntity] = &[
    PAYMENT,
    PAYMENT_TERM,
    PAYMENT_TERM_LINE,
    RECONCILE_MODEL,
    RECONCILE_MODEL_LINE,
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooFieldKind, OdooSemanticRole};

    #[test]
    fn entities_slice_has_expected_count() {
        assert_eq!(ENTITIES.len(), 5);
    }

    #[test]
    fn payment_state_machine_is_populated() {
        let sm = PAYMENT.state_machine.expect("account.payment must have a state machine");
        assert_eq!(sm.state_field, "state");
        assert_eq!(sm.states.len(), 5);
        assert!(!sm.transitions.is_empty());
    }

    #[test]
    fn payment_has_both_reconciliation_flags() {
        let is_reconciled = PAYMENT.fields.iter().find(|f| f.name == "is_reconciled");
        let is_matched = PAYMENT.fields.iter().find(|f| f.name == "is_matched");
        assert!(is_reconciled.is_some(), "is_reconciled field must be present");
        assert!(is_matched.is_some(), "is_matched field must be present");
    }

    #[test]
    fn payment_term_skonto_fields_present() {
        let field_names: Vec<&str> = PAYMENT_TERM.fields.iter().map(|f| f.name).collect();
        assert!(field_names.contains(&"early_discount"));
        assert!(field_names.contains(&"discount_percentage"));
        assert!(field_names.contains(&"discount_days"));
        assert!(field_names.contains(&"early_pay_discount_computation"));
    }

    #[test]
    fn payment_term_constraint_present() {
        assert_eq!(PAYMENT_TERM.constraints.len(), 1);
        assert_eq!(
            PAYMENT_TERM.constraints[0].source_method,
            Some("_check_lines")
        );
    }

    #[test]
    fn reconcile_model_can_be_proposed_is_computed() {
        let f = RECONCILE_MODEL
            .fields
            .iter()
            .find(|f| f.name == "can_be_proposed")
            .expect("can_be_proposed must be present");
        assert_eq!(f.computed, Some("_compute_can_be_proposed"));
        assert_eq!(f.kind, OdooFieldKind::Boolean);
    }

    #[test]
    fn reconcile_model_line_amount_type_policy() {
        let f = RECONCILE_MODEL_LINE
            .fields
            .iter()
            .find(|f| f.name == "amount_type")
            .expect("amount_type must be present");
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
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
    fn all_entities_cite_l5_doc() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.l_doc,
                "L5-PAY-TERMS-MATCH.md",
                "entity {} must cite L5 doc",
                entity.model_name
            );
        }
    }
}
