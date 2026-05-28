//! Lane L3 (K7-TAX) — typed Odoo entity declarations for tax computation,
//! exigibility (on-invoice vs on-payment / cash-basis), fiscal positions,
//! and tax repartition.
//!
//! Source: `.claude/odoo/L3-K7-TAX.md` (1120 lines, 2026-05-26).
//! Savant served: `TaxExigibilitySuggestor`.
//!
//! Entities:
//!   1. `account.tax.group`
//!   2. `account.tax`
//!   3. `account.tax.repartition.line`
//!   4. `account.fiscal.position`
//!   5. `account.fiscal.position.account`
//!   6. `account.account.tag`

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── 1. account.tax.group ────────────────────────────────────────────────────

const TAX_GROUP_FIELDS: &[OdooField] = &[
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
        name: "tax_payable_account_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.account"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference, // USt-Verbindlichkeiten (e.g. 1776/3860)
    },
    OdooField {
        name: "tax_receivable_account_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.account"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference, // VSt-Forderungen (e.g. 1545/1421)
    },
    OdooField {
        name: "advance_tax_payment_account_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.account"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference, // USt-Vorauszahlung (e.g. 1780/3820)
    },
    OdooField {
        name: "country_id",
        kind: OdooFieldKind::Computed,
        target: Some("res.country"),
        required: false,
        computed: Some("_compute_country_id"),
        depends: &["company_id"],
        semantic_role: OdooSemanticRole::Reference,
    },
    OdooField {
        name: "preceding_subtotal",
        kind: OdooFieldKind::Char,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Other, // UI label before subtotal group
    },
];

const TAX_GROUP_METHODS: &[OdooMethod] = &[OdooMethod {
    name: "_compute_country_id",
    kind: OdooMethodKind::Compute,
    return_kind: OdooReturnKind::Unit,
    triggers: &[],
}];

const TAX_GROUP_DECORATORS: &[OdooDecorator] = &[OdooDecorator {
    kind: OdooDecoratorKind::ApiDepends,
    targets: &["company_id"],
}];

const ACCOUNT_TAX_GROUP: OdooEntity = OdooEntity {
    model_name: "account.tax.group",
    description: "Groups taxes for display + closing-entry accounts (USt/VSt/advance); \
                  l10n_de groups: 0%, 7%, 5.5%, 10.7%, 19%.",
    fields: TAX_GROUP_FIELDS,
    methods: TAX_GROUP_METHODS,
    decorators: TAX_GROUP_DECORATORS,
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L3-K7-TAX.md",
        l_doc_lines: (624, 658),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_tax.py",
            line_range: (25, 69),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 2. account.tax ──────────────────────────────────────────────────────────

const ACCOUNT_TAX_FIELDS: &[OdooField] = &[
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
        name: "type_tax_use",
        kind: OdooFieldKind::Selection,
        target: None,
        required: true,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy, // sale | purchase | none | adjustment
    },
    OdooField {
        name: "tax_scope",
        kind: OdooFieldKind::Selection,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy, // service | consu
    },
    OdooField {
        name: "amount_type",
        kind: OdooFieldKind::Selection,
        target: None,
        required: true,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Tax, // group | fixed | percent | division
    },
    OdooField {
        name: "amount",
        kind: OdooFieldKind::Float,
        target: None,
        required: true,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Tax, // e.g. 19.0 for 19%, or fixed EUR amount
    },
    OdooField {
        name: "tax_exigibility",
        kind: OdooFieldKind::Selection,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        // KEY concept for TaxExigibilitySuggestor:
        // 'on_invoice' (default) = accrual basis — tax due on invoice validation
        // 'on_payment' = cash basis — tax due on payment receipt; routes amounts
        //                through cash_basis_transition_account_id until reconciled
        semantic_role: OdooSemanticRole::Tax,
    },
    OdooField {
        name: "cash_basis_transition_account_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.account"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Tax, // CABA transition account (must allow reconciliation)
    },
    OdooField {
        name: "price_include",
        kind: OdooFieldKind::Computed,
        target: None,
        required: false,
        computed: Some("_compute_price_include"),
        depends: &["price_include_override"],
        // Derived from price_include_override (per-tax) OR company_price_include (company default).
        // NOT stored; a porter must replicate the two-level override logic.
        semantic_role: OdooSemanticRole::Tax,
    },
    OdooField {
        name: "price_include_override",
        kind: OdooFieldKind::Selection,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy, // tax_included | tax_excluded | (empty → company default)
    },
    OdooField {
        name: "include_base_amount",
        kind: OdooFieldKind::Boolean,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Tax, // if true: this tax's amount affects subsequent taxes' bases
    },
    OdooField {
        name: "is_base_affected",
        kind: OdooFieldKind::Boolean,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Tax, // if true: accepts base-amount influence from preceding taxes
    },
    OdooField {
        name: "sequence",
        kind: OdooFieldKind::Integer,
        target: None,
        required: true,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy, // processing order (lower = earlier)
    },
    OdooField {
        name: "tax_group_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.tax.group"),
        required: true,
        computed: Some("_compute_tax_group_id"),
        depends: &["company_id", "country_id"],
        semantic_role: OdooSemanticRole::Reference,
    },
    OdooField {
        name: "invoice_repartition_line_ids",
        kind: OdooFieldKind::One2many,
        target: Some("account.tax.repartition.line"),
        required: false,
        computed: Some("_compute_invoice_repartition_line_ids"),
        depends: &[],
        semantic_role: OdooSemanticRole::Tax,
    },
    OdooField {
        name: "refund_repartition_line_ids",
        kind: OdooFieldKind::One2many,
        target: Some("account.tax.repartition.line"),
        required: false,
        computed: Some("_compute_refund_repartition_line_ids"),
        depends: &[],
        semantic_role: OdooSemanticRole::Tax,
    },
    OdooField {
        name: "children_tax_ids",
        kind: OdooFieldKind::Many2many,
        target: Some("account.tax"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Tax, // populated only when amount_type == 'group'
    },
    OdooField {
        name: "original_tax_ids",
        kind: OdooFieldKind::Many2many,
        target: Some("account.tax"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference, // domestic taxes this one replaces in fiscal-position mapping
    },
    OdooField {
        name: "has_negative_factor",
        kind: OdooFieldKind::Computed,
        target: None,
        required: false,
        computed: Some("_compute_has_negative_factor"),
        depends: &[],
        semantic_role: OdooSemanticRole::Tax, // §13b reverse-charge marker: any repartition line with factor < 0
    },
    OdooField {
        name: "country_id",
        kind: OdooFieldKind::Computed,
        target: Some("res.country"),
        required: true,
        computed: Some("_compute_country_id"),
        depends: &["company_id"],
        semantic_role: OdooSemanticRole::Reference,
    },
    OdooField {
        name: "fiscal_position_ids",
        kind: OdooFieldKind::Many2many,
        target: Some("account.fiscal.position"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference,
    },
    OdooField {
        name: "analytic",
        kind: OdooFieldKind::Boolean,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy,
    },
];

const ACCOUNT_TAX_METHODS: &[OdooMethod] = &[
    OdooMethod {
        name: "_flatten_taxes_and_sort_them",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Dict,
        // R1: expands group taxes into children, sorts by (sequence, id).
        // Group taxes are NEVER in sorted_taxes — only children.
        triggers: &[],
    },
    OdooMethod {
        name: "_batch_for_taxes_computation",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Dict,
        // R2: batches consecutive same-type/price_include/include_base_amount taxes.
        // Batch determines the shared denominator for price-included percent taxes.
        triggers: &[],
    },
    OdooMethod {
        name: "_get_tax_details",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Dict,
        // R3: three-pass evaluation: Fixed (reverse) → price-included (reverse) → price-excluded (forward).
        // Returns: { total_excluded, total_included, taxes_data }.
        triggers: &[],
    },
    OdooMethod {
        name: "_propagate_extra_taxes_base",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Unit,
        // R4: updates extra_base_for_tax / extra_base_for_base after each tax amount is computed.
        // extra_base_for_tax: only if target not yet computed. extra_base_for_base: always.
        triggers: &[],
    },
    OdooMethod {
        name: "_add_tax_details_in_base_line",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Unit,
        // R5: outer driver; applies discount, FX rate, rounding_method from company.
        // Produces raw_total_excluded/included_currency + raw_total_excluded/included (company CCY).
        triggers: &[],
    },
    OdooMethod {
        name: "_add_accounting_data_to_base_line_tax_details",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Unit,
        // R7: bridge from computed amounts to journal entries.
        // Selects invoice_repartition_line_ids vs refund_repartition_line_ids by is_refund.
        // on_payment taxes route to cash_basis_transition_account_id until reconciled.
        triggers: &[],
    },
    OdooMethod {
        name: "compute_all",
        kind: OdooMethodKind::Override,
        return_kind: OdooReturnKind::Dict,
        // R8: legacy public API. Returns base_tags, taxes (per repartition line), total_excluded,
        // total_included, total_void. Uses raw amounts for repartition (context key
        // 'compute_all_use_raw_base_lines'=True → no double-rounding).
        triggers: &[],
    },
    OdooMethod {
        name: "_compute_price_include",
        kind: OdooMethodKind::Compute,
        return_kind: OdooReturnKind::Unit,
        triggers: &[],
    },
    OdooMethod {
        name: "_compute_has_negative_factor",
        kind: OdooMethodKind::Compute,
        return_kind: OdooReturnKind::Unit,
        triggers: &[],
    },
    OdooMethod {
        name: "flatten_taxes_hierarchy",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Recordset,
        // Thin alias for _flatten_taxes_and_sort_them()[0]. See account_tax.py:L4855-4856.
        triggers: &[],
    },
    OdooMethod {
        name: "_adapt_price_unit_to_another_taxes",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Number,
        // R14: when a fiscal position maps a price-included tax to another tax, adjusts
        // price_unit so the net base remains consistent. Only activates when ALL original
        // taxes are price-included.
        triggers: &[],
    },
];

const ACCOUNT_TAX_DECORATORS: &[OdooDecorator] = &[
    OdooDecorator {
        kind: OdooDecoratorKind::ApiConstrains,
        targets: &["company_id", "name", "type_tax_use", "tax_scope", "country_id"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiConstrains,
        targets: &["tax_group_id"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiConstrains,
        targets: &["tax_exigibility", "cash_basis_transition_account_id"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["price_include_override"],
    },
];

const ACCOUNT_TAX_CONSTRAINTS: &[OdooConstraint] = &[
    OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "Tax names must be unique within (company hierarchy, name, type_tax_use, \
                    tax_scope, country_id). Checked in batches of 100.",
        source_method: Some("_constrains_name"),
    },
    OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "tax_group_id.country_id must match tax.country_id.",
        source_method: Some("validate_tax_group_id"),
    },
    OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "If tax_exigibility == 'on_payment', cash_basis_transition_account_id must \
                    allow reconciliation.",
        source_method: Some("_constrains_cash_basis_transition_account"),
    },
    OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "Nested group taxes are not allowed (children_tax_ids of a group tax may not \
                    themselves be group taxes).",
        source_method: Some("_constrains_children_tax_ids"),
    },
];

const ACCOUNT_TAX: OdooEntity = OdooEntity {
    model_name: "account.tax",
    description: "VAT / USt tax definition with computation type (percent/fixed/division/group), \
                  exigibility (accrual vs cash-basis), price-include semantics, and repartition \
                  lines. Core of K7 computation engine.",
    fields: ACCOUNT_TAX_FIELDS,
    methods: ACCOUNT_TAX_METHODS,
    decorators: ACCOUNT_TAX_DECORATORS,
    state_machine: None,
    constraints: ACCOUNT_TAX_CONSTRAINTS,
    provenance: OdooProvenance {
        l_doc: "L3-K7-TAX.md",
        l_doc_lines: (805, 850),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_tax.py",
            line_range: (71, 320),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 3. account.tax.repartition.line ─────────────────────────────────────────

const REPARTITION_LINE_FIELDS: &[OdooField] = &[
    OdooField {
        name: "factor_percent",
        kind: OdooFieldKind::Float,
        target: None,
        required: true,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Tax, // e.g. 100.0, -100.0 (negative = reverse charge)
    },
    OdooField {
        name: "factor",
        kind: OdooFieldKind::Computed,
        target: None,
        required: false,
        computed: Some("_compute_factor"),
        depends: &["factor_percent"],
        semantic_role: OdooSemanticRole::Tax, // factor_percent / 100.0
    },
    OdooField {
        name: "repartition_type",
        kind: OdooFieldKind::Selection,
        target: None,
        required: true,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Tax, // 'base' | 'tax'
    },
    OdooField {
        name: "document_type",
        kind: OdooFieldKind::Selection,
        target: None,
        required: true,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Tax, // 'invoice' | 'refund'
    },
    OdooField {
        name: "account_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.account"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference, // if empty → amount goes to total_void
    },
    OdooField {
        name: "tag_ids",
        kind: OdooFieldKind::Many2many,
        target: Some("account.account.tag"),
        required: false,
        computed: None,
        depends: &[],
        // USt-VA Kennziffer tags (e.g. 81_BASE, 81_TAX, 89_BASE, 66).
        // These are the K8 bridge: aggregate move lines by tag to derive Kz totals.
        semantic_role: OdooSemanticRole::Tax,
    },
    OdooField {
        name: "tax_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.tax"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference,
    },
    OdooField {
        name: "sequence",
        kind: OdooFieldKind::Integer,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy,
    },
    OdooField {
        name: "use_in_tax_closing",
        kind: OdooFieldKind::Computed,
        target: None,
        required: false,
        computed: Some("_compute_use_in_tax_closing"),
        depends: &["account_id", "repartition_type"],
        // True when: repartition_type=='tax' AND account_id AND
        // account_id.internal_group NOT IN ('income', 'expense')
        semantic_role: OdooSemanticRole::Tax,
    },
];

const REPARTITION_LINE_METHODS: &[OdooMethod] = &[
    OdooMethod {
        name: "_get_aml_target_tax_account",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Record,
        // Returns cash_basis_transition_account_id when tax_exigibility=='on_payment'
        // and context['caba_no_transition_account'] is not set; else returns account_id.
        triggers: &[],
    },
    OdooMethod {
        name: "_compute_factor",
        kind: OdooMethodKind::Compute,
        return_kind: OdooReturnKind::Unit,
        triggers: &[],
    },
    OdooMethod {
        name: "_compute_use_in_tax_closing",
        kind: OdooMethodKind::Compute,
        return_kind: OdooReturnKind::Unit,
        triggers: &[],
    },
];

const REPARTITION_LINE_DECORATORS: &[OdooDecorator] = &[
    OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["factor_percent"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["account_id", "repartition_type"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiOnchange,
        targets: &["repartition_type"],
    },
];

const REPARTITION_LINE_CONSTRAINTS: &[OdooConstraint] = &[OdooConstraint {
    kind: OdooConstraintKind::Python,
    condition: "Each document type must have exactly one base line and at least one tax line. \
                Invoice and refund must have the same number of lines in the same percentage order. \
                Sum of positive factors == 1.0; if negative factors exist: sum == -1.0.",
    source_method: Some("_validate_repartition_lines"),
}];

const ACCOUNT_TAX_REPARTITION_LINE: OdooEntity = OdooEntity {
    model_name: "account.tax.repartition.line",
    description: "Distribution rule mapping a tax computation result to a GL account and \
                  USt-VA grid tag. Negative factor lines implement §13b reverse-charge split. \
                  K8 bridge: tag_ids carry Kennziffer for Voranmeldung aggregation.",
    fields: REPARTITION_LINE_FIELDS,
    methods: REPARTITION_LINE_METHODS,
    decorators: REPARTITION_LINE_DECORATORS,
    state_machine: None,
    constraints: REPARTITION_LINE_CONSTRAINTS,
    provenance: OdooProvenance {
        l_doc: "L3-K7-TAX.md",
        l_doc_lines: (573, 620),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_tax.py",
            line_range: (5141, 5210),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 4. account.fiscal.position ──────────────────────────────────────────────

const FISCAL_POSITION_FIELDS: &[OdooField] = &[
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
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy, // auto-apply match order (lower = higher priority)
    },
    OdooField {
        name: "auto_apply",
        kind: OdooFieldKind::Boolean,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy, // l10n_de: only Domestic + EU-with-VAT-ID auto-apply
    },
    OdooField {
        name: "vat_required",
        kind: OdooFieldKind::Boolean,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy,
    },
    OdooField {
        name: "country_id",
        kind: OdooFieldKind::Many2one,
        target: Some("res.country"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference, // apply if delivery country matches
    },
    OdooField {
        name: "country_group_id",
        kind: OdooFieldKind::Many2one,
        target: Some("res.country.group"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference, // e.g. EU group for intra-community
    },
    OdooField {
        name: "state_ids",
        kind: OdooFieldKind::Many2many,
        target: Some("res.country.state"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference,
    },
    OdooField {
        name: "zip_from",
        kind: OdooFieldKind::Char,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        // ZIP range comparison is STRING-based (alphabetic, not numeric).
        // _convert_zip_values right-pads numeric ZIPs with leading zeros.
        semantic_role: OdooSemanticRole::Policy,
    },
    OdooField {
        name: "zip_to",
        kind: OdooFieldKind::Char,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy,
    },
    OdooField {
        name: "tax_ids",
        kind: OdooFieldKind::Many2many,
        target: Some("account.tax"),
        required: false,
        computed: None,
        depends: &[],
        // Destination taxes. If empty AND any input tax has fiscal_position_ids set → ALL taxes removed.
        semantic_role: OdooSemanticRole::Tax,
    },
    OdooField {
        name: "tax_map",
        kind: OdooFieldKind::Computed,
        target: None,
        required: false,
        computed: Some("_compute_tax_map"),
        depends: &["tax_ids"],
        // Binary: Dict[src_tax_id, List[dest_tax_id]] built by inverting original_tax_ids.
        // Many-to-many: one src can map to multiple destinations.
        semantic_role: OdooSemanticRole::Tax,
    },
    OdooField {
        name: "account_ids",
        kind: OdooFieldKind::One2many,
        target: Some("account.fiscal.position.account"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference,
    },
    OdooField {
        name: "account_map",
        kind: OdooFieldKind::Computed,
        target: None,
        required: false,
        computed: Some("_compute_account_map"),
        depends: &["account_ids.account_src_id", "account_ids.account_dest_id"],
        // Binary: Dict[src_account_id, dest_account_id]. Identity if no match.
        semantic_role: OdooSemanticRole::Reference,
    },
    OdooField {
        name: "is_domestic",
        kind: OdooFieldKind::Computed,
        target: None,
        required: false,
        computed: Some("_compute_is_domestic"),
        depends: &["company_id.domestic_fiscal_position_id"],
        semantic_role: OdooSemanticRole::Status,
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
        name: "foreign_vat",
        kind: OdooFieldKind::Char,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy, // company's VAT in a foreign jurisdiction
    },
];

const FISCAL_POSITION_METHODS: &[OdooMethod] = &[
    OdooMethod {
        name: "map_tax",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Recordset,
        // R11: translates a recordset of taxes through the fiscal position's tax_map.
        // Identity if self is empty (no FP). ALL taxes removed if tax_ids is empty
        // and any input tax has fiscal_position_ids set (tax-unit edge case).
        triggers: &[],
    },
    OdooMethod {
        name: "map_account",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Record,
        // R11: translates a single account through account_map. Identity if no match.
        triggers: &[],
    },
    OdooMethod {
        name: "_get_fiscal_position",
        kind: OdooMethodKind::ApiModel,
        return_kind: OdooReturnKind::Record,
        // R12: auto-apply logic. Priority: manual override > country/group/state/zip/VAT checks.
        // Sorted: company-depth first (deeper child wins), then sequence ascending.
        // Intra-EU + vat_exclusion → falls back to invoicing address instead of delivery.
        triggers: &[],
    },
    OdooMethod {
        name: "_get_first_matching_fpos",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Record,
        triggers: &[],
    },
    OdooMethod {
        name: "_compute_tax_map",
        kind: OdooMethodKind::Compute,
        return_kind: OdooReturnKind::Unit,
        triggers: &[],
    },
    OdooMethod {
        name: "_compute_account_map",
        kind: OdooMethodKind::Compute,
        return_kind: OdooReturnKind::Unit,
        triggers: &[],
    },
];

const FISCAL_POSITION_DECORATORS: &[OdooDecorator] = &[
    OdooDecorator {
        kind: OdooDecoratorKind::ApiModel,
        targets: &[],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiModelCreateMulti,
        targets: &[],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["tax_ids"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["account_ids.account_src_id", "account_ids.account_dest_id"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiConstrains,
        targets: &["zip_from", "zip_to"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiOnchange,
        targets: &["country_id"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiOnchange,
        targets: &["country_group_id"],
    },
];

const FISCAL_POSITION_CONSTRAINTS: &[OdooConstraint] = &[
    OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "zip_from and zip_to must be both set or both empty; zip_to >= zip_from \
                    (string comparison after zero-padding).",
        source_method: Some("_check_zip"),
    },
    OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "foreign_vat requires country_id. If country_id == company's fiscal country, \
                    state_ids must be set when that country has states. country_id must be within \
                    country_group_id if both are set.",
        source_method: Some("_validate_foreign_vat_country"),
    },
];

const ACCOUNT_FISCAL_POSITION: OdooEntity = OdooEntity {
    model_name: "account.fiscal.position",
    description: "Tax regime mapping rule: translates taxes and GL accounts for a partner. \
                  Auto-apply by country/group/state/ZIP/VAT; manual override wins. \
                  l10n_de: 2 auto-apply (Domestic + EU-with-VAT-ID), 4 manual-only.",
    fields: FISCAL_POSITION_FIELDS,
    methods: FISCAL_POSITION_METHODS,
    decorators: FISCAL_POSITION_DECORATORS,
    state_machine: None,
    constraints: FISCAL_POSITION_CONSTRAINTS,
    provenance: OdooProvenance {
        l_doc: "L3-K7-TAX.md",
        l_doc_lines: (661, 800),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/partner.py",
            line_range: (26, 301),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 5. account.fiscal.position.account ──────────────────────────────────────

const FISCAL_POSITION_ACCOUNT_FIELDS: &[OdooField] = &[
    OdooField {
        name: "position_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.fiscal.position"),
        required: true,
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
        name: "account_src_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.account"),
        required: true,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference, // source GL account (e.g. 8400)
    },
    OdooField {
        name: "account_dest_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.account"),
        required: true,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference, // destination GL account (e.g. 8125)
    },
];

const FISCAL_POSITION_ACCOUNT: OdooEntity = OdooEntity {
    model_name: "account.fiscal.position.account",
    description: "One GL account remapping rule within a fiscal position. \
                  Used to redirect revenue/expense accounts (e.g. 8400→8125 for EU-with-VAT-ID).",
    fields: FISCAL_POSITION_ACCOUNT_FIELDS,
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Sql,
        condition: "unique(position_id, account_src_id, account_dest_id) — no duplicate \
                    account mapping rows within the same fiscal position.",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L3-K7-TAX.md",
        l_doc_lines: (1026, 1037),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/partner.py",
            line_range: (303, 324),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 6. account.account.tag ───────────────────────────────────────────────────

const ACCOUNT_TAG_FIELDS: &[OdooField] = &[
    OdooField {
        name: "name",
        kind: OdooFieldKind::Char,
        target: None,
        required: true,
        computed: None,
        depends: &[],
        // For tax applicability: encodes USt-VA Kennziffer (e.g. "+81", "-81_TAX", "89_BASE").
        // The _BASE / _TAX suffix and +/- sign derive VAT-return line semantics.
        semantic_role: OdooSemanticRole::Identity,
    },
    OdooField {
        name: "applicability",
        kind: OdooFieldKind::Selection,
        target: None,
        required: true,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Policy, // 'accounts' | 'taxes' | 'products'
    },
    OdooField {
        name: "country_id",
        kind: OdooFieldKind::Many2one,
        target: Some("res.country"),
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Reference,
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
];

const ACCOUNT_ACCOUNT_TAG: OdooEntity = OdooEntity {
    model_name: "account.account.tag",
    description: "USt-VA grid tag linking move lines to Kennziffer (Kz) in the \
                  Umsatzsteuer-Voranmeldung. For tax applicability, name encodes the \
                  Kz (e.g. '81_BASE', '81_TAX', '89_BASE'). K8 aggregation: sum \
                  move lines by tag to compute each Kz total.",
    fields: ACCOUNT_TAG_FIELDS,
    methods: &[],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["applicability", "country_id"],
    }],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Sql,
        condition: "unique(name, applicability, country_id) — no duplicate tag name within the \
                    same applicability scope and country.",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L3-K7-TAX.md",
        l_doc_lines: (940, 960),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_account_tag.py",
            line_range: (7, 65),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// All entities documented in lane L3 (K7-TAX).
///
/// Ordered by dependency: tax groups → taxes → repartition lines →
/// fiscal positions → account mappings → tags.
pub const ENTITIES: &[OdooEntity] = &[
    ACCOUNT_TAX_GROUP,
    ACCOUNT_TAX,
    ACCOUNT_TAX_REPARTITION_LINE,
    ACCOUNT_FISCAL_POSITION,
    FISCAL_POSITION_ACCOUNT,
    ACCOUNT_ACCOUNT_TAG,
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooFieldKind, OdooSemanticRole};

    #[test]
    fn entity_count() {
        assert_eq!(ENTITIES.len(), 6);
    }

    #[test]
    fn model_names_correct() {
        let names: Vec<&str> = ENTITIES.iter().map(|e| e.model_name).collect();
        assert!(names.contains(&"account.tax"));
        assert!(names.contains(&"account.tax.group"));
        assert!(names.contains(&"account.tax.repartition.line"));
        assert!(names.contains(&"account.fiscal.position"));
        assert!(names.contains(&"account.fiscal.position.account"));
        assert!(names.contains(&"account.account.tag"));
    }

    #[test]
    fn all_entities_curated() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.confidence,
                OdooConfidence::Curated,
                "{} has wrong confidence",
                entity.model_name
            );
        }
    }

    #[test]
    fn all_entities_reference_l3_doc() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.l_doc, "L3-K7-TAX.md",
                "{} references wrong L-doc",
                entity.model_name
            );
            let (start, end) = entity.provenance.l_doc_lines;
            assert!(start > 0, "{} has zero start line", entity.model_name);
            assert!(end >= start, "{} has end < start", entity.model_name);
            assert!(end <= 1120, "{} references line beyond doc length", entity.model_name);
        }
    }

    #[test]
    fn tax_exigibility_field_is_selection_with_tax_role() {
        let tax_entity = ENTITIES.iter().find(|e| e.model_name == "account.tax").unwrap();
        let exig = tax_entity
            .fields
            .iter()
            .find(|f| f.name == "tax_exigibility")
            .expect("tax_exigibility field must exist on account.tax");
        assert_eq!(exig.kind, OdooFieldKind::Selection);
        assert_eq!(exig.semantic_role, OdooSemanticRole::Tax);
    }

    #[test]
    fn repartition_tag_ids_field_present() {
        let rep = ENTITIES
            .iter()
            .find(|e| e.model_name == "account.tax.repartition.line")
            .unwrap();
        let tag_field = rep.fields.iter().find(|f| f.name == "tag_ids").unwrap();
        assert_eq!(tag_field.kind, OdooFieldKind::Many2many);
        assert_eq!(tag_field.target, Some("account.account.tag"));
    }

    #[test]
    fn fiscal_position_has_auto_apply_policy_field() {
        let fp = ENTITIES
            .iter()
            .find(|e| e.model_name == "account.fiscal.position")
            .unwrap();
        let auto_apply = fp.fields.iter().find(|f| f.name == "auto_apply").unwrap();
        assert_eq!(auto_apply.kind, OdooFieldKind::Boolean);
        assert_eq!(auto_apply.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn account_tax_group_has_payable_receivable_accounts() {
        let tg = ENTITIES
            .iter()
            .find(|e| e.model_name == "account.tax.group")
            .unwrap();
        assert!(tg.fields.iter().any(|f| f.name == "tax_payable_account_id"));
        assert!(tg.fields.iter().any(|f| f.name == "tax_receivable_account_id"));
        assert!(tg.fields.iter().any(|f| f.name == "advance_tax_payment_account_id"));
    }

    #[test]
    fn fiscal_position_account_has_src_dest() {
        let fpa = ENTITIES
            .iter()
            .find(|e| e.model_name == "account.fiscal.position.account")
            .unwrap();
        assert!(fpa.fields.iter().any(|f| f.name == "account_src_id"));
        assert!(fpa.fields.iter().any(|f| f.name == "account_dest_id"));
        // No state machine on this model
        assert!(fpa.state_machine.is_none());
    }
}
