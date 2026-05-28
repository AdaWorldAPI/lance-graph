//! Lane L9 (PARTNER-FISCALPOS) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L9-PARTNER-FISCALPOS.md`.
//!
//! **Entity inventory (6 entities)**
//!
//! | Const                          | Odoo model                         | L-doc rules |
//! |---|---|---|
//! | [`ACCOUNT_FISCAL_POSITION`]    | `account.fiscal.position`          | R8–R12, R20 |
//! | [`ACCOUNT_FISCAL_POS_ACCOUNT`] | `account.fiscal.position.account`  | R10         |
//! | [`RES_PARTNER_ACCOUNTING`]     | `res.partner` (account extension)  | R1–R7, R14–R19 |
//! | [`RES_COUNTRY`]                | `res.country`                      | R8 VAT prefix |
//! | [`RES_COUNTRY_GROUP`]          | `res.country.group`                | R8 group match |
//! | [`ACCOUNT_PAYMENT_TERM_REF`]   | `account.payment.term` (reference) | R2 partner side |
//!
//! **L3 overlap:** L3 owns tax repartition internals (`account.tax`, CABA, …).
//! L9 is authoritative for partner-side fiscal-position SELECTION
//! (`_get_fiscal_position`, `map_tax`, `map_account`, `property_*` fields).
//!
//! **Savants:**
//! - `FiscalPositionResolver` (0x80, CustomerCategory, Deduction, NarsTruth, Analytical)
//!   — `_get_fiscal_position` + `_get_first_matching_fpos` (R8, L-doc lines 207–285).
//! - `PartnerTrustAdvisor` (0x80, CustomerCategory, Induction, NarsTruth, Analytical)
//!   — `trust` field drives dunning escalation (R16, L-doc line 456).

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── 1. account.fiscal.position ──────────────────────────────────────────────

/// `account.fiscal.position` — partner-facing fiscal-position record.
///
/// L9-authoritative for SELECTION; L3-authoritative for repartition internals.
/// L-doc R8–R12, R20; Odoo source L26–L301.
pub const ACCOUNT_FISCAL_POSITION: OdooEntity = OdooEntity {
    model_name: "account.fiscal.position",
    description: "Named tax-and-account mapping applied to a partner; selected by \
                  _get_fiscal_position (FiscalPositionResolver savant) or manual override.",
    fields: &[
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        // Sort key: company-specificity first (-len(parent_ids)), then sequence ASC.
        OdooField { name: "sequence", kind: OdooFieldKind::Integer, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "active", kind: OdooFieldKind::Boolean, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
        OdooField { name: "company_id", kind: OdooFieldKind::Many2one, target: Some("res.company"),
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // auto_apply=True → candidate for _get_fiscal_position auto-detection (R8).
        OdooField { name: "auto_apply", kind: OdooFieldKind::Boolean, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        // Predicate 1 in _get_fpos_validation_functions: partner must have VAT.
        OdooField { name: "vat_required", kind: OdooFieldKind::Boolean, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        // Predicate 4: partner.country_id == this (absent → any country).
        OdooField { name: "country_id", kind: OdooFieldKind::Many2one, target: Some("res.country"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // Predicate 5: partner.country_id in group.country_ids AND state not excluded.
        OdooField { name: "country_group_id", kind: OdooFieldKind::Many2one,
            target: Some("res.country.group"), required: false, computed: None, depends: &[],
            semantic_role: OdooSemanticRole::Reference },
        // Predicate 3: partner.state_id in this set (empty → any state).
        OdooField { name: "state_ids", kind: OdooFieldKind::Many2many,
            target: Some("res.country.state"), required: false, computed: None, depends: &[],
            semantic_role: OdooSemanticRole::Reference },
        // Predicate 2: padded with leading zeros by _convert_zip_values (R11).
        OdooField { name: "zip_from", kind: OdooFieldKind::Char, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "zip_to", kind: OdooFieldKind::Char, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "foreign_vat", kind: OdooFieldKind::Char, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        // True if this fpos == company.domestic_fiscal_position_id (R12).
        OdooField { name: "is_domestic", kind: OdooFieldKind::Boolean, target: None, required: false,
            computed: Some("_compute_is_domestic"),
            depends: &["company_id.domestic_fiscal_position_id"],
            semantic_role: OdooSemanticRole::Status },
        // {src_tax_id: [dest_tax_id, ...]} — used by map_tax (R9, R20).
        // L3 overlap: content is tax repartition data; the MAP structure is L9-authoritative.
        OdooField { name: "tax_map", kind: OdooFieldKind::Computed, target: None, required: false,
            computed: Some("_compute_tax_map"), depends: &["tax_ids"],
            semantic_role: OdooSemanticRole::Tax },
        // {src_account_id: dest_account_id} — used by map_account (R10, R20).
        OdooField { name: "account_map", kind: OdooFieldKind::Computed, target: None, required: false,
            computed: Some("_compute_account_map"),
            depends: &["account_ids.account_src_id", "account_ids.account_dest_id"],
            semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "account_ids", kind: OdooFieldKind::One2many,
            target: Some("account.fiscal.position.account"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "note", kind: OdooFieldKind::Html, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Document },
    ],
    methods: &[
        // R8 / FiscalPositionResolver savant boundary ────────────────────
        // @api.model: precedence → (1) manual property wins; (2) no country → empty;
        // (3) auto-detect via _get_first_matching_fpos (AXIS-B).
        OdooMethod { name: "_get_fiscal_position", kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Record, triggers: &[] },
        // Company-specific first, then sequence ASC; runs all 5 predicates.
        OdooMethod { name: "_get_first_matching_fpos", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Record, triggers: &[] },
        // Returns 5 lambdas: vat_required, zip_range, state, country, country_group.
        OdooMethod { name: "_get_fpos_validation_functions", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict, triggers: &[] },
        // R9: taxes → mapped taxes; special: empty fpos + fpos-aware taxes → remove all.
        OdooMethod { name: "map_tax", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset, triggers: &[] },
        // R10: account_id → mapped account via account_map; identity if no mapping.
        OdooMethod { name: "map_account", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Record, triggers: &[] },
        // R11: pads digit-only zips to equal length for correct lexicographic range match.
        OdooMethod { name: "_convert_zip_values", kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "create", kind: OdooMethodKind::ApiModelCreateMulti,
            return_kind: OdooReturnKind::Recordset, triggers: &[] },
        OdooMethod { name: "write", kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean, triggers: &[] },
        OdooMethod { name: "_compute_is_domestic", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_tax_map", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_account_map", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["company_id.domestic_fiscal_position_id"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends, targets: &["tax_ids"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends,
            targets: &["account_ids.account_src_id", "account_ids.account_dest_id"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiConstrains, targets: &["zip_from", "zip_to"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiConstrains,
            targets: &["country_id", "country_group_id", "state_ids", "foreign_vat"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiModel, targets: &[] },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "zip_from and zip_to both set or both absent; zip_from <= zip_to \
                        (lexicographic after leading-zero padding for digit-only zips, R11)",
            source_method: Some("_check_zip"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "foreign_vat requires country_id; if fiscal country, state_ids required \
                        when country has states; no duplicate foreign_vat per country per company",
            source_method: Some("_validate_foreign_vat_country"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L9-PARTNER-FISCALPOS.md",
        l_doc_lines: (207, 413),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/partner.py",
            line_range: (26, 301),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 2. account.fiscal.position.account ──────────────────────────────────────

/// `account.fiscal.position.account` — one src→dest account substitution line.
///
/// L-doc R10; Odoo source L303–L324.
pub const ACCOUNT_FISCAL_POS_ACCOUNT: OdooEntity = OdooEntity {
    model_name: "account.fiscal.position.account",
    description: "One account-substitution rule inside a fiscal position: \
                  account_src_id → account_dest_id when the position is active (map_account).",
    fields: &[
        OdooField { name: "position_id", kind: OdooFieldKind::Many2one,
            target: Some("account.fiscal.position"), required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "company_id", kind: OdooFieldKind::Many2one, target: Some("res.company"),
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // Lookup key in account_map at posting time.
        OdooField { name: "account_src_id", kind: OdooFieldKind::Many2one,
            target: Some("account.account"), required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "account_dest_id", kind: OdooFieldKind::Many2one,
            target: Some("account.account"), required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Sql,
        condition: "UNIQUE(position_id, account_src_id, account_dest_id) — no duplicate \
                    src→dest mapping per fiscal position",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L9-PARTNER-FISCALPOS.md",
        l_doc_lines: (348, 373),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/partner.py",
            line_range: (303, 324),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 3. res.partner (account extension) ──────────────────────────────────────
//
// Only fields added / overridden by the `account` module.  Base fields (name,
// vat, country_id, state_id, zip, …) are referenced in depends chains but
// not redeclared.
//
// Skipped (other lane authority): tax_ids (L3), bank_ids (separate lane),
// ref_company_ids (company-structure lane).
//
// Savant inputs:
//   FiscalPositionResolver → property_account_position_id, country_id, vat, zip (R8).
//   PartnerTrustAdvisor    → trust (R16) for dunning escalation.

/// `res.partner` — accounting extension (`_inherit`).
///
/// L-doc R1–R7, R14–R19; Odoo source L326–L870.
pub const RES_PARTNER_ACCOUNTING: OdooEntity = OdooEntity {
    model_name: "res.partner",
    description: "Partner extended by account module: AR/AP property accounts, payment terms, \
                  fiscal-position override, trust/dunning signal, rank counters, credit-limit, EDI.",
    fields: &[
        // R1: company_dependent AR/AP accounts; ondelete=restrict; synced via commercial rollup (R5).
        OdooField { name: "property_account_receivable_id", kind: OdooFieldKind::Property,
            target: Some("account.account"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "property_account_payable_id", kind: OdooFieldKind::Property,
            target: Some("account.account"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // R2: company_dependent payment terms; None = immediate / journal fallback.
        OdooField { name: "property_payment_term_id", kind: OdooFieldKind::Property,
            target: Some("account.payment.term"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "property_supplier_payment_term_id", kind: OdooFieldKind::Property,
            target: Some("account.payment.term"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        // R3: company_dependent manual override; ALWAYS wins over auto-detection in R8.
        OdooField { name: "property_account_position_id", kind: OdooFieldKind::Property,
            target: Some("account.fiscal.position"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Tax },
        // R4: incremented via _increase_rank on invoice/purchase confirm (deferred when rank > 0).
        OdooField { name: "customer_rank", kind: OdooFieldKind::Integer, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "supplier_rank", kind: OdooFieldKind::Integer, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
        // R6: SUM(amount_residual) on posted receivable/payable lines (raw SQL, PG SPLIT_PART).
        OdooField { name: "credit", kind: OdooFieldKind::Monetary, target: None, required: false,
            computed: Some("_credit_debit_get"), depends: &[], semantic_role: OdooSemanticRole::Money },
        OdooField { name: "debit", kind: OdooFieldKind::Monetary, target: None, required: false,
            computed: Some("_credit_debit_get"), depends: &[], semantic_role: OdooSemanticRole::Money },
        // R7: (credit / total_invoiced) * days_since_oldest_invoice; 0 when no invoices.
        OdooField { name: "days_sales_outstanding", kind: OdooFieldKind::Float, target: None,
            required: false, computed: Some("_compute_days_sales_outstanding"),
            depends: &["credit"], semantic_role: OdooSemanticRole::Quantity },
        // R14: 'always' | 'ask' | 'never'; ask triggers wizard after 3 unmodified validations.
        OdooField { name: "autopost_bills", kind: OdooFieldKind::Selection, target: None,
            required: true, computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        // R15: company_dependent; enforcement (blocking invoice) is in the sale/invoice flow.
        OdooField { name: "credit_limit", kind: OdooFieldKind::Float, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Money },
        OdooField { name: "use_partner_credit_limit", kind: OdooFieldKind::Boolean, target: None,
            required: false, computed: Some("_compute_use_partner_credit_limit"), depends: &[],
            semantic_role: OdooSemanticRole::Policy },
        // R16: 'good' | 'normal' | 'bad'; company_dependent; PartnerTrustAdvisor savant input.
        OdooField { name: "trust", kind: OdooFieldKind::Selection, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
        // R17: computed from commercial_partner_id.invoice_edi_format_store;
        // l10n_de overrides _get_suggested_invoice_edi_format to return 'xrechnung'.
        OdooField { name: "invoice_edi_format", kind: OdooFieldKind::Selection, target: None,
            required: false, computed: Some("_compute_invoice_edi_format"), depends: &[],
            semantic_role: OdooSemanticRole::Document },
        OdooField { name: "invoice_edi_format_store", kind: OdooFieldKind::Char, target: None,
            required: false, computed: None, depends: &[], semantic_role: OdooSemanticRole::Document },
        // DSO denominator: SUM(price_subtotal) on posted sale invoices for commercial partner.
        OdooField { name: "total_invoiced", kind: OdooFieldKind::Monetary, target: None,
            required: false, computed: Some("_invoice_total"), depends: &[],
            semantic_role: OdooSemanticRole::Money },
    ],
    methods: &[
        // R4: deferred post-commit when rank > 0; immediate when rank == 0.
        OdooMethod { name: "_increase_rank", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        // R5: returns partner.commercial_partner_id — accounting entries post to commercial parent.
        OdooMethod { name: "_find_accounting_partner", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Record, triggers: &[] },
        // R5: extends base list with the 5 property fields synced from parent → children.
        OdooMethod { name: "_commercial_fields", kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_credit_debit_get", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_days_sales_outstanding", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        // R13: community stub returns (vat, country_code) unchanged; VIES is Enterprise-only.
        OdooMethod { name: "_run_vat_checks", kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Dict, triggers: &[] },
        // Returns bool(partner.vat) in community; fpos predicate 1 input.
        OdooMethod { name: "_get_vat_required_valid", kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Boolean, triggers: &[] },
        OdooMethod { name: "_compute_use_partner_credit_limit", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_inverse_use_partner_credit_limit", kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_invoice_edi_format", kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_inverse_invoice_edi_format", kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        // Stub; overridden in l10n_de to return 'xrechnung'.
        OdooMethod { name: "_get_suggested_invoice_edi_format", kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict, triggers: &[] },
        // R18: @api.ondelete — blocks deletion when any draft/posted account.move references partner.
        OdooMethod { name: "_unlink_if_partner_in_account_move", kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit, triggers: &[] },
        // R19: VAT guard on reparent + re-points move lines to new commercial_partner_id
        //      with bypass_lock_check (K11 GoBD consideration).
        OdooMethod { name: "write", kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean, triggers: &[] },
        OdooMethod { name: "create", kind: OdooMethodKind::ApiModelCreateMulti,
            return_kind: OdooReturnKind::Recordset, triggers: &[] },
    ],
    decorators: &[
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends, targets: &["credit"] },
        OdooDecorator { kind: OdooDecoratorKind::ApiDepends, targets: &["company_id", "country_code"] },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "Partner cannot be deleted if referenced by any account.move in draft or \
                    posted state (@api.ondelete guard, R18)",
        source_method: Some("_unlink_if_partner_in_account_move"),
    }],
    provenance: OdooProvenance {
        l_doc: "L9-PARTNER-FISCALPOS.md",
        l_doc_lines: (32, 513),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/partner.py",
            line_range: (326, 870),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 4. res.country ──────────────────────────────────────────────────────────
//
// Projected for the VAT-prefix / EU-country-code role in _get_fiscal_position:
// the 2-char ISO code drives intra_eu / vat_exclusion logic (R8).

/// `res.country` — country record; L9 projection for VAT-prefix matching.
///
/// L-doc R8 (EU VAT prefix logic); base module source.
pub const RES_COUNTRY: OdooEntity = OdooEntity {
    model_name: "res.country",
    description: "Country; 2-char ISO code used in _get_fiscal_position to compute \
                  intra_eu / vat_exclusion flags; also the country filter predicate in fpos matching.",
    fields: &[
        // 2-char ISO 3166-1 alpha-2 (e.g. "DE"). VAT prefix = code[:2].
        OdooField { name: "code", kind: OdooFieldKind::Char, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        // Used in states_count and fpos predicate 3 (state_ids).
        OdooField { name: "state_ids", kind: OdooFieldKind::One2many,
            target: Some("res.country.state"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L9-PARTNER-FISCALPOS.md",
        l_doc_lines: (207, 291),
        odoo_source: &[OdooSourceRef {
            path: "addons/base/models/res_country.py",
            line_range: (1, 80),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 5. res.country.group ─────────────────────────────────────────────────────
//
// Predicate 5 in _get_fpos_validation_functions: partner's country must be in
// group.country_ids; partner's state must NOT be in group.exclude_state_ids.

/// `res.country.group` — fpos predicate 5 (country-group match).
///
/// L-doc R8; Odoo source `res_country_group.py`.
pub const RES_COUNTRY_GROUP: OdooEntity = OdooEntity {
    model_name: "res.country.group",
    description: "Named set of countries for fpos matching predicate 5: \
                  partner.country_id ∈ country_ids AND partner.state_id ∉ exclude_state_ids.",
    fields: &[
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        // Inclusion set: partner.country_id must be in this M2M.
        OdooField { name: "country_ids", kind: OdooFieldKind::Many2many,
            target: Some("res.country"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // Exclusion set: partner's state (if any) must NOT appear here.
        OdooField { name: "exclude_state_ids", kind: OdooFieldKind::Many2many,
            target: Some("res.country.state"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L9-PARTNER-FISCALPOS.md",
        l_doc_lines: (263, 277),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/res_country_group.py",
            line_range: (1, 30),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 6. account.payment.term (reference anchor) ───────────────────────────────
//
// L5 is the AUTHORITATIVE lane for payment-term internals (Skonto, installments,
// due-date formulas).  L9 projects only the shell to document that
// property_payment_term_id / property_supplier_payment_term_id point here.

/// `account.payment.term` — reference anchor; **L5 is authoritative**.
///
/// L-doc R2; see `l5::PAYMENT_TERM` for the full entity.
pub const ACCOUNT_PAYMENT_TERM_REF: OdooEntity = OdooEntity {
    model_name: "account.payment.term",
    description: "Payment terms (L5 authoritative); projected in L9 only as target of \
                  property_payment_term_id / property_supplier_payment_term_id on res.partner (R2).",
    fields: &[
        OdooField { name: "name", kind: OdooFieldKind::Char, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "active", kind: OdooFieldKind::Boolean, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L9-PARTNER-FISCALPOS.md",
        l_doc_lines: (59, 78),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_payment_term.py",
            line_range: (1, 30),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// All 6 entities in lane L9 (partner accounting + fiscal-position selection
/// + trust / dunning-risk).
///
/// [0] `account.fiscal.position`         — fpos selection + map_tax + map_account
/// [1] `account.fiscal.position.account` — account-substitution line
/// [2] `res.partner` (account ext.)      — AR/AP, trust, credit, DSO, R14–R19
/// [3] `res.country`                     — VAT-prefix / EU-code anchor
/// [4] `res.country.group`               — fpos predicate 5
/// [5] `account.payment.term` (ref)      — partner property target (L5 authoritative)
pub const ENTITIES: &[OdooEntity] = &[
    ACCOUNT_FISCAL_POSITION,
    ACCOUNT_FISCAL_POS_ACCOUNT,
    RES_PARTNER_ACCOUNTING,
    RES_COUNTRY,
    RES_COUNTRY_GROUP,
    ACCOUNT_PAYMENT_TERM_REF,
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooConstraintKind, OdooFieldKind, OdooSemanticRole};

    #[test]
    fn entities_slice_has_six_entries() {
        assert_eq!(ENTITIES.len(), 6);
    }

    #[test]
    fn fiscal_position_identity() {
        assert_eq!(ACCOUNT_FISCAL_POSITION.model_name, "account.fiscal.position");
        assert_eq!(ACCOUNT_FISCAL_POSITION.provenance.confidence, OdooConfidence::Curated);
        assert_eq!(ACCOUNT_FISCAL_POSITION.provenance.l_doc, "L9-PARTNER-FISCALPOS.md");
        assert!(ACCOUNT_FISCAL_POSITION.state_machine.is_none());
    }

    #[test]
    fn fiscal_position_has_auto_apply_policy_field() {
        let f = ACCOUNT_FISCAL_POSITION
            .fields.iter().find(|f| f.name == "auto_apply")
            .expect("auto_apply must be present");
        assert_eq!(f.kind, OdooFieldKind::Boolean);
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn fiscal_position_has_get_fiscal_position_method() {
        let m = ACCOUNT_FISCAL_POSITION
            .methods.iter().find(|m| m.name == "_get_fiscal_position")
            .expect("_get_fiscal_position must be present");
        assert_eq!(m.kind, OdooMethodKind::ApiModel);
        assert_eq!(m.return_kind, OdooReturnKind::Record);
    }

    #[test]
    fn fiscal_position_has_map_tax_and_map_account() {
        let names: Vec<&str> = ACCOUNT_FISCAL_POSITION.methods.iter().map(|m| m.name).collect();
        assert!(names.contains(&"map_tax"), "map_tax must be present");
        assert!(names.contains(&"map_account"), "map_account must be present");
    }

    #[test]
    fn fiscal_position_zip_constraint_present() {
        let c = ACCOUNT_FISCAL_POSITION.constraints.iter()
            .find(|c| c.source_method == Some("_check_zip"))
            .expect("_check_zip constraint must be present");
        assert_eq!(c.kind, OdooConstraintKind::Python);
    }

    #[test]
    fn fiscal_position_account_unique_constraint() {
        assert_eq!(ACCOUNT_FISCAL_POS_ACCOUNT.model_name, "account.fiscal.position.account");
        let c = ACCOUNT_FISCAL_POS_ACCOUNT.constraints.first()
            .expect("SQL unique constraint must be present");
        assert_eq!(c.kind, OdooConstraintKind::Sql);
    }

    #[test]
    fn res_partner_accounting_trust_field() {
        assert_eq!(RES_PARTNER_ACCOUNTING.model_name, "res.partner");
        let f = RES_PARTNER_ACCOUNTING.fields.iter().find(|f| f.name == "trust")
            .expect("trust field must be present");
        assert_eq!(f.kind, OdooFieldKind::Selection);
        assert_eq!(f.semantic_role, OdooSemanticRole::Status);
    }

    #[test]
    fn res_partner_accounting_has_ar_ap_property_fields() {
        let names: Vec<&str> = RES_PARTNER_ACCOUNTING.fields.iter().map(|f| f.name).collect();
        assert!(names.contains(&"property_account_receivable_id"));
        assert!(names.contains(&"property_account_payable_id"));
        assert!(names.contains(&"property_account_position_id"));
        assert!(names.contains(&"property_payment_term_id"));
    }

    #[test]
    fn res_partner_accounting_has_fiscal_position_resolver_inputs() {
        // FiscalPositionResolver savant inputs: manual override, AR balance, trust signal.
        let names: Vec<&str> = RES_PARTNER_ACCOUNTING.fields.iter().map(|f| f.name).collect();
        assert!(names.contains(&"property_account_position_id"));
        assert!(names.contains(&"credit"));
        assert!(names.contains(&"trust"));
    }

    #[test]
    fn res_partner_accounting_has_deletion_guard() {
        let c = RES_PARTNER_ACCOUNTING.constraints.first()
            .expect("deletion guard constraint must be present");
        assert_eq!(c.kind, OdooConstraintKind::Python);
        assert_eq!(c.source_method, Some("_unlink_if_partner_in_account_move"));
    }

    #[test]
    fn res_country_has_code_identity_field() {
        assert_eq!(RES_COUNTRY.model_name, "res.country");
        let f = RES_COUNTRY.fields.iter().find(|f| f.name == "code")
            .expect("code field must be present");
        assert_eq!(f.kind, OdooFieldKind::Char);
        assert_eq!(f.semantic_role, OdooSemanticRole::Identity);
    }

    #[test]
    fn res_country_group_has_exclude_state_ids() {
        assert_eq!(RES_COUNTRY_GROUP.model_name, "res.country.group");
        let f = RES_COUNTRY_GROUP.fields.iter().find(|f| f.name == "exclude_state_ids")
            .expect("exclude_state_ids must be present");
        assert_eq!(f.kind, OdooFieldKind::Many2many);
    }

    #[test]
    fn payment_term_ref_is_reference_only() {
        assert_eq!(ACCOUNT_PAYMENT_TERM_REF.model_name, "account.payment.term");
        assert_eq!(ACCOUNT_PAYMENT_TERM_REF.fields.len(), 2);
        assert!(ACCOUNT_PAYMENT_TERM_REF.methods.is_empty());
    }

    #[test]
    fn all_entities_have_curated_confidence() {
        for e in ENTITIES {
            assert_eq!(e.provenance.confidence, OdooConfidence::Curated,
                "entity {} must be Curated", e.model_name);
        }
    }

    #[test]
    fn all_entities_reference_l9_l_doc() {
        for e in ENTITIES {
            assert_eq!(e.provenance.l_doc, "L9-PARTNER-FISCALPOS.md",
                "entity {} must reference L9 l_doc", e.model_name);
        }
    }
}
