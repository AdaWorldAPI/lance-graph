//! Lane L9 (PARTNER-FISCALPOS) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L9-PARTNER-FISCALPOS.md`.
//!
//! ## Entity inventory (6 entities)
//!
//! | Const                          | Odoo model                         | L-doc rules |
//! |---|---|---|
//! | [`ACCOUNT_FISCAL_POSITION`]    | `account.fiscal.position`          | R8–R12, R20 |
//! | [`ACCOUNT_FISCAL_POS_ACCOUNT`] | `account.fiscal.position.account`  | R10         |
//! | [`RES_PARTNER_ACCOUNTING`]     | `res.partner` (account extension)  | R1–R7, R14–R19 |
//! | [`RES_COUNTRY`]                | `res.country`                      | R8 (VAT prefix) |
//! | [`RES_COUNTRY_GROUP`]          | `res.country.group`                | R8 (group matching) |
//! | [`ACCOUNT_PAYMENT_TERM_REF`]   | `account.payment.term` (reference) | R2 (partner side) |
//!
//! ## L3 overlap note
//!
//! L3 is the authoritative lane for tax repartition INTERNALS
//! (`account.tax`, repartition lines, `_compute_tax_base_amount`, …).
//! L9 is the authoritative lane for partner-side fiscal-position SELECTION
//! (`_get_fiscal_position`, `map_tax`, `map_account`) and for the
//! per-partner property fields (`property_account_position_id`, trust, …).
//! Fields / methods owned by L3 are intentionally absent here; duplicating
//! them would conflict with L3's canonical coverage.
//!
//! ## Savant annotations
//!
//! - **`FiscalPositionResolver`** (family=0x80, reasoning=CustomerCategory,
//!   inference=Deduction, semiring=NarsTruth, style=Analytical) — drives
//!   `_get_fiscal_position` / `_get_first_matching_fpos` (R8, lines 207–285).
//! - **`PartnerTrustAdvisor`** (family=0x80, reasoning=CustomerCategory,
//!   inference=Induction, semiring=NarsTruth, style=Analytical) — uses the
//!   `trust` field (R16, line 456) to infer dunning escalation from payment
//!   history patterns.

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── 1. account.fiscal.position ──────────────────────────────────────────────
//
// L9 is the authoritative lane for the SELECTION side of fiscal positions:
// which position applies to a partner, how it maps taxes / accounts, and how
// it is resolved via `_get_fiscal_position`.
//
// L3 overlap: tax repartition internals (`account.tax.*`) live in L3.
// Fields not projected here: `tax_ids` M2M (the raw tax link) is present
// only as an input to `_compute_tax_map` — the repartition semantics (base,
// tax, refund lines, CABA) are L3 territory.

/// `account.fiscal.position` — partner-facing fiscal-position record.
///
/// L-doc R8–R12, R20; source: `addons/account/models/partner.py:26–301`.
///
/// This is the entity the `FiscalPositionResolver` savant selects and that
/// `map_tax` / `map_account` methods operate on.
pub const ACCOUNT_FISCAL_POSITION: OdooEntity = OdooEntity {
    model_name: "account.fiscal.position",
    description: "Named tax-and-account mapping rule applied to a partner or delivery address; \
                  selected by priority-ranked auto-detection (_get_fiscal_position) or manual \
                  partner override; drives FiscalPositionResolver savant.",
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
            required: false,
            computed: None,
            depends: &[],
            // Sort key used by _get_first_matching_fpos (company-specificity first,
            // then sequence ASC). Lower sequence = higher priority within same company tier.
            semantic_role: OdooSemanticRole::Policy,
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
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "auto_apply",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // When True, this position is a candidate for _get_fiscal_position
            // auto-detection (R8). When False, it can only be assigned manually.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "vat_required",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Predicate 1 in _get_fpos_validation_functions: only match if partner has VAT.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "country_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.country"),
            required: false,
            computed: None,
            depends: &[],
            // Predicate 4: partner.country_id must equal this country (or absent → any).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "country_group_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.country.group"),
            required: false,
            computed: None,
            depends: &[],
            // Predicate 5: partner.country_id must be in this group's country_ids;
            // also checks exclude_state_ids for the partner's state.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "state_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("res.country.state"),
            required: false,
            computed: None,
            depends: &[],
            // Predicate 3: partner.state_id must be in this set (or empty → any state).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "zip_from",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Predicate 2 lower bound; padded with leading zeros for digit-only zips
            // by _convert_zip_values (R11). Lexicographic comparison: zip_from <= partner.zip.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "zip_to",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Predicate 2 upper bound; must be >= zip_from (_check_zip constraint, R11).
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "foreign_vat",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // The company's own foreign tax ID in the jurisdiction of this position.
            // Validated via _run_vat_checks on write (R13).
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "is_domestic",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: Some("_compute_is_domestic"),
            depends: &["company_id.domestic_fiscal_position_id"],
            // True only if this fpos is the company's designated domestic position (R12).
            semantic_role: OdooSemanticRole::Status,
        },
        // tax_map and account_map (R20) are pre-built Binary dict caches.
        // Projected as Computed; raw content (dict) is not typed further here —
        // the lookup logic lives in map_tax / map_account methods below.
        OdooField {
            name: "tax_map",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_tax_map"),
            depends: &["tax_ids"],
            // {src_tax_id: [dest_tax_id, ...]} — used by map_tax at invoice-line time.
            // L3 overlap: content is tax repartition data, but the MAP structure is L9.
            semantic_role: OdooSemanticRole::Tax,
        },
        OdooField {
            name: "account_map",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_account_map"),
            depends: &["account_ids.account_src_id", "account_ids.account_dest_id"],
            // {src_account_id: dest_account_id} — used by map_account at posting time.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "account_ids",
            kind: OdooFieldKind::One2many,
            target: Some("account.fiscal.position.account"),
            required: false,
            computed: None,
            depends: &[],
            // Source of account_map; each line is a src→dest account substitution.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "note",
            kind: OdooFieldKind::Html,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Legal mentions to print on invoices when this fpos applies.
            semantic_role: OdooSemanticRole::Document,
        },
    ],
    methods: &[
        // ── R8 / FiscalPositionResolver savant boundary ───────────────────
        OdooMethod {
            // @api.model: resolves the applicable fiscal position for a partner.
            // Precedence: (1) manual property_account_position_id wins;
            // (2) if no country → empty; (3) auto-detect via _get_first_matching_fpos.
            // Savant FiscalPositionResolver owns the AXIS-B heuristic in step 3.
            name: "_get_fiscal_position",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Record,
            triggers: &[],
        },
        OdooMethod {
            // Priority-ordered scan: company-specific first (-len(parent_ids)), then sequence.
            // Runs all 5 validation predicates; returns first fpos where all pass.
            name: "_get_first_matching_fpos",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Record,
            triggers: &[],
        },
        OdooMethod {
            // Returns list of 5 lambda predicates: vat_required, zip_range,
            // state, country, country_group.  All must pass (AND semantics).
            name: "_get_fpos_validation_functions",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // ── R9 ────────────────────────────────────────────────────────────
        OdooMethod {
            // Tax remapping: taxes → mapped taxes via tax_map dict.
            // Special: empty fpos + fpos-aware taxes → removes all such taxes.
            name: "map_tax",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
        },
        // ── R10 ───────────────────────────────────────────────────────────
        OdooMethod {
            // Account substitution: account_id → mapped account via account_map dict.
            // Identity fallback if no mapping exists for this account.
            name: "map_account",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Record,
            triggers: &[],
        },
        // ── R11 ───────────────────────────────────────────────────────────
        OdooMethod {
            // @api.model: pads digit-only zip codes to equal length with leading zeros
            // so lexicographic range comparison works for numeric postal codes (e.g. DE PLZ).
            name: "_convert_zip_values",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "create",
            kind: OdooMethodKind::ApiModelCreateMulti,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
        },
        OdooMethod {
            name: "write",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        // ── R12 ───────────────────────────────────────────────────────────
        OdooMethod {
            name: "_compute_is_domestic",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R20 ───────────────────────────────────────────────────────────
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
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["company_id.domestic_fiscal_position_id"],
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
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["country_id", "country_group_id", "state_ids", "foreign_vat"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "zip_from and zip_to must both be set or both absent; zip_from <= zip_to \
                        (lexicographic after leading-zero padding for digit-only zips)",
            source_method: Some("_check_zip"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "foreign_vat requires country_id; if country is fiscal country, \
                        state_ids required when country has states; no duplicate foreign_vat \
                        per country per company",
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
//
// One account-substitution line within a fiscal position.  The full set of
// lines populates the `account_map` (Binary computed) on the parent fpos.
// Constraint: (position_id, account_src_id, account_dest_id) must be unique.

/// `account.fiscal.position.account` — one src→dest account mapping line.
///
/// L-doc R10; source: `addons/account/models/partner.py:303–324`.
pub const ACCOUNT_FISCAL_POS_ACCOUNT: OdooEntity = OdooEntity {
    model_name: "account.fiscal.position.account",
    description: "One account-substitution rule inside a fiscal position: \
                  account_src_id is replaced with account_dest_id when the \
                  fiscal position is active (used by map_account at posting time).",
    fields: &[
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
            // Related from position_id.company_id, stored for check_company_auto.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "account_src_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: true,
            computed: None,
            depends: &[],
            // The source (pre-substitution) account — used as lookup key in account_map.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "account_dest_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: true,
            computed: None,
            depends: &[],
            // The destination (post-substitution) account that replaces account_src_id.
            semantic_role: OdooSemanticRole::Reference,
        },
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
// L9 projects only the fields added / overridden by the `account` module.
// Base `res.partner` fields (name, vat, country_id, state_id, zip, …) are
// referenced from the base module and are NOT redeclared here — they appear
// in the `depends` chains and savant inputs.
//
// Skipped (L3 / other lane authority):
//   - `tax_ids` on the partner — L3 owns tax-repartition internals.
//   - `bank_ids` / `res.partner.bank` — a separate lane.
//   - `ref_company_ids` — company-structure lane.
//
// Savant annotations:
//   - `FiscalPositionResolver`: input fields → country_id, vat, state_id, zip,
//     property_account_position_id, delivery address (R8).
//   - `PartnerTrustAdvisor`: input field → trust (R16); dunning escalation.

/// `res.partner` — accounting extension (account module `_inherit`).
///
/// L-doc R1–R7, R14–R19; source: `addons/account/models/partner.py:326–870`.
pub const RES_PARTNER_ACCOUNTING: OdooEntity = OdooEntity {
    model_name: "res.partner",
    description: "Partner model extended by the account module: adds AR/AP property accounts, \
                  payment terms, fiscal-position override, trust/dunning-risk signal, \
                  customer/supplier rank counters, credit-limit guard, and EDI format.",
    fields: &[
        // ── R1: Per-partner AR/AP accounts ────────────────────────────────
        OdooField {
            name: "property_account_receivable_id",
            kind: OdooFieldKind::Property,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            // company_dependent; domain: account_type='asset_receivable'; ondelete=restrict.
            // Inherited by child partners via commercial_partner_id rollup (R5).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "property_account_payable_id",
            kind: OdooFieldKind::Property,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            // company_dependent; domain: account_type='liability_payable'; ondelete=restrict.
            // Inherited by child partners via commercial_partner_id rollup (R5).
            semantic_role: OdooSemanticRole::Reference,
        },
        // ── R2: Per-partner payment terms ─────────────────────────────────
        OdooField {
            name: "property_payment_term_id",
            kind: OdooFieldKind::Property,
            target: Some("account.payment.term"),
            required: false,
            computed: None,
            depends: &[],
            // company_dependent; customer invoice due-date source. None = immediate/fallback.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "property_supplier_payment_term_id",
            kind: OdooFieldKind::Property,
            target: Some("account.payment.term"),
            required: false,
            computed: None,
            depends: &[],
            // company_dependent; vendor bill due-date source. None = immediate/fallback.
            semantic_role: OdooSemanticRole::Policy,
        },
        // ── R3: Manual fiscal-position override ───────────────────────────
        OdooField {
            name: "property_account_position_id",
            kind: OdooFieldKind::Property,
            target: Some("account.fiscal.position"),
            required: false,
            computed: None,
            depends: &[],
            // company_dependent; ALWAYS wins over auto-detection in _get_fiscal_position.
            // FiscalPositionResolver reads this as priority-1 input (R8 step 5).
            semantic_role: OdooSemanticRole::Tax,
        },
        // ── R4: customer/supplier rank ────────────────────────────────────
        OdooField {
            name: "customer_rank",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Incremented via _increase_rank on invoice/sale confirm; deferred post-commit
            // when rank > 0 (serialization-error mitigation in Odoo/PG).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "supplier_rank",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Incremented via _increase_rank on purchase/vendor-bill confirm.
            semantic_role: OdooSemanticRole::Quantity,
        },
        // ── R6: credit / debit computed balances ──────────────────────────
        OdooField {
            name: "credit",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_credit_debit_get"),
            depends: &[],
            // SUM(amount_residual) on posted asset_receivable lines not reconciled.
            // Computed via raw SQL (_asset_difference_search); PG-specific SPLIT_PART.
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "debit",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_credit_debit_get"),
            depends: &[],
            // SUM(-amount_residual) on posted liability_payable lines not reconciled.
            semantic_role: OdooSemanticRole::Money,
        },
        // ── R7: DSO ───────────────────────────────────────────────────────
        OdooField {
            name: "days_sales_outstanding",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_days_sales_outstanding"),
            depends: &["credit"],
            // (credit / total_invoiced_tax_included) * days_since_oldest_invoice.
            // Zero if total_invoiced == 0 (no invoices yet, not an error).
            semantic_role: OdooSemanticRole::Quantity,
        },
        // ── R14: autopost_bills ───────────────────────────────────────────
        OdooField {
            name: "autopost_bills",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Values: 'always' | 'ask' | 'never'. Default: 'ask'.
            // 'ask' triggers a wizard after 3 unmodified validations (counter in move flow).
            semantic_role: OdooSemanticRole::Policy,
        },
        // ── R15: credit_limit ─────────────────────────────────────────────
        OdooField {
            name: "credit_limit",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // company_dependent; falls back to company-level default.
            // Enforcement (blocking invoice) is in the sale/invoice flow, not here.
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "use_partner_credit_limit",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: Some("_compute_use_partner_credit_limit"),
            depends: &[],
            // True if partner's credit_limit differs from company default.
            // Inverse resets partner limit to company default when toggled off.
            semantic_role: OdooSemanticRole::Policy,
        },
        // ── R16: trust — PartnerTrustAdvisor savant input ─────────────────
        OdooField {
            name: "trust",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Values: 'good' | 'normal' | 'bad'. company_dependent.
            // Manually set by accounting staff; PartnerTrustAdvisor can suggest
            // based on payment-history patterns (Induction inference, NarsTruth semiring).
            semantic_role: OdooSemanticRole::Status,
        },
        // ── R17: EDI format ───────────────────────────────────────────────
        OdooField {
            name: "invoice_edi_format",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: Some("_compute_invoice_edi_format"),
            depends: &[],
            // Computed from commercial_partner_id.invoice_edi_format_store;
            // falls back to _get_suggested_invoice_edi_format() (stub, overridden in l10n_de).
            // Inverse stores 'none' to suppress suggestion.
            semantic_role: OdooSemanticRole::Document,
        },
        OdooField {
            name: "invoice_edi_format_store",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // company_dependent backing store for invoice_edi_format.
            // 'none' = explicitly disabled; '' = use suggestion; otherwise the format.
            semantic_role: OdooSemanticRole::Document,
        },
        // ── R5 helper field ───────────────────────────────────────────────
        OdooField {
            name: "total_invoiced",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_invoice_total"),
            depends: &[],
            // SUM(price_subtotal) on posted sale invoices for this commercial partner.
            // Used as denominator in DSO computation (R7).
            semantic_role: OdooSemanticRole::Money,
        },
    ],
    methods: &[
        // ── R4 ─────────────────────────────────────────────────────────────
        OdooMethod {
            // Increments customer_rank or supplier_rank.  Deferred post-commit hook
            // when rank > 0 (PG serialization-error mitigation); immediate when rank == 0.
            name: "_increase_rank",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R5 ─────────────────────────────────────────────────────────────
        OdooMethod {
            // Returns partner.commercial_partner_id — all accounting entries for a contact
            // are posted to the commercial (company-level) parent partner.
            name: "_find_accounting_partner",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Record,
            triggers: &[],
        },
        OdooMethod {
            // Extends base _commercial_fields with the five partner-accounting fields
            // that are synced from parent to children when a partner gets a parent company.
            name: "_commercial_fields",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // ── R6 ─────────────────────────────────────────────────────────────
        OdooMethod {
            name: "_credit_debit_get",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R7 ─────────────────────────────────────────────────────────────
        OdooMethod {
            name: "_compute_days_sales_outstanding",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R13 ────────────────────────────────────────────────────────────
        OdooMethod {
            // Community stub: returns (vat, country_code) unchanged.
            // Real validation is in base_vat module; VIES lookup is Enterprise-only.
            name: "_run_vat_checks",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            // Returns bool(partner.vat) in community; VIES check in Enterprise.
            // Used as predicate 1 in _get_fpos_validation_functions.
            name: "_get_vat_required_valid",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        // ── R15 ────────────────────────────────────────────────────────────
        OdooMethod {
            name: "_compute_use_partner_credit_limit",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_inverse_use_partner_credit_limit",
            kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R17 ────────────────────────────────────────────────────────────
        OdooMethod {
            name: "_compute_invoice_edi_format",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_inverse_invoice_edi_format",
            kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            // Stub returning False in community; overridden in l10n_de to return 'xrechnung'.
            name: "_get_suggested_invoice_edi_format",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // ── R18 ────────────────────────────────────────────────────────────
        OdooMethod {
            // @api.ondelete: prevents deletion of partner referenced by any draft or
            // posted account.move.  Raises UserError; applies to ALL states, not just posted.
            name: "_unlink_if_partner_in_account_move",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R19 ────────────────────────────────────────────────────────────
        OdooMethod {
            // write() override: VAT guard on reparent (raises UserError if VAT differs);
            // re-points existing move lines to new commercial_partner_id with bypass_lock_check.
            name: "write",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        OdooMethod {
            name: "create",
            kind: OdooMethodKind::ApiModelCreateMulti,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["credit"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["company_id", "country_code"],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "Partner cannot be deleted if referenced by any account.move in draft or \
                    posted state (@api.ondelete guard)",
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
// Projected here for the VAT-prefix / EU-country-code role it plays in
// _get_fiscal_position (R8): the two-char ISO code drives the intra_eu /
// vat_exclusion logic.  Only the fields relevant to L9 are captured; the
// full country model lives in the base module.

/// `res.country` — referenced by L9 for EU VAT-prefix matching.
///
/// L-doc R8; source: base module `res.country`.
pub const RES_COUNTRY: OdooEntity = OdooEntity {
    model_name: "res.country",
    description: "Country record; in L9 context: provides the 2-char ISO code used in \
                  _get_fiscal_position to compute intra_eu / vat_exclusion flags \
                  (VAT prefix comparison) and serves as the country filter predicate \
                  for fiscal-position auto-detection.",
    fields: &[
        OdooField {
            name: "code",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // 2-char ISO 3166-1 alpha-2 code (e.g. "DE", "FR").  Used as VAT prefix.
            // `eu_country_codes` set = set of codes from base.europe.country_ids.
            semantic_role: OdooSemanticRole::Identity,
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
            name: "state_ids",
            kind: OdooFieldKind::One2many,
            target: Some("res.country.state"),
            required: false,
            computed: None,
            depends: &[],
            // Used in states_count compute and predicate 3 of fiscal-position matching.
            semantic_role: OdooSemanticRole::Reference,
        },
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
// Used in predicate 5 of _get_fpos_validation_functions: partner's country must
// be in the group's country_ids, AND the partner's state must not appear in
// exclude_state_ids.

/// `res.country.group` — country group used as predicate 5 in fpos matching.
///
/// L-doc R8; source: `addons/account/models/res_country_group.py`.
pub const RES_COUNTRY_GROUP: OdooEntity = OdooEntity {
    model_name: "res.country.group",
    description: "Named set of countries; used in fiscal-position auto-detection predicate 5: \
                  partner.country_id must be in group.country_ids AND partner.state_id must \
                  not be in group.exclude_state_ids.",
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
            name: "country_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("res.country"),
            required: false,
            computed: None,
            depends: &[],
            // The inclusion set for predicate 5: partner.country_id IN country_ids.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "exclude_state_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("res.country.state"),
            required: false,
            computed: None,
            depends: &[],
            // Exclusion set for predicate 5: if partner has a state, it must NOT
            // appear in this set (e.g. exclude Swiss cantons from EU group).
            semantic_role: OdooSemanticRole::Reference,
        },
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

// ─── 6. account.payment.term (partner-side reference) ────────────────────────
//
// L5 is the AUTHORITATIVE lane for account.payment.term internals (installment
// computation, Skonto, due-date formulas).  L9 projects only the entity shell
// to document that property_payment_term_id / property_supplier_payment_term_id
// on res.partner point HERE, and to satisfy cross-reference queries from the
// partner-accounting surface.  Do NOT expand fields here; consult L5.

/// `account.payment.term` — referenced from L9 partner property fields.
///
/// **L5 is the authoritative lane** for all payment-term internals.
/// L9 projects this entity only as a reference anchor for the partner
/// property fields (R2).  See `l5::PAYMENT_TERM` for the full entity.
///
/// L-doc R2; source: `addons/account/models/account_payment_term.py`.
pub const ACCOUNT_PAYMENT_TERM_REF: OdooEntity = OdooEntity {
    model_name: "account.payment.term",
    description: "Payment terms record (L5 authoritative); referenced here because \
                  property_payment_term_id and property_supplier_payment_term_id on \
                  res.partner point to this model (R2). See l5::PAYMENT_TERM for \
                  the full entity with Skonto, installments, and due-date logic.",
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

/// All 6 entities documented in lane L9 (partner accounting + fiscal position
/// selection + trust / dunning-risk).
///
/// Entity index:
///   [0] `account.fiscal.position`         — fpos selection + map_tax + map_account
///   [1] `account.fiscal.position.account` — account-substitution line
///   [2] `res.partner` (account ext.)      — AR/AP props, trust, credit, DSO
///   [3] `res.country`                     — VAT-prefix / EU-code anchor
///   [4] `res.country.group`               — fpos predicate 5 (group match)
///   [5] `account.payment.term` (ref)      — partner property target (L5 authoritative)
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
            .fields
            .iter()
            .find(|f| f.name == "auto_apply")
            .expect("auto_apply must be present");
        assert_eq!(f.kind, OdooFieldKind::Boolean);
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn fiscal_position_has_get_fiscal_position_method() {
        let m = ACCOUNT_FISCAL_POSITION
            .methods
            .iter()
            .find(|m| m.name == "_get_fiscal_position")
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
        let c = ACCOUNT_FISCAL_POSITION
            .constraints
            .iter()
            .find(|c| c.source_method == Some("_check_zip"))
            .expect("_check_zip constraint must be present");
        assert_eq!(c.kind, OdooConstraintKind::Python);
    }

    #[test]
    fn fiscal_position_account_unique_constraint() {
        assert_eq!(ACCOUNT_FISCAL_POS_ACCOUNT.model_name, "account.fiscal.position.account");
        let c = ACCOUNT_FISCAL_POS_ACCOUNT
            .constraints
            .first()
            .expect("SQL unique constraint must be present");
        assert_eq!(c.kind, OdooConstraintKind::Sql);
    }

    #[test]
    fn res_partner_accounting_trust_field() {
        assert_eq!(RES_PARTNER_ACCOUNTING.model_name, "res.partner");
        let f = RES_PARTNER_ACCOUNTING
            .fields
            .iter()
            .find(|f| f.name == "trust")
            .expect("trust field must be present");
        assert_eq!(f.kind, OdooFieldKind::Selection);
        assert_eq!(f.semantic_role, OdooSemanticRole::Status);
    }

    #[test]
    fn res_partner_accounting_has_ar_ap_property_fields() {
        let names: Vec<&str> = RES_PARTNER_ACCOUNTING.fields.iter().map(|f| f.name).collect();
        assert!(names.contains(&"property_account_receivable_id"), "AR property must be present");
        assert!(names.contains(&"property_account_payable_id"), "AP property must be present");
        assert!(names.contains(&"property_account_position_id"), "fpos property must be present");
        assert!(names.contains(&"property_payment_term_id"), "payment term property must be present");
    }

    #[test]
    fn res_partner_accounting_has_fiscal_position_resolver_inputs() {
        // FiscalPositionResolver savant inputs: property_account_position_id (manual override),
        // credit/debit (AR balance), trust (risk signal).
        let names: Vec<&str> = RES_PARTNER_ACCOUNTING.fields.iter().map(|f| f.name).collect();
        assert!(names.contains(&"property_account_position_id"));
        assert!(names.contains(&"credit"));
        assert!(names.contains(&"trust"));
    }

    #[test]
    fn res_partner_accounting_has_deletion_guard() {
        let c = RES_PARTNER_ACCOUNTING
            .constraints
            .first()
            .expect("deletion guard constraint must be present");
        assert_eq!(c.kind, OdooConstraintKind::Python);
        assert_eq!(c.source_method, Some("_unlink_if_partner_in_account_move"));
    }

    #[test]
    fn res_country_has_code_identity_field() {
        assert_eq!(RES_COUNTRY.model_name, "res.country");
        let f = RES_COUNTRY
            .fields
            .iter()
            .find(|f| f.name == "code")
            .expect("code field must be present");
        assert_eq!(f.kind, OdooFieldKind::Char);
        assert_eq!(f.semantic_role, OdooSemanticRole::Identity);
    }

    #[test]
    fn res_country_group_has_exclude_state_ids() {
        assert_eq!(RES_COUNTRY_GROUP.model_name, "res.country.group");
        let f = RES_COUNTRY_GROUP
            .fields
            .iter()
            .find(|f| f.name == "exclude_state_ids")
            .expect("exclude_state_ids must be present");
        assert_eq!(f.kind, OdooFieldKind::Many2many);
    }

    #[test]
    fn payment_term_ref_is_reference_only() {
        assert_eq!(ACCOUNT_PAYMENT_TERM_REF.model_name, "account.payment.term");
        // Only 2 fields projected (name + active) — full entity is in L5.
        assert_eq!(ACCOUNT_PAYMENT_TERM_REF.fields.len(), 2);
        assert!(ACCOUNT_PAYMENT_TERM_REF.methods.is_empty());
    }

    #[test]
    fn all_entities_have_curated_confidence() {
        for e in ENTITIES {
            assert_eq!(
                e.provenance.confidence,
                OdooConfidence::Curated,
                "entity {} must be Curated",
                e.model_name
            );
        }
    }

    #[test]
    fn all_entities_reference_l9_l_doc() {
        for e in ENTITIES {
            assert_eq!(
                e.provenance.l_doc,
                "L9-PARTNER-FISCALPOS.md",
                "entity {} must reference L9 l_doc",
                e.model_name
            );
        }
    }
}
