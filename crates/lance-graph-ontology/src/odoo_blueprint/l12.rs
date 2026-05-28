//! Lane L12 (MULTICOMPANY-CURRENCY) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L12-MULTICOMPANY-CURRENCY.md`.
//!
//! ## Entity inventory (5 entities)
//!
//! | Const                         | Odoo model          | L-doc rules |
//! |---|---|---|
//! | [`RES_CURRENCY`]              | `res.currency`      | R1–R8 (rounding, rates, enable/disable) |
//! | [`RES_CURRENCY_RATE`]         | `res.currency.rate` | R3–R4 (rate lookup, three representations) |
//! | [`RES_COMPANY_MULTICOMPANY`]  | `res.company`       | R9–R11 (tree, root delegation, branches) |
//! | [`RES_USERS_COMPANY_ACCESS`]  | `res.users`         | R10 (UserCompanyAccessAdvisor) |
//! | [`ACCOUNT_ACCOUNT_EXCHANGE`]  | `account.account`   | R18 (ExchangeAccountSelector) |
//!
//! ## L4 overlap note
//!
//! L4 is the authoritative lane for `res.company` DE-specific aspects
//! (`restrictive_audit_trail`, DIN 5008 layout, GoBD Festschreibung — see
//! `l4::RES_COMPANY_DE`).  L12 covers the multi-company tree + currency
//! delegation aspects only.  The `income_currency_exchange_account_id` and
//! `expense_currency_exchange_account_id` fields live on `res.company`
//! (company.py:L135-145) but are projected here in [`ACCOUNT_ACCOUNT_EXCHANGE`]
//! as the *target* model — `res.company` fields that point to them are captured
//! in [`RES_COMPANY_MULTICOMPANY`].
//!
//! ## Savant annotations
//!
//! - **`CurrencySelectionAdvisor`** (family=0x62, reasoning=NextBestAction,
//!   inference=Induction, semiring=NarsTruth, style=Analytical) — drives
//!   currency enable/disable suggestions (R6, L-doc lines 41-43).
//! - **`ReportRateTypeSelector`** (family=0x62, reasoning=Other("ConsolidationRatePolicy"),
//!   inference=Deduction, semiring=Boolean, style=Analytical) — selects
//!   current/historical/average rate type per report line (R8, L-doc lines 47-49).
//!   No dedicated Odoo field carries the rate-type enum — it is a report-engine
//!   parameter resolved at report-generation time.
//! - **`UserCompanyAccessAdvisor`** (family=0x80, reasoning=CustomerCategory,
//!   inference=Induction, semiring=NarsTruth, style=Analytical) — branch-access
//!   scoping by user role/context (R10, L-doc lines 54-56).
//! - **`ExchangeAccountSelector`** (family=0x62, reasoning=Other("ChartAccountMapping"),
//!   inference=Deduction, semiring=Boolean, style=Analytical) — deterministic
//!   sign picks gain/loss exchange account (R18, L-doc lines 79-81).

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooEntityKind, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind,
    OdooProvenance, OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── 1. res.currency ─────────────────────────────────────────────────────────
//
// Core currency model.  Key rules:
//   R1: decimal_places from rounding field (log10-based, digits=(12,6)).
//   R2: round/is_zero/compare_amounts — HALF-UP, not banker's; compare_amounts
//       rounds BOTH before comparing (documented asymmetry with is_zero(a-b)).
//   R3: _get_rates — latest rate WHERE name<=date AND company_id IN (NULL, root_id).
//   R4: Three rate representations: `rate` (technical), `company_rate`, `inverse_company_rate`.
//   R5: _get_conversion_rate / _convert — same currency → 1; else to_rate/from_rate.
//   R6: group_multi_currency toggle (CurrencySelectionAdvisor savant boundary).
//   R7: rounding write-protection if _has_accounting_entries().
//   R8: currency-table builders current/historical/average (ReportRateTypeSelector savant).
//
// NOTE: engine uses only `rate` (stored); company_rate/inverse_company_rate are
// UI-computed write paths.  Write priority: inverse > company > rate.

/// `res.currency` — ISO 4217 currency with rounding rules and FX rate management.
///
/// L-doc R1–R8; sources: `base/models/res_currency.py:L1-504`,
/// `account/models/res_currency.py:L1-285`.
///
/// Drives `CurrencySelectionAdvisor` (R6) and `ReportRateTypeSelector` (R8).
pub const RES_CURRENCY: OdooEntity = OdooEntity {
    model_name: "res.currency",
    kind: OdooEntityKind::Model,
    description: "ISO 4217 currency: defines rounding precision (decimal_places derived from \
                  rounding via log10), three-representation FX rate store (rate/company_rate/ \
                  inverse_company_rate), and drives enable/disable decisions via \
                  CurrencySelectionAdvisor. Rate-table builders power ReportRateTypeSelector \
                  (current/historical/average per IFRS-vs-HGB policy).",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // ISO 4217 3-letter code (e.g. "EUR", "USD", "JPY").
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "symbol",
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
            // CurrencySelectionAdvisor input: active currency count > 1 enables
            // group_multi_currency.  Cannot deactivate if still used as company currency_id.
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            // R1: if 0 < rounding < 1: decimal_places = ceil(log10(1/rounding)) else 0.
            // digits=(12,6).  Rounding mode: HALF-UP to multiple of rounding value.
            name: "rounding",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            // Derived from rounding per R1: ceil(log10(1/rounding)) for 0<rounding<1.
            name: "decimal_places",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: Some("_compute_current_rate"),
            depends: &["rounding"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            // R4 technical rate: foreign units per 1 base currency (stored).
            // Engine uses ONLY this field for conversions.
            name: "rate",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_current_rate"),
            depends: &["rate_ids.rate", "rate_ids.company_id", "rate_ids.name"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            // R4 UI rate: rate / last_company_rate.  Write sets underlying `rate`.
            name: "company_rate",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_current_rate"),
            depends: &["rate_ids.rate", "rate_ids.company_id", "rate_ids.name"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            // R4 inverse UI rate: 1/company_rate.  Highest write priority.
            name: "inverse_company_rate",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: Some("_compute_current_rate"),
            depends: &["rate_ids.rate", "rate_ids.company_id", "rate_ids.name"],
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "rate_ids",
            kind: OdooFieldKind::One2many,
            target: Some("res.currency.rate"),
            required: false,
            computed: None,
            depends: &[],
            // Historical rate records; R3 selects from these by date + company.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            // Full ISO name for display.
            name: "full_name",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "position",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // 'before' | 'after' — symbol position relative to amount string.
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[
        OdooMethod {
            // R3: latest rate WHERE name<=date AND company_id IN (NULL, root_id)
            // ORDER BY company_id DESC, name DESC LIMIT 1.  Uses root_id NOT company.id.
            // Fallback: oldest rate → 1.0.
            name: "_get_rates",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            // R5: same currency → 1.0; else to_rate/from_rate.
            // Result: to_currency.round(from_amount * rate) if round=True.
            name: "_get_conversion_rate",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        OdooMethod {
            // R5: zero short-circuits to 0.0; else from_amount * _get_conversion_rate.
            name: "_convert",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Money,
            triggers: &[],
        },
        OdooMethod {
            // R2: float_round HALF-UP to multiple of rounding.
            name: "round",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        OdooMethod {
            // R2: |a| < rounding*0.5 + eps (float epsilon guard 2^-52).
            name: "is_zero",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        OdooMethod {
            // R2: rounds BOTH amounts FIRST, then compares — NOT equal to is_zero(a-b).
            // Documented asymmetry: must replicate exactly.
            name: "compare_amounts",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        OdooMethod {
            // R7: raises UserError if decimal_places would increase (rounding reduced)
            // or become 0 when _has_accounting_entries() is True.
            name: "write",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        OdooMethod {
            // R7: True if any account.move.line uses this currency.
            name: "_has_accounting_entries",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        OdooMethod {
            // R8: builds current/historical/average rate tables for reporting.
            // current = latest at date; historical = rate at specific date;
            // average = time-weighted (LEAD window).  ReportRateTypeSelector savant boundary.
            name: "_get_rates_table",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            // R6: add group_multi_currency if active count > 1; remove if ≤ 1.
            name: "_activate_group_multi_currency",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_current_rate",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            // Inverse for company_rate / inverse_company_rate write paths.
            name: "_inverse_company_rate",
            kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["rate_ids.rate", "rate_ids.company_id", "rate_ids.name"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["rounding"],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "cannot reduce decimal_places (raise rounding) or set to 0 rounding when \
                    _has_accounting_entries() is True — protects historical amounts from \
                    retroactive rounding loss (R7, account/res_currency.py:L26-41)",
        source_method: Some("write"),
    }],
    provenance: OdooProvenance {
        l_doc: "L12-MULTICOMPANY-CURRENCY.md",
        l_doc_lines: (25, 82),
        odoo_source: &[
            OdooSourceRef {
                path: "base/models/res_currency.py",
                line_range: (1, 504),
            },
            OdooSourceRef {
                path: "account/models/res_currency.py",
                line_range: (1, 285),
            },
        ],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── 2. res.currency.rate ────────────────────────────────────────────────────
//
// One time-stamped exchange rate record.  R3 and R4 rules:
//   R3: rate lookup selects company-specific OVER global (company_id=NULL).
//       Uses root_id — branches share root company rates.
//       Unique constraint: (name, currency_id, company_id).
//       Branches CANNOT have their own rates (only root companies).
//   R4: `rate` is stored (technical: foreign units per 1 base); the other
//       two representations (company_rate, inverse_company_rate) are computed
//       from `rate` at query time.

/// `res.currency.rate` — one dated FX exchange rate record.
///
/// L-doc R3–R4; source: `base/models/res_currency.py:L120-139`.
///
/// Unique per (name, currency_id, company_id).  company_id=NULL = global
/// rate; company_id=root_id = company-specific (preferred by _get_rates).
pub const RES_CURRENCY_RATE: OdooEntity = OdooEntity {
    model_name: "res.currency.rate",
    kind: OdooEntityKind::Model,
    description: "One dated FX exchange rate: foreign units per 1 base currency (technical `rate`). \
                  Lookup uses root_id, not company_id — branches inherit root rates. \
                  Only root companies may have dedicated rate records (no branch-level rates). \
                  Company-specific rates win over global (company_id=NULL) ones.",
    fields: &[
        OdooField {
            // Date key for rate lookup: name <= query_date ORDER BY name DESC.
            name: "name",
            kind: OdooFieldKind::Date,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            // Technical stored rate: foreign currency units per 1 base currency unit.
            // Engine always uses this field; company_rate / inverse_company_rate are derived.
            name: "rate",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
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
            // NULL = global rate; root_id = company-specific (preferred by _get_rates R3).
            // Branches MUST NOT have their own rates — constraint enforced.
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "UNIQUE(name, currency_id, company_id) — one rate per date per currency \
                        per company (NULL company = global rate)",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "company_id must be a root company (no branches can have dedicated rates); \
                        rate lookup uses company.root_id — branches share root rates (R3, R9)",
            source_method: None,
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L12-MULTICOMPANY-CURRENCY.md",
        l_doc_lines: (31, 49),
        odoo_source: &[OdooSourceRef {
            path: "base/models/res_currency.py",
            line_range: (120, 139),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── 3. res.company (multi-company + currency aspects) ───────────────────────
//
// L4 overlap: L4 owns DE-specific fields (restrictive_audit_trail, DIN 5008,
// GoBD Festschreibung) — see l4::RES_COMPANY_DE.  L12 owns:
//   R9: company tree (_parent_store, root_id, parent_id immutable after create).
//       currency_id is a root-delegated field → branches always inherit root currency.
//   R10: _accessible_branches — subset of branches accessible to current user
//        (UserCompanyAccessAdvisor savant boundary).
//   R11: _check_company_auto=True; related record company must be in parent_ids.
//   R18: income_currency_exchange_account_id and expense_currency_exchange_account_id
//        on company — ExchangeAccountSelector savant target.
//        Journal driven by currency_exchange_journal_id.
//
// Multi-currency = different ROOT companies.  Branches cannot differ in currency.

/// `res.company` — multi-company tree + currency delegation aspects.
///
/// L-doc R9–R11, R18; source: `base/models/res_company.py:L96-104, L341-450`.
///
/// L4 overlap: DE-specific fields live in `l4::RES_COMPANY_DE`.
/// L12 owns tree structure, root_id, currency delegation, branch-access
/// scoping, and FX exchange account pointers.
pub const RES_COMPANY_MULTICOMPANY: OdooEntity = OdooEntity {
    model_name: "res.company",
    kind: OdooEntityKind::Model,
    description: "Company / branch node in the multi-company tree (res.company _parent_store). \
                  root_id = the tree root (branches inherit root currency). parent_id immutable \
                  after create. currency_id is root-delegated (branches cannot override). \
                  Holds FX gain/loss account pointers (ExchangeAccountSelector) and the \
                  exchange journal. UserCompanyAccessAdvisor scopes branch access per user.",
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
            // Parent in the company tree.  Immutable after create (R9).
            name: "parent_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            // Root ancestor in the tree (_parent_store).  Branches share root rates.
            // _get_rates uses root_id in the lookup (R3, R9).
            name: "root_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: false,
            computed: Some("_compute_root_id"),
            depends: &["parent_id", "parent_id.root_id"],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "child_ids",
            kind: OdooFieldKind::One2many,
            target: Some("res.company"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            // Root-delegated: branches always inherit this from root_id.
            // Multi-currency means different ROOT companies, not different branches.
            name: "currency_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.currency"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            // R18 / ExchangeAccountSelector: sign > 0 → expense (loss) account.
            // Deterministic sign-driven selection: compare_amounts(open_balance, 0) > 0.
            name: "expense_currency_exchange_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            // ExchangeAccountSelector savant: heuristic only for initial SKR config;
            // at runtime selection is deterministic from balance sign.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            // R18 / ExchangeAccountSelector: sign < 0 → income (gain) account.
            name: "income_currency_exchange_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            // R18: FX difference entries posted to this journal.
            name: "currency_exchange_journal_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.journal"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[
        OdooMethod {
            // R9: tree root delegation — returns the root-delegated field names.
            // ['currency_id'] is the relevant entry: branches always use root currency.
            name: "_get_company_root_delegated_field_names",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            // R10 / UserCompanyAccessAdvisor: returns subset of branches accessible
            // to the current user in a multi-branch context.
            name: "_accessible_branches",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_root_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["parent_id", "parent_id.root_id"],
    }],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "parent_id is immutable after company creation (R9); raises UserError \
                        if parent_id is changed on an existing record",
            source_method: Some("write"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Domain,
            condition: "currency_id must match parent/root currency_id for branch companies; \
                        multi-currency requires distinct ROOT companies (R9)",
            source_method: None,
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L12-MULTICOMPANY-CURRENCY.md",
        l_doc_lines: (51, 81),
        odoo_source: &[OdooSourceRef {
            path: "base/models/res_company.py",
            line_range: (96, 450),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── 4. res.users (company access) ──────────────────────────────────────────
//
// R10: _accessible_branches — the user-facing side.  `allowed_company_ids` is
// the Many2many that gates which companies / branches a user can operate in.
// UserCompanyAccessAdvisor savant uses this field as its primary input.

/// `res.users` — multi-company access control (company + branch membership).
///
/// L-doc R10; source: `base/models/res_company.py:L429-450`.
///
/// Drives `UserCompanyAccessAdvisor` savant (family=0x80).
pub const RES_USERS_COMPANY_ACCESS: OdooEntity = OdooEntity {
    model_name: "res.users",
    kind: OdooEntityKind::Model,
    description: "User model: company_id (active company) and allowed_company_ids \
                  (all companies/branches the user may switch to). \
                  UserCompanyAccessAdvisor uses allowed_company_ids to scope branch \
                  access by user role in multi-branch context (R10).",
    fields: &[
        OdooField {
            // The currently active company for this user session.
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            // All companies/branches the user is allowed to access.
            // UserCompanyAccessAdvisor primary input (R10).
            name: "allowed_company_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("res.company"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Domain,
        condition: "company_id must be in allowed_company_ids; a user's active company \
                    must be one they are permitted to access",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L12-MULTICOMPANY-CURRENCY.md",
        l_doc_lines: (54, 56),
        odoo_source: &[OdooSourceRef {
            path: "base/models/res_company.py",
            line_range: (429, 450),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── 5. account.account (exchange gain/loss target) ──────────────────────────
//
// R18: ExchangeAccountSelector savant boundary.
//   company.income_currency_exchange_account_id (gain, sign < 0)
//   company.expense_currency_exchange_account_id (loss, sign > 0)
// Both point to account.account records.  The selection is DETERMINISTIC
// (sign from compare_amounts(open_balance, 0)); the savant is only heuristic
// for initial SKR chart-of-accounts configuration.
// Posted via currency_exchange_journal_id on the company.
//
// This entity is projected here as the TARGET of the company pointers.
// account.account core fields (code, name, account_type) are not repeated in
// full — only the role relevant to L12 (FX exchange account identity).

/// `account.account` — gain/loss exchange account role for ExchangeAccountSelector.
///
/// L-doc R18; source: `account/models/account_move.py:L5218-5237`,
/// `base/models/res_company.py:L135-145`.
///
/// Only the FX-exchange-account facet is projected here; full account.account
/// definition lives in L1 (chart-of-accounts lane).  These accounts are selected
/// deterministically by sign: balance > 0 → expense (loss); balance < 0 → income (gain).
pub const ACCOUNT_ACCOUNT_EXCHANGE: OdooEntity = OdooEntity {
    model_name: "account.account",
    kind: OdooEntityKind::Model,
    description: "General ledger account in its FX exchange gain/loss role: \
                  referenced by res.company.income_currency_exchange_account_id (gain) and \
                  expense_currency_exchange_account_id (loss). ExchangeAccountSelector savant \
                  heuristic assists initial SKR configuration; runtime selection is deterministic \
                  from compare_amounts(open_balance, 0) sign (R18).",
    fields: &[
        OdooField {
            name: "code",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // E.g. SKR03 2660 (exchange gain) / 2150 (exchange loss).
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
            // income/expense account_type determines gain vs loss routing.
            name: "account_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            // Gain account: 'income_other'; Loss account: 'expense'.
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
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L12-MULTICOMPANY-CURRENCY.md",
        l_doc_lines: (79, 81),
        odoo_source: &[
            OdooSourceRef {
                path: "account/models/account_move.py",
                line_range: (5218, 5237),
            },
            OdooSourceRef {
                path: "base/models/res_company.py",
                line_range: (135, 145),
            },
        ],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// All 5 entities documented in lane L12 (multi-company tree + currency +
/// FX exchange accounts + user branch access).
///
/// Entity index:
///   [0] `res.currency`     — ISO 4217 currency with rounding + rate management
///   [1] `res.currency.rate`— dated FX rate (root-company scoped)
///   [2] `res.company`      — multi-company tree + currency delegation + FX accounts
///   [3] `res.users`        — allowed_company_ids branch access gate
///   [4] `account.account`  — FX gain/loss account role (ExchangeAccountSelector)
pub const ENTITIES: &[OdooEntity] = &[
    RES_CURRENCY,
    RES_CURRENCY_RATE,
    RES_COMPANY_MULTICOMPANY,
    RES_USERS_COMPANY_ACCESS,
    ACCOUNT_ACCOUNT_EXCHANGE,
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{
        OdooConfidence, OdooConstraintKind, OdooFieldKind, OdooMethodKind, OdooReturnKind,
        OdooSemanticRole,
    };

    #[test]
    fn entities_slice_has_five_entries() {
        assert_eq!(ENTITIES.len(), 5);
    }

    #[test]
    fn res_currency_identity() {
        assert_eq!(RES_CURRENCY.model_name, "res.currency");
        assert_eq!(RES_CURRENCY.provenance.confidence, OdooConfidence::Curated);
        assert_eq!(RES_CURRENCY.provenance.l_doc, "L12-MULTICOMPANY-CURRENCY.md");
        assert!(RES_CURRENCY.state_machine.is_none());
    }

    #[test]
    fn res_currency_has_rounding_policy_field() {
        let f = RES_CURRENCY
            .fields
            .iter()
            .find(|f| f.name == "rounding")
            .expect("rounding field must be present");
        assert_eq!(f.kind, OdooFieldKind::Float);
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn res_currency_has_three_rate_fields() {
        let names: Vec<&str> = RES_CURRENCY.fields.iter().map(|f| f.name).collect();
        assert!(names.contains(&"rate"), "rate field must be present");
        assert!(names.contains(&"company_rate"), "company_rate field must be present");
        assert!(names.contains(&"inverse_company_rate"), "inverse_company_rate must be present");
    }

    #[test]
    fn res_currency_has_compare_amounts_asymmetry_method() {
        let m = RES_CURRENCY
            .methods
            .iter()
            .find(|m| m.name == "compare_amounts")
            .expect("compare_amounts method must be present");
        assert_eq!(m.kind, OdooMethodKind::Helper);
        assert_eq!(m.return_kind, OdooReturnKind::Number);
    }

    #[test]
    fn res_currency_has_rounding_write_protection_constraint() {
        let c = RES_CURRENCY
            .constraints
            .first()
            .expect("rounding write-protection constraint must be present");
        assert_eq!(c.kind, OdooConstraintKind::Python);
        assert_eq!(c.source_method, Some("write"));
    }

    #[test]
    fn res_currency_rate_identity() {
        assert_eq!(RES_CURRENCY_RATE.model_name, "res.currency.rate");
        assert_eq!(RES_CURRENCY_RATE.provenance.confidence, OdooConfidence::Curated);
    }

    #[test]
    fn res_currency_rate_unique_sql_constraint() {
        let c = RES_CURRENCY_RATE
            .constraints
            .iter()
            .find(|c| c.kind == OdooConstraintKind::Sql)
            .expect("UNIQUE SQL constraint must be present");
        assert!(c.condition.contains("currency_id"));
    }

    #[test]
    fn res_currency_rate_name_is_date_field() {
        let f = RES_CURRENCY_RATE
            .fields
            .iter()
            .find(|f| f.name == "name")
            .expect("name (date key) field must be present");
        assert_eq!(f.kind, OdooFieldKind::Date);
        assert_eq!(f.semantic_role, OdooSemanticRole::Date);
    }

    #[test]
    fn res_company_multicompany_has_root_id() {
        assert_eq!(RES_COMPANY_MULTICOMPANY.model_name, "res.company");
        let f = RES_COMPANY_MULTICOMPANY
            .fields
            .iter()
            .find(|f| f.name == "root_id")
            .expect("root_id field must be present");
        assert_eq!(f.kind, OdooFieldKind::Many2one);
        assert_eq!(f.target, Some("res.company"));
    }

    #[test]
    fn res_company_multicompany_has_exchange_account_fields() {
        let names: Vec<&str> = RES_COMPANY_MULTICOMPANY.fields.iter().map(|f| f.name).collect();
        assert!(
            names.contains(&"income_currency_exchange_account_id"),
            "income exchange account must be present"
        );
        assert!(
            names.contains(&"expense_currency_exchange_account_id"),
            "expense exchange account must be present"
        );
        assert!(
            names.contains(&"currency_exchange_journal_id"),
            "exchange journal must be present"
        );
    }

    #[test]
    fn res_company_multicompany_exchange_accounts_target_account_account() {
        for field_name in &[
            "income_currency_exchange_account_id",
            "expense_currency_exchange_account_id",
        ] {
            let f = RES_COMPANY_MULTICOMPANY
                .fields
                .iter()
                .find(|f| f.name == *field_name)
                .unwrap_or_else(|| panic!("{} must be present", field_name));
            assert_eq!(
                f.target,
                Some("account.account"),
                "{} must target account.account (ExchangeAccountSelector)",
                field_name
            );
        }
    }

    #[test]
    fn res_users_company_access_has_allowed_company_ids() {
        assert_eq!(RES_USERS_COMPANY_ACCESS.model_name, "res.users");
        let f = RES_USERS_COMPANY_ACCESS
            .fields
            .iter()
            .find(|f| f.name == "allowed_company_ids")
            .expect("allowed_company_ids must be present");
        assert_eq!(f.kind, OdooFieldKind::Many2many);
        // Policy: governs which branches a user can access (UserCompanyAccessAdvisor)
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn account_account_exchange_has_account_type_policy() {
        assert_eq!(ACCOUNT_ACCOUNT_EXCHANGE.model_name, "account.account");
        let f = ACCOUNT_ACCOUNT_EXCHANGE
            .fields
            .iter()
            .find(|f| f.name == "account_type")
            .expect("account_type field must be present");
        assert_eq!(f.kind, OdooFieldKind::Selection);
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
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
    fn all_entities_reference_l12_l_doc() {
        for e in ENTITIES {
            assert_eq!(
                e.provenance.l_doc,
                "L12-MULTICOMPANY-CURRENCY.md",
                "entity {} must reference L12 l_doc",
                e.model_name
            );
        }
    }
}
