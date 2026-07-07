//! Lane L10 (ANALYTIC) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L10-ANALYTIC.md`.
//!
//! Entities: `account.analytic.plan`, `account.analytic.account`,
//! `account.analytic.applicability`, `account.analytic.line`,
//! `account.analytic.distribution.model`.
//!
//! NOTE: The base `analytic` addon is absent from the source clone.  All
//! entity shapes are inferred from the account-side `_inherit` extensions
//! and test usage (L20-30 of L10-ANALYTIC.md).  Confidence: Curated.

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooEntityKind, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind,
    OdooProvenance, OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── account.analytic.plan ────────────────────────────────────────────────────
//
// Plan hierarchy: each plan is one axis of cost-centre attribution (e.g.
// "Department", "Project").  `_column_name()` returns a dynamic DB column name
// (e.g. `x_plan1_id`) used on `account.analytic.line` for that plan axis.
// Documented in R2 (L82-119) and R9 (L423-453).

pub const ANALYTIC_PLAN: OdooEntity = OdooEntity {
    model_name: "account.analytic.plan",
    kind: OdooEntityKind::Model,
    description: "Analytic plan hierarchy: one orthogonal cost-attribution axis (e.g. \
                  Department, Project). `_column_name()` drives dynamic DB columns on \
                  `account.analytic.line`; `_get_applicability()` resolves mandatory / \
                  optional / unavailable for a given (domain, product, account) context.",
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
        // Self-referencing parent for nested plan trees.
        OdooField {
            name: "parent_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.analytic.plan"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "children_ids",
            kind: OdooFieldKind::One2many,
            target: Some("account.analytic.plan"),
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
            name: "applicability_ids",
            kind: OdooFieldKind::One2many,
            target: Some("account.analytic.applicability"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[
        // Returns dynamic DB column name e.g. "x_plan1_id" (ORM metaprogramming).
        // In Rust, store column_name explicitly on the plan row instead.
        OdooMethod {
            name: "_column_name",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // Returns highest-scoring `account.analytic.applicability` for the given context.
        OdooMethod {
            name: "_get_applicability",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L10-ANALYTIC.md",
        l_doc_lines: (82, 119),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_analytic_plan.py",
            line_range: (1, 82),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── account.analytic.account ─────────────────────────────────────────────────
//
// Leaf cost-centre.  Belongs to exactly one `plan_id`; `root_plan_id` always
// points to the tree root.  UI smart-buttons for invoice / vendor-bill count
// use JSON-contains domain on `analytic_distribution`.  Documented in R2
// (L82-119), R13 (L508-513), and R12 (L486-505).

pub const ANALYTIC_ACCOUNT: OdooEntity = OdooEntity {
    model_name: "account.analytic.account",
    kind: OdooEntityKind::Model,
    description: "Leaf analytic cost-centre.  Belongs to one analytic plan axis; \
                  `root_plan_id` anchors the plan tree root used in amount allocation \
                  accumulation.  `active=False` blocks posting (R12 archived-account guard).",
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
            name: "plan_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.analytic.plan"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Always points to the root of the plan hierarchy tree.
        OdooField {
            name: "root_plan_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.analytic.plan"),
            required: false,
            computed: Some("_compute_root_plan_id"),
            depends: &["plan_id.parent_id"],
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
        // Archived accounts are hard-blocked from posting (UserError at validation).
        OdooField {
            name: "active",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "line_ids",
            kind: OdooFieldKind::One2many,
            target: Some("account.analytic.line"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // UI smart-button: count of customer invoices referencing this account via
        // JSON-contains domain on account.move.line.analytic_distribution.
        OdooField {
            name: "invoice_count",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: Some("_compute_invoice_count"),
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
        },
        // UI smart-button: count of vendor bills referencing this account.
        OdooField {
            name: "vendor_bill_count",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: Some("_compute_vendor_bill_count"),
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_root_plan_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // Uses _read_group with JSON-contains domain on analytic_distribution.
        OdooMethod {
            name: "_compute_invoice_count",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_vendor_bill_count",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["plan_id.parent_id"],
    }],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "Archived (active=False) analytic account cannot appear in any \
                    analytic_distribution when posting (UserError: 'archived analytic account')",
        source_method: None, // guard in analytic.mixin._validate_distribution (base absent)
    }],
    provenance: OdooProvenance {
        l_doc: "L10-ANALYTIC.md",
        l_doc_lines: (82, 119),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_analytic_account.py",
            line_range: (1, 79),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── account.analytic.applicability ──────────────────────────────────────────
//
// Rule that maps (business_domain, product_categ, account_prefix, company) to
// an applicability value: mandatory / optional / unavailable.  Scoring via
// `_get_score()` selects the most-specific matching rule.  Documented in R9
// (L423-453).

pub const ANALYTIC_APPLICABILITY: OdooEntity = OdooEntity {
    model_name: "account.analytic.applicability",
    kind: OdooEntityKind::Model,
    description: "Policy rule: for a given (business_domain, product_category, \
                  account_prefix, company) context, declares whether an analytic plan \
                  axis is mandatory / optional / unavailable.  `_get_score()` ranks \
                  rules by specificity; highest score wins.",
    fields: &[
        OdooField {
            name: "analytic_plan_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.analytic.plan"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // 'mandatory' | 'optional' | 'unavailable'
        OdooField {
            name: "applicability",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Extended by account/purchase/sale: 'invoice'|'bill'|'purchase_order'|'sale_order'|'general'
        OdooField {
            name: "business_domain",
            kind: OdooFieldKind::Selection,
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
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Semicolon/comma-separated account code prefixes; match via startswith.
        // Shown only when business_domain in ('general','invoice','bill').
        OdooField {
            name: "account_prefix",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "product_categ_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.category"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[
        // Base score (business_domain + company) extended by account-side to add
        // +1 for account_prefix match, +1 for product_categ match.
        // Returns -1 on hard-mismatch (exclusion).
        OdooMethod {
            name: "_get_score",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L10-ANALYTIC.md",
        l_doc_lines: (423, 453),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_analytic_plan.py",
            line_range: (59, 82),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── account.analytic.line ────────────────────────────────────────────────────
//
// Persisted analytic posting created by `_create_analytic_lines()` when a move
// is posted.  Amount = -balance (company currency, sign-flipped vs AML).
// Bidirectional sync with `account.move.line.analytic_distribution` guarded by
// `skip_analytic_sync` context.  Documented in R3 (L122-181), R6 (L260-294),
// R10 (L456-469), R11 (L471-483).

pub const ANALYTIC_LINE: OdooEntity = OdooEntity {
    model_name: "account.analytic.line",
    kind: OdooEntityKind::Model,
    description: "Analytic posting record created at journal-entry post time. \
                  Amount = -aml.balance (company currency). Bidirectional sync with \
                  `account.move.line.analytic_distribution` via `_update_analytic_distribution` \
                  / `_inverse_analytic_distribution` loop guarded by `skip_analytic_sync` context.",
    fields: &[
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Document,
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
        // In company currency; sign = -aml.balance (credit → positive, debit → negative).
        OdooField {
            name: "amount",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        // Physical quantity (hours, units); used for timesheet-style entries (R10).
        OdooField {
            name: "unit_amount",
            kind: OdooFieldKind::Float,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Quantity,
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
            name: "product_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.product"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
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
        // The GL account this analytic line backs.
        OdooField {
            name: "general_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Link back to the parent journal line (None for pure analytic entries).
        OdooField {
            name: "move_line_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.move.line"),
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
        // 'invoice' | 'vendor_bill' | 'other' — set by _prepare_analytic_distribution_line.
        OdooField {
            name: "category",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Status,
        },
    ],
    methods: &[
        // Onchange: computes amount = -standard_price * unit_amount; sets general_account_id.
        // Used for timesheet-style manual analytic entries (R10, L456-469).
        OdooMethod {
            name: "on_change_unit_amount",
            kind: OdooMethodKind::Onchange,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // create/write/unlink all call move_line_id._update_analytic_distribution()
        // after mutation, guarded by skip_analytic_sync context (R11, L471-483).
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
        OdooMethod {
            name: "unlink",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        // Returns CSV of all plan account IDs for this line (base mixin, absent).
        // Used as key in _update_analytic_distribution reverse-sync formula (R6).
        OdooMethod {
            name: "_get_distribution_key",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiOnchange,
            targets: &["product_id", "product_uom_id", "unit_amount", "currency_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModelCreateMulti,
            targets: &[],
        },
    ],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L10-ANALYTIC.md",
        l_doc_lines: (456, 483),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_analytic_line.py",
            line_range: (1, 111),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── account.analytic.distribution.model ─────────────────────────────────────
//
// Pattern-matching rule: given (partner, product_categ, account_prefix, company)
// criteria, proposes an `analytic_distribution` JSON.  Multi-criterion scoring
// (+1 per matched field, -1 on hard mismatch) → greedy plan-by-plan fill from
// highest-score model (R8, L368-420).  Drives savants:
//   - AnalyticDistributionSuggester (R7 AXIS-B: NextBestAction / NarsTruth)
//   - AnalyticModelScorer            (R8 AXIS-B: CustomerCategory / HammingMin)

pub const ANALYTIC_DISTRIBUTION_MODEL: OdooEntity = OdooEntity {
    model_name: "account.analytic.distribution.model",
    kind: OdooEntityKind::Model,
    description: "Distribution rule record: maps (partner, product_category, \
                  account_prefix, company) to a proposed `analytic_distribution` JSON. \
                  `_get_applicable_models()` filters by account_prefix startswith; \
                  `_get_distribution()` scores and greedily fills plans in sequence order. \
                  Drives AnalyticDistributionSuggester (AXIS-B) and AnalyticModelScorer.",
    fields: &[
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
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Lower sequence = higher priority in greedy plan-fill (R8).
        OdooField {
            name: "sequence",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // JSON: {"<account_id_csv>": <percentage_float>}.
        // Keys are comma-separated analytic account ID strings; values are float %.
        // Percentages per analytic plan must sum to 100 for mandatory plans (R1, R4).
        // Stored as fields.Json (Char in Odoo JSON column) — semantic_role: Policy
        // because it is the decision payload matched against invoice line context.
        OdooField {
            name: "analytic_distribution",
            kind: OdooFieldKind::Char, // fields.Json — no Json variant; Char is nearest
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Semicolon/comma-separated account code prefixes; Python startswith filter
        // (not SQL domain — `_create_domain` returns [] for this field).
        OdooField {
            name: "account_prefix",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "product_categ_id",
            kind: OdooFieldKind::Many2one,
            target: Some("product.category"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[
        // Returns {'product_id': False, 'product_categ_id': False, ...} — False means
        // "match models with no constraint on this field".
        OdooMethod {
            name: "_get_default_search_domain_vals",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // Filters candidates by account_prefix startswith; extends base domain filter.
        // Called from base `_get_distribution()` (base module absent).
        OdooMethod {
            name: "_get_applicable_models",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
        },
        // Scoring: +1 per matched criterion (partner/product_categ/account_prefix/company);
        // -1 on hard mismatch. AXIS-B delegation: AnalyticModelScorer.
        OdooMethod {
            name: "_get_score",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Number,
            triggers: &[],
        },
        // Base method (absent): scores all applicable models, greedy-fills plans by
        // sequence order (lower wins), skips models that would overwrite covered plans.
        // AXIS-B delegation: AnalyticDistributionSuggester.
        OdooMethod {
            name: "_get_distribution",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // Returns [] for account_prefix (Python filter, not SQL); delegates to base for others.
        OdooMethod {
            name: "_create_domain",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiModel,
        targets: &[],
    }],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L10-ANALYTIC.md",
        l_doc_lines: (368, 420),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_analytic_distribution_model.py",
            line_range: (1, 71),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// Entities documented in lane L10 (analytic accounts + analytic lines +
/// analytic distribution + distribution model scoring).
///
/// Excluded by design:
/// - `analytic.mixin` — a mixin, not a stored model; its methods
///   (`_merge_distribution`, `_validate_distribution`, `_get_distribution_key`,
///   `_get_plan_fnames`) are attributed to the consuming entities above.
/// - `account.move.line` analytic fields — documented in L1 (`account.move.line`);
///   L10 captures only the pure-analytic models.
pub const ENTITIES: &[OdooEntity] = &[
    ANALYTIC_PLAN,
    ANALYTIC_ACCOUNT,
    ANALYTIC_APPLICABILITY,
    ANALYTIC_LINE,
    ANALYTIC_DISTRIBUTION_MODEL,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooFieldKind, OdooSemanticRole};

    #[test]
    fn l10_entities_non_empty() {
        assert_eq!(ENTITIES.len(), 5);
    }

    #[test]
    fn analytic_distribution_model_policy_fields() {
        let model = &ANALYTIC_DISTRIBUTION_MODEL;
        assert_eq!(model.model_name, "account.analytic.distribution.model");
        // analytic_distribution is Char (nearest to fields.Json) with Policy role
        let dist = model.fields.iter().find(|f| f.name == "analytic_distribution").unwrap();
        assert_eq!(dist.kind, OdooFieldKind::Char);
        assert_eq!(dist.semantic_role, OdooSemanticRole::Policy);
        // account_prefix is Policy (pattern-matching criterion)
        let prefix = model.fields.iter().find(|f| f.name == "account_prefix").unwrap();
        assert_eq!(prefix.semantic_role, OdooSemanticRole::Policy);
        // _get_distribution is the AXIS-B delegation point
        let get_dist = model.methods.iter().find(|m| m.name == "_get_distribution").unwrap();
        assert_eq!(get_dist.kind, OdooMethodKind::ApiModel);
    }

    #[test]
    fn analytic_account_active_field_status() {
        let account = &ANALYTIC_ACCOUNT;
        assert_eq!(account.model_name, "account.analytic.account");
        let active = account.fields.iter().find(|f| f.name == "active").unwrap();
        assert!(active.required);
        assert_eq!(active.semantic_role, OdooSemanticRole::Status);
        assert_eq!(account.provenance.confidence, OdooConfidence::Curated);
    }

    #[test]
    fn analytic_applicability_scoring_method() {
        let app = &ANALYTIC_APPLICABILITY;
        assert_eq!(app.model_name, "account.analytic.applicability");
        let score = app.methods.iter().find(|m| m.name == "_get_score").unwrap();
        assert_eq!(score.return_kind, OdooReturnKind::Number);
    }

    #[test]
    fn analytic_line_bidirectional_sync_methods() {
        let line = &ANALYTIC_LINE;
        assert_eq!(line.model_name, "account.analytic.line");
        // create/write/unlink all present for bidirectional sync
        assert!(line.methods.iter().any(|m| m.name == "create"));
        assert!(line.methods.iter().any(|m| m.name == "write"));
        assert!(line.methods.iter().any(|m| m.name == "unlink"));
        assert!(line.methods.iter().any(|m| m.name == "_get_distribution_key"));
    }

    #[test]
    fn l10_all_curated() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.confidence,
                OdooConfidence::Curated,
                "{} should be Curated",
                entity.model_name,
            );
            assert_eq!(entity.provenance.l_doc, "L10-ANALYTIC.md",
                "{} must reference L10-ANALYTIC.md", entity.model_name);
        }
    }
}
