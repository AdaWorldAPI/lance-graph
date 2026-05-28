//! Lane L15 (TAX-REPARTITION) — repartition-line deep-dive, CABA mechanics,
//! tax-group totals, `account.account.tag` sign-convention layer.
//!
//! Source: `.claude/odoo/L15-TAX-REPARTITION.md` (655 lines, 2026-05-28).
//! Savant: `TaxExigibilitySuggestor`.
//!
//! ## L3 / L4 overlap
//!
//! Base entities (`account.tax`, `account.tax.repartition.line`,
//! `account.tax.group`, fiscal positions, `account.account.tag` base fields)
//! live in **L3** (`l3.rs`). L15 contributes **targeted extensions**:
//! - `account.account.tag`: adds `balance_negate` + `report_expression_id` (R12).
//! - `account.tax`: adds `hide_tax_exigibility` field + rounding/totals methods
//!   (`_round_base_lines_tax_details`, `_get_tax_totals_summary`, `_prepare_tax_lines`).
//! No field duplication with L3; CABA routing is on `repartition.line` in L3.
//! `balance_negate` is also referenced from **L4** (move-line tagging).
//!
//! Entities: 2 — `account.account.tag` (L15 ext) + `account.tax` (L15 ext).

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── 1. account.account.tag  (L15 extension: balance_negate sign convention) ─
//
// L3 captures: name, applicability, country_id, active.
// L15 adds:    balance_negate (computed, R12) + report_expression_id (computed).
//
// KEY SEMANTIC (R12, L15:529-543): leading '-' in name (e.g. "-Kz81") →
// balance_negate=True → negate GL balance when summing into USt-VA report box.
// _get_tax_tags_domain strips the '-' for search (L88-96); sign stays in name.

const TAG_EXTENSION_FIELDS: &[OdooField] = &[
    // ── Fields present in L3 (provenance completeness) ───────────────────────
    OdooField {
        name: "name",
        kind: OdooFieldKind::Char,
        target: None,
        required: true,
        computed: None,
        depends: &[],
        // For taxes applicability: verbatim tag name including optional leading '-'
        // (e.g. "-Kz81", "Kz89_BASE"). Sign is structural — strip via lstrip('-')
        // only when building search domains, not when storing.
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
        // DE tags: country_id = Germany. Multi-VAT companies can have tags for
        // foreign registrations too (account_tax.py L5178-5180).
        semantic_role: OdooSemanticRole::Reference,
    },
    // ── Fields NEW vs L3 ──────────────────────────────────────────────────────
    OdooField {
        name: "balance_negate",
        kind: OdooFieldKind::Computed,
        target: None,
        required: false,
        computed: Some("_compute_balance_negate"),
        depends: &["report_expression_id.formula"],
        // True when report expression formula starts with '-'.
        // Determines sign in USt-VA box aggregation. account_account_tag.py L20, L40-48.
        // LOAD-BEARING: all Kz aggregation paths must check this flag.
        semantic_role: OdooSemanticRole::Policy,
    },
    OdooField {
        name: "report_expression_id",
        kind: OdooFieldKind::Computed,
        target: Some("account.report.expression"),
        required: false,
        computed: Some("_compute_report_expression_id"),
        depends: &["name", "country_id"],
        // VAT-return box link. Full account.report is Enterprise; community has
        // the tag→formula join. K8: reuse tag->Kz structure (formula=tag.name.lstrip('-')).
        semantic_role: OdooSemanticRole::Reference,
    },
];

const TAG_EXTENSION_METHODS: &[OdooMethod] = &[
    OdooMethod {
        name: "_compute_balance_negate",
        kind: OdooMethodKind::Compute,
        return_kind: OdooReturnKind::Unit,
        // Sets balance_negate = (formula starts with '-').
        // account_account_tag.py L40-48.
        triggers: &[],
    },
    OdooMethod {
        name: "_compute_report_expression_id",
        kind: OdooMethodKind::Compute,
        return_kind: OdooReturnKind::Unit,
        // Joins via formula match: formula == tag.name or formula == '-' + tag.name.
        // account_account_tag.py L98-108.
        triggers: &[],
    },
    OdooMethod {
        name: "_get_tax_tags_domain",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Dict,
        // Strips leading '-' before building the name-search domain.
        // account_account_tag.py L88-96.
        // IMPORTANT: sign is a tag-name attribute, not a domain filter — callers
        // must read balance_negate separately after finding the tag.
        triggers: &[],
    },
    OdooMethod {
        name: "_get_related_tax_report_expressions",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Recordset,
        // Returns report expressions whose formula matches (with or without '-' prefix).
        // account_account_tag.py L98-108.
        triggers: &[],
    },
];

const TAG_EXTENSION_DECORATORS: &[OdooDecorator] = &[
    OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["report_expression_id.formula"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["name", "country_id"],
    },
];

const TAG_EXTENSION_CONSTRAINTS: &[OdooConstraint] = &[OdooConstraint {
    kind: OdooConstraintKind::Sql,
    condition: "unique(name, applicability, country_id) — no duplicate tag name within the \
                same applicability scope and country. (Mirrors l3.rs constraint.)",
    source_method: None,
}];

/// `account.account.tag` — L15 deep-dive on balance_negate sign convention.
///
/// BASE fields (name, applicability, country_id, active) documented in **L3**
/// (`l3.rs`); here we add `balance_negate` + `report_expression_id` which are
/// the K8 USt-VA aggregation primitives exposed by this lane's R12.
///
/// Also referenced from **L4** (account-move line tagging). Coordinate: the
/// `balance_negate` flag must be applied in both the invoice-posting path (L4)
/// and the VAT-report aggregation path (K8).
pub const ACCOUNT_ACCOUNT_TAG_L15: OdooEntity = OdooEntity {
    model_name: "account.account.tag",
    description: "L15 ext: balance_negate + report_expression_id on account.account.tag. \
                  Leading '-' in name → negate GL balance when summing into USt-VA box. \
                  Full account.report is Enterprise; K8 reuses tag->Kz structure.",
    fields: TAG_EXTENSION_FIELDS,
    methods: TAG_EXTENSION_METHODS,
    decorators: TAG_EXTENSION_DECORATORS,
    state_machine: None,
    constraints: TAG_EXTENSION_CONSTRAINTS,
    provenance: OdooProvenance {
        l_doc: "L15-TAX-REPARTITION.md",
        l_doc_lines: (529, 545),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_account_tag.py",
            line_range: (1, 141),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 2. account.tax  (L15 extension: CABA + rounding + totals) ───────────────
//
// L3 captures: all core fields + compute/propagation/repartition methods.
// L15 adds: hide_tax_exigibility (R11), _round_base_lines_tax_details (R8),
//           _get_tax_totals_summary (R10), _prepare_tax_lines (R13).
// _collect_tax_cash_basis_values lives on account.move (L4080-4147) but is
// part of this CABA semantic cluster; see L15-TAX-REPARTITION.md R11.

const TAX_EXTENSION_FIELDS: &[OdooField] = &[
    // ── Fields from L3 (cross-reference) ─────────────────────────────────────
    OdooField {
        name: "tax_exigibility",
        kind: OdooFieldKind::Selection,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        // 'on_invoice' (Soll): due on invoice posting.
        // 'on_payment' (Ist/CABA): due on payment; routes via transition account until reconciled.
        // CABA routing: _get_aml_target_tax_account on repartition.line (l3.rs).
        semantic_role: OdooSemanticRole::Tax,
    },
    OdooField {
        name: "cash_basis_transition_account_id",
        kind: OdooFieldKind::Many2one,
        target: Some("account.account"),
        required: false,
        computed: None,
        depends: &[],
        // CABA clearing account; reconcile=True required. account_tax.py L247-255.
        semantic_role: OdooSemanticRole::Tax,
    },
    // ── Field NEW vs L3 ───────────────────────────────────────────────────────
    OdooField {
        name: "hide_tax_exigibility",
        kind: OdooFieldKind::Computed,
        target: None,
        required: false,
        computed: Some("_compute_hide_tax_exigibility"),
        depends: &["company_id.tax_exigibility"],
        // True when company.tax_exigibility==False (CABA globally disabled).
        // woa-rs: only offer 'on_payment' when company.tax_exigibility is True.
        // account_tax.py L163; L15 R11 lines 524-525.
        semantic_role: OdooSemanticRole::Policy,
    },
];

const TAX_EXTENSION_METHODS: &[OdooMethod] = &[
    OdooMethod {
        name: "_round_base_lines_tax_details",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Unit,
        // R8: 3-step rounding. (1) raw currency.round(); (2) round_globally delta
        // distribution largest-first; (3) override from user-edited tax lines.
        // company.tax_calculation_rounding_method: 'round_globally' (EU/DE default).
        // account_tax.py L2178-2288.
        triggers: &[],
    },
    OdooMethod {
        name: "_distribute_delta_amount_smoothly",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Unit,
        // Delta → integer error-units at currency precision; distribute proportionally
        // largest-first then 1-unit tail. Guarantees cent-level accuracy.
        // account_tax.py L1836-1888.
        triggers: &[],
    },
    OdooMethod {
        name: "_get_tax_totals_summary",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Dict,
        // R10: invoice/POS footer subtotals. Groups by (group.sequence, group.id);
        // preceding_subtotal → named section; cash-rounding delta + non-deductible
        // lines handled. same_tax_base controls per-group base display.
        // account_tax.py L2709-2989.
        triggers: &[],
    },
    OdooMethod {
        name: "_prepare_tax_lines",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Dict,
        // R13: converts tax_reps_data into account.move.line create/update/delete diff.
        // Accumulates into tax_lines_mapping[grouping_key] × sign.
        // Prunes zero-amount lines unless __keep_zero_line flag. account_tax.py L3032-3126.
        triggers: &[],
    },
    OdooMethod {
        name: "_compute_hide_tax_exigibility",
        kind: OdooMethodKind::Compute,
        return_kind: OdooReturnKind::Unit,
        // Sets hide_tax_exigibility from company.tax_exigibility flag.
        triggers: &[],
    },
];

const TAX_EXTENSION_DECORATORS: &[OdooDecorator] = &[
    OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["company_id.tax_exigibility"],
    },
    OdooDecorator {
        kind: OdooDecoratorKind::ApiConstrains,
        targets: &["tax_exigibility", "cash_basis_transition_account_id"],
    },
];

const TAX_EXTENSION_CONSTRAINTS: &[OdooConstraint] = &[
    OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "If tax_exigibility == 'on_payment', cash_basis_transition_account_id.reconcile \
                    must be True (CABA transition account must allow reconciliation). \
                    account_tax.py L247-255.",
        source_method: Some("_constrains_cash_basis_transition_account"),
    },
    OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "CABA multi-currency restriction: _collect_tax_cash_basis_values returns None \
                    when multiple currencies are involved on the same move. woa-rs must enforce \
                    single-currency constraint for CABA moves at validation time. \
                    account_move.py L4136-4141.",
        source_method: None, // enforced at move level, not on account.tax itself
    },
];

/// `account.tax` — L15 extension: CABA mechanics + rounding engine + totals.
///
/// BASE fields and core computation methods (amount, tax_exigibility,
/// price_include, repartition lines, _get_tax_details, etc.) are documented
/// in **L3** (`l3.rs`). L15 adds:
/// - `hide_tax_exigibility` (company CABA gate, R11)
/// - `_round_base_lines_tax_details` (R8 — round_globally / round_per_line)
/// - `_get_tax_totals_summary` (R10 — invoice footer subtotals + Kennziffer display)
/// - `_prepare_tax_lines` (R13 — GL line diff from repartition data)
/// - `_distribute_delta_amount_smoothly` (R8 helper — cent-accurate rounding distribution)
///
/// CABA routing (`_get_aml_target_tax_account`) is on `account.tax.repartition.line`
/// in **L3**. The reconciliation side (`_collect_tax_cash_basis_values`) lives on
/// `account.move` (account_move.py L4080-4147) but is documented in L15 R11 as
/// part of the CABA semantic cluster.
pub const ACCOUNT_TAX_L15: OdooEntity = OdooEntity {
    model_name: "account.tax",
    description: "L15 ext: CABA gate + rounding engine + totals. Core in L3. Adds \
                  hide_tax_exigibility, _round_base_lines_tax_details (round_globally/per_line), \
                  _get_tax_totals_summary (footer subtotals), _prepare_tax_lines (GL diff). \
                  TaxExigibilitySuggestor: gate on_payment via company.tax_exigibility.",
    fields: TAX_EXTENSION_FIELDS,
    methods: TAX_EXTENSION_METHODS,
    decorators: TAX_EXTENSION_DECORATORS,
    state_machine: None,
    constraints: TAX_EXTENSION_CONSTRAINTS,
    provenance: OdooProvenance {
        l_doc: "L15-TAX-REPARTITION.md",
        l_doc_lines: (64, 614),
        odoo_source: &[
            OdooSourceRef {
                path: "addons/account/models/account_tax.py",
                line_range: (163, 174),  // hide_tax_exigibility + exigibility fields
            },
            OdooSourceRef {
                path: "addons/account/models/account_tax.py",
                line_range: (1836, 2288), // _distribute_delta + _round_base_lines_tax_details
            },
            OdooSourceRef {
                path: "addons/account/models/account_tax.py",
                line_range: (2709, 2989), // _get_tax_totals_summary
            },
            OdooSourceRef {
                path: "addons/account/models/account_tax.py",
                line_range: (3032, 3126), // _prepare_tax_lines
            },
            OdooSourceRef {
                path: "addons/account/models/account_move.py",
                line_range: (4080, 4147), // _collect_tax_cash_basis_values (CABA reconciliation)
            },
        ],
        confidence: OdooConfidence::Curated,
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// Entities in lane L15 (targeted extensions on top of L3 base entities).
/// Skipped (owned by L3): `account.tax.repartition.line`, `account.tax.group`,
/// `account.fiscal.position`, `account.fiscal.position.account`,
/// `account.account.tag` base fields.
pub const ENTITIES: &[OdooEntity] = &[ACCOUNT_ACCOUNT_TAG_L15, ACCOUNT_TAX_L15];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooFieldKind, OdooSemanticRole};

    #[test]
    fn entity_count() {
        assert_eq!(ENTITIES.len(), 2);
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
    fn all_entities_reference_l15_doc() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.l_doc,
                "L15-TAX-REPARTITION.md",
                "{} references wrong L-doc",
                entity.model_name
            );
            let (start, end) = entity.provenance.l_doc_lines;
            assert!(start > 0, "{} has zero start line", entity.model_name);
            assert!(end >= start, "{} has end < start", entity.model_name);
            assert!(end <= 655, "{} references line beyond doc length", entity.model_name);
        }
    }

    #[test]
    fn tag_extension_has_balance_negate() {
        let tag = ENTITIES
            .iter()
            .find(|e| e.model_name == "account.account.tag")
            .expect("account.account.tag must be present in L15 ENTITIES");
        let bn = tag
            .fields
            .iter()
            .find(|f| f.name == "balance_negate")
            .expect("balance_negate field must exist on account.account.tag in L15");
        assert_eq!(bn.kind, OdooFieldKind::Computed);
        assert_eq!(bn.semantic_role, OdooSemanticRole::Policy);
        assert_eq!(bn.computed, Some("_compute_balance_negate"));
    }

    #[test]
    fn tag_extension_has_report_expression_id() {
        let tag = ENTITIES
            .iter()
            .find(|e| e.model_name == "account.account.tag")
            .unwrap();
        let rep_expr = tag
            .fields
            .iter()
            .find(|f| f.name == "report_expression_id")
            .expect("report_expression_id must exist on account.account.tag in L15");
        assert_eq!(rep_expr.kind, OdooFieldKind::Computed);
        assert_eq!(rep_expr.target, Some("account.report.expression"));
    }

    #[test]
    fn tax_extension_has_hide_tax_exigibility() {
        let tax = ENTITIES
            .iter()
            .find(|e| e.model_name == "account.tax")
            .expect("account.tax must be present in L15 ENTITIES");
        let hide = tax
            .fields
            .iter()
            .find(|f| f.name == "hide_tax_exigibility")
            .expect("hide_tax_exigibility field must exist on account.tax in L15");
        assert_eq!(hide.kind, OdooFieldKind::Computed);
        assert_eq!(hide.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn tax_extension_has_rounding_and_totals_methods() {
        let tax = ENTITIES
            .iter()
            .find(|e| e.model_name == "account.tax")
            .unwrap();
        let method_names: Vec<&str> = tax.methods.iter().map(|m| m.name).collect();
        assert!(
            method_names.contains(&"_round_base_lines_tax_details"),
            "rounding engine method missing"
        );
        assert!(
            method_names.contains(&"_get_tax_totals_summary"),
            "totals summary method missing"
        );
        assert!(
            method_names.contains(&"_prepare_tax_lines"),
            "prepare_tax_lines method missing"
        );
    }

    #[test]
    fn tax_extension_caba_constraint_present() {
        let tax = ENTITIES
            .iter()
            .find(|e| e.model_name == "account.tax")
            .unwrap();
        let caba_constraint = tax
            .constraints
            .iter()
            .find(|c| c.source_method == Some("_constrains_cash_basis_transition_account"))
            .expect("CABA reconcile constraint must be declared in L15");
        assert_eq!(caba_constraint.kind, OdooConstraintKind::Python);
    }

    #[test]
    fn tag_sign_domain_method_present() {
        let tag = ENTITIES
            .iter()
            .find(|e| e.model_name == "account.account.tag")
            .unwrap();
        assert!(
            tag.methods
                .iter()
                .any(|m| m.name == "_get_tax_tags_domain"),
            "_get_tax_tags_domain must be declared (strips leading '-' from tag name)"
        );
    }
}
