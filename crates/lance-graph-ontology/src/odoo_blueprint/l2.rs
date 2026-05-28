//! Lane L2 (K3-RECON) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L2-K3-RECON.md`.
//!
//! Entities: `account.move.line` (reconciliation fields), `account.partial.reconcile`,
//! `account.full.reconcile`.
//!
//! `account.reconcile.model` is Axis-2 heuristic territory — deferred to lane L5
//! per FLAG-4 in the L2 doc.  `account.bank.statement` / `account.bank.statement.line`
//! are NOT documented in L2 (only referenced); they belong in a bank-statement lane.

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooSourceRef, OdooState, OdooStateMachine,
    OdooStateSemantic, OdooTransition,
};

// ─── account.move.line (reconciliation fields) ───────────────────────────────
//
// L2 documents the reconciliation region of AML: residual computes (R-1),
// reconcile() / remove_move_reconcile() entry points (R-2/R-3), and the full
// orchestration engine (R-4 through R-8 / R-12 / R-14).  Only reconciliation-
// relevant fields are captured here; the invoice/payment fields live in their
// own lane.

const ACCOUNT_MOVE_LINE_SM: OdooStateMachine = OdooStateMachine {
    state_field: "parent_state",
    states: &[
        OdooState { name: "draft", semantic: OdooStateSemantic::Draft },
        OdooState { name: "posted", semantic: OdooStateSemantic::Posted },
        OdooState { name: "cancel", semantic: OdooStateSemantic::Cancelled },
    ],
    transitions: &[OdooTransition {
        from: "draft",
        to: "posted",
        trigger: "action_post",
        guards: &["_check_balanced"],
    }],
};

pub const ACCOUNT_MOVE_LINE: OdooEntity = OdooEntity {
    model_name: "account.move.line",
    description: "Journal entry line — the debit/credit leaf of a double-entry move; \
                  carries residual amounts for open-item matching (K3 scope: reconciliation \
                  region only).",
    fields: &[
        OdooField {
            name: "amount_residual",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_compute_amount_residual"),
            depends: &[
                "debit",
                "credit",
                "amount_currency",
                "account_id",
                "currency_id",
                "company_id",
                "matched_debit_ids",
                "matched_credit_ids",
            ],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "amount_residual_currency",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_compute_amount_residual"),
            depends: &[
                "debit",
                "credit",
                "amount_currency",
                "account_id",
                "currency_id",
                "company_id",
                "matched_debit_ids",
                "matched_credit_ids",
            ],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "reconciled",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: Some("_compute_amount_residual"),
            depends: &[
                "debit",
                "credit",
                "amount_currency",
                "account_id",
                "currency_id",
                "company_id",
                "matched_debit_ids",
                "matched_credit_ids",
            ],
            semantic_role: OdooSemanticRole::Status,
        },
        OdooField {
            name: "full_reconcile_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.full.reconcile"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "matched_debit_ids",
            kind: OdooFieldKind::One2many,
            target: Some("account.partial.reconcile"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "matched_credit_ids",
            kind: OdooFieldKind::One2many,
            target: Some("account.partial.reconcile"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "matching_number",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None, // written by _update_matching_number SQL, not a compute field
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_amount_residual",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "reconcile",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "remove_move_reconcile",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_reconcile_plan",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_reconcile_plan_with_sync",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_optimize_reconciliation_plan",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_prepare_reconciliation_amls",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_prepare_reconciliation_single_partial",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_reconcile_pre_hook",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_reconcile_post_hook",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_prepare_exchange_difference_move_vals",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_create_exchange_difference_moves",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
        },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &[
            "debit",
            "credit",
            "amount_currency",
            "account_id",
            "currency_id",
            "company_id",
            "matched_debit_ids",
            "matched_credit_ids",
        ],
    }],
    state_machine: Some(&ACCOUNT_MOVE_LINE_SM),
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "reconciled = True only when BOTH amount_residual and \
                    amount_residual_currency are zero (multi-currency correctness).",
        source_method: Some("_compute_amount_residual"),
    }],
    provenance: OdooProvenance {
        l_doc: "L2-K3-RECON.md",
        l_doc_lines: (27, 466),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_move_line.py",
            line_range: (241, 295),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── account.partial.reconcile ───────────────────────────────────────────────
//
// Documented in R-9 (L468-528) and R-11 matching-number protocol (L569-617).
// Verified against /home/user/odoo/addons/account/models/account_partial_reconcile.py
// lines 9–99 + 148–153.

pub const ACCOUNT_PARTIAL_RECONCILE: OdooEntity = OdooEntity {
    model_name: "account.partial.reconcile",
    description: "Partial settlement event — one debit/credit AML pair matched for a \
                  given amount in company and foreign currency.  Cascade-unlinking this \
                  record reverses exchange-diff and CABA moves, clears the full reconcile, \
                  and resets matching numbers.",
    fields: &[
        OdooField {
            name: "debit_move_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.move.line"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "credit_move_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.move.line"),
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "full_reconcile_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.full.reconcile"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "exchange_move_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.move"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
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
            name: "debit_amount_currency",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "credit_amount_currency",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        },
        OdooField {
            name: "max_date",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: Some("_compute_max_date"),
            depends: &["debit_move_id.date", "credit_move_id.date"],
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "company_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.company"),
            required: false,
            computed: Some("_compute_company_id"),
            depends: &["debit_move_id", "credit_move_id"],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "debit_currency_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.currency"),
            required: false,
            computed: None, // related + stored
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "credit_currency_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.currency"),
            required: false,
            computed: None, // related + stored
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "draft_caba_move_vals",
            kind: OdooFieldKind::Text, // TODO: OdooFieldKind has no Json variant; Text is nearest
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Audit,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_max_date",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_company_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_required_computed_currencies",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "create",
            kind: OdooMethodKind::ApiModelCreateMulti,
            return_kind: OdooReturnKind::Recordset,
            triggers: &["in_process→paid"],
        },
        OdooMethod {
            name: "unlink",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &["paid→in_process"],
        },
        OdooMethod {
            name: "_update_matching_number",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_collect_tax_cash_basis_values",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_create_tax_cash_basis_moves",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Recordset,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["debit_currency_id", "credit_currency_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["debit_move_id.date", "credit_move_id.date"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["debit_move_id", "credit_move_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModelCreateMulti,
            targets: &[],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "Both debit_currency_id and credit_currency_id must be set on every partial.",
        source_method: Some("_check_required_computed_currencies"),
    }],
    provenance: OdooProvenance {
        l_doc: "L2-K3-RECON.md",
        l_doc_lines: (468, 617),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_partial_reconcile.py",
            line_range: (9, 215),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── account.full.reconcile ──────────────────────────────────────────────────
//
// Documented in R-10 (L530-567).
// Verified against /home/user/odoo/addons/account/models/account_full_reconcile.py
// lines 1-45.

pub const ACCOUNT_FULL_RECONCILE: OdooEntity = OdooEntity {
    model_name: "account.full.reconcile",
    description: "Completed settlement event — groups all partials and AMLs that together \
                  achieve zero residuals; sets matching_number to a plain integer string \
                  (no 'P' prefix) on all participating lines.",
    fields: &[
        OdooField {
            name: "partial_reconcile_ids",
            kind: OdooFieldKind::One2many,
            target: Some("account.partial.reconcile"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "reconciled_line_ids",
            kind: OdooFieldKind::One2many,
            target: Some("account.move.line"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[OdooMethod {
        name: "create",
        kind: OdooMethodKind::ApiModelCreateMulti,
        return_kind: OdooReturnKind::Recordset,
        triggers: &[],
    }],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiModelCreateMulti,
        targets: &[],
    }],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L2-K3-RECON.md",
        l_doc_lines: (530, 567),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_full_reconcile.py",
            line_range: (1, 45),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// Entities documented in lane L2 (K3-RECON: double-entry reconciliation,
/// open-item matching, partial/full reconcile records).
///
/// Excluded by design:
/// - `account.reconcile.model` — Axis-2 heuristic matching; deferred to lane L5
///   (FLAG-4 in L2 doc, lines 871-873).
/// - `account.bank.statement` / `account.bank.statement.line` — not documented
///   in the L2 prose; belong in a dedicated bank-statement lane.
pub const ENTITIES: &[OdooEntity] =
    &[ACCOUNT_MOVE_LINE, ACCOUNT_PARTIAL_RECONCILE, ACCOUNT_FULL_RECONCILE];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooFieldKind};

    #[test]
    fn l2_entities_non_empty() {
        assert_eq!(ENTITIES.len(), 3);
    }

    #[test]
    fn account_move_line_residual_fields_are_computed() {
        let aml = &ACCOUNT_MOVE_LINE;
        assert_eq!(aml.model_name, "account.move.line");
        let residual = aml.fields.iter().find(|f| f.name == "amount_residual").unwrap();
        assert_eq!(residual.computed, Some("_compute_amount_residual"));
        assert!(residual.depends.contains(&"matched_debit_ids"));
        assert_eq!(residual.kind, OdooFieldKind::Monetary);
        assert_eq!(aml.provenance.confidence, OdooConfidence::Curated);
    }

    #[test]
    fn account_partial_reconcile_required_fields() {
        let pr = &ACCOUNT_PARTIAL_RECONCILE;
        assert_eq!(pr.model_name, "account.partial.reconcile");
        let debit = pr.fields.iter().find(|f| f.name == "debit_move_id").unwrap();
        assert!(debit.required);
        let credit = pr.fields.iter().find(|f| f.name == "credit_move_id").unwrap();
        assert!(credit.required);
    }

    #[test]
    fn account_full_reconcile_has_two_fields() {
        let fr = &ACCOUNT_FULL_RECONCILE;
        assert_eq!(fr.model_name, "account.full.reconcile");
        assert_eq!(fr.fields.len(), 2);
        assert!(fr.state_machine.is_none());
        assert_eq!(fr.provenance.l_doc, "L2-K3-RECON.md");
    }

    #[test]
    fn l2_provenance_confidence_all_curated() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.confidence,
                OdooConfidence::Curated,
                "{} should be Curated",
                entity.model_name
            );
            assert!(!entity.provenance.odoo_source.is_empty(),
                "{} must have at least one odoo_source ref", entity.model_name);
        }
    }
}
