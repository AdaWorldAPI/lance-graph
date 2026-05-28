//! Lane L1 (K3-POST) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L1-K3-POST.md`.
//!
//! Entities: `account.move` (head record + state machine + hash chain),
//! `account.move.line` (line amounts + constraints), `account.journal`.

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooState, OdooStateMachine, OdooStateSemantic,
    OdooTransition,
};

// ─── Shared state machine ─────────────────────────────────────────────────────

const MOVE_SM: OdooStateMachine = OdooStateMachine {
    state_field: "state",
    states: &[
        OdooState { name: "draft",  semantic: OdooStateSemantic::Draft },
        OdooState { name: "posted", semantic: OdooStateSemantic::Posted },
        OdooState { name: "cancel", semantic: OdooStateSemantic::Cancelled },
    ],
    transitions: &[
        OdooTransition {
            from: "draft", to: "posted", trigger: "action_post",
            guards: &["_check_balanced", "_check_lock_date", "_check_draftable"],
        },
        OdooTransition {
            from: "posted", to: "draft", trigger: "button_draft",
            guards: &["_check_draftable"],
        },
        OdooTransition {
            from: "posted", to: "cancel", trigger: "button_cancel",
            guards: &["_check_draftable"],
        },
        OdooTransition {
            from: "draft", to: "cancel", trigger: "button_cancel",
            guards: &[],
        },
        // GoBD Storno: posted → new posted reversal; original row untouched.
        OdooTransition {
            from: "posted", to: "posted", trigger: "_reverse_moves",
            guards: &["_can_be_unlinked"],
        },
    ],
};

// ─── account.move ─────────────────────────────────────────────────────────────

pub const ACCOUNT_MOVE: OdooEntity = OdooEntity {
    model_name: "account.move",
    description: "Double-entry journal entry / invoice; draft→posted→cancel state machine, \
                  Belegnummer sequence, GoBD K11 inalterability hash chain, storno logic.",
    fields: &[
        OdooField { name: "state",
            kind: OdooFieldKind::Selection, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
        // Belegnummer; computed by _compute_name; frozen once posted_before=True.
        OdooField { name: "name",
            kind: OdooFieldKind::Char, target: None, required: false,
            computed: Some("_compute_name"),
            depends: &["state", "journal_id", "date", "posted_before"],
            semantic_role: OdooSemanticRole::Identity },
        // Set True by _post(); never reset — freezes Belegnummer permanently.
        OdooField { name: "posted_before",
            kind: OdooFieldKind::Boolean, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Audit },
        OdooField { name: "move_type",
            kind: OdooFieldKind::Selection, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "journal_id",
            kind: OdooFieldKind::Many2one, target: Some("account.journal"), required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "company_id",
            kind: OdooFieldKind::Many2one, target: Some("res.company"), required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "partner_id",
            kind: OdooFieldKind::Many2one, target: Some("res.partner"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // Auto-advanced by _post() when falling inside a locked period (lock-date heuristic).
        OdooField { name: "date",
            kind: OdooFieldKind::Date, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Date },
        OdooField { name: "invoice_date",
            kind: OdooFieldKind::Date, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Date },
        // Flipped to 'at_date' by _post(soft=True) for future-dated moves.
        // Values: no, at_date, monthly, quarterly, yearly.
        OdooField { name: "auto_post",
            kind: OdooFieldKind::Selection, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "auto_post_until",
            kind: OdooFieldKind::Date, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        OdooField { name: "auto_post_origin_id",
            kind: OdooFieldKind::Many2one, target: Some("account.move"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "reversed_entry_id",
            kind: OdooFieldKind::Many2one, target: Some("account.move"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "reversal_move_id",
            kind: OdooFieldKind::One2many, target: Some("account.move"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // K11 Festschreibung: "$4$<sha256_hex>". Once set → immutable; must be reversed.
        OdooField { name: "inalterable_hash",
            kind: OdooFieldKind::Char, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Audit },
        OdooField { name: "sequence_prefix",
            kind: OdooFieldKind::Char, target: None, required: false,
            computed: Some("_compute_name"), depends: &["name"],
            semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "sequence_number",
            kind: OdooFieldKind::Integer, target: None, required: false,
            computed: Some("_compute_name"), depends: &["name"],
            semantic_role: OdooSemanticRole::Quantity },
        OdooField { name: "line_ids",
            kind: OdooFieldKind::One2many, target: Some("account.move.line"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Document },
        OdooField { name: "currency_id",
            kind: OdooFieldKind::Many2one, target: Some("res.currency"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Money },
        OdooField { name: "invoice_currency_rate",
            kind: OdooFieldKind::Float, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Money },
        OdooField { name: "amount_total",
            kind: OdooFieldKind::Monetary, target: None, required: false,
            computed: Some("_compute_amount"),
            depends: &["line_ids.price_subtotal", "line_ids.tax_ids"],
            semantic_role: OdooSemanticRole::Money },
        // l10n_de: auto-filled to invoice_date before _post() for German sale docs (§14 UStG).
        OdooField { name: "delivery_date",
            kind: OdooFieldKind::Date, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Date },
        OdooField { name: "fiscal_position_id",
            kind: OdooFieldKind::Many2one, target: Some("account.fiscal.position"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Tax },
        OdooField { name: "partner_bank_id",
            kind: OdooFieldKind::Many2one, target: Some("res.partner.bank"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
    ],
    methods: &[
        // Core posting flow: validate, lock-date advance, sequence, state write, reconcile.
        OdooMethod { name: "_post",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Recordset, triggers: &[] },
        OdooMethod { name: "action_post",
            kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Action,
            triggers: &["action_post"] },
        OdooMethod { name: "button_draft",
            kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit,
            triggers: &["button_draft"] },
        OdooMethod { name: "button_cancel",
            kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit,
            triggers: &["button_cancel"] },
        // K11 user lock button → _hash_moves(force_hash=True).
        OdooMethod { name: "button_hash",
            kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit, triggers: &[] },
        // GoBD Storno: copy + negate balance/amount_currency; two audit rows kept.
        OdooMethod { name: "_reverse_moves",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Recordset,
            triggers: &["_reverse_moves"] },
        // Context-manager; SQL HAVING ROUND(SUM(balance), decimal_places) != 0.
        OdooMethod { name: "_check_balanced",
            kind: OdooMethodKind::Constrain, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_get_unbalanced_moves",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Recordset, triggers: &[] },
        // posted_before=True freezes name; never resets previously-posted Belegnummer.
        OdooMethod { name: "_compute_name",
            kind: OdooMethodKind::Compute, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_set_next_sequence",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
        // Gap-prevention: SAVEPOINT loop + UPDATE + UniqueViolation retry.
        OdooMethod { name: "_locked_increment",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
        // Detects periodicity: year_range_month→monthly→year_range→yearly→fixed.
        OdooMethod { name: "_deduce_sequence_number_reset",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_get_last_sequence_domain",
            kind: OdooMethodKind::Override, return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_get_last_sequence",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Record, triggers: &[] },
        OdooMethod { name: "_get_starting_sequence",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        // K11: SHA-256 chain per (journal_id, sequence_prefix).
        // Input includes name/date/journal_id/company_id + per-line fields.
        // Stored as "$4$<sha256_hex>"; raw hex used for chaining (version stripped).
        OdooMethod { name: "_calculate_hashes",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        OdooMethod { name: "_hash_moves",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
        // Groups by (journal_id, sequence_prefix); raises on sequence gap.
        OdooMethod { name: "_get_chains_to_hash",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        // Returns ["name","date","journal_id","company_id"] for hash_version >= 2.
        OdooMethod { name: "_get_integrity_hash_fields",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        // False if inalterable_hash set, or date ≤ fiscal_lock_date, or CABA/FX entry.
        OdooMethod { name: "_can_be_unlinked",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Boolean, triggers: &[] },
        // Guards button_draft: blocks exchange-diff, CABA, and hashed entries.
        OdooMethod { name: "_check_draftable",
            kind: OdooMethodKind::Constrain, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_get_violated_lock_dates",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        // Returns next open accounting date after a locked period.
        OdooMethod { name: "_get_accounting_date",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Date, triggers: &[] },
        // Axis-2 HEURISTIC: wizard after 3+ consecutive unmodified bills from same partner.
        OdooMethod { name: "_show_autopost_bills_wizard",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Action, triggers: &[] },
        // auto_post=monthly/quarterly/yearly → copy for next period after posting.
        OdooMethod { name: "_copy_recurring_entries",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Recordset, triggers: &[] },
        OdooMethod { name: "_affect_tax_report",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Boolean, triggers: &[] },
        // Raises ValidationError if posted_before AND name set.
        OdooMethod { name: "action_switch_move_type",
            kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["state", "journal_id", "date", "posted_before"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["line_ids"],
        },
    ],
    state_machine: Some(&MOVE_SM),
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "ROUND(SUM(balance), company_currency.decimal_places) == 0 \
                        (double-entry balance invariant)",
            source_method: Some("_check_balanced"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Cannot reset to draft a move with inalterable_hash (K11 gate)",
            source_method: Some("_check_draftable"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "UNIQUE(journal_id, sequence_prefix, sequence_number) — gap-free Belegnummer",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Cannot change move_type if posted_before=True and name is set",
            source_method: Some("action_switch_move_type"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L1-K3-POST.md",
        l_doc_lines: (27, 743),
        odoo_source: &[],
        confidence: OdooConfidence::Curated,
    },
};

// ─── account.move.line ────────────────────────────────────────────────────────

pub const ACCOUNT_MOVE_LINE: OdooEntity = OdooEntity {
    model_name: "account.move.line",
    description: "Journal entry line; balance/debit/credit in company currency, \
                  amount_currency for FX; subject to off-balance, payable/receivable, \
                  CABA, matching-number, and deductibility constraints.",
    fields: &[
        OdooField { name: "move_id",
            kind: OdooFieldKind::Many2one, target: Some("account.move"), required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "account_id",
            kind: OdooFieldKind::Many2one, target: Some("account.account"), required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // Company-currency balance; >0 = debit, <0 = credit (non-storno).
        // Journal entries: last line auto-balanced to negative sum of others.
        OdooField { name: "balance",
            kind: OdooFieldKind::Monetary, target: None, required: false,
            computed: Some("_compute_balance"),
            depends: &["price_unit", "quantity", "discount", "tax_ids",
                       "currency_id", "move_id.move_type"],
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "debit",
            kind: OdooFieldKind::Monetary, target: None, required: false,
            computed: Some("_compute_debit_credit"), depends: &["balance", "is_storno"],
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "credit",
            kind: OdooFieldKind::Monetary, target: None, required: false,
            computed: Some("_compute_debit_credit"), depends: &["balance", "is_storno"],
            semantic_role: OdooSemanticRole::Money },
        // FX balance: currency_id.round(balance * currency_rate); equals balance when same ccy.
        OdooField { name: "amount_currency",
            kind: OdooFieldKind::Monetary, target: None, required: false,
            computed: Some("_compute_amount_currency"), depends: &["currency_rate", "balance"],
            semantic_role: OdooSemanticRole::Money },
        // Invoices: header invoice_currency_rate. Journal entries: live rate at move date.
        OdooField { name: "currency_rate",
            kind: OdooFieldKind::Float, target: None, required: false,
            computed: Some("_compute_currency_rate"),
            depends: &["currency_id", "company_id",
                       "move_id.invoice_currency_rate", "move_id.date"],
            semantic_role: OdooSemanticRole::Money },
        OdooField { name: "currency_id",
            kind: OdooFieldKind::Many2one, target: Some("res.currency"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Money },
        OdooField { name: "company_id",
            kind: OdooFieldKind::Many2one, target: Some("res.company"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "partner_id",
            kind: OdooFieldKind::Many2one, target: Some("res.partner"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "tax_ids",
            kind: OdooFieldKind::Many2many, target: Some("account.tax"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Tax },
        // Non-null on tax lines generated by _sync_tax_lines(); null on base lines.
        OdooField { name: "tax_line_id",
            kind: OdooFieldKind::Many2one, target: Some("account.tax"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Tax },
        // Storno flag: flips debit/credit sign presentation (German accounting).
        OdooField { name: "is_storno",
            kind: OdooFieldKind::Boolean, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        // Values: line_section, line_subsection, line_note, tax, non_deductible_tax,
        // payment_term, cogs, product, '' (normal).
        OdooField { name: "display_type",
            kind: OdooFieldKind::Selection, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
        // Format: ^((P?\d+)|(I.+))$. 'I'=import temp; 'P'=partial; numeric=full reconcile.
        OdooField { name: "matching_number",
            kind: OdooFieldKind::Char, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        OdooField { name: "full_reconcile_id",
            kind: OdooFieldKind::Many2one, target: Some("account.full.reconcile"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // [0, 100]; non-purchase docs must equal 100.
        OdooField { name: "deductible_amount",
            kind: OdooFieldKind::Float, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Tax },
        // K11 hash fields (version >= 2): name included in _get_integrity_hash_fields.
        OdooField { name: "name",
            kind: OdooFieldKind::Char, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Document },
        OdooField { name: "price_unit",
            kind: OdooFieldKind::Monetary, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Money },
        OdooField { name: "quantity",
            kind: OdooFieldKind::Float, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Quantity },
    ],
    methods: &[
        OdooMethod { name: "_compute_balance",
            kind: OdooMethodKind::Compute, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_debit_credit",
            kind: OdooMethodKind::Compute, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_amount_currency",
            kind: OdooMethodKind::Compute, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_compute_currency_rate",
            kind: OdooMethodKind::Compute, return_kind: OdooReturnKind::Unit, triggers: &[] },
        // Called explicitly from _post(), not via decorator. Validates account active,
        // currency consistency, journal default/suspense account exemption.
        OdooMethod { name: "_check_constrains_account_id_journal_id",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_check_off_balance",
            kind: OdooMethodKind::Constrain, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_check_payable_receivable",
            kind: OdooMethodKind::Constrain, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_check_caba_non_caba_shared_tags",
            kind: OdooMethodKind::Constrain, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_constrains_matching_number",
            kind: OdooMethodKind::Constrain, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "_constrains_deductible_amount",
            kind: OdooMethodKind::Constrain, return_kind: OdooReturnKind::Unit, triggers: &[] },
        // K11 hash fields: ["name","debit","credit","account_id","partner_id"] (version >= 2).
        OdooMethod { name: "_get_integrity_hash_fields",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Dict, triggers: &[] },
        // Tax priority: product.taxes_id → account_id.tax_ids → fiscal_position.map_tax().
        OdooMethod { name: "_get_computed_taxes",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Recordset, triggers: &[] },
        OdooMethod { name: "_create_analytic_lines",
            kind: OdooMethodKind::Helper, return_kind: OdooReturnKind::Unit, triggers: &[] },
        OdooMethod { name: "remove_move_reconcile",
            kind: OdooMethodKind::Action, return_kind: OdooReturnKind::Unit, triggers: &[] },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["balance", "is_storno"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["currency_rate", "balance"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["currency_id", "company_id",
                       "move_id.invoice_currency_rate", "move_id.date"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["account_id", "display_type"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["matching_number", "full_reconcile_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["deductible_amount", "move_id"],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Off-balance: all move lines must share account_type='off_balance'; \
                        no tax_ids; no reconcile",
            source_method: Some("_check_off_balance"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Sale docs: liability_payable forbidden; \
                        payment_term display_type XOR asset_receivable account",
            source_method: Some("_check_payable_receivable"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "CABA and non-CABA taxes cannot share repartition tags on same line",
            source_method: Some("_check_caba_non_caba_shared_tags"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "matching_number: ^((P?\\d+)|(I.+))$; \
                        'P' prefix requires partials + no full_reconcile_id; \
                        numeric requires full_reconcile_id == str(id)",
            source_method: Some("_constrains_matching_number"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "deductible_amount in [0, 100]; non-purchase docs must equal 100",
            source_method: Some("_constrains_deductible_amount"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L1-K3-POST.md",
        l_doc_lines: (234, 362),
        odoo_source: &[],
        confidence: OdooConfidence::Curated,
    },
};

// ─── account.journal ──────────────────────────────────────────────────────────

pub const ACCOUNT_JOURNAL: OdooEntity = OdooEntity {
    model_name: "account.journal",
    description: "Accounting journal; owns Belegnummer sequence format, optional \
                  sequence_override_regex, and the (journal_id, sequence_prefix) \
                  K11 hash-chain anchor.",
    fields: &[
        OdooField { name: "name",
            kind: OdooFieldKind::Char, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        // Used as prefix in starting sequence (e.g. "INV/2024/00000").
        OdooField { name: "code",
            kind: OdooFieldKind::Char, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Identity },
        // Values: sale, purchase, bank, cash, general.
        // Determines default sequence format: sale/bank/cash = annual; other = monthly.
        OdooField { name: "type",
            kind: OdooFieldKind::Selection, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        // Archived journal blocks _post() with "archived journal" validation error.
        OdooField { name: "active",
            kind: OdooFieldKind::Boolean, target: None, required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Status },
        OdooField { name: "company_id",
            kind: OdooFieldKind::Many2one, target: Some("res.company"), required: true,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        // Overrides sequence_mixin's default regex table; checked first in
        // _get_last_sequence_domain before built-in patterns.
        OdooField { name: "sequence_override_regex",
            kind: OdooFieldKind::Char, target: None, required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Policy },
        // Lines on default_account_id or suspense_account_id bypass account constraint.
        OdooField { name: "default_account_id",
            kind: OdooFieldKind::Many2one, target: Some("account.account"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
        OdooField { name: "suspense_account_id",
            kind: OdooFieldKind::Many2one, target: Some("account.account"), required: false,
            computed: None, depends: &[], semantic_role: OdooSemanticRole::Reference },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L1-K3-POST.md",
        l_doc_lines: (488, 562),
        odoo_source: &[],
        confidence: OdooConfidence::Curated,
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

pub const ENTITIES: &[OdooEntity] = &[ACCOUNT_MOVE, ACCOUNT_MOVE_LINE, ACCOUNT_JOURNAL];
