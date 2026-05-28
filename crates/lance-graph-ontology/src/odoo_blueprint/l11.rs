//! Lane L11 (COA-JOURNALS-LOCKDATES) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L11-COA-JOURNALS-LOCKDATES.md`.
//!
//! Entities covered (4):
//!   - `account.account`       (R1-R5: account_type, reconcile, code, group hierarchy)
//!   - `account.account.tag`   (R6: tag structure, applicability, tax/report buckets)
//!   - `account.journal`       (R7-R10: type taxonomy, default accounts, GoBD hash)
//!   - `res.company` (lock-date ext) (R11-R19: 5 lock-date types, exception logic)
//!
//! **Overlap notes:**
//!   - `account.journal` overlaps with L1 (invoice flow).  L1 captures journal
//!     as a posting vehicle; L11 captures its structure (type taxonomy, default
//!     accounts, GoBD `restrict_mode_hash_table`, code auto-gen).
//!   - `account.account` overlaps with L4 (payment matching).  L4 uses accounts
//!     as references in reconciliation flows; L11 owns the COA structure: type
//!     enum, reconcile flag, code format, group prefix hierarchy.
//!   - `sequence.mixin` is an abstract mixin (R20-R28) — captured conceptually
//!     in comments but not projected as a stored `OdooEntity`.
//!   - `account.group` is a helper model for the prefix-based account hierarchy
//!     (R5); it carries only structural fields and no methods curated in L11,
//!     so it is captured inline in comments rather than as a separate entity.
//!   - `account.lock_exception` (R12) — referenced as a community model but
//!     not deep-read in L11 (see "Enterprise gaps flagged" in the L-doc, L149).
//!     Excluded from ENTITIES; add in a follow-up lane when the field shape is
//!     confirmed (open question 1, L151).

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooEntityKind, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind,
    OdooProvenance, OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── account.account ─────────────────────────────────────────────────────────
//
// Central chart-of-accounts record.  Key concerns from L11:
//   R1  — 19-value `account_type` enum; `internal_group` = prefix before `_`.
//   R2  — `include_initial_balance` (Bilanz vs GuV).
//   R3  — `reconcile` flag + constraints (forced true for receivable/payable;
//          blocked toggle if partial reconciliations exist).
//   R4  — `code` is company-dependent (JSONB `code_store`, per-root-company).
//          Uniqueness across parent/child hierarchy.
//   R5  — `account.group` hierarchy via `code_prefix_start`/`code_prefix_end`
//          (longest-fitting prefix = parent group, SQL DISTINCT ON).
//
// L1/L4 overlap: `account.account` appears in L1 (as posting target) and L4
// (as reconciliation target).  L11 owns the COA-structure concerns above.

pub const ACCOUNT_ACCOUNT: OdooEntity = OdooEntity {
    model_name: "account.account",
    kind: OdooEntityKind::Model,
    description: "Chart-of-accounts leaf record.  `account_type` is a 19-value closed enum \
                  that determines `internal_group`, `include_initial_balance` (Bilanz vs GuV \
                  reset), and whether reconciliation is allowed.  `code` is company-scoped via \
                  JSONB `code_store`; uniqueness spans parent/child company hierarchy.  \
                  Placed in `account.group` by longest-matching `code_prefix_start/end`.",
    fields: &[
        // Natural key; company-scoped JSONB store (R4).
        // Regex ^[A-Za-z0-9.]+$; uniqueness across parent/child hierarchy.
        OdooField {
            name: "code",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
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
        // 19-value closed enum (R1): asset_receivable, asset_cash, asset_current,
        // asset_non_current, asset_prepayments, asset_fixed, liability_payable,
        // liability_credit_card, liability_current, liability_non_current, equity,
        // equity_unaffected, income, income_other, expense, expense_other,
        // expense_depreciation, expense_direct_cost, off_balance.
        // Drives include_initial_balance, reconcile defaults, and tax eligibility.
        OdooField {
            name: "account_type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Computed from account_type (R2): True iff internal_group ∉ {income, expense}
        // AND type ≠ equity_unaffected.  Drives Bilanz carry-forward vs GuV reset.
        OdooField {
            name: "include_initial_balance",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: Some("_compute_include_initial_balance"),
            depends: &["account_type"],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Reconcile flag (R3): income/expense/equity → false; receivable/payable → true
        // (forced); cash/credit_card/off_balance → false.
        // Toggle true→false blocked if partial reconciliations exist.
        OdooField {
            name: "reconcile",
            kind: OdooFieldKind::Boolean,
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
        // Parent account group resolved by longest-matching code prefix (R5).
        OdooField {
            name: "group_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.group"),
            required: false,
            computed: Some("_compute_account_group"),
            depends: &["code"],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Tags used for tax report buckets (R6 cross-ref).
        OdooField {
            name: "tag_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("account.account.tag"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // For deprecatable accounts (R1 expense_depreciation).
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
    methods: &[
        OdooMethod {
            name: "_compute_include_initial_balance",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // SQL UPDATE: sets reconciled=(debit=0 AND credit=0) on toggling reconcile
        // from false→true (R3, account_account.py:L963-989).
        OdooMethod {
            name: "_compute_account_group",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // Increment trailing digit group of `code` until unique; fallback ".copy{n}" (R4).
        OdooMethod {
            name: "_search_new_account_code",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Record,
            triggers: &[],
        },
        // Blocks toggle reconcile true→false when partial reconciliations exist (R3).
        OdooMethod {
            name: "_check_reconciliation",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["account_type"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["code"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["reconcile"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
    ],
    state_machine: None,
    constraints: &[
        // receivable/payable MUST be reconcilable; off_balance cannot reconcile or carry
        // taxes (R3, account_account.py:L187-194).
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "account_type receivable/payable requires reconcile=True; \
                        off_balance cannot have reconcile=True or tax assignments",
            source_method: Some("_check_reconciliation"),
        },
        // code uniqueness per root company (R4).
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "UNIQUE(code, company_id) — uniqueness also enforced across \
                        parent/child company hierarchy in Python (R4)",
            source_method: None,
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L11-COA-JOURNALS-LOCKDATES.md",
        l_doc_lines: (27, 103),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_account.py",
            line_range: (1, 1642),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── account.account.tag ─────────────────────────────────────────────────────
//
// R6 (L49-57): Scoped by `applicability ∈ {accounts, taxes, products}` + optional
// `country_id`.  Tax tags drive report buckets via Enterprise `account_report_expression`.
// Tag name starting `-` ⇒ `balance_negate` (computed via LEFT JOIN — Enterprise-only).
// Master operating/financing/investing tags are undeletable.

pub const ACCOUNT_ACCOUNT_TAG: OdooEntity = OdooEntity {
    model_name: "account.account.tag",
    kind: OdooEntityKind::Model,
    description: "Annotation tag scoped to applicability domain (accounts/taxes/products) \
                  and optional country.  Tax-scoped tags drive report-line buckets in the \
                  Enterprise reporting engine.  Name starting `-` signals `balance_negate` \
                  (computed via Enterprise `account_report_expression` JOIN, not stored).",
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
        // 'accounts' | 'taxes' | 'products' — determines domain of use.
        OdooField {
            name: "applicability",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Optional country scope; tags without country_id are global.
        OdooField {
            name: "country_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.country"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // Computed via LEFT JOIN to Enterprise account_report_expression (not stored).
        // True iff tag name starts with '-'.
        OdooField {
            name: "balance_negate",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: Some("_compute_balance_negate"),
            depends: &["name"],
            semantic_role: OdooSemanticRole::Policy,
        },
        // active=False allowed but not fully protected; master tags (operating/
        // financing/investing) are undeletable (Python guard).
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
    methods: &[
        OdooMethod {
            name: "_compute_balance_negate",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["name"],
    }],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Sql,
        condition: "UNIQUE(name, applicability, country_id) — tag names unique per \
                    applicability scope and country (R6, account_account_tag.py:L1-141)",
        source_method: None,
    }],
    provenance: OdooProvenance {
        l_doc: "L11-COA-JOURNALS-LOCKDATES.md",
        l_doc_lines: (49, 57),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_account_tag.py",
            line_range: (1, 140),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── account.journal ─────────────────────────────────────────────────────────
//
// R7 (L53-56): 6 type values: sale, purchase, cash, bank, credit, general.
// R8 (L57-59): Auto-create default account for bank/cash/credit from company prefix.
// R9 (L61-63): Code auto-gen (INV/BILL/CSH/BNK/CCD/MISC + numeric suffix, max 5 chars).
// R10 (L65-67): `restrict_mode_hash_table` — GoBD Festschreibung anchor; once any entry
//   hashed, cannot be unset (UserError).  Hash itself lives on account.move (L1).
//
// L1 overlap: account.journal appears in L1 as the posting vehicle for
// account.move.  L11 owns: type taxonomy, default accounts, code format, GoBD flag.
//
// R15 non-obvious: general journals are NOT subject to sale/purchase lock dates.

pub const ACCOUNT_JOURNAL: OdooEntity = OdooEntity {
    model_name: "account.journal",
    kind: OdooEntityKind::Model,
    description: "Journal record: type-classified posting book (sale/purchase/cash/bank/ \
                  credit/general).  Drives default account selection, lock-date scope \
                  (general journals exempt from sale/purchase locks per R15), sequence \
                  generation, and GoBD Festschreibung via `restrict_mode_hash_table`.",
    fields: &[
        // Auto-generated: INV/BILL/CSH/BNK/CCD/MISC + numeric suffix ≤ 5 chars (R9).
        OdooField {
            name: "code",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
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
        // 6-value enum (R7): sale | purchase | cash | bank | credit | general.
        // Determines default account domain, suspense/payment lines, lock-date
        // applicability (R15), and refund_sequence / payment_sequence defaults.
        OdooField {
            name: "type",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Default account for postings; auto-created for bank/cash/credit (R8).
        OdooField {
            name: "default_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
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
        // GoBD Festschreibung flag (R10).  Once any move in this journal carries
        // `inalterable_hash != False`, this cannot be set back to False (UserError).
        // Drives savant: LockDateAdvancer (via journal type + lock-date axis).
        OdooField {
            name: "restrict_mode_hash_table",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // sale/purchase journals default True; controls whether a sequence is
        // created for credit notes.
        OdooField {
            name: "refund_sequence",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // bank/cash/credit journals default True; controls payment sequence creation.
        OdooField {
            name: "payment_sequence",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Suspense account for bank/cash/credit journals (liquidity line).
        OdooField {
            name: "suspense_account_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.account"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[
        // Code generation: try prefix N=1..99, fallback for uniqueness, max 5 chars (R9).
        OdooMethod {
            name: "_get_sequence_prefix",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // Auto-create account for bank/cash/credit when default_account_id is absent (R8).
        // Prefix from company.bank_account_code_prefix / cash_account_code_prefix.
        OdooMethod {
            name: "_prepare_liquidity_account_vals",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // Blocks re-disabling restrict_mode_hash_table once any entry hashed (R10).
        OdooMethod {
            name: "_check_restrict_mode_hash_table",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiConstrains,
        targets: &["restrict_mode_hash_table"],
    }],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "UNIQUE(code, company_id) — journal codes unique per company (R9)",
            source_method: None,
        },
        // Once any move in the journal has inalterable_hash set, restrict_mode_hash_table
        // cannot be unset (UserError, R10, account_journal.py:L794-801).
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "restrict_mode_hash_table cannot be set False once any journal entry \
                        has been hashed (GoBD Festschreibung irreversibility, R10)",
            source_method: Some("_check_restrict_mode_hash_table"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L11-COA-JOURNALS-LOCKDATES.md",
        l_doc_lines: (53, 103),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/account_journal.py",
            line_range: (1, 1300),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── res.company (lock-date extension) ───────────────────────────────────────
//
// L11 captures the lock-date extension fields and methods on res.company
// (company.py:L50-749).  The base res.company entity (address, currency,
// fiscal year) belongs to other lanes; this slice is scoped to:
//
//   R11 (L69-72): 5 lock-date fields — fiscalyear_lock_date (global),
//     tax_lock_date (auto-set on tax-closing post), sale_lock_date,
//     purchase_lock_date, hard_lock_date (irreversible).
//     SOFT_LOCK_DATE_FIELDS = first four.
//   R12 (L73-75): _get_user_lock_date — walks parent_ids; respects
//     account.lock_exception for per-user temporary overrides.
//   R13 (L77-79): _get_violated_soft_lock_date — two-pass fast short-circuit.
//   R14 (L81-83): _get_lock_date_violations — multi-lock sweep with flags.
//   R15 (L85-87): _get_violated_lock_dates — move-level; general journals
//     exempt from sale/purchase locks.
//   R16 (L89-91): _check_fiscal_lock_dates — write/post guard; skipped via
//     BYPASS_LOCK_CHECK object-identity sentinel.
//   R17 (L93-95): _get_user_fiscal_lock_date — single effective lock for copy/unlink.
//   R18 (L97-99): _validate_locks — write-time guard; hard_lock_date cannot
//     be unset or decreased.
//   R19 (L101-103): _get_accounting_date — bumps dates past violated locks.
//
// Lock-date fields drive savant: LockDateAdvancer (AXIS-B).

pub const RES_COMPANY_LOCK_DATE: OdooEntity = OdooEntity {
    model_name: "res.company",
    kind: OdooEntityKind::Model,
    description: "Lock-date extension on res.company (L11 scope: fields R11-R19 only). \
                  Five lock-date types control posting windows: fiscalyear (global), \
                  tax (auto-set on tax-close post), sale, purchase, hard (irreversible). \
                  `_get_user_lock_date` walks parent_ids and applies `account.lock_exception` \
                  overrides.  General journals are exempt from sale/purchase lock checks (R15). \
                  Drives savant: LockDateAdvancer.",
    fields: &[
        // Global accounting lock; all journals respect this (R11, R15).
        OdooField {
            name: "fiscalyear_lock_date",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Auto-set when a tax-closing entry is posted; covers entries with taxes (R11).
        OdooField {
            name: "tax_lock_date",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Only sale journals subject to this lock (R11, R15).
        OdooField {
            name: "sale_lock_date",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Only purchase journals subject to this lock (R11, R15).
        OdooField {
            name: "purchase_lock_date",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
        // Irreversible hard lock (cannot be unset or decreased, R18).
        // user_hard_lock_date = max hard lock across full parent_ids hierarchy.
        OdooField {
            name: "hard_lock_date",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Policy,
        },
    ],
    methods: &[
        // Walks parent_ids (sudo).  ignore_exceptions=true → max(company[field]).
        // Else: find active account.lock_exception for current user → effective =
        // max(soft, exception[field] or min).  (R12, company.py:L597-630)
        OdooMethod {
            name: "_get_user_lock_date",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Date,
            triggers: &[],
        },
        // Two-pass fast short-circuit: regular (ignore_exceptions=true) first;
        // then with exceptions.  Returns None if date passes or exception covers.
        // (R13, company.py:L646-663)
        OdooMethod {
            name: "_get_violated_soft_lock_date",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Date,
            triggers: &[],
        },
        // Given accounting_date + flags(fiscalyear, sale, purchase, tax, hard)
        // → Vec<(date, field)> sorted ascending.  Hard: user_hard_lock_date >= date.
        // (R14, company.py:L665-700)
        OdooMethod {
            name: "_get_lock_date_violations",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // Move-level: fiscalyear=true always; sale=journal.type==sale;
        // purchase=journal.type==purchase; tax=has_tax; hard=true.
        // General journal exempt from sale/purchase.  (R15, company.py:L713-729)
        OdooMethod {
            name: "_get_violated_lock_dates",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        // write/post guard: calls violations with tax=false (tax separate).
        // Skipped via BYPASS_LOCK_CHECK object-identity sentinel (not truthiness).
        // (R16, account_move.py:L2796-2813, L3905-3958)
        OdooMethod {
            name: "_check_fiscal_lock_dates",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // Single effective fiscal lock for copy/unlink: max(fiscalyear, hard,
        // +sale/purchase by journal type).  (R17, company.py:L632-644)
        OdooMethod {
            name: "_get_user_fiscal_lock_date",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Date,
            triggers: &[],
        },
        // Write-time guard: hard_lock_date cannot be unset or decreased (UserError).
        // RedirectWarning if draft entries ≤ date, or unreconciled bank lines ≤
        // max(fiscalyear, hard).  (R18, company.py:L542-595)
        OdooMethod {
            name: "_validate_locks",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // Locked period ⇒ base = last_violation + 1day.  Couples sequence format ↔
        // lock semantics (sale + number_reset month/year ⇒ last day of month/year).
        // (R19, account_move.py:L6570-6607)
        OdooMethod {
            name: "_get_accounting_date",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Date,
            triggers: &[],
        },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[
        // hard_lock_date cannot be unset or decreased (R18).
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "hard_lock_date cannot be set to None or decreased once set; \
                        RedirectWarning if draft entries or unreconciled bank lines exist \
                        in the locked period (R18, company.py:L542-595)",
            source_method: Some("_validate_locks"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L11-COA-JOURNALS-LOCKDATES.md",
        l_doc_lines: (69, 103),
        odoo_source: &[OdooSourceRef {
            path: "addons/account/models/company.py",
            line_range: (50, 749),
        }],
        confidence: OdooConfidence::Curated,
        regulation_iri: &[],
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// Entities documented in lane L11 (chart of accounts + journals + lock dates
/// + sequence integrity).
///
/// Excluded by design:
/// - `account.group` — structural helper for code-prefix hierarchy (R5); no
///   methods curated in L11; captured in comments on `account.account`.
/// - `sequence.mixin` — abstract mixin (R20-R28, L105-144); not a stored model;
///   its sequence-format families and gap-prevention logic are referenced in
///   comments; a separate stored-model projection would require a dedicated lane.
/// - `account.lock_exception` — referenced (R12) but field shape not deep-read
///   (L-doc L149-154, open question 1); excluded pending confirmation.
/// - `account.journal` lock-date fields — `fiscalyear_lock_date` etc. live on
///   `res.company`; `account.journal.type` is the selector for lock applicability
///   (R15); both captured above.
pub const ENTITIES: &[OdooEntity] = &[
    ACCOUNT_ACCOUNT,
    ACCOUNT_ACCOUNT_TAG,
    ACCOUNT_JOURNAL,
    RES_COMPANY_LOCK_DATE,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooFieldKind, OdooSemanticRole};

    #[test]
    fn l11_entities_slice_has_four() {
        assert_eq!(ENTITIES.len(), 4);
    }

    #[test]
    fn all_entities_cite_l11_doc() {
        for entity in ENTITIES {
            assert_eq!(
                entity.provenance.l_doc,
                "L11-COA-JOURNALS-LOCKDATES.md",
                "{} must reference L11-COA-JOURNALS-LOCKDATES.md",
                entity.model_name,
            );
            assert_eq!(
                entity.provenance.confidence,
                OdooConfidence::Curated,
                "{} must be Curated",
                entity.model_name,
            );
        }
    }

    #[test]
    fn account_account_type_is_policy() {
        let acc = &ACCOUNT_ACCOUNT;
        assert_eq!(acc.model_name, "account.account");
        let t = acc.fields.iter().find(|f| f.name == "account_type").unwrap();
        assert_eq!(t.kind, OdooFieldKind::Selection);
        assert_eq!(t.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn account_account_include_initial_balance_computed() {
        let acc = &ACCOUNT_ACCOUNT;
        let f = acc
            .fields
            .iter()
            .find(|f| f.name == "include_initial_balance")
            .unwrap();
        assert_eq!(f.computed, Some("_compute_include_initial_balance"));
        assert_eq!(f.depends, &["account_type"]);
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn lock_date_fields_are_policy_on_res_company() {
        let co = &RES_COMPANY_LOCK_DATE;
        assert_eq!(co.model_name, "res.company");
        let lock_fields = [
            "fiscalyear_lock_date",
            "tax_lock_date",
            "sale_lock_date",
            "purchase_lock_date",
            "hard_lock_date",
        ];
        for name in lock_fields {
            let f = co.fields.iter().find(|f| f.name == name).unwrap_or_else(|| {
                panic!("lock-date field '{}' missing on res.company entity", name)
            });
            assert_eq!(
                f.kind,
                OdooFieldKind::Date,
                "field '{}' must be Date",
                name
            );
            assert_eq!(
                f.semantic_role,
                OdooSemanticRole::Policy,
                "field '{}' must be Policy",
                name
            );
        }
    }

    #[test]
    fn journal_restrict_mode_hash_table_present() {
        let j = &ACCOUNT_JOURNAL;
        assert_eq!(j.model_name, "account.journal");
        let f = j
            .fields
            .iter()
            .find(|f| f.name == "restrict_mode_hash_table")
            .unwrap();
        assert_eq!(f.kind, OdooFieldKind::Boolean);
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn journal_type_field_is_selection_policy() {
        let j = &ACCOUNT_JOURNAL;
        let f = j.fields.iter().find(|f| f.name == "type").unwrap();
        assert_eq!(f.kind, OdooFieldKind::Selection);
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
    }

    #[test]
    fn account_account_tag_applicability_uniqueness() {
        let tag = &ACCOUNT_ACCOUNT_TAG;
        assert_eq!(tag.model_name, "account.account.tag");
        let f = tag
            .fields
            .iter()
            .find(|f| f.name == "applicability")
            .unwrap();
        assert_eq!(f.semantic_role, OdooSemanticRole::Policy);
        assert!(tag.constraints.iter().any(|c| {
            c.kind == OdooConstraintKind::Sql
                && c.condition.contains("UNIQUE(name, applicability, country_id)")
        }));
    }
}
