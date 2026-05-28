//! Lane L14 (HR-BASE) — typed Odoo entity declarations.
//!
//! Source: `.claude/odoo/L14-HR-BASE.md`.
//!
//! ## Entity inventory (4 entities)
//!
//! | Const              | Odoo model        | L-doc rules |
//! |---|---|---|
//! | [`HR_EMPLOYEE`]    | `hr.employee`     | R1, R4, R8–R12, R16 |
//! | [`HR_DEPARTMENT`]  | `hr.department`   | R5, R6, R17 |
//! | [`HR_JOB`]         | `hr.job`          | R7 |
//! | [`HR_CONTRACT_TYPE`] | `hr.contract.type` | R14, R15 (payroll boundary) |
//!
//! ## OGIT family note
//!
//! All four entities target a NEW OGIT family `0x90 HRFoundation` (proposed
//! in the L14 L-doc §Ontology rows). No savants currently dispatch to L14
//! entities directly — they are substrate for future HR-domain savants.
//! Alternative: map `hr.employee` → 0x80 SmbFoundryCustomer via
//! `work_contact_id` (decide at synthesis).
//!
//! ## Payroll boundary
//!
//! `hr_payroll` is **absent** (Enterprise-only). Models `hr.payslip`,
//! `hr.salary.rule`, `hr.payroll.structure` are NOT projected here.
//! `hr.contract.type` and `hr.payroll.structure.type` are thin community
//! stubs included only as discriminant anchors for the woa-rs payroll engine.
//! `hr.version` (temporal slice of hr.employee) is referenced in L-doc
//! R2/R3/R13–R16 but is a sub-model of hr.employee; it is NOT projected as a
//! separate top-level entity here because it has no standalone identity in
//! community (its fields are managed exclusively through hr.employee's
//! version relationship).

use super::{
    OdooConfidence, OdooConstraint, OdooConstraintKind, OdooDecorator, OdooDecoratorKind,
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooProvenance,
    OdooReturnKind, OdooSemanticRole, OdooSourceRef,
};

// ─── 1. hr.employee ──────────────────────────────────────────────────────────
//
// Core work-resource entity. Owns the user↔partner linkage (R8), the version
// chain (R1), gap-tolerant seniority calculation (R4), the newly-hired flag
// (R10), expiry-cron (R11), salary-distribution JSON (R12), the
// version-period calendar query for payroll (R16), and coach-default compute
// (R9).
//
// hr.version sub-model fields (wage, structure_type_id, resource_calendar_id,
// contract_date_start/end, trial_date_end) are referenced via the version
// relationship; only the employee-level contract dates (contract_date_start,
// contract_date_end via current_version_id) are surfaced directly on
// hr.employee in community.

/// `hr.employee` — work-resource entity with versioned employment terms.
///
/// L-doc R1, R4, R8–R12, R16; source:
/// `hr/models/hr_employee.py:1–1865` (full read).
///
/// This is the K13 data foundation for the woa-rs payroll engine.
/// No savant dispatches to this entity directly yet — it is substrate
/// for future HR-domain savants in the 0x90 HRFoundation family.
pub const HR_EMPLOYEE: OdooEntity = OdooEntity {
    model_name: "hr.employee",
    description: "Work-resource individual with versioned employment terms (hr.version chain); \
                  owns user↔partner linkage, org hierarchy (parent_id/coach_id), \
                  statutory identifiers, salary-distribution JSON (SEPA split), \
                  and the version-period calendar query for multi-schedule payroll periods.",
    fields: &[
        // ── Identity ──────────────────────────────────────────────────────
        OdooField {
            name: "name",
            kind: OdooFieldKind::Char,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Identity,
        },
        // ── Org hierarchy ─────────────────────────────────────────────────
        OdooField {
            name: "parent_id",
            kind: OdooFieldKind::Many2one,
            target: Some("hr.employee"),
            required: false,
            computed: None,
            depends: &[],
            // Manager (direct line supervisor). Auto-propagated from
            // hr.department on manager change (R6).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "coach_id",
            kind: OdooFieldKind::Many2one,
            target: Some("hr.employee"),
            required: false,
            computed: None,
            depends: &[],
            // Default set to new manager on parent_id change if coach was old
            // manager or empty (R9).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "department_id",
            kind: OdooFieldKind::Many2one,
            target: Some("hr.department"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "job_id",
            kind: OdooFieldKind::Many2one,
            target: Some("hr.job"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
        // ── User / partner linkage (R8) ────────────────────────────────────
        OdooField {
            name: "user_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.users"),
            required: false,
            computed: None,
            depends: &[],
            // Unique per company; work_contact_id auto-created from user's partner.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "work_contact_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.partner"),
            required: false,
            computed: None,
            depends: &[],
            // Auto-created res.partner (work contact). OWL pivot candidate:
            // vcard:Individual. Bridge to 0x80 SmbFoundryCustomer family.
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "work_email",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: Some("_compute_work_contact_details"),
            depends: &["work_contact_id.email"],
            // Computed from work_contact_id when ≤1 linked employee;
            // inverse pushes to partner.
            semantic_role: OdooSemanticRole::Address,
        },
        // ── Statutory identifiers (R8) ────────────────────────────────────
        OdooField {
            name: "identification_id",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // National ID / Ausweisnummer. Statutory input for payroll.
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "ssnid",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Social security number. Statutory payroll input (Sozialversicherungsnummer).
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "barcode",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // DB-unique [A-Za-z0-9]{≤18} (SQL unique constraint, R8).
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "pin",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Digits-only; Python constraint (R8).
            semantic_role: OdooSemanticRole::Identity,
        },
        // ── Version / contract pointer (R1) ────────────────────────────────
        OdooField {
            name: "current_version_id",
            kind: OdooFieldKind::Many2one,
            target: Some("hr.version"),
            required: false,
            computed: Some("_compute_current_version_id"),
            depends: &[],
            // Stored; daily cron _cron_update_current_version_id.
            // = latest hr.version where date_version <= today (desc), fallback earliest.
            semantic_role: OdooSemanticRole::Reference,
        },
        // ── Contract dates (surfaced from current_version_id, R3) ─────────
        OdooField {
            name: "contract_date_start",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Effective start of current employment period.
            // Payroll engine data hook (L-doc §Data hooks).
            semantic_role: OdooSemanticRole::Date,
        },
        OdooField {
            name: "contract_date_end",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // NULL = open-ended. Expiry cron monitors this (R11).
            // Payroll engine data hook (L-doc §Data hooks).
            semantic_role: OdooSemanticRole::Date,
        },
        // ── Work-permit expiry (R11) ───────────────────────────────────────
        OdooField {
            name: "permit_expiration_date",
            kind: OdooFieldKind::Date,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Expiry cron schedules activity at today+60 days (R11).
            semantic_role: OdooSemanticRole::Date,
        },
        // ── Newly-hired flag (R10) ────────────────────────────────────────
        OdooField {
            name: "is_new_hire",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: Some("_compute_is_new_hire"),
            depends: &["create_date"],
            // True if create_date > now − 90 days (R10).
            // Override point: _get_new_hire_field.
            semantic_role: OdooSemanticRole::Status,
        },
        // ── Payroll data hooks (L-doc §Data hooks) ────────────────────────
        OdooField {
            name: "marital",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Lohnsteuer input (DE): single/married/cohabitant/divorced/widower/other.
            semantic_role: OdooSemanticRole::Policy,
        },
        OdooField {
            name: "children",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Number of dependent children. Lohnsteuer input (DE).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "km_home_work",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // One-way km home→work. DE Lohnsteuer: 1.609 km factor.
            semantic_role: OdooSemanticRole::Quantity,
        },
        // ── Salary distribution JSON (R12) ────────────────────────────────
        OdooField {
            name: "bank_account_ids",
            kind: OdooFieldKind::One2many,
            target: Some("res.partner.bank"),
            required: false,
            computed: None,
            depends: &[],
            // SEPA bank accounts. salary_distribution JSON references these
            // by sequence for split payments (R12).
            semantic_role: OdooSemanticRole::Reference,
        },
        // ── HR responsible (R11) ──────────────────────────────────────────
        OdooField {
            name: "hr_responsible_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.users"),
            required: false,
            computed: None,
            depends: &[],
            // Receives expiry-date activity notifications (R11).
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[
        // ── R1: version compute + cron ─────────────────────────────────────
        OdooMethod {
            name: "_compute_current_version_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_cron_update_current_version_id",
            kind: OdooMethodKind::Cron,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R4: gap-tolerant seniority ─────────────────────────────────────
        OdooMethod {
            // Gap < 4 days between versions = continuous (seniority);
            // ≥ 4 days breaks the chain. date_end False → date(2100,1,1).
            name: "_get_employment_date",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Date,
            triggers: &[],
        },
        // ── R8: work-contact compute ───────────────────────────────────────
        OdooMethod {
            name: "_compute_work_contact_details",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R9: coach default ──────────────────────────────────────────────
        OdooMethod {
            name: "_onchange_parent_id",
            kind: OdooMethodKind::Onchange,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R10: newly-hired ───────────────────────────────────────────────
        OdooMethod {
            name: "_compute_is_new_hire",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R11: expiry cron ───────────────────────────────────────────────
        OdooMethod {
            // Exact-date match: contract_date_end == today + notice_period[7],
            // work_permit == today + 60; schedule activity to hr_responsible.
            name: "_cron_check_contract_expiration",
            kind: OdooMethodKind::Cron,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R16: version-period calendar query ────────────────────────────
        OdooMethod {
            // For [start, stop]: collect version-slices active in window,
            // clamp to [max(start, date_start), min(stop, date_end)],
            // map to resource_calendar per version (per-version TZ).
            // Critical for multi-schedule payroll periods (DE law).
            name: "_get_work_days_data_batch",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["work_contact_id.email"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["create_date"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiOnchange,
            targets: &["parent_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiAutovacuum,
            targets: &[],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "UNIQUE(barcode) WHERE barcode IS NOT NULL — barcode [A-Za-z0-9]{≤18}",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "pin must contain only digits",
            source_method: Some("_check_pin"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "user_id must be unique per company (one employee per user per company)",
            source_method: Some("_check_unique_user_id"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L14-HR-BASE.md",
        l_doc_lines: (29, 81),
        odoo_source: &[OdooSourceRef {
            path: "hr/models/hr_employee.py",
            line_range: (1, 1865),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 2. hr.department ────────────────────────────────────────────────────────
//
// Organisational unit with recursive parent hierarchy (_parent_store, R5).
// On manager change, propagates new manager to direct-member employees whose
// parent_id was the old manager (R6).  Non-HR users see only the subtree
// they manage (R17).

/// `hr.department` — org unit with recursive hierarchy and manager propagation.
///
/// L-doc R5, R6, R17; source: `hr/models/hr_department.py:1–243` (full read).
pub const HR_DEPARTMENT: OdooEntity = OdooEntity {
    model_name: "hr.department",
    description: "Organisational unit in the company tree; recursive parent hierarchy \
                  (_parent_store, complete_name path); manager change propagates to \
                  direct-member employees; non-HR ACL restricts to managed subtree.",
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
            name: "complete_name",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: Some("_compute_complete_name"),
            depends: &["name", "parent_id.complete_name"],
            // Recursive: parent.complete_name + ' / ' + name (R5).
            // _parent_store on parent_path; master_department_id = root of path.
            semantic_role: OdooSemanticRole::Identity,
        },
        OdooField {
            name: "parent_id",
            kind: OdooFieldKind::Many2one,
            target: Some("hr.department"),
            required: false,
            computed: None,
            depends: &[],
            // Org hierarchy parent. Cycle check raises (R5).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "master_department_id",
            kind: OdooFieldKind::Many2one,
            target: Some("hr.department"),
            required: false,
            computed: Some("_compute_master_department_id"),
            depends: &["parent_path"],
            // Root of the parent_path tree (computed from _parent_store).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "manager_id",
            kind: OdooFieldKind::Many2one,
            target: Some("hr.employee"),
            required: false,
            computed: None,
            depends: &[],
            // Department manager. On change: direct-member employees with
            // parent_id == old manager are updated to new manager (R6).
            semantic_role: OdooSemanticRole::Reference,
        },
        OdooField {
            name: "member_ids",
            kind: OdooFieldKind::One2many,
            target: Some("hr.employee"),
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
    methods: &[
        // ── R5: hierarchy ──────────────────────────────────────────────────
        OdooMethod {
            name: "_compute_complete_name",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_master_department_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        // ── R6: manager propagation ────────────────────────────────────────
        OdooMethod {
            // On manager change: update employees whose parent_id == old manager
            // to new manager (direct members only; manual overrides untouched;
            // exclude new manager itself).
            name: "write",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        // ── R17: ACL ───────────────────────────────────────────────────────
        OdooMethod {
            // Non-HR users: child_of(managed_departments, parent_path).
            // Returns ids the current user can read.
            name: "_has_read_access",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["name", "parent_id.complete_name"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["parent_path"],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "department parent_id must not create a cycle (raises UserError)",
        source_method: Some("_check_parent_id"),
    }],
    provenance: OdooProvenance {
        l_doc: "L14-HR-BASE.md",
        l_doc_lines: (41, 78),
        odoo_source: &[OdooSourceRef {
            path: "hr/models/hr_department.py",
            line_range: (1, 243),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 3. hr.job ───────────────────────────────────────────────────────────────
//
// Role/position within a department.  Tracks headcount: no_of_employee (active
// headcount) + no_of_recruitment = expected_employees.  SQL unique constraint
// on (name, company_id, department_id) (R7).

/// `hr.job` — job position with headcount tracking.
///
/// L-doc R7; source: `hr/models/hr_job.py:1–94` (full read).
pub const HR_JOB: OdooEntity = OdooEntity {
    model_name: "hr.job",
    description: "Job role / position within a department; tracks active headcount \
                  (no_of_employee) and open recruitment slots (no_of_recruitment); \
                  UNIQUE(name, company_id, department_id).",
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
            name: "department_id",
            kind: OdooFieldKind::Many2one,
            target: Some("hr.department"),
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
        OdooField {
            name: "no_of_employee",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: Some("_compute_employees"),
            depends: &["employee_ids.active", "employee_ids.job_id"],
            // COUNT of active employees in this job (R7).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "no_of_recruitment",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            // Open recruitment slots; ≥ 0 constraint (R7).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "expected_employees",
            kind: OdooFieldKind::Integer,
            target: None,
            required: false,
            computed: Some("_compute_employees"),
            depends: &["employee_ids.active", "employee_ids.job_id", "no_of_recruitment"],
            // = no_of_employee + no_of_recruitment (R7).
            semantic_role: OdooSemanticRole::Quantity,
        },
        OdooField {
            name: "employee_ids",
            kind: OdooFieldKind::One2many,
            target: Some("hr.employee"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[OdooMethod {
        name: "_compute_employees",
        kind: OdooMethodKind::Compute,
        return_kind: OdooReturnKind::Unit,
        triggers: &[],
    }],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["employee_ids.active", "employee_ids.job_id"],
    }],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Sql,
            condition: "UNIQUE(name, company_id, department_id) — no duplicate job name \
                        per company/department combination (R7)",
            source_method: None,
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "no_of_recruitment >= 0 (R7)",
            source_method: Some("_check_recruitment"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "L14-HR-BASE.md",
        l_doc_lines: (47, 49),
        odoo_source: &[OdooSourceRef {
            path: "hr/models/hr_job.py",
            line_range: (1, 94),
        }],
        confidence: OdooConfidence::Curated,
    },
};

// ─── 4. hr.contract.type ─────────────────────────────────────────────────────
//
// Thin community stub (23 lines source): just a name + country_id.
// In the payroll engine it is the contract-type discriminant (CDI/CDD/interim
// etc.); structure_type_id (hr.payroll.structure.type) is the payroll-ruleset
// discriminant — both are data hooks for the woa-rs fresh engine (R14/R15).
//
// NOTE: hr.payroll.structure.type (19 lines, Enterprise boundary) is NOT
// projected as a separate entity — it is an Enterprise stub with no community
// fields beyond name.  Its role (payroll ruleset discriminant, country-matched
// default logic R15) is documented here as a reference note.

/// `hr.contract.type` — employment contract-type discriminant.
///
/// L-doc R14; source: `hr/models/hr_contract_type.py:1–23` (full read).
///
/// Community stub: `name` + `country_id`.  Acts as discriminant for
/// contract-type logic (CDI/CDD/interim) in the woa-rs payroll engine.
///
/// **Payroll boundary:** `hr.payroll.structure.type` (Enterprise-only;
/// `hr_payroll` absent from community) is NOT projected here — it carries the
/// `country_id`-matched payroll-ruleset default (R15) but is an Enterprise stub
/// with no meaningful community fields.  The fresh payroll engine in woa-rs
/// will resolve structure_type_id from `contract.structure_type_id` directly.
pub const HR_CONTRACT_TYPE: OdooEntity = OdooEntity {
    model_name: "hr.contract.type",
    description: "Employment contract type (e.g. CDI / CDD / interim); community stub with \
                  name + country_id; contract-template whitelist (R14) copies this field; \
                  structure_type_id (Enterprise-only) is the payroll-ruleset discriminant \
                  and is absent from community — the woa-rs engine resolves it directly.",
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
            name: "country_id",
            kind: OdooFieldKind::Many2one,
            target: Some("res.country"),
            required: false,
            computed: None,
            depends: &[],
            // Jurisdiction of this contract type; used for country-specific
            // template whitelisting in R14.
            semantic_role: OdooSemanticRole::Reference,
        },
    ],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "L14-HR-BASE.md",
        l_doc_lines: (68, 81),
        odoo_source: &[
            OdooSourceRef {
                path: "hr/models/hr_contract_type.py",
                line_range: (1, 23),
            },
            OdooSourceRef {
                // Referenced in R14/R15 context; Enterprise boundary — absent from community.
                path: "hr/models/hr_payroll_structure_type.py",
                line_range: (1, 19),
            },
        ],
        confidence: OdooConfidence::Curated,
    },
};

// ─── ENTITIES slice ───────────────────────────────────────────────────────────

/// All 4 entities documented in lane L14 (HR base data — employee / org /
/// contract structure).
///
/// Entity index:
///   [0] `hr.employee`       — work-resource with version chain + statutory IDs
///   [1] `hr.department`     — org unit with recursive hierarchy
///   [2] `hr.job`            — role / position with headcount tracking
///   [3] `hr.contract.type`  — employment contract-type discriminant (community stub)
pub const ENTITIES: &[OdooEntity] = &[
    HR_EMPLOYEE,
    HR_DEPARTMENT,
    HR_JOB,
    HR_CONTRACT_TYPE,
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{OdooConfidence, OdooConstraintKind, OdooFieldKind, OdooSemanticRole};

    #[test]
    fn entities_slice_has_four_entries() {
        assert_eq!(ENTITIES.len(), 4);
    }

    #[test]
    fn all_entities_have_curated_confidence() {
        for e in ENTITIES {
            assert_eq!(
                e.provenance.confidence,
                OdooConfidence::Curated,
                "entity {} must be Curated",
                e.model_name,
            );
        }
    }

    #[test]
    fn all_entities_reference_l14_l_doc() {
        for e in ENTITIES {
            assert_eq!(
                e.provenance.l_doc,
                "L14-HR-BASE.md",
                "entity {} must reference L14 l_doc",
                e.model_name,
            );
        }
    }

    #[test]
    fn hr_employee_identity() {
        assert_eq!(HR_EMPLOYEE.model_name, "hr.employee");
        assert!(HR_EMPLOYEE.state_machine.is_none());
        let names: Vec<&str> = HR_EMPLOYEE.fields.iter().map(|f| f.name).collect();
        assert!(names.contains(&"name"), "name field required");
        assert!(names.contains(&"parent_id"), "parent_id (manager) required");
        assert!(names.contains(&"department_id"), "department_id required");
        assert!(names.contains(&"job_id"), "job_id required");
        assert!(names.contains(&"work_email"), "work_email required");
        assert!(names.contains(&"identification_id"), "identification_id required");
    }

    #[test]
    fn hr_employee_has_barcode_sql_constraint() {
        let c = HR_EMPLOYEE
            .constraints
            .iter()
            .find(|c| c.kind == OdooConstraintKind::Sql)
            .expect("barcode SQL unique constraint must be present");
        assert!(c.condition.contains("barcode"));
    }

    #[test]
    fn hr_employee_has_version_compute_and_cron() {
        let method_names: Vec<&str> = HR_EMPLOYEE.methods.iter().map(|m| m.name).collect();
        assert!(
            method_names.contains(&"_compute_current_version_id"),
            "_compute_current_version_id must be present"
        );
        assert!(
            method_names.contains(&"_cron_update_current_version_id"),
            "_cron_update_current_version_id must be present"
        );
    }

    #[test]
    fn hr_employee_has_work_contact_id_reference() {
        let f = HR_EMPLOYEE
            .fields
            .iter()
            .find(|f| f.name == "work_contact_id")
            .expect("work_contact_id must be present");
        assert_eq!(f.kind, OdooFieldKind::Many2one);
        assert_eq!(f.target, Some("res.partner"));
        assert_eq!(f.semantic_role, OdooSemanticRole::Reference);
    }

    #[test]
    fn hr_employee_newly_hired_field() {
        let f = HR_EMPLOYEE
            .fields
            .iter()
            .find(|f| f.name == "is_new_hire")
            .expect("is_new_hire field must be present");
        assert_eq!(f.kind, OdooFieldKind::Boolean);
        assert_eq!(f.semantic_role, OdooSemanticRole::Status);
        assert!(f.computed.is_some());
    }

    #[test]
    fn hr_department_identity() {
        assert_eq!(HR_DEPARTMENT.model_name, "hr.department");
        assert!(HR_DEPARTMENT.state_machine.is_none());
        let field_names: Vec<&str> = HR_DEPARTMENT.fields.iter().map(|f| f.name).collect();
        assert!(field_names.contains(&"name"));
        assert!(field_names.contains(&"parent_id"));
        assert!(field_names.contains(&"manager_id"));
        assert!(field_names.contains(&"member_ids"));
    }

    #[test]
    fn hr_department_complete_name_is_computed() {
        let f = HR_DEPARTMENT
            .fields
            .iter()
            .find(|f| f.name == "complete_name")
            .expect("complete_name must be present");
        assert!(f.computed.is_some());
        assert!(f.depends.contains(&"parent_id.complete_name"));
    }

    #[test]
    fn hr_department_has_cycle_constraint() {
        let c = HR_DEPARTMENT
            .constraints
            .iter()
            .find(|c| c.source_method == Some("_check_parent_id"))
            .expect("cycle-check constraint must be present");
        assert_eq!(c.kind, OdooConstraintKind::Python);
    }

    #[test]
    fn hr_job_identity() {
        assert_eq!(HR_JOB.model_name, "hr.job");
        assert!(HR_JOB.state_machine.is_none());
    }

    #[test]
    fn hr_job_headcount_fields() {
        let field_names: Vec<&str> = HR_JOB.fields.iter().map(|f| f.name).collect();
        assert!(field_names.contains(&"no_of_employee"), "no_of_employee required");
        assert!(field_names.contains(&"no_of_recruitment"), "no_of_recruitment required");
        assert!(field_names.contains(&"expected_employees"), "expected_employees required");
        let emp = HR_JOB
            .fields
            .iter()
            .find(|f| f.name == "no_of_employee")
            .unwrap();
        assert_eq!(emp.kind, OdooFieldKind::Integer);
        assert_eq!(emp.semantic_role, OdooSemanticRole::Quantity);
    }

    #[test]
    fn hr_job_has_sql_unique_constraint() {
        let c = HR_JOB
            .constraints
            .iter()
            .find(|c| c.kind == OdooConstraintKind::Sql)
            .expect("SQL unique constraint must be present");
        assert!(c.condition.contains("UNIQUE"));
    }

    #[test]
    fn hr_contract_type_identity() {
        assert_eq!(HR_CONTRACT_TYPE.model_name, "hr.contract.type");
        assert!(HR_CONTRACT_TYPE.state_machine.is_none());
        assert!(HR_CONTRACT_TYPE.methods.is_empty());
        // Community stub: only name + country_id.
        assert_eq!(HR_CONTRACT_TYPE.fields.len(), 2);
    }

    #[test]
    fn hr_contract_type_name_is_identity() {
        let f = HR_CONTRACT_TYPE
            .fields
            .iter()
            .find(|f| f.name == "name")
            .expect("name field must be present");
        assert_eq!(f.kind, OdooFieldKind::Char);
        assert!(f.required);
        assert_eq!(f.semantic_role, OdooSemanticRole::Identity);
    }
}
