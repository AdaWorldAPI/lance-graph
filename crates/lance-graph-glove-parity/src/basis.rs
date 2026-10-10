//! The canonical basis for cross-glove parity, read from the authority
//! (D-XGP-2, `.claude/plans/cross-glove-business-parity-v1.md` §C.9).
//!
//! Lockstep allocation is deprecated: each glove declares a [`HotPlug`] and
//! `OgarAuthority` resolves it. The fields a glove maps onto are the
//! authority's `ClassView` of the promoted `BillableWorkEntry` class
//! ([`canonical_fields`]); nothing here mints a concept, a field or a URI.
//!
//! [`SAP_FIELDS`] and [`ODOO_FIELDS`] record each native field's claimed
//! canonical position with a [`Grade`]. Every claim is `Hypothesized` today:
//! no conversion onto an edge target (`Worker`, `Duration`, `Tenant`, …) is
//! implemented and tested yet (plan §C.6.2), so none of them may bind.
use lance_graph_contract::hotplug::HotPlug;
use lance_graph_contract::ClassView;
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_ogar::OgarClassView;

/// The concepts both gloves plug.
pub const CONCEPTS: &[u16] = &[BILLABLE_WORK_ENTRY];

/// SAP CATS. No capabilities are covered: the authority declares none on
/// `0x0103` yet, so activation refuses with `NoCapabilitiesFor`.
pub const SAP_HOT_PLUG: HotPlug = HotPlug {
    consumer: "sap-glove",
    classids: CONCEPTS,
    covered: &[],
};

/// Odoo `account.analytic.line`. Same state as [`SAP_HOT_PLUG`].
pub const ODOO_HOT_PLUG: HotPlug = HotPlug {
    consumer: "odoo-glove",
    classids: CONCEPTS,
    covered: &[],
};

/// How well a native field is known to carry a canonical field (plan §C.6.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grade {
    /// The native field is the canonical value, unchanged.
    Exact,
    /// A conversion exists and a test proves it on real rows.
    Converted,
    /// Plausible, unproven. Never binds.
    Hypothesized,
}

/// One native field's claimed canonical position.
#[derive(Clone, Copy, Debug)]
pub struct FieldMap {
    pub native: &'static str,
    /// A `predicate_iri` of the authority's `ClassView` for `0x0103`.
    pub canonical: &'static str,
    pub grade: Grade,
    /// What a `Converted` grade would have to prove.
    pub gap: &'static str,
}

const fn hyp(native: &'static str, canonical: &'static str, gap: &'static str) -> FieldMap {
    FieldMap {
        native,
        canonical,
        grade: Grade::Hypothesized,
        gap,
    }
}

/// SAP CATS DTO fields (`lance-graph-sap::schema::FIELDS`) onto
/// `BillableWorkEntry`. `work_date_utc` has no row: the class has no temporal
/// role (plan §C.9 point 2).
pub const SAP_FIELDS: &[FieldMap] = &[
    hyp(
        "employee_number",
        "performed_by",
        "a PERNR identifies a Worker",
    ),
    hyp(
        "project_code",
        "project",
        "a PS element identifies a Project; optional, CatsBinder refuses it",
    ),
    hyp(
        "task_code",
        "about",
        "an order identifies a ProjectWorkItem; optional, CatsBinder refuses it",
    ),
    hyp(
        "hours_logged",
        "duration",
        "the Duration concept reads a decimal hour quantity",
    ),
    hyp(
        "billing_indicator",
        "billable",
        "the indicator's values map onto true/false",
    ),
    hyp(
        "tenant_id",
        "tenant",
        "the source tenant is the canonical Tenant",
    ),
];

/// Base `account.analytic.line` fields (`lance-graph-ontology`
/// `odoo_blueprint/extracted/analytic.rs`) onto `BillableWorkEntry`. The
/// base model has no project or task field (those come from `hr_timesheet`,
/// not harvested), and `date` has no row for the same reason as SAP's.
pub const ODOO_FIELDS: &[FieldMap] = &[
    hyp("user_id", "performed_by", "a login user is an HR employee"),
    hyp("unit_amount", "duration", "the product UoM is a time unit"),
    hyp("company_id", "tenant", "a company is the canonical Tenant"),
];

/// The authority's field names for `BillableWorkEntry`, in `ClassView` order.
#[must_use]
pub fn canonical_fields() -> Vec<String> {
    OgarClassView::new()
        .fields(BILLABLE_WORK_ENTRY)
        .iter()
        .map(|f| f.predicate_iri.clone())
        .collect()
}
