//! D-XGP-2: the hot-plugs and the canonical basis, against the authority.
use lance_graph_contract::hotplug::{ActivationDrift, CapabilityAuthority};
use lance_graph_glove_parity::basis::{
    canonical_fields, Grade, ODOO_FIELDS, ODOO_HOT_PLUG, SAP_FIELDS, SAP_HOT_PLUG,
};
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_ogar::OgarAuthority;
use lance_graph_sap::schema::FIELDS;

#[test]
fn both_plugs_refuse_until_the_authority_declares_capabilities_on_0x0103() {
    for plug in [SAP_HOT_PLUG, ODOO_HOT_PLUG] {
        assert_eq!(
            OgarAuthority.activate(&plug).map(|_| ()),
            Err(ActivationDrift::NoCapabilitiesFor(BILLABLE_WORK_ENTRY)),
            "{}: OGAR now declares capabilities on 0x0103; write `covered`",
            plug.consumer
        );
    }
}

#[test]
fn the_canonical_basis_is_the_promoted_class() {
    assert_eq!(
        canonical_fields(),
        [
            "billable",
            "project",
            "about",
            "performed_by",
            "duration",
            "priced_by",
            "cost_center",
            "classified_by",
            "materializes_as",
            "approval_state",
            "tenant",
            "audit_trail",
            "posted_by",
        ]
    );
}

#[test]
fn every_claim_names_a_canonical_field_and_a_real_native_field() {
    let canonical = canonical_fields();
    let sap: Vec<_> = FIELDS.iter().map(|f| f.technical_name).collect();
    for m in SAP_FIELDS {
        assert!(
            canonical.iter().any(|c| c == m.canonical),
            "sap {}",
            m.canonical
        );
        assert!(sap.contains(&m.native), "sap native {}", m.native);
    }
    // The harvested base model's own field names.
    let odoo = [
        "name",
        "date",
        "amount",
        "unit_amount",
        "product_uom_id",
        "partner_id",
        "user_id",
        "company_id",
        "currency_id",
    ];
    for m in ODOO_FIELDS {
        assert!(
            canonical.iter().any(|c| c == m.canonical),
            "odoo {}",
            m.canonical
        );
        assert!(odoo.contains(&m.native), "odoo native {}", m.native);
    }
}

#[test]
fn no_claim_is_better_than_hypothesized_and_no_native_field_maps_twice() {
    for table in [SAP_FIELDS, ODOO_FIELDS] {
        assert!(table
            .iter()
            .all(|m| m.grade == Grade::Hypothesized && !m.gap.is_empty()));
        let mut natives: Vec<_> = table.iter().map(|m| m.native).collect();
        natives.sort_unstable();
        natives.dedup();
        assert_eq!(natives.len(), table.len());
    }
}

#[test]
fn the_class_has_no_temporal_field_so_dates_stay_unmapped() {
    let canonical = canonical_fields();
    for word in ["date", "day", "period", "time"] {
        assert!(
            !canonical.iter().any(|c| c.contains(word)),
            "{word}: re-map work dates"
        );
    }
    assert!(!SAP_FIELDS.iter().any(|m| m.native == "work_date_utc"));
    assert!(!ODOO_FIELDS.iter().any(|m| m.native == "date"));
}
