//! D-XGP-2: the hot-plugs and the canonical basis, against the authority.
use lance_graph_contract::hotplug::{ActivationDrift, CapabilityAuthority};
use lance_graph_glove_parity::basis::{
    canonical_fields, Grade, ANCHORS, ODOO_FIELDS, ODOO_HOT_PLUG, SAP_FIELDS, SAP_HOT_PLUG,
    WOA_FIELDS, WOA_HOURS_TABLE, WOA_PINNED_TABLE,
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
    // WoA `TimeSheet`'s own field names (OGAR `.claude/harvest/woa-rs/models.py`).
    let woa = [
        "id",
        "tenant_id",
        "source",
        "datum",
        "minuten",
        "erfasst_von",
        "startzeit",
        "endzeit",
        "beschreibung",
        "timer_start",
        "timer_paused_at",
        "abgerechnet",
        "created_at",
        "updated_at",
        "customer",
        "user",
    ];
    for m in WOA_FIELDS {
        assert!(
            canonical.iter().any(|c| c == m.canonical),
            "woa {}",
            m.canonical
        );
        assert!(woa.contains(&m.native), "woa native {}", m.native);
    }
}

#[test]
fn only_the_proven_claim_is_converted_and_no_native_field_maps_twice() {
    for table in [SAP_FIELDS, ODOO_FIELDS, WOA_FIELDS] {
        assert!(table
            .iter()
            .all(|m| m.grade != Grade::Exact && !m.note.is_empty()));
        let mut natives: Vec<_> = table.iter().map(|m| m.native).collect();
        natives.sort_unstable();
        natives.dedup();
        assert_eq!(natives.len(), table.len());
    }
    let converted: Vec<_> = SAP_FIELDS
        .iter()
        .chain(ODOO_FIELDS)
        .chain(WOA_FIELDS)
        .filter(|m| m.grade == Grade::Converted)
        .map(|m| (m.native, m.canonical))
        .collect();
    assert_eq!(converted, [("billing_indicator", "billable")]);
}

/// A `Converted` SAP claim binds: the CATS binder serves the canonical name
/// as the derived lens the conversion is proven on.
#[test]
fn converted_sap_claims_bind_under_their_canonical_name() {
    use lance_graph_quack::bind::{Binder, FieldKind, TableId};
    use lance_graph_quack::Col;
    use lance_graph_sap::bind::{CatsBatch, BILLABLE};
    use lance_graph_sap::binder::CatsBinder;
    use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};
    let base: Vec<Option<&str>> = include_str!("../../lance-graph-sap/fixtures/cats.txt")
        .lines()
        .map(|v| if v == "\\N" { None } else { Some(v) })
        .collect();
    let input: [Vec<Option<&str>>; FIELD_COUNT] = std::array::from_fn(|i| vec![base[i]; 2]);
    let batch = CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )
    .unwrap();
    let b = CatsBinder::new(&batch);
    for m in SAP_FIELDS.iter().filter(|m| m.grade == Grade::Converted) {
        let f = b.field(TableId(0), m.canonical).expect(m.canonical);
        assert_eq!((f.col, f.kind), (Col(BILLABLE as u16), FieldKind::Code));
    }
}

/// Which `BillableWorkEntry` edge targets have an id in the SHARED codebook.
///
/// This measures shared-codebook coverage only. It does NOT mean the other
/// targets are unaddressable or must wait for a shared mint: a classid is
/// `domain(8) | appid(8) | concept(16)`, and each `domain:appid` hands out its
/// own 64k concept space (plan §C.9.3).
#[test]
fn three_edge_targets_have_a_shared_codebook_id() {
    use lance_graph_ogar::ogar_vocab::{billable_work_entry, canonical_concept_id};
    let snake = |camel: &str| {
        let mut out = String::new();
        for (i, ch) in camel.chars().enumerate() {
            if ch.is_ascii_uppercase() && i > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        }
        out
    };
    let class = billable_work_entry();
    let minted: Vec<_> = class
        .associations
        .iter()
        .filter(|e| {
            e.class_name
                .as_deref()
                .and_then(|t| canonical_concept_id(&snake(t)))
                .is_some()
        })
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(class.associations.len(), 12);
    assert_eq!(minted, ["project", "about", "classified_by"]);
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
    assert!(!WOA_FIELDS.iter().any(|m| m.native == "datum"));
}

/// Each glove claims each anchor exactly once, and claims nothing else.
#[test]
fn every_glove_claims_each_anchor_exactly_once() {
    for (glove, table) in [
        ("sap", SAP_FIELDS),
        ("odoo", ODOO_FIELDS),
        ("woa", WOA_FIELDS),
    ] {
        for anchor in ANCHORS {
            let n = table.iter().filter(|m| m.canonical == *anchor).count();
            assert_eq!(n, 1, "{glove} claims {anchor} {n} times");
        }
    }
    let woa: Vec<_> = WOA_FIELDS.iter().map(|m| m.canonical).collect();
    assert_eq!(woa, ANCHORS);
}

/// `abgerechnet` means "already invoiced", not "billable", so no glove maps
/// it, and WoA claims no `billable` at all.
#[test]
fn woa_invoiced_flag_is_not_billable() {
    assert!(!WOA_FIELDS.iter().any(|m| m.native == "abgerechnet"));
    assert!(!WOA_FIELDS.iter().any(|m| m.canonical == "billable"));
}

/// OGAR's WoA pin sits on the description row, not on the hours row, and the
/// OGIT concepts `ogit.WorkOrder:TimeSheet` / `:User` / `:Tenant` have no
/// `WoaPort` alias yet. Two-sided: when OGAR moves the pin or aliases those
/// concepts, this fails and the WoA claims are re-read.
#[test]
fn woa_pin_is_on_the_description_row_not_the_hours_row() {
    use lance_graph_ogar::ogar_vocab::ports::{PortSpec, WoaPort};
    assert_eq!(
        WoaPort::class_id(WOA_PINNED_TABLE),
        Some(BILLABLE_WORK_ENTRY)
    );
    for unmapped in [WOA_HOURS_TABLE, "User", "Tenant"] {
        assert_eq!(WoaPort::class_id(unmapped), None, "{unmapped}");
    }
}
