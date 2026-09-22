use lance_graph_contract::class_view::{ClassView, WideFieldMask};
use lance_graph_sap::schema::CatsSchema;

#[test]
fn sap_and_real_odoo_view_inputs_use_the_same_field_abi() {
    let schema = CatsSchema::new(42, 0);
    let sap = schema
        .realize_projection(&["EmployeeNumber", "ActivityType", "HoursLogged"])
        .unwrap();
    assert_eq!(sap, WideFieldMask::from_positions(&[5, 10, 11]));
    assert_eq!(
        sap,
        schema
            .realize_projection(&["employee_number", "activity_type", "hours_logged"])
            .unwrap()
    );
    assert!(schema.realize_projection(&["not_a_field"]).is_none());
    assert_eq!(schema.fields(42)[5].label, "employee_number");

    // Odoo ef03731c, source blobs verified by tools/harvest_odoo_view.py.
    // This executes the shared contract on harvested inputs; it is not a claim
    // to have compiled the entire Odoo/OGAR dependency graph.
    let fields: Vec<_> = include_str!("../fixtures/odoo-view.tsv")
        .lines()
        .skip(1)
        .map(|l| l.split_once('\t').unwrap())
        .collect();
    let universe: Vec<_> = fields.iter().map(|(name, _)| *name).collect();
    let present: Vec<_> = fields
        .iter()
        .filter(|(_, p)| *p == "1")
        .map(|(name, _)| *name)
        .collect();
    let odoo = WideFieldMask::from_universe_present(&universe, &present).unwrap();
    assert_eq!(universe.len(), 216);
    assert_eq!(odoo.count(), 57);
    for (i, (_, present)) in fields.iter().enumerate() {
        assert_eq!(odoo.has(i as u8), *present == "1");
    }
    assert!((64..216).any(|i| odoo.has(i)));
    assert_eq!(odoo.intersect(&WideFieldMask::full_for(216)), odoo);
}
