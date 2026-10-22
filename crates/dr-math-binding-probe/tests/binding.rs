use dr_math_binding_probe::{
    bind_dr64, Axis, AxisKind, BindingWitness, MethodSignature, Population, Refusal, Truth,
};
use lance_graph_contract::morton8x8::Morton8x8;

fn id() -> Axis {
    Axis { class: 12, kind: AxisKind::Identity }
}
fn property() -> Axis {
    Axis { class: 12, kind: AxisKind::Property }
}
fn witness() -> BindingWitness {
    BindingWitness {
        source_revision: 7,
        relation_map_revision: 11,
        coordinate_revision: 3,
        focus_revision: 5,
        authorization_scope: 100,
    }
}
fn signature() -> MethodSignature {
    MethodSignature::presence(12, 1, id(), property())
}
fn bits() -> [u64; 64] {
    let mut rows = [0u64; 64];
    rows[17] |= 1u64 << 29;
    rows
}

#[test]
fn bind_borrows_exact_bytes_no_materialization() {
    let rows = bits();
    let bound = bind_dr64(signature(), signature(), witness(), witness(), &rows).unwrap();
    assert!(std::ptr::eq(bound.rows, &rows));
    assert_eq!(bound.rows[17], 1u64 << 29);
    assert_eq!(std::mem::size_of_val(&rows), 512);
}
#[test]
fn physical_equal_identity_squared_and_identity_property_are_not_interchangeable() {
    let rows = bits();
    let identity_squared = MethodSignature::presence(12, 1, id(), id());
    assert_eq!(
        bind_dr64(signature(), identity_squared, witness(), witness(), &rows).unwrap_err(),
        Refusal::AxisChanged
    );
}
#[test]
fn method_is_not_just_a_class_semantics() {
    let rows = bits();
    let other_method = MethodSignature::presence(12, 2, id(), property());
    assert_eq!(bind_dr64(signature(), other_method, witness(), witness(), &rows).unwrap_err(), Refusal::MethodChanged);
}
#[test]
fn null_and_bag_require_separate_carriers() {
    let rows = bits();
    let mut k3 = signature();
    k3.truth = Truth::ThreeValued;
    assert_eq!(bind_dr64(k3, k3, witness(), witness(), &rows).unwrap_err(), Refusal::ThreeValuedNeedsValidity);
    let mut bag = signature();
    bag.population = Population::Bag;
    assert_eq!(bind_dr64(bag, bag, witness(), witness(), &rows).unwrap_err(), Refusal::BagNeedsMultiplicity);
}
#[test]
fn every_witness_component_is_a_separate_refusal() {
    let rows = bits();
    let w = witness();
    let cases = [
        (BindingWitness { source_revision: 8, ..w }, Refusal::SourceChanged),
        (BindingWitness { relation_map_revision: 12, ..w }, Refusal::MapChanged),
        (BindingWitness { coordinate_revision: 4, ..w }, Refusal::CoordinatesChanged),
        (BindingWitness { focus_revision: 6, ..w }, Refusal::FocusChanged),
        (BindingWitness { authorization_scope: 101, ..w }, Refusal::ScopeChanged),
    ];
    for (changed, expected) in cases {
        assert_eq!(bind_dr64(signature(), signature(), w, changed, &rows).unwrap_err(), expected);
    }
}
#[test]
fn cartesian_global_coordinate_is_not_a_foreign_relation_proof() {
    let code = Morton8x8::from_xy(17, 29);
    assert_eq!((code.x(), code.y()), (17, 29));
    // Cartesian addressability is exact; occupancy and semantics still
    // require a DR relationship bit and a ClassView domain binding.
    let rows = bits();
    let b = bind_dr64(signature(), signature(), witness(), witness(), &rows).unwrap();
    assert_eq!(b.rows[17] & (1u64 << 29), 1u64 << 29);
}
