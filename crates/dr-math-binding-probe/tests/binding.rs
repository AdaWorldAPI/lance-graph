use dr_math_binding_probe::{
    Axis, AxisKind, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Population, Refusal, Scope, Truth, bind_dr64,
};
use lance_graph_contract::morton8x8::Morton8x8;

fn id() -> Axis {
    Axis {
        class: 12,
        kind: AxisKind::Identity,
    }
}
fn property() -> Axis {
    Axis {
        class: 12,
        kind: AxisKind::Property,
    }
}
fn revisions() -> FrameRevisions {
    FrameRevisions {
        source: 7,
        relation_map: 11,
        coordinates: 3,
    }
}
const FOCUS: Focus = Focus { revision: 5 };
const SCOPE: Scope = Scope { id: 100 };
fn guards(store: &FrameStore) -> Guards {
    Guards {
        frame: store.seal(),
        focus: FOCUS,
        scope: SCOPE,
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
fn two_valued(rows: &[u64; 64]) -> Operand<'_> {
    Operand {
        rows,
        validity: None,
    }
}

#[test]
fn bind_borrows_exact_bytes_no_materialization() {
    let store = FrameStore::new(revisions());
    let rows = bits();
    let bound = bind_dr64(
        &store,
        signature(),
        signature(),
        &guards(&store),
        (FOCUS, SCOPE),
        two_valued(&rows),
    )
    .unwrap();
    assert!(std::ptr::eq(bound.rows, &rows));
    assert_eq!(bound.rows[17], 1u64 << 29);
    assert_eq!(std::mem::size_of_val(&rows), 512);
}

#[test]
fn physical_equal_identity_squared_and_identity_property_are_not_interchangeable() {
    let store = FrameStore::new(revisions());
    let rows = bits();
    let identity_squared = MethodSignature::presence(12, 1, id(), id());
    assert_eq!(
        bind_dr64(
            &store,
            signature(),
            identity_squared,
            &guards(&store),
            (FOCUS, SCOPE),
            two_valued(&rows)
        )
        .unwrap_err(),
        Refusal::AxisChanged
    );
}

#[test]
fn method_is_not_just_a_class_semantics() {
    let store = FrameStore::new(revisions());
    let rows = bits();
    let other_method = MethodSignature::presence(12, 2, id(), property());
    assert_eq!(
        bind_dr64(
            &store,
            signature(),
            other_method,
            &guards(&store),
            (FOCUS, SCOPE),
            two_valued(&rows)
        )
        .unwrap_err(),
        Refusal::MethodChanged
    );
}

/// Validity travels with R and is typed by R's method: a three-valued method
/// needs it, a two-valued one must not carry it. There is no second method a
/// validity plane could be bound to, so a "validity of a foreign method"
/// cannot be written.
#[test]
fn validity_is_a_carrier_of_the_relation_typed_by_its_method() {
    let store = FrameStore::new(revisions());
    let rows = bits();
    let known = bits();
    let mut k3 = signature();
    k3.truth = Truth::ThreeValued;
    let bind = |m: MethodSignature, operand| {
        bind_dr64(&store, m, m, &guards(&store), (FOCUS, SCOPE), operand)
    };
    assert_eq!(
        bind(k3, two_valued(&rows)).unwrap_err(),
        Refusal::ThreeValuedNeedsValidity
    );
    let with_validity = Operand {
        rows: &rows,
        validity: Some(&known),
    };
    let bound = bind(k3, with_validity).unwrap();
    assert!(std::ptr::eq(bound.validity.unwrap(), &known));
    assert_eq!(
        bind(signature(), with_validity).unwrap_err(),
        Refusal::TwoValuedHasNoValidity
    );
    let mut bag = signature();
    bag.population = Population::Bag;
    assert_eq!(
        bind(bag, two_valued(&rows)).unwrap_err(),
        Refusal::BagNeedsMultiplicity
    );
}

/// Each moved revision is its own refusal, and a re-seal at equal numbers is
/// still a new frame.
#[test]
fn every_frame_revision_is_a_separate_refusal() {
    let rows = bits();
    let r = revisions();
    let cases = [
        (FrameRevisions { source: 8, ..r }, Refusal::SourceChanged),
        (
            FrameRevisions {
                relation_map: 12,
                ..r
            },
            Refusal::MapChanged,
        ),
        (
            FrameRevisions {
                coordinates: 4,
                ..r
            },
            Refusal::CoordinatesChanged,
        ),
        (r, Refusal::FrameResealed),
    ];
    for (moved, expected) in cases {
        let mut store = FrameStore::new(r);
        let bound_at = guards(&store);
        store.advance(moved);
        assert_eq!(
            bind_dr64(
                &store,
                signature(),
                signature(),
                &bound_at,
                (FOCUS, SCOPE),
                two_valued(&rows)
            )
            .unwrap_err(),
            expected
        );
    }
}

/// The seal is store-issued identity, not equality of numbers: a frame of a
/// second store at the same revisions does not bind against the first.
#[test]
fn two_stores_at_equal_revisions_do_not_alias() {
    let rows = bits();
    let first = FrameStore::new(revisions());
    let second = FrameStore::new(revisions());
    assert_eq!(first.revisions(), second.revisions());
    assert!(!first.seal().same_issue(&second.seal()));
    assert!(first.seal().same_issue(&first.seal()));
    assert_eq!(
        bind_dr64(
            &first,
            signature(),
            signature(),
            &guards(&second),
            (FOCUS, SCOPE),
            two_valued(&rows)
        )
        .unwrap_err(),
        Refusal::ForeignFrame
    );
}

/// Focus is motion and scope is policy: each is its own guard, and moving
/// either leaves the frame valid.
#[test]
fn focus_and_scope_are_guards_apart_from_the_frame() {
    let store = FrameStore::new(revisions());
    let rows = bits();
    let bound_at = guards(&store);
    let bind = |now| {
        bind_dr64(
            &store,
            signature(),
            signature(),
            &bound_at,
            now,
            two_valued(&rows),
        )
    };
    assert_eq!(
        bind((Focus { revision: 6 }, SCOPE)).unwrap_err(),
        Refusal::FocusChanged
    );
    assert_eq!(
        bind((FOCUS, Scope { id: 101 })).unwrap_err(),
        Refusal::ScopeChanged
    );
    assert_eq!(store.check(&bound_at.frame), Ok(()));
    assert!(bind((FOCUS, SCOPE)).is_ok());
}

#[test]
fn cartesian_global_coordinate_is_not_a_foreign_relation_proof() {
    let code = Morton8x8::from_xy(17, 29);
    assert_eq!((code.x(), code.y()), (17, 29));
    // Cartesian addressability is exact; occupancy and semantics still
    // require a DR relationship bit and a ClassView domain binding.
    let store = FrameStore::new(revisions());
    let rows = bits();
    let b = bind_dr64(
        &store,
        signature(),
        signature(),
        &guards(&store),
        (FOCUS, SCOPE),
        two_valued(&rows),
    )
    .unwrap();
    assert_eq!(b.rows[17] & (1u64 << 29), 1u64 << 29);
}
