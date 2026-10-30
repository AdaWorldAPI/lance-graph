//! The bag-multiplicity operation stated as a DTO binding over coordinates DR
//! has already routed (EXO-50 message 19): which identities, which property
//! mappings, which method by reference, and the expected values.
//!
//! The stages stay separate and in order:
//!
//! 1. **DR routes.** Lab ids map to local slots (`AddressMap`) under a sealed
//!    frame (`FrameStore`). Dimensions are DR's; nothing here chooses one.
//! 2. **DT64 binds meaning.** Identities are `NodeGuid`s; the relations R and
//!    S are property mappings named by a `ClassView` field set and selected by
//!    a `WideFieldMask`; the counting method is named by
//!    `(owner_class, method_ordinal)` with `Population::Bag`, never embedded.
//! 3. **Execution** binds the registers. For a Bag method it refuses today
//!    (`Refusal::BagNeedsMultiplicity`): that is the failing half.
//!
//! No DT64 bind store exists yet (D-DT64-6 is PROPOSED). `Binding` below is a
//! test-local value showing what XLOG would hand one; it is not a store, it
//! persists nothing and it copies no business data (identities and field
//! positions only).

use dr_math_binding_probe::{
    Axis, AxisKind, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Population, Refusal, Scope, bind_dr64,
};
use lance_graph_contract::canonical_node::{NodeGuid, TailVariant};
use lance_graph_contract::class_view::{ClassId, ClassView, FieldMask, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use xlog_logic::parse_program;
use xlog_relcomp_probe::{AddressMap, SLOTS, facts, register};

const MULT: &str = "pred r(u32, u32).
pred s(u32, u32).
pred q(u32).
r(1, 2). r(5, 2). r(1, 3). r(9, 7).
s(2, 4). s(2, 6). s(3, 4). s(7, 8).
q(Y) :- r(X, Y), s(Y, Z).
?- q(Y).
";

/// Lab stand-ins. A real binding takes these from the OGAR codebook through
/// hotplug; they are not minted concepts.
const LAB_CLASS: ClassId = 0x7F01;
const LAB_CLASSID: u32 = (LAB_CLASS as u32) << 16;
const COUNT_METHOD: u16 = 1;

const X: Axis = Axis {
    class: 10,
    kind: AxisKind::Identity,
};
const Y: Axis = Axis {
    class: 11,
    kind: AxisKind::Property,
};
const Z: Axis = Axis {
    class: 12,
    kind: AxisKind::Community,
};
const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 2 };

/// The class's property mappings: field 0 is R, field 1 is S.
struct LabClassView {
    fields: Vec<FieldRef>,
}

impl LabClassView {
    fn new() -> Self {
        Self {
            fields: vec![
                FieldRef::new("lab:xlog/r", "r"),
                FieldRef::new("lab:xlog/s", "s"),
            ],
        }
    }
}

impl ClassView for LabClassView {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == LAB_CLASS {
            &self.fields
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

/// A lab id on one axis, as a `NodeGuid` reference (V3 tail; the axis class
/// sits in `family`, the lab id in `identity`).
fn guid(axis: Axis, id: u32) -> NodeGuid {
    NodeGuid::mint_for(
        TailVariant::V3,
        LAB_CLASSID,
        0,
        0,
        0,
        0,
        u32::from(axis.class),
        id,
    )
}

/// DR's output: local slots under a sealed frame.
struct Routed {
    xs: AddressMap,
    ys: AddressMap,
    zs: AddressMap,
    guards: Guards,
}

fn route(store: &FrameStore, r: &[(u32, u32)], s: &[(u32, u32)]) -> Routed {
    Routed {
        xs: AddressMap::new(X, r.iter().map(|t| t.0)).unwrap(),
        ys: AddressMap::new(Y, r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))).unwrap(),
        zs: AddressMap::new(Z, s.iter().map(|t| t.1)).unwrap(),
        guards: Guards {
            frame: store.seal(),
            focus: FOCUS,
            scope: SCOPE,
        },
    }
}

/// What DT64 records for the operation: identities by reference, the
/// selected property mappings, the method by reference, and the expected
/// values. It is only built against a current route (`bind` checks the frame).
struct Binding {
    answer_identities: Vec<NodeGuid>,
    class: ClassId,
    selected: WideFieldMask,
    method: MethodSignature,
    expected: Vec<(NodeGuid, u64)>,
}

#[derive(Debug, PartialEq, Eq)]
enum BindRefusal {
    /// DT64 never binds against a route that is no longer current.
    Route(Refusal),
}

fn bind(store: &FrameStore, routed: &Routed) -> Result<Binding, BindRefusal> {
    store
        .check(&routed.guards.frame)
        .map_err(BindRefusal::Route)?;
    let id_of = |slot: usize| routed.ys.ids(1u64 << slot)[0];
    Ok(Binding {
        answer_identities: (0..routed.ys.len()).map(|i| guid(Y, id_of(i))).collect(),
        class: LAB_CLASS,
        selected: WideFieldMask::from_positions(&[0, 1]),
        method: MethodSignature {
            population: Population::Bag,
            ..MethodSignature::presence(LAB_CLASS, COUNT_METHOD, Y, Y)
        },
        // Per-Y derivation counts; see tests/litho.rs for the two oracles.
        expected: vec![(guid(Y, 2), 4), (guid(Y, 3), 1), (guid(Y, 7), 1)],
    })
}

type Tuples = Vec<(u32, u32)>;

fn fixture() -> (Tuples, Tuples) {
    let p = parse_program(MULT).unwrap();
    (facts(&p, "r").unwrap(), facts(&p, "s").unwrap())
}

#[test]
fn the_binding_names_identities_property_mappings_and_a_method() {
    let (r, s) = fixture();
    let store = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let routed = route(&store, &r, &s);
    let b = bind(&store, &routed).unwrap();

    // Identities: references that decode back to the lab ids, in slot order.
    let ids: Vec<(u32, u16)> = b
        .answer_identities
        .iter()
        .map(|g| (g.classid(), g.identity_v2()))
        .collect();
    assert_eq!(
        ids,
        vec![(LAB_CLASSID, 2), (LAB_CLASSID, 3), (LAB_CLASSID, 7)]
    );

    // Property mappings: R and S, resolved through the ClassView, not stored.
    let view = LabClassView::new();
    assert!(b.selected.has(0) && b.selected.has(1) && !b.selected.has(2));
    let rows: Vec<&str> = view
        .render_rows(b.class, FieldMask::from_positions(&[0, 1]))
        .into_iter()
        .map(|r| r.label)
        .collect();
    assert_eq!(rows, vec!["r", "s"]);

    // Method: a reference, Bag population; nothing about how it counts.
    assert_eq!(
        (b.method.owner_class, b.method.method_ordinal),
        (LAB_CLASS, COUNT_METHOD)
    );
    assert_eq!(b.method.population, Population::Bag);

    // Expected values, keyed by identity.
    assert_eq!(b.expected.len(), b.answer_identities.len());
    assert!(
        b.expected
            .iter()
            .zip(&b.answer_identities)
            .all(|((g, _), a)| g == a)
    );
}

#[test]
fn dt64_does_not_bind_against_a_stale_route() {
    let (r, s) = fixture();
    let mut store = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let routed = route(&store, &r, &s);
    store.advance(FrameRevisions {
        source: 1,
        relation_map: 2,
        coordinates: 1,
    });
    assert_eq!(
        bind(&store, &routed).err(),
        Some(BindRefusal::Route(Refusal::MapChanged))
    );
}

/// The failing half: executing the bound Bag method is refused today, so the
/// expected counts cannot be produced from the register.
#[test]
fn executing_the_bound_bag_method_is_refused_today() {
    let (r, s) = fixture();
    let store = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let routed = route(&store, &r, &s);
    let b = bind(&store, &routed).unwrap();
    let rr = register(&r, &routed.xs, &routed.ys).unwrap();
    let _ss = register(&s, &routed.ys, &routed.zs).unwrap();
    let e = bind_dr64(
        &store,
        b.method,
        b.method,
        &routed.guards,
        (FOCUS, SCOPE),
        Operand {
            rows: &rr,
            validity: None,
        },
    );
    assert_eq!(e.err(), Some(Refusal::BagNeedsMultiplicity));
    assert_eq!(rr.len(), SLOTS);
}
