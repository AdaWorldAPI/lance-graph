//! The DT64 bind store against the decisions it implements: D-DT64-6 §1 (D4,
//! D5, D6) and §3, D-DRDT §4 (the four gaps) and §5 (the DR #4 falsifier).

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Cmp, Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{
    view_class, AlgebraEntry, AlgebraId, AlgebraRef, Arg, BindStore, ClassSchema, Effect, Guard,
    MethodBinding, MethodId, MethodRef, Refusal, RouteVersion, SchemaRev, SigRev, ValueDemand,
};

/// A ClassView with one class of `n` fields; `n` can be changed to model a
/// schema that moved under the binding.
struct View {
    class: ClassId,
    fields: Vec<FieldRef>,
}

impl View {
    fn with_fields(class: ClassId, n: usize) -> View {
        View {
            class,
            fields: (0..n)
                .map(|i| FieldRef::new(format!("urn:f{i}"), format!("f{i}")))
                .collect(),
        }
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == self.class {
            &self.fields
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

const ORDER: u32 = 0x0A1E_0001;
const ORDER_V2: u32 = 0x0A1E_0002;
/// The lens ORDER is read through: its concept half (`view_class`).
const VIEW_CLASS: ClassId = 0x0A1E;

fn schema(rev: u32, fields: usize) -> ClassSchema {
    ClassSchema {
        schema_rev: SchemaRev(rev),
        field_count: fields,
    }
}

/// The classid-free algebra catalogue the tests bind against (lab ids).
const ASSIGN: AlgebraId = AlgebraId(1);
const ACCUMULATE: AlgebraId = AlgebraId(2);
const GROUP_COUNT: AlgebraId = AlgebraId(10);
const MUL_I64: AlgebraId = AlgebraId(11);

fn entry(id: AlgebraId, polyfill: &str, inputs: u8) -> AlgebraEntry {
    AlgebraEntry {
        id,
        sig_rev: SigRev(1),
        polyfill: polyfill.to_string(),
        inputs,
    }
}

/// A store holding only the algebra catalogue: no classid anywhere.
fn catalogue() -> BindStore {
    let mut s = BindStore::new();
    for e in [
        entry(ASSIGN, "assign", 0),
        entry(ACCUMULATE, "accumulate", 0),
        // (relation field, key column)
        entry(GROUP_COUNT, "ndarray::simd::masked_group_count_u32", 2),
        entry(
            MUL_I64,
            "ndarray::hpc::checked_i64::checked_binary_i64(Mul)",
            2,
        ),
    ] {
        s.declare_algebra(e).unwrap();
    }
    s
}

fn uses(id: AlgebraId) -> AlgebraRef {
    AlgebraRef {
        id,
        sig_rev: SigRev(1),
    }
}

fn effect(target: u8, algebra: &str) -> Effect {
    let id = match algebra {
        "assign" => ASSIGN,
        "accumulate" => ACCUMULATE,
        other => panic!("no lab entry {other}"),
    };
    Effect {
        target,
        algebra: uses(id),
        args: vec![],
    }
}

/// A method fired by a change of field 0, guarded by `state == 1`, writing
/// field 2 then field 3 in the given order.
fn method(id: u32, effects: Vec<Effect>) -> MethodBinding {
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("m{id}"),
        on: WideFieldMask::from_positions(&[0]),
        when: Some(Guard::new(SetExpr::eq_u32(1, 1), Logic::Kleene)),
        reads: WideFieldMask::from_positions(&[0, 1]),
        effects,
        population: Population::Distinct,
        truth: Truth::ThreeValued,
        demand: ValueDemand::Values,
    }
}

fn store_with_order() -> BindStore {
    let mut s = catalogue();
    s.register(
        ORDER,
        schema(1, 4),
        vec![method(
            10,
            vec![effect(2, "assign"), effect(3, "accumulate")],
        )],
    )
    .unwrap();
    s
}

/// One lookup: class schema from the ClassView, method schema from the store,
/// for one classid at one schema revision (D-DRDT §4 gap 4).
#[test]
fn one_lookup_joins_class_schema_and_method_schema() {
    let s = store_with_order();
    let view = View::with_fields(VIEW_CLASS, 4);
    let r = s.resolve(&view, ORDER, SchemaRev(1)).unwrap();
    assert_eq!(r.fields.len(), 4, "fields come from the ClassView");
    assert_eq!(r.fields[2].label, "f2");
    assert_eq!(r.methods.len(), 1);
    assert_eq!(r.methods[0].id, MethodId(10));
    assert_eq!(r.derived_from, None);
}

/// An unregistered classid is UNKNOWN, never "no behaviour" (D-SAG-6 F06).
#[test]
fn an_unregistered_class_is_unknown_not_empty() {
    let s = store_with_order();
    let view = View::with_fields(VIEW_CLASS, 4);
    assert_eq!(
        s.resolve(&view, 0xDEAD_0001, SchemaRev(1)).unwrap_err(),
        Refusal::UnknownClass(0xDEAD_0001)
    );
    // The registered one still resolves, so the refusal is about the id.
    assert!(s.resolve(&view, ORDER, SchemaRev(1)).is_ok());
}

/// D5: a classid's behaviour is permanently immutable. Re-registering the same
/// binding is idempotent; any change is refused, and the old binding stands.
#[test]
fn class_behaviour_is_immutable() {
    let mut s = store_with_order();
    let same = vec![method(
        10,
        vec![effect(2, "assign"), effect(3, "accumulate")],
    )];
    assert_eq!(s.register(ORDER, schema(1, 4), same), Ok(()));

    let reordered = vec![method(
        10,
        vec![effect(3, "accumulate"), effect(2, "assign")],
    )];
    assert_eq!(
        s.register(ORDER, schema(1, 4), reordered),
        Err(Refusal::BehaviourImmutable(ORDER))
    );
    let view = View::with_fields(VIEW_CLASS, 4);
    let r = s.resolve(&view, ORDER, SchemaRev(1)).unwrap();
    assert_eq!(
        r.methods[0].effects[0].target, 2,
        "the first binding stands"
    );
}

/// An override is a new derived classid plus a re-route (D-DT64-6 §3). Only
/// routing is versioned: replay as of a version reads the classid routed
/// then, and that classid's behaviour is the same as today.
#[test]
fn override_is_a_derived_class_and_a_reroute_and_replay_reads_the_old_route() {
    let mut s = store_with_order();
    let v1 = s.route(7, ORDER).unwrap();
    s.register_derived(
        ORDER,
        ORDER_V2,
        schema(1, 4),
        vec![method(
            11,
            vec![effect(3, "accumulate"), effect(2, "assign")],
        )],
    )
    .unwrap();
    let v2 = s.route(7, ORDER_V2).unwrap();
    let v3 = s.retract(7);

    assert_eq!(s.routed_at(RouteVersion(0), 7), Ok(None));
    assert_eq!(s.routed_at(v1, 7), Ok(Some(ORDER)));
    assert_eq!(s.routed_at(v2, 7), Ok(Some(ORDER_V2)));
    assert_eq!(s.routed_at(v3, 7), Ok(None));
    assert_eq!(
        s.routed_at(v2, 8),
        Ok(None),
        "another position never routed"
    );
    assert_eq!(
        s.routed_at(RouteVersion(v3.0 + 1), 7),
        Err(Refusal::FutureVersion {
            asked: RouteVersion(v3.0 + 1),
            current: v3
        })
    );

    let view = View::with_fields(VIEW_CLASS, 4);
    let old = s.resolve_at(&view, v1, 7, SchemaRev(1)).unwrap();
    assert_eq!(old.class, ORDER);
    assert_eq!(old.methods[0].id, MethodId(10), "old behaviour unchanged");
    let new = s.resolve_at(&view, v2, 7, SchemaRev(1)).unwrap();
    assert_eq!(new.class, ORDER_V2);
    assert_eq!(new.derived_from, Some(ORDER));
    assert_eq!(
        s.resolve_at(&view, v3, 7, SchemaRev(1)).unwrap_err(),
        Refusal::Unrouted {
            position: 7,
            version: v3
        }
    );
}

/// Routing refuses a classid the store does not bind, and a derived class
/// needs a bound base.
#[test]
fn routes_and_derivations_need_bound_classes() {
    let mut s = store_with_order();
    assert_eq!(
        s.route(1, 0xBEEF_0001),
        Err(Refusal::UnknownClass(0xBEEF_0001))
    );
    assert_eq!(
        s.register_derived(0xBEEF_0001, ORDER_V2, schema(1, 4), vec![]),
        Err(Refusal::UnknownClass(0xBEEF_0001))
    );
    assert_eq!(
        s.register_derived(ORDER, ORDER, schema(1, 4), vec![]),
        Err(Refusal::BehaviourImmutable(ORDER))
    );
    assert_eq!(
        s.current_version(),
        RouteVersion(0),
        "no refused call moved the log"
    );
}

/// D-DRDT §5, the first DR #4 falsifier: two methods on one cell with the same
/// read and write sets and different effect order stay distinct, and a dropped
/// guard refuses.
#[test]
fn same_read_write_sets_different_effect_order_stay_distinct_and_a_dropped_guard_refuses() {
    let mut s = catalogue();
    let a = method(20, vec![effect(2, "assign"), effect(2, "accumulate")]);
    let b = method(21, vec![effect(2, "accumulate"), effect(2, "assign")]);
    // The negative mold: equal unordered read and write sets.
    assert_eq!(a.reads, b.reads);
    assert_eq!(a.writes(), b.writes());
    assert_ne!(a, b, "effect order is part of the binding");
    s.register(ORDER, schema(1, 4), vec![a.clone(), b.clone()])
        .unwrap();

    let view = View::with_fields(VIEW_CLASS, 4);
    let r = s.resolve(&view, ORDER, SchemaRev(1)).unwrap();
    let on_cell: Vec<MethodId> = r.methods_on(0).map(|m| m.id).collect();
    assert_eq!(
        on_cell,
        vec![MethodId(20), MethodId(21)],
        "both kept, none collapsed"
    );
    assert_eq!(r.methods_on(1).count(), 0, "field 1 triggers nothing");

    // Each faithful reference admits its own binding with its own order.
    let ra = MethodRef::of(ORDER, &a);
    let rb = MethodRef::of(ORDER, &b);
    assert_eq!(s.admit(&ra).unwrap().effects[0].algebra.id, ASSIGN);
    assert_eq!(s.admit(&rb).unwrap().effects[0].algebra.id, ACCUMULATE);

    // A route that dropped the guard witness refuses.
    let dropped = MethodRef {
        guard: None,
        ..ra.clone()
    };
    assert_eq!(
        s.admit(&dropped).unwrap_err(),
        Refusal::GuardDropped {
            class: ORDER,
            method: MethodId(20)
        }
    );
    // A guard that is not the bound one refuses.
    let other = MethodRef {
        guard: Some(Guard::new(SetExpr::eq_u32(1, 2), Logic::Kleene).witness()),
        ..ra.clone()
    };
    assert_eq!(
        s.admit(&other).unwrap_err(),
        Refusal::GuardMismatch {
            class: ORDER,
            method: MethodId(20)
        }
    );
    // A stale signature revision refuses.
    let stale = MethodRef {
        sig_rev: SigRev(0),
        ..ra
    };
    assert_eq!(
        s.admit(&stale).unwrap_err(),
        Refusal::SignatureRevisionMismatch {
            class: ORDER,
            method: MethodId(20),
            bound: SigRev(1),
            asked: SigRev(0)
        }
    );
}

/// A guard on an unguarded method is not a witness either.
#[test]
fn a_guard_added_to_an_unguarded_method_refuses() {
    let mut s = catalogue();
    let free = MethodBinding {
        when: None,
        ..method(30, vec![effect(2, "assign")])
    };
    s.register(ORDER, schema(1, 4), vec![free.clone()]).unwrap();
    assert!(s.admit(&MethodRef::of(ORDER, &free)).is_ok());
    let added = MethodRef {
        guard: Some(Guard::new(SetExpr::All, Logic::TwoValued).witness()),
        ..MethodRef::of(ORDER, &free)
    };
    assert_eq!(
        s.admit(&added).unwrap_err(),
        Refusal::GuardAdded {
            class: ORDER,
            method: MethodId(30)
        }
    );
}

/// WHEN is three-valued: under Kleene logic `x < 5 OR x >= 5` is not the whole
/// population (a NULL row is UNKNOWN, and UNKNOWN never fires), while the
/// two-valued reading of the same expression is. Two-sided, so the witness is
/// neither always-equal nor always-different.
#[test]
fn a_kleene_guard_never_treats_unknown_as_true() {
    let covering = SetExpr::Or(vec![
        SetExpr::i32(4, Cmp::Lt, 5),
        SetExpr::i32(4, Cmp::Ge, 5),
    ]);
    let all = |logic| Guard::new(SetExpr::All, logic).witness();
    assert_eq!(
        Guard::new(covering.clone(), Logic::TwoValued).witness(),
        all(Logic::TwoValued)
    );
    assert_ne!(
        Guard::new(covering, Logic::Kleene).witness(),
        all(Logic::Kleene)
    );
    // The logic is part of the witness: the same expression under the other
    // logic is a different guard.
    assert_ne!(all(Logic::TwoValued), all(Logic::Kleene));
}

/// A method reference resolved at another schema revision is refused
/// (D-DRDT §5), and so is a ClassView whose field basis moved under the
/// binding.
#[test]
fn schema_revision_and_field_basis_are_checked() {
    let s = store_with_order();
    let view = View::with_fields(VIEW_CLASS, 4);
    assert_eq!(
        s.resolve(&view, ORDER, SchemaRev(2)).unwrap_err(),
        Refusal::SchemaRevisionMismatch {
            class: ORDER,
            bound: SchemaRev(1),
            asked: SchemaRev(2)
        }
    );
    let grown = View::with_fields(VIEW_CLASS, 5);
    assert_eq!(
        s.resolve(&grown, ORDER, SchemaRev(1)).unwrap_err(),
        Refusal::FieldBasisChanged {
            class: ORDER,
            bound: 4,
            view: 5
        }
    );
}

/// Registration refuses what would make a binding ambiguous or point outside
/// its class: a repeated MethodId, a field position past the schema, and a
/// basis wider than a `WideFieldMask` addresses.
#[test]
fn registration_refuses_ambiguity_and_out_of_basis_positions() {
    let mut s = catalogue();
    assert_eq!(
        s.register(
            ORDER,
            schema(1, 4),
            vec![method(1, vec![]), method(1, vec![effect(2, "assign")])]
        ),
        Err(Refusal::DuplicateMethodId {
            class: ORDER,
            method: MethodId(1)
        })
    );
    assert_eq!(
        s.register(
            ORDER,
            schema(1, 4),
            vec![method(1, vec![effect(4, "assign")])]
        ),
        Err(Refusal::FieldOutOfBasis {
            class: ORDER,
            method: MethodId(1),
            position: 4
        })
    );
    let reads_past = MethodBinding {
        reads: WideFieldMask::from_positions(&[9]),
        ..method(1, vec![])
    };
    assert_eq!(
        s.register(ORDER, schema(1, 4), vec![reads_past]),
        Err(Refusal::FieldOutOfBasis {
            class: ORDER,
            method: MethodId(1),
            position: 9
        })
    );
    assert_eq!(
        s.register(ORDER, schema(1, 257), vec![]),
        Err(Refusal::BasisTooWide {
            class: ORDER,
            field_count: 257
        })
    );
    // None of the refused registrations bound the class.
    assert_eq!(s.class_count(), 0);
    // The largest legal basis is accepted, with a write on its last position.
    assert_eq!(
        s.register(
            ORDER,
            schema(1, 256),
            vec![method(1, vec![effect(255, "assign")])]
        ),
        Ok(())
    );
}

/// The store records what a method needs (D-DT64-6 A.1): population, truth and
/// value demand ride the binding unchanged; the store never refuses a Bag,
/// because admitting a counted population is the binder's call, not DT's.
#[test]
fn population_truth_and_demand_ride_the_binding() {
    let mut s = catalogue();
    let count = MethodBinding {
        population: Population::Bag,
        truth: Truth::TwoValued,
        demand: ValueDemand::MapOnly,
        when: None,
        ..method(40, vec![])
    };
    s.register(ORDER, schema(1, 4), vec![count.clone()])
        .unwrap();
    let got = s.admit(&MethodRef::of(ORDER, &count)).unwrap();
    assert_eq!(got.population, Population::Bag);
    assert_eq!(got.truth, Truth::TwoValued);
    assert_eq!(got.demand, ValueDemand::MapOnly);
}

/// A classid is a lens: its field basis is the ClassView of its concept half,
/// derived by the codebook's split, never declared. A view that carries the
/// fields under another class does not satisfy it.
#[test]
fn the_lens_is_the_classids_concept_half() {
    assert_eq!(view_class(ORDER), VIEW_CLASS);
    assert_eq!(
        view_class(ORDER_V2),
        VIEW_CLASS,
        "app half does not move the lens"
    );
    let s = store_with_order();
    assert!(s
        .resolve(&View::with_fields(VIEW_CLASS, 4), ORDER, SchemaRev(1))
        .is_ok());
    assert_eq!(
        s.resolve(&View::with_fields(0x0A1F, 4), ORDER, SchemaRev(1))
            .unwrap_err(),
        Refusal::FieldBasisChanged {
            class: ORDER,
            bound: 4,
            view: 0
        }
    );
}

/// Lab consumer classids. Each switches labels (its own ClassView) and
/// polyfill (its bindings) over the shared, classid-free algebra.
const XLOG: u32 = 0x7F01_0001;
const ONNX: u32 = 0x7F03_0001;

/// XLOG's count `q(Y) :- r(X,Y), s(Y,Z)`: fields r=0, s=1, the counts 2 and 3,
/// the product 4.
fn xlog_count() -> MethodBinding {
    MethodBinding {
        id: MethodId(1),
        sig_rev: SigRev(1),
        name: "count q(Y)".to_string(),
        on: WideFieldMask::from_positions(&[0, 1]),
        when: None,
        reads: WideFieldMask::from_positions(&[0, 1]),
        effects: vec![
            // R keyed by its y column (1)
            Effect {
                target: 2,
                algebra: uses(GROUP_COUNT),
                args: vec![Arg::Field(0), Arg::Const(1)],
            },
            // S keyed by its y column (0)
            Effect {
                target: 3,
                algebra: uses(GROUP_COUNT),
                args: vec![Arg::Field(1), Arg::Const(0)],
            },
            Effect {
                target: 4,
                algebra: uses(MUL_I64),
                args: vec![Arg::Field(2), Arg::Field(3)],
            },
        ],
        population: Population::Bag,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// An ONNX `Mul(int64)` node: fields a=0, b=1, y=2.
fn onnx_mul() -> MethodBinding {
    MethodBinding {
        name: "Mul(int64)".to_string(),
        effects: vec![Effect {
            target: 2,
            algebra: uses(MUL_I64),
            args: vec![Arg::Field(0), Arg::Field(1)],
        }],
        population: Population::Distinct,
        ..xlog_count()
    }
}

#[test]
fn consumers_switch_over_shared_classid_free_algebra() {
    let mut s = catalogue();
    s.register(XLOG, schema(1, 5), vec![xlog_count()]).unwrap();
    s.register(ONNX, schema(1, 3), vec![onnx_mul()]).unwrap();

    // One algebra entry, reused by both classids; nothing minted for it.
    assert_eq!(
        s.users_of(MUL_I64),
        vec![(XLOG, MethodId(1)), (ONNX, MethodId(1))]
    );
    assert_eq!(s.users_of(GROUP_COUNT), vec![(XLOG, MethodId(1))]);
    assert!(s.users_of(ASSIGN).is_empty());
    assert_eq!(s.class_count(), 2, "only the consumers carry classids");
    assert_eq!(
        s.algebra(MUL_I64).unwrap().polyfill,
        "ndarray::hpc::checked_i64::checked_binary_i64(Mul)"
    );

    // The classid switches the labels: each reads its own lens.
    let xv = View::with_fields(view_class(XLOG), 5);
    let ov = View::with_fields(view_class(ONNX), 3);
    assert_eq!(s.resolve(&xv, XLOG, SchemaRev(1)).unwrap().fields.len(), 5);
    assert_eq!(s.resolve(&ov, ONNX, SchemaRev(1)).unwrap().fields.len(), 3);
    assert!(
        s.resolve(&ov, XLOG, SchemaRev(1)).is_err(),
        "not XLOG's lens"
    );
}

/// An algebra entry is immutable once declared (D5 applied to D6).
#[test]
fn an_algebra_entry_is_immutable() {
    let mut s = catalogue();
    let same = entry(
        MUL_I64,
        "ndarray::hpc::checked_i64::checked_binary_i64(Mul)",
        2,
    );
    assert_eq!(s.declare_algebra(same), Ok(()));
    assert_eq!(
        s.declare_algebra(entry(MUL_I64, "wrapping_mul", 2)),
        Err(Refusal::AlgebraImmutable(MUL_I64))
    );
}

/// The operands are part of the binding (XLOG's correction), and every effect
/// is checked against the catalogue when its class registers.
#[test]
fn effect_operands_and_entries_are_checked() {
    let count = xlog_count();
    let mut twice_r = count.clone();
    twice_r.effects[1].args = vec![Arg::Field(0), Arg::Const(1)];
    assert_ne!(count, twice_r, "operands tell the two counts apart");
    let mut swapped = count.clone();
    swapped.effects[2].args = vec![Arg::Field(3), Arg::Field(2)];
    assert_ne!(count, swapped, "operand order is part of the step");

    let mut s = catalogue();
    let refused = |s: &mut BindStore, f: &dyn Fn(&mut Effect)| {
        let mut m = xlog_count();
        f(&mut m.effects[2]);
        s.register(XLOG, schema(1, 5), vec![m]).unwrap_err()
    };
    assert_eq!(
        refused(&mut s, &|e| e.args = vec![Arg::Field(2)]),
        Refusal::ArityMismatch {
            class: XLOG,
            method: MethodId(1),
            algebra: MUL_I64,
            expected: 2,
            got: 1
        }
    );
    assert_eq!(
        refused(&mut s, &|e| e.args = vec![Arg::Field(2), Arg::Field(7)]),
        Refusal::FieldOutOfBasis {
            class: XLOG,
            method: MethodId(1),
            position: 7
        }
    );
    assert_eq!(
        refused(&mut s, &|e| e.algebra = uses(AlgebraId(99))),
        Refusal::UnknownAlgebra {
            class: XLOG,
            method: MethodId(1),
            algebra: AlgebraId(99)
        }
    );
    assert_eq!(
        refused(&mut s, &|e| e.algebra.sig_rev = SigRev(0)),
        Refusal::AlgebraRevisionMismatch {
            class: XLOG,
            method: MethodId(1),
            algebra: MUL_I64,
            bound: SigRev(1),
            asked: SigRev(0)
        }
    );
    assert_eq!(s.class_count(), 0, "no refused class was bound");
    assert_eq!(s.register(XLOG, schema(1, 5), vec![count]), Ok(()));
}
