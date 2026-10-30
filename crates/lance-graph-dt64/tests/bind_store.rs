//! The DT64 bind store against the decisions it implements: D-DT64-6 §1 (D4,
//! D5, D6) and §3, D-DRDT §4 (the four gaps) and §5 (the DR #4 falsifier).

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Cmp, Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{
    view_class, AlgebraRef, Arg, BindStore, ClassSchema, Effect, EffectOp, Guard, MethodBinding,
    MethodId, MethodRef, Refusal, RouteVersion, SchemaRev, SigRev, ValueDemand,
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

fn effect(target: u8, algebra: &str) -> Effect {
    Effect {
        target,
        op: EffectOp::Algebra(AlgebraRef::new(algebra)),
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
        inputs: 0,
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
    let mut s = BindStore::new();
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
    let mut s = BindStore::new();
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
    assert_eq!(
        s.admit(&ra).unwrap().effects[0].op,
        EffectOp::Algebra(AlgebraRef::new("assign"))
    );
    assert_eq!(
        s.admit(&rb).unwrap().effects[0].op,
        EffectOp::Algebra(AlgebraRef::new("accumulate"))
    );

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
    let mut s = BindStore::new();
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
    let mut s = BindStore::new();
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
    let mut s = BindStore::new();
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

/// Lab stand-ins: one algebra lens whose entries are two native leaves, and
/// one consumer lens (XLOG's per-Y count). No classid per algebra item.
const ALG: u32 = 0x7F02_0001;
const XLOG: u32 = 0x7F01_0001;
const GROUP_COUNT: MethodId = MethodId(1);
const MUL_I64: MethodId = MethodId(2);

fn leaf(id: MethodId, native: &str, inputs: u8) -> MethodBinding {
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: native.to_string(),
        inputs,
        on: WideFieldMask::from_positions(&[]),
        when: None,
        reads: WideFieldMask::from_positions(&[]),
        effects: vec![effect(0, native)],
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

fn algebra_store() -> BindStore {
    let mut s = BindStore::new();
    s.register(
        ALG,
        schema(1, 1),
        vec![
            // (relation field, key column)
            leaf(GROUP_COUNT, "ndarray::simd::masked_group_count_u32", 2),
            leaf(
                MUL_I64,
                "ndarray::hpc::checked_i64::checked_binary_i64(Mul)",
                2,
            ),
        ],
    )
    .unwrap();
    s
}

fn call(s: &BindStore, class: u32, method: MethodId) -> EffectOp {
    let entry = MethodRef {
        class,
        method,
        sig_rev: SigRev(1),
        guard: None,
    };
    s.admit(&entry).unwrap();
    EffectOp::Call(entry)
}

/// XLOG's count `q(Y) :- r(X,Y), s(Y,Z)`: fields r=0, s=1, two counts 2 and
/// 3, the product 4. Its DO calls the algebra lens's entries in order.
fn xlog_count(s: &BindStore) -> MethodBinding {
    MethodBinding {
        id: MethodId(1),
        sig_rev: SigRev(1),
        name: "count q(Y)".to_string(),
        inputs: 0,
        on: WideFieldMask::from_positions(&[0, 1]),
        when: None,
        reads: WideFieldMask::from_positions(&[0, 1]),
        effects: vec![
            Effect {
                target: 2,
                op: call(s, ALG, GROUP_COUNT),
                // R keyed by its y column (1)
                args: vec![Arg::Field(0), Arg::Const(1)],
            },
            Effect {
                target: 3,
                op: call(s, ALG, GROUP_COUNT),
                // S keyed by its y column (0)
                args: vec![Arg::Field(1), Arg::Const(0)],
            },
            Effect {
                target: 4,
                op: call(s, ALG, MUL_I64),
                args: vec![Arg::Field(2), Arg::Field(3)],
            },
        ],
        population: Population::Bag,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

#[test]
fn a_consumer_reuses_algebra_entries_by_reference() {
    let mut s = algebra_store();
    let count = xlog_count(&s);
    s.register(XLOG, schema(1, 5), vec![count.clone()]).unwrap();

    assert_eq!(s.callers_of(ALG), vec![(XLOG, MethodId(1))]);
    assert!(s.callers_of(XLOG).is_empty());
    let got = s.admit(&MethodRef::of(XLOG, &count)).unwrap();
    let callees: Vec<MethodId> = got
        .effects
        .iter()
        .map(|e| match &e.op {
            EffectOp::Call(r) => r.method,
            EffectOp::Algebra(_) => panic!("XLOG declares no algebra of its own"),
        })
        .collect();
    assert_eq!(callees, vec![GROUP_COUNT, GROUP_COUNT, MUL_I64]);
    // Two classes bound, not one per algebra item.
    assert_eq!(s.class_count(), 2);

    // The reused lens cannot change under its caller (D5).
    assert_eq!(
        s.register(ALG, schema(1, 1), vec![leaf(GROUP_COUNT, "other", 2)]),
        Err(Refusal::BehaviourImmutable(ALG))
    );
}

/// A call is admitted when the caller registers: an unknown entry, a stale
/// signature revision, an unbound lens and a dropped guard are refused then,
/// and the caller is not bound.
#[test]
fn a_call_to_an_unknown_or_stale_entry_is_refused_at_registration() {
    let refused = |s: &mut BindStore, callee: MethodRef| {
        let m = MethodBinding {
            effects: vec![Effect {
                target: 2,
                op: EffectOp::Call(callee),
                args: vec![Arg::Const(0), Arg::Const(0)],
            }],
            ..method(5, vec![])
        };
        s.register(XLOG, schema(1, 5), vec![m]).unwrap_err()
    };
    let entry = |class, method, sig| MethodRef {
        class,
        method,
        sig_rev: SigRev(sig),
        guard: None,
    };
    let wrap = |callee| Refusal::CallRefused {
        class: XLOG,
        method: MethodId(5),
        callee: Box::new(callee),
    };

    let mut empty = BindStore::new();
    assert_eq!(
        refused(&mut empty, entry(ALG, GROUP_COUNT, 1)),
        wrap(Refusal::UnknownClass(ALG))
    );

    let mut s = algebra_store();
    assert_eq!(
        refused(&mut s, entry(ALG, MethodId(9), 1)),
        wrap(Refusal::UnknownMethod {
            class: ALG,
            method: MethodId(9)
        })
    );
    assert_eq!(
        refused(&mut s, entry(ALG, MUL_I64, 0)),
        wrap(Refusal::SignatureRevisionMismatch {
            class: ALG,
            method: MUL_I64,
            bound: SigRev(1),
            asked: SigRev(0)
        })
    );
    // A guard the entry does not have is refused too.
    let mut guarded = entry(ALG, MUL_I64, 1);
    guarded.guard = Some(Guard::new(SetExpr::All, Logic::TwoValued).witness());
    assert_eq!(
        refused(&mut s, guarded),
        wrap(Refusal::GuardAdded {
            class: ALG,
            method: MUL_I64
        })
    );
    assert_eq!(s.class_count(), 1, "no refused caller was bound");
}

/// The operands are part of the binding (XLOG's correction): counting R twice
/// is a different binding from counting R and S, and a call with the wrong
/// number of operands, or an operand outside the caller's basis, is refused at
/// registration.
#[test]
fn call_operands_are_bound_and_checked() {
    let s = algebra_store();
    let count = xlog_count(&s);
    let mut twice_r = count.clone();
    twice_r.effects[1].args = vec![Arg::Field(0), Arg::Const(1)];
    assert_ne!(count, twice_r, "operands tell the two counts apart");
    let mut swapped = count.clone();
    swapped.effects[2].args = vec![Arg::Field(3), Arg::Field(2)];
    assert_ne!(count, swapped, "operand order is part of the step");

    let mut s2 = algebra_store();
    let mut short = count.clone();
    short.effects[2].args = vec![Arg::Field(2)];
    assert_eq!(
        s2.register(XLOG, schema(1, 5), vec![short]),
        Err(Refusal::CallRefused {
            class: XLOG,
            method: MethodId(1),
            callee: Box::new(Refusal::ArityMismatch {
                class: ALG,
                method: MUL_I64,
                expected: 2,
                got: 1
            })
        })
    );
    let mut outside = count.clone();
    outside.effects[2].args = vec![Arg::Field(2), Arg::Field(7)];
    assert_eq!(
        s2.register(XLOG, schema(1, 5), vec![outside]),
        Err(Refusal::FieldOutOfBasis {
            class: XLOG,
            method: MethodId(1),
            position: 7
        })
    );
    assert_eq!(s2.register(XLOG, schema(1, 5), vec![count]), Ok(()));
}
