//! The DT64 bind store against the decisions it implements: D-DT64-6 §1 (D4,
//! D5, D6) and §3, D-DRDT §4 (the four gaps) and §5 (the DR #4 falsifier).

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Cmp, Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{
    view_class, AlgebraRef, Arg, Axis, BasisSource, BindStore, ClassSchema, Effect, Extent, Guard,
    MethodBinding, MethodId, MethodRef, Refusal, RouteVersion, SchemaRev, SigRev, ValueDemand,
    Width,
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

    fn with_labels(class: ClassId, labels: &[&str]) -> View {
        View {
            class,
            fields: labels
                .iter()
                .map(|l| FieldRef::new(format!("urn:{l}"), (*l).to_string()))
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

/// Register `class` against the `urn:f{i}` basis [`View::with_fields`] gives
/// its concept at the declared field count: the basis most tests here resolve
/// through. Tests about the basis witness itself call `register` directly.
fn reg(
    s: &mut BindStore,
    class: u32,
    schema: ClassSchema,
    methods: Vec<MethodBinding>,
) -> Result<(), Refusal> {
    let view = View::with_fields(view_class(class), schema.field_count);
    s.register(&view, class, schema, methods)
}

/// [`reg`] for a derived classid.
fn reg_derived(
    s: &mut BindStore,
    base: u32,
    class: u32,
    schema: ClassSchema,
    methods: Vec<MethodBinding>,
) -> Result<(), Refusal> {
    let view = View::with_fields(view_class(class), schema.field_count);
    s.register_derived(&view, base, class, schema, methods)
}

const ORDER: u32 = 0x0A1E_0001;
const ORDER_V2: u32 = 0x0A1E_0002;
/// The lens ORDER is read through: its concept half (`view_class`).
const VIEW_CLASS: ClassId = 0x0A1E;

/// A schema in the DR64³ extent.
fn schema(rev: u32, fields: usize) -> ClassSchema {
    schema_in(rev, fields, Extent::DR64_CUBE)
}

fn schema_in(rev: u32, fields: usize, extent: Extent) -> ClassSchema {
    ClassSchema {
        schema_rev: SchemaRev(rev),
        field_count: fields,
        extent,
    }
}

/// The polyfill vocabulary is one lens classid (lab), its words the fields
/// of that lens. No word has a classid of its own.
const POLYFILL: u32 = 0x7F0A_0001;
/// The words' labels, by position: what the polyfill lens's ClassView holds.
/// relalg describes, mask-risc folds, `ndarray::simd` computes; the two
/// `lab::` words stand in for any nullary step in the ordering tests.
const WORDS: [&str; 4] = [
    "lab::assign",
    "lab::accumulate",
    // mask-risc's keyed count, COUNT(*) GROUP BY key: (relation, key column)
    "lance_graph_mask_risc::Terminal::GroupReduce(GroupKey::Lane, GroupFold::Count)",
    // ndarray #407: checked i64 multiply on the ndarray::simd facade
    "ndarray::simd::checked_binary_i64(I64Op::Mul)",
];
const ARITY: [u8; 4] = [0, 0, 2, 2];

const fn word(word: u8) -> AlgebraRef {
    AlgebraRef {
        lens: POLYFILL,
        word,
    }
}
const ASSIGN: AlgebraRef = word(0);
const ACCUMULATE: AlgebraRef = word(1);
const GROUP_COUNT: AlgebraRef = word(2);
const MUL_I64: AlgebraRef = word(3);

/// A store holding only the polyfill lens: one classid, no consumer.
fn catalogue() -> BindStore {
    let mut s = BindStore::new();
    s.declare_algebra(POLYFILL, SchemaRev(1), Extent::DR64_CUBE, ARITY.to_vec())
        .unwrap();
    s
}

fn effect(target: u8, algebra: AlgebraRef) -> Effect {
    Effect {
        target,
        algebra,
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
    reg(
        &mut s,
        ORDER,
        schema(1, 4),
        vec![method(10, vec![effect(2, ASSIGN), effect(3, ACCUMULATE)])],
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
    let same = vec![method(10, vec![effect(2, ASSIGN), effect(3, ACCUMULATE)])];
    assert_eq!(reg(&mut s, ORDER, schema(1, 4), same), Ok(()));

    let reordered = vec![method(10, vec![effect(3, ACCUMULATE), effect(2, ASSIGN)])];
    assert_eq!(
        reg(&mut s, ORDER, schema(1, 4), reordered),
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
    reg_derived(
        &mut s,
        ORDER,
        ORDER_V2,
        schema(1, 4),
        vec![method(11, vec![effect(3, ACCUMULATE), effect(2, ASSIGN)])],
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
        reg_derived(&mut s, 0xBEEF_0001, ORDER_V2, schema(1, 4), vec![]),
        Err(Refusal::UnknownClass(0xBEEF_0001))
    );
    assert_eq!(
        reg_derived(&mut s, ORDER, ORDER, schema(1, 4), vec![]),
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
    let a = method(20, vec![effect(2, ASSIGN), effect(2, ACCUMULATE)]);
    let b = method(21, vec![effect(2, ACCUMULATE), effect(2, ASSIGN)]);
    // The negative mold: equal unordered read and write sets.
    assert_eq!(a.reads, b.reads);
    assert_eq!(a.writes(), b.writes());
    assert_ne!(a, b, "effect order is part of the binding");
    reg(&mut s, ORDER, schema(1, 4), vec![a.clone(), b.clone()]).unwrap();

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
    let ra = s.method_ref(ORDER, a.id).unwrap();
    let rb = s.method_ref(ORDER, b.id).unwrap();
    assert_eq!(s.admit(&ra).unwrap().effects[0].algebra, ASSIGN);
    assert_eq!(s.admit(&rb).unwrap().effects[0].algebra, ACCUMULATE);

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
        ..method(30, vec![effect(2, ASSIGN)])
    };
    reg(&mut s, ORDER, schema(1, 4), vec![free.clone()]).unwrap();
    assert!(s.admit(&s.method_ref(ORDER, free.id).unwrap()).is_ok());
    let added = MethodRef {
        guard: Some(Guard::new(SetExpr::All, Logic::TwoValued).witness()),
        ..s.method_ref(ORDER, free.id).unwrap()
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
///
/// Decision J (2026-11-07): the ordered `predicate_iri` basis is EXACT within
/// one class `SchemaRev`. An append is an evolution and takes a new revision,
/// so refusing a grown basis under the same revision is intended and stays.
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

/// A view over the ORDER class with the given `(predicate_iri, label)` pairs.
fn view_of(fields: &[(&str, &str)]) -> View {
    View {
        class: VIEW_CLASS,
        fields: fields
            .iter()
            .map(|(iri, label)| FieldRef::new(*iri, *label))
            .collect(),
    }
}

/// The basis ORDER is bound against: `View::with_fields(VIEW_CLASS, 4)`.
const BOUND: [(&str, &str); 4] = [
    ("urn:f0", "f0"),
    ("urn:f1", "f1"),
    ("urn:f2", "f2"),
    ("urn:f3", "f3"),
];

/// Decision J, item 1 (was the N1 FINDING, flipped by L5a): an equal-length
/// REORDER under the same revision is refused. Before L5a the basis check
/// compared revision and count only, so ORDER's method kept reading position
/// 1 after it came to name `urn:f2`. Correct count, wrong order.
#[test]
fn an_equal_length_reorder_is_refused() {
    let s = store_with_order();
    let swapped = view_of(&[BOUND[0], BOUND[2], BOUND[1], BOUND[3]]);
    assert_eq!(swapped.fields.len(), BOUND.len(), "same count");
    assert_eq!(
        s.resolve(&swapped, ORDER, SchemaRev(1)).unwrap_err(),
        Refusal::PredicateBasisChanged {
            class: ORDER,
            schema_rev: SchemaRev(1),
            source: BasisSource::Current,
        }
    );
}

/// FINDING (still true after L5c): `admit` never reads the view. Since L5c a
/// reference's stamp carries the binding digest, which covers the basis
/// witness at registration, and `admit` compares it with the digest the store
/// holds; neither is recomputed from the view now. So a router holding a
/// `MethodRef` is admitted whatever the view now says. `resolve` refuses the
/// reorder above; callers that execute resolve first (R2IL `prepare` does).
#[test]
fn finding_admit_does_not_carry_the_basis_identity() {
    let s = store_with_order();
    let at_bind = View::with_fields(VIEW_CLASS, 4);
    let bound = s.resolve(&at_bind, ORDER, SchemaRev(1)).unwrap();
    let r = s.method_ref(ORDER, bound.methods[0].id).unwrap();
    let swapped = view_of(&[BOUND[0], BOUND[2], BOUND[1], BOUND[3]]);
    assert!(s.resolve(&swapped, ORDER, SchemaRev(1)).is_err());
    assert!(s.admit(&r).is_ok(), "admit never reads a basis");
}

/// Decision J, item 1 (was the N4 FINDING, flipped by L5a): an equal-length
/// change of one field's `predicate_iri` (a replaced field) is refused.
#[test]
fn an_equal_length_predicate_change_is_refused() {
    let s = store_with_order();
    let replaced = view_of(&[BOUND[0], ("urn:g1", "f1"), BOUND[2], BOUND[3]]);
    assert_eq!(
        s.resolve(&replaced, ORDER, SchemaRev(1)).unwrap_err(),
        Refusal::PredicateBasisChanged {
            class: ORDER,
            schema_rev: SchemaRev(1),
            source: BasisSource::Current,
        }
    );
}

/// Decision J, item 1: removing a field under the same revision is refused.
#[test]
fn a_same_revision_removal_is_refused() {
    let s = store_with_order();
    let shrunk = view_of(&BOUND[..3]);
    assert_eq!(
        s.resolve(&shrunk, ORDER, SchemaRev(1)).unwrap_err(),
        Refusal::FieldBasisChanged {
            class: ORDER,
            bound: 4,
            view: 3
        }
    );
}

/// The witness hashes each predicate with its length, so two bases whose
/// predicates concatenate to the same bytes stay distinct.
#[test]
fn predicates_that_concatenate_alike_are_still_different_bases() {
    let s = store_with_order();
    let shifted = view_of(&[("urn:f0u", "f0"), ("rn:f1", "f1"), BOUND[2], BOUND[3]]);
    let joined = |v: &View| {
        v.fields
            .iter()
            .map(|f| f.predicate_iri.as_str())
            .collect::<String>()
    };
    assert_eq!(joined(&shifted), joined(&view_of(&BOUND)));
    assert!(matches!(
        s.resolve(&shifted, ORDER, SchemaRev(1)),
        Err(Refusal::PredicateBasisChanged { .. })
    ));
}

/// The witness against an independent oracle: for each listed variant of
/// ORDER's basis, `resolve` accepts when, and only when, the ordered
/// `predicate_iri`s equal the bound ones, whatever the labels say. (Up to a
/// digest collision: the store compares a witness, not the sequence.)
#[test]
fn resolve_accepts_the_bound_sequence_and_refuses_each_listed_variant() {
    let s = store_with_order();
    let bound: Vec<&str> = BOUND.iter().map(|(iri, _)| *iri).collect();
    let variants: Vec<Vec<(&str, &str)>> = vec![
        BOUND.to_vec(),
        BOUND.iter().map(|(iri, _)| (*iri, "relabelled")).collect(),
        vec![BOUND[1], BOUND[0], BOUND[2], BOUND[3]],
        vec![BOUND[0], BOUND[1], BOUND[3], BOUND[2]],
        vec![BOUND[3], BOUND[2], BOUND[1], BOUND[0]],
        vec![("urn:F0", "f0"), BOUND[1], BOUND[2], BOUND[3]],
        vec![BOUND[0], BOUND[1], BOUND[2], ("urn:f3 ", "f3")],
        vec![BOUND[0], BOUND[0], BOUND[2], BOUND[3]],
        BOUND[..3].to_vec(),
        [&BOUND[..], &[("urn:f4", "f4")]].concat(),
    ];
    let mut accepted = 0;
    for v in &variants {
        let view = view_of(v);
        let oracle = iris(&view.fields) == bound;
        let ok = s.resolve(&view, ORDER, SchemaRev(1)).is_ok();
        assert_eq!(ok, oracle, "variant {v:?}");
        accepted += usize::from(ok);
    }
    // Anti-vacuity: the sweep holds both outcomes.
    assert_eq!(accepted, 2, "exactly the bound basis and its relabelling");
}

/// Registration reads the basis from the source and refuses a declared field
/// count the source's basis does not have.
#[test]
fn registration_refuses_a_basis_of_another_count() {
    let mut s = BindStore::new();
    assert_eq!(
        s.register(
            &View::with_fields(VIEW_CLASS, 3),
            ORDER,
            schema(1, 4),
            vec![]
        ),
        Err(Refusal::FieldBasisChanged {
            class: ORDER,
            bound: 4,
            view: 3
        })
    );
    assert_eq!(s.class_count(), 0);
}

/// D5 with the witness: registering again over the same basis is idempotent;
/// the same schema and methods over another basis are a different binding and
/// are refused.
#[test]
fn re_registration_is_idempotent_only_over_the_same_basis() {
    let mut s = BindStore::new();
    let basis = view_of(&BOUND);
    assert_eq!(s.register(&basis, ORDER, schema(1, 4), vec![]), Ok(()));
    assert_eq!(s.register(&basis, ORDER, schema(1, 4), vec![]), Ok(()));
    let relabelled = view_of(&[
        ("urn:f0", "a"),
        ("urn:f1", "b"),
        ("urn:f2", "c"),
        ("urn:f3", "d"),
    ]);
    assert_eq!(s.register(&relabelled, ORDER, schema(1, 4), vec![]), Ok(()));
    let swapped = view_of(&[BOUND[0], BOUND[2], BOUND[1], BOUND[3]]);
    assert_eq!(
        s.register(&swapped, ORDER, schema(1, 4), vec![]),
        Err(Refusal::BehaviourImmutable(ORDER))
    );
}

/// Decision J, item 4: a label-only rename keeps every `predicate_iri`, so it
/// is not a meaning change and must keep resolving. This stays green when the
/// exact basis witness lands; it is the twin that proves the witness keys on
/// predicates, not on labels.
#[test]
fn a_label_only_rename_resolves_and_must_keep_resolving() {
    let s = store_with_order();
    let relabelled = view_of(&[
        ("urn:f0", "Quantity"),
        ("urn:f1", "Price"),
        ("urn:f2", "Total"),
        ("urn:f3", "State"),
    ]);
    let r = s.resolve(&relabelled, ORDER, SchemaRev(1)).unwrap();
    assert_eq!(r.fields[1].label, "Price");
    assert_eq!(r.fields[1].predicate_iri, "urn:f1");
}

/// Decision J item 5, the refusal half: `ClassView::fields(class)` takes no
/// revision, so once a source WITHOUT history evolves (here: appends a field,
/// bound as a derived classid at a new revision), the OLD classid cannot be
/// resolved at the revision it was bound at. Its behaviour is immutable (D5);
/// what is missing is the source's record of its old basis, and the store
/// refuses rather than read the new basis as the old one. The acceptance half
/// is `a_history_keeping_source_replays_each_revision_against_its_own_basis`.
#[test]
fn a_source_without_history_refuses_replay_after_evolution() {
    let mut s = store_with_order();
    reg_derived(&mut s, ORDER, ORDER_V2, schema(2, 5), Vec::new()).unwrap();
    let evolved = View::with_fields(VIEW_CLASS, 5);
    assert!(s.resolve(&evolved, ORDER_V2, SchemaRev(2)).is_ok());
    assert_eq!(
        s.resolve(&evolved, ORDER, SchemaRev(1)).unwrap_err(),
        Refusal::FieldBasisChanged {
            class: ORDER,
            bound: 4,
            view: 5
        }
    );
}

/// A source that keeps its own basis history (D-DT64-7 Decision J, item 5):
/// `current` is the basis now, `history` the basis it had at each recorded
/// revision, `lens_history` the same per FULL lens classid.
struct History {
    class: ClassId,
    current: Vec<FieldRef>,
    history: Vec<(u32, Vec<FieldRef>)>,
    lens_history: Vec<((u32, u32), Vec<FieldRef>)>,
}

impl ClassView for History {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == self.class {
            &self.current
        } else {
            &[]
        }
    }
    fn fields_at(&self, class: ClassId, schema_rev: u32) -> Option<&[FieldRef]> {
        if class != self.class {
            return None;
        }
        self.history
            .iter()
            .find(|(r, _)| *r == schema_rev)
            .map(|(_, f)| f.as_slice())
    }
    fn lens_fields_at(&self, lens: u32, schema_rev: u32) -> Option<&[FieldRef]> {
        self.lens_history
            .iter()
            .find(|(k, _)| *k == (lens, schema_rev))
            .map(|(_, f)| f.as_slice())
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

/// The independent oracle for basis identity: the ordered `predicate_iri`s,
/// labels dropped.
fn iris(fields: &[FieldRef]) -> Vec<&str> {
    fields.iter().map(|f| f.predicate_iri.as_str()).collect()
}

/// ORDER's source at revision 2: rev 1 had four fields, rev 2 appended
/// `urn:f4`, and the current basis is rev 2's.
fn evolved_source() -> History {
    let rev1 = View::with_fields(VIEW_CLASS, 4).fields;
    let rev2 = View::with_fields(VIEW_CLASS, 5).fields;
    History {
        class: VIEW_CLASS,
        current: rev2.clone(),
        history: vec![(1, rev1), (2, rev2)],
        lens_history: Vec::new(),
    }
}

/// Decision J item 5, the acceptance half: a source that records its history
/// answers both revisions, and each classid resolves against the basis of the
/// revision it was bound at, not the current one.
#[test]
fn a_history_keeping_source_replays_each_revision_against_its_own_basis() {
    let mut s = store_with_order();
    reg_derived(&mut s, ORDER, ORDER_V2, schema(2, 5), Vec::new()).unwrap();
    let source = evolved_source();
    let old = s.resolve(&source, ORDER, SchemaRev(1)).unwrap();
    let new = s.resolve(&source, ORDER_V2, SchemaRev(2)).unwrap();
    assert_eq!(iris(old.fields), ["urn:f0", "urn:f1", "urn:f2", "urn:f3"]);
    assert_eq!(
        iris(new.fields),
        ["urn:f0", "urn:f1", "urn:f2", "urn:f3", "urn:f4"]
    );
    // The current basis is rev 2's; rev 1 came from the history, not from it.
    assert_eq!(source.fields(VIEW_CLASS).len(), 5);
}

/// A source that keeps history but not for the bound revision: it is
/// accepted only if its current basis is proved equal to the bound one, and
/// refused otherwise. A missing history is never read as the current basis.
#[test]
fn a_missing_revision_in_history_is_proved_from_the_current_basis_or_refused() {
    let s = store_with_order();
    let rev1 = View::with_fields(VIEW_CLASS, 4).fields;
    let still_rev1 = History {
        class: VIEW_CLASS,
        current: rev1.clone(),
        history: vec![(7, rev1.clone())],
        lens_history: Vec::new(),
    };
    assert!(s.resolve(&still_rev1, ORDER, SchemaRev(1)).is_ok());
    let mut moved = rev1.clone();
    moved.swap(1, 2);
    let moved_on = History {
        class: VIEW_CLASS,
        current: moved,
        history: vec![(7, rev1)],
        lens_history: Vec::new(),
    };
    assert_eq!(
        s.resolve(&moved_on, ORDER, SchemaRev(1)).unwrap_err(),
        Refusal::PredicateBasisChanged {
            class: ORDER,
            schema_rev: SchemaRev(1),
            source: BasisSource::Current,
        }
    );
}

/// A source whose OWN record of revision 1 disagrees with the basis ORDER was
/// bound against (its history was rewritten in place) is refused, and the
/// refusal says the history disagreed, even though its current basis is the
/// bound one.
#[test]
fn a_rewritten_history_is_refused_even_when_the_current_basis_matches() {
    let s = store_with_order();
    let rev1 = View::with_fields(VIEW_CLASS, 4).fields;
    let mut rewritten = rev1.clone();
    rewritten.swap(0, 3);
    let source = History {
        class: VIEW_CLASS,
        current: rev1,
        history: vec![(1, rewritten)],
        lens_history: Vec::new(),
    };
    assert_eq!(
        s.resolve(&source, ORDER, SchemaRev(1)).unwrap_err(),
        Refusal::PredicateBasisChanged {
            class: ORDER,
            schema_rev: SchemaRev(1),
            source: BasisSource::Historical,
        }
    );
}

/// The history is read at the BOUND revision. Asking ORDER at revision 2,
/// which the source does hold, is still refused: the classid was bound at 1.
#[test]
fn a_class_is_never_resolved_at_another_revision_even_when_history_has_it() {
    let s = store_with_order();
    assert_eq!(
        s.resolve(&evolved_source(), ORDER, SchemaRev(2))
            .unwrap_err(),
        Refusal::SchemaRevisionMismatch {
            class: ORDER,
            bound: SchemaRev(1),
            asked: SchemaRev(2)
        }
    );
}

/// Behaviour lenses keep separate histories by their FULL classid (O9): two
/// lenses under one left half each read their own revision, and a lens the
/// source holds no record of fails closed instead of reading a sibling's.
#[test]
fn two_lenses_sharing_a_concept_half_keep_separate_histories() {
    let mut s = BindStore::new();
    for (lens, n) in [(RELALG_LENS, 17), (ONNX_LENS, 5), (UNKNOWN_LENS, 17)] {
        s.declare_algebra(lens, SchemaRev(1), Extent::DR64_CUBE, vec![1; n])
            .unwrap();
    }
    let source = History {
        class: 0xFFFF,
        // The left half's own basis, of neither lens's size.
        current: labels("concept", 3),
        history: vec![(1, labels("concept", 3))],
        lens_history: vec![
            ((RELALG_LENS, 1), labels("relalg", 17)),
            ((ONNX_LENS, 1), labels("onnx", 5)),
        ],
    };
    let relalg = s
        .resolve_algebra(&source, RELALG_LENS, SchemaRev(1))
        .unwrap();
    let onnx = s.resolve_algebra(&source, ONNX_LENS, SchemaRev(1)).unwrap();
    assert_eq!(relalg.fields[16].predicate_iri, "urn:relalg/16");
    assert_eq!(onnx.fields[4].predicate_iri, "urn:onnx/4");
    assert_eq!(
        s.resolve_algebra(&source, UNKNOWN_LENS, SchemaRev(1))
            .unwrap_err(),
        Refusal::UnknownLensBasis(UNKNOWN_LENS)
    );
}

/// Registration refuses what would make a binding ambiguous or point outside
/// its class: a repeated MethodId, a field position past the schema, and a
/// basis wider than a `WideFieldMask` addresses.
#[test]
fn registration_refuses_ambiguity_and_out_of_basis_positions() {
    let mut s = catalogue();
    assert_eq!(
        reg(
            &mut s,
            ORDER,
            schema(1, 4),
            vec![method(1, vec![]), method(1, vec![effect(2, ASSIGN)])]
        ),
        Err(Refusal::DuplicateMethodId {
            class: ORDER,
            method: MethodId(1)
        })
    );
    assert_eq!(
        reg(
            &mut s,
            ORDER,
            schema(1, 4),
            vec![method(1, vec![effect(4, ASSIGN)])]
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
        reg(&mut s, ORDER, schema(1, 4), vec![reads_past]),
        Err(Refusal::FieldOutOfBasis {
            class: ORDER,
            method: MethodId(1),
            position: 9
        })
    );
    assert_eq!(
        reg(&mut s, ORDER, schema(1, 257), vec![]),
        Err(Refusal::BasisTooWide {
            class: ORDER,
            field_count: 257
        })
    );
    // None of the refused registrations bound the class.
    assert_eq!(s.class_count(), 0);
    // The largest legal basis is accepted, with a write on its last position.
    assert_eq!(
        reg(
            &mut s,
            ORDER,
            schema_in(1, 256, Extent::DR256_SQ_X64),
            vec![method(1, vec![effect(255, ASSIGN)])]
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
    reg(&mut s, ORDER, schema(1, 4), vec![count.clone()]).unwrap();
    let got = s.admit(&s.method_ref(ORDER, count.id).unwrap()).unwrap();
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
                algebra: GROUP_COUNT,
                args: vec![Arg::Field(0), Arg::Const(1)],
            },
            // S keyed by its y column (0)
            Effect {
                target: 3,
                algebra: GROUP_COUNT,
                args: vec![Arg::Field(1), Arg::Const(0)],
            },
            Effect {
                target: 4,
                algebra: MUL_I64,
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
            algebra: MUL_I64,
            args: vec![Arg::Field(0), Arg::Field(1)],
        }],
        population: Population::Distinct,
        ..xlog_count()
    }
}

#[test]
fn consumers_switch_over_shared_classid_free_algebra() {
    let mut s = catalogue();
    reg(&mut s, XLOG, schema(1, 5), vec![xlog_count()]).unwrap();
    reg(&mut s, ONNX, schema(1, 3), vec![onnx_mul()]).unwrap();

    // One word, reused by both classids by position; nothing minted for it.
    assert_eq!(
        s.users_of(MUL_I64),
        vec![(XLOG, MethodId(1)), (ONNX, MethodId(1))]
    );
    assert_eq!(s.users_of(GROUP_COUNT), vec![(XLOG, MethodId(1))]);
    assert!(s.users_of(ASSIGN).is_empty());
    assert_eq!(
        s.class_count(),
        2,
        "the consumers; the polyfill is one lens"
    );
    assert_eq!(s.arity(MUL_I64), Some(2));
    // Which words each consumer uses, as a mask over the polyfill lens.
    assert_eq!(
        s.words_used(XLOG, POLYFILL),
        Ok(WideFieldMask::from_positions(&[2, 3]))
    );
    assert_eq!(
        s.words_used(ONNX, POLYFILL),
        Ok(WideFieldMask::from_positions(&[3]))
    );
    // Another lens's mask is empty: the mask is over one lens only.
    s.declare_algebra(
        0x7F0B_0001,
        SchemaRev(1),
        Extent::DR64_CUBE,
        vec![2, 2, 2, 2],
    )
    .unwrap();
    assert_eq!(
        s.words_used(XLOG, 0x7F0B_0001),
        Ok(WideFieldMask::from_positions(&[]))
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

/// The polyfill lens is read like any class: one classid, one ClassView, the
/// words as fields, a set of words as a WideFieldMask. Its binding is
/// immutable like a class's.
#[test]
fn the_polyfill_is_one_lens_read_through_its_classview() {
    let mut s = catalogue();
    let view = View::with_labels(view_class(POLYFILL), &WORDS);
    let lens = s.resolve_algebra(&view, POLYFILL, SchemaRev(1)).unwrap();
    let rows: Vec<(u8, &str)> = lens
        .field_view(&WideFieldMask::from_positions(&[2, 3]))
        .map(|r| (r.position, r.label))
        .collect();
    assert_eq!(rows, vec![(2, WORDS[2]), (3, WORDS[3])]);
    assert_eq!(lens.methods.len(), 0, "its words are fields, not methods");
    // A ClassView whose basis moved under the lens is refused, as for a class.
    let moved = View::with_labels(view_class(POLYFILL), &[WORDS[0]]);
    assert_eq!(
        s.resolve_algebra(&moved, POLYFILL, SchemaRev(1))
            .unwrap_err(),
        Refusal::FieldBasisChanged {
            class: POLYFILL,
            bound: 4,
            view: 1
        }
    );
    assert_eq!(
        s.declare_algebra(POLYFILL, SchemaRev(1), Extent::DR64_CUBE, ARITY.to_vec()),
        Ok(())
    );
    assert_eq!(
        s.declare_algebra(POLYFILL, SchemaRev(1), Extent::DR64_CUBE, vec![0, 0, 2, 3]),
        Err(Refusal::AlgebraImmutable(POLYFILL))
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
        reg(s, XLOG, schema(1, 5), vec![m]).unwrap_err()
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
        refused(&mut s, &|e| e.algebra = word(4)),
        Refusal::UnknownAlgebra {
            class: XLOG,
            method: MethodId(1),
            algebra: word(4)
        }
    );
    assert_eq!(s.class_count(), 0, "no refused class was bound");
    assert_eq!(reg(&mut s, XLOG, schema(1, 5), vec![count]), Ok(()));
}

/// The classid × WideFieldMask ergonomics are the fieldview (the Redmine ERB
/// pattern, OGAR `field_view`): the classid picks the lens, the mask selects
/// positions, and each row carries its position as its layout address, with
/// labels from the ClassView. Behaviour is exposed by ordinal only.
#[test]
fn the_fieldview_projects_mask_positions_with_late_labels() {
    let mut s = catalogue();
    reg(&mut s, XLOG, schema(1, 5), vec![xlog_count()]).unwrap();
    let view = View::with_fields(view_class(XLOG), 5);
    let r = s.resolve(&view, XLOG, SchemaRev(1)).unwrap();

    let mask = WideFieldMask::from_positions(&[4, 0, 9]);
    let rows: Vec<(u8, &str, &str)> = r
        .field_view(&mask)
        .map(|f| (f.position, f.label, f.predicate))
        .collect();
    // Ascending, labels from the lens; 9 is past the basis and selects nothing.
    assert_eq!(rows, vec![(0, "f0", "urn:f0"), (4, "f4", "urn:f4")]);
    assert_eq!(r.field_view(&WideFieldMask::from_positions(&[])).count(), 0);
    assert_eq!(r.action_ordinals().collect::<Vec<_>>(), vec![MethodId(1)]);
}

/// Field meaning through the lens: the fieldview row joined by predicate to
/// the contract's `PropertySpec` (obligation, semantic type, marking). The
/// same pattern yields masks from meaning: a risk mask, an obligation mask.
/// A field with no spec is UNKNOWN and never defaults into a mask.
#[test]
fn meaning_masks_come_from_the_same_fieldview() {
    use lance_graph_contract::cam::CodecRoute;
    use lance_graph_contract::property::{
        Marking, PropertyKind, PropertySchema, PropertySpec, SemanticType,
    };
    static SPECS: [PropertySpec; 3] = [
        PropertySpec::required("urn:f0")
            .with_semantic_type(SemanticType::Iban)
            .with_marking(Marking::Pii),
        PropertySpec::optional("urn:f1", CodecRoute::Passthrough),
        PropertySpec::required("urn:f2").with_marking(Marking::Financial),
        // urn:f3 is deliberately absent: UNKNOWN meaning.
    ];
    let schema = PropertySchema {
        entity_type: "order",
        properties: &SPECS,
    };
    let s = store_with_order();
    let view = View::with_fields(VIEW_CLASS, 4);
    let r = s.resolve(&view, ORDER, SchemaRev(1)).unwrap();

    let rows: Vec<(u8, Option<Marking>)> = r
        .field_meaning(&WideFieldMask::from_positions(&[0, 3]), &schema)
        .map(|m| (m.row.position, m.spec.map(|s| s.marking)))
        .collect();
    assert_eq!(rows, vec![(0, Some(Marking::Pii)), (3, None)]);

    let risk = r.mask_where(&schema, |s| s.marking >= Marking::Pii);
    assert_eq!(risk, WideFieldMask::from_positions(&[0, 2]));
    let required = r.mask_where(&schema, |s| s.kind == PropertyKind::Required);
    assert_eq!(required, WideFieldMask::from_positions(&[0, 2]));
    // Falsifier: a default spec would be Internal; the unspecified field 3
    // must still not join a mask that Internal satisfies.
    let low = r.mask_where(&schema, |s| s.marking <= Marking::Internal);
    assert_eq!(low, WideFieldMask::from_positions(&[1]));
    assert_eq!(r.unspecified(&schema), WideFieldMask::from_positions(&[3]));
}

/// The recipe's order must be what it says: a step reads only declared reads
/// or what an earlier step wrote. XLOG's count with the multiply moved first
/// reads the counts before they exist and is refused.
#[test]
fn a_step_cannot_read_what_no_earlier_step_wrote() {
    let mut s = catalogue();
    let mut mul_first = xlog_count();
    mul_first.effects.rotate_right(1);
    assert_eq!(
        reg(&mut s, XLOG, schema(1, 5), vec![mul_first]),
        Err(Refusal::UndeclaredRead {
            class: XLOG,
            method: MethodId(1),
            effect: 0,
            position: 2
        })
    );
    // A read of a field outside `reads` and never written is refused too.
    let mut stray = xlog_count();
    stray.effects[0].args = vec![Arg::Field(4), Arg::Const(1)];
    assert_eq!(
        reg(&mut s, XLOG, schema(1, 5), vec![stray]),
        Err(Refusal::UndeclaredRead {
            class: XLOG,
            method: MethodId(1),
            effect: 0,
            position: 4
        })
    );
    // The faithful order binds.
    assert_eq!(reg(&mut s, XLOG, schema(1, 5), vec![xlog_count()]), Ok(()));
}

/// Meaning reaches behaviour: a method's footprint (ON, reads, writes)
/// intersected with a risk mask says which actions touch risk.
#[test]
fn a_risk_mask_reaches_the_methods_that_touch_it() {
    let mut s = catalogue();
    reg(&mut s, XLOG, schema(1, 5), vec![xlog_count()]).unwrap();
    let view = View::with_fields(view_class(XLOG), 5);
    let r = s.resolve(&view, XLOG, SchemaRev(1)).unwrap();

    assert_eq!(
        r.methods[0].footprint(),
        WideFieldMask::from_positions(&[0, 1, 2, 3, 4])
    );
    // A mask over the product (written) and one over the inputs (read) both
    // reach the count; a mask over nothing it touches reaches nothing.
    for m in [[4u8], [0u8]] {
        let ids: Vec<MethodId> = r
            .methods_touching(&WideFieldMask::from_positions(&m))
            .map(|m| m.id)
            .collect();
        assert_eq!(ids, vec![MethodId(1)]);
    }
    assert_eq!(
        r.methods_touching(&WideFieldMask::from_positions(&[200]))
            .count(),
        0
    );
}

fn node(id: u32) -> MethodBinding {
    MethodBinding {
        id: MethodId(id),
        when: None,
        on: WideFieldMask::from_positions(&[]),
        reads: WideFieldMask::from_positions(&[]),
        ..method(id, vec![])
    }
}

const ONNX_BIG: u32 = 0x7F03_0002;
const JINA: u32 = 0x7F03_0003;

/// A consumer reserves what it needs per axis, in DR64 or DR256 tiles, so
/// nothing caps it at 64: 200 fields need a 256-wide property axis, and 70
/// methods need a predicate axis wider than one 64 tile. The reservation is
/// part of the immutable binding.
#[test]
fn a_consumer_reserves_its_extent_and_is_not_capped_at_64() {
    let mut s = catalogue();
    assert_eq!(
        reg(&mut s, ONNX, schema(1, 200), vec![node(1)]),
        Err(Refusal::FieldsExceedExtent {
            class: ONNX,
            field_count: 200,
            reserved: 64
        })
    );
    assert_eq!(
        reg(
            &mut s,
            ONNX,
            schema_in(1, 200, Extent::DR256_SQ_X64),
            vec![node(1)]
        ),
        Ok(())
    );

    let nodes: Vec<MethodBinding> = (1..=70).map(node).collect();
    assert_eq!(
        reg(
            &mut s,
            ONNX_BIG,
            schema_in(1, 200, Extent::DR256_SQ_X64),
            nodes.clone()
        ),
        Err(Refusal::MethodsExceedExtent {
            class: ONNX_BIG,
            methods: 70,
            reserved: 64
        })
    );
    let two_predicate_tiles = Extent {
        predicate: Axis::new(Width::W64, 2),
        ..Extent::DR256_SQ_X64
    };
    assert_eq!(
        reg(
            &mut s,
            ONNX_BIG,
            schema_in(1, 200, two_predicate_tiles),
            nodes
        ),
        Ok(())
    );
    assert_eq!(
        reg(
            &mut s,
            ONNX,
            schema_in(1, 200, two_predicate_tiles),
            vec![node(1)]
        ),
        Err(Refusal::BehaviourImmutable(ONNX)),
        "the reservation cannot be changed after binding"
    );
    // An axis wider than it can address is refused, not truncated.
    let too_wide = Extent {
        property: Axis::new(Width::W256, 2),
        ..Extent::DR256_SQ_X64
    };
    assert_eq!(
        reg(&mut s, JINA, schema_in(1, 4, too_wide), vec![]),
        Err(Refusal::ReservationTooWide {
            class: JINA,
            axis: "property",
            reserved: 512
        })
    );
}

/// Thin provisioning: Jina's 7,620 nodes are identity positions. Reserving
/// the whole DR256 identity space claims nothing; routing the 7,620 nodes
/// claims the 30 tiles of 256 they fall in, and the cost is stated for those
/// 30 tiles, not for the 256 reserved. A one-tile reservation refuses node
/// 7,619 instead of silently dropping it.
#[test]
fn jina_reserves_the_dr256_identity_space_and_claims_only_routed_tiles() {
    const NODES: u16 = 7_620;
    let mut s = catalogue();
    reg(
        &mut s,
        JINA,
        schema_in(1, 4, Extent::DR256_GLOBAL_X64),
        vec![node(1)],
    )
    .unwrap();
    assert_eq!(Extent::DR256_GLOBAL_X64.identity.reserved(), 65_536);
    let none = s.claimed(JINA, s.current_version()).unwrap();
    assert_eq!(
        (none.identity_tiles, none.bytes),
        (0, 0),
        "reserving claims nothing"
    );

    for position in 0..NODES {
        s.route(position, JINA).unwrap();
    }
    let claim = s.claimed(JINA, s.current_version()).unwrap();
    assert_eq!(claim.identity_tiles, 30, "ceil(7,620 / 256)");
    // One claimed tile: 256 identities x 256 properties x 64 predicates.
    assert_eq!(Extent::DR256_GLOBAL_X64.tile_bytes(), 512 * 1024);
    assert_eq!(claim.bytes, 30 * 512 * 1024);
    assert!(claim.bytes < Extent::DR256_GLOBAL_X64.reserved_cells() / 8);
    // Replay: before the last tile's first route, 29 tiles were claimed.
    assert_eq!(
        s.claimed(JINA, RouteVersion(7_424)).unwrap().identity_tiles,
        29
    );

    // A one-tile reservation refuses a node past it, naming the reservation.
    reg(
        &mut s,
        ONNX,
        schema_in(1, 4, Extent::DR256_SQ_X64),
        vec![node(1)],
    )
    .unwrap();
    let before = s.current_version();
    assert_eq!(
        s.route(NODES - 1, ONNX),
        Err(Refusal::OutsideReservation {
            class: ONNX,
            position: NODES - 1,
            reserved: 256
        })
    );
    assert_eq!(
        s.current_version(),
        before,
        "a refused route moves no version"
    );
}

// --------------------------------------------------------------------------
// D-DT64-7 O9 / L1: a behaviour lens (right half 0xF###) reads its basis by
// its full classid, through ClassView::lens_fields; it never inherits the
// basis of its left half.

/// A view holding concept bases and behaviour-lens bases separately.
struct Lenses {
    concepts: Vec<(ClassId, Vec<FieldRef>)>,
    lenses: Vec<(u32, Vec<FieldRef>)>,
}

fn labels(prefix: &str, n: usize) -> Vec<FieldRef> {
    (0..n)
        .map(|i| FieldRef::new(format!("urn:{prefix}/{i}"), format!("{prefix}{i}")))
        .collect()
}

impl ClassView for Lenses {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        self.concepts
            .iter()
            .find(|(c, _)| *c == class)
            .map_or(&[], |(_, f)| f.as_slice())
    }
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        self.lenses
            .iter()
            .find(|(l, _)| *l == lens)
            .map(|(_, f)| f.as_slice())
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

const RELALG_LENS: u32 = 0xFFFF_F000;
const ONNX_LENS: u32 = 0xFFFF_F200;
const UNKNOWN_LENS: u32 = 0xFFFF_F300;

#[test]
fn two_behaviour_lenses_under_one_left_half_resolve_independently() {
    let mut s = BindStore::new();
    s.declare_algebra(RELALG_LENS, SchemaRev(1), Extent::DR64_CUBE, vec![2; 17])
        .unwrap();
    s.declare_algebra(ONNX_LENS, SchemaRev(1), Extent::DR64_CUBE, vec![1; 5])
        .unwrap();
    let view = Lenses {
        // The concept 0xFFFF itself carries a basis of neither size.
        concepts: vec![(0xFFFF, labels("concept", 3))],
        lenses: vec![
            (RELALG_LENS, labels("relalg", 17)),
            (ONNX_LENS, labels("onnx", 5)),
        ],
    };
    let relalg = s.resolve_algebra(&view, RELALG_LENS, SchemaRev(1)).unwrap();
    let onnx = s.resolve_algebra(&view, ONNX_LENS, SchemaRev(1)).unwrap();
    assert_eq!(relalg.fields.len(), 17);
    assert_eq!(onnx.fields.len(), 5);
    assert_eq!(relalg.fields[0].label, "relalg0");
    assert_eq!(onnx.fields[0].label, "onnx0");
}

#[test]
fn a_data_class_and_a_behaviour_lens_sharing_a_left_half_do_not_cross_read() {
    const DATA: u32 = 0x7F0B_0001;
    const LENS: u32 = 0x7F0B_F001;
    let view = Lenses {
        concepts: vec![(0x7F0B, labels("data", 2))],
        lenses: vec![(LENS, labels("word", 3))],
    };
    let mut s = BindStore::new();
    s.register(&view, DATA, schema(1, 2), vec![]).unwrap();
    s.declare_algebra(LENS, SchemaRev(1), Extent::DR64_CUBE, vec![1; 3])
        .unwrap();
    let data = s.resolve(&view, DATA, SchemaRev(1)).unwrap();
    let lens = s.resolve_algebra(&view, LENS, SchemaRev(1)).unwrap();
    assert_eq!(data.fields.len(), 2);
    assert_eq!(data.fields[1].label, "data1");
    assert_eq!(lens.fields.len(), 3);
    assert_eq!(lens.fields[2].label, "word2");
}

#[test]
fn an_unknown_behaviour_lens_fails_closed_instead_of_inheriting_its_concept() {
    let mut s = BindStore::new();
    s.declare_algebra(UNKNOWN_LENS, SchemaRev(1), Extent::DR64_CUBE, vec![1; 3])
        .unwrap();
    // The concept's basis has exactly the lens's word count: inheriting it
    // would succeed, which is the failure this guards against.
    let view = Lenses {
        concepts: vec![(0xFFFF, labels("concept", 3))],
        lenses: vec![],
    };
    assert_eq!(
        s.resolve_algebra(&view, UNKNOWN_LENS, SchemaRev(1))
            .unwrap_err(),
        Refusal::UnknownLensBasis(UNKNOWN_LENS)
    );
}

#[test]
fn a_lens_outside_the_behaviour_range_still_reads_its_concept() {
    // Backward-compatibility twin: right half 0x0001 is not a behaviour
    // lens, so the concept basis answers and `lens_fields` is not asked.
    let mut s = BindStore::new();
    s.declare_algebra(POLYFILL, SchemaRev(1), Extent::DR64_CUBE, ARITY.to_vec())
        .unwrap();
    let view = Lenses {
        concepts: vec![(view_class(POLYFILL), labels("w", ARITY.len()))],
        lenses: vec![(POLYFILL, labels("ignored", 1))],
    };
    let lens = s.resolve_algebra(&view, POLYFILL, SchemaRev(1)).unwrap();
    assert_eq!(lens.fields.len(), ARITY.len());
    assert_eq!(lens.fields[0].label, "w0");
}

#[test]
fn a_view_that_does_not_implement_lens_fields_still_compiles_and_refuses_behaviour_lenses() {
    // `View` above implements only the required methods: the default
    // `lens_fields` answers None, so a behaviour lens fails closed.
    let mut s = BindStore::new();
    s.declare_algebra(RELALG_LENS, SchemaRev(1), Extent::DR64_CUBE, vec![1; 2])
        .unwrap();
    let view = View::with_fields(0xFFFF, 2);
    assert_eq!(view.lens_fields(RELALG_LENS), None);
    assert_eq!(
        s.resolve_algebra(&view, RELALG_LENS, SchemaRev(1))
            .unwrap_err(),
        Refusal::UnknownLensBasis(RELALG_LENS)
    );
}

/// A store for the `check_route` tests: the catalogue plus ONNX with a
/// one-tile reservation (256 identity positions).
fn routing_store() -> BindStore {
    let mut s = catalogue();
    reg(
        &mut s,
        ONNX,
        schema_in(1, 4, Extent::DR256_SQ_X64),
        vec![node(1)],
    )
    .unwrap();
    s
}

/// `check_route` answers exactly what `route` would, on every case: an
/// unknown class, the last position inside the reservation, the first
/// outside it, and a position far past it. Red if it skips either of
/// `route`'s two refusals (disables: drop the class lookup; drop the
/// reservation check from `check_route` only).
#[test]
fn check_route_agrees_with_route_on_every_case() {
    let s = routing_store();
    let cases = [
        (7, 0xBEEF_0001),
        (0, ONNX),
        (255, ONNX),
        (256, ONNX),
        (7_619, ONNX),
    ];
    let (mut accepted, mut unknown, mut outside) = (0, 0, 0);
    for (position, class) in cases {
        let checked = s.check_route(position, class);
        let routed = s.clone().route(position, class).map(|_| ());
        assert_eq!(checked, routed, "position {position}, class {class:#x}");
        match checked {
            Ok(()) => accepted += 1,
            Err(Refusal::UnknownClass(_)) => unknown += 1,
            Err(Refusal::OutsideReservation { .. }) => outside += 1,
            Err(other) => panic!("unexpected refusal {other:?}"),
        }
    }
    // Anti-vacuity: both outcomes and both refusals occur.
    assert_eq!((accepted, unknown, outside), (2, 1, 2));
}

/// `check_route` writes nothing: no route event, no new `RouteVersion`, no
/// claimed tile, for accepted and refused intents alike. Twin: `route` on the
/// same intent does move all three, so the comparison can fail.
#[test]
fn check_route_writes_nothing() {
    let mut s = routing_store();
    let v = s.current_version();
    let claim = s.claimed(ONNX, v).unwrap();
    for (position, class) in [(3, ONNX), (256, ONNX), (3, 0xBEEF_0001)] {
        let _ = s.check_route(position, class);
    }
    assert_eq!(s.current_version(), v, "no version issued");
    assert_eq!(s.routed_at(s.current_version(), 3), Ok(None));
    assert_eq!(s.claimed(ONNX, s.current_version()).unwrap(), claim);

    s.route(3, ONNX).unwrap();
    assert_ne!(s.current_version(), v, "route does move the version");
    assert_eq!(s.routed_at(s.current_version(), 3), Ok(Some(ONNX)));
}

/// A checked route intent stays valid while the store grows: other classes
/// are bound and other positions routed or retracted, and the same intent is
/// still accepted, then applied without a refusal (the property a prepared
/// land relies on to apply with no error path).
#[test]
fn a_checked_route_intent_stays_valid_as_the_store_grows() {
    let mut s = routing_store();
    assert_eq!(s.check_route(42, ONNX), Ok(()));
    reg(
        &mut s,
        JINA,
        schema_in(1, 4, Extent::DR256_GLOBAL_X64),
        vec![node(1)],
    )
    .unwrap();
    s.route(42, JINA).unwrap();
    s.route(7, ONNX).unwrap();
    s.retract(7);
    assert_eq!(s.check_route(42, ONNX), Ok(()));
    let v = s.route(42, ONNX).expect("a checked intent applies");
    assert_eq!(s.routed_at(v, 42), Ok(Some(ONNX)));
}

/// O9 on the `register` / `resolve` path, not only on `resolve_algebra`: a
/// classid with a `0xF###` right half registered as a class is read by its
/// FULL classid, so it gets its lens basis, and the data class under the same
/// left half keeps the concept basis.
#[test]
fn a_lens_classid_registered_as_a_class_reads_its_own_basis() {
    const DATA: u32 = 0x7F0B_0001;
    const LENS: u32 = 0x7F0B_F001;
    let view = Lenses {
        concepts: vec![(0x7F0B, labels("data", 2))],
        lenses: vec![(LENS, labels("word", 3))],
    };
    let mut s = BindStore::new();
    s.register(&view, DATA, schema(1, 2), vec![]).unwrap();
    s.register(&view, LENS, schema(1, 3), vec![]).unwrap();
    let lens = s.resolve(&view, LENS, SchemaRev(1)).unwrap();
    let data = s.resolve(&view, DATA, SchemaRev(1)).unwrap();
    assert_eq!(lens.fields[2].predicate_iri, "urn:word/2");
    assert_eq!(data.fields[1].predicate_iri, "urn:data/1");
    // A view that knows the concept but not the lens fails closed for the
    // lens classid instead of handing it the concept's basis.
    let concept_only = Lenses {
        concepts: vec![(0x7F0B, labels("data", 3))],
        lenses: vec![],
    };
    assert_eq!(
        s.resolve(&concept_only, LENS, SchemaRev(1)).unwrap_err(),
        Refusal::UnknownLensBasis(LENS)
    );
}

/// The revision is checked before the view is consulted, so a view is never
/// asked about a revision the class was not bound at: an unknown lens asked at
/// the wrong revision is a revision mismatch, not an unknown basis.
#[test]
fn the_revision_is_checked_before_the_basis_is_looked_up() {
    let mut s = BindStore::new();
    s.declare_algebra(UNKNOWN_LENS, SchemaRev(1), Extent::DR64_CUBE, vec![1; 2])
        .unwrap();
    let view = Lenses {
        concepts: vec![],
        lenses: vec![],
    };
    assert_eq!(
        s.resolve_algebra(&view, UNKNOWN_LENS, SchemaRev(2))
            .unwrap_err(),
        Refusal::SchemaRevisionMismatch {
            class: UNKNOWN_LENS,
            bound: SchemaRev(1),
            asked: SchemaRev(2)
        }
    );
    assert_eq!(
        s.resolve_algebra(&view, UNKNOWN_LENS, SchemaRev(1))
            .unwrap_err(),
        Refusal::UnknownLensBasis(UNKNOWN_LENS)
    );
}

/// Registration reads the source's HISTORY for the declared revision, not its
/// current basis: a source that has moved on (current = reordered) still
/// registers ORDER at revision 1 against its revision-1 record.
#[test]
fn registration_reads_the_sources_history_for_the_declared_revision() {
    let rev1 = View::with_fields(VIEW_CLASS, 4).fields;
    let mut moved = rev1.clone();
    moved.swap(0, 1);
    let source = History {
        class: VIEW_CLASS,
        current: moved,
        history: vec![(1, rev1)],
        lens_history: Vec::new(),
    };
    let mut s = BindStore::new();
    s.register(&source, ORDER, schema(1, 4), vec![]).unwrap();
    let r = s.resolve(&source, ORDER, SchemaRev(1)).unwrap();
    assert_eq!(iris(r.fields), ["urn:f0", "urn:f1", "urn:f2", "urn:f3"]);
    // The revision-1 basis, offered as a plain current view, is the same one.
    assert!(s
        .resolve(&View::with_fields(VIEW_CLASS, 4), ORDER, SchemaRev(1))
        .is_ok());
}

/// A same-count evolution: revision 2 reorders revision 1. With history, each
/// classid replays against its own revision; through the current basis alone,
/// the revision-1 class is refused by the witness, not by the count.
#[test]
fn a_same_count_reorder_across_revisions_replays_only_through_history() {
    let rev1 = View::with_fields(VIEW_CLASS, 4).fields;
    let mut rev2 = rev1.clone();
    rev2.swap(0, 1);
    let source = History {
        class: VIEW_CLASS,
        current: rev2.clone(),
        history: vec![(1, rev1.clone()), (2, rev2.clone())],
        lens_history: Vec::new(),
    };
    let mut s = BindStore::new();
    s.register(&source, ORDER, schema(1, 4), vec![]).unwrap();
    s.register_derived(&source, ORDER, ORDER_V2, schema(2, 4), vec![])
        .unwrap();
    assert_eq!(
        iris(s.resolve(&source, ORDER, SchemaRev(1)).unwrap().fields)[0],
        "urn:f0"
    );
    assert_eq!(
        iris(s.resolve(&source, ORDER_V2, SchemaRev(2)).unwrap().fields)[0],
        "urn:f1"
    );
    let current_only = View {
        class: VIEW_CLASS,
        fields: rev2,
    };
    assert_eq!(
        s.resolve(&current_only, ORDER, SchemaRev(1)).unwrap_err(),
        Refusal::PredicateBasisChanged {
            class: ORDER,
            schema_rev: SchemaRev(1),
            source: BasisSource::Current,
        }
    );
}

/// One revision has one basis: two classids read through one concept half
/// cannot bind two different bases at the same revision. The same basis, or
/// another revision, is accepted.
#[test]
fn two_classids_of_one_concept_cannot_bind_two_bases_at_one_revision() {
    let mut s = BindStore::new();
    s.register(&view_of(&BOUND), ORDER, schema(1, 4), vec![])
        .unwrap();
    let swapped = view_of(&[BOUND[1], BOUND[0], BOUND[2], BOUND[3]]);
    assert_eq!(
        s.register_derived(&swapped, ORDER, ORDER_V2, schema(1, 4), vec![]),
        Err(Refusal::BasisConflict {
            class: ORDER_V2,
            other: ORDER,
            schema_rev: SchemaRev(1),
        })
    );
    assert_eq!(
        s.register_derived(&swapped, ORDER, ORDER_V2, schema(2, 4), vec![]),
        Ok(())
    );
    const ORDER_V3: u32 = 0x0A1E_0003;
    assert_eq!(
        s.register_derived(&view_of(&BOUND), ORDER, ORDER_V3, schema(1, 4), vec![]),
        Ok(())
    );
}

/// A lens classid registered as a class through a view that knows only its
/// concept is refused at registration: the concept's basis is never taken.
#[test]
fn a_lens_classid_cannot_register_through_its_concepts_basis() {
    const LENS: u32 = 0x7F0B_F001;
    let concept_only = Lenses {
        concepts: vec![(0x7F0B, labels("data", 3))],
        lenses: vec![],
    };
    let mut s = BindStore::new();
    assert_eq!(
        s.register(&concept_only, LENS, schema(1, 3), vec![]),
        Err(Refusal::UnknownLensBasis(LENS))
    );
}

/// FINDING (contract gap, Decision J item 5): a `ClassView` does not say which
/// revision its CURRENT basis is. Registering at revision 1 through a view
/// that keeps no history and has moved on to another basis is accepted, and
/// that basis becomes revision 1's. The store cannot refuse what it cannot
/// see; closing this needs the source to answer for the revision it is
/// asked about (`fields_at`) or to report its current revision.
#[test]
fn finding_registration_through_a_current_only_view_trusts_the_declared_revision() {
    let moved_on = view_of(&[BOUND[1], BOUND[0], BOUND[2], BOUND[3]]);
    let mut s = BindStore::new();
    assert_eq!(s.register(&moved_on, ORDER, schema(1, 4), vec![]), Ok(()));
    let r = s.resolve(&moved_on, ORDER, SchemaRev(1)).unwrap();
    assert_eq!(r.fields[0].predicate_iri, "urn:f1");
}

/// FINDING (L5c / L5d): a polyfill lens binds arities, not a basis, so two
/// words of the same arity can trade places in the view and `resolve_algebra`
/// still accepts it. A consumer that picks behaviour by a word's label is not
/// covered by the class witness.
#[test]
fn finding_an_equal_arity_word_swap_in_a_polyfill_lens_resolves() {
    const LENS: u32 = 0x7F0A_0001;
    let mut s = BindStore::new();
    s.declare_algebra(LENS, SchemaRev(1), Extent::DR64_CUBE, vec![2, 2])
        .unwrap();
    let words = |a: &str, b: &str| Lenses {
        concepts: vec![(0x7F0A, vec![FieldRef::new(a, a), FieldRef::new(b, b)])],
        lenses: vec![],
    };
    let before = words("urn:groupcount", "urn:mul");
    let after = words("urn:mul", "urn:groupcount");
    let a = s.resolve_algebra(&before, LENS, SchemaRev(1)).unwrap();
    let b = s.resolve_algebra(&after, LENS, SchemaRev(1)).unwrap();
    assert_ne!(a.fields[0].predicate_iri, b.fields[0].predicate_iri);
}

/// FINDING (contract gap, with the one above): `BasisConflict` lets the FIRST
/// registration anchor a revision. A stale registration through a moved-on,
/// history-less view therefore locks out a later registration through the
/// correct basis of that revision. Fail closed, but the wrong one wins.
#[test]
fn finding_a_stale_first_registration_locks_out_the_correct_basis() {
    let moved_on = view_of(&[BOUND[1], BOUND[0], BOUND[2], BOUND[3]]);
    let mut s = BindStore::new();
    s.register(&moved_on, ORDER, schema(1, 4), vec![]).unwrap();
    assert!(matches!(
        s.register_derived(&view_of(&BOUND), ORDER, ORDER_V2, schema(1, 4), vec![]),
        Err(Refusal::BasisConflict { .. })
    ));
}

// ── L5c: the issuer stamp and the binding digest ─────────────────────────────

/// Store A binds method 10 as ASSIGN(2) then ACCUMULATE(3); every helper
/// below builds stores the same way, so two stores differ only where a test
/// says they do.
fn store_with(effects: Vec<Effect>, name: &str) -> BindStore {
    let mut s = catalogue();
    let m = MethodBinding {
        name: name.to_string(),
        ..method(10, effects)
    };
    reg(&mut s, ORDER, schema(1, 4), vec![m]).unwrap();
    s
}

fn forward() -> Vec<Effect> {
    vec![effect(2, ASSIGN), effect(3, ACCUMULATE)]
}

/// Positive: the store admits its own references, and two references it
/// issues for one method are equal (one stamp).
#[test]
fn a_same_store_twin_admits() {
    let s = store_with(forward(), "m10");
    let r1 = s.method_ref(ORDER, MethodId(10)).unwrap();
    let r2 = s.method_ref(ORDER, MethodId(10)).unwrap();
    assert_eq!(r1, r2, "one store, one method, one stamp");
    assert_eq!(s.admit(&r1).unwrap().effects, forward());
    assert_eq!(s.admit(&r2).unwrap().effects, forward());
}

/// n12 (P1-5): store B binds the same class, method, signature revision and
/// guard with the effects reversed. A's reference is refused by B for the
/// binding, which is checked before the issuer.
#[test]
fn a_reference_to_reversed_effects_is_refused_as_another_binding() {
    let a = store_with(forward(), "m10");
    let b = store_with(vec![effect(3, ACCUMULATE), effect(2, ASSIGN)], "m10");
    let ra = a.method_ref(ORDER, MethodId(10)).unwrap();
    let rb = b.method_ref(ORDER, MethodId(10)).unwrap();
    // Everything the pre-L5c reference carried is equal.
    assert_eq!(
        (ra.class, ra.method, ra.sig_rev, &ra.guard),
        (rb.class, rb.method, rb.sig_rev, &rb.guard)
    );
    assert_eq!(
        b.admit(&ra).unwrap_err(),
        Refusal::BindingMismatch {
            class: ORDER,
            method: MethodId(10)
        }
    );
    assert!(b.admit(&rb).is_ok(), "B admits its own reference");
}

/// A store with the same binding is still another store. Renaming the method
/// keeps the digest, so the refusal is the issuer, not the binding.
#[test]
fn a_same_binding_from_another_store_is_refused_as_foreign() {
    let a = store_with(forward(), "m10");
    let b = store_with(forward(), "renamed");
    let ra = a.method_ref(ORDER, MethodId(10)).unwrap();
    assert_eq!(
        b.admit(&ra).unwrap_err(),
        Refusal::ForeignStore { class: ORDER },
        "equal digest (name is not part of it), other issuer"
    );
    assert_eq!(
        b.check_issuer(&ra),
        Err(Refusal::ForeignStore { class: ORDER })
    );
    assert_eq!(a.check_issuer(&ra), Ok(()));
    // Fresh stores have distinct identities.
    assert_ne!(ra, b.method_ref(ORDER, MethodId(10)).unwrap());
}

/// D2: a clone is another store. It refuses the original's references, the
/// original refuses its references, and each admits its own.
#[test]
fn a_cloned_store_refuses_the_originals_reference() {
    let a = store_with(forward(), "m10");
    let c = a.clone();
    let ra = a.method_ref(ORDER, MethodId(10)).unwrap();
    let rc = c.method_ref(ORDER, MethodId(10)).unwrap();
    assert_eq!(
        c.admit(&ra).unwrap_err(),
        Refusal::ForeignStore { class: ORDER }
    );
    assert_eq!(
        a.admit(&rc).unwrap_err(),
        Refusal::ForeignStore { class: ORDER }
    );
    assert!(a.admit(&ra).is_ok());
    assert!(c.admit(&rc).is_ok());
}

/// A stamp copied onto a reference to another method of the same store is
/// refused: the digest is per method, not per store.
#[test]
fn a_stamp_moved_to_another_method_is_refused() {
    let mut s = catalogue();
    let a = method(20, vec![effect(2, ASSIGN)]);
    let b = method(21, vec![effect(2, ASSIGN)]);
    reg(&mut s, ORDER, schema(1, 4), vec![a, b]).unwrap();
    let ra = s.method_ref(ORDER, MethodId(20)).unwrap();
    let rb = s.method_ref(ORDER, MethodId(21)).unwrap();
    let moved = MethodRef {
        method: MethodId(21),
        ..ra
    };
    assert_eq!(
        s.admit(&moved).unwrap_err(),
        Refusal::BindingMismatch {
            class: ORDER,
            method: MethodId(21)
        }
    );
    assert!(s.admit(&rb).is_ok());
}

/// `method_ref` issues only for bound methods.
#[test]
fn method_ref_refuses_what_is_not_bound() {
    let s = store_with(forward(), "m10");
    assert_eq!(
        s.method_ref(ORDER_V2, MethodId(10)).unwrap_err(),
        Refusal::UnknownClass(ORDER_V2)
    );
    assert_eq!(
        s.method_ref(ORDER, MethodId(11)).unwrap_err(),
        Refusal::UnknownMethod {
            class: ORDER,
            method: MethodId(11)
        }
    );
}
