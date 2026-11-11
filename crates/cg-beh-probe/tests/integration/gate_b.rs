//! Gate B: the four kinds of change are told apart, each with its evidence
//! and the obligations it makes stale. Every predicted admission answer is
//! compared with what the stores answer.

use cg_beh_probe::fixture::{self, FoundryVariant, Variant};
use cg_beh_probe::query::{diff, impl_change, word_impact, Component, Obligation, OldRef, Status};
use cg_beh_probe::Grade;
use foundry_parity_lab::method;
use lance_graph_dt64::{BasisSource, MethodId, Refusal};
use lance_graph_r2il::polyfill::{position_of, POLYFILL, POLYFILL_LENS};

use crate::common::{delta, foundry, lab, observed, real_dispatch};

const CLASS: u32 = fixture::CLASS;

#[test]
fn a_changed_on_declaration_moves_the_binding_and_the_trigger_set() {
    let ((old, before), (new, after)) = (lab(Variant::Base), lab(Variant::OnMoved));
    let ds = diff(&before, &after).unwrap();
    let d = delta(&ds, fixture::GUARDED);
    assert_eq!(d.status, Status::Changed(vec![Component::On]));
    assert!(d.binding_changed);
    assert_eq!(
        d.stale,
        [Obligation::MethodRefStamp, Obligation::TriggerSet]
    );
    assert!(d
        .evidence
        .iter()
        .all(|e| e.grade == Grade::Dt64Registration));
    assert_eq!(d.old_ref(), Some(OldRef::Binding));
    assert_eq!(
        observed(&old.store, &new.store, CLASS, fixture::GUARDED),
        OldRef::Binding
    );
    // The trigger set really moved.
    let on = |w: &fixture::TestWorld, field: u8| {
        w.store
            .resolve(&w.view, CLASS, fixture::REV)
            .unwrap()
            .methods_on(field)
            .map(|m| m.id.0)
            .collect::<Vec<_>>()
    };
    assert_eq!(on(&old, 0), [fixture::GUARDED, fixture::INDEPENDENT]);
    assert_eq!(on(&new, 0), [fixture::INDEPENDENT]);
}

#[test]
fn a_changed_guard_moves_the_witness_and_not_the_binding() {
    let ((old, before), (new, after)) = (lab(Variant::Base), lab(Variant::GuardMoved));
    let ds = diff(&before, &after).unwrap();
    let d = delta(&ds, fixture::GUARDED);
    assert_eq!(d.status, Status::Changed(vec![Component::When]));
    assert!(!d.binding_changed);
    assert_eq!(d.stale, [Obligation::GuardWitness]);
    assert_eq!(d.old_ref(), Some(OldRef::Guard));
    assert_eq!(
        observed(&old.store, &new.store, CLASS, fixture::GUARDED),
        OldRef::Guard
    );
    // DO is untouched.
    let steps = |s: &cg_beh_probe::facts::Snapshot| {
        s.method(CLASS, fixture::GUARDED).unwrap().steps.clone()
    };
    assert_eq!(steps(&before), steps(&after));
}

#[test]
fn a_respelled_guard_keeps_its_witness_and_makes_nothing_stale() {
    let ((old, before), (new, after)) = (lab(Variant::Base), lab(Variant::GuardRespelled));
    let ds = diff(&before, &after).unwrap();
    let d = delta(&ds, fixture::GUARDED);
    assert_eq!(d.status, Status::Changed(vec![Component::WhenSpelling]));
    assert!(d.stale.is_empty());
    assert_eq!(d.old_ref(), Some(OldRef::IssuerOnly));
    assert_eq!(
        observed(&old.store, &new.store, CLASS, fixture::GUARDED),
        OldRef::IssuerOnly
    );
}

#[test]
fn a_changed_field_basis_reaches_every_method_of_the_class() {
    let ((old, before), (new, after)) = (lab(Variant::Base), lab(Variant::BasisSwapped));
    let ds = diff(&before, &after).unwrap();
    assert_eq!(ds.len(), 3);
    for d in &ds {
        assert_eq!(d.status, Status::Changed(vec![Component::Basis]));
        assert_eq!(
            d.stale,
            [Obligation::WitnessedResolve, Obligation::MethodRefStamp]
        );
        assert_eq!(d.old_ref(), Some(OldRef::Binding));
        assert_eq!(
            observed(&old.store, &new.store, CLASS, d.method),
            OldRef::Binding
        );
    }
    // Same positions, other predicates under them.
    let on = |s: &cg_beh_probe::facts::Snapshot| {
        s.method(CLASS, fixture::GUARDED).unwrap().on[0].clone()
    };
    assert_eq!((on(&before).position, on(&after).position), (0, 0));
    assert_eq!(
        (
            on(&before).predicate.as_str(),
            on(&after).predicate.as_str()
        ),
        ("lab:beh/a", "lab:beh/b")
    );
    // The witnessed lookup of the old store refuses the new basis ...
    assert_eq!(
        old.store
            .resolve(&new.view, CLASS, fixture::REV)
            .unwrap_err(),
        Refusal::PredicateBasisChanged {
            class: CLASS,
            schema_rev: fixture::REV,
            source: BasisSource::Current
        }
    );
    // ... and `admit`, which takes no view, still admits the old reference.
    let r = old
        .store
        .method_ref(CLASS, MethodId(fixture::GUARDED))
        .unwrap();
    assert!(old.store.admit(&r).is_ok());
}

#[test]
fn a_changed_do_implementation_changes_no_registered_fact() {
    let (_, snap) = lab(Variant::Base);
    let (src, dispatch) = real_dispatch();
    let less = position_of("IntSLess").unwrap();
    let arm = dispatch.arm_of_word(less).unwrap();
    assert_eq!(arm.names, ["INT_S_LESS"]);

    let hit = impl_change(&snap, &dispatch, &src, POLYFILL_LENS, arm.start + 1).unwrap();
    assert_eq!(hit.words.len(), 1);
    assert_eq!(hit.words[0].word, less);
    assert_eq!(hit.words[0].users, [(CLASS, fixture::GUARDED, 0)]);
    assert_eq!(
        hit.words[0].stale,
        [Obligation::ExecutedParity {
            lens: POLYFILL_LENS,
            word: less
        }]
    );
    let grades: Vec<Grade> = hit.evidence.iter().map(|e| e.grade).collect();
    assert_eq!(
        grades,
        [
            Grade::SourceStructure,
            Grade::SourceStructure,
            Grade::Dt64Registration
        ]
    );

    // The registration is the same before and after such a change.
    let (_, again) = lab(Variant::Base);
    assert!(diff(&snap, &again)
        .unwrap()
        .iter()
        .all(|d| d.status == Status::Unchanged && d.stale.is_empty()));
}

#[test]
fn an_arm_shared_by_three_opcodes_reaches_all_three_words() {
    let (_, snap) = lab(Variant::Base);
    let (src, dispatch) = real_dispatch();
    let add = position_of("IntAdd").unwrap();
    let arm = dispatch.arm_of_word(add).unwrap();
    assert_eq!(arm.names, ["INT_ADD", "INT_MULT", "INT_S_DIV"]);
    let hit = impl_change(&snap, &dispatch, &src, POLYFILL_LENS, arm.start).unwrap();
    let by_word = |name: &str| {
        hit.words
            .iter()
            .find(|w| w.word == position_of(name).unwrap())
            .unwrap()
    };
    assert_eq!(by_word("IntAdd").users, [(CLASS, fixture::INDEPENDENT, 0)]);
    // No user in this snapshot: nothing is claimed stale, and nothing is
    // claimed safe.
    assert!(by_word("IntMult").users.is_empty());
    assert!(by_word("IntMult").stale.is_empty());
    assert_eq!(hit.words.len(), 3);
}

#[test]
fn can_cook_is_told_apart_the_same_way() {
    let target = method::method_id(0, 1).0;
    let (old, before) = foundry(FoundryVariant::Base);

    let (new, after) = foundry(FoundryVariant::GuardThreshold);
    let ds = diff(&before, &after).unwrap();
    assert_eq!(ds.len(), 16);
    let d = delta(&ds, target);
    assert_eq!(d.status, Status::Changed(vec![Component::When]));
    assert!(!d.binding_changed);
    assert_eq!(
        observed(&old.store, &new.store, fixture::FOUNDRY_CLASS, target),
        OldRef::Guard
    );
    assert_eq!(
        ds.iter().filter(|d| d.status == Status::Unchanged).count(),
        15
    );

    let (new, after) = foundry(FoundryVariant::OnNarrowed);
    let ds = diff(&before, &after).unwrap();
    let d = delta(&ds, target);
    assert_eq!(d.status, Status::Changed(vec![Component::On]));
    assert_eq!(
        d.stale,
        [Obligation::MethodRefStamp, Obligation::TriggerSet]
    );
    assert_eq!(
        observed(&old.store, &new.store, fixture::FOUNDRY_CLASS, target),
        OldRef::Binding
    );
    assert_eq!(
        ds.iter().filter(|d| d.status == Status::Unchanged).count(),
        15
    );
}

#[test]
fn no_can_cook_registration_is_reached_through_do() {
    // The negative control: CanCook has no DO step, so a changed word
    // implementation has no DO edge to it. Its guard is lowered by relalg,
    // a path this gate does not map.
    let (_, snap) = foundry(FoundryVariant::Base);
    let (src, dispatch) = real_dispatch();
    for word in 0..POLYFILL.len() as u8 {
        let w = word_impact(&snap, &dispatch, &src, POLYFILL_LENS, word).unwrap();
        assert!(w.users.is_empty());
        assert!(w.stale.is_empty());
    }
}
