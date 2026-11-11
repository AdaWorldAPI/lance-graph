//! Gate C: independent falsifiers. Each alters one thing and checks that
//! exactly that is reported, or that the reading is refused.

use cg_beh_probe::dispatch::{self, DispatchError};
use cg_beh_probe::facts::{class_facts, extract, Missing};
use cg_beh_probe::fixture::{self, FoundryVariant, SwappedBasis, Variant};
use cg_beh_probe::query::{
    diff, impl_change, word_impact, Component, DiffRefusal, ImplRefusal, Obligation, OldRef, Status,
};
use foundry_parity_lab::bind::STOCK0;
use foundry_parity_lab::method;
use lance_graph_contract::class_view::ClassView;
use lance_graph_dt64::dr::Extent;
use lance_graph_dt64::{BasisSource, BindStore, ClassSchema, MethodId, Refusal};
use lance_graph_r2il::polyfill::{polyfill_arities, position_of, POLYFILL_LENS};

use crate::common::{delta, foundry, lab, observed, real_dispatch};

const CLASS: u32 = fixture::CLASS;

#[test]
fn a_basis_altered_at_the_same_field_count_is_refused_not_read() {
    let (world, _) = foundry(FoundryVariant::Base);
    let swapped = SwappedBasis::of(&world, STOCK0, STOCK0 + 1);
    assert_eq!(
        swapped.fields(foundry_parity_lab::bind::VIEW_CLASS).len(),
        world
            .view
            .fields(foundry_parity_lab::bind::VIEW_CLASS)
            .len()
    );
    let refused = Refusal::PredicateBasisChanged {
        class: fixture::FOUNDRY_CLASS,
        schema_rev: fixture::FOUNDRY_READ.schema_rev,
        source: BasisSource::Current,
    };
    assert_eq!(
        class_facts(
            &world.store,
            &swapped,
            fixture::FOUNDRY_CLASS,
            fixture::FOUNDRY_READ.schema_rev
        )
        .unwrap_err(),
        Missing::Basis(refused.clone())
    );
    let r = world
        .store
        .method_ref(fixture::FOUNDRY_CLASS, method::method_id(0, 1))
        .unwrap();
    assert_eq!(
        extract(&world.store, &swapped, &r, &fixture::FOUNDRY_READ).unwrap_err(),
        Missing::Basis(refused)
    );
    // The same reference over the bound basis reads.
    assert!(extract(&world.store, &world.view, &r, &fixture::FOUNDRY_READ).is_ok());
}

#[test]
fn reversing_independent_do_steps_is_a_do_change_and_nothing_else() {
    let ((old, before), (new, after)) = (lab(Variant::Base), lab(Variant::DoReversed));
    let ds = diff(&before, &after).unwrap();
    let d = delta(&ds, fixture::INDEPENDENT);
    assert_eq!(d.status, Status::Changed(vec![Component::Do]));
    assert_eq!(d.stale, [Obligation::MethodRefStamp]);
    assert_eq!(
        observed(&old.store, &new.store, CLASS, fixture::INDEPENDENT),
        OldRef::Binding
    );
    let words = |s: &cg_beh_probe::facts::Snapshot| {
        s.method(CLASS, fixture::INDEPENDENT)
            .unwrap()
            .steps
            .iter()
            .map(|st| (st.ordinal, st.word))
            .collect::<Vec<_>>()
    };
    let (add, sub) = (
        position_of("IntAdd").unwrap(),
        position_of("IntSub").unwrap(),
    );
    assert_eq!(words(&before), [(0, add), (1, sub)]);
    assert_eq!(words(&after), [(0, sub), (1, add)]);
}

#[test]
fn reversing_dependent_do_steps_does_not_register() {
    // GUARDED's second step reads what its first wrote. Reversed, the store
    // refuses the binding, so there are no facts to misread.
    let mut methods = fixture::methods(Variant::Base);
    methods[0].effects.reverse();
    let view = fixture::TestView::new(false);
    let mut store = BindStore::new();
    store
        .declare_algebra(
            POLYFILL_LENS,
            fixture::REV,
            Extent::DR64_CUBE,
            polyfill_arities(),
        )
        .unwrap();
    let schema = ClassSchema {
        schema_rev: fixture::REV,
        field_count: fixture::FIELDS.len(),
        extent: Extent::DR64_CUBE,
    };
    assert_eq!(
        store.register(&view, CLASS, schema, methods).unwrap_err(),
        Refusal::UndeclaredRead {
            class: CLASS,
            method: MethodId(fixture::GUARDED),
            effect: 0,
            position: 3
        }
    );
}

#[test]
fn an_unrelated_method_is_reported_unchanged_under_every_variant() {
    let (_, before) = lab(Variant::Base);
    for v in [
        Variant::OnMoved,
        Variant::GuardMoved,
        Variant::GuardRespelled,
        Variant::DoReversed,
    ] {
        let (_, after) = lab(v);
        let ds = diff(&before, &after).unwrap();
        assert_eq!(ds.len(), 3, "{v:?}");
        let d = delta(&ds, fixture::UNRELATED);
        assert_eq!(d.status, Status::Unchanged, "{v:?}");
        assert!(d.stale.is_empty() && !d.binding_changed, "{v:?}");
        assert_eq!(
            before.method(CLASS, fixture::UNRELATED),
            after.method(CLASS, fixture::UNRELATED)
        );
        // Exactly one method moved.
        assert_eq!(
            ds.iter().filter(|d| d.status != Status::Unchanged).count(),
            1,
            "{v:?}"
        );
    }
}

#[test]
fn a_reference_from_another_issuer_yields_no_facts() {
    let w = fixture::world(Variant::Base);
    let clone = w.store.clone();
    let foreign = clone.method_ref(CLASS, MethodId(fixture::GUARDED)).unwrap();
    assert_eq!(
        extract(&w.store, &w.view, &foreign, &fixture::READ).unwrap_err(),
        Missing::Admit(Refusal::ForeignStore { class: CLASS })
    );
    // A reference that dropped its guard witness is refused too.
    let mut dropped = w
        .store
        .method_ref(CLASS, MethodId(fixture::GUARDED))
        .unwrap();
    dropped.guard = None;
    assert_eq!(
        extract(&w.store, &w.view, &dropped, &fixture::READ).unwrap_err(),
        Missing::Admit(Refusal::GuardDropped {
            class: CLASS,
            method: MethodId(fixture::GUARDED)
        })
    );
}

#[test]
fn a_removed_dispatch_mapping_makes_its_word_unknown_and_no_other() {
    let (_, snap) = lab(Variant::Base);
    let (src, real) = real_dispatch();
    assert!(real.words_without_arm().is_empty());
    let less = position_of("IntSLess").unwrap();
    let arm = real.arm_of_word(less).unwrap().clone();

    let without: String = src
        .lines()
        .enumerate()
        .filter(|(i, _)| *i + 1 < arm.start || *i + 1 > arm.end)
        .map(|(_, l)| format!("{l}\n"))
        .collect();
    let cut = dispatch::read(&without).unwrap();
    assert_eq!(cut.words_without_arm(), [less]);
    assert_eq!(
        word_impact(&snap, &cut, &without, POLYFILL_LENS, less).unwrap_err(),
        ImplRefusal::NoDispatchArm { word: less }
    );
    // Every other word answers as before.
    for word in (0..17u8).filter(|w| *w != less) {
        assert_eq!(
            word_impact(&snap, &cut, &without, POLYFILL_LENS, word)
                .unwrap()
                .users,
            word_impact(&snap, &real, &src, POLYFILL_LENS, word)
                .unwrap()
                .users
        );
    }
}

#[test]
fn a_dispatch_reading_from_other_source_text_is_refused() {
    let (_, snap) = lab(Variant::Base);
    let (src, real) = real_dispatch();
    let line = real.arm_of_word(position_of("SUM").unwrap()).unwrap().start;
    let later = format!("{src}\n// one line later\n");
    assert!(matches!(
        impl_change(&snap, &real, &later, POLYFILL_LENS, line).unwrap_err(),
        ImplRefusal::StaleWitness { .. }
    ));
    assert!(impl_change(&snap, &real, &src, POLYFILL_LENS, line).is_ok());
}

#[test]
fn a_line_outside_the_word_arms_has_no_behaviour_edge() {
    let (_, snap) = lab(Variant::Base);
    let (src, real) = real_dispatch();
    assert_eq!(
        impl_change(&snap, &real, &src, POLYFILL_LENS, 1).unwrap_err(),
        ImplRefusal::NotInDispatch { line: 1 }
    );
    let load = real.arms.iter().find(|a| a.names == ["LOAD"]).unwrap();
    assert_eq!(
        impl_change(&snap, &real, &src, POLYFILL_LENS, load.start).unwrap_err(),
        ImplRefusal::ArmNotAWord {
            names: vec!["LOAD".to_string()]
        }
    );
    assert_eq!(
        dispatch::read("fn other() {}").unwrap_err(),
        DispatchError::NoCallFn
    );
}

#[test]
fn facts_that_contradict_the_binding_digest_are_refused() {
    let (_, before) = lab(Variant::Base);
    // ON altered by hand, digest left alone.
    let mut forged = before.clone();
    forged.methods[0].on[0].position = 1;
    assert!(
        matches!(diff(&before, &forged).unwrap_err(), DiffRefusal::Inconsistent { method, .. } if method == fixture::GUARDED)
    );
    // Digest altered by hand, nothing else.
    let mut forged = before.clone();
    forged.methods[2].binding ^= 1;
    assert!(
        matches!(diff(&before, &forged).unwrap_err(), DiffRefusal::Inconsistent { method, .. } if method == fixture::UNRELATED)
    );
    // A method whose class basis is absent.
    let mut forged = before.clone();
    forged.classes.clear();
    assert_eq!(
        diff(&before, &forged).unwrap_err(),
        DiffRefusal::ClassMissing { class: CLASS }
    );
}
