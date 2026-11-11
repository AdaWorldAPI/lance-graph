//! Gate A: ON, WHEN and DO are read as separate registered facts.

use cg_beh_probe::facts::{extract, ArgFact, DepKind, GuardConvention, Missing, Read, Snapshot};
use cg_beh_probe::fixture::{self, FoundryVariant, Variant};
use foundry_parity_lab::method;
use lance_graph_contract::set_algebra::{Cmp, Logic, SetExpr};
use lance_graph_dt64::{Guard, MethodId, Refusal, SchemaRev};
use lance_graph_r2il::polyfill::{position_of, POLYFILL, POLYFILL_LENS};

use crate::common::{foundry, lab, REV};

#[test]
fn the_lab_method_is_read_part_by_part() {
    let (w, s) = lab(Variant::Base);
    let m = s.method(fixture::CLASS, fixture::GUARDED).unwrap();

    // Identity: the digest is the one the store's own reference carries.
    let r = w
        .store
        .method_ref(fixture::CLASS, MethodId(fixture::GUARDED))
        .unwrap();
    assert_eq!(m.binding, r.stamp.binding());
    assert_eq!(m.sig_rev, 1);

    // ON, through the pinned basis.
    assert_eq!(m.on.len(), 1);
    assert_eq!(
        (m.on[0].position, m.on[0].predicate.as_str()),
        (0, "lab:beh/a")
    );
    let reads: Vec<(u8, &str)> = m
        .reads
        .iter()
        .map(|f| (f.position, f.predicate.as_str()))
        .collect();
    assert_eq!(reads, [(0, "lab:beh/a"), (1, "lab:beh/b")]);

    // WHEN.
    let when = m.when.as_ref().unwrap();
    assert_eq!(when.logic, "TwoValued");
    assert_eq!(when.deps.len(), 2);
    for (dep, (number, predicate)) in when.deps.iter().zip([(0, "lab:beh/a"), (1, "lab:beh/b")]) {
        assert_eq!((dep.kind, dep.number), (DepKind::ResI32, number));
        assert_eq!(dep.consumer.as_ref().unwrap().predicate, predicate);
    }

    // DO, in order.
    assert_eq!(m.steps.len(), 2);
    let (less, sum) = (&m.steps[0], &m.steps[1]);
    assert_eq!(
        (less.ordinal, less.lens, less.word),
        (0, POLYFILL_LENS, position_of("IntSLess").unwrap())
    );
    assert_eq!(less.word_predicate, "lg:polyfill:f000/IntSLess");
    assert_eq!((less.arity, less.target.position), (2, 3));
    assert!(matches!(&less.args[..], [ArgFact::Field(f), ArgFact::Const(5)] if f.position == 0));
    assert_eq!(
        (sum.ordinal, sum.word, sum.target.position),
        (1, position_of("SUM").unwrap(), 4)
    );
    assert!(
        matches!(&sum.args[..], [ArgFact::Field(a), ArgFact::Field(b)] if a.position == 3 && b.position == 1)
    );
    assert_eq!(POLYFILL.len(), 17);
}

#[test]
fn without_a_declared_convention_every_guard_dependency_is_unknown() {
    let w = fixture::world(Variant::Base);
    let r = w
        .store
        .method_ref(fixture::CLASS, MethodId(fixture::GUARDED))
        .unwrap();
    let opaque = Read {
        convention: GuardConvention::Opaque,
        ..fixture::READ
    };
    let m = extract(&w.store, &w.view, &r, &opaque).unwrap();
    let deps = &m.when.as_ref().unwrap().deps;
    assert_eq!(deps.len(), 2);
    assert!(deps.iter().all(|d| d.consumer.is_none()));
}

#[test]
fn a_guard_dependency_outside_the_registered_reads_stays_unknown() {
    // DT64 registers a guard that names lane 2 for a method that reads only
    // 0 and 1: `validate` does not check guard positions. The declared
    // convention is therefore not accepted for that dependency.
    let mut methods = fixture::methods(Variant::Base);
    methods[0].when = Some(Guard::new(
        SetExpr::And(vec![
            SetExpr::i32(0, Cmp::Lt, 5),
            SetExpr::i32(2, Cmp::Ge, 0),
        ]),
        Logic::TwoValued,
    ));
    let w = fixture::world_with(methods, false);
    let r = w
        .store
        .method_ref(fixture::CLASS, MethodId(fixture::GUARDED))
        .unwrap();
    let m = extract(&w.store, &w.view, &r, &fixture::READ).unwrap();
    let deps = &m.when.as_ref().unwrap().deps;
    assert_eq!(deps.len(), 2);
    assert!(deps[0].consumer.is_some());
    assert_eq!(deps[1].number, 2);
    assert!(deps[1].consumer.is_none());
}

#[test]
fn can_cook_has_on_and_when_and_no_do() {
    let (_, s) = foundry(FoundryVariant::Base);
    assert_eq!(s.methods.len(), 16);
    let id = method::method_id(0, 1).0;
    let m = s.method(fixture::FOUNDRY_CLASS, id).unwrap();

    assert!(m.steps.is_empty());
    assert!(s.methods.iter().all(|m| m.steps.is_empty()));
    assert!(s.to_tsv().contains(&format!("DO_NONE\t0x51160001\t{id}\n")));
    assert!(!s.to_tsv().contains("\nDO\t"));

    // ON is registered equal to reads, and every guard dependency is one of
    // the reads.
    assert!(!m.on.is_empty());
    assert_eq!(m.on, m.reads);
    let when = m.when.as_ref().unwrap();
    assert_eq!(when.logic, "TwoValued");
    assert!(when.deps.len() >= 3);
    for d in &when.deps {
        let f = d.consumer.as_ref().unwrap();
        assert!(m.reads.contains(f));
        assert!(f.predicate.starts_with("ops:"));
    }
    assert_eq!(s.class(fixture::FOUNDRY_CLASS).unwrap().basis.len(), 211);
}

#[test]
fn a_snapshot_without_a_source_revision_is_refused() {
    assert_eq!(Snapshot::new("", "x").unwrap_err(), Missing::SourceRevision);
    assert_eq!(
        Snapshot::new("  UNKNOWN ", "x").unwrap_err(),
        Missing::SourceRevision
    );
    assert!(Snapshot::new(REV, "x").is_ok());
}

#[test]
fn do_steps_are_not_read_without_the_lens_they_were_bound_at() {
    let w = fixture::world(Variant::Base);
    let r = w
        .store
        .method_ref(fixture::CLASS, MethodId(fixture::GUARDED))
        .unwrap();
    let none = Read {
        lens_rev: None,
        ..fixture::READ
    };
    assert_eq!(
        extract(&w.store, &w.view, &r, &none).unwrap_err(),
        Missing::LensRevision {
            lens: POLYFILL_LENS
        }
    );
    let other = Read {
        lens_rev: Some(SchemaRev(2)),
        ..fixture::READ
    };
    assert!(matches!(
        extract(&w.store, &w.view, &r, &other).unwrap_err(),
        Missing::Lens(Refusal::SchemaRevisionMismatch { .. })
    ));
}
