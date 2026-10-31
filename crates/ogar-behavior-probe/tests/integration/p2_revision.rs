//! P2: body revision. Positive and negative fixtures, and the envelope
//! change `body_rev` cannot see.

use ogar_behavior_probe::revision::{body_rev, envelope_rev, raw_body_rev, RevError, VocabTag};
use ogar_loco::{Call, FnIndex, FunctionBody, LaneShape, Program};
use ogar_vocab::KausalSpec;

use crate::common::{class, compiled, core};

const CORE: VocabTag = VocabTag {
    name: "ogar-loco/shared-core",
    rev: 1,
};

/// Iterative GCD, copied from `OGAR/crates/ogar-loco/examples/interpret_probe.rs`
/// (`build_gcd`). vars: a=0, b=1, t=2.
fn gcd(a0: u8, b0: u8) -> Program {
    let entry = FunctionBody::from_calls(
        LaneShape::Pairs,
        &[
            Call::with_value(FnIndex::NUMBER, a0),
            Call::with_value(FnIndex::VAR_SET, 0),
            Call::with_value(FnIndex::NUMBER, b0),
            Call::with_value(FnIndex::VAR_SET, 1),
            Call::with_value(FnIndex::VAR_GET, 1),
            Call::with_value(FnIndex::NUMBER, 0),
            Call::new(FnIndex::NEQ),
            Call::with_value(FnIndex::WHILE, 1),
            Call::with_value(FnIndex::VAR_GET, 0),
        ],
    )
    .unwrap();
    let loop_body = FunctionBody::from_calls(
        LaneShape::Pairs,
        &[
            Call::with_value(FnIndex::VAR_GET, 1),
            Call::with_value(FnIndex::VAR_SET, 2),
            Call::with_value(FnIndex::VAR_GET, 0),
            Call::with_value(FnIndex::VAR_GET, 1),
            Call::new(FnIndex::MOD),
            Call::with_value(FnIndex::VAR_SET, 1),
            Call::with_value(FnIndex::VAR_GET, 2),
            Call::with_value(FnIndex::VAR_SET, 0),
        ],
    )
    .unwrap();
    Program {
        functions: vec![entry, loop_body],
    }
}

/// `if x { then } else { else }` with the two branch bodies at indices
/// `then_at` and `else_at` (1 and 2 in either order).
fn branch(then_at: u8, else_at: u8) -> Program {
    let call = |v: u8| {
        FunctionBody::from_calls(LaneShape::Pairs, &[Call::with_value(FnIndex::NUMBER, v)]).unwrap()
    };
    let mut entry_call = Call::new(FnIndex::IF_ELSE);
    entry_call.values = [then_at, else_at, 0];
    let entry = FunctionBody::from_calls(
        LaneShape::Triples,
        &[Call::with_value(FnIndex::VAR_GET, 0), entry_call],
    )
    .unwrap();
    let mut functions = vec![entry, call(0), call(0)];
    functions[usize::from(then_at)] = call(7);
    functions[usize::from(else_at)] = call(9);
    Program { functions }
}

#[test]
fn an_identical_rebuild_has_the_same_revision() {
    let v = core();
    assert_eq!(
        body_rev(&gcd(48, 18), &v, CORE),
        body_rev(&gcd(48, 18), &v, CORE)
    );
    assert_eq!(
        raw_body_rev(&gcd(48, 18), CORE),
        raw_body_rev(&gcd(48, 18), CORE)
    );
}

#[test]
fn a_changed_constant_changes_the_revision() {
    let v = core();
    assert_ne!(
        body_rev(&gcd(48, 18), &v, CORE),
        body_rev(&gcd(48, 19), &v, CORE)
    );
}

#[test]
fn a_renumbered_body_keeps_the_canonical_revision_but_not_the_raw_one() {
    let v = core();
    let a = branch(1, 2);
    let b = branch(2, 1);
    assert_ne!(a, b, "the two programs differ in stored order");
    assert_ne!(raw_body_rev(&a, CORE), raw_body_rev(&b, CORE));
    assert_eq!(
        body_rev(&a, &v, CORE).unwrap(),
        body_rev(&b, &v, CORE).unwrap()
    );
}

#[test]
fn swapping_the_branch_targets_changes_the_canonical_revision() {
    // Silence twin of the renumbering test: same functions, but the THEN and
    // ELSE bodies trade places. That is a different program and must read so.
    let v = core();
    let a = branch(1, 2);
    let mut b = branch(1, 2);
    b.functions.swap(1, 2);
    assert_ne!(
        body_rev(&a, &v, CORE).unwrap(),
        body_rev(&b, &v, CORE).unwrap()
    );
}

#[test]
fn an_unreachable_function_is_not_part_of_the_canonical_revision() {
    let v = core();
    let a = gcd(48, 18);
    let mut b = gcd(48, 18);
    b.functions.push(
        FunctionBody::from_calls(LaneShape::Pairs, &[Call::with_value(FnIndex::NUMBER, 1)])
            .unwrap(),
    );
    assert_ne!(raw_body_rev(&a, CORE), raw_body_rev(&b, CORE));
    assert_eq!(
        body_rev(&a, &v, CORE).unwrap(),
        body_rev(&b, &v, CORE).unwrap()
    );
}

#[test]
fn the_vocabulary_tag_is_part_of_the_revision() {
    let v = core();
    let other = VocabTag {
        name: "ogar-loco/shared-core",
        rev: 2,
    };
    assert_ne!(
        body_rev(&gcd(48, 18), &v, CORE),
        body_rev(&gcd(48, 18), &v, other)
    );
}

#[test]
fn a_dangling_reference_is_refused() {
    let v = core();
    let mut p = gcd(48, 18);
    p.functions.truncate(1);
    assert_eq!(body_rev(&p, &v, CORE), Err(RevError::UnresolvableReference));
}

#[test]
fn a_guard_change_on_the_action_leaves_the_body_revision_unchanged() {
    // The P2 negative case the operator asked for: change WHEN on a real
    // ActionDef while the program bytes stay fixed.
    let v = core();
    let c = compiled();
    let term = class(&c, "account_payment_term");
    let before = term
        .actions
        .iter()
        .find(|a| a.predicate == "_check_lines")
        .expect("pinned constrains method")
        .clone();
    let mut after = before.clone();
    after.kausal = Some(KausalSpec::StateGuard {
        guard_field: "active".into(),
        guard_values: vec!["True".into()],
    });

    let program = gcd(48, 18);
    let rev = body_rev(&program, &v, CORE).unwrap();
    assert_eq!(
        rev,
        body_rev(&program, &v, CORE).unwrap(),
        "body_rev cannot see it"
    );
    assert_ne!(
        envelope_rev(&before),
        envelope_rev(&after),
        "the envelope digest does"
    );

    // And a write-set change, the other half of the envelope.
    let mut wrote = before.clone();
    wrote.writes.push("note".into());
    assert_ne!(envelope_rev(&before), envelope_rev(&wrote));
}
