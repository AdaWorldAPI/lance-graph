//! D-NFR-6 runtime falsifiers. The compile-time ones (F1, F2, F3 for owned
//! grants, F4 outliving, F7 escaping borrow) are `compile_fail` doctests in
//! `src/lib.rs`; this file covers what happens once the borrow ends, the
//! shared-revocation escape, and the semantic refusals.

use dr_math_binding_probe::{
    Axis, AxisKind, Guards, MethodSignature, Operand, Population, Refusal, Scope, Truth, bind_dr64,
};
use dr_validated_view_probe::fixture::{data, method, revisions, setup, validate_presence};
use dr_validated_view_probe::{SharedGrants, Table, ViewRefusal, validate, validate_shared};

/// The answers an independent row loop gives: count, sum, min, max, count(v > t).
fn oracle(rows: &[u64; 64], v: &[i32], t: i32) -> (u64, i64, Option<i32>, Option<i32>, u64) {
    let sel = |r: usize| rows[r / 64] >> (r % 64) & 1 == 1;
    let kept: Vec<i32> = (0..v.len()).filter(|&r| sel(r)).map(|r| v[r]).collect();
    (
        kept.len() as u64,
        kept.iter().map(|&x| i64::from(x)).sum(),
        kept.iter().copied().min(),
        kept.iter().copied().max(),
        kept.iter().filter(|&&x| x > t).count() as u64,
    )
}

const SIZES: [usize; 8] = [1, 2, 63, 64, 65, 1000, 4095, 4096];

/// FAILS IF: a view's five operations disagree with the direct ndarray calls
/// or with the row loop, at any size.
#[test]
fn view_answers_equal_direct_ndarray_and_the_row_loop() {
    for n in SIZES {
        let (t, g, m, guards) = setup(n);
        let view = validate_presence(&t, &g, m, &guards).unwrap();
        let w = n.div_ceil(64);
        let (rows, v) = (view.rows(), view.values());
        let mut scratch = vec![0u64; 2 * w];
        for thr in [-101, -3, 0, 50, 100] {
            let want = oracle(rows, v, thr);
            let got = (
                view.count(),
                view.sum(),
                view.min(),
                view.max(),
                view.count_gt(thr, &mut scratch),
            );
            assert_eq!(got, want, "n={n} t={thr}");
            let mut cmp = vec![0u64; w];
            let mut both = vec![0u64; w];
            ndarray::simd::gt_i32_to_mask(v, thr, &mut cmp);
            ndarray::simd::mask_and(&cmp, &rows[..w], &mut both);
            let direct = (
                ndarray::simd::popcount_batch_u64(&rows[..w]),
                ndarray::simd::masked_sum_i32(v, &rows[..w]),
                ndarray::simd::masked_min_i32(v, &rows[..w]),
                ndarray::simd::masked_max_i32(v, &rows[..w]),
                ndarray::simd::popcount_batch_u64(&both),
            );
            assert_eq!(got, direct, "n={n} t={thr}");
        }
    }
}

/// FAILS IF: the fixture is vacuous: the selection is all or nothing, or the
/// threshold never splits it.
#[test]
fn the_fixture_is_not_vacuous() {
    let (t, g, m, guards) = setup(4096);
    let view = validate_presence(&t, &g, m, &guards).unwrap();
    let c = view.count();
    assert!(c > 0 && c < 4096);
    assert!(view.min() < view.max());
    let mut s = vec![0u64; 128];
    let k = view.count_gt(0, &mut s);
    assert!(k > 0 && k < c);
}

/// FAILS IF: after the borrow ends, a source advance, a refocus or a revoked
/// scope still lets the old guards validate. The silent twin: the same
/// table validates again under freshly taken guards.
#[test]
fn after_the_borrow_ends_every_change_is_refused_at_acquisition() {
    let (mut t, mut g, m, guards) = setup(64);
    // A view is `Copy`; every copy carries the same borrow, which ends here.
    let _ = validate_presence(&t, &g, m, &guards).unwrap();

    t.advance(revisions(2), data(64, 9));
    assert_eq!(
        validate_presence(&t, &g, m, &guards).err(),
        Some(ViewRefusal::Bind(Refusal::SourceChanged))
    );
    let fresh = Guards {
        frame: t.seal(),
        ..guards.clone()
    };
    let v = validate_presence(&t, &g, m, &fresh).unwrap();
    assert_eq!(v.values(), &data(64, 9).values[..]);

    // A refocus with an identical map still moves the focus revision.
    let map = *t.local_to_global();
    t.refocus(map);
    assert_eq!(
        validate_presence(&t, &g, m, &fresh).err(),
        Some(ViewRefusal::Bind(Refusal::FocusChanged))
    );
    let fresh = Guards {
        focus: t.focus(),
        ..fresh
    };
    assert!(validate_presence(&t, &g, m, &fresh).is_ok());

    g.revoke();
    assert_eq!(
        validate_presence(&t, &g, m, &fresh).err(),
        Some(ViewRefusal::ScopeRevoked)
    );
}

/// FAILS IF: re-advancing to the same numbers keeps old frames valid, or a
/// frame from another table with equal numbers is accepted.
#[test]
fn a_resealed_or_foreign_frame_is_refused() {
    let (mut t, g, m, guards) = setup(64);
    t.advance(revisions(1), data(64, 0)); // same numbers, new issue
    assert_eq!(
        validate_presence(&t, &g, m, &guards).err(),
        Some(ViewRefusal::Bind(Refusal::FrameResealed))
    );
    let other = Table::new(revisions(1), data(64, 0), 7, [0; 64]);
    let foreign = Guards {
        frame: other.seal(),
        ..guards
    };
    assert_eq!(
        validate_presence(&t, &g, m, &foreign).err(),
        Some(ViewRefusal::Bind(Refusal::ForeignFrame))
    );
}

/// FAILS IF: scope is treated as a number to compare instead of a grant.
/// Guards for scope 3 under grants for scope 4 are refused by the existing
/// check; a revoked grant is refused before it.
#[test]
fn scope_comes_from_the_grant_not_from_the_caller() {
    let (t, _, m, guards) = setup(64);
    let other = dr_validated_view_probe::Grants::new(Scope { id: 4 });
    assert_eq!(
        validate(&t, &other, m, m, &guards).err(),
        Some(ViewRefusal::Bind(Refusal::ScopeChanged))
    );
}

/// F5. FAILS IF: a revocation by another owner, during a live view, goes
/// unnoticed. The borrow cannot stop it, so the shared view checks the grant
/// epoch on each operation. Before revocation it answers.
#[test]
fn shared_revocation_during_a_view_is_refused_per_operation() {
    let (t, _, m, guards) = setup(256);
    let g = SharedGrants::new(Scope { id: 3 });
    let sv = validate_shared(&t, &g, m, m, &guards).unwrap();
    let want = oracle(&data(256, 0).rows, &data(256, 0).values, 0);
    assert_eq!(sv.count(), Ok(want.0));
    assert_eq!(sv.sum(), Ok(want.1));
    g.revoke(); // through &self, while `sv` is alive: it compiles
    assert_eq!(sv.count(), Err(ViewRefusal::ScopeRevoked));
    assert_eq!(sv.max(), Err(ViewRefusal::ScopeRevoked));
    let mut s = vec![0u64; 8];
    assert_eq!(sv.count_gt(0, &mut s), Err(ViewRefusal::ScopeRevoked));
    assert_eq!(
        validate_shared(&t, &g, m, m, &guards).err(),
        Some(ViewRefusal::ScopeRevoked)
    );
}

/// F6. FAILS IF: a view is issued for another method, other axes, other
/// semantics, a three-valued method without validity, or a Bag population.
/// Each existing refusal of `bind_dr64` comes through unchanged.
#[test]
fn semantic_mismatches_keep_their_typed_refusals() {
    let (t, g, m, guards) = setup(64);
    let refuse = |declared: MethodSignature, requested: MethodSignature| {
        validate(&t, &g, declared, requested, &guards).err()
    };
    assert_eq!(
        refuse(m, method(2)),
        Some(ViewRefusal::Bind(Refusal::MethodChanged))
    );
    let other_axis = MethodSignature {
        column: Axis {
            class: 9,
            kind: AxisKind::Community,
        },
        ..m
    };
    assert_eq!(
        refuse(m, other_axis),
        Some(ViewRefusal::Bind(Refusal::AxisChanged))
    );
    let three = MethodSignature {
        truth: Truth::ThreeValued,
        ..m
    };
    assert_eq!(
        refuse(m, three),
        Some(ViewRefusal::Bind(Refusal::SemanticsChanged))
    );
    assert_eq!(
        refuse(three, three),
        Some(ViewRefusal::Bind(Refusal::ThreeValuedNeedsValidity))
    );
    let bag = MethodSignature {
        population: Population::Bag,
        ..m
    };
    assert_eq!(
        refuse(bag, bag),
        Some(ViewRefusal::Bind(Refusal::BagNeedsMultiplicity))
    );
}

/// Records the audit finding that motivates `Table`: the existing validator
/// checks the frame, not the bytes. Rows that no frame owner issued are
/// accepted under a valid frame. FAILS IF that changes, so the claim in the
/// crate docs is re-examined.
#[test]
fn bind_dr64_checks_the_frame_not_the_data() {
    let (t, _, m, guards) = setup(64);
    let unrelated = [u64::MAX; 64];
    let bound = bind_dr64(
        t.store(),
        m,
        m,
        &guards,
        (t.focus(), guards.scope),
        Operand {
            rows: &unrelated,
            validity: None,
        },
    )
    .unwrap();
    assert!(std::ptr::eq(bound.rows, &unrelated));
}
