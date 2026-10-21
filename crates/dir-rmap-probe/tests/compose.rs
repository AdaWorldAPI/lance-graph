//! D-RFT-1: the typed composition (user -> group membership composed with
//! role carriers over one slot coordinate) against an independent oracle.
//!
//! The oracle works from `Prepared` ordinals (a depth-first walk over the
//! nesting CSR restricted to security groups), not from `Qr` (Warshall in
//! the tile). Its output is a `Vec<u16>` lane in the same `i * 64 + r`
//! coordinates, compared lane to lane; supports are compared with
//! `compose::differ`. No tuple is built.

use dir_rmap_probe::compose::{
    Carriers, ComposeError, Law, Members, Stratum, add_bag, compose, differ,
};
use dir_rmap_probe::fixture::{Fixture, Kind, Spec};
use dir_rmap_probe::prep::{Ordinals, Prepared, bit};
use dir_rmap_probe::qr::{Page, Qr};
use dir_rmap_probe::{Edit, World};
use ogar_dir_core::{DirectoryScope, Guid128};

/// One user, three security groups g0, g1, g2; user directly in g0 and g1
/// (or only `direct`), g0 and g1 both nested in g2, g2 grants role 5.
fn diamond(direct: Vec<(u32, u32)>) -> Fixture {
    let mut spec = Spec::small();
    spec.users = 1;
    Fixture {
        spec,
        users: vec![Guid128([0x01; 16])],
        groups: vec![
            Guid128([0x10; 16]),
            Guid128([0x11; 16]),
            Guid128([0x12; 16]),
        ],
        kind: vec![Kind::Role; 3],
        mail: vec![false; 3],
        security: vec![Some(true); 3],
        direct,
        nested: vec![(0, 2), (1, 2)],
        grants: vec![(2, 5)],
        roles: vec![0, 1, 2],
        locations: vec![],
        teams: vec![],
        attr: vec![0],
        scope: DirectoryScope(Guid128([0x5C; 16])),
    }
}

fn rig(f: Fixture) -> (World, Qr, Page) {
    let w = World::new(f);
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).expect("fits 256 slots");
    let page = qr.page(&b, 0).expect("page 0");
    (w, qr, page)
}

/// FAILS IF: the diamond (two direct groups reaching one role) is counted
/// once or twice for Distinct, or the Bag count loses a path, or the
/// stratum is not Bag when both sides fan out.
#[test]
fn diamond_counts_two_paths_once_distinct() {
    let (w, qr, page) = rig(diamond(vec![(0, 0), (0, 1)]));
    let m = Members::of(&qr, &page, 1);
    let c = Carriers::of(&qr);
    let out = compose(&m, &c, &w.witness(), Law::Count).unwrap();
    assert_eq!(out.counts[5], 2);
    assert_eq!(out.support[0] >> 5 & 1, 1);
    assert_eq!(out.bag, 2);
    assert_eq!(out.distinct, 1);
    assert_eq!(out.stratum, Stratum::Bag);
    assert_eq!(out.answer(Law::Count), Ok(2));
    assert_eq!(out.answer(Law::Exists), Ok(1));
}

/// FAILS IF: parity (Xor) is accepted as a law, where the diamond would
/// read 0.
#[test]
fn xor_is_refused() {
    let (w, qr, page) = rig(diamond(vec![(0, 0), (0, 1)]));
    let m = Members::of(&qr, &page, 1);
    let c = Carriers::of(&qr);
    assert_eq!(
        compose(&m, &c, &w.witness(), Law::Xor).err(),
        Some(ComposeError::XorLaw)
    );
    let out = compose(&m, &c, &w.witness(), Law::Count).unwrap();
    assert_eq!(out.answer(Law::Xor), Err(ComposeError::XorLaw));
}

/// FAILS IF: a NULL join key (payload set, presence clear) still matches.
#[test]
fn null_key_matches_nothing() {
    let (w, qr, page) = rig(diamond(vec![(0, 0), (0, 1)]));
    let ord = Ordinals::of(&w.f, &w.view());
    let s = qr.slot(ord.group[1]).expect("g1 is a security group");
    let c = Carriers::of(&qr);
    let full = Members::of(&qr, &page, 1);
    assert_eq!(
        compose(&full, &c, &w.witness(), Law::Count)
            .unwrap()
            .stratum,
        Stratum::Bag,
        "anti-vacuity: with both keys present the left side fans out"
    );
    let mut m = Members::of(&qr, &page, 1);
    m.valid[0][s >> 6] &= !(1u64 << (s & 63));
    let out = compose(&m, &c, &w.witness(), Law::Count).unwrap();
    assert_eq!(out.counts[5], 1);
    assert_eq!(out.bag, 1);
    assert_eq!(out.distinct, 1);
    // The left fan (member & valid) dropped to 1, so the rule yields Distinct.
    assert_eq!(out.left_fan, 1);
    assert_eq!(out.stratum, Stratum::Distinct);
}

/// FAILS IF: operands bound to different registers are composed.
#[test]
fn mismatched_seal_is_refused() {
    let (wa, qa, pa) = rig(diamond(vec![(0, 0), (0, 1)]));
    let (_wb, qb, _pb) = rig(diamond(vec![(0, 0), (0, 1)]));
    let m = Members::of(&qa, &pa, 1);
    let c = Carriers::of(&qb);
    assert_eq!(
        compose(&m, &c, &wa.witness(), Law::Count).err(),
        Some(ComposeError::SealMismatch)
    );
}

/// FAILS IF: a composition bound before an edit answers for the edited
/// version.
#[test]
fn stale_binding_is_refused() {
    let (mut w, qr, page) = rig(diamond(vec![(0, 0), (0, 1)]));
    let m = Members::of(&qr, &page, 1);
    let c = Carriers::of(&qr);
    assert!(compose(&m, &c, &w.witness(), Law::Count).is_ok());
    w.apply(&[Edit::AddDirect(0, 2)]);
    assert_eq!(
        compose(&m, &c, &w.witness(), Law::Count).err(),
        Some(ComposeError::Stale)
    );
}

/// FAILS IF: the bag sum wraps instead of refusing, or refuses one short of
/// the boundary.
#[test]
fn overflow_is_refused() {
    assert_eq!(add_bag(u64::MAX, 1).err(), Some(ComposeError::Overflow));
    assert_eq!(add_bag(u64::MAX - 1, 1), Ok(u64::MAX));
}

/// FAILS IF: a functional left side (one direct group) is reported as the
/// Bag stratum, or a coordinate exceeds one path.
#[test]
fn functional_side_is_distinct_stratum() {
    let (w, qr, page) = rig(diamond(vec![(0, 0)]));
    let m = Members::of(&qr, &page, 1);
    let c = Carriers::of(&qr);
    let out = compose(&m, &c, &w.witness(), Law::Count).unwrap();
    assert_eq!(out.left_fan, 1);
    assert_eq!(out.stratum, Stratum::Distinct);
    assert!(out.counts.iter().all(|&n| n <= 1));
    assert_eq!(out.counts[5], 1, "anti-vacuity: the role is reached");
}

/// Roles reachable upward from security group `d` through security groups
/// only, by an explicit-stack walk over `b.up`.
fn reach_roles(b: &Prepared, d: usize) -> u64 {
    let mut seen = vec![false; b.groups];
    let mut stack = vec![d];
    seen[d] = true;
    let mut roles = 0u64;
    while let Some(h) = stack.pop() {
        roles |= b.grants[h];
        for &p in b.up.row(h) {
            let p = p as usize;
            if bit(&b.security, p) && !seen[p] {
                seen[p] = true;
                stack.push(p);
            }
        }
    }
    roles
}

/// The oracle lane for one page, in `i * 64 + r` coordinates.
fn oracle_lane(b: &Prepared, first: usize, users: usize) -> Vec<u16> {
    let mut lane = vec![0u16; users * 64];
    for i in 0..users {
        for &d in b.user_groups.row(first + i) {
            let d = d as usize;
            if !bit(&b.security, d) {
                continue;
            }
            let roles = reach_roles(b, d);
            for r in 0..64 {
                lane[i * 64 + r] += u16::from(roles >> r & 1 == 1);
            }
        }
    }
    lane
}

/// FAILS IF: the composed counts or support differ from an independent
/// computation over ordinals, or the fixture never reaches a multi-path
/// coordinate (a vacuous run).
#[test]
fn random_fixture_matches_independent_oracle() {
    let w = World::new(Fixture::generate(Spec::small()));
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).expect("fits 256 slots");
    let c = Carriers::of(&qr);
    let (mut ones, mut multi, mut bag_total) = (0usize, 0usize, 0u64);
    for p in 0..qr.pages() {
        let users = qr.users.saturating_sub(p * 256).min(256);
        let page = qr.page(&b, p).unwrap();
        let m = Members::of(&qr, &page, users);
        let out = compose(&m, &c, &w.witness(), Law::Count).unwrap();
        let want = oracle_lane(&b, p * 256, users);
        assert_eq!(out.counts, want, "page {p}: counts lane");
        let mut want_support = vec![0u64; users];
        for (i, word) in want_support.iter_mut().enumerate() {
            for r in 0..64 {
                *word |= u64::from(want[i * 64 + r] > 0) << r;
            }
        }
        assert!(
            !differ(&out.support, &want_support, users * 64),
            "page {p}: support plane"
        );
        let want_bag = want
            .iter()
            .try_fold(0u64, |a, &n| add_bag(a, u64::from(n)))
            .expect("no overflow");
        assert_eq!(out.bag, want_bag, "page {p}: bag");
        ones += want.iter().filter(|&&n| n == 1).count();
        multi += want.iter().filter(|&&n| n >= 2).count();
        bag_total += want_bag;
    }
    assert!(
        ones > 0 && multi > 0,
        "vacuous: coordinates with count 1 = {ones}, count >= 2 = {multi}, bag = {bag_total}"
    );
}
