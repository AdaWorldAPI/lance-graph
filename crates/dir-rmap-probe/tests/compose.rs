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
    let mut valid = page.member;
    valid[0][s >> 6] &= !(1u64 << (s & 63));
    let m = Members::of(&qr, &page, 1).with_valid(&valid);
    let out = compose(&m, &c, &w.witness(), Law::Count).unwrap();
    assert_eq!(out.counts[5], 1);
    assert_eq!(out.bag, 1);
    assert_eq!(out.distinct, 1);
    // The left fan (member & valid) dropped to 1, so the rule yields Distinct.
    assert_eq!(out.left_fan, 1);
    assert_eq!(out.stratum, Stratum::Distinct);
}

/// FAILS IF: the support under a presence plane is taken from the raw
/// membership row instead of the present keys. g1 alone grants role 7, so
/// nulling g1 must drop role 7 from the support, while role 5 (reached
/// through g0 too) stays.
#[test]
fn null_key_drops_the_role_only_it_reaches() {
    let mut f = diamond(vec![(0, 0), (0, 1)]);
    f.grants.push((1, 7));
    let (w, qr, page) = rig(f);
    let ord = Ordinals::of(&w.f, &w.view());
    let s = qr.slot(ord.group[1]).expect("g1 is a security group");
    let c = Carriers::of(&qr);
    let full = compose(&Members::of(&qr, &page, 1), &c, &w.witness(), Law::Count).unwrap();
    assert_eq!(
        full.support[0],
        (1 << 5) | (1 << 7),
        "anti-vacuity: both roles held"
    );
    let mut valid = page.member;
    valid[0][s >> 6] &= !(1u64 << (s & 63));
    let out = compose(
        &Members::of(&qr, &page, 1).with_valid(&valid),
        &c,
        &w.witness(),
        Law::Count,
    )
    .unwrap();
    assert_eq!(out.support[0], 1 << 5);
    assert_eq!(out.counts[7], 0);
    assert_eq!((out.bag, out.distinct), (1, 1));
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

/// FAILS IF: a page resolved before an edit is accepted when it is paired
/// with a register bound after the edit. The page carries the witness it was
/// resolved against, so `Members` cannot borrow the new register's witness
/// for old rows.
#[test]
fn page_from_an_earlier_binding_is_refused() {
    let (mut w, _qr_old, page_old) = rig(diamond(vec![(0, 0), (0, 1)]));
    w.apply(&[Edit::AddDirect(0, 2)]);
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).expect("fits 256 slots");
    let c = Carriers::of(&qr);
    let fresh = qr.page(&b, 0).expect("page 0");
    assert!(
        compose(&Members::of(&qr, &fresh, 1), &c, &w.witness(), Law::Count).is_ok(),
        "anti-vacuity: a page resolved against this register composes"
    );
    assert_eq!(
        compose(
            &Members::of(&qr, &page_old, 1),
            &c,
            &w.witness(),
            Law::Count
        )
        .err(),
        Some(ComposeError::SealMismatch)
    );
}

/// FAILS IF: a user with more present direct groups than r2il has scratch
/// slots makes the support union panic, or the union loses a role. User in
/// 12 security groups, group `g` grants role `g`.
#[test]
fn wide_support_union_composes() {
    const G: u32 = 12;
    let mut f = diamond((0..G).map(|g| (0, g)).collect());
    f.groups = (0..G).map(|g| Guid128([0x20 + g as u8; 16])).collect();
    f.kind = vec![Kind::Role; G as usize];
    f.mail = vec![false; G as usize];
    f.security = vec![Some(true); G as usize];
    f.nested = vec![];
    f.grants = (0..G).map(|g| (g, g as u8)).collect();
    f.roles = (0..G).collect();
    let (w, qr, page) = rig(f);
    let out = compose(
        &Members::of(&qr, &page, 1),
        &Carriers::of(&qr),
        &w.witness(),
        Law::Count,
    )
    .expect("composes");
    assert_eq!(out.left_fan, u64::from(G));
    assert_eq!(out.support[0], (1u64 << G) - 1);
    assert_eq!((out.bag, out.distinct), (u64::from(G), u64::from(G)));
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
        // The two Distinct routes agree: the page's resolved roles (no
        // presence plane) and the product's non-zero entries (a full one).
        let via_product = compose(
            &Members::of(&qr, &page, users).with_valid(&page.member),
            &c,
            &w.witness(),
            Law::Count,
        )
        .unwrap();
        assert!(
            !differ(&out.support, &via_product.support, users * 64),
            "page {p}: Page::roles vs product support"
        );
        assert_eq!(out.counts, via_product.counts, "page {p}: counts");
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
