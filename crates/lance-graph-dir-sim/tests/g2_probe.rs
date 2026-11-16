//! G2 probe P1-P6: a 512-user x 512-group directory read through dir-sim's
//! virtual projections, against a scalar oracle that shares no code with the
//! crate.
//!
//! The reconcile entry (`2026-11-16-g2-dr-virtual-projection-reconcile.md`)
//! names only P6 (the no-explosion falsifier); the other five are read from
//! its matrix and the G2 gate text:
//!
//! * P1 the sparse fixture reads exactly as the oracle (pairs, counts, group
//!   properties).
//! * P2 DISTINCT: a user that reaches a group by several chains is returned
//!   once; membership input multiplicities collapse (`[1, 1, 3]` shape).
//! * P3 BAG: path multiplicity as a count. BLOCKED, see the ignored test.
//! * P4 version refusal: an unknown version or a reversed range is refused,
//!   never answered from some other version.
//! * P5 authority / scope refusal: a foreign directory scope and a plan for a
//!   version that is not desired are refused.
//! * P6 no explosion: results are bounded by the sparse membership, deltas by
//!   the change, never by users x groups.
//!
//! Fixture (all derived below, nothing hardcoded from a run):
//!
//! * users `0..512`, groups `0..512`;
//! * user `u` is a direct member of group `g` iff `(u * 7 + g * 13) % 61 == 0`;
//! * group `g >= 8` is nested in groups `g % 8` and `(g / 8) % 8` (one edge if
//!   they coincide), so every parent has a smaller index (acyclic) and a user
//!   directly in two children of one root reaches that root by two chains;
//! * user `u` sits at location `[u % 4, (u / 4) % 8]`; group `g` is a
//!   security group iff `g % 3 == 0`; no group has an address.
//!
//! The directory API has no "at most 64 properties" bound: a node carries a
//! UPN, a primary SMTP, an enabled flag and a location, a group adds the
//! security flag. The fixture uses those and no more.
//!
//! No allocation or time counter is used. The test binary runs its tests on
//! several threads, so a process-global counting allocator (as in
//! `tests/alloc.rs`, which is its own single-test binary) would be noise; the
//! bounds below are on result sizes and delta lengths, which the API exposes.

use std::sync::OnceLock;

use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Dn128, Guid128};
use ogar_dir_sim::*;

const U: usize = 512;
const G: usize = 512;
const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const OTHER: DirectoryScope = DirectoryScope(Guid128([0x6D; 16]));
const GRANT: RuleId = RuleId {
    name: "G2Grant",
    version: 1,
};
/// A pair that is not a direct member: `(0 * 7 + 100 * 13) % 61 == 19`.
/// Group 100 has no children, so a user reaches it only by being in it.
const EXTRA: (usize, usize) = (0, 100);

fn id(kind: u8, i: usize) -> Guid128 {
    let mut b = [0u8; 16];
    b[0] = kind;
    b[1..5].copy_from_slice(&(i as u32).to_be_bytes());
    Guid128(b)
}
/// Ordinals follow the index: ids sort by `(kind, big-endian index)`.
fn user(i: usize) -> Guid128 {
    id(1, i)
}
fn group(i: usize) -> Guid128 {
    id(2, i)
}

fn is_direct(u: usize, g: usize) -> bool {
    (u * 7 + g * 13).is_multiple_of(61)
}
fn parents(g: usize) -> Vec<usize> {
    if g < 8 {
        return Vec::new();
    }
    let mut p = vec![g % 8, (g / 8) % 8];
    p.sort_unstable();
    p.dedup();
    p
}
fn dn_of(u: usize) -> [u8; 2] {
    [(u % 4) as u8, ((u / 4) % 8) as u8]
}
fn is_security(g: usize) -> bool {
    g.is_multiple_of(3)
}

fn observation() -> Observation {
    let mut nodes = Vec::with_capacity(U + G);
    for u in 0..U {
        let mail = format!("u{u}@g2.test");
        let mut n = ObservedNode::user(&mail, &mail);
        n.dn = Some(Dn128::new(&dn_of(u)).unwrap());
        nodes.push((user(u), n));
    }
    for g in 0..G {
        let mut n = ObservedNode::group();
        n.security = Some(is_security(g));
        nodes.push((group(g), n));
    }
    let mut members = Vec::new();
    for u in 0..U {
        for g in 0..G {
            if is_direct(u, g) {
                members.push((user(u), group(g)));
            }
        }
    }
    // A group is "a member" of its parent: (child, parent), kept by identity.
    for c in 0..G {
        for p in parents(c) {
            members.push((group(c), group(p)));
        }
    }
    Observation {
        scope: SCOPE,
        nodes,
        members,
    }
}

fn with_member(mut o: Observation, u: usize, g: usize) -> Observation {
    o.members.push((user(u), group(g)));
    o
}

/// The same directory with direct pairs listed again: pairs `0..10` three
/// times, pairs `10..100` twice, the rest once (multiplicities 1, 2, 3).
fn duplicated_observation() -> Observation {
    let mut o = observation();
    let base = o.members.clone();
    o.members.extend(base[..100].iter().copied());
    o.members.extend(base[..10].iter().copied());
    o
}

/// The scalar oracle: plain loops over the formulas above.
struct Oracle {
    /// `bag[u][a]`: how many chains put user `u` in group `a` (a direct
    /// membership is a chain of one group; each parent edge extends it).
    bag: Vec<Vec<u64>>,
    /// Direct `(user, group)` pairs.
    direct_rows: usize,
    /// `(user, group)` pairs with at least one chain.
    distinct: usize,
    /// Sum of chain counts.
    bag_total: u64,
}

/// Chains from group `from` up to group `to` (zero-length chain included).
fn paths(from: usize, to: usize) -> u64 {
    let mut n = u64::from(from == to);
    for p in parents(from) {
        n += paths(p, to);
    }
    n
}

fn oracle() -> &'static Oracle {
    static O: OnceLock<Oracle> = OnceLock::new();
    O.get_or_init(|| {
        let mut bag = vec![vec![0u64; G]; U];
        let mut direct_rows = 0;
        for (u, row) in bag.iter_mut().enumerate() {
            for d in 0..G {
                if is_direct(u, d) {
                    direct_rows += 1;
                    for (a, n) in row.iter_mut().enumerate() {
                        *n += paths(d, a);
                    }
                }
            }
        }
        let distinct = bag.iter().flatten().filter(|&&n| n > 0).count();
        let bag_total: u64 = bag.iter().flatten().sum();
        Oracle {
            bag,
            direct_rows,
            distinct,
            bag_total,
        }
    })
}

/// A pair reached by more than one chain.
fn multi_path_pair(o: &Oracle) -> (usize, usize) {
    (0..U)
        .flat_map(|u| (0..G).map(move |a| (u, a)))
        .find(|&(u, a)| o.bag[u][a] > 1)
        .expect("the fixture has a pair reached by several chains")
}

// ---- P1 ---------------------------------------------------------------------

/// P1: every pair, every group's count vector and the group properties equal
/// the oracle.
///
/// Fails if: `Snapshot::build` mis-resolves ordinals (a pair flips), a nested
/// `(group, group)` edge stops being kept by identity, `member_counts` counts
/// the wrong lane, or `groups_where` reads the wrong plane. The empty
/// `MailEnabled` result and the `Not` complement are the can-stay-silent half:
/// a `groups_where` that returned the security groups for every filter fails
/// them.
#[test]
fn p1_the_sparse_fixture_reads_as_the_scalar_oracle() {
    let o = oracle();
    let mut st = VersionStore::new();
    let v = st.observe("g2", 1, observation()).unwrap();
    let view = st.view(v).unwrap();
    assert_eq!((view.users_len(), view.groups_len()), (U, G));
    assert_eq!(st.snapshot(v).unwrap().membership_rows(), o.direct_rows);

    for u in 0..U {
        for g in 0..G {
            assert_eq!(
                view.is_member(&user(u), &group(g)),
                is_direct(u, g),
                "direct ({u}, {g})"
            );
            // A group is a member of its parents, and of nothing else.
            assert_eq!(
                view.is_member(&group(u), &group(g)),
                parents(u).contains(&g),
                "nested ({u}, {g})"
            );
        }
    }
    for g in 0..G {
        let want: Vec<i64> = (0..U).map(|u| i64::from(is_direct(u, g))).collect();
        assert_eq!(member_counts(&view, &group(g)), want, "group {g}");
    }

    let security = GroupWhere::Is(GroupProperty::SecurityEnabled);
    let secure: Vec<usize> = (0..G).filter(|&g| is_security(g)).collect();
    let rest: Vec<usize> = (0..G).filter(|&g| !is_security(g)).collect();
    assert_eq!(groups_where(&view, &security).rows(), secure);
    assert_eq!(
        groups_where(&view, &GroupWhere::Not(Box::new(security))).rows(),
        rest
    );
    assert_eq!(
        groups_where(&view, &GroupWhere::Is(GroupProperty::MailEnabled)).rows(),
        Vec::<usize>::new(),
        "no group has an address"
    );
}

// ---- P2 ---------------------------------------------------------------------

/// P2: the reach of a user, and the members of a group, are DISTINCT sets.
/// A pair reached by several chains is returned once, and the totals equal
/// the oracle's distinct count, not its chain count.
///
/// Fails if: the walk emitted one entry per chain (the lists would then total
/// the oracle's bag count and the multi-path pair would appear twice), or the
/// walk dropped a nested edge (a list would be short). The fixture guard
/// `bag_total != distinct` is what keeps this from passing on a fixture with
/// no multi-path pair. The single-chain pairs are the can-stay-silent half:
/// the same equality holds for them.
#[test]
fn p2_a_pair_reached_by_several_chains_is_returned_once() {
    let o = oracle();
    assert_ne!(
        o.bag_total, o.distinct as u64,
        "fixture must carry multi-path pairs"
    );
    let mut st = VersionStore::new();
    let v = st.observe("g2", 1, observation()).unwrap();
    let view = st.view(v).unwrap();

    let (mut by_user, mut by_group) = (0usize, 0usize);
    for u in 0..U {
        let got = view.groups_transitive(&user(u));
        let want: Vec<Guid128> = (0..G).filter(|&a| o.bag[u][a] > 0).map(group).collect();
        assert_eq!(got, want, "groups of user {u}");
        by_user += got.len();
    }
    for a in 0..G {
        let got = view.members_transitive(&group(a));
        let want: Vec<Guid128> = (0..U).filter(|&u| o.bag[u][a] > 0).map(user).collect();
        assert_eq!(got, want, "members of group {a}");
        by_group += got.len();
    }
    assert_eq!(by_user, o.distinct);
    assert_eq!(by_group, o.distinct);

    // The named multi-path pair: reached more than once, returned once, and
    // witnessed by one chain whose every link is an edge of the directory.
    let (u, a) = multi_path_pair(o);
    assert!(o.bag[u][a] > 1);
    let reach = view.groups_transitive(&user(u));
    assert_eq!(reach.iter().filter(|g| **g == group(a)).count(), 1);
    let path = view
        .membership_path(&user(u), &group(a))
        .expect("a reached group has a witness");
    assert_eq!(path.user(), &user(u));
    assert_eq!(path.group(), &group(a));
    let chain = path.groups();
    assert!(view.is_member(&user(u), &chain[0]));
    for w in chain.windows(2) {
        assert!(view.is_member(&w[0], &w[1]), "{w:?} is an edge");
    }
}

/// P2: a membership listed several times in the input is one membership.
/// Pairs `0..10` three times, `10..100` twice: the same rows, the same counts
/// and an empty diff against the clean observation.
///
/// Fails if: `Snapshot::build` stops deduplicating its rows (the row count
/// grows by 110 and a `member_counts` entry reaches 3). The clean observation
/// is the can-stay-silent half: its row count equals the oracle's as well, so
/// the assertion does not hold only because duplicates were removed.
#[test]
fn p2_duplicate_input_memberships_collapse_to_a_set() {
    let o = oracle();
    let dup_obs = duplicated_observation();
    assert_eq!(dup_obs.members.len(), observation().members.len() + 110);
    let mut st = VersionStore::new();
    let clean = st.observe("g2", 1, observation()).unwrap();
    let dup = st.observe("g2-dup", 2, dup_obs).unwrap();
    assert_eq!(st.snapshot(clean).unwrap().membership_rows(), o.direct_rows);
    assert_eq!(st.snapshot(dup).unwrap().membership_rows(), o.direct_rows);
    assert_eq!(st.diff(clean, dup).unwrap(), Vec::new());

    let view = st.view(dup).unwrap();
    for g in 0..G {
        let counts = member_counts(&view, &group(g));
        let direct = (0..U).filter(|&u| is_direct(u, g)).count() as i64;
        assert!(counts.iter().all(|&c| c <= 1), "group {g}: {counts:?}");
        assert_eq!(counts.iter().sum::<i64>(), direct, "group {g}");
    }
}

// ---- P3 ---------------------------------------------------------------------

/// P3: the BAG reading, the number of chains per `(user, group)` pair.
///
/// BLOCKED. dir-sim holds membership as a set (`snapshot.rs` sorts and
/// deduplicates), `membership_path` returns one witness, and `member_counts`
/// counts direct membership rows (0 or 1 per user); nested `(group, group)`
/// edges are not rows. Per the reconcile entry (section A, "Membership
/// semantics") a bag witness comes from the selection, not from dir-sim, and
/// the multiplicity primitive (`Cube::bag_count`, `lance-graph-xlog`) is not a
/// dependency of this crate.
///
/// Intended falsifier once a read exists: it must equal
/// `oracle().bag[u][a]` for every pair and exceed the distinct reading at the
/// multi-path pair.
#[test]
#[ignore = "BLOCKED: dir-sim has no read of path multiplicity over membership chains (membership is a set, membership_path is one witness, member_counts counts direct rows only)"]
fn p3_bag_path_multiplicity_is_readable() {
    let o = oracle();
    assert_ne!(
        o.bag_total, o.distinct as u64,
        "fixture must carry multi-path pairs"
    );
    unimplemented!("needs a bag read over membership chains");
}

// ---- P4 ---------------------------------------------------------------------

/// P4: a version the store does not hold is refused; a version it holds is
/// answered as of that version.
///
/// Fails if: `view` fell back to the newest version for an unknown id (the
/// ghost assertions), or answered every read from the newest version (the
/// `is_member` pair differs between `v1` and `v2`, so that mutation fails
/// it), or a refusal changed the store (the image comparison).
#[test]
fn p4_an_unknown_version_is_refused_not_answered() {
    let mut st = VersionStore::new();
    let v1 = st.observe("g2", 1, observation()).unwrap();
    let v2 = st
        .reobserve("g2", 2, with_member(observation(), EXTRA.0, EXTRA.1))
        .unwrap();
    let ghost = VersionId(999);

    // Answered, and answered as of the version asked for.
    {
        let (a, b) = (st.view(v1).unwrap(), st.view(v2).unwrap());
        assert!(!a.is_member(&user(EXTRA.0), &group(EXTRA.1)));
        assert!(b.is_member(&user(EXTRA.0), &group(EXTRA.1)));
        assert!(!a
            .groups_transitive(&user(EXTRA.0))
            .contains(&group(EXTRA.1)));
        assert!(b
            .groups_transitive(&user(EXTRA.0))
            .contains(&group(EXTRA.1)));
    }
    assert_eq!(
        st.history(v1, v2)
            .unwrap()
            .map(|v| v.id)
            .collect::<Vec<_>>(),
        vec![v2]
    );

    // Refused.
    let before = format!("{:?}", st.image());
    assert_eq!(st.view(ghost).err(), Some(SimError::UnknownVersion(ghost)));
    assert_eq!(
        st.diff(ghost, v1).err(),
        Some(SimError::UnknownVersion(ghost))
    );
    assert_eq!(
        st.diff(v1, ghost).err(),
        Some(SimError::UnknownVersion(ghost))
    );
    assert_eq!(
        st.history(ghost, v2).err(),
        Some(HistoryRefusal::UnknownVersion(ghost))
    );
    assert_eq!(
        st.history(v1, ghost).err(),
        Some(HistoryRefusal::UnknownVersion(ghost))
    );
    assert_eq!(
        st.history(v2, v1).err(),
        Some(HistoryRefusal::NotAncestor { from: v2, to: v1 })
    );
    assert_eq!(
        format!("{:?}", st.image()),
        before,
        "a refusal changes nothing"
    );
}

// ---- P5 ---------------------------------------------------------------------

/// P5: a query in a directory the version does not describe is refused, and
/// the same query in the right directory is answered.
///
/// Fails if: `subtree` ignored the scope (the `OTHER` query would return
/// rows), `diff` compared versions of two directories (the `ScopeMismatch`),
/// or `history` crossed roots (the `NotAncestor`). The same-scope subtree
/// rows and the one-change `diff(v1, v2)` are the can-stay-silent half: the
/// guards do not refuse everything.
#[test]
fn p5_a_foreign_scope_is_refused_and_the_right_scope_answers() {
    let mut st = VersionStore::new();
    let v1 = st.observe("g2", 1, observation()).unwrap();
    let v2 = st
        .reobserve("g2", 2, with_member(observation(), EXTRA.0, EXTRA.1))
        .unwrap();
    {
        let view = st.view(v1).unwrap();
        let p1 = Dn128::new(&[1]).unwrap();
        let p12 = Dn128::new(&[1, 2]).unwrap();
        let want1: Vec<usize> = (0..U).filter(|&u| dn_of(u)[0] == 1).collect();
        let want12: Vec<usize> = (0..U).filter(|&u| dn_of(u) == [1, 2]).collect();
        assert!(!want1.is_empty() && want1.len() < U);
        assert!(!want12.is_empty() && want12.len() < want1.len());
        assert_eq!(
            subtree(&view, SCOPE, NodeKind::User, &p1).unwrap().rows(),
            want1
        );
        assert_eq!(
            subtree(&view, SCOPE, NodeKind::User, &p12).unwrap().rows(),
            want12
        );
        assert_eq!(
            subtree(&view, OTHER, NodeKind::User, &p1).err(),
            Some(ScopeMismatch {
                version: SCOPE,
                query: OTHER
            })
        );
    }
    assert_eq!(
        st.diff(v1, v2).unwrap(),
        vec![Change::AddMembership {
            user: user(EXTRA.0),
            group: group(EXTRA.1)
        }]
    );

    // A read of another directory is a new root, not a re-read of this one.
    let mut other = observation();
    other.scope = OTHER;
    let o = st.observe("other", 3, other).unwrap();
    assert_eq!(st.diff(v1, o).err(), Some(SimError::ScopeMismatch(v1, o)));
    assert_eq!(
        st.history(v1, o).err(),
        Some(HistoryRefusal::NotAncestor { from: v1, to: o })
    );
}

/// P5: authority. A plan is made only for the version tagged desired, and
/// only against an observation of the same directory.
///
/// Fails if: `plan` accepted a version that was never promoted
/// (`NotDesired`), or ignored a later observation of another directory
/// (`ScopeMismatch`). The promoted plan with exactly the one granted
/// membership is the can-stay-silent half. Assumes the fixture validates, as
/// `promote_desired` requires; a violation there fails this test at the
/// `promote_desired` unwrap and is not a plan-authority defect.
#[test]
fn p5_a_plan_needs_the_desired_version_and_the_same_directory() {
    let mut st = VersionStore::new();
    let v = st.observe("g2", 1, observation()).unwrap();
    let grant = GrantGroup {
        rule: GRANT,
        group: group(EXTRA.1),
        to: vec![user(EXTRA.0)],
    };
    let s = st.simulate(v, &grant, &[]).unwrap();
    assert_eq!(st.plan(s), Err(PlanError::NotDesired(s)));

    st.promote_desired(s).unwrap();
    let plan = st.plan(s).unwrap();
    assert_eq!(plan.basis, v);
    assert_eq!(plan.ops.len(), 1);

    let mut other = observation();
    other.scope = OTHER;
    let o = st.observe("other", 2, other).unwrap();
    assert_eq!(
        st.plan(s),
        Err(PlanError::ScopeMismatch {
            basis: o,
            target: s
        })
    );
}

// ---- P6 ---------------------------------------------------------------------

/// P6: reads are bounded by the sparse membership, not by users x groups.
///
/// The membership rows equal the oracle's direct pairs and are under 1/40 of
/// the 262,144 possible pairs; every user's reach and every group's members,
/// summed, equal the oracle's distinct pairs (the two sides of one relation)
/// and stay under 1/16 of users x groups; a count vector has one entry per
/// user. An observed root carries no delta.
///
/// Fails if: a read built the cross product (the sums would approach
/// `U * G`), or a count sink were sized by pairs (`U * G` entries), or an
/// observed root held a delta. These are result sizes, not timings or
/// allocation counts: dir-sim exposes no work counter.
#[test]
fn p6_results_are_bounded_by_the_sparse_membership() {
    let o = oracle();
    let mut st = VersionStore::new();
    let v = st.observe("g2", 1, observation()).unwrap();
    let rows = st.snapshot(v).unwrap().membership_rows();
    assert_eq!(rows, o.direct_rows);
    assert!(rows * 40 < U * G, "{rows} rows of {} pairs", U * G);

    let view = st.view(v).unwrap();
    assert_eq!(view.delta_len(), 0);
    assert_eq!(member_counts(&view, &group(0)).len(), U);

    let mut by_user = 0usize;
    for u in 0..U {
        by_user += view.groups_transitive(&user(u)).len();
    }
    let mut by_group = 0usize;
    for g in 0..G {
        by_group += view.members_transitive(&group(g)).len();
    }
    assert_eq!(by_user, o.distinct);
    assert_eq!(by_group, o.distinct);
    assert!(o.distinct * 16 < U * G, "{} of {} pairs", o.distinct, U * G);
}

/// P6: a version's delta is the size of the change, and it shares the
/// observed snapshot instead of copying the populations.
///
/// One granted membership is one entry, 100 are 100: the bound is the
/// request, not a constant, so the check does not pass on a version that
/// ignores its change. A re-read that adds one pair stores one change and no
/// second root; a re-read that adds 100 stores 100.
///
/// Fails if: simulating copied the snapshot (the pointer comparison), the
/// delta grew with the directory (the `delta_len` equalities), or a re-read
/// stored a whole observation (`roots == 1`, `delta.len()`).
#[test]
fn p6_a_delta_is_the_size_of_the_change() {
    let mut st = VersionStore::new();
    let v = st.observe("g2", 1, observation()).unwrap();
    let targets: Vec<usize> = (0..U)
        .filter(|&u| !is_direct(u, EXTRA.1))
        .take(100)
        .collect();
    assert_eq!(targets.len(), 100);

    let one = GrantGroup {
        rule: GRANT,
        group: group(EXTRA.1),
        to: vec![user(EXTRA.0)],
    };
    let many = GrantGroup {
        rule: GRANT,
        group: group(EXTRA.1),
        to: targets.iter().map(|&u| user(u)).collect(),
    };
    let s1 = st.simulate(v, &one, &[]).unwrap();
    let s100 = st.simulate(v, &many, &[]).unwrap();
    assert_eq!(st.version(s1).unwrap().delta.len(), 1);
    assert_eq!(st.version(s100).unwrap().delta.len(), 100);
    {
        let base = st.view(v).unwrap();
        let (a, b) = (st.view(s1).unwrap(), st.view(s100).unwrap());
        assert_eq!(a.delta_len(), 1);
        assert_eq!(b.delta_len(), 100);
        assert!(std::ptr::eq(a.snapshot(), base.snapshot()));
        assert!(std::ptr::eq(b.snapshot(), base.snapshot()));
    }

    // One pair added by a re-read: one change, no second root.
    let v2 = st
        .reobserve("g2", 2, with_member(observation(), EXTRA.0, EXTRA.1))
        .unwrap();
    assert_eq!(
        st.version(v2).unwrap().delta,
        vec![Change::AddMembership {
            user: user(EXTRA.0),
            group: group(EXTRA.1)
        }]
    );
    assert_eq!(st.image().roots.len(), 1);

    // 100 pairs added by a re-read: 100 changes, from the same root.
    let mut st = VersionStore::new();
    st.observe("g2", 1, observation()).unwrap();
    let mut more = observation();
    for &u in &targets {
        more.members.push((user(u), group(EXTRA.1)));
    }
    let v = st.reobserve("g2", 2, more).unwrap();
    assert_eq!(st.version(v).unwrap().delta.len(), 100);
    assert_eq!(st.image().roots.len(), 1);
}
