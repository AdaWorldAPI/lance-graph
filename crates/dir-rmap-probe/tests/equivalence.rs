//! Every prepared arm equals dir-sim's own walks (A) and the independent
//! crepe oracle (D), on every user and group of a small fixture, before and
//! after an overlay. Plus the refusals and the anti-vacuity checks that keep
//! the comparisons meaningful.

use dir_rmap_probe::fixture::{Fixture, Kind, Spec};
use dir_rmap_probe::focus::{Focus, FocusLazy, FocusUsers, SLOTS, holds};
use dir_rmap_probe::prep::{CHAINS, Chain, Compiled, Ordinals, Prepared, Refusal, ones, words};
use dir_rmap_probe::{
    Edit, World, a_groups, a_members, composite_groups, conjunctions, focus_for, oracle, templates,
};
use ogar_dir_core::Guid128;
use std::collections::{BTreeSet, HashMap};

fn world() -> World {
    World::new(Fixture::generate(Spec::small()))
}

fn sorted(mut v: Vec<Guid128>) -> Vec<Guid128> {
    v.sort_unstable();
    v
}

/// A plane over group ordinals, as GUIDs.
fn groups_of_plane(w: &World, p: &[u64]) -> Vec<Guid128> {
    let v = w.view();
    sorted(
        ones(p)
            .map(|g| {
                v.group_guid(lance_graph_dir_sim::GroupOrdinal(g as u16))
                    .unwrap()
            })
            .collect(),
    )
}

fn users_of_plane(w: &World, p: &[u64]) -> Vec<Guid128> {
    let v = w.view();
    sorted(
        ones(p)
            .map(|u| {
                v.user_guid(lance_graph_dir_sim::UserOrdinal(u as u16))
                    .unwrap()
            })
            .collect(),
    )
}

fn oracle_rel(a: &oracle::Answers, chain: Chain) -> &BTreeSet<(u32, u32)> {
    match chain {
        Chain::All => &a.member,
        Chain::Security => &a.sec,
        Chain::Mail => &a.mail,
    }
}

/// Roles as dir-sim answers them: the grants of the SIDs it reports, keyed
/// by GUID, the shape `ViewActors::for_principal` reads.
fn a_roles(w: &World, u: &Guid128) -> u64 {
    let by_guid: HashMap<Guid128, u64> =
        w.f.grants.iter().fold(HashMap::new(), |mut m, &(g, r)| {
            *m.entry(w.f.groups[g as usize]).or_default() |= 1 << r;
            m
        });
    a_groups(&w.view(), u, Chain::Security)
        .iter()
        .fold(0, |a, g| a | by_guid.get(g).copied().unwrap_or(0))
}

/// The full comparison of A, B, B+ and D for one version.
fn check_all(w: &World, b: &Prepared, c: &Compiled) {
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let t = templates(&w.f);
    let d = oracle::run(&w.f, &w.direct, &w.nested, &t);
    let (gw, uw) = (words(v.groups_len()), words(v.users_len()));
    let (mut p, mut q, mut r) = (vec![0; gw], vec![0; gw], vec![0; uw]);
    let mut queue = Vec::new();
    for chain in CHAINS {
        let rel = oracle_rel(&d, chain);
        for (i, ug) in w.f.users.iter().enumerate() {
            let a = sorted(a_groups(&v, ug, chain));
            let uo = ord.user[i];
            b.groups_of(uo, chain, &mut p, &mut queue);
            c.groups_of(uo, chain, &mut q);
            let dd = sorted(
                rel.range((i as u32, 0)..=(i as u32, u32::MAX))
                    .map(|&(_, g)| w.f.groups[g as usize])
                    .collect(),
            );
            assert_eq!(groups_of_plane(w, &p), a, "B groups, {chain:?}, user {i}");
            assert_eq!(groups_of_plane(w, &q), a, "B+ groups, {chain:?}, user {i}");
            assert_eq!(dd, a, "oracle groups, {chain:?}, user {i}");
        }
        let mut by_group: HashMap<u32, Vec<Guid128>> = HashMap::new();
        for &(u, g) in rel {
            by_group.entry(g).or_default().push(w.f.users[u as usize]);
        }
        for (j, gg) in w.f.groups.iter().enumerate() {
            let a = sorted(a_members(&v, gg, chain));
            let go = ord.group[j];
            b.members_of(go, chain, &mut q, &mut r, &mut queue);
            assert_eq!(users_of_plane(w, &r), a, "B members, {chain:?}, group {j}");
            c.members_of(go, chain, &mut r);
            assert_eq!(users_of_plane(w, &r), a, "B+ members, {chain:?}, group {j}");
            let dd = sorted(by_group.remove(&(j as u32)).unwrap_or_default());
            assert_eq!(dd, a, "oracle members, {chain:?}, group {j}");
        }
    }
    let mut d_roles = vec![0u64; w.f.users.len()];
    for &(u, role) in &d.roles {
        d_roles[u as usize] |= 1 << role;
    }
    for (i, ug) in w.f.users.iter().enumerate() {
        let a = a_roles(w, ug);
        assert_eq!(
            b.roles_of(ord.user[i], &mut p, &mut queue),
            a,
            "B roles, user {i}"
        );
        assert_eq!(c.roles_of(ord.user[i]), a, "B+ roles, user {i}");
        assert_eq!(d_roles[i], a, "oracle roles, user {i}");
    }
    // Direct membership, every (user, group) pair.
    for (i, ug) in w.f.users.iter().enumerate() {
        for (j, gg) in w.f.groups.iter().enumerate() {
            assert_eq!(
                b.is_member(ord.user[i], ord.group[j]),
                v.is_member(ug, gg),
                "B is_member ({i}, {j})"
            );
        }
    }
}

#[test]
fn every_arm_equals_dir_sim_and_the_oracle() {
    let w = world();
    let b = w.prepare();
    let c = Compiled::compile(w.prepare());
    check_all(&w, &b, &c);
}

#[test]
fn the_focus_answers_role_and_location_like_the_oracle_and_dir_sim() {
    let w = world();
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let c = Compiled::compile(w.prepare());
    let all = templates(&w.f);
    let d = oracle::run(&w.f, &w.direct, &w.nested, &all);
    let mut checked = 0;
    for k in [16, 64, 256] {
        let (ts, gs) = focus_for(&all, k);
        assert!(!ts.is_empty() && gs.len() <= k);
        let groups: Vec<u16> = gs.iter().map(|&g| ord.group[g as usize]).collect();
        let f = Focus::bind(&c, Chain::All, &groups, &w.witness()).unwrap();
        let fu = FocusUsers::bind(&c, Chain::All, &groups, &w.witness()).unwrap();
        for (i, ug) in w.f.users.iter().enumerate() {
            let a = a_groups(&v, ug, Chain::All);
            let row = f.row_of(&c, ord.user[i]);
            for &(r, l) in &ts {
                let m = f
                    .all_of(&[ord.group[r as usize], ord.group[l as usize]])
                    .unwrap();
                let want =
                    a.contains(&w.f.groups[r as usize]) && a.contains(&w.f.groups[l as usize]);
                assert_eq!(holds(&row, &m), want, "F k={k} user {i} ({r}, {l})");
                assert_eq!(holds(&fu.rows[ord.user[i] as usize], &m), want, "F+users");
                assert_eq!(d.eligible.contains(&(i as u32, r, l)), want, "oracle");
                checked += 1;
            }
        }
    }
    assert!(
        checked > 10_000,
        "the comparison covered {checked} questions"
    );
}

#[test]
fn overlay_incremental_equals_recompile_dir_sim_and_the_oracle() {
    let mut w = world();
    let mut c = Compiled::compile(w.prepare());
    let f = &w.f;
    let team = f.teams[0];
    let role = f.roles[3];
    let removed = *f
        .direct
        .iter()
        .find(|&&(_, g)| f.kind[g as usize] == Kind::Composite)
        .unwrap();
    // A nesting that closes a cycle through a role and a resource, and one
    // that links a team (not security) into a role.
    let res = f.nested.iter().find(|&&(c, _)| c == role).unwrap().1;
    let edits = [
        Edit::AddDirect(5, team),
        Edit::AddDirect(7, role),
        Edit::RemoveDirect(removed.0, removed.1),
        Edit::AddNested(res, role),
        Edit::AddNested(team, f.roles[9]),
    ];
    w.apply(&edits);
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    // Ordinals are unchanged by membership edits, so the incremental path
    // may reuse them; the witness still moves to the new version.
    c.b.set_direct(&ord, &w.direct);
    for e in &edits {
        if let Edit::AddNested(ch, p) = *e {
            c.add_nesting(ord.group[ch as usize], ord.group[p as usize]);
        }
    }
    c.b.witness = w.witness();
    let full = Compiled::compile(w.prepare());
    for chain in CHAINS {
        assert_eq!(
            c.up[chain as usize], full.up[chain as usize],
            "{chain:?} up"
        );
        assert_eq!(
            c.down[chain as usize], full.down[chain as usize],
            "{chain:?} down"
        );
    }
    assert_eq!(c.role_sec, full.role_sec);
    check_all(&w, &w.prepare(), &full);
}

#[test]
fn a_stale_witness_and_an_unfocused_group_are_refused() {
    let mut w = world();
    let b = w.prepare();
    assert_eq!(b.check(&w.witness()), Ok(()));
    let f = &w.f;
    let (u, g) = (11, f.teams[1]);
    w.apply(&[Edit::AddDirect(u, g)]);
    assert_eq!(b.check(&w.witness()), Err(Refusal::Stale));
    let c = Compiled::compile(w.prepare());
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let focus = Focus::bind(
        &c,
        Chain::All,
        &[ord.group[w.f.roles[0] as usize]],
        &w.witness(),
    )
    .unwrap();
    assert_eq!(focus.check(&w.witness()), Ok(()));
    assert!(focus.all_of(&[ord.group[w.f.roles[0] as usize]]).is_ok());
    assert_eq!(
        focus.all_of(&[ord.group[w.f.roles[1] as usize]]),
        Err(Refusal::OutOfFocus)
    );
}

/// The fixture must exercise what the comparisons claim to cover.
#[test]
fn the_fixture_is_not_vacuous() {
    let w = world();
    let f = &w.f;
    let cyclic = f
        .nested
        .iter()
        .filter(|&&(a, b)| f.nested.binary_search(&(b, a)).is_ok())
        .count();
    assert!(cyclic >= 2, "the nesting has cycles");
    assert!(
        f.large_groups() >= 1,
        "some group exceeds 256 direct members"
    );
    let c = Compiled::compile(w.prepare());
    assert!(
        c.passes[Chain::All as usize].0 > 1,
        "the closure needs more than one pass"
    );
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let gw = words(v.groups_len());
    let (mut all, mut sec, mut mail) = (vec![0; gw], vec![0; gw], vec![0; gw]);
    let (mut sec_differs, mut mail_differs, mut roles) = (0, 0, 0);
    let (mut sec_some, mut mail_some) = (0, 0);
    for u in 0..f.users.len() {
        let uo = ord.user[u];
        c.groups_of(uo, Chain::All, &mut all);
        c.groups_of(uo, Chain::Security, &mut sec);
        c.groups_of(uo, Chain::Mail, &mut mail);
        sec_differs += usize::from(sec != all);
        mail_differs += usize::from(mail != all);
        sec_some += usize::from(sec.iter().any(|&w| w != 0));
        mail_some += usize::from(mail.iter().any(|&w| w != 0));
        roles += usize::from(c.roles_of(uo) != 0);
    }
    let n = f.users.len();
    // Each chain removes something for many users and empties it for few:
    // a filter that changed nothing, or kept nothing, would not test the
    // chain semantics.
    assert!(
        sec_differs * 10 > n,
        "the SID chain filters: {sec_differs}/{n}"
    );
    assert!(
        sec_some * 10 > n * 9,
        "the SID chain keeps groups: {sec_some}/{n}"
    );
    assert!(
        mail_differs * 10 > n,
        "the mail chain filters: {mail_differs}/{n}"
    );
    assert!(
        mail_some * 10 > n * 9,
        "the mail chain keeps groups: {mail_some}/{n}"
    );
    assert!(roles * 2 > n, "most users hold a role: {roles}/{n}");
    let t = templates(f);
    let d = oracle::run(f, &w.direct, &w.nested, &t);
    let total = n * t.len();
    assert!(
        d.eligible.len() * 50 > total / 50 && d.eligible.len() * 2 < total,
        "role x location is neither empty nor everything: {}/{total}",
        d.eligible.len()
    );
}

/// `SLOTS` group ordinals spread over the whole ordinal range.
fn spread(groups: usize) -> Vec<u16> {
    let step = groups.div_ceil(SLOTS).max(1);
    (0..groups)
        .step_by(step)
        .take(SLOTS)
        .map(|g| g as u16)
        .collect()
}

/// For every chain, a focus over 256 spread groups answers every
/// (user, slot) pair like dir-sim's own chain walk and the oracle, through
/// the eager and the lazy per-user rows alike; and the down-row projection
/// equals the reference projection.
#[test]
fn a_focus_on_every_chain_equals_dir_sim_and_the_oracle() {
    let w = world();
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let c = Compiled::compile(w.prepare());
    let d = oracle::run(&w.f, &w.direct, &w.nested, &templates(&w.f));
    let ids = spread(v.groups_len());
    // Fixture id of every group ordinal, to read the oracle.
    let mut fid = vec![0u32; v.groups_len()];
    for (j, &o) in ord.group.iter().enumerate() {
        fid[o as usize] = j as u32;
    }
    let mut yes = [0usize; 3];
    for chain in CHAINS {
        let f = Focus::bind(&c, chain, &ids, &w.witness()).unwrap();
        let mut reference = Focus::slots(&c, chain, &ids, &w.witness()).unwrap();
        reference.project_by_test(&c);
        for g in 0..v.groups_len() {
            assert_eq!(
                f.projected(g),
                reference.projected(g),
                "{chain:?} group {g}"
            );
        }
        let fu = FocusUsers::bind(&c, chain, &ids, &w.witness()).unwrap();
        let mut fl = FocusLazy::from_focus(&c, Focus::bind(&c, chain, &ids, &w.witness()).unwrap());
        let rel = oracle_rel(&d, chain);
        for (i, ug) in w.f.users.iter().enumerate() {
            let a = a_groups(&v, ug, chain);
            let u = ord.user[i];
            let row = f.row_of(&c, u);
            assert_eq!(fu.rows[u as usize], row, "eager row");
            assert_eq!(fl.row(&c, u), row, "lazy row");
            for (s, &g) in ids.iter().enumerate() {
                let m = f.all_of(&[g]).unwrap();
                let want = a.contains(&v.group_guid(lance_graph_dir_sim::GroupOrdinal(g)).unwrap());
                assert_eq!(holds(&row, &m), want, "F {chain:?} user {i} slot {s}");
                assert_eq!(
                    rel.contains(&(i as u32, fid[g as usize])),
                    want,
                    "oracle {chain:?}"
                );
                yes[chain as usize] += usize::from(want);
            }
        }
        assert_eq!(fl.filled, w.f.users.len());
    }
    // Anti-vacuity: every chain has positive answers, and the filtered
    // chains answer fewer of them than the unfiltered one.
    assert!(yes.iter().all(|&y| y > 1_000), "{yes:?}");
    assert!(yes[1] < yes[0] && yes[2] < yes[0], "{yes:?}");
}

/// A path that runs through groups outside the focus is represented: for
/// users whose membership in a focus group needs at least two nesting hops
/// (the group is neither a direct group of the user nor a direct parent of
/// one), the focus holding only that group still answers yes. A
/// focus-local closure would answer no.
#[test]
fn a_path_through_unfocused_groups_is_represented() {
    let w = world();
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let c = Compiled::compile(w.prepare());
    let gw = words(v.groups_len());
    let mut row = vec![0u64; gw];
    let mut checked = [0usize; 3];
    for chain in CHAINS {
        for i in (0..w.f.users.len()).step_by(7) {
            let u = ord.user[i];
            let direct = c.b.user_groups.row(u as usize);
            c.groups_of(u, chain, &mut row);
            let far: Vec<u16> = ones(&row)
                .map(|g| g as u16)
                .filter(|g| !direct.contains(g))
                .filter(|&g| !direct.iter().any(|&d| c.b.up.row(d as usize).contains(&g)))
                .take(4)
                .collect();
            for g in far {
                let f = Focus::bind(&c, chain, &[g], &w.witness()).unwrap();
                assert_eq!(
                    f.ask(&c, &w.witness(), u, &[g]),
                    Ok(true),
                    "{chain:?} user {i} group {g}"
                );
                checked[chain as usize] += 1;
            }
        }
    }
    assert!(
        checked.iter().all(|&n| n >= 20),
        "two-hop cases per chain: {checked:?}"
    );
}

/// After an overlay (membership adds and removals, a nesting pair that
/// closes a cycle, a team nested in a role), a focus refreshed
/// incrementally equals a fresh bind, dir-sim and the oracle, on every
/// chain, eager and lazy rows alike. Before the refresh it refuses.
#[test]
fn a_refreshed_focus_equals_a_fresh_bind_dir_sim_and_the_oracle() {
    let mut w = world();
    let v0 = w.view();
    let ord = Ordinals::of(&w.f, &v0);
    drop(v0);
    let mut c = Compiled::compile(w.prepare());
    let ids = spread(c.b.groups);
    let w0 = w.witness();
    let mut fs: Vec<FocusUsers> = CHAINS
        .iter()
        .map(|&ch| FocusUsers::bind(&c, ch, &ids, &w0).unwrap())
        .collect();
    let mut fl: Vec<FocusLazy> = CHAINS
        .iter()
        .map(|&ch| FocusLazy::from_focus(&c, Focus::bind(&c, ch, &ids, &w0).unwrap()))
        .collect();
    for (k, l) in fl.iter_mut().enumerate() {
        for u in 0..c.b.users as u16 {
            let _ = l.row(&c, u);
        }
        assert_eq!(l.filled, c.b.users, "chain {k} warmed");
    }
    let f = &w.f;
    let role = f.roles[3];
    let res = f.nested.iter().find(|&&(c, _)| c == role).unwrap().1;
    let removed = *f
        .direct
        .iter()
        .find(|&&(_, g)| f.kind[g as usize] == Kind::Composite)
        .unwrap();
    let edits = [
        Edit::AddDirect(5, f.teams[0]),
        Edit::AddDirect(7, role),
        Edit::RemoveDirect(removed.0, removed.1),
        Edit::AddNested(res, role),
        Edit::AddNested(f.teams[0], f.roles[9]),
    ];
    w.apply(&edits);
    let w1 = w.witness();
    for fu in &fs {
        assert_eq!(
            fu.focus.check(&w1),
            Err(Refusal::Stale),
            "refuses before refresh"
        );
    }
    // Incremental: membership CSRs, then each nesting pair.
    c.b.set_direct(&ord, &w.direct);
    c.b.witness = w1;
    let edited: Vec<u16> = edits
        .iter()
        .filter_map(|e| match *e {
            Edit::AddDirect(u, _) | Edit::RemoveDirect(u, _) => Some(ord.user[u as usize]),
            Edit::AddNested(..) => None,
        })
        .collect();
    for (fu, l) in fs.iter_mut().zip(fl.iter_mut()) {
        fu.focus.refresh_membership(&c);
        fu.refresh_users(&c, &edited);
        l.focus.refresh_membership(&c);
        l.refresh_users(&edited);
    }
    for e in &edits {
        if let Edit::AddNested(ch, p) = *e {
            let (ch, p) = (ord.group[ch as usize], ord.group[p as usize]);
            c.add_nesting(ch, p);
            for (fu, l) in fs.iter_mut().zip(fl.iter_mut()) {
                fu.refresh_nesting(&c, ch);
                l.focus.refresh_nesting(&c, ch);
                l.invalidate();
            }
        }
    }
    let full = Compiled::compile(w.prepare());
    let v = w.view();
    let d = oracle::run(&w.f, &w.direct, &w.nested, &templates(&w.f));
    let mut fid = vec![0u32; v.groups_len()];
    for (j, &o) in ord.group.iter().enumerate() {
        fid[o as usize] = j as u32;
    }
    for (k, &chain) in CHAINS.iter().enumerate() {
        let fresh = Focus::bind(&full, chain, &ids, &w1).unwrap();
        assert_eq!(fs[k].focus.check(&w1), Ok(()));
        for g in 0..v.groups_len() {
            assert_eq!(
                fs[k].focus.projected(g),
                fresh.projected(g),
                "{chain:?} group {g}"
            );
        }
        let rel = oracle_rel(&d, chain);
        for (i, ug) in w.f.users.iter().enumerate() {
            let u = ord.user[i];
            let row = fresh.row_of(&full, u);
            assert_eq!(fs[k].rows[u as usize], row, "eager {chain:?} user {i}");
            assert_eq!(fl[k].row(&c, u), row, "lazy {chain:?} user {i}");
            let a = a_groups(&v, ug, chain);
            for &g in &ids {
                let want = a.contains(&v.group_guid(lance_graph_dir_sim::GroupOrdinal(g)).unwrap());
                assert_eq!(holds(&row, &fresh.all_of(&[g]).unwrap()), want);
                assert_eq!(rel.contains(&(i as u32, fid[g as usize])), want);
            }
        }
    }
}

/// The two forms of role x location stay distinct questions. On the
/// composite fixture, "is the user in composite C" is one membership bit;
/// on the conjunction fixture (no composites), "in role R and in location
/// L" is a mask test of two bits. Each agrees with dir-sim and the oracle,
/// and neither creates a membership the other fixture does not have.
#[test]
fn composite_membership_and_policy_conjunction_are_distinct_questions() {
    // Composite fixture: one bit per composite.
    let w = world();
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let c = Compiled::compile(w.prepare());
    let comps = composite_groups(&w.f);
    assert!(comps.len() >= 20);
    let ids: Vec<u16> = comps
        .iter()
        .take(SLOTS)
        .map(|&g| ord.group[g as usize])
        .collect();
    let f = Focus::bind(&c, Chain::All, &ids, &w.witness()).unwrap();
    let d = oracle::run(&w.f, &w.direct, &w.nested, &[]);
    let mut yes = 0;
    for (i, ug) in w.f.users.iter().enumerate() {
        let a = a_groups(&v, ug, Chain::All);
        let row = f.row_of(&c, ord.user[i]);
        for (&cg, &o) in comps.iter().zip(&ids) {
            let want = a.contains(&w.f.groups[cg as usize]);
            assert_eq!(holds(&row, &f.all_of(&[o]).unwrap()), want);
            assert_eq!(d.member.contains(&(i as u32, cg)), want);
            yes += usize::from(want);
        }
    }
    let total = w.f.users.len() * ids.len();
    assert!(
        yes > 0 && yes * 4 < total,
        "composite membership: {yes}/{total}"
    );

    // Conjunction fixture: no composite exists; the question is two bits.
    let wc = World::new(Fixture::generate(Spec::small().without_composites()));
    assert!(composite_groups(&wc.f).is_empty());
    let vc = wc.view();
    let ordc = Ordinals::of(&wc.f, &vc);
    let cc = Compiled::compile(wc.prepare());
    let all = conjunctions(&wc.f);
    let (ts, gs) = focus_for(&all, SLOTS);
    assert!(ts.len() >= 20, "conjunction questions: {}", ts.len());
    let fc = Focus::bind(
        &cc,
        Chain::All,
        &gs.iter()
            .map(|&g| ordc.group[g as usize])
            .collect::<Vec<_>>(),
        &wc.witness(),
    )
    .unwrap();
    let dc = oracle::run(&wc.f, &wc.direct, &wc.nested, &ts);
    let mut yes = 0;
    for (i, ug) in wc.f.users.iter().enumerate() {
        let a = a_groups(&vc, ug, Chain::All);
        let row = fc.row_of(&cc, ordc.user[i]);
        for &(r, l) in &ts {
            let m = fc
                .all_of(&[ordc.group[r as usize], ordc.group[l as usize]])
                .unwrap();
            let want = a.contains(&wc.f.groups[r as usize]) && a.contains(&wc.f.groups[l as usize]);
            assert_eq!(holds(&row, &m), want);
            assert_eq!(dc.eligible.contains(&(i as u32, r, l)), want);
            yes += usize::from(want);
        }
    }
    let total = wc.f.users.len() * ts.len();
    assert!(yes > 0 && yes * 4 < total, "conjunctions: {yes}/{total}");
    // The conjunction fixture's memberships are exactly its direct and
    // nesting edges: the conjunction question added none.
    assert_eq!(dc.member.len(), {
        let gw = words(vc.groups_len());
        let mut p = vec![0u64; gw];
        (0..wc.f.users.len())
            .map(|i| {
                cc.groups_of(ordc.user[i], Chain::All, &mut p);
                p.iter().map(|w| w.count_ones() as usize).sum::<usize>()
            })
            .sum::<usize>()
    });
}

/// A focus refuses what it cannot answer completely: more than 256 groups,
/// a duplicate, a B+ from another version, and (through `ask`) a stale
/// focus or an unfocused group.
#[test]
fn a_focus_refuses_what_it_cannot_answer_completely() {
    let mut w = world();
    let c = Compiled::compile(w.prepare());
    let w0 = w.witness();
    let many: Vec<u16> = (0..=SLOTS as u16).collect();
    assert_eq!(
        Focus::bind(&c, Chain::All, &many, &w0).err(),
        Some(Refusal::TooLarge)
    );
    assert_eq!(
        Focus::bind(&c, Chain::All, &[3, 4, 3], &w0).err(),
        Some(Refusal::Duplicate)
    );
    let f = Focus::bind(&c, Chain::Security, &[3, 4], &w0).unwrap();
    assert!(f.ask(&c, &w0, 0, &[3]).is_ok());
    assert_eq!(f.ask(&c, &w0, 0, &[5]), Err(Refusal::OutOfFocus));
    w.apply(&[Edit::AddDirect(11, w.f.teams[1])]);
    let w1 = w.witness();
    assert_eq!(
        Focus::bind(&c, Chain::All, &[3], &w1).err(),
        Some(Refusal::Stale)
    );
    assert_eq!(f.ask(&c, &w1, 0, &[3]), Err(Refusal::Stale));
    // A current B+ does not make an old focus current: the focus's own
    // witness refuses.
    let c1 = Compiled::compile(w.prepare());
    assert_eq!(c1.b.check(&w1), Ok(()));
    assert_eq!(f.ask(&c1, &w1, 0, &[3]), Err(Refusal::Stale));
}
