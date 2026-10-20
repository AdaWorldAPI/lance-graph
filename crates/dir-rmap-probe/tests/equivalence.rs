//! Every prepared arm equals dir-sim's own walks (A) and the independent
//! crepe oracle (D), on every user and group of a small fixture, before and
//! after an overlay. Plus the refusals and the anti-vacuity checks that keep
//! the comparisons meaningful.

use dir_rmap_probe::fixture::{Fixture, Kind, Spec};
use dir_rmap_probe::focus::{holds, Focus, FocusUsers};
use dir_rmap_probe::prep::{ones, words, Chain, Compiled, Ordinals, Prepared, Refusal, CHAINS};
use dir_rmap_probe::{a_groups, a_members, focus_for, oracle, templates, Edit, World};
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
            .map(|g| v.group_guid(lance_graph_dir_sim::GroupOrdinal(g as u16)).unwrap())
            .collect(),
    )
}

fn users_of_plane(w: &World, p: &[u64]) -> Vec<Guid128> {
    let v = w.view();
    sorted(
        ones(p)
            .map(|u| v.user_guid(lance_graph_dir_sim::UserOrdinal(u as u16)).unwrap())
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
    let by_guid: HashMap<Guid128, u64> = w.f.grants.iter().fold(HashMap::new(), |mut m, &(g, r)| {
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
        assert_eq!(b.roles_of(ord.user[i], &mut p, &mut queue), a, "B roles, user {i}");
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
        let f = Focus::bind(&c, &groups);
        let fu = FocusUsers::bind(&c, &groups);
        for (i, ug) in w.f.users.iter().enumerate() {
            let a = a_groups(&v, ug, Chain::All);
            let row = f.row_of(&c, ord.user[i]);
            for &(r, l) in &ts {
                let m = f
                    .all_of(&[ord.group[r as usize], ord.group[l as usize]])
                    .unwrap();
                let want = a.contains(&w.f.groups[r as usize]) && a.contains(&w.f.groups[l as usize]);
                assert_eq!(holds(&row, &m), want, "F k={k} user {i} ({r}, {l})");
                assert_eq!(holds(&fu.rows[ord.user[i] as usize], &m), want, "F+users");
                assert_eq!(d.eligible.contains(&(i as u32, r, l)), want, "oracle");
                checked += 1;
            }
        }
    }
    assert!(checked > 10_000, "the comparison covered {checked} questions");
}

#[test]
fn overlay_incremental_equals_recompile_dir_sim_and_the_oracle() {
    let mut w = world();
    let mut c = Compiled::compile(w.prepare());
    let f = &w.f;
    let team = f.teams[0];
    let role = f.roles[3];
    let removed = *f.direct.iter().find(|&&(_, g)| f.kind[g as usize] == Kind::Composite).unwrap();
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
        assert_eq!(c.up[chain as usize], full.up[chain as usize], "{chain:?} up");
        assert_eq!(c.down[chain as usize], full.down[chain as usize], "{chain:?} down");
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
    let focus = Focus::bind(&c, &[ord.group[w.f.roles[0] as usize]]);
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
    let cyclic = f.nested.iter().filter(|&&(a, b)| f.nested.binary_search(&(b, a)).is_ok()).count();
    assert!(cyclic >= 2, "the nesting has cycles");
    assert!(f.large_groups() >= 1, "some group exceeds 256 direct members");
    let c = Compiled::compile(w.prepare());
    assert!(c.passes[Chain::All as usize].0 > 1, "the closure needs more than one pass");
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
    assert!(sec_differs * 10 > n, "the SID chain filters: {sec_differs}/{n}");
    assert!(sec_some * 10 > n * 9, "the SID chain keeps groups: {sec_some}/{n}");
    assert!(mail_differs * 10 > n, "the mail chain filters: {mail_differs}/{n}");
    assert!(mail_some * 10 > n * 9, "the mail chain keeps groups: {mail_some}/{n}");
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
