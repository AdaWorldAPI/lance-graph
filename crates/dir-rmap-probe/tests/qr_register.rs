//! Arm Q: IAM nested groups as a DR256 register.
//!
//! Every user's roles and effective security groups, read from the
//! users x groups bit matrix and the closure computed inside the 256 x 256
//! tile, must equal dir-sim (arm A), the crepe oracle (arm D) and the
//! compiled closure (arm B+), before and after edits. The register refuses a
//! chain it cannot hold whole, and a stale snapshot.
//!
//! The ClassView wiring is the same as `classview_dr.rs`: the class declares
//! `effective_roles` computed from `member_of`, and the DR table maps that
//! position to `Page::roles_of` on the user's page, resolved on first use.

use dir_rmap_probe::fixture::{Fixture, Spec};
use dir_rmap_probe::prep::{Chain, Compiled, Ordinals, Refusal};
use dir_rmap_probe::qr::{Page, Pages, Qr};
use dir_rmap_probe::{Edit, World, a_groups, oracle, templates};
use lance_graph_contract::class_view::{ClassId, ClassView, ComputeEdge, execute_compute_dag};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use std::collections::HashMap;

const USER: ClassId = 1;
const MEMBER_OF: u8 = 0;
const EFFECTIVE_ROLES: u8 = 1;

struct IamUser {
    fields: Vec<FieldRef>,
}

static ROLES_FROM_MEMBERSHIP: [ComputeEdge; 1] = [ComputeEdge {
    target: EFFECTIVE_ROLES,
    inputs: &[MEMBER_OF],
}];

impl ClassView for IamUser {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == USER { &self.fields } else { &[] }
    }
    fn template(&self, _: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _: ClassId) -> u8 {
        0
    }
    fn compute_dag(&self, class: ClassId) -> &[ComputeEdge] {
        if class == USER {
            &ROLES_FROM_MEMBERSHIP
        } else {
            &[]
        }
    }
}

/// Direct routing over a page: position -> executable.
type Route = fn(&Page, usize) -> u64;
const DR: [Option<Route>; 2] = [None, Some(Page::roles_of)];

fn run(view: &dyn ClassView, page: &Page, i: usize) -> Option<u64> {
    let mut roles = None;
    execute_compute_dag(view.compute_dag(USER), &mut roles, |out, target| {
        let route = DR.get(target as usize).copied().flatten().ok_or(target)?;
        *out = Some(route(page, i));
        Ok::<(), u8>(())
    })
    .expect("every declared target has a route");
    roles
}

fn view() -> IamUser {
    IamUser {
        fields: vec![
            FieldRef::new("urn:iam:user:memberOf", "member_of"),
            FieldRef::new("urn:iam:user:effectiveRoles", "effective_roles"),
        ],
    }
}

/// Checks every user against arms A, D and B+. Returns how many users hold a
/// role and how many (user, group) effective memberships were compared.
fn check(w: &World) -> (usize, usize) {
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).expect("security chain fits 256 slots");
    let mut pages = Pages::new(&qr);
    let compiled = Compiled::compile(w.prepare());
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let d = oracle::run(&w.f, &w.direct, &w.nested, &templates(&w.f));
    let mut d_roles = vec![0u64; w.f.users.len()];
    for &(u, role) in &d.roles {
        d_roles[u as usize] |= 1 << role;
    }
    let grants: HashMap<_, u64> = w.f.grants.iter().fold(HashMap::new(), |mut m, &(g, r)| {
        *m.entry(w.f.groups[g as usize]).or_default() |= 1 << r;
        m
    });
    let group_ord: HashMap<_, u16> =
        w.f.groups
            .iter()
            .enumerate()
            .map(|(i, g)| (*g, ord.group[i]))
            .collect();
    let iam = view();
    let (mut holders, mut compared) = (0, 0);
    for (i, ug) in w.f.users.iter().enumerate() {
        let u = ord.user[i];
        let a = a_groups(&v, ug, Chain::Security);
        let a_roles = a
            .iter()
            .fold(0, |acc, g| acc | grants.get(g).copied().unwrap_or(0));
        let page = pages.of(&qr, &b, u).unwrap();
        let got = run(&iam, page, u as usize % 256).expect("declared");
        let row = page.member[u as usize % 256];
        assert_eq!(qr.roles_live(&row), got, "row x roles, user {i}");
        assert_eq!(got, a_roles, "dir-sim roles, user {i}");
        assert_eq!(got, d_roles[i], "oracle roles, user {i}");
        assert_eq!(got, compiled.roles_of(u), "B+ roles, user {i}");
        holders += usize::from(got != 0);
        // Effective security groups: the matrix row times reach, slot by
        // slot, against dir-sim's walk.
        let eff = qr.effective(&row);
        let a_set: Vec<u16> = a.iter().map(|g| group_ord[g]).collect();
        for (s, &g) in qr.ids.iter().enumerate() {
            let here = eff[s >> 6] >> (s & 63) & 1 == 1;
            assert_eq!(here, a_set.contains(&g), "user {i}, group ordinal {g}");
            compared += 1;
        }
    }
    assert_eq!(pages.resolved, qr.pages(), "every page asked for once");
    (holders, compared)
}

#[test]
fn the_register_answers_every_user_like_dir_sim_the_oracle_and_b_plus() {
    let w = World::new(Fixture::generate(Spec::small()));
    let (holders, compared) = check(&w);
    // Anti-vacuity: roles held by some users and not all; nesting carries
    // at least one membership the direct rows do not.
    assert!(
        holders > 0 && holders < w.f.users.len(),
        "holders {holders}"
    );
    assert!(compared > 0);
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).unwrap();
    let mut indirect = 0;
    for p in 0..qr.pages() {
        let page = qr.page(&b, p).unwrap();
        for row in &page.member {
            indirect += usize::from(qr.effective(row) != *row);
        }
    }
    assert!(indirect > 0, "no user gains a group through nesting");
}

#[test]
fn edits_are_answered_by_a_rebind() {
    let mut w = World::new(Fixture::generate(Spec::small()));
    check(&w);
    let sec = |w: &World, g: u32| w.f.security[g as usize] == Some(true);
    // A new nesting pair between two security groups, and a membership
    // removed: both change effective groups for someone.
    let (c, p) = (0..w.f.groups.len() as u32)
        .flat_map(|c| (0..w.f.groups.len() as u32).map(move |p| (c, p)))
        .find(|&(c, p)| c != p && sec(&w, c) && sec(&w, p) && !w.nested.contains(&(c, p)))
        .expect("two unnested security groups");
    let (u, g) = *w.direct.iter().find(|&&(_, g)| sec(&w, g)).unwrap();
    w.apply(&[Edit::AddNested(c, p), Edit::RemoveDirect(u, g)]);
    check(&w);
}

#[test]
fn a_stale_register_refuses() {
    let mut w = World::new(Fixture::generate(Spec::small()));
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).unwrap();
    let mut pages = Pages::new(&qr);
    let g = qr.ids[0];
    assert!(pages.ask(&qr, &b, &w.witness(), 0, g).is_ok());
    let (u, gd) = w.direct[0];
    w.apply(&[Edit::RemoveDirect(u, gd)]);
    assert_eq!(pages.ask(&qr, &b, &w.witness(), 0, g), Err(Refusal::Stale));
    assert_eq!(Qr::bind(&b, &w.witness()).err(), Some(Refusal::Stale));
    // A page cannot be resolved from a snapshot the register was not bound
    // to.
    let b2 = w.prepare();
    assert_eq!(qr.page(&b2, 0).err(), Some(Refusal::Stale));
}

#[test]
fn a_chain_larger_than_the_register_is_refused_not_truncated() {
    let w = World::new(Fixture::generate(Spec::iam(2000)));
    let b = w.prepare();
    let security: u32 = b.security.iter().map(|x| x.count_ones()).sum();
    assert!(
        security > 256,
        "fixture must exceed the register: {security}"
    );
    assert_eq!(Qr::bind(&b, &w.witness()).err(), Some(Refusal::TooLarge));
}

#[test]
fn a_group_outside_the_security_chain_is_out_of_focus() {
    let w = World::new(Fixture::generate(Spec::small()));
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).unwrap();
    let g = (0..b.groups as u16)
        .find(|&g| qr.slot(g).is_none())
        .expect("a non-security group");
    let mut pages = Pages::new(&qr);
    assert_eq!(
        pages.ask(&qr, &b, &w.witness(), 0, g),
        Err(Refusal::OutOfFocus)
    );
}

#[test]
fn binding_resolves_no_user_and_a_question_resolves_one_page() {
    let mut spec = Spec::small();
    spec.users = 2000;
    let w = World::new(Fixture::generate(spec));
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).unwrap();
    let mut pages = Pages::new(&qr);
    assert_eq!(pages.resolved, 0);
    assert!(qr.pages() > 2);
    // Two users of one page, then one of the last page.
    pages.roles_of(&qr, &b, 3).unwrap();
    pages.roles_of(&qr, &b, 200).unwrap();
    assert_eq!(pages.resolved, 1);
    pages.roles_of(&qr, &b, (b.users - 1) as u16).unwrap();
    assert_eq!(pages.resolved, 2);
}
