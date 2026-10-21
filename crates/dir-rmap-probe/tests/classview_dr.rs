//! ClassView × DR, IAM case: a behavior compiled from what the class knows.
//!
//! The user class declares, through the contract's existing `ClassView`
//! surface, that `effective_roles` is computed from `member_of`
//! (`compute_dag`). Direct routing maps that target position to an existing
//! executable: the compiled membership-closure fold `Compiled::roles_of`
//! (arm B+). The contract's `execute_compute_dag` runs it. No relationship
//! type, association model or planner is introduced: the class names the
//! behavior, the routing table names the fold, the bindings are the probe's.
//!
//! Checked on every user against dir-sim's own walks (arm A) and the crepe
//! oracle (arm D), before and after an overlay that removes one of two
//! support paths.

use dir_rmap_probe::fixture::{Fixture, Spec};
use dir_rmap_probe::prep::{Chain, Compiled, Ordinals};
use dir_rmap_probe::{Edit, World, a_groups, oracle, templates};
use lance_graph_contract::class_view::{ClassId, ClassView, ComputeEdge, execute_compute_dag};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use ogar_dir_core::Guid128;
use std::collections::HashMap;

const USER: ClassId = 1;
/// Field positions of the user class: its field basis.
const MEMBER_OF: u8 = 0;
const EFFECTIVE_ROLES: u8 = 1;

/// What the class knows: two fields, and that one is computed from the other.
struct IamUser {
    fields: Vec<FieldRef>,
    dag: &'static [ComputeEdge],
}

static ROLES_FROM_MEMBERSHIP: [ComputeEdge; 1] = [ComputeEdge {
    target: EFFECTIVE_ROLES,
    inputs: &[MEMBER_OF],
}];

impl IamUser {
    fn new(dag: &'static [ComputeEdge]) -> Self {
        IamUser {
            fields: vec![
                FieldRef::new("urn:iam:user:memberOf", "member_of"),
                FieldRef::new("urn:iam:user:effectiveRoles", "effective_roles"),
            ],
            dag,
        }
    }
}

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
        if class == USER { self.dag } else { &[] }
    }
}

/// Direct routing: field position → the executable bound to it. `None` for
/// a position that is read from the bindings, not computed.
type Route = fn(&Compiled, u16) -> u64;
const DR: [Option<Route>; 2] = [None, Some(Compiled::roles_of)];

/// The per-user store the compiled behavior writes into.
struct Cell<'a> {
    compiled: &'a Compiled,
    user: u16,
    roles: Option<u64>,
}

/// Run what the class declares for one user. `None` when the class does not
/// declare the behavior: nothing is computed that the class does not know.
fn run(view: &dyn ClassView, compiled: &Compiled, user: u16) -> Option<u64> {
    let mut cell = Cell {
        compiled,
        user,
        roles: None,
    };
    execute_compute_dag(view.compute_dag(USER), &mut cell, |c, target| {
        let route = DR.get(target as usize).copied().flatten().ok_or(target)?;
        c.roles = Some(route(c.compiled, c.user));
        Ok::<(), u8>(())
    })
    .expect("every declared target has a route");
    cell.roles
}

/// Roles as dir-sim answers them (arm A): the grants of the SIDs it reports.
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

fn check(w: &World) -> usize {
    let view = IamUser::new(&ROLES_FROM_MEMBERSHIP);
    let compiled = Compiled::compile(w.prepare());
    let ord = Ordinals::of(&w.f, &w.view());
    let d = oracle::run(&w.f, &w.direct, &w.nested, &templates(&w.f));
    let mut d_roles = vec![0u64; w.f.users.len()];
    for &(u, role) in &d.roles {
        d_roles[u as usize] |= 1 << role;
    }
    let mut holders = 0;
    for (i, ug) in w.f.users.iter().enumerate() {
        let got = run(&view, &compiled, ord.user[i]).expect("declared");
        assert_eq!(got, a_roles(w, ug), "dir-sim roles, user {i}");
        assert_eq!(got, d_roles[i], "oracle roles, user {i}");
        holders += usize::from(got != 0);
    }
    holders
}

#[test]
fn the_declared_behavior_equals_dir_sim_and_the_oracle_for_every_user() {
    let w = World::new(Fixture::generate(Spec::small()));
    let holders = check(&w);
    // Anti-vacuity: the comparison is meaningful only if roles are held by
    // some users and not by all.
    assert!(
        holders > 0 && holders < w.f.users.len(),
        "holders {holders}"
    );
}

#[test]
fn removing_one_support_path_keeps_the_role_the_other_path_carries() {
    let mut w = World::new(Fixture::generate(Spec::small()));
    check(&w);
    // Make a role doubly supported, then remove one support. Roles come only
    // through security groups, so both supports must be security groups that
    // grant the same role: a user directly in g1, and a second group g2
    // granting a shared role that the user is not yet in.
    let grants: HashMap<u32, u64> = w.f.grants.iter().fold(HashMap::new(), |mut m, &(g, r)| {
        *m.entry(g).or_default() |= 1 << r;
        m
    });
    let sec = |g: u32| w.f.security[g as usize] == Some(true);
    let g_of = |g: u32| grants.get(&g).copied().unwrap_or(0);
    let (u, g1, g2) = w
        .direct
        .iter()
        .filter(|&&(_, g1)| sec(g1) && g_of(g1) != 0)
        .find_map(|&(u, g1)| {
            (0..w.f.groups.len() as u32)
                .find(|&g2| {
                    g2 != g1 && sec(g2) && g_of(g1) & g_of(g2) != 0 && !w.direct.contains(&(u, g2))
                })
                .map(|g2| (u, g1, g2))
        })
        .expect("fixture has two security groups granting one role");
    w.apply(&[Edit::AddDirect(u, g2)]);
    check(&w);
    let shared = |w: &World| {
        let compiled = Compiled::compile(w.prepare());
        let ord = Ordinals::of(&w.f, &w.view());
        run(
            &IamUser::new(&ROLES_FROM_MEMBERSHIP),
            &compiled,
            ord.user[u as usize],
        )
        .unwrap()
    };
    let before = shared(&w);
    w.apply(&[Edit::RemoveDirect(u, g1)]);
    check(&w);
    let after = shared(&w);
    let both = g_of(g1) & g_of(g2);
    assert_eq!(before & both, both, "precondition: both supports grant it");
    assert_eq!(after & both, both, "the role survives losing one support");
}

#[test]
fn a_class_that_does_not_declare_the_behavior_computes_nothing() {
    let w = World::new(Fixture::generate(Spec::small()));
    let compiled = Compiled::compile(w.prepare());
    let ord = Ordinals::of(&w.f, &w.view());
    let silent = IamUser::new(&[]);
    for i in 0..w.f.users.len() {
        assert_eq!(run(&silent, &compiled, ord.user[i]), None);
    }
}
