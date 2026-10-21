//! Arm Q: a team focus, populations bound to one version, methods that fold
//! them, and repair after edits.
//!
//! * A class declares three methods (`compute_dag`); the DR table binds each
//!   to a relalg fold over populations lent as mask planes. The populations
//!   are built once, only in the focus's tiles, and every method reuses them.
//!   Each answer equals dir-sim's own walk over the focus's users.
//! * After membership and nesting edits, repaired populations equal a fresh
//!   build, including when a user or a group keeps a property through a
//!   second path while one path is removed.
//! * Populations, pointer tables and repairs refuse a mismatched version.

use dir_rmap_probe::fixture::{Fixture, Kind, Spec};
use dir_rmap_probe::focus::Row;
use dir_rmap_probe::pop::{Pop, TeamFocus};
use dir_rmap_probe::prep::{Chain, Ordinals, Prepared, Refusal};
use dir_rmap_probe::qr::{Qr, Runs};
use dir_rmap_probe::{Edit, World, a_groups};
use lance_graph_contract::class_view::{ClassId, ClassView, ComputeEdge, execute_compute_dag};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_mask_risc::Planes;
use lance_graph_relalg::{Answer, Measure, Query, Rel, run};
use ogar_dir_core::Guid128;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------- the class

const TEAM: ClassId = 2;
const MEMBER_OF: u8 = 0;
/// Per license: members holding the license and the role, outside the
/// location.
const LICENSED_ROLE_OUTSIDE: u8 = 1;
/// Members holding no license.
const UNLICENSED: u8 = 2;
/// Members holding the role.
const ROLE_HOLDERS: u8 = 3;

static METHODS: [ComputeEdge; 3] = [
    ComputeEdge {
        target: LICENSED_ROLE_OUTSIDE,
        inputs: &[MEMBER_OF],
    },
    ComputeEdge {
        target: UNLICENSED,
        inputs: &[MEMBER_OF],
    },
    ComputeEdge {
        target: ROLE_HOLDERS,
        inputs: &[MEMBER_OF],
    },
];

struct TeamAudit {
    fields: Vec<FieldRef>,
}

impl ClassView for TeamAudit {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == TEAM { &self.fields } else { &[] }
    }
    fn template(&self, _: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _: ClassId) -> u8 {
        0
    }
    fn compute_dag(&self, class: ClassId) -> &[ComputeEdge] {
        if class == TEAM { &METHODS } else { &[] }
    }
}

fn team_audit() -> TeamAudit {
    TeamAudit {
        fields: vec![
            FieldRef::new("urn:iam:team:memberOf", "member_of"),
            FieldRef::new("urn:iam:team:licensedRoleOutside", "licensed_role_outside"),
            FieldRef::new("urn:iam:team:unlicensed", "unlicensed"),
            FieldRef::new("urn:iam:team:roleHolders", "role_holders"),
        ],
    }
}

// ------------------------------------------------- the bound populations

/// Plane indexes in [`Bound::planes`].
const FOCUS: u16 = 0;
const ROLE: u16 = 1;
const LOC: u16 = 2;
const LIC0: u16 = 3;

/// The populations a focus binds, for one version, lent as mask planes.
struct Bound<'a> {
    n_rows: usize,
    planes: Vec<&'a [u64]>,
    licenses: usize,
}

impl Bound<'_> {
    fn count(&self, rel: Rel) -> u64 {
        let planes = Planes {
            n_rows: self.n_rows,
            masks: &self.planes,
            lanes: &[],
        };
        let q = Query {
            rel,
            measure: Measure::Count,
        };
        match run(&q, &planes, None)
            .expect("relalg folds the planes")
            .answer
        {
            Answer::Count(n) => n,
            a => panic!("not a count: {a:?}"),
        }
    }
}

fn licensed_role_outside(b: &Bound) -> Vec<u64> {
    (0..b.licenses as u16)
        .map(|i| {
            b.count(Rel::And(vec![
                Rel::Plane(FOCUS),
                Rel::Plane(LIC0 + i),
                Rel::Plane(ROLE),
                Rel::Not(Box::new(Rel::Plane(LOC))),
            ]))
        })
        .collect()
}

fn unlicensed(b: &Bound) -> Vec<u64> {
    let any = Rel::Or(
        (0..b.licenses as u16)
            .map(|i| Rel::Plane(LIC0 + i))
            .collect(),
    );
    vec![b.count(Rel::And(vec![Rel::Plane(FOCUS), Rel::Not(Box::new(any))]))]
}

fn role_holders(b: &Bound) -> Vec<u64> {
    vec![b.count(Rel::And(vec![Rel::Plane(FOCUS), Rel::Plane(ROLE)]))]
}

/// Direct routing: method position -> fold.
type Route = fn(&Bound) -> Vec<u64>;
const DR: [Option<Route>; 4] = [
    None,
    Some(licensed_role_outside),
    Some(unlicensed),
    Some(role_holders),
];

fn call(view: &dyn ClassView, bound: &Bound) -> HashMap<u8, Vec<u64>> {
    let mut out = HashMap::new();
    execute_compute_dag(view.compute_dag(TEAM), &mut out, |out, target| {
        let route = DR.get(target as usize).copied().flatten().ok_or(target)?;
        out.insert(target, route(bound));
        Ok::<(), u8>(())
    })
    .expect("every declared method has a route");
    out
}

// ------------------------------------------------------------- the world

struct Setup {
    w: World,
    ord: Ordinals,
    licenses: Vec<usize>,
    loc: usize,
    team: usize,
    role: usize,
}

fn setup() -> Setup {
    let mut spec = Spec::small();
    spec.users = 3000;
    spec.licenses = 16;
    let w = World::new(Fixture::generate(spec));
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let kinds = |k: Kind| -> Vec<usize> {
        (0..w.f.groups.len())
            .filter(|&j| w.f.kind[j] == k)
            .collect()
    };
    let direct_n = |j: usize| w.direct.iter().filter(|&&(_, g)| g as usize == j).count();
    let team = *kinds(Kind::Team)
        .iter()
        .filter(|&&j| w.f.security[j] == Some(true))
        .max_by_key(|&&j| direct_n(j))
        .expect("a security team");
    let loc = *kinds(Kind::Location)
        .iter()
        .max_by_key(|&&j| direct_n(j))
        .unwrap();
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).unwrap();
    let role = (0..64)
        .max_by_key(|&r| qr.population(&b, &qr.role_slots[r]).unwrap().count())
        .unwrap();
    let licenses = kinds(Kind::License);
    drop(v);
    Setup {
        w,
        ord,
        licenses,
        loc,
        team,
        role,
    }
}

/// Per fixture user: effective security groups and roles, from dir-sim.
fn truth(s: &Setup) -> Vec<(u16, HashSet<Guid128>, u64)> {
    let v = s.w.view();
    let grants: HashMap<Guid128, u64> =
        s.w.f.grants.iter().fold(HashMap::new(), |mut m, &(g, r)| {
            *m.entry(s.w.f.groups[g as usize]).or_default() |= 1 << r;
            m
        });
    s.w.f
        .users
        .iter()
        .enumerate()
        .map(|(i, ug)| {
            let eff: HashSet<Guid128> = a_groups(&v, ug, Chain::Security).into_iter().collect();
            let roles = eff
                .iter()
                .fold(0, |a, g| a | grants.get(g).copied().unwrap_or(0));
            (s.ord.user[i], eff, roles)
        })
        .collect()
}

/// The populations a focus binds: role, location, every license.
struct Pops {
    role: Pop,
    loc: Pop,
    lic: Vec<Pop>,
}

fn build(s: &Setup, qr: &Qr, b: &Prepared, focus: &TeamFocus, runs: &mut Runs) -> Pops {
    let reach = |j: usize| qr.slots_reaching(s.ord.group[j]).unwrap();
    Pops {
        role: qr
            .population_in(b, &qr.role_slots[s.role], focus, runs)
            .unwrap(),
        loc: qr.population_in(b, &reach(s.loc), focus, runs).unwrap(),
        lic: s
            .licenses
            .iter()
            .map(|&l| qr.population_in(b, &reach(l), focus, runs).unwrap())
            .collect(),
    }
}

fn bound<'a>(n_rows: usize, focus: &'a TeamFocus, p: &'a Pops) -> Bound<'a> {
    let mut planes = vec![
        focus.rows.plane(n_rows),
        p.role.plane(n_rows),
        p.loc.plane(n_rows),
    ];
    planes.extend(p.lic.iter().map(|l| l.plane(n_rows)));
    Bound {
        n_rows,
        planes,
        licenses: p.lic.len(),
    }
}

#[test]
fn methods_reuse_one_focus_and_its_populations_and_equal_dir_sim() {
    let s = setup();
    let b = s.w.prepare();
    let qr = Qr::bind(&b, &s.w.witness()).unwrap();
    let focus = qr.team_focus(&b, s.ord.group[s.team]).unwrap();
    let mut runs = Runs::new(&qr);
    let pops = build(&s, &qr, &b, &focus, &mut runs);
    let got = call(&team_audit(), &bound(b.users, &focus, &pops));

    let t = truth(&s);
    let g = |j: usize| s.w.f.groups[j];
    let in_focus: Vec<&(u16, HashSet<Guid128>, u64)> =
        t.iter().filter(|u| u.1.contains(&g(s.team))).collect();
    let holds_role = |u: &&(u16, HashSet<Guid128>, u64)| u.2 >> s.role & 1 == 1;
    let want_lic: Vec<u64> = s
        .licenses
        .iter()
        .map(|&l| {
            in_focus
                .iter()
                .filter(|u| u.1.contains(&g(l)) && holds_role(u) && !u.1.contains(&g(s.loc)))
                .count() as u64
        })
        .collect();
    let want_unlic = in_focus
        .iter()
        .filter(|u| s.licenses.iter().all(|&l| !u.1.contains(&g(l))))
        .count() as u64;
    let want_role = in_focus.iter().filter(|u| holds_role(u)).count() as u64;
    assert_eq!(got[&LICENSED_ROLE_OUTSIDE], want_lic);
    assert_eq!(got[&UNLICENSED], vec![want_unlic]);
    assert_eq!(got[&ROLE_HOLDERS], vec![want_role]);

    // Anti-vacuity: the focus is a strict subset in few tiles, the role
    // filter keeps some members and not all, licenses differ.
    assert_eq!(focus.rows.count(), in_focus.len() as u64);
    assert!(in_focus.len() > 10 && in_focus.len() < t.len() / 4);
    assert!(want_role > 0 && want_role < in_focus.len() as u64);
    assert!(
        want_lic.iter().collect::<HashSet<_>>().len() > 2,
        "{want_lic:?}"
    );
    // Team members are spread over user ordinals (GUID order), so this
    // team occupies every tile; the tile restriction is checked on a narrow
    // focus below.
    // Only the slots the populations name were indexed.
    assert!(runs.built > 0 && runs.built < qr.ids.len());
}

/// The slot sets of every tracked population at one version: the roles in
/// `roles` (fixed ids, so the order holds across versions), then the named
/// groups.
fn tracked(qr: &Qr, s: &Setup, roles: &[usize]) -> Vec<Row> {
    let mut v: Vec<Row> = roles.iter().map(|&r| qr.role_slots[r]).collect();
    for &j in s.licenses.iter().chain([&s.loc, &s.team]) {
        v.push(qr.slots_reaching(s.ord.group[j]).unwrap());
    }
    v
}

fn restrict(p: &Pop, tiles: &[u64]) -> Pop {
    let mut q = p.clone();
    q.kept = Some(tiles.to_vec());
    for (i, t) in q.tiles.iter_mut().enumerate() {
        if tiles[i >> 6] >> (i & 63) & 1 == 0 {
            *t = [0; 4];
        }
    }
    q
}

#[test]
fn populations_in_a_narrow_focus_fill_only_its_tiles() {
    let s = setup();
    let b = s.w.prepare();
    let qr = Qr::bind(&b, &s.w.witness()).unwrap();
    // The security group with the fewest tiles that still has members.
    let focus = qr
        .ids
        .iter()
        .map(|&g| qr.team_focus(&b, g).unwrap())
        .filter(|f| f.rows.count() >= 3)
        .min_by_key(TeamFocus::tiles_selected)
        .unwrap();
    assert!(
        focus.tiles_selected() < qr.pages(),
        "a focus narrower than the page"
    );
    let mut runs = Runs::new(&qr);
    let pops = build(&s, &qr, &b, &focus, &mut runs);
    let outside = |p: &Pop| {
        p.tiles
            .iter()
            .enumerate()
            .filter(|&(i, _)| focus.tiles[i >> 6] >> (i & 63) & 1 == 0)
            .all(|(_, t)| *t == [0; 4])
    };
    assert!(outside(&pops.role) && outside(&pops.loc) && pops.lic.iter().all(outside));
    // Inside the focus's tiles they equal the full population.
    let full = qr.population(&b, &qr.role_slots[s.role]).unwrap();
    assert_eq!(pops.role, restrict(&full, &focus.tiles));
    // Anti-vacuity: the role is held outside the focus's tiles, and inside.
    assert!(full.count() > pops.role.count() && pops.role.count() > 0);
}

#[test]
fn repaired_populations_equal_a_fresh_build_across_edit_batches() {
    let mut s = setup();
    let sec = |s: &Setup, g: u32| s.w.f.security[g as usize] == Some(true);
    let mut b = s.w.prepare();
    let mut qr = Qr::bind(&b, &s.w.witness()).unwrap();
    let focus_tiles = qr.team_focus(&b, s.ord.group[s.team]).unwrap().tiles;
    let roles: Vec<usize> = (0..64).filter(|&r| qr.role_slots[r] != [0; 4]).collect();
    let mut slots = tracked(&qr, &s, &roles);
    let mut full: Vec<Pop> = slots
        .iter()
        .map(|x| qr.population(&b, x).unwrap())
        .collect();
    let mut part: Vec<Pop> = full.iter().map(|p| restrict(p, &focus_tiles)).collect();
    let mut rng = dir_rmap_probe::fixture::Rng::new(0xFEED);
    let (mut removed_slots, mut rederived) = (0, 0);
    for round in 0..8 {
        let n_groups = s.w.f.groups.len();
        let mut edits = Vec::new();
        for _ in 0..3 {
            let (u, g) = s.w.direct[rng.below(s.w.direct.len())];
            edits.push(Edit::RemoveDirect(u, g));
        }
        let secs: Vec<u32> = (0..n_groups as u32).filter(|&g| sec(&s, g)).collect();
        for _ in 0..3 {
            edits.push(Edit::AddDirect(
                rng.below(s.w.f.users.len()) as u32,
                secs[rng.below(secs.len())],
            ));
        }
        let nested_sec: Vec<(u32, u32)> =
            s.w.nested
                .iter()
                .copied()
                .filter(|&(c, p)| sec(&s, c) && sec(&s, p))
                .collect();
        for _ in 0..2 {
            let e = nested_sec[rng.below(nested_sec.len())];
            edits.push(Edit::RemoveNested(e.0, e.1));
        }
        for _ in 0..2 {
            let (c, p) = (secs[rng.below(secs.len())], secs[rng.below(secs.len())]);
            if c != p {
                edits.push(Edit::AddNested(c, p));
            }
        }
        let touched: Vec<u16> = edits
            .iter()
            .filter_map(|e| match *e {
                Edit::AddDirect(u, _) | Edit::RemoveDirect(u, _) => Some(s.ord.user[u as usize]),
                _ => None,
            })
            .collect();
        s.w.apply(&edits);
        let b1 = s.w.prepare();
        let qr1 = Qr::bind(&b1, &s.w.witness()).unwrap();
        let slots1 = tracked(&qr1, &s, &roles);
        for i in 0..full.len() {
            let r = full[i]
                .repair(&qr, &qr1, &b1, (&slots[i], &slots1[i]), &touched, None)
                .unwrap();
            removed_slots += r.slots_removed;
            rederived += r.rederived;
            part[i]
                .repair(
                    &qr,
                    &qr1,
                    &b1,
                    (&slots[i], &slots1[i]),
                    &touched,
                    Some(&focus_tiles),
                )
                .unwrap();
            let fresh = qr1.population(&b1, &slots1[i]).unwrap();
            assert_eq!(full[i], fresh, "round {round}, population {i}");
            assert_eq!(
                part[i],
                restrict(&fresh, &focus_tiles),
                "round {round}, focus {i}"
            );
        }
        (b, qr, slots) = (b1, qr1, slots1);
    }
    let _ = &b;
    // Anti-vacuity: slots lost properties and users were re-derived.
    assert!(removed_slots > 0, "no slot ever lost a property");
    assert!(rederived > 0);
}

#[test]
fn a_second_path_keeps_the_property_when_one_path_is_removed() {
    let mut s = setup();
    let b = s.w.prepare();
    let qr = Qr::bind(&b, &s.w.witness()).unwrap();
    let r = s.role;
    let carries = |qr: &Qr, j: usize| {
        let slot = qr.slot(s.ord.group[j]).unwrap();
        qr.role_slots[r][slot >> 6] >> (slot & 63) & 1 == 1
    };
    let fixture_of = |ordinal: u16| s.ord.group.iter().position(|&o| o == ordinal).unwrap();
    // Two parents carrying role r, a third role group for a user-level
    // second path, and a child group with members that does not carry r.
    let carriers: Vec<usize> = qr
        .ids
        .iter()
        .map(|&g| fixture_of(g))
        .filter(|&j| carries(&qr, j))
        .collect();
    let (p1, p2, q) = (carriers[0], carriers[1], carriers[2]);
    let c = qr
        .ids
        .iter()
        .map(|&g| fixture_of(g))
        .filter(|&j| !carries(&qr, j) && s.w.f.kind[j] == Kind::Team)
        .max_by_key(|&j| s.w.direct.iter().filter(|&&(_, g)| g as usize == j).count())
        .expect("a security team without the role");
    let members: Vec<u32> =
        s.w.direct
            .iter()
            .filter(|&&(_, g)| g as usize == c)
            .map(|&(u, _)| u)
            .collect();
    let u0 = members[0];
    s.w.apply(&[
        Edit::AddNested(c as u32, p1 as u32),
        Edit::AddNested(c as u32, p2 as u32),
        Edit::AddDirect(u0, q as u32),
    ]);
    let b1 = s.w.prepare();
    let qr1 = Qr::bind(&b1, &s.w.witness()).unwrap();
    let mut pop = qr1.population(&b1, &qr1.role_slots[r]).unwrap();
    let before = pop.count();

    // Remove one of two parents: c still carries r.
    s.w.apply(&[Edit::RemoveNested(c as u32, p1 as u32)]);
    let b2 = s.w.prepare();
    let qr2 = Qr::bind(&b2, &s.w.witness()).unwrap();
    assert!(
        carries(&qr2, c),
        "second parent keeps the role on the group"
    );
    pop.repair(
        &qr1,
        &qr2,
        &b2,
        (&qr1.role_slots[r], &qr2.role_slots[r]),
        &[],
        None,
    )
    .unwrap();
    assert_eq!(pop, qr2.population(&b2, &qr2.role_slots[r]).unwrap());
    assert_eq!(pop.count(), before);

    // Remove the second parent: c loses r, u0 keeps it through q.
    s.w.apply(&[Edit::RemoveNested(c as u32, p2 as u32)]);
    let b3 = s.w.prepare();
    let qr3 = Qr::bind(&b3, &s.w.witness()).unwrap();
    assert!(!carries(&qr3, c));
    let rep = pop
        .repair(
            &qr2,
            &qr3,
            &b3,
            (&qr2.role_slots[r], &qr3.role_slots[r]),
            &[],
            None,
        )
        .unwrap();
    assert!(rep.slots_removed > 0 && rep.rederived > 0);
    let fresh = qr3.population(&b3, &qr3.role_slots[r]).unwrap();
    assert_eq!(pop, fresh);
    assert!(pop.count() < before, "some member of c lost the role");
    let bit =
        |p: &Pop, u: u16| p.tiles[u as usize / 256][(u as usize % 256) >> 6] >> (u % 64) & 1 == 1;
    assert!(
        bit(&pop, s.ord.user[u0 as usize]),
        "u0 keeps the role through its second path"
    );
    // And against dir-sim.
    let t = truth(&s);
    for (o, _, roles) in &t {
        assert_eq!(bit(&pop, *o), roles >> r & 1 == 1, "user ordinal {o}");
    }
}

#[test]
fn repair_refuses_a_tile_restriction_other_than_the_one_built() {
    // Codex review on #1666: a population restricted to some tiles, repaired
    // with a different mask, kept the old tiles and stamped the new version.
    let mut s = setup();
    let b0 = s.w.prepare();
    let qr0 = Qr::bind(&b0, &s.w.witness()).unwrap();
    let focus = qr0
        .ids
        .iter()
        .map(|&g| qr0.team_focus(&b0, g).unwrap())
        .filter(|f| f.rows.count() >= 3)
        .min_by_key(TeamFocus::tiles_selected)
        .unwrap();
    let x0 = qr0.role_slots[s.role];
    let mut pop = qr0
        .population_in(&b0, &x0, &focus, &mut Runs::new(&qr0))
        .unwrap();
    let before = pop.clone();
    // Another mask: every tile the focus does not select.
    let other: Vec<u64> = focus.tiles.iter().map(|w| !w).collect();
    let (u, g) = s.w.direct[0];
    s.w.apply(&[Edit::RemoveDirect(u, g)]);
    let b1 = s.w.prepare();
    let qr1 = Qr::bind(&b1, &s.w.witness()).unwrap();
    let x1 = qr1.role_slots[s.role];
    assert_eq!(
        pop.repair(&qr0, &qr1, &b1, (&x0, &x1), &[], Some(&other))
            .err(),
        Some(Refusal::OutOfFocus)
    );
    assert_eq!(
        pop, before,
        "a refused repair leaves the population as it was"
    );
    // Unrestricted repair of a restricted population is refused too.
    assert_eq!(
        pop.repair(&qr0, &qr1, &b1, (&x0, &x1), &[], None).err(),
        Some(Refusal::OutOfFocus)
    );
    // The restriction it was built with is accepted.
    pop.repair(&qr0, &qr1, &b1, (&x0, &x1), &[], Some(&focus.tiles))
        .unwrap();
}

#[test]
fn a_mismatched_version_is_refused() {
    let mut s = setup();
    let b0 = s.w.prepare();
    let qr0 = Qr::bind(&b0, &s.w.witness()).unwrap();
    let mut runs0 = Runs::new(&qr0);
    let focus0 = qr0.team_focus(&b0, s.ord.group[s.team]).unwrap();
    let p0 = qr0.population(&b0, &qr0.role_slots[s.role]).unwrap();
    let (u, g) = s.w.direct[0];
    s.w.apply(&[Edit::RemoveDirect(u, g)]);
    let b1 = s.w.prepare();
    let qr1 = Qr::bind(&b1, &s.w.witness()).unwrap();
    let p1 = qr1.population(&b1, &qr1.role_slots[s.role]).unwrap();
    assert_eq!(p0.clone().and(&p1).err(), Some(Refusal::Stale));
    assert_eq!(p0.count_and(&p1).err(), Some(Refusal::Stale));
    // Pointer tables and focus from the old version.
    let x = qr1.role_slots[s.role];
    assert_eq!(
        qr1.population_in(&b1, &x, &focus0, &mut Runs::new(&qr1))
            .err(),
        Some(Refusal::Stale)
    );
    let focus1 = qr1.team_focus(&b1, s.ord.group[s.team]).unwrap();
    assert_eq!(
        qr1.population_in(&b1, &x, &focus1, &mut runs0).err(),
        Some(Refusal::Stale)
    );
    // A repair whose "before" register is not the population's version.
    let mut q = p0.clone();
    assert_eq!(
        q.repair(&qr1, &qr1, &b1, (&x, &x), &[], None).err(),
        Some(Refusal::Stale)
    );
}
