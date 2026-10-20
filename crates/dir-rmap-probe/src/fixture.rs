//! A synthetic IAM directory, generated deterministically from a seed.
//!
//! Every group is an ordinary dir-sim group. Role and location are groups,
//! and a role x location combination is a composite group nested in both, so
//! no new membership semantics is needed: inheritance is plain nesting. A
//! conjunction ("in role R and in location L") is a separate question the
//! probe asks of the closure, never a membership.
//!
//! The fixture keeps its own integer ids (`users[i]`, `groups[j]`) and its
//! own edge lists. The oracle reads those; the prepared arms translate them
//! through dir-sim's public ordinals. GUIDs are random, so the fixture order
//! and dir-sim's GUID-sorted ordinal order are unrelated.

use lance_graph_dir_sim::{Observation, ObservedNode};
use ogar_dir_core::{DirectoryScope, Guid128};

/// What kind of group a fixture group is. Only the generator and the
/// anti-vacuity checks read it; dir-sim never sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Every user, directly: the large-group fallback case.
    AllUsers,
    /// A division: locations are nested in it. Mail-enabled, not security.
    Division,
    /// The company: divisions are nested in it.
    Company,
    Role,
    Location,
    /// A role x location combination, nested in its role and its location.
    Composite,
    /// A resource group: roles and other resources are nested in it.
    Resource,
    License,
    /// A manual team: users added directly, mail-enabled distribution group.
    Team,
}

/// Sizes of one fixture.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub users: usize,
    pub divisions: usize,
    pub roles: usize,
    pub locations: usize,
    pub composites: usize,
    pub resources: usize,
    pub licenses: usize,
    pub teams: usize,
    /// Percent of users with 1 to 3 extra direct resource memberships.
    pub manual_pct: u64,
    /// Back edges between resources that close a cycle.
    pub cycles: usize,
    /// Distinct values of the probe's `u8` attribute lane.
    pub attr_values: u8,
    /// Distinct role ids granted to groups (at most 64).
    pub role_ids: u8,
    pub seed: u64,
}

impl Spec {
    /// About 8,000 users and 1,500 groups.
    pub fn iam(users: usize) -> Self {
        Spec {
            users,
            divisions: 7,
            roles: 300,
            locations: 40,
            composites: 600,
            resources: 500,
            licenses: 16,
            teams: 30,
            manual_pct: 10,
            cycles: 6,
            attr_values: 12,
            role_ids: 48,
            seed: 0xD1_5C0D,
        }
    }

    /// [`Spec::iam`] with the role, resource and composite counts multiplied
    /// by `factor`: the group-count sweep that pushes the B+ closure past L2.
    pub fn iam_scaled(users: usize, factor: usize) -> Self {
        let mut s = Spec::iam(users);
        s.roles *= factor;
        s.resources *= factor;
        s.composites *= factor;
        s
    }

    /// The same shape with no composite groups: every user holds its role
    /// and its location directly, and role x location exists only as a
    /// question (the policy-conjunction fixture).
    pub fn without_composites(mut self) -> Self {
        self.composites = 0;
        self
    }

    /// A few hundred users and groups: for the exhaustive equivalence tests.
    pub fn small() -> Self {
        Spec {
            users: 600,
            divisions: 3,
            roles: 40,
            locations: 8,
            composites: 60,
            resources: 50,
            licenses: 5,
            teams: 6,
            manual_pct: 15,
            cycles: 3,
            attr_values: 6,
            role_ids: 20,
            seed: 0x5EED_0001,
        }
    }
}

/// splitmix64.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn guid(&mut self) -> Guid128 {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&self.next_u64().to_le_bytes());
        b[8..].copy_from_slice(&self.next_u64().to_le_bytes());
        Guid128(b)
    }
}

/// One generated directory, in fixture ids.
pub struct Fixture {
    pub spec: Spec,
    pub users: Vec<Guid128>,
    pub groups: Vec<Guid128>,
    pub kind: Vec<Kind>,
    /// Has a primary SMTP address (dir-sim's `MailEnabled`).
    pub mail: Vec<bool>,
    /// `Some(true)` security-enabled, `Some(false)` not, `None` not read.
    pub security: Vec<Option<bool>>,
    /// `(user, group)`, direct.
    pub direct: Vec<(u32, u32)>,
    /// `(child group, parent group)`.
    pub nested: Vec<(u32, u32)>,
    /// `(group, role id)`.
    pub grants: Vec<(u32, u8)>,
    /// Groups of each kind, in fixture ids.
    pub roles: Vec<u32>,
    pub locations: Vec<u32>,
    pub teams: Vec<u32>,
    /// A `u8` attribute per user, in fixture user order (`< attr_values`).
    pub attr: Vec<u8>,
    pub scope: DirectoryScope,
}

impl Fixture {
    pub fn generate(spec: Spec) -> Self {
        let mut r = Rng::new(spec.seed);
        let mut f = Fixture {
            spec,
            users: (0..spec.users).map(|_| r.guid()).collect(),
            groups: Vec::new(),
            kind: Vec::new(),
            mail: Vec::new(),
            security: Vec::new(),
            direct: Vec::new(),
            nested: Vec::new(),
            grants: Vec::new(),
            roles: Vec::new(),
            locations: Vec::new(),
            teams: Vec::new(),
            attr: (0..spec.users)
                .map(|_| (r.next_u64() % u64::from(spec.attr_values)) as u8)
                .collect(),
            scope: DirectoryScope(Guid128([0x5C; 16])),
        };
        let add = |f: &mut Fixture, r: &mut Rng, k: Kind, mail: bool, sec: Option<bool>| {
            f.groups.push(r.guid());
            f.kind.push(k);
            f.mail.push(mail);
            f.security.push(sec);
            (f.groups.len() - 1) as u32
        };
        let all = add(&mut f, &mut r, Kind::AllUsers, true, Some(true));
        let company = add(&mut f, &mut r, Kind::Company, true, Some(true));
        let divisions: Vec<u32> = (0..spec.divisions)
            .map(|_| add(&mut f, &mut r, Kind::Division, true, Some(false)))
            .collect();
        for &d in &divisions {
            f.nested.push((d, company));
        }
        f.roles = (0..spec.roles)
            .map(|_| add(&mut f, &mut r, Kind::Role, false, Some(true)))
            .collect();
        // Locations are mail-enabled security groups, nested in a division:
        // the SID chain stops at the division, mail continues.
        f.locations = (0..spec.locations)
            .map(|_| add(&mut f, &mut r, Kind::Location, true, Some(true)))
            .collect();
        for (i, &l) in f.locations.clone().iter().enumerate() {
            f.nested.push((l, divisions[i % divisions.len()]));
        }
        let resources: Vec<u32> = (0..spec.resources)
            .map(|i| {
                // One in 20 resources: security flag not read.
                let sec = if i % 20 == 7 { None } else { Some(true) };
                add(&mut f, &mut r, Kind::Resource, i % 3 == 0, sec)
            })
            .collect();
        // A DAG over resources (child -> later parent), then back edges.
        for (i, &c) in resources.iter().enumerate() {
            if i + 1 < resources.len() && r.below(10) < 3 {
                let p = resources[i + 1 + r.below(resources.len() - i - 1)];
                f.nested.push((c, p));
            }
        }
        for k in 0..spec.cycles {
            let a = resources[(k * 7) % resources.len()];
            let b = resources[(k * 7 + 1) % resources.len()];
            f.nested.push((a, b));
            f.nested.push((b, a));
        }
        // Every role is nested in 1 to 3 resources.
        for &ro in &f.roles {
            for _ in 0..1 + r.below(3) {
                f.nested.push((ro, resources[r.below(resources.len())]));
            }
        }
        let licenses: Vec<u32> = (0..spec.licenses)
            .map(|_| add(&mut f, &mut r, Kind::License, false, Some(true)))
            .collect();
        f.teams = (0..spec.teams)
            .map(|i| {
                // Teams are distribution groups; one in 5 is also security.
                add(&mut f, &mut r, Kind::Team, true, Some(i % 5 == 0))
            })
            .collect();
        // Users: one role (skewed) and one location each. A composite group
        // exists for the first `composites` distinct pairs met.
        let mut composite = std::collections::HashMap::new();
        for u in 0..spec.users as u32 {
            let a = r.below(spec.roles);
            let b = r.below(spec.roles);
            let role = f.roles[a * b / spec.roles];
            let loc = f.locations[r.below(spec.locations)];
            let key = (role, loc);
            let c = match composite.get(&key) {
                Some(&c) => Some(c),
                None if composite.len() < spec.composites => {
                    let c = add(&mut f, &mut r, Kind::Composite, false, Some(true));
                    f.nested.push((c, role));
                    f.nested.push((c, loc));
                    composite.insert(key, c);
                    Some(c)
                }
                None => None,
            };
            match c {
                Some(c) => f.direct.push((u, c)),
                None => {
                    f.direct.push((u, role));
                    f.direct.push((u, loc));
                }
            }
            f.direct.push((u, all));
            if r.below(10) < 9 {
                f.direct.push((u, licenses[r.below(licenses.len())]));
            }
            if r.below(100) < spec.manual_pct as usize {
                for _ in 0..1 + r.below(3) {
                    f.direct.push((u, resources[r.below(resources.len())]));
                }
            }
        }
        // Teams: 20 to 250 direct users each, chosen at random.
        for &t in &f.teams.clone() {
            for _ in 0..20 + r.below(231) {
                f.direct.push((r.below(spec.users) as u32, t));
            }
        }
        f.direct.sort_unstable();
        f.direct.dedup();
        f.nested.sort_unstable();
        f.nested.dedup();
        // Grants: on half the resources, a fifth of the roles, and on some
        // teams (which only count when the team is a security group).
        for (j, k) in f.kind.iter().enumerate() {
            let p = match k {
                Kind::Resource => 50,
                Kind::Role => 20,
                Kind::Team => 50,
                Kind::Location => 25,
                _ => 0,
            };
            if r.below(100) < p {
                f.grants
                    .push((j as u32, (r.next_u64() % u64::from(spec.role_ids)) as u8));
            }
        }
        f
    }

    /// The dir-sim observation of this fixture.
    pub fn observation(&self) -> Observation {
        let mut nodes = Vec::with_capacity(self.users.len() + self.groups.len());
        for (i, g) in self.users.iter().enumerate() {
            let a = format!("u{i}@probe.test");
            nodes.push((*g, ObservedNode::user(&a, &a)));
        }
        for (j, g) in self.groups.iter().enumerate() {
            let mut n = ObservedNode::group();
            if self.mail[j] {
                n.primary_smtp = Some(format!("g{j}@probe.test"));
            }
            n.security = self.security[j];
            nodes.push((*g, n));
        }
        let mut members = Vec::with_capacity(self.direct.len() + self.nested.len());
        members.extend(
            self.direct
                .iter()
                .map(|&(u, g)| (self.users[u as usize], self.groups[g as usize])),
        );
        members.extend(
            self.nested
                .iter()
                .map(|&(c, p)| (self.groups[c as usize], self.groups[p as usize])),
        );
        Observation {
            scope: self.scope,
            nodes,
            members,
        }
    }

    /// Groups with more than 256 direct members.
    pub fn large_groups(&self) -> usize {
        let mut n = vec![0usize; self.groups.len()];
        for &(_, g) in &self.direct {
            n[g as usize] += 1;
        }
        n.iter().filter(|&&c| c > 256).count()
    }
}
