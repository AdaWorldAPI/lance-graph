//! D-RMAP-DIR-1, lab only: compiled membership closure and a 256-slot group
//! focus over `lance-graph-dir-sim`.
//!
//! STATUS: PROBE. Plan: `.claude/plans/d-rmap-dir-1-membership-closure-focus-v1.md`.
//! Nothing here is a production path. dir-sim stays authoritative and is
//! read only through its public API; no production file is changed.
//!
//! Arms:
//!
//! | arm | module | what |
//! |---|---|---|
//! | A  | dir-sim `View` | the existing walks (authoritative) |
//! | D  | [`oracle`] | crepe Datalog over fixture ids (independent oracle) |
//! | B  | [`prep::Prepared`] | bidirectional membership CSR + nesting CSR |
//! | B+ | [`prep::Compiled`] | B + closure rows per group and chain + roles by ordinal |
//! | F  | [`focus`] | B+ projected onto a 256-slot focus of selected groups |

pub mod fixture;
pub mod focus;
pub mod oracle;
pub mod prep;

use fixture::{Fixture, Kind};
use lance_graph_dir_sim::{GroupProperty, GroupWhere, View};
use ogar_dir_core::Guid128;
use prep::Chain;

/// Arm A: a user's groups through `chain`, as dir-sim answers it.
pub fn a_groups(v: &View<'_>, u: &Guid128, chain: Chain) -> Vec<Guid128> {
    match chain {
        Chain::All => v.groups_transitive(u),
        Chain::Security => v.security_identifiers(u),
        Chain::Mail => v.groups_transitive_through(u, &GroupWhere::Is(GroupProperty::MailEnabled)),
    }
}

/// Arm A: a group's users through `chain`.
pub fn a_members(v: &View<'_>, g: &Guid128, chain: Chain) -> Vec<Guid128> {
    match chain {
        Chain::All => v.members_transitive(g),
        Chain::Security => {
            v.members_transitive_through(g, &GroupWhere::Is(GroupProperty::SecurityEnabled))
        }
        Chain::Mail => v.members_transitive_through(g, &GroupWhere::Is(GroupProperty::MailEnabled)),
    }
}

/// Role x location questions, in fixture ids: the `(role, location)` pairs
/// a composite group joins, in the order the composites were created.
pub fn templates(f: &Fixture) -> Vec<(u32, u32)> {
    let mut parents: std::collections::BTreeMap<u32, (Option<u32>, Option<u32>)> =
        Default::default();
    for &(c, p) in &f.nested {
        if f.kind[c as usize] != Kind::Composite {
            continue;
        }
        let e = parents.entry(c).or_default();
        match f.kind[p as usize] {
            Kind::Role => e.0 = Some(p),
            Kind::Location => e.1 = Some(p),
            _ => {}
        }
    }
    parents
        .values()
        .filter_map(|&(r, l)| Some((r?, l?)))
        .collect()
}

/// Policy conjunctions, in fixture ids: the distinct `(role, location)`
/// pairs users hold, directly or through a composite, in first-seen user
/// order. A conjunction is a question asked of the closure ("in R and in
/// L"); it is never stored as a membership. On a fixture without composites
/// this is the only form role x location takes.
pub fn conjunctions(f: &Fixture) -> Vec<(u32, u32)> {
    let mut parents: std::collections::HashMap<u32, (Option<u32>, Option<u32>)> =
        Default::default();
    for &(c, p) in &f.nested {
        if f.kind[c as usize] == Kind::Composite {
            let e = parents.entry(c).or_default();
            match f.kind[p as usize] {
                Kind::Role => e.0 = Some(p),
                Kind::Location => e.1 = Some(p),
                _ => {}
            }
        }
    }
    let mut held: Vec<(Option<u32>, Option<u32>)> = vec![(None, None); f.users.len()];
    for &(u, g) in &f.direct {
        let h = &mut held[u as usize];
        match f.kind[g as usize] {
            Kind::Role => h.0 = Some(g),
            Kind::Location => h.1 = Some(g),
            Kind::Composite => *h = parents[&g],
            _ => {}
        }
    }
    let mut seen = std::collections::HashSet::new();
    held.iter()
        .filter_map(|&(r, l)| Some((r?, l?)))
        .filter(|p| seen.insert(*p))
        .collect()
}

/// Materialized composite groups, in fixture ids: "is the user in this
/// composite" is one membership bit, a different question from the
/// conjunction of its role and location.
pub fn composite_groups(f: &Fixture) -> Vec<u32> {
    (0..f.groups.len() as u32)
        .filter(|&g| f.kind[g as usize] == Kind::Composite)
        .collect()
}

/// The longest prefix of `templates` whose groups fit in `k` slots, and
/// those groups (first-seen order).
pub fn focus_for(templates: &[(u32, u32)], k: usize) -> (Vec<(u32, u32)>, Vec<u32>) {
    let mut groups: Vec<u32> = Vec::new();
    let mut kept = Vec::new();
    for &(r, l) in templates {
        let new = [r, l].iter().filter(|g| !groups.contains(g)).count();
        if groups.len() + new > k {
            continue;
        }
        for g in [r, l] {
            if !groups.contains(&g) {
                groups.push(g);
            }
        }
        kept.push((r, l));
    }
    (kept, groups)
}

use lance_graph_dir_sim::{Rule, VersionStore};
use ogar_dir_sim::{Change, EvidenceRef, RuleId, VersionId};

/// A fixture observed into a dir-sim store, plus the edge lists of the
/// current version (the fixture's, then every applied change).
pub struct World {
    pub f: Fixture,
    pub st: VersionStore,
    pub version: VersionId,
    pub direct: Vec<(u32, u32)>,
    pub nested: Vec<(u32, u32)>,
}

impl World {
    pub fn new(f: Fixture) -> Self {
        let mut st = VersionStore::new();
        let version = st
            .observe("probe", 0, f.observation())
            .expect("fixture builds");
        World {
            direct: f.direct.clone(),
            nested: f.nested.clone(),
            f,
            st,
            version,
        }
    }

    pub fn view(&self) -> View<'_> {
        self.st.view(self.version).expect("version exists")
    }

    /// Apply a change list as a simulated child version, and the same edits
    /// to the edge lists the prepared arms and the oracle are built from.
    pub fn apply(&mut self, edits: &[Edit]) {
        let f = &self.f;
        let changes = edits
            .iter()
            .map(|e| match *e {
                Edit::AddDirect(u, g) => Change::AddMembership {
                    user: f.users[u as usize],
                    group: f.groups[g as usize],
                },
                Edit::RemoveDirect(u, g) => Change::RemoveMembership {
                    user: f.users[u as usize],
                    group: f.groups[g as usize],
                },
                Edit::AddNested(c, p) => Change::AddMembership {
                    user: f.groups[c as usize],
                    group: f.groups[p as usize],
                },
            })
            .collect();
        self.version = self
            .st
            .simulate(self.version, &Propose(changes), &[])
            .expect("change list applies");
        for e in edits {
            match *e {
                Edit::AddDirect(u, g) => self.direct.push((u, g)),
                Edit::RemoveDirect(u, g) => self.direct.retain(|&x| x != (u, g)),
                Edit::AddNested(c, p) => self.nested.push((c, p)),
            }
        }
        self.direct.sort_unstable();
        self.direct.dedup();
        self.nested.sort_unstable();
        self.nested.dedup();
    }

    pub fn witness(&self) -> prep::Witness {
        prep::Witness::of(&self.st, self.version)
    }

    /// Arm B for the current version.
    pub fn prepare(&self) -> prep::Prepared {
        let v = self.view();
        let ord = prep::Ordinals::of(&self.f, &v);
        prep::Prepared::build(
            &v,
            prep::Witness::of(&self.st, self.version),
            &ord,
            &self.direct,
            &self.nested,
            &self.f.grants,
        )
    }
}

/// One edit, in fixture ids.
#[derive(Clone, Copy, Debug)]
pub enum Edit {
    AddDirect(u32, u32),
    RemoveDirect(u32, u32),
    AddNested(u32, u32),
}

struct Propose(Vec<Change>);

impl Rule for Propose {
    fn id(&self) -> RuleId {
        RuleId {
            name: "DirRmapProbe",
            version: 1,
        }
    }
    fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        self.0.clone()
    }
}
