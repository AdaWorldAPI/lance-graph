//! Arm D: the independent oracle. crepe's generated Datalog evaluator over
//! the fixture's own integer ids. It never sees a dir-sim ordinal, a CSR or
//! a closure row, so it shares no code with the arms it checks.
//!
//! The rules state dir-sim's semantics directly:
//!
//! * `Member`: nesting follows every group.
//! * `SecMember` / `MailMember`: a chain through groups that have the
//!   property; the user's direct group and every parent on the way must
//!   have it (`groups_transitive_through`, `members_transitive_through`).
//! * `RoleHeld`: the roles granted to the groups whose SID the user holds.
//! * `Eligible`: a role x location question, the AND of two memberships.
//!   It is a question asked of the closure, not a membership.

#![allow(clippy::all)]

use crate::fixture::Fixture;
use crepe::crepe;
use std::collections::{BTreeSet, HashSet};

crepe! {
    @input
    struct Direct(u32, u32);
    @input
    struct Nested(u32, u32);
    @input
    struct Sec(u32);
    @input
    struct Mail(u32);
    @input
    struct Grant(u32, u32);
    @input
    struct Template(u32, u32);

    @output
    struct Member(u32, u32);
    @output
    struct SecMember(u32, u32);
    @output
    struct MailMember(u32, u32);
    @output
    struct RoleHeld(u32, u32);
    @output
    struct Eligible(u32, u32, u32);

    Member(u, g) <- Direct(u, g);
    Member(u, p) <- Member(u, c), Nested(c, p);

    SecMember(u, g) <- Direct(u, g), Sec(g);
    SecMember(u, p) <- SecMember(u, c), Nested(c, p), Sec(p);

    MailMember(u, g) <- Direct(u, g), Mail(g);
    MailMember(u, p) <- MailMember(u, c), Nested(c, p), Mail(p);

    RoleHeld(u, r) <- SecMember(u, g), Grant(g, r);

    Eligible(u, r, l) <- Template(r, l), Member(u, r), Member(u, l);
}

/// The oracle's answers, in fixture ids.
pub struct Answers {
    pub member: BTreeSet<(u32, u32)>,
    pub sec: BTreeSet<(u32, u32)>,
    pub mail: BTreeSet<(u32, u32)>,
    pub roles: BTreeSet<(u32, u32)>,
    pub eligible: BTreeSet<(u32, u32, u32)>,
}

/// Run the program. `direct` and `nested` are passed separately so an
/// overlay's edge lists can be checked too.
pub fn run(
    f: &Fixture,
    direct: &[(u32, u32)],
    nested: &[(u32, u32)],
    templates: &[(u32, u32)],
) -> Answers {
    let mut rt = Crepe::new();
    rt.extend(direct.iter().map(|&(u, g)| Direct(u, g)));
    rt.extend(nested.iter().map(|&(c, p)| Nested(c, p)));
    // dir-sim's `SecurityEnabled` is "known and true"; `MailEnabled` is
    // "has a primary SMTP address".
    rt.extend(
        (0..f.groups.len() as u32)
            .filter(|&g| f.security[g as usize] == Some(true))
            .map(Sec),
    );
    rt.extend((0..f.groups.len() as u32).filter(|&g| f.mail[g as usize]).map(Mail));
    rt.extend(f.grants.iter().map(|&(g, r)| Grant(g, u32::from(r))));
    rt.extend(templates.iter().map(|&(r, l)| Template(r, l)));
    let (member, sec, mail, roles, eligible): (
        HashSet<Member>,
        HashSet<SecMember>,
        HashSet<MailMember>,
        HashSet<RoleHeld>,
        HashSet<Eligible>,
    ) = rt.run();
    Answers {
        member: member.into_iter().map(|Member(u, g)| (u, g)).collect(),
        sec: sec.into_iter().map(|SecMember(u, g)| (u, g)).collect(),
        mail: mail.into_iter().map(|MailMember(u, g)| (u, g)).collect(),
        roles: roles.into_iter().map(|RoleHeld(u, r)| (u, r)).collect(),
        eligible: eligible
            .into_iter()
            .map(|Eligible(u, r, l)| (u, r, l))
            .collect(),
    }
}
