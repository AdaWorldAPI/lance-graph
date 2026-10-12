//! The roles a principal holds, read from a [`View`]: the contract's
//! [`ActorSource`] over the directory.
//!
//! A role is granted to a group, never to an address: [`GroupRoles`] keys
//! the grants by the group's GUID. A principal holds the roles of every
//! security group whose SID it carries ([`View::security_identifiers`]),
//! directly or through nesting. Only a security group has a SID, so a
//! distribution group passes on nothing, and nesting reaches a role only
//! through a chain of security groups.
//!
//! Fails closed in three places:
//!
//! * a group whose grants were never read grants nothing. It is not the same
//!   as a group read with no grants, so [`ViewActors::unread_groups`] names
//!   it, and a caller that needs the full picture can refuse;
//! * a role the consumer's binding does not declare is dropped, so group
//!   data can never hand out a role the plug did not ask for;
//! * any actor other than the principal holds nothing.
//!
//! The grants are policy, not directory state: they live in the
//! authority's versioned [`GroupRolePolicy`](ogar_rbac::group_policy::GroupRolePolicy),
//! and [`GroupRoles::from_policy`] reads one version of it.
//!
//! Roles are unscoped here; a consumer that binds a role to rows (a mailbox,
//! a practice) supplies its own `memberships_of`.

use std::collections::HashMap;

use lance_graph_contract::rbac::{ActorId, RoleId};
use lance_graph_contract::rbac_plug::{ActorSource, RbacBinding, SubjectRoles};
use ogar_dir_core::Guid128;

use crate::{GroupProperty, GroupWhere, MembershipPath, View};

/// The roles granted to each group, keyed by the group's GUID.
///
/// A group is either read (with its roles, possibly none) or absent, which
/// means its grants were never read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroupRoles {
    read: HashMap<Guid128, Vec<RoleId>>,
}

impl GroupRoles {
    /// No group read yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the grants of `group` as read: `roles`, which may be empty.
    /// Reading a group again replaces what was recorded.
    pub fn read(&mut self, group: Guid128, roles: &[RoleId]) -> &mut Self {
        self.read.insert(group, roles.to_vec());
        self
    }

    /// The grants as one version of the authority's group-role policy
    /// states them. Every group the policy has named is read, with its
    /// standing roles; a group whose grants were all revoked is read with
    /// none. A group the policy never named stays unread, so it grants
    /// nothing and [`ViewActors::unread_groups`] names it.
    #[must_use]
    pub fn from_policy(state: &ogar_rbac::group_policy::GroupRoleState) -> Self {
        Self {
            read: state.groups().collect(),
        }
    }

    /// The roles of `group`, or `None` when its grants were never read.
    #[must_use]
    pub fn of(&self, group: &Guid128) -> Option<&[RoleId]> {
        self.read.get(group).map(Vec::as_slice)
    }
}

/// Why a principal holds a role in one view, or why it cannot be said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoleEvidence {
    /// Held. `grant` is the group whose read grants include the role, and
    /// `path` one chain of security groups (SIDs) from the principal to it.
    /// Other groups and other chains may grant it too: this is a witness,
    /// not the only reason.
    Held {
        /// The security group whose grants include the role.
        grant: Guid128,
        /// One SID chain from the principal to `grant`.
        path: MembershipPath,
    },
    /// Not held: no security group the principal carries grants it, and
    /// every one of those groups was read.
    NotHeld,
    /// Not held as far as was read, but some of the principal's security
    /// groups were never read and might grant it. Named so a caller can
    /// refuse rather than report a loss or an absence.
    Undetermined {
        /// The principal's security groups whose grants were never read.
        unread: Vec<Guid128>,
    },
    /// The consumer's binding does not declare the role, so no group data
    /// can confer it.
    NotDeclared,
}

impl RoleEvidence {
    /// Whether the role is held.
    #[must_use]
    pub fn is_held(&self) -> bool {
        matches!(self, Self::Held { .. })
    }
}

impl GroupRoles {
    /// Whether `principal` holds `role` in `view`, with a witness when it
    /// does: the same decision as [`ViewActors::for_principal`], asked for
    /// one role.
    ///
    /// The witness group is the granting group with the shortest SID chain,
    /// the lowest-ordinal one among equals, so one view always gives the
    /// same answer. The answer is about this view only: to know whether a
    /// change takes the role away, ask the view after the change
    /// ([`RoleImpact::between`]), never infer it from a witness's edge
    /// having gone.
    #[must_use]
    pub fn role_evidence(
        &self,
        view: &View<'_>,
        principal: &Guid128,
        role: RoleId,
        binding: &RbacBinding,
    ) -> RoleEvidence {
        if !binding.declared_grants().iter().any(|(r, _)| *r == role) {
            return RoleEvidence::NotDeclared;
        }
        let sid = GroupWhere::Is(GroupProperty::SecurityEnabled);
        let mut best: Option<(Guid128, MembershipPath)> = None;
        let mut unread = Vec::new();
        for group in view.security_identifiers(principal) {
            match self.of(&group) {
                Some(granted) if granted.contains(&role) => {
                    let Some(path) = view.membership_path_through(principal, &group, &sid) else {
                        continue;
                    };
                    if best
                        .as_ref()
                        .is_none_or(|(_, b)| path.groups().len() < b.groups().len())
                    {
                        best = Some((group, path));
                    }
                }
                Some(_) => {}
                None => unread.push(group),
            }
        }
        match (best, unread.is_empty()) {
            (Some((grant, path)), _) => RoleEvidence::Held { grant, path },
            (None, true) => RoleEvidence::NotHeld,
            (None, false) => RoleEvidence::Undetermined { unread },
        }
    }
}

/// A role before and after a change, each read from its own version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleImpact {
    /// The role in the version before the change.
    pub before: RoleEvidence,
    /// The role in the version after the change.
    pub after: RoleEvidence,
}

impl RoleImpact {
    /// `role` for `principal` in `before` and in `after`, each decided on
    /// its own view, so a role that survives along another chain stays held.
    #[must_use]
    pub fn between(
        before: &View<'_>,
        after: &View<'_>,
        principal: &Guid128,
        role: RoleId,
        grants: &GroupRoles,
        binding: &RbacBinding,
    ) -> Self {
        Self {
            before: grants.role_evidence(before, principal, role, binding),
            after: grants.role_evidence(after, principal, role, binding),
        }
    }

    /// Held before and known not held after. An undetermined side is not a
    /// loss.
    #[must_use]
    pub fn is_lost(&self) -> bool {
        self.before.is_held() && self.after == RoleEvidence::NotHeld
    }

    /// Known not held before and held after.
    #[must_use]
    pub fn is_gained(&self) -> bool {
        self.before == RoleEvidence::NotHeld && self.after.is_held()
    }
}

/// One principal's roles, read from a [`View`] once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewActors {
    held: SubjectRoles,
    principal: Guid128,
    unread: Vec<Guid128>,
}

impl ViewActors {
    /// The roles `principal` holds in `view`: the grants of every security
    /// group whose SID it carries, kept only where `binding` declares the
    /// role. Decisions are asked under the principal's GUID as text
    /// ([`subject`](Self::subject)).
    ///
    /// A principal that is not a user in `view` holds nothing.
    #[must_use]
    pub fn for_principal(
        view: &View<'_>,
        principal: &Guid128,
        grants: &GroupRoles,
        binding: &RbacBinding,
    ) -> Self {
        let declared: Vec<RoleId> = binding.declared_grants().iter().map(|(r, _)| *r).collect();
        let mut roles: Vec<RoleId> = Vec::new();
        let mut unread = Vec::new();
        for group in view.security_identifiers(principal) {
            match grants.of(&group) {
                Some(granted) => {
                    for role in granted {
                        if declared.contains(role) && !roles.contains(role) {
                            roles.push(*role);
                        }
                    }
                }
                None => unread.push(group),
            }
        }
        roles.sort_unstable();
        Self {
            held: SubjectRoles::new(principal.to_string(), roles),
            principal: *principal,
            unread,
        }
    }

    /// The principal's GUID as text, the id decisions are asked under.
    #[must_use]
    pub fn subject(&self) -> &str {
        self.held.subject()
    }

    /// The principal's directory object.
    #[must_use]
    pub fn principal(&self) -> &Guid128 {
        &self.principal
    }

    /// The roles held, sorted.
    #[must_use]
    pub fn roles(&self) -> &[RoleId] {
        self.held.roles()
    }

    /// The security groups of the principal whose grants were never read.
    /// They granted nothing; a caller that needs every grant can refuse when
    /// this is not empty.
    #[must_use]
    pub fn unread_groups(&self) -> &[Guid128] {
        &self.unread
    }
}

impl ActorSource for ViewActors {
    fn roles_of(&self, actor: ActorId<'_>) -> &[RoleId] {
        self.held.of(actor)
    }
}
