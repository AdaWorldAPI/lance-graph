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
//! Roles are unscoped here; a consumer that binds a role to rows (a mailbox,
//! a practice) supplies its own `memberships_of`.

use std::collections::HashMap;

use lance_graph_contract::rbac::{ActorId, RoleId};
use lance_graph_contract::rbac_plug::{ActorSource, RbacBinding};
use ogar_dir_core::Guid128;

use crate::View;

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

    /// The roles of `group`, or `None` when its grants were never read.
    #[must_use]
    pub fn of(&self, group: &Guid128) -> Option<&[RoleId]> {
        self.read.get(group).map(Vec::as_slice)
    }
}

/// One principal's roles, read from a [`View`] once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewActors {
    subject: String,
    principal: Guid128,
    roles: Vec<RoleId>,
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
            subject: principal.to_string(),
            principal: *principal,
            roles,
            unread,
        }
    }

    /// The principal's GUID as text, the id decisions are asked under.
    #[must_use]
    pub fn subject(&self) -> &str {
        &self.subject
    }

    /// The principal's directory object.
    #[must_use]
    pub fn principal(&self) -> &Guid128 {
        &self.principal
    }

    /// The roles held, sorted.
    #[must_use]
    pub fn roles(&self) -> &[RoleId] {
        &self.roles
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
        if actor == self.subject {
            &self.roles
        } else {
            &[]
        }
    }
}
