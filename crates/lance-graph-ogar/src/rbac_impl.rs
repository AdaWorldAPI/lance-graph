//! `rbac_impl` — the OGAR active-record RBAC realization (keystone Q5), as a
//! **local newtype**.
//!
//! The keystone names `impl ClassRbac for OgarClassView` as Q5's active-record
//! RBAC. That exact form is an **orphan-rule violation** here: `OgarClassView` is
//! a foreign type (re-exported from the OGAR git crate) and `ClassRbac` is a
//! foreign trait (`lance_graph_contract::rbac`) — a third crate cannot
//! `impl ForeignTrait for ForeignType` (E0117). So the realization is a local
//! newtype [`OgarRbac`] that carries an **injected** [`GrantSource`].
//!
//! # The §6 evaporation seam
//!
//! [`OgarRbac`] owns **no grant data** — every answer is read from its
//! `GrantSource`. Today the source is a fixture (or a consumer-supplied table)
//! because the OGAR Core does not yet carry the `project_role.granted` value-tenant
//! (keystone §6 — a tracked core-gap). When §6 lands, the `GrantSource`
//! implementation flips from a fixture to a resolver over `project_role.granted`
//! + the membership `EdgeBlock`, and **`OgarRbac`'s body does not change** (the
//! "evaporation test"). That is what makes this an honest bridge rather than a
//! consumer-side re-implementation of the Core.

use lance_graph_contract::rbac::{
    grants_permit, ActorId, ClassGrant, ClassId, ClassRbac, Operation, RoleId,
};

/// The injected grant source — the §6 evaporation seam. A fixture implements it
/// today; the §6 `project_role.granted` + membership-`EdgeBlock` resolver
/// implements it later, with no change to [`OgarRbac`].
pub trait GrantSource {
    /// Roles the `actor` holds (membership → role fold).
    fn roles_of(&self, actor: ActorId<'_>) -> &[RoleId];
    /// The typed `granted` value-tenant of `role` — its `(target_classid, op_mask)` set.
    fn grants_of(&self, role: RoleId) -> &[ClassGrant];
}

/// The OGAR active-record [`ClassRbac`] (keystone Q5) as a local newtype over an
/// injected [`GrantSource`]. Carries no grant state of its own.
pub struct OgarRbac<S: GrantSource> {
    /// The injected grant source (fixture today; §6 tenant resolver later).
    pub source: S,
}

impl<S: GrantSource> OgarRbac<S> {
    /// Wrap a [`GrantSource`] as a [`ClassRbac`].
    pub const fn new(source: S) -> Self {
        Self { source }
    }
}

impl<S: GrantSource> ClassRbac for OgarRbac<S> {
    fn actor_roles(&self, actor: ActorId<'_>) -> &[RoleId] {
        self.source.roles_of(actor)
    }

    fn grant_permits(&self, role: RoleId, class: ClassId, op: &Operation<'_>) -> bool {
        grants_permit(self.source.grants_of(role), class, op)
    }
    // §4 axis-2/3/4 (roles_reaching / row_scope / field_mask) inherit the
    // contract defaults — column-level RBAC over the OGAR class surface is the
    // §6 follow-on, not this bridge.
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::property::PrefetchDepth;
    use lance_graph_contract::rbac::OpMask;

    /// A Vec-backed fixture `GrantSource` — the stand-in the §6 tenant resolver replaces.
    struct FixtureGrants {
        memberships: Vec<(&'static str, Vec<RoleId>)>,
        grants: Vec<(RoleId, Vec<ClassGrant>)>,
    }
    impl GrantSource for FixtureGrants {
        fn roles_of(&self, actor: ActorId<'_>) -> &[RoleId] {
            self.memberships
                .iter()
                .find(|(a, _)| *a == actor)
                .map_or(&[], |(_, r)| r.as_slice())
        }
        fn grants_of(&self, role: RoleId) -> &[ClassGrant] {
            self.grants
                .iter()
                .find(|(r, _)| *r == role)
                .map_or(&[], |(_, g)| g.as_slice())
        }
    }

    const ENCOUNTER: ClassId = 0x0000_0901;

    fn fixture() -> OgarRbac<FixtureGrants> {
        OgarRbac::new(FixtureGrants {
            memberships: vec![("dr-house", vec!["physician"]), ("betty", vec!["cashier"])],
            grants: vec![
                (
                    "physician",
                    vec![ClassGrant::new(0x0901, OpMask::READ.union(OpMask::ACT))],
                ),
                ("cashier", vec![ClassGrant::new(0x0901, OpMask::READ)]),
            ],
        })
    }

    #[test]
    fn actor_roles_resolve_via_source() {
        let r = fixture();
        assert_eq!(r.actor_roles("dr-house"), &["physician"]);
        assert_eq!(r.actor_roles("betty"), &["cashier"]);
        assert_eq!(r.actor_roles("nobody"), &[] as &[RoleId]);
    }

    #[test]
    fn physician_acts_cashier_cannot() {
        let r = fixture();
        let act = Operation::Act { action: "approve" };
        assert!(r.grant_permits("physician", ENCOUNTER, &act));
        assert!(!r.grant_permits("cashier", ENCOUNTER, &act));
        // both may read
        let read = Operation::Read {
            depth: PrefetchDepth::Identity,
        };
        assert!(r.grant_permits("physician", ENCOUNTER, &read));
        assert!(r.grant_permits("cashier", ENCOUNTER, &read));
    }

    /// Evaporation proof: `OgarRbac` satisfies `ClassRbac` for ANY `GrantSource`
    /// — the body is generic over the source, so the §6 resolver drops in unchanged.
    fn _is_class_rbac(_: &impl ClassRbac) {}
    #[test]
    fn ogar_rbac_is_class_rbac_over_any_source() {
        _is_class_rbac(&fixture());
    }
}
