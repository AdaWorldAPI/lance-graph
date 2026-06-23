//! `rbac` — the classid-keyed authorization trait surface (OGAR keystone §4/§11).
//!
//! The keystone §11 build order places the **`ClassRbac` grant-resolution trait**
//! in this zero-dep contract crate so that *both* the concrete kernel
//! (`lance-graph-rbac`, which holds `authorize()` + `Policy` + the `0x0B` auth
//! membrane) *and* the active-record `ClassView` producer (`lance-graph-ogar`'s
//! `OgarClassView`, which deps contract but **not** rbac) can implement / consume
//! one trait. Before this module the trait lived in `lance-graph-rbac`, so ogar —
//! which does not depend on rbac — could not satisfy the keystone's
//! `impl ClassRbac for OgarClassView` (Q5). This is that placement.
//!
//! Only the **trait + the `Operation` it ranges over** live here (pure types, no
//! runtime — `Operation` reads [`PrefetchDepth`](crate::property::PrefetchDepth),
//! already in this crate). The concrete `authorize()` kernel, `ClassGrants`,
//! `Policy`, `AccessDecision`, and the auth membrane stay in `lance-graph-rbac`;
//! it **re-exports** these so existing `lance_graph_rbac::authorize::ClassRbac` /
//! `lance_graph_rbac::policy::Operation` paths are unchanged (callcenter +
//! the sibling `smb-realtime` / `medcare-realtime` gates keep compiling).
//!
//! # Relationship to the rest of the contract auth surface
//!
//! - [`crate::auth::ActorContext`] is the *resolved actor identity* (actor id +
//!   tenant + roles). `lance-graph-rbac`'s `auth::ResolvedIdentity` (the `0x0B`
//!   membrane output) carries the same triple plus the resolving provider's
//!   classid; converging the two onto `ActorContext` is a tracked follow-on, not
//!   forced here.
//! - [`crate::external_membrane::MembraneGate`] is the *gate* a consumer impls to
//!   admit/deny an external commit; `ClassRbac` is the *grant resolution* a gate
//!   consults. They compose: a gate calls `authorize(rbac, actor, class, op)`.

use crate::property::PrefetchDepth;

/// The codebook class identity an authorization targets — the
/// [`NodeGuid`](crate::NodeGuid) `classid` (or its low-`u16` codebook id widened).
/// Opaque to the kernel: it is compared and looked up, never decoded (the kernel
/// "never touches a token" — only resolved keys go inward).
pub type ClassId = u32;

/// An actor identity. In the full keystone this is the OIDC `sub` resolved to a
/// membership-set ([`crate::auth::ActorContext`]); here it is the opaque key a
/// [`ClassRbac`] impl maps to roles.
pub type ActorId<'a> = &'a str;

/// A role identity (a minted role classid in the full keystone; a role *name*
/// where reconciling against a string-keyed policy).
pub type RoleId = &'static str;

/// What a caller wants to do on a class — the op the [`ClassRbac`] grant gate
/// ranges over. Read is depth-graded ([`PrefetchDepth`]); Write names a
/// predicate; Act names an action. (Promoted from `lance-graph-rbac`'s
/// `policy::Operation`, keystone §11; that path re-exports this type.)
#[derive(Clone, Debug)]
pub enum Operation<'a> {
    /// Read up to a prefetch depth.
    Read {
        /// The requested read depth (`Identity` < … < `Full`).
        depth: PrefetchDepth,
    },
    /// Write a specific predicate.
    Write {
        /// The predicate being written.
        predicate: &'a str,
    },
    /// Trigger a named action.
    Act {
        /// The action name.
        action: &'a str,
    },
}

/// The §4 grant-resolution surface, **classid-keyed**. The single trait both the
/// membrane gate and the cognitive loop resolve access through; the impl owns the
/// membership→role folding and the `(role, class)` grant table. `lance-graph-rbac`
/// supplies the reference impl (`ClassGrants`) + the `authorize()` kernel that
/// consumes it; `lance-graph-ogar`'s `OgarClassView` is the keystone's intended
/// active-record impl (Q5).
pub trait ClassRbac {
    /// Roles the actor holds, already folded through
    /// membership → member_role → role (the §4 `actor_roles`). Empty ⇒ the actor
    /// is unknown to the policy.
    fn actor_roles(&self, actor: ActorId<'_>) -> &[RoleId];

    /// Does `role` carry a grant on `class` that permits `op`? The positive
    /// `R⁺` op-mask gate (§5 stage 1). No grant, or a grant that does not permit
    /// the op, ⇒ `false` (restrictive default-deny).
    fn grant_permits(&self, role: RoleId, class: ClassId, op: &Operation<'_>) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_reads_prefetch_depth() {
        // Operation ranges over the contract's own PrefetchDepth — no rbac dep.
        let op = Operation::Read {
            depth: PrefetchDepth::Full,
        };
        assert!(matches!(op, Operation::Read { .. }));
    }

    // A trivial in-contract ClassRbac impl proves the trait is satisfiable with
    // contract-only types (the property ogar relies on: deps contract, not rbac).
    struct OneRole;
    impl ClassRbac for OneRole {
        fn actor_roles(&self, _actor: ActorId<'_>) -> &[RoleId] {
            const R: &[RoleId] = &["reader"];
            R
        }
        fn grant_permits(&self, role: RoleId, class: ClassId, op: &Operation<'_>) -> bool {
            role == "reader" && class == 0x0901 && matches!(op, Operation::Read { .. })
        }
    }

    #[test]
    fn trait_is_satisfiable_with_contract_only_types() {
        let rbac = OneRole;
        assert_eq!(rbac.actor_roles("anyone"), &["reader"]);
        assert!(rbac.grant_permits(
            "reader",
            0x0901,
            &Operation::Read {
                depth: PrefetchDepth::Identity
            }
        ));
        assert!(!rbac.grant_permits("reader", 0x0901, &Operation::Act { action: "x" }));
    }
}
