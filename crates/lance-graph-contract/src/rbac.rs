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

// ─────────────────────────────────────────────────────────────────────────────
// §6 — the typed `granted` value-tenant (first-class replacement for
// `project_role.permissions: text`).
// ─────────────────────────────────────────────────────────────────────────────

/// The verb bitmask of a class-grant — the §3 axis-1 "verb × class" gate, one
/// `u8`, palette-native (#511 `SoaMemberSpec`: a role's grants are low-tens, one
/// column). Shaped after Odoo `ir.model.access`'s `perm_{read,write,create,unlink}`.
///
/// This is the **coarse verb gate** (§5 stage 1). It answers "may this role
/// *read / write / act on* this class at all", not the finer "at what depth /
/// which predicate / which action name" — those are the field-projection (axis 4)
/// and row-scope (axis 3) refinements that layer *above* a passed verb gate. So
/// [`OpMask::permits`] maps [`Operation::Read`] → the `READ` bit regardless of
/// depth; a depth/predicate/action-name check is a separate, finer stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct OpMask(pub u8);

impl OpMask {
    /// May read the class (any depth).
    pub const READ: OpMask = OpMask(1 << 0);
    /// May write a predicate on the class.
    pub const WRITE: OpMask = OpMask(1 << 1);
    /// May create an instance (Odoo `perm_create`).
    pub const CREATE: OpMask = OpMask(1 << 2);
    /// May delete an instance (Odoo `perm_unlink`).
    pub const DELETE: OpMask = OpMask(1 << 3);
    /// May trigger a named action (the DO arm — `ActionDef` fire).
    pub const ACT: OpMask = OpMask(1 << 4);

    /// The empty mask — grants nothing (restrictive default-deny).
    pub const NONE: OpMask = OpMask(0);

    /// Union of two masks (grant composition; e.g. role-hierarchy fold).
    #[inline]
    #[must_use]
    pub const fn union(self, other: OpMask) -> OpMask {
        OpMask(self.0 | other.0)
    }

    /// Whether `self` carries every bit of `bits`.
    #[inline]
    #[must_use]
    pub const fn contains(self, bits: OpMask) -> bool {
        self.0 & bits.0 == bits.0
    }

    /// Whether this mask permits `op` — the verb gate. `Read` → `READ`,
    /// `Write` → `WRITE`, `Act` → `ACT` (depth / predicate / action-name are
    /// finer stages, not decided here).
    #[inline]
    #[must_use]
    pub fn permits(self, op: &Operation<'_>) -> bool {
        let bit = match op {
            Operation::Read { .. } => OpMask::READ,
            Operation::Write { .. } => OpMask::WRITE,
            Operation::Act { .. } => OpMask::ACT,
        };
        self.contains(bit)
    }
}

/// One typed class-grant tuple — `(target_classid: u16, op_mask: u8)`. The
/// first-class, palette-native replacement for the `project_role.permissions:
/// text` blob (keystone §6 / I-K0 registry axiom: "decisions key on `classid`,
/// not on text"). A role's `granted` value-tenant is a `&[ClassGrant]`.
///
/// `target_classid` is the **low `u16` codebook id** (the shared-concept half of
/// a [`NodeGuid`](crate::NodeGuid)'s `classid`) — the RBAC + ontology identity,
/// app-render-skin-independent (the hi `u16` chooses render, never grants).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct ClassGrant {
    /// The class this grant targets (low-`u16` codebook id).
    pub target_classid: u16,
    /// The verbs this grant permits on that class.
    pub op_mask: OpMask,
}

impl ClassGrant {
    /// Construct a grant.
    #[inline]
    #[must_use]
    pub const fn new(target_classid: u16, op_mask: OpMask) -> Self {
        Self {
            target_classid,
            op_mask,
        }
    }

    /// Whether this grant permits `op` on `class`. Matches on the **low `u16`**
    /// of `class` (the codebook id), so a grant authored against the shared
    /// concept applies regardless of which app's render-skin (hi `u16`) the
    /// `ClassId` carries.
    #[inline]
    #[must_use]
    pub fn permits(&self, class: ClassId, op: &Operation<'_>) -> bool {
        self.target_classid == (class as u16) && self.op_mask.permits(op)
    }
}

/// Does any grant in a role's `granted` set permit `op` on `class`? The slice
/// form of the §5 stage-1 positive op-gate — the body a typed [`ClassRbac`] impl
/// uses for `grant_permits` (restrictive default-deny: empty ⇒ `false`).
#[must_use]
pub fn grants_permit(granted: &[ClassGrant], class: ClassId, op: &Operation<'_>) -> bool {
    granted.iter().any(|g| g.permits(class, op))
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

    // ── §6 typed `granted` value-tenant ──

    const PATIENT: ClassId = 0x0000_0901;

    #[test]
    fn opmask_permits_the_matching_verb_only() {
        let rw = OpMask::READ.union(OpMask::WRITE);
        assert!(rw.permits(&Operation::Read {
            depth: PrefetchDepth::Full
        }));
        assert!(rw.permits(&Operation::Write { predicate: "x" }));
        assert!(!rw.permits(&Operation::Act { action: "approve" }));
        // contains is bit-subset
        assert!(rw.contains(OpMask::READ));
        assert!(!rw.contains(OpMask::ACT));
        assert_eq!(OpMask::NONE, OpMask::default());
    }

    #[test]
    fn class_grant_matches_on_low_u16_codebook_id() {
        let grant = ClassGrant::new(0x0901, OpMask::READ.union(OpMask::ACT));
        // Same concept, different app render-skin (hi u16) → still permitted:
        // the grant keys on the shared-concept low u16, never the render half.
        let app_a: ClassId = 0x0000_0901;
        let app_b: ClassId = 0xAB12_0901;
        let read = Operation::Read {
            depth: PrefetchDepth::Identity,
        };
        assert!(grant.permits(app_a, &read));
        assert!(grant.permits(app_b, &read));
        // Wrong concept → denied even with the verb.
        assert!(!grant.permits(0x0000_0902, &read));
        // Right concept, ungranted verb → denied.
        assert!(!grant.permits(app_a, &Operation::Write { predicate: "due" }));
    }

    /// A typed [`ClassRbac`] impl whose `grant_permits` body IS [`grants_permit`]
    /// over a role's `granted` value-tenant — the §6 shape end-to-end, proving the
    /// typed tenant replaces `permissions: text` with contract-only types.
    struct TypedRoleGrants {
        // physician → {READ+ACT on PATIENT}; cashier → {READ on PATIENT}
        physician: [ClassGrant; 1],
        cashier: [ClassGrant; 1],
    }
    impl ClassRbac for TypedRoleGrants {
        fn actor_roles(&self, actor: ActorId<'_>) -> &[RoleId] {
            match actor {
                "dr-house" => &["physician"],
                "betty" => &["cashier"],
                _ => &[],
            }
        }
        fn grant_permits(&self, role: RoleId, class: ClassId, op: &Operation<'_>) -> bool {
            let granted: &[ClassGrant] = match role {
                "physician" => &self.physician,
                "cashier" => &self.cashier,
                _ => &[],
            };
            grants_permit(granted, class, op)
        }
    }

    #[test]
    fn typed_granted_drives_grant_permits() {
        let rbac = TypedRoleGrants {
            physician: [ClassGrant::new(0x0901, OpMask::READ.union(OpMask::ACT))],
            cashier: [ClassGrant::new(0x0901, OpMask::READ)],
        };
        let act = Operation::Act { action: "approve" };
        // physician may act; cashier may not — restrictive default-deny.
        assert!(rbac.grant_permits("physician", PATIENT, &act));
        assert!(!rbac.grant_permits("cashier", PATIENT, &act));
        // both may read
        let read = Operation::Read {
            depth: PrefetchDepth::Identity,
        };
        assert!(rbac.grant_permits("physician", PATIENT, &read));
        assert!(rbac.grant_permits("cashier", PATIENT, &read));
    }
}
