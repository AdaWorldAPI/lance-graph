//! `authorize` — the classid-keyed RBAC kernel (OGAR keystone §5) and its
//! falsification gate, `PROBE-OGAR-RBAC-AUTHORIZE` (keystone §10).
//!
//! # What this is
//!
//! The shipped membrane path is [`crate::policy::Policy::evaluate`] — a
//! **string-keyed** check (`role_name`, `entity_type`, `Operation`). The OGAR
//! `CLASSID-RBAC-KEYSTONE-SPEC.md` §5 specifies the canonical successor:
//! `authorize(rbac, actor, class: ClassId, op)` — **classid-keyed**, where the
//! entity is named by its codebook `ClassId` (the `NodeGuid.classid`), not a
//! string. The keystone §11 build order ends at step (4): a probe that proves
//! the classid-keyed kernel reproduces a reference system's decision
//! **bit-for-bit** before any consumer collapses onto it (step 5). Until that
//! probe is green the keystone is **CONJECTURE**.
//!
//! This module is steps (1)+(3)+(4) made concrete against the in-repo reference
//! (the shipped `Policy` — the "reconcile the shipped MembraneGate path with the
//! keystone" framing of `ISS-RBAC-AUTHORIZE-BY-CLASSID`):
//!
//! - [`ClassRbac`] — the §4 grant-resolution trait, classid-keyed.
//! - [`authorize`] — the §5 two-stage kernel (positive ∧ op-gate), collapsed to
//!   the shipped [`AccessDecision`] so the parity comparison is exact.
//! - [`ClassGrants`] — `PermissionSpec` **re-keyed by `ClassId`** (§11 "re-key
//!   `PermissionSpec` to `ClassId`"); the independent representation the probe
//!   tests.
//! - `tests::probe_ogar_rbac_authorize` — the gate. For a fixed corpus of
//!   `(actor, class, op)` it asserts `authorize(...) == Policy::evaluate(...)`,
//!   **deny-reason included**. A wrong keying or a wrong kernel branch fails it.
//!
//! # Scope of this probe (honest fence)
//!
//! The reference here is the **shipped in-repo gate**, which is positive
//! role→permission only (no row-scope predicate, no field projection in the
//! decision). So this probe certifies the §5 *positive ∧ op-gate* half and the
//! classid re-keying. The §5 stage-2 *row-scope* predicate and the projecting
//! `Allow { scope, mask }` return remain keystone work; the keystone's stronger
//! reference options (Odoo `ir.model.access ∧ ir.rule`, OpenFGA) exercise scope
//! and are the follow-on probes. This gate is necessary, not yet sufficient for
//! the full keystone — but it is the step-4 reconciliation the shipped path
//! needs, and it moves "classid keying reproduces the membrane" from CONJECTURE
//! to FINDING.

use crate::access::AccessDecision;
use crate::permission::PermissionSpec;
use crate::policy::Operation;
use lance_graph_contract::class_view::FieldMask;
use lance_graph_contract::rbac::ScopeSpec;

// `ClassId` / `ActorId` / `RoleId` / `ClassRbac` were promoted to
// `lance_graph_contract::rbac` (keystone §11) so `lance-graph-ogar`'s
// `OgarClassView` (deps contract, NOT rbac) can implement the trait. Re-exported
// here so the `lance_graph_rbac::authorize::{ClassRbac, ClassId, ActorId, RoleId}`
// paths are unchanged; `authorize()` + `ClassGrants` (the kernel + reference impl)
// stay in this crate.
pub use lance_graph_contract::rbac::{ActorId, ClassId, ClassRbac, RoleId};

/// The §5 kernel — positive intersection ∧ op-gate, collapsed to the shipped
/// [`AccessDecision`]. An actor is allowed iff it holds at least one role whose
/// grant on `class` permits `op`. Deny reasons mirror [`Policy::evaluate`]
/// exactly so the parity gate can compare bit-for-bit:
/// - no roles at all ⇒ `Deny { "unknown role" }`
/// - roles present, none permit ⇒ the op-specific reason.
///
/// [`Policy::evaluate`]: crate::policy::Policy::evaluate
#[must_use]
pub fn authorize(
    rbac: &impl ClassRbac,
    actor: ActorId<'_>,
    class: ClassId,
    op: Operation<'_>,
) -> AccessDecision {
    let roles = rbac.actor_roles(actor);
    if roles.is_empty() {
        // Mirrors the shipped gate: an actor with no resolvable role is
        // indistinguishable from an unknown role-name.
        return AccessDecision::Deny {
            reason: "unknown role",
        };
    }
    if roles.iter().any(|&r| rbac.grant_permits(r, class, &op)) {
        return AccessDecision::Allow;
    }
    // Positive set non-empty but no grant permits — the op-specific reason,
    // identical to `Policy::evaluate`'s per-arm deny.
    AccessDecision::Deny {
        reason: match op {
            Operation::Read { .. } => "insufficient read depth",
            Operation::Write { .. } => "predicate not writable",
            Operation::Act { .. } => "action not allowed",
        },
    }
}

/// `PermissionSpec` **re-keyed by `ClassId`** (keystone §11) plus the
/// actor→role membership folding. The independent, classid-keyed representation
/// the probe certifies against the shipped string-keyed `Policy`.
#[derive(Clone, Debug, Default)]
pub struct ClassGrants {
    /// `(role, class) → grant`. The shipped `Role` keys `PermissionSpec` by
    /// `entity_type: &str`; this keys the same grant primitive by `ClassId`.
    grants: Vec<(RoleId, ClassId, PermissionSpec)>,
    /// `actor → roles`. One actor may hold several roles (the §5 union); the
    /// probe assigns each actor exactly one named role to mirror the
    /// single-role-name shipped gate.
    memberships: Vec<(&'static str, Vec<RoleId>)>,
}

impl ClassGrants {
    /// Empty grant table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a `(role, class) → grant` row (the re-keyed `PermissionSpec`).
    #[must_use]
    pub fn with_grant(mut self, role: RoleId, class: ClassId, grant: PermissionSpec) -> Self {
        self.grants.push((role, class, grant));
        self
    }

    /// Assign an actor a set of roles (membership fold).
    #[must_use]
    pub fn with_actor(mut self, actor: &'static str, roles: Vec<RoleId>) -> Self {
        self.memberships.push((actor, roles));
        self
    }

    fn grant_for(&self, role: RoleId, class: ClassId) -> Option<&PermissionSpec> {
        self.grants
            .iter()
            .find(|(r, c, _)| *r == role && *c == class)
            .map(|(_, _, g)| g)
    }
}

impl ClassRbac for ClassGrants {
    fn actor_roles(&self, actor: ActorId<'_>) -> &[RoleId] {
        self.memberships
            .iter()
            .find(|(a, _)| *a == actor)
            .map(|(_, roles)| roles.as_slice())
            .unwrap_or(&[])
    }

    fn grant_permits(&self, role: RoleId, class: ClassId, op: &Operation<'_>) -> bool {
        let Some(g) = self.grant_for(role, class) else {
            return false;
        };
        match op {
            Operation::Read { depth } => g.can_read_at(*depth),
            Operation::Write { predicate } => g.can_write(predicate),
            Operation::Act { action } => g.can_act(action),
        }
    }
}

/// The §5 two-stage authorization result — the positive∧op-gate decision PLUS
/// the row-scope (axis-3) and field-projection (axis-4) a granted read carries.
/// `Allow` carries `scope`+`field_mask`; a non-`Allow` carries `None`/`FULL`
/// (scope is irrelevant when access is refused).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopedDecision {
    /// The stage-1 positive∧op-gate verdict (unchanged from [`authorize`]).
    pub decision: AccessDecision,
    /// Axis-3 row-scope — the restrictive-AND of every granting role's
    /// [`ScopeSpec`]. `None` ⇒ global (no row restriction).
    pub scope: Option<ScopeSpec>,
    /// Axis-4 field projection — the union of every granting role's
    /// [`FieldMask`]. `FieldMask::FULL` on a refused decision.
    pub field_mask: FieldMask,
}

/// §5 two-stage authorize: stage-1 is the unchanged positive∧op-gate
/// ([`authorize`]); stage-2 folds the **granting subset** (the roles that
/// actually permit `op` — the SAME predicate stage-1 uses, NOT `roles_reaching`)
/// into a restrictive-AND row-scope and a union field-mask.
///
/// A non-`Allow` stage-1 short-circuits (no scope/mask computed). `AccessDecision`
/// is unchanged — the projection lives only here, in [`ScopedDecision`].
#[must_use]
pub fn authorize_scoped(
    rbac: &impl ClassRbac,
    actor: ActorId<'_>,
    class: ClassId,
    op: Operation<'_>,
) -> ScopedDecision {
    let decision = authorize(rbac, actor, class, op.clone());
    // Deny OR Escalate (any non-Allow) → no projection.
    if !matches!(decision, AccessDecision::Allow) {
        return ScopedDecision {
            decision,
            scope: None,
            field_mask: FieldMask::FULL,
        };
    }
    // Stage 2 — fold over the granting subset (actor_roles ∧ grant_permits).
    let mut scope: Option<ScopeSpec> = None;
    let mut mask = FieldMask::EMPTY;
    for &r in rbac.actor_roles(actor) {
        if rbac.grant_permits(r, class, &op) {
            // restrictive-AND of row-scopes. A role with NO scope is global —
            // it must NOT narrow the fold, so we only intersect *concrete* `Some`
            // scopes and leave `None` (the global sentinel) untouched. Folding a
            // `None` in as `ScopeSpec::default()` would replace the "no restriction"
            // sentinel with a materialized empty-tenant scope and force every
            // consumer down the `Some` branch even when nothing restricts.
            if let Some(rs) = rbac.row_scope(r, class) {
                scope = Some(match scope {
                    None => rs,
                    Some(acc) => acc.intersect(rs),
                });
            }
            // union of field projections (a user sees any column any role permits).
            mask = mask.union(rbac.field_mask(r, class));
        }
    }
    ScopedDecision {
        decision,
        scope,
        field_mask: mask,
    }
}

#[cfg(test)]
mod scoped_tests {
    use super::*;
    use lance_graph_contract::rbac::{ClassGrant, OpMask};

    // Two roles, BOTH granting Act on the class, with DIFFERENT row_scope +
    // DIFFERENT field_mask — so the fold's restrictive-AND scope + union mask are
    // both exercised (the test FAILS if scope is OR'd or mask is intersected).
    struct DualGrantRbac;
    const CLS: ClassId = 0x0000_0901;
    impl ClassRbac for DualGrantRbac {
        fn actor_roles(&self, _actor: ActorId<'_>) -> &[RoleId] {
            const R: &[RoleId] = &["role_a", "role_b"];
            R
        }
        fn grant_permits(&self, role: RoleId, class: ClassId, op: &Operation<'_>) -> bool {
            (role == "role_a" || role == "role_b")
                && class == CLS
                && matches!(op, Operation::Act { .. })
        }
        fn row_scope(&self, role: RoleId, _class: ClassId) -> Option<ScopeSpec> {
            match role {
                "role_a" => Some(ScopeSpec {
                    tenant: Some(7),
                    predicate_key: 0,
                    deny: false,
                }),
                "role_b" => Some(ScopeSpec {
                    tenant: None,
                    predicate_key: 2,
                    deny: false,
                }),
                _ => None,
            }
        }
        fn field_mask(&self, role: RoleId, _class: ClassId) -> FieldMask {
            match role {
                "role_a" => FieldMask::from_positions(&[0, 1]),
                "role_b" => FieldMask::from_positions(&[1, 2]),
                _ => FieldMask::EMPTY,
            }
        }
    }

    #[test]
    fn scoped_allow_ands_scope_and_unions_mask() {
        let _ = ClassGrant::new(0, OpMask::ACT); // touch the imports
        let d = authorize_scoped(&DualGrantRbac, "u", CLS, Operation::Act { action: "x" });
        assert_eq!(d.decision, AccessDecision::Allow);
        // restrictive-AND of {tenant 7, pk 0} ∩ {tenant None, pk 2}
        let expected_scope = ScopeSpec {
            tenant: Some(7),
            predicate_key: 0,
            deny: false,
        }
        .intersect(ScopeSpec {
            tenant: None,
            predicate_key: 2,
            deny: false,
        });
        assert_eq!(d.scope, Some(expected_scope));
        assert_eq!(d.scope.unwrap().tenant, Some(7));
        assert_eq!(d.scope.unwrap().predicate_key, 2);
        // union of {0,1} ∪ {1,2} = {0,1,2}
        assert_eq!(
            d.field_mask,
            FieldMask::from_positions(&[0, 1]).union(FieldMask::from_positions(&[1, 2]))
        );
        assert!(d.field_mask.has(0) && d.field_mask.has(1) && d.field_mask.has(2));
    }

    // Zero roles → Deny short-circuits with no scope, FULL mask.
    struct NoRoles;
    impl ClassRbac for NoRoles {
        fn actor_roles(&self, _a: ActorId<'_>) -> &[RoleId] {
            &[]
        }
        fn grant_permits(&self, _r: RoleId, _c: ClassId, _o: &Operation<'_>) -> bool {
            false
        }
    }
    #[test]
    fn scoped_deny_yields_no_scope_full_mask() {
        let d = authorize_scoped(&NoRoles, "ghost", CLS, Operation::Act { action: "x" });
        assert!(matches!(d.decision, AccessDecision::Deny { .. }));
        assert_eq!(d.scope, None);
        assert_eq!(d.field_mask, FieldMask::FULL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{smb_policy, Policy};
    use lance_graph_contract::property::PrefetchDepth;

    // ── Probe-local classid allocation. The point of the probe is the *keying*
    // (string `entity_type` → `ClassId`), not which specific codebook slot; the
    // SMB entity types are app-local and not promoted into the OGAR codebook, so
    // these are probe-local ids. A real consumer substitutes the codebook id. ──
    const CID_CUSTOMER: ClassId = 0x0000_C001;
    const CID_INVOICE: ClassId = 0x0000_C002;
    const CID_TAXDECL: ClassId = 0x0000_C003;

    fn class_of(entity_type: &str) -> ClassId {
        match entity_type {
            "Customer" => CID_CUSTOMER,
            "Invoice" => CID_INVOICE,
            "TaxDeclaration" => CID_TAXDECL,
            other => panic!("probe corpus references unmapped entity {other}"),
        }
    }

    /// Build the classid-keyed grant table BY RE-KEYING the shipped policy: walk
    /// each role's `PermissionSpec`s and store them under `class_of(entity_type)`
    /// instead of the entity string. This guarantees the two representations
    /// carry the *same grant data*, so the probe isolates what we actually want
    /// to certify: that the classid **keying** + the [`authorize`] **kernel**
    /// reproduce the shipped `Policy::evaluate` structure (multi-role union,
    /// empty→"unknown role", op→reason). A bug in either fails the corpus.
    fn class_grants_from(policy: &Policy) -> ClassGrants {
        let mut g = ClassGrants::new();
        for role in &policy.roles {
            for perm in &role.permissions {
                g = g.with_grant(role.name, class_of(perm.entity_type), perm.clone());
            }
            // Each role becomes an actor of the same name holding exactly that
            // one role — mirroring the single-role-name shipped gate.
            g = g.with_actor(role.name, vec![role.name]);
        }
        g
    }

    /// `PROBE-OGAR-RBAC-AUTHORIZE` (keystone §10).
    ///
    /// Asserts the classid-keyed [`authorize`] reproduces the shipped
    /// string-keyed [`Policy::evaluate`] **bit-for-bit** (deny-reason included)
    /// over a fixed corpus spanning all three SMB roles, all three op kinds, the
    /// allow path, every distinct deny reason, the depth boundary, and the
    /// unknown-actor path. Green ⇒ the §5 positive ∧ op-gate kernel + the §11
    /// classid re-keying are FINDING (no longer CONJECTURE) for the shipped
    /// reference.
    #[test]
    fn probe_ogar_rbac_authorize() {
        let policy = smb_policy();
        let grants = class_grants_from(&policy);

        // (actor / role-name, entity_type, op) — chosen to hit every branch.
        let corpus: &[(&str, &str, Operation)] = &[
            // accountant: Detail on Customer (allow), Full on Customer (deny depth)
            (
                "accountant",
                "Customer",
                Operation::Read {
                    depth: PrefetchDepth::Detail,
                },
            ),
            (
                "accountant",
                "Customer",
                Operation::Read {
                    depth: PrefetchDepth::Full,
                },
            ),
            // accountant: write/act on Invoice (allow), unwritable predicate (deny)
            (
                "accountant",
                "Invoice",
                Operation::Write {
                    predicate: "status",
                },
            ),
            (
                "accountant",
                "Invoice",
                Operation::Act { action: "approve" },
            ),
            (
                "accountant",
                "Invoice",
                Operation::Write {
                    predicate: "due_date",
                },
            ),
            ("accountant", "Invoice", Operation::Act { action: "delete" }),
            // accountant: no grant on Customer write/act → op-specific deny
            (
                "accountant",
                "Customer",
                Operation::Write {
                    predicate: "customer_name",
                },
            ),
            // auditor: Full read everywhere (allow), but write/act deny
            (
                "auditor",
                "Invoice",
                Operation::Read {
                    depth: PrefetchDepth::Full,
                },
            ),
            (
                "auditor",
                "Invoice",
                Operation::Write {
                    predicate: "status",
                },
            ),
            ("auditor", "Invoice", Operation::Act { action: "approve" }),
            // admin: full power
            ("admin", "Customer", Operation::Act { action: "delete" }),
            (
                "admin",
                "TaxDeclaration",
                Operation::Act { action: "submit" },
            ),
            (
                "admin",
                "Customer",
                Operation::Write {
                    predicate: "customer_name",
                },
            ),
            // unknown actor → "unknown role"
            (
                "ghost",
                "Customer",
                Operation::Read {
                    depth: PrefetchDepth::Identity,
                },
            ),
        ];

        for (actor, entity, op) in corpus {
            let shipped = policy.evaluate(actor, entity, op.clone());
            let keyed = authorize(&grants, actor, class_of(entity), op.clone());
            assert_eq!(
                keyed, shipped,
                "classid-keyed authorize diverged from shipped Policy::evaluate \
                 for actor={actor:?} entity={entity:?} op={op:?}: \
                 keyed={keyed:?} shipped={shipped:?}",
            );
        }
    }

    /// Falsification self-check: the gate is only meaningful if a *wrong* keying
    /// actually fails the comparison. Mapping every entity to one wrong class
    /// must make at least one corpus tuple diverge — proving the probe is not
    /// vacuous (it would pass trivially if `authorize` ignored the class).
    #[test]
    fn probe_is_falsifiable_under_wrong_keying() {
        let policy = smb_policy();
        let grants = class_grants_from(&policy);
        // Send an allow tuple to the WRONG class: accountant approve Invoice,
        // but ask under the Customer classid (accountant has no act grant there).
        let shipped = policy.evaluate(
            "accountant",
            "Invoice",
            Operation::Act { action: "approve" },
        );
        let miskeyed = authorize(
            &grants,
            "accountant",
            CID_CUSTOMER, // wrong class
            Operation::Act { action: "approve" },
        );
        assert_eq!(shipped, AccessDecision::Allow);
        assert_ne!(
            miskeyed, shipped,
            "a wrong classid must change the decision — else the probe is vacuous",
        );
    }
}
