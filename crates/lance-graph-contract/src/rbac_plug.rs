//! RBAC hot-plug — the [`hotplug`](crate::hotplug) pattern applied to
//! authorization.
//!
//! Three roles, three homes, exactly as for capabilities:
//!
//! 1. **This module (the socket, zero-dep):** a consumer declares one
//!    [`RbacPlug`] const naming the classids it gates and the roles its policy
//!    uses; the [`RbacAuthority`] trait is what an authority implements.
//! 2. **OGAR (the authority):** resolves the plug against its grant data,
//!    checks that every classid is minted and every role is defined, and hands
//!    back an [`RbacBinding`] for exactly the plugged ids and roles.
//! 3. **The consumer:** binds once in its own binary and tests, then pairs the
//!    binding with its actor source ([`RbacBinding::with_actors`]) to get a
//!    [`ClassRbac`] the `lance-graph-rbac` kernels accept. Drift bangs at
//!    bind time; there is no hand-copied policy table to fall out of sync.
//!
//! # Fails closed by mechanism
//!
//! An [`RbacBinding`] has private fields and no `Default`: a binding is
//! something an authority resolved, never an empty value a caller can conjure.
//! Every lookup returns a `Result`, so asking about a class or role outside the
//! plug is a named [`RbacDrift`], not an empty grant list that reads like "no
//! restriction". Through the [`ClassRbac`] view, an out-of-plug question is a
//! denial.
//!
//! Roles are named by [`RoleId`] strings, matching every existing grant table.

use crate::class_view::{FieldMask, WideFieldMask};
use crate::ogar_codebook::classid_canon_compat;
use crate::rbac::{
    grants_permit, ActorId, ClassGrant, ClassId, ClassRbac, Membership, Operation, RoleId,
};

/// A consumer's RBAC declaration: the classids it gates and the roles its
/// policy uses. One `const` per consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RbacPlug {
    /// Consumer name (crate name by convention).
    pub consumer: &'static str,
    /// The canon-high concept ids whose access this consumer decides.
    pub classids: &'static [u16],
    /// The role names the consumer's actors may hold.
    pub roles: &'static [RoleId],
}

/// Why a bind or a lookup failed. Each arm is one named refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RbacDrift {
    /// A plugged classid is not minted in the authority's vocabulary.
    UnknownClassid(u16),
    /// A plugged role is not defined by the authority.
    UnknownRole(String),
    /// A lookup named a concept the plug did not declare.
    NotPlugged(u16),
    /// A lookup named a role the plug did not declare.
    RoleNotPlugged(String),
    /// The authority resolved a concept the zero-dep wire mirror
    /// ([`crate::ogar_codebook`]) does not carry at the same id.
    MirrorDrift {
        /// Concept name as the authority resolved it.
        concept: String,
        /// The id the authority is authoritative for.
        authority_id: u16,
        /// What the mirror said; `None` when the concept is missing.
        mirror_id: Option<u16>,
    },
}

impl core::fmt::Display for RbacDrift {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownClassid(id) => write!(f, "plugged classid 0x{id:04X} is not minted"),
            Self::UnknownRole(r) => write!(f, "plugged role `{r}` is not defined by the authority"),
            Self::NotPlugged(id) => write!(f, "concept 0x{id:04X} is not in this RBAC plug"),
            Self::RoleNotPlugged(r) => write!(f, "role `{r}` is not in this RBAC plug"),
            Self::MirrorDrift {
                concept,
                authority_id,
                mirror_id,
            } => match mirror_id {
                Some(m) => write!(
                    f,
                    "wire mirror has `{concept}`=0x{m:04X} but the authority says 0x{authority_id:04X}"
                ),
                None => write!(
                    f,
                    "wire mirror is missing `{concept}` (authority: 0x{authority_id:04X})"
                ),
            },
        }
    }
}

impl std::error::Error for RbacDrift {}

/// Check an authority's resolved `(concept, id)` rows against the wire mirror.
/// The RBAC twin of [`crate::hotplug::verify_against_mirror`].
///
/// # Errors
///
/// [`RbacDrift::MirrorDrift`] for the first row the mirror disagrees with.
pub fn verify_concepts_against_mirror(concepts: &[(String, u16)]) -> Result<(), RbacDrift> {
    match crate::hotplug::verify_against_mirror(concepts) {
        None => Ok(()),
        Some(crate::hotplug::ActivationDrift::MirrorDrift {
            concept,
            authority_id,
            mirror_id,
        }) => Err(RbacDrift::MirrorDrift {
            concept,
            authority_id,
            mirror_id,
        }),
        // `verify_against_mirror` only ever reports `MirrorDrift`; anything
        // else would be a contract change there, refused here rather than
        // read as agreement.
        Some(other) => Err(RbacDrift::MirrorDrift {
            concept: other.to_string(),
            authority_id: 0,
            mirror_id: None,
        }),
    }
}

/// What an authority returns for a green bind: the grants and field masks of
/// exactly the plugged roles on exactly the plugged classids.
///
/// **No `Default`, private fields.** See the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RbacBinding {
    consumer: String,
    classids: Vec<u16>,
    grants: Vec<(RoleId, Vec<ClassGrant>)>,
    field_masks: Vec<(RoleId, u16, WideFieldMask)>,
}

impl RbacBinding {
    /// Authority-side constructor.
    ///
    /// `grants` holds one row per plugged role. Grants and field masks on a
    /// concept outside `classids` are dropped here, so a binding can never
    /// carry access the plug did not ask about.
    #[must_use]
    pub fn new(
        consumer: impl Into<String>,
        classids: Vec<u16>,
        grants: Vec<(RoleId, Vec<ClassGrant>)>,
        field_masks: Vec<(RoleId, u16, WideFieldMask)>,
    ) -> Self {
        let grants = grants
            .into_iter()
            .map(|(role, gs)| {
                let kept = gs
                    .into_iter()
                    .filter(|g| classids.contains(&g.target_classid))
                    .collect();
                (role, kept)
            })
            .collect();
        let field_masks = field_masks
            .into_iter()
            .filter(|(_, c, _)| classids.contains(c))
            .collect();
        Self {
            consumer: consumer.into(),
            classids,
            grants,
            field_masks,
        }
    }

    /// The consumer this binding was resolved for.
    #[must_use]
    pub fn consumer(&self) -> &str {
        &self.consumer
    }

    /// The concept of `class` if the plug declared it.
    ///
    /// # Errors
    ///
    /// [`RbacDrift::NotPlugged`] when it did not.
    pub fn plugged(&self, class: ClassId) -> Result<u16, RbacDrift> {
        let concept = classid_canon_compat(class);
        if self.classids.contains(&concept) {
            Ok(concept)
        } else {
            Err(RbacDrift::NotPlugged(concept))
        }
    }

    /// The grants `role` holds within the plug.
    ///
    /// # Errors
    ///
    /// [`RbacDrift::RoleNotPlugged`] when the plug did not declare `role`.
    pub fn grants_for(&self, role: RoleId) -> Result<&[ClassGrant], RbacDrift> {
        self.grants
            .iter()
            .find_map(|(r, gs)| (*r == role).then_some(gs.as_slice()))
            .ok_or_else(|| RbacDrift::RoleNotPlugged(role.to_string()))
    }

    /// Whether `role` may perform `op` on `class`.
    ///
    /// # Errors
    ///
    /// [`RbacDrift::NotPlugged`] or [`RbacDrift::RoleNotPlugged`] when the
    /// question lies outside the plug.
    pub fn permits(
        &self,
        role: RoleId,
        class: ClassId,
        op: &Operation<'_>,
    ) -> Result<bool, RbacDrift> {
        self.plugged(class)?;
        Ok(grants_permit(self.grants_for(role)?, class, op))
    }

    /// The field mask declared for `role` on `class`, or `None` when the
    /// authority declared none (the class is then not column-restricted for
    /// that role, the same as [`ClassRbac::field_mask`]'s default).
    ///
    /// # Errors
    ///
    /// As [`permits`](RbacBinding::permits).
    pub fn field_mask_for(
        &self,
        role: RoleId,
        class: ClassId,
    ) -> Result<Option<&WideFieldMask>, RbacDrift> {
        let concept = self.plugged(class)?;
        self.grants_for(role)?;
        Ok(self
            .field_masks
            .iter()
            .find_map(|(r, c, m)| (*r == role && *c == concept).then_some(m)))
    }

    /// Every declared `(role, grants)` row — the audit surface for tests and
    /// conformance checks. Resolve a single question through
    /// [`permits`](RbacBinding::permits).
    #[must_use]
    pub fn declared_grants(&self) -> &[(RoleId, Vec<ClassGrant>)] {
        &self.grants
    }

    /// Pair the binding with the source of actors and their memberships,
    /// giving a [`ClassRbac`] the `lance-graph-rbac` kernels accept.
    #[must_use]
    pub fn with_actors<A: ActorSource>(&self, actors: A) -> PluggedRbac<'_, A> {
        PluggedRbac {
            binding: self,
            actors,
        }
    }
}

/// Where a consumer's actors and their memberships come from — its identity
/// layer (a user store, a session). The grants come from the binding.
pub trait ActorSource {
    /// The roles `actor` holds.
    fn roles_of(&self, actor: ActorId<'_>) -> &[RoleId];

    /// The actor's memberships relevant to `class`: each role held, with the
    /// scope that holding is bound to. Defaults to one unscoped membership per
    /// role from [`roles_of`](ActorSource::roles_of).
    fn memberships_of(&self, actor: ActorId<'_>, _class: ClassId) -> Vec<Membership> {
        self.roles_of(actor)
            .iter()
            .map(|&role| Membership { role, scope: None })
            .collect()
    }
}

/// A binding paired with an [`ActorSource`]: the [`ClassRbac`] view.
///
/// Scopes come from the actor's memberships, so decide with
/// `lance_graph_rbac::authorize::authorize_memberships`; `authorize_scoped`
/// reads `row_scope`, which this view leaves at its unscoped default.
#[derive(Debug)]
pub struct PluggedRbac<'b, A> {
    binding: &'b RbacBinding,
    actors: A,
}

impl<A: ActorSource> ClassRbac for PluggedRbac<'_, A> {
    fn actor_roles(&self, actor: ActorId<'_>) -> &[RoleId] {
        self.actors.roles_of(actor)
    }

    /// A question outside the plug is a denial, never an allow.
    fn grant_permits(&self, role: RoleId, class: ClassId, op: &Operation<'_>) -> bool {
        self.binding.permits(role, class, op).unwrap_or(false)
    }

    fn memberships(&self, actor: ActorId<'_>, class: ClassId) -> Vec<Membership> {
        self.actors.memberships_of(actor, class)
    }

    fn field_mask(&self, role: RoleId, class: ClassId) -> WideFieldMask {
        match self.binding.field_mask_for(role, class) {
            Ok(Some(mask)) => mask.clone(),
            Ok(None) => WideFieldMask::from(FieldMask::FULL),
            // Outside the plug: no grant can permit, so the mask is never
            // used for a decision; it is still kept empty, not full.
            Err(_) => WideFieldMask::EMPTY,
        }
    }
}

/// Implemented by the authority: resolve an [`RbacPlug`] to its
/// [`RbacBinding`] or the first [`RbacDrift`].
pub trait RbacAuthority {
    /// Verify the plug and hand back the grants for exactly its classids and
    /// roles.
    ///
    /// # Errors
    ///
    /// [`RbacDrift::UnknownClassid`], [`RbacDrift::UnknownRole`] or
    /// [`RbacDrift::MirrorDrift`].
    fn bind(&self, plug: &RbacPlug) -> Result<RbacBinding, RbacDrift>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ogar_codebook::compose_classid;
    use crate::property::PrefetchDepth;
    use crate::rbac::OpMask;

    const PATIENT: u16 = 0x0901;
    const DIAGNOSIS: u16 = 0x0902;
    const LAB: u16 = 0x0903;

    fn class(concept: u16) -> ClassId {
        compose_classid(concept, 0)
    }

    fn read() -> Operation<'static> {
        Operation::Read {
            depth: PrefetchDepth::Full,
        }
    }

    /// Defines `physician` (read+write patient, read diagnosis and lab) and
    /// `nurse` (read patient); refuses anything else.
    struct TinyAuthority;
    impl RbacAuthority for TinyAuthority {
        fn bind(&self, plug: &RbacPlug) -> Result<RbacBinding, RbacDrift> {
            for &id in plug.classids {
                if ![PATIENT, DIAGNOSIS, LAB].contains(&id) {
                    return Err(RbacDrift::UnknownClassid(id));
                }
            }
            let mut grants = Vec::new();
            for &role in plug.roles {
                let gs = match role {
                    "physician" => vec![
                        ClassGrant::new(PATIENT, OpMask::READ.union(OpMask::WRITE)),
                        ClassGrant::new(DIAGNOSIS, OpMask::READ),
                        ClassGrant::new(LAB, OpMask::READ),
                    ],
                    "nurse" => vec![ClassGrant::new(PATIENT, OpMask::READ)],
                    other => return Err(RbacDrift::UnknownRole(other.to_string())),
                };
                grants.push((role, gs));
            }
            Ok(RbacBinding::new(
                plug.consumer,
                plug.classids.to_vec(),
                grants,
                vec![("nurse", PATIENT, WideFieldMask::from_positions(&[0, 1]))],
            ))
        }
    }

    const PLUG: RbacPlug = RbacPlug {
        consumer: "demo",
        classids: &[PATIENT, DIAGNOSIS],
        roles: &["physician", "nurse"],
    };

    #[test]
    fn a_bind_refuses_unknown_classids_and_roles() {
        assert_eq!(
            TinyAuthority.bind(&RbacPlug {
                classids: &[0x0B02],
                ..PLUG
            }),
            Err(RbacDrift::UnknownClassid(0x0B02))
        );
        assert_eq!(
            TinyAuthority.bind(&RbacPlug {
                roles: &["janitor"],
                ..PLUG
            }),
            Err(RbacDrift::UnknownRole("janitor".into()))
        );
        assert!(TinyAuthority.bind(&PLUG).is_ok());
    }

    // The authority grants physician lab access, but the plug did not ask for
    // lab: the binding must not carry it.
    #[test]
    fn a_binding_carries_nothing_outside_the_plug() {
        let b = TinyAuthority.bind(&PLUG).unwrap();
        let physician = b.grants_for("physician").unwrap();
        assert_eq!(physician.len(), 2);
        assert!(physician.iter().all(|g| g.target_classid != LAB));
        assert_eq!(
            b.permits("physician", class(LAB), &read()),
            Err(RbacDrift::NotPlugged(LAB))
        );
    }

    #[test]
    fn lookups_outside_the_plug_bang_inside_they_answer() {
        let b = TinyAuthority.bind(&PLUG).unwrap();
        assert_eq!(b.permits("physician", class(DIAGNOSIS), &read()), Ok(true));
        assert_eq!(
            b.permits("nurse", class(DIAGNOSIS), &read()),
            Ok(false),
            "in the plug and not granted is a plain no"
        );
        assert_eq!(
            b.permits("cashier", class(PATIENT), &read()),
            Err(RbacDrift::RoleNotPlugged("cashier".into()))
        );
        assert_eq!(b.plugged(class(PATIENT)), Ok(PATIENT));
        assert_eq!(b.consumer(), "demo");
    }

    #[test]
    fn field_masks_are_per_role_and_plug_bounded() {
        let b = TinyAuthority.bind(&PLUG).unwrap();
        assert_eq!(
            b.field_mask_for("nurse", class(PATIENT)),
            Ok(Some(&WideFieldMask::from_positions(&[0, 1])))
        );
        assert_eq!(b.field_mask_for("physician", class(PATIENT)), Ok(None));
        assert_eq!(
            b.field_mask_for("nurse", class(LAB)),
            Err(RbacDrift::NotPlugged(LAB))
        );
    }

    struct Ward;
    impl ActorSource for Ward {
        fn roles_of(&self, actor: ActorId<'_>) -> &[RoleId] {
            match actor {
                "dr_a" => &["physician"],
                "n_b" => &["nurse"],
                "ghost" => &["cashier"],
                _ => &[],
            }
        }
    }

    #[test]
    fn the_class_rbac_view_denies_outside_the_plug() {
        let b = TinyAuthority.bind(&PLUG).unwrap();
        let rbac = b.with_actors(Ward);
        assert_eq!(rbac.actor_roles("dr_a"), &["physician"]);
        assert!(rbac.grant_permits("physician", class(PATIENT), &read()));
        assert!(!rbac.grant_permits("physician", class(LAB), &read()));
        assert!(!rbac.grant_permits("cashier", class(PATIENT), &read()));
        assert_eq!(
            rbac.field_mask("nurse", class(PATIENT)),
            WideFieldMask::from_positions(&[0, 1])
        );
        assert_eq!(
            rbac.field_mask("physician", class(PATIENT)),
            WideFieldMask::from(FieldMask::FULL)
        );
        assert_eq!(rbac.field_mask("nurse", class(LAB)), WideFieldMask::EMPTY);
        assert_eq!(
            rbac.memberships("n_b", class(PATIENT)),
            vec![Membership {
                role: "nurse",
                scope: None
            }]
        );
    }

    #[test]
    fn mirror_check_names_the_drift() {
        assert_eq!(
            verify_concepts_against_mirror(&[("patient".into(), PATIENT)]),
            Ok(())
        );
        assert!(matches!(
            verify_concepts_against_mirror(&[("patient".into(), 0x0999)]),
            Err(RbacDrift::MirrorDrift {
                authority_id: 0x0999,
                ..
            })
        ));
    }
}
