//! `auth` — the OGIT-imported AuthStore class family (`0x0B`) wired to the
//! authorization kernel (OGAR keystone §7).
//!
//! # The membrane, not the kernel
//!
//! The keystone draws a hard line (I-K7): **the inner [`authorize`] kernel never
//! touches a token.** A token is parsed once at the membrane; the chosen
//! `auth_store` provider profile resolves its claims to canonical classids/roles;
//! and only those *resolved keys* go inward. This module is that membrane step —
//! the OGIT `NTO/Auth/Configuration` entity (arago's `auth_store`, 1:1 with OGAR's
//! `0x0B01`) made executable:
//!
//! ```text
//!   raw token  ──parse──▶  RawClaims  ──AuthProvider::resolve──▶  ResolvedIdentity
//!   (membrane)            (per-IdP grammar)                       (actor + roles + tenant)
//!                                                                        │
//!                                                                        ▼
//!                                          authorize(rbac, &id.actor, class, op)
//! ```
//!
//! The [`AuthProvider`] variants ARE the preminted `0x0B` family
//! (`auth_store` 0x0B01 base + `auth_zitadel`/`auth_zanzibar`/`auth_ory_keto`
//! provider profiles). Selecting a provider = picking its codebook classid; the
//! classid is resolved through the zero-dep contract mirror
//! ([`lance_graph_contract::ogar_codebook::canonical_concept_id`]), so this crate
//! pulls the identity from ONE source (BBB-safe: no `ogar-vocab` dependency).
//!
//! # The §7 mapping
//!
//! Each provider carries its own *claim grammar* as data: which claim key holds
//! the subject, the role list, and the org/tenant. `resolve` applies it —
//! `sub → actor`, `role-key → roles`, `org → tenant` (the scope axis) — and
//! returns owned [`ResolvedIdentity`] strings. Mapping the resolved IdP role
//! strings to the app's own role set is the *consumer's* job (a small fixed
//! IdP-role → app-role table); see [`ResolvedIdentity`] and the tests for the
//! handoff into [`authorize`].

use crate::authorize::ClassId;

/// The preminted AuthStore class family (`0x0B`). Each variant is one codebook
/// concept; the classid is resolved through the contract mirror so there is no
/// hardcoded `0x0B0N` and no `ogar-vocab` dependency. `Store` is the base
/// (provider-agnostic); the others are per-IdP profiles that is-a `Store`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuthProvider {
    /// `auth_store` (0x0B01) — the base. Provider-agnostic claim resolution.
    Store,
    /// `auth_zitadel` (0x0B02) — Zitadel claim grammar (org-project-roles URN).
    Zitadel,
    /// `auth_zanzibar` (0x0B03) — Zanzibar / OpenFGA tuple grammar.
    Zanzibar,
    /// `auth_ory_keto` (0x0B04) — Ory Keto.
    OryKeto,
}

impl AuthProvider {
    /// The canonical concept name (the codebook key).
    #[must_use]
    pub const fn concept(self) -> &'static str {
        match self {
            Self::Store => "auth_store",
            Self::Zitadel => "auth_zitadel",
            Self::Zanzibar => "auth_zanzibar",
            Self::OryKeto => "auth_ory_keto",
        }
    }

    /// The codebook classid (the low-`u16`), resolved through the zero-dep
    /// contract mirror — the single source of truth, no hardcoded `0x0B0N`.
    /// Panics only if the contract mirror and this enum drift, which the
    /// `provider_class_ids_resolve_through_the_contract_mirror` test forbids.
    #[must_use]
    pub fn class_id(self) -> u16 {
        lance_graph_contract::ogar_codebook::canonical_concept_id(self.concept())
            .expect("AuthProvider concept must exist in the contract codebook mirror")
    }

    /// Reverse: a codebook classid (low `u16`) back to its provider, if it is in
    /// the `0x0B` AuthStore family. `None` for any non-auth id.
    #[must_use]
    pub fn from_class_id(id: u16) -> Option<Self> {
        [Self::Store, Self::Zitadel, Self::Zanzibar, Self::OryKeto]
            .into_iter()
            .find(|p| p.class_id() == id)
    }

    /// As a full 32-bit `ClassId` (hi-`u16` core prefix `0x0000`, lo-`u16`
    /// concept) — the form [`authorize`](crate::authorize::authorize) and the
    /// `NodeGuid` classid take. Auth concepts are core (cross-app), so the
    /// render prefix is `0x0000`.
    #[must_use]
    pub fn classid(self) -> ClassId {
        u32::from(self.class_id())
    }

    /// The claim-key grammar for this provider — which claim names carry the
    /// subject, the role list, and the org/tenant. The per-IdP grammar the
    /// keystone §7 says each profile "carries as data". `Store` uses the plain
    /// OIDC defaults; the named providers override the ones that differ.
    #[must_use]
    pub const fn grammar(self) -> ClaimGrammar {
        match self {
            // Plain OIDC defaults.
            Self::Store | Self::OryKeto => ClaimGrammar {
                subject_claim: "sub",
                roles_claim: "roles",
                tenant_claim: "org",
            },
            // Zitadel: roles live under the project-roles URN; org is the URN org id.
            Self::Zitadel => ClaimGrammar {
                subject_claim: "sub",
                roles_claim: "urn:zitadel:iam:org:project:roles",
                tenant_claim: "urn:zitadel:iam:org:id",
            },
            // Zanzibar/OpenFGA: the subject is the tuple's user; relations are roles.
            Self::Zanzibar => ClaimGrammar {
                subject_claim: "user",
                roles_claim: "relation",
                tenant_claim: "namespace",
            },
        }
    }

    /// Apply the §7 mapping: `sub → actor`, `role-key → roles`, `org → tenant`.
    /// `subject` is the already-extracted subject value; `role_values` the
    /// already-extracted role list; `tenant` the org/tenant. (Extraction from a
    /// concrete token uses [`grammar`](Self::grammar) at the membrane — kept out
    /// of this crate so no JWT/JSON dependency leaks into the contract tier.)
    #[must_use]
    pub fn resolve(
        self,
        subject: impl Into<String>,
        role_values: impl IntoIterator<Item = String>,
        tenant: Option<String>,
    ) -> ResolvedIdentity {
        ResolvedIdentity {
            provider: self,
            actor: subject.into(),
            roles: role_values.into_iter().collect(),
            tenant,
        }
    }
}

/// The claim-key grammar a provider profile carries as data (keystone §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaimGrammar {
    /// Claim holding the subject (→ actor).
    pub subject_claim: &'static str,
    /// Claim holding the role list (→ roles).
    pub roles_claim: &'static str,
    /// Claim holding the org / tenant (→ scope axis).
    pub tenant_claim: &'static str,
}

/// The resolved identity — the ONLY thing that crosses the membrane inward
/// (no token, per I-K7). Owned strings: the actor (from `sub`), the IdP role
/// strings (mapped to the app's role set by the consumer), and the tenant
/// (scope axis). The provider it was resolved through is retained for audit /
/// provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedIdentity {
    /// Which `0x0B` profile resolved this identity.
    pub provider: AuthProvider,
    /// The actor — the OIDC `sub`, resolved to a membership key.
    pub actor: String,
    /// The IdP role strings. The consumer maps these to its own role set (a
    /// fixed IdP-role → app-role table) before calling
    /// [`authorize`](crate::authorize::authorize).
    pub roles: Vec<String>,
    /// The org / tenant — the scope axis (§5 stage 2). `None` = unscoped.
    pub tenant: Option<String>,
}

impl ResolvedIdentity {
    /// Does the resolved identity carry `role` (raw IdP string)? Convenience for
    /// the consumer's IdP-role → app-role mapping.
    #[must_use]
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    /// The auth-class classid this identity was resolved through — for the audit
    /// witness (which `0x0B` profile authorized the actor).
    #[must_use]
    pub fn auth_classid(&self) -> ClassId {
        self.provider.classid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorize::{authorize, ClassGrants};
    use crate::permission::PermissionSpec;
    use crate::policy::Operation;

    #[test]
    fn provider_class_ids_resolve_through_the_contract_mirror() {
        // The 0x0B family resolves through the zero-dep contract mirror — one
        // source, no hardcoded ids. Pins the OGIT-imported auth class to the
        // codebook.
        assert_eq!(AuthProvider::Store.class_id(), 0x0B01);
        assert_eq!(AuthProvider::Zitadel.class_id(), 0x0B02);
        assert_eq!(AuthProvider::Zanzibar.class_id(), 0x0B03);
        assert_eq!(AuthProvider::OryKeto.class_id(), 0x0B04);
        // Full classid is core-prefixed (hi u16 = 0x0000 — auth is cross-app).
        assert_eq!(AuthProvider::Store.classid(), 0x0000_0B01);
        // Round-trips.
        for p in [
            AuthProvider::Store,
            AuthProvider::Zitadel,
            AuthProvider::Zanzibar,
            AuthProvider::OryKeto,
        ] {
            assert_eq!(AuthProvider::from_class_id(p.class_id()), Some(p));
        }
        // A non-auth id is not in the family.
        assert_eq!(AuthProvider::from_class_id(0x0901), None); // patient
    }

    #[test]
    fn provider_grammars_match_keystone_section_7() {
        // Zitadel's project-roles URN + org-id URN (the §7 worked example).
        let z = AuthProvider::Zitadel.grammar();
        assert_eq!(z.roles_claim, "urn:zitadel:iam:org:project:roles");
        assert_eq!(z.tenant_claim, "urn:zitadel:iam:org:id");
        // Zanzibar's tuple grammar (user / relation / namespace).
        let zn = AuthProvider::Zanzibar.grammar();
        assert_eq!(zn.subject_claim, "user");
        assert_eq!(zn.roles_claim, "relation");
        // Store is the plain-OIDC base.
        assert_eq!(AuthProvider::Store.grammar().subject_claim, "sub");
    }

    #[test]
    fn resolve_maps_sub_roles_and_org() {
        let id = AuthProvider::Zitadel.resolve(
            "user-42",
            ["physician".to_string(), "billing".to_string()],
            Some("clinic-7".to_string()),
        );
        assert_eq!(id.actor, "user-42");
        assert!(id.has_role("physician"));
        assert!(!id.has_role("admin"));
        assert_eq!(id.tenant.as_deref(), Some("clinic-7"));
        assert_eq!(id.auth_classid(), 0x0000_0B02);
    }

    #[test]
    fn resolved_identity_feeds_authorize() {
        // The end-to-end seam: an identity resolved at the membrane feeds the
        // inner authorize() kernel. The consumer maps the IdP role string
        // ("accountant") to the app's known &'static role name; here the
        // mapping is identity, modelling an IdP whose role names match the app.
        let grants = ClassGrants::new()
            .with_grant(
                "accountant",
                0x0000_C002, // probe-local Invoice classid
                PermissionSpec::full("Invoice", &["status"], &["approve"]),
            )
            .with_actor("user-42", vec!["accountant"]);

        let id = AuthProvider::Store.resolve(
            "user-42",
            ["accountant".to_string()],
            Some("clinic-7".to_string()),
        );

        // Membrane resolved → kernel authorizes on the resolved actor.
        let decision = authorize(
            &grants,
            &id.actor,
            0x0000_C002,
            Operation::Act { action: "approve" },
        );
        assert!(decision.is_allowed());

        // A write to a predicate outside the grant's writable set is denied
        // (the grant allows writing "status", not "due_date") — kernel unchanged.
        let denied = authorize(
            &grants,
            &id.actor,
            0x0000_C002,
            Operation::Write {
                predicate: "due_date",
            },
        );
        assert!(denied.is_denied());
    }
}
