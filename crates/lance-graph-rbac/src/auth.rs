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
//!
//! # Delegation (RFC 8693 `act`)
//!
//! A token obtained by token exchange may say that one party acts on behalf
//! of another: `sub` names the principal whose authority the token carries,
//! and the `act` claim names the party acting with it. A nested `act` inside
//! `act` names an earlier actor in a chain.
//!
//! ```text
//!   { "sub": "alice", "act": { "sub": "bob", "act": { "sub": "svc" } } }
//!     alice's authority     bob acts now       svc acted before bob
//! ```
//!
//! [`ResolvedIdentity::acting_through`] records that. Authorization stays on
//! [`ResolvedIdentity::actor`], the `sub`: the token carries the principal's
//! authority, and the authorization server already decided the actor may use
//! it. The current actor is kept for audit and for the decisions that name
//! the acting party (who sent a message, who read a mailbox). Prior actors are
//! informational only and never take part in an access decision (RFC 8693
//! §4.1).

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

    /// The codebook classid (the canon `u16` — the high half of a `NodeGuid`
    /// classid since the 2026-07-02 flip), resolved through the zero-dep
    /// contract mirror — the single source of truth, no hardcoded `0x0B0N`.
    /// Panics only if the contract mirror and this enum drift, which the
    /// `provider_class_ids_resolve_through_the_contract_mirror` test forbids.
    #[must_use]
    pub fn class_id(self) -> u16 {
        lance_graph_contract::ogar_codebook::canonical_concept_id(self.concept())
            .expect("AuthProvider concept must exist in the contract codebook mirror")
    }

    /// Reverse: a codebook classid (canon `u16`) back to its provider, if it is in
    /// the `0x0B` AuthStore family. `None` for any non-auth id.
    #[must_use]
    pub fn from_class_id(id: u16) -> Option<Self> {
        [Self::Store, Self::Zitadel, Self::Zanzibar, Self::OryKeto]
            .into_iter()
            .find(|p| p.class_id() == id)
    }

    /// As a full 32-bit `ClassId` under the core render lens (concept in the
    /// CANON high `u16` since the 2026-07-02 half-order flip, prefix `0x0000`
    /// in the custom low half) — the form
    /// [`authorize`](crate::authorize::authorize) and the `NodeGuid` classid
    /// take. Auth concepts are core (cross-app). Routed through the
    /// contract's one flippable composition — never a local widening.
    #[must_use]
    pub fn classid(self) -> ClassId {
        lance_graph_contract::render_classid(0x0000, self.class_id())
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
                act_claim: Some("act"),
            },
            // Zitadel: roles live under the project-roles URN; org is the URN org id.
            Self::Zitadel => ClaimGrammar {
                subject_claim: "sub",
                roles_claim: "urn:zitadel:iam:org:project:roles",
                tenant_claim: "urn:zitadel:iam:org:id",
                act_claim: Some("act"),
            },
            // Zanzibar/OpenFGA: the subject is the tuple's user; relations are roles.
            Self::Zanzibar => ClaimGrammar {
                subject_claim: "user",
                roles_claim: "relation",
                tenant_claim: "namespace",
                // A relation tuple is not a token: nothing is exchanged.
                act_claim: None,
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
            acting: None,
            prior_actors: Vec::new(),
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
    /// Claim naming the acting party of a delegated token (RFC 8693 `act`).
    /// `None`: the provider issues no exchanged tokens, so its identities
    /// are never delegated.
    pub act_claim: Option<&'static str>,
}

/// One party named by an RFC 8693 `act` claim: its `sub`, and its `iss` when
/// the claim carries one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Actor {
    /// The actor's `sub`.
    pub subject: String,
    /// The actor's `iss`, when it differs from the token's issuer.
    pub issuer: Option<String>,
}

impl Actor {
    /// An actor by subject, issued by the token's own issuer.
    #[must_use]
    pub fn new(subject: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            issuer: None,
        }
    }
}

/// Why a delegation could not be recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelegationRefused {
    /// The provider has no `act` claim ([`ClaimGrammar::act_claim`] is
    /// `None`), so no token it resolves can be delegated.
    NotIssuedByProvider,
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
    /// The party acting on [`actor`](Self::actor)'s behalf: the outermost
    /// `act` (RFC 8693). `None` when the token is not delegated.
    pub acting: Option<Actor>,
    /// Earlier actors from the nested `act` claims, most recent first.
    /// Informational only: never part of an access decision.
    pub prior_actors: Vec<Actor>,
}

impl ResolvedIdentity {
    /// Does the resolved identity carry `role` (raw IdP string)? Convenience for
    /// the consumer's IdP-role → app-role mapping.
    #[must_use]
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    /// Record a delegation from the token's `act` claim: `current` is the
    /// outermost `act`, `prior` the nested ones, most recent first. The
    /// authority stays [`actor`](Self::actor)'s.
    ///
    /// # Errors
    ///
    /// [`DelegationRefused::NotIssuedByProvider`] when the provider's grammar
    /// has no `act` claim.
    pub fn acting_through(
        mut self,
        current: Actor,
        prior: impl IntoIterator<Item = Actor>,
    ) -> Result<Self, DelegationRefused> {
        if self.provider.grammar().act_claim.is_none() {
            return Err(DelegationRefused::NotIssuedByProvider);
        }
        self.acting = Some(current);
        self.prior_actors = prior.into_iter().collect();
        Ok(self)
    }

    /// Whether another party acts on the principal's behalf.
    #[must_use]
    pub fn is_delegated(&self) -> bool {
        self.acting.is_some()
    }

    /// The party performing the request: the current `act` when delegated,
    /// else the principal itself.
    #[must_use]
    pub fn acting_party(&self) -> &str {
        self.acting
            .as_ref()
            .map_or(self.actor.as_str(), |a| a.subject.as_str())
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
        // Full classid under the core lens: concept in the CANON high u16
        // (post-flip form), custom prefix 0x0000 — auth is cross-app.
        assert_eq!(AuthProvider::Store.classid(), 0x0B01_0000);
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
        // Token providers name the acting party in `act`; a tuple store has none.
        assert_eq!(AuthProvider::Zitadel.grammar().act_claim, Some("act"));
        assert_eq!(AuthProvider::Store.grammar().act_claim, Some("act"));
        assert_eq!(AuthProvider::Zanzibar.grammar().act_claim, None);
    }

    fn mailbox_grants() -> ClassGrants {
        ClassGrants::new()
            .with_grant(
                "owner",
                0x0000_C003, // probe-local Mailbox classid
                PermissionSpec::full("Mailbox", &["flags"], &["send"]),
            )
            .with_actor("alice", vec!["owner"])
    }

    // `{ "sub": "alice", "act": { "sub": "bob" } }`: bob acts with alice's
    // authority. bob holds no grant of his own, and needs none.
    #[test]
    fn a_delegated_identity_authorizes_as_the_principal() {
        let id = AuthProvider::Zitadel
            .resolve("alice", Vec::new(), None)
            .acting_through(Actor::new("bob"), [])
            .unwrap();
        assert!(id.is_delegated());
        assert_eq!(id.actor, "alice");
        assert_eq!(id.acting_party(), "bob");
        let grants = mailbox_grants();
        let op = Operation::Act { action: "send" };
        assert!(authorize(&grants, &id.actor, 0x0000_C003, op.clone()).is_allowed());
        // The acting party's own authority is not the token's.
        assert!(authorize(&grants, id.acting_party(), 0x0000_C003, op).is_denied());
    }

    #[test]
    fn an_undelegated_identity_acts_as_itself() {
        let id = AuthProvider::Zitadel.resolve("alice", Vec::new(), None);
        assert!(!id.is_delegated());
        assert_eq!(id.acting_party(), "alice");
        assert!(id.prior_actors.is_empty());
    }

    // `{ "sub": "alice", "act": { "sub": "bob", "act": { "sub": "svc" } } }`:
    // bob is the current actor; svc acted earlier and only rides along.
    #[test]
    fn the_outermost_act_is_the_current_actor() {
        let id = AuthProvider::Store
            .resolve("alice", Vec::new(), None)
            .acting_through(
                Actor::new("bob"),
                [Actor {
                    subject: "svc".into(),
                    issuer: Some("https://other.example".into()),
                }],
            )
            .unwrap();
        assert_eq!(id.acting_party(), "bob");
        assert_eq!(id.prior_actors.len(), 1);
        assert_eq!(id.prior_actors[0].subject, "svc");
        assert_eq!(id.actor, "alice", "authority is still the principal's");
    }

    #[test]
    fn a_provider_without_tokens_cannot_be_delegated() {
        let id = AuthProvider::Zanzibar.resolve("alice", Vec::new(), None);
        assert_eq!(
            id.acting_through(Actor::new("bob"), []),
            Err(DelegationRefused::NotIssuedByProvider)
        );
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
        assert_eq!(id.auth_classid(), 0x0B02_0000);
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
