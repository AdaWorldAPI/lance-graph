//! `actions` — the OGAR **DO-arm provider**: per-class [`ActionDef`] manifests
//! keyed by classid, with **RBAC hardcoded into the class** (the Türsteher).
//!
//! # Why the provider, and why RBAC is `const` here
//!
//! OGAR is the Active-Record Core: a `Class` carries its own behaviour
//! ([`HIRO-IN-CLASSES.md`](../../../../OGAR/docs/HIRO-IN-CLASSES.md)). The DO arm
//! of that behaviour is a `const` table of [`ActionDef`]s per class — the
//! action-axis sibling of the THINK-arm `ClassView`. [`OgarActionProvider`] is
//! the lance-graph-side registry of those tables, resolved exactly like the
//! field-set: `classid → actions`, inheritance-aware via
//! [`contract::action::effective_actions`].
//!
//! The pin (operator, "OGAR der Türsteher mit Köpfchen"): each action's
//! [`required_role`](ActionDef::required_role) is a **compile-time `const` on the
//! class manifest**, not a runtime policy row. RBAC is *baked into the structure*
//! — a compliance reviewer reads the class's grant surface off the source, and an
//! action with no role is structurally `None` (audited), never an accidental open
//! door. Containment is by **structure**, not by trust: whatever cognition sits
//! *above* (the Rung-1-9 Flughöhe in the hot path) cannot widen a class's DO
//! surface, because the surface is fixed in the Core and the
//! [`commit`](contract::action::ActionInvocation::commit) gate reads
//! `required_role` from *here*, not from the caller. "Ogar kriegt sie alle."
//!
//! # The cold-path gate this feeds
//!
//! These manifests are the `def` half of the cold path: a consumer resolves
//! `actions_for(classid)`, matches an [`ActionInvocation`] to its [`ActionDef`] by
//! `predicate`, then [`commit`](contract::action::ActionInvocation::commit)s —
//! which enforces `required_role` (RBAC) → the [`StateGuard`] (Libet do/don't) →
//! the MUL [`GateDecision`](contract::mul::GateDecision) (the Rubicon crossing).
//! The kgV executor that runs the committed action is `graph-flow-action`'s
//! `ActionHandler` in rs-graph-llm; the outer cycle envelope is
//! `graph-flow-kanban`. This crate supplies the *authorized DO surface* both sit
//! over.

use contract::action::{actions_for, effective_actions, ActionDef, ClassActions, StateGuard};
use contract::kanban::ExecTarget;
use lance_graph_contract as contract;

// ── Role identities, hardcoded into the manifests below (the Türsteher's keys).
// These are the `required_role` literals the cold-path `commit` checks against an
// `ActorContext`; an OIDC `auth_store` profile (§7) resolves token claims to them.
const ROLE_AUTH_USER: &str = "auth_user";
const ROLE_AUTH_ADMIN: &str = "auth_admin";

/// `auth_store` (`0x0B01`) DO surface — the base auth membrane's actions, RBAC
/// baked in. The most on-point "RBAC hardcoded in OGAR" example: the class that
/// *does* authorization carries its own authorization on every mutating action.
const AUTH_STORE_ACTIONS: &[ActionDef] = &[
    // Mint a session token — any authenticated principal may.
    ActionDef {
        predicate: "issue_token",
        object_class: AUTH_STORE_CID,
        exec: ExecTarget::Native,
        guard: None,
        required_role: Some(ROLE_AUTH_USER),
        overrides: None,
    },
    // Revoke a token — admin only.
    ActionDef {
        predicate: "revoke_token",
        object_class: AUTH_STORE_CID,
        exec: ExecTarget::Native,
        guard: None,
        required_role: Some(ROLE_AUTH_ADMIN),
        overrides: None,
    },
    // Rotate the signing secret — admin only, and only while the store is active
    // (Libet do/don't state guard: never rotate a suspended store).
    ActionDef {
        predicate: "rotate_secret",
        object_class: AUTH_STORE_CID,
        exec: ExecTarget::Native,
        guard: Some(StateGuard {
            field: "status",
            value: "active",
        }),
        required_role: Some(ROLE_AUTH_ADMIN),
        overrides: None,
    },
];

/// `auth_zitadel` (`0x0B02`) net-new DO surface. `auth_zitadel` **is-a**
/// `auth_store` (§7 provider profile), so its *effective* actions are the base's
/// plus these — and `issue_token` is **overridden** to run the Zitadel
/// org-role-aware path (an Elixir low-code template; the JITson ↔ Elixir ↔
/// SurrealQL exec seam). Same `required_role` is restated to keep the override's
/// grant surface explicit (an override must not silently widen access).
const AUTH_ZITADEL_ACTIONS: &[ActionDef] = &[
    ActionDef {
        predicate: "issue_token",
        object_class: AUTH_ZITADEL_CID,
        exec: ExecTarget::Elixir,
        guard: None,
        required_role: Some(ROLE_AUTH_USER),
        overrides: Some("auth_store::issue_token"),
    },
    // Sync org→role grants from Zitadel — admin only, low-code template.
    ActionDef {
        predicate: "sync_org_roles",
        object_class: AUTH_ZITADEL_CID,
        exec: ExecTarget::Elixir,
        guard: None,
        required_role: Some(ROLE_AUTH_ADMIN),
        overrides: None,
    },
];

// The auth-family codebook ids (keystone §7 `0x0B` core domain). Written as
// literals — NOT `ogar_vocab::class_ids::AUTH_STORE` — so this DO-arm provider is
// strictly `lance_graph_contract`-dependent and does not couple to whichever
// `ogar-vocab` git ref this crate pins (the action manifest is a contract-shaped
// artifact, exactly as `contract::action::ClassActions` documents — "generated
// downstream; the Core provides the type"). They MUST equal
// `ogar_vocab::class_ids::{AUTH_STORE, AUTH_ZITADEL}`; the lib's `parity` guard is
// what binds the codebook itself.
const AUTH_STORE_CID: u32 = 0x0000_0B01;
const AUTH_ZITADEL_CID: u32 = 0x0000_0B02;

/// The registry: one [`ClassActions`] row per class with a DO surface. Seeded
/// with the auth family (the worked hardcoded-RBAC example); other domains append
/// their `const` tables here as their harvests land.
const REGISTRY: &[ClassActions] = &[
    ClassActions {
        classid: AUTH_STORE_CID,
        actions: AUTH_STORE_ACTIONS,
    },
    ClassActions {
        classid: AUTH_ZITADEL_CID,
        actions: AUTH_ZITADEL_ACTIONS,
    },
];

/// The lance-graph-side OGAR DO-arm provider — resolves a classid to its
/// authorized action manifest. Zero-state: it wraps the `const` [`REGISTRY`]; the
/// "provider" name marks it as the lookup surface a consumer holds (mirrors how
/// `OgarClassView` is the THINK-arm projection surface).
#[derive(Debug, Clone, Copy, Default)]
pub struct OgarActionProvider;

impl OgarActionProvider {
    /// Construct the provider (the registry is `const`, so this is zero-cost).
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The full action registry.
    #[must_use]
    pub const fn registry(&self) -> &'static [ClassActions] {
        REGISTRY
    }

    /// A class's **own** actions (not its parents'). Zero-fallback: an
    /// unregistered classid resolves to `&[]`, never a panic.
    #[must_use]
    pub fn actions_for(&self, classid: u32) -> &'static [ActionDef] {
        actions_for(REGISTRY, classid)
    }

    /// A class's **effective** DO surface — its own actions composed with its
    /// OGAR parent's (`is-a`), child overrides applied. For `auth_zitadel` this
    /// is the `auth_store` base + the Zitadel net-new, with `issue_token`
    /// overridden. A root class (no parent) returns its own actions unchanged.
    #[must_use]
    pub fn effective_actions(&self, classid: u32) -> Vec<ActionDef> {
        match self.parent_of(classid) {
            Some(parent) => effective_actions(self.actions_for(parent), self.actions_for(classid)),
            None => self.actions_for(classid).to_vec(),
        }
    }

    /// The OGAR `is-a` parent of a class for DO-arm inheritance. The auth provider
    /// profiles (§7) are-a `auth_store`; everything else is a root here until its
    /// `class_ancestors` edge is wired. (Kept local + explicit so the inheritance
    /// the DO arm uses is auditable, not implicit.)
    #[must_use]
    fn parent_of(&self, classid: u32) -> Option<u32> {
        match classid {
            AUTH_ZITADEL_CID => Some(AUTH_STORE_CID),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_store_surface_has_hardcoded_rbac_on_every_mutating_action() {
        let p = OgarActionProvider::new();
        let acts = p.actions_for(AUTH_STORE_CID);
        assert_eq!(acts.len(), 3);
        // Every action carries a required_role — RBAC is baked into the class,
        // there is no roleless mutating action (the Türsteher invariant).
        assert!(acts.iter().all(|a| a.required_role.is_some()));
        // issue_token is auth_user; revoke/rotate are admin.
        let by = |name: &str| acts.iter().find(|a| a.predicate == name).unwrap();
        assert_eq!(by("issue_token").required_role, Some(ROLE_AUTH_USER));
        assert_eq!(by("revoke_token").required_role, Some(ROLE_AUTH_ADMIN));
        assert_eq!(by("rotate_secret").required_role, Some(ROLE_AUTH_ADMIN));
        // rotate_secret guards on status==active (Libet do/don't).
        assert_eq!(
            by("rotate_secret").guard,
            Some(StateGuard {
                field: "status",
                value: "active"
            })
        );
    }

    #[test]
    fn zitadel_inherits_auth_store_and_overrides_issue_token() {
        let p = OgarActionProvider::new();
        let eff = p.effective_actions(AUTH_ZITADEL_CID);
        // base(3) ∪ net-new(sync_org_roles); issue_token overridden, not doubled.
        let names: Vec<&str> = eff.iter().map(|a| a.predicate).collect();
        assert!(names.contains(&"issue_token"));
        assert!(names.contains(&"revoke_token")); // inherited, unchanged
        assert!(names.contains(&"rotate_secret")); // inherited, unchanged
        assert!(names.contains(&"sync_org_roles")); // net-new
        assert_eq!(
            names.iter().filter(|n| **n == "issue_token").count(),
            1,
            "override must replace, not duplicate"
        );
        // The override is the Zitadel Elixir-low-code path, and it did NOT widen
        // the grant (still auth_user, never silently elevated).
        let issue = eff.iter().find(|a| a.predicate == "issue_token").unwrap();
        assert_eq!(issue.exec, ExecTarget::Elixir);
        assert_eq!(issue.required_role, Some(ROLE_AUTH_USER));
        assert!(issue.is_override());
    }

    #[test]
    fn unregistered_classid_is_zero_fallback_empty() {
        let p = OgarActionProvider::new();
        assert!(p.actions_for(0xDEAD_BEEF).is_empty());
        assert!(p.effective_actions(0xDEAD_BEEF).is_empty());
    }

    #[test]
    fn exec_targets_span_the_jitson_elixir_seam() {
        // The DO surface carries the exec target through to the kanban/handler
        // layer untouched: native for the base store, Elixir low-code for the
        // Zitadel provider profile.
        let p = OgarActionProvider::new();
        assert!(p
            .actions_for(AUTH_STORE_CID)
            .iter()
            .all(|a| a.exec == ExecTarget::Native));
        assert!(p
            .actions_for(AUTH_ZITADEL_CID)
            .iter()
            .all(|a| a.exec == ExecTarget::Elixir));
    }
}
