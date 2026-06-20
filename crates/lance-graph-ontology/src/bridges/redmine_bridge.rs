//! Redmine tenant bridge — locks to the `Redmine` namespace.
//!
//! Northstar plan §3, C5. Sibling of [`crate::bridges::OpenProjectBridge`]
//! (C4); supplies the Redmine port (`redmine-rs` + `redmine-canon`) with
//! the scoped registry view every consumer that touches Redmine data on
//! the unified bridge goes through.
//!
//! # Scope
//!
//! The `Redmine` namespace covers the **same 32 promoted concepts**
//! from the OGAR codebook that `redmine-canon` re-exports (project,
//! issue, time_entry, project_role, etc.). Public-name resolution
//! (`bridge.entity("Issue")` etc.) returns the `EntityRef` whose
//! `entity_type_id()` is the codebook id (e.g. `0x0102` for
//! `project_work_item` — the canonical concept that Redmine's `Issue`
//! and OpenProject's `WorkPackage` BOTH map to).
//!
//! # Apple meets apple — the convergence pin
//!
//! C4 + C5 together: a consumer holding `UnifiedBridge<OpenProjectBridge>`
//! AND `UnifiedBridge<RedmineBridge>` resolves `"WorkPackage"` OR
//! `"Issue"` through the policy-evaluated, audit-chained path → both
//! return `EntityRef`s whose `entity_type_id()` is the SAME OGAR
//! codebook id, routing to the same `OgarClassView` arm. That's the
//! cross-fork convergence the codebook was calcified for; see
//! `redmine-rs/fork_convergence.json` (schema `fork-convergence/2`) for
//! the 26/26 concept-pair pin.

use crate::bridge::{BridgeFromRegistry, NamespaceBridge};
use crate::error::{Error, Result};
use crate::namespace::NamespaceId;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

/// Canonical namespace name for Redmine. Matches the
/// `ogit.Redmine:` TTL prefix the corpus's per-entity files use.
pub const NAMESPACE: &str = "Redmine";

/// Scoped registry view locked to the `Redmine` namespace. Built over
/// an [`OntologyRegistry`] that has already hydrated the Redmine
/// namespace (via [`OntologyRegistry::hydrate_once_sync`] or a
/// programmatic `register_*` call).
pub struct RedmineBridge {
    registry: Arc<OntologyRegistry>,
    g_lock: NamespaceId,
}

impl RedmineBridge {
    /// New bridge over the given registry. Returns
    /// [`Error::UnknownNamespace`] if the `Redmine` namespace
    /// is not registered yet — callers must hydrate before
    /// constructing the bridge.
    ///
    /// # Errors
    ///
    /// - [`Error::UnknownNamespace`] when the registry has no
    ///   `Redmine` namespace registered.
    pub fn new(registry: Arc<OntologyRegistry>) -> Result<Self> {
        let g_lock = registry
            .namespace_id(NAMESPACE)
            .ok_or_else(|| Error::UnknownNamespace(NAMESPACE.to_string()))?;
        Ok(Self { registry, g_lock })
    }
}

impl NamespaceBridge for RedmineBridge {
    fn bridge_id(&self) -> &'static str {
        "redmine"
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }
}

impl BridgeFromRegistry for RedmineBridge {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::new(registry)
    }
}

// Compile-only check that BridgeError is reachable through the entity
// resolution path (matches the convention WoaBridge / MedcareBridge /
// OpenProjectBridge use).
#[allow(dead_code)]
fn _compile_check(b: &RedmineBridge) -> std::result::Result<(), crate::bridge::BridgeError> {
    let _ = b.entity("Issue")?;
    Ok(())
}
