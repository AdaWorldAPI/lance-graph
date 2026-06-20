//! OpenProject tenant bridge — locks to the `OpenProject` namespace.
//!
//! Northstar plan §3, C4. Sibling of [`crate::bridges::MedcareBridge`] and
//! [`crate::bridges::WoaBridge`]; supplies the OpenProject port
//! (`openproject-nexgen-rs` + `op-canon`) with the scoped registry view
//! every consumer that touches OpenProject data on the unified bridge
//! goes through.
//!
//! # Scope
//!
//! The `OpenProject` namespace covers the **32 promoted concepts** from
//! the OGAR codebook that the OpenProject corpus owns or shares — see
//! `ogar_vocab::class_ids::ALL`. Public-name resolution
//! (`bridge.entity("WorkPackage")` etc.) returns the `EntityRef` whose
//! `entity_type_id()` is the codebook id (e.g. `0x0102` for
//! `project_work_item`), so downstream consumers reach the same
//! `OgarClassView` arm as Redmine or any other port that emits through
//! the same codebook.
//!
//! # Sibling work
//!
//! - **C5** (RedmineBridge, redmine-rs) — symmetric "Redmine" namespace
//!   bridge, same codebook ids.
//! - **C2** (op-canon::class_view) — the run-time projection layer over
//!   the same canonical concepts the bridge resolves names against.

use crate::bridge::{BridgeFromRegistry, NamespaceBridge};
use crate::error::{Error, Result};
use crate::namespace::NamespaceId;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

/// Canonical namespace name for OpenProject. Matches the
/// `ogit.OpenProject:` TTL prefix the corpus's per-entity files use.
pub const NAMESPACE: &str = "OpenProject";

/// Scoped registry view locked to the `OpenProject` namespace. Built
/// over an [`OntologyRegistry`] that has already hydrated the
/// OpenProject namespace (via [`OntologyRegistry::hydrate_once_sync`]
/// or a programmatic `register_*` call).
pub struct OpenProjectBridge {
    registry: Arc<OntologyRegistry>,
    g_lock: NamespaceId,
}

impl OpenProjectBridge {
    /// New bridge over the given registry. Returns
    /// [`Error::UnknownNamespace`] if the `OpenProject` namespace
    /// is not registered yet — callers must hydrate before
    /// constructing the bridge.
    ///
    /// # Errors
    ///
    /// - [`Error::UnknownNamespace`] when the registry has no
    ///   `OpenProject` namespace registered.
    pub fn new(registry: Arc<OntologyRegistry>) -> Result<Self> {
        let g_lock = registry
            .namespace_id(NAMESPACE)
            .ok_or_else(|| Error::UnknownNamespace(NAMESPACE.to_string()))?;
        Ok(Self { registry, g_lock })
    }
}

impl NamespaceBridge for OpenProjectBridge {
    fn bridge_id(&self) -> &'static str {
        "openproject"
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }
}

impl BridgeFromRegistry for OpenProjectBridge {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::new(registry)
    }
}

// Compile-only check that BridgeError is reachable through the entity
// resolution path (matches the convention WoaBridge / MedcareBridge use).
#[allow(dead_code)]
fn _compile_check(b: &OpenProjectBridge) -> std::result::Result<(), crate::bridge::BridgeError> {
    let _ = b.entity("WorkPackage")?;
    Ok(())
}
