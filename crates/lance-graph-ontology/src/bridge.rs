//! `NamespaceBridge` trait and the canonical `BridgeError` type.
//!
//! A bridge is a thin scoped view over the shared `OntologyRegistry`. It
//! does two things only:
//!
//! 1. Locks every operation to one G partition (cross-namespace access
//!    requires explicit unlock — the bridge does not provide that).
//! 2. Translates the bridge's public-facing entity / edge / attribute names
//!    to OGIT URIs via the registry.
//!
//! The defaults here do all the work; a tenant bridge typically supplies
//! `bridge_id()` + `g_lock()` + a constructor and is otherwise ~5 lines.

use crate::error::Result;
use crate::namespace::{NamespaceId, OgitUri, SchemaPtr};
use crate::proposal::MappingRow;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("bridge `{bridge_id}`: namespace `{namespace}` is not registered")]
    NamespaceMissing {
        bridge_id: &'static str,
        namespace: &'static str,
    },

    #[error("bridge `{bridge_id}`: public name `{public_name}` is not registered")]
    NotInScope {
        bridge_id: &'static str,
        public_name: String,
    },

    #[error("bridge `{bridge_id}`: cross-namespace leak — resolved to namespace {resolved_id:?} but locked to {locked_id:?}")]
    CrossNamespaceLeak {
        bridge_id: &'static str,
        resolved_id: NamespaceId,
        locked_id: NamespaceId,
    },
}

/// A scoped view of the shared registry. Implementations supply the
/// constants; the defaults handle resolution + scope-lock enforcement.
pub trait NamespaceBridge: Send + Sync {
    fn bridge_id(&self) -> &'static str;
    fn registry(&self) -> &OntologyRegistry;
    fn g_lock(&self) -> NamespaceId;

    fn entity(&self, public_name: &str) -> std::result::Result<EntityRef, BridgeError> {
        let ptr = self
            .registry()
            .resolve(self.bridge_id(), public_name)
            .ok_or(BridgeError::NotInScope {
                bridge_id: self.bridge_id_static(),
                public_name: public_name.to_string(),
            })?;
        if ptr.namespace_id() != self.g_lock() {
            return Err(BridgeError::CrossNamespaceLeak {
                bridge_id: self.bridge_id_static(),
                resolved_id: ptr.namespace_id(),
                locked_id: self.g_lock(),
            });
        }
        Ok(EntityRef { schema_ptr: ptr })
    }

    fn edge(&self, public_name: &str) -> std::result::Result<EdgeRef, BridgeError> {
        let ptr = self
            .registry()
            .resolve(self.bridge_id(), public_name)
            .ok_or(BridgeError::NotInScope {
                bridge_id: self.bridge_id_static(),
                public_name: public_name.to_string(),
            })?;
        if ptr.namespace_id() != self.g_lock() {
            return Err(BridgeError::CrossNamespaceLeak {
                bridge_id: self.bridge_id_static(),
                resolved_id: ptr.namespace_id(),
                locked_id: self.g_lock(),
            });
        }
        Ok(EdgeRef { schema_ptr: ptr })
    }

    /// Resolve by raw OGIT URI. Useful for the `ogit` bridge that does
    /// not maintain a public-name dictionary; tenants generally prefer
    /// `entity()` / `edge()`.
    fn entity_by_uri(&self, uri: &OgitUri) -> std::result::Result<EntityRef, BridgeError> {
        let ptr = self
            .registry()
            .resolve_uri(uri.as_str())
            .ok_or(BridgeError::NotInScope {
                bridge_id: self.bridge_id_static(),
                public_name: uri.as_str().to_string(),
            })?;
        if ptr.namespace_id() != self.g_lock() {
            return Err(BridgeError::CrossNamespaceLeak {
                bridge_id: self.bridge_id_static(),
                resolved_id: ptr.namespace_id(),
                locked_id: self.g_lock(),
            });
        }
        Ok(EntityRef { schema_ptr: ptr })
    }

    /// Returns the underlying dictionary row (full audit detail).
    fn row(&self, public_name: &str) -> std::result::Result<MappingRow, BridgeError> {
        let entity = self.entity(public_name)?;
        let _ptr = entity.schema_ptr;
        // Re-look-up via URI to get the row — `resolve` returns just the
        // pointer so we go through the registry's row_for_uri interface.
        // First we find the URI by enumerating the namespace; that is
        // O(rows in namespace) but acceptable for this audit-only path.
        let registry = self.registry();
        let ns_name = registry
            .namespace_names()
            .into_iter()
            .find(|n| registry.namespace_id(n) == Some(self.g_lock()))
            .ok_or(BridgeError::NamespaceMissing {
                bridge_id: self.bridge_id_static(),
                namespace: "",
            })?;
        let rows = registry.enumerate(&ns_name);
        rows.into_iter()
            .find(|r| r.public_name == public_name && r.bridge_id == self.bridge_id())
            .ok_or(BridgeError::NotInScope {
                bridge_id: self.bridge_id_static(),
                public_name: public_name.to_string(),
            })
    }

    /// `bridge_id` as `&'static str`. Default just returns the same value
    /// as `bridge_id()`; bridges with non-static identifiers can override.
    fn bridge_id_static(&self) -> &'static str {
        self.bridge_id()
    }
}

/// Pointer to an entity in the dictionary. The hot-path consumer compares
/// `schema_ptr.entity_type_id()` — the GLOBAL template id (DECISION-3),
/// shared across namespaces for the same canonical class. Ids are sparse
/// (monotone with gaps), so compare/lookup by id; never dense-index an
/// array with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityRef {
    pub schema_ptr: SchemaPtr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeRef {
    pub schema_ptr: SchemaPtr,
}

/// Convenience wrapper: converts a generic registry handle into a typed
/// bridge of the requested implementation.
pub fn make_bridge<B>(registry: Arc<OntologyRegistry>) -> Result<B>
where
    B: BridgeFromRegistry,
{
    B::from_registry(registry)
}

/// Implemented by every bridge struct that has a single-arg constructor
/// (registry → Self). All three default tenant bridges in
/// `crate::bridges` implement it.
pub trait BridgeFromRegistry: Sized {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self>;
}
