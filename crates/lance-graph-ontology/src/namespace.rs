//! Namespace + URI + SchemaPtr identity types.
//!
//! `NamespaceId` is the lazy-lock register switch G — one byte per OGIT
//! namespace. `OgitUri` is the fully-qualified canonical name
//! (`ogit.Network:IPAddress`). `SchemaPtr` is a packed pointer of
//! `(namespace_id, entity_type_id, kind_disc)` that the hot-path resolver
//! returns. The packed layout matches the plan's bit-packing convention:
//!
//! ```text
//! SchemaPtr (u32):
//!   bits 31..24 : namespace_id  (u8)
//!   bits 23..8  : entity_type_id (u16, dense within the namespace)
//!   bits  7..0  : kind discriminant (u8)  — Entity / Edge / Attribute
//! ```
//!
//! Carrier-method doctrine: methods live on these types, not free functions.

use crate::error::{Error, Result};

/// G: the lazy-lock register switch. 0 is reserved for "unknown / unbound";
/// 1..=255 are valid namespace ordinals assigned by the registry as TTL
/// hydrates each namespace for the first time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NamespaceId(pub u8);

impl NamespaceId {
    pub const UNKNOWN: NamespaceId = NamespaceId(0);

    pub const fn raw(self) -> u8 {
        self.0
    }

    pub const fn is_known(self) -> bool {
        self.0 != 0
    }
}

/// The fully-qualified OGIT URI for an entity, edge, or attribute.
/// Form: `ogit.<Namespace>:<Name>`. We store as `String` because the
/// dictionary table is dynamic — namespaces can be added at runtime.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OgitUri(String);

impl OgitUri {
    /// Construct an OgitUri without validation. Prefer [`OgitUri::parse`].
    pub fn from_string_unchecked(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Parse and validate an OgitUri. The shape `ogit.<NS>:<Name>` is
    /// enforced; bare strings or empty namespace/name are rejected.
    pub fn parse(s: &str) -> Result<Self> {
        let ns = Self::namespace_part(s).filter(|p| !p.is_empty());
        let name = Self::name_part(s).filter(|p| !p.is_empty());
        if ns.is_some() && name.is_some() {
            Ok(Self(s.to_string()))
        } else {
            Err(Error::InvalidOgitUri(s.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }

    /// Returns `Some("Network")` for `ogit.Network:IPAddress`.
    pub fn namespace(&self) -> Option<&str> {
        Self::namespace_part(&self.0)
    }

    /// Returns `Some("IPAddress")` for `ogit.Network:IPAddress`.
    pub fn name(&self) -> Option<&str> {
        Self::name_part(&self.0)
    }

    fn namespace_part(s: &str) -> Option<&str> {
        let after_prefix = s.strip_prefix("ogit.")?;
        let colon = after_prefix.find(':')?;
        Some(&after_prefix[..colon])
    }

    fn name_part(s: &str) -> Option<&str> {
        let colon = s.find(':')?;
        let after = &s[colon + 1..];
        if after.is_empty() {
            None
        } else {
            Some(after)
        }
    }
}

impl std::fmt::Display for OgitUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Packed schema pointer. Returned from
/// [`crate::OntologyRegistry::resolve`]. The hot path consumer pattern is
/// to compare the `namespace_id()` against the bridge's lock and then use
/// the `entity_type_id()` as the dense local index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SchemaPtr(u32);

impl SchemaPtr {
    pub const fn new(namespace_id: NamespaceId, entity_type_id: u16, kind: SchemaKind) -> Self {
        let packed = ((namespace_id.0 as u32) << 24)
            | ((entity_type_id as u32) << 8)
            | (kind as u32 & 0xFF);
        Self(packed)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Reconstruct a SchemaPtr from its packed `u32`. Used by the Lance
    /// cache when replaying the dictionary on startup.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn namespace_id(self) -> NamespaceId {
        NamespaceId(((self.0 >> 24) & 0xFF) as u8)
    }

    pub const fn entity_type_id(self) -> u16 {
        ((self.0 >> 8) & 0xFFFF) as u16
    }

    pub const fn kind(self) -> SchemaKind {
        match self.0 & 0xFF {
            0 => SchemaKind::Entity,
            1 => SchemaKind::Edge,
            2 => SchemaKind::Attribute,
            _ => SchemaKind::Entity,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SchemaKind {
    Entity = 0,
    Edge = 1,
    Attribute = 2,
}

impl SchemaKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Entity => "entity",
            Self::Edge => "edge",
            Self::Attribute => "attribute",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "entity" => Some(Self::Entity),
            "edge" => Some(Self::Edge),
            "attribute" => Some(Self::Attribute),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ogit_uri_parses_namespace_and_name() {
        let uri = OgitUri::parse("ogit.Network:IPAddress").unwrap();
        assert_eq!(uri.namespace(), Some("Network"));
        assert_eq!(uri.name(), Some("IPAddress"));
    }

    #[test]
    fn ogit_uri_rejects_malformed() {
        assert!(OgitUri::parse("ogit.Network").is_err());
        assert!(OgitUri::parse("Network:IPAddress").is_err());
        assert!(OgitUri::parse("ogit.:Empty").is_err());
        assert!(OgitUri::parse("ogit.Network:").is_err());
    }

    #[test]
    fn schema_ptr_round_trips() {
        let ptr = SchemaPtr::new(NamespaceId(7), 42, SchemaKind::Entity);
        assert_eq!(ptr.namespace_id(), NamespaceId(7));
        assert_eq!(ptr.entity_type_id(), 42);
        assert_eq!(ptr.kind(), SchemaKind::Entity);
    }

    #[test]
    fn schema_ptr_kinds() {
        let entity = SchemaPtr::new(NamespaceId(1), 1, SchemaKind::Entity);
        let edge = SchemaPtr::new(NamespaceId(1), 1, SchemaKind::Edge);
        let attr = SchemaPtr::new(NamespaceId(1), 1, SchemaKind::Attribute);
        assert_ne!(entity.raw(), edge.raw());
        assert_ne!(entity.raw(), attr.raw());
        assert_eq!(entity.kind(), SchemaKind::Entity);
        assert_eq!(edge.kind(), SchemaKind::Edge);
        assert_eq!(attr.kind(), SchemaKind::Attribute);
    }
}
