//! Canonical, fixed-width identities — the only names the execution core uses.
//!
//! **Strings are presentation metadata, not execution coordinates.** A textual
//! field name resolves to a [`FieldId`] at the boundary (the catalog); a
//! textual categorical value resolves to an ordinal in that field's domain
//! (the CAM label store). After that, nothing variable-length survives into a
//! plan, a selection, a coordinate, a mask, a fold or a cell.

/// A field (lane) identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldId(pub u32);

/// A resident mask identity. [`MaskId::ALPHA`] is the validity plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaskId(pub u32);

impl MaskId {
    /// The validity plane every batch carries.
    pub const ALPHA: MaskId = MaskId(0);
}

/// A published source identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(pub u32);

impl core::fmt::Display for FieldId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "F{}", self.0)
    }
}

impl core::fmt::Display for MaskId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "M{}", self.0)
    }
}

impl core::fmt::Display for SourceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "S{}", self.0)
    }
}
