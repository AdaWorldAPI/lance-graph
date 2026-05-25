//! `ada-3dtiles` — a reader for OGC 3D Tiles `tileset.json` documents (1.0 and 1.1).
//!
//! This is Phase 1 of the geospatial 3DGS rebuild described in
//! `.claude/plans/3DGS-3D-Tiles-runtime-plan.md`: parse explicit and implicit
//! tilesets into typed DTOs without dragging in renderer, traversal-policy, or
//! storage concerns. Those live in separate crates per the cross-repo boundary:
//!
//! - **traversal / screen-space-error policy** → `ada-3dtiles-selection` (future)
//! - **Lance/Arrow sidecar storage** → `ada-geo-lance` (future)
//! - **SIMD culling / 3DGS projection / certificates** → `ndarray::hpc::splat3d`
//!   + `ndarray::hpc::pillar`
//!
//! # Scope
//!
//! - Tolerant parsing: unknown `extensions` / `extras` are preserved losslessly
//!   so vendor data (`3DTILES_content_gltf`, `ESRI_crs`, S2 bounding volumes) is
//!   not dropped.
//! - Both content forms: the 1.0 single `content` and the 1.1 `contents` array
//!   are unified via [`Tile::all_contents`].
//! - `refine` inheritance is resolved during preorder traversal
//!   ([`Tileset::visit_preorder`]).
//! - Implicit tiling descriptors are parsed; subtree-file expansion is a later
//!   phase.
//!
//! # Example
//!
//! ```
//! let json = r#"{
//!   "asset": { "version": "1.1" },
//!   "geometricError": 100.0,
//!   "root": {
//!     "boundingVolume": { "box": [0,0,0, 1,0,0, 0,1,0, 0,0,1] },
//!     "geometricError": 0.0,
//!     "refine": "ADD",
//!     "content": { "uri": "model.glb" }
//!   }
//! }"#;
//!
//! let tileset = ada_3dtiles::from_str(json).unwrap();
//! assert_eq!(tileset.version(), "1.1");
//! assert_eq!(tileset.tile_count(), 1);
//! assert_eq!(tileset.root.content.as_ref().unwrap().location(), Some("model.glb"));
//! ```

mod error;
mod tileset;

pub use error::{Error, Result};
pub use tileset::{
    Asset, BoundingVolume, BoundingVolumeKind, Content, ImplicitTiling, Refine, SubdivisionScheme,
    Subtrees, Tile, Tileset,
};

/// Parse a tileset document from a JSON string.
pub fn from_str(s: &str) -> Result<Tileset> {
    Ok(serde_json::from_str(s)?)
}

/// Parse a tileset document from raw JSON bytes (e.g. the body of a
/// `tileset.json` HTTP response or a file read).
pub fn from_slice(bytes: &[u8]) -> Result<Tileset> {
    Ok(serde_json::from_slice(bytes)?)
}
