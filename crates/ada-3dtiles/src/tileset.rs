//! 3D Tiles tileset document model (1.0 and 1.1).
//!
//! These DTOs follow the OGC 3D Tiles specification. They are intentionally
//! tolerant: unknown fields carried under `extensions` / `extras` are preserved
//! as raw [`serde_json::Value`] so that round-tripping a tileset does not drop
//! vendor extension data (e.g. `3DTILES_content_gltf`, `ESRI_crs`,
//! `3DTILES_bounding_volume_S2`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Top-level tileset document (`tileset.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tileset {
    /// Metadata about the tileset (version, optional application version).
    pub asset: Asset,

    /// Error, in meters, introduced if this tileset is not rendered at all.
    /// Used to size screen-space error against the root.
    #[serde(rename = "geometricError")]
    pub geometric_error: f64,

    /// The root tile of the hierarchy.
    pub root: Tile,

    /// Names of 3D Tiles extensions used somewhere in the tileset.
    #[serde(rename = "extensionsUsed", skip_serializing_if = "Option::is_none")]
    pub extensions_used: Option<Vec<String>>,

    /// Names of extensions a client must support to load the tileset.
    #[serde(rename = "extensionsRequired", skip_serializing_if = "Option::is_none")]
    pub extensions_required: Option<Vec<String>>,

    /// 1.0 `properties` block, retained verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Value>,

    /// Extension objects keyed by extension name (lossless passthrough).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,

    /// Application-specific data (lossless passthrough).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<Value>,
}

impl Tileset {
    /// The declared 3D Tiles asset version (e.g. `"1.0"` or `"1.1"`).
    pub fn version(&self) -> &str {
        &self.asset.version
    }

    /// Returns `true` if the document declares any required extension.
    pub fn has_required_extensions(&self) -> bool {
        self.extensions_required
            .as_ref()
            .is_some_and(|v| !v.is_empty())
    }

    /// Visit every tile in preorder (root first, then children left to right).
    ///
    /// The visitor receives each tile together with its depth (root = 0) and the
    /// [`Refine`] mode in effect for that tile, with parent inheritance already
    /// resolved per the spec.
    pub fn visit_preorder<F: FnMut(&Tile, usize, Refine)>(&self, mut visit: F) {
        fn walk<F: FnMut(&Tile, usize, Refine)>(
            tile: &Tile,
            depth: usize,
            inherited: Refine,
            visit: &mut F,
        ) {
            let refine = tile.refine.unwrap_or(inherited);
            visit(tile, depth, refine);
            if let Some(children) = &tile.children {
                for child in children {
                    walk(child, depth + 1, refine, visit);
                }
            }
        }
        // The root MUST declare a refine mode; default to ADD if a malformed
        // tileset omits it rather than failing the whole traversal.
        let root_refine = self.root.refine.unwrap_or(Refine::Add);
        walk(&self.root, 0, root_refine, &mut visit);
    }

    /// Total number of tile nodes in the hierarchy (explicit tiles only;
    /// implicit subtrees are not expanded here).
    pub fn tile_count(&self) -> usize {
        let mut n = 0;
        self.visit_preorder(|_, _, _| n += 1);
        n
    }
}

/// `asset` metadata block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    /// The 3D Tiles version the tileset uses.
    pub version: String,

    /// Application-specific tileset version string.
    #[serde(rename = "tilesetVersion", skip_serializing_if = "Option::is_none")]
    pub tileset_version: Option<String>,

    /// Extension passthrough.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,

    /// Extras passthrough.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<Value>,
}

/// A single node in the tile hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tile {
    /// The spatial extent enclosing the tile's content and children.
    #[serde(rename = "boundingVolume")]
    pub bounding_volume: BoundingVolume,

    /// Error, in meters, introduced if this tile's children are not rendered.
    #[serde(rename = "geometricError")]
    pub geometric_error: f64,

    /// Refinement mode. Required on the root; inherited from the parent
    /// otherwise. Use [`Tileset::visit_preorder`] to get the resolved value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refine: Option<Refine>,

    /// Optional 4x4 column-major transform applied to this tile and its
    /// descendants.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform: Option<[f64; 16]>,

    /// A bounding volume the viewer must be inside for content to render.
    #[serde(
        rename = "viewerRequestVolume",
        skip_serializing_if = "Option::is_none"
    )]
    pub viewer_request_volume: Option<BoundingVolume>,

    /// Single content payload (1.0 form, also valid in 1.1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,

    /// Multiple content payloads (1.1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contents: Option<Vec<Content>>,

    /// Child tiles.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<Tile>>,

    /// Implicit tiling descriptor (1.1).
    #[serde(rename = "implicitTiling", skip_serializing_if = "Option::is_none")]
    pub implicit_tiling: Option<ImplicitTiling>,

    /// Extension passthrough.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,

    /// Extras passthrough.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<Value>,
}

impl Tile {
    /// All content payloads on this tile, unifying the 1.0 `content` field and
    /// the 1.1 `contents` array into a single iterator.
    pub fn all_contents(&self) -> impl Iterator<Item = &Content> {
        self.content.iter().chain(self.contents.iter().flatten())
    }

    /// `true` if this tile carries any content payload.
    pub fn has_content(&self) -> bool {
        self.content.is_some() || self.contents.as_ref().is_some_and(|c| !c.is_empty())
    }

    /// `true` if this tile is an implicit-tiling root.
    pub fn is_implicit_root(&self) -> bool {
        self.implicit_tiling.is_some()
    }
}

/// Refinement strategy for a tile's children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Refine {
    /// Children are rendered in addition to this tile's content.
    #[serde(rename = "ADD")]
    Add,
    /// Children replace this tile's content when refined.
    #[serde(rename = "REPLACE")]
    Replace,
}

/// Tile content reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    /// Content URI (1.1). Relative to the tileset document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,

    /// Legacy content URL (1.0). Deprecated alias for [`Content::uri`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Optional tighter bounding volume for just this content.
    #[serde(rename = "boundingVolume", skip_serializing_if = "Option::is_none")]
    pub bounding_volume: Option<BoundingVolume>,

    /// Metadata group index (1.1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<u64>,

    /// Extension passthrough.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,

    /// Extras passthrough.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<Value>,
}

impl Content {
    /// The effective content location, preferring the 1.1 `uri` and falling
    /// back to the 1.0 `url`.
    pub fn location(&self) -> Option<&str> {
        self.uri.as_deref().or(self.url.as_deref())
    }
}

/// Bounding volume. Exactly one geometric variant is set per the spec, but a
/// volume may instead be supplied by an extension (e.g. S2), in which case all
/// three are `None` and the data lives under [`BoundingVolume::extensions`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingVolume {
    /// Oriented bounding box: `[cx, cy, cz, xHalf(3), yHalf(3), zHalf(3)]`.
    #[serde(rename = "box", skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f64; 12]>,

    /// Geographic region: `[west, south, east, north, minHeight, maxHeight]`
    /// (longitude/latitude in radians, heights in meters).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<[f64; 6]>,

    /// Bounding sphere: `[centerX, centerY, centerZ, radius]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sphere: Option<[f64; 4]>,

    /// Extension passthrough (e.g. `3DTILES_bounding_volume_S2`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,

    /// Extras passthrough.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<Value>,
}

/// Which geometric form a [`BoundingVolume`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundingVolumeKind {
    /// Oriented box.
    Box,
    /// Geographic region.
    Region,
    /// Sphere.
    Sphere,
    /// Supplied by an extension; geometric fields are absent.
    Extension,
}

impl BoundingVolume {
    /// Classify which geometric form is present.
    pub fn kind(&self) -> BoundingVolumeKind {
        if self.bbox.is_some() {
            BoundingVolumeKind::Box
        } else if self.region.is_some() {
            BoundingVolumeKind::Region
        } else if self.sphere.is_some() {
            BoundingVolumeKind::Sphere
        } else {
            BoundingVolumeKind::Extension
        }
    }
}

/// Implicit tiling descriptor (3D Tiles 1.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplicitTiling {
    /// How a tile is subdivided into children.
    #[serde(rename = "subdivisionScheme")]
    pub subdivision_scheme: SubdivisionScheme,

    /// Number of distinct levels in each subtree.
    #[serde(rename = "subtreeLevels")]
    pub subtree_levels: u32,

    /// Number of available levels (1.1). Older drafts used `maximumLevel`.
    #[serde(rename = "availableLevels", skip_serializing_if = "Option::is_none")]
    pub available_levels: Option<u32>,

    /// Legacy maximum level (pre-1.1 drafts).
    #[serde(rename = "maximumLevel", skip_serializing_if = "Option::is_none")]
    pub maximum_level: Option<u32>,

    /// Template URI from which subtree files are addressed.
    pub subtrees: Subtrees,
}

impl ImplicitTiling {
    /// Number of children each subdivision produces (4 for quadtree, 8 for
    /// octree).
    pub fn children_per_tile(&self) -> u32 {
        match self.subdivision_scheme {
            SubdivisionScheme::Quadtree => 4,
            SubdivisionScheme::Octree => 8,
        }
    }
}

/// Subdivision scheme for implicit tiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubdivisionScheme {
    /// 2D quadtree (4 children).
    #[serde(rename = "QUADTREE")]
    Quadtree,
    /// 3D octree (8 children).
    #[serde(rename = "OCTREE")]
    Octree,
}

/// Subtree addressing for implicit tiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtrees {
    /// Template URI with `{level}`, `{x}`, `{y}`, and (octree) `{z}` tokens.
    pub uri: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
    {
      "asset": { "version": "1.1", "tilesetVersion": "site-2026-05" },
      "geometricError": 512.0,
      "extensionsUsed": ["3DTILES_content_gltf"],
      "root": {
        "boundingVolume": { "box": [0,0,0, 100,0,0, 0,100,0, 0,0,20] },
        "geometricError": 256.0,
        "refine": "REPLACE",
        "content": { "uri": "root.glb" },
        "children": [
          {
            "boundingVolume": { "region": [-1.30, 0.69, -1.29, 0.70, 0.0, 30.0] },
            "geometricError": 64.0,
            "contents": [
              { "uri": "child_a.glb" },
              { "uri": "child_a_splat.spz", "group": 1 }
            ]
          },
          {
            "boundingVolume": { "sphere": [10, 10, 5, 50] },
            "geometricError": 0.0,
            "refine": "ADD",
            "content": { "url": "legacy_child.b3dm" }
          }
        ]
      }
    }
    "#;

    fn parse() -> Tileset {
        serde_json::from_str(SAMPLE).expect("sample tileset parses")
    }

    #[test]
    fn parses_asset_and_root() {
        let ts = parse();
        assert_eq!(ts.version(), "1.1");
        assert_eq!(ts.asset.tileset_version.as_deref(), Some("site-2026-05"));
        assert_eq!(ts.geometric_error, 512.0);
        assert_eq!(ts.root.geometric_error, 256.0);
        assert_eq!(ts.tile_count(), 3);
    }

    #[test]
    fn bounding_volume_kinds() {
        let ts = parse();
        assert_eq!(ts.root.bounding_volume.kind(), BoundingVolumeKind::Box);
        let children = ts.root.children.as_ref().unwrap();
        assert_eq!(
            children[0].bounding_volume.kind(),
            BoundingVolumeKind::Region
        );
        assert_eq!(
            children[1].bounding_volume.kind(),
            BoundingVolumeKind::Sphere
        );
    }

    #[test]
    fn refine_inheritance() {
        let ts = parse();
        let mut modes = Vec::new();
        ts.visit_preorder(|_, depth, refine| modes.push((depth, refine)));
        // root REPLACE; child[0] inherits REPLACE; child[1] overrides to ADD.
        assert_eq!(modes[0], (0, Refine::Replace));
        assert_eq!(modes[1], (1, Refine::Replace));
        assert_eq!(modes[2], (1, Refine::Add));
    }

    #[test]
    fn content_uri_and_legacy_url() {
        let ts = parse();
        let children = ts.root.children.as_ref().unwrap();

        // 1.1 multiple contents.
        let multi: Vec<_> = children[0].all_contents().collect();
        assert_eq!(multi.len(), 2);
        assert_eq!(multi[0].location(), Some("child_a.glb"));
        assert_eq!(multi[1].group, Some(1));

        // 1.0 legacy `url` resolves through location().
        let legacy: Vec<_> = children[1].all_contents().collect();
        assert_eq!(legacy.len(), 1);
        assert_eq!(legacy[0].location(), Some("legacy_child.b3dm"));
    }

    #[test]
    fn unknown_extension_is_preserved() {
        let json = r#"
        {
          "asset": { "version": "1.1" },
          "geometricError": 1.0,
          "root": {
            "boundingVolume": {
              "extensions": { "3DTILES_bounding_volume_S2": { "token": "89c284" } }
            },
            "geometricError": 0.0,
            "refine": "ADD"
          }
        }"#;
        let ts: Tileset = serde_json::from_str(json).unwrap();
        assert_eq!(
            ts.root.bounding_volume.kind(),
            BoundingVolumeKind::Extension
        );
        // Round-trip keeps the extension payload intact.
        let reser = serde_json::to_string(&ts).unwrap();
        assert!(reser.contains("3DTILES_bounding_volume_S2"));
        assert!(reser.contains("89c284"));
    }

    #[test]
    fn implicit_tiling() {
        let json = r#"
        {
          "asset": { "version": "1.1" },
          "geometricError": 100.0,
          "root": {
            "boundingVolume": { "box": [0,0,0, 1,0,0, 0,1,0, 0,0,1] },
            "geometricError": 50.0,
            "refine": "REPLACE",
            "content": { "uri": "content/{level}/{x}/{y}.glb" },
            "implicitTiling": {
              "subdivisionScheme": "QUADTREE",
              "subtreeLevels": 7,
              "availableLevels": 21,
              "subtrees": { "uri": "subtrees/{level}/{x}/{y}.subtree" }
            }
          }
        }"#;
        let ts: Tileset = serde_json::from_str(json).unwrap();
        let it = ts.root.implicit_tiling.as_ref().unwrap();
        assert_eq!(it.subdivision_scheme, SubdivisionScheme::Quadtree);
        assert_eq!(it.children_per_tile(), 4);
        assert_eq!(it.subtree_levels, 7);
        assert_eq!(it.available_levels, Some(21));
        assert!(ts.root.is_implicit_root());
    }
}
