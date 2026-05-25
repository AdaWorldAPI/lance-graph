# 3DGS ArcGIS and Cesium Ingestion Plan — lance-graph

## Goal

Define how `lance-graph` ingests external geospatial sources into the 3DGS runtime model.

This plan focuses on source acquisition and normalization, not rendering.

## Source classes

### 1. 3D Tiles / Cesium-compatible packages

Inputs:

- `tileset.json`
- subtree files
- glTF / GLB payloads
- SPZ / Gaussian splat glTF extensions where available
- metadata and batch/property tables

Outputs:

- `tiles` table
- `contents` table
- `subtrees` table
- `features` table where metadata exists
- raw content resource references

### 2. ArcGIS REST services

Inputs:

- FeatureServer metadata and query output
- MapServer metadata and query output
- VectorTileServer metadata, styles, and tiles
- Scene/3D layer references where available
- service directory JSON / PJSON

Outputs:

- vector overlays
- semantic features
- basemap/vector-tile references
- optional source provenance rows

### 3. glTF / GLB content

Inputs:

- mesh payloads
- splat-related extensions
- metadata extensions

Outputs:

- content refs
- derived bounds
- feature metadata where available
- optional conversion to 3DGS blocks if supported

### 4. GeoJSON / MVT overlays

Inputs:

- GeoJSON feature collections
- Mapbox Vector Tile payloads
- ArcGIS query output converted to GeoJSON

Outputs:

- `features` table
- overlay content refs
- queryable geometry metadata

### 5. Native 3DGS payloads

Inputs:

- splat point records
- SPZ-like compressed payloads
- custom columnar splat blocks

Outputs:

- `contents` rows with kind `GaussianSplat3d`
- `splat_blocks` rows
- optional raw payload files or Lance column chunks

## Normalization stages

```text
source fetch/read
  -> source manifest
  -> bounds and CRS detection
  -> content kind classification
  -> normalized tile/content rows
  -> optional feature extraction
  -> optional certificate preflight
  -> Lance/Arrow write
```

## Coordinate policy

Every ingest path must record coordinate assumptions:

- CRS / spatial reference if known
- local origin if converted
- ECEF / ENU / projected frame where applicable
- transform applied
- uncertainty if unknown

Do not silently treat unknown coordinates as WGS84.

## Provenance

Add a provenance row/table or metadata object:

```text
source_id
source_kind
source_uri
retrieved_at
source_version
license_hint
crs_hint
conversion_notes
```

This matters for debugging and for later ArcGIS/Cesium export.

## Ingestion crate layout

```text
crates/ada-geo-ingest
  mod.rs
  source_manifest.rs
  arcgis.rs
  cesium_3dtiles.rs
  gltf.rs
  geojson.rs
  mvt.rs
  native_3dgs.rs
  provenance.rs
```

## ArcGIS-first test endpoints

Use public/sample services for tests that do not require credentials:

- sample service directory root
- FeatureServer query result
- MapServer layer query result
- VectorTile metadata if available without secrets

Credentialed services should be integration-test optional and skipped by default.

## Acceptance criteria

- A 3D Tiles tileset can be ingested into tile/content tables without loading all payload bytes.
- ArcGIS FeatureServer query output can become features/overlay rows.
- GeoJSON can become features rows.
- Every ingested source records provenance.
- Unknown CRS is explicit, not guessed.
- 3DGS payloads become first-class content rows.
