# cesium-osm-substrate-v1 — integration plan (debt + future)

> **Status:** PROPOSAL / integration plan. Design-spec only; **no code in this plan**.
> **Authored:** 2026-06-05 (session `claude/lance-graph-ontology-review-Pyry3`).
> **Trigger:** user-supplied feasibility question on OpenStreetMap + Cesium + Gaussian-splat coupling, cross-cutting the existing `3DGS-ArcGIS-Cesium-ingestion-plan.md` (parent / structural) and `splat-native-ultrasound-v1.md` (substrate-sibling).
>
> **Anchored to (FINDING-grade):** `E-SOA-IS-THE-ONLY` (one substrate, three operations — fit/accumulate/render — shared across ultrasound + geospatial), `I-VSA-IDENTITIES` (OSM tag-class identity fingerprints, not bundle of tag content), `I-LEGACY-API-FEATURE-GATED` (OSM-XYZ vs Cesium-TMS Y-axis disambiguation handled at boundary), `I-NOISE-FLOOR-JIRAK` (any geocode-significance claims under weakly-dependent OSM tag distributions).
>
> **Cross-workspace coordination:** OGAR session signed off on Q1 (Tag-as-Class + v1 fallback) + Q2 (Cesium TMS quadkey NiblePath) in the 2026-06-05 exchange recorded in §2. OGAR-side `ogar-from-osm-pbf` adapter (Phase 2c in `RDF-OWL-ALIGNMENT.md §10`) consumes D-OSM-3 by ID; OGAR docs PR queued behind this addendum.

---

## 0. Executive summary (one screen)

OpenStreetMap as a 6th source class for the existing 3DGS-ArcGIS-Cesium ingestion plan. Same shape as the other five (3D Tiles, ArcGIS, glTF, GeoJSON/MVT, native 3DGS): a parser → Arrow RecordBatch → Lance Dataset → SPO triples in `lance-graph` → cesium tileset → splat renderer. Substrate (Gaussian3D carrier + ndarray SIMD ops + Lance storage) shared with `splat-native-ultrasound-v1`; **scenes diverge, substrate is singular**.

The Rust ecosystem makes only **`osmpbf`** (b-r-u, multi-year, lazy-decoded, parallelized) production-ready. The genuine gap is a Cesium-3D-Tiles **writer** (b3dm/i3dm/cmpt) — no Rust crate exists; **D-OSM-6 is the new code on the critical path**. Everything else is glue over crates that already work (`osmpbf`, `georaster`, `srtm_reader`, `gltf`).

**Total new deliverables: D-OSM-1..7** (seven). Two design decisions ratified in coordination with OGAR session (§2). Three out-of-scope items called out (geocoding sidecar, vector-tile generation, OSM-XYZ Y-axis adjustment).

---

## 1. Relationship to existing plans

```text
3DGS-ArcGIS-Cesium-ingestion-plan.md ──┐  (parent / structural — 5 source classes)
                                       │
splat-native-ultrasound-v1.md ─────────┤  (substrate-sibling — Gaussian3D carrier, SIMD ops, Lance storage)
                                       │
                                       ▼
                          cesium-osm-substrate-v1.md ← this doc
                          (6th source class + D-OSM-1..7)
```

- **Parent.** `3DGS-ArcGIS-Cesium-ingestion-plan.md` (lance-graph) is the structural sketch — defines 5 source classes (3D Tiles, ArcGIS, glTF, GeoJSON/MVT, native 3DGS), the normalization-stages pipeline, the provenance discipline, the coordinate-policy rule (*"do not silently treat unknown coordinates as WGS84"*), and acceptance criteria. This addendum extends it without modifying it.
- **Sibling.** `splat-native-ultrasound-v1.md` (lance-graph) ships D-SPLAT-1..14. The **carrier reuse** (D-SPLAT-1 `Gaussian3D`, D-SPLAT-2 SIMD ops, D-SPLAT-3 `SplatBatch` SoA, D-SPLAT-12 `splat-render`) is what makes the cross-arc reuse cheap. Two scenes (anatomy, geography), one substrate.
- **Promote-when** rule. If D-OSM-* surface grows past 5 distinct deliverable families (geocoding adapter, vector-tile generator, terrain integration, etc.), this addendum splits into its own top-level plan. Convention: grow before fork.

---

## 2. OGAR coordination — Q1 + Q2 rulings (locked 2026-06-05)

The OGAR session raised two design calls before scaffolding `ogar-from-osm-pbf` (Phase 2c per `RDF-OWL-ALIGNMENT.md §10`). Both are resolved.

### Q1 — How do OSM tags map to OGAR IR? **Tag-as-Class (option c) with v1 Arrow-list fallback (option b).**

OSM tags are open-keyspace key-value pairs (any string ↔ string). Three options were considered:
- (a) Synthetic Attribute per tag — **rejected.** Breaks the fixed-schema Lance path; forces sparse-wide or JSON-blob columns; the Cesium tile loader can't index a moving schema cheaply.
- (b) `tags: List<Struct<key, value>>` Arrow column — **v1 fallback.** Acceptable Arrow type; works with DataFusion + Lance natively; one IR field change; minimal substrate-wide consult.
- (c) `Tag` as separate Class with `has_tag: HasMany<Tag>` association — **end-state.** Matches the SPO/triple-emission pattern OGAR is built on. Tag interning + canonical-key normalization caps cardinality at ~10⁵ unique key=value pairs on the planet (not ~10¹¹ triples). Queries like *"all Ways tagged building=yes near here"* become a single triple lookup.

**Ruling.** Ship (b) in D-OSM-1/2 implementation as the wire shape for v1; migrate to (c) in a follow-up sprint when Tag interning is implemented. Lance stores both natively; runtime cost difference is negligible for the LOD-paged query patterns.

### Q2 — How does NiblePath prefix the OSM ID space? **Cesium TMS quadkey.**

Five options were considered:
- Flat `osm/node/42` — **rejected.** No HHTL locality.
- OSM-XYZ slippy-map quadkey — **rejected** (see Q3 below).
- Geohash `osm/u281xv5/node/42` — **deferred** to secondary index. Spatial-locality prefix is right for "find nearby" queries; runs as a Nominatim-style sidecar service if needed.
- Admin hierarchy `osm/de/by/munich/marienplatz/...` — **deferred** to UX-edge. Semantic addressing belongs in geocoding (Nominatim) not in the primary identity path.
- **Cesium TMS quadkey** `osm/qk:<level>/<x>/<y>/<type>/<id>` (octree variant `osm/ok:<level>/<x>/<y>/<z>/<type>/<id>` when vertical resolution matters) — **chosen.**

**Ruling.** `crates/cesium/src/implicit_tiling.rs` + `sse.rs` + `hlod.rs` (all already shipped in ndarray) consume Cesium subtree-style coordinates. Aligning OSM identity with Cesium tile coordinates makes *"give me all OSM Ways inside this Cesium tile"* a single NiblePath prefix scan — the HHTL payoff. Same compile-time HHTL primitive that resolves `Femur is_a LongBone` for FMA resolves `Marienplatz is_in Munich` for OSM.

### Q3 — OSM-XYZ vs Cesium-TMS Y-axis (raised in coordination, must be documented)

OSM slippy-map quadkeys are **XYZ-ordered** (Y top-down, web standard); Cesium's `implicit_tiling` is **TMS-style** (Y bottom-up). They are **not the same key**.

**Ruling.** Use **Cesium TMS internal** for NiblePath prefix-equality with subtree IDs. Document the OSM-XYZ → TMS Y-flip at the ingest boundary (one subtract per Way). This is the boundary mapping; it is NOT a runtime path. The flip is per `I-LEGACY-API-FEATURE-GATED` (legacy-API version-gate pattern) — when the wire format changes, the boundary converts; the runtime sees only one shape.

### Coordination outcome

OGAR session queued action (`ogar-from-osm-pbf` Phase 2c scaffold) is unblocked. Their docs PR (`DOMAIN-INSTANCES.md §2.6` + `RDF-OWL-ALIGNMENT.md §10 Phase 2c`) lands after this addendum so they cite **D-OSM-1..7** by ID. Cross-arc reuse from `splat-native-ultrasound-v1.md` (D-SPLAT-1/2/3/12) is named explicitly in §3.

---

## 3. Per-deliverable specifications

### D-OSM-1 — `crates/cesium/src/osm_pbf.rs` (ndarray)

Mirror the shape of `crates/cesium/src/arcgis_pbf.rs` (428 LOC, already shipped). Stub types for the OSM PBF data model (`OsmNode`, `OsmWay`, `OsmRelation`, `OsmTagList`, `OsmPbfHeader`, `OsmPbfBlock`), a `decode_pbf(&[u8]) -> Result<OsmPbfBlock, OsmPbfError>` skeleton, and the OSM-XYZ → TMS Y-flip boundary helper. **No osmpbf dependency yet** — wired in D-OSM-2.

- Repo: `ndarray`
- LOC: ~400 (mirrors arcgis_pbf.rs)
- Risk: LOW
- Sprint: P1 sprint 1
- Gates on: nothing (foundation)

### D-OSM-2 — `osmpbf` consumer + Arrow RecordBatch emitter (lance-graph)

Wire `osmpbf` v0.4 (b-r-u) into a **lance-graph-side** module — e.g. `crates/lance-graph/src/ingest/osm_pbf.rs` (new) — **not** the ndarray `crates/cesium/src/osm_pbf.rs` file, which stays the dependency-free D-OSM-1 stub. The `osmpbf` dependency and the Arrow/Lance emitter live entirely on the lance-graph side; this module consumes the D-OSM-1 carrier shapes (`OsmNode` / `OsmWay` / `OsmRelation` / `OsmTagList`) and the XYZ→TMS Y-flip helper, mapping each lazily-decoded, parallelized PBF element into a per-element-type Arrow RecordBatch:

- `osm_nodes` table — `(id u64, lat f64, lon f64, qk_tms_path FixedSizeBinary(24), tags List<Struct<key,value>>)`
- `osm_ways` table — `(id u64, ref_node_ids List<u64>, qk_tms_path FixedSizeBinary(24), tags List<Struct<key,value>>)`
- `osm_relations` table — `(id u64, members List<Struct<ref_id, role, kind>>, tags List<Struct<key,value>>)`

Tags column is **option (b)** per Q1 ruling. NiblePath `qk_tms_path` per Q2 ruling, computed at ingest via Q3 boundary helper.

- Repo: `lance-graph`
- LOC: ~600
- Risk: MED (osmpbf v0.4 API stability)
- Sprint: P1 sprint 1-2
- Gates on: D-OSM-1

### D-OSM-3 — OSM tag → SPO triple lift (lance-graph-ontology)

Emit OSM tags as SPO triples consumable by `ogar-from-osm-pbf` (OGAR Phase 2c):

```text
(Way#123, ogar:hasTag, "building=yes")
(Way#123, ogar:hasTag, "height=12.5")
(Node#42, ogar:atLocation, "lat,lon")
(Way#123, ogar:withinTile, "qk_tms:<level>/<x>/<y>")
```

This is the **OGAR-crossing deliverable** — the SPO contract the `ogar-from-osm-pbf` adapter consumes. OGAR session signed off on this surface; their Phase 2c work waits for D-OSM-3's contract to land before scaffolding.

When Q1 (c) lands (Tag-as-Class migration), the lift becomes `(Way#123, ogar:hasTag, Tag#building=yes)` with `Tag` as a separate interned class. Backward-compatible — the v1 tag-literal form is a special case of the v2 Tag-identity form (via `Tag::from_literal("building=yes")`).

- Repo: `lance-graph-ontology`
- LOC: ~200
- Risk: LOW
- Sprint: P2 sprint 3
- Gates on: D-OSM-2 + OGAR session readiness signal

### D-OSM-4 — `ndarray::simd::dem` batched DEM-height sampling (ndarray)

W1c primitive-addition contract (mirrors D-SPLAT-2's pattern). One new SIMD primitive:

```rust
pub fn batched_sample_height(
    dem_grid: &[f32],          // height grid in row-major, length = rows * cols
    rows: u32,
    cols: u32,
    bbox: [f64; 4],            // [min_lat, min_lon, max_lat, max_lon] in WGS84
    query_xy: &[f64],          // length 2*M — query points [lat, lon] interleaved
    out_heights: &mut [f32],   // length M — sampled heights
);
```

Bilinear interpolation. All three backends mandatory (AVX-512 / NEON / scalar) per the existing consumer-contract knowledge doc.

- Repo: `ndarray`
- LOC: ~300
- Risk: MED (boundary cases for query outside bbox)
- Sprint: P2 sprint 3
- Gates on: nothing (foundation)

### D-OSM-5 — Geospatial splat-fit: OSM footprint × DEM → extruded Gaussians

The geospatial analog of D-SPLAT-6 (ultrasound splat-fit). Takes OSM `building=yes` Ways + DEM-sampled vertex heights → extruded anisotropic Gaussian volumes. **Emits the SAME `Gaussian3D` carrier** from `lance-graph-contract::splat::Gaussian3D` (D-SPLAT-1) — this is the substrate-reuse payoff.

```text
OSM Way (building footprint) → polygon
DEM sampled at vertices (D-OSM-4) → height field
Polygon × heights → extruded prism
Prism voxelization → anisotropic Gaussians
Gaussian3D batch → SplatBatch (D-SPLAT-3)
```

Lives in either `crates/splat-fit-geo` (new sibling to `crates/splat-fit` from D-SPLAT-6) or as a `geo` feature flag on `crates/splat-fit`. Recommend the feature-flag form to keep substrate types unified.

- Repo: new `crates/splat-fit` extension OR `crates/splat-fit-geo`
- LOC: ~800
- Risk: MED-HIGH (prism-voxelization math)
- Sprint: P3 sprint 4-5
- Gates on: D-OSM-1 + D-OSM-2 + D-OSM-4 + D-SPLAT-1 + D-SPLAT-3

### D-OSM-6 — `cesium-3dtiles-writer` crate (the genuine Rust gap)

**The new code on the critical path.** No Rust crate produces Cesium 3D Tiles (b3dm/i3dm/cmpt) today. CesiumGS' official `3d-tiles-tools` is TypeScript; `kiselev-dv/osm-cesium-3d-tiles` is Java. We build the writer.

Building blocks that exist:
- `gltf` Rust crate (read/write glTF 2.0) — handles the glTF half.
- b3dm/i3dm/cmpt headers are 28-byte structs — `bincode`/`byteorder` over `gltf`.
- `crates/cesium/src/tileset.rs` + `implicit_tiling.rs` (already shipped) consume tileset JSON; this writer EMITS it.

Surface (~minimal v1):
```rust
pub fn write_b3dm(gltf: &gltf::Glb, feature_table: &FeatureTable, batch_table: &BatchTable) -> Vec<u8>;
pub fn write_cmpt(payloads: &[CmptPayload]) -> Vec<u8>;
pub fn write_tileset_json(root: &TilesetRoot) -> String;
```

- Repo: `ndarray` (new `crates/cesium-3dtiles-writer` or as a `writer` feature flag on existing `crates/cesium`)
- LOC: ~500
- Risk: HIGH (genuinely new crate; first-of-its-kind in Rust)
- Sprint: P3 sprint 4-5
- Gates on: D-OSM-5 (feeds the writer with Gaussians + features) + D-SPLAT-3 (SplatBatch SoA)

### D-OSM-7 — Nominatim sidecar adapter (optional, UX-edge)

HTTP client for Nominatim geocoding (address → lat/lon) + reverse-geocoding (lat/lon → address) as an external service. Per AFI blog stack (Nominatim = PostgreSQL+PostGIS; Photon = Elasticsearch). **NOT a substrate concern** — runs as a sidecar, called via `reqwest`, response decoded to OSM IDs that route through D-OSM-2's primary path. No Rust reimplementation of Nominatim/Photon (they're SQL+ES, not Rust-shaped).

- Repo: `lance-graph` (small adapter module) or new `crates/nominatim-client`
- LOC: ~150
- Risk: LOW
- Sprint: P4 sprint 6 (optional; ship only when UX-edge demands it)
- Gates on: nothing (independent path)

---

## 4. Phase sequencing

| Phase | Sprint window | Deliverables | Risk |
|---|---|---|---|
| **P1 — Substrate** | 1-2 | D-OSM-1 (ndarray cesium stub), D-OSM-2 (osmpbf + Arrow ingest), D-OSM-4 (SIMD DEM sample) | MED |
| **P2 — SPO contract** | 3 | D-OSM-3 (tag → SPO lift; unblocks OGAR Phase 2c) | LOW |
| **P3 — Splat-fit + 3D Tiles** | 4-5 | D-OSM-5 (OSM × DEM → Gaussian3D), D-OSM-6 (`cesium-3dtiles-writer` — Rust gap) | HIGH |
| **P4 — UX-edge** | 6+ | D-OSM-7 (Nominatim sidecar; optional) | LOW |

Critical path: **D-OSM-1 → D-OSM-2 → D-OSM-5 → D-OSM-6**. D-OSM-3 unblocks OGAR; D-OSM-4 unblocks D-OSM-5. D-OSM-7 ships only on UX-edge demand.

---

## 5. Dependencies graph (textual)

```text
b-r-u/osmpbf v0.4 ────────┐
                          ▼
                D-OSM-1 (ndarray cesium stub) ──► D-OSM-2 (osmpbf → Arrow → Lance)
                                                       │
                                                       ▼
                                                D-OSM-3 (SPO lift) ──► OGAR ogar-from-osm-pbf (Phase 2c)
                                                       │
                                                       ▼
                D-OSM-4 (SIMD DEM sample) ──► D-OSM-5 (OSM × DEM → Gaussian3D)
                                                       │
                                                       ▼  (consumes D-SPLAT-1 carrier, D-SPLAT-3 SoA)
                                                D-OSM-6 (cesium-3dtiles-writer — the Rust gap)
                                                       │
                                                       ▼
                                                D-SPLAT-12 splat-render (geospatial backend; same renderer)

           D-OSM-7 (Nominatim sidecar) — independent; UX-edge only.
```

**Cross-arc shared substrate (from `splat-native-ultrasound-v1.md`):**
- D-SPLAT-1 `Gaussian3D` carrier — reused verbatim in D-OSM-5 emit path.
- D-SPLAT-2 SIMD ops — `batched_cholesky_3x3` / `batched_mahalanobis` / `batched_opacity_blend` / `batched_sh_eval_l3` / `batched_se3_transform` all reused for geospatial render; D-OSM-4 adds `batched_sample_height` as a sibling.
- D-SPLAT-3 `SplatBatch<N>` SoA — reused verbatim in D-OSM-5 emit + D-OSM-6 writer input.
- D-SPLAT-12 `crates/splat-render` — same renderer; OSM + ultrasound become two scene backends behind the same render surface.

---

## 6. Open questions

- **OQ-OSM-1** — Tag interning canonicalization rules. Q1 (c) requires Tag identity via canonical key=value form. Lowercase keys? `whitespace-normalize` values? Unicode NFC? Defer to D-OSM-3 implementation.
- **OQ-OSM-2** — DEM source preference. SRTM 1-arcsec (~30m) covers most of the globe; ASTER GDEM covers higher latitudes. Both are free-tier. Default: SRTM 1-arcsec; document the gap at latitude > 60°. Defer to D-OSM-4 implementation.
- **OQ-OSM-3** — `cesium-3dtiles-writer` crate scope. Minimal MVP = b3dm + cmpt + tileset.json. Full = i3dm (instances), pnts (point cloud), subtree files. Recommend MVP for D-OSM-6 P3 ship; full as P5 follow-up.
- **OQ-OSM-4** — Planet-scale ingest discipline. Full OSM planet PBF is ~70 GB; lazy `osmpbf` reader works but Arrow RecordBatch emission needs partitioning. Per-country PBF (Geofabrik downloads) is the practical v1 input. Defer to D-OSM-2 implementation.
- **OQ-OSM-5** — Coordinate-policy compliance. Parent plan rules: *"do not silently treat unknown coordinates as WGS84"*. OSM IS WGS84 by definition — but DEM sources (SRTM, ASTER) use various reference systems. The boundary helper (D-OSM-4) must declare CRS explicitly per the parent plan §"Coordinate policy".

---

## 7. Risk matrix

| Risk | Likelihood | Mitigation |
|---|---|---|
| `osmpbf` v0.4 API breaking change | LOW | Pin minor version in D-OSM-2 Cargo.toml; v0.4 is multi-year stable |
| `cesium-3dtiles-writer` MVP misses Cesium spec corners (D-OSM-6) | MED | Test against CesiumJS reference data; cross-validate b3dm output with `3d-tiles-tools` (TypeScript) for round-trip parity |
| OSM-XYZ → TMS Y-flip boundary bug | MED | Property-based test: round-trip `xyz → tms → xyz ≡ identity` over random tile coords. Boundary helper in D-OSM-1 is the only conversion point — Q3 §2 ruling |
| Tag cardinality explosion under Q1 (b) before (c) lands | LOW | Arrow `List<Struct>` handles bounded per-row tag count (~5-20 typical); compaction OK. Migration to (c) lands when Tag interning is implemented; backward-compatible per D-OSM-3 §3 |
| Planet-scale ingest OOM | MED | Per-country PBF (Geofabrik) is the v1 input per OQ-OSM-4; planet-scale runs are sharded by tile-quadkey prefix at ingest |
| DEM-sample bilinear interpolation precision at query points near bbox edges | MED | D-OSM-4 returns NaN sentinel for out-of-bbox queries; D-OSM-5 caller decides skip-or-extrapolate |

---

## 8. Success criteria

- D-OSM-1 stub compiles cleanly in ndarray `crates/cesium`; module wired into `lib.rs`; doctest passes.
- D-OSM-2 ingests a per-country Geofabrik PBF (e.g. Liechtenstein for size) into three Lance datasets in <60 s on a single core; round-trip query returns same row count.
- D-OSM-3 emits SPO triples consumable by OGAR `ogar-from-osm-pbf` Phase 2c (verified by cross-session integration test once OGAR ships the adapter).
- D-OSM-4 `batched_sample_height` matches scalar reference within 1 ULP on AVX-512 + NEON + scalar (parity test); ≥ 2× scalar throughput at N=1M on AVX-512.
- D-OSM-5 emits Gaussian3D batches from OSM building footprints + SRTM DEM; visually validated by D-OSM-6 b3dm output rendered in CesiumJS reference viewer.
- D-OSM-6 b3dm output round-trips through CesiumGS `3d-tiles-tools` (TypeScript reference) without loss.
- D-OSM-7 (if shipped) Nominatim sidecar resolves test addresses; rate-limit + timeout policies documented.

---

## 9. Cross-references

- **Parent plan:** `lance-graph/.claude/plans/3DGS-ArcGIS-Cesium-ingestion-plan.md` (structural; 5 source classes)
- **Substrate sibling:** `lance-graph/.claude/plans/splat-native-ultrasound-v1.md` (D-SPLAT-1..14; shared carrier + SIMD + Lance + renderer)
- **OGAR coordination thread (2026-06-05):** ratified Q1 + Q2; OGAR-side docs PR (`DOMAIN-INSTANCES.md §2.6` + `RDF-OWL-ALIGNMENT.md §10 Phase 2c`) queued behind this PR
- **OGAR PR #37** (merged): `ogar-adapter-ttl` scaffold — Phase 2a TTL adapter; precedent for `ogar-from-osm-pbf` Phase 2c scaffold shape
- **OGAR PR #25/#31** (shipped): `KnowableFromStore` registry — `ogar-from-osm-pbf` registers via this surface
- **ndarray PR #142** (shipped): VBMI gate pattern for `permute_bytes` — relevant for D-OSM-4 SIMD DEM sampling on older microarchitectures
- **lance-graph PR #471/#472** (shipped): splat-native-ultrasound-v1 canonical plan + fixes — substrate-reuse anchor
- **AFI geocoding blog**: https://blog.afi.io/blog/building-a-free-geocoding-and-reverse-geocoding-service-with-openstreetmap/ — Nominatim + Photon stack rationale for D-OSM-7
- **b-r-u GitHub profile**: https://github.com/b-r-u — `osmpbf` is the only production-ready crate; `osmflat` exists but is NOT Arrow (heremaps `flatdata`; reject for D-OSM-2)
- **OSM PBF spec**: https://wiki.openstreetmap.org/wiki/PBF_Format — informs D-OSM-1 stub types
- **OGC 3D Tiles 1.1 spec**: https://docs.ogc.org/cs/22-025r4/22-025r4.html — informs D-OSM-6 writer
- **CesiumGS 3d-tiles-tools** (TypeScript reference): https://github.com/CesiumGS/3d-tiles-tools — round-trip validation for D-OSM-6

---

## 10. What this plan does NOT cover

- **Planet-scale vector-tile generation** — Tilemaker / Planetiler are C++ / Java. No Rust planet-scale generator exists. Out of scope for D-OSM-* family; would be a multi-month port worth its own arc if ever needed.
- **Nominatim / Photon reimplementation** — These are PostgreSQL+PostGIS + Elasticsearch services. No Rust reimplementation is worth doing. D-OSM-7 covers HTTP sidecar adapter only.
- **Routing / routing-engine** — OSRM, Valhalla, GraphHopper are the references. None in Rust at planet scale. Out of scope.
- **OSM data editing / write-back** — OSM edits go through OSM API (osm.org). This plan is read-only consumer; write-back would be a separate adapter.
- **iMapping / OSM-XML legacy** — XML format is legacy; PBF is the canonical wire format. XML support deferred until/unless a consumer demands it.
- **The ultrasound arc.** Substrate reuse (Gaussian3D, SIMD, SoA, renderer) is the cross-arc payoff; the ultrasound scene (FMA atlas + HoloLens patient overlay) is OUT of scope for OSM — they are sibling domain instances, not pipeline-mergeable.

---

## 11. ADR-024 adoption — palette256 + HHTL codec contract

> **Pinned 2026-06-05 after ADR-024 landed in OGAR #39.** This section closes the runtime-side follow-up commitment that ADR-024 § Consequences names by reference ("Reports ρ-vs-reference on first per-country PBF run per the runtime session's §11 follow-up commitment").

The cesium-osm arc adopts **ADR-024** (Palette256 + HHTL codec — universal compression primitive) as the canonical compression contract for D-OSM-2 (OSM tag palette + tile-local coordinate quantization) and any future per-tile palette in this family.

### What ADR-024 requires

Per `OGAR/docs/ARCHITECTURAL-DECISIONS-2026-06-04.md § ADR-024` adoption checklist:

1. **Identify the prefix.** For D-OSM-2: the Cesium TMS quadkey path `qk:<level>/<x>/<y_tms>` (Q2 ruling §2). The prefix is the per-tile spatial frame.
2. **Identify the palette domain.** For D-OSM-2: OSM tag-values clustered per tile (~95% body covered by ≤256 distinct values at zoom 21 per ADR-024 § 256-ceiling escape hatches) **and** tile-local 16-bit quantized coordinates (tile bounds ~4 m at zoom 21, sub-cm precision).
3. **Build the palette + measure ρ-vs-reference.** For D-OSM-2: the reference is exact-match tag equality; ρ for the per-tile tag palette is the proportion of correctly-decoded tag values vs ground truth. Per ADR-024 target: **ρ ≥ 0.99** matching the `lance-graph-arm-discovery` aerial-codebook anchor (ρ = 0.9973 vs cosine).
4. **Decode = const-table lookup.** Per-tile palette is runtime const-table; decode path is zero-allocation. Compile-time HHTL where the palette is shared across tiles (e.g. the global ~100 most-common OSM keys: `highway`, `building`, `name`, `addr:*`).

### Falsifiable adoption contract for D-OSM-2

**D-OSM-2 implementation MUST report:**

- Empirical ρ-vs-reference on the **first per-country PBF run** (default candidate: Liechtenstein PBF — smallest tractable corpus; per §6 OQ-OSM-4).
- Per-tile palette cardinality distribution (mean / p95 / p99). The 256-ceiling escape hatch (per-tile, hierarchical, or palette-64K) is selected on the basis of measured cardinality, not speculation.
- Decode bandwidth: target ≥ 10⁸ palette-decodes/sec on AVX-512 (gather + table-lookup), matching the bgz-tensor `AttentionTable` throughput regime.

If the ρ-vs-reference falls below 0.99 on the first per-country run, **D-OSM-2 must document the gap before proceeding** to multi-country ingest — either by (a) escalating to per-tile palettes if the global palette undercovers, (b) escalating to palette-64K if cardinality genuinely exceeds 256, or (c) accepting the gap with rationale (e.g. ρ = 0.96 may be acceptable for navigation-style queries where exact tag equality is not load-bearing).

### Cross-arc ADR-024 adopters (the codec spreads)

| Adopter | Domain | Prefix | Palette domain | ρ-vs-reference |
|---|---|---|---|---|
| `arm-discovery` aerial codebook (anchor; **shipped**) | Distance | class identity | quantized cosine | **ρ = 0.9973** (the empirical floor) |
| `Binary16K` `_effectiveReaders` (anchor; **shipped**) | Security | row identity | 256 role-bit subsets | binary exact-match (popcount intersect) |
| `bgz-tensor` `WeightPalette` (anchor; **shipped**) | Attention | layer / head index | quantized dense weights | cosine (matches ADR-024 reference) |
| **D-OSM-2** (this plan, queued) | Geographic | Cesium TMS quadkey | tag-values + tile-local coords | **ρ ≥ 0.99** target (this section) |
| **D-SPLAT-4** (splat-native arc, queued) | Anatomical / volumetric | FMA NiblePath + SH basis-id | SH coefficients (ℓ=3) | **ρ ≥ 0.99** target (per splat-native plan amendment, separate PR) |

The two queued adopters (D-OSM-2 + D-SPLAT-4) are explicitly named in ADR-024 § Consequences. This section is the runtime-side acknowledgment that the codec contract binds at adoption time, not after the fact.

### Why this section is §11, not §3

Adding ADR-024 as a deliverable specification on D-OSM-2 (in §3) would conflate the carrier-shape contract (Q1 v1 fallback) with the compression contract (palette256 codec). They are independent — the Arrow `List<Struct<key, value>>` shape (Q1 v1) holds *whether or not* the tag value is palette-encoded; the palette is a transparent codec underneath the column. Keeping the ADR-024 callout at §11 preserves §3's "carrier shape" framing while pinning the codec contract separately.

When D-OSM-2 ships, the implementation PR cites both §3 (carrier) and §11 (codec) as its contract surface.

---

_End of plan v1._
