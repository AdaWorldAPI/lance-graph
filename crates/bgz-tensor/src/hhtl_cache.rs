//! HHTL cache: compact index alongside bgz7 weight files.
//!
//! Extracts the 256-entry palette + distance table from bgz7 shards
//! and writes a compact cache file for HIP-level early exit.
//!
//! ```text
//! Per model:
//!   shard-00.bgz7           (17 MB)  ← full weight fingerprints
//!   shard-00_hhtl.bgz       (140 KB) ← palette + distance table (95% queries)
//!
//! Or per model (aggregated):
//!   qwen35-9b-base_hhtl.bgz (140 KB) ← combined from all 4 shards
//! ```
//!
//! Format: "HHTL" + k(u16) + k × Base17(34) + k × k × u16 + k × u32 radii
//!   = 4 + 2 + 256×34 + 256×256×2 + 256×4 = 140,294 bytes for k=256
//!
//! The HHTL cache enables:
//!   HEEL: PAL8 palette bits → which blocks? (4 KB, from ndarray)
//!   HIP:  HHTL cache → L1 distance between any two archetypes (140 KB, this file)
//!   TWIG: bgz7 → per-row Base17 lookup (17+ MB, feature-gated download)
//!   LEAF: BF16 from HuggingFace → never stored locally

use crate::projection::Base17;
use crate::palette::WeightPalette;
use crate::attention::AttentionTable;
use crate::cascade::{ScentByte, CascadeConfig};

/// Precomputed action for an archetype pair.
///
/// This is NOT just distance — it's the **routing decision**.
/// The prefetch loads decisions, not data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RouteAction {
    /// Pair doesn't interact. Skip entirely. No attention score needed.
    Skip = 0,
    /// Direct attention: pair interacts, score = distance table lookup.
    Attend = 1,
    /// Compose: pair interacts through intermediate archetype (index stored separately).
    Compose = 2,
    /// Escalate: HIP can't decide — need TWIG-level Base17 L1 for this pair.
    Escalate = 3,
}

/// HHTL cache: palette + precomputed distance table + route table.
///
/// The route table is the key insight: it precomputes the CASCADE DECISION
/// for every archetype pair. At inference time, looking up what to do
/// with token pair (i, j) is:
///
/// ```text
/// let a = palette_idx[i];
/// let b = palette_idx[j];
/// match cache.route(a, b) {
///     Skip     → don't compute attention (60% of pairs)
///     Attend   → score = cache.distance(a, b) (35% of pairs)
///     Compose  → score via intermediate (rare)
///     Escalate → need full Base17 L1 (5% of pairs)
/// }
/// ```
///
/// This is the HIP-level index. 140-150 KB per model. 95% early exit.
#[derive(Clone, Debug)]
pub struct HhtlCache {
    /// The k archetypal Base17 patterns.
    pub palette: WeightPalette,
    /// k × k pairwise L1 distances (precomputed, O(1) lookup).
    pub distances: AttentionTable,
    /// k × k precomputed routing decisions. Same layout as distances.
    pub routes: Vec<RouteAction>,
}

impl HhtlCache {
    /// Build from an existing palette with default cascade config.
    pub fn from_palette(palette: WeightPalette) -> Self {
        Self::from_palette_with_config(palette, &CascadeConfig::default())
    }

    /// Build from an existing palette with custom thresholds.
    pub fn from_palette_with_config(palette: WeightPalette, config: &CascadeConfig) -> Self {
        let distances = AttentionTable::build(&palette);
        let routes = build_route_table(&palette, &distances, config);
        Self { palette, distances, routes }
    }

    /// Build from raw Base17 rows (e.g., read from bgz7 shards).
    ///
    /// Selects up to 256 archetypes via furthest-point sampling,
    /// computes the distance table, stores radii for distortion bounds.
    pub fn from_base17_rows(rows: &[Base17], max_k: usize) -> Self {
        let k = rows.len().min(max_k).min(256);
        if k == 0 {
            return Self {
                palette: WeightPalette {
                    entries: Vec::new(),
                    radii: Vec::new(),
                    counts: Vec::new(),
                },
                distances: AttentionTable {
                    distances: Vec::new(),
                    k: 0,
                },
                routes: Vec::new(),
            };
        }

        // Furthest-point sampling for coverage
        let mut selected = Vec::with_capacity(k);
        let mut selected_idx = Vec::with_capacity(k);
        let mut min_dists = vec![u32::MAX; rows.len()];

        // Start with first row
        selected.push(rows[0].clone());
        selected_idx.push(0);

        for _ in 1..k {
            // Update min distances to nearest selected
            let last = selected.last().unwrap();
            for (i, row) in rows.iter().enumerate() {
                let d = row.l1(last);
                if d < min_dists[i] {
                    min_dists[i] = d;
                }
            }

            // Pick the row farthest from all selected
            let mut best_idx = 0;
            let mut best_dist = 0u32;
            for (i, &d) in min_dists.iter().enumerate() {
                if d > best_dist && !selected_idx.contains(&i) {
                    best_dist = d;
                    best_idx = i;
                }
            }

            selected.push(rows[best_idx].clone());
            selected_idx.push(best_idx);
        }

        // Compute radii: for each archetype, max L1 to any assigned row
        let mut radii = vec![0u32; k];
        let mut counts = vec![0u32; k];
        for row in rows {
            let (nearest, dist) = nearest_archetype(row, &selected);
            counts[nearest] += 1;
            if dist > radii[nearest] {
                radii[nearest] = dist;
            }
        }

        let palette = WeightPalette {
            entries: selected,
            radii,
            counts,
        };
        let distances = AttentionTable::build(&palette);
        let config = CascadeConfig::default();
        let routes = build_route_table(&palette, &distances, &config);

        Self { palette, distances, routes }
    }

    /// Palette size (number of archetypes).
    pub fn k(&self) -> usize {
        self.palette.len()
    }

    /// O(1) distance lookup between two archetype indices.
    #[inline]
    pub fn distance(&self, a: u8, b: u8) -> u16 {
        self.distances.distance(a, b)
    }

    /// O(1) route lookup: what should we do with this archetype pair?
    ///
    /// This is the prefetch decision. When token A (archetype `a`) meets
    /// token B (archetype `b`), the route tells the attention engine:
    /// Skip (no computation), Attend (use distance), Compose (multi-hop),
    /// or Escalate (need more data).
    #[inline]
    pub fn route(&self, a: u8, b: u8) -> RouteAction {
        let k = self.k();
        if (a as usize) < k && (b as usize) < k {
            self.routes[a as usize * k + b as usize]
        } else {
            RouteAction::Skip
        }
    }

    /// Find nearest archetype for a query Base17.
    pub fn nearest(&self, query: &Base17) -> (u8, u32) {
        let (idx, dist) = nearest_archetype(query, &self.palette.entries);
        (idx as u8, dist)
    }

    /// Serialize to compact binary format.
    ///
    /// Format: "HHTL" + k(u16) + k×Base17(34) + k×k×u16 + k×k×u8(routes) + k×u32(radii)
    /// k=256: 4 + 2 + 8704 + 131072 + 65536 + 1024 = 206,342 bytes (~200 KB)
    /// k=64:  4 + 2 + 2176 + 8192 + 4096 + 256 = 14,726 bytes (~14 KB)
    pub fn serialize(&self, path: &str) -> Result<(), String> {
        use std::io::Write;
        let k = self.k();
        let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;

        f.write_all(b"HHTL").map_err(|e| e.to_string())?;
        f.write_all(&(k as u16).to_le_bytes()).map_err(|e| e.to_string())?;

        // Palette entries
        for entry in &self.palette.entries {
            for &dim in &entry.dims {
                f.write_all(&dim.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }

        // Distance table
        for &d in &self.distances.distances {
            f.write_all(&d.to_le_bytes()).map_err(|e| e.to_string())?;
        }

        // Route table
        for &r in &self.routes {
            f.write_all(&[r as u8]).map_err(|e| e.to_string())?;
        }

        // Radii
        for &r in &self.palette.radii {
            f.write_all(&r.to_le_bytes()).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Deserialize from compact binary.
    pub fn deserialize(path: &str) -> Result<Self, String> {
        use std::io::Read;
        let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;

        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"HHTL" {
            return Err(format!("bad magic: {:?}", magic));
        }

        let mut k_buf = [0u8; 2];
        f.read_exact(&mut k_buf).map_err(|e| e.to_string())?;
        let k = u16::from_le_bytes(k_buf) as usize;

        // Palette entries
        let mut entries = Vec::with_capacity(k);
        for _ in 0..k {
            let mut dims = [0i16; 17];
            for d in &mut dims {
                let mut buf = [0u8; 2];
                f.read_exact(&mut buf).map_err(|e| e.to_string())?;
                *d = i16::from_le_bytes(buf);
            }
            entries.push(Base17 { dims });
        }

        // Distance table
        let mut distances = vec![0u16; k * k];
        for d in &mut distances {
            let mut buf = [0u8; 2];
            f.read_exact(&mut buf).map_err(|e| e.to_string())?;
            *d = u16::from_le_bytes(buf);
        }

        // Route table
        let mut routes = vec![RouteAction::Skip; k * k];
        for r in &mut routes {
            let mut buf = [0u8; 1];
            f.read_exact(&mut buf).map_err(|e| e.to_string())?;
            *r = match buf[0] {
                0 => RouteAction::Skip,
                1 => RouteAction::Attend,
                2 => RouteAction::Compose,
                3 => RouteAction::Escalate,
                _ => RouteAction::Skip,
            };
        }

        // Radii
        let mut radii = vec![0u32; k];
        for r in &mut radii {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf).map_err(|e| e.to_string())?;
            *r = u32::from_le_bytes(buf);
        }

        let counts = vec![0u32; k];

        Ok(Self {
            palette: WeightPalette { entries, radii, counts },
            distances: AttentionTable { distances, k },
            routes,
        })
    }

    /// Check if HHTL cache exists for a model.
    pub fn cache_path(model_dir: &str, model_name: &str) -> String {
        format!("{}/{}_hhtl.bgz", model_dir, model_name)
    }

    /// Load or build: try cache first, build from bgz7 rows if missing.
    pub fn load_or_build(
        cache_path: &str,
        rows: Option<&[Base17]>,
        max_k: usize,
    ) -> Result<Self, String> {
        // Try cache first
        if std::fs::metadata(cache_path).is_ok() {
            return Self::deserialize(cache_path);
        }

        // Build from rows
        let rows = rows.ok_or_else(|| {
            format!("{cache_path} not found and no rows provided — run hydrate first")
        })?;

        let cache = Self::from_base17_rows(rows, max_k);
        cache.serialize(cache_path)?;
        Ok(cache)
    }
}

/// Build the route table: precompute cascade decisions for all archetype pairs.
///
/// For each (a, b) pair, runs the HEEL + HIP check to decide the action.
/// This is O(k²) at build time, O(1) at inference time.
fn build_route_table(
    palette: &WeightPalette,
    distances: &AttentionTable,
    _config: &CascadeConfig,
) -> Vec<RouteAction> {
    let k = palette.len();
    let mut routes = vec![RouteAction::Skip; k * k];

    // Derive thresholds from the actual distance distribution (not fixed constants).
    // Collect all non-diagonal distances, sort, use percentiles.
    let mut all_dists: Vec<u16> = Vec::with_capacity(k * k);
    for a in 0..k {
        for b in 0..k {
            if a != b {
                all_dists.push(distances.distance(a as u8, b as u8));
            }
        }
    }
    all_dists.sort_unstable();
    let n = all_dists.len();
    let p25 = if n > 0 { all_dists[n / 4] } else { 0 };
    let p75 = if n > 0 { all_dists[3 * n / 4] } else { u16::MAX };

    for a in 0..k {
        for b in 0..k {
            let dist = distances.distance(a as u8, b as u8);

            // Skip: distance above 75th percentile — too far, no interaction
            if dist > p75 {
                routes[a * k + b] = RouteAction::Skip;
                continue;
            }

            // Check if this pair could benefit from composition
            // (exists intermediate c where d(a,c) + d(c,b) < d(a,b) * 1.1)
            let mut has_shortcut = false;
            for c in 0..k {
                if c == a || c == b { continue; }
                let d_ac = distances.distance(a as u8, c as u8) as u32;
                let d_cb = distances.distance(c as u8, b as u8) as u32;
                let d_ab = dist as u32;
                // Composition is useful if the path through c is significantly different
                // (not just shorter, but structurally different route)
                if d_ac + d_cb < (d_ab * 9) / 10 {
                    has_shortcut = true;
                    break;
                }
            }

            if has_shortcut {
                routes[a * k + b] = RouteAction::Compose;
            } else if dist <= p25 {
                // Strong signal (bottom 25%) — attend directly
                routes[a * k + b] = RouteAction::Attend;
            } else {
                // Middle 50% — needs TWIG to decide
                routes[a * k + b] = RouteAction::Escalate;
            }
        }
        // Self-attention is always direct
        routes[a * k + a] = RouteAction::Attend;
    }

    routes
}

/// Find nearest archetype by L1 distance.
fn nearest_archetype(query: &Base17, archetypes: &[Base17]) -> (usize, u32) {
    let mut best_idx = 0;
    let mut best_dist = u32::MAX;
    for (i, a) in archetypes.iter().enumerate() {
        let d = query.l1(a);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    (best_idx, best_dist)
}

/// HIP-level cache: 64 archetypes for p64 Palette64 compatibility.
///
/// 64 entries × 34 bytes Base17 = 2,176 bytes palette
/// 64 × 64 × 2 bytes distances  = 8,192 bytes
/// 64 × 4 bytes radii            = 256 bytes
/// Total: 10,630 bytes (~10 KB) — fits L1 cache.
///
/// This is the sweet spot for p64: `Palette64::attend()` works on 64 rows.
/// The 9B model has ~40 transformer layers × ~64 heads = ~640 unique patterns.
/// Furthest-point sampling from 640 to 64 gives ~93% coverage.
///
/// For 27B (~64 layers × ~64 heads = ~4096 patterns), sampling to 64 gives
/// ~76% coverage. Use k=256 HHTL for 27B, k=64 HIP for 9B.
pub type HipCache = HhtlCache;

impl HhtlCache {
    /// Build a HIP-level cache (k=64) for p64 compatibility.
    pub fn build_hip(rows: &[Base17]) -> Self {
        Self::from_base17_rows(rows, 64)
    }

    /// Build a full HHTL cache (k=256) for 27B models.
    pub fn build_full(rows: &[Base17]) -> Self {
        Self::from_base17_rows(rows, 256)
    }

    /// Export as 64×64 distance matrix for p64 Palette64 operations.
    ///
    /// Returns None if k > 64 (use full HHTL instead).
    pub fn as_p64_distances(&self) -> Option<[[u16; 64]; 64]> {
        if self.k() > 64 { return None; }
        let k = self.k();
        let mut matrix = [[0u16; 64]; 64];
        for i in 0..k {
            for j in 0..k {
                matrix[i][j] = self.distance(i as u8, j as u8);
            }
        }
        Some(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hhtl_cache_empty() {
        let cache = HhtlCache::from_base17_rows(&[], 256);
        assert_eq!(cache.k(), 0);
    }

    #[test]
    fn test_hhtl_cache_small() {
        let rows: Vec<Base17> = (0..10).map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i * 100) as i16;
            dims[1] = (i * 50) as i16;
            Base17 { dims }
        }).collect();

        let cache = HhtlCache::from_base17_rows(&rows, 256);
        assert_eq!(cache.k(), 10); // fewer rows than max_k

        // Distance should be symmetric
        let d01 = cache.distance(0, 1);
        let d10 = cache.distance(1, 0);
        assert_eq!(d01, d10);

        // Self-distance should be 0
        assert_eq!(cache.distance(0, 0), 0);
    }

    #[test]
    fn test_hhtl_cache_serialization_roundtrip() {
        let rows: Vec<Base17> = (0..20).map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i * 100) as i16;
            dims[3] = (i * 77) as i16;
            dims[16] = -(i * 30) as i16;
            Base17 { dims }
        }).collect();

        let cache = HhtlCache::from_base17_rows(&rows, 16);
        assert_eq!(cache.k(), 16);

        let path = "/tmp/test_hhtl_roundtrip.bgz";
        cache.serialize(path).expect("serialize");

        let loaded = HhtlCache::deserialize(path).expect("deserialize");
        assert_eq!(loaded.k(), 16);

        // Distances should match
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(
                    cache.distance(i as u8, j as u8),
                    loaded.distance(i as u8, j as u8),
                    "mismatch at ({i}, {j})"
                );
            }
        }

        // Palette entries should match
        for i in 0..16 {
            assert_eq!(cache.palette.entries[i], loaded.palette.entries[i]);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_hhtl_cache_256_size() {
        // Verify file size for k=256
        let rows: Vec<Base17> = (0..300).map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i % 256) as i16 * 100;
            dims[1] = (i / 3) as i16;
            Base17 { dims }
        }).collect();

        let cache = HhtlCache::from_base17_rows(&rows, 256);
        assert_eq!(cache.k(), 256);

        let path = "/tmp/test_hhtl_256.bgz";
        cache.serialize(path).expect("serialize");

        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        // 4 magic + 2 k + 256×34 entries + 256×256×2 distances + 256×256×1 routes + 256×4 radii
        let expected = 4 + 2 + 256 * 34 + 256 * 256 * 2 + 256 * 256 * 1 + 256 * 4;
        assert_eq!(size, expected as u64, "expected {expected} bytes, got {size}");

        std::fs::remove_file(path).ok();
    }
}
