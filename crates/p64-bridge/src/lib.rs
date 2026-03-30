//! # p64-bridge — lance-graph → p64 palette wiring
//!
//! Maps lance-graph types to p64-compatible palette operations.
//! **No p64 dependency** — provides compatible constants and mapping functions.
//! The consumer imports both `p64` and `p64-bridge`, connecting them at the call site.
//!
//! ## Architecture
//!
//! ```text
//! lance-graph                    p64 (ndarray)
//! ──────────                     ─────────────
//! CausalEdge64 ──→ p64-bridge ──→ palette row/col addressing
//! ThinkingStyle ──→ p64-bridge ──→ layer_mask + combine + contra
//! HdrSemiring   ──→ p64-bridge ──→ combine_mode + contra_mode
//! ```

use std::sync::LazyLock;

use causal_edge::edge::CausalEdge64;

// ============================================================================
// CausalEdge64 → Palette64 addressing
// ============================================================================

/// Map a CausalEdge64 to a 64×64 palette block address.
///
/// S[0:7] → block row (S/4), O[16:23] → block col (O/4).
/// The 256 palette indices map to 64 blocks of 4.
#[inline]
pub fn edge_to_block(edge: &CausalEdge64) -> (usize, usize) {
    let row = edge.s_idx() as usize / 4;
    let col = edge.o_idx() as usize / 4;
    (row.min(63), col.min(63))
}

/// Extract NARS frequency + confidence as f32 pair.
#[inline]
pub fn edge_nars_f32(edge: &CausalEdge64) -> (f32, f32) {
    (
        edge.frequency_u8() as f32 / 255.0,
        edge.confidence_u8() as f32 / 255.0,
    )
}

/// Map CausalEdge64's causal mask + inference type to an 8-bit predicate layer mask.
///
/// ```text
/// Pearl causal mask:
///   bit 0 (direct)      → Layer 0 CAUSES
///   bit 1 (enabling)    → Layer 1 ENABLES
///   bit 2 (confounding) → Layer 3 CONTRADICTS
///
/// Inference type:
///   0 Deduction  → Layer 0 CAUSES
///   1 Induction  → Layer 2 SUPPORTS
///   2 Abduction  → Layer 5 ABSTRACTS
///   3 Revision   → Layer 4 REFINES
///   4 Synthesis  → Layer 7 BECOMES
/// ```
#[inline]
pub fn edge_to_layer_mask(edge: &CausalEdge64) -> u8 {
    let causal = edge.causal_mask() as u8;
    let infer = edge.inference_type() as u8;

    let mut mask = 0u8;

    // Pearl's causal mask → predicate layers
    if causal & 1 != 0 { mask |= 1 << CAUSES; }
    if causal & 2 != 0 { mask |= 1 << ENABLES; }
    if causal & 4 != 0 { mask |= 1 << CONTRADICTS; }

    // Inference type → predicate layer
    mask |= match infer {
        0 => 1 << CAUSES,
        1 => 1 << SUPPORTS,
        2 => 1 << ABSTRACTS,
        3 => 1 << REFINES,
        4 => 1 << BECOMES,
        _ => 1 << GROUNDS,
    };

    mask
}

/// Build a 64×64 bitmask from a batch of CausalEdge64 values.
/// Returns `[u64; 64]` directly usable as `Palette64::rows`.
pub fn edges_to_palette_rows(edges: &[CausalEdge64]) -> [u64; 64] {
    let mut rows = [0u64; 64];
    for edge in edges {
        let (row, col) = edge_to_block(edge);
        rows[row] |= 1u64 << col;
    }
    rows
}

/// Build 8 layered palette row arrays from edges (for Palette3D).
/// Each edge is routed to the appropriate predicate layers based on
/// its causal mask and inference type.
pub fn edges_to_layered_rows(edges: &[CausalEdge64]) -> [[u64; 64]; 8] {
    let mut layers = [[0u64; 64]; 8];
    for edge in edges {
        let (row, col) = edge_to_block(edge);
        let layer_mask = edge_to_layer_mask(edge);
        for z in 0..8 {
            if layer_mask & (1 << z) != 0 {
                layers[z][row] |= 1u64 << col;
            }
        }
    }
    layers
}

// ── Predicate layer indices (matches p64::predicate) ──────────────────

pub const CAUSES: usize = 0;
pub const ENABLES: usize = 1;
pub const SUPPORTS: usize = 2;
pub const CONTRADICTS: usize = 3;
pub const REFINES: usize = 4;
pub const ABSTRACTS: usize = 5;
pub const GROUNDS: usize = 6;
pub const BECOMES: usize = 7;

// ============================================================================
// ThinkingStyle → p64 modulation parameters
// ============================================================================

/// p64-compatible modulation parameters (mirrors p64::ThinkingStyle layout).
/// The consumer constructs a `p64::ThinkingStyle` from these values.
#[derive(Debug, Clone, Copy)]
pub struct StyleParams {
    pub layer_mask: u8,
    pub combine: u8,     // 0=Union, 1=Intersection, 2=Majority, 3=Weighted
    pub contra: u8,      // 0=Suppress, 1=Ignore, 2=Invert, 3=Tension
    pub density_target: f32,
    pub name: &'static str,
}

/// Combine mode constants (match p64::CombineMode ordinals).
pub mod combine {
    pub const UNION: u8 = 0;
    pub const INTERSECTION: u8 = 1;
    pub const MAJORITY: u8 = 2;
    pub const WEIGHTED: u8 = 3;
}

/// Contra mode constants (match p64::ContraMode ordinals).
pub mod contra {
    pub const SUPPRESS: u8 = 0;
    pub const IGNORE: u8 = 1;
    pub const INVERT: u8 = 2;
    pub const TENSION: u8 = 3;
}

/// All 12 base thinking styles, frozen via LazyLock.
/// Ordinal matches lance-graph-planner `ThinkingStyle` enum order:
/// Analytical(0), Convergent(1), Systematic(2),
/// Creative(3), Divergent(4), Exploratory(5),
/// Focused(6), Diffuse(7), Peripheral(8),
/// Intuitive(9), Deliberate(10), Metacognitive(11).
static STYLES: LazyLock<[StyleParams; 12]> = LazyLock::new(|| {
    [
        // ── Convergent cluster ──
        StyleParams { layer_mask: 0b0111_0111, combine: combine::INTERSECTION, contra: contra::SUPPRESS, density_target: 0.05, name: "analytical" },
        StyleParams { layer_mask: 0b0011_0111, combine: combine::INTERSECTION, contra: contra::SUPPRESS, density_target: 0.04, name: "convergent" },
        StyleParams { layer_mask: 0b0111_1111, combine: combine::INTERSECTION, contra: contra::SUPPRESS, density_target: 0.03, name: "systematic" },
        // ── Divergent cluster ──
        StyleParams { layer_mask: 0b1111_1111, combine: combine::UNION,        contra: contra::IGNORE,   density_target: 0.40, name: "creative" },
        StyleParams { layer_mask: 0b1000_1001, combine: combine::UNION,        contra: contra::INVERT,   density_target: 0.30, name: "divergent" },
        StyleParams { layer_mask: 0b1111_1111, combine: combine::UNION,        contra: contra::IGNORE,   density_target: 0.50, name: "exploratory" },
        // ── Attention cluster ──
        StyleParams { layer_mask: 0b0000_0011, combine: combine::INTERSECTION, contra: contra::SUPPRESS, density_target: 0.02, name: "focused" },
        StyleParams { layer_mask: 0b0111_0111, combine: combine::MAJORITY,     contra: contra::TENSION,  density_target: 0.20, name: "diffuse" },
        StyleParams { layer_mask: 0b1110_0000, combine: combine::UNION,        contra: contra::IGNORE,   density_target: 0.35, name: "peripheral" },
        // ── Speed cluster ──
        StyleParams { layer_mask: 0b0000_0001, combine: combine::UNION,        contra: contra::IGNORE,   density_target: 0.50, name: "intuitive" },
        StyleParams { layer_mask: 0b0111_1111, combine: combine::WEIGHTED,     contra: contra::SUPPRESS, density_target: 0.08, name: "deliberate" },
        StyleParams { layer_mask: 0b1110_0000, combine: combine::MAJORITY,     contra: contra::TENSION,  density_target: 0.10, name: "metacognitive" },
    ]
});

/// Get style params by ordinal (0..11). O(1) after LazyLock init.
#[inline]
pub fn style_by_ordinal(ord: usize) -> &'static StyleParams {
    &STYLES[ord % 12]
}

/// Get style params by name. O(n) scan but n=12, so ~12 comparisons max.
#[inline]
pub fn style_by_name(name: &str) -> &'static StyleParams {
    let lower = name.to_lowercase();
    STYLES.iter().find(|s| s.name == lower.as_str()).unwrap_or(&STYLES[0])
}

/// All 12 styles as a slice.
#[inline]
pub fn all_styles() -> &'static [StyleParams; 12] {
    &STYLES
}

// ============================================================================
// Semiring → combine/contra mapping
// ============================================================================

/// Map a semiring name to (combine_mode, contra_mode) ordinals.
///
/// ```text
/// BOOLEAN          → (Intersection, Suppress)  — strict reachability
/// HAMMING_MIN      → (Weighted, Tension)        — shortest path with awareness
/// SIMILARITY_MAX   → (Majority, Suppress)       — consensus matching
/// XOR_BUNDLE       → (Union, Invert)            — superposition + divergent
/// TRUTH_PROPAGATING→ (Weighted, Tension)         — NARS revision chain
/// ```
pub fn semiring_to_modes(semiring_name: &str) -> (u8, u8) {
    match semiring_name {
        "BOOLEAN"           => (combine::INTERSECTION, contra::SUPPRESS),
        "HAMMING_MIN"       => (combine::WEIGHTED,     contra::TENSION),
        "SIMILARITY_MAX"    => (combine::MAJORITY,     contra::SUPPRESS),
        "XOR_BUNDLE"        => (combine::UNION,        contra::INVERT),
        "TRUTH_PROPAGATING" => (combine::WEIGHTED,     contra::TENSION),
        _                   => (combine::MAJORITY,     contra::SUPPRESS),
    }
}

// ============================================================================
// Blumenstrauß: 8 planes bundled — topology × metric × algebra
// ============================================================================

/// Blumenstrauß: binds p64 topology with bgz17 O(1) distance.
///
/// ```text
/// Mask (p64):     [u64; 64]       WHICH pairs interact (topology)
/// Distance (bgz17): [u16; k×k]   HOW FAR apart (metric, O(1) lookup)
/// Compose (bgz17):  [u8; k×k]    WHAT path composition means (algebra, O(1))
/// ```
///
/// No POPCNT. No Hamming. Distance is PRECOMPUTED in the codebook.
/// The mask gates access. The table provides the answer. O(1).
pub mod blumenstrauss {
    use bgz17::distance_matrix::DistanceMatrix;
    use bgz17::palette::Palette;
    use bgz17::palette_semiring::PaletteSemiring;

    /// 8 planes × 64 rows = 512 u64 = topology mask.
    /// Each plane corresponds to a predicate layer (CAUSES..BECOMES).
    /// Each bit at `planes[z][row] & (1 << col)` means:
    ///   "archetype block `row` relates to block `col` via predicate `z`."
    ///
    /// Distance between any two archetype indices = `semiring.distance(a, b)`.
    /// Path composition = `semiring.compose(a, b)`.
    /// Both O(1). The mask just says WHERE to look.
    pub struct Blumenstrauss<'a> {
        /// 8 predicate planes (topology).
        pub planes: [[u64; 64]; 8],
        /// bgz17 semiring: distance + compose + identity.
        pub semiring: &'a PaletteSemiring,
        /// Palette size (typically 256).
        pub k: usize,
    }

    /// Result of a cascade query.
    #[derive(Debug, Clone)]
    pub struct CascadeHit {
        /// Target archetype index (0..255).
        pub target: u8,
        /// Precomputed distance from query to target.
        pub distance: u16,
        /// Which predicate layers connect them (bitmask).
        pub predicates: u8,
    }

    impl<'a> Blumenstrauss<'a> {
        /// Construct from layered rows (output of `edges_to_layered_rows`) + bgz17 semiring.
        pub fn new(planes: [[u64; 64]; 8], semiring: &'a PaletteSemiring) -> Self {
            Self {
                planes,
                semiring,
                k: semiring.k,
            }
        }

        /// HHTL Cascade: find all targets within `radius` of `query`.
        ///
        /// ```text
        /// HEEL: which predicate planes are active? (layer_mask gates Z)
        /// HIP:  scan mask row for active bits     (topology gates X-Y)
        /// TWIG: expand block → 4 archetype indices (4×4 refinement)
        /// LEAF: semiring.distance(query, target)   (O(1) metric lookup)
        /// ```
        ///
        /// Returns hits sorted by distance ascending.
        pub fn cascade(
            &self,
            query: u8,
            radius: u16,
            layer_mask: u8,
        ) -> Vec<CascadeHit> {
            let block_row = query as usize / 4;
            if block_row >= 64 {
                return Vec::new();
            }

            // Collect which block-columns are active across selected layers
            let mut active_cols = 0u64;
            let mut per_col_predicates = [0u8; 64];

            for z in 0..8 {
                if layer_mask & (1 << z) == 0 {
                    continue;
                }
                let row_mask = self.planes[z][block_row];
                let mut bits = row_mask;
                while bits != 0 {
                    let col = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    active_cols |= 1u64 << col;
                    per_col_predicates[col] |= 1 << z;
                }
            }

            // Expand active block-columns to archetype indices, lookup distance
            let mut hits = Vec::new();
            let mut bits = active_cols;
            while bits != 0 {
                let block_col = bits.trailing_zeros() as usize;
                bits &= bits - 1;

                // 4 archetype indices per block
                for sub in 0..4 {
                    let target = (block_col * 4 + sub) as u8;
                    if (target as usize) >= self.k {
                        continue;
                    }
                    let dist = self.semiring.distance(query, target);
                    if dist <= radius {
                        hits.push(CascadeHit {
                            target,
                            distance: dist,
                            predicates: per_col_predicates[block_col],
                        });
                    }
                }
            }

            hits.sort_by_key(|h| h.distance);
            hits
        }

        /// Transitive deduction: A→B→C via compose.
        ///
        /// Given query A, find all B where A→B exists (CAUSES layer),
        /// then for each B, compose(A,B) gives the intermediate,
        /// and scan for all C where B→C exists (ENABLES layer).
        /// Distance of the composed path = distance(A, compose(A,B)) + distance(compose(A,B), C).
        pub fn deduce_path(
            &self,
            query: u8,
            cause_layer: usize,
            effect_layer: usize,
            max_hops: usize,
        ) -> Vec<(u8, u16, Vec<u8>)> {
            let mut results = Vec::new();
            let mut visited = [false; 256];
            visited[query as usize] = true;

            let mut frontier = vec![(query, 0u16, vec![query])];

            for _hop in 0..max_hops {
                let mut next_frontier = Vec::new();

                for (current, cumulative_dist, path) in &frontier {
                    let block_row = *current as usize / 4;
                    if block_row >= 64 {
                        continue;
                    }

                    // Which targets does `current` connect to in the specified layer?
                    let layer = if _hop == 0 { cause_layer } else { effect_layer };
                    let mask = self.planes[layer.min(7)][block_row];
                    let mut bits = mask;

                    while bits != 0 {
                        let block_col = bits.trailing_zeros() as usize;
                        bits &= bits - 1;

                        for sub in 0..4 {
                            let target = (block_col * 4 + sub) as u8;
                            if (target as usize) >= self.k || visited[target as usize] {
                                continue;
                            }

                            // Compose: what does the path through current→target produce?
                            let composed = self.semiring.compose(*current, target);
                            let step_dist = self.semiring.distance(*current, target);
                            let total = cumulative_dist + step_dist;

                            visited[target as usize] = true;
                            let mut new_path = path.clone();
                            new_path.push(target);
                            results.push((composed, total, new_path.clone()));
                            next_frontier.push((target, total, new_path));
                        }
                    }
                }

                if next_frontier.is_empty() {
                    break;
                }
                frontier = next_frontier;
            }

            results.sort_by_key(|(_, dist, _)| *dist);
            results
        }

        /// Density per layer.
        pub fn layer_densities(&self) -> [f32; 8] {
            let mut d = [0.0f32; 8];
            for z in 0..8 {
                let bits: u32 = self.planes[z].iter().map(|r| r.count_ones()).sum();
                d[z] = bits as f32 / 4096.0;
            }
            d
        }

        /// Memory footprint.
        pub fn topology_bytes(&self) -> usize {
            8 * 64 * 8 // 8 planes × 64 rows × 8 bytes per u64
        }

        pub fn metric_bytes(&self) -> usize {
            self.semiring.byte_size()
        }

        pub fn total_bytes(&self) -> usize {
            self.topology_bytes() + self.metric_bytes()
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use causal_edge::edge::{CausalEdge64, InferenceType};
    use causal_edge::pearl::CausalMask;
    use causal_edge::plasticity::PlasticityState;

    #[test]
    fn edge_to_palette_addressing() {
        let edge = CausalEdge64::pack(
            10,                          // S
            5,                           // P
            20,                          // O
            200,                         // frequency
            150,                         // confidence
            CausalMask::SO,              // causal: S+O active (association)
            0,                           // direction
            InferenceType::Deduction,    // inference
            PlasticityState::from_bits(0), // plasticity
            0,                           // temporal
        );

        let (row, col) = edge_to_block(&edge);
        assert_eq!(row, 2);  // 10/4
        assert_eq!(col, 5);  // 20/4

        let (f, c) = edge_nars_f32(&edge);
        assert!((f - 200.0 / 255.0).abs() < 0.01);
        assert!((c - 150.0 / 255.0).abs() < 0.01);

        let mask = edge_to_layer_mask(&edge);
        assert!(mask & (1 << CAUSES) != 0, "Deduction → CAUSES");

        eprintln!("Edge: S=10→row={row}, O=20→col={col}, f={f:.3}, c={c:.3}, layers={mask:08b}");
    }

    #[test]
    fn batch_edges_to_rows() {
        use causal_edge::edge::InferenceType;
        let infer_types = [
            InferenceType::Deduction, InferenceType::Induction,
            InferenceType::Abduction, InferenceType::Revision,
            InferenceType::Synthesis,
        ];

        let edges: Vec<CausalEdge64> = (0..50).map(|i| {
            CausalEdge64::pack(
                ((i * 7) % 256) as u8,
                0,
                ((i * 13 + 3) % 256) as u8,
                128, 128,
                CausalMask::from_bits((i % 7) as u8),
                0,
                infer_types[i % 5],
                PlasticityState::from_bits(0),
                0,
            )
        }).collect();

        let rows = edges_to_palette_rows(&edges);
        let nnz: u32 = rows.iter().map(|r| r.count_ones()).sum();
        eprintln!("50 edges → {} non-zero palette bits ({:.1}% density)", nnz, nnz as f64 / 4096.0 * 100.0);
        assert!(nnz > 0);

        let layers = edges_to_layered_rows(&edges);
        for z in 0..8 {
            let bits: u32 = layers[z].iter().map(|r| r.count_ones()).sum();
            if bits > 0 {
                eprintln!("  Layer {z} ({:12}): {} bits",
                    ["CAUSES","ENABLES","SUPPORTS","CONTRADICTS","REFINES","ABSTRACTS","GROUNDS","BECOMES"][z],
                    bits);
            }
        }
    }

    #[test]
    fn thinking_style_lazylock() {
        for i in 0..12 {
            let s = style_by_ordinal(i);
            eprintln!("{:<14} layers={:08b} combine={} contra={} density={:.2}",
                s.name, s.layer_mask, s.combine, s.contra, s.density_target);
        }

        let a = style_by_name("analytical");
        let c = style_by_name("creative");
        assert!(a.density_target < c.density_target);
        assert_eq!(a.combine, combine::INTERSECTION);
        assert_eq!(c.combine, combine::UNION);
    }

    #[test]
    fn semiring_mapping() {
        let (comb, ctr) = semiring_to_modes("BOOLEAN");
        assert_eq!(comb, combine::INTERSECTION);
        assert_eq!(ctr, contra::SUPPRESS);

        let (comb, ctr) = semiring_to_modes("XOR_BUNDLE");
        assert_eq!(comb, combine::UNION);
        assert_eq!(ctr, contra::INVERT);
    }

    #[test]
    fn blumenstrauss_cascade() {
        use super::blumenstrauss::Blumenstrauss;
        use bgz17::base17::Base17;
        use bgz17::palette::Palette;
        use bgz17::palette_semiring::PaletteSemiring;

        // Build a small palette: 16 entries with distinct Base17 patterns
        let entries: Vec<Base17> = (0..16).map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i * 100) as i16;  // spread on dim 0
            dims[1] = ((i * 37) % 200) as i16;
            Base17 { dims }
        }).collect();
        let palette = Palette { entries };
        let semiring = PaletteSemiring::build(&palette);

        // Build layered mask: CAUSES connects neighbors, ENABLES connects every other
        let mut planes = [[0u64; 64]; 8];
        for i in 0..4 {  // 16 entries / 4 = 4 blocks
            // CAUSES: each block connects to next block
            if i + 1 < 4 {
                planes[CAUSES][i] |= 1u64 << (i + 1);
            }
            // ENABLES: each block connects to block+2
            if i + 2 < 4 {
                planes[ENABLES][i] |= 1u64 << (i + 2);
            }
            // SUPPORTS: self-connection
            planes[SUPPORTS][i] |= 1u64 << i;
        }

        let b = Blumenstrauss::new(planes, &semiring);

        // Cascade from entry 0, radius = 5000 (wide), all layers
        let hits = b.cascade(0, 5000, 0xFF);

        eprintln!("\n=== Blumenstrauß Cascade ===");
        eprintln!("Query: archetype 0, radius=5000, all layers");
        eprintln!("Hits: {}", hits.len());
        for hit in &hits {
            eprintln!("  target={:3} dist={:5} predicates={:08b}", 
                hit.target, hit.distance, hit.predicates);
        }

        assert!(!hits.is_empty(), "Should find at least one hit");
        // All distances should be from bgz17 lookup, not POPCNT
        for hit in &hits {
            let expected_dist = semiring.distance(0, hit.target);
            assert_eq!(hit.distance, expected_dist, 
                "Distance should match bgz17 O(1) lookup for target {}", hit.target);
        }

        // Memory footprint
        eprintln!("Topology: {} bytes", b.topology_bytes());
        eprintln!("Metric:   {} bytes", b.metric_bytes());
        eprintln!("Total:    {} bytes", b.total_bytes());
        eprintln!("Densities: {:?}", b.layer_densities());
    }

    #[test]
    fn blumenstrauss_deduction() {
        use super::blumenstrauss::Blumenstrauss;
        use bgz17::base17::Base17;
        use bgz17::palette::Palette;
        use bgz17::palette_semiring::PaletteSemiring;

        let entries: Vec<Base17> = (0..16).map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i * 100) as i16;
            dims[1] = ((i * 37) % 200) as i16;
            Base17 { dims }
        }).collect();
        let palette = Palette { entries };
        let semiring = PaletteSemiring::build(&palette);

        // Chain: 0→1 (CAUSES), 1→2 (ENABLES), 2→3 (ENABLES)
        let mut planes = [[0u64; 64]; 8];
        planes[CAUSES][0] = 0b0010;   // block 0 → block 1
        planes[ENABLES][0] = 0b0100;  // block 0 → block 2
        planes[ENABLES][0] |= 0b1000; // block 0 → block 3

        let b = Blumenstrauss::new(planes, &semiring);

        let paths = b.deduce_path(0, CAUSES, ENABLES, 2);

        eprintln!("\n=== Blumenstrauß Deduction ===");
        eprintln!("Query: 0, CAUSES→ENABLES, max 2 hops");
        for (composed, dist, path) in &paths {
            eprintln!("  composed={composed} dist={dist} path={path:?}");
        }

        // Should find paths via compose(0, target) = O(1) algebra
        assert!(!paths.is_empty(), "Should find deduced paths");
    }
}
