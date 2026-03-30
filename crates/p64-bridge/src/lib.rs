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
}
