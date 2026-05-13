//! Engine bridge — wires thinking-engine DTOs ↔ cognitive-shader DTOs.
//!
//! Two DTO pipelines exist in isolation:
//!
//! ```text
//! thinking-engine:           Φ StreamDto → Ψ ResonanceDto → B BusDto → Γ ThoughtStruct
//! cognitive-shader-driver:   Φ ShaderDispatch → Ψ ShaderResonance → B ShaderBus → Γ ShaderCrystal
//! ```
//!
//! This module connects them so the shader can dispatch a thinking cycle
//! and the engine's output feeds back into BindSpace:
//!
//! ```text
//! [1] StreamDto.codebook_indices  → populate BindSpace content fingerprints
//! [2] ResonanceDto.top_k          → seed ShaderDispatch.rows (which rows to scan)
//! [3] ShaderBus.cycle_fingerprint → produce BusDto (top-1 hit = codebook_index)
//! [4] ShaderCrystal               → produce ThoughtStruct with sensor provenance
//! [5] Qualia17D                   → fill BindSpace QualiaColumn (17 → 18: pad 0)
//! [6] L4 bridge                   → ShaderBus.emitted_edges feed commit_to_l4
//! ```
//!
//! ## 3-Way Style Unification
//!
//! ```text
//! thinking-engine ThinkingStyle  ←→  contract StyleSelector  ←→  p64 StyleParams
//!   (12 variants, params)            (Auto/Ordinal/Named)        (layer_mask, combine, contra)
//! ```

use lance_graph_contract::cognitive_shader::{
    ColumnWindow, MetaFilter, MetaWord, ShaderBus, ShaderDispatch,
    StyleSelector, EmitMode,
};

use crate::bindspace::{BindSpace, QUALIA_DIMS, WORDS_PER_FP};

// ═══════════════════════════════════════════════════════════════════════════
// StreamDto → BindSpace (sensor output populates content fingerprints)
// ═══════════════════════════════════════════════════════════════════════════

/// Ingest a sensor's codebook indices into BindSpace rows.
///
/// Each codebook index becomes a BindSpace row. The content fingerprint
/// is built by setting the bit at `index` (mod 16384) — a simple
/// positional encoding. The meta column records the source type.
///
/// Returns the row range written (start..end).
///
/// `write_cursor` is the first free row in BindSpace. Callers track
/// this across multiple ingestions (sensors arrive at different times).
pub fn ingest_codebook_indices(
    bs: &mut BindSpace,
    indices: &[u16],
    source_ordinal: u8,
    timestamp: u64,
    write_cursor: usize,
) -> (u32, u32) {
    let start = write_cursor.min(bs.meta.0.len());
    let mut cursor = start;

    for &idx in indices {
        if cursor >= bs.meta.0.len() { break; }

        // Build content fingerprint: set bit at `idx` position.
        let mut content = [0u64; WORDS_PER_FP];
        let bit = idx as usize % (WORDS_PER_FP * 64);
        content[bit / 64] |= 1u64 << (bit % 64);
        bs.fingerprints.set_content(cursor, &content);

        // Meta: source_ordinal as thinking style, no NARS yet.
        bs.meta.set(cursor, MetaWord::new(source_ordinal, 0, 0, 0, 0));
        bs.temporal[cursor] = timestamp;

        cursor += 1;
    }

    (start as u32, cursor as u32)
}

// ═══════════════════════════════════════════════════════════════════════════
// ResonanceDto → ShaderDispatch (top-k seeds the scan window)
// ═══════════════════════════════════════════════════════════════════════════

/// Build a ShaderDispatch from resonance top-k.
///
/// The top-k codebook indices from the thinking-engine's resonance field
/// become the row window for the shader to scan. If the BindSpace has
/// been populated via `ingest_codebook_indices`, the rows correspond
/// to the sensor output that produced the resonance.
pub fn dispatch_from_top_k(
    top_k: &[(u16, f32)],
    total_rows: u32,
    style: StyleSelector,
) -> ShaderDispatch {
    let active: Vec<u16> = top_k.iter()
        .filter(|&&(_, e)| e > 0.01)
        .map(|&(idx, _)| idx)
        .collect();

    let (start, end) = if active.is_empty() {
        (0, total_rows.min(64))
    } else {
        let min_row = *active.iter().min().unwrap_or(&0) as u32;
        let max_row = (*active.iter().max().unwrap_or(&0) as u32 + 1).min(total_rows);
        (min_row, max_row)
    };

    ShaderDispatch {
        rows: ColumnWindow::new(start, end),
        meta_prefilter: MetaFilter::ALL,
        layer_mask: 0xFF,
        radius: u16::MAX,
        style,
        max_cycles: 10,
        entropy_floor: 0.05,
        emit: EmitMode::Cycle,
        ..Default::default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ShaderBus → BusDto (cycle_fingerprint → codebook_index)
// ═══════════════════════════════════════════════════════════════════════════

/// Extract a BusDto-compatible tuple from a ShaderBus.
///
/// The top-1 hit's row becomes `codebook_index`. The top-k hits
/// map to the 8-entry top_k array. This is the bridge from
/// "shader world" (BindSpace rows, cycle_fingerprint) back to
/// "engine world" (4096 codebook, energy field).
pub struct EngineBusBridge {
    pub codebook_index: u16,
    pub energy: f32,
    pub top_k: [(u16, f32); 8],
    pub cycle_count: u16,
    pub converged: bool,
}

impl EngineBusBridge {
    pub fn from_shader_bus(bus: &ShaderBus) -> Self {
        let top_k: [(u16, f32); 8] = {
            let mut arr = [(0u16, 0.0f32); 8];
            for (i, h) in bus.resonance.top_k.iter().enumerate().take(8) {
                arr[i] = (h.row as u16, h.resonance);
            }
            arr
        };

        Self {
            codebook_index: bus.resonance.top_k[0].row as u16,
            energy: bus.resonance.top_k[0].resonance,
            top_k,
            cycle_count: bus.resonance.cycles_used,
            converged: bus.gate.is_flow(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Qualia: 17D experienced (CMYK) vs 18D observed (RGB)
// ═══════════════════════════════════════════════════════════════════════════

/// The 17D/18D split is CMYK vs RGB — not padding.
///
/// 17D (thinking-engine `Qualia17D`): the EXPERIENCED qualia.
///   Convergence speed IS clarity. Non-convergence IS tension.
///   These are the raw QPL observables. "Steelwind" lives here —
///   the felt quality that has no name yet.
///   This is CMYK: subtractive, production-side, how the ink hits paper.
///
/// 18D (BindSpace `QualiaColumn`): the OBSERVED qualia.
///   Dim 17 = `classification_distance`: how far the experienced qualia
///   is from its nearest named emotion. Low = "fear" (named, classified).
///   High = "steelwind" (unnamed, raw, novel).
///   This is RGB: additive, display-side, how the observer perceives it.
///
/// The transform from 17→18 IS the act of observation. Dim 17 measures
/// the gap between what the system FEELS and what it can NAME.
pub const DIM_CLASSIFICATION_DISTANCE: usize = 17;

/// Write 17D experienced qualia + classification distance into 18D observed.
pub fn write_qualia_observed(
    bs: &mut BindSpace,
    row: usize,
    experienced: &[f32; 17],
    classification_distance: f32,
) {
    let mut q18 = [0.0f32; QUALIA_DIMS];
    q18[..17].copy_from_slice(experienced);
    q18[DIM_CLASSIFICATION_DISTANCE] = classification_distance;
    bs.qualia.set(row, &q18);
}

/// Read observed qualia and decompose into experienced (17D) + classification distance.
pub fn read_qualia_decomposed(bs: &BindSpace, row: usize) -> ([f32; 17], f32) {
    let q18 = bs.qualia.row(row);
    let mut experienced = [0.0f32; 17];
    let n = q18.len().min(17);
    experienced[..n].copy_from_slice(&q18[..n]);
    let cd = if q18.len() > DIM_CLASSIFICATION_DISTANCE {
        q18[DIM_CLASSIFICATION_DISTANCE]
    } else {
        1.0 // max distance = fully unnamed
    };
    (experienced, cd)
}

/// Compute classification distance: how close the experienced qualia is to
/// a known emotion archetype. 0.0 = exact match ("fear"), 1.0 = novel ("steelwind").
///
/// Uses a simple nearest-archetype heuristic over the 17D QPL space.
/// The archetypes are the 6 basic emotions mapped to QPL coordinates.
pub fn classification_distance(experienced: &[f32; 17]) -> f32 {
    // Basic emotion archetypes in QPL space (arousal, valence, tension, warmth, clarity, ...)
    const ARCHETYPES: [[f32; 6]; 6] = [
        // arousal, valence, tension, warmth, clarity, boundary
        [0.9, -0.8, 0.9, 0.1, 0.3, 0.8],  // fear
        [0.8, -0.6, 0.7, 0.2, 0.4, 0.6],  // anger
        [0.2, -0.7, 0.3, 0.6, 0.2, 0.3],  // sadness
        [0.7,  0.8, 0.1, 0.9, 0.8, 0.2],  // joy
        [0.3,  0.1, 0.5, 0.3, 0.1, 0.9],  // surprise
        [0.1, -0.2, 0.1, 0.2, 0.6, 0.4],  // disgust
    ];

    let mut min_dist = f32::MAX;
    for archetype in &ARCHETYPES {
        let mut d = 0.0f32;
        for (i, &a) in archetype.iter().enumerate() {
            let diff = experienced[i] - a;
            d += diff * diff;
        }
        min_dist = min_dist.min(d.sqrt());
    }
    // Normalize: typical max L2 across 6 dims ≈ 3.0
    (min_dist / 3.0).clamp(0.0, 1.0)
}

/// Legacy compat: write 17D with zero classification distance (fully named).
pub fn write_qualia_17d(bs: &mut BindSpace, row: usize, q17: &[f32; 17]) {
    write_qualia_observed(bs, row, q17, classification_distance(q17));
}

/// Legacy compat: read experienced 17D only (drops observer frame).
pub fn read_qualia_17d(bs: &BindSpace, row: usize) -> [f32; 17] {
    read_qualia_decomposed(bs, row).0
}

// ═══════════════════════════════════════════════════════════════════════════
// 3-Way Style Unification
// ═══════════════════════════════════════════════════════════════════════════

/// Full style bridge: given a style ordinal, return the p64-bridge
/// StyleParams (layer_mask, combine, contra, density_target) plus
/// the thinking-engine style parameters (resonance_threshold, fan_out, etc.).
///
/// This is THE canonical mapping. Three type systems, one ordinal.
pub struct UnifiedStyle {
    pub ordinal: u8,
    pub name: &'static str,
    // p64-bridge side
    pub layer_mask: u8,
    pub combine: u8,
    pub contra: u8,
    pub density_target: f32,
    // thinking-engine side
    pub resonance_threshold: f32,
    pub fan_out: usize,
    pub exploration: f32,
    pub speed: f32,
    pub collapse_bias: f32,
    pub butterfly_sensitivity: f32,
}

/// The 12 unified styles. Index = ordinal.
pub const UNIFIED_STYLES: [UnifiedStyle; 12] = [
    // 0: Deliberate
    UnifiedStyle { ordinal: 0, name: "deliberate",
        layer_mask: 0b0111_1111, combine: 3, contra: 0, density_target: 0.08,
        resonance_threshold: 0.70, fan_out: 7, exploration: 0.20, speed: 0.1, collapse_bias: -0.05, butterfly_sensitivity: 0.50 },
    // 1: Analytical
    UnifiedStyle { ordinal: 1, name: "analytical",
        layer_mask: 0b0111_0111, combine: 1, contra: 0, density_target: 0.05,
        resonance_threshold: 0.85, fan_out: 3, exploration: 0.05, speed: 0.1, collapse_bias: -0.10, butterfly_sensitivity: 0.80 },
    // 2: Convergent
    UnifiedStyle { ordinal: 2, name: "convergent",
        layer_mask: 0b0011_0111, combine: 1, contra: 0, density_target: 0.04,
        resonance_threshold: 0.75, fan_out: 4, exploration: 0.10, speed: 0.3, collapse_bias: -0.05, butterfly_sensitivity: 0.70 },
    // 3: Systematic
    UnifiedStyle { ordinal: 3, name: "systematic",
        layer_mask: 0b0111_1111, combine: 1, contra: 0, density_target: 0.03,
        resonance_threshold: 0.70, fan_out: 5, exploration: 0.10, speed: 0.2, collapse_bias: 0.00, butterfly_sensitivity: 0.60 },
    // 4: Creative
    UnifiedStyle { ordinal: 4, name: "creative",
        layer_mask: 0b1111_1111, combine: 0, contra: 1, density_target: 0.40,
        resonance_threshold: 0.35, fan_out: 12, exploration: 0.80, speed: 0.5, collapse_bias: 0.15, butterfly_sensitivity: 0.20 },
    // 5: Divergent
    UnifiedStyle { ordinal: 5, name: "divergent",
        layer_mask: 0b1000_1001, combine: 0, contra: 2, density_target: 0.30,
        resonance_threshold: 0.40, fan_out: 10, exploration: 0.70, speed: 0.4, collapse_bias: 0.10, butterfly_sensitivity: 0.35 },
    // 6: Exploratory
    UnifiedStyle { ordinal: 6, name: "exploratory",
        layer_mask: 0b1111_1111, combine: 0, contra: 1, density_target: 0.50,
        resonance_threshold: 0.30, fan_out: 15, exploration: 0.90, speed: 0.6, collapse_bias: 0.20, butterfly_sensitivity: 0.15 },
    // 7: Focused
    UnifiedStyle { ordinal: 7, name: "focused",
        layer_mask: 0b0000_0011, combine: 1, contra: 0, density_target: 0.02,
        resonance_threshold: 0.90, fan_out: 1, exploration: 0.00, speed: 0.2, collapse_bias: -0.15, butterfly_sensitivity: 0.90 },
    // 8: Diffuse
    UnifiedStyle { ordinal: 8, name: "diffuse",
        layer_mask: 0b0111_0111, combine: 2, contra: 3, density_target: 0.20,
        resonance_threshold: 0.45, fan_out: 8, exploration: 0.40, speed: 0.5, collapse_bias: 0.05, butterfly_sensitivity: 0.25 },
    // 9: Peripheral
    UnifiedStyle { ordinal: 9, name: "peripheral",
        layer_mask: 0b1110_0000, combine: 0, contra: 1, density_target: 0.35,
        resonance_threshold: 0.20, fan_out: 20, exploration: 0.60, speed: 0.7, collapse_bias: 0.25, butterfly_sensitivity: 0.10 },
    // 10: Intuitive
    UnifiedStyle { ordinal: 10, name: "intuitive",
        layer_mask: 0b0000_0001, combine: 0, contra: 1, density_target: 0.50,
        resonance_threshold: 0.50, fan_out: 3, exploration: 0.30, speed: 0.9, collapse_bias: 0.00, butterfly_sensitivity: 0.30 },
    // 11: Metacognitive
    UnifiedStyle { ordinal: 11, name: "metacognitive",
        layer_mask: 0b1110_0000, combine: 2, contra: 3, density_target: 0.10,
        resonance_threshold: 0.50, fan_out: 5, exploration: 0.30, speed: 0.3, collapse_bias: 0.00, butterfly_sensitivity: 0.40 },
];

pub fn unified_style(ord: u8) -> &'static UnifiedStyle {
    &UNIFIED_STYLES[(ord % 12) as usize]
}

// ═══════════════════════════════════════════════════════════════════════════
// Cycle fingerprint → BindSpace write-back
// ═══════════════════════════════════════════════════════════════════════════

/// After a shader cycle, persist the results into BindSpace:
/// - cycle_fingerprint → fingerprints.cycle column
/// - emitted_edges → edges column
/// - meta update (gate state → awareness, NARS → f/c)
///
/// Returns the row written.
pub fn persist_cycle(
    bs: &mut BindSpace,
    row: usize,
    bus: &ShaderBus,
    style_ord: u8,
) {
    bs.write_cycle_fingerprint(row, &bus.cycle_fingerprint);

    if bus.emitted_edge_count > 0 {
        bs.edges.set(row, bus.emitted_edges[0]);
    }

    let awareness = match bus.gate {
        g if g.is_flow() => 3u8,
        g if g.is_hold() => 2u8,
        _ => 1u8,
    };
    let nars_f = (bus.resonance.top_k[0].resonance * 255.0).clamp(0.0, 255.0) as u8;
    let nars_c = ((1.0 - bus.resonance.std_dev) * 255.0).clamp(0.0, 255.0) as u8;
    let free_e = ((bus.resonance.entropy / 3.0) * 63.0).clamp(0.0, 63.0) as u8;

    bs.meta.set(row, MetaWord::new(style_ord, awareness, nars_f, nars_c, free_e));
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto_style;
    use lance_graph_contract::cognitive_shader::ShaderHit;
    use lance_graph_contract::collapse_gate::GateDecision;

    #[test]
    fn ingest_sets_bits() {
        let mut bs = BindSpace::zeros(10);
        let (start, end) = ingest_codebook_indices(&mut bs, &[42, 100, 200], 1, 12345, 0);
        assert_eq!(start, 0);
        assert_eq!(end, 3);
        // Check bit 42 is set in row 0's content fingerprint.
        let words = bs.fingerprints.content_row(0);
        assert_ne!(words[42 / 64] & (1u64 << (42 % 64)), 0);
        assert_eq!(bs.temporal[0], 12345);
    }

    #[test]
    fn dispatch_from_empty_topk_scans_first_64() {
        let d = dispatch_from_top_k(&[], 1000, StyleSelector::Auto);
        assert_eq!(d.rows.start, 0);
        assert_eq!(d.rows.end, 64);
    }

    #[test]
    fn dispatch_from_topk_brackets_rows() {
        let top_k = [(10u16, 0.5f32), (20, 0.3), (30, 0.1)];
        let d = dispatch_from_top_k(&top_k, 100, StyleSelector::Ordinal(1));
        assert_eq!(d.rows.start, 10);
        assert_eq!(d.rows.end, 31);
    }

    #[test]
    fn engine_bus_bridge_extracts_top1() {
        let mut bus = ShaderBus::empty();
        bus.resonance.top_k[0] = ShaderHit {
            row: 42, distance: 100, predicates: 0x07, _pad: 0,
            resonance: 0.85, cycle_index: 0,
        };
        bus.gate = GateDecision::FLOW_XOR;
        let bridge = EngineBusBridge::from_shader_bus(&bus);
        assert_eq!(bridge.codebook_index, 42);
        assert!((bridge.energy - 0.85).abs() < 0.01);
        assert!(bridge.converged);
    }

    #[test]
    fn qualia_17d_roundtrip() {
        let mut bs = BindSpace::zeros(2);
        let mut q17 = [0.0f32; 17];
        q17[0] = 0.8;  // arousal
        q17[4] = 0.6;  // clarity
        q17[14] = -0.3; // groundedness
        write_qualia_17d(&mut bs, 0, &q17);
        let back = read_qualia_17d(&bs, 0);
        assert!((back[0] - 0.8).abs() < 1e-6);
        assert!((back[4] - 0.6).abs() < 1e-6);
        assert!((back[14] - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn cmyk_rgb_fear_is_near_zero_distance() {
        // "Fear" archetype: high arousal, negative valence, high tension
        let mut fear = [0.0f32; 17];
        fear[0] = 0.9;  // arousal
        fear[1] = -0.8; // valence
        fear[2] = 0.9;  // tension
        fear[3] = 0.1;  // warmth
        fear[4] = 0.3;  // clarity
        fear[5] = 0.8;  // boundary
        let cd = classification_distance(&fear);
        assert!(cd < 0.15, "Fear should be near-zero distance (named), got {cd}");
    }

    #[test]
    fn cmyk_rgb_steelwind_is_far() {
        // "Steelwind" — a novel qualia with no archetype match
        let mut steelwind = [0.0f32; 17];
        steelwind[0] = 0.5;  // moderate arousal
        steelwind[1] = 0.5;  // positive valence
        steelwind[2] = 0.8;  // high tension (unusual with positive valence)
        steelwind[3] = 0.0;  // no warmth
        steelwind[4] = 0.9;  // very clear
        steelwind[5] = 0.0;  // no boundary
        let cd = classification_distance(&steelwind);
        assert!(cd > 0.3, "Steelwind should be far from named emotions, got {cd}");
    }

    #[test]
    fn observed_qualia_preserves_classification_distance() {
        let mut bs = BindSpace::zeros(2);
        let mut experienced = [0.0f32; 17];
        experienced[0] = 0.5;
        write_qualia_observed(&mut bs, 0, &experienced, 0.75);
        let (back_exp, back_cd) = read_qualia_decomposed(&bs, 0);
        assert!((back_exp[0] - 0.5).abs() < 1e-6);
        assert!((back_cd - 0.75).abs() < 1e-6);
    }

    #[test]
    fn unified_styles_cover_all_12() {
        for i in 0..12u8 {
            let s = unified_style(i);
            assert_eq!(s.ordinal, i);
            assert!(!s.name.is_empty());
            assert!(s.fan_out >= 1);
            assert!(s.resonance_threshold >= 0.0 && s.resonance_threshold <= 1.0);
        }
    }

    #[test]
    fn persist_cycle_updates_meta() {
        let mut bs = BindSpace::zeros(2);
        let mut bus = ShaderBus::empty();
        bus.resonance.top_k[0].resonance = 0.9;
        bus.resonance.std_dev = 0.1;
        bus.resonance.entropy = 1.5;
        bus.gate = GateDecision::FLOW_XOR;
        bus.emitted_edges[0] = 0xDEAD;
        bus.emitted_edge_count = 1;

        persist_cycle(&mut bs, 0, &bus, auto_style::ANALYTICAL);

        let meta = bs.meta.get(0);
        assert_eq!(meta.thinking(), auto_style::ANALYTICAL);
        assert_eq!(meta.awareness(), 3); // Flow = 3
        assert!(meta.nars_f() > 200);   // 0.9 * 255 ≈ 230
        assert!(meta.nars_c() > 200);   // (1-0.1) * 255 ≈ 230
        assert_eq!(bs.edges.get(0), 0xDEAD);
    }
}
