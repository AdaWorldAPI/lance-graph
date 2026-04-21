//! `ShaderDriver` — the CognitiveShader IS the driver.
//!
//! Holds BindSpace columns (owned), the p64 CognitiveShader topology (8
//! predicate planes) plus its bgz17 PaletteSemiring (O(1) distance table),
//! and an optional sink. `dispatch()` runs one cycle end-to-end.
//!
//! ```text
//!   ShaderDispatch
//!         │
//!         ▼
//!  [1] meta prefilter  (cheap u32 column sweep)
//!  [2] resolve style   (auto-detect from qualia if Auto)
//!  [3] shader cascade  (p64 CognitiveShader + bgz17 distance)
//!  [4] cycle signature (Hamming-folded fingerprint of the top-k)
//!  [5] edge emission   (CausalEdge64 per strong hit)
//!  [6] CollapseGate    (Flow/Hold/Block from std-dev)
//!  [7] sink            (on_resonance → on_bus → on_crystal)
//!         │
//!         ▼
//!  ShaderCrystal
//! ```
//!
//! No forward pass, no JSON, no allocations beyond top-k + edges.

use std::sync::Arc;

use bgz17::palette_semiring::PaletteSemiring;
use causal_edge::edge::{CausalEdge64, InferenceType};
use causal_edge::pearl::CausalMask;
use causal_edge::plasticity::PlasticityState;
use lance_graph_contract::cognitive_shader::{
    CognitiveShaderDriver, EmitMode, MetaSummary, NullSink, ShaderBus, ShaderCrystal,
    ShaderDispatch, ShaderHit, ShaderResonance, ShaderSink,
};
use lance_graph_contract::collapse_gate::{GateDecision, MergeMode};
use p64_bridge::cognitive_shader::CognitiveShader;

use crate::auto_style;
use crate::bindspace::{BindSpace, WORDS_PER_FP};

// ═══════════════════════════════════════════════════════════════════════════
// ShaderDriver — holds everything the shader needs to drive
// ═══════════════════════════════════════════════════════════════════════════

/// The genius driver: CognitiveShader in the driving seat, BindSpace and
/// the bgz17 semiring in the back, thinking-engine optional.
pub struct ShaderDriver {
    pub(crate) bindspace: Arc<BindSpace>,
    pub(crate) semiring: Arc<PaletteSemiring>,
    pub(crate) planes: [[u64; 64]; 8],
    #[allow(dead_code)]
    pub(crate) default_style: u8,
}

impl ShaderDriver {
    /// Construct with BindSpace + semiring + 8 planes. Prefer the builder.
    pub fn new(
        bindspace: Arc<BindSpace>,
        semiring: Arc<PaletteSemiring>,
        planes: [[u64; 64]; 8],
        default_style: u8,
    ) -> Self {
        Self { bindspace, semiring, planes, default_style }
    }

    /// Borrow the underlying BindSpace (read-only).
    #[inline]
    pub fn bindspace(&self) -> &BindSpace { &self.bindspace }

    /// Borrow the topology planes (8 × 64 u64).
    #[inline]
    pub fn planes(&self) -> &[[u64; 64]; 8] { &self.planes }

    /// Run one dispatch, feeding a sink. This is the single hot path.
    fn run<S: ShaderSink>(&self, req: &ShaderDispatch, sink: &mut S) -> ShaderCrystal {
        // [1] Cheap meta prefilter (u32 column sweep).
        let passed_rows = self.bindspace.meta_prefilter(req.rows, &req.meta_prefilter);

        // [2] Resolve style — Auto reads the qualia of the FIRST surviving row.
        let qualia_seed = if let Some(&row) = passed_rows.first() {
            self.bindspace.qualia.row(row as usize)
        } else {
            // No rows passed — use default style; we'll still emit an empty cycle.
            &[0.0f32; 18][..]
        };
        let style_ord = auto_style::resolve(req.style, qualia_seed);

        // [3] Shader cascade — bgz17 O(1) per probed block.
        let shader = CognitiveShader::new(self.planes, &self.semiring);
        let max_dist = (self.semiring.k as f32) * (self.semiring.k as f32);
        let mut hits = Vec::<ShaderHit>::with_capacity(passed_rows.len().min(64));

        for (cycle_idx, &row) in passed_rows.iter().enumerate() {
            if cycle_idx as u16 >= req.max_cycles.saturating_mul(4) { break; }
            // Use the SPO `s_idx` of the row's edge as the query palette index.
            // Rows with edge=0 default to palette 0 (identity probe).
            let edge = CausalEdge64(self.bindspace.edges.get(row as usize));
            let query = edge.s_idx();
            let raw = shader.cascade(query, req.radius, req.layer_mask);
            for hit in raw.into_iter().take(4) {
                let resonance = 1.0 / (1.0 + (hit.distance as f32 / max_dist));
                hits.push(ShaderHit {
                    row,
                    distance: hit.distance,
                    predicates: hit.predicates,
                    _pad: 0,
                    resonance,
                    cycle_index: cycle_idx as u32,
                });
            }
        }

        // Sort by resonance descending, keep top-8.
        hits.sort_by(|a, b| b.resonance.partial_cmp(&a.resonance).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(8);

        // [4] Build the cycle_fingerprint by folding content rows of hits.
        let mut cycle_fp = [0u64; WORDS_PER_FP];
        for h in &hits {
            let row_words = self.bindspace.fingerprints.content_row(h.row as usize);
            for (i, w) in row_words.iter().enumerate() {
                cycle_fp[i] ^= *w;
            }
        }

        // [5] Entropy + std-dev of top-k resonances → CollapseGate.
        let (entropy, std_dev) = entropy_std(&hits);
        let gate = collapse_gate(std_dev);

        // [6] Emit one CausalEdge64 per strong hit (up to 8).
        let mut emitted = [0u64; 8];
        let mut emitted_n = 0u8;
        for h in hits.iter().take(8) {
            if h.resonance < 0.2 { continue; }
            let f = (h.resonance.clamp(0.0, 1.0) * 255.0) as u8;
            let c = (h.resonance.clamp(0.0, 1.0) * 255.0) as u8;
            let s_palette = (h.row % 256) as u8;
            let o_palette = ((h.row / 4) % 256) as u8;
            let edge = CausalEdge64::pack(
                s_palette,
                0,
                o_palette,
                f,
                c,
                CausalMask::from_bits(h.predicates & 0x07),
                0,
                style_ord_to_inference(style_ord),
                PlasticityState::from_bits(0),
                (h.cycle_index & 0xFFF) as u16,
            );
            emitted[emitted_n as usize] = edge.0;
            emitted_n += 1;
        }

        let mut top_k = [ShaderHit::default(); 8];
        for (i, h) in hits.iter().take(8).enumerate() {
            top_k[i] = *h;
        }

        let resonance_dto = ShaderResonance {
            top_k,
            hit_count: hits.len() as u16,
            cycles_used: passed_rows.len() as u16,
            entropy,
            std_dev,
            style_ord,
        };

        // [7] Sink callbacks.
        if !sink.on_resonance(&resonance_dto) {
            return ShaderCrystal {
                bus: ShaderBus {
                    cycle_fingerprint: cycle_fp,
                    emitted_edges: emitted,
                    emitted_edge_count: emitted_n,
                    gate,
                    resonance: resonance_dto,
                },
                persisted_row: None,
                meta: MetaSummary::default(),
            };
        }

        let bus = ShaderBus {
            cycle_fingerprint: cycle_fp,
            emitted_edges: emitted,
            emitted_edge_count: emitted_n,
            gate,
            resonance: resonance_dto,
        };
        if !sink.on_bus(&bus) {
            return ShaderCrystal { bus, persisted_row: None, meta: MetaSummary::default() };
        }

        // Meta summary (confidence from top-1 resonance, simple surrogate).
        let confidence = resonance_dto.top_k[0].resonance;
        let meta = MetaSummary {
            confidence,
            meta_confidence: (1.0 - std_dev).clamp(0.0, 1.0),
            brier: 0.0,
            should_admit_ignorance: confidence < 0.2,
        };

        let persisted_row = match req.emit {
            EmitMode::Persist => Some(resonance_dto.top_k[0].row),
            _ => None,
        };

        let crystal = ShaderCrystal { bus, persisted_row, meta };
        sink.on_crystal(&crystal);
        crystal
    }
}

impl CognitiveShaderDriver for ShaderDriver {
    fn dispatch(&self, req: &ShaderDispatch) -> ShaderCrystal {
        let mut null = NullSink;
        self.run(req, &mut null)
    }

    fn dispatch_with_sink<S: ShaderSink>(&self, req: &ShaderDispatch, sink: &mut S) -> ShaderCrystal {
        self.run(req, sink)
    }

    fn row_count(&self) -> u32 { self.bindspace.len as u32 }

    fn byte_footprint(&self) -> usize {
        self.bindspace.byte_footprint()
            + 8 * 64 * 8                           // planes: 4096 bytes
            + self.semiring.compose_table.len()    // k×k u8
            + self.semiring.distance_matrix.byte_size() // k×k u16
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder — EmbedAnything fluent pattern
// ═══════════════════════════════════════════════════════════════════════════

/// Fluent builder for `ShaderDriver`.
///
/// ```rust,ignore
/// let driver = CognitiveShaderBuilder::new()
///     .bindspace(Arc::new(my_bs))
///     .semiring(Arc::new(sr))
///     .planes(planes)
///     .default_style(auto_style::ANALYTICAL)
///     .build();
/// ```
pub struct CognitiveShaderBuilder {
    bindspace: Option<Arc<BindSpace>>,
    semiring: Option<Arc<PaletteSemiring>>,
    planes: Option<[[u64; 64]; 8]>,
    default_style: u8,
}

impl CognitiveShaderBuilder {
    pub fn new() -> Self {
        Self {
            bindspace: None,
            semiring: None,
            planes: None,
            default_style: auto_style::DELIBERATE,
        }
    }

    pub fn bindspace(mut self, bs: Arc<BindSpace>) -> Self {
        self.bindspace = Some(bs);
        self
    }

    pub fn semiring(mut self, sr: Arc<PaletteSemiring>) -> Self {
        self.semiring = Some(sr);
        self
    }

    pub fn planes(mut self, p: [[u64; 64]; 8]) -> Self {
        self.planes = Some(p);
        self
    }

    pub fn default_style(mut self, ord: u8) -> Self {
        self.default_style = ord.min(11);
        self
    }

    pub fn build(self) -> ShaderDriver {
        ShaderDriver {
            bindspace: self.bindspace.expect("bindspace required"),
            semiring: self.semiring.expect("semiring required"),
            planes: self.planes.unwrap_or([[0u64; 64]; 8]),
            default_style: self.default_style,
        }
    }
}

impl Default for CognitiveShaderBuilder {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn entropy_std(hits: &[ShaderHit]) -> (f32, f32) {
    if hits.is_empty() { return (0.0, 0.0); }
    let sum: f32 = hits.iter().map(|h| h.resonance).sum();
    if sum <= 0.0 { return (0.0, 0.0); }
    let mut ent = 0.0f32;
    for h in hits {
        let p = h.resonance / sum;
        if p > 1e-9 { ent -= p * p.ln(); }
    }
    let mean = sum / hits.len() as f32;
    let var: f32 = hits.iter()
        .map(|h| (h.resonance - mean).powi(2))
        .sum::<f32>() / hits.len() as f32;
    (ent, var.sqrt())
}

fn collapse_gate(sd: f32) -> GateDecision {
    // Matches thinking_engine::cognitive_stack::{SD_FLOW_THRESHOLD, SD_BLOCK_THRESHOLD}.
    const FLOW: f32 = 0.15;
    const BLOCK: f32 = 0.35;
    if sd < FLOW { GateDecision { gate: 0, merge: MergeMode::Xor } }
    else if sd > BLOCK { GateDecision::BLOCK }
    else { GateDecision::HOLD }
}

fn style_ord_to_inference(ord: u8) -> InferenceType {
    // analytical/convergent/systematic → Deduction
    // creative/divergent/exploratory   → Induction
    // focused/diffuse/peripheral       → Abduction
    // intuitive/deliberate             → Revision
    // metacognitive                    → Synthesis
    match ord {
        1..=3 => InferenceType::Deduction,
        4..=6 => InferenceType::Induction,
        7..=9 => InferenceType::Abduction,
        0 | 10    => InferenceType::Revision,
        _         => InferenceType::Synthesis,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindspace::{BindSpaceBuilder, QUALIA_DIMS, WORDS_PER_FP};
    use bgz17::base17::Base17;
    use bgz17::palette::Palette;
    use lance_graph_contract::cognitive_shader::{
        ColumnWindow, MetaFilter, ShaderDispatch, StyleSelector,
    };
    use lance_graph_contract::cognitive_shader::MetaWord;

    fn demo_bindspace() -> BindSpace {
        let q = [0.0f32; QUALIA_DIMS];
        let content = [0u64; WORDS_PER_FP];
        BindSpaceBuilder::new(4)
            .push(&content, MetaWord::new(1, 1, 200, 200, 5), 0, &q, 0, 0)
            .push(&content, MetaWord::new(2, 2, 100, 100, 5), 0, &q, 0, 0)
            .push(&content, MetaWord::new(3, 3,  50,  50, 5), 0, &q, 0, 0)
            .push(&content, MetaWord::new(4, 4,   0,   0, 5), 0, &q, 0, 0)
            .build()
    }

    fn demo_semiring() -> PaletteSemiring {
        let entries: Vec<Base17> = (0..16).map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i as i16) * 100;
            dims[1] = ((i as i16) * 37) % 200;
            Base17 { dims }
        }).collect();
        let palette = Palette { entries };
        PaletteSemiring::build(&palette)
    }

    fn demo_planes() -> [[u64; 64]; 8] {
        let mut planes = [[0u64; 64]; 8];
        for i in 0..4 {
            if i + 1 < 4 { planes[0][i] |= 1u64 << (i + 1); } // CAUSES
            planes[2][i] |= 1u64 << i;                        // SUPPORTS self
        }
        planes
    }

    #[test]
    fn driver_builder_builds() {
        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .default_style(auto_style::ANALYTICAL)
            .build();
        assert_eq!(driver.row_count(), 4);
        assert!(driver.byte_footprint() > 0);
    }

    #[test]
    fn dispatch_runs_and_emits_cycle_fingerprint() {
        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();

        let req = ShaderDispatch {
            rows: ColumnWindow::new(0, 4),
            meta_prefilter: MetaFilter::ALL,
            layer_mask: 0xFF,
            radius: u16::MAX,
            style: StyleSelector::Ordinal(auto_style::ANALYTICAL),
            ..Default::default()
        };
        let crystal = driver.dispatch(&req);
        assert_eq!(crystal.bus.resonance.style_ord, auto_style::ANALYTICAL);
    }

    #[test]
    fn dispatch_with_prefilter_excludes_rows() {
        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();

        let tight = MetaFilter { nars_c_min: 150, ..MetaFilter::ALL };
        let req = ShaderDispatch {
            rows: ColumnWindow::new(0, 4),
            meta_prefilter: tight,
            ..Default::default()
        };
        let crystal = driver.dispatch(&req);
        // Only row 0 passes (c=200). It produces at most a few hits.
        assert!(crystal.bus.resonance.cycles_used <= 1);
    }

    #[test]
    fn sink_short_circuits_on_false() {
        struct Stop;
        impl ShaderSink for Stop {
            fn on_resonance(&mut self, _r: &ShaderResonance) -> bool { false }
        }

        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();

        let mut stop = Stop;
        let req = ShaderDispatch { rows: ColumnWindow::new(0, 4), ..Default::default() };
        let crystal = driver.dispatch_with_sink(&req, &mut stop);
        // Short-circuited → persisted_row is None, meta is default.
        assert!(crystal.persisted_row.is_none());
        assert_eq!(crystal.meta.confidence, 0.0);
    }
}
