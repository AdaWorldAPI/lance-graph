//! DTOs: bus adapters between cognitive speed zones.
//!
//! Φ Dispersion:   StreamDto      — sensor output enters the field
//! Ψ Interference: ResonanceDto   — the ripple field IS f64[4096]
//! B Consequence:   BusDto         — committed thought with provenance
//! Γ Collapse:      ThoughtStruct  — stabilized, persisted, text is lazy

use crate::engine::CODEBOOK_SIZE;

/// Source of a perturbation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceType {
    Jina,       // Jina v3 embedding (API or codebook lookup)
    BgeM3,      // BGE-M3 multilingual embedding
    ReaderLm,   // reader-LM HTML→markdown
    Qwen,       // Qwen 27B + Opus distilled
    DeepNsm,    // distributional semantics (COCA co-occurrence)
    Wikidata,   // entity type prototypes
    AriGraph,   // graph-derived persona
    HiDream,    // image generation latent
    User,       // direct user input
}

/// Thinking scale (from gate stride or convergence pattern).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingScale {
    Exploiting,  // fast, narrow, confident
    Focused,     // careful, detailed
    Exploring,   // broad, routing
    Abstract,    // meta-level
}

// ═══════════════════════════════════════════════════════════════════════════
// Φ — StreamDto: sensor output enters the field
// ═══════════════════════════════════════════════════════════════════════════

/// Sensor output. Carries codebook indices, not raw vectors.
/// Multiple sensors can fire StreamDtos into the same engine.
#[derive(Clone, Debug)]
pub struct StreamDto {
    /// Which sensor produced this.
    pub source: SourceType,
    /// Codebook indices this sensor activated.
    pub codebook_indices: Vec<u16>,
    /// Timestamp (monotonic counter or epoch ms).
    pub timestamp: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Ψ — ResonanceDto: the ripple field
// ═══════════════════════════════════════════════════════════════════════════

/// ResonanceDto IS f64[4096] energy. Not a struct with candidate lists.
///
/// High energy at entry 42 = "thought 42 resonates."
/// Zero at entry 200 = "thought 200 destructively interfered."
/// Spike at entry 7 = "thought 7 crystallizing."
#[derive(Clone, Debug)]
pub struct ResonanceDto {
    /// Energy distribution. f32 — matches u8 distance table precision.
    pub energy: Vec<f32>,
    pub cycle_count: u16,
    pub converged: bool,
    pub top_k: [(u16, f32); 8],
}

impl ResonanceDto {
    /// Build from f32 energy array (fixed-size legacy compat).
    pub fn from_energy(energy: &[f32; CODEBOOK_SIZE], cycles: u16) -> Self {
        Self::from_energy_f32(energy.as_slice(), cycles)
    }

    /// Build from f32 energy slice.
    pub fn from_energy_f32(energy: &[f32], cycles: u16) -> Self {
        let mut indexed: Vec<(usize, f32)> = energy.iter()
            .enumerate()
            .map(|(i, &e)| (i, e))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut top_k = [(0u16, 0.0f32); 8];
        for (k, &(idx, val)) in indexed.iter().take(8).enumerate() {
            top_k[k] = (idx as u16, val);
        }

        Self { energy: energy.to_vec(), cycle_count: cycles, converged: cycles < 10, top_k }
    }

    /// Legacy: build from f64 slice (converts to f32).
    pub fn from_energy_vec(energy: &[f64], cycles: u16) -> Self {
        let f32_energy: Vec<f32> = energy.iter().map(|&e| e as f32).collect();
        Self::from_energy_f32(&f32_energy, cycles)
    }

    pub fn entropy(&self) -> f32 {
        let mut h = 0.0f32;
        for &e in &self.energy {
            if e > 1e-10 { h -= e * e.ln(); }
        }
        h
    }

    pub fn active_count(&self, threshold: f32) -> usize {
        self.energy.iter().filter(|&&e| e > threshold).count()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// B — BusDto: committed thought
// ═══════════════════════════════════════════════════════════════════════════

/// The first accountable structured thought.
/// Dominant peak of the resonance field + provenance.
#[derive(Clone, Debug)]
pub struct BusDto {
    pub codebook_index: u16,
    pub energy: f32,
    pub top_k: [(u16, f32); 8],
    pub cycle_count: u16,
    pub converged: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// Γ — ThoughtStruct: stabilized, persisted
// ═══════════════════════════════════════════════════════════════════════════

/// Stabilized thought. Text is LAZY display, not the thought.
#[derive(Clone, Debug)]
pub struct ThoughtStruct {
    /// The committed thought.
    pub bus: BusDto,
    /// Lazy text rendering. Only generated when needed.
    pub text: Option<String>,
    /// Which sensors contributed and what indices they produced.
    pub sensor_contributions: Vec<(SourceType, Vec<u16>)>,
    /// Energy snapshots per cycle (for trajectory analysis).
    pub tension_history: Vec<Vec<f32>>,
    /// Thinking scale trajectory.
    pub style_trajectory: Vec<ThinkingScale>,
}

impl ThoughtStruct {
    /// Build from engine state + sensor history.
    pub fn from_engine(
        bus: BusDto,
        contributions: Vec<(SourceType, Vec<u16>)>,
    ) -> Self {
        Self {
            bus,
            text: None,
            sensor_contributions: contributions,
            tension_history: Vec::new(),
            style_trajectory: Vec::new(),
        }
    }

    /// Set lazy text.
    pub fn with_text(mut self, text: String) -> Self {
        self.text = Some(text);
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ThoughtIndex: SoA for searching thoughts
// ═══════════════════════════════════════════════════════════════════════════

/// Structure-of-Arrays for cognitive search across many thoughts.
/// Same principle as NeuronIndex: AoS for API, SoA for search.
pub struct ThoughtIndex {
    pub codebook_index: Vec<u16>,
    pub energy: Vec<f32>,
    pub style: Vec<ThinkingScale>,
    pub source: Vec<SourceType>,
    pub timestamp: Vec<u64>,
    pub converged: Vec<bool>,
    pub cycle_count: Vec<u16>,
}

impl ThoughtIndex {
    pub fn new() -> Self {
        Self {
            codebook_index: Vec::new(), energy: Vec::new(),
            style: Vec::new(), source: Vec::new(),
            timestamp: Vec::new(), converged: Vec::new(),
            cycle_count: Vec::new(),
        }
    }

    pub fn push(&mut self, thought: &ThoughtStruct, primary_source: SourceType, ts: u64) {
        self.codebook_index.push(thought.bus.codebook_index);
        self.energy.push(thought.bus.energy);
        self.style.push(thought.style_trajectory.last().copied()
            .unwrap_or(ThinkingScale::Focused));
        self.source.push(primary_source);
        self.timestamp.push(ts);
        self.converged.push(thought.bus.converged);
        self.cycle_count.push(thought.bus.cycle_count);
    }

    pub fn len(&self) -> usize { self.codebook_index.len() }
    pub fn is_empty(&self) -> bool { self.codebook_index.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_dto_basic() {
        let stream = StreamDto {
            source: SourceType::Jina,
            codebook_indices: vec![42, 100, 200],
            timestamp: 12345,
        };
        assert_eq!(stream.codebook_indices.len(), 3);
    }

    #[test]
    fn resonance_dto_from_energy() {
        let mut energy = [0.0f32; CODEBOOK_SIZE];
        energy[42] = 0.5;
        energy[100] = 0.3;
        energy[200] = 0.2;

        let res = ResonanceDto::from_energy(&energy, 5);
        assert_eq!(res.top_k[0].0, 42);
        assert!((res.top_k[0].1 - 0.5).abs() < 1e-10);
        assert_eq!(res.top_k[1].0, 100);
        assert_eq!(res.cycle_count, 5);
        assert!(res.converged); // < 10 cycles
    }

    #[test]
    fn bus_dto_from_resonance() {
        let bus = BusDto {
            codebook_index: 42,
            energy: 0.5,
            top_k: [(42, 0.5), (100, 0.3), (200, 0.2),
                     (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)],
            cycle_count: 5,
            converged: true,
        };
        assert_eq!(bus.codebook_index, 42);
    }

    #[test]
    fn thought_struct_lazy_text() {
        let bus = BusDto {
            codebook_index: 42, energy: 0.5,
            top_k: [(42, 0.5); 8], cycle_count: 3, converged: true,
        };
        let thought = ThoughtStruct::from_engine(bus, vec![])
            .with_text("The cat sat on the mat.".into());
        assert_eq!(thought.text.as_deref(), Some("The cat sat on the mat."));
    }

    #[test]
    fn thought_index_soa() {
        let mut idx = ThoughtIndex::new();
        let bus = BusDto {
            codebook_index: 42, energy: 0.5,
            top_k: [(42, 0.5); 8], cycle_count: 3, converged: true,
        };
        let thought = ThoughtStruct::from_engine(bus, vec![
            (SourceType::Jina, vec![42]),
        ]);

        idx.push(&thought, SourceType::Jina, 12345);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.codebook_index[0], 42);
    }
}
