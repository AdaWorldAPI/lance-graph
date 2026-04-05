//! ThinkingEngineBuilder: fluent API for engine construction.
//!
//! ```rust,no_run
//! let engine = ThinkingEngineBuilder::new()
//!     .lens(Lens::Jina)
//!     .table_type(TableType::SignedI8)
//!     .pooling(Pooling::TopK(5))
//!     .on_commit(|bus| l4.learn_from(bus))
//!     .build();
//! ```

use crate::engine::ThinkingEngine;
use crate::signed_engine::SignedThinkingEngine;
use crate::pooling::Pooling;

/// Thinking style presets — map to temperature + pooling.
/// From cognitive_stack.rs: 12 styles. Here: the 4 that matter for sampling.
#[derive(Clone, Copy, Debug)]
pub enum ThinkingPreset {
    /// Analytical: low temperature, narrow nucleus. Precise.
    Analytical,
    /// Creative: high temperature, wide nucleus. Exploratory.
    Creative,
    /// Metacognitive: adaptive, balanced.
    Balanced,
    /// Focused: argmax, no randomness. Deterministic.
    Focused,
}

impl ThinkingPreset {
    /// Convert to pooling strategy.
    pub fn to_pooling(self) -> Pooling {
        match self {
            ThinkingPreset::Analytical => Pooling::Nucleus {
                temperature: 0.3,
                top_p: 0.3,
                seed: None,
            },
            ThinkingPreset::Creative => Pooling::Nucleus {
                temperature: 1.2,
                top_p: 0.95,
                seed: None,
            },
            ThinkingPreset::Balanced => Pooling::Nucleus {
                temperature: 0.7,
                top_p: 0.9,
                seed: None,
            },
            ThinkingPreset::Focused => Pooling::ArgMax,
        }
    }
}

/// Which baked lens to use.
#[derive(Clone, Debug)]
pub enum Lens {
    Jina,
    BgeM3,
    Reranker,
    /// ModernBERT-large — not baked yet, needs stream_signed_lens first.
    /// GeGLU FFN, 28 layers, OLMo tokenizer (50K vocab).
    ModernBert,
    /// CLIP ViT-Huge-14 — vision sensor, not baked yet.
    ClipVision,
    Custom(Vec<u8>),
}

/// Distance table encoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TableType {
    UnsignedU8,
    SignedI8,
}

/// Built engine: either unsigned or signed.
pub enum BuiltEngine {
    Unsigned(ThinkingEngine),
    Signed(SignedThinkingEngine),
}

impl BuiltEngine {
    pub fn perturb(&mut self, indices: &[u16]) {
        match self {
            BuiltEngine::Unsigned(e) => e.perturb(indices),
            BuiltEngine::Signed(e) => e.perturb(indices),
        }
    }

    pub fn reset(&mut self) {
        match self {
            BuiltEngine::Unsigned(e) => e.reset(),
            BuiltEngine::Signed(e) => e.reset(),
        }
    }

    pub fn energy(&self) -> &[f32] {
        match self {
            BuiltEngine::Unsigned(e) => &e.energy,
            BuiltEngine::Signed(e) => &e.energy,
        }
    }

    pub fn cycles(&self) -> u16 {
        match self {
            BuiltEngine::Unsigned(e) => e.cycles,
            BuiltEngine::Signed(e) => e.cycles,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            BuiltEngine::Unsigned(e) => e.size,
            BuiltEngine::Signed(e) => e.size,
        }
    }

    pub fn think(&mut self, max_cycles: usize) {
        match self {
            BuiltEngine::Unsigned(e) => { e.think(max_cycles); }
            BuiltEngine::Signed(e) => { e.think(max_cycles); }
        }
    }
}

/// Commit sink: where committed thoughts go.
pub type CommitSink = Box<dyn Fn(&crate::dto::BusDto) + Send + Sync>;

/// Builder for ThinkingEngine with fluent API.
pub struct ThinkingEngineBuilder {
    lens: Option<Lens>,
    table_type: TableType,
    pooling: Pooling,
    max_cycles: usize,
    sinks: Vec<CommitSink>,
}

impl ThinkingEngineBuilder {
    pub fn new() -> Self {
        Self {
            lens: None,
            table_type: TableType::UnsignedU8,
            pooling: Pooling::ArgMax,
            max_cycles: 10,
            sinks: Vec::new(),
        }
    }

    /// Select a baked lens.
    pub fn lens(mut self, lens: Lens) -> Self {
        self.lens = Some(lens);
        self
    }

    /// Set table encoding type: u8 (default) or i8 signed.
    pub fn table_type(mut self, tt: TableType) -> Self {
        self.table_type = tt;
        self
    }

    /// Set pooling strategy.
    pub fn pooling(mut self, p: Pooling) -> Self {
        self.pooling = p;
        self
    }

    /// Apply a thinking preset (sets pooling from cognitive style).
    pub fn thinking_preset(mut self, preset: ThinkingPreset) -> Self {
        self.pooling = preset.to_pooling();
        self
    }

    /// Set max think cycles (default: 10).
    pub fn max_cycles(mut self, n: usize) -> Self {
        self.max_cycles = n;
        self
    }

    /// Add a commit sink (adapter pattern).
    /// Sinks receive the BusDto after every commit.
    pub fn on_commit(mut self, sink: impl Fn(&crate::dto::BusDto) + Send + Sync + 'static) -> Self {
        self.sinks.push(Box::new(sink));
        self
    }

    /// Build the engine.
    pub fn build(self) -> Result<ConfiguredEngine, String> {
        let table = match self.lens {
            Some(Lens::Jina) => crate::jina_lens::JINA_HDR_TABLE.to_vec(),
            Some(Lens::BgeM3) => crate::bge_m3_lens::BGE_M3_HDR_TABLE.to_vec(),
            Some(Lens::Reranker) => crate::reranker_lens::RERANKER_HDR_TABLE.to_vec(),
            Some(Lens::ModernBert) => return Err("ModernBERT lens not baked yet — run stream_signed_lens first".into()),
            Some(Lens::ClipVision) => return Err("CLIP vision lens not baked yet — needs ViT codebook".into()),
            Some(Lens::Custom(t)) => t,
            None => return Err("no lens specified".into()),
        };

        let engine = match self.table_type {
            TableType::UnsignedU8 => BuiltEngine::Unsigned(ThinkingEngine::new(table)),
            TableType::SignedI8 => BuiltEngine::Signed(
                crate::signed_engine::SignedThinkingEngine::from_unsigned(&table)
            ),
        };

        Ok(ConfiguredEngine {
            engine,
            pooling: self.pooling,
            max_cycles: self.max_cycles,
            sinks: self.sinks,
        })
    }
}

impl Default for ThinkingEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A fully configured engine with pooling and commit sinks.
pub struct ConfiguredEngine {
    pub engine: BuiltEngine,
    pub pooling: Pooling,
    pub max_cycles: usize,
    sinks: Vec<CommitSink>,
}

impl ConfiguredEngine {
    /// Full pipeline: perturb → think → pool → commit → notify sinks.
    pub fn process(&mut self, codebook_indices: &[u16]) -> crate::dto::BusDto {
        self.engine.reset();
        self.engine.perturb(codebook_indices);
        self.engine.think(self.max_cycles);

        let bus = self.pooling.to_bus(self.engine.energy(), self.engine.cycles());

        // Notify all sinks
        for sink in &self.sinks {
            sink(&bus);
        }

        bus
    }

    /// Access the underlying engine.
    pub fn inner(&self) -> &BuiltEngine {
        &self.engine
    }

    /// Access the pooling strategy.
    pub fn pooling(&self) -> &Pooling {
        &self.pooling
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicU32, Ordering}};

    #[test]
    fn builder_jina_unsigned() {
        let engine = ThinkingEngineBuilder::new()
            .lens(Lens::Jina)
            .build()
            .unwrap();
        assert_eq!(engine.engine.size(), 256);
    }

    #[test]
    fn builder_reranker_signed() {
        let engine = ThinkingEngineBuilder::new()
            .lens(Lens::Reranker)
            .table_type(TableType::SignedI8)
            .build()
            .unwrap();
        assert_eq!(engine.engine.size(), 256);
    }

    #[test]
    fn builder_with_pooling() {
        let mut engine = ThinkingEngineBuilder::new()
            .lens(Lens::Jina)
            .pooling(Pooling::TopK(3))
            .build()
            .unwrap();

        let bus = engine.process(&[50, 52, 54]);
        assert!(bus.energy > 0.0);
    }

    #[test]
    fn builder_with_sink() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut engine = ThinkingEngineBuilder::new()
            .lens(Lens::BgeM3)
            .on_commit(move |_bus| {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            })
            .build()
            .unwrap();

        engine.process(&[10, 20, 30]);
        engine.process(&[40, 50, 60]);

        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn builder_thinking_preset() {
        let mut engine = ThinkingEngineBuilder::new()
            .lens(Lens::Jina)
            .thinking_preset(ThinkingPreset::Creative)
            .build()
            .unwrap();

        let bus = engine.process(&[50, 52, 54]);
        assert!(bus.energy > 0.0);
        // Creative preset uses Nucleus pooling — may pick non-argmax peak
    }

    #[test]
    fn builder_no_lens_errors() {
        let result = ThinkingEngineBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_custom_table() {
        let mut table = vec![128u8; 64 * 64];
        for i in 0..64 { table[i * 64 + i] = 255; }

        let engine = ThinkingEngineBuilder::new()
            .lens(Lens::Custom(table))
            .table_type(TableType::SignedI8)
            .pooling(Pooling::Mean { threshold: 0.001 })
            .max_cycles(5)
            .build()
            .unwrap();

        assert_eq!(engine.engine.size(), 64);
    }

    #[test]
    fn builder_multiple_sinks() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let log1 = log.clone();
        let log2 = log.clone();

        let mut engine = ThinkingEngineBuilder::new()
            .lens(Lens::Jina)
            .on_commit(move |bus| {
                log1.lock().unwrap().push(format!("sink1:{}", bus.codebook_index));
            })
            .on_commit(move |bus| {
                log2.lock().unwrap().push(format!("sink2:{}", bus.codebook_index));
            })
            .build()
            .unwrap();

        engine.process(&[100]);
        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].starts_with("sink1:"));
        assert!(entries[1].starts_with("sink2:"));
    }
}
