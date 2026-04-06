//! Cognitive Trace: full provenance chain from text to thought.
//!
//! Every computation step is recorded. The trace IS the thought.
//! Reproducible, debuggable, replayable.

use crate::domino::StageResult;
use crate::superposition::{SuperpositionField, ThinkingStyle};
use crate::qualia::Qualia17D;

/// One SPO triple extracted from the superposition field.
#[derive(Clone, Debug)]
pub struct SpoTriple {
    pub subject: u16,       // centroid index
    pub predicate: &'static str, // relationship type
    pub object: u16,        // centroid index
    pub frequency: f32,     // NARS frequency (how strong)
    pub confidence: f32,    // NARS confidence (how agreed between lenses)
}

/// Complete cognitive trace from input to thought.
#[derive(Clone, Debug)]
pub struct CognitiveTrace {
    /// Input text.
    pub input: String,
    /// Token IDs from BPE.
    pub token_ids: Vec<u32>,
    /// Token strings.
    pub tokens: Vec<String>,

    /// Per-lens results.
    pub lens_results: Vec<LensTrace>,

    /// Superposition field from multi-lens interference.
    pub superposition: SuperpositionField,
    /// Detected thinking style.
    pub style: ThinkingStyle,
    /// Gated survivors (atoms that pass the interference filter).
    pub gated_atoms: Vec<u16>,

    /// Qualia from the interference pattern.
    pub qualia: Qualia17D,
    /// Emotional blend name.
    pub blend: String,
    /// Primary + overlay families.
    pub primary_family: String,
    pub overlay_family: String,

    /// Extracted SPO triples (confirmed relationships).
    pub spo_triples: Vec<SpoTriple>,

    /// Overall metrics.
    pub confidence: f32,
    pub dissonance: f32,
    pub staunen_max: f32,
    pub wisdom_max: f32,
}

/// Trace for one lens.
#[derive(Clone, Debug)]
pub struct LensTrace {
    pub name: String,
    pub dominant: u16,
    pub chain: Vec<u16>,
    pub dissonance: f32,
    pub staunen: f32,
    pub wisdom: f32,
    pub stages: Vec<StageResult>,
}

impl CognitiveTrace {
    /// Extract SPO triples from gated atom pairs.
    /// Each pair with >60% lens agreement becomes a triple.
    pub fn extract_spo(
        gated: &[u16],
        jina_dist_fn: impl Fn(u16, u16) -> u8,
        bge_dist_fn: impl Fn(u16, u16) -> u8,
        agreement_threshold: f32,
    ) -> Vec<SpoTriple> {
        let mut triples = Vec::new();

        for i in 0..gated.len() {
            for j in (i + 1)..gated.len() {
                let a = gated[i];
                let b = gated[j];
                let d_jina = jina_dist_fn(a, b);
                let d_bge = bge_dist_fn(a, b);
                let agreement = 1.0 - (d_jina as f32 - d_bge as f32).abs() / 255.0;

                if agreement < agreement_threshold { continue; }

                // Determine predicate from distance value
                let avg_dist = (d_jina as f32 + d_bge as f32) / 2.0;
                let predicate = if avg_dist > 200.0 {
                    "STRONGLY_RELATED"
                } else if avg_dist > 150.0 {
                    "RELATED"
                } else if avg_dist > 100.0 {
                    "WEAKLY_RELATED"
                } else if avg_dist < 50.0 {
                    "OPPOSED"
                } else {
                    "ORTHOGONAL"
                };

                // Frequency from distance, confidence from agreement
                let frequency = avg_dist / 255.0;
                let confidence = agreement;

                triples.push(SpoTriple {
                    subject: a,
                    predicate,
                    object: b,
                    frequency,
                    confidence,
                });
            }
        }

        // Sort by confidence descending
        triples.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        triples
    }

    /// Accumulate triples into a knowledge graph (append to file).
    pub fn append_to_knowledge_graph(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        for triple in &self.spo_triples {
            writeln!(file, "{}\t{}\t{}\t{:.3}\t{:.3}\t{}",
                triple.subject, triple.predicate, triple.object,
                triple.frequency, triple.confidence,
                self.input.replace('\t', " ").replace('\n', " "))?;
        }
        Ok(())
    }

    /// Summary string for display.
    pub fn summary(&self) -> String {
        format!(
            "\"{}\" → {} | {} | conf={:.0}% dis={:.2} | {} triples | {} style",
            &self.input[..self.input.len().min(40)],
            self.blend,
            if self.lens_results.len() >= 2 {
                if self.lens_results[0].dominant == self.lens_results[1].dominant {
                    "CONVERGE"
                } else { "DIVERGE" }
            } else { "SINGLE" },
            self.confidence * 100.0,
            self.dissonance,
            self.spo_triples.len(),
            self.style,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spo_extraction() {
        let gated = vec![10u16, 20, 30];
        let triples = CognitiveTrace::extract_spo(
            &gated,
            |a, b| if a == 10 && b == 20 { 200 } else { 128 },
            |a, b| if a == 10 && b == 20 { 190 } else { 130 },
            0.5,
        );
        // 10↔20: agreement = 1 - |200-190|/255 = 0.96 → included as highest confidence
        assert!(!triples.is_empty());
        // Find the 10↔20 triple (should be highest confidence)
        let t10_20 = triples.iter().find(|t| t.subject == 10 && t.object == 20).unwrap();
        assert_eq!(t10_20.predicate, "RELATED"); // avg 195 > 150
        assert!(t10_20.confidence > 0.9);
    }

    #[test]
    fn spo_filters_low_agreement() {
        let gated = vec![10u16, 20];
        let triples = CognitiveTrace::extract_spo(
            &gated,
            |_, _| 200, // Jina says close
            |_, _| 50,  // BGE says far
            0.6,
        );
        // agreement = 1 - |200-50|/255 = 0.41 < 0.6 → filtered out
        assert!(triples.is_empty());
    }
}
