//! OSINT Pipeline: URL → Triplets → Graph → Palette → Cache.
//!
//! No external LLM API. Runs 24/7 on CPU.
//! Uses local embeddings (DeepNSM/bgz7) for all intelligence.

use lance_graph_planner::cache::convergence;
use lance_graph_planner::cache::triple_model::Truth;

use crate::reader::{self, EmbeddedParagraph};
use crate::extractor::{self, Triplet};

/// Configuration for the OSINT pipeline.
pub struct OsintConfig {
    pub max_triplets: usize,
    pub confidence_threshold: f32,
}

impl Default for OsintConfig {
    fn default() -> Self {
        Self { max_triplets: 10000, confidence_threshold: 0.5 }
    }
}

/// The OSINT pipeline.
pub struct OsintPipeline {
    pub triplets: Vec<Triplet>,
    pub config: OsintConfig,
    pub clock: u64,
    pub processed_urls: Vec<String>,
}

impl OsintPipeline {
    pub fn new(config: OsintConfig) -> Self {
        Self {
            triplets: Vec::new(),
            config,
            clock: 0,
            processed_urls: Vec::new(),
        }
    }

    /// Ingest a URL: fetch → embed → extract triplets → refine graph.
    pub fn ingest_url(&mut self, url: &str) -> Result<IngestResult, String> {
        self.clock += 1;

        // 1. Fetch and embed (local, no API)
        let paragraphs = reader::fetch_and_embed(url)
            .map_err(|e| format!("{e}"))?;

        let mut new_triplets = 0;

        // 2. Extract triplets from each paragraph
        for para in &paragraphs {
            let extracted = extractor::extract_triplets(&para.text, self.clock);
            for t in &extracted {
                if t.truth.confidence >= self.config.confidence_threshold {
                    new_triplets += 1;
                }
            }
            // 3. NARS revision against existing knowledge
            extractor::refine_triplets(&mut self.triplets, &extracted);
        }

        // 4. Cap
        if self.triplets.len() > self.config.max_triplets {
            // Evict lowest confidence
            self.triplets.sort_by(|a, b| b.truth.confidence.partial_cmp(&a.truth.confidence).unwrap());
            self.triplets.truncate(self.config.max_triplets);
        }

        self.processed_urls.push(url.to_string());

        Ok(IngestResult {
            url: url.to_string(),
            paragraphs: paragraphs.len(),
            new_triplets,
            total_triplets: self.triplets.len(),
        })
    }

    /// Ingest raw text directly (no URL fetch).
    pub fn ingest_text(&mut self, text: &str) -> usize {
        self.clock += 1;
        let extracted = extractor::extract_triplets(text, self.clock);
        let count = extracted.len();
        extractor::refine_triplets(&mut self.triplets, &extracted);
        count
    }

    /// Export as p64 Palette layers for AutocompleteCache.
    pub fn to_palette_layers(&self) -> [[u64; 64]; 8] {
        let triplet_data: Vec<(String, String, String, f32)> = self.triplets.iter()
            .map(|t| (t.subject.clone(), t.relation.clone(), t.object.clone(), t.truth.frequency))
            .collect();
        convergence::triplets_to_palette_layers(&triplet_data)
    }

    /// Retrieve relevant triplets for a query.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<&Triplet> {
        let query_lower = query.to_lowercase();
        let mut matches: Vec<(&Triplet, usize)> = self.triplets.iter()
            .filter_map(|t| {
                let score = [&t.subject, &t.relation, &t.object].iter()
                    .map(|s| s.to_lowercase())
                    .filter(|s| {
                        query_lower.split_whitespace()
                            .any(|w| s.contains(w))
                    })
                    .count();
                if score > 0 { Some((t, score)) } else { None }
            })
            .collect();
        matches.sort_by(|a, b| b.1.cmp(&a.1));
        matches.truncate(top_k);
        matches.into_iter().map(|(t, _)| t).collect()
    }

    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            urls_processed: self.processed_urls.len(),
            total_triplets: self.triplets.len(),
            clock: self.clock,
        }
    }
}

#[derive(Debug)]
pub struct IngestResult {
    pub url: String,
    pub paragraphs: usize,
    pub new_triplets: usize,
    pub total_triplets: usize,
}

#[derive(Debug)]
pub struct PipelineStats {
    pub urls_processed: usize,
    pub total_triplets: usize,
    pub clock: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let p = OsintPipeline::new(OsintConfig::default());
        assert_eq!(p.stats().urls_processed, 0);
    }

    #[test]
    fn test_ingest_text() {
        let mut p = OsintPipeline::new(OsintConfig::default());
        let count = p.ingest_text("Albert Einstein developed the theory of relativity. Marie Curie discovered radium.");
        assert!(count > 0);
        assert!(p.triplets.len() > 0);
    }

    #[test]
    fn test_search() {
        let mut p = OsintPipeline::new(OsintConfig::default());
        p.ingest_text("Albert Einstein developed the theory of relativity.");
        let results = p.search("Einstein", 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_palette_export() {
        let mut p = OsintPipeline::new(OsintConfig::default());
        p.ingest_text("NARS causes inference. Pearl enables causality.");
        let layers = p.to_palette_layers();
        assert_eq!(layers.len(), 8);
        // Should have bits set in CAUSES layer
        assert!(layers[0].iter().any(|r| *r != 0));
    }

    #[test]
    #[ignore] // requires network
    fn test_ingest_url() {
        let mut p = OsintPipeline::new(OsintConfig::default());
        let result = p.ingest_url("https://example.com").unwrap();
        eprintln!("{:?}", result);
        assert!(result.paragraphs > 0);
    }
}
