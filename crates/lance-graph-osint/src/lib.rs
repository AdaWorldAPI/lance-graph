//! # lance-graph-osint: OSINT Intelligence Pipeline
//!
//! Web → Triplets → Graph → Palette → Cache. No external LLM API.
//!
//! ```text
//! r.jina.ai/URL → Markdown
//!   → DeepNSM tokenize + embed → 512-bit VSA fingerprints
//!   → NARS-based triplet extraction (no LLM needed)
//!   → AriGraph TripletGraph (SPO + truth values)
//!   → convergence.rs → Palette64 layers
//!   → AutocompleteCache → 4096 heads × O(1)
//!
//! Cost: $0. Speed: 17K tokens/sec. Memory: 388 KB.
//! ```

pub mod reader;
pub mod extractor;
pub mod pipeline;
