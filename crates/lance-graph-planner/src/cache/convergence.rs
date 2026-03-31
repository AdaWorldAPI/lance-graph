//! p64 convergence: AriGraph triplets → Palette64 → Blumenstrauss → AutocompleteCache.
//!
//! p64 is the convergence point where both hardware (ndarray) and thinking (lance-graph) meet.
//!
//! ```text
//! Cold path (columns/rows):
//!   AriGraph TripletGraph → SPO strings → DataFusion → Arrow
//!
//! Hot path (p64 palette):
//!   AriGraph Triplets → Base17 fingerprints → Palette → Blumenstrauss
//!     → 8 predicate layers × 64×64 attention = 4096 heads
//!     → CausalEdge64 forward/learn = O(1) per head
//!     → NarsTables revision = O(1) per truth update
//!
//! Convergence:
//!   Cold path BUILDS the graph (via LLM, slow)
//!   Hot path SERVES the graph (via palette, fast)
//!   p64 IS the bridge between them
//! ```

use super::kv_bundle::HeadPrint;
use super::nars_engine::{SpoHead, MASK_SPO, CausalEdge64};

/// Convert an SPO triplet (as strings) into a HeadPrint fingerprint.
///
/// Uses the same hash-to-Base17 approach as bgz17 label_fp,
/// producing a 17-dim i16 fingerprint from arbitrary text.
pub fn triplet_to_headprint(subject: &str, predicate: &str, object: &str) -> HeadPrint {
    let mut dims = [0i16; 17];
    // Subject → dims 0-5 (S-plane in Pearl SPO)
    for (i, b) in subject.bytes().enumerate() {
        dims[i % 6] = dims[i % 6].wrapping_add(b as i16 * 31);
    }
    // Predicate → dims 6-11 (P-plane)
    for (i, b) in predicate.bytes().enumerate() {
        dims[6 + i % 6] = dims[6 + i % 6].wrapping_add(b as i16 * 37);
    }
    // Object → dims 12-16 (O-plane)
    for (i, b) in object.bytes().enumerate() {
        dims[12 + i % 5] = dims[12 + i % 5].wrapping_add(b as i16 * 43);
    }
    HeadPrint { dims }
}

/// Convert a HeadPrint into an SpoHead with palette assignment.
///
/// Uses simple modular hashing for palette index assignment.
/// In production: use Palette::nearest() for proper CLAM assignment.
pub fn headprint_to_spo(fp: &HeadPrint, truth_f: f32, truth_c: f32) -> SpoHead {
    // S-plane hash → s_idx
    let s: i32 = fp.dims[0..6].iter().map(|d| *d as i32).sum();
    let s_idx = (s.unsigned_abs() % 256) as u8;
    // P-plane hash → p_idx
    let p: i32 = fp.dims[6..12].iter().map(|d| *d as i32).sum();
    let p_idx = (p.unsigned_abs() % 256) as u8;
    // O-plane hash → o_idx
    let o: i32 = fp.dims[12..17].iter().map(|d| *d as i32).sum();
    let o_idx = (o.unsigned_abs() % 256) as u8;

    SpoHead {
        s_idx, p_idx, o_idx,
        freq: (truth_f.clamp(0.0, 1.0) * 255.0) as u8,
        conf: (truth_c.clamp(0.0, 0.99) * 255.0) as u8,
        pearl: MASK_SPO,
        inference: 0, // deduction default
        temporal: 0,
    }
}

/// Build 8 p64 predicate layers from a set of SPO triplets.
///
/// Each triplet sets bits in the appropriate predicate layer:
///   Layer 0 CAUSES:      triplets with causal relations
///   Layer 1 ENABLES:     triplets with enabling relations
///   Layer 2 SUPPORTS:    triplets with supporting evidence
///   Layer 3 CONTRADICTS: triplets with contradicting evidence
///   Layer 4 REFINES:     triplets with refinement relations
///   Layer 5 ABSTRACTS:   triplets with abstraction relations
///   Layer 6 GROUNDS:     triplets with grounding evidence
///   Layer 7 BECOMES:     triplets with transformation relations
///
/// Returns [[u64; 64]; 8] ready for Blumenstrauss::new().
pub fn triplets_to_palette_layers(
    triplets: &[(String, String, String, f32)], // (subject, predicate, object, truth_freq)
) -> [[u64; 64]; 8] {
    let mut layers = [[0u64; 64]; 8];

    for (subject, predicate, object, freq) in triplets {
        let fp = triplet_to_headprint(subject, predicate, object);
        let spo = headprint_to_spo(&fp, *freq, 0.8);

        // Determine predicate layer from relation text
        let layer = classify_relation(predicate);

        // Set bit: row = s_idx % 64, col = o_idx % 64
        let row = (spo.s_idx % 64) as usize;
        let col = (spo.o_idx % 64) as usize;
        layers[layer][row] |= 1u64 << col;
    }

    layers
}

/// Classify a relation string into a p64 predicate layer index.
fn classify_relation(relation: &str) -> usize {
    let r = relation.to_lowercase();
    if r.contains("cause") || r.contains("lead") || r.contains("result") { 0 }      // CAUSES
    else if r.contains("enable") || r.contains("allow") || r.contains("permit") { 1 } // ENABLES
    else if r.contains("support") || r.contains("confirm") || r.contains("agree") { 2 } // SUPPORTS
    else if r.contains("contradict") || r.contains("deny") || r.contains("oppose") { 3 } // CONTRADICTS
    else if r.contains("refine") || r.contains("improve") || r.contains("update") { 4 } // REFINES
    else if r.contains("abstract") || r.contains("general") || r.contains("type") { 5 } // ABSTRACTS
    else if r.contains("ground") || r.contains("evidence") || r.contains("prove") { 6 } // GROUNDS
    else if r.contains("become") || r.contains("transform") || r.contains("change") { 7 } // BECOMES
    else { 0 } // default: CAUSES
}

/// Build a Blumenstrauss-ready structure from AriGraph episodic memory.
///
/// Takes a list of episodes (observation text) and extracts SPO triplets,
/// converts them to palette layers, ready for hot-path routing.
pub fn episodes_to_palette_layers(
    episodes: &[(String, Vec<(String, String, String)>, f32)], // (observation, triplets, recency)
) -> [[u64; 64]; 8] {
    let mut all_triplets = Vec::new();
    for (_, triplets, recency) in episodes {
        for (s, p, o) in triplets {
            all_triplets.push((s.clone(), p.clone(), o.clone(), *recency));
        }
    }
    triplets_to_palette_layers(&all_triplets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triplet_to_headprint() {
        let fp = triplet_to_headprint("Claude", "reasons_like", "Opus4.6");
        assert_ne!(fp.dims, [0i16; 17]);
        // S-plane (0-5) should be non-zero
        assert!(fp.dims[0..6].iter().any(|d| *d != 0));
        // P-plane (6-11) should be non-zero
        assert!(fp.dims[6..12].iter().any(|d| *d != 0));
        // O-plane (12-16) should be non-zero
        assert!(fp.dims[12..17].iter().any(|d| *d != 0));
    }

    #[test]
    fn test_headprint_to_spo() {
        let fp = triplet_to_headprint("Alice", "knows", "Bob");
        let spo = headprint_to_spo(&fp, 0.9, 0.8);
        assert!(spo.frequency() > 0.8);
        assert!(spo.confidence() > 0.7);
        assert_eq!(spo.pearl, MASK_SPO);
    }

    #[test]
    fn test_classify_relation() {
        assert_eq!(classify_relation("causes"), 0);
        assert_eq!(classify_relation("enables"), 1);
        assert_eq!(classify_relation("supports"), 2);
        assert_eq!(classify_relation("contradicts"), 3);
        assert_eq!(classify_relation("refines"), 4);
        assert_eq!(classify_relation("is type of"), 5);
        assert_eq!(classify_relation("grounds with evidence"), 6);
        assert_eq!(classify_relation("becomes"), 7);
    }

    #[test]
    fn test_triplets_to_palette_layers() {
        let triplets = vec![
            ("Claude".into(), "causes".into(), "reasoning".into(), 0.9),
            ("NARS".into(), "enables".into(), "inference".into(), 0.8),
            ("Pearl".into(), "supports".into(), "causality".into(), 0.85),
            ("v1".into(), "contradicts".into(), "v2".into(), 0.7),
        ];
        let layers = triplets_to_palette_layers(&triplets);

        // CAUSES layer should have bits set
        assert!(layers[0].iter().any(|row| *row != 0), "CAUSES should be populated");
        // ENABLES layer should have bits set
        assert!(layers[1].iter().any(|row| *row != 0), "ENABLES should be populated");
        // SUPPORTS layer should have bits set
        assert!(layers[2].iter().any(|row| *row != 0), "SUPPORTS should be populated");
        // CONTRADICTS layer should have bits set
        assert!(layers[3].iter().any(|row| *row != 0), "CONTRADICTS should be populated");
    }

    #[test]
    fn test_different_triplets_different_fingerprints() {
        let fp1 = triplet_to_headprint("Alice", "knows", "Bob");
        let fp2 = triplet_to_headprint("Charlie", "likes", "Diana");
        assert_ne!(fp1.dims, fp2.dims);
    }

    #[test]
    fn test_palette_layers_ready_for_blumenstrauss() {
        let triplets = vec![
            ("A".into(), "causes".into(), "B".into(), 0.9),
        ];
        let layers = triplets_to_palette_layers(&triplets);
        // layers is [[u64; 64]; 8] — exactly what Blumenstrauss::new() expects
        assert_eq!(layers.len(), 8);
        assert_eq!(layers[0].len(), 64);
    }
}
