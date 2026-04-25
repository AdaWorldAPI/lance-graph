//! p64 convergence: AriGraph triplets → Palette64 → CognitiveShader → AutocompleteCache.
//!
//! p64 is the convergence point where both hardware (ndarray) and thinking (lance-graph) meet.
//!
//! ```text
//! Cold path (columns/rows):
//!   AriGraph TripletGraph → SPO strings → DataFusion → Arrow
//!
//! Hot path (p64 palette):
//!   AriGraph Triplets → Base17 fingerprints → Palette → CognitiveShader
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
use ndarray::hpc::palette_distance::{Palette, DistanceMatrix, SpoDistanceMatrices};
use ndarray::hpc::bgz17_bridge::SpoBase17;

/// Per-plane palette distance context (TD-INT-5).
///
/// Wraps ndarray's `SpoDistanceMatrices` (pre-computed 256×256 per-plane
/// L1 distance tables). All comparison algebra lives in ndarray; this
/// struct is a session-scoped handle that the planner cache and cascade
/// use to compare triplets on individual role planes (subject-only,
/// predicate-only, etc.) without doing bit ops in lance-graph.
///
/// Build once from the palette codebooks; compare in O(1) per pair.
pub struct PlaneDistance {
    matrices: SpoDistanceMatrices,
}

impl PlaneDistance {
    /// Build from three palettes (one per S/P/O plane).
    pub fn build(s_pal: &Palette, p_pal: &Palette, o_pal: &Palette) -> Self {
        Self { matrices: SpoDistanceMatrices::build(s_pal, p_pal, o_pal) }
    }

    /// Combined S+P+O distance. O(1): three table lookups.
    #[inline]
    pub fn spo_distance(&self, a: &SpoHead, b: &SpoHead) -> u32 {
        self.matrices.spo_distance(a.s_idx, a.p_idx, a.o_idx, b.s_idx, b.p_idx, b.o_idx)
    }

    /// Subject-plane only distance. O(1): one table lookup.
    #[inline]
    pub fn subject_distance(&self, a: &SpoHead, b: &SpoHead) -> u16 {
        self.matrices.subject.distance(a.s_idx, b.s_idx)
    }

    /// Predicate-plane only distance. O(1): one table lookup.
    #[inline]
    pub fn predicate_distance(&self, a: &SpoHead, b: &SpoHead) -> u16 {
        self.matrices.predicate.distance(a.p_idx, b.p_idx)
    }

    /// Object-plane only distance. O(1): one table lookup.
    #[inline]
    pub fn object_distance(&self, a: &SpoHead, b: &SpoHead) -> u16 {
        self.matrices.object.distance(a.o_idx, b.o_idx)
    }
}

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
/// Returns [[u64; 64]; 8] ready for CognitiveShader::new().
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

/// Run the convergence highway: AriGraph triplets → palette planes → caller.
///
/// This is the TD-INT-14 closure: newly committed SPO knowledge goes from
/// the cold-path AriGraph (where the LLM commits triples) to the hot-path
/// `[[u64; 64]; 8]` topology that `CognitiveShader` cascades over. Without
/// this function the shader keeps the construction-time demo planes forever.
///
/// The shader-driver crate cannot depend on the planner (would create a
/// dependency cycle), so the convergence call lives here and the caller
/// passes a closure that knows how to apply the new planes — typically
/// `|p| driver.update_planes(p)`.
///
/// # Example
///
/// ```ignore
/// use lance_graph_planner::cache::convergence::run_convergence;
///
/// let triplets = vec![
///     ("Claude".into(), "reasons_about".into(), "physics".into(), 0.9),
/// ];
/// run_convergence(&triplets, |planes| driver.update_planes(planes));
/// ```
pub fn run_convergence(
    triplets: &[(String, String, String, f32)],
    apply: impl FnOnce([[u64; 64]; 8]),
) {
    let planes = triplets_to_palette_layers(triplets);
    apply(planes);
}

/// Build a CognitiveShader-ready structure from AriGraph episodic memory.
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
    fn test_plane_distance_subject_only() {
        // Build a 256-entry palette (production size) from spread Base17 patterns
        let patterns: Vec<ndarray::hpc::bgz17_bridge::Base17> = (0..256)
            .map(|i| {
                let mut b = ndarray::hpc::bgz17_bridge::Base17::zero();
                b.dims[0] = (i as i16).wrapping_mul(127);
                b.dims[1] = (i as i16).wrapping_mul(31);
                b
            })
            .collect();
        let pal = Palette::build(&patterns, 256, 1);
        let pd = PlaneDistance::build(&pal, &pal, &pal);

        // Same palette index → zero distance
        let a = headprint_to_spo(&triplet_to_headprint("Alice", "knows", "Bob"), 0.9, 0.8);
        assert_eq!(pd.subject_distance(&a, &a), 0);
        assert_eq!(pd.spo_distance(&a, &a), 0);

        // Different triplets → likely nonzero distance
        let b = headprint_to_spo(&triplet_to_headprint("Zephyr", "loves", "Qux"), 0.9, 0.8);
        let combined = pd.spo_distance(&a, &b);
        let sub_only = pd.subject_distance(&a, &b);
        // Subject-only ≤ combined (since combined adds P + O)
        assert!(sub_only as u32 <= combined,
            "subject-only {} should be <= combined {}", sub_only, combined);
    }

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
    fn test_palette_layers_ready_for_cognitive_shader() {
        let triplets = vec![
            ("A".into(), "causes".into(), "B".into(), 0.9),
        ];
        let layers = triplets_to_palette_layers(&triplets);
        // layers is [[u64; 64]; 8] — exactly what CognitiveShader::new() expects
        assert_eq!(layers.len(), 8);
        assert_eq!(layers[0].len(), 64);
    }

    #[test]
    fn test_run_convergence_delivers_planes_to_callback() {
        // TD-INT-14 closure: triplets in → palette planes out via the
        // callback. The callback IS the convergence highway terminus —
        // in production it wraps `ShaderDriver::update_planes`. Here we
        // capture the planes in a Cell so we can prove they reached the
        // far side and carry the AriGraph knowledge.
        use std::cell::Cell;

        let triplets = vec![
            ("Claude".into(), "causes".into(), "reasoning".into(), 0.9),
            ("NARS".into(), "enables".into(), "inference".into(), 0.8),
            ("Pearl".into(), "supports".into(), "causality".into(), 0.85),
            ("v1".into(), "contradicts".into(), "v2".into(), 0.7),
            ("draft".into(), "refines".into(), "outline".into(), 0.6),
            ("dog".into(), "is type of".into(), "animal".into(), 0.95),
            ("data".into(), "grounds with evidence".into(), "claim".into(), 0.75),
            ("ice".into(), "becomes".into(), "water".into(), 0.99),
        ];

        let captured: Cell<Option<[[u64; 64]; 8]>> = Cell::new(None);
        run_convergence(&triplets, |planes| {
            captured.set(Some(planes));
        });

        let planes = captured.into_inner().expect("callback was invoked");

        // Knowledge must have reached the cascade: at least one bit set
        // somewhere in the 8 × 64 × 64 palette (i.e. the planes are not
        // the zero topology the driver was constructed with).
        let any_bit_set = planes.iter()
            .any(|layer| layer.iter().any(|row| *row != 0));
        assert!(any_bit_set, "convergence produced an all-zero topology — knowledge never reached the cascade");

        // Every relation we fed should have lit up its predicate layer.
        // Layers 0..7 cover CAUSES/ENABLES/SUPPORTS/CONTRADICTS/REFINES/
        // ABSTRACTS/GROUNDS/BECOMES.
        for (idx, layer) in planes.iter().enumerate() {
            assert!(
                layer.iter().any(|row| *row != 0),
                "predicate layer {idx} stayed empty after convergence"
            );
        }
    }

    #[test]
    fn test_run_convergence_zero_in_zero_out() {
        // Empty input must still produce a [[u64; 64]; 8] (the cascade
        // expects that exact shape) and the callback must run exactly
        // once. The planes are all zero — the all-zero topology is a
        // legitimate "no knowledge committed yet" state.
        let triplets: Vec<(String, String, String, f32)> = vec![];
        let mut call_count = 0;
        let mut captured = [[1u64; 64]; 8]; // sentinel non-zero

        run_convergence(&triplets, |planes| {
            call_count += 1;
            captured = planes;
        });

        assert_eq!(call_count, 1, "callback must run exactly once");
        assert!(
            captured.iter().all(|layer| layer.iter().all(|row| *row == 0)),
            "no triplets means zero topology"
        );
    }
}
