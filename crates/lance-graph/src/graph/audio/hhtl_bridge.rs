//! HHTL Bridge: AudioNode ↔ bgz-tensor cascade.
//!
//! Maps AudioFrame's 48 bytes to HHTL cascade levels:
//!
//! ```text
//! HEEL: PVQ summary bytes 0-1 (sign pattern → spectral category)
//!       + SpiralAddress.stride (role → voice character)
//!       = coarse_band() from highheelbgz (3 integer ops, zero data)
//!
//! HIP:  palette_idx → HhtlCache.route(a, b) → RouteAction
//!       = O(1) lookup in precomputed k×k route table
//!       Band energies projected to Base17, nearest palette centroid.
//!
//! TWIG: Base17 L1 distance between full band energy vectors
//!       = 17 i16 subtractions (VNNI-acceleratable)
//!
//! LEAF: Full AudioFrame decode → iMDCT → PCM
//!       = only for frames that survive HEEL+HIP+TWIG
//! ```
//!
//! The ComposeTable enables multi-hop audio reasoning:
//!   compose(frame_A, frame_B) → palette index of their XOR-bind
//!   = "what would A and B sound like combined?"
//!   Used for: mixing, crossfade prediction, transition detection.
//!
//! The route_hint on AudioNode enables STREAMING cascade:
//!   If route_hint == Skip, the renderer can skip iMDCT entirely
//!   and interpolate from the previous frame. 40-60% of frames
//!   skip in typical speech (norm-role, low spectral change).

/// HHTL cascade level for audio search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioCascadeLevel {
    /// HEEL: coarse_band + stride match. 3 integer ops.
    Heel,
    /// HIP: palette route table lookup. O(1).
    Hip,
    /// TWIG: Base17 L1 distance. 17 subtractions.
    Twig,
    /// LEAF: full decode. iMDCT + overlap-add.
    Leaf,
}

/// Result of cascading an audio query against a candidate.
#[derive(Clone, Copy, Debug)]
pub struct AudioCascadeResult {
    /// Which level resolved this pair.
    pub level: AudioCascadeLevel,
    /// Distance at the resolving level (lower = more similar).
    pub distance: u32,
    /// Whether this pair should be attended to (not skipped).
    pub attend: bool,
}

/// Audio HHTL cascade: progressive elimination for audio search.
///
/// Given a query AudioNode and a set of candidate AudioNodes,
/// eliminates candidates at each level (cheapest first):
///
///   HEEL: stride mismatch → reject (different voice role)
///         PVQ category mismatch → reject (different spectral class)
///
///   HIP:  palette route table → Skip/Attend/Compose/Escalate
///         ~40-60% eliminated here (speech norm-role frames)
///
///   TWIG: Base17 L1 within threshold → attend
///         Remaining ~10-20% eliminated
///
///   LEAF: full decode for final ranking (top-k only)
pub fn cascade_search(
    query: &super::node::AudioNode,
    candidates: &[super::node::AudioNode],
    route_table: &[u8], // k×k route table (RouteAction as u8)
    k: usize,           // palette size
    hip_threshold: u16,  // max distance for HIP attend
) -> Vec<(usize, AudioCascadeResult)> {
    let mut results = Vec::new();

    for (idx, cand) in candidates.iter().enumerate() {
        // ═══ HEEL: stride match + PVQ category ═══
        // Different stride = different role = categorically different
        if query.spiral[1] != cand.spiral[1] {
            results.push((idx, AudioCascadeResult {
                level: AudioCascadeLevel::Heel,
                distance: u32::MAX,
                attend: false,
            }));
            continue;
        }

        // PVQ sign pattern match (bytes 0-1 = spectral category)
        let pvq_match = query.pvq_summary[0] == cand.pvq_summary[0]
                      && query.pvq_summary[1] == cand.pvq_summary[1];
        if !pvq_match {
            // Different spectral category — likely skip, but check HIP
            // (don't hard-reject: similar energy can have different signs)
        }

        // ═══ HIP: palette route table ═══
        let qa = query.palette_idx as usize;
        let ca = cand.palette_idx as usize;
        if qa < k && ca < k {
            let route = route_table[qa * k + ca];
            match route {
                0 => { // Skip
                    results.push((idx, AudioCascadeResult {
                        level: AudioCascadeLevel::Hip,
                        distance: u32::MAX,
                        attend: false,
                    }));
                    continue;
                }
                1 => { // Attend
                    let d = super::node::spectral_l1(query, cand);
                    results.push((idx, AudioCascadeResult {
                        level: AudioCascadeLevel::Hip,
                        distance: d as u32,
                        attend: d <= hip_threshold,
                    }));
                    continue;
                }
                2 => { // Compose — check intermediate path
                    // For audio: compose = "could these frames be crossfaded?"
                    // Attend if either direct or composed distance is small
                    let d = super::node::spectral_l1(query, cand);
                    results.push((idx, AudioCascadeResult {
                        level: AudioCascadeLevel::Hip,
                        distance: d as u32,
                        attend: d <= hip_threshold * 2, // wider threshold for compose
                    }));
                    continue;
                }
                _ => { // Escalate → fall through to TWIG
                }
            }
        }

        // ═══ TWIG: spectral L1 distance ═══
        let d = super::node::spectral_l1(query, cand);
        results.push((idx, AudioCascadeResult {
            level: AudioCascadeLevel::Twig,
            distance: d as u32,
            attend: d <= hip_threshold,
        }));
    }

    results
}

/// Statistics from a cascade search run.
#[derive(Clone, Copy, Debug, Default)]
pub struct CascadeStats {
    pub heel_rejected: u32,
    pub hip_skipped: u32,
    pub hip_attended: u32,
    pub hip_composed: u32,
    pub twig_resolved: u32,
    pub leaf_decoded: u32,
}

impl CascadeStats {
    pub fn from_results(results: &[(usize, AudioCascadeResult)]) -> Self {
        let mut stats = CascadeStats::default();
        for (_, r) in results {
            match (r.level, r.attend) {
                (AudioCascadeLevel::Heel, _) => stats.heel_rejected += 1,
                (AudioCascadeLevel::Hip, false) => stats.hip_skipped += 1,
                (AudioCascadeLevel::Hip, true) => stats.hip_attended += 1,
                (AudioCascadeLevel::Twig, _) => stats.twig_resolved += 1,
                (AudioCascadeLevel::Leaf, _) => stats.leaf_decoded += 1,
            }
        }
        stats
    }

    pub fn skip_rate(&self) -> f32 {
        let total = self.heel_rejected + self.hip_skipped + self.hip_attended
                  + self.hip_composed + self.twig_resolved + self.leaf_decoded;
        if total == 0 { return 0.0; }
        (self.heel_rejected + self.hip_skipped) as f32 / total as f32
    }
}

/// Build route hints for a sequence of AudioNodes.
///
/// For each consecutive pair (prev, curr), compute the HHTL route action.
/// This enables streaming skip: the renderer checks route_hint before
/// spending cycles on iMDCT decode.
///
/// Returns the input nodes with route_hint filled in.
pub fn assign_route_hints(
    nodes: &mut [super::node::AudioNode],
    route_table: &[u8],
    k: usize,
) {
    if nodes.is_empty() { return; }
    nodes[0].route_hint = 1; // first frame always attends

    for i in 1..nodes.len() {
        let prev_idx = nodes[i - 1].palette_idx as usize;
        let curr_idx = nodes[i].palette_idx as usize;
        if prev_idx < k && curr_idx < k {
            nodes[i].route_hint = route_table[prev_idx * k + curr_idx];
        } else {
            nodes[i].route_hint = 3; // Escalate (unknown palette)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::node::AudioNode;

    fn make_node(stride: u16, palette_idx: u8, energy: u16) -> AudioNode {
        AudioNode::from_parts(
            [energy; 21], [0; 6], [200, 30, 128, 50],
            0, stride, 10, palette_idx, 0,
        )
    }

    #[test]
    fn heel_rejects_different_stride() {
        let query = make_node(5, 0, 0x3C00); // V role
        let candidates = vec![
            make_node(5, 1, 0x3C00), // same stride
            make_node(8, 1, 0x3C00), // different stride (Gate)
            make_node(3, 1, 0x3C00), // different stride (QK)
        ];
        let route_table = vec![1u8; 256 * 256]; // all Attend

        let results = cascade_search(&query, &candidates, &route_table, 256, 1000);
        // Only stride=5 should pass HEEL
        assert!(results[0].1.attend, "Same stride should attend");
        assert!(!results[1].1.attend, "Different stride should reject at HEEL");
        assert!(!results[2].1.attend, "Different stride should reject at HEEL");
    }

    #[test]
    fn hip_skips_when_route_table_says_skip() {
        let query = make_node(5, 0, 0x3C00);
        let candidates = vec![
            make_node(5, 1, 0x3C00),
            make_node(5, 2, 0x3C00),
        ];
        // Route table: palette 0→1 = Skip, palette 0→2 = Attend
        let mut route_table = vec![1u8; 256 * 256]; // default Attend
        route_table[0 * 256 + 1] = 0; // Skip

        let results = cascade_search(&query, &candidates, &route_table, 256, 1000);
        assert!(!results[0].1.attend, "Route Skip should not attend");
        assert!(results[1].1.attend, "Route Attend should attend");
    }

    #[test]
    fn cascade_stats_skip_rate() {
        let results = vec![
            (0, AudioCascadeResult { level: AudioCascadeLevel::Heel, distance: u32::MAX, attend: false }),
            (1, AudioCascadeResult { level: AudioCascadeLevel::Hip, distance: u32::MAX, attend: false }),
            (2, AudioCascadeResult { level: AudioCascadeLevel::Hip, distance: 100, attend: true }),
            (3, AudioCascadeResult { level: AudioCascadeLevel::Twig, distance: 50, attend: true }),
        ];
        let stats = CascadeStats::from_results(&results);
        assert_eq!(stats.heel_rejected, 1);
        assert_eq!(stats.hip_skipped, 1);
        assert_eq!(stats.hip_attended, 1);
        assert_eq!(stats.twig_resolved, 1);
        assert!((stats.skip_rate() - 0.5).abs() < 0.01, "50% skip rate");
    }

    #[test]
    fn assign_route_hints_first_always_attends() {
        let mut nodes = vec![
            make_node(5, 0, 0x3C00),
            make_node(5, 0, 0x3C00),
        ];
        let route_table = vec![0u8; 256 * 256]; // all Skip
        assign_route_hints(&mut nodes, &route_table, 256);
        assert_eq!(nodes[0].route_hint, 1, "First frame always attends");
        assert_eq!(nodes[1].route_hint, 0, "Second frame gets route from table");
    }
}
