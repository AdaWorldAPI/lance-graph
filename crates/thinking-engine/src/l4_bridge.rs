//! L3→L4 Bridge: commit peaks → XOR bind → accumulate.
//!
//! Connects the immutable WAVE cascade (L1→L2→L3) to the mutable
//! PARTICLE experience (L4). Every L3 commit becomes a learning
//! signal for L4 personality.
//!
//! **LIMITATION**: Uses distance table rows as centroid proxies.
//! A table row is a list of DISTANCES (how far centroid i is from all others),
//! NOT the centroid vector itself. This is a lossy approximation.
//! For accurate L4 learning, use the real centroid vectors (f32, from CLAM).
//! Real centroids are only available for Qwopus (5120D, 80 MB in Release).
//! For the 256×256 HDR lenses, centroid vectors are NOT stored.
//!
//! ```text
//! L3 commit (BusDto)
//!   → top-k peaks: [(idx, energy); 8]
//!   → for each consecutive pair:
//!       row_a = table[idx_a][..] centered
//!       row_b = table[idx_b][..] centered
//!       bin_a = binarize(row_a)  → 16,384 bits
//!       bin_b = binarize(row_b)  → 16,384 bits
//!       bound = xor_bind(bin_a, bin_b)
//!       reward = sqrt(energy_a × energy_b) × scale
//!       l4.learn(bound, reward)
//! ```

use crate::dto::BusDto;
use crate::l4::L4Experience;

/// Bridge an L3 commit into L4 experience.
///
/// For each consecutive pair of top peaks in the commit:
///   1. Extract distance table rows as centroid proxies (f32, centered at 0)
///   2. Binarize each row to 16,384 bits via sign-bit threshold
///   3. XOR-bind the pair = "this relationship"
///   4. Learn with reward proportional to peak energy
///
/// Returns the number of pairs learned.
pub fn commit_to_l4(
    bus: &BusDto,
    distance_table: &[u8],
    table_size: usize,
    l4: &mut L4Experience,
    reward_scale: i8,
) -> usize {
    let peaks: Vec<(u16, f32)> = bus.top_k.iter()
        .filter(|&&(_, e)| e > 0.01)
        .cloned()
        .collect();

    if peaks.len() < 2 { return 0; }

    let mut learned = 0usize;

    for pair in peaks.windows(2) {
        let (idx_a, energy_a) = pair[0];
        let (idx_b, energy_b) = pair[1];

        if (idx_a as usize) >= table_size || (idx_b as usize) >= table_size {
            continue;
        }

        // Distance table rows as centroid proxies, centered around zero.
        let row_a: Vec<f32> = distance_table
            [idx_a as usize * table_size..(idx_a as usize + 1) * table_size]
            .iter().map(|&v| v as f32 - 128.0).collect();
        let row_b: Vec<f32> = distance_table
            [idx_b as usize * table_size..(idx_b as usize + 1) * table_size]
            .iter().map(|&v| v as f32 - 128.0).collect();

        let bin_a = L4Experience::binarize(&row_a);
        let bin_b = L4Experience::binarize(&row_b);
        let bound = L4Experience::xor_bind(&bin_a, &bin_b);

        // Reward proportional to geometric mean of peak energies.
        let pair_energy = (energy_a * energy_b).sqrt();
        let reward = (pair_energy * reward_scale as f32)
            .round().clamp(-128.0, 127.0) as i8;

        if reward != 0 {
            l4.learn(&bound, reward);
            learned += 1;
        }
    }

    learned
}

/// Check if L4 recognizes the pattern from a commit.
///
/// Returns a score: positive = familiar-good, negative = familiar-bad,
/// near-zero = novel pattern.
pub fn recognize_thought(
    bus: &BusDto,
    distance_table: &[u8],
    table_size: usize,
    l4: &L4Experience,
) -> i32 {
    let peaks: Vec<(u16, f32)> = bus.top_k.iter()
        .filter(|&&(_, e)| e > 0.01)
        .cloned()
        .collect();

    if peaks.len() < 2 { return 0; }

    let mut total_score: i32 = 0;
    let mut pairs = 0;

    for pair in peaks.windows(2) {
        let (idx_a, _) = pair[0];
        let (idx_b, _) = pair[1];

        if (idx_a as usize) >= table_size || (idx_b as usize) >= table_size {
            continue;
        }

        let row_a: Vec<f32> = distance_table
            [idx_a as usize * table_size..(idx_a as usize + 1) * table_size]
            .iter().map(|&v| v as f32 - 128.0).collect();
        let row_b: Vec<f32> = distance_table
            [idx_b as usize * table_size..(idx_b as usize + 1) * table_size]
            .iter().map(|&v| v as f32 - 128.0).collect();

        let bin_a = L4Experience::binarize(&row_a);
        let bin_b = L4Experience::binarize(&row_b);
        let bound = L4Experience::xor_bind(&bin_a, &bin_b);

        total_score += l4.recognize(&bound);
        pairs += 1;
    }

    if pairs > 0 { total_score / pairs } else { 0 }
}

/// Generate sensor bias weights from L4 experience.
///
/// For each consecutive pair of codebook indices (wrapping last→first),
/// checks L4 recognition and maps to a weight multiplier:
///   >1.0 for familiar-good patterns (boost)
///   <1.0 for familiar-bad patterns (suppress)
///   =1.0 for novel patterns (neutral)
pub fn bias_from_l4(
    codebook_indices: &[u16],
    distance_table: &[u8],
    table_size: usize,
    l4: &L4Experience,
) -> Vec<f32> {
    let n = codebook_indices.len();
    let mut weights = vec![1.0f32; n];

    if n < 2 { return weights; }

    for i in 0..n {
        let j = if i + 1 < n { i + 1 } else { 0 };

        let idx_a = codebook_indices[i] as usize;
        let idx_b = codebook_indices[j] as usize;

        if idx_a >= table_size || idx_b >= table_size { continue; }

        let row_a: Vec<f32> = distance_table
            [idx_a * table_size..(idx_a + 1) * table_size]
            .iter().map(|&v| v as f32 - 128.0).collect();
        let row_b: Vec<f32> = distance_table
            [idx_b * table_size..(idx_b + 1) * table_size]
            .iter().map(|&v| v as f32 - 128.0).collect();

        let bin_a = L4Experience::binarize(&row_a);
        let bin_b = L4Experience::binarize(&row_b);
        let bound = L4Experience::xor_bind(&bin_a, &bin_b);

        let score = l4.recognize(&bound);
        let norm = (score as f32) / (crate::l4::ACCUM_LEN as f32 * 32.0);
        weights[i] = 1.0 + 0.5 * norm.clamp(-1.0, 1.0);
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jina_lens::JINA_HDR_TABLE;

    fn make_test_bus() -> BusDto {
        BusDto {
            codebook_index: 50,
            energy: 0.3,
            top_k: [
                (50, 0.30), (52, 0.25), (54, 0.20), (100, 0.10),
                (130, 0.08), (200, 0.05), (10, 0.01), (1, 0.01),
            ],
            cycle_count: 7,
            converged: true,
        }
    }

    #[test]
    fn commit_to_l4_learns() {
        let mut l4 = L4Experience::new();
        let bus = make_test_bus();

        let (p0, n0, _) = l4.stats();
        assert_eq!(p0, 0);
        assert_eq!(n0, 0);

        let learned = commit_to_l4(&bus, JINA_HDR_TABLE, 256, &mut l4, 50);
        assert!(learned > 0, "should have learned at least one pair");

        let (p1, n1, _) = l4.stats();
        assert!(p1 + n1 > 0, "personality should change after learning");
    }

    #[test]
    fn recognize_after_commit() {
        let mut l4 = L4Experience::new();
        let bus = make_test_bus();

        let score_before = recognize_thought(&bus, JINA_HDR_TABLE, 256, &l4);

        for _ in 0..10 {
            commit_to_l4(&bus, JINA_HDR_TABLE, 256, &mut l4, 50);
        }

        let score_after = recognize_thought(&bus, JINA_HDR_TABLE, 256, &l4);
        assert!(score_after > score_before,
            "recognition should increase: before={} after={}", score_before, score_after);
    }

    #[test]
    fn bias_weights_change_with_experience() {
        let mut l4 = L4Experience::new();
        let indices = vec![50u16, 52, 54, 100];

        let weights_before = bias_from_l4(&indices, JINA_HDR_TABLE, 256, &l4);
        for &w in &weights_before {
            assert!((w - 1.0).abs() < 0.01, "initial weight should be ~1.0, got {}", w);
        }

        let bus = make_test_bus();
        for _ in 0..20 {
            commit_to_l4(&bus, JINA_HDR_TABLE, 256, &mut l4, 50);
        }

        let weights_after = bias_from_l4(&indices, JINA_HDR_TABLE, 256, &l4);
        let any_changed = weights_after.iter().any(|&w| (w - 1.0).abs() > 0.01);
        assert!(any_changed, "weights should change after learning: {:?}", weights_after);
    }

    #[test]
    fn empty_commit_does_nothing() {
        let mut l4 = L4Experience::new();
        let table = vec![128u8; 16]; // 4x4 table

        let bus = BusDto {
            codebook_index: 0,
            energy: 1.0,
            top_k: [(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0),
                     (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)],
            cycle_count: 1,
            converged: true,
        };

        let learned = commit_to_l4(&bus, &table, 4, &mut l4, 50);
        assert_eq!(learned, 0);
    }
}
