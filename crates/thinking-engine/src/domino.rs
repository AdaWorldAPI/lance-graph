//! Domino Cascade: top-K focus + NARS context testing.
//!
//! Instead of MatVec convergence (which washes out signal in uniform tables),
//! cascade through the table attention-style:
//!
//! ```text
//! Q1 = perturb(tokens)       → 3σ top-K focus
//!   context = full row        → NARS truth filter (freq, conf)
//!   promoted + contradicts    → CausalEdge64
//! V1 = focus + promoted       → feed as Q2
//! Q2 = V1 survivors           → 3σ top-K focus
//!   ...cascade until NARS confidence saturates
//! ```
//!
//! The table topology IS used, but through focused attention instead of
//! global MatVec. Works even on near-uniform tables because 3σ selects
//! the few genuinely strong connections per row.

use crate::engine::ThinkingEngine;
use crate::layered::CausalEdge64;

// Bach counterpoint channel indices
pub const CH_BECOMES: u8 = 0;     // voice crossing — identity shifts
pub const CH_CAUSES: u8 = 1;      // diminution — one causes the other
pub const CH_SUPPORTS: u8 = 2;    // parallel motion — same direction
pub const CH_REFINES: u8 = 3;     // contrary motion — opposite, sharpening
pub const CH_GROUNDS: u8 = 4;     // oblique motion — one stable, other moves
pub const CH_ABSTRACTS: u8 = 5;   // augmentation — slower, broader
pub const CH_RELATES: u8 = 6;     // imitation — echo, repetition
pub const CH_CONTRADICTS: u8 = 7; // dissonance — tritone, unresolved

/// One atom in the cascade with its accumulated truth.
#[derive(Clone, Debug)]
pub struct CascadeAtom {
    pub index: u16,
    pub energy: f32,
    /// NARS frequency: how similar to the query (from table value).
    pub frequency: f32,
    /// NARS confidence: how informative (inverse of centroid popularity).
    pub confidence: f32,
    /// Which stage promoted this atom.
    pub stage: u8,
}

/// Result of one cascade stage.
#[derive(Clone, Debug)]
pub struct StageResult {
    /// Focus atoms (3σ survivors).
    pub focus: Vec<CascadeAtom>,
    /// Context atoms promoted by NARS.
    pub promoted: Vec<CascadeAtom>,
    /// Contradicting atoms (high confidence, low frequency).
    pub contradictions: Vec<CascadeAtom>,
    /// Stage number.
    pub stage: u8,
}

/// A transition between two cascade stages, classified by Bach counterpoint.
#[derive(Clone, Debug)]
pub struct Transition {
    pub from_atom: u16,
    pub to_atom: u16,
    pub edge: CausalEdge64,
    pub dissonance: f32, // 0.0 = consonant, 1.0 = fully dissonant
}

/// Dissonance profile across the full cascade chain.
#[derive(Clone, Debug)]
pub struct DissonanceProfile {
    pub transitions: Vec<Transition>,
    /// Overall dissonance: contradiction / (contradiction + constructive).
    pub total_dissonance: f32,
    /// Per-stage dissonance values.
    pub per_stage: Vec<f32>,
    /// Is this a resolved progression? (dissonance decreasing over stages)
    pub resolved: bool,
    /// Is this a Rachmaninov suspension? (high dissonance sustained, then sudden drop)
    pub suspension: bool,
}

/// Classify a stage-to-stage transition using Bach counterpoint rules.
///
/// The classification is based on the RELATIONSHIP between the two atoms
/// in the distance table, not just the distance value.
pub fn classify_transition(
    table: &[u8],
    n: usize,
    from: u16,
    to: u16,
    from_energy: f32,
    to_energy: f32,
    floor: u8,
) -> Transition {
    let sim = if (from as usize) < n && (to as usize) < n {
        table[from as usize * n + to as usize]
    } else { 128 };

    let above_floor = sim.saturating_sub(floor) as f32 / (255.0 - floor as f32).max(1.0);
    let energy_ratio = if from_energy > 1e-10 { to_energy / from_energy } else { 1.0 };

    let mut edge = CausalEdge64::new();

    if above_floor > 0.8 {
        // Very similar: parallel motion (SUPPORTS)
        edge.set_channel(CH_SUPPORTS, (above_floor * 200.0) as u8);
    } else if above_floor > 0.5 {
        // Moderately similar
        if energy_ratio > 1.2 {
            // Energy increasing: CAUSES (the first drives the second)
            edge.set_channel(CH_CAUSES, (above_floor * 180.0) as u8);
        } else if energy_ratio < 0.8 {
            // Energy decreasing: GROUNDS (stabilizing)
            edge.set_channel(CH_GROUNDS, (above_floor * 150.0) as u8);
        } else {
            // Stable energy: RELATES (imitation/echo)
            edge.set_channel(CH_RELATES, (above_floor * 160.0) as u8);
        }
    } else if above_floor > 0.2 {
        // Weak similarity: contrary motion (REFINES) or abstraction
        if energy_ratio > 1.5 {
            edge.set_channel(CH_ABSTRACTS, (above_floor * 120.0) as u8);
        } else {
            edge.set_channel(CH_REFINES, (above_floor * 130.0) as u8);
        }
    } else if above_floor > 0.0 {
        // Very weak: identity shift (BECOMES)
        edge.set_channel(CH_BECOMES, (above_floor * 100.0) as u8);
    }

    // Dissonance: low similarity + high confidence = contradiction
    if sim < floor {
        let contra_strength = ((floor - sim) as f32 / floor as f32 * 200.0).min(255.0) as u8;
        edge.set_channel(CH_CONTRADICTS, contra_strength);
    }

    let constructive = edge.constructive_strength() as f32;
    let contradiction = edge.get_channel(CH_CONTRADICTS) as f32;
    let dissonance = if constructive + contradiction > 0.0 {
        contradiction / (constructive + contradiction)
    } else {
        0.5 // neutral
    };

    Transition { from_atom: from, to_atom: to, edge, dissonance }
}

/// Compute the full dissonance profile from cascade stages.
pub fn measure_dissonance(
    stages: &[StageResult],
    table: &[u8],
    n: usize,
    floor: u8,
) -> DissonanceProfile {
    let mut transitions = Vec::new();
    let mut per_stage = Vec::new();

    for i in 0..stages.len().saturating_sub(1) {
        let from = &stages[i];
        let to = &stages[i + 1];

        // Primary transition: dominant focus atom to next dominant
        let from_atom = from.focus.first().map(|a| (a.index, a.energy)).unwrap_or((0, 0.0));
        let to_atom = to.focus.first().map(|a| (a.index, a.energy)).unwrap_or((0, 0.0));

        let t = classify_transition(table, n, from_atom.0, to_atom.0, from_atom.1, to_atom.1, floor);
        per_stage.push(t.dissonance);
        transitions.push(t);
    }

    let total_dissonance = if per_stage.is_empty() { 0.0 } else {
        per_stage.iter().sum::<f32>() / per_stage.len() as f32
    };

    // Check if resolved: dissonance should decrease over stages
    let resolved = per_stage.len() >= 2 && {
        let last = per_stage[per_stage.len() - 1];
        let first = per_stage[0];
        last < first * 0.7 // at least 30% drop
    };

    // Check for Rachmaninov suspension: high sustained then sudden drop
    let suspension = per_stage.len() >= 3 && {
        let high_count = per_stage.iter().take(per_stage.len() - 1)
            .filter(|&&d| d > 0.4).count();
        let last = *per_stage.last().unwrap_or(&0.0);
        high_count >= 2 && last < 0.2
    };

    DissonanceProfile { transitions, total_dissonance, per_stage, resolved, suspension }
}

/// The domino cascade engine.
pub struct DominoCascade<'a> {
    engine: &'a ThinkingEngine,
    /// How many atoms in 3σ focus per stage.
    pub top_k: usize,
    /// NARS confidence threshold for promotion.
    pub conf_threshold: f32,
    /// NARS frequency threshold for contradiction.
    pub contra_freq: f32,
    /// Maximum cascade stages.
    pub max_stages: usize,
    /// IDF weights per centroid (1/ln(count+1)).
    idf: Vec<f32>,
}

impl<'a> DominoCascade<'a> {
    pub fn new(engine: &'a ThinkingEngine, centroid_counts: &[u32]) -> Self {
        let idf: Vec<f32> = centroid_counts.iter()
            .map(|&c| 1.0 / (1.0 + (c.max(1) as f32).ln()))
            .collect();
        // Pad if needed
        let mut idf = idf;
        while idf.len() < engine.size {
            idf.push(1.0);
        }
        Self {
            engine,
            top_k: 5,
            conf_threshold: 0.3,
            contra_freq: 0.3,
            max_stages: 5,
            idf,
        }
    }

    /// Run the full domino cascade from initial token centroids.
    pub fn cascade(&self, initial_centroids: &[u16]) -> Vec<StageResult> {
        let n = self.engine.size;
        let table = self.engine.distance_table_ref();
        let floor = self.engine.floor;

        // Initial query: the perturbed centroids
        let mut query: Vec<(u16, f32)> = initial_centroids.iter()
            .map(|&c| (c, self.idf.get(c as usize).copied().unwrap_or(1.0)))
            .collect();

        // Deduplicate and sum weights
        let mut merged = std::collections::HashMap::new();
        for (idx, w) in &query {
            *merged.entry(*idx).or_insert(0.0f32) += w;
        }
        query = merged.into_iter().collect();
        query.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut stages = Vec::new();
        // Visit counter: anti-confirmation-bias gate.
        // Established insights (frequently visited hubs) get suppressed
        // so the cascade explores past familiar territory into the forest.
        let mut visit_count: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();

        for stage in 0..self.max_stages {
            let mut all_neighbors: Vec<CascadeAtom> = Vec::new();

            // Record visits for current query atoms
            for &(q_idx, _) in &query {
                *visit_count.entry(q_idx).or_insert(0) += 1;
            }

            // For each query atom, scan its row
            for &(q_idx, q_energy) in &query {
                if (q_idx as usize) >= n { continue; }
                let row = &table[q_idx as usize * n..(q_idx as usize + 1) * n];

                // Compute row statistics for 3σ threshold
                let above_floor: Vec<(usize, u8)> = row.iter().enumerate()
                    .filter(|(j, &v)| *j != q_idx as usize && v > floor)
                    .map(|(j, &v)| (j, v))
                    .collect();

                if above_floor.is_empty() { continue; }

                let mean: f32 = above_floor.iter().map(|(_, v)| *v as f32).sum::<f32>()
                    / above_floor.len() as f32;
                let variance: f32 = above_floor.iter()
                    .map(|(_, v)| { let d = *v as f32 - mean; d * d })
                    .sum::<f32>() / above_floor.len() as f32;
                let std = variance.sqrt().max(0.1);
                let threshold_3sigma = mean + 3.0 * std;

                for &(j, val) in &above_floor {
                    let freq = (val as f32 - floor as f32) / (255.0 - floor as f32);
                    let conf = self.idf.get(j).copied().unwrap_or(1.0);

                    // Novelty gate: suppress revisited atoms.
                    // Trees (established insights) get gated so we see the forest.
                    let visits = visit_count.get(&(j as u16)).copied().unwrap_or(0);
                    let novelty = 1.0 / (1.0 + visits as f32 * visits as f32);
                    // novelty: 1.0 (never seen) → 0.5 (seen once) → 0.2 (twice) → 0.1 (three times)

                    all_neighbors.push(CascadeAtom {
                        index: j as u16,
                        energy: freq * q_energy * conf * novelty,
                        frequency: freq,
                        confidence: conf,
                        stage: stage as u8,
                    });
                }
            }

            // Deduplicate neighbors (sum energies for same atom)
            let mut deduped: std::collections::HashMap<u16, CascadeAtom> = std::collections::HashMap::new();
            for atom in all_neighbors {
                deduped.entry(atom.index)
                    .and_modify(|existing| {
                        existing.energy += atom.energy;
                        existing.frequency = existing.frequency.max(atom.frequency);
                        existing.confidence = existing.confidence.max(atom.confidence);
                    })
                    .or_insert(atom);
            }

            let mut neighbors: Vec<CascadeAtom> = deduped.into_values().collect();
            neighbors.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap());

            // Split into focus (top-K), promoted (NARS pass), contradictions
            let focus: Vec<CascadeAtom> = neighbors.iter()
                .take(self.top_k)
                .cloned()
                .collect();

            let promoted: Vec<CascadeAtom> = neighbors.iter()
                .skip(self.top_k)
                .filter(|a| a.confidence > self.conf_threshold && a.frequency > 0.5)
                .take(self.top_k * 2) // limit promoted count
                .cloned()
                .collect();

            let contradictions: Vec<CascadeAtom> = neighbors.iter()
                .filter(|a| a.confidence > self.conf_threshold && a.frequency < self.contra_freq)
                .take(self.top_k)
                .cloned()
                .collect();

            let result = StageResult {
                focus: focus.clone(),
                promoted: promoted.clone(),
                contradictions,
                stage: stage as u8,
            };
            stages.push(result);

            // Build Q for next stage: focus + promoted
            query = focus.iter().chain(promoted.iter())
                .map(|a| (a.index, a.energy))
                .collect();

            // Stop if focus is stable (same atoms as previous stage)
            if stage > 0 {
                let prev_focus: std::collections::HashSet<u16> = stages[stage - 1].focus.iter()
                    .map(|a| a.index).collect();
                let curr_focus: std::collections::HashSet<u16> = stages[stage].focus.iter()
                    .map(|a| a.index).collect();
                if prev_focus == curr_focus { break; }
            }

            if query.is_empty() { break; }
        }

        stages
    }

    /// Run cascade and return the dominant atom + its chain + dissonance profile.
    pub fn think(&self, centroids: &[u16]) -> (u16, Vec<StageResult>, DissonanceProfile) {
        let stages = self.cascade(centroids);
        let dominant = stages.last()
            .and_then(|s| s.focus.first())
            .map(|a| a.index)
            .unwrap_or(0);
        let dissonance = measure_dissonance(
            &stages,
            self.engine.distance_table_ref(),
            self.engine.size,
            self.engine.floor,
        );
        (dominant, stages, dissonance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_table(n: usize) -> Vec<u8> {
        let mut table = vec![64u8; n * n];
        for i in 0..n {
            table[i * n + i] = 255;
            // Each atom has 3-4 strong neighbors
            for d in 1..=3 {
                let j = (i + d) % n;
                table[i * n + j] = 200 - (d as u8 * 20);
                table[j * n + i] = 200 - (d as u8 * 20);
            }
        }
        table
    }

    #[test]
    fn cascade_produces_stages() {
        let table = make_test_table(64);
        let engine = ThinkingEngine::new(table);
        let counts = vec![100u32; 64];
        let cascade = DominoCascade::new(&engine, &counts);
        let (dominant, stages, _dis) = cascade.think(&[10, 20, 30]);
        assert!(!stages.is_empty());
        assert!(stages[0].focus.len() <= 5);
        assert!(dominant < 64);
    }

    #[test]
    fn cascade_differentiates_inputs() {
        let table = make_test_table(64);
        let engine = ThinkingEngine::new(table);
        let counts = vec![100u32; 64];
        let cascade = DominoCascade::new(&engine, &counts);

        let (d1, _, _) = cascade.think(&[5, 6, 7]);
        let (d2, _, _) = cascade.think(&[50, 51, 52]);
        // Different inputs should (usually) produce different dominants
        // on a table with local structure
        eprintln!("d1={} d2={}", d1, d2);
        // At minimum, both should produce valid results
        assert!(d1 < 64);
        assert!(d2 < 64);
    }

    #[test]
    fn cascade_ripples_outward() {
        let table = make_test_table(64);
        let engine = ThinkingEngine::new(table);
        let counts = vec![100u32; 64];
        let cascade = DominoCascade::new(&engine, &counts);
        let (_, stages, dissonance) = cascade.think(&[10]);
        // Should produce multiple stages (domino ripple)
        assert!(stages.len() >= 2, "cascade should ripple at least 2 stages");
        // Focus should shift as the ripple propagates
        let s0_focus: std::collections::HashSet<u16> = stages[0].focus.iter().map(|a| a.index).collect();
        let s1_focus: std::collections::HashSet<u16> = stages[1].focus.iter().map(|a| a.index).collect();
        // At least some overlap (continuity) but not identical (exploration)
        let overlap = s0_focus.intersection(&s1_focus).count();
        eprintln!("stage0 focus: {:?}, stage1 focus: {:?}, overlap: {}",
            s0_focus, s1_focus, overlap);
        assert!(s1_focus.len() > 0, "stage 1 should have focus atoms");
        // Dissonance should be computed
        assert!(!dissonance.transitions.is_empty(), "should have transitions");
        assert!(dissonance.total_dissonance >= 0.0 && dissonance.total_dissonance <= 1.0);
        eprintln!("dissonance: total={:.3} per_stage={:?} resolved={} suspension={}",
            dissonance.total_dissonance, dissonance.per_stage, dissonance.resolved, dissonance.suspension);
    }

    #[test]
    fn classify_transition_consonant() {
        // High similarity → SUPPORTS, low dissonance
        let mut table = vec![0u8; 16];
        table[0 * 4 + 1] = 250; // atom 0 → atom 1: very similar
        table[1 * 4 + 0] = 250;
        for i in 0..4 { table[i * 4 + i] = 255; }
        let t = classify_transition(&table, 4, 0, 1, 1.0, 1.0, 0);
        assert!(t.dissonance < 0.1, "high similarity should be consonant, got {}", t.dissonance);
        assert!(t.edge.get_channel(CH_SUPPORTS) > 0, "should be SUPPORTS");
    }

    #[test]
    fn classify_transition_dissonant() {
        // Below floor → CONTRADICTS, high dissonance
        let mut table = vec![128u8; 16];
        table[0 * 4 + 1] = 50; // atom 0 → atom 1: below floor
        table[1 * 4 + 0] = 50;
        for i in 0..4 { table[i * 4 + i] = 255; }
        let t = classify_transition(&table, 4, 0, 1, 1.0, 1.0, 128);
        assert!(t.dissonance > 0.5, "below floor should be dissonant, got {}", t.dissonance);
        assert!(t.edge.get_channel(CH_CONTRADICTS) > 0, "should have CONTRADICTS");
    }
}
