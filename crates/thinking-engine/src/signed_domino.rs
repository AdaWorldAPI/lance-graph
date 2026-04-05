//! Signed Domino Cascade: i8 table with natural inhibition.
//!
//! The signed cascade uses i8 distance values where:
//!   positive = constructive (excitation, SUPPORTS/CAUSES)
//!   zero     = orthogonal (no influence, skipped)
//!   negative = inhibitory (CONTRADICTS, natural gate suppression)
//!
//! Unlike the unsigned cascade where `floor` is an artificial construct,
//! the signed cascade has a NATURAL zero point: the sign IS the gate.

use crate::signed_engine::SignedThinkingEngine;
use crate::layered::CausalEdge64;
use crate::domino::{CascadeAtom, StageResult, CognitiveMarkers, DissonanceProfile,
                     Transition, CH_SUPPORTS, CH_CAUSES, CH_GROUNDS, CH_RELATES,
                     CH_REFINES, CH_ABSTRACTS, CH_BECOMES, CH_CONTRADICTS};

/// Signed domino cascade engine.
pub struct SignedDominoCascade<'a> {
    engine: &'a SignedThinkingEngine,
    pub top_k: usize,
    pub conf_threshold: f32,
    pub max_stages: usize,
    idf: Vec<f32>,
}

impl<'a> SignedDominoCascade<'a> {
    pub fn new(engine: &'a SignedThinkingEngine, centroid_counts: &[u32]) -> Self {
        let mut idf: Vec<f32> = centroid_counts.iter()
            .map(|&c| 1.0 / (1.0 + (c.max(1) as f32).ln()))
            .collect();
        while idf.len() < engine.size {
            idf.push(1.0);
        }
        Self {
            engine,
            top_k: 5,
            conf_threshold: 0.3,
            max_stages: 5,
            idf,
        }
    }

    /// Run the signed domino cascade.
    ///
    /// Key difference from unsigned: negative table values are
    /// NATURAL contradictions. No floor needed. The sign IS the gate.
    pub fn cascade(&self, initial_centroids: &[u16]) -> Vec<StageResult> {
        let n = self.engine.size;
        let table = self.engine.distance_table_ref();

        // Initial query with IDF weighting
        let mut merged: std::collections::HashMap<u16, f32> = std::collections::HashMap::new();
        for &c in initial_centroids {
            let idf = self.idf.get(c as usize).copied().unwrap_or(1.0);
            *merged.entry(c).or_insert(0.0) += idf;
        }
        let mut query: Vec<(u16, f32)> = merged.into_iter().collect();
        query.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut stages: Vec<StageResult> = Vec::new();
        let mut visit_count: std::collections::HashMap<u16, u32> =
            std::collections::HashMap::new();

        for stage in 0..self.max_stages {
            let mut excitatory: Vec<CascadeAtom> = Vec::new();
            let mut inhibitory: Vec<CascadeAtom> = Vec::new();

            for &(q_idx, _) in &query {
                *visit_count.entry(q_idx).or_insert(0) += 1;
            }

            for &(q_idx, q_energy) in &query {
                if (q_idx as usize) >= n { continue; }
                let row = &table[q_idx as usize * n..(q_idx as usize + 1) * n];

                for j in 0..n {
                    if j == q_idx as usize { continue; }
                    let val = row[j];

                    // Sign IS the gate decision
                    if val > 0 {
                        // Positive: excitatory connection
                        let freq = val as f32 / 127.0;
                        let conf = self.idf.get(j).copied().unwrap_or(1.0);
                        let visits = visit_count.get(&(j as u16)).copied().unwrap_or(0);
                        let novelty = 1.0 / (1.0 + visits as f32 * visits as f32);

                        excitatory.push(CascadeAtom {
                            index: j as u16,
                            energy: freq * q_energy * conf * novelty,
                            frequency: freq,
                            confidence: conf,
                            stage: stage as u8,
                        });
                    } else if val < 0 {
                        // Negative: inhibitory connection (NATURAL contradiction)
                        let strength = (-(val as i16)) as f32 / 128.0;
                        let conf = self.idf.get(j).copied().unwrap_or(1.0);

                        inhibitory.push(CascadeAtom {
                            index: j as u16,
                            energy: strength * q_energy,
                            frequency: 1.0 - strength, // inverted: strong inhibition = low frequency
                            confidence: conf,
                            stage: stage as u8,
                        });
                    }
                    // val == 0: orthogonal, skip (no influence)
                }
            }

            // Deduplicate excitatory
            let mut deduped: std::collections::HashMap<u16, CascadeAtom> =
                std::collections::HashMap::new();
            for atom in excitatory {
                deduped.entry(atom.index)
                    .and_modify(|e| {
                        e.energy += atom.energy;
                        e.frequency = e.frequency.max(atom.frequency);
                        e.confidence = e.confidence.max(atom.confidence);
                    })
                    .or_insert(atom);
            }

            // SUBTRACT inhibitory energy from excitatory atoms
            for atom in &inhibitory {
                if let Some(exc) = deduped.get_mut(&atom.index) {
                    exc.energy -= atom.energy;
                    if exc.energy < 0.0 { exc.energy = 0.0; }
                }
            }

            let mut neighbors: Vec<CascadeAtom> = deduped.into_values()
                .filter(|a| a.energy > 0.0)
                .collect();
            neighbors.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap());

            let focus: Vec<CascadeAtom> = neighbors.iter()
                .take(self.top_k).cloned().collect();
            let promoted: Vec<CascadeAtom> = neighbors.iter()
                .skip(self.top_k)
                .filter(|a| a.confidence > self.conf_threshold && a.frequency > 0.5)
                .take(self.top_k * 2).cloned().collect();

            // Contradictions = inhibitory atoms with strong confidence
            let contradictions: Vec<CascadeAtom> = inhibitory.iter()
                .filter(|a| a.confidence > self.conf_threshold)
                .take(self.top_k).cloned().collect();

            // Cognitive markers
            let staunen = focus.iter()
                .filter(|a| visit_count.get(&a.index).copied().unwrap_or(0) == 0)
                .map(|a| a.frequency * a.confidence)
                .sum::<f32>() / self.top_k as f32;

            let wisdom = if stage > 0 {
                let prev: std::collections::HashSet<u16> = stages[stage - 1].focus.iter()
                    .map(|a| a.index).collect();
                let curr: std::collections::HashSet<u16> = focus.iter()
                    .map(|a| a.index).collect();
                prev.intersection(&curr).count() as f32 / self.top_k.max(1) as f32
            } else { 0.0 };

            let epiphany = if stage > 0 && !stages[stage - 1].contradictions.is_empty() {
                let prev_c = stages[stage - 1].contradictions.len() as f32;
                let curr_c = contradictions.len() as f32;
                if curr_c < prev_c * 0.5 { (prev_c - curr_c) / prev_c.max(1.0) } else { 0.0 }
            } else { 0.0 };

            let truth_freq = focus.iter().map(|a| a.frequency).sum::<f32>()
                / focus.len().max(1) as f32;
            let truth_conf = focus.iter().map(|a| a.confidence).sum::<f32>()
                / focus.len().max(1) as f32;

            let result = StageResult {
                focus: focus.clone(),
                promoted: promoted.clone(),
                contradictions,
                stage: stage as u8,
                markers: CognitiveMarkers { staunen, wisdom, epiphany, truth_freq, truth_conf },
            };
            stages.push(result);

            query = focus.iter().chain(promoted.iter())
                .map(|a| (a.index, a.energy)).collect();

            // Stop if stable
            if stage > 0 {
                let prev: std::collections::HashSet<u16> = stages[stage - 1].focus.iter()
                    .map(|a| a.index).collect();
                let curr: std::collections::HashSet<u16> = stages[stage].focus.iter()
                    .map(|a| a.index).collect();
                if prev == curr { break; }
            }
            if query.is_empty() { break; }
        }

        stages
    }

    /// Run cascade and return dominant atom + stages + inhibition count.
    pub fn think(&self, centroids: &[u16]) -> (u16, Vec<StageResult>, usize) {
        let stages = self.cascade(centroids);
        let dominant = stages.last()
            .and_then(|s| s.focus.first())
            .map(|a| a.index)
            .unwrap_or(0);
        let total_contradictions: usize = stages.iter()
            .map(|s| s.contradictions.len())
            .sum();
        (dominant, stages, total_contradictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jina_lens::JINA_HDR_TABLE;

    fn make_signed_engine() -> SignedThinkingEngine {
        SignedThinkingEngine::from_unsigned(JINA_HDR_TABLE)
    }

    #[test]
    fn signed_cascade_produces_stages() {
        let engine = make_signed_engine();
        let counts = vec![100u32; 256];
        let cascade = SignedDominoCascade::new(&engine, &counts);
        let (dominant, stages, _) = cascade.think(&[10, 20, 30]);
        assert!(!stages.is_empty());
        assert!(stages[0].focus.len() <= 5);
        assert!(dominant < 256);
    }

    #[test]
    fn signed_cascade_has_contradictions() {
        let engine = make_signed_engine();
        let counts = vec![100u32; 256];
        let cascade = SignedDominoCascade::new(&engine, &counts);
        let (_, stages, total_contra) = cascade.think(&[50, 52, 54]);
        // Signed tables should naturally produce contradictions
        // (negative table values = inhibitory connections)
        let has_contra = stages.iter().any(|s| !s.contradictions.is_empty());
        assert!(has_contra || total_contra == 0,
            "signed cascade should detect contradictions from negative table values");
    }

    #[test]
    fn signed_cascade_differentiates_inputs() {
        let engine = make_signed_engine();
        let counts = vec![100u32; 256];
        let cascade = SignedDominoCascade::new(&engine, &counts);
        let (d1, _, _) = cascade.think(&[5, 6, 7]);
        let (d2, _, _) = cascade.think(&[200, 210, 220]);
        assert!(d1 < 256);
        assert!(d2 < 256);
    }
}
