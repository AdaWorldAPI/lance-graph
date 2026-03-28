// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Episodic memory for AriGraph agents.
//!
//! Stores observation episodes with fingerprint-based similarity retrieval,
//! enabling agents to recall relevant past experiences.

use crate::graph::fingerprint::{hamming_distance, label_fp, Fingerprint};
use crate::graph::spo::truth::TruthValue;

/// A single episode: an observation with its extracted triplets and metadata.
#[derive(Debug, Clone)]
pub struct Episode {
    /// The raw observation text.
    pub observation: String,
    /// Triplet strings extracted from this observation ("subject - relation - object").
    pub triplets: Vec<String>,
    /// Fingerprint of the observation for similarity search.
    pub fingerprint: Fingerprint,
    /// Logical step (turn number) when this episode was recorded.
    pub step: u64,
    /// Truth value indicating confidence/relevance of this episode.
    pub truth: TruthValue,
}

/// Capacity-bounded episodic memory with fingerprint-based retrieval.
///
/// Stores episodes up to a fixed capacity, evicting the oldest when full.
/// Retrieval is based on Hamming distance between observation fingerprints.
#[derive(Debug, Clone)]
pub struct EpisodicMemory {
    /// Stored episodes, ordered by insertion time.
    episodes: Vec<Episode>,
    /// Maximum number of episodes to retain.
    capacity: usize,
}

impl EpisodicMemory {
    /// Create a new episodic memory with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            episodes: Vec::with_capacity(capacity.min(1024)),
            capacity,
        }
    }

    /// Add an observation episode.
    ///
    /// Computes a fingerprint from the observation text. If the memory is at
    /// capacity, the oldest episode is evicted before insertion.
    pub fn add(&mut self, observation: &str, triplets: &[String], step: u64) {
        if self.episodes.len() >= self.capacity {
            self.episodes.remove(0);
        }

        let episode = Episode {
            observation: observation.to_string(),
            triplets: triplets.to_vec(),
            fingerprint: label_fp(observation),
            step,
            truth: TruthValue::certain(),
        };

        self.episodes.push(episode);
    }

    /// Retrieve the `k` most similar episodes to the query string.
    ///
    /// Similarity is measured by Hamming distance on fingerprints (lower = closer).
    /// Returns episodes sorted by ascending distance (most similar first).
    pub fn top_k(&self, query: &str, k: usize) -> Vec<&Episode> {
        if self.episodes.is_empty() || k == 0 {
            return Vec::new();
        }

        let query_fp = label_fp(query);
        let mut scored: Vec<(u32, usize)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(i, ep)| (hamming_distance(&query_fp, &ep.fingerprint), i))
            .collect();

        scored.sort_by_key(|&(dist, _)| dist);

        scored
            .into_iter()
            .take(k)
            .map(|(_, idx)| &self.episodes[idx])
            .collect()
    }

    /// Retrieve episodes relevant to a plan string.
    ///
    /// Functionally identical to `top_k` but semantically distinct: used when
    /// the agent is planning rather than observing.
    pub fn retrieve_for_plan(&self, plan: &str, k: usize) -> Vec<&Episode> {
        self.top_k(plan, k)
    }

    /// Number of episodes currently stored.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// True if no episodes are stored.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Remove all episodes.
    pub fn clear(&mut self) {
        self.episodes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_retrieve() {
        let mut mem = EpisodicMemory::new(10);
        mem.add(
            "alpha bravo charlie",
            &["a - r - b".to_string()],
            1,
        );
        mem.add(
            "delta echo foxtrot",
            &["d - r - e".to_string()],
            2,
        );

        assert_eq!(mem.len(), 2);

        // Exact match query: should return the matching episode first
        let results = mem.top_k("alpha bravo charlie", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].observation, "alpha bravo charlie");
    }

    #[test]
    fn test_capacity_eviction() {
        let mut mem = EpisodicMemory::new(2);
        mem.add("first observation", &[], 1);
        mem.add("second observation", &[], 2);
        mem.add("third observation", &[], 3);

        assert_eq!(mem.len(), 2);
        // The first episode should have been evicted
        assert_eq!(mem.episodes[0].step, 2);
        assert_eq!(mem.episodes[1].step, 3);
    }

    #[test]
    fn test_top_k_ordering() {
        let mut mem = EpisodicMemory::new(10);
        mem.add("alpha beta gamma", &[], 1);
        mem.add("completely unrelated xyz", &[], 2);
        mem.add("alpha beta delta", &[], 3);

        let results = mem.top_k("alpha beta gamma", 2);
        assert_eq!(results.len(), 2);
        // Exact match should come first
        assert_eq!(results[0].observation, "alpha beta gamma");
    }

    #[test]
    fn test_top_k_empty_memory() {
        let mem = EpisodicMemory::new(10);
        let results = mem.top_k("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_top_k_zero_k() {
        let mut mem = EpisodicMemory::new(10);
        mem.add("some observation", &[], 1);
        let results = mem.top_k("some observation", 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_for_plan() {
        let mut mem = EpisodicMemory::new(10);
        mem.add("explored the dungeon", &[], 1);
        mem.add("found a key in the chest", &[], 2);

        // Exact match should come back first
        let results = mem.retrieve_for_plan("explored the dungeon", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].observation, "explored the dungeon");
    }

    #[test]
    fn test_clear() {
        let mut mem = EpisodicMemory::new(10);
        mem.add("something", &[], 1);
        mem.clear();
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
    }
}
