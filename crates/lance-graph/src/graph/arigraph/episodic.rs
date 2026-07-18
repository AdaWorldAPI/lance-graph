// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Episodic memory for AriGraph agents.
//!
//! Stores observation episodes with fingerprint-based similarity retrieval,
//! enabling agents to recall relevant past experiences.

use std::collections::{BTreeSet, HashMap};

use crate::graph::fingerprint::{hamming_distance, label_fp, Fingerprint};
use crate::graph::spo::truth::TruthValue;

/// Hardness threshold for unbundling — mirrors
/// `lance_graph_contract::crystal::UNBUNDLE_HARDNESS_THRESHOLD`.
///
/// A bundled crystal / episode whose NARS hardness exceeds this value
/// has accumulated enough evidence to be promoted to individually
/// addressable facts.
pub const UNBUNDLE_HARDNESS_THRESHOLD: f32 = 0.8;

/// Report returned by [`EpisodicMemory::unbundle_hardened`] and
/// [`EpisodicMemory::unbundle_targeted`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UnbundleReport {
    /// Number of episodes whose triplets were unbundled into `facts`.
    pub crystals_unbundled: u32,
    /// Count of triplet-level facts emitted.
    pub facts_emitted: u64,
    /// Maximum hardness observed among unbundled episodes (for telemetry).
    pub max_hardness: f32,
}

/// Report returned by [`EpisodicMemory::rebundle_cold`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RebundleReport {
    /// Number of individual triplet facts compacted back into episodes.
    pub facts_compacted: u64,
    /// Number of episodes still considered "cold" (below the step cutoff).
    pub cold_episodes: u32,
}

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

/// An **episodic-basin** partition over the entities observed in
/// [`EpisodicMemory`].
///
/// The *experiential* complement to
/// [`Communities`](super::community::Communities): entities are grouped by
/// **co-occurrence in episodes** (what was observed together), not by structural
/// graph connectivity (what is linked). Same accessor shape as `Communities`, so
/// the two partitions can be read the same way and compared by a caller.
#[derive(Debug, Clone)]
pub struct EpisodicBasins {
    /// Sorted, deduped entity names observed across episodes.
    pub entities: Vec<String>,
    /// Basin id per entity, parallel to [`Self::entities`].
    pub labels: Vec<u32>,
    /// Number of distinct basins.
    pub num_basins: usize,
}

impl EpisodicBasins {
    /// The basin id of `entity`, if it was observed in any episode.
    pub fn basin_of(&self, entity: &str) -> Option<u32> {
        self.entities
            .iter()
            .position(|e| e == entity)
            .map(|i| self.labels[i])
    }

    /// The entity names in `basin`.
    pub fn members(&self, basin: u32) -> Vec<&str> {
        self.entities
            .iter()
            .zip(&self.labels)
            .filter_map(|(e, &b)| if b == basin { Some(e.as_str()) } else { None })
            .collect()
    }
}

/// AriGraph Eq. 1 episode relevance: `n / (N · ln N)` for `N ≥ 2`, else `0`.
///
/// `n` = hit triplets incident to the episode, `N` = its total triplets. The
/// `N·ln N` normalizer down-weights large episodes; `ln 1 = 0` gives a
/// single-triplet episode zero weight (the denominator is 0 ⇒ we return 0
/// rather than divide). `n = 0` also yields 0.
fn episode_relevance(n: usize, total: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let nn = total.max(1) as f64;
    let denom = nn * nn.ln();
    if denom > 0.0 {
        n as f64 / denom
    } else {
        0.0
    }
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

    /// Maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
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

    /// Partition observed entities into **episodic basins**: two entities share
    /// a basin iff they co-occur in at least one stored episode (both names
    /// appear in that episode's `"subject - relation - object"` triplets). This
    /// is the *experiential* grouping — the complement to the *structural*
    /// [`TripletGraph::communities`](super::triplet_graph::TripletGraph::communities)
    /// partition. An entity never seen alongside another is its own basin.
    ///
    /// Deterministic: entities are sorted, and the union-find merges by lower
    /// root, so the same episode set always yields the same partition.
    pub fn basins(&self) -> EpisodicBasins {
        // 1. Per-episode entity sets + the global sorted entity list.
        let mut per_episode: Vec<Vec<String>> = Vec::with_capacity(self.episodes.len());
        let mut all: BTreeSet<String> = BTreeSet::new();
        for ep in &self.episodes {
            let mut ents: Vec<String> = Vec::new();
            for triplet in &ep.triplets {
                let parts: Vec<&str> = triplet.split(" - ").collect();
                if parts.len() == 3 {
                    for raw in [parts[0].trim(), parts[2].trim()] {
                        if !raw.is_empty() {
                            ents.push(raw.to_string());
                            all.insert(raw.to_string());
                        }
                    }
                }
            }
            ents.sort_unstable();
            ents.dedup();
            per_episode.push(ents);
        }
        let entities: Vec<String> = all.into_iter().collect();
        let index: HashMap<&str, usize> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.as_str(), i))
            .collect();

        // 2. Union-find over co-occurrence (all entities in one episode merge).
        let mut parent: Vec<usize> = (0..entities.len()).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for ents in &per_episode {
            let mut anchor: Option<usize> = None;
            for e in ents {
                let i = index[e.as_str()];
                match anchor {
                    None => anchor = Some(i),
                    Some(a) => {
                        let (ra, rb) = (find(&mut parent, a), find(&mut parent, i));
                        if ra != rb {
                            let (lo, hi) = (ra.min(rb), ra.max(rb));
                            parent[hi] = lo;
                        }
                    }
                }
            }
        }

        // 3. Densify roots to 0..k by first appearance (deterministic).
        let mut dense: HashMap<usize, u32> = HashMap::new();
        let mut next = 0u32;
        let labels: Vec<u32> = (0..entities.len())
            .map(|i| {
                let r = find(&mut parent, i);
                *dense.entry(r).or_insert_with(|| {
                    let d = next;
                    next += 1;
                    d
                })
            })
            .collect();
        let num_basins = dense.len();
        EpisodicBasins {
            entities,
            labels,
            num_basins,
        }
    }

    /// AriGraph **episodic search** (Anokhin et al. 2024, Eq. 1): rank stored
    /// episodes by how many of the caller's `semantic_hits` triplets are
    /// incident to each episode, normalized by the episode's size.
    ///
    /// This is the *chained* half of the AriGraph memory-graph search — the
    /// semantic (triplet) hits SEED the episodic recall, rather than a parallel
    /// fingerprint top-k. `semantic_hits` are triplet string-reprs (matching the
    /// `"subject - relation - object"` form stored on each [`Episode`]); an
    /// episode's relevance is
    ///
    /// ```text
    /// rel = n / (N · ln N)        for N ≥ 2
    /// rel = 0                     for N ≤ 1
    /// ```
    ///
    /// where `n` = hit triplets incident to the episode and `N` = its total
    /// triplets. The `N·ln N` normalizer down-weights large episodes (an
    /// incident hit in a focused episode counts for more) and gives
    /// single-triplet episodes zero weight (`ln 1 = 0`), per the paper. Episodes
    /// with no incident hit are dropped; the rest are returned as
    /// `(episode, rel)` sorted by `rel` descending (stable — insertion order
    /// breaks ties deterministically), truncated to `k`.
    ///
    /// Pure: reads only `Episode::triplets`. This is the retrieval *primitive*;
    /// chaining it after semantic search inside `OsintRetriever::retrieve`
    /// (replacing the parallel Hamming top-k) stays gated on the G0 verdict.
    #[must_use]
    pub fn episodic_search<'a>(
        &'a self,
        semantic_hits: &BTreeSet<String>,
        k: usize,
    ) -> Vec<(&'a Episode, f64)> {
        let mut scored: Vec<(&Episode, f64)> = self
            .episodes
            .iter()
            .filter_map(|ep| {
                let n = ep
                    .triplets
                    .iter()
                    .filter(|t| semantic_hits.contains(*t))
                    .count();
                let rel = episode_relevance(n, ep.triplets.len());
                (rel > 0.0).then_some((ep, rel))
            })
            .collect();
        // rel descending; stable sort keeps insertion order within ties.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// True if no episodes are stored.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Remove all episodes.
    pub fn clear(&mut self) {
        self.episodes.clear();
    }

    /// Scan all episodes and unbundle each whose hardness exceeds
    /// [`UNBUNDLE_HARDNESS_THRESHOLD`] into individually-addressable
    /// triplet facts.
    ///
    /// Hardness is derived from the NARS truth confidence on each
    /// episode. Unbundled episodes remain in memory (their triplets
    /// are lifted into the fact set, not deleted).
    ///
    /// Returns an [`UnbundleReport`] and the emitted facts paired with
    /// their source fingerprints.
    pub fn unbundle_hardened(&self) -> (UnbundleReport, Vec<(Fingerprint, String)>) {
        let mut report = UnbundleReport::default();
        let mut facts = Vec::new();
        for ep in &self.episodes {
            let hardness = ep.truth.confidence;
            if hardness >= UNBUNDLE_HARDNESS_THRESHOLD {
                report.crystals_unbundled += 1;
                if hardness > report.max_hardness {
                    report.max_hardness = hardness;
                }
                for t in &ep.triplets {
                    facts.push((ep.fingerprint, t.clone()));
                    report.facts_emitted += 1;
                }
            }
        }
        (report, facts)
    }

    /// Unbundle exactly one episode identified by fingerprint match.
    ///
    /// Compares with Hamming distance ≤ `max_distance`. Returns the
    /// report + emitted facts from the closest match (SIMD-dispatched
    /// batch scan under `ndarray-hpc`, scalar otherwise).
    pub fn unbundle_targeted(
        &self,
        fp: &Fingerprint,
        max_distance: u32,
    ) -> (UnbundleReport, Vec<String>) {
        let mut report = UnbundleReport::default();
        let mut facts = Vec::new();
        if self.episodes.is_empty() {
            return (report, facts);
        }

        let best = self.best_match_index(fp, max_distance);
        if let Some(idx) = best {
            let ep = &self.episodes[idx];
            report.crystals_unbundled = 1;
            report.max_hardness = ep.truth.confidence;
            for t in &ep.triplets {
                facts.push(t.clone());
                report.facts_emitted += 1;
            }
        }
        (report, facts)
    }

    /// Index of the closest episode whose Hamming distance to `fp` is
    /// ≤ `max_distance`, or `None` if no match exists.
    ///
    /// Uses `ndarray::hpc::bitwise::hamming_batch_raw` (VPOPCNTDQ /
    /// AVX-512BW / AVX2 / scalar dispatch) when `ndarray-hpc` is
    /// enabled; falls back to per-episode scalar otherwise.
    fn best_match_index(&self, fp: &Fingerprint, max_distance: u32) -> Option<usize> {
        #[cfg(feature = "ndarray-hpc")]
        {
            // Fingerprint is [u64; 8] = 64 bytes; build a contiguous
            // database for the SIMD batch call.
            const FP_BYTES: usize = core::mem::size_of::<Fingerprint>();
            let n = self.episodes.len();
            let mut db: Vec<u8> = Vec::with_capacity(n * FP_BYTES);
            for ep in &self.episodes {
                // SAFETY: Fingerprint is [u64; 8], which is plain-old-
                // data and has no padding. Reinterpreting as &[u8] of
                // the same byte length is always safe.
                let bytes = unsafe {
                    core::slice::from_raw_parts(ep.fingerprint.as_ptr() as *const u8, FP_BYTES)
                };
                db.extend_from_slice(bytes);
            }
            let q = unsafe { core::slice::from_raw_parts(fp.as_ptr() as *const u8, FP_BYTES) };
            let dists = ndarray::hpc::bitwise::hamming_batch_raw(q, &db, n, FP_BYTES);
            let mut best: Option<(usize, u64)> = None;
            for (i, &d) in dists.iter().enumerate() {
                if d as u32 > max_distance {
                    continue;
                }
                match best {
                    None => best = Some((i, d)),
                    Some((_, bd)) if d < bd => best = Some((i, d)),
                    _ => {}
                }
            }
            best.map(|(i, _)| i)
        }

        #[cfg(not(feature = "ndarray-hpc"))]
        {
            let mut best: Option<(usize, u32)> = None;
            for (i, ep) in self.episodes.iter().enumerate() {
                let d = hamming_distance(fp, &ep.fingerprint);
                if d > max_distance {
                    continue;
                }
                match best {
                    None => best = Some((i, d)),
                    Some((_, bd)) if d < bd => best = Some((i, d)),
                    _ => {}
                }
            }
            best.map(|(i, _)| i)
        }
    }

    /// Compact cold episodes — those older than `step_cutoff` and with
    /// low hardness — back into bundled form by removing their
    /// individual triplet lists.
    ///
    /// Returns a [`RebundleReport`]. The underlying episodes remain
    /// searchable by fingerprint; only the per-triplet addressability
    /// is retired to reclaim space.
    pub fn rebundle_cold(&mut self, step_cutoff: u64) -> RebundleReport {
        let mut report = RebundleReport::default();
        for ep in self.episodes.iter_mut() {
            if ep.step < step_cutoff && ep.truth.confidence < UNBUNDLE_HARDNESS_THRESHOLD {
                report.facts_compacted += ep.triplets.len() as u64;
                ep.triplets.clear();
                report.cold_episodes += 1;
            }
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_retrieve() {
        let mut mem = EpisodicMemory::new(10);
        mem.add("alpha bravo charlie", &["a - r - b".to_string()], 1);
        mem.add("delta echo foxtrot", &["d - r - e".to_string()], 2);

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

    fn set_hardness(mem: &mut EpisodicMemory, idx: usize, hardness: f32) {
        mem.episodes[idx].truth = TruthValue::new(mem.episodes[idx].truth.frequency, hardness);
    }

    #[test]
    fn unbundle_hardened_emits_facts_from_hard_episodes() {
        let mut mem = EpisodicMemory::new(10);
        mem.add("soft", &["a - r - b".into()], 1);
        mem.add("hard", &["c - r - d".into(), "e - r - f".into()], 2);
        mem.add("harder", &["g - r - h".into()], 3);
        set_hardness(&mut mem, 0, 0.3); // below threshold
        set_hardness(&mut mem, 1, 0.85); // above
        set_hardness(&mut mem, 2, 0.95); // above

        let (report, facts) = mem.unbundle_hardened();
        assert_eq!(report.crystals_unbundled, 2);
        assert_eq!(report.facts_emitted, 3);
        assert_eq!(facts.len(), 3);
        assert!(report.max_hardness >= 0.94 && report.max_hardness <= 0.96);
    }

    #[test]
    fn unbundle_targeted_returns_facts_of_match() {
        let mut mem = EpisodicMemory::new(10);
        mem.add("alpha bravo", &["x - r - y".into()], 1);
        mem.add("delta echo", &["u - r - v".into()], 2);
        set_hardness(&mut mem, 0, 0.9);

        let fp = label_fp("alpha bravo");
        let (report, facts) = mem.unbundle_targeted(&fp, 8);
        assert_eq!(report.crystals_unbundled, 1);
        assert_eq!(report.facts_emitted, 1);
        assert_eq!(facts, vec!["x - r - y".to_string()]);
    }

    #[test]
    fn rebundle_cold_clears_old_soft_episode_triplets() {
        let mut mem = EpisodicMemory::new(10);
        mem.add("aged-soft", &["p - r - q".into(), "s - r - t".into()], 1);
        mem.add("aged-hard", &["m - r - n".into()], 2);
        mem.add("young-soft", &["j - r - k".into()], 100);
        set_hardness(&mut mem, 0, 0.3); // soft, aged → rebundle
        set_hardness(&mut mem, 1, 0.9); // hard, aged → keep
        set_hardness(&mut mem, 2, 0.3); // soft, young → keep

        let report = mem.rebundle_cold(10);
        assert_eq!(report.cold_episodes, 1);
        assert_eq!(report.facts_compacted, 2);
        // aged-soft had its triplets compacted
        assert_eq!(mem.episodes[0].triplets.len(), 0);
        // aged-hard untouched
        assert_eq!(mem.episodes[1].triplets.len(), 1);
        // young-soft untouched
        assert_eq!(mem.episodes[2].triplets.len(), 1);
    }

    #[test]
    fn threshold_matches_contract_constant() {
        assert!((UNBUNDLE_HARDNESS_THRESHOLD - 0.8).abs() < 1e-6);
    }

    // ── episodic basins ──────────────────────────────────────────────────

    fn basin_mem(episodes: &[&[&str]]) -> EpisodicMemory {
        let mut m = EpisodicMemory::new(64);
        for (i, trips) in episodes.iter().enumerate() {
            let ts: Vec<String> = trips.iter().map(|s| s.to_string()).collect();
            m.add(&format!("obs{i}"), &ts, i as u64);
        }
        m
    }

    #[test]
    fn basins_group_co_occurring_entities() {
        let m = basin_mem(&[&["alice - knows - bob"], &["carol - knows - dave"]]);
        let b = m.basins();
        assert_eq!(b.num_basins, 2);
        assert_eq!(b.basin_of("alice"), b.basin_of("bob"));
        assert_eq!(b.basin_of("carol"), b.basin_of("dave"));
        assert_ne!(b.basin_of("alice"), b.basin_of("carol"));
    }

    #[test]
    fn basins_merge_on_a_shared_entity() {
        // `bob` bridges the two episodes → one basin of {alice, bob, carol}.
        let m = basin_mem(&[&["alice - knows - bob"], &["bob - knows - carol"]]);
        let b = m.basins();
        assert_eq!(b.num_basins, 1);
        assert_eq!(b.members(b.basin_of("alice").unwrap()).len(), 3);
    }

    #[test]
    fn basins_union_all_entities_in_one_episode() {
        let m = basin_mem(&[&["a - r - b", "b - r - c", "c - r - d"]]);
        let b = m.basins();
        assert_eq!(b.num_basins, 1);
        assert_eq!(b.entities.len(), 4);
    }

    #[test]
    fn basins_empty_memory_is_safe() {
        let b = EpisodicMemory::new(8).basins();
        assert_eq!(b.num_basins, 0);
        assert!(b.entities.is_empty());
    }

    #[test]
    fn basins_deterministic() {
        let m = basin_mem(&[&["alice - r - bob"], &["bob - r - carol"]]);
        let (x, y) = (m.basins(), m.basins());
        assert_eq!(x.entities, y.entities);
        assert_eq!(x.labels, y.labels);
    }

    // ── episodic search (AriGraph Eq. 1) ─────────────────────────────────

    fn hits(ts: &[&str]) -> BTreeSet<String> {
        ts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn episodic_search_ranks_by_incidence() {
        // ep0 has 2 incident hits, ep1 has 1 — same size ⇒ ep0 ranks higher.
        let m = basin_mem(&[&["a - r - b", "c - r - d"], &["a - r - b", "x - r - y"]]);
        let out = m.episodic_search(&hits(&["a - r - b", "c - r - d"]), 10);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0.observation, "obs0");
        assert!(out[0].1 > out[1].1);
    }

    #[test]
    fn episodic_search_downweights_large_episodes() {
        // Both episodes have exactly 1 incident hit ("a - r - b"); the smaller
        // episode wins per the N·ln N normalizer.
        let m = basin_mem(&[
            &["a - r - b", "x - r - y"],
            &[
                "a - r - b",
                "p1 - r - q1",
                "p2 - r - q2",
                "p3 - r - q3",
                "p4 - r - q4",
            ],
        ]);
        let out = m.episodic_search(&hits(&["a - r - b"]), 10);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0.observation, "obs0"); // the small one
        assert!(out[0].1 > out[1].1);
    }

    #[test]
    fn episodic_search_single_triplet_episode_is_zero_weight() {
        // N = 1 ⇒ ln 1 = 0 ⇒ rel 0, even though the one triplet is a hit.
        let m = basin_mem(&[&["a - r - b"]]);
        let out = m.episodic_search(&hits(&["a - r - b"]), 10);
        assert!(out.is_empty());
    }

    #[test]
    fn episodic_search_drops_episodes_without_incidence() {
        let m = basin_mem(&[&["a - r - b", "c - r - d"], &["e - r - f", "g - r - h"]]);
        let out = m.episodic_search(&hits(&["a - r - b"]), 10);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0.observation, "obs0");
    }

    #[test]
    fn episodic_search_empty_is_safe() {
        let m = basin_mem(&[&["a - r - b", "c - r - d"]]);
        assert!(m.episodic_search(&BTreeSet::new(), 10).is_empty()); // no hits
        assert!(EpisodicMemory::new(8)
            .episodic_search(&hits(&["a - r - b"]), 10)
            .is_empty()); // no episodes
    }

    #[test]
    fn episodic_search_respects_k() {
        let m = basin_mem(&[
            &["a - r - b", "z - r - w"],
            &["a - r - b", "z - r - w"],
            &["a - r - b", "z - r - w"],
        ]);
        assert_eq!(m.episodic_search(&hits(&["a - r - b"]), 2).len(), 2);
    }
}
