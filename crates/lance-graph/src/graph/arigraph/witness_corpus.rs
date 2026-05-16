// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! WitnessCorpus — replaces SpoWitnessChain<32> per L-9 / W5 spec §3.3.
//!
//! D-CSV-6a/6b: WitnessEntry, WitnessCorpus, WitnessIndexHashMap (sprint-12).
//! D-CSV-16 (sprint-13 W-I3): WitnessIndexCamPq, spo_to_fingerprint (Option A),
//!   enable_cam_pq, query_similar. Feature gate: `with-cam-pq`.
//!
//! Iron rules:
//! - W5-INV-CAM-PQ-INDEX: insert/evict mirrors to both indices when cam_pq enabled.
//! - I-VSA-IDENTITIES: adapter binds role-keyed IDENTITY fingerprints (not content).
//! - OQ-CSV-11: Option A VSA bind (NOT one-hot). Ratified.
//! - OQ-CSV-12: Lazy enable_cam_pq (NOT eager in new()). HashMap path zero-cost
//!   when `with-cam-pq` feature is off.

use bytes::Bytes;
use std::sync::Arc;

#[cfg(feature = "with-cam-pq")]
use ndarray::hpc::cam_pq::{CamCodebook, CamFingerprint, DistanceTables};

#[cfg(feature = "with-cam-pq")]
use lance_graph_contract::vsa::roles::{
    make_palette_id, make_role_key, CAM_PQ_DIM, O_SLICE_END, O_SLICE_START, P_SLICE_END,
    P_SLICE_START, S_SLICE_END, S_SLICE_START,
};

// ── WitnessEntry ─────────────────────────────────────────────────────────────

/// One peer-witnessed SPO observation. Compact (s, p, o) palette IDs +
/// timestamp + provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessEntry {
    /// SPO triple as packed (s_idx, p_idx, o_idx) palette indices.
    pub spo: u64,
    /// Nanoseconds since UNIX epoch. W5-INV-CHAIN-ORDER: sorted ASC on insert.
    pub timestamp_ns: u64,
    pub source_url: Option<String>,
    pub evidence_blob: Bytes,
}

impl WitnessEntry {
    pub fn tie_break_hash(&self) -> u64 {
        match &self.source_url {
            None => 0,
            Some(s) => {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                s.hash(&mut h);
                h.finish()
            }
        }
    }

    fn ord_key(&self) -> (u64, u64) {
        (self.timestamp_ns, self.tie_break_hash())
    }
}

/// Opaque handle for an inserted witness entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WitnessId(pub u64);

// ── WitnessIndexHashMap (sprint-12) ──────────────────────────────────────────

/// HashMap-backed exact-match index. Sprint-12; always present.
/// Per W5-INV-CAM-PQ-INDEX: canonical search structure for exact SPO lookups.
#[derive(Clone, Debug, Default)]
pub struct WitnessIndexHashMap {
    by_spo: std::collections::HashMap<u64, Vec<usize>>,
}

impl WitnessIndexHashMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, spo: u64, position: usize) {
        self.by_spo.entry(spo).or_default().push(position);
    }

    pub fn lookup(&self, spo: u64) -> &[usize] {
        self.by_spo.get(&spo).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn remove_at(&mut self, position: usize) {
        for positions in self.by_spo.values_mut() {
            positions.retain(|&p| p != position);
            for p in positions.iter_mut() {
                if *p > position {
                    *p -= 1;
                }
            }
        }
        self.by_spo.retain(|_, v| !v.is_empty());
    }

    pub fn clear(&mut self) {
        self.by_spo.clear();
    }

    pub fn len(&self) -> usize {
        self.by_spo.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.by_spo.is_empty()
    }

    pub(crate) fn rebuild_from_entries(entries: &[WitnessEntry]) -> Self {
        let mut idx = Self::new();
        for (pos, entry) in entries.iter().enumerate() {
            idx.insert(entry.spo, pos);
        }
        idx
    }
}

// ── WitnessIndexCamPq (sprint-13, D-CSV-16) ──────────────────────────────────

/// CAM-PQ-backed distance-ranked index. Sprint-13; feature `with-cam-pq` only.
///
/// Struct layout:
/// - `cam_fps[i]`   : 6-byte CAM fingerprint for the i-th cam slot.
/// - `cam_to_entry[i]`: entry position in `WitnessCorpus::entries` for cam slot i.
/// - `spo_to_slots` : SPO u64 → list of cam slot indices (for exact-match back-compat).
///
/// W5-INV-CAM-PQ-INDEX: mirrored on every insert/evict by WitnessCorpus.
#[cfg(feature = "with-cam-pq")]
#[derive(Clone, Debug)]
pub struct WitnessIndexCamPq {
    codec: CamCodebook,
    /// Subject role-key, 256D bipolar ±1 in dims [0..85).
    role_s: [f32; 256],
    /// Predicate role-key, 256D bipolar ±1 in dims [85..170).
    role_p: [f32; 256],
    /// Object role-key, 256D bipolar ±1 in dims [170..255).
    role_o: [f32; 256],
    /// 256-entry palette identity catalogue. Dense bipolar ±1. ~256 KB.
    palette_id: Box<[[f32; 256]; 256]>,
    /// 6-byte CAM fingerprints, one per cam slot.
    cam_fps: Vec<CamFingerprint>,
    /// cam slot → entry position (index into WitnessCorpus::entries).
    cam_to_entry: Vec<usize>,
    /// SPO u64 → list of cam slot indices.
    spo_to_slots: std::collections::HashMap<u64, Vec<usize>>,
}

/// Wraps WitnessIndexCamPq for storage in WitnessCorpus.
#[cfg(feature = "with-cam-pq")]
#[derive(Clone, Debug)]
pub struct CamPqState {
    pub index: WitnessIndexCamPq,
}

// ── Option A adapter: spo_to_fingerprint (OQ-CSV-11) ─────────────────────────

/// SPO u64 → 256D f32 identity fingerprint (Option A, OQ-CSV-11 ratified).
///
/// `v[i] = R_S[i]·id(s_idx)[i] + R_P[i]·id(p_idx)[i] + R_O[i]·id(o_idx)[i]`
///
/// Role keys have disjoint slices → `dot(R_S, R_P) = 0` exactly.
/// Per I-VSA-IDENTITIES: `palette_id[i]` are IDENTITY fingerprints (dense
/// bipolar ±1), NOT quantized content codes.
///
/// `spo` bit layout: `[0..8)` = s_idx, `[8..16)` = p_idx, `[16..24)` = o_idx.
#[cfg(feature = "with-cam-pq")]
pub fn spo_to_fingerprint(
    spo: u64,
    role_s: &[f32; CAM_PQ_DIM],
    role_p: &[f32; CAM_PQ_DIM],
    role_o: &[f32; CAM_PQ_DIM],
    palette_id: &[[f32; CAM_PQ_DIM]; 256],
) -> [f32; CAM_PQ_DIM] {
    let s_idx = (spo & 0xFF) as usize;
    let p_idx = ((spo >> 8) & 0xFF) as usize;
    let o_idx = ((spo >> 16) & 0xFF) as usize;

    let id_s = &palette_id[s_idx];
    let id_p = &palette_id[p_idx];
    let id_o = &palette_id[o_idx];

    let mut v = [0.0f32; CAM_PQ_DIM];
    for i in 0..CAM_PQ_DIM {
        v[i] = role_s[i] * id_s[i] + role_p[i] * id_p[i] + role_o[i] * id_o[i];
    }
    v
}

// ── WitnessIndexCamPq impl ────────────────────────────────────────────────────

#[cfg(feature = "with-cam-pq")]
impl WitnessIndexCamPq {
    /// Construct empty index with pre-trained codebook.
    /// Role-keys and palette derived deterministically from `seed`.
    pub fn new(codec: CamCodebook, seed: u64) -> Self {
        Self {
            role_s: make_role_key(b"S", S_SLICE_START, S_SLICE_END),
            role_p: make_role_key(b"P", P_SLICE_START, P_SLICE_END),
            role_o: make_role_key(b"O", O_SLICE_START, O_SLICE_END),
            palette_id: make_palette_id(seed),
            codec,
            cam_fps: Vec::new(),
            cam_to_entry: Vec::new(),
            spo_to_slots: std::collections::HashMap::new(),
        }
    }

    /// Insert (spo, entry_position) pair.
    ///
    /// Shifts spo_to_slots positions >= entry_position by +1 first,
    /// then encodes SPO → CamFingerprint and appends to cam_fps.
    pub fn insert(&mut self, spo: u64, entry_position: usize) {
        // Shift entry positions in cam_to_entry that are >= entry_position
        for ep in self.cam_to_entry.iter_mut() {
            if *ep >= entry_position {
                *ep += 1;
            }
        }

        // Encode SPO → 256D fingerprint → CamFingerprint
        let fp_vec = spo_to_fingerprint(
            spo,
            &self.role_s,
            &self.role_p,
            &self.role_o,
            &self.palette_id,
        );
        let cam_fp = self.codec.encode(&fp_vec);
        let cam_slot = self.cam_fps.len();
        self.cam_fps.push(cam_fp);
        self.cam_to_entry.push(entry_position);
        self.spo_to_slots.entry(spo).or_default().push(cam_slot);
    }

    /// Exact-match: return cam slots for `spo`.
    pub fn lookup_exact(&self, spo: u64) -> &[usize] {
        self.spo_to_slots
            .get(&spo)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Distance-ranked top-k lookup. Returns `Vec<(entry_position, distance)>`.
    ///
    /// Steps: encode query → precompute_distances → ADC per candidate → top-k sort.
    pub fn cam_pq_search(&self, query_spo: u64, k: usize) -> Vec<(usize, f32)> {
        if self.cam_fps.is_empty() || k == 0 {
            return vec![];
        }
        let fp_vec = spo_to_fingerprint(
            query_spo,
            &self.role_s,
            &self.role_p,
            &self.role_o,
            &self.palette_id,
        );
        let tables: DistanceTables = self.codec.precompute_distances(&fp_vec);

        let mut scored: Vec<(usize, f32)> = self
            .cam_fps
            .iter()
            .enumerate()
            .map(|(cam_slot, cam_fp)| {
                let dist = tables.distance(cam_fp);
                let entry_pos = self.cam_to_entry[cam_slot];
                (entry_pos, dist)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Rebuild index after eviction. `keep` = `(spo, entry_position)` for survivors.
    pub fn rebuild_after_evict(&mut self, keep: &[(u64, usize)]) {
        self.cam_fps.clear();
        self.cam_to_entry.clear();
        self.spo_to_slots.clear();

        for &(spo, entry_pos) in keep {
            let fp_vec = spo_to_fingerprint(
                spo,
                &self.role_s,
                &self.role_p,
                &self.role_o,
                &self.palette_id,
            );
            let cam_fp = self.codec.encode(&fp_vec);
            let cam_slot = self.cam_fps.len();
            self.cam_fps.push(cam_fp);
            self.cam_to_entry.push(entry_pos);
            self.spo_to_slots.entry(spo).or_default().push(cam_slot);
        }
    }

    pub fn len(&self) -> usize {
        self.cam_fps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cam_fps.is_empty()
    }
}

// ── WitnessCorpus ─────────────────────────────────────────────────────────────

/// Unbounded CAM-PQ-indexed witness corpus.
///
/// Two indices, two query classes:
/// - `cam_pq_index` (HashMap): exact-match O(1) — sprint-12. Always present.
/// - `cam_pq_state` (CAM-PQ): distance-ranked top-k — sprint-13. Lazy.
#[derive(Clone, Debug)]
pub struct WitnessCorpus {
    entries: Arc<Vec<WitnessEntry>>,
    /// Sprint-12 exact-match backbone. Always present (zero-cost when `with-cam-pq` off).
    pub(crate) cam_pq_index: WitnessIndexHashMap,
    /// Sprint-13: real CAM-PQ index. None until `enable_cam_pq()` called (OQ-CSV-12).
    #[cfg(feature = "with-cam-pq")]
    pub cam_pq_state: Option<CamPqState>,
}

impl Default for WitnessCorpus {
    fn default() -> Self {
        Self {
            entries: Arc::new(Vec::new()),
            cam_pq_index: WitnessIndexHashMap::new(),
            #[cfg(feature = "with-cam-pq")]
            cam_pq_state: None,
        }
    }
}

impl WitnessCorpus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a new witness. Maintains W5-INV-CHAIN-ORDER.
    ///
    /// W5-INV-CAM-PQ-INDEX: when cam_pq_state is Some, mirrors to CAM-PQ index.
    pub fn insert(&mut self, entry: WitnessEntry) -> WitnessId {
        let (t, h) = entry.ord_key();
        let id = WitnessId((t.wrapping_mul(31)) ^ h);
        let entries = Arc::make_mut(&mut self.entries);
        let pos = entries
            .binary_search_by(|e| e.ord_key().cmp(&(t, h)))
            .unwrap_or_else(|p| p);
        entries.insert(pos, entry);
        self.cam_pq_index = WitnessIndexHashMap::rebuild_from_entries(entries);

        #[cfg(feature = "with-cam-pq")]
        if let Some(state) = self.cam_pq_state.as_mut() {
            let keep: Vec<(u64, usize)> = entries
                .iter()
                .enumerate()
                .map(|(p, e)| (e.spo, p))
                .collect();
            state.index.rebuild_after_evict(&keep);
        }

        id
    }

    /// Exact-match query: entries matching `spo` in chain order.
    pub fn query(&self, spo: u64) -> impl Iterator<Item = &WitnessEntry> {
        let positions: Vec<usize> = self.cam_pq_index.lookup(spo).to_vec();
        positions
            .into_iter()
            .map(|pos| &self.entries[pos])
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, WitnessEntry> {
        self.entries.iter()
    }

    /// Evict entries strictly before `cutoff_ns`.
    ///
    /// W5-INV-CAM-PQ-INDEX: rebuilds both indices after retain.
    pub fn evict_stale_before(&mut self, cutoff_ns: u64) -> usize {
        let entries = Arc::make_mut(&mut self.entries);
        let n_before = entries.len();
        entries.retain(|e| e.timestamp_ns >= cutoff_ns);
        let evicted = n_before - entries.len();
        if evicted > 0 {
            self.cam_pq_index = WitnessIndexHashMap::rebuild_from_entries(entries);
            #[cfg(feature = "with-cam-pq")]
            if let Some(state) = self.cam_pq_state.as_mut() {
                let keep: Vec<(u64, usize)> = entries
                    .iter()
                    .enumerate()
                    .map(|(p, e)| (e.spo, p))
                    .collect();
                state.index.rebuild_after_evict(&keep);
            }
        }
        evicted
    }

    /// Exact-match top-k query by SPO.
    ///
    /// Always returns entries whose `spo` exactly equals the query; never
    /// returns unrelated neighbours. For distance-ranked nearest-neighbour
    /// search, use `query_similar` (feature `with-cam-pq`).
    pub fn cam_pq_search(&self, spo: u64, k: usize) -> Vec<&WitnessEntry> {
        self.cam_pq_index
            .lookup(spo)
            .iter()
            .take(k)
            .map(|&pos| &self.entries[pos])
            .collect()
    }
}

// ── CAM-PQ extension methods (feature-gated) ─────────────────────────────────

#[cfg(feature = "with-cam-pq")]
impl WitnessCorpus {
    /// Enable CAM-PQ indexing with a caller-supplied trained codebook.
    ///
    /// OQ-CSV-12: lazy — NOT called in new(). Backfills all existing entries.
    /// O(N × 256 × 6) one-time encode cost.
    pub fn enable_cam_pq(&mut self, codec: CamCodebook, seed: u64) {
        let mut idx = WitnessIndexCamPq::new(codec, seed);
        let keep: Vec<(u64, usize)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(p, e)| (e.spo, p))
            .collect();
        idx.rebuild_after_evict(&keep);
        self.cam_pq_state = Some(CamPqState { index: idx });
    }

    /// Distance-ranked top-k search. Returns `None` if CAM-PQ is not enabled.
    ///
    /// OQ-CSV-12: returns None (not a silent fallback) so callers know if the
    /// distance-ranked path is active.
    pub fn query_similar(&self, spo: u64, k: usize) -> Option<Vec<(&WitnessEntry, f32)>> {
        let state = self.cam_pq_state.as_ref()?;
        let hits = state.index.cam_pq_search(spo, k);
        let results = hits
            .into_iter()
            .filter_map(|(pos, dist)| self.entries.get(pos).map(|e| (e, dist)))
            .collect();
        Some(results)
    }

    /// True if CAM-PQ indexing has been enabled.
    pub fn is_cam_pq_enabled(&self) -> bool {
        self.cam_pq_state.is_some()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(spo: u64, ts: u64) -> WitnessEntry {
        WitnessEntry {
            spo,
            timestamp_ns: ts,
            source_url: None,
            evidence_blob: Bytes::new(),
        }
    }

    /// Tiny deterministic codebook: 252D (6 × 42), 256 centroids per subspace.
    #[cfg(feature = "with-cam-pq")]
    fn tiny_codebook() -> CamCodebook {
        use ndarray::hpc::cam_pq::{SubspaceCodebook, NUM_CENTROIDS, NUM_SUBSPACES};
        const TOTAL_DIM: usize = 252; // 6 × 42
        const SUB_DIM: usize = 42;

        let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
        let xorshift = |s: &mut u64| -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        };

        let codebooks: [SubspaceCodebook; NUM_SUBSPACES] = std::array::from_fn(|_| {
            let centroids: Vec<Vec<f32>> = (0..NUM_CENTROIDS)
                .map(|_| {
                    (0..SUB_DIM)
                        .map(|_| ((xorshift(&mut state) & 0xFFFF) as f32 / 32767.5) - 1.0)
                        .collect()
                })
                .collect();
            SubspaceCodebook {
                centroids,
                subspace_dim: SUB_DIM,
            }
        });

        CamCodebook {
            codebooks,
            total_dim: TOTAL_DIM,
            subspace_dim: SUB_DIM,
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Sprint-12 Tests (preserved)
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_witness_corpus_new_empty() {
        let corpus = WitnessCorpus::new();
        assert_eq!(corpus.len(), 0);
        assert!(corpus.is_empty());
        assert_eq!(corpus.iter().count(), 0);
    }

    #[test]
    fn test_witness_entry_ord_key() {
        let make = |ts: u64, url: Option<&str>| WitnessEntry {
            spo: 0,
            timestamp_ns: ts,
            source_url: url.map(String::from),
            evidence_blob: Bytes::new(),
        };
        let e_none = make(100, None);
        assert_eq!(e_none.tie_break_hash(), 0);
        assert_eq!(e_none.ord_key(), (100, 0));
        let e_a1 = make(200, Some("peer-A"));
        let e_a2 = make(200, Some("peer-A"));
        assert_eq!(e_a1.tie_break_hash(), e_a2.tie_break_hash());
        let e_b = make(200, Some("peer-B"));
        assert_ne!(e_a1.tie_break_hash(), e_b.tie_break_hash());
    }

    #[test]
    fn test_witness_corpus_insert_keeps_chain_order() {
        let mut corpus = WitnessCorpus::new();
        corpus.insert(make_entry(1, 200));
        corpus.insert(make_entry(1, 100));
        corpus.insert(make_entry(1, 150));
        let ts: Vec<u64> = corpus.iter().map(|e| e.timestamp_ns).collect();
        assert_eq!(ts, vec![100, 150, 200]);
    }

    #[test]
    fn test_witness_corpus_insert_tie_break() {
        let mut corpus = WitnessCorpus::new();
        let e1 = WitnessEntry {
            spo: 0xABC,
            timestamp_ns: 500,
            source_url: Some("peer-X".to_string()),
            evidence_blob: Bytes::new(),
        };
        let e2 = WitnessEntry {
            spo: 0xABC,
            timestamp_ns: 500,
            source_url: Some("peer-Y".to_string()),
            evidence_blob: Bytes::new(),
        };
        let hash1 = e1.tie_break_hash();
        let hash2 = e2.tie_break_hash();
        corpus.insert(e1);
        corpus.insert(e2);
        assert_eq!(corpus.len(), 2);
        let hashes: Vec<u64> = corpus.iter().map(|e| e.tie_break_hash()).collect();
        if hash1 < hash2 {
            assert_eq!(hashes, vec![hash1, hash2]);
        } else {
            assert_eq!(hashes, vec![hash2, hash1]);
        }
    }

    #[test]
    fn test_witness_corpus_query_filters_by_spo() {
        let mut corpus = WitnessCorpus::new();
        corpus.insert(make_entry(0xABC, 10));
        corpus.insert(make_entry(0xDEF, 15));
        corpus.insert(make_entry(0xABC, 30));
        corpus.insert(make_entry(0xDEF, 25));
        corpus.insert(make_entry(0xABC, 20));
        let abc: Vec<u64> = corpus.query(0xABC).map(|e| e.timestamp_ns).collect();
        let def: Vec<u64> = corpus.query(0xDEF).map(|e| e.timestamp_ns).collect();
        assert_eq!(abc, vec![10, 20, 30]);
        assert_eq!(def, vec![15, 25]);
    }

    #[test]
    fn test_witness_corpus_unbounded() {
        let mut corpus = WitnessCorpus::new();
        for i in 0u64..100 {
            corpus.insert(make_entry(i, i * 10));
        }
        assert_eq!(corpus.len(), 100);
    }

    #[test]
    fn test_witness_corpus_evict_stale_before() {
        let mut corpus = WitnessCorpus::new();
        for &ts in &[100u64, 200, 300, 400, 500] {
            corpus.insert(make_entry(0, ts));
        }
        let evicted = corpus.evict_stale_before(300);
        assert_eq!(evicted, 2);
        assert_eq!(corpus.len(), 3);
        let rem: Vec<u64> = corpus.iter().map(|e| e.timestamp_ns).collect();
        assert_eq!(rem, vec![300, 400, 500]);
    }

    #[test]
    fn test_witness_corpus_clone_is_cheap_via_arc() {
        let mut original = WitnessCorpus::new();
        original.insert(make_entry(1, 100));
        let mut cloned = original.clone();
        assert!(Arc::strong_count(&original.entries) >= 2);
        cloned.insert(make_entry(2, 200));
        assert_eq!(Arc::strong_count(&original.entries), 1);
        assert_eq!(original.len(), 1);
        assert_eq!(cloned.len(), 2);
    }

    #[test]
    fn test_cam_pq_index_empty() {
        let idx = WitnessIndexHashMap::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.lookup(0xABC), &[] as &[usize]);
    }

    #[test]
    fn test_cam_pq_index_insert_and_lookup() {
        let mut idx = WitnessIndexHashMap::new();
        idx.insert(0xABC, 0);
        idx.insert(0xABC, 2);
        idx.insert(0xDEF, 1);
        assert_eq!(idx.lookup(0xABC), &[0, 2]);
        assert_eq!(idx.lookup(0xDEF), &[1]);
        assert_eq!(idx.lookup(0x999), &[] as &[usize]);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn test_cam_pq_index_remove_at_shifts_positions() {
        let mut idx = WitnessIndexHashMap::new();
        idx.insert(0xAAA, 0);
        idx.insert(0xBBB, 1);
        idx.insert(0xAAA, 2);
        idx.remove_at(1);
        assert_eq!(idx.lookup(0xBBB), &[] as &[usize]);
        assert_eq!(idx.lookup(0xAAA), &[0, 1]);
    }

    #[test]
    fn test_witness_corpus_query_uses_index() {
        let mut corpus = WitnessCorpus::new();
        let target = 0xDEAD_BEEF_u64;
        let other = 0x1234_5678_u64;
        for i in 0u64..50 {
            corpus.insert(make_entry(target, i * 2 + 1));
            corpus.insert(make_entry(other, i * 2 + 2));
        }
        let results: Vec<&WitnessEntry> = corpus.query(target).collect();
        assert_eq!(results.len(), 50);
        assert!(results.iter().all(|e| e.spo == target));
        let ts: Vec<u64> = results.iter().map(|e| e.timestamp_ns).collect();
        let mut sorted = ts.clone();
        sorted.sort_unstable();
        assert_eq!(ts, sorted);
    }

    #[test]
    fn test_witness_corpus_evict_rebuilds_index() {
        let mut corpus = WitnessCorpus::new();
        let spo_a = 0xAAAA_u64;
        let spo_b = 0xBBBB_u64;
        for &(spo, ts) in &[
            (spo_a, 100u64),
            (spo_b, 200),
            (spo_a, 300),
            (spo_b, 400),
            (spo_a, 500),
        ] {
            corpus.insert(make_entry(spo, ts));
        }
        assert_eq!(corpus.evict_stale_before(300), 2);
        let a: Vec<u64> = corpus.query(spo_a).map(|e| e.timestamp_ns).collect();
        assert_eq!(a, vec![300, 500]);
        let b: Vec<u64> = corpus.query(spo_b).map(|e| e.timestamp_ns).collect();
        assert_eq!(b, vec![400]);
        let all: Vec<u64> = corpus.iter().map(|e| e.timestamp_ns).collect();
        assert_eq!(all, vec![300, 400, 500]);
    }

    #[test]
    fn test_witness_corpus_insert_maintains_index() {
        let mut corpus = WitnessCorpus::new();
        let spo = 0xCAFE_u64;
        for ts in [500u64, 300, 100, 400, 200] {
            corpus.insert(make_entry(spo, ts));
        }
        let results: Vec<u64> = corpus.query(spo).map(|e| e.timestamp_ns).collect();
        assert_eq!(results, vec![100, 200, 300, 400, 500]);
    }

    #[test]
    fn test_cam_pq_search_top_k() {
        let mut corpus = WitnessCorpus::new();
        let spo = 0xF00D_u64;
        for i in 1u64..=10 {
            corpus.insert(make_entry(spo, i * 10));
        }
        let results = corpus.cam_pq_search(spo, 3);
        assert_eq!(results.len(), 3);
        let ts: Vec<u64> = results.iter().map(|e| e.timestamp_ns).collect();
        assert_eq!(ts, vec![10, 20, 30]);
        assert_eq!(corpus.cam_pq_search(spo, 100).len(), 10);
        assert!(corpus.cam_pq_search(0xDEAD, 5).is_empty());
    }

    // ════════════════════════════════════════════════════════════════════════
    // D-CSV-16 Sprint-13 Tests (T1-T12)
    // ════════════════════════════════════════════════════════════════════════

    // T1: Option A round-trip — unbind and cosine-match recovers s_idx.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn spo_to_fingerprint_option_a_roundtrip() {
        use lance_graph_contract::vsa::roles::{
            make_palette_id, make_role_key, O_SLICE_END, O_SLICE_START, P_SLICE_END, P_SLICE_START,
            S_SLICE_END, S_SLICE_START,
        };

        let role_s = make_role_key(b"S", S_SLICE_START, S_SLICE_END);
        let role_p = make_role_key(b"P", P_SLICE_START, P_SLICE_END);
        let role_o = make_role_key(b"O", O_SLICE_START, O_SLICE_END);
        let palette = make_palette_id(0xCAFE_BABE);

        let s_idx: u8 = 42;
        let p_idx: u8 = 10;
        let o_idx: u8 = 7;
        let spo: u64 = s_idx as u64 | ((p_idx as u64) << 8) | ((o_idx as u64) << 16);

        let v = spo_to_fingerprint(spo, &role_s, &role_p, &role_o, &palette);

        // Unbind S: v ⊙ R_S
        let unbound_s: Vec<f32> = v
            .iter()
            .zip(role_s.iter())
            .map(|(&vi, &ri)| vi * ri)
            .collect();

        let cosine = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                dot / (na * nb)
            }
        };

        // Find best cosine match in palette
        let mut best_idx = 0usize;
        let mut best_cos = f32::NEG_INFINITY;
        for (i, entry) in palette.iter().enumerate() {
            let cos = cosine(&unbound_s, entry);
            if cos > best_cos {
                best_cos = cos;
                best_idx = i;
            }
        }

        assert_eq!(
            best_idx, s_idx as usize,
            "unbind+cosine must recover s_idx={s_idx}, got {best_idx}"
        );
        assert!(
            best_cos > 0.30,
            "cosine recovery must exceed 0.30, got {best_cos:.4}"
        );
    }

    // T2: cam_fps.len() == N after N inserts.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn cam_pq_codec_insert_roundtrip() {
        let mut idx = WitnessIndexCamPq::new(tiny_codebook(), 0xCAFE_BABE);
        for i in 0u64..100 {
            idx.insert(i & 0xFF_FFFF, i as usize);
        }
        assert_eq!(idx.len(), 100);
        assert_eq!(idx.cam_to_entry.len(), 100);
    }

    // T3: exact-match entry is top[0]; results in ascending distance order.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn cam_pq_search_top_k_matches_adc_distance_order() {
        let mut idx = WitnessIndexCamPq::new(tiny_codebook(), 0xCAFE_BABE);
        for i in 0u64..50 {
            idx.insert(i * 256, i as usize);
        }
        let query_spo = 7u64 * 256;
        let results = idx.cam_pq_search(query_spo, 5);
        assert!(!results.is_empty());
        let (top_pos, top_dist) = results[0];
        assert_eq!(top_pos, 7, "exact-match must be top-1, got pos={top_pos}");
        for (_, d) in &results[1..] {
            assert!(*d >= top_dist, "distances must be non-decreasing");
        }
    }

    // T4: partial query returns neighbours from the index.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn cam_pq_search_partial_query_returns_neighbours() {
        let mut idx = WitnessIndexCamPq::new(tiny_codebook(), 0xCAFE_BABE);
        for o in 0u64..10 {
            let spo = 1u64 | (2u64 << 8) | (o << 16);
            idx.insert(spo, o as usize);
        }
        let query_spo = 1u64 | (2u64 << 8) | (99u64 << 16);
        let results = idx.cam_pq_search(query_spo, 5);
        assert!(!results.is_empty(), "partial query must return neighbours");
        for i in 1..results.len() {
            assert!(
                results[i].1 >= results[i - 1].1,
                "distances must be non-decreasing"
            );
        }
    }

    // T5: after enable_cam_pq, HashMap path still works.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn back_compat_with_witness_index_hashmap() {
        let mut corpus = WitnessCorpus::new();
        let spo = 0xABCD_u64;
        for ts in [100u64, 200, 300] {
            corpus.insert(make_entry(spo, ts));
        }
        corpus.enable_cam_pq(tiny_codebook(), 0xCAFE_BABE);
        // Sprint-12 HashMap path unchanged
        assert_eq!(corpus.cam_pq_index.lookup(spo).len(), 3);
        let ts: Vec<u64> = corpus.query(spo).map(|e| e.timestamp_ns).collect();
        assert_eq!(ts, vec![100, 200, 300]);
    }

    // T6: out-of-order inserts keep both indices coherent.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn out_of_order_insert_keeps_both_indices_coherent() {
        let mut corpus = WitnessCorpus::new();
        corpus.enable_cam_pq(tiny_codebook(), 0xCAFE_BABE);
        corpus.insert(make_entry(0xA0, 200));
        corpus.insert(make_entry(0xB0, 100)); // goes before A0
        corpus.insert(make_entry(0xC0, 150)); // goes between B0 and A0

        let entries_spos: Vec<u64> = corpus.iter().map(|e| e.spo).collect();
        assert_eq!(
            entries_spos,
            vec![0xB0, 0xC0, 0xA0],
            "sorted by ts: 100,150,200"
        );

        let state = corpus.cam_pq_state.as_ref().unwrap();
        assert_eq!(state.index.len(), 3, "CAM-PQ must have 3 entries");
        // Every cam_to_entry value must be a valid entries index
        for &ep in &state.index.cam_to_entry {
            assert!(
                ep < corpus.len(),
                "cam_to_entry {ep} must be < corpus.len()"
            );
        }
    }

    // T7: lazy — cam_pq_state is None after new(); query_similar returns None.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn lazy_construction_starts_empty() {
        let corpus = WitnessCorpus::new();
        assert!(!corpus.is_cam_pq_enabled());
        // cam_pq_search falls back to HashMap (empty)
        assert!(corpus.cam_pq_search(0xABCD, 5).is_empty());
        // query_similar returns None
        assert!(corpus.query_similar(0xABCD, 5).is_none());
    }

    // T8: enable_cam_pq backfills all existing entries.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn enable_cam_pq_backfills_existing_entries() {
        let mut corpus = WitnessCorpus::new();
        for i in 0u64..10 {
            corpus.insert(make_entry(i * 256, i * 100));
        }
        assert!(!corpus.is_cam_pq_enabled());
        corpus.enable_cam_pq(tiny_codebook(), 0);
        assert!(corpus.is_cam_pq_enabled());
        assert_eq!(corpus.cam_pq_state.as_ref().unwrap().index.len(), 10);
    }

    // T9: eviction rebuilds CAM-PQ index; query_similar works post-evict.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn eviction_rebuilds_cam_pq_index() {
        let mut corpus = WitnessCorpus::new();
        for &(spo, ts) in &[
            (0x01u64, 100u64),
            (0x02, 200),
            (0x03, 300),
            (0x04, 400),
            (0x05, 500),
        ] {
            corpus.insert(make_entry(spo, ts));
        }
        corpus.enable_cam_pq(tiny_codebook(), 0xCAFE_BABE);
        assert_eq!(corpus.cam_pq_state.as_ref().unwrap().index.len(), 5);

        let evicted = corpus.evict_stale_before(300);
        assert_eq!(evicted, 2);
        assert_eq!(corpus.len(), 3);

        let state = corpus.cam_pq_state.as_ref().unwrap();
        assert_eq!(state.index.len(), 3);

        // query_similar must work for surviving spo
        let similar = corpus.query_similar(0x03, 1);
        assert!(similar.is_some());
        assert!(!similar.unwrap().is_empty());
    }

    // T10: dot(R_S, R_P) = dot(R_S, R_O) = dot(R_P, R_O) = 0.0 (disjoint slices).
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn option_a_role_orthogonality() {
        use lance_graph_contract::vsa::roles::{
            make_role_key, O_SLICE_END, O_SLICE_START, P_SLICE_END, P_SLICE_START, S_SLICE_END,
            S_SLICE_START,
        };
        let r_s = make_role_key(b"S", S_SLICE_START, S_SLICE_END);
        let r_p = make_role_key(b"P", P_SLICE_START, P_SLICE_END);
        let r_o = make_role_key(b"O", O_SLICE_START, O_SLICE_END);
        let dot = |a: &[f32; 256], b: &[f32; 256]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };
        assert_eq!(dot(&r_s, &r_p), 0.0, "dot(R_S, R_P) must be exactly 0.0");
        assert_eq!(dot(&r_s, &r_o), 0.0, "dot(R_S, R_O) must be exactly 0.0");
        assert_eq!(dot(&r_p, &r_o), 0.0, "dot(R_P, R_O) must be exactly 0.0");
    }

    // T11: feature ON smoke — WitnessIndexCamPq::new runs.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn build_with_with_cam_pq_feature_on() {
        let idx = WitnessIndexCamPq::new(tiny_codebook(), 0xCAFE_BABE);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    // Codex P2 (PR #396): cam_pq_search must remain exact-match even when
    // CAM-PQ is enabled. ANN search lives on query_similar. Asserts that a
    // missing SPO yields an empty result (not k unrelated neighbours) and
    // that an existing SPO yields only its own entries.
    #[cfg(feature = "with-cam-pq")]
    #[test]
    fn cam_pq_search_exact_match_when_cam_pq_enabled() {
        let mut corpus = WitnessCorpus::new();
        let spo_a = 0xAAAA_u64;
        let spo_b = 0xBBBB_u64;
        for ts in [100u64, 200, 300] {
            corpus.insert(make_entry(spo_a, ts));
        }
        for ts in [10u64, 20, 30] {
            corpus.insert(make_entry(spo_b, ts));
        }
        corpus.enable_cam_pq(tiny_codebook(), 0xCAFE_BABE);

        let hits_a = corpus.cam_pq_search(spo_a, 5);
        assert_eq!(hits_a.len(), 3, "must return exactly the 3 spo_a entries");
        assert!(
            hits_a.iter().all(|e| e.spo == spo_a),
            "no unrelated neighbours allowed"
        );

        let missing = corpus.cam_pq_search(0xCCCC_u64, 5);
        assert!(
            missing.is_empty(),
            "missing SPO must return empty, not k nearest neighbours; got {} entries",
            missing.len()
        );
    }

    // T12: feature OFF smoke — WitnessCorpus::new() and query() work.
    // (Compiles and runs under both feature flags; tests only HashMap path.)
    #[test]
    fn build_without_with_cam_pq_feature_off() {
        let mut corpus = WitnessCorpus::new();
        let spo = 0xDEAD_u64;
        corpus.insert(make_entry(spo, 100));
        corpus.insert(make_entry(spo, 200));
        let ts: Vec<u64> = corpus.query(spo).map(|e| e.timestamp_ns).collect();
        assert_eq!(ts, vec![100, 200]);
        let results = corpus.cam_pq_search(spo, 5);
        assert_eq!(results.len(), 2);
    }
}
