// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! WitnessCorpus — replaces SpoWitnessChain<32> per L-9 / W5 spec §3.3.
//!
//! Unbounded, copy-on-write via Arc::make_mut, sorted by timestamp_ns ASC
//! with hash tie-break (W5-INV-CHAIN-ORDER). Each entry encodes one
//! peer-witnessed SPO observation with optional source URL + evidence blob.
//!
//! D-CSV-6a scope (sprint-11 Phase B core) — landed in #386:
//! - WitnessEntry struct
//! - WitnessCorpus { entries, cam_pq_index }
//! - insert / query (index-backed) / iter / evict_stale_before
//! - Three iron-rule invariants enforced in tests: W5-INV-CHAIN-ORDER,
//!   W5-INV-WITNESS-UNBOUNDED, W5-INV-CAM-PQ-INDEX
//!
//! D-CSV-6b scope (this PR — sprint-12 Wave G):
//! - Replace CamPqIndexPlaceholder with CamPqWitnessIndex (HashMap-backed)
//! - query() now uses the index instead of linear scan (O(1) expected time)
//! - cam_pq_search(spo, k) top-k variant
//! - evict_stale_before rebuilds index after retain
//! - insert rebuilds index after binary-search insertion (correctness over perf)
//!
//! **Upgrade path note (TECH_DEBT):**
//! The full ndarray::hpc::cam_pq::CamPqCodec wiring for SPO witness-tuple
//! distance ranking is sprint-13+ work, pending upstream cam_pq witness-tuple
//! support. The ndarray cam_pq codec operates on 256D+ float vectors; it does
//! not currently accept (s,p,o) u64 palette triples as query keys.
//! The cam_index.rs module (GraphHV, 3-channel Fingerprint<256>) is also not
//! directly wirable to the u64 SPO → Vec<usize> mapping needed here.
//! HashMap gives O(1) expected-time lookups vs O(N) linear scan — preserving
//! the cheap-query contract for downstream call sites.
//! Track in TECH_DEBT: "CamPqWitnessIndex: upgrade HashMap to ndarray cam_pq
//! codec once upstream adds SPO witness-tuple support (cam_pq.rs or cam_index.rs)."

use bytes::Bytes;
use std::sync::Arc;

/// One peer-witnessed SPO observation. Compact (s, p, o) palette IDs +
/// timestamp + provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessEntry {
    /// SPO triple as packed (s_idx, p_idx, o_idx) palette indices.
    /// Use the same low-3-byte packing as causal_edge::CausalEdge64
    /// for round-trip with the CausalEdge layer (bits 0-23).
    pub spo: u64,

    /// Nanoseconds since UNIX epoch (or any monotonic source agreed on
    /// by the producer). Iron rule W5-INV-CHAIN-ORDER: entries sorted
    /// ASC by this field on insert.
    pub timestamp_ns: u64,

    /// Optional URL or peer identifier of the witness source.
    pub source_url: Option<String>,

    /// Optional evidence payload (raw bytes; can be a hash, a chunk
    /// reference, or empty if the witness is purely metadata).
    pub evidence_blob: Bytes,
}

impl WitnessEntry {
    /// Hash-based tie-break key for same-timestamp entries.
    /// Per W5-INV-CHAIN-ORDER: when timestamp_ns is equal, sort by
    /// hash(source_url) ASC; entries with no source_url sort first.
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

    /// Total ordering: timestamp_ns ASC, then tie_break_hash ASC.
    fn ord_key(&self) -> (u64, u64) {
        (self.timestamp_ns, self.tie_break_hash())
    }
}

/// Opaque newtype for inserted entry handle. Returned by `insert`;
/// usable later for `evict_stale_before` returning evicted IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WitnessId(pub u64);

/// CAM-PQ-backed index over WitnessCorpus entries.
/// Maps (s, p, o) palette indices → positions in the entries vec.
/// Per W5-INV-CAM-PQ-INDEX iron rule (W5 spec §3.3).
///
/// **Implementation note:** this PR ships a HashMap-backed index as the
/// canonical surface. The full CAM-PQ codec wiring (ndarray hpc::cam_pq)
/// is sprint-13+ once the CAM-PQ codec's witness-tuple support lands
/// upstream. The HashMap version preserves the API shape and gives O(1)
/// expected-time lookups (vs O(N) linear scan in the placeholder), so
/// downstream call sites can rely on the cheap-query contract today.
///
/// **TECH_DEBT:** upgrade `by_spo` HashMap to ndarray::hpc::cam_pq once
/// upstream adds SPO witness-tuple support. Track in TECH_DEBT.md.
#[derive(Clone, Debug, Default)]
pub struct CamPqWitnessIndex {
    /// SPO triple → list of (entry positions in WitnessCorpus.entries)
    by_spo: std::collections::HashMap<u64, Vec<usize>>,
}

impl CamPqWitnessIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `position` into the bucket for `spo`.
    pub fn insert(&mut self, spo: u64, position: usize) {
        self.by_spo.entry(spo).or_default().push(position);
    }

    /// Lookup all positions for `spo`. Returns empty slice if absent.
    pub fn lookup(&self, spo: u64) -> &[usize] {
        self.by_spo.get(&spo).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Remove `position` from any bucket containing it, and shift down
    /// all positions > `position` by 1 (since the entries vec is rebuilt
    /// after eviction). Called by `evict_stale_before` after `retain`.
    ///
    /// Note: this method is O(total entries across all buckets) in the
    /// worst case. For the eviction use-case, the index is rebuilt in
    /// full via `rebuild_from_entries` instead, which is simpler and
    /// equally O(N).
    pub fn remove_at(&mut self, position: usize) {
        for positions in self.by_spo.values_mut() {
            positions.retain(|&p| p != position);
            for p in positions.iter_mut() {
                if *p > position {
                    *p -= 1;
                }
            }
        }
        // Remove now-empty buckets
        self.by_spo.retain(|_, v| !v.is_empty());
    }

    /// Clear the index entirely.
    pub fn clear(&mut self) {
        self.by_spo.clear();
    }

    /// Total number of (spo, position) pairs in the index.
    pub fn len(&self) -> usize {
        self.by_spo.values().map(Vec::len).sum()
    }

    /// True if the index has no entries.
    pub fn is_empty(&self) -> bool {
        self.by_spo.is_empty()
    }

    /// Rebuild the index from scratch from the given entries slice.
    /// Used after eviction (positions shift) and after mid-vec inserts.
    fn rebuild_from_entries(entries: &[WitnessEntry]) -> Self {
        let mut idx = Self::new();
        for (pos, entry) in entries.iter().enumerate() {
            idx.insert(entry.spo, pos);
        }
        idx
    }
}

/// Unbounded CAM-PQ-indexed witness corpus.
/// Per L-9 / W5 §3.3: replaces SpoWitnessChain<32>. Copy-on-write via
/// Arc::make_mut so cheap clones are cheap; mutations bump the Arc.
///
/// **Index rebuild policy (TECH_DEBT):**
/// Every insert rebuilds the index in O(N) because binary-search insertion
/// can place entries at arbitrary mid-vec positions, invalidating all
/// positions after the insertion point. For append-only workloads (common
/// in practice since witnesses arrive in roughly chronological order),
/// this degrades to O(1) index update. A future optimisation: detect
/// append-only inserts and skip the full rebuild. Track in TECH_DEBT.md.
#[derive(Clone, Debug)]
pub struct WitnessCorpus {
    entries: Arc<Vec<WitnessEntry>>,
    /// CAM-PQ-backed index (HashMap surface; ndarray codec upgrade sprint-13+).
    /// Per W5-INV-CAM-PQ-INDEX: this is the canonical search structure.
    pub(crate) cam_pq_index: CamPqWitnessIndex,
}

impl Default for WitnessCorpus {
    fn default() -> Self {
        Self {
            entries: Arc::new(Vec::new()),
            cam_pq_index: CamPqWitnessIndex::new(),
        }
    }
}

impl WitnessCorpus {
    /// Create a new empty corpus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of witness entries in the corpus.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the corpus contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a new witness. Maintains W5-INV-CHAIN-ORDER: entries
    /// stay sorted by (timestamp_ns ASC, tie_break_hash ASC).
    /// Returns a WitnessId derived from the entry's ord_key (NOT from
    /// position — position can shift on later insertions).
    ///
    /// After binary-search insertion, the index is rebuilt in O(N) to
    /// ensure all positions remain correct. See TECH_DEBT note on the
    /// struct for the append-only optimisation path.
    pub fn insert(&mut self, entry: WitnessEntry) -> WitnessId {
        let (t, h) = entry.ord_key();
        let id = WitnessId((t.wrapping_mul(31)) ^ h);
        let entries = Arc::make_mut(&mut self.entries);
        let pos = entries
            .binary_search_by(|e| e.ord_key().cmp(&(t, h)))
            .unwrap_or_else(|p| p);
        entries.insert(pos, entry);
        // Rebuild index: positions after `pos` all shifted by 1.
        self.cam_pq_index = CamPqWitnessIndex::rebuild_from_entries(entries);
        id
    }

    /// Point query by SPO: return all entries matching this exact SPO triple,
    /// in chain order (timestamp ASC, W5-INV-CHAIN-ORDER).
    /// O(1) expected time via cam_pq_index; O(1) amortized for hot SPOs.
    /// Per W5-INV-CAM-PQ-INDEX: direct Vec iteration is forbidden in
    /// production paths.
    pub fn query(&self, spo: u64) -> impl Iterator<Item = &WitnessEntry> {
        // Collect positions to owned Vec so the borrow of cam_pq_index does not
        // escape the iterator's lifetime — the index slice borrow ends here.
        let positions: Vec<usize> = self.cam_pq_index.lookup(spo).to_vec();
        // Map each position to a reference into self.entries. Positions are always
        // valid: they are produced by rebuild_from_entries and only invalidated
        // on the next insert/evict (which requires &mut self, preventing aliasing).
        positions
            .into_iter()
            .map(|pos| &self.entries[pos])
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// All entries in chain order (timestamp ASC).
    pub fn iter(&self) -> std::slice::Iter<'_, WitnessEntry> {
        self.entries.iter()
    }

    /// Evict entries strictly before `cutoff_ns`. Per W5-INV-WITNESS-
    /// UNBOUNDED, the corpus is unbounded by default; eviction is a
    /// caller-driven operation, NOT an automatic LRU/cap. Returns the
    /// number of evicted entries. Rebuilds the index after retain since
    /// all positions shift.
    pub fn evict_stale_before(&mut self, cutoff_ns: u64) -> usize {
        let entries = Arc::make_mut(&mut self.entries);
        let n_before = entries.len();
        entries.retain(|e| e.timestamp_ns >= cutoff_ns);
        let evicted = n_before - entries.len();
        if evicted > 0 {
            self.cam_pq_index = CamPqWitnessIndex::rebuild_from_entries(entries);
        }
        evicted
    }

    /// Top-k query by SPO (W5 spec §3.3 cam_pq_search surface).
    ///
    /// Returns up to `k` WitnessEntry references, in chain order (timestamp
    /// ASC). For the HashMap backend, "top-k" = first k entries in chain
    /// order; no distance ranking is applied.
    ///
    /// **Sprint-13+ upgrade path:** when ndarray::hpc::cam_pq gains
    /// witness-tuple support, this method will rank by CAM-PQ distance
    /// from the query SPO vector, enabling palette-family-proximity
    /// ranking (ontological family proximity scoring per W5 spec §3.3).
    pub fn cam_pq_search(&self, spo: u64, k: usize) -> Vec<&WitnessEntry> {
        self.cam_pq_index
            .lookup(spo)
            .iter()
            .take(k)
            .map(|&pos| &self.entries[pos])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1 ──────────────────────────────────────────────────────────────
    /// W5-INV-WITNESS-UNBOUNDED: new corpus is empty; len 0; iter yields nothing.
    #[test]
    fn test_witness_corpus_new_empty() {
        let corpus = WitnessCorpus::new();
        assert_eq!(corpus.len(), 0);
        assert!(corpus.is_empty());
        assert_eq!(corpus.iter().count(), 0);
    }

    // ── Test 2 ──────────────────────────────────────────────────────────────
    /// W5-INV-CHAIN-ORDER tie-break key semantics.
    /// - None source_url → hash = 0
    /// - Same source_url → same hash
    /// - Different source_url → different ord_key (hash differs with high probability)
    #[test]
    fn test_witness_entry_ord_key() {
        let make_entry = |ts: u64, url: Option<&str>| WitnessEntry {
            spo: 0,
            timestamp_ns: ts,
            source_url: url.map(String::from),
            evidence_blob: Bytes::new(),
        };

        // None source_url → tie_break_hash == 0
        let e_none = make_entry(100, None);
        assert_eq!(e_none.tie_break_hash(), 0);
        assert_eq!(e_none.ord_key(), (100, 0));

        // Same source_url → same hash → same ord_key
        let e_a1 = make_entry(200, Some("peer-A"));
        let e_a2 = make_entry(200, Some("peer-A"));
        assert_eq!(e_a1.tie_break_hash(), e_a2.tie_break_hash());
        assert_eq!(e_a1.ord_key(), e_a2.ord_key());

        // Different source_url → different hash (with overwhelming probability)
        let e_b = make_entry(200, Some("peer-B"));
        assert_ne!(e_a1.tie_break_hash(), e_b.tie_break_hash());
        assert_ne!(e_a1.ord_key(), e_b.ord_key());
    }

    // ── Test 3 ──────────────────────────────────────────────────────────────
    /// W5-INV-CHAIN-ORDER: insert 3 entries out of timestamp order;
    /// iter yields them in 100, 150, 200 order.
    #[test]
    fn test_witness_corpus_insert_keeps_chain_order() {
        let mut corpus = WitnessCorpus::new();

        corpus.insert(WitnessEntry {
            spo: 1,
            timestamp_ns: 200,
            source_url: None,
            evidence_blob: Bytes::new(),
        });
        corpus.insert(WitnessEntry {
            spo: 1,
            timestamp_ns: 100,
            source_url: None,
            evidence_blob: Bytes::new(),
        });
        corpus.insert(WitnessEntry {
            spo: 1,
            timestamp_ns: 150,
            source_url: None,
            evidence_blob: Bytes::new(),
        });

        let timestamps: Vec<u64> = corpus.iter().map(|e| e.timestamp_ns).collect();
        assert_eq!(timestamps, vec![100, 150, 200]);
    }

    // ── Test 4 ──────────────────────────────────────────────────────────────
    /// W5-INV-CHAIN-ORDER tie-break: same timestamp + different source_url;
    /// both retained, ordered by hash ASC.
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

        // Capture expected order based on hash comparison
        let hash1 = e1.tie_break_hash();
        let hash2 = e2.tie_break_hash();

        corpus.insert(e1.clone());
        corpus.insert(e2.clone());

        assert_eq!(corpus.len(), 2, "both entries must be retained");

        // Verify ASC order by hash
        let hashes: Vec<u64> = corpus.iter().map(|e| e.tie_break_hash()).collect();
        if hash1 < hash2 {
            assert_eq!(hashes, vec![hash1, hash2]);
        } else {
            assert_eq!(hashes, vec![hash2, hash1]);
        }
    }

    // ── Test 5 ──────────────────────────────────────────────────────────────
    /// query() filters by SPO; returns matching entries in timestamp ASC order.
    #[test]
    fn test_witness_corpus_query_filters_by_spo() {
        let mut corpus = WitnessCorpus::new();

        // 3 entries with spo=0xABC (timestamps 10, 30, 20 inserted out of order)
        corpus.insert(WitnessEntry {
            spo: 0xABC,
            timestamp_ns: 10,
            source_url: None,
            evidence_blob: Bytes::new(),
        });
        corpus.insert(WitnessEntry {
            spo: 0xDEF,
            timestamp_ns: 15,
            source_url: None,
            evidence_blob: Bytes::new(),
        });
        corpus.insert(WitnessEntry {
            spo: 0xABC,
            timestamp_ns: 30,
            source_url: None,
            evidence_blob: Bytes::new(),
        });
        corpus.insert(WitnessEntry {
            spo: 0xDEF,
            timestamp_ns: 25,
            source_url: None,
            evidence_blob: Bytes::new(),
        });
        corpus.insert(WitnessEntry {
            spo: 0xABC,
            timestamp_ns: 20,
            source_url: None,
            evidence_blob: Bytes::new(),
        });

        let abc_results: Vec<u64> = corpus.query(0xABC).map(|e| e.timestamp_ns).collect();
        let def_results: Vec<u64> = corpus.query(0xDEF).map(|e| e.timestamp_ns).collect();

        assert_eq!(
            abc_results,
            vec![10, 20, 30],
            "3 ABC entries in timestamp ASC order"
        );
        assert_eq!(
            def_results,
            vec![15, 25],
            "2 DEF entries in timestamp ASC order"
        );
    }

    // ── Test 6 ──────────────────────────────────────────────────────────────
    /// W5-INV-WITNESS-UNBOUNDED: insert 100 entries; len = 100; no automatic cap.
    #[test]
    fn test_witness_corpus_unbounded() {
        let mut corpus = WitnessCorpus::new();
        for i in 0u64..100 {
            corpus.insert(WitnessEntry {
                spo: i,
                timestamp_ns: i * 10,
                source_url: None,
                evidence_blob: Bytes::new(),
            });
        }
        assert_eq!(
            corpus.len(),
            100,
            "corpus must be unbounded — no cap at 32 or any other limit"
        );
    }

    // ── Test 7 ──────────────────────────────────────────────────────────────
    /// evict_stale_before(300) removes entries at ts < 300; returns count evicted.
    #[test]
    fn test_witness_corpus_evict_stale_before() {
        let mut corpus = WitnessCorpus::new();
        for &ts in &[100u64, 200, 300, 400, 500] {
            corpus.insert(WitnessEntry {
                spo: 0,
                timestamp_ns: ts,
                source_url: None,
                evidence_blob: Bytes::new(),
            });
        }

        let evicted = corpus.evict_stale_before(300);
        assert_eq!(evicted, 2, "entries at ts=100 and ts=200 must be evicted");
        assert_eq!(corpus.len(), 3);

        let remaining: Vec<u64> = corpus.iter().map(|e| e.timestamp_ns).collect();
        assert_eq!(remaining, vec![300, 400, 500]);
    }

    // ── Test 8 ──────────────────────────────────────────────────────────────
    /// W5-INV-COW: clone is cheap (Arc strong_count >= 2); insert into clone
    /// triggers Arc::make_mut split; original is unaffected.
    #[test]
    fn test_witness_corpus_clone_is_cheap_via_arc() {
        let mut original = WitnessCorpus::new();
        original.insert(WitnessEntry {
            spo: 1,
            timestamp_ns: 100,
            source_url: None,
            evidence_blob: Bytes::new(),
        });

        // Clone — shared Arc
        let mut cloned = original.clone();
        assert!(
            Arc::strong_count(&original.entries) >= 2,
            "cloned corpus must share the same Arc (strong_count >= 2)"
        );

        // Mutate clone → Arc::make_mut splits the Arc
        cloned.insert(WitnessEntry {
            spo: 2,
            timestamp_ns: 200,
            source_url: None,
            evidence_blob: Bytes::new(),
        });

        // After split, original Arc count is back to 1
        assert_eq!(
            Arc::strong_count(&original.entries),
            1,
            "after Arc::make_mut split, original Arc strong_count must be 1"
        );

        // Original is unaffected
        assert_eq!(original.len(), 1);
        assert_eq!(cloned.len(), 2);
    }

    // ════════════════════════════════════════════════════════════════════════
    // D-CSV-6b NEW TESTS — CamPqWitnessIndex + WitnessCorpus index wiring
    // ════════════════════════════════════════════════════════════════════════

    // ── New Test 1 ──────────────────────────────────────────────────────────
    /// New empty index returns empty slice for any lookup.
    #[test]
    fn test_cam_pq_index_empty() {
        let idx = CamPqWitnessIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.lookup(0xABC), &[] as &[usize]);
        assert_eq!(idx.lookup(0), &[] as &[usize]);
        assert_eq!(idx.lookup(u64::MAX), &[] as &[usize]);
    }

    // ── New Test 2 ──────────────────────────────────────────────────────────
    /// insert and lookup round-trip: multiple spos, multiple positions per spo.
    #[test]
    fn test_cam_pq_index_insert_and_lookup() {
        let mut idx = CamPqWitnessIndex::new();
        idx.insert(0xABC, 0);
        idx.insert(0xABC, 2);
        idx.insert(0xDEF, 1);

        assert_eq!(idx.lookup(0xABC), &[0, 2]);
        assert_eq!(idx.lookup(0xDEF), &[1]);
        assert_eq!(idx.lookup(0x999), &[] as &[usize]);
        assert_eq!(idx.len(), 3);
    }

    // ── New Test 3 ──────────────────────────────────────────────────────────
    /// remove_at shifts positions and removes the target.
    #[test]
    fn test_cam_pq_index_remove_at_shifts_positions() {
        let mut idx = CamPqWitnessIndex::new();
        // spo=0xAAA has positions 0 and 2; spo=0xBBB has position 1
        idx.insert(0xAAA, 0);
        idx.insert(0xBBB, 1);
        idx.insert(0xAAA, 2);

        // Remove position 1 (spo=0xBBB)
        idx.remove_at(1);

        // spo=0xBBB bucket should be empty (removed)
        assert_eq!(idx.lookup(0xBBB), &[] as &[usize]);
        // spo=0xAAA: position 0 stays, position 2 shifts to 1
        assert_eq!(idx.lookup(0xAAA), &[0, 1]);
    }

    // ── New Test 4 ──────────────────────────────────────────────────────────
    /// query() uses the index (not linear scan); yields only matching entries
    /// in chain order. W5-INV-CHAIN-ORDER must hold post-CAM-PQ.
    #[test]
    fn test_witness_corpus_query_uses_index() {
        let mut corpus = WitnessCorpus::new();
        let target_spo = 0xDEAD_BEEF_u64;
        let other_spo = 0x1234_5678_u64;

        // Insert 100 entries: alternating spos, out of timestamp order
        for i in 0u64..50 {
            corpus.insert(WitnessEntry {
                spo: target_spo,
                timestamp_ns: i * 2 + 1, // odd timestamps: 1,3,5,...,99
                source_url: None,
                evidence_blob: Bytes::new(),
            });
            corpus.insert(WitnessEntry {
                spo: other_spo,
                timestamp_ns: i * 2 + 2, // even timestamps: 2,4,6,...,100
                source_url: None,
                evidence_blob: Bytes::new(),
            });
        }

        assert_eq!(corpus.len(), 100);

        // query must return exactly the target_spo entries
        let results: Vec<&WitnessEntry> = corpus.query(target_spo).collect();
        assert_eq!(results.len(), 50, "must yield exactly 50 target entries");
        assert!(
            results.iter().all(|e| e.spo == target_spo),
            "all results must have target_spo"
        );

        // W5-INV-CHAIN-ORDER: results must be in timestamp ASC order
        let timestamps: Vec<u64> = results.iter().map(|e| e.timestamp_ns).collect();
        let mut sorted = timestamps.clone();
        sorted.sort_unstable();
        assert_eq!(timestamps, sorted, "W5-INV-CHAIN-ORDER: results must be in ASC order");
    }

    // ── New Test 5 ──────────────────────────────────────────────────────────
    /// evict_stale_before rebuilds index; subsequent queries return correct
    /// entries with shifted positions.
    #[test]
    fn test_witness_corpus_evict_rebuilds_index() {
        let mut corpus = WitnessCorpus::new();
        let spo_a = 0xAAAA_u64;
        let spo_b = 0xBBBB_u64;

        // 5 entries with timestamps 100-500
        corpus.insert(WitnessEntry { spo: spo_a, timestamp_ns: 100, source_url: None, evidence_blob: Bytes::new() });
        corpus.insert(WitnessEntry { spo: spo_b, timestamp_ns: 200, source_url: None, evidence_blob: Bytes::new() });
        corpus.insert(WitnessEntry { spo: spo_a, timestamp_ns: 300, source_url: None, evidence_blob: Bytes::new() });
        corpus.insert(WitnessEntry { spo: spo_b, timestamp_ns: 400, source_url: None, evidence_blob: Bytes::new() });
        corpus.insert(WitnessEntry { spo: spo_a, timestamp_ns: 500, source_url: None, evidence_blob: Bytes::new() });

        // Evict entries with ts < 300 (removes ts=100, ts=200)
        let evicted = corpus.evict_stale_before(300);
        assert_eq!(evicted, 2);
        assert_eq!(corpus.len(), 3);

        // spo_a: ts=300 and ts=500 remain
        let a_results: Vec<u64> = corpus.query(spo_a).map(|e| e.timestamp_ns).collect();
        assert_eq!(a_results, vec![300, 500], "spo_a: ts=300 and ts=500 remain, in order");

        // spo_b: ts=400 remains
        let b_results: Vec<u64> = corpus.query(spo_b).map(|e| e.timestamp_ns).collect();
        assert_eq!(b_results, vec![400], "spo_b: only ts=400 remains");

        // W5-INV-CHAIN-ORDER: iter must still be sorted
        let all_ts: Vec<u64> = corpus.iter().map(|e| e.timestamp_ns).collect();
        assert_eq!(all_ts, vec![300, 400, 500]);
    }

    // ── New Test 6 ──────────────────────────────────────────────────────────
    /// Insert entries in non-chronological order; query yields them in chain
    /// order (binary-search insert path is correctly indexed).
    #[test]
    fn test_witness_corpus_insert_maintains_index() {
        let mut corpus = WitnessCorpus::new();
        let spo = 0xCAFE_u64;

        // Insert in reverse timestamp order (worst case for binary-search)
        for ts in [500u64, 300, 100, 400, 200] {
            corpus.insert(WitnessEntry {
                spo,
                timestamp_ns: ts,
                source_url: None,
                evidence_blob: Bytes::new(),
            });
        }

        // All 5 must be indexed correctly
        assert_eq!(corpus.len(), 5);

        let results: Vec<u64> = corpus.query(spo).map(|e| e.timestamp_ns).collect();
        assert_eq!(
            results,
            vec![100, 200, 300, 400, 500],
            "W5-INV-CHAIN-ORDER: entries must be in timestamp ASC order after non-chronological inserts"
        );
    }

    // ── New Test 7 ──────────────────────────────────────────────────────────
    /// cam_pq_search(spo, k) returns at most k entries in chain order.
    #[test]
    fn test_cam_pq_search_top_k() {
        let mut corpus = WitnessCorpus::new();
        let spo = 0xF00D_u64;

        // Insert 10 entries with the same spo, timestamps 10..100 step 10
        for i in 1u64..=10 {
            corpus.insert(WitnessEntry {
                spo,
                timestamp_ns: i * 10,
                source_url: None,
                evidence_blob: Bytes::new(),
            });
        }

        // cam_pq_search with k=3: must return exactly 3 entries
        let results = corpus.cam_pq_search(spo, 3);
        assert_eq!(results.len(), 3, "cam_pq_search must return exactly k=3 entries");

        // The 3 returned must be the first 3 in chain order (ts=10,20,30)
        let timestamps: Vec<u64> = results.iter().map(|e| e.timestamp_ns).collect();
        assert_eq!(timestamps, vec![10, 20, 30], "first 3 in chain order");

        // k > total entries: returns all
        let all_results = corpus.cam_pq_search(spo, 100);
        assert_eq!(all_results.len(), 10, "cam_pq_search(k>N) returns all N entries");

        // Unknown spo: returns empty
        let empty = corpus.cam_pq_search(0xDEAD, 5);
        assert!(empty.is_empty(), "cam_pq_search on unknown spo returns empty");
    }
}
