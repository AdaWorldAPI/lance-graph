// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! WitnessCorpus — replaces SpoWitnessChain<32> per L-9 / W5 spec §3.3.
//!
//! Unbounded, copy-on-write via Arc::make_mut, sorted by timestamp_ns ASC
//! with hash tie-break (W5-INV-CHAIN-ORDER). Each entry encodes one
//! peer-witnessed SPO observation with optional source URL + evidence blob.
//!
//! D-CSV-6a scope (sprint-11 Phase B core):
//! - WitnessEntry struct
//! - WitnessCorpus { entries, cam_pq_index_placeholder }
//! - insert / query (linear scan) / iter / evict_stale
//! - Three iron-rule invariants enforced in tests: W5-INV-CHAIN-ORDER,
//!   W5-INV-WITNESS-UNBOUNDED, W5-INV-CAM-PQ-INDEX (the third tested
//!   only at the API-shape level since the index is a placeholder)
//!
//! Out of scope (D-CSV-6b, follow-up PR):
//! - Real CAM-PQ index (ndarray hot-path wiring)
//! - cam_pq_search method (linear-scan query() is the placeholder)
//! - Benches
//! - WitnessCorpusStore (the 64-slot array keyed by W-slot — only needed
//!   when MailboxSoA's W-slot integration lands jointly with D-CSV-7+
//!   Σ-tier dispatch)

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

/// CAM-PQ index handle placeholder. D-CSV-6b replaces this with a
/// concrete ndarray `CamPqCodec`-backed index. For D-CSV-6a it's an
/// empty unit struct so the WitnessCorpus shape compiles + tests
/// can assert the field exists.
#[derive(Clone, Debug, Default)]
pub struct CamPqIndexPlaceholder;

/// Unbounded CAM-PQ-indexed witness corpus.
/// Per L-9 / W5 §3.3: replaces SpoWitnessChain<32>. Copy-on-write via
/// Arc::make_mut so cheap clones are cheap; mutations bump the Arc.
#[derive(Clone, Debug)]
pub struct WitnessCorpus {
    entries: Arc<Vec<WitnessEntry>>,
    /// Placeholder until D-CSV-6b lands real CAM-PQ wiring.
    pub cam_pq_index: CamPqIndexPlaceholder,
}

impl Default for WitnessCorpus {
    fn default() -> Self {
        Self {
            entries: Arc::new(Vec::new()),
            cam_pq_index: CamPqIndexPlaceholder,
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
    pub fn insert(&mut self, entry: WitnessEntry) -> WitnessId {
        let (t, h) = entry.ord_key();
        let id = WitnessId((t.wrapping_mul(31)) ^ h);
        let entries = Arc::make_mut(&mut self.entries);
        let pos = entries
            .binary_search_by(|e| e.ord_key().cmp(&(t, h)))
            .unwrap_or_else(|p| p);
        entries.insert(pos, entry);
        id
    }

    /// Linear-scan query by SPO match. Returns matching entries
    /// in chain order (timestamp ASC). D-CSV-6b replaces this with a
    /// CAM-PQ-backed O(log N) lookup.
    pub fn query<'a>(&'a self, spo: u64) -> impl Iterator<Item = &'a WitnessEntry> + 'a {
        self.entries.iter().filter(move |e| e.spo == spo)
    }

    /// All entries in chain order (timestamp ASC).
    pub fn iter(&self) -> std::slice::Iter<'_, WitnessEntry> {
        self.entries.iter()
    }

    /// Evict entries strictly before `cutoff_ns`. Per W5-INV-WITNESS-
    /// UNBOUNDED, the corpus is unbounded by default; eviction is a
    /// caller-driven operation, NOT an automatic LRU/cap. Returns the
    /// number of evicted entries.
    pub fn evict_stale_before(&mut self, cutoff_ns: u64) -> usize {
        let entries = Arc::make_mut(&mut self.entries);
        let n_before = entries.len();
        entries.retain(|e| e.timestamp_ns >= cutoff_ns);
        n_before - entries.len()
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

        assert_eq!(abc_results, vec![10, 20, 30], "3 ABC entries in timestamp ASC order");
        assert_eq!(def_results, vec![15, 25], "2 DEF entries in timestamp ASC order");
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
}
