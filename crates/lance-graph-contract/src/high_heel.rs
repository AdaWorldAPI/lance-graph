//! HighHeelBGZ — 2KB container for family basin nodes in AriGraph.
//!
//! One container = one family basin (TWIG-level centroid) + up to 240 causal edges.
//!
//! ```text
//! HEEL (128 bytes = W0-W15):
//!   W0:      DN address (u64)
//!   W1:      Label hash (u32) + flags (u32)
//!   W2-W14:  SpoBase17 (102 bytes + 2 pad) — the family basin vector
//!   W15:     NARS truth (u8 freq, u8 conf, u8 scent, u8 plasticity, u32 temporal)
//!
//! EDGES (1920 bytes = W16-W255):
//!   W16-W255: up to 240 × CausalEdge64 (8 bytes each)
//! ```
//!
//! HHTL cascade mapping:
//!   HEEL (1 byte scent in W15)   → 95% rejection pre-filter
//!   HIP  (3 bytes palette in W1) → CAKES metric-safe pruning
//!   TWIG (102 bytes SpoBase17)   → family basin L1 distance
//!   LEAF (full 16Kbit planes)    → computed on demand, not stored
//!
//! For streaming: triplets accumulate into basins. Related triplets
//! (Base17 L1 < threshold) merge into the same basin. Edges represent
//! inter-basin causal structure. 240 edges is sufficient for episodic context.
//!
//! For fulltext distillation: chain N × 2KB containers for long documents.
//! Each container is one basin (paragraph cluster), edges link basins across
//! the document. A book = Vec<HighHeelBGZ> where each container is a
//! thematic basin with causal edges to related basins.
//!
//! Zero dependencies. Pure data types.

/// Size of the container in u64 words.
pub const CONTAINER_WORDS: usize = 256;
/// Size of the container in bytes.
pub const CONTAINER_BYTES: usize = CONTAINER_WORDS * 8; // 2048
/// Heel size in words (metadata region).
pub const HEEL_WORDS: usize = 16;
/// Maximum number of CausalEdge64 entries.
pub const MAX_EDGES: usize = CONTAINER_WORDS - HEEL_WORDS; // 240

/// SpoBase17 — three Base17 planes (S, P, O) packed as 102 bytes.
///
/// Each plane is 17 × i16 = 34 bytes. Three planes = 102 bytes.
/// This is the family basin centroid at TWIG level (ρ=0.965 vs full planes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SpoBase17 {
    /// Subject plane: 17 dimensions.
    pub s: [i16; 17],
    /// Predicate plane: 17 dimensions.
    pub p: [i16; 17],
    /// Object plane: 17 dimensions.
    pub o: [i16; 17],
}

impl SpoBase17 {
    pub const ZERO: Self = Self {
        s: [0i16; 17],
        p: [0i16; 17],
        o: [0i16; 17],
    };

    /// L1 distance between two SpoBase17 vectors (all 3 planes).
    pub fn l1_distance(&self, other: &Self) -> u32 {
        let mut d = 0u32;
        for i in 0..17 {
            d += (self.s[i] as i32 - other.s[i] as i32).unsigned_abs();
            d += (self.p[i] as i32 - other.p[i] as i32).unsigned_abs();
            d += (self.o[i] as i32 - other.o[i] as i32).unsigned_abs();
        }
        d
    }

    /// L1 distance on subject plane only.
    pub fn l1_subject(&self, other: &Self) -> u32 {
        let mut d = 0u32;
        for i in 0..17 { d += (self.s[i] as i32 - other.s[i] as i32).unsigned_abs(); }
        d
    }

    /// Scent byte: 7-bit Boolean lattice of plane proximity.
    /// Bit 0: S close, 1: P close, 2: O close,
    /// 3: SP, 4: SO, 5: PO, 6: SPO.
    pub fn scent(&self, other: &Self, threshold: u32) -> u8 {
        let ds = self.l1_subject(other);
        let dp = {
            let mut d = 0u32;
            for i in 0..17 { d += (self.p[i] as i32 - other.p[i] as i32).unsigned_abs(); }
            d
        };
        let do_ = {
            let mut d = 0u32;
            for i in 0..17 { d += (self.o[i] as i32 - other.o[i] as i32).unsigned_abs(); }
            d
        };
        let s_close = ds < threshold;
        let p_close = dp < threshold;
        let o_close = do_ < threshold;
        let mut b = 0u8;
        if s_close { b |= 1; }
        if p_close { b |= 2; }
        if o_close { b |= 4; }
        if s_close && p_close { b |= 8; }
        if s_close && o_close { b |= 16; }
        if p_close && o_close { b |= 32; }
        if s_close && p_close && o_close { b |= 64; }
        b
    }
}

/// Heel metadata — the identity + content of a family basin.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Heel {
    /// DN address (W0): identity in the BindSpace.
    pub dn_address: u64,
    /// Label hash (lower 32 bits) + flags (upper 32 bits) (W1).
    /// Flags: bits 0-2 = palette_s, 3-5 = palette_p, 6-8 = palette_o (HIP level).
    pub label_flags: u64,
    /// The family basin vector (W2-W14, 102 bytes + 2 pad).
    pub spo: SpoBase17,
    /// NARS truth + scent + plasticity + temporal (W15).
    /// Byte 0: frequency (u8, 0-255 → 0.0-1.0)
    /// Byte 1: confidence (u8, 0-255 → 0.0-1.0)
    /// Byte 2: scent (u8, 7-bit Boolean lattice)
    /// Byte 3: plasticity (u8, 0=frozen, 1=cooling, 2=warm, 3=hot)
    /// Bytes 4-7: temporal index (u32, basin creation step)
    pub truth_meta: u64,
}

impl Heel {
    /// Extract NARS frequency [0.0, 1.0].
    pub fn frequency(&self) -> f32 {
        (self.truth_meta & 0xFF) as f32 / 255.0
    }

    /// Extract NARS confidence [0.0, 1.0].
    pub fn confidence(&self) -> f32 {
        ((self.truth_meta >> 8) & 0xFF) as f32 / 255.0
    }

    /// Extract scent byte.
    pub fn scent(&self) -> u8 {
        ((self.truth_meta >> 16) & 0xFF) as u8
    }

    /// Extract plasticity state (0=frozen..3=hot).
    pub fn plasticity(&self) -> u8 {
        ((self.truth_meta >> 24) & 0xFF) as u8
    }

    /// Extract temporal index.
    pub fn temporal(&self) -> u32 {
        (self.truth_meta >> 32) as u32
    }

    /// Pack truth_meta from components.
    pub fn pack_truth_meta(freq: f32, conf: f32, scent: u8, plasticity: u8, temporal: u32) -> u64 {
        let f = (freq.clamp(0.0, 1.0) * 255.0) as u64;
        let c = (conf.clamp(0.0, 1.0) * 255.0) as u64;
        f | (c << 8) | ((scent as u64) << 16) | ((plasticity as u64) << 24) | ((temporal as u64) << 32)
    }
}

/// HighHeelBGZ — 2KB container: one family basin + up to 240 causal edges.
///
/// The raw backing store is `[u64; 256]`.
/// W0-W15 = Heel (identity + SpoBase17 basin vector + NARS truth).
/// W16-W255 = CausalEdge64 edges (each is one u64).
///
/// For fulltext distillation, chain containers: `Vec<HighHeelBGZ>`.
/// Each container is a thematic basin, edges link across basins.
#[derive(Clone, Debug)]
pub struct HighHeelBGZ {
    /// The heel: identity, basin vector, truth.
    pub heel: Heel,
    /// Causal edges (up to 240). Each is a packed u64.
    /// Format: CausalEdge64 bit layout (S/P/O palette + NARS + Pearl mask + inference + plasticity + temporal).
    pub edges: Vec<u64>,
}

impl HighHeelBGZ {
    /// Create an empty container with the given DN address and basin vector.
    pub fn new(dn_address: u64, spo: SpoBase17) -> Self {
        Self {
            heel: Heel {
                dn_address,
                label_flags: 0,
                spo,
                truth_meta: Heel::pack_truth_meta(0.5, 0.1, 0, 3, 0), // weak prior, hot plasticity
            },
            edges: Vec::new(),
        }
    }

    /// Add a causal edge. Returns false if at capacity (240).
    pub fn add_edge(&mut self, edge: u64) -> bool {
        if self.edges.len() >= MAX_EDGES {
            return false;
        }
        self.edges.push(edge);
        true
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Is this basin frozen (plasticity = 0, truth stable)?
    pub fn is_crystallized(&self) -> bool {
        self.heel.plasticity() == 0 && self.heel.confidence() > 0.8
    }

    /// Update NARS truth via revision (new evidence).
    pub fn revise_truth(&mut self, new_freq: f32, new_conf: f32) {
        let old_f = self.heel.frequency();
        let old_c = self.heel.confidence();
        // NARS revision: weighted average by confidence
        let w1 = old_c;
        let w2 = new_conf;
        let total = w1 + w2;
        if total < 1e-6 { return; }
        let merged_f = (old_f * w1 + new_freq * w2) / total;
        let merged_c = (total / (total + 1.0)).min(0.99); // confidence approaches but never reaches 1.0
        // Cool plasticity as confidence rises
        let plasticity = if merged_c > 0.8 { 0 }      // frozen
            else if merged_c > 0.6 { 1 }               // cooling
            else if merged_c > 0.3 { 2 }               // warm
            else { 3 };                                 // hot
        self.heel.truth_meta = Heel::pack_truth_meta(
            merged_f, merged_c,
            self.heel.scent(), plasticity,
            self.heel.temporal(),
        );
    }

    /// Pack to raw `[u64; 256]` wire format.
    pub fn pack(&self) -> [u64; 256] {
        let mut buf = [0u64; 256];
        buf[0] = self.heel.dn_address;
        buf[1] = self.heel.label_flags;
        // Pack SpoBase17 into W2-W14 (102 bytes → 13 words)
        let spo_bytes = spo_to_bytes(&self.heel.spo);
        let mut word_idx = 2;
        let mut byte_idx = 0;
        while byte_idx + 8 <= 104 && word_idx < 15 {
            let mut w = [0u8; 8];
            let end = (byte_idx + 8).min(spo_bytes.len());
            let len = end - byte_idx;
            w[..len].copy_from_slice(&spo_bytes[byte_idx..end]);
            buf[word_idx] = u64::from_le_bytes(w);
            word_idx += 1;
            byte_idx += 8;
        }
        buf[15] = self.heel.truth_meta;
        // Pack edges
        for (i, &edge) in self.edges.iter().enumerate() {
            if i >= MAX_EDGES { break; }
            buf[16 + i] = edge;
        }
        buf
    }

    /// Unpack from raw `[u64; 256]` wire format.
    pub fn unpack(buf: &[u64; 256]) -> Self {
        let dn_address = buf[0];
        let label_flags = buf[1];
        // Unpack SpoBase17 from W2-W14
        let mut spo_bytes = [0u8; 104]; // 102 + 2 pad
        for i in 0..13 {
            let w = buf[2 + i].to_le_bytes();
            let start = i * 8;
            let end = (start + 8).min(104);
            spo_bytes[start..end].copy_from_slice(&w[..end - start]);
        }
        let spo = bytes_to_spo(&spo_bytes);
        let truth_meta = buf[15];
        // Unpack edges (non-zero entries in W16-W255)
        let mut edges = Vec::new();
        for i in 16..256 {
            if buf[i] != 0 {
                edges.push(buf[i]);
            }
        }
        Self {
            heel: Heel { dn_address, label_flags, spo, truth_meta },
            edges,
        }
    }

    /// Total byte size on wire.
    pub const fn wire_size() -> usize { CONTAINER_BYTES }
}

/// Convert SpoBase17 to 102 bytes (+ 2 padding = 104).
fn spo_to_bytes(spo: &SpoBase17) -> [u8; 104] {
    let mut out = [0u8; 104];
    for i in 0..17 {
        let b = spo.s[i].to_le_bytes();
        out[i * 2] = b[0];
        out[i * 2 + 1] = b[1];
    }
    for i in 0..17 {
        let b = spo.p[i].to_le_bytes();
        out[34 + i * 2] = b[0];
        out[34 + i * 2 + 1] = b[1];
    }
    for i in 0..17 {
        let b = spo.o[i].to_le_bytes();
        out[68 + i * 2] = b[0];
        out[68 + i * 2 + 1] = b[1];
    }
    out
}

/// Convert 104 bytes back to SpoBase17.
fn bytes_to_spo(bytes: &[u8; 104]) -> SpoBase17 {
    let mut spo = SpoBase17::ZERO;
    for i in 0..17 {
        spo.s[i] = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
    }
    for i in 0..17 {
        spo.p[i] = i16::from_le_bytes([bytes[34 + i * 2], bytes[34 + i * 2 + 1]]);
    }
    for i in 0..17 {
        spo.o[i] = i16::from_le_bytes([bytes[68 + i * 2], bytes[68 + i * 2 + 1]]);
    }
    spo
}

// ═══════════════════════════════════════════════════════════════════════════
// BASIN ACCUMULATOR — streaming triplets into family basins
// ═══════════════════════════════════════════════════════════════════════════

/// Basin merge threshold: if L1 distance < this, triplets belong to same basin.
pub const DEFAULT_BASIN_THRESHOLD: u32 = 2000;

/// Streaming basin accumulator.
///
/// Receives SpoBase17 triplets and merges them into family basins.
/// Each basin is a HighHeelBGZ container. New triplets either join an
/// existing basin (L1 < threshold) or create a new one.
///
/// For fulltext distillation (e.g., Rumi, Tagore via Gutenberg):
/// stream paragraphs → embed as SpoBase17 → accumulate into basins →
/// Vec<HighHeelBGZ> = the distilled document.
pub struct BasinAccumulator {
    /// Active basins.
    pub basins: Vec<HighHeelBGZ>,
    /// Merge threshold (L1 distance).
    pub threshold: u32,
    /// Next DN address to assign.
    next_dn: u64,
    /// Total triplets ingested.
    pub ingested: u64,
    /// Total basins created.
    pub basins_created: u64,
    /// Total merges (triplet joined existing basin).
    pub merges: u64,
}

impl BasinAccumulator {
    pub fn new(threshold: u32) -> Self {
        Self {
            basins: Vec::new(),
            threshold,
            next_dn: 1,
            ingested: 0,
            basins_created: 0,
            merges: 0,
        }
    }

    /// Ingest a triplet. Merges into nearest basin or creates new one.
    /// Returns the basin index.
    pub fn ingest(&mut self, spo: SpoBase17, edge: u64) -> usize {
        self.ingested += 1;
        // Find nearest basin
        let mut best_idx = None;
        let mut best_dist = u32::MAX;
        for (i, basin) in self.basins.iter().enumerate() {
            let d = basin.heel.spo.l1_distance(&spo);
            if d < best_dist {
                best_dist = d;
                best_idx = Some(i);
            }
        }
        if best_dist < self.threshold {
            let idx = best_idx.unwrap();
            self.basins[idx].add_edge(edge);
            // Revise truth: more evidence → higher confidence
            let freq = self.basins[idx].heel.frequency();
            self.basins[idx].revise_truth(freq, 0.3); // each merge adds 0.3 confidence evidence
            self.merges += 1;
            idx
        } else {
            // New basin
            let dn = self.next_dn;
            self.next_dn += 1;
            let mut basin = HighHeelBGZ::new(dn, spo);
            basin.add_edge(edge);
            self.basins.push(basin);
            self.basins_created += 1;
            self.basins.len() - 1
        }
    }

    /// Get monitoring snapshot.
    pub fn stats(&self) -> BasinStats {
        let crystallized = self.basins.iter().filter(|b| b.is_crystallized()).count();
        let total_edges: usize = self.basins.iter().map(|b| b.edge_count()).sum();
        let avg_edges = if self.basins.is_empty() { 0.0 }
            else { total_edges as f32 / self.basins.len() as f32 };
        BasinStats {
            basin_count: self.basins.len(),
            total_ingested: self.ingested,
            total_merges: self.merges,
            total_edges,
            avg_edges_per_basin: avg_edges,
            crystallized_count: crystallized,
            merge_ratio: if self.ingested == 0 { 0.0 }
                else { self.merges as f32 / self.ingested as f32 },
        }
    }
}

/// Monitoring snapshot for cognitive debugging.
#[derive(Debug, Clone)]
pub struct BasinStats {
    /// Number of active basins.
    pub basin_count: usize,
    /// Total triplets ingested.
    pub total_ingested: u64,
    /// Total merges (triplet → existing basin).
    pub total_merges: u64,
    /// Total edges across all basins.
    pub total_edges: usize,
    /// Average edges per basin.
    pub avg_edges_per_basin: f32,
    /// Basins that have crystallized (frozen, high confidence).
    pub crystallized_count: usize,
    /// Merge ratio: merges / ingested (higher = more consolidation).
    pub merge_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spo(s0: i16, p0: i16, o0: i16) -> SpoBase17 {
        let mut spo = SpoBase17::ZERO;
        spo.s[0] = s0; spo.p[0] = p0; spo.o[0] = o0;
        spo
    }

    #[test]
    fn test_spo_l1_distance() {
        let a = make_spo(100, 200, 300);
        let b = make_spo(110, 200, 300);
        assert_eq!(a.l1_distance(&b), 10); // only s[0] differs by 10
    }

    #[test]
    fn test_spo_self_distance_zero() {
        let a = make_spo(100, 200, 300);
        assert_eq!(a.l1_distance(&a), 0);
    }

    #[test]
    fn test_scent_all_close() {
        let a = make_spo(100, 200, 300);
        let b = make_spo(105, 205, 305);
        let s = a.scent(&b, 1000);
        assert_eq!(s & 0x7F, 0x7F); // all 7 bits set
    }

    #[test]
    fn test_scent_none_close() {
        let a = make_spo(0, 0, 0);
        let b = make_spo(10000, 10000, 10000);
        let s = a.scent(&b, 100);
        assert_eq!(s, 0);
    }

    #[test]
    fn test_heel_truth_pack_unpack() {
        let meta = Heel::pack_truth_meta(0.9, 0.7, 0x3F, 2, 42);
        let heel = Heel {
            dn_address: 1, label_flags: 0,
            spo: SpoBase17::ZERO, truth_meta: meta,
        };
        assert!((heel.frequency() - 0.9).abs() < 0.01);
        assert!((heel.confidence() - 0.7).abs() < 0.01);
        assert_eq!(heel.scent(), 0x3F);
        assert_eq!(heel.plasticity(), 2);
        assert_eq!(heel.temporal(), 42);
    }

    #[test]
    fn test_container_pack_roundtrip() {
        let spo = make_spo(1234, -5678, 9012);
        let mut c = HighHeelBGZ::new(42, spo);
        c.add_edge(0xDEAD_BEEF_CAFE_BABE);
        c.add_edge(0x1234_5678_9ABC_DEF0);
        let buf = c.pack();
        let c2 = HighHeelBGZ::unpack(&buf);
        assert_eq!(c2.heel.dn_address, 42);
        assert_eq!(c2.heel.spo, spo);
        assert_eq!(c2.edges.len(), 2);
        assert_eq!(c2.edges[0], 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(c2.edges[1], 0x1234_5678_9ABC_DEF0);
    }

    #[test]
    fn test_max_edges() {
        let mut c = HighHeelBGZ::new(1, SpoBase17::ZERO);
        for i in 0..MAX_EDGES {
            assert!(c.add_edge(i as u64 + 1));
        }
        assert!(!c.add_edge(999)); // 241st fails
        assert_eq!(c.edge_count(), MAX_EDGES);
    }

    #[test]
    fn test_revise_truth_increases_confidence() {
        let mut c = HighHeelBGZ::new(1, SpoBase17::ZERO);
        let c0 = c.heel.confidence();
        c.revise_truth(0.8, 0.5);
        let c1 = c.heel.confidence();
        assert!(c1 > c0, "confidence should increase with evidence");
    }

    #[test]
    fn test_crystallization() {
        let mut c = HighHeelBGZ::new(1, SpoBase17::ZERO);
        assert!(!c.is_crystallized());
        // Force high confidence + frozen plasticity
        c.heel.truth_meta = Heel::pack_truth_meta(0.9, 0.95, 0, 0, 0);
        assert!(c.is_crystallized());
    }

    #[test]
    fn test_basin_accumulator_merge() {
        let mut acc = BasinAccumulator::new(500);
        // Two similar triplets → same basin
        let spo1 = make_spo(100, 200, 300);
        let spo2 = make_spo(105, 205, 305); // L1 = 15, well under 500
        acc.ingest(spo1, 0x1111);
        acc.ingest(spo2, 0x2222);
        assert_eq!(acc.basins.len(), 1); // merged
        assert_eq!(acc.basins[0].edge_count(), 2);
        let stats = acc.stats();
        assert_eq!(stats.total_merges, 1);
        assert_eq!(stats.total_ingested, 2);
    }

    #[test]
    fn test_basin_accumulator_split() {
        let mut acc = BasinAccumulator::new(500);
        // Two distant triplets → different basins
        let spo1 = make_spo(100, 200, 300);
        let spo2 = make_spo(10000, 20000, 30000); // very far
        acc.ingest(spo1, 0x1111);
        acc.ingest(spo2, 0x2222);
        assert_eq!(acc.basins.len(), 2); // split
    }

    #[test]
    fn test_basin_stats_monitoring() {
        let mut acc = BasinAccumulator::new(1000);
        for i in 0..10 {
            let spo = make_spo(i * 5, i * 5, i * 5); // close together
            acc.ingest(spo, i as u64);
        }
        let stats = acc.stats();
        assert!(stats.merge_ratio > 0.5, "most should merge at threshold=1000");
        assert!(stats.basin_count < 10, "should consolidate into fewer basins");
    }

    #[test]
    fn test_wire_size() {
        assert_eq!(HighHeelBGZ::wire_size(), 2048);
    }
}
