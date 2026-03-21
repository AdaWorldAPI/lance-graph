//! Container 0 integration: pack/unpack bgz17 data inside the `[u64; 256]` BitVec.
//!
//! The cognitive node container is 256 words × 64 bits = 2KB = `FixedSizeBinary(2048)`.
//! bgz17 does NOT replace the container — it provides a COMPRESSED VIEW of it.
//!
//! ## Word Layout (bgz17-relevant regions)
//!
//! ```text
//! W112-124  Base17 S+P+O annex   (13 words = 104 bytes, uses 102)
//! W125      palette_s(8) | palette_p(8) | palette_o(8) | temporal_q(8) | spare(32)
//! W126-127  checksum + version   (integrity, written by container layer)
//!
//! WIDE META (CogRecord8K only, W128-255):
//! W128-143  SPO Crystal          (16 words = 8 packed SPO triples)
//! W224-239  Extended edge overflow(16 words = 64 more edges, 128 total with W16-31)
//! W254      Wide checksum
//! W255      Embedding format tag
//! ```
//!
//! ## Cascade Sampling Insight
//!
//! `words_hamming_sampled(query, candidate, stride=16)` hits W0, W16, W32, ..., W112, ...
//! W112 = first word of Base17 annex. Filling W112-125 with Base17 data makes the
//! Cascade's stage-1 sample MORE discriminating with ZERO code change to the Cascade.

use crate::base17::{Base17, SpoBase17};
use crate::palette::PaletteEdge;
use crate::BASE_DIM;

// ─── Container geometry ───────────────────────────────────────────────

/// Total words in Container 0.
pub const CONTAINER_WORDS: usize = 256;

/// Total bytes in Container 0.
pub const CONTAINER_BYTES: usize = CONTAINER_WORDS * 8; // 2048

// ─── Word offsets (from the spec) ─────────────────────────────────────

/// DN address (identity).
pub const W_DN_ADDR: usize = 0;
/// node_kind | flags | schema_version | provenance_hash.
pub const W_NODE_META: usize = 1;
/// created_ms | modified_ms.
pub const W_TEMPORAL: usize = 2;
/// label_hash | tree_depth.
pub const W_TREE_POS: usize = 3;

/// NARS truth values: frequency, confidence, evidence weight, horizon.
pub const W_NARS_START: usize = 4;
pub const W_NARS_END: usize = 7; // inclusive

/// DN rung | gate_state.
pub const W_RUNG_START: usize = 8;
/// 7-layer cognitive markers.
pub const W_LAYER_MARKERS_END: usize = 15; // inclusive

/// Inline edges: 16 words, 64 edges max (verb:8 + target:8 per edge = 4 per word).
pub const W_INLINE_EDGES_START: usize = 16;
pub const W_INLINE_EDGES_END: usize = 31; // inclusive
/// Maximum inline edges in core container.
pub const MAX_INLINE_EDGES: usize = 64;

/// RL / Q-values / rewards.
pub const W_RL_START: usize = 32;
pub const W_RL_END: usize = 39;

/// Bloom filter (512 bits).
pub const W_BLOOM_START: usize = 40;
pub const W_BLOOM_END: usize = 47;

/// Graph metrics (degree, centrality, etc.).
pub const W_GRAPH_METRICS_START: usize = 48;
pub const W_GRAPH_METRICS_END: usize = 55;

/// Qualia: 18 channels × f16 + 8 spare slots.
pub const W_QUALIA_START: usize = 56;
pub const W_QUALIA_END: usize = 63;

/// Rung history + collapse gate.
pub const W_RUNG_HISTORY_START: usize = 64;
pub const W_RUNG_HISTORY_END: usize = 79;

/// Representation descriptor.
pub const W_REPR_DESC_START: usize = 80;
pub const W_REPR_DESC_END: usize = 95;

/// DN-Sparse adjacency (RESERVED — future palette CSR).
pub const W_DN_SPARSE_START: usize = 96;
pub const W_DN_SPARSE_END: usize = 111;

/// ★ bgz17 ANNEX: Base17 S+P+O (102 bytes) packed into 13 words.
pub const W_BGZ17_ANNEX_START: usize = 112;
pub const W_BGZ17_ANNEX_END: usize = 124; // inclusive, 13 words = 104 bytes

/// Palette indices + temporal quantile packed into 1 word.
/// Layout: palette_s(8) | palette_p(8) | palette_o(8) | temporal_q(8) | spare(32)
pub const W_PALETTE_PACK: usize = 125;

/// Checksum + version (integrity).
pub const W_CHECKSUM: usize = 126;
pub const W_VERSION: usize = 127;

// ─── Wide meta (CogRecord8K, W128-255) ───────────────────────────────

/// SPO Crystal: 8 packed triples (PRIO 0).
pub const W_SPO_CRYSTAL_START: usize = 128;
pub const W_SPO_CRYSTAL_END: usize = 143;

/// Extended edge overflow: 64 more edges (PRIO 0).
pub const W_EXT_EDGES_START: usize = 224;
pub const W_EXT_EDGES_END: usize = 239;
/// Maximum edges with overflow (core 64 + extended 64).
pub const MAX_TOTAL_EDGES: usize = 128;

/// Wide checksum (PRIO 0).
pub const W_WIDE_CHECKSUM: usize = 254;
/// Embedding format tag (PRIO 0).
pub const W_FORMAT_TAG: usize = 255;

// ─── Format tag values ────────────────────────────────────────────────

/// Format tag indicating bgz17 annex is populated in W112-125.
pub const FORMAT_BGZ17_ANNEX: u64 = 0x6267_7A31_375F_7631; // "bgz17_v1" as ASCII

/// Format tag indicating CogRecord8K wide meta is populated.
pub const FORMAT_COGREC_8K: u64 = 0x636F_6752_6563_384B; // "cogRec8K" as ASCII

// ─── Annex packing (W112-125): Base17 S+P+O + palette indices ────────

/// Pack an SpoBase17 (102 bytes) into container words W112-124 (104 bytes).
///
/// The 102 bytes are laid out as:
///   W112-115: base17_s (34 bytes, dims[0..16] as i16 LE)
///   W116-119: base17_p (34 bytes)
///   W120-123: base17_o (34 bytes)
///   W124[0..1]: remaining 2 bytes from base17_o (already included), pad with 0.
///
/// More precisely: 102 bytes = 12 full words (96 bytes) + 6 remaining bytes in W124.
/// W124 has its upper 2 bytes zeroed.
pub fn pack_base17_annex(container: &mut [u64; CONTAINER_WORDS], spo: &SpoBase17) {
    let bytes = spo_to_bytes(spo);
    // 102 bytes → 12 full u64 words + 6 bytes in word 13
    for i in 0..12 {
        let offset = i * 8;
        container[W_BGZ17_ANNEX_START + i] = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
    }
    // Word 13 (W124): 6 remaining bytes + 2 zero pad
    let offset = 96;
    container[W_BGZ17_ANNEX_START + 12] = u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        0,
        0,
    ]);
}

/// Unpack SpoBase17 from container words W112-124.
pub fn unpack_base17_annex(container: &[u64; CONTAINER_WORDS]) -> SpoBase17 {
    let mut bytes = [0u8; SpoBase17::BYTE_SIZE]; // 102
    for i in 0..12 {
        let word = container[W_BGZ17_ANNEX_START + i].to_le_bytes();
        let offset = i * 8;
        bytes[offset..offset + 8].copy_from_slice(&word);
    }
    // Word 13: 6 remaining bytes
    let word = container[W_BGZ17_ANNEX_START + 12].to_le_bytes();
    bytes[96..102].copy_from_slice(&word[..6]);

    bytes_to_spo(&bytes)
}

/// Pack palette indices + temporal quantile into W125.
///
/// Layout: `palette_s(8) | palette_p(8) | palette_o(8) | temporal_q(8) | spare(32)`
/// (little-endian: palette_s in lowest byte)
pub fn pack_palette_word(
    container: &mut [u64; CONTAINER_WORDS],
    edge: PaletteEdge,
    temporal_quantile: u8,
) {
    container[W_PALETTE_PACK] = (edge.s_idx as u64)
        | ((edge.p_idx as u64) << 8)
        | ((edge.o_idx as u64) << 16)
        | ((temporal_quantile as u64) << 24);
}

/// Unpack palette indices + temporal quantile from W125.
/// Returns `(PaletteEdge, temporal_quantile)`.
pub fn unpack_palette_word(container: &[u64; CONTAINER_WORDS]) -> (PaletteEdge, u8) {
    let w = container[W_PALETTE_PACK];
    let edge = PaletteEdge {
        s_idx: (w & 0xFF) as u8,
        p_idx: ((w >> 8) & 0xFF) as u8,
        o_idx: ((w >> 16) & 0xFF) as u8,
    };
    let temporal_q = ((w >> 24) & 0xFF) as u8;
    (edge, temporal_q)
}

/// Pack the full bgz17 annex (Base17 + palette + temporal) in one call.
pub fn pack_annex(
    container: &mut [u64; CONTAINER_WORDS],
    spo: &SpoBase17,
    edge: PaletteEdge,
    temporal_quantile: u8,
) {
    pack_base17_annex(container, spo);
    pack_palette_word(container, edge, temporal_quantile);
}

/// Unpack the full bgz17 annex in one call.
/// Returns `(SpoBase17, PaletteEdge, temporal_quantile)`.
pub fn unpack_annex(container: &[u64; CONTAINER_WORDS]) -> (SpoBase17, PaletteEdge, u8) {
    let spo = unpack_base17_annex(container);
    let (edge, tq) = unpack_palette_word(container);
    (spo, edge, tq)
}

// ─── CogRecord8K: SPO Crystal (W128-143) ─────────────────────────────

/// A packed SPO triple for the crystal region.
/// Each triple: subject_idx(u8) + predicate_idx(u8) + object_idx(u8) + flags(u8)
///            + distance(u16) + weight(u16) = 8 bytes.
/// 16 words = 128 bytes = 16 packed triples (spec says 8 but we can fit 16).
/// We follow the spec: 8 triples × 16 bytes each = 128 bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CrystalTriple {
    /// Subject palette index.
    pub s_idx: u8,
    /// Predicate palette index.
    pub p_idx: u8,
    /// Object palette index.
    pub o_idx: u8,
    /// Flags (e.g., edge type, confidence tier).
    pub flags: u8,
    /// Palette distance (scaled u16).
    pub distance: u16,
    /// Edge weight / importance (scaled u16).
    pub weight: u16,
    /// Source node index (local scope).
    pub source: u16,
    /// Target node index (local scope).
    pub target: u16,
    /// Reserved.
    pub reserved: u32,
}

impl CrystalTriple {
    pub const BYTE_SIZE: usize = 16;
    /// Maximum triples in the SPO crystal region (128 bytes / 16 bytes each).
    pub const MAX_CRYSTAL_TRIPLES: usize = 8;

    fn to_bytes(self) -> [u8; Self::BYTE_SIZE] {
        let mut buf = [0u8; Self::BYTE_SIZE];
        buf[0] = self.s_idx;
        buf[1] = self.p_idx;
        buf[2] = self.o_idx;
        buf[3] = self.flags;
        buf[4..6].copy_from_slice(&self.distance.to_le_bytes());
        buf[6..8].copy_from_slice(&self.weight.to_le_bytes());
        buf[8..10].copy_from_slice(&self.source.to_le_bytes());
        buf[10..12].copy_from_slice(&self.target.to_le_bytes());
        buf[12..16].copy_from_slice(&self.reserved.to_le_bytes());
        buf
    }

    fn from_bytes(buf: &[u8; Self::BYTE_SIZE]) -> Self {
        CrystalTriple {
            s_idx: buf[0],
            p_idx: buf[1],
            o_idx: buf[2],
            flags: buf[3],
            distance: u16::from_le_bytes([buf[4], buf[5]]),
            weight: u16::from_le_bytes([buf[6], buf[7]]),
            source: u16::from_le_bytes([buf[8], buf[9]]),
            target: u16::from_le_bytes([buf[10], buf[11]]),
            reserved: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
        }
    }
}

/// Pack up to 8 SPO crystal triples into W128-143.
pub fn pack_spo_crystal(
    container: &mut [u64; CONTAINER_WORDS],
    triples: &[CrystalTriple],
) {
    let n = triples.len().min(CrystalTriple::MAX_CRYSTAL_TRIPLES);
    // Zero the region first
    for w in W_SPO_CRYSTAL_START..=W_SPO_CRYSTAL_END {
        container[w] = 0;
    }
    for i in 0..n {
        let bytes = triples[i].to_bytes();
        // Each triple is 16 bytes = 2 words
        let w_off = i * 2;
        container[W_SPO_CRYSTAL_START + w_off] = u64::from_le_bytes(
            bytes[0..8].try_into().unwrap(),
        );
        container[W_SPO_CRYSTAL_START + w_off + 1] = u64::from_le_bytes(
            bytes[8..16].try_into().unwrap(),
        );
    }
}

/// Unpack SPO crystal triples from W128-143.
/// Returns up to 8 triples (stops at first all-zero triple).
pub fn unpack_spo_crystal(container: &[u64; CONTAINER_WORDS]) -> Vec<CrystalTriple> {
    let mut triples = Vec::with_capacity(CrystalTriple::MAX_CRYSTAL_TRIPLES);
    for i in 0..CrystalTriple::MAX_CRYSTAL_TRIPLES {
        let w_off = i * 2;
        let w0 = container[W_SPO_CRYSTAL_START + w_off];
        let w1 = container[W_SPO_CRYSTAL_START + w_off + 1];
        if w0 == 0 && w1 == 0 {
            break; // sentinel: all-zero triple ends the list
        }
        let mut bytes = [0u8; CrystalTriple::BYTE_SIZE];
        bytes[0..8].copy_from_slice(&w0.to_le_bytes());
        bytes[8..16].copy_from_slice(&w1.to_le_bytes());
        triples.push(CrystalTriple::from_bytes(&bytes));
    }
    triples
}

// ─── CogRecord8K: Extended edges (W224-239) ───────────────────────────

/// An inline edge: verb(8) + target(8) = 16 bits.
/// 4 edges per u64 word. 16 words = 64 edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineEdge {
    pub verb: u8,
    pub target: u8,
}

impl InlineEdge {
    /// Pack 4 edges into one u64 word.
    fn pack4(edges: &[InlineEdge; 4]) -> u64 {
        (edges[0].verb as u64)
            | ((edges[0].target as u64) << 8)
            | ((edges[1].verb as u64) << 16)
            | ((edges[1].target as u64) << 24)
            | ((edges[2].verb as u64) << 32)
            | ((edges[2].target as u64) << 40)
            | ((edges[3].verb as u64) << 48)
            | ((edges[3].target as u64) << 56)
    }

    /// Unpack 4 edges from one u64 word.
    fn unpack4(word: u64) -> [InlineEdge; 4] {
        [
            InlineEdge { verb: (word & 0xFF) as u8, target: ((word >> 8) & 0xFF) as u8 },
            InlineEdge { verb: ((word >> 16) & 0xFF) as u8, target: ((word >> 24) & 0xFF) as u8 },
            InlineEdge { verb: ((word >> 32) & 0xFF) as u8, target: ((word >> 40) & 0xFF) as u8 },
            InlineEdge { verb: ((word >> 48) & 0xFF) as u8, target: ((word >> 56) & 0xFF) as u8 },
        ]
    }
}

/// Pack extended edges into W224-239 (up to 64 edges).
/// These are OVERFLOW edges beyond the 64 in W16-31.
pub fn pack_extended_edges(
    container: &mut [u64; CONTAINER_WORDS],
    edges: &[InlineEdge],
) {
    let n_words = (W_EXT_EDGES_END - W_EXT_EDGES_START) + 1; // 16
    // Zero the region
    for w in W_EXT_EDGES_START..=W_EXT_EDGES_END {
        container[w] = 0;
    }
    for wi in 0..n_words {
        let base = wi * 4;
        if base >= edges.len() {
            break;
        }
        let mut quad = [InlineEdge::default(); 4];
        for j in 0..4 {
            if base + j < edges.len() {
                quad[j] = edges[base + j];
            }
        }
        container[W_EXT_EDGES_START + wi] = InlineEdge::pack4(&quad);
    }
}

/// Unpack extended edges from W224-239.
/// Returns up to 64 edges (stops at first all-zero quad).
pub fn unpack_extended_edges(container: &[u64; CONTAINER_WORDS]) -> Vec<InlineEdge> {
    let mut edges = Vec::with_capacity(MAX_INLINE_EDGES);
    let n_words = (W_EXT_EDGES_END - W_EXT_EDGES_START) + 1;
    for wi in 0..n_words {
        let word = container[W_EXT_EDGES_START + wi];
        if word == 0 {
            break;
        }
        let quad = InlineEdge::unpack4(word);
        for e in &quad {
            if e.verb == 0 && e.target == 0 {
                return edges;
            }
            edges.push(*e);
        }
    }
    edges
}

// ─── CogRecord8K: Wide checksum + format tag (W254-255) ──────────────

/// Set the wide checksum (W254). XOR-fold of W128-253.
pub fn compute_wide_checksum(container: &[u64; CONTAINER_WORDS]) -> u64 {
    let mut xor = 0u64;
    for i in 128..254 {
        xor ^= container[i];
    }
    xor
}

/// Write wide checksum and format tag.
pub fn seal_wide_meta(container: &mut [u64; CONTAINER_WORDS], format_tag: u64) {
    container[W_WIDE_CHECKSUM] = compute_wide_checksum(container);
    container[W_FORMAT_TAG] = format_tag;
}

/// Verify wide checksum.
pub fn verify_wide_checksum(container: &[u64; CONTAINER_WORDS]) -> bool {
    container[W_WIDE_CHECKSUM] == compute_wide_checksum(container)
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn spo_to_bytes(spo: &SpoBase17) -> [u8; SpoBase17::BYTE_SIZE] {
    let mut buf = [0u8; SpoBase17::BYTE_SIZE];
    let s = spo.subject.to_bytes();
    let p = spo.predicate.to_bytes();
    let o = spo.object.to_bytes();
    buf[0..Base17::BYTE_SIZE].copy_from_slice(&s);
    buf[Base17::BYTE_SIZE..Base17::BYTE_SIZE * 2].copy_from_slice(&p);
    buf[Base17::BYTE_SIZE * 2..Base17::BYTE_SIZE * 3].copy_from_slice(&o);
    buf
}

fn bytes_to_spo(buf: &[u8; SpoBase17::BYTE_SIZE]) -> SpoBase17 {
    SpoBase17 {
        subject: Base17::from_bytes(&buf[0..Base17::BYTE_SIZE]),
        predicate: Base17::from_bytes(&buf[Base17::BYTE_SIZE..Base17::BYTE_SIZE * 2]),
        object: Base17::from_bytes(&buf[Base17::BYTE_SIZE * 2..Base17::BYTE_SIZE * 3]),
    }
}

/// Check whether the bgz17 annex is populated (W112 non-zero).
pub fn has_bgz17_annex(container: &[u64; CONTAINER_WORDS]) -> bool {
    container[W_BGZ17_ANNEX_START] != 0
}

/// Stride-16 sample positions that the Cascade hits.
/// Returns the 16 word indices: W0, W16, W32, ..., W240.
pub const fn cascade_stride16_positions() -> [usize; 16] {
    let mut pos = [0usize; 16];
    let mut i = 0;
    while i < 16 {
        pos[i] = i * 16;
        i += 1;
    }
    pos
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spo() -> SpoBase17 {
        SpoBase17 {
            subject: Base17 { dims: [100, -200, 300, -400, 500, -600, 700, -800, 900, -1000, 1100, -1200, 1300, -1400, 1500, -1600, 1700] },
            predicate: Base17 { dims: [-50, 150, -250, 350, -450, 550, -650, 750, -850, 950, -1050, 1150, -1250, 1350, -1450, 1550, -1650] },
            object: Base17 { dims: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170] },
        }
    }

    #[test]
    fn test_base17_annex_roundtrip() {
        let spo = make_spo();
        let mut container = [0u64; CONTAINER_WORDS];
        pack_base17_annex(&mut container, &spo);
        let unpacked = unpack_base17_annex(&container);
        assert_eq!(spo, unpacked);
    }

    #[test]
    fn test_palette_word_roundtrip() {
        let edge = PaletteEdge { s_idx: 42, p_idx: 128, o_idx: 255 };
        let tq = 200u8;
        let mut container = [0u64; CONTAINER_WORDS];
        pack_palette_word(&mut container, edge, tq);
        let (edge2, tq2) = unpack_palette_word(&container);
        assert_eq!(edge, edge2);
        assert_eq!(tq, tq2);
    }

    #[test]
    fn test_full_annex_roundtrip() {
        let spo = make_spo();
        let edge = PaletteEdge { s_idx: 7, p_idx: 99, o_idx: 200 };
        let tq = 150u8;
        let mut container = [0u64; CONTAINER_WORDS];
        pack_annex(&mut container, &spo, edge, tq);
        let (spo2, edge2, tq2) = unpack_annex(&container);
        assert_eq!(spo, spo2);
        assert_eq!(edge, edge2);
        assert_eq!(tq, tq2);
    }

    #[test]
    fn test_annex_does_not_clobber_neighbors() {
        let mut container = [0xDEAD_BEEF_CAFE_BABEu64; CONTAINER_WORDS];
        let spo = make_spo();
        let edge = PaletteEdge { s_idx: 1, p_idx: 2, o_idx: 3 };
        pack_annex(&mut container, &spo, edge, 42);

        // W111 (before annex) untouched
        assert_eq!(container[111], 0xDEAD_BEEF_CAFE_BABE);
        // W126 (after annex) untouched
        assert_eq!(container[126], 0xDEAD_BEEF_CAFE_BABE);
        // Roundtrip still works
        let (spo2, edge2, tq2) = unpack_annex(&container);
        assert_eq!(spo, spo2);
        assert_eq!(edge, edge2);
        assert_eq!(tq2, 42);
    }

    #[test]
    fn test_cascade_stride16_hits_annex() {
        let positions = cascade_stride16_positions();
        // W112 = 7th position (index 7, since 7*16=112)
        assert_eq!(positions[7], 112);
        assert_eq!(positions[7], W_BGZ17_ANNEX_START);
    }

    #[test]
    fn test_has_bgz17_annex() {
        let empty = [0u64; CONTAINER_WORDS];
        assert!(!has_bgz17_annex(&empty));

        let mut filled = [0u64; CONTAINER_WORDS];
        let spo = make_spo();
        pack_base17_annex(&mut filled, &spo);
        assert!(has_bgz17_annex(&filled));
    }

    #[test]
    fn test_spo_crystal_roundtrip() {
        let triples = vec![
            CrystalTriple {
                s_idx: 10, p_idx: 20, o_idx: 30, flags: 0x01,
                distance: 1234, weight: 5678,
                source: 100, target: 200, reserved: 0,
            },
            CrystalTriple {
                s_idx: 40, p_idx: 50, o_idx: 60, flags: 0x02,
                distance: 4321, weight: 8765,
                source: 300, target: 400, reserved: 0,
            },
        ];
        let mut container = [0u64; CONTAINER_WORDS];
        pack_spo_crystal(&mut container, &triples);
        let unpacked = unpack_spo_crystal(&container);
        assert_eq!(unpacked.len(), 2);
        assert_eq!(unpacked[0], triples[0]);
        assert_eq!(unpacked[1], triples[1]);
    }

    #[test]
    fn test_spo_crystal_max_8() {
        let triples: Vec<CrystalTriple> = (0..10).map(|i| CrystalTriple {
            s_idx: i as u8, p_idx: i as u8, o_idx: i as u8, flags: 1,
            distance: i as u16 * 100, weight: i as u16 * 10,
            source: i as u16, target: i as u16 + 1, reserved: 0,
        }).collect();

        let mut container = [0u64; CONTAINER_WORDS];
        pack_spo_crystal(&mut container, &triples);
        let unpacked = unpack_spo_crystal(&container);
        // Only first 8 stored
        assert_eq!(unpacked.len(), 8);
        assert_eq!(unpacked[0].s_idx, 0);
        assert_eq!(unpacked[7].s_idx, 7);
    }

    #[test]
    fn test_extended_edges_roundtrip() {
        let edges: Vec<InlineEdge> = (1..=20).map(|i| InlineEdge {
            verb: i as u8,
            target: (i * 3) as u8,
        }).collect();

        let mut container = [0u64; CONTAINER_WORDS];
        pack_extended_edges(&mut container, &edges);
        let unpacked = unpack_extended_edges(&container);
        assert_eq!(unpacked.len(), 20);
        for (a, b) in edges.iter().zip(unpacked.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_extended_edges_max_64() {
        let edges: Vec<InlineEdge> = (1..=64).map(|i| InlineEdge {
            verb: i as u8,
            target: 255 - i as u8,
        }).collect();

        let mut container = [0u64; CONTAINER_WORDS];
        pack_extended_edges(&mut container, &edges);
        let unpacked = unpack_extended_edges(&container);
        assert_eq!(unpacked.len(), 64);
        assert_eq!(unpacked[0], edges[0]);
        assert_eq!(unpacked[63], edges[63]);
    }

    #[test]
    fn test_wide_checksum_seal_verify() {
        let mut container = [0u64; CONTAINER_WORDS];

        // Pack some data into wide meta regions
        let spo = make_spo();
        pack_base17_annex(&mut container, &spo);
        let triples = vec![CrystalTriple {
            s_idx: 1, p_idx: 2, o_idx: 3, flags: 0xFF,
            distance: 999, weight: 111,
            source: 5, target: 10, reserved: 0,
        }];
        pack_spo_crystal(&mut container, &triples);

        // Seal with checksum + format tag
        seal_wide_meta(&mut container, FORMAT_COGREC_8K);

        // Verify
        assert!(verify_wide_checksum(&container));
        assert_eq!(container[W_FORMAT_TAG], FORMAT_COGREC_8K);

        // Tamper → verification fails
        container[130] ^= 1;
        assert!(!verify_wide_checksum(&container));
    }

    #[test]
    fn test_stride16_discriminator_with_annex() {
        // Demonstrate that filling the annex changes the stride-16 sample
        let mut c1 = [0u64; CONTAINER_WORDS];
        let mut c2 = [0u64; CONTAINER_WORDS];

        // Same identity, different content
        c1[W_DN_ADDR] = 0x1234;
        c2[W_DN_ADDR] = 0x1234;

        // Pack different Base17 patterns
        let spo_a = SpoBase17 {
            subject: Base17 { dims: [1000; BASE_DIM] },
            predicate: Base17 { dims: [2000; BASE_DIM] },
            object: Base17 { dims: [3000; BASE_DIM] },
        };
        let spo_b = SpoBase17 {
            subject: Base17 { dims: [-1000; BASE_DIM] },
            predicate: Base17 { dims: [-2000; BASE_DIM] },
            object: Base17 { dims: [-3000; BASE_DIM] },
        };
        pack_base17_annex(&mut c1, &spo_a);
        pack_base17_annex(&mut c2, &spo_b);

        // Stride-16 sample at W112 now differs
        let positions = cascade_stride16_positions();
        let w112_idx = 7;
        assert_eq!(positions[w112_idx], W_BGZ17_ANNEX_START);
        assert_ne!(c1[positions[w112_idx]], c2[positions[w112_idx]]);

        // Hamming distance on the stride-16 sample
        let mut hamming = 0u32;
        for &pos in &positions {
            hamming += (c1[pos] ^ c2[pos]).count_ones();
        }
        assert!(hamming > 0, "stride-16 sample should discriminate different Base17 patterns");
    }

    #[test]
    fn test_container_geometry() {
        assert_eq!(CONTAINER_BYTES, 2048);
        assert_eq!(CONTAINER_WORDS, 256);
        // Base17 annex: 13 words = 104 bytes ≥ 102 bytes (SpoBase17)
        assert!((W_BGZ17_ANNEX_END - W_BGZ17_ANNEX_START + 1) * 8 >= SpoBase17::BYTE_SIZE);
        // Palette word fits in 1 word
        assert_eq!(W_PALETTE_PACK, W_BGZ17_ANNEX_END + 1);
    }

    #[test]
    fn test_inline_edge_pack4_roundtrip() {
        let edges = [
            InlineEdge { verb: 1, target: 10 },
            InlineEdge { verb: 2, target: 20 },
            InlineEdge { verb: 3, target: 30 },
            InlineEdge { verb: 4, target: 40 },
        ];
        let packed = InlineEdge::pack4(&edges);
        let unpacked = InlineEdge::unpack4(packed);
        assert_eq!(edges, unpacked);
    }

    #[test]
    fn test_format_tags_are_ascii() {
        // Verify the format tags decode to readable ASCII
        let bgz_bytes = FORMAT_BGZ17_ANNEX.to_le_bytes();
        let cog_bytes = FORMAT_COGREC_8K.to_le_bytes();
        assert!(bgz_bytes.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_'));
        assert!(cog_bytes.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_'));
    }
}
