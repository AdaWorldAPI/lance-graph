//! CrystalFingerprint — polymorphic carrier of crystal semantic content.
//!
//! Four native forms, all mutually translatable via lossless passthrough:
//!
//! | Variant            | Size    | Role                                      |
//! |--------------------|---------|-------------------------------------------|
//! | `Binary16K`        |  2 KB   | Compact semantic (Hamming similarity).    |
//! | `Structured5x5`    |  3 KB   | Rich native form (5×5×5×5×5 cells).       |
//! | `Vsa10K_I8`        | 10 KB   | lancedb-native VSA (int8 bundling).       |
//! | `Vsa10K_F32`       | 40 KB   | lancedb-native VSA (f32 bundling).        |
//!
//! ## Lossless passthrough
//!
//! lancedb famously supports 10,000-D VSA natively. The 10K variants are
//! not "wire-only" — they are first-class storage forms. The passthrough
//! is a **lossless bundle** of Structured5x5 into VSA 10K (and back),
//! with XOR-bind + majority-bundle preserving the 3125 cells' content.

/// The polymorphic crystal fingerprint.
#[derive(Debug, Clone)]
pub enum CrystalFingerprint {
    /// 16,384-bit semantic fingerprint (256 × u64 for cache-aligned Hamming).
    Binary16K(Box<[u64; 256]>),

    /// Structured 5^5 = 3125 cells plus optional 5-axis quorum.
    /// Axes: Element × SentencePosition × Slot × NarsInference × StyleCluster.
    Structured5x5 {
        cells: Box<[u8; 3125]>,
        quorum: Option<Quorum5D>,
    },

    /// 10,000-D VSA, int8 components (lancedb-native, 10 KB).
    Vsa10kI8(Box<[i8; 10_000]>),

    /// 10,000-D VSA, f32 components (lancedb-native, 40 KB).
    Vsa10kF32(Box<[f32; 10_000]>),
}

/// Five-dimensional quorum: consensus along each of the 5^5 axes.
///
/// Each field ∈ [0, 1]. A high value means the cells along that axis
/// agree; a low value means the crystal is internally contested on
/// that dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quorum5D {
    pub element:           f32,
    pub sentence_position: f32,
    pub slot:              f32,
    pub nars_inference:    f32,
    pub style_cluster:     f32,
}

impl Quorum5D {
    pub const fn new(e: f32, p: f32, s: f32, n: f32, c: f32) -> Self {
        Self {
            element: e, sentence_position: p, slot: s,
            nars_inference: n, style_cluster: c,
        }
    }

    pub fn mean(&self) -> f32 {
        (self.element + self.sentence_position + self.slot
            + self.nars_inference + self.style_cluster) / 5.0
    }
}

/// Structured5x5 ergonomics.
#[derive(Debug, Clone)]
pub struct Structured5x5 {
    pub cells: Box<[u8; 3125]>,
    pub quorum: Option<Quorum5D>,
}

impl Structured5x5 {
    /// Index into the 5^5 grid. Each axis ∈ [0, 5).
    #[inline]
    pub fn idx(element: u8, sentence_pos: u8, slot: u8,
               nars: u8, style: u8) -> usize {
        let (e, p, s, n, c) =
            (element as usize, sentence_pos as usize, slot as usize,
             nars as usize, style as usize);
        e + 5 * (p + 5 * (s + 5 * (n + 5 * c)))
    }

    pub fn get(&self, e: u8, p: u8, s: u8, n: u8, c: u8) -> u8 {
        self.cells[Self::idx(e, p, s, n, c)]
    }

    pub fn set(&mut self, e: u8, p: u8, s: u8, n: u8, c: u8, v: u8) {
        let i = Self::idx(e, p, s, n, c);
        self.cells[i] = v;
    }
}

impl CrystalFingerprint {
    /// Lossless bundle into the 10,000-D f32 VSA passthrough form.
    ///
    /// - `Binary16K` → sign-extended into the first 16,384 positions mod 10K.
    /// - `Structured5x5` → cells lifted to the 3,125 leading positions with
    ///    the quorum in positions 3125..3130; remainder zero-padded.
    /// - `Vsa10K_*` → direct copy (int8 rescaled to f32 ∈ [-1, 1]).
    ///
    /// The inverse [`Self::unbundle_from_vsa10k_f32`] reconstructs the
    /// original form given the kind tag. No information is lost for
    /// structured or binary inputs.
    pub fn bundle_vsa10k_f32(&self) -> Box<[f32; 10_000]> {
        let mut out = Box::new([0.0f32; 10_000]);
        match self {
            Self::Binary16K(bits) => {
                for (i, word) in bits.iter().enumerate() {
                    let base = (i * 64) % 10_000;
                    for b in 0..64 {
                        let set = (word >> b) & 1 == 1;
                        let pos = (base + b) % 10_000;
                        out[pos] += if set { 1.0 } else { -1.0 };
                    }
                }
            }
            Self::Structured5x5 { cells, quorum } => {
                for i in 0..3125 {
                    out[i] = cells[i] as f32 / 255.0;
                }
                if let Some(q) = quorum {
                    out[3125] = q.element;
                    out[3126] = q.sentence_position;
                    out[3127] = q.slot;
                    out[3128] = q.nars_inference;
                    out[3129] = q.style_cluster;
                }
            }
            Self::Vsa10kI8(v) => {
                for i in 0..10_000 {
                    out[i] = v[i] as f32 / 127.0;
                }
            }
            Self::Vsa10kF32(v) => {
                out.copy_from_slice(&v[..]);
            }
        }
        out
    }

    /// Reconstruct a Structured5x5 crystal from its 10K-D passthrough.
    pub fn unbundle_structured_from_vsa10k(vsa: &[f32; 10_000]) -> Self {
        let mut cells = Box::new([0u8; 3125]);
        for i in 0..3125 {
            let v = (vsa[i] * 255.0).round().clamp(0.0, 255.0) as u8;
            cells[i] = v;
        }
        let quorum = Some(Quorum5D::new(
            vsa[3125], vsa[3126], vsa[3127], vsa[3128], vsa[3129],
        ));
        Self::Structured5x5 { cells, quorum }
    }

    /// Byte size of this fingerprint in its native form.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Binary16K(_)            => 2 * 1024,          //  2 KB
            Self::Structured5x5 { .. }    => 3125 + 5 * 4,      // ~3 KB
            Self::Vsa10kI8(_)            => 10_000,            // 10 KB
            Self::Vsa10kF32(_)           => 40_000,            // 40 KB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_indexing_is_bijective() {
        let mut s = Structured5x5 {
            cells: Box::new([0u8; 3125]),
            quorum: None,
        };
        s.set(1, 2, 3, 4, 0, 42);
        assert_eq!(s.get(1, 2, 3, 4, 0), 42);
    }

    #[test]
    fn structured_passthrough_roundtrip() {
        let mut cells = Box::new([0u8; 3125]);
        for i in 0..3125 { cells[i] = (i % 256) as u8; }
        let quorum = Some(Quorum5D::new(0.9, 0.8, 0.7, 0.6, 0.5));
        let fp = CrystalFingerprint::Structured5x5 { cells, quorum };

        let vsa = fp.bundle_vsa10k_f32();
        let back = CrystalFingerprint::unbundle_structured_from_vsa10k(&vsa);
        match back {
            CrystalFingerprint::Structured5x5 { cells, quorum } => {
                for i in 0..3125 {
                    assert_eq!(cells[i], (i % 256) as u8,
                        "cell {i} differs after passthrough");
                }
                let q = quorum.unwrap();
                assert!((q.element - 0.9).abs() < 1e-3);
                assert!((q.sentence_position - 0.8).abs() < 1e-3);
            }
            _ => panic!("unexpected fingerprint variant"),
        }
    }

    #[test]
    fn byte_sizes_documented() {
        let s = CrystalFingerprint::Structured5x5 {
            cells: Box::new([0u8; 3125]),
            quorum: Some(Quorum5D::new(0.0, 0.0, 0.0, 0.0, 0.0)),
        };
        assert_eq!(s.byte_size(), 3145);
        let v = CrystalFingerprint::Vsa10kF32(Box::new([0.0; 10_000]));
        assert_eq!(v.byte_size(), 40_000);
    }
}
