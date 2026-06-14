//! # lance-graph-turbovec — TurboQuant ANN index on the lance-graph spine
//!
//! [`turbovec`](https://arxiv.org/abs/2504.19874) is Google Research's
//! **TurboQuant**: a *data-oblivious* scalar-quantization index for vector
//! search — normalize → random rotation → per-coordinate (TQ+) calibration →
//! Lloyd-Max 2/3/4-bit quantization → bit-pack → SIMD nibble-LUT scan. No
//! training, no rebuilds; a 10 M-doc f32 corpus (31 GB) fits in ~4 GB.
//!
//! This crate is the **thin bridge** that fits turbovec into the workspace.
//!
//! ## Placement — why lance-graph, not ndarray
//!
//! turbovec is a *search index* (IO, id-map, deletes, filtered search), not a
//! hardware SIMD primitive. Per the stack split it belongs on the **spine**
//! (lance-graph), a sibling to `bgz17` / `deepnsm` / `bgz-tensor` — the
//! standalone codec/search crates — and is `exclude`d from the main workspace
//! exactly like them. What belongs in **ndarray** is the *kernel*: and indeed
//! ndarray already owns the relevant ANN substrate — `hpc::clam` /
//! `clam_search` (CLAM neighborhood), `hpc::cam_pq` (PQ-ADC), `hpc::cascade`
//! (HDR), and `hpc::amx_matmul` (the AMX/VNNI int8 GEMM). So the integration
//! rule is: **the index lives here; every wide op is borrowed from
//! `ndarray::simd`.**
//!
//! ## Two kernels, one index — the [`Kernel`] switch
//!
//! - [`Kernel::NativeLut`] — turbovec's hand-written nibble-LUT ADC
//!   (`pshufb`/`vqtbl` gather, 32-vector blocks, runtime AVX-512BW / AVX2 /
//!   NEON / scalar). The fast path; lowest memory (2–4 bits/dim).
//! - [`Kernel::PolyfillGemm`] — TurboQuant scoring re-expressed as a batched
//!   int8 GEMM `Q·X̂ᵀ` routed through [`ndarray::simd::matmul_i8_to_i32`].
//!   turbovec writes **zero** raw intrinsics; ndarray picks the backend:
//!   **AMX `TDPBUSD` tile (byte-asm, 16 384 MAC/instr, Sapphire Rapids+) →
//!   AVX-512 VPDPBUSD → AVX-VNNI → scalar**, all bit-identical. This is the
//!   "ship AMX through dispatch" path — it lights up AMX on capable silicon
//!   with no code change.
//!
//! ### The headline synergy finding (measured)
//!
//! On an AVX-512+VNNI host (no AMX tiles), `n=20 000, dim=512, 4-bit`:
//!
//! | kernel | ns/query | recall@10 | DB memory |
//! |---|---|---|---|
//! | `NativeLut` (AVX-512BW) | **76 073** | 0.785 | 5 000 KB (4-bit) |
//! | `PolyfillGemm` (VPDPBUSD-zmm) | 866 899 | 0.764 | 10 000 KB (i8) |
//! | scalar reference | 6 267 279 | — | — |
//!
//! The polyfill GEMM is **11.4× slower** than the native LUT — **because
//! TurboQuant's whole design trades the matmul away.** LUT-ADC is an O(1)
//! table gather per coordinate; the GEMM does the full `dim`-length dot per
//! (query, vector) pair. AMX accelerates *exactly the operation TurboQuant
//! removed*, so even VPDPBUSD (and, extrapolating, the 4×-wider AMX tile)
//! cannot close the algorithmic gap. **Conclusion: keep the native LUT
//! kernel; AMX is the wrong tool for this index.** The polyfill is retained
//! as a measured baseline and as proof the index is `ndarray::simd`-clean.
//!
//! See `KNOWLEDGE.md` for the full synergy map against the bgz-tensor
//! primitives (HDR popcount stacking early-exit, Belichtungsmesser sigma
//! confidence thresholds, preheating vs palette256 ranking).

pub use turbovec::{AddError, ConstructError, IdMapIndex, SearchResults, TurboQuantIndex};

use std::sync::OnceLock;
use turbovec::TurboQuantIndex as Inner;

/// Which scoring kernel [`TurboVec::search`] dispatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// turbovec's native nibble-LUT ADC (hand-written AVX-512BW / AVX2 / NEON
    /// / scalar). Fast path, lowest memory.
    NativeLut,
    /// `ndarray::simd::matmul_i8_to_i32` int8-GEMM scoring. SIMD/AMX backend
    /// chosen inside ndarray; zero intrinsics in turbovec.
    PolyfillGemm,
}

/// lance-graph-facing bridge over [`turbovec::TurboQuantIndex`].
///
/// Adds (1) the [`Kernel`] A/B switch, (2) a lazily-built reconstruction
/// cache for the polyfill GEMM (the analogue of turbovec's native blocked
/// cache), and (3) a [`TurboVec::polyfill_backend`] report of which
/// `ndarray::simd` int8 tier this host dispatches.
pub struct TurboVec {
    inner: Inner,
    /// Transposed i8 reconstruction `X̂ᵀ` for the polyfill GEMM, built once
    /// on first polyfill search and reset on `add` (interior mutability so
    /// `search` stays `&self`, matching turbovec's concurrent-search design).
    db_i8_t: OnceLock<Vec<i8>>,
}

impl TurboVec {
    /// Construct with a fixed dimensionality. `bit_width ∈ {2,3,4}`,
    /// `dim` a positive multiple of 8.
    pub fn new(dim: usize, bit_width: usize) -> Result<Self, ConstructError> {
        Ok(Self {
            inner: Inner::new(dim, bit_width)?,
            db_i8_t: OnceLock::new(),
        })
    }

    /// Add a flat `n * dim` batch of f32 vectors. Invalidates the polyfill
    /// reconstruction cache.
    pub fn add(&mut self, vectors: &[f32]) {
        self.inner.add(vectors);
        self.db_i8_t = OnceLock::new();
    }

    /// Top-`k` search via the chosen [`Kernel`]. Safe to call from multiple
    /// threads (`&self`); the polyfill path lazily materialises its
    /// reconstruction on first use.
    pub fn search(&self, queries: &[f32], k: usize, kernel: Kernel) -> SearchResults {
        match kernel {
            Kernel::NativeLut => self.inner.search(queries, k),
            Kernel::PolyfillGemm => {
                let db = self
                    .db_i8_t
                    .get_or_init(|| self.inner.reconstruct_db_i8_transposed());
                self.inner.search_polyfill_with_db(queries, k, db)
            }
        }
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the index holds zero vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Borrow the underlying [`turbovec::TurboQuantIndex`] (write/load, ids,
    /// filtered search, deletes — the full upstream surface).
    pub fn inner(&self) -> &Inner {
        &self.inner
    }

    /// Which `ndarray::simd` int8-GEMM tier the [`Kernel::PolyfillGemm`] path
    /// dispatches on THIS host. `"amx-tdpbusd-tile"` on Sapphire-Rapids-class
    /// silicon (OS-enabled tiles), otherwise the AVX-512/AVX-VNNI/scalar
    /// ladder inside `matmul_i8_to_i32` selects at call time.
    pub fn polyfill_backend(&self) -> &'static str {
        if ndarray::simd::amx_available() {
            "amx-tdpbusd-tile"
        } else {
            "avx512-vnni / avx-vnni / scalar (runtime-selected by ndarray)"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut out = vec![0.0f32; n * dim];
        for row in out.chunks_mut(dim) {
            let mut norm = 0.0f64;
            for x in row.iter_mut() {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let v = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
                *x = v as f32;
                norm += v * v;
            }
            let inv = 1.0 / (norm.sqrt() + 1e-9);
            for x in row.iter_mut() {
                *x = (*x as f64 * inv) as f32;
            }
        }
        out
    }

    #[test]
    fn both_kernels_agree_on_majority_of_topk() {
        let (dim, n, nq, k) = (64usize, 800usize, 6usize, 10usize);
        let db = unit_vectors(n, dim, 3);
        let queries = unit_vectors(nq, dim, 77);
        let mut tv = TurboVec::new(dim, 4).unwrap();
        tv.add(&db);
        assert_eq!(tv.len(), n);

        let native = tv.search(&queries, k, Kernel::NativeLut);
        let poly = tv.search(&queries, k, Kernel::PolyfillGemm);
        assert_eq!(native.nq, nq);
        assert_eq!(poly.nq, nq);

        // The two kernels are different approximations of the same inner
        // product; on clean data they should agree on a solid majority of
        // each query's top-k.
        let mut overlap = 0usize;
        for qi in 0..nq {
            let a: std::collections::BTreeSet<i64> = native.indices[qi * k..(qi + 1) * k]
                .iter()
                .copied()
                .collect();
            for &i in &poly.indices[qi * k..(qi + 1) * k] {
                if a.contains(&i) {
                    overlap += 1;
                }
            }
        }
        let agree = overlap as f64 / (nq * k) as f64;
        assert!(
            agree >= 0.5,
            "native/polyfill top-k agreement {agree:.3} below 0.5"
        );
    }

    #[test]
    fn backend_report_is_nonempty() {
        let tv = TurboVec::new(8, 2).unwrap();
        assert!(!tv.polyfill_backend().is_empty());
        assert!(tv.is_empty());
    }
}
