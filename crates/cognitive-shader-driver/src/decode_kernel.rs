//! **LAB-ONLY.** D1.3 — residual PQ via decode-kernel composition.
//!
//! Scope correction (per `cognitive-shader-architecture.md`): this module
//! sits on the **hydration / calibration path**, not the cascade inference
//! path. The inference cascade uses `p64_bridge::CognitiveShader::cascade`
//! at Layer 2 (line 582 of that doc); decode kernels here are for offline
//! codec-candidate calibration. A codec that passes the token-agreement
//! cert gate (D2.x) then runs at **weight hydration time** (GGUF → palette
//! + Fingerprint<256> + holographic residual), never per-inference.
//!
//! This module defines:
//!
//! - [`DecodeKernel`] — the codec decode/encode trait, signature-keyed
//!   into `CodecKernelCache<H>` at the `H` slot where `H: DecodeKernel`.
//! - [`StubDecodeKernel`] — deterministic reference for tests (byte-level
//!   round-trip, no quantization loss, matches Rule F "serialise once at
//!   edge" — the decode/encode IS the edge).
//! - [`ResidualComposer`] — composes two decoders with subtract/add:
//!     `decode(enc) = base.decode(enc[..k]) + residual.decode(enc[k..])`
//!   `encode(v)    = [base.encode(v); residual.encode(v - base.decode(base.encode(v)))]`
//!   Depth `d > 1` recurses: the residual field itself is a `ResidualComposer`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Error from encode / decode — size mismatch or downstream kernel failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Input slice size doesn't match the kernel's expected `bytes_per_row`
    /// (for decode) or `dim` (for encode).
    SizeMismatch { expected: usize, actual: usize },
    /// A composed-kernel stage failed with a downstream reason.
    Stage { stage: &'static str, detail: String },
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizeMismatch { expected, actual } => {
                write!(f, "decode size mismatch: expected {expected}, got {actual}")
            }
            Self::Stage { stage, detail } => write!(f, "stage {stage} failed: {detail}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// A codec decode/encode kernel.
///
/// Object-safe so `ResidualComposer` can hold `Box<dyn DecodeKernel>` for
/// each stage; keyed by `signature()` into `CodecKernelCache<H>` at the
/// `H = Box<dyn DecodeKernel>` slot.
pub trait DecodeKernel: Send + Sync + std::fmt::Debug {
    /// Decode `bytes` (exactly `bytes_per_row` long) into `dim` `f32` values.
    fn decode(&self, bytes: &[u8]) -> Result<Vec<f32>, DecodeError>;
    /// Encode `vec` (exactly `dim` `f32` values) into `bytes_per_row` bytes.
    fn encode(&self, vec: &[f32]) -> Result<Vec<u8>, DecodeError>;
    /// Bytes per encoded row — composers rely on this to split payloads.
    fn bytes_per_row(&self) -> u32;
    /// Dimension of the decoded `f32` vector.
    fn dim(&self) -> u32;
    /// Stable hash for JIT cache keying.
    fn signature(&self) -> u64;
    /// Backend tier ("avx512" | "amx" | "stub") — never "scalar" on SoA.
    fn backend(&self) -> &'static str;
}

// ─── Stub decoder (byte-exact round-trip for testing) ────────────────────

/// Deterministic byte-exact decoder. `dim` `f32` values ⇔ `dim * 4` bytes
/// via native-endian reinterpret. No quantization, no compression — the
/// round-trip is exact.
///
/// This IS NOT a real codec. It exists so the `ResidualComposer` and
/// `CodecKernelCache` composition tests can verify the plumbing without
/// depending on a trained palette.
#[derive(Debug, Clone, Copy)]
pub struct StubDecodeKernel {
    pub dim: u32,
    pub tag: u64,
}

impl StubDecodeKernel {
    pub const fn new(dim: u32, tag: u64) -> Self { Self { dim, tag } }
}

impl DecodeKernel for StubDecodeKernel {
    fn decode(&self, bytes: &[u8]) -> Result<Vec<f32>, DecodeError> {
        let expected = self.bytes_per_row() as usize;
        if bytes.len() != expected {
            return Err(DecodeError::SizeMismatch { expected, actual: bytes.len() });
        }
        let mut out = Vec::with_capacity(self.dim as usize);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_ne_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }

    fn encode(&self, vec: &[f32]) -> Result<Vec<u8>, DecodeError> {
        let expected = self.dim as usize;
        if vec.len() != expected {
            return Err(DecodeError::SizeMismatch { expected, actual: vec.len() });
        }
        let mut out = Vec::with_capacity(expected * 4);
        for &v in vec {
            out.extend_from_slice(&v.to_ne_bytes());
        }
        Ok(out)
    }

    fn bytes_per_row(&self) -> u32 { self.dim * 4 }
    fn dim(&self) -> u32 { self.dim }

    fn signature(&self) -> u64 {
        let mut h = DefaultHasher::new();
        "stub_decode".hash(&mut h);
        self.dim.hash(&mut h);
        self.tag.hash(&mut h);
        h.finish()
    }

    fn backend(&self) -> &'static str { "stub" }
}

// ─── Residual composer ───────────────────────────────────────────────────

/// Composes a `base` decoder with a `residual` decoder: the encoded payload
/// is the concatenation of `base.encode(v)` + `residual.encode(v -
/// base.decode(base.encode(v)))`; decode reverses by decoding each stage's
/// own byte range and summing.
///
/// **Contract:** `base.dim() == residual.dim()`. Enforced by `new()`.
///
/// Deeper residual chains (depth > 1) nest `ResidualComposer`s — the
/// `residual` slot itself is a `Box<dyn DecodeKernel>` which can be
/// another `ResidualComposer`.
#[derive(Debug)]
pub struct ResidualComposer {
    base: Box<dyn DecodeKernel>,
    residual: Box<dyn DecodeKernel>,
}

impl ResidualComposer {
    /// Build a two-stage residual composer.
    ///
    /// Returns `Err(SizeMismatch)` when `base.dim() != residual.dim()`.
    pub fn new(
        base: Box<dyn DecodeKernel>,
        residual: Box<dyn DecodeKernel>,
    ) -> Result<Self, DecodeError> {
        if base.dim() != residual.dim() {
            return Err(DecodeError::SizeMismatch {
                expected: base.dim() as usize,
                actual: residual.dim() as usize,
            });
        }
        Ok(Self { base, residual })
    }
}

impl DecodeKernel for ResidualComposer {
    fn decode(&self, bytes: &[u8]) -> Result<Vec<f32>, DecodeError> {
        let base_b = self.base.bytes_per_row() as usize;
        let expected = self.bytes_per_row() as usize;
        if bytes.len() != expected {
            return Err(DecodeError::SizeMismatch { expected, actual: bytes.len() });
        }
        let base_v = self
            .base
            .decode(&bytes[..base_b])
            .map_err(|e| DecodeError::Stage { stage: "base::decode", detail: e.to_string() })?;
        let residual_v = self
            .residual
            .decode(&bytes[base_b..])
            .map_err(|e| DecodeError::Stage { stage: "residual::decode", detail: e.to_string() })?;
        let mut out = base_v;
        for (dst, &r) in out.iter_mut().zip(&residual_v) {
            *dst += r;
        }
        Ok(out)
    }

    fn encode(&self, vec: &[f32]) -> Result<Vec<u8>, DecodeError> {
        let expected = self.dim() as usize;
        if vec.len() != expected {
            return Err(DecodeError::SizeMismatch { expected, actual: vec.len() });
        }
        // First-pass encode + its self-reconstruction.
        let base_bytes = self
            .base
            .encode(vec)
            .map_err(|e| DecodeError::Stage { stage: "base::encode", detail: e.to_string() })?;
        let base_reconstructed = self
            .base
            .decode(&base_bytes)
            .map_err(|e| DecodeError::Stage { stage: "base::decode", detail: e.to_string() })?;
        // Residual = original − base.decode(base.encode(original)).
        let residual_vec: Vec<f32> =
            vec.iter().zip(&base_reconstructed).map(|(a, b)| a - b).collect();
        let residual_bytes = self
            .residual
            .encode(&residual_vec)
            .map_err(|e| DecodeError::Stage { stage: "residual::encode", detail: e.to_string() })?;
        // Concat.
        let mut out = Vec::with_capacity(base_bytes.len() + residual_bytes.len());
        out.extend_from_slice(&base_bytes);
        out.extend_from_slice(&residual_bytes);
        Ok(out)
    }

    fn bytes_per_row(&self) -> u32 {
        self.base.bytes_per_row() + self.residual.bytes_per_row()
    }

    fn dim(&self) -> u32 { self.base.dim() }

    fn signature(&self) -> u64 {
        let mut h = DefaultHasher::new();
        "residual_compose".hash(&mut h);
        self.base.signature().hash(&mut h);
        self.residual.signature().hash(&mut h);
        h.finish()
    }

    fn backend(&self) -> &'static str {
        // Composer backend = the less-optimized stage's backend (weakest link
        // drives actual latency). For the stub case this is "stub"; for real
        // D1.1b kernels, the OPQ matmul stage dominates.
        if self.base.backend() == "stub" || self.residual.backend() == "stub" {
            "stub"
        } else {
            self.base.backend()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_round_trip_is_exact() {
        let k = StubDecodeKernel::new(8, 1);
        let v = vec![1.0_f32, -2.5, 3.25, 0.0, -7.875, 100.0, -0.125, 42.0];
        let enc = k.encode(&v).unwrap();
        assert_eq!(enc.len(), 32);
        let dec = k.decode(&enc).unwrap();
        assert_eq!(dec, v);
    }

    #[test]
    fn stub_rejects_wrong_input_size() {
        let k = StubDecodeKernel::new(4, 0);
        let err = k.encode(&[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, DecodeError::SizeMismatch { expected: 4, actual: 3 }));
        let err = k.decode(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, DecodeError::SizeMismatch { expected: 16, actual: 10 }));
    }

    #[test]
    fn residual_compose_round_trip_is_exact_when_both_stubs() {
        let base = Box::new(StubDecodeKernel::new(4, 1));
        let residual = Box::new(StubDecodeKernel::new(4, 2));
        let comp = ResidualComposer::new(base, residual).unwrap();

        let v = vec![1.5_f32, -2.0, 3.125, -0.5];
        let enc = comp.encode(&v).unwrap();
        assert_eq!(enc.len(), 32, "4 dim × 4 bytes × 2 stages = 32");
        let dec = comp.decode(&enc).unwrap();
        // Both stages byte-exact ⇒ residual is all zeros ⇒ decoded = base_reconstructed = v.
        assert_eq!(dec, v);
    }

    #[test]
    fn residual_compose_mismatched_dims_rejected() {
        let base = Box::new(StubDecodeKernel::new(4, 0));
        let residual = Box::new(StubDecodeKernel::new(8, 0));
        let err = ResidualComposer::new(base, residual).unwrap_err();
        assert!(matches!(err, DecodeError::SizeMismatch { expected: 4, actual: 8 }));
    }

    #[test]
    fn residual_compose_bytes_per_row_sums_stages() {
        let base = Box::new(StubDecodeKernel::new(6, 1));      // 24 bytes
        let residual = Box::new(StubDecodeKernel::new(6, 2));  // 24 bytes
        let comp = ResidualComposer::new(base, residual).unwrap();
        assert_eq!(comp.bytes_per_row(), 48);
        assert_eq!(comp.dim(), 6);
    }

    #[test]
    fn residual_compose_nested_depth_two_round_trip() {
        // depth=2: ResidualComposer whose `residual` slot is itself a ResidualComposer.
        let inner_base = Box::new(StubDecodeKernel::new(4, 1));
        let inner_residual = Box::new(StubDecodeKernel::new(4, 2));
        let inner = Box::new(ResidualComposer::new(inner_base, inner_residual).unwrap());
        let outer_base = Box::new(StubDecodeKernel::new(4, 3));
        let outer = ResidualComposer::new(outer_base, inner).unwrap();

        assert_eq!(outer.bytes_per_row(), 48, "4 dim × 4 bytes × 3 stages");
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let enc = outer.encode(&v).unwrap();
        let dec = outer.decode(&enc).unwrap();
        assert_eq!(dec, v);
    }

    #[test]
    fn signatures_distinguish_composer_from_stages() {
        let base = Box::new(StubDecodeKernel::new(4, 1));
        let residual = Box::new(StubDecodeKernel::new(4, 2));
        let s_base = base.signature();
        let s_res = residual.signature();
        let comp = ResidualComposer::new(base, residual).unwrap();
        let s_comp = comp.signature();
        assert_ne!(s_comp, s_base);
        assert_ne!(s_comp, s_res);
    }

    #[test]
    fn signature_depends_on_stage_order() {
        let a = Box::new(StubDecodeKernel::new(4, 1));
        let b = Box::new(StubDecodeKernel::new(4, 2));
        let ab = ResidualComposer::new(a, b).unwrap();
        let a2 = Box::new(StubDecodeKernel::new(4, 1));
        let b2 = Box::new(StubDecodeKernel::new(4, 2));
        let ba = ResidualComposer::new(b2, a2).unwrap();
        assert_ne!(ab.signature(), ba.signature(), "base/residual order is part of identity");
    }

    #[test]
    fn composer_backend_reports_stub_when_any_stage_is_stub() {
        let base = Box::new(StubDecodeKernel::new(4, 1));
        let residual = Box::new(StubDecodeKernel::new(4, 2));
        let comp = ResidualComposer::new(base, residual).unwrap();
        assert_eq!(comp.backend(), "stub");
    }
}
