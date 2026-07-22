//! `codebook` — the LE loader for the TRAINED Cam96 codebook + word codes.
//!
//! This retires the `demo()` placeholder for real use: the artifacts in
//! `data/` were produced from real Jina-v3 96-d embeddings of the KJV Bible
//! vocabulary (12,543 words), k-means-256 per axis (see `probes/README.md`
//! § trained codebook for the held-out measurement: 96-bit ρ 0.774 vs the
//! 48-bit point's 0.617 on 299,882 held-out pairs).
//!
//! ## Formats (little-endian, this crate's own — no serde)
//!
//! `cam96_codebook.bin`: magic `CAM96CB1` · `u32` n_axes (12) · `u32` dim ·
//! `u32` n_centroids (≤256) · `f32` d_max · then `n_axes × n_centroids × dim`
//! `f32` centroid data in axis-major order.
//!
//! `cam96_codes.bin`: magic `CAM96WD1` · `u32` n_words · then `n_words × 12`
//! bytes, one [`Cam96`] per word, in vocabulary order (pair with the
//! frequency-ranked vocab list so `codes[id]` matches [`crate::vocab::WordId`]).

use crate::space::{Cam96, Cam96Space};

/// Loader error — which structural check failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodebookError {
    /// Wrong or missing magic bytes.
    BadMagic,
    /// Header fields inconsistent (n_axes ≠ 12, zero dim, >256 centroids).
    BadHeader,
    /// Byte length does not match the header's promised payload.
    Truncated,
}

fn read_u32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// Parse a `CAM96CB1` codebook blob into a [`Cam96Space`].
///
/// # Errors
/// [`CodebookError`] naming the failed structural check; never panics on
/// malformed input.
pub fn load_cam96_space(bytes: &[u8]) -> Result<Cam96Space, CodebookError> {
    if bytes.get(..8) != Some(b"CAM96CB1".as_slice()) {
        return Err(CodebookError::BadMagic);
    }
    let n_axes = read_u32(bytes, 8).ok_or(CodebookError::Truncated)? as usize;
    let dim = read_u32(bytes, 12).ok_or(CodebookError::Truncated)? as usize;
    let n_cent = read_u32(bytes, 16).ok_or(CodebookError::Truncated)? as usize;
    let d_max = f32::from_le_bytes(
        bytes
            .get(20..24)
            .ok_or(CodebookError::Truncated)?
            .try_into()
            .map_err(|_| CodebookError::Truncated)?,
    );
    if n_axes != 12 || dim == 0 || n_cent == 0 || n_cent > 256 {
        return Err(CodebookError::BadHeader);
    }
    let payload = &bytes[24..];
    let expect = n_axes * n_cent * dim * 4;
    if payload.len() < expect {
        return Err(CodebookError::Truncated);
    }
    let mut axes = Vec::with_capacity(n_axes);
    let mut off = 0;
    for _ in 0..n_axes {
        let mut axis = Vec::with_capacity(n_cent);
        for _ in 0..n_cent {
            let c: Vec<f32> = (0..dim)
                .map(|d| {
                    f32::from_le_bytes(payload[off + d * 4..off + d * 4 + 4].try_into().unwrap())
                })
                .collect();
            off += dim * 4;
            axis.push(c);
        }
        axes.push(axis);
    }
    Ok(Cam96Space::from_axis_codebooks(axes, d_max))
}

/// Parse a `CAM96WD1` codes blob into per-word [`Cam96`] codes
/// (`codes[id]` in vocabulary order).
///
/// # Errors
/// [`CodebookError`] naming the failed structural check.
pub fn load_cam96_codes(bytes: &[u8]) -> Result<Vec<Cam96>, CodebookError> {
    if bytes.get(..8) != Some(b"CAM96WD1".as_slice()) {
        return Err(CodebookError::BadMagic);
    }
    let n = read_u32(bytes, 8).ok_or(CodebookError::Truncated)? as usize;
    let payload = &bytes[12..];
    if payload.len() < n * 12 {
        return Err(CodebookError::Truncated);
    }
    Ok((0..n)
        .map(|i| {
            let mut c = [0u8; 12];
            c.copy_from_slice(&payload[i * 12..i * 12 + 12]);
            c
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_codebook_blob(dim: usize, n_cent: usize) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"CAM96CB1");
        b.extend_from_slice(&12u32.to_le_bytes());
        b.extend_from_slice(&(dim as u32).to_le_bytes());
        b.extend_from_slice(&(n_cent as u32).to_le_bytes());
        b.extend_from_slice(&8.0f32.to_le_bytes());
        for a in 0..12 {
            for c in 0..n_cent {
                for d in 0..dim {
                    b.extend_from_slice(&((a + c + d) as f32).to_le_bytes());
                }
            }
        }
        b
    }

    #[test]
    fn codebook_round_trips() {
        let blob = tiny_codebook_blob(3, 4);
        let space = load_cam96_space(&blob).unwrap();
        assert_eq!(space.axis_dim(), 3);
        // encode a vector made of axis-k centroid 2 → recovers index 2 per axis.
        let v: Vec<f32> = (0..12)
            .flat_map(|a| (0..3).map(move |d| (a + 2 + d) as f32))
            .collect();
        assert_eq!(space.encode(&v), [2u8; 12]);
    }

    #[test]
    fn codes_round_trip_and_reject_garbage() {
        let mut b = Vec::new();
        b.extend_from_slice(b"CAM96WD1");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&[1u8; 12]);
        b.extend_from_slice(&[7u8; 12]);
        let codes = load_cam96_codes(&b).unwrap();
        assert_eq!(codes, vec![[1u8; 12], [7u8; 12]]);
        assert_eq!(load_cam96_codes(b"nope"), Err(CodebookError::BadMagic));
        assert!(matches!(
            load_cam96_space(&tiny_codebook_blob(3, 4)[..30]),
            Err(CodebookError::Truncated)
        ));
    }
}
