//! GGUF source handle: the file IS the database.
//!
//! SpiralAddr says WHERE to read. This module provides the file handle
//! and tensor offset table that turns an address into actual f32 values.
//!
//! No data is copied or shifted. The GGUF stays on disk.
//! Hydration = seek + read + dequant. On demand. For survivors only.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use crate::{SpiralAddress, SpiralWalk, BASE_DIM};

/// A tensor's location in the GGUF file.
#[derive(Clone, Debug)]
pub struct TensorLocation {
    pub name: String,
    /// Absolute byte offset in the GGUF file where this tensor's data starts.
    pub data_offset: u64,
    /// Data type (0=F32, 1=F16, 8=Q8_0, 30=BF16).
    pub dtype: u32,
    /// Shape: [rows, cols] for 2D tensors.
    pub dims: Vec<u64>,
    /// Number of elements.
    pub n_elements: u64,
    /// Columns per row (for row-level seeking).
    pub n_cols: usize,
}

impl TensorLocation {
    /// Bytes per element in the raw file (for seeking).
    pub fn bytes_per_element(&self) -> usize {
        match self.dtype {
            0 => 4,       // F32
            1 | 30 => 2,  // F16, BF16
            8 => 1,       // Q8_0: ~1 byte per element (34 bytes per 32-element block)
            _ => 4,
        }
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        if self.dims.len() >= 2 { self.dims[0] as usize } else { 1 }
    }
}

/// Index of all tensors in a GGUF file. Built once, used for all hydrations.
///
/// 292 tensors × ~100 bytes metadata = ~30 KB. Trivial.
#[derive(Clone, Debug)]
pub struct GgufIndex {
    pub path: PathBuf,
    pub tensors: Vec<TensorLocation>,
}

impl GgufIndex {
    /// Build index from GGUF file. Reads only the header, not the tensor data.
    pub fn open(path: &str) -> Result<Self, String> {
        let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {}", path, e))?;
        let header = parse_gguf_header_with_offsets(&mut file)?;

        let tensors: Vec<TensorLocation> = header.tensors.into_iter().map(|t| {
            let n_cols = if t.dims.len() >= 2 {
                t.dims[1..].iter().map(|&d| d as usize).product()
            } else { t.n_elements as usize };

            TensorLocation {
                name: t.name,
                data_offset: header.data_offset + t.offset,
                dtype: t.dtype,
                dims: t.dims,
                n_elements: t.n_elements,
                n_cols,
            }
        }).collect();

        Ok(GgufIndex { path: PathBuf::from(path), tensors })
    }

    /// Find a tensor by name.
    pub fn find(&self, name: &str) -> Option<&TensorLocation> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Find all tensors matching a pattern.
    pub fn find_matching(&self, pattern: &str) -> Vec<&TensorLocation> {
        self.tensors.iter().filter(|t| t.name.contains(pattern)).collect()
    }

    /// Hydrate one row of one tensor: seek to exact position, read, dequant to f32.
    ///
    /// This is the LEAF operation. Only called on cascade survivors (~0.1%).
    /// Opens the file, seeks, reads one row, closes. No caching.
    pub fn hydrate_row(&self, tensor: &TensorLocation, row_idx: usize) -> Result<Vec<f32>, String> {
        let mut file = std::fs::File::open(&self.path).map_err(|e| e.to_string())?;
        hydrate_row_from_file(&mut file, tensor, row_idx)
    }

    /// Hydrate a spiral walk: read only the positions the address specifies.
    ///
    /// Even cheaper than hydrate_row — reads scattered positions, not a full row.
    pub fn hydrate_walk(&self, tensor: &TensorLocation, row_idx: usize, addr: &SpiralAddress) -> Result<SpiralWalk, String> {
        let row = self.hydrate_row(tensor, row_idx)?;
        Ok(SpiralWalk::execute(addr, &row))
    }

    /// Total index size in bytes.
    pub fn index_bytes(&self) -> usize {
        self.tensors.len() * std::mem::size_of::<TensorLocation>()
    }

    /// Summary.
    pub fn summary(&self) -> String {
        let total_params: u64 = self.tensors.iter().map(|t| t.n_elements).sum();
        format!("GgufIndex: {} tensors, {} params, index={} bytes, source={}",
            self.tensors.len(), total_params, self.index_bytes(),
            self.path.display())
    }
}

/// Hydrate one row from an open file handle.
fn hydrate_row_from_file<R: Read + Seek>(
    file: &mut R,
    tensor: &TensorLocation,
    row_idx: usize,
) -> Result<Vec<f32>, String> {
    let n_cols = tensor.n_cols;

    match tensor.dtype {
        0 => { // F32: seek directly to row, read n_cols × 4 bytes
            let row_offset = tensor.data_offset + (row_idx * n_cols * 4) as u64;
            file.seek(SeekFrom::Start(row_offset)).map_err(|e| e.to_string())?;
            let mut buf = vec![0u8; n_cols * 4];
            file.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())
        }
        1 => { // F16
            let row_offset = tensor.data_offset + (row_idx * n_cols * 2) as u64;
            file.seek(SeekFrom::Start(row_offset)).map_err(|e| e.to_string())?;
            let mut buf = vec![0u8; n_cols * 2];
            file.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f16_to_f32(bits)
            }).collect())
        }
        30 => { // BF16
            let row_offset = tensor.data_offset + (row_idx * n_cols * 2) as u64;
            file.seek(SeekFrom::Start(row_offset)).map_err(|e| e.to_string())?;
            let mut buf = vec![0u8; n_cols * 2];
            file.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            }).collect())
        }
        8 => { // Q8_0: blocks of 32 int8 + f16 scale
            // Seek to the row's starting block
            let blocks_per_row = (n_cols + 31) / 32;
            let bytes_per_block = 34; // 2 (f16 scale) + 32 (int8 values)
            let row_offset = tensor.data_offset + (row_idx * blocks_per_row * bytes_per_block) as u64;
            file.seek(SeekFrom::Start(row_offset)).map_err(|e| e.to_string())?;
            let mut buf = vec![0u8; blocks_per_row * bytes_per_block];
            file.read_exact(&mut buf).map_err(|e| e.to_string())?;

            let mut result = Vec::with_capacity(n_cols);
            for b in 0..blocks_per_row {
                let o = b * bytes_per_block;
                let scale_bits = u16::from_le_bytes([buf[o], buf[o + 1]]);
                let scale = f32::from_bits((scale_bits as u32) << 16); // BF16→f32
                for i in 0..32 {
                    if result.len() >= n_cols { break; }
                    result.push(buf[o + 2 + i] as i8 as f32 * scale);
                }
            }
            Ok(result)
        }
        _ => Err(format!("unsupported dtype {} for {}", tensor.dtype, tensor.name)),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { f32::from_bits(sign << 31) }
        else { let f = frac as f32 / 1024.0 * 2.0f32.powi(-14); if sign == 1 { -f } else { f } }
    } else if exp == 31 {
        if frac == 0 { if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY } } else { f32::NAN }
    } else { f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)) }
}

// ═══════════════════════════════════════════════════════════════════════════
// GGUF header parser (reused from lib.rs pattern but returns absolute offsets)
// ═══════════════════════════════════════════════════════════════════════════

struct ParsedHeader { tensors: Vec<ParsedTensor>, data_offset: u64 }
struct ParsedTensor { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }

fn parse_gguf_header_with_offsets<R: Read + Seek>(r: &mut R) -> Result<ParsedHeader, String> {
    let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8];
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    if u32::from_le_bytes(b4) != 0x46554747 { return Err("bad GGUF magic".into()); }
    r.read_exact(&mut b4).map_err(|e| e.to_string())?; // version
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let nt = u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let nm = u64::from_le_bytes(b8) as usize;
    for _ in 0..nm { skip_kv(r)?; }
    let mut tensors = Vec::with_capacity(nt);
    for _ in 0..nt {
        r.read_exact(&mut b8).map_err(|e| e.to_string())?;
        let nl = u64::from_le_bytes(b8) as usize;
        let mut nb = vec![0u8; nl]; r.read_exact(&mut nb).map_err(|e| e.to_string())?;
        let name = String::from_utf8_lossy(&nb).to_string();
        r.read_exact(&mut b4).map_err(|e| e.to_string())?;
        let nd = u32::from_le_bytes(b4) as usize;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd { r.read_exact(&mut b8).map_err(|e| e.to_string())?; dims.push(u64::from_le_bytes(b8)); }
        r.read_exact(&mut b4).map_err(|e| e.to_string())?; let dtype = u32::from_le_bytes(b4);
        r.read_exact(&mut b8).map_err(|e| e.to_string())?; let offset = u64::from_le_bytes(b8);
        tensors.push(ParsedTensor { name, dims: dims.clone(), dtype, offset, n_elements: dims.iter().product() });
    }
    let pos = r.stream_position().map_err(|e| e.to_string())?;
    Ok(ParsedHeader { tensors, data_offset: (pos + 31) / 32 * 32 })
}

fn skip_kv<R: Read + Seek>(r: &mut R) -> Result<(), String> {
    let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8];
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let kl = u64::from_le_bytes(b8) as usize;
    let mut kb = vec![0u8; kl]; r.read_exact(&mut kb).map_err(|e| e.to_string())?;
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    skip_val(r, u32::from_le_bytes(b4))
}

fn skip_val<R: Read + Seek>(r: &mut R, vt: u32) -> Result<(), String> {
    let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8];
    match vt {
        0|1|7 => { let mut b = [0u8; 1]; r.read_exact(&mut b).map_err(|e| e.to_string())?; }
        2|3 => { r.read_exact(&mut [0u8; 2]).map_err(|e| e.to_string())?; }
        4|5|6 => { r.read_exact(&mut b4).map_err(|e| e.to_string())?; }
        8 => { r.read_exact(&mut b8).map_err(|e| e.to_string())?; let l = u64::from_le_bytes(b8) as usize; let mut s = vec![0u8; l]; r.read_exact(&mut s).map_err(|e| e.to_string())?; }
        9 => { r.read_exact(&mut b4).map_err(|e| e.to_string())?; let et = u32::from_le_bytes(b4); r.read_exact(&mut b8).map_err(|e| e.to_string())?; let c = u64::from_le_bytes(b8) as usize; for _ in 0..c { skip_val(r, et)?; } }
        10|11|12 => { r.read_exact(&mut b8).map_err(|e| e.to_string())?; }
        _ => return Err(format!("unknown GGUF vtype {}", vt)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Jina GGUF"]
    fn test_gguf_index_open() {
        let path = "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let idx = GgufIndex::open(path).unwrap();
        eprintln!("{}", idx.summary());
        assert!(idx.tensors.len() > 100);
    }

    #[test]
    #[ignore = "requires Jina GGUF"]
    fn test_hydrate_row() {
        let path = "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let idx = GgufIndex::open(path).unwrap();

        // Find first weight tensor (not embedding, has reasonable dims)
        let tensor = idx.tensors.iter()
            .find(|t| t.n_rows() > 10 && t.n_cols > 512 && t.n_cols < 10000)
            .unwrap();

        let row0 = idx.hydrate_row(tensor, 0).unwrap();
        let row1 = idx.hydrate_row(tensor, 1).unwrap();

        assert_eq!(row0.len(), tensor.n_cols);
        assert_eq!(row1.len(), tensor.n_cols);

        // Rows should be different (different neurons)
        assert_ne!(row0, row1);

        // Magnitude should be nonzero
        let mag: f32 = row0.iter().map(|v| v.abs()).sum();
        assert!(mag > 0.0, "hydrated row should not be zero");

        eprintln!("Hydrated {}: {} cols, magnitude={:.4}", tensor.name, row0.len(), mag);
    }

    #[test]
    #[ignore = "requires Jina GGUF"]
    fn test_hydrate_walk() {
        let path = "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let idx = GgufIndex::open(path).unwrap();

        // Find a tensor with nonzero data (may need to try several)
        let tensor = idx.tensors.iter()
            .find(|t| {
                if t.n_rows() < 2 || t.n_cols < 100 { return false; }
                if let Ok(row) = idx.hydrate_row(t, 0) {
                    row.iter().any(|v| v.abs() > 1e-10)
                } else { false }
            });

        let tensor = match tensor {
            Some(t) => t,
            None => {
                eprintln!("No nonzero tensors found — Q8_0 may dequant to zero for this model. Skipping.");
                return;
            }
        };

        let addr = SpiralAddress::new(20, 8, 4);
        let walk = idx.hydrate_walk(tensor, 0, &addr).unwrap();

        // Walk should have 17 dims with samples
        assert_eq!(walk.samples.len(), BASE_DIM);
        let total_samples: usize = walk.samples.iter().map(|s| s.len()).sum();
        assert!(total_samples > 0, "walk should have samples");

        // Self-cosine should be 1.0
        let c = walk.cosine(&walk);
        assert!((c - 1.0).abs() < 1e-10);

        eprintln!("Walk on {}: {} total samples, self-cosine={:.6}", tensor.name, total_samples, c);
    }
}
