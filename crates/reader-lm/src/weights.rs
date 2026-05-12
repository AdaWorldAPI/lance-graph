//! Reader LM weight loading — stacked (SPD=32) and legacy (i16 bgz7).
//!
//! Two paths:
//! - `load_stacked_from_gguf()` — reads GGUF, dequants to f32, encodes as StackedN.
//!   Role-classified (Q/K/V/Gate/Up/Down). SPD=32 = Pearson 0.996.
//! - `load()` — legacy bgz7 i16 path. Kept for palette L1 operations.
//!   OpenChat = ALL ZEROS at i16. Llama4 = MARGINAL. Do not use for new work.

use ndarray::hpc::bgz17_bridge::Base17;
use bgz_tensor::stacked_n::StackedN;
use bgz_tensor::variance::Role;

pub const DEFAULT_BGZ7_PATH: &str = "/tmp/reader_lm_1_5b.bgz7";

// Qwen2-1.5B architecture constants
pub const VOCAB_SIZE: usize = 151936;
pub const HIDDEN_DIM: usize = 1536;
pub const NUM_LAYERS: usize = 28;
pub const NUM_HEADS: usize = 12;
pub const NUM_KV_HEADS: usize = 2; // GQA: 12 query heads, 2 KV heads
pub const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 128
pub const MLP_DIM: usize = 8960; // SwiGLU intermediate
pub const MAX_SEQ_LEN: usize = 32768;

/// Default samples per dimension for stacked encoding.
pub const DEFAULT_SPD: usize = 32;

// ═══════════════════════════════════════════════════════════════════════════
// Stacked weights — the production path (SPD=32, Pearson 0.996)
// ═══════════════════════════════════════════════════════════════════════════

/// Reader LM weights at stacked resolution, role-separated.
pub struct ReaderLmStackedWeights {
    /// Per-layer, per-role weight rows.
    /// layers[layer_idx].role contains Vec<StackedN> for that layer+role.
    pub layers: Vec<LayerWeights>,
    /// Embedding table (not role-classified).
    pub embeddings: Vec<StackedN>,
    /// Samples per dimension.
    pub spd: usize,
    /// Total weight rows across all layers.
    pub total_rows: usize,
}

/// One transformer layer's weights, role-separated.
pub struct LayerWeights {
    pub q_proj: Vec<StackedN>,
    pub k_proj: Vec<StackedN>,
    pub v_proj: Vec<StackedN>,
    pub o_proj: Vec<StackedN>,
    pub gate_proj: Vec<StackedN>,
    pub up_proj: Vec<StackedN>,
    pub down_proj: Vec<StackedN>,
}

impl LayerWeights {
    fn empty() -> Self {
        LayerWeights {
            q_proj: Vec::new(), k_proj: Vec::new(), v_proj: Vec::new(),
            o_proj: Vec::new(), gate_proj: Vec::new(), up_proj: Vec::new(),
            down_proj: Vec::new(),
        }
    }

    fn push(&mut self, role: Role, row: StackedN) {
        match role {
            Role::Q => self.q_proj.push(row),
            Role::K => self.k_proj.push(row),
            Role::V => self.v_proj.push(row),
            Role::Gate => self.gate_proj.push(row),
            Role::Up => self.up_proj.push(row),
            Role::Down => self.down_proj.push(row),
        }
    }

    /// Get rows for a specific role.
    pub fn role(&self, role: Role) -> &[StackedN] {
        match role {
            Role::Q => &self.q_proj,
            Role::K => &self.k_proj,
            Role::V => &self.v_proj,
            Role::Gate => &self.gate_proj,
            Role::Up => &self.up_proj,
            Role::Down => &self.down_proj,
        }
    }
}

impl ReaderLmStackedWeights {
    /// Load from GGUF at stacked resolution.
    ///
    /// Reads GGUF, dequantizes each tensor to f32, classifies by role,
    /// encodes as StackedN at given SPD. No BF16→i16 mush path.
    pub fn load_from_gguf(path: &str, spd: usize) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        Self::load_from_gguf_bytes(&data, spd)
    }

    /// Load from GGUF bytes (for embedded/downloaded models).
    pub fn load_from_gguf_bytes(data: &[u8], spd: usize) -> Result<Self, String> {
        use std::io::{Cursor, Read, Seek, SeekFrom};

        let mut reader = Cursor::new(data);
        let header = parse_gguf_header(&mut reader)?;

        let mut layers: Vec<LayerWeights> = (0..NUM_LAYERS).map(|_| LayerWeights::empty()).collect();
        let mut embeddings = Vec::new();
        let mut total_rows = 0;

        for tensor in &header.tensors {
            if tensor.n_elements < 1024 { continue; }

            let f32_data = read_tensor_f32(&mut reader, &header, tensor)?;
            let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
                (tensor.dims[0] as usize, tensor.dims[1..].iter().map(|&d| d as usize).product())
            } else {
                (1, f32_data.len())
            };

            let role = Role::from_name(&tensor.name);
            let layer_idx = parse_layer_idx(&tensor.name);

            for r in 0..n_rows {
                let start = r * n_cols;
                let end = (start + n_cols).min(f32_data.len());
                if end <= start { continue; }

                let stacked = StackedN::from_f32(&f32_data[start..end], spd);
                total_rows += 1;

                match (role, layer_idx) {
                    (Some(role), Some(li)) if (li as usize) < layers.len() => {
                        layers[li as usize].push(role, stacked);
                    }
                    _ => {
                        if tensor.name.contains("embed") {
                            embeddings.push(stacked);
                        }
                    }
                }
            }
        }

        Ok(ReaderLmStackedWeights { layers, embeddings, spd, total_rows })
    }

    /// Hydrate a specific role from a specific layer to f32 vectors.
    ///
    /// This is the "connect the dots" path: hydrate before comparing.
    /// Returns full-precision f32 vectors for cosine/distance computation.
    pub fn hydrate_role(&self, layer: usize, role: Role) -> Vec<Vec<f32>> {
        if layer >= self.layers.len() { return Vec::new(); }
        self.layers[layer].role(role).iter()
            .map(|s| s.hydrate_f32())
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Legacy i16 weights — kept for palette L1 operations (bgz17 fallback)
// ═══════════════════════════════════════════════════════════════════════════

/// Legacy weight index from bgz7 (i16 Base17).
/// Kept for palette L1 distance operations where i16 is sufficient.
/// DO NOT use for cosine comparison — OpenChat = ALL ZEROS at this resolution.
pub struct ReaderLmWeights {
    pub tensors: Vec<(String, Vec<Base17>)>,
    pub total_rows: usize,
}

impl ReaderLmWeights {
    pub fn load(path: &str) -> Result<Self, String> {
        let compressed = ndarray::hpc::gguf_indexer::read_bgz7_file(path)?;
        let mut tensors = Vec::new();
        let mut total_rows = 0;
        for ct in compressed {
            total_rows += ct.rows.len();
            tensors.push((ct.name, ct.rows));
        }
        Ok(Self { tensors, total_rows })
    }

    pub fn load_default() -> Result<Self, String> {
        Self::load(DEFAULT_BGZ7_PATH)
    }

    pub fn all_rows(&self) -> Vec<&Base17> {
        self.tensors.iter().flat_map(|(_, rows)| rows.iter()).collect()
    }

    pub fn q_proj_rows(&self) -> Vec<&Base17> { self.rows_matching("q_proj") }
    pub fn k_proj_rows(&self) -> Vec<&Base17> { self.rows_matching("k_proj") }
    pub fn v_proj_rows(&self) -> Vec<&Base17> { self.rows_matching("v_proj") }
    pub fn gate_proj_rows(&self) -> Vec<&Base17> { self.rows_matching("gate_proj") }

    fn rows_matching(&self, pattern: &str) -> Vec<&Base17> {
        self.tensors.iter()
            .filter(|(name, _)| name.contains(pattern))
            .flat_map(|(_, rows)| rows.iter())
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Minimal GGUF reader (self-contained, same as bgz-tensor examples)
// ═══════════════════════════════════════════════════════════════════════════

struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }

fn parse_layer_idx(name: &str) -> Option<u16> {
    let n = name.to_lowercase();
    if let Some(pos) = n.find("layers.") {
        n[pos + 7..].split('.').next().and_then(|s| s.parse().ok())
    } else if let Some(pos) = n.find("blk.") {
        n[pos + 4..].split('.').next().and_then(|s| s.parse().ok())
    } else { None }
}

fn parse_gguf_header<R: std::io::Read + std::io::Seek>(r: &mut R) -> Result<GgufHeader, String> {
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
        let ne: u64 = dims.iter().product();
        tensors.push(TensorMeta { name, dims: dims.clone(), dtype, offset, n_elements: ne });
    }
    let pos = r.stream_position().map_err(|e| e.to_string())?;
    Ok(GgufHeader { tensors, data_offset: (pos + 31) / 32 * 32 })
}

fn skip_kv<R: std::io::Read + std::io::Seek>(r: &mut R) -> Result<(), String> {
    let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8];
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let kl = u64::from_le_bytes(b8) as usize;
    let mut kb = vec![0u8; kl]; r.read_exact(&mut kb).map_err(|e| e.to_string())?;
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    skip_val(r, u32::from_le_bytes(b4))
}

fn skip_val<R: std::io::Read + std::io::Seek>(r: &mut R, vt: u32) -> Result<(), String> {
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

fn read_tensor_f32<R: std::io::Read + std::io::Seek>(r: &mut R, h: &GgufHeader, t: &TensorMeta) -> Result<Vec<f32>, String> {
    use std::io::SeekFrom;
    r.seek(SeekFrom::Start(h.data_offset + t.offset)).map_err(|e| e.to_string())?;
    let n = t.n_elements as usize;
    match t.dtype {
        0 => { // F32
            let mut buf = vec![0u8; n * 4]; r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())
        }
        1 => { // F16
            let mut buf = vec![0u8; n * 2]; r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                let sign = ((bits >> 15) & 1) as u32;
                let exp = ((bits >> 10) & 0x1F) as u32;
                let frac = (bits & 0x3FF) as u32;
                if exp == 0 { if frac == 0 { f32::from_bits(sign << 31) } else { let f = frac as f32 / 1024.0 * 2.0f32.powi(-14); if sign == 1 { -f } else { f } } }
                else if exp == 31 { if frac == 0 { if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY } } else { f32::NAN } }
                else { f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)) }
            }).collect())
        }
        8 => { // Q8_0
            let nb = (n + 31) / 32; let bpb = 34;
            let mut buf = vec![0u8; nb * bpb]; r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            let mut res = Vec::with_capacity(n);
            for b in 0..nb {
                let o = b * bpb;
                let sb = u16::from_le_bytes([buf[o], buf[o+1]]);
                let s = f32::from_bits((sb as u32) << 16); // BF16→f32
                for i in 0..32 { if res.len() >= n { break; } res.push(buf[o+2+i] as i8 as f32 * s); }
            }
            Ok(res)
        }
        30 => { // BF16
            let mut buf = vec![0u8; n * 2]; r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            }).collect())
        }
        _ => Err(format!("unsupported dtype {} for {}", t.dtype, t.name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_constants() {
        assert_eq!(HEAD_DIM, 128);
        assert_eq!(NUM_HEADS / NUM_KV_HEADS, 6); // GQA ratio
    }

    #[test]
    fn test_role_parse() {
        assert_eq!(Role::from_name("model.layers.5.self_attn.q_proj.weight"), Some(Role::Q));
        assert_eq!(Role::from_name("model.layers.5.mlp.gate_proj.weight"), Some(Role::Gate));
        assert_eq!(Role::from_name("model.embed_tokens.weight"), None);
    }

    #[test]
    fn test_layer_idx_parse() {
        assert_eq!(parse_layer_idx("model.layers.15.self_attn.q_proj.weight"), Some(15));
        assert_eq!(parse_layer_idx("model.embed_tokens.weight"), None);
    }

    #[test]
    #[ignore = "requires GGUF model file"]
    fn test_load_stacked_from_gguf() {
        let path = "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let w = ReaderLmStackedWeights::load_from_gguf(path, 32).unwrap();
        eprintln!("Loaded: {} total rows, {} layers", w.total_rows, w.layers.len());
        // Verify Gate rows are nonzero (unlike i16 path where they'd be zero)
        for (li, layer) in w.layers.iter().enumerate().take(3) {
            for role in Role::ALL {
                let rows = layer.role(role);
                if !rows.is_empty() {
                    let hydrated = rows[0].hydrate_f32();
                    let mag: f64 = hydrated.iter().map(|x| x.abs() as f64).sum();
                    eprintln!("  Layer {} {}: {} rows, magnitude={:.4}", li, role.label(), rows.len(), mag);
                    assert!(mag > 0.0, "Layer {} {} should not be zero at stacked resolution", li, role.label());
                }
            }
        }
    }

    #[test]
    #[ignore = "requires: /tmp/reader_lm_1_5b.bgz7"]
    fn test_load_legacy() {
        let w = ReaderLmWeights::load_default().unwrap();
        assert!(w.total_rows > 0);
    }
}
