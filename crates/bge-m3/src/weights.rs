//! BGE-M3 weight loading — stacked (SPD=32) and legacy (i16 bgz7).
//!
//! Two paths:
//! - `load_stacked_from_gguf()` — GGUF → f32 → StackedN, role-classified. Pearson 0.996.
//! - `load()` — legacy bgz7 i16. Kept for palette L1. Do not use for new work.

use ndarray::hpc::bgz17_bridge::Base17;
use bgz_tensor::stacked_n::StackedN;
use bgz_tensor::variance::Role;

pub const DEFAULT_BGZ7_PATH: &str = "/tmp/bge_m3_f16.bgz7";

// XLM-RoBERTa architecture constants (BGE-M3)
pub const VOCAB_SIZE: usize = 250002;
pub const HIDDEN_DIM: usize = 1024;
pub const NUM_LAYERS: usize = 24;
pub const NUM_HEADS: usize = 16;
pub const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 64
pub const MLP_DIM: usize = 4096;
pub const MAX_SEQ_LEN: usize = 8192;

pub const DEFAULT_SPD: usize = 32;

// ═══════════════════════════════════════════════════════════════════════════
// Stacked weights (production path)
// ═══════════════════════════════════════════════════════════════════════════

/// BGE-M3 weights at stacked resolution, role-separated.
pub struct BgeM3StackedWeights {
    pub layers: Vec<BgeM3LayerWeights>,
    pub embeddings: Vec<StackedN>,
    pub spd: usize,
    pub total_rows: usize,
}

/// One XLM-RoBERTa layer's weights.
pub struct BgeM3LayerWeights {
    pub q_proj: Vec<StackedN>,
    pub k_proj: Vec<StackedN>,
    pub v_proj: Vec<StackedN>,
    pub o_proj: Vec<StackedN>,
    pub gate_proj: Vec<StackedN>,
    pub up_proj: Vec<StackedN>,
    pub down_proj: Vec<StackedN>,
}

impl BgeM3LayerWeights {
    fn empty() -> Self {
        Self {
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

    pub fn role(&self, role: Role) -> &[StackedN] {
        match role {
            Role::Q => &self.q_proj, Role::K => &self.k_proj,
            Role::V => &self.v_proj, Role::Gate => &self.gate_proj,
            Role::Up => &self.up_proj, Role::Down => &self.down_proj,
        }
    }
}

impl BgeM3StackedWeights {
    /// Load from GGUF at stacked resolution.
    pub fn load_from_gguf(path: &str, spd: usize) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        // Reuse reader-lm's GGUF parser — same format
        Self::load_from_gguf_bytes(&data, spd)
    }

    pub fn load_from_gguf_bytes(data: &[u8], spd: usize) -> Result<Self, String> {
        use std::io::Cursor;
        let mut reader = Cursor::new(data);
        let header = crate::weights::parse_gguf_header(&mut reader)?;

        let mut layers: Vec<BgeM3LayerWeights> = (0..NUM_LAYERS).map(|_| BgeM3LayerWeights::empty()).collect();
        let mut embeddings = Vec::new();
        let mut total_rows = 0;

        for tensor in &header.tensors {
            if tensor.n_elements < 1024 { continue; }
            let f32_data = crate::weights::read_tensor_f32(&mut reader, &header, tensor)?;
            let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
                (tensor.dims[0] as usize, tensor.dims[1..].iter().map(|&d| d as usize).product())
            } else { (1, f32_data.len()) };

            let role = Role::from_name(&tensor.name);
            let layer_idx = crate::weights::parse_layer_idx(&tensor.name);

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
                    _ => { if tensor.name.contains("embed") { embeddings.push(stacked); } }
                }
            }
        }
        Ok(BgeM3StackedWeights { layers, embeddings, spd, total_rows })
    }

    pub fn hydrate_role(&self, layer: usize, role: Role) -> Vec<Vec<f32>> {
        if layer >= self.layers.len() { return Vec::new(); }
        self.layers[layer].role(role).iter().map(|s| s.hydrate_f32()).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Legacy i16 weights (bgz17 fallback)
// ═══════════════════════════════════════════════════════════════════════════

pub struct BgeM3Weights {
    pub tensors: Vec<(String, Vec<Base17>)>,
    pub total_rows: usize,
}

impl BgeM3Weights {
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
    pub fn load_default() -> Result<Self, String> { Self::load(DEFAULT_BGZ7_PATH) }

    pub fn embedding_rows(&self) -> Vec<&Base17> {
        self.tensors.iter().filter(|(n, _)| n.contains("embed") || n.contains("word"))
            .flat_map(|(_, r)| r.iter()).collect()
    }
    pub fn attention_rows(&self) -> Vec<&Base17> {
        self.tensors.iter().filter(|(n, _)| n.contains("attn") || n.contains("self"))
            .flat_map(|(_, r)| r.iter()).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GGUF reader — shared with reader-lm (re-export for bge-m3)
// ═══════════════════════════════════════════════════════════════════════════

// The GGUF reader lives in reader-lm/weights.rs. Since bge-m3 can't depend
// on reader-lm directly (sibling crate), we duplicate the minimal parser.
// TODO: extract to a shared gguf crate.

pub(crate) struct GgufHeader { pub tensors: Vec<TensorMeta>, pub data_offset: u64 }
pub(crate) struct TensorMeta { pub name: String, pub dims: Vec<u64>, pub dtype: u32, pub offset: u64, pub n_elements: u64 }

pub(crate) fn parse_layer_idx(name: &str) -> Option<u16> {
    let n = name.to_lowercase();
    if let Some(pos) = n.find("layers.") { n[pos + 7..].split('.').next().and_then(|s| s.parse().ok()) }
    else if let Some(pos) = n.find("blk.") { n[pos + 4..].split('.').next().and_then(|s| s.parse().ok()) }
    else { None }
}

pub(crate) fn parse_gguf_header<R: std::io::Read + std::io::Seek>(r: &mut R) -> Result<GgufHeader, String> {
    let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8];
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    if u32::from_le_bytes(b4) != 0x46554747 { return Err("bad GGUF magic".into()); }
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?; let nt = u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?; let nm = u64::from_le_bytes(b8) as usize;
    for _ in 0..nm { skip_kv(r)?; }
    let mut tensors = Vec::with_capacity(nt);
    for _ in 0..nt {
        r.read_exact(&mut b8).map_err(|e| e.to_string())?; let nl = u64::from_le_bytes(b8) as usize;
        let mut nb = vec![0u8; nl]; r.read_exact(&mut nb).map_err(|e| e.to_string())?;
        let name = String::from_utf8_lossy(&nb).to_string();
        r.read_exact(&mut b4).map_err(|e| e.to_string())?; let nd = u32::from_le_bytes(b4) as usize;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd { r.read_exact(&mut b8).map_err(|e| e.to_string())?; dims.push(u64::from_le_bytes(b8)); }
        r.read_exact(&mut b4).map_err(|e| e.to_string())?; let dtype = u32::from_le_bytes(b4);
        r.read_exact(&mut b8).map_err(|e| e.to_string())?; let offset = u64::from_le_bytes(b8);
        tensors.push(TensorMeta { name, dims: dims.clone(), dtype, offset, n_elements: dims.iter().product() });
    }
    let pos = r.stream_position().map_err(|e| e.to_string())?;
    Ok(GgufHeader { tensors, data_offset: (pos + 31) / 32 * 32 })
}

fn skip_kv<R: std::io::Read + std::io::Seek>(r: &mut R) -> Result<(), String> {
    let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8];
    r.read_exact(&mut b8).map_err(|e| e.to_string())?; let kl = u64::from_le_bytes(b8) as usize;
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

pub(crate) fn read_tensor_f32<R: std::io::Read + std::io::Seek>(r: &mut R, h: &GgufHeader, t: &TensorMeta) -> Result<Vec<f32>, String> {
    use std::io::SeekFrom;
    r.seek(SeekFrom::Start(h.data_offset + t.offset)).map_err(|e| e.to_string())?;
    let n = t.n_elements as usize;
    match t.dtype {
        0 => { let mut buf = vec![0u8; n*4]; r.read_exact(&mut buf).map_err(|e| e.to_string())?; Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()) }
        8 => { let nb=(n+31)/32; let bpb=34; let mut buf=vec![0u8;nb*bpb]; r.read_exact(&mut buf).map_err(|e|e.to_string())?;
            let mut res=Vec::with_capacity(n); for b in 0..nb { let o=b*bpb; let sb=u16::from_le_bytes([buf[o],buf[o+1]]); let s=f32::from_bits((sb as u32)<<16);
            for i in 0..32 { if res.len()>=n{break;} res.push(buf[o+2+i] as i8 as f32 * s); } } Ok(res) }
        30 => { let mut buf=vec![0u8;n*2]; r.read_exact(&mut buf).map_err(|e|e.to_string())?; Ok(buf.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0],c[1]]) as u32)<<16)).collect()) }
        _ => Err(format!("unsupported dtype {} for {}", t.dtype, t.name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() { assert_eq!(HEAD_DIM, 64); }

    #[test]
    fn test_role_parse_xlm_roberta() {
        // XLM-RoBERTa uses different naming than Llama
        assert_eq!(Role::from_name("encoder.layer.5.attention.self.query"), None); // doesn't match q_proj
        // But standard names work
        assert_eq!(Role::from_name("model.layers.5.self_attn.q_proj.weight"), Some(Role::Q));
    }

    #[test]
    #[ignore = "requires: /tmp/bge_m3_f16.bgz7"]
    fn test_load_legacy() {
        let w = BgeM3Weights::load_default().unwrap();
        eprintln!("BGE-M3 legacy: {} tensors, {} rows", w.tensors.len(), w.total_rows);
    }
}
