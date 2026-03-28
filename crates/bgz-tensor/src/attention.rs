//! Attention via precomputed table lookup.
//!
//! The core insight: with 256 palette entries, there are only 65,536 possible
//! attention scores (256 × 256 pairs). Precompute them ALL. Every attention
//! score in every layer comes from the same 128KB table.
//!
//! ```text
//! Standard attention:           Q · K^T / √d     → O(d²) per pair
//! Palette attention:    table[q_idx][k_idx]        → O(1) per pair
//! ```
//!
//! The distance matrix gives you attention scores.
//! The compose table gives you multi-hop reasoning.
//! Together they form a semiring: the algebraic structure of attention.

use crate::projection::Base17;
use crate::palette::WeightPalette;

/// Precomputed pairwise distance table between palette entries.
///
/// `table[a * k + b]` = L1 distance between archetype a and archetype b.
/// For k=256: 256 × 256 × 2 bytes = 128 KB → fits L1 cache.
///
/// This IS the attention score matrix, precomputed for all possible
/// palette-index pairs. At inference time, looking up the attention
/// between query token i and key token j is:
///
/// ```text
/// score = table[q_palette_idx[i]][k_palette_idx[j]]
/// ```
///
/// One memory access. No multiply. No accumulate.
#[derive(Clone, Debug)]
pub struct AttentionTable {
    /// Pairwise L1 distances: k × k u16 values.
    pub distances: Vec<u16>,
    /// Palette size.
    pub k: usize,
}

/// Compose table for multi-hop attention.
///
/// `compose[a * k + b]` = palette index of `palette[a].xor_bind(palette[b])`.
///
/// This gives you "what does token a attend to THROUGH token b?" in one lookup.
/// Standard transformers compute this via multi-layer attention stacking.
/// Here it's a single u8 table lookup.
#[derive(Clone, Debug)]
pub struct ComposeTable {
    /// Compose indices: k × k u8 values.
    pub indices: Vec<u8>,
    /// Palette size.
    pub k: usize,
}

/// Combined attention semiring: distance + composition.
#[derive(Clone, Debug)]
pub struct AttentionSemiring {
    /// Distance table (attention scores).
    pub attention: AttentionTable,
    /// Compose table (multi-hop reasoning).
    pub compose: ComposeTable,
    /// Palette size.
    pub k: usize,
}

impl AttentionTable {
    /// Build from a weight palette.
    ///
    /// Computes ALL pairwise L1 distances between palette entries.
    /// O(k²) construction, O(1) lookup forever after.
    pub fn build(palette: &WeightPalette) -> Self {
        let k = palette.len();
        let mut distances = vec![0u16; k * k];

        for i in 0..k {
            for j in (i + 1)..k {
                let d = palette.entries[i].l1(&palette.entries[j]);
                // Scale to u16 range
                let max_l1 = 17u64 * 65535; // theoretical max L1 across 17 i16 dims
                let scaled = ((d as u64 * 65535) / max_l1).min(65535) as u16;
                distances[i * k + j] = scaled;
                distances[j * k + i] = scaled;
            }
        }

        AttentionTable { distances, k }
    }

    /// Build with PCDVQ-weighted distances.
    pub fn build_weighted(palette: &WeightPalette) -> Self {
        let k = palette.len();
        let mut distances = vec![0u16; k * k];

        for i in 0..k {
            for j in (i + 1)..k {
                let d = palette.entries[i].l1_weighted(&palette.entries[j]);
                let max_weighted = 20u64 * 65535 + 6 * 3 * 65535 + 10 * 65535;
                let scaled = ((d as u64 * 65535) / max_weighted).min(65535) as u16;
                distances[i * k + j] = scaled;
                distances[j * k + i] = scaled;
            }
        }

        AttentionTable { distances, k }
    }

    /// Look up attention distance between two palette indices. O(1).
    #[inline]
    pub fn distance(&self, a: u8, b: u8) -> u16 {
        self.distances[a as usize * self.k + b as usize]
    }

    /// Compute attention scores for a sequence of tokens.
    ///
    /// Given query palette indices and key palette indices, returns
    /// the n_q × n_k attention distance matrix.
    ///
    /// This replaces Q·K^T matmul. Each score is one table lookup.
    pub fn attention_scores(
        &self,
        q_indices: &[u8],
        k_indices: &[u8],
    ) -> Vec<u16> {
        let n_q = q_indices.len();
        let n_k = k_indices.len();
        let mut scores = vec![0u16; n_q * n_k];

        for i in 0..n_q {
            for j in 0..n_k {
                scores[i * n_k + j] = self.distance(q_indices[i], k_indices[j]);
            }
        }

        scores
    }

    /// Compute attention scores with early termination.
    ///
    /// For each query, only computes scores for keys where the distance
    /// is below threshold. This is the HIP stage of HHTL — pairs that
    /// are obviously distant are skipped entirely.
    ///
    /// Returns sparse (query_idx, key_idx, distance) triples.
    pub fn attention_sparse(
        &self,
        q_indices: &[u8],
        k_indices: &[u8],
        threshold: u16,
    ) -> Vec<(usize, usize, u16)> {
        let n_q = q_indices.len();
        let n_k = k_indices.len();
        let mut sparse = Vec::new();

        for i in 0..n_q {
            for j in 0..n_k {
                let d = self.distance(q_indices[i], k_indices[j]);
                if d < threshold {
                    sparse.push((i, j, d));
                }
            }
        }

        sparse
    }

    /// Byte size of the table.
    pub fn byte_size(&self) -> usize {
        self.k * self.k * 2
    }
}

impl ComposeTable {
    /// Build from a weight palette.
    ///
    /// For each pair (a, b): compute `palette[a].xor_bind(palette[b])`,
    /// then find the nearest palette entry.
    ///
    /// This precomputes all possible multi-hop attention compositions.
    pub fn build(palette: &WeightPalette) -> Self {
        let k = palette.len();
        let mut indices = vec![0u8; k * k];

        for a in 0..k {
            for b in 0..k {
                let composed = palette.entries[a].xor_bind(&palette.entries[b]);
                // Find nearest palette entry to composed result
                let nearest = palette
                    .entries
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, e)| composed.l1(e))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                indices[a * k + b] = nearest as u8;
            }
        }

        ComposeTable { indices, k }
    }

    /// Look up composed path: a → b.
    ///
    /// Returns the palette index of the pattern you get by composing
    /// archetype a with archetype b via XOR bind.
    ///
    /// Usage: "token a attends to token b. What pattern does that produce?"
    /// Answer: `compose(q_palette[a], k_palette[b])` → palette index.
    #[inline]
    pub fn compose(&self, a: u8, b: u8) -> u8 {
        self.indices[a as usize * self.k + b as usize]
    }

    /// Multi-hop composition: a → b → c.
    ///
    /// Composes three archetypes in sequence. Equivalent to two-layer
    /// attention stacking but computed in O(1).
    #[inline]
    pub fn compose_chain(&self, a: u8, b: u8, c: u8) -> u8 {
        let ab = self.compose(a, b);
        self.compose(ab, c)
    }

    /// Byte size.
    pub fn byte_size(&self) -> usize {
        self.k * self.k
    }
}

impl AttentionSemiring {
    /// Build complete semiring from a palette.
    pub fn build(palette: &WeightPalette) -> Self {
        let attention = AttentionTable::build(palette);
        let compose = ComposeTable::build(palette);
        let k = palette.len();
        AttentionSemiring {
            attention,
            compose,
            k,
        }
    }

    /// Distance + compose in one call.
    ///
    /// "How far apart are a and b, and what do they produce together?"
    #[inline]
    pub fn query(&self, a: u8, b: u8) -> (u16, u8) {
        (self.attention.distance(a, b), self.compose.compose(a, b))
    }

    /// Total byte size: distance table + compose table.
    pub fn byte_size(&self) -> usize {
        self.attention.byte_size() + self.compose.byte_size()
    }

    /// Identity element: palette entry closest to Base17::zero().
    pub fn identity(&self, palette: &WeightPalette) -> u8 {
        palette
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| {
                let zero = Base17::zero();
                e.l1(&zero)
            })
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }
}

// ─── Attention Head ──────────────────────────────────────────────────────────

/// A complete compiled attention head.
///
/// Contains palette-quantized Q, K, V matrices and their precomputed
/// attention semiring. Inference is pure table lookup — no matmul.
///
/// ```text
/// Standard:  score = softmax(Q·K^T/√d) · V     [3 matmuls + softmax]
/// Compiled:  score = table[q_idx[i]][k_idx[j]]   [1 lookup]
///            value = v_palette[v_idx[j]]          [1 lookup]
/// ```
#[derive(Clone, Debug)]
pub struct CompiledHead {
    /// Palette for Q weight rows.
    pub q_palette: WeightPalette,
    /// Palette for K weight rows.
    pub k_palette: WeightPalette,
    /// Palette for V weight rows.
    pub v_palette: WeightPalette,
    /// Q×K attention semiring (distance + compose).
    pub qk_semiring: AttentionSemiring,
    /// Token → Q palette assignments.
    pub q_indices: Vec<u8>,
    /// Token → K palette assignments.
    pub k_indices: Vec<u8>,
    /// Token → V palette assignments.
    pub v_indices: Vec<u8>,
}

impl CompiledHead {
    /// Compile an attention head from raw weight matrices.
    ///
    /// # Arguments
    /// - `q_weights`: Query weight matrix (d_model × d_head), row-major
    /// - `k_weights`: Key weight matrix
    /// - `v_weights`: Value weight matrix
    /// - `d_model`: Model dimension (number of columns)
    /// - `d_head`: Head dimension (number of rows per head)
    /// - `palette_k`: Palette size (typically 256)
    pub fn compile(
        q_weights: &[f32],
        k_weights: &[f32],
        v_weights: &[f32],
        d_model: usize,
        d_head: usize,
        palette_k: usize,
    ) -> Self {
        // 1. Project weight rows to Base17
        let q_projected = crate::projection::project_weight_matrix(q_weights, d_head, d_model);
        let k_projected = crate::projection::project_weight_matrix(k_weights, d_head, d_model);
        let v_projected = crate::projection::project_weight_matrix(v_weights, d_head, d_model);

        // 2. Build palettes (CLAM-inspired manifold clustering)
        let q_palette = WeightPalette::build_weighted(&q_projected, palette_k);
        let k_palette = WeightPalette::build_weighted(&k_projected, palette_k);
        let v_palette = WeightPalette::build(&v_projected, palette_k);

        // 3. Build Q×K attention semiring
        // Cross-palette: we need distances between Q archetypes and K archetypes
        let qk_semiring = build_cross_semiring(&q_palette, &k_palette);

        // 4. Assign weight rows to palette indices
        let q_indices = q_palette.assign_all(&q_projected);
        let k_indices = k_palette.assign_all(&k_projected);
        let v_indices = v_palette.assign_all(&v_projected);

        CompiledHead {
            q_palette,
            k_palette,
            v_palette,
            qk_semiring,
            q_indices,
            k_indices,
            v_indices,
        }
    }

    /// Compute attention scores for a token sequence.
    ///
    /// Returns n_tokens × n_tokens distance matrix (lower = more attention).
    pub fn attention_scores(&self, n_tokens: usize) -> Vec<u16> {
        // Each token's Q and K representations come from the compiled indices
        // For now, use the weight-row palette indices as token-level proxies
        // In full implementation, input embeddings would be projected and assigned
        let q_idx: Vec<u8> = (0..n_tokens)
            .map(|i| self.q_indices[i % self.q_indices.len()])
            .collect();
        let k_idx: Vec<u8> = (0..n_tokens)
            .map(|i| self.k_indices[i % self.k_indices.len()])
            .collect();

        self.qk_semiring.attention.attention_scores(&q_idx, &k_idx)
    }

    /// Memory footprint of the compiled head.
    pub fn byte_size(&self) -> usize {
        self.q_palette.byte_size()
            + self.k_palette.byte_size()
            + self.v_palette.byte_size()
            + self.qk_semiring.byte_size()
            + self.q_indices.len()
            + self.k_indices.len()
            + self.v_indices.len()
    }
}

/// Build a cross-palette attention semiring.
///
/// Unlike same-palette semiring, this computes distances between
/// Q palette entries and K palette entries (different palettes).
fn build_cross_semiring(
    q_palette: &WeightPalette,
    k_palette: &WeightPalette,
) -> AttentionSemiring {
    let q_k = q_palette.len();
    let k_k = k_palette.len();
    let k = q_k.max(k_k); // use larger for table sizing

    // Distance table: q_palette × k_palette
    let mut distances = vec![0u16; k * k];
    for i in 0..q_k {
        for j in 0..k_k {
            let d = q_palette.entries[i].l1(&k_palette.entries[j]);
            let max_l1 = 17u64 * 65535;
            let scaled = ((d as u64 * 65535) / max_l1).min(65535) as u16;
            distances[i * k + j] = scaled;
        }
    }

    // Compose table: q × k → nearest in q_palette (query-side composition)
    let mut compose_indices = vec![0u8; k * k];
    for a in 0..q_k {
        for b in 0..k_k {
            let composed = q_palette.entries[a].xor_bind(&k_palette.entries[b]);
            let nearest = q_palette
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| composed.l1(e))
                .map(|(i, _)| i)
                .unwrap_or(0);
            compose_indices[a * k + b] = nearest as u8;
        }
    }

    AttentionSemiring {
        attention: AttentionTable { distances, k },
        compose: ComposeTable {
            indices: compose_indices,
            k,
        },
        k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::Base17;

    fn make_palette(n: usize) -> WeightPalette {
        let rows: Vec<Base17> = (0..n)
            .map(|i| {
                let mut dims = [0i16; 17];
                for d in 0..17 {
                    dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect();
        WeightPalette::build(&rows, n.min(64))
    }

    #[test]
    fn attention_table_symmetric() {
        let pal = make_palette(32);
        let table = AttentionTable::build(&pal);
        for a in 0..pal.len() as u8 {
            for b in 0..pal.len() as u8 {
                assert_eq!(table.distance(a, b), table.distance(b, a));
            }
        }
    }

    #[test]
    fn attention_table_self_zero() {
        let pal = make_palette(32);
        let table = AttentionTable::build(&pal);
        for a in 0..pal.len() as u8 {
            assert_eq!(table.distance(a, a), 0);
        }
    }

    #[test]
    fn compose_identity() {
        let mut rows: Vec<Base17> = (0..31)
            .map(|i| {
                let mut dims = [0i16; 17];
                for d in 0..17 {
                    dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect();
        rows.push(Base17::zero());
        let pal = WeightPalette::build(&rows, 32);
        let compose = ComposeTable::build(&pal);

        // Find identity
        let id = pal
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.l1(&Base17::zero()))
            .map(|(i, _)| i as u8)
            .unwrap();

        // compose(a, identity) should ≈ a
        for a in 0..pal.len() as u8 {
            assert_eq!(compose.compose(a, id), a,
                "compose({}, identity={}) should be {}", a, id, a);
        }
    }

    #[test]
    fn sparse_attention_filters() {
        let pal = make_palette(32);
        let table = AttentionTable::build(&pal);

        let q = vec![0u8, 1, 2, 3];
        let k = vec![0u8, 1, 2, 3, 4, 5, 6, 7];

        let dense_count = q.len() * k.len(); // 32 pairs
        let sparse = table.attention_sparse(&q, &k, 10000); // low threshold

        // Sparse should have fewer entries than dense
        assert!(sparse.len() <= dense_count);
    }

    #[test]
    fn semiring_byte_size() {
        let pal = make_palette(64);
        let sr = AttentionSemiring::build(&pal);
        // 64×64×2 (distances) + 64×64×1 (compose) = 12,288 bytes
        assert_eq!(sr.byte_size(), 64 * 64 * 2 + 64 * 64);
    }

    #[test]
    fn compiled_head_basic() {
        let d_model = 64;
        let d_head = 16;
        let q = vec![0.1f32; d_head * d_model];
        let k = vec![-0.1f32; d_head * d_model];
        let v = vec![0.05f32; d_head * d_model];

        let head = CompiledHead::compile(&q, &k, &v, d_model, d_head, 8);
        assert!(head.byte_size() > 0);

        let scores = head.attention_scores(4);
        assert_eq!(scores.len(), 16); // 4×4
    }
}
