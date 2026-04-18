//! BindSpace — the genius-typed struct-of-arrays.
//!
//! One row per cognitive atom. Columns are separate contiguous buffers so
//! the shader can sweep any single field without pulling unrelated bytes
//! into cache. Meta is packed u32 → one load per row, filter first, load
//! fingerprints second.
//!
//! Address width for named fingerprints is fixed at 256 × u64 = 16,384 bits
//! = `ndarray::hpc::fingerprint::Fingerprint<256>`.

use lance_graph_contract::cognitive_shader::{ColumnWindow, MetaFilter, MetaWord};

pub const WORDS_PER_FP: usize = 256;
pub const WIDTH_BITS: usize = WORDS_PER_FP * 64;
pub const QUALIA_DIMS: usize = 18;

/// Named fingerprint planes (content / cycle / topic / angle).
/// Flat `Box<[u64]>` of length `len * 256`. Each row starts at
/// `row * 256` words and spans 256 consecutive u64.
///
/// Why not `[[u64; 256]]`? Because row-major Box<[u64]> gives us O(1)
/// `chunks_exact(256)` iteration which LLVM autovectorises cleanly.
#[derive(Debug)]
pub struct FingerprintColumns {
    pub content: Box<[u64]>,
    pub cycle: Box<[u64]>,
    pub topic: Box<[u64]>,
    pub angle: Box<[u64]>,
}

impl FingerprintColumns {
    pub fn zeros(len: usize) -> Self {
        let mk = || vec![0u64; len * WORDS_PER_FP].into_boxed_slice();
        Self { content: mk(), cycle: mk(), topic: mk(), angle: mk() }
    }

    /// Zero-copy view of a row's content fingerprint words (len = 256).
    #[inline]
    pub fn content_row(&self, row: usize) -> &[u64] {
        &self.content[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP]
    }

    #[inline]
    pub fn cycle_row(&self, row: usize) -> &[u64] {
        &self.cycle[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP]
    }

    #[inline]
    pub fn topic_row(&self, row: usize) -> &[u64] {
        &self.topic[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP]
    }

    #[inline]
    pub fn angle_row(&self, row: usize) -> &[u64] {
        &self.angle[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP]
    }

    /// Write a row's content fingerprint.
    pub fn set_content(&mut self, row: usize, words: &[u64]) {
        assert_eq!(words.len(), WORDS_PER_FP);
        self.content[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP]
            .copy_from_slice(words);
    }

    pub fn set_cycle(&mut self, row: usize, words: &[u64]) {
        assert_eq!(words.len(), WORDS_PER_FP);
        self.cycle[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP]
            .copy_from_slice(words);
    }
}

/// One CausalEdge64 per row. 8-byte-aligned `Box<[u64]>` — SIMD-friendly.
#[derive(Debug)]
pub struct EdgeColumn(pub Box<[u64]>);

impl EdgeColumn {
    pub fn zeros(len: usize) -> Self { Self(vec![0u64; len].into_boxed_slice()) }
    #[inline] pub fn get(&self, row: usize) -> u64 { self.0[row] }
    #[inline] pub fn set(&mut self, row: usize, edge: u64) { self.0[row] = edge; }
}

/// 18 × f32 per row (valence, activation, dominance, depth, certainty,
/// urgency, arousal, valence-high, …). Contiguous `len * 18` f32.
#[derive(Debug)]
pub struct QualiaColumn(pub Box<[f32]>);

impl QualiaColumn {
    pub fn zeros(len: usize) -> Self { Self(vec![0.0f32; len * QUALIA_DIMS].into_boxed_slice()) }

    #[inline]
    pub fn row(&self, row: usize) -> &[f32] {
        &self.0[row * QUALIA_DIMS..(row + 1) * QUALIA_DIMS]
    }

    pub fn set(&mut self, row: usize, q: &[f32; QUALIA_DIMS]) {
        self.0[row * QUALIA_DIMS..(row + 1) * QUALIA_DIMS].copy_from_slice(q);
    }
}

/// Packed u32 per row: thinking(6) + awareness(4) + nars_f(8) + nars_c(8) + free_e(6).
/// One u32 load per row = the cheapest prefilter we can run.
#[derive(Debug)]
pub struct MetaColumn(pub Box<[u32]>);

impl MetaColumn {
    pub fn zeros(len: usize) -> Self { Self(vec![0u32; len].into_boxed_slice()) }
    #[inline] pub fn get(&self, row: usize) -> MetaWord { MetaWord(self.0[row]) }
    #[inline] pub fn set(&mut self, row: usize, w: MetaWord) { self.0[row] = w.0; }
}

/// The BindSpace — read-only universal address space.
/// 16 KB fingerprints × 4 planes + u64 edges + 18 f32 qualia + u32 meta +
/// u64 temporal + u16 expert. All separate column buffers.
///
/// Mutations go through CollapseGate (lance-graph-contract::collapse_gate).
#[derive(Debug)]
pub struct BindSpace {
    pub len: usize,
    pub fingerprints: FingerprintColumns,
    pub edges: EdgeColumn,
    pub qualia: QualiaColumn,
    pub meta: MetaColumn,
    pub temporal: Box<[u64]>,
    pub expert: Box<[u16]>,
}

impl BindSpace {
    /// All-zero BindSpace with `len` rows allocated.
    pub fn zeros(len: usize) -> Self {
        Self {
            len,
            fingerprints: FingerprintColumns::zeros(len),
            edges: EdgeColumn::zeros(len),
            qualia: QualiaColumn::zeros(len),
            meta: MetaColumn::zeros(len),
            temporal: vec![0u64; len].into_boxed_slice(),
            expert: vec![0u16; len].into_boxed_slice(),
        }
    }

    /// Total byte footprint (sum across all columns).
    pub fn byte_footprint(&self) -> usize {
        let fp_bytes = 4 * self.len * WORDS_PER_FP * 8; // 4 planes × len × 256 × 8
        let edge_bytes = self.len * 8;
        let qualia_bytes = self.len * QUALIA_DIMS * 4;
        let meta_bytes = self.len * 4;
        let temporal_bytes = self.len * 8;
        let expert_bytes = self.len * 2;
        fp_bytes + edge_bytes + qualia_bytes + meta_bytes + temporal_bytes + expert_bytes
    }

    /// Apply MetaFilter across a row window. Returns a dense Vec of row
    /// indices that pass — caller uses this list to drive the fingerprint
    /// sweep. Read cost is one u32 per row (very cheap).
    pub fn meta_prefilter(&self, win: ColumnWindow, filter: &MetaFilter) -> Vec<u32> {
        let start = (win.start as usize).min(self.len);
        let end = (win.end as usize).min(self.len);
        let mut out = Vec::with_capacity(end.saturating_sub(start));
        for row in start..end {
            let w = self.meta.get(row);
            if filter.accepts(w) {
                out.push(row as u32);
            }
        }
        out
    }

    /// Emit a cycle_fingerprint row — the unit of thought. This is how
    /// Layer 4 (cognitive_stack) persists its per-cycle signature into
    /// BindSpace so future cycles can retrieve/replay it.
    pub fn write_cycle_fingerprint(&mut self, row: usize, cycle_fp: &[u64; WORDS_PER_FP]) {
        self.fingerprints.set_cycle(row, cycle_fp);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder — EmbedAnything pattern
// ═══════════════════════════════════════════════════════════════════════════

/// Fluent builder for BindSpace. Stages population without exposing
/// the raw column layout to callers.
pub struct BindSpaceBuilder {
    bs: BindSpace,
    cursor: usize,
}

impl BindSpaceBuilder {
    pub fn new(capacity: usize) -> Self {
        Self { bs: BindSpace::zeros(capacity), cursor: 0 }
    }

    pub fn push(
        mut self,
        content: &[u64],
        meta: MetaWord,
        edge: u64,
        qualia: &[f32; QUALIA_DIMS],
        temporal: u64,
        expert: u16,
    ) -> Self {
        let row = self.cursor;
        self.bs.fingerprints.set_content(row, content);
        self.bs.meta.set(row, meta);
        self.bs.edges.set(row, edge);
        self.bs.qualia.set(row, qualia);
        self.bs.temporal[row] = temporal;
        self.bs.expert[row] = expert;
        self.cursor += 1;
        self
    }

    pub fn build(self) -> BindSpace {
        self.bs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindspace_zeros_shapes() {
        let bs = BindSpace::zeros(10);
        assert_eq!(bs.len, 10);
        assert_eq!(bs.fingerprints.content.len(), 10 * WORDS_PER_FP);
        assert_eq!(bs.qualia.0.len(), 10 * QUALIA_DIMS);
        assert_eq!(bs.meta.0.len(), 10);
    }

    #[test]
    fn bindspace_footprint_adds_columns() {
        let bs = BindSpace::zeros(1);
        // 4 × 2048 (fp) + 8 (edge) + 72 (qualia 18×4) + 4 (meta) + 8 (temporal) + 2 (expert)
        // = 8192 + 8 + 72 + 4 + 8 + 2 = 8286
        assert_eq!(bs.byte_footprint(), 8286);
    }

    #[test]
    fn meta_prefilter_returns_passing_rows() {
        let mut bs = BindSpace::zeros(4);
        bs.meta.set(0, MetaWord::new(0, 0, 200, 200, 0));
        bs.meta.set(1, MetaWord::new(0, 0, 50, 50, 0));
        bs.meta.set(2, MetaWord::new(0, 0, 255, 255, 0));
        bs.meta.set(3, MetaWord::new(0, 0, 10, 10, 0));

        let filter = MetaFilter { nars_c_min: 150, ..MetaFilter::ALL };
        let hits = bs.meta_prefilter(ColumnWindow::new(0, 4), &filter);
        assert_eq!(hits, vec![0, 2]);
    }

    #[test]
    fn builder_pushes_rows() {
        let qualia = [0.0f32; QUALIA_DIMS];
        let content = [0u64; WORDS_PER_FP];
        let bs = BindSpaceBuilder::new(3)
            .push(&content, MetaWord::new(1, 0, 100, 100, 0), 0, &qualia, 0, 0)
            .push(&content, MetaWord::new(2, 0, 200, 200, 0), 0, &qualia, 0, 0)
            .build();
        assert_eq!(bs.meta.get(0).thinking(), 1);
        assert_eq!(bs.meta.get(1).thinking(), 2);
    }

    #[test]
    fn write_cycle_fingerprint_persists() {
        let mut bs = BindSpace::zeros(2);
        let mut fp = [0u64; WORDS_PER_FP];
        fp[42] = 0xDEADBEEF;
        bs.write_cycle_fingerprint(1, &fp);
        assert_eq!(bs.fingerprints.cycle_row(1)[42], 0xDEADBEEF);
        assert_eq!(bs.fingerprints.cycle_row(0)[42], 0);
    }
}
