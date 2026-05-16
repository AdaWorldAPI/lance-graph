//! BindSpace — the genius-typed struct-of-arrays.
//!
//! One row per cognitive atom. Columns are separate contiguous buffers so
//! the shader can sweep any single field without pulling unrelated bytes
//! into cache. Meta is packed u32 → one load per row, filter first, load
//! fingerprints second.
//!
//! Address width for named fingerprints is fixed at 256 × u64 = 16,384 bits
//! = `ndarray::hpc::fingerprint::Fingerprint<256>`.
//! The `cycle` column uses `Vsa16kF32` carrier (16,384 × f32 = 64 KB per row)
//! for algebraic operations; other planes remain `u64 × 256`.

use std::sync::Arc;

use lance_graph_contract::cognitive_shader::{ColumnWindow, MetaFilter, MetaWord};
use lance_graph_contract::qualia::QualiaI4_16D;
use lance_graph_ontology::OntologyRegistry;

pub const WORDS_PER_FP: usize = 256;
pub const WIDTH_BITS: usize = WORDS_PER_FP * 64;
pub const QUALIA_DIMS: usize = 18;
pub const FLOATS_PER_VSA: usize = 16_384; // Vsa16kF32 carrier width

/// Named fingerprint planes (content / cycle / topic / angle) plus the
/// per-row Σ codebook index.
///
/// Flat `Box<[u64]>` of length `len * 256` for content/topic/angle. Each row starts at
/// `row * 256` words and spans 256 consecutive u64.
/// The `cycle` plane uses `Vsa16kF32` carrier: `Box<[f32]>` of length `len * 16_384`.
///
/// `sigma` is a 1-byte-per-row index into a 256-entry Σ codebook (per
/// Pillar-6 / PR #289 and Σ-Codebook viability #288, R²=0.9949 at k=256).
/// The codebook itself is *not* loaded here — that lives in
/// `lance-graph-contract::sigma_propagation` (B1/B3 PRs) and is boot-loaded
/// from disk by the runtime. This column only stores the per-row index;
/// callers that need μ alone can ignore it (1 byte / row ≈ 0.02 % of the
/// ~6.2 KB row footprint).
///
/// Why not `[[u64; 256]]`? Because row-major Box<[u64]> gives us O(1)
/// `chunks_exact(256)` iteration which LLVM autovectorises cleanly.
#[derive(Debug)]
pub struct FingerprintColumns {
    pub content: Box<[u64]>,
    pub cycle: Box<[f32]>, // was Box<[u64]>, now Vsa16kF32 carrier (16_384 f32 per row)
    pub topic: Box<[u64]>,
    pub angle: Box<[u64]>,
    /// Σ-codebook index, one byte per row. 0 = "untrained" / first centroid;
    /// non-zero indexes into the 256-entry Σ codebook owned by
    /// `lance-graph-contract::sigma_propagation`. See Pillar-6 (PR #289)
    /// and Σ-Codebook viability (#288, R²=0.9949 at k=256).
    pub sigma: Box<[u8]>,
}

impl FingerprintColumns {
    pub fn zeros(len: usize) -> Self {
        let mk = || vec![0u64; len * WORDS_PER_FP].into_boxed_slice();
        Self {
            content: mk(),
            cycle: vec![0.0f32; len * FLOATS_PER_VSA].into_boxed_slice(),
            topic: mk(),
            angle: mk(),
            sigma: vec![0u8; len].into_boxed_slice(),
        }
    }

    /// Read a row's Σ-codebook index (1 byte).
    #[inline]
    pub fn sigma_at(&self, row: usize) -> u8 {
        self.sigma[row]
    }

    /// Write a row's Σ-codebook index (1 byte).
    #[inline]
    pub fn write_sigma(&mut self, row: usize, idx: u8) {
        self.sigma[row] = idx;
    }

    /// Zero-copy view of a row's content fingerprint words (len = 256).
    #[inline]
    pub fn content_row(&self, row: usize) -> &[u64] {
        &self.content[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP]
    }

    #[inline]
    pub fn cycle_row(&self, row: usize) -> &[f32] {
        &self.cycle[row * FLOATS_PER_VSA..(row + 1) * FLOATS_PER_VSA]
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

    pub fn set_cycle(&mut self, row: usize, vsa: &[f32]) {
        assert_eq!(vsa.len(), FLOATS_PER_VSA);
        self.cycle[row * FLOATS_PER_VSA..(row + 1) * FLOATS_PER_VSA]
            .copy_from_slice(vsa);
    }

    /// Write a cycle fingerprint from Binary16K (u64×256) by projecting to Vsa16kF32 bipolar.
    /// This is the adapter for upstream producers (ShaderBus) that still emit u64.
    pub fn set_cycle_from_bits(&mut self, row: usize, bits: &[u64; WORDS_PER_FP]) {
        use lance_graph_contract::crystal::binary16k_to_vsa16k_bipolar;
        let vsa = binary16k_to_vsa16k_bipolar(bits);
        self.cycle[row * FLOATS_PER_VSA..(row + 1) * FLOATS_PER_VSA]
            .copy_from_slice(&*vsa);
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

/// 18 × f32 per row. Dims 0..16 = EXPERIENCED (CMYK, the QPL convergence
/// observables: arousal, valence, tension, warmth, clarity, boundary, depth,
/// velocity, entropy, coherence, intimacy, presence, assertion, receptivity,
/// groundedness, expansion, integration). Dim 17 = OBSERVED: classification
/// distance (0 = named emotion like "fear", 1 = raw unnamed like "steelwind").
/// Contiguous `len * 18` f32.
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

/// Sibling i4 column alongside QualiaColumn (D-CSV-5a double-write).
/// Length = N rows; each entry is 8 bytes (one `QualiaI4_16D`).
/// Column total = 8 × N bytes.
///
/// Written on every `push_typed` call alongside the f32 `QualiaColumn`.
/// Reads still use the f32 `QualiaColumn` (cutover is D-CSV-5b).
#[derive(Debug)]
pub struct QualiaI4Column(pub Box<[QualiaI4_16D]>);

impl QualiaI4Column {
    pub fn zeros(rows: usize) -> Self {
        Self(vec![QualiaI4_16D::ZERO; rows].into_boxed_slice())
    }

    #[inline]
    pub fn row(&self, row: usize) -> QualiaI4_16D {
        self.0[row]
    }

    #[inline]
    pub fn set(&mut self, row: usize, value: QualiaI4_16D) {
        self.0[row] = value;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Bulk-convert from an f32 `QualiaColumn`.
    ///
    /// Uses the flat `[k * QUALIA_DIMS .. (k+1) * QUALIA_DIMS]` slice layout
    /// of `QualiaColumn.0` to extract each row, then calls
    /// `QualiaI4_16D::from_f32_17d` per row.
    pub fn from_f32(qualia_f32: &QualiaColumn) -> Self {
        let total = qualia_f32.0.len();
        let rows = total / QUALIA_DIMS;
        let mut out = Vec::with_capacity(rows);
        for k in 0..rows {
            let slice = &qualia_f32.0[k * QUALIA_DIMS..(k + 1) * QUALIA_DIMS];
            // from_f32_17d expects &[f32; 17]; QUALIA_DIMS may be 18, so cap at 17.
            let mut arr = [0.0f32; 17];
            let copy_len = slice.len().min(17);
            arr[..copy_len].copy_from_slice(&slice[..copy_len]);
            out.push(QualiaI4_16D::from_f32_17d(&arr));
        }
        Self(out.into_boxed_slice())
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
///
/// `Debug` is implemented manually because `OntologyRegistry` does not derive
/// `Debug` (it holds interior mutability and large hydrated tables); the
/// registry slot is rendered as a presence flag only.
pub struct BindSpace {
    pub len: usize,
    pub fingerprints: FingerprintColumns,
    pub edges: EdgeColumn,
    pub qualia: QualiaColumn,
    pub qualia_i4: QualiaI4Column,
    pub meta: MetaColumn,
    pub temporal: Box<[u64]>,
    pub expert: Box<[u16]>,
    /// Column H: per-row entity type binding (Foundry Object Type equivalent).
    /// 0 = untyped. Non-zero = 1-based index into `Ontology.schemas`.
    pub entity_type: Box<[u16]>,
    /// Optional handle to the ontology registry (Phase 7, v4 plan).
    ///
    /// READ-ONLY access only. The driver consults this registry to resolve
    /// `entity_type` indices into named OGIT schemas, semantic types, and
    /// namespace bridges. The shader never mutates the registry — mutation
    /// flows through `OntologyRegistry::append_mapping` on a separately-owned
    /// `Arc`, never through `BindSpace`.
    ///
    /// FUTURE WORK (NOT this session's deliverable): downstream calibration
    /// improvements will let the MUL gate pick ontology-aware trust
    /// thresholds — e.g. Compliance edges → Plateau-only commit, Healthcare
    /// edges → stricter trust calibration. The MUL gate logic in
    /// `driver.rs:271-320` and the CausalEdge64 emission path are unchanged
    /// in this PR; only the registry handle is wired.
    pub ontology: Option<Arc<OntologyRegistry>>,
}

impl std::fmt::Debug for BindSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindSpace")
            .field("len", &self.len)
            .field("fingerprints", &self.fingerprints)
            .field("edges", &self.edges)
            .field("qualia", &self.qualia)
            .field("qualia_i4", &self.qualia_i4)
            .field("meta", &self.meta)
            .field("temporal", &self.temporal)
            .field("expert", &self.expert)
            .field("entity_type", &self.entity_type)
            .field("ontology", &self.ontology.as_ref().map(|_| "<OntologyRegistry>"))
            .finish()
    }
}

impl BindSpace {
    /// All-zero BindSpace with `len` rows allocated.
    pub fn zeros(len: usize) -> Self {
        Self {
            len,
            fingerprints: FingerprintColumns::zeros(len),
            edges: EdgeColumn::zeros(len),
            qualia: QualiaColumn::zeros(len),
            qualia_i4: QualiaI4Column::zeros(len),
            meta: MetaColumn::zeros(len),
            temporal: vec![0u64; len].into_boxed_slice(),
            expert: vec![0u16; len].into_boxed_slice(),
            entity_type: vec![0u16; len].into_boxed_slice(),
            ontology: None,
        }
    }

    /// Attach a read-only ontology registry handle. Phase 7 (v4 plan).
    ///
    /// The registry is shared via `Arc` — multiple `BindSpace` instances and
    /// the orchestration bridge can hold the same registry. Mutation of the
    /// registry (TTL hydration, mapping proposals) happens on the original
    /// `OntologyRegistry` owner; this handle is read-only by convention.
    pub fn set_ontology(&mut self, registry: Arc<OntologyRegistry>) {
        self.ontology = Some(registry);
    }

    /// Read-only view of the attached ontology registry, if any.
    pub fn ontology(&self) -> Option<&Arc<OntologyRegistry>> {
        self.ontology.as_ref()
    }

    /// Total byte footprint (sum across all columns).
    pub fn byte_footprint(&self) -> usize {
        let content_topic_angle = 3 * self.len * WORDS_PER_FP * 8;
        let cycle_bytes = self.len * FLOATS_PER_VSA * 4; // f32 carrier
        let sigma_bytes = self.len; // 1 byte per row, Σ-codebook index
        let edge_bytes = self.len * 8;
        let qualia_bytes = self.len * QUALIA_DIMS * 4;
        let qualia_i4_bytes = self.len * 8;
        let meta_bytes = self.len * 4;
        let temporal_bytes = self.len * 8;
        let expert_bytes = self.len * 2;
        let entity_type_bytes = self.len * 2;
        content_topic_angle + cycle_bytes + sigma_bytes + edge_bytes + qualia_bytes + qualia_i4_bytes + meta_bytes + temporal_bytes + expert_bytes + entity_type_bytes
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
        self.fingerprints.set_cycle_from_bits(row, cycle_fp);
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

    /// Push a row with default entity_type (0 = untyped).
    ///
    /// # Panics
    /// Panics if cursor >= capacity (F-08: bounds-checked push).
    pub fn push(
        mut self,
        content: &[u64],
        meta: MetaWord,
        edge: u64,
        qualia: &[f32; QUALIA_DIMS],
        temporal: u64,
        expert: u16,
    ) -> Self {
        self.push_typed(content, meta, edge, qualia, temporal, expert, 0)
    }

    /// Push a row with explicit entity type (Column H).
    ///
    /// # Panics
    /// Panics if cursor >= capacity (F-08: bounds-checked push).
    pub fn push_typed(
        mut self,
        content: &[u64],
        meta: MetaWord,
        edge: u64,
        qualia: &[f32; QUALIA_DIMS],
        temporal: u64,
        expert: u16,
        entity_type: u16,
    ) -> Self {
        assert!(
            self.cursor < self.bs.len,
            "BindSpaceBuilder overflow: tried to push row {} into capacity {}",
            self.cursor, self.bs.len,
        );
        let row = self.cursor;
        self.bs.fingerprints.set_content(row, content);
        self.bs.meta.set(row, meta);
        self.bs.edges.set(row, edge);
        self.bs.qualia.set(row, qualia);
        self.bs.qualia_i4.set(row, QualiaI4_16D::from_f32_17d(qualia));
        self.bs.temporal[row] = temporal;
        self.bs.expert[row] = expert;
        self.bs.entity_type[row] = entity_type;
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
        assert_eq!(bs.fingerprints.cycle.len(), 10 * FLOATS_PER_VSA);
        assert_eq!(bs.fingerprints.sigma.len(), 10);
        assert_eq!(bs.qualia.0.len(), 10 * QUALIA_DIMS);
        assert_eq!(bs.meta.0.len(), 10);
    }

    #[test]
    fn bindspace_footprint_adds_columns() {
        let bs = BindSpace::zeros(1);
        // 3 × 2048 (content/topic/angle) + 65536 (cycle f32) + 1 (sigma u8)
        //   + 8 (edge) + 72 (qualia 18×4) + 4 (meta) + 8 (temporal)
        //   + 2 (expert) + 2 (entity_type)
        // = 6144 + 65536 + 1 + 8 + 72 + 4 + 8 + 2 + 2 = 71777
        assert_eq!(bs.byte_footprint(), 71777);
    }

    #[test]
    fn sigma_column_zeros_initialised_to_index_zero() {
        let fp = FingerprintColumns::zeros(8);
        assert_eq!(fp.sigma.len(), 8);
        for row in 0..8 {
            assert_eq!(fp.sigma_at(row), 0, "row {row} must default to codebook index 0");
        }
    }

    #[test]
    fn sigma_column_round_trips_per_row() {
        let mut fp = FingerprintColumns::zeros(4);
        // Distinct indices to prove per-row independence.
        fp.write_sigma(0, 0);
        fp.write_sigma(1, 17);
        fp.write_sigma(2, 200);
        fp.write_sigma(3, 255);
        assert_eq!(fp.sigma_at(0), 0);
        assert_eq!(fp.sigma_at(1), 17);
        assert_eq!(fp.sigma_at(2), 200);
        assert_eq!(fp.sigma_at(3), 255);
        // Overwrite must replace, not OR/accumulate.
        fp.write_sigma(2, 9);
        assert_eq!(fp.sigma_at(2), 9);
    }

    #[test]
    fn fingerprint_columns_sigma_len_matches_other_columns() {
        // Σ is one byte per ROW, while content/topic/angle are 256 u64
        // per row and cycle is 16_384 f32 per row. Verify the implied
        // row-count is identical across all columns.
        for &n in &[0usize, 1, 7, 64, 1_000] {
            let fp = FingerprintColumns::zeros(n);
            assert_eq!(fp.sigma.len(), n, "sigma row count for len {n}");
            assert_eq!(fp.content.len(), n * WORDS_PER_FP);
            assert_eq!(fp.topic.len(), n * WORDS_PER_FP);
            assert_eq!(fp.angle.len(), n * WORDS_PER_FP);
            assert_eq!(fp.cycle.len(), n * FLOATS_PER_VSA);
            // Implied row counts agree.
            assert_eq!(fp.content.len() / WORDS_PER_FP, fp.sigma.len());
            assert_eq!(fp.topic.len() / WORDS_PER_FP, fp.sigma.len());
            assert_eq!(fp.angle.len() / WORDS_PER_FP, fp.sigma.len());
            assert_eq!(fp.cycle.len() / FLOATS_PER_VSA, fp.sigma.len());
        }
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
        fp[0] = 1; // bit 0 set
        bs.write_cycle_fingerprint(1, &fp);
        // After bipolar projection: bit 0 set → dim 0 = +1.0, bit 1 unset → dim 1 = -1.0
        let row = bs.fingerprints.cycle_row(1);
        assert_eq!(row[0], 1.0);   // bit 0 was set
        assert_eq!(row[1], -1.0);  // bit 1 was not set
        // Row 0 should still be all zeros (not projected)
        assert!(bs.fingerprints.cycle_row(0).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn entity_type_defaults_to_untyped() {
        let bs = BindSpace::zeros(4);
        for row in 0..4 {
            assert_eq!(bs.entity_type[row], 0, "default should be untyped (0)");
        }
    }

    #[test]
    fn entity_type_set_and_get() {
        let mut bs = BindSpace::zeros(4);
        bs.entity_type[1] = 42;
        bs.entity_type[3] = 7;
        assert_eq!(bs.entity_type[0], 0);
        assert_eq!(bs.entity_type[1], 42);
        assert_eq!(bs.entity_type[2], 0);
        assert_eq!(bs.entity_type[3], 7);
    }

    #[test]
    fn builder_push_typed_sets_entity_type() {
        let qualia = [0.0f32; QUALIA_DIMS];
        let content = [0u64; WORDS_PER_FP];
        let bs = BindSpaceBuilder::new(2)
            .push_typed(&content, MetaWord::new(1, 0, 100, 100, 0), 0, &qualia, 0, 0, 5)
            .push(&content, MetaWord::new(2, 0, 200, 200, 0), 0, &qualia, 0, 0)
            .build();
        assert_eq!(bs.entity_type[0], 5, "push_typed should set entity_type");
        assert_eq!(bs.entity_type[1], 0, "push should default to 0");
    }

    #[test]
    fn ontology_handle_attaches_and_reads_back() {
        // Phase 7 (v4 plan): BindSpace holds an Option<Arc<OntologyRegistry>>
        // for read-only registry access. This test asserts the wiring; it
        // does NOT test ontology semantics (those live in the
        // lance-graph-ontology crate's own tests).
        let mut bs = BindSpace::zeros(4);
        assert!(bs.ontology().is_none(), "default must be None");
        let reg = Arc::new(OntologyRegistry::new_in_memory());
        bs.set_ontology(reg);
        assert!(bs.ontology().is_some(), "after set_ontology must be Some");
    }

    #[test]
    fn set_cycle_direct_f32() {
        let mut bs = BindSpace::zeros(2);
        let mut vsa = vec![0.0f32; FLOATS_PER_VSA];
        vsa[42] = 0.75;
        vsa[16383] = -0.5;
        bs.fingerprints.set_cycle(0, &vsa);
        assert_eq!(bs.fingerprints.cycle_row(0)[42], 0.75);
        assert_eq!(bs.fingerprints.cycle_row(0)[16383], -0.5);
        assert!(bs.fingerprints.cycle_row(1).iter().all(|&v| v == 0.0));
    }

    // ── D-CSV-5a: QualiaI4Column tests ─────────────────────────────────────

    #[test]
    fn test_qualia_i4_column_zeros() {
        use lance_graph_contract::qualia::QualiaI4_16D;
        const N: usize = 8;
        let col = QualiaI4Column::zeros(N);
        assert_eq!(col.len(), N);
        for i in 0..N {
            assert_eq!(col.row(i), QualiaI4_16D::ZERO, "row {} should be ZERO", i);
        }
        assert!(!col.is_empty());
        assert!(QualiaI4Column::zeros(0).is_empty());
    }

    #[test]
    fn test_qualia_i4_column_set_row() {
        use lance_graph_contract::qualia::QualiaI4_16D;
        const N: usize = 10;
        let mut col = QualiaI4Column::zeros(N);
        let known = QualiaI4_16D::ZERO.with(0, 3).with(9, -4).with(15, 7);
        col.set(5, known);
        assert_eq!(col.row(5), known, "row 5 should equal known value");
        for i in 0..N {
            if i != 5 {
                assert_eq!(col.row(i), QualiaI4_16D::ZERO, "row {} should still be ZERO", i);
            }
        }
    }

    #[test]
    fn test_qualia_i4_column_from_f32_parity() {
        use lance_graph_contract::qualia::QualiaI4_16D;
        const N: usize = 4;
        let mut qcol = QualiaColumn::zeros(N);
        let rows: Vec<[f32; QUALIA_DIMS]> = (0..N)
            .map(|k| {
                let mut arr = [0.0f32; QUALIA_DIMS];
                for d in 0..QUALIA_DIMS {
                    arr[d] = (k * QUALIA_DIMS + d) as f32 / (N * QUALIA_DIMS) as f32;
                }
                arr
            })
            .collect();
        for (k, row) in rows.iter().enumerate() {
            qcol.set(k, row);
        }
        let i4col = QualiaI4Column::from_f32(&qcol);
        assert_eq!(i4col.len(), N);
        for k in 0..N {
            let mut arr17 = [0.0f32; 17];
            let src = &qcol.0[k * QUALIA_DIMS..(k + 1) * QUALIA_DIMS];
            let copy_len = src.len().min(17);
            arr17[..copy_len].copy_from_slice(&src[..copy_len]);
            let expected = QualiaI4_16D::from_f32_17d(&arr17);
            assert_eq!(i4col.row(k), expected, "row {} mismatch", k);
        }
    }

    #[test]
    fn test_bindspace_zeros_double_column() {
        use lance_graph_contract::qualia::QualiaI4_16D;
        const N: usize = 5;
        let bs = BindSpace::zeros(N);
        assert_eq!(bs.qualia.0.len(), N * QUALIA_DIMS);
        assert!(bs.qualia.0.iter().all(|&v| v == 0.0));
        assert_eq!(bs.qualia_i4.len(), N);
        for i in 0..N {
            assert_eq!(bs.qualia_i4.row(i), QualiaI4_16D::ZERO, "i4 row {} should be ZERO", i);
        }
    }

    #[test]
    fn test_bindspace_byte_size_includes_i4() {
        const N: usize = 7;
        let bs = BindSpace::zeros(N);
        let footprint = bs.byte_footprint();
        let content_topic_angle = 3 * N * WORDS_PER_FP * 8;
        let cycle_bytes = N * FLOATS_PER_VSA * 4;
        let sigma_bytes = N;
        let edge_bytes = N * 8;
        let qualia_bytes = N * QUALIA_DIMS * 4;
        let qualia_i4_bytes = N * 8;
        let meta_bytes = N * 4;
        let temporal_bytes = N * 8;
        let expert_bytes = N * 2;
        let entity_type_bytes = N * 2;
        let expected = content_topic_angle + cycle_bytes + sigma_bytes + edge_bytes
            + qualia_bytes + qualia_i4_bytes + meta_bytes + temporal_bytes
            + expert_bytes + entity_type_bytes;
        assert_eq!(footprint, expected,
            "byte_footprint should include i4 column (8 bytes/row × {} rows = {} bytes)",
            N, qualia_i4_bytes);
    }

    #[test]
    fn test_bindspace_push_typed_double_write() {
        use lance_graph_contract::qualia::QualiaI4_16D;
        let mut qualia_arg = [0.0f32; QUALIA_DIMS];
        for d in 0..QUALIA_DIMS {
            qualia_arg[d] = (d + 1) as f32 / (QUALIA_DIMS + 1) as f32;
        }
        let content = [0u64; WORDS_PER_FP];
        let bs = BindSpaceBuilder::new(1)
            .push_typed(&content, MetaWord::new(1, 0, 100, 100, 0), 0, &qualia_arg, 0, 0, 0)
            .build();
        let f32_row = bs.qualia.row(0);
        for d in 0..QUALIA_DIMS {
            assert_eq!(f32_row[d], qualia_arg[d], "f32 qualia dim {} mismatch", d);
        }
        let mut arr17 = [0.0f32; 17];
        let copy_len = qualia_arg.len().min(17);
        arr17[..copy_len].copy_from_slice(&qualia_arg[..copy_len]);
        let expected_i4 = QualiaI4_16D::from_f32_17d(&arr17);
        assert_eq!(bs.qualia_i4.row(0), expected_i4,
            "i4 qualia must match QualiaI4_16D::from_f32_17d of the pushed f32 arg");
    }

}
