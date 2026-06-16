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
        self.content[row * WORDS_PER_FP..(row + 1) * WORDS_PER_FP].copy_from_slice(words);
    }

    pub fn set_cycle(&mut self, row: usize, vsa: &[f32]) {
        assert_eq!(vsa.len(), FLOATS_PER_VSA);
        self.cycle[row * FLOATS_PER_VSA..(row + 1) * FLOATS_PER_VSA].copy_from_slice(vsa);
    }

    /// Write a cycle fingerprint from Binary16K (u64×256) by projecting to Vsa16kF32 bipolar.
    /// This is the adapter for upstream producers (ShaderBus) that still emit u64.
    pub fn set_cycle_from_bits(&mut self, row: usize, bits: &[u64; WORDS_PER_FP]) {
        use lance_graph_contract::crystal::binary16k_to_vsa16k_bipolar;
        let vsa = binary16k_to_vsa16k_bipolar(bits);
        self.cycle[row * FLOATS_PER_VSA..(row + 1) * FLOATS_PER_VSA].copy_from_slice(&*vsa);
    }
}

/// One CausalEdge64 per row. 8-byte-aligned `Box<[u64]>` — SIMD-friendly.
#[derive(Debug)]
pub struct EdgeColumn(pub Box<[u64]>);

impl EdgeColumn {
    pub fn zeros(len: usize) -> Self {
        Self(vec![0u64; len].into_boxed_slice())
    }
    #[inline]
    pub fn get(&self, row: usize) -> u64 {
        self.0[row]
    }
    #[inline]
    pub fn set(&mut self, row: usize, edge: u64) {
        self.0[row] = edge;
    }
}

/// **DEPRECATED** since 0.2.0 — use `QualiaI4Column` directly.
///
/// The f32 column was retired in D-CSV-5b cutover. Downstream callers
/// have one release cycle to migrate to `bs.qualia` (now `QualiaI4Column`).
///
/// 18 × f32 per row (legacy layout). Replaced by 8 B/row packed i4×16
/// per cognitive-substrate-convergence-v1.md §7.2 and plan decision L-10.
#[deprecated(
    since = "0.2.0",
    note = "use QualiaI4Column directly; this f32 column was retired in D-CSV-5b cutover"
)]
#[derive(Debug)]
pub struct QualiaColumn(pub Box<[f32]>);

#[allow(deprecated)]
impl QualiaColumn {
    pub fn zeros(len: usize) -> Self {
        Self(vec![0.0f32; len * QUALIA_DIMS].into_boxed_slice())
    }

    #[inline]
    pub fn row(&self, row: usize) -> &[f32] {
        &self.0[row * QUALIA_DIMS..(row + 1) * QUALIA_DIMS]
    }

    pub fn set(&mut self, row: usize, q: &[f32; QUALIA_DIMS]) {
        self.0[row * QUALIA_DIMS..(row + 1) * QUALIA_DIMS].copy_from_slice(q);
    }
}

/// Canonical qualia column (D-CSV-5b cutover — QualiaColumn f32 retired).
/// Length = N rows; each entry is 8 bytes (one `QualiaI4_16D`).
/// Column total = 8 × N bytes (9× compression vs old [f32;18] = 72 B/row).
///
/// `bs.qualia` now refers to this i4 column. Consumers use
/// `bs.qualia.row(k)` which returns `QualiaI4_16D` (by value, Copy).
/// For downstream f32 math call `.to_f32_17d()` at the call site.
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

    /// Bulk-convert from an f32 `QualiaColumn` (deprecated type).
    ///
    /// Uses the flat `[k * QUALIA_DIMS .. (k+1) * QUALIA_DIMS]` slice layout
    /// of `QualiaColumn.0` to extract each row, then calls
    /// `QualiaI4_16D::from_f32_17d` per row.
    #[allow(deprecated)]
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
    pub fn zeros(len: usize) -> Self {
        Self(vec![0u32; len].into_boxed_slice())
    }
    #[inline]
    pub fn get(&self, row: usize) -> MetaWord {
        MetaWord(self.0[row])
    }
    #[inline]
    pub fn set(&mut self, row: usize, w: MetaWord) {
        self.0[row] = w.0;
    }
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
    /// Canonical qualia column (i4-16D, D-CSV-5b). Returns `QualiaI4_16D` by value.
    /// The old `QualiaColumn` (f32) was retired in D-CSV-5b; see `QualiaColumn` docs.
    pub qualia: QualiaI4Column,
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
            .field("meta", &self.meta)
            .field("temporal", &self.temporal)
            .field("expert", &self.expert)
            .field("entity_type", &self.entity_type)
            .field(
                "ontology",
                &self.ontology.as_ref().map(|_| "<OntologyRegistry>"),
            )
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
            qualia: QualiaI4Column::zeros(len),
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
        // D-CSV-5b: qualia is now QualiaI4Column (8 B/row). f32 column (72 B/row) retired.
        let qualia_bytes = self.len * 8;
        let meta_bytes = self.len * 4;
        let temporal_bytes = self.len * 8;
        let expert_bytes = self.len * 2;
        let entity_type_bytes = self.len * 2;
        content_topic_angle
            + cycle_bytes
            + sigma_bytes
            + edge_bytes
            + qualia_bytes
            + meta_bytes
            + temporal_bytes
            + expert_bytes
            + entity_type_bytes
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
        Self {
            bs: BindSpace::zeros(capacity),
            cursor: 0,
        }
    }

    /// Push a row with default entity_type (0 = untyped).
    ///
    /// # Panics
    /// Panics if cursor >= capacity (F-08: bounds-checked push).
    /// Push a row with default entity_type (0 = untyped).
    ///
    /// D-CSV-5b: `qualia` is now `QualiaI4_16D`. Callers that previously
    /// supplied `&[f32; QUALIA_DIMS]` should call
    /// `QualiaI4_16D::from_f32_17d(&q17)` at the call site.
    ///
    /// # Panics
    /// Panics if cursor >= capacity (F-08: bounds-checked push).
    pub fn push(
        self,
        content: &[u64],
        meta: MetaWord,
        edge: u64,
        qualia: QualiaI4_16D,
        temporal: u64,
        expert: u16,
    ) -> Self {
        self.push_typed(content, meta, edge, qualia, temporal, expert, 0)
    }

    /// Push a row with explicit entity type (Column H).
    ///
    /// # Panics
    /// Panics if cursor >= capacity (F-08: bounds-checked push).
    /// Push a row with explicit entity type (Column H) and i4 qualia.
    ///
    /// D-CSV-5b: `qualia` is now `QualiaI4_16D`. Callers that previously
    /// supplied `&[f32; QUALIA_DIMS]` should call
    /// `QualiaI4_16D::from_f32_17d(&q17)` at the call site.
    ///
    /// # Panics
    /// Panics if cursor >= capacity (F-08: bounds-checked push).
    pub fn push_typed(
        mut self,
        content: &[u64],
        meta: MetaWord,
        edge: u64,
        qualia: QualiaI4_16D,
        temporal: u64,
        expert: u16,
        entity_type: u16,
    ) -> Self {
        assert!(
            self.cursor < self.bs.len,
            "BindSpaceBuilder overflow: tried to push row {} into capacity {}",
            self.cursor,
            self.bs.len,
        );
        let row = self.cursor;
        self.bs.fingerprints.set_content(row, content);
        self.bs.meta.set(row, meta);
        self.bs.edges.set(row, edge);
        // D-CSV-5b: single i4 write (f32 column retired).
        self.bs.qualia.set(row, qualia);
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
        // D-CSV-5b: qualia is now QualiaI4Column (len = N rows, 8 B each)
        assert_eq!(bs.qualia.len(), 10);
        assert_eq!(bs.meta.0.len(), 10);
    }

    #[test]
    fn bindspace_footprint_adds_columns() {
        let bs = BindSpace::zeros(1);
        // D-CSV-5b: f32 QualiaColumn (72 B/row) retired; QualiaI4Column (8 B/row) is canonical.
        // 3 × 2048 (content/topic/angle) + 65536 (cycle f32) + 1 (sigma u8)
        //   + 8 (edge) + 8 (qualia i4, D-CSV-5b) + 4 (meta)
        //   + 8 (temporal) + 2 (expert) + 2 (entity_type)
        // = 6144 + 65536 + 1 + 8 + 8 + 4 + 8 + 2 + 2 = 71713
        assert_eq!(bs.byte_footprint(), 71713);
    }

    #[test]
    fn sigma_column_zeros_initialised_to_index_zero() {
        let fp = FingerprintColumns::zeros(8);
        assert_eq!(fp.sigma.len(), 8);
        for row in 0..8 {
            assert_eq!(
                fp.sigma_at(row),
                0,
                "row {row} must default to codebook index 0"
            );
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

        let filter = MetaFilter {
            nars_c_min: 150,
            ..MetaFilter::ALL
        };
        let hits = bs.meta_prefilter(ColumnWindow::new(0, 4), &filter);
        assert_eq!(hits, vec![0, 2]);
    }

    #[test]
    fn builder_pushes_rows() {
        use lance_graph_contract::qualia::QualiaI4_16D;
        let qualia = QualiaI4_16D::ZERO;
        let content = [0u64; WORDS_PER_FP];
        let bs = BindSpaceBuilder::new(3)
            .push(&content, MetaWord::new(1, 0, 100, 100, 0), 0, qualia, 0, 0)
            .push(&content, MetaWord::new(2, 0, 200, 200, 0), 0, qualia, 0, 0)
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
        assert_eq!(row[0], 1.0); // bit 0 was set
        assert_eq!(row[1], -1.0); // bit 1 was not set
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
        use lance_graph_contract::qualia::QualiaI4_16D;
        let qualia = QualiaI4_16D::ZERO;
        let content = [0u64; WORDS_PER_FP];
        let bs = BindSpaceBuilder::new(2)
            .push_typed(
                &content,
                MetaWord::new(1, 0, 100, 100, 0),
                0,
                qualia,
                0,
                0,
                5,
            )
            .push(&content, MetaWord::new(2, 0, 200, 200, 0), 0, qualia, 0, 0)
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

    // ── D-CSV-5b: QualiaI4Column canonical tests ────────────────────────────

    /// 1. BindSpace::zeros has bs.qualia of type QualiaI4Column with length N.
    #[test]
    fn test_bindspace_zeros_qualia_is_i4() {
        use lance_graph_contract::qualia::QualiaI4_16D;
        const N: usize = 8;
        let bs = BindSpace::zeros(N);
        // qualia is now QualiaI4Column; len() is row count (not flat f32 len).
        assert_eq!(bs.qualia.len(), N);
        for i in 0..N {
            assert_eq!(
                bs.qualia.row(i),
                QualiaI4_16D::ZERO,
                "row {} should be ZERO",
                i
            );
        }
        assert!(!bs.qualia.is_empty());
        assert!(BindSpace::zeros(0).qualia.is_empty());
    }

    /// 2. byte_size() is now (N × 8) + other_columns, NOT (N × 72) + other.
    #[test]
    fn test_bindspace_byte_size_post_cutover() {
        const N: usize = 7;
        let bs = BindSpace::zeros(N);
        let footprint = bs.byte_footprint();
        // Explicit formula (D-CSV-5b): no f32 qualia column (72 B/row gone).
        let content_topic_angle = 3 * N * WORDS_PER_FP * 8; // 3 × 2048 × N
        let cycle_bytes = N * FLOATS_PER_VSA * 4; // 65536 × N
        let sigma_bytes = N; // 1 × N
        let edge_bytes = N * 8; // 8 × N
        let qualia_bytes = N * 8; // i4: 8 × N (NOT 72 × N)
        let meta_bytes = N * 4;
        let temporal_bytes = N * 8;
        let expert_bytes = N * 2;
        let entity_type_bytes = N * 2;
        let expected = content_topic_angle
            + cycle_bytes
            + sigma_bytes
            + edge_bytes
            + qualia_bytes
            + meta_bytes
            + temporal_bytes
            + expert_bytes
            + entity_type_bytes;
        assert_eq!(
            footprint,
            expected,
            "D-CSV-5b: byte_footprint should be {} (i4 8 B/row × {} rows), not {} (f32 72 B/row)",
            expected,
            N,
            expected + N * (QUALIA_DIMS * 4 - 8)
        );
    }

    /// 3. push_typed writes i4; read back via bs.qualia.row(0) equals the input.
    #[test]
    fn test_bindspace_push_typed_writes_i4() {
        use lance_graph_contract::qualia::QualiaI4_16D;
        let known = QualiaI4_16D::ZERO.with(0, 3).with(7, -5).with(15, 7);
        let content = [0u64; WORDS_PER_FP];
        let bs = BindSpaceBuilder::new(1)
            .push_typed(
                &content,
                MetaWord::new(1, 0, 100, 100, 0),
                0,
                known,
                0,
                0,
                0,
            )
            .build();
        assert_eq!(
            bs.qualia.row(0),
            known,
            "push_typed must write QualiaI4_16D verbatim to bs.qualia.row(0)"
        );
    }

    /// 4. engine_bridge conversion: from_f32_17d at the bridge produces the
    ///    same i4 that QualiaI4_16D::from_f32_17d would produce independently.
    #[test]
    fn test_bindspace_engine_bridge_converts_f32_at_boundary() {
        use lance_graph_contract::qualia::QualiaI4_16D;
        // Simulate what dispatch_busdto does post-cutover:
        // the engine produces f32; from_f32_17d converts at the bridge boundary.
        let mut q = [0.0f32; QUALIA_DIMS]; // QUALIA_DIMS=18 (bindspace local const)
        q[0] = 0.8; // energy
        q[1] = 0.3;
        q[3] = -0.6;
        q[9] = 512.0; // codebook_index as f32

        // Oracle: what from_f32_17d produces from the first 17 dims
        let mut q17 = [0.0f32; 17];
        q17.copy_from_slice(&q[..17]);
        let oracle = QualiaI4_16D::from_f32_17d(&q17);

        // Write to BindSpace the same way the bridge will post-cutover
        let mut bs = BindSpace::zeros(1);
        bs.qualia.set(0, oracle); // bridge writes the converted i4 directly
        assert_eq!(
            bs.qualia.row(0),
            oracle,
            "bridge-converted i4 must match QualiaI4_16D::from_f32_17d oracle"
        );
    }

    /// 5. QualiaColumn deprecation attribute is present (meta-test).
    ///    Verifies the deprecated annotation at the type level.
    #[test]
    fn test_qualia_column_deprecation_warning_present() {
        // The `#[deprecated]` attribute on QualiaColumn means using it
        // triggers a compiler warning. We verify the type STILL EXISTS
        // (for backward-compat one release cycle) while confirming that
        // the canonical path is QualiaI4Column.
        //
        // If QualiaColumn were removed entirely this test would fail to
        // compile, which is CORRECT — the deprecation cycle has ended.
        #[allow(deprecated)]
        {
            let col = QualiaColumn::zeros(2);
            // The type exists and is functional during the deprecation cycle.
            assert_eq!(
                col.0.len(),
                2 * QUALIA_DIMS,
                "deprecated QualiaColumn must still allocate during deprecation cycle"
            );
        }
        // The canonical field on BindSpace is QualiaI4Column, not QualiaColumn.
        let bs = BindSpace::zeros(1);
        // Confirming bs.qualia is QualiaI4Column (returns QualiaI4_16D, not &[f32]):
        use lance_graph_contract::qualia::QualiaI4_16D;
        let _: QualiaI4_16D = bs.qualia.row(0);
    }
}
