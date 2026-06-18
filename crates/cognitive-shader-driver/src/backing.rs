//! `BackingStore` — the read/write shim that lets `driver.run()` keep ONE body
//! while the W3+W4a migration flips its substrate from the singleton
//! [`BindSpace`] to a per-mailbox [`MailboxSoA`].
//!
//! ## Why an enum, not a trait (OQ-C resolved)
//!
//! The dispatch hot path reads the same six column surfaces on every cycle. An
//! `enum BackingStore<'a>` monomorphizes to a single match per read with no
//! `dyn` indirection — equivalent to a generic `R: DriverRead` bound but with
//! no new public trait to maintain. The feature flag selects which *variant is
//! constructed* (in `ShaderDriver::backing`), NEVER a `#[cfg]` branch inside
//! `run()`.
//!
//! ## The two arms must agree byte-for-byte (C2 — the load-bearing invariant)
//!
//! The Singleton arm reproduces today's `driver.run()` reads verbatim. The
//! Mailbox arm reads the migrated columns of a `MailboxSoA<1024>` whose rows
//! mirror the BindSpace window. The W2 differential harness
//! (`tests/w2_differential.rs`) asserts the WHOLE `ShaderCrystal` is
//! bit-identical (`f32::to_bits()`) across the two arms — if the arms diverge,
//! that test fails, which is the entire point of building the shim before any
//! production flip.
//!
//! ## Prefilter window semantics (C2, P0)
//!
//! [`BackingStore::prefilter`] on the Mailbox arm iterates
//! `win.start.min(populated)..win.end.min(populated)` — byte-identical to
//! [`BindSpace::meta_prefilter`]'s `start..end`. It must NOT iterate
//! `0..populated`: that would silently ignore a non-zero `win.start` (the
//! wire/grpc paths pass `req.row_start`), producing the same Vec *shape* with
//! divergent rows — a sentinel-lie the differential's non-zero-window case
//! catches.
//!
//! ## Write shim (C1)
//!
//! [`BackingStoreWrite`] is the `&mut` mirror. `driver.run()` itself writes NO
//! bindspace columns (it is `&self`); the write surface is exercised only by
//! the W2 differential harness, mirroring a BindSpace window into a mailbox.
//! `set_edge` reconciles the column-type mismatch: BindSpace stores raw `u64`
//! (`EdgeColumn::set`), the mailbox stores typed `CausalEdge64`
//! (`MailboxSoA::set_edge`); the Singleton arm unwraps `e.0`.

use causal_edge::edge::CausalEdge64;
use lance_graph_contract::cognitive_shader::{ColumnWindow, MetaFilter};

use crate::bindspace::BindSpace;
use crate::mailbox_soa::{WriteCell, WriteOutcome};
#[cfg(feature = "mailbox-thoughtspace")]
use crate::mailbox_soa::MailboxSoA;

/// Read-only substrate the dispatch hot path sweeps.
///
/// `Singleton` is the live default (the migrating-off-of singleton). `Mailbox`
/// is the migration target, compiled only under `mailbox-thoughtspace`.
pub(crate) enum BackingStore<'a> {
    /// Path A — the live default. Reads the shared singleton `BindSpace`.
    Singleton(&'a BindSpace),
    /// Path B — feature-gated. Reads one designated `MailboxSoA<1024>`'s
    /// migrated columns.
    #[cfg(feature = "mailbox-thoughtspace")]
    Mailbox(&'a MailboxSoA<1024>),
}

impl<'a> BackingStore<'a> {
    /// Apply `f` across the meta column within `win`, returning the dense Vec of
    /// passing row indices (ascending) — the prefilter that drives the
    /// fingerprint sweep.
    ///
    /// Both arms honour the dispatch [`ColumnWindow`] start AND end. The
    /// Mailbox arm clamps to [`MailboxSoA::populated`] (the `BindSpace::len`
    /// analogue) so zeroed padding rows `populated..N` are never swept — a
    /// zeroed `MetaWord` would otherwise pass `MetaFilter::accepts`.
    #[inline]
    pub(crate) fn prefilter(&self, win: ColumnWindow, f: &MetaFilter) -> Vec<u32> {
        match self {
            BackingStore::Singleton(bs) => bs.meta_prefilter(win, f),
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStore::Mailbox(mb) => {
                let populated = mb.populated();
                let start = (win.start as usize).min(populated);
                let end = (win.end as usize).min(populated);
                let mut out = Vec::with_capacity(end.saturating_sub(start));
                for row in start..end {
                    if f.accepts(mb.meta_at(row)) {
                        out.push(row as u32);
                    }
                }
                out
            }
        }
    }

    /// Zero-copy view of `row`'s content identity fingerprint (256 u64).
    /// The driver's resonance/Hamming search reads this on the hot path.
    #[inline]
    pub(crate) fn content_row(&self, row: usize) -> &[u64] {
        match self {
            BackingStore::Singleton(bs) => bs.fingerprints.content_row(row),
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStore::Mailbox(mb) => mb.content_row(row),
        }
    }

    /// `row`'s 17-dim affective vector as f32 (the `auto_style` / α-composite
    /// read). The underlying store is `QualiaI4_16D`; conversion happens here
    /// so both arms expose an identical `[f32; 17]`.
    #[inline]
    pub(crate) fn qualia_17d(&self, row: usize) -> [f32; 17] {
        match self {
            BackingStore::Singleton(bs) => bs.qualia.row(row).to_f32_17d(),
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStore::Mailbox(mb) => mb.qualia_at(row).to_f32_17d(),
        }
    }

    /// `row`'s `CausalEdge64` baton edge — typed on BOTH arms.
    ///
    /// The Singleton arm wraps the raw `u64` (`EdgeColumn` stores `u64`); the
    /// Mailbox arm returns the natively-typed `CausalEdge64` (zero raw bounce).
    #[inline]
    pub(crate) fn edge(&self, row: usize) -> CausalEdge64 {
        match self {
            BackingStore::Singleton(bs) => CausalEdge64(bs.edges.get(row)),
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStore::Mailbox(mb) => mb.edge(row),
        }
    }

    /// `row`'s OGIT entity-type index (0 = untyped).
    #[inline]
    pub(crate) fn entity_type(&self, row: usize) -> u16 {
        match self {
            BackingStore::Singleton(bs) => bs.entity_type[row],
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStore::Mailbox(mb) => mb.entity_type_at(row),
        }
    }

    /// Declared logical row count (`BindSpace::len` / `MailboxSoA::populated`).
    #[inline]
    #[allow(dead_code)] // mirrors the read surface; row_count routes through bindspace until W4b
    pub(crate) fn len(&self) -> usize {
        match self {
            BackingStore::Singleton(bs) => bs.len,
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStore::Mailbox(mb) => mb.populated(),
        }
    }
}

/// Write mirror of [`BackingStore`] — the `&mut` write surface (C1).
///
/// `driver.run()` writes nothing (it is `&self`), so this is exercised only by
/// the W2 differential harness, which mirrors a BindSpace window into a mailbox
/// row-for-row. Under the feature BOTH the read and write surfaces flip in the
/// harness; production never half-flips (H-DW-1).
///
/// `dead_code` is allowed deliberately: C1 mandates this write surface exist
/// NOW (so W2 can flip BOTH read+write under the feature), but the production
/// write path is still `&self` and does not consume it until W4b/W7. This is a
/// forward-staged API, not a masked lint. It IS exercised by the in-module
/// tests below.
#[allow(dead_code)]
pub(crate) enum BackingStoreWrite<'a> {
    /// Path A — writes the singleton `BindSpace` columns.
    Singleton(&'a mut BindSpace),
    /// Path B — feature-gated. Writes one designated `MailboxSoA<1024>`.
    #[cfg(feature = "mailbox-thoughtspace")]
    Mailbox(&'a mut MailboxSoA<1024>),
}

#[allow(dead_code)] // forward-staged write surface (C1); consumed at W4b/W7. Tested below.
impl BackingStoreWrite<'_> {
    /// Write `row`'s content identity fingerprint (256 u64).
    #[inline]
    pub(crate) fn set_content(&mut self, row: usize, words: &[u64]) {
        match self {
            BackingStoreWrite::Singleton(bs) => bs.fingerprints.set_content(row, words),
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStoreWrite::Mailbox(mb) => mb.set_content(row, words),
        }
    }

    /// Write `row`'s packed `QualiaI4_16D` affective vector.
    #[inline]
    pub(crate) fn set_qualia(&mut self, row: usize, q: lance_graph_contract::qualia::QualiaI4_16D) {
        match self {
            BackingStoreWrite::Singleton(bs) => bs.qualia.set(row, q),
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStoreWrite::Mailbox(mb) => mb.set_qualia(row, q),
        }
    }

    /// Write `row`'s `CausalEdge64` baton edge.
    ///
    /// The Singleton arm unwraps to the raw `u64` that `EdgeColumn::set` stores;
    /// the Mailbox arm stores the typed edge directly.
    #[inline]
    pub(crate) fn set_edge(&mut self, row: usize, e: CausalEdge64) {
        match self {
            BackingStoreWrite::Singleton(bs) => bs.edges.set(row, e.0),
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStoreWrite::Mailbox(mb) => mb.set_edge(row, e),
        }
    }

    /// Write `row`'s packed `MetaWord`.
    #[inline]
    pub(crate) fn set_meta(
        &mut self,
        row: usize,
        m: lance_graph_contract::cognitive_shader::MetaWord,
    ) {
        match self {
            BackingStoreWrite::Singleton(bs) => bs.meta.set(row, m),
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStoreWrite::Mailbox(mb) => mb.set_meta(row, m),
        }
    }

    /// Write `row`'s OGIT entity-type index.
    #[inline]
    pub(crate) fn set_entity_type(&mut self, row: usize, t: u16) {
        match self {
            BackingStoreWrite::Singleton(bs) => bs.entity_type[row] = t,
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStoreWrite::Mailbox(mb) => mb.set_entity_type(row, t),
        }
    }

    /// Write `row`'s temporal stamp.
    #[inline]
    pub(crate) fn set_temporal(&mut self, row: usize, t: u64) {
        match self {
            BackingStoreWrite::Singleton(bs) => bs.temporal[row] = t,
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStoreWrite::Mailbox(mb) => mb.set_temporal(row, t),
        }
    }

    /// Write `row`'s expert/corpus id.
    #[inline]
    pub(crate) fn set_expert(&mut self, row: usize, e: u16) {
        match self {
            BackingStoreWrite::Singleton(bs) => bs.expert[row] = e,
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStoreWrite::Mailbox(mb) => mb.set_expert(row, e),
        }
    }

    /// Write `row`'s Σ-codebook index.
    #[inline]
    pub(crate) fn set_sigma(&mut self, row: usize, s: u8) {
        match self {
            BackingStoreWrite::Singleton(bs) => bs.fingerprints.write_sigma(row, s),
            #[cfg(feature = "mailbox-thoughtspace")]
            BackingStoreWrite::Mailbox(mb) => mb.set_sigma(row, s),
        }
    }

    /// Cycle-aware row write (S2.5 deinterlacing) — routes a [`WriteCell`]
    /// through the per-mailbox cycle gate.
    ///
    /// - **Mailbox arm:** delegates to [`MailboxSoA::write_row`], which gates the
    ///   write (wrap-aware) against the mailbox's `current_cycle`: a stale batch
    ///   never overwrites a row the owner advanced past.
    /// - **Singleton arm:** **cycle-blind BY CONSTRUCTION** (CATCH-CRITICAL,
    ///   baton-handoff). `BindSpace` owns no `current_cycle`, so it cannot gate;
    ///   it applies the cell's present fields via the per-field setters and
    ///   returns [`WriteOutcome::Accepted`] unconditionally. The cycle gate is a
    ///   Mailbox-only guarantee until W7 deletes `BindSpace`. `topic`/`angle` are
    ///   Mailbox-only on the write shim; the legacy singleton path does not carry
    ///   them (it has no dense-plane setter on this surface).
    #[inline]
    pub(crate) fn write_row(&mut self, row: usize, cycle: u32, cell: &WriteCell<'_>) -> WriteOutcome {
        #[cfg(feature = "mailbox-thoughtspace")]
        if let BackingStoreWrite::Mailbox(mb) = self {
            return mb.write_row(row, cycle, cell);
        }
        // Singleton arm: cycle-blind by construction. The `cycle` argument is
        // intentionally unused here (no clock to compare against).
        let _ = cycle;
        if let Some(w) = cell.content {
            self.set_content(row, w);
        }
        if let Some(q) = cell.qualia {
            self.set_qualia(row, q);
        }
        if let Some(e) = cell.edge {
            self.set_edge(row, e);
        }
        if let Some(m) = cell.meta {
            self.set_meta(row, m);
        }
        if let Some(t) = cell.entity_type {
            self.set_entity_type(row, t);
        }
        if let Some(t) = cell.temporal {
            self.set_temporal(row, t);
        }
        if let Some(x) = cell.expert {
            self.set_expert(row, x);
        }
        if let Some(s) = cell.sigma {
            self.set_sigma(row, s);
        }
        WriteOutcome::Accepted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindspace::WORDS_PER_FP;
    use lance_graph_contract::cognitive_shader::MetaWord;
    use lance_graph_contract::qualia::QualiaI4_16D;

    /// The Singleton read arm reproduces direct BindSpace reads byte-for-byte,
    /// and the write shim's Singleton arm is the inverse of those reads.
    #[test]
    fn singleton_write_then_read_round_trips() {
        let mut bs = BindSpace::zeros(4);
        let content: [u64; WORDS_PER_FP] = {
            let mut w = [0u64; WORDS_PER_FP];
            w[0] = 0xFEED_FACE;
            w[WORDS_PER_FP - 1] = 0x1234;
            w
        };
        let q = QualiaI4_16D::ZERO.with(0, 5).with(9, -3);
        let meta = MetaWord::new(3, 2, 111, 222, 4);

        // Write through the write shim (exercises BackingStoreWrite::Singleton).
        {
            let mut w = BackingStoreWrite::Singleton(&mut bs);
            w.set_content(2, &content);
            w.set_qualia(2, q);
            w.set_meta(2, meta);
            w.set_edge(2, CausalEdge64(0xABCD_0002));
            w.set_entity_type(2, 0);
            w.set_temporal(2, 0xDEAD);
            w.set_expert(2, 77);
            w.set_sigma(2, 9);
        }

        // Read back through the read shim (exercises BackingStore::Singleton).
        let r = BackingStore::Singleton(&bs);
        assert_eq!(r.content_row(2), &content[..], "content round-trip");
        assert_eq!(r.qualia_17d(2), q.to_f32_17d(), "qualia round-trip");
        assert_eq!(r.edge(2).0, 0xABCD_0002, "edge round-trip");
        assert_eq!(r.entity_type(2), 0, "entity_type round-trip");
        assert_eq!(r.len(), 4, "len reports BindSpace::len");

        // Direct-read parity: the shim reads exactly what BindSpace exposes.
        assert_eq!(r.content_row(2), bs.fingerprints.content_row(2));
        assert_eq!(r.edge(2).0, bs.edges.get(2));
        assert_eq!(
            r.prefilter(crate::ColumnWindow::new(0, 4), &MetaFilter::ALL)
                .len(),
            4
        );
    }

    /// Under the feature, the Mailbox write+read arms round-trip and the
    /// prefilter honours a NON-ZERO window start (the C2 sentinel-lie guard at
    /// the shim level, complementing the differential harness).
    #[cfg(feature = "mailbox-thoughtspace")]
    #[test]
    fn mailbox_write_read_and_windowed_prefilter() {
        let mut mb: MailboxSoA<1024> = MailboxSoA::new(0, 0, 1.0);
        for row in 0..6usize {
            let mut w = BackingStoreWrite::Mailbox(&mut mb);
            let mut c = [0u64; WORDS_PER_FP];
            c[0] = row as u64 + 1;
            w.set_content(row, &c);
            w.set_qualia(row, QualiaI4_16D::ZERO.with(0, row as i8));
            w.set_meta(row, MetaWord::new(row as u8, 1, 200, 200, 0));
            w.set_edge(row, CausalEdge64(0xAB00 | row as u64));
            w.set_entity_type(row, 0);
        }
        mb.set_populated(6);

        let r = BackingStore::Mailbox(&mb);
        assert_eq!(r.len(), 6, "len reports populated, not capacity N");
        assert_eq!(r.edge(3).0, 0xAB00 | 3, "mailbox edge round-trip");
        assert_eq!(r.content_row(3)[0], 4, "mailbox content round-trip");

        // Non-zero window: rows [2, 5) only — proves the start is honoured.
        let passed = r.prefilter(crate::ColumnWindow::new(2, 5), &MetaFilter::ALL);
        assert_eq!(
            passed,
            vec![2, 3, 4],
            "windowed prefilter honours start AND end"
        );
    }
}
