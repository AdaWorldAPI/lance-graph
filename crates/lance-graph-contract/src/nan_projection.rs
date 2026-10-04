//! NaN-detection projection surface — the singleton BindSpace, demoted.
//!
//! Per the operator (2026-06-20): **kill the singleton BindSpace as a stateful
//! carrier; keep it ONLY as a read-only PROJECTION SURFACE for NaN detection.**
//! You do not hold a mutable BindSpace and you do not bundle into it. You
//! *project* the SoA's f32 accumulator tenant ([`ValueTenant::Energy`]) through
//! this surface to flag any non-finite board.
//!
//! The finiteness test itself stays the fastest possible read: a fixed-offset,
//! fixed-stride read of one 4-byte `f32` per [`NodeRow`], decided by a single
//! integer exponent mask — **no float load, no branch on the value**.
//! `Energy` is F32 precisely because F32 is the fast tenant (half of f64, and the
//! NaN test is one `&`-compare on the bit pattern).
//!
//! **Schema-gated (T5 closure, 2026-07-29).** `value_offset()` is a fixed
//! reserved position — the SAME byte range regardless of which [`ValueSchema`]
//! a row resolves to (RESERVE, DON'T RECLAIM) — so reading it is never memory-
//! unsafe. But a row whose resolved schema does NOT materialise `Energy` (e.g.
//! [`ValueSchema::Compressed`], used by [`NodeGuid::CLASSID_FMA`]) has no
//! writer obligated to keep that reserved range meaningful; a schema-blind sweep
//! would silently misread whatever bytes happen to sit there as if they were a
//! real energy accumulator — a false non-finite flag, or worse, a false-clean
//! pass over real corruption elsewhere in the slab that a NaN-shaped bit pattern
//! happened to zero out. Each row is therefore gated on its OWN resolved
//! `[ValueSchema::has]` before its `Energy` bytes are read at all.
//!
//! **Where the schema is resolved.** The finiteness test — the exponent-mask
//! compare on four loaded bytes — has no branch on the value. The schema
//! lookup ([`classid_read_mode`], a `HashMap` behind a `LazyLock`) is kept
//! out of the per-row loop:
//!
//! - [`project_energy_nonfinite_resolved`] / [`energy_all_finite_resolved`]
//!   take the population's already-resolved [`ValueSchema`] (e.g.
//!   `ResolvedReading::read_mode.value_schema`). One schema check per call;
//!   no registry lookup at all.
//! - [`project_energy_nonfinite`] / [`energy_all_finite`] accept a mixed
//!   batch. They resolve once per RUN of equal classid and hand each run to
//!   the resolved path, so a homogeneous batch costs one lookup, and a batch
//!   alternating classids costs one per change. The only per-row metadata
//!   work left there is comparing the row's classid with the run's.
//!
//! [`NanReport::skipped`] makes the gate's effect observable rather than a
//! silent no-op, per the workspace's can-it-fire testing rule.
//!
//! This is "BindSpace as projection surface": the only surviving role of the old
//! singleton is to answer "did any node go non-finite this cycle?" over the SoA.
//!
//! [`ValueSchema`]: crate::canonical_node::ValueSchema
//! [`ValueSchema::has`]: crate::canonical_node::ValueSchema::has
//! [`NodeGuid::CLASSID_FMA`]: crate::canonical_node::NodeGuid::CLASSID_FMA
//! [`NodeGuid::read_mode`]: crate::canonical_node::NodeGuid::read_mode
//! [`classid_read_mode`]: crate::canonical_node::classid_read_mode

use crate::canonical_node::{classid_read_mode, NodeRow, ValueSchema, ValueTenant};

/// `true` iff an `f32` bit pattern is non-finite (Inf or NaN): the exponent
/// field is all-ones. No float materialised.
#[inline]
pub const fn f32_bits_nonfinite(bits: u32) -> bool {
    (bits & 0x7F80_0000) == 0x7F80_0000
}

/// The result of projecting an SoA batch onto the NaN-detection surface.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NanReport {
    /// Boards whose `Energy` tenant was actually inspected (their resolved
    /// schema materialises `Energy`). Excludes [`Self::skipped`].
    pub total: usize,
    /// Board indices whose `Energy` tenant is non-finite (NaN or Inf). A subset
    /// of the inspected (non-skipped) boards.
    pub nonfinite: Vec<u32>,
    /// Boards whose resolved schema does NOT materialise `Energy` — excluded
    /// from the finiteness question entirely (not counted as clean, not
    /// counted as dirty; genuinely not-applicable). Nonzero only in batches
    /// mixing classids whose read-mode omits `Energy` (e.g.
    /// [`ValueSchema::Compressed`]) with ones that carry it.
    ///
    /// [`ValueSchema::Compressed`]: crate::canonical_node::ValueSchema::Compressed
    pub skipped: usize,
}

impl NanReport {
    /// No inspected board went non-finite. (Silent about `skipped` boards by
    /// design — they were never inspected, so they cannot make the sweep dirty.)
    #[inline]
    pub fn is_clean(&self) -> bool {
        self.nonfinite.is_empty()
    }

    /// Count of non-finite boards.
    #[inline]
    pub fn count(&self) -> usize {
        self.nonfinite.len()
    }
}

/// Read one board's `Energy` tenant as a raw `f32` bit pattern (no float load).
/// Caller MUST have already confirmed the population's schema materialises
/// `Energy` — this function does not gate.
#[inline]
fn energy_bits(row: &NodeRow) -> u32 {
    let off = ValueTenant::Energy.value_offset();
    u32::from_le_bytes([
        row.value[off],
        row.value[off + 1],
        row.value[off + 2],
        row.value[off + 3],
    ])
}

/// Project a population whose reading is ALREADY RESOLVED onto the
/// NaN-detection surface. `schema` is the population's value schema (for a
/// [`crate::hotplug::ResolvedReading`], its `read_mode.value_schema`); every
/// row is read under it. No registry or read-mode lookup happens here.
///
/// If `schema` does not materialise `Energy`, no `Energy` bytes are read: every
/// row is `skipped` and the report is clean. Otherwise each row's `Energy` is
/// tested with the integer exponent mask.
///
/// The caller owns the homogeneity claim. For a batch that may mix classids,
/// use [`project_energy_nonfinite`].
pub fn project_energy_nonfinite_resolved(rows: &[NodeRow], schema: ValueSchema) -> NanReport {
    if !schema.has(ValueTenant::Energy) {
        return NanReport {
            total: 0,
            nonfinite: Vec::new(),
            skipped: rows.len(),
        };
    }
    let mut nonfinite = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if f32_bits_nonfinite(energy_bits(row)) {
            nonfinite.push(i as u32);
        }
    }
    NanReport {
        total: rows.len(),
        nonfinite,
        skipped: 0,
    }
}

/// Clean/dirty answer for a population whose reading is already resolved —
/// the sibling of [`project_energy_nonfinite_resolved`]. Early-outs on the
/// first non-finite board; `true` without reading anything when `schema` has
/// no `Energy`.
pub fn energy_all_finite_resolved(rows: &[NodeRow], schema: ValueSchema) -> bool {
    !schema.has(ValueTenant::Energy) || rows.iter().all(|row| !f32_bits_nonfinite(energy_bits(row)))
}

/// Split `rows` into maximal runs of equal classid, resolving each run's
/// value schema once. Yields `(start index, run, schema)`.
fn schema_runs(rows: &[NodeRow]) -> impl Iterator<Item = (usize, &[NodeRow], ValueSchema)> {
    let mut start = 0usize;
    core::iter::from_fn(move || {
        if start >= rows.len() {
            return None;
        }
        let classid = rows[start].key.classid();
        let len = rows[start..]
            .iter()
            .take_while(|r| r.key.classid() == classid)
            .count();
        let run = &rows[start..start + len];
        let at = start;
        start += len;
        Some((at, run, classid_read_mode(classid).value_schema))
    })
}

/// Project a batch of canonical boards onto the NaN-detection surface by reading
/// each one's `Energy` tenant — schema-gated per row (see module docs). Read-only;
/// returns the indices of non-finite boards among those actually inspected.
/// This is the demoted singleton BindSpace — a projection, never a carrier.
///
/// Accepts a batch mixing classids. Each run of equal classid is resolved once
/// and handed to [`project_energy_nonfinite_resolved`]; a caller that already
/// knows its population's reading should call that directly.
pub fn project_energy_nonfinite(rows: &[NodeRow]) -> NanReport {
    let mut report = NanReport::default();
    for (at, run, schema) in schema_runs(rows) {
        let r = project_energy_nonfinite_resolved(run, schema);
        report.total += r.total;
        report.skipped += r.skipped;
        report
            .nonfinite
            .extend(r.nonfinite.into_iter().map(|i| i + at as u32));
    }
    report
}

/// Fast clean/dirty answer without materialising the index list — the cheapest
/// projection (early-outs on the first non-finite board). Rows whose schema
/// omits `Energy` are skipped, not treated as a violation. Mixed batches are
/// resolved once per run of equal classid.
pub fn energy_all_finite(rows: &[NodeRow]) -> bool {
    schema_runs(rows).all(|(_, run, schema)| energy_all_finite_resolved(run, schema))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_node::{EdgeBlock, NodeGuid};

    // Fixtures use `CLASSID_OSINT` (a stable, permanently-Cognitive read-mode
    // that has always materialised `Energy`) rather than `NodeGuid::local(0)`
    // (classid 0 / DEFAULT). `ReadMode::DEFAULT` is documented as a TEMPORARY
    // POC pin to `ValueSchema::Full`, scheduled to flip back to `Bootstrap`
    // (no tenants) once the POC ends — a test fixture pinned to DEFAULT would
    // silently go vacuous (every row skipped, `is_clean()` trivially true) on
    // that flip. `CLASSID_OSINT`'s Cognitive schema carries no such sunset.
    fn board_with(energy: f32) -> NodeRow {
        board_with_classid(NodeGuid::CLASSID_OSINT, energy)
    }

    fn board_with_classid(classid: u32, energy: f32) -> NodeRow {
        let mut row = NodeRow {
            key: NodeGuid::new(classid, 0, 0, 0, 0, 0),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        };
        let off = ValueTenant::Energy.value_offset();
        row.value[off..off + 4].copy_from_slice(&energy.to_le_bytes());
        row
    }

    #[test]
    fn finite_batch_is_clean() {
        let rows: Vec<NodeRow> = (0..8).map(|i| board_with(i as f32)).collect();
        let r = project_energy_nonfinite(&rows);
        assert!(r.is_clean());
        assert_eq!(r.total, 8);
        // Inertness half of the T5 gate: an all-Energy-bearing batch is swept
        // in full — the schema gate skips nothing here.
        assert_eq!(r.skipped, 0);
        assert!(energy_all_finite(&rows));
    }

    #[test]
    fn nan_and_inf_are_flagged_neg_inf_too() {
        let rows = vec![
            board_with(1.0),
            board_with(f32::NAN),
            board_with(f32::INFINITY),
            board_with(0.0),
            board_with(f32::NEG_INFINITY),
        ];
        let r = project_energy_nonfinite(&rows);
        assert_eq!(r.nonfinite, vec![1, 2, 4]);
        assert_eq!(r.count(), 3);
        assert_eq!(r.skipped, 0);
        assert!(!r.is_clean());
        assert!(!energy_all_finite(&rows));
    }

    #[test]
    fn subnormal_and_zero_are_finite() {
        // exponent-zero patterns (zero, subnormals) must NOT be flagged
        let rows = vec![
            board_with(0.0),
            board_with(-0.0),
            board_with(f32::MIN_POSITIVE),
        ];
        assert!(project_energy_nonfinite(&rows).is_clean());
    }

    // ── T5 closure: the schema gate on the two fixed-offset sweepers ──────────

    #[test]
    fn schema_gate_excludes_boards_whose_schema_omits_energy() {
        // Real, registered classids — not a synthetic override — so this
        // exercises the actual `classid_read_mode` registry, not a stand-in.
        // CLASSID_OSINT → Cognitive (has Energy). CLASSID_FMA → Compressed
        // (Fingerprint + Helix + Turbovec + EntityType — no Energy).
        let clean = board_with_classid(NodeGuid::CLASSID_OSINT, 1.0);
        let poisoned_but_out_of_schema_1 = board_with_classid(NodeGuid::CLASSID_FMA, f32::NAN);
        let poisoned_but_out_of_schema_2 = board_with_classid(NodeGuid::CLASSID_FMA, f32::INFINITY);

        // Prove the poison is real: an ungated read of these same reserved
        // bytes IS non-finite. If this assertion ever failed, the test below
        // would pass for the wrong reason (nothing to gate against).
        assert!(f32_bits_nonfinite(energy_bits(
            &poisoned_but_out_of_schema_1
        )));
        assert!(f32_bits_nonfinite(energy_bits(
            &poisoned_but_out_of_schema_2
        )));

        let rows = vec![
            clean,
            poisoned_but_out_of_schema_1,
            poisoned_but_out_of_schema_2,
        ];

        let r = project_energy_nonfinite(&rows);
        // Falsifier: without the gate, `total` would be 3 and `nonfinite`
        // would contain indices 1 and 2.
        assert_eq!(r.total, 1, "only the OSINT/Cognitive board is inspected");
        assert_eq!(r.skipped, 2, "both FMA/Compressed boards are out of schema");
        assert!(
            r.nonfinite.is_empty(),
            "the poisoned bytes must never surface — they aren't Energy under this row's schema"
        );
        assert!(r.is_clean());
        assert!(
            energy_all_finite(&rows),
            "energy_all_finite must agree with project_energy_nonfinite"
        );
    }

    // ── Resolved path: schema resolved once by the caller ─────────────────────

    #[test]
    fn resolved_path_does_no_read_mode_lookup() {
        use crate::canonical_node::{read_mode_lookups, reset_read_mode_lookups};
        let cognitive = classid_read_mode(NodeGuid::CLASSID_OSINT).value_schema;
        assert!(cognitive.has(ValueTenant::Energy));
        for n in [1usize, 1000] {
            let mut rows: Vec<NodeRow> = (0..n).map(|i| board_with(i as f32)).collect();
            rows[n - 1] = board_with(f32::NAN);
            reset_read_mode_lookups();
            let r = project_energy_nonfinite_resolved(&rows, cognitive);
            let clean = energy_all_finite_resolved(&rows, cognitive);
            assert_eq!(read_mode_lookups(), 0, "n = {n}");
            // anti-vacuity: the rows were actually read
            assert_eq!(r.total, n);
            assert_eq!(r.nonfinite, vec![(n - 1) as u32]);
            assert!(!clean);
        }
    }

    #[test]
    fn resolved_path_keeps_the_schema_gate() {
        let compressed = classid_read_mode(NodeGuid::CLASSID_FMA).value_schema;
        assert!(!compressed.has(ValueTenant::Energy));
        let rows = vec![
            board_with_classid(NodeGuid::CLASSID_FMA, f32::NAN),
            board_with_classid(NodeGuid::CLASSID_FMA, f32::INFINITY),
        ];
        // the bytes really are poisoned, so a missing gate would report them
        assert!(f32_bits_nonfinite(energy_bits(&rows[0])));
        let r = project_energy_nonfinite_resolved(&rows, compressed);
        assert_eq!((r.total, r.skipped), (0, 2));
        assert!(r.nonfinite.is_empty());
        assert!(energy_all_finite_resolved(&rows, compressed));
    }

    #[test]
    fn mixed_wrapper_resolves_once_per_classid_run() {
        use crate::canonical_node::{read_mode_lookups, reset_read_mode_lookups};
        for n in [1usize, 1000] {
            let rows: Vec<NodeRow> = (0..n).map(|i| board_with(i as f32)).collect();
            reset_read_mode_lookups();
            let r = project_energy_nonfinite(&rows);
            assert_eq!(read_mode_lookups(), 1, "homogeneous batch, n = {n}");
            assert_eq!(r.total, n);
            reset_read_mode_lookups();
            assert!(energy_all_finite(&rows));
            assert_eq!(read_mode_lookups(), 1, "homogeneous batch, n = {n}");
        }
    }

    #[test]
    fn mixed_wrapper_reports_indices_into_the_whole_batch() {
        let rows = vec![
            board_with_classid(NodeGuid::CLASSID_FMA, f32::NAN),
            board_with_classid(NodeGuid::CLASSID_OSINT, f32::NAN),
            board_with_classid(NodeGuid::CLASSID_FMA, f32::NAN),
            board_with_classid(NodeGuid::CLASSID_OSINT, 1.0),
            board_with_classid(NodeGuid::CLASSID_OSINT, f32::INFINITY),
        ];
        let r = project_energy_nonfinite(&rows);
        assert_eq!(r.nonfinite, vec![1, 4]);
        assert_eq!((r.total, r.skipped), (3, 2));
        assert!(!energy_all_finite(&rows));
        // the FMA rows alone are skipped, not read
        assert!(energy_all_finite(&[rows[0], rows[2]]));
    }
}
