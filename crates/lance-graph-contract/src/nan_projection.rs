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
//! **What "branchless" still means after the gate, precisely (codex review,
//! 2026-07-30).** The FINITENESS TEST — the exponent-mask compare on the
//! four already-loaded bytes — is unchanged: still zero branches on the
//! value. That is not the same claim as "the sweep costs what it did
//! before." [`row_has_energy`] calls [`NodeGuid::read_mode`], which resolves
//! through [`classid_read_mode`] — a `HashMap` lookup behind a `LazyLock`,
//! not a bitmask. That lookup is real per-row work, added on top of the old
//! four-byte load, and can plausibly dominate it for an in-cache homogeneous
//! batch. Not benchmarked; do not read "branchless" below as "free" — if
//! this projection lands on a genuinely hot path, that lookup is the first
//! place to look before assuming the schema gate is costless.
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

use crate::canonical_node::{NodeRow, ValueTenant};

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
/// Caller MUST have already confirmed the row's schema materialises `Energy`
/// ([`row_has_energy`]) — this function does not gate.
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

/// Does this row's OWN resolved schema materialise `Energy`? The one branch
/// this module adds — on schema presence, never on the float value.
#[inline]
fn row_has_energy(row: &NodeRow) -> bool {
    row.key.read_mode().value_schema.has(ValueTenant::Energy)
}

/// Project a batch of canonical boards onto the NaN-detection surface by reading
/// each one's `Energy` tenant — schema-gated per row (see module docs). Read-only;
/// returns the indices of non-finite boards among those actually inspected.
/// This is the demoted singleton BindSpace — a projection, never a carrier.
pub fn project_energy_nonfinite(rows: &[NodeRow]) -> NanReport {
    let mut total = 0usize;
    let mut skipped = 0usize;
    let mut nonfinite = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if !row_has_energy(row) {
            skipped += 1;
            continue;
        }
        total += 1;
        if f32_bits_nonfinite(energy_bits(row)) {
            nonfinite.push(i as u32);
        }
    }
    NanReport {
        total,
        nonfinite,
        skipped,
    }
}

/// Fast clean/dirty answer without materialising the index list — the cheapest
/// projection (early-outs on the first non-finite board). Rows whose schema
/// omits `Energy` are skipped, not treated as a violation.
pub fn energy_all_finite(rows: &[NodeRow]) -> bool {
    rows.iter()
        .filter(|row| row_has_energy(row))
        .all(|row| !f32_bits_nonfinite(energy_bits(row)))
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
}
