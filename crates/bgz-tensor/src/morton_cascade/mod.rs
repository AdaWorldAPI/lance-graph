//! Morton-cascade inverse-pyramid Fisher-z reading over the V3 L4 palette tenant
//! — the CLEAN V3 thinking compute that REPLACES running the cluttered V1
//! `CausalEdge64` (3×8 bare indices) over the V3 substrate.
//!
//! Operator ruling: *"using cluttered V1 causaledge64 over v3 substrate is
//! extremely poor design … keep the old and add a `_v3` … and compare its
//! results … lazylock dispatch."* So this module is the **keep-old + add-v3 +
//! dispatch + compare** pattern:
//!
//!   - [`v3`] — the clean V3 path: reads the **full-resolution 3×256² SPO +
//!     3×256² AriGraph-basin true-centroid PAIRS** and scores them through the
//!     256×256 [`FisherZTable`] pairwise distribution over the Morton
//!     inverse-pyramid cascade.
//!   - [`legacy`] — the V1 path: the `3×8` **bare-index** read (only the first
//!     byte of each pair, as `CausalEdge64::{s_idx,p_idx,o_idx}` did). Kept for
//!     comparison; it structurally CANNOT see the pairwise partner (the `.1` of
//!     each pair = the AriGraph/full-resolution half) — which is exactly why
//!     running V1 over the V3 substrate is lossy.
//!   - [`read`] dispatches via [`BACKEND`] (LazyLock, default V3); [`compare`]
//!     runs both.
//!
//! The tenant carving is the established one (le-contract §3 L4 `palette256²`):
//! `6×(u8:u8)` = **3 SPO pairs (the fact) + 3 episodic-witness/AriGraph-basin
//! pairs (the support)**.

pub mod legacy;
pub mod v3;

use crate::fisher_z::FisherZTable;
use std::sync::LazyLock;

/// The V3 L4 palette tenant: 12 bytes = `6×(u8:u8)` true-centroid pairs, carved
/// as 3 SPO + 3 episodic-witness/AriGraph-basin (full-resolution SPO + AriGraph).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct L4Tenant {
    /// 3×256² SPO true-centroid pairs — the SPO 2³ fact.
    pub spo: [(u8, u8); 3],
    /// 3×256² AriGraph SPO-G / episodic-witness basin pairs — the support.
    pub basins: [(u8, u8); 3],
}

impl L4Tenant {
    /// Read the 12-byte L4 slice of a node's value tenant.
    pub fn from_bytes(b: &[u8; 12]) -> Self {
        L4Tenant {
            spo: [(b[0], b[1]), (b[2], b[3]), (b[4], b[5])],
            basins: [(b[6], b[7]), (b[8], b[9]), (b[10], b[11])],
        }
    }
    /// Serialize back to the 12-byte L4 slice.
    pub fn to_bytes(&self) -> [u8; 12] {
        [
            self.spo[0].0,
            self.spo[0].1,
            self.spo[1].0,
            self.spo[1].1,
            self.spo[2].0,
            self.spo[2].1,
            self.basins[0].0,
            self.basins[0].1,
            self.basins[1].0,
            self.basins[1].1,
            self.basins[2].0,
            self.basins[2].1,
        ]
    }
}

/// A cognitive reading of the tenant: the SPO fact's internal coherence, the
/// basin support toward it, and the net (negative ⇒ inverted/ironic — the
/// qualia −8..+8 inversion the operator named).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    pub fact_coherence: f32,
    pub basin_support: f32,
    pub net: f32,
}

/// Which substrate computes the reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The clean V3 full-resolution 3×256² true-centroid path.
    V3,
    /// The legacy V1 3×8 bare-index path (lossy on the V3 substrate).
    Legacy,
}

/// LazyLock dispatch: default V3; override with `BGZ_MORTON_BACKEND=legacy`
/// (or `p64`) for the legacy comparison path. Resolved once per process.
pub static BACKEND: LazyLock<Backend> =
    LazyLock::new(|| match std::env::var("BGZ_MORTON_BACKEND").as_deref() {
        Ok("legacy") | Ok("p64") | Ok("v1") => Backend::Legacy,
        _ => Backend::V3,
    });

/// Read the tenant through the active backend (see [`BACKEND`]).
pub fn read(tenant: &L4Tenant, fz: &FisherZTable) -> Reading {
    match *BACKEND {
        Backend::V3 => v3::read(tenant, fz),
        Backend::Legacy => legacy::read(tenant, fz),
    }
}

/// Compare thinking: run BOTH substrates on the same tenant. Returns
/// `(v3, legacy)`. When the pair partners (`.1`) carry signal the V1 bare-index
/// read cannot see, the two Readings DIVERGE — the measured cost of running V1
/// over V3.
pub fn compare(tenant: &L4Tenant, fz: &FisherZTable) -> (Reading, Reading) {
    (v3::read(tenant, fz), legacy::read(tenant, fz))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth 256-centroid codebook so neighbouring indices correlate.
    fn table() -> FisherZTable {
        let dim = 32usize;
        let reps: Vec<Vec<f32>> = (0..256)
            .map(|c| {
                (0..dim)
                    .map(|d| ((c as f32) * 0.19 + (d as f32) * 0.37).sin())
                    .collect()
            })
            .collect();
        FisherZTable::build(&reps, 256)
    }

    #[test]
    fn tenant_round_trips_12_bytes_3spo_3basin() {
        let bytes = [10u8, 12, 11, 13, 12, 14, 12, 15, 13, 11, 14, 12];
        let t = L4Tenant::from_bytes(&bytes);
        assert_eq!(t.to_bytes(), bytes);
        assert_eq!(t.spo.len(), 3);
        assert_eq!(t.basins.len(), 3);
    }

    #[test]
    fn v3_sees_pairwise_that_legacy_cannot() {
        let fz = table();
        // pairs whose SECOND byte (the AriGraph/full-res half) carries signal
        // the V1 bare-index read drops.
        let t = L4Tenant::from_bytes(&[10, 200, 11, 205, 12, 195, 12, 15, 13, 11, 14, 12]);
        let (v3r, legr) = compare(&t, &fz);
        // the V3 fact-coherence reads the (a,b) pairwise; legacy reads only a→a-ish.
        assert!(
            (v3r.fact_coherence - legr.fact_coherence).abs() > 1e-3,
            "V3 full-res read must differ from the V1 bare-index read when the pair partners carry signal"
        );
    }

    #[test]
    fn v3_and_legacy_agree_on_the_degenerate_diagonal() {
        let fz = table();
        // a==b in every pair ⇒ the pairwise collapses to the bare index; the two
        // backends must AGREE (the legacy read is the diagonal of the V3 read).
        let t = L4Tenant {
            spo: [(10, 10), (11, 11), (12, 12)],
            basins: [(13, 13), (14, 14), (15, 15)],
        };
        let (v3r, legr) = compare(&t, &fz);
        assert!(
            (v3r.fact_coherence - legr.fact_coherence).abs() < 1e-6,
            "on the a==b diagonal the V3 pairwise read collapses to the V1 bare-index read"
        );
    }

    #[test]
    fn backend_defaults_to_v3() {
        // no env override in the test harness ⇒ V3.
        assert_eq!(*BACKEND, Backend::V3);
    }
}
