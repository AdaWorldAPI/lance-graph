//! The **legacy V1** reading: the `3×8` bare-index path, kept for comparison.
//!
//! This is what running the cluttered `CausalEdge64` over the V3 substrate does.
//! V1 stored SPO as three single palette indices (`s_idx`/`p_idx`/`o_idx`, 8
//! bits each) — it has **no pairwise partner**. So on the V3 L4 tenant it can
//! only read the FIRST byte of each `(a, b)` pair and treats it as the
//! degenerate pair `(a, a)`: the `b` half (the AriGraph / full-resolution
//! centroid) is structurally invisible to it.
//!
//! The comparison in [`super::compare`] measures the cost: when the pair
//! partners carry signal, this read DIVERGES from (and is poorer than) the V3
//! read; only on the `a == b` diagonal do the two agree.

use super::{L4Tenant, Reading};
use crate::fisher_z::FisherZTable;

/// Read the tenant with V1 bare indices only (each pair collapsed to `(a, a)`).
pub fn read(tenant: &L4Tenant, fz: &FisherZTable) -> Reading {
    // fact coherence: each SPO role read as the degenerate pair (a, a) — the V1
    // bare index a, with no access to b. lookup_f32(a, a) is the self-cosine.
    let mut coh = 0.0f32;
    for &(a, _b) in &tenant.spo {
        coh += fz.lookup_f32(a, a);
    }
    let fact_coherence = coh / 3.0;

    // basin support: only the bare index (.0) of each basin pair, scored against
    // only the bare indices (.0) of the SPO pairs — the b half is invisible.
    let spo_indices = [tenant.spo[0].0, tenant.spo[1].0, tenant.spo[2].0];
    let mut support = 0.0f32;
    for &(a, _b) in &tenant.basins {
        let mut best = -1.0f32;
        for &c in &spo_indices {
            best = best.max(fz.lookup_f32(a, c));
        }
        support += best;
    }
    let basin_support = support / 3.0;

    let net = basin_support - (1.0 - fact_coherence).abs();
    Reading {
        fact_coherence,
        basin_support,
        net,
    }
}
