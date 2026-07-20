//! The **clean V3** Morton-cascade reading: full-resolution `3×256²` SPO +
//! `3×256²` AriGraph-basin true-centroid PAIRS, scored through the 256×256
//! [`FisherZTable`](crate::fisher_z::FisherZTable) pairwise distribution over
//! the Morton 2bit×2bit inverse-pyramid cascade.
//!
//! Each SPO / basin role is a `(a, b)` centroid PAIR — a point in the pairwise
//! centroid distribution, i.e. a *relation* the Fisher-z table scores directly.
//! This is the reimagined SPO (operator): `3×256²` true centroids, NOT the V1
//! `3×8` bare indices. No `CausalEdge64` is read here — the V3 substrate gets a
//! clean compute, not the cluttered V1 register.

use super::{L4Tenant, Reading};
use crate::fisher_z::FisherZTable;

/// Morton 2bit×2bit → 4-bit cascade cell (x,y each 2 bits, interleaved).
/// Bijective over 0..16; the inverse pyramid reads cells coarse→fine.
#[inline]
pub fn morton2(x: u8, y: u8) -> u8 {
    let mut d = 0u8;
    for i in 0..2 {
        d |= ((x >> i) & 1) << (2 * i);
        d |= ((y >> i) & 1) << (2 * i + 1);
    }
    d & 0x0F
}

/// Read the tenant with the full-resolution pairwise centroids.
pub fn read(tenant: &L4Tenant, fz: &FisherZTable) -> Reading {
    // fact coherence: each SPO role's OWN pair (a,b) scored pairwise — the
    // true 256² centroid relation, read in Morton cascade order (coarse→fine).
    let mut coh = 0.0f32;
    for (i, &(a, b)) in tenant.spo.iter().enumerate() {
        // the cascade cell selects the pyramid depth this pair reads at; the
        // reduction (mean) is order-independent, but the cell keeps the
        // inverse-pyramid structure explicit for downstream shaders.
        let _cell = morton2((i as u8) & 0b11, ((i as u8) >> 2) & 0b11);
        coh += fz.lookup_f32(a, b);
    }
    let fact_coherence = coh / 3.0;

    // basin support: each AriGraph/episodic basin pair's BOTH centroids scored
    // against ALL SPO centroids (both bytes of every SPO pair). This is the
    // full-resolution support the V1 bare-index read cannot compute.
    let spo_centroids = [
        tenant.spo[0].0,
        tenant.spo[0].1,
        tenant.spo[1].0,
        tenant.spo[1].1,
        tenant.spo[2].0,
        tenant.spo[2].1,
    ];
    let mut support = 0.0f32;
    for &(a, b) in &tenant.basins {
        let mut best = -1.0f32;
        for &c in &spo_centroids {
            best = best.max(fz.lookup_f32(a, c)).max(fz.lookup_f32(b, c));
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
