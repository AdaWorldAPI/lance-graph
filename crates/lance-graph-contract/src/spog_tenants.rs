//! **Multitenant SPOG — G via classid.** The "octopus" substrate, agnostic.
//!
//! Operator-ruled (2026-08-31, verbatim): *"octopus 8 domains is just
//! multitenant SPOG"* / *"g via classid"* / *"nicht 8 mal handrolling"* —
//! ONE substrate, N tenants, never N copies. And the demarcation that keeps
//! this crate honest: *"Medcare is everything that can't be agnostic — if
//! thinking can't be agnostic in lancegraph it's still handrolled."*
//!
//! # What "G via classid" means, mechanically
//!
//! A SPOG quad needs no fourth stored column: the graph/tenant coordinate is
//! READ from the key itself — [`graph_of`] is the canon-high concept half of
//! the classid (`classid >> 16`). Every node already carries its graph the
//! way it already carries its class; multitenancy is routing, not schema.
//!
//! # What a tenant IS here
//!
//! One [`AlphaOverlay`] shadow per tenant concept, all over ONE shared
//! [`AlphaAllocation`] — the same reserve-don't-claim thin provisioning the
//! rung tunnel uses, keyed by GRAPH instead of by rung. A domain leg and a
//! patient leg are the same thing: a tenant. The N+1st leg is not special
//! machinery, it is one more concept id.
//!
//! # What this module deliberately does NOT know
//!
//! What a concept id MEANS. The meaning of a tenant (e.g. "phenotype") is
//! minted in ogar-vocab and bound by the consumer that loads the domain —
//! here a tenant key is an opaque `u16`. That is the agnostic line: the
//! substrate routes by it, never interprets it. Operator, verbatim: *"data
//! as config via ogar ogar-vocab"* — tenant bindings are DATA resolved
//! through the codebook, never hardcoded literals in any crate.

use crate::alpha::{AlphaAddr, AlphaAllocation, AlphaClaim, AlphaError, AlphaOverlay, AlphaStamp};

/// The graph coordinate of an address — the canon-high concept half of its
/// classid. No fourth column: G is read from the key.
#[must_use]
pub const fn graph_of(addr: AlphaAddr) -> u16 {
    (addr.classid() >> 16) as u16
}

/// What became of one tenant-routed claim.
#[derive(Debug)]
pub enum TenantClaim {
    /// Routed to the tenant owning the address's graph; the claim stands.
    Routed(u16, AlphaClaim),
    /// The address's graph has no tenant here — reported, never silently
    /// absorbed into a wrong shadow.
    NoTenant(u16),
    /// The substrate itself said no (e.g. an unallocated address).
    Substrate(AlphaError),
}

impl TenantClaim {
    /// Did the claim land in a tenant shadow?
    #[must_use]
    pub fn routed(&self) -> bool {
        matches!(self, TenantClaim::Routed(..))
    }
}

/// N tenant shadows over ONE allocation — the multitenant SPOG.
pub struct SpogTenants<'a> {
    /// `(concept, shadow)` in the caller's declaration order — which is the
    /// merge order, so the caller's order is load-bearing and deterministic.
    tenants: Vec<(u16, AlphaOverlay<'a>)>,
}

impl<'a> SpogTenants<'a> {
    /// One shadow per DISTINCT concept, all borrowing the same allocation.
    /// Duplicate concepts collapse to the first occurrence (one substrate,
    /// never two shadows for one graph).
    #[must_use]
    pub fn over(alloc: &'a AlphaAllocation<'a>, cycle: u32, concepts: &[u16]) -> Self {
        let mut tenants: Vec<(u16, AlphaOverlay<'a>)> = Vec::new();
        for &c in concepts {
            if !tenants.iter().any(|(k, _)| *k == c) {
                tenants.push((c, AlphaOverlay::over_shared(alloc, cycle)));
            }
        }
        Self { tenants }
    }

    /// Route a claim to the tenant owning `graph_of(addr)`.
    pub fn claim(&mut self, addr: AlphaAddr, rung: u8) -> TenantClaim {
        let g = graph_of(addr);
        let Some((_, shadow)) = self.tenants.iter_mut().find(|(k, _)| *k == g) else {
            return TenantClaim::NoTenant(g);
        };
        match shadow.claim(addr, rung) {
            Ok(c) => TenantClaim::Routed(g, c),
            Err(e) => TenantClaim::Substrate(e),
        }
    }

    /// A tenant's shadow, readable.
    #[must_use]
    pub fn tenant(&self, concept: u16) -> Option<&AlphaOverlay<'a>> {
        self.tenants
            .iter()
            .find(|(k, _)| *k == concept)
            .map(|(_, s)| s)
    }

    /// The declared tenant concepts, in declaration order.
    #[must_use]
    pub fn concepts(&self) -> Vec<u16> {
        self.tenants.iter().map(|(k, _)| *k).collect()
    }

    /// Total claims across all shadows.
    #[must_use]
    pub fn claimed_len(&self) -> usize {
        self.tenants.iter().map(|(_, s)| s.claimed_len()).sum()
    }

    /// Merge the shadows into one deterministic scanpath: declaration order,
    /// then per-shadow seq. Tenants are DISJOINT by construction (an address
    /// routes only to its own graph's shadow), so no cross-tenant revisit
    /// accounting exists to lose — a revisit within a tenant is already in
    /// its stamp's `visits`.
    #[must_use]
    pub fn merge(&self) -> Vec<(AlphaAddr, AlphaStamp)> {
        let mut out = Vec::with_capacity(self.claimed_len());
        for (_, shadow) in &self.tenants {
            for addr in shadow.scanpath() {
                if let Some(row) = shadow.get(addr) {
                    out.push((addr, crate::alpha::stamp_of(row)));
                }
            }
        }
        // Re-seq globally: per-shadow seqs collide (each starts at 0).
        for (i, (_, st)) in out.iter_mut().enumerate() {
            st.seq = u32::try_from(i).unwrap_or(u32::MAX);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_node::{EdgeBlock, NodeGuid, NodeRow};

    /// Three graphs over one spine: concepts 0x0301, 0x0302, and a
    /// "patient-leg-shaped" 0x0900 — deliberately just another tenant.
    fn base() -> Vec<NodeRow> {
        let mut rows = Vec::new();
        for (g, n) in [(0x0301u32, 4u32), (0x0302, 3), (0x0900, 2), (0x0777, 1)] {
            for i in 0..n {
                rows.push(NodeRow {
                    key: NodeGuid::new(g << 16, 1, 2, 3, 0x66, (g << 8) + i + 1),
                    edges: EdgeBlock::default(),
                    value: [0u8; 480],
                });
            }
        }
        rows
    }

    /// G is READ from the key — no fourth column anywhere.
    #[test]
    fn the_graph_coordinate_is_read_from_the_classid() {
        let b = base();
        assert_eq!(graph_of(b[0].key), 0x0301);
        assert_eq!(graph_of(b[7].key), 0x0900);
    }

    /// **The multitenancy falsifier, two-sided.** A claim lands in ITS
    /// graph's shadow and in NO other; a graph without a tenant is REPORTED
    /// (`NoTenant`), never absorbed. Disable-verified: routing every claim
    /// to the first tenant (ignoring `graph_of`) fails the isolation arm.
    #[test]
    fn a_claim_lands_in_its_own_graphs_shadow_and_nowhere_else() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut t = SpogTenants::over(&alloc, 3, &[0x0301, 0x0302, 0x0900]);

        assert!(t.claim(b[0].key, 1).routed(), "0x0301 row routes");
        assert!(t.claim(b[4].key, 2).routed(), "0x0302 row routes");
        assert!(
            t.claim(b[7].key, 4).routed(),
            "the patient leg is just another tenant"
        );

        // Isolation: each shadow holds exactly its own graph's claim.
        for (concept, its_key, other_key) in [
            (0x0301u16, b[0].key, b[4].key),
            (0x0302, b[4].key, b[0].key),
            (0x0900, b[7].key, b[0].key),
        ] {
            let s = t.tenant(concept).expect("tenant exists");
            assert!(s.get(its_key).is_some(), "{concept:#06x} holds its own");
            assert!(
                s.get(other_key).is_none(),
                "{concept:#06x} holds NOTHING foreign"
            );
            assert_eq!(s.claimed_len(), 1);
        }

        // A graph nobody declared is reported, not swallowed.
        match t.claim(b[9].key, 1) {
            TenantClaim::NoTenant(g) => assert_eq!(g, 0x0777),
            other => panic!("expected NoTenant, got {other:?}"),
        }
        assert_eq!(t.claimed_len(), 3, "the stray claim landed nowhere");
    }

    /// One substrate, never two shadows for one graph: duplicate concepts
    /// collapse; a revisit is counted in the ONE shadow's `visits`.
    #[test]
    fn duplicate_concepts_collapse_and_a_revisit_is_counted_once_in_one_shadow() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut t = SpogTenants::over(&alloc, 1, &[0x0301, 0x0301, 0x0301]);
        assert_eq!(t.concepts(), vec![0x0301], "nicht 3 mal handrolling");
        assert!(t.claim(b[0].key, 1).routed());
        assert!(t.claim(b[0].key, 2).routed(), "a revisit routes too");
        assert_eq!(t.claimed_len(), 1, "one address, one row");
        let st = crate::alpha::stamp_of(t.tenant(0x0301).unwrap().get(b[0].key).unwrap());
        assert_eq!(st.visits, 2, "the return is counted");
        assert_eq!(st.rung, 1, "the first stamp is kept");
    }

    /// Merge is deterministic and ordered by tenant DECLARATION order, then
    /// per-shadow visit order; seq is re-issued globally.
    #[test]
    fn merge_is_declaration_ordered_and_deterministic() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut t = SpogTenants::over(&alloc, 2, &[0x0302, 0x0301]);
        assert!(t.claim(b[0].key, 1).routed()); // 0x0301
        assert!(t.claim(b[4].key, 1).routed()); // 0x0302 — declared FIRST
        assert!(t.claim(b[1].key, 1).routed()); // 0x0301, second visit order
        let m = t.merge();
        let addrs: Vec<AlphaAddr> = m.iter().map(|(a, _)| *a).collect();
        assert_eq!(
            addrs,
            vec![b[4].key, b[0].key, b[1].key],
            "0x0302 first (declaration order), then 0x0301 in visit order"
        );
        let seqs: Vec<u32> = m.iter().map(|(_, s)| s.seq).collect();
        assert_eq!(seqs, vec![0, 1, 2], "seq re-issued globally");
        assert_eq!(t.merge(), m, "deterministic");
    }
}
