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

use crate::alpha::{
    AlphaAddr, AlphaAllocation, AlphaClaim, AlphaError, AlphaMask, AlphaOverlay, AlphaStamp,
};
use crate::canonical_node::NodeRow;

/// The graph coordinate of an address — the canon-high concept half of its
/// classid. No fourth column: G is read from the key.
#[must_use]
pub const fn graph_of(addr: AlphaAddr) -> u16 {
    (addr.classid() >> 16) as u16
}

/// The **block** a tenant belongs to — the high byte of its concept id.
///
/// A concept is `block:vocabulary` (`0x9101` = block `0x91`, vocabulary
/// `0x01`), so several tenants routinely share one block: measured on a real
/// consumer artifact, five distinct graphs resolved to one block. Grouping is
/// therefore a SHIFT on the key, never a second stored coordinate — the same
/// economy `graph_of` itself is.
///
/// What a block MEANS stays with the consumer that loaded the domain. This
/// crate groups by it and never interprets it.
#[must_use]
pub const fn block_of(concept: u16) -> u8 {
    (concept >> 8) as u8
}

/// **The tenant list as a census of the artifact, never as a table.**
///
/// Every row carries its graph in its own key, so the set of tenants a spine
/// needs is a *reading* of that spine — not a configuration beside it that
/// could disagree with it. This is what the module's "tenant bindings are
/// DATA" line buys structurally: here the data IS the bake.
///
/// Ascending, so the declaration order — which [`SpogTenants::merge`] makes
/// load-bearing — follows from the keys and never from a hash iteration.
///
/// This exists because the shape it answers to was measured rather than
/// assumed: a consumer's baked artifacts carry **5, 8 and 16 distinct graphs
/// in ONE file** (2026-09-07, over 60 478 / 7 641 / 762 041 rows). Bakes are
/// not one-per-graph, which is precisely the case [`SpogTenants`] exists for —
/// N tenants over ONE allocation, never N bakes and never N copies.
#[must_use]
pub fn census(rows: &[NodeRow]) -> Vec<u16> {
    let mut seen: Vec<u16> = rows.iter().map(|r| graph_of(r.key)).collect();
    seen.sort_unstable();
    seen.dedup();
    seen
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
    /// The ONE allocation every shadow borrows. Held so the mask surface has
    /// a base length even when no tenant was declared — an empty aufstellung
    /// must still answer "nothing attended, out of N addresses" rather than
    /// have no answer at all.
    alloc: &'a AlphaAllocation<'a>,
    /// **Which tenant took the n-th FRESH claim** — the one fact the shadows
    /// cannot hold.
    ///
    /// Each shadow numbers its own claims from 0, so per-shadow `seq` is
    /// exact WITHIN a graph and meaningless BETWEEN graphs: the interleaving
    /// of a saccade that crosses tenants is destroyed by the split, and no
    /// reading of the shadows can recover it. So it is recorded here, and
    /// only here.
    ///
    /// This is not a second projection of something already stored. It is the
    /// sole home of a fact that would otherwise be lost — the distinction the
    /// zero-copy law turns on. It costs `u16` per fresh claim and NO address:
    /// a tenant's own scanpath already carries which address, in order, so
    /// the tenant id plus a per-tenant cursor reconstructs the global saccade
    /// exactly ([`Self::merge_in_claim_order`]).
    ///
    /// Revisits are absent by construction — a revisit adds no position to any
    /// scanpath, so recording one here would desynchronise the cursors.
    route: Vec<u16>,
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
        Self {
            tenants,
            alloc,
            route: Vec::new(),
        }
    }

    /// **The no-configuration constructor**: one shadow per graph the
    /// allocation's own spine carries ([`census`]).
    ///
    /// With this there is no list to keep in step with the bake, so
    /// [`TenantClaim::NoTenant`] becomes structurally unreachable for any
    /// address of THIS spine — a claim can only miss a tenant if the caller
    /// declared a narrower set on purpose.
    #[must_use]
    pub fn over_census(alloc: &'a AlphaAllocation<'a>, cycle: u32) -> Self {
        let concepts = census(alloc.base());
        Self::over(alloc, cycle, &concepts)
    }

    /// Route a claim to the tenant owning `graph_of(addr)`.
    pub fn claim(&mut self, addr: AlphaAddr, rung: u8) -> TenantClaim {
        let g = graph_of(addr);
        let Some((_, shadow)) = self.tenants.iter_mut().find(|(k, _)| *k == g) else {
            return TenantClaim::NoTenant(g);
        };
        match shadow.claim(addr, rung) {
            Ok(c) => {
                if c.fresh {
                    self.route.push(g);
                }
                TenantClaim::Routed(g, c)
            }
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

    /// The tenants of one block, in declaration order — grouping by
    /// [`block_of`], a shift on the key.
    #[must_use]
    pub fn tenants_in_block(&self, block: u8) -> Vec<u16> {
        self.tenants
            .iter()
            .map(|(k, _)| *k)
            .filter(|&c| block_of(c) == block)
            .collect()
    }

    /// The allocation every shadow borrows — the ONE address space.
    #[must_use]
    pub fn allocation(&self) -> &'a AlphaAllocation<'a> {
        self.alloc
    }

    /// **One tenant's population, as a mask.** [`None`] for an undeclared
    /// graph — an absent tenant is not an empty one, and answering an empty
    /// mask would make "this graph has no shadow" indistinguishable from
    /// "this graph was never looked at".
    #[must_use]
    pub fn tenant_mask(&self, concept: u16) -> Option<AlphaMask> {
        self.tenant(concept).map(AlphaOverlay::attended_mask)
    }

    /// Everything any tenant attended, as one mask — the SPOG half of the
    /// rung × tenant cross ([`crate::alpha_focus`]).
    ///
    /// Recomputed, never stored: the shadows are the truth and a cached union
    /// would be a second reading of them.
    #[must_use]
    pub fn attended_mask(&self) -> AlphaMask {
        let mut m = AlphaMask::empty(self.alloc.base().len());
        for (_, s) in &self.tenants {
            m = m.or(&s.attended_mask());
        }
        m
    }

    /// Total claims across all shadows.
    #[must_use]
    pub fn claimed_len(&self) -> usize {
        self.tenants.iter().map(|(_, s)| s.claimed_len()).sum()
    }

    /// **The saccade as it happened**, across tenants — visit order, not
    /// declaration order.
    ///
    /// The sibling of [`merge`](Self::merge), and the two answer different
    /// questions: `merge` groups a thought BY GRAPH (every claim of one
    /// tenant together, which is what a per-graph reading wants); this
    /// replays it IN TIME (what attention did, in the order it did it), which
    /// is what a scanpath consumer and any order-sensitive replay wants.
    ///
    /// Reconstructed from [`Self::route`] plus each tenant's own scanpath —
    /// one cursor per tenant, advanced as its id comes up. `seq` is re-issued
    /// as the global position, so it means the same thing it means in a
    /// single overlay.
    #[must_use]
    pub fn merge_in_claim_order(&self) -> Vec<(AlphaAddr, AlphaStamp)> {
        let mut cursor: Vec<(u16, usize)> = self.tenants.iter().map(|(k, _)| (*k, 0)).collect();
        let mut out = Vec::with_capacity(self.route.len());
        for &g in &self.route {
            let Some((_, shadow)) = self.tenants.iter().find(|(k, _)| *k == g) else {
                continue;
            };
            let Some(slot) = cursor.iter_mut().find(|(k, _)| *k == g) else {
                continue;
            };
            let Some(addr) = shadow.scanpath().nth(slot.1) else {
                continue;
            };
            slot.1 += 1;
            if let Some(row) = shadow.get(addr) {
                let mut st = crate::alpha::stamp_of(row);
                st.seq = u32::try_from(out.len()).unwrap_or(u32::MAX);
                out.push((addr, st));
            }
        }
        out
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

    /// **The order falsifier.** `merge` groups a thought BY GRAPH; the
    /// interleaved saccade — what attention did, in time — is a different
    /// sequence, and only [`SpogTenants::merge_in_claim_order`] has it.
    ///
    /// Two-sided on purpose: the two readings must hold the SAME addresses
    /// (nothing invented, nothing dropped) and must NOT be in the same order
    /// (otherwise this method is decoration and the fixture is one that
    /// cannot tell them apart). A revisit must add no position to either.
    ///
    /// Disable-verified: dropping the `c.fresh` guard on `route.push` — so a
    /// revisit records a position — desynchronises the cursors and replays the
    /// saccade in the WRONG ORDER.
    ///
    /// The first version of this fixture revisited at the END and the disable
    /// stayed GREEN: the extra route entry simply ran the revisited tenant's
    /// cursor off its own scanpath, and the surplus was dropped. A revisit
    /// only produces an observable defect when another tenant claims after it
    /// and the revisited tenant claims again — so the fixture's SHAPE is part
    /// of what this test covers, not just its values.
    #[test]
    fn claim_order_replays_the_interleaved_saccade_that_declaration_order_loses() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        // Declaration order is deliberately NOT the visit order below.
        let mut t = SpogTenants::over(&alloc, 4, &[0x0900, 0x0302, 0x0301]);

        // Attention crosses tenants, RETURNS mid-way, and then goes on — and
        // that exact shape is what makes the revisit rule falsifiable. A
        // revisit followed by nothing, or followed only by more of the same
        // tenant, is absorbed by the cursor running off its own scanpath and
        // proves nothing; the wrong position only surfaces when a DIFFERENT
        // tenant claims after the revisit and the revisited one claims again.
        assert!(t.claim(b[0].key, 2).routed()); // 0x0301
        assert!(t.claim(b[4].key, 2).routed()); // 0x0302
        assert!(t.claim(b[0].key, 9).routed()); // 0x0301 AGAIN — not a position
        assert!(t.claim(b[7].key, 2).routed()); // 0x0900
        assert!(t.claim(b[1].key, 2).routed()); // 0x0301, after the crossing

        let timed: Vec<AlphaAddr> = t.merge_in_claim_order().iter().map(|(a, _)| *a).collect();
        assert_eq!(
            timed,
            vec![b[0].key, b[4].key, b[7].key, b[1].key],
            "the saccade replays in the order it happened, revisit adding nothing"
        );

        let grouped: Vec<AlphaAddr> = t.merge().iter().map(|(a, _)| *a).collect();
        assert_eq!(
            grouped,
            vec![b[7].key, b[4].key, b[0].key, b[1].key],
            "declaration order groups by graph"
        );

        // Same population, different sequence — the whole point.
        let mut a = timed.clone();
        let mut c = grouped.clone();
        a.sort_unstable_by_key(|k| (k.classid(), k.identity()));
        c.sort_unstable_by_key(|k| (k.classid(), k.identity()));
        assert_eq!(a, c, "nothing invented, nothing dropped");
        assert_ne!(
            timed, grouped,
            "anti-vacuity: a fixture where both readings agree proves nothing"
        );

        // seq is the global position in each reading.
        let seqs: Vec<u32> = t
            .merge_in_claim_order()
            .iter()
            .map(|(_, s)| s.seq)
            .collect();
        assert_eq!(seqs, vec![0, 1, 2, 3]);
        assert_eq!(
            t.merge_in_claim_order().len(),
            4,
            "one position per FRESH claim"
        );
        // ...and the RUNG is still the first visit's, never the revisit's.
        assert_eq!(t.merge_in_claim_order()[0].1.rung, 2, "first stamp kept");
        assert_eq!(
            t.merge_in_claim_order()[0].1.visits,
            2,
            "the return is counted"
        );
        assert_eq!(
            t.merge_in_claim_order(),
            t.merge_in_claim_order(),
            "deterministic"
        );
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
