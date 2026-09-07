//! **The rung × tenant cross — where attention went, and where it did not.**
//!
//! `.claude/temporal/03-alpha-channel-state.md` records this as an absence,
//! measured: *"The rung × tenant cross does NOT exist … `SpogTenants` exposes
//! no mask surface of its own, and a grep for `rung.*tenant|tenant.*rung`
//! returns nothing but that signature. No structure crosses them, in code or
//! in any plan."* It also names what the object is — *"a 10×N mask matrix over
//! one allocation; each cell is one `AlphaMask` AND … it needs no new stored
//! state (both masks are recomputed projections)"*. This module is that.
//!
//! # Why the cross is the point, and not a convenience
//!
//! Two axes already exist over the SAME allocation:
//!
//! | axis | who owns it | reading |
//! |---|---|---|
//! | rung `0..=9` — level of PROCESSING | [`AlphaTunnel::lane`] | `.attended_mask()` |
//! | graph / tenant — the G of a quad | [`SpogTenants::tenant`] | `.attended_mask()` |
//!
//! Neither can express the other's question. And because they cannot, a
//! consumer that wants "which domain did this walk touch" is pushed to encode
//! the domain *into the rung byte* — which is exactly what a measured consumer
//! did (`rung = domain ordinal + 1`, then filtering one flat scanpath back
//! apart by it). That collapses two axes into one field and is lossy in a way
//! that was also measured: on one real artifact **16 graphs projected onto 6
//! domains, and 3 graphs (35 % of the rows) resolved to no domain at all** —
//! they landed on the null rung and were filtered out of every projection.
//!
//! The fix is not a rule against the shortcut. It is making the cross
//! *available*, so the rung stays the rung.
//!
//! # The absence is the interesting cell
//!
//! [`AlphaFocus::unlooked`] is `tenant_mask AND NOT any_rung_mask` — "this
//! graph was addressable and no rung ever looked at it". That is the question
//! no log can answer, because absence leaves no line; it is the same argument
//! [`AlphaOverlay::unattended`](crate::alpha::AlphaOverlay::unattended) makes
//! one axis down.
//!
//! # Nothing is stored
//!
//! Every cell is an AND of two masks that are themselves recomputed from the
//! shadows. Constructing a focus costs no rows and no bits beyond the mask it
//! is asked for; dropping it discards nothing that was not derivable again.

use crate::alpha::AlphaMask;
use crate::alpha_tunnel::AlphaTunnel;
use crate::spog_tenants::SpogTenants;

/// Why two readings could not be crossed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusError {
    /// The tunnel and the tenants do not stand over the SAME allocation.
    ///
    /// Refused rather than intersected: a mask is a set of BASE ORDINALS, so
    /// two masks from two allocations are two different coordinate systems and
    /// their AND is arithmetic on unrelated numbers. It would return a
    /// plausible mask and mean nothing.
    DifferentAllocations,
}

impl std::fmt::Display for FocusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DifferentAllocations => write!(
                f,
                "the tunnel and the tenants stand over different allocations; \
                 their ordinals are not the same coordinate"
            ),
        }
    }
}

impl std::error::Error for FocusError {}

/// One cell of the cross — a rung, a graph, and the population where both
/// looked.
#[derive(Debug, Clone)]
pub struct FocusCell {
    /// The processing rung.
    pub rung: u8,
    /// The graph (tenant concept id).
    pub concept: u16,
    /// How many addresses this rung attended within this graph.
    pub count: u32,
}

/// **The 10×N cross** over ONE allocation, borrowed.
pub struct AlphaFocus<'a, 'b> {
    tunnel: &'b AlphaTunnel<'a>,
    tenants: &'b SpogTenants<'a>,
}

impl<'a, 'b> AlphaFocus<'a, 'b> {
    /// Cross a tunnel with an aufstellung of tenants.
    ///
    /// # Errors
    /// [`FocusError::DifferentAllocations`] when the two do not share one
    /// address space. Checked by identity of the base slice, not by its
    /// contents: two allocations over equal-but-distinct slices still index
    /// their own copies, and `ptr::eq` is the only thing that answers "the
    /// same coordinate system" rather than "the same values".
    pub fn cross(
        tunnel: &'b AlphaTunnel<'a>,
        tenants: &'b SpogTenants<'a>,
    ) -> Result<Self, FocusError> {
        let same = tunnel.lane(0).is_some_and(|l| {
            std::ptr::eq(l.allocation().base(), tenants.allocation().base())
        });
        if !same {
            return Err(FocusError::DifferentAllocations);
        }
        Ok(Self { tunnel, tenants })
    }

    /// The population one rung attended INSIDE one graph — the cell.
    ///
    /// [`None`] when the rung is out of range or the graph has no tenant; an
    /// absent axis is not an empty cell.
    #[must_use]
    pub fn cell(&self, rung: u8, concept: u16) -> Option<AlphaMask> {
        let lane = self.tunnel.lane(rung)?.attended_mask();
        let tenant = self.tenants.tenant_mask(concept)?;
        Some(lane.and(&tenant))
    }

    /// Every non-empty cell, in `(rung ascending, declaration order)` — the
    /// matrix as a sparse reading, which is what a caller almost always wants
    /// (most of a 10×N grid is empty by construction).
    #[must_use]
    pub fn matrix(&self) -> Vec<FocusCell> {
        let mut out = Vec::new();
        for rung in 0..crate::rung_schedule::LEVELS {
            let rung = u8::try_from(rung).unwrap_or(u8::MAX);
            for concept in self.tenants.concepts() {
                if let Some(m) = self.cell(rung, concept) {
                    let count = m.count();
                    if count > 0 {
                        out.push(FocusCell {
                            rung,
                            concept,
                            count,
                        });
                    }
                }
            }
        }
        out
    }

    /// Everything any rung of the tunnel attended.
    #[must_use]
    pub fn any_rung_mask(&self) -> AlphaMask {
        let base = self.tenants.allocation().base().len();
        let mut m = AlphaMask::empty(base);
        for rung in 0..crate::rung_schedule::LEVELS {
            if let Some(l) = self.tunnel.lane(u8::try_from(rung).unwrap_or(u8::MAX)) {
                m = m.or(&l.attended_mask());
            }
        }
        m
    }

    /// **The absence.** Addresses of `concept` that were allocated and that
    /// NO rung ever looked at.
    ///
    /// Note what this is not: it is not "the tenant's unclaimed addresses". It
    /// is the tenant's ATTENDED population minus everything the tunnel
    /// reached — so it answers "the graph leg saw these, the rung ladder never
    /// did", which is the disagreement worth reading. For the never-addressed
    /// remainder use [`AlphaOverlay::unattended`](crate::alpha::AlphaOverlay::unattended).
    ///
    /// [`None`] for an undeclared graph.
    #[must_use]
    pub fn unlooked(&self, concept: u16) -> Option<AlphaMask> {
        let tenant = self.tenants.tenant_mask(concept)?;
        Some(tenant.and_not(&self.any_rung_mask()))
    }

    /// How far one rung reached across ALL graphs.
    #[must_use]
    pub fn rung_reach(&self, rung: u8) -> Option<AlphaMask> {
        let lane = self.tunnel.lane(rung)?.attended_mask();
        Some(lane.and(&self.tenants.attended_mask()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaAllocation;
    use crate::canonical_node::{EdgeBlock, NodeGuid, NodeRow};
    use crate::spog_tenants::{block_of, census};

    /// Two graphs sharing a block, one in another block — the shape the
    /// consumer measurement found (several vocabularies under one domain).
    fn base() -> Vec<NodeRow> {
        let mut rows = Vec::new();
        for (g, n) in [(0x9101u32, 3u32), (0x9102, 2), (0x9202, 2)] {
            for i in 0..n {
                rows.push(NodeRow {
                    key: NodeGuid::new(g << 16, 0, 0, 0, 0, (g << 4) + i + 1),
                    edges: EdgeBlock::default(),
                    value: [0u8; 480],
                });
            }
        }
        rows
    }

    /// **The premise, measured on the fixture.** One spine, several graphs —
    /// and the census reads them out of the keys rather than being told.
    /// Anti-vacuity: a single-graph spine would make every cross below
    /// trivially one column wide.
    #[test]
    fn the_census_reads_several_graphs_out_of_one_spine() {
        let b = base();
        assert_eq!(census(&b), vec![0x9101, 0x9102, 0x9202]);
        assert!(census(&b).len() >= 2, "anti-vacuity: more than one graph");
        // Grouping is a shift: two of the three share a block.
        let block = block_of(0x9101);
        let same: Vec<u16> = census(&b).into_iter().filter(|&c| block_of(c) == block).collect();
        assert_eq!(same, vec![0x9101, 0x9102]);
        assert_ne!(block_of(0x9202), block, "and the third does not");
    }

    /// **THE falsifier of this module.** The cross separates what one flat
    /// scanpath cannot: the same rung across two graphs, and two rungs within
    /// one graph, are four distinguishable cells.
    ///
    /// Two-sided: the cells must be non-trivially populated AND a cell whose
    /// two axes never met must be EMPTY. Disable-verified: making `cell`
    /// return the lane mask alone (ignoring the tenant) collapses the graph
    /// axis and fails the emptiness arm.
    #[test]
    fn the_cross_separates_rung_from_graph() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut tunnel = AlphaTunnel::over(&alloc, 1);
        let mut tenants = SpogTenants::over_census(&alloc, 1);

        // rung 3 looks at two rows of 0x9101; rung 5 at one row of 0x9202.
        for (rung, row) in [(3u8, 0usize), (3, 1), (5, 5)] {
            tunnel.lane_mut(rung).unwrap().claim(b[row].key, rung).unwrap();
            assert!(tenants.claim(b[row].key, rung).routed());
        }

        let f = AlphaFocus::cross(&tunnel, &tenants).expect("one allocation");

        assert_eq!(f.cell(3, 0x9101).unwrap().count(), 2, "rung 3 in 0x9101");
        assert_eq!(f.cell(5, 0x9202).unwrap().count(), 1, "rung 5 in 0x9202");
        // The cells whose axes never met — the half a collapsed encoding loses.
        assert_eq!(f.cell(5, 0x9101).unwrap().count(), 0, "rung 5 never entered 0x9101");
        assert_eq!(f.cell(3, 0x9202).unwrap().count(), 0, "rung 3 never entered 0x9202");

        let m = f.matrix();
        assert_eq!(m.len(), 2, "exactly the two populated cells: {m:?}");
        assert_eq!((m[0].rung, m[0].concept, m[0].count), (3, 0x9101, 2));
        assert_eq!((m[1].rung, m[1].concept, m[1].count), (5, 0x9202, 1));
    }

    /// **The absence.** A graph the tenant leg attended and the rung ladder
    /// never reached is readable — and a graph both reached is NOT reported as
    /// absent (the can-stay-silent half).
    #[test]
    fn a_graph_no_rung_ever_looked_at_is_readable() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut tunnel = AlphaTunnel::over(&alloc, 1);
        let mut tenants = SpogTenants::over_census(&alloc, 1);

        // The tenant leg sees 0x9102 (row 3); no rung ever does.
        assert!(tenants.claim(b[3].key, 0).routed());
        // Both legs see 0x9101 (row 0).
        assert!(tenants.claim(b[0].key, 2).routed());
        tunnel.lane_mut(2).unwrap().claim(b[0].key, 2).unwrap();

        let f = AlphaFocus::cross(&tunnel, &tenants).expect("one allocation");
        assert_eq!(
            f.unlooked(0x9102).unwrap().count(),
            1,
            "0x9102 was addressable and no rung looked"
        );
        assert_eq!(
            f.unlooked(0x9101).unwrap().count(),
            0,
            "0x9101 was reached — silence here, or the reading fires on everything"
        );
        assert!(f.unlooked(0x0999).is_none(), "an undeclared graph is absent, not empty");
    }

    /// The rung stays the rung: a claim's stamp carries the PROCESSING rung
    /// the caller named, in whichever graph it lands. This is the property the
    /// domain-in-rung shortcut destroys.
    #[test]
    fn the_rung_byte_is_the_processing_rung_in_every_graph() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut tenants = SpogTenants::over_census(&alloc, 1);
        const RUNG: u8 = 7;
        for row in [0usize, 3, 5] {
            assert!(tenants.claim(b[row].key, RUNG).routed());
        }
        let touched: Vec<u16> = tenants
            .concepts()
            .into_iter()
            .filter(|&c| tenants.tenant(c).is_some_and(|s| s.claimed_len() > 0))
            .collect();
        assert_eq!(touched.len(), 3, "anti-vacuity: three graphs touched");
        for (_, st) in tenants.merge() {
            assert_eq!(st.rung, RUNG, "the graph never leaks into the rung byte");
        }
    }

    /// Two allocations are two coordinate systems — refused, not intersected.
    #[test]
    fn crossing_two_allocations_is_refused() {
        let b = base();
        let other = base();
        let alloc_a = AlphaAllocation::over(&b);
        let alloc_b = AlphaAllocation::over(&other);
        let tunnel = AlphaTunnel::over(&alloc_a, 1);
        let tenants = SpogTenants::over_census(&alloc_b, 1);
        match AlphaFocus::cross(&tunnel, &tenants) {
            Err(FocusError::DifferentAllocations) => {}
            Ok(_) => panic!("equal contents are not the same coordinate"),
        }
    }
}
