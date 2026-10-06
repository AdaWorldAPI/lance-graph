//! D-ALPHA-G-0: WorldG is attention provenance, carried by the Alpha route,
//! not by `CausalEdge64`.
//!
//! Claim under test:
//!
//! ```text
//! epistemic event = (Alpha-attended address, CausalEdge64)
//! WorldG          = graph_of(address)        read from the attended key
//! order of worlds = SpogTenants route         which tenant took each fresh claim
//! CausalEdge64    = opaque epistemic delta    carries no world and needs none
//! ```
//!
//! # The seam used
//!
//! `lance_graph_contract::spog_tenants::SpogTenants` is the Alpha split tunnel
//! keyed by graph: one `AlphaOverlay` shadow per world over ONE
//! `AlphaAllocation`. `claim(addr)` routes to the tenant `graph_of(addr)`,
//! refuses an address whose world has no tenant (`TenantClaim::NoTenant`),
//! and records the world of every fresh claim in its `route`.
//! `merge_in_claim_order` replays the saccade in time. Every attended row keeps
//! the base row's key byte-for-byte, so the world is read from the attention
//! record itself.
//!
//! # What the probe shows
//!
//! - The same CE64 bits attended in two worlds are two different events.
//! - Replaying the same claims on a fresh tunnel recovers the world sequence,
//!   with all-zero edges, so no world can have come from the edge.
//! - Changing every CE64 field leaves the recovered worlds unchanged, while
//!   the events themselves change.
//! - Partitioned by recovered world, each `MailboxSoA` receives only its own
//!   world's deliveries and accepts all of them; `apply_edges` is unchanged and
//!   reads no G.
//! - An address of an undeclared world is refused at the tunnel and never
//!   becomes a delivery.
//!
//! # What it does NOT prove
//!
//! - That production delivers edges this way. `MailboxSoA::apply_edges` has no
//!   production caller today and nothing connects `SpogTenants` to a mailbox,
//!   so no production path mixes worlds at a mailbox. That is absence of a
//!   path, not a guard on one; the multi-mailbox routing that would become one
//!   (W5) is unbuilt.
//! - That `graph_of` (classid high half) is the right WorldG granularity for
//!   every consumer; it is the canonical rule today (`spog_tenants`).
//! - Anything about a row-local SPOG G (LocalG); that is a separate
//!   coordinate inside the selected world and is not modelled here.
//!
//! # Correction to D-SPOG-W-0
//!
//! `spog_witness_probe`'s cross-graph case built a delivery from a foreign
//! world directly, bypassing this tunnel. It is a constructed case, not a
//! reachable one; `ISS-MAILBOX-ROUTES-WITNESS-WITHOUT-GRAPH` is reframed
//! accordingly, and `MailboxSoA` stays graph-blind.
//!
//! Run: `cargo run -p cognitive-shader-driver --example alpha_world_provenance_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example alpha_world_provenance_probe`

use causal_edge::edge::CausalEdge64;
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;
use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::alpha::{AlphaAddr, AlphaAllocation};
use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow, TailVariant};
use lance_graph_contract::spog_tenants::{graph_of, SpogTenants, TenantClaim};

/// Two worlds. Probe placeholders, not OGAR mints.
const W_OBS: u16 = 0x9101;
const W_ERP: u16 = 0x9102;
/// A world the spine carries but the tunnel below does not declare.
const W_IAM: u16 = 0x9103;

/// Rows per world in the probe spine.
const PER_WORLD: u16 = 4;

/// The Witness slot every probe mailbox and edge uses, so `apply_edges`'s own
/// slot filter never decides anything here.
const SLOT: u8 = 7;

/// One attended row of world `w`, index `i`. Only the classid's high half is
/// the world; `heel` makes the keys distinct.
fn addr(w: u16, i: u16) -> AlphaAddr {
    NodeGuid::mint_for(TailVariant::V3, (u32::from(w) << 16) | 1, i, 0, 0, 0, 0, 1)
}

/// The base spine the allocation borrows: three worlds, interleaved, so the
/// world is never a slice range.
fn spine() -> Vec<NodeRow> {
    (0..PER_WORLD)
        .flat_map(|i| [addr(W_OBS, i), addr(W_ERP, i), addr(W_IAM, i)])
        .map(|key| NodeRow {
            key,
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        })
        .collect()
}

fn edge(s: u8, p: u8, o: u8) -> CausalEdge64 {
    CausalEdge64::pack_v2(
        s,
        p,
        o,
        200,
        180,
        CausalMask::SO,
        0b010,
        PlasticityState::ALL_HOT,
    )
    .with_inference_mantissa(1)
    .with_w_slot(SLOT)
}

/// One observation: attention landed at `addr` and produced `edge`.
#[derive(Debug, Clone, Copy)]
struct Observation {
    addr: AlphaAddr,
    edge: CausalEdge64,
}

/// An epistemic event: the world the attention route supplies, plus the edge.
/// Transient; never stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Event {
    world: u16,
    bits: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// The address's world has no tenant in this tunnel.
    NoTenant(u16),
    /// The substrate refused the address (not in the spine).
    Substrate,
}

/// Attend every observation through the split tunnel. The first refusal stops
/// the walk; claims made before it stand.
fn attend(tunnel: &mut SpogTenants<'_>, obs: &[Observation]) -> Result<(), Refusal> {
    for o in obs {
        match tunnel.claim(o.addr, 1) {
            TenantClaim::Routed(..) => {}
            TenantClaim::NoTenant(g) => return Err(Refusal::NoTenant(g)),
            TenantClaim::Substrate(_) => return Err(Refusal::Substrate),
        }
    }
    Ok(())
}

/// The events of a saccade: the world from the attended key, the delta from
/// the edge.
fn events(obs: &[Observation]) -> Vec<Event> {
    obs.iter()
        .map(|o| Event {
            world: graph_of(o.addr),
            bits: o.edge.0,
        })
        .collect()
}

/// The world sequence replayed from the tunnel alone: its route plus each
/// tenant's scanpath. No edge is consulted.
fn replay_worlds(tunnel: &SpogTenants<'_>) -> Vec<u16> {
    tunnel
        .merge_in_claim_order()
        .into_iter()
        .map(|(a, _)| graph_of(a))
        .collect()
}

/// A saccade that switches worlds several times, one fresh claim per address.
fn saccade(edge_for: impl Fn(usize) -> CausalEdge64) -> Vec<Observation> {
    let path = [
        addr(W_OBS, 0),
        addr(W_ERP, 0),
        addr(W_ERP, 1),
        addr(W_OBS, 1),
        addr(W_ERP, 2),
        addr(W_OBS, 2),
        addr(W_OBS, 3),
    ];
    path.iter()
        .enumerate()
        .map(|(i, &a)| Observation {
            addr: a,
            edge: edge_for(i),
        })
        .collect()
}

fn main() {
    let rows = spine();
    let alloc = AlphaAllocation::over(&rows);
    let mut tunnel = SpogTenants::over(&alloc, 1, &[W_OBS, W_ERP]);
    let obs = saccade(|_| edge(1, 2, 3));
    attend(&mut tunnel, &obs).expect("both worlds declared");
    println!("D-ALPHA-G-0: WorldG from the Alpha route, not from CE64");
    println!("  events          {:x?}", events(&obs));
    println!("  replayed worlds {:x?}", replay_worlds(&tunnel));
    let refused = attend(
        &mut tunnel,
        &[Observation {
            addr: addr(W_IAM, 0),
            edge: edge(1, 2, 3),
        }],
    );
    println!("  undeclared world -> {refused:?}");
    // Inside the selected world the mailbox is graph-blind: deliver the
    // observations world W_OBS produced to that world's mailbox.
    let mut mb: MailboxSoA<16> = MailboxSoA::new(1, SLOT, 0.5);
    let own: Vec<(u16, CausalEdge64)> = obs
        .iter()
        .filter(|o| graph_of(o.addr) == W_OBS)
        .map(|o| (alloc.ordinal(o.addr).unwrap_or(0) as u16 % 16, o.edge))
        .collect();
    println!(
        "  W_OBS mailbox accepted {} of {}",
        mb.apply_edges(&own),
        own.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same CE64 bits, two Alpha-selected worlds: two distinct events.
    #[test]
    fn same_ce64_bits_in_two_worlds_are_two_events() {
        let e = edge(9, 9, 9);
        let ev = events(&[
            Observation {
                addr: addr(W_OBS, 0),
                edge: e,
            },
            Observation {
                addr: addr(W_ERP, 0),
                edge: e,
            },
        ]);
        assert_eq!(ev[0].bits, ev[1].bits, "same delta");
        assert_ne!(ev[0], ev[1], "different events");
        assert_eq!((ev[0].world, ev[1].world), (W_OBS, W_ERP));
    }

    /// The falsifier: replay of the same route over the same spine recovers
    /// which world produced each observation, with edges that carry nothing.
    #[test]
    fn replay_recovers_the_world_sequence_without_the_edge() {
        let rows = spine();
        let alloc = AlphaAllocation::over(&rows);
        let obs = saccade(|_| CausalEdge64::ZERO);
        let generated: Vec<u16> = obs.iter().map(|o| graph_of(o.addr)).collect();

        let mut first = SpogTenants::over(&alloc, 1, &[W_OBS, W_ERP]);
        attend(&mut first, &obs).unwrap();
        let mut again = SpogTenants::over(&alloc, 1, &[W_OBS, W_ERP]);
        attend(&mut again, &obs).unwrap();

        assert_eq!(
            replay_worlds(&first),
            generated,
            "the route recovers the worlds"
        );
        assert_eq!(
            replay_worlds(&again),
            generated,
            "and does so again on replay"
        );

        // Anti-vacuity: the saccade switches worlds more than once, and the
        // grouped merge (no route) gives a different order, so the route is
        // what carries the sequence.
        let switches = generated.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(switches >= 3, "{switches} world switches");
        let grouped: Vec<u16> = first.merge().iter().map(|(a, _)| graph_of(*a)).collect();
        assert_ne!(grouped, generated, "grouping by world loses the order");
    }

    /// Stay-silent: changing CE64 fields does not move the recovered world,
    /// while the events do change (so the edges are really in them).
    #[test]
    fn ce64_changes_do_not_move_the_world() {
        let variants: [fn(usize) -> CausalEdge64; 5] = [
            |_| CausalEdge64::ZERO,
            |i| edge(i as u8, 2, 3),
            |i| edge(1, 2, 3).with_w_slot(i as u8 + 1),
            |i| edge(1, 2, 3).with_inference_mantissa(-(i as i8)),
            |_| CausalEdge64(u64::MAX),
        ];
        let worlds: Vec<Vec<u16>> = variants
            .iter()
            .map(|f| events(&saccade(f)).iter().map(|e| e.world).collect())
            .collect();
        let bits: Vec<Vec<u64>> = variants
            .iter()
            .map(|f| events(&saccade(f)).iter().map(|e| e.bits).collect())
            .collect();
        assert!(worlds.iter().all(|w| *w == worlds[0]), "worlds never move");
        for i in 0..bits.len() {
            for j in i + 1..bits.len() {
                assert_ne!(bits[i], bits[j], "variants {i} and {j} differ in the edge");
            }
        }
    }

    /// Inside one already-selected world the mailbox needs no G: partitioned by
    /// the recovered world, each mailbox receives only its own deliveries and
    /// accepts all of them, through the unchanged `apply_edges`.
    #[test]
    fn inside_one_world_the_mailbox_needs_no_g() {
        let rows = spine();
        let alloc = AlphaAllocation::over(&rows);
        let mut tunnel = SpogTenants::over(&alloc, 1, &[W_OBS, W_ERP]);
        let obs = saccade(|i| edge(i as u8, 2, 3));
        attend(&mut tunnel, &obs).unwrap();

        let worlds = [W_OBS, W_ERP];
        let mut boxes: [MailboxSoA<16>; 2] =
            [MailboxSoA::new(1, SLOT, 0.5), MailboxSoA::new(2, SLOT, 0.5)];
        let mut received = [0usize; 2];
        let mut accepted = [0usize; 2];
        for o in &obs {
            let g = graph_of(o.addr);
            let k = worlds.iter().position(|w| *w == g).expect("declared world");
            let row = alloc.ordinal(o.addr).unwrap() as u16 % 16;
            received[k] += 1;
            accepted[k] += boxes[k].apply_edges(&[(row, o.edge)]);
        }
        let expected = |w: u16| obs.iter().filter(|o| graph_of(o.addr) == w).count();
        assert_eq!(received, [expected(W_OBS), expected(W_ERP)]);
        assert_eq!(accepted, received, "every in-world delivery is accepted");
        assert!(received.iter().all(|&n| n > 0), "both worlds deliver");
        // Read back from the mailboxes themselves: each holds exactly its own
        // world's deliveries, so the partition happened before `apply_edges`.
        let held = |b: &MailboxSoA<16>| {
            (0..16)
                .map(|r| usize::from(b.plasticity_at(r)))
                .sum::<usize>()
        };
        assert_eq!([held(&boxes[0]), held(&boxes[1])], received);
    }

    /// An address from a world the tunnel does not serve is refused upstream
    /// and never reaches a mailbox; an address outside the spine likewise.
    #[test]
    fn a_foreign_world_is_refused_at_the_tunnel() {
        let rows = spine();
        let alloc = AlphaAllocation::over(&rows);
        let mut tunnel = SpogTenants::over(&alloc, 1, &[W_OBS]);
        let foreign = Observation {
            addr: addr(W_IAM, 0),
            edge: edge(1, 2, 3),
        };
        assert_eq!(
            attend(&mut tunnel, &[foreign]),
            Err(Refusal::NoTenant(W_IAM))
        );
        let outside = Observation {
            addr: addr(W_OBS, 99),
            edge: edge(1, 2, 3),
        };
        assert_eq!(attend(&mut tunnel, &[outside]), Err(Refusal::Substrate));
        assert_eq!(tunnel.claimed_len(), 0, "nothing was attended");

        // Silent twin: an address of the served world is accepted.
        let own = Observation {
            addr: addr(W_OBS, 0),
            edge: edge(1, 2, 3),
        };
        assert_eq!(attend(&mut tunnel, &[own]), Ok(()));
        assert_eq!(tunnel.claimed_len(), 1);
    }
}
