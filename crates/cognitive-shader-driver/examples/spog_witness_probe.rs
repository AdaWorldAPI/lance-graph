//! D-SPOG-W-0: the Witness slot as a SPOG sub-context under classid-G.
//!
//! Claim under test: a SPOG coordinate is a view assembled from coordinates
//! that are already resident, not a storage layout:
//!
//! ```text
//! G    = graph_of(classid)            the canonical source (spog_tenants:
//!                                     "g via classid"), read from the key
//! sub  = w_slot() under a reading     a sub-context INSIDE G (corpus, named
//!        the class declares            frame, schema); 0 = no anchor
//! S/P/O = the three resident bytes    passed through blind, no reading
//! ```
//!
//! Witness refines G; it never replaces it. A class that has not declared the
//! sub-context reading, or declared another reading of the slot, refuses.
//! Bits 53..58 are read only under an asserted v2/V3 provenance (on a v1 row
//! they are old `temporal` bits), the `band_reading` rule.
//!
//! # The routing check
//!
//! The shipped production consumer of the slot is
//! `MailboxSoA::apply_edges`, which drops every edge whose `w_slot()` differs
//! from the mailbox's. The probe compares that real routing with the SPOG
//! reading:
//!
//! - **Within one G they agree** on all 64 × 64 slot pairs, including slot 0
//!   (unanchored mailbox accepts unanchored edges).
//! - **Across graphs they diverge.** `apply_edges` reads no classid, so a
//!   mailbox accepts an edge from another graph that happens to carry the same
//!   slot value, while the SPOG reading places it in a different context. The
//!   agreement therefore holds only for a mailbox scoped to one graph; nothing
//!   in `MailboxSoA` enforces that today. Pinned as a test, not fixed here.
//!
//! # Not decided here
//!
//! - The declaration is probe-local and keyed by classid; whether it becomes a
//!   `band_reading`-style contract declaration is open.
//! - What a sub-context ordinal means (corpus, frame, schema) is the
//!   consumer's binding, not this probe's.
//! - Making `MailboxSoA` graph-aware is a production change and out of scope.
//!
//! Run: `cargo run -p cognitive-shader-driver --example spog_witness_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example spog_witness_probe`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use causal_edge::edge::CausalEdge64;
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;
use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::canonical_node::{NodeGuid, TailVariant};
use lance_graph_contract::spog_tenants::graph_of;

// ── allocation counter (the recipe_quartet_probe pattern) ─────────────────

/// Counts allocations on the current thread; test threads run in parallel.
struct CountingAlloc;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: forwards every call to `System` unchanged; it only adds a counter.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Run `f` and return its result with the allocations it made on this thread.
fn counting<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATIONS.with(Cell::get);
    let out = f();
    (out, ALLOCATIONS.with(Cell::get) - before)
}

// ── declared readings of the Witness slot ──────────────────────────────────

/// How a class reads the six Witness bits. Declared per class; the bits
/// themselves cannot say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WitnessReading {
    /// A sub-context inside the class's graph (corpus, frame, schema).
    GraphSubContext,
    /// Some other reading of the same bits (e.g. an evidence family). It is
    /// legal for its own consumers and never readable as a SPOG sub-context.
    Other,
}

/// Two graphs (concepts `0x9101`, `0x9102`), two classes in the first, one in
/// the second, and one class that reads the slot otherwise. Probe placeholders,
/// not OGAR mints.
const CLASS_A: u32 = 0x9101_0001;
const CLASS_A2: u32 = 0x9101_0002;
const CLASS_B: u32 = 0x9102_0001;
const CLASS_OTHER: u32 = 0x9101_0003;
const CLASS_UNDECLARED: u32 = 0x9101_0004;

const DECLARATIONS: [(u32, WitnessReading); 4] = [
    (CLASS_A, WitnessReading::GraphSubContext),
    (CLASS_A2, WitnessReading::GraphSubContext),
    (CLASS_B, WitnessReading::GraphSubContext),
    (CLASS_OTHER, WitnessReading::Other),
];

fn reading_of(classid: u32) -> Option<WitnessReading> {
    DECLARATIONS
        .iter()
        .find(|(c, _)| *c == classid)
        .map(|(_, r)| *r)
}

// ── the SPOG view ───────────────────────────────────────────────────────────

/// A SPOG coordinate read from resident state. Transient; never stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Spog {
    s: u8,
    p: u8,
    o: u8,
    /// The graph, from the classid.
    g: u16,
    /// The Witness sub-context inside `g`; `None` = no anchor (slot 0).
    sub: Option<u8>,
}

impl Spog {
    /// The context half: which graph, and which sub-context inside it.
    fn context(self) -> (u16, Option<u8>) {
        (self.g, self.sub)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// Bits 53..58 were not asserted to be v2/V3-stamped.
    Provenance(EdgeProvenance),
    /// The class declared no reading of the slot.
    Undeclared(u32),
    /// The class reads the slot some other way.
    OtherReading(u32),
}

/// A key carrying `classid`, minted on the sanctioned V3 path. Only the
/// classid matters to G; the tail is zero.
fn key(classid: u32) -> NodeGuid {
    NodeGuid::mint_for(TailVariant::V3, classid, 0, 0, 0, 0, 0, 0)
}

/// Sub-context reading of a slot value. Slot 0 is "no corpus anchor", as the
/// shipped `w_slot` accessor documents.
fn sub_context(w: u8) -> Option<u8> {
    (w != 0).then_some(w)
}

/// Assemble the SPOG coordinate from `(classid, edge)`.
fn spog(classid: u32, edge: CausalEdge64, provenance: EdgeProvenance) -> Result<Spog, Refusal> {
    if !matches!(
        provenance,
        EdgeProvenance::V2Stamped | EdgeProvenance::V3Register
    ) {
        return Err(Refusal::Provenance(provenance));
    }
    match reading_of(classid) {
        None => return Err(Refusal::Undeclared(classid)),
        Some(WitnessReading::Other) => return Err(Refusal::OtherReading(classid)),
        Some(WitnessReading::GraphSubContext) => {}
    }
    Ok(Spog {
        s: edge.s_idx(),
        p: edge.p_idx(),
        o: edge.o_idx(),
        g: graph_of(key(classid)),
        sub: sub_context(edge.w_slot()),
    })
}

/// An edge this probe stamped itself under the v2 layout.
fn stamped(classid: u32, edge: CausalEdge64) -> Result<Spog, Refusal> {
    spog(classid, edge, EdgeProvenance::V2Stamped)
}

/// Does the real mailbox accept this edge? One delivery to row 0.
fn mailbox_accepts(mailbox_slot: u8, edge: CausalEdge64) -> bool {
    let mut mb: MailboxSoA<4> = MailboxSoA::new(1, mailbox_slot, 0.5);
    mb.apply_edges(&[(0, edge)]) == 1
}

fn edge(s: u8, p: u8, o: u8, w: u8) -> CausalEdge64 {
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
    .with_w_slot(w)
}

fn main() {
    let e = edge(0x11, 0x22, 0x33, 23);
    for class in [CLASS_A, CLASS_A2, CLASS_B, CLASS_OTHER, CLASS_UNDECLARED] {
        println!(
            "classid {class:#010x}, w_slot 23 -> {:?}",
            stamped(class, e)
        );
    }
    let (_, allocs) = counting(|| stamped(CLASS_A, e));
    println!("SPOG read allocations: {allocs}");
    let a = stamped(CLASS_A, e).unwrap();
    let b = stamped(CLASS_B, e).unwrap();
    println!(
        "mailbox(slot 23) accepts the edge: {}; SPOG contexts {:?} vs {:?} (graphs differ, routing does not see it)",
        mailbox_accepts(23, e),
        a.context(),
        b.context()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// G comes from the classid and only from it: the slot never moves G, the
    /// classid never moves the sub-context, and `graph_of` is the concept half.
    #[test]
    fn g_comes_from_the_classid_never_from_witness() {
        for w in 0..64u8 {
            let a = stamped(CLASS_A, edge(1, 2, 3, w)).unwrap();
            let b = stamped(CLASS_B, edge(1, 2, 3, w)).unwrap();
            assert_eq!(a.g, 0x9101);
            assert_eq!(b.g, 0x9102);
            assert_eq!(a.sub, b.sub, "the classid does not move the sub-context");
            assert_ne!(a.context(), b.context(), "same slot, different graphs");
        }
        // Two classes of one graph share G.
        let a = stamped(CLASS_A, edge(1, 2, 3, 9)).unwrap();
        let a2 = stamped(CLASS_A2, edge(1, 2, 3, 9)).unwrap();
        assert_eq!(a.context(), a2.context());
    }

    /// S/P/O pass through as blind bytes; the context ignores them.
    #[test]
    fn spo_bytes_pass_through_blind() {
        for v in 0..=255u8 {
            let c = stamped(CLASS_A, edge(v, 255 - v, v ^ 0x5A, 7)).unwrap();
            assert_eq!((c.s, c.p, c.o), (v, 255 - v, v ^ 0x5A));
            assert_eq!(c.context(), (0x9101, Some(7)));
        }
    }

    /// Slot 0 reads as "no anchor", every other slot as its own sub-context.
    #[test]
    fn slot_zero_is_no_anchor() {
        assert_eq!(stamped(CLASS_A, edge(1, 2, 3, 0)).unwrap().sub, None);
        for w in 1..64u8 {
            assert_eq!(stamped(CLASS_A, edge(1, 2, 3, w)).unwrap().sub, Some(w));
        }
    }

    /// Classes without the declared reading refuse; so do v1 and unknown
    /// provenance. The bits are the same in every case.
    #[test]
    fn undeclared_other_reading_and_untrusted_provenance_refuse() {
        let e = edge(1, 2, 3, 23);
        assert_eq!(
            stamped(CLASS_UNDECLARED, e),
            Err(Refusal::Undeclared(CLASS_UNDECLARED))
        );
        assert_eq!(
            stamped(CLASS_OTHER, e),
            Err(Refusal::OtherReading(CLASS_OTHER))
        );
        for prov in [EdgeProvenance::V1Legacy, EdgeProvenance::Unknown] {
            assert_eq!(spog(CLASS_A, e, prov), Err(Refusal::Provenance(prov)));
        }
        assert!(spog(CLASS_A, e, EdgeProvenance::V3Register).is_ok());
    }

    /// Within one graph, the real `MailboxSoA` routing and the SPOG reading
    /// agree on every one of the 64 × 64 slot pairs: the mailbox accepts an
    /// edge exactly when their sub-contexts are equal.
    #[test]
    fn within_one_graph_routing_agrees_with_the_spog_reading() {
        let (mut accepted, mut dropped) = (0, 0);
        for mailbox_slot in 0..64u8 {
            let mailbox_ctx = (0x9101u16, sub_context(mailbox_slot));
            for w in 0..64u8 {
                let e = edge(1, 2, 3, w);
                let same_ctx = stamped(CLASS_A, e).unwrap().context() == mailbox_ctx;
                let accepts = mailbox_accepts(mailbox_slot, e);
                assert_eq!(accepts, same_ctx, "mailbox {mailbox_slot}, edge slot {w}");
                if accepts {
                    accepted += 1;
                } else {
                    dropped += 1;
                }
            }
        }
        assert_eq!((accepted, dropped), (64, 64 * 63), "both outcomes occur");
    }

    /// Across graphs they diverge: the mailbox reads no classid, so it accepts
    /// an edge from another graph carrying the same slot, which the SPOG
    /// reading places in a different context. Agreement needs a mailbox scoped
    /// to one graph.
    #[test]
    fn across_graphs_routing_diverges_from_the_spog_reading() {
        let mailbox_ctx = (0x9101u16, sub_context(23));
        let foreign = edge(1, 2, 3, 23);
        assert!(
            mailbox_accepts(23, foreign),
            "routing accepts the foreign edge"
        );
        assert_ne!(stamped(CLASS_B, foreign).unwrap().context(), mailbox_ctx);
        // The same edge read under the mailbox's own graph does agree.
        assert_eq!(stamped(CLASS_A, foreign).unwrap().context(), mailbox_ctx);
    }

    /// The read allocates nothing.
    #[test]
    fn the_spog_read_allocates_nothing() {
        let edges: [CausalEdge64; 64] = core::array::from_fn(|w| edge(w as u8, 2, 3, w as u8));
        let (acc, allocs) = counting(|| {
            edges.iter().fold(0u32, |acc, e| {
                let c = stamped(CLASS_A, *e).unwrap();
                acc.wrapping_add(u32::from(c.g) ^ u32::from(c.sub.unwrap_or(0)) ^ u32::from(c.s))
            })
        });
        assert_eq!(allocs, 0);
        assert_ne!(acc, 0);
    }
}
