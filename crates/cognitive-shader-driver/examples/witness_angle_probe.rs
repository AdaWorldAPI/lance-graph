//! D-WA-0: Witness × Angle as a declared, sparse local source coordinate.
//!
//! Claim under test: inside a world the Alpha route has already selected,
//! two resident CE64 fields can address a local source reading under a law
//! the class declares:
//!
//! ```text
//! Witness = w_slot()      bits 53..58, 6 bits
//! Angle   = direction()   bits 43..45, 3 bits
//! (class, generation, Witness, Angle)  --sparse law-->  SourceCoord
//! ```
//!
//! Nothing global is encoded in the bits. The same raw pair means different
//! things under different classes and under different generations of one
//! class, and means nothing for a class that reads those bits another way.
//! The world comes from the attended key (`graph_of`), never from the edge,
//! and no source record is stored beside the edge: the coordinate is computed
//! on read from a tiny probe codebook.
//!
//! # Readings of the same bits that refuse here
//!
//! - **Pathology triad**: the shipped meaning of bits 43..45
//!   (`s/p/o_pathological`).
//! - **Cohort witness**: the shipped meaning of bits 53..58 in
//!   `witness_table::WitnessTable` (slot → `(mailbox_ref, spo_fact_ref)`).
//!
//! A class that declares either reading refuses the source-coordinate
//! reading, so the two shipped meanings are never silently reinterpreted.
//!
//! # What it does NOT prove
//!
//! - What a source coordinate means: `SourceCoord` is two opaque ordinals of
//!   a six-entry test codebook, not an ontology.
//! - That the declaration belongs in the contract: it is probe-local, like
//!   D-SPOG-W-0's.
//! - Anything about LocalG (a row-local SPOG G): not modelled.
//!
//! Run: `cargo run -p cognitive-shader-driver --example witness_angle_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example witness_angle_probe`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use causal_edge::edge::CausalEdge64;
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;
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

// ── declarations ──────────────────────────────────────────────────────────

/// How a class reads Witness and bits 43..45 together. Declared per class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotReading {
    /// Witness × Angle → a local source coordinate under the class's law.
    SourceCoordinate,
    /// Bits 43..45 are the S/P/O pathology triad (the shipped meaning).
    PathologyTriad,
    /// Bits 53..58 index a cohort `WitnessTable` (the shipped meaning).
    CohortWitness,
}

/// Probe placeholders, not OGAR mints: two worlds (`0x9101`, `0x9102`).
const CLASS_OBS: u32 = 0x9101_0001;
const CLASS_ERP: u32 = 0x9102_0001;
/// A second source-coordinate class in the same world as `CLASS_OBS`.
const CLASS_OBS2: u32 = 0x9101_0005;
const CLASS_PATHOLOGY: u32 = 0x9101_0002;
const CLASS_COHORT: u32 = 0x9101_0003;
const CLASS_UNDECLARED: u32 = 0x9101_0004;

/// `(class, reading, declared generations)`.
const DECLARATIONS: [(u32, SlotReading, &[u8]); 5] = [
    (CLASS_OBS, SlotReading::SourceCoordinate, &[1, 2]),
    (CLASS_OBS2, SlotReading::SourceCoordinate, &[1]),
    (CLASS_ERP, SlotReading::SourceCoordinate, &[1]),
    (CLASS_PATHOLOGY, SlotReading::PathologyTriad, &[1]),
    (CLASS_COHORT, SlotReading::CohortWitness, &[1]),
];

/// A local source/read position: two opaque ordinals of the probe codebook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceCoord {
    table: u8,
    column: u8,
}

const fn sc(table: u8, column: u8) -> SourceCoord {
    SourceCoord { table, column }
}

/// The sparse law: `(class, generation, witness, angle) → coordinate`. Every
/// pair not listed is unmapped. A test codebook, not an ontology.
const LAW: [(u32, u8, u8, u8, SourceCoord); 6] = [
    (CLASS_OBS, 1, 23, 5, sc(1, 7)),
    (CLASS_OBS, 1, 23, 2, sc(1, 3)),
    (CLASS_OBS, 1, 4, 5, sc(2, 7)),
    // Generation 2 re-reads (23, 5) and drops the other two pairs.
    (CLASS_OBS, 2, 23, 5, sc(1, 8)),
    // Same world, another class, same raw pair: a different source.
    (CLASS_OBS2, 1, 23, 5, sc(5, 1)),
    // Another world, same raw pair, a different source.
    (CLASS_ERP, 1, 23, 5, sc(9, 2)),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// The edge's bits were not asserted to be v2/V3-stamped.
    Provenance(EdgeProvenance),
    /// The class declared no reading of these bits.
    Undeclared(u32),
    /// The class reads these bits another way.
    OtherReading(SlotReading),
    /// The class never declared this generation.
    UndeclaredGeneration(u8),
    /// Witness 0: no anchor, so no source.
    NoAnchor,
    /// The pair has no entry in this class generation's law.
    Unmapped { witness: u8, angle: u8 },
}

/// The world and the local source of one epistemic delta. Transient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LocalSource {
    /// From the attended key, never from the edge.
    world: u16,
    source: SourceCoord,
}

fn key(classid: u32) -> NodeGuid {
    NodeGuid::mint_for(TailVariant::V3, classid, 0, 0, 0, 0, 0, 0)
}

/// Read the local source of `edge`, attended under `classid`, at the declared
/// `generation`.
fn local_source(
    classid: u32,
    generation: u8,
    edge: CausalEdge64,
    provenance: EdgeProvenance,
) -> Result<LocalSource, Refusal> {
    if !matches!(
        provenance,
        EdgeProvenance::V2Stamped | EdgeProvenance::V3Register
    ) {
        return Err(Refusal::Provenance(provenance));
    }
    let Some(&(_, reading, gens)) = DECLARATIONS.iter().find(|(c, _, _)| *c == classid) else {
        return Err(Refusal::Undeclared(classid));
    };
    if reading != SlotReading::SourceCoordinate {
        return Err(Refusal::OtherReading(reading));
    }
    if !gens.contains(&generation) {
        return Err(Refusal::UndeclaredGeneration(generation));
    }
    let (witness, angle) = (edge.w_slot(), edge.direction());
    if witness == 0 {
        return Err(Refusal::NoAnchor);
    }
    LAW.iter()
        .find(|(c, g, w, a, _)| *c == classid && *g == generation && *w == witness && *a == angle)
        .map(|&(_, _, _, _, source)| LocalSource {
            world: graph_of(key(classid)),
            source,
        })
        .ok_or(Refusal::Unmapped { witness, angle })
}

fn stamped(classid: u32, generation: u8, edge: CausalEdge64) -> Result<LocalSource, Refusal> {
    local_source(classid, generation, edge, EdgeProvenance::V2Stamped)
}

/// An edge with the given Witness and Angle; the rest is fixed filler.
fn edge(witness: u8, angle: u8) -> CausalEdge64 {
    CausalEdge64::pack_v2(
        1,
        2,
        3,
        200,
        180,
        CausalMask::SO,
        angle,
        PlasticityState::ALL_HOT,
    )
    .with_inference_mantissa(1)
    .with_w_slot(witness)
}

fn main() {
    let e = edge(23, 5);
    println!("D-WA-0: Witness x Angle as a declared local source coordinate");
    println!(
        "  bits                      w={} a={}",
        e.w_slot(),
        e.direction()
    );
    for (name, c, g) in [
        ("observations gen 1", CLASS_OBS, 1),
        ("observations gen 2", CLASS_OBS, 2),
        ("erp gen 1", CLASS_ERP, 1),
        ("pathology class", CLASS_PATHOLOGY, 1),
        ("undeclared class", CLASS_UNDECLARED, 1),
    ] {
        println!("  {name:<25} {:?}", stamped(c, g, e));
    }
    let (r, allocs) = counting(|| stamped(CLASS_OBS, 1, e));
    println!(
        "  read allocations          {allocs} ({:?})",
        r.map(|s| s.source)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same bits, same class, same generation: the same coordinate, every time.
    #[test]
    fn same_bits_and_generation_give_the_same_coordinate() {
        let a = stamped(CLASS_OBS, 1, edge(23, 5)).unwrap();
        let b = stamped(CLASS_OBS, 1, edge(23, 5)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.source, sc(1, 7));
        assert_eq!(a.world, 0x9101);
    }

    /// The same raw pair in two classes is two sources, whether the classes
    /// share a world or not. The world comes from the class's key; the edge is
    /// identical.
    #[test]
    fn same_bits_in_another_class_are_another_source() {
        let e = edge(23, 5);
        let obs = stamped(CLASS_OBS, 1, e).unwrap();
        let obs2 = stamped(CLASS_OBS2, 1, e).unwrap();
        let erp = stamped(CLASS_ERP, 1, e).unwrap();
        // Same world, different class: class identity alone separates them.
        assert_eq!(obs.world, obs2.world);
        assert_ne!(obs.source, obs2.source);
        // Different world as well.
        assert_ne!(obs.source, erp.source);
        assert_eq!((obs.world, erp.world), (0x9101, 0x9102));
    }

    /// A later generation may legally re-read the same bits, and may drop a
    /// pair the earlier one mapped.
    #[test]
    fn a_new_generation_may_read_the_bits_differently() {
        assert_eq!(stamped(CLASS_OBS, 1, edge(23, 5)).unwrap().source, sc(1, 7));
        assert_eq!(stamped(CLASS_OBS, 2, edge(23, 5)).unwrap().source, sc(1, 8));
        assert_eq!(stamped(CLASS_OBS, 1, edge(23, 2)).unwrap().source, sc(1, 3));
        assert_eq!(
            stamped(CLASS_OBS, 2, edge(23, 2)),
            Err(Refusal::Unmapped {
                witness: 23,
                angle: 2
            })
        );
    }

    /// Every way the read can be illegitimate refuses, each for its own reason.
    #[test]
    fn illegitimate_reads_refuse() {
        let e = edge(23, 5);
        assert_eq!(
            stamped(CLASS_UNDECLARED, 1, e),
            Err(Refusal::Undeclared(CLASS_UNDECLARED))
        );
        assert_eq!(
            stamped(CLASS_PATHOLOGY, 1, e),
            Err(Refusal::OtherReading(SlotReading::PathologyTriad))
        );
        assert_eq!(
            stamped(CLASS_COHORT, 1, e),
            Err(Refusal::OtherReading(SlotReading::CohortWitness))
        );
        assert_eq!(
            stamped(CLASS_ERP, 2, e),
            Err(Refusal::UndeclaredGeneration(2))
        );
        assert_eq!(stamped(CLASS_OBS, 1, edge(0, 5)), Err(Refusal::NoAnchor));
        for p in [EdgeProvenance::V1Legacy, EdgeProvenance::Unknown] {
            assert_eq!(
                local_source(CLASS_OBS, 1, e, p),
                Err(Refusal::Provenance(p))
            );
        }
        // Silent twin: the legitimate read of the same edge succeeds.
        assert!(stamped(CLASS_OBS, 1, e).is_ok());
    }

    /// No global meaning: over all 63 × 8 anchored pairs, each class maps only
    /// its own few, and the two classes do not map the same set to the same
    /// sources.
    #[test]
    fn the_law_is_sparse_and_class_scoped() {
        let mapped = |c: u32| -> Vec<(u8, u8, SourceCoord)> {
            (1..64u8)
                .flat_map(|w| (0..8u8).map(move |a| (w, a)))
                .filter_map(|(w, a)| stamped(c, 1, edge(w, a)).ok().map(|s| (w, a, s.source)))
                .collect()
        };
        let (obs, erp) = (mapped(CLASS_OBS), mapped(CLASS_ERP));
        assert_eq!(obs.len(), 3, "{obs:?}");
        assert_eq!(erp.len(), 1, "{erp:?}");
        assert!(obs.len() * 100 < 63 * 8, "sparse");
        assert!(erp.iter().all(|x| !obs.contains(x)), "no shared meaning");
    }

    /// Stay-silent: fields other than Witness and Angle do not move the
    /// coordinate. Can-fire: changing Witness or Angle does.
    #[test]
    fn only_witness_and_angle_address_the_source() {
        let base = stamped(CLASS_OBS, 1, edge(23, 5)).unwrap();
        let others = [
            CausalEdge64::pack_v2(
                9,
                8,
                7,
                10,
                20,
                CausalMask::SPO,
                5,
                PlasticityState::ALL_HOT,
            )
            .with_w_slot(23),
            edge(23, 5).with_inference_mantissa(-6),
            CausalEdge64::pack_v2(0, 0, 0, 0, 0, CausalMask::None, 5, PlasticityState::ALL_HOT)
                .with_w_slot(23),
        ];
        for e in others {
            assert_eq!(stamped(CLASS_OBS, 1, e).unwrap(), base, "{:016x}", e.0);
        }
        assert_ne!(
            stamped(CLASS_OBS, 1, edge(4, 5)).unwrap(),
            base,
            "Witness moves it"
        );
        assert_ne!(
            stamped(CLASS_OBS, 1, edge(23, 2)).unwrap(),
            base,
            "Angle moves it"
        );
    }

    /// The read allocates nothing.
    #[test]
    fn the_read_allocates_nothing() {
        let edges: [CausalEdge64; 64] = core::array::from_fn(|i| {
            // Half the edges hit the law, half miss it: both paths are measured.
            if i % 2 == 0 {
                edge(23, 5)
            } else {
                edge(i as u8, (i % 8) as u8)
            }
        });
        let (hits, allocs) = counting(|| {
            edges
                .iter()
                .filter(|e| stamped(CLASS_OBS, 1, **e).is_ok())
                .count()
        });
        assert_eq!(allocs, 0);
        assert_eq!(hits, 32, "the hit path ran");
    }
}
