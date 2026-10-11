//! **D-MATURATION3-0 — Novelty → Wisdom as an eight-stage 3-bit ladder.**
//!
//! EXPERIMENTAL, test-local. Nothing here changes production code.
//!
//! # Why this is test-local (CONTRACT DECISION, OPEN)
//!
//! Bits 50..52 of `CausalEdge64` are plasticity under the v2 layout:
//! `CausalEdge64::learn` gates S/P/O adoption on them, `MooreNars16` packs the
//! same field at bit 50, and `CausalEdgeV3` carries it in byte 3. The
//! `staunen_scheduler_probe` (D-CE64-STAUNEN-0) measured that writing another
//! meaning there changes what `learn` does. So no existing provenance may read
//! these bits as maturation: the reader below accepts only an explicitly
//! declared experimental reading and refuses `V2Stamped`, `V3Register`,
//! `V1Legacy` and `Unknown`. Migrating the bits is an operator decision.
//!
//! # What is tested
//!
//! - The codebook (0..7) and its bit isolation on a real `CausalEdge64`.
//! - `mature`: one small, deterministic, allocation-free transition. It moves
//!   only on an executed result; it never moves because a pass completed.
//! - A test-local feedback loop that reuses production `pearl::hydrate`
//!   unchanged: resolved relations become premises for later hydrations, with
//!   lineage. A derived premise carries the union of its parents' observation
//!   roots and never adds a root of its own.
//! - Ablation through bits 59..63 in an isolated copy, never the original.
//!
//! Run with `--nocapture` to see the measured timings.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

use causal_edge::edge::InferenceType;
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    Certification3, Epi5Declarations, Epi5Gen, Epi5Reading, EpistemicState5, Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;
use lance_graph_planner::chain_replay::ChainStep;
use lance_graph_planner::pearl::{hydrate, Hydration, Reading};

// ── Allocation counter (measurement only) ─────────────────────────────────

struct Counting;
thread_local! {
    // Per-thread so parallel tests do not pollute the count.
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

// SAFETY: forwards every call to `System` unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

// ── The codebook ──────────────────────────────────────────────────────────

/// Eight cognitive maturation stages. The value IS the 3-bit code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum Maturation3 {
    /// Consolidated explanation, demonstrably reusable as an anchor.
    Wisdom = 0,
    /// The validated explanation produced an inference in another context.
    Integration = 1,
    /// The mechanism passed a discriminating test.
    Falsification = 2,
    /// An explanatory mechanism exists and can be tested.
    Mechanism = 3,
    /// Candidate bridges identified.
    Candidates = 4,
    /// Missing link localized and bookmarked.
    DogEar = 5,
    /// The discrepancy is independently substantiated.
    Anomaly = 6,
    /// A new expectation violation.
    Staunen = 7,
}

impl Maturation3 {
    const ALL: [Maturation3; 8] = [
        Maturation3::Wisdom,
        Maturation3::Integration,
        Maturation3::Falsification,
        Maturation3::Mechanism,
        Maturation3::Candidates,
        Maturation3::DogEar,
        Maturation3::Anomaly,
        Maturation3::Staunen,
    ];

    const fn from_code(c: u8) -> Self {
        Self::ALL[(c & 0b111) as usize]
    }
}

const M3_SHIFT: u32 = 50;
const M3_MASK: u64 = 0b111 << M3_SHIFT;

/// The declared reading. Only an explicit experimental declaration may read
/// bits 50..52 as maturation; every existing provenance means plasticity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum M3Reading {
    /// Declared by this probe for its own edges.
    Experimental,
    /// Any edge minted by existing producers.
    Existing(EdgeProvenance),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaturationError {
    /// The bits mean plasticity under this provenance.
    PlasticityReading(EdgeProvenance),
}

fn read_m3(e: CausalEdge64, r: M3Reading) -> Result<Maturation3, MaturationError> {
    match r {
        M3Reading::Experimental => Ok(Maturation3::from_code(((e.0 & M3_MASK) >> M3_SHIFT) as u8)),
        M3Reading::Existing(p) => Err(MaturationError::PlasticityReading(p)),
    }
}

/// A maturation-only write: bits 50..52 and nothing else.
fn write_m3(
    e: CausalEdge64,
    m: Maturation3,
    r: M3Reading,
) -> Result<CausalEdge64, MaturationError> {
    read_m3(e, r)?;
    Ok(CausalEdge64((e.0 & !M3_MASK) | ((m as u64) << M3_SHIFT)))
}

// ── The transition ────────────────────────────────────────────────────────

/// What an executed operation measured. `Copy`, no allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Executed {
    /// An observation contradicted an expectation. `independent_roots` counts
    /// distinct observation roots behind the contradiction.
    Violation { independent_roots: u32 },
    /// `pearl::hydrate` ran on this relation.
    Hydrated(Hydration),
    /// A discriminating test between competing mechanisms.
    Discriminated {
        survivor_passed: bool,
        rival_ruled_out: bool,
    },
    /// The relation was used as a premise elsewhere; `new_inferences` counts
    /// relations that resolved only because of it.
    Applied { new_inferences: u32 },
    /// The relation was cut in an isolated copy; `impact` counts dependent
    /// relations that fell back to `IndirectUnknown`.
    Ablated { impact: u32 },
}

/// The one transition. Moves only on an earned condition. A forward target
/// never moves the state backward; only a violation reopens it. Unsupported
/// results abstain.
fn mature(
    current: Maturation3,
    result: Executed,
    reading: M3Reading,
) -> Result<Maturation3, MaturationError> {
    if let M3Reading::Existing(p) = reading {
        return Err(MaturationError::PlasticityReading(p));
    }
    use Maturation3 as M;
    let target = match result {
        Executed::Violation {
            independent_roots: 0,
        } => return Ok(current),
        Executed::Violation {
            independent_roots: 1,
        } => return Ok(M::Staunen),
        Executed::Violation { .. } => return Ok(M::Anomaly),
        Executed::Hydrated(Hydration::ProposedOnly) => M::DogEar,
        Executed::Hydrated(Hydration::Partial) => M::Candidates,
        Executed::Hydrated(Hydration::Hydrated) => M::Mechanism,
        Executed::Discriminated {
            survivor_passed: true,
            rival_ruled_out: true,
        } if current <= M::Mechanism => M::Falsification,
        Executed::Discriminated { .. } => return Ok(current),
        Executed::Applied { new_inferences }
            if new_inferences > 0 && current <= M::Falsification =>
        {
            M::Integration
        }
        Executed::Applied { .. } => return Ok(current),
        Executed::Ablated { impact } if impact > 0 && current == M::Integration => M::Wisdom,
        Executed::Ablated { .. } => return Ok(current),
    };
    // Forward results never move the state backward.
    Ok(current.min(target))
}

// ── Fixture: edges, readings, premises with lineage ───────────────────────

const CLASS: ClassId = 0x0902;
const RAIL: RailAxis = RailAxis::Taxonomy;
const PREDICATE: u8 = 0x91;
const EPI_BITS: u64 = 0b11111 << 59;

fn declarations() -> Epi5Declarations {
    let mut d = Epi5Declarations::new();
    d.declare(
        CLASS,
        RAIL,
        Epi5Reading {
            generation: Epi5Gen::V1,
        },
    );
    d
}

fn reading(d: &Epi5Declarations) -> Reading<'_> {
    Reading {
        declarations: d,
        class: CLASS,
        rail: RAIL,
        provenance: EdgeProvenance::V2Stamped,
    }
}

fn raw_edge(s: u8, o: u8, freq: u8) -> CausalEdge64 {
    CausalEdge64::pack(
        s,
        PREDICATE,
        o,
        freq,
        200,
        CausalMask::SO,
        0b101,
        InferenceType::Deduction,
        PlasticityState::from_bits(0),
        0,
    )
}

fn open(s: u8, o: u8) -> CausalEdge64 {
    let st = EpistemicState5::new(
        Epi5Gen::V1,
        Topology2::IndirectUnknown,
        Certification3::Related,
    );
    raw_edge(s, o, 200).with_epistemic_raw5(st.raw())
}

fn topology(e: CausalEdge64) -> Topology2 {
    EpistemicState5::decode(Epi5Gen::V1, e.epistemic_raw5())
        .expect("valid code")
        .topology()
}

/// A premise: an edge plus its lineage.
#[derive(Debug, Clone, Copy)]
struct Premise {
    edge: CausalEdge64,
    /// Observation roots behind it (one bit per sealed observation).
    roots: u64,
    /// Derivation events behind it (one bit per derived relation).
    derivations: u64,
}

/// An open relation with its lineage once resolved.
#[derive(Debug, Clone, Copy)]
struct Relation {
    edge: CausalEdge64,
    id: u32,
    roots: u64,
    derivations: u64,
}

fn observation(s: u8, o: u8, root: u32) -> Premise {
    Premise {
        edge: raw_edge(s, o, 200),
        roots: 1 << root,
        derivations: 0,
    }
}

/// Knobs that the disable runs flip. Production behaviour is `feedback: false`.
#[derive(Clone, Copy)]
struct Loop {
    feedback: bool,
    /// DISABLE knob: let still-open relations act as premises.
    open_premises: bool,
}

const LOOP: Loop = Loop {
    feedback: true,
    open_premises: false,
};
const BASELINE: Loop = Loop {
    feedback: false,
    open_premises: false,
};

/// One pass. Every open relation tries every intermediate the premises
/// propose, through production `pearl::hydrate` over a premise chain.
/// Returns the number of relations resolved in this pass.
fn pass(
    rels: &mut [Relation],
    obs: &[Premise],
    banned: Option<u32>,
    cfg: Loop,
    d: &Epi5Declarations,
) -> usize {
    let mut resolved = 0;
    for i in 0..rels.len() {
        if topology(rels[i].edge) != Topology2::IndirectUnknown {
            continue;
        }
        // The premise pool: observations, plus (with feedback) resolved
        // relations that neither are this relation nor were derived from it.
        let mut pool: Vec<Premise> = obs.to_vec();
        if cfg.feedback {
            for r in rels.iter() {
                // Only RESOLVED relations are premises. That is the
                // self-support guard: an open relation can never ground
                // another, so a cycle cannot bootstrap itself (T8 disable).
                let usable = Some(r.id) != banned
                    && (cfg.open_premises || topology(r.edge) != Topology2::IndirectUnknown);
                if usable {
                    pool.push(Premise {
                        edge: raw_edge(r.edge.s_idx(), r.edge.o_idx(), 200),
                        roots: r.roots,
                        derivations: r.derivations | (1 << r.id),
                    });
                }
            }
        }
        let chain: Vec<ChainStep> = pool.iter().map(|p| (0, p.edge)).collect();
        let (a, y) = (rels[i].edge.s_idx(), rels[i].edge.o_idx());
        for p in pool.iter().filter(|p| p.edge.s_idx() == a) {
            let b = p.edge.o_idx();
            let (next, h) = hydrate(rels[i].edge, (a, b, y), &chain, reading(d)).expect("declared");
            if h == Hydration::Hydrated && topology(next) == Topology2::IndirectKnown {
                let second = pool
                    .iter()
                    .find(|q| q.edge.s_idx() == b && q.edge.o_idx() == y)
                    .expect("hydrated means both bindings exist");
                rels[i].edge = next;
                rels[i].roots = p.roots | second.roots;
                rels[i].derivations = p.derivations | second.derivations;
                resolved += 1;
                break;
            }
        }
    }
    resolved
}

/// Run to a fixed point; return the number of passes that resolved anything.
fn fixed_point(
    rels: &mut [Relation],
    obs: &[Premise],
    banned: Option<u32>,
    cfg: Loop,
    d: &Epi5Declarations,
) -> usize {
    for passes in 0..32 {
        if pass(rels, obs, banned, cfg, d) == 0 {
            return passes;
        }
    }
    panic!("no fixed point in 32 passes");
}

fn residue(rels: &[Relation]) -> usize {
    rels.iter()
        .filter(|r| topology(r.edge) == Topology2::IndirectUnknown)
        .count()
}

fn rel(id: u32, s: u8, o: u8) -> Relation {
    Relation {
        edge: open(s, o),
        id,
        roots: 0,
        derivations: 0,
    }
}

/// The dependency chain the prompt asks for:
/// sealed obs → A (1→3) → B (1→4) → C (0→4).
fn chain_world() -> (Vec<Premise>, Vec<Relation>) {
    let obs = vec![
        observation(1, 2, 0),
        observation(2, 3, 1),
        observation(3, 4, 2),
        observation(0, 1, 3),
    ];
    let rels = vec![rel(0, 1, 3), rel(1, 1, 4), rel(2, 0, 4)];
    (obs, rels)
}

// ── T1: decoding and bit isolation ────────────────────────────────────────

#[test]
fn t1_eight_codes_round_trip_and_only_bits_50_52_move() {
    let base = open(9, 7).with_w_slot(0b101101);
    for m in Maturation3::ALL {
        let e = write_m3(base, m, M3Reading::Experimental).unwrap();
        assert_eq!(read_m3(e, M3Reading::Experimental), Ok(m));
        assert_eq!(
            e.0 & !M3_MASK,
            base.0 & !M3_MASK,
            "only bits 50..52 may change"
        );
        assert_eq!(e.frequency(), base.frequency());
        assert_eq!(e.confidence(), base.confidence());
        assert_eq!(e.w_slot(), base.w_slot());
        assert_eq!(e.epistemic_raw5(), base.epistemic_raw5());
    }
    for c in 0u8..8 {
        assert_eq!(Maturation3::from_code(c) as u8, c);
    }
}

#[test]
fn t1_every_existing_provenance_refuses_the_maturation_reading() {
    let e = open(9, 7);
    for p in [
        EdgeProvenance::V2Stamped,
        EdgeProvenance::V3Register,
        EdgeProvenance::V1Legacy,
        EdgeProvenance::Unknown,
    ] {
        assert_eq!(
            read_m3(e, M3Reading::Existing(p)),
            Err(MaturationError::PlasticityReading(p))
        );
        assert!(write_m3(e, Maturation3::Wisdom, M3Reading::Existing(p)).is_err());
        assert!(mature(
            Maturation3::Integration,
            Executed::Ablated { impact: 3 },
            M3Reading::Existing(p)
        )
        .is_err());
    }
}

// ── T2: cheap fixed point, no drift toward Wisdom ─────────────────────────

#[test]
fn t2_repeating_a_completed_result_never_advances_maturation() {
    let x = M3Reading::Experimental;
    let mut m = Maturation3::Staunen;
    m = mature(m, Executed::Hydrated(Hydration::Hydrated), x).unwrap();
    assert_eq!(m, Maturation3::Mechanism, "a hydration may skip stages");
    for _ in 0..100 {
        m = mature(m, Executed::Hydrated(Hydration::Hydrated), x).unwrap();
    }
    assert_eq!(
        m,
        Maturation3::Mechanism,
        "completed passes are not progress"
    );
    // A forward result never moves a more mature state backward.
    assert_eq!(
        mature(
            Maturation3::Falsification,
            Executed::Hydrated(Hydration::Hydrated),
            x
        )
        .unwrap(),
        Maturation3::Falsification
    );
    assert_eq!(
        mature(
            Maturation3::Integration,
            Executed::Hydrated(Hydration::Partial),
            x
        )
        .unwrap(),
        Maturation3::Integration
    );

    let d = declarations();
    let (obs, mut rels) = chain_world();
    fixed_point(&mut rels, &obs, None, LOOP, &d);
    let snapshot: Vec<u64> = rels.iter().map(|r| r.edge.0).collect();
    for _ in 0..16 {
        pass(&mut rels, &obs, None, LOOP, &d);
    }
    assert_eq!(rels.iter().map(|r| r.edge.0).collect::<Vec<_>>(), snapshot);
}

// ── T3: the Eselsohr survives ─────────────────────────────────────────────

#[test]
fn t3_a_localized_gap_keeps_its_dogear_and_witness_until_addressed() {
    let x = M3Reading::Experimental;
    let d = declarations();
    // 7 → 9 with a proposal 7 → 8 but no 8 → 9.
    let obs = [observation(7, 8, 0)];
    let mut r = [rel(0, 7, 9)];
    let (_, h) = hydrate(r[0].edge, (7, 8, 9), &[(0, obs[0].edge)], reading(&d)).unwrap();
    let m = mature(Maturation3::Staunen, Executed::Hydrated(h), x).unwrap();
    assert_eq!(m, Maturation3::Candidates, "one binding found: candidates");

    // The witness handle points at the transition record (test-local log).
    let log: Vec<(u32, Maturation3)> = vec![(r[0].id, m)];
    r[0].edge = write_m3(r[0].edge, m, x)
        .unwrap()
        .with_w_slot(log.len() as u8 - 1);
    let marked = r[0].edge;
    for _ in 0..8 {
        pass(&mut r, &obs, None, LOOP, &d);
    }
    assert_eq!(
        r[0].edge, marked,
        "passes over the same evidence keep the mark"
    );
    assert_eq!(
        log[r[0].edge.w_slot() as usize],
        (0, Maturation3::Candidates)
    );
}

// ── T4: a new inference without a new observation ─────────────────────────

#[test]
fn t4_feedback_resolves_what_the_sealed_chain_alone_cannot() {
    let d = declarations();
    let (obs, mut base) = chain_world();
    fixed_point(&mut base, &obs, None, BASELINE, &d);
    assert_eq!(residue(&base), 2, "baseline: only A resolves");

    let (obs, mut fed) = chain_world();
    let passes = fixed_point(&mut fed, &obs, None, LOOP, &d);
    assert_eq!(residue(&fed), 0, "feedback: A, B, then C");
    assert!(passes >= 1);
    // Same observations, no new one: four roots, never more.
    assert_eq!(fed[2].roots.count_ones(), 4);
    assert_eq!(
        fed[1].roots, 0b0111,
        "B carries exactly the roots of its parents"
    );
    assert_eq!(
        fed[2].derivations & (1 << 1),
        1 << 1,
        "C records B as a derivation, not a root"
    );
}

// ── T5: falsification never certifies ─────────────────────────────────────

#[test]
fn t5_a_discriminating_test_matures_but_does_not_certify() {
    let x = M3Reading::Experimental;
    let d = declarations();
    // 30 → 32 via 31 (both bindings) or via 33 (only one).
    let chain: Vec<ChainStep> = [(30, 31), (31, 32), (30, 33)]
        .iter()
        .map(|&(s, o)| (0, raw_edge(s, o, 200)))
        .collect();
    let e = open(30, 32);
    let (_, via31) = hydrate(e, (30, 31, 32), &chain, reading(&d)).unwrap();
    let (_, via33) = hydrate(e, (30, 33, 32), &chain, reading(&d)).unwrap();
    assert_eq!((via31, via33), (Hydration::Hydrated, Hydration::Partial));

    let m = mature(
        Maturation3::Mechanism,
        Executed::Discriminated {
            survivor_passed: via31 == Hydration::Hydrated,
            rival_ruled_out: via33 != Hydration::Hydrated,
        },
        x,
    )
    .unwrap();
    assert_eq!(m, Maturation3::Falsification);
    let written = write_m3(e, m, x).unwrap();
    assert_eq!(
        written.0 & EPI_BITS,
        e.0 & EPI_BITS,
        "maturation never touches certification"
    );

    // A test that rules nothing out earns nothing.
    let m2 = mature(
        Maturation3::Mechanism,
        Executed::Discriminated {
            survivor_passed: true,
            rival_ruled_out: false,
        },
        x,
    )
    .unwrap();
    assert_eq!(m2, Maturation3::Mechanism);
}

// ── T6 + T7: integration, then the anchor ablation ────────────────────────

#[test]
fn t6_t7_an_applied_anchor_integrates_and_its_ablation_measures_wisdom() {
    let x = M3Reading::Experimental;
    let d = declarations();
    let (obs, mut rels) = chain_world();
    fixed_point(&mut rels, &obs, None, LOOP, &d);
    let original: Vec<u64> = rels.iter().map(|r| r.edge.0).collect();

    // T6: A was used as a premise; count relations whose lineage holds A.
    let new_inferences = rels.iter().filter(|r| r.derivations & 1 != 0).count() as u32;
    assert_eq!(new_inferences, 2, "B and C resolved through A");
    let mut m = mature(
        Maturation3::Falsification,
        Executed::Applied { new_inferences },
        x,
    )
    .unwrap();
    assert_eq!(m, Maturation3::Integration);

    // T7: cut A in an isolated copy and recompute.
    let (obs2, mut cut) = chain_world();
    let r_baseline = residue(&rels);
    let passes_cut = fixed_point(&mut cut, &obs2, Some(0), LOOP, &d);
    let impact = (residue(&cut) - r_baseline) as u32;
    // A itself still resolves from observations; B and C lose their bridge.
    assert_eq!(impact, 2, "AnchorImpact = R_cut - R_baseline");
    assert_eq!(topology(cut[1].edge), Topology2::IndirectUnknown);
    assert_eq!(topology(cut[2].edge), Topology2::IndirectUnknown);
    assert!(passes_cut <= 1);
    assert_eq!(
        rels.iter().map(|r| r.edge.0).collect::<Vec<_>>(),
        original,
        "the original is never overwritten"
    );

    m = mature(m, Executed::Ablated { impact }, x).unwrap();
    assert_eq!(m, Maturation3::Wisdom);

    // Applying something that enabled nothing is not integration.
    assert_eq!(
        mature(
            Maturation3::Falsification,
            Executed::Applied { new_inferences: 0 },
            x
        )
        .unwrap(),
        Maturation3::Falsification
    );
    // An anchor whose cut changes nothing earns nothing.
    assert_eq!(
        mature(Maturation3::Integration, Executed::Ablated { impact: 0 }, x).unwrap(),
        Maturation3::Integration
    );
}

// ── T8: no echo, no self-support ──────────────────────────────────────────

#[test]
fn t8_two_paths_from_the_same_roots_are_not_two_supports() {
    let d = declarations();
    // 5 → 7 via 6 and via 8; both first legs come from root 0, both second
    // legs from root 1.
    let shared = observation(5, 6, 0);
    let obs = [
        shared,
        Premise {
            edge: raw_edge(5, 8, 200),
            ..shared
        },
        observation(6, 7, 1),
        Premise {
            edge: raw_edge(8, 7, 200),
            ..observation(6, 7, 1)
        },
    ];
    let paths: Vec<u64> = [6u8, 8]
        .iter()
        .map(|&b| {
            let first = obs
                .iter()
                .find(|p| p.edge.s_idx() == 5 && p.edge.o_idx() == b)
                .unwrap();
            let second = obs
                .iter()
                .find(|p| p.edge.s_idx() == b && p.edge.o_idx() == 7)
                .unwrap();
            first.roots | second.roots
        })
        .collect();
    let union = paths.iter().fold(0, |a, p| a | p);
    assert_eq!(
        union.count_ones(),
        2,
        "two paths, two observations, two supports"
    );
    let echo: u32 = paths.iter().map(|p| p.count_ones()).sum();
    assert_eq!(
        echo, 4,
        "summing per path would double-count: the disable arm"
    );

    // Self-support: two open relations that each need the other.
    let obs = vec![observation(21, 22, 0), observation(22, 21, 1)];
    let mut cyc = vec![rel(0, 20, 22), rel(1, 20, 21)];
    fixed_point(&mut cyc, &obs, None, LOOP, &d);
    assert_eq!(
        residue(&cyc),
        2,
        "neither may ground the other from nothing"
    );
}

#[test]
fn t8_disable_open_premises_lets_a_cycle_ground_itself() {
    let d = declarations();
    let obs = vec![observation(21, 22, 0), observation(22, 21, 1)];
    let mut cyc = vec![rel(0, 20, 22), rel(1, 20, 21)];
    let broken = Loop {
        feedback: true,
        open_premises: true,
    };
    fixed_point(&mut cyc, &obs, None, broken, &d);
    assert!(
        residue(&cyc) < 2,
        "the guard is load-bearing: without it the cycle resolves"
    );
}

// ── T9: wisdom creates new surprise ───────────────────────────────────────

/// An observation violates an expectation only if a resolved relation
/// predicts the opposite.
fn violates(rels: &[Relation], s: u8, o: u8, freq: u8) -> bool {
    rels.iter().any(|r| {
        r.edge.s_idx() == s
            && r.edge.o_idx() == o
            && topology(r.edge) == Topology2::IndirectKnown
            && (freq < 128) != (r.edge.frequency() < 0.5)
    })
}

#[test]
fn t9_an_anchor_turns_a_neutral_observation_into_an_anomaly() {
    let d = declarations();
    // "1 → 4 does not hold" (low frequency).
    let (s, o, f) = (1u8, 4u8, 20u8);

    let (obs, mut without) = chain_world();
    fixed_point(&mut without, &obs, None, BASELINE, &d);
    assert!(
        !violates(&without, s, o, f),
        "no model of 1 → 4: nothing to contradict"
    );

    let (obs, mut with) = chain_world();
    fixed_point(&mut with, &obs, None, LOOP, &d);
    assert!(
        violates(&with, s, o, f),
        "the anchor chain now predicts 1 → 4"
    );
    let m = mature(
        Maturation3::Wisdom,
        Executed::Violation {
            independent_roots: 1,
        },
        M3Reading::Experimental,
    )
    .unwrap();
    assert_eq!(m, Maturation3::Staunen);
}

// ── T10: regression and replay ────────────────────────────────────────────

#[test]
fn t10_counterevidence_reopens_and_old_plasticity_is_never_read_as_wisdom() {
    let x = M3Reading::Experimental;
    assert_eq!(
        mature(
            Maturation3::Wisdom,
            Executed::Violation {
                independent_roots: 1
            },
            x
        )
        .unwrap(),
        Maturation3::Staunen
    );
    assert_eq!(
        mature(
            Maturation3::Wisdom,
            Executed::Violation {
                independent_roots: 3
            },
            x
        )
        .unwrap(),
        Maturation3::Anomaly
    );
    assert_eq!(
        mature(
            Maturation3::Wisdom,
            Executed::Violation {
                independent_roots: 0
            },
            x
        )
        .unwrap(),
        Maturation3::Wisdom
    );

    // A historic v2 edge whose plasticity bits happen to be 0b000.
    let mut old = raw_edge(1, 2, 200);
    old.set_plasticity(PlasticityState::ALL_FROZEN);
    assert_eq!(old.0 & M3_MASK, 0, "same bit pattern as Wisdom");
    assert!(read_m3(old, M3Reading::Existing(EdgeProvenance::V2Stamped)).is_err());
}

// ── Measurement ───────────────────────────────────────────────────────────

#[test]
fn measure_transition_cost_and_allocations() {
    let x = M3Reading::Experimental;
    let inputs = [
        Executed::Violation {
            independent_roots: 1,
        },
        Executed::Hydrated(Hydration::Partial),
        Executed::Hydrated(Hydration::Hydrated),
        Executed::Discriminated {
            survivor_passed: true,
            rival_ruled_out: true,
        },
        Executed::Applied { new_inferences: 2 },
        Executed::Ablated { impact: 2 },
    ];
    let before = ALLOCS.with(Cell::get);
    let mut m = Maturation3::Staunen;
    let mut acc = 0u64;
    for i in 0..1_000_000usize {
        m = mature(m, inputs[i % inputs.len()], x).unwrap();
        acc = acc.wrapping_add(m as u64);
    }
    let allocs = ALLOCS.with(Cell::get) - before;
    assert_eq!(allocs, 0, "mature must not allocate");

    // Timing: batches of 10_000 transitions.
    let mut ns: Vec<f64> = (0..200)
        .map(|_| {
            let t = Instant::now();
            for i in 0..10_000usize {
                m = mature(std::hint::black_box(m), inputs[i % inputs.len()], x).unwrap();
            }
            t.elapsed().as_nanos() as f64 / 10_000.0
        })
        .collect();
    ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "mature: median {:.2} ns, p95 {:.2} ns per transition (debug unless --release); allocs {allocs}; checksum {acc}",
        ns[100], ns[190]
    );
}

#[test]
fn measure_feedback_overhead() {
    let d = declarations();
    let time = |cfg: Loop| {
        let mut ns: Vec<u128> = (0..200)
            .map(|_| {
                let (obs, mut rels) = chain_world();
                let t = Instant::now();
                std::hint::black_box(fixed_point(&mut rels, &obs, None, cfg, &d));
                t.elapsed().as_nanos()
            })
            .collect();
        ns.sort_unstable();
        let (obs, mut rels) = chain_world();
        let before = ALLOCS.with(Cell::get);
        let passes = fixed_point(&mut rels, &obs, None, cfg, &d);
        let allocs = ALLOCS.with(Cell::get) - before;
        (ns[100], ns[190], passes, residue(&rels), allocs)
    };
    let b = time(BASELINE);
    let f = time(LOOP);
    for (name, r) in [("baseline", b), ("feedback", f)] {
        println!(
            "fixed point, 3 relations / 4 observations, {name}: median {} ns, p95 {} ns, {} resolving passes, residue {}, allocs {}",
            r.0, r.1, r.2, r.3, r.4
        );
    }
    assert!(
        f.3 < b.3,
        "feedback discovers relations the baseline cannot"
    );
}
