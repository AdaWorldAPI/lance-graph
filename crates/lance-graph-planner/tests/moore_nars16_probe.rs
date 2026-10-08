//! MooreNars16 probe: a declared Moore-local representation of the
//! ISA-visible NARS state of a `CausalEdge64`.
//!
//! The claim under test is NOT that 16 bits round-trip the canonical 24.
//! It is:
//!
//! ```text
//! canonical logical operand          MooreContext + SPOFC40 + MooreNars16
//!            |                                     |
//!            +------------- same ISA op -----------+
//!                               |
//!               same architecturally observable result
//! ```
//!
//! The two canonical fields MooreNars16 does not hold are absent for
//! different reasons, and each has its own falsifier:
//!
//! - **Direction3** is the canonical S/P/O sign triple. No ISA operation reads
//!   it to compute anything (`forward` copies the weight's, `syllogize` writes
//!   0, `learn` never reads it). A Moore tenant declares a DIFFERENT reading:
//!   lane geometry (the slot) x a polarity bit. A consumer of the sign triple
//!   (Simpson detection, `network.rs`) must refuse the Moore reading.
//! - **Witness6** is lifted to the tenant: one W for all eight lanes. That is
//!   valid only if two lanes of one tenant never need different W at once;
//!   the projection refuses a tenant whose lanes disagree.
//!
//! Lane layout (u16, probe-local, NOT a contract):
//!
//! ```text
//! bits  0..2   Pearl3        (CE64 40..42)
//! bits  3..6   Energy4       (CE64 46..49, the raw signed mantissa nibble)
//! bits  7..9   Plasticity3   (CE64 50..52)
//! bit   10     Polarity1     (Moore-local; no canonical home)
//! bits 11..15  Epi5          (CE64 59..63)
//! ```
//!
//! Nothing in production uses this type. It changes no CE64 layout and no
//! operation.

use causal_edge::edge::CausalEdge64;
use lance_graph_contract::epistemic_state5::{Epi5Gen, EpistemicState5};

// ─── CE64 field geometry (v2 layout) ────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    Spo,
    F,
    C,
    Pearl,
    Direction,
    Energy,
    Plasticity,
    W,
    Epi5,
}

const FIELDS: [Field; 9] = [
    Field::Spo,
    Field::F,
    Field::C,
    Field::Pearl,
    Field::Direction,
    Field::Energy,
    Field::Plasticity,
    Field::W,
    Field::Epi5,
];

impl Field {
    const fn span(self) -> (u32, u32) {
        match self {
            Field::Spo => (0, 24),
            Field::F => (24, 8),
            Field::C => (32, 8),
            Field::Pearl => (40, 3),
            Field::Direction => (43, 3),
            Field::Energy => (46, 4),
            Field::Plasticity => (50, 3),
            Field::W => (53, 6),
            Field::Epi5 => (59, 5),
        }
    }
    fn mask(self) -> u64 {
        let (s, w) = self.span();
        ((1u64 << w) - 1) << s
    }
    fn get(self, e: CausalEdge64) -> u64 {
        (e.0 & self.mask()) >> self.span().0
    }
    fn set(self, e: CausalEdge64, v: u64) -> CausalEdge64 {
        CausalEdge64((e.0 & !self.mask()) | ((v << self.span().0) & self.mask()))
    }
}

/// Every field an ISA result is judged on: everything except the canonical
/// sign triple, which the Moore reading deliberately does not carry.
fn observable(e: CausalEdge64) -> u64 {
    e.0 & !Field::Direction.mask()
}

// ─── The lane ───────────────────────────────────────────────────────────

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MooreNars16(u16);

impl MooreNars16 {
    fn new(pearl: u8, energy: u8, plasticity3: u8, polarity: bool, epi5: u8) -> Self {
        Self(
            u16::from(pearl & 7)
                | u16::from(energy & 0xF) << 3
                | u16::from(plasticity3 & 7) << 7
                | u16::from(polarity) << 10
                | u16::from(epi5 & 0x1F) << 11,
        )
    }
    fn pearl(self) -> u8 {
        (self.0 & 7) as u8
    }
    fn energy(self) -> u8 {
        (self.0 >> 3 & 0xF) as u8
    }
    fn plasticity4(self) -> u8 {
        (self.0 >> 7 & 0xF) as u8
    }
    fn plasticity3(self) -> u8 {
        self.plasticity4() & 7
    }
    fn polarity(self) -> bool {
        self.plasticity4() & 8 != 0
    }
    fn epi5(self) -> u8 {
        (self.0 >> 11) as u8
    }
}

/// Lane geometry: lane index -> (dx, dy). Fixed by the tenant, never stored.
const SLOTS: [(i8, i8); 8] = [
    (-1, -1), // NW
    (0, -1),  // N
    (1, -1),  // NE
    (-1, 0),  // W
    (1, 0),   // E
    (-1, 1),  // SW
    (0, 1),   // S
    (1, 1),   // SE
];

/// The directed Moore relation a lane denotes: (from, to) relative to the
/// tenant's centre. Outbound: centre -> neighbour. Inbound: neighbour -> centre.
fn directed_relation(slot: usize, lane: MooreNars16) -> ((i8, i8), (i8, i8)) {
    let n = SLOTS[slot];
    if lane.polarity() {
        (n, (0, 0))
    } else {
        ((0, 0), n)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirectionReading {
    /// CE64 bits 43..45 are the canonical S/P/O sign triple.
    SignTriple,
    /// Direction is (lane slot, polarity); CE64 bits 43..45 carry nothing.
    Moore { slot: usize, polarity: bool },
}

/// A logical operand: the CE64 the ISA executes on, plus how its direction
/// bits are to be read.
#[derive(Clone, Copy, Debug)]
struct Operand {
    edge: CausalEdge64,
    reading: DirectionReading,
}

#[derive(Debug, PartialEq, Eq)]
enum Refusal {
    /// An Epi5 code outside the declared codebook (certification 6 or 7).
    ReservedEpi5(u8),
    /// The lanes of one tenant carry different witnesses.
    WitnessScope { lane: usize, expected: u8, found: u8 },
}

/// A sign-triple consumer (Simpson's pattern S and O pathological, P not,
/// as `CausalNetwork::detect_simpsons_paradox` reads it). It must refuse a
/// Moore reading: there is no sign triple to read.
#[derive(Debug, PartialEq, Eq)]
struct ReadingMismatch;

fn simpson_pattern(op: Operand) -> Result<bool, ReadingMismatch> {
    match op.reading {
        DirectionReading::SignTriple => {
            let d = op.edge.direction();
            Ok(d & 0b001 != 0 && d & 0b100 != 0 && d & 0b010 == 0)
        }
        DirectionReading::Moore { .. } => Err(ReadingMismatch),
    }
}

// ─── Tenant ─────────────────────────────────────────────────────────────

/// Representation mutations, used to prove each projection choice is
/// load-bearing. `None` is the probe's representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    None,
    /// Store only 3 of the 4 energy bits.
    Energy3,
    /// Drop plasticity bit 2.
    Plasticity2,
    /// Do not store Epi5.
    DropEpi5,
    /// Restore a different witness than the tenant's.
    WrongWitness,
    /// Write polarity over plasticity bit 0 instead of beside it.
    PolarityIntoPlasticity,
}

const MUTATIONS: [Mutation; 5] = [
    Mutation::Energy3,
    Mutation::Plasticity2,
    Mutation::DropEpi5,
    Mutation::WrongWitness,
    Mutation::PolarityIntoPlasticity,
];

struct MooreTenant {
    witness: u8,
    payload: [u64; 8], // SPOFC40 per lane
    lanes: [MooreNars16; 8],
    mutation: Mutation,
}

impl MooreTenant {
    fn project(
        edges: [CausalEdge64; 8],
        polarity: [bool; 8],
        mutation: Mutation,
    ) -> Result<Self, Refusal> {
        let witness = edges[0].w_slot();
        let mut payload = [0u64; 8];
        let mut lanes = [MooreNars16(0); 8];
        for (k, e) in edges.iter().enumerate() {
            if e.w_slot() != witness {
                return Err(Refusal::WitnessScope {
                    lane: k,
                    expected: witness,
                    found: e.w_slot(),
                });
            }
            let epi5 = e.epistemic_raw5();
            if EpistemicState5::decode(Epi5Gen::V1, epi5).is_err() {
                return Err(Refusal::ReservedEpi5(epi5));
            }
            let mut energy = Field::Energy.get(*e) as u8;
            let mut plast = e.plasticity().bits();
            let mut epi = epi5;
            let mut pol = polarity[k];
            match mutation {
                Mutation::Energy3 => energy &= 0b0111,
                Mutation::Plasticity2 => plast &= 0b011,
                Mutation::DropEpi5 => epi = 0,
                Mutation::PolarityIntoPlasticity => {
                    plast = (plast & !1) | u8::from(pol);
                    pol = false;
                }
                Mutation::None | Mutation::WrongWitness => {}
            }
            payload[k] = e.0 & ((1u64 << 40) - 1);
            lanes[k] = MooreNars16::new(e.causal_mask() as u8, energy, plast, pol, epi);
        }
        Ok(Self {
            witness,
            payload,
            lanes,
            mutation,
        })
    }

    /// The logical operand lane `k` stands for. Direction bits are left 0:
    /// the reading says they carry nothing.
    fn operand(&self, k: usize) -> Operand {
        let lane = self.lanes[k];
        let witness = if self.mutation == Mutation::WrongWitness {
            (self.witness + 1) & 63
        } else {
            self.witness
        };
        let mut e = CausalEdge64(self.payload[k]);
        e = Field::Pearl.set(e, u64::from(lane.pearl()));
        e = Field::Energy.set(e, u64::from(lane.energy()));
        e = Field::Plasticity.set(e, u64::from(lane.plasticity3()));
        e = Field::W.set(e, u64::from(witness));
        e = Field::Epi5.set(e, u64::from(lane.epi5()));
        Operand {
            edge: e,
            reading: DirectionReading::Moore {
                slot: k,
                polarity: lane.polarity(),
            },
        }
    }
}

// ─── The ISA operations under test (shipped code, unchanged) ────────────

fn compose_tables() -> [Box<[u8; 256 * 256]>; 3] {
    let mk = |a: usize, b: usize| {
        let mut t = vec![0u8; 256 * 256].into_boxed_slice();
        for x in 0..256 {
            for y in 0..256 {
                t[x * 256 + y] = (x * a + y * b) as u8;
            }
        }
        t.try_into().expect("256*256")
    };
    [mk(31, 17), mk(7, 13), mk(11, 29)]
}

#[derive(Clone, Copy, Debug)]
enum Op {
    /// `x.forward(y)`: x is the running edge.
    ForwardRunning,
    /// `y.forward(x)`: x is the weight, whose mantissa picks the rule.
    ForwardWeight,
    /// `x.learn(y)`.
    LearnSelf,
    /// `y.learn(x)`: x is the observation.
    LearnObservation,
    /// `x.syllogize(y)` (chain figure: x.o == y.s).
    Syllogize,
}

const OPS: [Op; 5] = [
    Op::ForwardRunning,
    Op::ForwardWeight,
    Op::LearnSelf,
    Op::LearnObservation,
    Op::Syllogize,
];

fn run(op: Op, x: CausalEdge64, y: CausalEdge64, t: &[Box<[u8; 256 * 256]>; 3]) -> CausalEdge64 {
    match op {
        Op::ForwardRunning => x.forward(y, &t[0], &t[1], &t[2]),
        Op::ForwardWeight => y.forward(x, &t[0], &t[1], &t[2]),
        Op::LearnSelf => {
            let mut e = x;
            e.learn(y, 0);
            e
        }
        Op::LearnObservation => {
            let mut e = y;
            e.learn(x, 0);
            e
        }
        Op::Syllogize => {
            x.syllogize(y)
                .expect("fixture builds a chain figure")
                .conclusion
        }
    }
}

// ─── Fixtures ───────────────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A canonical edge with every field random, a VALID Epi5 code (0..24) and
/// the given witness. Direction is a random sign triple.
fn canonical(r: &mut Rng, witness: u8) -> CausalEdge64 {
    let e = CausalEdge64(r.next());
    let e = Field::W.set(e, u64::from(witness));
    Field::Epi5.set(e, r.next() % 24)
}

/// Eight lanes sharing one witness, plus a partner operand per lane whose S
/// equals the lane's O (so `syllogize` takes the chain figure).
fn tenant_fixture(r: &mut Rng) -> ([CausalEdge64; 8], [bool; 8], [CausalEdge64; 8]) {
    let w = (r.next() % 64) as u8;
    let edges: [CausalEdge64; 8] = std::array::from_fn(|_| canonical(r, w));
    let polarity: [bool; 8] = std::array::from_fn(|_| r.next() & 1 == 1);
    let partners: [CausalEdge64; 8] = std::array::from_fn(|k| {
        let pw = (r.next() % 64) as u8;
        let p = canonical(r, pw);
        Field::Spo.set(
            p,
            (Field::Spo.get(p) & !0xFF) | (Field::Spo.get(edges[k]) >> 16 & 0xFF),
        )
    });
    (edges, polarity, partners)
}

/// Count, over `rounds` random tenants, the (op, lane) cases where the
/// tenant operand gives a different observable result from the canonical one.
fn divergences(mutation: Mutation, rounds: usize) -> usize {
    let t = compose_tables();
    let mut r = Rng(0xA11CE);
    let mut diff = 0;
    for _ in 0..rounds {
        let (edges, polarity, partners) = tenant_fixture(&mut r);
        let tenant = MooreTenant::project(edges, polarity, mutation).expect("valid tenant");
        for k in 0..8 {
            let moore = tenant.operand(k).edge;
            for op in OPS {
                let a = run(op, edges[k], partners[k], &t);
                let b = run(op, moore, partners[k], &t);
                if observable(a) != observable(b) {
                    diff += 1;
                }
            }
        }
    }
    diff
}

// ─── A: layout ──────────────────────────────────────────────────────────

#[test]
fn a_lane_is_two_bytes_and_a_tenant_row_is_sixteen() {
    assert_eq!(std::mem::size_of::<MooreNars16>(), 2);
    assert_eq!(std::mem::size_of::<[MooreNars16; 8]>(), 16);
}

// ─── B: every field encoding survives exactly ───────────────────────────

#[test]
fn every_pearl_energy_plasticity_and_valid_epi5_survives() {
    for pearl in 0..8u8 {
        for energy in 0..16u8 {
            for plast in 0..8u8 {
                for pol in [false, true] {
                    for epi in 0..24u8 {
                        let l = MooreNars16::new(pearl, energy, plast, pol, epi);
                        assert_eq!(
                            (l.pearl(), l.energy(), l.plasticity3(), l.polarity(), l.epi5()),
                            (pearl, energy, plast, pol, epi)
                        );
                    }
                }
            }
        }
    }
}

/// Reserved Epi5 codes (certification 6 or 7, raw 24..31) are refused at
/// projection, never truncated into the lane.
#[test]
fn reserved_epi5_is_refused_at_projection() {
    let mut r = Rng(1);
    for raw in 0..32u8 {
        let mut edges: [CausalEdge64; 8] = std::array::from_fn(|_| canonical(&mut r, 5));
        edges[3] = Field::Epi5.set(edges[3], u64::from(raw));
        let got = MooreTenant::project(edges, [false; 8], Mutation::None).err();
        if raw < 24 {
            assert_eq!(got, None, "raw {raw} is a valid code");
        } else {
            assert_eq!(got, Some(Refusal::ReservedEpi5(raw)));
        }
    }
}

/// All 16 raw mantissa nibbles reach `forward` unchanged: the tenant operand
/// picks the same rule and the same result as the canonical one, including
/// the nibbles `from_mantissa` collapses (-8, +3, -2, -3, -4, -5, -7).
#[test]
fn all_sixteen_energy_encodings_drive_forward_identically() {
    let t = compose_tables();
    let mut r = Rng(7);
    for nibble in 0..16u64 {
        let (mut edges, polarity, partners) = tenant_fixture(&mut r);
        for e in &mut edges {
            *e = Field::Energy.set(*e, nibble);
        }
        let tenant = MooreTenant::project(edges, polarity, Mutation::None).unwrap();
        for k in 0..8 {
            let m = tenant.operand(k).edge;
            assert_eq!(Field::Energy.get(m), nibble);
            let a = run(Op::ForwardWeight, edges[k], partners[k], &t);
            let b = run(Op::ForwardWeight, m, partners[k], &t);
            assert_eq!(observable(a), observable(b), "nibble {nibble}");
        }
    }
}

// ─── C: which fields the ISA reads (pinned, two-sided) ──────────────────

/// For an input field, the union of OTHER output fields its value affects,
/// over random operands of every op. A field that affects only itself (or
/// nothing) is carried, not read.
fn reads(field: Field) -> Vec<(String, Field)> {
    let t = compose_tables();
    let mut r = Rng(0xBEEF);
    let mut out = Vec::new();
    for op in OPS {
        for _ in 0..400 {
            let (edges, _, partners) = tenant_fixture(&mut r);
            let (x, y) = (edges[0], partners[0]);
            let bits = (1u64 << field.span().1) - 1;
            let x2 = field.set(x, (field.get(x) + 1 + r.next() % bits) & bits);
            let (a, b) = (run(op, x, y, &t), run(op, x2, y, &t));
            for f in FIELDS {
                if f != field && f.get(a) != f.get(b) {
                    let key = (format!("{op:?}"), f);
                    if !out.contains(&key) {
                        out.push(key);
                    }
                }
            }
        }
    }
    out
}

/// MEASURED: no ISA operation computes anything from Direction3, Witness6 or
/// Epi5. They are carried (or dropped), never read.
#[test]
fn direction_witness_and_epi5_are_never_read_by_an_isa_op() {
    for f in [Field::Direction, Field::W, Field::Epi5] {
        assert_eq!(reads(f), vec![], "{f:?} is read by an ISA op");
    }
}

/// Can-fire arms: the same measurement does detect the fields the ISA reads.
#[test]
fn pearl_energy_and_plasticity_are_read() {
    let energy = reads(Field::Energy);
    assert!(
        energy.contains(&("ForwardWeight".into(), Field::F)),
        "the weight's mantissa picks forward's rule: {energy:?}"
    );
    let plast = reads(Field::Plasticity);
    assert!(
        plast.contains(&("LearnSelf".into(), Field::Spo)),
        "learn gates S/P/O rewrites on plasticity: {plast:?}"
    );
    // Pearl only combines into the output mask (AND); it is carried in the
    // sense above, so this arm checks that the mask itself is combined.
    let t = compose_tables();
    let mut r = Rng(3);
    let (edges, _, partners) = tenant_fixture(&mut r);
    let x = Field::Pearl.set(edges[0], 0b111);
    let y = Field::Pearl.set(partners[0], 0b101);
    assert_eq!(Field::Pearl.get(run(Op::ForwardRunning, x, y, &t)), 0b101);
}

// ─── D: ISA observational equivalence ───────────────────────────────────

/// The central claim: a tenant operand and its canonical edge give the same
/// observable result for every ISA op, in every operand position. Only the
/// canonical sign triple may differ, and only where `forward` copies it.
#[test]
fn tenant_operands_are_isa_equivalent_to_canonical_edges() {
    assert_eq!(divergences(Mutation::None, 300), 0);
}

/// Each representation choice is load-bearing: weakening any one of them
/// produces observable divergence somewhere in the ISA.
#[test]
fn every_representation_choice_is_load_bearing() {
    for m in MUTATIONS {
        let d = divergences(m, 300);
        assert!(d > 0, "{m:?} went unnoticed");
    }
}

/// The equivalence is not vacuous on Direction: the canonical edges carry
/// non-zero sign triples, and `forward` does copy the weight's triple, so the
/// raw words genuinely differ where the observable results agree.
#[test]
fn the_equivalence_is_observational_not_bitwise() {
    let t = compose_tables();
    let mut r = Rng(0xD1);
    let mut raw_differs = 0;
    for _ in 0..100 {
        let (edges, polarity, partners) = tenant_fixture(&mut r);
        let tenant = MooreTenant::project(edges, polarity, Mutation::None).unwrap();
        for k in 0..8 {
            let a = run(Op::ForwardWeight, edges[k], partners[k], &t);
            let b = run(Op::ForwardWeight, tenant.operand(k).edge, partners[k], &t);
            assert_eq!(observable(a), observable(b));
            raw_differs += usize::from(a.0 != b.0);
        }
    }
    assert!(raw_differs > 600, "only {raw_differs}/800 differ in raw bits");
}

// ─── E: Direction, three ways ───────────────────────────────────────────

/// 1. Slot geometry: the eight lanes denote eight distinct neighbours, and
/// reading a lane through a rotated slot table changes its direction.
#[test]
fn slot_rotation_changes_the_geometric_direction() {
    let lane = MooreNars16::new(0, 0, 0, false, 0);
    let rels: Vec<_> = (0..8).map(|k| directed_relation(k, lane)).collect();
    for i in 0..8 {
        for j in 0..8 {
            assert_eq!(rels[i] == rels[j], i == j);
        }
    }
    for k in 0..8 {
        assert_ne!(directed_relation((k + 1) % 8, lane), rels[k]);
    }
}

/// 2. Polarity flips the directed relation (from/to swap) without changing
/// the slot, and without changing plasticity semantics.
#[test]
fn polarity_reverses_the_relation_and_leaves_plasticity_alone() {
    let t = compose_tables();
    let mut r = Rng(0xF1);
    for k in 0..8 {
        for plast in 0..8u8 {
            let out = MooreNars16::new(1, 2, plast, false, 3);
            let inb = MooreNars16::new(1, 2, plast, true, 3);
            let (f, to) = directed_relation(k, out);
            assert_eq!(directed_relation(k, inb), (to, f));
            assert_eq!(out.plasticity3(), inb.plasticity3());
        }
    }
    // learn, which branches on plasticity, cannot see polarity.
    for _ in 0..200 {
        let (edges, mut polarity, partners) = tenant_fixture(&mut r);
        let a = MooreTenant::project(edges, polarity, Mutation::None).unwrap();
        polarity.iter_mut().for_each(|p| *p = !*p);
        let b = MooreTenant::project(edges, polarity, Mutation::None).unwrap();
        for k in 0..8 {
            assert_eq!(
                run(Op::LearnSelf, a.operand(k).edge, partners[k], &t).0,
                run(Op::LearnSelf, b.operand(k).edge, partners[k], &t).0
            );
        }
    }
}

/// 3. A sign-triple consumer refuses the Moore reading and still works on the
/// canonical one. Without the reading tag it would read the zeroed bits and
/// silently report "no Simpson pattern".
#[test]
fn sign_triple_consumers_refuse_the_moore_reading() {
    let mut r = Rng(0x51);
    let (mut edges, polarity, _) = tenant_fixture(&mut r);
    edges[2].set_direction(0b101); // S and O pathological, P not: Simpson
    let canonical_op = Operand {
        edge: edges[2],
        reading: DirectionReading::SignTriple,
    };
    assert_eq!(simpson_pattern(canonical_op), Ok(true));

    let tenant = MooreTenant::project(edges, polarity, Mutation::None).unwrap();
    let moore = tenant.operand(2);
    assert_eq!(simpson_pattern(moore), Err(ReadingMismatch));
    // What an untagged read would have said: the triple is gone.
    let untagged = Operand {
        reading: DirectionReading::SignTriple,
        ..moore
    };
    assert_eq!(simpson_pattern(untagged), Ok(false));
}

// ─── F: Witness ownership scope ─────────────────────────────────────────

/// Can two lanes of one tenant need different W at once? The projection
/// says no and refuses; it never picks one W silently.
#[test]
fn a_tenant_with_mixed_witnesses_is_refused() {
    let mut r = Rng(0x77);
    let (mut edges, polarity, _) = tenant_fixture(&mut r);
    let w = edges[0].w_slot();
    edges[5] = Field::W.set(edges[5], u64::from((w + 9) & 63));
    assert_eq!(
        MooreTenant::project(edges, polarity, Mutation::None).err(),
        Some(Refusal::WitnessScope {
            lane: 5,
            expected: w,
            found: (w + 9) & 63
        })
    );
}

/// Silent arm: a uniform witness is accepted and restored on every lane.
#[test]
fn a_uniform_witness_is_restored_on_every_lane() {
    let mut r = Rng(0x78);
    let (edges, polarity, _) = tenant_fixture(&mut r);
    let tenant = MooreTenant::project(edges, polarity, Mutation::None).unwrap();
    for k in 0..8 {
        assert_eq!(tenant.operand(k).edge.w_slot(), edges[0].w_slot());
    }
}
