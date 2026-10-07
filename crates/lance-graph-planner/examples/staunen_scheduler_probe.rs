//! **D-CE64-STAUNEN-0 — can surprise schedule the next fold?**
//!
//! The hypothesis under test: an observation's **surprisal**,
//! `I(x) = -log2 p(x)` under the basin's own predictive distribution, can
//! work as an interrupt for thought. A basin that is surprised gets fold
//! budget; the fold reacts (signed activation); evidence revises Epi5 only
//! through the production operators; the basin's **Shannon entropy**
//! `H = -Σ p log2 p` then falls, stays, or rises, and the next fold is again
//! self-selected.
//!
//! The probe extends `self_orchestration_probe` (D-CE64-LOOP-0) from one
//! register to many **basins** (one `CausalEdge64` register and one declared
//! hypothesis set each), run across **decision windows** with a fold budget
//! per window and state persisting across windows. The world changes while
//! it runs:
//!
//! - **Real basins** have a true mediator. Rarely (about 1 in 300 windows)
//!   the world switches it. The old chain is retracted silently, and the
//!   new chain is delivered over the next two windows as observations
//!   (cues).
//! - **Noise basins** receive a random cue about half the time and toggle
//!   a half-bound step. Nothing about them can be learned.
//!
//! # What is real and what is policy
//!
//! - Operators: production `pearl::hydrate`, and `pearl::reason` + `revise`
//!   under SO / PO / SP.
//! - H, surprisal and activation are measured from outcomes. The likelihood
//!   tables (hydration and cue) and every scheduler weight are **policy
//!   pins**.
//! - The scheduler reads each basin's register, belief, carried surprisal,
//!   activation coherence and which folds are fresh. It cannot read the
//!   world's truth: that lives in a separate `Oracle` that only the events
//!   and the report touch.
//! - There is no phase variable. Regimes (exploration, convergence,
//!   commitment, reopening, sleep) are classified **afterwards**, from
//!   per-window aggregates, by a function the scheduler never calls.
//!
//! # Bits 50..52
//!
//! The carried surprisal is a transient field, **not** bits 50..52. In
//! production those bits are plasticity, and `CausalEdge64::learn` gates on
//! them; a test below shows that a quantized surprisal written there
//! changes what `learn` does. The probe measures whether a 3-bit carrier
//! would be enough (`Carrier::Q3`, `Carrier::Thermo`), without writing one.
//!
//! "Staunen is the interrupt of thought" is the intuition. This probe
//! measures it; it does not assume it.
//!
//! Run: `cargo run --release -p lance-graph-planner --example staunen_scheduler_probe`
//! Tests (CI): `cargo test -p lance-graph-planner --example staunen_scheduler_probe`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use causal_edge::edge::InferenceType;
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::causal_audit::SupportBasis;
use lance_graph_contract::certification::{CertificationModel, ModelBuilder};
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    Certification3, Epi5Declarations, Epi5Gen, Epi5Reading, EpistemicState5, Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;
use lance_graph_planner::chain_replay::ChainStep;
use lance_graph_planner::pearl::{hydrate, reason, revise, Edit, Evidence, Hydration, Reading};

// ── Allocation counter (measurement only) ─────────────────────────────────

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call to `System` unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
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

// ── Fixture constants ─────────────────────────────────────────────────────

const A: u8 = 10;
const Y: u8 = 30;
const PREDICATE: u8 = 0x91;
const CANDIDATES: [u8; 4] = [20, 99, 77, 55];
const CLASS: ClassId = 0x0902;
const RAIL: RailAxis = RailAxis::Taxonomy;
/// Hypotheses: 0 = direct, `1 + j` = via `CANDIDATES[j]`.
const K: usize = 5;

fn edge(s: u8, o: u8, mask: CausalMask) -> CausalEdge64 {
    CausalEdge64::pack(
        s,
        PREDICATE,
        o,
        200,
        200,
        mask,
        0b101,
        InferenceType::Deduction,
        PlasticityState::S_HOT,
        0,
    )
}

fn step(s: u8, o: u8) -> ChainStep {
    (PREDICATE, edge(s, o, CausalMask::SPO))
}

/// `Related` observationally; with `arms`, executed trial arms earn `Causes`.
fn model(arms: bool) -> CertificationModel {
    let mut b = ModelBuilder::new();
    b.cell(0, true, 5, 4);
    b.cell(0, false, 5, 1);
    b.cell(1, true, 3, 1);
    b.cell(1, false, 3, 2);
    b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
    if arms {
        b.arm(true, 6, 5);
        b.arm(false, 6, 1);
        b.sources(SupportBasis::InterventionBacked, &[10, 11]);
    }
    b.build()
}

struct Shared {
    decl: Epi5Declarations,
    models: [CertificationModel; 2],
}

impl Shared {
    fn new() -> Self {
        let mut decl = Epi5Declarations::new();
        decl.declare(
            CLASS,
            RAIL,
            Epi5Reading {
                generation: Epi5Gen::V1,
            },
        );
        Shared {
            decl,
            models: [model(false), model(true)],
        }
    }

    fn reading(&self) -> Reading<'_> {
        Reading {
            declarations: &self.decl,
            class: CLASS,
            rail: RAIL,
            provenance: EdgeProvenance::V2Stamped,
        }
    }
}

fn epi_raw(raw: u8) -> (Topology2, Certification3) {
    let s = EpistemicState5::decode(Epi5Gen::V1, raw).expect("valid code");
    (s.topology(), s.certification())
}

// ── Folds and likelihoods ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fold {
    Hydrate(u8),
    Observe,
    Intervene,
    Confound,
}

const FOLDS: [Fold; 7] = [
    Fold::Hydrate(0),
    Fold::Hydrate(1),
    Fold::Hydrate(2),
    Fold::Hydrate(3),
    Fold::Observe,
    Fold::Intervene,
    Fold::Confound,
];
const NF: usize = FOLDS.len();

/// Hydration likelihoods. **Policy pins.**
fn l_hydrate(j: usize, h: Hydration) -> [f64; K] {
    let (own, direct, other) = match h {
        Hydration::Hydrated => (0.90, 0.02, 0.05),
        Hydration::Partial => (0.08, 0.18, 0.25),
        Hydration::ProposedOnly => (0.02, 0.80, 0.70),
    };
    core::array::from_fn(|i| {
        if i == 0 {
            direct
        } else if i == 1 + j {
            own
        } else {
            other
        }
    })
}

/// A delivered step mentioning candidate `c`: weak evidence for "via c".
/// **Policy pin.**
fn l_cue(c: usize) -> [f64; K] {
    core::array::from_fn(|i| if i == 1 + c { 0.5 } else { 0.125 })
}

// ── Belief ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Belief {
    Posterior([f64; K]),
    /// No hypothesis survives what was measured. Not a settled belief.
    Contradiction,
}

impl Belief {
    fn uniform() -> Self {
        Belief::Posterior([1.0 / K as f64; K])
    }

    /// `p(outcome) = Σ p_i l_i`; `None` when nothing is predicted.
    fn predictive(self, l: [f64; K]) -> Option<f64> {
        match self {
            Belief::Posterior(p) => Some((0..K).map(|i| p[i] * l[i]).sum()),
            Belief::Contradiction => None,
        }
    }

    fn update(self, l: [f64; K]) -> Self {
        let Belief::Posterior(p) = self else {
            return Belief::Contradiction;
        };
        let z: f64 = (0..K).map(|i| p[i] * l[i]).sum();
        if z <= 0.0 {
            return Belief::Contradiction;
        }
        Belief::Posterior(core::array::from_fn(|i| p[i] * l[i] / z))
    }

    /// Shannon entropy in bits; `None` for a contradiction (never `0`).
    fn entropy(self) -> Option<f64> {
        match self {
            Belief::Posterior(p) => {
                Some(p.iter().filter(|x| **x > 0.0).map(|x| -x * x.log2()).sum())
            }
            Belief::Contradiction => None,
        }
    }

    fn mass(self, i: usize) -> f64 {
        match self {
            Belief::Posterior(p) => p[i],
            Belief::Contradiction => 0.0,
        }
    }

    fn map(self) -> Option<usize> {
        match self {
            Belief::Posterior(p) => (0..K).max_by(|&a, &b| p[a].total_cmp(&p[b])),
            Belief::Contradiction => None,
        }
    }
}

/// Surprisal of an outcome the belief gave probability `p`, in bits.
fn surprisal(p: f64) -> f64 {
    -p.log2()
}

/// Expected information gain of hydrating candidate `j`, in bits.
fn eig_hydrate(b: Belief, j: usize) -> f64 {
    let Some(h0) = b.entropy() else { return 0.0 };
    let mut expected = 0.0;
    for o in [
        Hydration::Hydrated,
        Hydration::Partial,
        Hydration::ProposedOnly,
    ] {
        let l = l_hydrate(j, o);
        let po = b.predictive(l).unwrap_or(0.0);
        if po > 0.0 {
            expected += po * b.update(l).entropy().unwrap_or(0.0);
        }
    }
    h0 - expected
}

// ── The carried surprisal ─────────────────────────────────────────────────

/// How the carried surprisal is stored between cycles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Carrier {
    /// Full precision (f64 bits).
    Full,
    /// 3-bit binary: whole bits, 0..=7.
    Q3,
    /// 3-bit thermometer: 4 levels (0b000, 0b001, 0b011, 0b111) = 0, 2, 4, 6 bits.
    Thermo,
}

impl Carrier {
    fn store(self, s: f64) -> f64 {
        match self {
            Carrier::Full => s,
            Carrier::Q3 => s.round().clamp(0.0, 7.0),
            Carrier::Thermo => 2.0 * (s / 2.0).floor().clamp(0.0, 3.0),
        }
    }
}

/// The 3-bit word a carrier would put in bits 50..52.
#[cfg(test)]
fn carrier_bits(c: Carrier, s: f64) -> u8 {
    match c {
        Carrier::Full | Carrier::Q3 => s.round().clamp(0.0, 7.0) as u8,
        Carrier::Thermo => match (s / 2.0).floor().clamp(0.0, 3.0) as u8 {
            0 => 0b000,
            1 => 0b001,
            2 => 0b011,
            _ => 0b111,
        },
    }
}

// ── Basins ────────────────────────────────────────────────────────────────

/// Everything the scheduler may read about a basin.
#[derive(Clone)]
struct Basin {
    r: CausalEdge64,
    belief: Belief,
    steps: Vec<ChainStep>,
    /// Bumped whenever the sealed chain changes.
    seal: u32,
    /// Per candidate: bumped when a step mentioning it changes. A hydration
    /// is a deterministic read of exactly these steps.
    ver: [u32; 4],
    /// Per candidate: `ver + 1` when its hydration last ran.
    ran_ver: [u32; 4],
    /// The latest hydration outcome per candidate. The belief is built from
    /// these, so a re-measurement **replaces** its channel instead of
    /// multiplying the same evidence in again (no pseudo-replication), and a
    /// retracted chain stops counting.
    last: [Option<Hydration>; 4],
    /// The latest unconfirmed cue, until its candidate is hydrated.
    cue_on: Option<usize>,
    /// Per Pearl fold: bitset of Epi5 codes it has run at.
    ran_epi: [u32; 3],
    /// Carried surprisal, in bits, stored through the carrier.
    surprise: f64,
    /// Coherence of consecutive activations, in [-1, 1].
    coh: f64,
    /// Habitual surprisal (EWMA of what this basin usually sees). Only the
    /// excess above it interrupts.
    habit: f64,
    /// Learning progress: EWMA of the entropy a fold here removed that was
    /// still gone one window later, per fold.
    progress: f64,
    /// Entropy at the end of the previous window.
    h_prev_end: f64,
    prev_act: i8,
    arms: bool,
    contradictions: u32,
}

impl Basin {
    fn new(id: usize, arms: bool) -> Self {
        let r = edge(A, Y, CausalMask::SO)
            .with_w_slot(1 + (id % 62) as u8)
            .with_epistemic_raw5(
                EpistemicState5::new(
                    Epi5Gen::V1,
                    Topology2::IndirectUnknown,
                    Certification3::Open,
                )
                .raw(),
            );
        Basin {
            r,
            belief: Belief::uniform(),
            steps: Vec::with_capacity(8),
            seal: 0,
            ver: [0; 4],
            ran_ver: [0; 4],
            last: [None; 4],
            cue_on: None,
            ran_epi: [0; 3],
            surprise: 0.0,
            coh: 0.0,
            habit: 0.0,
            // Optimistic: an untried basin is assumed to be learnable.
            progress: PROGRESS_PRIOR,
            h_prev_end: (K as f64).log2(),
            prev_act: 0,
            arms,
            contradictions: 0,
        }
    }

    fn fresh(&self, i: usize) -> bool {
        match FOLDS[i] {
            Fold::Hydrate(j) => self.ran_ver[j as usize] != self.ver[j as usize] + 1,
            _ => self.ran_epi[i - 4] & (1 << self.r.epistemic_raw5()) == 0,
        }
    }

    fn certification(&self) -> Certification3 {
        epi_raw(self.r.epistemic_raw5()).1
    }

    /// The sealed chain changed for candidate `c`.
    fn touch(&mut self, c: usize) {
        self.ver[c] += 1;
        self.seal += 1;
    }

    /// The belief from the current evidence, optionally leaving out one
    /// hydration channel and the cue.
    fn evidence(&self, skip: Option<usize>, with_cue: bool) -> Belief {
        let mut b = Belief::uniform();
        for j in 0..4 {
            if let (false, Some(h)) = (skip == Some(j), self.last[j]) {
                b = b.update(l_hydrate(j, h));
            }
        }
        match (with_cue, self.cue_on) {
            (true, Some(c)) => b.update(l_cue(c)),
            _ => b,
        }
    }

    /// Apply a delivered observation: take its surprisal under the current
    /// belief, then hold it as the latest unconfirmed cue.
    fn cue(&mut self, c: usize, carrier: Carrier, w: Weights) -> Option<f64> {
        let s = self.belief.predictive(l_cue(c)).map(surprisal);
        self.cue_on = Some(c);
        self.belief = self.evidence(None, true);
        if let Some(s) = s {
            let excess = habituated(&mut self.habit, s, w);
            self.surprise = carrier.store(self.surprise.max(excess));
        }
        s
    }
}

/// What only the world knows. The scheduler never receives this.
#[derive(Clone, Copy)]
struct Oracle {
    truth: Option<usize>,
    noise: bool,
    /// Window of the last switch, until the basin re-resolves.
    switched_at: Option<u64>,
}

fn resolved(b: &Basin, o: &Oracle) -> bool {
    o.truth.is_some() && b.belief.map() == o.truth && b.belief.entropy().is_some_and(|h| h <= 0.5)
}

// ── Scheduling ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
struct Weights {
    /// Expected information gain (entropy feedback).
    entropy: f64,
    /// Carried surprisal.
    surprise: f64,
    /// Activation coherence (continuation / inhibition).
    activation: f64,
    /// Subtract the habitual surprisal (1) or use raw surprisal (0).
    habituate: f64,
    /// Weight expected gain by learning progress (1) or not (0).
    progress: f64,
    /// Below this priority the scheduler sleeps instead of folding.
    sleep: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Policy {
    Score(Weights),
    RoundRobin,
    Random(u64),
}

const FULL: Weights = Weights {
    entropy: 1.0,
    surprise: 1.0,
    activation: 1.0,
    habituate: 1.0,
    progress: 1.0,
    sleep: 0.05,
};

/// Raw entropy and raw surprisal: no habituation, no learning progress, no
/// sleep. The version a first reading of "surprise creates work" suggests.
const NAIVE: Weights = Weights {
    habituate: 0.0,
    progress: 0.0,
    sleep: 0.0,
    ..FULL
};

/// The optimistic learning-progress prior. **Pin.**
const PROGRESS_PRIOR: f64 = 0.25;
/// Per-window relaxation of unsampled progress toward the prior. **Pin.**
const PROGRESS_RELAX: f64 = 0.01;

/// The entropy weight learning progress puts on expected gain. **Pin.**
fn progress_gain(b: &Basin, w: Weights) -> f64 {
    if w.progress == 0.0 {
        1.0
    } else {
        (4.0 * b.progress).clamp(0.05, 1.0)
    }
}

/// Fold the habit and return the excess to carry.
fn habituated(habit: &mut f64, s: f64, w: Weights) -> f64 {
    let excess = (s - w.habituate * *habit).max(0.0);
    *habit = 0.8 * *habit + 0.2 * s;
    excess
}

/// Certification headroom of a Pearl fold, 0..=1; zero for hydration.
fn headroom(b: &Basin, f: Fold) -> f64 {
    let c = b.certification().code();
    let top = match f {
        Fold::Observe => Certification3::CausalCandidate.code(),
        Fold::Intervene => Certification3::Causes.code(),
        _ => return 0.0,
    };
    f64::from(top.saturating_sub(c)) / 5.0
}

/// A basin's priority and its best fresh fold. `None` when nothing is worth
/// running there.
fn priority(b: &Basin, w: Weights) -> Option<(f64, usize)> {
    let h_max = (K as f64).log2();
    let h = b.belief.entropy()?;
    let mut best: Option<(f64, usize)> = None;
    for (i, &f) in FOLDS.iter().enumerate() {
        if !b.fresh(i) {
            continue;
        }

        let score = match f {
            Fold::Hydrate(j) if w.entropy > 0.0 => {
                w.entropy * eig_hydrate(b.belief, j as usize) * progress_gain(b, w)
            }
            // Without entropy feedback a hydration is worth a fixed probe.
            Fold::Hydrate(_) => 0.05,
            _ if w.entropy > 0.0 => (1.0 - h / h_max) * headroom(b, f),
            _ => headroom(b, f),
        };
        if best.is_none_or(|(s, _)| score > s + 1e-12) {
            best = Some((score, i));
        }
    }
    let (score, i) = best?;
    let p = (score + w.surprise * b.surprise / 8.0) * (1.0 + 0.75 * w.activation * b.coh).max(0.0);
    (p > w.sleep.max(1e-9)).then_some((p, i))
}

// ── One fold ──────────────────────────────────────────────────────────────

/// One compact trace record.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Step {
    cycle: u64,
    basin: u16,
    fold: Fold,
    h_before: Option<f64>,
    h_after: Option<f64>,
    surprisal: Option<f64>,
    activation: i8,
    question: u8,
    orientation: u8,
    fc_before: (u8, u8),
    fc_after: (u8, u8),
    w: u8,
    epi_before: u8,
    epi_after: u8,
    result: u8,
}

fn quantize(x: f64) -> i8 {
    (x * 7.0).round().clamp(-7.0, 7.0) as i8
}

fn fc(e: CausalEdge64) -> (u8, u8) {
    let f = (e.frequency() * 255.0).round() as u8;
    let c = (e.confidence() * 255.0).round() as u8;
    (f, c)
}

fn with_mask(mut e: CausalEdge64, m: CausalMask) -> CausalEdge64 {
    e.set_causal_mask(m);
    e
}

fn fold(
    sh: &Shared,
    b: &mut Basin,
    id: usize,
    i: usize,
    cycle: u64,
    carrier: Carrier,
    w: Weights,
) -> Step {
    let f = FOLDS[i];
    let h_before = b.belief.entropy();
    let epi_before = b.r.epistemic_raw5();
    let fc_before = fc(b.r);
    let rd = sh.reading();
    let mut surprise = None;
    let (activation, result) = match f {
        Fold::Hydrate(j) => {
            let j = j as usize;
            let (n, h) =
                hydrate(b.r, (A, CANDIDATES[j], Y), &b.steps, rd).expect("declared reading");
            b.r = n;
            // Predict this channel from everything else held now.
            let rest = b.evidence(Some(j), b.cue_on != Some(j));
            surprise = rest.predictive(l_hydrate(j, h)).map(surprisal);
            let before = b.belief.mass(1 + j);
            b.last[j] = Some(h);
            if b.cue_on == Some(j) {
                b.cue_on = None;
            }
            b.belief = b.evidence(None, true);
            b.ran_ver[j] = b.ver[j] + 1;
            let r = match h {
                Hydration::Hydrated => 0,
                Hydration::Partial => 1,
                Hydration::ProposedOnly => 2,
            };
            (quantize(b.belief.mass(1 + j) - before), r)
        }
        _ => {
            let mask = match f {
                Fold::Observe => CausalMask::SO,
                Fold::Intervene => CausalMask::PO,
                _ => CausalMask::SP,
            };
            let ev = Evidence {
                model: &sh.models[usize::from(b.arms)],
                chain: None,
            };
            let m = reason(with_mask(b.r, mask), &ev, Edit::None);
            b.r = revise(&m, rd).expect("declared reading");
            b.ran_epi[i - 4] |= 1 << epi_before;
            let c0 = epi_raw(epi_before).1.code() as i8;
            let a = match f {
                Fold::Confound => match m.confounded() {
                    Some(true) => -1,
                    Some(false) => 1,
                    None => 0,
                },
                _ => b.certification().code() as i8 - c0,
            };
            (a.clamp(-7, 7), 3 + u8::from(m.ungrounded().is_some()))
        }
    };
    if matches!(b.belief, Belief::Contradiction) {
        b.contradictions += 1;
    }
    // The surprise is absorbed by the fold that read it; a surprising
    // outcome keeps attention here.
    let excess = surprise.map_or(0.0, |s| habituated(&mut b.habit, s, w));
    b.surprise = carrier.store((0.5 * b.surprise).max(excess));
    if matches!(f, Fold::Hydrate(_)) {
        let s = match (activation.signum(), b.prev_act.signum()) {
            (0, _) => -0.5,
            (_, 0) => 0.0,
            (x, y) if x == y => 1.0,
            _ => -1.0,
        };
        b.coh = 0.7 * b.coh + 0.3 * s;
        b.prev_act = activation;
    }
    Step {
        cycle,
        basin: id as u16,
        fold: f,
        h_before,
        h_after: b.belief.entropy(),
        surprisal: surprise,
        activation,
        question: b.r.causal_mask() as u8,
        orientation: b.r.direction(),
        fc_before,
        fc_after: fc(b.r),
        w: b.r.w_slot(),
        epi_before,
        epi_after: b.r.epistemic_raw5(),
        result,
    }
}

// ── The world ─────────────────────────────────────────────────────────────

fn lcg(x: &mut u64) -> u64 {
    *x = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *x >> 33
}

#[derive(Debug, Clone, Copy)]
struct Shape {
    real: usize,
    noise: usize,
    budget: usize,
    windows: u64,
    /// A real basin switches mediator with probability 1/`switch_every`
    /// per window; 0 = never.
    switch_every: u64,
    seed: u64,
}

const SHAPE: Shape = Shape {
    real: 32,
    noise: 8,
    budget: 3,
    windows: 2_000,
    switch_every: 300,
    seed: 0xC0FFEE,
};

struct World {
    basins: Vec<Basin>,
    oracle: Vec<Oracle>,
    /// Pending deliveries: (window, basin, step, cue candidate).
    pending: Vec<(u64, usize, ChainStep, usize)>,
    rng: u64,
}

impl World {
    fn new(s: Shape) -> Self {
        let n = s.real + s.noise;
        let mut basins = Vec::with_capacity(n);
        let mut oracle = Vec::with_capacity(n);
        for id in 0..n {
            let noise = id >= s.real;
            let mut b = Basin::new(id, id % 2 == 0);
            let truth = if noise {
                None
            } else {
                let j = id % 4;
                b.steps.push(step(A, CANDIDATES[j]));
                b.steps.push(step(CANDIDATES[j], Y));
                // A half-bound decoy.
                b.steps.push(step(A, CANDIDATES[(j + 1) % 4]));
                Some(1 + j)
            };
            basins.push(b);
            oracle.push(Oracle {
                truth,
                noise,
                switched_at: None,
            });
        }
        World {
            basins,
            oracle,
            pending: Vec::with_capacity(16),
            rng: s.seed,
        }
    }

    /// Events at the start of window `w`. Returns the cue surprisals.
    fn events(
        &mut self,
        s: Shape,
        w: u64,
        carrier: Carrier,
        wt: Weights,
        cue_surprisal: &mut Vec<f64>,
    ) {
        for id in 0..self.basins.len() {
            let b = &mut self.basins[id];
            let o = &mut self.oracle[id];
            if o.noise {
                if lcg(&mut self.rng).is_multiple_of(2) {
                    let c = (lcg(&mut self.rng) % 4) as usize;
                    let st = step(A, CANDIDATES[c]);
                    if let Some(k) = b.steps.iter().position(|x| *x == st) {
                        b.steps.swap_remove(k);
                    } else {
                        b.steps.push(st);
                    }
                    b.touch(c);
                    if let Some(x) = b.cue(c, carrier, wt) {
                        cue_surprisal.push(x);
                    }
                }
            } else if s.switch_every > 0 && lcg(&mut self.rng).is_multiple_of(s.switch_every) {
                let old = o.truth.expect("real basin") - 1;
                let new = (old + 1 + (lcg(&mut self.rng) % 3) as usize) % 4;
                let c_old = CANDIDATES[old];
                // Silent retraction of the old chain.
                b.steps
                    .retain(|(_, e)| e.s_idx() != c_old && e.o_idx() != c_old);
                b.touch(old);
                o.truth = Some(1 + new);
                o.switched_at = Some(w);
                self.pending
                    .push((w + 1, id, step(A, CANDIDATES[new]), new));
                self.pending
                    .push((w + 2, id, step(CANDIDATES[new], Y), new));
            }
        }
        let mut k = 0;
        while k < self.pending.len() {
            if self.pending[k].0 <= w {
                let (_, id, st, c) = self.pending.swap_remove(k);
                let b = &mut self.basins[id];
                b.steps.push(st);
                b.touch(c);
                if let Some(x) = b.cue(c, carrier, wt) {
                    cue_surprisal.push(x);
                }
            } else {
                k += 1;
            }
        }
    }
}

// ── A run ─────────────────────────────────────────────────────────────────

/// Per-basin, per-window aggregate for the post-hoc regime reading.
#[derive(Clone, Copy, Default)]
struct Agg {
    folds: u32,
    h_start: f64,
    pos: u32,
    neg: u32,
    surprise_start: f64,
    settled_before: bool,
}

/// Regimes, read off afterwards. Never an input to scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Regime {
    Sleep,
    Neglect,
    Reopen,
    Commit,
    Converge,
    Explore,
    Other,
}

const REGIMES: [Regime; 7] = [
    Regime::Sleep,
    Regime::Neglect,
    Regime::Reopen,
    Regime::Commit,
    Regime::Converge,
    Regime::Explore,
    Regime::Other,
];

fn regime(a: &Agg, h_end: f64) -> Regime {
    let coherent = a.pos == 0 || a.neg == 0;
    if a.folds == 0 {
        if h_end <= 0.5 {
            Regime::Sleep
        } else {
            Regime::Neglect
        }
    } else if a.surprise_start >= 2.0 && a.settled_before {
        Regime::Reopen
    } else if h_end <= 0.5 && (a.h_start - h_end).abs() < 0.05 {
        Regime::Commit
    } else if h_end < a.h_start - 0.05 && coherent {
        Regime::Converge
    } else if h_end > 1.0 && !coherent {
        Regime::Explore
    } else {
        Regime::Other
    }
}

const EPOCHS: usize = 10;

#[derive(Debug, Clone, PartialEq, Default)]
struct Report {
    folds: u64,
    slept: u64,
    useful: u64,
    noise_folds: u64,
    settled_folds: u64,
    epi_transitions: u64,
    resolved_windows: u64,
    real_windows: u64,
    switches: u64,
    woken: u64,
    wake_latency: u64,
    contradictions: u64,
    bytes: u64,
    recomputes: u64,
    regimes: [u64; 7],
    /// Per epoch: (folds, useful, noise folds).
    epochs: [(u64, u64, u64); EPOCHS],
    /// Folds that ran with the basin carrying ≥ 2 bits of surprise.
    surprised_folds: u64,
    cue_surprisal_sum: f64,
    cues: u64,
    /// Σ |ΔH| and Σ |activation| over hydrations, and their cross term.
    dh_abs: f64,
    act_abs: f64,
    dh_act: f64,
    hydrations: u64,
}

impl Report {
    fn resolved_fraction(&self) -> f64 {
        self.resolved_windows as f64 / self.real_windows.max(1) as f64
    }
    fn useful_fraction(&self) -> f64 {
        self.useful as f64 / self.folds.max(1) as f64
    }
    fn noise_share(&self) -> f64 {
        self.noise_folds as f64 / self.folds.max(1) as f64
    }
    fn mean_wake(&self) -> f64 {
        self.wake_latency as f64 / self.woken.max(1) as f64
    }
}

/// Bounded trace: the first `cap` steps, then every 4096th.
struct Trace {
    steps: Vec<Step>,
    cap: usize,
}

impl Trace {
    fn push(&mut self, n: u64, s: Step) {
        if self.steps.len() < self.cap
            || (n.is_multiple_of(4096) && self.steps.len() < 2 * self.cap)
        {
            self.steps.push(s);
        }
    }
}

/// Run `s.windows` decision windows. `stop` cancels between windows.
fn run(
    s: Shape,
    p: Policy,
    carrier: Carrier,
    stop: &AtomicBool,
    mut trace: Option<&mut Trace>,
) -> (World, Report) {
    let sh = Shared::new();
    let mut world = World::new(s);
    let n = world.basins.len();
    let mut rep = Report::default();
    let mut cache: Vec<Option<(f64, usize)>> = vec![None; n];
    let mut dirty = vec![true; n];
    let mut agg = vec![Agg::default(); n];
    let mut cue_s = Vec::with_capacity(n);
    let mut rr = 0usize;
    let mut rng = match p {
        Policy::Random(seed) => seed,
        _ => 0,
    };
    // Baselines do not read the weights; they get the full habituation so
    // their cue bookkeeping is the same as the scheduler's.
    let wt = match p {
        Policy::Score(w) => w,
        _ => FULL,
    };
    let mut cycle = 0u64;
    for w in 0..s.windows {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let seals: Vec<u32> = world.basins.iter().map(|b| b.seal).collect();
        cue_s.clear();
        world.events(s, w, carrier, wt, &mut cue_s);
        rep.cues += cue_s.len() as u64;
        rep.cue_surprisal_sum += cue_s.iter().sum::<f64>();
        for id in 0..n {
            let b = &world.basins[id];
            if b.seal != seals[id] {
                dirty[id] = true;
            }
            agg[id] = Agg {
                folds: 0,
                h_start: b.belief.entropy().unwrap_or(f64::INFINITY),
                pos: 0,
                neg: 0,
                surprise_start: b.surprise,
                settled_before: b.belief.entropy().is_some_and(|h| h <= 0.5),
            };
        }
        let epoch = ((w * EPOCHS as u64) / s.windows.max(1)) as usize;
        for spent in 0..s.budget {
            let pick = match p {
                Policy::Score(wt) => {
                    let mut best: Option<(f64, usize, usize)> = None;
                    for id in 0..n {
                        if dirty[id] {
                            cache[id] = priority(&world.basins[id], wt);
                            dirty[id] = false;
                            rep.recomputes += 1;
                        }
                        if let Some((pr, i)) = cache[id] {
                            if best.is_none_or(|(b, _, _)| pr > b + 1e-12) {
                                best = Some((pr, id, i));
                            }
                        }
                    }
                    best.map(|(_, id, i)| (id, i))
                }
                Policy::RoundRobin => (0..n).map(|d| (rr + d) % n).find_map(|id| {
                    (0..NF)
                        .find(|&i| world.basins[id].fresh(i))
                        .map(|i| (id, i))
                }),
                Policy::Random(_) => {
                    let live = world
                        .basins
                        .iter()
                        .filter(|b| (0..NF).any(|i| b.fresh(i)))
                        .count();
                    if live == 0 {
                        None
                    } else {
                        let k = (lcg(&mut rng) as usize) % live;
                        let id = (0..n)
                            .filter(|&id| (0..NF).any(|i| world.basins[id].fresh(i)))
                            .nth(k)
                            .expect("k < live");
                        let b = &world.basins[id];
                        let fr = (0..NF).filter(|&i| b.fresh(i)).count();
                        let m = (lcg(&mut rng) as usize) % fr;
                        Some((id, (0..NF).filter(|&i| b.fresh(i)).nth(m).expect("m < fr")))
                    }
                }
            };
            let Some((id, i)) = pick else {
                rep.slept += (s.budget - spent) as u64;
                break;
            };
            rr = id + 1;
            let settled = resolved(&world.basins[id], &world.oracle[id]);
            let surprised = world.basins[id].surprise >= 2.0;
            let st = fold(&sh, &mut world.basins[id], id, i, cycle, carrier, wt);
            cycle += 1;
            dirty[id] = true;
            rep.folds += 1;
            rep.bytes += 8 + 16 * world.basins[id].steps.len() as u64;
            let useful = st.epi_before != st.epi_after
                || matches!((st.h_before, st.h_after), (Some(a), Some(b)) if b < a - 1e-9);
            rep.useful += u64::from(useful);
            rep.epi_transitions += u64::from(st.epi_before != st.epi_after);
            rep.noise_folds += u64::from(world.oracle[id].noise);
            rep.settled_folds += u64::from(settled && !surprised);
            rep.surprised_folds += u64::from(surprised);
            let e = &mut rep.epochs[epoch.min(EPOCHS - 1)];
            e.0 += 1;
            e.1 += u64::from(useful);
            e.2 += u64::from(world.oracle[id].noise);
            if let (Fold::Hydrate(_), Some(a), Some(b)) = (st.fold, st.h_before, st.h_after) {
                let dh = (a - b).abs();
                let ac = f64::from(st.activation.unsigned_abs());
                rep.dh_abs += dh;
                rep.act_abs += ac;
                rep.dh_act += dh * ac;
                rep.hydrations += 1;
            }
            agg[id].folds += 1;
            agg[id].pos += u32::from(st.activation > 0);
            agg[id].neg += u32::from(st.activation < 0);
            if let Some(t) = trace.as_deref_mut() {
                t.push(cycle, st);
            }
        }
        for id in 0..n {
            let b = &mut world.basins[id];
            let h_end = b.belief.entropy().unwrap_or(f64::INFINITY);
            // Learning progress: the entropy removed since the last window's
            // end, per fold spent here. Noise gives it back; a learnable
            // basin keeps it.
            if agg[id].folds > 0 && h_end.is_finite() {
                let sample = (b.h_prev_end - h_end) / f64::from(agg[id].folds);
                b.progress = 0.7 * b.progress + 0.3 * sample;
                dirty[id] = true;
            } else if b.progress < PROGRESS_PRIOR {
                // Unsampled progress relaxes back toward the optimistic
                // prior: sleep is not permanent. Without this the learned
                // gate is absorbing (measured: resolution decays over long
                // runs). **Pin.**
                b.progress += PROGRESS_RELAX * (PROGRESS_PRIOR - b.progress);
                dirty[id] = true;
            }
            if h_end.is_finite() {
                b.h_prev_end = h_end;
            }
            let b = &world.basins[id];
            rep.regimes[regime(&agg[id], h_end) as usize] += 1;
            let o = &mut world.oracle[id];
            if o.truth.is_some() {
                rep.real_windows += 1;
                let ok = resolved(b, o);
                rep.resolved_windows += u64::from(ok);
                if let (true, Some(t0)) = (ok, o.switched_at) {
                    rep.woken += 1;
                    rep.wake_latency += w - t0;
                    o.switched_at = None;
                }
            }
        }
    }
    rep.switches = world
        .oracle
        .iter()
        .filter(|o| o.switched_at.is_some())
        .count() as u64
        + rep.woken;
    rep.contradictions = world
        .basins
        .iter()
        .map(|b| u64::from(b.contradictions))
        .sum();
    (world, rep)
}

fn policies() -> [(&'static str, Policy); 11] {
    [
        ("full", Policy::Score(FULL)),
        (
            "no-entropy",
            Policy::Score(Weights {
                entropy: 0.0,
                ..FULL
            }),
        ),
        (
            "no-surprise",
            Policy::Score(Weights {
                surprise: 0.0,
                ..FULL
            }),
        ),
        (
            "no-activation",
            Policy::Score(Weights {
                activation: 0.0,
                ..FULL
            }),
        ),
        (
            "no-habituation",
            Policy::Score(Weights {
                habituate: 0.0,
                ..FULL
            }),
        ),
        (
            "no-progress",
            Policy::Score(Weights {
                progress: 0.0,
                ..FULL
            }),
        ),
        ("no-sleep", Policy::Score(Weights { sleep: 0.0, ..FULL })),
        ("naive", Policy::Score(NAIVE)),
        (
            "no-entropy-no-act",
            Policy::Score(Weights {
                entropy: 0.0,
                activation: 0.0,
                ..FULL
            }),
        ),
        ("round-robin", Policy::RoundRobin),
        ("random", Policy::Random(0x5EED)),
    ]
}

// ── Measurement ───────────────────────────────────────────────────────────

fn row(name: &str, r: &Report, ns: f64) {
    println!(
        "  {:<16} {:>7} {:>6} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>5}/{:<5} {:>6.1} {:>8.1} {:>6.0}",
        name,
        r.folds,
        r.slept,
        r.resolved_fraction(),
        r.useful_fraction(),
        r.noise_share(),
        r.settled_folds as f64 / r.folds.max(1) as f64,
        r.woken,
        r.switches,
        r.mean_wake(),
        r.folds as f64 / r.epi_transitions.max(1) as f64,
        ns,
    );
}

fn main() {
    let stop = AtomicBool::new(false);
    println!(
        "D-CE64-STAUNEN-0: {} real + {} noise basins, {} folds/window, {} windows, switch 1/{}",
        SHAPE.real, SHAPE.noise, SHAPE.budget, SHAPE.windows, SHAPE.switch_every
    );
    println!(
        "  {:<16} {:>7} {:>6} {:>7} {:>7} {:>7} {:>7} {:>11} {:>6} {:>8} {:>6}",
        "policy",
        "folds",
        "slept",
        "resolv",
        "useful",
        "noise",
        "settled",
        "woken/sw",
        "wake",
        "fold/epi",
        "ns/f"
    );
    for (name, p) in policies() {
        let t0 = Instant::now();
        let (_, r) = run(SHAPE, p, Carrier::Full, &stop, None);
        let ns = t0.elapsed().as_nanos() as f64 / r.folds.max(1) as f64;
        row(name, &r, ns);
    }
    for (name, c) in [("full/Q3", Carrier::Q3), ("full/thermo", Carrier::Thermo)] {
        let t0 = Instant::now();
        let (_, r) = run(SHAPE, Policy::Score(FULL), c, &stop, None);
        let ns = t0.elapsed().as_nanos() as f64 / r.folds.max(1) as f64;
        row(name, &r, ns);
    }

    println!("\nSleep threshold sweep (energy vs quality), full weights:");
    for sleep in [0.0, 0.02, 0.05, 0.1, 0.2] {
        let (_, r) = run(
            SHAPE,
            Policy::Score(Weights { sleep, ..FULL }),
            Carrier::Full,
            &stop,
            None,
        );
        println!(
            "  sleep {sleep:<4}: {:>5} folds, resolved {:.3}, wake {:.1} windows, noise share {:.3}, \
             resolved-basin-windows per fold {:.1}",
            r.folds,
            r.resolved_fraction(),
            r.mean_wake(),
            r.noise_share(),
            r.resolved_windows as f64 / r.folds.max(1) as f64
        );
    }

    let (_, full) = run(SHAPE, Policy::Score(FULL), Carrier::Full, &stop, None);
    let (_, rr) = run(SHAPE, Policy::RoundRobin, Carrier::Full, &stop, None);
    println!("\nPer epoch (useful fraction / noise share): full | round-robin");
    for k in 0..EPOCHS {
        let f = full.epochs[k];
        let r = rr.epochs[k];
        let frac = |a: u64, b: u64| a as f64 / b.max(1) as f64;
        println!(
            "  epoch {k}: {:.3} / {:.3}  |  {:.3} / {:.3}",
            frac(f.1, f.0),
            frac(f.2, f.0),
            frac(r.1, r.0),
            frac(r.2, r.0)
        );
    }
    println!("\nRegimes (basin-windows, read off afterwards), full policy:");
    for (k, g) in REGIMES.iter().enumerate() {
        println!("  {:<9} {:>8}", format!("{g:?}"), full.regimes[k]);
    }
    let n = full.hydrations.max(1) as f64;
    println!(
        "\nSurprisal vs entropy vs activation (hydrations): mean cue surprisal {:.3} bits over {} cues; \
         E|ΔH| {:.3}, E|act| {:.3}, E|ΔH|·|act| {:.3}",
        full.cue_surprisal_sum / full.cues.max(1) as f64,
        full.cues,
        full.dh_abs / n,
        full.act_abs / n,
        full.dh_act / n
    );
    println!(
        "Energy proxies (full): bytes/fold {:.1}, priority recomputes/fold {:.2}, folds per Epi5 move {:.1}",
        full.bytes as f64 / full.folds.max(1) as f64,
        full.recomputes as f64 / full.folds.max(1) as f64,
        full.folds as f64 / full.epi_transitions.max(1) as f64
    );

    println!("\nLong run (full policy, release): windows -> folds, ns/fold, allocs/fold, resolved");
    for windows in [100u64, 1_000, 10_000, 62_500] {
        let s = Shape { windows, ..SHAPE };
        let a0 = ALLOCS.load(Ordering::Relaxed);
        let t0 = Instant::now();
        let (_, r) = run(s, Policy::Score(FULL), Carrier::Full, &stop, None);
        let ns = t0.elapsed().as_nanos() as f64 / r.folds.max(1) as f64;
        let allocs = ALLOCS.load(Ordering::Relaxed) - a0;
        println!(
            "  {:>6} windows: {:>8} folds, {:>7.1} ns/fold, {:>5.2} allocs/fold, resolved {:.3}, \
             folds per 350 ms {:.0}",
            windows,
            r.folds,
            ns,
            allocs as f64 / r.folds.max(1) as f64,
            r.resolved_fraction(),
            350e6 / ns
        );
    }

    println!("\nTrace sample (full policy, first 12 folds):");
    let mut t = Trace {
        steps: Vec::new(),
        cap: 12,
    };
    run(
        Shape {
            windows: 2,
            ..SHAPE
        },
        Policy::Score(FULL),
        Carrier::Full,
        &stop,
        Some(&mut t),
    );
    for s in &t.steps {
        let (tb, cb) = epi_raw(s.epi_before);
        let (ta, ca) = epi_raw(s.epi_after);
        println!(
            "  c{:>3} b{:>2} {:<11} H {} dH {} I {} act {:+} q{:03b} o{:03b} fc {:?}->{:?} w{} {:?}x{:?}->{:?}x{:?} r{}",
            s.cycle,
            s.basin,
            format!("{:?}", s.fold),
            s.h_after.map_or("contra".into(), |h| format!("{h:.3}")),
            match (s.h_before, s.h_after) {
                (Some(a), Some(b)) => format!("{:+.3}", b - a),
                _ => "-".into(),
            },
            s.surprisal.map_or("-".into(), |x| format!("{x:.2}")),
            s.activation,
            s.question,
            s.orientation,
            s.fc_before,
            s.fc_after,
            s.w,
            tb,
            cb,
            ta,
            ca,
            s.result
        );
    }
}

// ── Falsifiers ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST: Shape = Shape {
        windows: 600,
        ..SHAPE
    };

    fn go(p: Policy) -> Report {
        run(TEST, p, Carrier::Full, &AtomicBool::new(false), None).1
    }

    /// Measurement helper: prints the TEST-shape table (not an assertion).
    #[test]
    #[ignore]
    fn dump_test_shape() {
        for (name, p) in policies() {
            let r = go(p);
            println!(
                "{name:<18} folds {:>5} resolved {:.3} wake {:.1} noise {:.3} settled {:.3} per-fold {:.2} woken {}",
                r.folds,
                r.resolved_fraction(),
                r.mean_wake(),
                r.noise_share(),
                r.settled_folds as f64 / r.folds.max(1) as f64,
                r.resolved_windows as f64 / r.folds.max(1) as f64,
                r.woken
            );
        }
        for c in [Carrier::Q3, Carrier::Thermo] {
            let r = run(TEST, Policy::Score(FULL), c, &AtomicBool::new(false), None).1;
            println!(
                "{c:?} folds {} resolved {:.3} wake {:.1}",
                r.folds,
                r.resolved_fraction(),
                r.mean_wake()
            );
        }
        let calm = Shape {
            switch_every: 0,
            noise: 0,
            ..TEST
        };
        let r = run(
            calm,
            Policy::Score(FULL),
            Carrier::Full,
            &AtomicBool::new(false),
            None,
        )
        .1;
        println!(
            "calm folds {} epochs {:?} resolved {:.3}",
            r.folds,
            r.epochs,
            r.resolved_fraction()
        );
    }

    /// F2: surprisal is not entropy. The same observation is surprising to a
    /// settled basin and barely surprising to an open one, while the open
    /// basin has the higher entropy.
    #[test]
    fn f2_surprisal_is_distinct_from_entropy() {
        let settled = Belief::Posterior([0.01, 0.96, 0.01, 0.01, 0.01]);
        let open = Belief::uniform();
        let cue = l_cue(2);
        let s_settled = surprisal(settled.predictive(cue).unwrap());
        let s_open = surprisal(open.predictive(cue).unwrap());
        assert!(settled.entropy().unwrap() < open.entropy().unwrap());
        assert!(s_settled > s_open + 0.5, "{s_settled} vs {s_open}");
        // log2(popcount) is the uniform special case only: here it is the
        // entropy of the open basin, and wrong for the settled one.
        assert!((open.entropy().unwrap() - (K as f64).log2()).abs() < 1e-12);
        assert!((settled.entropy().unwrap() - (K as f64).log2()).abs() > 1.0);
    }

    /// F3: a contradiction has no entropy and is never resolved.
    #[test]
    fn f3_zero_legal_states_is_a_contradiction_not_certainty() {
        let b = Belief::uniform().update([0.0; K]);
        assert_eq!(b, Belief::Contradiction);
        assert_eq!(b.entropy(), None);
        assert_eq!(b.predictive(l_cue(0)), None);
        let mut basin = Basin::new(0, true);
        basin.belief = b;
        let o = Oracle {
            truth: Some(1),
            noise: false,
            switched_at: None,
        };
        assert!(!resolved(&basin, &o));
        assert_eq!(priority(&basin, FULL), None);
        // Silence twin: a single surviving hypothesis is settled at 0 bits.
        assert_eq!(
            Belief::uniform()
                .update([0.0, 1.0, 0.0, 0.0, 0.0])
                .entropy(),
            Some(0.0)
        );
    }

    /// F1: high entropy and high surprise alone never move Epi5. Feeding a
    /// basin many cues leaves its register untouched; only a fold through
    /// `revise` / `hydrate` changes it.
    #[test]
    fn f1_high_entropy_and_surprise_alone_create_no_truth() {
        let mut b = Basin::new(3, true);
        let r0 = b.r;
        for k in 0..50 {
            b.cue(k % 4, Carrier::Full, FULL);
        }
        assert!(b.belief.entropy().unwrap() > 1.0);
        assert!(b.surprise > 0.0);
        assert_eq!(b.r, r0);
    }

    /// Scheduling never writes the register's mantissa, plasticity or witness.
    #[test]
    fn the_register_keeps_mantissa_plasticity_and_witness() {
        let (w, _) = run(
            TEST,
            Policy::Score(FULL),
            Carrier::Full,
            &AtomicBool::new(false),
            None,
        );
        for (id, b) in w.basins.iter().enumerate() {
            let start = Basin::new(id, id % 2 == 0).r;
            let keep = ((1u64 << 16) - 1) << 43;
            assert_eq!((b.r.0 ^ start.0) & keep, 0, "basin {id}");
            assert_eq!(b.r.w_slot(), 1 + (id % 62) as u8);
        }
    }

    /// Bits 50..52 are plasticity, and `learn` gates on them: a quantized
    /// surprisal written there changes whether `learn` adopts an archetype.
    /// The reading cannot be applied to an edge `learn` touches.
    #[test]
    fn surprisal_in_bits_50_52_would_change_learn() {
        let obs = {
            let mut e = edge(A + 1, Y, CausalMask::SO);
            e.set_confidence(0.95);
            e
        };
        let adopt = |bits: u8| {
            let mut e = edge(A, Y, CausalMask::SO);
            e.set_confidence(0.1);
            e.set_plasticity(PlasticityState::from_bits(bits));
            e.learn(obs, 0);
            e.s_idx() == obs.s_idx()
        };
        // Settled (0 bits of surprise) reads as all planes frozen.
        assert!(!adopt(carrier_bits(Carrier::Q3, 0.3)));
        // 1 bit of surprise is S hot; 2 bits is P hot, S frozen.
        assert!(adopt(carrier_bits(Carrier::Q3, 1.0)));
        assert!(!adopt(carrier_bits(Carrier::Q3, 2.0)));
        // The thermometer reading at least keeps hot-count monotone.
        let hot = |s: f64| PlasticityState::from_bits(carrier_bits(Carrier::Thermo, s)).hot_count();
        assert_eq!([hot(0.0), hot(2.0), hot(4.0), hot(7.0)], [0, 1, 2, 3]);
    }

    fn ablate(f: impl FnOnce(&mut Weights)) -> Report {
        let mut w = FULL;
        f(&mut w);
        go(Policy::Score(w))
    }

    /// F8: the naive reading of "surprise creates work" (raw entropy, raw
    /// surprisal) is captured by noise: it resolves less and spends more on
    /// noise than the habituated scheduler.
    #[test]
    fn f8_naive_surprise_is_captured_by_noise() {
        let full = go(Policy::Score(FULL));
        let naive = go(Policy::Score(NAIVE));
        assert!(
            naive.resolved_fraction() < full.resolved_fraction() - 0.1,
            "{naive:?}"
        );
        assert!(
            naive.noise_share() > full.noise_share(),
            "{naive:?}\n{full:?}"
        );
        assert!(full.noise_share() < go(Policy::RoundRobin).noise_share());
    }

    /// Habituation and learning progress each carry the fix.
    #[test]
    fn habituation_and_progress_each_matter() {
        let full = go(Policy::Score(FULL));
        for r in [ablate(|w| w.habituate = 0.0), ablate(|w| w.progress = 0.0)] {
            assert!(
                r.resolved_fraction() < full.resolved_fraction() - 0.03,
                "{r:?}"
            );
            assert!(r.mean_wake() > 2.0 * full.mean_wake(), "{r:?}");
        }
    }

    /// F5: removing entropy feedback changes scheduling and costs resolution.
    #[test]
    fn f5_entropy_feedback_matters() {
        let full = go(Policy::Score(FULL));
        let r = ablate(|w| w.entropy = 0.0);
        assert!(
            r.resolved_fraction() < full.resolved_fraction() - 0.2,
            "{r:?}"
        );
    }

    /// F6: removing activation feedback removes inhibition: the scheduler
    /// no longer sleeps on non-reacting basins and spends more folds.
    #[test]
    fn f6_activation_feedback_drives_inhibition() {
        let full = go(Policy::Score(FULL));
        let r = ablate(|w| w.activation = 0.0);
        assert!(r.folds > full.folds, "{} vs {}", r.folds, full.folds);
        assert!(r.slept < full.slept);
    }

    /// F7: with both entropy and activation removed, what remains is no
    /// better than the seeded-random baseline (measured: worse).
    #[test]
    fn f7_without_entropy_and_activation_nothing_beats_baseline() {
        let r = ablate(|w| {
            w.entropy = 0.0;
            w.activation = 0.0;
        });
        assert!(r.resolved_fraction() <= go(Policy::Random(0x5EED)).resolved_fraction());
    }

    /// F10: new evidence wakes a settled basin, and surprise is what wakes
    /// it: without the surprise term re-resolution is several times slower.
    #[test]
    fn f10_surprise_wakes_settled_basins() {
        let full = go(Policy::Score(FULL));
        let r = ablate(|w| w.surprise = 0.0);
        assert!(full.woken > 0);
        assert!(
            r.mean_wake() > 4.0 * full.mean_wake(),
            "{} vs {}",
            r.mean_wake(),
            full.mean_wake()
        );
    }

    /// F9: settled basins yield compute. Fewer folds land on resolved,
    /// unsurprised basins than under round-robin; in a calm world (no
    /// switches, no noise) the scheduler stops spending once settled.
    #[test]
    fn f9_settled_basins_yield_compute() {
        let full = go(Policy::Score(FULL));
        let rr = go(Policy::RoundRobin);
        let share = |r: &Report| r.settled_folds as f64 / r.folds.max(1) as f64;
        assert!(
            share(&full) < share(&rr),
            "{} vs {}",
            share(&full),
            share(&rr)
        );
        let calm = Shape {
            switch_every: 0,
            noise: 0,
            ..TEST
        };
        let c = run(
            calm,
            Policy::Score(FULL),
            Carrier::Full,
            &AtomicBool::new(false),
            None,
        )
        .1;
        assert!(c.epochs[0].0 > 0, "it worked at first");
        assert!(c.epochs[2..].iter().all(|e| e.0 == 0), "{:?}", c.epochs);
        // Silence twin: round-robin keeps spending nothing too only because
        // nothing is fresh; it spent more getting there.
        let crr = run(
            calm,
            Policy::RoundRobin,
            Carrier::Full,
            &AtomicBool::new(false),
            None,
        )
        .1;
        assert!(crr.folds >= c.folds);
    }

    /// F11: better directed, not merely more folds: fewer folds and more
    /// resolved basin-windows per fold than round-robin, and faster wake.
    /// The same run pins where it LOSES: round-robin's resolved fraction is
    /// higher on this shape.
    #[test]
    fn f11_more_information_per_fold_than_round_robin() {
        let full = go(Policy::Score(FULL));
        let rr = go(Policy::RoundRobin);
        let per = |r: &Report| r.resolved_windows as f64 / r.folds.max(1) as f64;
        assert!(full.folds < rr.folds);
        assert!(per(&full) > per(&rr), "{} vs {}", per(&full), per(&rr));
        assert!(full.mean_wake() < rr.mean_wake());
        assert!(
            rr.resolved_fraction() > full.resolved_fraction(),
            "the loss is pinned"
        );
    }

    /// A 3-bit carrier is not free: binary 3-bit doubles wake latency, and
    /// the 4-level thermometer (the reading that would keep plasticity's
    /// hot-count monotone) loses the interrupt almost entirely.
    #[test]
    fn a_three_bit_carrier_costs_wake_latency() {
        let full = go(Policy::Score(FULL));
        let q3 = run(
            TEST,
            Policy::Score(FULL),
            Carrier::Q3,
            &AtomicBool::new(false),
            None,
        )
        .1;
        let th = run(
            TEST,
            Policy::Score(FULL),
            Carrier::Thermo,
            &AtomicBool::new(false),
            None,
        )
        .1;
        assert!(
            q3.mean_wake() > 1.5 * full.mean_wake(),
            "{}",
            q3.mean_wake()
        );
        assert!(
            th.mean_wake() > 4.0 * full.mean_wake(),
            "{}",
            th.mean_wake()
        );
    }

    /// F4: activation and entropy are not the same signal: over the run
    /// hydrations with a large |ΔH| and a zero activation both occur.
    #[test]
    fn f4_activation_and_entropy_carry_independent_information() {
        let sh = Shared::new();
        let mut b = Basin::new(0, true);
        b.steps.push(step(A, CANDIDATES[0]));
        b.steps.push(step(CANDIDATES[0], Y));
        // Hydrating a wrong candidate first: entropy falls, while the tested
        // alternative's mass barely moves.
        let s = fold(&sh, &mut b, 0, 1, 0, Carrier::Full, FULL);
        let dh = s.h_before.unwrap() - s.h_after.unwrap();
        assert!(dh > 0.1, "{s:?}");
        assert!(s.activation.unsigned_abs() <= 1, "{s:?}");
        // The true candidate: activation large.
        let s = fold(&sh, &mut b, 0, 0, 1, Carrier::Full, FULL);
        assert!(s.activation >= 4, "{s:?}");
    }

    /// F9 (determinism): same shape, seed, policy and carrier, same report.
    #[test]
    fn replay_is_deterministic() {
        for (_, p) in policies() {
            assert_eq!(go(p), go(p), "{p:?}");
        }
    }

    /// Cancellation between windows: a raised stop runs nothing.
    #[test]
    fn a_raised_stop_cancels() {
        let stop = AtomicBool::new(true);
        let r = run(TEST, Policy::Score(FULL), Carrier::Full, &stop, None).1;
        assert_eq!(r.folds, 0);
    }
}
