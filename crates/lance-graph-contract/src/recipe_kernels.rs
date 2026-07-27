//! The 34 reasoning tactics as **34 working Rust implementations** behind one
//! uniform behaviour (`Tactic`) — the "Elixir-like" recipe layer: a common
//! interface + 34 hot-dispatchable units, registry-routed by tactic id.
//!
//! Each `apply` performs the tactic's *characteristic operation* on a shared
//! [`ThoughtCtx`] using OUR substrate markers (CollapseGate SD / free-energy /
//! dissonance / temperature / NARS confidence / rung) — never a ladybug call
//! (charter D0). Metadata (Tier/Mechanism/Bucket/2³) lives in [`crate::recipes`];
//! this module is the executable side.
//!
//! These are deliberately small, deterministic kernels over a lightweight context
//! so all 34 are genuinely runnable and tested today; richer substrate (real
//! fingerprints via cognitive-shader-driver) slots in behind the same trait later.

use crate::recipes::{recipe, Bucket, Recipe};

/// CollapseGate thresholds (Invariant #2): FLOW < 0.15 ≤ HOLD ≤ 0.35 < BLOCK.
pub const SD_FLOW: f32 = 0.15;
pub const SD_BLOCK: f32 = 0.35;
/// Berry-Esseen noise floor at d=16384.
pub const NOISE_FLOOR: f32 = 0.004;

/// The shared cognitive context a recipe reads/transforms (our substrate markers).
#[derive(Debug, Clone)]
pub struct ThoughtCtx {
    /// CollapseGate dispersion = entropy gate.
    pub sd: f32,
    /// Free energy (surprise).
    pub free_energy: f32,
    /// Quorum split magnitude.
    pub dissonance: f32,
    /// Staunen↔Wisdom: 0.0 = cold/exploit … 1.0 = hot/explore.
    pub temperature: f32,
    /// NARS confidence 0..1.
    pub confidence: f32,
    /// Meaning-depth rung 1..=9.
    pub rung: u8,
    /// Candidate scores (for prune / filter / parallel / fuse tactics).
    pub candidates: Vec<f32>,
    /// Beliefs `(topic_id, frequency, confidence)` (for contradiction / revision).
    pub beliefs: Vec<(u32, f32, f32)>,
}

impl ThoughtCtx {
    /// A neutral context with the given candidate scores.
    pub fn new(candidates: Vec<f32>) -> Self {
        Self {
            sd: 0.25,
            free_energy: 0.5,
            dissonance: 0.0,
            temperature: 0.5,
            confidence: 0.5,
            rung: 1,
            candidates,
            beliefs: Vec::new(),
        }
    }
    fn gate_state(&self) -> GateState {
        if self.sd < SD_FLOW {
            GateState::Flow
        } else if self.sd <= SD_BLOCK {
            GateState::Hold
        } else {
            GateState::Block
        }
    }
}

/// CollapseGate state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    Flow,
    Hold,
    Block,
}

/// What a recipe produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// Did the implicit gate let the recipe run?
    pub fired: bool,
    /// One-line description of what it did.
    pub note: &'static str,
    /// Net change applied to `ctx.confidence`.
    pub delta_conf: f32,
}

impl Outcome {
    fn skipped() -> Self {
        Self {
            fired: false,
            note: "gated off",
            delta_conf: 0.0,
        }
    }
    fn done(note: &'static str, delta_conf: f32) -> Self {
        Self {
            fired: true,
            note,
            delta_conf,
        }
    }
}

/// The eight fields of a [`ThoughtCtx`] — the basis of a tactic's input checklist.
///
/// One bit per field; the bit positions are stable (do not reorder — this is an
/// append-only basis per the per-class-bitmask discipline, cognitive-risc-classes N3).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThoughtField {
    /// `ctx.sd` — CollapseGate dispersion / entropy gate.
    Sd = 0,
    /// `ctx.free_energy` — surprise.
    FreeEnergy = 1,
    /// `ctx.dissonance` — quorum split magnitude.
    Dissonance = 2,
    /// `ctx.temperature` — Staunen↔Wisdom explore/exploit knob.
    Temperature = 3,
    /// `ctx.confidence` — NARS confidence (the reliability coefficient).
    Confidence = 4,
    /// `ctx.rung` — meaning-depth rung 1..=9 (the ladder).
    Rung = 5,
    /// `ctx.candidates` — candidate scores.
    Candidates = 6,
    /// `ctx.beliefs` — `(topic, frequency, confidence)` belief set.
    Beliefs = 7,
}

/// A tactic's **input checklist** as a bitmask over [`ThoughtField`] — the latent
/// "what this tactic reads" made explicit data (reliability-checklist-arc M1).
///
/// This is the executable form of `E-TEMPLATE-IS-CHECKLIST-IS-DATOMS`: a tactic's
/// `requires()` mask is its checklist; coverage = `required & known == required`
/// (`E-RELIABILITY-IS-CHECKLIST-COVERAGE`). Zero-dep (a plain `u8`, no `bitflags`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThoughtMask(pub u8);

impl ThoughtMask {
    /// The empty mask (a tactic that reads nothing — should never occur for a real tactic).
    pub const EMPTY: Self = Self(0);

    /// Build a mask from a slice of fields.
    pub const fn of(fields: &[ThoughtField]) -> Self {
        let mut bits = 0u8;
        let mut i = 0;
        while i < fields.len() {
            bits |= 1 << (fields[i] as u8);
            i += 1;
        }
        Self(bits)
    }

    /// Does this mask contain `field`?
    #[inline]
    pub const fn has(self, field: ThoughtField) -> bool {
        self.0 & (1 << (field as u8)) != 0
    }

    /// Number of required fields (the checklist length).
    #[inline]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// Is the checklist empty?
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Coverage test: are all of `self`'s required fields present in `known`?
    /// (`required & known == required`) — the reliability-as-coverage gate.
    #[inline]
    pub const fn covered_by(self, known: ThoughtMask) -> bool {
        self.0 & known.0 == self.0
    }
}

/// The epistemic status of a tactic's **implementation** — machine-readable, so
/// "this one is a placeholder" is a value the registry can filter on rather than
/// a sentence in a doc-comment nobody parses.
///
/// Lives on the [`Tactic`] impl, NOT on the [`Recipe`] catalogue entry. A
/// `Recipe` describes what the tactic *is* (Tier / Mechanism / Bucket / 2³) —
/// stable properties of the concept. Maturity describes what *this code*
/// currently does, and changes the day someone finishes the implementation. The
/// catalogue entry must not have to change when that happens; folding an
/// implementation property into a concept record is the same
/// merged-carrier mistake the effect census exists to catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelMaturity {
    /// Has a real effect on [`ThoughtCtx`]: mutates at least one field, or
    /// returns a non-zero `delta_conf` on at least one branch. Enforced by
    /// `maturity_operational_implies_an_effect` — a tactic that can do neither
    /// CANNOT declare itself `Operational`.
    Operational,
    /// Runs a real, deterministic computation but lands no effect — either it
    /// ignores `ctx` entirely (algebra demonstrations), or it computes a result
    /// and discards it. Honest scaffolding, not production behaviour. Whether a
    /// given `Demonstration` should be wired up or deleted is an open decision,
    /// recorded per-impl; it is NOT resolved by silently giving it an effect.
    Demonstration,
    /// Hardcoded constants standing in for an unimplemented mechanism.
    Stub,
}

impl KernelMaturity {
    /// May this tactic's effects be relied on in a production dispatch?
    /// Only [`Operational`](KernelMaturity::Operational).
    #[inline]
    #[must_use]
    pub const fn is_production(self) -> bool {
        matches!(self, Self::Operational)
    }
}

/// The uniform behaviour every tactic implements (the Elixir-style contract).
pub trait Tactic: Sync {
    /// The catalogue metadata for this tactic.
    fn meta(&self) -> &'static Recipe;

    /// The epistemic status of this implementation — see [`KernelMaturity`].
    ///
    /// NON-defaulted on purpose, exactly like [`requires`](Tactic::requires): a
    /// default of `Operational` would let an unimplemented tactic inherit a
    /// production claim by saying nothing, which is the failure mode this
    /// method exists to close.
    fn maturity(&self) -> KernelMaturity;

    /// The tactic's **output checklist**: which [`ThoughtField`]s its
    /// [`apply`](Tactic::apply) can mutate, on ANY branch.
    ///
    /// **POSSIBLE writes, not guaranteed writes** — the mirror of `requires()`'s
    /// *may-read*. A tactic that writes `Temperature` only under
    /// `GateState::Block` still declares `Temperature`.
    ///
    /// This exists because `delta_conf` is ONE of eight possible effects, and
    /// reading it as the whole effect is wrong: an effect census over the 34
    /// found **15 tactics returning `delta_conf = 0.0` on every branch while
    /// mutating `ThoughtCtx`** — [`Tactic::run`] calls `apply(ctx)` first and
    /// only then adds the delta, so a zero delta says nothing about whether the
    /// context survived unchanged. `Htd` reorders the entire candidate vector
    /// and reports zero.
    fn writes(&self) -> ThoughtMask;

    /// The tactic's **input checklist**: which [`ThoughtField`]s its [`apply`] reads.
    ///
    /// NON-defaulted on purpose — every tactic MUST declare what it consumes, so the
    /// checklist is real data, not a silent empty default (the reliability-checklist-arc
    /// M1 keystone: reliability is a *declared accessor*, not a constructed gate). The
    /// mask must match the fields the tactic's `apply` body actually reads.
    ///
    /// [`apply`]: Tactic::apply
    fn requires(&self) -> ThoughtMask;
    /// Implicit gate — should this recipe fire given the markers? Default: Gate-bucket
    /// recipes fire only when not in FLOW (there is surprise to act on); others always.
    fn gate(&self, ctx: &ThoughtCtx) -> bool {
        match self.meta().bucket {
            Bucket::Gate => ctx.gate_state() != GateState::Flow,
            _ => true,
        }
    }
    /// Perform the tactic's characteristic operation on the context.
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome;
    /// Gate + apply.
    fn run(&self, ctx: &mut ThoughtCtx) -> Outcome {
        if !self.gate(ctx) {
            return Outcome::skipped();
        }
        let out = self.apply(ctx);
        ctx.confidence = (ctx.confidence + out.delta_conf).clamp(0.0, 1.0);
        out
    }
}

// Small numeric helpers (deterministic; no rng — tests must be reproducible).
fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}
fn max_idx(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .fold(0usize, |b, (i, &v)| if v > xs[b] { i } else { b })
}

macro_rules! tactic {
    ($name:ident, $id:expr) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;
        impl $name {
            #[inline]
            fn rec() -> &'static Recipe {
                recipe($id).expect("recipe id present")
            }
        }
    };
}

// ── the 34 ───────────────────────────────────────────────────────────────────

tactic!(Rte, 1);
impl Tactic for Rte {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::FreeEnergy, ThoughtField::Rung])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Rung, ThoughtField::FreeEnergy])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Recursive expansion: deepen the rung while there's surprise; Berry-Esseen-style stop.
        let mut depth = 0;
        let mut fe = ctx.free_energy;
        while fe > NOISE_FLOOR && depth < 9 {
            fe *= 0.5;
            depth += 1;
        }
        ctx.rung = (ctx.rung + depth).min(9);
        ctx.free_energy = fe;
        Outcome::done("recursively expanded to convergence", 0.05)
    }
}

tactic!(Htd, 2);
impl Tactic for Htd {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Hierarchical decompose: bipolar split around the mean (CLAM-style).
        let m = mean(&ctx.candidates);
        let (hi, lo): (Vec<f32>, Vec<f32>) = ctx.candidates.iter().partition(|&&v| v >= m);
        ctx.candidates = hi.into_iter().chain(lo).collect(); // grouped sub-chains
        Outcome::done("decomposed into bipolar sub-chains", 0.0)
    }
}

tactic!(Smad, 3);
impl Tactic for Smad {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // 3-agent vote: agreement (low spread) revises confidence up.
        let spread = ctx.candidates.iter().cloned().fold(0.0f32, f32::max)
            - ctx.candidates.iter().cloned().fold(1.0f32, f32::min);
        let agree = spread < 0.3;
        Outcome::done(
            if agree {
                "council converged"
            } else {
                "council split"
            },
            if agree { 0.1 } else { -0.05 },
        )
    }
}

tactic!(Rcr, 4);
impl Tactic for Rcr {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Reverse-causality: walk backward (effect→cause) = reverse the chain.
        ctx.candidates.reverse();
        Outcome::done("reverse-traced effect → antecedent (SPO backward S_O)", 0.0)
    }
}

tactic!(Tcp, 5);
impl Tactic for Tcp {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates, ThoughtField::Sd])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Prune low-confidence branches: keep candidates above an SD-derived floor.
        let floor = mean(&ctx.candidates) * (1.0 - ctx.sd);
        let before = ctx.candidates.len();
        ctx.candidates.retain(|&v| v >= floor);
        let _ = before;
        Outcome::done("pruned low-confidence branches (SD floor)", 0.05)
    }
}

tactic!(Tr, 6);
impl Tactic for Tr {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates, ThoughtField::Temperature])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Thought randomization: deterministic temperature-scaled perturbation above noise floor.
        let amp = (ctx.temperature * 0.1).max(NOISE_FLOOR);
        for (i, c) in ctx.candidates.iter_mut().enumerate() {
            let jitter = if i % 2 == 0 { amp } else { -amp };
            *c = (*c + jitter).clamp(0.0, 1.0);
        }
        Outcome::done("perturbed above noise floor (temperature-scaled)", 0.0)
    }
}

tactic!(Asc, 7);
impl Tactic for Asc {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Confidence])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Adversarial self-critique: negate the top belief; survival = strength, else weaken.
        let survives = ctx.confidence > 0.6;
        Outcome::done(
            if survives {
                "belief survived negation challenge"
            } else {
                "belief failed challenge"
            },
            if survives { 0.05 } else { -0.15 },
        )
    }
}

tactic!(Cas, 8);
impl Tactic for Cas {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Rung])
    }
    /// Demonstration: computes an abstraction level and discards it — no effect on ctx, no confidence delta.
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Demonstration
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Conditional abstraction scaling: pick HDR resolution from rung (coarse→fine).
        let _level = match ctx.rung {
            0..=2 => 1,
            3..=5 => 4,
            6..=7 => 8,
            _ => 32,
        };
        Outcome::done("scaled abstraction to rung-appropriate HDR level", 0.0)
    }
}

tactic!(Irs, 9);
impl Tactic for Irs {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates, ThoughtField::Temperature])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Iterative roleplay: a persona modulation (structurally distinct search kernel).
        for c in ctx.candidates.iter_mut() {
            *c = (*c * (0.5 + ctx.temperature)).clamp(0.0, 1.0);
        }
        Outcome::done("applied persona FieldModulation", 0.0)
    }
}

tactic!(Mcp, 10);
impl Tactic for Mcp {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Confidence, ThoughtField::FreeEnergy])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Meta-cognition: if confident but high free-energy (poorly calibrated), pull confidence down.
        let miscalibrated = ctx.confidence > 0.7 && ctx.free_energy > 0.5;
        Outcome::done(
            if miscalibrated {
                "lowered overconfident estimate (Brier)"
            } else {
                "calibration ok"
            },
            if miscalibrated { -0.2 } else { 0.0 },
        )
    }
}

tactic!(Cr, 11);
impl Tactic for Cr {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Beliefs])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Contradiction: same topic, opposing frequency (one true, one false).
        let mut found = false;
        'outer: for (i, &(t, f, _)) in ctx.beliefs.iter().enumerate() {
            for &(t2, f2, _) in &ctx.beliefs[i + 1..] {
                if t == t2 && (f - f2).abs() > 0.5 {
                    found = true;
                    break 'outer;
                }
            }
        }
        // Contradiction preserved, not resolved → coherence (confidence) drops.
        Outcome::done(
            if found {
                "contradiction detected (preserved)"
            } else {
                "coherent"
            },
            if found { -0.2 } else { 0.0 },
        )
    }
}

tactic!(Tca, 12);
impl Tactic for Tca {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    /// only when `candidates` is non-empty
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Temporal augmentation: lag-shift the series (Granger-style precedence).
        if !ctx.candidates.is_empty() {
            ctx.candidates.rotate_right(1);
        }
        Outcome::done("anchored to temporal precedence (Granger lag)", 0.0)
    }
}

tactic!(Cdt, 13);
impl Tactic for Cdt {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates, ThoughtField::Temperature])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    /// both branches write; the convergent branch only when a max exists
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Convergent↔divergent by temperature: hot spreads, cold collapses to the best.
        if ctx.temperature > 0.5 {
            for (i, c) in ctx.candidates.iter_mut().enumerate() {
                *c = (*c + 0.05 * i as f32 * ctx.temperature).fract();
            }
            Outcome::done("divergent: spread candidates", 0.0)
        } else {
            if let Some(&best) = ctx.candidates.get(max_idx(&ctx.candidates)) {
                ctx.candidates = vec![best];
            }
            Outcome::done("convergent: collapsed to best", 0.05)
        }
    }
}

tactic!(Mct, 14);
impl Tactic for Mct {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Multimodal: unify modalities into one fingerprint (mean as the unified score).
        let unified = mean(&ctx.candidates);
        ctx.candidates = vec![unified];
        Outcome::done(
            "unified modalities → one fingerprint (GrammarTriangle)",
            0.0,
        )
    }
}

tactic!(Lsi, 15);
impl Tactic for Lsi {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    /// Reads `candidates` only — `Sd` is an OUTPUT (see `writes`), not an input; it
    /// was over-declared before the write-mask existed.
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Sd])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Latent introspection: read the distribution (mean/sd) and write sd back.
        let m = mean(&ctx.candidates);
        let var = ctx.candidates.iter().map(|&v| (v - m).powi(2)).sum::<f32>()
            / ctx.candidates.len().max(1) as f32;
        ctx.sd = var.sqrt();
        Outcome::done("introspected cluster distribution (CRP μ/σ)", 0.0)
    }
}

tactic!(Pso, 16);
impl Tactic for Pso {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Scaffold: pre-organize (sort) the reasoning candidates descending.
        ctx.candidates
            .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Outcome::done("scaffolded (ordered) the reasoning steps", 0.0)
    }
}

tactic!(Cdi, 17);
impl Tactic for Cdi {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Beliefs, ThoughtField::Dissonance])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Beliefs, ThoughtField::Dissonance])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Induce dissonance: inject a conflicting belief to force deeper investigation.
        let topic = ctx.beliefs.first().map(|b| b.0).unwrap_or(0);
        ctx.beliefs.push((topic, 0.1, 0.6)); // a low-frequency counter-belief on the same topic
        ctx.dissonance = (ctx.dissonance + 0.3).min(1.0);
        Outcome::done("induced productive dissonance (HOLD)", 0.0)
    }
}

tactic!(Cws, 18);
impl Tactic for Cws {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[
            ThoughtField::Candidates,
            ThoughtField::Confidence,
            ThoughtField::Beliefs,
        ])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    /// only when a max-scoring candidate exists
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Beliefs])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Context persistence: checkpoint the current best into the (persistent) belief set.
        if let Some(&best) = ctx.candidates.get(max_idx(&ctx.candidates)) {
            ctx.beliefs.push((u32::MAX, best, ctx.confidence)); // a memory anchor
        }
        Outcome::done("checkpointed state to persistent memory", 0.0)
    }
}

tactic!(Are, 19);
impl Tactic for Are {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::EMPTY
    }
    /// Demonstration: context-blind ABBA unbind identity; ignores ctx entirely.
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Demonstration
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, _ctx: &mut ThoughtCtx) -> Outcome {
        // Reverse-engineer via exact algebraic inverse: A⊗B⊗B = A (XOR self-inverse).
        let (a, b) = (0xDEADBEEFu32, 0xCAFEBABEu32);
        let recovered = (a ^ b) ^ b;
        debug_assert_eq!(recovered, a);
        Outcome::done("recovered component via ABBA unbind (exact)", 0.0)
    }
}

tactic!(Tcf, 20);
impl Tactic for Tcf {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Cascade filter: N strategies = N perturbed views; keep the agreement (median).
        let mut v = ctx.candidates.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(&med) = v.get(v.len() / 2) {
            ctx.candidates = vec![med];
        }
        Outcome::done("filtered N strategies to their agreement (median)", 0.05)
    }
}

tactic!(Ssr, 21);
impl Tactic for Ssr {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Confidence, ThoughtField::FreeEnergy])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Self-skepticism: challenge intensity scales with (confidence − evidence).
        let intensity = (ctx.confidence - ctx.free_energy.min(1.0)).max(0.0);
        Outcome::done("applied skeptic challenge schedule", -0.1 * intensity)
    }
}

tactic!(Etd, 22);
impl Tactic for Etd {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    /// Demonstration: sorts a CLONE of `candidates` and never writes it back — the
    /// computed decomposition is discarded. Wiring it up is a behaviour change and
    /// needs an explicit decision, not a silent fix.
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Demonstration
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Emergent decomposition: split at the largest gap (natural cluster boundary).
        let mut v = ctx.candidates.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Outcome::done("decomposed at the emergent cluster boundary", 0.0)
    }
}

tactic!(Amp, 23);
impl Tactic for Amp {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::FreeEnergy, ThoughtField::Rung])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    /// only when `free_energy > 0.5`
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Rung])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Adaptive meta: TD-style — raise the rung when free-energy stays high.
        if ctx.free_energy > 0.5 {
            ctx.rung = (ctx.rung + 1).min(9);
        }
        Outcome::done("adapted strategy (rung) to performance", 0.0)
    }
}

tactic!(Zcf, 24);
impl Tactic for Zcf {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::EMPTY
    }
    /// Demonstration: context-blind VSA bind identity; ignores ctx entirely.
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Demonstration
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, _ctx: &mut ThoughtCtx) -> Outcome {
        // Zero-shot fusion: bind(A,B) — valid in both, recoverable.
        let (a, b) = (0x0Au32, 0xB0u32);
        let bound = a ^ b;
        debug_assert_eq!(bound ^ b, a); // recoverable
        Outcome::done("fused two concepts via VSA bind (recoverable)", 0.0)
    }
}

tactic!(Hpm, 25);
impl Tactic for Hpm {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    /// only when `candidates` is non-empty
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Pattern match: nearest candidate to a query target (the substrate sweep).
        let target = 0.5f32;
        if let Some(best) = ctx
            .candidates
            .iter()
            .cloned()
            .min_by(|a, b| (a - target).abs().partial_cmp(&(b - target).abs()).unwrap())
        {
            ctx.candidates = vec![best];
        }
        Outcome::done("matched nearest pattern (cosine sweep)", 0.0)
    }
}

tactic!(Cur, 26);
impl Tactic for Cur {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    /// only while more than one candidate remains (the loop may not run)
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Cascading uncertainty reduction: coarse→fine prune ~half per pass; raise confidence.
        while ctx.candidates.len() > 1 {
            let m = mean(&ctx.candidates);
            ctx.candidates.retain(|&v| v >= m);
            if ctx.candidates.len() == 1 {
                break;
            }
            if ctx.candidates.iter().all(|&v| (v - m).abs() < NOISE_FLOOR) {
                break;
            }
        }
        Outcome::done("reduced uncertainty coarse→fine", 0.1)
    }
}

tactic!(Mpc, 27);
impl Tactic for Mpc {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Multi-perspective compression: bundle = consensus (mean per the bundle op).
        let consensus = mean(&ctx.candidates);
        ctx.candidates = vec![consensus];
        Outcome::done("compressed perspectives to consensus (bundle)", 0.0)
    }
}

tactic!(Ssam, 28);
impl Tactic for Ssam {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Sd])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Analogy A→B, C≈A ⊢ C→B: confidence ∝ source similarity.
        let sim = 1.0 - ctx.sd; // closer cluster ⇒ stronger analogy
        Outcome::done("mapped structural analogy (NARS)", 0.1 * (sim - 0.5))
    }
}

tactic!(Idr, 29);
impl Tactic for Idr {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    /// only when `candidates` is non-empty
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Intent reframe: pick the dominant interpretation (max candidate).
        let i = max_idx(&ctx.candidates);
        if let Some(&v) = ctx.candidates.get(i) {
            ctx.candidates = vec![v];
        }
        Outcome::done("reframed to dominant intent", 0.0)
    }
}

tactic!(Spp, 30);
impl Tactic for Spp {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Shadow-parallel: two independent paths; agreement = structural verification.
        let path_a = mean(&ctx.candidates);
        let path_b = ctx.candidates.iter().cloned().fold(0.0f32, f32::max) * 0.5
            + ctx.candidates.iter().cloned().fold(1.0f32, f32::min) * 0.5;
        let agree = (path_a - path_b).abs() < 0.1;
        Outcome::done(
            if agree {
                "shadow paths agree (verified)"
            } else {
                "shadow paths diverge (HOLD)"
            },
            if agree { 0.1 } else { -0.05 },
        )
    }
}

tactic!(Icr, 31);
impl Tactic for Icr {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::EMPTY
    }
    /// Stub: hardcoded constants; `delta_conf` is literally `x * 0.0`.
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Stub
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    /// **⚠ STUB — this is NOT a Pearl counterfactual.** Labelled honestly per the
    /// falsifiability rule ("a doc-comment claim is not a behaviour").
    ///
    /// What it actually does: XORs three HARDCODED constants, counts the
    /// resulting bit divergence, and contributes exactly zero confidence
    /// (`* 0.0`). It never reads `ctx` — the same output for every input, so no
    /// test over this kernel can fail. Recipe 31 carries the
    /// `RungLevel::Counterfactual` label and the SPO=0b111 (Pearl "IMAGINE")
    /// mask, and the label is the only counterfactual thing about it.
    ///
    /// A real `do(X = x)` must SEVER the mechanisms that normally determine
    /// `X` — parents disconnected, evidence derived from those parents
    /// invalidated, descendants recomputed — while holding the exogenous
    /// background fixed. Overwriting a value while retaining evidence derived
    /// from its old parents is a contradictory MUTATION, not an intervention.
    /// Nothing in this crate severs anything today.
    ///
    /// Tracked: `ISS-PEARL-VOCABULARY-WITHOUT-PEARL-MECHANICS`. Do not cite
    /// this kernel, or recipe 31's rung, as evidence of counterfactual
    /// capability.
    fn apply(&self, _ctx: &mut ThoughtCtx) -> Outcome {
        // Counterfactual: world' = world ⊗ factual ⊗ counterfactual; divergence = popcount.
        let world = 0xF0F0_F0F0u32;
        let (factual, counterfactual) = (0x0000_00FFu32, 0x0000_FF00u32);
        let world_cf = world ^ factual ^ counterfactual; // SPO=0b111 apex
        let divergence = (world ^ world_cf).count_ones();
        Outcome::done(
            "STUB counterfactual (hardcoded constants; contributes no confidence)",
            (divergence as f32 / 32.0) * 0.0, // divergence reported, conf unchanged
        )
    }
}

tactic!(Sdd, 32);
impl Tactic for Sdd {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Candidates])
    }
    /// Demonstration: detects distortion and reports it in the note, but `delta_conf`
    /// is hardcoded 0.0 outside the branch — the detection lands nowhere.
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Demonstration
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Semantic distortion: deviation above the Berry-Esseen noise floor = real distortion.
        let dev = (mean(&ctx.candidates) - 0.5).abs();
        let distorted = dev > NOISE_FLOOR;
        Outcome::done(
            if distorted {
                "distortion above noise floor flagged"
            } else {
                "within noise floor"
            },
            0.0,
        )
    }
}

tactic!(Dtmf, 33);
impl Tactic for Dtmf {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Sd, ThoughtField::Temperature])
    }
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Operational
    }
    /// only when the gate reads BLOCK
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[ThoughtField::Temperature])
    }
    fn apply(&self, ctx: &mut ThoughtCtx) -> Outcome {
        // Meta-frame switch when the current frame is BLOCKed.
        let switched = ctx.gate_state() == GateState::Block;
        if switched {
            ctx.temperature = (ctx.temperature + 0.3).min(1.0); // shift all modulation: try differently
        }
        Outcome::done(
            if switched {
                "switched frame (was BLOCK)"
            } else {
                "frame held"
            },
            0.0,
        )
    }
}

tactic!(Hkf, 34);
impl Tactic for Hkf {
    fn meta(&self) -> &'static Recipe {
        Self::rec()
    }
    fn requires(&self) -> ThoughtMask {
        ThoughtMask::EMPTY
    }
    /// Demonstration: context-blind cross-domain bind identity; ignores ctx entirely.
    fn maturity(&self) -> KernelMaturity {
        KernelMaturity::Demonstration
    }
    fn writes(&self) -> ThoughtMask {
        ThoughtMask::of(&[])
    }
    fn apply(&self, _ctx: &mut ThoughtCtx) -> Outcome {
        // Cross-domain fusion: bind(domain_A, relation, domain_B); reversible/auditable.
        let (da, rel, db) = (0x11u32, 0x22u32, 0x44u32);
        let fused = da ^ rel ^ db;
        debug_assert_eq!(fused ^ rel ^ db, da); // recover domain A
        Outcome::done("fused cross-domain knowledge (reversible bind)", 0.0)
    }
}

// ── registry ──────────────────────────────────────────────────────────────────

macro_rules! kernels {
    ($($id:expr => $ty:ident),+ $(,)?) => {
        /// Dispatch a tactic kernel by id (1..=34).
        pub fn kernel(id: u8) -> Option<&'static dyn Tactic> {
            match id {
                $( $id => Some(&$ty as &dyn Tactic), )+
                _ => None,
            }
        }
        /// All 34 kernels in id order.
        pub fn all_kernels() -> [&'static dyn Tactic; 34] {
            [ $( &$ty as &dyn Tactic ),+ ]
        }
    };
}

kernels! {
    1 => Rte, 2 => Htd, 3 => Smad, 4 => Rcr, 5 => Tcp, 6 => Tr, 7 => Asc, 8 => Cas,
    9 => Irs, 10 => Mcp, 11 => Cr, 12 => Tca, 13 => Cdt, 14 => Mct, 15 => Lsi, 16 => Pso,
    17 => Cdi, 18 => Cws, 19 => Are, 20 => Tcf, 21 => Ssr, 22 => Etd, 23 => Amp, 24 => Zcf,
    25 => Hpm, 26 => Cur, 27 => Mpc, 28 => Ssam, 29 => Idr, 30 => Spp, 31 => Icr, 32 => Sdd,
    33 => Dtmf, 34 => Hkf,
}

/// Effect-census tests: do the declared masks tell the truth?
///
/// A declaration drifts from its implementation exactly like a doc-comment
/// does — that is the failure this whole module's `writes()` mask exists to
/// answer, so the mask itself needs a falsifier. Every test here asks
/// *what input would make this fail?*
#[cfg(test)]
mod effect_census {
    use super::*;

    /// Probe contexts chosen to exercise the CONDITIONAL write branches the
    /// census identified — empty vs populated `candidates`, both sides of the
    /// `temperature > 0.5` split, `free_energy > 0.5`, and all three
    /// `GateState`s. A single default ctx would leave conditional writers
    /// looking inert and quietly pass every test below.
    fn probes() -> Vec<ThoughtCtx> {
        let mut hot = ThoughtCtx::new(vec![0.9, 0.6, 0.3, 0.1]);
        hot.sd = 0.5; // BLOCK — the only gate state Dtmf switches under
        hot.temperature = 0.9; // > 0.5 — Cdt divergent branch
        hot.free_energy = 0.9; // > 0.5 — Amp raises the rung
        hot.rung = 3;
        hot.beliefs = vec![(7, 0.9, 0.8), (7, 0.1, 0.7)]; // same-topic contradiction

        let mut cold = ThoughtCtx::new(vec![0.4, 0.45, 0.5, 0.55]);
        cold.sd = 0.05; // FLOW
        cold.temperature = 0.1; // <= 0.5 — Cdt convergent branch
        cold.free_energy = 0.05; // <= 0.5 — Amp holds
        cold.rung = 8;
        cold.beliefs = vec![(3, 0.6, 0.5)];

        let mut empty = ThoughtCtx::new(vec![]); // every `is_empty` guard bites
        empty.sd = 0.25; // HOLD
        empty.beliefs = vec![];

        let mut single = ThoughtCtx::new(vec![0.5]); // len == 1: Cur's loop never runs
        single.sd = 0.25;
        single.beliefs = vec![(1, 0.5, 0.5)];

        // Overconfident-and-surprised. Added because
        // `maturity_operational_implies_an_effect` FAILED on `Mcp` without it:
        // every probe above inherits `ThoughtCtx::new`'s `confidence = 0.5`, so
        // Mcp's `confidence > 0.7 && free_energy > 0.5` branch was unreachable
        // and Mcp looked inert. The gap was in the fixtures, not the kernel —
        // which is precisely what a can-fire test is for, and it found the hole
        // in the probe matrix before it found one in a kernel.
        let mut overconfident = ThoughtCtx::new(vec![0.8, 0.2]);
        overconfident.confidence = 0.95;
        overconfident.free_energy = 0.9;
        overconfident.sd = 0.4; // BLOCK
        overconfident.temperature = 0.6;
        overconfident.rung = 5;
        overconfident.beliefs = vec![(9, 0.9, 0.9), (9, 0.05, 0.6)];

        vec![hot, cold, empty, single, overconfident]
    }

    /// Which [`ThoughtField`]s differ between two contexts.
    ///
    /// Bit-equality on the floats: "unchanged" means the kernel did not touch
    /// it, so an exact comparison is the right one (no epsilon — an epsilon
    /// here would hide small real writes).
    fn changed_fields(before: &ThoughtCtx, after: &ThoughtCtx) -> ThoughtMask {
        let mut bits = 0u8;
        let mut set = |f: ThoughtField| bits |= 1 << (f as u8);
        if before.sd != after.sd {
            set(ThoughtField::Sd);
        }
        if before.free_energy != after.free_energy {
            set(ThoughtField::FreeEnergy);
        }
        if before.dissonance != after.dissonance {
            set(ThoughtField::Dissonance);
        }
        if before.temperature != after.temperature {
            set(ThoughtField::Temperature);
        }
        if before.confidence != after.confidence {
            set(ThoughtField::Confidence);
        }
        if before.rung != after.rung {
            set(ThoughtField::Rung);
        }
        if before.candidates != after.candidates {
            set(ThoughtField::Candidates);
        }
        if before.beliefs != after.beliefs {
            set(ThoughtField::Beliefs);
        }
        ThoughtMask(bits)
    }

    /// **No kernel may mutate a field it did not declare.**
    ///
    /// Uses `apply` directly, NOT `run`: `run` adds `delta_conf` to
    /// `ctx.confidence` afterwards, which is a separate declared effect and
    /// would otherwise show up here as an undeclared `Confidence` write.
    #[test]
    fn no_kernel_writes_outside_its_declared_mask() {
        for k in all_kernels() {
            let declared = k.writes();
            for probe in probes() {
                let before = probe.clone();
                let mut after = probe;
                let _ = k.apply(&mut after);
                let actual = changed_fields(&before, &after);
                assert!(
                    actual.covered_by(declared),
                    "{} ({}) mutated fields outside writes(): actual={:08b} declared={:08b}",
                    k.meta().code,
                    k.meta().id,
                    actual.0,
                    declared.0
                );
            }
        }
    }

    /// **A declared write must be REACHABLE** — the can-fire half.
    ///
    /// A mask that over-declares is as dishonest as one that under-declares:
    /// it makes a kernel look more effectful than it is, and it is exactly
    /// what `Lsi` was doing on the read side (declaring `Sd` as an input it
    /// never read) before the census.
    #[test]
    fn every_declared_write_actually_happens_on_some_probe() {
        for k in all_kernels() {
            let declared = k.writes();
            if declared.is_empty() {
                continue;
            }
            let mut observed = 0u8;
            for probe in probes() {
                let before = probe.clone();
                let mut after = probe;
                let _ = k.apply(&mut after);
                observed |= changed_fields(&before, &after).0;
            }
            assert_eq!(
                observed & declared.0,
                declared.0,
                "{} ({}) declares writes it never performs: declared={:08b} observed={:08b}",
                k.meta().code,
                k.meta().id,
                declared.0,
                observed
            );
        }
    }

    /// **`Operational` requires an effect.** A kernel that can neither mutate
    /// `ThoughtCtx` nor move confidence is not production behaviour, whatever
    /// its note string claims.
    ///
    /// This is the invariant [`KernelMaturity::Operational`] documents, made
    /// executable — without it, maturity is another unenforced doc-comment.
    #[test]
    fn maturity_operational_implies_an_effect() {
        for k in all_kernels() {
            if k.maturity() != KernelMaturity::Operational {
                continue;
            }
            let has_write = !k.writes().is_empty();
            let moves_confidence = probes().into_iter().any(|mut c| {
                let out = k.apply(&mut c);
                out.delta_conf != 0.0
            });
            assert!(
                has_write || moves_confidence,
                "{} ({}) claims Operational but writes nothing and never moves confidence",
                k.meta().code,
                k.meta().id
            );
        }
    }

    /// The converse: a `Demonstration` or `Stub` must NOT be quietly
    /// effectful. If one starts doing real work, its maturity is stale and
    /// this fails rather than letting an unreviewed effect ride in under a
    /// "not production" label.
    #[test]
    fn non_operational_kernels_land_no_effect() {
        for k in all_kernels() {
            if k.maturity() == KernelMaturity::Operational {
                continue;
            }
            assert!(
                k.writes().is_empty(),
                "{} is {:?} but declares writes",
                k.meta().code,
                k.maturity()
            );
            for probe in probes() {
                let before = probe.clone();
                let mut after = probe;
                let out = k.apply(&mut after);
                assert_eq!(
                    changed_fields(&before, &after),
                    ThoughtMask::EMPTY,
                    "{} is {:?} but mutated ctx",
                    k.meta().code,
                    k.maturity()
                );
                assert_eq!(
                    out.delta_conf,
                    0.0,
                    "{} is {:?} but moved confidence",
                    k.meta().code,
                    k.maturity()
                );
            }
        }
    }

    /// The four context-blind kernels return the SAME outcome for radically
    /// different inputs — the honest test for an algebra demonstration.
    ///
    /// Note this is deliberately NOT a can-fire / can-stay-silent pair: those
    /// belong to detectors and thresholded gates. A kernel that ignores its
    /// argument has no input condition to fire on, so the meaningful property
    /// is *invariance*, and asserting anything else would be theatre.
    #[test]
    fn context_blind_kernels_are_input_invariant() {
        const BLIND: [u8; 4] = [19, 24, 31, 34]; // Are, Zcf, Icr, Hkf
        for id in BLIND {
            let k = kernel(id).expect("id in range");
            let outs: Vec<Outcome> = probes().into_iter().map(|mut c| k.apply(&mut c)).collect();
            for o in &outs {
                assert_eq!(
                    o,
                    &outs[0],
                    "{} is context-blind and must not vary with ctx",
                    k.meta().code
                );
            }
            assert_ne!(
                k.maturity(),
                KernelMaturity::Operational,
                "{} ignores ctx entirely and cannot be Operational",
                k.meta().code
            );
        }
    }

    /// The maturity split is non-trivial in BOTH directions — neither label is
    /// vacuous. A classification that applied to everything (or nothing) would
    /// carry exactly as much information as no classification at all.
    #[test]
    fn maturity_discriminates_and_is_not_all_one_label() {
        let ks = all_kernels();
        let operational = ks
            .iter()
            .filter(|k| k.maturity() == KernelMaturity::Operational)
            .count();
        let demonstration = ks
            .iter()
            .filter(|k| k.maturity() == KernelMaturity::Demonstration)
            .count();
        let stub = ks
            .iter()
            .filter(|k| k.maturity() == KernelMaturity::Stub)
            .count();

        assert_eq!(operational + demonstration + stub, 34);
        assert!(operational > 0 && operational < 34, "not all one label");
        assert!(demonstration > 0, "the demonstrations must stay visible");
        assert_eq!(stub, 1, "Icr is the one self-declared stub");
    }

    /// **The finding this census exists for.** `delta_conf == 0.0` does NOT
    /// mean "no effect": `run` applies the delta only AFTER `apply` has had
    /// full `&mut` access. A substantial set of kernels return zero while
    /// reordering candidates, rewriting beliefs, or raising the rung.
    ///
    /// Pinned as a REGRESSION GUARD, not as a target: if a future refactor
    /// makes zero-delta imply inert, this fails and the reasoning gets
    /// re-examined rather than silently inverted.
    #[test]
    fn zero_delta_does_not_imply_inert() {
        let silent_mutators: Vec<&'static str> = all_kernels()
            .iter()
            .filter(|k| !k.writes().is_empty())
            .filter(|k| {
                probes()
                    .into_iter()
                    .all(|mut c| k.apply(&mut c).delta_conf == 0.0)
            })
            .map(|k| k.meta().code)
            .collect();

        assert!(
            silent_mutators.len() >= 10,
            "expected a substantial silent-mutator set, found {}: {:?}",
            silent_mutators.len(),
            silent_mutators
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ThoughtCtx {
        let mut c = ThoughtCtx::new(vec![0.9, 0.6, 0.3, 0.1]);
        c.beliefs = vec![(7, 0.9, 0.8), (7, 0.1, 0.7)]; // a same-topic contradiction
        c
    }

    #[test]
    fn all_34_kernels_dispatch_and_run() {
        let ks = all_kernels();
        assert_eq!(ks.len(), 34);
        for (i, k) in ks.iter().enumerate() {
            assert_eq!(k.meta().id as usize, i + 1, "kernel order matches id");
            let mut c = ctx();
            let _ = k.run(&mut c); // must not panic; confidence stays in range
            assert!((0.0..=1.0).contains(&c.confidence));
        }
        assert!(kernel(0).is_none() && kernel(35).is_none());
        assert_eq!(kernel(4).unwrap().meta().code, "RCR");
    }

    #[test]
    fn tcp_prunes_low_candidates() {
        let mut c = ThoughtCtx::new(vec![0.9, 0.8, 0.1, 0.05]);
        c.sd = 0.2;
        let out = Tcp.run(&mut c);
        assert!(out.fired);
        assert!(
            c.candidates.iter().all(|&v| v >= 0.1),
            "low branches pruned"
        );
    }

    #[test]
    fn cr_detects_same_topic_contradiction_and_drops_confidence() {
        let mut c = ctx();
        let before = c.confidence;
        let out = Cr.run(&mut c);
        assert_eq!(out.note, "contradiction detected (preserved)");
        assert!(c.confidence < before, "coherence drop on contradiction");
    }

    #[test]
    fn icr_builds_counterfactual_via_xor_self_inverse() {
        let mut c = ThoughtCtx::new(vec![0.5]);
        let out = Icr.run(&mut c);
        assert!(out.fired && out.note.contains("counterfactual"));
    }

    #[test]
    fn gate_bucket_recipes_skip_in_flow() {
        let mut c = ThoughtCtx::new(vec![0.5, 0.5]);
        c.sd = 0.05; // FLOW
                     // TCP is a Gate-bucket recipe → should not fire in FLOW.
        assert!(!Tcp.run(&mut c).fired);
        c.sd = 0.5; // BLOCK
        assert!(Tcp.run(&mut c).fired);
    }

    // ── M1: Tactic::requires() — the checklist-as-data tests (with teeth) ──

    #[test]
    fn thought_mask_ops() {
        let m = ThoughtMask::of(&[ThoughtField::Candidates, ThoughtField::Sd]);
        assert!(m.has(ThoughtField::Candidates) && m.has(ThoughtField::Sd));
        assert!(!m.has(ThoughtField::Beliefs));
        assert_eq!(m.len(), 2);
        assert!(!m.is_empty() && ThoughtMask::EMPTY.is_empty());
        // coverage: required ⊆ known
        let known = ThoughtMask::of(&[
            ThoughtField::Candidates,
            ThoughtField::Sd,
            ThoughtField::Rung,
        ]);
        assert!(m.covered_by(known), "required ⊆ known → covered");
        let partial = ThoughtMask::of(&[ThoughtField::Candidates]); // missing Sd
        assert!(
            !m.covered_by(partial),
            "missing a required field → not covered"
        );
    }

    /// TEETH: every tactic's `requires()` mask must match the fields its `apply`
    /// actually reads — spot-checked on representatives so a wrong/empty mask fails.
    #[test]
    fn requires_matches_apply_reads() {
        // Cr reads beliefs (same-topic contradiction scan).
        assert!(Cr.requires().has(ThoughtField::Beliefs));
        assert!(!Cr.requires().has(ThoughtField::Candidates));
        // Tcp reads candidates + sd (SD-derived prune floor).
        assert!(
            Tcp.requires().has(ThoughtField::Candidates) && Tcp.requires().has(ThoughtField::Sd)
        );
        // Mcp reads confidence + free_energy (Brier miscalibration).
        assert!(
            Mcp.requires().has(ThoughtField::Confidence)
                && Mcp.requires().has(ThoughtField::FreeEnergy)
        );
        // Rte reads free_energy + rung (recursive expansion stop).
        assert!(
            Rte.requires().has(ThoughtField::FreeEnergy) && Rte.requires().has(ThoughtField::Rung)
        );
        // Are/Zcf/Icr/Hkf are constant-only (algebraic) → empty checklist is correct,
        // not a forgotten declaration.
        assert!(Are.requires().is_empty() && Zcf.requires().is_empty());
        assert!(Icr.requires().is_empty() && Hkf.requires().is_empty());
    }

    /// TEETH (anti-theater): the 34 masks must be NON-TRIVIAL and VARIED — this fails
    /// if `requires()` were a silent empty default or lazy copy-paste (all-same). The
    /// council's no-op-test warning, made into a real guard.
    #[test]
    fn requires_masks_are_varied_not_a_constant_stub() {
        let masks: Vec<ThoughtMask> = all_kernels().iter().map(|k| k.requires()).collect();
        assert_eq!(masks.len(), 34);

        // Not all-empty: the vast majority declare real inputs (only the 4 algebraic
        // constant-only tactics are legitimately empty).
        let empty = masks.iter().filter(|m| m.is_empty()).count();
        assert_eq!(
            empty, 4,
            "exactly the 4 constant-only tactics (Are/Zcf/Icr/Hkf) are empty"
        );

        // Varied: many distinct masks (fails the copy-paste/all-same stub).
        let distinct: std::collections::BTreeSet<u8> = masks.iter().map(|m| m.0).collect();
        assert!(
            distinct.len() >= 8,
            "checklists must vary across tactics (got {} distinct masks)",
            distinct.len()
        );

        // Every mask is within the 8-field basis. (`u8` is structurally 8 bits, so
        // the bound is on the populated-field count, not stray high bits.)
        for m in &masks {
            assert!(m.len() <= 8, "mask exceeds the 8-field ThoughtField basis");
        }
    }

    /// The reliability-as-coverage gate in miniature: a tactic is "evaluable" iff its
    /// required checklist is covered by the known fields (the AND-test that will drive
    /// the Rubicon Evaluation→Commit decision once wired). Pure, no plan/commit here.
    #[test]
    fn coverage_gate_required_subset_of_known() {
        // A context where only candidates + sd are "known".
        let known = ThoughtMask::of(&[ThoughtField::Candidates, ThoughtField::Sd]);
        // Tcp(candidates,sd) is covered; Cr(beliefs) is NOT (a known-unknown → Plan).
        assert!(
            Tcp.requires().covered_by(known),
            "Tcp evaluable: required ⊆ known"
        );
        assert!(
            !Cr.requires().covered_by(known),
            "Cr blocked: beliefs is a dark/required-unknown field"
        );
    }
}
