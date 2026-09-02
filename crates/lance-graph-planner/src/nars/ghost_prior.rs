//! **Ghost prior** — the lingering-trace family as a per-thought Friston prior
//! (D-TEH-2, harvested from thinking-engine's former `ghosts.rs`; that file is
//! deleted, this module is its only home).
//!
//! A ghost is a trace a completed thought leaves on an atom: it decays
//! asymptotically and pre-weights which atoms the next cascade visits first.
//! The prior's `prediction` is the autocomplete cache; the gap between that
//! prediction and what actually lit up is **free energy** — the surprise that
//! says "update the prior" (Friston) and, for D-HOUSE-4, the anchoring alarm:
//! a prior that says X while the evidence says Y raises free energy, and the
//! prior is what yields, never the evidence.
//!
//! # Family fence (`E-A-GHOST-TRACE-IS-NOT-THE-COUNTERFACTUAL-LANE-1`)
//!
//! This is the LINGERING-TRACE family — `GhostEcho` (Staunen = persistent
//! wonder, Wisdom = harvested knowing, …) carried by `WisdomMarker`. It is NOT
//! the non-authoritative counterfactual rung (the −6 lane,
//! `deposit_counterfactual` / `CounterfactualMailbox`), whose own docs call
//! themselves "ghost-tier" (`TD-GHOST-TIER-NAME-COLLISION-1`). The rung may
//! consume a trace as a starting prior; it never is one, and nothing here reads
//! or writes the −6 lane.
//!
//! # Ownership — per thought, never a singleton
//!
//! A [`GhostPrior`] is owned by the thought (mailbox) that runs the cascade.
//! There is no global field, no `static`, no shared mutable sink; two mailboxes
//! have two priors. That is the V3 ownership rule applied to the source's
//! process-wide `GhostField`.
//!
//! # The floor — an intentional semantic decision, calibrated, not inherited
//!
//! The source and the contract's `WisdomMarker` agree on the decay rate (0.85
//! per cycle) but not on the FLOOR: the source dropped a trace below 0.001 and
//! `prune` deleted it; `WisdomMarker::intensity_at` clamps at 0.1 forever.
//! Porting the field over the marker would raise long-lived bias by up to two
//! orders of magnitude, so this module carries BOTH readings as
//! [`PriorFloor`] and a calibration gate ([`calibration::discrimination`],
//! exercised in the tests) decides the default from the free-energy response
//! on a recurrence fixture: which floor lets a context SHIFT and a RECURRENCE
//! be told apart after the prior has aged. The declared default is
//! [`PriorFloor::DEFAULT`]; the test `calibration_gate_picks_the_default_floor`
//! fails if the other floor discriminates better on the fixture, so the choice
//! is falsifiable rather than asserted.
//!
//! **Result (2026-09-02, first run).** The first declaration was `Trace` and
//! the gate REJECTED it. Discrimination (`fe_shift − fe_recurrence`, 256
//! atoms), Trace vs Marker: no stale patterns, age 0 → 0.0188 vs 0.0188;
//! age 20 → 0.0188 vs 0.0188; 30 stale patterns, age 0 → 0.0002 vs 0.0026;
//! age 20 → **0.0000 vs 0.0188**; age 60 → **0.0000 vs 0.0188**. Under the
//! Trace floor the remembered pattern is pruned once it is older than ~42
//! cycles (0.8·0.85ᵏ < 0.001) and the prior can no longer tell a recurrence
//! from a shift at all; under the Marker floor it survives at 0.1 and the
//! discrimination holds. The price is absolute level: with 30 stale patterns
//! the Marker prior's free energy sits at 0.35 where the Trace prior's sits at
//! 0.07 — a permanent memory is a noisier baseline. D-HOUSE-4's anchoring
//! alarm reads the DIFFERENCE, so the default is **`Marker`** — the contract's
//! semantics, now measured rather than inherited.

use lance_graph_contract::escalation::{GhostEcho, WisdomMarker};

/// How a trace's intensity bottoms out as it ages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorFloor {
    /// The contract reading: `max(FLOOR, i · decay^age)` with
    /// `WisdomMarker::FLOOR` (0.1) — a trace never vanishes and is never
    /// pruned. Persistence-of-identity semantics.
    Marker,
    /// The source reading: a trace below [`GhostPrior::TRACE_FLOOR`] (0.001)
    /// contributes nothing and `prune` removes it. Bias semantics.
    Trace,
}

impl PriorFloor {
    /// The declared default — chosen by the calibration gate (see module doc
    /// and `tests::calibration_gate_picks_the_default_floor`), not inherited
    /// from either source. `Marker` won on every fixture row; `Trace` lost
    /// all discrimination once the remembered pattern aged past its prune
    /// point.
    pub const DEFAULT: PriorFloor = PriorFloor::Marker;
}

/// One lingering trace: an echo on an atom, born at a cycle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trace {
    /// The atom the trace sits on.
    pub atom: u16,
    /// Echo kind + birth intensity (the contract's marker).
    pub marker: WisdomMarker,
    /// Thought cycle at which the trace was laid.
    pub created_at: u32,
}

/// A per-thought Friston prior over atoms — the harvested ghost field.
#[derive(Debug, Clone)]
pub struct GhostPrior {
    traces: Vec<Trace>,
    decay: f32,
    floor: PriorFloor,
    cycle: u32,
}

impl GhostPrior {
    /// Per-cycle decay rate shared with `WisdomMarker::DECAY` (0.85).
    pub const DEFAULT_DECAY: f32 = WisdomMarker::DECAY;
    /// Below this a trace is inert under [`PriorFloor::Trace`] (source: 0.001).
    pub const TRACE_FLOOR: f32 = 0.001;
    /// At most this many atoms are imprinted per thought (source: 10) —
    /// the I-VSA-IDENTITIES bundle-size discipline applied to the prior.
    pub const IMPRINT_CAP: usize = 10;
    /// `summary` reports traces at or above this intensity (source: 0.01).
    pub const SUMMARY_FLOOR: f32 = 0.01;

    /// An empty prior at cycle 0 with the default decay and the given floor.
    pub fn new(floor: PriorFloor) -> Self {
        Self {
            traces: Vec::new(),
            decay: Self::DEFAULT_DECAY,
            floor,
            cycle: 0,
        }
    }

    /// Override the decay rate (a persona's mode may want 0.75 / 0.92).
    /// Clamped to `[0, 1]`: a rate above 1 would be growth, not decay.
    pub fn with_decay(mut self, decay: f32) -> Self {
        self.decay = decay.clamp(0.0, 1.0);
        self
    }

    /// The floor this prior applies.
    pub fn floor(&self) -> PriorFloor {
        self.floor
    }

    /// The decay rate this prior applies.
    pub fn decay(&self) -> f32 {
        self.decay
    }

    /// The current thought cycle.
    pub fn cycle(&self) -> u32 {
        self.cycle
    }

    /// Advance one thought cycle. `imprint` does this itself; call `tick`
    /// for a thought that laid no trace.
    pub fn tick(&mut self) {
        self.cycle = self.cycle.saturating_add(1);
    }

    /// Choose the echo a completed thought leaves, from its cognitive
    /// markers (source rule, unchanged): Staunen above 0.5 wins, then Wisdom
    /// above 0.3, then unresolved dissonance above 0.3 becomes Grief;
    /// otherwise the caller's style-derived `fallback`.
    pub fn echo_for(staunen: f32, wisdom: f32, dissonance: f32, fallback: GhostEcho) -> GhostEcho {
        if staunen > 0.5 {
            GhostEcho::Staunen
        } else if wisdom > 0.3 {
            GhostEcho::Wisdom
        } else if dissonance > 0.3 {
            GhostEcho::Grief
        } else {
            fallback
        }
    }

    /// Lay traces for a completed thought: the first [`Self::IMPRINT_CAP`]
    /// `(atom, amplitude)` pairs become traces of `echo` at
    /// `amplitude.clamp(0, 1)`. Advances the cycle first, so a trace laid now
    /// has age 0 for the next `bias` / `prediction`.
    pub fn imprint(&mut self, resonant_atoms: &[(u16, f32)], echo: GhostEcho) {
        self.tick();
        for &(atom, amplitude) in resonant_atoms.iter().take(Self::IMPRINT_CAP) {
            let mut marker = WisdomMarker::fresh(echo);
            marker.intensity = amplitude.clamp(0.0, 1.0);
            self.traces.push(Trace {
                atom,
                marker,
                created_at: self.cycle,
            });
        }
    }

    /// A trace's live intensity under this prior's floor, or `None` when it
    /// contributes nothing (`Trace` floor only — a `Marker` trace never does).
    fn live_intensity(&self, t: &Trace) -> Option<f32> {
        let age = self.cycle.saturating_sub(t.created_at);
        let decayed = t.marker.intensity * self.decay.powi(age as i32);
        match self.floor {
            PriorFloor::Marker => Some(decayed.max(WisdomMarker::FLOOR)),
            PriorFloor::Trace => (decayed >= Self::TRACE_FLOOR).then_some(decayed),
        }
    }

    /// The prior's pull on one atom: summed live intensity and the echo of
    /// the strongest contributing trace.
    pub fn bias(&self, atom: u16) -> (f32, Option<GhostEcho>) {
        let mut total = 0.0f32;
        let mut dominant = None;
        let mut max_i = 0.0f32;
        for t in self.traces.iter().filter(|t| t.atom == atom) {
            let Some(i) = self.live_intensity(t) else {
                continue;
            };
            total += i;
            if i > max_i {
                max_i = i;
                dominant = Some(t.marker.ghost);
            }
        }
        (total, dominant)
    }

    /// The prediction for the next cascade over `n_atoms` atoms — summed
    /// live intensity per atom, normalised to `[0, 1]` by its maximum
    /// (all-zero when nothing is live).
    pub fn prediction(&self, n_atoms: usize) -> Vec<f32> {
        let mut pred = vec![0.0f32; n_atoms];
        for t in &self.traces {
            let idx = t.atom as usize;
            if idx >= n_atoms {
                continue;
            }
            if let Some(i) = self.live_intensity(t) {
                pred[idx] += i;
            }
        }
        let max = pred.iter().copied().fold(0.0f32, f32::max);
        if max > 0.0 {
            for p in &mut pred {
                *p /= max;
            }
        }
        pred
    }

    /// Free energy between the prediction and the actual activation: the mean
    /// absolute gap per atom (the source's L1 stand-in for a divergence).
    /// Rises when the context shifts away from what the prior expected, falls
    /// on a recurrence — the two-sided property the tests pin.
    pub fn free_energy(&self, actual: &[f32]) -> f32 {
        let pred = self.prediction(actual.len());
        let n = actual.len().max(1) as f32;
        pred.iter()
            .zip(actual)
            .map(|(p, a)| (p - a).abs())
            .sum::<f32>()
            / n
    }

    /// Traces that currently contribute.
    pub fn active_count(&self) -> usize {
        self.traces
            .iter()
            .filter(|t| self.live_intensity(t).is_some())
            .count()
    }

    /// Drop traces that no longer contribute. A no-op under
    /// [`PriorFloor::Marker`] by definition (nothing ever falls below the
    /// floor); returns the number removed.
    pub fn prune(&mut self) -> usize {
        let before = self.traces.len();
        let cycle = self.cycle;
        let decay = self.decay;
        let floor = self.floor;
        self.traces.retain(|t| {
            let age = cycle.saturating_sub(t.created_at);
            let decayed = t.marker.intensity * decay.powi(age as i32);
            match floor {
                PriorFloor::Marker => true,
                PriorFloor::Trace => decayed >= Self::TRACE_FLOOR,
            }
        });
        before - self.traces.len()
    }

    /// Strongest live trace per atom at or above [`Self::SUMMARY_FLOOR`],
    /// strongest first, one entry per atom. The per-atom maximum is taken
    /// BEFORE the strength sort (a sort-then-adjacent-dedup, as the source
    /// did it, lets one atom appear twice whenever another atom's trace sorts
    /// between its two — Codex on #1142).
    pub fn summary(&self) -> Vec<(u16, GhostEcho, f32)> {
        let mut best: std::collections::BTreeMap<u16, (GhostEcho, f32)> =
            std::collections::BTreeMap::new();
        for t in &self.traces {
            let Some(i) = self.live_intensity(t) else {
                continue;
            };
            if i < Self::SUMMARY_FLOOR {
                continue;
            }
            best.entry(t.atom)
                .and_modify(|cur| {
                    if i > cur.1 {
                        *cur = (t.marker.ghost, i);
                    }
                })
                .or_insert((t.marker.ghost, i));
        }
        let mut v: Vec<(u16, GhostEcho, f32)> =
            best.into_iter().map(|(a, (g, i))| (a, g, i)).collect();
        v.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        v
    }
}

/// The calibration gate that decides [`PriorFloor::DEFAULT`] — kept as
/// library code (not test-only) so a probe can re-run it on other fixtures.
pub mod calibration {
    use super::{GhostPrior, PriorFloor};
    use lance_graph_contract::escalation::GhostEcho;

    /// The smallest atom space the fixture can address: pattern A sits on
    /// atoms 10/20/30 and the shift pattern B on 60/61/62. Stale patterns
    /// beyond the space are simply not predicted over (they still consume a
    /// cycle each), so they impose no bound.
    pub const FIXTURE_MIN_ATOMS: usize = 63;

    /// The most stale patterns the fixture can lay: pattern `k` sits on atoms
    /// `100 + 3k .. 100 + 3k + 2`, and the last of those must fit a `u16`
    /// (`100 + 3·21811 + 2 = 65535`). One more would wrap the atom id back
    /// onto a live pattern (CodeRabbit on #1142).
    pub const FIXTURE_MAX_STALE_PATTERNS: usize = 21_812;

    /// The recurrence fixture: pattern A is imprinted once, then
    /// `stale_patterns` OTHER disjoint patterns are imprinted (one per cycle,
    /// the memory of a life lived since), then the prior is aged `age` more
    /// cycles. Returns `(fe_recurrence, fe_shift)`: free energy when A recurs
    /// vs when an unseen pattern B appears. `n_atoms` is the atom space;
    /// `None` when it is smaller than [`FIXTURE_MIN_ATOMS`] (the fixture's
    /// atoms would fall outside it — Codex on #1142) or when `stale_patterns`
    /// exceeds [`FIXTURE_MAX_STALE_PATTERNS`] (a stale atom id would wrap).
    pub fn recurrence_fixture(
        floor: PriorFloor,
        stale_patterns: usize,
        age: u32,
        n_atoms: usize,
    ) -> Option<(f32, f32)> {
        if n_atoms < FIXTURE_MIN_ATOMS || stale_patterns > FIXTURE_MAX_STALE_PATTERNS {
            return None;
        }
        let a: [(u16, f32); 3] = [(10, 1.0), (20, 0.8), (30, 0.6)];
        let mut prior = GhostPrior::new(floor);
        prior.imprint(&a, GhostEcho::Thought);
        for k in 0..stale_patterns {
            // Disjoint from A and from B: atoms 100.. in steps of 3. Checked:
            // the bound above guarantees it, and a wrap would alias a live atom.
            let base = u16::try_from(100 + 3 * k).ok()?;
            prior.imprint(
                &[(base, 0.9), (base + 1, 0.7), (base + 2, 0.5)],
                GhostEcho::Affinity,
            );
        }
        for _ in 0..age {
            prior.tick();
        }
        let mut recur = vec![0.0f32; n_atoms];
        for &(atom, amp) in &a {
            recur[atom as usize] = amp;
        }
        let mut shift = vec![0.0f32; n_atoms];
        shift[60] = 1.0;
        shift[61] = 0.8;
        shift[62] = 0.6;
        Some((prior.free_energy(&recur), prior.free_energy(&shift)))
    }

    /// Discrimination = `fe_shift − fe_recurrence` on the fixture: how far
    /// apart the prior holds "this is familiar" from "this is new" after it
    /// has aged. Higher is better; ≤ 0 means the prior can no longer tell.
    /// `None` under the same conditions as [`recurrence_fixture`].
    pub fn discrimination(
        floor: PriorFloor,
        stale_patterns: usize,
        age: u32,
        n_atoms: usize,
    ) -> Option<f32> {
        let (recur, shift) = recurrence_fixture(floor, stale_patterns, age, n_atoms)?;
        Some(shift - recur)
    }
}

#[cfg(test)]
mod tests {
    use super::calibration::{
        discrimination, recurrence_fixture, FIXTURE_MAX_STALE_PATTERNS, FIXTURE_MIN_ATOMS,
    };
    use super::*;

    const N: usize = 256;

    fn fresh(floor: PriorFloor) -> GhostPrior {
        let mut p = GhostPrior::new(floor);
        p.imprint(&[(42, 0.8)], GhostEcho::Affinity);
        p
    }

    // ── decay ────────────────────────────────────────────────────────────

    #[test]
    fn bias_decays_monotonically_to_the_trace_floor_and_then_goes_inert() {
        let mut p = fresh(PriorFloor::Trace);
        let mut last = p.bias(42).0;
        assert!((last - 0.8).abs() < 1e-6, "age 0 = birth intensity");
        let mut went_inert_at = None;
        for cycle in 1..200u32 {
            p.tick();
            let b = p.bias(42).0;
            assert!(
                b <= last + 1e-7,
                "non-increasing at cycle {cycle}: {b} > {last}"
            );
            if b == 0.0 && went_inert_at.is_none() {
                went_inert_at = Some(cycle);
            }
            last = b;
        }
        let inert = went_inert_at.expect("a Trace-floor trace must go inert within 200 cycles");
        // 0.8 · 0.85^k < 0.001  ⇔  k > ln(0.00125)/ln(0.85) ≈ 41.1 → inert from cycle 42.
        assert_eq!(
            inert, 42,
            "inert cycle follows from the decay constant, not a tuned bound"
        );
        assert_eq!(p.active_count(), 0);
        assert_eq!(p.prune(), 1);
    }

    #[test]
    fn bias_decays_monotonically_to_the_marker_floor_and_never_below() {
        let mut p = fresh(PriorFloor::Marker);
        let mut last = p.bias(42).0;
        for _ in 1..200u32 {
            p.tick();
            let b = p.bias(42).0;
            assert!(b <= last + 1e-7);
            assert!(
                b >= WisdomMarker::FLOOR - 1e-7,
                "never below the marker floor"
            );
            last = b;
        }
        assert!(
            (last - WisdomMarker::FLOOR).abs() < 1e-6,
            "settles exactly on the floor"
        );
        assert_eq!(p.active_count(), 1, "a marker trace never goes inert");
        assert_eq!(p.prune(), 0, "prune is a no-op under the marker floor");
    }

    #[test]
    fn decay_constant_is_load_bearing() {
        // Disable-run in test form: with decay 1.0 nothing decays, so the
        // monotone-to-floor property above would be vacuous. Prove the knob
        // binds in both directions.
        let mut slow = GhostPrior::new(PriorFloor::Trace).with_decay(1.0);
        slow.imprint(&[(42, 0.8)], GhostEcho::Affinity);
        let mut fast = GhostPrior::new(PriorFloor::Trace).with_decay(0.5);
        fast.imprint(&[(42, 0.8)], GhostEcho::Affinity);
        for _ in 0..10 {
            slow.tick();
            fast.tick();
        }
        assert!(
            (slow.bias(42).0 - 0.8).abs() < 1e-6,
            "decay 1.0 holds the birth intensity"
        );
        assert!(
            fast.bias(42).0 < 0.001,
            "decay 0.5 is inert within 10 cycles"
        );
        let mut default = fresh(PriorFloor::Trace);
        for _ in 0..10 {
            default.tick();
        }
        let d = default.bias(42).0;
        assert!(
            d < 0.8 && d > 0.001,
            "the default sits strictly between: {d}"
        );
    }

    // ── free energy, two-sided ───────────────────────────────────────────

    #[test]
    fn free_energy_falls_on_recurrence_and_rises_on_shift_under_both_floors() {
        for floor in [PriorFloor::Trace, PriorFloor::Marker] {
            let mut p = GhostPrior::new(floor);
            p.imprint(&[(10, 1.0), (20, 0.8), (30, 0.6)], GhostEcho::Thought);
            let baseline = p.free_energy(&vec![0.0f32; N]); // nothing lit up at all
            let mut recur = vec![0.0f32; N];
            recur[10] = 1.0;
            recur[20] = 0.8;
            recur[30] = 0.6;
            let mut shift = vec![0.0f32; N];
            shift[60] = 1.0;
            shift[61] = 0.8;
            shift[62] = 0.6;
            let fe_recur = p.free_energy(&recur);
            let fe_shift = p.free_energy(&shift);
            assert!(
                fe_recur < baseline,
                "{floor:?}: recurrence LOWERS surprise below the empty baseline"
            );
            assert!(
                fe_shift > baseline,
                "{floor:?}: a shift RAISES surprise above the empty baseline"
            );
            assert!(fe_recur < fe_shift);
            assert!(
                fe_recur.abs() < 1e-6,
                "{floor:?}: an exact recurrence of a fresh prior is zero surprise"
            );
        }
    }

    #[test]
    fn free_energy_of_an_empty_prior_is_the_mean_activation() {
        let p = GhostPrior::new(PriorFloor::Trace);
        let mut actual = vec![0.0f32; 4];
        actual[0] = 1.0;
        assert!((p.free_energy(&actual) - 0.25).abs() < 1e-6);
        assert_eq!(
            p.free_energy(&[]),
            0.0,
            "no atoms, no surprise, no division by zero"
        );
    }

    // ── the calibration gate ─────────────────────────────────────────────

    #[test]
    fn calibration_gate_picks_the_default_floor() {
        // The fixture the module doc names: A imprinted, 30 stale patterns
        // lived since, then aged 20 more cycles. Print the whole table so a
        // reviewer can see the numbers, then assert the DECLARED default is
        // the floor that discriminates better — anything else is a
        // declaration the fixture contradicts.
        let mut table = String::new();
        let mut default_wins_everywhere = true;
        for &(stale, age) in &[(0usize, 0u32), (0, 20), (30, 0), (30, 20), (30, 60)] {
            let d_trace =
                discrimination(PriorFloor::Trace, stale, age, N).expect("N ≥ FIXTURE_MIN_ATOMS");
            let d_marker =
                discrimination(PriorFloor::Marker, stale, age, N).expect("N ≥ FIXTURE_MIN_ATOMS");
            let (rt, st) = recurrence_fixture(PriorFloor::Trace, stale, age, N)
                .expect("N ≥ FIXTURE_MIN_ATOMS");
            let (rm, sm) = recurrence_fixture(PriorFloor::Marker, stale, age, N)
                .expect("N ≥ FIXTURE_MIN_ATOMS");
            table.push_str(&format!(
                "stale={stale:>2} age={age:>2} | Trace: recur={rt:.4} shift={st:.4} disc={d_trace:.4} | Marker: recur={rm:.4} shift={sm:.4} disc={d_marker:.4}\n"
            ));
            let (d_default, d_other) = match PriorFloor::DEFAULT {
                PriorFloor::Trace => (d_trace, d_marker),
                PriorFloor::Marker => (d_marker, d_trace),
            };
            if d_default + 1e-6 < d_other {
                default_wins_everywhere = false;
            }
        }
        println!("calibration gate (discrimination = fe_shift − fe_recurrence):\n{table}");
        assert!(
            default_wins_everywhere,
            "PriorFloor::DEFAULT = {:?} does not discriminate at least as well as the other floor on every \
             fixture row:\n{table}",
            PriorFloor::DEFAULT
        );
    }

    #[test]
    fn the_two_floors_genuinely_differ_after_ageing() {
        // Anti-vacuity for the gate: if both floors gave the same numbers the
        // gate would pass trivially. After 60 cycles a Trace-floor trace is
        // inert and a Marker-floor trace sits at 0.1 — the predictions differ.
        let (rt, _) =
            recurrence_fixture(PriorFloor::Trace, 30, 60, N).expect("N ≥ FIXTURE_MIN_ATOMS");
        let (rm, _) =
            recurrence_fixture(PriorFloor::Marker, 30, 60, N).expect("N ≥ FIXTURE_MIN_ATOMS");
        assert!(
            (rt - rm).abs() > 1e-3,
            "floors must be distinguishable on the fixture: {rt} vs {rm}"
        );
    }

    // ── shape ────────────────────────────────────────────────────────────

    #[test]
    fn prediction_orders_atoms_by_trace_strength_and_leaves_unseen_atoms_zero() {
        let mut p = GhostPrior::new(PriorFloor::DEFAULT);
        p.imprint(&[(10, 0.9), (20, 0.7), (30, 0.5)], GhostEcho::Epiphany);
        let pred = p.prediction(N);
        assert!((pred[10] - 1.0).abs() < 1e-6, "normalised to the maximum");
        assert!(pred[10] > pred[20] && pred[20] > pred[30]);
        assert_eq!(pred[100], 0.0);
        assert_eq!(p.prediction(0).len(), 0);
    }

    #[test]
    fn imprint_caps_at_ten_atoms_and_clamps_amplitude() {
        let mut p = GhostPrior::new(PriorFloor::DEFAULT);
        let atoms: Vec<(u16, f32)> = (0..15u16).map(|a| (a, 2.0)).collect();
        p.imprint(&atoms, GhostEcho::Somatic);
        assert_eq!(p.active_count(), GhostPrior::IMPRINT_CAP);
        assert!((p.bias(0).0 - 1.0).abs() < 1e-6, "amplitude clamped to 1.0");
        assert_eq!(p.bias(14).0, 0.0, "the 15th atom was not imprinted");
    }

    #[test]
    fn echo_selection_follows_the_source_rule() {
        assert_eq!(
            GhostPrior::echo_for(0.6, 0.9, 0.9, GhostEcho::Thought),
            GhostEcho::Staunen
        );
        assert_eq!(
            GhostPrior::echo_for(0.1, 0.4, 0.9, GhostEcho::Thought),
            GhostEcho::Wisdom
        );
        assert_eq!(
            GhostPrior::echo_for(0.1, 0.1, 0.4, GhostEcho::Thought),
            GhostEcho::Grief
        );
        assert_eq!(
            GhostPrior::echo_for(0.1, 0.1, 0.1, GhostEcho::Boundary),
            GhostEcho::Boundary
        );
    }

    #[test]
    fn dominant_echo_is_the_strongest_live_trace_and_summary_is_one_per_atom() {
        let mut p = GhostPrior::new(PriorFloor::DEFAULT);
        p.imprint(&[(42, 0.3)], GhostEcho::Affinity);
        p.imprint(&[(42, 0.9), (7, 0.2)], GhostEcho::Staunen);
        assert_eq!(p.bias(42).1, Some(GhostEcho::Staunen));
        let s = p.summary();
        assert_eq!(s.len(), 2, "one entry per atom");
        assert_eq!(s[0].0, 42, "strongest first");
        assert_eq!(s[0].1, GhostEcho::Staunen);
    }

    #[test]
    fn summary_is_one_entry_per_atom_even_when_another_atom_sorts_between() {
        // Codex's repro on #1142: atom 1 at 0.9 and 0.1 with atom 2 at 0.5
        // between them — a sort-then-adjacent-dedup returns atom 1 twice.
        let mut p = GhostPrior::new(PriorFloor::DEFAULT);
        p.imprint(&[(1, 0.9), (2, 0.5), (1, 0.1)], GhostEcho::Thought);
        let s = p.summary();
        assert_eq!(s.len(), 2, "one entry per atom: {s:?}");
        assert_eq!(s[0], (1, GhostEcho::Thought, 0.9));
        assert_eq!(s[1].0, 2);
    }

    #[test]
    fn calibration_fixture_refuses_an_atom_space_it_cannot_address() {
        assert!(recurrence_fixture(PriorFloor::DEFAULT, 0, 0, FIXTURE_MIN_ATOMS - 1).is_none());
        assert!(discrimination(PriorFloor::DEFAULT, 0, 0, 0).is_none());
        assert!(recurrence_fixture(PriorFloor::DEFAULT, 30, 60, FIXTURE_MIN_ATOMS).is_some());
    }

    #[test]
    fn calibration_fixture_refuses_a_stale_count_whose_atom_id_would_wrap() {
        // 100 + 3·21811 + 2 = 65535 = u16::MAX exactly: the last admissible
        // pattern; one more (index 21812) starts at 65536 and wraps.
        assert_eq!(
            100 + 3 * (FIXTURE_MAX_STALE_PATTERNS - 1) + 2,
            usize::from(u16::MAX)
        );
        assert!(
            recurrence_fixture(PriorFloor::DEFAULT, FIXTURE_MAX_STALE_PATTERNS + 1, 0, N).is_none()
        );
        assert!(
            discrimination(PriorFloor::DEFAULT, FIXTURE_MAX_STALE_PATTERNS + 1, 0, N).is_none()
        );
        assert!(
            recurrence_fixture(PriorFloor::DEFAULT, FIXTURE_MAX_STALE_PATTERNS, 0, N).is_some()
        );
    }

    #[test]
    fn two_priors_are_independent_state() {
        // Ownership: no shared field. Imprinting one never touches the other.
        let mut a = GhostPrior::new(PriorFloor::DEFAULT);
        let b = GhostPrior::new(PriorFloor::DEFAULT);
        a.imprint(&[(1, 1.0)], GhostEcho::Wisdom);
        assert_eq!(a.active_count(), 1);
        assert_eq!(b.active_count(), 0);
        assert_eq!(b.cycle(), 0);
    }
}
