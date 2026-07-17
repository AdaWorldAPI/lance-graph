//! `ShaderDriver` — the CognitiveShader IS the driver.
//!
//! Holds BindSpace columns (owned), the p64 CognitiveShader topology (8
//! predicate planes) plus its bgz17 PaletteSemiring (O(1) distance table),
//! and an optional sink. `dispatch()` runs one cycle end-to-end.
//!
//! ```text
//!   ShaderDispatch
//!         │
//!         ▼
//!  [1] meta prefilter  (cheap u32 column sweep)
//!  [2] resolve style   (auto-detect from qualia if Auto)
//!  [3] shader cascade  (p64 CognitiveShader + bgz17 distance)
//!  [4] cycle signature (Hamming-folded fingerprint of the top-k)
//!  [5] edge emission   (CausalEdge64 per strong hit)
//!  [6] FreeEnergy gate (Flow/Hold/Block from active-inference F)
//!  [7] sink            (on_resonance → on_bus → on_crystal)
//!         │
//!         ▼
//!  ShaderCrystal
//! ```
//!
//! No forward pass, no JSON, no allocations beyond top-k + edges.

use std::sync::Arc;
use std::sync::RwLock;

use bgz17::palette_semiring::PaletteSemiring;
use causal_edge::edge::{CausalEdge64, InferenceType};
use causal_edge::pearl::CausalMask;
use causal_edge::plasticity::PlasticityState;
use causal_edge::tables::{unpack_c, unpack_f, NarsTables};
use lance_graph_contract::cognitive_shader::{
    AlphaComposite, CognitiveShaderDriver, EmitMode, MaterializeProvenance, MetaSummary, NullSink,
    RungElevator, RungLevel, ShaderBus, ShaderCrystal, ShaderDispatch, ShaderHit, ShaderResonance,
    ShaderSink, ALPHA_COMPOSITE_DIMS,
};
use lance_graph_contract::collapse_gate::{GateDecision, MergeMode, ALPHA_SATURATION_THRESHOLD};
use lance_graph_contract::grammar::free_energy::{FreeEnergy, EPIPHANY_MARGIN};
use lance_graph_contract::grammar::inference::NarsInference;
use lance_graph_contract::grammar::thinking_styles::{
    GrammarStyleAwareness, ParamKey, ParseOutcome,
};
use lance_graph_contract::materialize::materialize;
use lance_graph_contract::mul::{MulAssessment, MulThresholdProfile, SituationInput};
use lance_graph_contract::recipe_kernels::ThoughtCtx;
use lance_graph_contract::thinking::ThinkingStyle;
use p64_bridge::cognitive_shader::CognitiveShader;

use crate::auto_style;
use crate::backing::BackingStore;
use crate::bindspace::{BindSpace, WORDS_PER_FP};
use crate::mailbox_soa::MailboxSoA;
use lance_graph_contract::collapse_gate::MailboxId;

/// The single designated mailbox the dispatch read-shim selects under the
/// `mailbox-thoughtspace` feature (OQ-D, Option A — no contract change).
///
/// `ShaderDispatch` carries no `MailboxId` today, so a singleton-shaped
/// dispatch routes to this fixed id. Multi-mailbox routing is W5; until then
/// the driver `debug_assert!`s exactly one mailbox is registered. `MailboxId`
/// is a `u32` alias (`collapse_gate.rs`), so this is a free const.
#[cfg(feature = "mailbox-thoughtspace")]
const DEFAULT_MAILBOX: MailboxId = 0;

// ═══════════════════════════════════════════════════════════════════════════
// ShaderDriver — holds everything the shader needs to drive
// ═══════════════════════════════════════════════════════════════════════════

/// The genius driver: CognitiveShader in the driving seat, BindSpace and
/// the bgz17 semiring in the back, thinking-engine optional.
pub struct ShaderDriver {
    pub(crate) bindspace: Arc<BindSpace>,
    pub(crate) semiring: Arc<PaletteSemiring>,
    /// 8 predicate planes × 64 rows × u64 columns = 4 KB topology.
    /// Boxed to keep the bulk off ShaderDriver's stack frame, and held
    /// under an RwLock so the convergence highway (TD-INT-14) can swap
    /// in fresh planes when AriGraph commits new SPO knowledge.
    pub(crate) planes: RwLock<Box<[[u64; 64]; 8]>>,
    #[allow(dead_code)]
    pub(crate) default_style: u8,
    /// Per-style (12 ord) NARS-revised awareness — phi-1 humility ceiling.
    /// Updated at end of every cycle based on FreeEnergy outcome.
    pub(crate) awareness: RwLock<Vec<GrammarStyleAwareness>>,
    /// Optional precomputed 4096-head NARS truth tables (TD-INT-10).
    ///
    /// When present, the cascade can look up Pearl 2³ + DK + Plasticity +
    /// Truth at dispatch time without paying for a runtime NARS engine.
    /// Lives in `causal-edge` (zero-dep), so attaching it does NOT pull
    /// the planner into shader-driver.
    pub(crate) nars_tables: Option<Arc<NarsTables>>,
    /// Transitional per-mailbox routing surface (slice A2).
    ///
    /// Consumers can opt into per-mailbox routing by inserting
    /// `MailboxSoA<1024>` instances here via the builder's
    /// `with_mailbox` method. The singleton `Arc<BindSpace>` (above)
    /// is unchanged — this field is purely additive and does not alter
    /// any existing dispatch semantics. Removed at cutover (plan S3).
    pub(crate) mailboxes: std::collections::HashMap<MailboxId, MailboxSoA<1024>>,
    /// Persistent rung-elevation state (`RungElevator`) across dispatch
    /// cycles. Each `run()` call is "one cycle" per this module's own
    /// docstring ("emits one CycleFingerprint per cycle — the unit of
    /// thought"); the elevator lives here — same `RwLock`-guarded,
    /// updated-every-cycle pattern as `awareness` above — so that
    /// "sustained BLOCK elevates" (the policy `RungElevator::on_gate`
    /// documents) can actually observe more than one gate. A
    /// per-call-local elevator would reset every dispatch and could
    /// never accumulate a streak. Reset to the dispatch's requested
    /// base whenever `ShaderDispatch::rung` changes between calls.
    pub(crate) rung_elevator: RwLock<RungElevator>,
}

impl ShaderDriver {
    /// Construct with BindSpace + semiring + 8 planes. Prefer the builder.
    pub fn new(
        bindspace: Arc<BindSpace>,
        semiring: Arc<PaletteSemiring>,
        planes: [[u64; 64]; 8],
        default_style: u8,
    ) -> Self {
        let awareness = (0..12)
            .map(|ord| GrammarStyleAwareness::bootstrap(ord_to_thinking_style(ord)))
            .collect::<Vec<_>>();
        Self {
            bindspace,
            semiring,
            planes: RwLock::new(Box::new(planes)),
            default_style,
            awareness: RwLock::new(awareness),
            nars_tables: None,
            mailboxes: std::collections::HashMap::new(),
            rung_elevator: RwLock::new(RungElevator::new(RungLevel::default())),
        }
    }

    /// Attach precomputed NARS truth tables (TD-INT-10).
    ///
    /// Builder-style mutation: takes ownership, returns Self. Pass
    /// `Arc::new(NarsTables::build(c_levels))` (or share an existing
    /// `Arc`) to wire Pearl 2³ + Truth lookups into the cascade.
    pub fn with_nars_tables(mut self, tables: Arc<NarsTables>) -> Self {
        self.nars_tables = Some(tables);
        self
    }

    /// Borrow the attached NARS lookup tables (TD-INT-10), if any.
    #[inline]
    pub fn nars_tables(&self) -> Option<&Arc<NarsTables>> {
        self.nars_tables.as_ref()
    }

    /// Return a read reference to the `MailboxSoA<1024>` registered under
    /// `id`, or `None` if no mailbox with that id has been inserted via
    /// the builder's `with_mailbox` method.
    ///
    /// The singleton `Arc<BindSpace>` is unchanged by this accessor.
    /// This is the transitional per-mailbox routing read surface (slice A2).
    #[inline]
    pub fn mailbox(&self, id: MailboxId) -> Option<&MailboxSoA<1024>> {
        self.mailboxes.get(&id)
    }

    /// Borrow the underlying BindSpace (read-only).
    #[inline]
    pub fn bindspace(&self) -> &BindSpace {
        &self.bindspace
    }

    /// Snapshot the topology planes (8 × 64 u64).
    ///
    /// Returns a fresh copy because the planes are kept under an `RwLock`
    /// (TD-INT-14: convergence highway lets the planner swap in new
    /// AriGraph-derived planes at runtime). Callers that just want a
    /// stable view of the current topology pay a 4 KB copy.
    #[inline]
    pub fn planes(&self) -> [[u64; 64]; 8] {
        **self.planes.read().expect("planes RwLock poisoned")
    }

    /// Replace the topology planes at runtime.
    ///
    /// This is the convergence highway terminus: AriGraph commits SPO
    /// knowledge → `triplets_to_palette_layers` produces fresh `[[u64; 64]; 8]`
    /// → this method swaps them into the live driver under a write lock.
    /// The next `dispatch()` call will see the new topology.
    #[inline]
    pub fn update_planes(&self, new_planes: [[u64; 64]; 8]) {
        let mut guard = self.planes.write().expect("planes RwLock poisoned");
        **guard = new_planes;
    }

    /// Select the dispatch read substrate (W3 read-shim).
    ///
    /// Default (`mailbox-thoughtspace` OFF): the live singleton `BindSpace` —
    /// byte-identical to the pre-shim reads. Under the feature: the single
    /// designated `MailboxSoA` ([`DEFAULT_MAILBOX`]). `run()` keeps ONE body
    /// written against [`BackingStore`]; this method (NOT a `#[cfg]` inside
    /// `run`) selects which variant is constructed.
    #[inline]
    fn backing(&self) -> BackingStore<'_> {
        #[cfg(feature = "mailbox-thoughtspace")]
        {
            // OQ-D: multi-mailbox routing is W5. Until then AT MOST one mailbox
            // is the BindSpace surrogate; a singleton-shaped dispatch selects it.
            // An unmigrated driver (zero mailboxes) falls back to the singleton,
            // so the feature build never panics on a driver that hasn't been
            // populated with the designated mailbox yet.
            debug_assert!(
                self.mailboxes.len() <= 1,
                "mailbox-thoughtspace expects at most one designated mailbox \
                 (DEFAULT_MAILBOX) until W5 multi-mailbox routing; got {}",
                self.mailboxes.len()
            );
            if let Some(mb) = self.mailboxes.get(&DEFAULT_MAILBOX) {
                return BackingStore::Mailbox(mb);
            }
            BackingStore::Singleton(&self.bindspace)
        }
        #[cfg(not(feature = "mailbox-thoughtspace"))]
        {
            BackingStore::Singleton(&self.bindspace)
        }
    }

    /// Run one dispatch, feeding a sink. This is the single hot path.
    fn run<S: ShaderSink>(&self, req: &ShaderDispatch, sink: &mut S) -> ShaderCrystal {
        // W3 read-shim: select the substrate (singleton BindSpace by default;
        // the designated MailboxSoA under `mailbox-thoughtspace`). The body
        // below is written ONCE against `backing` — no `#[cfg]` branches here.
        let backing = self.backing();

        // ── Rung ascent loop (D-TRI-6) ──────────────────────────────────────
        // The persistent per-driver RungElevator's CURRENT level — advanced by
        // the PREVIOUS cycle's gate via `on_gate` at the end of this fn — selects
        // THIS cycle's cascade plane breadth. A dispatch whose requested base
        // rung differs from the elevator's tracked base resets the elevator
        // first, so streaks never leak across unrelated dispatch bases. At base
        // the mask is unchanged (no regression); above base it widens by union
        // (see `rung_widened_layer_mask` + its HAZARD(a) note).
        let effective_layer_mask = {
            let mut elevator = self
                .rung_elevator
                .write()
                .expect("rung_elevator RwLock poisoned");
            if elevator.base != req.rung {
                *elevator = RungElevator::new(req.rung);
            }
            rung_widened_layer_mask(elevator.base, elevator.level, req.layer_mask)
        };

        // [1] Cheap meta prefilter (u32 column sweep).
        let passed_rows = backing.prefilter(req.rows, &req.meta_prefilter);

        // [2] Resolve style — Auto reads the qualia of the FIRST surviving row.
        // D-CSV-5b: qualia is QualiaI4_16D; the shim converts to f32 at the read.
        let qualia_f32_arr: [f32; 17] = if let Some(&row) = passed_rows.first() {
            backing.qualia_17d(row as usize)
        } else {
            [0.0f32; 17]
        };
        let style_ord = auto_style::resolve(req.style, &qualia_f32_arr[..]);

        // [3] Shader cascade — bgz17 O(1) per probed block.
        // Snapshot the planes under the read lock so the cascade sees a
        // consistent topology even if `update_planes` fires mid-dispatch.
        let planes_snapshot: [[u64; 64]; 8] = **self.planes.read().expect("planes RwLock poisoned");
        let shader = CognitiveShader::new(planes_snapshot, &self.semiring);
        let max_dist = (self.semiring.k as f32) * (self.semiring.k as f32);
        let mut hits = Vec::<ShaderHit>::with_capacity(passed_rows.len().min(64));

        // TD-INT-10: optional NARS truth-table lookups per hit.
        let nars_tables = self.nars_tables.as_deref();

        // Content-plane Hamming pre-pass (PR #259).
        const CONTENT_MATCH_PREDICATE: u8 = 0x01;
        const MAX_CONTENT_PREPASS_ROWS: usize = 256;
        const FP_BITS: f32 = (WORDS_PER_FP * 64) as f32;
        if passed_rows.len() <= MAX_CONTENT_PREPASS_ROWS {
            let style_cfg = &crate::engine_bridge::UNIFIED_STYLES[(style_ord % 12) as usize];
            let min_resonance = style_cfg.resonance_threshold;

            for (i, &row_i) in passed_rows.iter().enumerate() {
                let fp_i = backing.content_row(row_i as usize);
                for (j_off, &row_j) in passed_rows.iter().enumerate().skip(i + 1) {
                    let fp_j = backing.content_row(row_j as usize);
                    let fp_i_bytes = unsafe {
                        std::slice::from_raw_parts(fp_i.as_ptr() as *const u8, WORDS_PER_FP * 8)
                    };
                    let fp_j_bytes = unsafe {
                        std::slice::from_raw_parts(fp_j.as_ptr() as *const u8, WORDS_PER_FP * 8)
                    };
                    let hamming =
                        ndarray::hpc::bitwise::hamming_distance_raw(fp_i_bytes, fp_j_bytes) as u32;
                    let resonance = 1.0 - (hamming as f32 / FP_BITS);
                    if resonance >= min_resonance {
                        hits.push(ShaderHit {
                            row: row_i,
                            distance: hamming.min(u16::MAX as u32) as u16,
                            predicates: CONTENT_MATCH_PREDICATE,
                            _pad: 0,
                            resonance,
                            cycle_index: i as u32,
                        });
                        hits.push(ShaderHit {
                            row: row_j,
                            distance: hamming.min(u16::MAX as u32) as u16,
                            predicates: CONTENT_MATCH_PREDICATE,
                            _pad: 0,
                            resonance,
                            cycle_index: j_off as u32,
                        });
                    }
                }
            }
        }

        for (cycle_idx, &row) in passed_rows.iter().enumerate() {
            if cycle_idx as u16 >= req.max_cycles.saturating_mul(4) {
                break;
            }
            // Use the SPO `s_idx` of the row's edge as the query palette index.
            // Rows with edge=0 default to palette 0 (identity probe).
            let edge = backing.edge(row as usize);
            let query = edge.s_idx();
            let raw = shader.cascade(query, req.radius, effective_layer_mask);
            for hit in raw.into_iter().take(4) {
                let resonance = 1.0 / (1.0 + (hit.distance as f32 / max_dist));

                // TD-INT-10: NARS truth lookup against precomputed tables.
                // The row's edge already carries a (frequency, confidence)
                // pair; we revise it against a hit-derived surrogate truth
                // (resonance as frequency, conservative half-confidence).
                // The result is currently observed only — see comment above.
                if let Some(tables) = nars_tables {
                    let f1 = edge.frequency_u8();
                    let c1 = edge.confidence_u8();
                    let f2 = (resonance.clamp(0.0, 1.0) * 255.0) as u8;
                    let c2 = 128u8;
                    let packed = tables.revise(f1, c1, f2, c2);
                    let _revised_truth = (unpack_f(packed), unpack_c(packed));
                }

                hits.push(ShaderHit {
                    row,
                    distance: hit.distance,
                    predicates: hit.predicates,
                    _pad: 0,
                    resonance,
                    cycle_index: cycle_idx as u32,
                });
            }
        }

        // Sort by resonance descending, keep top-8.
        hits.sort_by(|a, b| {
            b.resonance
                .partial_cmp(&a.resonance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(8);

        // [4] Build the cycle_fingerprint with positional Markov braiding.
        //     Each row is rotated by its cycle_index before XOR — preserves
        //     position information structurally (binary-space vsa_permute analogue).
        //     Per I-SUBSTRATE-MARKOV: this activates the Markov ±5 property
        //     even in binary space; full f32 VSA bundle is the next step.
        let mut cycle_fp = [0u64; WORDS_PER_FP];
        for h in &hits {
            let row_words = backing.content_row(h.row as usize);
            let pos = (h.cycle_index as usize) % WORDS_PER_FP;
            for (i, w) in row_words.iter().enumerate() {
                cycle_fp[(i + pos) % WORDS_PER_FP] ^= *w;
            }
        }

        // [5] Entropy + std-dev of top-k resonances.
        let (entropy, std_dev) = entropy_std(&hits);

        // [6] FreeEnergy gate (principled F from resonance + KL surrogate).
        let top_resonance = hits.first().map(|h| h.resonance).unwrap_or(0.0);
        let free_energy = FreeEnergy::compose(top_resonance, std_dev);

        // Epiphany check: top-2 hypotheses within margin, both non-catastrophic
        let is_epiphany = hits.len() >= 2 && {
            let fe2 = FreeEnergy::compose(hits[1].resonance, std_dev);
            (fe2.total - free_energy.total).abs() < EPIPHANY_MARGIN && !fe2.is_catastrophic()
        };

        // TD-INT-3: Meta-Uncertainty Layer assessment.
        //
        // Build a SituationInput from what the shader can directly observe
        // and compute a MulAssessment. Fields the shader can't see cleanly
        // (calibration_accuracy, allostatic_load, max_acceptable_damage,
        // sandbox_available, etc.) fall back to SituationInput::default() —
        // tightening these is a deferred wiring point that will land when
        // the awareness column publishes Brier history and the orchestration
        // bridge passes a per-cycle damage budget.
        //
        //   felt_competence       ← top resonance (cycle's self-reported "I got it")
        //   demonstrated_competence ← (1 - free_energy.total) (active-inference truth)
        //   environment_stability ← 1 - std_dev clamp (low spread = stable hypotheses)
        //   challenge_level       ← std_dev clamp (high spread = harder problem)
        //   skill_level           ← top awareness divergence proxy (style competence)
        // Skill proxy: this style's recent-success frequency from the
        // NARS-revised awareness. Maps directly to MUL's skill_level
        // axis — competence as the system has demonstrated it, not as
        // it feels right now.
        let awareness_skill = self
            .awareness
            .read()
            .ok()
            .and_then(|aw| {
                aw.get(style_ord as usize)
                    .map(|s| s.recent_success.frequency as f64)
            })
            .unwrap_or(0.5);
        let std_dev_clamped = std_dev.clamp(0.0, 1.0) as f64;
        let situation = SituationInput {
            felt_competence: top_resonance.clamp(0.0, 1.0) as f64,
            demonstrated_competence: (1.0 - free_energy.total).clamp(0.0, 1.0) as f64,
            environment_stability: (1.0 - std_dev_clamped).clamp(0.0, 1.0),
            challenge_level: std_dev_clamped,
            skill_level: awareness_skill,
            ..SituationInput::default()
        };
        let mul = MulAssessment::compute(&situation);

        // D-CASCADE-V1-7: ctx_id resolves via BindSpace.entity_type +
        // optional OntologyRegistry handle; per-row context column is
        // Wave-3.5 follow-up (gate is one-per-dispatch today).
        let ctx_id: u32 = passed_rows
            .first()
            .copied()
            .and_then(|r| {
                // entity_type routes through the shim; ontology() stays on the
                // singleton (the registry re-home is W4b — see plan §90).
                let etid = backing.entity_type(r as usize);
                if etid == 0 {
                    return None;
                }
                self.bindspace.ontology().and_then(|reg| {
                    reg.enumerate_first_with_entity_type_id(etid)
                        .map(|row| row.ontology_context_id())
                })
            })
            .unwrap_or(0);
        let profile = MulThresholdProfile::for_context(ctx_id);
        let trust_below_floor = (mul.trust.value as f32) < profile.trust_min;

        // Gate decision: catastrophic F blocks; MUL veto on
        // unskilled-overconfident OR sub-profile trust downgrades any
        // would-be Flow to Hold; epiphany holds (preserve the contradiction);
        // homeostasis flows.
        let gate = if free_energy.is_catastrophic() {
            GateDecision::BLOCK
        } else if mul.is_unskilled_overconfident() || trust_below_floor {
            // MUL veto: the system "feels confident" while DK / trust
            // textures flag the gap, OR trust falls below the
            // ontology-context profile's floor. Hold rather than commit.
            GateDecision::HOLD
        } else if is_epiphany {
            GateDecision::HOLD
        } else if free_energy.is_homeostatic() {
            GateDecision {
                gate: 0,
                merge: MergeMode::Bundle,
            }
        } else {
            GateDecision::HOLD
        };

        // [5] Emit one CausalEdge64 per strong hit (up to 8).
        let mut emitted = [0u64; 8];
        let mut emitted_n = 0u8;
        for h in hits.iter().take(8) {
            if h.resonance < 0.2 {
                continue;
            }
            let f = (h.resonance.clamp(0.0, 1.0) * 255.0) as u8;
            let c = (h.resonance.clamp(0.0, 1.0) * 255.0) as u8;
            let s_palette = (h.row % 256) as u8;
            let o_palette = ((h.row / 4) % 256) as u8;
            let edge = CausalEdge64::pack(
                s_palette,
                0,
                o_palette,
                f,
                c,
                CausalMask::from_bits(h.predicates & 0x07),
                0,
                style_ord_to_inference(style_ord),
                PlasticityState::from_bits(0),
                (h.cycle_index & 0xFFF) as u16,
            );
            emitted[emitted_n as usize] = edge.0;
            emitted_n += 1;
        }

        let mut top_k = [ShaderHit::default(); 8];
        for (i, h) in hits.iter().take(8).enumerate() {
            top_k[i] = *h;
        }

        let resonance_dto = ShaderResonance {
            top_k,
            hit_count: hits.len() as u16,
            cycles_used: passed_rows.len() as u16,
            entropy,
            std_dev,
            style_ord,
        };

        // ── [7] Sink stage — dispatch on MergeMode ──────────────────────
        //
        // Pillar-7 (B5): when the effective merge mode for this dispatch
        // is `AlphaFrontToBack`, replace the top-K hit aggregation with
        // Kerbl 2023 EWA front-to-back α-compositing. All other merge
        // modes (Bundle / Xor / Superposition) preserve their existing
        // behaviour exactly — only `alpha_composite` toggles.
        //
        // The override hierarchy is:
        //   1. `req.merge_override` (caller-set)
        //   2. `gate.merge`         (gate-decided)
        //
        // Only `AlphaFrontToBack` changes the sink path; everything else
        // routes through the original code below unchanged.
        let effective_merge = req.merge_override.unwrap_or(gate.merge);
        // D-CSV-5b: bs.qualia is now QualiaI4Column (returns QualiaI4_16D by value).
        // alpha_front_to_back_composite expects F: Fn(u32) -> &'a [f32].
        // Pre-materialize hit qualia as f32 so references are valid for the closure.
        let hit_qualia_f32: Vec<(u32, [f32; 17])> = hits
            .iter()
            .map(|h| (h.row, backing.qualia_17d(h.row as usize)))
            .collect();
        let alpha_composite = if effective_merge == MergeMode::AlphaFrontToBack {
            let threshold = req
                .alpha_saturation_override
                .unwrap_or(ALPHA_SATURATION_THRESHOLD);
            Some(alpha_front_to_back_composite(
                &hits,
                |row| {
                    hit_qualia_f32
                        .iter()
                        .find(|(r, _)| *r == row)
                        .map(|(_, q)| &q[..])
                        .unwrap_or(&[][..])
                },
                threshold,
            ))
        } else {
            None
        };

        // Advance the persistent per-driver `RungElevator` with THIS cycle's
        // already-decided `gate`, BEFORE the sink callbacks. The sink early
        // returns (`on_resonance` / `on_bus` returning false) must NOT bypass
        // the rung transition — a sink that stops processing would otherwise
        // freeze the ascent ladder, now that the elevator drives the cascade
        // (CodeRabbit #708). Sustained BLOCK elevates over the dispatched base;
        // sustained FLOW relaxes back to it. The base-reset is REPEATED here,
        // inside the SAME critical section as `on_gate` (not only at the
        // top-of-`run()` read): the elevator is a `&self` per-driver `RwLock`
        // singleton, so a concurrent dispatch with a different `req.rung` may
        // have reset `base` in the window since the top read; re-validating here
        // keeps reset+advance atomic, so this cycle's gate is never counted
        // against another request's base (codex r3603015026). The resulting
        // level is what the NEXT dispatch on this base consults — closing the
        // ascent loop across cycles.
        let elevated_rung: u8 = {
            let mut elevator = self
                .rung_elevator
                .write()
                .expect("rung_elevator RwLock poisoned");
            if elevator.base != req.rung {
                *elevator = RungElevator::new(req.rung);
            }
            elevator.on_gate(gate) as u8
        };

        // [7] Sink callbacks.
        if !sink.on_resonance(&resonance_dto) {
            return ShaderCrystal {
                bus: ShaderBus {
                    cycle_fingerprint: cycle_fp,
                    emitted_edges: emitted,
                    emitted_edge_count: emitted_n,
                    gate,
                    resonance: resonance_dto,
                },
                persisted_row: None,
                meta: MetaSummary::default(),
                materialize: MaterializeProvenance::default(),
                alpha_composite,
            };
        }

        let bus = ShaderBus {
            cycle_fingerprint: cycle_fp,
            emitted_edges: emitted,
            emitted_edge_count: emitted_n,
            gate,
            resonance: resonance_dto,
        };
        if !sink.on_bus(&bus) {
            return ShaderCrystal {
                bus,
                persisted_row: None,
                meta: MetaSummary::default(),
                materialize: MaterializeProvenance::default(),
                alpha_composite,
            };
        }

        // Meta summary (confidence from top-1 resonance, FreeEnergy-derived).
        let confidence = resonance_dto.top_k[0].resonance;
        let meta = MetaSummary {
            confidence,
            meta_confidence: (1.0 - free_energy.total).clamp(0.0, 1.0),
            brier: 0.0,
            should_admit_ignorance: free_energy.is_catastrophic(),
        };

        let persisted_row = match req.emit {
            EmitMode::Persist => Some(resonance_dto.top_k[0].row),
            _ => None,
        };

        // Materialized-awareness provenance — runs the F→34→F loop + HHTL fork as
        // a SIDE analysis over this cycle's observables. Provenance-only: it has
        // NOT influenced `gate` above and does not influence persistence; it only
        // records "what the 34 would dispatch and whether the leaf residue forks".
        let candidate_resonances: Vec<f32> = hits.iter().map(|h| h.resonance).collect();
        let materialize_prov = materialize_provenance(
            &free_energy,
            std_dev,
            top_resonance,
            awareness_skill,
            &candidate_resonances,
            elevated_rung,
        );

        // [8] NARS revision — phi-1 humility ceiling.
        //     System observes its own outcome and revises per-style awareness.
        //     This is what makes the cognitive loop close: every cycle updates
        //     the next cycle's F landscape via accumulated belief.
        let outcome = free_energy_to_outcome(&free_energy, is_epiphany);
        let inference = style_ord_to_inference(style_ord);
        let nars_inference = match inference {
            InferenceType::Deduction => NarsInference::Deduction,
            InferenceType::Induction => NarsInference::Induction,
            InferenceType::Abduction => NarsInference::Abduction,
            InferenceType::Revision => NarsInference::Revision,
            InferenceType::Synthesis => NarsInference::Synthesis,
            // style_ord_to_inference never returns Reserved5/6/7;
            // fall back to Revision so reserved variants map cleanly.
            _ => NarsInference::Revision,
        };
        let key = ParamKey::NarsPrimary(nars_inference);
        if let Ok(mut aw) = self.awareness.write() {
            if let Some(style_aw) = aw.get_mut(style_ord as usize) {
                style_aw.revise(key, outcome);
            }
        }

        let crystal = ShaderCrystal {
            bus,
            persisted_row,
            meta,
            materialize: materialize_prov,
            alpha_composite,
        };
        sink.on_crystal(&crystal);
        crystal
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rung ascent loop — rung → predicate-plane widening (D-TRI-6)
// ═══════════════════════════════════════════════════════════════════════════

/// Combine the dispatch's 8-bit predicate-plane mask with the CURRENT rung.
/// At (or below) the dispatched base rung this returns `req_mask` UNCHANGED —
/// zero behaviour change for a thought sitting at its requested depth. Above
/// base, each Pearl level the rung has climbed UNIONs in additional predicate
/// planes, so the consulted set is a strict SUPERSET of `req_mask`: elevation
/// only ever WIDENS the cascade, never narrows it.
///
/// HAZARD (a) — axis translation (CORRECTNESS GATE). The p64
/// `CognitiveShader::cascade` `layer_mask` is an 8-bit PREDICATE-PLANE mask
/// (CAUSES..BECOMES; bit `z` gates plane `z`), NOT the 3-bit SPO projection
/// mask `RungLevel::causal_mask_bits()` returns. Feeding `causal_mask_bits()`
/// raw would be a category error AND, at base Surface (0b001), would collapse
/// the cascade to plane 0 only — a severe regression. So the rung's *Pearl
/// level* selects a predicate-plane widen-set, mirroring the Pearl→predicate
/// routing already in `p64_bridge::edge_to_layer_mask`. Because the widen is a
/// UNION, a caller that already asked for all planes (`0xFF`, the
/// `ShaderDispatch` default) sees no change — you cannot consult more than every
/// plane. The specific planes each Pearl level unlocks are a calibration a later
/// probe may retune; the identity-at-base and superset-monotone properties are
/// invariant.
fn rung_widened_layer_mask(base: RungLevel, level: RungLevel, req_mask: u8) -> u8 {
    if (level as u8) <= (base as u8) {
        return req_mask;
    }
    // p64 predicate-plane bit positions (mirror p64_bridge::{CAUSES..BECOMES}):
    // CAUSES=0 ENABLES=1 SUPPORTS=2 CONTRADICTS=3 REFINES=4 ABSTRACTS=5
    // GROUNDS=6 BECOMES=7.
    const ENABLES: u8 = 1 << 1;
    const CONTRADICTS: u8 = 1 << 3;
    const REFINES: u8 = 1 << 4;
    const ABSTRACTS: u8 = 1 << 5;
    const BECOMES: u8 = 1 << 7;
    // Pearl-level → predicate widen-set (superset-monotone):
    //   L1 observation    → nothing beyond req_mask,
    //   L2 intervention   → +ENABLES +REFINES,
    //   L3 counterfactual → also +CONTRADICTS +ABSTRACTS +BECOMES.
    let widen = match level.pearl_level() {
        1 => 0,
        2 => ENABLES | REFINES,
        _ => ENABLES | REFINES | CONTRADICTS | ABSTRACTS | BECOMES,
    };
    req_mask | widen
}

// ═══════════════════════════════════════════════════════════════════════════
// Pillar-7 — α-front-to-back composite helper (stage [7] sink mode)
// ═══════════════════════════════════════════════════════════════════════════

/// Kerbl-2023 EWA-splatting front-to-back α-compositing over `hits`.
///
/// Hits are assumed sorted by confidence DESC (the dispatch pipeline
/// already does this in stage [3]). The loop terminates early when
/// accumulated α exceeds `saturation_threshold` — the early-ray-
/// termination optimization that makes 3DGS practical.
///
/// `qualia_for_row` extracts the per-hit color (qualia vector) from
/// whatever payload the BindSpace has bound to that row. The slice it
/// returns is copied into `color_acc`'s active prefix; trailing slots
/// stay zero. Robust to non-finite confidences (treated as α = 0,
/// fully transparent — does not advance α_acc).
///
/// Per Pillar-7 (B5):
///
/// ```text
///   α_acc     = 0
///   color_acc = 0
///   for hit in hits:                           # sorted by confidence DESC
///       α_i  = hit.confidence_to_alpha()
///       w    = α_i * (1 - α_acc)               # transmittance × current α
///       color_acc += hit.qualia_payload() * w
///       α_acc     += w
///       if α_acc > saturation_threshold: break # early ray termination
/// ```
pub fn alpha_front_to_back_composite<'a, F>(
    hits: &[ShaderHit],
    qualia_for_row: F,
    saturation_threshold: f32,
) -> AlphaComposite
where
    F: Fn(u32) -> &'a [f32],
{
    let mut alpha_acc: f32 = 0.0;
    let mut color_acc = [0.0f32; ALPHA_COMPOSITE_DIMS];
    let mut hits_consumed: u16 = 0;
    let mut saturated = false;

    for hit in hits.iter() {
        let alpha_i = hit.confidence_to_alpha();
        if alpha_i <= 0.0 {
            // Fully transparent — does not advance α_acc, but still
            // counts as "considered" so loop progress is visible.
            hits_consumed = hits_consumed.saturating_add(1);
            continue;
        }
        let weight = alpha_i * (1.0 - alpha_acc);
        let q = qualia_for_row(hit.row);
        let active = q.len().min(ALPHA_COMPOSITE_DIMS);
        for i in 0..active {
            color_acc[i] += q[i] * weight;
        }
        alpha_acc += weight;
        hits_consumed = hits_consumed.saturating_add(1);
        if alpha_acc > saturation_threshold {
            saturated = true;
            break;
        }
    }

    AlphaComposite {
        color_acc,
        alpha_acc,
        hits_consumed,
        saturated,
    }
}

impl CognitiveShaderDriver for ShaderDriver {
    fn dispatch(&self, req: &ShaderDispatch) -> ShaderCrystal {
        let mut null = NullSink;
        self.run(req, &mut null)
    }

    fn dispatch_with_sink<S: ShaderSink>(
        &self,
        req: &ShaderDispatch,
        sink: &mut S,
    ) -> ShaderCrystal {
        self.run(req, sink)
    }

    fn row_count(&self) -> u32 {
        self.bindspace.len as u32
    }

    fn byte_footprint(&self) -> usize {
        self.bindspace.byte_footprint()
            + 8 * 64 * 8                           // planes: 4096 bytes
            + self.semiring.compose_table.len()    // k×k u8
            + self.semiring.distance_matrix.byte_size() // k×k u16
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder — EmbedAnything fluent pattern
// ═══════════════════════════════════════════════════════════════════════════

/// Fluent builder for `ShaderDriver`.
///
/// ```rust,ignore
/// let driver = CognitiveShaderBuilder::new()
///     .bindspace(Arc::new(my_bs))
///     .semiring(Arc::new(sr))
///     .planes(planes)
///     .default_style(auto_style::ANALYTICAL)
///     .build();
/// ```
pub struct CognitiveShaderBuilder {
    bindspace: Option<Arc<BindSpace>>,
    semiring: Option<Arc<PaletteSemiring>>,
    planes: Option<[[u64; 64]; 8]>,
    default_style: u8,
    nars_tables: Option<Arc<NarsTables>>,
    /// Transitional per-mailbox routing map populated by `with_mailbox`.
    /// Forwarded into `ShaderDriver::mailboxes` at `build()` time.
    mailboxes: std::collections::HashMap<MailboxId, MailboxSoA<1024>>,
}

impl CognitiveShaderBuilder {
    pub fn new() -> Self {
        Self {
            bindspace: None,
            semiring: None,
            planes: None,
            default_style: auto_style::DELIBERATE,
            nars_tables: None,
            mailboxes: std::collections::HashMap::new(),
        }
    }

    pub fn bindspace(mut self, bs: Arc<BindSpace>) -> Self {
        self.bindspace = Some(bs);
        self
    }

    pub fn semiring(mut self, sr: Arc<PaletteSemiring>) -> Self {
        self.semiring = Some(sr);
        self
    }

    pub fn planes(mut self, p: [[u64; 64]; 8]) -> Self {
        self.planes = Some(p);
        self
    }

    pub fn default_style(mut self, ord: u8) -> Self {
        self.default_style = ord.min(11);
        self
    }

    /// Attach precomputed NARS lookup tables (TD-INT-10).
    pub fn nars_tables(mut self, tables: Arc<NarsTables>) -> Self {
        self.nars_tables = Some(tables);
        self
    }

    /// Register a `MailboxSoA<1024>` for transitional per-mailbox routing
    /// (slice A2). The mailbox is keyed by `id`; a second call with the
    /// same `id` replaces the previous entry. Multiple mailboxes are
    /// supported. The singleton `Arc<BindSpace>` is not affected.
    pub fn with_mailbox(mut self, id: MailboxId, soa: MailboxSoA<1024>) -> Self {
        self.mailboxes.insert(id, soa);
        self
    }

    pub fn build(self) -> ShaderDriver {
        let awareness = (0..12)
            .map(|ord| GrammarStyleAwareness::bootstrap(ord_to_thinking_style(ord)))
            .collect::<Vec<_>>();
        ShaderDriver {
            bindspace: self.bindspace.expect("bindspace required"),
            semiring: self.semiring.expect("semiring required"),
            planes: RwLock::new(Box::new(self.planes.unwrap_or([[0u64; 64]; 8]))),
            default_style: self.default_style,
            awareness: RwLock::new(awareness),
            nars_tables: self.nars_tables,
            mailboxes: self.mailboxes,
            rung_elevator: RwLock::new(RungElevator::new(RungLevel::default())),
        }
    }
}

impl Default for CognitiveShaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn entropy_std(hits: &[ShaderHit]) -> (f32, f32) {
    if hits.is_empty() {
        return (0.0, 0.0);
    }
    let sum: f32 = hits.iter().map(|h| h.resonance).sum();
    if sum <= 0.0 {
        return (0.0, 0.0);
    }
    let mut ent = 0.0f32;
    for h in hits {
        let p = h.resonance / sum;
        if p > 1e-9 {
            ent -= p * p.ln();
        }
    }
    let mean = sum / hits.len() as f32;
    let var: f32 = hits
        .iter()
        .map(|h| (h.resonance - mean).powi(2))
        .sum::<f32>()
        / hits.len() as f32;
    (ent, var.sqrt())
}

/// Run the materialized-awareness analysis *alongside* the cycle — provenance
/// only, never alters the gate. Builds a `ThoughtCtx` from the cycle's
/// already-computed observables, runs the F→34→F loop, and computes the HHTL
/// fork action from a dispersion proxy (CONJECTURE pending the real orthogonal
/// `CoarseResidue` magnitude from the codec path).
///
/// Observable → `ThoughtCtx` mapping (faithful to fields the cycle already builds):
/// - `free_energy` ← `F.total` (surprise)
/// - `sd`          ← `std_dev` (CollapseGate dispersion — exact)
/// - `confidence`  ← `1 - F.total` (the driver's own `demonstrated_competence`)
/// - `dissonance`  ← `|top_resonance - (1 - F.total)|` (the Dunning-Kruger gap:
///   what the cycle *feels* vs what it has *demonstrated*)
/// - `temperature` ← `std_dev` clamp (spread ⇒ explore; proxy)
/// - `rung`        ← live `RungElevator` level (sustained-BLOCK elevation
///   over the dispatched base — see `ShaderDriver::rung_elevator`)
fn materialize_provenance(
    free_energy: &FreeEnergy,
    std_dev: f32,
    top_resonance: f32,
    awareness_skill: f64,
    candidate_resonances: &[f32],
    rung: u8,
) -> MaterializeProvenance {
    let demonstrated = (1.0 - free_energy.total).clamp(0.0, 1.0);
    let mut ctx = ThoughtCtx::new(candidate_resonances.to_vec());
    ctx.free_energy = free_energy.total.clamp(0.0, 1.0);
    ctx.sd = std_dev;
    ctx.confidence = demonstrated;
    ctx.dissonance = (top_resonance.clamp(0.0, 1.0) - demonstrated)
        .abs()
        .clamp(0.0, 1.0);
    ctx.temperature = std_dev.clamp(0.0, 1.0);
    ctx.rung = rung;

    let trace = materialize(&mut ctx, 64);

    // HHTL fork: `std_dev` (dispersion) is the CONJECTURE residue-magnitude proxy
    // until the real orthogonal `CoarseResidue` is surfaced into the cycle. The
    // floor/sigma_k are calibrated to std_dev's ~[0.05, 0.35] working range (NOT
    // the codec's NOISE_FLOOR), so a confident low-spread cycle reads as low
    // challenge (Commit) and a scattered one as high (fork). `depth == max_depth`
    // treats the read as a leaf, so ForkDomain is reachable when skill is short.
    const STD_DEV_RESIDUE_FLOOR: f64 = 0.05;
    const STD_DEV_RESIDUE_SIGMA_K: f64 = 6.0;
    let fork = ndarray::hpc::entropy_ladder::fork_decision(
        std_dev as f64,
        awareness_skill,
        1,
        1,
        STD_DEV_RESIDUE_FLOOR,
        STD_DEV_RESIDUE_SIGMA_K,
    ) as u8;

    MaterializeProvenance {
        first_tactic: trace.steps.first().map(|s| s.tactic_id).unwrap_or(0),
        steps: trace.steps.len() as u16,
        rested: trace.rested,
        final_free_energy: trace.final_free_energy,
        fork,
    }
}

#[allow(dead_code)]
fn collapse_gate(sd: f32) -> GateDecision {
    // Matches thinking_engine::cognitive_stack::{SD_FLOW_THRESHOLD, SD_BLOCK_THRESHOLD}.
    const FLOW: f32 = 0.15;
    const BLOCK: f32 = 0.35;
    if sd < FLOW {
        GateDecision {
            gate: 0,
            merge: MergeMode::Xor,
        }
    } else if sd > BLOCK {
        GateDecision::BLOCK
    } else {
        GateDecision::HOLD
    }
}

fn style_ord_to_inference(ord: u8) -> InferenceType {
    // analytical/convergent/systematic → Deduction
    // creative/divergent/exploratory   → Induction
    // focused/diffuse/peripheral       → Abduction
    // intuitive/deliberate             → Revision
    // metacognitive                    → Synthesis
    //
    // Driver policy, deliberately local. The ordinal RANGES below encode
    // family clusters and are safe ONLY because StyleFamily's
    // discriminants are frozen (pinned in contract tests) — never
    // re-order the families without revisiting these ranges.
    match ord {
        1..=3 => InferenceType::Deduction,
        4..=6 => InferenceType::Induction,
        7..=9 => InferenceType::Abduction,
        0 | 10 => InferenceType::Revision,
        _ => InferenceType::Synthesis,
    }
}

/// Map shader ordinal (0..11, UNIFIED_STYLES) to the family's default
/// 36-runbook for awareness bootstrap — routed through THE canonical
/// `StyleFamily::default_runbook()` (M9 dedup). The local table this fn
/// carried had drifted at ords 8/9/10 (diffuse/peripheral/intuitive →
/// Speculative/Curious/Reflective vs canonical Gentle/Speculative/Poetic);
/// adjudication in dtsc1-thinkingstyle-dedup-spec-v1.md §3 S1/S6.
fn ord_to_thinking_style(ord: u8) -> ThinkingStyle {
    lance_graph_contract::style_family::StyleFamily::from_ordinal(ord)
        .unwrap_or_default()
        .default_runbook()
}

/// Map FreeEnergy outcome to ParseOutcome for NARS revision.
fn free_energy_to_outcome(fe: &FreeEnergy, is_epiphany: bool) -> ParseOutcome {
    if is_epiphany {
        ParseOutcome::LocalSuccessConfirmedByLLM
    } else if fe.is_homeostatic() {
        ParseOutcome::LocalSuccess
    } else if fe.is_catastrophic() {
        ParseOutcome::LocalFailureLLMSucceeded
    } else {
        ParseOutcome::EscalatedButLLMAgreed
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindspace::{BindSpaceBuilder, QUALIA_DIMS, WORDS_PER_FP};
    use bgz17::base17::Base17;
    use bgz17::palette::Palette;
    use lance_graph_contract::cognitive_shader::MetaWord;
    use lance_graph_contract::cognitive_shader::{
        ColumnWindow, MetaFilter, ShaderDispatch, StyleSelector,
    };

    fn demo_bindspace() -> BindSpace {
        use lance_graph_contract::qualia::QualiaI4_16D;
        let q = QualiaI4_16D::ZERO;
        let content = [0u64; WORDS_PER_FP];
        BindSpaceBuilder::new(4)
            .push(&content, MetaWord::new(1, 1, 200, 200, 5), 0, q, 0, 0)
            .push(&content, MetaWord::new(2, 2, 100, 100, 5), 0, q, 0, 0)
            .push(&content, MetaWord::new(3, 3, 50, 50, 5), 0, q, 0, 0)
            .push(&content, MetaWord::new(4, 4, 0, 0, 5), 0, q, 0, 0)
            .build()
    }

    fn demo_semiring() -> PaletteSemiring {
        let entries: Vec<Base17> = (0..16)
            .map(|i| {
                let mut dims = [0i16; 17];
                dims[0] = (i as i16) * 100;
                dims[1] = ((i as i16) * 37) % 200;
                Base17 { dims }
            })
            .collect();
        let palette = Palette { entries };
        PaletteSemiring::build(&palette)
    }

    fn demo_planes() -> [[u64; 64]; 8] {
        let mut planes = [[0u64; 64]; 8];
        for i in 0..4 {
            if i + 1 < 4 {
                planes[0][i] |= 1u64 << (i + 1);
            } // CAUSES
            planes[2][i] |= 1u64 << i; // SUPPORTS self
        }
        planes
    }

    #[test]
    fn driver_builder_builds() {
        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .default_style(auto_style::ANALYTICAL)
            .build();
        assert_eq!(driver.row_count(), 4);
        assert!(driver.byte_footprint() > 0);
    }

    #[test]
    fn dispatch_runs_and_emits_cycle_fingerprint() {
        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();

        let req = ShaderDispatch {
            rows: ColumnWindow::new(0, 4),
            meta_prefilter: MetaFilter::ALL,
            layer_mask: 0xFF,
            radius: u16::MAX,
            style: StyleSelector::Ordinal(auto_style::ANALYTICAL),
            ..Default::default()
        };
        let crystal = driver.dispatch(&req);
        assert_eq!(crystal.bus.resonance.style_ord, auto_style::ANALYTICAL);
    }

    #[test]
    fn dispatch_populates_materialize_provenance() {
        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();
        let req = ShaderDispatch {
            rows: ColumnWindow::new(0, 4),
            meta_prefilter: MetaFilter::ALL,
            layer_mask: 0xFF,
            radius: u16::MAX,
            style: StyleSelector::Ordinal(auto_style::ANALYTICAL),
            ..Default::default()
        };
        let crystal = driver.dispatch(&req);
        let m = crystal.materialize;
        // Sane provenance. `first_tactic == 0` is the legitimate "already at rest,
        // nothing dispatched" case (this demo cycle is confident → settles at once);
        // any dispatched step must name a real tactic.
        assert!(
            m.first_tactic <= 34,
            "tactic id in 0..=34, got {}",
            m.first_tactic
        );
        if m.steps > 0 {
            assert!(
                (1..=34).contains(&m.first_tactic),
                "a dispatched step must name a real tactic, got {}",
                m.first_tactic
            );
        }
        assert!(
            m.fork <= 3,
            "fork action is a valid ForkAction u8, got {}",
            m.fork
        );
        assert!(m.final_free_energy.is_finite() && (0.0..=1.0).contains(&m.final_free_energy));
    }

    #[test]
    fn materialize_provenance_confident_commits_scattered_forks() {
        // Confident cycle: high resonance, low dispersion, ample skill → Boredom →
        // Commit (fork 0). The leaf residue is small, the domain over-explains.
        let fe_lo = FreeEnergy::compose(0.95, 0.05);
        let confident = materialize_provenance(&fe_lo, 0.05, 0.95, 0.9, &[0.95, 0.9, 0.88], 1);
        assert_eq!(confident.fork, 0, "confident cycle commits (no fork)");

        // Scattered cycle: low resonance, high dispersion, short skill → Anxiety at
        // leaf → ForkDomain (fork 3): the orthogonal leaf residue is strong enough
        // that free energy forks into a new domain.
        let fe_hi = FreeEnergy::compose(0.3, 0.4);
        let scattered = materialize_provenance(&fe_hi, 0.4, 0.3, 0.1, &[0.3, 0.28, 0.31], 1);
        assert_eq!(
            scattered.fork, 3,
            "scattered low-skill cycle forks to a new domain"
        );
    }

    /// Full driver-cycle test for the `RungElevator` wiring: a dispatch with
    /// `rung: RungLevel::Surface` whose cycles (each `dispatch()` call is
    /// "one cycle") produce sustained BLOCK gates elevates the driver's
    /// persistent `rung_elevator` above `Surface`. Uses an empty row window
    /// (`ColumnWindow::new(0, 0)`) so `hits` is deterministically empty on
    /// every call → `top_resonance = 0.0`, `std_dev = 0.0` →
    /// `FreeEnergy::compose(0.0, 0.0).total == 1.0 > FAILURE_CEILING (0.8)`
    /// → `GateDecision::BLOCK`, every single cycle, no flakiness.
    #[test]
    fn rung_elevator_persists_across_cycles_and_elevates_on_sustained_block() {
        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();

        let req = ShaderDispatch {
            rows: ColumnWindow::new(0, 0),
            meta_prefilter: MetaFilter::ALL,
            layer_mask: 0xFF,
            radius: u16::MAX,
            style: StyleSelector::Ordinal(auto_style::ANALYTICAL),
            rung: RungLevel::Surface,
            ..Default::default()
        };

        // Base rung starts at Surface (RungElevator::new(RungLevel::default())).
        assert_eq!(
            driver.rung_elevator.read().unwrap().level,
            RungLevel::Surface
        );

        // Cycle 1: BLOCK, but `RungElevator::DEFAULT_THRESHOLD == 2` — a
        // single BLOCK is not yet "sustained", so no elevation.
        let crystal1 = driver.dispatch(&req);
        assert_eq!(crystal1.bus.gate, GateDecision::BLOCK);
        assert_eq!(
            driver.rung_elevator.read().unwrap().level,
            RungLevel::Surface,
            "one BLOCK cycle must not elevate (threshold is 2 consecutive)"
        );

        // Cycle 2: second consecutive BLOCK — sustained — elevates one rung.
        let crystal2 = driver.dispatch(&req);
        assert_eq!(crystal2.bus.gate, GateDecision::BLOCK);
        assert_eq!(
            driver.rung_elevator.read().unwrap().level,
            RungLevel::Shallow,
            "two consecutive BLOCK cycles must elevate Surface -> Shallow"
        );

        // A dispatch requesting a *different* base rung resets the
        // elevator to that new base (streaks never leak across unrelated
        // dispatch contexts) — the `materialize_provenance` seam this
        // elevator feeds (`ctx.rung ← elevator.on_gate(gate)`) is exercised
        // directly in the narrower unit test below.
        let req_meta = ShaderDispatch {
            rung: RungLevel::Meta,
            ..req
        };
        let _ = driver.dispatch(&req_meta);
        assert_eq!(
            driver.rung_elevator.read().unwrap().base,
            RungLevel::Meta,
            "a dispatch with a different requested base resets the elevator"
        );
    }

    /// Narrower unit test of the wired seam itself: `materialize_provenance`
    /// now takes the elevator's current level as an explicit `rung: u8`
    /// parameter and threads it into `ThoughtCtx::rung`, which
    /// `materialize()` uses for tier selection (`materialize.rs` lines
    /// ~99-101: `ctx.rung >= 7` / `>= 4` gate which tactic tier is offered).
    /// This asserts the parameter is load-bearing, not decorative: raising
    /// `rung` from `1` to `9` (`RungLevel::Transcendent`) changes the offered
    /// tactic tier for an otherwise-identical scattered cycle.
    #[test]
    fn materialize_provenance_threads_elevated_rung_into_thought_ctx() {
        let fe_hi = FreeEnergy::compose(0.3, 0.4);
        let low_rung = materialize_provenance(&fe_hi, 0.4, 0.3, 0.1, &[0.3, 0.28, 0.31], 1);
        let high_rung = materialize_provenance(&fe_hi, 0.4, 0.3, 0.1, &[0.3, 0.28, 0.31], 9);
        // Both are well-formed provenance (sanity floor from the existing
        // dispatch_populates_materialize_provenance assertions).
        assert!(low_rung.first_tactic <= 34);
        assert!(high_rung.first_tactic <= 34);
        assert!(low_rung.final_free_energy.is_finite());
        assert!(high_rung.final_free_energy.is_finite());
        // Load-bearing check. Honesty note: tier is a +1 tie-weight in the
        // 34-tactic scoring (mechanism +5 dominates), so different rungs are
        // NOT guaranteed to select different tactics for every input — but for
        // THIS scattered mid-surprise cycle they empirically do (rung 1 →
        // tactic 17, rung 9 → tactic 3 at authoring time). Asserting
        // inequality (not exact ids) pins the property "the rung parameter
        // reaches tier selection and can flip the choice" without freezing the
        // recipe table.
        assert_ne!(
            low_rung.first_tactic, high_rung.first_tactic,
            "rung must be load-bearing in tactic selection for this input"
        );
    }

    #[test]
    fn dispatch_with_prefilter_excludes_rows() {
        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();

        let tight = MetaFilter {
            nars_c_min: 150,
            ..MetaFilter::ALL
        };
        let req = ShaderDispatch {
            rows: ColumnWindow::new(0, 4),
            meta_prefilter: tight,
            ..Default::default()
        };
        let crystal = driver.dispatch(&req);
        // Only row 0 passes (c=200). It produces at most a few hits.
        assert!(crystal.bus.resonance.cycles_used <= 1);
    }

    /// Build a BindSpace of `n` rows with caller-supplied content fingerprints.
    /// Meta confidence set to (200, 200) so everything passes the prefilter.
    fn bindspace_with_content(rows: &[[u64; WORDS_PER_FP]]) -> BindSpace {
        use lance_graph_contract::qualia::QualiaI4_16D;
        let q = QualiaI4_16D::ZERO;
        let mut builder = BindSpaceBuilder::new(rows.len());
        for (idx, content) in rows.iter().enumerate() {
            let meta = MetaWord::new(
                (idx as u8).wrapping_add(1),
                (idx as u8).wrapping_add(1),
                200,
                200,
                5,
            );
            builder = builder.push(content, meta, 0, q, 0, 0);
        }
        builder.build()
    }

    #[test]
    fn content_hamming_finds_similar_rows() {
        // Two rows with near-identical content (differ in only 4 bits)
        // → resonance ≈ 0.9998, well above any style threshold.
        let mut a = [0u64; WORDS_PER_FP];
        for i in 0..250 {
            a[i / 64] |= 1u64 << (i % 64);
        }
        let mut b = a;
        b[0] ^= 0xF; // 4-bit difference → Hamming = 4
                     // A third row with substantially different content.
        let mut c = [0u64; WORDS_PER_FP];
        for i in 8000..8250 {
            c[i / 64] |= 1u64 << (i % 64);
        }

        let bs = Arc::new(bindspace_with_content(&[a, b, c]));
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();

        let req = ShaderDispatch {
            rows: ColumnWindow::new(0, 3),
            meta_prefilter: MetaFilter::ALL,
            layer_mask: 0xFF,
            radius: u16::MAX,
            style: StyleSelector::Ordinal(auto_style::ANALYTICAL),
            ..Default::default()
        };
        let crystal = driver.dispatch(&req);
        // Top-k must contain at least one content-match hit (predicates=0x01).
        let content_hits: Vec<_> = crystal
            .bus
            .resonance
            .top_k
            .iter()
            .filter(|h| h.predicates & 0x01 != 0 && h.resonance > 0.0)
            .collect();
        assert!(
            !content_hits.is_empty(),
            "expected at least one content-match hit, got top_k={:?}",
            crystal.bus.resonance.top_k
        );
        // Similarity should be very high (differ in only 4/16384 bits).
        assert!(
            content_hits.iter().any(|h| h.resonance > 0.5),
            "content-match resonance should be > 0.5 for near-identical rows"
        );
    }

    #[test]
    fn content_hamming_skips_dissimilar() {
        // Two rows with ~10000 Hamming distance → resonance ≈ 0.39, which
        // is BELOW analytical threshold (0.85). Analytical must not emit
        // a content-match hit.
        let mut a = [0u64; WORDS_PER_FP];
        for i in 0..5000 {
            a[i / 64] |= 1u64 << (i % 64);
        }
        let mut b = [0u64; WORDS_PER_FP];
        for i in 8000..13000 {
            b[i / 64] |= 1u64 << (i % 64);
        }
        // Disjoint ranges → Hamming ≈ 10000.

        let bs = Arc::new(bindspace_with_content(&[a, b]));
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();

        let req = ShaderDispatch {
            rows: ColumnWindow::new(0, 2),
            meta_prefilter: MetaFilter::ALL,
            layer_mask: 0xFF,
            radius: u16::MAX,
            style: StyleSelector::Ordinal(auto_style::ANALYTICAL),
            ..Default::default()
        };
        let crystal = driver.dispatch(&req);
        let content_hits: Vec<_> = crystal
            .bus
            .resonance
            .top_k
            .iter()
            .filter(|h| h.predicates & 0x01 != 0 && h.resonance > 0.0)
            .collect();
        assert!(
            content_hits.is_empty(),
            "analytical style should not emit content hits when resonance < 0.85; got {:?}",
            content_hits
        );
    }

    #[test]
    fn content_hamming_respects_style_threshold() {
        // Design Hamming ≈ 5000 so resonance ≈ 0.695:
        //   * below analytical  (0.85) → 0 content hits
        //   * above creative    (0.35) → ≥ 1 content hits
        // a = bits [0..5000), b = bits [2500..7500) → overlap 2500 bits,
        // disjoint 2500+2500 = 5000, Hamming ≈ 5000.
        let mut a = [0u64; WORDS_PER_FP];
        for i in 0..5000 {
            a[i / 64] |= 1u64 << (i % 64);
        }
        let mut b = [0u64; WORDS_PER_FP];
        for i in 2500..7500 {
            b[i / 64] |= 1u64 << (i % 64);
        }

        // Use empty planes so the palette cascade produces no hits —
        // isolates the content pre-pass so it cannot be drowned out by
        // synthetic palette matches that dominate top-k truncate(8).
        let empty_planes = [[0u64; 64]; 8];
        let mk_driver = || {
            let bs = Arc::new(bindspace_with_content(&[a, b]));
            let sr = Arc::new(demo_semiring());
            CognitiveShaderBuilder::new()
                .bindspace(bs)
                .semiring(sr)
                .planes(empty_planes)
                .build()
        };
        let mk_req = |style_ord: u8| ShaderDispatch {
            rows: ColumnWindow::new(0, 2),
            meta_prefilter: MetaFilter::ALL,
            layer_mask: 0xFF,
            radius: u16::MAX,
            style: StyleSelector::Ordinal(style_ord),
            ..Default::default()
        };

        let strict = mk_driver().dispatch(&mk_req(auto_style::ANALYTICAL));
        let loose = mk_driver().dispatch(&mk_req(auto_style::CREATIVE));
        let strict_hits = strict
            .bus
            .resonance
            .top_k
            .iter()
            .filter(|h| h.predicates & 0x01 != 0 && h.resonance > 0.0)
            .count();
        let loose_hits = loose
            .bus
            .resonance
            .top_k
            .iter()
            .filter(|h| h.predicates & 0x01 != 0 && h.resonance > 0.0)
            .count();
        // Monotonicity: loosening the style cannot reduce the set of
        // content-match hits. This is the load-bearing invariant.
        assert!(
            strict_hits <= loose_hits,
            "creative (loose) should emit >= analytical (strict) content hits: strict={} loose={}",
            strict_hits,
            loose_hits
        );
        assert!(loose_hits > 0,
            "creative (threshold 0.35) should emit content hits for resonance ≈ 0.695\nloose top_k: {:?}",
            loose.bus.resonance.top_k);
    }

    #[test]
    fn sink_short_circuits_on_false() {
        struct Stop;
        impl ShaderSink for Stop {
            fn on_resonance(&mut self, _r: &ShaderResonance) -> bool {
                false
            }
        }

        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();

        let mut stop = Stop;
        let req = ShaderDispatch {
            rows: ColumnWindow::new(0, 4),
            ..Default::default()
        };
        let crystal = driver.dispatch_with_sink(&req, &mut stop);
        // Short-circuited → persisted_row is None, meta is default.
        assert!(crystal.persisted_row.is_none());
        assert_eq!(crystal.meta.confidence, 0.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Pillar-7 — α-front-to-back merge tests (B5)
    // ═══════════════════════════════════════════════════════════════════════

    use lance_graph_contract::cognitive_shader::ALPHA_COMPOSITE_DIMS;
    use lance_graph_contract::collapse_gate::ALPHA_SATURATION_THRESHOLD;

    /// Build hits inline with the given resonances. Each hit gets a unique
    /// `row` so the qualia closure can map row → distinct color.
    fn mk_hits(resonances: &[f32]) -> Vec<ShaderHit> {
        resonances
            .iter()
            .enumerate()
            .map(|(i, &r)| ShaderHit {
                row: i as u32,
                distance: 0,
                predicates: 0,
                _pad: 0,
                resonance: r,
                cycle_index: i as u32,
            })
            .collect()
    }

    /// Per-row qualia: row k → all-ones vector × (k+1) so we can tell which
    /// hits actually contributed to the composited color.
    fn qualia_with(rows: usize) -> Vec<[f32; QUALIA_DIMS]> {
        (0..rows)
            .map(|k| {
                let mut q = [0.0f32; QUALIA_DIMS];
                for slot in q.iter_mut() {
                    *slot = (k + 1) as f32;
                }
                q
            })
            .collect()
    }

    #[test]
    fn alpha_merge_terminates_early_when_saturated() {
        // Two hits at α=0.99 each. After hit #1: α_acc = 0.99, which
        // *equals* the default threshold (0.99) — strict `>` won't fire.
        // After hit #2: α_acc = 0.99 + 0.99·0.01 = 0.9999 > 0.99 → break.
        // So `hits_consumed` must be exactly 2 even though we passed 5.
        let hits = mk_hits(&[0.99, 0.99, 0.99, 0.99, 0.99]);
        let qualia = qualia_with(hits.len());
        let composite = alpha_front_to_back_composite(
            &hits,
            |row| &qualia[row as usize][..],
            ALPHA_SATURATION_THRESHOLD,
        );
        assert_eq!(
            composite.hits_consumed, 2,
            "early-ray-termination should fire after 2 hits at α=0.99 each, got {}",
            composite.hits_consumed
        );
        assert!(
            composite.saturated,
            "saturated flag should be set when α exceeds threshold"
        );
        assert!(
            composite.alpha_acc > ALPHA_SATURATION_THRESHOLD,
            "α_acc must exceed threshold at termination, got {}",
            composite.alpha_acc
        );
    }

    #[test]
    fn alpha_merge_respects_confidence_ordering() {
        // Two hits at the same modest α, identical qualia rows except
        // their magnitude differs. Front (high) should dominate; reversing
        // the order produces a *different* color (front-to-back is order
        // dependent — that is the load-bearing property under test).
        let hits_desc = mk_hits(&[0.5, 0.5]);
        let qualia = qualia_with(2);
        let asc_composite = alpha_front_to_back_composite(
            &hits_desc,
            |row| &qualia[row as usize][..], // row 0 → 1.0, row 1 → 2.0
            ALPHA_SATURATION_THRESHOLD,
        );
        // First hit weight = 0.5 · 1.0    = 0.5  → contributes 0.5 · 1.0 = 0.5
        // Second hit weight = 0.5 · 0.5   = 0.25 → contributes 0.25 · 2.0 = 0.5
        // color_acc[0] = 0.5 + 0.5 = 1.0
        // α_acc       = 0.75
        assert!(
            (asc_composite.color_acc[0] - 1.0).abs() < 1e-5,
            "expected color_acc[0] = 1.0 (front=row0 dominates first), got {}",
            asc_composite.color_acc[0]
        );

        // Reverse order: row 1 first (qualia × 2), row 0 second (qualia × 1)
        let mut hits_rev = hits_desc.clone();
        hits_rev[0].row = 1;
        hits_rev[1].row = 0;
        let rev_composite = alpha_front_to_back_composite(
            &hits_rev,
            |row| &qualia[row as usize][..],
            ALPHA_SATURATION_THRESHOLD,
        );
        // First hit weight 0.5 · 1.0 = 0.5  → 0.5 · 2.0 = 1.0
        // Second hit weight 0.5 · 0.5 = 0.25 → 0.25 · 1.0 = 0.25
        // color_acc[0] = 1.25, distinct from the 1.0 we got front-first.
        assert!(
            (rev_composite.color_acc[0] - 1.25).abs() < 1e-5,
            "reversed order should give color_acc[0] = 1.25, got {}",
            rev_composite.color_acc[0]
        );
        assert!(
            (asc_composite.color_acc[0] - rev_composite.color_acc[0]).abs() > 0.1,
            "front-to-back composite must be order-dependent; got identical results"
        );
    }

    #[test]
    fn alpha_merge_zero_hits_returns_default() {
        let hits: Vec<ShaderHit> = Vec::new();
        let composite = alpha_front_to_back_composite(
            &hits,
            |_row| -> &[f32] { &[] },
            ALPHA_SATURATION_THRESHOLD,
        );
        assert_eq!(composite.alpha_acc, 0.0);
        assert_eq!(composite.hits_consumed, 0);
        assert!(!composite.saturated);
        for &slot in composite.color_acc.iter() {
            assert_eq!(
                slot, 0.0,
                "zero-hits composite must be all-zero color, found {}",
                slot
            );
        }
        // Sanity: AlphaComposite::default() matches the zero-hits result.
        let default = AlphaComposite::default();
        assert_eq!(default.alpha_acc, composite.alpha_acc);
        assert_eq!(default.hits_consumed, composite.hits_consumed);
        assert_eq!(default.saturated, composite.saturated);
        assert_eq!(default.color_acc, composite.color_acc);
        // ALPHA_COMPOSITE_DIMS sanity probe — keeps the import live and
        // documents the active prefix size.
        assert!(ALPHA_COMPOSITE_DIMS >= QUALIA_DIMS);
    }

    #[test]
    fn alpha_merge_single_hit_dominates() {
        // One opaque hit (α = 1.0): the composite must equal that hit's
        // qualia exactly, with α_acc = 1.0 and saturated = true.
        let hits = mk_hits(&[1.0]);
        let qualia = qualia_with(1);
        let composite = alpha_front_to_back_composite(
            &hits,
            |row| &qualia[row as usize][..],
            ALPHA_SATURATION_THRESHOLD,
        );
        assert_eq!(composite.hits_consumed, 1);
        assert!(
            (composite.alpha_acc - 1.0).abs() < 1e-6,
            "α_acc must be 1.0 after one opaque hit, got {}",
            composite.alpha_acc
        );
        assert!(
            composite.saturated,
            "α=1.0 > 0.99 threshold → saturated must be true"
        );
        // qualia[0] = [1.0; QUALIA_DIMS]. weight = 1.0 · 1.0 = 1.0.
        // color_acc[0..QUALIA_DIMS] = qualia[0].
        for i in 0..QUALIA_DIMS {
            assert!(
                (composite.color_acc[i] - 1.0).abs() < 1e-6,
                "color_acc[{}] must equal qualia[0][{}] = 1.0, got {}",
                i,
                i,
                composite.color_acc[i]
            );
        }
        // Trailing slots [QUALIA_DIMS..ALPHA_COMPOSITE_DIMS) stay zero.
        for i in QUALIA_DIMS..ALPHA_COMPOSITE_DIMS {
            assert_eq!(
                composite.color_acc[i], 0.0,
                "color_acc[{}] beyond QUALIA_DIMS must be zero",
                i
            );
        }
    }

    /// Negative-space: existing merge modes (Bundle, Xor, Superposition)
    /// must not populate `alpha_composite`. Only AlphaFrontToBack does.
    /// The full top-K aggregation pipeline (entropy, std_dev, top_k) must
    /// also be unchanged across modes — we probe by running the same
    /// dispatch with no override and with each non-alpha override and
    /// asserting `alpha_composite` stays None and the cycle outputs match.
    #[test]
    fn existing_merge_modes_unchanged() {
        let bs = Arc::new(demo_bindspace());
        let sr = Arc::new(demo_semiring());
        let driver = CognitiveShaderBuilder::new()
            .bindspace(bs)
            .semiring(sr)
            .planes(demo_planes())
            .build();

        let baseline = ShaderDispatch {
            rows: ColumnWindow::new(0, 4),
            meta_prefilter: MetaFilter::ALL,
            layer_mask: 0xFF,
            radius: u16::MAX,
            style: StyleSelector::Ordinal(auto_style::ANALYTICAL),
            ..Default::default()
        };
        let baseline_crystal = driver.dispatch(&baseline);
        // No override → alpha_composite must be None (gate decides Bundle
        // by default for homeostatic paths, never AlphaFrontToBack).
        assert!(
            baseline_crystal.alpha_composite.is_none(),
            "default dispatch must not populate alpha_composite"
        );

        // Bundle override — the gate's Bundle path stays.
        let bundle_req = ShaderDispatch {
            merge_override: Some(MergeMode::Bundle),
            ..baseline
        };
        let bundle_crystal = driver.dispatch(&bundle_req);
        assert!(
            bundle_crystal.alpha_composite.is_none(),
            "MergeMode::Bundle override must not populate alpha_composite"
        );
        // Top-K outputs remain identical to baseline.
        assert_eq!(
            bundle_crystal.bus.resonance.hit_count,
            baseline_crystal.bus.resonance.hit_count
        );
        assert_eq!(
            bundle_crystal.bus.resonance.entropy.to_bits(),
            baseline_crystal.bus.resonance.entropy.to_bits()
        );

        // Xor override — same negative-space property.
        let xor_req = ShaderDispatch {
            merge_override: Some(MergeMode::Xor),
            ..baseline
        };
        let xor_crystal = driver.dispatch(&xor_req);
        assert!(
            xor_crystal.alpha_composite.is_none(),
            "MergeMode::Xor override must not populate alpha_composite"
        );
        assert_eq!(
            xor_crystal.bus.resonance.hit_count,
            baseline_crystal.bus.resonance.hit_count
        );

        // Superposition override — same.
        let super_req = ShaderDispatch {
            merge_override: Some(MergeMode::Superposition),
            ..baseline
        };
        let super_crystal = driver.dispatch(&super_req);
        assert!(
            super_crystal.alpha_composite.is_none(),
            "MergeMode::Superposition override must not populate alpha_composite"
        );

        // Positive control: AlphaFrontToBack override DOES populate it.
        let alpha_req = ShaderDispatch {
            merge_override: Some(MergeMode::AlphaFrontToBack),
            ..baseline
        };
        let alpha_crystal = driver.dispatch(&alpha_req);
        assert!(
            alpha_crystal.alpha_composite.is_some(),
            "MergeMode::AlphaFrontToBack override MUST populate alpha_composite"
        );
    }

    /// Sanity probe: `confidence_to_alpha` clamps NaN / out-of-range
    /// resonances to zero so a poisoned hit can't break the composite.
    #[test]
    fn confidence_to_alpha_clamps_pathological() {
        let hit_nan = ShaderHit {
            resonance: f32::NAN,
            ..Default::default()
        };
        let hit_inf = ShaderHit {
            resonance: f32::INFINITY,
            ..Default::default()
        };
        let hit_neg = ShaderHit {
            resonance: -0.5,
            ..Default::default()
        };
        let hit_big = ShaderHit {
            resonance: 5.0,
            ..Default::default()
        };
        assert_eq!(hit_nan.confidence_to_alpha(), 0.0);
        assert_eq!(hit_inf.confidence_to_alpha(), 0.0);
        assert_eq!(hit_neg.confidence_to_alpha(), 0.0);
        assert_eq!(hit_big.confidence_to_alpha(), 1.0);
    }
}

#[cfg(test)]
mod rung_ascent_tests {
    use super::rung_widened_layer_mask;
    use lance_graph_contract::cognitive_shader::{RungElevator, RungLevel};
    use lance_graph_contract::collapse_gate::GateDecision;

    // A narrow dispatch mask so union-widening is observable. The production
    // default is 0xFF (all 8 planes); at 0xFF elevation cannot widen further,
    // so the probe uses a single-plane base — mimics a Surface observation.
    const BASE_MASK: u8 = 0b0000_0001; // CAUSES only

    #[test]
    fn d_tri_6_rung_ascends_relaxes_and_cascade_mask_tracks() {
        let mut e = RungElevator::new(RungLevel::Surface);

        // (i) at base: mask is EXACTLY the dispatch mask — no regression.
        assert_eq!(
            rung_widened_layer_mask(e.base, e.level, BASE_MASK),
            BASE_MASK
        );

        // sustained BLOCK ascends. threshold = 2, so 6 BLOCKs climb 3 rungs:
        // Surface(0) → Shallow(1) → Contextual(2) → Analogical(3, Pearl L2).
        for _ in 0..6 {
            e.on_gate(GateDecision::BLOCK);
        }
        assert_eq!(e.level, RungLevel::Analogical);
        assert_eq!(e.level.pearl_level(), 2);

        // (ii) elevated across a Pearl boundary → strict SUPERSET of the dispatch
        // mask (widened, changed, never narrowed).
        let widened = rung_widened_layer_mask(e.base, e.level, BASE_MASK);
        assert_ne!(
            widened, BASE_MASK,
            "elevation must change the consulted mask"
        );
        assert_eq!(widened & BASE_MASK, BASE_MASK, "must be a superset");
        assert!(widened.count_ones() > BASE_MASK.count_ones(), "must widen");

        // sustained FLOW relaxes back to base (never below): Analogical(3) → base
        // Surface is 3 de-elevations = 6 FLOWs.
        for _ in 0..6 {
            e.on_gate(GateDecision::FLOW_BUNDLE);
        }
        assert_eq!(e.level, e.base);
        assert_eq!(e.level, RungLevel::Surface);

        // back at base → mask returns to identity.
        assert_eq!(
            rung_widened_layer_mask(e.base, e.level, BASE_MASK),
            BASE_MASK
        );
    }

    #[test]
    fn all_planes_dispatch_mask_saturates_widening_to_noop() {
        // At the ShaderDispatch default (0xFF), elevation can't widen further.
        let mut e = RungElevator::new(RungLevel::Surface);
        for _ in 0..8 {
            e.on_gate(GateDecision::BLOCK); // climb well past Pearl L3
        }
        assert_eq!(rung_widened_layer_mask(e.base, e.level, 0xFF), 0xFF);
    }
}
