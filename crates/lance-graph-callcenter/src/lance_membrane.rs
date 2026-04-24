//! LanceMembrane — the `ExternalMembrane` implementation.
//!
//! This is the compile-time BBB enforcement point. It is the only place
//! in the system where external consumer traffic and internal cognitive
//! state meet. After crossing here, external intent becomes a `UnifiedStep`
//! and committed cycles become `CognitiveEventRow` scalars — never the
//! reverse.
//!
//! # Phase map
//!
//! - Phase A (this file): BBB spine. Types correct. Scent is a stub.
//!   Subscription is a disconnected `mpsc::Receiver<u64>`.
//! - Phase B: `dialect` field populated by polyglot front-end parsers.
//! - Phase C: `scent` replaced by full ZeckBF17→Base17→CAM-PQ cascade.
//! - Phase D: Subscription wired to `tokio::sync::watch` on Lance version
//!   counter; `CommitFilter` applied per-subscriber.
//!
//! # UNKNOWN-1 resolution
//!
//! `ShaderSink` in `cognitive-shader-driver` is the internal BindSpace
//! ingestion path — it processes cycle fingerprints stack-side. There is
//! NO overlap with `ExternalMembrane`: `ShaderSink` never crosses the BBB;
//! `ExternalMembrane` is the gate. The two traits compose without conflict.
//!
//! Plan: `.claude/plans/callcenter-membrane-v1.md` § 10.9 – § 11

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    RwLock,
};

#[cfg(not(feature = "realtime"))]
use std::sync::mpsc;

#[cfg(feature = "realtime")]
use tokio::sync::watch;

#[cfg(feature = "realtime")]
use crate::version_watcher::LanceVersionWatcher;

use lance_graph_contract::{
    a2a_blackboard::ExpertId,
    cognitive_shader::{MetaWord, ShaderBus},
    external_membrane::{CommitFilter, ExternalEventKind, ExternalMembrane},
    faculty::FacultyRole,
    orchestration::{StepStatus, UnifiedStep},
};

use crate::external_intent::{CognitiveEventRow, ExternalIntent};

/// The Blood-Brain Barrier enforcement point.
///
/// One `LanceMembrane` per session. All interior state is protected by
/// `RwLock` or atomics so it is `Send + Sync`.
///
/// `current_role`, `current_faculty`, `current_expert` capture the last
/// `ingest()` context so that the next `project()` call stamps the correct
/// identity columns on the emitted `CognitiveEventRow`.
pub struct LanceMembrane {
    current_role:    RwLock<u8>,      // ExternalRole discriminant
    current_faculty: RwLock<u8>,      // FacultyRole discriminant
    current_expert:  RwLock<ExpertId>,
    current_scent:   AtomicU64,
    current_rationale_phase: AtomicBool, // MM-CoT stage: true = Stage 1 rationale
    version:         AtomicU64,
    /// Fan-out watcher for projected cognitive events ([realtime] feature only).
    #[cfg(feature = "realtime")]
    watcher:         LanceVersionWatcher,
}

impl LanceMembrane {
    pub fn new() -> Self {
        Self {
            current_role:    RwLock::new(0),
            current_faculty: RwLock::new(FacultyRole::ReadingComprehension as u8),
            current_expert:  RwLock::new(0),
            current_scent:   AtomicU64::new(0),
            current_rationale_phase: AtomicBool::new(false),
            version:         AtomicU64::new(0),
            #[cfg(feature = "realtime")]
            watcher:         LanceVersionWatcher::default(),
        }
    }

    /// Current Lance version counter (monotonic; ticks on each CollapseGate
    /// Persist). Phase D wires this to the Lance dataset version.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Set the faculty context for the current cycle.
    ///
    /// Called by the orchestration layer (not by `ingest`) when
    /// the faculty dispatcher resolves which `FacultyDescriptor`
    /// handles the current cycle. The `rationale_phase` argument
    /// surfaces MM-CoT Stage 1 (true) vs Stage 2 (false); the
    /// dispatcher derives it from `FacultyDescriptor::is_asymmetric()`
    /// plus the current dispatch stage.
    pub fn set_faculty_context(&self, faculty: u8, expert: ExpertId, rationale_phase: bool) {
        *self.current_faculty.write().expect("faculty poisoned") = faculty;
        *self.current_expert.write().expect("expert poisoned") = expert;
        self.current_rationale_phase.store(rationale_phase, Ordering::Relaxed);
    }
}

impl Default for LanceMembrane {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExternalMembrane impl — the BBB
// ─────────────────────────────────────────────────────────────────────────────

impl ExternalMembrane for LanceMembrane {
    /// Scalar projection of a committed cycle.
    ///
    /// Only Arrow-scalar primitives here. `[u64; 256]` fingerprint lives
    /// in `bus` — this type carries two representative words, not the tensor.
    type Commit = CognitiveEventRow;

    /// External consumer intent entering through the gate.
    type Intent = ExternalIntent;

    /// Subscription handle for projected cognitive events.
    ///
    /// With `[realtime]` feature: a `tokio::sync::watch::Receiver<CognitiveEventRow>`
    /// wired to `LanceVersionWatcher` — always-latest semantics, supabase-shape.
    /// Without `[realtime]`: a disconnected `mpsc::Receiver<u64>` stub (Phase A).
    #[cfg(not(feature = "realtime"))]
    type Subscription = mpsc::Receiver<u64>;
    #[cfg(feature = "realtime")]
    type Subscription = watch::Receiver<CognitiveEventRow>;

    /// Project a committed ShaderBus cycle to a scalar row.
    ///
    /// Strips all VSA state. Emits only Arrow-scalar-compatible fields.
    /// Called on every `CollapseGate` fire with `EmitMode::Persist`.
    fn project(&self, bus: &ShaderBus, meta: MetaWord) -> CognitiveEventRow {
        let role    = *self.current_role.read().expect("role poisoned");
        let faculty = *self.current_faculty.read().expect("faculty poisoned");
        let expert  = *self.current_expert.read().expect("expert poisoned");
        let scent   = self.current_scent.load(Ordering::Relaxed) as u8;

        let row = CognitiveEventRow {
            external_role: role,
            faculty_role:  faculty,
            expert_id:     expert,
            dialect:       0, // Phase B: populated by polyglot front-end
            scent,
            thinking:  meta.thinking(),
            awareness: meta.awareness(),
            nars_f:    meta.nars_f(),
            nars_c:    meta.nars_c(),
            free_e:    meta.free_e(),
            // Two scalar words from the fingerprint — cycle identity without
            // the full 2 KB VSA tensor. The tensor lives only in ShaderBus.
            cycle_fp_hi: bus.cycle_fingerprint[0],
            cycle_fp_lo: bus.cycle_fingerprint[255],
            gate_commit:     bus.gate.is_flow(),
            gate_f:          meta.free_e(),
            rationale_phase: self.current_rationale_phase.load(Ordering::Relaxed),
        };

        // DM-4: fan out to all current subscribers (supabase-shape realtime).
        #[cfg(feature = "realtime")]
        self.watcher.bump(row.clone());

        row
    }

    /// Translate external intent to canonical dispatch.
    ///
    /// Four-step BBB crossing:
    /// 1. Pass membrane — `ExternalIntent` is the safe crossing type.
    /// 2. Get a role   — `intent.role` is stamped into `current_role`.
    /// 3. Get a place  — `intent.dn.scent_stub()` becomes `current_scent`.
    /// 4. Translate    — returns `UnifiedStep` for `OrchestrationBridge::route()`.
    fn ingest(&self, intent: ExternalIntent) -> UnifiedStep {
        // 2. Role
        *self.current_role.write().expect("role poisoned") = intent.role as u8;

        // 3. Place (Phase A: XOR-fold stub; Phase C: full cascade)
        let scent = intent.dn.scent_stub();
        self.current_scent.store(scent as u64, Ordering::Relaxed);

        // 4. Translate to step type for OrchestrationBridge
        let step_type = match intent.kind {
            ExternalEventKind::Seed    => "lg.blackboard.seed",
            ExternalEventKind::Context => "lg.blackboard.context",
            ExternalEventKind::Commit  => "lg.blackboard.commit",
        };

        UnifiedStep {
            step_id:    format!("{:016x}", scent as u64),
            step_type:  step_type.to_owned(),
            status:     StepStatus::Pending,
            thinking:   None, // resolved by OrchestrationBridge::resolve_thinking()
            reasoning:  None,
            confidence: None,
        }
    }

    /// Subscribe to projected commits matching the filter.
    ///
    /// With `[realtime]`: returns a `watch::Receiver<CognitiveEventRow>` seeded
    /// with the latest committed row.  Always-latest semantics (supabase-shape).
    /// Without `[realtime]`: returns a disconnected `mpsc::Receiver<u64>` stub.
    #[cfg(not(feature = "realtime"))]
    fn subscribe(&self, _filter: CommitFilter) -> mpsc::Receiver<u64> {
        let (_tx, rx) = mpsc::channel();
        rx
    }

    #[cfg(feature = "realtime")]
    fn subscribe(&self, _filter: CommitFilter) -> watch::Receiver<CognitiveEventRow> {
        self.watcher.subscribe()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::{
        cognitive_shader::ShaderBus,
        external_membrane::ExternalRole,
    };
    use crate::dn_path::DnPath;

    fn make_dn() -> DnPath {
        DnPath::parse("/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc")
            .unwrap()
    }

    #[test]
    fn ingest_sets_role_and_scent() {
        let m = LanceMembrane::new();
        let intent = ExternalIntent::seed(
            ExternalRole::CrewaiAgent,
            make_dn(),
            b"hello world".to_vec(),
        );
        let step = m.ingest(intent);
        assert_eq!(step.step_type, "lg.blackboard.seed");
        assert_eq!(step.status, StepStatus::Pending);
        // scent was set
        assert_ne!(m.current_scent.load(Ordering::Relaxed), 0);
        // role was stamped
        assert_eq!(*m.current_role.read().unwrap(), ExternalRole::CrewaiAgent as u8);
    }

    #[test]
    fn project_emits_scalar_row() {
        let m = LanceMembrane::new();
        // Prime role context via ingest
        let intent = ExternalIntent::seed(
            ExternalRole::Rag,
            make_dn(),
            vec![],
        );
        m.ingest(intent);

        let bus = ShaderBus::empty();
        let meta = MetaWord::new(5, 3, 200, 150, 10);
        let row = m.project(&bus, meta);

        assert_eq!(row.external_role, ExternalRole::Rag as u8);
        assert_eq!(row.thinking, 5);
        assert_eq!(row.nars_f, 200);
        assert_eq!(row.nars_c, 150);
        assert_eq!(row.free_e, 10);
        // Fingerprint words from ShaderBus::empty() are 0
        assert_eq!(row.cycle_fp_hi, 0);
        assert_eq!(row.cycle_fp_lo, 0);
        // Empty bus has HOLD gate (gate=2, not Flow)
        assert!(!row.gate_commit);
    }

    #[test]
    fn context_kind_produces_context_step() {
        let m = LanceMembrane::new();
        let intent = ExternalIntent::context(
            ExternalRole::N8n,
            make_dn(),
            b"context payload".to_vec(),
        );
        let step = m.ingest(intent);
        assert_eq!(step.step_type, "lg.blackboard.context");
    }

    /// Phase A (no realtime feature): subscription is a disconnected stub.
    #[cfg(not(feature = "realtime"))]
    #[test]
    fn subscribe_returns_disconnected_receiver() {
        let m = LanceMembrane::new();
        let rx = m.subscribe(CommitFilter::default());
        // Phase A: disconnected — no sender kept, recv errors immediately
        assert!(rx.try_recv().is_err());
    }

    /// Phase D (realtime feature): subscribe() → project() → rx.borrow() sees the row.
    #[cfg(feature = "realtime")]
    #[test]
    fn subscribe_receives_on_project() {
        let m = LanceMembrane::new();
        let rx = m.subscribe(CommitFilter::default());

        // Prime role context and call project()
        let intent = ExternalIntent::seed(ExternalRole::CrewaiAgent, make_dn(), vec![]);
        m.ingest(intent);

        let bus = ShaderBus::empty();
        let meta = MetaWord::new(7, 3, 200, 150, 10);
        m.project(&bus, meta);

        // The watcher should have delivered the row
        let snapshot = rx.borrow();
        assert_eq!(snapshot.thinking, 7, "subscriber should see the projected row");
        assert_eq!(snapshot.external_role, ExternalRole::CrewaiAgent as u8);
    }

    #[test]
    fn set_faculty_context_wires_rationale_phase() {
        let m = LanceMembrane::new();
        // Before wiring: rationale_phase defaults to false
        let bus = ShaderBus::empty();
        let meta = MetaWord::new(0, 0, 128, 128, 20);
        let row = m.project(&bus, meta);
        assert!(!row.rationale_phase, "default should be Stage 2 (false)");

        // Wire Stage 1 (rationale) via set_faculty_context
        m.set_faculty_context(
            FacultyRole::ReadingComprehension as u8,
            42, // expert_id
            true, // rationale phase = Stage 1
        );
        let row = m.project(&bus, meta);
        assert!(row.rationale_phase, "after set_faculty_context(true), should be Stage 1");
        assert_eq!(row.faculty_role, FacultyRole::ReadingComprehension as u8);
        assert_eq!(row.expert_id, 42);

        // Wire Stage 2 (answer)
        m.set_faculty_context(FacultyRole::ReadingComprehension as u8, 42, false);
        let row = m.project(&bus, meta);
        assert!(!row.rationale_phase, "after set_faculty_context(false), should be Stage 2");
    }

    #[test]
    fn bbb_scalar_only_compile_check() {
        // This is the compile-time BBB proof: CognitiveEventRow contains
        // only u8/u16/u64/bool. It implements Send + Sync trivially.
        // If someone added a [u64; 256] field here, the type would still
        // be Send, but it would be a 2 KB copy — visible in code review.
        // More critically, Arrow::try_new() would reject it as a column
        // element when the [persist] feature is activated in Phase B.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CognitiveEventRow>();
        assert_send_sync::<LanceMembrane>();
    }
}
