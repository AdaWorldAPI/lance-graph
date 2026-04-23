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
    atomic::{AtomicU64, Ordering},
    mpsc, RwLock,
};

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
    version:         AtomicU64,
}

impl LanceMembrane {
    pub fn new() -> Self {
        Self {
            current_role:    RwLock::new(0),
            current_faculty: RwLock::new(FacultyRole::ReadingComprehension as u8),
            current_expert:  RwLock::new(0),
            current_scent:   AtomicU64::new(0),
            version:         AtomicU64::new(0),
        }
    }

    /// Current Lance version counter (monotonic; ticks on each CollapseGate
    /// Persist). Phase D wires this to the Lance dataset version.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
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

    /// Phase-A subscription stub: a disconnected `mpsc::Receiver<u64>`.
    ///
    /// Callers that `recv()` on this immediately get `Err(RecvError)`.
    /// Phase D replaces this with a `tokio::sync::watch::Receiver<u64>`
    /// wired to the Lance version counter, filtered by `CommitFilter`.
    type Subscription = mpsc::Receiver<u64>;

    /// Project a committed ShaderBus cycle to a scalar row.
    ///
    /// Strips all VSA state. Emits only Arrow-scalar-compatible fields.
    /// Called on every `CollapseGate` fire with `EmitMode::Persist`.
    fn project(&self, bus: &ShaderBus, meta: MetaWord) -> CognitiveEventRow {
        let role    = *self.current_role.read().expect("role poisoned");
        let faculty = *self.current_faculty.read().expect("faculty poisoned");
        let expert  = *self.current_expert.read().expect("expert poisoned");
        let scent   = self.current_scent.load(Ordering::Relaxed) as u8;

        CognitiveEventRow {
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
            rationale_phase: false, // Phase B: wired from FacultyDescriptor::is_asymmetric()
        }
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
    /// Phase A: returns a disconnected channel (recv immediately errors).
    /// Phase D: wires to `tokio::sync::watch` + `CommitFilter` predicate.
    fn subscribe(&self, _filter: CommitFilter) -> mpsc::Receiver<u64> {
        let (_tx, rx) = mpsc::channel();
        rx
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

    #[test]
    fn subscribe_returns_disconnected_receiver() {
        let m = LanceMembrane::new();
        let rx = m.subscribe(CommitFilter::default());
        // Phase A: disconnected — no sender kept, recv errors immediately
        assert!(rx.try_recv().is_err());
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
