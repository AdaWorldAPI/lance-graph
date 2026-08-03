//! ⚠ STATUS: UNWIRED STUB — never spawned, owns no bridge, emits no audit.
//! Every `handle` arm is a `tracing::debug!` plus a `// TODO`; the supervisor
//! tree spawns `supervisor::StubConsumerActor` instead (supervisor.rs:368).
//! Retained as the worked SHAPE for a generic `ConsumerActor<P: PortSpec>`; it
//! is public API (re-exported from `lib.rs`), so removal is a breaking change.
//!
//! Healthcare-specific surface is two constants (`MEDCARE_G`,
//! `MEDCARE_VERSION`), one env-var name (`MEDCARE_AUDIT_SALT`) and the type
//! names — the message handling itself is generic `ConsumerEnvelope` logging.
//!
//! What it would become: the actor half of the collapse the bridge already
//! made (`MedcareBridge` → `UnifiedBridge<HealthcarePort>`, lance-graph#570).
//! `MEDCARE_G` / `MEDCARE_VERSION` / the actor name / the audit-salt env var
//! are spawn parameters a `PortSpec` already carries (`P::NAMESPACE`,
//! `P::BRIDGE_ID`, `P::class_id`), so the generic form re-derives this file as
//! a one-line type alias. The healthcare VOCABULARY is owned upstream and is
//! not duplicated here: capability + classid table in `ogar_vocab`
//! (`healthcare_actions`, `ports::HealthcarePort`), public ontology reference
//! in OGAR `ogar-obo` (MONDO/HPO/Uberon/PATO/RO + the agnostic xref crosswalk).
//!
//! Original v1 intent (sprint-7 spec `pr-g2-ractor-supervisor.md` §8), none of
//! it built: own a `UnifiedBridge`, answer `ConsumerEnvelope`, emit
//! `UnifiedAuditEvent` per authorization decision, seed the `AuditChain` from
//! `MEDCARE_AUDIT_SALT` (hex u64) and later from HSM. The env var is read in
//! `pre_start` and logged; it is wired to nothing else.

use ractor::{Actor, ActorProcessingErr, ActorRef};
use tracing;

use crate::consumer_msg::{ConsumerEnvelope, HealthStatus};

/// G-slot constant for MedCare.
pub const MEDCARE_G: u32 = 2;
/// Version constant for MedCare (HEALTHCARE_V1).
pub const MEDCARE_VERSION: u32 = 1;

// ─── Actor state ──────────────────────────────────────────────────────────────

pub struct MedcareState {
    /// Number of requests handled since spawn (diagnostic only for skeleton).
    pub handled: u64,
}

// ─── MedcareConsumerActor ─────────────────────────────────────────────────────

/// Concrete consumer actor for G=2 (Healthcare / MedCare).
///
/// Full implementation wires `UnifiedBridge<MedcareBridge>` with an
/// `AuditChain` seeded from `MEDCARE_AUDIT_SALT` env var (sprint-7) or HSM
/// (sprint-8). The actor name is always `"consumer_g_2"` — survives respawn.
pub struct MedcareConsumerActor;

impl Actor for MedcareConsumerActor {
    type Msg = ConsumerEnvelope;
    type State = MedcareState;
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let salt_hex = std::env::var("MEDCARE_AUDIT_SALT").unwrap_or_else(|_| "0".to_string());
        let salt = u64::from_str_radix(salt_hex.trim_start_matches("0x"), 16).unwrap_or(0);

        tracing::info!(
            g = MEDCARE_G,
            version = MEDCARE_VERSION,
            audit_salt = format!("{salt:#018x}"),
            "MedcareConsumerActor pre_start"
        );

        // TODO (sprint-8): construct UnifiedBridge<MedcareBridge> with AuditChain here.
        // let bridge = UnifiedBridge::new(medcare_bridge, SuperDomain::Healthcare, salt, sink);

        Ok(MedcareState { handled: 0 })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        state.handled += 1;

        match msg {
            ConsumerEnvelope::Health => {
                // Health is handled by the supervisor routing the reply port
                // directly; this branch is for when the actor is called directly.
                tracing::debug!(g = MEDCARE_G, handled = state.handled, "Health ping");
            }

            ConsumerEnvelope::Dispatch(req) => {
                tracing::debug!(
                    g = MEDCARE_G,
                    tenant = req.tenant_id,
                    "MedcareConsumerActor: Dispatch (stub — wiring TODO)"
                );
                // TODO: route through UnifiedBridge::authorize(AuthOp::Act, ...)
                //       emit UnifiedAuditEvent via AuditChain
            }

            ConsumerEnvelope::Ingest(req) => {
                tracing::debug!(
                    g = MEDCARE_G,
                    tenant = req.tenant_id,
                    records = req.records.len(),
                    "MedcareConsumerActor: Ingest (stub)"
                );
            }

            ConsumerEnvelope::Qualia(req) => {
                tracing::debug!(
                    g         = MEDCARE_G,
                    tenant    = req.tenant_id,
                    key       = %req.qualia_key,
                    "MedcareConsumerActor: Qualia (stub)"
                );
            }

            ConsumerEnvelope::Styles(req) => {
                tracing::debug!(
                    g = MEDCARE_G,
                    tenant = req.tenant_id,
                    "MedcareConsumerActor: Styles (stub)"
                );
            }

            ConsumerEnvelope::Tensors(_req) => {
                tracing::debug!(
                    g = MEDCARE_G,
                    "MedcareConsumerActor: Tensors lab arm (stub)"
                );
            }

            ConsumerEnvelope::Calibrate(_req) => {
                tracing::debug!(
                    g = MEDCARE_G,
                    "MedcareConsumerActor: Calibrate lab arm (stub)"
                );
            }

            ConsumerEnvelope::Probe(_req) => {
                tracing::debug!(g = MEDCARE_G, "MedcareConsumerActor: Probe lab arm (stub)");
            }
        }

        Ok(())
    }
}

/// Helper: build a `HealthStatus` for the medcare actor.
pub fn medcare_health(handled: u64) -> HealthStatus {
    HealthStatus {
        ok: true,
        detail: format!("MedcareConsumerActor ok; handled={handled}"),
    }
}
