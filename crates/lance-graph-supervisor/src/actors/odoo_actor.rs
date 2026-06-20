//! `OdooConsumerActor` — G=50, ODOO_V1 consumer actor (proof-of-concept).
//!
//! Second concrete `Actor` impl after `MedcareConsumerActor`. Mirrors that
//! card 1:1 — same envelope arms, same audit-salt env handshake, same
//! stub-with-TODO shape — so the `odoo-rs` consumer crate, when it lands,
//! only has to swap the body of each handler arm.
//!
//! Ontology positioning: ODOO inherits from FIBOFND (financial-foundations)
//! per `modules/odoo/manifest.yaml`; the registry resolves OGIT URIs through
//! `OdooBridge` (locks to namespace `"Odoo"`); RBAC routes through
//! `odoo_policy` (declared in the manifest, defined in the consumer's
//! policy table).
//!
//! For v1, this is a skeleton:
//! - `UnifiedBridge<OdooBridge>` wiring is a `// TODO` (parallels the
//!   medcare TODO awaiting HSM-salt wiring).
//! - `ConsumerEnvelope::Health` is fully handled.
//! - All other arms log a diagnostic and return Ok.
//!
//! Audit chain initialization: accepts env var `ODOO_AUDIT_SALT` (hex u64),
//! defaulting to `0` for dev. Production deployments override with a
//! per-tenant HSM-rooted value.

use ractor::{Actor, ActorProcessingErr, ActorRef};
use tracing;

use crate::consumer_msg::{ConsumerEnvelope, HealthStatus};

/// G-slot constant for Odoo (matches `CANONICAL_SLOTS` in
/// `lance-graph-contract/build.rs` and `OGIT::ODOO_V1.0`).
pub const ODOO_G: u32 = 50;
/// Version constant for Odoo (ODOO_V1, matches manifest `version: 1`).
pub const ODOO_VERSION: u32 = 1;

// ─── Actor state ──────────────────────────────────────────────────────────────

pub struct OdooState {
    /// Number of requests handled since spawn (diagnostic only for skeleton).
    pub handled: u64,
}

// ─── OdooConsumerActor ────────────────────────────────────────────────────────

/// Concrete consumer actor for G=50 (Odoo / ERP).
///
/// Full implementation wires `UnifiedBridge<OdooBridge>` with an
/// `AuditChain` seeded from `ODOO_AUDIT_SALT` env var (skeleton) or HSM
/// (production). The actor name is always `"consumer_g_50"` — survives respawn.
pub struct OdooConsumerActor;

impl Actor for OdooConsumerActor {
    type Msg = ConsumerEnvelope;
    type State = OdooState;
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let salt_hex = std::env::var("ODOO_AUDIT_SALT").unwrap_or_else(|_| "0".to_string());
        let salt = u64::from_str_radix(salt_hex.trim_start_matches("0x"), 16).unwrap_or(0);

        tracing::info!(
            g = ODOO_G,
            version = ODOO_VERSION,
            audit_salt = format!("{salt:#018x}"),
            "OdooConsumerActor pre_start"
        );

        // TODO: construct UnifiedBridge<OdooBridge> with AuditChain here once
        // the odoo-rs consumer crate lands. Parallel of the medcare TODO.
        // let bridge = UnifiedBridge::new(odoo_bridge, SuperDomain::WorkOrderBilling, salt, sink);

        Ok(OdooState { handled: 0 })
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
                tracing::debug!(g = ODOO_G, handled = state.handled, "Health ping");
            }

            ConsumerEnvelope::Dispatch(req) => {
                tracing::debug!(
                    g = ODOO_G,
                    tenant = req.tenant_id,
                    "OdooConsumerActor: Dispatch (stub — wiring TODO)"
                );
                // TODO: route through UnifiedBridge::authorize(AuthOp::Act, ...)
                //       emit UnifiedAuditEvent via AuditChain
            }

            ConsumerEnvelope::Ingest(req) => {
                tracing::debug!(
                    g = ODOO_G,
                    tenant = req.tenant_id,
                    records = req.records.len(),
                    "OdooConsumerActor: Ingest (stub)"
                );
            }

            ConsumerEnvelope::Qualia(req) => {
                tracing::debug!(
                    g      = ODOO_G,
                    tenant = req.tenant_id,
                    key    = %req.qualia_key,
                    "OdooConsumerActor: Qualia (stub)"
                );
            }

            ConsumerEnvelope::Styles(req) => {
                tracing::debug!(
                    g = ODOO_G,
                    tenant = req.tenant_id,
                    "OdooConsumerActor: Styles (stub)"
                );
            }

            ConsumerEnvelope::Tensors(_req) => {
                tracing::debug!(g = ODOO_G, "OdooConsumerActor: Tensors lab arm (stub)");
            }

            ConsumerEnvelope::Calibrate(_req) => {
                tracing::debug!(g = ODOO_G, "OdooConsumerActor: Calibrate lab arm (stub)");
            }

            ConsumerEnvelope::Probe(_req) => {
                tracing::debug!(g = ODOO_G, "OdooConsumerActor: Probe lab arm (stub)");
            }
        }

        Ok(())
    }
}

/// Helper: build a `HealthStatus` for the odoo actor.
pub fn odoo_health(handled: u64) -> HealthStatus {
    HealthStatus {
        ok: true,
        detail: format!("OdooConsumerActor ok; handled={handled}"),
    }
}
