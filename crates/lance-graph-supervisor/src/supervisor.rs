//! `CallcenterSupervisor` — ractor root actor with one-for-one supervision.
//!
//! Owns the per-G consumer actor tree. Spawns one actor per active G slot;
//! inert slots (DOLCE G=0, FMA G=5) are skipped — a `Route { g: 5 }` for
//! an inert G returns `SupervisorErr::InertG(5)`, not a panic.
//!
//! Spec: pr-g2-ractor-supervisor.md §2-§5
//! CC-2: lifecycle events go to `LifecycleAuditEvent`, NOT `UnifiedAuditEvent`.
//! CC-3: lifecycle audit chain uses `SuperDomain::System`.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ractor::{Actor, ActorProcessingErr, ActorRef, SupervisionEvent};

use crate::consumer_msg::{ConsumerEnvelope, ConsumerReply, HealthStatus};
use crate::error::SupervisorErr;

// ─── Public constants ─────────────────────────────────────────────────────────

/// Default per-consumer actor mailbox capacity (messages).
/// Override via `stack_profile.mailbox_capacity` in the consumer manifest.
pub const DEFAULT_MAILBOX_CAPACITY: usize = 1024;

/// Initial backoff on first crash (milliseconds).
const BACKOFF_INITIAL_MS: u64 = 100;
/// Maximum backoff cap.
const BACKOFF_CAP: Duration = Duration::from_secs(30);
/// Crash count threshold for escalation (operator alert).
const ESCALATION_CRASH_COUNT: u32 = 10;

// ─── Module info for a G slot ─────────────────────────────────────────────────

/// Static configuration for one G slot, read from the MODULE_TABLE at spawn.
#[derive(Clone, Debug)]
pub struct ModuleEntry {
    pub g: u32,
    pub version: u32,
    pub mailbox_capacity: Option<usize>,
    pub is_active: bool,
}

// ─── ConsumerSlot — live child state ─────────────────────────────────────────

/// Per-G runtime state kept in the supervisor.
pub struct ConsumerSlot {
    pub actor_ref: ActorRef<ConsumerEnvelope>,
    pub crash_count: u32,
    pub last_crash_ts: Option<Instant>,
}

impl ConsumerSlot {
    /// Exponential backoff delay for the next respawn.
    /// Formula: `min(100ms * 2^crash_count, 30s)`.
    pub fn backoff_delay(&self) -> Duration {
        let shifts = self.crash_count.min(30);
        let ms = BACKOFF_INITIAL_MS.saturating_mul(1u64.checked_shl(shifts).unwrap_or(u64::MAX));
        let delay = Duration::from_millis(ms.min(BACKOFF_CAP.as_millis() as u64));
        delay.min(BACKOFF_CAP)
    }
}

// ─── Supervisor messages ──────────────────────────────────────────────────────

/// Messages the supervisor actor accepts.
pub enum SupervisorMsg {
    /// Route a typed envelope to the actor owning G.
    DispatchToG {
        g: u32,
        version: u32,
        envelope: ConsumerEnvelope,
        reply: ractor::RpcReplyPort<Result<ConsumerReply, SupervisorErr>>,
    },
    /// Health check — returns summary of all live children.
    Health {
        reply: ractor::RpcReplyPort<SupervisorHealthSummary>,
    },
    /// Graceful shutdown — stops all children then stops supervisor.
    Shutdown,
    /// Internal: supervisor schedules a respawn of a dead child after backoff.
    RespawnG {
        g: u32,
        version: u32,
        crash_count: u32,
    },
}

// ─── Health summary ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ChildSummary {
    pub g: u32,
    pub version: u32,
    pub crash_count: u32,
    pub is_live: bool,
}

#[derive(Clone, Debug)]
pub struct SupervisorHealthSummary {
    pub children: Vec<ChildSummary>,
}

// ─── Supervisor state ─────────────────────────────────────────────────────────

pub struct SupervisorState {
    /// Live slots keyed by `(g, version)`.
    pub slots: HashMap<(u32, u32), ConsumerSlot>,
    /// Reverse index: `ActorId → (g, version)` for O(1) lookup in supervision events.
    pub reverse_index: HashMap<u64, (u32, u32)>,
    /// Static module table (loaded at `pre_start`).
    pub modules: Vec<ModuleEntry>,
}

impl SupervisorState {
    fn new(modules: Vec<ModuleEntry>) -> Self {
        Self {
            slots: HashMap::new(),
            reverse_index: HashMap::new(),
            modules,
        }
    }
}

// ─── CallcenterSupervisor ─────────────────────────────────────────────────────

/// Root supervisor actor. One-for-one supervision of per-G consumer actors.
///
/// Use `CallcenterSupervisor::new(modules)` to construct, then
/// `Actor::spawn(None, supervisor, ())` to start.
pub struct CallcenterSupervisor {
    /// Static module table seed (passed at construction).
    pub modules: Vec<ModuleEntry>,
}

impl CallcenterSupervisor {
    pub fn new(modules: Vec<ModuleEntry>) -> Self {
        Self { modules }
    }
}

impl Actor for CallcenterSupervisor {
    type Msg = SupervisorMsg;
    type State = SupervisorState;
    type Arguments = ();

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        tracing::info!(
            modules = self.modules.len(),
            "CallcenterSupervisor pre_start: seeding from MODULE_TABLE"
        );

        let mut state = SupervisorState::new(self.modules.clone());

        // Spawn one actor per active G slot; skip inert slots (Option A from spec §2.1).
        for entry in self.modules.clone() {
            if !entry.is_active {
                tracing::debug!(g = entry.g, "skipping inert G slot");
                continue;
            }
            spawn_consumer_actor(myself.clone(), &mut state, &entry).await?;
        }

        tracing::info!(
            live_slots = state.slots.len(),
            "CallcenterSupervisor started"
        );
        Ok(state)
    }

    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            SupervisorMsg::DispatchToG {
                g,
                version,
                envelope,
                reply,
            } => {
                match state.slots.get(&(g, version)) {
                    Some(slot) => {
                        // Full impl: forward envelope to child actor and relay reply.
                        // Skeleton: handle Health directly; others return DispatchNotImplemented.
                        match envelope {
                            ConsumerEnvelope::Health => {
                                let _ = slot.actor_ref.clone(); // suppress unused
                                let _ =
                                    reply.send(Ok(ConsumerReply::Health(HealthStatus::healthy())));
                            }
                            _ => {
                                let _ = reply.send(Err(SupervisorErr::DispatchNotImplemented));
                            }
                        }
                    }
                    None => {
                        // Determine if inert (known but inactive) or truly unknown.
                        let known_inert = state.modules.iter().any(|m| m.g == g && !m.is_active);
                        if known_inert {
                            tracing::debug!(g, "DispatchToG: inert slot returns InertG");
                        } else {
                            tracing::warn!(
                                g,
                                version,
                                "DispatchToG: unknown G slot returns InertG"
                            );
                        }
                        let _ = reply.send(Err(SupervisorErr::InertG(g)));
                    }
                }
            }

            SupervisorMsg::Health { reply } => {
                let children: Vec<ChildSummary> = state
                    .modules
                    .iter()
                    .map(|m| {
                        let slot = state.slots.get(&(m.g, m.version));
                        ChildSummary {
                            g: m.g,
                            version: m.version,
                            crash_count: slot.map(|s| s.crash_count).unwrap_or(0),
                            is_live: slot.is_some(),
                        }
                    })
                    .collect();
                let _ = reply.send(SupervisorHealthSummary { children });
            }

            SupervisorMsg::Shutdown => {
                tracing::info!("CallcenterSupervisor: graceful shutdown");
                for ((g, version), slot) in state.slots.drain() {
                    tracing::debug!(g, version, "stopping child actor");
                    slot.actor_ref.stop(None);
                }
                myself.stop(None);
            }

            SupervisorMsg::RespawnG {
                g,
                version,
                crash_count,
            } => {
                let entry = state
                    .modules
                    .iter()
                    .find(|m| m.g == g && m.version == version)
                    .cloned();

                let Some(entry) = entry else {
                    tracing::warn!(g, version, "RespawnG: module entry not found");
                    return Ok(());
                };

                if crash_count > ESCALATION_CRASH_COUNT {
                    tracing::error!(
                        g,
                        version,
                        crash_count,
                        "consumer actor crash threshold > 10 — NOT respawning; \
                         operator action required (send ResetCrashCount)"
                    );
                    return Ok(());
                }

                // Backoff: 100ms * 2^crash_count, capped at 30s.
                let delay = {
                    let shifts = crash_count.min(30);
                    let ms = BACKOFF_INITIAL_MS
                        .saturating_mul(1u64.checked_shl(shifts).unwrap_or(u64::MAX));
                    Duration::from_millis(ms.min(BACKOFF_CAP.as_millis() as u64)).min(BACKOFF_CAP)
                };

                tracing::warn!(
                    g,
                    version,
                    crash_count,
                    delay_ms = delay.as_millis(),
                    "scheduling consumer actor respawn"
                );

                // Use tokio to sleep the backoff, then respawn.
                let myself2 = myself.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    if let Err(e) = myself2.cast(SupervisorMsg::RespawnG {
                        g,
                        version,
                        crash_count: 0, // reset count after backoff succeeds
                    }) {
                        tracing::error!(g, version, ?e, "failed to cast RespawnG after backoff");
                    }
                });

                // Immediate spawn attempt (the tokio task above will re-cast with count=0
                // after the backoff if we need to retry).
                match spawn_consumer_actor(myself.clone(), state, &entry).await {
                    Ok(()) => {
                        if let Some(slot) = state.slots.get_mut(&(g, version)) {
                            slot.crash_count = crash_count;
                            slot.last_crash_ts = Some(Instant::now());
                        }
                        tracing::info!(g, version, crash_count, "consumer actor respawned");
                    }
                    Err(e) => {
                        tracing::error!(g, version, ?e, "respawn failed");
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_supervisor_evt(
        &self,
        myself: ActorRef<Self::Msg>,
        evt: SupervisionEvent,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        if let SupervisionEvent::ActorTerminated(cell, _, reason) = evt {
            let actor_id = cell.get_id().pid();
            if let Some(&(g, version)) = state.reverse_index.get(&actor_id) {
                tracing::warn!(
                    g,
                    version,
                    actor_id,
                    ?reason,
                    "consumer actor terminated; scheduling respawn"
                );
                let crash_count = state
                    .slots
                    .get(&(g, version))
                    .map(|s| s.crash_count + 1)
                    .unwrap_or(1);

                state.slots.remove(&(g, version));
                state.reverse_index.remove(&actor_id);

                myself.cast(SupervisorMsg::RespawnG {
                    g,
                    version,
                    crash_count,
                })?;
            }
        }
        Ok(())
    }
}

// ─── Spawn helper ─────────────────────────────────────────────────────────────

/// Spawn a consumer actor for `entry` as a linked child of `supervisor`.
/// Registers the slot in `state.slots` and `state.reverse_index`.
async fn spawn_consumer_actor(
    supervisor: ActorRef<SupervisorMsg>,
    state: &mut SupervisorState,
    entry: &ModuleEntry,
) -> Result<(), ActorProcessingErr> {
    let name = format!("consumer_g_{}", entry.g);

    // Spawn stub consumer actor (full impl wires ConcreteConsumerActor<MedcareBridge> etc.)
    let stub = StubConsumerActor { g: entry.g };
    let (actor_ref, _handle) =
        Actor::spawn_linked(Some(name.clone()), stub, (), supervisor.get_cell())
            .await
            .map_err(|e| {
                tracing::error!(g = entry.g, name, ?e, "failed to spawn consumer actor");
                ActorProcessingErr::from(format!("spawn consumer_g_{} failed: {e}", entry.g))
            })?;

    let actor_id = actor_ref.get_id().pid();
    state.slots.insert(
        (entry.g, entry.version),
        ConsumerSlot {
            actor_ref,
            crash_count: 0,
            last_crash_ts: None,
        },
    );
    state
        .reverse_index
        .insert(actor_id, (entry.g, entry.version));
    tracing::debug!(
        g = entry.g,
        version = entry.version,
        name,
        "consumer actor spawned"
    );
    Ok(())
}

// ─── Stub consumer actor (proof-of-concept / medcare skeleton) ────────────────

/// Minimal `ractor::Actor` impl used for the per-G slot skeleton.
/// Full implementations will be `ConsumerActor<MedcareBridge>` etc.
pub struct StubConsumerActor {
    pub g: u32,
}

pub struct StubConsumerState;

impl Actor for StubConsumerActor {
    type Msg = ConsumerEnvelope;
    type State = StubConsumerState;
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        tracing::debug!(g = self.g, "StubConsumerActor pre_start");
        Ok(StubConsumerState)
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            ConsumerEnvelope::Health => {
                tracing::debug!(g = self.g, "StubConsumerActor: Health ping handled");
            }
            _ => {
                tracing::debug!(
                    g = self.g,
                    "StubConsumerActor: unhandled message variant (noop)"
                );
            }
        }
        Ok(())
    }
}
