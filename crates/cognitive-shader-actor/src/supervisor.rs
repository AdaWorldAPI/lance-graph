//! # ShaderSupervisor — one-for-one restart policy (plan §6)
//!
//! [`ShaderSupervisor`] manages a collection of [`CognitiveShaderActor`] children
//! under the OTP-inspired policy described in the integration plan §6 supervisor
//! topology:
//!
//! ```text
//! ShaderSupervisor (one-for-one, max 5 restarts in 60 s)
//! ├── ThinkingEngineActor
//! ├── CausalEdgeActor
//! ├── DeepNSMActor
//! ├── HolographActor
//! └── CognitiveShaderDriverActor
//! ```
//!
//! Policy parameters are sourced from the contract crate types:
//! - [`RestartBackoff`] — exponential back-off between restarts.
//! - [`SupervisionPolicy`] — which children to restart on failure.
//!
//! ## Sprint status
//!
//! Scaffold — Sprint 2 wires real ractor supervisor calls. The one public method
//! [`ShaderSupervisor::spawn_child`] returns `unimplemented!("LG-2 stub — Sprint 2")`.
//!
//! [`CognitiveShaderActor`]: crate::actor::CognitiveShaderActor
//! [`RestartBackoff`]: lance_graph_contract::actor::RestartBackoff
//! [`SupervisionPolicy`]: lance_graph_contract::actor::SupervisionPolicy

use lance_graph_contract::actor::{RestartBackoff, SupervisionPolicy};

// ---------------------------------------------------------------------------
// ShaderSupervisor
// ---------------------------------------------------------------------------

/// Supervisor that owns the lifecycle of all shader children (plan §6).
///
/// Holds the restart policy ([`SupervisionPolicy`]) and back-off configuration
/// ([`RestartBackoff`]) sourced from `lance-graph-contract::actor`. In Sprint 2
/// this struct becomes a `ractor::SupervisionEvent` handler that implements the
/// one-for-one restart loop.
pub struct ShaderSupervisor {
    /// Which siblings to restart when a child fails.
    ///
    /// Defaults to `OneForOne` matching the plan §6 topology ("restart only the
    /// failed child").
    pub policy: SupervisionPolicy,

    /// Exponential back-off between restart attempts.
    ///
    /// Defaults: 5 restarts in 60 s window, base 100 ms, cap 30 s (matches
    /// [`RestartBackoff::default`]).
    pub backoff: RestartBackoff,
}

impl Default for ShaderSupervisor {
    /// Returns a supervisor configured with the plan §6 defaults:
    /// one-for-one policy, 5 restarts in 60 s, 100 ms base back-off.
    fn default() -> Self {
        Self {
            policy: SupervisionPolicy::OneForOne,
            backoff: RestartBackoff::default(),
        }
    }
}

impl ShaderSupervisor {
    /// Construct a supervisor with explicit policy and back-off.
    ///
    /// In most cases [`ShaderSupervisor::default`] is the right entry point.
    /// Use this constructor when a non-default policy (e.g. `OneForAll`) or a
    /// tighter back-off window is required for a specific deployment.
    pub fn new(policy: SupervisionPolicy, backoff: RestartBackoff) -> Self {
        Self { policy, backoff }
    }

    /// Spawn a child shader actor under this supervisor's policy.
    ///
    /// Accepts a pre-built [`CognitiveShaderActor<S>`] and the [`Arc<S>`] argument
    /// that ractor passes to `pre_start`. Returns the actor handle on success.
    ///
    /// **Sprint 2 stub** — real ractor spawn + supervision registration is wired in
    /// Sprint 2. The return type will be `ractor::ActorRef<ShaderMessage<S::Payload>>`.
    ///
    /// [`CognitiveShaderActor<S>`]: crate::actor::CognitiveShaderActor
    /// [`Arc<S>`]: std::sync::Arc
    pub fn spawn_child(&self, _name: &str) -> ! {
        unimplemented!("LG-2 stub — Sprint 2")
    }
}
