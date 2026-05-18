//! # CognitiveShaderActor — ractor::Actor adapter (plan §6)
//!
//! [`CognitiveShaderActor<S>`] is a thin `ractor::Actor` wrapper around any type
//! that implements [`SupervisableShader`]. One actor instance per shader instance;
//! the supervisor restarts it on panic without affecting peer shaders.
//!
//! ## Generics
//!
//! - `S: SupervisableShader` — the concrete shader (e.g. `ThinkingEngineShader`).
//!   The associated type `S::Payload` becomes the actor's message payload type.
//!
//! ## State
//!
//! [`ShaderState<S>`] holds:
//! - `shader: Arc<S>` — shared reference so the actor can hand a clone to a
//!   future if needed in Sprint 2.
//! - `inflight: usize` — count of in-progress `Apply` messages; used by `Drain`
//!   to yield until the mailbox drains.
//!
//! ## Message handling (plan §6)
//!
//! | Message | Behaviour |
//! |---|---|
//! | `Apply` | `inflight += 1`; call `S::apply`; `inflight -= 1`; send result |
//! | `ApplyDelta` | call `S::apply_delta` |
//! | `Drain` | spin-yield while `inflight > 0`; call `S::drain`; reply; stop self |
//!
//! ## Sprint status
//!
//! All method bodies are stubs — `unimplemented!("LG-2 stub — Sprint 2")`. The
//! `ractor::Actor` trait signatures may need adjustment once Sprint 2 confirms the
//! exact ractor 0.14 API surface (see report for known gaps).
//!
//! [`SupervisableShader`]: lance_graph_contract::actor::SupervisableShader

use std::sync::Arc;

use lance_graph_contract::actor::SupervisableShader;

use crate::messages::ShaderMessage;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Runtime state held by a live `CognitiveShaderActor` instance.
///
/// Plan §6: "State holds `Arc<S>` + `inflight: usize`."
pub struct ShaderState<S>
where
    S: SupervisableShader,
{
    /// The underlying shader. `Arc` so a future spawn in Sprint 2 can hold a
    /// clone without copying the shader implementation.
    pub shader: Arc<S>,

    /// Count of `Apply` messages currently being processed. Used by `Drain`
    /// to yield until the actor is idle before stopping.
    pub inflight: usize,
}

// ---------------------------------------------------------------------------
// Actor struct
// ---------------------------------------------------------------------------

/// A cognitive shader wrapped as a ractor actor (plan §6, Glue #4).
///
/// Instantiate one `CognitiveShaderActor<S>` per shader instance and spawn it
/// via `ractor::Actor::spawn`. The supervisor topology in [`crate::supervisor`]
/// manages collections of these under a one-for-one policy.
///
/// ### Type parameter
///
/// `S` must implement [`SupervisableShader`]; its associated `Payload` type is
/// forwarded to `ShaderMessage<S::Payload>` as the actor's message type.
pub struct CognitiveShaderActor<S>
where
    S: SupervisableShader,
{
    // Marker: the actor struct itself carries no runtime data; all state lives
    // in ShaderState<S> which ractor manages per-instance.
    _marker: std::marker::PhantomData<S>,
}

impl<S> CognitiveShaderActor<S>
where
    S: SupervisableShader,
{
    /// Construct the actor descriptor. The concrete shader is supplied as
    /// `Arguments` when calling `ractor::Actor::spawn`.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<S> Default for CognitiveShaderActor<S>
where
    S: SupervisableShader,
{
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ractor::Actor impl — Sprint 2 stub
// ---------------------------------------------------------------------------
//
// Plan §6 names four associated types and three async lifecycle methods:
//   type Msg = ShaderMessage<S::Payload>
//   type State = ShaderState<S>
//   type Arguments = Arc<S>
//   async fn pre_start(...)  -> Result<State, ActorProcessingErr>
//   async fn handle(...)     -> Result<(), ActorProcessingErr>
//
// The exact ractor 0.14 trait signature (crate version "*" in Cargo.toml) must
// be confirmed in Sprint 2 — in particular whether `handle` is `async fn` or
// takes a `BoxFuture`, and whether `ActorProcessingErr` is re-exported from the
// crate root or a sub-module. Bodies below are `unimplemented!` stubs.
//
// KNOWN GAP (report): ractor's `Actor` trait uses `async fn` syntax only with
// the `async-trait` macro pre-1.75, or bare `async fn` in trait in Rust ≥ 1.75
// with ractor ≥ 0.12. Confirm exact form before wiring.

#[allow(unused_variables)]
impl<S> CognitiveShaderActor<S>
where
    S: SupervisableShader,
    S::Error: Into<anyhow::Error>,
{
    /// Called once by the supervisor when the actor spawns or respawns (plan §6).
    ///
    /// Calls `S::pre_start` and wraps the shader in `ShaderState`. Sprint 2 will
    /// wire this as the `ractor::Actor::pre_start` body.
    pub fn stub_pre_start(
        &self,
        shader: Arc<S>,
    ) -> Result<ShaderState<S>, anyhow::Error> {
        unimplemented!("LG-2 stub — Sprint 2")
    }

    /// Dispatch a single [`ShaderMessage`] onto the actor state (plan §6).
    ///
    /// Sprint 2 wires this as the `ractor::Actor::handle` body.
    pub fn stub_handle(
        &self,
        msg: ShaderMessage<S::Payload>,
        state: &mut ShaderState<S>,
    ) -> Result<(), anyhow::Error> {
        unimplemented!("LG-2 stub — Sprint 2")
    }
}

// Placeholder `ractor::Actor` impl block — commented out until Sprint 2 confirms
// the exact ractor version and trait API. Kept here so Sprint 2 has a scaffold to
// fill in without needing to rediscover the type layout.
//
// ```rust
// #[ractor::async_trait]
// impl<S> ractor::Actor for CognitiveShaderActor<S>
// where
//     S: SupervisableShader,
//     S::Error: Into<anyhow::Error>,
// {
//     type Msg = ShaderMessage<S::Payload>;
//     type State = ShaderState<S>;
//     type Arguments = Arc<S>;
//
//     async fn pre_start(
//         &self,
//         _myself: ractor::ActorRef<Self::Msg>,
//         shader: Self::Arguments,
//     ) -> Result<Self::State, ractor::ActorProcessingErr> {
//         shader.pre_start().map_err(|e| Into::<anyhow::Error>::into(e))?;
//         Ok(ShaderState { shader, inflight: 0 })
//     }
//
//     async fn handle(
//         &self,
//         myself: ractor::ActorRef<Self::Msg>,
//         msg: Self::Msg,
//         state: &mut Self::State,
//     ) -> Result<(), ractor::ActorProcessingErr> {
//         match msg {
//             ShaderMessage::Apply { input, reply } => {
//                 state.inflight += 1;
//                 let result = state.shader.apply(input).map_err(Into::into);
//                 state.inflight -= 1;
//                 let _ = reply.send(result);
//             }
//             ShaderMessage::ApplyDelta { delta } => {
//                 state.shader.apply_delta(delta).map_err(Into::<anyhow::Error>::into)?;
//             }
//             ShaderMessage::Drain { reply } => {
//                 while state.inflight > 0 {
//                     tokio::task::yield_now().await;
//                 }
//                 state.shader.drain().map_err(Into::<anyhow::Error>::into)?;
//                 let _ = reply.send(());
//                 myself.stop(Some("drained".into()));
//             }
//         }
//         Ok(())
//     }
// }
// ```
