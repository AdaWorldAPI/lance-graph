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
//! [`SupervisableShader`]: lance_graph_contract::actor::SupervisableShader

use std::sync::Arc;

use lance_graph_contract::actor::SupervisableShader;
use ractor::{Actor, ActorProcessingErr, ActorRef};

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
// ractor::Actor impl
// ---------------------------------------------------------------------------
//
// ractor 0.15 ships WITHOUT the `async-trait` feature in its defaults; it uses
// native `async fn` in traits (Rust >= 1.75).  The impl below uses bare
// `async fn` accordingly — no `#[async_trait]` macro needed.
//
// Associated types:
//   Msg       = ShaderMessage<S::Payload>
//   State     = ShaderState<S>
//   Arguments = Arc<S>   (the concrete shader instance is passed on spawn)

impl<S> Actor for CognitiveShaderActor<S>
where
    S: SupervisableShader,
    S::Error: Into<anyhow::Error>,
    S::Payload: std::fmt::Debug,
{
    type Msg = ShaderMessage<S::Payload>;
    type State = ShaderState<S>;
    type Arguments = Arc<S>;

    /// Initialise actor state from the supplied shader instance.
    ///
    /// Calls `S::pre_start` so the shader can do one-time setup; wraps it in
    /// `ShaderState` with `inflight = 0`.
    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        shader: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        shader
            .pre_start()
            .map_err(|e| Into::<anyhow::Error>::into(e))?;
        Ok(ShaderState {
            shader,
            inflight: 0,
        })
    }

    /// Dispatch a single [`ShaderMessage`] onto the actor state (plan §6).
    ///
    /// | Variant | Action |
    /// |---|---|
    /// | `Apply` | increment `inflight`, call `shader.apply`, decrement, reply |
    /// | `ApplyDelta` | call `shader.apply_delta` |
    /// | `Drain` | yield until `inflight == 0`, call `shader.drain`, reply, stop |
    async fn handle(
        &self,
        myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            ShaderMessage::Apply { input, reply } => {
                state.inflight += 1;
                let result = state
                    .shader
                    .apply(input)
                    .map(Ok)
                    .unwrap_or_else(|e| Err(Into::<anyhow::Error>::into(e)));
                state.inflight -= 1;
                // Ignore send errors — the caller may have dropped its receiver.
                let _ = reply.send(result);
            }
            ShaderMessage::ApplyDelta { delta } => {
                state
                    .shader
                    .apply_delta(delta)
                    .map_err(|e| Into::<anyhow::Error>::into(e))?;
            }
            ShaderMessage::Drain { reply } => {
                // Spin-yield until all in-flight Apply messages finish.
                while state.inflight > 0 {
                    tokio::task::yield_now().await;
                }
                state
                    .shader
                    .drain()
                    .map_err(|e| Into::<anyhow::Error>::into(e))?;
                let _ = reply.send(());
                myself.stop(Some("drained".into()));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ractor::call;

    /// A minimal shader for testing: `apply` doubles the input.
    struct Counter;

    impl SupervisableShader for Counter {
        type Payload = u32;
        type Error = std::convert::Infallible;

        fn apply(&self, payload: u32) -> Result<u32, Self::Error> {
            Ok(payload * 2)
        }
    }

    /// End-to-end test: spawn a `CognitiveShaderActor::<Counter>`, send an
    /// `Apply { input: 5 }` via `ractor::call!`, assert result is `Ok(10)`,
    /// then drain the actor cleanly.
    #[tokio::test]
    async fn e2e_apply_doubles_input() {
        // Spawn the actor with a Counter instance as Arguments.
        let (actor_ref, handle) = Actor::spawn(
            Some("counter-test".into()),
            CognitiveShaderActor::<Counter>::new(),
            Arc::new(Counter),
        )
        .await
        .expect("failed to spawn CognitiveShaderActor");

        // RPC: Apply { input: 5 }.  The call! macro builds the reply port and
        // passes it as the last argument to the message constructor closure.
        let result = call!(actor_ref, |reply| ShaderMessage::Apply {
            input: 5u32,
            reply,
        })
        .expect("call! failed");

        // The shader doubles the input, so 5 -> 10.
        assert!(result.is_ok(), "apply returned Err: {:?}", result);
        assert_eq!(result.unwrap(), 10u32);

        // Drain the actor cleanly: wait for inflight to settle, then stop.
        let drain_result = call!(actor_ref, |reply| ShaderMessage::Drain { reply })
            .expect("drain call! failed");
        assert_eq!(drain_result, ());

        // Wait for the actor to finish.
        handle.await.expect("actor join failed");
    }
}
