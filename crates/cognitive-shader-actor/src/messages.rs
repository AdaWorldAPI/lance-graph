//! # Shader actor message vocabulary (plan §6)
//!
//! Defines [`ShaderMessage<P>`], the single enum that flows through every
//! `CognitiveShaderActor`'s mailbox. The three variants match the plan §6
//! API sketch for `ShaderMessage`:
//!
//! - [`ShaderMessage::Apply`] — transform a payload, reply with the result.
//! - [`ShaderMessage::ApplyDelta`] — fire-and-forget incremental update.
//! - [`ShaderMessage::Drain`] — graceful shutdown; reply when in-flight work
//!   is complete.
//!
//! The payload type `P` is left generic so the contract crate (and this crate)
//! stay zero-dep on Arrow. The actor module fixes `P = arrow_array::RecordBatch`
//! for the shipped supervisor topology; other consumers may choose any `P: Send`.
//!
//! ## Sprint status
//!
//! Scaffold — Sprint 2 wires real ractor RPC ports.

/// The mailbox vocabulary for a supervised cognitive shader (plan §6).
///
/// `P` is the payload type — typically `arrow_array::RecordBatch` in the shipped
/// supervisor topology, but left generic so the message type is usable for any
/// payload that is `Send + 'static`.
///
/// Every public variant is doc-commented with its plan §6 reference.
#[derive(Debug)]
pub enum ShaderMessage<P>
where
    P: Send + 'static,
{
    /// Request the shader to transform `input` and return the result.
    ///
    /// The actor increments its `inflight` counter before calling
    /// `SupervisableShader::apply`, decrements it after, then sends the result
    /// through `reply`. Plan §6 `ShaderMessage::Apply`.
    Apply {
        /// The payload to transform.
        input: P,
        /// One-shot reply port; the actor sends `Ok(P)` on success, `Err` on failure.
        ///
        /// Plan §6: `reply: ractor::RpcReplyPort<anyhow::Result<P>>`.
        reply: ractor::RpcReplyPort<anyhow::Result<P>>,
    },

    /// Fire-and-forget incremental update; no reply expected.
    ///
    /// Routes to `SupervisableShader::apply_delta`. The default trait impl
    /// forwards to `apply` and discards the output, so shaders that do not
    /// override `apply_delta` get correct behaviour for free.
    ///
    /// Plan §6 `ShaderMessage::ApplyDelta`.
    ApplyDelta {
        /// The incremental payload to fold into the shader's internal state.
        delta: P,
    },

    /// Graceful drain: wait until all in-flight `Apply` messages complete, then
    /// call `SupervisableShader::drain` and reply to the caller.
    ///
    /// The actor stops itself after replying. Plan §6 `ShaderMessage::Drain`.
    Drain {
        /// One-shot reply port; the actor sends `()` once drained.
        reply: ractor::RpcReplyPort<()>,
    },
}
