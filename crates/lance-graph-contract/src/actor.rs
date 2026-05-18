//! Actor-wrapper marker traits for supervisable cognitive shaders.
//!
//! # Scope
//!
//! This module defines a **payload-generic supervisable shader trait**
//! that the `cognitive-shader-actor` crate wraps as a `ractor::Actor`.
//! It is intentionally distinct from
//! [`crate::cognitive_shader::CognitiveShaderDriver`], which is the
//! in-process driver trait with `ShaderDispatch` semantics. The
//! supervisable variant is for the lifecycle a supervisor cares about:
//! apply / drain / restart.
//!
//! # Additive contract
//!
//! Added in 0.2.0. Pure addition — no existing surface in this crate is
//! touched. Existing `CognitiveShaderDriver` implementors continue to
//! work; they may *additionally* implement [`SupervisableShader`] to
//! opt into supervised execution.
//!
//! # Zero-dep payload
//!
//! The payload type is a generic associated type (`Payload`) so this
//! crate stays zero-dep — the `cognitive-shader-actor` crate picks
//! `arrow_array::RecordBatch`; other consumers can pick whatever fits.

use core::time::Duration;

/// Restart strategy for a shader supervisor.
///
/// Mirrors Erlang/OTP's classic three supervisor strategies. The
/// `cognitive-shader-actor` crate maps these onto ractor's supervisor
/// hierarchy; this enum is the contract vocabulary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SupervisionPolicy {
    /// Restart only the failed child.
    #[default]
    OneForOne,
    /// Restart all children on any failure (use when children share state).
    OneForAll,
    /// Restart the failed child and every child started after it.
    RestForOne,
}

impl SupervisionPolicy {
    /// Convenience constructor for the default policy. Equivalent to
    /// `SupervisionPolicy::default()`, kept for call-site readability.
    pub const fn one_for_one() -> Self {
        Self::OneForOne
    }
}

/// Restart back-off configuration.
///
/// Mirrors OTP's `max_restarts` / `max_seconds` window plus an exponential
/// inter-restart back-off. Defaults: 5 restarts in 60s, base 100ms,
/// max 30s back-off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestartBackoff {
    /// Max restarts in the sliding window before the supervisor escalates.
    pub max_restarts: u32,
    /// Sliding window length.
    pub window: Duration,
    /// Initial inter-restart delay; doubles on each restart up to `max_delay`.
    pub base_delay: Duration,
    /// Cap on the inter-restart delay.
    pub max_delay: Duration,
}

impl Default for RestartBackoff {
    fn default() -> Self {
        Self {
            max_restarts: 5,
            window: Duration::from_secs(60),
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
        }
    }
}

impl RestartBackoff {
    /// Compute the delay before restart attempt `n` (1-indexed).
    /// Exponential: `min(base * 2^(n-1), max_delay)`.
    pub fn delay_for_attempt(&self, n: u32) -> Duration {
        if n == 0 {
            return Duration::ZERO;
        }
        let shift = (n - 1).min(31); // avoid overflow for huge n
        let scaled = self.base_delay.saturating_mul(1u32 << shift);
        if scaled > self.max_delay {
            self.max_delay
        } else {
            scaled
        }
    }
}

/// A shader that can be wrapped as a supervisable actor.
///
/// Implementors are typically wrappers around concrete cognitive crates
/// (`thinking-engine`, `causal-edge`, `deepnsm`, `holograph`). The
/// `cognitive-shader-actor` crate consumes this trait to build a
/// `ractor::Actor` around any implementor.
///
/// # Lifecycle (called by the supervisor in this order)
///
/// 1. `pre_start` — runs once when the actor spawns.
/// 2. `apply` / `apply_delta` — called per message during normal operation.
/// 3. `drain` — called once during graceful shutdown; should flush
///    pending state.
/// 4. (on panic / Err return) → supervisor restarts: `pre_start` runs
///    again on the new instance.
///
/// All methods take `&self` so implementors must use interior mutability
/// (e.g. `Mutex`, `RwLock`, `AtomicU*`) if they hold state. This keeps
/// ractor's `Actor` impl ergonomic (no `&mut` ping-pong through the
/// message handler).
pub trait SupervisableShader: Send + Sync + 'static {
    /// The opaque payload this shader processes. The actor wrapper picks
    /// a concrete type (typically `arrow_array::RecordBatch`).
    type Payload: Send + Sync + 'static;

    /// Error type returned by shader operations.
    type Error: core::fmt::Debug + Send + Sync + 'static;

    /// Stable name used as the supervision-tree key. Defaults to the
    /// implementor type name via `core::any::type_name`.
    fn shader_name(&self) -> &'static str {
        core::any::type_name::<Self>()
    }

    /// Hook called once when the supervisor spawns / respawns the actor.
    /// Default: no-op.
    fn pre_start(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Apply the shader to a payload, returning the transformed payload.
    /// Called per `Apply` message in the actor wrapper.
    fn apply(&self, payload: Self::Payload) -> Result<Self::Payload, Self::Error>;

    /// Apply incrementally to a delta (no return value). The shader is
    /// expected to fold the delta into its internal state. Default:
    /// route through `apply` and discard the result.
    fn apply_delta(&self, delta: Self::Payload) -> Result<(), Self::Error> {
        self.apply(delta).map(|_| ())
    }

    /// Flush pending state. Called once during graceful shutdown.
    /// Default: no-op.
    fn drain(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn supervision_policy_default_is_one_for_one() {
        assert_eq!(SupervisionPolicy::default(), SupervisionPolicy::OneForOne);
    }

    #[test]
    fn restart_backoff_exponential() {
        let b = RestartBackoff {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            ..Default::default()
        };
        assert_eq!(b.delay_for_attempt(0), Duration::ZERO);
        assert_eq!(b.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(b.delay_for_attempt(2), Duration::from_millis(200));
        assert_eq!(b.delay_for_attempt(3), Duration::from_millis(400));
        // Eventually clamped to max_delay.
        assert_eq!(b.delay_for_attempt(20), Duration::from_secs(10));
    }

    #[test]
    fn restart_backoff_huge_n_does_not_overflow() {
        let b = RestartBackoff::default();
        // u32::MAX attempts should still produce a sane value.
        let d = b.delay_for_attempt(u32::MAX);
        assert!(d <= b.max_delay);
    }

    /// A minimal shader for testing the trait shape.
    struct Counter {
        n: AtomicUsize,
    }

    impl SupervisableShader for Counter {
        type Payload = u32;
        type Error = core::convert::Infallible;

        fn apply(&self, payload: u32) -> Result<u32, Self::Error> {
            self.n.fetch_add(payload as usize, Ordering::Relaxed);
            Ok(payload * 2)
        }
    }

    #[test]
    fn supervisable_shader_defaults() {
        let c = Arc::new(Counter {
            n: AtomicUsize::new(0),
        });
        assert!(c.pre_start().is_ok());
        assert_eq!(c.apply(5).unwrap(), 10);
        assert_eq!(c.n.load(Ordering::Relaxed), 5);
        // Default apply_delta routes through apply and discards.
        c.apply_delta(3).unwrap();
        assert_eq!(c.n.load(Ordering::Relaxed), 8);
        assert!(c.drain().is_ok());
        // Default shader_name returns the type name.
        assert!(c.shader_name().contains("Counter"));
    }
}
