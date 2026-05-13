//! Error types for `lance-graph-supervisor`.

use thiserror::Error;

/// Errors returned by the `CallcenterSupervisor`.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SupervisorErr {
    /// The requested G slot exists in the manifest but has no active consumer
    /// (`consumer_pointer = None`, `inert_when_consumer_absent = true`).
    /// SPARQL queries against inert triples route through `OntologyRegistry`
    /// directly, bypassing the actor mesh.
    #[error("G slot {0} is inert (no consumer registered)")]
    InertG(u32),

    /// Child actor's mailbox is full — the caller must retry, shed load, or
    /// escalate. Corresponds to spec §4.4 `SupervisorErr::MailboxFull`.
    #[error("G slot {0} mailbox is full — apply backpressure")]
    MailboxFull(u32),

    /// Consumer actor is unhealthy (crash_count > 10 within the window).
    /// Operator action required: send `ResetCrashCount { g }`.
    #[error("consumer actor for G={0} is unhealthy; operator action required")]
    ConsumerUnhealthy(u32),

    /// Dispatch path for this envelope variant is not yet implemented in the
    /// stub actor (skeleton); replace `StubConsumerActor` with a concrete impl.
    #[error("dispatch not implemented in stub consumer actor — wire a concrete ConsumerActor<B>")]
    DispatchNotImplemented,

    /// Supervisor itself failed to start (pre_start error).
    #[error("supervisor pre_start failed: {0}")]
    StartupFailed(String),
}
