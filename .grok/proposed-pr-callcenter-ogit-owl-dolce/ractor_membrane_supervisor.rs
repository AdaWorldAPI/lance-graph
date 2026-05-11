//! Proposed Ractor-based membrane supervisor skeleton
//!
//! This can live inside lance-graph-callcenter as the orchestration layer
//! on top of ExternalMembrane.

use ractor::{Actor, ActorRef, SupervisionEvent};

/// Top-level supervisor for the lance-graph-callcenter membrane.
pub struct MembraneSupervisor;

#[derive(Debug)]
pub enum MembraneMessage {
    Ingest,
    Project,
    Subscribe,
    // ... more intents
}

impl Actor for MembraneSupervisor {
    type Msg = MembraneMessage;
    type State = ();
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ractor::ActorProcessingErr> {
        // TODO: spawn child actors for different membrane responsibilities
        // e.g. ingestion actor, projection actor, subscription actor, etc.
        Ok(())
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ractor::ActorProcessingErr> {
        match message {
            MembraneMessage::Ingest => {
                // TODO: dispatch to ingestion actor
            }
            MembraneMessage::Project => {
                // TODO: dispatch to projection actor
            }
            MembraneMessage::Subscribe => {
                // TODO: dispatch to subscription / realtime actor
            }
        }
        Ok(())
    }

    async fn handle_supervisor_evt(
        &self,
        _myself: ActorRef<Self::Msg>,
        message: SupervisionEvent,
        _state: &mut Self::State,
    ) -> Result<(), ractor::ActorProcessingErr> {
        match message {
            SupervisionEvent::ActorStarted(_) => {}
            SupervisionEvent::ActorTerminated(actor, _, reason) => {
                tracing::warn!(actor = ?actor, reason = ?reason, "Membrane child actor terminated");
            }
            _ => {}
        }
        Ok(())
    }
}
