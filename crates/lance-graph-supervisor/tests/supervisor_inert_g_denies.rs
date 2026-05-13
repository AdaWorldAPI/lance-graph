//! Test: DispatchToG to inert/unknown G → SupervisorErr::InertG (not a panic).
//!
//! Spec: pr-g2-ractor-supervisor.md §2.2

#[cfg(feature = "supervisor")]
mod tests
{
    use lance_graph_supervisor::{
        CallcenterSupervisor, ConsumerEnvelope, ModuleEntry, SupervisorErr, SupervisorMsg,
    };
    use ractor::Actor;

    fn single_active_module() -> Vec<ModuleEntry>
    {
        vec![
            ModuleEntry { g: 2, version: 1, mailbox_capacity: None, is_active: true },
            ModuleEntry { g: 5, version: 1, mailbox_capacity: None, is_active: false }, // inert FMA
        ]
    }

    #[tokio::test]
    async fn inert_g_returns_inert_error_not_panic()
    {
        let (actor_ref, handle) = Actor::spawn(
            None,
            CallcenterSupervisor::new(single_active_module()),
            (),
        )
        .await
        .expect("supervisor spawn");

        tokio::time::sleep(std::time::Duration::from_millis(30)).await;

        // Dispatch to inert G=5 (FMA) using actor.call() with struct variant closure.
        let result: Result<lance_graph_supervisor::ConsumerReply, SupervisorErr> = actor_ref
            .call(
                |tx| SupervisorMsg::DispatchToG {
                    g:        5,
                    version:  1,
                    envelope: ConsumerEnvelope::Health,
                    reply:    tx,
                },
                None,
            )
            .await
            .expect("ractor call must not error")
            .unwrap();

        assert!(result.is_err(), "inert G=5 must return an error, not a success reply");
        assert_eq!(result.unwrap_err(), SupervisorErr::InertG(5), "inert G=5 error must be InertG(5)");

        // Dispatch to completely unknown G=999.
        let result2: Result<lance_graph_supervisor::ConsumerReply, SupervisorErr> = actor_ref
            .call(
                |tx| SupervisorMsg::DispatchToG {
                    g:        999,
                    version:  1,
                    envelope: ConsumerEnvelope::Health,
                    reply:    tx,
                },
                None,
            )
            .await
            .expect("ractor call must not error")
            .unwrap();

        assert!(result2.is_err(), "unknown G=999 must return an error");
        assert_eq!(result2.unwrap_err(), SupervisorErr::InertG(999), "unknown G=999 error must be InertG(999)");

        actor_ref.stop(None);
        let _ = handle.await;
    }
}
