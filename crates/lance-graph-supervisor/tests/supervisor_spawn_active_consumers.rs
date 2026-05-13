//! Test: per-G actor spawn — 3 active G slots → 3 children spawned.
//! Inert slots (G=0 DOLCE, G=5 FMA) are skipped.
//!
//! Spec: pr-g2-ractor-supervisor.md §2.1-§2.2 (Option A inert skipping).

#[cfg(feature = "supervisor")]
mod tests
{
    use lance_graph_supervisor::{CallcenterSupervisor, ModuleEntry, SupervisorMsg};
    use ractor::Actor;

    fn module_table() -> Vec<ModuleEntry>
    {
        vec![
            ModuleEntry { g: 0, version: 1, mailbox_capacity: None, is_active: false }, // DOLCE — inert
            ModuleEntry { g: 2, version: 1, mailbox_capacity: None, is_active: true  }, // Healthcare
            ModuleEntry { g: 3, version: 1, mailbox_capacity: None, is_active: true  }, // GOTHAM
            ModuleEntry { g: 4, version: 1, mailbox_capacity: None, is_active: true  }, // SMB
            ModuleEntry { g: 5, version: 1, mailbox_capacity: None, is_active: false }, // FMA — inert
        ]
    }

    #[tokio::test]
    async fn supervisor_spawns_exactly_3_active_children()
    {
        let supervisor = CallcenterSupervisor::new(module_table());
        let (actor_ref, handle) = Actor::spawn(None, supervisor, ())
            .await
            .expect("supervisor spawn must succeed");

        // Give actors a tick to start up.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Query health — use actor.call() with closure since Health is a struct variant.
        let summary = actor_ref
            .call(|tx| SupervisorMsg::Health { reply: tx }, None)
            .await
            .expect("call must succeed")
            .unwrap();

        let live_count = summary.children.iter().filter(|c| c.is_live).count();
        assert_eq!(live_count, 3, "expected 3 live children (G=2,3,4); got {live_count}");

        let inert_count = summary.children.iter().filter(|c| !c.is_live).count();
        assert_eq!(inert_count, 2, "G=0 and G=5 should be inert (not live)");

        actor_ref.stop(None);
        let _ = handle.await;
    }
}
