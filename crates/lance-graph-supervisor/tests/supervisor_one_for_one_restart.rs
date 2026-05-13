//! Test: one-for-one restart — crash one child, sibling is unaffected.
//!
//! Spec: pr-g2-ractor-supervisor.md §5.1-§5.3

#[cfg(feature = "supervisor")]
mod tests
{
    use lance_graph_supervisor::{CallcenterSupervisor, ModuleEntry, SupervisorMsg};
    use ractor::Actor;
    use std::time::Duration;

    fn two_active_modules() -> Vec<ModuleEntry>
    {
        vec![
            ModuleEntry { g: 2, version: 1, mailbox_capacity: None, is_active: true }, // Healthcare
            ModuleEntry { g: 4, version: 1, mailbox_capacity: None, is_active: true }, // SMB
        ]
    }

    #[tokio::test]
    async fn one_for_one_sibling_alive_after_peer_death()
    {
        let (actor_ref, handle) = Actor::spawn(
            None,
            CallcenterSupervisor::new(two_active_modules()),
            (),
        )
        .await
        .expect("supervisor spawn");

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Verify both children live before the crash.
        let before = actor_ref
            .call(|tx| SupervisorMsg::Health { reply: tx }, None)
            .await
            .expect("health call")
            .unwrap();

        let live_before: Vec<u32> = before.children.iter().filter(|c| c.is_live).map(|c| c.g).collect();
        assert!(live_before.contains(&2), "G=2 should be live before crash");
        assert!(live_before.contains(&4), "G=4 should be live before crash");

        // Kill the G=2 actor by registry name.
        if let Some(g2_cell) = ractor::registry::where_is("consumer_g_2".to_string()) {
            g2_cell.stop(None);
        }

        // Wait for supervisor to detect termination and schedule respawn.
        // Backoff: 100ms * 2^1 = 200ms; give 400ms total.
        tokio::time::sleep(Duration::from_millis(400)).await;

        // G=4 must still be alive (one-for-one: sibling unaffected).
        let after = actor_ref
            .call(|tx| SupervisorMsg::Health { reply: tx }, None)
            .await
            .expect("health after crash")
            .unwrap();

        let g4_alive = after.children.iter().any(|c| c.g == 4 && c.is_live);
        assert!(g4_alive, "G=4 (SMB) must remain alive after G=2 (Healthcare) crash");

        // G=2 should have been respawned by now.
        let g2_respawned = after.children.iter().any(|c| c.g == 2 && c.is_live);
        assert!(g2_respawned, "G=2 (Healthcare) should have been respawned after one-for-one restart");

        actor_ref.stop(None);
        let _ = handle.await;
    }
}
