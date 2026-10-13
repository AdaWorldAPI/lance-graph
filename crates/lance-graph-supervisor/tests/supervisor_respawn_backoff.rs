//! Test: a crashed consumer actor is respawned once per crash, after the
//! exponential backoff, and not at all once the escalation threshold is passed.
//!
//! Issue: ISS-SUPERVISOR-RESPAWN-BACKOFF-NOT-APPLIED. Before the fix, the
//! `RespawnG` arm respawned at once (no delay) and also scheduled a delayed
//! `RespawnG { crash_count: 0 }` that re-ran the same arm every 100 ms forever,
//! so a child escalated past 10 crashes was brought back by that chain.
//!
//! Time is paused (`start_paused`), so the 30 s backoff cap costs nothing:
//! tokio advances the clock whenever every task is idle.

#[cfg(feature = "supervisor")]
mod tests {
    use lance_graph_supervisor::{CallcenterSupervisor, ModuleEntry, SupervisorMsg};
    use ractor::{Actor, ActorRef};
    use std::time::Duration;

    fn one_active_module() -> Vec<ModuleEntry> {
        vec![ModuleEntry {
            g: 2,
            version: 1,
            mailbox_capacity: None,
            is_active: true,
        }]
    }

    /// `(is_live, crash_count)` of G=2 as the supervisor reports it.
    async fn g2(sup: &ActorRef<SupervisorMsg>) -> (bool, u32) {
        let health = sup
            .call(|tx| SupervisorMsg::Health { reply: tx }, None)
            .await
            .expect("health call")
            .expect("health reply");
        let child = health
            .children
            .iter()
            .find(|c| c.g == 2)
            .expect("G=2 is in the module table");
        (child.is_live, child.crash_count)
    }

    fn crash_g2() {
        ractor::registry::where_is("consumer_g_2".to_string())
            .expect("G=2 is registered while live")
            .stop(None);
    }

    /// Backoff after crash `n`: `min(100 ms * 2^n, 30 s)`.
    fn backoff(n: u32) -> Duration {
        Duration::from_millis((100u64 << n.min(30)).min(30_000))
    }

    #[tokio::test(start_paused = true)]
    async fn each_crash_respawns_once_after_its_backoff() {
        let (sup, handle) = Actor::spawn(None, CallcenterSupervisor::new(one_active_module()), ())
            .await
            .expect("supervisor spawn");
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(g2(&sup).await, (true, 0), "G=2 live before any crash");

        for n in 1..=2u32 {
            crash_g2();
            // Half the backoff: the child must still be down.
            tokio::time::sleep(backoff(n) / 2).await;
            assert_eq!(
                g2(&sup).await.0,
                false,
                "crash {n}: respawned before its {:?} backoff",
                backoff(n)
            );
            // Past the backoff: up again, and the count is carried, not reset.
            tokio::time::sleep(backoff(n)).await;
            assert_eq!(g2(&sup).await, (true, n), "crash {n}: one respawn after the backoff");
        }

        sup.stop(None);
        let _ = handle.await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_child_past_the_escalation_threshold_stays_down() {
        let (sup, handle) = Actor::spawn(None, CallcenterSupervisor::new(one_active_module()), ())
            .await
            .expect("supervisor spawn");
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Crashes 1..=10 are respawned (threshold is "more than 10").
        for n in 1..=10u32 {
            crash_g2();
            tokio::time::sleep(backoff(n) + Duration::from_millis(50)).await;
            assert_eq!(g2(&sup).await, (true, n), "crash {n} is respawned");
        }

        // Crash 11 escalates: no respawn, however long we wait.
        crash_g2();
        tokio::time::sleep(Duration::from_secs(120)).await;
        assert!(!g2(&sup).await.0, "crash 11 must not be respawned");

        sup.stop(None);
        let _ = handle.await;
    }
}
