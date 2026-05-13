# S7-W3 Scratchpad — CallcenterSupervisor (PR-G2)

## Status: COMPLETE

## What was implemented

1. **New crate** `crates/lance-graph-supervisor/` added to workspace.
   - `Cargo.toml`: `ractor = "0.14"` (feature `tokio_runtime`), `supervisor` feature gate.
   - `src/lib.rs`, `src/error.rs`, `src/consumer_msg.rs`, `src/lifecycle_audit.rs`
   - `src/supervisor.rs`: `CallcenterSupervisor` + `StubConsumerActor` (one-for-one, per-G slots, backoff 100ms→30s, escalation at crash_count > 10)
   - `src/actors/mod.rs` + `src/actors/medcare_actor.rs`: G=2 proof-of-concept skeleton

2. **CC-2 compliance**: `LifecycleAuditEvent` (18-byte canonical_bytes) is SEPARATE from `UnifiedAuditEvent` (26-byte). `AuthOp` NOT modified. `UnifiedAuditEvent::canonical_bytes` regression test passes.

3. **CC-3 compliance**: `SuperDomain::System = 8` added to `lance-graph-callcenter/src/super_domain.rs` with doc comment explicitly stating it is exempt from the §13.4 hard-lock matrix.

4. **Tests** (all pass with `--features supervisor`):
   - `supervisor_spawn_active_consumers.rs` — 3 of 5 G slots spawn (2 inert skipped)
   - `supervisor_inert_g_denies.rs` — InertG(5), InertG(999) typed errors
   - `supervisor_one_for_one_restart.rs` — peer isolation + respawn
   - `supervisor_send_sync_compile.rs` — static Send+Sync assertions
   - `supervisor_lifecycle_audit.rs` — 18-byte layout, discriminants, noop sink

## Build verification
- `cargo check -p lance-graph-supervisor` — clean
- `cargo test -p lance-graph-supervisor --features supervisor` — all pass
- `cargo test -p lance-graph-callcenter canonical_bytes` — 1 passed (26-byte layout preserved)
