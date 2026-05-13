# agent-W12 — sprint-log-5-6 / S6-W10 conformance test spec

> Worker: W12 | Sprint: Sprint-6 | Deliverable: .claude/specs/sprint-6-conformance-test.md
> Started: 2026-05-13 | Model: claude-sonnet-4-6

## Read order completed

1. ls .claude/plans/ — 32 plan files indexed
2. super-domain-rbac-tenancy-v1.md §14 — harvested §14.1-§14.5 (D-SDR-18..23), §13.3 AuditChain merkle, §3.4-§3.8 SuperDomain/OgitFamily/OwlIdentity
3. foundry-consumer-parity-v1.md — §2 shared Foundry surface (DM-8/LF-12/LF-31/LF-90/LF-92)
4. LATEST_STATE.md — D-SDR-5 shipped PR #364: UnifiedBridge<B>, AuditChain, 26-byte UnifiedAuditEvent, 3-byte OwlIdentity
5. unified_bridge.rs — full trait surface + RecordingSink pattern
6. orchestration.rs — OrchestrationBridge, StepDomain, DomainProfile
7. td-super-domain-subcrates.md — CI gate shape from sprint-5 W4

## Key decisions

- Separate crate: crates/lance-graph-consumer-conformance/
- Generic harness: assert_consumer_conformance<B: NamespaceBridge>()
- 10 assertions A1-A10: audit shape, super_domain stamp, merkle chain, bridge-error short-circuit, canonical-vs-alias policy, SuperDomain != Unknown, family table coverage, TenantId isolation, actor_role_hash, g_lock non-zero
- E1/E2/E3 active (blocking CI), E4/E5 scaffold #[ignore]
- DELTA vs foundry-consumer-parity-v1.md documented in Section 8

## Status: COMPLETE — .claude/specs/sprint-6-conformance-test.md written (26,467 bytes)
