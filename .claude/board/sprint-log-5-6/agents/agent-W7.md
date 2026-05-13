## 2026-05-13 — W7 start + complete

**Worker:** W7 — S6-W3 PR-E2 smb-office UnifiedBridge retrofit spec
**Deliverable:** `.claude/specs/pr-e2-smb-retrofit.md` (~11 KB, 12 sections)

**Reads completed (mandatory order):**
1. `.claude/plans/` ls — confirmed plan inventory
2. `.claude/board/LATEST_STATE.md` — sprint-5 cross-repo landing; smb-office-rs#31 = +111 LOC minimal hook
3. `.claude/board/PR_ARC_INVENTORY.md` #364 — D-SDR-3/4/5 shipped; OwlIdentity 3-byte (u16 slot)
4. `.claude/plans/super-domain-rbac-tenancy-v1.md` §14 — D-SDR-22 = "smb-office-rs retrofit, zero behavior change"
5. `.claude/plans/callcenter-membrane-v1.md` — BBB semantics, AuditChain, ExternalMembrane
6. smb-office-rs git log — 342f601 = PR-C commit
7. smb-bridge/src tree: unified_bridge_wiring.rs + auth.rs + rls.rs + orchestration.rs + mongo.rs + lance.rs + lib.rs
8. `.claude/specs/td-sdr-family-hydration.md` — W9 cross-flag; new_hydrated(); TOML seed
9. `.claude/specs/td-super-domain-subcrates.md` — Phase 2 = smb-bridge retrofit; dependency chain

**Key findings:**
- PR-C constructor smb_unified_bridge() is never called in production — 1 error-path test only
- 5 bypass sites identified: MongoConnector, LanceConnector, SmbOrchestrator::route, main.rs, login_flow.rs
- No UnifiedAuditEvent emitted anywhere in smb-office-rs post-PR-C
- Super-domain: WorkOrderBilling (Steuerberater) + Networking (WoA); Networking discriminant NOT yet in super_domain.rs (§8.1 blocker)
- OwlIdentity slot is u16 per P1 fix (3-byte canonical)
- Batches A+B have no blockers; Batch C blocked on §8.1 + pr-d4-family-hydration

**Status:** COMPLETE — spec written at .claude/specs/pr-e2-smb-retrofit.md
