## 2026-05-13 — W11 start

**Worker:** W11 — sprint-6 PR-G2 CallcenterSupervisor ractor port  
**Deliverable:** `.claude/specs/pr-g2-ractor-supervisor.md`

**Mandatory reads completed (in order):**
1. `ls .claude/plans/ | head -40` — confirmed 32 plan files present
2. `.claude/plans/compile-time-consumer-binding-v1.md` — read fully; Pattern F is D-RACTOR-SUPERVISOR; key sketch in §2.2; Open Q 6 (one-for-one vs all-for-one) deferred to this spec
3. `.claude/plans/callcenter-membrane-v1.md` — read fully; 4-layer architecture; DrainTask/LanceMembrane stay unchanged; UnifiedBridge is per-bridge monomorph
4. `.claude/board/LATEST_STATE.md` — PR #364 shipped D-SDR-3/4/5 + SuperDomain + UnifiedAuditEvent 26-byte; sprint-5-6 cross-repo landing confirmed
5. `.claude/board/PR_ARC_INVENTORY.md` — #364 locked: `SuperDomain` type exists, `AuthOp` enum exists, `AuditChain` shipped
6. `crates/lance-graph-callcenter/src/` — examined `unified_bridge.rs` (341 LOC, `UnifiedBridge<B>` with Mutex-guarded `AuditChain`, no supervisor), `lib.rs` (feature-gated modules, no supervisor mod), `Cargo.toml` (no ractor dep today)

**Key findings:**
- No supervisor exists today — `UnifiedBridge<B>` is per-consumer ad-hoc, no crash isolation
- `ractor` not in any Cargo.toml — fresh dep addition required
- `AuthOp` enum already shipped (Read/Write/Act); PR-G2 extends with ActorStart/ActorStop/ActorRestart
- `SuperDomain::System` is a new variant (not in current `super_domain.rs`)
- PR-F-1 (sprint-3 spec) used `Box<dyn ConsumerActorMsg>`; this spec upgrades to typed `ConsumerEnvelope` enum (eliminates ~40 ns box overhead, no expressivity loss since envelope arms are fixed)
- One-for-one restart chosen over "restart-all" (plan's simplification) due to §73 SGB V MedCare audit chain isolation requirement

**Spec produced:** `.claude/specs/pr-g2-ractor-supervisor.md` (555 lines, ~25 KB)
- §2: Actor topology (per-G, one-for-one, inert-skip Option A from PR-F-1 CORRECTION)
- §3: Message types (ConsumerEnvelope / ConsumerReply typed enums, not Box<dyn>)
- §4: Backpressure (bounded 1024-msg default, per-manifest override, unbounded supervisor)
- §5: Failure handling (one-for-one, exponential backoff 100ms→30s, escalation at crash_count>10)
- §6: Lifecycle audit (AuthOp::ActorStart/Stop/Restart, SuperDomain::System, feature-gated)
- §7: ractor 0.14 + `features = ["tokio-runtime"]`, I-2 via clippy disallowed-types
- §8: File layout (14 files, ~820 LOC total)
- §11: DELTA from both plan docs and pr-f-1 sprint-3 spec — all 5 differences cited
