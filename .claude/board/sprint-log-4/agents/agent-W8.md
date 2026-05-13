# agent-W8 — TD-SDR-AUDIT-PERSIST-1 spec run

**Date:** 2026-05-13
**Task:** Write `.claude/specs/td-sdr-audit-persist.md` — AuditSink trait + Lance/JSONL/Composite sinks + replay tool + perf budget

## Findings from code grep

- `UnifiedAuditEvent` lives at `crates/lance-graph-callcenter/src/unified_audit.rs:146`
- `AuditMerkleRoot` = u64 FNV-1a chain (not [u8;32] — see field: `pub struct AuditMerkleRoot(pub u64)`)
- Existing `UnifiedAuditSink` trait at line 258: `fn emit(&self, event: &UnifiedAuditEvent)` — takes `&UnifiedAuditEvent`, no return value, no flush/checkpoint
- `NoopUnifiedAuditSink` is current default
- `UnifiedBridge::emit_audit()` calls `sink.emit(&stamped)` at line 405
- `BridgeError` short-circuits BEFORE `emit_audit()` — W10 coordination needed
- Canonical bytes: 25 bytes — ts_unix_ms(8) + tenant(4) + super_domain(1) + owl(2) + op(1) + decision(1) + actor_role_hash(8)
- NOTE: merkle_root is u64 (8 bytes), not [u8;32] as originally specced — arrow schema will use u64

## Cross-flags
- W4: super-domain subcrates → each owns its `UnifiedBridge` instance → each has its own `AuditChain` + `AuditSink`
- W10: BridgeError path must emit before short-circuit

## Spec written to `.claude/specs/td-sdr-audit-persist.md`

## Run 2 — 2026-05-13 (writing spec)

Starting spec write. Key corrections vs original brief:
- merkle_root is u64 (FNV-1a 64-bit), not [u8;32]
- prev_merkle field = AuditMerkleRoot.0 stored as u64 in Arrow schema
- AuditMerkleRoot::GENESIS = 0xa5a5_a5a5_a5a5_a5a5


## Final status — 2026-05-13

Spec written to `.claude/specs/td-sdr-audit-persist.md` (23.8 KB / 647 lines).
All 8 required sections delivered. Key corrections applied vs. brief:
- merkle fields are u64 (not [u8;32]) — Arrow schema uses UInt64
- prev_merkle requires extending UnifiedAuditEvent (action item noted in §4)
- 4 open questions captured (OQ-1 through OQ-4)
