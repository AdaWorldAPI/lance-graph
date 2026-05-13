W3 starting spec td-api-drift-deprecation.md

## Completion — 2026-05-13

Deliverable written: `.claude/specs/td-api-drift-deprecation.md` (21 KB, 455 lines)

Sources read:
- SPRINT_LOG.md (sprint-log-4) — task context
- EPIPHANIES.md (head 120) — D-SDR FMA demo anchor
- `crates/lance-graph-callcenter/src/unified_bridge.rs` — OwlIdentity, TenantId, AuthError, UnifiedBridge, authorize_* signatures
- `crates/lance-graph-callcenter/src/unified_audit.rs` — AuditChain, AuditMerkleRoot, UnifiedAuditEvent, UnifiedAuditSink
- `crates/lance-graph-callcenter/src/family_table.rs` — FamilyEntry, OgitFamilyTable, OwlCharacteristics
- `crates/lance-graph-callcenter/src/policy.rs` — PolicyRewriter, ColumnMaskRewriter
- `crates/lance-graph-ontology/src/bridge.rs` — BridgeError variants (NamespaceMissing, NotInScope, CrossNamespaceLeak)

Key findings:
- Policy::evaluate drift is SILENT (not a compile error) — most dangerous
- BridgeError rename is a HARD compile error with 3-variant rename
- AuditChainBuilder doesn't exist at all — moved into UnifiedBridge::with_audit_chain()
- OwlIdentity widened u8→u16; owl_from_schema_ptr truncates 16-bit entity_type_id to 8 bits (cross-ref TD-SDR-SLOT-TRUNC-1 / W10)
- FamilyEntry shape changed: 2 fields → 7 fields, owned String → &'static str

Cross-flags: W7 (release window), W10 (slot truncation + BridgeError audit emission)
