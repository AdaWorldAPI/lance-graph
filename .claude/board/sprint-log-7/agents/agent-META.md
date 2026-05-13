## 2026-05-13 — Opus META cross-implementation review

Wrote `.claude/board/sprint-log-7/meta-review.md` (32 KB, 8 sections).

Per-impl grades: W1 A, W2 A, W3 B, W4 A, W5 A, W6 B-minus, W7 B.
Single must-fix CC-7-1: `UnifiedAuditSink` (bridge-owned) vs `AuditSink`
(W6-built) trait families do not match — W6 sinks ship orphaned from
the bridge. Recommend deprecating UnifiedAuditSink in PR-C scope.

prev_merkle field addition verified safe: excluded from canonical_bytes;
`UnifiedAuditEvent` stays 26 bytes; W4 A1 assertion still passes.

SuperDomain::System landed at super_domain.rs:77; W4 A6 fixture
explicitly exempts it at harness.rs:214.

SMB/SMB.bson both → WorkOrderBilling confirmed correct (OQ-4 locked
2026-05-13); not a defect; doc-comment at hydration.rs:287-306.

MedCare#116 entity-name realignment does NOT collide with W1 TTL
(family/basin names vs entity names — different layers). Followup
cross-check needed on `medcare_ontology()` registry seed.

Sequencing: 3 thematic PRs (A=scaffold, B=hydration, C=gate+audit-sink).
DO NOT ship as 1 mega-PR — buries the trait-family fix.

5 OQs raised; OQ-7-2 (trait-family resolution) blocks PR-C; OQ-7-3
(default sink) blocks MedCare sprint-2 item 5 wiring.
