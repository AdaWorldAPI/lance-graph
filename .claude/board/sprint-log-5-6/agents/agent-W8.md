# agent-W8 — sprint-log-5-6 S6-W4 PR-E3 woa-rs-extract spec

> Worker: W8 (claude-sonnet-4-6) | Date: 2026-05-13 | Deliverable: .claude/specs/pr-e3-woa-rs-extract.md

## Context read
- super-domain-rbac-tenancy-v1.md §14 (harvest + templates)
- q2-foundry-integration-v1.md (woa lives inside WorkOrderBilling super-domain, q2/geo testbed)
- LATEST_STATE.md + PR_ARC_INVENTORY.md: #364 shipped D-SDR-1..5; woa-rs bridge already exists in lance-graph-ontology
- Searched crates: woa_bridge.rs (lance-graph-ontology), super_domain.rs (WorkOrderBilling=6), unified_bridge.rs (authorize tests with WorkOrder namespace)
- pr-e1-medcare-super-domain.md (W6): gap analysis shape, 6 items, ~900 LOC, Healthcare-specific
- pr-e2-smb-retrofit.md (W7): 5-bypass-site pattern, 3-batch plan, ~480 LOC net

## Status: WRITING SPEC
