# sprint-log-7 — implementation wave for sprint-5/6 specs

> **Started:** 2026-05-13 | **Branch:** claude/lance-datafusion-integration-gv0BF
> **Predecessor:** PR #365 (13 specs + Opus meta review) + EPIPHANIES 2026-05-13 DECISION entry (4 OQs resolved)
> **Worker fleet:** 6 wave-1 Sonnet workers + 1 wave-2 Sonnet (sequenced post-W1) + 1 Opus meta.
> **Pattern:** CCA2A — each worker tee -a's to .claude/board/sprint-log-7/agents/agent-W{N}.md and prepends to AGENT_ORCHESTRATION_LOG.md.
> **Output:** Rust code in respective crates (not specs). Each worker runs `cargo check -p <crate>` before reporting.
> **Boundary:** lance-graph-side ONLY. medcare-rs / smb-office-rs / woa-rs work owned by other sessions.

## Wave 1 — 6 parallel workers (no file conflicts)

| W | Implements spec | Target crates | C-grade fix |
|---|---|---|---|
| S7-W1 | pr-d4-family-hydration.md | lance-graph-ontology + lance-graph-callcenter | — |
| S7-W2 | pr-g1-manifest-modules.md | lance-graph-contract (build.rs + types) | §4.3 sorted-slice + binary search per OQ-2 |
| S7-W3 | pr-g2-ractor-supervisor.md | new crate lance-graph-supervisor | LifecycleAuditEvent split per meta CC-2 |
| S7-W4 | sprint-6-conformance-test.md | new crate lance-graph-consumer-conformance | — |
| S7-W5 | pr-f1-thinking-engine-wire.md | thinking-engine + cognitive-shader-driver + callcenter | — |
| S7-W6 | pr-d3a-lance-audit-sink.md + pr-d3b-jsonl-and-verify.md (combined) | lance-graph-callcenter (sink trait + impls) + new verify CLI bin | — |

## Wave 2 — 1 sequenced worker (after S7-W1 lands)

| W | Implements spec | Target |
|---|---|---|
| S7-W7 | pr-ogit-ttl-smb-hydration.md (lance-graph-side) | extends parse_family_registry() for ogit.SMB.bson: namespace |

## Wave 3 — Opus meta

Cross-implementation review across all 7 worker outputs.
