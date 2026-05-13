# agent-W3.md — sprint-log-5-6 W3 scratchpad

**Sprint:** sprint-log-5-6 | **Worker:** W3 | **Date:** 2026-05-13
**Deliverable:** .claude/specs/pr-d4-family-hydration.md

## Read-order completed

1. ls .claude/plans/ | head -40 — confirmed super-domain-rbac-tenancy-v1.md present
2. .claude/board/LATEST_STATE.md — confirmed PR #364 merged; FAMILY_TO_SUPER_DOMAIN deferred
3. .claude/board/PR_ARC_INVENTORY.md #364 entry — locked: table is vestigial fallback; hydration deferred
4. super-domain-rbac-tenancy-v1.md §3.4 + §3.3 + §6 + §8 D-SDR-4 + §9.1 read
5. crates/lance-graph-callcenter/src/super_domain.rs — confirmed static [Unknown; 256], never written
6. crates/lance-graph-callcenter/src/family_table.rs — OgitFamilyTable: sparse HashMap<u16, FamilyEntry> (post P1)
7. crates/lance-graph-ontology/src/registry.rs — hydrate_once_sync, OnceLock<LazyLock<NamespaceRegistry>>, RwLock pattern

## Key findings

- FAMILY_TO_SUPER_DOMAIN is a compile-time static [Unknown; 256], never mutated. super_domain_for_family() returns Unknown always.
- AuditChain.super_domain() (the P2 fix) reads a struct field, not the table. Hot path is safe; fallback is broken.
- OntologyRegistry uses RwLock<RegistryState> + OnceLock<LazyLock<NamespaceRegistry>> precedent.
- parse_ttl_directory_with_provenance() exists and is battle-tested. Avoids TOML dependency.
- td-sdr-family-hydration.md (sprint-4 stub) proposed TOML; this spec overrides with TTL reuse.
- lance-graph-callcenter already depends on lance-graph-ontology (family_table.rs:51 re-export of SchemaKind).

## Decisions made in spec

- TTL not TOML for format (reuses existing ingest machinery, zero new deps)
- OnceLock<Arc<RwLock<FamilyTableInner>>> storage (consistent snapshots, no unsafe)
- Hot reload opt-in (None = restart-only default)
- BestEffort vs RequireMinDomains policy distinction (backward compat for tests)
- try_resolve() API added; old super_domain_for_family() becomes shim
- OQ-1: parser extension boundary — recommend option (c) new thin entry point

## Output

.claude/specs/pr-d4-family-hydration.md: 16145 bytes

