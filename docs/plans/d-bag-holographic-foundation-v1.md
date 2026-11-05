# D-BAG-HOLOGRAPHIC-FOUNDATION-1: Exclusive DR64 holographic Bag

## Scope correction: SAP-shaped GUID tile supraconductor POC (2026-11-05)

**This section supersedes earlier language implying the POC must build GridCraft, an Excel-compatible runtime, Databricks/DataFusion integration, or full F### formula semantics.** The actual POC is an **SAP-shaped, PSCustomObject-style projection of typed tiles whose relationships are established by tenant-qualified 128-bit source GUID correspondence and executed as witnessed numeric/mask folds in the existing lance-graph/ndarray substrate.**

### Source-backed precedents, not new frameworks

- [lance-graph-sap::query](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-sap/src/query.rs): SAP CATS is already cold-bound into typed lanes, lowered by Quack, and executed as a mask-risc grouped fold without constructing DTO source rows. This is the **SAP tile / fold precedent**.
- [dir-sim snapshot](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dir-sim/src/snapshot.rs): users and groups are distinct SoA populations keyed by **Guid128 = 16-byte identity**; only snapshot-local user/group ordinals are u16. Strings are cold. Sparse memberships are independent relation lanes.
- [OGAR hybrid correspondence](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-dir-core/src/correspond.rs): AD/Entra/EXO retain separate populations. The join is on scoped 128-bit IDs, with forward/baksync/cloud evidence, ambiguity and time skew checked. It prepares sorted `(scope, u128 key, row)` indexes and runs a numeric fold with no string decoding or hot allocation. **It is not currently a proven pure ndarray::simd_stride GUID-join kernel.**
- [OGAR Dn128](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-dir-core/src/dn128.rs): the 16-byte hierarchy coordinate is *not* the 16-byte Guid128 identity. Dn128's subtree filter is a separate direct positional/facet-stride use case.
- [ndarray simd_stride.rs](https://github.com/AdaWorldAPI/ndarray/blob/master/src/simd_stride.rs): positional `lut_u8`/`lut9` forms, no native cross-source GUID join; an optimized join requires a separate owner-admitted falsifier, and Gather/ADMISSION B remain binding.

### Minimal POC and boundary

`AD tile + Entra tile + Exchange tile + one SAP/CATS-shaped typed business tile → scoped Guid128 correspondence → prebound witness/selection → one admitted Quack/mask-risc fold over borrowed lanes → one PSCustomObject-shaped *logical* view of selected fields.`

The **PSCustomObject shape** is an ergonomic property façade, not a PowerShell execution engine or heap object population. Tile descriptors carry class/field identity and source qualification, not copied rows. Preserve 128-bit GUID bytes and endian conversion; never truncate a GUID to a DR ordinal, codebook ID, or 16-byte canonical classid key. The DR local ordinal is an ephemeral address, not a source ID. Missing, ambiguous, conflicting and scope-mismatched matches must refuse or remain visibly unknown, not choose first.

**Minimum falsifier:** two scoped populations with colliding visible labels, one 128-bit GUID match and one ambiguous two-way match, a changed snapshot, one typed SAP-like field fold, a 16-byte Dn128 prefix test (separate from ID join), an independent scalar oracle, allocation counters showing no per-row PSCustomObject/DTO hydration, and cost split into cold index preparation vs hot fold. Must prove only what current admission supports; a FACET STRIDE-specialized GUID join is a *future candidate*, not an implied done fact.

### POC-critical path

**P0 canonical Bag/scope → P1 positional/typed fold discipline + P2 only if a specifically required Bagop semantic gate exists → dir-sim + SAP-based tiny proof in lance-graph.** P3 GridCraft, P5 report-rs workbook convergence, P6 Excel integration corpus and P7 F### formula/Codegraph are **deferred consumer studies, not POC prerequisites**. P4 lance-graph-java remains strictly non-blocking evidence of folds without row-object materialization. **No DataFusion/Polars/Delta dependency is planned.**

**Status:** PROPOSED, owner ruling required. **Date:** 2026-11-05. **Mode:** plan only; no production admission.

**Operator invariant:** DR64 is exclusively the holographic representation of a Bag; Bagop is its polyfill interface. DR64 shall not be repurposed as an independent database, generic register VM, lookup service, IAM authority or spreadsheet row engine. This is the new requested architectural rule, not a claim that existing production contracts already implement it.

## Scope and semantic ownership

| Concept | Meaning / owner | Non-equivalence |
|---|---|---|
| Bag | Witnessed population, global identity, occurrence multiplicity, properties and values; lance-graph Bagop contract | Not a Vec of spreadsheet rows |
| DR64 | Holographic representation of that Bag | Not an independent service |
| DR256 × 256 | 65,536 local position geometry in existing DRISA lab | Not the entire global identity universe |
| DT64 | Typed field/method basis | Not authorization |
| SPOG G | Semantic graph concept | Not automatically a security tenant |
| OGAR | Tenant-qualified permission/actor scope | Not supplied by a customer label |
| Derived index / scratch | Disposable, rebuildable and versioned performance mechanism | Not independent identity authority |
| Cold text | Content-addressed bytes resolved at an explicit membrane | Not a per-row hot-path object |

**Authority invariant:** canonical identity, population membership, property interpretation, values and their versions have one owner. A frame/layout/value/scope witness binds every derived projection. An application cannot mint codebook ordinals and then ask DR to treat them as canonical truth. MapTable-owned routing structures are permitted only as *derived implementation machinery*, not competing authority. Report-rs CamLabels/MemKv must not become another canonical address space.

**Allowed:** borrowed SoA lanes; canonical content references; prepared DR selections; internal, version-checked global-to-local indexes; temporary Dirty masks and plan scratch; explicit XLSX/CSV/HTML/PowerShell egress.

**Refused:** per-row Cell or PSCustomObject execution stores; independent workbook codebooks; cross-tenant label equivalence; unversioned masks; row-derived hot-path Gather; DR64 as an IAM/SQL engine; speculative production promises.

## Grounding and contradictions to resolve

- [DR MapTable](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/dr.rs) has frame/TileMap mapping, including host structures. A ban on **all** auxiliary memory would contradict the current implementation.
- [DRISA #1955](https://github.com/AdaWorldAPI/lance-graph/pull/1955) demonstrates borrowed SoA planes and zero rerouting after bind in a **lab**, not a production ABI.
- [DRISA M1–M7](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/d-drisa-decision-memos-v1.md): M1 canonical SoA layout witness, M2 DR-owned index or co-addressed selections, M3 result lifetime, M5 D4/S-INGRESS remain owner decisions.
- [D-DYNBAG-0 #1906](https://github.com/AdaWorldAPI/lance-graph/pull/1906) uses Quack → mask-risc for Bag count and support. Bag multiplicity is not DISTINCT.
- [Canonical node](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/canonical_node.rs) and [ContentStore](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/content_store.rs) permit cold bytes by content identity; one substrate does not mean a one-word physical encoding for all data.
- [#1943](https://github.com/AdaWorldAPI/lance-graph/pull/1943) and [#1948](https://github.com/AdaWorldAPI/lance-graph/pull/1948) are routing/co-addressed lab evidence. NotASide, XLOG computed-mask reuse, OP-14 composition, Gather ban and ADMISSION B remain unresolved where applicable.

**Owner question:** reconcile exclusive-DR64 purpose with existing DR/DT64, DRISA, Quack and XLOG scopes. Cite each conflicting symbol and its admission status; do not silently rewrite the laws.

## Falsifiers and controls

Fixture: two differently authorized tenant populations sharing external labels, Bag duplicates, 256 permuted tiles, map and value revisions, missing/blank/empty distinctions. Independent scalar oracle knows global identities and typed values, not local slot numbers.

| Claim | Test | Disable check |
|---|---|---|
| Single authority | Delete all derived indexes; rebuild and get the same semantics | Substitute side-codebook ordinal and require failure |
| Frame agreement | Permute local slots; witness mismatch must refuse | Remove layout check to show wrong result |
| Tenant isolation | Equal labels in different scopes remain distinct | Disable scope check, observe wrong answer |
| Bag multiplicity | Duplicates count as occurrences | Collapse to DISTINCT, oracle red |
| Value lifetime | Map COUNT stays valid after value edit; predicates refuse stale value | Remove value witness, oracle red |
| Amortization | Repeated admitted folds perform zero global→local resolutions after bind | Force rerouting, counter test red |

Separate cold bind, warm execution, memory, source hydration and terminal output costs. No general speed claim from a warm isolated cell read.

## Cross-PR plan index and dependencies

| ID | Owner / plan path | Requires |
|---|---|---|
| P0 | lance-graph: this plan | Nothing |
| P1 | ndarray: docs/plans/n-facet-stride-bag-fold-v1.md | P0 proposed semantics |
| P2 | lance-graph: docs/plans/d-bagop-xverweis-polyfill-v1.md | P0 |
| P3 | gridcraft: docs/plans/gridcraft-bag-virtual-sheet-v1.md | P1, P2 |
| P4 | lance-graph-java: docs/plans/lgj-holographic-object-facade-v1.md | Non-blocking precedent: folding without materialization only |
| P5 | report-rs: docs/plans/rpt-bagop-workbook-convergence-v1.md | P2, P3; P4 optional evidence only |
| P6 | gridcraft: docs/plans/gridcraft-bagop-falsifier-program-v1.md | P0–P3, P5, P7; P4 optional evidence only |\n| P7 | gridcraft: docs/plans/gridcraft-f-method-codegraph-v1.md | P0–P3; F### method owner audit and Codegraph boundaries |

**Decision log 2026-11-05:** operator foundation captured, not yet adopted by code owners. **Next implementation PR (blocked):** narrowly scoped admitted Bagop projection after M1/M2, D4 boundary, and semantic owner decisions. Plan-only PR cannot authorize code, an alternate map, or production promotion.


## Program PR index (real draft PRs; 2026-11-05)

| Plan | Draft PR | Plan path |
|---|---|---|
| P0 | [lance-graph #1960](https://github.com/AdaWorldAPI/lance-graph/pull/1960) | This document |
| P1 | [ndarray #408](https://github.com/AdaWorldAPI/ndarray/pull/408) | docs/plans/n-facet-stride-bag-fold-v1.md |
| P2 | [lance-graph #1961](https://github.com/AdaWorldAPI/lance-graph/pull/1961) | docs/plans/d-bagop-xverweis-polyfill-v1.md |
| P3 | [gridcraft #1](https://github.com/AdaWorldAPI/gridcraft/pull/1) | docs/plans/gridcraft-bag-virtual-sheet-v1.md |
| P4 | [lance-graph-java #93](https://github.com/AdaWorldAPI/lance-graph-java/pull/93) | docs/plans/lgj-holographic-object-facade-v1.md |
| P5 | [report-rs #34](https://github.com/AdaWorldAPI/report-rs/pull/34) | docs/plans/rpt-bagop-workbook-convergence-v1.md |
| P6 | [gridcraft #2](https://github.com/AdaWorldAPI/gridcraft/pull/2) | docs/plans/gridcraft-bagop-falsifier-program-v1.md |\n| P7 | [gridcraft #3](https://github.com/AdaWorldAPI/gridcraft/pull/3) | docs/plans/gridcraft-f-method-codegraph-v1.md |

**P7 adds the F### formula-method lowering plan and a non-dilution boundary for Kythe/R2IL/Codegraph. Its method ID registry and any graph adapter remain OPEN pending owner/source verification.**\n\n**P4 is exclusively a non-blocking precedent for folding without materialization, not an architectural/runtime dependency, XVERWEIS/PSCustomObject owner or required gate.**

All seven are **draft plan-only PRs**. Links and creation are facts; no implementation/admission or owner approval is claimed. Changes across these PRs should be reviewed in dependency order, not auto-merged.
