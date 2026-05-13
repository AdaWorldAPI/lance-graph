# Sprint-4 Master Execution Plan + FMA-Heart-Click Demo Manifest

> **Branch:** `claude/lance-datafusion-integration-gv0BF`
> **Date:** 2026-05-13
> **Author:** W1 (master plan agent)
> **Pattern:** CCA2A — append-only per-agent logs; scoped output files; no direct git commits by workers
> **Status:** ACTIVE — in-progress spec for sprint-4 wave

---

## 1. Sprint Goal + Acceptance Criteria

### Sprint Goal

Convert the 11 TD entries in the 2026-05-13 TECH_DEBT.md batch into PR-ready implementation
specs, and define the FMA-heart-click end-to-end smoke-test demo manifest, so that an engineer
can pick any spec and begin coding the D-SDR Tier-2 follow-up wave without additional design
work.

### Acceptance Criteria

| # | Criterion | Owner | Verifiable by |
|---|---|---|---|
| AC-1 | All 12 worker specs (W2..W12) exist at `.claude/specs/td-*.md` and `.claude/specs/fma-*.md` | W2..W12 | `ls .claude/specs/*.md` |
| AC-2 | Each spec covers: problem statement, PR shape, LOC estimate, tests, cross-refs | W2..W12 | Meta agent review |
| AC-3 | P0 TDs (TD-Q2-STUBS, TD-API-DRIFT, TD-SDR-PR-FOLLOWUP, TD-SDR-CONSUMER-PUSH) have actionable PR-ready specs with exact file paths and diffs described | W2, W3, W7 | Meta-1 review checklist |
| AC-4 | FMA-heart-click manifest (W11) names all crates, datasets, query cells, and render targets | W11 | W1 cross-check (this doc sec 4) |
| AC-5 | PR-sequencing graph (W12) has no circular dependencies and correctly schedules P0s first | W12 | W1 cross-check (this doc sec 3) |
| AC-6 | W1 master plan cross-spec coordination flags are visible to M1/M2 meta agents | W1 | This doc sec 5 |
| AC-7 | Risk register (this doc sec 6) has >=3 entries with mitigation paths | W1 | Human reviewer |
| AC-8 | No git commits or pushes from worker agents -- main thread aggregates | All workers | `git log` check |

---

## 2. Worker Roster Table (with Dependency Arrows)

| Agent | TD-ID | Deliverable | Output path | Target size | Deps |
|---|---|---|---|---|---|
| **W1** | meta | Sprint-4 master execution plan + FMA demo manifest | `.claude/specs/sprint-4-execution-plan.md` | ~12 KB | none (reads first) |
| **W2** | TD-Q2-STUBS-DEDUP-1 (P0) | Q2 stubs dedup spec | `.claude/specs/td-q2-stubs-dedup.md` | ~8 KB | => W11 (FMA pipeline depends on stubs being resolved) |
| **W3** | TD-API-DRIFT-MIDFLIGHT-1 (P0) | D-SDR API deprecation playbook | `.claude/specs/td-api-drift-deprecation.md` | ~10 KB | => W7 (deprecation path informs PR sequencing) |
| **W4** | TD-SUPER-DOMAIN-SUBCRATES-1 (P1) | Super-domain subcrate cascade | `.claude/specs/td-super-domain-subcrates.md` | ~12 KB | => W6 (thinking-engine wire needed before subcrate scaffold) |
| **W5** | TD-SIMD-CALLCENTER-BATCH-PATHS-1 (P2) | SIMD callcenter batch retrofit | `.claude/specs/td-simd-callcenter-batch.md` | ~8 KB | => W4 (subcrate canonical location for SIMD batch paths) |
| **W6** | TD-THINKING-ENGINE-UNWIRED-1 (P1) | thinking-engine UnifiedBridge wire-up | `.claude/specs/td-thinking-engine-wire.md` | ~12 KB | => W4 (super-domain scaffold composes against thinking-engine) |
| **W7** | TD-SDR-PR-FOLLOWUP-1 + TD-SDR-CONSUMER-PUSH-1 (P0+P0) | D-SDR PR follow-up + consumer push release plan | `.claude/specs/td-sdr-pr-release.md` | ~8 KB | => W3 (deprecation path), => W8 (audit persist comes after follow-up PR) |
| **W8** | TD-SDR-AUDIT-PERSIST-1 (P1) | Audit sink spec -- Lance + JSONL | `.claude/specs/td-sdr-audit-persist.md` | ~10 KB | => W7 (JSONL sink unblocked after follow-up PR), => W10 (slot fix feeds audit path) |
| **W9** | TD-SDR-FAMILY-HYDRATION-1 (P2) | Family hydration + reverse-lookup TTL | `.claude/specs/td-sdr-family-hydration.md` | ~8 KB | blocked by external OGIT MCP scope expansion; spec outlines unblocked portions |
| **W10** | TD-SDR-SLOT-TRUNC-1 + TD-SDR-BRIDGE-ERR-AUDIT-1 (P1+P2) | Slot widen u16 + bridge-err audit fix | `.claude/specs/td-sdr-slot-and-bridgeerr.md` | ~8 KB | => W8 (bridge-err audit event needs JSONL sink to be meaningful) |
| **W11** | FMA smoke-test anchor | FMA heart-click end-to-end smoke test (75K OWL -> q2 3D render) | `.claude/specs/fma-heart-click-smoke.md` | ~12 KB | => W2 (stubs dedup), => W4 (super-domain subcrates), => W6 (thinking-engine) |
| **W12** | sequencing meta | Cross-repo PR sequencing graph | `.claude/specs/sprint-4-pr-graph.md` | ~6 KB | reads W2..W11 specs (runs last or in parallel with M1) |
| **M1** | review | Per-worker spec assessment | `.claude/board/sprint-log-4/meta-1-review.md` | ~6 KB | reads all agent-W*.md + specs |
| **M2** | synthesis | Cross-spec coherence + governance updates | `.claude/board/sprint-log-4/meta-2-review.md` | ~5 KB | reads M1 + all specs |

### Dependency arrow summary

```
W3 ---> W7 ---> W8 ---> W10
W6 ---> W4 ---> W5
W2 ---> W11
W4 ---> W11
W6 ---> W11
W12 (reads W2..W11, runs after or in parallel)
M1  (reads W1..W12 outputs)
M2  (reads M1)
```

---

## 3. PR-Sequencing Recommendation

### Wave 0 -- Governance gate (must land first, ~15 min work)

| PR | Spec | TD | Description |
|---|---|---|---|
| PR-S4-0a | W7 sec1 | TD-SDR-PR-FOLLOWUP-1 | Open follow-up PR for D-SDR-3..5 commits on `claude/lance-datafusion-integration-gv0BF`; board hygiene (LATEST_STATE + STATUS_BOARD + PR_ARC_INVENTORY) in same commit |
| PR-S4-0b | W7 sec2 | TD-SDR-CONSUMER-PUSH-1 | Push medcare-rs `31e999b` + smb-office-rs `342f601` and open consumer PRs in parallel; no lance-graph surface changes needed |

**Rationale:** Wave 0 unblocks all consumers from the "5 commits ahead of main, no PR" limbo.
PR-S4-0a opens the anchor PR; PR-S4-0b confirms the consumer wirings compile against merged-#363
surface. These two ship before any other Wave.

### Wave 1 -- P0 blockers (can parallel-build once Wave 0 merges)

| PR | Spec | TD | Description |
|---|---|---|---|
| PR-S4-1a | W2 | TD-Q2-STUBS-DEDUP-1 | q2 workspace: replace stubs with `pub use` re-exports; add 2 integration tests; ~60 LOC |
| PR-S4-1b | W3 | TD-API-DRIFT-MIDFLIGHT-1 | Deprecation annotations + `must_use` + `migration::*` module + SHA-pin instructions + CHANGELOG; ~80 LOC |

**Rationale:** PR-S4-1a clears the FMA compilation blocker. PR-S4-1b institutes the operational
discipline that prevents the next mid-air migration breakage. Both are self-contained and don't
require each other, so they can merge in any order within Wave 1.

### Wave 2 -- P1 architectural (sequence within wave matters)

| PR | Spec | TD | Description | Sequence |
|---|---|---|---|---|
| PR-S4-2a | W6 | TD-THINKING-ENGINE-UNWIRED-1 | cognition_bridge module (~300 LOC + 5 tests) wiring thinking-engine into UnifiedBridge path | First in wave |
| PR-S4-2b | W10 sec1 | TD-SDR-SLOT-TRUNC-1 | `debug_assert!(entity_type_id < 256)` guard in `owl_from_schema_ptr`; ~5 LOC | After 2a (audit path is cleaner) |
| PR-S4-2c | W8 | TD-SDR-AUDIT-PERSIST-1 | `JsonLinesAuditSink` + replay-verify schema (~200 LOC + 7 tests) | After 2b (slot fix reduces aliasing in audit stream) |
| PR-S4-2d | W4 | TD-SUPER-DOMAIN-SUBCRATES-1 | MedCare migration finalization (PR 1 of 5-cascade); collapse parallel auth paths; ~900 LOC cascade total | After 2a (composes thinking-engine surface) |

### Wave 3 -- P2 performance + partial unblocks

| PR | Spec | TD | Description |
|---|---|---|---|
| PR-S4-3a | W5 | TD-SIMD-CALLCENTER-BATCH-PATHS-1 | ~200 LOC SIMD batch replacements + 5 micro-benchmarks |
| PR-S4-3b | W10 sec2 | TD-SDR-BRIDGE-ERR-AUDIT-1 | Emit `BridgeError`-tagged `UnifiedAuditEvent` before short-circuit; ~30 LOC + 2 tests |
| PR-S4-3c | W9 | TD-SDR-FAMILY-HYDRATION-1 | OGIT TTL hydration baker (unblocked portions); full resolution awaits OGIT MCP scope expansion |

### Wave 4 -- FMA convergence target (after all Wave 1-3 merge)

| PR | Spec | TD | Description |
|---|---|---|---|
| PR-S4-4 | W11 | FMA smoke-test | End-to-end: stalwart OWL ingest -> OGIT routing -> UnifiedBridge auth -> EWA-Sandwich -> q2 3D render; see sec 4 for full manifest |

### P0 gates summary

The four P0 TDs ship in this order:

```
TD-SDR-PR-FOLLOWUP-1    (Wave 0a)
  |
  +--> TD-SDR-CONSUMER-PUSH-1 (Wave 0b, parallel)
         |
         +--> TD-Q2-STUBS-DEDUP-1       (Wave 1a, can parallel with 0b)
         +--> TD-API-DRIFT-MIDFLIGHT-1  (Wave 1b, can parallel with 1a)
```

---

## 4. FMA-Heart-Click Demo Manifest

### What the demo proves

A user clicks the "heart" anatomical region in the q2 3D anatomy viewer. The click:

1. Resolves to an FMA OWL entity (`FMA:7088 Heart`) via the OGIT ontology routing table.
2. Issues a Cypher query through `q2::notebook-query` -> `lance-graph-planner::PolyglotDetector`.
3. Passes through `UnifiedBridge<MedcareBridge>` auth/audit chain (HIPAA-scoped).
4. Propagates edges through the EWA-Sandwich substrate in `lance-graph` (SPO + blasgraph).
5. Returns a subgraph (cardiac chambers + coronary arteries + connected anatomical structures).
6. Renders the subgraph as a 3D highlight in q2's `cockpit-server` Foundry-parity view.

### Dataset

| Item | Detail |
|---|---|
| Ontology | FMA (Foundational Model of Anatomy) 75,000-entity OWL/RDF |
| Source | `http://sig.biostr.washington.edu/ftp/pub/fma/release/` (public, CC-licensed) |
| Format | OWL/XML -> `stalwart` RDF parser -> SPO triples |
| Ingest size | ~75K entities, ~400K triples (Heart subgraph: ~2K entities, ~8K triples) |
| Test anchor entity | `FMA:7088` (Heart) -- canonical smoke-test pivot |

### Crates touched (full list)

| Crate | Role in demo path | TD that gates it |
|---|---|---|
| `stalwart` (external, OWL/RDF) | Parse FMA OWL/XML -> RDF triples | none (already usable) |
| `lance-graph` (`graph/spo/`) | Ingest RDF triples as SPO store; EWA-Sandwich edge propagation | TD-Q2-STUBS-DEDUP-1 (canonical type surface) |
| `lance-graph-planner` | Cypher query dispatch -> DataFusion plan | TD-Q2-STUBS-DEDUP-1 |
| `lance-graph-contract` | `UnifiedStep`, `OrchestrationBridge` trait, `BridgeSlot` | none |
| `lance-graph-callcenter` | `OgitFamilyTable` lookup (FMA entities -> Healthcare SuperDomain) | TD-SDR-FAMILY-HYDRATION-1 (partial) |
| `ndarray` (canonical) | `Fingerprint<256>`, SIMD cosine for entity similarity search | TD-Q2-STUBS-DEDUP-1 |
| `thinking-engine` | Cognitive substrate: `role_tables`, `persona`, `osint_bridge` wired into UnifiedBridge | TD-THINKING-ENGINE-UNWIRED-1 |
| `medcare-rs` (medcare-bridge subcrate) | `UnifiedBridge<MedcareBridge>` auth chain; HIPAA super-domain salt | TD-SDR-CONSUMER-PUSH-1, TD-SUPER-DOMAIN-SUBCRATES-1 |
| `q2::stubs/lance-graph` -> canonical | Cypher cell dispatch in q2 graph-notebook | TD-Q2-STUBS-DEDUP-1 |
| `q2::cockpit-server` | 3D anatomy render + subgraph highlight overlay | TD-Q2-STUBS-DEDUP-1 |

### Cypher cell (q2 graph-notebook)

```cypher
// FMA Heart-Click smoke query
// Dispatched by q2::notebook-query -> lance-graph-planner::PolyglotDetector -> Strategy #1 (CypherParse)
MATCH (heart:FMAEntity {fma_id: "FMA:7088"})
OPTIONAL MATCH (heart)-[:PART_OF|:REGIONAL_PART_OF|:CONSTITUTIONAL_PART_OF*1..2]->(parent)
OPTIONAL MATCH (heart)<-[:PART_OF|:REGIONAL_PART_OF|:CONSTITUTIONAL_PART_OF*1..2]-(child)
OPTIONAL MATCH (heart)-[:ARTERIAL_SUPPLY]->(artery)
RETURN heart, parent, child, artery
LIMIT 500
```

### Auth/audit chain

```
q2::notebook-query
  |
  +--> lance-graph-planner (Cypher -> DataFusion plan)
         |
         +--> UnifiedBridge<MedcareBridge>::authorize_read(
                actor        = "q2-demo-user",
                public_name  = "FMA:7088",
                super_domain = SuperDomain::Healthcare
              )
                |
                +--> OGIT family lookup (lance-graph-callcenter::OgitFamilyTable)
                +--> Policy::evaluate (HIPAA-scoped, role = Viewer)
                +--> emit UnifiedAuditEvent (-> JsonLinesAuditSink)
                +--> AuthDecision::Allow -> DataFusion scan proceeds
```

### EWA-Sandwich edge propagation

**CORRECTION (post-sprint-write):** EWA = Elliptical Weighted Average (Heckbert), not "Efficient Weighted Adjacency". EWA-Sandwich is **Pillar 6** of the JC pillars framework — Σ push-forward `M·Σ·Mᵀ` along multi-hop edge paths certifying PSD-preservation + Köstenberger-Stark concentration rate. Already implemented at `crates/jc/src/ewa_sandwich.rs` (450 LOC) + `crates/lance-graph-contract/src/sigma_propagation.rs` (488 LOC). See `.claude/plans/jc-pillars-runtime-wiring-v1.md` + ERRATUM, and `crates/jc/examples/osint_edge_traversal.rs` for the canonical OSINT-route demo. PR #288 (Σ-codebook viability, R² = 0.9949) certifies the 256-entry codebook + 1-byte sidecar (rules out CausalEdge64 8→16 expansion).

The Heart subgraph is traversed via:

1. `graph/spo/` SPO triple store -- exact triple lookups for `PART_OF` / `ARTERIAL_SUPPLY` edges.
2. `graph/blasgraph/` -- sparse CSR adjacency matrix for k-hop neighborhood expansion (up to 2 hops per Cypher `*1..2` pattern).
3. `lance-graph-planner::SigmaBandScan` strategy -- range-predicate pushdown for entity ID band.
4. Result: `RecordBatch` of `(heart_id, related_id, edge_type, hop_distance)` streamed to q2.

### Render target

`q2::cockpit-server` receives the subgraph as Arrow `RecordBatch`. The 3D anatomy viewer:

- Highlights `FMA:7088` (Heart) in red.
- Colors first-hop `PART_OF` children (atria, ventricles) in orange.
- Colors second-hop children and `ARTERIAL_SUPPLY` edges in yellow.
- Renders coronary artery tree as a graph overlay on the 3D mesh.

### Demo smoke-test acceptance criteria

| Check | Pass condition |
|---|---|
| Compile | `cargo check -p q2-notebook-query` with canonical `lance-graph` dep (not stub) |
| Auth chain | `UnifiedBridge<MedcareBridge>::authorize_read` returns `AuthDecision::Allow` for `FMA:7088` |
| Audit event | `JsonLinesAuditSink` emits >=1 event with `super_domain = Healthcare`, `actor_role_hash != 0` |
| Query result | Cypher cell returns >=10 nodes including `FMA:7088` (Heart) and >=1 artery node |
| Render | cockpit-server renders subgraph highlight without panic |
| Latency | Full round-trip (click -> highlight) <=500 ms on dev hardware |

---

## 5. Cross-Spec Coordination Flags

### FLAG-1: W2 <-> W11 -- FMA compilation gate

**Dependency:** W11's smoke-test spec (`fma-heart-click-smoke.md`) cannot describe a
compilable demo path until W2's stubs-dedup spec (`td-q2-stubs-dedup.md`) is implemented.
W11 must reference W2's exact Cargo.toml change as a prerequisite.

**Action for W11:** Quote W2 sec1 (Cargo.toml diff) as a literal prerequisite. List
`TD-Q2-STUBS-DEDUP-1` as a blocking gate in the demo manifest's acceptance criteria.

**Action for W2:** The two integration tests (`q2_lance_graph_canonical_test.rs`,
`q2_ndarray_simd_dispatch_test.rs`) must be written so W11's smoke-test can build on top of
them without duplication.

**Coordination point:** W2 and W11 may be written in parallel; W2's Cargo.toml section must
be stable before W11 finalises the compile-check acceptance criterion.

---

### FLAG-2: W6 <-> W4 -- thinking-engine wire required before super-domain scaffold

**Dependency:** W4's super-domain subcrate cascade spec (`td-super-domain-subcrates.md`)
recommends composing against `thinking-engine::role_tables`, `::persona`, and `::osint_bridge`
for per-super-domain HKDF + hard-lock partner declarations (D-SDR-13/17). If W4 is written
before W6 finalises the cognition_bridge module shape, W4's spec will either duplicate
thinking-engine or describe a different integration point.

**Action for W6:** W6 must define the public surface of `cognition_bridge` (trait name,
method signatures) before W4 can reference it as a composition target.

**Action for W4:** W4 should treat `cognition_bridge` as a soft dependency -- describe the
composition as "TBD pending W6 cognition_bridge surface" if W6 is not yet complete when W4
is authored. Do NOT duplicate thinking-engine logic in the subcrate spec.

**Coordination point:** Ideally W6 ships PR-S4-2a before W4 writes the cascade PR shape
for subcrates 2-5 (hiro-rs, hubspot-rs, woa-rs). PR 1 of the cascade (MedCare migration
finalization) can be written without W6 being complete.

---

### FLAG-3: W3 <-> W7 -- deprecation path informs release sequencing

**Dependency:** W7's release plan (`td-sdr-pr-release.md`) must incorporate the SHA-pinning
and `migration::*` module design from W3's deprecation playbook
(`td-api-drift-deprecation.md`). If W7 is written first, it will likely recommend "just push
the PRs" without the SHA-pinning discipline, which is exactly the failure mode TD-API-DRIFT
documents.

**Action for W3:** W3 must explicitly state which SHA consumers should pin to during the
migration window (post-#363 merge SHA `421e71e`). W3's spec is the source of truth for
this value; W7 references it.

**Action for W7:** W7's consumer PR instructions (for medcare-rs + smb-office-rs) must
include a step: "Update `Cargo.toml` to pin `lance-graph` at SHA `421e71e` (per W3
deprecation playbook) before opening the consumer PR."

**Coordination point:** W3 should be authored (or at least its SHA-pinning section finalized)
before W7 writes the consumer PR step-by-step instructions.

---

### FLAG-4: W8 <-> W10 -- audit sink meaningfulness gated on slot fix

**Dependency:** W10's slot truncation fix (`debug_assert!(entity_type_id < 256)`) prevents
audit events from carrying aliased `super_domain` values in basins that exceed 256 entities.
W8's JSONL sink will persist those aliased values if W10 ships after W8.

**Action for W10:** W10 sec1 (slot fix) must note that `JsonLinesAuditSink` should ideally
not ship to production until the slot guard is in place. If W8 ships first, add a comment
in `JsonLinesAuditSink` marking the `super_domain` field as potentially aliased for basins
>256 entities.

**Action for W8:** W8's spec should note "this sink should be treated as draft until
TD-SDR-SLOT-TRUNC-1 (W10) lands -- the `super_domain` field in emitted events may alias for
large basins."

---

### FLAG-5: W9 external blocker -- OGIT MCP scope expansion

**Dependency:** W9's family hydration spec (`td-sdr-family-hydration.md`) cannot describe
a complete payoff because `TD-SDR-FAMILY-HYDRATION-1` is explicitly blocked on
`AdaWorldAPI/OGIT` MCP scope expansion for D-SDR-6 (Hiro entities) and D-SDR-7 (HubSpot
entities).

**Action for W9:** Write the spec in two parts: (a) what can ship today (TTL baker skeleton,
static table generator harness), (b) what is blocked pending OGIT scope expansion. The
unblocked portion enables partial hydration for the Healthcare FMA subgraph even without
Hiro/HubSpot entities.

**Action for W12 (sequencing):** Mark W9 as "partial ship" in the PR graph -- wave 3 PR
carries the skeleton; full payoff is deferred to a sprint-5 wave contingent on OGIT MCP.

---

## 6. Risk Register

### RISK-1: q2 stub replacement breaks internal q2 consumers

**Probability:** Medium. The stubs expose a "minimal vertex/edge CRUD + zeros/ones/matmul"
surface; canonical lance-graph is a strict superset, but call sites may use different
function signatures or module paths.

**Impact:** High -- breaks q2 CI, delays W11 FMA smoke-test.

**Mitigation:**
- W2's spec must enumerate every call site in q2 that uses the stub API before proposing
  the replacement (not just the re-export shim).
- The replacement PR should be drafted as: (1) add canonical dep, (2) add re-export shim
  with identical stub surface, (3) compile check, (4) migrate call sites, (5) remove shim.
  This gives a compile-safe intermediate state.
- W12 should flag the q2 PR as "review-required by q2 crate owner before merge."

**Residual risk:** Low after step-by-step shim approach.

---

### RISK-2: thinking-engine cognition_bridge surface drifts during sprint

**Probability:** Medium. W6 is a ~300-LOC spec + implementation PR. If W4 and W11 reference
the cognition_bridge surface before W6 stabilizes it, those specs will need revision.

**Impact:** Medium -- W4 subcrate cascade and W11 FMA manifest may need one revision pass.

**Mitigation:**
- W6 should publish its cognition_bridge public trait surface in its spec before W4 and W11
  finalize their cross-references (FLAG-2 above).
- W4 should use a forward-reference placeholder ("pending W6 sec2 cognition_bridge trait")
  rather than guessing the interface.
- M1 meta review specifically checks W4 + W11 cross-references against W6's finalized spec.

**Residual risk:** One spec revision pass by M2 synthesis agent.

---

### RISK-3: OGIT MCP scope expansion delay blocks W9 completely

**Probability:** High (already blocked per TECH_DEBT.md TD-SDR-FAMILY-HYDRATION-1).

**Impact:** Medium -- FMA smoke-test can proceed with partial hydration (Healthcare FMA
entities map to `SuperDomain::Healthcare` without needing Hiro/HubSpot). But
`FAMILY_TO_SUPER_DOMAIN` will still return `Unknown` for non-Healthcare families in the demo,
which shows up as `super_domain: Unknown` in audit events.

**Mitigation:**
- W9 writes a skeleton spec complete for the unblocked Healthcare FMA portion.
- W11 FMA smoke-test acceptance criteria explicitly scope to Healthcare super-domain only
  (`assert_eq!(event.super_domain, SuperDomain::Healthcare)` when FMA entities are routed).
- A spike to manually seed the static `FAMILY_TO_SUPER_DOMAIN` table for FMA families
  (without OGIT TTL hydration) is included in W9's spec as a workaround.
- Sprint-5 planning gate: OGIT MCP scope expansion is a prerequisite for D-SDR-6/7.

**Residual risk:** Demo succeeds at reduced scope; full hydration deferred to sprint-5.

---

## 7. Open Questions for Human Reviewer

### Q1 -- OGIT MCP scope: timeline and owner?

The dependency on `AdaWorldAPI/OGIT` MCP scope expansion for D-SDR-6 (Hiro entities) and
D-SDR-7 (HubSpot entities) is referenced by three TDs (TD-SDR-FAMILY-HYDRATION-1,
TD-SUPER-DOMAIN-SUBCRATES-1 Hiro/HubSpot slots). Is there a timeline or assignee for this
expansion? If not, sprint-4's W9 spec should formally propose a manual seeding workaround
as a permanent unblock.

### Q2 -- FMA OWL source: snapshot or live?

The FMA demo manifest proposes ingesting FMA OWL from the Washington biostr FTP site. Should
the sprint target a committed snapshot (checked into a test-fixtures repo or GitHub Release
asset) for reproducibility, or is a live download acceptable for the smoke-test? The snapshot
approach avoids FTP availability risk at demo time.

### Q3 -- q2 crate ownership for stub replacement (W2)?

TD-Q2-STUBS-DEDUP-1 requires a PR to the `q2` repo (not `lance-graph`). Does the human
reviewer own the q2 repo merge gate, or is there a separate q2 crate owner who must approve
W2's PR before it can merge? The sprint plan currently treats W2 as a unilateral spec, but
the actual PR merge requires q2 repo access.

### Q4 -- Wave 0 urgency: should TD-SDR-PR-FOLLOWUP-1 be a pre-sprint gate?

The Wave 0 step (opening the follow-up PR for D-SDR-3..5) is estimated at ~15 minutes of
mechanical work. Given that it unblocks all consumer PRs, should this be done by the human
reviewer before worker agents begin sprint-4 spec authoring, rather than waiting for W7 to
spec it? The risk is that medcare-rs and smb-office-rs remain in an ambiguous state for the
full duration of spec authoring.

### Q5 -- Slot truncation scope: is 256-entity-per-basin cap a hard limit or a soft assumption?

TD-SDR-SLOT-TRUNC-1 notes the truncation is "lossless within the sec16 addressable domain
(<=256 entries per family; SGO meta excluded per sec9.3)." The FMA ontology's Heart subgraph
has ~2K entities across the cardiac family. Does the cardiac FMA family exceed the 256-entity
cap? If so, the `debug_assert!` guard will fire during the FMA demo, meaning W10's fix must
include not just the assert but a wider `OwlIdentity` or basin partition -- which is a larger
change than the current ~5 LOC estimate.

---

## Appendix: TD Cross-Reference Table

| TD-ID | Priority | Worker | Wave | LOC estimate | External blockers |
|---|---|---|---|---|---|
| TD-SDR-PR-FOLLOWUP-1 | P0 | W7 | Wave 0 | ~0 (mechanical) | none |
| TD-SDR-CONSUMER-PUSH-1 | P0 | W7 | Wave 0 | ~0 (push + PR) | TD-SDR-PR-FOLLOWUP-1 (Wave 0a) |
| TD-Q2-STUBS-DEDUP-1 | P0 | W2 | Wave 1 | ~60 LOC + 2 tests | none |
| TD-API-DRIFT-MIDFLIGHT-1 | P0 | W3 | Wave 1 | ~80 LOC | none |
| TD-THINKING-ENGINE-UNWIRED-1 | P1 | W6 | Wave 2 | ~300 LOC + 5 tests | none |
| TD-SDR-SLOT-TRUNC-1 | P1 | W10 | Wave 2 | ~5 LOC (see Q5) | none |
| TD-SDR-AUDIT-PERSIST-1 | P1 | W8 | Wave 2 | ~200 LOC + 7 tests | TD-SDR-PR-FOLLOWUP-1 |
| TD-SUPER-DOMAIN-SUBCRATES-1 | P1 | W4 | Wave 2 | ~900 LOC cascade | TD-THINKING-ENGINE-UNWIRED-1 |
| TD-SIMD-CALLCENTER-BATCH-PATHS-1 | P2 | W5 | Wave 3 | ~200 LOC + 5 benchmarks | TD-SUPER-DOMAIN-SUBCRATES-1 |
| TD-SDR-BRIDGE-ERR-AUDIT-1 | P2 | W10 | Wave 3 | ~30 LOC + 2 tests | TD-SDR-AUDIT-PERSIST-1 |
| TD-SDR-FAMILY-HYDRATION-1 | P2 | W9 | Wave 3 (partial) | ~150 LOC | OGIT MCP scope expansion |
| FMA smoke-test | -- | W11 | Wave 4 | integration test suite | Waves 1-3 complete |
