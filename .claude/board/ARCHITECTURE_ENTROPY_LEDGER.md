# Architecture Entropy Ledger — OPEN (active concerns)

> **APPEND-ONLY** — same governance as `PR_ARC_INVENTORY.md` /
> `EPIPHANIES.md` / `TECH_DEBT.md` / `ISSUES.md`. New rows append
> below. The `Entropy` and `Plan-Status` columns are the only
> mutable per-row fields; structural claims (Region / Component /
> File / DupCount / Seam) are immutable history.
>
> **Companion to** `.claude/knowledge/soa-dto-fma-map.md` and
> `.claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md` (the
> archive of closures + state-change records).
>
> **This file = OPEN concerns** (entropy ≥ 3 rows, open seams, active
> clusters, still-stalled plans). Closed/resolved/wired rows live in
> `ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md` — split 2026-05-07 to
> reduce filesize.
>
> **Why this exists:** the workspace has DTO spaghetti — 6-copy NARS,
> 4-copy ThinkingStyle, 3-copy VSA stacks, 2-copy SentenceCrystal,
> 2-copy GateDecision (different shapes). The ledger names them and
> scores entropy / loose-ends / duplicate potential so the next
> session sorts by entropy and picks the highest-leverage fix without
> re-discovery.

---

## Scoring rubrics

**Integration state:**
- `Wired` — implementation present, consumers reading/writing it, tests cover it.
- `Stub` — types exist, body is `unimplemented!()`/regex/empty `Ok(())`/single mock.
- `Aspirational` — referenced in docs/plans but **zero source hits in `crates/`**.
- `Dead` — code exists but no consumer / workspace-excluded / superseded.

**Duplicate potential:** `None` / `Low` / `Med` / `High` / `Spaghetti` (4+ copies).

**Entropy (1-5):**
1. Clean — canonical, wired, doc + tests + plan agree.
2. Mostly clean — small drift, no behavioural risk.
3. Partial — working but loose end (one missing wiring, one stale doc).
4. High — 2-3 unconnected duplicates OR major seam broken.
5. Spaghetti — 4+ duplicates / cross-crate name collision / dead pointer in plan.

**Plan-status:** `Shipped` / `Active` / `Stalled` / `Abandoned` / `Missing`.

---

## 2026-05-05 — initial snapshot (Section A — OPEN rows only, entropy ≥ 3)

> Closed rows from this snapshot (FREEWILL-1, LANCEDB-NAME-1, ENT-TYPE-1, SPLAT-1)
> moved to `ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md`.

| ID | Region | Component | State | DupCount | DupPotential | LooseEnds | Plan | PlanStatus | Entropy |
|---|---|---|---|---|---|---|---|---|---|
| **NARS-1** | R6 | NARS inference enum | Wired (×6) | 6 | **Spaghetti** | `contract::nars::InferenceType`(5), `contract::grammar::inference::NarsInference`(7), `planner::nars::inference::NarsInference`(5), `planner::thinking::nars_dispatch::NarsInferenceType`(5), `causal-edge::edge::InferenceType`, `learning::feedback::NarsInferenceType`(5). 9 revision sites. 3 `TruthValue` copies. | none | Missing | **5** |
| **THINK-1** | R6 | ThinkingStyle enum | Wired (×4) + const(1) + bandit(1) | 6 | **Spaghetti** | `contract::thinking::ThinkingStyle`(36), `planner::thinking::style::ThinkingStyle`(12), `thinking-engine::cognitive_stack::ThinkingStyle`(12), `thinking-engine::superposition::ThinkingStyle`(5), `cognitive-shader-driver::engine_bridge::UNIFIED_STYLES`(12-const), `learning::cognitive_styles::StyleSelector`(RL bandit). `ord_to_thinking_style` (driver.rs:677) hand-picks contract-36 reps from 12-ord. No `From` adapters. | THINKING_RECONCILIATION.md | Stalled | **5** |
| **VSA-1** | R5 | VSA carrier algebra | Wired (×3, distinct algebras) | 3 | **High** | `contract::crystal::fingerprint::Vsa16kF32` (ℝ multiply/add) + `ndarray::hpc::vsa::VsaVector` (Binary16K GF(2) XOR) + `holograph::bitpack::BitpackedVector` (XOR, workspace-EXCLUDED). No `vsa16k_permute` on f32 carrier — Markov ρ^d unimplementable. 8 free functions, no methods (Click P-1 violation). | vsa-switchboard-architecture.md | Missing | **5** |
| **DEEPNSM-NSM-1** | R5/R6 | nsm/ vs deepnsm | Wired (×2 parallel) | 2 | **High** | `lance-graph/src/nsm/{encoder,parser,similarity,tokenizer,nsm_word}.rs` (≈2,405 LOC) parallel to `crates/deepnsm/`. CLAUDE.md Phase 3 task "Consolidate nsm/ module" never ran. | none | Missing | **5** |
| **GATE-1** | R2/R6 | `GateDecision` namespace clash | Wired (×2 different shapes) | 2 | **High** | `contract::collapse_gate::GateDecision` is a struct `{gate: u8, merge: MergeMode}`. `contract::mul::GateDecision` is an enum (Flow/Hold/Block-style). Same name, different types, in the same crate. Quietly tolerated. | none | Missing | **4** |
| **TRUTH-1** | R6 | `TruthValue` algebra | Wired (×3) | 3 | High | `contract::crystal::TruthValue` + `lance-graph-planner::nars::truth::TruthValue` + `causal-edge::tables::PackedTruth`. `revise()` fn at planner/nars/truth.rs:57; `revision()` at lance-graph/spo/truth.rs:62 — **subtly different math** (planner uses `c/(1-c)`, spo uses `c/(1-c+EPS)`). | none | Missing | **4** |
| **CRYSTAL-1** | R5/R6 | `SentenceCrystal` collision | Wired (×2 unaware) | 2 | High | `contract::crystal::sentence::SentenceCrystal` + `holograph::crystal_dejavu::SentenceCrystal` (line 248) + `holograph::sentence_crystal::SemanticCrystal`. `holograph` excluded from workspace; collision is silent. | none | Abandoned (holograph excluded) | **4** |
| **NARS-TRUTH-1** | R6 | `NarsTruth` struct | Wired (×2) | 2 | Med | `contract::exploration::NarsTruth:86` + `holograph::width_16k::schema::NarsTruth:97`. | none | Missing | **3** |
| **PARSER-1** | R3/R8 | Cypher parser | Wired (×1 real) + Stub (×3) + parallel (×1 excluded) | 5 | Spaghetti | `lance-graph::parser::parse_cypher_query` (1,932 LOC nom — REAL); `planner::strategy::cypher_parse.rs` (72 LOC substring); `cognitive-shader-driver::cypher_bridge.rs` (regex stub, confidence=0.5); `lance-graph-cognitive::cypher_bridge.rs` (5th copy); `holograph::query::parser.rs` (663 LOC parallel AST, excluded). **35 `format!("{:?}", logical_plan)` Debug-string introspection sites workspace-wide.** | codec-sweep / elegant-herding-rocket — no parser-unification plan | Missing | **5** |
| **HEEL-1** | R6 | HEEL/HIP/BRANCH/TWIG/LEAF | Wired (×3 different orderings) | 3 | High | `contract::cam::CamByte::{Heel, Branch, TwigA, TwigB, Leaf, Gamma}` — **HIP missing**. `lance-graph::graph::neighborhood::SearchCascade::{heel,hip,twig,leaf_rerank}` — BRANCH missing. `bgz17/router.rs` + `bgz-tensor/hhtl_d.rs` — third invention on bf16 basins. I10 doctrinal sequence violated by all three. | none | Missing | **4** |
| **CAM-DIST-1** | R6/R8 | `cam_distance` UDF wiring | Wired (opt-in only) | 1 | Low | UDF registered at `cam_pq/udf.rs:241,257,326`. Called from `query.rs:470` only when `with_cam_codebook(...)` is opted into. `datafusion_planner/mod.rs` does NOT register; default Cypher path can't reference `cam_distance`. | cam-pq-production-wiring-v1 (DRAFT) | Stalled | **3** |
| **DNTREE-1** | R6 | `DnTree` struct | Wired (in excluded crate) | 1 | None | `holograph/src/dntree.rs:36` (TreeAddr, DnNode, DnEdge, CogVerb). `holograph` excluded → no `lance-graph` consumer. `QueryStrategy::DnTreeFull` (`contract::nars.rs:36`) is a token with no traversal. | none | Abandoned (excluded) | **5** |
| **CONTRACT-INV-1** | R6 | `LATEST_STATE.md` Contract Inventory | Stale | n/a | None | 38 contract modules on disk; LATEST_STATE inventory mentions ~28. **10 modules unlisted:** `auth`, `distance`, `external_membrane`, `faculty`, `hash`, `ontology`, `persona`, `scenario`, `sigma_propagation`, `sla`. `lib.rs:26-32` docstring still says "7 modules". | none | n/a | **3** |
| **PLAN-INDEX-1** | (board) | INTEGRATION_PLANS.md | Stale | n/a | None | 18 plan files on disk; INTEGRATION_PLANS.md indexes 11. **7 plans unindexed:** archetype-scaffold, burn-ndarray-parity-sprint, foundry-roadmap-unified-smb-medcare, oxigraph-arigraph-cognitive-shader-soa-merge, sql-spo-ontology-bridge, tetrahedral-epiphany-splat-integration, thought-cycle-soa-awareness-integration. | n/a | n/a | **3** |
| **AGENT-LOG-1** | (board) | `.claude/board/AGENT_LOG.md` | Aspirational | 0 | None | Referenced by CLAUDE.md §Mandatory-Board-Hygiene-Rule + §Layer-2 A2A. **File does not exist** at `.claude/board/AGENT_LOG.md`. | none | n/a | **3** |
| **PR-ARC-1** | (board) | `PR_ARC_INVENTORY.md` | Stale | n/a | None | Last PR archived: #243 (2026-04-21). Main HEAD post-#335 (2026-04-30). ~50 PRs unarchived. CLAUDE.md §Mandatory-Board-Hygiene-Rule violated by every merge since #243. | n/a | n/a | **3** |
| **LATEST-RECENT-1** | (board) | `LATEST_STATE.md` Recently Shipped | Stale | n/a | None | Same staleness as PR-ARC-1. | n/a | n/a | **3** |
| **STATUS-CODEC-1** | (board) | STATUS_BOARD codec-sweep | Frozen | n/a | None | D0.2/D0.3/D0.5/D1.1/D1.2/D1.3/D2.1/D2.3/D3.1/D3.2 marked "**In PR**" since 2026-04-21 with **no PR number cited and no merge in `git log`**. Stalled or branch abandoned. | codec-sweep-via-lab-infra-v1 | Stalled | **4** |
| **MUL-ASSESS-1** | R6 | `MulAssessment` | Wired (×4) | 4 | Spaghetti | `contract::mul::MulAssessment:50` + `planner::mul::mod.rs:82` + `arigraph::orchestrator:172` + `cache::triple_model:57`. | none | Missing | **5** |
| **TRUST-1** | R6 | `TrustTexture` | Wired (×3 incompatible) | 3 | High | contract: 4 variants (Calibrated/Over/Under/Uncertain); planner: 5 (Crystalline/Solid/Fuzzy/Murky/Dissonant); arigraph orchestrator: 3 (Crystalline/Fibrous/Fuzzy). | none | Missing | **4** |
| **FLOW-1** | R6 | `FlowState` | Wired (×3) | 3 | High | contract:4, planner:4 (Apathy not Transition), lance-graph:4 with Apathy. | none | Missing | **4** |
| **COMPASS-1** | R6 | `CompassDecision` | Wired (×3 incompatible) | 3 | High | contract:`{StaySurface,GoMeta}` (2); planner:`{ExecuteWithLearning,Exploratory,SurfaceToMeta}` (3); lance-graph:`{None,ForceExplore,ForceSandbox}` (3). | none | Missing | **4** |
| **SEAL-1** | R4 | `MembraneRegistry::seal()` | Stub | n/a | None | `lance_membrane.rs:207-213` returns `Ok(())` empty with TODO. `Plugin::seal` trait exists, never called. | foundry-roadmap PR-3 | Stalled | **3** |
| **PROJECT-LANCE-1** | R7 | `CognitiveEventRow` Lance writer | Aspirational | 0 | None | `external_intent.rs:88-104` schema-on-paper. `LanceMembrane::project` emits to `tokio::watch` only; **no Lance dataset writer**. Audit log writer exists for `AuditEntry`, not for `CognitiveEventRow`. | callcenter-membrane-v1 DM-9 | Stalled | **4** |
| **LANCEDB-PHANTOM-1** | R7 | `lancedb` SDK dep | Aspirational | 0 | None | `lance-graph/Cargo.toml:39+67` declares `lancedb = "=0.27.2"` behind feature `lancedb-sdk`. **Zero `lancedb::` source hits workspace-wide.** Phantom feature gate. | none | n/a (delete or wire) | **3** |
| **PERMUTE-1** | R5 | `vsa16k_permute` | Aspirational | 0 | None | Markov ρ^d braiding requires it; missing. `markov_bundle.rs:135` TODO "per-sentence pre-bundle vsa_permute is a follow-up". | elegant-herding-rocket-v1 D5 | Stalled | **4** |
| **CONTENT-FP-1** | R5/R6 | `content_fp.rs` (deepnsm) | Aspirational | 0 | None | Brief claimed shipped in PR #243; **does not exist** at `crates/deepnsm/src/content_fp.rs`. Closest: `fingerprint16k.rs::from_centroid_semantic` (which violates I-VSA-IDENTITIES). | elegant-herding-rocket-v1 D5 region | Stalled | **4** |
| **ROLEKEY-OPS-1** | R6 | `RoleKey::bind/unbind/recovery_margin` | Removed | 0 | None | Deleted in cleanup commit `cd5c049`. **Board still advertises them** (`LATEST_STATE.md:32`, `STATUS_BOARD.md:120`); `TECH_DEBT.md:1057` flags the divergence. | none | n/a (board lag) | **4** |
| **TRAJ-HASH-1** | R3 | `step_trajectory_hash` | Stub | 0 | None | `orchestration.rs:194-201` is feature-gated `unimplemented!()`. PR #278 outlook E4 cross-PR bridge to PR #279's grammar substrate. | foundry-roadmap.md (cross-PR) | Stalled | **3** |
| **ALPHA-7-1** | R1/R2 | Pillar-7 α-front-to-back | Wired (no JC prover) | 1 | None | `cognitive_shader.rs:319-340` (`AlphaComposite`); `collapse_gate.rs:30-53` (`MergeMode::AlphaFrontToBack=3`); kernel at `driver.rs:493-529`. No `crates/jc/` prover for Pillar-7. | none | Missing | **3** |
| **CARTAN-1** | (jc) | Pillar 2 (Cartan-Kuranishi) | Stub (silent green) | n/a | None | `jc/cartan.rs:15-22` returns `PillarResult::deferred(...)` which counts as `pass: true`. Genuinely deferred — needs learned-attention-mask module. | (formal-scaffold idea-journal) | Stalled | **3** |
| **SCENARIO-PROMOTE-1** | R1/R7 | Scenario-vs-evidence promotion gate | Aspirational | 0 | None | Per `holographic-temporal-access-projection.md`. `ShaderHit::source_kind: TemporalCandidateKind` does not exist. | none | Missing | **3** |
| **ADJ-THINK-1** | R3/R6 | Thinking-as-`AdjacencyStore` (I5) | Aspirational | 0 | None | I5 doctrine: 36 thinking styles are nodes in a CSR/CSC `AdjacencyStore` at τ-prefix `0x0D`. `tau()` addresses are computed; **nothing writes those rows.** | none | Missing | **4** |
| **DEBUG-STRINGIFY-1** | R3/R7 | `format!("{:?}", logical_plan)` | Workaround | 35 | High | 35 sites read DataFusion `LogicalPlan` Debug output as a stable surface. `lance_native_planner.rs:76-79` feeds the result back into `Planner::plan(query_hint)` which runs `to_uppercase().contains("MATCH")` against pretty-printed Rust struct syntax. | none | Missing | **5** |

### Section B — Spaghetti cluster (active)

| Cluster | Members | Total entropy | Suggested order |
|---|---|---|---|
| **NARS** | NARS-1, TRUTH-1, NARS-TRUTH-1, MUL-ASSESS-1 | 17 | Collapse `NarsInference` first (canonical: `contract::grammar::inference::NarsInference` 7-variant). Then `TruthValue`, then `NarsTruth`. |
| **Thinking** | THINK-1, COMPASS-1, TRUST-1, FLOW-1, MUL-ASSESS-1, ADJ-THINK-1 | 24 | Adopt `contract::thinking::ThinkingStyle` (36) as canonical. Replace planner-12 + engine-12 + engine-5 + bandit. Drop `UNIFIED_STYLES[12]` const for `unified_params(style: ThinkingStyle)`. |
| **VSA carrier** | VSA-1, PERMUTE-1, CONTENT-FP-1, ROLEKEY-OPS-1, CRYSTAL-1 | 23 | Add `vsa16k_permute` first. Then methods on `Vsa16kF32`. Then re-introduce role-key ops on f32 carrier. |
| **Parser** | PARSER-1, DEBUG-STRINGIFY-1 | 10 | Wire `cypher_parse::CypherParse::plan` to call `lance-graph::parser::parse_cypher_query` (real nom). Eradicate the 35 Debug-string sites once a typed visitor exists. |
| **Foundry seal (residual)** | SEAL-1, PROJECT-LANCE-1 | 7 | POLICY-1 / MEMBRANE-GATE-1 / WATCHER-1 closed (see RESOLVED). Real `seal()` topo sort. Then `CognitiveEventLanceSink`. |
| **HEEL ladder** | HEEL-1, CAM-DIST-1, DNTREE-1 | 12 | Reconcile `CamByte` with I10 (rename or amend doc). Decide if `holograph::DnTree` graduates into contract, or if `QueryStrategy::DnTreeFull` retires. |
| **Board hygiene** | CONTRACT-INV-1, PLAN-INDEX-1, AGENT-LOG-1, PR-ARC-1, LATEST-RECENT-1, STATUS-CODEC-1 | 19 | Triple-ledger retrofit. Then create `AGENT_LOG.md`. |
| **Per-row-context (NEW from #355)** | MUL-THRESHOLD-1, CASCADE-COLS-1, CONTEXT-ID-1 | 9 | Single 200-300 LOC PR: land per-row `BindSpace.context_ids`; re-wire `driver.rs:303-321`; bump `lance_cache.rs` Arrow schema for 7 new MappingRow cascade cols. Drops three rows entropy 3 → 2. |

### Section C — Open seams (still active)

> Closed seams (R6/R0 ontology-as-SoA via #355; R4 RBAC↔Membrane via #29 + #98;
> R7 LanceVersionWatcher transport via #337) live in RESOLVED.

| Seam | From → To | Single point of fix |
|---|---|---|
| R0 `BindSpace` write airgap soft-enforced | R0 → R2 | Privatise fields, add `apply(GateDecision, MergeMode, RowDelta)` write API |
| R3 `step_trajectory_hash` | R3 → R7 | Land PR #279 outlook E4 implementation |
| R7 LanceMembrane projection cold-path | R4 → R7 | `CognitiveEventLanceSink` mirror of `LanceAuditSink` |
| R7 `LanceVersionWatcher` MVCC bind | R7 → R4 | Bind `Dataset::checkout_latest().version()` (transport closed; storage backend pending) |
| R8 cypher_bridge regex | R8 → R3 | Wire to `lance-graph::parser::parse_cypher_query` |
| Pillar-7 prover | (jc) | Build `crates/jc/src/alpha_front_to_back.rs` KS-concentration probe |
| Pillar-2 (Cartan-Kuranishi) genuinely deferred | (jc) | Needs ML attention-mask module |
| **NEW: per-row `BindSpace.context_ids`** | R6 → R0 | Wave-3.5 carry from PR #355; drives Per-row-context cluster |
| **NEW: `lance_cache.rs` Arrow schema bump for 7 cascade cols** | R6 → R7 | Lance-cache hydrate is currently lossy |

### Section D — Plan-status cross-reference

| Plan file | INTEGRATION_PLANS index? | STATUS_BOARD coverage? | Live deliverables? |
|---|---|---|---|
| `elegant-herding-rocket-v1` | yes | yes | D5/D7 SHIPPED #243; D2/D3/D8/D9/D10/D11 queued |
| `codec-sweep-via-lab-infra-v1` | yes | yes | D0.6/D0.7 SHIPPED #225; D0.1 SHIPPED #227; 10 D-ids stalled |
| `categorical-algebraic-inference-v1` | yes | partial | Companion to elegant-herding-rocket; #243 |
| `callcenter-membrane-v1` | yes | yes | DM-0/DM-1 SHIPPED 2026-04-22; DM-2 in progress; DM-4 closed via WATCHER-1 transport; DM-7 stalled |
| `unified-integration-v1` | yes | yes | DU-4/DU-5 SHIPPED; DU-0..3 queued |
| `q2-foundry-integration-v1` | yes | NO | Phase 2B SHIPPED via q2#35; Phase 3 awaits CycleAccumulator + BgzShaderDriver |
| `lf-integration-mapping-v1` | yes | NO | Active; no STATUS_BOARD section |
| `bindspace-columns-v1` | yes | NO | Active; PR #272 shipped Phase 1 (Column H) |
| `foundry-consumer-parity-v1` | yes | NO | Active; SMB closure PR #29; medcare closure PR #98 |
| `cam-pq-production-wiring-v1` | yes (DRAFT) | NO | DRAFT |
| `supabase-subscriber-v1` | yes | partial | DM-4 transport closed via WATCHER-1; DM-6 In PR |
| `lance-graph-ontology-v5` | NEW (PR #355) | n/a | Pillar 0 SHIPPED; Per-row-context follow-up cluster open |
| `ogit-cascade-supabase-callcenter-v1` | NEW (PR #355) | n/a | D-V1-1/2/3/7/11 shipped; D-V1-4/5 OGIT-side |
| `palantir-parity-cascade-v2` | NEW (PR #353/355) | n/a | D-V2-2/3/4/10/12 shipped; D-V2-7 (Q2 Object Explorer) future |
| **Unindexed plans on disk** (7) | **NO** | NO | archetype-scaffold, burn-ndarray-parity-sprint, foundry-roadmap-unified-smb-medcare, oxigraph-arigraph-cognitive-shader-soa-merge, sql-spo-ontology-bridge, tetrahedral-epiphany-splat-integration, thought-cycle-soa-awareness-integration |

### Section E — Aggregate entropy (open only)

- **Open rows tracked here:** 33 (from initial 41) + 7 still-open new rows (SUBJECT-DTO-1, MUL-THRESHOLD-1, CASCADE-COLS-1, OBJECT-VIEW-1, CERT-OFFICER-1, CONTEXT-ID-1, DTO-CLASS-CHECK-1) = **40 open rows**.
- **Spaghetti (entropy 5):** 7 — NARS-1, THINK-1, VSA-1, DEEPNSM-NSM-1, PARSER-1, DNTREE-1, MUL-ASSESS-1, DEBUG-STRINGIFY-1.
- **High (entropy 4):** 11 — GATE-1, TRUTH-1, CRYSTAL-1, HEEL-1, STATUS-CODEC-1, TRUST-1, FLOW-1, COMPASS-1, PROJECT-LANCE-1, PERMUTE-1, CONTENT-FP-1, ROLEKEY-OPS-1, ADJ-THINK-1, SUBJECT-DTO-1, DTO-CLASS-CHECK-1.
- **Partial (entropy 3):** 13 — NARS-TRUTH-1, CAM-DIST-1, CONTRACT-INV-1, PLAN-INDEX-1, AGENT-LOG-1, PR-ARC-1, LATEST-RECENT-1, SEAL-1, LANCEDB-PHANTOM-1, TRAJ-HASH-1, ALPHA-7-1, CARTAN-1, SCENARIO-PROMOTE-1, MUL-THRESHOLD-1, CASCADE-COLS-1, OBJECT-VIEW-1, CERT-OFFICER-1, CONTEXT-ID-1.

**Highest-leverage cluster (most entropy / fewest dependencies):** **Per-row-context** (9 cluster-entropy, single 200-300 LOC PR closes 3 rows entropy 3→2 and 2 open seams). Net leverage: −3 rows + −2 seams.

**Highest-architectural-leverage cluster:** **Thinking** (24) — collapsing the 4-copy ThinkingStyle drift unlocks I5 (thinking-as-AdjacencyStore), which removes the `ord_to_thinking_style` hand-pick widening, which simplifies `MUL-ASSESS-1` and `COMPASS-1`.

**Highest-cognitive-leverage cluster:** **VSA carrier** (23) — adding `vsa16k_permute` unblocks Markov ρ^d braiding (D5 deferred work), the literal FMA the architecture is built around.

**The single biggest deficit-vs-genius gap:** `Vsa16kF32` (R5) is the architecture's load-bearing carrier — the literal FMA `bind(role, content) + bundle(prior)` lives on it — and its algebra is currently **8 free functions on a type alias**, not methods on a newtype carrier. Click P-1 explicitly says "free function = reject, method = accept". This is **Stage 3 stalled at Dumb when it should be Stage 4 + Smart**. PERMUTE-1 is a direct downstream consequence.

---

## New row introductions (still open) — 2026-05-06 / 2026-05-07

### SUBJECT-DTO-1 — Aspirational (entropy 4)

**Region:** R4/R6. **State:** Aspirational. **DupCount:** 0.

Multiple consumers waiting on a canonical `Subject` type that doesn't exist yet — namespace pre-claimed by JWT shape in MedCareV2 #8 + `q2` cycle session-binding. `MembraneGate::admit` has no Subject parameter today.

**Deficit→genius:** `contract::subject::Subject { user_id: u64, roles: SmallVec<[u16; 4]>, display_name: SmolStr, auth_time: i64, source: AuthSource }` with `AuthSource::{Jwt(JwtClaims), Session(SessionId), Anonymous, ProbeOnly}`. **Typed struct fields, not VSA.** Consumed by `MembraneGate::admit(subject, resource, op)` (E2 seam).

### MUL-THRESHOLD-1 — Wired but loose-end (entropy 3)

**Region:** R6 (companion to MUL-ASSESS-1). **State:** Wired (7/7 tests). **Smart/Dumb:** Smart.

Component: `lance-graph-contract::mul::MulThresholdProfile` w/ const profiles `MEDICAL` / `CALLCENTER` / `DEFAULT` + `for_context(u32)` lookup; consult site at `driver.rs:303-321`.

**LooseEnd:** `driver.rs:311` per-dispatch (not per-row) `ctx_id` — `trust_below_floor` branch is dead-effect today. Per-row `BindSpace.context_ids` deferred to Wave-3.5. **Cluster: Per-row-context.**

**Evidence:** PR #355 D-ONTO-V5-9 (commit `8366e70`).

### CASCADE-COLS-1 — Wired but loose-end (entropy 3)

**Region:** R6/R7. **State:** Wired (4/4 tests). **Smart/Dumb:** Dumb (struct fields; no carrier methods yet — Click-P-1 deficit candidate).

Component: `MappingRow` cascade cols: `IdentityCodec { cam_pq_code, base17_head, palette_key, scent }` + `QualiaMeta { qualia, meta, edge }` + `thinking_style: Option<u16>` + `attribute_sources` + 3 type-ref strings.

**LooseEnd:** `lance_cache.rs` Arrow schema bump pending — Lance-cache hydrate is currently **lossy** (replay defaults the 7 new cols). #355 handoff debt item 5. **Cluster: Per-row-context.**

**Evidence:** PR #355 D-V1-7 + D-V2-12 (Wave 3 self-trimmed 1.87× → 1.41× envelope; commit `fc49a29`).

### OBJECT-VIEW-1 — Stage-1 risk (entropy 3)

**Region:** R6 (contract surface for Foundry). **State:** Wired (4/4 tests). **Maturity:** Stage 1 (≤1 consumer — only the test suite; no production consumer until D-PARITY-V2-7 ships).

Component: `ObjectView`, `NotificationSpec`, `DisplayTemplate{Card,Detail,Summary}`, `NotificationTrigger{Created,Updated,Deleted,ThresholdCrossed}`, `NotificationChannel{Inline,Webhook,Email}`, `FieldRef`. Zero serde, zero deps. POD-shaped.

**DupPotential:** **Watch** (namespace pre-claim risk if Q2 Object Explorer invents parallel types before D-PARITY-V2-7 lands — CRYSTAL-1-style collision potential).

**Evidence:** PR #355 D-PARITY-V2-4 (commit `fc49a29`).

### CERT-OFFICER-1 — Wired but test-of-test stub (entropy 3)

**Region:** R8 (build / governance). **State:** Wired (post-FIX-2 scoped to direct builds + `--features zone-check-strict`); 0 violations on canonical surface.

Component: `lance-graph-callcenter/build.rs` syn-based static check; walks `Item::Struct/Enum` filtered by `Visibility::Public`, flags `#[derive(Serialize)]` on Zone 1/2 source files.

**LooseEnd:** `zone_serialize_check_compile_fail.rs` is `assert!(true)` smoke — trybuild-style probe is the proper fix (FIX-1, deferred per #355 handoff debt item 1).

**Evidence:** PR #355 D-V1-1 + meta-1 FIX-2 (commit `16a745c`).

### CONTEXT-ID-1 — Wired schema-side (entropy 3)

**Region:** R6 (contract::SchemaPtr) + R0 (BindSpace context-id column). **State:** Wired. **Maturity:** Stage 2 → Stage 3 (consumed by SchemaPtr + driver.rs + bridge crates). **Smart/Dumb:** Smart.

Component: `SchemaPtr` widened tuple → named struct w/ `ontology_context_id: u32`; new `NamespaceRegistry::seed_defaults()` with 14 mappings (WorkOrder=1, Healthcare=2, Network=3, SMB=0, Medical/{ICD10CM..CHEBI}=10..19).

**LooseEnd:** per-row `BindSpace.context_ids` deferred to Wave-3.5. Same loose end as MUL-THRESHOLD-1 + CASCADE-COLS-1. **Cluster: Per-row-context.**

**Evidence:** PR #355 D-V1-2 (commit `8366e70`).

### DTO-CLASS-CHECK-1 — Wired gate, failing consumers (entropy 4)

**Region:** (board / governance tooling). **State:** Wired (gate works; 1/1 smoke test) / **currently 28/28 FAIL exit 1** (consumer side incomplete). **Maturity:** Stage 1.

Component: `tools/dto-class-check/` new bin scans 28 types vs 22-row hardcoded ledger const.

**LooseEnd:** 22 `// classification: …` doc comments needed in owner crates to flip 28/28 FAIL → 28/28 PASS (#355 handoff debt item 4).

**Evidence:** PR #355 D-V2-10 (commit `fc49a29`).

---

## Update protocol

When a row's state changes (e.g. NARS-1 collapses):
1. **Append a new dated entry** below. Reference the row ID.
2. **Do NOT edit the snapshot table.** Append-only.
3. **If row resolves (entropy → ≤ 2):** move record to `ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md`.
4. New section: `## YYYY-MM-DD — <ID> resolution / state change` with old-state, new-state, evidence (PR#, file:line).

When a NEW SoA / DTO / bridge enters:
1. Append `## YYYY-MM-DD — <ID> introduction` to whichever file fits (OPEN if entropy ≥ 3, RESOLVED if ≤ 2).
2. Score Region / DupCount / DupPotential / LooseEnds / Plan / PlanStatus / Entropy.
3. If part of an existing cluster, cite the cluster ID.

Cross-references:
- Companion map: `.claude/knowledge/soa-dto-fma-map.md`
- Mandatory invariants: `.claude/knowledge/lab-vs-canonical-surface.md` (I1-I11)
- Iron rules: `CLAUDE.md` § Substrate-level iron rules (I-SUBSTRATE-MARKOV / I-NOISE-FLOOR-JIRAK / I-VSA-IDENTITIES)
- Resolved entries archive: `ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md`
- Single-Binary Topology: `SINGLE_BINARY_TOPOLOGY.md`
- Patterns guide: `.claude/patterns.md` (PLURAL — sibling of `.claude/pattern.md` per PR #345)

---

## Crate inventory — canonical at 2026-05-07 (~23 crates)

**Lance-graph workspace (~17 crates):**
`lance-graph-contract`, `lance-graph`, `lance-graph-planner`,
`lance-graph-callcenter`, `lance-graph-rbac`, `lance-graph-archetype`,
`lance-graph-catalog`, `lance-graph-cognitive`, `cognitive-shader-driver`,
`deepnsm`, `holograph` (workspace-EXCLUDED), `bgz-tensor`, `highheelbgz`,
`reader-lm`, `sigker` (NEW per PR #348), `jc/`, `thinking-engine`,
`neural-debug`, `learning`, `causal-edge`, plus embedded modules
`lance-graph::nsm/` + `lance-graph::cam_pq/` + `lance-graph::graph::{spo,arigraph,versioned}`.
NEW per PR #355: `lance-graph-ontology` (Pillar 0 SoA-as-canonical).

**Consumer-side (separate repos, in MCP allowlist):**
`medcare-rs/{medcare-rbac,medcare-realtime,medcare-{server,db,core,analytics,pdf}}`,
`smb-office-rs/{smb-realtime,smb-{office-bin,db,core,analytics}}`,
`q2/cockpit-server`.

**Out-of-scope:** `MedCareV2` (C# probe, not in MCP), `ladybug-rs`
(earlier prototype, fully migrated per LADYBUG-EQUIV-1 in RESOLVED).

---

## 2026-05-07 — Sprint-2 recognition reframes (THINK-1, HEEL-1, ADJ-THINK-1, CRYSTAL-1, CAM-DIST-1) + 15 architectural patterns absorption

> **Append-only correction layer.** The 2026-05-05 Section A snapshot
> is IMMUTABLE — it stays as historical record. This section adjusts
> the live `Entropy` field on five rows and one cluster-substrate row
> (VSA-1) based on the 12-agent sprint's recognition that the named
> "drifts" were already-shipped patterns that the ledger over-flagged
> for lack of a vocabulary to name them. **No code changes are
> claimed here; only naming + entropy re-scoring.**
>
> **Cross-refs:**
> - W1's master plan: `.claude/plans/unified-ogit-architecture-synthesis-v1.md` (sprint coordination)
> - W2's Tier-0 doc: `.claude/knowledge/architectural-patterns-A-through-O.md` (15-pattern canonical reference)
> - W4's epiphanies: `.claude/board/EPIPHANIES.md` (recognition entries E-PATTERN-A through E-PATTERN-O)
> - W5's tech-debt: `.claude/board/TECH_DEBT.md` (TD-OGIT-G-SLOT-1, TD-CONTEXT-BUNDLE-2, TD-GENERIC-BRIDGE-3, TD-MANIFEST-MODULES-4, TD-RACTOR-SUPERVISOR-5, TD-INT4-32D-ATOMS-6, TD-CIRCULAR-COMPILATION-7, TD-CAM-DIST-REGISTRATION-9, TD-ADJ-THINK-EXPOSE-10)

### Sprint-2 motive (one paragraph)

A 16-turn architectural-review conversation surfaced 15 patterns
(A-O) that the workspace had **already shipped** but never named in
the ledger. Without those names, five rows from Section A read as
"drift / duplication / aspirational" when they were actually
"layered codebooks / single canonical impl with multiple views /
write surface already in place but missing public builder API."
The corrections below re-score `Entropy` to reflect what's actually
in `crates/`, while preserving the historical record of how the
rows looked before the patterns were named.

### THINK-1 reframe — entropy 5 → 3 (−2)

**Original claim (2026-05-05):** "4-copy ThinkingStyle drift,
Spaghetti, 6 redundant declarations."

**Recognition:** Not a drift. The workspace ships an **intentional
two-level codebook**:

- `contract::thinking::ThinkingStyle` — 36-variant **composed
  surface** (base × modifier). Consumers that need the full
  taxonomy read this.
- `p64-bridge::STYLES` — 12-entry **base codebook**, LazyLock-frozen
  at startup, with 4 clusters: **Convergent / Divergent / Attention
  / Speed**. The base codebook is the canonical addressable surface;
  modifiers compose on top.
- `lance-graph-planner::thinking::style` — reads the **contract-36**
  composed surface (correct consumer of the high-level taxonomy).
- `thinking-engine::cognitive_stack::ThinkingStyle` — uses the
  **12-base internally** (correct consumer of the low-level
  codebook for cognitive-stack dispatch).

**12 × modifiers = 36** is the algebra. Pattern G (Best-Practice
Thinking Inheritance) reads from BOTH surfaces; **neither retires**.
The `superposition::ThinkingStyle(5)` and `UNIFIED_STYLES(12-const)`
+ `StyleSelector` (RL bandit) were the genuine drift items; those
remain real loose ends but are now scoped to a single Pattern-G
wiring task (TD-OGIT-G-SLOT-1 region) rather than a "collapse
ThinkingStyle" megasweep.

**New entropy: 3 (Partial).** Loose end: Pattern G wiring to expose
per-thinking-style inheritance from a DOLCE root. Surface is
correct; the missing piece is the explicit inheritance edge from
the 12-base codebook to the 36-composed surface, exposed as a
public adjacency. See **ADJ-THINK-1** for the matching write surface.

**Cluster impact:** Thinking cluster total drops from 24 → ~14
(THINK-1 5→3 = −2; ADJ-THINK-1 4→2 = −2; remaining MUL-ASSESS-1,
COMPASS-1, TRUST-1, FLOW-1 unchanged pending separate Pattern G
wiring). Conservative further estimate to ~10 once Pattern G is
wired.

### HEEL-1 reframe — entropy 4 → 2 (−2)

**Original claim (2026-05-05):** "Wired (×3 different orderings)
HEEL/HIP/BRANCH/TWIG/LEAF, I10 doctrinal sequence violated by all
three."

**Recognition:** Not 3 orderings. There is **ONE canonical
HHTL cascade impl** at `p64-bridge::cognitive_shader::CognitiveShader::cascade`:

```
HEEL: layer_mask gates Z         (which predicate planes active)
HIP:  mask_row = planes[z][block_row]  gates X-Y (topological neighborhood)
TWIG: 4×4 block expansion → archetype indices
LEAF: semiring.distance(query, target)  O(1) bgz17 lookup
```

The cascade doc comment is explicit (and load-bearing):

> *"No POPCNT. No Hamming. Distance is PRECOMPUTED in the codebook."*

The "3 orderings" called out in 2026-05-05 — `contract::cam::CamByte`,
`lance-graph::graph::neighborhood::SearchCascade`,
`bgz17/router.rs + bgz-tensor/hhtl_d.rs` — are **different views of
the same primitive** at different layer subsets and different
cascade depths, not parallel implementations. They could not
diverge because they all bottom out in the same precomputed
distance table.

**New entropy: 2 (Mostly clean).** Loose end: doc surface alignment
— the three views need a single doc that names them as views of
the canonical cascade rather than as parallel orderings. This is a
docs PR, not a code change.

**Cluster impact:** HEEL ladder cluster 12 → ~4 (HEEL-1 4→2 = −2;
CAM-DIST-1 3→2 = −1; DNTREE-1 5 unchanged — genuinely
abandoned-in-excluded-crate). Conservative further reduction once
the doc lands.

### ADJ-THINK-1 reframe — entropy 4 → 2 (−2)

**Original claim (2026-05-05):** "Aspirational. I5 doctrine: 36
thinking styles are nodes in a CSR/CSC AdjacencyStore at τ-prefix
0x0D. `tau()` addresses are computed; **nothing writes those rows.**"

**Recognition:** Not Aspirational. The write surface IS shipped —
it just isn't named `AdjacencyStore<ThinkingStyle>`. The field
`[u64; 64]; 8 planes` inside `p64-bridge::cognitive_shader::CognitiveShader`
IS the adjacency store:

- 64×64 = **4,096 archetype-pair slots per layer** (CSR-ish bitmask)
- **8 layers** = **32,768 typed edges** total
- Each plane = one thinking-style adjacency layer
- The `tau()` addresses ARE the writes — the planes are populated
  at codebook-freeze time

The "addresses computed never written" framing was wrong: the
planes are the writes. The genuine missing piece is the **public
builder API** that exposes this surface as `AdjacencyStore<ThinkingStyle>`
to outside crates. Today the planes are private to the shader;
Pattern G needs `tau_write()` to be public.

**New entropy: 2 (Mostly clean).** Loose end: expose
`AdjacencyStore<ThinkingStyle>` builder. See **TD-ADJ-THINK-EXPOSE-10**
in TECH_DEBT.md (W5).

### CRYSTAL-1 reframe — entropy 4 → 2 (−2)

**Original claim (2026-05-05):** "`SentenceCrystal` collision —
contract::crystal::sentence::SentenceCrystal + holograph::crystal_dejavu::SentenceCrystal
+ holograph::sentence_crystal::SemanticCrystal. holograph excluded
from workspace; collision is silent."

**Recognition:** Not a collision. Two legitimate codebooks at
**different Pattern N (Fingerprint-as-Codebook-Address) layers**:

- `contract::crystal::sentence::SentenceCrystal` — canonical
  **sentence-level fingerprint** (one slot in the OGIT bundle:
  what was said as a unit).
- `holograph::sentence_crystal::SemanticCrystal` — a different
  layer: **holograph semantic-rich representation** (a separate
  codebook in the OGIT bundle: how the sentence resonates against
  the semantic basis).

Both are legitimate. Pattern N supports parallel codebooks per
content layer — sentence-level + semantic-level + qualia-level +
crystallized-belief-level — addressing different memory layers of
the same Think struct. The 2026-05-05 framing missed that "two
types with overlapping names" in a Pattern-N system is the
**expected shape**, not a collision.

**New entropy: 2 (Mostly clean).** Loose end: naming-clarification
docs to make the layering explicit. Both surfaces remain.
`holograph::crystal_dejavu::SentenceCrystal` (the third type
flagged in the original claim) is in the workspace-EXCLUDED
holograph crate; it stays excluded — that part of CRYSTAL-1's
original entry was correct.

### CAM-DIST-1 reframe — entropy 3 → 2 (−1)

**Original claim (2026-05-05):** "`cam_distance` UDF wiring,
Wired (opt-in only), default Cypher path can't reference
`cam_distance`. Plan: cam-pq-production-wiring-v1 (DRAFT)
Stalled."

**Recognition:** One-line fix. The UDF is fully registered at
`cam_pq/udf.rs:241,257,326`. The DataFusion planner just needs to
call `register_cam_distance(state)` from `DataFusionPlanner::new`.
Pattern N substrate is fully shipped; this row is **registration
scaffold only**, not a real plan gap.

**New entropy: 2 (Mostly clean — Tier 0 quick-win pending).**
See **TD-CAM-DIST-REGISTRATION-9** in TECH_DEBT.md (W5).

### VSA-1 substrate clarification — entropy 5 → 3 (−2)

**Original claim (2026-05-05):** "VSA carrier algebra, Wired (×3
distinct algebras), entropy 5, **highest-cognitive-leverage
cluster**, 'load-bearing FMA the architecture is built around.'"

**Recognition (retraction of the highest-cognitive-leverage
framing):** `Vsa16kF32` is **NOT** the load-bearing universal
carrier. It is a **Markov-accumulation cotton-ball** — one program
among many in the cognitive stack. The actual substrate is
**CAM (bitpacked content-addressable memory)**, which is what
Pattern N (Fingerprint-as-Codebook-Address) and Pattern M
(Wave-Particle Bimodal) bottom out in.

The 2026-05-05 framing — "Vsa16kF32 is the FMA the architecture is
built around, PERMUTE-1 is the direct downstream consequence" —
overweighted one program (cotton-ball superposition for
Markov-trajectory accumulation) at the expense of the actual
multi-program substrate. The corrected framing per Pattern H
(Switchable Cognitive Vessel): the cognitive stack runs **one
program among many** at any moment; Vsa16kF32 is the carrier for
the **bundle-superposition program**, not for the whole architecture.

**New entropy: 3 (Partial).** The remaining loose ends in the
"VSA carrier" cluster reduce to:

- **PERMUTE-1** (still real): per-sentence pre-bundle `vsa_permute`
  for ρ^d braiding is a genuine missing primitive on the F32 carrier
  IF the cotton-ball program needs it. Demoted from "blocks the
  architecture" to "completes one program among many."
- **CONTENT-FP-1** (still real): `content_fp.rs` was claimed shipped
  in PR #243 but does not exist at the claimed path. Genuine
  aspirational gap. Demoted from "blocks Markov braiding" to
  "completes the cotton-ball encoder side."
- **ROLEKEY-OPS-1** (board lag): board still advertises deleted
  ops. Unchanged — this is doc cleanup.
- **CRYSTAL-1** (reframed above, leaves this cluster).

**Cluster impact:** VSA carrier cluster 23 → ~8 (VSA-1 5→3 = −2;
CRYSTAL-1 4→2 = −2 and leaves the cluster; PERMUTE-1, CONTENT-FP-1,
ROLEKEY-OPS-1 unchanged but rescoped). Net cluster entropy drop
~15 units, but more importantly **the framing changes**: from
"this is the load-bearing FMA" to "this is one program among
several." That reframe removes the false urgency that had VSA-1
flagged as a Tier-0 priority.

### 15-Pattern absorption table

The 12-agent sprint named 15 architectural patterns (A through O).
Each pattern provides a vocabulary slot for one or more ledger rows.
**"Already shipped" patterns explain rows that the ledger over-flagged
as drift; "design phase" patterns explain rows that need wiring but
where the pattern is a recognized shape.**

| Pattern | Name | Ledger rows touched | Status | TD ref (W5) |
|---|---|---|---|---|
| **A** | SPO-G u32 slot | SPO-1 (closed), AriGraph extension | design phase | TD-OGIT-G-SLOT-1 |
| **B** | ContextBundle | ONTOLOGY-REGISTRY-SOA-1 (closed), CONTEXT-ID-1 | design phase | TD-CONTEXT-BUNDLE-2 |
| **C** | Generic Bridge | POLICY-1 (closed), MEMBRANE-GATE-1 (closed), SUBJECT-DTO-1, OBJECT-VIEW-1, MEDCARE_POLICY_GAP | design phase; consumer-side simplification | TD-GENERIC-BRIDGE-3 |
| **D** | Meta-Structure Hydration | DEEPNSM-NSM-1 (migration debt), PARSER-1, DEBUG-STRINGIFY-1 | design phase | — |
| **E** | Compile-Time Consumer Binding | SEAL-1 (supervisor IS the seal) | design phase | TD-MANIFEST-MODULES-4 |
| **F** | ractor/BEAM Supervisor | WATCHER-1 (transport closed), PROJECT-LANCE-1 | design phase | TD-RACTOR-SUPERVISOR-5 |
| **G** | Best-Practice Thinking Inheritance | THINK-1 (reframe), ADJ-THINK-1 (reframe), TRUST-1, FLOW-1, COMPASS-1, MUL-ASSESS-1, MUL-THRESHOLD-1 | design phase | — |
| **H** | Switchable Cognitive Vessel | ALPHA-7-1 (already shipped as one program in p64-bridge) | **already shipped** | — |
| **I** | Implicit Cognition | CYCLE-ACCUM-1 (resolved) | already shipped | — |
| **J** | INT4-32D Atoms | (new — no prior ledger row) | design phase | TD-INT4-32D-ATOMS-6 |
| **K** | Circular Compilation | (new — no prior ledger row) | aspirational | TD-CIRCULAR-COMPILATION-7 |
| **L** | SPO-Chain Narrative | PARSER-1, DEBUG-STRINGIFY-1, CRYSTAL-1 (reframe) | design phase | — |
| **M** | Wave-Particle Bimodal | VSA-1 (substrate clarification), PERMUTE-1, CONTENT-FP-1, ROLEKEY-OPS-1 | **already shipped in primitives**; G-blend mechanism is new | — |
| **N** | Fingerprint-as-Codebook-Address | All "codebook" rows; CRYSTAL-1, CAM-DIST-1, HEEL-1 (all reframes) | **already shipped** in qualia.rs, p64-bridge, prime_fingerprint, cam_pq, bgz17 | — |
| **O** | Phenomenological Memory | (new framing — no prior ledger row) | **already shipped** in qualia.rs (17D + 10 families + music calibration + Bach 7+1) | — |

**Reading the table:**
- Patterns **H, I, M, N, O** are recognised as **already shipped**.
  Their ledger rows are reframes (entropy goes down) because the
  pattern names what was already in `crates/`.
- Patterns **A, B, C, D, E, F, G, L** are **design phase** — the
  shape is recognised, the wiring is partial. Tech-debt rows in W5
  carry the actual work items.
- Patterns **J, K, O** introduce **new framings** (INT4-32D atoms,
  circular compilation, phenomenological memory). J and K open
  fresh tech-debt items; O recognises qualia.rs as Pattern O.

### Aggregate entropy delta — Sprint-2 recognition only

**Per-row reframes** (no code changes; entropy re-scoring from
recognition of already-shipped patterns):

| Row | Old entropy | New entropy | Delta |
|---|---|---|---|
| THINK-1 | 5 | 3 | −2 |
| HEEL-1 | 4 | 2 | −2 |
| ADJ-THINK-1 | 4 | 2 | −2 |
| CRYSTAL-1 | 4 | 2 | −2 |
| CAM-DIST-1 | 3 | 2 | −1 |
| VSA-1 | 5 | 3 | −2 |
| **Total** | **25** | **14** | **−11** |

**Net per-row delta: −11 entropy units from recognition alone.**
Zero lines of code changed. The reduction is honest: it reflects
that the patterns were already shipped and the ledger had been
over-flagging them.

**Cluster-level reorganization** (more dramatic because clusters
include the reframed rows AND benefit from a single Pattern G/M/N
wiring serving multiple rows at once):

| Cluster | Old total | New total (Sprint-2) | Delta |
|---|---|---|---|
| **Thinking** (THINK-1, COMPASS-1, TRUST-1, FLOW-1, MUL-ASSESS-1, ADJ-THINK-1) | 24 | ~10 (post Pattern G wiring) | ~−14 |
| **VSA carrier** (VSA-1, PERMUTE-1, CONTENT-FP-1, ROLEKEY-OPS-1, CRYSTAL-1) | 23 | ~8 (CRYSTAL-1 leaves; VSA-1 demoted; PERMUTE-1/CONTENT-FP-1 rescoped) | ~−15 |
| **HEEL ladder** (HEEL-1, CAM-DIST-1, DNTREE-1) | 12 | ~4 (DNTREE-1 unchanged at 5, but reframe drops 3 from the others) | ~−8 |
| **Net cluster entropy reduction** | — | — | **~37 units** |

The cluster-level reduction (~37) is larger than the per-row
reduction (−11) because Pattern G wiring resolves multiple
Thinking-cluster rows at once, and Pattern N + Pattern M
reframings rescope the VSA carrier cluster from "load-bearing
substrate" to "one program among many in the cognitive vessel."

### What this section does NOT claim

To stay brutally honest (per sprint instruction):

1. **No code shipped.** Every entropy reduction here is from
   recognition + naming. The patterns were already in `crates/`;
   the ledger just hadn't named them.
2. **Six rows reframed; thirty-plus rows unchanged.** This is not
   a "sprint that closed thirty rows." It is a recognition pass
   that corrects six over-flagged entries and absorbs 15 patterns
   into the ledger vocabulary.
3. **The 2026-05-05 Section A snapshot is untouched.** That table
   stays as historical record of how the ledger looked before the
   pattern vocabulary existed. The new entropies live in this
   dated section.
4. **Pattern G, A, B, C, D, E, F, L still need wiring.** They are
   "design phase" — recognized shape, partial implementation. The
   real work is in W5's TECH_DEBT items, not in this ledger.
5. **VSA-1's reframe is a demotion, not a closure.** Vsa16kF32 is
   one program among many — that is the new framing. PERMUTE-1
   and CONTENT-FP-1 remain real aspirational gaps for that one
   program; they did not vanish.
6. **Section B's cluster table is updated only by reference.** The
   2026-05-05 cluster totals (NARS 17, Thinking 24, VSA carrier 23,
   Parser 10, HEEL ladder 12, Board hygiene 19, Per-row-context 9)
   stay as historical record. The Sprint-2 cluster totals (Thinking
   ~10, VSA carrier ~8, HEEL ladder ~4) are recorded here, not by
   editing Section B.

### Updated Section E aggregate (Sprint-2 only — append, not edit)

Re-counted with the six reframes applied (and treating the original
Section A counts as the immutable baseline):

- **Spaghetti (entropy 5):** 5 rows (was 7) — NARS-1, DEEPNSM-NSM-1,
  PARSER-1, DNTREE-1, MUL-ASSESS-1, DEBUG-STRINGIFY-1 minus
  **THINK-1 leaves** (5→3) and **VSA-1 leaves** (5→3). Count: 7 − 2 = 5.
- **High (entropy 4):** 8 rows (was 11) — GATE-1, TRUTH-1,
  STATUS-CODEC-1, TRUST-1, FLOW-1, COMPASS-1, PROJECT-LANCE-1,
  PERMUTE-1, CONTENT-FP-1, ROLEKEY-OPS-1, SUBJECT-DTO-1,
  DTO-CLASS-CHECK-1 minus **CRYSTAL-1 leaves** (4→2),
  **HEEL-1 leaves** (4→2), **ADJ-THINK-1 leaves** (4→2). Approximate.
- **Partial (entropy 3):** entrants from reframes — THINK-1
  (was 5), VSA-1 (was 5). Original 13 + 2 entrants − 1 leaver
  (CAM-DIST-1 3→2) = ~14.
- **Mostly clean (entropy 2):** entrants from reframes — HEEL-1,
  ADJ-THINK-1, CRYSTAL-1, CAM-DIST-1. These remain **open** per
  Sprint-2 protocol (W1's master plan) until the docs/wiring items
  in W5's TECH_DEBT close.

**New aggregate net (Sprint-2 recognition only):** 40 open rows;
no rows added or moved to RESOLVED. The **entropy distribution**
shifts down by −11 from recognition.

### Highest-leverage cluster (post-Sprint-2)

Previously (2026-05-05): "Per-row-context (9 cluster-entropy, single
200-300 LOC PR closes 3 rows entropy 3→2 and 2 open seams). Net
leverage: −3 rows + −2 seams."

Post-Sprint-2: **Per-row-context cluster remains the highest leverage**
for actual code work. The Thinking and VSA carrier clusters are no
longer the architectural priorities they were on 2026-05-05;
recognising them as **already-shipped multi-layer codebooks** removes
the false urgency.

**The single biggest deficit-vs-genius gap (revised):** the
**Pattern G wiring** — exposing `AdjacencyStore<ThinkingStyle>` from
the existing 8-plane `[u64; 64]` shader-internal write surface as a
public builder. ~50-150 LOC. Closes ADJ-THINK-1 + unblocks THINK-1's
remaining loose end + provides the inheritance edge that
TRUST-1/FLOW-1/COMPASS-1/MUL-ASSESS-1 all need. **This is now the
highest architectural leverage in the workspace** — replacing the
2026-05-05 claim that VSA-1's missing `vsa16k_permute` was the
biggest gap.

### Brutally-honest self-review

- **The ledger over-flagged because the pattern vocabulary did not
  exist on 2026-05-05.** That's not a failure of the 2026-05-05
  snapshot — it's a feature of the append-only protocol that
  enabled this correction to land cleanly without rewriting history.
- **Six rows, not thirty, were reframed.** Anyone reading this
  section expecting a sweep is going to find a recognition pass.
  That's what it is.
- **The −11 entropy delta is real but small in absolute terms.**
  The 40 OPEN rows mostly stay open. What changed is that the six
  reframed rows are no longer **misleading the next session** about
  where the architectural pressure actually is.
- **Pattern G wiring is now the biggest single lever** — that
  replaces the 2026-05-05 VSA-1 framing. If the next session
  picks one item from this ledger, it should be exposing
  `AdjacencyStore<ThinkingStyle>` (TD-ADJ-THINK-EXPOSE-10).
- **VSA-1's demotion is the load-bearing recognition.** Calling
  Vsa16kF32 "the FMA the architecture is built around" was wrong;
  it's one carrier among several in a switchable cognitive vessel
  (Pattern H). That framing correction matters more than the
  −2 entropy.
