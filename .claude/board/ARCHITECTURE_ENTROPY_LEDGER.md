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
