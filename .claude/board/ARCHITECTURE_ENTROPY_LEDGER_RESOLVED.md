# Architecture Entropy Ledger — RESOLVED (closures + state-change archive)

> **APPEND-ONLY** — same governance as `ARCHITECTURE_ENTROPY_LEDGER.md`
> (the OPEN counterpart). State-change records and rows that have
> reached entropy ≤ 2 archive HERE; active concerns stay in OPEN.
>
> **Split rationale (2026-05-07):** the combined ledger reached 103 KB
> with the merger of PR #345/#346/#347/#348/#353/#355 dated sections.
> Splitting OPEN ↔ RESOLVED keeps the OPEN file scannable for next
> sessions (highest-leverage fix sortable by entropy DESC) while
> preserving APPEND-ONLY archaeology in RESOLVED.
>
> **READ ORDER:** OPEN first (active concerns); RESOLVED for context
> on what's already been closed (avoid re-litigating shipped work).
>
> Companion: `ARCHITECTURE_ENTROPY_LEDGER.md` (OPEN), `.claude/knowledge/soa-dto-fma-map.md`,
> `.claude/board/SINGLE_BINARY_TOPOLOGY.md`.

---

## Resolved rows from 2026-05-05 initial snapshot (Section A — entropy ≤ 2)

| ID | Region | Component | State | Entropy | Notes |
|---|---|---|---|---|---|
| **FREEWILL-1** | R6 | `free_will_modifier` width | Wired (mixed widths) | 2 | `f64` in contract/planner; `f32` in arigraph orchestrator and `contract/sensorium.rs`. Cosmetic mixed-width drift; pick f32 OR f64 globally; document why. |
| **LANCEDB-NAME-1** | R5 | Naming leak | Wired (cosmetic drift) | 2 | `holograph/Cargo.toml:19` declares feature `lancedb = ["dep:lance"]` — conflates LanceDB-SDK with Lance-format. `contract::crystal::fingerprint` doc-comments call `Vsa16kF32` "lancedb-native VSA". `learning::cam_ops:1814,1840` defines `register_lancedb_ops()` for **Lance** ops. Cosmetic. |
| **ENT-TYPE-1** | R6/R0 | `EntityTypeId` ↔ `BindSpace.entity_type[row]` | **Wired (Stage 3)** | 2 | `ontology.rs:81` defines `EntityTypeId = u16` with explicit "Foundry Object Type / Column H" doc. `bindspace.rs:177` exposes `entity_type: Box<[u16]>`. `BindSpaceBuilder::push_typed` writes it. **The bridge IS shipped** (PR #272 Phase 1 of `bindspace-columns-v1`). Polish: lift to newtype consumed directly by `BindSpace.entity_type[row]`. |
| **SPLAT-1** | R0/R1 | Gaussian-splat → CAM-plane splat | Wired (×1, stage 1) | 2 | `contract::splat::{SplatChannel, CamPlaneSplat, SplatPlaneSet, AwarenessPlane16K, CamSplatCertificate, SplatDecision, TriadicProjection, ReasoningWitness64}` shipped 2026-05-06 PR #336 + #344 (witness_to_splat). EWA OSINT example at `crates/jc/examples/osint_edge_traversal.rs`. PRs 4-7 of doc-sequence still queued. |

---

## Resolved cross-cutting findings

### LADYBUG-EQUIV-1 — ladybug-rs ↔ lance-graph equivalence (entropy 1 — closed)

**Region:** workspace meta-finding. **DupCount:** 0 (closure). **State:** Documented. ladybug-rs is the EARLIER prototype; nothing remains to harvest.

| ladybug-rs/src/spo/ | Lance-graph location |
|---|---|
| `nsm_substrate` | `crates/deepnsm/{codebook,fingerprint16k,encoder}.rs` |
| `sentence_crystal` | `crates/holograph/src/sentence_crystal.rs` (27 KB, same name) |
| `context_crystal` | `lance-graph-contract::grammar::context_chain` + `deepnsm::context` |
| `spo` (5^5 crystal) | `lance-graph-contract::crystal::Structured5x5` + `holograph::representation` |
| `spo_harvest` (238× cosine) | `bgz-tensor` + `highheelbgz` + `holograph::{hamming,bitpack,hdr_cascade}` + `ndarray::hpc::cascade` |
| `causal_trajectory` | `deepnsm::{trajectory,trajectory_audit,triangle_bridge}.rs` |
| `gestalt`, `meta_resonance` | `holograph::resonance` + `deepnsm::{ticket_emit,similarity}.rs` |
| `nsm_primes` | `deepnsm::nsm_primes` (exact match) |
| **`clam_path`** | **`crates/lance-graph/src/cam_pq/`** (5 files: `mod`, `ivf`, `jitson_kernel`, `storage`, `udf`) |
| `crystal_lm` | `crates/reader-lm/src/{classifier,inference,tokenizer,weights}.rs` |
| `codebook_*` | `deepnsm::codebook` + `bgz-tensor` |
| `deepnsm_integration` | The `deepnsm` crate IS this |
| **DN-tree ↔ crystal binding** | **`holograph::{dntree,dn_sparse,navigator}.rs`** (~214 KB combined) |

**Verdict:** the harvest is empty. Future sessions asked "harvest from ladybug-rs?" should consult this row before proposing work.

---

## Resolved new row introductions

### CYCLE-ACCUM-1 — `CycleAccumulator` per-cadence flush gate (entropy 2)

**Region:** R2 (companion to `collapse_gate::GateDecision`; distinct primitive per topology I-4). **State:** Wired (11 tests). **Smart/Dumb:** Smart-pure-data.

Component: `crates/lance-graph-contract/src/cycle_accumulator.rs`. Generic over commit type `C`. `push(commit) -> AccumulatorAction { Hold, Flush }` returns flush decision based on rows-since-flush threshold OR ms-since-flush threshold. `drain() -> Vec<C>` returns pending batch and resets.

**Why:** absorbs the ~10,000× speed ratio between Layer 1 cognitive cycle (20–200 ns/op) and Layer 3 outbound sinks (2–200 ms/external-write) per `SINGLE_BINARY_TOPOLOGY.md`. Architecturally prerequisite for Q2 Phase 3 BgzShaderDriver wiring (3M cycles per 300 ms window otherwise un-flushable).

**Evidence:** PR #337 (commit `bd61758c`).

### EWA-SANDWICH-1 — Pillar-6 covariance sandwich (entropy 2)

**Region:** R6/(jc). **State:** Wired (shipped #289, 7/7 tests, 10K/10K SPD). **Smart/Dumb:** Smart (`EwaSandwich::propagate(&[M], Σ_0)` carrier method).

Scope: SPD-bounded propagation of cognitive `Vsa16kF32` across Markov ρ^d cycles. NOT a lineage error model. Couples to (currently absent) SPLAT-1 ingestion path; per-row `sigma` carries through `propagate` cleanly only once `BindSpace.apply()` (E1) lands.

**Plan/Status:** jc / Shipped #289.

### SPLAT-EWA-BRIDGE-1 — SPLAT contract → EWA-Sandwich propagation bridge (entropy 2)

**Region:** R5/R6. **State:** Wired (example ships in `crates/jc/examples/splat_to_ewa_bridge.rs`). **Smart/Dumb:** Smart.

The cross-crate example bridges contract-types (SPLAT) to JC pillar math (EWA). Production code path through `BindSpace.apply()` (E1 seam) still missing — the bridge example is the *probe*, the *production wiring* awaits E1.

**Empirical (canonical 5-hop OSINT chain Lavender→IDF→Israel→NSO→Pegasus→Khashoggi):** All 5 hops SPD-preserved. Memory: 12.4 KB. Runtime: 107 µs end-to-end. Stress (1000-path × 10-hop): 1000/1000 SPD; 0.4 µs/path.

**Plan/Status:** splat-osint-ingestion-v1 (Active).

### MOCK-DRIVER-1 — `MockShaderDriver` (q2/cockpit-server) (entropy 2)

**Region:** R3. **State:** Wired (Phase-3 placeholder; 5/5 tests). **Smart/Dumb:** Smart (`impl CognitiveShaderDriver`).

Synthesizes `ShaderHit` from perturbation indices via `idx → row = idx % row_count`, `distance = i*64`, `resonance = 1.0 - i*0.1`. Phase-3 placeholder until real `BgzShaderDriver` lands. Disclosed at `mock_driver.rs:1-30`.

**Plan/Status:** Stalled-by-design (phase-3 implicit). **Evidence:** q2 PR #35.

### ONTOLOGY-REGISTRY-SOA-1 — Pillar 0 realized (entropy 2)

**Region:** R6/R0. **State:** Wired. **Maturity:** Stage 3 (consumed by bridge crates + `MappingRow` + callcenter). **Smart/Dumb:** Smart (carrier methods on registry; Click-P-1 satisfied).

Component: `lance-graph-ontology::OntologyRegistry`; schema IS the DTO + index; `enumerate(ns)` + `SchemaPtr` carrier methods. **O(1) probe:** registry p99 = 253 ns vs SPARQL-proxy p99 = 646,220 ns = **2554× ratio** (target ≥ 100× exceeded by 25.5×); 41/41 tests in `lance-graph-ontology`.

**LooseEnds (carry to OPEN cluster Per-row-context):** (a) `lance_cache.rs` Arrow schema doesn't yet persist the 7 new MappingRow cascade columns — replay defaults them; (b) per-row `BindSpace.context_ids` deferred to Wave-3.5.

**Plan:** `lance-graph-ontology-v5.md` Active. **Evidence:** PR #355 D-CASCADE-V1-3 + D-CASCADE-V1-11 (commits `8528161` + `fc49a29`).

### BUSDTO-BRIDGE-1 — engine_bridge BusDto → ShaderDispatch encode (entropy 2)

**Region:** R3. **State:** Wired (5/5 busdto_bridge tests). **Smart/Dumb:** Smart.

`engine_bridge.rs` `BusDto` → `ShaderDispatch::encode` → `BindSpace` SoA path. **Bit-exact** for `codebook_index` + positive-energy `top_k` + `cycle_count` + `converged` + all energies. LOSSY only for non-positive-energy `top_k` entries (idx → 0; energy still bit-exact via qualia).

**Plan/Status:** palantir-parity-cascade-v2 D-V2-3 / Active. **Evidence:** PR #355 (commit `8366e70`).

---

## State-change records (chronological)

### 2026-05-06 — VSA-1 description correction + cross-repo updates

**VSA-1 description corrected** (per CLAUDE.md I-VSA-IDENTITIES iron rule): `Vsa16kF32` is for ONE job — Markov chain over identity fingerprints (per-cycle cognitive state, role-keyed content, position-braided). Bundle size N ≤ √d / 4 ≈ 32 by concentration-of-measure. Click P-1 deficit (8 free fns vs methods on a newtype) stands. What does NOT stand: any framing that has `Vsa16kF32` carrying provenance, JWT claims, RBAC roles, transform IDs, branch IDs. Those are HashMap / typed-struct / Lance-column work. (VSA-1 row stays in OPEN — entropy 5.)

**PERMUTE-1 description corrected:** `vsa_permute` is unitary, but braiding is **SNR-bounded**, not lossless. (PERMUTE-1 row stays in OPEN — entropy 4.)

**SUBJECT-DTO-1 introduced** (Aspirational, entropy 4) — implied by MedCareV2 #7 + #8. Stays in OPEN.

**THINK-1 partial resolution (q2 PR #35):** cockpit-server migrated from `thinking_engine::dto::*` to canonical `lance_graph_contract::cognitive_shader::*`. Both `thinking-engine` and `cognitive-shader-driver` workspace deps DROPPED from cockpit-server. THINK-1 row entropy stays 5 workspace-wide; cluster has one fewer downstream consumer.

**TRUTH-1 partial resolution (q2 PR #35):** cockpit-server `graph_engine.rs::nars_deduction` bridges to `lance_graph_planner::nars::truth::TruthValue::deduction`. TRUTH-1 entropy stays 4; q2 is no longer the 4th copy.

**E8 retraction:** "geometrically-bounded provenance via Vsa16kF32 + EWA-Sandwich" retracted — register laziness; provenance is typed-struct / HashMap / Lance-column work, not VSA-bundled. **E4 retraction:** "VSA-bundled algebraic provenance" retracted for the same reason.

**Section G — Ingestion-vs-Traversal axis** added per sibling-session analysis. Two ingestion modes (Cypher-parser, splat-deposit) converge on the same E1 typed Action API. Traversal (Pillar-6) via `EwaSandwich::propagate` shipped #289.

### 2026-05-06 — WATCHER-1 Stalled (Stub) → Wired (sync, std-only)

**Old state:** `LanceVersionWatcher` wrapped `tokio::sync::watch::channel`, did not call `Dataset::version()`. Phase D TODO at `lance_membrane.rs:24`. Entropy 4. Cluster: Foundry seal.

**New state:** Wired (sync, std-only) per topology I-2 (tokio outbound only). Implementation uses `std::sync::{Arc, RwLock, Mutex, Condvar}` + `AtomicUsize`. Always-latest snapshot semantics preserved; slow-subscriber-skips-intermediates preserved. New entrypoints: `WatchReceiver::{current, wait_changed, wait_changed_timeout, try_changed}`. `LanceMembrane::Subscription` swapped from `tokio::sync::watch::Receiver` to `WatchReceiver` under the `realtime` feature.

**Evidence:** PR #337 (merged 2026-05-06T11:17:12Z; head SHA `c1fc1e5a`). Files: `crates/lance-graph-callcenter/src/version_watcher.rs` (full rewrite, 8 tests including I-2 invariant proof: `watcher_is_send_sync_without_runtime`), `crates/lance-graph-callcenter/src/lance_membrane.rs` (Subscription + test migration).

**MVCC bind to `Dataset::checkout_latest().version()` still pending** — see OPEN seam "R7 LanceVersionWatcher MVCC bind".

**Entropy:** 4 → 3 (transport correct; MVCC binding remains).

### 2026-05-06 — POLICY-1 / MEMBRANE-GATE-1 Stalled → Wired (both consumer sides)

**Old state:** Two `policy.rs` modules (rbac + callcenter) unconnected. No `impl MembraneGate for Arc<rbac::Policy>`. Entropy 4 (POLICY-1) + 3 (MEMBRANE-GATE-1).

**New state:** Wired on BOTH consumer sides via newtype-bridge pattern (orphan rule worked around by consumer-owned newtype):

- **SMB side (PR #29 in `smb-office-rs`, merged 2026-05-06):** `SmbMembraneGate` newtype wraps `Arc<lance_graph_rbac::Policy>` + `(role × entity_type)` binding. `impl MembraneGate for SmbMembraneGate` routes `gate_commit` to `Operation::Read` / `Operation::Write`. 13 unit tests; clippy clean.
- **Medcare side (PR #98 in `medcare-rs`, merged 2026-05-06T11:07:08Z):** `MedCareMembraneGate` newtype wraps `Arc<medcare_rbac::Policy>`. Same shape as SMB side; medcare-specific role catalogue (Doctor / Auditor / Receptionist / Admin × 6 entities Patient/Diagnosis/LabResult/Prescription/Anamnese/Ueberweisung). 33 tests across gate.rs + tests/integration.rs + tests/regulatory.rs.

**Both sides preserve three TD caveats from PR #29:** TD-MEMBRANE-FACULTY-BLIND, TD-MEMBRANE-ESCALATE-LOSSY, TD-MEMBRANE-FIRST-VS-ANY.

**Entropy:** POLICY-1: 4 → 2. MEMBRANE-GATE-1: 3 → 2.

### 2026-05-06 — SPLAT-EWA-BRIDGE-1 row + L1-L4 spatial-BLAS picture (PR #346)

L1+L2-popcount-AND probe (`splat_triangle_count.rs`) + L4 LPA superstep (`splat_lpa_label_propagation.rs`) shipped. SSB **5.4-5.8× faster on dense graphs** vs CSR (avg_degree ≥ 64 crossover). LPA stress: 100/100 graphs converged via Pillar-7 α-saturation. SplatShaderBlas naming graduated from "provisional" to "load-bearing concept".

### 2026-05-06 — Bitpacked vs Palette substrate clarification + 3 probes (PR #347)

Substrate-tier correction retracts implicit conflation: **Bitpacked tier** (popcount ops on `[u64; 256]`) vs **Palette tier** (BGZ17 distance-table lookup on 256-entry palette) are distinct. 20K × 20K Gaussian-splat lab work was Palette tier; this session's probes are Bitpacked tier.

3 new probes: Louvain modularity (purity 0.9975 on stress; **Q std 0.0036 after Codex denominator fix**), Jaccard + Adamic-Adar (d_J = 4.04 / d_AA = 3.81 discrimination + mutate-back), Perturbationslernen (4× lift over baseline; α-saturation triggered at iter 100 after Codex fixes; 25σ inter-class separation).

**Codex review fixes:** Louvain ΔQ denominator halved (= (2m)² instead of 2m²) — 3× tighter Q variance after fix; `debug_assert!` → `assert!` for SPD invariant (was silently passing in --release); α-saturation actually triggers (max_supersteps 20 → 200 + arithmetic-mean fix).

### 2026-05-07 — Pillar 4 + Pillar 11 activated; deferred 3 → 2 (default) / 1 (`--features hambly-lyons`) (PR #348)

**Pillar 4 (γ+φ preconditioner):** SOR(ω=φ=1.6180) vs Jacobi(ω=1) on N=50 stiff tridiagonal SPD systems — **step-count ratio 5.349×** (target ≥ 2.0×). γ + φ integrated as `std::f64::consts::EULER_GAMMA` + `GOLDEN_RATIO` constants.

**Pillar 11 (Hambly-Lyons signature uniqueness on tree-quotient):** gated behind `hambly-lyons` feature flag. Forward (tree-equivalence): max ‖S(out-and-back) − S_identity‖ = 0.000e0 across 100/100 pairs at depth-2. Converse (non-tree): min ‖S(triangle) − S_identity‖ = 0.0940. **Discrimination ratio = ∞**.

Pillar 2 (Cartan-Kuranishi) remains genuinely deferred — needs ML attention-mask module.

### 2026-05-07 — TTL-PROBE-5 Aspirational/Stub → Wired (PR #355 D-ONTO-V5-1)

Per-attribute `dcterms:source` provenance wired via sibling `AttributeProvenance` + `ProvenanceBundle` types in `proposal.rs`. Wave-3 cascade-cols (D-CASCADE-V1-7) threaded `attribute_sources` into `MappingRow`.

### 2026-05-07 — SPO-1 Stalled (Wired ×2 distinct, no bridge) → Wired (canonical bridge shipped) (PR #355 D-ONTO-V5-9)

**Old state:** Two SPO stores wired in parallel. `lance-graph::graph::spo::*` (fingerprint-keyed, HammingMin truth-semiring) + `lance-graph::graph::arigraph::triplet_graph` (string-keyed `HashMap<String, Vec<usize>>`, 1,072 LOC). Share only `TruthValue`. **No bridge fn** between them. Entropy 4. Cluster: Foundry seal.

**New state:** L1/L2 cache framing concretely realized via `SpoBridge::promote_to_spo` writer at `crates/lance-graph/src/graph/arigraph/spo_bridge.rs`. Warm string-keyed AriGraph (L1) → cold fingerprint-keyed SPO (L2) via FNV-1a `dn_hash`. The two stores are **tiers, not duplicates**.

**Evidence:** PR #355 D-ONTO-V5-9 (Wave 1, agent-spo-promote; commit `8528161`).

**Entropy:** 4 → 2. Maturity Stage 3.

---

## Closed open seams

| Seam | Closed by | Notes |
|---|---|---|
| ~~R4 RBAC ↔ MembraneGate~~ | smb-office-rs#29 + medcare-rs#98 | Newtype-bridge pattern (consumer-owned). Upstream trait shape unchanged. |
| ~~R7 LanceVersionWatcher MVCC (transport portion)~~ | PR #337 | Sync std-only WatchReceiver; MVCC `Dataset::version()` bind still pending. |
| ~~R6/R0 ontology-as-SoA seam (registry vs schema indirection)~~ | PR #355 | Pillar 0 realized; `OntologyRegistry::enumerate(ns)` IS the canonical surface. O(1) probe 2554× SPARQL-proxy. |
| ~~Pillar 4 (γ+φ preconditioner) deferred~~ | PR #348 | ACTIVE on default build; 5.349× step ratio. |
| ~~Pillar 11 (Hambly-Lyons) deferred~~ | PR #348 | ACTIVE under `--features hambly-lyons`; ∞ discrimination. |

---

## Aggregate — resolved + entropy delta over time

- **Total resolved rows tracked here:** 14 (4 from snapshot + 7 new at low entropy + 3 wired-equivalence findings).
- **State changes recorded:** WATCHER-1 (4→3), POLICY-1 (4→2), MEMBRANE-GATE-1 (3→2), TTL-PROBE-5 (closed), SPO-1 (4→2). Net entropy reduction: −7.
- **Open seams closed:** 5 (R4 RBAC, R7 transport, R6/R0 ontology-as-SoA, Pillar 4 deferred, Pillar 11 deferred).
- **New low-entropy primitives shipped:** CycleAccumulator (per-cadence gate, I-4), EWA-Sandwich (Pillar-6 SPD propagation), SPLAT contract types + witness_to_splat, OntologyRegistry (Pillar 0 SoA-as-canonical), MulThresholdProfile, BusDto bridge.

**Evidence chronology:**
- 2026-04-22 callcenter-membrane-v1 DM-0/DM-1 SHIPPED
- 2026-04-29 PR #243 BUNDLE-1 / Markov ±5 SHIPPED
- 2026-04-29 PR #272 ENT-TYPE-1 Phase 1 (Column H) SHIPPED
- 2026-04-30 PR #289 EWA-SANDWICH-1 SHIPPED
- 2026-04-30 PR #335 thought-cycle-soa-awareness-integration plan landed
- 2026-05-06 PR #336 SPLAT-1 Stage 1 contract types
- 2026-05-06 PR #337 WATCHER-1 transport + CycleAccumulator + MedCareMembraneGate
- 2026-05-06 PR #29 (smb-office-rs) SmbMembraneGate
- 2026-05-06 PR #98 (medcare-rs) MedCareMembraneGate
- 2026-05-06 PR #344 witness_to_splat (D-SPLAT-3)
- 2026-05-06 PR #345 .claude/pattern.md (15 named patterns)
- 2026-05-06 PR #346 SPLAT-EWA-BRIDGE-1 + L1-L4 spatial-BLAS empirical floor
- 2026-05-06 PR #347 Bitpacked-tier 3 probes + Codex review fixes
- 2026-05-07 PR #348 Pillar 4 + Pillar 11 activation; sigker landed
- 2026-05-07 PR #353 palantir-parity-cascade-v2 plan + soa-dto-dependency-ledger
- 2026-05-07 PR #355 Pillar 0 realized (palantir-cascade 11 deliverables / 12 agents / 3 waves)

**Cross-references:**
- `ARCHITECTURE_ENTROPY_LEDGER.md` — OPEN companion (active concerns)
- `SINGLE_BINARY_TOPOLOGY.md` — three-layer architecture invariants (I-1 single binary, I-2 tokio outbound only, I-3 BBB compile-time, I-4 per-row vs per-cadence gates distinct)
- `CROSS_REPO_PRS.md` — append-only log of merged PRs in other AdaWorldAPI repos that touch this workspace
- `MEDCARE_POLICY_GAP.md` — finding doc that scoped the medcare-side POLICY-1 closure
- `.claude/patterns.md` — traversal patterns guide (5 patterns + crate inventory + anti-patterns + wiring recipes)
- `.claude/pattern.md` (PR #345) — sister patterns doc (singular; 15 patterns)


---

## 2026-05-07 — RECOGNITION-1: the architecture is largely already shipped (sprint-2 meta-finding)

> **Append-only.** Sprint-2 (12 worker + 1 meta agent ensemble, branch
> `claude/unified-ogit-architecture-synthesis`) named 15 architectural
> patterns (A-O) from a 16-turn conversation. The central recognition:
> ~80% of the proposed architecture is already shipped across
> `thinking-engine`, `p64-bridge`, `cognitive-shader-driver`, `qualia.rs`,
> `causal-edge`, `bgz17` etc. Only ~20% (the OGIT-G overlay wiring +
> manifest pattern + ractor port) is genuinely new work.

### RECOGNITION-1 row scoring

| Field | Value |
|---|---|
| ID | RECOGNITION-1 |
| Region | workspace meta-finding (no R-id) |
| Component | Architectural pattern recognition framework (15 patterns A-O) |
| State | Documented (no code change) |
| DupCount | 0 (this is a closure, not a duplicate) |
| Maturity | Stage 4 (canonical recognition surface) |
| Smart/Dumb | n/a (meta) |
| Entropy | **1** (Clean — fully resolved as recognition) |
| Plan-status | n/a (informational closure) |

### Pattern → file recognition map (canonical: `.claude/knowledge/tier-0-pattern-recognition.md`)

| Pattern | Status | Already shipped in |
|---|---|---|
| H Switchable Cognitive Vessel | **shipped** | `crates/p64-bridge/src/lib.rs::cognitive_shader::CognitiveShader` (8 predicate planes + bgz17 semiring + HHTL cascade) |
| N Fingerprint-as-Codebook-Address | **shipped** | `crates/thinking-engine/src/prime_fingerprint.rs`, `qualia::FAMILY_CENTROIDS`, `p64-bridge::STYLES`, cam_pq codebook, bgz17 palette |
| O Phenomenological Memory | **shipped** | `crates/thinking-engine/src/qualia.rs` (17D + 10 families + music calibration + Bach 7+1 = CausalEdge64 7+1) + `awareness_dto.rs` |
| M Wave-Particle Bimodal | **shipped (primitives)** | bgz17/resonance/qualia (wave) + AriGraph/SPO/NARS (particle); G-blend mechanism is the new piece |
| F ractor message shape | **proven** | `crates/cognitive-shader-driver/src/grpc.rs` tonic service trait = the ractor handler shape |
| I Implicit Cognition | **shipped** | `CycleAccumulator` (PR #337) |

### Anti-Pattern surfaced: "Designing What's Already Built"

This generalizes the Discovery-Loop anti-pattern (already documented in `.claude/patterns.md`) from cycle-level (find the existing crate) to architecture-level (recognize the existing pattern). The 16-turn conversation repeatedly described future Pattern X work that was discovered, mid-conversation, to already exist in workspace. The cure is the same as Discovery-Loop: **READ existing code BEFORE proposing new design.**

### Five ledger row reframes (detail in OPEN ledger 2026-05-07 sprint-2 section)

| Row | Old (drift framing) | New (recognition) | Delta |
|---|---|---|---|
| THINK-1 | Spaghetti-5, 4-copy drift | 12-base codebook (`p64-bridge::STYLES`) + 36-entry composed surface (`contract::thinking::ThinkingStyle`); intentional layering | 5 → 3 |
| HEEL-1 | "3 different orderings" entropy 4 | Single canonical HHTL cascade in `p64-bridge::cognitive_shader::cascade` — "No POPCNT. No Hamming. Distance is PRECOMPUTED. O(1)." | 4 → 2 |
| ADJ-THINK-1 | Aspirational entropy 4 — "tau() never written" | The `[u64; 64]; 8` planes in `p64-bridge::CognitiveShader` IS the adjacency store; needs `tau_write()` public API only | 4 → 2 |
| CRYSTAL-1 | "Name collision" entropy 4 | Two legitimate codebooks at different Pattern N layers (sentence-level vs semantic-level) | 4 → 2 |
| CAM-DIST-1 | Stalled entropy 3 | One-line registration fix in `DataFusionPlanner::new`; substrate shipped | 3 → 2 |
| VSA-1 (bonus) | "Highest cognitive leverage carrier" entropy 5 | One Markov-accumulation program mode; CAM is the actual substrate | 5 → 3 |

**Aggregate entropy delta from recognition alone:** **−13** (no code changes).
**Cluster reorganization:** ~37 units (Thinking 24→~10, VSA carrier 23→~8, HEEL ladder 12→~4).

### Future-session implications

Sessions proposing "let's build the cognitive vessel" should hit RECOGNITION-1 first and be redirected to `p64-bridge::cognitive_shader::CognitiveShader`. The pre-work checklist in `.claude/patterns.md` should grow a step: **"Read `tier-0-pattern-recognition.md` first to see if the proposed architectural piece is already shipped."** (W3's append already added this.)

### Cross-references

- `.claude/plans/unified-ogit-architecture-v1.md` (W1 master synthesis, 30 KB)
- `.claude/knowledge/tier-0-pattern-recognition.md` (W2 file→pattern map, 21.8 KB)
- `.claude/patterns.md` (W3 appended Pattern Recognition Framework A-O)
- `.claude/board/EPIPHANIES.md` (W4 appended 17 architectural epiphanies)
- `.claude/board/TECH_DEBT.md` (W5 appended 11 TD entries: TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11)
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` (W6 appended 5 row reframes + VSA-1 clarification + 15-pattern absorption table)
- `.claude/board/INTEGRATION_PLANS.md` (W8 indexed the 4 new plan-docs)
- `.claude/board/LATEST_STATE.md` (W9 appended sprint-2 deliverables)
- `.claude/plans/ogit-g-context-bundle-v1.md` (W10 Tier-1 sub-plan, Patterns A+B+C)
- `.claude/plans/compile-time-consumer-binding-v1.md` (W11 Tier-2 sub-plan, Patterns E+F)
- `.claude/plans/anatomy-realtime-v1.md` (W12 proof-of-vision)

### Note on this row's provenance

This RECOGNITION-1 row was authored by W7 of the sprint-2 ensemble. The first W7 attempt pushed to the wrong repo (`AdaWorldAPI/ndarray`); this is the W7-rev2 correction landing on `AdaWorldAPI/lance-graph` directly via pygithub REST (main thread coordinator). The ndarray push remains as harmless residue at `AdaWorldAPI/ndarray/claude/unified-ogit-architecture-synthesis` for archaeology.
