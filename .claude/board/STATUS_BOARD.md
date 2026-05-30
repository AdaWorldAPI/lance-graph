# Status Board — Cross-Deliverable View

> Deliverable-level status across all active integration plans.
> **Status** and **PR / Evidence** columns are the only mutable
> fields — title, plan-version, and scope are immutable.
>
> For plan-level status see `INTEGRATION_PLANS.md`.
> For per-PR decision history see `PR_ARC_INVENTORY.md`.
> For current contract inventory see `LATEST_STATE.md`.

---

## Status Legend

| Status | Meaning |
|---|---|
| **Shipped** | Merged to main. PR column cites the merge commit. |
| **In PR** | PR open, under review. Not yet merged. |
| **In progress** | Active branch, code in flight, not yet PR. |
| **Queued** | Next up; spec is clear; work not started. |
| **Backlog** | Future; still in scope but not yet queued for a phase. |
| **Deferred** | Explicitly parked. Rationale recorded. Will be revisited. |
| **Abandoned** | Removed from scope. Rationale recorded. Will not be revisited. |

Rules:
- New rows APPEND (at the bottom of the relevant section).
- Status field is the ONLY field that gets edited in place.
- When a deliverable ships, record the PR number — never delete the
  row.
- When a deliverable is superseded by a different design, keep the
  row with Status = Abandoned and cite the replacement.

---

## normalized-entity-holy-grail-v1 — typed unified normalization + Op chain

Stage 1 contract surface scaffold. Typed consumer pipeline grammar that
unifies OGIT/OWL/DOLCE/Odoo inheritance + cognitive shader + JIT +
MailboxSoA into one surface. Plan path:
`.claude/plans/normalized-entity-holy-grail-v1.md`.

### Stage 1 deliverables (D-NEH-1a..g)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| **D-NEH-1a** | `cognition::{NormalizedEntity, stages, Op, OpKind, MailboxRow, Output}` typed surface | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1b** | `transaction::{Interactive, Bulk, Periodisch, Context, OgitCtx/OwlCtx/DolceCtx/FibuCtx}` context shapes | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1c** | 5-verb advancement methods on `NormalizedEntity<S>` | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1d** | `CascadeKind` + `TraversalMode` + `CascadeWalker` trait | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1e** | Compile-fail tests + 7 positive typestate tests | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1f** | Crate doc + example chain + `docs/COGNITION_HOLY_GRAIL.md` | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |
| **D-NEH-1g** | Board hygiene (AGENT_LOG + STATUS_BOARD) | **In PR** | Branch `claude/normalized-entity-holy-grail-v1` |

### Stage 2..7 deliverables (future plans)

| D-id | Title | Status |
|---|---|---|
| D-NEH-2a..z | ~50 Op kernel bodies + shader dispatch wiring | **Backlog** |
| D-NEH-3a..c | Consumer DSL macros (medcare/woa/smb) | **Backlog** |
| D-NEH-4a..b | Stream + GenServer integration | **Backlog** |
| D-NEH-5 | Jahresabrechnung kernel + fiscal-close JIT | **Backlog** |
| D-NEH-6 | palantir-foundry parity audit | **Backlog** |
| D-NEH-7 | elixir-OTP parity audit | **Backlog** |

---

## codec-sweep-via-lab-infra-v1 — JIT-first codec sweep

Active integration plan. 7 Phase 0 deliverables (D0.1–D0.7) + Phases
1–5 queued. One upfront Wire-surface rebuild; every candidate
afterwards is a JIT kernel, not a rebuild. Plan path:
`.claude/plans/codec-sweep-via-lab-infra-v1.md`.

### CI Gate — JC Substrate Proof

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| CI-JC | `.github/workflows/jc-proof.yml` — runs prove_it on every PR touching `crates/jc/` or `cam.rs` | **In PR** | 5-min timeout, exits 0 = substrate sound |

### Phase 0 — API hardening (partial in PR #225; remainder queued)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D0.1 | Extend `WireCalibrate` + `WireTensorView` (64-byte-aligned decode, object-oriented methods) | **Shipped** | #227 — 55/55 tests passing |
| D0.2 | `WireTokenAgreement` endpoint stub — I11 cert gate (Phase 0 surface, Phase 2 harness) | **In PR** | branch — `WireTokenAgreement` + `WireTokenAgreementResult` + `WireBaseline` DTOs + 3 round-trip tests. Stub handler returns `stub:true` / `backend:"stub"` until D2.1–D2.3 wire real decode-and-compare. |
| D0.3 | `WireSweep` streaming endpoint + Lance append stub | **In PR** | branch — `WireSweepGrid` + `cardinality()` + `enumerate()` → `Vec<WireCodecParams>` + `WireMeasure` enum + `WireSweepRequest` / `WireSweepResult` / `WireSweepResponse` DTOs + 5 tests. Streaming handler + Lance writer defer to Phase 3 D3.1. |
| D0.4 | Surface freeze (commit + rebuild) | **Ready** | D0.1–D0.7 all Shipped / In PR; freeze fires on merge of this PR. |
| D0.5 | `auto_detect.rs` — `ModelFingerprint` from `config.json` | **In PR** | branch — `auto_detect::{detect, ModelFingerprint, DetectError}` + HF config.json parser + per-architecture lane/distance heuristics (llama/qwen3/bert/modernbert/xlm-roberta/generic) + 8 tests. CODING_PRACTICES gap 1 remediated. |
| D0.6 | `CodecParamsBuilder` fluent API | **Shipped** | #225 — `contract::cam` +290 LOC of codec-params types, 14 tests (CODING_PRACTICES gap 3) |
| D0.7 | Precision-ladder validation (OPQ↔BF16x32, Hadamard pow2, overfit guard) | **Shipped** | #225 — `CodecParamsError` at `.build()` BEFORE JIT compile |

### Phase 1 — JIT codec kernels

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D1.1 | `CodecKernelCache` — structural cache layer (generic over handle) | **In PR** | branch — `CodecKernelCache<H>` + `StubKernel` + `get_or_compile` / `try_get_or_compile` with RwLock concurrent-safe double-check + compile/hit/ratio counters + 9 tests. Scaffold ships NOW; D1.1b Cranelift IR emission follows. |
| D1.1b | Adapter: `CodecKernelEngine` wrapping `ndarray::hpc::jitson_cranelift::JitEngine` with two-phase BUILD/RUN lifecycle (Arc-freeze). CodecParams → CodecScanParams adapter + codec-specific IR emission in jitson_cranelift/scan_jit analog | **Queued** | target ~250 LOC; `JitEngine` already ships (`/home/user/ndarray/src/hpc/jitson_cranelift/engine.rs`); the work is the CodecParams adapter + codec-specific JITSON template |
| D1.2 | Rotation primitives: Identity / Hadamard / OPQ as `RotationKernel` impls | **In PR** | branch — `RotationKernel` trait (Send+Sync+Debug, object-safe) + `IdentityRotation` (no-op) + `HadamardRotation` (real Sylvester butterfly, O(N log N) in-place, norm²-scaling verified) + `OpqRotationStub` (matrix-blob-id placeholder for D1.1b) + `build(&Rotation, dim)` factory + `RotationError` typed errors + 15 tests. Hadamard stays at Tier-3 F32x16 (add/sub, not matmul → no AMX benefit per Rule C). |
| D1.3 | Residual PQ via decode-kernel composition | **In PR** | branch — `DecodeKernel` trait (Send+Sync+Debug, object-safe, encode/decode/signature/bytes_per_row/dim/backend) + `StubDecodeKernel` (byte-exact round-trip for testing) + `ResidualComposer` (base + residual with subtract/add; nests recursively for depth >1) + `DecodeError` typed errors + 9 tests. Scope clarified: hydration/calibration path, NOT cascade inference (cascade uses `p64_bridge::CognitiveShader` per `cognitive-shader-architecture.md` line 582). |

### Phase 2 — Token-agreement harness (I11 cert gate) — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D2.1 | Token-agreement harness scaffold (reference model stub + top-k comparator + stub result) | **In PR** | branch — `ReferenceModel::{load, stub}` + `TokenAgreementError` + `TopKAgreement::{compare, top1_rate, top5_rate, meets_cert_gate, aggregate}` + `TokenAgreementHarness::{measure_stub, measure_full}` + 13 tests. Real safetensors load + decode loop defer to D2.2. |
| D2.2 | Decode-and-compare loop (top-k, per-layer MSE) | **Queued** | target ~220 LOC |
| D2.3 | Handler wiring for `/v1/shader/token-agreement` | **In PR** | branch — `token_agreement_handler` routes `WireTokenAgreement` → TryFrom(CodecParams) at ingress (precision-ladder + overfit guard fire here) → `ReferenceModel::load` or stub fallback on nonexistent paths → `TokenAgreementHarness::measure_stub()` → `WireTokenAgreementResult { stub:true }`. Route added: `POST /v1/shader/token-agreement`. Phase 0 Wire + Phase 2 harness now round-trip end-to-end. |

### Phase 3 — Sweep driver + Lance logger — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D3.1 | Server-side sweep handler + Lance fragment append | **In PR** | branch — `sweep_handler` batch mode: enumerates `WireSweepGrid::enumerate()`, validates each via TryFrom(CodecParams) at ingress, returns `WireSweepResponse { results: [WireSweepResult { kernel_hash, stub:true }], cardinality, elapsed_ms }`. SSE streaming + real calibrate/token-agreement per point deferred to D3.1b. Route: `POST /v1/shader/sweep`. |
| D3.2 | Client-side driver + config files | **In PR** | branch — 3 starter YAML configs (`configs/codec/{00_pr220_baseline, 10_wider_codebook, 12_hadamard_pre_rotation}.yaml`), `scripts/codec_sweep.sh` curl wrapper, `configs/codec/README.md`, YAML-shape spec-drift guard test. 118/118 tests pass. |

### Phase 4 — Frontier analysis — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D4.1 | DataFusion SQL over `sweep_results` Lance | **Queued** | target ~80 LOC |
| D4.2 | Pareto frontier notebook | **Queued** | target ~120 LOC |

### Phase 5 — Graduation — Fires per-candidate

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D5  | Graduation to canonical `OrchestrationBridge` (per winner) | **Queued** | target ~120 LOC per graduation; gate: ICC ≥ 0.99 held-out + token-agreement top1 ≥ 0.99 |

---

## elegant-herding-rocket-v1 — Phase-structured

Active integration plan, 12 deliverables D0 + D2–D11 (D1 dropped
early — CausalityFlow extension deferred). Plan path:
`.claude/plans/elegant-herding-rocket-v1.md`.

### Phase 1 — Shipped (PR #210, merged 2026-04-19)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D0  | grammar-landscape.md + linguistic-epiphanies + fractal-codec knowledge docs | **Shipped** | #210 — 3 docs, 1151 LOC |
| D4  | ContextChain reasoning ops (coherence / replay / disambiguate / WeightingKernel) | **Shipped** | #210 — 396 LOC, 8 tests |
| D6  | Role-key catalogue with contiguous `[start:stop]` slice addressing | **Shipped** | #210 — 404 LOC, 7 tests |

### Phase 2 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D2  | DeepNSM emits `FailureTicket` on low coverage (wiring step 4) | **Queued** | — |
| D3  | Grammar Triangle wired into DeepNSM via `triangle_bridge.rs` | **Queued** | — |
| D5  | Markov ±5 bundler + Trajectory + content_fp (wiring steps 1-3) | **Shipped** | PR #243 — `content_fp.rs` (98 LOC, 5 tests), `markov_bundle.rs` (250 LOC, 8 tests), `trajectory.rs` (298 LOC, 4 tests). 63 deepnsm tests pass. |
| D7  | Thinking styles + free-energy + RoleKey-as-operator | **Shipped** | PR #243 — `thinking_styles.rs` (490 LOC, 12 tests), `free_energy.rs` (347 LOC, 7 tests), `role_keys.rs` bind/unbind/recovery_margin (295 LOC added, 14 tests). 175 contract tests pass. |

### Phase 3 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D8  | Story-context bridge: AriGraph commit + global_context + contradiction (wiring steps 5-6) | **Queued** | — |
| D10 | Forward-validation harness (Animal Farm: chapter-10 > chapter-1 accuracy = AGI test) | **Queued** | — |

### Phase 4 — Backlog

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D9  | ONNX story-arc export + ArcPressure / ArcDerivative awareness hook | **Backlog** | — |
| D11 | Bundle-perturb emergence interface (transformer-free generative stack) | **Backlog** | — |

### Dropped / Deferred from the plan itself

| D-id | Title | Status | Notes |
|---|---|---|---|
| D1  | CausalityFlow 3→9 slot extension (modal/local/instrument + beneficiary/goal/source) | **Deferred** | User decision; follow-up PR after Phase 2 |

---

## Infrastructure / governance (not in elegant-herding-rocket)

Workspace-level bootstrap work. Tracked here rather than PR_ARC
because it's process, not architecture.

| Item | Status | PR / Evidence |
|---|---|---|
| CLAUDE.md §Session Start — three mandatory reads | **Shipped** | #211 |
| CLAUDE.md §A2A Orchestration — two layers (runtime + session) | **Shipped** | #211 |
| CLAUDE.md §Model Policy — grindwork vs accumulation + never Haiku | **Shipped** | #211 |
| CLAUDE.md §GitHub Access Policy — zipball-for-reads | **Shipped** | #211 |
| `.claude/BOOT.md` session entry + prior-art links | **Shipped** | #211 |
| `.claude/agents/BOOT.md` orchestration spec (renamed from README) | **Shipped** | #211 |
| `.claude/agents/README.md` function inventory | **Shipped** | #211 |
| `.claude/board/LATEST_STATE.md` current-state snapshot | **Shipped** | #211 |
| `.claude/board/PR_ARC_INVENTORY.md` append-only decision arc | **Shipped** | #211 |
| `.claude/board/INTEGRATION_PLANS.md` versioned plan index | **Shipped** | #211 |
| `.claude/board/STATUS_BOARD.md` this file | **Shipped** | #211 |
| `.claude/settings.json` team-shared governance (ask/deny + hooks) | **Shipped** | #211 |
| `.claude/hooks/session-start.sh` + `post-compact.sh` | **Shipped** | #211 |
| `.claude/skills/cca2a/` pattern-explanation skill | **Shipped** | #211 |
| `.claude/plans/elegant-herding-rocket-v1.md` plan in workspace | **Shipped** | #211 |

## Infrastructure — queued

| Item | Status | Notes |
|---|---|---|
| `.claude/rules/` with `paths:` frontmatter | **Backlog** | Audit rec 2; replace / complement `READ BY:` headers with path-scoped loading |
| Skill `context: fork` + `agent:` field | **Backlog** | Audit rec 4; read-only isolation for search-only skill variants |
| Auto memory (`~/.claude/projects/<proj>/memory/`) | **Backlog** | Audit rec; unstructured addition to curated LATEST_STATE |

---

## Cross-cutting research threads (orthogonal to grammar work)

Separate research thread — not entangled with grammar/crystal/A2A.
Tracked here so it doesn't get lost.

| Item | Status | Notes |
|---|---|---|
| Named-Entity pre-pass (NER) — biggest OSINT blocker | **Deferred** | Dedicated PR after Phase 2 |
| FP_WORDS = 160 migration (currently 157) | **Deferred** | Needs coordinated ndarray change |
| Crystal4K 41:1 persistence compression | **Deferred** | ladybug-rs owns it; would port later |
| 200–500 YAML TEKAMOLO templates per language | **Deferred** | Training pipeline; future |
| Cross-linguistic active parsers (EN+FI+RU+TR) | **Deferred** | Role keys exist; parsers later |
| Fractal-descriptor leaf codec (MFDFA on Hadamard) | **Research** | `.claude/knowledge/fractal-codec-argmax-regime.md`. 30-min probe first. |
| UK Biobank cardiac MRI benchmark | **Research** | Downstream of fractal-codec probe |
| Chess vertical (ruci + lichess-bot integration) | **Deferred** | Capstone Tier 0, parallel stream |
| Wikidata ingest (1.2 B triples → 14.4 GB) | **Deferred** | `.claude/knowledge/wikidata-spo-nars-at-scale.md` |
| OSINT pipeline (spider + reader-lm + DeepNSM) | **Deferred** | `.claude/knowledge/osint-pipeline-openclaw.md` |
| Python/TypeScript grammar-stack convergence | **Deferred** | `.claude/knowledge/grammar-landscape.md` §7 |

---

## Prior-art audit (61 + 41 = 102 existing docs)

Before this session, the workspace accumulated 61 `.claude/*.md`
top-level docs + 41 `.claude/prompts/*.md` files across prior
sessions. They are indexed in `.claude/BOOT.md §Existing content`
and `CLAUDE.md §Prior art`, but their individual **status** (still
active / superseded / archival) has not been audited.

Status rows per bucket, not per file (102 rows would drown the
board — use filesystem + INTEGRATION_PLANS + PR_ARC for per-file
history):

| Bucket | Count | Status | Notes |
|---|---|---|---|
| `.claude/*.md` top-level calibration reports / handovers / audits / snapshots | 61 | **Indexed** | Pointed at from BOOT.md + CLAUDE.md. Per-file active/superseded status: **Backlog** (needs one-pass audit). |
| `.claude/prompts/*.md` scoped session / probe / handover prompts | 41 | **Indexed** | Pointed at from BOOT.md via `SCOPED_PROMPTS.md` index. Per-file status: **Backlog**. |
| `.claude/knowledge/*.md` structured knowledge | 12 | **Active** | Current; each has `READ BY:` header; used by Knowledge Activation triggers. |
| `.claude/agents/*.md` specialist + meta-agent cards | 24 | **Active** | Current; used by spawning + Knowledge Activation. |
| `.claude/hooks/*.sh` | 2 | **Active** | Wired via settings.json. |
| `.claude/skills/cca2a/*.md` | 3 | **Active** | Current. |
| `.claude/plans/*.md` integration plans | 1 (v1) | **Active** | Elegant herding rocket v1, Phase 1 shipped. |

**Backlog item — prior-art audit.** One-pass sweep across the
61+41 files. Per file: label as active / superseded / archival
with a one-line note. Deliverable = an `ARCHIVE_INDEX.md` that
splits the 102 into current vs historical, plus rename/move of
superseded files into an `archive/` subdirectory. Estimate ~200
LOC of meta work, ~2 hours of reading. **Not urgent**; useful
before the next major planning session.

---

## ADR 0001 — Archetype transcode + Lance/DataFusion stack + Persona 16^32

Three-decision architectural lock, accepted 2026-04-24. First ADR in the
workspace. Path: `.claude/adr/0001-archetype-transcode-stack.md`.

| Decision | Status | Mutability |
|---|---|---|
| **D1 — Archetype is TRANSCODED, not bridged** | **Accepted** | Immutable (unlocking requires new ADR) |
| **D2 — Stack lock** (Lance + DataFusion + Supabase-shape scheduler + Arrow temporal; Polars rejected; Ballista deferred to 1s-P99) | **Accepted** | Ballista threshold mutable; rest immutable |
| **D3 — Persona 16^32 is THE identity space** (56-bit PersonaSignature; atom vector BBB-banned) | **Accepted** | Immutable; shared-DTO unification OPEN for future ADRs |

**Follow-up items tracked** (per ADR implications):

| Item | Priority | Location |
|---|---|---|
| DU-2 clarification (rename "bridge" → "transcode") | P2 | `unified-integration-v1.md` DU-2 |
| First `lance-graph-archetype` skeleton crate | P1 (when deliverable lands) | — |
| Grok gRPC A2A expert adapter | P2 | `TECH_DEBT.md` 2026-04-24 |
| Enrichment-shape follow-up ADR | P2 | `TECH_DEBT.md` 2026-04-24 |
| Ballista threshold tuning (post-benchmark amend) | P3 | `TECH_DEBT.md` 2026-04-24 |

Merged via PR #249 (2026-04-24).

---

## callcenter-membrane-v1 — Supabase-shape over Lance + DataFusion

External callcenter membrane crate. BBB enforced by Arrow type system at
compile time. Plan: `.claude/plans/callcenter-membrane-v1.md`. **Validated
by ADR 0001 Decision 2** (DM-4 `LanceVersionWatcher` + DM-6 `DrainTask`
pattern IS the Supabase-shape transcode approach).

### DM-0 / DM-1 — Shipped in this session

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| DM-0 | `ExternalMembrane` trait + `CommitFilter` in `lance-graph-contract/src/external_membrane.rs` | **Shipped** | session 2026-04-22 — `pub mod external_membrane` added to contract lib.rs |
| DM-1 | `lance-graph-callcenter` crate skeleton: `Cargo.toml` (feature gates) + `src/lib.rs` (stub + UNKNOWN markers) | **Shipped** | session 2026-04-22 — added to workspace members |

### DM-2 through DM-9 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| DM-2 | `LanceMembrane: ExternalMembrane` impl with `project()` + compile-time BBB leak test | **In progress** | Phase A shipped `9a8d6a0` — `LanceMembrane` struct + `project()` + `ingest()` + `subscribe()` stub. Phase B: full Lance append + version counter pending DM-4. |
| DM-3 | `CommitFilter` → DataFusion `Expr` translator (`[query]` feature) | **Queued** | — |
| DM-4 | `LanceVersionWatcher` — tails Lance version counter, emits Phoenix `postgres_changes` (`[realtime]`) | **In PR** | branch `claude/supabase-subscriber-wire-up` — DM-4a/b/c: `version_watcher.rs` (117 LOC, 4 tests), `lib.rs` `pub mod version_watcher`, `LanceMembrane::watcher` field + `project()` calls `bump()`, `subscribe()` returns `watch::Receiver<CognitiveEventRow>`. |
| DM-5 | `PhoenixServer` — minimal WS server, Phoenix channel subset (`[realtime]`) | **Queued** | Resolve UNKNOWN-2 (which consumers need Phoenix wire?) first |
| DM-6 | `DrainTask` — `steering_intent` Lance read → `UnifiedStep` → `OrchestrationBridge::route()` | **In PR** | branch `claude/supabase-subscriber-wire-up` — DM-6a/b scaffold: `drain.rs` (89 LOC, 2 tests), `lib.rs` `pub mod drain`, `Poll::Pending` until follow-up PR wires real drain loop. |
| DM-7 | `JwtMiddleware` + `ActorContext` → `LogicalPlan` RLS rewriter (`[auth]`) | **Queued** | Resolve UNKNOWN-3 (pgwire?) + UNKNOWN-4 (actor_id type) first |
| DM-8 | `PostgRestHandler` — query-string → DataFusion SQL → Lance scan → Arrow response (`[serve]`) | **Queued** | Confirm PostgREST compat needed (§ 8 stop point 4) before building |
| DM-9 | End-to-end test: shader fires → `LanceMembrane::project()` → Lance append → Phoenix subscriber receives event | **Queued** | Depends on DM-2 through DM-6 |

---

## grammar-foundry-followup-v1 — Wire stubs to existing tissue

Plan: `.claude/plans/grammar-foundry-followup-v1.md`. Session 2026-04-29.
Six explicit stubs in PRs #275-#283 + 1 keystone (LF-12 Pipeline DAG). 13 PRs total in 3 waves.

### Wave 1 — no deps (parallel)

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-S1 | LF-12 Pipeline DAG: `UnifiedStep.depends_on` + topological executor | **Queued** | Keystone. Unblocks F4, G2, G6 |
| PR-F1 | PolicyRewriter UDF wrap: `RedactionMode` executors (closes `policy.rs:122`) | **Queued** | Unblocks F2, F5 |
| PR-F3 | Audit log Lance-backed writer (closes `lib.rs:100`) | **Queued** | |
| PR-F6 | `dn_path.rs` real scent via CAM-PQ (closes `dn_path.rs:53`) | **Queued** | Risk: bgz-tensor dep |
| PR-G1 | Triangle bridge real Causality footprint (closes `triangle_bridge.rs:90,221`) | **Queued** | |
| PR-G3 | ContextChain real `Binary16K` fingerprint (closes `context_chain.rs:345`) | **Queued** | |
| PR-G4 | verb_table seed 10/12 families (closes empty `default_table()` rows) | **Queued** | |
| PR-G5 | AriGraph episodic unbundle/rebundle (per `integration-plan-grammar-crystal-arigraph.md`) | **Queued** | |

### Wave 2 — depends on Wave 1

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-F2 | RowEncryption + DifferentialPrivacy executors (closes `policy.rs:147,181`) | **Queued** | After F1; needs key-mgmt ADR |
| PR-F4 | PostgREST → DataFusion dispatch (closes `EchoHandler` stub) | **Queued** | After S1 |
| PR-F5 | `audit_from_plan()` helper (closes `orchestration.rs:202` `unimplemented!`) | **Queued** | After F1 |
| PR-G2 | Disambiguator wiring at parser boundary + FailureTicket emission | **Queued** | After S1 |

### Wave 3 — depends on Waves 1+2

| D-id | Title | Status | Notes |
|---|---|---|---|
| PR-G6 | Animal Farm harness real run (D10 from PR #243) | **Queued** | After G1+G2+G3; text licensing needed |

---

## unified-integration-v1 — PersonaHub × ONNX × Archetype × MM-CoT × RoleDB

Plan: `.claude/plans/unified-integration-v1.md`. Session 2026-04-23.

| D-id | Title | Status | Notes |
|---|---|---|---|
| DU-0 | PersonaHub 56-bit compression: `(atom_bitset: u32, palette_weight: u8, template_id: u16)` offline extraction from 370M HF parquet rows | **Queued** | Runs offline; no code deps. Output: `personas.bin` + `sigs_dedup.bin` + `templates/*.yaml` |
| DU-1 | ONNX persona classifier @ L4/L5 — 288-class `(ExternalRole × ThinkingStyle)` product prediction; `style_oracle: Option<&OnnxPersonaClassifier>` in Think struct | **Queued** | Needs ~10K labeled cycles from Lance internal_cold (DM-2 must ship first); replaces Chronos proposal |
| DU-2 | Archetype ECS bridge crate `lance-graph-archetype-bridge` — `ArchetypeWorld → Blackboard`, `ArchetypeTick → UnifiedStep`, `project() → DataFrame component` adapters | **Queued** | Needs DM-2 (ExternalMembrane impl) before adapter can be built |
| DU-3 | RoleDB DataFusion VSA UDFs: `unbind`, `bundle`, `hamming_dist`, `braid_at`, `top_k` — registers in DataFusion session | **Queued** | Fingerprint column type decision needed first (FixedSizeBinary vs FixedSizeList); see open question in plan § 5 |
| DU-4 | MM-CoT stage split: add `rationale_phase: bool` to `CognitiveEventRow`; surface `FacultyDescriptor.is_asymmetric()` in projected RecordBatch | **Shipped** (Phase A: 2026-04-23 `a05979e`; Phase B: 2026-04-24) | Phase A: field exists. Phase B: `set_faculty_context()` on `LanceMembrane` wires `rationale_phase` from `AtomicBool`; orchestration layer calls it with `FacultyDescriptor::is_asymmetric()` + stage. Column is live, not ghost. |
| DU-5 | Board hygiene: DU-0 through DU-4 registered; INTEGRATION_PLANS.md + LATEST_STATE.md updated | **Shipped** (2026-04-23, commit `a05979e`) | Plan corrections + precision-tier §18 + father-grandfather concept committed in follow-up. |

## splat-osint-ingestion-v1 — Splat contract + EWA OSINT bridge

Active plan, 7 deliverables (D-SPLAT-1..7) staged across 6 PRs of the
`gaussian-splat-cam-plane-workaround.md` doc-sequence. PR 1+2 in flight
on branch `claude/splat-osint-ingestion`.
Plan path: `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`.

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-SPLAT-1 | `crates/lance-graph-contract/src/splat.rs` — `SplatChannel`, `CamPlaneSplat`, `SplatPlaneSet`, `AwarenessPlane16K`, `CamSplatCertificate`, `SplatDecision`, `TriadicProjection`, `ReasoningWitness64` + 10 unit tests | **In PR** | branch `claude/splat-osint-ingestion` |
| D-SPLAT-2 | `crates/jc/examples/osint_edge_traversal.rs` — EWA-Sandwich Σ-push-forward demo for OSINT 5-hop chain, side-by-side vs naive convolution | **In PR** | branch `claude/splat-osint-ingestion` |
| D-SPLAT-3 | `witness_to_splat()` deterministic conversion (PR 2 of doc-sequence) | **In PR** | branch `claude/phase-3b-witness-to-splat` |
| D-SPLAT-4 | Splat deposition into BindSpace columns via `MergeMode::AlphaFrontToBack` lanes (PR 3 of doc-sequence) | **Queued** | — |
| D-SPLAT-5 | `PlanarSplatBundle4096` with local/short/medium/long bands (PR 4 of doc-sequence) | **Queued** | — |
| D-SPLAT-6 | Semantic-CAM-distance integration — survivor tile selection vs splatted pressure planes (PR 5 of doc-sequence) | **Queued** | — |
| D-SPLAT-7 | Replay fallback — exact 4096-cycle ThoughtCycleSoA replay slice when certificate insufficient (PR 6 of doc-sequence) | **Queued** | — |

Cross-ref: SPLAT-1 row in `ARCHITECTURE_ENTROPY_LEDGER.md` (Aspirational → Wired stage 1, entropy 4 → 2).

---


## causaledge64-mailbox-rename-soa-v1 — sprint-10 spec corpus + sprint-11 impl queue

Active integration plan. Specs shipped via PR #372 (merged 2026-05-14, governance-only).
Plan path: `.claude/plans/causaledge64-mailbox-rename-soa-v1.md`.

### Sprint-10 — spec sprint (12 CCA2A workers + Opus meta) — Shipped

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CE64-MB-1 | par-tile crate apex + Mailbox<T> + 3 backings + AttentionMask SoA + BindSpaceView | **Spec shipped** | #372 — `pr-ce64-mb-1-par-tile-crate.md` (W1) |
| D-CE64-MB-2 | CausalEdge64 v2 layout proposal + OQ-LAYOUT-1 BLOCKER finding | **Spec shipped** | #372 — `pr-ce64-mb-2-causaledge64-v2.md` (W2) |
| D-CE64-MB-2-regress | PAL8 / NARS regression tests (accessor-based, post-OQ-LAYOUT-1) | **Spec shipped** | #372 — `pr-ce64-mb-2-pal8-nars-regression.md` (W3) |
| D-CE64-MB-3 | BindSpace E/F/G/H column extension | **Spec shipped** | #372 — `pr-ce64-mb-3-bindspace-efgh.md` (W4) |
| D-CE64-MB-4 | AriGraph SPO-G + ghost edges + SpoWitnessChain + SCHEMA_VERSION 2→3 | **Spec shipped** | #372 — `pr-ce64-mb-4-arigraph-spo-g.md` (W5) |
| D-CE64-MB-5 | MailboxSoA<N> + AttentionMaskActor (single tick per cycle) | **Spec shipped** | #372 — `pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) |
| D-CE64-MB-6 | SigmaTierRouter + banding + INT4-32D cold-start + Hebbian plasticity + KernelHandle cache + Σ9-10 escalation | **Spec shipped** | #372 — `pr-ce64-mb-6-sigma-tier-router.md` (W7) |
| D-CE64-MB-7 | bevy 0.14 cull plugin proof-PR | **Spec shipped** | #372 — `pr-ce64-mb-7-bevy-cull-plugin.md` (W9) |
| D-NDARRAY-MIRI-COMPLETE | Miri coverage ~760 → ~1550 | **Spec shipped** | #372 — `pr-ndarray-miri-complete.md` (W8) |
| D-SPRINT-10-DEPGRAPH | 8 PRs × 6 waves + parallel-landability + cross-spec consistency checks | **Spec shipped** | #372 — `sprint-10-pr-dep-graph.md` (W10) |
| D-SPRINT-10-TESTPLAN | Unified test plan + Miri growth target + proptest Miri runtime | **Spec shipped** | #372 — `sprint-10-test-plan.md` (W11) |
| D-SPRINT-10-EXECPLAN | Sprint-11 fleet definition + post-merge governance + worker prompt template | **Spec shipped** | #372 — `sprint-10-execution-plan.md` (W12) |
| D-SPRINT-10-META | Opus meta-review (CSI-1..6 + E-META-1..5 + sprint-11 gate decision) | **Shipped** | #372 — `.claude/board/sprint-log-10/meta-review.md` |

### Sprint-11 — implementation wave — Queued (blocked)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CE64-MB-1-impl | par-tile crate impl (W1 → code) | **Queued** | blocked on OQ-5 user ratification (rayon vendor) |
| D-CE64-MB-2-impl | CausalEdge64 v2 layout impl (W2 → code) | **Queued** | blocked on CSI-1 user ratification (which Option A/B/C/D/E for bit reclaim) |
| D-CE64-MB-2-regress-impl | PAL8 / NARS regression test impl (W3 → code) | **Queued** | blocked on D-CE64-MB-2-impl |
| D-CE64-MB-3-impl | BindSpace E/F/G/H impl (W4 → code) | **Queued** | blocked on D-CE64-MB-1-impl |
| D-CE64-MB-4-impl | AriGraph SPO-G + ghosts impl (W5 → code) | **Queued** | blocked on D-CE64-MB-2-impl |
| D-CE64-MB-5-impl | MailboxSoA + AttentionMaskActor impl (W6 → code) | **Queued** | blocked on OQ-3 user ratification (plasticity granularity) + CSI-2 spec patch (g_slot_at_drop field) |
| D-CE64-MB-6-impl | SigmaTierRouter impl (W7 → code) | **Queued** | blocked on OQ-1 user ratification (Σ4-Σ5 banding) + CSI-3 spec patch (PR-J1 Wave 0.5 prerequisite) |
| D-CE64-MB-7-impl | bevy cull plugin impl (W9 → code) | **Queued** | blocked on D-CE64-MB-1-impl + CSI-4 spec patch (BindSpaceView::empty_static() ctor in W1) |
| D-NDARRAY-MIRI-COMPLETE-impl | Miri coverage impl (W8 → code) | **Queued** | independent; can spawn first |
| D-PR-J1-INT4-32D-ATOMS | INT4-32D codebook for SigmaTierRouter cold-start | **Queued** | new Wave 0.5 prerequisite; not in original W10 dep graph |
| D-CSI-2 | W6 CompartmentReport `g_slot_at_drop: u8` field patch | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-3 | W10 dep graph PR-J1 Wave 0.5 row patch | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-4 | W1 spec `BindSpaceView::empty_static()` + `from_arc()` constructors | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-5 | W1 spec move `SigmaTier` to `lance-graph-contract::orchestration` | **Queued** | small spec edit; pre-sprint-11 |
| D-CSI-6 | W11 test-count drift reconciliation | **Queued** | small spec edit; pre-sprint-11 |

### User-ratification gates (block sprint-11 spawn)

| Gate | Wave blocked | Resolution path |
|---|---|---|
| **CSI-1** — CausalEdge64 bit-reclaim Option (A/B/C/D/E) | Wave 2 (D-CE64-MB-2-impl) | User picks; meta-review recommends Option C-conservative (drop temporal + G-slot, allocate W-slot + lens) |
| **OQ-1** — Σ4-Σ5 banding (Tokio reflex vs InMem cycle-speed) | Wave 5 (D-CE64-MB-6-impl) | Default Tokio is safe-to-ship; ratification only PROMOTES |
| **OQ-3** — Plasticity update granularity (bit-counter per emission + NARS revise at AriGraph commit) | Wave 4 (D-CE64-MB-5-impl) | Tentative resolution recorded; user formal-acknowledge |
| **OQ-5** — Rayon vendor decision (std::thread::scope first vs vendored-rayon) | Wave 1 (D-CE64-MB-1-impl) | Tentative defer; user formal-acknowledge |

### Reunification track (sprint-12+)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-REUNIFY-1 | Acknowledge dual `CausalEdge64` types in TYPE_DUPLICATION_MAP + LATEST_STATE + EPIPHANIES | **Shipped** | this commit (post-merge #372 board-hygiene tail) |
| D-REUNIFY-2 | 8-channel → SPO transcoder spec at thinking-engine L3 commit boundary | **Backlog** | per Option R-3; sprint-12+ |
| D-REUNIFY-3 | `Think` carrier struct prototype unifying thinking-engine cascade + cognitive-shader-driver SoA | **Backlog** | per `.claude/knowledge/splat-shader-rayon-struct-method-vision.md` sprint-12 |
| D-REUNIFY-4 | Splat op fleet (`splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany`) as methods on `Think` | **Backlog** | sprint-13+ |
| D-REUNIFY-5 | rayon work-stealing par_* method variants | **Backlog** | sprint-14+ |
| D-REUNIFY-6 | OWL DOLCE / OntologyFilter wiring into `emit_causal_edges_filtered` | **Backlog** | sprint-15+ |

---

## cognitive-substrate-convergence-v1 — i4 mantissa + gapless baton + active inference

Active integration plan. Authored 2026-05-15 (cross-session A2A discussion).
Plan path: `.claude/plans/cognitive-substrate-convergence-v1.md`.
Consolidates sprint-10 architectural decisions before context dilution.

### Phase A — Substrate primitives (sprint-11)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-1 | `causal-edge` crate v2 layout (signed mantissa, W-slot, lens, drop temporal) | **Shipped** | PR #383 merge `03bd175`; OQ-CSV-2 ratified to 6 bits (default) |
| D-CSV-2 | `QualiaI4_16D` type in `lance-graph-contract::qualia` + f32↔i4 migration helpers | **Shipped** | PR #384 merge `0751a8b`; OQ-CSV-1 ratified to Option α (canonical convergence-observable vocab; drop dim 16 "integration") |
| D-CSV-3 | InferenceType signed-mantissa expansion (absorbs PR-LL-1 Intervention/Counterfactual into canonical edge enum) | **Shipped** | PR #383 merge `03bd175`, paired with D-CSV-1 in same crate |
| D-CSV-4 | `CollapseGateEmission` wire format spec + impl per plan §8 | **Shipped** | PR #383 merge `03bd175`, contract crate (Vec instead of SmallVec to preserve zero-dep — TD-COLLAPSE-GATE-SMALLVEC-1) |

### Phase B — Storage & dispatch path (sprint-11)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-5a | QualiaColumn migration phase 5a — sibling `QualiaI4Column` add + double-write (no read-side change) | **Shipped** | PR #385 merge `6f58418`; OQ-CSV-4 ratified to sibling-cutover (default); 5b cutover follows in separate PR |
| D-CSV-5b | QualiaColumn migration phase 5b — flip readers to i4, drop f32 column, drop f32 push arg | **In PR (#390 W-G1)** | sprint-12 Wave G fleet; depends on D-CSV-5a (merged) + downstream reader audit |
| D-CSV-6a | `WitnessCorpus` partial (W-slot anchor + chain invariant; sorted by emission cycle, drop-oldest truncation) | **Shipped** | PR #386 merge `33110c8` (paired with D-CSV-7) |
| D-CSV-6b | `WitnessCorpus` full (CAM-PQ-indexed, unbounded, salience decay) | **In PR (#390 W-G2)** | sprint-12 Wave G fleet; depends on D-CSV-6a (merged) |
| D-CSV-7 | MailboxSoA integration: W-slot referencing + per-row plasticity accumulator + apply_edges | **Shipped** | PR #386 merge `33110c8` (paired with D-CSV-6a) |

### Phase C — Reasoning path (sprint-12)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-8 | MUL evaluation in integer SIMD: DK/TrustTexture/FlowState/GateDecision consume i4 qualia + signed mantissa | **Shipped** | PR #387 merge `e042c70` (scalar i4 path; AVX-512/NEON deferred → D-CSV-13/13b sprint-13 per TD-D-CSV-8-SIMD-1) |
| D-CSV-9 | 8-channel ↔ SPO-palette transcoder (Option R-3) at thinking-engine L3 commit boundary | **Shipped** | PR #387 merge `e042c70` (paired with D-CSV-8) |
| D-CSV-10 | Σ-tier Rubicon-resonance dispatch in `SigmaTierRouter`: ΔF + resonance threshold → Σ10 commit | **In PR (#388 W-F1)** | sprint-12 Wave F; sigma-tier-router crate present in workspace post-Wave G #390 cargo metadata (hand-tuned threshold per OQ-CSV-6; Jirak-derived → D-CSV-15 sprint-13+) |

### Phase D — Streaming infrastructure (sprint-12 productization)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-11a | Vertical streaming structs in ndarray: `QualiaStream` / `QualiaI4Row` | **Shipped** | ndarray PR #147 merge `d867b1c` |
| D-CSV-11b | Vertical streaming structs in ndarray: `InferenceStream` / `InferenceRow` | **Shipped** | ndarray PR #147 merge `d867b1c` |
| D-CSV-11c | Vertical streaming structs in ndarray: `SplatFieldStream` (+ `par_*` rayon variants deferred to sprint-14+ behind `parallel` feature) | **Shipped** | ndarray PR #147 merge `d867b1c`; `par_*` rayon variants deferred (Queued sprint-14+) |
| D-CSV-12 | Splat shader op fleet (`splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany`) — scalar standalone ops | **Shipped** | PR #388 merge `77f2d26` (W-F7 scalar; on-Think method migration → D-CSV-14 sprint-13) |

### Phase E — Sprint-12/13 new entries (NEW in v2 + sprint-13 preflight)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D-CSV-13 | Batch i4 scalar MUL (paired with D-CSV-8 SIMD-readiness) | **Shipped** | PR #388 merge `77f2d26` (W-G3 batch i4 scalar) |
| D-CSV-13b | SIMD vectorization of D-CSV-8 i4 MUL evaluation (AVX-512 + NEON intrinsics) | **In PR (sprint-13/W-I1 salvage)** | branch `claude/sprint-13-w-i1-salvage`; AVX-512F+BW dispatch via `simd_caps()`; bench on Skylake-AVX512 host = 8.7× dk / 7.4× trust / 5.2× flow / 10.2× gate_disc / 3.1× mul_assess at batch 1024 — all SHIP gates met; 5 SIMD-vs-scalar parity tests over 10 sizes green |
| D-CSV-14 | On-Think method migration for D-CSV-12 splat ops (struct-method surface per L-20) | **Queued (PP-4 spec drafting)** | sprint-13; depends on D-CSV-11 streaming substrate (shipped via ndarray #147) |
| D-CSV-15 | Σ10 Jirak-derived threshold (TD-SIGMA-TIER-THRESHOLDS-1 resolution) | **In PR (#390 W-G4 Jirak threshold)** | sprint-12 Wave G partial; full VAMPE coupled-revival deferred sprint-13+ |
| D-CSV-16 | NEW sprint-13 entry | **Queued (PP-5 spec drafting)** | sprint-13 preflight |
| D-CSV-17 | NEW sprint-13 entry | **Queued (PP-3 spec drafting)** | sprint-13 preflight |

### Open-question gates (block specific D-CSV-* spawns)

| Gate | Blocks | Recommendation |
|---|---|---|
| **OQ-CSV-1** Qualia 16D per-dim assignment | D-CSV-2, D-CSV-5 | Ratify proposed §7.2 layout with `qualia-engineer` agent cross-check against `thinking-engine/src/qualia.rs` |
| **OQ-CSV-2** W-slot width 6 vs 8 bits | D-CSV-1 | Default 6 (= 64 active corpora); promote to 8 if multi-tenant SaaS demands |
| **OQ-CSV-4** QualiaColumn migration phasing | D-CSV-5 | Default sibling-column-then-cutover (lower risk; 1 extra PR worth it) |
| **OQ-CSV-6** Σ10 Rubicon threshold derivation | D-CSV-10 (sprint-12) | Hand-tuned acceptable for sprint-11/12 with TECH_DEBT note per `I-NOISE-FLOOR-JIRAK`; principled Jirak derivation deferred to VAMPE coupled-revival sprint-13+ |

### Cross-spec patches (one bundled prep PR pre-sprint-11) — **SHIPPED via PR #381 (merged 2026-05-16, commit `a7c0545`)**

| Spec | Patch | LOC | Status |
|---|---|---|---|
| `pr-ce64-mb-2-causaledge64-v2.md` (W2) | §3 bit layout → plan §6; OQ-LAYOUT-1 resolved; signed-mantissa rationale; G-slot API stripped from test plan + risk matrix (codex P1) | ~160 actual | **Shipped** |
| `pr-ce64-mb-2-pal8-nars-regression.md` (W3) | Tests parameterized on v2 layout; mantissa roundtrip + lens 4-state; v1-temporal=0 safe-migration fix + version-gate test (codex P1) | ~370 actual | **Shipped** |
| `pr-ce64-mb-3-bindspace-efgh.md` (W4) | QualiaColumn migration step (D-CSV-5) cross-ref | ~40 actual | **Shipped** |
| `pr-ce64-mb-4-arigraph-spo-g.md` (W5) | `SpoWitnessChain<32>` → `WitnessCorpus`; `W5-INV-CHAIN-ORDER` invariant; W-slot semantics | ~316 actual | **Shipped** |
| `pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) | `g_slot_at_drop` field (CSI-2); spatial-temporal accumulator semantics | ~50 actual | **Shipped** |
| `pr-ce64-mb-6-sigma-tier-router.md` (W7) | Σ10 Rubicon-resonance threshold; integer-SIMD MUL path | ~120 actual | **Shipped** |
| `sprint-10-pr-dep-graph.md` (W10) | PR-J1-INT4-32D-ATOMS + CAM-PQ wiring elevated to Wave 3 hard dep | ~50 actual | **Shipped** |
| `sprint-10-test-plan.md` (W11) | Refresh test counts for v2; i4-roundtrip + signed-mantissa-product tests | ~87 actual | **Shipped** |

**Total spec-patch LOC:** ~1,200 actual across 5 commits (`9bd66d9`, `f730528`, `5253c79`, `e4d15a3`, `33509ab`) merged 2026-05-16 in PR #381. Original ~870 estimate undershot W3 (codex P1 fix added ~280 LOC) and W5 (full WitnessCorpus section added ~16 LOC over estimate). All 8 workers complete. Sprint-11 spawn now unblocked on the spec-patch dimension; remaining gates: OQ-CSV-1, OQ-CSV-2, OQ-CSV-4 user ratifications.

---

## rung-persona-orchestration-v1 — time-bound persona orchestration (checklist → meta-recipe → hot/cold/feedback anneal)

Active proposal. Authored 2026-05-26. Plan path:
`.claude/plans/rung-persona-orchestration-v1.md`. Sibling/time-bound
composition layer over `rung-mul-grounding-v1`. Grounds ladybug's
hot/cold/feedback loop onto our contract types + SoA floor
(restore-on-SoA, not port). Epiphany: `E-RIGID-RULES-OPEN-DOORS`.

| D-id | Title | Crate(s) | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-PERSONA-1 | escalation+epiphany loop = the checklist (`felt_parse` collapse-hint + `InnerCouncil`/`HdrResonance` split + `EpiphanyDetector`; green-flip = Epiphany/Wisdom ghost) — NOT a bespoke verifier | contract + planner | 160 | LOW | **In progress** | branch `claude/splat3d-cpu-simd-renderer-MAOO0` |
| D-PERSONA-2 | meta-recipe manifest (declarative child-spec, recipe-as-data, macro-evaluable) | contract | 150 | MED | **Queued** | — |
| D-PERSONA-3 | hot/cold/feedback wiring — anneal + `CrystalCodebook`→wisdom-marker cold path + Preload hydrate | planner + Lance | 240 | MED | **Queued** | — |
| D-PERSONA-4 | macro-eval harness (scenario→trace→discover→diagnose; suspect-bridge = blasgraph betweenness; 5 rubrics from D-RUNG-MUL) | planner + Lance | 280 | HIGH | **Queued** | — |
| D-PERSONA-5 | ractor outer-swarm runtime under `OrchestrationBridge` (batons as messages, async only at boundary) | planner | 200 | MED | **Queued** | — |
| D-PERSONA-6 | `odoo_scanner` + `OdooBridge` — harvest Odoo `l10n_de` → Finance-ns `MappingProposal`s; bind existing `TaxEngine`; GoBD by construction | ontology + contract + planner | 280 | MED | **Queued** | — |

---

## unified-soa-convergence-v1 — ONE LE SoA end-to-end across 9 consumers + version gate + Lance 6.0.1 stack + 4-phase Rubicon kanban

Plan path: `.claude/plans/unified-soa-convergence-v1.md`. Handover `.claude/handovers/2026-05-29-1825-soa-convergence-author-to-impl.md`. Epiphany `E-SOA-IS-THE-ONLY` (+ §11.3/4/6 refinements).

| D-id | Title | Crate(s) | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-MBX-A1 | migrated thoughtspace columns landed on `MailboxSoA<N>` (`edges`/`qualia`/`meta`/`entity_type`) | cognitive-shader-driver | 60 | LOW | **Shipped** | between #418 and #433 (verified `mailbox_soa.rs` 2026-05-29) |
| D-MBX-A2 | close BindSpace expressivity gaps in `MailboxSoA<N>` (`content_ref`, S/P/O role slices, temporal/expert fold per OQ-2) | cognitive-shader-driver + contract | 140 | MED | **Queued** | gates on D-CE64-MB-1-impl + OQ-1/OQ-2 |
| D-MBX-A3 | `witness_arc: [u32; W]` per-row column (the belief-state arc handle into AriGraph episodic Markov chain) | cognitive-shader-driver | 100 | MED | **Queued** | gates on D-MBX-A2 + OQ-11.2 |
| D-MBX-A4 | Staunen × Wisdom counterfactual plasticity spreader (Hebbian, hot-path-only, Planning-gated) | cognitive-shader-driver | 80 | LOW | **Queued — design** | gates on D-MBX-A3 + OQ-11.1 + `phase` field |
| D-MBX-A5 | SPO-W witness pointer dual-residency (SoA / kanban / mailbox index); SoA decides commit modality (chain pointer vs cold fact) | cognitive-shader-driver + AriGraph SPO-G | 150 | HIGH | **Queued** | gates on D-MBX-A3 + D-MBX-4 |
| D-MBX-A6 | `lance-graph-planner` DTO surface overhaul: DTOs as SoA-row-lenses; planner output = `KanbanMove`s; 5-phase feature-gated cutover (OQ-11.7) | lance-graph-planner + contract | 600 | HIGH | **Queued** | gates on D-MBX-10 + D-MBX-8 + OQ-11.7 |
| D-MBX-7 | `lance-graph` containers ≡ `MailboxSoA` layout ≡ `ndarray::simd_soa.rs`-aligned (1.4–4.2× SIMD payoff; hard prereq for SurrealDB transparent view) | lance-graph + ndarray | 300 | HIGH | **Queued** | gates on D-MBX-A2 + D-MBX-10 + D-MBX-11 + PR-NDARRAY-MIRI-COMPLETE |
| D-MBX-8 | Σ10 commit stamps **t = −550 ms** wall-clock (Libet anchor) in `SigmaTierRouter`; downstream ractor START fires | sigma-tier-router + shader-driver | 120 | MED | **Queued** | gates on D-MBX-A4 + D-MBX-A6 Phase 1 |
| D-MBX-9 | Rubicon kanban view in `surrealkv`-on-lance (4 columns: Planning · Cognitive work · Evaluation · Commit·Plan·Prune); ractor lifecycle hooks = kanban moves | surreal_container + ractor | 250 | HIGH | **Queued** | gates on D-MBX-7 + D-MBX-8 + surreal_container BLOCKED(B/C/D) resolved (OQ-11.6) + D-PERSONA-5 |
| D-MBX-10 | SoA version byte at layout root (`MailboxSoAHeader`); refuse v(N>M) bytes on v(M) reader; field-isolation matrix tests on every column op (`I-LEGACY-API-FEATURE-GATED` discipline) | lance-graph-contract | 100 | HIGH | **Queued** | foundation — should land early in P2; gates on OQ-11.5 |
| D-MBX-11 | Lance `=6.0.0 → =6.0.1` patch bump (5 Cargo.toml files identified) | workspace Cargo.toml | 10 | LOW | **Queued (mechanical)** | none — can land in parallel with par-tile prereq |
| D-MBX-12 | 8-PR workspace-wide consumer alignment: 12.1 AriGraph · 12.2 Vsa16k audit · 12.4 lance-graph · 12.5 planner · 12.6 shader-driver · 12.7 callcenter · 12.8 ontology audit · 12.9 thinking-styles | per-crate | 800 | per-PR | **Queued (multi-PR)** | sequencing per OQ-11.8: 12.4 → 12.5 → 12.6 → 12.7 → 12.1 → 12.9 → 12.2 → 12.8 |
| D-MBX-A6-P1 | contract slice of D-MBX-A6: `kanban::{KanbanColumn, KanbanMove}` + `soa_view::{MailboxSoaView, MailboxSoaOwner}` + `StepDomain::Kanban` — the planner⟷ractor⟷surreal seam, zero-dep, no parallel DTO family | lance-graph-contract | 340 | HIGH | **In PR** | extends §8.4; +6 tests; downstream cargo-check clean; consumer impls (planner emit / `MailboxSoA` owner-impl / ractor arm / surreal read-view) deferred |

---

## bindspace-singleton-to-mailbox-soa-v1 — dissolve `Arc<BindSpace>` into per-mailbox `MailboxSoA<N>`

Plan path: `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md`. Epiphany `E-MAILBOX-IS-BINDSPACE`. Migration of the shared singleton address space into mailbox-owned ephemeral thoughtspace (LE-contract SoA columns); drops the 64 KB `Vsa16kF32` `cycle` plane.

| D-id | Title | Crate(s) | ~LOC | Risk | Status | PR / Evidence |
|---|---|---|---|---|---|---|
| D-MBX-1 | add migrated columns (`edges`/`qualia`/`meta`/`entity_type`) to `MailboxSoA<N>` behind `mailbox-thoughtspace` feature | cognitive-shader-driver | 120 | MED | **Queued** | gated on D-CE64-MB-1-impl + PR-NDARRAY-MIRI-COMPLETE |
| D-MBX-2 | move `engine_bridge` per-row read/write surface onto mailbox rows; `cycle` plane becomes a transient local | cognitive-shader-driver | 180 | MED | **Queued** | blocked on D-MBX-1 + OQ-1 (content-ref shape) |
| D-MBX-3 | `ShaderDriver` holds a sea-star of mailboxes; kill the `BindSpace::zeros(4096)` singleton in `serve.rs` | cognitive-shader-driver | 160 | HIGH | **Queued** | blocked on D-MBX-2 + OQ-2 (temporal/expert fold) |
| D-MBX-4 | death → SPO-G quad + Lance tombstone-witness (link-integrity back-pointer) | cognitive-shader-driver + Lance | 200 | HIGH | **Queued** | blocked on D-MBX-3 + Zone-2 persistence |
| D-MBX-5 | delete `BindSpace` singleton + `Vsa16kF32` `cycle` plane; remove feature gate | cognitive-shader-driver | 80 | MED | **Queued** | blocked on D-MBX-4 + OQ-4 (CLAUDE.md "The Click" doctrinal update) |
| D-MBX-6 | `ThoughtStruct` = transparent hot/cold view over SurrealDB container table(s) (same SoA both tiers; ~64k–256k hot ceiling, ~6 KB/thought) | cognitive-shader-driver + surreal_container | 220 | HIGH | **Queued** | blocked on D-MBX-3 + surreal_container unblock (BLOCKED A/B/C/D) or callcenter Zone-2 |
| TD-RESONANCEDTO-DUP-1 | dedup the two `ResonanceDto` (thinking-engine) | thinking-engine | 60 | LOW | **Deferred** | user 2026-05-27 — fold into D-MBX-2 |

---

## odoo-savant-reasoners-v2 — reshape: `Reasoner` trait → typed composition over `CausalEdge64` + `Tactic` + `callcenter/role_keys`

Reshape of v1 (shipped PR #420). v1's `Reasoner` trait surface fails CLAUDE.md "P-1 The Click" + "P0 AGI-as-glove" litmus tests; v2 routes the canonical path through the agnostic substrate that already exists (CausalEdge64 + Tactic + 33-TSV atoms + role-key catalogues). v1 stays under `legacy-reasoner` feature with `#[deprecated]` until woa-rs migrates. Plan path: `.claude/plans/odoo-savant-reasoners-v2.md`. Driver epiphany: `E-SAVANT-COMPOSITION-1`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-ODOO-SAV-5a | `SavantPattern` + `TacticInvocation` + `EdgeEmissionSpec` + `AtomTouchMask` primitives (Group D, zero-dep, in contract) | lance-graph-contract | 200 | HIGH | **Queued** | additive — ships with this plan + INTEGRATION_PLANS prepend + this STATUS_BOARD section + EPIPHANIES entry (board hygiene) |
| D-ODOO-SAV-5b | `callcenter/role_keys.rs` with 25 disjoint Vsa16kF32 slices + lookup-by-enum + slice-allocation manifest (Group E) | lance-graph-callcenter | 250 | HIGH | **Queued** | parallel with 5a — independent; coordinate disjoint slice range vs `grammar/role_keys.rs` |
| D-ODOO-SAV-5c | 25 `SavantPattern` consts drawn from `.claude/odoo/savants/<N>.md` slot 1/4 + `.claude/odoo/L*.md` business semantics (Group F) | lance-graph-callcenter | 600 | MED | **Queued** | blocked on 5a + 5b; likely one D-id per savant in a Wave if translation is large; 14 NEEDS-INPUT savants ship pattern + emission spec only |
| D-ODOO-SAV-5d | `#[deprecated]` + `legacy-reasoner` feature gate + migration pointers on v1 `Reasoner` trait + 4 `*Reasoner` impls + `SavantConclusion` + `SavantSuggestion` + `build_conclusion` (Group G) | lance-graph-contract + lance-graph-callcenter | 120 | HIGH | **Queued** | blocked on 5c (so the migration pointer names a real target); removal in a follow-up after woa-rs migrates |
| D-ODOO-SAV-5e | End-to-end test: FiscalPositionResolver `SavantPattern` over a synthetic ontology fixture → expected `CausalEdge64` row (SPO + NARS truth + v2 signed mantissa); the proof the reshape works | lance-graph-callcenter tests | 150 | MED | **Queued** | ships with 5c completion as the round-trip proof; uses `CausalEdge64::pack_v2` per `I-LEGACY-API-FEATURE-GATED` |

---

## odoo-business-logic-blueprint-v1 — typed Odoo entity DTOs as the substrate for OGIT → OWL → DOLCE → FIBU/FIBO normalization + JITson / recipe codegen

PREREQUISITE for `odoo-savant-reasoners-v2` Group F (per `E-SAVANT-COMPOSITION-1`). Establishes the typed `OdooEntity` + sub-types layer that the inheritance chain operates on — replaces today's ad-hoc string-keyed maps against `model_name`. Both passes (L-docs first as curated filter, Odoo source extraction second as exhaustive backing). All 15 lanes (L1–L15). Plan path: `.claude/plans/odoo-business-logic-blueprint-v1.md`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-ODOO-BP-1a | `OdooEntity` + sub-types (`OdooField`/`OdooMethod`/`OdooDecorator`/`OdooStateMachine`/`OdooConstraint`/`OdooProvenance`) — zero-dep, const-only, no serde | lance-graph-ontology | 300 | HIGH | **Queued** | ships with plan + INTEGRATION_PLANS prepend + this STATUS_BOARD section (board hygiene); additive — zero churn to existing call sites |
| D-ODOO-BP-1b | L-doc projection: one `OdooEntity` const per entity, 15 lanes, per-lane module `odoo_blueprint::l{1..15}`, provenance=Curated with line-range citations | lance-graph-ontology | 2500 | HIGH | **Queued** | blocked on 1a; ships in Waves (L1-L5, L6-L10, L11-L15), one subagent per lane (Sonnet, mechanical prose→const projection); ~5 entities/lane average × 15 lanes ≈ 75-200 consts |
| D-ODOO-BP-1c | Wire OGIT classifier to take `&OdooEntity` (replaces string-keyed `resolve_odoo`); uses field/method semantics for richer dispatch; covers 0x63/0x90 from PR #414 | lance-graph-ontology + lance-graph-callcenter::family_table | 250 | HIGH | **Queued** | blocked on 1b; parallel with 1d/1e |
| D-ODOO-BP-1d | Wire OWL hydrator to take `&OdooEntity`: relational fields → edges, computed fields → SHACL-equivalent constraints, decorators → axioms | lance-graph-ontology | 350 | MED | **Queued** | blocked on 1b; parallel with 1c/1e |
| D-ODOO-BP-1e | Wire DOLCE classifier + FIBU/FIBO alignment to take `&OdooEntity`; closes D-ODOO-SAV-2's `None`-class alignment for stock.* / analytic.distribution.model / account.account.tag over typed input | lance-graph-ontology | 200 | HIGH | **Queued** | blocked on 1b; parallel with 1c/1d |
| D-ODOO-BP-1f | Odoo source extraction tool: tree-sitter Python AST → candidate `OdooEntity` consts with Confidence=Extracted; validates + extends 1b's curated set | tools/odoo-blueprint-extractor/ | 800 | MED | **Queued** | blocked on 1b/c/d/e; conflicts (curated vs extracted) flag for ratification, default to curated |
| D-ODOO-BP-1g | Wire JITson → recipes: `jit::JitCompiler` compiles `Tactic` kernels parameterized by `(&OdooEntity, AtomTouchMask)`; produces DTO-ish NARS that lands in shader-driver | lance-graph-contract::jit + thinking-engine | 400 | MED | **Queued** | blocked on 1c/d/e; proof-of-concept on FiscalPositionResolver, the rest follow in `odoo-savant-reasoners-v2` Group F |
| D-ODOO-STYLE-1 | `style_recipe.rs` — Phase 1 D-Atom interpretation step: typed Odoo SoA → `OdooStyleRecipe` cognitive fingerprints (12 DAtom basis, 7-rule cascade, FNV-1a recipe_id, never stored back as triples) | lance-graph-ontology::odoo_blueprint | 746 | HIGH | **Shipped** | commit `feb8be54` (PR #433 merged); 13/13 tests; DAtom::ALL discriminant-order pinned; OdooStyleRecipe != contract::recipe::StyleRecipe (documented) |
| D-ODOO-OP-1 | `op_emitter.rs` — Phase 2 bucket-dispatch codegen: `bucket_corpus` groups OdooStyleRecipe by OdooMethodKind; `emit_op_dispatch` emits compilable Rust (RECIPE_* consts + per-kind Op structs + static Op slices); deterministic, recipe_id dedup collapses identical DAtom profiles | lance-graph-ontology::odoo_blueprint | 400 | HIGH | **Shipped** | commit `63f3e2ca`; 12/12 tests; zero-dep emitted output; 230/230 existing tests green |

---

## streaming-arm-nars-discovery-v1 — upstream proposer leg into the SPO substrate (20K-200K rows/window pair-stats + optional Aerial+ → NARS-truth → SpoStore hypothesis test → council ratification → op_emitter codegen)

The missing UPSTREAM discovery leg. Today's proposers (curated L-docs + AST-extracted Odoo source) are bounded by the literal artifact; this plan adds runtime-tabular-data ARM discovery, gated through the epiphany-brainstorm-council before reaching the deterministic codegen path. Plan: `.claude/plans/streaming-arm-nars-discovery-v1.md`. Handover: `.claude/handovers/2026-05-29-2030-arm-discovery-author-to-impl.md`.

| D-id | Title | Crate | Lines | Conf | Status | Notes |
|---|---|---|---|---|---|---|
| D-ARM-1 | `ProvenanceTier::{Curated,Extracted,ArmDiscovered,Ratified,Conjecture}` enum + ordering | lance-graph-contract | 50 | HIGH | **Queued** | blocks all other D-ARM-*; additive |
| D-ARM-2 | `Proposer` trait + `CandidateRule` carrier + `WindowMetadata` | lance-graph-contract | 100 | HIGH | **Queued** | blocks D-ARM-3, D-ARM-9. D-ARM-13 shipped **local mirrors** (`rule::{CandidateRule, Proposer, Item}`) ahead of this — see **TD-ARM-CARRIER-FORK**: re-export via `pub use` when this lands (firewall allows path-dep on zero-dep contract). Field set diverges — local carries bare `n: u32`, this plans `WindowMetadata`; reconcile (recommend `n: u32`) so the shape matches. |
| D-ARM-3 | Pair-stats proposer (default trunk, deterministic, k² pair counters per window) | lance-graph-arm-discovery::proposer::pair_stats | 400 | HIGH | **Queued** | depends on D-ARM-1/2/7; blocks D-ARM-12 |
| D-ARM-4 | ARM-truth → NARS-truth translator + Odoo `FeedProjector` impl | lance-graph-arm-discovery::translator | 200 | HIGH | **Partially shipped (branch)** | The translator substance landed early inside D-ARM-13: `translator::{arm_to_nars, NarsTruth, CandidateTriple, FeedProjector}` (verbatim paper §2/§3.3 mapping, 35/35 tests). REMAINING: the real Odoo `FeedProjector` (currently a `DebugProjector` stub emitting `implies`) + contract homing on D-ARM-1/2. Depends on D-ARM-1/2. |
| D-ARM-5 | Hypothesis test: SpoStore round-trip, NARS revision, contradiction commit per The Click | lance-graph-arm-discovery::hypothesis | 350 | MED | **Queued** | depends on D-ARM-4; verifies `spo::truth::Contradiction` primitive exists |
| D-ARM-6 | `RatificationQueue` ring buffer + corrections-to-#434 spec PR (`discovery_arc D=8`, `discovery_origin u8`) | lance-graph-arm-discovery::queue + #434 spec follow-up | 200 + spec | MED | **Queued** | depends on PR #434 D-MBX-A3 landing |
| D-ARM-7 | Jirak-2016 weak-dependence significance thresholds (mandatory Stage A floor) | lance-graph-arm-discovery::jirak | 150 | HIGH | **Queued — HARD PREREQUISITE** | blocks D-ARM-3; cites I-NOISE-FLOOR-JIRAK. **ISSUE ARM-JIRAK-FLOOR (2026-05-30, 3-savant review):** D-ARM-13 ships the Aerial proposer with NO Jirak floor (classical `min_support`/`min_confidence` only). MUST land before D-ARM-5 wires the proposer to a live `SpoStore`, else the substrate calcifies on thin-but-frequent noise (plan §11.1). **ENGINE EXISTS:** `jc::jirak` (Jirak-Cartan Pillar 5) is the weak-dependence Berry-Esseen rate (`n^(p/2-1)`); this deliverable is the *gate function* (rule → significant?) that derives its threshold from it — NOT a from-scratch Jirak impl. See E-ARM-JC-RESOLVES-BOTH-SEAMS + `splat-codebook-aerial-wikidata-compression.md`. |
| D-ARM-8 | `Feed` + `FeedProjector` + window-size config + Odoo `account.move` projector example | lance-graph-arm-discovery::feed | 250 | MED | **Queued** | depends on D-ARM-2 |
| D-ARM-9 | Aerial+ IPC client (feature-gated `arm-aerial`, NDJSON over Unix socket) | lance-graph-arm-discovery::proposer::aerial_ipc | 200 | MED | **Superseded by D-ARM-13** | The native in-process Aerial+ transcode (D-ARM-13, branch `claude/jolly-cori-clnf9`) replaces the need for the Python IPC client. The determinism-boundary rationale the IPC was designed for (keep the nondeterministic autoencoder out of the Rust path) is now met in-process via seed (`aerial::Rng`) + `aerial` feature gate + workspace `exclude`. Keep this row ONLY if a Python-only Aerial variant is later required; otherwise close as Abandoned-by-replacement. |
| D-ARM-10 | `op_emitter::bucket_corpus` ratification filter (`confidence ≥ Ratified`) + 2 tests | lance-graph-ontology::op_emitter | 30 | HIGH | **Queued** | depends on D-ARM-1 |
| D-ARM-11 | `style_recipe.rs` rule 8 — ArmDiscovered backing adds `DAtom::Compute` weight 2 (provisional) | lance-graph-ontology::style_recipe | 80 | MED | **Queued** | depends on D-ARM-1 |
| D-ARM-12 | End-to-end pipeline test + bench (synthetic Odoo feed → all 5 stages → council micro-batch) | lance-graph-arm-discovery::tests + benches | 400 | MED | **Queued** | depends on Waves 1-6; informs OQ-ARM-2 + OQ-ARM-7 |
| D-ARM-13 | **Aerial+ Rust transcode — deterministic codebook-probe backend** (float-free). The paper's `f32` denoising autoencoder is REPLACED by an integer `CodebookDistance` oracle (palette256 distance, ρ=0.9973 vs cosine): the reconstruction probe is a codebook top-k, not a softmax over float weights. Integer evidence counts + ppm gates + `TruthU8` (= CausalEdge64 wire). `AerialProposer` impl of `Proposer`. Count loop is a row-bitset SoA (`RowMasks`) → AND+popcount, routed through `ndarray::simd::U64x8` under the `ndarray-simd` feature. | lance-graph-arm-discovery::aerial | ~1.1K | HIGH | **Shipped (branch)** | branch `claude/jolly-cori-clnf9`; standalone zero-dep crate (excluded); **33/33** tests + clippy `-D warnings` clean on BOTH default (scalar) and `--features ndarray-simd`; **zero f32 in the discovery path** (audit), float only at the `TruthValue`/`Triple` serialization edge. Bitwise-deterministic ⇒ joins the trunk; the nondeterminism firewall + D-ARM-9 IPC rationale are moot. SIMD target-cpu caveat: real AVX-512/AMX kernels need `-C target-cpu=native`/`x86-64-v4`. v1 (autoencoder) superseded per the user's no-float directive. |
| D-ARM-14 | **Splat-codebook oracle + Wikidata skeleton discovery** — wire the certified jc splat codebook into aerial as the `CodebookDistance` oracle, discover OWL/DOLCE+ SPO HHTL classes + basins, drive the `wikidata-hhtl-load.md` deterministic compression (skeleton + basins + CAM-dedup + thin rows). | lance-graph-arm-discovery::aerial + crates/jc + wikidata loader | ~? | MED | **Queued (architecture)** | CONJECTURE pipeline; seams concrete (`CodebookDistance` ← jc `[u32;dim²]` table built by `ewa_sandwich`+`sigma_codebook_probe` ρ=0.9973+`pflug` Lε; Jirak floor ← `jc::jirak`). Float lives OFFLINE in jc only; aerial online path integer. No new aerial dep (pass the frozen table in). Gated on D-ARM-7. Full map: `splat-codebook-aerial-wikidata-compression.md`; finding E-ARM-JC-RESOLVES-BOTH-SEAMS. |
| D-ARM-SYN-1 | Add `Implies`/`CoOccursWith` to `ruff_spo_triplet::Predicate` closed vocabulary (+ `Provenance` tier) so ARM rules load through the same `parse_triples` ndjson path as the static extractor | ruff/ruff_spo_triplet | 40 | MED | **Queued** | council-gated (deliberate ontology change); blocks SYN-2; see `.claude/knowledge/aerial-arm-ruff-spo-codegen-synergies.md` §1 |
| D-ARM-SYN-2 | `CandidateRule → ruff_spo_triplet::ModelGraph` adapter so the Aerial runtime-data leg joins the `ruff_python_dto_check` static-AST leg in one graph before `expand()` | lance-graph-arm-discovery + ruff_spo_triplet | 120 | MED | **Queued** | depends on SYN-1; synergy doc §2 |
| D-ARM-SYN-3 | Calibrate `ProvenanceTier::ArmDiscovered` `(f,c)` below the `op_emitter` ratification gate + below static `Inferred (0.85,0.75)` so un-ratified ARM truth is council-visible but codegen-filtered | lance-graph-contract + lance-graph-ontology::op_emitter | 30 | MED | **Queued** | depends on D-ARM-1 + SYN-1; synergy doc §3/§4 |

---

## Update protocol

When a deliverable ships:
1. Edit this file's Status column in place for the row → **Shipped**.
2. Fill in PR / Evidence column with the merge commit or PR #.
3. Append a new section to `PR_ARC_INVENTORY.md` (Added / Locked /
   Deferred / Docs / Confidence).
4. Update `LATEST_STATE.md` (Recently Shipped PRs + Current Inventory
   if types change).

When a deliverable moves phase (e.g. Queued → In progress → In PR):
1. Edit Status column in place. Don't reorder rows.
2. If the move reflects scope correction, also update
   `INTEGRATION_PLANS.md` Status line for the parent plan.

When a new deliverable is added to a plan:
1. Append a new row at the bottom of the plan's section.
2. D-id is sequential in the plan (D12, D13, etc.).
3. Original scope becomes immutable once committed.

When a deliverable is abandoned:
1. Edit Status → **Abandoned**. Don't remove the row.
2. Cite the replacement in Notes.
