# ARCHIVE_INDEX — Prior-Art File Status Audit

> **Deliverable of the STATUS_BOARD backlog row "prior-art audit" (102
> existing docs).** One-pass classification of every `.claude/*.md`
> and `.claude/prompts/*.md` file by vicinity to the PR that last
> touched it. **APPEND-ONLY** once committed — if a file is
> reactivated or archived, append a new dated status line under the
> entry; never rewrite the original row.
>
> **Method.** For each file: `git log --follow` → last commit SHA +
> date. SHA mapped to the merge commit (PR #) that brought it into
> main. Status bucket derived from days stale against HEAD =
> 2026-04-19.
>
> **Buckets:**
> - **Active** — touched ≤ 1 day ago (current PR era, #211).
> - **Recent** — touched 2–7 days ago (PR ~200–210 era).
> - **Dormant** — touched 8–14 days ago (PR ~185–200 era).
> - **Stale** — touched 15–30 days ago (PR ~130–185 era).
> - **Archival** — touched > 30 days ago.
>
> **Caveat.** Many files' last touch is a mechanical rename
> (`Blumenstrauß → CognitiveShader`, PR #205) or a bulk docs sweep,
> not a content update. The `first_date` column reveals true
> content-origin PR; `last_pr` reveals the last mechanical touch.
> A file whose first_date is March 2026 but last_date is April
> 2026 is typically a historical artifact that happened to get
> swept by a rename commit.
>
> **Next action per bucket:**
> - Active / Recent → keep in place, referenced from BOOT / CLAUDE / knowledge.
> - Dormant → review if still referenced; if not, move to `.claude/archive/`.
> - Stale / Archival → default candidates for `.claude/archive/` move.
>   **Not in this PR** — the move is a follow-up deliverable.

---


## Active (≤ 1 day, PR #211 era)

_8 files_

| Path | First | Last | Last PR | Days stale |
|---|---|---|---|---|
| `.claude/ARCHITECTURE_THOUGHT_ENGINE.md` | 2026-04-03 | 2026-04-18 | #200 | 1 |
| `.claude/CROSS_REPO_AUDIT_2026_04_01.md` | 2026-04-02 | 2026-04-18 | #200 | 1 |
| `.claude/DISTANCE_METRIC_INVENTORY.md` | 2026-04-03 | 2026-04-18 | #200 | 1 |
| `.claude/INTEGRATIONSPLAN_2026_04_01.md` | 2026-04-02 | 2026-04-18 | #200 | 1 |
| `.claude/blackboard-ripple-architecture-changelog.md` | 2026-04-02 | 2026-04-18 | #200 | 1 |
| `.claude/knowledge.md` | 2026-04-02 | 2026-04-18 | #200 | 1 |
| `.claude/prompt-for-other-session-additive.md` | 2026-04-02 | 2026-04-18 | #200 | 1 |
| `.claude/ripple-file-index.md` | 2026-04-02 | 2026-04-18 | #200 | 1 |

## Recent (2–7 days, PR #~200–210 era)

_16 files_

| Path | First | Last | Last PR | Days stale |
|---|---|---|---|---|
| `.claude/prompts/archetype-codebook-probe.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/arxiv.md` | 2026-04-12 | 2026-04-12 | #158 | 7 |
| `.claude/prompts/certify-hhtld-quality.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/00-master-plan.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/02-hhtl-d-integration.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/03-shared-palette.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/04-ndarray-simd.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/05-encoder.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/06-decoder.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/07-certifier.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/08-docs.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/09-bake.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/10-release.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/11-tts-inference.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/fisher-z-wiring/12-validate.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |
| `.claude/prompts/matryoshka-wav-test.md` | 2026-04-14 | 2026-04-14 | #175 | 5 |

## Dormant (8–14 days, PR #~185–200 era)

_25 files_

| Path | First | Last | Last PR | Days stale |
|---|---|---|---|---|
| `.claude/AGI_DESIGN.md` | 2026-04-07 | 2026-04-07 | #152 | 12 |
| `.claude/BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md` | 2026-04-11 | 2026-04-11 | #155 | 8 |
| `.claude/CALIBRATION_STATUS_GROUND_TRUTH.md` | 2026-04-06 | 2026-04-06 | #138 | 13 |
| `.claude/CODING_PRACTICES.md` | 2026-04-05 | 2026-04-05 | #116 | 14 |
| `.claude/DEVELOPMENT_STAGES.md` | 2026-04-06 | 2026-04-06 | #146 | 13 |
| `.claude/HANDOVER_CALIBRATION_SESSION.md` | 2026-04-05 | 2026-04-05 | #113 | 14 |
| `.claude/HANDOVER_MAVERICK_SESSION.md` | 2026-04-05 | 2026-04-05 | #113 | 14 |
| `.claude/HANDOVER_NEXT_SESSION.md` | 2026-04-06 | 2026-04-06 | #138 | 13 |
| `.claude/HANDOVER_SIGNED_SESSION.md` | 2026-04-06 | 2026-04-06 | #138 | 13 |
| `.claude/INVARIANT_MATRIX_RESEARCH.md` | 2026-04-11 | 2026-04-11 | #155 | 8 |
| `.claude/KNOWLEDGE_SYNC_SIGNED_SESSION.md` | 2026-04-05 | 2026-04-05 | #118 | 14 |
| `.claude/LANE_AKKUMULATOR.md` | 2026-04-07 | 2026-04-07 | #151 | 12 |
| `.claude/ONE_FORTIETH_SIGMA_LENS.md` | 2026-04-11 | 2026-04-11 | #155 | 8 |
| `.claude/PLAN_BF16_DISTANCE_TABLES.md` | 2026-04-06 | 2026-04-06 | #122 | 13 |
| `.claude/RING_PERTURBATION_PROPAGATION.md` | 2026-04-11 | 2026-04-11 | #155 | 8 |
| `.claude/RISC_THOUGHT_ENGINE_AGI_ROADMAP.md` | 2026-04-06 | 2026-04-06 | #138 | 13 |
| `.claude/STATUSMATRIX.md` | 2026-04-07 | 2026-04-07 | #150 | 12 |
| `.claude/TECHNICAL_DEBT_SIGNED_SESSION.md` | 2026-04-06 | 2026-04-06 | #122 | 13 |
| `.claude/WIKIDATA_EXTRACTION_PLAN.md` | 2026-04-06 | 2026-04-06 | #146 | 13 |
| `.claude/agent2agent-orchestrator-prompt.md` | 2026-04-02 | 2026-04-11 | #156 | 8 |
| `.claude/probe_m1_result_2026_04_11.md` | 2026-04-11 | 2026-04-11 | #156 | 8 |
| `.claude/prompts/SESSION_BGZ_TENSOR_HYDRATE.md` | 2026-03-30 | 2026-04-11 | #156 | 8 |
| `.claude/prompts/session_B_v3_bgz17_container_semiring.md` | 2026-03-21 | 2026-04-08 | #153 | 11 |
| `.claude/prompts/session_ontology_layer_audit.md` | 2026-04-08 | 2026-04-08 | #154 | 11 |
| `.claude/session_2026_04_11_bf16_hhtl_combined_research.md` | 2026-04-11 | 2026-04-11 | #155 | 8 |

## Stale (15–30 days, PR #~130–185 era)

_63 files_

| Path | First | Last | Last PR | Days stale |
|---|---|---|---|---|
| `.claude/BELICHTUNGSMESSER.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/BF16_SEMIRING_EPIPHANIES.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/CALIBRATION_REPORT_2026_04_03.md` | 2026-04-03 | 2026-04-03 | #80 | 16 |
| `.claude/DEEPNSM_CAM_REFERENCE.md` | 2026-03-25 | 2026-03-25 | #47 | 25 |
| `.claude/DEEP_ADJACENT_EXPLORATION.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/FALKORDB_ANALYSIS.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/FINAL_STACK.md` | 2026-03-21 | 2026-03-22 | #34 | 28 |
| `.claude/FIX_BLASGRAPH_SPO.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/GPU_CPU_SPLIT_ARCHITECTURE.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/INTEGRATION_SESSIONS.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/INVENTORY_MAP.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/LANGGRAPH_CRATE_STRUCTURE.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/LANGGRAPH_FULL_INVENTORY.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/LANGGRAPH_OUR_ADDITIONS.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/LANGGRAPH_PARITY_CHECKLIST.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/LANGGRAPH_TRANSCODING_MAP.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/OVERLOOKED_THREADS.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/RESEARCH_REFERENCE.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/RESEARCH_THREADS.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/SCOPE_E_FINDINGS.md` | 2026-03-23 | 2026-03-23 | #36 | 27 |
| `.claude/SESSION_2026_03_25_CROSS_REFERENCE.md` | 2026-03-25 | 2026-03-25 | #47 | 25 |
| `.claude/SESSION_B_HDR_RENAME.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/SESSION_D_LENS_CORRECTION.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/SESSION_FALKORDB_CROSSCHECK.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/SESSION_J_PACKED_DATABASE.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/SESSION_LANCE_ECOSYSTEM_INVENTORY.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/SESSION_LANGGRAPH_ORCHESTRATION.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/UNIFIED_HDR_RENAME_AND_CROSSPOLINATE.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/VISION_ORCHESTRATED_THINKING.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/blackboard.md` | 2026-03-23 | 2026-03-23 | #37 | 27 |
| `.claude/prompts/CAM_PQ_SPEC.md` | 2026-03-24 | 2026-03-24 | #37 | 26 |
| `.claude/prompts/EPIPHANIES_COMPRESSED.md` | 2026-03-29 | 2026-03-29 | #63 | 21 |
| `.claude/prompts/FINAL_MAP.md` | 2026-03-29 | 2026-03-29 | #64 | 21 |
| `.claude/prompts/P18_INTERNAL_LLM_LANGUAGE_SURFACE.md` | 2026-03-29 | 2026-03-29 | #66 | 21 |
| `.claude/prompts/SCOPED_PROMPTS.md` | 2026-03-23 | 2026-03-23 | #37 | 27 |
| `.claude/prompts/SESSION_CAPSTONE.md` | 2026-03-29 | 2026-03-29 | #64 | 21 |
| `.claude/prompts/VERIFY_COMPRESSION_REVOLUTION.md` | 2026-03-30 | 2026-03-29 | #67 | 21 |
| `.claude/prompts/audio_codec_meta_codex.md` | 2026-03-29 | 2026-03-29 | #62 | 21 |
| `.claude/prompts/audio_session1_opus_celt.md` | 2026-03-29 | 2026-03-29 | #62 | 21 |
| `.claude/prompts/audio_session2_whisper_phoneme.md` | 2026-03-29 | 2026-03-29 | #62 | 21 |
| `.claude/prompts/audio_session3_bark_voice.md` | 2026-03-29 | 2026-03-29 | #62 | 21 |
| `.claude/prompts/research_quantized_graph_algebra.md` | 2026-03-22 | 2026-03-22 | #37 | 28 |
| `.claude/prompts/session_6d_neuronprint_handover.md` | 2026-03-31 | 2026-03-31 | #76 | 19 |
| `.claude/prompts/session_A_v3_blasgraph_csc_planner.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/prompts/session_C_v3_ndarray_bgz17_dualpath.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/prompts/session_D_v3_falkordb_retrofit.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/prompts/session_MASTER_map_v3.md` | 2026-03-21 | 2026-03-22 | #29 | 28 |
| `.claude/prompts/session_arigraph_transcode.md` | 2026-03-25 | 2026-03-25 | #47 | 25 |
| `.claude/prompts/session_bgz17_similarity.md` | 2026-03-22 | 2026-03-22 | #37 | 28 |
| `.claude/prompts/session_bgz_tensor.md` | 2026-03-28 | 2026-03-28 | #48 | 22 |
| `.claude/prompts/session_deepnsm_cam.md` | 2026-03-25 | 2026-03-25 | #47 | 25 |
| `.claude/prompts/session_deepnsm_compile.md` | 2026-03-27 | 2026-03-27 | #47 | 23 |
| `.claude/prompts/session_epiphany_integration.md` | 2026-03-28 | 2026-03-28 | #58 | 22 |
| `.claude/prompts/session_integration_plan.md` | 2026-03-28 | 2026-03-28 | #58 | 22 |
| `.claude/prompts/session_master_integration.md` | 2026-03-28 | 2026-03-28 | #58 | 22 |
| `.claude/prompts/session_model_integration_plans.md` | 2026-03-29 | 2026-03-29 | #65 | 21 |
| `.claude/prompts/session_ndarray_migration_inventory.md` | 2026-03-22 | 2026-03-22 | #29 | 28 |
| `.claude/prompts/session_simd_rewire.md` | 2026-04-03 | 2026-04-03 | #84 | 16 |
| `.claude/prompts/session_simd_surgery.md` | 2026-03-22 | 2026-03-22 | #37 | 28 |
| `.claude/prompts/session_tensor_codec_vision.md` | 2026-03-29 | 2026-03-29 | #63 | 21 |
| `.claude/prompts/session_thinking_topology.md` | 2026-03-25 | 2026-03-25 | #47 | 25 |
| `.claude/prompts/session_unified_26_epiphanies.md` | 2026-03-28 | 2026-03-29 | #62 | 21 |
| `.claude/prompts/session_unified_vector_search.md` | 2026-03-22 | 2026-03-22 | #37 | 28 |

## Archival (> 30 days)

_0 files_

| Path | First | Last | Last PR | Days stale |
|---|---|---|---|---|

---

## How to use this file

**A new session looking for prior art** — scan the **Active** and
**Recent** sections first; those are the docs that describe the
current architecture. Treat Dormant as "cite for context but
verify against LATEST_STATE". Treat Stale / Archival as "historical
reasoning trail — read for the arc, not for current truth."

**When a file is reactivated** — append a dated line under its row:
`**Reactivated YYYY-MM-DD (PR #N):** <reason>`. The original
bucket classification stays; the appended line is the mutable
annotation.

**When a file is archived** — move to `.claude/archive/<year>/<file>`
in a dedicated PR; append a dated line:
`**Archived YYYY-MM-DD (PR #N) →** `.claude/archive/...``

**Regenerate** — `git log --follow` per file → last SHA; map SHA
to merge commit (PR #); bucket by days-stale vs HEAD. Script is
data-driven and deterministic given the current main HEAD.

## Cross-references

- `STATUS_BOARD.md` — flips the "prior-art audit" row from **Backlog**
  to **Shipped** when this file lands.
- `PR_ARC_INVENTORY.md` — the PR that ships this file gets a new entry.
- `LATEST_STATE.md` — unchanged; this is meta-audit, not content change.
