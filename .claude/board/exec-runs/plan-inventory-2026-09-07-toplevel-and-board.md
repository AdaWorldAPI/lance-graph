# Inventory: top-level `.claude/*.md` + board dashboards (lance-graph)

Repo: `/home/user/lance-graph`, checked out on `main` at `aeebfb23`. All counts below are
mechanically derived (git log / grep / python parse) except where marked otherwise.

---

## Top-level docs table

71 files at `.claude/*.md` (top level only). Columns: filename | last commit date | role
(one line, from head-of-file) | self-referential marker (⊘/SUPERSEDED/DEPRECATED/RETIRED/
historical, applied to the FILE ITSELF — not incidental text matches) | cross-ref count
(files under `.claude/` that mention this filename, excluding itself).

| File | Last commit | Role | Marker | Refs |
|---|---|---|---|---|
| AGENT_COORDINATION.md | 2026-04-25 | Governance doc: the 3 coordination layers agents use (git log / PR descriptions / EPIPHANIES.md / CROSS_SESSION_BROADCAST.md) | — | 1 |
| AGI_DESIGN.md | 2026-04-07 | AGI design note: 4D×16Kbit cluster resonance + neural meta-learning (ONNX micro-learner correcting u8 lookup) | — | 1 |
| ARCHITECTURE_THOUGHT_ENGINE.md | 2026-04-18 | Architecture: "16M RISC Thought Engine" (MatVec/cycle, 3-layer 64/256/4096 branching) | — | 2 |
| ATTENTION_MASK_AUDIT_2026_08_21.md | 2026-08-21 | D-ACR-0 audit report: `attention_mask.rs` graded EXISTS-UNCALLED for the eye-tracking use case | — | 6 |
| BELICHTUNGSMESSER.md | 2026-03-14 | Explainer: "HDR Popcount-Stacking Early-Exit Distance Cascade" (exposure-meter early-exit search) | — | 3 |
| BF16_SEMIRING_EPIPHANIES.md | 2026-03-15 | Research-report epiphanies: BF16 + semirings + binary planes (5:2 bitwise/float split) | — | 4 |
| BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md | 2026-04-11 | Research note: why bgz17 uses 11/17 golden-step constants, read from source | — | 2 |
| BOOT.md | 2026-07-22 | THE session entry point / mandatory read-order doc (canonical bootstrap, named in root CLAUDE.md) | — | 62 |
| CALIBRATION_REPORT_2026_04_03.md | 2026-04-03 | Calibration report: bgz17→f32 bridge + real-data validation (OpenChat/Llama4/Jina v3) | — | 0 |
| CALIBRATION_STATUS_GROUND_TRUTH.md | 2026-04-06 | "Ground truth" correction doc, meant to be read before SESSION_*.md prompts (**named in root CLAUDE.md Knowledge Base**) | — | 5 |
| CODING_PRACTICES.md | 2026-04-21 | Coding-pattern checklist from EmbedAnything (**named in root CLAUDE.md Knowledge Base**) | — | 3 |
| CROSS_REPO_AUDIT_2026_04_01.md | 2026-04-18 | Cross-repo audit (lance-graph/ndarray/q2/AriGraph), 8 parallel agents | (false-positive: "DEPRECATED" is a legend-table cell, not a self-marker) | 3 |
| DECISION_SPO_ARIGRAPH.md | 2026-05-07 | Binding decision doc: SPO-1 crate composition, Option B federated two-layer cache | — | 8 |
| DEEPNSM_CAM_REFERENCE.md | 2026-03-25 | Quick reference for DeepNSM-CAM semantic-lookup layer (data file locations) | — | 3 |
| DEEP_ADJACENT_EXPLORATION.md | 2026-03-15 | Notes on dropped algorithms / RISC design / adjacent research (RDF-3X etc.) | — | 3 |
| DEVELOPMENT_STAGES.md | 2026-04-06 | RISC Thought Engine dev-stage tracker with empirical results | — | 1 |
| DISTANCE_METRIC_INVENTORY.md | 2026-04-18 | Table: valid distance metric per data type across the stack | — | 3 |
| FALKORDB_ANALYSIS.md | 2026-03-16 | Competitive analysis of FalkorDB informing lance-graph design | — | 3 |
| FINAL_STACK.md | 2026-03-22 | Session summary naming the "final" 3-repo stack (ndarray/lance-graph/rs-graph-llm) | — | 3 |
| FIX_BLASGRAPH_SPO.md | 2026-03-13 | Proposal: unify BlasGraph BitVec + SPO Fingerprint into one type | — | 3 |
| GPU_CPU_SPLIT_ARCHITECTURE.md | 2026-03-15 | Architectural epiphany: GPU tensor cores for NARS truth revision, not cosine search | — | 3 |
| HANDOVER_CALIBRATION_SESSION.md | 2026-04-05 | Session handover: calibration/encoding/Cronbach-alpha validation hypotheses | — | 1 |
| HANDOVER_MAVERICK_SESSION.md | 2026-04-05 | Session handover: Llama4 Maverick 128-expert MoE + temperature fix | — | 1 |
| HANDOVER_NEXT_SESSION.md | 2026-04-06 | Handover: forward-pass branch graph + vision + LoRA (**named in root CLAUDE.md Prior art**) | — | 0 (see note) |
| HANDOVER_SIGNED_SESSION.md | 2026-04-06 | Session-stats handover for the "signed session" (17 PRs, 294 tests) | — | 1 |
| IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md | 2026-04-29 | Idea journal (pillars 7/8/9); ideas re-filed to canonical board | **historical** ("preserved for historical reference") | 4 |
| IDEA_JOURNAL_2026_04_29_STREAMING_HYDRATION.md | 2026-04-29 | Idea journal (streaming-hydration/fractal codec); ideas re-filed to canonical board | **historical** ("preserved for historical reference") | 4 |
| INTEGRATIONSPLAN_2026_04_01.md | 2026-04-18 | Integration plan: "quick wins to AGI stack" (**named in root CLAUDE.md Prior art**) | — | 2 |
| INTEGRATION_SESSIONS.md | 2026-03-15 | Session-prompt index, inventory→wired-system (**named in root CLAUDE.md Prior art**) | — | 3 |
| INVARIANT_MATRIX_RESEARCH.md | 2026-04-11 | Research note: "which invariant does each lens preserve" reframing | — | 4 |
| INVENTORY_MAP.md | 2026-03-15 | Complete inventory of rustynum-core + lance-graph substrate (**named in root CLAUDE.md Prior art**) | — | 4 |
| KNOWLEDGE_SYNC_SIGNED_SESSION.md | 2026-04-05 | Knowledge-sync doc: "33% error" SiLU-gate correction finding (**named in root CLAUDE.md Prior art**) | — | 0 (see note) |
| LANE_AKKUMULATOR.md | 2026-04-07 | L0-L4 multi-lane lookup-prediction accumulator | — | 1 |
| LANGGRAPH_CRATE_STRUCTURE.md | 2026-03-16 | Python LangGraph package layout → recommended Rust crate structure | — | 0 |
| LANGGRAPH_FULL_INVENTORY.md | 2026-03-16 | Full Python LangGraph → Rust API mapping table | — | 0 |
| LANGGRAPH_OUR_ADDITIONS.md | 2026-03-16 | Rust graph-flow features beyond Python LangGraph's set | (false-positive: "historical state" is prose describing a feature) | 0 |
| LANGGRAPH_PARITY_CHECKLIST.md | 2026-03-16 | DONE/PARTIAL/MISSING parity checklist vs Python LangGraph | — | 0 |
| LANGGRAPH_TRANSCODING_MAP.md | 2026-03-16 | Python→Rust type/function transcoding map for LangGraph | — | 0 |
| ONE_FORTIETH_SIGMA_LENS.md | 2026-04-11 | Research note: "1/40 σ lens" as invariant-matrix column | — | 3 |
| OVERLOOKED_THREADS.md | 2026-03-15 | Catalog of glossed-over research threads (tropical attention etc.) | — | 2 |
| PLAN_BF16_DISTANCE_TABLES.md | 2026-04-06 | Plan: replace u8/i8 distance tables with BF16 tables (**named in root CLAUDE.md Knowledge Base**) | — | 1 |
| RECON_ONTOLOGY_CRATE.md | 2026-05-07 | Phase-1 recon: verifying "already shipped" claims before ontology-crate work | — | 6 |
| RESEARCH_REFERENCE.md | 2026-03-15 | Prior-art catalog ("what to steal"), e.g. DreamerV3 binary RL | — | 2 |
| RESEARCH_THREADS.md | 2026-03-15 | Actionable research threads/connections (DreamerV3 proves binary is better) | — | 4 |
| RING_PERTURBATION_PROPAGATION.md | 2026-04-11 | Research note: ring perturbation awareness propagation (angular half of 1/40σ) | — | 3 |
| RISC_THOUGHT_ENGINE_AGI_ROADMAP.md | 2026-04-06 | AGI roadmap + ground-truth results for the RISC Thought Engine | — | 2 |
| SCOPE_E_FINDINGS.md | 2026-03-23 | API-surface inventory for wiring lance-graph+ndarray+rs-graph-llm | — | 2 |
| SESSION_2026_03_25_CROSS_REFERENCE.md | 2026-03-25 | Cross-reference summary of a big session (thinking-styles-as-graph, DeepNSM, CAM-PQ) | — | 1 |
| SESSION_B_HDR_RENAME.md | 2026-03-22 | Completed session prompt: rename LightMeter→hdr::Cascade (title: "✅ DONE") | done/complete (not ⊘/SUPERSEDED-family) | 2 |
| SESSION_D_LENS_CORRECTION.md | 2026-03-14 | Session prompt: gamma+cushion lens correction + boundary fold | — | 2 |
| SESSION_FALKORDB_CROSSCHECK.md | 2026-03-22 | Session prompt superseded by a `.claude/prompts/` file, kept for reference | **SUPERSEDED** (in title) | 2 |
| SESSION_J_PACKED_DATABASE.md | 2026-03-15 | Session prompt: PackedDatabase panel-packing for cascade search | — | 2 |
| SESSION_LANCE_ECOSYSTEM_INVENTORY.md | 2026-03-15 | Session prompt: inventory unused original-Lance-ecosystem capability | — | 3 |
| SESSION_LANGGRAPH_ORCHESTRATION.md | 2026-03-15 | Session prompt: replace Layer-4 chaos with LangGraph-style execution model | — | 2 |
| STATUSMATRIX.md | 2026-04-07 | RISC Thought Engine status matrix (German), empirically-proven results | — | 1 |
| TECHNICAL_DEBT_SIGNED_SESSION.md | 2026-04-06 | Tech-debt review of the signed session (**named in root CLAUDE.md Knowledge Base**) | — | 2 |
| UNIFIED_HDR_RENAME_AND_CROSSPOLINATE.md | 2026-03-14 | Combined rename + cross-pollination plan (rustynum ↔ lance-graph) | (false-positive: "Deprecated wrapper" describes a code fn, not the doc) | 2 |
| VISION_ORCHESTRATED_THINKING.md | 2026-03-15 | Vision: zero-copy orchestrated thinking (LangGraph+Lance+binary planes) | — | 3 |
| WIKIDATA_EXTRACTION_PLAN.md | 2026-04-06 | Plan: streaming Wikidata→AriGraph hydration via SPARQL | — | 0 |
| agent2agent-orchestrator-prompt.md | 2026-04-11 | Orchestrator prompt for a small agent ensemble (ripple-architecture doctrine) | — | 3 |
| blackboard-ripple-architecture-changelog.md | 2026-04-18 | Changelog for the "Ripple Architecture" exploration | — | 5 |
| blackboard.md | 2026-03-23 | "Polyglot Notebook" single-binary architecture doc (marimo/graph-notebook transcode plan) | — | 12 |
| foundry-roadmap.md | 2026-04-29 | DRAFT unified SMB-Office+MedCare Foundry parity roadmap | not-yet-historical (plans to become HISTORICAL after PR-5; currently still DRAFT) | 6 |
| knowledge.md | 2026-04-18 | "Knowledge Spine" doctrine doc for the ripple-architecture cognition-stack framing | — | 6 |
| medcare-foundry-vision.md | 2026-04-30 | Client-facing architectural-vision draft for MedCare Foundry | — | 4 |
| pattern.md | 2026-05-06 | APPEND-ONLY SoA/DTO graph-traversal primer, "read by every session" touching SoA/DTO | — | 17 |
| patterns.md | 2026-05-12 | "READ FIRST" SoA/DTO navigation guide, near-duplicate topic of `pattern.md` | — | 30 |
| probe_m1_result_2026_04_11.md | 2026-04-11 | Probe M1 result: FAIL with k=4 surprise (clustering-method panel) | — | 0 |
| prompt-for-other-session-additive.md | 2026-04-18 | Cross-session prompt: work additively on ripple architecture | — | 0 |
| ripple-file-index.md | 2026-04-18 | File index for a parallel session entering the ripple-architecture work | — | 1 |
| session_2026_04_11_bf16_hhtl_combined_research.md | 2026-04-11 | Non-canonical research index integrating 4 session notes (BF16-HHTL thread) | — | 1 |

Note on the 0-ref "named in CLAUDE.md" rows: `HANDOVER_NEXT_SESSION.md` and
`KNOWLEDGE_SYNC_SIGNED_SESSION.md` have zero cross-references *inside* `.claude/*.md`
files, but the root `CLAUDE.md`'s "Prior art" list names both explicitly by filename —
so they are referenced, just from outside the `.claude/` scope the grep covered.

---

## Orphans

ORPHAN = zero `.claude/`-internal cross-references AND no self-referential marker AND
not named in root `CLAUDE.md`'s "Prior art" / "Knowledge Base" lists (9 files named there;
`SESSION_CAPSTONE.md` is also named but does not exist in this repo — a stale citation in
CLAUDE.md itself, worth flagging).

**9 true orphans:**

1. `CALIBRATION_REPORT_2026_04_03.md`
2. `LANGGRAPH_CRATE_STRUCTURE.md`
3. `LANGGRAPH_FULL_INVENTORY.md`
4. `LANGGRAPH_OUR_ADDITIONS.md`
5. `LANGGRAPH_PARITY_CHECKLIST.md`
6. `LANGGRAPH_TRANSCODING_MAP.md`
7. `WIKIDATA_EXTRACTION_PLAN.md`
8. `probe_m1_result_2026_04_11.md`
9. `prompt-for-other-session-additive.md`

The 5-file `LANGGRAPH_*` cluster is a coherent orphaned unit — a complete Python→Rust
LangGraph parity/mapping doc-set with zero inbound references from any other `.claude/`
doc, and CLAUDE.md's "Prior art" list does not name any of them.

---

## Stale candidates

STALE-CANDIDATE = last commit before 2026-06-01 AND not in root CLAUDE.md's Prior-art/
Knowledge-Base lists. Root CLAUDE.md itself says "61 top-level docs" in its Prior-art
section — the real count is **71**, so that count in CLAUDE.md is itself stale (10 files
added since it was last touched).

**61 files** meet the stale-candidate bar (every top-level doc except `BOOT.md`,
`ATTENTION_MASK_AUDIT_2026_08_21.md`, and the 9 files named in CLAUDE.md's lists — all
69 non-recent files minus those 8 existing named exemptions). Effectively: **every
top-level doc in this directory except BOOT.md and the 2026-08-21 audit report is >3
months old**, and the vast majority (61/71 = 86%) predate June 2026 entirely. The
directory reads as a graveyard of one active file (BOOT.md) plus a long tail of
session-scoped notes from March–May 2026 that were never archived or pruned.

---

## STATUS_BOARD counts + In-PR rows

`STATUS_BOARD.md` is 1854 lines, organized into **89 `## <plan-name>` sections**, each
with its own `| D-id | deliverable | status |`-shaped table (though the column count
varies per section — some sections add LOC/Priority/Owner columns, which defeats a
single mechanical "grab the last pipe-field" parse). Mechanically extracted: **592
data rows** (excluding the ~12 repeated header rows).

**Best-effort status classification** (leading-token match on the status cell, after
stripping bold markers; ~427/592 rows carry a rich narrative status instead of a clean
lifecycle word, which is itself a finding — see Candidate epiphanies):

| Status (leading token) | Count |
|---|---|
| Narrative / not a clean lifecycle token | 427 |
| Queued | 77 (+1 mentioned mid-cell) |
| Shipped | 33 (+8 mentioned mid-cell) |
| Blocked | 23 (+6 mentioned mid-cell) |
| In PR | 5 (leading-token match only — see caveat below) |
| Proposed | 5 |
| Decision | 3 |
| Superseded | 1 (+1 mentioned) |
| Withdrawn | 1 (+1 mentioned) |

**Caveat on "In PR" count:** the leading-token classifier only reliably catches rows
using the strict 3-column `| D-id | deliverable | status |` shape. Sections with wider
tables (LOC/Priority/Owner columns) put "In PR" in a MIDDLE column, so the raw text
search for `In PR` anywhere on a line returns **~90 distinct occurrences** across the
file, spanning D-ids from `D-LNC-*`/`D-MW-P2` (2026-09) all the way back to `D-MBX-*`/
`D-WIKI-HHTL-*`/`D-IDENTITY-2` (2026-06). A fully precise per-row status census was not
attempted at this scale — see Uncertain.

**Rows whose status text is `In PR` (with the PR number cited) and merge-status check:**

| D-id | PR cited | `git log --grep` result |
|---|---|---|
| D-LNC-5a, D-MW-P2 (STATUS_BOARD.md:19-20) | #1198 | **MERGED** — `3797237b Merge pull request #1198` exists. **These two rows are STALE**: the board still reads `**In PR** #1198` although #1198 is merged (see Candidate epiphanies). |
| D-GUARD-1 (:802) | #785 | MERGED — `650f925a Merge pull request #785` |
| D-CSW-0 (:803) | #777 | MERGED — `5ce46705 Merge pull request #777` |
| D-CCF-1 (:947) | #628 | MERGED — `6858118b Merge pull request #628` |
| D-CCF-2 (:948) | OGAR #147 | external repo — not checkable from this checkout |
| D-CCF-3 (:949) | q2 #71 | external repo — not checkable from this checkout |
| D-CSV-5b/6b/15 (:1557,1559,1586) | #390 (Wave G tags) | no `Merge pull request #390` commit found in this repo's history — could be a squash-merge with different wording, or an internal wave/work-item tag rather than a GitHub PR number; **inconclusive** |
| D-CSV-10 (:1568) | #388 | same as above — inconclusive |
| D-MBX-A6-P3a/M1 (:1661,1666) | #439 | no direct "Merge pull request #439" found; a later commit (`8017ca19`) DOES reference "D-PG-6" work after #438, suggesting #439 likely landed, but not confirmed by an exact merge-commit grep |
| D-WIKI-HHTL-1/2 (:1672-1673) | #441 | no direct "Merge pull request #441" found; a later fix commit explicitly says `fix(#441): CodeRabbit — STATUS_BOARD #440->#441`, strongly suggesting #441 did land, but not confirmed by an exact merge-commit grep |

---

## ISSUES open

`ISSUES.md` is 2777 lines, **47 double-entry headers** (`## ISS-<NAME> (<date>) — <STATUS>`,
7 of which put a descriptive title after the id instead of a clean status token in the
header itself — resolved by reading each body for its actual verdict).

**Counts:** OPEN **37** · RESOLVED **8** · RESOLVING **1** · SUPERSEDED **1** (total 47).

**All 37 OPEN issues** (id + one-line title, from the header text or, where the header
carries no title, a title derived from the body):

1. `ISS-NO-NON-LINUX-TARGET` (2026-09-06) — `posix_fadvise` gated `#[cfg(unix)]` but is Linux/Android-only, breaks macOS build
2. `ISS-NO-AARCH64-RUNNER` (2026-09-06) — no CI job ever compiles an aarch64 target; two real defects found only by an external fleet
3. `ISS-EXCLUDED-CRATES-UNBUILT` (2026-09-06) — workspace-excluded crates get zero CI coverage
4. `ISS-PIN-RULING-PROSE-DRIFTS-BEHIND-THE-MANIFEST` (2026-09-05) — pin-rule prose in CLAUDE.md lags the actual Cargo.toml state
5. `ISS-PLAN-TRACKING-IS-UNENFORCED` (2026-09-03) — plans can mint D-ids the board never tracks
6. `ISS-F32-ENGINE-NEVER-CONVERGES-BY-ITS-OWN-DELTA-THRESHOLD` (2026-09-03) — `F32ThinkingEngine::cycle()` never trips its own convergence early-exit; runs full budget every time
7. `ISS-SUPERSESSION-GENERATOR-DID-RANGE-NARROW` (2026-08-28) — `supersession_index.py` only extracts the first id of a `D-XXX-0..4` range
8. `ISS-TOKEN-TENANT-16-COLLIDES-WITH-HOLEV3` (2026-08-28) — two tenants both claim `ValueTenant` discriminant 16
9. `ISS-TVT-3-DISABLE-RUN-IS-VACUOUS` (2026-08-28) — `token-value-tenant-v1`'s F-TVT-3 disable-run cannot actually fail
10. `ISS-TVT-HYDRATION-REGION-CLAIM-WRONG` (2026-08-28) — plan text contradicts the code it cites (`AWS_DEFAULT_REGION` is optional, plan says required)
11. `ISS-KANBAN-PLAN-EXIT-HAS-NO-NAMED-ROUTE` (2026-08-28) — `KanbanColumn::Plan` is DAG-legal but no primitive ever emits it
12. `ISS-F-MUL-6-HALF-BUILT` (2026-08-27) — only one of two known consumers was actually build-verified
13. `ISS-MUL-GATE-NAMED-FOR-THE-WRONG-LAYER` (2026-08-27) — `contract::mul::GateDecision` is really an execution/commit gate, module name misleads
14. `ISS-PLANNER-SANDBOX-STILL-CARRIES-FREE-TEXT` (2026-08-27) — `MulGateDecision::Sandbox{reason:String}` still heap-allocates free text
15. `ISS-NO-CAUSAL-SIGN-ON-EDGES` (2026-08-26) — `CausalEdge64` has no Inc/Dec polarity carrier
16. `ISS-D-ECG-6-BUDGET-WITHOUT-ADMISSION` (2026-08-26) — census-rank frontier has no relevance-ADMIT stage in front of it
17. `ISS-RUNG-VS-BAND-CARDINALITY-COLLISION` (2026-08-26) — `RungLevel` (10 values) vs `ReasoningBand` (8 values) share endpoint names, differ in cardinality
18. `ISS-ALPHA-NOT-LOAD-BEARING` (2026-08-26) — `AttentionMaskSoA` has no production consumer at all
19. `ISS-REASONING-BAND-GATES-NOTHING` (2026-08-26) — `ReasoningBand` is minted/tested but gates no control-loop behaviour
20. `ISS-BAND-READING-UNMINTED-IN-OGAR` (2026-08-26) — the OGAR producer side of `ClassView::band_reading` never answers
21. `ISS-DOMAIN-LENS-BY-CONVENTION-ONLY` (2026-08-26) — a real domain-conditioned-lens ruling exists only as prose in one consumer
22. `ISS-MUL-GATE-OUTCOME-COUPLED-TO-PRODUCER-GROUND` (2026-08-26) — `GateDecision::{Hold,Block}` narrowed to MUL-specific vocabulary at a public boundary
23. `ISS-PERTURBATION-P64-ADDRESS-IDENTITY-UNPROVEN` (2026-08-26)
24. `ISS-CORPUS-ADDRESSING-OPEN-POINTS` (2026-08-22)
25. `ISS-CAUSAL-EDGE-CARRIES-SEVEN-PRE-EXISTING-CLIPPY-FINDINGS` (2026-08-22)
26. `ISS-HYDRATE-ENV-READER-IS-A-SECOND-COPY-OF-DEV-S3-ENV` (2026-08-17) — deliberate, has a named exit
27. `ISS-HYDRATE-NAME-COLLIDES-WITH-TWO-EXISTING-WORKSPACE-MEANINGS` (2026-08-17) — cosmetic, no correctness impact
28. `ISS-HELIX-GOLDEN-STEP-LABEL` (2026-08-12) — misleading label, not wrong code
29. `ISS-D-IGN-B-REAL-CORPUS-PATH-IS-UNVERIFIED` (2026-08-06)
30. `ISS-IDENTITY-CODEBOOK-ORDINAL-STABILITY` (2026-08-06) — partially mitigated
31. `ISS-REMOTE-URI-CONSTRUCTORS-PREDATE-THE-HYDRATION-DOCTRINE` (2026-08-06)
32. `ISS-CODEC-RESEARCH-MDCT-ASSERT` (2026-08-05) — pre-existing, discovered not caused
33. `ISS-MARM-T1-4X-A0-GAP` (2026-08-05) — measurement defect, not a result
34. `ISS-MAILBOXSOA-ROW-COST-VS-512B-CANON` (2026-08-04) — an open question, not yet a conclusion
35. `ISS-DOMINO-WRITES-ENERGY-OUTSIDE-ITS-OWN-SCHEMA` (2026-07-29)
36. `ISS-NO-PER-THREAD-TEMPORAL-PROJECTION-IS-EVER-CONSTRUCTED` (2026-07-29) — upstream of `meta_basin`
37. `ISS-CLASSID-OGAR-DRIFT` (2026-06-20 entry, marked OPEN — needs operator sign-off; note a LATER dated entry at line 2771 for the same id is marked RESOLVING/landed, i.e. this id has two append-only dated sub-entries with different statuses — the most recent one (RESOLVING) likely supersedes this OPEN one in practice)

---

## TECH_DEBT counts + open P0/P1

`TECH_DEBT.md` is 4311 lines and uses **at least three different header conventions**
across its history (`## TD-... — OPEN`, `### TD-...` with a `**Severity:**`/`**Open.**`
body, and `## YYYY-MM-DD — TD-...:` date-prefixed) — itself a finding (see Candidate
epiphanies). Unified parse across all three: **167 total entries**.

**Status counts** (header-token first, else body `**Status:**`/`**Open.**` marker, else
UNLABELED):

| Status | Count |
|---|---|
| OPEN | 95 |
| UNLABELED (no explicit status found in header or body) | 65 |
| PAID | 4 |
| RESOLVED | 3 |

The 65 UNLABELED entries are concentrated in the oldest part of the file (PR #270-#330
era items like `TD-F10-ACTOR-ID`, `TD-ARROW-58`, `TD-INT-1`..`TD-INT-14`) which predate
the "Kanban Format (priority + scope on every entry)" convention introduced at line 1964
— they were never retrofitted with the newer schema.

**Entries tagged Priority/Severity P0 or P1 that are still Open (19 of 20 P0/P1-tagged
entries; the 20th, `TD-INT-1/2/4`, is tagged Paid):**

| Line | Priority | TD-id |
|---|---|---|
| 1758 | P1 | `TD-NDARRAY-SIMD-UNPACK-I4-16D` (W1a-#1) |
| 1772 | P1 | `TD-NDARRAY-SIMD-SATURATING-ABS-I8` (W1a-#2) |
| 1786 | P1 | `TD-NDARRAY-SIMD-GATHER` (W1a-#3) |
| 1813 | P1 | `TD-NDARRAY-SIMD-POPCOUNT-U64` (W1a-#5) |
| 1897 | **P0** | `TD-SIMD-SWEEP-W4` (lance-graph-contract mul.rs follow-up) |
| 1980 | P1 | `TD-ARIGRAPH-EPISODIC-FIDELITY-1` — AriGraph episodic retrieval transcoded as the RAG baseline the paper beats |
| 2055 | P1 | `TD-Q2-STUBS-DEDUP-1` — q2 carries local lance-graph/ndarray stubs needing re-export |
| 2116 | **P0** | `TD-API-DRIFT-MIDFLIGHT-1` — consumer migrations failing mid-air on source-crate API drift |
| 2159 | P1 | `TD-SUPER-DOMAIN-SUBCRATES-1` |
| 2242 | P1 | `TD-THINKING-ENGINE-UNWIRED-1` — 582 KB cognitive substrate dormant |
| 2289 | **P0** | `TD-SDR-PR-FOLLOWUP-1` — 5 commits stacked on merged main, no follow-up PR opened |
| 2309 | **P0** | `TD-SDR-CONSUMER-PUSH-1` — medcare-rs/smb-office-rs bridge wirings committed LOCALLY, not pushed |
| 2327 | P1 | `TD-SDR-AUDIT-PERSIST-1` — audit events emit to in-memory chain only, no persistent sink |
| 2347 | P1 | `TD-SDR-FAMILY-HYDRATION-1` |
| 4018 | P1 | `TD-CONTEXT-BUNDLE-2` |
| 4048 | P1 | `TD-GENERIC-BRIDGE-3` |
| 4069 | P1 | `TD-MANIFEST-MODULES-4` |
| 4097 | P1 | `TD-RACTOR-SUPERVISOR-5` |
| 3995 | **P0** | `TD-OGIT-G-SLOT-1` — wire u32 G slot into the SPO quad-store |

5 of these are P0 (`TD-SIMD-SWEEP-W4`, `TD-API-DRIFT-MIDFLIGHT-1`, `TD-SDR-PR-FOLLOWUP-1`,
`TD-SDR-CONSUMER-PUSH-1`, `TD-OGIT-G-SLOT-1`); 14 are P1.

---

## INTEGRATION_PLANS index vs directory

`.claude/plans/` holds **211 files**. `INTEGRATION_PLANS.md` cites **141 distinct
`.claude/plans/*.md` paths** by literal path string.

- **Plans that exist in `.claude/plans/` but are NOT cited by path in
  `INTEGRATION_PLANS.md`: 72.** Of those, 14 are mentioned by bare filename (without the
  full `.claude/plans/` path) somewhere in the index text — likely still "known" to the
  index in spirit. **58 plans are mentioned nowhere in `INTEGRATION_PLANS.md` at all**
  (neither full path nor bare name) — see the list below. A large sub-cluster (20 files)
  is the entire `3DGS-*` doc family (has its own `3DGS-PLAN-INDEX.md` sub-index instead),
  and another sub-cluster is the `tesseract-rs-*` transcode plans (5 files) and the
  `alpha-reason-witness-shader-field-*` archaeology plans (3 files).

  Full 58-item unmentioned list: `3DGS-3D-Tiles-runtime-plan`, `3DGS-4x4-cognitive-shader-integration-plan`,
  `3DGS-ArcGIS-Cesium-ingestion-plan`* , `3DGS-Blender-transcode-crosspollination-plan`,
  `3DGS-Cesium-BindSpace4-headstone-exploration`, `3DGS-Cesium-feature-mapping-plan`,
  `3DGS-HHTL-datalake-traversal-plan`, `3DGS-Lance-Arrow-storage-plan`, `3DGS-PLAN-INDEX`,
  `3DGS-PRX12-crosspollination-capstone`, `3DGS-SplatShaderBlas-BLASGraph-crosspollination-plan`,
  `3DGS-blast-radius-application-map`, `3DGS-certified-query-render-plan`,
  `3DGS-cross-pollination-raw-field-plan`, `3DGS-domain-adapter-strategy-plan`,
  `3DGS-epiphany-roadmap-plan`, `3DGS-genetics-4x4-fanout-plan`, `3DGS-integration-wiring-plan`,
  `3DGS-neuronal-network-4x4-plan`, `3DGS-ultrasound-SaMD-plan`,
  `Palette256-3DSB-PhiSpiral-attention-integration-plan`, `PhiSpiral256-SoA-cross-system-integration-plan`,
  `a3-carrier-v1`, `alpha-reason-witness-shader-field-archaeology-pass-1`,
  `alpha-reason-witness-shader-field-d-arw-0-audit-v1`, `alpha-reason-witness-shader-field-lineage-addendum-v1`,
  `archetype-scaffold-v1`, `belief-abi-restoration-v1`, `burn-ndarray-parity-sprint-v1`,
  `cognitive-substrate-convergence-v2`, `cognitive-substrate-convergence-v3`,
  `cognitive-write-roundtrip-substrate-v1`, `deepnsm-morton-comma-facet-v1`, `dialectic-engine-v1`,
  `dtsc1-thinkingstyle-dedup-spec-v1`, `epistemic-quadrant-materialization-v1`,
  `genetic-research-substrate-integration-v1`, `genetics-probes-v1`, `hydrate-crate-hardening-council-v1`,
  `lance-graph-rdf-fma-snomed-v1`, `lance9-datafusion54-upgrade-probe-v1`, `measure-64k-axes-v4`,
  `ocr-canonical-soa-integration-v1`, `ocr-probes-v1`, `oracle-funnel-probe-v1`,
  `oxigraph-arigraph-cognitive-shader-soa-merge-v1`, `post-438-integration-options-v1`,
  `probe-revision-attention-view-1`, `reliability-checklist-arc-v1`, `soa-migration-diff-resolution-2026-06-13`,
  `sprint-5-through-9-roadmap-v1`, `sql-spo-ontology-bridge-v1`, `tesseract-rs-ast-dll-codegen-v1`,
  `tesseract-rs-layout-transcode-v1`, `tesseract-rs-recodebeam-transcode-v1`,
  `tesseract-rs-traineddata-gguf-v1`, `tesseract-rs-transcode-master-v1`,
  `tetrahedral-epiphany-splat-integration-v1`, `transcode-extend-core-probe-v1`,
  `wikidata-lazy-spine-hydration-v1`

- **Paths cited in `INTEGRATION_PLANS.md` that do NOT exist as a file in `.claude/plans/`: 2.**
  1. `.claude/plans/pdf-to-text-ocr-v1.md` — also referenced from `tesseract-rs`'s own
     `CLAUDE.md` ("plan `pdf-to-text-ocr-v1.md`"); may genuinely live only in the
     `tesseract-rs` repo, or the reference is stale in both places.
  2. `.claude/plans/splat-native-ultrasound-simd-substrate-v1.md` — the directory has
     `splat-native-ultrasound-v1.md` (no `-simd-substrate` infix); looks like a naming
     drift/typo between the citation and the actual filename.

---

## LATEST_STATE summary

`LATEST_STATE.md` is 3460 lines, read across multiple offsets (structure: newest-first
append-only dated delta entries at the top, running from **2026-09-07** back through
2026-05-05, followed by several STATIC/summary sections near the bottom of the file that
have not been kept in sync with the dated deltas above them).

- **"as of" date:** **2026-09-07** (top entry: `D-BLOCKS-HOTPLUG-1`, an operator ruling
  that a hot-plugged consumer classid is not a canon builtin).

- **"Current Contract Inventory (lance-graph-contract)" section** (line 2915-2982):
  **27 blockquote (`> **...`) entries**, spanning dates 2026-05-30 through 2026-06-26.
  This is a STATIC section — it has NOT been updated with any of the ~100+ dated delta
  entries that were prepended above it since 2026-06-26 (over two months of newer
  contract changes are not reflected here). See Candidate epiphanies.

- **"Immediate Next Work" / Queued Work section** (verbatim, lines 3073-3101):

  > **Queued Work — sprint-13 (specs being drafted in the sprint-13-preflight fleet on
  > this branch):**
  > - **D-CSV-13b** — SIMD vectorization of D-CSV-8 i4 MUL evaluation. **IN PR
  >   (sprint-13/W-I1 salvage)**...
  > - **D-CSV-14** — on-Think method migration for D-CSV-12 splat ops...
  > - **D-CSV-16** — NEW sprint-13 entry. Spec being drafted by PP-5.
  > - **D-CSV-17** — NEW sprint-13 entry. Spec being drafted by PP-3.
  >
  > **Sprint-14+ Phase F items (Backlog):**
  > - ndarray `parallel`-feature `par_*` rayon variants for `QualiaStream` /
  >   `InferenceStream` / `SplatFieldStream` (work-stealing).
  > - D-REUNIFY-4/5/6 carryover from causaledge64-mailbox-rename-soa-v1...
  >
  > **`codec-sweep-via-lab-infra-v1` Phase 0 remainder (carry-over):**
  > - D0.1, D0.2, D0.3, D0.5 + four test gates (listed verbatim in the file).
  >
  > **`elegant-herding-rocket-v1` Phase 2 (still queued):**
  > - D2, D3, D5, D7 (listed verbatim in the file).

  **This entire section is dated to sprint-13/sprint-14 (~2026-05-16) and is STALE**:
  D-CSV-13b is cited here as "IN PR" but STATUS_BOARD.md's own D-CSV-13b row (line 1584)
  shows it long since shipped with measured SIMD benchmarks. This "Immediate Next Work"
  section has not been refreshed to reflect the file's own top-of-file 2026-09-07 state.

- **"Deferred (do NOT propose these — they're explicitly parked)" section** (verbatim,
  lines 3102-3119):

  > **Sprint-12/13 explicit deferrals (2026-05-16):**
  > - **TD-COLLAPSE-GATE-SMALLVEC-1** — CLOSED 2026-06-11 as moot: `CollapseGateEmission`
  >   removed entirely (PR #477 tombstone commit), nothing left to optimize.
  > - **TD-SIGMA-TIER-THRESHOLDS-1** — Σ10 VAMPE-coupled Jirak-derived threshold
  >   refinement (D-CSV-15). Hand-tuned acceptable through sprint-12 per
  >   `I-NOISE-FLOOR-JIRAK`; principled Jirak 2016 derivation forwarded to sprint-13+
  >   VAMPE coupled-revival track.
  > - **ndarray `parallel`-feature `par_*` rayon variants** — productized substrate
  >   ships sequentially in PR #147; rayon work-stealing wraps deferred to sprint-14+
  >   behind an opt-in feature gate.
  >
  > **Long-running parks (pre-existing):**
  > - CausalityFlow TEKAMOLO extension (modal/local/instrument + beneficiary/goal/source,
  >   9 total) — struct change deferred until after Phase 2.
  > - D8 story-context bridge, D9 ONNX arc export, D10 Animal Farm validation, D11
  >   bundle-perturb emergence — Phase 3/4.
  > - Named Entity pre-pass (NER) — biggest OSINT blocker, separate PR.
  > - FP_WORDS = 160 migration (currently 157) — coordinated ndarray change.
  > - Crystal4K 41:1 persistence compression.
  > - 200-500 YAML TEKAMOLO templates per language — future training pipeline.
  > - Python/TypeScript grammar-stack convergence.

  One entry in this list (`TD-COLLAPSE-GATE-SMALLVEC-1`) is self-annotated CLOSED as of
  2026-06-11, i.e. this "Deferred" list already contains a stale row it did not remove.

---

## IDEAS current list

`IDEAS.md` is 1236 lines, **47 `## ` headers total** — a mix of structural/instructional
headers (Triple-entry discipline, Governance, Kanban Format, Open Ideas, Implemented
Ideas, Rejected/Deferred Ideas, How-to-use, template placeholders) and **33 actual dated
idea entries**. Status extraction (via `**Status:**` line where present):

| Line | Date | Status | Title |
|---|---|---|---|
| 90 | 2026-09-05 | Open | Σ-propagation: the batched `F64x8` kernel is the real lever; AMX/MKL was the wrong shape for a 2×2 |
| 98 | 2026-09-05 | Open | The σ column indexes a codebook that exists in three doc comments and zero crates |
| 106 | 2026-09-05 | Open | `ndarray::simd::ternlog` ⋂ the §14 voxel cube: the "other 18 cells" are one named immediate each |
| 114 | 2026-09-05 | Open | Citations should carry the path the decay gate can resolve, or the gate sees 7% of them |
| 122 | 2026-06-15 | RESEARCH | CLAM residue ladder ⋂ knee/hip attractor basins (HHTL cascade in REVERSE) |
| 140 | 2026-05-13 | (none found) | CORRECTION-OF previous same-day splat row: split into two distinct ideas |
| 144 | 2026-05-13 | (none found) | EXECUTION PATH: prerendered cinematic as palette-indexed frames in LanceDB |
| 206 | 2026-05-13 | (none found) | REFRAME: holographic cinematic is a SALES asset, not a product feature |
| 230 | 2026-05-13 | (none found) | RECONCILIATION: Amiga-demoscene prerender + live 60fps renderer compose, not compete |
| 250 | 2026-05-13 | (none found) | Sci-fi presentation vision: transparent holographic human-body projection for q2 |
| 278 | 2026-05-13 | (none found) | CORRECTION-OF prerender row: ndarray already ships the 60fps double-buffer renderer |
| 282 | 2026-05-13 | (none found) | Separate-and-orthogonal: 3D Gaussian-Splat prerender buffer as Tier-3 FMA render path |
| 286 | 2026-05-13 | Open | Super-domain subcrate scaffolding cascade (MedCare→smb-bridge→woa-rs→hiro/hubspot) |
| 332 | 2026-05-13 | Open | Pattern E+F+cognition cascade: manifest + ractor supervisor + thinking-engine bridge |
| 365 | 2026-05-13 | Open | Wire `thinking-engine` into UnifiedBridge — collapse D-SDR-13/15/17 into one module |
| 414 | 2026-04-29 | Implemented | Probe P1: γ-phase-offset ranking discrimination (from 2026-04-29) |
| 499 | 2026-04-29 | Open | Inverted-pyramid awareness streaming via CausalEdge64 through SPO+COCA→CAM_PQ |
| 522 | 2026-04-29 | Implemented | Probe P1: γ-phase-offset ranking discrimination |
| 548 | 2026-04-29 | Open | Safetensor-Streaming als ndimensionale Bedeutungsakkumulation |
| 566 | 2026-04-29 | Open | Family-Bounds als globale fraktale Codierung (Hypothesis Test) |
| 583 | 2026-04-29 | Open | Pillar 7 Front-to-Back α-Akkumulation (LIKELY-REDISCOVERY) |
| 600 | 2026-04-29 | Open | Pillar 8 Adaptive Densification für Σ-Codebook |
| 615 | 2026-04-29 | Open | Pillar 9 SH-Koeffizienten als Thinking-Style-Manifold |
| 632 | 2026-04-19 | Open | FP_WORDS = 256 (supersede the 160 plan) |
| 663 | 2026-04-19 | Open | CORRECTION-OF FP_WORDS = 256 |
| 721 | 2026-04-19 | Open | REFINEMENT-OF CORRECTION-OF FP_WORDS scope |
| 772 | 2026-04-19 | Open | REFINEMENT-2: HDC substrate is FP16/BF16, not FP32 |
| 829 | 2026-04-19 | Open | lance-graph-cognitive refactor: dedup + merge + excise |
| 886 | 2026-04-19 | Open | CORRECTION-OF lance-graph-cognitive refactor |
| 904 | 2026-04-19 | Open | Fractal round-trip codec: phase+magnitude preservation |
| 957 | 2026-04-19 | Open | Fractal codec validation path: codec_rnd_bench + ICC_3_1 |
| 1006 | 2026-04-19 | Open | Zipper codec: phase+magnitude multiplexed in one bgz17 container |
| 1088 | 2026-05-05 | Reshaped | Future-work items extracted from PRs #244–#335 |

Structural note: entries at lines 499-1006 sit AFTER the file's own "## How to use this
file" section (line 479) — an unusual placement for an append-only-at-top idea log; not
fully explained within budget (see Uncertain).

---

## EPIPHANIES recent + id checks

`EPIPHANIES.md` is **27,617 lines** — not read whole per task instructions; only
`grep -n "^## "` used.

**15 most recent headings (verbatim, newest-first):**

1. `## 2026-09-07 — E-PLUG-AND-PLAY-IS-THE-DECLARATION-NOT-A-TABLE-1 — my fix rebuilt the lockstep it was closing`
2. `## 2026-09-07 — E-A-DOC-COMMENT-IS-NOT-A-FAIL-CLOSED-MECHANISM-1 — hotplug could still land on V1 with a one-liner`
3. `## 2026-09-07 — E-A-V3-MINT-MUST-NEVER-DEGRADE-TO-V1-1 — the fallback arm's own justification was falsified by D-BLOCKS-HOTPLUG-1`
4. `## 2026-09-07 — E-THE-V1-GUARD-WAS-TESTED-THE-V3-GUARD-THAT-REPLACED-IT-WAS-NOT-1 — a guard nothing proves can fire is the defect one level up`
5. `## 2026-09-06 — E-AN-EXCLUDED-CRATE-ON-AN-X86-ONLY-FLEET-IS-CODE-NO-CI-HAS-EVER-COMPILED-1 — un-gating one downstream suite found a second aarch64 defect that could never have built`
6. `## 2026-09-06 — E-I-CITED-THE-RIGHTMOST-REGISTER-AND-CALLED-IT-THE-ADDRESS-1 — three corrections to one entry, each because I reasoned instead of measuring`
7. `## 2026-09-05 — E-A-SWEEP-IS-COMPLETE-ONLY-WITHIN-THE-TARGET-KINDS-ITS-GATE-COMPILES-1 — #1194 swept the whole 1.98 delta and still left two sites, because "whole" was measured through six clippy steps`
8. `## 2026-09-06 — E-THE-AARCH64-PATH-HAD-NEVER-BEEN-COMPILED-1 — a cfg-gated arch path is dead code until some CI targets it`
9. `## 2026-09-05 — E-A-MACHINE-APPLICABLE-FIX-IS-A-SUGGESTION-NOT-A-PROOF-1 — clippy's own autofix did not compile, and the lint it fixes is a substrate argument`
10. `## 2026-09-05 — E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1 (OPERATOR-RULED)`
11. `## 2026-09-05 — E-THE-UNFINISHED-UDF-WAS-NOT-THE-DEBT-1 — D-OIF-1 re-derived: the execution model that needed policy_hash_v1 never reached production`
12. `## 2026-09-05 — E-BLW5-FIRST-MEASUREMENT-1 — the observer-effect instrument is alive, and every pre-registered gate reads SILENT`
13. `## 2026-09-05 — E-NARS-EXPECTATION-CHOICE-PREFERS-IGNORANCE-TO-A-CONFIDENT-NEGATIVE-1 — a derived belief with f < 0.5 loses CHOICE to a vacuous one`
14. `## 2026-09-05 — E-THE-UNFINISHED-FUNCTION-WAS-NOT-THE-DEBT-1 — the execution model that needed policy_hash_v1 never reached a binary, and neither has its replacement`
15. `## 2026-09-05 — E-A-COLUMN-OF-INDICES-INTO-A-CODEBOOK-THAT-DOES-NOT-EXIST-1 — three stale idea cards, each wrong about its own blocker`

(Note: item 8 dated 2026-09-06 sits chronologically AFTER item 7 dated 2026-09-05 in the
file's own prepend order — the append-only ledger is not strictly date-sorted at every
boundary, likely from two branches merging in close succession.)

**Requested id existence checks:**

| Id | Exists? | Location |
|---|---|---|
| `E-SIGNATURE-PDE-SWEEP-SHIPPED-W1.5-GATE-WAS-QUIETLY-OPEN-1` | **YES** | line 2195, dated 2026-09-02 |
| `E-PILLAR-11-PUBLISHED-BOUND-NEEDS-ITS-OWN-NUMERIC-GUARD-1` | **YES** | line 2253, dated 2026-09-02 |
| `E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1` | **YES** | line 539, dated 2026-09-05 (operator-ruled) |
| `E-LANCE-IS-UPSTREAM-AUTHORITATIVE-1` | **YES** | line 12295, dated 2026-08-05 (operator-ruled); also referenced elsewhere at line 583 |

**`log_signature` / `LOG-SIGNATURE` / `LYNDON` headings:** one substantial entry found —
`## 2026-09-03 — E-A-DOCUMENTED-MODULE-THAT-WAS-NEVER-\`pub mod\`-D-IS-DEAD-CODE-WITH-A-PARITY-CLAIM-1`
(line 2094) — `crates/sigker/src/log_signature.rs` existed on `main`, 388 lines, cited a
Reizenstein-Graham paper and a "7-13× compression, lossless" claim, but `lib.rs` never
declared `pub mod log_signature;` — it had never been compiled once. The entry also
corrects a basis-projection bug (flat read ≠ true Lyndon-basis coefficients) and
corrects the compression ratio (2.10×-7.62× depending on depth, not a flat "7-13×").
`TD-NDARRAY-SIMD-LYNDON-PACK` (TECH_DEBT.md line 1854, tagged P3/DEFERRED) is the
still-open SIMD follow-up this entry unblocks but does not itself resolve.

---

## SUPERSESSION-INDEX summary

`SUPERSESSION-INDEX.md` is 159 lines, fully read (it is generated + small).

- **Row counts:** Table 1 (ruled symbols) = **14 rows**. Table 2 (plans naming a ruled
  symbol without citing the ruling) = **73 rows**, broken down as **RESCOPE: 55 · READ:
  18 · ARCHIVE?: 0**.

- **14 ruled symbols, with verdict:**

  | Symbol | Verdict | Successor | Live in crates | Named in plans | Blind plans |
  |---|---|---|---|---|---|
  | `A2AMessage` | BLOCKED | — | 2 | 1 | 0 |
  | `StepMask` | BLOCKED | — | 3 | 9 | 4 |
  | `commit_to_l4` | BLOCKED | — | 2 | 2 | 0 |
  | `dispatch_busdto` | BLOCKED | — | 3 | 8 | 5 |
  | `persist_cycle` | BLOCKED | — | 11 | 8 | 5 |
  | `CognitiveMarkers` | REPURPOSE | `Commit` | 1 | 0 | 0 |
  | `DominoCascade` | REPURPOSE | `Commit` | 7 | 1 | 0 |
  | `GateDecision` | REPURPOSE | — | 25 | 27 | 24 |
  | `GateState` | REPURPOSE | — | 14 | 1 | 1 |
  | `MergeMode` | REPURPOSE | — | 8 | 13 | 12 |
  | `ResonanceDto` | REPURPOSE | `PerturbationDto` | 2 | 12 | 7 |
  | `BindSpace` | **RETIRE** | — | 68 | 47 | 41 |
  | `CollapseGateEmission` | **RETIRE** | — | 5 | 14 | 12 |
  | `ThinkingStyle` | RETIRE-toward-contract | — | 52 | 28 | 24 |

  Headline (the file's own commentary, verified against the table): `BindSpace` is
  marked RETIRE and is simultaneously the most-referenced symbol in the whole index —
  68 crate files, 47 plans, 41 of them "blind" (naming it without citing the ruling).

- **73 plans flagged as naming a ruled symbol without citing the ruling** — the full
  55-item RESCOPE list and 18-item READ list are reproduced verbatim in the file at
  `.claude/board/SUPERSESSION-INDEX.md:82-154`; not re-transcribed here for space, but
  every row is present in the file (read in full, see above). Two rows worth flagging
  individually: `GateDecision` review found the mechanical `ARCHIVE?` route's first
  batch was **3/3 false positives** (the file documents this itself, lines 73-78) — so
  the route is explicitly self-described as "a prompt to read the plan, never a licence
  to act on it," and this session did not attempt to re-verify any individual RESCOPE/
  READ row's current correctness (that would require reading up to 55+18 plan files).

---

## Candidate epiphanies

Facts observed this pass that look like genuine board-hygiene findings, each with a
file:line anchor, offered as raw material — not claimed as ratified epiphanies.

1. **STATUS_BOARD.md:19-20** — `D-LNC-5a` and `D-MW-P2` both read `**In PR** #1198`, but
   `#1198` has a merge commit on `main` (`3797237b`, verified via
   `git log --oneline --all --grep="Merge pull request #1198"`). These two rows are
   stale post-merge — the board hygiene rule (root `CLAUDE.md` "Mandatory Board-Hygiene
   Rule") requires a merged-PR row to update in the same commit; here it evidently did
   not, or a later merge (of #1199, which the AGENT_LOG shows recording #1198's own
   merge — `.claude/board/LATEST_STATE.md:213-227`) never touched these two rows.

2. **LATEST_STATE.md:2915-2982 vs :1-2914** — the "Current Contract Inventory" section
   (27 entries, frozen at 2026-06-26) has not been updated in over two months of newer
   dated deltas prepended above it (through 2026-09-07). A session reading only the
   named section header would get a badly stale contract inventory; the real current
   state is scattered across ~100+ dated top-of-file entries instead.

3. **LATEST_STATE.md:3073-3119** — "Immediate Next Work" / "Deferred" sections are dated
   internally to sprint-13/sprint-14 (~2026-05-16) and cite work (e.g. D-CSV-13b "IN PR")
   that STATUS_BOARD.md's own row for the same D-id (line 1584) shows shipped with
   measured benchmarks long since. One row in "Deferred" (`TD-COLLAPSE-GATE-SMALLVEC-1`)
   is even self-annotated "CLOSED 2026-06-11 as moot" while still sitting in the
   "Deferred (do NOT propose these)" list.

4. **TECH_DEBT.md** — the file uses at least 3 distinct header conventions across its
   history (`## TD-X — OPEN`, `### TD-X` + body `**Severity:**`/`**Open.**`, and
   `## YYYY-MM-DD — TD-X:`), and the "Kanban Format (priority + scope on every entry)"
   convention (line 1964) was introduced partway through the file's life — 65 of 167
   entries (39%) carry no machine-extractable status at all, mostly the oldest
   PR-numbered items (`TD-F10-ACTOR-ID` .. `TD-INT-14`, lines 3513-4264) that predate the
   convention and were never retrofitted.

5. **CLAUDE.md** (root) "Prior art" section says "`.claude/*.md` (61 top-level docs)" —
   the actual count today is 71. The section also names `SESSION_CAPSTONE.md` as an
   example file, which does not exist anywhere in this repo (`ls .claude/*.md` has no
   such file) — a dangling citation in the repo's own operational-contract doc.

6. **`.claude/pattern.md` vs `.claude/patterns.md`** — near-duplicate purpose (both
   "SoA/DTO graph traversal" primers, both claim "read by every session" / "READ FIRST"
   before touching that surface), both still actively cross-referenced (17 and 30
   internal refs respectively) and both younger than most of the directory (2026-05-06
   and 2026-05-12) — worth a session asking whether these should be merged rather than
   maintained as two parallel entry points to the same doctrine.

7. **INTEGRATION_PLANS.md coverage gap** — 58 plan files (27% of the 211 in
   `.claude/plans/`) are named nowhere in the index, concentrated in two coherent
   clusters (`3DGS-*`, 20 files, tracked instead by its own `3DGS-PLAN-INDEX.md`; and
   `tesseract-rs-*` transcode plans, 5 files) — suggests `INTEGRATION_PLANS.md`'s
   "index of every plan" framing (root CLAUDE.md: "Consult before proposing a new plan")
   is not actually comprehensive for at least these two whole subject areas.

---

## Uncertain

- **STATUS_BOARD.md "In PR" full census**: given the file mixes 3-column, 4-column and
  6-column table shapes across 89 different plan sections (592 data rows total), a fully
  precise single-pass mechanical extraction of "every row whose status is exactly In PR"
  was not completed — only the rows matching the strict `| D-id | deliverable | status |`
  3-column shape were reliably classified (5 rows), plus a broader raw-text sweep found
  ~90 lines mentioning "In PR" somewhere. The 10 clearest single-D-id "In PR" citations
  were individually checked against `git log`; PR numbers #390, #388, #439, #441 could
  not be confirmed merged or open via an exact `git log --grep="Merge pull request #N"`
  match (no such commit found), which could mean squash-merge wording differs, the
  numbers are internal wave tags rather than GitHub PR numbers, or they are genuinely
  still open — not resolved within this pass's budget.

- **TECH_DEBT.md "still open" for the 65 UNLABELED entries**: this report treats
  UNLABELED as "status unknown," not as "open." Per the file's own stated policy
  ("Debt moves Open → Paid by status-flip; rows are NEVER deleted"), an entry with no
  flip marker plausibly IS still open by default, but that inference was not verified
  entry-by-entry (would require reading ~65 bodies individually).

- **IDEAS.md structural placement**: dated entries at lines 499-1006 (2026-04-19 through
  2026-04-29 topics) sit physically AFTER the "## How to use this file" instructional
  section (line 479), which is an unusual position for an append-only-at-top idea log.
  Not investigated further — could be a historical bulk-import placed below the
  instructional boilerplate, or a structural quirk from an earlier file-reorganization;
  flagged for a follow-up read rather than resolved here.

- **ISS-CLASSID-OGAR-DRIFT double entry** (ISSUES.md lines 2771 and 2774): two dated
  sub-entries for the same issue id, one OPEN (older position in the file, meaning it
  was likely written LATER given the append-at-top convention — but the OPEN one's own
  visible date-string is "2026-06-20" same as the RESOLVING one) — the file's own
  ordering vs the embedded date strings appear to conflict slightly; treated both as
  distinct rows in the OPEN/RESOLVED counts above without fully resolving which is
  chronologically authoritative.

- **7 top-level docs whose header carries a descriptive title instead of a clean
  OPEN/RESOLVED-style token** (in ISSUES.md) required reading their bodies to determine
  status; this was done for all 7, but a couple (e.g. `ISS-F32-ENGINE-NEVER-CONVERGES...`)
  had no single unambiguous "OPEN"/"RESOLVED" sentence — the OPEN classification there
  is inferred from prose ("Not root-caused or fixed... Not scheduled") rather than a
  literal status field.

## Orchestrator errata (appended 2026-09-07 after review of #1218 — the agent text above is unedited)

- L150–160: the nine leading-token buckets sum to 575 of the 592 data rows; the remaining 17 rows carry a leading token outside those nine classes and were not classified in this pass. Read the table as "575 classified + 17 unclassified = 592", not as a partition. Mirrored in the inventory §0 / §6.
