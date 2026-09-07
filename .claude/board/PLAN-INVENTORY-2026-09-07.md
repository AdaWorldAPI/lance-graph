# Plan inventory — 2026-09-07 (`.claude/`, `.claude/v3/`, `.claude/plans/`, W-waves, jc)

> **What this is.** A measured census of every plan file, the V3 folder, the
> SIMD W-waves and the jc pillar registry, against the tree at `aeebfb23`
> (main, 2026-09-07). Five read-only Sonnet agents wrote the evidence
> (verbatim tag-files under `exec-runs/plan-inventory-2026-09-07-*.md`); the
> orchestrator spot-checked every claim that enters a board file below
> (`file:line` in the tree, not in the tag-file) and corrected one. This is
> a snapshot, not a plan: it mints **no D-ids**, so `supersession_index.py`
> and `plan_dids.py` do not see it — by design.
>
> **Operator ask (2026-09-07):** *"create an inventory about the plans in
> `.claude` `.claude/V3` `.claude/plans` … what is still open what is closed,
> check if v3 already has the harvest and update, write down any epiphanies
> and expansion ideas."* Epiphanies: `EPIPHANIES.md` 2026-09-07 block. Ideas:
> `IDEAS.md` 2026-09-07 block. V3 harvest mirrors: §4.
>
> **How to read a verdict.** OPEN = the plan's own status line and/or its
> D-ids on `STATUS_BOARD.md` say work remains. CLOSED = the plan's own
> deliverable is delivered (a measurement, an audit, a shipped ladder).
> SUPERSEDED = a higher-numbered sibling exists or the status line says so.
> AMBIGUOUS = no status line in house format and no board row to decide it.
> Every verdict came from a **read of the status line in context** — naive
> substring matching on "SHIPPED" produced ≥ 6 false positives in this corpus
> (`genetic-research-substrate-integration-v1.md` "No code shipped yet",
> `weather-soa-bake-v1.md` "No code shipped", `entropy-ladder-spo-rung-v1.md`
> "R1 SHIPPED; R2–R6 planned", `mul-consumer-build-gate-v1.md` "GATE RUN" with
> §4 "OPEN — not discharged", …), exactly the trap `CLAUDE.md` § "What the
> gate does NOT prove" records for the `ARCHIVE?` route.

---

## 0. The numbers

| Surface | Count | Open | Closed | Superseded | Ambiguous |
|---|---|---|---|---|---|
| `.claude/plans/*.md` A–K (entries 1–103) | 103 | 64 | 1 | 2 | 36 |
| `.claude/plans/*.md` L–Z (entries 104–211) | 108 | 85 | 12 | 7 | 4 |
| **`.claude/plans/` total** | **211** | **149** | **13** | **9** | **40** |
| `.claude/v3/` waves W0–W6 | 7 | 4 (W3/W4 partial, W5, W6) | 1 (W0) | — | 2 (W1 mechanism-closed/adoption-partial; W2 contradictory) |
| `.claude/v3/ENTROPY-MILESTONES.md` M1–M27 | 27 | 9 queued + 6 in-flight + 1 ruling-needed | ~8 | — | — |
| `.claude/*.md` top-level | 71 | — | — | 1 (`SESSION_FALKORDB_CROSSCHECK`) | 9 orphans, 61 pre-June-2026 |
| `STATUS_BOARD.md` D-id rows | 592 (89 sections) | 77 Queued, 23 Blocked, 5 In PR | 33 Shipped | 1 | 427 narrative-status cells |
| `ISSUES.md` | 47 | 37 | 8 (+1 resolving) | 1 | — |
| `TECH_DEBT.md` | 167 | 95 | 7 (4 Paid + 3 Resolved) | — | 65 unlabeled (pre-Kanban-format era) |
| W1a SIMD primitives (ndarray) | 5 | 0 | 5 | — | — |
| W1b consumer migrations | 6 | 6 (one half-done) | 0 | — | — |
| W1.5 sigker primitives | 3 | 1 (#8, ungated) | 2 (#6, #7) | — | — |
| jc pillar registry | 12 | 1 deferred (Pillar 2) + 1 feature-gated (Pillar 11) | 10 execute by default | — | — |

The 40 AMBIGUOUS plans are 35 A–K files with no `Status:` line at all (22 of
them the `3DGS-*` / `Palette256-3DSB-*` / `PhiSpiral256-SoA-*` design-doc
genre, landed in one batch on 2026-05-25; 13 ordinary plans that never got a
header), `cascade-seal-register-grid-v1.md` ("RATIFIED v3 — council complete"
with both D-ids absent from the board), and four L–Z files
(`ogit-g-context-bundle-v1`, `probe-revision-attention-view-1`,
`supabase-subscriber-v1`, `tarski-markov-hhtl-seam-v1` — the last is "NOT a
plan" by its own header).

## 1. Board lags the tree — in BOTH directions

The load-bearing finding. Fixed in this PR where the fix is a status cell;
recorded where it needs a reader.

### 1a. Docs still saying *Open / Deferred / Queued / does not exist* for work that is on `main`

| Where | Said | Tree | Fixed here |
|---|---|---|---|
| `TECH_DEBT.md` TD-NDARRAY-SIMD-{UNPACK-I4-16D, SATURATING-ABS-I8, GATHER, PREFETCH, POPCOUNT-U64} (W1a #1–#5) | `Status: Open` ×5 | every symbol on ndarray `master` `b9afcb9b` — `simd_scalar.rs:1684,1709,1799,1887,1916` / `simd_avx512.rs:2673,2716,2846,3091,2987` / `simd_neon.rs:2269,2301,2356,2445` | yes — regraded SHIPPED (primitive side); consumer halves stay Open |
| `TECH_DEBT.md` TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION (W1.5 #7) | `DEFERRED` (heading + body) | `ndarray/src/hpc/randomized_signature.rs:96,205,244,292` (PR #294) **and consumed** — the import `use ndarray::hpc::randomized_signature::randomized_signature_sweep` at `crates/sigker/src/randomized.rs:49`; call site `RandomizedSignatureBuilder::encode` (`randomized.rs:124`) | yes — a three-way inconsistency (two knowledge docs said SHIPPED since 2026-09-02) |
| `TECH_DEBT.md` TD-SIMD-SWEEP-W2 | "`types.rs` (22 raw ops)" | `types.rs` has **0** raw-intrinsic lines (`hamming_distance_dispatch` → `ndarray::hpc::bitwise::hamming_distance_raw`, `types.rs:462`); `ndarray_bridge.rs` still 66, incl. `_mm512_popcnt_epi64` at `:465,:493` | yes — HALF DONE |
| `STATUS_BOARD.md` D-LNC-5a, D-MW-P2 | `**In PR** #1198` | #1198 merged `3797237b` | yes — Shipped |
| `.claude/v3/ENTROPY-MILESTONES.md` M2 | `QUEUED (small, mechanical)` | STATUS_BOARD D-PERT-1 Shipped #630, verified in-code 2026-07-10 — nine weeks before the file was last edited | yes |
| `.claude/v3/ENTROPY-MILESTONES.md` M24 | `PARTIAL 2026-07-17` | STATUS_BOARD D-MBX-A6-P3d/P4 (2026-08-01/02) ship the durable witness + recovery + cycle/WAL closure with the 64k/17 falsifier | yes — regrade appended |
| `.claude/v3/knowledge/compiled-templates.md` gap list | "`StepMask` type does not exist yet" | `contract::step_mask::StepMask`, D-V3-W3a, 2026-07-10 | yes |
| `.claude/plans/self-reasoning-substrate-v1.md:15` | "PROPOSED — doc-only. No code" | D-SRS-1..4 all **Shipped** on STATUS_BOARD (`reason.rs` + 7 tests; `shape,ancestry.rs` + 63; falsifier fired; `introspect.rs` + 77) | no — plan header, left for the owner; recorded |
| `LATEST_STATE.md` static sections `## Current Contract Inventory (lance-graph-contract)` (`:2915`), `## Immediate Next Work` (`:3073`), `## Deferred (…)` (`:3102`) | frozen at 2026-05-16 / 2026-06-26; cite D-CSV-13b "IN PR" (shipped long since) and keep `TD-COLLAPSE-GATE-SMALLVEC-1` in "Deferred" while self-annotating it "CLOSED 2026-06-11" (`:3106`) | ~100 dated deltas prepended above them since | no — the file is append-only; a reader must treat the top-of-file deltas as current, the static sections as a 2026-06 snapshot |
| `CLAUDE.md:762,765` | "`.claude/*.md` (61 top-level docs) … `SESSION_CAPSTONE.md`" | 71 files; no `SESSION_CAPSTONE.md` exists | no — CLAUDE.md is operator-owned; recorded |

### 1b. Docs still carrying a premise that was rescinded, or ids that were never minted

| Where | Said | Truth | Fixed here |
|---|---|---|---|
| `.claude/v3/INTEGRATION-PLAN.md:110` (D-CCF-4 row), `soa_layout/routing.md:104-106`, `knowledge/v3-substrate-primer.md:103-105` | `0x1000` is temporary; adoption 100 % ⇒ P4 ⇒ marker retires | **D-CCF-4 RESCINDED** (operator 2026-07-03, `E-V3-DUAL-SCHEMA-0x1000-IS-PERMANENT-1`) — STATUS_BOARD row 950 carried it for nine weeks; none of the three V3 files did | yes — ⊘ notes at all three |
| `.claude/v3/INTEGRATION-PLAN.md` Addendum-12a (`:486-510`) vs Addendum-15 (`:744`) | W2a = a NEW gated `BoardAggregates` tenant **vs** W2a "SHIPPED as `ValueTenant::Kanban`" | `ValueTenant::Kanban` is the pre-existing per-ROW tenant #9 that `mailbox-kanban-model.md` calls a *sibling, not a substitute*; `tenants.md` (2026-08-23) lists no `BoardAggregates` | no — recorded on FUTURE-DESIGN 2026-09-07; needs `canonical_node.rs:1622+` + owner wiring read, not a doc pass |
| `.claude/v3/COMPONENT-MAP.md` §4 W2b row | supervisor `KanbanActor<O>` EXTEND | actor half DELETED 2026-08-05 (⊘ note already in the row since 2026-09-05); the wave table's W2b row still reads as an open work item | no — ⊘ exists; the wave table row is append-only history |
| `persistent-nars-kg-v1.md`, `self-reasoning-substrate-v1.md` | cite `D-GRAPH-1`, `D-TRUTH-1`, `D-INFER-DEDUCTIONS-RELATION-BLIND` | **0** hits on STATUS_BOARD — never minted or renamed without a pointer | no — recorded |
| `weather-substrate-poc-v2.md` | `D-WXA-*`/`D-WXB-*`/`D-WXC-*` ladder | never on STATUS_BOARD (`grep -c WXA` = 0); `weather-soa-bake-v1.md` §0.1 records it from inside the corpus | no — already recorded there |

### 1c. The gap the board already names, re-measured

`ISS-PLAN-TRACKING-IS-UNENFORCED` (`ISSUES.md:183`, 2026-09-03, OPEN) says
plans can mint D-ids the board never tracks. Measured on the A–K half:
**314 of 559 D-id citations (56 %) across 61 plans have no STATUS_BOARD
row; 26 plans have zero board coverage** (full list in
`exec-runs/plan-inventory-2026-09-07-plans-a-k.md` § Mismatches). The L–Z
half found the same shape but also the house style that makes a raw count
overstate it: `D-<FULL-SENTENCE>-N` is a legitimate in-plan *finding*
citation, not a deliverable row, and the naive regex `D-[A-Z]+(-[A-Z0-9]+)+`
also matches the tail of `E-READ-NOT-GREP` as `D-NOT-GREP` (81 such false
positives across 43 files, filtered). So: "no board row" is evidence of a
tracking gap only after checking whether the id is grammatically a
deliverable. `plan-dids.yml` gates ADDED plans only; the backlog is a
cross-session scope call the workflow itself defers.

`INTEGRATION_PLANS.md` cites 141 distinct `.claude/plans/` paths; **58 of
211 plan files are named nowhere in it** — 20 `3DGS-*` (own sub-index
`3DGS-PLAN-INDEX.md`), 5 `tesseract-rs-*`, 3 `alpha-reason-witness-shader-
field-*`, and 30 singles (list in `exec-runs/…-toplevel-and-board.md`). The
"two dangling cited paths" the agent reported are **not dangling** — they are
cross-repo citations (`tesseract-rs/.claude/plans/pdf-to-text-ocr-v1.md`,
`INTEGRATION_PLANS.md:2018`; `ndarray/.claude/plans/splat-native-ultrasound-
simd-substrate-v1.md`, `:2339`) misread as local. Verification reshaped the
finding; the tag-file keeps the original claim.

## 2. `.claude/plans/` — what is genuinely open, by family

Only plans whose OPEN state is load-bearing for a next session are named;
the two tag-files carry every row.

| Family | Open on the board | Closed / shipped | The one thing a session should know |
|---|---|---|---|
| **V3 substrate** (`v3-substrate-integration-v1` stub → `.claude/v3/`, `bindspace-mailbox-soa-wiring-v1`, `bindspace-singleton-to-mailbox-soa-v1`, `cycle-coherent-soa-snapshot-v1`, `singleton-to-snapshot-nudge-v1`) | D-BSW-0..4 Queued/Blocked; D-SOA-SNAP-1..6 no board rows; D-SNGL-3 "In progress" since 2026-06-13 | shim built (`BackingStoreWrite`, 9 methods, zero callers) | `BindSpace` is RETIRE with **68 crate files / 47 plans / 41 blind** (`SUPERSESSION-INDEX.md`); retirement is D-BSW-4, deliberately BLOCKED — never a worker task |
| **Alpha / rung / witness** (`alpha-channel-rung-overlay-v1` 26 D-ids, 12 on board; `alpha-reason-witness-*` ×4; `rubicon-loco-rung-cognitive-fabric-v1`; `known-unknown-handover-network-v1`; `entropy-closure-causal-ground-v1`; `ew64-witness-unification-v1` 2/18 on board) | D-ACR-2 rail UNMINTED; D-ACR-3 no ontology write path (`mailbox_owner()` zero external callers — survives #1112); D-RLR-0..6 all Queued/HELD; D-ECG-*, D-EWU-2..9 Queued | D-ARW-0 Shipped (cited by 4 plans that all say "PLAN ONLY" — the work landed outside them); D-ACR-1/7/8 Shipped; `alpha`/`alpha_tunnel`/`rung_schedule` in contract (#1112) | the temporal audit (`.claude/temporal/`, #1198) FALSIFIED the migration hypothesis: `SpogTenants`, `AlphaTunnel`, `rung_horizon`, `AlphaMask`, `residue_band` are built, tested and **connected to nothing** — the open item is a first consumer, not a mechanism |
| **MUL / EWA / trust** (`mul-calibration-not-verdict-v1`, `mul-ewa-trust-propagation-v1`, `mul-consumer-build-gate-v1`) | D-MCAL-0..5, D-GATE-1..5, D-MEP-0/1 Queued; F-MUL-6 second half "OPEN — not discharged" (MedCare-rs never built against the head, `mul-consumer-build-gate-v1.md:174`) | `mul-consumer-census-v1` CLOSED; D-MCAL-1/6 Shipped | zero MUL↔EWA wiring exists; F-MEP-0b (sign of Σ) and F-MEP-0d (closed-graph monotonicity) are the STOP gates before any run; `ISS-KANBAN-PLAN-EXIT-HAS-NO-NAMED-ROUTE` (`Plan = 4` reachable by no primitive) is the operator's call |
| **Plasticity / Sudoku** (`epistemic-quadrant-materialization-v1`) | P1–P4 falsifiers unrun; `PlasticityState` inert (11 ALL_HOT sites, nothing consumes the bit); `TD-FORK-CANNOT-CLOSE-WHAT-SINGLES-CANNOT` | G1–G7 GREEN on engineered fixtures (a mechanism demonstrator, not a solver) | `PlasticityEngine` verdict REIMAGINE — tenant 7 already IS the Hebbian counter; §4b refuses "rung = plasticity mode" (a matrix, not a mapping) — see the rung/band/plasticity entry |
| **Signatures / sigker / jc** (`pillar11-signature-certification-unification-v1`, `jc-pillars-runtime-wiring-v1`) | W5 PowerSig DEFERRED by its own trigger (4609 in-tree vs ~11 585 memory-half, ~33 388 time-half) | W0–W4 SHIPPED; pillar11 plan CLOSED | two "Pillar 11"s (jc `hambly_lyons` uniqueness vs ndarray `hpc::pillar::signature` stability) + `jc::solver_order` as a 13th battery outside the registry; the SIMD W1a/W1b/W1.5 and this plan's W0–W5 are DISJOINT numberings |
| **Mask / nexgen** (`.claude/nexgen/plans/nexgen-mask-histogram-thresholds-v1`, `mask-algebra-revision-read-v1`) | D-NXG-2/6/7/8/9/10/12 Queued, D-NXG-11 Blocked (σ_step placeholder); D-MAR-2 blocked on an operator ruling (§5) | D-NXG-1/4/5 shipped (`planner/src/nested_bands.rs`, E-NXG-21); D-MAR-1 Shipped; HIST-1/ROLL-1/FLOOR-1 green with three restatements | budget LEADS entropy by 5 steps (E-NXG-19); `mu+3σ` floor unreachable on a real column (E-NXG-20); rooms 4–27 PROPOSAL; the plan lives outside `supersession_index.py`'s scan path on purpose |
| **R2IL / loco** (`r2il-machine-semantic-contract-v1` 1427 lines, `r2il-bpe-typed-genetic-recombination-v1`, `probe-r2il-live-regfile-v1`) | Q2 unresolved, Q3 an external OGAR-mint gate, Q4 may proceed, Q5 data half in another repo; the 4 recombination operators + contract pass + v3 admission loop unbuilt (`revise_if_minority_wins` is `todo!()`, blocked on D-PERSONA-5) | live-regfile 18/18 GREEN locally; §7 falsifiers ran | the r2sleigh CI workflow has **never executed** (`total_count: 0`, unregistered runner) — every green number is local; Q8 falsified the hex-topology hypothesis (E-Q8: degree-1 ablation matches to 4 decimals at 5.5× less memory); `ogar-r2il` has **zero consumers** in lance-graph (only `OGAR/Cargo.toml` + its own manifest name it) |
| **Persistence / lance** (`lance-convergence-staged-migration-v1`, `persistence-cycle-wal-bootstrap-v1`, `persistence-artifact-backed-commit-v1`) | D-LNC-4/5/6 Queued; D-MBX-A6 upgrade Queued; only Phase A implemented | D-LNC-0..3 Shipped (lance 11 / lancedb 0.38); D-LNC-5a/D-MW-P2 Shipped (#1198) | `LanceCycleWriter::open` has zero callers repo-wide; native lance delta is an optional accelerator, snapshot comparison is the oracle (D-TEMPORAL-2) |
| **Consumers** (`lance-graph-in-{medcare,smb-office,woa}-rs-v1`, `unified-bridge-consumer-migration-v1`, `ogar-sink-in-and-consumer-bridge-removal-v1`, `super-domain-rbac-tenancy-v1` 40 D-ids) | Drafts from 2026-05-25 never advanced; D-SINK-2/3/5 unconfirmed; D-V3-W5a..i all Queued | the six `*Bridge`s are `#[deprecated]` aliases over `UnifiedBridge<P>` — that half is DONE | smb-office-rs `LanceConnector::upsert` is still the ONE live orphan online write (W5f); `ClassRbac × ClassView × WideFieldMask` is unenforced across nine repos (D-OIF-1-DEC withdrawal) |
| **Odoo / OGIT / ontology** (`odoo-*` ×6, `lance-graph-ontology-v5`, `normalized-entity-holy-grail-v1`, `ogit-*`) | D-ODOO-SAV-5a..e Queued; 14/15 D-ONTO-V5 unconfirmed; D-CASCADE-V1-1..15 "plan, not implementation" | `odoo-source-extraction-v1` CLOSED (48/53 entities, 5 documented exemptions); `ogar-vocab-contract-codebook-migration-v1` CLOSED | `ogar-ar-shape-endgame-v1` Inc5 "F5-real" (one executor pair doing a REAL write) is the DEFERRED half that keeps the doctrine's §10 litmus at CONJECTURE |
| **Weather / OSINT / 3DGS / OCR** | weather D-WXS-0 Blocked on a classid mint, rest Queued; 22 3DGS docs untracked by design; D-OCR-51/52/53 unstarted | `substrate-comfort-zones-v1` CLOSED (hypothesis REFUTED — a result); weather POC v2 current | the `tesseract-rs-*` plans read OPEN here only because the real progress lives in the sibling repo's own `CLAUDE.md` (image→text byte-parity CLOSED there) — cross-repo status is not on this board |
| **Temporal / Markov / styles** (`temporal-markov-and-style-classes-v1`, `dtsc1-thinkingstyle-dedup-spec-v1`, `triangle-tenants-gestalt-separation-v1`, `thinking-engine-harvest-closure-v1`) | D-MTS-1/2/3 Queued — so the VSA→stream migration the 2026-07-10 ruling gates has not started; D-TSC-2/3, D-TRI-2..5 Queued; W0 census done, 51-file fate assignment not started | D-MTS-6 GREEN (k\*=1), D-TSC-1 resolved (M9), D-TRI-6 In PR | `VISION.md` §8 road (D-MTS-1 → D-TTV-1 → D-MTS-6b → …) is at its **first rung** |

## 3. `.claude/v3/` — waves, ledger, open items (21 files, all read)

| Wave | Verdict | Evidence |
|---|---|---|
| W0 ratify/document | **CLOSED** | D-V3-W0a/b Shipped (STATUS_BOARD:894-895); the folder exists as promised |
| W1 envelope + ownership (the keystone) | **CLOSED mechanism / PARTIAL adoption** | batch writer, delegation cache, probes SHIPPED (Addendum-15, INTEGRATION-PLAN:730-761); a real production caller exists (`cycle_driver.rs:516`, D-MBX-A6-P3c/P4) — but the board's own honesty ledger (STATUS_BOARD:1665): actor-owned production wiring NOT proven, durability FAKE until `LanceShardSink` |
| W2 kanban executors | **PARTIAL / CONTRADICTORY** | arm #1 Shipped (D-MBX-A6); W2a stated two ways (Addendum-12a vs -15, §1b); W2b's actor half DELETED 2026-08-05 — the wave-table row would send a session to rebuild the retired path |
| W3 compiled templates | **PARTIAL, mostly OPEN** | W3a `StepMask` shipped; W3b/c/d Queued; `compiled-templates.md` said StepMask did not exist (struck) |
| W4 DTO ladder | **PARTIAL** | D-PERT-1 Shipped (#630); W4a "In PR" as of 2026-07-10 (not re-verified in source); W4b Queued |
| W5 consumer adoption | **OPEN** | nine sub-items, none Shipped; smb-office-rs `LanceConnector::upsert` still the orphan write (W5f) |
| W6 monitor + retirement | **OPEN, premise RESCINDED** | D-CCF-4 rescinded 2026-07-03; W6a counting logic shipped, Lance-dataset sweep residue; "post-P4" now means post-checkpoint (⊘ notes landed) |

`ENTROPY-MILESTONES.md` (M1–M27 after this PR): shipped/resolved M2 (regraded
here), M5, M6, M7, M9 (partially reopened 2026-07-18), M15, M25, M26; in-flight
M3, M4, M12, M13, M17, M20, M24 (regraded here); queued M8, M10, M11, M14, M16,
M19, M21, M22, M23, M27 (mechanism shipped, callers queued); **M18 RULING-NEEDED**
(sigma chain Ω→Δ→Φ→Θ→Λ vs `KanbanColumn` six phases — no ruling found).

`knowledge/persona-vs-rung-ladder.md` O1–O9: **all open** except O5 (a probe
run; its two hardenings unconfirmed). O2 re-scoped 2026-08-30 to "any-rung
orchestration → `recipes::Recipe`"; O9 regraded StyleFamily's rung-4 anchoring
as a scalar-era artifact. `VISION.md` §8's dependency road (D-MTS-1 → D-TTV-1 →
D-MTS-6b → D-MTS-2/3 → D-TSC-2/3 → "the compilation loop closes") has **none of
its first three rungs shipped**.

## 4. Harvest check — `.claude/v3/` did NOT have it; nine mirrors landed

"Harvest" carries five senses in this workspace (ruff SPO producer; the
domain corpora it produces; cross-repo pattern-transfer briefs; the
literature/idea sweeps; and a same-word collision — see §7). The V3 folder
integrates sense 1 first-class (`knowledge/multi-anchor-ast-resolution.md`,
README row) and cited **neither** `.claude/nexgen/` (2026-09-05, four reader
reports + seven PR sweeps + the mask-histogram plan) **nor**
`.claude/knowledge/literature-harvest-2026-09-01-post-1132.md` — although
`NestedBands` is keyed `(classid, version, idx)` (the V3 keyspace) and the
nexgen plan's room 27 proposes it as the carrier primer §6 demotes
`Vsa16kF32` from. Every "nexgen" string inside `.claude/v3/` meant the
`openproject-nexgen-rs` consumer.

Landed in this PR: README doc-map row + collision note + two rulings added to
the canonical list; primer §5 ⊘ (D-CCF-4) + §6 row
(`E-EVERYTHING-WIRES-TO-SOA-V3-CE64-IS-ALU-LEGACY-1`, which postdated every
mirror in the folder); INTEGRATION-PLAN W6 ⊘; routing.md §5 ⊘;
compiled-templates `StepMask` struck; ENTROPY M2/M24 regrades + M27;
COMPONENT-MAP §6 `NestedBands` row; FUTURE-DESIGN 2026-09-07 block;
witness-nibble-lane P5 (lit-harvest D3: "a single `u8:u8` rail read as ONE
scalar axis is d=1", `:104`); the nexgen plan links back. **Deliberately not
added:** a `tenants.md` row — no D-NXG consumer sits in a `ValueTenant` yet,
and a row without a lane is a wish (ENTROPY's own meta-rule).

## 5. W-waves + jc — verified against code (ndarray `master` `b9afcb9b`)

| Item | State | Note |
|---|---|---|
| W1a #1 `from_i4_packed_u64` / `lane_i8` / `batch_packed_i4_16` | SHIPPED ×4 backends (scalar/avx512/neon + wasm partial) | TD said Open — fixed |
| W1a #2 `saturating_abs` (I8x16/I8x32) | SHIPPED ×4 | TD said Open — fixed |
| W1a #3 `gather_u16` / `palette_lookup_u8x8` | SHIPPED — **API only**; x86 body is a scalar-loop polyfill by its own doc (`simd_avx512.rs:2833-2836`) | performance intent unmet |
| W1a #4 `prefetch_read_t0/t1/t2` | SHIPPED ×3 | TD said Open — fixed |
| W1a #5 `U64x8::popcnt` / `xor_popcount` / `U64x4::popcnt` | SHIPPED (scalar/avx512/avx2) | TD said Open — fixed |
| W1b holograph `hamming.rs` | Open, 25 raw lines | — |
| W1b blasgraph | `types.rs` **0** (done, own path) / `ndarray_bridge.rs` **66** incl. the exact `_mm512_popcnt_epi64` #5 replaces | TD regraded half-done |
| W1b bgz17 `simd.rs` + `prefetch.rs` | Open, 22 + 1 | both gathers still raw; aarch64 `_prefetch` already a no-op |
| W1b contract `mul.rs::i4_eval` | Open, **64** (P0 TD-SIMD-SWEEP-W4) | untouched by #1/#2 |
| W1b thinking-engine | Open, 1 (`is_x86_feature_detected!`, `engine.rs:508`) | matvec already via `simd_amx` |
| W1.5 #6 `signature_pde_sweep` | SHIPPED (PR #293), consumed by `sigker::signature_kernel_pde` | — |
| W1.5 #7 `randomized_signature_sweep` | SHIPPED (PR #294), **consumed** (`sigker/src/randomized.rs:49,124`) | TD said Deferred — fixed |
| W1.5 #8 Lyndon pack | unbuilt, **ungated** (scalar prerequisite `sigker/src/log_signature.rs`, #1150); not a lane op by shape | TD note added |

Raw-intrinsic census across the five named crates: **179 lines** (the
knowledge doc's "158-violation finding" of 2026-05-16 is stale as a live
number — drift from growth, not remediation; `types.rs` is the one real
reduction). **W1b verdict: 0 of 6 closed** with all five primitives shipped.

jc: the 12-entry registry is literally "11/12 implemented, Pillar 2 deferred"
(`lib.rs:24-29`) — with the unstated caveat that Pillar 11 (`hambly_lyons`)
returns `PillarResult::deferred` under the crate's own `default = []`
(`hambly_lyons.rs:737-745`; `Cargo.toml:26,30`), so a plain `prove_it` shows
**10 executing + 2 deferred**. There is **no Pillar 6** in jc (1,2,3,4,5,5b,
7,8,9,9b,10,11); the board's "Pillar-6/7" rows (D-NXG-11, D-MEP-0) name a
different document's pillars. `jc::solver_order` is a 13th battery outside
`run_all_pillars()`. Four example files are present with no `[[example]]`
entry (`l9_loci_real_text`, `partof_isa_vs_palette256`,
`rung_divergence_reliability`, `weather_substrate_reliability`). W5's
PowerSig trigger is legitimately unfired: longest in-tree 4609 points vs
memory-half ≈ 11 585 (`√(2³⁰/8)`, recomputed) and time-half ≈ 33 388 (prior
run, not re-run here).

## 6. Top-level `.claude/*.md` + dashboards

- **71 files**; `CLAUDE.md:762` says 61 and `:765` cites `SESSION_CAPSTONE.md`,
  which does not exist. **9 orphans** (zero refs, no marker, not in
  CLAUDE.md's lists): `CALIBRATION_REPORT_2026_04_03`, the five-file
  `LANGGRAPH_*` cluster (a coherent Python→Rust parity doc-set),
  `WIKIDATA_EXTRACTION_PLAN`, `probe_m1_result_2026_04_11`,
  `prompt-for-other-session-additive`. 61/71 predate 2026-06-01; only
  `BOOT.md` (62 refs) and `ATTENTION_MASK_AUDIT_2026_08_21.md` are recent.
  `pattern.md` (17 refs) and `patterns.md` (30 refs) are two live entry points
  to one SoA/DTO doctrine.
- **STATUS_BOARD.md**: 1854 lines, 89 sections, 592 rows in 3/4/6-column
  shapes; 427 cells carry narrative rather than a lifecycle token — a per-row
  mechanical census is not possible without normalizing the table shape.
- **TECH_DEBT.md**: 167 entries under three header conventions; 65 unlabeled
  (all pre-Kanban-format, `TD-F10-ACTOR-ID`..`TD-INT-14`); open P0s after
  this PR: `TD-SIMD-SWEEP-W4`, `TD-API-DRIFT-MIDFLIGHT-1`,
  `TD-SDR-PR-FOLLOWUP-1`, `TD-SDR-CONSUMER-PUSH-1`, `TD-OGIT-G-SLOT-1`.
- **ISSUES.md**: 47 entries, 37 OPEN. The ones this inventory touches:
  `ISS-PLAN-TRACKING-IS-UNENFORCED`, `ISS-KANBAN-PLAN-EXIT-HAS-NO-NAMED-ROUTE`,
  `ISS-RUNG-VS-BAND-CARDINALITY-COLLISION`, `ISS-REASONING-BAND-GATES-NOTHING`,
  `ISS-EXCLUDED-CRATES-UNBUILT`, `ISS-SUPERSESSION-GENERATOR-DID-RANGE-NARROW`.
  `ISS-CLASSID-OGAR-DRIFT` has two dated sub-entries (OPEN and RESOLVING).
- **SUPERSESSION-INDEX.md**: 14 ruled symbols, 73 blind plans (55 RESCOPE /
  18 READ / 0 ARCHIVE?); `BindSpace` RETIRE with 68 crate files / 47 plans /
  41 blind.

## 7. Name collisions a session must read past

| Word | Meaning A | Meaning B |
|---|---|---|
| **v3** | `.claude/v3/` mailbox-kanban-facet substrate | version 3 of the *retracted* `epistemic_bassin` 24-axis basis (`0x0334`) in `nexgen/harvest/14-*` and the literature harvest |
| **nexgen** | `.claude/nexgen/` (2026-09-05 harvest + plan) | `openproject-nexgen-rs` (consumer repo) — the only sense inside `.claude/v3/` until this PR |
| **Pillar 6 / Pillar 11** | jc registry: no Pillar 6; Pillar 11 = Hambly–Lyons uniqueness | board rows "Pillar-6/7" = another document's numbering; ndarray `hpc::pillar::signature` = a second "Pillar 11" (kernel stability) |
| **W1 … W5** | SIMD waves W1a/W1b/W1.5 (`ndarray-vertical-simd-alien-magic.md`) | `pillar11-signature-certification-unification-v1` W0–W6 (W1/W4 live in `ndarray/crates/sigker-parity`) |
| **harvest** | five senses (§4) | — |
| **EWA** | three different operations share the word (lit-harvest #27; jc sandwich = rendering push-forward) | — |
| **TrustTexture / GateDecision** | four `TrustTexture` types; two `GateDecision` types (M15 resolved the planner one) | `causal-edge/src/layout.rs` rules against a cast between them |

## 8. Synergy — the digest (full text: EPIPHANIES 2026-09-07 block)

Every open blocker in the SPOG/alpha/medcare/DataFusion arc is **read-time
re-derivation of something decided once upstream**: the ontology category
re-read from `value[96]` per row (medcare `obo_store.rs:16-18,77,681`),
`Domain::of_row` (`orphanet.rs:165`, `rails.rs:432`), `FacetRegime::PerRowTui`
(`domain_block.rs:259,695`), `graph_of(addr)` resolved per claim by a linear
scan (`spog_tenants.rs:38,87`), V1 external-id identity, and Lance row identity
re-materialised by `with_row_id`/`with_row_addr` (medcare `state.rs:900,936`).
The operator's sentence — *"every domain is just another table keyed by CUI
STDID (snomed) loinc etc — its a chain effect with masking, no datafusion
joins ever"* — is the schema-level form of the mask plan's invariant
(`substrate == mask geometry == projection surface`): one table per G, the
quad's 4×24 slots = four pre-resolved foreign keys, a crosswalk = a chain of
`eq_u32_to_mask` sweeps conjoined by `mask_ternlog` where the mask-out of hop
n is the key set of hop n+1 — never a mask-AND across two tables (nexgen room
18's family-separation rule is the mechanical "no joins"). The measured bounds
ride with it: masks win on Boolean relations at every density (D-GTM-0j — a
TYPE boundary), 0 bytes/step (0k), chaining pays only while masks stay in L2
and survivors stay above 0.1 % (0n), and packed prefixes do NOT carry
cross-domain relations (0l) — the FK columns do. The seven harvest sources
collapse onto one object: the version-keyed `NestedBands` (+ overlap matrix)
per `(classid, version)` — reveal-ahead / cache-key = Lance version; Shannon =
WHERE only and it lags the popcount budget; EWA ranks WHERE to look, never
WHAT is true; a known unknown = a hop whose survivor mask has popcount > 1;
Boolean → masks, valued → blasgraph semirings over CSR. And the fence: rung
(processing horizon, 0–9) × ReasoningBand (assertion permission, bits 61..63)
× PlasticityState (how content changes, 3-bit S/P/O) are three axes; folding
any two into one "level" field is a LAYOUT-BREAK-class defect.

## 9. What to do next (an order, not a plan — each step names its gate)

1. **SPOG-aware bakes — one artifact per G** (medcare): the domain becomes
   the table, `value[96]` stops being a read. Gate: PROBE-CROSSWALK-MASK-1
   (IDEAS 2026-09-07) — survivor set == the DataFusion join's set on one
   fixture; 0 bytes/step; cross-family AND rejected at the seal.
2. **V3 lane-local addresses** — retire V1 external-id identity
   (`E-A-V3-MINT-MUST-NEVER-DEGRADE-TO-V1-1` already bangs on the mint side).
   Gate: D-TEMPORAL-1a's V3-accessor pin stays green.
3. **`ogar-r2il` gets a consumer** through `lance-graph-ogar` (`RANK` +
   `TERNLOG 0x86`); today it has none. Gate: a lifted crosswalk program
   equals the mask chain bit-for-bit; attaches to D-R2IL-5, not a new owner.
4. **DataFusion containment** — `with_row_id = false`, `with_row_addr = false`
   pinned by a test that fails if either flips; no new DataFusion surface
   (`E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1`).
5. Board hygiene the inventory could not do alone: adjudicate W2a
   (Addendum-12a vs -15) from `canonical_node.rs:1622+`; rule M18; decide
   `ISS-KANBAN-PLAN-EXIT-HAS-NO-NAMED-ROUTE`; retrofit the 65 unlabeled
   TECH_DEBT rows or declare them open by policy; normalise STATUS_BOARD's
   three table shapes so a mechanical census becomes possible.
