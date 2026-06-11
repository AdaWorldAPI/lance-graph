# 2026-06-05 session — handover to next session

> **Session:** `claude/lance-graph-ontology-review-Pyry3` (Opus 4.7 / 1M context, main thread).
> **Source of truth:** this file. The `.claude/board/AGENT_LOG.md` entries are also live and authoritative.

## Read first (3 minutes)

If you are the next session opening this workspace, read in this order:

1. `.claude/BOOT.md` — workspace mandatory reads (board files).
2. This file — what shipped + what's outstanding from the 2026-06-05 session.
3. The four "outstanding decision" tables below — each names a specific PR or session you may need to coordinate with.

## What shipped this session (14 PRs across 4 repos)

Two cross-workspace arcs landed, fully clean end-to-end (canonical plan → companion stubs/handovers → reviewer fixes → ADR adoption callouts).

### Arc 1 — splat-native-ultrasound-v1 (CPU-only Gaussian-splat ultrasound SaMD)

| PR | Repo | What | Status |
|---|---|---|---|
| #471 | lance-graph | Canonical plan + 14 D-SPLAT-* deliverables + Q1/Q2 OGAR coordination | merged |
| #472 | lance-graph | Reviewer fix arc on #471 (5 fixes) | merged |
| #212 | ndarray | SIMD substrate plan (D-SPLAT-2 — 5 W1c primitives) | merged |
| #213 | ndarray | Reviewer fix arc on #212 (ray-segmentation + Cholesky-scratch) | merged |
| #163 | MedCare-rs | HIPAA wire handover (D-SPLAT-10/11) | merged |
| #164 | MedCare-rs | Reviewer fix arc on #163 (pose_se3 16→24) | merged |
| #34 | OGAR | §6 FMA-litmus customer narrative | merged |
| #35 | OGAR | Reviewer fix arc on #34 (IVD-MDR → MDR Annex VIII Rule 11) | merged |
| #476 | lance-graph | D-SPLAT-4 ADR-024 adoption callout | merged |

### Arc 2 — cesium-osm-substrate-v1 (OSM as 6th source class for 3DGS-ArcGIS-Cesium ingestion)

| PR | Repo | What | Status |
|---|---|---|---|
| #473 | lance-graph | Canonical plan + 7 D-OSM-* deliverables + Q1/Q2/Q3 OGAR coordination | merged |
| #474 | lance-graph | Other session: D-OSM-2 ownership-boundary fix on #473 | merged |
| #214 | ndarray | D-OSM-1 stub (mirrors `arcgis_pbf.rs`; live Q3 Y-flip helper) | merged |
| #475 | lance-graph | §11 ADR-024 adoption callout + CR Major fixes (prefix unification + coordinate-fidelity metric) | merged |

### ADR-024 (palette256 + HHTL codec) — companion to both arcs

OGAR session shipped **ADR-024** (`OGAR/docs/ARCHITECTURAL-DECISIONS-2026-06-04.md`) as the universal compression primitive across security perms (Binary16K `_effectiveReaders`), attention (bgz-tensor WeightPalette), distance (arm-discovery aerial codebook ρ=0.9973 vs cosine), OSM tag palette (D-OSM-2), and SH coefficient palette (D-SPLAT-4). Empirical floor: **ρ ≥ 0.99** per adopting domain.

## Outstanding decisions (these need someone — you, the user, or another session)

### Decision 1 — substrate-addressing-v1.md plan (mine to file when unblocked)

Third session proposed three P0 substrate-of-substrate deliverables:
- **D-HELIX-1** — `helix::bounds(nibblepath)` + `helix::centroid(nibblepath)`. Owner: helix-crate owner (third session).
- **D-CESIUM-1** — extend `crates/cesium/src/implicit_tiling.rs` to consume `helix::bounds` as a tiling backend. Owner: runtime (me).
- **D-JC-1** — `jc::predict_lod(scene_cert, tolerance) → level`. Owner: jc-crate owner.

I committed to file the consolidating `substrate-addressing-v1.md` plan **after** D-HELIX-1 and D-JC-1 owners bless their pieces. Greenlit but not yet filed. **Trigger:** ping from helix or jc owner.

**Update 2026-06-05 (post-handover-doc-write, after surveying #471-#487):** D-HELIX-1 substantially unblocked. PR **#485** (`claude/probe-mantissa-fill`, **still OPEN**) lands **PROBE-MANTISSA-FILL + PROBE-PHASE-1, both GREEN** against the SHIPPED helix code:
- Golden-mantissa placement beats seeded uniform-random by ~30% at k=256 (192/3 vs 141-150/5-6) and at k=1024 with half the worst-case pile-up. The "golden mantissa places implicit centroids" leg of the substrate-addressing claim is now **measured, not asserted**.
- `CurveRuler` regeneration is **bit-exact** across independent constructions over 20 `(path, depth)` pairs incl. `u64::MAX`; stride-4-over-17 arc is full permutation from every one of 17 start offsets. Same address ⟹ same sequence is the deterministic-phase pin's empirical proof.

PR **#482** (`docs(knowledge): GUID canon + prefix routing`, **merged**) lands two architectural pins that bind any future `substrate-addressing-v1.md`:
- **TWO-ALGEBRA RULE** (load-bearing): sign = XOR (`vsa_bind`); magnitude = `vsa_bundle`, **NEVER `MergeMode::Xor`** (breaks Markov per `I-SUBSTRATE-MARKOV`). Cited as PP-13 P1-1 — permanent CI guard.
- **Helix `CurveRuler` stride-4-over-17** is the canonical bit-exact integer phase generator; analysis remainder escalates per the quorum certificate.
- Probe ledger: HILBERT-L4 (ndarray #215) · PHASE-1 (#485) · MANTISSA-FILL (#485) all GREEN; remaining: ROUTE-1 · QUORUM-1 · PHI-1 · PYR-1 · PERT-RHO · CODEBOOK-44 · WHP-1..4 · PROBE-ATTN-EDGE · PROBE-SPLAT-PSD · PROBE-CASCADE-SPARSITY.

**New trigger for substrate-addressing-v1.md filing:** after #485 merges. The two-algebra rule + bit-exact phase generator + measured golden-mantissa placement are now load-bearing receipts the plan can cite verbatim. The D-CESIUM-1 (mine — extend `implicit_tiling.rs` to consume `helix::bounds`) becomes the file-soonest deliverable; D-JC-1 still needs per-scene-class `p` calibration timeline from jc-crate owner.

### Decision 2 — Tier-1 cross-session asks (status table)

From my coordination message earlier this session. Status as of session close:

| Ask | From | Status |
|---|---|---|
| FMA Phase 8 implementation opens (TTL → FmaEntity SoA emitter on top of OGAR #37's parser) | OGAR | NOT YET — TTL parser exists (#37), FMA-specific lift not yet written |
| NiblePath identity-prefix scheme for FMA classes | OGAR | **RESOLVED (post-handover survey).** lance-graph #484 ships `D-IDENTITY-2: entity_type↔NiblePath bijection` in `lance-graph-ontology` registry — `register_class_path` is bijective + dedup-by-URI + conflict-rejecting + u16-overflow-guarded; +5 tests; 14 registry tests green. OGAR Phase 8 calls `register_class_path("fma/Class/<name>", entity_type)` per FMA class. The `NodeGuid` layout change (groups 3-4 reclaim for HIP/TWIG) is the version-gated Phase-B follow-up; not blocking for FMA registration. |
| FMA region partitioning scheme for Lance row-groups | OGAR | NOT YET |
| MailboxSoAHeader format frozen? | lance-graph contract owner | UNCLEAR — D-SPLAT-1/3 carriers inherit this |
| bgz17 SH palette extension blessing (D-SPLAT-4 + ADR-024) | bgz17 owner | NOT YET — needed before D-SPLAT-4 implementation |
| D-OSM-4 batched_sample_height W1c primitive | ndarray SIMD-savant | NOT YET — needed before D-OSM-5 |
| No breaking RedactionMode rework queued? | MedCare-rs | NOT YET CONFIRMED — D-SPLAT-10 / D-OSM splat-fit depend on existing set |
| 3DGS-ArcGIS-Cesium parent-plan owner blessing for cesium-osm as 6th source class | lance-graph plan owner | NOT YET CONFIRMED |
| D-OSM-6 (`cesium-3dtiles-writer` crate) ownership | UNASSIGNED | NOT YET — genuine Rust gap; no existing session owns it |
| ndarray PR #189 `OntologySchema::is_ancestor` measured on real FMA (~75K) | ndarray PR owner | NOT YET — §6 acceptance gate hangs on this |

### Decision 3 — User ratification owed (before any P1 sprint implementation opens)

- **OQ-SPLAT-1..4** (probe SDK / SH degree / beamformer scope / AR egress)
- **OQ-OSM-1..5** (Tag canonicalization / DEM source / writer scope / planet-scale ingest / coordinate-policy)

Default proposals are in the respective plans. No implementation PRs should open without these.

### Decision 4 — Unified-soa post-merge addendum (PR opened today)

Resurfaced from the early-session Pyry3 branch during today's rebase pass. **PR opened against lance-graph main** as the resolution path (alternative was to drop or leave-on-branch — both worse per board hygiene rule). Reviewer choice in the PR body: ship as-is OR scope down to plan-file edits only (board entries are 80+ commits stale).

## Mechanics — git state at session close

| Repo | Pyry3 state | Pushed? |
|---|---|---|
| lance-graph | rebased onto `5363f436`; 1 commit ahead (`5b7e3163` unified-soa addendum, now PR'd) | force-with-lease pushed ✓ |
| ndarray | local rebased to `cb77a31` (= origin/master); push rejected as stale (another session active on the branch) | local-only; nothing of mine to push |
| MedCare-rs | rebased onto `b074aa3`; clean fast-forward | force-with-lease pushed ✓ |

ndarray Pyry3 has another session actively pushing — coordinate via that session's handover rather than force-pushing over them.

## Architectural pins from this session (cite by name in future work)

- **E-SOA-IS-THE-ONLY** — one substrate, three operations. Splat-native + cesium-osm both incarnations.
- **ADR-024** (palette256 + HHTL codec) — load-bearing across security / attention / distance / OSM / SH. Empirical floor ρ = 0.9973 (arm-discovery anchor); target ≥ 0.99 for new adopters.
- **NR-SPLAT-PHI** (in `splat-native-ultrasound-v1.md §3.10`) — scanner-frame splat geometry is non-identifying on its own; `patient_ref ↔ splat_volume` link IS PHI; raw RF/IQ never persisted.
- **Q3 OSM-XYZ → TMS Y-flip** (in `ndarray/crates/cesium/src/osm_pbf.rs::xyz_to_tms_y`) — single boundary helper; runtime sees only TMS.
- **MDR Annex VIII Rule 11** (correct regulation for ultrasound SaMD; NOT IVDR) — pinned across splat-native + OGAR docs.

## Pointer to next action

If you have nothing else queued, the cheapest productive next step is to **ping the helix-crate owner and jc-crate owner** so Decision 1 unblocks. Once D-HELIX-1 + D-JC-1 are blessed, file `substrate-addressing-v1.md` and the three-deliverable cross-arc unification lands.

_End of handover._
