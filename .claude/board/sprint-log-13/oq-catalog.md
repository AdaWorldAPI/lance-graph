# Sprint-13 OQ Catalog (OQ-CSV-7..N)

**Status:** PRE-RATIFICATION. PP-11 Opus planner enumeration. User ratification required before sprint-13 spawn.
**Authored:** 2026-05-16 (claude/sprint-13-preflight-planning, PP-11).
**Predecessor:** OQ-CSV-1..6 in `.claude/plans/cognitive-substrate-convergence-v1.md` §14 / `-v2.md` §12 (all RATIFIED or DEFAULT-APPLIED per `i4-substrate-decisions.md`).
**Convention:** OQ-CSV-N entries below extend the v1/v2 gate table. Recommendation column is the Opus planner's pre-ratification preference; final ratification belongs to the user at sprint-spawn time.

---

## Index

| # | Topic | Blocks | Sprint-spawn blocker? |
|---|---|---|---|
| OQ-CSV-7  | PP-3 rayon feature gate name              | D-CSV-17       | YES |
| OQ-CSV-8  | PP-3 par_* iteration chunk size           | D-CSV-17       | YES |
| OQ-CSV-9  | PP-4 splat_field carrier (Think vs sibling) | D-CSV-14     | YES |
| OQ-CSV-10 | PP-4 splat generation source               | D-CSV-14      | YES |
| OQ-CSV-11 | PP-5 SPO -> 256D adapter (VSA bind vs one-hot) | D-CSV-16  | YES |
| OQ-CSV-12 | PP-5 WitnessCorpus cam_pq lazy vs eager    | D-CSV-16      | YES |
| OQ-CSV-13 | PP-6 SIMD i4 runtime vs compile-time dispatch | D-CSV-13b  | YES |
| OQ-CSV-14 | PP-6 bench speedup floor (SHIP vs LAND)    | D-CSV-13b ship | YES |
| OQ-CSV-15 | PP-8 worker-template-v2 workspace member edit ownership | all sprint-13 worker prompts | YES |
| OQ-CSV-16 | Governance: E-META-10 + iron-rules-doctrine in BOOT.md Tier-1 | sprint-13 worker readiness | YES |
| OQ-CSV-17 | (14+) ndarray PR #116 hpc-extras coordination ownership | D-CSV-13 cross-repo handoff | NO (sprint-14+) |
| OQ-CSV-18 | (14+) VAMPE+Jirak coupled-revival threshold derivation scope | D-CSV-15 | NO (sprint-14+) |
| OQ-CSV-19 | (14+) Splat field persistence to Lance vs in-memory only | D-CSV-14 follow-on | NO (sprint-14+) |

Sprint-13 spawn blockers: **10** (OQ-CSV-7..16). Sprint-14+ tracked: **3** (OQ-CSV-17..19).

---

## OQ-CSV-7 — PP-3 rayon feature gate name

**Question:** When PP-3 introduces `par_*` parallel iteration streams on the i4 substrate, what is the rayon feature gate name on the consuming side — `parallel` (ndarray's convention) or `rayon` (rayon's own crate name)?

**Why it matters:** Cross-repo consistency. ndarray downstreams (cognitive-shader-driver, lance-graph-contract) already wire ndarray's `parallel` feature. A mismatched name on i4-substrate forces every consumer to spell two gates, inflates Cargo.toml diff churn, and breaks the "one feature, one capability" pattern from CLAUDE.md doctrine.

**Recommendation:** **`parallel`** — match ndarray's existing feature naming (see `/home/user/ndarray/Cargo.toml`). Rationale: every existing parallel consumer already opts into `parallel`; aligning preserves the cargo feature unification rule and avoids a feature-fork in the dependency graph.

**Blocks:** D-CSV-17 (PP-3 par_* row sweep / SoA stream landing).

---

## OQ-CSV-8 — PP-3 par_* iteration chunk size

**Question:** Should the par_* iteration chunk size on i4 row sweeps be auto-detected from cache-line geometry at runtime, or fixed at a row-size-aligned constant?

**Why it matters:** Auto-detection adds a one-time probe cost and a `LazyLock<usize>` static per stream; fixed gives a deterministic bench baseline and removes a tuning variable from D-CSV-13b SIMD work. Wrong chunk size = false sharing on the QualiaI4 cache line and a 2-3× regression versus scalar.

**Recommendation:** **Fixed.** **8 rows for QualiaI4** (8 × 8 bytes = 64 B = one cache line), **4 rows for SplatField** (4 × 16 bytes = 64 B = one cache line). Rationale: cache-line aligned by construction; trivially scalar-replicable in the SIMD path; deterministic in benches. Auto-detect can be reintroduced as a feature flag if a tuning need surfaces in sprint-14+.

**Blocks:** D-CSV-17.

---

## OQ-CSV-9 — PP-4 splat_field carrier (Think vs sibling)

**Question:** Does the PP-4 splat-field on-Think method migration add a `splat_field: Vec<SplatField>` field directly to the existing `Think` struct, or does it introduce a sibling `Splat` carrier struct alongside `Think`?

**Why it matters:** This is a CLAUDE.md "Thinking is a struct" doctrine call. A sibling `Splat` carrier means free functions get reshaped as methods on a NEW struct — which the iron-rules litmus test (`free-function-on-state -> reject`) rejects as a refactor of the same anti-pattern. Adding to `Think` keeps the canonical carrier and lets all splat ops surface as `Think::splat_*` methods, matching the L-20 surface contract.

**Recommendation:** **Add `splat_field: Vec<SplatField>` to `Think`.** Rationale: `Think` is the existing carrier per CLAUDE.md doctrine; a sibling `Splat` carrier would re-create the free-function-on-state pattern under a new name. Cost is +24 B per Think (one `Vec` header) when splat is unused — acceptable per the doctrine's "carrier integrity" weight.

**Blocks:** D-CSV-14.

---

## OQ-CSV-10 — PP-4 splat generation source

**Question:** Should `SplatField::generation: u32` be derived from `Think::cycle` (the existing monotonic counter), or carry its own `splat_generation: u32` field on `Think`?

**Why it matters:** "Thinking is a struct" doctrine implies a single source of truth for time on the carrier. A separate `splat_generation` invites drift (splat advances but cycle doesn't, or vice versa) and a new invariant ("`splat_generation <= cycle`") that callers must respect. Derivation from `self.cycle` makes the invariant unrepresentable.

**Recommendation:** **Derive from `self.cycle`.** Rationale: single source of truth; no new invariant; cycle already advances on every Think step. If splat needs sub-cycle granularity in sprint-14+, introduce `Think::splat_subcycle` then — but until a use case forces it, prefer the simpler model.

**Blocks:** D-CSV-14.

---

## OQ-CSV-11 — PP-5 SPO -> 256D adapter (VSA bind vs one-hot)

**Question:** For the PP-5 WitnessIndexCamPq integration, how is an SPO triple lifted to the 256-dim CAM-PQ vector — Option A: VSA bind of role-keyed fingerprints (`bind(S_key, s) XOR bind(P_key, p) XOR bind(O_key, o)`), or Option B: one-hot block encoding (84 dims subject, 84 dims predicate, 88 dims object)?

**Why it matters:** This is the I-VSA-IDENTITIES "register-loss problem" surface (see iron-rules-doctrine). One-hot loses role information under any pooling/superposition and forces the cam_pq index to memorize exact triples instead of generalizing across role-structure. VSA bind preserves role-keyed distinguishability under superposition — the property that lets a single 256D query match multiple witness rows by structural similarity.

**Recommendation:** **Option A (VSA bind of role-keyed fingerprints).** Rationale: per I-VSA-IDENTITIES iron rule, role information must survive superposition; one-hot fails this litmus. Role keys (`S_KEY`, `P_KEY`, `O_KEY`) live as constants in `lance-graph-contract::vsa::roles`; fingerprints come from existing `LanceGraph::node_fingerprint`. Cost: 3 XOR operations per SPO at insert time — negligible vs. the cam_pq quantize step.

**Blocks:** D-CSV-16.

---

## OQ-CSV-12 — PP-5 WitnessCorpus cam_pq integration: lazy vs eager

**Question:** Is `WitnessCorpus`'s cam_pq index built eagerly in `WitnessCorpus::new()`, or lazily via a `WitnessCorpus::enable_cam_pq()` opt-in?

**Why it matters:** Eager forces every `WitnessCorpus` constructor to pay the ndarray + cam_pq dep cost (compile time + binary size + a one-time index allocation). Many callers use `WitnessCorpus` purely as a HashMap-backed corpus and never query by similarity — making them pay is a violation of the "pay for what you use" cargo feature norm.

**Recommendation:** **Lazy via `enable_cam_pq()`.** Rationale: keeps ndarray/cam_pq an optional dep behind a feature gate; HashMap-only users see zero cost. Builders that want similarity query call `corpus.enable_cam_pq()` once after construction. The cam_pq state lives in an `Option<CamPqState>` field; query methods early-return `None` if disabled.

**Blocks:** D-CSV-16.

---

## OQ-CSV-13 — PP-6 SIMD i4 dispatch: runtime vs compile-time

**Question:** For the PP-6 AVX-512 / NEON i4 SIMD landing (D-CSV-13b), is ISA dispatch done at runtime via a `simd_caps()` probe, or at compile time via `target_feature` cargo flags?

**Why it matters:** Compile-time dispatch produces one binary per target — fine for a single-tenant deployment, painful for distributed binaries (one slow path + one fast path = two artifacts). Runtime dispatch matches ndarray's existing pattern (cf. `ndarray::simd::caps`), gives one binary that adapts to the host, and lets benchmarks compare paths in a single run.

**Recommendation:** **Runtime dispatch via `simd_caps()`.** Rationale: matches ndarray's pattern, single binary, bench-friendly. Cost is one cached `LazyLock<SimdCaps>` and a branch per call site (`if caps.avx512 { ... } else if caps.neon { ... } else { scalar }`). Branch predictor handles the cost trivially after warmup.

**Blocks:** D-CSV-13b.

---

## OQ-CSV-14 — PP-6 bench speedup floor (SHIP vs LAND)

**Question:** What is the SHIP-gate (block merge) and LAND-gate (block PR open) relative speedup floor for AVX-512 i4 MUL vs scalar in D-CSV-13b benches?

**Why it matters:** Without a floor, "AVX-512 lands and is slower than scalar" is a possible outcome (false-sharing, vextract overhead, misaligned loads). A SHIP gate forces the bench to pass before merge; a LAND gate keeps WIP PRs honest about progress.

**Recommendation:** **SHIP gate: 4x AVX-512 vs scalar.** **LAND gate: 2x.** Below 2x = block PR; 2x-4x = land with `TD-D-CSV-13b-PERF-FLOOR-1` TECH_DEBT note and follow-up; >=4x = ship. Rationale: 4x is the published expectation (8-wide i4 lanes after vpcompress, ~2x overhead from horizontal sum); 2x is the floor at which "AVX-512 is worth the binary cost at all."

**Blocks:** D-CSV-13b ship (not LAND, per the gate split).

---

## OQ-CSV-15 — PP-8 worker-template-v2 workspace member edit ownership

**Question:** When a sprint-13 worker creates a NEW crate (e.g. `splat-field-types`, `witness-index-cam-pq`), does the WORKER edit the workspace root `Cargo.toml` `members = [...]` list, or does the worker leave it untouched and the MAIN thread edits it post-merge?

**Why it matters:** Sprint-12 Wave G W-G6 demonstrated workers CAN edit workspace members without orphaning crates (the CSI-7 orphan pattern was orthogonal — caused by main-thread oversight, not worker authority). If workers can't edit workspace, every new-crate PR needs a follow-up "wire to workspace" PR from main — doubling the PR count for the new-crate scaffolding waves.

**Recommendation:** **WORKER edits `members =`.** Rationale: W-G6 proved feasible (PR landed clean, no CSI-7 recurrence). Avoids the orphan pattern at root (worker tests the crate against the workspace; if `members =` is wrong, worker's own `cargo test` catches it). Worker template v2 documents the required edit as a checklist item.

**Blocks:** all sprint-13 worker prompts that create new crates (PP-3 splat-field-types, PP-5 witness-index-cam-pq, plus any others surfaced during spec drafting).

---

## OQ-CSV-16 — Governance: E-META-10 + iron-rules-doctrine in BOOT.md Tier-1

**Question:** Should the PP-2 consolidation of E-META-10 (free-function-on-state litmus) + iron-rules-doctrine be promoted to a Tier-1 Mandatory Read in `BOOT.md` and `agents/BOOT.md`, alongside CLAUDE.md and LATEST_STATE.md?

**Why it matters:** Sprint-12 surfaced multiple iron-rule debt entries (`I-NOISE-FLOOR-JIRAK`, `I-VSA-IDENTITIES`) that workers stepped on because the doctrine doc was Tier-2-discovery. Tier-1 promotion makes the litmus test mandatory before any spec-touching or carrier-touching work — the exact class of work PP-3/4/5/6 spawn.

**Recommendation:** **YES, add to BOOT.md Tier-1 trigger table.** Concretely: BOOT.md gains a row like "TOUCHING a struct method, free function on state, or iron-rule debt entry -> READ `iron-rules-doctrine.md`." Cost: +1 read per worker session that triggers; benefit: zero iron-rule violations make it past worker self-review.

**Blocks:** sprint-13 worker readiness (without this, PP-4 splat-on-Think and PP-5 VSA-bind decisions risk being re-litigated mid-worker).

---

## Sprint-14+ candidates (non-blocking for sprint-13 spawn)

### OQ-CSV-17 — ndarray PR #116 hpc-extras coordination ownership

**Question:** When ndarray PR #116 (hpc-extras: i4 helpers, par_* shims) lands upstream, does lance-graph pin to the released crate version, or vendor-fork until upstream stabilizes?

**Why it matters:** PR #116 is a cross-repo handoff; an upstream version churn mid-sprint-14 forces a coordinated bump. Vendor-fork insulates against churn but creates a drift risk.

**Recommendation:** **Pin to released semver `^0.16` once PR #116 merges; vendor-fork only if upstream slips past sprint-14 W2.** Rationale: standard semver discipline; vendor-fork is the escape hatch, not the default.

**Blocks:** D-CSV-13 cross-repo handoff timing.

---

### OQ-CSV-18 — VAMPE+Jirak coupled-revival threshold derivation scope

**Question:** For D-CSV-15 (Σ10 Rubicon Jirak-derived threshold, TD-7 resolution), does the derivation cover only Σ10 (current TD), or all Σk bands (Σ1..Σ10)?

**Why it matters:** Wider scope = larger sprint slot; narrower = TD-7 closes faster but Σ1..Σ9 stay hand-tuned.

**Recommendation:** **Σ10 only for D-CSV-15; open D-CSV-15b for Σ1..Σ9 if needed.** Rationale: keep sprint-13 scope tight; Σ10 is the surfaced TD; lower bands can be derived in a follow-up once the Σ10 calibration validates the methodology.

**Blocks:** D-CSV-15.

---

### OQ-CSV-19 — Splat field persistence: Lance vs in-memory only

**Question:** Once PP-4 splat-on-Think ships (D-CSV-14), does `SplatField` persist to a Lance table (durable, queryable across sessions), or stay in-memory on the Think carrier only?

**Why it matters:** In-memory is simpler and matches Think's session-scoped lifetime; Lance persistence enables cross-session splat replay and audit but adds a schema commitment.

**Recommendation:** **In-memory for D-CSV-14; add D-CSV-14b for Lance persistence if a replay/audit use case surfaces.** Rationale: don't commit to schema before the use case is concrete.

**Blocks:** D-CSV-14 follow-on (non-blocking for the initial ship).

---

## Ratification protocol

1. User reviews OQ-CSV-7..16 (10 sprint-spawn blockers) before sprint-13 worker spawn.
2. For each, user either: ratifies the recommendation, picks an alternative, or defers (with a documented TECH_DEBT note).
3. Ratified decisions land in `.claude/knowledge/i4-substrate-decisions.md` (W-F11 doc) with file:line evidence as PP-3/4/5/6 specs go from draft to in-PR.
4. OQ-CSV-17..19 stay in this catalog as sprint-14+ tracking; no ratification required for sprint-13 spawn.

**Until OQ-CSV-7..16 are ratified, sprint-13 worker spawn is BLOCKED.**

---

## Cross-refs

- Predecessor gate table: `.claude/plans/cognitive-substrate-convergence-v2.md` §12.
- Ratification evidence chain: `.claude/knowledge/i4-substrate-decisions.md`.
- Iron-rule doctrine (PP-2 consolidation target): `.claude/knowledge/iron-rules-doctrine.md`.
- TECH_DEBT entries referenced: TD-SIGMA-TIER-THRESHOLDS-1 (TD-7), TD-D-CSV-8-SIMD-1, TD-D-CSV-13b-PERF-FLOOR-1 (proposed).
- Worker template v2 (PP-8): TBD path under `.claude/agents/`.
