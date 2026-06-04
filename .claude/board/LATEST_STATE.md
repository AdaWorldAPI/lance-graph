# LATEST_STATE — What Just Shipped (read this FIRST)

> **Auto-injected at session start via SessionStart hook.**
> Updated after every merged PR.
> **Last updated:** 2026-05-14 (PR #372 merged: sprint-10 spec sprint, 12-worker CCA2A fleet + Opus meta-review + 8 knowledge docs, governance-only (zero .rs changes), mirrors PR #365 pattern. Sprint-11 implementation wave gated on 5 spec patches + 4 user ratifications: CSI-1 CausalEdge64 bit-reclaim Option, OQ-1 Σ4-Σ5 banding, OQ-3 plasticity granularity, OQ-5 rayon vendor. **Major findings:** (1) dual `CausalEdge64` types in workspace — `causal_edge::CausalEdge64` SPO-palette layout ≠ `thinking_engine::layered::CausalEdge64` 8-channel cascade, same name different semantics, surfaced as duplication entry #13 in TYPE_DUPLICATION_MAP and E-META-7 in EPIPHANIES; (2) p64 drift origin pinpointed at `crates/lance-graph-planner/src/cache/convergence.rs:18-22 #[allow(unused_imports)]` annotation — wiring intended for hot-path convergence never finished; (3) three-zone hot-path mental model corrects prior framing — Zone-1 thinking-engine MatVec 200-500ns + AriGraph entity_index O(1) ~20-200ns is the actual cycle-speed path, not DataFusion. Prior: 2026-05-13 (PR #366 merged: sprint-7 7-worker implementation wave for the sprint-5/6 specs + AuditSink trait unification, ~5 KLOC across 5 crates +2 new (`lance-graph-supervisor`, `lance-graph-consumer-conformance`), ~70 new tests, workspace clippy --tests --no-deps -D warnings exits 0; Opus meta verdict 4A/2B/1B-minus; OQ-7-1/2/3 all locked pre-merge; `UnifiedAuditSink` D-SDR-4 placeholder dropped, all sinks unified on `AuditSink` trait; `UnifiedBridge::with_jsonl_audit()` ergonomic constructor added for MedCare-rs sprint-2 item 5. **Adjacent landings (same day):** MedCare-rs sprint-1 10-PR sweep (#113-#122) including E1-1 OQ-3 direct migration (6 RoleGroups) consuming our `0d725d4` decision. MedCare-rs sprint-2 (5 PRs) is queued on user "go" — item 5 consumes this PR's new constructor. Prior same-day: PR #365 (13 sprint-5/6 specs + meta). Prior: PR #364 (D-SDR-3/4/5 + sprint-log-4 governance + sprint-5-9 roadmap + codex P1/P2 fixes). lance-graph #364 ships D-SDR-3/4/5 + sprint-log-4 governance + sprint-5-9 roadmap + codex P1/P2 surgical fixes (OwlIdentity 3-byte canonical, UnifiedAuditEvent 26 bytes, OgitFamilyTable sparse `HashMap<u16, FamilyEntry>`, audit super_domain via AuditChain). MedCare-rs#112 (PR-B) wires `UnifiedBridge<MedcareBridge>` + medcare-rbac + medcare-realtime substrate (+2963 LOC, 17 files, §73 SGB V + BMV-Ä §57 + BtM regulatory tests). smb-office-rs#31 (PR-C) wires `UnifiedBridge<OgitBridge>` (+111 LOC). ndarray#142 ships VBMI gate for `permute_bytes` (P0 SIGILL fix on Skylake-X / Cascade Lake / Ice Lake-SP) + Inf clamp for `simd_exp_f32`. D-SDR-5 `UnifiedBridge` surface is now consumed end-to-end across MedCare + smb-office. Prior: 2026-05-07 (PR #354). Prior: 2026-05-07 (PR #353). Prior: 2026-05-07 (PR #352). Prior: 2026-05-06 (splat-osint-ingestion-v1 PR 1+2 of 6 in flight). Prior: 2026-04-21 post PR #243.
>
> Purpose: prevent new sessions from hallucinating structure that
> already exists or proposing features already shipped. Read this
> BEFORE proposing any grammar/crystal/contract changes.

---

> **2026-06-03 — hardened (follow-up after #460)** (D-HELIX-1 wiring): `crates/helix` now takes **ndarray as a MANDATORY, non-optional git dependency** (`git = AdaWorldAPI/ndarray @ master`), replacing the optional `path` dep + `ndarray-hpc` feature. Why: (1) codex P2 — an optional *path* dep still forces Cargo to read the local sibling manifest at resolution, so a clean checkout failed before feature selection; (2) directive "ndarray is mandatory for lance-graph". `simd.rs` always uses `ndarray::simd` (no scalar fallback); the self-contained fork → no import cycle. 63 unit + 6 doctests green; clippy/fmt clean. See E-HELIX-NDARRAY-MANDATORY.
>
> **2026-06-03 — shipped (autoattended)** (D-HELIX-1): new standalone crate `crates/helix` — the golden-spiral **Place/Residue** codec from the user's `KNOWLEDGE.md`. HHTL = deterministic PLACE; helix = orthogonal RESIDUE. Pipeline: equal-area `√u` hemisphere placement (`HemispherePoint`) → stride-4-over-17 `CurveRuler` coupling → Fisher-Z/arctanh `Similarity` alignment → EULER_GAMMA hand-off → 256-palette `RollingFloor` quantise (occupancy-drift + version stamp) → 3-byte `ResidueEdge` endpoint pair; metric-safe L1 via 256×256 `DistanceLut` (`distance_adaptive`) + non-metric byte-Hamming `distance_heuristic`. `prove()` closes the 2-D discrepancy Open Item (companion to `jc::weyl`). Zero-dep default (`edition 2021`, empty `[workspace]`, root `exclude`); optional `ndarray-hpc` feature routes batch Fisher-Z through `ndarray::simd::simd_ln_f32`. **61 unit + 6 doctests green** on BOTH feature configs; clippy -D warnings + fmt clean. ~80% overlaps existing CERTIFIED primitives by design (clean-room, user-directed) — see `crates/helix/KNOWLEDGE.md` § Overlap & Consolidation + E-HELIX-OVERLAP + TD-HELIX-OVERLAP-1. Branch claude/gallant-rubin-Y9pQd.
>
> **2026-06-01 — shipped (autoattended)** (D-A3): `lance_graph_contract::atoms` — `I4x32::pack`/`unpack` implemented (the 2 `todo!()`s gone) + new `I4x64` (256-bit / 64 signed-i4 dims, `repr(C, align(16))`, 32 B) + private `sext4`. Two's-complement signed-i4 nibble codec (byte-compatible with `QualiaI4_16D` + the `CausalEdge64` mantissa), sign-agnostic (caller pre-scales). The carrier is a deterministic **CAM address** + sparse-intensity "smell" — NO vector search, no float; the `{instance,reference}` dual is rejected ("64" = 64 poles). Contract lib **562 green** (+9), offline, zero new deps. The bipolar `−introspection..+exploration` pole semantics + asymmetric scaling ride the caller's pre-scale (A4). Plan `.claude/plans/a3-carrier-v1.md`; doctrine `.claude/knowledge/ephemeral-warm-cold-lifecycle.md`.
>
> **2026-06-01 — PR-in-flight (autoattended)** (D-EW64-2): `lance_graph_contract::episodic_edges::EpisodicEdges64::{promote, strongest}` — MRU "promote" strengthens an edge to slot 0 (the hot / most-immediate position); fire→front, un-refired ages toward slot 3 and evicts to the cold connectome; **slot order IS the strength ranking** (no per-edge weight stored — the co-addressed `CausalEdge64` plasticity carries the Hebbian weight, recency is the slot index). Realizes `E-EW64-STRENGTH-IS-CE64-PLASTICITY` (the user's "stronger immediate edges"). Zero-dep; contract lib 533 green (+5), default clippy clean, episodic_edges.rs pedantic+nursery clean. The surreal-LIVE "wingman" that drives `promote` stays GATED on OQ-11.6 (LanceDB-LIVE fallback exists) — this is the substrate-agnostic hot-tier mechanism it calls.

---

## Recently Shipped PRs (reverse chronological)

| PR | Merged | Title | What it added |
|---|---|---|---|
| **#459** | 2026-06-03 | feat(helix): golden-spiral Place/Residue codec (zero-dep + optional ndarray-hpc) | **`crates/helix`** — new standalone codec realising the Place/Residue `KNOWLEDGE.md` (HHTL = PLACE, helix = orthogonal RESIDUE). `HemispherePoint` (√u equal-area placement) → `CurveRuler` (stride-4-over-17) → `Similarity` (Fisher-Z/arctanh) → `RollingFloor` (256-palette; occupancy-drift + version stamp) → 3-byte `ResidueEdge` + `DistanceLut` (metric-safe 256×256 L1) + `prove()` (2-D discrepancy companion to `jc::weyl`). Zero-dep default (empty `[workspace]`, root `exclude`); optional `ndarray-hpc` = batch Fisher-Z via `simd_ln_f32`. 63 unit + 6 doctests green both configs; clippy/fmt clean. ~80% clean-room overlap with CERTIFIED primitives (E-HELIX-OVERLAP / TD-HELIX-OVERLAP-1). Merge commit `ef35ff1`. Branch `claude/gallant-rubin-Y9pQd`. |
| **#450** | 2026-06-01 | NAL syllogism capstone + atoms/styles/NAL → planner-DTO unification (A1/A2) | **`causal-edge::syllogism`** — hardwired NAL **figure** resolver (`Figure{Chain,ChainRev,SharedSubject,SharedObject}` + `figure()`/`syllogize()`; SPO term-sharing → Deduction/Induction/Abduction + signed mantissa; the reasoning kernel, Pearl-2³-analogue). **A1** `contract::nars::InferenceType::{to,from}_mantissa` (zero-dep cross-crate rule bridge) + `From<grammar::NarsInference>`. **A2** `rung: RungLevel` on both `ThinkingContext` structs (the meta-aware handle). Unification spec §0–14 (`.claude/specs/atoms-styles-nal-planner-dto-unification-v1.md`) + **vart vendored** (`/home/user/vart`). Branch `claude/jolly-cori-clnf9`. _(PR_ARC #450 entry owed.)_ |
| **#411** | 2026-05-27 | Cognitive substrate: locked 33-TSV atom layer + 34-tactic recipes + escalation loop | **D-PERSONA-1** `contract::escalation` + `planner::mul::escalation` (CollapseHint/InnerCouncil/EpiphanyDetector/GhostEcho/WisdomMarker/Checklist, 13 tests). **`contract::atoms`** — LOCKED 33-dim TSV `CANONICAL_ATOMS` (3 Pearl + 9 Rung + 5 Σ + 8 Ops + 4 Presence + 4 Meta) + `I4x32` carrier. **`contract::recipes`** — 34-tactic metadata catalogue. **`contract::recipe_kernels`** — the 34 tactics as 34 `Tactic` impls + registry over a shared `ThoughtCtx`. Charter D0: **ladybug-rs has no relation, rewrite-not-port**; lattice is **SPOQ** (SPO 2³ causal + Q qualia overlay); business = OGIT sidecar; markers gate the datapath/control/gate partition. Green: escalation 13 / atoms 3 / recipes 4 / recipe_kernels 5 + 446 prior, no warnings. Branch `claude/splat3d-cpu-simd-renderer-MAOO0` (39 commits). See PR_ARC #411. |
| **#389** | 2026-05-16 | fix(sprint-12/wave-F): codex P2 — AttentionMaskBackend impl for AttentionMaskSoA + canonical MailboxId import | Codex P2 follow-on to PR #388. Adds `AttentionMaskBackend` trait impl for `AttentionMaskSoA` (Wave-F surface coherence) and converges duplicate `MailboxId` imports onto the canonical contract definition. Merge commit `b526485`. |
| **#388** | 2026-05-16 | impl(sprint-12/wave-F partial): D-CSV-10 sigma-tier-router + AttentionMask + splat ops + governance (6 of 12 workers landed) | Sprint-12 Wave F fleet partial landing. **D-CSV-10** `SigmaTierRouter` crate (Rubicon-resonance ΔF + threshold → Σ10 commit, hand-tuned threshold per OQ-CSV-6, tracked as TD-SIGMA-TIER-THRESHOLDS-1); **D-CSV-12** scalar splat op fleet on i4 (`splat_gaussian`, `score_hole_closure`, `replay_coherence`, `emit_if_epiphany`); **AttentionMask** SoA + actor + backend surface; W-F8 TYPE_DUPLICATION_MAP refresh (records two-`TrustTexture` coexistence as TD-TRUST-TEXTURE-DUPE-1); W-F10 sprint-11 Opus meta-review; W-F11 i4-substrate-decisions knowledge doc; W-F12 cognitive-substrate-convergence-v2 plan draft (608 lines). Merge commit `77f2d26`. |
| **#387** | 2026-05-16 | impl(sprint-11/wave-E): D-CSV-8 MUL i4 SIMD evaluation + D-CSV-9 8ch↔SPO transcoder | **D-CSV-8** integer MUL evaluation on `QualiaI4_16D` + signed mantissa (scalar i4 path; AVX-512/NEON deferred → D-CSV-13 sprint-12). **D-CSV-9** 8-channel ↔ SPO-palette transcoder (Option R-3) at thinking-engine L3 commit boundary; 16-mapping bidirectional round-trip; renames `set_channel` → `set_channel_u8` to widen equivalence class. Merge commit `e042c70`. |
| **#386** | 2026-05-16 | impl(sprint-11/wave-D): D-CSV-7 MailboxSoA + D-CSV-6a WitnessCorpus core (parallel workers) | **D-CSV-7** `MailboxSoA<N>` integration: W-slot referencing + per-row plasticity accumulator + `apply_edges` for baton receipt; `last_emission_cycle` u32::MAX sentinel + lib re-export + ndarray hpc-extras feature. **D-CSV-6a** `WitnessCorpus` partial (W-slot anchor + chain invariant; sorted by emission cycle, drop-oldest truncation). Full CAM-PQ-indexed corpus (D-CSV-6b) sprint-12. Merge commit `33110c8`. |
| **#385** | 2026-05-16 | impl(sprint-11/wave-C): D-CSV-5a sibling QualiaI4Column add (double-write, no read-side change) | **D-CSV-5a** sibling `QualiaI4Column` add to `cognitive-shader-driver::FingerprintColumns` per OQ-CSV-4 ratification (sibling-then-cutover). Double-writes f32 + i4 during sprint-11/12; cutover (D-CSV-5b) drops f32 column once consumers migrated. Worker recovery from stash + `[..17]` slicing + hpc-extras feature gate. Merge commit `6f58418`. |
| **#384** | 2026-05-16 | impl(sprint-11/wave-B): D-CSV-2 QualiaI4_16D type + OQ-CSV-1 ratification (Option α) | **D-CSV-2** `QualiaI4_16D` 16-dim signed-i4 type in `lance-graph-contract::qualia` + f32↔i4 migration helpers (`to_f32_17d`). **OQ-CSV-1 ratified to Option α** — canonical convergence-observable vocab (arousal/valence/tension/curiosity/…); drop dim 16 "integration" placeholder. 14 unit tests pass; codex P1 + CI gate fmt fix. Merge commit `0751a8b`. |
| **#383** | 2026-05-16 | impl(sprint-11/wave-A): D-CSV-1/3/4 — causal-edge v2 layout + InferenceType signed mantissa + CollapseGateEmission | Sprint-11 Wave A landing. **D-CSV-1** `causal-edge` crate v2 layout (signed mantissa, W-slot 6 bits per OQ-CSV-2, lens, drop temporal); feature-gated via `causal-edge-v2-layout`; crate bumped 0.1.0 → 0.2.0. **D-CSV-3** `InferenceType` signed-mantissa expansion absorbing PR-LL-1 Intervention/Counterfactual into Reserved5/6 of the canonical edge enum. **D-CSV-4** `CollapseGateEmission` wire format in contract crate (Vec instead of SmallVec to preserve zero-dep — TD-COLLAPSE-GATE-SMALLVEC-1 tracks the optimization). Merge commit `03bd175`. |
| **#372** | 2026-05-14 | specs(sprint-10): 12-worker CCA2A fleet + meta-review (governance only) | Sprint-10 spec sprint mirroring PR #365 pattern (specs precede a separate implementation wave). **38 .md files / ~580 KB / zero .rs changes.** 11 PR-ready worker specs (~370 KB) covering par-tile crate apex, CausalEdge64 v2 layout, BindSpace E/F/G/H columns, AriGraph SPO-G + ghost edges + SpoWitnessChain, MailboxSoA + AttentionMaskActor, SigmaTierRouter + banding + plasticity + KernelHandle cache, bevy cull plugin, ndarray Miri completion, sprint-10 execution plan, PR dep graph, unified test plan. Opus meta-review (~28 KB) with sprint grade B+, 6 cross-spec inconsistencies (CSI-1..6), 5 cross-cutting epiphanies (E-META-1..5), sprint-11 spawn decision = NO until 5 spec patches + 4 user ratifications. 8 knowledge docs (~123 KB) documenting: **dual `CausalEdge64` finding** (SPO-palette variant in `causal-edge` crate ≠ 8-channel cascade variant in `thinking-engine` crate, same name different bit semantics); **p64 drift origin** pinpointed at `crates/lance-graph-planner/src/cache/convergence.rs:18-22 #[allow(unused_imports)]`; **three-zone hot-path model** (Zone-1 thinking-engine MatVec 200-500ns + AriGraph entity_index O(1), Zone-2 blasgraph+neighborhood cascade 20-1200µs, Zone-3 DataFusion >1ms); **SPOW tetrahedron + ontology-aware splat vision**; **5-sprint reunification arc** to unify thinking-engine + cognitive-shader-driver SoA. **Deferred:** sprint-11 implementation wave, `Think` carrier struct unification (sprint-12+), splat shader op fleet (sprint-13+), OWL DOLCE / OntologyFilter wiring (sprint-12+), PR-J1-INT4-32D-ATOMS as Wave 0.5 prerequisite. |
| **#366** | 2026-05-13 | impl(sprint-7): 7-worker implementation wave + AuditSink trait unification | Sprint-7 CCA2A 6-parallel + 1-sequenced + 1-Opus-meta. **~5 KLOC across 5 crates + 2 new** (`lance-graph-supervisor`, `lance-graph-consumer-conformance`). Workers: **S7-W1** `parse_family_registry()` + Healthcare basins `0x10..=0x19` (unblocks MedCare-rs E1-2/E1-3/E1-4 cascade); **S7-W2** `lance-graph-contract/build.rs` codegen (zero-dep preserved; sorted-slice + binary_search, no phf — OQ-2); **S7-W3** ractor supervisor with separate 18-byte `LifecycleAuditEvent` (CC-2) + `SuperDomain::System` exempt (CC-3); **S7-W4** `assert_consumer_conformance` harness (A1-A10); **S7-W5** `CognitiveBridgeGate` trait + `UnifiedBridgeGate<B>` impl; **S7-W6** new `audit_sink/` module (`AuditSink` trait + `JsonlAuditSink` + `LanceAuditSink` + `CompositeSink`) + `audit_verify` CLI + `prev_merkle` field on UnifiedAuditEvent (canonical_bytes still 26 B); **S7-W7** SMB Foundry `0x80..=0x82` vs BSON `0xA0..=0xAD` disjoint slots (OQ-4). **Post-meta AuditSink trait unification** (`bc530a4`): dropped legacy `UnifiedAuditSink` D-SDR-4 placeholder, `UnifiedBridge::audit_sink: Arc<dyn AuditSink>`, added `with_jsonl_audit()` ergonomic constructor (OQ-7-2 + OQ-7-3 locked). **Pre-existing workspace lint debt** cleaned by Sonnet janitor across ~30 files in `lance-graph` core / `bgz-tensor` / planner / nsm (sprint-7 outputs guardrailed). **Opus meta verdict** at `.claude/board/sprint-log-7/meta-review.md`: 4A/2B/1B-minus/0 C/D/F. **Adjacent landings:** MedCare-rs sprint-1 10-PR sweep #113-#122 (E1-1 OQ-3 consumed our `0d725d4` decision; sprint-2 5 PRs queued). |
| **#365** | 2026-05-13 | specs(sprint-5-6): 13-worker parallel batch + Opus meta review | Governance-only PR. **13 PR-ready specs at `.claude/specs/`** (~300 KB) from a 12-Sonnet-worker + 1-post-meta-Sonnet-worker + 1-Opus-meta-agent parallel batch. Spec grades: 3 A (W2 d3b-jsonl, W5 pr-graph, W12 conformance), 7 B, 2 C (W10 manifest-modules needs §4.3 sorted-slice rewrite; W11 ractor-supervisor needs LifecycleAuditEvent split). 24 KB Opus meta cross-spec review at `.claude/board/sprint-log-5-6/meta-review.md`. 4 blocking OQs (W3 parser entry, W10 phf vs sorted-slice, W6 Role migration, W13 BSON namespace). CCA2A 12+1+1 pattern validated at scale: ~300 KB of PR-ready output in under an hour wall-clock; 3 workers required respawns for permission denials (settings.json patched for `.claude/board/sprint-log-5-6/**`). |
| **#364** | 2026-05-13 | D-SDR-3/4/5 + sprint-log-4 governance + sprint 5-9 roadmap + codex P1/P2 | Tier-A substrate close: **D-SDR-3** OgitFamilyTable + FamilyEntry codebook (~300 LOC), **D-SDR-4** merkle-chained UnifiedAuditEvent (~460 LOC, AuditMerkleRoot = u64 FNV-1a), **D-SDR-5** authorize_* through Policy::evaluate with audit emission (~300 LOC). **Codex P1 fix** (`3208743`): OwlIdentity widened u8→u16 slot → 3-byte canonical `[family, slot_lo, slot_hi]`; OgitFamilyTable → sparse `HashMap<u16, FamilyEntry>`; UnifiedAuditEvent canonical_bytes 25→26. **Codex P2 fix** (`e23ce89`): emit_audit uses AuditChain.super_domain() instead of static FAMILY_TO_SUPER_DOMAIN. **CI fix** (`a3c753f`): ndarray/hpc-extras opt-in for blake3. Sprint-log-4 governance corpus (12 worker specs + 2 meta reviews) + sprint-5-through-9 roadmap (70 agents = 60W + 10M across 5 sprints, mandatory 12-step plan-read-order in worker prompts). 97/97 callcenter lib tests pass. All 5 CI checks green on `c8176cb`. Adjacent: ndarray#142 (VBMI gate + Inf clamp) merged same day. |
| **#354** | 2026-05-07 | gov: #353 post-merge + cross-repo adjacent-landings | Pure governance close-out. PR_ARC entry for #353 + LATEST_STATE row. Documents the 5-PR coordinated landing across 4 repos: lance-graph #352/#353/#354 + OGIT #2 (woa+medcare bridges unblocked for OGIT-O(1)) + woa-rs #2 (cross-repo `--features ontology` integration) + MedCare-rs #109 (`?source=lance` exercising Zone 2 → Zone 3 rewriter chain). Locks: append-only board hygiene durability across 4 sequential prepends; cross-repo coordinated-landing recipe. |
| **#353** | 2026-05-07 | plan: palantir-parity-cascade v2 + SoA DTO entropy ledger + #352 post-merge governance | Three artifacts. **v2 capstone** (262 lines): integrates 4 prior Foundry parity docs. Pillar 0 carry-forward: Foundry parity IS SoA-as-canon parity. Column H (PR #272 SHIPPED) is already the Foundry Object Type bridge. 15 D-PARITY-V2 deliverables. **SoA DTO entropy ledger** (210 lines, append-only knowledge): 22 DTOs classified across 4 tiers (sensor → engine → contract → callcenter). Buckets: 9 bare-metal / 7 SoA-glue / 6 bridge-projection (3 OPEN). `ResonanceDto` IS the SoA. Codec cascade columns all OPEN today. **#352 post-merge governance**: PR_ARC + LATEST_STATE updates. |
| **#352** | 2026-05-07 | plan: lance-graph-ontology v5 + ogit-cascade v1 | Two-plan PR. **v5** (177 lines): 15 deliverables for ontology crate post-merge follow-on (D-1 dcterms:source, D-2 SpoBridge::promote_to_spo, D-9 ontology-aware MUL thresholds). 4 ratifications (smb-ontology export-only, D-9 above D-2, MulThresholdProfile in lance-graph-contract, OGIT-fork upstream non-PR). **v1 cascade** (209 lines): 15 D-CASCADE deliverables for SoA-as-canon + Zone 1/2/3 + BioPortal arsenal + bridge collapse. **Pillar 0**: OntologyRegistry IS the SoA, schema IS the DTO + name→row index. **Codec cascade per row** (target state, NOT YET WIRED — D-CASCADE-V1-7): identity Vsa16kF32 → CAM-PQ 6 B → Base17 34 B → palette key 4 B → Scent 1 B + qualia 18×f32 + meta 8 B + edge 8 B, every step O(1). |
| **#243** | *(open)* | D5+D7 categorical-algebraic inference | `thinking_styles.rs` (490 LOC, 12 tests), `free_energy.rs` (347 LOC, 7 tests), `role_keys.rs` bind/unbind/recovery (295 LOC, 14 tests), `content_fp.rs` (98 LOC, 5 tests), `markov_bundle.rs` (250 LOC, 8 tests), `trajectory.rs` (298 LOC, 4 tests). Plans: `categorical-algebraic-inference-v1.md` (496 lines). Knowledge: `paper-landscape-grammar-parsing.md`, `session-2026-04-21-categorical-click.md`. CLAUDE.md § The Click (P-1). 12 epiphanies. |
| **#225** | *(open)* | Codec-sweep plan + D0.6/D0.7 CodecParams | 9-commit plan (`codec-sweep-via-lab-infra-v1.md`, Rules A-F, 9 starter YAMLs, CODING_PRACTICES audit) + `lance-graph-contract::cam` CodecParams/Builder/precision-ladder validation (14 tests). 147/147 contract suite |
| **#224** | 2026-04-20 | lab = API+Planner+JIT, thinking harvest, I11 measurability | `lab-vs-canonical-surface.md` extended: three-part lab stack (API + Planner + JIT), thinking-harvest subsection (REST/Cypher → `{rows, thinking_trace}` = the AGI magic bullet), I11 invariant (every layer L0→L4 emits harvest-ready trace; no black-box short-circuits) |
| **#223** | 2026-04-20 | LAB-ONLY firewall + AGI-as-SoA + I1-I10 | `lab-vs-canonical-surface.md` initial doc: canonical consumer = `UnifiedStep`/`OrchestrationBridge`, Wire DTOs are lab quarantine. AGI = (topic, angle, thinking, planner) = struct-of-arrays consuming cognitive-shader-driver. 10 cross-cutting invariants I1-I10 (BindSpace read-only, canonical `simd::*` import, temporal budgets, temperature hierarchy, thinking IS AdjacencyStore, weights are seeds, per-cycle cascade, 4096 surface, three DTO families, HEEL/HIP/BRANCH/TWIG/LEAF) |
| **#210** | 2026-04-19 | Phase 1 grammar + knowledge docs | ContextChain reasoning ops, role_keys slice catalogue, 3 knowledge docs (grammar-landscape, linguistic-epiphanies E13-E27, fractal-codec) |
| **#209** | 2026-04-19 | sandwich layout + bipolar cells | Crystal fingerprint sandwich, VSA_permute reference, lossless bundling corrections |
| **#208** | 2026-04-19 | grammar + crystal + AriGraph unbundle | Contract grammar/ + crystal/ modules, AriGraph episodic unbundle hooks with SIMD dispatch |
| **#206** | 2026-04-18 | state classification pillars | qualia.rs (17D), proprioception.rs (7 anchors), world_map.rs, sigma_rosetta 64 glyphs + 144 verbs, Pumpkin NPC example |
| **#205** | 2026-04-18 | engine bridge + CMYK/RGB qualia | engine_bridge.rs, 12-style unified mapping, 17D vs 18D qualia distinction |
| **#204** | 2026-04-18 | cognitive-shader-driver | New crate, ShaderDispatch/Resonance/Bus/Crystal DTOs, BindSpace struct-of-arrays, full ladybug-rs import |

## Current Contract Inventory (lance-graph-contract)

> **2026-05-31 — ADDED (D-EW64-1 + D-VIEW-1, episodic-RISC-spine)**: `episodic_edges::{EpisodicEdges64(u64), EdgeRef{family:u8,local:u16}}` — AriGraph episodic edges, 4x[4-bit family | 12-bit local]: family 0 = intra-basin (inherited, ~98.6% per #444), 1..=15 = cross-family index into the OGIT-class-inherited palette (~1.4%; identities inherited, never on the edge — I-VSA-IDENTITIES). Plus `view_angle::ViewAngle` (4-bit view-schema selector; presence bitmask doubles as attention mask, inherited). Zero-dep; 527 contract lib tests; clippy pedantic+nursery clean. Plan: episodic-risc-spine-v1.md.

> **2026-05-31 — ADDED (D-H2H-1, head2head superposition winner-select)**: `lance_graph_contract::head2head::{Head2Head (judge), WinnerCriterion (DissonanceMin≈infight / SupportSpread≈Raumgewinn / ConfidenceMax / Tempered=default), CompetitionOutcome}`. `Head2Head::select(&Blackboard) -> Option<CompetitionOutcome>` picks the winning competing-expert bid over the existing `a2a_blackboard` (confidence/dissonance/support) — pure read + arg-extremum, **no new identity, copies nothing** (select-don't-duplicate, `I-VSA-IDENTITIES`); `margin` = the dark-horse signal. The *selection* half of head2head superposition; parallel-mailbox *execution* is the CI-gated consumer side. Zero-dep; 516 contract lib tests (+7); clippy pedantic+nursery clean.

> **2026-05-31 — ADDED (D-MBX-9-IN, VersionScheduler contract slice, on `b6e3cc6`/lance7)**: `lance_graph_contract::scheduler::{DatasetVersion(u64), VersionScheduler (trait), NextPhaseScheduler (reference impl)}`. The IN-direction dual of `MailboxSoaOwner` (`E-SUBSTRATE-IS-THE-SCHEDULER`): `on_version<V: MailboxSoaView>(&V, DatasetVersion, ExecTarget) -> Option<KanbanMove>` lowers a Lance `versions()` tick to the next legal Rubicon `KanbanMove`; `NextPhaseScheduler` is the forward-arc reference (Libet `-550ms` anchor on Planning→CognitiveWork, `None` on absorbing). Read-only over the view (**propose-not-dispose**, R1); composes only existing contract types; zero-dep. 509 contract lib tests (+6); clippy pedantic-clean. CI-gated twin = `LanceVersionScheduler` over `VersionedGraph::versions()` via callcenter `LanceVersionWatcher`. Closes D-MBX-9 IN-direction at the type level (OUT twin + core impl remain CI-gated).

> **2026-05-31 — MERGED (#441, D-CLS arc, merge `a77e119`)**: `lance_graph_contract::class_view::{FieldMask (u64 presence bitmask), ClassView (resolver trait), ClassProjection, RenderRow}` + `ClassView::render_rows` (off-bits-skipped). `ClassId = u16` (reuses `soa_view::class_id`). The class meta-DTO **flies ABOVE the agnostic SoA** — labels/shape/DOLCE resolve LATE from the OGIT cache, nothing semantic in the row (C2 presence≠semantics; N3 stable positions; out-of-range mask bits IGNORED not folded — Codex P2). Ontology side: `class_resolver::RegistryClassView` (impls `ClassView` over the live `OntologyRegistry`, DOLCE via `classify_odoo`) + `odoo_blueprint::class_signature::{StructuralSignature, OdooEntity::signature()/object_view() carrier methods, audit, shape_families, curated_entities, corpus_summary}` (deterministic FNV-1a structural-hash group-by, NOT Aerial-cluster). Zero-dep preserved; extends `ontology::{ObjectView,FieldRef,DisplayTemplate}`, reuses `class_id` (no new newtype). 497 contract + 240 ontology lib tests. D-CLS-{FM,RES,SIG,AUDIT,RENDER} all Shipped.
>
> **2026-05-30 — PR-in-flight addition** (D-MBX-A6 Phase 2 — Rubicon lifecycle + ExecTarget): `lance_graph_contract::kanban::{ExecTarget (Native/Jit/SurrealQl/Elixir), RubiconTransitionError}` + `KanbanColumn::{next_phases, can_transition_to}` (the Rubicon lifecycle DAG) + `KanbanMove.exec: ExecTarget` field + `MailboxSoaOwner::try_advance_phase()` (checked lifecycle enforcement → `Result<KanbanMove, RubiconTransitionError>`). Zero-dep; `KanbanMove` still ≤16 B; 489 contract lib tests (+4); downstream cargo-check clean. Lifecycle enforcement + planner exec-target are now contract-level invariants. Resolves the #437 deferred exec-target NOTE. Cross-ref D-MBX-A6 / #437.
>
> **2026-05-30 — PR-in-flight addition** (D-MBX-A6 Phase 1 — planner⟷ractor⟷surreal meta-DTO): `lance_graph_contract::kanban::{KanbanColumn (6: Planning/CognitiveWork/Evaluation/Commit/Plan/Prune), KanbanMove}` — the 4-phase Rubicon kanban transition; `KanbanMove` is `Copy`, ≤16 B, carries `MailboxId` + `witness_chain_position` (R4 pointer) + `libet_offset_us` (−550 ms anchor, D-MBX-8). `lance_graph_contract::soa_view::{MailboxSoaView, MailboxSoaOwner}` — zero-dep **borrow trait** for the transparent zero-copy SoA view (R1 "one SoA never transformed"); `&[T]` column accessors (energy/edges_raw/meta_raw/entity_type) mirror `MailboxSoA::*_at`; the read/owner split makes "view is read-only" structural (surreal implements only the read half). `orchestration::StepDomain::Kanban` variant + `"kanban."` prefix. Consumed via the EXISTING `OrchestrationBridge` (planner emits, ractor owns/drives via `MailboxSoaOwner`, surreal_container projects via read-only `MailboxSoaView`) — NO parallel DTO family (lab-vs-canonical ruling). Contract `[dependencies]` still empty. 485 contract lib tests green (+6 new); `cargo check` clean on planner/cognitive-shader-driver/supervisor (StepDomain variant additive-safe). Consumer impls deferred. See E-SOA-VIEW-IS-A-BORROW; `unified-soa-convergence-v1.md §5+§8.4`.
>
> **2026-05-28 — PR-in-flight addition** (bindspace→mailbox migration wave A1-A4): `lance_graph_contract::witness_table::{WitnessEntry, WitnessTable<N=64>}` — column-type primitive resolving the 6-bit W-slot in `CausalEdge64 v2` into a per-cohort `(mailbox_ref: u32, spo_fact_ref: Option<u64>)` table (`mailbox_ref` carries the full canonical `MailboxId`, NOT a truncated cohort-local index — see PR #427 Codex P2 fix). Zero-dep, 3 unit tests, `WitnessTable::{new, get, set, default}`. Cross-ref: `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md` §10 (architectural refinements landed in same wave). Also in same wave: `cognitive-shader-driver::MailboxSoA<N>` gains four thoughtspace columns (`edges: [CausalEdge64; N]`, `qualia: [QualiaI4_16D; N]`, `meta: [MetaWord; N]`, `entity_type: [u16; N]`) + 8 row accessors; `ShaderDriver` gains transitional `mailboxes: HashMap<MailboxId, MailboxSoA<1024>>` + `with_mailbox()` builder + `mailbox()` read accessor (sibling-shape, additive — singleton untouched). 457 contract+driver tests pass.

Types that EXIST — do NOT re-propose them:

**`grammar/`**: `FailureTicket`, `PartialParse`, `CausalAmbiguity`, `TekamoloSlots`, `TekamoloSlot`, `WechselAmbiguity`, `WechselRole`, `FinnishCase`, `finnish_case_for_suffix`, `NarsInference`, `inference_to_style_cluster`, `ContextChain` (with coherence_at / total_coherence / replay_with_alternative / disambiguate / DisambiguationResult / WeightingKernel), `RoleKey` + 47 `LazyLock<RoleKey>` instances + `Tense` enum + `finnish_case_key / tense_key / nars_inference_key` lookups, **`RoleKey::bind/unbind/recovery_margin`** (slice-masked XOR), **`Vsa10k`** + `VSA_ZERO` + `vsa_xor` + `vsa_similarity`, **`GrammarStyleConfig`** + **`GrammarStyleAwareness`** + `revise_truth` + `ParseOutcome` + `divergence_from`, **`FreeEnergy`** + **`Hypothesis`** + **`Resolution`** (Commit / Epiphany / FailureTicket) + `from_ranked` + thresholds.

**`crystal/`**: `Crystal` trait, `CrystalKind`, `TruthValue`, `UNBUNDLE_HARDNESS_THRESHOLD = 0.8`, `CrystalFingerprint` (Binary16K / Structured5x5 / Vsa10kI8 / Vsa10kF32 / **Vsa16kF32**), `Structured5x5`, `Quorum5D`, `SentenceCrystal`, `ContextCrystal`, `DocumentCrystal`, `CycleCrystal`, `SessionCrystal`, sandwich layout constants, **`vsa16k_zero` / `binary16k_to_vsa16k_bipolar` / `vsa16k_to_binary16k_threshold` / `vsa16k_bind` / `vsa16k_bundle` / `vsa16k_cosine`** (Click switchboard carrier + algebra, 64 KB, inside-BBB only).

**`cognitive_shader`**: `ShaderDispatch`, `ShaderResonance`, `ShaderBus`, `ShaderCrystal`, `MetaWord` (u32 packed), `MetaFilter`, `ColumnWindow`, `StyleSelector`, `RungLevel`, `EmitMode`, `ShaderSink` trait, `CognitiveShaderDriver` trait.

**`cognitive-shader-driver` BindSpace substrate (2026-04-24)**: `FingerprintColumns.cycle` is now `Box<[f32]>` (Vsa16kF32 carrier, 16_384 f32 per row = 64 KB) — migrated from `Box<[u64]>` (Binary16K). New constant `FLOATS_PER_VSA = 16_384`. New methods: `set_cycle(&[f32])`, `set_cycle_from_bits(&[u64; 256])` (adapter with `binary16k_to_vsa16k_bipolar` projection), `cycle_row() -> &[f32]`. `write_cycle_fingerprint()` API unchanged (takes `&[u64; 256]`), converts internally. `byte_footprint()` for 1 row = 71_774 bytes. Other three planes (content/topic/angle) remain `Box<[u64]>`.

**CausalEdge64 — TWO distinct types in workspace (2026-05-14, PR #372 finding)**: same name, different bit semantics, different consumers. Always qualify by crate when referring to either:
- `causal_edge::CausalEdge64` (`crates/causal-edge/src/edge.rs:60`) — SPO-palette layout: S/P/O palette indices + NARS f/c + Pearl 2³ mask + direction triad + inference type + plasticity flags + temporal index. Consumed by `lance-graph-planner::cache::nars_engine` (NarsTables), `cognitive-shader-driver::BindSpace::EdgeColumn`, AriGraph SPO commit path. The unit of NARS reasoning at cycle-speed.
- `thinking_engine::layered::CausalEdge64` (`crates/thinking-engine/src/layered.rs:45`) — 8-channel cascade: BECOMES / CAUSES / SUPPORTS / REFINES / GROUNDS / ABSTRACTS / RELATES / CONTRADICTS (each 1 byte). Source/target NOT in u64 (carried as tuple key `(target: u16, edge: CausalEdge64)`). Emitted by `TierEngine::emit_causal_edges` after MatVec; consumed by downstream tiers via `apply_edges`. The unit of cognitive-cascade dispatch in the L1 → L2 → L3 thinking pipeline.

Full reference: `.claude/knowledge/causal-edge-64-spo-variant.md` + `.claude/knowledge/causal-edge-64-thinking-engine-variant.md` + `.claude/knowledge/causal-edge-64-synergies-and-pr-trajectory.md`. Reunification path (Option R-3): transcode 8-channel → SPO at thinking-engine L3 commit boundary; see `.claude/knowledge/cognitive-shader-driver-thinking-engine-reunification.md`.

**`escalation`** (D-PERSONA-1, 2026-05-26, branch `claude/splat3d-cpu-simd-renderer-MAOO0`): the escalation+epiphany loop = the boot checklist (a *restore* of ladybug's qualia loop on our SoA — NOT a bespoke verifier). `CollapseHint` {Flow, Fanout, RungElevate} + `fanout_width` / `noise_tolerance` / `rung_delta` (ladybug `detector.rs` formulas); `Archetype` {Guardian, Catalyst, Balanced} + `InnerCouncil::{deliberate, from_signals}` + `is_split(0.7,0.5)` ×1.2 split-amplify → `CouncilVerdict`; `EpiphanyDetector::observe` (sim > baseline×1.5 ∧ window ≥ 4) → `Epiphany`; `GhostEcho` (8 named: Affinity/Epiphany/Somatic/Staunen/Wisdom/Thought/Grief/Boundary — canonical zero-dep home, mirrors `thinking_engine::ghosts::GhostType`, see TD-GHOST-ECHO-DUP-1) + `WisdomMarker` (asymptotic decay → 0.1 floor, never zero); `GateKind` {Hard, Soft} + `ChecklistItem` + `Checklist::{step, mark_red, boot_ready, all_flow, degraded}` (green-flip = Flow + epiphany; let-it-crash = `mark_red` re-escalate). Planner wiring at `lance_graph_planner::mul::escalation::{boot_checklist, verdict_from}` (§2: 6 HARD / 3 SOFT items + a `MulAssessment` → `CouncilVerdict` adapter). 13 tests (10 contract + 3 planner).

## cognitive-shader-driver Wire Surface (lab-only, post D0.1)

Types live in `crates/cognitive-shader-driver/src/wire.rs` behind `--features serve`:

- **`WireCodecParams`** + `WireLaneWidth {F32x16, U8x64, F64x8, BF16x32}` + `WireDistance {AdcU8, AdcI8}` + `WireRotation {Identity, Hadamard{dim}, Opq{matrix_blob_id, dim}}` + `WireResidualSpec {depth, centroids}` — serde mirrors of the `contract::cam::*` types from PR #225. `TryFrom<WireCodecParams> for CodecParams` runs the precision-ladder validation (OPQ↔BF16x32, overfit guard, pow2 Hadamard) at ingress BEFORE any JIT compile.
- **`WireTensorView {shape, lane_width, bytes_base64}`** + methods `row(&AlignedBytes, usize)` / `subspace(&AlignedBytes, row, k, sub_bytes)` / `row_count()` / `col_count()` / `row_bytes()` / `element_bytes()` / `decode() -> AlignedBytes`. Per Rule E (Wire surface IS the SIMD surface, object-oriented) + Rule A (stdlib `slice::array_windows::<N>` + `ndarray::simd::*` loaders).
- **`AlignedBytes`** — heap-allocated, 64-byte-aligned owned buffer produced once by `WireTensorView::decode` per Rule F (decode at REST ingress, never inside). Safe Send/Sync; `Drop` dealloc with matching layout.
- **`WireCalibrateRequest`** extended with optional `params: Option<WireCodecParams>` + `tensor_view: Option<WireTensorView>` (new path) alongside legacy fields (`num_subspaces` / `num_centroids` / `kmeans_iterations` / `max_rows`) for back-compat.
- **`WireCalibrateResponse`** extended with `kernel_hash: u64` (= `CodecParams::kernel_signature()` of the executed kernel) + `compile_time_us: u64` + `backend: String` ("amx" | "vnni" | "avx512" | "avx2" | "legacy"; **never "scalar"** — iron rule).
- **`WireTensorViewError {Base64, SizeMismatch, ZeroShape}`** — typed decode errors.

**`proprioception`**: 7 `StateAnchor` (Intake/Focused/Rest/Flow/Observer/Balanced/Baseline), 11-D `ProprioceptionAxes`, `StateClassifier` trait, `DefaultClassifier`, `hydrate()` softmax-weighted blend.

**`qualia`**: 17-D `QualiaVector`, `qualia_to_state` projection (17→11).

**`world_map`**: `WorldMapDto`, `WorldMapRenderer` trait, `DefaultRenderer`.

**`world_model`**: `WorldModelDto` with `qualia`, `axes`, `proprioception`, `cycle_fingerprint`, `timestamp`, `cycle_index`, `is_self_recognised()` / `is_liminal()`.

**`container`**: `Container = [u64; 256]` (16Kbit = 2KB), `CogRecord`.

**`property`** (new, SMB domain): `PropertyKind` (Required / Optional / Free), `PropertySpec` (predicate + kind + `CodecRoute` + NARS floor), `PropertySchema` (`&'static`-based, const schemas), `Schema` + `SchemaBuilder` (runtime builder: `.required()` / `.optional()` / `.searchable()` / `.free()` / `.validate()`), `CUSTOMER_SCHEMA`, `INVOICE_SCHEMA`. Maps bardioc Required/Optional/Free to I1 Codec Regime Split (ADR-0002).

**`repository`** (new, SMB domain): `EntityStore` + `EntityWriter` + `Batch` + `EntityKey` — Arrow-agnostic row store contract.

**`mail`** (new, SMB domain): `MailParser` + `ThreadLinker` + `ParseHints` + `AttachmentRef` + `PartRef`.

**`ocr`** (new, SMB domain): `OcrProvider` + `PageImage` + `OcrOpts` + `Bbox` + `BlockKind` + `LayoutBlock`.

**`splat`** (new, 2026-05-06): `SplatChannel` (6 variants: Support / Contradiction / Forecast / Counterfactual / Style / Source), `CamPlaneSplat` (q8 amplitude / width / theta_accept + 16-byte witness identity + 8-byte `replay_ref`), `AwarenessPlane16K` (256 × u64 = 2 KB pressure tile), `SplatPlaneSet` (6 channel planes = 12 KB), `CamSplatCertificate` (q8 pressure measurements + replay decision), `SplatDecision` (Proceed / RequireExactReplay / PrefetchOnly / ScenarioOnly / Drop), `TriadicProjection`, `ReasoningWitness64`. Resolves SPLAT-1 row in entropy ledger (Aspirational → Wired stage 1, entropy 4 → 2). Per `.claude/knowledge/gaussian-splat-cam-plane-workaround.md` PR 1. Plan: `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`.

**`tax`** (new, SMB domain): `TaxEngine` + `TaxPeriod` + `PeriodKind` + `Jurisdiction` + `PostingBatchRef` + `RuleBundle`.

**`reasoning`** (new, SMB domain): `Reasoner` + `ReasoningKind` + `ReasoningContext` + `EvidenceRef` + `Budget`.

**`cam`** (extended by PR #225): `CodecRoute` + `route_tensor` (existing), `CamByte`, `CamStrategy`, `DistanceTableProvider` trait, `CamCodecContract` trait, `IvfContract` trait, plus codec-sweep parameter shape — `LaneWidth` (F32x16 / U8x64 / F64x8 / BF16x32), `Distance` (AdcU8 / AdcI8), `Rotation` (Identity / Hadamard{dim} / Opq{matrix_blob_id, dim}), `ResidualSpec {depth, centroids}`, `CodecParams {subspaces, centroids, residual, lane_width, pre_rotation, distance, calibration_rows, measurement_rows, seed}` with `kernel_signature() -> u64` + `is_matmul_heavy() -> bool`, `CodecParamsBuilder` fluent API, `CodecParamsError {ZeroDimension, OpqRequiresBf16, HadamardDimNotPow2, CalibrationEqualsMeasurement}` — **precision-ladder validation fires at `.build()` BEFORE any JIT compile**.

**`graph_render`** (new, q2 cockpit): `RenderNode`, `RenderEdge`, `InferredConnection`, `Contradiction`, `GraphSnapshot`, `GraphHealth`, `CypherResult`, `CypherValue`, `CypherError`, `EpisodicTrace`, `ShaderEvent`, traits `GraphSnapshotProvider`, `GraphInferenceProvider`, `CypherExecutor`, `EpisodicTraceProvider`, `ShaderEventStream`. Visual render surface for Neo4j/Palantir Gotham cockpit — q2 consumes, lance-graph arigraph produces.

**`a2a_blackboard`**, **`collapse_gate`**, **`exploration`**, **`literal_graph`**, **`orchestration_mode`**, **`jit`**, **`nars`**, **`plan`**, **`orchestration`**, **`thinking`** (36 styles, 6 clusters), **`mul`**, **`sensorium`**, **`high_heel`**.

**Sprint-11/12 D-CSV substrate types (2026-05-16, PRs #383-#389)**:
- `lance-graph-contract::qualia`: `QualiaI4_16D` (16-dim signed-i4, 9× compression vs `[f32; 18]`), `QualiaI4Column` (sibling SoA column in cognitive-shader-driver).
- `lance-graph-contract::collapse_gate`: `CollapseGateEmission` (Vec-of-`(u16 target, CausalEdge64)` wire format; zero-dep — SmallVec optimization deferred as TD-COLLAPSE-GATE-SMALLVEC-1).
- `lance-graph-contract::mailbox` / `attention_mask`: `MailboxId` (canonical id type), `MailboxSoA<N>` (SoA mailbox with W-slot + plasticity accumulator + `apply_edges`), `AttentionMaskSoA`, `AttentionMaskActor`, `AttentionMaskBackend` trait.
- `lance-graph-contract::sigma_tier`: `SigmaTierBands`, `SigmaTierRouter` (Rubicon-resonance ΔF + threshold dispatch), `DispatchOutcome`, `RestReason` (Σ-tier crate surface).
- `lance-graph-contract::witness`: `WitnessCorpus` (CAM-PQ-indexed; D-CSV-6a partial in PR #386, full 6b sprint-12), `WitnessEntry`, `WitnessId`, `WitnessIndexHashMap` (anchor + chain invariant).
- `cognitive-shader-driver::bindspace`: `QualiaI4Column` (sibling SoA column, D-CSV-5a).
- `thinking-engine` + `ndarray`: `SplatField` (×2 — one in thinking-engine for Think carrier scalar ops, one in ndarray for vertical streaming).
- `ndarray::hpc::stream` (vertical streaming structs, D-CSV-11 Wave F W-F4/5/6, productization sprint-12): `QualiaI4Row`, `QualiaStream`, `InferenceRow`, `InferenceStream`, `SplatFieldStream` (+ planned `par_*` rayon variants gated behind ndarray `parallel` feature — deferred to sprint-14+).


## Current AriGraph Inventory (lance-graph/src/graph/arigraph/)

4696 LOC shipped, 7 modules:
- `episodic.rs` (210 LOC + unbundle hooks from #208) — `Episode`, `EpisodicMemory`, `unbundle_hardened`, `unbundle_targeted`, `rebundle_cold`, `UnbundleReport`, `RebundleReport`, `UNBUNDLE_HARDNESS_THRESHOLD = 0.8`
- `triplet_graph.rs` (1064 LOC) — SPO graph, NARS truth, BFS, spatial paths
- `retrieval.rs` (447 LOC) — fingerprint retrieval policies
- `sensorium.rs` (539 LOC) — observation → triplets
- `orchestrator.rs` (1562 LOC) — AriGraph coordinator
- `xai_client.rs` (521 LOC) — xAI enrichment
- `language.rs` (339 LOC) — LM bridge

## Workspace Conventions (locked in CLAUDE.md)

1. **Model policy:** main thread Opus + deep thinking; subagent grindwork → Sonnet; accumulation → Opus; NEVER Haiku.
2. **GitHub reads:** zipball to `/tmp/sources/` + local grep for 3+ reads per repo. MCP only for writes (PR, comments) and single-path reads.
3. **Contract zero-dep invariant:** `lance-graph-contract` has no external crate deps. Do not add any.
4. **Read before Write:** always Read a file before overwriting. Write-over-self without Read is the documented failure mode.
5. **No JSON serialization in types.** Serde stays out of types (debug-only). Wire formats are explicit.
6. **Pumpkin framing** for externally-visible examples (clinical / game-AI disguise for the AGI primitives).

## Active Branches (local at /home/user/lance-graph)

**Sprint-12/13 open work (2026-05-16):**

- `claude/sprint-12-wave-g-fleet` — **PR #390 OPEN** — sprint-12 Wave G follow-on. Lands the remaining D-CSV deliverables not in Wave F: **W-G1** D-CSV-5b QualiaColumn cutover (drop `[f32; 18]`, flip readers to i4); **W-G2** D-CSV-6b full CAM-PQ-indexed `WitnessCorpus` (unbounded, salience decay); **W-G3** batch i4 scalar MUL (paired w/ #388 Wave F); **W-G4** Σ10 Jirak-threshold derivation (D-CSV-15 NEW v2 entry; partial — VAMPE coupled-revival still sprint-13+).
- `claude/sprint-13-preflight-planning` — **work in flight on this branch.** Sprint-13 preflight planning fleet (PP-3/4/5/6 spec drafts for D-CSV-13b/14/16/17). Governance + spec corpus only; no .rs changes on this branch.
- Sibling repo `AdaWorldAPI/ndarray`: **PR #147 merged** (`d867b1c`) — vertical streaming substrate (`QualiaI4Row`, `QualiaStream`, `InferenceRow`, `InferenceStream`, `SplatFieldStream`) ships D-CSV-11. `par_*` rayon variants deferred behind `parallel` feature (sprint-14+).

**Historical (post-#225 era, retained for arc reference):**

- `claude/teleport-session-setup-wMZfb` — shipped PRs #223 / #224 / #225 (LAB-ONLY + AGI-as-SoA + I1-I11 + codec-sweep D0.6/D0.7).
- `claude/deepnsm-grammar-phase1` — Phase 1 PR #210, merged into main.
- `main` — up-to-date post #389 (sprint-12 Wave F + codex P2 follow-on).

## Active Integration Plans

- **`elegant-herding-rocket-v1`** — grammar / NARS / crystal / AriGraph (Phase 1 shipped in #210; Phase 2 queued).
- **`codec-sweep-via-lab-infra-v1`** (NEW 2026-04-20) — JIT-first codec sweep through lab endpoint; 1 upfront rebuild, unlimited candidates afterwards. D0.6 + D0.7 shipped in #225.

## Immediate Next Work

**Queued Work — sprint-13 (specs being drafted in the sprint-13-preflight fleet on this branch):**

- **D-CSV-13b** — SIMD vectorization of D-CSV-8 i4 MUL evaluation. **IN PR (sprint-13/W-I1 salvage)** on branch `claude/sprint-13-w-i1-salvage`. AVX-512F+BW path runtime-dispatched via cached `simd_caps()` (zero ndarray dep); NEON path correctness-only per spec §7; scalar fallback. Bench on Skylake-AVX512 host: 8.7× dk / 7.4× trust / 5.2× flow / 10.2× gate_disc / 3.1× mul_assess at batch 1024 — all SHIP gates met. `#[repr(u8)]` discriminants locked on `DkPosition`/`TrustTexture`/`FlowState` per spec §5 (I-LEGACY-API-FEATURE-GATED). 449 lance-graph-contract tests green including 5 new SIMD-vs-scalar parity tests over 10 sizes.
- **D-CSV-14** — on-Think method migration for D-CSV-12 splat ops (struct-method surface per L-20 lock; depends on D-CSV-11 ndarray streaming PR #147). Spec being drafted by PP-4.
- **D-CSV-16** — NEW sprint-13 entry. Spec being drafted by PP-5.
- **D-CSV-17** — NEW sprint-13 entry. Spec being drafted by PP-3.

**Sprint-14+ Phase F items (Backlog):**

- ndarray `parallel`-feature `par_*` rayon variants for `QualiaStream` / `InferenceStream` / `SplatFieldStream` (work-stealing).
- D-REUNIFY-4/5/6 carryover from causaledge64-mailbox-rename-soa-v1 (splat op fleet on `Think`, par_* method variants, OWL DOLCE / OntologyFilter wiring).

**`codec-sweep-via-lab-infra-v1` Phase 0 remainder (carry-over):**

- **D0.1** `WireCalibrate` + `WireTensorView` (object-oriented, 64-byte-aligned decode) (~180 LOC).
- **D0.2** `WireTokenAgreement` endpoint stub — I11 cert gate (~160 LOC).
- **D0.3** `WireSweep` streaming endpoint + Lance append (~200 LOC).
- **D0.5** `auto_detect.rs` reading `config.json` for `ModelFingerprint` (~140 LOC).
- Four test gates: `kernel_contract_test`, `amx_dispatch_test` (x86_64), `wire_object_surface_test`, `no_internal_serialisation_test`.

**`elegant-herding-rocket-v1` Phase 2 (still queued):**

- **D2** DeepNSM emits `FailureTicket` on low coverage (~150 LOC).
- **D3** Grammar Triangle wired into DeepNSM via `triangle_bridge.rs` (~220 LOC).
- **D5** Markov ±5 bundler with role-indexed VSA (~300 LOC).
- **D7** NARS-tested grammar thinking styles as meta-inference policies (~260 LOC).

## Deferred (do NOT propose these — they're explicitly parked)

**Sprint-12/13 explicit deferrals (2026-05-16):**

- **TD-COLLAPSE-GATE-SMALLVEC-1** — SmallVec optimization for `CollapseGateEmission` (currently Vec to preserve contract zero-dep invariant). Revisit only if profiling shows the heap allocation is hot.
- **TD-SIGMA-TIER-THRESHOLDS-1** — Σ10 VAMPE-coupled Jirak-derived threshold refinement (D-CSV-15). Hand-tuned acceptable through sprint-12 per `I-NOISE-FLOOR-JIRAK`; principled Jirak 2016 derivation forwarded to sprint-13+ VAMPE coupled-revival track.
- **ndarray `parallel`-feature `par_*` rayon variants** — productized substrate ships sequentially in PR #147; rayon work-stealing wraps deferred to sprint-14+ behind an opt-in feature gate.

**Long-running parks (pre-existing):**

- CausalityFlow TEKAMOLO extension (modal/local/instrument + beneficiary/goal/source, 9 total) — struct change deferred until after Phase 2.
- D8 story-context bridge, D9 ONNX arc export, D10 Animal Farm validation, D11 bundle-perturb emergence — Phase 3/4.
- Named Entity pre-pass (NER) — biggest OSINT blocker, separate PR.
- FP_WORDS = 160 migration (currently 157) — coordinated ndarray change.
- Crystal4K 41:1 persistence compression.
- 200-500 YAML TEKAMOLO templates per language — future training pipeline.
- Python/TypeScript grammar-stack convergence.

## If You're Tempted to Propose Something

Check this file first. Then check the KNOWLEDGE_INDEX.md for which
docs cover your domain. Then load only those docs. If you're still
uncertain whether something exists, grep the actual source before
proposing a new type.

The fastest way to waste 30 turns is to re-invent what's already in
the contract. This file exists to prevent that.

---

## 2026-05-05 — Recently Shipped backfill (PRs #244–#335)

> The "Recently Shipped PRs" table above stops at #243 (last refreshed 2026-04-21). Roughly 50 PRs have merged since. This section retrofits them.

| PR | Merged | Title | What it added (one-line) |
|---|---|---|---|
| **#335** | 2026-05-05 | Claude/thought cycle soa integration plan | Two new knowledge docs: gaussian-splat-cam-plane-workaround + entropy-budget-codebook-superposition |
| **#330** | 2026-05-01 | docs: add Cursor Cloud specific instructions to AGENTS.md | AGENTS.md section: ndarray path, CI commands, fmt-drift inventory, bgz-tensor known failures |
| **#329** | 2026-05-01 | style: apply rustfmt to contract lib.rs + python bindings | Tier-A rustfmt drift in contract lib.rs + python bindings (no semantic change) |
| **#328** | 2026-05-01 | ci(test): add lance-graph-contract unit tests to test gate | `cargo test -p lance-graph-contract --lib` added to CI rust-test.yml |
| **#327** | 2026-05-01 | style(shader-driver): drop double-space alignment in bindspace.rs | Two-line rustfmt drift fix in bindspace.rs introduced by #323 |
| **#326** | 2026-05-01 | fix(sigma-propagation): correct log_norm_growth_negative test seed | Fix broken test from #322: seed at 4·I not I so attenuation reduces log-norm |
| **#325** | 2026-04-30 | chore(toolchain): bump pin 1.94.0 → 1.94.1 | rust-toolchain.toml bumped to 1.94.1 to match sibling repos |
| **#324** | 2026-04-30 | feat(shader-driver): Pillar-7 α-front-to-back-merge sink mode (B5) | AlphaFrontToBack MergeMode + EWA Kerbl-2023 compositing in stage [7] |
| **#323** | 2026-04-30 | feat(cognitive-shader-driver): add Σ-codebook-index column to FingerprintColumns (B2) | FingerprintColumns.sigma u8 column (+1 byte/row, 0.02% overhead) |
| **#322** | 2026-04-30 | feat(contract): promote EWA-Sandwich Σ-propagation kernel to contract (B1) | sigma_propagation.rs: Spd2, ewa_sandwich, log_norm_growth, pillar_5plus_bound |
| **#321** | 2026-04-30 | fix: 10 pre-existing test failures (cosine_distance, arigraph, parse_triplets) | Fixed cosine inversion, Stagnant ordering, quality_window clear, SPO arg order |
| **#320** | 2026-04-30 | ci: declare rustfmt + clippy as pinned-toolchain components | rust-toolchain.toml gets components=[rustfmt,clippy]; fixes CI fmt failure |
| **#319** | 2026-04-30 | fix(transcode): per-month day-validity in parse_iso_date_to_days | Gregorian per-month + leap-year gate before civil_to_days |
| **#316** | 2026-04-30 | feat(transcode): round-3 typed-value resolver for triples_to_batch | triples_to_batch_with_resolver: Currency→f32, Date→Date32, Id→u64 |
| **#315** | 2026-04-30 | ci: revert ndarray-branch pin — PR #115 landed on master | Remove temp ndarray branch pin from rust-test.yml + style.yml |
| **#314** | 2026-04-30 | docs(vision): clear post-F1 staleness items in medcare-foundry-vision.md | §1–§4 DRAFT/forward-tense/PR-N placeholders replaced with real anchors |
| **#313** | 2026-04-30 | feat(transcode): Phase-2-B triples_to_batch (ExpandedTriple → RecordBatch) | ExpandedTriple stream → N-row RecordBatch, lenient-Utf8, 19 tests |
| **#312** | 2026-04-30 | feat(transcode): Phase-2-A pushdown classification (Inexact for recognised filters) | OntologyTableProvider classifies entity_type/predicate/nars filters as Inexact |
| **#311** | 2026-04-30 | docs(vision): mark F1 shipped, restate next deliverable as F2 | medcare-foundry-vision.md §7: F1 parity shipped; F2 RBAC is next posture |
| **#310** | 2026-04-30 | feat(transcode): r2 fixes — typed Arrow + codec_route + partial writes + CachedOntology | Currency/Date/Id→typed Arrow; CachedOntology; validate_route; from_columns_partial |
| **#309** | 2026-04-30 | feat(callcenter::transcode): outer ↔ inner ontology mapper + parallelbetrieb | transcode submodule: zerocopy, cam_pq_decode, spo_filter, ontology_table, parallelbetrieb |
| **#308** | 2026-04-30 | feat: bilingual ontology DTO surface + bgz-tensor workspace inclusion | OntologyDto locale projection; smb_ontology + medcare_ontology; bgz-tensor in workspace |
| **#307** | 2026-04-30 | refactor: dedup FNV-1a — one canonical hash::fnv1a in contract | contract::hash::fnv1a const fn; 8 call sites unified |
| **#306** | 2026-04-30 | feat(G4): verb_table tense modulation (Quirk CGEL grounded) | 12 VerbFamily priors + tense_modifier → 144 unique cell values |
| **#305** | 2026-04-30 | feat(G3): DisambiguateOpts builder + deepnsm caller wiring real fingerprint | DisambiguateOpts builder; sign_binarize_to_binary16k; disambiguator_glue.rs |
| **#304** | 2026-04-30 | feat(G1): Pearl 2³ causality footprint with PAD-model qualia mapping | compute_pearl_mask() 3-bit SPO→CausalMask; PAD qualia footprint replaces 0.5 |
| **#303** | 2026-04-30 | feat(F6): FNV-1a scent with scent_u64 accessor + birthday collision tests | scent() FNV-1a fold-to-u8; scent_u64() full 64-bit digest; 10 tests |
| **#302** | 2026-04-30 | feat(F3): LanceAuditSink with temporal timestamps + full schema round-trip | LanceAuditSink → Lance dataset append; temporal timestamp; O(1) scan_back |
| **#301** | 2026-04-30 | feat(F1): ColumnMaskRewriter full-tree expression walk + Hash UDF hard-fail | Full-tree OptimizerRule covering Filter/Aggregate/Join; NotYetWiredHashUdf |
| **#300** | 2026-04-30 | feat(LF-12): Pipeline DAG with StepId derivation + OrchestrationBridge adapter | PipelineDag Kahn's algorithm; FNV-1a StepId; execute_via_bridge; cycle detection |
| **#299** | 2026-04-29 | revert #294/#295/#296 + clean on top | Reverts #294–#296 confabulation; corrects probe routing (M1/P2-P4 → shader-lab) |
| **#296** | 2026-04-29 | ideas: COCA-Bundle vs Jina-CLAM bucket comparison (**REVERTED by #299**) | IDEAS.md Open entry for COCA/Jina probe (premise flawed; reverted) |
| **#295** | 2026-04-29 | docs: probe-queue data-available followup (**REVERTED by #299**) | bf16-hhtl-terrain.md data-available update (inherited bad routing; reverted) |
| **#294** | 2026-04-29 | docs(probe-queue): honest "needs production data" assessment (**REVERTED by #299**) | bf16-hhtl-terrain.md probe routing table (wrong routing; reverted) |
| **#293** | 2026-04-29 | jc: drain Probe P1 (γ-phase-offset ranking discrimination) → PASS | probe_p1_gamma_phase.rs; P1 PASS: min Spearman ρ=-0.963 (Dupain-Sós) |
| **#292** | 2026-04-29 | docs(board): posthoc-correct PRs #290 #291 via canonical board mechanism | CONJECTURE banners; 5 Open IDEAS.md entries; 2 EPIPHANIES.md entries |
| **#291** | 2026-04-29 | docs: idea journal — proposed application pillars 7/8/9 captured | IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md with Pillars 7/8/9 + PASS criteria |
| **#290** | 2026-04-29 | docs: idea journal — streaming-hydration + fractal-codec captured | IDEA_JOURNAL_2026_04_29_STREAMING_HYDRATION.md separating two ideas |
| **#289** | 2026-04-29 | jc: Pillar 6 — EWA-Sandwich Σ-push-forward | ewa_sandwich.rs; Pillar 6: 10000/10000 PSD-preserving hops; KS bound tightness 1.467× |
| **#288** | 2026-04-29 | jc: Σ-Codebook Viability Probe — rules out CausalEdge64 8→16B expansion | sigma_codebook_probe.rs; R²=0.9949 at k=256; CausalEdge64 stays 8 bytes |
| **#287** | 2026-04-29 | jc: Pillar 5++ — Düker-Zoubouloglou Hilbert-space CLT | dueker_zoubouloglou.rs; Pillar 5++: bundle-of-N in ℝ^16384 → Gaussian limit in ℓ² |
| **#286** | 2026-04-29 | jc: Pillar 5+ — Köstenberger-Stark concentration on Hadamard 2×2 SPD | koestenberger.rs; Pillar 5+: tightness 0.969× on SPD manifold |
| **#285** | 2026-04-29 | Re-land #283 unlocks (Quantum, Disambiguator, verb_table, animal-farm) | Quantum mode, Disambiguator trait, verb_table, animal-farm harness; PhaseTag overflow fix |
| **#284** | 2026-04-29 | Re-land #281 unlocks (PolicyRewriter, DomainProfile) | PolicyRewriter trait, ColumnMaskRewriter, DomainProfile HIPAA thresholds |
| **#282** | 2026-04-29 | fix: Grammar/Markov hardening — slice unification, kernel wiring | CRITICAL slice fix; rotate_right removed; coherence kernel wired; 363 tests |
| **#280** | 2026-04-29 | fix: Foundry hardening — sealed RLS, VecDeque audit, URL decode, Plugin handshake | Sealed RLS default; O(1) audit ring; FNV-1a; URL decode; Plugin handshake; 58 tests |
| **#279** | 2026-04-29 | feat: DeepNSM grammar parser — Markov ±5 bundler, role keys, thinking styles | D0/D2/D3/D4/D5/D6/D7: MarkovBundler, RoleKeySlice, GrammarStyleConfig, 12 YAML configs |
| **#278** | 2026-04-29 | feat: Foundry parity — RLS rewriter, audit log, PostgREST, with_registry | LF-3/DM-7 RLS; LF-90 audit; DM-8 PostgREST stub; LanceMembrane::with_registry; 35 tests |
| **#277** | 2026-04-28 | plan: unified Foundry roadmap for SMB + MedCare (corrects #276 framing) | foundry-roadmap-unified-v1.md; correct scale decisions per FormatBestPractices.md |
| **#276** | 2026-04-28 | plan: Foundry Consumer Parity — shared ontology + UNKNOWN resolutions | foundry-consumer-parity-v1.md; 5 callcenter UNKNOWNs resolved; DM-8 unblocked |
| **#275** | 2026-04-28 | feat: add lancedb 0.27.2 + pin lance =4.0.0 | lancedb=0.27.2 optional dep; lance exact-pinned =4.0.0 for compat |
| **#274** | 2026-04-27 | fix: F-01 identity-tear race + F-08 bounds check + F-09 poison recovery | Single ActorState RwLock; poison recovery; push bounds check |
| **#273** | 2026-04-27 | feat: bump lance 2→4 + datafusion 51→52 + deltalake 0.30→0.31 | Version bumps + API break fixes (invalid_input, DeltaTableProvider migration) |
| **#272** | 2026-04-27 | feat: Column H — EntityTypeId on BindSpace (Phase 1 of 4) | EntityTypeId u16 on BindSpace; push_typed(); 1-based index; 4 tests |
| **#271** | 2026-04-27 | plan: BindSpace Columns E/F/G/H — 4→8 SoA integration plan | bindspace-columns-v1.md; 24 deliverables; 7 SOUND / 7 CAUTION / 0 WRONG |
| **#270** | 2026-04-26 | ci: remove typos spell-check job (too many false positives) | Removed crate-ci/typos from style.yml; cargo fmt --check remains |
| **#269** | 2026-04-26 | feat: Distance trait + SIMD Hamming/cosine wiring + PaletteDistanceTable + Dockerfile docs | Distance trait; SIMD Hamming/cosine wiring; PaletteDistanceTable 128KB; Dockerfile.md |


---

## 2026-05-07 — Append: lance-graph-ontology shipped (commit 4cf9a26, branch claude/create-graph-ontology-crate-gkuJG)

(Per APPEND-ONLY rule: this dated annotation augments the "Recently Shipped PRs" table and "Current Contract Inventory" snapshot above. Treat the row below as the new top-of-table entry; treat the inventory paragraph below as a new top-of-inventory entry.)

### Recently Shipped PRs — new top row

| PR | Merged | Title | What it added |
|---|---|---|---|
| **(open / pending merge)** | *(open)* | feat(lance-graph-ontology): scaffold OGIT-canonical ontology spine | New workspace member `crates/lance-graph-ontology/` (~3000 LOC, 28 tests = 16 inline + 12 integration). Phases 3-5 of the v4 plan: scaffold + TTL hydration + tenant bridges. Public surface: `OntologyRegistry`, `NamespaceBridge` trait, `NamespaceId`, `OgitUri`, `SchemaPtr`, `SchemaKind`, `MappingProposal`, `MappingProposalKind`, `MappingRow`, `MappingHandle`, `HydrationReport`, `HydrationFailure`, `BridgeError`, `Error`, `SchemaSource` trait, `EntityRef`, `EdgeRef`, `OntologyAssembler`, `SemanticTypeMap`, `TtlSource`. Default tenant bridges: `bridges::WoaBridge`, `bridges::MedcareBridge`, `bridges::OgitBridge`. Feature-gated `lance_cache::LanceWriter` (under `lance-cache` feature, gated to keep zero-protoc compile path). Builds on prior commit `edef321` (recon + SPO-1 decision: federated two-layer cache, Option B). |

### Current Contract Inventory — new entry

**`lance-graph-ontology`** (new crate, 2026-05-07): consolidates per-tenant bridge multiplication into one ontology spine. OGIT becomes the canonical TTL ontology source; Lance is the (feature-gated) runtime dictionary cache; tenant bridges become thin scoped views over the shared registry. Public types: `OntologyRegistry`, `NamespaceBridge` trait, `NamespaceId`, `OgitUri`, `SchemaPtr`, `SchemaKind`, `MappingProposal`, `MappingProposalKind`, `MappingRow`, `MappingHandle`, `HydrationReport`, `HydrationFailure`, `BridgeError`, `Error`, `SchemaSource` trait, `EntityRef`, `EdgeRef`, `OntologyAssembler`, `SemanticTypeMap`, `TtlSource`. Default tenant bridges: `bridges::WoaBridge`, `bridges::MedcareBridge`, `bridges::OgitBridge`. 28 tests passing (16 inline + 12 integration). Feature-gated Lance persistence under `lance-cache` (kept off by default so the crate compiles without `protoc`, which `lance-encoding`'s build-script requires). Branch `claude/create-graph-ontology-crate-gkuJG`; commit `4cf9a26`; prior recon + decision in `edef321` (`.claude/RECON_ONTOLOGY_CRATE.md`, `.claude/DECISION_SPO_ARIGRAPH.md`).

---

## 2026-05-07 — Sprint-2: Unified OGIT Architecture synthesis (recently shipped — documentation tier)

> **APPEND-ONLY annotation.** Per the governance rule above, this section augments — does not edit — prior content. Treat as the new top-of-state. Branch: `claude/unified-ogit-architecture-synthesis`.
>
> Sprint-2 was a 12-agent + meta-review coordinated burst. **Zero code changes; documentation tier only.** It captures 16 turns of architectural conversation (2026-05-07) as a unified pattern-recognition framework over already-shipped substrate, plus three concrete next-PR sub-plans and one proof-of-vision plan. The dominant finding: ~80% of the "unified OGIT architecture" we were about to design is **already shipped**; recognising this drops architecture entropy by **−11** with no code written.

### Sprint-2 deliverables (12 workers + meta)

**New plan-docs (4)**

| File | Size | Worker | Purpose |
|---|---|---|---|
| `.claude/plans/unified-ogit-architecture-v1.md` | ~22 KB | W1 | Master synthesis: 15 patterns (A-O) + Tier 0-4 stack + proof-of-vision. Canonical reference for the unified OGIT architecture. |
| `.claude/plans/ogit-g-context-bundle-v1.md` | ~10 KB | W10 | Tier-1 sub-plan: G-overlay wiring; Patterns A (G-slot) + B (context-bundle) + C (per-cycle cascade). |
| `.claude/plans/compile-time-consumer-binding-v1.md` | ~10 KB | W11 | Tier-2 sub-plan: compile-time consumer binding + ractor; Patterns E (consumer-binding) + F (zero-overhead actor seam). |
| `.claude/plans/anatomy-realtime-v1.md` | ~12 KB | W12 | Proof-of-vision: north-star realtime anatomy demo end-to-end across the unified stack. |

**New knowledge doc (1)**

| File | Size | Worker | Purpose |
|---|---|---|---|
| `.claude/knowledge/tier-0-pattern-recognition.md` | ~13 KB | W2 | File→pattern map covering ~30 already-shipped files. Read this FIRST in any future session that touches OGIT architecture to avoid the Discovery-Loop anti-pattern. |

**Board appends (5, append-only governance preserved)**

| File | Worker | Append summary |
|---|---|---|
| `.claude/patterns.md` | W3 | Appended **Pattern Recognition Framework**: 15 patterns A-O catalogued + new Anti-Pattern **"Designing What's Already Built"**. |
| `.claude/board/EPIPHANIES.md` | W4 | Appended **17 architectural epiphanies**: E-OGIT-1 through E-RECOGNITION-OVER-DESIGN-17. |
| `.claude/board/TECH_DEBT.md` | W5 | Appended **11 TD entries**: TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11, each with effort estimate. |
| `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` | W6 | Appended **5 row reframes** (THINK-1 5→3, HEEL-1 4→2, ADJ-THINK-1 4→2, CRYSTAL-1 4→2, CAM-DIST-1 3→2) + 15-pattern absorption table. **Net entropy delta: −11**. |
| `.claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md` | W7 | Appended RECOGNITION-1 meta-finding row + Anti-Pattern surfaced ("Designing What's Already Built"). |

**Index update (1)**

| File | Worker | Update |
|---|---|---|
| `.claude/board/INTEGRATION_PLANS.md` | W8 | Indexed the 4 new plan-docs (W1 master + W10 + W11 + W12). |

**Sprint coordination (CCA2A pattern, `/sprint-log-2`)**

- `.claude/board/sprint-log-2/SPRINT_LOG.md` — master coordination index.
- `.claude/board/sprint-log-2/agents/agent-W{1..12}.md` — per-agent append-only logs (12 files).
- `.claude/board/sprint-log-2/meta-1-review.md` — meta agent brutally-honest review.
- `.claude/board/sprint-log-2/agents/agent-W9.md` — this worker's handover log.

### Aggregate impact

- **15 architectural patterns (A-O)** named and catalogued.
- **~80% of the "unified OGIT architecture" is recognised as already shipped** — Patterns H, M, N, O at substrate level; Pattern F shape proven by gRPC.
- **~20% genuinely new wiring work** captured as TECH_DEBT entries with effort estimates (TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11).
- **Net entropy reduction from recognition alone: −11** (no code changes; 5 row reframes + 15-pattern absorption).
- **Totals shipped this sprint:** 4 new plan-docs + 1 knowledge doc + 5 board appends + 1 index update + sprint-log-2 scaffolding (1 master + 12 agent logs + 1 meta review).

### What this enables

Future sessions that read `.claude/knowledge/tier-0-pattern-recognition.md` first will avoid the **Discovery-Loop anti-pattern at architectural scale** — the same anti-pattern `.claude/patterns.md` warns about at cycle level (proposing concepts that already exist in workspace).

The master plan-doc `.claude/plans/unified-ogit-architecture-v1.md` provides the canonical reference for the unified OGIT architecture. The three sub-plans give concrete next-PR scope:

- **Tier 1 next PR** — `.claude/plans/ogit-g-context-bundle-v1.md` (G-overlay wiring).
- **Tier 2 next PR** — `.claude/plans/compile-time-consumer-binding-v1.md` (compile-time consumer binding + ractor).
- **Proof of vision** — `.claude/plans/anatomy-realtime-v1.md` (north-star demo).

### Cross-references

- All sister deliverables listed above (W1–W12 + meta).
- 16-turn architectural conversation (2026-05-07).
- Pre-existing plans absorbed into the unified framework: `lance-graph-ontology-v5` (PR #355), `palantir-parity-cascade-v2` (PR #353), `ogit-cascade-supabase-callcenter-v1` (PR #355).
- Substrate already shipped (Patterns H/M/N/O): see "Current Contract Inventory" and "Current AriGraph Inventory" sections above; especially `lance-graph-ontology` (commit `4cf9a26`), `cognitive-shader-driver` BindSpace SoA (PR #204+ thru #323), `crystal/` Vsa16kF32 sandwich (PR #208/#209), `cam/` codec cascade (PR #225).

### Brutally-honest self-review (W9)

- **In scope:** append-only update to `LATEST_STATE.md`. Did not edit any prior content. Verified the file's existing closing section (`2026-05-07 — Append: lance-graph-ontology shipped`) is preserved verbatim.
- **Risk:** the "~80% already shipped" claim is W1/W2's recognition assertion, not independently re-verified by W9. This section reports it as the synthesis output; the canonical evidence lives in `tier-0-pattern-recognition.md` (W2) and the entropy ledger reframe rows (W6).
- **Governance:** append-only preserved. No deletions. No edits to the prior `## 2026-05-07 — Append: lance-graph-ontology shipped` section. Section heading matches the spec exactly.
- **What this section does NOT do:** it does not edit the top-of-file "Last updated" line (would violate append-only); it does not edit the "Recently Shipped PRs" table (Sprint-2 shipped no PRs); it does not edit "Active Branches" (Sprint-2 is documentation tier on a branch that has not yet merged).

## 2026-05-12 — Sprint-3: Tier-1 Implementation Specs (PR #360 + #361 + post-#360 substrate sweep)

**PR #360** (sprint-3 main): 11 PR-X-1 specs covering 7 design-phase patterns A/B/C/D/E/F/J + 3 trivia closures + supporting docs. ~140 KB across `.claude/specs/`. Engineer can now execute Tier-1 in ~6 working days parallelized (per W10 sequencing).

**PR #361** (post-#360 corrections): PR-F-1 supervisor must skip inert bundles (DOLCE/FMA have consumer_pointer=None by design); PR-E-1 build script must emit data-only (no consumer crate refs) to avoid Cargo dependency cycle. Both fixed via append-only correction sections; inventory-crate self-registration recommended for actor binding.

**Post-#360 substrate-recognition sweep** (this PR): 3 of 11 specs reclassified PARTIALLY SHIPPED:
- Pattern A: SchemaPtr.ontology_context_id + NamespaceRegistry::seed_defaults already ship; PR-A-1 reduces to ~150 LOC / 1 day
- Pattern C: BridgeFromRegistry + 3 impls + woa-rs#2 + medcare-rs#110 consumer scaffolds already ship; PR-C-1 reduces to ~80 LOC / ½ day
- Pattern D: parse_ttl_directory_with_provenance + attach_provenance already ship; PR-D-1 reduces to ~250 LOC / 1-2 days (OWL/RDF-XML adapter only)

Compressed sprint-3 critical path: ~6 days → ~3-4 days parallelized. The genuinely-new ~5-pattern set is B (context bundle), E (manifest-modules), F (ractor port), G (inheritance protocol), J (INT4-32D atoms).

### New knowledge docs (sprint-3 substrate-sweep)

- `.claude/knowledge/pattern-recognition-cross-source.md` — A-O ↔ Pillars 0-4 ↔ `.grok/` ↔ shipped substrate matrix (4 parallel taxonomies cross-referenced)
- `.claude/knowledge/cca2a-sprint-prompt-template.md` — substrate-grep checklist + wrong-repo guardrail + pattern-letter discipline (mandatory pre-spawn template for future sprints)

### Anti-Pattern recurrence captured

The "Designing What's Already Built" anti-pattern (introduced PR #358) recurred in sprint-3's own design (PR-A-1/PR-C-1/PR-D-1 over-scoped because they didn't sweep post-#355 substrate). The correction PR formalizes the substrate-grep checklist as mandatory before any new spec.

### Recurring failure mode: wrong-repo error

Sprint-2 W7 → ndarray; sprint-3 W9 → ada-consciousness. Both corrected via main-thread pygithub recovery. Wrong-repo guardrail snippet now mandatory in every worker prompt (per `.claude/knowledge/cca2a-sprint-prompt-template.md`).

### Cross-references

- `.claude/specs/sprint-3-execution-plan.md` (W1 master)
- `.claude/specs/sprint-3-pr-graph.md` (W10 sequencing — to be updated for compressed timeline)
- `.claude/specs/pr-{a,b,c,d,e,f,j}-1-*.md` (11 PR-ready specs; A/C/D have appended CORRECTION sections)
- `.claude/specs/consumer-crate-template.md` (W8; re-target from hubspo-rs hypothetical to woa-rs/medcare-rs precedent)
- `.claude/specs/ogit-g-smoke-test.md` (W11 validation)
- `.claude/specs/trivia-prs-bundle.md` (W12 — 3 quick wins parallel-shippable)
- `.claude/board/sprint-log-3/{SPRINT_LOG.md,agents/agent-W1..W12.md,meta-1-review.md,sprint-summary.md}`

PR sequence: #360 → #361 → post-#360 substrate-sweep (this PR).

---

## APPEND-ONLY annotation — D-ODOO-1 odoo hydrator (2026-05-27)

> Per the APPEND-ONLY governance rule, this section augments — does not edit — prior content. Treat as the new top-of-state. Branch: `claude/lance-graph-att-activate-Jd2iZ`.

### Current Contract Inventory — new entry

- **`OGIT::ODOO_V1` = (50, 1)** — new OGIT G slot (first manifest-declared slot above SKR03BAU=42). Source: `modules/odoo/manifest.yaml` (`ogit_g: ODOO`, `inherits_from: fibofnd`, 17 entity_types u16=4300..4316). Registered in `crates/lance-graph-contract/build.rs` CANONICAL_SLOTS as `("ODOO", 50)`; build regenerates `OUT_DIR/ogit_namespace.rs` accordingly.

### New module surface (`lance-graph-ontology`)

- **`hydrators::odoo`** — Layer-1 odoo extraction hydrator (four-way alignment seam). `hydrate_odoo(registry)` + `hydrate_odoo_from(paths, registry)`; `inherits_from: Some(OGIT::FIBOFND_V1.0)`; edge whitelist {rdfs:subClassOf, owl:equivalentClass, rdfs:subPropertyOf, owl:equivalentProperty}. Re-exported from `lib.rs`.
- **`hydrators::dolce_odoo`** — odoo DOLCE suffix classifier (Seam decision 2, own module per Open-question 3). `pub fn classify_odoo(iri: &str) -> DolceCategory` + `pub enum DolceCategory { Endurant, Perdurant, Quality, AbstractEntity }`. Re-exported from `lib.rs`. (Doc-noted: canonical DUL renames Endurant→Object / Perdurant→Event.)

### New data artifacts

- `data/ontologies/odoo/odoo-core.ttl` — 17 odoo core classes (`odoo: <https://ada.world/onto/odoo#>`).
- `data/ontologies/odoo/alignment/odoo-to-fibo.ttl` + `odoo-to-skr.ttl` — Layer-2 `owl:equivalentClass`/`owl:equivalentProperty` alignment axioms (Seam decision 1 / Option B: odoo inherits existing FIBO/SKR slots, no new CAM family).

### Tests

`cargo test -p lance-graph-ontology` → 127 passed / 0 failed (+7 odoo integration tests across `tests/odoo_hydrator_smoke.rs` + `tests/odoo_dolce_classifier.rs`, incl. the full 21-row seam classifier matrix; +4 lib unit tests). `cargo test -p lance-graph-contract` → 449 passed / 0 failed.

### Relationship to prior art

`lance-graph-callcenter::odoo_alignment` already ships a parallel `dolce_odoo()` + `DolceMarker` + `ODOO_SEED` table. This is the ontology-side counterpart (TTL hydration into `OntologyRegistry`); consistent doctrine (Option B, same pivots), distinct crate + distinct `DolceCategory` enum per task spec. Cross-crate dedup is a possible follow-up, not done here.

---

## 2026-05-28 — Append: PR #422 shipped (post-merge governance for the #418/#419 review handover)

(Per APPEND-ONLY rule: this dated annotation augments the "Recently Shipped PRs" table above. Treat the row below as the new top-of-table entry.)

### Recently Shipped PRs — new top row

| PR | Merged | Title | What it added |
|---|---|---|---|
| **#422** | 2026-05-28 | docs(handover): PR #418/#419 review + surreal/mailbox/Baton/SoA-as-BindSpace-surrogate plan map | Read-only synthesis handover. New `.claude/handovers/2026-05-28-1200-pr-418-419-surreal-mailbox-baton-plan-map.md` (~310 LOC, 7 sections): §1 PR #418 review (verdict *sound, merge-ready as a spec* + 3 substantive notes on the bare-columns-vs-hot-thought footprint distinction, `E-RUBICON-RACTOR` as honest post-hoc CONJECTURE, OQ-4 doctrinal gating); §2 the **SurrealDB role correction** (Zone-2 cold store → *view over leading LanceDB*, recorded in `E-RUBICON-RACTOR` + plan §2.7); §3 the plan corpus map (8 plans + 9 epiphanies + `PR-NDARRAY-MIRI-COMPLETE → D-CE64-MB-1-impl → D-MBX-1..6` dep chain + `TD-RESONANCEDTO-DUP-1`); §4 brief #419 review (unrelated to surreal/mailbox; the 14 `NEEDS-INPUT` blockers are the real gate for D-ODOO-SAV-4); §5 navigability meta-finding (the surreal POC docs lack a supersedure pointer); §6 action surface; §7 cross-refs. Board appends: `EPIPHANIES.md` ← `E-SURREAL-POC-UNANNOTATED-SUPERSEDURE` (FINDING / navigability); `AGENT_LOG.md` ← session row. **Zero code change**; 3 files; +310/-0. Branch `claude/lance-graph-ontology-review-Pyry3` → `main`. Merge commit `984512b` on top of `a29946b` (the doc commit, rebased onto post-#421 `main` to resolve the AGENT_LOG append-vs-append conflict by keeping both #421's AXIS-B row and this PR's session row in chronological order). |

---

## 2026-05-28 — Append: PR #425 shipped (deps comment cleanup + [patch.crates-io] ndarray declared intent)

(Per APPEND-ONLY rule: this dated annotation augments the "Recently Shipped PRs" table above. Treat the row below as the new top-of-table entry.)

### Recently Shipped PRs — new top row

| PR | Merged | Title | What it added |
|---|---|---|---|
| **#425** | 2026-05-28 | deps(workspace): clean BLOCKED comments; record 6.0.0→6.0.1 block (lancedb 0.29.0 transitive) | Workspace `Cargo.toml` cleanup + finding. Replaces the stale `BLOCKED-(A)/(B)/(D)` comment block (predates #423's 4→6 / 0.27→0.29 / 52→53 / 57→58 bump) with a dated `RESOLVED(A)/(B)/(D)` record pointing to #423 + the live crate-level pins. Records the user-authorised follow-on patch `lance 6.0.0 → 6.0.1` as **CURRENTLY BLOCKED** by `lancedb 0.29.0`'s transitive `lance = "=6.0.0"` requirement (proof: `cargo check` → `versions that meet the requirements '=6.0.0' are: 6.0.0`; resolution paths: wait for lancedb 0.29.1+, drop strict-=, or `[patch.crates-io]` override). Adds `[patch.crates-io] ndarray = { git = "https://github.com/AdaWorldAPI/ndarray.git", branch = "master" }` per user directive — declared intent; cargo emits `warning: patch ndarray v0.17.2 was not used in the crate graph` because lance-index 6.0.0 pins `ndarray = "0.16.1"` (semver gap, fork at 0.17.2). `Cargo.lock` now contains a `[[patch.unused]]` entry that makes the gap visible at every build. Files `TD-NDARRAY-PATCH-0_16` in `TECH_DEBT.md`. Codex P2 (`59ef97e`) flagged the original false RESOLVED(D) claim; fix in `2e001a5`/`8f3913b`/`1444f78`. Merge commit `1a3abfb8`. |

## 2026-05-28 — Append: PR #427 shipped (bindspace→mailbox migration wave A1-A4)

(Per APPEND-ONLY rule: appended after PR #425's annotation above. Treat the row below as the new top-of-table entry.)

### Recently Shipped PRs — new top row

| PR | Merged | Title | What it added |
|---|---|---|---|
| **#427** | 2026-05-28 | feat(mailbox-soa): bindspace→mailbox migration wave A1-A4 (thoughtspace columns + transitional routing + WitnessTable + plan §10) | First implementation pulse of `bindspace-singleton-to-mailbox-soa-v1` (PR #418 plan). **A1** (`1df12eca`, +103): 4 thoughtspace columns on `MailboxSoA` (`edges`/`qualia`/`meta`/`entity_type`) + 8 row accessors + zero-init in `new()` + reset in `reset_row()`. **A2** (`61b641d5`, +42): transitional `mailboxes: HashMap` + `with_mailbox()` builder + `mailbox()` accessor on `ShaderDriver` — sibling-shape, additive, singleton untouched. **A3** (`ef848a34`, +187): new `WitnessTable` + `WitnessEntry{ mailbox_ref, spo_fact_ref }` primitive in `lance-graph-contract::witness_table` (zero-dep, 3 unit tests, `const fn new`, `get`/`set` bounds-checked). **A4** (`0f448730`, +36): plan §10 "2026-05-28 architectural refinements" appended — 7 ratified findings (SoA-Lance ≠ cascade; cascade is not an index space; 64k-256k mailbox envelope ~360 MB - 1.4 GB RAM-resident; W-slot = per-cohort witness table not corpus pointer; cascade granularities = CPU/cache boundaries 64/256/4096/16384; `simd_soa.rs` introspects per-SoA shape; SoA invariant spawn → commit, two egress modes external/internal) + 2 surviving OQs (OQ-MBX-8 `persisted_row` vs Lance native versioning; OQ-MBX-15′ container scoping granularity). Codex P1 follow-on `f541b280`: widen `WitnessEntry.mailbox_ref` u16 → u32 + correct `Option<u64>` size doc. **457 contract+driver tests passing**, zero new behavioural code outside the columns/builder/primitive. Singleton `Arc<BindSpace>` NOT removed (sibling pattern); cutover in a downstream slice (D-MBX-3/4). Merge commit `84296118`. Author session — this governance row is the post-merge close-out; per-deliverable AGENT_LOG entries D-MBX-A1..A4 already prepended at branch HEAD pre-merge. |

## 2026-05-30 — Append: new standalone crate `lance-graph-arm-discovery` (Aerial+ transcode, D-ARM-13) on branch `claude/jolly-cori-clnf9`

(Per APPEND-ONLY rule: dated annotation augmenting the "Current Contract Inventory" snapshot above. Branch work, not yet merged — recorded so a new session does not re-derive the crate.)

### Current Contract Inventory — new entry

- **`crates/lance-graph-arm-discovery`** (NEW, **excluded** standalone zero-dep crate; build via `cargo test --manifest-path crates/lance-graph-arm-discovery/Cargo.toml`). The **Aerial+** Rust transcode (Karabulut 2025, 2504.19354v1) — the upstream runtime-data proposer leg of `streaming-arm-nars-discovery-v1`. Public surface: `encode::{FeatureSpec, Dataset}`, `rule::{Item, CandidateRule, Proposer}`, `translator::{arm_to_nars, NarsTruth, CandidateTriple, FeedProjector, DebugProjector, NARS_PERSONALITY_K}`, `ndjson::to_ndjson`, and (feature `aerial`, default-on) `aerial::{Rng, AerialAutoencoder, AerialParams, AerialProposer, extract_rules, ExtractParams}`. 35/35 tests, clippy `-D warnings` clean. Emits the `{"s","p","o","f","c"}` ndjson the SPO store loader reads; `(f,c)` == `TruthValue::new(f,c)` == `ruff_spo_triplet::Triple{f,c}`. Determinism boundary: nondeterministic AE is seeded + feature-gated + emits *proposals* only. Synergy map: `.claude/knowledge/aerial-arm-ruff-spo-codegen-synergies.md`. Status board: D-ARM-13 (Shipped on branch) + D-ARM-SYN-1/2/3 (Queued). **Not** in `lance-graph-contract` yet — `rule`/`translator` are the local seam until D-ARM-1/2 land the shared carriers.

> **2026-06-01 — PR-in-flight (autoattended)** (D-EW64-3/4): `lance_graph_contract::episodic_edges` gains `EpisodicEdges64::{coldest, contains, promote_into}` + the `DemotionSink` trait. `coldest()` = the eviction victim (symmetric to `strongest()`); `contains()` = family-discriminating membership; `promote_into(e, sink)` = `promote` routing the evicted (coldest) edge to a `DemotionSink` — the hot→cold connectome exit. `DemotionSink` impls (surreal/LanceDB-LIVE "wingman", `E-SUBSTRATE-IS-THE-SCHEDULER`) are deferred + GATED on OQ-11.6. Zero-dep; contract lib 545 green; default clippy clean; `episodic_edges.rs` pedantic+nursery clean.

> **2026-06-01 — Shipped (autoattended, 5-agent council)** (D-ATOM-4/RawEdge): `contract::counterfactual` wired into `lib.rs` (was orphaned); `RawEdge(i8)` mantissa-only **structural** impl of `EpisodicEdge` (`size_of==1` — a u64 newtype could read plasticity 50–52); `deposit_counterfactual` v2 filled (−6 on split). Closes the counterfactual seam (NOT the prefetch loop). +3 latent scaffold fixes. 550 contract lib green, clippy clean. The council REFUTED the prior "compose `Heel.plasticity` × MRU" ① resolution (`E-BASIN-NOT-EDGE-PLASTICITY`): coarse strength = MRU slot-order (shipped); per-edge Hebbian = per-plane `PlasticityState` (gated).
