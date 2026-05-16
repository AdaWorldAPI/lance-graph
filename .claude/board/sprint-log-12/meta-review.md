# Sprint-12 Meta-Review (Opus 4.7, PP-7, 2026-05-16)

> **Scope:** Closing meta-review for sprint-12 (Waves F + G; PRs #388, #389, #390). Reads: `.claude/board/sprint-log-11/meta-review-opus.md` (Wave F honest review), `.claude/board/sprint-log-11/meta-review-opus-wave-g.md` (Wave G honest review), `git log main --oneline -30`, `.claude/plans/cognitive-substrate-convergence-v2.md` §11.
> **Authority:** PP-7 (sprint-13 preflight planner, Opus 4.7). This is the canonical sprint-12 record consolidating Wave F (B grade) + Wave G (A− grade) into a single sprint-level signal and anchoring sprint-13 spawn gate.
> **Predecessor:** `.claude/board/sprint-log-11/meta-review.md` (sprint-11 closing meta-review, W-F10 + W-Meta-Opus, format reference for this file).

---

## 1. Executive Summary

### Sprint grade: **B+** (Wave F B + Wave G A− = net positive trajectory; integration discipline corrected mid-sprint)

**Headline:** Sprint-12 delivered the Phase B completion (D-CSV-5b cutover, D-CSV-6b HashMap surface), the Phase C dispatch fleet (D-CSV-10 SigmaTierRouter with Jirak-derived bands, D-CSV-13 batch i4 API scaffolding), and the Phase D scaffolding (D-CSV-11 stream types, D-CSV-12 splat ops) across two waves. Wave F (12 Sonnet workers + 1 Opus meta) shipped fast but botched the lib.rs/workspace registration discipline, requiring three P0 follow-up fixes in PR #389. Wave G (6 Sonnet workers + 1 Opus meta) demonstrated the discipline correction — six in-lane deliveries, three Wave F debt items actively repaired (workspace conflict, E-META-10 iron-rule promotion, D-CSV-5b cutover follow-through), zero new blocker-class regressions. The CCA2A 12+1 fleet pattern is now proven twice: Wave F as a cautionary scaling exercise, Wave G as the disciplined follow-through. The Opus meta-reviewer caught a math error in its own Wave G brief (Jirak p≥4 inverted), and the W-G4 Sonnet worker corrected it by consulting the workspace's own iron rule — exactly the consult-don't-guess discipline the workspace doctrine targets.

**Why B+ not A−:**

- Wave F's three CSI-7/8/9 blockers (sigma-tier-router workspace orphan, AttentionMask unregistered, ndarray streams unregistered) materially failed the "ship integration not just files" floor. PR #389 codex-fixup absorbed CSI-7 + CSI-8 + CSI-10; CSI-9 cross-repo ndarray PR remains open.
- CSI-13 (orphan-pattern recurrence) confirmed the worker-prompt template as written ("main thread aggregates") creates systematic invisible-work debt. Wave G's correction is structural proof the prompt template needs updating before sprint-13 spawn.
- The rustfmt 1.95 CI gate caught nearly every PR (formatting commits on #383, #384, #388, Wave G), confirming the auto-fmt step belongs in the worker template, not as post-hoc cleanup.

**Why not B:** Wave G is genuinely A−. The six workers stayed in lane, the meta-Opus reviewer caught its own math error and accepted the correction, the iron-rule promotion process worked (E-META-10 → I-LEGACY-API-FEATURE-GATED), and three CSIs were proactively surfaced (CSI-15 naming pre-commitment, CSI-17 spec-error-did-not-propagate, CSI-18 iron-rule meta-pattern). Sprint-12 ends in a substantially stronger place than it began: 4 iron rules in CLAUDE.md (was 3), 4-track Phase C completion (was 2), the cognitive-shader-driver workspace conflict resolved (TD-SHADER-DRIVER-WORKSPACE-CONFLICT-1 closed), and the i4-substrate-decisions knowledge doc anchored.

---

## 2. Per-Wave Summary

### Wave F — sprint-11 Phase A/B/C + sprint-12 Phase C/D scaffold (12 Sonnet + 1 Opus)

**Workers:** W-F1..W-F12 (12 Sonnet) + W-Meta-Opus (Opus honest review).
**Branch:** `claude/sprint-12-wave-f-fleet` (PR #388) + `claude/sprint-12-wave-f-codex-p2-followup` (PR #389).
**Deliverables:** D-CSV-10 SigmaTierRouter scaffold (W-F1), AttentionMask SoA + Actor (W-F2/W-F3), QualiaStream/InferenceStream/SplatFieldStream ndarray scaffolds (W-F4/W-F5/W-F6), Splat ops fleet on thinking-engine (W-F7), TYPE_DUPLICATION_MAP refresh (W-F8), TECH_DEBT+ISSUES sweep (W-F9), sprint-11 meta-review draft (W-F10), i4-substrate-decisions knowledge doc (W-F11), cognitive-substrate-convergence-v2 plan draft (W-F12, 608 LOC).
**LOC delta:** ~2,400 LOC code + ~1,200 LOC governance/plans = **~3,600 LOC** (largest single fleet to date).
**Test delta:** ~104 #[test] markers across implementation workers (W-F1..F7); ~60 distinct test functions per worker self-reports.
**Files touched:** 19 (5 cognitive-shader-driver, 3 sigma-tier-router crate, 3 ndarray cross-repo, 1 thinking-engine, 7 governance/plans).
**Wave F grade: B** (per W-Meta-Opus Wave F review; three P0 integration gaps CSI-7/8/9; aggregation pass was invisible work that nobody owned).

### Wave G — sprint-12 Phase B completion + Phase C dispatch + Phase E entry (6 Sonnet + 1 Opus)

**Workers:** W-G1..W-G6 (6 Sonnet) + W-Meta-Opus (Opus honest review).
**Branch:** `claude/sprint-12-wave-g-cutover-jirak` (PR #390 in flight).
**Deliverables:** D-CSV-5b QualiaColumn cutover (W-G1, end-to-end across bindspace/engine_bridge/driver), D-CSV-6b CAM-PQ HashMap WitnessCorpus surface (W-G2), D-CSV-13 batch i4 API on contract::mul (W-G3, 5 batch + 1 vec wrapper), D-CSV-15 Jirak-derived Σ-tier bands (W-G4, math correction caught), I-LEGACY-API-FEATURE-GATED iron rule promotion (W-G5, E-META-10 → CLAUDE.md), cognitive-shader-driver workspace fix (W-G6, 3-LOC).
**LOC delta:** ~800 LOC code + ~400 LOC governance = **~1,200 LOC** (focused follow-through).
**Test delta:** **+28 unit tests** across the six modified files (99 total #[test] markers across Wave G's six worker scopes).
**Files touched:** 9 (3 cognitive-shader-driver, 1 sigma-tier-router lib, 1 lance-graph-contract mul, 1 lance-graph witness_corpus + mod, 1 Cargo.toml workspace, 2 governance CLAUDE.md/EPIPHANIES/TECH_DEBT).
**Wave G grade: A−** (per W-Meta-Opus Wave G review; six in-lane deliveries, three Wave F debt items repaired, two soft findings CSI-15/CSI-18 deferred to sprint-13).

### Wave F + Wave G aggregate

**Sprint-12 aggregate LOC:** ~4,800 LOC (code + governance + plans).
**Sprint-12 aggregate tests:** ~132 #[test] markers (Wave F ~104 + Wave G 28 = 132). Estimated ~85-90 distinct test functions added in sprint-12.
**Sprint-12 PRs:** #388 (Wave F fleet, merged), #389 (Wave F codex P2 follow-up, merged), #390 (Wave G, in PR pending review).
**Sprint-12 CSIs surfaced:** CSI-7 through CSI-18 = **12 new CSIs** (continuing sprint-10/11 CSI-1..6 sequence).
**Sprint-12 iron rules promoted:** 1 (E-META-10 → I-LEGACY-API-FEATURE-GATED).

---

## 3. Per-PR Roll-up

### PR #388 — Wave F fleet (merged `77f2d26`)

**Branch:** `claude/sprint-12-wave-f-fleet`
**Scope:** 12-worker Sonnet fleet covering D-CSV-10 scaffolding, D-CSV-11/12 stream + splat scaffolding, TYPE_DUPLICATION_MAP refresh, TECH_DEBT/ISSUES sweep, sprint-11 meta-review draft, i4-substrate-decisions knowledge doc, cognitive-substrate-convergence-v2 plan.
**Codex catches (pre-merge):** P2 AttentionMaskBackend impl missing for AttentionMaskSoA (deferred to PR #389); rustfmt gate caught on witness_corpus.rs (fixed in `f6a1f9f` ahead of merge).
**Opus meta-review catches (post-merge, gov commit `d4e5bbc`):** CSI-7 (sigma-tier-router not in parent workspace), CSI-8 (AttentionMask + Actor not in lib.rs), CSI-9 (QualiaStream + SplatFieldStream not in ndarray mod.rs), CSI-10 (W-F2 redeclared MailboxId locally), CSI-11 (W-F12 plan drift on D-CSV-5a status), CSI-13 (systemic orphan-pattern from "main thread aggregates" prompt).
**Key finding:** the "write file → meta-review" pipeline omitted an aggregation phase; three P0-class registration gaps shipped to main and were caught only by Opus verification against the working tree. This is the headline lesson of sprint-12.
**Grade:** **B** (Wave F per W-Meta-Opus Wave F review; three blocker-class CSIs).

### PR #389 — Wave F codex P2 follow-up (merged `b526485`)

**Branch:** `claude/sprint-12-wave-f-codex-p2-followup`
**Scope:** AttentionMaskBackend impl for AttentionMaskSoA (codex P2 from PR #388) + canonical MailboxId import (resolves CSI-10).
**Codex catches:** none new (this PR IS the codex follow-up).
**Opus meta-review catches:** none (mechanical fix; verified by main-thread Opus before merge).
**Key finding:** the codex P2 catch worked as designed — the post-merge codex pass surfaced the missing trait impl and routed it through a focused follow-up PR with no scope creep. The MailboxId canonical import being bundled into the same PR closed CSI-10 in the same commit.
**Grade:** **A−** (mechanical fix, clean scope, no new debt).

### PR #390 — Wave G cutover + Jirak + iron rule (in PR, branch `claude/sprint-12-wave-g-cutover-jirak`)

**Branch:** `claude/sprint-12-wave-g-cutover-jirak` (4 commits on top of `b526485` + 1 gov commit `4d429e3` + 1 fmt commit `bad0875`)
**Scope:** D-CSV-5b QualiaColumn cutover (W-G1), D-CSV-6b WitnessCorpus HashMap surface (W-G2), D-CSV-13 batch i4 API (W-G3), D-CSV-15 Jirak-derived bands (W-G4), I-LEGACY-API-FEATURE-GATED iron rule (W-G5), cognitive-shader-driver workspace fix (W-G6), W-Meta-Opus honest review with CSI-15 fix in same commit (gov).
**Codex catches:** rustfmt 1.95 gate caught witness_corpus.rs + mod.rs (fixed in `bad0875` post-meta-review).
**Opus meta-review catches:** CSI-14 (CONFIRMED OK — QualiaColumn deprecation covers all live call sites), CSI-15 (P2, FIXED IN GOV COMMIT — CamPqWitnessIndex → WitnessIndexHashMap rename), CSI-16 (CONFIRMED OK — batch API length-assertion discipline complete), CSI-17 (LOW — Wave F→G Jirak spec error did NOT propagate beyond worker brief), CSI-18 (MED — four iron rules share meta-pattern; doctrine consolidation deferred to sprint-13).
**Key finding:** Wave G is the disciplinary correction Wave F earned. W-G4's Jirak math correction (caught the meta-reviewer's own inverted brief by consulting CLAUDE.md's I-NOISE-FLOOR-JIRAK iron rule) is the standout Sonnet-worker performance of the sprint.
**Grade:** **A−** (Wave G per W-Meta-Opus Wave G review; six in-lane deliveries, three Wave F debts repaired, two soft findings deferred).

---

## 4. CSI Continuation — CSI-1..18 with Resolution Status

This is the canonical CSI ledger across sprint-10 + sprint-11 + sprint-12. Future meta-reviews continue from CSI-19.

| CSI | Severity | Discoverer | Resolution status | Anchor |
|---|---|---|---|---|
| **CSI-1** | MED | sprint-11 Wave F (W-F8) | **OPEN** — TrustTexture × 2 rename queued for sprint-13 housekeeping | TD-TRUST-TEXTURE-DUPE-1 |
| **CSI-2** | P0→fixed | sprint-11 Wave A codex | **CLOSED** — fixed in `42b3215` + `b44ce87`; promoted to E-META-10 → I-LEGACY-API-FEATURE-GATED iron rule | I-LEGACY-API-FEATURE-GATED |
| **CSI-3** | MED | PR #381 + sprint-11 | **MITIGATED** — Python-via-Bash heredoc pattern is the operational workaround; SDK gap remains open upstream (anthropics/claude-code#46861) | E-META-8 |
| **CSI-4** | LOW-MED | sprint-11 Wave C + E | **ACCEPTED** — intentional decoupling (ndarray producer vs contract consumer); documented in TYPE_DUPLICATION_MAP | TYPE_DUPLICATION_MAP §SplatField |
| **CSI-5** | LOW | sprint-11 Wave B | **CLOSED** — OQ-CSV-1 ratified to Option α; process improvement noted | OQ-CSV-1 |
| **CSI-6** | LOW | sprint-11 orchestration | **CLOSED** — board hygiene as main-thread-only responsibility codified in worker template | E-META-9, CLAUDE.md Mandatory Board-Hygiene Rule |
| **CSI-7** | P0 | sprint-12 Wave F W-Meta-Opus | **CLOSED in PR #389** — sigma-tier-router moved to parent workspace members; verified via `cargo metadata --no-deps` (16 packages including sigma-tier-router) | PR #389, Wave G CSI-7-follow-through note |
| **CSI-8** | P0 | sprint-12 Wave F W-Meta-Opus | **CLOSED in PR #389** — AttentionMask + AttentionMaskActor registered in cognitive-shader-driver/src/lib.rs | PR #389 |
| **CSI-9** | P0 (cross-repo) | sprint-12 Wave F W-Meta-Opus | **OPEN** — cross-repo ndarray PR required to register qualia + splat_field in hpc/stream/mod.rs; **HARD BLOCKER on D-CSV-11 productization** | ISSUES.md cross-repo blocker, ndarray PR needed |
| **CSI-10** | MED | sprint-12 Wave F W-Meta-Opus | **CLOSED in PR #389** — local MailboxId redeclaration in attention_mask.rs replaced with canonical contract import | PR #389 |
| **CSI-11** | MED | sprint-12 Wave F W-Meta-Opus | **CLOSED** — v2 plan §0.1/§11/§15 updated for D-CSV-5a Shipped status with commit anchor `6f58418` | cognitive-substrate-convergence-v2.md |
| **CSI-12** | OK | sprint-12 Wave F W-Meta-Opus | **CONFIRMED OK** — SplatField bit-compat between W-F6 + W-F7 verified identical layout | TYPE_DUPLICATION_MAP §SplatField |
| **CSI-13** | LOW (systemic) | sprint-12 Wave F W-Meta-Opus | **MITIGATED** — Wave G demonstrated that worker prompts including registration discipline ship clean integration commits; sprint-13 worker-template-v2 (PP-8) codifies this | PP-8 deliverable, Wave G as proof |
| **CSI-14** | OK | sprint-12 Wave G W-Meta-Opus | **CONFIRMED OK** — QualiaColumn deprecation covers all live call sites in cognitive-shader-driver | cognitive-shader-driver/src/bindspace.rs |
| **CSI-15** | P2 | sprint-12 Wave G W-Meta-Opus | **CLOSED in gov commit `4d429e3`** — renamed CamPqWitnessIndex → WitnessIndexHashMap (12 occurrences + 1 re-export) | witness_corpus.rs |
| **CSI-16** | OK | sprint-12 Wave G W-Meta-Opus | **CONFIRMED OK** — batch API length-assertion discipline complete (5 _batch + 1 _vec with appropriate asserts) | crates/lance-graph-contract/src/mul.rs |
| **CSI-17** | LOW | sprint-12 Wave G W-Meta-Opus | **POSITIVE FINDING** — Jirak spec error did NOT propagate beyond worker brief; W-G4 caught + corrected by consulting CLAUDE.md iron rule | sprint-12 Wave G review §3 CSI-17 |
| **CSI-18** | MED | sprint-12 Wave G W-Meta-Opus | **OPEN** — iron-rules-doctrine consolidation (~250 LOC) deferred to sprint-13 (PP-2 deliverable) | PP-2 sprint-13 deliverable |

**Net:** CSI-1, CSI-9, CSI-18 remain OPEN (sprint-13 work). All other CSIs are CLOSED, MITIGATED, ACCEPTED, or CONFIRMED OK. CSI-9 is the only hard cross-repo blocker.

---

## 5. Cross-Cutting Epiphanies — E-META-7..E-META-10

| Epiphany | Status | Sprint-12 evolution | Iron-rule promotion |
|---|---|---|---|
| **E-META-7** — Dual CausalEdge64 types in workspace | FINDING (sprint-11) | D-CSV-9 transcoder shipped Option R-3 resolution at thinking-engine L3 boundary | No (architectural decoupling, not a discipline rule) |
| **E-META-8** — Bare Edit/Write perm rule invalid + subagent isolation gap | FINDING (sprint-11) | Python-via-Bash heredoc pattern is operational standard; SDK gap remains upstream | No (SDK-level, not codebase doctrine) |
| **E-META-9** — Mandatory Board-Hygiene Rule (retroactive-hygiene anti-pattern) | FINDING (sprint-11) | Maintained across sprint-12; Wave A gov commit pattern (`fd61310`) is the template | Already in CLAUDE.md as "Mandatory Board-Hygiene Rule" (effectively iron-rule status) |
| **E-META-10** — v1-API-under-v2-feature alias pattern | FINDING (sprint-11) → **PROMOTED TO IRON RULE** (Wave G W-G5, sprint-12) | I-LEGACY-API-FEATURE-GATED added to CLAUDE.md; CSI-2 + 4 codex P1 catches anchored as the empirical basis | **YES — I-LEGACY-API-FEATURE-GATED, 2026-05-16** |

**Iron rules in CLAUDE.md (as of sprint-12 end):**

1. **I-SUBSTRATE-MARKOV** (2026-04-20) — VSA bundling in d=10000 guarantees Chapman-Kolmogorov by construction.
2. **I-NOISE-FLOOR-JIRAK** (2026-04-20) — Bits are weakly dependent; use Jirak 2016 rates not classical Berry-Esseen.
3. **I-VSA-IDENTITIES** (2026-04-21) — VSA operates on identity fingerprints that POINT TO content; never on bitpacked content.
4. **I-LEGACY-API-FEATURE-GATED** (2026-05-16, sprint-12 Wave G) — v1 API paths under v2-layout features must route through canonical mapping or feature-gate to no-op with migration pointer.

**Emerging meta-pattern (CSI-18):** every iron rule formalizes a discipline against silent drift across some axis (substrate operator / statistical model / data semantics / API version). Sprint-13 PP-2 deliverable is the `.claude/knowledge/iron-rules-doctrine.md` consolidation.

---

## 6. What Sprint-12 Did Well

1. **CCA2A 12+1 fleet pattern proven twice at scale.** Wave F (12 Sonnet + 1 Opus) shipped 19 files in a single coordinated wave; Wave G (6 Sonnet + 1 Opus) demonstrated the same pattern with disciplined follow-through. The Opus meta-reviewer in both waves added value the Sonnet drafts could not (Wave F: CSI-7/8/9 verification against working tree; Wave G: Jirak math validation + iron-rule promotion ratification). The 12+1 pattern is now the proven default for substrate-convergence work.

2. **Opus catching math errors in main-thread specs is a working safety net.** W-Meta-Opus's Wave G brief contained an inverted Jirak p≥4 statement; W-G4 Sonnet worker caught the error by consulting CLAUDE.md's I-NOISE-FLOOR-JIRAK iron rule, derived the correct math (Σ_k = k^(p/2) / 10^(p/2) with Σ10 = 1.0 normalized), and shipped 8 new tests with the corrected assertion. The meta-Opus reviewer then independently verified W-G4's correction in the Wave G honest review (CSI-17 POSITIVE FINDING). This is the consult-don't-guess discipline working in both directions — Sonnet workers consulting iron rules, Opus meta-reviewers consulting Sonnet output against working tree.

3. **Iron-rule promotion process worked end-to-end.** E-META-10 was surfaced as FINDING in sprint-11 Wave A (codex catches on PR #383), recommended for promotion in Wave F W-Meta-Opus §4, drafted as new iron rule in Wave G W-G5, integrated into CLAUDE.md alongside the three existing rules, and immediately tested by being added to the worker-template enforcement checklist. The full FINDING → iron-rule pipeline took two waves and produced a CLAUDE.md anchor that is permanent doctrine.

---

## 7. What Sprint-12 Did Poorly

1. **CSI-13 orphan-pattern recurrence required Wave G to clean up Wave F.** The Wave F worker prompt template said "the main thread aggregates" for lib.rs/mod.rs/workspace registration. The main thread did NOT aggregate before meta-review, producing three P0-class CSIs (CSI-7/8/9). Wave G's six workers were instructed to include their registration in scope and shipped clean integration on every deliverable. The lesson: "main thread aggregates" is a permission-isolation pattern (subagents can't easily edit lib.rs without conflict-risk against parallel siblings) but it is NOT a delivery pattern unless the aggregation is an explicit scheduled deliverable. Sprint-13 worker-template-v2 (PP-8) must bake registration into worker scope.

2. **rustfmt 1.95 gate caught nearly every PR (formatting commits on #383, #384, #388, #390).** Every wave needed a post-meta-review `cargo fmt` commit to pass CI. This is mechanical overhead that should be in the worker template's pre-commit step (`cargo fmt --check` before any commit). Estimated overhead: 1 extra commit per wave × 5 waves = 5 wasted commits across sprint-11+12. Sprint-13 worker-template-v2 should add `cargo fmt && cargo clippy --no-deps` to the pre-commit checklist.

3. **`cargo target/` disk pressure required `cargo clean` once during sprint-12.** The Wave G gov commit notes a pre-existing infra issue: local `cargo test -p lance-graph` hit ENOSPC during link. Workspace target/ grew large enough to exhaust disk space during a routine compile cycle. This is operational debt: the workspace should have a `cargo clean` step in the BOOT.md session entry checklist, or a periodic CI prune, before sprint-13 onward. Estimated disk pressure: ~30-50 GB of target/ artifacts at sprint-12 mid-point.

---

## 8. Sprint-13 Spawn Gate

**Recommendation: spawn sprint-13 ONLY AFTER the following pre-fleet hygiene completes.**

### Pre-spawn checklist (ordered, blockers first)

| # | Item | Owner | Severity | Notes |
|---|---|---|---|---|
| 1 | PR #390 (Wave G) merges to main | Main thread post-review | **Blocker** | Carries D-CSV-5b/6b/13/15 + I-LEGACY-API-FEATURE-GATED iron rule + CSI-15 fix |
| 2 | CSI-9 cross-repo ndarray PR registers qualia + splat_field in hpc/stream/mod.rs | Cross-repo coordination | **Blocker on D-CSV-11 productization** | ~4 LOC in ndarray |
| 3 | CSI-1 TrustTexture × 2 rename (causal_edge::layout::TrustTexture → EdgeTexture or CrystallineState) | Sprint-13 housekeeping wave | P1 | ~1-2h refactor; TD-TRUST-TEXTURE-DUPE-1 |
| 4 | CSI-18 iron-rules-doctrine consolidation knowledge doc | PP-2 (sprint-13 preflight) | MED | ~250 LOC synthesis of 4 iron rules + meta-pattern |
| 5 | Worker-template-v2 with baked-in registration + cargo fmt pre-commit | PP-8 (sprint-13 preflight) | MED (process) | Resolves CSI-13 systemic + rustfmt-overhead-per-wave |
| 6 | `cargo clean` + disk-pressure check | Session-start hook | LOW (operational) | Add to BOOT.md or session-start hook |

### Ratifications needed (OQ-CSV-* enumerated by PP-11)

PP-11 will enumerate the full open-question table for sprint-13 spawn. Standing user ratifications from sprint-11/12 that remain valid:

- **OQ-CSV-1** (qualia vocab choice): RESOLVED to Option α (canonical convergence-observable vocab; drop dim 16 "integration").
- **OQ-CSV-2** (W-slot 6 bits): RESOLVED in sprint-11 Wave A.
- **OQ-CSV-4** (D-CSV-5 phasing): RESOLVED — sibling-then-cutover (sprint-11 5a + sprint-12 5b).
- **OQ-CSV-6** (Jirak threshold): RESOLVED via W-G4 Jirak-derived bands (math anchored in I-NOISE-FLOOR-JIRAK; principled VAMPE coupled-revival deferred to sprint-13+).

### Forward-looking sprint-13 deliverables

| Deliverable | PP author | Notes |
|---|---|---|
| **D-CSV-13b** — AVX-512/NEON intrinsics backing the batch i4 API | PP-3 spec | Wave G shipped contract; sprint-13 ships SIMD intrinsics |
| **D-CSV-14** — Splat ops on-Think method migration (free-function → method-on-carrier per CLAUDE.md litmus) | PP-4 spec | Builds on Wave F W-F7 thinking-engine splat_ops; requires Think carrier final form |
| **D-CSV-16** — VAMPE coupled-revival track activation for principled Jirak thresholds | PP-5 spec | Pairs with D-CSV-15 Jirak math; replaces hand-tuned σ-thresholds with Jirak-derived bounds |
| **D-CSV-17** — Real ndarray::hpc::cam_pq witness-tuple wiring (replaces WitnessIndexHashMap fallback) | PP-6 spec | Cross-repo dependency on ndarray hpc::cam_pq SPO witness-tuple support |
| **PP-2 iron-rules-doctrine** — `.claude/knowledge/iron-rules-doctrine.md` consolidation | PP-2 | Anchor for sprint-13 doctrinal workers |
| **PP-8 worker-template-v2** — bake lib.rs/workspace registration + cargo fmt into worker scope | PP-8 | Resolves CSI-13 systemic; closes "main thread aggregates" debt |

---

## 9. Per-Worker Grades — Sprint-12 Flat Table

### Wave F workers (W-F1..W-F12 + W-Meta-Opus-WaveF)

| Worker | Scope | Grade | Justification |
|---|---|---|---|
| **W-F1** | sigma-tier-router crate (D-CSV-10 scaffold + Σ-tier banding) | **B** | Standalone subworkspace (CSI-7 blocker); code quality fine, integration broken |
| **W-F2** | AttentionMask SoA core | **B−** | Unregistered (CSI-8); local MailboxId redeclaration (CSI-10); two gaps in one worker |
| **W-F3** | AttentionMaskActor | **B+** | Unregistered (CSI-8); but correct MailboxId import from contract |
| **W-F4** | QualiaStream (ndarray, D-CSV-11 scaffold) | **B** | Cross-repo orphan (CSI-9) — most serious because fix can't ride same commit |
| **W-F5** | InferenceStream (ndarray) | **A−** | Registered in mod.rs; bit-compat documented; only Wave F worker who completed integration |
| **W-F6** | SplatFieldStream (ndarray) | **B** | Same CSI-9 orphan; bit-compat with W-F7 verified (CSI-12 OK) |
| **W-F7** | Splat ops fleet (thinking-engine D-CSV-12 scaffold) | **A** | Disciplined local SplatField def with dep-cycle commentary; +2 tests over spec |
| **W-F8** | TYPE_DUPLICATION_MAP refresh | **A−** | 5 new Wave F entries with file:line rigor; MailboxId shadow recorded |
| **W-F9** | TECH_DEBT + ISSUES sweep | **B+** | 8 TD + 5 IS entries; did not catch CSI-7/8/9 (structural gap, not worker failure) |
| **W-F10** | sprint-11 meta-review draft (341 LOC) | **B** | Format mirrors sprint-10; CSI-1..6 real findings; missed CSI-7/8/9 (structural) |
| **W-F11** | i4-substrate-decisions knowledge doc (200 LOC) | **B+** | Anchored READ-BY header; OQ-CSV-1..6 ratification chain captured |
| **W-F12** | cognitive-substrate-convergence-v2 plan draft (608 LOC) | **B−** | Comprehensive but CSI-11 drift (D-CSV-5a "In PR" vs merged) |
| **W-Meta-Opus-WaveF** | sprint-11 + Wave F honest review (161 LOC) | **A** | Surfaced CSI-7/8/9 against working tree; recommended E-META-10 promotion; clean +13 LOC fix list |

### Wave G workers (W-G1..W-G6 + W-Meta-Opus-WaveG)

| Worker | Scope | Grade | Justification |
|---|---|---|---|
| **W-G1** | D-CSV-5b QualiaColumn cutover (bindspace + engine_bridge + driver) | **A−** | 86/86 tests pass; 4 pre-existing tests updated with discipline; deprecation covers all live call sites (CSI-14 OK) |
| **W-G2** | D-CSV-6b CAM-PQ HashMap WitnessCorpus surface | **B+** | 15 tests + module registration; CSI-15 naming pre-commitment (fixed in gov commit) |
| **W-G3** | D-CSV-13 batch i4 API (5 _batch + 1 _vec) | **A−** | 20 tests; clippy clean; all 5 panic-on-length-mismatch verified (CSI-16 OK) |
| **W-G4** | D-CSV-15 Jirak-derived Σ-tier bands | **A** | Caught + corrected Opus's math error; 8 new tests; Σ10 = 1.0 normalized; Jirak 2016 citation |
| **W-G5** | I-LEGACY-API-FEATURE-GATED iron-rule promotion | **A** | CLAUDE.md now has 4 iron rules; EPIPHANIES E-META-10 PROMOTED header; TECH_DEBT TD-RESOLVED-1 entry |
| **W-G6** | TD-SHADER-DRIVER-WORKSPACE-CONFLICT-1 (3-LOC Cargo.toml fix) | **A−** | Clean fix; sister CSI-7 fix bundled into PR #389 |
| **W-Meta-Opus-WaveG** | sprint-12 Wave G honest review (180 LOC) | **A** | Caught CSI-15 + CSI-17 + CSI-18; ratified W-G4 math correction independently; CSI-15 fix in same commit |

**Wave F average:** B (per W-Meta-Opus Wave F).
**Wave G average:** A− (per W-Meta-Opus Wave G; 5 of 7 grades at A−/A).
**Sprint-12 average:** **B+** (Wave F + Wave G aggregate; net positive trajectory from Wave F integration debt repair).

---

## 10. Forward-Looking Deliverables for Sprint-13

### Spec authoring track (PP-3 through PP-6)

| Spec | Deliverable | Estimated LOC | Dependencies |
|---|---|---|---|
| **PP-3** | D-CSV-13b spec — AVX-512/NEON i4 multiply-accumulate intrinsics backing the batch i4 API | ~150 LOC spec | Wave G W-G3 contract; depends on rustc nightly target_feature support |
| **PP-4** | D-CSV-14 spec — Splat ops on-Think method migration | ~200 LOC spec | Wave F W-F7 thinking-engine splat_ops; Think carrier final form |
| **PP-5** | D-CSV-16 spec — VAMPE coupled-revival track for principled Jirak thresholds | ~250 LOC spec | Wave G W-G4 Jirak math; I-NOISE-FLOOR-JIRAK iron rule |
| **PP-6** | D-CSV-17 spec — Real ndarray::hpc::cam_pq witness-tuple wiring | ~180 LOC spec | Cross-repo ndarray dependency; depends on upstream SPO witness-tuple support |

### Doctrinal track (PP-2)

| Deliverable | Description | Estimated LOC |
|---|---|---|
| **PP-2** iron-rules-doctrine | `.claude/knowledge/iron-rules-doctrine.md` consolidating I-SUBSTRATE-MARKOV + I-NOISE-FLOOR-JIRAK + I-VSA-IDENTITIES + I-LEGACY-API-FEATURE-GATED + meta-pattern ("no silent drift across axis X") | ~250 LOC |

### Process track (PP-8)

| Deliverable | Description |
|---|---|
| **PP-8** worker-template-v2 | Bake lib.rs/workspace registration into worker scope; add `cargo fmt --check && cargo clippy --no-deps` to pre-commit checklist; mandate Python-via-Bash heredoc for file writes; explicit "do not update board governance files" instruction (CSI-6/E-META-9 enforcement) |

### Housekeeping track (sprint-13 wave or sprint-13-prep wave)

- CSI-1 TrustTexture × 2 rename (~1-2h refactor)
- CSI-9 cross-repo ndarray PR (qualia + splat_field registration, ~4 LOC)
- `cargo clean` in BOOT.md or session-start hook

---

## 11. Cumulative Test Counts (Sprint-10 + 11 + 12)

| Sprint | Distinct test functions added | Cumulative running total | Target (Miri) |
|---|---|---|---|
| **Sprint-10** | ~80 tests (CSV prep waves) | ~1,250 (baseline) | 1,550 |
| **Sprint-11** | ~58 tests (Wave A–E per W-F12 §14 rollup) | ~1,308 | 1,550 |
| **Sprint-12** | ~85-90 tests (Wave F ~60 + Wave G 28) | **~1,395** | 1,550 |
| **Gap to Miri target** | — | **~155 tests short** | — |

**Sprint-13 test target:** add ~80 tests across PP-3/4/5/6 specs (D-CSV-13b SIMD intrinsics ~20, D-CSV-14 method migration ~25, D-CSV-16 VAMPE probe ~15, D-CSV-17 CAM-PQ wiring ~20). Projected end-of-sprint-13 cumulative: ~1,475 tests (90% of Miri target).

**Net:** sprint-12 was the strongest test-delta sprint to date (~85-90 added vs sprint-11's ~58 vs sprint-10's ~80), reflecting the wider implementation scope (D-CSV-5b cutover + D-CSV-13 batch API + D-CSV-15 Jirak bands all carried 15-25 tests each).

---

## 12. Cumulative LOC (Code vs Governance vs Docs)

| Sprint | Code LOC | Governance LOC | Docs/Plans LOC | Total |
|---|---|---|---|---|
| **Sprint-10** | ~2,800 | ~900 | ~1,400 | ~5,100 |
| **Sprint-11** | ~2,200 | ~600 | ~800 | ~3,600 |
| **Sprint-12 Wave F** | ~2,400 | ~400 | ~800 | ~3,600 |
| **Sprint-12 Wave G** | ~800 | ~250 | ~150 | ~1,200 |
| **Sprint-12 total** | **~3,200** | **~650** | **~950** | **~4,800** |
| **Cumulative (10+11+12)** | **~8,200** | **~2,150** | **~3,150** | **~13,500** |

**Code-to-governance ratio:** sprint-12 ran 3,200 / 650 = ~4.9× code-to-governance LOC, in line with sprint-10's 3.1× and sprint-11's 3.7×. Governance LOC is dominated by W-Meta-Opus honest reviews (Wave F 161 LOC + Wave G 180 LOC = 341 LOC, plus the per-wave AGENT_LOG/EPIPHANIES/STATUS_BOARD updates).

**Code-to-docs ratio:** sprint-12 ran 3,200 / 950 = ~3.4× code-to-docs, with W-F12 cognitive-substrate-convergence-v2 plan (608 LOC) and W-F11 i4-substrate-decisions knowledge doc (200 LOC) accounting for ~85% of sprint-12 docs.

---

## Closing Assessment

Sprint-12 graduated the CCA2A 12+1 fleet pattern from "single proof-of-scale" (sprint-11 Wave F as it was originally conceived) to "proven discipline with documented failure mode and correction." Wave F demonstrated the headline failure (CSI-13 systemic orphan-pattern from "main thread aggregates" prompts); Wave G demonstrated the discipline correction (registration baked into worker scope). The Opus meta-reviewer caught a math error in its own brief during Wave G; the Sonnet worker corrected it by consulting the workspace's iron rule. This is the consult-don't-guess discipline working in both directions, and it is the proof sprint-13 needed before scaling the pattern to a fifth wave.

The four iron rules in CLAUDE.md (3 from sprint-11 + 1 promoted in sprint-12 Wave G) now span four discipline axes (substrate operator, statistical model, data semantics, API version). CSI-18's meta-pattern recognition ("no silent drift across axis X") is the doctrinal capstone — sprint-13 PP-2 turns this into a knowledge-doc anchor for future iron-rule authors.

Sprint-12 closes with three OPEN CSIs (CSI-1 TrustTexture rename, CSI-9 cross-repo ndarray, CSI-18 iron-rules-doctrine), one HARD BLOCKER on D-CSV-11 productization (CSI-9), and a Wave G PR (#390) pending review-and-merge. Sprint-13 spawn is recommended after the six-item pre-fleet hygiene checklist completes (§8 above), with worker-template-v2 (PP-8) as the structural fix that turns CSI-13 from "recurring failure mode" into "process-prevented regression."

**Sprint-12 final grade: B+** (Wave F B + Wave G A− = net positive trajectory; integration discipline corrected mid-sprint; iron-rule promotion process worked end-to-end).

**Sprint-13 spawn recommendation: YES, gated on the six-item pre-fleet hygiene checklist + worker-template-v2 (PP-8) landing as the structural fix for CSI-13.**

---

*End of sprint-12 meta-review. PP-7 (Opus 4.7), sprint-13 preflight planning, 2026-05-16. Authored after reading sprint-log-11/meta-review-opus.md (Wave F honest review, 161 LOC), sprint-log-11/meta-review-opus-wave-g.md (Wave G honest review, 180 LOC), git log main --oneline -30, and cognitive-substrate-convergence-v2.md §11 D-CSV-* deliverable table. Grades aggregate the per-wave Opus honest reviews; sprint-13 spawn checklist consolidates Wave F + Wave G open items for PP-2/3/4/5/6/8 spec authors.*
