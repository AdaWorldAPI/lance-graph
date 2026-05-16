# Sprint-13 Preflight Meta-Review — OPUS Honest Cross-Cutting Review (Opus 4.7, W-Meta-Opus, 2026-05-16)

> **Scope:** Independent Opus review of the sprint-13 preflight planning fleet (Wave H, 13 Opus planners PP-1..PP-13). Sibling to the sprint-log-11/meta-review-opus (Wave F) and sprint-log-12/meta-review (Wave G + sprint-12 close) honest reviews. This file is the brutally-honest cross-cutting check: spec actionability, plan-vs-git-state drift, cross-spec consistency, ID-coordination gaps, and sprint-13 spawn-readiness gates.
>
> **Authority:** W-Meta-Opus (main-thread spawn, Opus 4.7 per Model Policy "accumulation requires Opus"). I verified each PP output against the working tree at HEAD = `04620aa` and against `main` at HEAD = `b526485`, not against planner self-reports.
>
> **Predecessors:**
> - `.claude/board/sprint-log-11/meta-review-opus.md` (Wave F honest review, 161 LOC, grade B)
> - `.claude/board/sprint-log-12/meta-review.md` (PP-7 sprint-12 close, grade B+)

---

## 1. Executive Summary

### Wave H grade: **B+** (revised down from a hypothetical A− on output count alone; capped by plan-vs-git drift in PP-1/PP-12 and ID/OQ-numbering drift across PP-1..PP-6)

**Headline:** Wave H delivered 13 planner outputs across 8 commits — ~8000 LOC of planning documentation, zero source-code touches — at a quality level that is genuinely better than the sprint-12 plan-author work (W-F12 had CSI-11 drift on one cell; Wave H has comparable drift on multiple cells, but ALSO multiple internally-consistent specs that are engineer-ready). The four impl specs (PP-3 rayon, PP-4 Think methods, PP-5 CAM-PQ, PP-6 SIMD) are ACTIONABLE for sprint-13 spawn — each could be handed to a Sonnet impl worker today with minimal additional context. The two doctrinal/process outputs (PP-2 iron-rules-doctrine, PP-8 worker-template-v2) genuinely close CSI-13 and CSI-18 root causes. The new agent card (PP-13 brutally-honest-tester) adds a missing pre-commit gate to the CCA2A pipeline.

**Why B+ not A−:**

- **PP-1 plan v3 misreports CSI-9 status as "Risk REDUCED" via d4e5bbc aggregation commit** — but d4e5bbc was a lance-graph commit; `/home/user/ndarray/src/hpc/stream/mod.rs` registration only exists on a local branch (`claude/sprint-12-qualia-stream-w-f4`), NOT on ndarray master. PP-7 sprint-12 meta-review correctly flags CSI-9 as OPEN/HARD BLOCKER. PP-1 v3 §0.1 and §13.8 drift from this reality. This is the same class of drift as CSI-11 in v2 — exactly what the v3 plan was supposed to prevent.
- **PP-12 cross-repo audit propagates the same false claim** — "ndarray PR #147 merged 2026-05-16T04:35:05Z" — verified against `cd /home/user/ndarray && git log master --oneline`: master HEAD is `2a3885d2` (PR #146 merged), there is no PR #147 merge in master log. The streams scaffold commit `2a1a1e38` is on local branch only. Two planners reading the same wrong state is worse than one.
- **D-CSV-* and OQ-CSV-* numbering drift across PP-1..PP-6 and PP-11.** PP-3 says "D-CSV-16 reserved for PP-2 splat on-Think method migration" — PP-2 is iron-rules-doctrine, PP-4 is on-Think, D-CSV-16 is CAM-PQ in PP-5. PP-1 lists OQ-CSV-7..12; PP-11 lists OQ-CSV-7..19. Multiple planners chose IDs independently without a coordinator. New CSI-19 below.
- **PP-2 reads CLAUDE.md as if I-LEGACY-API-FEATURE-GATED is already an iron rule** (CLAUDE.md §6.4 "I-LEGACY-API-FEATURE-GATED — API version iron rule (pending sprint-13 ratification)") — verified against `grep "I-LEGACY-API-FEATURE-GATED" /home/user/lance-graph/CLAUDE.md`: no match. The promotion is on PR #390 branch, not yet on main. PP-2 footnotes the pending status correctly but the §1 axis table presents it as a peer of the other three. Multiple downstream specs (PP-6, PP-13) cite I-LEGACY-API-FEATURE-GATED as if canonical.

**Why not B:** four of the impl specs (PP-3, PP-4, PP-5, PP-6) are genuinely engineer-ready. PP-4 in particular passed an honest grep test for callers (zero external callers of the 4 free fns verified). PP-5's Option-A-vs-Option-B analysis cites I-VSA-IDENTITIES correctly. PP-8 worker-template-v2 closes CSI-13 with explicit §5.1-§5.5 integration responsibility. PP-13 introduces a stable-Rust-only post-impl pre-commit gate that the CCA2A pipeline genuinely lacked. The B+ is about the same-class-of-drift problem the v2 plan had, not about output quality at the spec level.

---

## 2. Per-Planner Grades

| Planner | Grade | Key finding |
|---|---|---|
| **PP-1 plan v3** (cognitive-substrate-convergence-v3.md, ~597 LOC) | **B−** | CSI-9 status drift (§0.1 + §13.8 claim CSI-9 resolved via d4e5bbc; ndarray master has NO stream registration). D-CSV-15b vs D-CSV-15 ambiguity in §11. OQ-CSV-7..12 vs PP-11's OQ-CSV-7..19 — undercount. Otherwise comprehensive: status delta §0.1/§0.2 is the right shape; sprint-13 D-CSV-13b/14/16/17 entries are internally consistent. Same drift class as W-F12 CSI-11 — exactly what v3 was meant to prevent. |
| **PP-2 iron-rules-doctrine** (.claude/knowledge/iron-rules-doctrine.md, ~392 LOC) | **A−** | Four-axis framing (substrate operator / statistical model / data semantics / API version) is genuinely orthogonal — I considered whether a fifth axis could be needed and the candidate list §5.5 (ABI under codebook rebase, memory ordering, workspace member/exclude, board-hygiene) shows the doctrine author tracked the failure modes that DON'T fit. Per-rule analysis (§2.1-§2.4) is rigorous. Caveat: PP-2 reads from main (per its own header note) but §6 lists I-LEGACY-API-FEATURE-GATED in "Canonical iron rules" — should say "pending PR #390 merge." Promotion checklist §3 is the artifact this workspace will benefit from for years. |
| **PP-3 rayon streams** (pr-sprint-13-rayon-streams.md, ~672 LOC) | **B** | D-CSV-17 ID is correct. Spec is engineer-ready: IndexedParallelIterator vs ParallelIterator §2 analysis is load-bearing. 18-test matrix in §7 is structured. Determinism contract §6 correctly cites I-SUBSTRATE-MARKOV for VSA-bundle associativity. **The drift:** §0 "follow the D-CSV-16 slot reserved by PP-2 (sprint-13 splat on-Think method migration)" — PP-2 is iron-rules-doctrine, NOT splat on-Think; PP-4 is splat on-Think; D-CSV-16 is CAM-PQ per PP-5. Three errors in one sentence. Implementation correctness is independent of this; ID coordination is what failed. |
| **PP-4 Think methods** (pr-sprint-13-think-methods.md, ~674 LOC) | **A−** | D-CSV-14 ID correct. **Honest verification PASSED**: `grep splat_gaussian crates/ examples/` returns ONLY `crates/thinking-engine/src/splat_ops.rs` — PP-4's "zero callers of the 4 free fns outside splat_ops.rs" claim is TRUE. `grep "struct Think\b" crates/` returns empty — PP-4's "Today thinking-engine has ThinkingEngine ... it is NOT the doctrinal `Think`" claim is TRUE. §2.1 honest framing ("This is NOT yet the eight-field doctrinal Think") protects against premature commitment. 24 tests (16 migrated + 4 deprecation + 4 cycle integration). §5.2 deprecation shim pattern is the right belt-and-suspenders. |
| **PP-5 WitnessCorpus CAM-PQ** (pr-sprint-13-witness-cam-pq.md, ~551 LOC) | **A−** | D-CSV-16 ID correct. Option A (VSA bind of role-keyed identities) vs Option B (one-hot) §2.2/§2.3 — cites I-VSA-IDENTITIES Test-0 (register-laziness) and Layer-3 (register-loss) correctly; the Option-B rejection rationale is the doctrine-correct read of the iron rule. **PP-5 cites the CamPqIndexPlaceholder → WitnessIndexHashMap rename (CSI-15 from PR #390) explicitly in §0 line** "formerly `CamPqIndexPlaceholder` until CSI-15 rename in PR #390" — that is the honest provenance discipline this workspace needs. Verified against main: CamPqIndexPlaceholder is still in `witness_corpus.rs:83`; the rename is on PR #390 branch only. 12-test matrix maps cleanly. |
| **PP-6 SIMD i4** (pr-sprint-13-simd-i4.md, ~982 LOC) | **A−** | D-CSV-13b ID correct. Follows the ndarray `simd_caps()` proven pattern (§2 cites `/home/user/ndarray/src/hpc/simd_caps.rs` and the proven intrinsic patterns in `simd_avx512.rs:1612` / `simd_avx512.rs:641`). **The `GateDecision::{Block, Hold}` payload problem is correctly carved out** in §2: SIMD computes discriminants only (u8 0/1/2), scalar tail materializes String reasons — this is the right architectural call. 10 per-fn pseudocode sketches (5 fns × 2 ISAs). Risk matrix §8 is honest (R-4 "AVX-2 falls through to scalar"). The B+ to A− coin-flip: the 982 LOC is at the upper edge of "spec" — could be split into a per-ISA spec pair without loss. |
| **PP-7 sprint-12 meta-review** (sprint-log-12/meta-review.md, ~310 LOC) | **A−** | Format mirrors sprint-log-11/meta-review-opus.md correctly. Per-wave summary (§2) + per-PR rollup (§3) + CSI-1..18 ledger (§4) + per-worker grade table (§9). **Grades both waves correctly** per the prior honest reviews — Wave F B (matches Wave-F W-Meta-Opus), Wave G A− (matches Wave-G W-Meta-Opus). Spawns sprint-13 gate (§8) with 6 pre-spawn checklist items. **Catches all 18 CSIs in §4 with current resolution status** — that's the audit ledger every future meta-review continues from. Note: PP-7 §3 PR #390 "**Codex catches:** rustfmt 1.95 gate caught witness_corpus.rs + mod.rs (fixed in `bad0875` post-meta-review)" — this is also captured in PR_ARC, consistent. |
| **PP-8 worker-template-v2** (worker-template-v2.md, ~563 LOC) | **A** | §5 integration responsibility (§5.1 module registration, §5.2 Cargo.toml dep additions, §5.3 re-exports, §5.4 workspace [members] = main-thread reserved, §5.5 field-isolation matrix tests) — this is the CSI-13 structural fix. **Concrete enough that a Sonnet worker can follow §5 WITHOUT main-thread aggregation** — that was the user's explicit acceptance criterion. §7 output format codifies the AGENT_LOG entry shape. §10 worker classes (impl / governance / spec / knowledge / meta-review / verification) is the right taxonomy. §9 codex P1 anticipation checklist makes E-META-10 enforcement mechanical. |
| **PP-9 PR_ARC backfill** (PR_ARC_INVENTORY.md prepend, +1469 LOC) | **A−** | 8 PR entries prepended (#383, #384, #385, #386, #387, #388, #389, #390). **APPEND-ONLY rule respected** — verified by reading the entries; no prior entries were edited. **Dates accurate** — #385 merge `6f58418`, #386 `33110c8`, #388 `77f2d26`, #389 `b526485` all match `git log main --oneline`. Confidence annotations follow the sprint-12 honest grades (B for Wave F, A− for Wave G). Cross-refs in each entry preserve the trace forward. |
| **PP-10 LATEST_STATE + STATUS_BOARD refresh** (LATEST_STATE.md, STATUS_BOARD.md) | **B+** | Sprint-11/12 D-CSV substrate types section (lines 101-109 of LATEST_STATE.md) correctly enumerates: `QualiaI4_16D`, `CollapseGateEmission`, `MailboxSoA<N>`, `AttentionMaskSoA/Actor/Backend`, `SigmaTierBands/Router`, `WitnessCorpus`, `QualiaI4Column`, `SplatField ×2`, `QualiaStream/InferenceStream/SplatFieldStream`. Verified against `grep QualiaI4_16D crates/lance-graph-contract/src/qualia.rs` and `grep CollapseGateEmission crates/lance-graph-contract/src/lib.rs` — both present. **One slight drift:** the line about ndarray streams says "productization sprint-12" but ndarray master has NO stream module yet (only the local branch); should read "productization in flight (cross-repo PR pending)." Otherwise solid. |
| **PP-11 OQ catalog** (sprint-log-13/oq-catalog.md, 207 LOC) | **B+** | 13 OQs enumerated (OQ-CSV-7..19, with 7..16 as sprint-13 blockers). **Distinct from OQ-CSV-1..6** (verified — no overlap). Recommendations per OQ are Opus-correct: OQ-CSV-11 cites I-VSA-IDENTITIES; OQ-CSV-13 cites ndarray simd_caps pattern. **The numbering drift:** PP-1 plan v3 §12 lists OQ-CSV-7..12 (6 OQs); PP-11 lists OQ-CSV-7..19 (13 OQs, 10 sprint-13 blockers). PP-11 is the wider enumeration and is correct; PP-1 should be updated to match in a follow-on commit. CSI-19 below records this drift. |
| **PP-12 cross-repo audit** (cross-repo-dependency-audit-2026-05.md, 557 LOC) | **C+** | **Hard finding: PP-12's "ndarray PR #147 merged 2026-05-16T04:35:05Z" is FALSE.** Verified `cd /home/user/ndarray && git log master --oneline | head -15`: master HEAD `2a3885d2` is PR #146 merge; there is NO PR #147 in master log. The streams commit `2a1a1e38` exists ONLY on local branch `claude/sprint-12-qualia-stream-w-f4`. PP-12 says "the local checkout is on branch claude/sprint-12-qualia-stream-w-f4 (post-merge of PR #147)" — but `git branch --contains 2a1a1e38` returns ONLY that branch; the merge did not happen. **The convergence.rs claim is correct** (§4.3): `grep "allow(unused_imports)" crates/lance-graph-planner/src/cache/convergence.rs` confirms lines 22/24/26 still carry the annotations; line 26 (DistanceMatrix) is now stale per PP-12 since `PlaneDistance` actually got wired. C+ because the ndarray PR status finding propagates into PP-1 §13.8 and §0.1, doubling the drift. |
| **PP-13 brutally-honest-tester** (.claude/agents/brutally-honest-tester.md, 708 LOC) | **A−** | New agent card per user request. Stable-Rust-only toolchain (clippy + fmt + audit + deny + machete + geiger + semver-checks) — explicit non-fit for Miri/cargo-fuzz/cargo-careful per §1.4. AP1..AP8 anti-pattern catalogue covers the codex-P1 patterns observed in sprint-11/12. §3 verdict format (LAND/HOLD/REJECT) is the binary gate this CCA2A loop lacked. §6 promotion ceremony (calibration period: 3 successful gates + 1 false-positive + 1 missed-bug case study) is appropriately humble for a new agent. **Caveat:** §3 finding table cites I-LEGACY-API-FEATURE-GATED as the cardinal rule — same pending-promotion issue as PP-2; honest §7 cross-references list it as "promotion pending." |

**Wave H aggregate:** 4 A− / 3 B+ / 3 B / 1 B− / 1 C+ / 1 A. Mean grade ≈ B+. The C+ drags the average; the A on PP-8 (worker-template-v2, the structural fix for CSI-13) is the keystone.

---

## 3. Cross-Cutting Findings (CSI-19..23)

These are the additive findings I surface that the individual PP outputs missed. Each is verified against the working tree or against git log, not against planner self-reports.

### CSI-19 (P2) — D-CSV-* and OQ-CSV-* numbering drift across PP-1..PP-11

**Evidence (verified):**
- PP-1 plan v3 §11 lists D-CSV-13b (PP-6), D-CSV-14 (PP-4), D-CSV-15b (sprint-14+), D-CSV-16 (PP-5), D-CSV-17 (PP-3) — internally consistent.
- PP-3 §0 says "D-CSV-17 — chosen to not collide with the in-flight D-CSV-13/14/15 entries in convergence-v2 §11, and to follow the **D-CSV-16 slot reserved by PP-2 (sprint-13 splat on-Think method migration)**" — three errors: PP-2 is iron-rules-doctrine (not splat); PP-4 is splat on-Think; D-CSV-16 is CAM-PQ per PP-5.
- PP-1 plan v3 §12 OQ table lists OQ-CSV-7..12 (6 entries); PP-11 oq-catalog.md lists OQ-CSV-7..19 (13 entries, 10 sprint-13 blockers).
- Each planner chose IDs independently without a coordinator.

**Why this matters:** sprint-13 worker prompts will reference D-CSV-* and OQ-CSV-* numbers; ID drift causes the same class of confusion that CSI-15 (CamPqIndexPlaceholder → WitnessIndexHashMap rename) caused at the type level. A worker reading PP-3 §0 will believe PP-2 = splat on-Think, fail to find that work product, and stall.

**Fix:** pre-spawn aggregation commit on this branch unifies the ID space: PP-3 §0 cross-ref line correction (~3 LOC); PP-1 §12 OQ table extension from OQ-CSV-7..12 to OQ-CSV-7..16 (the 10 sprint-13 blockers, matching PP-11). ~15 LOC total. **Blocker for sprint-13 spawn.**

### CSI-20 (P1) — Multiple planners drifted from main on PR #390-dependent state

**Evidence (verified):**
- PP-1 plan v3 §0.2 says "CSI-18 iron-rule doctrine consolidation: across Wave F + Wave G, three iron-rule-shaped findings recur ... the proposed three are: I-AGGREGATION-DISCIPLINE, I-FEATURE-GATE-FIELD-ISOLATION, I-PLAN-GIT-RECONCILE." This is consistent with PP-7 sprint-12 meta-review.
- PP-2 iron-rules-doctrine §1 axis table lists `I-LEGACY-API-FEATURE-GATED` as a canonical iron rule peer of I-SUBSTRATE-MARKOV / I-NOISE-FLOOR-JIRAK / I-VSA-IDENTITIES, but `grep I-LEGACY-API-FEATURE-GATED /home/user/lance-graph/CLAUDE.md` returns empty — the promotion is on PR #390 branch only.
- PP-6 SIMD spec line 8 cites "I-LEGACY-API-FEATURE-GATED" as "iron rules in force" without the pending qualifier.
- PP-13 brutally-honest-tester §3 cites I-LEGACY-API-FEATURE-GATED as "the cardinal rule this agent enforces" — §7 honestly notes "promotion pending."

**Why this matters:** PR #390 is the gate that materializes I-LEGACY-API-FEATURE-GATED in main. The preflight fleet ran in PARALLEL with the gate. Anything that assumes the gate has merged is preflighting against future state. Same class of drift as PP-1's CSI-9 claim and PP-12's "PR #147 merged."

**Fix:** add a pre-spawn audit pass: every Wave H spec/knowledge/agent that cites I-LEGACY-API-FEATURE-GATED, WitnessIndexHashMap, or i4_eval::batch needs a "(pending PR #390 merge)" qualifier. ~5 docs × 1 line each. **P1, do before sprint-13 spawn.**

### CSI-21 (P2) — Wave H produced ~8000 LOC of planning docs against an estimated ~2000 LOC of sprint-13 IMPL

**Evidence (verified by `wc -l`):**
- PP-1 v3 plan: 597 LOC
- PP-2 doctrine: 392 LOC
- PP-3 rayon spec: 672 LOC (for ~270 LOC actual)
- PP-4 Think spec: 674 LOC (for ~545 LOC actual)
- PP-5 CAM-PQ spec: 551 LOC (for ~615 LOC actual)
- PP-6 SIMD spec: 982 LOC (for ~600 LOC actual)
- PP-7 sprint-12 meta-review: 310 LOC
- PP-8 worker-template-v2: 563 LOC
- PP-9 PR_ARC backfill: ~1469 LOC prepended
- PP-10 LATEST_STATE refresh: not separately counted (mostly modifications)
- PP-11 OQ catalog: 207 LOC
- PP-12 cross-repo audit: 557 LOC
- PP-13 brutally-honest-tester: 708 LOC
- **Total ≈ 7682 LOC of planning docs for ~2030 LOC of sprint-13 impl ≈ 3.8× spec-to-impl ratio.**

**Why this matters:** the sprint-10 spec sprint (PR #372) ran ~5100 LOC spec for the sprint-11 impl wave (~3600 LOC) ≈ 1.4× ratio. The sprint-11 Wave F meta-review noted ~2400 LOC code + ~1200 LOC governance + plans ≈ 1.5× ratio. Wave H is 2.5× higher spec density than precedent. Some of this is justified (PP-8 worker-template-v2 has compounding value across all future waves; PP-2 iron-rules-doctrine has cross-sprint shelf life). But PP-6 SIMD spec at 982 LOC for ~600 LOC impl is over-planned — a 2:1 ratio would suffice, and the extra 400 LOC of spec text will be obsolete the moment the first AVX-512 intrinsic returns a different lane count than the pseudocode assumes.

**Fix:** for sprint-14 preflight, cap per-spec LOC at 2× impl LOC. Specs above the cap get split. **P2, doctrinal carry-forward.**

### CSI-22 (P2) — PP-13 introduces a NEW pre-commit gate; clarify the ensemble pipeline post sprint-13

**Evidence (verified):**
- PP-13 §4.1 CCA2A loop slot diagram places brutally-honest-tester AFTER worker DONE and BEFORE commit.
- W-Meta-Opus (current pattern, this file's authoring) runs AFTER commits land, against the working tree.
- §4.3 "Relationship to existing tooling" — PP-13 honestly distinguishes: codex bot runs post-push in PR thread; brutally-honest-tester runs LOCAL pre-emption; meta-review (Opus) runs AFTER commit looking for cross-spec inconsistencies.
- §6 calibration period (3 successful gates + 1 false-positive + 1 missed-bug) places PP-13 in probation status — its verdicts are advisory, orchestrator may override.

**Why this matters:** the CCA2A pipeline currently has worker-self-validation (worker-template-v2 §6) and meta-Opus-post-commit (this file's role). PP-13 fills the per-PR pre-commit gap. This is genuinely additive, but the ensemble must clarify: does PP-13 REPLACE W-Meta-Opus on Wave-G-style 6-worker fleets, or does it ADD a step? The honest answer per §4.3 is "different time-scale, different audience" — brutally-honest-tester is per-PR-pre-commit; W-Meta-Opus is per-wave-post-commit. Both should run.

**Fix:** sprint-13 worker template v2 §11 spawn skeleton should add a "post-worker-DONE, pre-commit: spawn brutally-honest-tester" step. ~3 LOC addition. **P2, sprint-13 process refinement.**

### CSI-23 (P0) — PR #390 in-flight is the gate for both (a) main carrying the iron rule + WitnessIndexHashMap rename + Σ-tier Jirak math, AND (b) sprint-13 spawn

**Evidence (verified):**
- `git log main --oneline | head -5`: latest main is `b526485` (PR #389 merge). PR #390 not yet merged.
- `grep WitnessIndexHashMap crates/lance-graph/src/graph/arigraph/witness_corpus.rs` returns empty — still `CamPqIndexPlaceholder`.
- `grep I-LEGACY-API-FEATURE-GATED CLAUDE.md` returns empty.
- PP-7 sprint-12 meta-review §8 pre-spawn checklist item #1: "PR #390 (Wave G) merges to main" — **Blocker.**

**Why this matters:** the entire preflight fleet ran in PARALLEL with the PR #390 gate. Most planners correctly footnote pending-merge status (PP-5 §0 cites "post-CSI-15 rename PR #390"; PP-2 §3.4 cites "promotion text drafted in sprint-log-11/meta-review-opus.md CSI-18"; PP-13 §7 cites "promotion pending"). The risk: PR #390 contains the four-iron-rule CLAUDE.md update + the CamPqIndexPlaceholder → WitnessIndexHashMap rename + the Σ-tier Jirak math + the cognitive-shader-driver workspace fix. If PR #390 lands a different commit hash than the preflight fleet read against, the cross-refs need a sweep.

**Fix:** the sprint-13 spawn gate REQUIRES PR #390 merge first. After PR #390 lands, run a one-pass sweep: every Wave H doc that cites PR #390 or its deliverables gets its anchor hash updated. ~10 minutes of governance work. **Blocker for sprint-13 spawn.**

---

## 4. Sprint-13 Spawn-Readiness Gate

**Recommendation: spawn sprint-13 ONLY AFTER the following pre-fleet hygiene completes.**

### Pre-spawn checklist (ordered, blockers first)

| # | Item | Owner | Severity | Notes |
|---|---|---|---|---|
| 1 | **PR #390 merge** (Wave G) | Main thread post-review | **Blocker (CSI-23)** | Carries I-LEGACY-API-FEATURE-GATED iron rule + WitnessIndexHashMap rename + Σ-tier Jirak math + cognitive-shader-driver workspace fix. **Sprint-13 spawn is gated on this.** |
| 2 | **Wave H (this branch) merge** | Main thread post-this-review | **Blocker** | Carries PP-1..PP-13 planning outputs; sprint-13 worker prompts reference them. |
| 3 | **CSI-19 ID-numbering consolidation** | Main thread aggregation commit on this branch BEFORE merge | **Blocker** | PP-3 §0 cross-ref correction (~3 LOC); PP-1 §12 OQ table extension to OQ-CSV-7..16 (~10 LOC). |
| 4 | **CSI-20 pending-status sweep** | Main thread aggregation commit on this branch BEFORE merge | P1 | Every cite of I-LEGACY-API-FEATURE-GATED / WitnessIndexHashMap / Σ-tier Jirak math gets "(pending PR #390 merge)" qualifier. ~5 docs × 1 line. |
| 5 | **CSI-9 cross-repo ndarray PR** (register `qualia` + `splat_field` in `/home/user/ndarray/src/hpc/stream/mod.rs` upstream) | Cross-repo coordination | **Blocker on D-CSV-11 productization** | ~4 LOC in ndarray master. PP-1 §13.8 + PP-12 §1 both incorrectly state this is resolved; it is NOT — verified against `cd /home/user/ndarray && git log master`. |
| 6 | **OQ-CSV-7..16 user ratification** (10 sprint-13 blockers per PP-11) | User | **Blocker** | OQ-CSV-7 (rayon feature gate name), OQ-CSV-8 (par_* chunk size), OQ-CSV-9 (splat carrier on Think vs sibling), OQ-CSV-10 (splat generation source), OQ-CSV-11 (SPO→256D adapter Option A vs B), OQ-CSV-12 (CAM-PQ lazy vs eager), OQ-CSV-13 (SIMD dispatch runtime vs compile-time), OQ-CSV-14 (bench speedup floor SHIP vs LAND), OQ-CSV-15 (worker workspace member edit ownership), OQ-CSV-16 (iron-rules-doctrine in BOOT.md Tier-1). |
| 7 | **Worker-template-v2 promotion to canonical** | Main thread | P1 (process) | PP-8 ships the template; promotion means `.claude/agents/README.md` adds the row and BOOT.md ensemble list cites it. ~5 LOC. |
| 8 | **Brutally-honest-tester promotion to probation-tier ensemble member** | Main thread | P1 (process) | PP-13 §6 calibration period: 3 successful gates + 1 false-positive + 1 missed-bug case study. Promote to canonical AFTER calibration; admit to BOOT.md Tier-1 trigger table NOW per PP-13 §5. |

### Standing ratifications from sprint-11/12 (carry forward)

- **OQ-CSV-1** (qualia vocab choice): RESOLVED to Option α.
- **OQ-CSV-2** (W-slot 6 bits): RESOLVED.
- **OQ-CSV-4** (D-CSV-5 phasing): RESOLVED (sibling-then-cutover).
- **OQ-CSV-6** (Jirak threshold): PARTIAL — hand-tuned accepted sprint-11/12; D-CSV-15b sprint-13+ for VAMPE-coupled refinement.

### Forward-looking sprint-13 deliverables (per PP-1 v3 + PP-3..PP-6 specs)

- **D-CSV-5b** — QualiaColumn cutover (drop legacy `[f32; 17]`) after consumer audit.
- **D-CSV-6b** — Full WitnessCorpus (pruning + salience decay) — depends on D-CSV-16.
- **D-CSV-13b** — SIMD intrinsics AVX-512 + NEON (PP-6 spec).
- **D-CSV-14** — On-Think method migration (PP-4 spec).
- **D-CSV-16** — Real CAM-PQ wiring (PP-5 spec).
- **D-CSV-17** — par_*_stream rayon variants (PP-3 spec).

---

## 5. The Honest Reflection

The CCA2A preflight pattern at 13-planner scale revealed three structural lessons:

**First, Opus accumulation produces deeper plans than Sonnet — but plan-quality and plan-vs-git-state-coherence are orthogonal axes.** PP-4 / PP-5 / PP-6 / PP-8 are deeper than any Sonnet-authored spec in this workspace's history; they cite iron rules with the correct test patterns, they grep for callers before committing to migrations, they carve out edge cases like `GateDecision::Block { reason: String }` payloads. But PP-1 and PP-12 both shipped the same class of plan-vs-git drift as W-F12's CSI-11 in sprint-11 — the lesson "every plan-author worker must run `git log origin/main -20` and reconcile" (CSI-11 honest-reflection, sprint-11 W-Meta-Opus) was NOT systematized into the preflight prompts, and the same failure mode recurred. **Sprint-14 preflight worker prompts MUST mandate `cd /home/user/<repo> && git log master --oneline -20` for every sibling-repo cited.** PP-8 worker-template-v2 §3 mentions reading recent state but does not codify the cross-repo git-log step.

**Second, ID coordination is invisible work that orphans without a designated coordinator.** Wave H produced 13 planner outputs in 8 commits, each planner choosing D-CSV-* and OQ-CSV-* numbers independently. PP-1 plan v3 was meant to be the coordinator (per its §17.5 "PP-3..PP-6 draft the new D-CSV-* specs"), but PP-3 was authored against an earlier ID assignment that didn't survive into PP-1 v3. The fix: every multi-planner preflight wave must have a numbering-coordinator step that runs AFTER all planners but BEFORE the meta-review. This is the same shape as the CSI-13 / CSI-7/8/9 "main thread aggregates" fix — there was an invisible coordination phase that nobody owned.

**Third, the preflight cost/benefit equation depends on the spec's shelf life.** PP-8 worker-template-v2 will pay back across every future wave for the rest of the workspace's life. PP-2 iron-rules-doctrine has cross-sprint shelf life — every future iron-rule promotion will read it. PP-13 brutally-honest-tester pays back per-PR going forward. These three justify their preflight cost easily. PP-3 / PP-4 / PP-5 / PP-6 are one-sprint-shelf-life specs — they describe sprint-13's impl scope and will be obsolete the moment sprint-13 ships. For these, the 2.5× spec-to-impl LOC ratio is over-planning relative to the precedent ratio (~1.5×). Sprint-14 preflight should distinguish: doctrinal/process specs uncapped (PP-2 / PP-8 / PP-13 class) vs impl specs capped at 2× impl LOC.

These three are the workflow-discipline equivalents of the cross-spec drifts. The cross-spec drifts get caught by Opus meta-review; the workflow drifts get caught only by someone like me looking for them. Sprint-14 preflight worker prompts should bake the numbering-coordinator, the cross-repo-git-log, and the spec-LOC-cap disciplines in.

---

## 6. Cross-references

- **PP-1 plan v3:** `.claude/plans/cognitive-substrate-convergence-v3.md` (needs CSI-19 + CSI-20 fixes)
- **PP-2 doctrine:** `.claude/knowledge/iron-rules-doctrine.md` (needs CSI-20 fix — qualify I-LEGACY-API-FEATURE-GATED as pending)
- **PP-3 rayon spec:** `.claude/specs/pr-sprint-13-rayon-streams.md` (needs CSI-19 fix — PP-3 §0 cross-ref correction)
- **PP-4 Think methods spec:** `.claude/specs/pr-sprint-13-think-methods.md` (clean)
- **PP-5 CAM-PQ spec:** `.claude/specs/pr-sprint-13-witness-cam-pq.md` (clean)
- **PP-6 SIMD spec:** `.claude/specs/pr-sprint-13-simd-i4.md` (needs CSI-20 fix)
- **PP-7 sprint-12 meta-review:** `.claude/board/sprint-log-12/meta-review.md` (clean)
- **PP-8 worker-template-v2:** `.claude/agents/worker-template-v2.md` (clean, ready for canonical promotion)
- **PP-9 PR_ARC backfill:** `.claude/board/PR_ARC_INVENTORY.md` (8 entries prepended, clean)
- **PP-10 LATEST_STATE refresh:** `.claude/board/LATEST_STATE.md` (minor drift on ndarray streams "shipped")
- **PP-11 OQ catalog:** `.claude/board/sprint-log-13/oq-catalog.md` (canonical OQ-CSV-7..19 enumeration; PP-1 §12 must align)
- **PP-12 cross-repo audit:** `.claude/knowledge/cross-repo-dependency-audit-2026-05.md` (CSI-20 fix needed; the "PR #147 merged" claim is FALSE and propagates)
- **PP-13 brutally-honest-tester:** `.claude/agents/brutally-honest-tester.md` (clean, ready for probation-tier promotion)
- **Predecessor reviews:** `.claude/board/sprint-log-11/meta-review-opus.md`, `.claude/board/sprint-log-12/meta-review.md`

---

*End of sprint-13 preflight Opus meta-review. W-Meta-Opus (Opus 4.7), main-thread, 2026-05-16. Authored after independent verification pass on PP-1..PP-13 outputs against the working tree at HEAD = 04620aa and against main at HEAD = b526485. Grades are independent of planner self-reports. The Wave H grade of B+ is gated on the CSI-19/CSI-20/CSI-23 pre-spawn fixes; once those land in an aggregation commit on this branch and PR #390 merges, the sprint-13 spawn-readiness gate is closed and the impl fleet can fan out against PP-3/PP-4/PP-5/PP-6 specs with PP-8 worker-template-v2 as the spawn template and PP-13 brutally-honest-tester as the per-PR pre-commit gate.*
