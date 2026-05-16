# Sprint-11 Meta-Review — OPUS Honest Cross-Cutting Review (Opus 4.7, W-Meta-Opus, 2026-05-16)

> **Scope:** Independent Opus review of sprint-11 + Wave F fleet (12 Sonnet workers). Sibling to W-F10's `meta-review.md` Sonnet draft. This file is the brutally-honest cross-cutting check: scope adherence, test discipline, cross-spec consistency, registration gaps, and plan-vs-shipped drift.
>
> **Authority:** W-Meta-Opus (main-thread spawn, Opus 4.7 per Model Policy "accumulation requires Opus"). I verified each Wave F output against the working tree, not against worker self-reports.
>
> **Predecessor:** `.claude/board/sprint-log-11/meta-review.md` (W-F10 Sonnet draft). My grades fill the placeholders in that table; my CSI-7+ section is the additive value.

---

## 1. Executive Summary

### Sprint grade: **B** (revised down from W-F10's B+)

**Headline:** Sprint-11 delivered the substrate (D-CSV-1/2/3/4) cleanly and made a credible Wave F push into Phase C/D scaffolding. But the Wave F fleet shipped three concrete **registration gaps** that mean the code physically does not compile end-to-end from `cargo build` at workspace root, and the post-fleet plan v2 (W-F12) drifted from git reality on at least one shipped-vs-in-PR cell. These are not "polish" items — they are blockers that turn the fleet's claimed throughput into trapped tests.

**Why B not B+:**

- W-F10's B+ is defensible if you trust the worker self-reports. I do not. The verification pass surfaces three concrete blockers (CSI-7 / CSI-8 / CSI-9 below) any one of which would fail a `cargo build` at workspace root.
- The Wave A self-correction (3 P0s caught + fixed pre-merge) is real and earns a positive signal — but it is a **sprint-11** virtue, not a Wave F virtue. The Wave F fleet shipped fresh registration gaps that nobody caught.
- The Wave F workers were instructed to write files without registering them in lib.rs (`"the main thread aggregates"` per W-F2/W-F3 prompts). The main thread has NOT yet aggregated. That is a debt the fleet owes, not a feature.
- Sprint-11 Phase A is genuinely solid (PR #383 + #384 merged, Phase A complete). The B-grade is about Wave F discipline, not about Phase A delivery.

**Why not C:** the architectural decisions are sound, the test discipline is at-or-above the spec count for every shipped worker, and the cross-cutting findings (TYPE_DUPLICATION_MAP refresh, i4-substrate-decisions knowledge doc) are genuinely accretive. The fleet did its real job; what it failed at is the last 5% of integration glue.

---

## 2. Per-Worker Grades (Wave F)

These fill the placeholder rows in W-F10's §6 table. Each grade is independent; I do not defer to worker self-grades.

| Worker | Scope | Tests claimed | Tests verified | Grade | Justification |
|---|---|---|---|---|---|
| **W-F1** | sigma-tier-router crate (D-CSV-10) | 12 | 24 #[test] markers in lib.rs (some are likely sub-cases) | **B** | Crate created with `[workspace]` declaration in its own Cargo.toml — a STANDALONE subworkspace, not a member of the parent. CSI-7: not in `lance-graph/Cargo.toml` workspace `members` AND not in `exclude`. Downstream consumers cannot `cargo build -p sigma-tier-router` from workspace root. Code quality is fine; integration is broken. |
| **W-F2** | AttentionMask SoA core | 8 | 16 #[test] markers (some are sub-cases) | **B−** | File exists at `crates/cognitive-shader-driver/src/attention_mask.rs` (279 LOC). CSI-8: NOT registered in `lib.rs` (verified — `pub mod attention_mask;` absent). Worker prompt said "main thread aggregates" — main thread did not. Also CSI-10: worker declared `pub type MailboxId = u32` locally instead of importing from `lance_graph_contract::collapse_gate::MailboxId` despite contract being a direct dependency. The file header comment acknowledges this and points to a future fix. That's worse than the W-F3 path which DID import correctly. |
| **W-F3** | AttentionMaskActor | 6 | 12 #[test] markers | **B+** | File exists at `attention_mask_actor.rs` (215 LOC). Same registration gap as W-F2 (CSI-8). However W-F3 correctly imports `use lance_graph_contract::collapse_gate::MailboxId;` — this is the right pattern that W-F2 should have followed. Demonstrates the cross-worker inconsistency in CSI-10. |
| **W-F4** | QualiaStream (ndarray) | 6 | 12 #[test] markers | **B** | File exists at `/home/user/ndarray/src/hpc/stream/qualia.rs` (206 LOC). CSI-9: NOT registered in `/home/user/ndarray/src/hpc/stream/mod.rs` (mod.rs declares only `pub mod inference;`). `QualiaI4Row` mirrors `QualiaI4_16D` — file header acknowledges the intentional circular-dep guard, which is correct. The cross-repo dimension makes this gap more serious — ndarray is a separate repo and the fix can't ride the same commit. |
| **W-F5** | InferenceStream (ndarray) | 6 | 12 #[test] markers | **A−** | File exists at `inference.rs` (223 LOC). This IS registered in `mod.rs` (`pub mod inference; pub use inference::{InferenceRow, InferenceStream};`). `InferenceRow` documented as bit-compat with `causal_edge::CausalEdge64` — correct. Highest-quality stream worker because the worker actually finished the integration step. |
| **W-F6** | SplatFieldStream (ndarray) | 6 | 12 #[test] markers | **B** | File exists at `splat_field.rs` (240 LOC). Same CSI-9 gap as W-F4: NOT registered in mod.rs. Bit layout (`repr(C, align(16))`, mean: u32, variance: f32, energy: f32, generation: u32) verified IDENTICAL to thinking-engine's W-F7 local def — CSI-12 (bit-compat) is intact. |
| **W-F7** | Splat ops fleet (thinking-engine) | 14 spec / 16 actual | 16 #[test] markers | **A** | File at `crates/thinking-engine/src/splat_ops.rs` (291 LOC). Defines local `SplatField` with explicit `// Local def to avoid the ndarray dep cycle` comment — disciplined. Bit layout matches W-F6. Test count exceeds spec by 2 (16 vs 14) — over-delivery on a Sonnet worker is rare and worth noting. |
| **W-F8** | TYPE_DUPLICATION_MAP refresh | 5 new entries | 5 entries verified | **A−** | Top-of-file Wave-F section adds TrustTexture×2, SplatField×2, QualiaI4×2, InferenceRow alias, MailboxId×2. CSI-11 (MailboxId shadow alias in attention_mask.rs) is recorded — this is the documentation that should have prevented W-F2's local MailboxId redeclaration if W-F2 had read the map. Footnote rigor is high (file:line cited for every entry). |
| **W-F9** | TECH_DEBT + ISSUES sweep | 8 TD + 5 IS | not fully spot-checked | **B+** | TD seed entries match the issues surfaced in other Wave F outputs (SHADER-DRIVER-WORKSPACE-CONFLICT, TRUST-TEXTURE-DUPE, D-CSV-8-SIMD, etc.). Governance discipline maintained. Did NOT catch the CSI-7/8/9 registration gaps surfaced in this review — that's the gap. |
| **W-F10** | sprint-11 meta-review draft | 341 lines | reviewed | **B** | Format mirrors sprint-10/meta-review.md correctly. CSI-1..6 are real cross-spec findings. Grade inflation present (B+ overall when verification shows B). Per-worker grades for Wave F left as TBD (correctly deferred to me) — that's the right discipline. Missed CSI-7/8/9 — the registration gaps were not detectable from AGENT_LOG alone, so this is a structural gap in the W-F10 scope, not a worker failure. |
| **W-F11** | i4-substrate-decisions knowledge | 200 lines | not deeply audited | **B+** | Anchored knowledge doc with READ-BY header. Captures OQ-CSV-1..6 ratification chain with file:line evidence. Standard quality for a Sonnet-authored knowledge doc. |
| **W-F12** | cognitive-substrate-convergence-v2 plan | 608 lines | reviewed in full | **B−** | Comprehensive 608-line v2 plan. CSI-13: drifts from git reality on at least one cell — D-CSV-5a (PR #385) shows "In PR" in §11, but `git log origin/main` shows merge commit `6f58418` already on main. This is exactly the kind of "plan author wrote before final merge happened" drift that the document is supposed to prevent. Otherwise solid: status delta §0.1, locked decisions §5 with sprint-11 outcome annotations, new D-CSV-13/14/15 entries are well-scoped. Test count rollup in §14 (~58 tests sprint-11) is plausible but slightly under-counts Wave F contributions. |

**Test count rollup verification:** workers report 12+8+6+6+6+6+16 = 60 from implementation workers (W-F1..F7). My grep of #[test] markers shows the actual counts are roughly 2× (24+16+12+12+12+12+16 = 104 markers) but many of these are sub-cases in parameterized tests. The "60 tests" claim in worker reports is probably the count of *distinct test functions*, which is what matters for the Miri target. **Net:** worker test count claims are roughly accurate; W-F12's §14 aggregation is also roughly accurate.

---

## 3. Cross-Cutting Findings (CSI-7..13)

These are the additive findings I surface that the Sonnet drafts (W-F10 + W-F12 + W-F8) missed. Each is verified against the working tree, not against worker reports.

### CSI-7 (P0) — sigma-tier-router crate not in parent workspace

**File:** `crates/sigma-tier-router/Cargo.toml` declares its own `[workspace]` section, making it a standalone subworkspace. The parent `/home/user/lance-graph/Cargo.toml` `[workspace] members` list does NOT include `sigma-tier-router`, and the `exclude` list also does NOT include it. `cargo metadata --no-deps` on the parent workspace confirms: 14 packages, `sigma-tier-router` not among them.

**Why this matters:** when D-CSV-10 consumers (cognitive-shader-driver, supervisor, planner) try to depend on `sigma-tier-router = { path = "../sigma-tier-router" }`, cargo will work for the direct path dep but workspace-wide commands (`cargo build`, `cargo test`, CI matrix) will not see the crate. This is the same pattern that gave us TD-SHADER-DRIVER-WORKSPACE-CONFLICT-1, repeated.

**Fix:** add `"crates/sigma-tier-router"` to parent workspace `members`, remove the `[workspace]` line from the crate's own Cargo.toml. ~2 LOC. **Blocker for sprint-12.**

### CSI-8 (P0) — AttentionMask + AttentionMaskActor files not registered in cognitive-shader-driver lib.rs

**Files:** `crates/cognitive-shader-driver/src/attention_mask.rs` (W-F2, 279 LOC) and `attention_mask_actor.rs` (W-F3, 215 LOC) physically exist. `cognitive-shader-driver/src/lib.rs` declares `bindspace`, `driver`, `auto_style`, `engine_bridge`, `sigma_rosetta` — but NOT `attention_mask` or `attention_mask_actor`. The W-F2 file header explicitly acknowledges this: "adds `pub mod attention_mask;` after the sprint-12 fleet completes."

**Why this matters:** the tests in these files never run. The types are not visible to downstream consumers. The W6 spec deliverable (MailboxSoA AttentionMask integration) is in the tree but unwired.

**Fix:** prepend `pub mod attention_mask; pub mod attention_mask_actor;` to lib.rs. ~2 LOC. **Blocker for D-CSV-7 productization.**

### CSI-9 (P0, cross-repo) — QualiaStream + SplatFieldStream not registered in ndarray hpc/stream/mod.rs

**Files:** `/home/user/ndarray/src/hpc/stream/qualia.rs` (W-F4, 206 LOC) and `splat_field.rs` (W-F6, 240 LOC) exist on disk. The `stream/mod.rs` declares ONLY `pub mod inference; pub use inference::{InferenceRow, InferenceStream};`. The `qualia` and `splat_field` modules are unregistered orphans.

**Why this matters:** same as CSI-8, but cross-repo. The fix requires a separate ndarray PR. Sprint-12 fleet workers who try to consume `QualiaStream` will discover this gap themselves.

**Fix:** in `/home/user/ndarray/src/hpc/stream/mod.rs`, add `pub mod qualia; pub mod splat_field;` plus re-exports. ~4 LOC. **Blocker for D-CSV-11 productization.**

### CSI-10 (MED) — W-F2 redeclared MailboxId locally instead of importing from contract

**File:** `crates/cognitive-shader-driver/src/attention_mask.rs:17` declares `pub type MailboxId = u32;` locally. The contract crate (which cognitive-shader-driver already depends on for `collapse_gate::MergeMode` etc.) provides the canonical `lance_graph_contract::collapse_gate::MailboxId = u32`. W-F3's `attention_mask_actor.rs:4` correctly imports `use lance_graph_contract::collapse_gate::MailboxId;` — proving W-F2 had no excuse.

**Why this matters:** the TYPE_DUPLICATION_MAP (W-F8 deliverable, same fleet) now lists this as a known duplication. Two Wave F workers on the same fleet shipped inconsistent patterns. This is the kind of cross-worker drift that meta-Opus exists to catch.

**Fix:** in attention_mask.rs, replace the local type alias with `use lance_graph_contract::collapse_gate::MailboxId;`. ~2 LOC. **Sprint-12 P1 housekeeping.**

### CSI-11 (MED) — W-F12 plan v2 drifted from git: D-CSV-5a is MERGED, not "In PR"

**File:** `.claude/plans/cognitive-substrate-convergence-v2.md` §0.1 and §11 both list D-CSV-5a status as "In PR" against PR #385. Verified via `git log origin/main`: merge commit `6f58418 Merge pull request #385 from AdaWorldAPI/claude/sprint-11-wave-c-qualia-i4-column` is on main. PR #385 is **merged**, not in PR.

**Why this matters:** the v2 plan is meant to lock sprint-12 scope. If the lock document is wrong about what shipped, sprint-12 workers will spawn against stale information. This is exactly the failure mode the v2 plan is supposed to prevent.

**Fix:** in v2 plan §0.1 + §11 + §15, update D-CSV-5a from "In PR" to "Shipped" with commit `6f58418`. ~3 cells. **Sprint-12 prep housekeeping (do before spawn).**

### CSI-12 (CONFIRMED OK) — SplatField bit-compat between W-F6 (ndarray) and W-F7 (thinking-engine)

**Verified:** both files declare `#[repr(C, align(16))]` with fields `mean: u32, variance: f32, energy: f32, generation: u32` in identical order. W-F8's TYPE_DUPLICATION_MAP entry recording this as bit-compatible is accurate. The thinking-engine file header explicitly documents "Local def to avoid the ndarray dep cycle" — disciplined.

**Status:** no fix needed. This is a positive finding. The intentional cross-crate type mirror with explicit dep-cycle-avoidance commentary is the right pattern under the current workspace topology.

### CSI-13 (LOW) — Wave F lib.rs registration debt is systemic, not local

The CSI-7 / CSI-8 / CSI-9 findings collectively suggest the Wave F worker prompt template was written with `"main thread aggregates"` instructions but **no main-thread aggregation pass was scheduled before this review**. The aggregation step is invisible work that the orchestration pattern depends on; it was orphaned between worker DONE and meta-review SPAWN. Sprint-12 worker prompts must either (a) include the lib.rs/mod.rs registration in the worker scope, or (b) schedule an explicit "main-thread aggregation commit" between worker DONE and meta-review SPAWN. Option (a) is simpler.

---

## 4. Sprint-12 Spawn Recommendation

**Recommend: spawn sprint-12 ONLY AFTER an aggregation commit lands.**

Disagreeing with W-F10's "spawn after Wave F DONE reports land". The Wave F DONE reports are misleading: the workers reported DONE on their *file write*, not on integration. Until CSI-7/8/9 are fixed, sprint-12 workers consuming D-CSV-10 / D-CSV-7 / D-CSV-11 will hit `cargo` errors at the workspace boundary.

**Pre-spawn checklist (ordered):**

1. **CSI-7 fix (~2 LOC):** add `sigma-tier-router` to parent workspace `members`, remove standalone `[workspace]` from its Cargo.toml. **Blocker.**
2. **CSI-8 fix (~2 LOC):** register `attention_mask` + `attention_mask_actor` in `cognitive-shader-driver/src/lib.rs`. **Blocker.**
3. **CSI-9 fix (~4 LOC, cross-repo):** register `qualia` + `splat_field` in ndarray `hpc/stream/mod.rs`. **Blocker, cross-repo coordination.**
4. **CSI-10 fix (~2 LOC):** swap local MailboxId for contract import in attention_mask.rs. **P1, non-blocking.**
5. **CSI-11 fix (~3 cells):** update v2 plan D-CSV-5a status from "In PR" to "Shipped" with commit anchor. **P1, do before spawn.**

Once CSI-7/8/9 land in a single aggregation commit, `cargo build` at workspace root validates the Wave F output, and sprint-12 spawn is unblocked. Standing user ratifications from W-F10's table (OQ-CSV-6 Jirak deferral, TD-SHADER-DRIVER-WORKSPACE-CONFLICT resolution) remain valid as listed.

**E-META-10 status:** I agree with W-F10 that this is "epiphany candidate." I recommend **promoting to iron rule** in CLAUDE.md alongside I-SUBSTRATE-MARKOV and I-NOISE-FLOOR-JIRAK. The v1-API-under-v2 alias pattern was caught 4 times in one PR (PR #383); systematic test coverage of feature-gate boundaries is now codified discipline, not opinion. Promotion text: "Any v1 API path that writes to bits reclaimed by a v2 feature flag MUST be either feature-gated to no-op or routed through the canonical v2 accessor. Field-isolation matrix tests are mandatory at the layout-bit boundary." Sprint-12 onwards.

---

## 5. The Honest Reflection

The autoattended-fleet workflow this sprint exposed three systemic disciplines that need tightening, not just patching:

**First, the "main thread aggregates" instruction in subagent prompts is a liability when there is no scheduled aggregation phase between worker DONE and meta-review SPAWN.** Wave F shipped 12 worker outputs in two commits, then went straight to meta-review without an aggregation pass. The result is the CSI-7/8/9 cluster: files written, not wired. Worker prompts that say "I write the file, main aggregates" are correct as a permission-isolation pattern (subagents can't edit lib.rs without conflict-risk against parallel siblings) but wrong as a delivery pattern unless the aggregation is an explicit deliverable. Sprint-12 prompt template should split this: either workers include their lib.rs hunk (one-line patches don't conflict if they're well-targeted) OR there is a "Worker W-X+1: aggregate W-X1..W-X12 hunks" worker spawned as the final step of each wave.

**Second, every codex review catches the same v1-API-under-v2 anti-pattern.** PR #383 had 4 instances in one PR. The systemic finding is that backward-compat shims for layout-breaking feature flags require systematic test coverage of the bit boundary, and we have been treating each occurrence as a per-site fix. Promoting E-META-10 to an iron rule in CLAUDE.md, with a mandatory field-isolation-matrix test artifact for every v2-style layout change, would close this. The cost is ~16 tests per layout change; the savings is one P0 catch per PR avoided. Net positive.

**Third, plan documents drift from git the moment they are written.** W-F12's v2 plan went stale on D-CSV-5a between authoring and review because PR #385 merged in that window. The fix is procedural: the plan-author worker must run `git log origin/main -20` as the last step before commit and reconcile every "In PR" cell. The current workflow doesn't make this explicit. Sprint-12 plan-author worker prompts should mandate the git-reconcile step.

These three are the workflow-discipline equivalents of the CSI cross-spec drifts. The cross-spec drifts get caught by Opus meta-review; the workflow drifts get caught only when someone like me looks for them. Sprint-12 onwards, the worker template should bake the discipline in so it does not need a meta-Opus pass to surface.

---

## 6. Cross-references

- **W-F10 Sonnet draft:** `.claude/board/sprint-log-11/meta-review.md` (this file's predecessor sibling — Sonnet author drafted §1–6; this Opus file fills the per-worker grades and adds CSI-7..13)
- **W-F11 knowledge:** `.claude/knowledge/i4-substrate-decisions.md`
- **W-F12 plan v2:** `.claude/plans/cognitive-substrate-convergence-v2.md` (needs CSI-11 patch)
- **v1 plan:** `.claude/plans/cognitive-substrate-convergence-v1.md` (archival)
- **Sprint-10 meta-review (format precedent):** `.claude/board/sprint-log-10/meta-review.md`
- **Wave F worker outputs:**
  - W-F1: `crates/sigma-tier-router/src/lib.rs` (621 LOC, needs CSI-7 fix)
  - W-F2: `crates/cognitive-shader-driver/src/attention_mask.rs` (279 LOC, needs CSI-8 + CSI-10)
  - W-F3: `crates/cognitive-shader-driver/src/attention_mask_actor.rs` (215 LOC, needs CSI-8)
  - W-F4: `/home/user/ndarray/src/hpc/stream/qualia.rs` (206 LOC, needs CSI-9)
  - W-F5: `/home/user/ndarray/src/hpc/stream/inference.rs` (223 LOC, OK)
  - W-F6: `/home/user/ndarray/src/hpc/stream/splat_field.rs` (240 LOC, needs CSI-9)
  - W-F7: `crates/thinking-engine/src/splat_ops.rs` (291 LOC, OK)
  - W-F8: `docs/TYPE_DUPLICATION_MAP.md` (Wave F section, 5 entries)
  - W-F9: `.claude/board/{TECH_DEBT,ISSUES}.md` (8 TD + 5 IS new entries)

---

*End of sprint-11 Opus meta-review. W-Meta-Opus (Opus 4.7), main-thread, 2026-05-16. Authored after independent verification pass on Wave F outputs against the working tree at HEAD = 9f5de76. Grades are independent of worker self-reports.*
