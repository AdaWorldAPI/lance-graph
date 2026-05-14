## 2026-05-14T12:26 — W11 spec-draft started (sonnet)

**Role:** test-plan-unification — unified regression test plan across all 8 PRs + Miri extension + clippy gate
**Output target:** `.claude/specs/sprint-10-test-plan.md`
**Plans cited:** causaledge64-mailbox-rename-soa-v1.md §3/§7/§12; scripts/miri-tests.sh (lance-graph + ndarray); .github/workflows/rust-test.yml + style.yml; MANIFEST.md worker table; LATEST_STATE.md sprint context
**Mandatory reads completed:** MANIFEST.md (fleet table), parent plan §3/§7/§12, LATEST_STATE.md, AGENT_LOG.md (W12 only agent so far), scripts/miri-tests.sh (lance-graph), ndarray/scripts/miri-tests.sh, rust-test.yml + style.yml (CI structure), .claude/specs/ directory (no sprint-10 worker specs yet — W1-W9 not yet run)
**Key observation:** W1-W9 worker specs do not yet exist (W12 is first runner). W11 must draft the test-plan spec from parent plan + miri scripts + CI structure alone, with placeholders for per-worker spec citations.
**Drafting spec now.**

## 2026-05-14T12:29 — W11 spec complete

**Spec written:** `.claude/specs/sprint-10-test-plan.md` — 43,718 bytes, 775 lines
**Plans cited:** causaledge64-mailbox-rename-soa-v1.md §3/§5/§6/§7/§8/§12; scripts/miri-tests.sh (lance-graph); ndarray/scripts/miri-tests.sh; rust-test.yml; style.yml; MANIFEST.md
**Key delta:** Miri coverage growth target = ~760 current → ~1550 post-sprint-11 (3 mechanisms: W8 dispatch reroute +529 ndarray, new pure-Rust crates +116 lance-graph, expanded lance-graph Miri scope +145)
**Open questions for meta-review:**
  - OQ-T1: W1-W9 test count reconciliation (specs not yet authored)
  - OQ-T2: proptest Miri runtime (PROPTEST_CASES=100 mitigation untested)
  - OQ-T3: bevy headless approach (xvfb-run vs bevy/headless feature)
  - OQ-T4: causal-edge Miri scope (confirm pure-Rust dep closure; W2 must verify)
  - OQ-T5: ractor Miri guard in lance-graph-supervisor (W7 must add cfg(not(miri)) guards)
