# Agent W5 — TECH_DEBT.md unified-OGIT remaining-wiring rows (sprint-2)

**Branch:** `claude/unified-ogit-architecture-synthesis`
**Date:** 2026-05-07
**Role:** Append 11 tech-debt rows to `.claude/board/TECH_DEBT.md` capturing the ~20% of unified-OGIT architecture wiring not yet shipped in workspace, plus three substrate-misclassification reframes that W6 produced against the entropy ledger.

## Scope (immutable, as briefed)

- APPEND a single dated section `## 2026-05-07 — Unified OGIT Architecture: remaining wiring work (sprint-2)` at the END of `TECH_DEBT.md`.
- Add 11 TD rows (TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11).
- Each row carries Title / Region / Severity-Effort / Where / What / Plan / Dependencies.
- Cross-reference sister-worker plan-docs (W10 / W11 / W12) and ledger reframes (W6).
- Cite existing PR numbers where the brief named them (PR #29, PR #98, PR #355).
- Preserve all prior content verbatim — append-only.

## What W5 did

1. Read existing `TECH_DEBT.md` (1492 lines, ~100 KB) on branch `claude/unified-ogit-architecture-synthesis`. Verified the prior tail entry was `2026-05-07 — TTL-PROBE-5: dcterms:source dropped during TTL hydration`.
2. Appended a new `## 2026-05-07 — Unified OGIT Architecture: remaining wiring work (sprint-2)` section after the dcterms:source entry, opening with a one-paragraph blockquote orienting future readers to the 15-pattern crystallization and the W6 reframe convention.
3. Authored 11 TD rows in the standard shape used elsewhere in the file (Status / Priority / Region / Effort / Scope / Where / What / Plan reference / Dependencies). 289 inserted lines.
4. Committed locally on branch `claude/unified-ogit-architecture-synthesis`; pushed `52a0055..879b970`. Single-file commit, no other files touched in that commit.
5. Pushed this log in a follow-up commit on the same branch.

## TD row index (sorted by dependency depth)

| # | Row ID | Priority | Effort | Plan ref |
|---|--------|----------|--------|----------|
| 1 | TD-OGIT-G-SLOT-1 | P0 | medium ~300 LOC | unified-ogit-architecture-v1 Tier 1; W10 |
| 2 | TD-CONTEXT-BUNDLE-2 | P1 | small ~200 LOC | W10 |
| 3 | TD-GENERIC-BRIDGE-3 | P1 | medium ~200 LOC | W10 |
| 4 | TD-MANIFEST-MODULES-4 | P1 | medium ~330 LOC | W11 |
| 5 | TD-RACTOR-SUPERVISOR-5 | P1 | large ~400 LOC | W11 |
| 6 | TD-INT4-32D-ATOMS-6 | P2 | small ~120 LOC | unified-ogit Tier 3 |
| 7 | TD-CIRCULAR-COMPILATION-7 | P3 | large ~500-800 LOC | unified-ogit Tier 4 (aspirational) |
| 8 | TD-ANATOMY-DEMO-8 | P2 | very large multi-PR | W12 |
| 9 | TD-CAM-DIST-REGISTRATION-9 | P2 | trivial 1 line | unified-ogit Tier 0; W6 reframe |
| 10 | TD-ADJ-THINK-EXPOSE-10 | P2 | trivial ~30 LOC | unified-ogit Tier 0; W6 reframe |
| 11 | TD-DEEPNSM-NSM-COLLAPSE-11 | P2 | small ~30 LOC + 5 deletes | Recipe C; W6 reframe |

Dependency DAG (topo-sortable, no cycles):

```
TD-OGIT-G-SLOT-1 (foundation)
  ├── TD-CONTEXT-BUNDLE-2
  │     ├── TD-GENERIC-BRIDGE-3
  │     ├── TD-INT4-32D-ATOMS-6
  │     └── TD-RACTOR-SUPERVISOR-5 (also depends on #4)
  ├── TD-MANIFEST-MODULES-4
  │     └── TD-RACTOR-SUPERVISOR-5
  │           └── TD-CIRCULAR-COMPILATION-7 (also depends on #4, #6)
  └── TD-ANATOMY-DEMO-8 (depends on 1-5; fans in all P0/P1 rows)

TD-CAM-DIST-REGISTRATION-9   (independent)
TD-ADJ-THINK-EXPOSE-10        (independent; W6 reframe of ADJ-THINK-1)
TD-DEEPNSM-NSM-COLLAPSE-11    (independent; W6 reframe of DEEPNSM-NSM-1)
```

## Cross-references

- **W10 plan** (`.claude/plans/ogit-g-context-bundle-v1.md`): rows 1, 2, 3.
- **W11 plan** (`.claude/plans/compile-time-consumer-binding-v1.md`): rows 4, 5.
- **W12 plan** (`.claude/plans/anatomy-realtime-v1.md`): row 8.
- **W1 master plan** (`.claude/plans/unified-ogit-architecture-v1.md`): rows 1 (Tier 1), 6 (Tier 3), 7 (Tier 4), 9, 10 (Tier 0 quick-wins).
- **W6 entropy-ledger reframes** (`.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` rows `CAM-DIST-1`, `ADJ-THINK-1`, `DEEPNSM-NSM-1`): rows 9, 10, 11.
- **W8 INTEGRATION_PLANS index** (`.claude/board/sprint-log-2/agents/agent-W8.md`): cites the same TD-id set verbatim — cross-doc coherence verified before commit.
- **PRs** cited in row text: PR #29 (SmbMembraneGate), PR #98 (MedCareMembraneGate), PR #355 (D-ONTO-V5-9 / SpoBridge::promote_to_spo / lance-graph-ontology crate introduction).

## Findings (brutally honest)

- **Sixth manifest slot extrapolated:** the brief's arithmetic for TD-MANIFEST-MODULES-4 said "~30 LOC × 6 = 180 LOC" but listed only 5 manifest files (medcare, q2-cockpit, smb-office, dolce, fma). I added a "(sixth slot reserved for the first community-contributed module)" line rather than leave the math dangling. Small extrapolation; flagged.
- **Section context blockquote added beyond brief:** I opened the section with a `> Section context.` blockquote orienting future readers. The brief did not require it. Rationale: the file is 100 KB / 1492 lines deep; a reader landing on this section deserves orientation without having to grep up the chain. If meta-1 / meta-2 prefer strict brief-fidelity, the blockquote can be trimmed in a follow-up — it is informational, not structural, so removing it does not break any TD-id reference.
- **Recipe C location not verified:** the brief says "Recipe C in patterns.md". I wrote the cross-ref as `EPIPHANIES.md` patterns.md (Recipe C "collapse-parallel-impl-to-reexport") because EPIPHANIES.md is the canonical patterns hub in this workspace. The actual canonical file location (`patterns.md` vs `EPIPHANIES.md` vs both) was not verified at commit time. A later sweep should tighten this.
- **PR numbers taken as load-bearing from brief:** PR #29 (SMB), PR #98 (MedCare), PR #355 (D-ONTO-V5-9). I did not cross-check against `PR_ARC_INVENTORY.md`. W8's log notes the same — see `agent-W8.md` Finding #3.
- **Forward-references to sister plan-docs:** the four `.claude/plans/*.md` files cited (`unified-ogit-architecture-v1.md`, `ogit-g-context-bundle-v1.md`, `compile-time-consumer-binding-v1.md`, `anatomy-realtime-v1.md`) are produced by W1 / W10 / W11 / W12 in this same sprint and may not yet exist on the branch when this row set lands. That is expected by the sprint CCA2A protocol (parallel workers, append-only board), but means a reader on a partial checkout will see broken plan-doc links until the sister workers push.
- **`lance-graph-callcenter` crate path assumption:** rows 3, 5 assume the crate exists at `crates/lance-graph-callcenter/`. I did not verify this against the live workspace at commit time. If the crate is named differently or lives elsewhere, the "Where" fields need a follow-up edit.
- **Local INTEGRATION_PLANS.md modification untouched:** my working tree had a sibling-W8 mod to `.claude/board/INTEGRATION_PLANS.md` from a parallel agent. Left untouched and unstaged per scope-discipline; only `.claude/board/TECH_DEBT.md` was staged and pushed by my commit.

## Self-review checklist

- [x] 11 TD rows appended (TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11).
- [x] Each row has Title, Region, Severity-Effort, Where, What, Plan reference, Dependencies.
- [x] Cross-references to W10 / W11 / W12 plan-docs.
- [x] W6 entropy-ledger reframes called out on rows 9, 10, 11.
- [x] PR numbers (#29, #98, #355) referenced verbatim from brief.
- [x] Append-only: diff `@@ -1490,3 +1490,292 @@` confirms only additions after prior tail entry; no existing line altered.
- [x] Single-file commit (`879b970`); no other workspace files touched in that commit.
- [x] Dependency DAG between TD rows is internally consistent and acyclic.
- [x] Sprint-2 governance preserved (CCA2A append-only, log path under sprint-log-2/agents/).

## Out of scope (intentional non-deliverables)

- Did NOT touch any other board file. W1, W2, W3, W4, W6, W7, W8, W9, W10, W11, W12 own their own append entries.
- Did NOT create the sister plan-docs (`.claude/plans/*-v1.md`) — those are W1 / W10 / W11 / W12 deliverables. My rows forward-reference them per CCA2A protocol.
- Did NOT edit any existing TD row in `TECH_DEBT.md`. Append-only.
- Did NOT verify the `crates/lance-graph-callcenter/` crate name or PR numbers in `PR_ARC_INVENTORY.md` — taken from brief as load-bearing.
- Did NOT collapse or retire the `SmbMembraneGate` / `MedCareMembraneGate` wrappers — that work is the body of TD-GENERIC-BRIDGE-3 and a follow-up paid-debt entry, not a board edit.
