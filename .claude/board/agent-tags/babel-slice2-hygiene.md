# babel-slice2-hygiene — worker record

Sonnet grindwork worker, branch `claude/x265-x266-plans-review-h9osnl`.
Task: board hygiene for PROBE-BABEL-STANCES slice 2 (originally staged on
PR #861, which merged; the branch restarted from `origin/main` and the
slice-2 work now lives on **PR #862**, rewritten by the orchestrator; codex
P1/P2 findings resolved in
`crates/lance-graph-planner/examples/probe_babel_stances.rs`).

## Files touched

1. `.claude/board/EPIPHANIES.md`
   - In-place amended the existing (unmerged, orchestrator-authored) slice-1
     entry `E-THE-GRID-COLLAPSES-WHAT-A-LANGUAGE-SPLITS-1` — inserted a
     `> **⊘ RETRACTED (codex P1, 2026-07-28):**` blockquote directly after
     its Status/Confidence line, stating the two P1 findings (lemma-vs-synset
     coordinate error; en-kjv self-resultant pollution) and naming what
     survives unretracted (the graded-phase/LCS-radix fix, per-row
     VERIFIED/CHECK gating, the 3:7 convergence with `probe_eyes_opened` B2).
     Did NOT touch any other part of that entry's body — the false-friends
     2×2 and the `DIE` non-finding stand as written, per the brief.
   - PREPENDED a new entry at the very top of the file:
     `E-TWO-ROSETTA-STONES-AND-THE-FOUR-CHANNEL-SPLIT-1`, dated 2026-07-28.
     Covers: the two stones (corpus convergence grid + per-lane learning
     language stones); the four-channel phase split with the three measured
     readings (German three-channel / Latin silent / Czech morphology-only,
     the last one flagged explicitly as CHECK-row-dependent, report-only,
     never asserted); the translationese / coherent-antiphase pragmatic
     finding; the tracking census (only German tracks, sem coherence 0.413
     KNOW vs 1.000 NAKED) as the measured inversion of slice 1's headline;
     valency (frame vs lexeme, the distinction moves levels); passion
     peaking at (German, KNOW) = 0.3130, asserted; the two self-caught
     defects (MAD=0 degenerate branch, quale/confidence conflation on CHECK
     rows); the Finnish/suffix typological limitation (spine slot 6 left
     RESERVED); and the metric check (CLAM/helix vs Levenshtein, q-gram
     quantization caveat, slice-3 deferred to a trained codebook).

2. `.claude/board/STATUS_BOARD.md`
   - No existing PROBE-BABEL-STANCES row existed (grepped, confirmed empty).
     Added a new standalone section `## PROBE-BABEL-STANCES — two Rosetta
     stones + four-channel phase split (IN PR — slice 2, 2026-07-28)`
     immediately after the existing `## PROBE-EYES-OPENED` section, matching
     that section's exact header-table format (single-row `| D-id |
     Deliverable | Repo | Status | Evidence |` table). Status marked
     **IN PR** (not SHIPPED — the slice-2 work is open on PR #862, opened
     after #861 merged, per the brief), evidence cites both the new
     EPIPHANIES id and the retraction note inside the old one, plus the CI
     wiring fix (`rust-test.yml` now runs both probe examples).

3. `.claude/board/agent-tags/babel-slice2-hygiene.md` (this file) — created
   the `agent-tags/` directory (did not exist) and wrote this record.

## Did NOT touch

- `AGENT_LOG.md` — per the one-writer rule, not written.
- The probe source file itself (`examples/probe_babel_stances.rs`) — read
  only, not edited (already rewritten by the orchestrator per the brief).
- No `cargo` commands run (build/check/test/clippy/fmt) — edit-only per the
  guardrail preamble.
- No git operations (no commit/stage/push, no worktree).

## Discrepancies / things noticed but not acted on

- None found. The module doc header in `probe_babel_stances.rs` matched the
  brief's summary closely; spot-checked the measured numbers cited in the
  brief (passion assert `("de-luther1545", "KNOW")`, sem-coherence values,
  the CHECK-row quale conflation comment) directly against the source around
  lines 762–1300 and they line up with what the EPIPHANIES entry now states.
- `STATUS_BOARD.md` has no per-file append-only enforcement visible in this
  worktree's tooling reach (no settings.json check run), so the edit was
  made via the normal Edit tool per the brief's explicit deliverable list,
  not via `tee -a`. The repository's newest-first ordering rule is
  mandatory for `STATUS_BOARD.md` — there is no dashboard-vs-ledger
  exception — so the section was placed at the very top of the file, ahead
  of `## PROBE-EYES-OPENED`, not merely appended after it.
