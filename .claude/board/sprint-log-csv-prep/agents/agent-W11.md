# Agent W11 — Sprint-10 Test Plan v2 Patch

**Worker:** W11 (re-dispatch)
**Branch:** `claude/sprint-10-specs-patch-csv-prep`
**Date:** 2026-05-16
**Status:** COMPLETE

## Scope

Patched `.claude/specs/sprint-10-test-plan.md` with ~40 LOC of v2 substrate additions per `cognitive-substrate-convergence-v1.md` §12 W11 row.

## Changes Applied

### §3.A — v2 Substrate Additions (~25 lines)
Added new subsection enumerating test deltas beyond v1 baseline:
- W3: +5 (mantissa-signed-±, lens-4-state, w-slot-64, temporal-absent, product-range)
- W5: +12 (WitnessCorpus insert/query/cam_pq_search, chain-order iron rule, decay, anchor)
- W4: +3 (BindSpace phase 5a/5b QualiaColumn switch)
- W6: +8 (MailboxSoA SoA semantics, apply_edges, CollapseGateEmission)
- W7: +30 (SigmaTierRouter dispatch, Rubicon-resonance, integer-SIMD MUL)
- Total v2 delta: +58 new tests

### §4.3.1 — Miri growth-target refresh (~10 lines)
- Revised Miri target: ~1550 (v1) → ~1600 (v2, +58 delta)
- Called out `unsafe` SIMD signum/abs paths (W2 mantissa) needing Miri coverage
- Called out `WitnessCorpus Arc::make_mut` copy-on-write needing Miri test

### §3.B — Cross-references (~5 lines)
- Plan §5 L-1..L-20, §6, §11 D-CSV-1..D-CSV-12, §12 worker rows W2..W11
- Knowledge docs: `spo-schema-and-mailbox-sidecar.md`, `splat-shader-rayon-struct-method-vision.md`

## Files Modified
- `/home/user/lance-graph/.claude/specs/sprint-10-test-plan.md` (governance/markdown only)

## Files NOT Modified
- No Cargo files touched
- No settings.json / settings.local.json touched
- No branch switch, commit, or push performed
