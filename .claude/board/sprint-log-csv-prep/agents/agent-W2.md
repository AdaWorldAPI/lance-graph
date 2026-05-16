# Agent W2 Scratchpad — Sprint-10 Specs Patch (CSV Prep)

**Date:** 2026-05-16
**Branch:** `claude/sprint-10-specs-patch-csv-prep`
**Target spec:** `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md`
**Plan anchor:** `.claude/plans/cognitive-substrate-convergence-v1.md` §5 (20 locked decisions), §6 (v2 bit layout), §12 (W2 patch row)

---

## Work Done

Applied ~150 LOC delta to `pr-ce64-mb-2-causaledge64-v2.md` across 11 patch operations:

### 1. OQ-LAYOUT-1 RESOLVED everywhere

- **§0 header callout** (already present in prior partial patch): confirms Option F ratified
- **§11 Open Questions**: replaced old `OQ-LAYOUT-1 (BLOCKER)` text with `OQ-LAYOUT-1 (RESOLVED 2026-05-16)` citing plan §6, L-2, L-3, L-4, L-6, L-7
- **§10 Risk Matrix**: replaced stale "Option C not ratified / MED likelihood" row with "Option F resolved / RESOLVED" row
- **§12 DELTA item 1**: updated to describe Option F rather than Option C
- **§12 DELTA item 8** (new): explicit note that Counterfactual = causal_mask, NOT a separate bit

### 2. §"Signed Mantissa Rationale" added (~50 lines, inserted before §3)

- **Why sign = direction**: full table showing |mantissa| → rule name in +/− directions
- **8 base slots**: Identity(0), Deduction(1), Induction(2), Exemplification(3), Revision+(4), Synthesis(5), Reserved5/6 (absorbs PR-LL-1 Intervention/Counterfactual), Extension(7)
- **PR-LL-1 absorption**: `+6` = Intervention, `−6` = Counterfactual — zero new bits needed
- **Three SIMD wins**: Win1 signum/abs free (arithmetic-right-shift), Win2 palette×mantissa stays i4/i8 family, Win3 8ch→SPO transcode is near-bitcast via `net_strength.signum()`

### 3. §"Counterfactual via causal_mask, NOT via separate bit" added (~20 lines, inserted before §3)

- Full causal_mask 8-state table: 0b000..0b111 → Pearl rung 0..3
- 0b111 SPO = Pearl-3 Counterfactual by construction — no extra bit needed
- Three arguments against a separate Pearl-3 modifier: redundant, inconsistent, wasteful
- Orthogonality: causal_mask = WHICH rung; mantissa = WHICH rule AT that rung
- Example: `causal_mask=0b111` + `mantissa=−6` = Pearl-3 counterfactual via PR-LL-1 Counterfactual rule

### 4. §4 Accessor sketches updated (G-slot removed, signed mantissa added)

- Removed: `g_slot()`, `with_g_slot()`, `set_g_slot()` and all G_SHIFT/BITS5_MASK/G_MASK references
- Added: `inference_mantissa() -> i8`, `inference_direction() -> i8`, `inference_rule_index() -> u8`
- Updated: `with_routing(w, t)` — no `g` parameter (G-slot absent per L-3)
- Updated: v1 stub block to match

### 5. Supporting sections cleaned up

- **§5 feature flag comment**: replaced Option C reference with Option F description
- **§6.1 PAL8 byte analysis**: removed G-slot row, updated bit positions to Option F layout
- **§7 per-method semantics**: replaced g_slot section with w_slot + inference_mantissa sections
- **§13 coordination notes**: W4/W6 notes updated to reflect no G-slot; meta-reviewer note updated

---

## Key Design Decisions Captured

| Decision | Location in plan | Location in spec |
|---|---|---|
| Drop temporal(12b) | L-2 | §2, §3 const V1_TEMPORAL_SHIFT |
| Drop G-slot entirely | L-3 | §0, §1, §4 (removed), §12 item 1 |
| 4b signed i4 mantissa | L-4 | §2, §3, §4, §"Signed Mantissa Rationale" |
| causal_mask 0b111 IS Pearl-3 | L-5 | §"Counterfactual via causal_mask" |
| W-slot 6b corpus handle | L-6 | §2, §3 W_SHIFT=53 |
| Truth-band lens 2b | L-7 | §2, §3 TRUTH_SHIFT=59 |
| Reserved5/6 absorb PR-LL-1 | L-9 | §"Signed Mantissa Rationale" PR-LL-1 absorption |

---

## Files Modified

- `/home/user/lance-graph/.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` — 874 lines, 43671 chars (from ~712 lines)

---

## No Commits, No Pushes, No Branch Switches

Per task instructions: applied edits only, wrote scratchpad. Branch remains `claude/sprint-10-specs-patch-csv-prep`.
