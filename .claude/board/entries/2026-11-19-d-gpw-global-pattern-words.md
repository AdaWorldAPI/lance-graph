# D-GPW-0: global pattern words, proposed (2026-11-19)

**Status:** PROPOSED / DOCS ONLY. The reading is this session's; the owner
has not confirmed it. Plan: `.claude/plans/global-pattern-words-v1.md`.

Claim: a pattern (`LIKE 'a.%'`, a spaCy token pattern, a MeTTa `match`) is one
global word with a hole. Within-a-value and across-values are its two
dimensions, and what a name with no member means is a declared policy, by
analogy with `Codec8<T, S, R, N>`'s `Rounding` and `NanPolicy`
(`ndarray/src/hpc/codec8.rs`, read).

MEASURED, in zammad-rs against PostgreSQL 16.15:

- 57 of 512 recorded SELECTs are Zammad's permission lookup,
  `name = ? OR name LIKE 'x.%'`.
- Answered as a text match (zammad-rs #41) and over a relation the intake
  declares (zammad-rs #45): both equal PostgreSQL on all 57.
- The two forms differ after the row `a` is deleted: PostgreSQL still returns
  `a.b.c` for `LIKE 'a.%'`; the relation form returns nothing. Checked on a
  scratch table.
- 53 distinct conditions with their constants for 18 shapes in that workload;
  the operators are `=`, `AND` and the fragment. A guard carries no call
  parameter (EXO-291).

VERIFIED-IN-CODE: `POLYFILL` has 17 words and none with a hole; `LIKE` and
`IN` have no position. `Logic::{TwoValued, Kleene}` is already an absent-case
declaration, for truth.

Prior art built on, not restated: `metta-polyfill-v1.md` (across dimension);
`d-cgmm-lazy-codegraph-fact-demand-v1.md:125` (a spaCy-like pipeline is an
analogy, not a second text representation).

Open: the four decisions in the plan's §5; whether the absent-case policy can
be a compile-time marker or must be a registered value; D-GPW-1's census has
not been run, so "three mechanisms" is counted for zammad-rs only.
