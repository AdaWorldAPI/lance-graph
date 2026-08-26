## 2026-08-12 — E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1

**Status:** FINDING `[G]` — both instances measured this session, committed
scripts + JSONs (`sunflower_pairing_probe`, `golden_vs_tempered_probe`).

**The claim.** The falsifiability rule's can-it-fire / can-it-stay-silent
doctrine applies to CONTROLS with exactly the same force as to guards: **a
control arm that cannot lose by construction carries zero information when
it wins** — and this session produced the shape twice in one afternoon, in
two independent probes, caught two different ways.

**Instance 1 — the symmetric-grid control (W2s-a, caught by a smoke test
BEFORE the full run).** The pre-registered G2 bar expected golden's
nearest-pair-distance CV to beat an axis-aligned-grid control's. Measured:
grid CV ≈ **1.6e-12** vs golden's 0.368 — twelve orders of magnitude, at
every N in the sweep, invariant under four different center offsets (0 to
300 km, checked in a 50k-point smoke test before committing to the full
2.55M run). Mechanism, not mystery: the brief specified TWO IDENTICALLY
CONSTRUCTED grids differing only by a pure translation — and two identical
periodic tilings offset by a fixed vector are, by lattice symmetry,
**translation-invariant in their cross-nearest-neighbour distance**. Every
point sees the same local geometry; the CV is floating-point noise. **The
control cannot lose ANY evenness comparison against ANY irregular
construction — so G2's FAIL verdict is a fact about the control, not about
the golden lattice.** Reported as FAIL + diagnosis rather than silently
redesigning the control to force the pre-registered answer (the run was
committed as-specified after the smoke test flagged it, deliberately, so
the record shows the specified control failing rather than a quiet swap).

**Instance 2 — the first-crossing m* (golden-vs-tempered T1, caught by
external review of the RUN's own result, codex P1 on #935).** The
crossover point "where golden permanently overtakes tempered" was computed
as a FIRST crossing — but golden's raw discrepancy sequence is
non-monotonic (only its `O(log m/m)` ENVELOPE is a bound), so a first dip
below the ceiling can reverse. Reproduced exactly: q=17 reported m*=21;
D*(22)=0.081 sits back ABOVE the frozen ceiling 0.059. The "permanent"
claim had no machinery that could catch a reversal — **an implicit
control (nothing rose back above) that was never actually checked**.
Fixed: `verified_permanent_crossover` checks a sampled suffix (every
integer for 50 steps + geometric points to the 200q horizon) and REPORTS
THE CHECKPOINT COUNT beside every m*, so the verification scope is stated
rather than implied exhaustive. The verified m* moved from ~1.0–1.4×q to
**~1.9–2.7×q** — the number's THIRD revision, each widening, each from a
methodological gap the previous pass could not see.

**The rule, in one line each:**
- *A control must be able to lose.* Before pre-registering a control arm,
  ask what result would make the control WIN unfairly — symmetry,
  degeneracy, and construction-identity are the usual culprits (two
  identical tilings, a rotated referent with the same marginals, a
  permutation that preserves the tested statistic).
- *"Permanently" is a claim about a suffix, not a point.* Any
  "stays/never/always thereafter" assertion over a non-monotonic sequence
  needs suffix verification with a STATED scope (checkpoint count), never
  a first-hit search.
- *Smoke-test the control's losability cheaply before paying for the full
  run* — instance 1 cost 0.1 s at N=50k to diagnose what would otherwise
  have surfaced only as a confusing FAIL after the 2.55M-point run.

**Cross-refs.** The falsifiability rule (CLAUDE.md P0) — this extends its
guard-doctrine to control arms. `E-THE-CONTROL-SCORED-THE-HEADLINE-1` — the
complementary failure (a control that scored AS WELL as the signal,
exposing the instrument); today's is the inverse (a control that cannot
lose, exposing nothing). W5's B3 control history — three successive control
designs (12/18 wrong-scale, 1500/2600 wrong-scale-in-disguise,
distance-matched-neighbour) are the same lesson approached from the
wrong-scale side. Plan homes: `weather-w-probes-v1.md` §2 RUN note,
`golden-vs-tempered-stride-v1.md` T1 correction notes.

