## 2026-08-12 — E-A-HORSE-RACE-IS-NOT-A-CROSS-SWAP-1

**Status:** FINDING `[H]` — methodological, operator-ruled. Caught on
`substrate-comfort-zones-v1` BEFORE any bar ran, so nothing measured had to
be reinterpreted. Grade is `[H]` because the reasoning is airtight given the
premise but the premise itself ("the model captures the phenomenon") is an
assumption this plan does not test.

**The defect, in one line: when the premise is *the model captures the
phenomenon but is not calibrated*, an error metric measured under a
deliberately wrong calibration is bad BY DEFINITION, so scoring it answers
nothing.** The first draft of that plan compared four calibration arms on
reconstruction RMSE — a horse race. But `CAL-ABS-FOREIGN` is not a
competitor that might lose; it is the measurement CONDITION the whole
hypothesis is about. Racing it guarantees the answer and measures the
tautology.

**What the design must be instead.** Miscalibration is a *transfer*
question, so the instrument is a transfer matrix, not a ranking: `M[D][T]` =
read regime `T`'s field through a codebook derived from regime `D`. The
diagonal is own-calibration; every off-diagonal cell is a swap. The primary
metric becomes **structure preservation** (Spearman `ρ` of reconstructed vs
true), and the quantity the hypothesis actually concerns is the derived
**transfer loss** `L[D][T] = ρ[T][T] − ρ[D][T]` — how much ordering survives
being read through the wrong calibration. Absolute error is retained as
*evidence that the swap genuinely hurt*, never as the verdict.

**The sharp consequence that only appears in the matrix framing.** A
window-local encoding (rank-normalised, Fisher-z) has **no donor at all**,
so its matrix row is flat and `L ≡ 0` **by construction**. That is not an
artifact to hide — it IS the property under test: a substrate that carries
no absolute anchor cannot have that anchor be wrong. Which reframes the
whole comparison. The question stops being "which formula wins" and becomes
"in which regime does the anchor-free encoding's own-window `ρ` exceed the
absolute encoding's, measured against the DIAGONAL (its own calibration,
the hardest available opponent) rather than against the swapped cell (which
it beats almost by definition)."

**And the degeneracy must be VERIFIED both ways** — this is the
`E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1` lesson in its can-it-DIFFER
form, which W5 already paid for once. A flat row is only evidence if the
identical code path is proven able to produce a NON-flat one (the absolute
arm must show real off-diagonal degradation). Otherwise "flat" means the
harness cannot vary anything, and the encoding was never tested.

**Second half of the ruling, on the regime ladder itself.** The operator's
framing: *in science you hold variables constant to test the others;
constancy is relative, so the design deliberately manufactures strong
correlation differences — on the assumption that those differences are fit
to evaluate the hypothesis.* Three things follow, and the plan had only the
first:

1. **The 9.3× `|∇p|` spread across the four regimes is the design's PURPOSE,
   not a by-product of box selection.** At small spread every effect is in
   the noise; at large spread it is legible or it does not exist.
2. **"Constant" is relative, so it is MEASURED** — within-box spread of the
   discriminator must be small relative to the between-box spread
   (`separation ≥ 3`). Below that, "held constant" is a label on a box, not
   a condition, and the caveat must ride every downstream number.
3. **"Fit to evaluate the hypothesis" is an ASSUMPTION and therefore needs a
   falsifier.** The ladder is built on a pressure-gradient discriminator but
   the hypothesis is about correlation STRUCTURE. If two regimes are
   indistinguishable in autocorrelation decay and rank-distribution shape,
   they are ONE condition for this question however far apart their
   gradients sit — and the ladder would be measuring four copies of the same
   thing. A null there VOIDS the reading rather than weakening it: it would
   mean the spread was manufactured on the wrong axis.

**The defect is STICKY — it survived the rewrite built to remove it, and a
REVIEWER caught it, not the author (CodeRabbit on #944, same day).** §2.3's
summary passage still named the *foreign-donor* comparison as "the
operator's hypothesis" while C4, `STATUS_BOARD` and the `INTEGRATION_PLANS`
entry all correctly scored against the diagonal. Mechanism: §2.3 was written
before C4 was sharpened, and the sharpening was not propagated backwards.
Left standing, the plan would have carried **two incompatible verdict
criteria with the weaker one holding the headline** — the swapped cell
establishing merit, i.e. the exact tautology the rewrite existed to delete.

That makes this the **fifth** instance in this arc of a claim that is
internally consistent with its own operands but inconsistent with a SIBLING
claim (#930's fused relation, #927's decimal claim, #928's audit figures,
#941's dropped qualifier, now this). The generalization is sharper than the
individual fixes: **when a design is revised, the summary sentences are the
last thing to be updated and the first thing a reader believes.** A revision
is not complete when the new machinery is right; it is complete when every
sentence that *names the conclusion* has been re-derived from the new
machinery. Self-review kept missing it because the author re-reads the part
that changed, not the part that merely still sounds right.

**The transferable rule.** Before scoring arms against each other, ask which
of them is a *competitor* and which is a *condition*. A condition put on the
starting line will always lose, and its loss carries no information. Score
conditions by what SURVIVES them, not by how much they cost.

**Cross-ref:** `.claude/plans/substrate-comfort-zones-v1.md` §2 (the rebuilt
instrument, with v1's framing preserved in place as a correction note rather
than deleted) and §3 C1b/C1c/C2/C3/C4; `STATUS_BOARD.md` D-CZ-* (rows re-cut
while still Queued, with the old→new mapping recorded);
`E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1`;
`E-R²-IS-NEAR-BLIND` (why the physical unit is retained alongside `ρ`).

