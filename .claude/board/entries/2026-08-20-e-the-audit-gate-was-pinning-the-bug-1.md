## 2026-08-20 — E-THE-AUDIT-GATE-WAS-PINNING-THE-BUG-1

**Status:** FINDING (measured; `examples/recipe_claim_audit.rs`, branch
`claude/carve-nars-kernels`). Sibling to
`E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1` — a *new* failure shape, not a
restatement.

Four of the 34 NARS recipe kernels were carved from non-production to
production (CAS 8, ETD 22, ICR 31, SDD 32). The moment they were, the repo's
own census example went **RED** — and the gate it failed was `G2 recipe-weak
set is exactly {CAS,ETD}+{ARE,ZCF,ICR,HKF}`.

**The audit's arms were written to confirm the defect, not to detect it.**
`8 =>` asserted `unchanged` (candidates, rung and Δconf all untouched) and
reported *"abstraction level computed then discarded — no observable
effect"*. That is a correct description of a bug, encoded as an expectation,
behind an equality gate. Fixing the bug is what breaks the suite.

This is NOT the vacuous-assertion failure (a test that cannot fail). Every
arm here CAN fail and did. It is the inverse: **an assertion that is real,
two-sided, and pointed at the wrong side of the finding.** A census that
pins "the weak set is exactly W" is only a census while W is a measurement;
the day someone shrinks W, the pin argues against them.

**What generalises.** A test that records a defect must say, in the test, that
it is recording a defect and what its removal should look like — the
`f_ord_real_defect_pin_*` convention (`E-…-FALSIFIER`) does exactly this and
is the shape to copy. The four arms are now re-pinned to the CLAIMED
POST-CONDITION instead (`realize(c1.candidates == vec![0.0, 1.0] …)`), so a
regression to compute-then-discard fails them again — the pin now points the
other way and the equality in G2 was kept deliberately (`n_inert == 0 &&
n_constant == 3`), so a NEW inert kernel is a review, not an absorption.

**Second, smaller finding in the same arm.** ICR sat in the shared
`19 | 24 | 31 | 34` "input-independent" arm, which varies only `candidates`.
ICR now reads `free_energy`, which that probe never varies — so it would have
reported CONSTANT forever while the kernel discriminated. *The fixture's
shape is part of the coverage*, again, and this time on a probe rather than a
test.

---

