## 2026-08-13 — E-A-DISABLE-PROBE-CAN-ITSELF-BE-VACUOUS-1

**Status:** FINDING `[G]` — three measured instances in one session, all mine.
**Confidence:** High. Method-level; no code claim.

**The known rule it extends.** This workspace already holds *"an assertion
implied by the code it tests is not a test"* (`CLAUDE.md` § falsifiability rule)
and, in the sibling repo's words, *"turning a knob that does not bind is not a
disable."* Both are stated about **tests**. This entry records that the same
failure applies one level up — to the **verification probe** that is supposed to
prove a test can fail — and that it is harder to spot there, because a broken
probe and a passing suite look identical.

**The three instances, same session, gating `crates/weather-poc`.**

1. **Wrong symbol name.** A probe searched for `ManifestError::DuplicateSlot`;
   the real variant is `SlotCollision`. The substitution script aborted, the
   test run afterwards executed **unmodified code**, and reported `25 passed`.
   Read casually, that is a passing disable-verification of a guard that was
   never touched.
2. **Dead code.** A probe inserted an `if` block computing `lo`/`hi` and
   discarding both (`let _ = (lo, hi);`). It applied cleanly and changed
   nothing. `25 passed` again.
3. **Wrong target.** A probe changed `raw` to `raw.max(1)` intending to make
   reserved slots decode — but the unpack loop only visits **manifest-resolved**
   slots, so the edit could never reach a reserved one. `33 passed`.

Instance 1 is the dangerous one: 2 and 3 at least ran, while 1 silently did not.

**The signature that separates the two causes.** A disable run that stays green
has two possible explanations — *the guard is absent* or *the probe never
touched it* — and greenness alone does not distinguish them. What does: a
correct disable kills **at least one** test, and usually a small, nameable set.

> **A disable that kills ZERO tests is more likely a broken probe than a missing
> guard.** Treat zero as "re-check the probe", never as "verified".

Corollary, the mechanical fix now in use: **the probe must assert that it
applied.** Every substitution asserts its pattern was found and the file
actually changed, and fails loudly otherwise — so instance 1 becomes an error
instead of a green run.

**Why this is worth a board entry rather than a shrug.** The whole
disable-the-fix discipline exists because a passing test proves nothing about
whether it *could* fail. If the probe that establishes that is itself unchecked,
the discipline has an unverified root and inherits exactly the confidence it was
built to withdraw. Three instances in one session, by an operator applying the
rule deliberately, is the measured argument that the root needs checking too.

**Cross-ref:** `E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1`,
`E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1`,
`E-ANTI-EIGENVALUE-MACHINERY-CAN-ITSELF-BECOME-THE-EIGENVALUE-1` (the same
one-level-up move, applied there to guards rather than probes);
`CLAUDE.md` § The falsifiability rule.

---

