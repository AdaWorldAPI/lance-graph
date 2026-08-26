## 2026-08-20 — E-A-LOCAL-DERIVATION-CANNOT-OVERRULE-A-MEASURED-COUNTEREXAMPLE-1 — and its twin: a counterexample cited past its own Boundaries section is the same failure with the sign flipped

**Status:** RULING (operator, cold-start recovery session after #973's
closure). **Confidence:** High — process rule, with two instances measured in
ONE session.

**Instance 1 (#973).** A locally correct derivation about one representation
(`NiblePath`'s 16-nibble ceiling) was promoted, via a dramatic finding title,
into an implied GLOBAL substrate conclusion — while three already-measured
counterexamples in the same repository showed the opposite pattern working.
None were consulted before the finding was written.

**Instance 2 (the recovery PR itself, caught by the operator).** The session
correcting instance 1 then (a) cited #875 as proving an *exact* WordNet
encoding when its own W5 gate reports ~255 leaves per cell, (b) called a
`4 × u16` tuple "absolute identity" when a merged doc states plainly that real
OBO ids exceed `u16`, (c) named a generic `(D,S,P,O)` structure `CausalLiteral`
while its own test used `TREATED_WITH`, and (d) called a lexicographic prefix an
"HHTL locality projection" with no consumer and no measurement. **Four unearned
claims inside the PR whose entire purpose was retracting one.**

**The rules that follow, for this and every future session:**

1. Before declaring the substrate "cannot" do something architectural, search
   `EPIPHANIES.md` and the relevant `.claude/knowledge/` / `docs/` for existing
   measurements FIRST. A local proof whose premise omits established substrate
   behaviour is not a discovery.
2. **A counterexample must be read to its Boundaries section, not its
   headline.** Citing a measured result past what it measured is the same
   error, inverted — and it is *easier* to commit while correcting someone
   else, because the counterexample feels like it is on your side.
3. Before minting any new absolute-address type, answer in writing: *what exact
   information cannot be expressed by the addressing that already exists?* No
   concrete falsifier ⇒ no new type. A slot in a plan is not a falsifier.
4. Name a type for what it structurally IS, not for the first use case that
   motivated it. If a test can substitute a non-causal predicate and the type
   still works, "Causal" does not belong in the name.
5. Use explicit `[MERGED]` / `[MEASURED]` / `[RULING]` / `[PROPOSED]` /
   `[REJECTED]` labels, and never blur "true of one type" into "true of the
   substrate" without saying so in the same sentence.

**Cross-ref:** `E-S3-0-NEEDED-NO-NEW-ADDRESS-1`,
`E-WORDNET-IS-A-LOCALITY-PRIOR-NOT-AN-IDENTITY-ENCODING-1`,
`E-NIBLEPATH-DEPTH-IS-NOT-HHTL-DIMENSIONALITY-1`, PR #973 (closed unmerged),
`.claude/handovers/2026-08-20-s3-0-cold-start-recovery-audit.md`.

---

