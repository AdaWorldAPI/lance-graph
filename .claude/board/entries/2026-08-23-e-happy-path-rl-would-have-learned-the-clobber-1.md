## 2026-08-23 — E-HAPPY-PATH-RL-WOULD-HAVE-LEARNED-THE-CLOBBER-1 — R2IL machine-state receipts are the falsification substrate the toy Phase-1 world could not provide

**Status:** FINDING — [MEASURED] (`PROBE-R2IL-FRONTIER-PHASE2-1`, 7/7).
**Phase 2 of 2** — Phase 1 is `E-THE-FRONTIER-LEARNER-IS-ALREADY-SHIPPED-1`
(the loop = shipped `revise` + `Stamp` + CHOICE, nothing new). This entry
lifts that loop onto R2IL-shaped typed behavior ops and measures what the
richer vocabulary BUYS.
**Confidence:** High for the mechanism; the machine is a 4-register toy and
both oracles are probe-local (stated) — no claim about real binaries until
real `FunctionBehavior` episode streams are measured.

**The headline (gate R6):** a "reckless" candidate macro computes the RIGHT
value into the WRONG register, clobbering callee-saved `r3`. The deliberately
sloppy happy-path oracle ("the doubled sum exists in SOME register") rewards
it until its NARS expectation reaches **e = 0.812 — ABOVE the 0.75 trust
bar** — and it is refused ONLY by the falsification intervention that checks
the actual contract (result in `r2` AND `r3` bit-preserved). Happy-path RL
would have learned the clobber. The falsification-first admission predicate
(`LearnedSurvivedTests`, #1011 F6) is not a nicety at the R2IL level — it is
the difference between a learned macro and a learned bug.

**Why R2IL is richer (gate R1, measured not asserted):** Phase 1's op
vocabulary was style-local labels; R2IL's `FactKind` discipline
(Op / OperandIn / OperandOut / Edge / MemUse / MemDef / Predicate /
CallSite, mirrored from `ruff_r2il` at ruff `origin/main`) gives every op a
typed operand signature over Varnodes (space→offset→size). That is what
makes BEFORE + TYPED EDIT = AFTER checkable at the MACHINE-STATE level
(gate R2: `(7+5)*2 = 24` lands in `r2`, `r3 = 0xDEAD` preserved,
byte-identical replay) — the #1001 typed-receipt law extended to behavior.

**The other gates:** R3 arms the trap (the sloppy signal raises BOTH
explorers' trust — reckless reaches e > 0.9); R4 the intervention admits
the lean macro and refuses the reckless one; R5 dispatch flips to the
cheaper PROVEN macro (2 ops); R7 fences with MEASURED sizes — `Vn` = 8 B,
`R2Op` = 28 B, `MachState` = 32 B. The R7 sizes are measured, not guessed:
the gate's first run FAILED on hand-guessed sizes (12/40), which is the
gate working.

**Honesty box:** toy 4-register machine; both oracles probe-local; the
R2IL shapes are a cited probe-local MIRROR (ruff is a separate cargo
workspace — never imported); real-corpus `FunctionBehavior` episode streams
are NAMED, NOT BUILT (absent from this checkout); the V4 plane classid
remains provisional (O5 gate); the widened R2IL × BPE / OGAR-loco / V4
synthesis stays a three-IF hypothesis — nothing here mints, reserves, or
decides it.

**Files:** `crates/lance-graph-planner/examples/probe_r2il_frontier_phase2.rs`.

