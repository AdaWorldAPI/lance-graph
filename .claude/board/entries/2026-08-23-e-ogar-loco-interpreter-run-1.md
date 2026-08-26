## 2026-08-23 — E-OGAR-LOCO-INTERPRETER-RUN-1 — the missing interpreter was built, ran, and a real ABI property surfaced

**Status:** FINDING (measured — `PROBE-LOCO-INTERPRETER-1`, pre-registered in
`.claude/brainstorms/2026-08-22-behavioral-ir-fathoming.md` §F, was actually
run). **Confidence:** High; reproducible from the commit.

`AdaWorldAPI/OGAR` branch `claude/probe-loco-interpreter-1` adds
`crates/ogar-loco/examples/interpret_probe.rs`: a minimal interpreter for
`ogar_loco::FunctionBody` over its shared computational core (arithmetic,
logic, variables, IF/IF_ELSE/REPEAT/WHILE/REPEAT_UNTIL). Corpus: four
hand-authored real algorithms (GCD, summation, FizzBuzz-style classification,
Collatz step counts), 44 real inputs total, each run twice.

Results: KC2 (determinism) PASS, KC3 (median episode length ≥5) PASS at
median=23, KC4 (traces are input-dependent, not `ladder_program()`'s static
ordering) PASS with 44 distinct sequences across 44 episodes — the single
most-likely-to-kill condition did not fire. All 44 episodes independently
correct against ground-truth arithmetic. KC1 (the 34 lance-graph-ogar recipes
have separable executable effects) explicitly NOT TESTED — the probe covers
only the shared core below `DOMAIN_FLOOR`.

**The genuine surprise was in building it, not in running it:** the shared
core's own declared `pushes_result` table marks `VAR_SET`/`VAR_CHANGE` as
*pushing* a result (chainable-assignment semantics), and there is no
`DROP`/`POP` primitive. `ogar_loco::statements::statement_bounds` (whole-body
segmentation, built for step-mask masking) therefore correctly *refuses*
(`DanglingOperands`) any ordinary imperative "set a; set b; …" sequence,
because nothing consumes the leftover pushed values. This is not a bug in
`statement_bounds` or in the probe — masking and execution are different
questions — but it means `statement_bounds` cannot be reused unmodified as a
learned-macro unit boundary, since it refuses on exactly the bodies real
programs are made of. The interpreter therefore does not use
`statement_bounds` for dispatch; it walks each function body as one linear
program-counter pass (treating `VAR_SET`/`VAR_CHANGE` as void), using a local
backward operand-span scan only where `WHILE`/`REPEAT_UNTIL` must re-run a
condition.

**Consequence for the fathoming report (`AdaWorldAPI/lance-graph` PR #989):**
§F was rewritten to §F1 with these results; §H ("if it fails") does not
apply — the falsifier survived. §G's sketch (learned macro as a `MacroId`
reference, deopt-shaped fallback, never a new opcode) is now reasoning about
a substrate with a confirmed-executing IR underneath it, still [H] pending
KC1.

**The generalization:** a crate's own declared semantic table (here,
`pushes_result`) can encode a design intent (chainable assignment) that no
consumer has ever exercised against a real multi-statement program — so its
correctness as a MASKING primitive and its usability as an EXECUTION
primitive are separate claims, and the first building of a real interpreter
is where the gap between them becomes visible.

**Status:** FINDING (measured — the mutation was run and initially did NOT
fire). **Confidence:** High; reproducible from the commit.
**Relation:** second instance, same day, of the family opened by
`E-A-DOC-PRECEDENCE-CLAIM-CAN-PASS-EIGHT-GREEN-TESTS-1` — but a **different
mechanism**, so it is filed separately rather than folded in. There the claim
was untested; here it was *false*, and the code it justified was *correct*.

`rubicon_witness::breadth_bits` computes `log2(1 + coverage(mask))`. The `1 +`
is load-bearing. Its doc comment explained why:

> the `1 +` is what keeps **empty** (`0.0`) distinct from **exact** (`1.0`) —
> without it both would read `0.0`

**That is wrong.** Without the offset, empty reads `log2(0) = -inf` and exact
reads `log2(1) = 0.0` — already distinct. The assertion written from that
explanation (`breadth_bits(empty) < breadth_bits(exact)`) therefore passes
**with or without** the guard, because `-inf < 0.0` holds.

The offset's real job is **finiteness**: `-inf` in a mean carries the whole
[`FocusTrace::breadth`] with it, so a single unsampled moment would make an
entire phase read as infinitely narrow. Correct guard, false reason, and a test
that tested the reason.

Exposed by a mutation that removed the `1 +` and watched **all eight tests stay
green**. The fix was to correct the comment in place (with the error recorded,
not deleted) and add the assertion the real property needs: a trace holding one
real focus and one empty sample must report a **finite** breadth, strictly
between the two.

**The generalization:**

> A doc comment does not only make CLAIMS that need testing — it supplies the
> REASON a test author writes their assertion from. A false reason yields a
> true-but-inert assertion, and inert assertions are invisible: they pass, they
> look like coverage, and they sit next to a guard they do not guard.

Practical consequence: when a guard is small and "obviously" necessary
(`1 +`, `saturating_`, a `+ 1`, a clamp), the mutation to run is **deleting the
guard**, not exercising its stated purpose. If nothing goes red, the stated
purpose is not the actual purpose — and the comment is the first suspect, not
the last.

Both instances this session share one method note: the finding only exists
because the mutation was **executed**. Neither would have survived a reading.

