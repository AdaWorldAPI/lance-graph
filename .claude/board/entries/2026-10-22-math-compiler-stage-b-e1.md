# 2026-10-22 — Proof-carrying math compiler: Stage B, E1 end to end (probe-local)

**Status:** MEASURED (lab). Crate `crates/egglog-r2il-probe/`, own workspace root. egglog = AdaWorldAPI/egglog `main` (public API only, unmodified). Coordination: EXO-50 /043. Continues `entries/2026-10-22-math-compiler-gap-matrix.md`; falsifier E1 from `.arxiv/2304.04332_egglog_equality_saturation_datalog_excerpt.md`; D-RFT-2 owns the plan.

**Shape (operator decision):** E1 is exact-integer symbolic substitution, `a = 2b, b = 2c, z = 2a ⇒ z = 8c`. It is not a population measure, so there is no SUM fold, no lane and no mask, and `contract::Measure` is unchanged. This supersedes the Stage A seam's "over a SUM fold" step.

| Hop | What runs | Owner |
|---|---|---|
| law admission | two probe laws, `MulAssoc` and `ConstFold`; the rule id carries the contract's `set_rules::Arith` domain. Exact admits both; Wrapping admits only `MulAssoc`; Checked and Ieee admit neither | probe (`laws.rs`) |
| saturation | egglog proofs mode, `(run arith 1)` stepped up to `fuel`, `Saturated` / `Incomplete` read from the step's own iteration report | egglog, cold |
| extraction | egglog's own: `prove-variants` on the goal, cheapest first under the default AST-size cost, non-target variables as `(Def String :cost 1000)`; the answer is the cheapest variant other than the goal's own name | egglog |
| proof | the variant's proof from `prove-variants`, checked by egglog's `check_proof` before and after simplification (`src/proofs/proof_extraction.rs:126-139`) | egglog |
| lowering | straight-line R2IL scalar calls (`push`, `INT_MULT`) through `FoldDialect` over empty planes, the `procs.rs` pattern; no new opcode | existing R2IL |
| oracle | i128 substitution over the original premises, no shared code | probe (`oracle.rs`) |

**Results (13 tests, release):**
- Exact: `(Mul (Num 8) (Var "c"))`, one `INT_MULT`, agreeing with the oracle for c ∈ {0, ±1, 7, -123456, i64::MAX/8, i64::MIN/8}.
- Checked: both laws are refused and the program is three `INT_MULT`s. The domain changes the program.
- Missing premise (`b = 2c` dropped):
  - no variant contains `(Num 8)` and the result is not closed;
  - lowering refuses `UnboundVar` (`a` or `b`; which one is egglog's cost tie-break, both cost 1002);
  - the oracle returns None;
  - egglog's `prove` of `z = 8c` fails, while the paired claim `z = 4·b` proves.
- Overflow counterexamples (c = i64::MAX/8 + 1, i64::MIN/8 − 1, i64::MAX): `Overflow` in both Exact and Checked, and the oracle's value lies outside i64. The boundary i64::MAX/8 passes.
- Wrapping: lowering refuses, because R2IL scalar arithmetic is checked, not wrapping.
- Determinism: three fresh e-graphs give an identical extraction, variant list and call list.
- Bounded fuel: the six-link chain is `Incomplete` at fuel 1 and `Saturated` at fuel 16.
- Traceroute: the hops are Premises, Laws, Saturation, Extracted, Proof, R2IL, Value, Oracle, Verdict.

**Disable runs (6, all red):**
- Checked admits `MulAssoc`;
- `Def` without its cost;
- Wrapping is lowered;
- `Overflow` is mapped away;
- the goal's own name is accepted as the answer;
- every step is read as saturated.

**Cost:** encode 2 µs; lower + execute 5 µs (median of 5, release). The egglog phase (saturation + `prove-variants`) is timed in `Compiled::timings.egglog` but not printed or reported here. It is above the workspace's microsecond budget, and whether cold, out-of-fold compile costs may be reported is an open operator decision (CLAUDE.md "no millisecond results").

**Corrections to Stage A:**
- "Independent proof check MISSING" overstated the gap. egglog already checks every proof it returns. What is missing is a check callable from outside egglog on a proof object. The i128 oracle covers answer correctness independently.
- No egglog change was needed.

**Cold cost (operator decision, 2026-10-22: report cold costs in microseconds, not gated):** egglog phase (setup + saturation + `prove-variants`) 21 084 / 26 875 / 28 793 µs over three runs, each a median of 5, release, Xeon @ 2.10 GHz, 4 vCPU, shared host. In the same runs encode was 3–4 µs and lower + execute 6–15 µs. `examples/e1_cold_cost.rs` now prints the egglog row. The cost line above is unchanged; it records the earlier state.

OPEN:
- E2–E5 beyond what E1 exercises;
- a lowering target for wrapping arithmetic (none exists; by design?);
- `Measure::Scale` stays gated on a real population-measure consumer in a separate PR.
