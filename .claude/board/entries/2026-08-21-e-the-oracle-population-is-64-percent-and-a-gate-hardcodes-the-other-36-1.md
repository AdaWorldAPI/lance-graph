## 2026-08-21 — E-THE-ORACLE-POPULATION-IS-64-PERCENT-AND-A-GATE-HARDCODES-THE-OTHER-36-1 — a third of the "known intermediates" name no intermediate, and the gate that would have caught it asserts the wrong number

**Status:** FINDING (measured three independent ways on `/workspace/dismech`
@`557e15436`, 1,968 disorder files). **Confidence:** High — the three methods
agree exactly.

The DisMech supervision story rests on `INDIRECT_KNOWN_INTERMEDIATES` being a
population where "the source names the mediators, so they can be hidden and
recovery measured" (`E-DISMECH-CORPUS-CENSUS-1`). Measured, **only 2,449 of
3,825 (64.0%) such edges actually name one.** 1,376 (36.0%) carry the label and
an absent or empty mediator list.

Three methods, agreeing: (a) a line-oriented per-edge walk — **2,449** edges
over **534** disorder files, 3,714 mediator strings; (b) key-occurrence count —
`intermediate_mechanisms:` appears **2,525** times with **0** inline empty
lists, and 2,525 = 2,449 + 74 + 1 + 1, exactly the per-bucket split;
(c) `contract::dismech_evidence.rs:155-176` at the narrower
`pathophysiology[].downstream[]` scope — **1,347 of 3,844 (35.0%)** empty.

**Consequences.** The held-out corpus is **2,449 edges over 534 diseases**, so
a 20% split is ~490 edges / ~107 diseases, not the 774 a 3,869 denominator
implies. And **74 `INDIRECT_UNKNOWN_INTERMEDIATES` edges DO name mediators** —
a source contradiction the four-label taxonomy does not anticipate; they must
be removed from any restraint control or they will read as hallucinated
closure by the benchmark's own definition.

**The gate that should have caught this hardcodes the wrong number.**
`MedCare-rs/.claude/plans/dismech-missing-links-v1.md` Gate W1.1 asserts
`known_links.tsv == 3.869` — unsatisfiable on any corpus revision. Its own
instruction is *"stoppen und melden, nicht die Zahl anpassen"*, so this entry
is the report, not an edit.

**Transferable:** a label that asserts a property and a field that carries it
are two different measurements. Counting the label is not counting the data —
and a supervision corpus sized from the label is oversized by exactly the rows
where the source labelled but did not fill.

Cross-ref: `.claude/plans/dismech-causality-v3-v1.md` §3a; `E-DISMECH-CORPUS-CENSUS-1`.
