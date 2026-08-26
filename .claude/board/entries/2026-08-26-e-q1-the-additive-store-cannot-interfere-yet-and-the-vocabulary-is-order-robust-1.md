## 2026-08-26 — E-Q1-THE-ADDITIVE-STORE-CANNOT-INTERFERE-YET-AND-THE-VOCABULARY-IS-ORDER-ROBUST-1 — first plasticity falsifier run: INT is a CONTROL result, ORD passes at 0.872, SAT survives both naive policies, sabotage validates the harness

**Status:** FINDING [MEASURED] — every number below is from ONE
instrumented run of `probe_bpe_r2il_loco_microcode` over the four-binary
ore (`ore_all.tsv`, 94,536 rows; A = 2×gcc `stress_test` train,
H_A = `vuln_test` held-out C, B = rustc `build-script-build`), executed
AFTER the pre-registration was committed (`7b00847`, plan §8.3 — the
commit order is the proof of pre-registration). Instrument reverted
after the run; the probe on this branch is byte-identical to main.
**Confidence:** High for what is claimed; the entry's own headline is a
NEGATIVE boundary ("plasticity NOT yet demonstrated") and must not be
quietly upgraded.

### The run, verbatim

```
Q1  corpus split: A=1872 B=8659 H_A=608
Q1  BASE  V_A macros=33 density(H_A)=2.515 byte_exact=true
Q1  INT   V_A->B: new_from_B=4 total=37 density(H_A)=2.554 byte_exact=true
Q1  ORD   V_B->A: macros=36 density(H_A)=2.714; Jaccard(A->B vs B->A)=0.872 (34/39) byte_exact=true
Q1  SAT   cap=16: REFUSE density(H_A)=2.222; EVICT-AMORTIZED keeps 15/16 A-macros, density(H_A)=2.365; refuse_byte_exact=true
Q1  SABO  -top5 density=0.766; -bottom5 density=2.510
```

### Each arm against its pre-registered threshold (plan §8.3)

- **SABOTAGE first (harness validity):** can-fire half 0.766 < 1.8 ✓;
  silence half 2.510 within 2% of 2.515 ✓. The harness discriminates.
  All other arms count.
- **INT — CONTROL result, as pre-registered.** 2.554 ≥ 2.515 (no drop;
  the >1% drop that would have falsified additivity did not occur, and
  continuing on B even ADDED 4 macros that fire on unseen C). Verbatim
  per the pre-registration: *no destructive interference is possible yet
  because no destructive mechanism exists.* This is NOT a plasticity
  success — the store has never been given a way to forget, so "did not
  forget" is a property of the architecture, not of a learning rule.
  **Association is demonstrated. Plasticity is NOT.**
- **ORD — order-ROBUST at 0.872 ≥ 0.8.** A→B and B→A share 34 of 39
  atom-patterns. The vocabulary is substantially path-independent — it
  is a property of the corpus, not of the presentation order. (B→A's
  higher H_A density 2.714 is unsurprising: A's merges land last and
  freshest under B→A; not a pre-registered quantity, noted only.)
- **SAT — both naive policies clear the 2.03 kill.** REFUSE 2.222,
  EVICT-AMORTIZED 2.365 — and evict WINS while keeping 15/16 A-macros,
  i.e. amortized-occurrence eviction discarded almost nothing that
  mattered on held-out A. Saturation is where real interference first
  becomes POSSIBLE; at cap 16 neither naive policy degrades the store
  to the column-null floor. The interesting regime (caps low enough
  that eviction must cut into the top-5 that carry 69.5% of H_A hits)
  is Q6/hex territory, not this run.
- **BYTE-EXACT — true in every arm** (decode == original atoms for
  every training stream under every table, including the evicted one on
  the REFUSE side). R2IL remains the immutable atom layer throughout;
  BPE never became a second truth.

### What this licenses and what it does not

Licensed: "the additive macro store learns a second corpus without
degrading the first, order-robustly, with exact provenance" — measured.
NOT licensed: any claim containing "plasticity", "forgetting",
"consolidation", or "interference resistance" — the mechanisms those
words name do not exist in the store yet, and Q1's own INT arm is the
record of that absence. Q6 (hex A/B) is now UNGATED by these results:
its falsifiable question — same learning, LESS interference under a
mechanism that CAN interfere — is exactly the regime Q1 shows the
current store has not yet entered.

Cross-ref: plan `.claude/plans/r2il-machine-semantic-contract-v1.md`
§8 (pre-registration + queue); E-THE-QA-MACHINERY-IS-THE-LEARNING-RULE-1
(the reframing that generated these falsifiers);
E-R2IL-MACRO-VOCABULARY-TRANSFERS-ACROSS-COMPILER-AND-LANGUAGE-1 (the
baseline densities and nulls the thresholds were derived from).

