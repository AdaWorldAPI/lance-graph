## 2026-08-24 — E-R2IL-BPE-RECOMBINATION-FALSIFIERS-CONFIRMED-1 — the typed genetic recombination proposal's three §7 falsifiers all run green: splice points exist selectively (10.1%), recombination round-trips, and the counterfactual lane distinguishes at the real (v2-only) primitive level

**Status:** FINDING — [MEASURED] (`PROBE-R2IL-BPE-RECOMBINATION-FALSIFIERS-1`,
4/4 gates, autoattended: Sonnet worker built + verified, orchestrator
independently re-ran to bit-for-bit identical output before landing).
**Confidence:** High for the mechanism on this corpus (same 2 binaries,
143 episodes, 33 learned macros as the parent POC); F3 is explicitly
scoped to the SHIPPED v2 half of the counterfactual lane only.

Answers `.claude/plans/r2il-bpe-typed-genetic-recombination-v1.md` §7's
three named falsifiers, previously unrun:

- **F1 (splice legality):** of 1,056 ordered pairs among the 33 learned
  macros, **107 (10.1%)** admit ≥1 real type-legal splice point — a
  genuine observed def-use edge from macro A's tail site into macro B's
  head site, in the SAME episode (built from `extract_chains`'s own
  dataflow walk, not re-derived). 949 pairs admit none. Real def-use
  chains discriminate: neither uniformly entangled (splice never fires)
  nor uniformly permissive (splice always fires) — both would have been
  a weaker finding. Strongest legal pair `(7, 11)`, witnessed across 45
  disjoint episodes.
- **F2 (round-trip):** 5 sampled macros × {duplicate, delete, substitute}
  produced 10 genuinely distinguishable decoded sequences and 5
  correctly-silent identity substitutions (can-fire + can-stay-silent
  both hold). A corrupt-table falsifiability demo, mirroring the parent
  POC's B4 `corrupt_demo` exactly, confirms the decode machinery is a
  real falsifier here too. Every token in every recombination is a real
  R2IL atom id or a real learned macro id — never invented.
- **F3 (counterfactual-lane distinguishability), RE-SCOPED to the
  actually-shipped API:** `lance_graph_contract::counterfactual` has two
  staged halves — v2 (`deposit_counterfactual`,
  `FreeEnergyComparison::minority_wins()`) is REAL and shipped; v3
  (`CounterfactualMailbox`, `revise_if_minority_wins`, `awareness.revise`)
  is `todo!()`-stubbed, BLOCKED on D-PERSONA-5 (the ractor outer-swarm).
  This probe tests v2 ONLY — v3 is never instantiated or called (would
  panic). Result: an ordinary evidence-matched baseline pair (both
  minority and majority at the identical disjoint-episode count) correctly
  produces `minority_wins()=false`; F1's strongest recombined splice pair
  (45 pooled disjoint episodes) against the SAME weak majority produces
  `minority_wins()=true`. The verdicts differ — real signal, not washed
  out, at the primitive level that actually ships today.

**Two design refutations recorded honestly in the file, not adjusted
away (both re-verified by the orchestrator, present in the committed
comments exactly as reported):**
1. First F3 pairing used the globally-strongest macro (90 disjoint
   episodes) as the shared majority. BOTH scenarios lost
   (`minority_wins()=false/false`) — real refutation of that PAIRING
   (too strong a majority), not of the primitives: the raw
   `FreeEnergyComparison` values already showed a real ~36× gap
   (`f_minority_a=0.0524` vs `f_minority_b=0.0015`), just not crossing the
   boolean threshold against that majority.
2. Second attempt used the two globally-WEAKEST macros (1 vs 2 disjoint
   episodes) as the "evidence-matched" pair. It still flipped
   (`wins_a=true`) because `TruthValue::revise`'s curve is steep between
   N=1 and N=2 — "adjacent by rank" is not "evidence-matched." Traced to
   `revise`'s symmetric weighted-average math: only an EXACT
   disjoint-episode-count TIE produces identical `f` values, which is
   what the final, committed pairing uses.

**Process notes:** autoattended dispatch (one Sonnet worker, thoroughly
grounded brief pre-verifying the counterfactual API surface before
dispatch to avoid the worker hitting the BLOCKED v3 stubs); orchestrator
independently recompiled/re-ran to bit-for-bit identical output before
landing, per this session's standing rule of never trusting a worker's
self-report alone.

**Fences:** no mint performed anywhere; no write to MUL, the autopoiesis
triangle, or any `ValueTenant`; `CounterfactualMailbox`/
`revise_if_minority_wins` never called (would panic); this probe
generates falsifier EVIDENCE only, never an admission decision.

**Files:** `probe_r2il_bpe_recombination_falsifiers.rs`,
`.claude/plans/r2il-bpe-typed-genetic-recombination-v1.md` (§7 updated
in place with these results, status line corrected).

