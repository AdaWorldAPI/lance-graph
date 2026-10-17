# 2026-10-17 — Epistemic frontier: what clicked

**Kind:** finding + decisions. **Status:** DECISION for O1 and O3 (BASIS:
the user's statements of 2026-10-17, quoted below); the rest is
WORKING-MODEL until a probe or a PR makes it VERIFIED-IN-CODE.
**Map:** `.claude/knowledge/epistemic-frontier-capstone.md` holds the full
tree with file and status per node; this entry records only what changed
today and why.

## Decisions

| # | DECISION | BASIS (user, 2026-10-17) | replaces |
|---|---|---|---|
| O1 | CE64 46..49 = activation i4: +7 excitation … 0 neutral … −8 inhibition. It is the signed reaction of the edge, never the operation that produced it | the 2026-10-17 bit diagram ("activation i4 +7 excitation … −8 inhibition") and "the operation lives in the running thought, never write it back into the result edge" | the inference-mantissa reading in `isa.rs` and the line "Bits 46..49 stay the inference mantissa; activation moves to a row field (D-CGW-9)" in `ce64-semantic-upper-half.md` |
| O3 | CE64 53..58 (W) = the thought layer this state belongs to: 0 = givens, 1..63 = hypotheses. W is the breadcrumb to the thought that changed the state | "The smartest thing would be to make it a register for the 64 thoughts"; "It gives the identity of the parallel universe it is in" | the corpus-root handle into `WitnessTable` (`mailbox_ref`, `spo_fact_ref`), which is a foreign key and is withdrawn |
| G | SPOG's G is never a reference between ontologies; evidence is projected as a foldable layer | "if 2 ontologies that use SPOG, don't wire G as a foreign key / The evidence is projected into the graph as a native foldable layer instead" | any design that joins graphs by a key |
| N | A thought is not capped at a NAL level; NAL levels stack across the 64 lanes | "why should we limit a thought that could progress in it's awareness to a higher NAL level / 64 thoughts can easily satisfy stacking 9 NAL layers" | my own earlier cap, withdrawn |

## Clicks (candidates for the Eureka gate)

Each passes the three-way test only if it still matters with the mistake
that led to it forgotten. All four do; promotion to `EPIPHANIES.md` is
blocked because that file is generated from `epiphanies-canon.json` and has
no route for a post-baseline entry (OPEN, see the 2026-10-17 one-layout
entry).

1. **The operation lives in the thought; the edge carries state and a
   breadcrumb.** The result is 59..63, the reaction is 46..49, the maturity
   is 50..52, and W says which reality wrote it. Nothing in the edge says
   *how* it was derived: that is the thought's program, held in its lane.
2. **Foreign keys dissolve into presence bits across 64 layers.** Whatever a
   reference would have carried (who witnessed, which graph, which evidence,
   who disagrees) is a bit in the per-row written-by word or a bit in a
   thought's mask, and every question about it is a popcount of an AND over
   one allocation. This is why W can be six bits and still be complete.
3. **Four ladders, two kinds of axis.** Rung (attention depth, per dispatch)
   and Pearl (the question, per edge) are fixed small ladders; maturity
   (50..52, per edge) and NAL level (per thought) are what grows. Maturity is
   the gear that opens a higher-level thought; it is not itself a NAL level.
4. **A thought = (strategy, front end), compiled once, in one lane.** The
   115 NAL capabilities and the front ends (nal, relalg, metta, cog,
   Narsese, Atomese) fill lanes; all lower to R2IL → mask-risc. Agreement
   between lanes, and between our engine and the oracles, is a signal read
   from the same bits.

## What this does NOT change

- Bits 59..63 stay one field (`EpistemicState5`).
- S/P/O stay palette codes.
- The alpha channel's rules stand: no fallback to the base, motion never
  writes the frame, discardable whole.
- Rung ≠ NAL level; the tunnel stays at 10 rungs.

## Follow-ups (each its own PR)

| D-id | work |
|---|---|
| D-HL-1 | migrate `isa.rs` opcode reading of 46..49 behind a declared "isa-v2" reading; fix PROBE-W-PRESERVE (`forward` / `syllogize` must not return W = 0) |
| D-HL-1 | wire W at merge / materialization from the layered view |
| — | `witness_table.rs` carries a VIOLATION banner (this PR); removal or rewrite later |
| O2 | 50..52: plasticity (`learn`) vs novelty → wisdom (`wire`) still needs a ruling |
