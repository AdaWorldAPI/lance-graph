## 2026-08-31 — E-Q8-THE-SIX-DOES-NO-WORK-A-DEGREE-ABLATION-COLLAPSES-THE-HEX-OVERLAYS-ENTIRE-ADVANTAGE-1 — B passes every pre-registered gate and the pass is unattributable: at degree 1 it scores identically with 5.5× less memory

**Status:** FINDING [MEASURED]. One run of the Q8 harness over the pass-1
ore, after §11's pre-registration was committed (`30b1fa47`; commit order is
the proof). **Contains a failure of my own instrument, caught by its own G0
and recorded rather than replaced silently**, and a **result that
contradicts my pre-registered prediction, then a post-hoc ablation that
withdraws it.** Instrument reverted; board-only.
**Confidence:** High for the ablation (it is an equality, not a margin);
**low for any attribution of B's advantage to topology** — that is the
finding.

### What was asked

§7.2's probe — the one Q6/Q7 did **not** answer. They measured plasticity
and interference under capacity; this measures **pattern completion and
false resonance**, on the cheap form §7.2 itself names: the macro
co-occurrence graph as the learned neighbourhood, no new tissue. Grey
(learned overlay) against white (the substrate's native reading), in
parallel, with the exact table as authority throughout.

Corpus: `r2il-pass1.ore.tsv`, 17,560 rows, **2 binaries**, 143 chains, 9
opcodes. This is Q6/Q7's A-side, not their 4-binary `ore_all.tsv` — so no
learning or interference claim is available here, and none is made.

### The first instrument was degenerate, and G0 caught it

Completion was **0.0000 for every arm at every cap**. Cause: the task asked
each arm to propose a macro sharing the cue's *left* atom and scored it
correct only if the *right* atom matched too — which would make it the same
macro. **Structurally unwinnable.** Per the pre-registration, G0 failing
means no hypothesis verdict is reported, so none was. Rebuilt as held-out
next-macro prediction: adjacency learned on 115 train chains, scored on 28
held-out, the same merges applied in learned order.

### The gates then PASSED — against my own prediction

| macros | arm | completion | false res. | steps | footprint |
|---|---|---|---|---|---|
| 16 | A white/Cartesian | 0.0630 | 0.9370 | 1.65 | 168 |
| 16 | **B grey/hex** | **0.2730** | **0.7270** | **1.25** | **164** |
| 16 | RAND null | 0.0000 | 1.0000 | 1.58 | 192 |
| 33 | A | 0.0268 | 0.9732 | 1.82 | 372 |
| 33 | **B** | **0.1790** | **0.8210** | **1.38** | **336** |
| 33 | RAND | 0.0268 | 0.9732 | 1.82 | 396 |
| 64 | A | 0.0153 | 0.9847 | 1.89 | 744 |
| 64 | **B** | **0.1498** | **0.8440** | **1.58** | **632** |
| 64 | RAND | 0.0183 | 0.9817 | 1.93 | 768 |

G0 PASS, G1 PASS, G2 PASS, G3 PASS, and B's footprint is *smaller* than A's
— so under §7.2's kill condition verbatim, **B survives at every cap**. I
had predicted it would fail. Recorded as it stands.

### The ablation that withdraws it

The task uses each arm's **first** neighbour only. So degree was never
exercised — and re-running B at `DEGREE = 1`:

| macros | completion @6 | completion @1 | footprint @6 | footprint @1 |
|---|---|---|---|---|
| 16 | 0.2730 | **0.2730** | 164 | **30** |
| 33 | 0.1790 | **0.1790** | 336 | **60** |
| 64 | 0.1498 | **0.1498** | 632 | **120** |

**Identical to four decimals, at 5.5× less memory.** The six-neighbourness
contributes *nothing*. B is a bigram successor table; the topology is
decoration on top of it.

> **A pre-registered gate can pass for a reason the gate cannot see.** Q6
> taught the topology null (beat a random partition with the identical
> rule) and I included it — RAND. It was not enough: RAND varies the
> *wiring* while holding degree fixed, so it cannot detect that degree
> itself is inert. **A locality claim needs a DEGREE ablation, not only a
> wiring null.** §7.2's probe as specced lacks one, which is why any "B
> wins" it produced would have been unattributable.

(A's collapse to 0.0000 in the degree-1 run is an artifact of that run —
`DEGREE/2 = 0` empties the Cartesian construction — not a finding about A.)

### What this does and does not close

**Closed:** on this corpus and this task, the hex overlay's advantage is not
hexagonal. Whatever it earns, it earns as a learned pairwise successor
relation of degree one.

**NOT closed:** §7.2's other metrics (transfer, counterfactual quality)
remain unrun, and A is a weak arm here — macro ids are assigned in BPE
learning order, so "Cartesian neighbours" means "macros learned at adjacent
times", which is not the substrate's real Morton reading. A stronger A would
make B's margin smaller, never larger, so the ablation's conclusion is
unaffected — but no claim about the true Cartesian reading is made.

### Convergence worth naming

This lands where `E-PALETTE256-IS-A-NEEDLE-THE-COLON-IS-THE-DISTRIBUTION-1`
already was, by a different road: **the information is in the PAIR, not in
the neighbourhood's shape.** One relation `(a → b)` carried everything B
achieved; five more neighbours added nothing but bytes. That is the colon
doing the work, and it is consistent with Q6/Q7 finding hexagonal locality
bought nothing on plasticity either.

Cross-ref: plan §11 (pre-registration + gates); §7.2 (the probe);
`E-Q6-HEX-FAILS-CONTENT-ADDRESSING-IS-CAPACITY-DESTROYING-UNDER-A-SKEWED-DISTRIBUTION-1`;
`E-Q7-FREQUENCY-SIZING-RESCUES-THE-LEARNING-GATE-BUT-NOT-THE-INTERFERENCE-CLAIM-AND-THE-2-BYTE-RAILS-ARE-COMPLEMENTARY-NOT-COMPETING-1`;
`E-PALETTE256-IS-A-NEEDLE-THE-COLON-IS-THE-DISTRIBUTION-1`.
