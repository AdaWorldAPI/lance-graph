## 2026-08-26 — E-Q6-HEX-FAILS-CONTENT-ADDRESSING-IS-CAPACITY-DESTROYING-UNDER-A-SKEWED-DISTRIBUTION-1 — the hex A/B experiment fails every hypothesis gate (G1–G3) at every cap while both validity gates pass; a random partition with the identical locality rule beats it outright

**Status:** FINDING [MEASURED] — **NEGATIVE RESULT, recorded exactly as
a pass would be.** One run of the Q6 harness over the four-binary ore
(`ore_all.tsv`), executed AFTER its pre-registration was committed
(`6d6f3c6`, plan §9 — commit order is the proof). Instrument reverted;
no source change. **Confidence:** High for the failure and its
mechanism; the named caveat below is stated with its direction of bias.

### The verdict, against the pre-registered gates (plan §9.6)

The five gates split into two kinds, and the distinction is load-bearing:
**G0 and G4 are VALIDITY gates** (does this experiment measure anything,
and is it sound?) — **both PASSED**. **G1–G3 are HYPOTHESIS gates** — all
three failed at all three caps. The validity passes are what make the
hypothesis failure trustworthy instead of an artefact; an earlier
wording of this entry said "fails every gate", which contradicted its
own G0/G4 rows (caught in review on #1036, corrected before merge).

| cap | G1 equal-learning (hex_B ≥ 0.95·global_B) | G2 headline (int hex < global) | G3 topology null (int hex < rand) |
|---|---|---|---|
| 14 | **FAIL** 1.465 < 1.743 | **FAIL** 0.498 vs 0.178 (×2.80 worse) | **FAIL** 0.498 vs 0.081 |
| 21 | **FAIL** 1.622 < 2.004 | **FAIL** 0.421 vs 0.100 (×4.21 worse) | **FAIL** 0.421 vs 0.377 |
| 28 | **FAIL** 1.816 < 2.182 | **FAIL** 0.531 vs 0.158 (×3.36 worse) | **FAIL** 0.531 vs −0.008 |

G0 (inertness): all three caps COUNT — GLOBAL evicted A-macros (7/6/3)
and showed interference > 0 (0.178/0.100/0.158) at every cap, so the
control genuinely forgets. G4 (byte-exact): **true in all nine runs** —
R2IL stayed the sole truth throughout.

**Hex fails in the worst available combination: it learns LESS *and*
interferes MORE.** The pre-registered prediction expected the trivial
failure (less interference bought by less learning); the measured
failure is strictly worse than that, and the operator's own rule
applies without qualification — *a hex topology that merely looks
brain-like but does not reduce interference FAILS.*

### The mechanism, visible in the run's own census

A's chains use only 5 opcodes, heavily skewed:
`copy 2147 · int_add 1696 · load 1258 · store 458 · return 57`.
Macros are addressed by their first atom, so nearly all land in cells
0–2; cells 3–6 are never addressed at all. With per-cell capacity
`cap/7` = 2/3/4, the addressed cells saturate immediately while the
rest sit permanently empty. **Hex holds 6/9/12 macros of a nominal
14/21/28 — under half its own capacity.** RAND, hashing uniformly
across the same 7 cells with the same adjacency, eviction and refusal
rules, holds 13/20/25.

So the finding generalizes past hexagons:

> **Content addressing under a heavy-tailed content distribution is
> capacity-destroying.** The very uniformity that makes the random
> partition semantically meaningless is what makes it work; the
> semantic locality that makes hex *look* brain-like is what starves
> it. A partition is only as good as the flatness of the distribution
> it partitions.

The high interference follows from the same cause: hex's baseline rests
on 6–12 macros, so evicting 4–6 removes a large FRACTION of its
vocabulary, where GLOBAL loses 3–7 of 14–28. Small effective capacity
makes every eviction proportionally more expensive.

Two secondary readings worth keeping:

- `max_hop = 1` for hex and rand — the locality bound is **real and
  non-vacuous**: evictions genuinely reached neighbour cells, they were
  not all same-cell.
- `global_far = 0` at every cap — GLOBAL's evictions, scored under hex
  addressing, were **never ≥2 hops away**. The global pool never needed
  non-local reach on this corpus, so hex's bounded propagation solves a
  problem that does not occur here.

### Named caveat, with its direction of bias

A refused candidate pair is **permanently blocked** from re-selection
(it does not enter the vocabulary and does not shape later learning).
Applied identically to all three arms, but hex refuses most (35–43 vs
11–34), so it bites hex hardest. This entangles the **G1** failure:
some of hex's learning shortfall is this rule, not the topology. It
cannot explain **G2** — blocking *reduces* learning, hence reduces
evictions, hence would reduce interference, yet hex's interference is
2.8–4.2× the control's. **The headline failure is robust to the
caveat.**

Second semantics note: eviction removes a macro from the resident
vocabulary but leaves the stream encoding intact (the SymTable is
append-only). All arms share this, so the comparison is fair, but the
absolute interference numbers are optimistic for every arm.

### What this does and does not close

Closed: *this* hex construction — radius-1 tile, first-atom addressing,
neighbourhood-bounded eviction — on *this* corpus. It is not rescued by
retuning, and no retune was attempted (per the brief: **do not quietly
optimize around a failure; measure it**).

NOT closed, and explicitly NOT run here: a **frequency-sized** cell
variant (cells sized by content mass rather than uniformly), which
would directly target the capacity waste this run identified. That is a
different experiment needing its own pre-registration — naming it here
is the honest disposal of the failure, not a rescue of it.

Cross-ref: plan §9 (pre-registration + gates);
E-Q1-THE-ADDITIVE-STORE-CANNOT-INTERFERE-YET-AND-THE-VOCABULARY-IS-ORDER-ROBUST-1
(which ungated Q6 by showing the unbounded store cannot interfere at
all); E-THE-QA-MACHINERY-IS-THE-LEARNING-RULE-1.

