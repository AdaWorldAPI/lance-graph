## 2026-08-26 — E-Q7-FREQUENCY-SIZING-RESCUES-THE-LEARNING-GATE-BUT-NOT-THE-INTERFERENCE-CLAIM-AND-THE-2-BYTE-RAILS-ARE-COMPLEMENTARY-NOT-COMPETING-1 — Q6's hex verdict survives its own repair; PAL is language-portable, I8 is compiler-idiom, and BPE could not be ranked because the common metric was defective

**Status:** FINDING [MEASURED] — one run of the Q7 harness over the
four-binary ore, after §10's pre-registration was committed (`95ea634`;
commit order is the proof). Includes a **partial failure of my own
experimental design**, recorded rather than smoothed, and a **refinement
of how Q6's entry should be read**. Instrument reverted; board-only.
**Confidence:** High for Q7b's null separations and transfer ratios;
**deliberately low for Q7a's between-arm interference ordering** — see
the resolution limit below, which is the honest headline of that arm.

### Q7a — frequency-sized cells: the capacity fix works, the interference claim still fails

The degeneracy check did NOT fire: `max_cell_share` = 0.286 / 0.333 /
0.357, far below the pre-registered 0.80. So this is a clean test of
neighbourhood-bounded eviction with Q6's confound removed.

**The fix worked exactly as intended.** Quotas
`[4,4,3,2,1,0,0]` / `[7,6,5,2,1,0,0]` / `[10,8,6,3,1,0,0]` put HEX-FREQ
at **11/18/24 macros of 14/21/28 — 79/86/86% utilization**, against
Q6's uniform-cell **43/43/43%**.

| gate | cap 14 | cap 21 | cap 28 | verdict |
|---|---|---|---|---|
| G1 equal-learning (≥0.95·global) | 1.817 ≥ 1.743 **PASS** | 2.077 ≥ 2.004 **PASS** | 2.200 ≥ 2.182 **PASS** | **PASS 3/3** |
| G2 interference < GLOBAL | 0.526 vs 0.178 FAIL | −0.061 vs 0.100 PASS | 0.401 vs 0.158 FAIL | **FAIL (1/3, needs ≥2)** |
| G3 vs RAND | FAIL | PASS | FAIL | **FAIL (1/3)** |

G0: all three caps count. G4 byte-exact: true in all nine runs.

**⊘ This REFINES `E-Q6-HEX-FAILS-…-1`, and the refinement matters.**
Q6 reported hex "learns less **and** interferes more". Q7a shows the
*learns-less* half was a **construction artefact of uniform cell
sizing** — sized to content mass, hex clears the learning gate at every
cap. Q6's headline survives, but its mechanism attribution narrows to:
*hexagonal locality does not reduce interference even when it is given
enough capacity to learn normally.* That is a stronger refutation than
Q6's, because the confound is gone.

**The resolution limit, stated as the honest headline of this arm.**
Within a single arm, interference varies across caps as much as it
varies between arms at a fixed cap: RAND alone spans 0.081 / 0.377 /
−0.008 (spread 0.385), while the three arms at cap 21 span 0.100 /
−0.061 / 0.377 (spread 0.438). **n = 1 run per cell, no error bar.** So
the correct claim is *"HEX-FREQ does not meet the pre-registered bar"*,
**not** *"HEX-FREQ is worse than GLOBAL"* — this design cannot resolve
an effect of that size, and no amount of reading the table harder will
change that. Resolving it needs seeded replication, which is a
different experiment.

Two readings worth keeping: **interference can be negative** (−0.061,
−0.008) — learning B *raised* held-out-A density, because B's merges
also fire on C chains (the same positive transfer Q1 measured). And
`max_hop = 1` throughout: the locality bound stayed real, not vacuous.

### Q7b — what a 2-byte rail should carry

Equal budget in bytes (N ∈ {14,21,28} entries × 2 B), trained on A,
measured on H_A (unseen C) and H_B (rustc).

| carrier | hit H_A | hit H_B | transfer | column-shuffle null (H_A) | C0 |
|---|---|---|---|---|---|
| **BPE** | .9852 / .9967 / 1.000 | .9883 / .9994 / 1.000 | 1.00 | [.9852–.9951] … [.9951–1.000] | **FAIL ×3** |
| **PAL** `palette256:palette256` | .9539 / .9951 / 1.000 | .9480 / .9901 / 1.000 | **0.994 / 0.995 / 1.000** | [.498–.763] / [.771–.791] / [.808–.859] | **PASS ×3** |
| **I8** `i8:i8` | .7303 / .7911 / .8191 | .4892 / .5190 / .5348 | **0.670 / 0.656 / 0.653** | **[.000–.081] / [.000–.082] / [.000–.082]** | **PASS ×3** |

**I8 has by far the largest separation from chance** — real .73–.82
against a null that never exceeds .082, roughly a 10× ratio — and the
**worst transfer**, the only carrier in the 0.5–0.8 *partial* band.
**PAL passes clearly and transfers essentially perfectly** (0.99–1.00),
which **falsifies pre-registered prediction #2** (I predicted PAL would
transfer worse, on the reasoning that a whole-chain signature is more
corpus-specific; it is not). Prediction #3 is **confirmed**: metric
def-use distance is compiler idiom, symbolic/categorical identity is
language-portable.

> **The two rails are complementary, not competing.** At two bytes, a
> `palette256:palette256` categorical pair buys portability across
> languages; an `i8:i8` metric pair buys a far stronger signal against
> chance but binds to one compiler's idiom. Neither dominates — which
> is an argument *for* the canon's actual shape (one 12-byte register,
> six rails, the ClassView choosing the reading per class) and against
> picking a single winner for all rails.

### The design failure: the common metric was defective, so BPE cannot be ranked

I pre-registered **hit-rate** as the cross-carrier metric precisely
because PAL/I8 density is capped at 1.0 by construction. The null then
showed hit-rate is uninformative for BPE: with 9 opcodes and 3-atom
chains, nearly every held-out chain contains *some* frequent pair, so
even a column-shuffled corpus hits ~99%. **C3 therefore binds — BPE may
not be ranked against PAL or I8 at any budget in this run.**

A **post-hoc, exploratory** sweep (labelled as such; not pre-registered,
run after seeing the failure, and reported because leaving it
unexamined would be worse) shows the carrier is fine and the metric was
not:

```
Q7c N= 2 hit=.8717 null=[.5987...6628] | density=1.145 null=[0.599..0.676]
Q7c N= 6 hit=.9589 null=[.9852...9852] | density=1.898 null=[1.347..1.599]
Q7c N=14 hit=.9852 null=[.9852...9951] | density=2.197 null=[1.785..1.834]
Q7c N=28 hit=1.000 null=[.9951..1.000] | density=2.510 null=[2.135..2.155]
```

**BPE density separates from its null at every budget, N=2 through 28.**
Hit-rate saturates from N≈14 up — and, worse, **inverts** at N=6–8,
where the real vocabulary hits *fewer* chains than a shuffled one
(.9589 vs .9852). A two-sided range test scores that as "separated",
but separation in the wrong direction is not evidence of structure: a
shuffled corpus mints more *generic* macros that touch more chains
while carrying less. **Hit-rate is defective here in both directions**,
which is the same saturation trap this arc already recorded for
coverage — met again, from the other side, in a metric I chose *to
avoid* it.

What a valid three-carrier comparison needs (NOT run here): a metric
PAL and I8 can also express per-chain rather than per-corpus — e.g.
sliding-window multiplicity — or budgets small enough that no carrier
saturates. Named as the follow-up, not retrofitted into this result.

Cross-ref: plan §10 (pre-registration + gates);
`E-Q6-HEX-FAILS-…-1` (refined above, not overturned);
`E-Q1-THE-ADDITIVE-STORE-CANNOT-INTERFERE-YET-…-1`;
`I-NOISE-FLOOR-JIRAK` (all separations read distribution-free, as
range non-overlap, never as σ).

