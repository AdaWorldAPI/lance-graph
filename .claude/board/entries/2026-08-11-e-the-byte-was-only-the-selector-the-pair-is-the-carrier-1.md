## 2026-08-11 — E-THE-BYTE-WAS-ONLY-THE-SELECTOR-THE-PAIR-IS-THE-CARRIER-1

**Status:** FINDING `[H]` — operator correction + `l4_rail_probe.py` (commit
96e86b90); report §6.1. EXPLORATORY, not an EV.

**Operator:** *"was ist mit 6× Palette256:Palette256 centroid, was ja die
Verteilung anzeigen soll — palette256 alleine ist ja nur 'attention header'"*.

**Every encoding in this arc treated "one scalar → one byte" as the unit. The
shipped carrier is a PAIR.** le-contract §3 row L4 is `6 × (8:8)`,
`palette256²` — "each byte pair indexes the 256×256 palette distance/compose
tables; similarity = ONE table read". The single byte is the **selector**; the
**pair** is a cell in the centroid tile, and the tile is where the
distribution lives. I had built one rail and called it the carrier — the
rolling-floor cascade is the sanctioned §3 "area : location in stacked
exactness" reading, but of ONE rail out of six.

**Measured: the 12-byte facet is LOSSLESS against the f64 spine.** Carve D
(dipole rail + 10 ring bytes *spread over the full radius*, missing rings
interpolated) reproduces the f64 constrained spine to four decimals on both
storms — R² 0.9434 / 0.9090, |D − f64| = 0.0000. R² is demonstrably sensitive
(carve B, the same budget spent on 12 rings with no dipole rail, collapses to
0.635 / 0.294), so this is recovery, not insensitivity.

**Two of four pre-registered bars FAILED as written, and both failures taught
more than the pass.**

1. **L1 failed** (0.0222 vs a 0.02 bar) and the decomposition names the cause
   exactly: **quantization +0.0000, dropped rings +0.0222**. *The carrier's
   PRECISION is free; its CAPACITY was the entire miss.* Spending the same 12
   bytes across the full radius erases it. Generalizable: when a byte-budget
   fit misses, decompose before widening — the two costs point at opposite
   fixes (a bigger codebook vs a better carve), and here the codebook was
   already perfect.

2. **L3 failed, and so did my proposed rescue** — the more useful half.
   Fisher-z centroid axes are **5× WORSE** than uniform on the ring means
   (18.07 vs 3.84 Pa). I hypothesised the population was wrong (ranks against
   the 24 encoded values rather than the field) and measured that too: **19.00
   Pa, no rescue.** So the mechanism is not population size but *what the read
   is for*: ring means are a smooth NARROW-BAND quantity sitting
   mid-distribution, and a rim-stretch spends levels in tails where no ring
   mean lives.

**The demarcation this forces, which is the entry's real content:** it does
NOT contradict `three_register_probe`'s R4, where Fisher-z is **8.3× TIGHTER**
than plain rank in the storm tail (24.74 vs 204.54 Pa) on the raw field. Same
substrate, opposite verdicts. **Fisher-z wins a RANK/TAIL read and loses an
INTERPOLATE/LEVEL read.** Which is exactly why le-contract says a ClassView
**MAY** declare an analytic codebook — per class, by measurement, not as a
default. This corrects my own over-generalization, made earlier the same
session, that Fisher-z is *the* L4 codebook axis.

**A ninth vacuous falsifier, and the mechanism is worth naming.** L4x
(shared-vs-per-storm codebook) passed on its first run **comparing an array
against itself**: a uniform codebook is fixed by its population's min/max
alone, so because storm 1's profile range strictly CONTAINS storm 2's, storm
1's "own" codebook IS the pooled codebook. It looked like a real comparison
only because an earlier variant (fisherz, which depends on the whole rank
distribution rather than the endpoints) had produced *differing* numbers —
**switching to the codebook the previous bar had just NAMED AS BEST is what
made the test vacuous.** Degeneracy flags now report both directions; the
informative one gives **620.79 Pa vs 4.48 Pa, a 139× penalty**, strong
evidence the codebook must be global — the "one table read" property the
carrier exists for.

**Rule:** *when a bar's inputs are derived from the same population, check
they are not the same OBJECT before reading its verdict.* Equal numbers on
both sides of a comparison are the signature, and "it passed" is the least
informative way to find out.

