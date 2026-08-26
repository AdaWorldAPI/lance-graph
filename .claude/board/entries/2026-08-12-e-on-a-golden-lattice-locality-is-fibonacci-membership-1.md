## 2026-08-12 — E-ON-A-GOLDEN-LATTICE-LOCALITY-IS-FIBONACCI-MEMBERSHIP-1

**Status:** FINDING `[G]` — W5 RUN (`spiral_adi_probe.py`/`.json`) + a
targeted verification measurement, both committed.

> **⊘ CORRECTION (2026-08-12, codex review on PR #936, same day) — the
> HEADLINE 0.9996 anisotropy ratio this finding cites is retracted; the
> MECHANISM below it is not.** The headline run's control was subsampled
> at 250k points/band against ~956k-point headline bands — ~74 % of
> sources got no real KD-tree and silently self-linked — so 0.9996 mostly
> measured self-links, not the control, and cannot by itself establish
> anything. **The 99.38 %-Fibonacci-offset verification measurement below
> is UNAFFECTED**: it ran at N=62 208, with per-band populations
> (~7 776 points) far under the 250k cap, so no subsampling occurred
> there — the mechanism claim (locality ⟺ Fibonacci membership via the
> three-distance theorem) still rests on a real, unsampled measurement.
> Fixed same day (`106ca605`): `build_control_links` is now full-band, no
> cap; a v2 headline rerun is in flight. Once it lands, this note gets a
> sibling recording the corrected headline ratio — append-only, this
> paragraph is not edited further.

> **⊘ SECOND CORRECTION (2026-08-12, codex P2 on PR #937) — "does not
> exist" overclaims what 99.38 % actually showed.** The claim below reads
> as a universal non-existence proof; the evidence is a 99.38 % rate at
> ONE N (62 208) with ONE control construction (distance-matched among 8
> real nearest neighbours) — not 100 %, and not a sweep across constructions
> or scales. **What is actually proven, exactly:** the three-distance
> theorem is a real theorem — a point's near neighbours on this lattice
> ARE at convergent-denominator (Fibonacci) index offsets, as a matter of
> number theory, not measurement. **What is empirical and should be read
> as strong CORROBORATION, not proof of universal impossibility:** 99.38 %
> of the measured control links landed on a Fibonacci offset, with the
> ~0.62 % remainder consistent with the independently-measured ~1–2 %
> structural boundary effect (points whose own forward Fibonacci partner
> falls outside band range, counted separately in the same session) rather
> than evidence of a working non-Fibonacci alternative. The claim below is
> RESTATED, not deleted, to match this scope: read "does not exist" as
> "the three-distance theorem predicts it cannot, and 99.38 % measured
> confirmation at one tested N/construction is consistent with that" — a
> sweep across N and control constructions would be needed to call the
> universal form settled.

> **★ THIRD UPDATE (2026-08-12, v2 headline RUN landed) — STRENGTHENED, at
> the real scale, on the full uncapped population, AND now on BOTH link
> families.** The v2 rerun (`106ca605`, no subsampling cap) measured the
> actual headline population directly: **99.68 % of the qualifying-band
> population's family-A control links (n_qualifying=4 782 017) land on a
> pure Fibonacci offset** — 99.95 % of the moved links specifically —
> dominated by offset 2584=F(18) itself (4 745 846 of 4 769 097 moved
> links), with the small remainder at 6765=F(20), 10946=F(21), and
> 13530=2·F(20) (a Fibonacci harmonic, not a counterexample).
>
> **⊘ THIRD-UPDATE CORRECTION (2026-08-12, codex P2 on PR #938, same day)
> — two overclaims in the paragraph above, both fixed.** (1) The 99.68 %
> figure covered only family A — `build_control_links`'s histogram was
> computed `if fam == 0` only, so family B (the OTHER stride direction,
> used every ADI iteration alongside family A) was never measured;
> generalizing to "the qualifying population's control links" from
> family-A-only data overclaimed. Fixed: the function now histograms both
> families. **Family B independently measured at 99.56 %** Fibonacci-offset
> rate, dominated by offset 4181=F(19) — the OTHER member of the discovered
> pair `[2584, 4181]`, exactly as the mechanism predicts (each family's
> characteristic near-neighbour scale IS that family's own discovered
> stride). Two families, two Fibonacci offsets, both >99.5 %. (2) "four
> orders of magnitude apart" for the two tested N was an arithmetic error
> — `4 782 017 / 62 208 ≈ 76.9×`, i.e. **~1.9 orders of magnitude**, not
> four. **Second correction's caveat stands as written** (one construction
> family at a time, now two scales AND two families, still not a sweep
> across fundamentally different control constructions — the "does not
> exist" phrasing remains restated, not un-restated) — the evidentiary
> base is nonetheless materially stronger: two independent N ~1.9 orders
> of magnitude apart, both link families, all landing within roughly half
> a percentage point of each other.
>
> **And a genuinely new, larger result rode along in the same run: B2 now
> FAILS.** v1's B2 PASS (anisotropy 1.2134) is now understood to have been
> an artifact of the SAME kind of confound this epiphany already
> catalogued for B3 — an inert operator (8 fixed iterations added ~0.003 %
> variance) inherited its apparent "isotropy" from a mask sitting only
> 1.72σ from the bump, whose own truncation analytically predicts a 1.208
> ratio, matching v1's number almost exactly. v2 fixed the mask clearance
> (3.35σ; baseline through it now measures an essentially clean 1.0046)
> AND scaled the iterations to a real physical diffusion target — and with
> both confounds gone, the ADI operator's own anisotropy is revealed for
> the first time: **1.5251, a genuine ~0.52 anisotropy contribution from
> the operator itself**, failing the 1.25 bar. Full detail:
> `weather-w-probes-v1.md` §1's v2 RUN section. Not filed as a separate
> epiphany — it is the SAME lesson (a control/baseline that cannot
> discriminate carries zero information when it "passes") applied to B2
> instead of B3, discovered by the same v2 fix that was built for B3.

**The claim, AS ORIGINALLY WRITTEN below (kept for the record; read through
both corrections above).** On a Vogel/golden lattice, a *local*
non-Fibonacci control for stride-structure experiments **does not exist**
— not "is hard to build," does not exist. Locality and Fibonacci-family
membership are the same property, by the three-distance theorem: a point's
near neighbours in
physical space are exactly the points at convergent-denominator index
offsets, and for the golden angle the convergent denominators ARE the
Fibonacci numbers.

**How three successive control designs discovered this by failing
differently.** W5's B3 control went through three generations, each
correcting the last: (1) strides 12/18 — wrong SCALE (connects points
nowhere near each other); (2) strides 1500/2600 — magnitude-matched but
still wrong scale in disguise (angular residues 0.05/0.11 vs the true
pair's 0.00028/0.00017 — two to three orders larger); (3) the
distance-matched shuffled-neighbour control — picks, per point, the REAL
nearest lattice neighbour closest in physical distance to the true partner,
excluding that partner. Generation 3 is the strongest possible local
control — and the RUN shows it changes nothing: anisotropy ratio
control/fib = **0.9996** (B3 bar: ≥1.5 → **VOID** by its own
pre-registered rule).

**The diagnosis, then the verification — in that order.** The suspicion:
generation 3's "non-Fibonacci" partners are themselves Fibonacci-offset
points, because on this lattice there is nothing else nearby to pick.
Measured directly (N=62 208, family-A control links): **99.38 % of the
control's links have a Fibonacci |Δk|** — top offsets 233, 987, 610
(all Fibonacci) and 1220 (= 2·610). The control never left the family.
The three-generation arc is therefore not a story of bad control design
but a **constructive proof sketch of the impossibility**: wrong-arithmetic
⟹ wrong-scale on a golden lattice, in both directions, because the
three-distance theorem couples the two.

**Consequences.**
- **W5's B3 question ("does the smoothing depend on the strides being
  Fibonacci?") is unanswerable BY A LOCAL CONTROL on this lattice** — and
  that is the honest verdict, not a probe defect. What CAN be said: the
  ADI smoothing quality is governed by local step geometry, and on a
  golden lattice, having correct local step geometry and being
  Fibonacci-linked are the same thing.
- **Companion to `E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1` (below,
  same day): that entry's failure mode is a control that cannot LOSE;
  this one's is a control that cannot DIFFER.** Both carry zero
  information when they "work" — and both were caught only because the
  runs happened and the results were interrogated instead of banked.
- A genuine falsifier for the Fibonacci-dependence question would have to
  leave locality: e.g. compare against a DIFFERENT lattice (jittered
  grid, Halton) with its own natural neighbour structure under the same
  stencil — deferred, scoped as a different experiment, not a fourth
  control generation on the same lattice.

**The positive results this run also delivered (recorded here since B3's
VOID would otherwise overshadow them):** **B2 PASS** at the headline
N=7 651 227 — iso-fit error **0.00053** against the 0.15 bar (≈280×
margin), anisotropy 1.213 vs the 1.25 bar: two Fibonacci-stride
tridiagonal sweeps DO approximate an isotropic 2D diffusion, which is the
load-bearing half for the domino.rs spiral-ADI design. **B4**: smooth
monotone improvement across the floor sweep (iso 0.44 → 0.0001 from n=8
to n=19, N=52.4M), **no knee at n=17** — on this metric the operator's
17/21 floor reads as a comfortable safety margin, per the pre-registered
two-sided reading. Honest residual: the anisotropy asymptotes at **~1.213
across three decades of N** (1.28/1.22/1.213/1.213 at n=12/14/17/19) —
an N-independent, structural ~21 % second-moment anisotropy of the
operator/geometry itself (band-restricted chains on polar geometry), not
a resolution artifact; it passes the 1.25 bar but does not tend to 1.0,
and any future tightening of that bar below ~1.22 would need this
mechanism addressed first.

