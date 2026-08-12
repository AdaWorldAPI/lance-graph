# golden-vs-tempered-stride-v1 — head vs gut, made falsifiable

> **Status:** T1–T4 RUN (`probes/weather-p1/golden_vs_tempered_probe.py` /
> `.json`, committed with bars before execution, 2026-08-12). Cross-referenced
> from `weather-w-probes-v1.md` §0 (the golden-ratio index floor rule) — this
> plan is the standalone, substrate-general validation of that rule, not
> weather-specific. Zero fetch, pure arithmetic.
>
> **⚠ RUNNING THE PROBE CAUGHT TWO REAL DEFECTS IN THE HAND-DERIVED NUMBERS
> BELOW — both fixed, both explained where they occurred (T1, T3).** Neither
> changes the qualitative finding; both change specific numbers. This is
> exactly what "commit the bars, then run" is for.

## Why this file exists

Operator framing (2026-08-12, paraphrased): the intuitive pull is toward the
golden ratio as nature's own choice — sunflowers, phyllotaxis, the felt sense
that irrational growth is the "right" mechanism. Set against that is a
sharper, more analytic worry: a coprime tempered walk **never collapses** by
construction, closes its cycle **exactly**, and might for that reason be
**mathematically better**, at least in some regime, than an irrational
angle that only equidistributes in the limit.

Both instincts are checkable, and — the actual finding, pre-registered
before any bar was written — **both are correct, in different regimes, and
the crossover between them is itself measurable.** This plan makes that
precise instead of leaving it as a vibe.

## The two generators, precisely

- **GOLDEN (continuum):** `θ_k = k · 2π(1 − 1/φ)`. Irrational angle, never
  exactly repeats. Weyl equidistribution theorem [G]: prefix discrepancy
  `D*(m) = O(log m / m)` — improves without bound as `m → ∞`, no ceiling.
- **TEMPERED (quantized):** `θ_k = k · 2π·s/q` for a coprime integer stride
  `s` chosen against modulus `q`. Rational angle. **Closes exactly** at
  `m = q` (coprimality ⇒ full bijective permutation of `q` cells), then
  **repeats identically forever** — a hard ceiling on refinement, but a
  **deterministic, zero-variance guarantee of closure** the golden angle
  cannot offer at any finite `m`.

## T1 — the crossover, swept across q (descriptive, the headline)

**Method.** For each `q` in `{12, 17, 34, 55, 64, 89, 144, 233}`, enumerate
every coprime stride `s ∈ [1, q−1]` and select the one minimizing the
**median star-discrepancy over the USEFUL prefix range `m ∈ [⌈q/2⌉, q]`**
(excludes the tiny-`m` degenerate cases — `m=2` is trivially "discrepant"
for any stride and dominates a naive worst-case-over-all-`m` metric into
near-uselessness; caught and corrected mid-session before this plan was
written, see the caveat below). Compare that stride's score in the same
range to golden's score in the same range, and separately find `m*`, the
first prefix length beyond `q` at which golden's discrepancy drops below
the tempered stride's (permanently, since tempered is frozen at its `m=q`
value forever after).

**RUN, committed script, 2026-08-12** (`golden_vs_tempered_probe.json`):

| q | best coprime s | temp score (median, useful range) | golden score (same range) | m* (golden overtakes) | m*/q | temp/golden @ m=200q |
|---|---|---|---|---|---|---|
| 12 | 5 | 0.1667 | 0.1721 | 16 | 1.33 | 89.5× |
| 17 | 14 | 0.1042 | 0.1169 | 21 | 1.24 | 70.4× |
| 34 | 25 | 0.0570 | 0.0654 | 42 | 1.24 | 89.4× |
| 55 | 34 | 0.0384 | 0.0373 | 55 | 1.00 | 106.4× |
| 64 | 41 | 0.0312 | 0.0337 | 90 | 1.41 | 83.3× |
| 89 | 35 | 0.0251 | 0.0275 | 110 | 1.24 | 84.2× |
| 144 | 85 | 0.0160 | 0.0158 | 144 | 1.00 | 103.6× |
| 233 | 149 | 0.0104 | 0.0116 | 288 | 1.24 | 68.2× |
| 377 | 239 | 0.0066 | 0.0066 | 377 | 1.00 | 96.4× |
| 987 | 722 | 0.0028 | 0.0027 | 987 | 1.00 | 90.0× |

> **⚠ CORRECTION — `m*` was computed inconsistently with this plan's own
> prose in the pre-registered draft (all rows), caught by actually running
> the script.** The draft's `m*` search recomputed the TEMPERED sequence's
> star discrepancy at each growing `m > q` — but a tempered walk past `m=q`
> is REPEATING its own `q` positions, not sampling new ones, so that
> recomputation feeds duplicate points into a formula built for distinct
> order statistics, and the resulting "discrepancy" **spuriously worsens**
> instead of staying at its true, meaningful value. Worked example at
> `q=17`: at `m=18` (the point where the tempered stride's 18th sample lands
> exactly back on its own first position) the repeating-sequence
> recomputation jumps to 0.1111 — WORSE than tempered's actual frozen
> quality of 0.0588 — making golden's 0.0832 look like a win at `m=18` when
> it is still **worse** than tempered's real ceiling. The draft's `m*=18`
> for `q=17` was an artifact of this; the corrected script holds tempered at
> its true frozen `m=q` value (matching this plan's own stated definition:
> *"then repeats identically forever — a hard ceiling on refinement"*) and
> finds golden's genuine first crossing, `m*=21`. **Every `m*` in the
> corrected table is ≥ the draft's value** — under the correct definition,
> golden takes somewhat LONGER to overtake than the draft suggested, not
> shorter, so nothing here weakens the qualitative claim; it corrects the
> tightness of one specific number per row.

**Reading, stated as the finding rather than left implicit:**
- **The head is right in the bounded regime.** At every tested `q`, the
  best coprime tempered stride is **competitive with or better than**
  golden **within its own budget** (`m ≤ q`) — and it achieves this with
  **zero variance and a construction-guaranteed closure**, where golden's
  quality at any finite `m` is a continuous function with no guaranteed
  floor.
- **The gut is right in the unbounded regime.** `m*` — the point where
  golden permanently overtakes — sits **within about 1.0–1.4× of `q`** in
  every row tested (never more than half a cycle-length beyond `q`; exactly
  at `q` for four of the ten rows — `55, 144, 377, 987`). Beyond that,
  tempered is **frozen** at its `m=q` value forever (coprimality guarantees
  full closure, not continued refinement), while golden keeps improving as
  `O(log m / m)`. By `m = 200q` the gap is **68–106×** in golden's favor, at
  every `q` tested.
- **Neither instinct is wrong; they are answers to different questions.**
  "Is there ever going to be more data than this fixed budget?" — no ⇒
  tempered, exact closure, zero variance, done. "Is more data always
  coming, indefinitely?" — yes ⇒ golden, no ceiling, strictly better past
  `m ≈ q`.

**Bar T1 — RUN, result above:** the extended list (including `q=377,987`)
confirms the qualitative crossover pattern: `m*` never exceeds ~1.41× `q` at
any tested `q`, and lands exactly at `q` whenever the useful-range-optimal
stride's own frozen discrepancy already beats golden's score throughout the
sweep window (the `m*/q = 1.00` rows). No `q` broke the pattern into a
qualitatively different regime.

**⚠ CAVEAT, stated up front rather than discovered late (an earlier
worst-case-over-all-`m` metric picked DIFFERENT "best" strides for q=17 —
stride 10, tied with 11–15 at score 0.5000 — dominated by the degenerate
`m=2` case; and the EARLIER, narrower per-prefix-length comparison
committed in `EPIPHANIES.md` `E-THE-GOLDEN-STEP-IS-THE-WRONG-STEP-AT-SMALL-Q-1`
— stride 4 beating stride 11 at `m = 5/9/13` specifically — used yet a THIRD
metric, short fixed prefix lengths, and picked yet a different stride).
**"Best stride" is prefix-range-dependent — there is no single champion
across all `m`.** All three findings stand, each scoped to its own metric;
none contradicts another. This plan's canonical metric for T1 is the
useful-range median defined above; report which metric is in use whenever
citing a "best stride" number, here or elsewhere.

## T2 — the asymptotic claim, tested not assumed

**Bar (pass/fail):** for `m = 200q`, golden discrepancy `<` the tempered
stride's frozen `m=q` value, for **every** `q` in the T1 list. **RUN: PASS
at all 10 tested q (68.2×–106.4× separation)** — this is the arithmetic
validation of the intuitive "nature prefers golden ratio" pull, made
falsifiable rather than assumed. A single `q` where this bar fails would be
a genuine surprise and would need its own investigation before the T2
verdict stands.

## T3 — closure occupancy: does tempered actually GUARANTEE zero gaps?

**Method.** At `m = q` (tempered's own full cycle), bucket both walks into
`q` equal-width cells and count how many are empty. Tempered fills `q/q`
**by construction** (coprimality ⇒ bijection — this is not measured, it is
proven, and the measurement exists only to confirm no implementation bug).
Golden's fill count is genuinely **not guaranteed** and must be measured —
report it, and check it is not an artifact of bin-boundary phase by
re-binning at 5 different phase offsets.

> **⚠ CORRECTION — the FIRST run of this bar produced a false negative on
> tempered's OWN proof, caught by actually running it rather than trusting
> the proof-not-measurement framing.** The original method checked BOTH
> walks via the same float round-trip (`k/q` then `+offset` then `%1.0`
> then `*q` then `int()`) — and for the tempered walk, at `off=0.0`, this
> reported only **138/140** filled, contradicting its own "proof, not
> measurement" claim. Diagnosed: pure IEEE-754 truncation —
> `int(46.99999999999999)` rounds DOWN to 46 instead of 47, because
> `(47/140)*140` does not round-trip to exactly `47.0` in binary floating
> point. This affected only the MEASUREMENT CODE, not the mathematical
> fact (coprimality ⇒ exact bijection, provably true regardless of how it
> is measured). **Fixed: the tempered check now uses pure integer
> arithmetic (`(s·i) mod q`, never divided then re-multiplied) — it cannot
> have this artifact, and correctly reports 140/140 always.** The golden
> check is unaffected by this fix (its positions are inherently
> continuous, so the float phase-offset sweep is the legitimate empirical
> method there, not a proof-verification with a spurious failure mode).

**RUN, corrected method (non-Fibonacci q=140, avoiding the self-referential
case where q is itself a Fibonacci number — see the aside below):**
tempered fills **140/140** (exact integer check — proof confirmed, not
merely assumed). Golden fills **124–127/140 across 5 phase offsets tested**
(canonical phase: 124) — **13–16 empty cells depending on phase** — real,
not a binning artifact (verified via the phase sweep, and the artifact this
correction removed was in the TEMPERED check, not the golden one).

**Aside, reported not judged:** at `q = 144 = F(12)` (a Fibonacci number
itself), golden fills **144/144 at all 5 phases tested** — a
special/resonant case worth flagging but not treated as representative;
T3's headline number uses `q=140` specifically to avoid this
Fibonacci-on-Fibonacci confound.

**Bar T3 (two-sided by construction) — RUN, PASS on both sides:** tempered
fill = q/q **always** (140/140, exact integer arithmetic — the proof holds
and is now verified without a measurement artifact); golden fill `< q` at
`q=140` (**124–127/140**, well below 140, falsifiable and not falsified).

## T4 — the naive-rounding collapse hazard (the sharpest form of "kollabiert nicht")

**Why this is the sharpest test of the head's worry.** A NAIVELY IMPLEMENTED
golden stride — `round(frac · q)` without checking coprimality — can
literally **collapse**: if `gcd(round(frac·q), q) = g > 1`, the walk visits
only `q/g` distinct cells, repeating a short cycle instead of covering the
space. This is not a hypothetical: already measured in this session at
`q=64` (`round(0.382·64)=24`, `gcd(24,64)=8` — only 8/64 cells reached) and
`q=256` (`gcd=2`, only 128/256 reached).

**Method.** Sweep `q ∈ [8, 300)`, compute `s = round(frac·q)`, check
`gcd(s, q)`. A properly-implemented tempered walk, by contrast, only ever
searches the coprime candidates (by construction, cannot collapse — the
search space excludes non-coprime `s` entirely).

**Measured: 114 of 292 tested q (39.0 %) collapse under naive golden
rounding.** Examples: `q=9→s=3,gcd=3`; `q=10→s=4,gcd=2`; `q=15→s=6,gcd=3`;
`q=16→s=6,gcd=2`; `q=20→s=8,gcd=4`; `q=22→s=8,gcd=2`; `q=24→s=9,gcd=3`;
`q=25→s=10,gcd=5`.

**Bar T4 (pass/fail, and this is the one that matters most for practice):**
collapse rate under naive rounding **> 25 %** across the swept range (bar:
demonstrates the hazard is common, not a corner case) — **measured 39.0 %,
PASS** — versus **0 %** collapse for a coprimality-checked tempered search
by construction (proof, not measurement — the search space excludes
non-coprime candidates entirely, so this is a structural guarantee, stated
as such rather than measured as a frequency).

**Reading:** this is the strongest, most concrete form of the head's
worry — "does not collapse" is not a vague reassurance, it is a **39 %
failure rate of the naive alternative**, avoidable ENTIRELY by checking
`gcd(s,q)=1` before use, which the workspace's shipped `CurveRuler`
already does correctly (stride 4, `gcd(4,17)=1`).

## Synthesis — the design rule this plan earns

| regime | which instinct is right | generator | why |
|---|---|---|---|
| **fixed, bounded budget** (`m ≤ q`, exact closure needed, e.g. a byte-addressable rail, a facet's palette index) | **head** | tempered, coprimality-checked | zero-variance closure guarantee (T3); competitive-to-better discrepancy within budget (T1); avoids the 39 % naive-rounding collapse hazard (T4) |
| **unbounded, growing budget** (`m ≫ q`, e.g. a continuum lattice sampled indefinitely, real phyllotaxis with thousands of florets) | **gut** | golden angle | no ceiling — `O(log m/m)` refinement forever, 68–106× ahead of any frozen tempered walk by `m=200q` (T1, T2) |

This is not a tie-breaker between the two intuitions — it is the discovery
that **each is the correct mechanism for its own regime**, and the
crossover sits almost exactly at `m ≈ q` in every case tested. Filed as the
final validation of the two-regime table already committed in
`COMET_TAIL_REPORT.md` §10.5 and `EPIPHANIES.md`
`E-THE-GOLDEN-STEP-IS-THE-WRONG-STEP-AT-SMALL-Q-1` — this plan supplies the
head-to-head arithmetic that entry asserted but did not yet run as a
standalone, swept comparison.

### Refinement — the storm's own geography sorts the two regimes (operator, 2026-08-12)

Operator observation (paraphrased): the analytic/tempered register belongs at
the **storm's eye**, the golden/sunflower register in the **collision zone**
(the arc's Nahkampf terrain) — and specifically, the tempered register gives
the more exact storm **centering**, because a golden spiral's own center is
where its data quality is worst.

All three halves of that are mechanically grounded in measurements this arc
has already made — the mapping is not an aesthetic assignment:

1. **The spiral's center is structurally sub-floor at ANY N.** Local
   parastichy index at radius `r` is `√(r²·N)` → 0 as `r` → 0 — the exact
   arithmetic behind the W5 fix (`weather-w-probes-v1` §1), where raising N
   could never rescue the inner bands and the only correct move was
   excluding them from judgment. A Vogel lattice's innermost points are
   degenerate by construction: resonant stride families, uneven first-few-k
   spacing, no floor-qualifying neighbourhood. **Center work on a golden
   lattice is garbage-register work, at any budget.**
2. **The shipped center-finder is already on the correct register.**
   `find_center` (`comet_tail_f5_n10.py` / `comet_tail_f16.py`) is an
   exact-enumeration argmin over the regular lat/lon grid plus a quadratic
   sub-grid fit — bounded, exhaustive, zero-variance: the tempered/head
   register, never a spiral sample. What was implicit practice is now
   stated doctrine: **centering = exact enumeration; never sample the
   center from the spiral whose center is its own worst data.**
3. **The sunflower is the territory-gain model, and its advantage is that
   its addresses speak for themselves outward.** Prefix-extensibility
   (report §10.5 property 3) means position is *implied* by index `k` —
   place deterministic, residue stored, nothing re-meshed — and each added
   point lands in the largest current gap (`O(log N/N)` discrepancy). In
   the Go framing the arc already measured (`go_territory_probe`): golden
   is the **opening** — optimal incremental territory claim in open,
   growing space; tempered is the **endgame count** — exact, closed,
   bounded. Same two regimes as the T1 table, now with the game-phase
   reading attached. **And the self-description is unbounded** (operator,
   same session, paraphrased: self-explanatory without end) — which is the
   sharpest way to state the asymmetry. A tempered walk is self-describing
   exactly up to `q`: beyond one cycle the index carries no new
   information, the address vocabulary is exhausted, the walk repeats. The
   golden index is self-describing for **every** `k`, indefinitely — no
   re-meshing, no re-indexing, no schema change at any scale, with quality
   still *improving* as it grows (`O(log m/m)`, T2's measured 68–106×
   advantage at `m=200q` IS this property in number form). Bounded
   self-description that closes exactly, versus unbounded self-description
   that never stops refining: that is the entire duel in one sentence.

4. **Overlaying the collision lattices is controlled chaos at near-zero
   storage cost** (operator observation, same session, paraphrased: the
   superposition is pleasingly chaotic AND cheap). Both halves are
   measurable, not vibes. *Chaotic:* two golden lattices over different
   centers are incommensurate — the overlay pattern is aperiodic (never
   repeats, no moiré, no systematic ties; §10.5 property 2, and exactly
   what W2s-a's G1/G3 bars measure) — yet fully **deterministic**: the
   entire pattern regenerates from two center coordinates and nothing
   else. Low-discrepancy chaos, not randomness — reproducible, seedless,
   audit-friendly. *Cheap:* the overlay stores **no geometry at all**.
   Every point's position is implied by its index (place deterministic),
   so a collision node costs exactly its 12-byte V3 facet — rails 0–1 the
   `(k_H:k_T)` pair address, rails 2–5 the 8 state bytes — with radii,
   azimuths, and positions all *derived*, never persisted. Densifying the
   overlay adds nodes, not meshes: cost grows with the number of
   collisions you choose to materialize, not with the resolution of the
   space they live in.

**The demarcation that keeps this honest:** the geography does not *cause*
the regime — the **task** does. The eye is tempered territory because the
task there (resolve one address, fixed budget, exact closure) is a bounded
task; the collision annulus is golden territory because the task there
(pair two growing lattices ever more densely) is an unbounded one. It
happens — and this is the elegant part, not the load-bearing part — that
the storm's geography sorts its tasks into exactly the two regimes the T1
crossover table measures.

## Execution — RUN

Zero fetch, pure stdlib `math`/`statistics` — no `numpy`/`scipy` needed after
all (`q ≤ 987`, no lattice-scale KD-tree required, unlike
`weather-w-probes-v1`'s W5/W2s-a). Committed with bars before execution
(`38c56d00`), then run (< 5 seconds wall time), then two real defects were
caught by the run itself and fixed (T1's `m*` methodology, T3's float
round-trip false negative) — both explained inline above rather than
silently absorbed into the numbers. `probes/weather-p1/
golden_vs_tempered_probe.py` / `.json` / `.partial.jsonl` are the committed
artifacts; `.json` is the record of truth for every number in this document.
