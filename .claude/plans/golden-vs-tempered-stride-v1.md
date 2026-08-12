# golden-vs-tempered-stride-v1 — head vs gut, made falsifiable

> **Status:** T1–T4 RUN (`probes/weather-p1/golden_vs_tempered_probe.py` /
> `.json`, committed with bars before execution, 2026-08-12). Cross-referenced
> from `weather-w-probes-v1.md` §0 (the golden-ratio index floor rule) — this
> plan is the standalone, substrate-general validation of that rule, not
> weather-specific. Zero fetch, pure arithmetic.
>
> **⚠ THE RUN CAUGHT DEFECTS TWICE — once from running it, once from
> external review of the run's own result.** First pass (running the
> script): two real defects in the hand-derived numbers (T1's m*
> methodology, T3's float round-trip false negative). Second pass (codex
> review on PR #935, of the FIRST corrected run): two more real defects,
> both in T1 (an off-by-one in the useful-range floor, and — the
> substantive one — "m*" was a first-crossing search, not a verified
> PERMANENT one, so the corrected-once table was still wrong in a way that
> mattered). All four are fixed and explained inline (T1, T3). None
> changes the qualitative finding; all four change specific numbers, twice
> over for `m*`. This is what "commit the bars, then run, then let review
> hit the run" is for — each pass caught something the previous one
> could not see from the outside.

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

**RUN, committed script, 2026-08-12 — THIRD revision of `m*`, this one
verified rather than assumed** (`golden_vs_tempered_probe.json`):

| q | best coprime s | temp score (median, useful range) | golden score (same range) | m* (VERIFIED permanent) | m*/q | checkpoints verified | temp/golden @ m=200q |
|---|---|---|---|---|---|---|---|
| 12 | 5 | 0.1667 | 0.1721 | 24 | 2.00 | 76 | 89.5× |
| 17 | 14 | 0.0966 | 0.1087 | 32 | 1.88 | 78 | 70.4× |
| 34 | 25 | 0.0570 | 0.0654 | 68 | 2.00 | 80 | 89.4× |
| 55 | 34 | 0.0383 | 0.0367 | 115 | 2.09 | 81 | 106.4× |
| 64 | 41 | 0.0312 | 0.0337 | 170 | 2.66 | 81 | 83.3× |
| 89 | 35 | 0.0247 | 0.0275 | 212 | 2.38 | 82 | 84.2× |
| 144 | 85 | 0.0160 | 0.0158 | 320 | 2.22 | 83 | 103.6× |
| 233 | 149 | 0.0103 | 0.0116 | 589 | 2.53 | 82 | 68.2× |
| 377 | 239 | 0.0066 | 0.0066 | 929 | 2.46 | 83 | 96.4× |
| 987 | 722 | 0.0028 | 0.0027 | 2521 | 2.55 | 83 | 90.0× |

> **⚠⚠ TWO CORRECTIONS TO `m*` NOW, NOT ONE — stated plainly rather than
> quietly folded in, because the number has moved twice and a reader
> deserves to see the trajectory.** The pattern each time: a real
> methodological gap the run itself exposed, each fix moving `m*` further
> from `q`, never closer.
>
> **First correction (already recorded here) — the draft recomputed the
> TEMPERED sequence past its own closure**, feeding repeated points into a
> distinct-order-statistic formula, spuriously worsening it and making
> golden look like it won earlier than it did. Fix: hold tempered frozen at
> its true `m=q` value. That produced the (now superseded) `m* ≈ 1.0–1.4×q`
> table.
>
> **Second correction (codex P1 on PR #935) — that "frozen-ceiling" `m*`
> was still only a FIRST crossing, not a verified PERMANENT one.** Golden's
> raw discrepancy sequence is not monotonic — only its `O(log m/m)`
> ENVELOPE is a bound — so a single dip below the tempered ceiling can be
> followed by a rise back above it before the sequence settles for good.
> Codex's exact, reproduced example: `q=17` reported `m*=21`, but
> `D*(22) = 0.08137`, ABOVE the frozen ceiling `0.05882` — not permanent at
> all. Fixed with `verified_permanent_crossover`: a candidate crossing is
> checked against a **sampled checkpoint set** (every integer for the next
> 50 steps — this is what catches near-term reversals exactly like the
> q=17 case — plus ~15 geometrically-spaced points out to the `m=200q`
> horizon, plus the horizon itself); any checkpoint violation restarts the
> scan past it. **The checkpoint count is reported alongside every `m*`**
> (76–83 points per row) so the verification scope is stated, not implied
> as exhaustive — points strictly between checkpoints are not individually
> checked, though the sampling density (every integer for 50 steps right
> after the candidate, where reversals are most likely, per the q=17
> example) is chosen to make an undetected reversal unlikely.
>
> Also note: the useful-range floor bug (codex P2, `q//2` vs `⌈q/2⌉`) is
> folded into this table too — it shifted `temp/gold score` slightly for
> odd `q` (17, 55, 89, 233), visible above; it did not change any stride
> choice or any pass/fail verdict.

**Reading, corrected a second time:**
- **The head is right in the bounded regime — this claim is unaffected by
  either correction.** At every tested `q`, the best coprime tempered
  stride is **competitive with or better than** golden **within its own
  budget** (`m ≤ q`) — zero variance, construction-guaranteed closure,
  where golden's quality at any finite `m` has no guaranteed floor.
- **The gut is right in the unbounded regime, and its margin is LARGER than
  first stated.** The verified-permanent `m*` sits at **roughly 1.9–2.7× `q`**
  — golden needs about two tempered cycles' worth of samples, not one, before
  it can be trusted never to dip back above the frozen ceiling. This is a
  WEAKER claim for tempered's near-term competitiveness than the
  once-corrected table suggested (`1.0–1.4×`), and a stronger one for
  golden's eventual, durable dominance. Beyond `m*`, tempered is frozen at
  its `m=q` value forever, while golden keeps improving as `O(log m/m)`; by
  `m=200q` the gap is **68–106×** in golden's favor at every `q` tested,
  UNCHANGED by either correction (T2 was always computed at the single
  fixed point `m=200q`, never via a crossing search, so it was never
  exposed to either bug).
- **Neither instinct is wrong; they are answers to different questions,
  and the honest margins are now wider apart than first drafted, not
  narrower.** "Is there ever going to be more data than this fixed
  budget?" — no ⇒ tempered, exact closure, zero variance, done. "Is more
  data always coming, indefinitely?" — yes ⇒ golden, no ceiling, verified
  durably better past roughly `2× q`.

**Bar T1 — RUN, corrected result above:** the extended list (including
`q=377,987`) confirms the qualitative crossover pattern holds at every
tested `q` — golden's advantage is DURABLE once past its verified `m*`, not
merely a lucky first dip. No `q` broke the pattern into a qualitatively
different regime; what changed across both corrections is the TIGHTNESS of
the crossover estimate, not its existence or direction.

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
that **each is the correct mechanism for its own regime**, and the VERIFIED
crossover sits at roughly **1.9–2.7× `q`** in every case tested (the
number moved twice under review, both times widening — see T1's two
correction notes above). Filed as the final validation of the two-regime
table already committed in
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
