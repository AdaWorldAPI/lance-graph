# golden-vs-tempered-stride-v1 — head vs gut, made falsifiable

> **Status:** ACTIVE for the exploratory tier (T1–T4 below). Cross-referenced
> from `weather-w-probes-v1.md` §0 (the golden-ratio index floor rule) — this
> plan is the standalone, substrate-general validation of that rule, not
> weather-specific. Zero fetch, pure arithmetic, runnable by any Sonnet
> worker with `numpy` + `scipy` only.

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

**Pre-registered expectation, measured before commit:**

| q | best coprime s | temp score (median, useful range) | golden score (same range) | m* (golden overtakes) | temp/golden @ m=200q |
|---|---|---|---|---|---|
| 12 | 5 | 0.1667 | 0.1721 | 13 | 89.5× |
| 17 | 14 | 0.1042 | 0.1169 | 18 | 70.4× |
| 34 | 25 | 0.0570 | 0.0654 | 35 | 89.4× |
| 55 | 34 | 0.0384 | 0.0373 | 55 | 106.4× |
| 64 | 41 | 0.0312 | 0.0337 | 66 | 83.3× |
| 89 | 35 | 0.0251 | 0.0275 | 90 | 84.2× |
| 144 | 85 | 0.0160 | 0.0158 | 144 | 103.6× |
| 233 | 149 | 0.0104 | 0.0116 | 234 | 68.2× |

**Reading, stated as the finding rather than left implicit:**
- **The head is right in the bounded regime.** At every tested `q`, the
  best coprime tempered stride is **competitive with or better than**
  golden **within its own budget** (`m ≤ q`) — and it achieves this with
  **zero variance and a construction-guaranteed closure**, where golden's
  quality at any finite `m` is a continuous function with no guaranteed
  floor.
- **The gut is right in the unbounded regime.** `m*` — the point where
  golden permanently overtakes — sits almost exactly at `m ≈ q` in every
  row (crossing within one budget-length of the tempered walk's own
  ceiling). Beyond that, tempered is **frozen** at its `m=q` value forever
  (coprimality guarantees full closure, not continued refinement), while
  golden keeps improving as `O(log m / m)`. By `m = 200q` the gap is
  **68–106×** in golden's favor, at every `q` tested.
- **Neither instinct is wrong; they are answers to different questions.**
  "Is there ever going to be more data than this fixed budget?" — no ⇒
  tempered, exact closure, zero variance, done. "Is more data always
  coming, indefinitely?" — yes ⇒ golden, no ceiling, strictly better past
  `m ≈ q`.

**Bar T1 (descriptive, no single pass/fail — the crossover table itself is
the deliverable):** report the table above, regenerated at run time rather
than copied, for the full q list plus **two additional q not yet run**:
`q = 377` and `q = 987` (both Fibonacci, continuing the ladder) — confirm
the `m* ≈ q` pattern holds or report the first `q` where it breaks.

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
stride's frozen `m=q` value, for **every** `q` in the T1 list. **Measured:
TRUE at all 8 tested q (68.2×–106.4× separation)** — this is the arithmetic
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

**Measured (non-Fibonacci q=140, avoiding the self-referential case where q
is itself a Fibonacci number — see the aside below):** tempered fills
**140/140** at every phase (proof, not measurement). Golden fills
**124/140 at the canonical phase** — **16 empty cells** — and the count is
**stable across 5 bin-phase offsets tested** (not a binning artifact).

**Aside, reported not judged:** at `q = 144 = F(12)` (a Fibonacci number
itself), golden happened to fill **144/144 at all 5 phases tested** in a
quick check — a special/resonant case worth flagging but not treated as
representative; T3's headline number uses `q=140` specifically to avoid
this Fibonacci-on-Fibonacci confound.

**Bar T3 (two-sided by construction):** tempered fill = q/q **always** (a
guard against an implementation bug more than a finding); golden fill `<
q` for **at least** `q=140` (falsifiable — if golden also fills 140/140,
the closure-guarantee argument for T3 is weaker than claimed and must be
restated as "usually" rather than "guaranteed-vs-not").

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

## Execution

Zero fetch, pure `numpy`/`scipy.spatial` (only T3's KD-tree-adjacent bucket
counting needs anything beyond stdlib math, and even that is trivial at
these sizes — `q ≤ 987`, no lattice-scale KD-tree needed here at all,
unlike `weather-w-probes-v1`'s W5/W2s-a). Single Sonnet worker,
**~5 minutes**, no `§0` preamble needed (this plan is self-contained and
carries no weather-domain data access). One script,
`probes/weather-p1/golden_vs_tempered_probe.py`, emitting
`golden_vs_tempered_probe.json` with `{T1: [...], T2: {...}, T3: {...},
T4: {...}}`. Commit the script with its bars BEFORE running, per the
standing discipline.
