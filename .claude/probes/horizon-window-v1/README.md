# PROBE-HORIZON-WINDOW-1 — is the A9 ±8 locus window a discount, and of what kind?

**Re-runnable.** Unlike `hexagon-plasticity-v1/` (auditable, not re-runnable — its
corpus was ephemeral), this probe needs no corpus, no fetch and no RNG:

```sh
python3 .claude/probes/horizon-window-v1/window_sweep.py
```

## The claim under test

A `Locus` is *"a signed offset naming WHERE in the ±8 `temporal.rs` Markov window
that dimension's filler sits"* — a **temporal-distance** register whose values are
`[−8, +7]` with `0 = unbound`. So the window is a perception rule over delay, and
the question is which one: a steep discount, or something else.

Canonical preference-reversal fixture — a **sooner-smaller** reward at absolute
position `T1` and a **later-larger** one at `T2 > T1`, with the agent deciding
from each vantage `τ ∈ [0, T1)`. Fixed absolute positions are what make the
exponential control provable.

| model | rule | role |
|---|---|---|
| exponential(γ) | `v · γ^offset` | **harness control** — time-consistent, provably never reverses |
| hyperbolic(k) | `v / (1 + k·offset)` | **positive control** — the classic reversal generator |
| boxcar(w) | `v` if `offset ≤ w` else **unbound** | the A9 window under test |

Loci are POINTERS, never magnitudes (operator-locked, `E-SIX-SEMANTIC-FAMILIES-
MUST-NOT-IMPERSONATE-EACH-OTHER-1`). The window gates **visibility of the
pointer**; values live in a separate table. No magnitude is stored in a locus.

## Result

| fixture | exponential | hyperbolic | boxcar band | derived `[T2−T1+1, T2−1]` | direction |
|---|---|---|---|---|---|
| T1=4 T2=12 | 0/19 | 2/59 | (9, 11) | [9, 11] ✓ | **opposite** |
| T1=3 T2=10 | 0/19 | 49/59 | (8, 9) | [8, 9] ✓ | **opposite** |
| T1=6 T2=20 | 0/19 | 1/59 | (15, 19) | [15, 19] ✓ | **opposite** |

**1. The boxcar is not a discount — it reverses in the OPPOSITE direction.**
Hyperbolic goes `later → sooner` (patient far away, impatient up close — akrasia).
The boxcar goes `sooner → later`: blind at distance, then *more* patient as the
far reward enters the window. Proximity buys **visibility**, not impatience.

**2. It has no indifference region — analytically, not as a measured ratio.**
The boxcar's flip margin is **exactly `V2 − V1`** (the far reward jumps unbound →
full value while the near one sits at `V1`), so it is bounded below by the value
gap and *cannot* be small. Hyperbolic values cross **continuously**, so for any ε
there is a `k` whose flip margin is below ε. Hyperbolic admits indifference; the
boxcar cannot. A hard-horizon agent never *almost* sees a consequence, so it
cannot hedge and cannot be nudged.

⊘ An earlier revision quoted a **41–206×** margin ratio. That was an artifact of
the `k = i/10` sampling grid — a finer grid lands nearer a reversal boundary and
drives the denominator toward zero. Removed; the analytic statement is stronger
and needs no grid.

**3. At the A9 carrier's representable forward bound, distant rewards are simply
invisible — in every fixture.** A `Locus` is a signed **i4, `[−8, +7]`**, and `+` is
*after / consequent*; these fixtures place both rewards in the future, so the bound is
**7**. At `w=7` all three fixtures read all-`sooner` from every vantage: not impatience,
**uninformed**, with no way to report the difference.

**The window is asymmetric, and the wrong way round for a causal agent.** Two's
complement gives **8 steps backward** (antecedents, `−8`) and only **7 forward**
(consequents, `+7`): you can point one step further into the cause than into the
consequence.

## What it falsified

The session claim *"an A9-locus agent is constitutionally a scorpion"* is **false**.
The scorpion stings midstream — impatience rising with proximity — which is the
**hyperbolic** signature. A boxcar agent would have crossed: midstream is exactly
when the far shore becomes visible. Nature-as-curvature and nature-as-register-
width are different pathologies, and the substrate has the second one.

## Falsifiers (each verified)

- `exponential` reverses at **0/19** γ on all three fixtures — if it ever fires, the harness is broken.
- `hyperbolic` fires on all three — positive control.
- the measured boxcar band equals the derived `[T2−T1+1, T2−1]` on all three — asserted in-probe.
- **DISABLE**: `w = ∞` → reversals vanish. The horizon is load-bearing, not decorative.

## Instrument bug worth keeping

The first run printed *no* margin comparison and looked like a null result. Cause:
`flip_margin` scanned only for `sooner → later`, so it returned `None` for every
hyperbolic trace. **The direction it could not see was the finding.** Fixed to scan
both directions; the docstring records it. Third instance this session of *a null
result is a claim about the apparatus until proven otherwise.*

**A second correction, from review (CodeRabbit, PR #1233, Major).** The first
published run used `w=8` and reported a reversal on fixture 2 as the substrate's
real-width behaviour. Invalid: `+8` is not representable in i4, and that
unrepresentable offset was what produced the reversal. At the true forward bound
`w=7` **no fixture reverses** — which makes the blindness result uniform rather
than wrinkled. Taking the doc's "±8" phrasing at face value instead of reading
the carrier's own `[−8, +7]` range is what caused it.
