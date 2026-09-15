#!/usr/bin/env python3
"""
PROBE-HORIZON-WINDOW-1 -- is the A9 +-8 locus window a DISCOUNT, and of what kind?

The claim under test (operator, this session):
  "A boxcar isn't exponential or hyperbolic; the reversal comes from the window
   being relative to the reader's own position."

Canonical preference-reversal setup (fixed absolute positions, NOT a moving
reward -- that distinction is what makes the exponential control provable):

  SOONER-SMALLER reward  at absolute position T1, value V1
  LATER-LARGER   reward  at absolute position T2 > T1, value V2 > V1
  agent stands at tau, sees offsets (T1-tau) and (T2-tau), decides at each tau.

Three perception models over the SAME fixture:

  exponential(g)  v * g**offset            time-consistent -> NO reversal, provably.
                                           HARNESS CONTROL: any reversal here
                                           means the harness is broken.
  hyperbolic(k)   v / (1 + k*offset)       the classic reversal generator.
                                           POSITIVE CONTROL: must fire somewhere.
  boxcar(w)       v if offset <= w else 0  the A9 locus window. 0 = unbound =
                                           INVISIBLE, not discounted.

Loci are POINTERS, never magnitudes (operator-locked, six-families ruling), so
the window gates VISIBILITY of the pointer; the value is read at the target and
lives in a separate table here. Nothing in this probe stores a magnitude in a
locus register.

Deterministic, no corpus, no fetch, no RNG.
"""
import json
import os
from fractions import Fraction

# ---------------------------------------------------------------- perception
def see_boxcar(value, offset, w):
    """Hard horizon: full value inside, UNBOUND (None) outside. Not a discount."""
    return value if offset <= w else None

def see_exponential(value, offset, g):
    """HARNESS CONTROL. Exponential discounting is time-consistent: with the
    reward positions held FIXED in absolute time, the ratio of two discounted
    values is independent of the vantage tau, so it can never reverse. A
    reversal on this arm is a bug in the harness, not a finding about the
    agent -- which is what makes its measured 0/19 the evidence that the other
    two arms are reading real structure."""
    return value * (g ** offset)

def see_hyperbolic(value, offset, k):
    """POSITIVE CONTROL. Hyperbolic discounting is the textbook generator of
    preference reversal, and it reverses LATER -> SOONER: impatience rises as
    the near reward approaches. It fires on all three fixtures, so it proves
    the harness CAN see a reversal -- without it, the boxcar's own result would
    be indistinguishable from an instrument that detects nothing."""
    return value / (1.0 + k * offset)

# ---------------------------------------------------------------- the agent
def choose(t1, v1, t2, v2, tau, perceive):
    """Take the best VISIBLE reward. None = unbound (cannot be chosen)."""
    p1 = perceive(v1, t1 - tau)
    p2 = perceive(v2, t2 - tau)
    if p1 is None and p2 is None:
        return "abstain", None
    if p1 is None:
        return "later", p2
    if p2 is None:
        return "sooner", p1
    if p2 > p1:
        return "later", p2 - p1
    if p1 > p2:
        return "sooner", p1 - p2
    return "tie", 0.0

def trace(t1, v1, t2, v2, perceive):
    """Decisions over every vantage point tau the agent can still act from."""
    return [(tau,) + choose(t1, v1, t2, v2, tau, perceive) for tau in range(0, t1)]

def reverses(tr):
    """A reversal = the agent picks DIFFERENT rewards from different vantages.
    Abstention is not a choice and never counts as a reversal."""
    picks = {c for _, c, _ in tr if c in ("sooner", "later")}
    return len(picks) > 1

def flip_margin(tr):
    """(direction, margin) at the first flip, IN EITHER DIRECTION.

    An earlier version scanned only for sooner->later and silently returned
    None for hyperbolic, which reverses the other way -- the comparison then
    printed nothing and looked like a null result. It was a directional bug in
    the instrument, and the direction is itself the finding.

    Margin = |perceived_later - perceived_sooner| at the vantage just after the
    flip. A smooth crossing lands near 0; a discontinuity lands at a jump."""
    prev = None
    for _, c, m in tr:
        if c in ("sooner", "later"):
            if prev is not None and c != prev:
                return (f"{prev}->{c}", m)
            prev = c
    return None

# ---------------------------------------------------------------- the sweep
FIXTURES = [(4, 3.0, 12, 7.0), (3, 2.0, 10, 9.0), (6, 5.0, 20, 11.0)]
RESULT = {"fixtures": []}

for (T1, V1, T2, V2) in FIXTURES:
    rec = {"T1": T1, "V1": V1, "T2": T2, "V2": V2}
    print("=" * 74)
    print(f"fixture  sooner-smaller (t={T1}, v={V1})   later-larger (t={T2}, v={V2})")
    print(f"         agent decides from tau = 0 .. {T1-1}")

    # -- exponential: HARNESS CONTROL, must be silent at every g -------------
    exp_rev = [g for g in [Fraction(i, 20) for i in range(1, 20)]
               if reverses(trace(T1, V1, T2, V2, lambda v, o, g=g: see_exponential(v, o, g)))]
    rec["exponential_reversing_gammas"] = [str(g) for g in exp_rev]
    print(f"  exponential  reversals at {len(exp_rev)}/19 gammas   "
          f"{'OK (provable zero)' if not exp_rev else 'HARNESS BROKEN'}")

    # -- hyperbolic: POSITIVE CONTROL, must fire somewhere -------------------
    hyp_rev, hyp_margins = [], []
    for i in range(1, 60):
        k = i / 10.0
        tr = trace(T1, V1, T2, V2, lambda v, o, k=k: see_hyperbolic(v, o, k))
        if reverses(tr):
            hyp_rev.append(round(k, 2))
            fm = flip_margin(tr)
            if fm is not None:
                hyp_margins.append(fm)
    rec["hyperbolic_reversing_ks"] = hyp_rev
    rec["hyperbolic_flips"] = [[d, round(m, 4)] for d, m in hyp_margins]
    print(f"  hyperbolic   reversals at {len(hyp_rev)}/59 k values  "
          f"{'OK (fires)' if hyp_rev else 'CONTROL FAILED'}")

    # -- boxcar: the register under test ------------------------------------
    box_rev, box_margins = [], []
    for w in range(0, T2 + 6):
        tr = trace(T1, V1, T2, V2, lambda v, o, w=w: see_boxcar(v, o, w))
        if reverses(tr):
            box_rev.append(w)
            fm = flip_margin(tr)
            if fm is not None:
                box_margins.append((w, fm))
    rec["boxcar_reversing_w"] = box_rev
    rec["boxcar_flips"] = [[w, d, round(m, 4)] for w, (d, m) in box_margins]
    band = (min(box_rev), max(box_rev)) if box_rev else None
    rec["boxcar_band"] = band
    print(f"  boxcar       reversals at w in {box_rev}   band={band}")

    # derived band, asserted against the measurement
    lo, hi = T2 - T1 + 1, T2 - 1
    rec["boxcar_band_derived"] = [lo, hi]
    ok = band == (lo, hi)
    rec["boxcar_band_matches_derivation"] = ok
    print(f"  derived band [T2-T1+1, T2-1] = [{lo}, {hi}]   "
          f"{'MATCH' if ok else 'MISMATCH -- implementation wrong'}")

    # -- DISABLE: remove the horizon (w = inf). Reversals must vanish. -------
    tr_inf = trace(T1, V1, T2, V2, lambda v, o: see_boxcar(v, o, 10**9))
    rec["boxcar_disabled_reverses"] = reverses(tr_inf)
    print(f"  DISABLE w=inf -> reverses={reverses(tr_inf)}  "
          f"{'OK (horizon is load-bearing)' if not reverses(tr_inf) else 'VACUOUS'}")

    # -- the finding: the flip margin is CATEGORICALLY different -------------
    #
    # ⊘ CORRECTED 2026-09-15 (CodeRabbit, PR #1233). An earlier revision reported
    # a "margin ratio" of boxcar-min / hyperbolic-min. That number is an artifact
    # of the sampling grid, not a property of the two models: hyperbolic margins
    # are collected only at k = i/10, and a finer grid lands arbitrarily close to
    # a reversal boundary, driving the denominator toward 0 and the ratio toward
    # infinity. The ratio has been removed.
    #
    # The ANALYTIC distinction is stronger and needs no grid:
    #   boxcar      flip margin == V2 - V1 EXACTLY (far jumps unbound -> full
    #               value while near sits at V1). Bounded below by the value gap;
    #               it CANNOT be made small.
    #   hyperbolic  the perceived values cross CONTINUOUSLY, so for any epsilon
    #               there is a k whose flip margin is < epsilon.
    # So hyperbolic admits an indifference region and the boxcar cannot have one.
    # Raw margins are still recorded, labelled as grid-dependent.
    if box_margins and hyp_margins:
        bdirs = {d for _, (d, _) in box_margins}
        hdirs = {d for d, _ in hyp_margins}
        rec["boxcar_directions"] = sorted(bdirs)
        rec["hyperbolic_directions"] = sorted(hdirs)
        print(f"  DIRECTION    boxcar={sorted(bdirs)}   hyperbolic={sorted(hdirs)}"
              f"   {'OPPOSITE' if bdirs.isdisjoint(hdirs) else 'same'}")
        bmin = min(m for _, (_, m) in box_margins)
        hmin = min(m for _, m in hyp_margins)
        rec["min_flip_margin_boxcar_GRID_DEPENDENT"] = round(bmin, 4)
        rec["min_flip_margin_hyperbolic_GRID_DEPENDENT"] = round(hmin, 4)
        rec["boxcar_margin_analytic"] = round(V2 - V1, 4)
        rec["margin_distinction"] = (
            "categorical, not a ratio: boxcar flip margin == V2-V1 exactly and "
            "cannot be small; hyperbolic crosses continuously, so some k gives a "
            "margin below any epsilon. Sampled minima are grid artifacts."
        )
        print(f"  FLIP MARGIN  boxcar == V2-V1 == {V2-V1:.4f} ANALYTIC (cannot be small); "
              f"hyperbolic crosses continuously -> some k is arbitrarily close to 0")
        print(f"               (sampled minima, GRID-DEPENDENT: box={bmin:.4f} hyp={hmin:.4f}"
              f" -- do not quote as a ratio)")

    # -- the substrate's real width, and it is ASYMMETRIC --------------------
    #
    # ⊘ CORRECTED 2026-09-15 (CodeRabbit, PR #1233, Major). An earlier revision
    # used w=8 here and reported a reversal on fixture 2. That was INVALID: a
    # Locus is a signed i4, range [-8, +7], with `+` = after/consequent. This
    # fixture places both rewards in the FUTURE, so every offset is positive and
    # the representable bound is +7. The w=8 arm asked the register to hold a
    # +8 offset it cannot store, and that unrepresentable value was what produced
    # the reversal.
    #
    # The asymmetry is itself the finding, and it runs the wrong way for a causal
    # agent: two's complement gives 8 steps BACKWARD (antecedents, -8) and only 7
    # FORWARD (consequents, +7). You can point one step further into the cause
    # than into the consequence.
    tr7 = trace(T1, V1, T2, V2, lambda v, o: see_boxcar(v, o, 7))
    rec["w7_trace"] = [[tau, c, (round(m, 4) if m is not None else None)] for tau, c, m in tr7]
    rec["w7_reverses"] = reverses(tr7)
    rec["forward_reach_i4"] = 7
    rec["backward_reach_i4"] = 8
    print(f"  w=7 (A9 fwd) {[c for _, c, _ in tr7]}  reverses={reverses(tr7)}"
          f"   [i4 forward bound; backward is 8 -- the window is ASYMMETRIC]")
    RESULT["fixtures"].append(rec)

print("=" * 74)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "window-sweep.json")
json.dump(RESULT, open(out, "w"), indent=1)
print("wrote", out)
