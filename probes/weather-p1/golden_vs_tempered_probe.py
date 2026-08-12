"""golden-vs-tempered-stride-v1 -- T1-T4, pre-registered in the plan file
(.claude/plans/golden-vs-tempered-stride-v1.md), run here to independently
validate the hand-derived numbers already committed there.

WHY THIS SCRIPT EXISTS. The plan's T1-T4 tables were computed via scratch
Python during the design session and copied into the plan as pre-registered
expectations. This script is the COMMITTED, INDEPENDENT reproduction: same
method, freshly executed, checkpointed, with no hand-editing of the output.
If a number here disagrees with the plan's table, the plan's table is wrong
and gets corrected -- this script is the source of truth going forward.

SCOPE: zero fetch, pure arithmetic. No network. Deterministic (no RNG at all
in T1/T3/T4; nothing here needs a seed).
"""
import json
import pathlib
import statistics
from math import gcd, log2

PHI = (1 + 5 ** 0.5) / 2
GOLDEN_FRAC = 2 - PHI  # = 1/phi^2, the golden-angle fraction in turns


def star_discrepancy(pts):
    """Star discrepancy D*(P) of a finite point set P in [0,1) (1-D form):
    max over i of |i/n - x_(i)| and |(i+1)/n - x_(i)| on the sorted sample.
    This is the standard low-discrepancy-sequence quality metric (Niederreiter).
    """
    p = sorted(pts)
    n = len(p)
    d = 0.0
    for i, x in enumerate(p):
        d = max(d, abs((i + 1) / n - x), abs(i / n - x))
    return d


def golden_pts(m):
    """First m points of the golden-angle 1-D equidistribution sequence."""
    return [(i * GOLDEN_FRAC) % 1.0 for i in range(m)]


def tempered_pts(m, s, q):
    """First m points of the coprime stride-s walk mod q, as fractions of q."""
    return [((s * i) % q) / q for i in range(m)]


def best_coprime_stride(q):
    """The coprime stride s in [1,q) minimizing the MEDIAN star discrepancy
    over the 'useful' prefix range m in [ceil(q/2), q] -- excludes the
    degenerate tiny-m cases (m=2 is trivially discrepant for any stride) that
    dominate a naive worst-case-over-all-m metric into near-uselessness.
    Returns (score, stride).
    """
    lo = max(2, q // 2)
    best = None
    for s in range(1, q):
        if gcd(s, q) != 1:
            continue
        sc = statistics.median(
            star_discrepancy(tempered_pts(m, s, q)) for m in range(lo, q + 1)
        )
        if best is None or sc < best[0]:
            best = (sc, s)
    return best


def t1_crossover(q_list):
    """T1: for each q, find the best coprime stride (useful-range metric),
    the matching golden score in the same range, and m* -- the first prefix
    length beyond q where golden's discrepancy permanently drops below the
    tempered stride's frozen m=q value.
    """
    rows = []
    for q in q_list:
        temp_score, s = best_coprime_stride(q)
        lo = max(2, q // 2)
        gold_score = statistics.median(
            star_discrepancy(golden_pts(m)) for m in range(lo, q + 1)
        )
        temp_frozen = star_discrepancy(tempered_pts(q, s, q))
        m_star = None
        for m in range(q, 20 * q + 1):
            if star_discrepancy(golden_pts(m)) < temp_frozen:
                m_star = m
                break
        m_big = 200 * q
        ratio_big = star_discrepancy(golden_pts(m_big)) and (
            temp_frozen / star_discrepancy(golden_pts(m_big))
        )
        rows.append({
            "q": q, "best_stride": s,
            "temp_score_useful_range": temp_score,
            "golden_score_useful_range": gold_score,
            "m_star": m_star,
            "temp_over_golden_at_200q": ratio_big,
        })
    return rows


def t2_asymptotic_bar(t1_rows):
    """T2: pass/fail -- at m=200q, golden discrepancy < tempered's frozen
    m=q value, for EVERY q tested. Reuses T1's rows (no recomputation)."""
    ok = all(r["temp_over_golden_at_200q"] > 1.0 for r in t1_rows)
    return {"bar": "golden < temp_frozen at m=200q, for every tested q",
            "pass": ok,
            "per_q": [(r["q"], r["temp_over_golden_at_200q"]) for r in t1_rows]}


def t3_closure_occupancy(q, phases):
    """T3: at m=q (tempered's own full cycle), count empty bins for both
    walks under q equal-width cells, checked at several bin-phase offsets to
    rule out a binning artifact.

    TWO DIFFERENT VERIFICATION METHODS, DELIBERATELY, per the SAME lesson
    caught mid-development of this script: a naive float round-trip
    (k/q then *q then int()) truncates values like 46.99999999999999 to 46
    instead of 47 -- a pure IEEE-754 rounding artifact that produced a FALSE
    fill-count deficit for the tempered walk even though its bijection is a
    mathematical PROOF (coprimality => {s*i mod q} = {0..q-1} exactly, no
    floats involved at all). Fixed: the tempered check uses EXACT INTEGER
    arithmetic (`(s*i) % q`, never divided then re-multiplied), so it cannot
    have this artifact -- it either equals q always (as proven) or the proof
    itself would be wrong, which it is not. The golden check legitimately
    needs floats (its positions are inherently continuous), so the
    phase-offset sweep stays meaningful there -- it is the actual empirical
    question, not a verification of an existing proof.
    """
    s = best_coprime_stride(q)[1]
    temp_fill_exact = len(set((s * i) % q for i in range(q)))  # proof-checking; no floats
    gold_fill_by_phase = []
    for off in phases:
        gold_bins = set(int(((i * GOLDEN_FRAC) % 1.0 + off) % 1.0 * q) for i in range(q))
        gold_fill_by_phase.append(len(gold_bins))
    return {
        "q": q, "stride": s, "phases": phases,
        "temp_fill_exact_integer": temp_fill_exact,
        "golden_fill_by_phase": gold_fill_by_phase,
        "temp_always_full": temp_fill_exact == q,
        "golden_ever_short": any(f < q for f in gold_fill_by_phase),
    }


def t4_naive_rounding_collapse(q_lo, q_hi):
    """T4: sweep q in [q_lo, q_hi), round(golden_frac * q) with NO
    coprimality check, count how often gcd(s,q) > 1 (the walk collapses to
    fewer than q distinct cells)."""
    collapses = []
    total = 0
    for q in range(q_lo, q_hi):
        total += 1
        s = round(GOLDEN_FRAC * q)
        if s == 0:
            s = 1
        g = gcd(s, q)
        if g > 1:
            collapses.append({"q": q, "s": s, "gcd": g, "cells_reached": q // g})
    return {
        "q_range": [q_lo, q_hi], "total_q_tested": total,
        "n_collapsing": len(collapses),
        "collapse_rate": len(collapses) / total,
        "examples": collapses[:8],
    }


def run():
    """Run T1-T4 in order, checkpoint each to the .partial.jsonl, then write
    the final combined JSON. Deterministic -- no seed needed anywhere."""
    out_dir = pathlib.Path(__file__).parent
    partial = out_dir / "golden_vs_tempered_probe.partial.jsonl"
    with open(partial, "w") as pf:
        q_list = [12, 17, 34, 55, 64, 89, 144, 233, 377, 987]
        t1 = t1_crossover(q_list)
        pf.write(json.dumps({"stage": "T1", "rows": t1}) + "\n")
        pf.flush()

        t2 = t2_asymptotic_bar(t1)
        pf.write(json.dumps({"stage": "T2", "result": t2}) + "\n")
        pf.flush()

        t3 = t3_closure_occupancy(140, [0.0, 0.1, 0.37, 0.5, 0.83])
        t3_aside_144 = t3_closure_occupancy(144, [0.0, 0.1, 0.37, 0.5, 0.83])
        pf.write(json.dumps({"stage": "T3", "headline_q140": t3,
                              "aside_q144_fibonacci": t3_aside_144}) + "\n")
        pf.flush()

        t4 = t4_naive_rounding_collapse(8, 300)
        pf.write(json.dumps({"stage": "T4", "result": t4}) + "\n")
        pf.flush()

    out = {
        "T1": t1,
        "T2": t2,
        "T3": {"headline_q140": t3, "aside_q144_fibonacci": t3_aside_144},
        "T4": t4,
    }
    with open(out_dir / "golden_vs_tempered_probe.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    result = run()
    print("=== T1 crossover ===")
    for r in result["T1"]:
        print(f"  q={r['q']:5d} s={r['best_stride']:5d} "
              f"temp={r['temp_score_useful_range']:.4f} "
              f"gold={r['golden_score_useful_range']:.4f} "
              f"m*={r['m_star']} temp/gold@200q={r['temp_over_golden_at_200q']:.1f}x")
    print("\n=== T2 asymptotic bar ===", "PASS" if result["T2"]["pass"] else "FAIL")
    print("\n=== T3 closure (q=140) ===")
    print("  temp fill (exact integer):", result["T3"]["headline_q140"]["temp_fill_exact_integer"])
    print("  gold fill by phase:", result["T3"]["headline_q140"]["golden_fill_by_phase"])
    print("=== T3 aside (q=144, Fibonacci) ===")
    print("  gold fill by phase:", result["T3"]["aside_q144_fibonacci"]["golden_fill_by_phase"])
    print("\n=== T4 naive-rounding collapse ===")
    print(f"  {result['T4']['n_collapsing']}/{result['T4']['total_q_tested']} = "
          f"{100*result['T4']['collapse_rate']:.1f}% collapse")
    print("  examples:", result["T4"]["examples"])
