#!/usr/bin/env python3
"""Density sweep — does the depth-rank divergence track ANCESTRY DENSITY, and does
a bounded frontier (the torch) hold its recall out to 12 hops?

THREE questions, one axis, because they share an independent variable:

  Q1  ISS-ELK-DENSITY-UNISOLATED. MQ (36.0%) and MONDO (84.6%) differ by 48.6pp on
      depth-rank-vs-most_specific, and the gap survived both confounds (A4 vacuity,
      A5 braiding). MQ was built as a DIFFERENT graph, not a density sweep, so it
      proved non-universality without naming the cause. If agreement tracks density
      monotonically, density is the property and the claim becomes a checkable
      precondition: "depth-rank works while ancestry stays sparse."

  Q2  The torch's anti-vacuity arm. A bounded-k frontier only works if truncation is
      cheap. Measured across densities: if bounded-k recall is high EVERYWHERE the
      result is about these graphs; it must DEGRADE somewhere or it says nothing.

  Q3  The amortization band. Masking pays for longer reach only if bounded recall
      holds AS DISTANCE GROWS. Measured at hops 4 / 8 / 12 rather than at 12 alone.

DELIBERATELY NOT MEASURED: "top-k successor mass" on these unweighted graphs. That
is the D-HXP-1 SIGNAL half, STRUCK as unanswerable by that instrument --
`uniform_expected = min(6,d)/d` is biased by small-sample concentration, and on an
unweighted graph every successor carries equal mass so the statistic is arithmetic.
Bounded-k RECALL has no such bias: truncating a 6,297-wide frontier to 50 either
costs reachability or it does not.
"""
import random, struct, os, sys
from collections import defaultdict

CAP = 64

def supers(parents, c, cap=CAP):
    """LensClosure::supers_of, keeping the BFS depth (min over paths)."""
    best = {}; q = [(c, 0)]
    while q:
        n, d = q.pop(0)
        if d >= cap: continue
        for p in parents.get(n, ()):
            if p not in best or d + 1 < best[p]:
                best[p] = d + 1; q.append((p, d + 1))
    return best

def most_specific(parents, s):
    """lens.rs: drop x when some other member y has x among its ancestors."""
    return [x for x in s if not any(y != x and x in supers(parents, y) for y in s)]

def children_of(parents):
    """Invert to the DOWNWARD adjacency the frontier walk needs."""
    ch = defaultdict(list)
    for node, ps in parents.items():
        for p in ps:
            ch[p].append(node)
    return ch

def reach_unbounded(ch, seed, hops):
    """Level-synchronous BFS keeping EVERY newly discovered node — the hdr_bfs shape.
    Returns (reached_set, peak_frontier_width)."""
    seen = {seed}; frontier = [seed]; peak = 1
    for _ in range(hops):
        nxt = []
        for n in frontier:
            for c in ch.get(n, ()):
                if c not in seen:
                    seen.add(c); nxt.append(c)
        if not nxt: break
        frontier = nxt; peak = max(peak, len(frontier))
    return seen, peak

def reach_bounded(ch, seed, hops, k, rng):
    """The TORCH: same walk, frontier truncated to k after every advance.
    Truncation is a uniform random sample, NOT a ranked top-k — these graphs carry no
    weights, so any ranking I invented would be the thing under test. A random bound is
    the conservative choice: a ranked bound can only do better."""
    seen = {seed}; frontier = [seed]
    for _ in range(hops):
        nxt = []
        for n in frontier:
            for c in ch.get(n, ()):
                if c not in seen:
                    seen.add(c); nxt.append(c)
        if not nxt: break
        frontier = nxt if len(nxt) <= k else rng.sample(nxt, k)
    return seen

def mq_river(rows, max_width, advances, shifts, seed=7):
    """MQ with the density knobs exposed: fewer speeds / narrower lane shift / a
    narrower river all reduce in-degree, hence ancestry density."""
    rng = random.Random(seed)
    layers, w = [], min(3, max_width)
    for r in range(rows):
        w = max(1, min(max_width, w + rng.choice((-1, 0, 0, 1))))
        layers.append([(r, c) for c in range(w)])
    par = defaultdict(list)
    for r in range(rows - 1):
        for (rr, c) in layers[r]:
            for adv in advances:
                if r + adv >= rows: continue
                for dc in shifts:
                    t = (r + adv, c + dc)
                    if t in layers[r + adv]:
                        par[t].append((rr, c))
    nodes = [n for L in layers for n in L]
    return par, [n for n in nodes if par.get(n)]

def measure(parents, seeds, label, n_pairs=200, n_seeds=60, pair_seed=0xC0FFEE):
    """Run all three questions against ONE graph and return its row.

    Q1 draws `n_pairs` seed pairs (seeded, so two graphs get the SAME pair sequence)
    and scores depth-rank-ascending against `most_specific`; pairs with an empty
    ancestor intersection are counted out of the denominator, never as agreement.
    Q2/Q3 walk `n_seeds` seeds bounded and unbounded at hops 4/8/12 and report
    recall of the bounded walk against the unbounded one.

    `density` is mean ancestry size over graph size -- the variable the sweep was
    built to isolate, and the one it FALSIFIED (see the controlled arm below).
    """
    rng = random.Random(pair_seed)
    n_nodes = len({n for n in parents} | {p for ps in parents.values() for p in ps})

    # --- Q1: depth-rank ascending vs most_specific ---
    pairs = [(rng.choice(seeds), rng.choice(seeds)) for _ in range(n_pairs)]
    pairs = [(a, b) for a, b in pairs if a != b]
    anc_sizes, agree, both = [], 0, 0
    for p in pairs:
        A, B = supers(parents, p[0]), supers(parents, p[1])
        anc_sizes.append(len(A))
        I = set(A) & set(B)
        if not I: continue
        both += 1
        if sorted(I, key=lambda x: (max(A[x], B[x]), x))[0] in set(most_specific(parents, list(I))):
            agree += 1
    mean_anc = sum(anc_sizes) / len(anc_sizes)
    density = mean_anc / n_nodes
    asc = 100 * agree / max(both, 1)

    # --- Q2/Q3: bounded-k recall vs distance ---
    ch = children_of(parents)
    walk_seeds = [s for s in seeds if ch.get(s)]
    walk_seeds = random.Random(11).sample(walk_seeds, min(n_seeds, len(walk_seeds)))
    rows_out, peak12 = {}, 0
    for hops in (4, 8, 12):
        for k in (6, 50):
            tot = 0.0
            for s in walk_seeds:
                truth, peak = reach_unbounded(ch, s, hops)
                if hops == 12: peak12 = max(peak12, peak)
                got = reach_bounded(ch, s, hops, k, random.Random(hash((s, k, hops)) & 0xFFFF))
                tot += len(got & truth) / max(len(truth), 1)
            rows_out[(hops, k)] = 100 * tot / len(walk_seeds)

    print(f"{label:<22} n={n_nodes:>6}  mean|anc|={mean_anc:>6.1f}  density={100*density:>6.2f}%"
          f"  ASC={asc:>5.1f}%  peak_frontier@12={peak12:>6}")
    print(f"{'':<22}   bounded-k recall   "
          + "  ".join(f"h{h}/k{k}={rows_out[(h,k)]:5.1f}%" for h in (4, 8, 12) for k in (6, 50)))
    return {"density": 100 * density, "asc": asc, "recall": rows_out, "peak": peak12, "n": n_nodes}

print("=" * 108)
print("DENSITY SWEEP — one axis, three questions (Q1 depth-rank cause · Q2 torch anti-vacuity · Q3 amortization)")
print("=" * 108)

CONFIGS = [
    ("MQ sparse-1",  dict(rows=60, max_width=2, advances=(1,),     shifts=(0,))),
    ("MQ sparse-2",  dict(rows=60, max_width=3, advances=(1,),     shifts=(-1, 0, 1))),
    ("MQ mid-1",     dict(rows=60, max_width=4, advances=(1, 2),   shifts=(0,))),
    ("MQ mid-2",     dict(rows=60, max_width=5, advances=(1, 2),   shifts=(-1, 0, 1))),
    ("MQ dense-1",   dict(rows=60, max_width=6, advances=(1, 2, 3), shifts=(0,))),
    ("MQ dense-2",   dict(rows=60, max_width=6, advances=(1, 2, 3), shifts=(-1, 0, 1))),  # the original
]
res = []
for name, kw in CONFIGS:
    par, seeds = mq_river(**kw)
    if len(seeds) < 10:
        print(f"{name:<22} SKIPPED — only {len(seeds)} seeds"); continue
    res.append((name, measure(par, seeds, name)))

SOA = os.environ.get("OBO_CORE_SOA", "/home/user/MedCare-rs/.data/obo-core.soa")
if os.path.exists(SOA):
    print()
    STRIDE, EOFF, VOFF, LOFF, LB, LN = 512, 16, 32, 112, 16, 23
    MONDO = 0x91010000
    def u24(b, o):
        """Read a 3-byte LITTLE-endian unsigned int at offset `o`.

        Endianness is named because it is not inferable from the width: this
        session measured a join where reading the same bytes big-endian returns
        a plausible wrong answer that passes an ordering check.
        """
        return b[o] | (b[o+1] << 8) | (b[o+2] << 16)
    d = open(SOA, "rb").read()
    if not d or len(d) % STRIDE:
        raise ValueError(
            "MONDO bake must be non-empty and a whole multiple of STRIDE; "
            "len(...) // STRIDE silently discards a partial trailing row, so an\n"
            "incomplete download would report measurements from a prefix."
        )
    po = defaultdict(list)
    for r in range(len(d) // STRIDE):
        row = d[r*STRIDE:(r+1)*STRIDE]
        cls = struct.unpack_from("<I", row, 0)[0]
        fam, idt = struct.unpack_from("<HH", row, 12)
        subj = (cls, (fam << 16) | idt)
        degs = [row[EOFF+i] for i in range(8)]; sl = []
        for l in range(LN):
            lo = VOFF + LOFF + l*LB
            t = struct.unpack_from("<I", row, lo)[0]
            if t == 0: continue
            for kk in range(4):
                st = u24(row, lo + 4 + kk*3)
                if st: sl.append((t, st - 1))
        for _ in range(degs[1]):
            if sl: po[subj].append(sl.pop(0))
    res.append(("MONDO is_a", measure(po, [n for n in po if n[0] == MONDO and po[n]], "MONDO is_a")))
else:
    print(f"\nMONDO arm SKIPPED — {SOA} absent (31 MB bake, not committed; set OBO_CORE_SOA).")

print()
print("=" * 108)
print("CONTROLLED ARM — width=6 and shifts=(-1,0,1) FIXED; advances varied ALONE")
print("In the sweep above width COVARIED with advances (2,3 / 4,5 / 6,6), so reading")
print("'advance multiplicity is the variable' off it would repeat the exact confound the")
print("density hypothesis just died of. Here density is pinned and advances move alone.")
print("=" * 108)
ctrl = []
for advs in [(1,), (2,), (1, 2), (1, 3), (1, 2, 3), (1, 2, 3, 4)]:
    par, seeds = mq_river(rows=60, max_width=6, advances=advs, shifts=(-1, 0, 1))
    if len(seeds) < 10:
        print(f"  advances={advs} SKIPPED ({len(seeds)} seeds)"); continue
    ctrl.append((advs, measure(par, seeds, f"adv={advs}")))

print()
print("ORDERED BY |advances| — the multiplicity of distinct path lengths:")
for advs, r in sorted(ctrl, key=lambda x: (len(x[0]), x[0])):
    print(f"   |adv|={len(advs)}  advances={str(advs):<14} density={r['density']:>6.2f}%  ASC={r['asc']:>5.1f}%")
dens = {round(r["density"], 2) for _, r in ctrl if len(_) > 1 or _ == (1,)}
print()
print("THE TWO CONTRASTS THAT DECIDE IT:")
print("  (a) density PINNED, multiplicity varied — if agreement still moves, density is not the cause.")
print("  (b) advances=(1,) vs (2,) — |adv|=1 in BOTH, but density differs ~2x. If agreement is")
print("      identical, density is inert at fixed multiplicity.")
print("  (c) (1,2) vs (1,3) — SAME |adv|, same density, different SPAN of path lengths.")
print("      A gap here says it is the SPREAD, not the count.")

print()
print("=" * 108)
print("Q1 — does depth-rank agreement track ancestry density?")
for name, r in sorted(res, key=lambda x: x[1]["density"]):
    print(f"   density {r['density']:>6.2f}%   ASC {r['asc']:>5.1f}%   {name}")
d0, d1 = sorted(res, key=lambda x: x[1]["density"])[0], sorted(res, key=lambda x: x[1]["density"])[-1]
print(f"   span: density {d0[1]['density']:.2f}% -> {d1[1]['density']:.2f}%, "
      f"ASC {d0[1]['asc']:.1f}% -> {d1[1]['asc']:.1f}%")
