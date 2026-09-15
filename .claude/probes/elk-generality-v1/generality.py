#!/usr/bin/env python3
"""MQ vs MONDO — is the meet/horizon behaviour SUBSTRATE or TAXONOMY-SHAPE?

Every number reported for the elk lens so far came from one graph: the MONDO
is_a spine. A taxonomy is the graph shape most likely to produce those numbers
trivially, so a generality control is not optional.

MQ = a Mississippi-Queen river course: positions flowing downstream, the channel
braiding and rejoining, bounded legal moves. It is a DAG like a taxonomy, but its
"ancestors" are UPSTREAM POSITIONS YOU COULD HAVE COME FROM, not generalisations,
and the meet of two positions is the CONFLUENCE rather than a common genus. Same
algebra, different semantics, different degree profile -- exactly what a control
needs to be.

BOTH graphs run through the SAME arm functions below. If the numbers agree, the
behaviour is the lens's; if they diverge, it was the taxonomy's all along.

RE-RUNNABILITY, stated honestly: the MQ arm is synthetic and seeded -- it runs
anywhere, forever. The MONDO arm needs obo-core.soa, a 31 MB binary bake that is
NOT committed to this repo (size, not policy -- the path is configurable via
OBO_CORE_SOA); that arm skips with a notice when the file is absent.
"""
import struct, random, math, os, sys
from collections import defaultdict

DEPTH_CAP = 64

# ── the arms (shared by both graphs — this is what makes the comparison valid) ──
def supers(parents, c, cap=DEPTH_CAP):
    """LensClosure::supers_of, KEEPING the depth the BFS already computes."""
    best = {}; q = [(c, 0)]
    while q:
        n, d = q.pop(0)
        if d >= cap: continue
        for p in parents.get(n, ()):
            if p not in best or d + 1 < best[p]:
                best[p] = d + 1; q.append((p, d + 1))
    return best

def supers_minmax(parents, c, cap=DEPTH_CAP):
    """Same walk, keeping BOTH the shortest and the longest path to each ancestor.

    `supers_of` keeps the MINIMUM. Where min == max for every ancestor the graph
    is locally tree-shaped and "shallow" is a faithful proxy for "general"; where
    they differ the DAG BRAIDS and a general ancestor can score shallow via a
    shortcut. That spread is the braid arm's independent variable.
    """
    mn = {}; mx = {}; q = [(c, 0)]
    while q:
        n, d = q.pop(0)
        if d >= cap: continue
        for p in parents.get(n, ()):
            depth = d + 1
            # Decide BEFORE writing. The first version updated mn/mx on the two
            # lines above the test, so the test then compared depth against the
            # value it had just written and was always false — only the FIRST
            # discovery of a node ever propagated, and a later longer path
            # widened mx[p] without pushing that depth to p's own ancestors.
            # That systematically UNDER-states spread, which is the variable
            # the A5 arm reads. Found by review on 43dbfde.
            changed = p not in mn or depth < mn[p] or depth > mx[p]
            mn[p] = min(mn.get(p, depth), depth)
            mx[p] = max(mx.get(p, depth), depth)
            if changed:
                q.append((p, depth))
    return mn, mx

def most_specific(parents, sset):
    """lens.rs: drop x when some other member y has x among its ancestors."""
    out = []
    for x in sset:
        if not any(y != x and x in supers(parents, y) for y in sset):
            out.append(x)
    return out

def arms(parents, seeds, label, n_pairs=200, seed=0xC0FFEE):
    """A0-A3 + the horizon sweep, run identically on whichever graph is passed in.

    THIS FUNCTION BEING SHARED IS THE WHOLE METHOD. Both graphs reach it through the
    same `parents` adjacency and the same seeded pair sampler, so any difference in
    the numbers it prints is a difference between the GRAPHS and cannot be a
    difference between two implementations of the measurement.

    A0 ancestry size · A1 meet size (with the true-EMPTY count, which must be
    width-invariant) · A3 the depth-rank agreement with `most_specific`, reported in
    BOTH sort directions because ranking descending selects the most GENERAL
    ancestor and was the sign error this probe exists to pin · then the horizon
    sweep, which separates CUT (a shared ancestor exists beyond `w`) from EMPTY
    (none exists at any width) -- two answers that a meet returning `0` conflates.
    """
    rng = random.Random(seed)
    pairs = [(rng.choice(seeds), rng.choice(seeds)) for _ in range(n_pairs)]
    pairs = [(a, b) for a, b in pairs if a != b]
    truth = {p: set(supers(parents, p[0])) & set(supers(parents, p[1])) for p in pairs}

    anc = [len(supers(parents, p[0])) for p in pairs]
    inter = [len(truth[p]) for p in pairs]
    agree_desc = agree_asc = both = 0
    for p in pairs:
        I = truth[p]
        if not I: continue
        A, B = supers(parents, p[0]), supers(parents, p[1])
        M = set(most_specific(parents, list(I))); both += 1
        if sorted(I, key=lambda x: (-max(A[x], B[x]), x))[0] in M: agree_desc += 1
        if sorted(I, key=lambda x: ( max(A[x], B[x]), x))[0] in M: agree_asc += 1

    print(f"--- {label}   nodes={len(parents):,}  pairs={len(pairs)}")
    print(f"    A0 mean |ancestors|   {sum(anc)/len(anc):.1f}   max {max(anc)}")
    print(f"    A1 mean |intersection|{sum(inter)/len(inter):.2f}  <=3 on "
          f"{sum(1 for x in inter if x<=3)}/{len(pairs)}  empty {sum(1 for x in inter if not x)}")
    print(f"    A3 depth DESCENDING   {agree_desc}/{both} = {100*agree_desc/max(both,1):.1f}%")
    print(f"    A3 depth ASCENDING    {agree_asc}/{both} = {100*agree_asc/max(both,1):.1f}%")

    print(f"    horizon sweep (CUT = a shared ancestor exists but sits beyond w):")
    print(f"      {'w':>4} {'mean|∩|':>8} {'EMPTY':>6} {'CUT':>5} {'recall':>8}")
    for cap in (1, 7, 14, 64):
        emp = cut = 0; rec = []
        for p in pairs:
            I = set(supers(parents, p[0], cap)) & set(supers(parents, p[1], cap))
            if not I:
                if truth[p]: cut += 1
                else: emp += 1
            if truth[p]: rec.append(len(I) / len(truth[p]))
        print(f"      {cap:>4} {sum(len(set(supers(parents,p[0],cap))&set(supers(parents,p[1],cap))) for p in pairs)/len(pairs):>8.2f}"
              f" {emp:>6} {cut:>5} {100*sum(rec)/len(rec):>7.1f}%")
    return {"asc": 100*agree_asc/max(both,1), "desc": 100*agree_desc/max(both,1)}

def arm_stratify(parents, seeds, label, n_pairs=200, seed=0xC0FFEE):
    """A4 — ANTI-VACUITY. At |I| == 1 the argmin IS the only most_specific member,
    so agreement there is arithmetic, not evidence. Report the |I| == 1 count and
    the agreement with those pairs excluded; also bucket by |I| so a "small
    intersections make it easy" confound would be VISIBLE rather than assumed."""
    rng = random.Random(seed)
    pairs = [(rng.choice(seeds), rng.choice(seeds)) for _ in range(n_pairs)]
    pairs = [(a, b) for a, b in pairs if a != b]
    buckets = defaultdict(lambda: [0, 0]); triv = 0
    for p in pairs:
        A, B = supers(parents, p[0]), supers(parents, p[1])
        I = set(A) & set(B)
        if not I: continue
        M = set(most_specific(parents, list(I)))
        hit = sorted(I, key=lambda x: (max(A[x], B[x]), x))[0] in M
        n = len(I)
        if n == 1: triv += 1
        b = "|I|=1 VACUOUS" if n == 1 else "|I|=2-3" if n <= 3 else "|I|=4-9" if n <= 9 else "|I|>=10"
        buckets[b][0] += hit; buckets[b][1] += 1
    print(f"    A4 stratified by |intersection| ({label}):")
    nt_a = nt_t = 0
    for b in ("|I|=1 VACUOUS", "|I|=2-3", "|I|=4-9", "|I|>=10"):
        if b not in buckets: continue
        a, t = buckets[b]
        print(f"       {b:<15} {a:>4}/{t:<4} = {100*a/t:5.1f}%")
        if b != "|I|=1 VACUOUS": nt_a += a; nt_t += t
    print(f"       NON-VACUOUS     {nt_a:>4}/{nt_t:<4} = {100*nt_a/max(nt_t,1):5.1f}%"
          f"   (dropped {triv} arithmetic pairs)")
    return 100 * nt_a / max(nt_t, 1)

def arm_braid(parents, seeds, label, n_pairs=200, seed=0xC0FFEE):
    """A5 — MECHANISM CANDIDATE. Split pairs by whether their shared ancestors are
    reached by paths of a SINGLE length (flat/tree-like) or several (braided). If
    braiding explains the divergence, the braided buckets should agree across the
    two graphs. Reported whether or not it does — a falsified mechanism is a
    result, not a failure."""
    rng = random.Random(seed)
    pairs = [(rng.choice(seeds), rng.choice(seeds)) for _ in range(n_pairs)]
    pairs = [(a, b) for a, b in pairs if a != b]
    fa = ft = ba = bt = 0; spreads = []
    for p in pairs:
        An, Ax = supers_minmax(parents, p[0]); Bn, Bx = supers_minmax(parents, p[1])
        I = set(An) & set(Bn)
        if not I: continue
        M = set(most_specific(parents, list(I)))
        hit = sorted(I, key=lambda x: (max(An[x], Bn[x]), x))[0] in M
        sp = sum((Ax[x] - An[x]) + (Bx[x] - Bn[x]) for x in I) / (2 * len(I))
        spreads.append(sp)
        if sp == 0: fa += hit; ft += 1
        else:       ba += hit; bt += 1
    print(f"    A5 braid split ({label}): mean path-length spread {sum(spreads)/len(spreads):.2f}")
    print(f"       FLAT  (spread == 0) {fa:>4}/{ft:<4} = "
          f"{(100*fa/ft) if ft else float('nan'):5.1f}%" if ft else
          "       FLAT  (spread == 0)   NONE — this graph braids everywhere")
    print(f"       BRAID (spread >  0) {ba:>4}/{bt:<4} = "
          f"{(100*ba/bt) if bt else float('nan'):5.1f}%" if bt else
          "       BRAID (spread >  0)   NONE — this graph is tree-like everywhere")
    return (100 * ba / bt) if bt else float("nan")

# ── MQ: a Mississippi-Queen river course ──────────────────────────────────────
def mq_river(rows=60, seed=7):
    """Positions flow downstream; the channel braids and rejoins; legal moves are
    bounded (a boat advances 1-3 and may shift one lane). parents = UPSTREAM."""
    rng = random.Random(seed)
    layers, width = [], 3
    for r in range(rows):
        width = max(1, min(6, width + rng.choice((-1, 0, 0, 1))))   # river narrows/widens
        layers.append([(r, c) for c in range(width)])
    parents = defaultdict(list)
    for r in range(rows - 1):
        for (rr, c) in layers[r]:
            for adv in (1, 2, 3):                                    # boat speed
                if r + adv >= rows: continue
                for dc in (-1, 0, 1):                                # lane shift
                    tgt = (r + adv, c + dc)
                    if tgt in layers[r + adv]:
                        parents[tgt].append((rr, c))                 # upstream link
    nodes = [n for L in layers for n in L]
    return parents, [n for n in nodes if parents.get(n)]

print("=" * 72)
print("MQ (Mississippi-Queen river course) — the generality control")
print("=" * 72)
p_mq, s_mq = mq_river()
mq = arms(p_mq, s_mq, "MQ river")
mq_nv = arm_stratify(p_mq, s_mq, "MQ")
mq_br = arm_braid(p_mq, s_mq, "MQ")

print()
print("=" * 72)
SOA = os.environ.get("OBO_CORE_SOA", "/home/user/MedCare-rs/.data/obo-core.soa")
if not os.path.exists(SOA):
    print(f"MONDO arm SKIPPED — {SOA} absent (31 MB bake, not committed; set OBO_CORE_SOA).")
    sys.exit(0)
print("MONDO is_a spine — the original measurement")
print("=" * 72)
STRIDE, EDGES_OFF, VALUE_OFF = 512, 16, 32
ET_OFF, LANE_OFF, LANE_B, LANES = 96, 112, 16, 23
MONDO_CLS = 0x91010000
def u24(b, o):
    """One 3-byte little-endian edge numeric from the bake, RAW (still +1 biased).

    The lane slots store `numeric + 1` so that a stored `0` can mean PAD rather than
    node 0 -- so every caller must subtract 1, and a slot reading 0 is skipped, never
    decoded. Returning the raw value here keeps that bias visible at the call site
    instead of hiding it in the accessor.
    """
    return b[o] | (b[o+1] << 8) | (b[o+2] << 16)
data = open(SOA, "rb").read()
if not data or len(data) % STRIDE:
    raise ValueError(
        "MONDO bake must be non-empty and a whole multiple of STRIDE; "
        "len(...) // STRIDE silently discards a partial trailing row, so an "
        "incomplete download would report measurements from a prefix."
    )
rows = len(data) // STRIDE
p_ont = defaultdict(list)
for r in range(rows):
    row = data[r*STRIDE:(r+1)*STRIDE]
    cls = struct.unpack_from("<I", row, 0)[0]
    fam, ident = struct.unpack_from("<HH", row, 12)
    subj = (cls, (fam << 16) | ident)
    degs = [row[EDGES_OFF+i] for i in range(8)]
    slots = []
    for l in range(LANES):
        lo = VALUE_OFF + LANE_OFF + l*LANE_B
        t = struct.unpack_from("<I", row, lo)[0]
        if t == 0: continue
        for s in range(4):
            st = u24(row, lo + 4 + s*3)
            if st: slots.append((t, st - 1))
    for _ in range(degs[1]):
        if slots: p_ont[subj].append(slots.pop(0))
s_ont = [n for n in p_ont if n[0] == MONDO_CLS and p_ont[n]]
ont = arms(p_ont, s_ont, "MONDO is_a")
ont_nv = arm_stratify(p_ont, s_ont, "MONDO")
ont_br = arm_braid(p_ont, s_ont, "MONDO")

print()
print("=" * 72)
print("VERDICT")
print(f"  depth ASCENDING   MQ {mq['asc']:.1f}%   MONDO {ont['asc']:.1f}%   "
      f"Δ {abs(mq['asc']-ont['asc']):.1f}pp")
print(f"  depth DESCENDING  MQ {mq['desc']:.1f}%   MONDO {ont['desc']:.1f}%")
print(f"  NON-VACUOUS       MQ {mq_nv:.1f}%   MONDO {ont_nv:.1f}%   "
      f"Δ {abs(mq_nv-ont_nv):.1f}pp   (A4: does |I|==1 arithmetic carry it?)")
print(f"  BRAIDED PAIRS ONLY MQ {mq_br:.1f}%   MONDO {ont_br:.1f}%   "
      f"Δ {abs(mq_br-ont_br):.1f}pp   (A5: does braiding explain it?)")
print()
print("  PRE-REGISTERED READING: if ASCENDING agrees across both, the ranking")
print("  result is the LENS's. If it diverges, it was the taxonomy's shape.")
print("  A4 and A5 are the two confounds that could fake a divergence; a gap")
print("  that SURVIVES both is structural and its mechanism is still unnamed.")
