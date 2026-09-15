"""seriation_quality.py -- ONE question: is the seriated atom chain built in
w1_cue2.py actually good -- i.e. do ADJACENT chain positions hold
behaviourally SIMILAR atoms?

Why this matters: the W1 run measured that soft-matching on tile ADJACENCY
makes retrieval WORSE, not better. That null is only meaningful if "adjacent"
actually means "similar". If the greedy seriation is near-random past the
first few positions, then adjacency is a false label and the null is about
the chain, not about adjacency. This script settles which, with numbers.

REUSE, NOT RE-DERIVATION: the corpus load, train-only atom vocabulary V, the
per-atom context profiles ctx, the fisher_z() function, and the greedy
seriation loop below are copied VERBATIM from w1_cue2.py's "THE TILE"
section (and the corpus-load / atoms-build lines immediately before it in
that file). Nothing about the seriation itself is re-derived or "improved"
-- the object under test is the exact chain that w1_cue2.py's actual run
built and used. (w1_cue2.py's ARM 0 positive-control block, and its THE
UNIT / cell-map / LUT / scorer / falsifier sections, are NOT needed to
build V / ctx / the chain -- they are independent of this object and are
skipped here purely for wall-clock cost; skipping them changes nothing
about the chain, since ARM 0 mutates no global that V/ctx/chain depend on
and the chain-building loop itself calls no RNG.)

Pure stdlib, no numpy/sklearn. PYTHONHASHSEED=0 required (asserted below,
same as w1_cue2.py) -- though note fisher_z(a, b) == fisher_z(b, a) exactly
(the union-of-keys iteration order cancels out of the Pearson-r arithmetic),
so hash-seed does not actually change any numeric result here; it is kept
for exact parity with the source script's invocation contract.
"""
import os as _os
assert _os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"

import collections
import json
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus2
import metrics2

S = "/tmp/claude-0/-home-user/7a79cf76-e191-5165-86a0-e4f23ef5c7db/scratchpad"
LAB = os.path.join(S, "lab")
RUNS = os.path.join(S, "runs")
os.makedirs(RUNS, exist_ok=True)

T0 = time.time()
LOG_LINES = []


def log(msg):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def flush_log():
    with open(os.path.join(RUNS, "w1-seriation.log"), "w") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# VERBATIM from w1_cue2.py -- corpus load + atoms build (lines 113-127 there),
# identical code, same S constant, same call sequence, so V/ctx/chain below
# are built from the exact same train split and atom vocabulary that run used.
# ═════════════════════════════════════════════════════════════════════════════
log("loading corpus2.Corpus2(ore-full-v2) ...")
c = corpus2.Corpus2(os.path.join(S, "ore-full-v2"))
train, test = c.split_class()
trs, tes = set(train), set(test)
pool = train + test
pairs = c.pairs(pool)
log(f"train {len(train)} test {len(test)} pool {len(pool)} pairs {len(pairs)}")

log("building atoms via c.view_ast(pool), level=struct ...")
va = c.view_ast(pool)
atoms, resid = metrics2.atoms_of(va)
log(f"atoms built for {len(atoms)} gids")


# ═════════════════════════════════════════════════════════════════════════════
# VERBATIM from w1_cue2.py "THE TILE" section (lines 194-259 there): V, ctx,
# fisher_z(), and the greedy seriation loop. Byte-mapping (byte_of / cell
# maps / LUT / spread / scorers) is NOT reproduced -- it plays no part in
# building the chain itself and is irrelevant to this measurement.
# ═════════════════════════════════════════════════════════════════════════════
log("=" * 78)
log("THE TILE -- axis ordering (seriation) on TRAIN atom vocabulary (verbatim reuse)")

V = sorted({x for g in train for x in atoms[g]})
log(f"len(V) = {len(V)} (distinct atom strings across train gids)")

# ctx[a]: Counter over "L:"+left_atom / "R:"+right_atom, immediate neighbours,
# window 1, every occurrence of atom a in TRAIN sequences only.
ctx = {a: collections.Counter() for a in V}
for g in train:
    a = atoms[g]
    for i, x in enumerate(a):
        cx = ctx.get(x)
        if cx is None:
            continue  # cannot happen (x in V by construction) -- defensive
        if i > 0:
            cx["L:" + a[i - 1]] += 1
        if i < len(a) - 1:
            cx["R:" + a[i + 1]] += 1


def fisher_z(a, b):
    """Pearson r over the union of ctx keys (missing = 0), then Fisher-Z.
    Mirrors lance-graph helix::fisher_z::Similarity::fisher_z. Degenerate case
    (fewer than 2 shared/union keys, or a zero-variance context vector) has no
    well-defined correlation -- the brief doesn't specify this, so it is
    treated as the *least* similar (z = -1.0 sentinel), which only ever
    matters as a last-resort tie-break in the seriation below (such an atom
    only ever gets picked once every better-defined candidate is exhausted)."""
    ca, cb = ctx[a], ctx[b]
    keys = set(ca) | set(cb)
    if len(keys) < 2:
        return -1.0
    xs = [ca.get(k, 0) for k in keys]
    ys = [cb.get(k, 0) for k in keys]
    nk = len(xs)
    mx = sum(xs) / nk
    my = sum(ys) / nk
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return -1.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = sxy / math.sqrt(sx * sy)
    r = max(-1 + 1e-9, min(1 - 1e-9, r))
    return 0.5 * (math.log(1 + r) - math.log(1 - r))


t0 = time.time()
remaining = set(V)
chain_start = min(V)  # lexicographically smallest atom
chain = [chain_start]
remaining.discard(chain_start)
while remaining:
    cur = chain[-1]
    best, best_z = None, None
    # deterministic: iterate candidates lexicographically ascending, replace
    # only on a STRICTLY higher z -- so among ties the lexicographically
    # smallest candidate wins (first one seen keeps the record).
    for cand in sorted(remaining):
        z = fisher_z(cur, cand)
        if best_z is None or z > best_z:
            best_z, best = z, cand
    chain.append(best)
    remaining.discard(best)
log(f"seriation chain built: {len(chain)} atoms ({time.time() - t0:.2f}s)")

POS = {a: i for i, a in enumerate(chain)}
N = len(chain)
assert N == len(V), "chain must contain every atom in V exactly once"


# ═════════════════════════════════════════════════════════════════════════════
# NEW MEASUREMENT CODE (this file's actual job) starts here. Everything above
# this line is the reused object under test.
# ═════════════════════════════════════════════════════════════════════════════

RESULT = {
    "meta": dict(
        n_train=len(train), n_test=len(test), n_pool=len(pool), n_pairs=len(pairs),
        len_V=len(V), chain_len=N,
    ),
    "m1_neighbour_quality_vs_random": {},
    "m2_topk_recovery": {},
    "m3_monotonicity_decay": {},
    "m4_where_it_breaks": {},
    "degenerate_pairs": {},
}
json.dump(RESULT, open(os.path.join(RUNS, "w1-seriation.json"), "w"), indent=1)

# ── full pairwise Fisher-z matrix over chain POSITIONS (0..N-1), built once ──
# Z[i][j] = fisher_z(chain[i], chain[j]) = fisher_z(chain[j], chain[i]) exactly
# (verified symmetric by construction of the Pearson-r formula: swapping which
# atom's Counter feeds xs vs ys just swaps the pointwise multiplicands in the
# sxy sum, which is commutative, and sx*sy is order-independent regardless).
# Cheap: N*(N-1)/2 calls, each O(|union of two small Counters|).
log("=" * 78)
log("building full pairwise Fisher-z matrix over all chain positions (for M2/M3/M4 + "
    "degenerate-pair census) ...")
t0 = time.time()
Z = [[0.0] * N for _ in range(N)]
n_degenerate = 0
total_pairs = N * (N - 1) // 2
for i in range(N):
    for j in range(i + 1, N):
        z = fisher_z(chain[i], chain[j])
        Z[i][j] = z
        Z[j][i] = z
        if z == -1.0:
            n_degenerate += 1
log(f"pairwise matrix built ({time.time() - t0:.2f}s); "
    f"{n_degenerate}/{total_pairs} pairs degenerate (z == -1.0 sentinel)")

degenerate_frac = n_degenerate / total_pairs
RESULT["degenerate_pairs"] = dict(
    count=n_degenerate, total_pairs=total_pairs, fraction=round(degenerate_frac, 4),
)
log(f"DEGENERATE-PAIR FRACTION: {n_degenerate}/{total_pairs} = {degenerate_frac:.4f}")
if degenerate_frac >= 0.25:
    log("NOTE: degenerate fraction is large (>=25%) -- M1-M4 below are partly ordering "
        "mostly-undefined (-1.0 sentinel) scores; read them in that light.")

overall_mean_z = statistics.mean(Z[i][j] for i in range(N) for j in range(i + 1, N))
log(f"overall mean pairwise Fisher-z (all {total_pairs} atom pairs): {overall_mean_z:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# M1 -- neighbour quality vs random shuffle null
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("M1 -- adjacent-pair Fisher-z vs random-shuffle null")

adjacent_zs = [Z[i][i + 1] for i in range(N - 1)]
real_mean = statistics.mean(adjacent_zs)
real_median = statistics.median(adjacent_zs)
log(f"real chain: mean adjacent-pair z = {real_mean:.4f}, median = {real_median:.4f} "
    f"(n_links={len(adjacent_zs)})")

NULL_SEEDS = list(range(20))
null_means = []
for s in NULL_SEEDS:
    rng = random.Random(s)
    shuffled = chain[:]
    rng.shuffle(shuffled)
    # z between shuffled-adjacent ATOMS, looked up via their ORIGINAL chain
    # positions in Z (Z is indexed by original chain position, and fisher_z
    # is a pure function of atom identity, so Z[POS[a]][POS[b]] == fisher_z(a, b)
    # regardless of what order we're now considering the atoms in).
    null_zs = [Z[POS[shuffled[k]]][POS[shuffled[k + 1]]] for k in range(N - 1)]
    null_means.append(statistics.mean(null_zs))

null_mean = statistics.mean(null_means)
null_std = statistics.stdev(null_means)  # sample stdev (ddof=1) over the 20 shuffle draws
z_score = (real_mean - null_mean) / null_std if null_std > 0 else float("inf")
log(f"null (20 shuffles, seeds 0..19): mean of per-shuffle means = {null_mean:.4f}, "
    f"stdev = {null_std:.4f}")
log(f"real chain z-score against null distribution: {z_score:.3f}")

RESULT["m1_neighbour_quality_vs_random"] = dict(
    real_mean_adjacent_z=round(real_mean, 4),
    real_median_adjacent_z=round(real_median, 4),
    n_links=len(adjacent_zs),
    null_seeds=NULL_SEEDS,
    null_per_shuffle_means=[round(x, 4) for x in null_means],
    null_mean=round(null_mean, 4),
    null_stdev=round(null_std, 4),
    z_score_real_vs_null=round(z_score, 4) if math.isfinite(z_score) else None,
)


# ─────────────────────────────────────────────────────────────────────────────
# M2 -- top-K recovery: for each atom, do its K=5 most-similar atoms (by
# Fisher-z) land within +-3 chain positions?
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("M2 -- top-K=5 recovery within +-3 chain positions")

K = 5
WINDOW = 3
hits = 0
total_checked = 0
for i in range(N):
    a = chain[i]
    # rank all other atoms by z descending; deterministic tie-break: higher z
    # first, then lexicographically smaller atom string first (mirrors the
    # seriation loop's own tie-break convention).
    ranked = sorted((j for j in range(N) if j != i), key=lambda j: (-Z[i][j], chain[j]))
    top5 = ranked[:K]
    for j in top5:
        total_checked += 1
        if abs(i - j) <= WINDOW:
            hits += 1

measured_frac = hits / total_checked
log(f"measured: {hits}/{total_checked} = {measured_frac:.4f} of top-{K} neighbours land "
    f"within +-{WINDOW} chain positions")

# exact expected value under "K random atoms (excluding self)" at each atom's
# OWN actual chain position (boundary-aware: an atom near either end of the
# chain has fewer than 2*WINDOW positions available within the window).
per_position_window_frac = []
for i in range(N):
    lo = max(0, i - WINDOW)
    hi = min(N - 1, i + WINDOW)
    window_count = (hi - lo)  # positions in [lo,hi] excluding i itself = (hi-lo+1)-1
    per_position_window_frac.append(window_count / (N - 1))
expected_random_frac = statistics.mean(per_position_window_frac)
ratio = measured_frac / expected_random_frac if expected_random_frac > 0 else float("inf")
log(f"exact random-baseline expectation (boundary-aware, same chain layout): "
    f"{expected_random_frac:.4f}")
log(f"ratio measured/random = {ratio:.3f}")

RESULT["m2_topk_recovery"] = dict(
    K=K, window=WINDOW,
    hits=hits, total_checked=total_checked,
    measured_fraction=round(measured_frac, 4),
    expected_random_fraction=round(expected_random_frac, 4),
    ratio_measured_over_random=round(ratio, 4) if math.isfinite(ratio) else None,
)


# ─────────────────────────────────────────────────────────────────────────────
# M3 -- monotonicity / decay: mean Fisher-z at each chain-distance d = 1..10
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("M3 -- mean Fisher-z by chain distance d=1..10")

decay = []
for d in range(1, 11):
    if d >= N:
        break
    zs_at_d = [Z[i][i + d] for i in range(N - d)]
    m = statistics.mean(zs_at_d)
    decay.append(dict(d=d, mean_z=round(m, 4), n_pairs=len(zs_at_d)))
    log(f"  d={d:2d}: mean z = {m:.4f} (n={len(zs_at_d)})")

# plain-language monotonicity check: strictly non-increasing (allow small
# float noise via a tiny epsilon), else check for a decay-then-plateau shape
# (steep, mostly-monotonic drop in the first half; small range in the second
# half relative to the overall range) before falling back to flat/noisy.
means_seq = [row["mean_z"] for row in decay]
diffs = [means_seq[k + 1] - means_seq[k] for k in range(len(means_seq) - 1)]
n_increases = sum(1 for x in diffs if x > 1e-9)
n_decreases = sum(1 for x in diffs if x < -1e-9)
n_flat = len(diffs) - n_increases - n_decreases

half = max(1, len(diffs) // 2)
early_diffs, late_diffs = diffs[:half], diffs[half:]
early_decreasing = sum(1 for x in early_diffs if x < -1e-9)
total_range = means_seq[0] - min(means_seq)
late_segment = means_seq[half:]
late_range = (max(late_segment) - min(late_segment)) if late_segment else 0.0

if n_increases == 0 and n_decreases > 0:
    shape = "decays monotonically (non-increasing at every step)"
elif early_decreasing >= len(early_diffs) - 1 and total_range > 0 and late_range <= 0.35 * total_range:
    shape = (f"decays sharply for d=1..{half + 1} (mostly monotonic: {early_decreasing}/"
             f"{len(early_diffs)} early steps decreasing) then plateaus/noisy for "
             f"d={half + 1}..{len(means_seq)} (late-segment range {late_range:.4f} is only "
             f"{late_range / total_range:.2f}x the full d=1..{len(means_seq)} range "
             f"{total_range:.4f})")
elif n_flat >= len(diffs) // 2:
    shape = "plateaus (mostly flat step-to-step)"
else:
    shape = "flat/noisy (no clear monotonic trend)"
log(f"shape verdict: {shape} (increases={n_increases}, decreases={n_decreases}, flat={n_flat})")

RESULT["m3_monotonicity_decay"] = dict(
    sequence=decay,
    step_diffs=[round(x, 4) for x in diffs],
    n_increasing_steps=n_increases,
    n_decreasing_steps=n_decreases,
    n_flat_steps=n_flat,
    shape_verdict=shape,
)


# ─────────────────────────────────────────────────────────────────────────────
# M4 -- where it breaks: first chain position whose link z drops below the
# overall mean pairwise z (i.e. the point past which the chain is no longer
# better than picking at random).
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("M4 -- where the chain stops beating the overall mean pairwise z")

link_zs = adjacent_zs  # link i is between chain[i] and chain[i+1], i=0..N-2
total_links = len(link_zs)
first_break = None
for i, z in enumerate(link_zs):
    if z < overall_mean_z:
        first_break = i
        break

if first_break is None:
    log(f"no link ever drops below the overall mean pairwise z ({overall_mean_z:.4f}) -- "
        f"the chain never gets worse than random by this measure.")
    frac_past = 0.0
else:
    links_past_inclusive = total_links - first_break
    frac_past = links_past_inclusive / total_links
    log(f"first break at link index {first_break} (link between chain positions "
        f"{first_break} and {first_break + 1}), link z = {link_zs[first_break]:.4f} < "
        f"overall mean {overall_mean_z:.4f}")
    log(f"links at-or-after the break: {links_past_inclusive}/{total_links} = "
        f"{frac_past:.4f} of the chain's links")

RESULT["m4_where_it_breaks"] = dict(
    overall_mean_pairwise_z=round(overall_mean_z, 4),
    total_links=total_links,
    first_break_link_index=first_break,
    link_zs=[round(x, 4) for x in link_zs],
    fraction_of_links_at_or_after_break=round(frac_past, 4),
)


# ─────────────────────────────────────────────────────────────────────────────
# write outputs
# ─────────────────────────────────────────────────────────────────────────────
json.dump(RESULT, open(os.path.join(RUNS, "w1-seriation.json"), "w"), indent=1)
flush_log()
log("=" * 78)
log("done.")
flush_log()
