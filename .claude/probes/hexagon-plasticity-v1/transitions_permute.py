"""transitions_permute.py -- settle whether the palette CELLS in the TRANSITIONS
arm (w1_cue2.py) contribute anything beyond being consistent labels, or whether
the whole 0.1579 -> 0.0827 order-sensitive signal is really "order of unit
types" with cells acting as an arbitrary (but consistent) alphabet.

w1_cue2.py's TRANSITIONS arm was tested against an ORDER shuffle (its own F2
-style shuffle, seed 1 per gid) and dropped r@10 0.1579 -> 0.0827, proving
order matters. It was NEVER tested against a CELL-ASSIGNMENT permutation --
the disable that would show whether the cell geometry itself carries
anything, independent of order. This script closes that gap with four arms,
all on the TRANSITIONS representation (bagged consecutive cell-PAIRS over the
n=3 unit sequence), all scored with metrics2.retrieval unmodified:

  A. BASELINE          -- reproduce w1_cue2.py's published TRANSITIONS
                           original-order r@10 = 0.1579. STOP if it does not
                           reproduce -- everything downstream is meaningless
                           if the object under test has moved.
  B. CELL PERMUTATION   -- the missing disable. Permute which CELL each unit
                           TYPE maps to (marginal cell-value distribution
                           preserved exactly; content->cell relationship
                           destroyed) -- same code shape as w1_cue2.py's F1
                           (random.Random(0), permute ASSIGNMENTS, never
                           LABELS -- a label relabel would be a no-op).
  C. IDENTITY CONTROL   -- the decisive arm. Replace cell(u) with an
                           arbitrary DENSE INTEGER id per unit type (its
                           index in sorted(unit_types)), carrying NO palette
                           structure, NO seriation, NO two-axis geometry,
                           nothing but identity. If C matches A, the palette
                           contributes nothing and the whole signal is
                           order-of-unit-types. If A clearly beats C, the
                           cells carry something beyond identity.
  D. ORDER SHUFFLE ON C -- the same order shuffle w1_cue2.py's TRANSITIONS
                           arm used (random.Random(1), fresh per gid),
                           applied to the identity-id representation. If D
                           drops from C the way 0.1579->0.0827 did, order is
                           the whole story and never needed the palette.

Every piece reused from w1_cue2.py -- the corpus load/split, THE UNIT
(unit_seq), THE TILE (V, ctx, fisher_z, the seriation chain, byte_of),
build_cell_map, trans_seq, the F1 permutation shape, and the F2/TRANSITIONS
order-shuffle shape -- is copied verbatim from that file, not re-derived and
not "improved". Only n=1 (unused by TRANSITIONS), ARM0 (the BPE positive
control), THE LUT / THE LUT KERNEL / THE SPREAD / the bag+field scorers, and
the n=3/n=1 x radius(0,1,2) matrix are omitted: none of them sit on the
TRANSITIONS code path, and re-running them would only burn time (the LUT
rows alone cost 3.5-4 minutes EACH in the source run).

Pure stdlib. PYTHONHASHSEED=0 asserted. Every RNG is seeded explicitly; every
dict/set that could affect an ordering-sensitive result is iterated sorted
before any max/argmax/tie-break happens.
"""
import os as _os
assert _os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"

import collections
import json
import math
import os
import random
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
    with open(os.path.join(RUNS, "w1-transitions-permute.log"), "w") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")


NOTES = []


def note(msg):
    NOTES.append(msg)
    log(f"NOTE: {msg}")


RESULT = {"meta": {}, "notes": NOTES}


def dump():
    json.dump(RESULT, open(os.path.join(RUNS, "w1-transitions-permute.json"), "w"), indent=1)


# ─────────────────────────────────────────────────────────────────────────────
# Corpus load + split -- VERBATIM from w1_cue2.py
# ─────────────────────────────────────────────────────────────────────────────
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

RESULT["meta"] = dict(n_train=len(train), n_test=len(test), n_pool=len(pool), n_pairs=len(pairs))
dump()


# ─────────────────────────────────────────────────────────────────────────────
# THE UNIT -- overlapping n-gram windows over atoms -- VERBATIM shape from
# w1_cue2.py, restricted to n=3 (TRANSITIONS never touches n=1).
# ─────────────────────────────────────────────────────────────────────────────
def unit_seq(gid, n):
    a = atoms[gid]
    return [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]


UNIT_CACHE = {3: {g: unit_seq(g, 3) for g in pool}}
n3_types_pool = {u for g in pool for u in UNIT_CACHE[3][g]}
log(f"n=3 distinct units: pool {len(n3_types_pool)}")
RESULT["meta"]["n3_types_pool"] = len(n3_types_pool)


# ─────────────────────────────────────────────────────────────────────────────
# THE TILE -- Fisher-Z seriated axis bytes (train only), then cell(u) --
# VERBATIM from w1_cue2.py's THE TILE section.
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("THE TILE -- axis ordering (seriation) on TRAIN atom vocabulary")

V = sorted({x for g in train for x in atoms[g]})
log(f"len(V) = {len(V)} (distinct atom strings across train gids)")
RESULT["meta"]["len_V"] = len(V)

ctx = {a: collections.Counter() for a in V}
for g in train:
    a = atoms[g]
    for i, x in enumerate(a):
        cx = ctx.get(x)
        if cx is None:
            continue
        if i > 0:
            cx["L:" + a[i - 1]] += 1
        if i < len(a) - 1:
            cx["R:" + a[i + 1]] += 1


def fisher_z(a, b):
    """VERBATIM from w1_cue2.py -- mirrors lance-graph helix::fisher_z."""
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
chain_start = min(V)
chain = [chain_start]
remaining.discard(chain_start)
while remaining:
    cur = chain[-1]
    best, best_z = None, None
    for cand in sorted(remaining):
        z = fisher_z(cur, cand)
        if best_z is None or z > best_z:
            best_z, best = z, cand
    chain.append(best)
    remaining.discard(best)
log(f"seriation chain built: {len(chain)} atoms ({time.time() - t0:.2f}s)")

POS = {a: i for i, a in enumerate(chain)}
OOV_BYTE = 128
n_oov_atoms_seen = [0]


def byte_of(atom):
    """VERBATIM from w1_cue2.py (FIX 1: byte(atom) = pos directly)."""
    p = POS.get(atom)
    if p is None:
        n_oov_atoms_seen[0] += 1
        return OOV_BYTE
    return p


def build_cell_map(n, unit_types_pool):
    """VERBATIM from w1_cue2.py."""
    Lc = collections.defaultdict(collections.Counter)
    Rc = collections.defaultdict(collections.Counter)
    for g in train:
        for i, u in enumerate(UNIT_CACHE[n][g]):
            a = atoms[g]
            if i > 0:
                Lc[u][a[i - 1]] += 1
            if i + n < len(a):
                Rc[u][a[i + n]] += 1

    def mode(counter, fallback):
        if not counter:
            return fallback
        mx = max(counter.values())
        return min(k for k, v in counter.items() if v == mx)

    cell_map = {}
    for u in sorted(unit_types_pool):
        lctx = mode(Lc.get(u), u[0])
        rctx = mode(Rc.get(u), u[-1])
        cell_map[u] = (byte_of(lctx), byte_of(rctx))
    return cell_map


n_oov_atoms_seen[0] = 0
t0 = time.time()
CELL_MAP3 = build_cell_map(3, n3_types_pool)
log(f"cell map built for n=3 ({len(CELL_MAP3)} units) ({time.time() - t0:.2f}s); "
    f"OOV-atom byte() calls during cell-map build: {n_oov_atoms_seen[0]}")

unit_types_sorted3 = sorted(CELL_MAP3.keys())
distinct_types_n3 = len(unit_types_sorted3)
distinct_cells_n3 = len(set(CELL_MAP3.values()))
collision_factor = round(distinct_types_n3 / distinct_cells_n3, 4)
log(f"distinct unit types: {distinct_types_n3}, distinct cells occupied: {distinct_cells_n3}, "
    f"collision factor (types/cells): {collision_factor}")
RESULT["meta"]["distinct_types_n3"] = distinct_types_n3
RESULT["meta"]["distinct_cells_n3"] = distinct_cells_n3
RESULT["meta"]["collision_factor"] = collision_factor
dump()


# ─────────────────────────────────────────────────────────────────────────────
# trans_seq -- VERBATIM from w1_cue2.py's TRANSITIONS section.
# ─────────────────────────────────────────────────────────────────────────────
def trans_seq(units, cell_map):
    """The list of consecutive cell PAIRS (cell(u_i), cell(u_{i+1})) over a
    body's own n=3 unit sequence, in whatever order `units` is given in --
    order-SENSITIVE by construction. VERBATIM from w1_cue2.py."""
    cells = [cell_map[u] for u in units]
    return [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# ARM A -- BASELINE: reproduce the published TRANSITIONS original-order r@10.
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("ARM A -- BASELINE (reproduce published TRANSITIONS r@10 = 0.1579)")
trans_seqs_a = {g: trans_seq(UNIT_CACHE[3][g], CELL_MAP3) for g in pool}
t0 = time.time()
r_a = metrics2.retrieval(trans_seqs_a, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  ARM A: {time.time() - t0:.1f}s -> {json.dumps(r_a)}")
RESULT["arm_a_baseline"] = r_a

PUBLISHED_BASELINE_R10 = 0.1579
baseline_reproduced = abs(r_a["r@10"] - PUBLISHED_BASELINE_R10) < 1e-9
RESULT["baseline_reproduced"] = baseline_reproduced
log(f"ARM A baseline reproduced published 0.1579? {baseline_reproduced} (got {r_a['r@10']})")
flush_log()
dump()

if not baseline_reproduced:
    log("BASELINE DID NOT REPRODUCE -- STOPPING. Every other arm would be meaningless "
        "if the object under test has moved since the published 0.1579 run.")
    note(f"ARM A did not reproduce the published TRANSITIONS r@10=0.1579 -- got "
         f"{r_a['r@10']} instead. Stopped before running arms B/C/D: comparing them "
         f"against a baseline that has itself moved would not answer the question.")
    RESULT["stopped_after_baseline_mismatch"] = True
    RESULT["secs"] = round(time.time() - T0)
    flush_log()
    dump()
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# ARM B -- CELL PERMUTATION (the missing disable): same code shape as
# w1_cue2.py's F1 -- permute unit-type -> cell ASSIGNMENTS (never LABELS,
# which would be a no-op), marginal cell-value distribution preserved.
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("ARM B -- CELL PERMUTATION (permute assignments, seed 0, marginals preserved)")
cells_list = [CELL_MAP3[u] for u in unit_types_sorted3]
rng_b = random.Random(0)
shuffled_cells = list(cells_list)
rng_b.shuffle(shuffled_cells)
PERMUTED_CELL_MAP_3 = dict(zip(unit_types_sorted3, shuffled_cells))

same_marginals = (collections.Counter(CELL_MAP3.values())
                   == collections.Counter(PERMUTED_CELL_MAP_3.values()))
assert same_marginals, "ARM B must preserve the marginal cell distribution by construction"
log(f"  permuted {len(unit_types_sorted3)} unit->cell assignments (seed 0); "
    f"marginal cell-value distribution preserved: {same_marginals} -- assert passed")
RESULT["arm_b_marginals_preserved"] = same_marginals

trans_seqs_b = {g: trans_seq(UNIT_CACHE[3][g], PERMUTED_CELL_MAP_3) for g in pool}
t0 = time.time()
r_b = metrics2.retrieval(trans_seqs_b, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  ARM B: {time.time() - t0:.1f}s -> {json.dumps(r_b)}")
RESULT["arm_b_cell_permutation"] = r_b
flush_log()
dump()


# ─────────────────────────────────────────────────────────────────────────────
# ARM C -- IDENTITY CONTROL (the decisive arm): cell(u) -> a dense integer id
# per unit type, id_of[u] = index of u in sorted(unit_types). No palette
# structure, no seriation, no two-axis geometry -- identity only.
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("ARM C -- IDENTITY CONTROL (dense integer id per unit type, no palette geometry)")
ID_MAP_3 = {u: i for i, u in enumerate(unit_types_sorted3)}
trans_seqs_c = {g: trans_seq(UNIT_CACHE[3][g], ID_MAP_3) for g in pool}
t0 = time.time()
r_c = metrics2.retrieval(trans_seqs_c, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  ARM C: {time.time() - t0:.1f}s -> {json.dumps(r_c)}")
RESULT["arm_c_identity_control"] = r_c
flush_log()
dump()


# ─────────────────────────────────────────────────────────────────────────────
# ARM D -- ORDER SHUFFLE ON C: same order-shuffle shape as w1_cue2.py's F2 /
# TRANSITIONS order-shuffle (random.Random(1), fresh per gid), applied to the
# identity-id representation instead of the cell-map representation.
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("ARM D -- ORDER SHUFFLE ON C (seed 1, fresh per gid, identity-id representation)")
order_override_id = {}
for g in sorted(pool):
    units = list(UNIT_CACHE[3][g])
    random.Random(1).shuffle(units)
    order_override_id[g] = units

trans_seqs_d = {g: trans_seq(order_override_id[g], ID_MAP_3) for g in pool}
t0 = time.time()
r_d = metrics2.retrieval(trans_seqs_d, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  ARM D: {time.time() - t0:.1f}s -> {json.dumps(r_d)}")
RESULT["arm_d_identity_order_shuffled"] = r_d
flush_log()
dump()


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY -- raw numbers and deltas only, no pass/fail gate.
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("SUMMARY (numbers only -- no gate applied)")
log(f"  A baseline                       r@10 = {r_a['r@10']}")
log(f"  B cell-permutation                r@10 = {r_b['r@10']}  (B - A = {round(r_b['r@10'] - r_a['r@10'], 4)})")
log(f"  C identity-id                     r@10 = {r_c['r@10']}  (C - A = {round(r_c['r@10'] - r_a['r@10'], 4)})")
log(f"  D identity-id + order-shuffle      r@10 = {r_d['r@10']}  (D - C = {round(r_d['r@10'] - r_c['r@10'], 4)})")
log(f"  distinct unit types: {distinct_types_n3}, distinct cells occupied: {distinct_cells_n3}, "
    f"collision factor (types/cells): {collision_factor}")

RESULT["summary"] = dict(
    a_r10=r_a["r@10"], b_r10=r_b["r@10"], c_r10=r_c["r@10"], d_r10=r_d["r@10"],
    b_minus_a=round(r_b["r@10"] - r_a["r@10"], 4),
    c_minus_a=round(r_c["r@10"] - r_a["r@10"], 4),
    d_minus_c=round(r_d["r@10"] - r_c["r@10"], 4),
    distinct_types_n3=distinct_types_n3, distinct_cells_n3=distinct_cells_n3,
    collision_factor=collision_factor,
)

RESULT["secs"] = round(time.time() - T0)
flush_log()
dump()
log(f"# done {RESULT['secs']}s -- wrote runs/w1-transitions-permute.json and .log")
flush_log()
