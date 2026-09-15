"""w1_cue.py -- W1 CUE probe: motif-shaped units mapped to a palette256:palette256
tile via Fisher-Z seriated axis bytes, spread across the tile (mirroring
ndarray::simd::mask_shift_morton), scored two ways (bag-of-cells cosine vs
field-overlap), against override-twin retrieval. LAB TOOL, throwaway probe.

Pure stdlib. Every RNG is seeded explicitly; every dict/set that could affect
a result is iterated in a sorted or otherwise fully-deterministic order before
any max/argmax/tie-break happens. Run with PYTHONHASHSEED=0 (asserted below).

Sections mirror the brief exactly:
  ARM 0        -- positive control (reproduce the shipped BPE incumbent A2a-f)
  THE UNIT     -- unit_seq(gid, n): overlapping n-gram windows over atoms
  THE TILE     -- Fisher-Z seriated axis bytes (train only) -> cell(u)
  THE LUT      -- circular L1 256x256 distance table (built + self-checked;
                  NOT consumed by any scorer below -- see report note)
  THE SPREAD   -- decayed, non-wraparound, SUM-accumulating diffusion
  SCORERS      -- (i) bag-of-cells cosine via metrics2.retrieval unmodified
                  (ii) field overlap (Ruzicka/weighted-Jaccard), own retrieval
                  fn with the same interface/ranking logic as metrics2.retrieval
  MATRIX       -- n in (3,1) x radius in (0,1,2) x scorer in (bag,field)
  FALSIFIERS   -- F1 permuted assignment, F2 order shuffle, F3 n=3 vs n=1
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
import bpe

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
    with open(os.path.join(RUNS, "w1-cue.log"), "w") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")


NOTES = []  # surprises / places the spec left a gap and a decision had to be made


def note(msg):
    NOTES.append(msg)
    log(f"NOTE: {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Corpus load + split (same path as mq_run.py / ladder2.py)
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

RESULT = {
    "meta": dict(n_train=len(train), n_test=len(test), n_pool=len(pool), n_pairs=len(pairs)),
    "arm0_positive_control": None,
    "matrix": {},
    "falsifiers": {},
    "lut_selfcheck": {},
    "notes": NOTES,
}
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue.json"), "w"), indent=1)


# ─────────────────────────────────────────────────────────────────────────────
# ARM 0 -- positive control: reproduce the shipped BPE incumbent (A2a-f, ~0.180)
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("ARM 0 -- positive control (reproduce A2a-f)")
tr_atoms = {g: t for g, t in atoms.items() if g in trs}
t0 = time.time()
merges0, _ = bpe.learn(tr_atoms, 128, objective="frequency", class_of=None)
log(f"  bpe.learn(128, frequency): {len(merges0)} merges ({time.time() - t0:.1f}s)")
t0 = time.time()
seqs0 = metrics2.reapply(atoms, merges0)
log(f"  reapply: {time.time() - t0:.1f}s")
t0 = time.time()
r0 = metrics2.retrieval(seqs0, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  retrieval: {time.time() - t0:.1f}s -> {json.dumps(r0)}")
RESULT["arm0_positive_control"] = r0
arm0_pass = 0.15 <= r0["r@10"] <= 0.21
RESULT["arm0_pass"] = arm0_pass
log(f"ARM 0 r@10 = {r0['r@10']} -- {'PASS' if arm0_pass else 'FAIL'} (expect ~0.180, band 0.15..0.21)")
flush_log()
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue.json"), "w"), indent=1)
if not arm0_pass:
    log("POSITIVE CONTROL FAILED -- stopping. Everything downstream would be meaningless.")
    flush_log()
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# THE UNIT -- overlapping n-gram windows over atoms
# ─────────────────────────────────────────────────────────────────────────────
def unit_seq(gid, n):
    a = atoms[gid]
    return [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]


UNIT_CACHE = {}  # n -> {gid: [unit, ...]}
for n in (3, 1):
    UNIT_CACHE[n] = {g: unit_seq(g, n) for g in pool}
n3_types_pool = {u for g in pool for u in UNIT_CACHE[3][g]}
n3_types_train = {u for g in train for u in UNIT_CACHE[3][g]}
n1_types_pool = {u for g in pool for u in UNIT_CACHE[1][g]}
n1_types_train = {u for g in train for u in UNIT_CACHE[1][g]}
log(f"n=3 distinct units: pool {len(n3_types_pool)} train {len(n3_types_train)}")
log(f"n=1 distinct units: pool {len(n1_types_pool)} train {len(n1_types_train)}")
RESULT["meta"]["distinct_units"] = dict(
    n3_pool=len(n3_types_pool), n3_train=len(n3_types_train),
    n1_pool=len(n1_types_pool), n1_train=len(n1_types_train),
)


# ─────────────────────────────────────────────────────────────────────────────
# THE TILE -- Fisher-Z seriated axis bytes (train only), then cell(u)
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("THE TILE -- axis ordering (seriation) on TRAIN atom vocabulary")

V = sorted({x for g in train for x in atoms[g]})
log(f"len(V) = {len(V)} (distinct atom strings across train gids)")
RESULT["meta"]["len_V"] = len(V)

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
OOV_BYTE = 128  # midpoint / "no signal" placeholder for an atom absent from V
n_oov_atoms_seen = [0]  # mutable counter, bumped by byte() as a side effect


def byte_of(atom):
    p = POS.get(atom)
    if p is None:
        n_oov_atoms_seen[0] += 1
        return OOV_BYTE
    return round(p * 255 / max(1, len(V) - 1))


# ── cell assignment (Step 2), TRAIN ONLY context, per n ─────────────────────
def build_cell_map(n, unit_types_pool):
    """Lctx/Rctx counted from TRAIN occurrences only; cell(u) for every unit
    type in the POOL (train+test), falling back to u's own first/last atom
    when a side never occurs in train (including units never seen in train
    at all -- same code path, since Lc/Rc[u] is simply absent/empty then)."""
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
        # deterministic tie-break: lexicographically smallest atom string
        return min(k for k, v in counter.items() if v == mx)

    cell_map = {}
    for u in sorted(unit_types_pool):
        lctx = mode(Lc.get(u), u[0])
        rctx = mode(Rc.get(u), u[-1])
        cell_map[u] = (byte_of(lctx), byte_of(rctx))
    return cell_map


n_oov_atoms_seen[0] = 0
t0 = time.time()
CELL_MAP = {3: build_cell_map(3, n3_types_pool), 1: build_cell_map(1, n1_types_pool)}
log(f"cell maps built for n=3 ({len(CELL_MAP[3])} units) and n=1 ({len(CELL_MAP[1])} units) "
    f"({time.time() - t0:.2f}s); OOV-atom byte() calls during cell-map build: {n_oov_atoms_seen[0]}")
note(f"{n_oov_atoms_seen[0]} byte() calls during cell-map construction hit an atom absent "
     f"from V (train atom vocabulary) and fell back to the sentinel OOV_BYTE={OOV_BYTE}. "
     f"The spec defines a fallback for UNSEEN UNITS but not for an atom itself being "
     f"out-of-vocabulary (only 1 atom overall, 'Call(dtor)', is test-only); OOV_BYTE=128 "
     f"(chain midpoint, i.e. 'no positional information') was chosen as a neutral default.")


# ─────────────────────────────────────────────────────────────────────────────
# THE LUT -- circular L1 distance, 256x256, mirroring helix::DistanceLut::circular()
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("THE LUT -- circular L1 256x256 distance table")
t0 = time.time()
LUT = [[min(abs(a - b), 256 - abs(a - b)) for b in range(256)] for a in range(256)]
log(f"LUT built ({time.time() - t0:.2f}s)")
lut_checks = dict(
    d_0_255=LUT[0][255],    # wraps: 1
    d_0_128=LUT[0][128],    # antipodal: 128
    d_5_5=LUT[5][5],        # 0
    d_10_246=LUT[10][246],  # min(236,20) = 20
    d_symmetric_ok=all(LUT[i][j] == LUT[j][i] for i in (0, 1, 128, 200, 255) for j in (0, 1, 128, 200, 255)),
)
RESULT["lut_selfcheck"] = lut_checks
log(f"LUT self-check: {lut_checks}")
note("THE LUT (circular 256x256 distance table) is built and self-checked above, but is NOT "
     "consumed by either scorer below: the BAG scorer uses metrics2.retrieval's cosine "
     "similarity over cell-token Counters, and the FIELD scorer uses the Ruzicka/weighted-"
     "Jaccard overlap the brief specifies (sum-min / sum-max), neither of which is a distance "
     "computation over the LUT. It is built exactly as specified but is otherwise inert in "
     "this probe's scoring pipeline -- reported as-is rather than wired into something the "
     "brief didn't ask for.")


# ─────────────────────────────────────────────────────────────────────────────
# THE SPREAD -- decayed, non-wraparound, additive (SUM, never XOR) diffusion
# ─────────────────────────────────────────────────────────────────────────────
DECAY = 0.5
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))  # +basin, -basin, +identity, -identity


def spread_field(occurrences, cell_map, radius):
    """A body's field: dict {(basin, identity): magnitude}.
    Start: += 1.0 at cell(u) for every unit OCCURRENCE (not type).
    Then `radius` spread steps; each step reads the field as it stood after
    the PREVIOUS step and produces a new field that (a) RETAINS every value
    already present (the "do NOT overwrite" self term) and (b) ACCUMULATES --
    by SUM, never XOR; this is I-SUBSTRATE-MARKOV / the shipped vsa16k_bundle
    doc's "Do NOT substitute XOR" -- a DECAY-scaled shifted copy of the prior
    field for each of the four axis directions into that same destination.
    No wraparound: a shift that would leave [0,255] on either axis simply
    contributes nothing there (mirrors ndarray::simd::mask_shift_morton's
    edge behaviour) -- this is deliberately DIFFERENT from the LUT above,
    which IS circular; the two are not the same operation."""
    field = collections.defaultdict(float)
    for u in occurrences:
        field[cell_map[u]] += 1.0
    field = dict(field)
    for _ in range(radius):
        nxt = dict(field)  # retain the pre-step field (do NOT overwrite)
        for (b, i), val in field.items():
            for db, di in DIRS:
                nb, ni = b + db, i + di
                if 0 <= nb <= 255 and 0 <= ni <= 255:
                    key = (nb, ni)
                    nxt[key] = nxt.get(key, 0.0) + DECAY * val  # SUM, never XOR
        field = nxt
    return field


# ─────────────────────────────────────────────────────────────────────────────
# SCORERS
# ─────────────────────────────────────────────────────────────────────────────
def bag_seqs(n, cell_map, order_override=None):
    seqs = {}
    for g in pool:
        units = order_override[g] if order_override is not None else UNIT_CACHE[n][g]
        seqs[g] = [cell_map[u] for u in units]
    return seqs


def build_fields(n, cell_map, radius, order_override=None):
    out = {}
    for g in pool:
        units = order_override[g] if order_override is not None else UNIT_CACHE[n][g]
        out[g] = spread_field(units, cell_map, radius)
    return out


def _ruzicka(a, b):
    """sim(A,B) = sum_shared min(A[c],B[c]) / sum_union max(A[c],B[c]).
    No cosine anywhere. O(|a|+|b|)."""
    if not a or not b:
        return 0.0
    num = 0.0
    denom = 0.0
    seen = set()
    for k, va in a.items():
        vb = b.get(k)
        if vb is None:
            denom += va
        else:
            num += va if va < vb else vb
            denom += va if va > vb else vb
            seen.add(k)
    for k, vb in b.items():
        if k not in seen:
            denom += vb
    return num / denom if denom else 0.0


def field_retrieval(fields, prs, ks=(1, 5, 10), shuffled_seeds=()):
    """Same interface / same ranking logic as metrics2.retrieval, scored by
    field overlap instead of bag cosine. Performance note (semantically
    inert): the ranked-by-similarity target list for a fixed query does not
    depend on which target it is later paired with, so it is computed once
    per DISTINCT query gid and reused for the real pairing and every null
    shuffle -- this is the exact same result metrics2.retrieval's per-call
    O(n_pairs * n_vecs) re-sweep would produce, just without repeating the
    O(n_vecs) sweep 1 (real) + len(shuffled_seeds) (null) times per query."""
    rank_cache = {}

    def rank_lookup(q):
        rl = rank_cache.get(q)
        if rl is not None:
            return rl
        if q not in fields:
            rl = {}
        else:
            sq = fields[q]
            sims = sorted(((_ruzicka(sq, v), g) for g, v in fields.items() if g != q), reverse=True)
            rl = {g: i + 1 for i, (_, g) in enumerate(sims)}
        rank_cache[q] = rl
        return rl

    def run(pp):
        rec = collections.Counter()
        mrr = n = 0
        for q, t in pp:
            if q not in fields or t not in fields:
                continue
            rl = rank_lookup(q)
            rank = rl.get(t)
            n += 1
            if rank:
                mrr += 1.0 / rank
                for k in ks:
                    rec[k] += rank <= k
        return dict(n=n, mrr=round(mrr / max(1, n), 4),
                    **{f"r@{k}": round(rec[k] / max(1, n), 4) for k in ks})

    out = run(prs)
    if shuffled_seeds:
        nulls = []
        targets = [t for _, t in prs]
        for sd in shuffled_seeds:
            rng = random.Random(sd)
            tt = list(targets)
            rng.shuffle(tt)
            nulls.append(run(list(zip((q for q, _ in prs), tt))))
        out["null_r@10"] = round(sum(x["r@10"] for x in nulls) / len(nulls), 4)
        out["null_mrr"] = round(sum(x["mrr"] for x in nulls) / len(nulls), 4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MATRIX: n in (3,1) x radius in (0,1,2) x scorer in (bag,field)
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("RUN MATRIX (n x radius x scorer)")

BAG_BY_N = {}  # n -> retrieval dict; BAG doesn't reference the spread field at
               # all (it's built straight from cell(u), no radius argument
               # anywhere in its definition) so it is radius-invariant by
               # construction -- computed once per n, see NOTE below.
for n in (3, 1):
    t0 = time.time()
    seqs = bag_seqs(n, CELL_MAP[n])
    r = metrics2.retrieval(seqs, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
    BAG_BY_N[n] = r
    log(f"  BAG n={n} (computed once, radius-invariant): {time.time() - t0:.1f}s -> {json.dumps(r)}")

note("BAG scorer results are IDENTICAL across radius=0/1/2 for a given n: the brief's BAG "
     "definition (`seqs = {gid: [cell(u) for u in unit_seq(gid,n)]}`) never references the "
     "spread field or the radius parameter at all -- only FIELD does. Computed once per n "
     "(not re-run redundantly 3x) and the same dict is reported under all three radius rows "
     "below; this is a structural property of the design, not a shortcut that changes numbers.")

for n in (3, 1):
    for radius in (0, 1, 2):
        label = f"n{n}_r{radius}"
        row = {}
        row["bag"] = BAG_BY_N[n]
        t0 = time.time()
        fields = build_fields(n, CELL_MAP[n], radius)
        rf = field_retrieval(fields, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
        row["field"] = rf
        log(f"  FIELD {label}: {time.time() - t0:.1f}s -> {json.dumps(rf)}")
        RESULT["matrix"][label] = row
        flush_log()
        json.dump(RESULT, open(os.path.join(RUNS, "w1-cue.json"), "w"), indent=1)

log("matrix done.")


# ─────────────────────────────────────────────────────────────────────────────
# FALSIFIERS
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("FALSIFIERS")

# F1 -- permuted assignment: same marginal cell distribution, scrambled
# content->cell relationship. Permute ASSIGNMENTS (which unit type maps to
# which cell), never LABELS (a global relabelling of byte values would be a
# no-op under both scorers, since cosine/overlap similarity only depends on
# which cells two bodies SHARE, not what those cells are called).
unit_types_sorted3 = sorted(CELL_MAP[3].keys())
cells_list = [CELL_MAP[3][u] for u in unit_types_sorted3]
rng_f1 = random.Random(0)
shuffled_cells = list(cells_list)
rng_f1.shuffle(shuffled_cells)
PERMUTED_CELL_MAP_3 = dict(zip(unit_types_sorted3, shuffled_cells))
same_marginals = collections.Counter(CELL_MAP[3].values()) == collections.Counter(PERMUTED_CELL_MAP_3.values())
log(f"F1: permuted {len(unit_types_sorted3)} unit->cell assignments (seed 0); "
    f"marginal cell-value distribution preserved: {same_marginals}")
assert same_marginals, "F1 must preserve the marginal cell distribution by construction"

t0 = time.time()
f1_bag_seqs = bag_seqs(3, PERMUTED_CELL_MAP_3)
f1_bag = metrics2.retrieval(f1_bag_seqs, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  F1 bag  n=3 r=1 permuted: {time.time() - t0:.1f}s -> {json.dumps(f1_bag)}")
t0 = time.time()
f1_fields = build_fields(3, PERMUTED_CELL_MAP_3, 1)
f1_field = field_retrieval(f1_fields, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  F1 field n=3 r=1 permuted: {time.time() - t0:.1f}s -> {json.dumps(f1_field)}")
RESULT["falsifiers"]["f1_permuted_assignment"] = dict(bag=f1_bag, field=f1_field)

baseline_n3_r1 = RESULT["matrix"]["n3_r1"]
f1_bag_fired = f1_bag["r@10"] < baseline_n3_r1["bag"]["r@10"]
f1_field_fired = f1_field["r@10"] < baseline_n3_r1["field"]["r@10"]
log(f"  F1 fired (dropped vs n3_r1 baseline)? bag={f1_bag_fired} "
    f"({baseline_n3_r1['bag']['r@10']} -> {f1_bag['r@10']}), "
    f"field={f1_field_fired} ({baseline_n3_r1['field']['r@10']} -> {f1_field['r@10']})")
RESULT["falsifiers"]["f1_fired"] = dict(bag=f1_bag_fired, field=f1_field_fired)
flush_log()
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue.json"), "w"), indent=1)

# F2 -- order shuffle: same multiset of units per body, order scrambled
# (random.Random(1), reset fresh per gid).
order_override_3 = {}
for g in sorted(pool):
    units = list(UNIT_CACHE[3][g])
    random.Random(1).shuffle(units)
    order_override_3[g] = units

t0 = time.time()
f2_bag_seqs = bag_seqs(3, CELL_MAP[3], order_override=order_override_3)
f2_bag = metrics2.retrieval(f2_bag_seqs, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  F2 bag  n=3 r=1 order-shuffled: {time.time() - t0:.1f}s -> {json.dumps(f2_bag)}")
t0 = time.time()
f2_fields = build_fields(3, CELL_MAP[3], 1, order_override=order_override_3)
f2_field = field_retrieval(f2_fields, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  F2 field n=3 r=1 order-shuffled: {time.time() - t0:.1f}s -> {json.dumps(f2_field)}")
RESULT["falsifiers"]["f2_order_shuffle"] = dict(bag=f2_bag, field=f2_field)

f2_bag_identical = f2_bag == baseline_n3_r1["bag"]
f2_field_identical = f2_field == baseline_n3_r1["field"]
log(f"  F2 identical to n3_r1 baseline (expected -- both scorers aggregate over occurrences, "
    f"order-independent by construction)? bag={f2_bag_identical} field={f2_field_identical}")
RESULT["falsifiers"]["f2_identical_to_baseline"] = dict(bag=f2_bag_identical, field=f2_field_identical)
if not (f2_bag_identical and f2_field_identical):
    note("F2 (order shuffle) was expected to be a structural no-op for both scorers (BAG is a "
         "Counter/histogram, FIELD is an order-independent accumulation) but did NOT come back "
         "byte-identical -- see f2_order_shuffle vs n3_r1 in the matrix for the actual numbers.")
else:
    note("F2 (order shuffle) reproduced the n3_r1 baseline exactly for both scorers, confirming "
         "empirically (not just by construction-argument) that neither BAG nor FIELD as "
         "specified can see within-body sequence order -- both are pure occurrence "
         "aggregations. This is a property of the cue design, not a bug in the falsifier.")
flush_log()
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue.json"), "w"), indent=1)

# F3 -- n=3 (motif) vs n=1 (single-atom ceiling control), already in the matrix
f3 = {}
for radius in (0, 1, 2):
    for scorer in ("bag", "field"):
        r3 = RESULT["matrix"][f"n3_r{radius}"][scorer]["r@10"]
        r1 = RESULT["matrix"][f"n1_r{radius}"][scorer]["r@10"]
        f3[f"r{radius}_{scorer}"] = dict(n3_r10=r3, n1_r10=r1, n3_beats_n1=r3 > r1)
log(f"F3 (n=3 vs n=1, r@10): {json.dumps(f3)}")
RESULT["falsifiers"]["f3_motif_vs_ceiling"] = f3


# ─────────────────────────────────────────────────────────────────────────────
# best cue overall
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
candidates = [("ARM0_A2a-f", r0["r@10"])]
for label, row in RESULT["matrix"].items():
    candidates.append((f"{label}_bag", row["bag"]["r@10"]))
    candidates.append((f"{label}_field", row["field"]["r@10"]))
best_label, best_r10 = max(candidates, key=lambda x: (x[1], x[0]))
RESULT["best_cue"] = dict(label=best_label, r10=best_r10)
log(f"BEST CUE: {best_label} r@10={best_r10}")

RESULT["secs"] = round(time.time() - T0)
flush_log()
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue.json"), "w"), indent=1)
log(f"# done {RESULT['secs']}s -- wrote runs/w1-cue.json and runs/w1-cue.log")
flush_log()
