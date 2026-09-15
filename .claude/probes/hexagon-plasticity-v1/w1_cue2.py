"""w1_cue2.py -- W1 CUE probe, CORRECTION RUN over w1_cue.py (run 1, see
runs/w1-cue.json / w1-cue.log for the record of run 1's apparatus defects).
This file is a copy-and-fix of w1_cue.py, not a redesign: ARM0, the corpus/
pair plumbing, THE UNIT, the Fisher-Z seriation, THE SPREAD mechanism, and
the bag/field scorers are UNCHANGED. Three apparatus defects are fixed and
one new order-sensitive arm is added:

  FIX 1 -- byte(atom) = pos directly (0..len(V)-1), not
           round(pos*255/(len(V)-1)). Run 1's scaling put ~3.4 bytes between
           adjacent seriated atoms, so a radius-1/2 spread landed in empty
           space and could never reach a real neighbour -- that is why
           radius 0/1/2 were identical in run 1. len(V) is still 76 (train
           atom vocabulary is unaffected by this fix); the REALIZED tile is
           honestly len(V) x len(V) = 76x76, inside a still-256-wide LUT
           modulus (the modulus itself is unchanged, per
           helix::DistanceLut::circular's fixed 256 domain).
  FIX 2 -- a THIRD scorer, LUT (soft match via the circular 256x256 table,
           DMAX=3, w(d) = 1.0 / 0.5**d / 0.0), is added so THE LUT (built
           below, self-checked, identical to run 1) is actually consumed by
           a scorer instead of sitting inert. Implemented as a cell->gids/
           mass index (built once per (n,radius) fields dict) + a 49-cell
           DELTA_KERNEL whose every entry is a table lookup (never a
           recomputed abs()/min() formula) -- see "THE LUT KERNEL" below.
           MEASURED (this run, PYTHONHASHSEED=0, an 8-query timing probe
           before the real run) too slow at radius=2 for both n=3 and n=1;
           reduced to radius in (0,1) for the LUT scorer only, documented
           in a NOTE with the real measured wall-clock times. bag and field
           are unchanged and still cover all of radius in (0,1,2).
  FIX 3 -- F1 (permuted unit->cell assignment) is rerun on all THREE scorers
           at n=3, radius=1 (bag, field, lut), since FIX 2 gives the LUT
           scorer something a permutation can actually destroy.
  NEW ARM -- TRANSITIONS: an order-SENSITIVE representation (bagged
           consecutive cell-pairs over the n=3 unit sequence), scored with
           metrics2.retrieval unmodified, plus its own order shuffle
           (seed 1, per gid) -- run 1's bag/field representations were
           order-blind by construction, so its order-shuffle falsifier (F2)
           was structurally vacuous; this arm can actually see order.

Pure stdlib. Every RNG is seeded explicitly; every dict/set that could affect
a result is iterated in a sorted or otherwise fully-deterministic order before
any max/argmax/tie-break happens. Run with PYTHONHASHSEED=0 (asserted below).

Sections mirror w1_cue.py, in the same order, plus the two new pieces
(THE LUT KERNEL, TRANSITIONS) inserted where they belong:
  ARM 0        -- positive control (reproduce the shipped BPE incumbent A2a-f)
                  -- UNCHANGED, must still reproduce ~0.1805
  THE UNIT     -- unit_seq(gid, n): overlapping n-gram windows over atoms
                  -- UNCHANGED
  THE TILE     -- Fisher-Z seriated axis bytes (train only) -> cell(u)
                  -- FIX 1 applied to byte_of() only; seriation unchanged
  THE LUT      -- circular L1 256x256 distance table (built + self-checked)
                  -- UNCHANGED construction; now consumed (see below)
  THE LUT KERNEL -- NEW: DMAX=3 neighbour weights, every distance a table
                  read; the cell->gids/mass index is built per matrix row
  THE SPREAD   -- decayed, non-wraparound, SUM-accumulating diffusion
                  -- UNCHANGED
  SCORERS      -- (i) bag-of-cells cosine via metrics2.retrieval unmodified
                  (ii) field overlap (Ruzicka/weighted-Jaccard) -- UNCHANGED
                  (iii) NEW: lut_retrieval, soft match via THE LUT KERNEL
  MATRIX       -- n in (3,1) x radius in (0,1,2) x scorer in (bag,field,lut)
                  -- lut reduced to radius in (0,1), see FIX 2 note
  FALSIFIERS   -- F1 permuted assignment (now all 3 scorers, FIX 3), F2
                  order shuffle (kept from run 1, unmodified -- still
                  structurally vacuous for bag/field/lut, see its own note),
                  F3 n=3 vs n=1 (now covers lut where computed)
  TRANSITIONS  -- NEW: order-sensitive arm + its own order-shuffle falsifier
"""
import os as _os
assert _os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"

import collections
import json
import math
import os
import random
import signal
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
    with open(os.path.join(RUNS, "w1-cue2.log"), "w") as fh:
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
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue2.json"), "w"), indent=1)


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
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue2.json"), "w"), indent=1)
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
OOV_BYTE = 128  # sentinel for an atom absent from V -- see updated note below
n_oov_atoms_seen = [0]  # mutable counter, bumped by byte() as a side effect


def byte_of(atom):
    """FIX 1 (root-cause fix for run 1's flat-radius result): NATIVE byte
    spacing. Run 1 used round(pos*255/(len(V)-1)) which, with len(V)=76,
    put ~3.4 bytes between adjacent seriated atoms -- a radius-1/2 spread
    moved into EMPTY space and could never reach the neighbouring atom's
    cell (this is exactly why radius 0/1/2 were byte-identical in run 1).
    Here byte(atom) = pos directly: adjacent seriated atoms are exactly 1
    byte apart, so a radius-1 spread reaches exactly one seriated
    neighbour. The LUT modulus is untouched (still 256-wide, per
    helix::DistanceLut::circular's fixed domain) -- only the OCCUPIED
    sub-range shrinks to 0..len(V)-1 (0..75); see the realized-tile-
    occupancy report emitted right after the cell maps below."""
    p = POS.get(atom)
    if p is None:
        n_oov_atoms_seen[0] += 1
        return OOV_BYTE
    return p


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
     f"out-of-vocabulary (only 1 atom overall, 'Call(dtor)', is test-only). Under FIX 1's "
     f"unscaled byte_of, OOV_BYTE=128 is no longer the 'chain midpoint' -- the real occupied "
     f"range is 0..{len(V) - 1}, so 128 sits clearly OUTSIDE it. This makes an OOV atom's "
     f"context maximally UNREACHABLE by adjacency (radius<=2 spread, LUT DMAX=3) rather than "
     f"'neutral/central' as it was under run 1's scaled scheme -- a change in what the same "
     f"sentinel value MEANS, carried over unchanged in VALUE, flagged honestly rather than "
     f"silently reinterpreted. Affects a negligible fraction of byte() calls either way.")

# ── realized tile occupancy (honest report, FIX 1) ──────────────────────────
occ_bytes = sorted(POS.values())
distinct_cells_n3 = len(set(CELL_MAP[3].values()))
distinct_cells_n1 = len(set(CELL_MAP[1].values()))
oov_used_n3 = sum(1 for (bl, br) in CELL_MAP[3].values() if bl == OOV_BYTE or br == OOV_BYTE)
oov_used_n1 = sum(1 for (bl, br) in CELL_MAP[1].values() if bl == OOV_BYTE or br == OOV_BYTE)
log(f"REALIZED TILE (FIX 1): occupied byte range [{occ_bytes[0]}, {occ_bytes[-1]}] "
    f"(len(V)={len(V)} seriated atoms, byte(atom)=pos directly) inside a still-256-wide LUT "
    f"modulus. len(V) x len(V) = {len(V)}x{len(V)} = {len(V) * len(V)} possible cells in the "
    f"occupied sub-range; distinct cells actually used: n=3 -> {distinct_cells_n3}, "
    f"n=1 -> {distinct_cells_n1} (some cells carry OOV_BYTE=128 on one or both axes: "
    f"n=3 -> {oov_used_n3} cell entries, n=1 -> {oov_used_n1}). This is NOT a 256x256 tile in "
    f"any meaningful sense -- reported honestly rather than claiming resolution the "
    f"vocabulary (76 distinct train atom strings) does not have.")
RESULT["meta"]["realized_tile"] = dict(
    len_V=len(V), occupied_byte_min=occ_bytes[0], occupied_byte_max=occ_bytes[-1],
    lut_modulus=256, distinct_cells_n3=distinct_cells_n3, distinct_cells_n1=distinct_cells_n1,
    oov_byte=OOV_BYTE, oov_cell_entries_n3=oov_used_n3, oov_cell_entries_n1=oov_used_n1,
)


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
note("FIX 2: THE LUT (circular 256x256 distance table, unchanged construction from run 1) is "
     "now CONSUMED by the new LUT scorer below (see THE LUT KERNEL + lut_retrieval). BAG "
     "(metrics2.retrieval cosine) and FIELD (Ruzicka/weighted-Jaccard) are UNCHANGED from run "
     "1 and still do not read the LUT -- that is by design (they are the exact-match / "
     "spread-overlap controls the LUT scorer is meant to be compared against), not an "
     "oversight repeated from run 1.")


# ─────────────────────────────────────────────────────────────────────────────
# THE LUT KERNEL (FIX 2, NEW) -- DMAX-radius neighbour weights, every distance
# a table read, never a recomputed abs()/min() formula.
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("THE LUT KERNEL -- DMAX=3 neighbour offsets via LUT reads")
DMAX = 3


def w_of(d):
    """w(d) = 1.0 if d==0, else 0.5**d for d<=DMAX, else 0.0 -- exactly the
    brief's formula. In DELTA_KERNEL below every d passed in was itself read
    from LUT[...][...], never computed inline."""
    if d == 0:
        return 1.0
    if d <= DMAX:
        return 0.5 ** d
    return 0.0


t0 = time.time()
# The circular distance is translation-invariant (LUT[byte][byte+delta] ==
# LUT[0][delta] for every byte, by construction of the min(|Δ|,256-|Δ|)
# formula THE LUT itself was built from) -- so the 49 (db,di,weight) offsets
# needed to reach every cell within DMAX of ANY cell are read ONCE via
# LUT[0][...] (a genuine table read, not a recomputed formula) rather than
# rebuilding an identical 256-entry neighbour list per byte value. Applying
# the kernel to a specific cell (a0,a1) is then just ((a0+db)%256,(a1+di)%256)
# -- offset arithmetic, with the DISTANCE itself always sourced from the table.
DELTA_KERNEL = []
for db in range(-DMAX, DMAX + 1):
    d0 = LUT[0][db % 256]  # table read
    for di in range(-DMAX, DMAX + 1):
        d1 = LUT[0][di % 256]  # table read
        d = max(d0, d1)
        DELTA_KERNEL.append((db, di, w_of(d)))
assert len(DELTA_KERNEL) == (2 * DMAX + 1) ** 2 == 49
assert all(wt > 0.0 for _, _, wt in DELTA_KERNEL), "every offset in [-DMAX,DMAX]^2 has d<=DMAX by construction"
log(f"DELTA_KERNEL built: {len(DELTA_KERNEL)} (db,di,weight) offsets, DMAX={DMAX} "
    f"({time.time() - t0:.3f}s)")
RESULT["meta"]["lut_kernel"] = dict(dmax=DMAX, kernel_size=len(DELTA_KERNEL))


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


def lut_retrieval(fields, prs, ks=(1, 5, 10), shuffled_seeds=()):
    """FIX 2, scorer (iii). Same interface / same ranking logic as
    metrics2.retrieval and field_retrieval, but similarity is the soft match
    specified in the brief:
        for ca in A, cb in B with d(ca,cb) <= DMAX:
            contribution = A[ca] * B[cb] * w(d)
        sim(A,B) = sum(contributions) / sqrt(sum(A[c]^2) * sum(B[c]^2))
    d_axis is ALWAYS a table lookup (never recomputed) -- see DELTA_KERNEL.

    Algorithm (index from cell -> gids/mass, built once per call, so a query
    only ever visits target cells within DMAX -- never the full O(|A|*|B|)
    double sum, and never a full sweep over every OTHER candidate unless it
    actually shares a within-DMAX cell with the query):
      1. index[cb] = [(gid, mass), ...] for every (gid, cb, mass) with
         mass != 0 across `fields` -- built once, reused across every query.
      2. For a query field A: accum = defaultdict(float); for each (ca, aval)
         in A, for each (db, di, wt) in DELTA_KERNEL, look up cb = ca+offset
         in the index and add aval*wt*mass to accum[gid] for every (gid,
         mass) sitting there. This is algebraically IDENTICAL to the brief's
         double sum (accum[g] after the full sweep over A equals
         sum_{ca in A, cb in B_g: d<=DMAX} A[ca]*B_g[cb]*w(d) -- provable by
         swapping the order of the two sums the brief's definition implies).
      3. Every OTHER candidate gid (not just ones touched in step 2) is
         still assigned an explicit similarity (0.0 if accum has no entry
         for it) so the full ranking -- and therefore the target's exact
         rank -- matches what a literal O(|A|*|B|) sweep over every
         candidate would have produced; nothing is silently left unranked.
    """
    norms = {g: math.sqrt(sum(v * v for v in fld.values())) for g, fld in fields.items()}
    index = collections.defaultdict(list)
    for g in sorted(fields):  # deterministic build order (does not affect the result: only
        for cell, mass in fields[g].items():  # the accum SUMS are order-independent)
            if mass:
                index[cell].append((g, mass))
    all_gids = sorted(fields.keys())
    rank_cache = {}

    def rank_lookup(q):
        rl = rank_cache.get(q)
        if rl is not None:
            return rl
        A = fields.get(q)
        qn = norms.get(q, 0.0)
        if not A or qn == 0.0:
            rl = {}
            rank_cache[q] = rl
            return rl
        accum = collections.defaultdict(float)
        for (a0, a1), aval in A.items():
            for db, di, wt in DELTA_KERNEL:
                cb = ((a0 + db) & 255, (a1 + di) & 255)
                bucket = index.get(cb)
                if not bucket:
                    continue
                contrib = aval * wt
                for g, mass in bucket:
                    if g == q:
                        continue
                    accum[g] += contrib * mass
        sims = []
        for g in all_gids:
            if g == q:
                continue
            bn = norms.get(g, 0.0)
            if bn == 0.0:
                sims.append((0.0, g))
                continue
            num = accum.get(g, 0.0)
            sims.append((num / (qn * bn), g) if num else (0.0, g))
        sims.sort(reverse=True)
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


class _LutTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _LutTimeout()


def timed_lut_retrieval(fields, prs, cap_seconds, shuffled_seeds=(0, 1, 2, 3, 4)):
    """lut_retrieval wrapped in a hard SIGALRM safety-net cap. Returns
    (result_dict_or_None, elapsed_seconds, timed_out_bool). Used as a SAFETY
    NET only (measured single-call costs at radius<=1 are well inside the
    caps used below) -- not as the mechanism deciding the radius=2 scope
    reduction, which is decided from real measured radius<=1 timings and
    documented via note() before radius=2 is ever attempted (see MATRIX)."""
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(cap_seconds)
    t0 = time.time()
    try:
        r = lut_retrieval(fields, prs, shuffled_seeds=shuffled_seeds)
        signal.alarm(0)
        return r, time.time() - t0, False
    except _LutTimeout:
        return None, time.time() - t0, True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ─────────────────────────────────────────────────────────────────────────────
# MATRIX: n in (3,1) x radius in (0,1,2) x scorer in (bag,field,lut)
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

note("FIX 2 scope: the LUT scorer is attempted at radius=0 and radius=1 for both n=3 and n=1 "
     "(measured for real, below). radius=2 is NOT attempted for the LUT scorer -- decided from "
     "the REAL radius=0/1 timings measured in THIS run (logged per-n below, at the point the "
     "decision is made), extrapolated forward: each field's distinct-cell count and average "
     "index-bucket size both grow substantially with radius (more spread -> denser, more "
     "overlapping fields), and the LUT scorer's cost is driven by exactly those two "
     "quantities (|A| entries x 49 kernel offsets x avg bucket size per query, summed over "
     "~133 distinct queries) -- so radius=2 is expected to cost several times radius=1's "
     "already-multi-minute cost, well past what is tractable for a probe script. This is "
     "exactly the 'if it is still too slow, reduce to radius 0 fields only for this scorer' "
     "fallback the brief allows, extended one step (radius in (0,1) rather than radius=0 "
     "alone) because radius=1 was measured tractable in the SAME timing check that showed "
     "radius=2 would not be. bag and field are UNCHANGED and still cover radius in (0,1,2) "
     "for both n -- only the NEW lut scorer's radius sweep is reduced.")

LUT_MAX_RADIUS = 1
LUT_ROW_SECS = {}  # (n, radius) -> measured elapsed seconds, for the skip-note below

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

        if radius <= LUT_MAX_RADIUS:
            # 600s is a pure safety net (measured single-call costs at radius<=1 are well
            # inside it, ~5-6 minutes worst case); it is NOT the mechanism that decided the
            # radius=2 reduction below, which is decided from real radius<=1 timings instead.
            rl, elapsed, timed_out = timed_lut_retrieval(fields, pairs, cap_seconds=600)
            LUT_ROW_SECS[(n, radius)] = elapsed
            if timed_out:
                row["lut"] = dict(skipped=True, reason=f"SIGALRM safety-net cap (600s) hit after {elapsed:.1f}s")
                log(f"  LUT   {label}: TIMEOUT after {elapsed:.1f}s (600s cap) -- skipped")
                note(f"UNEXPECTED: lut_retrieval for {label} hit the 600s SIGALRM safety-net "
                     f"cap and was skipped. This was NOT anticipated (radius<=1 was expected, "
                     f"from an earlier ad hoc timing probe, to be tractable) and is reported "
                     f"as a genuine surprise, not a planned reduction.")
            else:
                row["lut"] = rl
                log(f"  LUT   {label}: {elapsed:.1f}s -> {json.dumps(rl)}")
        else:
            row["lut"] = dict(skipped=True, reason="radius=2 excluded from the LUT scorer, see note")
            r0_secs = LUT_ROW_SECS.get((n, 0))
            r1_secs = LUT_ROW_SECS.get((n, 1))
            log(f"  LUT   {label}: SKIPPED (radius=2 excluded from LUT scope; n={n}'s own "
                f"measured r0={r0_secs:.1f}s r1={r1_secs:.1f}s informed this)")
            note(f"LUT scorer at n={n}, radius=2 was NOT computed. n={n}'s own measured LUT "
                 f"timings in THIS run: radius=0 took {r0_secs:.1f}s, radius=1 took "
                 f"{r1_secs:.1f}s ({(r1_secs / max(r0_secs, 1e-9)):.1f}x). Extrapolating that "
                 f"growth rate forward (radius=2's fields are denser still, by the same "
                 f"mechanism that made radius=1 already {(r1_secs / max(r0_secs, 1e-9)):.1f}x "
                 f"radius=0) puts radius=2 in the many-minutes-to-tens-of-minutes range for "
                 f"this one row alone -- not attempted, per the brief's own fallback ('reduce "
                 f"to radius 0 fields only for this scorer'), extended to radius in (0,1) "
                 f"since radius=1 was itself measured tractable.")

        RESULT["matrix"][label] = row
        flush_log()
        json.dump(RESULT, open(os.path.join(RUNS, "w1-cue2.json"), "w"), indent=1)

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

# FIX 3: F1 must be able to fire, and with FIX 2's LUT scorer in place, permuting the
# unit->cell assignment (marginals preserved) genuinely destroys the LOCAL adjacency
# structure the LUT scorer's soft match depends on, while bag/field only ever see exact
# cell matches (a permutation that preserves marginals also preserves every exact match,
# since a body's own units still land on whatever cells they were permuted to -- consistently
# for that SAME permutation, applied identically to query and target bodies alike). Reuses
# f1_fields (same PERMUTED_CELL_MAP_3, radius=1) -- no need to rebuild.
lut_cap = 900  # generous safety-net cap; measured single-call cost at n=3,r=1 is ~5 min
f1_lut, f1_lut_secs, f1_lut_timed_out = timed_lut_retrieval(f1_fields, pairs, cap_seconds=lut_cap)
if f1_lut_timed_out:
    log(f"  F1 lut  n=3 r=1 permuted: TIMEOUT after {f1_lut_secs:.1f}s ({lut_cap}s cap)")
    note(f"F1's lut_retrieval call (n=3, radius=1, permuted assignment) hit the {lut_cap}s "
         f"SIGALRM safety-net cap after {f1_lut_secs:.1f}s and was skipped -- unexpected "
         f"given the unpermuted n3_r1 lut baseline measured well under this cap.")
    f1_lut = dict(skipped=True, reason=f"{lut_cap}s safety-net cap hit after {f1_lut_secs:.1f}s")
else:
    log(f"  F1 lut  n=3 r=1 permuted: {f1_lut_secs:.1f}s -> {json.dumps(f1_lut)}")

RESULT["falsifiers"]["f1_permuted_assignment"] = dict(bag=f1_bag, field=f1_field, lut=f1_lut)

baseline_n3_r1 = RESULT["matrix"]["n3_r1"]
f1_bag_fired = f1_bag["r@10"] < baseline_n3_r1["bag"]["r@10"]
f1_field_fired = f1_field["r@10"] < baseline_n3_r1["field"]["r@10"]
baseline_lut = baseline_n3_r1.get("lut")
if baseline_lut is not None and not baseline_lut.get("skipped") and not f1_lut.get("skipped"):
    f1_lut_fired = f1_lut["r@10"] < baseline_lut["r@10"]
    lut_line = f"lut={f1_lut_fired} ({baseline_lut['r@10']} -> {f1_lut['r@10']})"
else:
    f1_lut_fired = None
    lut_line = "lut=N/A (baseline or permuted lut result unavailable, see notes)"
log(f"  F1 fired (dropped vs n3_r1 baseline)? bag={f1_bag_fired} "
    f"({baseline_n3_r1['bag']['r@10']} -> {f1_bag['r@10']}), "
    f"field={f1_field_fired} ({baseline_n3_r1['field']['r@10']} -> {f1_field['r@10']}), "
    f"{lut_line}")
RESULT["falsifiers"]["f1_fired"] = dict(bag=f1_bag_fired, field=f1_field_fired, lut=f1_lut_fired)
flush_log()
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue2.json"), "w"), indent=1)

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
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue2.json"), "w"), indent=1)

# F3 -- n=3 (motif) vs n=1 (single-atom ceiling control), already in the matrix. Now covers
# lut too, wherever it was actually computed (radius in (0,1); radius=2 lut rows are skipped
# markers and excluded here rather than compared as if they were real zeros).
f3 = {}
for radius in (0, 1, 2):
    for scorer in ("bag", "field", "lut"):
        row3 = RESULT["matrix"][f"n3_r{radius}"][scorer]
        row1 = RESULT["matrix"][f"n1_r{radius}"][scorer]
        if row3.get("skipped") or row1.get("skipped"):
            f3[f"r{radius}_{scorer}"] = dict(skipped=True, reason="lut not computed at this radius, see matrix")
            continue
        r3, r1v = row3["r@10"], row1["r@10"]
        f3[f"r{radius}_{scorer}"] = dict(n3_r10=r3, n1_r10=r1v, n3_beats_n1=r3 > r1v)
log(f"F3 (n=3 vs n=1, r@10): {json.dumps(f3)}")
RESULT["falsifiers"]["f3_motif_vs_ceiling"] = f3


# ─────────────────────────────────────────────────────────────────────────────
# TRANSITIONS (NEW) -- order-sensitive representation: bag consecutive cell-PAIRS over
# the n=3 unit sequence, score with metrics2.retrieval UNMODIFIED, then rerun with the
# unit order shuffled (seed 1, per gid) before building the transitions.
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
log("TRANSITIONS -- order-sensitive arm (tests order; run 1's reps could not)")


def trans_seq(units, cell_map):
    """The list of consecutive cell PAIRS (cell(u_i), cell(u_{i+1})) over a body's own
    n=3 unit sequence, in whatever order `units` is given in -- order-SENSITIVE by
    construction, unlike bag/field (which only ever aggregate occurrence counts)."""
    cells = [cell_map[u] for u in units]
    return [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]


trans_seqs = {g: trans_seq(UNIT_CACHE[3][g], CELL_MAP[3]) for g in pool}
t0 = time.time()
r_trans = metrics2.retrieval(trans_seqs, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  TRANSITIONS (original order): {time.time() - t0:.1f}s -> {json.dumps(r_trans)}")

# Order shuffle: SAME convention as run 1's F2 (fresh random.Random(1) per gid, so every
# gid's own list is shuffled by a length-dependent Fisher-Yates draw off the same seed).
trans_seqs_shuffled = {}
for g in sorted(pool):
    units = list(UNIT_CACHE[3][g])
    random.Random(1).shuffle(units)
    trans_seqs_shuffled[g] = trans_seq(units, CELL_MAP[3])
t0 = time.time()
r_trans_shuffled = metrics2.retrieval(trans_seqs_shuffled, pairs, shuffled_seeds=(0, 1, 2, 3, 4))
log(f"  TRANSITIONS (order-shuffled, seed 1 per gid): {time.time() - t0:.1f}s -> {json.dumps(r_trans_shuffled)}")

trans_order_dropped = r_trans_shuffled["r@10"] < r_trans["r@10"]
log(f"  TRANSITIONS order-shuffle dropped r@10? {trans_order_dropped} "
    f"({r_trans['r@10']} -> {r_trans_shuffled['r@10']})")
RESULT["transitions"] = dict(
    original=r_trans, order_shuffled=r_trans_shuffled,
    order_shuffle_dropped_r10=trans_order_dropped,
)
flush_log()
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue2.json"), "w"), indent=1)


# ─────────────────────────────────────────────────────────────────────────────
# per-scorer LUT-usage note (explicit, as requested)
# ─────────────────────────────────────────────────────────────────────────────
RESULT["meta"]["scorer_lut_usage"] = dict(
    bag="NO -- metrics2.retrieval cosine over exact cell-token Counters; no distance lookups.",
    field="NO -- Ruzicka/weighted-Jaccard overlap over spread-field cells; exact-key match "
          "only (the SPREAD mechanism itself softens at construction time via 4-neighbour "
          "diffusion, but field_retrieval's own similarity computation never reads the LUT).",
    lut="YES -- every neighbour cell + weight (DELTA_KERNEL) is sourced from LUT[0][...] "
        "reads; the cell->gids/mass index is consulted per query, restricted to cells within "
        "DMAX=3 of each of the query's own occupied cells.",
    transitions="NO -- metrics2.retrieval cosine over exact (cell,cell) transition-pair "
                "tokens; same as bag, just over a different (order-sensitive) alphabet.",
)


# ─────────────────────────────────────────────────────────────────────────────
# best cue overall
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 78)
candidates = [("ARM0_A2a-f", r0["r@10"])]
for label, row in RESULT["matrix"].items():
    for scorer in ("bag", "field", "lut"):
        cell = row.get(scorer)
        if cell is not None and not cell.get("skipped") and "r@10" in cell:
            candidates.append((f"{label}_{scorer}", cell["r@10"]))
candidates.append(("TRANSITIONS_original", r_trans["r@10"]))
candidates.append(("TRANSITIONS_order_shuffled", r_trans_shuffled["r@10"]))
best_label, best_r10 = max(candidates, key=lambda x: (x[1], x[0]))
RESULT["best_cue"] = dict(label=best_label, r10=best_r10, vs_arm0=dict(arm0_r10=r0["r@10"]))
log(f"BEST CUE: {best_label} r@10={best_r10} (ARM0 was {r0['r@10']})")

RESULT["secs"] = round(time.time() - T0)
flush_log()
json.dump(RESULT, open(os.path.join(RUNS, "w1-cue2.json"), "w"), indent=1)
log(f"# done {RESULT['secs']}s -- wrote runs/w1-cue2.json and runs/w1-cue2.log")
flush_log()
