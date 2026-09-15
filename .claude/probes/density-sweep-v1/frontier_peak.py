#!/usr/bin/env python3
"""Resolve the 6,297-vs-142 peak-frontier discrepancy the PR body flagged.

TWO figures for "the MONDO `is_a` peak frontier" were published one arc apart and
the flag said *"both cannot stand as 'the' peak"*. That flag was itself wrong:
both are correct, because they are maxima over DIFFERENT SEED POOLS.

  142    max peak over 60 sampled INTERIOR seeds (`sweep.py`'s `walk_seeds`)
  6,297  peak from the ROOT, at hop 6

`sweep.py` draws its walk seeds from `[n for n in po if n[0] == MONDO and po[n]]`
-- nodes that HAVE PARENTS. A root has none, so **a root is excluded from that
pool by construction** and no amount of resampling inside it can ever reach the
root's frontier. MONDO has exactly 3 roots.

The exhaustive arm is what turns this from "two samplings" into a statement about
the graph: 6,297 is the GLOBAL maximum over every one of the ~6.2k nodes that has
children -- so it is not merely "a single seed", it is the widest frontier the
graph admits. That is STRONGER than how it was first reported, and it is the
right figure for a cost-of-no-torch argument, because a traversal seeded at the
ontology root is exactly the worst case a bounded frontier exists to bound.

Anti-vacuity: the exhaustive arm is what makes the root arm meaningful. Without
it, "the root is widest" would be an assumption about ontology shape rather than
a measurement -- a deep-but-narrow root and a wide interior hub are both a priori
possible in a DAG whose edges are `is_a`.

Requires the MONDO bake; skips (does not fail) when it is absent, matching
`sweep.py`'s own arm. Set OBO_CORE_SOA to relocate it.
"""
import os
import random
import struct
import sys
from collections import defaultdict

SOA = os.environ.get("OBO_CORE_SOA", "/home/user/MedCare-rs/.data/obo-core.soa")
STRIDE, EOFF, VOFF, LOFF, LB, LN = 512, 16, 32, 112, 16, 23
MONDO = 0x91010000
HOPS = 12

# Pinned from the measurement below. Exact, not relational: the whole point of
# this probe is that two DIFFERENT numbers are each right, so a bound like
# "peak >= 142" would pass while the distinction it exists to record was lost.
EXPECT_SAMPLED_INTERIOR = 142
EXPECT_ROOT = 6297
EXPECT_ROOT_HOP = 6
EXPECT_N_ROOTS = 3


def u24(b, o):
    """Read a 3-byte LITTLE-endian unsigned int at offset `o`."""
    return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16)


def load_parents(path):
    """Parse the SoA bake into the `node -> [parents]` map, rejecting partial bakes."""
    d = open(path, "rb").read()
    if not d or len(d) % STRIDE:
        raise ValueError(
            "MONDO bake must be non-empty and a whole multiple of STRIDE; "
            "len(...) // STRIDE silently discards a partial trailing row, so an "
            "incomplete download would report measurements from a prefix."
        )
    po = defaultdict(list)
    for r in range(len(d) // STRIDE):
        row = d[r * STRIDE:(r + 1) * STRIDE]
        cls = struct.unpack_from("<I", row, 0)[0]
        fam, idt = struct.unpack_from("<HH", row, 12)
        subj = (cls, (fam << 16) | idt)
        degs = [row[EOFF + i] for i in range(8)]
        sl = []
        for l in range(LN):
            lo = VOFF + LOFF + l * LB
            t = struct.unpack_from("<I", row, lo)[0]
            if t == 0:
                continue
            for kk in range(4):
                st = u24(row, lo + 4 + kk * 3)
                if st:
                    sl.append((t, st - 1))
        for _ in range(degs[1]):
            if sl:
                po[subj].append(sl.pop(0))
    return po


def children_of(parents):
    """Invert to the DOWNWARD adjacency the frontier walk needs."""
    ch = defaultdict(list)
    for node, ps in parents.items():
        for p in ps:
            ch[p].append(node)
    return ch


def profile(ch, seed, hops=HOPS):
    """Level-synchronous BFS. Returns (per-hop widths, peak width, total reached)."""
    seen, frontier, prof = {seed}, [seed], []
    for _ in range(hops):
        nxt = []
        for n in frontier:
            for c in ch.get(n, ()):
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
        if not nxt:
            break
        frontier = nxt
        prof.append(len(frontier))
    return prof, (max(prof) if prof else 1), len(seen)


if not os.path.exists(SOA):
    print(f"SKIPPED — {SOA} absent (31 MB bake, not committed; set OBO_CORE_SOA).")
    sys.exit(0)

po = load_parents(SOA)
ch = children_of(po)

subj_with_parents = {n for n in po if n[0] == MONDO and po[n]}
has_children = {n for n in ch if n[0] == MONDO}
roots = sorted(has_children - subj_with_parents)

print("=" * 92)
print("PEAK FRONTIER — the same graph, three seed pools, two correct answers")
print("=" * 92)
print(f"MONDO parents (nodes with children) : {len(has_children)}")
print(f"MONDO roots (children, NO parents)  : {len(roots)}")

# (a) reproduce sweep.py's sampling exactly.
ws = [s for s in sorted(subj_with_parents) if ch.get(s)]
ws = random.Random(11).sample(ws, min(60, len(ws)))
sampled_peak = max(profile(ch, s)[1] for s in ws)
print(f"\n(a) 60 sampled INTERIOR seeds  -> peak {sampled_peak}")
print("    A root cannot appear here: the pool is nodes that HAVE parents.")

# (b) the roots.
root_prof, root_peak, root_reach = max(
    (profile(ch, r) for r in roots), key=lambda t: t[1]
)
root_hop = root_prof.index(root_peak) + 1
print(f"\n(b) all {len(roots)} ROOTS               -> peak {root_peak} at hop {root_hop}")
print(f"    per-hop: {root_prof}")
print(f"    reached: {root_reach}")

# (c) exhaustive — the arm that makes (b) a measurement rather than an assumption.
glob_peak, glob_node = 0, None
for n in has_children:
    p = profile(ch, n)[1]
    if p > glob_peak:
        glob_peak, glob_node = p, n
print(f"\n(c) EXHAUSTIVE over all {len(has_children)} parents -> peak {glob_peak}")
print(f"    global maximum, so (b) is the widest frontier the graph admits.")

assert sampled_peak == EXPECT_SAMPLED_INTERIOR, f"sampled peak moved: {sampled_peak}"
assert root_peak == EXPECT_ROOT, f"root peak moved: {root_peak}"
assert root_hop == EXPECT_ROOT_HOP, f"root peak hop moved: {root_hop}"
assert len(roots) == EXPECT_N_ROOTS, f"root count moved: {len(roots)}"
assert glob_peak == EXPECT_ROOT, (
    f"the root is no longer the global maximum ({glob_peak}) — the claim that "
    "6,297 is the widest frontier the graph admits rests on this"
)
assert sampled_peak < root_peak, "the two pools must differ, or there is nothing to reconcile"
print("\nBOTH FIGURES STAND — different seed pools, both maxima, 6,297 is global.")
