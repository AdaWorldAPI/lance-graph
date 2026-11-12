#!/usr/bin/env python3
"""Three axes of 256 positions: when do three masks stand for a cube?

Positive control: a factorized selection S(x,y,z) = A(x) and B(y) and C(z).
Its count is popcount(A) * popcount(B) * popcount(C); the reference oracle
enumerates only the selected triples. No 256^3 domain is built.

Negative control: the relation R = {(x,y,z) : (x + y + z) % 256 == 0}. Its
count is 65,536. Its three axis projections are all full, so the tightest
three independent masks describe 256^3 triples; its three pairwise
projections are all full as well, so pairwise relations do not describe it
either.

Run: python3 .claude/probes/2026-11-12-dr-geometry/three_axis_oracle.py
"""
N = 256

def splitmix(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)

def axis(seed, keep):
    out, state = [], seed
    for i in range(N):
        state, r = splitmix(state)
        if r % 100 < keep:
            out.append(i)
    return out

# ---- positive control: sparse independent axes
A, B, C = axis(1, 6), axis(2, 6), axis(3, 6)
product = len(A) * len(B) * len(C)
enumerated = sum(1 for _ in ((x, y, z) for x in A for y in B for z in C))
assert product == enumerated and 0 < product < N ** 3
print(f"factorized\tpopcounts={len(A)},{len(B)},{len(C)}\tcount={product}"
      f"\tcoordinates_visited={len(A) + len(B) + len(C)}"
      f"\tmask_bytes={3 * N // 8}\tdense_cube_bytes={N ** 3 // 8}")

# ---- negative control: a relation that does not factorize
count = 0
px, py, pz = set(), set(), set()
pxy, pxz, pyz = set(), set(), set()
for x in range(N):
    for y in range(N):
        z = (-x - y) % N          # the one z for this (x, y)
        count += 1
        px.add(x); py.add(y); pz.add(z)
        pxy.add((x, y)); pxz.add((x, z)); pyz.add((y, z))
axis_product = len(px) * len(py) * len(pz)
assert count == N * N
assert axis_product == N ** 3 and axis_product != count
# pairwise: every pair is present in all three projections, so the join of
# the three pairwise relations admits every triple, 256^3, not 65,536.
assert len(pxy) == len(pxz) == len(pyz) == N * N
pairwise_join = sum(1 for x in range(8) for y in range(8) for z in range(8)
                    if (x, y) in pxy and (x, z) in pxz and (y, z) in pyz)
assert pairwise_join == 8 ** 3      # sampled 8^3 corner: all admitted
print(f"non_factorizable\tcount={count}\taxis_mask_product={axis_product}"
      f"\tpairwise_projections_full=yes\tsparse_triple_bytes={count * 3}"
      f"\tdense_cube_bytes={N ** 3 // 8}")

# ---- the factorized set under one extra cross-axis constraint
constrained = sum(1 for x in A for y in B for z in C if (x + y + z) % 2 == 0)
assert constrained != product
print(f"factorized_plus_cross_constraint\tcount={constrained}"
      f"\tproduct_of_popcounts={product}\tproduct_is_wrong=yes")
