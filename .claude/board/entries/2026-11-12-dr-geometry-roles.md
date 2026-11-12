# 2026-11-12 — D-DR-SELECT-K: geometry roles, masks against a cube, gap register (docs only)

**Kind:** plan amendment and a read-only census. Ships no code.
**Grade:** the crosswalk is VERIFIED-IN-CODE at `main` `9d81fd929`; the
reading of the four geometries as roles is a WORKING-MODEL; §9's three
results are exact integers from a small oracle.
**Plan:** `.claude/plans/dr-select-k-v1.md` §8 to §12.
**Linear (existing):** EXO-253, EXO-220, EXO-255, EXO-114, EXO-46, EXO-140.

- **Roles.** DR64 and DR256 are the native bounded focus; `Axis::for_survivors`
  already picks the narrow tile when a selection fits. DR{dynamic} is the
  escape hatch for a larger or irregular population. DR256³ exists only as
  the reservation shape `Extent`; no constant has three 256-wide axes and
  no 256³ fold exists.
- **Three masks are not a cube.** 96 B of axis masks express exactly the
  products A × B × C. Oracle: a product with popcounts 10, 12, 16 counts
  1,920 with 38 coordinates visited; the relation `(x + y + z) mod 256 = 0`
  has 65,536 triples while its axis masks and its pairwise projections are
  all full; one parity constraint on the product halves the count, so the
  product of popcounts is then wrong.
- **Gap that matters most.** No admitted consumer folds over a list of
  positions, and both executors rule it out in their own text (mask-RISC
  A1; Quack matrix row R1). The lab's positions arm in PR 2035 is a scalar
  loop measuring an access pattern. The lawful candidate inside the mask
  vocabulary is a per-word sparse walk for the gated predicate, which
  `masked_sum_i32` already does for sums; not measured.
- **Next measurement, proposed:** D-DR-SELECT-K-1b, N = 65,536 with an
  existing sealed order, top 64 and top 256 bound into a native focus,
  against the N-wide mask path.

**Open:** which operations need a 256³ domain; the SelectionVector role
against the two executors' rule; the four gates of plan §5.
