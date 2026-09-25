# 2026-09-25 — A facet aperture prefix lowers to a Range in the planner

**Status:** TEST-PINNED (`lance-graph-contract` `ordered_lane` tests, `lance-graph-quack` `diamond_lowering_tests`; disable-verified) · OPEN (see below)
**D-ids:** none new. Closes the "aperture→Range lowering in the planner" item in `2026-09-25-aperture-masks-and-kernel-gap.md`, and extends D-DIAMOND-1 R2 from whole tiles to bits.

## What landed
- `facet::SemanticAperture { pattern, care }` states the consulted bits of one facet. This is the HHTL partial mask ("HEEL + HIP, TWIG free", or "rail 0's coarse byte") that a whole-tile `SemanticPrefix` cannot name.
  - `matches` is `((f ^ pattern) & care) == 0`.
  - `prefix_bits` is `Some(b)` when the care, read in semantic order, is a run of ones from the coarse end.
  - `interval` then gives the closed key interval `[pattern & care, pattern | !care]` in numeric projection order.
- `SealedFacetLane::bound_aperture(w, aperture)` validates the witness and the lens, then locates that interval with two `partition_point`s. It returns `Ok(None)` for an aperture with a hole.
- `Filter::aperture_facet` (quack):
  - a prefix aperture under a validated witness becomes one `Cmp::Range`;
  - without a witness, or with a rejected one, it becomes the `MatchU64` sweep over the two semantic planes;
  - an aperture with a hole always sweeps.
  - `ApertureLowering` reports which fold was used, and why.
- A tile-aligned aperture lowers to exactly the same `Filter` as `prefix_facet`.

## The in-place sweep
- New quack leaves `Cmp::{EqU32Strided, NeU32Strided, MatchFacetStrided}` lower one-to-one to the mask-risc strided predicates. A facet stored inside a wider record (a `NodeRow`) is queried where it sits.
- `Filter::aperture_facet_strided` sweeps an aperture over two views of the same bytes:
  - the classid at +0, compared with `EqU32Strided`;
  - the 12 tier bytes at +4, compared with `MatchFacetStrided`.
- A caller holding only record bytes therefore never extracts the semantic planes.
- The bound decision is shared with `aperture_facet`: a witnessed prefix becomes the same `Range` on either sweep.
- Tests at a 40-byte and at the 512-byte stride, with junk around every facet:
  - the three leaves match a plain reading of the stored bytes;
  - the in-place sweep, the plane sweep and the aperture agree on random apertures, with and without holes;
  - a partial classid care is refused.
- Disable runs, each red then green:

| disable | tests that went red |
|---|---|
| leaf pattern/care swapped | 2 tests |
| tier bytes read at +0 | 1 test |
| classid leg dropped | 2 tests |
| partial classid accepted | 1 test |

## Evidence
- Every bit-prefix aperture from 0 to 128 bits bounds to exactly its matches, over 3000 sealed keys at four probe keys. Prefixes that end inside a tile are included; an anti-vacuity count requires such sub-tile cuts to actually split the population.
- At the lowering, the Range, the sweep and the row oracle agree for bit prefixes. A hole aperture never becomes a Range, even under a valid witness. A stale witness sweeps and reports `VersionMismatch`.
- Disable runs, each red then restored green:

| disable | tests that went red |
|---|---|
| `interval` given for holes | 1 contract test |
| both hole guards removed | the quack hole test |
| `prefix_bits` always `None` | 3 quack tests |
| upper bound excludes `hi_key` | 2 contract tests |

## OPEN
- The `Range` is ordinals in the sealed lane's order. The precondition on `prefix_facet` (`ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER`) applies unchanged.
- An aperture that cares about only PART of the classid has no in-place spelling: the strided classid reader is an equality, and no strided u32 ternary match exists. `aperture_facet_strided` refuses it (`None`), and it lowers through the plane sweep.
- No caller mints apertures yet. The HHTL cascade and the traversal frontier are the intended producers.
