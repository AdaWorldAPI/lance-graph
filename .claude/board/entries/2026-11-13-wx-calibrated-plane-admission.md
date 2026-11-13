# 2026-11-13 — D-WX-CALIBRATED-PLANE-ADMISSION-1: representation stays, interpretation is refused (lab probe)

**Kind:** lab probe in `crates/quack-boundary-lab` (falsifiers, on the
existing `DerivedPlane`) and `crates/dr256-ternlog-lab` (cost at weather
scale). No production crate changes.
**Grade:** HYPOTHESIS until the results section is filled; everything above
it was written before any candidate ran.
**Follows:** D-WX-DR256-TERNLOG (PR 2055), D-WX-BIT-PLANES (PR 2057),
D-QUACK-DERIVED-PLANE-PROBE (PR 2058).

## Question

Can an admitted, resident bit-plane remain valid as a pure representation
while a temperature predicate derived from it is refused when its calibration
interpretation changes?

## Two things kept apart

| thing | depends on | carrier used here |
|---|---|---|
| bit-plane `Bit { field, k }` | the code bytes, the source frame, the row order | `DerivedPlane` (PR 2058), checked by `check_planes` |
| temperature predicate `T > theta` | all of the above, and the floor that turns `theta` into a code threshold | a registered DT64 method whose guard carries `SetExpr::Sym(digest(floor version, theta, code threshold))`; admission compares the caller's description with the bound guard (`Guard::witness`), the mechanism PR 2056 uses for a producer |

No issuer is added. If the refusal needs an authority these interfaces cannot
supply, that is the result.

## Fixture

`quack-boundary-lab`'s `Source` (40,003 rows). The code of a row is the low
byte of its `amount` lane. Floors are `weather_poc::floor::CalibratedFloor`,
calibrated from two different kelvin samples so that one physical threshold
maps to two different code thresholds. `DerivedPlane` cannot hold a weather
timestep (its source is a fixed-size fixture), so cost at 16 slabs is measured
in `dr256-ternlog-lab` with a lab-local record of the same checks (per-slab
`LaneBinding`, the frame each plane was derived under, the row order).

## Pass conditions, fixed before running

- **A, same source, same floor.** Eight `Bit` planes of the admitted code
  lane; the predicate under floor A, admitted, equals a scalar loop over the
  code bytes. FAIL on inequality or refusal.
- **B, same bytes, changed calibration, frame unchanged.**
  B1: the eight planes still pass `check_planes`. B2: the method bound under
  floor A, described under floor B, is REFUSED. B3: the method bound under
  floor B, described under floor B, is admitted over the same planes and
  equals the scalar loop at floor B's threshold; the two thresholds and the
  two answers differ. B4, the limit to establish: does anything refuse a
  caller that still describes with floor A after the calibration changed?
  If nothing does, that is recorded as the missing boundary, not patched.
- **C, new timestep, same floor.** Code bytes change and the source advances
  its frame. Under a fresh binding the old planes are REFUSED by the frame
  check; freshly derived planes equal the scalar loop on the new bytes.
- **D, undeclared mutation.** Code bytes change, frame does not move. Expected
  from PR 2058: the old planes are admitted and answer from the old bytes.
  PASS means this is reproduced and pinned, not fixed.
- **E, re-encoding.** The same physical values quantized under floor B give
  other code bytes. In the lab both branches are run (owner advances the
  frame; owner does not). Whether the real code-lane owner (`weather-poc`'s
  bake) issues a new version on re-encoding is answered by reading its
  source; absent a stated guarantee it is recorded as a missing owner
  contract.
- **Cost.** One slab and one ERA5 timestep (16 slabs), same machine and run
  as PR 2057's unbound bit-plane range count: cold derivation time and
  resident bytes; warm checks alone, execution alone, both; method admission
  alone; warm allocations (must be 0). No number from PR 2058's 40,003-row
  run is carried over.

## Results

Not run yet.
