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

## Results (run 2026-11-13, after the conditions above were committed in `3c502e6`)

**Grade:** MEASURED on one machine (Intel Xeon 2.1 GHz, 4 cores, AVX-512,
release). Falsifiers: `cargo test --release -p quack-boundary-lab --test calibrated_plane`
(6 tests) and `-p dr256-ternlog-lab --test bound_planes` (4 tests). Cost:
`ERA5_T2M_F32=<file> cargo run --release -p dr256-ternlog-lab --example calibrated_planes`,
three runs; PR 2057's `bit_planes` example in the same session.

### Answer

Yes for a caller that states its calibration; no authority makes it state the
current one.

| falsifier | verdict | what happened |
|---|---|---|
| A same source, same floor | PASS | admitted; equals the scalar loop over the code bytes |
| B1 planes after a calibration change | PASS | the same eight `DerivedPlane`s pass `check_planes`; their words are unchanged |
| B2 method bound under floor A, described under floor B | REFUSED | `PlaneRefusal::Description`, through `Guard::witness` on the method the store admitted |
| B3 method bound under floor B, same planes | PASS | admitted; equals the scalar loop at floor B's threshold; thresholds and answers differ |
| B4 caller keeps describing with floor A | **NOT REFUSED** (pinned) | no interface in the path holds "the current floor"; the caller states it |
| C new timestep, frame advanced | REFUSED, then PASS | old binding: `LaneRefusal::Frame`; fresh binding with old planes: `PlaneRefusal::Frame`; fresh planes equal the scalar loop |
| D bytes rewritten, frame not moved | NOT DETECTED (pinned) | old planes admitted, answer from the old bytes; reproduces PR 2058 |
| E re-encoding under another floor | caught only if the owner advances its frame | both branches run; see below |

Disable runs, each red: witness comparison removed; plane frame check removed;
token made independent of the floor; predicate threshold fixed; and in the
slab module: all checks removed, row-order check removed, bit slicing shifted.

### The five identities, separately

| identity | established by | holds? |
|---|---|---|
| bit-plane representation | `Recipe::Bit`, frame, row order (`check_planes`) | yes, independent of calibration |
| calibrated predicate | DT64 method guard carrying `Sym(digest(floor version, theta, code threshold))` | yes against a mismatched description |
| source and frame | `LaneBinding`, `FrameStore::check` | yes; a moved frame refuses binding and planes |
| calibration interpretation | the caller's description only | stated, not issued: B4 |
| producer authority | none | not added; the bytes' producer and the floor's owner are both unwitnessed |

### E: who owns a re-encoding

`weather-poc` states the rule and does not carry it. The plan
(`weather-soa-bake-v1.md`, the codebook-edges item) says the floor's
`(lo, hi, floor_version)` is dataset metadata, and to "align any re-roll with
a version boundary". `lane.rs` says its `stamped_floor_versions` "stands in
for" that metadata. `bake.rs` takes floors and writes bytes; it records no
stamp and issues no version. Recorded as a **missing owner contract**: nothing
obliges the code-lane owner to move its frame when the floor changes, and
nothing ties a floor version to a Lance version in code. Lance versioning of
the rows does not cover a floor kept beside them.

### Cost (medians of 21 samples; ranges over three runs)

`DerivedPlane` is bound to a fixed 40,003-row fixture and cannot hold a
timestep, so these numbers are from `dr256-ternlog-lab::bound_planes`, which
keeps the same record per plane (frame, row order) under one `LaneBinding`
per slab. Query: `145 < code < 183` over the whole grid, eight planes, no
region plane.

| arm | one slab (65,536 rows) | one timestep (16 slabs) |
|---|---|---|
| unbound, contiguous planes | 2.4 to 2.7 us | 47 to 51 us |
| unbound, per slab | 2.9 to 3.2 us | 40 to 47 us |
| checks alone (binding + 8 planes per slab) | 213 to 233 ns | 3.4 to 3.7 us (215 to 233 ns per slab) |
| method admission, description built per call | 190 to 209 ns, 2 allocations (60 B) | the same |
| method admission, reference and witness resolved once | 20 to 21 ns, 0 allocations | the same |
| admitted, description built per call | 3.4 to 3.6 us | 42 to 49 us |
| admitted, resolved once | 3.4 us | 46 to 48 us |

- Warm allocations: 0 for checks and execution; 2 per call when the caller
  rebuilds its description each time (a formatted token and a guard). With
  the method reference and witness resolved once per binding: 0. The
  pre-registered "must be 0" FAILS for the per-call form and PASSES for the
  resolved-once form.
- At one timestep the checks are 3.4 to 3.7 us on a 40 to 47 us fold, about
  8%. At one slab they are about 230 ns on 3 us. The admitted and unbound
  timestep arms overlap within run-to-run spread.
- Per-slab planes are no slower than contiguous ones at 16 slabs (40 to 47 us
  against 47 to 51 us).
- Cold: 11 to 13 ms to derive eight planes for a timestep, 182 allocations,
  1.07 MB written; 1.04 MB resident against 4.15 MB for the widened lane.
- Same session, PR 2057's unbound benchmark (its range includes a region
  plane, nine planes): range count 42.7 us, band count 4.8 us, gated `i32`
  count 45.4 us.

Not measured here: the checks on the three-plane band count. Its fold is about
5 us, so a fixed 3.5 us of per-slab checks would not be small there.

### Decision

The split holds and needs no new issuer for the case tested: a resident
bit-plane is admitted on source, frame and row order alone, and a calibrated
predicate is refused when the description and the bound method disagree.
What the existing interfaces do not supply is an owner for "which floor is
current" (B4) and an obligation on the code-lane owner to version a
re-encoding (E). Neither is patched here.

**Open:** a carrier for the current floor that the admission path can read;
the bake recording its floor stamp with the version it writes; plane checks
on the cheap band fold; `DerivedPlane` over a source of arbitrary size.
