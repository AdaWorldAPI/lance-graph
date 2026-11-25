# D-DIRSIM-DT64-PATHS — a user's location as bound DT64 coordinates

**Status:** In PR. **Date:** 2026-11-25.

## What

`lance-graph-dir-sim` (feature `dt64`) binds the OU path of a user's
location as DT64 lanes, beside UPN and primary SMTP:

- `dir:ouPath/k` (k = 1..16) names the lane `F_PATH1 + k - 1`. Its cell is
  the level code at depth k of the user's `Dn128`, read through the
  version's `View` per row. No DN is parsed or formatted, nothing is copied.
- `UserBasis::new(class, levels)` is the published basis: UPN, Primary
  SMTP, Path1..Path`levels` (clamped to 16). Consumers bind it instead of
  writing their own.
- `View::location(&Guid128)` returns a node's location in the version; it
  shares the read with `node_state` (one `slot_dn`, no second copy).

## Contract

- A Path`k` value identifies an OU **among its siblings only**. It is a
  coordinate, meaningful together with Path1..Path`k-1`; the name comes
  from the `OuDictionary` at render time.
- A user with no location, or a shallower one, has no value at that level.
- The domain is the version's scope (one value per version), not a row
  field.
- Only the canonical spelling binds: `dir:ouPath/01`, `/0`, `/17`, `/+1`
  are refused as `Unbound`.
- `LANE_REV` is unchanged: the path lanes are additive, a basis without
  them binds exactly as before.

## Evidence

`tests/dt64_paths.rs`, 4 tests: basis names and order; each level read
against the fixture's own codes with 0 allocations on the read path (5
users × 3 levels, 5 present values); a simulated `SetLocation` read through
the simulated version; canonical predicates only.

## Why

The report-rs IAM visual POC (slice S3) needs Path1..n as bound dimensions
over the version, not as strings parsed from DNs. The ADUC tree in report-rs
currently derives paths by rendering `ad::project` DNs and parsing them; with
these lanes a consumer can fold on location coordinates directly.

## Open

- VRL (`lance-graph-vrl`) needs contiguous `i32`/`u32` lanes and cannot yet
  read `UserLane`, which is virtual (one view read per row). Not addressed.
- Method descriptors for the user class (e.g. a move) are not registered.
- Borrowing a lane through `LaneBinding::lane()` allocated 3 times per call
  in this test (the cell reads after it allocate 0). Measured, not traced;
  `dt64_lanes.rs` measures `selected()` at 0. Belongs to DT64's check path,
  not to the dir-sim lanes.
