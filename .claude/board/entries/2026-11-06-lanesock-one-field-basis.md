# One field basis for the lane socket; test-only scope admission (D-LANESOCK-3)

**Date:** 2026-11-06. **Status:** in PR. Follows D-LANESOCK-2 (#1969, merged `53e849a`).

## Defect (RED, recorded at `49d11c2`)

DT64 validated ClassView A; `AbiLanes` resolved its own ClassView B with the same class, revision and field count but the first two fields swapped. The bind succeeded and A's `revenue` field borrowed B's `region` lane: a wrong, nonzero value with every check passing. DT64's basis check compares the field count, so it cannot see a reorder.

## Fix

- `LaneSource::lane_of(.., field, basis: &FieldRef, frame)`: DT64 passes the basis entry it resolved, at bind and on every check. A source maps that field, never a position.
- `AbiLanes` drops its ClassView. A check through a reordered view refuses `MappingChanged`; a bind under that view borrows the lane it names.
- The three implementers were updated (dt64 test source, dir-sim lab, `AbiLanes`).
- `AbiLanes::unwitnessed` and its suite moved into a `#[cfg(test)]` unit module; the `unwitnessed-scope` feature was removed. No build, `--all-features` included, constructs an `AbiLanes`.

Disable run: accepting a changed mapping in `LaneBinding::check` turns the basis test red.

## Handoff

Spec §8: Cockpit / Hub / Matrix consume one scoped projection through DT64 → borrowed lanes → Quack; reader-scope witness inventory (OGAR `RbacAuthority` → `RbacBinding` → `authorize_scoped` / `authorize_memberships`); the missing seam is a witness sealing a decision into `dr::Scope`. Not wired.
