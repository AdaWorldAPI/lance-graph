# 2026-11-13 — D-QUACK-DERIVED-PLANE-PROBE P1: twins rules over bound lanes (lab)

**Kind:** lab probe in `crates/quack-boundary-lab` (`tests/twins_lanes.rs`). No production crate changes.
**Grade:** MEASURED (tests). Pass conditions were committed in `8b689d2fa` before the test file existed.
**Plan:** `.claude/plans/d-quack-derived-plane-probe.md` sections 4 and 11.

The twins rule shape (`hidden_state_twins_probe::Family`) is decided twice:
over raw vectors through `Bound::decide`, and over lanes taken through one
DT64 `LaneBinding`.

| case | bound path | raw path |
|---|---|---|
| right source | equals the row loop | equals the row loop |
| another source, same row count | refused (`ForeignFrame`) | accepted, wrong |
| frame advanced after binding | refused | accepted |
| another row order | refused | accepted, wrong |
| lane rewritten by another producer, frame not moved | **admitted** | accepted |
| the same with the frame moved | refused | accepted |

6 tests; 4 of 4 disable runs red.

**Finding:** the binding establishes source, frame, row order and field
mapping. It does not establish what wrote a lane's contents. The producer
digest carried in a guard is compared with the caller's description of the
method, not with the lanes. A rewrite is refused only when the source itself
advances its frame.

**Not done:** P2 and P3 (bit-planes, fk planes) wait for PR 2057. P4 (cost) not run.
