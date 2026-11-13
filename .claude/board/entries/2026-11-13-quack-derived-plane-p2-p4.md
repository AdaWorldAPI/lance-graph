# 2026-11-13 — D-QUACK-DERIVED-PLANE-PROBE-P2-P4: what a derived plane inherits (lab)

**Kind:** lab probe in `crates/quack-boundary-lab` (`src/derived.rs`, `tests/derived_plane.rs`, `tests/derived_plane_alloc.rs`, `examples/derived_plane_cost.rs`). No production crate changes; `LaneSource` and `LaneBinding` unchanged.
**Grade:** MEASURED. Pass conditions committed in `b0272382c` before the tests ran.
**Plan:** `.claude/plans/d-quack-derived-plane-probe.md` sections 12 and 13.

7 new tests (31 in the crate); 5 of 5 disable runs red; 0 warm allocations.

- Expressible through the existing socket: the input lane's admission at derivation, the plane's frame against the source's store, the plane's row order against the binding.
- Not expressible: which plane a program expects (a lab-local recipe comparison), that the words follow the recipe (no issuer), and a rewrite under an unmoved frame (not detectable).
- A resident plane keeps answering from the old bytes after such a rewrite while the lane path reads the new ones.
- Three kinds, three records: a bit-plane needs one field and a bit; a guard plane verifies the method but not the lowering; an fk plane's record cannot name the foreign population, and two visibility sets give indistinguishable records.
- Cost at 40,003 rows, AVX-512: binding check 64 to 68 ns, three plane checks 80 to 82 ns, unchecked execution 176 ns, whole admitted call 489 to 495 ns.

**Open:** no producer witness for lane contents; no carrier for a foreign population's frame in a derived plane; plane identity has no socket.
