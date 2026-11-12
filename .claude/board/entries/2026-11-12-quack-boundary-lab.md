# 2026-11-12 — D-QUACK-BOUNDARY-LAB: a Quack program is admitted where its lanes are, not inside the program

**Kind:** lab crate + audit. No production API changed.
**Grade:** MEASURED (`cargo test --manifest-path crates/quack-boundary-lab/Cargo.toml`: 11 pass; four disable runs red).
**Plan:** `.claude/plans/quack-boundary-lab-v1.md`. **Linear (existing):** EXO-253.

- The consumer-side admission seam for lanes already exists: `LaneBinding` (`lance-graph-dt64/src/lane.rs`). One check covers the issuing source (store identity), the current frame, the class route and basis, row order and reader scope. Nothing used it for a `Program` before this lab.
- One unchanged program (`SUM(qty)` over a row range and a value predicate) run unbound over another source of equal shape and equal numbers, and over the same source after a coordinate revision, returns a plausible wrong sum. Through the binding it is refused (`ForeignFrame`, `CoordinatesChanged`).
- A kept mask of another program presented as a foreign plane is accepted by the executor and keeps answering with the old population after the foreign table changed. No owner issues a mask plane or a foreign plane, so the admitted path does not run such a program (BLOCKED). The same holds for `Filter::sql_where` over a validity plane and for every phase-2 program of a `GroupPlan`.
- Stale bytes under NULL do not reach the answer when validity is gated, for a predicate and for its negation.
- `lower_group_by_auto` folds SUM, COUNT, MIN and MAX to one program; ANY, ALL, Rows and `groups == 0` stay a forest; a grouped blend is refused. The forest hands 5,008 bytes of mask (40,003 rows) from its first program to the others.
- Ledger: every workload allocates zero bytes, the forest included. Scratch is 2,056 bytes per program at any row count. Only the forest has an intermediate.

**Read record:** in the plan, section 2. Not read: most of `dr.rs`, the crates' own test modules.

- After reading `lance-graph-dt64/src/lib.rs` and `lance-graph-report/src/lane_source.rs` in full: the only non-lab `LaneSource` (`AbiLanes`) has a `#[cfg(test)]` constructor, so the seam has no production-constructible source; and the lab never exercises method admission (`BindStore::admit`). Both added to the plan's list (items 10, 11); item 6 corrected (`CoordSpace` is source-reported, only `Scope` lacks an issuer).

**Agent runs:** none; read, written and run on the main thread.

**Open:** the eleven contracts in the plan, section 8. None is closed by this lab.
