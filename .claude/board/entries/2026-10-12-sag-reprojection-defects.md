# 2026-10-12 — D-SAG-1R — SAG reprojection defects A and B, found by #1510

**Status:** In PR. **Found by:** #1510's audit of #1509.

A: `SoftwareGraph::link` deduplicated on `(src, rel, dst, tier)`, so a fact two
methods share (the claim behind a field both read) belonged to whichever wrote
it first. When that method stopped reading the field, its reprojection
retracted the fact for both. `ctx` is now part of the key; `out` and
`capability` read each fact once.

B: `project_compute` never retracted the DAG it replaced. Cyclic then acyclic
kept the `ComputeCycle` refusal and the stale `ComputesFrom` edges. It now
retracts its own class's compute edges and verdict first.

MEASURED (tests/awareness.rs): for every method, reprojecting it with a body
that reads nothing gives the same facts as a fresh projection; three DAG
sequences give the expected edge count and verdict, the other class untouched.
Three disable runs red on the tests that name each guard.

OPEN: C, `certify` accepts any test name. Who may issue a verified receipt is
an operator decision, not a code fix.
