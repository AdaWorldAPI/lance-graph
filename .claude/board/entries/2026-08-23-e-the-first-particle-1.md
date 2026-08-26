## 2026-08-23 — E-THE-FIRST-PARTICLE-1 — the substrate changed how it looked at the problem and can say exactly what changed

**Status:** FINDING (measured — `PROBE-FIRST-PARTICLE-1`, 5/5 gates green,
`examples/probe_first_particle.rs`). **Confidence:** High for the measured
claim; the scope fence is part of the finding.

**The particle, as the #1001 charter's laws require it.** One observed chain
of 63 links, closed once — **2016 beliefs, sealed by digest** — holds live
activity at every rung 0..6 simultaneously:

```
  R0 ▓ (63)   R1 ▓ (62)   R2 ▓▓▓ (121)   R3 ▓▓▓▓▓▓▓ (230)
  R4 ▓▓▓▓▓▓▓▓▓▓▓▓ (412)   R5 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (632)   R6 ▓▓▓ (496)
```

Over that field:

```
  View A [BoundAt(SupportedBy), Band(0..=2)]  → rows at rungs [1,1,2,2]
       edits = [RemoveAt(1), Push(Band(4..=6))]
  View B [BoundAt(SupportedBy), Band(4..=6)]  → rows at rungs [4,4,6,6]
```

- `reconstruct(A, edits) == B` on BOTH layers (descriptor stack and lowered
  visible set), and the inverse sequence restores A exactly.
- Arena digest, population digest, and base pointer identical after all
  viewing: **56 B of descriptors moved; 10,240 B of population did not.**
- **Territories overlap and one contribution lives in two phases at once:**
  windows low 0..=2 / mid 2..=5 / high 4..=6 are deliberately overlapping;
  an R2 row is visible under low AND mid, an R4 row under mid AND high, with
  exclusives on each side (P2). Not one-hot, executably.

**The rung ceiling is a measured constraint, recorded rather than smoothed:**
`Stamp::source(id)` is `1 << (id % 64)` (`belief.rs:36-38`), so evidential ids
≥ 64 alias and the largest collision-free observed chain is 63 links →
Tarski ceiling `ceil(log2 63)` = **rung 6**. The charter's R2/R5/R8 example
was explicitly non-mandatory; this probe reaches its spread by derivation,
not by decorating rows with invented rungs.

**Conservation laws exercised, none violated:** population not moved by an
attention change; rung activity not one-hot; the two selector families never
flattened into one meaning; the lowered view opaque while its contributing
layers stay identifiable; the edit changed the view, not the universe
underneath it; the change reconstructible.

**Unchanged and restated:** F-REVISION-FOCUS-1 is ABSENT — `ViewEdit` is
probe-local; nothing here touches `RungElevator`, `StyleLane`,
`EpistemicMode`, or `temporal.rs`; territory windows are view selectors, not
identities; occupancy is semantic, not wall-clock; no behavioral BPE; Rubicon
persistence open. The next questions this makes askable — which edit
sequences recur, which survive grounding, which compress — belong to the
learner, which does not exist and is not implied to.
