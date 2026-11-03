# D-BAG-FOLD v3: Arm L against production witness contracts

**Status:** MEASURED, lab only. Arm L is not promoted. EXO-140 is unchanged. No DT64
behaviour is implemented. mask-risc, Quack and production lowering are unchanged;
`MaskOp::Gather` runs only inside `probe-v3/`.

Source: lance-graph `d651b70d4` (`main` after #1937). Toolchain rustc 1.99.0,
release, 4 vCPU Xeon @ 2.10 GHz, load ≈ 0.4. Run:

```sh
cd .claude/probes/2026-11-03-d-bag-fold/probe-v3
CARGO_PROFILE_DEV_DEBUG=0 cargo +1.99.0 run --release -q   # exits 1 on any failed check
```

Result: **26 passed, 0 failed.**

## What changed from v2

| v2 (probe-local mirror) | v3 (production contract) |
|---|---|
| `u64` keys, hand-rolled FNV digest | `FacetCascade` keys under `SemanticLens::CanonHighTiles8` |
| local `Witness` struct | `SealedFacetLane::seal`, `OrderedLaneWitness`, `validate` |
| `binary_search` | `SealedFacetLane::seek_from` (witnessed, per-depth cursor) and `bound` (witnessed prefix range) |
| local `presence_version` | `lance_graph_dt64::dr::FrameStore` / `Frame` / `FrameRefusal` |
| — | fixtures at 340, 2,340, 4,680, 17,476 and 69,904 nodes; oracle parity on every root at 340 and 4,680, and on the timed root at every benchmark size |

Still lab-local, because no production contract exists for any of them: provenance
check, pointer bounds check, strict key uniqueness, presence-plane digest.

## Results

| # | control | production contract | lab guard | contract-only (disable) |
|---|---|---|---|---|
| 1 | oracle parity: 4 roots × fanout 4 (340 nodes) and 8 roots × fanout 8 (4,680 nodes), every root | — | all on | — |
| 2 | permuted S plane, key lane untouched | `validate` Ok, `Frame::check` Ok: **not detected** | refused (`PresenceChanged`) | silent `(false,0)` vs oracle `(true,27)` |
| 3a | presence deleted, writer advances `FrameStore` | **refused** (`Frame(SourceChanged)`) | — | — |
| 3b | presence deleted, frame not advanced | **not detected** | refused (`PresenceChanged`) | — |
| 3d | lane resealed at a new `LanceVersion` | **refused** (`VersionMismatch {1, 2}`) | — | — |
| 4 | duplicate key | `seal` **accepts it** | refused (`DuplicateKey`) | Count 28 vs distinct 27 |
| 5 | `ib[j] = n_rows` | no check; `Gather` zero-fallback | refused (`PointerOutOfBounds`) | silent `(true,26)` vs `(true,27)` |
| 6a | another ClassId holds the only valid chains | prefix `bound` excludes it: correct `(false,0)` | — | — |
| 6b | edge crossing the ClassId boundary | — | refused at build (`NonRefiningEdge`) | — |
| 6c/d | pointer redirected into another ClassId, same tiers | **not detected** | refused (`Provenance`) | passes |
| 6e | witness of another lane, same version | **refused** (`RowCountMismatch`) | — | — |
| 6f | frame sealed by another `FrameStore` | **refused** (`Frame(ForeignFrame)`) | — | — |
| 6g | wrong codebook / lens | **not constructible**: `SemanticLens` has one variant, and `seek_from` checks no lens | — | — |
| 7 | `ib` filled from S's source endpoint values | **not detected** | refused (`Provenance`) | silent `(false,0)` vs `(true,18)` |
| 8 | drop `Semijoin(ic→S_in)` / `Semijoin(ib→R_in)` | — | — | both red against the oracle; the full fold equals it |

Every lab guard has a contract-only run (the disable) that produces a silent wrong
answer, so each one is load-bearing.

## Cost (release, medians of 31)

No-ms rule: a cell of 1 ms or more is a failed gate and is printed as `FAILED-GATE`,
never as a number.

| nodes | production `seal` µs | pointer prep µs | verify µs | exec ns (Any + Count) | gather-free baseline ns | exec allocs | prep allocs | reads in same 64-row word |
|---|---|---|---|---|---|---|---|---|
| 2,340 | 59.0 | 69.2 | 47.0 | 11,025 | 441 | 0 | 2 (18,720 B) | 70.4 % |
| 17,476 | 451.1 | 508.5 | 383.0 | 90,993 | 668 | 0 | 2 (139,808 B) | 47.6 % |
| 69,904 | FAILED-GATE | FAILED-GATE | FAILED-GATE | 380,419 | 1,503 | 0 | 2 (559,232 B) | 47.6 % |

What the numbers show:

- **Preparation scales with the order contract, not with Arm L.** Pointer
  preparation, with one witnessed `seek_from` cursor per ancestor depth, costs
  1.1–1.2× the production `seal` at 2,340 and 17,476 nodes. At 69,904 nodes the production
  `seal` alone crosses 1 ms. A first version that galloped every seek from row 0 was
  3–4× slower; it was replaced.
- **Preparation is per key version, not per query or per presence change.** `$i`
  depends on keys only. A presence change needs a new frame or plane digest, not new
  pointers.
- **Execution is allocation-free and stays in microseconds**, but it is 25–250× the
  gather-free baseline. The cause is visible in the lowered program:

  ```
  Pred{Range, under: Plane(0)}, Gather{lane 0, under: Plane(0)}, And,
  Gather{lane 1, under: Plane(0)}, And → Count
  ```

  Each `Gather` is gated by `T_in` (a mask), not by the candidate `Range` (a
  predicate). So it visits every T-bearing row of the whole lane, not only the 4,369
  candidates. That is a property of Quack's lowering, not of the witnesses. Arm L is
  also two Programs (Any, Count) where one would do.
- **Locality.** Every pointer lands inside the candidate subtree range (by
  construction: ancestors precede descendants in key order). The largest distance is
  4,352 rows at 4,369-row ranges. 47.6 % of reads fall in the reading row's own
  64-row word.

## Decision matrix

| item | verdict | evidence |
|---|---|---|
| Correctness of Arm L on the hierarchical fixture | **GO** (lab) | §1, 6a, 8; 26/26 |
| Key-lane order and version witness | **GO** — production | 3d, 6e (`validate`, `seek_from`, `bound`) |
| Frame identity and staleness | **GO** — production, conditionally | 3a, 6f. Holds only if every presence writer advances `FrameStore` (3b) |
| Presence-plane binding (plane order, plane content) | **NO-GO** — missing witness | 2, 3b. No contract ties a plane to the key lane or a version. The lab digest catches a change after binding, but not a plane built misaligned from the start |
| Key uniqueness | **NO-GO** — missing witness | 4. `attest_sorted` accepts non-decreasing lanes |
| Pointer bounds | **NO-GO** — missing guard | 5. `Gather` zero-fallback answers silently; mask-risc is frozen, so the guard must sit before the fold |
| Pointer provenance | **NO-GO** — missing witness | 6c, 7. No contract type records how `$i` was produced; only a lab runtime re-check refuses value-derived pointers |
| ClassId through the key prefix | **GO** — production | 6a, 6e |
| Codebook / lens identity | **NEEDS OWNER DECISION** | 6g. One `SemanticLens` variant; `seek_from` takes no lens. Untestable until a second lens exists |
| Execution cost | **GO** under the no-ms rule; **NEEDS OWNER DECISION** to make it proportional to the candidate range | µs, 0 allocs; gate-by-`Range` would need a Quack lowering change, which is out of scope |
| Preparation cost | **NEEDS OWNER DECISION** | ≥ 1 ms at 69,904 nodes for the production `seal` itself; whether per-version cold preparation falls under the no-ms rule's scoped exception is the operator's call |
| Admission of `Gather` through `$i` in execution | **NEEDS OWNER DECISION** | unchanged: ADMISSION B vs the 2026-10-21 `$i` doctrine, EXO-140 (`ARM-L-DECISION-MEMO.md`) |
| DT64 WHEN/DO route | not exercised | independently gated |

## The remaining blocker

It is not execution cost. Execution is allocation-free and in microseconds; making it
proportional to the candidate range is an optimization, not a gate.

It is **missing witnesses plus the unresolved admission ruling**:

1. **Four witnesses with no production contract:**
   - presence-plane binding (order and content);
   - key uniqueness;
   - pointer bounds;
   - pointer provenance.

   Each has a lab guard proven load-bearing by its disable run, so each is specified;
   none exists.
2. **Codebook/lens identity** cannot be exercised with one lens variant.
3. **The admission ruling** (memo §D) is still open.

Preparation cost is a third, separate question. It is the production `seal`'s cost
class, so it is a scope question for the no-ms rule, not an Arm L defect.

## Limitations

- One hierarchical layout, fixed depth 4. A candidate is a depth-4 row under one root.
- The lab guards are probe code; their costs are inside "verify" and "pointer prep".
- `FrameStore` advance on presence writes is called by the probe's own writer. No
  production writer exists to check.
- Timings are single-machine medians on a shared 4-vCPU host; repeated runs moved
  `exec_ns` at 69,904 nodes between 380,419 and 559,372.
