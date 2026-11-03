# 2026-11-03 — D-BAG-FOLD-V3: Arm L against production witness contracts

**Status:** MEASURED, lab only (`.claude/probes/2026-11-03-d-bag-fold/probe-v3/`, `REPORT-V3.md`, source `d651b70d4`, rustc 1.99.0). Not promoted; EXO-140 unchanged; no DT64 behaviour; `MaskOp::Gather` only in the lab crate.

- 26/26 checks: oracle parity on every root of two fixtures, plus negative controls and disables.
- **Production contracts that refuse correctly:**
  - `SealedFacetLane::validate` refuses a stale lane version and another lane's witness;
  - `FrameStore::check` refuses a foreign frame, and a stale frame when the presence writer advances it.
- **Missing witnesses** (each closed only by a lab guard whose disable gives a silent wrong answer):
  - presence-plane binding: a permuted plane or a deletion without a frame advance is undetected;
  - key uniqueness: `seal` accepts duplicates;
  - pointer bounds: `Gather` zero-fallback;
  - pointer provenance: value-derived and cross-ClassId pointers are undetected.
- Codebook/lens mismatch is not constructible: one `SemanticLens` variant, and `seek_from` takes no lens.
- **Cost:**
  - pointer preparation costs 1.1–1.2× the production `seal` (measured at 2,340 and 17,476 nodes), which itself crosses 1 ms at 69,904 nodes (failed gate, not reported as a number);
  - execution is 0 allocations and in microseconds, but 25–250× a gather-free baseline, because each `Gather` is gated by `T_in`, not by the candidate `Range`.
- **Remaining blocker:** missing witnesses plus the admission ruling, not execution cost.
- **OPEN, owner decisions:**
  - the ruling itself;
  - whether per-version preparation falls under the no-ms rule's cold-cost exception;
  - range-proportional gating in Quack lowering.
