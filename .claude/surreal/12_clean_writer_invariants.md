# 12 — clean_writer_invariants  (POC green-light)

**Repo:** lance-graph (+ ndarray golden-byte refs) · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** the safety net — "damit keine Unfälle passieren".

## Scope
The verification suite that proves the POC writer is clean. **Tests only.**

## Owns
- `crates/surreal_container/tests/invariants.rs`
  (+ references the ndarray golden-byte tests from tasks 02/03)

## Depends on
all (02–11).

## Guards — do NOT touch
- **Tests only — no production code.**

## Acceptance — the invariants (all must be green)
- (a) **one Lance fragment per epoch** — no fragmentation.
- (b) **append-only** — no record/fragment overwrite.
- (c) **no data loss** — `fold` recovers all committed state.
- (d) **disjoint container-keys** — no same-key collision in a batch → LWW safe by
  construction (the clean-writer invariant for the POC).
- (e) **roundtrip** — write → read → decode == input.
- (f) **alignment** — decoded buffers usable by SIMD load without misalignment.
- (g) **cross-crate parity** — bytes encoded in ndarray (task 03) decode identically
  in lance-graph (task 05). This is the anti-divergent-layout check.

Green suite = the POC's clean, anti-fragmented, non-destructive writer is proven.
