# 2026-11-24 — LG-COG-001 reuse census: thought-git receipts exist; the replay evidence rule holds by construction

**Status:** MEASURED (thought-git-probe 58/58 at 855d051) plus CENSUS (docs only).

`cd crates/thought-git-probe && cargo test` (debug=0) passes 58 tests in 6 suites. `receipt.rs` implements D-TGB-4 as lab code, so the statement in D-LFA that it "does not exist" is corrected. `replay` always returns `independent_roots: 0`, and `tests/signals.rs::replayed_material_cannot_raise_evidence` pins that. NAL `Revision` already refuses overlapping or unstamped evidence. The missing piece is seam S1: mapping receipt roots onto `Evidence::stamp`.

Matrix and seams: `.claude/plans/lg-cog-001-reuse-census-v1.md`.

OPEN:
- S1 receipt → NAL stamp adapter;
- S2 typed rollout witness;
- EXO-148 `DurableRef`;
- EXO-300 lance-context admission;
- the mock-teacher harness before any API spend ($5 cap).
