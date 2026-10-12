# 2026-10-12 — NAL query operations as coordinate ranges (D-NALP-1/2)

**Status:** In PR. Plan: `.claude/plans/nal-polyfill-v1.md` (status note before § 5).

**Finding (MEASURED):** with beliefs sealed as `FacetCascade` keys
`[classid, S, copula, P, 255 − c, 255 − f]`, a belief question, a confidence
threshold within a statement, subject support and NAL choice are witnessed
coordinate ranges. The typed `nal::` operations bind once (R2IL procedure →
kept mask-risc program) and each call runs only that program over the range.
At 1,048,576 beliefs: range counts 24 ns, a confidence threshold across one
subject 139 ns, choice 265 ns, against 554–614 µs for the row loop
(`crates/lance-graph-nal/examples/nal_cost.rs`, AVX-512, release). Passing the
range as the execution extent, not as an ANDed mask, took the cross-statement
threshold from 195 µs to 139 ns.

**Tests:** `crates/lance-graph-nal/tests/w2.rs`, 8 row-oracle tests over the
unsorted input; 7 disable runs red (threshold ignored / off by one, choice
reads the last row, cross-statement threshold off by one and without its
extent, witness not validated, subject bound truncated).

**OPEN:** R2IL `RANGE` takes `u8` immediates and cannot name a range past row
255 (request to `ogar-r2il`). The rest of D-NALP-1 (ClassView `WideFieldMask`
projection, binder mask) and W3+ are queued.
