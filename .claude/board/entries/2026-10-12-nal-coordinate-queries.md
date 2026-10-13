# 2026-10-12 — NAL query operations as coordinate ranges (D-NALP-1/2)

**Status:** In PR (#1525). Plan: `.claude/plans/nal-polyfill-v1.md` (status note before § 5).

**Decision (operator, 2026-10-12):** the NodeGuid is always the address and the
distinguished name; a selection is the 16-byte key minus an `n`-byte aperture.
A belief's key carries only its name (classid, S, copula, P, id); its truth
value lives in value lanes in the sealed order.

**Finding (MEASURED):** a statement, a subject or a whole class is an aperture
of 10, 6 or 4 cared bytes and one witnessed range
(`SealedFacetLane::bound_aperture`). The typed `nal::` operations bind once
(R2IL procedure → kept mask-risc program) and each call runs only that program
over the range as its execution extent. At 1,048,576 beliefs: range counts
19–20 ns, a confidence threshold over a 245-row subject 127 ns, the maximum
confidence 248 ns, against 495–559 µs for the row loop
(`crates/lance-graph-nal/examples/nal_cost.rs`, AVX-512, release).

**Tests:** `crates/lance-graph-nal/tests/w2.rs`, 7 row-oracle tests over the
unsorted input; 7 disable runs red (statement aperture too narrow, aperture
one byte short, threshold off by one, threshold without its extent, choice
reading `f`, truth lanes misaligned with keys, witness not validated).

**OPEN:** R2IL `RANGE` takes `u8` immediates and cannot name a range past row
255 (request to `ogar-r2il`); no argmax fold (which belief holds the maximum);
the rest of D-NALP-1 (ClassView `WideFieldMask` projection, binder mask) and
W3+ are queued.
