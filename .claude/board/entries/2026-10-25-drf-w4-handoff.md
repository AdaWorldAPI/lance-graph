# D-DRF-W4 — W4–W7: flushed row images handed to the real CycleWriter (lab)

**Status:** TEST-PINNED (lab, `cargo test --features handoff`; no CI job runs this crate).
**Code:** `crates/drf-focus-probe/src/handoff.rs` (`RowImage::into_slot`, `into_slots`).
**Tests:** `tests/integration/handoff.rs`, driving `lance-graph-supervisor::cycle_writer`
over a reconciliation-first fake WAL. Plan: `.claude/plans/d-dr64-focus-cache-falsifiers-v1.md` §4.

- **W4 PASS:** 64 writes over 37 rows → 37 owned casts. The focus and the borrow end at
  flush; the source is then changed while the WAL holds the commit. Nothing is durable until
  the WAL releases (`durable().through == None`, ticket unresolved); what lands is the flushed
  snapshot. Stream positions continue across cycles (37 → 74).
- **W2 through the writer PASS:** two facets of one row land together. Contrast arm: one cast
  per raw patch loses the first facet, because the writer's freeze keeps the later payload
  whole. The per-row coalesce in `flush` is what makes the handoff safe.
- **W6 PASS:** lost acknowledgement → `ResubmitFrozen`, handed-back casts equal the frozen
  batch byte for byte, re-submit on a fresh writer → `Reconciled`, one append. Regenerating
  cycle 1 from the changed source → `HashConflict` → `Escalate`. A full queue hands the casts
  back unchanged; a poisoned writer refuses new submissions.
- **W7 PASS:** a cycle with no dirty row gives zero casts → `NoChange`, the WAL is never
  called, the head stays; the next dirty cycle publishes exactly one version.
- **Disables, each red:** one image per patch (W1–W3 tests and the through-writer W2 test);
  `into_slots` not advancing the stream position (W4).
- **Change:** `RowImage::row` is now `Vec<u8>` (was `[u8; 512]`) so the move into the
  cast's payload is not a second copy.

OPEN: W5 (descriptor-only cast pointing into live SoA, lease held until capture) needs a
descriptor payload the writer does not have; W8 (eviction under memory pressure with live
readers) is not run. The fake WAL stands in for Lance; a real `WalSink` run is not done.
