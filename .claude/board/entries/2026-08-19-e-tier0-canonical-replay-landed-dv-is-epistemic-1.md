## 2026-08-19 — E-TIER0-CANONICAL-REPLAY-LANDED-DV-IS-EPISTEMIC-1

**Status:** FINDING + RULING `[operator T0.3 amendment]`, all landed
red-then-green.

**T0.1 F-PHYS-ORDER (landed, fixed).** Proven RED on the old code two ways,
compaction-free: a semantically legitimate one-lap cyclic chain sealed with
scrambled physical row order made `recover_and_apply` FAIL outright
(`StalePhase` — legitimate sealed content unrecoverable purely because of
physical placement), and the watermark was LAST-seen not MAX-seen (physical
order [2,0,1] → watermark 1 with position 2 already consumed → the next
pass re-admits it: a silent double-apply exposure SHARPER than D2's L2
table). Fix: `recover_and_apply` sorts the owner's landings by their own
durable canonical coordinates `(cycle, stream_position)` before walking —
restart-path only, hot path untouched; the write-side ordering remains a
layout courtesy, no longer the correctness carrier. `scan_sealed` stays a
raw physical read (the canonical projection is at REPLAY — visibility is a
projection, per §0). Both falsifiers disable-verified (sort removed → red).

**T0.2 F-QREF-STRICT (landed, pinned two-sided).** The existing test pinned
the default's VALUES; the new falsifier pins the CONSEQUENCE: under
`QueryReference::default()` — Strict in name — a future row classifies
`Contemporary` (the u64::MAX sentinel is an UNBOUNDED observer, zero
hindsight protection; documented on `default()` with the migration pointer),
while the SAME row under `at(head, 0)` is `Anachronistic` and in-horizon
rows stay admitted (the guard discriminates). Disable-verified (classify's
rejection branch removed → red). Sentinel-removal-at-construction stays the
registered Tier-1 follow-on; silently changing `default()` would be the
I-LEGACY trap.

**T0.3 F-AWARENESS-LAG (operator amendment, landed).** The ruling, recorded:
ΔV = W_write − A_last is the primary cognitive quantity — an EPISTEMIC
metric (durable-history distance: "how far the world moved while I was
thinking"), NOT wall-clock (Δt is an economics metric). STORE COORDINATES,
DERIVE DISTANCE; no awareness_delay_ms / elapsed_time / persisted ΔV.
ΔV does not replace HLC (one version history vs cross-writer causality).
No hard replay threshold from ΔV yet — measure its distribution first
(hot-wavefront mode legitimately lands ΔV > 1; dense mode pins ΔV = 1).
**Audit result (the amendment's mandated first question): BOTH coordinates
are ALREADY durably reconstructible — no schema field added.** A_last =
`FrameMeta.base_version` (per cycle, restart-readable via `timeline()`,
hash-bound into the batch identity). W_write derives with no field: the
FENCE invariant forces `base_version(N+1) == publication_version(N)` in the
one-writer chain, so the durable timeline reconstructs every interior
cycle's W_write and the head covers the newest; store-side the Lance sink
independently has one DatasetVersion per commit + per-row
`created_at_version` (RP-SEAL A1). Falsifier proves restart recovery of
both exact coordinates + order-independent ΔV derivation by canonical
keying. `QueryReference::ref_version` documented as the reader-side A_last.

**Gate:** planner 357/357 + fmt + CI-form clippy clean in-crate (the 5
residual warnings are pre-existing jc/shader-driver noise, on main too).
Next per the ruled order: X-C2-1 injection harness, then E2 re-verify +
perf economics.

