## 2026-08-18 — E-FORD-REAL-PUBLICATION-IDENTITY-IS-ARRIVAL-DEPENDENT-1

**Status:** FINDING (VERIFIED at file:line) + pre-registered falsifier landed
(LOTUS research charter Phase 0/1; docs: `docs/lotus/LOTUS-FRONTIER-AUDIT.md`
+ `docs/lotus/F-ORD-REAL-FALSIFIER.md`).

**The defect chain.** `BatchWriter::cast` mints `CastId` at ARRIVAL
(`batch_writer.rs:132-138`) → `collect_casts` derives `stream_position =
position_base + cast.0` (`cycle_driver.rs:385`) → `freeze` stable-sorts by it
(`persist_sink.rs:378` — storage ORDER is safe) → **`content_hash` folds the
`stream_position` VALUES into `batch_hash`** (`persist_sink.rs:414`). The
publication identity — the durable `(cycle, batch_hash)` idempotency key —
therefore depends on producer completion order. The struct's own doc
(`persist_sink.rs:359-362`, "Identical completed sets yield identical hashes
regardless of worker completion order") is FALSE today. The row-keyed `image`
leg HOLDS (identity-derived `row_of(owner)`); the restart leg was already
answered (durable `position_base`, falsifier at `cycle_driver.rs:1460`).

**Landed with this entry:** the GREEN two-sided defect pin
(`f_ord_real_defect_pin_arrival_order_leaks_into_batch_hash` — anti-vacuity:
perturbation provably reaches the key mint; semantic set + image asserted
arrival-INDEPENDENT; hash asserted arrival-DEPENDENT) and the `#[ignore]`d
RED falsifier (`f_ord_real_publication_identity_is_arrival_order_independent`
— red under `--ignored`, hashes 2604999916736672513 vs 4858955943411201665 on
the real chain). Fix mechanism deliberately unprescribed (four candidates in
the falsifier doc, per-workload per the linear-vs-tile split below).

**Three adjacent verified tensions recorded in the audit, not fixed here:**
(1) `SweepSlot::stream_position`'s contract says "the caller's EXISTING
canonical (textual/stream) order key… NOT a new coordinate"
(`persist_sink.rs:168-183`) while `collect_casts` MINTS it — for linear/text
workloads the fix is carrying the true semantic position (then hashing it is
CORRECT); for tile workloads identity-derived placement; per-class resolution,
never one global rule (operator: "texts are organized linearly; GridLake
tiles would make it contradictory"). (2) The descriptor doctrine
(`batch_writer.rs:30-39`, "P is a DESCRIPTOR — never owned delta bytes") is
unrealized: the only production-shaped instantiation is `BatchWriter<Vec<u8>>`
→ batch resident up to 3× at seal (staging + image clone + retry clone).
(3) The seal is O(batch bytes) ×3 passes (repair sort + clone + byte-wise
FNV) — `order_cycle_stably`'s own doc already concedes the constructive
scatter alternative (`persist_sink.rs:126-129`).

**Session BLOCKER, flagged loud:** the lance crate source is ABSENT from this
sandbox (zero `lance*` under the registry; family pinned `=9.0.0`) — the
prepared-artifact capability audit (fragments/Transaction/cleanup/blob) is
UNVERIFIABLE from source this session; Phase 6 of the charter is gated on a
`cargo fetch` or operator-sanctioned source consult. Also owed: the
`temporal.rs:396`/`:410` doc-drift (cites the RETIRED `DurableWitness` /
`DurableCoordinate` as production implementor) — correction folded into the
next phase PR.

