# F-ORD-REAL — the arrival-order publication-identity falsifier (Phase 1)

> Deliverable 2 of the LOTUS SEAL research charter (operator, 2026-08-18).
> Companion audit: `LOTUS-FRONTIER-AUDIT.md` §1 (the verified chain). The
> tests this document pre-registers land in the SAME PR, in
> `crates/lance-graph-supervisor/src/cycle_driver.rs`'s test module — the
> falsifier lands BEFORE any fix, per the charter's preference.

## 1. The mechanism, in four verified steps

1. `BatchWriter::cast` mints `CastId(next_id++)` at ARRIVAL
   (`lance-graph-planner/src/batch_writer.rs:132-138`).
2. `collect_casts` derives `stream_position = position_base + cast.0`
   (`lance-graph-supervisor/src/cycle_driver.rs:385`).
3. `DetachedCycleBatch::freeze` stable-sorts by `stream_position`
   (`lance-graph-planner/src/persist_sink.rs:378`) — storage ORDER is safe.
4. `content_hash` folds the `stream_position` VALUES into `batch_hash`
   (`persist_sink.rs:414`, `eat(&s.stream_position.to_le_bytes())`) — so the
   publication identity (the durable idempotency key, the `(cycle, batch_hash)`
   reconciliation key, the frame the store persists) depends on which producer
   happened to finish first.

**The smoking gun is a doc-contradiction:** `DetachedCycleBatch`'s own field
doc (`persist_sink.rs:359-362`) claims *"Identical completed sets yield
identical hashes regardless of worker completion order (the freeze
canonicalizes first)."* The freeze canonicalizes the ORDER; it does not
canonicalize the COORDINATES the hash then eats. The sentence is false today.

**What is NOT implicated (already answered upstream):**
- The restart leg — `position_base` is a durable cursor
  (`cycle_driver.rs:340-347`), pinned by
  `restart_stable_stream_positions_survive_writer_reconstruction` (`:1460`).
- The row-keyed image — `image: row → last payload` with identity-derived
  `row = row_of(owner)` is arrival-independent when rows are per-owner. The
  defect pin asserts this GREEN leg explicitly, so the test documents which
  leg holds, not just which breaks.

## 2. Test design — perturb the process that creates the key

Operator sharpening (A2, 2026-08-18): *"Do not test permutation after the
order key already exists. Perturb the process that creates the key."*

Applied literally: the tests permute **the order of `cast()` calls** — the
upstream event that mints `CastId` — and then run the REAL
`collect_casts → freeze` chain. Two designs were rejected as vacuous or weak:

- **Permuting `Vec<SweepSlot>` before `freeze`** — vacuous: `freeze` sorts by
  the already-minted key, so any post-mint permutation passes regardless of
  the defect. This is exactly the trap the sharpening names.
- **Real threads + jitter** — a nondeterministic way to produce the same
  perturbation the deterministic form applies directly; it can pass by
  scheduler luck. The deterministic permuted-arrival form is the stronger
  falsifier and is what lands. (A threaded stress variant may join the Phase 5
  benchmark plan; it adds realism, not evidence.)

Construction: 64 owners; payloads a pure function of owner identity (so the
SEMANTIC completed set is identical by construction across runs); no kanban
moves (the defect is fully expressed by positions; moves would only widen the
API surface under test); `row_of = identity`. Run once in forward arrival
order, once reversed; freeze both against the same `CycleFrame`.

## 3. What lands, and what each half means

**(a) The GREEN defect pin** —
`f_ord_real_defect_pin_arrival_order_leaks_into_batch_hash`:

- anti-vacuity guard: the same owner carries a DIFFERENT `stream_position` in
  the two runs — proof the perturbation reached the key mint (without this,
  equal hashes would be vacuously assertable);
- asserts the semantic sets equal AND `image` equal (the leg that holds);
- asserts `batch_hash` DIFFERS (the defect, pinned two-sided). **When a fix
  lands this assert fails**, forcing a deliberate re-pin: delete the pin and
  un-ignore (b). A silent fix cannot slip past it.

**(b) The RED falsifier** —
`f_ord_real_publication_identity_is_arrival_order_independent`, `#[ignore]`d
with a message citing this document. It asserts the DESIRED property (equal
semantic set ⇒ equal `batch_hash`) and fails today — run
`cargo test -p lance-graph-supervisor -- --ignored f_ord_real` to see it red.
The ignore attribute is what lets a red-by-design falsifier land on a green
main, per the charter's land-the-falsifier-first preference.

## 4. Green criterion for any fix — mechanism deliberately unprescribed

A fix is green when: identical completed semantic sets yield identical
`batch_hash` (test (b) un-ignored and green), with the restart falsifier, the
`(cycle, batch_hash)` reconciliation path, and `recover_and_apply`'s
per-owner watermark contract (`stream_position` monotonic per owner ACROSS
cycles — `persist_sink.rs:176-182`) all still green.

Candidate mechanisms, listed NOT chosen (Phase 2+ design, council-gated):

1. hash the canonical INDEX (position within the sorted batch) instead of the
   `stream_position` value;
2. hash owner-keyed content only (owner, row, move, payload) and drop the
   coordinate from the identity entirely;
3. identity-derived positions (the lotus placement: slot = f(identity), so
   arrival never mints a coordinate at all) — the only candidate that ALSO
   removes the seal's repair sort; see the audit §6.3;
4. carry the TRUE semantic order key through the cast (linear/textual
   workloads only) — `SweepSlot`'s contract already says the key is
   *"the caller's EXISTING canonical (textual/stream) order key… NOT a new
   coordinate"* (`persist_sink.rs:168-183`); for a genuinely linear stream,
   supplying the real witness position makes hashing it CORRECT, because the
   coordinate becomes semantic identity rather than arrival residue.

Candidates 3 and 4 are not rivals — they are the two halves of the workload
split (audit §6.5, "texts are linear; GridLake tiles would make it
contradictory"): linear-stream classes keep a supplied semantic key, tile
classes get derived placement, resolved per class like every other sanctioned
reading. Each candidate has a distinct blast radius on the watermark contract
— that analysis is the Phase 2 design document's job, not this falsifier's.
The falsifier is mechanism-agnostic by construction: it constrains WHAT must
become true, never HOW — with one nuance: under candidate 4 the test's
"identical semantic set" premise must key on the SUPPLIED semantic positions
(two arrival orders of the same text carry the same textual positions, so the
falsifier's assertion is unchanged in substance).
