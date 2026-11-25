# Adaptive exact-decimal binding — from the #2182 oracle to #438 physical execution (v1)

**Status:** PROPOSED / DOCS ONLY (2026-11-25). Research plan; nothing built by this file.
**Predecessor:** PR #2182 (`crates/exact-decimal-probe`, `.claude/knowledge/exact-decimal-reconciliation.md`). It establishes the mathematics on i32 and stays the semantic oracle. This plan does not widen its scope.
**Destination:** ndarray #438, adaptive P×M width, slice and seal, bound through the ISA register contract (D-DRISA) and the lane socket (D-LANESOCK).

## 0. What #2182 proved, and what it did not

| Capability | State after #2182 | Evidence |
|---|---|---|
| Adaptive exponent per slice (coarsest exact, positive exponents included) | **Implemented**, as mathematics | `seal` / `SealedSlice::exp`; `sealing_removes_redundant_powers_of_ten_before_the_width` |
| Redundant powers of ten removed before choosing a width | **Implemented**, as mathematics | `250.00, 100, 1000` → `25, 10, 100` at exponent +1 (PM9); PM25 at fixed scale 2 |
| Smallest #438 width per slice | **Implemented as an annotation** | `SealedSlice::width: Option<PmWidth>` names what *would* suffice |
| Aggregates combined across exponents (N, S, Q, exact variance) | **Implemented** | `combine`; `slice_statistics_combine_without_rescaling_the_values` |
| Physical adaptive magnitude (U8/U16/U24/U32 × P) | **Missing** | `SealedSlice` owns `Vec<i32>` + `Vec<bool>`; storage stays i32 |
| Width / exponent bookmarks bound through the ISA register | **Missing** | `LaneBinding` holds no width or scale; `EpiLaneSpec.width` is `Bits8`/`Bits9`, compile-time (`lane_codec.rs:71-76`) |
| Carry-driven promotion, demotion, resealing | **Missing** | — |
| Adaptive SIMD execution (dispatch once per sealed slice) | **Missing** | `LaneRef` is `I32`/`U32`/`U64`/`Strided` (`mask-risc ir.rs:16-36`); no U8/U16/U24 operand |

The 64 % payload saving for the example above (PM9 against PM25: 9 against 25 bits per value) is **theoretical**: it excludes metadata and polarity layout, and nothing stores it physically yet.

## 0a. Scope: every deterministic number at ingestion, not only currency (DECISION, operator, 2026-11-25)

The objective is to avoid unnecessary numerical width and computation across arbitrary incoming data, not exact currency or Excel. The sources include:
- logs and telemetry;
- PSCustomObject-like records and JSON;
- VRL events;
- Microsoft Graph reports.

Many of their values are deterministic quantities with a known decimal resolution: durations, CPU percentages, temperatures, costs, counters, measured sizes.

- **The incoming serialization type does not dictate the resident execution type.** A decimal point in a JSON or log payload is lexical text. It is not a reason to store `f64`.
- **The canonical numerical interpretation for an eligible deterministic value is exact `coefficient × 10^exponent`.** The physical width is adaptive per sealed slice (#438).
- **Schema on write establishes the semantic numerical contract and resolves the smallest lawful physical representation** at ingestion and binding time, not per consumer.
- **`f64` stays an explicitly admitted numerical method** where floating-point semantics are actually required. Examples are a source that produces binary floats, exponent notation, or values past exact range. It is never the automatic representation of a decimal.
- **A floating-point source is never quantized silently.** Source semantics and provenance are preserved (`Provenance` in #2182 is the oracle's form).
- **No consumer-specific numerical types and no new runtime.** The existing intake owner, ClassView, ISA register contract and ndarray methods carry it.

**Where it connects (current state, read at source):**

| Concern | Today | What this plan adds |
|---|---|---|
| Schema on write (intake detection) | `detect_column` recognizes bool, INT4 / unsigned INT8 and timestamp. **Decimal text such as `"1.5"` falls through to a text column** (`lance-graph-intake/src/detect.rs:157-197`), so it becomes a code-table string that nothing can compute on. | D-AED-7: detection proposes an exact decimal reading (minimal exponent, range, provenance) beside text, under the existing config-wins / verdict rule. |
| Dynamic numerical field binding | A field is a ClassView `FieldRef` (IRI, label); the binding holds no width or exponent. | The field carries the semantic numerical contract (decimal resolution, or `f64` admitted). The sealed slice carries the physical width (D-AED-3). |
| PSCustomObject-like virtual tuple bags | `Rows` over borrowed lanes (#2177/#2178) reads `Datum::{I32, U32}`, positionally. | A decimal field reads as a coefficient plus the descriptor's exponent. It is formatted late, with no `f64` variant in `Datum`. |
| Log ingestion and VRL | VRL compares `i32` and text codes. A decimal literal such as `1.5` is refused ("non-i32 comparison"). Vector JSON ingress is queued under D-VRL-RECIPE-0. | D-AED-8: `.duration_ms > 1.5` resolves at prepare time to a coefficient threshold under the field's exponent (#2182's `threshold`, checked against its exact oracle). |
| OpenObserve / Grafana telemetry | Aggregations over telemetry would decode values first. | Count, sum, mean and variance as per-slice N/S/Q folds aligned across slices. The result is formatted once. |
| Statistical folds | `GroupPowerSumsI32`; aggregate alignment proven in #2182. | D-AED-1 (no key lane), D-AED-5 (per-width kernels). |
| Late materialization | Positional `Rows`, positions resolved by the owner. | Coefficients are decoded or formatted only for demanded cells. |

## 1. The target shape (HYPOTHESIS, to be tested by the phases below)

- **Exponent** decides the smallest exact coefficient for a slice. **Magnitude width** decides how that coefficient is stored. They are independent. Presentation precision stays a third, formatting-only parameter.
- **A slice boundary is the bookmark.** A sealed slice is a contiguous identity range with one exponent and one width. The boundary is where either one changes. Identity order is never permuted to group equal widths.
- **The descriptor is witnessed, not owned.** The destination is not a `SealedSlice` that owns a second `Vec` population. It is a descriptor (range, exponent, width, seal) over the canonical owner's physical slices, lent with the lanes by `borrow_all` and checked against the frame like every other source fact. No competing schema store, register socket or numerical runtime.
- **Dispatch is per slice.** A kernel is chosen once per sealed slice, by (width, exponent). Per-value type tags are not admitted.
- **Aggregates align, values do not.** Cross-slice N/S/Q combine by aligning the aggregates (as `combine` does), one alignment per slice.

## 2. Phases

Each phase has a gate. A phase may stop at its gate with a negative result, and that is a valid outcome.

### D-AED-0 — this plan and the oracle boundary
Deliverable: this file, plus board rows. Gate: #2182 stays as is and is the oracle every later phase is checked against, by value equality on the same inputs. **Shipped with this file.**

### D-AED-1 — slice boundaries instead of the slice-id lane
**Question.** `slice_sums` builds an extra U32 slice-id lane so it can use `GroupPowerSumsI32`. Can contiguous boundaries replace that lane?

**What exists (read, not run).**
- ndarray has ungrouped `masked_sum_i32` (`simd_masking_ops.rs:946`), which gives S; `Count` gives N. mask-risc exposes both, and both run per slice over a zero-copy sub-slice of the lanes (`&lane[lo..hi]`), so no lane is materialized.
- Σx² exists only grouped: `masked_group_power_sums_i32` and its `_via` / `_pair` variants (`:2376-2460`). All of them need a key lane.

**Candidates.**
- (a) N and S per slice through the existing terminals over sub-slices; Q through a constant-key lane allocated once and reused across slices.
- (b) A new ungrouped `masked_power_sums_i32` kernel in ndarray plus a terminal in mask-risc. The terminal needs an execution-level insufficiency proof first, per isa-anti-lasagne-warden.

**Measure.**
- ns per 64 lanes for the grouped pass with the key lane, against the per-slice ungrouped pass;
- bytes allocated for each.

**Gate.** Values equal the #2182 oracle. Candidate (b) is admitted only if (a) is measured and found insufficient.

### D-AED-2 — physical width: one sealed slice stored at its #438 width
**Question.** Store a sealed slice's coefficients as P×M at the chosen width — U8 (PM9) first, then U16 (PM17) — inside the canonical owner, with polarity as the bitplane, and read it back exactly.
- Reuse codec9 / Abakus for PM9: `add_pm`, `sub_pm`, `neg_pm`, `touched` (ndarray `hpc::abakus::pm`).
- PM17 and wider have no library arithmetic. Any wider kernel is a #438 deliverable, not a probe convenience.

**Owner seam.** Where the physical slice lives, and how `borrow_all` lends it:
- Today `LaneRef` has no U8 or U16 operand.
- The register contract can express the pairing, but no source lends it.
- Read D-DRISA-0/1 and D-LANESOCK before proposing any `LaneRef` variant. A new variant is a LAYOUT change and needs the v3-envelope-auditor.

**Measure.**
- Bytes per value including metadata (descriptor, polarity, presence), for PM9 / PM17 / i32 on the same slices.
- The #2182 fixture shape (half integers, half scale 1) and synthetic worst cases.

**Gate.**
- Round trip equals the oracle, with P×M semantics intact: −0 is untouched, +0 is established, and presence is P ∨ M ≠ 0.
- Identity order is preserved.

### D-AED-3 — width and exponent as witnessed bookmarks
**Question.** Bind (range, exponent, width) as a source fact witnessed with the frame, so that a reader cannot use a slice under a stale descriptor.
- #2182 showed the requirement: a prepared filter is refused under a changed exponent.
- Here the witness moves from the probe's `FieldExps` stand-in to the owner's descriptor.

**Candidates (reuse first):**
- the existing frame revision (`FrameRevisions.source` moves on every data write);
- `SoaEnvelope` / `ColumnDescriptor` / `LaneAddr`;
- intake's column shape.
- Not a new descriptor system.

**Gate.**
- A run under a changed width or exponent is refused, even when class, field identity and row count are unchanged. This mirrors #2182 requirement 1, now on the real witness.
- Disable-verify the refusal.

### D-AED-4 — promotion, finer resolution, demotion
Three independent events. Each has the same two strategies: rewrite the slice in place, or keep the sealed slice and start a new one.

| Event | Rewrite in place (reseal) | New slice (append a bookmark) |
|---|---|---|
| **Carry / magnitude overflow** (a sum or a new value leaves the width) | Re-encode the slice at the next width | Keep the old slice and open a new slice at the wider width |
| **Finer resolution** (a new value needs a smaller exponent, independent of width) | Multiply the slice's coefficients by 10^k; this may also promote width | Keep the old slice and open a new slice at the finer exponent; aggregates align per slice |
| **Demotion** (a slice's values all fit narrower or coarser) | Re-encode on compaction only | — (demotion is a compaction-time decision) |

**HYPOTHESIS.** Appending a new slice keeps sealed slices immutable and readers' frames valid, and moves all reconciliation to per-slice aggregate alignment. Rewriting in place keeps slices fewer and larger.

**Measure.**
- slice count growth;
- bytes;
- aggregate alignment cost against a per-value rescale baseline.

Use carry detection from `PmSum::carry` (Abakus) for the overflow trigger.

**Gate.**
- Exactness against the oracle across every event.
- No rewrite of a slice a live frame still witnesses.

### D-AED-5 — dispatch once per sealed slice, and the measurement that decides
**Question.** Run count, sum, threshold filters and N/S/Q per slice at the slice's own width and exponent, dispatching the kernel once per slice. Combine across slices by aggregate alignment.

**Measure** (the only phase that can claim speed):
- ns per 64 lanes for each kernel at each width, against the i32 path, on the same data;
- dense and sparse selections (#439 found P×M wins dense and loses sparse);
- the dispatch overhead per slice;
- the end-to-end bytes moved.

Report in ns and µs only, per the workspace rule.

**Gate.**
- A claimed gain needs a measurement with an asserted answer (query-stage-profiler).
- An unmeasured or noisy result stays OPEN.

### D-AED-7 — schema-on-write detection of exact decimals (intake)

**Question.** Extend intake's detection with an exact decimal reading. Today a column of decimal text detects as text.

**The reading.**
- **Eligibility:** every present value parses exactly (`canonical` in #2182). That means no exponent notation, and no more significant digits than the source could have produced deterministically.
- **Recorded per column:** the minimal common exponent and the coefficient range.
- **Provenance:** `Lexical`.
- **Config wins**, as for every other detected shape. A column whose values look like binary-float artefacts (for example `0.30000000000000004`) is not silently read as a decimal. It stays text, or becomes an explicitly declared `f64` method, with the verdict recorded.

**Gate.**
- Detection agrees with a declared config on synthetic logs, JSON and Graph-report-shaped samples.
- A float-artefact column is refused as decimal, and that refusal is disable-verified.
- No value is rounded.

### D-AED-8 — VRL and log ingestion over decimal fields

**Question.** A VRL recipe filters and aggregates decimal fields.
- Literals are resolved when the program is prepared, under the field's exponent, as #2182's `threshold` does, and are refused when the exponent changes (#2182 requirement 1).
- Values arrive through intake, or later through Vector JSON ingress (D-VRL-RECIPE-0, queued).

**Gate.** Counts and sums equal the #2182 oracle on the same events. No `f64` enters the recipe path.

### D-AED-6 — real coefficient ranges
**Question.** Measure actual coefficient ranges and the share of redundant zeros. #2182's fixture scan is synthetic. Sources to measure:
- real workbooks;
- real logs and telemetry samples, such as duration, CPU %, temperature, byte counters and Graph report fields.

Real workbooks reach report-rs only at runtime, through the operator. Telemetry samples need a source the operator names.

**Gate.** No width default is chosen from synthetic data.

## 3. Out of scope

- **Production decimal admission:** D-LANESOCK keeps it closed, and this plan does not open it. D-AED-7 and D-AED-8 are probes against the oracle.
- **The report-rs XVERWEIS migration:** it continues independently, on integer and codebook fields.
- **Division, arbitrary precision and decimal tensors.**
- **Changes to #2182 beyond oracle maintenance.**

## 4. Open

1. **Lane variant or strided view.** Whether a PM9/PM17 slice is lent as a new `LaneRef` variant or as a `Strided` view over an owner layout. This is a layout decision, not taken here.
2. **Where the descriptor lives.** Intake column shape, the SoA envelope, or the DR tile (`TILE_WIDTH = 256` in intake is one existing boundary). Undecided.
3. **Reseal or append.** Which strategy wins depends on write patterns; D-AED-4 measures both.
