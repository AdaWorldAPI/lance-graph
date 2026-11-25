# Exact decimals over the existing numerical substrate

READ BY: anyone touching exact decimals, currency or percentages; report-rs value ingress (EXO-174); `lance-graph-vrl` numeric filters; DT64 lane socket (D-LANESOCK); the register contract (D-DRISA); adaptive P×M width (ndarray #438) and P×M statistics (ndarray #439).

STATUS: research plus an isolated semantic probe (`crates/exact-decimal-probe`). Not a production admission: D-LANESOCK keeps exact decimal **closed**, and nothing here changes that.

## Framing (DECISION, operator, 2026-11-25)

- **An exact decimal is a mathematical reading of a coefficient, not a new physical datatype:** `value = coefficient × 10^exponent`, with `exponent = −scale`.
- **Three independent parameters.**
  - *Coefficient width* is numerical capacity: the ndarray #438 widths PM9 / PM17 / PM25 / PM33, or i32 today.
  - *Decimal scale* is numerical resolution, homogeneous per field or per sealed slice.
  - *Presentation precision* is formatting. 250.00 can be coefficient 250 at scale 0, displayed with two fraction digits.
- **The i32 probe is a semantic reference implementation, not a storage decision.** The destination stays adaptive numerical binding through the ISA register contract (D-DRISA): width and scale witnessed with the binding and seal, the source owner lending the physical layout. The probe pins the arithmetic that destination must reproduce. It must not turn into the permanent decimal storage format.
- **Objectives.** The immediate objective is exactness. The next is adaptive numerical compression and cheaper execution. The combination of statistical folding and adaptive decimal scales (§ Probe results, slice statistics) is a research hypothesis, not a performance result.
- **Not admitted:**
  - `f64` in `Datum`;
  - a report-rs decimal engine;
  - a second decoded population, or a competing validity bitmap;
  - string arithmetic;
  - codebook ordinals used as coefficients;
  - a competing schema store, register socket or numerical runtime.

## Next: physical adaptive execution

#2182 proves the mathematics on i32. Physical adaptive width, witnessed width/exponent bookmarks, promotion, per-slice dispatch, and the ingestion scope (every deterministic number in logs, telemetry, JSON, VRL and Graph reports, not only currency) are planned in `.claude/plans/adaptive-exact-decimal-binding-v1.md` (D-AED-0..8). The probe stays the oracle.

## Cross-references

- **ndarray #438:** adaptive P×M width, slice and seal (`.claude/knowledge/adaptive-pm-width-slice-seal-research-prompt.md` in ndarray).
- **ndarray #439:** P×M statistics, N/S/Q/A folds and comparisons (`pm-stats-probe.md`; code in `examples/pm_stats`).
- **D-DRISA:** the register contract (`.claude/board/entries/2026-11-04-d-drisa-0-remote-register-contract.md`, `…-d-drisa-1-register-proof.md`).
- **D-LANESOCK:** the lane socket and its "Still closed … exact decimal" gate (`.claude/plans/d-lanesock-dr-dt64-lane-socket-v1.md`).
- **LG #2160:** `borrow_all_at`, frame-qualified positional reads.
- **LG #2175:** Codec9 presence reconciliation (polarity is not validity).
- **LG #2177 / #2178:** `lance-graph-vrl` (recipes, code lanes, `rows_at`). The text literal resolved under a frame (#2181) is the pattern the decimal literal resolver mirrors.
- **EXO-174** (report-rs): value ingress and replacement of the quarantined value layer.

## Probe results (`crates/exact-decimal-probe`, 11 tests, all passing)

Run with `cargo test --manifest-path crates/exact-decimal-probe/Cargo.toml`.

- **Exact lexical parse:**
  - `"1.2"` and `"1.20"` read as one number;
  - fewer fraction digits are padded, extra digits are admitted only when they are zeros, and a nonzero digit past the scale is refused (never rounded);
  - the i32 bounds are exact at scale 2 (±21,474,836.47 / −21,474,836.48);
  - a coarser exponent (thousands) is admitted.
- **Literal thresholds, resolved at prepare time:** for every operator, literals on, between and far beyond the representable coefficients agree with an exact rational oracle for every coefficient tested. Some literals become `Never` or `EveryValue`, and the drop term keeps them on the field so absent values stay undecided.
- **COUNT(rows) and COUNT(established values) are counted apart:** 5 rows and 4 values on the intake fixture. The explicit `0.00` counts as a value; the blank is reported as undecided.
- **A prepared filter refuses a changed scale on the same witness** (`RecipeRefusal::ExpChanged`). The same class, field identities and row count do not help; the VRL recipe alone would have run. In production the scale must be an owner-side column property witnessed with the binding. The probe's `FieldExps` stands in for that owner descriptor and is not a schema store.
- **`f64` is an explicit policy (`F64Policy::ShortestRoundTrip`), recorded as `Provenance::F64ShortestRoundTrip`, never proof of the original decimal.** `0.1 + 0.2` is refused at scale 2.
- **Sealing removes redundant powers of ten before the width is chosen:**
  - `250.00, 100, 1000` seal at exponent 1 as `25, 10, 100`, which fits PM9;
  - the same values at the field's fixed scale 2 need PM25.
  - A slice never seals finer than its field's declared resolution.
- **Slice statistics (the research hypothesis, verified for exactness only):**
  - one mask-risc `GroupPowerSumsI32` pass, keyed by a slice-id lane and masked by presence (`presence ∨ c ≠ 0`), yields per-slice N, S, Q;
  - combining slices aligns the *aggregates* (`S·10^(e−t)`, `Q·10^(2(e−t))`), one alignment per slice that differs instead of one per value;
  - the result equals every value rescaled one by one, including exact variance `(N·Q − S², N²)`;
  - an alignment past `i128` is refused.
  - No timing was taken: this establishes exactness and the per-slice operation count, not speed.

## Workbook fixture measurements (report-rs, synthetic)

The field policy declares one decimal field, `xl:azure_spend` ("Azure Kosten"). Scanning the raw cell XML of `fixtures/workbooks/customers-synthetic-v{1,2,3}.xlsx` (24/25/25 numeric cells):

- **Lexical text is present in the file.** The XLSX stores short decimal text (`37.5`, `50`, `62.5`, …), with no `<v>` longer than 17 characters. calamine then hands report-rs an `f64` (`read.rs`). Lexical intake would have to read `<v>` itself; until then the `f64` path is a policy conversion.
- **Redundant decimal zeros:** at the minimal scale, 12 of 24/25 values are integers (scale 0) and the rest need scale 1. None needs scale 2, though a fixed currency scale would store two fraction digits for all of them.
- **Range:** the largest coefficient at scale 2 is 33,750, which fits PM17. At the minimal per-value scale the integers fit smaller widths.
- **These fixtures are synthetic.** Real customer workbooks arrive only at runtime (`REPORT_LABELS`), so actual coefficient ranges and the share of redundant zeros remain **OPEN** until a real workbook is measured. No width decision is taken from synthetic data.

## Research reconciliation (verified)

Sources were read at ndarray master `82d12946`, lance-graph `db2f1c8` and report-rs `203a7fc`. Path prefixes: `nd:` ndarray, `lg:` lance-graph, `rr:` report-rs. The orchestrator re-read the five least certain claims against source and corrected two (§7.1, §7.4).


Date: 2026-11-25. Read-only research. No code built or run (no cargo invoked).

Sources and the heads they were read at:
- ndarray master worktree `…/scratchpad/ndarray-master` at `82d12946` (merge of #439). Paths below starting `nd:` are relative to it.
- lance-graph `/home/user/lance-graph` on branch `claude/great-bohr-gwiktx` at `db2f1c8` (main plus PR #2181). Paths starting `lg:`.
- report-rs `/home/user/report-rs` at `203a7fc`. Paths starting `rr:`.

Labels: a claim with a `path:line` was read at that place. HYPOTHESIS is an untested proposal. OPEN means not settled from source.

---

## 0. Answer in one paragraph

Mathematically, yes. An exact decimal is a signed integer coefficient with a homogeneous base-10 scale. Every operation report-rs needs (equality, ordering against a literal, sum, mean and variance as exact rationals) is integer arithmetic on the coefficient plus bookkeeping on the scale. Physically, the substrate today has two executable integer carriers:
- the **i32 lane with a presence plane**, an executable path end to end (intake → DT64 → VRL → mask-risc);
- **PM9 / Abakus**, which is exact only to ±255 and has no binding into mask-risc.

PM17/25/33 are a research prompt, not code. Nothing in any of the three repos records a decimal scale. Codec9's scale is an `f32` quantization step, not a base-10 exponent. report-rs keeps decimals as text codes: `1.20` and `1.2` are different codes, and numbers stored as numbers arrive as `f64`. The smallest honest probe is therefore the **scaled coefficient on the existing i32 lane**, with the scale held as a source-owned column property witnessed through the frame. PM widths enter later, as a storage and bandwidth refinement of the same coefficient.

---

## 1. WHY — what this would remove

The duplication today, all in report-rs:

1. **A second representation of numbers as text.** `TypedValue::Decimal(String)` is documented as "Exact decimal text … Never an `f64`" (rr:src/workbook_delta/reconcile.rs:108-111). It is stored as a text-code lane: every non-bool field goes to `column(PgType::Text, LaneEncoding::TextCode)` (rr:src/workbook_delta/lanes.rs:144-145). The value text is the code-table entry (rr:src/workbook_delta/lanes.rs:75-80, 82-88).
2. **No arithmetic on that representation.** A code is not a number: VRL `Datum::U32` is "A text column's code" (lg:crates/lance-graph-vrl/src/lib.rs:1199-1203). Text order comparisons are refused "since codes have no order" (lg:crates/lance-graph-vrl/src/lib.rs, crate doc, lines 66-69). Summing or ordering a decimal column therefore needs a decode followed by arithmetic somewhere else. That would be a second, decoded population, which the operator has ruled out.
3. **Lexical equality where numeric equality is meant.** Two spellings of one number are two codes. `parse_decimal` keeps the text "as written" (rr:src/workbook_delta/reconcile.rs:233-242), so `1.20` ≠ `1.2` and `007.5` ≠ `7.5` under `EqU32`.
4. **A parallel numeric path for numbers stored as numbers.** A cell stored as a number becomes `TypedValue::Decimal(n.to_string())` from an `f64` (rr:src/workbook_delta/reconcile.rs:205-211).

A substrate-carried coefficient would replace (1) through (4) with one integer lane, one presence plane and one scale witness. Projection, equality, ordering, sum and statistics would then run on lanes that already exist (§3).

HYPOTHESIS: the same carrier would serve other consumers with fixed-scale quantities. The 1BRC probe already folds a scale-1 decimal as integer "tenths-of-a-degree" (lg:crates/onebrc-probe/src/lib.rs:100, :119-136; lg:crates/onebrc-probe/src/gen.rs:86-88). That is precedent for the pattern, not a shared contract.

---

## 2. WHERE — who owns each responsibility

| Responsibility | Owner today | Evidence | Status |
|---|---|---|---|
| Integer coefficient, P×M form, ±255 | ndarray `hpc::codec9` (representation), `hpc::abakus::pm` (arithmetic) | nd:src/hpc/abakus/mod.rs:3-5 ("codec9 owns the representation (magnitude, presence, scale, rounding…). This module owns operations"); nd:src/hpc/abakus/pm.rs:11-31 | exists, ±255 only |
| Integer coefficient, i32 two's complement + presence | contract `pg_row::LaneEncoding::Signed`; intake maps it to `FieldKind::I32` | lg:crates/lance-graph-contract/src/pg_row.rs:156-157; lg:crates/lance-graph-intake/src/bind.rs:111 | exists |
| Absent / established zero law | contract `lane_codec` ("absent … when its presence bit is 0 **and** its value is 0"), DT64 `PresenceSource` | lg:crates/lance-graph-contract/src/lane_codec.rs:55-63; lg:crates/lance-graph-dt64/src/lane.rs:122-141 | exists |
| Lane layout witness (width) | `EpiLaneSpec { width, space, values, presence, lanes }`, per class, compile-time | lg:crates/lance-graph-contract/src/lane_codec.rs:104-114; `LaneWidth` is `Bits8`/`Bits9` only (:70-76) | width only, 8/9 bits |
| Scale witness (base-10) | **none** | `EpiLaneSpec` has no scale field (lane_codec.rs:104-114; grep "scale" in that file: no hits); `LaneBinding` fields hold class, rev, route, frame, scope, selection, fields, lanes, refs, basis, laws, and no width or scale (lg:crates/lance-graph-dt64/src/lane.rs:258-280); `FrameRevisions` = source, relation_map, coordinates (lg:crates/lance-graph-dt64/src/dr.rs:360-364) | MISSING |
| Codec9 scale | ndarray `Codec9`/`Codec9Runtime`, an `f32` step, decode = `v * scale` | nd:src/hpc/codec9.rs:23-37, :187-193, :265-286 | exists, wrong kind (see §3 note) |
| Comparison, reduction (executable) | mask-risc `Pred`/`Terminal` over `LaneRef::{I32,U32,U64,Strided}` | lg:crates/lance-graph-mask-risc/src/ir.rs:16-36, :127-206, :288-497 | i32 only for ordered/sum |
| Recipe front (filter/sum/project) | `lance-graph-vrl` | lg:crates/lance-graph-vrl/src/lib.rs:13-25 | i32 compare + sum |
| Admission of SQL `numeric` | contract `pg_row` refuses it | `PgTypeRefusal::NotAdmitted` "(numeric, floating point, …)" (lg:crates/lance-graph-contract/src/pg_row.rs:55-58); `PgType` has no numeric variant (:32-51) | refused by design |
| Intake parsing of the cell | report-rs `reconcile::convert` / `parse_decimal` | rr:src/workbook_delta/reconcile.rs:179-221, :233-242 | text only |
| Formatting | report-rs (`value_text`), and the code table | rr:src/workbook_delta/lanes.rs:75-80 | text passthrough |
| Planning gate | D-LANESOCK lists "exact decimal" under **Still closed** | lg:.claude/plans/d-lanesock-dr-dt64-lane-socket-v1.md:163-165 | not admitted |

HYPOTHESIS (ownership): the scale belongs where the source declares the column, which is intake `ColumnConfig` / `PgColumn`, the layer that already declares `ty`, `encoding` and `absent` (rr:src/workbook_delta/lanes.rs:92-98). DT64 would then witness it the way it witnesses every other source fact: through a source revision on the frame. Neither `ClassView` nor `EpiLaneSpec` is the right home: the scale is a property of the source field, not of the class's byte layout. This is an open decision (§7).

---

## 3. WHAT — per operation

Model: value = (2P−1)·M·10^(−s) under P×M; under i32, value = c·10^(−s) with absent = presence 0 ∧ c = 0. `s` is fixed per bound field.

**Do not use Codec9's scale as `s`.** Codec9 encodes `R::round(value / scale)` from `f32` and decodes `f32::from(v) * scale` (nd:src/hpc/codec9.rs:156-193). The scale is a positive finite `f32` checked at construction (:271-281), and the encoder's only input is an `f32` (:226, :290). `0.01_f32` is not 1/100, so `encode(1.10)` with scale 0.01 is a rounding of a binary fraction, not a parse of `"1.10"`. The only exact integer entry is `EpiLane9::from_codeword` (:85-94) or building `Pm` bytes directly (nd:src/hpc/abakus/pm.rs:196-209, test helper). An exact-decimal path must parse the text into the coefficient itself and never pass through `encode_touched`.

| Operation | i32 lane path | P×M path | Verdict |
|---|---|---|---|
| **Projection** | VRL projects `i32` or `u32` lanes, `Datum::I32` (lg:crates/lance-graph-vrl/src/lib.rs:1179-1184); presence law in `Rows::present` (:1188-1194) | `PmSum::lane` → `PmLane::{NeverTouched, Exact(i16), Escaped}` (nd:src/hpc/abakus/pm.rs:103-115); no `LaneRef` byte form, so no binding into VRL or mask-risc (lg:crates/lance-graph-mask-risc/src/ir.rs:16-36; refusal `PresenceIsSign` "the operand is" missing, lg:crates/lance-graph-r2il/src/binding.rs:325-337, :411-421) | i32: REUSE-UNCHANGED (coefficient); P×M: MISSING (binding) |
| **Equality** | `EqI32`/`NeI32` (ir.rs:137-139), through VRL `Cmp` (vrl lib.rs:296-306). Numeric, so `1.20` = `1.2` once both are coefficients at one scale | `eq` three-valued, untouched = unknown (nd:examples/pm_stats/stats.rs:175-183); exhaustive oracle (nd:tests/pm_stats.rs:41-110). **Example, not library** | i32: COMPOSE (literal → coefficient at prepare); P×M: lab only |
| **Ordering vs literal** | `LtI32`…`GeI32` (ir.rs:129-136). VRL literals are `parse::<i32>()`, and `1.5` is refused as "non-i32 comparison" (vrl lib.rs:392-394) | `lt_const` (nd:examples/pm_stats/stats.rs:196-207), ±255 only, example | i32: COMPOSE (needs a directed-rounding literal resolver, see §5); P×M: lab only |
| **Sum** | `MaskedSumI32` → `i64`, exact to 2^32 rows (ir.rs:296-299, :710-716); `Recipe::sum` (vrl lib.rs:873-884) | `add_pm`/`sub_pm` exact with a carry mask (pm.rs:26-31, :130-172); signed fold `S = ΣM − 2ΣM⁻` (stats.rs:81-88), example | i32: REUSE-UNCHANGED (result is the coefficient, scale unchanged); P×M: lab fold + carry escape |
| **Mean / variance** | `GroupPowerSumsI32` → `PowerSums { n: u64, sum: i64, sum_sq: u128 }` (ir.rs:458-480; nd:src/simd_masking_ops.rs:2325-2332). Grouped only. Ungrouped means grouping by one constant key, OPEN whether `GroupKey` admits that cheaply | `Stats {n,s,q,a}` and exact `variance_parts → (N·Q − S², N²)` as `i128` (nd:examples/pm_stats/stats.rs:64-69, :141-149), **example** | i32: REUSE (grouped) / COMPOSE (scale: mean is S/(N·10^s), variance (NQ−S²)/(N²·10^{2s}), exact rational); P×M: lab only |
| **Scale alignment** (mixed scales) | nothing | nothing | MISSING. HYPOTHESIS: refuse at bind (one scale per bound field, the `CoordinateSpace` refusal pattern, dt64 lane.rs:13-17, :183-188); promote only by an exact ×10^k at prepare, with a range check |
| **Multiplication** (amount × rate) | no i32×i32 terminal in `Terminal` (ir.rs:288-497); `GroupCrossPowerSumsI32` gives Σxy only (ir.rs:481-497) | `prod_sum` Σ(a·b) of magnitudes, exact (stats.rs:21-32); `dot_split` (stats.rs:233) | MISSING as a per-row product; Σxy COMPOSE (scale s₁+s₂) |
| **Division** | none | none | MISSING; not needed in the hot path. Mean/ratio stays an exact rational, rounded only at formatting (HYPOTHESIS) |
| **Formatting** | none; report-rs passes text through (rr:lanes.rs:75-80) | `Codec9::decode` → `f32`, lossy (codec9.rs:187-193) | MISSING (late, scalar, per displayed cell; trivial integer → digits with `s` fraction digits) |
| **Intake parsing** | `parse_decimal` validates and keeps text, no coefficient (rr:reconcile.rs:233-242); `parse_integer` canonical i64 (:223-231) | n/a | COMPOSE: the validator already splits int/frac (:235-237); turning that into a coefficient at a declared scale is a small scalar step |
| **Exact widening past i32** | no `I64` `LaneRef` (ir.rs:16-36) | `PmSum` carry escape (pm.rs:26-31); `Quantized9::Wide(i64)` for input (codec9.rs:139-152); PM17/25/33 are HYPOTHESIS only (nd:.claude/knowledge/adaptive-pm-width-slice-seal-research-prompt.md:4, :21) | MISSING above i32 |

Is it a kernel or a formula?
- **Executable SIMD kernels, proven:** `add_pm`/`sub_pm`/`neg_pm`, exhaustive over 512×512 on seven tiers (nd:.claude/knowledge/add-pm-probe.md:47-62, :83-90). The mask-risc i32 terminals are also executable.
- **Executable but example-scoped:** the pm-stats `fold`, `lt`, `eq`, `prod_sum` (nd:examples/pm_stats/stats.rs, header :1-5 "Research only … no new kernel, API or layout"). They are not in the library.
- **Formula only:** decimal scale alignment, mean and variance with a base-10 scale, PM17+ arithmetic.

---

## 4. WHICH — where the benefit is (only numbers that appear in the sources)

- **Statistical fold N, S, Q, A on P×M, AVX-512:** 12–18 ns per 64 lanes against a scalar i16 loop at 41–51; no win on AVX2 (44–47 against 40–48). The source's own reading: "The strongest fruit is the statistical fold on AVX-512" (nd:.claude/knowledge/pm-stats-probe.md:53-56, :68). The baseline is a poorly vectorized scalar loop (:38-41), not mask-risc's `GroupPowerSumsI32`.
- **Signed sum S, density-independent:** AVX-512 2.9–3.9 ns per 64 lanes, flat; `masked_sum_i32` 1.7 at ≤1% selected, 12 at ≥50% (pm-stats-probe.md:45-50). P×M wins on dense selections and loses on sparse ones.
- **Where the advantage comes from:** "Every advantage measured here comes from bytes (72 vs 128–256 per 64 lanes) or from density-independence, not from fewer instructions" (pm-stats-probe.md:71). 72 bytes per 64 rows is the DR512 footprint (nd:.claude/knowledge/dr512-epistemic-pm-invariant.md:31-33).
- **`add_pm` in place against decode → i32 → encode:** 11.8–11.9 ns against 89.7 ns on AVX-512, and 284.7–285.3 against 133.5–134.7 ns on AVX2, where it is slower because of the array-backed `U8x64` (nd:.claude/knowledge/add-pm-probe.md:94-109).
- **Comparison:** "Signed comparison is not cheaper in ALU work … `x < 0` is genuinely cheaper" (pm-stats-probe.md:69).
- **Wider widths, storage only:** PM25 is "a capacity calculation, not native five-lane SIMD, not measured throughput" (adaptive-pm-width…md:21).

What this means for currency (HYPOTHESIS, no measurement): a two-decimal amount fits PM9 only up to ±2.55, so report-rs amounts cannot use the measured PM9 benefits. They need PM17+ (unbuilt) or i32 (built, at 4 bytes plus a presence bit, versus PM9's 1 byte plus a polarity bit). The near-term benefit is correctness and de-duplication: numeric equality, ordering, sum and exact variance with no second population. A storage or bandwidth benefit is only plausible later, if PM17/25 is built and measured. Those width probes were "Not measured" in #439 (pm-stats-probe.md:79).

---

## 5. HOW — the smallest isolated probe

**Shape:** one decimal field, one homogeneous scale `s` declared by the source, coefficient in the existing i32 `Signed` lane plus its presence plane. Read through the existing DT64 → VRL → mask-risc path. No new runtime, no new `Datum` variant, no `f64`.

**Where it would live (HYPOTHESIS):** a lab crate or test in lance-graph next to `lance-graph-vrl` (as `vrl-recipe-probe` is the lab for VRL, vrl lib.rs:3-5), holding:
1. **A lexical parser** `text → (coefficient: i64, scale: u8)` with an exact refusal. It reuses the shape `parse_decimal` already validates (rr:reconcile.rs:233-242) and adds a canonical coefficient at the field's declared `s`. Fewer fraction digits are padded (`"1.2"` at s=2 → 120). More digits than `s` are refused unless the extra digits are zeros (`"1.200"` → 120; `"1.205"` → refuse). It never rounds. It refuses on i32 overflow, or widens exactly to an i64 escape if one is admitted.
2. **A literal resolver at prepare time,** mirroring how text literals resolve to codes before preparing (vrl lib.rs:66-69, :551; the unrepresentable case already maps to `Range{lo:0,hi:0}` / `Range{0,hi}`, :1050-1056). For a literal L with exact coefficient L·10^s:
   - representable: compare directly;
   - not representable at `s`: `==` gives an empty mask, `!=` gives every present row, `<`/`<=` take `ceil`/`floor` of L·10^s into `Lt`/`Le`, `>`/`>=` mirror.

   All of this is computed in exact integer or rational arithmetic on the literal's digits, never `f64`.
3. **Fold:** `Recipe::count` / `Recipe::sum` (vrl lib.rs:844-884) unchanged; the sum is a coefficient at scale `s`. Mean and variance via `GroupPowerSumsI32` (ir.rs:476-480) as the exact rational `(N·Q − S², N²·10^{2s})`, the same form as `variance_parts` (nd:examples/pm_stats/stats.rs:141-149).
4. **Late formatting:** coefficient + `s` → digits, on the displayed cells only.

**Falsifiers.** Each should have a can-fire and a can-stay-silent case on non-trivial input:
- **Established zero vs untouched:** `"0.00"` (presence 1, c=0) counts in N and reads `0.00`; a blank cell (presence 0, c=0) does not count. Disable: dropping the presence bit from N must turn the test red. This is the law in lane_codec.rs:55-63 and `Rows::present` (vrl lib.rs:1191-1194).
- **Negatives:** `"-0.01"` → c=−1, present. On i32 this is two's complement with presence 1; intake already sets presence for every written cell (lg:crates/lance-graph-intake/src/lib.rs:1216-1222), so the probe asserts it rather than having to establish it.
- **`"-0"` / `"-0.00"`:** must become established +0 (c=0, presence 1), never absent. This mirrors Codec9's "rounded zero canonicalized to +0" (nd:src/hpc/codec9.rs:13-14). `parse_decimal` accepts `"-0"` today (rr:reconcile.rs:235-241).
- **Exact cancellation:** +12.34 and −12.34 sum to 0 with N=2 (the P×M analogue is `−5 + +5 = +0`, pm.rs:217).
- **Maximum coefficient boundary:** c = i32::MAX and i32::MIN, then one more digit refuses or escapes. Sums near 2^32 rows (ir.rs:710-716).
- **Carry / widening:** for P×M variants, ±255 then ±256 lands in `PmLane::Escaped` (pm.rs:211-224). For i32, the parse at 2^31 must refuse, never wrap.
- **Mixed scales:** two fields at s=2 and s=3 in one comparison or sum are refused at bind, or aligned only by an exact ×10. A test proves that summing them without alignment is caught.
- **`1.20` vs `1.2`:** equal as coefficients at s=2 (same i32). Kept as a regression against the current code behaviour, where they are different `EqU32` codes (rr:reconcile.rs:233-242 keeps the text).
- **Threshold between representable coefficients:** `x < 1.205` at s=2 keeps 1.20, drops 1.21; `x == 1.205` selects nothing; `x != 1.205` selects every present row.
- **Stale scale witness:** re-declare the field's scale (for example a re-import at s=3), then run a prepared program. It must be refused. Through intake a re-declared scale is a new table, so the old program meets `ForeignFrame`; the probe should pin that, and §7.1 records what a source that changes scale in place would need.
- **Stale width witness:** for P×M variants, `EpiLaneSpec.width` changing under a binding.
- **Independent oracle:** a scalar big-integer or `i128` rational oracle over the decimal *text*, not `f64`. Compare every count, sum and variance rational. The P×M tests already use an `i32` oracle the same way (nd:tests/pm_stats.rs:169-185; add-pm-probe.md:51).

**Concrete report-rs seam (HYPOTHESIS):**
- `FieldType::Decimal` (rr:src/workbook_delta/schema.rs:24-29) gains a declared scale in the field policy (the `FieldMapping` already carries `ty`, :43-49).
- `lanes.rs` maps decimal fields to `column(PgType::Int4, LaneEncoding::Signed)` instead of `Text/TextCode` (rr:src/workbook_delta/lanes.rs:144-145).
- `convert` produces the coefficient from `CellState::Text` by exact lexical parse (rr:reconcile.rs:196-201). The raw text lane `r{p}` keeps the verbatim spelling for display (rr:lanes.rs:5-7).
- `CellState::Number` (an `f64` from calamine `D::Float` or gridcraft `Value::Number`, rr:src/workbook_delta/read.rs:159, :193-194) needs its own rule. Converting an `f64` to an exact decimal at scale `s` is exact only if the shortest round-trip text has ≤ s fraction digits. That is the existing `n.to_string()` (rr:reconcile.rs:211), so the probe can parse that text and refuse anything else.
- Blocker: `PgType` admits no `numeric` (lg:pg_row.rs:55-58), and D-LANESOCK keeps exact decimal closed (lg:.claude/plans/d-lanesock-dr-dt64-lane-socket-v1.md:165). Using `Int4` plus a side scale avoids widening `PgType`, but it is a decision, not a code fact.

---

## 6. Classification

| Layer | State | Evidence |
|---|---|---|
| Existing representation (signed integer coefficient + presence law) | **ESTABLISHED** for i32 and PM9; **PARTIAL** as a decimal (no scale anywhere) | i32: pg_row.rs:156-157, intake bind.rs:111, lane_codec.rs:55-63; PM9: codec9.rs:1-21, pm.rs:11-16; scale: none in lane_codec.rs / lane.rs:258-280 / dr.rs:360-364 |
| Mathematical coverage (sum, compare, N/S/Q/A, exact variance, Σxy) | **ESTABLISHED** for integers; **PARTIAL** for decimals (scale bookkeeping is formula only; product per row, alignment, division MISSING) | pm.rs:130-172; stats.rs:64-149; ir.rs:296-299, :458-497 |
| Physical execution | **ESTABLISHED** for i32 (mask-risc + VRL); **PARTIAL** for P×M (library add/sub/neg only; stats/compare are examples; no `LaneRef` byte form; ±255); **MISSING** above i32 / PM17+ | ir.rs:16-36; vrl lib.rs:19-22; r2il binding.rs:325-337; pm-stats-probe.md:6-7, :79; adaptive…md:4 |
| Tertiary integration (intake parse, literal resolver, scale witness, formatting, report-rs lanes) | **MISSING** | rr:reconcile.rs:233-242 (text kept); rr:lanes.rs:144-145 (TextCode); vrl lib.rs:392-394 (non-i32 literal refused); pg_row.rs:55-58 |
| Production readiness | **MISSING** | lg D-LANESOCK "Still closed … exact decimal" (plan:165); abakus "Experimental" (nd:src/hpc/abakus/mod.rs:7-8); pm-stats "Research only" (pm-stats-probe.md:6-7) |

---

## 7. OPEN

1. **Does the seal witness scale or width?** Not today. `FrameRevisions` holds source, relation_map and coordinates revisions (dr.rs:360-364), and `LaneBinding` holds no width or scale (lane.rs:258-280). Width is a compile-time class constant (`EpiLaneSpec`, lane_codec.rs:104-114). Verified by the orchestrator: intake bumps `source` on every data write, append or delete (`advance_data`, lg:crates/lance-graph-intake/src/lib.rs:1005-1008), so `SourceChanged` is a data-version witness, not a scale witness. A column's shape (and so a declared scale) is fixed when the table is built from its `ShapeConfig`; a re-declared scale is a new table under a new frame store, which `ForeignFrame` already refuses. OPEN: whether that is enough (a scale tied to the table's lifetime) or whether a source that can change a field's scale in place needs a scale field in the binding.
2. **Does report-rs keep lexical decimal text?** Text cells: yes, verbatim (reconcile.rs:196-201, :233-242). Numeric cells: no. report-rs receives an `f64` from calamine (`D::Float`, read.rs:193) or gridcraft (`Value::Number`, read.rs:159). Whether the `<v>` text inside the XLSX was retrievable before calamine parsed it is outside report-rs and UNKNOWN. I did not read calamine's or gridcraft's source.
3. **Ungrouped power sums.** `GroupPowerSumsI32` is grouped (ir.rs:458-480). Whether a single-group `GroupKey` is cheap, or an ungrouped terminal is wanted, is OPEN.
4. **Presence for negative i32 values — SETTLED (verified by the orchestrator).** Intake sets the presence bit for every cell that holds a value, whatever the value: `if cell.is_some() { plane[at / 64] |= 1 << (at % 64) }` (lg:crates/lance-graph-intake/src/lib.rs:1216-1222). A negative or zero coefficient written through intake is therefore present; only an empty cell is absent.
5. **i32 range sufficiency.** At s=2, i32 holds ±21,474,836.47. Whether report-rs workbook amounts fit is a data question. UNKNOWN; no fixture inspected.
6. **PM17/25/33 economics** for currency-sized coefficients: unbuilt and unmeasured (adaptive…md:4, :21; pm-stats-probe.md:79).
7. **Where the scale is declared:** intake `ColumnConfig`, the report-rs field policy, or a contract type. A design decision; no source settles it.
8. **Percentages:** whether a percentage is a decimal at its own scale (for example s=4 for 12.34%) or a ratio computed late. Unexamined. HYPOTHESIS: the same carrier at a declared scale.
