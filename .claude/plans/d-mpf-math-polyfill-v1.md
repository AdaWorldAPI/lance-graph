# D-MPF — the mathematical polyfill API, Stage 1 (source audit + folding probe)

**Status:** D-MPF-1 MEASURED, lab only (`crates/math-polyfill-probe`). D-MPF-2.. PROPOSED. Independent of ONNX and of the open S0 drafts (#1744, #1745, #1750); it does not choose between them.

## 1. Finding: the Stage 1 polyfill API already exists

The polyfill is the typed procedure layer that already ships over R2IL, not a new API:

| layer | what it is | where | status |
|---|---|---|---|
| set description + proven rewrites | `Rel` (All/Empty/Plane/Atom/And/Or/Not), `Measure` (Count/Sum/Min/Max/Keep), `normalize`, `symbolic` | `crates/lance-graph-relalg` (algebra in `lance-graph-contract::set_algebra`, D-RA-2 = A) | SHIPPED |
| procedures (named R2IL call sequences, no new opcode) | `select_lt filter_lt select_u32 count exists all differ intersect union keep count_and sum min max and_cmp_i32 group_sum margin is_null is_not_null and_between_i32 in_u32 count_valid avg_parts bool_or bool_and not_exists except count_distinct_sorted count_union_all` | `crates/lance-graph-r2il/src/procs.rs` | SHIPPED |
| checked scalar arithmetic on fold results | `IntAdd IntSub IntMult` (overflow refused), `IntSDiv` (division by zero refused) | `FoldDialect` in `crates/lance-graph-r2il/src/lib.rs` | SHIPPED |
| bulk engine | mask-risc `Program` + terminals `Count Any All MaskedSumI32 MaskedMinI32 MaskedMaxI32 MaskedStridedGroupSum BlendI32 Keep`, ternlog `fuse` | `crates/lance-graph-mask-risc` | SHIPPED |
| contract refusals | `Refusal::{NotTwoValued, NoUniverse, Checked, Ieee, SymFamily, Impure, BasisMismatch, PastUniverse, Bag}` | `crates/lance-graph-contract/src/set_rules.rs` | SHIPPED |
| proof / certified recipe | egglog-proved associativity of Boolean-Distinct composition over DR64, `RecipeCertificate` holds egglog's `ProofStore` + `ProofId` | `crates/dr-math-binding-probe` (#1707, #1740, #1742) | LAB |

Typing the procedures carry today: domain `i32`/`u32` lanes, `i64` scalars; Boolean set algebra on masks; integer `+` over selected rows; NULL as a presence plane (`WHERE` drops UNKNOWN; full K3 lives in Quack `sql_where`); multiplicity Set (Bag refused by the contract); no float lane; every answer exact.

**ONNX/protobuf fork:** `AdaWorldAPI/optimum-onnx-rs` exists and is empty (no branch, no commit, checked 2026-10-23). `AdaWorldAPI/onnx` is not accessible to this session. There is no integration surface to audit yet (UNKNOWN until the operator names it).

## 2. D-MPF-1 — one expression through the existing polyfill

`crates/math-polyfill-probe` writes one expression using only the procedures and scalar calls above:

    E = SUM(v | sel) − k·COUNT(sel),   sel = valid(v) ∧ x ≥ lo ∧ y < hi

`eval` lowers it onto a lent `FoldDialect` (two folds, then `IntMult`, `IntSub`). `oracle` is a row loop in `i128` that shares no code with the procedures, the dialect or mask-risc.

**Gates (`tests/polyfill.rs`, 9 tests, all green):**

- equal to the oracle at 64, 1 000, 4 096 and 65 537 rows, for empty, partial and full selections and multipliers 3, 0, −7, 1;
- anti-vacuity: one case selects nothing, one a strict subset, and NULL rows exist inside the selected range;
- NULL is load-bearing: with every row present the answer changes, and the polyfill still matches;
- relalg's `Query` and the procedures count the same rows (two frontends, one substrate);
- typed refusals, each an existing `FoldError`: lane + lane (`LaneVersusLane`), unsigned compare (`NoUnsignedLaneCompare`), scalar overflow (`Overflow`, with a fitting product accepted), an expression whose exact value leaves `i64` (`Overflow`, the oracle confirms it does not fit), division by zero (`DivisionByZero`).

**Disable runs (committed first, anchors asserted):** D1 drop the NULL gate → 3 tests red; D2 `≥` → `>` → 2 red; D3 `IntSub` → `IntAdd` → 1 red.

## 3. Cost split (reported, not gated)

`cargo run --release --example cost_split`, 4-core container, three runs; ns per call, median of 21 batches. `exec` is mask-risc `execute_into` of the prebuilt count program alone; `count` is selection + `COUNT` through the dialect; `expr` is all of E.

| rows | bind | exec | count | expr | count+bind | keep+bind | relalg | oracle | expr ns/row |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 21–24 | 116–129 | 271–279 | 576–627 | 307–362 | 299–315 | 2 723–2 934 | 78–86 | 9.0–9.8 |
| 512 | 28–56 | 246–463 | 324–731 | 792–1 867 | 395–904 | 385–952 | 2 937–5 627 | 651–738 | 1.6–3.7 |
| 4 096 | 66–74 | 1 056–1 246 | 982–1 116 | 3 197–3 599 | 1 128–1 213 | 1 070–1 215 | 3 574–4 089 | 5 237–6 430 | 0.78–0.88 |
| 65 536 | 112–123 | 17 102–18 534 | 15 196–15 785 | 56 625–59 361 | 16 345–16 523 | 15 253–16 432 | 20 022–22 501 | 91 886–96 264 | 0.86–0.91 |

What the numbers say:

- **Per row, the target holds:** above ~4 K rows the expression costs under 1 ns/row and beats the scalar loop by 1.6–1.7×; the fold alone (`exec`) is ~0.27 ns/row at 64 K.
- **Per call at register scale, it does not:** at 64 rows (one `u64` word) one fold costs ~120 ns in mask-risc plus ~150 ns of dialect lowering, and the scalar loop (~80 ns) is faster than the whole expression (~600 ns). "Nanosecond hot execution" is therefore not met by the existing per-call API for one-word populations.
- `relalg::run` adds ~2.5 µs of fixed cost (normalize through the contract) over the dialect; worth it for rewrites, not for a hot one-word path.
- Materializing the mask (`keep+bind` vs `count+bind`) costs nothing measurable.
- 512-row figures vary 2× across runs; treat them as noisy.

## 4. Gaps, each with its counterexample

| id | gap | counterexample (test) | status |
|---|---|---|---|
| G1 | no element-wise value arithmetic (`x + y` per row) on the fold path | `lane_plus_lane_is_refused` | OPEN, owner decision: R2IL/mask-risc vs an existing ndarray kernel behind a typed method |
| G2 | no unsigned lane compare | `unsigned_compare_is_refused` | OPEN (known since D-SQLP-1) |
| G3 | no float lane kind; nothing IEEE is admitted | contract `Refusal::Ieee` | OPEN; floats route to ndarray kernels, measured, never certified |
| G4 | ~270 ns fixed cost per fold at one word | cost table row 64 | OPEN; D-MPF-2 measures where it goes before anything is changed |

No new primitive is proposed in this PR. G1 is the first place a counterexample shows composition is insufficient.

## 5. Next stages (smallest experiment each; none authorized here)

- **D-MPF-2:** attribute the 64-row fixed cost (Program build, scratch carve, tile loop, terminal) with counters; decide whether a one-word fast path is a mask-risc change or an existing `ndarray::simd` call. Gate: a one-word fold under 50 ns, equal to the oracle, or a recorded reason why not.
- **D-MPF-3 (G1):** the owner chooses where element-wise integer arithmetic lives. Counterexample to beat: `lane_plus_lane_is_refused`.
- **Stage 2 (ONNX wiring):** blocked until the ONNX/protobuf integration surface exists (§1) and S0 ownership is settled.
- **Stage 3 (DR64³):** the existing `relcomp::compose` R∘S chain is the start (#1747); separate plan.
- **Stage 4 (Palette256):** reuse BGZ/Helix/ndarray; the Burn compiled-linear identity defect (#1749/#1751 threads) must be fixed first.
- **Stage 5 (egglog):** one rewrite with a checkable proof, compiled to an existing recipe, fresh-frame rebind, oracle match, one refused rewrite.
- **Stage 6 (Optimum, #1751):** last.

## 6. Open points

| id | question | owner | status |
|---|---|---|---|
| MPF-OP-1 | where element-wise integer arithmetic lives (G1) | contract / R2IL owners | OPEN |
| MPF-OP-2 | which repository is the ONNX/protobuf fork | operator | OPEN |
| MPF-OP-3 | is a one-word fast path a mask-risc concern | mask-risc owner | OPEN (D-MPF-2) |
| MPF-OP-4 | S0 owner among #1744/#1745/#1750 | operator / EXO-50 | OPEN |
