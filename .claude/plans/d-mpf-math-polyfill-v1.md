# D-MPF — the mathematical polyfill API, Stage 1 (source audit + folding probe)

**Status:** D-MPF-1 MEASURED, D-MPF-2 MEASURED (§7), lab only (`crates/math-polyfill-probe`). D-MPF-3.. PROPOSED. Independent of ONNX and of the open S0 drafts (#1744, #1745, #1750); it does not choose between them.

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
  > ⊘ **Corrected by D-MPF-2 (§7):** "beats the scalar loop" compared against the `i128` filter-chain oracle, not an equal-work loop. A branch-free `i64` loop over the same `&[i32]` with a runtime threshold takes ~11 µs at 64 K rows (0.17 ns/row) and is ~3× faster than the SIMD-path fold. The per-row figure (<1 ns/row) stands; the comparison does not.
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
| G4 | ~270 ns fixed cost per fold at one word | cost table row 64 | ATTRIBUTED (§7); no change made |

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

## 7. D-MPF-2 — where the cost goes (measured, nothing changed)

`examples/fixed_cost.rs`, `examples/floor_split.rs`, `examples/codegen_probe.rs` + `codegen-witness.sh`. Host: AVX-512 (`target-cpu=native`), 4 cores, other load present; two clean runs of three kept, one dropped as noisy. Every arm equals the `i128` oracle at every size before timing. No primitive, contract or ABI was changed.

**Per call, one question (`Σ x > t`), each arm one layer more than the last** (ns, two runs):

| rows | scalar `i64` | ndarray floor | mask-risc compiled | + recognition | + `Program::new` | dialect |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 12 | 39 | 94–99 | 97–100 | 113–125 | 171–172 |
| 4 096 | 664–686 | 1 721–1 831 | 1 789–1 901 | 1 792–1 911 | 1 811–1 936 | 1 839–1 975 |
| 65 536 | 10 967–11 817 | 30 884–42 348 | 31 048–31 051 | 30 578–31 190 | 32 925–36 005 | 33 765–36 822 |

Allocations per call: 0 for the first four arms, 1 for `Program::new` and for the dialect (its `ops` vector is moved into each program and re-grown on the next fold).

At 64 rows the 172 ns split as: ndarray floor 39 (memset, 4 compares, the bit walk), mask-risc execution ~57 (precheck, validation, tile dispatch, scratch carve), program build ~20 + 1 allocation, dialect ~50 (stack, epoch, op accumulation, result extraction). Recognition (`compile`) is ~3 ns.

**Per row, the floor is the bottleneck, not mask-risc:** at 64 K rows mask-risc compiled ≈ ndarray floor. Inside the floor (`floor_split`, 65 536 rows, three runs):

| selected | compare | `masked_sum_i32` | branch-free sum | scalar |
|---:|---:|---:|---:|---:|
| 0 | 5.4–9.2 µs | 1.2–1.7 µs | 12.2–20.8 µs | 10.8–11.5 µs |
| 32 429 | 5.5–6.0 µs | 24.6–26.5 µs | 12.2–12.5 µs | 10.7–10.9 µs |
| 65 536 | 5.2–5.6 µs | 43.8–52.6 µs | 12.2–17.7 µs | 11.1–12.4 µs |

**LLVM** (`codegen-witness.sh`, PASS ×3): `gt_i32_to_mask` is one `vpcmpltd` + `kmov` per 16 lanes; `masked_sum_i32` is a scalar `tzcnt`/`blsr` walk with no vector register; the branch-free sum vectorizes (`vpsrlvq`, `vptestmq`, `vpmovsxdq`, `vpaddq`); the scalar row loop vectorizes on ymm.

**Bottleneck:** `ndarray::simd::masked_sum_i32` walks set bits one at a time, so its cost tracks the popcount (0.03 → 0.67 ns/row from 0 % to 100 % selected) and it is ~75 % of the fold at half selectivity.

**One proposed optimization (not made; owner of `ndarray::simd`):** choose the body by density. When a word has more than ~16 set bits, sum it branch-free; otherwise keep the walk. Negative controls already measured: the branch-free form equals the oracle at 0 %, 50 % and 100 %; its time is flat in selectivity (the mechanism is the missing branch, not the data); and the walk is 7–10× faster at 0 %, which is why the walk must stay for sparse words. Expected effect at 50 %: fold ~31 µs → ~18 µs at 64 K rows. Even then a one-pass fold stays slower than the scalar loop; the mask form pays off when the mask is reused or combined (NULL planes, several predicates), which this probe does not measure.

**Fixed cost (G4):** the 50 ns gate is not met and is not reachable by tuning one layer: the floor alone is 39 ns at one word. A second proposal, for the r2il owner: keep the `ops` vector's capacity across folds (reclaim it from the previous program) to remove the one allocation per fold. Not made.
