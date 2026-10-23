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

## 8. Layer benchmark — ndarray primitives → mask-risc → R2IL polyfill

`examples/layers.rs`: five folds over `x > t` (t = 0, about half the rows), each through the three layers on the same resident `&[i32]`. Each layer's result is checked against the `i128` oracle at every size before it is timed. Runs 2 and 3 of three, ns per call (run 1 dropped as noisy at 64 rows). "Before r2il" is the mask-risc column, "after" is the R2IL column.

| rows | op | scalar | ndarray | mask-risc | Δ | R2IL | Δ | allocs nd/mr/r2il |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 64 | count | 8–15 | 17–21 | 87–121 | +70–100 | 153–175 | +54–66 | 0/0/1 |
| 64 | exists | 6 | 17 | 90–95 | +73–78 | 149–172 | +59–77 | 0/0/1 |
| 64 | sum | 11–12 | 36–38 | 88–90 | +52 | 162–172 | +74–81 | 0/0/1 |
| 64 | min | 28–30 | 39–41 | 89–92 | +50–52 | 177 | +85–88 | 0/0/1 |
| 64 | max | 34–38 | 39–41 | 99–107 | +58–68 | 175–179 | +72–76 | 0/0/1 |
| 4 096 | count | 421–428 | 253–254 | 321–324 | +68–71 | 383–386 | +61 | 0/0/1 |
| 4 096 | exists | 6 | 255–280 | 317–349 | +62–69 | 381–385 | +36–64 | 0/0/1 |
| 4 096 | sum | 606–628 | 1 548–1 691 | 1 598–1 808 | +50–116 | 1 661–1 825 | +18–63 | 0/0/1 |
| 4 096 | min | 1 701–1 717 | 1 671–1 843 | 1 729–1 864 | +21–58 | 1 743–1 931 | +14–67 | 0/0/1 |
| 4 096 | max | 1 690–1 923 | 1 613–1 825 | 1 698–1 743 | −82–+85 | 1 803–1 935 | +60–236 | 0/0/1 |
| 65 536 | count | 9 386–9 850 | 4 809–5 367 | 4 988–5 380 | noise | 5 066 | noise | 0/0/1 |
| 65 536 | exists | 6 | 4 754–5 172 | 4 952–5 491 | noise | 5 341–5 540 | noise | 0/0/1 |
| 65 536 | sum | 9 924–10 951 | 28 658–30 789 | 28 661–30 575 | noise | 30 228–30 975 | noise | 0/0/1 |
| 65 536 | min | 28 676–30 353 | 32 996–33 760 | 31 903–32 749 | noise | 32 326–32 784 | noise | 0/0/1 |
| 65 536 | max | 27 833–30 056 | 29 384–32 385 | 34 665–34 770 | noise | 32 332–33 131 | noise | 0/0/1 |

- **Per call:** mask-risc adds ~50–100 ns over ndarray, and the R2IL polyfill adds another ~55–90 ns plus one allocation. Both costs are fixed: at 4 096 rows they shrink to tens of ns, and at 65 536 rows they are inside run-to-run noise. The polyfill layers cost per call, not per row.
- **Per row:** count is the one fold where the ndarray path beats the scalar loop (~0.08 vs ~0.15 ns/row, `popcount_batch_u64`). For sum the ndarray path is ~3× slower than the scalar loop (the bit walk, §7). For min and max it matches the scalar loop, which does not vectorize them either (~0.45–0.5 ns/row).
- **Exists is not equal work:** the scalar loop stops at the first hit (row 4 here), so it takes 6 ns at every size. All three layers build the whole mask first. That is an algorithmic difference, not overhead: a fused predicate + `Any` that stops at the first non-empty tile would close it. Proposal only.

## 9. D-MPF-3 (lab) — hybrid masked sum and mask reuse

`examples/hybrid.rs`, two runs, 65 536 rows unless stated, ns per call. Every arm is checked against an `i128` oracle at every configuration. Nothing in ndarray or R2IL was changed. A winning body is a proposal for the `ndarray::simd` owner, approved separately.

**Hybrid sum.** Per mask word, three cases:
- an empty word is skipped;
- a word with at most `th` set bits uses the bit walk (the current `masked_sum_i32` body);
- a denser word is summed branch-free over all 64 lanes.

Two mask shapes at the same density: `uniform` (independent bits) and `clustered` (whole words set or clear).

| mask | density | ndarray walk | hybrid th=16 | branch-free only (th=0) |
|---|---:|---:|---:|---:|
| uniform | 1 % | 1 408–2 377 | 1 485–2 281 | 5 373–5 607 |
| uniform | 10 % | 12 124–12 325 | 5 538–5 891 | 11 450–12 762 |
| uniform | 50 % | 32 893–39 806 | 11 533–11 726 | 11 660–12 745 |
| uniform | 100 % | 42 732–44 840 | 11 708–12 334 | 12 130–12 477 |
| clustered | 10 % | 6 505–7 015 | 1 361–1 436 | 1 355–1 424 |
| clustered | 50 % | 25 445–25 887 | 5 424–6 925 | 5 852–6 763 |

- `th = 16` matches or beats the walk at every density measured: ~2× at 10 %, ~3× at 50 %, ~3.7× at 100 % (uniform), 4–5× on clustered masks.
- Branch-free alone loses 2–4× on sparse uniform masks, so the walk has to stay for sparse words.
- The threshold is flat between 8 and 32, so it is not a sensitive knob.
- `masked_min_i32` / `masked_max_i32` use the same walk (`masked_fold_i32`, "cost proportional to the popcount") and are the next candidates.

**Mask reuse across count, sum, min and max of `x > t`.**

| rows | selected | recompute ×4 | reuse | reuse + hybrid(16) | fused scalar loop |
|---:|---:|---:|---:|---:|---:|
| 64 | 3 | 84–85 | 29–30 | 29–30 | 19–20 |
| 64 | 28 | 132–137 | 85–89 | 74–76 | 19–20 |
| 4 096 | 196 | 1 707–1 778 | 855–872 | 905–910 | 907–926 |
| 4 096 | 2 035 | 5 515–5 586 | 4 783–5 119 | 3 978–5 501 | 965–996 |
| 65 536 | 3 239 | 30 415–51 405 | 15 624–16 224 | 15 174–24 062 | 15 583–18 176 |
| 65 536 | 32 925 | 120 471–145 908 | 95 255–97 732 | 94 472–98 539 | 15 605–15 792 |

- Reusing one mask removes the repeated compares: about 2× when sparse and ~20–30 % when dense.
- At ~5 % selected, the reused mask matches a fused scalar loop. At 50 % selected, the fused loop is ~6× faster, because min and max still walk every set bit.
- The mask form pays off when the selection is sparse or the reductions are bit-only (count). With dense selections and several value reductions, a fused pass wins until min and max get the same hybrid treatment.
- OPEN: hybrid min/max; a fused mask-then-multi-reduce terminal; the ndarray owner's call on adopting any body.
