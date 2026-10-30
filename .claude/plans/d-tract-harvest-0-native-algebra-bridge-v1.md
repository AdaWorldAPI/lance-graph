# D-TRACT-HARVEST-0 — the ONNX graph as a DR#3 graph through the DT64 registry (Tract as cold frontend)

**Date:** 2026-10-30. **Status:** LAB PROBE IMPLEMENTED (crates/tract-harvest-probe, own workspace, excluded from the lance-graph workspace). Not a production path; no native crate depends on it. Implements the S1 (source preservation) and S3 (one admitted native subgraph) stages of D-ONNX-INTAKE (#1802) for the Tract arm only; Burn onnx-ir was not run. **Coordinator:** D-NAM / EXO-152.

## 0. Redesign (2026-10-30): the ONNX graph as a DR#3 graph, resolved through the DT64 registry

> **Supersedes** the fixed admission recipes (`admit.rs`, removed) and with them §7 and the recipe-specific rows of §1, §8 and §9. The operator's correction: *"We want the graph … The mapping of the Graph should then resolve to polyfill and predicate methods in DT64 verb 'on when do' schema."* Two decisions taken with it: the graph's home is the lance-graph DR#3 graph, and the binding key is the op signature → one immutable classid. Sections 1-13 below are kept as the record of the recipe probe; their measurements still hold, since the same native functions run.

**Shape** (`crates/tract-harvest-probe`):

| layer | file | what it holds |
|---|---|---|
| source | `source.rs`, `value.rs` | raw protobuf: node names, domain, opset, discrete attributes, initializer and `Constant` values decoded once (bool packed into mask words) |
| facts | `dr3::tract_facts` | Tract, cold: dtype and shape per labelled value (no declutter) |
| graph | `dr3::OnnxGraph` | nodes = **Identity**, values = **Property** on one 64x64 tile; `on`, `do_` as `Rows`; `dep = do_ ∘ read_by` by `dr_math_binding_probe::relcomp::compose`; ordered args beside the relations |
| registry | `registry.rs` | `Registration { op signature (domain, op_type, dtype, attribute rules, opsets), WHEN: K3 guard, DO: polyfill, Numeric }`, versioned; one ONNX lens classid `ONNX_LENS` (0x0F00); each op signature is a field under it (§0.7, §0.8) |
| plan | `plan.rs` | resolve once: signature lookup (unregistered = UNKNOWN, refused), opset, WHEN must be TRUE, schedule from `dep`; sealed in a `FrameStore` frame (source digest, registry version, coordinates); `run` checks the frame, then routes each node to its DO |

ON × WHEN × DO per node: ON = its input Property positions; WHEN = a three-valued guard over the facts and constant operands (multidirectional broadcast, `Pow` exponent exactly 2.0, single reduction axis, scalar threshold, `Where` operand shapes); DO = an existing function, never a new kernel (`lance_graph_onnx::{add,sub,mul,div,abs_f32,sqrt_f32,greater_i32,where_i32,ReduceSum}`, `ndarray` `mean_axis`). Classids are **lab ordinals**, not OGAR-minted concepts; production would mint them in OGAR and activate through `lance-graph-contract::hotplug`.

**Results:**

| check | result |
|---|---|
| tiny `base`, `gen2`, `neg_where_nonzero` through the graph | bit-exact vs ORT literal; `Where(c, X, 1)` now **runs** (17): there is no masked-sum rewrite to refuse, `Where` and `ReduceSum` are separate methods |
| tiny `div3`, 256 seeded inputs | bit-exact on every element (`Div` keeps its field) |
| `dep` vs the source edges | equal on 3 variants; the schedule is a topological order of it, and a protobuf with the node list reversed still runs bit-exact |
| Jina v5 small layer-0 input RMSNorm (13 nodes, Constants included) | 9,259/10,240 exact, max 4 ulp; mean 9/10 exact, max 1 ulp; `ReduceMean` the only Bounded method; every row with a bit-exact mean is bit-exact on all 1,024 outputs |
| full Jina graph (7,620 nodes, inputs pinned `[1,10]`), registry v1 coverage | **3,751** WHEN-true, **169** registered but WHEN not true, **3,700** unregistered. All 113 RMSNorm `Pow`/`ReduceMean`/`Sqrt` resolve. The 169 are facts Tract does not keep: its typed stage labels only 2,546 of 7,620 source outputs, so those shapes are UNKNOWN and the guard refuses them. The largest unregistered groups: `Unsqueeze` 888, `Shape` 430, `Gather` 373, `Concat` 284, `MatMul` 253, `Reshape` 228, `Mul` (non-f32 / other signatures) 228 |

**Refusals, each with its own test:**

| input | refusal |
|---|---|
| `B[4]` vs `F[2,3]` | `When { n_add, Add field, False }` (broadcast guard over the facts) |
| `Erf` | `Unregistered { n_abs, "ai.onnx:Erf(Some(F32)){}" }` |
| `com.example:Abs` (Tract imports it as `Abs`) | `Unregistered`, signature `com.example:Abs…`: the source domain keys the signature |
| `Pow(x, 2.0000002)` | `When { Pow, Pow field, False }` |
| `Cast_1` to FLOAT16, `Cast` to DOUBLE | `Unregistered`, signature names `to=10` / `to=11` |
| Tract's shapes removed | `When { n_where, Unknown }`: UNKNOWN never counts as TRUE |
| store advanced to `gen2`, or to a changed Jina weight | `Frame(SourceChanged)` at run time |
| registry version changed | `Frame(MapChanged)` |
| sum past int32 | `Native(OutOfDtypeRange)` at run time |

`ReduceMean` over axis -2 instead of -1 is a different valid program, not a refusal: nothing assumes the RMSNorm pattern, so the graph runs it with the mean shaped `[1,1,1024]`.

**Disable runs** (committed first, anchor asserted, `--no-fail-fast`; each went red on exactly the named test, tree restored):

| guard disabled | red test |
|---|---|
| broadcast WHEN → TRUE | `incompatible_broadcast_is_when_false` |
| `Pow` exponent WHEN → TRUE | `pow_exponent_other_than_two_is_when_false` |
| attribute rules ignored | `cast_to_half_is_unregistered`, `cast_to_double_is_unregistered` |
| domain dropped from the signature | `foreign_domain_is_unregistered_although_tract_accepts_it` |
| frame check skipped in `run` | `stale_generation_is_refused`, `registry_change_is_a_frame_change`, `new_weight_generation_refuses_the_old_plan` |
| UNKNOWN accepted as TRUE | `unknown_shapes_refuse` |
| schedule = source order | `schedule_comes_from_dep_not_source_order` |

**Cost** (`examples/cost.rs`, release, reported, not gated): tiny cold Tract facts 140 µs, tile + resolve 7.4 µs; hot plan 3.4 µs (54 allocs) vs Tract typed 3.6 µs. Jina slice cold facts 175 µs, tile + resolve 11.9 µs; hot plan 14.9 µs (62 allocs, 263 KB: every value materialized, the plan clones its constants per run) vs Tract typed 113 µs.

**OPEN (graph design):**
- More than 64 nodes or values needs several DR64 tiles; the full graph is only censused, not tiled.
- ~~Facts for the 5,074 values Tract's typed stage leaves unlabelled~~ CLOSED 2026-10-30, see §0.1.
- ~~Registry entries for the shape-plumbing ops~~ registered 2026-10-30, see §0.2. Still open: the admitted f32 `MatMul`, and int64 shape arithmetic (§0.2).
- The plan clones constants per run; hot allocation is not yet minimal.
- Classids are lab-local; minting through OGAR is a separate decision.

### 0.1 Facts from Tract's inference stage (2026-10-30)

`dr3::FactSources` reads facts from both Tract stages and keeps them apart. The **inference** stage (the inference model after `analyse`) keeps every source label. A fact there may be partial, so a shape is taken only when its rank and every dimension are determined; an undetermined dimension never becomes a symbol, because two of them need not be equal. The **typed** stage is complete per outlet but labels only the values that survive typing. `merged()` prefers the inference fact and fills a missing dtype or shape from the typed one; `disagreements()` compares the two wherever both describe a value. If typing fails, the inference facts are kept and the typed error is recorded (`typed_error`). Tract refuses to type a graph with an undetermined input dimension, but its inference stage still runs.

| Jina, inputs pinned `[1,10]` | inference | typed | merged |
|---|---|---|---|
| source values with a full fact (dtype and shape), of 7,620 | **7,620** | 2,546 | 7,620 |
| disagreements where both stages speak | 0 (all 2,546 agree) | | |
| registry v1 coverage [WHEN-true, WHEN-not-true, unregistered] | 4,146 / 0 / 3,474 | 3,751 / 169 / 3,700 | **4,146 / 0 / 3,474** |

The 169 refusals came entirely from missing facts. A further 226 nodes became registered once their dtype was known: `Mul` 452 WHEN-true (was 282), `Add` 253 (was 141), `Div` 113 (was 1). What remains unregistered is the op groups themselves: `Unsqueeze` 888, `Shape` 430, `Gather` 373, `Concat` 284, `MatMul` 253, `Reshape` 228, `Transpose` 141, `Slice` 140, `Cast` (other targets) 119, `Mul` (non-f32) 115, `Where` 114.

Tests: `inference_stage_facts_cover_every_value_and_agree_with_typed` (tiny), `full_graph_facts_and_coverage` (Jina, counts pinned exactly), `undetermined_input_dim_gives_no_shape_and_typing_failure_is_recorded`, `dr3::tests::fact_sources_disagree_and_merge`. `unknown_shapes_refuse` re-pinned: `T` now has a Tract fact (the typed stage had merged it with `Z`), so `n_greater` is the first UNKNOWN. Disable runs, each red on the named tests: merge from typed only; undetermined dim as a symbol; typed failure propagated; `disagreements` always empty.

### 0.2 Cold shape folding (D-TRACT-SHAPE-1, 2026-10-30)

**Classification.** `fold::classify` gives every value a dependency class: `Const` (source constants only), `Shape` (constants and fully known input shapes), `Symbolic` (a shape with a symbolic dimension) or `Data` (an input value, or a shape the facts do not determine). A `Shape` node takes the class of its input's shape, every other node the strongest class of its inputs. Jina, inputs pinned `[1,10]`:

| op | Const | Shape | Symbolic | Data |
|---|---|---|---|---|
| Unsqueeze (888) | 0 | 830 | 0 | 58 |
| Shape (430) | 0 | 430 | 0 | 0 |
| Gather (373) | 0 | 371 | 0 | 2 |
| Concat (284) | 0 | 228 | 0 | 56 |
| Reshape (228) | 0 | 58 | 0 | 170 |
| all 7,620 nodes | 2,760 | 2,399 | 0 | 2,461 |

Unpinned, Tract's `analyse` fails at `/Flatten` (two symbols), so the Symbolic class is shown on the `shape_sym` fixture instead.

**Folding.** `fold::fold` executes a node at resolution only when its class is `Const` or `Shape` and either every input is a known constant or it is `Shape` over a fully known shape. It uses the node's own registration: lookup, WHEN must be TRUE, then the same DO a run would call. `plan::resolve` folds before binding, writes the folded values into the tile's constants (so later guards read them) and keeps folded nodes in `Plan::folded`, not in the steps. A run whose inputs differ from the resolved facts is refused before any step (`check_against_fact`), so a folded shape is never applied to an input it was not computed for. Symbolic and Data nodes stay dynamic.

**Registry v2.** Fields: `Shape` F32/I64/I32/Bool (`start`/`end` clamped), `Gather` on int64 data (`ndarray` `select`; constant indices in `[-d, d)`, else FALSE; run-time indices UNKNOWN), `Unsqueeze` I64/F32, `Concat` I64/F32 (`ndarray::concatenate`; equal non-axis dims, two different symbols UNKNOWN), `Reshape` I64/F32 (`allowzero = 0`, constant target, known input shape), `Cast` i64→i64 identity, `Erf` (`ndarray::hpc::vml::vserf`, Bounded ≤ 2 ULP on [-5, 5]). Scalar-Y `Where` now calls `ndarray::simd::blend_i32_scalar` (one allocation and 28 B less per call on fixture A; no new kernel). `AttrRule.value: None` admits any integer attribute (an axis) read by the DO; a fixed value still keys a separate signature.

**Fact identity.** A Tract fact is admitted for a source value only if exactly one outlet of the stage carries the label and that outlet is the value's source producer by Tract's own naming rule (`dr3::source_producers`, `onnx/src/model.rs:122,150`); a matching label alone is not authority (`dr3::admit`). Jina: inference 7,937 admitted, 0 rejected; typed 1,988 of 2,546 full facts admitted, 558 rejected (557 from an outlet Tract renamed during typing, 1 label on two outlets). Coverage is unaffected because the inference stage covers every value. Tract's `TDim` (the type its `Shape` emits) is read as int64, the ONNX type. A Tract panic during `analyse` or typing is a frontend refusal (Tract indexes out of bounds on a constant out-of-range `Gather` index, `core/src/ops/array/gather.rs:75`).

**Coverage, Jina pinned `[1,10]`, [WHEN-true, WHEN-not-true, unregistered]:** registry v1 (§0.1) 4,146 / 0 / 3,474 → registry v2 with folding **6,404 / 56 / 1,160**. 4,440 nodes fold in 7.3 ms (facts 221 ms); 1,972 folded values are still read by non-folded nodes (~20 KB). Per op: Unsqueeze 888/0/0, Shape 430/0/0, Gather 371/0/2 (f32 embedding and bool data), Concat 284/0/0, Reshape 170/56/2 (56 UNKNOWN: target computed by unregistered int64 arithmetic; 2 bool data). Every registered shape-class node that does not fold is blocked by an upstream unregistered node, none by its own WHEN: `Unsqueeze` 181, `Cast` 117, `Concat` 57, `Mul` 2. The unregistered shape-class nodes are int64 arithmetic: `Mul` 115, `ConstantOfShape` 59, `Equal` 58, `Where` 58, `Div` 56, `Range` 4, `Expand` 2, `Add` 1, `LessOrEqual` 1, `And` 1, plus `Cast` to bool/float 2, `Cos`/`Sin`/`MatMul`/`Transpose` 1 each.

**Fixtures and oracle.** New in `tools/gen_tiny.py`, frozen ORT 1.29.0 `ORT_DISABLE_ALL`: `shape_fold` (the chain folds; `Reshape(F)` and `Unsqueeze(F)` run bit-exact), `shape_sym` (nothing folds; Reshape WHEN UNKNOWN, refused), `neg_gather_index` (ORT fails the run, Tract panics, the frontend refuses), `neg_concat` (ORT and Tract both reject), `erf` (fixture A with Erf; ≤ 2 ULP vs ORT), `erf_sweep` (4,115 points: dense [-6, 6], signed zeros, infinities, NaN, subnormals; max 2 ULP vs `math.erf` rounded to f32 and vs ORT; specials exact). `neg_unsupported` now uses `Acosh`.

**Cost (reported, `examples/cost.rs`, AVX-512 host, `target-cpu=native`):** `shape_fold` cold tile+fold+resolve 12.4 µs (2 steps, 4 folded), hot 1.7 µs, 37 allocations; `erf` 4,096 elements hot 8.5 µs vs Tract 2.0 µs (`vserf` alone 7.9 µs, 1.9 ns/element); fixture A hot 3.3 µs, 55 allocations (56 before scalar-Y Where); Jina slice tile+fold+resolve 20 µs (9 steps, 4 folded Constants).

Tests (40, all green): `shape_chain_folds_and_the_data_nodes_run_against_ort`, `folded_shapes_refuse_a_run_with_other_input_dims`, `a_node_reading_input_values_never_folds`, `a_symbolic_shape_is_not_folded_and_unknown_is_not_true`, `an_out_of_range_gather_index_is_refused_not_a_panic`, `a_concat_dimension_mismatch_is_refused_by_the_frontend`, `erf_in_the_graph_against_ort`, `erf_sweep_against_f64_and_ort`, and unit tests `a_label_alone_does_not_admit_a_fact`, `source_producers_follow_tract_naming`, `concat_never_aliases_two_symbols`, `gather_index_range_is_three_valued`, `reshape_target_rules`, `attr_rules_any_and_fixed`, `shape_start_end_clamp`. Re-pinned: `full_graph_facts_and_coverage` (typed 1,988 admitted, 558 rejected, coverage above), the RMSNorm slice (9 steps + 4 folded Constants), `unregistered_op_is_refused` (Acosh). Disable runs, each red on the named test: no folding; label as authority; two symbols aliased; Reshape UNKNOWN as TRUE; symbolic shape classed as known; no stale-input check; Gather range always TRUE; no panic catch; Shape `end` ignored. Not load-bearing (documented in the code): the fold's dependency-class skip, since a Symbolic or Data node never has every input known.

**OPEN (owner decisions):**
- int64 shape arithmetic (`Mul`, `Div`, `Add`, `Equal`, `Where`, `Range`) has no ndarray word; its overflow and division semantics (wrap or refuse; truncation; division by zero) need a decision before an ndarray ask.
- An axis attribute is a DO parameter under one field per (op, dtype); the alternative is a field per axis value.
- `Erf` is Bounded at 2 ULP; `vserf` is about 4x slower than Tract's `erf` on this host.
- `ConstantOfShape` and `Expand` map onto ndarray `from_elem` / `broadcast` and are not registered yet.
- Multi-tile orchestration and full Jina MatMul execution remain separate gates.

### 0.3 After #400 / #1853 (2026-10-30)

Owner decisions: int64 shape arithmetic GO under checked, fail-closed admission (Div truncates toward zero; a zero divisor or an overflow refuses); an axis stays a guarded parameter of one method unless a different axis changes the behaviour law; ConstantOfShape and Expand registered on existing ndarray capabilities; Erf numerically accepted as Bounded, its performance open.

**Registry v3.** int64 `Equal` and `LessOrEqual` (ndarray #400 `eq/le_i64_pair_to_mask`, scalar right side `eq/le_i64_to_mask`; a same-shape or one-element right side only, other broadcasts not admitted), int64 `Where` (`blend_i64` / `blend_i64_scalar`), `ConstantOfShape` (`ArrayD::from_elem` of the value attribute, constant non-negative shape), `Expand` int64/f32 (`ndarray` `broadcast` to the multidirectional broadcast). Fixture `shape_i64`: every int64 node over `Shape(F)` folds, `Expand(F)` runs, every output equals ORT.

**Optional output slots.** Tract numbers a node's outlets after dropping empty output names (`onnx/src/model.rs:150`). Fixtures `lstm_skip` (`LSTM` outputs `["", "Y_h"]`) and `lstm_full` pin that `source_producers` names slot 0 resp. 1 for `Y_h` and that the admitted fact is `Y_h`'s (ORT `[1,1,2]`), not the omitted `Y`'s (`[5,1,1,2]`). Disable run (count empty slots): red.

**Coverage, Jina pinned `[1,10]`:** 6,404 / 56 / 1,160 → **6,581 / 114 / 925**; 4,499 nodes fold. The 114 not-true: Reshape 56 and Expand 57 UNKNOWN (target computed by unregistered int64 arithmetic), LessOrEqual 1 not admitted (a `[S,1]` vs `[1,S]` broadcast). Unregistered shape-class nodes left: Mul 115, Div 56, Range 4, Add 1, plus Cast to bool/float 2, And, Cos, Sin, MatMul, Transpose 1 each.

**ndarray ask (D-TRACT-SHAPE-1, after #400).** #400 already supplies the int64 compares (`eq/lt/le/gt/ge_i64[_pair]_to_mask`) and `blend_i64[_scalar]`; they are registered. What is missing, by exact signature (fail-closed: `Err` and an unchanged `dst` content contract is not required, but no partial result may be returned as success):

```rust
pub enum IntArithError { Overflow { index: usize }, DivideByZero { index: usize }, ZeroStep }
pub fn add_i64_checked(a: &[i64], b: &[i64], dst: &mut [i64]) -> Result<(), IntArithError>;
pub fn sub_i64_checked(a: &[i64], b: &[i64], dst: &mut [i64]) -> Result<(), IntArithError>;
pub fn mul_i64_checked(a: &[i64], b: &[i64], dst: &mut [i64]) -> Result<(), IntArithError>;
/// Truncates toward zero; `i64::MIN / -1` is Overflow.
pub fn div_i64_trunc_checked(a: &[i64], b: &[i64], dst: &mut [i64]) -> Result<(), IntArithError>;
// one-element right side, as blend_i64_scalar
pub fn add_i64_scalar_checked(a: &[i64], b: i64, dst: &mut [i64]) -> Result<(), IntArithError>;
pub fn sub_i64_scalar_checked(a: &[i64], b: i64, dst: &mut [i64]) -> Result<(), IntArithError>;
pub fn mul_i64_scalar_checked(a: &[i64], b: i64, dst: &mut [i64]) -> Result<(), IntArithError>;
pub fn div_i64_trunc_scalar_checked(a: &[i64], b: i64, dst: &mut [i64]) -> Result<(), IntArithError>;
/// ONNX Range: max(ceil((limit - start) / delta), 0) elements; delta 0 is ZeroStep.
pub fn range_i64_checked(start: i64, limit: i64, delta: i64) -> Result<Vec<i64>, IntArithError>;
```

Accuracy: exact (integers). Panics only on a length mismatch, as the #400 kernels. Fixtures: `crates/tract-harvest-probe/fixtures/tiny/i64_arith.onnx` with ORT `ORT_DISABLE_ALL` outputs in `ort.json` (`Div([-7, 7], [2, 3]) = [-3, 2]`, `Mul([2, 3], [3, -4]) = [6, -12]`, `Add([2, 3], [-5, 10]) = [-3, 13]`, `Sub = [7, -7]`, `Range(0, 3, 1) = [0, 1, 2]`); refusal cases ORT does not cover: `add(i64::MAX, 1)`, `mul(i64::MIN, -1)`, `div(i64::MIN, -1)`, `div(x, 0)`, `range(0, 5, 0)` each `Err`; `div(-7, 2) = -3`, `div(7, -2) = -3`, `range(5, 0, -2) = [5, 3, 1]`, `range(0, 0, 1) = []`. Test `int64_arithmetic_is_refused_until_checked_words_exist` pins the current refusal.

Disable runs (each red): Equal kernel replaced by `le`; slot counted over empty outputs; general broadcast admitted; broadcast mismatch accepted; ConstantOfShape ignoring its value. 46 tests green.

**Hardening (same PR).** `source_producers` enumerates each node's ONNX output positions first and maps a position to Tract's slot with `dr3::tract_slot` (named outputs before it). `FactSources` records the source it was read from; `merged_for` refuses facts of another model. Tests: `["", "Y"]` and multi-output mapping; producer collisions (two producers, a value that is also an input or initializer, duplicate node names, unnamed node without a first output); a fact from a renamed producer; facts of another model; int64 compares and Where over 70 elements (tail bits clear, scalar and pair); ConstantOfShape (default float 0, empty shape gives a scalar, zero dims, bool fill, 1-D shape input, K3 refusals) and Expand (multidirectional broadcast). Disable runs (each red): ONNX position used as the slot; no binding check; collisions keep a producer; bool fill tail bits; ConstantOfShape rank. 55 tests green.

**Census (Jina, `[1,10]`, `examples/shape_census.rs`).** WHEN-true / not-true / unregistered **6,581 / 114 / 925** (registry v1 4,146 / 0 / 3,474; v2 6,404 / 56 / 1,160). Cold-folded nodes 4,499 (v2 4,440, +59 ConstantOfShape; v1 plan had no folding). **The 56 Reshapes blocked in v2 still do not resolve: the first unregistered op their target depends on is int64 `Mul` in all 56.** Not folded: runtime 2,461 (Mul 450, Add 253, MatMul 252, Cast 228, Reshape 170, Slice 140, Transpose 140, Div/Pow/ReduceMean/Sqrt 113 each, …) and shape-only blocked 660 (Unsqueeze 181, Cast 119, Mul 117, Equal 58, Where 58, Concat 57, Div 56, Range 4, …). The integration acceptance (a whole shape-arithmetic chain folding or executing through admitted methods and matching ORT) waits on the checked int64 words; `i64_arith` is its fixture. No full Jina runtime is started.

### 0.4 Checked int64 arithmetic on ndarray #401 (2026-10-30)

ndarray #401 shipped `ndarray::hpc::checked_i64` (consumed as built; ndarray is another session's). Registry v4 registers int64 `Add`/`Sub`/`Mul`/`Div` on `checked_binary_i64` and `Range` on `range_i64`, bounded by `RANGE_CAPACITY = 65,536` (a policy pin). Guards: equal known shapes, or a one-element side whose rank is at most the other's; anything else FALSE, unknown shapes UNKNOWN. `Range` needs three one-element inputs; a constant zero step is FALSE. `Div` truncates toward zero; overflow, a zero divisor, a zero step and a range over capacity refuse with the ndarray error text.

The fold records a DO refusal (`Folded::refused`) instead of treating it as "not folded", and `plan::resolve` returns it, so invalid constant arithmetic refuses the plan.

**Integration acceptance.** `int64_arithmetic_folds_and_matches_ort`: the whole `i64_arith` chain (Shape → Div/Mul/Add/Sub, Gather → Range) folds through admitted methods with no runtime step and matches ORT `ORT_DISABLE_ALL` exactly (`Div([-7, 7], [2, 3]) = [-3, 2]`). `i64_data`: int64 `Mul` on a run-time input stays a step, matches ORT, and a run with `i64::MAX` refuses with an overflow.

**Refusals, and where each happens.** Constant overflow (`neg_i64_overflow`; ORT wraps it to `i64::MIN`) refuses at resolution through our DO. An unadmitted broadcast (`[2]` against `[2, 1]`, `neg_i64_bcast`) is WHEN FALSE. A constant zero divisor (`neg_i64_div0`; ORT is not run, integer division by zero is undefined there) and a zero `Range` step (`neg_range_zero`; ORT refuses too) are already refused by Tract's analysis, the second as a panic that `no_panic` turns into a frontend refusal. Our DOs' own refusals for both are pinned by `checked_i64_dos_refuse`.

**Jina census (`[1,10]`).** WHEN-true / not-true / unregistered **6,871 / 1 / 748** (v3 6,581 / 114 / 925); typed-facts coverage 3,954 / 227 / 3,439 (was 3,841 / 226 / 3,553). Cold-folded nodes 5,139 (v3 4,499); retained folded values 1,446 (~17.5 KB). **All 58 shape-dependent Reshapes fold; the 56 that v3 blocked on int64 `Mul` resolve.** Expand: 60 nodes, 58 of them read a run-time value (Data class, never foldable), 1 folds, 1 is blocked (below). No full Jina runtime was started.

`fold` and `coverage` now give an initializer without an admitted fact its protobuf dtype and dims (`fold::with_initializer_facts`), the rule `OnnxGraph::build` already applied: Jina's `_one` is carried by two Tract outlets and rejected, which left `_pos_range`'s Range UNKNOWN in the census though a plan would have folded it.

**Remaining shape-only blocked nodes: 20, each traced to its first unresolved producer or guard** (`examples/shape_census.rs`):

| first unresolved | blocks |
|---|---|
| `/rotary_emb/Cos`, `/rotary_emb/Sin` (unregistered f32 Cos, Sin) | Cos, Sin, 2 Mul, 2 Unsqueeze, 2 Cast |
| `/rotary_emb/MatMul` (unregistered f32 MatMul; the ndarray word exists) | MatMul |
| `/rotary_emb/Transpose` (unregistered) | Transpose, Concat |
| `/rotary_emb/Cast_1`, `/Cast_4` (unregistered Cast signatures) | 3 Cast |
| `Constant_2836` (rotary `inv_freq`, 64 f32 stored in the external weight file, not loaded here) | 2 Cast, 1 Expand |
| `/And` (unregistered bool And) | And |
| `/LessOrEqual` WHEN FALSE (multidirectional broadcast of two Unsqueezes; the guard admits pair or scalar only) | LessOrEqual |

The rotary-embedding chain is shape-only (its angles depend on positions, not token values), so it folds once Cos, Sin, MatMul, Transpose and the Cast signatures are registered and its constant is available. Runtime nodes not folded: 2,461, unchanged.

**Cost (cold, reported, not gated).** Census fold over the whole graph 7,832 µs. `examples/cost.rs`: `i64_arith` (7 nodes folded, 0 steps) cold 20.6 + 343.1 + 29.5 µs (source, Tract facts, tile+fold+resolve); a plan run still costs 2,114 ns against Tract's 723 ns, because the run clones the 7 folded outputs (38 allocations). `i64_data` (one run-time Mul): plan 954 ns, Tract 1,219 ns. Erf (4,096 values) plan 4,181 ns, Tract 1,930 ns, on ndarray `5c6d98b0` (after #402); no A/B against pre-#402 was taken.

Disable runs (each red): DO refusal not surfaced by `resolve`; binary guard always TRUE; overflow ignored; zero-step guard removed; range capacity widened; one-element left side not broadcast; initializer facts not filled in `fold`. 60 tests green (21 lib, 31 tiny, 8 Jina).

### 0.5 The rotary chain (2026-10-30)

Registry v5 registers what §0.4 listed as the remaining shape-only blockers, each on an ndarray word consumed as built:

| op | DO | numerics |
|---|---|---|
| Cast int64 → f32 | `v as f32` (round to nearest, as ONNX) | exact |
| Cast bool → bool | identity | exact |
| MatMul f32 | `matmul_plan` + `ndarray::simd::matmul_f32_batched` | bounded (sgemm accumulation order; exact when K = 1) |
| Transpose f32 / int64 | `permuted_axes` + `as_standard_layout` | exact |
| Cos / Sin | `ndarray::hpc::vml::vscos` / `vssin` | bounded |
| And (bool) | `mask_and` after broadcasting the bit words | exact |

MatMul admits a rank-2 right operand (the left's leading dims flatten into rows), equal leading dims, or all-ones right leading dims; anything else is WHEN FALSE. Equal and LessOrEqual now admit any multidirectional broadcast (both sides broadcast with ndarray, then the pair kernel); the earlier pair-or-scalar rule was the census's one WHEN-false node. `attrs_match` accepts an integer list for a rule without a pinned value (Transpose's `perm`).

**External constant.** The rotary `inv_freq` (`Constant_2836`, 64 f32) lives in the external weight file. `tools/fetch_jina_constants.py` fetches only external-data Constants of at most 64 KiB as HTTP ranges into `fixtures/jina/` (gitignored); `fixtures/jina_constants.json` commits offset, length, dims and SHA-256. `SourceInventory::inline_constants_from_manifest` refuses a wrong digest or length, and `inline_external_constant` refuses a target that is not exactly one undecoded default-domain Constant. The same script extracts the 40-node rotary subgraph with the model's own opset and runs it under ORT `ORT_DISABLE_ALL` for `input_ids` [1, 10] (outputs and digests in the manifest).

**Acceptance.**
- `rotary_chain_folds_and_matches_ort` (tiny `rope` variant, synthetic inv_freq): the chain folds with no run-time step through the v5 fields; the causal mask matches ORT exactly (55 set bits); cos and sin are within 1 ULP.
- `trig_sweep_against_f64_and_ort` (41,579 points: the rotary angles, a grid over ±4π, special values): `vscos`/`vssin` are ≤ 1 ULP from f64 and ≤ 2 ULP from ORT, which is itself 2 ULP from f64 at points.
- `matmul_f32_against_ort` (a [1,4,64] × [64,8] weight and a [1,2,4,16] × [1,2,16,4] attention shape): 0 ULP measured, pinned ≤ 2.
- `jina_rotary_fold_matches_ort` (real Jina, inv_freq inlined): both folded rotary outputs match ORT within 1 ULP over 1,280 values each.

**Jina census (`[1,10]`, inv_freq inlined).** WHEN-true / not-true / unregistered **7,274 / 0 / 346** (v4 6,871 / 1 / 748); typed-facts coverage 4,238 / 256 / 3,126. Cold-folded nodes 5,159 (v4 5,139); retained folded values 1,440 (~38 KB). **Shape-only blocked nodes: 0** (v4: 20). The 346 unregistered nodes are run-time ops (Slice, Neg, IsNaN, Sigmoid, Softmax, GatherElements, f16 casts and similar); the 252 run-time MatMuls are now admitted (196 rank-3 × weight, 56 rank-4 attention). Not folded: 2,461 run-time nodes, unchanged. No full Jina runtime was started.

**Findings for the ndarray owner (reported, not changed here).** `ndarray::simd`'s documentation says `matmul_f32` uses "BF16 compute on AMX hosts", but `matmul_f32_batched` measured as exact f32 sgemm (0 ULP against ORT). `hpc::amx_matmul` is compiled on x86_64 only.

**Cost (cold, reported, not gated).** `examples/cost.rs`: `i64_arith` cold 20.6 + 350.2 + 30.2 µs, hot plan 2,150 ns against Tract's 712 ns (the run still clones the 7 folded outputs); Jina RMSNorm slice plan 14.5 µs against Tract 113.6 µs. No rotary cost row was taken.

Disable runs (each red): Transpose `perm` ignored; Cos computed with sin; compare broadcast removed; And broadcast ignored; manifest digest check removed; MatMul weight-row flattening dropped. The last one stayed green until the weight case was given two batches (with one batch the flattening is a no-op). 70 tests green (23 lib, 35 tiny, 9 Jina; the Jina tests need the fetch scripts).

### 0.6 ONNX as a consumer of the DR64 route / DT64 bind store (2026-10-30)

D-DT64-6 §1 records DR64 as direct routing (position → bound target) and DT64 as the DTO bind store (classid → behaviour). The DT64 drafts on EXO-50 (-06 to -09) resolve `address → classid → method`. `Registry::lookup` does both halves in one step. `src/bind.rs` separates them without changing the registry:

- `bind::route(src, facts, registry)` is the binder: the signature match runs once per node and only `node position → MethodKey { lens, ordinal }` is kept, sparse. `Routes::version` digests the registry version and every entry, so a re-route is a new version.
- `bind::resolve(routes, registry, node)` is the bind-store side: the field is found by its key alone (`Registry::get`), and the failed link is named (`Unresolved::Route` for an unrouted position, which is UNKNOWN; `Unresolved::Lens` for a foreign lens; `Unresolved::Item` for a key the lens does not hold or does not expose).

This works because every field has one position under the one ONNX lens, so a key names one field.

**Acceptance.**
- `route_then_resolve_equals_signature_lookup`: for every node of 15 tiny fixtures, resolve returns the same registration (pointer-equal) as `lookup`; unregistered nodes resolve as `Route`.
- `bind_store_names_the_failed_link_and_versions_routes`: a key the lens does not hold resolves as `Item`, a foreign lens as `Lens`; re-routing against it drops the route and moves the version; a registry version bump moves the version.
- `full_graph_facts_and_coverage` (Jina, [1,10], inv_freq inlined): **7,274 routed and resolved, 346 unrouted**, equal to the coverage census.
- Disable runs (each red): key match loosened; registry version dropped from the route digest.

**Finding.** `OnnxGraph::build` refuses a graph over one 64-node tile (`TileOverflow`), and Jina has 7,620 nodes. The binder therefore reads per-value facts, not the one-tile graph; a node position is the source node index. Mapping that index onto DR64 tile coordinates (and multi-tile execution) is DR's, not Tract's.

**Not done here:** the DT64 crate, the DR route table, `K3` re-homing, OGAR classids. When those exist, `route` writes DR's table and `resolve` becomes DT64's; the equality test is the migration gate (G1 in the -06 draft, with the census now 7,274 / 0 / 346).

### 0.7 CORRECTION: a classid is a codebook lens, not a codebook item (2026-10-30)

**Superseded.** Every per-op-signature classid (0x0F01…0x0F82, and "13 lab classids" in the EXO-50 messages) was a mistake. They have been removed from §0.1–§0.6; ONNX has one classid, `ONNX_LENS`. The registry spent one classid per codebook item. It was this session's assumption: a binding question offered "op signature → one immutable classid" as the recommended option, and the module doc then attributed it to a "user decision" the user never made.

**Now (registry version 6).**
- The ONNX operator codebook is read through one lens, `registry::ONNX_LENS` (a lab classid, not minted).
- Each op signature is an item, addressed as `MethodKey { lens, ordinal }`. The ordinal is the item's position in `Registry::regs`, derived and never allocated; a reorder is a new registry version.
- `Bound`, `Refusal::When` and `Plan::census` carry the key. `Registry::key_of` and `Registry::get` translate; `get` refuses a foreign lens.
- `bind::Routes` maps `node position → MethodKey`. `bind::resolve` names the failed link as `Route`, `Lens` or `Item`.

**Tests.** `one_lens_many_items` (every item under the one lens, ordinal = position, a foreign lens reads nothing). The bind-store test now covers the `Lens` and `Item` failures. Results are unchanged: tiny and Jina suites green, Jina 7,274 routed and resolved, 346 unrouted. Disable runs red: lens check removed from `Registry::get`; lens check removed from `resolve`.

### 0.8 The registry is a field view over lens-agnostic algebra (2026-10-30)

The user said the classid + `WideFieldMask` ergonomics is the Redmine ERB field-view pattern. The table below is this session's reading of that, and registry version 7 implements the reading. Redmine's own field formats (string, list, date) are agnostic. A tracker's custom fields label a format and add their own validation. The tracker enables a subset of those fields.

| Redmine | registry v7 |
|---|---|
| field format (agnostic) | `Registry::methods`: one `Method { name, do_, numeric }` per algebra address, by position, no label, no lens |
| custom field (label + validation) | `Registry::regs`: a `Registration` labels one address (`method: u16`) with an op signature and its WHEN; its position is the ordinal of `MethodKey { lens, ordinal }` |
| fields enabled for a tracker | `Registry::exposed`, a `lance-graph-contract` `WideFieldMask` over field positions |

- `Registry::build(lens, version, drafts)` interns equal addresses by name. It panics if one name has two bodies. 48 fields label fewer addresses; the three `Constant` fields share one.
- A field outside `exposed` is absent: `lookup` does not match it, `get` and `bind::resolve` do not return it (`Unresolved::Item`), and its node is UNREGISTERED. Other routes resolve unchanged.
- `Registry::revision()` hashes the version and the exposed positions. It is the frame's `relation_map` revision and seeds the route version, so hiding a field is a new revision.

**Tests.** `fields_label_shared_algebra_addresses`, `a_masked_field_is_absent`; all earlier tests unchanged. Jina still 7,274 routed and resolved, 346 unrouted. Disable runs red: mask ignored in `lookup`; mask ignored in `get`; mask left out of `revision`; interning removed.

**Open.** The WHEN guards are on the fields. Some are only meaningful for one address, for example a Reshape guard over its target shape. Whether a guard belongs with its field or its address is not settled here. `exposed` is set to full by `build`, and no production lens is minted.

### 0.9 DR64 is global to local; the codebook spans mask blocks (2026-10-30)

The user: "the codebook should not be limited to 64 if Jina asks for more" and "DR64 is global to local, not a rejection to access 64k soa". `OnnxGraph::build` refused any graph beyond one 64 × 64 tile (`TileOverflow`). That was wrong.

- **Graph.** A global position `p` is bit `p % 64` of tile `p / 64`. `on` and `do_` are `dr3::Tiled` relations: their non-empty DR64 tiles, keyed `(row tile, column tile)`. `dep = do_ ∘ read_by` composes tile by tile with the same `relcomp::compose` kernel, and `schedule` reads `dep`'s pairs. The only refusal left is the u16 address width (`Refusal::AddressOverflow`, 65,536 positions per axis).
- **Jina, measured.** The full graph (7,620 nodes, 7,939 values) builds; its tiled `dep` (391 tiles, 9,176 edges) equals the source edges; it schedules, and every edge runs forward.
- **Codebook.** One `WideFieldMask` position is a `u8`, so the exposure mask stopped the lens at 256 fields. `Registry::exposed` is now a list of masks: field `o` is position `o % 256` of block `o / 256`, up to 65,536 fields. `Registry::expose` shows or hides one field.

**Tests.** `graphs_beyond_one_tile_lay_over_tiles` (200-node chain over four Identity tiles, with a cross-tile edge), `codebook_spans_mask_blocks` (600 fields over three blocks), and the full-graph Jina assertions. Disable runs, each red: compose restricted to the diagonal column tile; `exposes` reading block 0 only; `set` dropping cross-tile bits.

**Not yet.** `plan::resolve` still refuses Jina at its first unregistered node (346 remain); tiling removes the size refusal, not the coverage gap.

### 0.10 Triage, codegen, and the last 346 Jina nodes (2026-10-30)

**Coverage (measured).** ONNX 1.23.2 has 203 default-domain ops. Tract `bf7ca16` registers 182 op names: 163 standard plus 19 contrib/ML. 40 standard ops are not in Tract (sequence, optional and string types, training losses, `Loop`, some detection ops). No Cargo.toml in Tract names ort, onnxruntime, candle, burn, libtorch or tch, so Tract has no runtime fallback. Before this section the probe had 29 op types; Jina at `[1, 10]` left 346 nodes unbound. Map: board entry `2026-10-30-d-tract-onnx-coverage-map.md`.

**Triage, for any model.** `triage::triage` classifies each unbound node:
- the gap is dtype, attribute or op;
- the value dtype is taken the way the registry keys the op;
- the role is the node name with layer indices folded to `N`;
- it records each argument's dtype, rank and constness;
- it records whether Tract registers the op;
- it attaches the matching rows of `data/native_words.tsv`.

`tract_index` reads Tract's ONNX frontend source as text. For every `reg.insert` it gets the parser, the attributes it reads with their `unwrap_or` defaults, and whether it branches on the opset. `examples/onnx_triage` prints the report. It finds Tract and ndarray from `Cargo.lock` and checks every native word exists in that ndarray source.

**Codegen.** `examples/registry_codegen` writes `src/generated.rs` for three templates that need no judgement:
- an element-wise unary word;
- the complement of a "not X" mask word;
- an existing field at one more dtype.

It refuses an op whose Tract parser reads attributes or branches on the opset. It triages against `registry::hand_written()`, so running it again reproduces the same file. For Jina it generates Neg, IsNaN and Sigmoid f32, plus Bool rows for Reshape, Expand and Gather (116 nodes).

**Hand-written fields.** Slice (f32, bool); Where with f32 values (a scalar X blends Y under the complemented condition); Softmax over the last axis; f32 row Gather; Squeeze; Flatten; checked i64 ReduceSum; Cast i64→bool; GatherElements. Also Bool arms in `do_expand` and `do_gather_i64`. Registry version 8: 59 hand-written plus 6 generated fields.

**Measured.**
- New fixture `jina_ops`: the 14 op types, every output compared with ORT. Exact ops match bit for bit. Sigmoid is within 1 ULP and Softmax within 2 ULP of f64.
- ORT flushes `sigmoid(-20)` = 2.06e-9 to 0, so for the bounded ops ORT is reported and f64 is asserted.
- Jina at `[1, 10]`: 7,620 of 7,620 nodes registered with WHEN TRUE, all routed and resolved.
- Disable runs, each red: Where without the complement; Slice ignoring start; ReduceSum ignoring keepdims; GatherElements not replacing the axis coordinate; Softmax over axis 0; generated IsNaN without the complement. Triage: value dtype from argument 0; role keeping digits; opset branch ignored; attributes never matching.

**Not measured.** A whole-model numeric run against ORT. The local Jina file is the 1.27 MB graph; its weights are external data (CC-BY-NC, never committed) and were not fetched.

### 0.11 Op coverage across ORT, burn and candle (2026-10-30)

`data/onnx_op_coverage.tsv` is generated by `tools/onnx_op_coverage.py` from each runtime's own registry. Of the 203 ONNX 1.23.2 default-domain ops, each runtime registers by name:

| runtime | ops | source |
|---|---|---|
| ORT 1.29.0 CPU | 188 | `get_all_opkernel_def` |
| Tract bf7ca16 | 163 | `reg.insert` |
| burn (onnx-ir 0.22.0) | 163 | `ProcessorRegistry`, `unsupported::` variants excluded |
| candle-onnx 0.11.0 | 80 | `match node.op_type` arms |
| probe | 37 | |

- burn's rank-split variants (`Conv1d/2d/3d`, `MaxPool1d`, …) are folded back to their ONNX op. Its `Linear` is a fused node, and four of its variants are ONNX-ML ops.
- Tract and burn together cover every op any Rust runtime has (170); candle adds none.
- burn adds 7 Tract lacks: Loop, Det, Col2Im, DeformConv, LpPool, Bernoulli, Upsample.
- Tract adds 7 burn lacks: RMSNormalization, RotaryEmbedding, Compress, ConvInteger, DynamicQuantizeLinear, Multinomial, QLinearConv.
- 25 ops are ORT-only (sequence, optional, string, Unique, RoiAlign, …). 8 are in no runtime (SwiGLU, LinearAttention, the training losses, …).
- 133 ops are in some Rust runtime and not in the probe.

`examples/onnx_triage` now prints, per unbound op, which runtimes register it.

**Fixed.** The examples applied `fixtures/jina_constants.json` to any model under `fixtures/`; the triage crashed on a tiny fixture. `inline_matching_constants` applies a manifest only when the model has a `Constant` producing every output it lists.

**Limits.**
- Registered by name is not full support: attributes, dtypes and opset ranges are not compared.
- burn is counted at onnx-ir (parse and IR). The burn-onnx code generator is per IR node and is not cross-checked per op.

## 1. The acceptance chain, demonstrated

The chain under test: a source-attested ONNX subgraph is imported by Tract, admitted, lowered onto existing native methods, executed without Tract, and verified against ONNX Runtime (ORT). The chain holds for both fixtures: fixture A (tiny, synthetic) is bit-exact, and fixture B (a real model slice) is bounded. The numbers are in §4. Nothing here is certified, production or merged.

```text
.onnx bytes ──► source::SourceInventory      (raw protobuf: domain, opset, initializers)
     │
     └──► Tract InferenceModel ─► TypedModel  (cold; shapes, dtypes, constants)
                                    │
              admit::admit_*(source, typed) ─► Recipe + Receipt | Refusal
                                                │
                              Recipe::run ─► lance-graph-onnx / ndarray only
```

Invariants:

- Recipes are concrete structs whose `run` calls lance-graph-onnx / ndarray directly.
- No graph is read at run time.
- There is no general lowering, interpreter, tensor AST or registry.
- Tract supplies structure, shapes and constants. The raw protobuf supplies domain, opset, initializer identity and digest.
- ORT runs with graph optimization DISABLED (`ORT_DISABLE_ALL`) as the source-semantics oracle.

## 2. Baseline reconciled (what existed before this probe)

| Reference | State at the time of this probe |
|---|---|
| #1753 D-NFC | Native-first rule: third-party ONNX is an external compatibility leaf. |
| #1754 / #1758 | `lance-graph-onnx` facade: ALIAS / INLINE / bounded Exp, Tanh. Float ReduceSum is NATIVE-not-exported pending a tolerance contract. Gather and MatMul are REFUSE in the facade. |
| #1752 | One `Greater -> Where -> ReduceSum` native composite. |
| #1779 – #1783 | D-NAM specs. PROPOSED, not implemented. |
| #1802 D-ONNX-INTAKE-0 | Bake-off capstone (docs). S1–S5 NOT RUN, not authorized. |
| #1810 | Board entry nominating Tract for first harvest attention (docs). |
| #1800 / #1801 | TFLite Softmax / Codec8 lab evidence. Merged, lab only, not reused here. |
| #1835 | DR/DT ownership decisions. DR = HHTL16 Cartesian addressing/projection. DT = ClassId -> class schema + method schema. DR #4 routes method ops. Thought-Git sits above. |

Nothing was re-run from the certified TFLite TANH sweeps.

## 3. Fixtures

### 3.1 Fixture A: tiny (committed)

Location: `crates/tract-harvest-probe/fixtures/tiny`, generated by `tools/gen_tiny.py`. Generated with ORT 1.29.0, onnx 1.23.2, opset 18.

- `S = ReduceSum(Where(Greater(X, T), X, Z), axes=[0], keepdims=0)` with X int32[N] (symbolic N), T = 0, Z = 0.
- `Q = Div(Sqrt(Abs(Mul(Add(F, B), F))), B)` with F float[2,3] and B float[3] an initializer.

| Variant | Change |
|---|---|
| base | as above |
| gen2 | new B bytes, same shape |
| div3 | B = [0.5, -2, 3] |
| neg_shape | B has shape [4] |
| neg_where_nonzero | Z = 1 |
| neg_unsupported | Erf in place of Abs |
| neg_domain | `com.example:Abs` |

Plus a 256-input seeded div3 sweep (seed 20261030) run through ORT.

### 3.2 Fixture B: real model slice (weights never committed)

- Model: `jinaai/jina-embeddings-v5-text-small-retrieval` at revision `6856e76bb72982e58de0620458a4e8b3614da340`, license cc-by-nc-4.0.
- Weights are fetched by `tools/fetch_jina_slice.py` with SHA-256-pinned HTTP ranges into gitignored `fixtures/jina/`. The committed manifest `fixtures/jina_manifest.json` holds digests only.
- Architecture: Qwen3Model, 28 layers, hidden 1024, 16 heads / 8 KV heads, head_dim 128, intermediate 3072, rms_norm_eps 1e-6, rope_theta 3500000, vocab 151936, tie_word_embeddings true, pooling last-token (1_Pooling config).
- ONNX graph `onnx/model.onnx`: sha256 `463fb3fec2adf9c47a946626dc2a0884e43091f95466083fb1754950000a7c1a`, 1,266,078 bytes; producer pytorch 2.9.1; ir 8; opset 18; 7,620 nodes; 317 initializers, of which 310 are external FP32 in `model.onnx_data` (2,384,199,936 bytes).

Op histogram (top): Constant 2759, Unsqueeze 888, Mul 567, Shape 430, Gather 373, Cast 350, Concat 284, Add 254, MatMul 253, Reshape 228, Div 169, Transpose 141, Slice 140, Where 114, Pow / ReduceMean / Sqrt 113 each, Expand 60, ConstantOfShape 59, Equal 58, Neg 56, Softmax 28, IsNaN 28, Sigmoid 28, Range 4, And 2, and one each of LessOrEqual, Flatten, Cos, Sin, ReduceSum, Sub, GatherElements, Squeeze.

- Tokenizer: `tokenizer.json` sha256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
- Pinned text: "Harvest the algebra. Keep the native substrate." Token ids: `[26490, 7019, 279, 46876, 13, 13655, 279, 9867, 52957, 13]` (offsets recorded in the manifest).
- Slice: `/embed_tokens/Gather`, then `/layers.0/input_layernorm/{Cast, Pow, ReduceMean, Add, Sqrt, Div, Mul, Cast_1, Mul_1}` with its Constants (exponent 2.0, axes [-1], eps 9.999999974752427e-07, numerator 1.0). Weight `layers.0.input_layernorm.weight` [1024] at external offset 622330880, length 4096.
- The whole fetch took 7 s and 44 KB of weight bytes instead of 2.4 GB.

## 4. Results

### 4.1 Fixture A

| Check | Result |
|---|---|
| Native recipe vs ORT on base, gen2, div3 | bit-exact; numeric class Exact |
| div3 sweep, native vs ORT | bit-exact on all 256 inputs (1,536 output elements) |
| div3 sweep, Tract typed vs ORT | bit-exact on all 256 inputs |
| div3 sweep, Tract decluttered vs ORT | differs on 161 of 1,536 elements, all in the divide-by-3 column (powers of two are exact under the rewrite) |

### 4.2 Fixture B (10 x 1024 outputs), against ORT literal (`ORT_DISABLE_ALL`)

| Arm | Mean (10 values) | Output (10,240 values) |
|---|---|---|
| Native | 9/10 exact, max 1 ulp | 9,259/10,240 exact, max 4 ulp |
| Tract typed (no declutter) | 0/10 exact, max 31 ulp | 0/10,240 exact, max 26 ulp |
| Tract decluttered (fused RmsNorm) | 0/10 exact, max 31 ulp | 7,401/10,240 exact, max 3 ulp |

Every row whose mean is bit-equal gives a bit-equal output row (asserted), so the whole native deviation is the one bounded step: float summation order in ReduceMean.

ORT literal vs ORT optimized (default `ORT_ENABLE_ALL`): 7,136/10,240 exact, max 2 ulp. ORT's own optimizer changes the bytes, so the oracle configuration is part of the contract.

Against an f64 reference mean (f32 squares summed in f64, divided, rounded once):

| Arm | Max error |
|---|---|
| Native | 2 ulp |
| ORT literal | 2 ulp |
| Tract typed | 33 ulp |

Pinned bounds: native output <= 4 ulp vs ORT literal; native mean <= 4 ulp vs ORT literal and <= 4 ulp vs the f64 reference.

### 4.3 Gather

Byte ranges computed from the source initializer placement (base offset + row x 4096) equal the pinned ranges and the fetched rows' SHA-256 for all 10 tokens. Index `vocab` (151936) and `-vocab-1` are refused. Index -1 maps to row 151935 (the ONNX negative-index rule).

## 5. Harvest map: Tract construct, retained semantics, owner, transformation, refusal / missing

All rows are source-backed.

| Tract construct | Retained / lost | Owner of the missing check | Falsifier or note |
|---|---|---|---|
| ONNX import (`onnx/src/model.rs:119-158`, `parse_graph`) | Node name kept; empty name becomes the first output name or `{n}-{op}`. Ops are looked up by op_type only (`model.rs:135`, `op_register.0.get(&pbnode.op_type)`), so the DOMAIN IS NOT CONSULTED. Unknown op becomes an `UnimplementedOp` placeholder that fails later at typing. Outlet labels kept for every output. Domain lost. | Admission (source inventory) | `neg_domain`: ORT refuses `com.example:Abs`; Tract imports it as Abs. |
| Opset (`model.rs:264-278`) | Only the default-domain version is kept, as one integer. Outside the tested range 9–18 Tract only logs a warning. | Admission enforces `ADMITTED_OPSETS` 13..=18 from the source. | |
| Model metadata (`model.rs:289-324`) | producer, version, domain, ir_version, doc_string, metadata_props kept as properties. | | |
| Initializers / constants (`hir/src/infer/model.rs:202-218`, `InferenceModel::add_const`) | `add_const` returns an existing Const of equal value, so two source initializers with equal values (T = 0, Z = 0) become ONE node named after the first; the second initializer's identity is lost. | Admission reads initializer identity and digests from the source inventory. | |
| Lazy external data (`model.rs:340-345`, `with_lazy_external_data`) | External weights stay as named references without reading bytes. | | This let the full 7,620-node graph be imported with no weight file. |
| Typing (`into_typed`) | Shapes and dtypes. Rank fix-up Consts / AddAxis nodes are synthesized (`.fix-rank`, `-dispose-dims-0`). ReduceMean is renamed `<name>.mean` and becomes Reduce<Mean> (doc: "Accumulates in f32"). | Native broadcastable check is the admission gate. | An impossible broadcast (B[4] vs F[2,3]) is typed symbolically as `broadcast((3),(4))` and fails only at run time ("Can not broadcast 3 against 4"); ORT refuses at load. |
| Flatten (`hir/src/ops/array/flatten.rs:11`) | Refuses a shape with more than one symbolic dim ("Can not compute a shape with square of symbols"). | | The Jina export flattens [batch_size, sequence_length], so the full model types only after pinning input facts (pinned [1,10]). |
| Declutter: Div | `Div(Y, B)` becomes `Mul(Y, recip(B))` (`n_div-recip` Const). Not numerically identical (161/1,536 differ) and removes the source Div op. | | `declutter_div_to_reciprocal_is_not_exact` |
| Declutter: RMSNorm | Pattern Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul becomes one fused RmsNorm op (plus Reduce<MeanOfSquares> for a separately requested mean). Source names of Pow/Add/Sqrt/Div/Mul/Cast are lost. | | Worth harvesting as a recognizer, not as a kernel. |
| Declutter: Silu, EinSum | `x * Sigmoid(x)` becomes Silu (28 in Jina). MatMul becomes EinSum (253). | | |

### 5.1 Classification of Tract transformations

| Class | Members observed |
|---|---|
| Algebraically equivalent | Cast f32 -> f32 identity; `Pow(x, 2)` -> `x * x` when the exponent is exactly 2.0; rank fix-ups; constant dedup of equal values (numerically equivalent but provenance-losing) |
| Numerically bounded, non-identical | Div -> Mul(recip); ReduceMean summation order; fused RmsNorm; ORT's own optimizer |
| Backend-specific | EinSum lowering, packing (not exercised) |
| Provenance-losing | domain; const dedup; declutter fusion names; shape-folded nodes |
| No native counterpart today | see §6 |

## 6. Native coverage of the full Jina graph

Source: the Tract decluttered graph of the pinned [1,10] model, 2,106 nodes. Histogram: Add 140, AddAxis 312, And 1, Cast 1, Concat 57, Const 346, Cos 1, EinSum 253, Gather 2, GatherElements 1, Iff 29, IsNan 28, MoveAxis 56, Mul 309, MultiBroadcastTo 29, Neg 56, Reduce<Sum> 1, Reshape 196, RmAxis 3, RmsNorm 113, Silu 28, Sin 1, Slice 112, Softmax 28, Source 2, Sub 1.

| Class | Ops | Notes |
|---|---|---|
| ADMITTED in this probe | RmsNorm (BOUNDED recipe); elementwise Add / Mul / Sub / Div / Neg via the facade (exact) | |
| NATIVE-NOT-EXPORTED | Softmax; Silu | `ndarray::hpc::activations::softmax_axis_f32` exists. `sigmoid_f32` exists but Sigmoid has no measured bound. Exp / Tanh bounded exist. |
| MISSING | MatMul / EinSum | Facade REFUSE until the CompiledLinear identity defect is resolved. `bf16_gemm_f32` / `blas_gemm` exist in ndarray but there is no admitted f32 MatMul contract. |
| MISSING | Gather (f32 embedding) | The probe resolves byte ranges; the facade REFUSEs. |
| MISSING | Cos / Sin (RoPE) | No bounded native kernel exported. |
| MISSING | IsNaN; Iff over f32; GatherElements | |
| MISSING | Layout ops: Reshape, AddAxis, MoveAxis, Slice, Concat, MultiBroadcastTo | ndarray views exist but no admitted contract. |

Whole-model native inference is NOT reachable today. The first blocker is MatMul.

### 6.1 Per-stage provenance of the full model

Pinned [1,10], lazy external data. Timings: import 25.6 ms, into_typed 153 ms, declutter 221 ms.

| Stage | Nodes | Source names kept | Outputs labelled |
|---|---|---|---|
| Inference | 7,937 | 7,605 (15 source nodes have empty names); 17 synthesized | 7,620/7,620 |
| Typed | 3,170 | 1,985 | 2,546/7,620 |
| Decluttered | 2,106 | 1,046 | 1,164/7,620 |

At the typed stage, shape computation is folded away: Constant 2,759, Unsqueeze 884, Shape 428, Gather 369, Reshape 228, Concat 227, Transpose 141, Cast 118, Mul 115, ReduceMean 113 (renamed), and others.

### 6.2 Harvest ledger

| Item | State | Owner |
|---|---|---|
| Tract `InferenceModel → TypedModel` as cold frontend | IMPLEMENTED (lab, this probe) | D-ONNX-INTAKE / D-NAM (EXO-152) |
| Raw-protobuf source inventory (domain, opset, initializer identity and digest) | IMPLEMENTED (lab) | this probe; would move to the D-NAM intake owner |
| Two admitted methods (`TinyRecipe` exact, `RmsNormRecipe` bounded) | IMPLEMENTED (lab) | D-NAM method contracts |
| `Gather` axis 0 from source placement, ONNX index rule | IMPLEMENTED (lab); facade still REFUSE | `lance-graph-onnx` (reclassification is an owner decision) |
| Elementwise float ops, `Sqrt`, `Abs`, int32 compare / masked sum | REUSED from the facade, unchanged | `lance-graph-onnx` (#1754 / #1758) |
| `ndarray` `mean_axis`, `softmax_axis_f32`, `sigmoid_f32`, `blas_gemm`, `bf16_gemm_f32` | EXIST, not exported for ONNX | ndarray; export decisions D-ONNX-NATIVE-1 |
| Tract fusions (RmsNorm, Silu) | REUSABLE as pattern evidence only | — |
| Tract numerical kernels | REFUSED for harvest (`Reduce<Mean>` 33 ulp from the reference) | — |
| Tract declutter as an admission source | REFUSED (Div → Mul(recip), name loss) | — |
| f32 MatMul, RoPE Cos/Sin, IsNaN, f32 Where, GatherElements, layout views | MISSING | see §12 |
| Burn `onnx-ir` arm | NOT RUN; DEFER | D-ONNX-INTAKE |
| Earlier TFLite TANH / Softmax evidence (#1800, #1801, #1810) | unchanged, not re-run, not superseded | D-TFLP |

## 7. Admission contract

`Refusal` variants:

| Variant | Meaning |
|---|---|
| Frontend | The frontend could not import or type the graph. |
| ForeignDomain | A node's domain is not the default ONNX domain. |
| Opset | The source opset is outside `ADMITTED_OPSETS` (13..=18). |
| UnsupportedOp | An operator with no admitted method. |
| Pattern | The subgraph does not match the method's pattern. |
| RewriteAssumption | A law the method relies on does not hold (for example the Where else-value is not 0). |
| Dtype | A dtype check failed. |
| Native(lance_graph_onnx::Refusal) | The native facade refused. |
| StaleGeneration | Initializer digests differ from those the recipe was prepared with. |

`Numeric` is `Exact` or `Bounded { op, reason }`.

`Receipt` fields: `method`, `model_sha256`, `opset`, `frontend`, `source_nodes` (source names in graph order), `initializers` (name, placement digest), `numeric`. It has a `check_generation` method.

Two methods exist:

- **TinyRecipe.** Laws: `ReduceSum(Where(c, X, 0)) == masked_sum(X, c)` requires Z == 0 and the same X; B read twice is one weight; broadcast decided once.
- **RmsNormRecipe.** Both Casts are f32 -> f32 identity; exponent exactly 2.0; numerator exactly 1.0; mean over the last axis with keepdims; weight length equals hidden. Numeric class: Bounded.

## 8. Falsifiers and disable runs

17 tests: 10 tiny, 7 Jina. The Jina tests are `#[ignore]` until the fixture is fetched, so a fresh clone reports them as not run. `cargo test` stops at the first failing test binary unless `--no-fail-fast` is given; the disable runs use it.

| ID | Disabled check | Test that goes RED |
|---|---|---|
| T1 | Where else-value need not be 0 | `invalid_rewrite_assumption_is_refused` |
| T2 | Domain not checked | `foreign_domain_is_refused_although_tract_accepts_it` |
| T3 | Generation not checked | `new_weight_generation_is_refused_by_the_old_recipe`, `stale_weight_generation_is_refused` |
| T4 | Native Div replaced by Mul(recip) | `declutter_div_to_reciprocal_is_not_exact` |
| T5 | Pow exponent not checked | `pow_exponent_other_than_two_is_refused` |
| T6 | Gather negative index not wrapped | `gather_rows_from_source_placement` |
| T7 | Gather upper bound not checked | `gather_rows_from_source_placement` |
| T8 | Mean axis not checked | `reduction_over_another_axis_is_refused` |
| T9 | Cast dtype not checked | `cast_to_double_is_refused_by_the_dtype_check` |
| T10 | Broadcast not checked | `incompatible_shape_is_refused` |
| T11 | Unsupported op admitted | `unsupported_operator_is_refused` |
| T12 | Native mean replaced by a sequential f32 sum | `rmsnorm_slice_against_ort` |

T9 note: the first attempt used an f16 `Cast_1` and stayed green, because Tract inserts its own cast and the pattern refuses first. The f64 control was added so the dtype check binds.

## 9. Measured cost (reported, not gated)

Release build, AVX-512 host, medians shown as ranges.

| Step | Tiny | Jina slice 10 x 1024 |
|---|---|---|
| Source inventory | 13.7–14.0 us | 18.6–19.8 us |
| Tract import + into_typed | 142–143 us | 176–177 us |
| Admission | 1.3 us | 2.9–3.0 us |
| Native run | 353–363 ns (5 allocations, 120 B) | 8.9–9.4 us (9 allocations, 123,048 B) |
| Tract typed runnable | 3.6–3.8 us (26 allocations, 3,119 B incl. input tensors) | 112–114 us |
| Tract decluttered fused RmsNorm | n/a | 22.5–22.7 us |

The native Jina run materializes every intermediate, because the facade's broadcasting binary adapters return owned arrays.

Full model, cold (no weights): 25.6 ms import, 153 ms into_typed, 221 ms declutter.

## 10. DR/DT DO-arm sketch (design only, nothing built)

| Selection element | Mapped to |
|---|---|
| Source-qualified operation identity | (model_sha256, opset, domain, source node names) |
| Algebraic method + proof receipt | method id + the laws checked + `Numeric` |
| Prepared numerical recipe | the recipe struct (constants copied out at preparation) |
| Hardware specialization | inside ndarray's own SIMD dispatch, not a lance-graph choice |
| Data / calibration generation | initializer placement digests, checked by `check_generation` before reuse |
| Explicit refusal | the `Refusal` enum |

DT (ClassId -> method schema, #1835) would hold the method identity binding. DR routes coordinates. The DO arm executes a pre-admitted recipe and never re-reads the ONNX graph. Not built: no store, no cache, no DR cube mask. Open owner decisions are listed in §12.

## 11. Research trajectory (forward map, not authorization)

Text -> exact tokenizer ids / offsets (pinned, recorded) -> native transformer -> last-token pooling -> graph-aware retrieval.

- Token boundaries are byte spans, not semantic relations. Any lexical-graph bridge must keep them separate.
- TensorFlow / TFLite learning: gradients, optimizer state, quantized training, intermediate rounding and weight-generation updates are new semantic contracts. Inference parity does not imply learning parity.
- The generation check here (initializer digests) is the hook a learning loop would have to respect. It is not a learning mechanism.

## 12. Next PR sequence

Smallest independent steps:

1. Owner decision: float reduction tolerance contract (D-ONNX-NATIVE-1), using this probe's measured bounds. Then export a bounded ReduceMean / ReduceSum in `lance-graph-onnx::bounded`.
2. Admitted f32 MatMul contract (resolve the CompiledLinear identity defect; ndarray `blas_gemm` / `bf16_gemm_f32` exist) with one Jina q_proj slice.
3. Gather axis-0 reclassification from REFUSE to INLINE for initializer tables (the index rule is tested here).
4. Softmax and Silu as bounded exports after measured bounds.
5. RoPE Cos / Sin bounded kernels.
6. Layout ops as view contracts.
7. A fused native RMSNorm kernel in ndarray if the 9-pass materializing composition is too costly (missing primitive, owner ndarray).
8. One full decoder layer slice, then the whole model.
9. Burn onnx-ir arm of the bake-off only if a later stage needs source fidelity Tract cannot supply. Tract plus the raw-protobuf inventory was sufficient for both fixtures here, so Burn is DEFER, not run.

Then the TFLite learning frontier.

Owner decisions OPEN:

- Tolerance contract (D-ONNX-NATIVE-1 / EXO-152).
- MatMul identity (CompiledLinear).
- Gather reclassification.
- Whether `ORT_DISABLE_ALL` is the standing oracle setting.

## 13. Verdict

Tract is ADOPTED as the cold typed frontend for this lab stage, on the condition that admission always pairs it with the raw-protobuf inventory. Tract loses domain, initializer identity and, after declutter, names and Div exactness.

- Do not admit from Tract's decluttered graph. Use its fusions (RmsNorm, Silu) as pattern evidence only.
- Tract's numerical kernels are not harvested (Reduce<Mean> is 33 ulp from the reference mean).
- The primary falsifier (Tract cannot retain enough source information) did NOT fire, because the gaps are recoverable from the protobuf.

## Reproduce

```text
cd crates/tract-harvest-probe && python3 tools/gen_tiny.py && python3 tools/fetch_jina_slice.py && cargo test --release --no-fail-fast -- --include-ignored && cargo run --release --example stages -- fixtures/jina/onnx_model.onnx 10 && cargo run --release --example cost
```

Versions: lance-graph base 61720fa (main), AdaWorldAPI/tract main bf7ca16, AdaWorldAPI/ndarray master 50381182 (one ndarray package in the probe graph, via `[patch.crates-io]`), Rust 1.99.0, ORT 1.29.0 (python, CPUExecutionProvider), onnx 1.23.2, tokenizers 0.23.3.
