# D-ONNX-NATIVE — ONNX vocabulary over existing native functions

**Status:** D-ONNX-NATIVE-0 IMPLEMENTED as a draft (`crates/lance-graph-onnx`). It does not change the production ABI and has not been merged. Everything past §5 is a proposal.

**Rule:** ONNX is a vocabulary over our existing mathematics, not another execution system. The path is `ONNX-compatible Rust API → existing native function` and nothing else: no graph, AST, interpreter, VM, runtime dispatcher, `LazyLock`/`OnceLock`, ClassId routing or per-call backend choice.

**Scope boundaries:**
- R2IL routing (D-MPF-4, #1752) is a separate architecture, and this crate does not import it.
- `ndarray` owns SIMD and the kernels, and it does not depend on this crate.
- Quack, DR64³ and Palette256³ are not recreated here.

## 1. Categories

| category | meaning | cost contract |
|---|---|---|
| **ALIAS** | The native function already has the ONNX contract, so it is re-exported under the ONNX name. | The same symbol: zero added instructions, zero allocations. |
| **INLINE** | A small adapter does only the semantic work the native function lacks, then calls it. | Only that work is added (§4). |
| **NATIVE** | The mathematics exists natively, but it needs a composition or a decision that is not made yet. | Not exported. |
| **REFUSE** | The ONNX semantics differ from what we have, or would need an execution system. | Not exported. Runtime facts are refused as `Refusal` values. |

## 2. Crosswalk (source-backed)

Every ALIAS and INLINE row was checked against the native source. The checks covered dtype, rank, broadcasting, axes, overflow, rounding, empty tensors, output shape and the bool representation.

**Bool convention:** ONNX `tensor(bool)` is the native packed mask. Bit `i % 64` of word `i / 64` is element `i`, and the bits past the last element are zero. `mask_words(n) = n.div_ceil(64)`.

### ALIAS

| ONNX | native | checks |
|---|---|---|
| `Abs` (f32) | `ndarray::hpc::vml::vsabs` | IEEE exact on ±0, ±inf and NaN, for n = 0..70 (SIMD body and tail) |
| `Sqrt` (f32) | `ndarray::hpc::vml::vssqrt` | IEEE exact (correctly rounded sqrt); `sqrt(-x) = NaN`; `sqrt(-0) = -0` |
| `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual`, `Equal` (int32 tensor vs a scalar) | `ndarray::simd::{gt,ge,lt,le,eq}_i32_to_mask` | signed comparison; `i32::MIN`/`MAX` thresholds; n ∈ {0,1,15,16,17,63,64,65,130}; tail bits zero |
| `And`, `Or`, `Xor` (bool) | `ndarray::simd::mask_{and,or,xor}` | word-wise and exact; zero tails stay zero |
| `Not` (bool, n elements) | `ndarray::simd::mask_not(src, n, dst)` | clears the tail bits past n |
| `Where` (int32, equal shapes) | `ndarray::simd::blend_i32` | a set bit takes X, a clear bit takes Y; native asserts equal lengths |

`aliases_are_the_native_symbols` pins each alias by `fn_addr_eq`, so an alias cannot quietly become a wrapper. That was the D4 disable.

### INLINE

| ONNX | native call | added work and reason |
|---|---|---|
| `Add`/`Sub`/`Mul`/`Div` (f32, f64; multidirectional broadcast) | `ndarray` `&a op &b` | `broadcastable(shapes)`. ndarray broadcasts like numpy but **panics** on a mismatch (`broadcast_with(..).unwrap()`; `co_broadcast` is `pub(crate)`). The adapter turns that panic into `Refusal::Broadcast`. The arithmetic is IEEE in both. `simd::add_f32` and its siblings were **not** used: they truncate to `min(len)`, which is not ONNX. |
| `Neg` (f32, f64) | `ndarray` element map `-v` (the body of its `Neg` impl) | none. It is INLINE, not ALIAS, because the obvious alias `vml::vsneg` is wrong: its 16-lane body computes `0.0 - v`, so `Neg(+0) = +0` for n ≥ 16, while its scalar tail computes `-v`. Pinned by `vml_vsneg_is_not_onnx_neg_on_positive_zero`. |
| `ReduceSum` (int32, rank 1) | iterator `i64` sum | attribute validation at construction, once; an exact `i64` sum; an `int32` range check (`Refusal::OutOfDtypeRange`). ONNX does not define integer overflow, so the adapter refuses rather than wraps. There is no native unmasked `i32 → i64` sum (§6). |
| `ReduceSum(Where(Greater(X, T), X, 0))` (int32) | `gt_i32_to_mask` + `masked_sum_i32` | the range check only. Scratch comes from the caller, so there is no allocation. This is the #1752 composition. |

`ReduceSum` attributes are checked once in `ReduceSum::new(axes, keepdims, noop_with_empty_axes)`, never per call:
- `axes` ∈ {absent, `[0]`, `[-1]`} for rank 1;
- `keepdims` ∈ {0, 1};
- `noop_with_empty_axes = 1` with absent axes is refused, because it returns a tensor, not a scalar.

### NATIVE (exists, not exported yet)

| ONNX | native | why not exported |
|---|---|---|
| `Exp`, `Log`, `Tanh`, `Sigmoid` | `vml::vsexp` and the related vml functions | approximate (polynomial). On ndarray master `vsexp` returns 1.7e38 for +inf, halves every result above x ≈ 88.38 and returns ~1.2e-38 for −inf (fixed in AdaWorldAPI/ndarray#368, open). Exact and approximate stay separate, so these would need an explicitly named approximate surface with ULP bounds. |
| `ReduceSum` (float) | `ArrayBase::sum` | it reassociates (`unrolled_fold`). That is allowed under ONNX tolerance but is not bit-identical to sequential order, so the tolerance contract has to be stated first. |
| `ReduceMin`/`ReduceMax` of a masked population | `masked_min_i32`/`masked_max_i32` | the empty-population result is undecided (ONNX ReduceMin over an empty set is ±inf for floats and undefined for ints) |
| comparisons on u64 | `{lt,le,gt,ge}_u64_to_mask` | ONNX has no unsigned-64 compare-with-scalar distinct from int64; needs a dtype decision |

### REFUSE

| ONNX | reason |
|---|---|
| integer `Add`/`Sub`/`Mul` | overflow is undefined in ONNX. ndarray `+` wraps in release and panics in debug. Needs the G1 overflow decision. |
| tensor-vs-tensor comparisons; f32 comparisons to a mask | no native kernel. Writing one here would duplicate a SIMD kernel. |
| `MatMul`/`Gemm` | out of scope for D-ONNX-NATIVE-0. The CompiledLinear identity defect has to be resolved first. |
| `Gather`, `Scatter`, `If`, `Loop`, control flow | these need graph or execution semantics, which this crate does not have |
| ONNX protobuf import | out of scope. It would be a separate optional crate. |
| Palette substitution for any op | approximate. Never automatic. |

## 3. Cross-check against relalg and mask-risc

The composite answer is checked against four independent computations in one test (`composite_matches_the_oracle_mask_risc_r2il_and_relalg`):

1. a plain-iterator oracle;
2. a `lance-graph-mask-risc` program, `Pred::GtI32` + `Terminal::MaskedSumI32` via `execute_into`;
3. `lance-graph-r2il` `FoldDialect` (`LOAD`, `INT_S_LESS`, `procs::sum`);
4. `lance-graph-relalg` `run(Query{ rel: Atom::I32{ op: Gt, .. }, measure: Sum })`.

All four are dev-dependencies only. None is on the facade's call path.

## 4. Measurements and assembly (reported, not gated)

Setup: `examples/alias_cost.rs`, release, AVX-512 at compile time, load ≈ 0.1–0.4. Medians over 21 reps, in ns per call; 3 runs agree.

| rows | op | direct | onnx | #1752 wrapper (diagnostic) | allocs d/o/w |
|---:|---|---:|---:|---:|---|
| 64 | Greater (ALIAS) | 14.6 | 14.6 | — | 0/0 |
| 64 | ReduceSum(Where(Greater)) (INLINE) | 38.6 | 39.3 | 224.4 | 0/0/3 |
| 64 | Add f32 (INLINE) | 34.8 | 35.6 | — | 1/1 |
| 4096 | Greater | 311.1 | 310.7 | — | 0/0 |
| 4096 | ReduceSum(Where(Greater)) | 1807.9 | 1807.3 | 2047.1 | 0/0/3 |
| 4096 | Add f32 | 362.3 | 382.7 | — | 1/1 |
| 65536 | Greater | 6004.8 | 5796.6 | — | 0/0 |
| 65536 | ReduceSum(Where(Greater)) | 30531.1 | 28112.2 | 29773.5 | 0/0/3 |
| 65536 | Add f32 | 8003.4 | 8465.6 | — | 1/1 |

At 65536 rows the direct/onnx differences are run-to-run noise (other runs reverse them). Add's single allocation is the output array, the same as direct `&a + &b`.

The assembly is from the out-of-line `probe_*` functions (`cargo rustc --release --example alias_cost -- --emit=asm`):
- **Greater (ALIAS):** `fn_addr_eq(greater_i32, gt_i32_to_mask)` is true, so there is no wrapper symbol at all.
- **ReduceSum(Where(Greater)):** with `#[inline]`, the onnx probe is 163 instructions against direct's 147. Diffing the bodies, the extra instructions are the `int32` range check (`movslq`/`cmp`) and the `Result` write. Both probes carry the same `memset` (mask zero-fill in `pack`), `memcpy` and bounds-check panics. Without `#[inline]`, the onnx probe was a single tail `jmp` into the adapter.
- **Add:** direct is a 4-instruction tail jump into ndarray's `Add for &ArrayRef`. The onnx probe inlines `broadcastable` (a shape loop plus the cold `to_vec` allocation on the refusal path) and then makes the same ndarray call.
- **Scan for dispatch:** no `Lazy`, `Once`, `cpuid` or feature probe appears in any onnx probe or in `src/`.

The old 50 ns target does not apply to aliases. An alias's cost is the native function's cost.

## 5. Dependency audit

- `lance-graph-onnx`'s only normal dependency is `ndarray`, on the same git coordinate as `lance-graph-mask-risc`. The root `[patch]` unifies it with the sibling checkout, so the workspace holds one ndarray. `cargo tree -p lance-graph-onnx -e normal --depth 1` shows exactly that.
- Reverse: `cargo tree -i lance-graph-onnx -e normal` lists only the crate itself. No manifest under `crates/` names it. The root `Cargo.toml` names it only as a workspace member. The ndarray sibling has no reference to it.
- The relalg, mask-risc, r2il and ogar-loco crates appear only under `[dev-dependencies]`, for the cross-checks and the diagnostic baseline.

## 6. Proposed (not implemented)

- An ndarray owner fix for `vml::vsneg` (+0) and `vml::vsexp` (±inf, overflow halving): AdaWorldAPI/ndarray#368, open. After it lands, `Neg` could become an ALIAS.
- A native unmasked `i32 → i64` sum in `ndarray::simd`, so that `ReduceSum::sum_i32` becomes a direct call.
- The NATIVE rows in §2, each with its open decision named.
- `#1752` `onnx_compat` stays as the independent probe. This crate is the facade, and #1752's measurements are kept as the diagnostic baseline.

## 7. Gates

- [x] ALIAS = native symbol (`fn_addr_eq`), zero allocations (measured)
- [x] INLINE adds only its semantic work (asm diff)
- [x] no dispatch, no initialization, no per-loop decision (source grep + asm)
- [x] agreement with independent references (11 tests + 1 doctest)
- [x] explicit refusals (broadcast, axes, keepdims, noop, overflow)
- [x] 4 disable runs, each red on its own test only
- [x] clippy `-D warnings`, fmt
- [ ] independent review
- [ ] operator authorization before merge
