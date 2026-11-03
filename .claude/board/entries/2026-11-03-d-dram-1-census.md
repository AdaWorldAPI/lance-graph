# D-DRAM-1 (= D-DRAM-P3-3): census of the shipped carry words, the "5×24" kernels, and the DR lease owner

**Status:** VERIFIED-IN-CODE, read-only. ndarray `master` at `b3d7919`, lance-graph `main` at `58e01cfab`. Owner: this session (D-DRAM-P3). Each listed item was read at the cited line. The search space was closed: ndarray `src/`, `benches/`, `examples/`, `tests/`; lance-graph `crates/` and `.claude/plans/`; each searched with named spellings. An item not found is reported as "not found under these spellings", never as absent.

## ndarray: what a k-of-n consumer may use

| Word | Where | Status |
|---|---|---|
| `half_add_mask(a, carry) -> (u64, u64)` | `src/simd_masking_ops.rs:6523`, re-exported `src/simd.rs:1062` | **Shipped, public** `const fn`: 64 independent lanes, carry to the next significance column in the same lane |
| `full_add_mask(a, b, c) -> (u64, u64)` | `src/simd_masking_ops.rs:6545`, re-exported `src/simd.rs:1062` | **Shipped, public** `const fn`: 3:2 carry-save compressor on 64 lanes |
| `U64x8::sum8_u8_add_masked(self, a: U8x64, keep: u64)` | per backend, e.g. `src/simd_scalar.rs:550` (read), plus avx512, avx2, avx, neon, wasm, nightly (harvested) | **Shipped**: masked vertical byte sums (#390), a different operation from bit-sliced counting |
| `U24x5` | `examples/microtile/tile.rs:60` | **Lab fixture only**: five 24-bit fields plus 8 metadata bits per 128-bit microtile. `tile.rs:23` says: "Lab fixture only: no public type, no production caller." |
| `Abacus`, `Abacus<BITS>`, `AccAbacus` | `examples/u9_abacus/abacus.rs:60`, `examples/u9_domino/domino.rs:316` | **Lab fixture only**. `src/` holds one comment (`simd_masking_ops.rs:6504`) and no type. |

The "5×24" spellings searched over `src/` (`5x24`, `5×24`, `5_24`, `24x5`, `x24`) matched only hex literals. **The only 5×24-like name is the `U24x5` microtile packing in `examples/`**, so #1949's UNVERIFIED mark stands, now with a reason: no shipped 5×24 kernel family was found under these spellings.

**Consequence for D-DRAM-3 / P3-5:** the k-of-n bench uses only `half_add_mask` and `full_add_mask`. Importing `U24x5` or `Abacus` would hit #1949's stop condition "unverified 5×24 API claimed as shipped". Promoting either to a public type is an ndarray-owner decision and needs a real consumer.

## lance-graph: who owns the DR map, and where a lease exists

| Item | Where | Crate status |
|---|---|---|
| `MapTable` (versioned coordinate table), `LocalMap` (local → global identity), `Frame`, `FrameStore` | `crates/lance-graph-dt64/src/dr.rs:805`, `:556`, `:469`, `:379` | workspace **member** (`Cargo.toml:23`): the DR map owner |
| `Guards { frame, focus, scope }`, `Guards::check` | `dr.rs:509`, `:526` | member. Here `Focus` is a tile *revision*: "Which tile of the frame is read" (`dr.rs:493-498`). It is not a lease. |
| `FocusKey { source: SourceId { dataset, branch, version, index_seal }, map, scope, facet }`, `check_seal` | `crates/drf-focus-probe/src/lib.rs:62`, `:24-30`, `:132` | **excluded lab** (`Cargo.toml` exclude). Its `Focus` gathers one `i32` facet column, the D-DRF-1 §5 derived cache, not an address receipt (`lib.rs:1-4`, `:143-160`). |
| `FocusLease` (one feature table per source; `StaleSeal` refusal) | `crates/lance-graph-nsl/src/lib.rs:380-398` | **excluded lab** (`Cargo.toml` exclude) |
| `SealedFacetLane` (Lance-side lane witness used by #1633) | `crates/lance-graph-contract/src/ordered_lane.rs:180` | member (contract) |

**Consequence for D-DRAM-2 / P3-9:** under these spellings, every focus or lease candidate outside dt64 is in an excluded lab crate. dt64's `Focus` is a revision, not a lease. So #1949's "existing focus lease to host disposable locator metadata" has no production host today.

- The nearest key shape is `drf-focus-probe`'s `FocusKey`: full source witness, map, scope and facet. That matches the receipt key #1949 §2.1 sketches.
- Its value, however, is a gathered column, not a locator.

A receipt therefore needs a production owner. D-DRAM-2 stays **GATED on EXO-46 / EXO-151**, and this census is the input to that decision.

**Not done:** the harvest capped on `carry`, on the full `MapTable`/`Frame` hit list and on the plan-reference list. The items above were read; the uncapped tails were not enumerated.
