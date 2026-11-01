# D-DYNBAG-0: the XLOG probe runs through Quack → mask-risc, gated by the bind store

**Date 2026-11-01 · ID D-DYNBAG-0 · Status IN PR · Plan `.claude/plans/d-dynbag-holographic-bags-v1.md` §17.7 · Crate `crates/xlog-relcomp-probe`**

- `native.rs` describes each side as a Quack `Query` and runs it with mask-risc `execute_into`:
  - support: `ScatterOrU32`, then `mask_and`;
  - bag count: `GroupReduce { Local(y), Count }`, then a checked product.
- No `ndarray::simd` fold word is called from the probe. Two direct calls remain, and neither is a fold: `mask_and`, an elementwise alias, and `checked_binary_i64`, TRACKED as MISSING-3.
- `native::Intake` looks Y up once under a `dr::MapTable` frame (#1905). Each query re-checks that frame and performs no further lookup (MISSING-5).
- `bag_count_admitted` runs only from an admitted `Population::Bag` `MethodBinding`, with each word read by the caller (MISSING-1, probe-local).
- Refused:
  - a retracted route (it produces no binding);
  - a stale frame;
  - a word the caller gave no reading for;
  - a `Distinct` method;
  - `p(X, Z)` (MISSING-6).
- Disable runs, all red:
  - Count swapped for Max;
  - backward dropped;
  - population check removed;
  - frame check removed;
  - unread word accepted;
  - intake frame not re-checked;
  - one lookup per query.

**OPEN:** the cost columns, MISSING-1/2/3/4 in their owner crates, and `crates/lance-graph-xlog` (D-XLOG-NATIVE-4), whose dependencies Amendment C now sets as Quack + mask-risc rather than ndarray only.
