# 2026-10-23 — D-MPF-4: R2IL routes once per ClassId; the steady-state cost is the guard, not the route

**Status:** MEASURED, lab (`crates/math-polyfill-probe/examples/classid_route.rs`). Plan `.claude/plans/d-mpf-math-polyfill-v1.md` §10. No production crate, ABI or execution contract changed.

- **Attribution:** five different things are called R2IL, and they stay apart:
  - r2sleigh's `r2il`: IL types, no executor.
  - the machine reading: `r2conc` / `r2sym`, scalar machine state.
  - `ogar-r2il`: the opcode table, readings `0xC400` machine / `0xC401` fold.
  - `lance-graph-r2il::FoldDialect`: the fold reading only.
  - mask-risc / `ndarray::simd` / `simd_runtime`.

  The 94–99 ns and 171–172 ns of D-MPF-2 are the cost of choosing the mask-risc and FoldDialect paths per call. They are not a cost of R2IL routing.
- **Recovered 2–5 ns figures:** each is a direct kernel, not an R2IL call. The one cited as a comparison is `ndarray::simd` AND, AND, popcount over 4 words at 2.2–2.8 ns, against 54–70 ns through r2il + mask-risc (`2026-10-21-d-rft-1-cost-breakdown.md`).
- **Routing table:** every wired fold-reading op has a DIRECT or COMPOSE route onto existing `ndarray::simd` entrypoints, or is refused. Only machine semantics need R2Sleigh: carry/borrow flags, address-space load/store, branches, width-exact integers.
- **Measured** (64 rows / 65 536 rows): prebound route without guard 36–39 / ≈ direct ndarray. Guard (`FrameStore::check` + focus + scope) 27–28 ns, 0 allocations; objdump shows 2× `lock cmpxchg` + 2× `lock decq` from `Arc::downgrade`. Per-call `LazyLock` + `match` adds 0–1 ns. Prepared R2IL (FoldDialect's `last_program` as a `CompiledProgram`) = prepared mask-risc, 85–94 ns.
- **Asserted:** one resolution per ClassId; 0 route evaluations in steady state; LOWER/REFUSE refused at bind; focus, scope and source changes refused at run; rebinding does not re-resolve. Three disable runs red.
- **Proposal (not made, DR lab owner):** compare `Weak::as_ptr` with `Arc::as_ptr` instead of `Weak::ptr_eq(.., &Arc::downgrade(..))`. Same identity semantics, no atomics. Unmeasured.
- OPEN: the program-only half of mask-risc's per-call `precheck`/`validate`; mask-sharing callables for multi-op bindings; the U64 unsigned-compare refusal (`lt_u64_to_mask` exists).
