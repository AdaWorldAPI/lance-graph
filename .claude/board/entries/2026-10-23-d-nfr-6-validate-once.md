# D-NFR-6 — validate once, execute without repeated guards

**Date:** 2026-10-23 · **Status:** MEASURED, lab only · **Plan:** `.claude/plans/d-nfr-6-validate-once-v1.md` · **Crate:** `crates/dr-validated-view-probe`

- **Q1, the guard:** the 27–28 ns per-call guard (#1752 route) is `FrameStore::check` (25.7–28.3 ns; 96 instructions; six `lock` RMWs) plus two `u64` compares (~1.3 ns). The cost is the temporary `Weak`s from `Arc::downgrade`; the pointer comparisons cost ~1 ns each. `bind_dr64` adds a `Frame` clone (two more `lock inc`): 59–67 ns total.
- **Q2/Q5, validate once:** a `Table` owns data, frame store and focus map; every change takes `&mut`. `validate` runs the existing `bind_dr64` once and returns a `View<'t>`. While a view lives, advance, refocus and revoke are compile errors (E0502), and a view cannot outlive its table or leak a `'static` borrow (E0597). After the borrow, stale guards are refused at re-acquisition.
- **Q6, the cost:** view operations equal direct `ndarray::simd` within noise, at 0 allocations. `View::sum` is a tail call with no `lock` op. One validation is ~65 ns per view. Per-operation alternatives: +25–40 ns (route guard), +50–75 ns (`bind_dr64`).
- **Q3, what stays at run time:** revocation owned by another component (one epoch load per operation, `SharedView`); data not owned with its frame (`bind_dr64` checks the frame, not the bytes); a focus whose revision is chosen by the caller (fixed by an owner-issued revision).
- **Proposed, not implemented:** `FrameStore::check` by `as_ptr` comparison (no refcount traffic); `BoundDr64` borrowing its frame. Estimated validation ~10 ns, not measured.
