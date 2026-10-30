# D-LITHO-1 — XLOG's Gate B as explicit register steps; bag-multiplicity fixture

**Date 2026-10-30 · Status LAB PROBE · Crate `crates/xlog-relcomp-probe` (`tests/litho.rs`) · Plan `.claude/plans/d-litho-photolithography-contract-v1.md` (mental model only) · EXO-50 `XLOG-CONSUMER-HANDOVER-20261030-12`**

Gate B as existing calls: fill and bind R/S (`register`, `bind_dr64`), type the chain (`Chain::new`; stale frame refused), bound constants as row slot x and selector `C[z]=1`, forward `R[x]`, backward `compose` + column 0, answer And, read as ids. Asserted equal to `propagate`. Registers prepared once per frame revision; slot and selector per query.

Multiplicity fixture: `q(Y) :- r(X,Y), s(Y,Z)`, expected {2:4, 3:1, 7:1}, checked by scalar enumeration and by a bag walk of XLOG's RIR for `q` and for `count(X)` (XLOG counts distinct full body bindings; a set walk gives 2). Shape: 3x3x3 ids, 4/9 cells per relation, one Y word, max count 4. The register answers support only; `admit` refuses `GroupBy`; the binder refuses Bag (`BagNeedsMultiplicity`). An ignored test demands the counts and fails. 2 disable runs red.

**OPEN:** counts need a per-Y count width with defined overflow, and a per-Y product (a two-atom chain count is deg_R(y)·deg_S(y)). Gate D: `ShaderDispatch` has no row-selection-mask field — named on EXO-50, nothing changed.

**DTO binding (`tests/dto_binding.rs`, EXO-50 message 19):** identities as `NodeGuid` references (lab classid stand-in), R/S as `ClassView` fields selected by `WideFieldMask`, the count method by `(owner_class, method_ordinal)` with `Population::Bag`, expected values keyed by identity. Bound only against a current route (stale map refused); executing the Bag method is refused today. XLOG is ready to be the first consumer of a DT64 bind store once one exists; no store is built here. 2 more disable runs red.
