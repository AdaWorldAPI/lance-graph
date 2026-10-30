# D-XLOG-NATIVE-0 — XLOG plan vocabulary crosswalked onto mask-risc and relalg

**Date 2026-10-30 · Status PROPOSED / DOCS ONLY · Plan `.claude/plans/d-xlog-native-v1.md` · Precedent D-ONNX-NATIVE**

Each XLOG compiled-plan node and operator is mapped onto mask-risc / relalg in D-ONNX-NATIVE's four categories, read against `lance-graph-mask-risc/src/ir.rs` and `lance-graph-relalg/src/ir.rs`; no row is tested yet.

- ALIAS / INLINE: equality filters (`Pred::EqU32` / `NeU32`), `I32` comparisons, `And` / `Or` / `AndNot`, full-row `Distinct`, many-to-one semi / anti / inner joins (`Gather`, `EqU32Via`, `GroupKey::Via`), per-key `Count` / `Sum` / `Min` / `Max` on one table (`GroupReduce`, `GroupSumI32`).
- NATIVE: the many-to-many join Gate B uses (runs only on the lab `relcomp::compose`; `ScatterOrU32` may not feed a follow-on program), the bag count over such a join (a per-key product of two `GroupReduce` results), ordered `U32` comparison, 64-bit and symbol constants, recursion.
- REFUSE: outer joins, projection arithmetic, stratified negation, `LogSumExp`, probabilistic programs, float constants.

**OPEN:** no `lance-graph-xlog` crate until selection rows are tested against the probe's oracles, the many-to-many join has a production home, and relalg's workspace membership is settled.
