# R1: optional R2IL reading of a DT64 MethodBinding (prepare/run)

**Date 2026-10-31 · D-id D-DYNBAG §19.3 R1 (scope corrected by §20) · Status DONE (lab-tested, optional reading)**

`lance_graph_r2il::binding` lets an admitted DT64 `MethodBinding` be run through the existing `FoldDialect`, when a consumer asks for the R2IL reading. It is not the path any consumer must take. Native execution (Quack, mask-risc, `ndarray::simd`) runs its own prepared plan without consulting a codegraph.

Lifecycle, as in the Tract probe (`tract-harvest-probe/src/plan.rs:116-128,171,204`):
- `prepare` (cold), in this order:
  - frame/focus/scope guards;
  - route at the captured version and now;
  - class basis via `BindStore::resolve`;
  - `BindStore::admit`;
  - refuses WHEN guards and three-valued truth;
  - lens concept = OGAR `CONCEPT_R2IL_FOLD` (provisional; lab only);
  - per effect: lens, word label resolved to exactly one opcode by OGAR's `R2ILVocabulary` (the position is free; only the name binds), translatable subset (17 opcodes: stack operands only, one pushed result), and OGAR arity = argument count.
- `run` (hot): guards, route and an explicit `ExecPermit` again, then the bound opcodes through `FoldDialect`. No `ClassView`, vocabulary or lens lookup.
- `Receipt`: plan id (FNV-1a over method ref, lens, steps), frame revisions, steps executed, mask-risc programs run, output. Bounded by step count. It is not a row trace, not rewrite history, not logical provenance.

MEASURED (`cargo test -p lance-graph-r2il`): 14 new tests, the existing 9 + 22, and `r2il-mask-abi-probe`'s parity suite all green. 11 disable runs, each red:

| guard | red |
|---|---|
| D1 label binds, not word number | 12 tests |
| D2 OGAR arity = argument count | an_arity_the_lens_declares_against_ogar_refuses |
| D3 run needs a permit | execution_needs_a_permit_beyond_admission |
| D4 run rechecks guards | a_moved_frame_focus_or_scope_is_never_reused |
| D5 run rechecks route | a_changed_route_is_never_reused |
| D6 lane constant fits lane type | operands_that_do_not_translate_faithfully_refuse_at_run |
| D7 WHEN refused | what_the_reading_cannot_honour_is_refused_at_prepare |
| D8 lens is the R2IL reading | a_word_under_another_lens_or_reading_refuses |
| D9 executable subset | what_the_reading_cannot_honour_is_refused_at_prepare |
| D10 effect names the R2IL lens | a_word_under_another_lens_or_reading_refuses |
| D11 three-valued refused | what_the_reading_cannot_honour_is_refused_at_prepare |

A first disable of "OGAR arity vs declared arity" stayed green: `BindStore::register` already ties argument count to the declared arity, so that comparison could never fire on its own. It was removed; the refusal still reports the declared arity.

**Found:** `FoldDialect`'s lane compares narrow an `i64` constant with `unwrap_or(i32::MAX)` / `unwrap_or(u32::MAX)` (`lance-graph-r2il/src/lib.rs` `INT_EQUAL`, `INT_S_LESS` arms), which changes the predicate for an out-of-range constant. The seam refuses such a constant first; the dialect's own saturation is unchanged (OPEN, owner: lance-graph-r2il).

OPEN:
- who issues `ExecPermit`;
- `0xC401` is provisional in OGAR;
- `GROUP_SUM`/`KEEP` (sinks) and immediates (`NUMBER`, `LOAD`, `RANGE`) are refused, not translated.

**D-DYNBAG-0 does not depend on R1** (§20). Its seam is binding → prepared native (Quack) plan.
