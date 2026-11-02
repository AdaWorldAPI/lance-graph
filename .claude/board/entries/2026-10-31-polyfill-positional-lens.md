# The polyfill lens `0xFFFF :: 0xF000` binds by position; R2IL, SQL and Quack names are aliases — 2026-10-31

**Status:** MEASURED (tests in `crates/lance-graph-r2il/tests/integration/polyfill.rs`). Implements D-DT64-7 §1a/§1b (plan on PR #1898) for the relational polyfill only.

## What changed
- `lance_graph_r2il::polyfill`: `POLYFILL_LENS = compose_classid(0xFFFF, 0xF000)` and `POLYFILL`, 17 positions in `binding::EXECUTABLE` order. Each position carries its opcode, arity and aliases: the R2IL name first, then a SQL name where one fits, then the Quack variant.
- `binding::prepare` accepts this lens beside the R2IL fold reading. Under it, word `p` is `POLYFILL[p]`; the view's label at `p` must be an alias of `p`, and a name belonging to another position is refused (`ExecRefusal::NotAnAlias`), never followed. The R2IL fold reading keeps its label binding.

## Measured
- R2IL, SQL and Quack views bind the same steps and the same plan id, and answer the scalar oracle.
- Methods registered by Quack names (`Cmp::LtI32`, `Cmp::EqI32`, `Filter::and`, `Filter::negate`, `Agg::{Count, Any, All, SumI32, MinI32, MaxI32, CountDistinctOrderedU32}`) answer exactly as Quack's own `lower` + mask-risc `execute` over the same lanes: 21 cases over a partial, a full and an empty selection, plus one compound filter.
- The table matches `R2ILVocabulary` (name, stack arity, one result) and the `EXECUTABLE` order; every alias names one position; the table's arity overrules a wrong runtime `declare_algebra`.
- Disable runs, each red: alias check removed (1 fail), positional path removed (4), lens not accepted (5), SUM/MIN opcodes swapped (3), a duplicate alias (1), table arity check removed (1, after adding the wrong-declaration test; it was green before because registration already ties arity).

## What "register SQL via Quack" turned out to mean
Quack has no SQL text parser. Its SQL surface is the typed builder (`bind::table(..).where_eq(..).count()`) lowering to `Query { Filter, Agg }`. So SQL registers through Quack's variant names, and those are now aliases with a parity test against Quack's own execution. The SQL keyword aliases (`<`, `AND`, `COUNT`, `EXISTS`, ...) are recorded, but nothing in the tree parses SQL text into them.

## Not covered (OPEN)
- Quack aggregates with no single executable word: `Rows`, `BlendI32`, `ScatterOrU32`, the `Group*` family (`GROUP_SUM` and `KEEP` write sinks and are refused by the binding). `Cmp::{Ne, Le, Gt, Ge}*`, the ternary matches, strided compares and `Range` are composable or absent, not aliased.
- SQL `NULL` over an empty selection: `SUM`/`MIN`/`MAX` answer the fold identity (`0`, `i64::MAX`, `i64::MIN`), as Quack's `Agg` does through this mapping; three-valued bindings are refused.
- The lens is still declared at runtime through `BindStore::declare_algebra`, and DT64 does not yet act as the authority for `0xFFFF` (D-DT64-7-4, O8). The opcodes are still `ogar-loco` `FnIndex` values.
