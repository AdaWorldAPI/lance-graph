# 2026-10-22 — D-MPS-9: MORK as the differential oracle for every admitted `query_multi`

**Status:** MEASURED (lab; mork-pathmap-probe, crepe-plan-probe). Not admitted, not wired into Quack or R2IL.

## What changed

- `boundary.rs` gains a general MORK arm, `mork_rules(facts, rules)`. It takes Lance-owned plain data (`OracleRule`, `OracleAtom`, `OracleTerm`, `OracleFacts`) and returns the distinct head tuples (`TupleSet`, a set, oracle only) plus `OracleCost`. No crepe type enters the crate; nothing upstream-typed leaves it.
- Each rule runs as its own MM2 `exec`, so one rule's output never feeds another. Before the `exec`, the body is parsed with `Space::parse_sexpr` and enumerated with `Space::query_multi`, counting the effect callbacks. That times MORK's unification alone, with no answer decode. The callback's `bool` is a continue flag (`space.rs:1253`, `:1343`, `:1364`); it is called only after a successful unify.
- `mork_api::to_oracle(rule)` runs `admit` first and returns its refusal unchanged; constants above `u32` are refused (`ConstOutOfRange`).

## Tests (`crepe-plan-probe/tests/mork_differential.rs`, 7; `mork-pathmap-probe/tests/differential.rs` +1)

- Six fixtures, MORK set == reference set, each non-empty: `family` (two rules, head constant, repeated relation, a duplicated fact), `family_noisy` (== `family`'s set), `two_way`, `star3`, a body constant, a three-atom chain (acyclic; the native star refuses it, MORK answers it).
- MORK's callback count == the reference's derivation count over DEDUPLICATED facts, and the bag count over raw facts exceeds it on `family` (the duplicate `Allowed(2)`), so the dedup is not a no-op.
- `to_oracle` refuses exactly what `admit` refuses (Tri `Cyclic`, Loop `RepeatedVarInAtom`; Chain admitted).
- The general arm agrees with the fixed `Q(x,y,z)` query.
- Disable runs, each red then restored: constants encoded as variables (5 tests red); counting only `Ok` callbacks (match-count test red); `to_oracle` skipping `admit` (refusal test red).

## Cost split (`examples/mork_split.rs`, release, median of 5, n = 10 000, 12 500 input rows, 10 000 answers; Xeon @ 2.10 GHz, 4 vCPU, shared host)

| arm | µs | ns / input row |
|---|---|---|
| native prepare (layout) | 162 | 13.0 |
| native execute, `Count(BAG)` terminal | 17.6 | 1.4 |
| native execute, `Bindings(BAG)` (deferred handle) | 2.8 | 0.2 |
| native: enumerate all 10 000 answers (no allocation per answer) | 188 | 15.0 |
| MORK encode + build | 3 564 | 285 |
| MORK unification only (`query_multi`) | 12 289 | 983 |
| MORK exec + text decode | 23 548 | 1 884 |

- The `Bindings` outcome's `execute` does not read answers; enumeration is timed separately. Reading the answers costs 10× the count terminal here.
- MORK numbers are oracle reference costs, not gates. MORK's unification alone is about half its exec + decode time; the earlier §6 MORK query column (exec + dump + parse) is therefore roughly 2× MORK's join.

## Not covered (OPEN)

- `add_sexpr` / `remove_sexpr` / `dump_sexpr` / `transform_multi_multi` façade verbs: not built. Removal is a full re-prepare on the native side.
- Binding and witness validation are still inside `prepare`; not split out.
- The §7c disable "silently flatten a nested pattern" is not run: the native side has no representation for nested facts, so the flattened arm cannot be expressed. The nested refusal test stands.
- Structural sharing and memory across versions: not measured. Allocations: still blocked (MORK installs PathMap's jemalloc, D-MPS-4).

**Correction (chat, 2026-10-22):** PathMap's default build uses `slim_ptrs` (`TrieNodeODRc` = `SlimNodePtr` + allocator, `trie_node.rs:3020`); `Arc<dyn TrieNode>` is only the non-`slim_ptrs` build (`:2721`/`:2726`).
