# CG-BEH-1: from a changed dispatch arm to the registered methods that use its word

**Status:** MEASURED (read-only probe), 2026-11-11, at `317c4ffc0`.
Plan: `.claude/plans/codegraph-behaviour-v1.md`. Owner of the area: EXO-155 (D-CGMM-0).
Board: `P-EXO-18`, channel EXO-232.

No word, classid or binding is added. Nothing is prepared or executed.

## Question

Can a change in Rust code be traced to the registered methods that depend on it, with each hop
showing what kind of evidence it rests on?

## Answer

Yes for one lens and two existing test bindings, at the granularity of a `match` arm.
The trace has three hops and two kinds of evidence:

| hop | read from | evidence |
|---|---|---|
| code → opcode | the `match f` arms of `FoldDialect::call` (`lance-graph-r2il/src/lib.rs`) | source structure |
| opcode → word | `POLYFILL[p]` (`lance-graph-r2il/src/polyfill.rs`), cross-checked against `binding::EXECUTABLE` | source structure |
| word → method, in order | `BindStore::method_ref` + `admit`, in a test run | DT64 registration |

Source evidence ends at the `POLYFILL` table. Everything after it is what a bind store holds.

## Measured

- **Lens:** `0xFFFF_F000`, 17 words. All 17 opcodes have a dispatch arm.
  `FoldDialect::call` has 25 opcode arms and one catch-all; 8 arm opcodes are not words of the lens
  (`NUMBER`, `LOAD`, `VIA`, `INT_LESS`, `INT_LESS_EQUAL`, `GROUP_SUM`, `RANGE`, `KEEP`).
- **Uses** (`uses.tsv`, exported by the r2il integration test): two methods of class `0x7F0B_0001`.
  Method 1 uses word 5 (`IntSLess`) then word 11 (`SUM`). Method 2 uses word 5 then word 10 (`PopCount`).
- **Arm query** (`examples/impact-arm-int_s_less.tsv`): a change at `lib.rs:980` is inside the
  `INT_S_LESS` arm (lines 976 to 1018), word 5, used at ordinal 0 by methods 1 and 2.
- **Symbol query** (`examples/impact-symbol.tsv`): at symbol level the same function realizes all
  17 words. 3 are used in this store; 14 print "no registered user in this uses file", never "unaffected".

## Negative controls

| control | result | file |
|---|---|---|
| dispatch mapping of `INT_S_LESS` removed | word 5 is `UNKNOWN`; its two `uses` rows are withdrawn; every other row is unchanged | `examples/impact-missing-dispatch.tsv` |
| source changed after the witness was taken (one line appended to a copy of `lib.rs`) | `STALE`, exit 4, no impact row | `examples/impact-stale-witness.tsv` |
| the witness matches the current source | exit 0 | |
| a symbol that dispatches nothing (`polyfill::position_of`) | `behaviour NONE … (not: no impact)` | |
| uses file missing | `ARTIFACT_MISSING`, exit 3 | |

`beh.py --self-test` has 13 tests on synthetic trees. `disable_runs.py` removes one guard at a time
from a copy of `beh.py`; all 12 cases go red. The Rust test's ordering assertion goes red when the
effects are read in reverse.

## What this does not show

- **`realizes` is not a behavioural witness.** An arm contains the code that runs for an opcode. It
  does not show that the code computes what the word declares. No per-word behavioural test is mapped.
- **It is not a compiler fact.** The arms are parsed from source. A compiler call graph sees one
  function for all 17 words and cannot narrow further.
- **The native path is not mapped.** `binding_native::prepare_native` lowers steps to mask-risc
  programs; that realization and everything below it (mask-risc, `ndarray::simd`) is a residual.
- **The uses cover one test store.** No production crate registers a binding with DO effects.
  Foundry's `can_cook_method` has no effects (`effects: Vec::new()`): it is ON + WHEN only.
- **ON and WHEN are not read.** Only DO is joined.
- **Parser limits.** `beh.py` finds the first `match f {` after `pub fn call(` in `impl FoldDialect`,
  and assumes block arms. Raw strings are not handled.

## Compared with `.mermaid/05-methods.md` (not used as an oracle)

- Agrees: DT64 stores the binding and executes nothing; the DO path is `binding::prepare` →
  `prepare_native` → `run_native`; `admit` is not a permit.
- The diagram does not show the word table or the dispatch arms; this probe adds that layer.
- One stale statement in the diagram: it places the G6 method lab on an open PR.
  `crates/foundry-parity-lab/src/method.rs` is on `main` (commit `fcee68c90`).
- `.mermaid/03` and `/06` were not read for this slice.

## Reproduce

```sh
cd crates/lance-graph-r2il
CG_BEH_USES_OUT=$PWD/../../.claude/probes/2026-11-11-cg-beh-1/uses.tsv \
  cargo test -p lance-graph-r2il --test integration cg_beh
cd ../../.claude/probes/2026-11-11-cg-beh-1
python3 -I beh.py words    --src ../../.. > words.tsv
python3 -I beh.py dispatch --src ../../.. > dispatch.tsv
python3 -I beh.py impact   --src ../../.. --uses uses.tsv \
  --symbol lance_graph_r2il::FoldDialect::call --line 980
python3 -I beh.py --self-test && python3 -I disable_runs.py
```

## Next smallest step

Replace the source-structure reading of the arms with compiler facts once the extraction driver
(CG-2, EXO-234) and the effects reader (CG-8, EXO-241) exist, and add one behavioural witness per
word so `realizes` can be graded above source structure.
