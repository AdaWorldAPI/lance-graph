# D-SAG-6-2-JOIN-IDENT: span-based join of two frontends; symbol identity that survives a witnessed move

**Date 2026-10-31 · D-id D-SAG-6-2 · Status LAB (bounded lab projection, not production) · Prior [D-SAG-6-2-LAB](2026-10-31-d-sag-6-2-lab-projection.md)**

This entry addresses the first two admission gates of D-SAG-6-2:
- gate 1: a symbol identity that survives a move, with a rebinding witness;
- gate 2: a join of impls and closures across the two frontends.

The scope stays as approved: everything lives in `crates/sag-ra-probe` (excluded from the workspace, no dependents). There is no new store, no registry and no 4+12 mint.

What was added:

**`sag-ra-probe`:**
- `DUMP=1` now also prints `at=<file>:<line>` (the name's line, 1-based) and `lines=<begin>-<end>` (the whole definition). Paths are relative to the workspace root.
- `RESOLVE=a::b,c::d` resolves each path through module scopes, which include `use` imports. It prints the canonical path of the item it reaches, or `unresolved` / `ambiguous`.

**`tools/span_join.py`:** joins Charon functions to rust-analyzer symbols by file and line.

**`tools/ident_chain.py`:** carries identities across N revisions.

**`tools/test_tools.py`:** 15 tests on hand-written inputs.

**`llbc_calls.py`, `project_rows.py`:**
- calls carry Charon ids;
- with a join file, subjects and local callees become rust-analyzer monikers;
- without a join file, the output is the earlier path join.

**Fixtures `sag_ident/rev_c` and `rev_d`:**
- rev C moves `helper` and keeps the old path as a re-export;
- rev D re-exports a *different* function at the old path.

## 1. Span join

| Class | Rule |
|---|---|
| `EXACT` | a rust-analyzer function, method, constant, static or enum variant whose name sits on the line where Charon's span starts, in the same file, with the same name |
| `ENCLOSED` | a closure or its `call_once` shim, joined to the innermost `EXACT` function containing it |
| `DERIVED` | a method of an impl that Charon marks `AutomaticallyDerived`, joined to the Self type's moniker |
| `AMBIGUOUS` / `UNJOINED` | `UNKNOWN`, never "no such item" |

**Results.** Charon was run with `--no-dedup-serialized-ast`; without it, type references are dedup indices and the derived join cannot read them.

| Crate | Bodies | EXACT | ENCLOSED | DERIVED | UNJOINED |
|---|---|---|---|---|---|
| `sag_fixture` | 29 | 14 | 12 | 3 | **0** (path join: 17) |
| `sag_ident/right`, rev A and rev C | 2 | 2 | 0 | 0 | 0 |
| `lance-graph-mask-risc` | 459 | 140 | 186 | 123 | 10 |
| `lance-graph-contract` (not used to build the rules) | 7189 | 3061 | 1640 | 2250 | 238 |

Every unjoined row has a named cause, and none can be joined without inventing a symbol:

| Cause | mask-risc | contract |
|---|---|---|
| compiler drop glue (`Impl::Builtin`, `Destruct`): no source item | 9 | 184 |
| anonymous `const _` | 1 | 32 |
| item in a build-script `OUT_DIR` file. rust-analyzer loads those only with `MACROS=1`, and under its own out dir, so the path never matches | 0 | 21 |
| `call_once` shim of a trait method declared without a body | 0 | 1 |

**The two `Impl::test` methods** share one Charon path. The path join cannot separate them; positions do (`impl::Below::Pred::test` vs `impl::Even::Pred::test`). The cross-crate impl joins too: `right::Impl::area` → `right::impl::Sq::Shape::area`.

**Closure subjects carry their line and Charon method,** e.g. `count_below::{closure@L26}::call_mut`. The first version wrote `…::{closure}` for both the closure body and its `call_once` shim. That turned a real call into a self-loop, and it would have merged two closures in one function. **A line-qualified subject is an occurrence, not an identity:** it changes when the code above it changes.

> **Superseded (same day):** the count comparison below excused gaps with a marker set fitted to the same data. Impls are now paired one by one; see "Impls, paired one by one" below.

**Impl counts from both tools** (`impls` rows):

| Crate | Types | AGREE | MARKER_GAP | DISAGREE |
|---|---|---|---|---|
| `sag_fixture` | 3 | 2 | 1 | 0 |
| `lance-graph-mask-risc` | 22 | 3 | 19 | 0 |
| `lance-graph-contract` | 532 | 182 | 287 | 63 |

- **The marker set** (`StructuralPartialEq`, `TrivialClone`, `Destruct`) was **read off the mask-risc disagreements**. Charon lists these compiler-generated impls; rust-analyzer's `goto_implementation` does not.
- **The marker set does not fully generalize.** On the contract crate, which was not used to build it, 63 types still disagree:
  - 15 are traits implemented only in *other* crates. rust-analyzer counts impls across the whole workspace; Charon sees one crate.
  - 43 have more impls in rust-analyzer than in Charon.
  - 5 have more impls in Charon.
  - Neither of the last two groups is explained yet. One suspected cause is inherent impl blocks with no methods, which are invisible in Charon's output. **OPEN.**

**Impls, paired one by one.**

- The probe's `IMPLS=1` prints one row per impl of every local type and trait. It uses the same `Impl::all_for_type` / `all_for_trait` calls as `goto_implementation`. Each row has the trait path or `inherent`, the self type, the crate the impl is written in, the item count, and the impl block's source range.
- `span_join.py` pairs rust-analyzer's and Charon's impls as a multiset keyed (owner, trait, self type). Each unpaired Charon impl carries its origin:

| Origin | Meaning |
|---|---|
| `closure` | a closure's state type; rust-analyzer has no ADT for it |
| `derived` | Charon marks the impl `AutomaticallyDerived` |
| `drop_glue` | a `Destruct` impl whose method is compiler drop glue |
| `source` | none of the above |

- Impls written in other crates are listed as `impl_scope` and not compared.
- Inherent blocks are counted by placing each Charon inherent method in the rust-analyzer block whose range contains it. Two blocks of one type can have identical Charon path segments, so the earlier segment-based count merged them. That accounted for all 14 of the contract crate's "rust-analyzer has more inherent blocks" cases. It was a defect in the tool, not a gap between the frontends.

| Crate | Paired keys | rust-analyzer only | Charon-only impls | Out of scope (other crates) |
|---|---|---|---|---|
| `lance-graph-mask-risc` | 175 | 0 | 205 | 0 |
| `lance-graph-contract` | 3609 | 0 | 2074 | 39 |

In the breakdown below, counts are unpaired keys, not impls. Several closures in one function share one Charon type path (`<fn>::Builtin`), so one key can hold several closure impls: up to 7 `FnOnce` impls for `reference::reference_execute_into` in mask-risc. That is why the keys (137 and 1752) are fewer than the impls.

Charon-only keys by origin and trait:

| Origin | Trait | mask-risc | contract |
|---|---|---|---|
| derived | `core::marker::StructuralPartialEq` | 28 | 446 |
| derived | `core::clone::TrivialClone` | 30 | 443 |
| closure | `core::ops::function::FnOnce` | 30 | 393 |
| closure | `core::ops::function::FnMut` | 26 | 332 |
| closure | `core::ops::function::Fn` | 16 | 54 |
| drop_glue | `core::marker::Destruct` | 7 | 79 |
| closure | `core::marker::Destruct` | 0 | 5 |

- No Charon-only impl has origin `source`, no impl is rust-analyzer-only, and no inherent method is unplaced, in either crate. Every gap is a kind of impl one frontend does not model, named by its origin, not excused by a list.
- The count method left 63 contract types unexplained. It could also have hidden a missing written impl behind an excused trait. The paired form leaves none unexplained, and it would show such an impl as `source`.
- The 39 out-of-scope rows include 5 impls of `cognition::op::Op` in `cognition_typestate` and 4 of `reasoning::Reasoner` in `lance_graph_callcenter`.

## 2. Identity that survives a move

**An identity is minted the first time an item is seen.** It continues only by one of:

| Status | When |
|---|---|
| `SAME` | same moniker, same body digest |
| `BODY_CHANGED` | same moniker, different body |
| `MOVED` | the moniker is gone; the later revision **resolves the old path** to a new item with the same kind and body digest. The witness is the compiler's own name resolution through a re-export |
| `DECLARED` | a row in a declarations file (old, new, who, why), attributed to its author; never inferred |

**Everything else ends the old identity:**

| Status | Meaning |
|---|---|
| `REBOUND` | the old path now resolves to a *different* body; the path continued, the item did not |
| `POSSIBLE_MOVE` | exactly one new item has the same kind and body, but there is no witness |
| `AMBIGUOUS` | several candidates, or the body occurs in two items |
| `CONFLICT` | the old path now resolves to an item that already had its own identity |
| `ENDED` | the item is gone and has no candidate |
| `UNRESOLVED` | the run has no resolve line for the path: `UNKNOWN` |

**On the real fixtures** (`sag-ra-probe` + `ident_chain.py`):

| Transition | Old `left::util::helper` | Identity |
|---|---|---|
| A → B (moved, no witness) | `AMBIGUOUS`: its body equals `right::util::helper`'s in A | ends, a new one is minted |
| A → B with `fixtures/sag_ident/rebind_a_b.tsv` | `DECLARED`, attributed `reviewer: moved in the util->ops split, body unchanged` | `sid-6` continues |
| A → C (re-export kept) | `MOVED` via `resolve`: the old path resolves to `left::ops::helper` | `sid-6` continues |
| A → D (re-export of `other`) | `REBOUND` | ends; `left::ops::other` gets a new identity |
| A → C → D | `MOVED`, then `SAME` | `sid-6` stays with `helper` through all three revisions |

**This satisfies gate 1 for free items.** The 4+12 key is still not minted. Two reasons:
- methods, closures and cross-crate re-exports are outside what has been exercised (below);
- a mint is an integration decision, which needs owner approval (gate 4).

## 3. Falsifiers

**Tests** (`python3 -I -B tools/test_tools.py`): 15 pass. Each guard was then removed, and the run turned red:

| Guard removed | Red |
|---|---|
| digest check before `MOVED` | `test_rebound_path_does_not_continue_identity`; on the real rev D, the re-export of `other` became `MOVED` |
| the resolve witness | 3 tests; the real rev C fell back to `AMBIGUOUS` |
| the `CONFLICT` check | `test_resolve_to_an_existing_item_is_a_conflict` |
| "a declaration may only name a new item" | `test_declaration_naming_an_existing_item_is_ignored` |
| position keying (names only) | `test_same_named_methods_are_told_apart_by_position`; on the fixture, 4 rows became `AMBIGUOUS` |
| the derived join | `test_derived_method_joins_to_its_self_type`; on the fixture, 3 rows went back to `UNJOINED` |
| the enclosing join | `test_closure_joins_to_its_enclosing_function` |
| placement of inherent methods in block ranges | `test_two_inherent_blocks_with_identical_segments_pair`, `test_empty_inherent_block_is_ra_only` |
| the drop-glue origin | `test_drop_glue_impl_is_labelled` |
| the crate-scope split | `test_impl_written_in_another_crate_is_scope_not_missing` |

**Run Python with `-B` for disable runs.** The disable `if outer:` → `if False:` keeps the file size identical. `git checkout` restored the file within the same second, and the stale `.pyc` (checked by mtime and size only) kept running the disabled code. The tree was clean while the test stayed red.

## 4. Not run, not claimed

- **Identity for methods, impls and closures.** `RESOLVE` resolves module-level items only. A method's moniker includes its impl type. A closure has no path.
- **A cross-crate re-export as a witness.** NOT RUN.
- **Recording a re-export that changes target** (rev C → D) as an edge of its own. The chain tracks defined items only.
- **`MACROS=1` on a proc-macro or build-script-heavy workspace.** It is the likely route to the 21 `OUT_DIR` items, but only if the two out-dir paths are mapped onto each other.
- **Pairing derive markers and closure impls.** They are left unpaired, not modelled on the rust-analyzer side. Pairing them would need a rust-analyzer view of closures and of derive expansions it does not list.
- **F05 and F07.** Still NOT RUN.
- **Cost of the workspace dump.** The index stage was 23 s in D-SAG-6-1. It is now 30.7 s, and that time includes the dump with a `goto_implementation` call per struct, trait and trait method. The whole run took 140 s: load 13 s, body walk 27 s, MIR lowering 61 s. Measured once; not a benchmark.

Remaining admission gates:
1. identity beyond free items;
2. F05 and F07 run;
3. owner approval for any DT64 or DR integration or 4+12 mint.

Reproduce, from `crates/sag-ra-probe`:
1. Dump each revision, resolving the paths that vanished:
   `DUMP=1 RESOLVE=$(python3 -I -B tools/ident_chain.py --needs A.txt B.txt) ./target/release/sag-ra-probe fixtures/sag_ident/rev_X > X.txt`
2. Chain them: `python3 -I -B tools/ident_chain.py [--declared fixtures/sag_ident/rebind_a_b.tsv] A.txt X.txt ...`
3. Run Charon in the crate: `charon cargo --no-dedup-serialized-ast --dest-file x.llbc`.
4. Join: `python3 -I -B tools/span_join.py ra.txt x.llbc <crate> > join.tsv`
5. Project: `llbc_calls.py` → `project_rows.py ra.txt calls.txt join.tsv`.
