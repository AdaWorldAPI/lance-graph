# CG-8: the Codegraph reads the effects rows (EXO-241)

Ruling: D-CODEGRAPH-WAVE0-ACCEPTANCE, priority 2. Stacked on PR 2004.

## What it is

`crates/lance-graph-codegraph/src/effects.rs`, read-only:

- `Effects::open(dir)` reads every `<crate>.effects.tsv` (ruff_charon effects version 1,
  `subject effect kind file start end`). A file that does not match the format is refused whole.
- `Effects::of(symbol)`: `Witnessed(rows)`, `NoneWitnessed` (not a purity verdict) or
  `CrateNotRead`.
- `EffectClass::{Division, Remainder, Shift, Write}` groups the kinds a review asks about.
- `Codegraph::effects_reached(symbol, &effects, keep)`: the effect rows in bodies the symbol is
  or reaches, each graded `Own`, `Confirmed`, `Possible` or `Unknown` from the call rows, plus
  every known `ReachLimit`.
- `Effects::identity(krate)`: FNV-1a 64 of the file's bytes and its row count.

Nothing is written into the call masks. `Codegraph::impact` and the ruff_charon format are
unchanged.

## Acceptance

"Which witnessed division, remainder, shift or write effects are reachable from a named symbol":

```
$ cg_query tests/data/effects <foundry-parity-lab> effects \
    --reach "foundry_parity_lab::arm_b::{ItemQuery<'s>}::max_quantities" \
    --class division,remainder,shift,write
reach  Confirmed  effect  ASSERT  DivisionByZero  …::cook_ops  src/arm_b.rs  245  7745..7780
reach  Confirmed  effect  ASSERT  Overflow:Div    …::cook_ops  src/arm_b.rs  245  7745..7780
(each twice: the store has two rows for the site)
LIMIT  CalleesOutsideStore { crates: ["alloc", "core", "lance_graph_mask_risc"] }
effects-file  foundry_parity_lab  fnv64=05ca96551de2c6a0  rows=213
reached  4  PARTIAL (see LIMIT lines)
```

"A missing effects artifact must produce UNKNOWN/PARTIAL, never an empty clean answer":

```
$ cg_query tests/data/rev1 <src> effects --reach cg_app::run --class division
LIMIT  EffectsNotRead { krate: "cg_app" }
LIMIT  EffectsNotRead { krate: "cg_lib" }
LIMIT  DynamicCalls { store_dynamic_calls: 1 }
reached  0  UNKNOWN (partial: see LIMIT lines)
```

| asked to preserve | how |
|---|---|
| source spans | each row keeps `file start end`; the CLI adds the 1-based line |
| coverage | `EffectsNotRead` per stored crate without a file; `CalleesOutsideStore` names the crates reached but not stored; `DynamicCalls` when the symbol reaches a dynamic call; `Unplaced` for an effect whose subject has no position |
| residuals | `HarvestIncomplete` for a crate that is partial, has residuals, or has no residuals file |
| revision identity | the file's digest and row count. **The effects file carries no source revision**, so this identifies the bytes read, not the commit. Binding it to a revision is CG-3 (EXO-235). |

## What it found on real data

- All 62 effect subjects of the `foundry_parity_lab` store resolve to Codegraph positions.
- `--kind ASSERT:DivisionByZero` lists five rows at four sites: `method.rs:58`, `arm_a.rs:75`,
  `arm_b.rs:245` (two rows) and `oracle.rs:62`.
- `method::execute_can_cook` reaches no division, although `can_cook_guard` (`method.rs:58`) is
  registered from it through a closure. That is the closure-ownership gap of CG-1 (EXO-233): the
  call rows have no edge into a closure body. The answer is `PARTIAL`, not clean, only because
  of the other limits; nothing here reports the missing closure edge itself.
- Almost every function reaches `core` or `alloc`, so almost every answer is `PARTIAL`. That is
  accurate: those bodies were not extracted.

## Tests and controls

- `tests/effects.rs`: 20 tests. Real fixture: `tests/data/effects/` (213 effect rows and the
  matching call rows of `foundry_parity_lab`). The four reach grades use the `rev1` call rows
  with one synthetic effect row.
- `python3 .claude/probes/2026-11-11-codegraph-effects/disable_runs.py`: 29 guards, each patched
  out through an asserted anchor. Receipt beside it: 29 red, 0 not red.

## Not done

- Target-kind and cfg rows, the other half of CG-8's plan entry. They need a decision on the
  cfg source and are a separate slice.
- No join with `impact_report`'s `Scope`: declared targets and excluded regions do not narrow a
  reach answer yet.
- The reach is computed by one `impact` per candidate symbol (cold path). It is not timed here
  and makes no latency claim.
