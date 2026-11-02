# D-SAG-6-2-LAB: bounded lab projection — two frontends, typed rows, identity layers, F01–F15

**Date 2026-10-31 · D-id D-SAG-6-2 · Status LAB (approved as a bounded laboratory projection; not production) · Prior [D-SAG-6-1-CHECKLIST](2026-10-31-d-sag-6-1-checklist.md)**

Approved scope: extend `crates/sag-ra-probe`. No other frontend runtime, store, registry or Codegraph, and no production code. Everything lives in the excluded lab crate; nothing depends on it.

Added:
- **Probe options in `sag-ra-probe`:**
  - `MACROS=1` runs build scripts and the proc-macro server;
  - `DUMP=1` now prints, per symbol, its moniker, the number of impls (`goto_implementation`) and a body digest.
- **`fixtures/sag_ident/{rev_a,rev_b}`:** two revisions of a three-crate workspace.
- **`tools/ident_diff.py`:** compares two revisions.
- **`tools/project_rows.py`:** typed relation rows from both frontends.
- **`tools/llbc_calls.py`:** now separates argument-rooted from local-rooted writes, and classifies trait calls as `Clause` / `Dyn` / `TraitImpl`.

## 1. Before declaring a capability missing

- **Trait resolution exists.** `Analysis::goto_implementation` on `Pred::test` returns both impls. On the cross-crate trait `left::Shape` it returns the impl in `right`.
- **Derives expand.** `Below` has 4 impls and `Even` 1; the difference is the three derives. They were absent only from `StaticIndex`, which indexes source tokens. The D-SAG-6-1 checklist's claim that proc macros caused the gap is corrected there.
- **`MACROS=1` changed nothing on these fixtures** (tokens 99, bodies 18 either way): they have no proc-macro crates. Its effect on a proc-macro-heavy workspace is NOT RUN.

## 2. Identity is four layers, kept apart

| Layer | Carried by | Measured behaviour |
|---|---|---|
| symbol (semantic) | rust-analyzer moniker, a crate-qualified path, e.g. `left::util::helper` | distinct across crates with equal paths and equal bodies. **Changes when the item moves** (`left::util::helper` → `left::ops::helper`), so it is a candidate identity, not a stable one |
| occurrence | file + text range per revision | positional: `helper`'s range moved from 71..77 to 127..133 between revisions |
| revision / build | repository commit, features, profile, toolchain | recorded by the caller. Not yet captured by the tools themselves |
| runtime binding | DT64 `MethodId` + `SigRev` | separate by construction: `MethodBinding::name` is carried, never identity |

**Continuity across a move is a claim, not a fact.**

`ident_diff.py` on rev A → rev B:

| Classification | Items |
|---|---|
| `SAME` | `app::area_dyn`, `left::Shape`, `left::Shape::area`, `right::Sq`, the `Sq` impl's `area` |
| `BODY_CHANGED` | `app::run` (it now calls `left::ops::helper`), `right::util::helper` (same path, new body) |
| `POSSIBLE_MOVE` | `left::util::helper -> left::ops::helper` (moniker gone, same kind, equal body digest) |
| `DIGEST_SHARED` in A | `left::util::helper` and `right::util::helper`: identical bodies in two crates |

So a body digest can never be an identity: equal text occurs in distinct symbols. A move needs a rebinding witness before it becomes identity, for example a rename edit recorded by rust-analyzer's `SourceChange`, or a human decision.

**The 4+12 key is not minted in this lab.** A key derived from the moniker would change on every move, as measured above. A key derived from the occurrence changes on every edit. Minting has to wait for a symbol identity that survives revisions. That is the contract's own reason: the key prerenders a node and must stay its address.

## 3. Projection: one relation set, witness-tagged

`project_rows.py` merges a rust-analyzer DUMP with `llbc_calls.py` output into `relation subject object witness evidence`.

Fixture `sag_fixture`:

| Relation | Rows | Evidence |
|---|---|---|
| `DEFINES` | 23 | WITNESSED (rust-analyzer) |
| `HAS_IMPLS` | 4 | WITNESSED (rust-analyzer) |
| `CALLS` | 38 | WITNESSED (Charon, resolved callee) |
| `MAY_DISPATCH` | 2 | POSSIBLE (Charon `trait:Clause`, `dynamic`) |
| `MAY_PANIC` | 8 | WITNESSED (Charon `Assert`) |
| `WRITES_CALLER_MEMORY` | 1 | WITNESSED: the tally closure, rooted in its captured state |
| `WRITES_THROUGH_LOCAL_REF` | 1 | POSSIBLE: `mask_below` writing its own `Vec` through `IndexMut` |
| `UNJOINED` | 17 | UNKNOWN: Charon names impls and closures (`…::Impl::call_mut`) differently from rust-analyzer's monikers (`…::impl::Below::Pred::test`) |

**A write is not an effect until its root is known.** The first version counted every write through a dereference as one class, which put `mask_below`'s local buffer write next to the tally closure's write to caller memory. The root check separates them: a written place rooted in an argument (a closure's captured state is argument 1) is caller memory. Disabling the check reclassifies `mask_below` as a caller-memory write.

**Dynamic dispatch has two LLBC forms.** A direct call on `&dyn Shape` is `trait:Dyn`. A call through a `&dyn` captured in a closure is `dynamic`. A generic bound is `trait:Clause`. None of the three names a confirmed callee.

**The 17 `UNJOINED` rows are the real cost of two frontends.** Free functions join by path; impls and closures do not. A join on spans or on Charon's `source_text` is the next step. Until then those facts stay UNKNOWN rather than merged by guess.

## 4. Binding through DT64 (explored, not built)

`lance-graph-dt64::MethodBinding` already carries:
- ON (`WideFieldMask`);
- WHEN (`Guard` = `SetExpr` under `Logic`);
- ordered DO (`Vec<Effect>`, each with a target field, an `AlgebraRef` and ordered `Arg`s);
- `MethodId` + `SigRev` as identity.

| Function | Binding |
|---|---|
| `count_below` | WHEN `x < t` (two-valued); one `Effect` writing the count field through mask-risc's popcount word. No new type |
| `count_with_tally` | **refused**: its `WRITES_CALLER_MEMORY` targets captured state outside the class's fields, and an `Effect` can only target a field position |
| `count_div` | refused for pure fusion while `MAY_PANIC`: the binding cannot express the panic |
| `count_generic` / `count_dyn` | refused: the callee is POSSIBLE |

DR64 consumes only admitted bindings. No DR row exists for a refused one.

## 5. F01–F15

| Gate | Result | Evidence / limit |
|---|---|---|
| F01 same name across crates/features/revisions | **PASS** across crates, revisions partial; features NOT RUN | distinct monikers for `left::util::helper` and `right::util::helper` despite equal bodies; a move is `POSSIBLE_MOVE`, not identity |
| F02 mutable capture | **PASS** | `WRITES_CALLER_MEMORY` on the tally closure; behaviour test shows fusion drops the effect |
| F03 panic / overflow | **PASS** | `MAY_PANIC` from `Assert`; profile-dependent, so the profile belongs to the revision layer |
| F04 trait dispatch | **PASS** | `trait:Clause`, `trait:Dyn` and `dynamic` are POSSIBLE; `goto_implementation` enumerates candidates (2 in-crate, 1 cross-crate) |
| F05 borrow/drop/unsafe/atomic | NOT RUN | none in the fixtures |
| F06 missing = UNKNOWN | **PASS** for joins | `UNJOINED` rows are UNKNOWN. A query layer that could make a closed-world claim does not exist yet |
| F07 DR ordinal reuse | NOT RUN | no DR rows produced |
| F08 K3 / bag | NOT RUN | |
| F09 forged proof | MISSING | no independent checker |
| F10 unfair baseline | NOT RUN | nothing timed |
| F11 changed source | **PASS** (lab) | `BODY_CHANGED` and `POSSIBLE_MOVE` from the body digest. A whole-revision digest is not yet recorded |
| F12 MLIR | MISSING | |
| F13 applied without approval | **PASS** by construction | no apply path; edits would be `SourceChange` values |
| F14 dynamic external effect | **PASS** | dyn calls stay POSSIBLE; binding refused (§4) |
| F15 synthetic labels | **PASS** | every row comes from compiled sources through the two frontends |

Admission gates before anything leaves the lab:
1. a symbol identity that survives a move, with a rebinding witness;
2. a span-based join of impls and closures across the two frontends;
3. F05 and F07 run;
4. separate owner approval for any DT64 or DR integration.

Reproduce, from `crates/sag-ra-probe`:
1. `DUMP=1 cargo run --release -- fixtures/sag_ident/rev_a > a.txt`
2. the same for `rev_b > b.txt`
3. `python3 -I tools/ident_diff.py a.txt b.txt`
4. for rows: Charon `--no-dedup-serialized-ast` on `fixtures/sag_fixture` → `llbc_calls.py` → `project_rows.py`.
