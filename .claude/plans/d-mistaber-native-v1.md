# D-MIST-NATIVE — Mistaber's Kripke holdings as a vocabulary over existing algebra

**Status:** D-MIST-NATIVE-0 LAB PROBE (this PR); D-MIST-NATIVE-1..4 PROPOSED.
**Date:** 2026-10-30.
**Upstream evidence:** `AdaWorldAPI/mistaber-rs` (`docs/D-MIST-RUFF-XLOG-0.md`, decision table §7; Clingo oracle; `goldens/`).
**Claim:** EXO-50 `MIST-PROJECT-BAG-SET-20261030-02`.

## 1. Overview

The endgame is a `lance-graph-mistaber` crate wired like `lance-graph-onnx`. The crate holds no interpreter, graph or runtime. Each Mistaber operation is either an alias of an existing operation, a small adapter in front of one, or a typed refusal.

The `mistaber-rs` harvest found that every operation it could carry natively is classes A or B in its decision table: relational selection, plus stratified negation over input relations. The stable-model features stay with Clingo. This plan does not port a solver or the HLL compiler.

D-MIST-NATIVE-0 is the probe that tests this claim. Its findings:

- **The recursion is `holds`, not `accessible_transitive`.** Upstream's `holds_via(inherited(Parent), P, Child)` reads `holds` at the parent. The only negations are `not overridden`, `not asserts` and `not endorses`, all at the child and all over inputs. The answer is therefore the least fixpoint

  ```text
  H₀   = (asserts ∪ endorses) ∧ world
  Hₖ₊₁ = H₀ ∪ ((Hₖ ∘ children) ∧ ¬Block)      Block = asserts ∪ endorses ∪ overridden
  ```

  Each step is one `dr-math-binding-probe::relcomp::compose` and one AND-NOT. `accessible_transitive/2` is derived upstream, but no rule reads it and the goldens do not include it.
- **The two lineages use the same fixpoint.** `holds_via_rule`'s source rule and `inheritance_source`'s origin are each that fixpoint, gated by the final `H`: one register per rule, one per origin world.
- **relalg is the wrong host for the recursion.** It describes selections over one table; its docs say recursion (`μ`) and the hop are not built. The relation register and `compose` already exist in `dr-math-binding-probe`, which is what `xlog-relcomp-probe` uses as well.

## 2. Checklist

- [x] D-MIST-NATIVE-0: probe `crates/mistaber-kripke-probe` (own workspace root, CI workflow).
  - Reproduces all 11 Clingo goldens exactly.
  - 6 disable runs fail as expected.
  - Bag/set falsifier on the diamond.
  - Refusals: unknown fact, arity, 65th value on an axis, a selection that is no predicate.
- [x] Goldens exported by `mistaber-rs` (`oracle/export_goldens.py`, `test_goldens.py`, F5-gates and F6-undeclared-child fixtures); F3-naf refused at export.
- [ ] D-MIST-NATIVE-1: `lance-graph-mistaber`, an onnx-shaped facade (crosswalk §3). Blocked on a production home for `compose` (§6, Q1).
- [ ] D-MIST-NATIVE-2: lineage origins into `SupportLedger` as `EvidenceSourceId` with `SupportBasis::TextAttested`. This is the only lawful route to Epi5 (#1863/#1868 closed the topology route for non-causal steps).
- [ ] D-MIST-NATIVE-3: axes past 64 (worlds, propositions, rules). Refused today; tiling or a wider register is open.
- [ ] D-MIST-NATIVE-4: intake from compiled HLL. Real-corpus holdings hit the upstream provenance defects (`mistaber-rs#3`, `#4`), so the expected outcome there is refusals, not values.

## 3. Crosswalk (the facade D-MIST-NATIVE-1 would export)

| Mistaber (`kripke_rules.lp`) | Class | Native |
|---|---|---|
| `holds_via(asserted/endorsed, P, W)` | ALIAS | the input relation ∧ world mask |
| `holds_via(inherited(Wp), P, W)` | INLINE | `H[P]` has `Wp` → `children[Wp] ∧ ¬Block[P]` |
| `holds(P, W)` | INLINE | the least fixpoint above (compose + AND-NOT) |
| `holds/2`, `holds_via/3`, `holds_via_rule/5` as views | ALIAS | `WideFieldMask` selection over the holdings lens (§4), **Set** projection |
| `holding_strength(strong/moderate/weak)` | INLINE | masks over direct and inherited-from-direct; a label, never an Epi5 code |
| `inheritance_source`, `holds_via_rule` source | INLINE | the same fixpoint per origin or rule, gated by `H` |
| `overridden(P, W)` | ALIAS | the input relation |
| `accessible_transitive` | not exported | derived upstream, read by no rule |
| negative cycles over inputs, choice, optimisation (F3-naf) | REFUSE | stays with Clingo |
| taxonomy → worlds, strength → Epi5, `is_a` → Pearl3 | REFUSE | category errors (`mistaber-rs` `qualify.py`) |
| a holding with no `holds_via_rule` (`mistaber-rs#3`) | REFUSE | `MissingProvenance` |

## 4. The holdings lens and the bag/set split

Following the field-view ruling (`ogar-consumer-preflight.md:277`, "Classview can be used for any compute masking"), holdings are one relation lens with fields `proposition, world, mode, immediate_rule, source_rule`. Each predicate is a `WideFieldMask` selection over it:

- `holds/2` = {proposition, world};
- `holds_via/3` adds mode;
- `holds_via_rule/5` adds both rules.

Provenance is therefore part of the same relation, not a side table. Worlds are values in the `world` field; they never take a classid. The lens id comes from the OGAR codebook through hotplug, as in `dto_binding.rs`. The probe uses no classid.

A narrower selection can make two lens rows equal. In the diamond (F1), `p` holds in `d` by `inherited(a)` and by `inherited(b)`, so the lens has two rows but `holds(p,d)` is one fact. Mistaber's semantics are Datalog (set):

| Projection | Maps to |
|---|---|
| bag (column subset, multiplicity kept) | ALIAS to the selection |
| set (Datalog head) | the selection **followed by** a distinct/existence fold |

A lens row is one (mode, source rule), or one (mode) with empty rule fields when the holding has no rule. Two rows can also share a mode: `p` holds in `e` only by `inherited(d)` but traces to two source rules, so a bag projection gives two `holds_via(inherited(d),p,e)` and two `holds(p,e)` (`bag_counts_lens_rows_not_modes`; caught in review on #1873, where Bag first counted modes). `set_projection_is_required_on_the_diamond` pins both: bag multiplicity 2, set count 1, and the bag reading does not reproduce the golden. `bag_and_set_agree_where_no_holding_has_two_modes` is the silence twin. The same split was sent to the XLOG session for the #1866 crosswalk (EXO-50).

## 5. Evidence (D-MIST-NATIVE-0)

`cargo test --release --manifest-path crates/mistaber-kripke-probe/Cargo.toml`: 12 tests.

Goldens: F1-modes, F1-reversed, F2-cycle, F5-gates, F6-undeclared-child, C-mammal-C2/C4/C6/C6i, E-xlog-plain/override. All are exact, compared as sorted lists, so a duplicate atom would also fail.

Disable runs (each made after committing, red, then restored):

| Disable | Fails |
|---|---|
| drop `¬Block` in the fixpoint | `every_golden_is_reproduced_exactly` |
| count one bag row per mode instead of per (mode, source rule) | `bag_counts_lens_rows_not_modes` (`left: 1, right: 2`) |
| drop the `holds` gate on lineage | `lineage_never_leaves_the_held_cells` (`left: 3, right: 1`) |
| drop the world mask on direct holdings | `every_golden_…` (F5 ghost world) |
| drop the world mask on rule atoms | `every_golden_…` (F5) |
| moderate from any holding instead of direct | `every_golden_…` |

The lineage gate is also enforced at render time (`inherited_from` checks `holds` at the parent). This is why the golden comparison could not catch its removal, and why it has its own unit test. F5-gates was added to `mistaber-rs` because no earlier fixture exercised the world mask or that gate. F6-undeclared-child pins the opposite case: upstream's inherited rule has no `world(W_child)` condition, so a holding reaches an undeclared child world, and the probe deliberately applies no world mask on inheritance (found in review on #1873).

No timing was taken, and none is claimed. `mistaber-rs` §7 defers mask folds (class E) until a repeated workload exists.

## 6. Open questions

- **Q1 (blocks D-MIST-NATIVE-1):** `compose` lives in `dr-math-binding-probe`, a lab crate. A production `lance-graph-mistaber` cannot depend on it. Where should the Boolean relation register and `compose` live in production: ndarray's SIMD facade, the contract, or a native relation crate? `xlog-relcomp-probe` faces the same question; D-XLOG-NATIVE-3 asks the ndarray and mask-risc owners for a native home for its many-to-many join. Both should get one answer.
- **Q2:** the lens classid comes from the OGAR codebook via hotplug, and none is minted. Does it need a concept, or a lab stand-in as in `dto_binding.rs`?

## 7. Not in scope

A port of the HLL compiler or Clingo. Stable-model semantics. Any CE64 bit, any Epi5 code. Writes to `BrainyBlaze/mistaber`.

## 8. Mistaber on DR64³ (2026-10-30, lab test, no new type)

The probe's registers are DR64² in the D-DRDT sense: row = proposition (64), bit = world (64). They are read through the HHTL16 address `Morton8x8::from_xy(p, w)` (`dr-math-binding-probe::hhtl16::cell`). The lineage already lives on three logical 64 × 64 × 64 index spaces, each a stack of DR64² slices along a third axis:

| cube | third axis | cell `(z, p, w)` | stored? |
|---|---|---|---|
| V | parent world | `holds_via(inherited(z), p, w)` = `holds[p]∋z ∧ children[z]∋w ∧ ¬block[p]∋w` | no; it is the uncontracted iteration space of `compose`, built on demand |
| O | origin world | `inheritance_source(p, w, z)` | yes, `Holdings::origins` |
| L | rule | `holds_via_rule` source rule `z` of `(p, w)` | yes, `Holdings::sources` |

Contracting V or O over the third axis (OR over `z`) gives the `holds` register back: `direct ∪ ⋃_z V[z] = holds` and `⋃_z O[z] = holds`. L does not contract to `holds`, because a holding can have no rule (F1 `g` in `a`). Bag multiplicity is the count along the third axis of V × L. The Set projection is the contraction.

**Test:** `tests::the_third_axis_contracts_back_to_holds`, over all 11 goldens. It also checks that every cell of all three cubes reads the same through its HHTL16 address. Anti-vacuity: at least one cell has two parents (the F1 diamond) and at least one has two origins.

**Disable runs** (committed first, then each broken, red, restored):

- drop `¬block` from `inherited_from`: red, `E-xlog-override V`, `left: 3, right: 1`;
- drop the last origin world: red, `F2-cycle O`, `left: 0, right: 4`.

**Axis typing, a data point for NFR-OP-2.** The D-DR256 Amendment A working model reads DR #3 as Identity × Property, with the third axis indexing a relation ordinal for the Identity × Identity view. Mistaber needs two kinds of third axis:

- two cubes (V, O) have an Identity third axis: a world;
- one cube (L) has a rule ordinal, closer to a method or relation ordinal.

So "the third axis is a relation ordinal" covers L and not V or O. Closing NFR-OP-2 stays the DR owner's decision.

**Size.** Dense, each cube is 32 KiB of bits, 96 KiB for the three. The probe stores only O and L, and only as many slices as there are worlds and rules. V is never stored, because every V slice is one `compose` row read. Nothing here allocates a cube.
