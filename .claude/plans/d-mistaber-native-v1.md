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

`cargo test --release --manifest-path crates/mistaber-kripke-probe/Cargo.toml`: 18 tests + 2 doctests (since §9).

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

## 9. D-MIST-NATIVE-1a: HHTL16 rows replace the name interner (2026-10-30)

**Status:** LAB, in PR. The probe holds no names in its solved state. Names live only in `src/membrane.rs`, which parses the goldens and renders atoms.

- **Rows** (`src/rows.rs`). World, proposition and rule nodes are 512-byte `NodeRow`s in the HHTL16 opt-in format. `Register0` holds the x bytes and `Register1` the y bytes. Column k is `Morton8x8::from_xy(R0[k], R1[k])`. Column 0 is the node's identity; columns 1..=15 are addresses, which for a world are its parents. Address = first-appearance ordinal + 1, so no two collide and none is unbound.
- **Classid per population.** Each population is bound once: `Activation` → `resolve_for_context(concept, Register128 slab)` → `bind_register128(Two)`. The concept lives on `RegisterLanes::concept()`. The key, edges and value bytes carry no classid. Concepts are lab stand-ins `0x7F11..13`; none is minted.
- **Accessibility is derived** from world rows (`Kb::derive_children`, its only writer). A 16th parent is refused with `TooManyParents { parents }`. Global address → local 0..64 goes through `hhtl16::LocalMap`, which refuses a 65th node, a duplicate or an unmapped address.
- **Binding.** The accessibility register and the three input registers are bound through `bind_dr64`. Every `holds ∘ children` step runs through `relcomp::Chain::new(..).eval(..)`. The fixpoint and lineage algebra are unchanged.
- **No-String guard.** A doc-hidden `AddressOnly` trait plus an exhaustive destructure of `Kb`, `Holdings`, `Mode` and `Population`. A String field fails to compile, and two `compile_fail` doctests show the bound can fire.

**Tests:** 18 tests and 2 doctests. All 11 goldens stay exact. New tests cover:

- rows round-tripping through `get`/`set`;
- a two-parent world deriving the old accessibility;
- the 16th parent being refused;
- the classid being on the lanes and not in the rows;
- `LocalMap` refusals.

**Disable runs** (committed first, each then broken, red, restored):

| # | breakage | red |
|---|---|---|
| D1 | read only parent column 1 | goldens, diamond Set/Bag, two-parent derivation, contraction, classid test |
| D2 | truncate parents instead of refusing | `a_16th_parent_is_refused` (`left: None`) |
| D3 | classid minted into the key | `the_classid_lives_on_the_lanes_not_in_the_rows` |
| D4 | accessibility signature typed Property × Identity | 12 tests (the `Chain` typing refuses `holds ∘ children`) |
| D5 | address without `+1` | round-trip, goldens and 7 others |
| D6 | accessibility orientation reversed | goldens, orientation, contraction and 5 others |
| D7 | a `NameTable` field on `Kb` | compile errors E0063 and E0027 (exhaustive destructure) |

Known limit: replacing `hop` with raw `compose` turns nothing red. The binding is a typing guard, not semantics, and D4 is the only proof that the path is live.

**Open (operator):**

- the row key is all zero, so identity exists only in column 0. Should the key also carry the address?
- a real binding takes concept ids from the OGAR codebook through hotplug.

## 10. D-MIST-NATIVE-1b: makor citations addressed the KJV way, no classid (2026-10-31)

**DECISION (operator, 2026-10-31):** no classid for Mistaber nodes. They are addressed with the existing KJV path: `deepnsm-v2` `toc`/`hydrate` gives each node a `NiblePath` from the outline. `promote`, `key_at` and `NodeRow` are not used. Draft PR #1879 (address in the row key) answered a question the operator selected by mistake. It was closed unmerged, and its two commits are reverted on the branch.

- **Fixture:** `crates/mistaber-kripke-probe/fixtures/makor.tsv` holds 141 real `makor(rule, sa("yd:siman:seif"))` citations from the compiled Mistaber ontology (`AdaWorldAPI/mistaber@b50bf33`). They cover 10 seifim: 87:1, 87:3–9, 89:1 and 100:6.
- **Mapping** (`src/makor.rs`): Yoreh De'ah is book 0, siman is the chapter and seif is the verse. Each citation is a `Spo(rule ordinal, CITES, seif ordinal)` laid at its seif's verse node. Names exist only in `parse_makor`.
- **Refusals:** a siman past the 255-per-level capacity has no node, and its citations are counted in `Hydration::unaddressed`, never folded into another node. More rule or seif ordinals than a `u16` holds are refused, never saturated.
- **Dependency:** the probe depends on `deepnsm-v2`, a standalone lab crate, not an OGAR concept. The only global piece used is the contract's `NiblePath`. If the generic part (outline → spawned tree → addressed items) is ever needed outside a lab, it belongs beside `NiblePath` in the contract, with both `deepnsm-v2` and Mistaber consuming it.

**Tests** (`tests/makor.rs`, 8):

- all 141 citations are addressed, and each lands on its own seif's path;
- containment reads as prefix ancestry: 87:1 and 87:3 sit under the siman-87 node, 87:x and 89:1 share only the book node, and the book node is an ancestor of all 10 paths;
- the 10 seifim get distinct paths;
- siman 403 is refused into `unaddressed`;
- reading order is the fixture order;
- a seif can carry many citing rules;
- malformed lines are refused;
- the 65,537th rule ordinal is refused.

Full suite: 25 tests and 2 doctests.

**Disable runs** (committed first, each then broken, red, restored):

| # | breakage | red |
|---|---|---|
| M1 | `path_of` ignores the siman | 5 tests (addressing, containment, distinctness, refusal, order) |
| M3 | rule ordinals saturate instead of refusing | `rule_ordinals_past_a_word_id_are_refused_not_saturated` |
| M4 | markers swap siman and seif | 5 tests |

M2 is a recorded limit, not a failure. Dropping the seif dedup turns only the ordinal test red: repeated markers resolve to the same node, so the dedup is representation, not addressing.

**Note on ancestry granularity:** `NiblePath::common_ancestor` works per nibble, and each level spends two nibbles. So two seifim of one siman can share an interior nibble below the siman node. Tests assert level-node ancestry (`is_ancestor_of`), not equality of the common ancestor.
