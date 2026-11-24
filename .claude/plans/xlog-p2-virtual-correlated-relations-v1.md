# XLOG P2 — virtual correlated n-ary relations (contract v1)

Status: CONTRACT, docs only, 2026-11-24. Claim: EXO-395 under EXO-377. No implementation until ratified.
Evidence labels: VERIFIED-IN-CODE (file:line read today), MEASURED (cited entry), HYPOTHESIS, MISSING, UNVERIFIED.
Line numbers: `cube.rs` cites are for the 1862-line file on `ccr-85acaff0-oszq40-setexpr-column` (main is 1836 lines; the difference is `SetExpr::Column`, EXO-394). `body.rs` cites hold on both.

## 1. Problem

- The xlog-cube adapter keeps relation state as `type Rows = BTreeSet<Vec<u32>>` (xlog/crates/xlog-cube/src/lib.rs:194).
- The Cube holds binary relations (pair planes) and one-column sets (cube.rs:399, `PairPlane` in dt64 dr/pair.rs:1-20; `add_set` cube.rs:540-ish).
- MEASURED: `census-097c351.tsv` (PR #2168) has 128 refused rules; `.claude/plans/xlog-reactor-swap-inventory-v1.md` §6 classes them, and 68 = 39 (n-ary or computed head column) + 25 (n-ary read) + 4 (n-ary join key) are the P2 candidates before cascades. The class table is a reading of reason strings, not a per-rule proof.
- The 007bb19 census (entry 2026-11-17-d-xlog-w7-p0-census.md) counted 39 wide rule heads: 22 joins, 11 one-atom heads, 6 computed-column heads. Its first reading ("a named sink of position tables") was struck by the operator the same day: wide heads are virtual tuple bags over the cube, rows only at the edge.
- STATUS_BOARD row D-XLOG-W7 says the same: "virtual tuple bag over the cube (a description, never stored)". I found no entry containing "EXO-330" or the phrase "folded, never stored". UNVERIFIED: the source of that wording.
- What exists: `Cube::body_answer` and `Cube::materialize_witnesses` (cube/body.rs:533, :569). They answer one conjunctive body. They do not name it, reuse it, or let another rule read it.

## 2. Definitions

A **recipe** is plain data:
- `atoms`: a list of `Atom { rel: RelId, vars: [u8; 2] }` (body.rs:41-45), binary only.
- `k`: number of variable classes, `1..=MAX_BODY_VARS` (8, body.rs:38).
- `head`: an ordered list of variable-class indices. Repeats are refused (see §6). Classes not in `head` are existential.

A **virtual relation** is `(recipe, cube id, frame issue)`. It holds no ids, no rows, no masks.

Forbidden representations (each is a stored population or a disguise of one):
- `BTreeSet<Vec<u32>>`, or any row buffer, as the relation's state.
- A per-column table of positions, one entry per head column, held after the terminal returns.
- Any reconstruction of the relation from per-variable support masks.

Why independent supports lose the relation. Take `R = {(a,x), (b,y), (c,z)}`.
- Column 0 support is `{a,b,c}`. Column 1 support is `{x,y,z}`.
- Rebuilding `R` as support0 × support1 gives 9 pairs. `R` has 3.
- The per-variable support in `BodyAnswer.support` (body.rs:75-80) is exactly the projection that loses the pairing. It is exact for one variable and wrong as a description of two.
- Corpus-shaped twin (use in tests): `e1={(0,1),(3,4)}`, `e2={(1,2),(4,5)}`, `e3={(0,2),(3,5)}`. `tri` has 2 witnesses `(0,1,2)`, `(3,4,5)`. Supports are X={0,3}, Y={1,4}, Z={2,5}, so a product rebuild claims 8 tuples. The binary head `p(X,Z)` has 2 pairs; a product rebuild claims 4, including `(0,5)` and `(3,2)`.

## 3. Five layers, kept apart

| layer | holds | never holds |
|---|---|---|
| recipe | atoms, variable classes, head selection | ids, positions, rows |
| coordinate binding | `MapTable` / `LocalMap` (dr.rs:576-650) and the cube's `Frame` (cube.rs:399 field list) | any tuple |
| base relation plane | the cube's `PairPlane` per relation, both directions, lent without copy (body.rs:166-172 `plane.rows`) | derived relations |
| demanded terminal | one answer: existence, count, support, pair-projection | a reusable row set |
| egress | `materialize_witnesses` (body.rs:569), the only named row builder, counted in `stats().rows_materialized` (body.rs:586-590) | nothing after return |

Binding rules:
- A recipe's `RelId`s are cube-local (order of `Cube::new` / `add_relation`). A virtual relation is therefore bound to one cube id. `View` already carries the cube id (cube.rs:133); a foreign cube is `CubeRefusal::ForeignView` (cube.rs:260, raised at :703).
- Every terminal re-checks the frame (`check_frame`, body.rs:541, cube.rs:800). A recipe survives `add_edge` / `retire_edge` (cube.rs:1182, :1203) with no maintenance, because it stores nothing. The next terminal reads the effective lanes and planes, which the cube drops and rebuilds on a logged edge (cube.rs `planes` / `effective` field docs).
- After `rebase` or a map-growing `commit`, the recipe is unchanged; only the frame check can refuse.

## 4. Composition: unfolding

A rule that reads a virtual relation substitutes its recipe (body unfolding, inlining).

Procedure, per reader atom `V(t1..tn)` of a virtual relation with recipe `(atoms, k, head)`:
1. Rename the recipe's variable classes apart from the reader's variables.
2. Identify head class `head[i]` with reader term `ti`.
3. Existential recipe classes get fresh reader variables (a reader `_` is a fresh existential variable).
4. Append the renamed atoms to the reader body.

Lawful when:
- the virtual relation is non-recursive and sits in an earlier stratum than the reader;
- every use is positive (no negation, no aggregate over a virtual relation: complement is not in the cube, and `Diff` is already refused per d-xlog-native-v1.md Amendment E);
- set semantics (§7);
- after substitution every atom is binary over two distinct variables, the total class count is at most 8, and no constant argument remains.

Typed refusals (new `VirtualRefusal`, none falls back to a row store):
- `Recursive { scc }`: any strongly connected component through a virtual n-ary relation. This is P4 (generic monotone SCC, EXO-73; STATUS_BOARD cites EXO-73 as a gate for D-CGMM-2 / D-DRISA-7). P2 only detects and names the SCC.
- `Diagonal`: unfolding produces `rel(V, V)`; `Plan::new` already refuses it as `SameVariable` (body.rs:146).
- `Constant`: a constant at a virtual column. The body grammar has no constants (`Atom` is two variables).
- `TooManyVars(k)`: `BodyRefusal::Arity` (body.rs:134-136).
- `NegatedVirtual`, `AggregateOverVirtual`.
- `NonBinaryBase`: a base atom of arity other than 2.

Multi-rule heads (a virtual relation with several rules, like `tri` with two rules) unfold to a union of bodies:
- existence: OR. Single-variable support: OR of the masks, exact.
- Pair projection: OR of the pair planes, exact.
- Full-tuple count: not additive, because bodies can overlap. Typed refusal `CountOverUnion` unless bodies are proven disjoint. UNVERIFIED how often this arises in the corpus.

## 5. Demanded terminals

| demand | existing operation | status |
|---|---|---|
| existence | `body_answer(..).count != 0` (body.rs:533) | VERIFIED-IN-CODE; computes the full count, no cheaper path |
| count of tuples, head = all variables | `body_answer.count`: distinct bindings of all variables (body.rs:21-22) | VERIFIED-IN-CODE |
| count of tuples, head drops a variable | none | MISSING. Witness count over-counts: `r1(X,Y) :- tri(X,Y,Z)` counts every `Z`. Needs ∃-elimination (below) |
| one variable | `body_answer.support[v]` | VERIFIED-IN-CODE, exact (body.rs:23-24) |
| two variables (binary head, e.g. `p(X,Z)`, `r1`, `path01`) | none | MISSING |
| rows at the edge | `materialize_witnesses` then project and de-duplicate | VERIFIED-IN-CODE for witnesses; the projection is caller-side and labelled egress |

Binary head projection, classified.
- It needs correlation: the answer is `{(x,z) : ∃ the rest. body}`. Per-variable supports cannot give it (§2).
- Nothing existing answers it without enumeration. `body_answer` has no way to pin a variable (no constants in `Atom`). `Cube::bind`/`join` handle two sides only (cube.rs:718, :745). `closure`/`reach` are recursive binary terminals, not body projections (cube.rs:630, :1109).
- Classification: a primitive and grammar gap, not a mathematics gap. The Boolean semiring answer is well defined: existential elimination of a variable is an OR over its candidates. The walk already has the pieces: row masks per atom and direction, candidate AND, popcount at the leaf (body.rs module doc :1-30).
- Candidate shape (HYPOTHESIS, to design in P2): `body_pair(table, atoms, k, [u, v])` returns one row mask of `v`-candidates per position of `u`: a derived binary relation, the same kind of object as `Closure { rows }` (cube.rs:349). A binary result is representable; this is not a tuple store. Count of distinct head pairs is the popcount of that plane.
- Special case, HYPOTHESIS: a chain with a closing edge (`tri`'s `(X,Z)` = `(e1∘e2) ∧ e3`) is a Boolean matrix product masked by a plane. d-dynbag-holographic-bags-v1.md §7.2-7.4 names `ndarray::simd::matmul_i8_to_i32` as an unmeasured candidate and notes it copies inputs. Not general; not for cyclic bodies with two or more eliminated variables.
- Cost: UNVERIFIED. A per-position walk could cost `n` walks of one body. Measured today: `tri` 22-109 µs per full count, `cyc` 45-532 µs (entry 2026-11-17-d-xlog-w7-bodies.md). `n` such walks may reach milliseconds, which the no-millisecond rule (lance-graph CLAUDE.md) treats as a failed gate. P2 must measure before admitting; if over the gate, the terminal is a typed refusal and the head stays refused.

## 6. Scope limits inherited

- One tile: at most 256 ids across all relations. A tiled cube is refused (`CubeRefusal::Tiled`, body.rs:540, :576). Module doc: body.rs:28-30.
- At most 8 variables (body.rs:38). Binary atoms. No repeated variable in an atom (body.rs:146). No unused variable (body.rs:148-150 `Unused`).
- `clique5` has 5 variables and 10 base atoms: inside the limits. UNVERIFIED: whether any atom-count cap exists; `Plan::new` shows none in the lines read.
- P3 (tiles) would need: a global → (tile, local) index (d-dynbag §17.5 MISSING-7, line 526; today `MapTable::locate` is linear in tiles, STATUS_BOARD D-XLOG-W7-2); a component product that crosses tiles; a tiled variant of the lent plane rows (`new_tiled`, cube.rs:487, holds wide coordinates). Out of P2.

## 7. Set versus bag

- Base relations are Boolean: `body_answer` counts distinct bindings, "Boolean per base relation" (body.rs:22).
- XLOG heads are Datalog sets. The recursive-dispatch falsifiers compare relations as sets of tuples. I read no falsifier in the list that needs a multiplicity (ℕ) count. UNVERIFIED for the full 128 refused rules.
- So P2 is Boolean throughout. Any ℕ demand (witness multiplicity per head pair, `count` aggregate over a virtual relation) is a typed refusal.
- EXO-108 is kept separate. I could not find its definition in the files read. UNVERIFIED what it covers.

## 8. Falsifier plan

Oracles: naive AST oracle `xlog-coverage-probe/src/naive.rs` (`check_rule` :66, `evaluate_component` :249) and the Datafrog arm (`datafrog_arm.rs`, rows are `Vec<u32>` at :47). Both are row-based references. The lance-graph CLAUDE.md says answers are compared by folds, not by expanding rows. Reconciliation (proposed): oracles may enumerate; the cube side is compared by count and supports; rows are compared only in the egress test. UNVERIFIED that the board accepts this.

Corpus falsifiers as written (xlog fixtures, line numbers from the brief; text read at these lines):

| falsifier | shape | P2 verdict |
|---|---|---|
| recursive_dispatch :301-303 `tri`↔`echo` | SCC {tri, echo} | needs P4; P2 refuses `Recursive` |
| :413-415 `cyc`↔`echo` (4 vars) | SCC | needs P4 |
| :504-508 `r1`,`r2` ← `tri`, `tri` ← `r1`,`r2`,`r3` | SCC through virtual `tri` and binary heads | needs P4 |
| :660-662 `e1` ← `tri`, `tri` ← `e1`,`e2`,`e3` | a base relation becomes recursive | needs P4 |
| :967-969 `p(X,Z) :- tri(X,Y,Z)`, `tri :- p,p,q` | SCC | needs P4 |
| clique_dispatch :651-657 `path01` ← `clique5` ← `path01` | SCC, 5 vars | needs P4 |
| test_sort_labels.rs:30-37 `support_1`, `usable` | non-recursive, but base `external_consumer_commit` has arity 3 | outside P2 (`NonBinaryBase`); needs a ternary base plane or a decomposition |

Every corpus falsifier in the brief is a recursive SCC. None is admitted by P2 as written. P2's own gates therefore use non-recursive derivations of the same bodies:
- G1 `tri` alone over frozen `e1,e2,e3` (the first rule of :301). Terminals: existence, count, support X/Y/Z, egress rows. Oracle: naive + Datafrog.
- G2 `r1(X,Y) :- tri(X,Y,Z)` as a reader of G1: unfolding with an existential variable. Binary head: only if §5 `body_pair` lands; else typed refusal pinned.
- G3 `cyc` (4 vars) and `clique5` bodies without the recursive rule (the base step of :413 and :651-657).
- G4 `echo(X,Y,Z) :- tri(X,Y,Z)` as a pure reader (alias with renaming, including a permuted head such as `tri(X,Z,Y)`).
- G5 multi-rule head: union terminals (existence, support) pass; `CountOverUnion` pinned.
- G6 recursion detector: each row of the table above must return `Recursive` with the right SCC members. This is the can-fire test; G1-G5 non-recursive cases are the can-stay-silent twin.

Action for the census: add a column "SCC-recursive through an n-ary head" to the xlog-coverage-probe census so the 68 splits into P2-admissible, P4, P3 and out-of-scope. UNVERIFIED how the 68 splits.

Negative twins (correlation-losing):
- The `R` fixture of §2 and the 2-triangle twin. Test asserts count 2 (not 8), and for `p(X,Z)` asserts 2 pairs (not 4).
- Same supports, different pairing: swap `e3` to `{(0,5),(3,2)}`: supports unchanged, witnesses 0. A support-only answer cannot tell the two cases apart.

Disable runs (commit first; assert each anchor applied, per the repo rule; read the exit status and the `test result:` line, not a grep):
1. Skip variable renaming in unfolding → two readers share a class → wrong count (G2/G4 red).
2. Rebuild a head from supports → twin tests red.
3. Drop the head selection (treat every class as head) → G2 count over-counts (red).
4. Unfold a recursive SCC instead of refusing → G6 red (or non-termination: add a depth bound so it fails, not hangs).
5. Skip the frame re-check → stale-frame test red.
6. Let the egress projection skip de-duplication → G2 rows count red.
A disable that does not bind is not evidence: confirm each by the red run, as with the other body falsifiers (entry 2026-11-17-d-xlog-w7-bodies.md table).

Also required: stats show no rows built by any non-egress terminal (`rows_materialized` unchanged), as `tests/body.rs` already does for the count.

## 9. Open questions

1. Where do recipes live: `lance-graph-xlog` (owns semantics per Amendment E, d-xlog-native-v1.md:206-208) with the xlog-cube adapter only translating. Proposed yes; needs ratification.
2. Multi-rule heads and pure aliases: `tri :- echo`, `echo :- tri` adds nothing beyond `tri`'s other rule. Collapsing such alias SCCs would admit :301 and :413 without P4. That is a rewrite beyond unfolding; deferred, not specified here.
3. `body_pair` cost gate (§5). If it fails the no-ms rule, which binary heads stay refused?
4. The 5 ILP heads, the 6 computed-column heads (`is`), and `sales_by_cat_region` are out of this contract (census entry; group-by already runs, tests/group_by.rs).
5. Ternary base relations (`external_consumer_commit`): a ternary plane, or a decomposition? Not P2.
6. Atoms with a constant or repeated variable (diagonals): refuse now; could be a `bind`-style selection later.
7. Oracle policy for row-based references (§8).

## 10. Not verified

- The source of the wording "folded, never stored" and the EXO-330 reference (§1).
- EXO-108's exact scope (§7).
- Whether any atom-count cap exists in `Plan::new` beyond the lines read (§6).
- `body_pair` cost (§5): no measurement exists.
- How the 68 split into P2-admissible, P4, P3 and out-of-scope (§8): needs the new census column.
- Whether the board accepts row-enumerating oracles for the cube comparison (§8).
