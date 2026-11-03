# D-BAG-AWARENESS · the `Cube::side()` boundary — is the native algebra closed over virtual views?

Read-only. lance-graph `514948203`. Grades: CODE (coordinator read the lines), TEST (executed here), SCOUT, DECISION, INFERENCE.

## Verdict

Composition of virtual relations is **not blocked by mathematics**, and an existing lab abstraction already evaluates `(R∘S)∘T` (and every association order) entirely in registers, against an independent oracle, with no relation registration and no tuples — for the **Boolean** semiring only. What stops the production path is three distinct things, each with its own owner:

1. **Cube scope (implementation):** `Cube::side` admits only a resident relation or a bound scan as a join operand; a `Join` node is `NotASide` (`lance-graph-xlog/src/cube.rs:422-433`). Nothing in the cube's own contract says why; the module never claims closure over views.
2. **An explicit algebra law (mask-risc / Quack):** a scattered mask "is never an intermediate: a follow-on program or fold that consumes it is the forbidden projection → population → projection shape … a filter through the targets is `Pred::EqU32Via` / `MaskOp::Gather` over a RESIDENT plane" (`lance-graph-mask-risc/src/ir.rs:334-342`). Quack restates it as the precondition of `Filter::Semijoin`: "`foreign` names a RESIDENT plane, never a produced one … a `Keep`-then-`Gather` pipeline across two programs is exactly the forbidden shape" (`lance-graph-quack/src/lib.rs:376-386`); cypher-quack enforces it as a refusal ("WHERE with a hop: … Keep→Semijoin is forbidden", `lance-graph-cypher-quack/src/lib.rs:299-300`).
3. **Semiring scope (explicit, lab):** associativity is admitted only for Boolean composition; Bag (ℕ) and three-valued composition are refused (`LawRefusal::NotBooleanSemiring`, `dr-math-binding-probe/tests/relcomp.rs:266-290`), and a chain of ℕ products is a typed refusal "until a word exists" because inputs exceed `i8` after one hop (`.claude/plans/d-dynbag-holographic-bags-v1.md` §7.3).

## 1. Operations that accept resident sides only

| operation | operand | evidence |
|---|---|---|
| `Cube::scan` | `RelId` | `cube.rs:367` |
| `Cube::bind` | a `scan` view only (`NotASide` otherwise) | `cube.rs:376-399` |
| `Cube::join` | each side scan or bind; `Join` → `NotASide` | `cube.rs:403-433` |
| `Cube::closure` / `closure_wide` / `refresh_closure` | `RelId` | `cube.rs:669, 719, 810` |
| Quack `Filter::Semijoin` / mask-risc `MaskOp::Gather` | foreign plane must be RESIDENT | quack `lib.rs:376-391`, ir `:277` |
| mask-risc `Planes` | the table's own lanes + caller-supplied input planes | ir `:73` |

## 2. Operations that accept lazy views, registers, masks or expression handles

| operation | accepts | evidence |
|---|---|---|
| `Cube::support` / `bag_count` / `materialize_rows` | one lazy `View` (`Node::Join` of two resident sides) | `cube.rs:642-664, 1010` |
| `bag_count_admitted` | an admitted `MethodBinding` + a two-sided `Intake` | `lance-graph-xlog/src/lib.rs:384` |
| codegraph `impact` | closure registers (`WideRows`), folded with `mask_and` / `mask_andnot` / `intersects` | `lance-graph-codegraph/src/query.rs:239-285` (PROD) |
| `relcomp::Chain::eval(&RelExpr)` | n operands bound to methods, any association order, `[u64; 64]` registers | `dr-math-binding-probe/src/relcomp.rs:120-222` (PROBE) |
| `ncompose::pair_counts` | two DR64² tiles → ℕ counts per `(x, z)` (one hop) | `xlog-relcomp-probe/src/ncompose.rs` (PROBE) |
| `dir-rmap-probe` `Composer::compose` | membership × carriers over one slot coordinate; `Law::Count` and `Law::Exists`; seal + `Stale` witness | `dir-rmap-probe/src/compose.rs:1-40, 250-300` (PROBE) |

## 3. The exact first operation forcing `R∘S` to become physical before `(R∘S)∘T`

- In the cube: `Cube::join(view_of(R⋈S), …)` → `side()` → `Err(NotASide)` (`cube.rs:432`). The XLOG polyfill's answer is `materialize_rows` (`xlog-cube/src/lib.rs:747`) → `add_relation` (`:206`).
- In the mask-risc/Quack algebra: the step from B-space to C-space. A two-way join's support is computed entirely in its join-key space (two scatters + AND, `cube.rs:642-650`). Moving on to `T` needs the rows of `S` whose `B` is in that mask — a gather through a **produced** mask, which ir.rs:334-342 forbids. The legal route requires the intermediate projection to be a **resident** plane.
- Note: `Cube::support`'s own `mask_and` of two scatter results is already the shape the Nov 2 archaeology flagged as conflicting with this law (decision D3, "Scattered mask in `support`", OPEN).

## 4. Classification of the restriction

| restriction | kind | owner |
|---|---|---|
| `NotASide` | implementation scope of the cube (`Side = (rel, bound)`) | `lance-graph-xlog` (Amendment E: it owns the semantics) |
| produced mask never an intermediate | **explicit algebra law** (mask-risc), enforced in Quack and cypher-quack | mask-risc (frozen absent operator decision and evidence, EXO-105); D3 open |
| Gather / EqU32Via call sites | admission, unclassified, **no owner named** | EXO-140 (ADMISSION B) |
| counted global→focus membrane | operator admission | EXO-46 |
| Bag chain (counts feeding a product) | carrier limit (i8), typed refusal | D-DYNBAG §7.3; EXO-108 (degree-product / foreign-value sum, law 6) |
| associativity outside Boolean | explicit law refusal | `admit_composition_assoc` (lab) |
| iterated composition (recursion) | open decision | EXO-73 |

Not mathematical: composition is associative in every semiring involved (Boolean and ℕ), and `(R∘S)∘T = R∘(S∘T)` is exactly what relcomp's tests and the egglog proof of every five-chain re-association check.

## 5. Can a DT64-admitted method consume the composition virtually?

- Two-way: yes, `bag_count_admitted` (admitted `MethodBinding`, `Population::Bag`, steps `GroupCount, GroupCount, MulI64`), any other step shape → `BoundRefusal::Shape` (`lib.rs:395-397`). Callers: probe + tests.
- n-way: the lab `Chain` binds each operand to a method (`BoundDr64 { method: MethodSignature, frame, focus, scope }`, `dr-math-binding-probe/src/lib.rs:217-224`), but `MethodSignature` is a LAB identity listed for removal in the D-DT64-AUTH census (`lance-graph-r2il/dt64-authority-census.tsv`), not a DT64 `MethodRef`. No production DT64 path consumes a chain.

## 6. One expression, several readings

| reading | two-way virtual join (cube) | n-way chain (lab `Chain`) |
|---|---|---|
| Boolean existence | `support` | `eval` → rows; `any` |
| Bag count | `bag_count` (degree product) | refused (`NotBooleanSemiring`); ℕ only one hop (`ncompose`) |
| restriction | `bind` (`col == const`) | operand rows are the caller's; validity plane `BoundDr64::validity` |
| further composition | refused (`NotASide`) | yes, any order |
| difference / negation | `mask_andnot` on the result (codegraph, host side) | not a chain operator |

## 7. Versions, deletions, speculative frames

- **Version change:** every cube terminal re-checks its frame (`check_frame`, `cube.rs:458`); a lab `Chain` borrows the `FrameStore`, so advancing the store while a chain is alive does not compile (doc test, `relcomp.rs:124-133`), and a stale issue is `StaleFrame`. Virtual results are recomputed, never patched.
- **Deleted support:** a retire is logged and merged on read; the effective lanes cache is dropped (`invalidate`, `cube.rs:577`), so `support`/`bag_count` see the deletion with nothing to retract. Derived registers that *are* cached (codegraph closures) are refreshed or dropped at commit (`graph.rs:903-912`); a closure row is recomputed for every source that could reach the retired edge (`cube.rs:760-762`).
- **Speculative state:** `Cube::stage` checks a batch "without changing it or the table … no version is issued" (`cube.rs:849-855`), and `stage_closure` gives "the closure of relation `rel` as it will be once `staged` is committed … Changes nothing" (`cube.rs:904-908`): a virtual relation evaluated on an uncommitted state. That is the closest existing THOUGHT-over-sealed-WORLD shape; Thought-Git proper is probe-only.

## 8. Falsifier (proposed, not implemented)

`R(A,B) ∘ S(B,C) ∘ T(C,D)` → existence and count, against an independent oracle.

- **Existence:** relcomp `Chain` with three operands and both association orders vs a scalar oracle; assert equality, non-trivial fixtures (neither empty nor full), and `CubeStats.rows_materialized == 0` / no `add_relation` on any cube involved. Part of this already exists: `every_order_computes_the_same_relation` runs 4- and 5-chains over all 5 / 14 orders. **Run here (TEST):** `cargo test -q --manifest-path crates/dr-math-binding-probe/Cargo.toml --test relcomp` → `13 passed; 0 failed` (lance-graph `514948203`; ndarray sibling `b3d7919` via a temporary symlink, removed after the run, as was the probe's `target/`). That includes the chain-vs-oracle, refusal, Boolean-only associativity and egglog five-chain tests. It does NOT include the 3-relation count arm or the cube arm, which do not exist yet.
- **Count:** the same chain under ℕ. Expected today: typed refusal (`NotBooleanSemiring`, D-DYNBAG §7.3). A pass would require a word that does not exist (foreign-value sum / wider product); the falsifier's job is to show the refusal is the only blocker, by checking `(A·B)·C` against enumeration with a wide-integer oracle in the test itself.
- **Through the cube:** the same fixture via `Cube::join(join(..), ..)` must refuse with `NotASide` — that is the boundary under test, and it must stay red until an owner decides otherwise.
- Disable runs: swap two operands of a non-commutative chain; drop the frame check; feed a produced mask into `Semijoin` (Quack must refuse).

## 9. Does an existing abstraction already resolve it?

For Boolean existence: **yes, in the lab** — `relcomp::Chain` + `RelExpr` (register-only, frame-bound, method-bound, associativity admitted and egglog-proved). It is not wired to `Cube`, DR³ tiles beyond 64, or DT64 `MethodRef`. For counts: **no** — refused by law and by carrier, and recorded as gaps (EXO-108; cypher-quack "two or more hops: the walk count needs a foreign-value sum").

No new expression engine or view type is justified by this evidence: the Boolean chain abstraction exists; what is missing is its admission into the production identity (DT64) and a decision on the produced-mask law (D3 / EXO-140).

## 10. Governance addendum (board/plan text, re-read by the coordinator)

These sharpen §4 and change §9's verdict on the lab chain.

- **The exact question already has a pending contract.** `dr256-where-capstone-v1.md:40-44`: "A `Keep` mask is a legal projection-space mask: a later program may read it as a plane (`Filter::Plane`). A `ScatterOrU32` mask may not (`ir.rs:329-337`). … The #1791 contract proposal (a witnessed derived population) is the only route to reusing a scattered mask, and it is still not authorized." → A virtual intermediate that a second fold may read is precisely a *witnessed derived population*; its owner route is #1791, unauthorized.
- **The lab chain is itself admission-open.** `ISSUES.md:210` (#1896): `relcomp::compose` evaluates XLOG's linear transitive closure on the DR64 register, "Whether `compose` counts as co-addressed under ADMISSION B is not decided here." And `d-onnx-relalg-cap-capstone-1750-v1.md:474` OP-14: "Does the DR64 `compose` lower to mask-risc without `MaskOp::Gather`? … arm B result `NOT EXPRESSIBLE` … report, do not add an opcode | OPEN". So `Chain` proves the mathematics and the oracle equality (13/13 here), but it runs as a host loop (`acc |= b[bit]`, an indexed read per set bit) that mask-risc cannot express without `Gather`, and its admission is undecided.
- **The gather route is closed by ruling.** The gather ban (#1590, 2026-10-16) and ADMISSION B (`entries/2026-10-20-six-decisions.md` §1): no cross-table value-derived positional reads; Semijoin / EqU32Via / Gather call sites unclassified, no owner (EXO-140, `ISSUES.md:122`). `OutgoingHop` (two `Gather` + `ScatterOrU32`) is quarantined behind an off-by-default feature (`STATUS_BOARD.md:426`).
- **The lawful shape the board already names.** `entries/2026-10-16-synergy-map-since-1579.md:56`: "a hop is a coordinate range and chained hops are range intersections (`fold_ranges`, already shipped); a join is a layout that puts the pair at one coordinate." `cypher-mask-lowering-v2.md` RF-CHAIN: a chain of push hops needs "either a mask-risc ruling that admits a scattered mask as a resident plane, with its own law and falsifier, or a pull lane on the target rows."

**Revised §9.** For `R∘S∘T` there are exactly two lawful routes on record, both unbuilt for this case:
(a) **co-addressed layout** — the join is a layout; chained hops become range intersections over `fold_ranges` (shipped primitive) or pull lanes on target rows; no produced mask is consumed;
(b) **#1791 witnessed derived population** — an intermediate becomes readable by a second program under a witness; unauthorized.
The lab `Chain` is evidence for the semantics, not an admitted route (OP-14 / ADMISSION B open). `Cube::side`'s `NotASide` is therefore consistent with the standing law, not an accident to remove: lifting it to accept a join view would require one of (a) or (b).
