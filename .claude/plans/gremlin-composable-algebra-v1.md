# gremlin-composable-algebra-v1 — Gremlin steps as typed building blocks of ClassId methods (D-CMH-G1)

> **Status:** MEASURED for the slice in §2 (code: `crates/lance-graph-glove-parity/src/{method,gremlin,decl}.rs`,
> tests `tests/{compose,gremlin}.rs`). The vocabulary matrix (§4) is ANALYSIS against code and tests,
> one row per step family. Stacked on #1507 (runtime declarations); independent of #1506.
> **Ledger:** `.claude/board/entries/2026-10-12-class-method-polyfill-ledger.md`.

## §0 — Question and answer

Can Gremlin's vocabulary be a set of typed building blocks from which ClassId methods
are composed, with no second interpreter and no new instruction? **For one population,
yes.** A method can name another method; Gremlin steps compose into a method body; the
body goes through the same preparation and Quack lowering as every method (#1504); and
it can be loaded as text into a binary that was compiled before the text existed.

The result to look at: `billable_hours` calls `is_billable`, a separately declared
predicate method. Both are text in `tests/fixtures/lego.methods`, loaded at run time.
There is no Rust for either method. `billable_hours` lowers to exactly the `Program`
of the direct Quack query (`SUM(hours) WHERE billable = true`), and runs on both frontends.

## §1 — What already shipped (not this PR)

| piece | where | what it gives this PR |
|---|---|---|
| #1311 Gremlin↔SQL witness | `crates/lance-graph-quack/tests/gremlin_parity.rs` (test-only, 13 tests, re-run green on this tree) | the anchor rule: one traverser = one anchor row; functional hop keeps the anchor, one fan-out re-anchors on the path rows; bag and set kept apart |
| #1312 corrections | `.claude/plans/population-law-crosscheck-v1.md` | a second fan-out is not a barrier by hop count; the barrier is a fan-in aggregate carried forward |
| #1504 ClassId methods | `src/{method,call}.rs` | `MethodSpec`, `MethodTable::prepare` (seal, reading, claims, laws, Quack lowering once), two frontends |
| #1507 runtime declarations | `src/decl.rs` | owned declarations lent to `prepare` |
| #1506 compute DAG over 256 positions | contract | prerequisite for derived fields; this PR adds none |

## §2 — What this PR adds

1. **Predicate methods and method references.** `MethodBody::Select { filter }` folds nothing.
   `Cond::Method(MethodRef::{Key | Label})` names one. `MethodTable::prepare` resolves each
   reference once and inlines the predicate's conditions into the caller's conjunction
   before binding, so no call exists at run time.
2. **Result types stay apart.** Only a predicate can be named: a reference to a fold
   (scalar) or an action (effect) is `MethodRefNotPredicate`; a pure call of a predicate is
   `CallError::NotTerminal`; the referenced predicate must share concept, schema, reading
   and effect with the caller (`MethodRefContract`), so a same-shaped predicate of another
   class is not captured.
3. **Typed Gremlin steps.** `gremlin::compose(concept, &[Step])` turns
   `V / has(eq,neq) / where(method()) / values / dedup / count / sum` into a body. The
   anchor contract decides what composes (module doc of `gremlin.rs`); hops are refused
   because the method glove declares no relation carrier; effects have no `Step`.
4. **Gremlin-shaped text.** `V(concept).has(f, true).where(m()).values(f).sum()` loads
   through `decl.rs` into owned storage. Effect steps are `DeclError::Effect`, other steps
   and predicates are `Unsupported` by name; every loaded declaration is pure.

No new mask-risc op, no new R2IL op, no new Quack variant, no method-specific Rust.

## §3 — Evidence

**Acceptance (the four properties):**

| property | test |
|---|---|
| a method references an independently declared predicate | `compose.rs::two_methods_reuse_one_predicate_method`; text: `gremlin.rs::the_lego_methods_load_from_text_and_match_the_compiled_ones` |
| both are declarations, no domain Rust | `tests/fixtures/lego.methods`; `gremlin.rs::a_later_declaration_reuses_an_earlier_predicate` (written after the first load) |
| same Quack program as the direct expression | `compose.rs` (`programs()` == hand conjunction); `gremlin.rs::a_composed_traversal_is_the_direct_quack_query` (== `lower(Query{table(..).where_eq(..)…, SumI32})`) |
| rejects cycles, incompatible populations, unsupported multiplicity, cross-class capture | `compose.rs::a_broken_reference_is_refused` (self, two-step and transitive cycles; seal; reading; fold/action targets; unknown/ambiguous), `…another_class_is_not_captured`, `gremlin.rs::steps_outside_the_anchor_contract_are_refused` |

Each answer is checked on Quack and R2IL against a row loop over the indicator text, at
n ∈ {1, 64, 65, 4097}. Editing only `is_billable` moves both consumers
(`editing_the_predicate_moves_every_consumer`), so the reuse is by reference, not by copy.

**Mutations (each a typed refusal or a red oracle test):**

| corrupted | caught by |
|---|---|
| relation direction | method layer: every hop refused (`NoRelationCarrier`); substrate: #1311 `reverse_fanout_reanchors…` and its disable run (an edge hop that drops the target-exists semijoin) |
| anchor population | `V(document)` in a `billable_work_entry` method → `AnchorMismatch` |
| traverser multiplicity | `values().dedup()` → `DedupOfValues`; `values().count()` → `CountOfValues`; `dedup().count()` = `count()` only on the anchor, asserted; across a fan-out, #1311 `bag_and_set_stay_apart_after_a_fanout` |
| field identity | `UnknownField` through a referenced predicate; `polyfill.rs` swapped-lane binder (both frontends disagree with the oracle) |
| SchemaSeal / ReadingId | referenced predicate of another seal or reading → `MethodRefContract` |
| source generation | `polyfill.rs::planes_of_another_population_are_refused` (`UniverseMismatch`) |
| method reference | unknown key/label, ambiguous label, cycle, fold/action target |
| dedup precondition | `DedupOfValues` |
| effect classification | 13 effect steps → `DeclError::Effect`; `MethodDecl::effect()` is always `Pure`; an action cannot be named as a predicate |

**Disable runs (each red, then restored):** inlining skipped (4/5 of `compose.rs` red);
the contract check compared the target with itself (refusal test red); the concept
dropped from the contract (cross-class test red); the effect gate removed (effects test
red); an ambiguous key resolved to its first hit (refusal test red).

**Cost** (`examples/compose_cost.rs`, 1,048,576 rows, 101 reps, shared machine; tier printed
from compiled target features, not inferred from flags):

| path | native (`avx512f=true`) median / min | portable v3 (`avx512f=false avx2=true`) median / min |
|---|---|---|
| cold: parse 3 declarations | 4.9 / 3.7 µs | 3.1 / 2.9 µs |
| cold: `prepare_with` (resolve, inline, bind, lower) | 2.5 / 2.0 µs | 1.7 / 1.5 µs |
| warm: composed `billable_hours`, Quack | 755 / 667 µs | 662 / 476 µs |
| warm: composed, R2IL | 762 / 677 µs | 663 / 472 µs |
| warm: the same `Program` run directly | 749 / 676 µs | 660 / 464 µs |

A composed call runs the direct query's `Program`, so its warm cost equals direct execution
within noise. The native build is not faster than v3 here, so no SIMD gain is claimed
from the flag. The cold cost is microseconds per file.

## §4 — Vocabulary → algebra matrix (TinkerPop 3.7.7)

Reference: https://tinkerpop.apache.org/docs/3.7.7/reference/ (latest 3.7.x; read 2026-10-12).
Rows group steps of one mechanism, so percentages are of rows, not of step names.

- **DIRECT** = lowers through the G1 method layer today, tested here.
- **COMPOSE** = existing Quack/mask-risc primitives give it (tested in #1311 or the Quack
  suites), but the method layer has no surface for it yet.
- **YET** = needs reusable algebra that does not exist; the evidence column names the
  refusal or counterexample.
- **EFFECT** = state or authority.
- **OUT** = outside the execution contract (lists, strings, lambdas, randomness, OLAP).

| step(s) | family | in → out | multiplicity | order/path witness | effect | G1 status | evidence | missing capability |
|---|---|---|---|---|---|---|---|---|
| V(c) | population | -> vertices of c | bag=rows, bulk 1 | — | pure | DIRECT | anchor = concept table (live plane) | — |
| E() | population | -> edges | bag=edge rows | — | pure | COMPOSE | #1311: edge table as anchor population | relation carrier |
| hasLabel | selection | V->V | keeps | — | pure | DIRECT | V(c) is the classid population; a second label is another anchor | — |
| hasId | selection | V->V | keeps | — | pure | COMPOSE | Cmp::Range / row ordinal | id↔row binding in Glove |
| has(f, eq/neq) | selection | V->V | keeps | — | pure | DIRECT | Cmp::EqU32/NeU32 (tests/gremlin.rs) | — |
| has(f, lt/lte/gt/gte) on i32 | selection | V->V | keeps | — | pure | COMPOSE | Cmp::LtI32..GeI32 | Cond ordered compare |
| has(f, ordered) on u32 | selection | V->V | keeps | — | pure | YET | none: #1311 G5 (`far_side_ordered_predicate_is_refused…`) | ordered u32 compare (G5) |
| has(f, within/without) | selection | V->V | keeps | — | pure | COMPOSE | Filter::Or of EqU32 / And of NeU32 | Cond::Or |
| has(f, inside/outside/between) | selection | V->V | keeps | — | pure | COMPOSE | And/Or of i32 compares | Cond ordered compare |
| hasKey/hasValue/hasNot | selection | V->V | keeps | null | pure | YET | nullable fields refused (`a_nullable_field_is_refused`) | three-valued condition + null policy |
| is | selection | value->value | keeps | — | pure | COMPOSE | = has(f,p) when the value is an anchor field | value-cursor filter |
| where(method()) | selection | V->V | keeps | — | pure | DIRECT | inlined predicate (tests/compose.rs) | — |
| where(traversal) functional | selection | V->V | keeps | — | pure | DIRECT (G1-4a) | a predicate method over `out(rel).has(..)`, referenced by `where(method())` (tests/relation.rs `a_predicate_through_the_relation_is_reused_by_reference`) | — |
| where(traversal) fan-out | selection | V->V (set) | set over fk | ordered fk | pure | COMPOSE | #1311 `exists_over_children…` (CountDistinctOrdered; refuses unordered) | relation carrier + order witness |
| and | boolean | pred,pred->pred | keeps | — | pure | DIRECT | conjunction of Conds | — |
| or | boolean | pred,pred->pred | keeps | — | pure | COMPOSE | Filter::Or | Cond::Or |
| not | boolean | pred->pred | keeps | — | pure | COMPOSE | Filter::Not / dir-sim negate(Semijoin) | Cond::Not |
| filter(lambda) | selection | any | — | — | opaque | OUT | — | lambda |
| dedup() on anchor | multiplicity | V->V | identity (rows distinct) | — | pure | DIRECT | tests/gremlin.rs dedup().count()==count() | — |
| dedup() after fan-out | multiplicity | bag->set | bag→set | ordered key | pure | COMPOSE | #1311 `bag_and_set_stay_apart…` (ScatterOrU32 / CountDistinctOrderedU32) | relation carrier |
| dedup() of values | multiplicity | values->set | bag→set | ordered key | pure | YET | refused (DedupOfValues); CountKeyRunsU32 needs ordered lane | ordered first-occurrence mask (FirstOfRun) |
| coin/sample/timeLimit | selection | any | random | — | nondet | OUT | — | — |
| range/limit/skip/tail | ordered | any | prefix of an order | order | pure | YET | Cmp::Range is row-ordinal only | ordered population op |
| none | selection | any->∅ | — | — | pure | COMPOSE | empty filter | — |
| all/any (list) | selection | list->list | — | — | pure | OUT | no list type | list values |
| out(rel) functional | traversal | V->V' | keeps (anchor stays) | — | pure | DIRECT (G1-4a) | `Cond::Reaches` (Semijoin over the target's live plane) + `Cond::Via` (EqU32Via); tests/relation.rs vs a row loop over the text; R2IL refuses by name (no `GATHER`) | R2IL `GATHER` |
| in(rel) reverse fk | traversal | V->V' | re-anchor on child rows | — | pure | COMPOSE | #1311 `reverse_fanout_reanchors…` | relation carrier |
| out/in(rel) M:N | traversal | V->V' | re-anchor on edge rows | — | pure | COMPOSE | #1311 `many_to_many_one_hop…` | relation carrier |
| both(rel) | traversal | V->V' | union of two bags | — | pure | YET | no bag union across anchors | typed bag merge |
| second hop, functional | traversal | V->V'' | keeps | — | pure | YET | #1311 `functional_two_hop_is_refused…` | composed fk read (G1) |
| second fan-out (fan-in carried) | traversal | V->V'' | bag | — | pure | YET | #1311 `many_to_many_two_hop_is_refused…`; #1312 §D | barrier / fan-in operand (G2) |
| outE/inE/bothE | traversal | V->E | edge rows | — | pure | COMPOSE | edge table as anchor | relation carrier |
| outV/inV/otherV/bothV | traversal | E->V | fk read | — | pure | COMPOSE | EqU32Via from edge row | relation carrier |
| values(f) then fold | projection | V->value | keeps | — | pure | DIRECT | Fold::Sum (tests/gremlin.rs) | — |
| values(f) as result | projection | V->values | bag of values | — | pure | YET | Agg::Rows / Keep exist; Answer has no row/value type | row answer type |
| valueMap/elementMap/propertyMap/properties | projection | V->map | keeps | — | pure | YET | Agg::Rows | row answer type |
| id/label/key/value/element | projection | elem->scalar | keeps | — | pure | YET | Agg::Rows | row answer type |
| constant/identity/barrier | projection | x->x | keeps | — | pure | COMPOSE | no-op in a bulk executor | — |
| as/select/project/by | binding | traverser->map | keeps | labels | pure | YET | none | binding environment |
| match | binding | pattern | bag | labels | pure | YET | none | binding environment + join |
| count() | aggregation | V->scalar | bag count | — | pure | DIRECT | Fold::Count (tests/gremlin.rs, tests/compose.rs) | — |
| count() of values | aggregation | values->scalar | COUNT(f) | — | pure | YET | refused (CountOfValues) | field-reading count fold |
| sum() on anchor i32 | aggregation | values->scalar | bag sum | — | pure | DIRECT | Fold::Sum | — |
| sum() through fk | aggregation | values->scalar | bag sum | — | pure | YET | none (#1312 §I gap 4) | masked_sum_i32_via |
| min/max | aggregation | values->scalar | — | — | pure | COMPOSE | Agg::MinI32/MaxI32 | Fold::Min/Max |
| mean | aggregation | values->scalar | — | — | pure | COMPOSE | lower_avg (SUM+COUNT) | a law over two folds |
| groupCount().by(f) | aggregation | V->map | bag per key | — | pure | COMPOSE | GroupReduce{Count}; #1311 groupCount tests | Fold::GroupCount |
| group().by(f).by(sum) | aggregation | V->map | bag per key | — | pure | DIRECT | Fold::GroupSum (tests/declared.rs) | — |
| fold/unfold | aggregation | bag<->list | — | — | pure | OUT | no list type | list values |
| math | arithmetic | scalars->scalar | — | — | pure | OUT | string expression | — |
| union (same population) | set | pred,pred->pred | set union of rows | — | pure | COMPOSE | Filter::Or | Cond::Or |
| union (different traversals) | set | bag,bag->bag | bag sum | — | pure | YET | none | typed bag merge |
| intersect/difference/disjunct/combine/merge/product/conjoin/reverse/length (lists) | list | list->list | — | — | pure | OUT | no list type | list values |
| choose/coalesce/optional/branch/local | branching | traverser->traverser | per-branch | — | pure | YET | none | typed branch + merge |
| repeat/until/emit/times/loops | iteration | V->V* | bag or set | path | pure | YET | #1306 frontier fixpoint (plan only) | bounded fixpoint |
| path/simplePath/cyclicPath/tree | path | traverser->path | per path | path | pure | YET | none | provenance-carrying path |
| shortestPath | path | V->path | — | path | pure | OUT | — | — |
| order().by() | ordered | bag->sequence | keeps | order | pure | YET | none (row order only) | ordered population op |
| addV/addE/property/drop | mutation | — | — | — | effect | EFFECT | refused at load (tests/gremlin.rs) | dir-sim delta / action gate |
| mergeV/mergeE | mutation | — | — | — | effect | EFFECT | refused at load | dir-sim delta / action gate |
| sideEffect/aggregate/store/sack/cap/subgraph | state | — | — | — | state | EFFECT | refused at load (cap/subgraph: not parsed) | explicit computation state |
| inject/io/call | source | — | — | — | external | EFFECT | refused at load | authority boundary |
| fail | control | — | — | — | effect | OUT | — | — |
| concat/asString/toUpper/toLower/trim/lTrim/rTrim/split/substring/replace/format | string | string->string | — | — | pure | OUT | strings are dictionary codes | — |
| asDate/dateAdd/dateDiff | date | date->date | — | — | pure | OUT | — | — |
| pageRank/peerPressure/connectedComponent/program | OLAP | graph->graph | — | — | pure | OUT | — | — |
| map/flatMap/sideEffect (lambda), index, explain, profile | meta | — | — | — | — | OUT | — | — |

| predicate | status | evidence |
|---|---|---|
| P.eq / P.neq | DIRECT | Cmp::EqU32/NeU32 |
| P.lt/lte/gt/gte (i32) | COMPOSE | Cmp::*I32; no Cond form |
| P.lt/lte/gt/gte (u32) | YET | G5 |
| P.inside/outside/between | COMPOSE | And/Or of compares |
| P.within/without | COMPOSE | Or/And of EqU32 |
| TextP.startingWith | COMPOSE | PrefixLowering over an ordered dictionary lane |
| TextP.endingWith/containing/regex and negations | OUT | strings are codes |

Step rows: 67. DIRECT 12 (18%), COMPOSE 20 (30%), YET 19 (28%), EFFECT 4 (6%), OUT 12 (18%) — after G1-4a; at #1508: DIRECT 10, COMPOSE 22
Predicate rows: 7. DIRECT 1, COMPOSE 4, YET 1, OUT 1

YET rows by missing capability: row answer type ×3; ordered population op ×2; typed bag merge ×2; ordered u32 compare (G5) ×1; three-valued condition + null policy ×1; ordered first-occurrence mask (FirstOfRun) ×1; composed fk read (G1) ×1; barrier / fan-in operand (G2) ×1; binding environment ×1; binding environment + join ×1; field-reading count fold ×1; masked_sum_i32_via ×1; typed branch + merge ×1; bounded fixpoint ×1; provenance-carrying path ×1

COMPOSE rows by what the method layer lacks: relation carrier ×6 (×8 at #1508; G1-4a moved functional `out` and `where` over it to DIRECT); Cond::Or ×3; Cond ordered compare ×2; — ×2; id↔row binding in Glove ×1; value-cursor filter ×1; relation carrier + order witness ×1; Cond::Not ×1; Fold::Min/Max ×1; a law over two folds ×1; Fold::GroupCount ×1

## §5 — What the matrix says to build next

- **Most leverage, and no new algebra: a relation carrier in the `Glove`** — `(carrier: fk |
  edge table, field(s), direction, target concept)`, the four facts #1311's adapter needed
  per relation. It turns 9 COMPOSE rows (functional `out`, reverse `in`, one M:N hop, `E`,
  `outE/inE`, `outV/inV`, `where` over a hop, `dedup` after a fan-out) into method-layer
  steps by moving #1311's anchor rule into `compose`. It is a binding, not a primitive.
- **Small method-layer surfaces over existing primitives:** `Cond::Or/Not`, ordered `Cond`
  over `i32`, `Fold::Min/Max/GroupCount`, a mean law (9 COMPOSE rows).
- **Reusable algebra (YET), by cross-domain reuse:**
  1. ordered-key family — first-occurrence mask (FirstOfRun; NARS stamps, IAM, SAP/Odoo dedup),
     ordered population ops (`order/range/limit`), ordered `u32` compare (G5);
  2. a row/value answer type (`values` as a result, `valueMap`, `id`) — `Agg::Rows` exists, `Answer` has no row type;
  3. composed fk read (G1) and the fan-in operand (G2), the two traversal gaps #1311/#1312 named;
  4. binding environments (`as/select/project/match`), typed branch and merge, bounded fixpoint, paths.

None is built here. Each needs the ledger's extension criteria (a method that needs it, a
counterexample, an independent oracle, both frontends, a second domain).

## §6 — Effects and IAM

No Gremlin effect step loads: `addV/addE/property/drop/mergeV/mergeE` and the state steps
are refused at parse time, and `MethodDecl::effect()` is `Pure` for every declaration. An
administrative intent such as `IAM::group(g)::add(member)` belongs to the action gate and
dir-sim's version/delta machinery as a proposal or simulated transition; loading a
traversal grants no authority. Not exercised here: no write of any kind.

## §7 — Falsifiers

1. A composed method whose `programs()` differ from its direct conjunction.
2. A reference that resolves across concepts, seals or readings.
3. A traversal accepted by `compose` whose answer differs from the row oracle.
4. `dedup` or `count` composed past `values(f)` without a refusal.
5. A declaration that loads an effect.

## §8 — D-CMH-G1-4a: the functional relation carrier

A [`Glove`] now carries [`Relation`]s: the canonical key field on its concept,
the target concept's glove, and where a call supplies the target's live plane
and value lanes (`Foreign::planes` / `Foreign::lanes`). That is #1311's
adapter, minus the relations that fan out.

- `out(rel)` composes to `Cond::Reaches { rel }` and binds to
  `Filter::Semijoin(fk, target live plane)`. Gremlin drops a traverser whose
  edge leads nowhere, so a dangling key and a key to a dead row both fail it.
  `EqU32Via` alone drops only the first, so the liveness test is not
  optional: removing it fails the row oracle.
- `out(rel).has(f, eq/neq)` composes to `Cond::Via` and adds
  `Filter::EqU32Via` (negated for `neq`) on the target's lane. The liveness
  test is emitted once per relation, wherever it is first named.
- `count()` after `out(rel)` counts anchor rows: a functional relation yields
  at most one traverser per anchor row, so the anchor does not move.
- Refused by name: a second hop (`SecondHop`), `values(f)` on the target
  (`ValueThroughRelation`; no sum through a key in Quack), `dedup()` on the
  target (`DedupThroughRelation`; a set count), `where(method())` on the
  target (`WhereThroughRelation`), and `in`/`both` (`FanOut`).
- Preparation binds the target field against the target's own basis, claims
  and binder, and names errors `relation.field`. The key joins `required`, so
  an unproven key refuses the method like any field it reads.
- A call names each target concept's population; another population, or
  none, is `CallError::ForeignUniverse`. Two relations that reach one target concept in
  different populations are refused at preparation (`ConflictingUniverse`),
  since one population per concept is all a call can name.
- **R2IL cannot express it.** The fold band has `VIA` but no `GATHER`: it can
  read a target field through a key, but not test that the key names a live
  row. A method with a relation condition returns `CallError::FrontendGap {
  R2il, "GATHER" }`; methods without one still run on both frontends.

Evidence: `tests/relation.rs` (8 tests). FIXTURE: a synthetic `line -billed_to->
partner` pair (37 partners, 4 dangling key values, every 7th partner dead,
every 13th line dead); every answer is checked against a row loop over the
field text at n ∈ {1, 64, 65, 4097}. Six disable runs, each red on the test
that names it: liveness test removed, `neq` ignored, population check removed,
key dropped from `required`, R2IL refusal turned into a silent pass, liveness
emitted per condition.

## D-ids

| D-id | scope | status |
|---|---|---|
| D-CMH-G1-0 | vocabulary matrix (§4) | Shipped (#1508) |
| D-CMH-G1-1 | predicate methods + method references | Shipped (#1508) |
| D-CMH-G1-2 | typed Gremlin steps → method body | Shipped (#1508) |
| D-CMH-G1-3 | Gremlin-shaped runtime declarations | Shipped (#1508) |
| D-CMH-G1-4a | functional relation carrier: `out(rel)`, `out(rel).has`, `count()` after it (§8) | In PR |
| D-CMH-G1-4 | the rest of the carrier: `in`/M:N (anchor moves to child or edge rows), `E()`, `outE/inV`, `dedup` after a fan-out | Queued |
| D-CMH-G1-5 | FirstOfRun (insufficiency proof first; separate PR #3) | Queued |
