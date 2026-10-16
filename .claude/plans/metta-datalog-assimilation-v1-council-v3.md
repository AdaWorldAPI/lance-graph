# RATIFIED v3 — metta-datalog-assimilation council (2026-10-16) — metta-datalog-assimilation, tranche W1–W3 (D-ASM-1..4) + ratification of decisions A–H

> v3 = v2 with the three reviewers' verdicts applied (0 BLOCK, 2 FIX-P1, 16 FIX-P2; §9 ledger). v2 = v1.1 with every savant finding applied; §8 is that ledger.
> Status words: DECISION / MEASURED (with command) / VERIFIED-IN-CODE (with location) / POLICY PIN (a chosen bound, not a measurement) / OPEN.
> v1.1 = v1 re-asked after the premise gate (PREMISE-SPLIT; §6 stamp sub-premise PREMISE-WRONG). §3 below is the
> AMENDED resolution; §7 records the gate and the split questions. Savants answer §3/§5/§6 as they stand here.

Council: /5plus3, 2026-10-16. Orchestrator: main thread. Repo: /home/user/lance-graph at origin/main `4f9d5f26`.
Scope: ratify plan `.claude/plans/metta-datalog-assimilation-v1.md` decisions A–H and make D-ASM-1, D-ASM-2, D-ASM-3, D-ASM-4
executable. OUT of this council: W4 (μ/d′), W5 (D-ASM-5/6), W6, W7 — each gets its own deliberate pass when its wave opens.
The council verifies and hardens; it does not design. A savant that wants a different design files ONE `RISK` and stops.

## 0. Vocabulary pins (so no savant re-derives them)
- "row" = one 512-byte node row, key = 16-byte `FacetCascade` (`classid u32 + 12-byte content-blind register`, `E-V3-FACET-4-PLUS-12`).
- "lane" = one SoA column of u32/i32 over a population; "plane" = a u64 bitmap over the same population.
- "fold" = one mask-risc `Program` run over a population; relalg lowers a `Query` to R2IL calls over `FoldDialect`
  (`crates/lance-graph-relalg/src/lower.rs:1-13`), which emits `Pred`s and ends in a `Terminal`.
- "preimage" = D-CMR-5: `f⁻¹(S)` of a kept set `S` under a proven mapping `f` (lane → key), the only sanctioned hop
  (`.claude/plans/relalg-v1.md:68-72` `never a foreign key`; `STATUS_BOARD.md:78`).
- "bag" = multiset; Hyperon `BindingsSet(SmallVec<[Bindings;1]>)` is a bag, `PartialEq` "conceptually unordered"
  (fork `hyperon-atom/src/matcher.rs:883-888`).

## 1. FROZEN DECISIONS (cite; council may only VIOLATES with file:line)
F1. Strings are labels, never addresses; a text front end lowers from the BOUND plan, never the raw AST; no fallback engine.
    (nal-polyfill-v1 §4 H `:249`; `cypher-lowering-warden.md:7`; relalg-v1 §4 `:68-72`.)
F2. No new opcode / terminal / vocabulary byte without an execution-level insufficiency proof (`isa-anti-lasagne-warden`
    card; plan §4 "Kill conditions — Anywhere").
F3. Zero copy is a law without escape hatches; a binder never holds atoms, it holds addresses (`zero-copy-warden` card;
    plan §5).
F4. The 16-byte key and the 512-byte stride are canon; a tenant may be added, the stride never changes
    (`CLAUDE.md` § CANON — Minimal SoA node; `v3-envelope-auditor`).
F5. AdaWorldAPI forks only, never crates.io for a forked crate; the lance family excepted (`CLAUDE.md` P0). Pins: hyperon-experimental
    `3f76dc4`, PeTTa `1d19a1d`, ONA `dc4efd0`, OpenNARS Java `4d1bdb0` — MEASURED as the clone heads 2026-10-16 (`git rev-parse --short HEAD`
    in each clone); datafrog `edd7cfc` (master) per `crates/lance-graph-metta/Cargo.lock` (gitignored, local).
F6. Hyperon's interpreter is called, never ported or modified for this; an unsupported shape is never approximated
    (metta-polyfill-v1 §8 `:394-401`; plan §3 H). **Amended by R-5 (this council):** two words, two cases — ROUTING: an atom kind
    with no row form is stored and answered by the inner `GroundingSpace` (not an error, counted by kind); REFUSAL: a QUERY the
    fold cannot answer soundly (`UnknownValue`, `CrossTableProjection`, `AnswerTooLarge`, `ProjectionTooWide`, `CodebookFull`)
    is a Hyperon `Error` atom, never an empty `BindingsSet`.
F7. `LanceSpace` is bound as a NAMED space, never as a module's `&self` (`ModuleSpace` splits conjunctions and panics on
    non-`ModuleSpace` deps, `lib/src/space/module.rs:33-48`; metta-polyfill-v1 §5 `:149-162`; D-MTP-2 row).
F8. Three oracles, never averaged; PeTTa is not a substitute for Hyperon (HY-Q12, parity map `:77`; plan §3 G, §7).
F9. Validation order: oracle parity first, CE64 enrichment second; zero enrichment must be byte-invisible (plan §7, DECISION
    2026-10-15).
F10. The per-row truth law is `causal_edge::isa::truth::code` (D-NALP-4; per entry `2026-10-14-isa-truth-exact-codes.md`,
    which records 0 of 2³² off the oracle with its command — cited, not re-measured here); argument order: lance-graph/OpenNARS vs ONA/PeTTa swapped for abduction/induction
    (inventory §2; `Truth.c:84-92` vs `TruthFunctions.java:250-287`).
F11. relalg has no join and no μ by design; D-RA-3 (rules as values) is queued AFTER D-RA-2 (maintainer decision: relalg
    into the contract set IR, or relalg keeps only lowering) (`STATUS_BOARD.md:62-65`).
F12. No model identifier in any committed artifact; status words are `MEASURED/VERIFIED-IN-CODE/TEST-PINNED/DECISION/OPEN`,
    never `operator-*` (`FIRST-HAND-SOURCE-LAW.md`).

## 2. INPUT INVENTORY (VERIFIED-IN-CODE unless marked)
### 2.1 lance-graph (all crates are own-workspace-root, excluded from the main workspace)
| path | shape | role in tranche |
|---|---|---|
| `crates/lance-graph-relalg/src/ir.rs:17-31` `pub enum Cmp` (6 variants); `Atom {I32{lane,op,t}, U32{lane,eq,v}}` :34-39; `Rel {All, Empty, Plane(u16), Atom, And(Vec), Or(Vec), Not(Box)}` :47-62; `Measure {Count, Sum(u16), Min(u16), Max(u16), Keep}` :99-110; `Query` :114 | 7 `Rel` variants, 2 `Atom` kinds, 5 measures; ONE table's row set | home of the Datalog AST VALUES (decision A) |
| `crates/lance-graph-relalg/src/lower.rs:22` `pub enum Answer` (`Run` :37, `RelError` :49, `run` :63, `run_as_written` :84); `normalize.rs:30` `pub fn normalize` (`symbolic` :44) | lowering = R2IL calls over `FoldDialect` (`lower.rs:5-10`: `INT_AND/INT_EQUAL/…`, `ogar_r2il::{KEEP,MAX,MIN,RANGE,SUM}`) | D-ASM-2 extends `Atom`/`Rel`, keeps `normalize` rewrites and `symbolic` ∅ short-circuit |
| `crates/lance-graph-relalg/Cargo.toml` deps: mask-risc, r2il, ogar-loco, ogar-r2il (git main) | no quack, no contract direct | dependency decision below (R-2) |
| `crates/lance-graph-quack/src/lib.rs:340` `pub enum Filter` (`EqU32Via` :358, `Semijoin { fk, foreign }` :387; `pub fn lower` :1383; `pub struct Query` :1301); deps mask-risc + contract only | the SHIPPED hop lowering (D-MTP-1 used it: `crates/lance-graph-metta/examples/join_probe.rs:1-12,27,75-79`) | parity oracle for D-ASM-2's `Program` bytes |
| `crates/lance-graph-mask-risc/src/ir.rs:127-205` `Pred` (incl. `EqU32Via{fk,key,v} :158`, `Range :180`, `EqU32Strided :184`, `MatchFacet16Strided :200`); `:210-287` `MaskOp` (incl. `Gather :277`); `:288-470` `Terminal` (incl. `Count :290`, `Keep :321`, `ScatterOrU32 :338`, `ScatterCountU32 :364`, `CountKeyRunsU32 :389`, `GroupSumI32 :399`, `GroupReduce :442`, `GroupPowerSumsI32 :465`) | the whole word set D-ASM-2/3 may use | F2: nothing added without proof |
| `crates/lance-graph-r2il/src/lib.rs:260` `pub struct FoldDialect` (`new(planes, foreign)` :327, `with_keep` :366, `last_program` :447, `emit_pred` :589) | the only path relalg takes to a `Program` | gate G4 (byte-identical `Program`) reads `last_program()` |
| `crates/lance-graph-metta/Cargo.toml` deps: mask-risc, quack; dev datafrog (fork master); `src/lib.rs` 7 lines (stub); `examples/common/mod.rs:23 Relation2`, `:46 ingest`, `:139 pairs`, `:180 preimage`, `:200 range_pairs`, `:236 oracle`; `examples/join_probe.rs`, `join_range_probe.rs` | D-MTP-1/1b probes, MEASURED; no `Space` impl yet; no hyperon dependency yet | D-ASM-1/4 land here |
| `crates/lance-graph-nal/src/lib.rs:94` `pub enum Copula` (Inheritance=1, Similarity=2, Implication=3, Equivalence=4; `Statement` :117, `Belief` :130, `guid` :157-170 = `FacetCascade::from_semantic_tiles([class>>16, class, s, copula, p, id, 0, 0])`, `WEIGHT_SCALE/HORIZON/MAX_CONFIDENCE_CODE/evidence classes` :212-244, `NalError` :494-530, 10 variants counted 2026-10-16) | arity-2 fact already laid out as 6 u16 tiles + 2 spare | D-ASM-1 reuses the tile layout for `(f a₁…aₖ)`, k ≤ 5 |
| `crates/lance-graph-contract/src/nars.rs:12` `pub enum InferenceType` (Deduction, Induction, Abduction, Revision, Synthesis; doc drift: described as CAM search modes) | out of tranche; inventory §6 item | non-goal here |
| `crates/causal-edge/src/layout.rs` bands 40..42 Pearl, 53..58 witness, 59..63 epistemic ladder | no foreign engine carries these | F9 pass-2 only; NOT written by W1–W3 |
### 2.2 forks (paths are fork-relative at the pinned revs; local shallow clones were read, nothing here depends on a machine path)
| path | fact |
|---|---|
| hyperon-experimental@3f76dc4 `hyperon-atom/src/lib.rs:827-849` | `enum Atom {Symbol, Expression, Variable, Grounded}` |
| hyperon-experimental@3f76dc4 `hyperon-space/src/lib.rs:152 trait Space` (`common`, `query(&self,&Atom)->BindingsSet :175`, `visit`, `as_any`); `:219 trait SpaceMut {add :237, remove :255, replace :275}`; `:282 DynSpace(Rc<RefCell<dyn SpaceMut>>)`; `:340-368 complex_query` (folds `(, …)` conjuncts left to right, `BindingsSet::single()` seed, `merge`) | the ONLY seam; `complex_query` is a helper a space may call, not trait |
| hyperon-experimental@3f76dc4 `hyperon-atom/src/matcher.rs:883 BindingsSet(SmallVec<[Bindings;1]>)`, `:886-888 PartialEq` unordered, `:975 count(n)`, `:1032 merge` | bag; `count(n)` builds n full matches |
| hyperon-experimental@3f76dc4 `lib/src/metta/runner/stdlib/core.rs:163` | `match` → `space.borrow().query(&pattern)` unchanged |
| hyperon-experimental@3f76dc4 `lib/src/space/module.rs:33-48` | `ModuleSpace` runs `complex_query` itself; non-`ModuleSpace` dep panics `:44` |
| hyperon-experimental@3f76dc4 `hyperon-space/src/index/trie.rs` const `:511`, assert `:539` (`TK_MAX_EXPRESSION_SIZE = 1<<10`; metta-polyfill-v1 §3.1 `:90-106`) | join_10000 panics in Hyperon at the pinned rev; the fix exists on the fork branch `ccr-9f3b8075-74u47l-trie-expr-width` (`5c9afd4`, tests first) and `1ffdbd6` is upstream PR trueagi-io#1107 — neither on fork main, so the pin still panics (corrected 2026-10-16) |
| hyperon-experimental@3f76dc4 `python/tests/scripts/*.metta` — 26 files (`ls … | wc -l`, 2026-10-16), among them `b4_nondeterm`, `c1_grounded_basic`, `e1_kb_write`, `a1_symbols` (parity map `:35`, HY-Q04 `:69`) | b4: 2×2 Cartesian branch → four answers; the multiplicity kill condition |
| `hyperon-probe/golden` (metta-polyfill-v1 `:387`) — ABSENT on disk (S3 #5); G6 builds the baseline first | D-ASM-4 gate |
| ONA@dc4efd0 `src/Term.h:53-72` `Term {bool hashed; HASH_TYPE hash; Atom atoms[COMPOUND_TERM_SIZE_MAX]}`, `Term_Equal :64`, `Term_ExtractSubterm :68`, `Term_Complexity :70`, `Term_Hash :72`; `Config.h:140 COMPOUND_TERM_SIZE_MAX 64` | decision E (W5, out of tranche; cited for A's arity bound) |
| Narsese.rs (fork head read 2026-10-15): `enum_narsese::Term` (`term/structs.rs:20`) and `lexical::Term` — two namespaces | NAL front end, W5/W6, out of tranche |
| datafrog fork `edd7cfc` (dev-dep of metta) | exact Datalog oracle; join 10⁶ at 65–69 ns/pair per entry `2026-10-14-metta-polyfill-map.md` (D-MTP-6) |

## 3. THE PROPOSED RESOLUTION (committed, v1.1 — amended by the premise gate)
R-1 **Two halves, two homes.** A conjunctive body has (a) an EXISTENCE half — which rows of the head table satisfy it — whose
   signature is "one table's row set through fk lanes", and (b) an ANSWER half — which binding tuples it produces, with
   multiplicity. They are never filed under one name.
   (a) lowers through **quack** as shipped: `Filter::And([Plane(valid), Semijoin{fk,foreign}, EqU32Via{..}, Range])`
   (`crates/lance-graph-quack/src/lib.rs:340` `pub enum Filter`, `lower` :1383). relalg is NOT extended in this tranche; D-RA-3 stays queued
   behind D-RA-2 (F11). The D-MTP-1 probe is the precedent (`join_probe.rs:1-12,27,75-79`).
   (b) is a KEYED COUNT: for a projection onto one coordinate `?x` (a lane value of the head table, i.e. a LABEL) the bag is
   `Terminal::GroupReduce { mask: kept, key: GroupKey(lane of ?x), fold: Count }` (`mask-risc/src/ir.rs:420-446`) → `(key, n)`
   pairs; the adapter builds `BindingsSet` by pushing `n` `Bindings{?x ↦ atom(key)}` per key. `CountKeyRunsU32` is
   `COUNT(DISTINCT)` (`ir.rs:369-389`) and is the dedup arm: it is the falsifier, never the carrier.
   The lowering is quack's own: `Query { filter, agg: Agg::GroupReduce { key: GroupAddr::Local(col) | GroupAddr::Pair{hi,lo,stride}, agg:
   GroupAgg::Count } }` → `Terminal::GroupReduce` (`quack/src/lib.rs:1224-1262`, `:1476` builds the same shape; `mask-risc/src/ir.rs:490-505`). No quack change — VERIFIED-IN-CODE by
   the orchestrator at `lib.rs:1224-1235` (`Agg::GroupReduce{key: GroupAddr, agg: GroupAgg}` is a plain `Query` aggregate), and
   re-asserted by G4b below for an arbitrary non-`GROUP BY` lane.
   **Projection rule (committed, v2):** one variable → `GroupAddr::Local`; two variables on the SAME head table → `GroupAddr::Pair`
   (`(x, y, n)` bags, no tuple carrier); a projection whose variables live on DIFFERENT tables (a two-hop with both ends projected) is
   refused `Unsupported::CrossTableProjection` in W1–W3 — that is the only real gap (S5 #5/#6, quack `GroupAddr::Pair` :1252). The pair
   EMISSION of D-MTP-1 stays example-only. **Sink bound:** the `Out::I64` buffer is `K` slots, `K` = the label codebook universe
   (≤ 2¹⁶, see R-4) for `Local`, `hi_universe × stride` for `Pair`; `K` is pinned at bind time from the codebook, never from the row
   count, because `?x` binds to a LABEL (lane value), never to a row address (S2 #9, S3 #2). A `Pair` whose `K` exceeds
   `PAIR_K_MAX` refuses `Unsupported::ProjectionTooWide` rather than allocating. `PAIR_K_MAX = 2²⁴` is a POLICY PIN (128 MiB of
   `i64` sink at the bound), re-measured when a real fixture needs more; it is not derived.
R-1b **The Datalog values** `Fact{rel:(classid,arity), args:[Coord;5]}` and `Query{body: Vec<Pattern>, project: Vec<Var>}` with
   `Coord = Const(u16 label id) | Var(u8)` live in `lance-graph-metta::datalog` as the front end's BOUND plan (F1), lowering into
   quack `Query` exactly as `lance-graph-cypher-quack` does (`lower_plan(plan, bind) -> Result<Compiled, Refusal>`,
   `cypher-quack/src/lib.rs:256`, `Refusal` :164, "not a second evaluator … a refusal is never retried elsewhere" :41-44) — the
   `Refusal`/`Demand` pattern is COPIED, not reinvented (S1 #1). This is a bound plan over quack's sanctioned `Query` home; under the warden's own checklist (`isa-anti-lasagne-warden.md:24-28`:
   a SECOND SPELLING is a lowering into R2IL/`Program` with no second reader; `nal-polyfill-v1.md:52`) it is not one, because it
   emits a quack `Query`, never a `Program` (S1 #3 answers S3 #9). That is a reading of the rule, not a proof; G4 is the test. Not in relalg
   (`ir.rs:41`), not in quack (`bind.rs` `Op` is `Eq/Ne` over one table, S5 #1), not in the D-CMR set IR (not code, D-CMR-1 Queued,
   `STATUS_BOARD.md:74`; sets not bags, S5 #2). **There is no `Rule` value in W1–W3** (deferred to W4 with d′, where it has a reader; S2 #4, AP6 — a stored, never-evaluated
   type is a one-impl shell). An `=` atom is CLASSIFIED at `add` (to route it) and kept as an atom in the inner space (R-5); the
   `Rule` value arrives with W4 and d′. A `match` over a relation only `=` rules define is answered by the inner space, i.e. by
   Hyperon's own semantics, never refused. **Migration debt recorded:** when the procedures crate of `sql-polyfill-r2il-v1.md:115-117`
   lands, this front end (like cypher-quack today) must re-route through it (S1 #4).
R-2 **The join leaf** — unchanged words: constant on one side → `Pred::EqU32Via{fk,key,v}`; otherwise the gated semijoin
   `MaskOp::Gather` behind quack `Filter::Semijoin` — gated by the ENCLOSING `Plane(valid)` conjunct, which the survivor-skip bakes
   into `Gather{under}` (quack `lib.rs:1966-1990`, `:2050-2060`); the nullable-fk `valid(fk)` rewrite is a `sql_where` concern
   (`:560-585`) the Datalog front end never invokes; address-ordered
   fk → the range preimage (D-MTP-1b). No relalg `Atom::Via`/`Rel::Semijoin` in this tranche (struck from v1). No new word (F2).
R-3 **Multiplicity** = R-1(b): `GroupReduce{Count}` is the carrier (keyed count = bag; prior art: the bag/set split in
   `quack/tests/gremlin_parity.rs:556` and entry `2026-10-12-edge-table-anchor-distinct.md`, S1 #5). `CountKeyRunsU32` is the
   dedup FALSIFIER only (`COUNT(DISTINCT)`, key-ordered, `LaneNotOrdered` at run — S3 #3). No third option; no new terminal (F2).
   **R-3b — the one named exit (a different layer from the carrier):** `Space::query` must return a by-value `BindingsSet`, and `BindingsSet::count(n)`
   yields n EMPTY `Bindings` (`matcher.rs:975-977`, S3 #4), so the adapter pushes n `Bindings{?x ↦ Symbol(label)}` per key. That is
   O(answer) and population-proportional; it is the ONE named exit `materialize_bindings(keyed_counts) -> BindingsSet`, O(Σn)
   stated in its doc, refusing above `ANSWER_ROWS_MAX` with `Unsupported::AnswerTooLarge` (`ANSWER_ROWS_MAX = 2²⁰` is a POLICY PIN: one `Bindings`
   per answer row at that bound is tens of MiB, the scale where a caller wants a count, not a bag; re-measured when needed)
   (S2 #8 VIOLATES → resolved by naming, exactly as `Mask::materializeRows()` in lgj). Internal currency stays mask + keyed count.
R-4 **The binder (D-ASM-1).** **The label codebook is W1 scope, new** (S1 #8: no existing codebook fits — ontology
   `soa_bake::LabelCodebook` drops unknown tokens, `literal_graph` is a u8 palette, quack `Binder::code` is a trait with no
   production impl). `metta::codebook::LabelCodebook`: `Symbol` name → **u16** id stored verbatim in the tile and the lane (the tile width of
   `from_semantic_tiles`; a u32 would truncate silently and alias keys — S2 #2 GAP); **id 0 is reserved = "no label"** so an
   unwritten argument lane reads as absent; ids `1..=65535`, capacity 65,535, overflow → `Refusal::CodebookFull`; MINTS only
   at `add` (a fact introduces its labels), NEVER at `query` (unknown label in a query constant → `Refusal::UnknownValue`, the
   quack `Binder::code` contract `bind.rs:100-102`). The codebook holds the label strings (layer-3 content, I-VSA Test 0) and is
   the ONLY place a `String` lives; G1 is restated accordingly (S2 #10). `Symbol` → label id, resolved ONCE; flat
   `Expression (f a₁…aₖ)`, k ≤ 5, `f` a Symbol → a row of class `(f, k)` keyed by
   `FacetCascade::from_semantic_tiles([class>>16, class, a₁, a₂, a₃, a₄, a₅, 0])` (`nal/src/lib.rs:157-170`), one u32 lane per
   argument holding the u16 label id; **plus one reserved u32 lane `evidence` per fact row, written `3` = `NO_PROVENANCE` (nal `lib.rs:242`: `INDEPENDENT 0,
   DUPLICATE 1, OVERLAP 2, NO_PROVENANCE 3`) in this tranche — never 0, which would read as INDEPENDENT evidence (R1 P1)** so W5/W6 attach the stamp lane without retrofitting the binder (S5 #9); `Variable` →
   `Coord::Var`, never a row; `Grounded`, nested `Expression`, `(stv f c)`, state atoms, and any `=` rule
   → ROUTED (see R-5), classified by kind and counted. The binder holds the codebook + class registry only; addresses are
   recomputed from tiles on every bind; no `Atom` is cached (F3). Functional-relation requirement of D-MTP-1 (`SecondFact`
   refusal) does NOT carry over: a relation is a population of rows, many facts per subject are rows. D-MTP-1 left one-to-many
   OPEN (`entries/2026-10-14-d-mtp-1-join.md:18-19`, S1 #10); it is closed here by gate G2b (a non-functional fixture diffed
   against datafrog), not by assertion.
R-5 **`LanceSpace` (D-ASM-4) — the refusal CHANNEL.** `SpaceMut::add(&mut self, Atom)` returns `()` (fork
   `hyperon-space/src/lib.rs:237`): there is no error path at `add`. Therefore `LanceSpace` OWNS an inner `GroundingSpace`;
   `add` routes a bindable atom to rows and every other atom to the inner space, incrementing a per-kind refusal counter
   (inspectable via a grounded op, e.g. `(lance-space-stats &kb)`). `query(&Atom)`: a `(, q₁…qₙ)` conjunction whose patterns
   all bind lowers as ONE quack `Query`; a pattern that does not bind is answered by the inner space; a conjunction mixing both
   is answered by `complex_query`-style folding over the two (recorded in the harness as "per-conjunct"). The answer is a TWO-CATEGORY result: a fold-bag (key-derived `Bindings`, may carry an evidence lane later) and an inner-bag
   (matcher-derived `Bindings`, never carries evidence), each tagged with its origin in the adapter's own return type and only then
   concatenated into the `BindingsSet` the trait demands; the harness diffs each category separately per R-8 (R2 P1). Errors that are not routing (unknown label at bind of a QUERY constant, `CrossTableProjection`, `AnswerTooLarge`, `CodebookFull`,
   `ProjectionTooWide`) surface as Hyperon `Error` atoms, never as an empty `BindingsSet`. F6 is amended: "refused at `add`" reads
   "routed at `add`, refused at `query`". Named-space binding (F7) unchanged. **Batching (S2 #6 GAP):** `add` appends to a RESIDENT staging fold; one Lance version
   is published per batch at an explicit boundary — `(lance-space-flush &kb)` or implicitly before every `query` (read-your-writes
   for `e1_kb_write`), never one version per `add` (split-tunnel, `split-tunnel-sparse-append-resident-fold.md`). `remove` =
   tombstone mask + inner-space remove; `replace` = remove + add. **The inner space is not a fallback engine (S2 #5):** routing at
   `add` keeps the two sides DISJOINT by atom kind, so the bag union cannot double-count (G6 asserts disjointness, S2 #10), every
   inner-space answer is counted per kind, and an answer that needed the inner space is reported as such in the harness. **Plan
   amendment carried (S1 #10 VIOLATES):** `metta-polyfill-v1.md:173-174` "Shapes the class does not declare are refused, never
   stored as text" becomes "… are never stored as ROWS; they stay atoms in the space's inner `GroundingSpace`, counted by kind" —
   the module's `&self` remains a separate `GroundingSpace` as D-MTP-0 settled (S1 #9); the inner space exists only because
   `add` has no refusal channel. **Provenance note (S5 #10):** a `Bindings` from the inner space carries no evidence lane; the union
   is where gating cannot yet attach — recorded under N4, not fixed here.
R-6 **Migration order.** W1 (R-4 binder + R-1b values) and W2 (R-1(a)/(b) lowering + harness skeleton) in parallel on disjoint
   files; W3 (R-5) after both. No commit touches `canonical_node.rs`, `facet.rs`, `causal-edge`, relalg, or mask-risc `ir.rs`.
R-7 **Deleted/kept/aliased.** Nothing deleted. `join_probe.rs`/`join_range_probe.rs` kept (quack-path precedent; G4 reads them).
   `lance-graph-metta/src/lib.rs` stub replaced. metta gains deps `hyperon-atom`, `hyperon-space` (`git = "https://github.com/AdaWorldAPI/hyperon-experimental"`,
   `rev = "3f76dc4"`, never `branch`, never crates.io — F5, S4 #2, R3 P2),
   keeps quack + mask-risc; dev-dep `hyperon` (interpreter) at the SAME rev with `default-features = false` (defaults pull `das`
   from a singnet git source and `pkg_mgmt`). Transitive set read from the fork manifests by S4: common, macros, bitset, unescaper, bimap, smallvec, log — no
   interpreter crate at `[dependencies]` level; G8b turns that into a `cargo tree` assertion rather than a manifest reading (R1 P2). `crates/lance-graph-metta/Cargo.lock` is gitignored (`.gitignore:15`), as
   the no-lock rule requires. No CI workflow names `lance-graph-metta` today; a job is a FOLLOW-UP, so G7 is a local gate in this
   tranche and says so (S4 #8).
R-8 **Oracle comparison is per CATEGORY** (plan §3 G/§7 amendment): every harness assertion names its category — bag vs bag
   (keyed count vs Hyperon multiset), set vs Hyperon's bag deduplicated, count vs bag size — and projects Hyperon's answer to
   that category before the diff. The oracle call is the EVALUATED call (`!(match &kb …)` as the interpreter runs it), never a
   raw `space.query` on a rule atom.
R-9 **Plan amendments carried out of this council (Phase 5, text only, no code):**
   - Decision E (W5): a compound term is INTERNED once, frame-side (a term table with ONA `Term_Hash/Equal/ExtractSubterm`
     parity); a row carries a 16-byte reference to it; a subterm-match lane is a derived projection of that table. The "64 × u8
     per-row tenant" wording is struck (premise gate: +64 B/row = 12.5 % of the stride, a label filed as a row value).
   - §6 stamp: the stamp travels BESIDE the atom as an evidence lane at the same coordinate (the nal `Evidence`/class-lane shape,
     `nal/src/lib.rs:233-255`), never inside the atom; 53..58 and 59..63 are runtime state written by `hydrate`/`revise` over it.
     The OPEN is closed as a DECISION with that shape; the lane's layout is W5/W6 work.
   - Decision D (W4): a Datalog `=` rule is a PROGRAM for d′, not space content; `match` over the derived relation answers the
     closure only through the evaluated call; the plan's wording is corrected when W4 opens.
## 4. NON-GOALS
N1. μ / recursion (W4, D-RA-4/D-MTP-3): the driver d′ and the `Rule` VALUE are its pass; in W1–W3 an `=` atom is an atom in the
    inner space and `match` over it is Hyperon's own answer.
N1b. Cross-table projections (two-hop, both ends projected): refused `CrossTableProjection`, counted; the lift needs a tuple
    carrier that does not exist (W4+). Same-table two-variable projections are NOT limited (`GroupAddr::Pair`).
N2. Compound terms deeper than one level, the 64 × u8 tenant (W5, D-ASM-5): a nested `Expression` is refused in W1.
N3. NAL rule catalogue, Narsese.rs front end (W5/W6, D-ASM-6, D-NALP-5).
N4. Any write to CE64 bands 40..42 / 50..52 / 53..58 / 59..63 for assimilated facts; the stamp-as-atom shape (plan §6 last
    bullet) stays OPEN and is asked of the council as a question, not resolved here.
N5. Merging/upstreaming Hyperon's trie-panic fix (D-MTP-4: `5c9afd4` exists on the fork branch; `1ffdbd6` is upstream PR #1107;
    merging to fork main and bumping the pin is the operator's call); the gate for join_10000 is "LanceSpace answers",
    Hyperon's panic is recorded as the oracle's behaviour.
N6. `contract/nars.rs` InferenceType doc drift (inventory §6) — separate doc PR.
N7. Mining / `lance-graph-cog` (W7).
N8. Performance claims beyond "answers equal": timings are reported, never gated, in this tranche.

## 5. PRE-REGISTERED GATES
G1 (W1) every fixture atom from `e1_kb_write` and `a1_symbols` binds to the same `(classid, NodeGuid)` twice; a routed atom carries
   its kind; no `String` is reachable from any ROW, `Program`, or `Query` value (the codebook is the one exempt holder); a codebook
   at 65,535 labels refuses the next with `CodebookFull` (can-fire) and accepts the 65,535th (can-stay-silent).
G2 (W2) the datalog→quack two-hop: pairs == datafrog as a BAG at n = 10², 10³, 10⁴, 10⁵, 10⁶ on the D-MTP-1 generator; join_10000
   answers where Hyperon panics (the panic is recorded as the oracle's behaviour, N5).
G2b (W2) a NON-functional fixture (k facts per subject, k ∈ {2, 7}) diffed against datafrog as a bag — closes D-MTP-1's one-to-many OPEN.
G3 (W2) disable runs, each red-then-green, each with the patch anchor asserted (`assert s.count(old) == 1`): (a) swapping fk/key in
   the lowered `EqU32Via` breaks a named test; (b) replacing `GroupReduce{Count}` by `CountKeyRunsU32` (the dedup arm) breaks the
   BAG fixture's multiset test; (c) answering a cross-table projection through the `Pair` path instead of refusing breaks a named
   test; (d) minting a label at `query` time instead of refusing `UnknownValue` breaks a named test.
G4 (W2) for the FUNCTIONAL two-hop fixture only (S3 #7): the `Program` produced by `datalog → quack::lower` is byte-identical to
   the `Program` `join_probe.rs:69-79` builds by hand (`fn query` :69, `lower(&Query{…})` :75) — `And[Plane(VALID), Semijoin{DST, VALID_FOREIGN}]` in that conjunct order,
   no `sql_where` (S3 #6: the survivor-skip bakes the earlier `Plane` into `Gather{under}`; `valid(fk)` is only a `sql_where`
   rewrite). The builder is DUPLICATED into the test, not lifted out of `examples/common/mod.rs` (S4 #7: do not touch a probe-shared
   file).
G4b (W2) a one-variable `match` over a flat relation lowers to `Terminal::GroupReduce{key: GroupKey::Lane(arg lane)}` on a lane
   that is NOT a `GROUP BY` key, asserted on the lowered `Program` (makes R-1(b)'s "no quack change" a tested fact).
G5 (W2/3) **`b4_nondeterm` is the ROUTING fixture, not the multiplicity fixture** (S5 #3/#4, S3 #8 — every b4 fact is an `=` rule
   with a nested head; its only `match` pattern is a rule atom; the 2×2 case is an evaluated call): every b4 answer must equal
   Hyperon's AND the harness must report that 100 % of b4 answers came from the inner space (a fold answering any b4 query is a
   routing bug). **The multiplicity fixture is purpose-built and committed:** `tests/fixtures/bag.metta` — n flat facts
   `(p sᵢ xⱼ)` with repeated `xⱼ` added to `&kb`, `!(match &kb (p $s $x) $x)` (one variable, `Local`) and `!(match &kb (p $s $x)
   ($s $x))` (two variables same table, `Pair`); oracle multiset = the Hyperon interpreter run on the same file; equality as
   normalised multisets (HY-Q04); the dedup arm (G3 b) FAILS it. `e1_kb_write` (`add-atom &kb (Green $x)`, `match &kb (Green $x)
   $x`) is the fold fixture from Hyperon's own corpus. Kill condition rewritten: if `GroupReduce{Count}` cannot reproduce the bag
   fixture's multiset, the tranche ships counts only and says so.
G6 (W3) **`hyperon-probe/golden` is ABSENT on disk (S3 #5); "32/33" is struck.** W3's FIRST step builds and commits the baseline:
   the fork's 26 `python/tests/scripts/*.metta` run through the fork's Rust runner at `3f76dc4`, pass/fail recorded per script
   BEFORE any `LanceSpace` exists (MEASURED, with the command). Then: that baseline unchanged with facts moved into `LanceSpace`;
   every LanceSpace answer == Hyperon per CATEGORY (R-8) or a named refusal; inner-space answers counted per kind and DISJOINT from
   fold answers (S2 #10); PeTTa divergences logged by fixture; whole-conjunction vs per-conjunct call counts recorded.
G7 (all) `cargo test --manifest-path crates/lance-graph-metta/Cargo.toml` green; the floor is the guard count: G1 (2), G2 (5 sizes),
   G2b (2), G3 (4 disables × red-then-green), G4, G4b, G5 (routing + bag ×2), G6 — ≥ 20 is that sum, not a taste; relalg's 11
   (`grep -c '#[test]'`, 2026-10-16) and quack's 124 unchanged;
   `cargo clippy … --no-deps -- -D warnings` clean; `cargo fmt --check`.
G8 (all) no new variant in mask-risc `ir.rs`, no change under `crates/lance-graph-relalg`, quack, r2il, contract, causal-edge
   (diff-asserted); `v3-envelope-auditor` not needed (no layout touch).
G8b (all) `cargo tree --manifest-path crates/lance-graph-metta/Cargo.toml -e normal` contains no `hyperon ` (interpreter) and no
   `serde` crate in the LIBRARY build (dev-deps excluded) — disable-verifiable by adding either and watching it fail (R3 P3, R1 P2).
G9 (all) board, same commit: STATUS_BOARD rows D-ASM-1..4 flipped with gate numbers; **D-MTP-2 flips WITH D-ASM-4 (same
   deliverable)**; D-RA-3 gains "Datalog values live in `metta::datalog`; D-RA-2 still gates relalg" (S4 #9); one entry per wave +
   the R-9 amendment entry, `entries_index.py --write`; TECH_DEBT rows for: the procedures-crate migration debt of R-1b, the missing CI job for metta, and the `contract/nars.rs`
   `InferenceType` doc drift (N6); no new ISSUES row (the trie panic is the existing D-MTP-4 board row, corrected 2026-10-16 to "fix exists on the fork branch"); AGENT_LOG entry written by
   the orchestrating main thread ONLY (one-writer rule), naming the council (5 savants, 3 reviewers, verdict counts, v1→v2→v3
   deltas); `SUPERSESSION-INDEX.md` regenerated LITERALLY LAST — after the entries index write AND after every R-9 plan edit, since
   plans and board are both its inputs (S4 #10, R3 P2).

## 6. PER-SAVANT QUESTION SETS (answer each: CONFIRMS / VIOLATES / GAP / PRIOR-ART-AT / RISK, with file:line)
### S1 prior-art-savant
Q1 Does any shipped crate already carry a `Fact/Rule/Query`-shaped value (quack `Query`, D-CMR set IR, cypher-quack)? Name it.
Q2 Is R-1b (a `datalog` value module in metta lowering to quack `Query`) a second spelling of quack's own `bind.rs`
   `Op`/`Query` path or of the D-CMR set IR under the board's own rules (isa-anti-lasagne SECOND-SPELLING)? Cite the ruling.
Q3 Has a bag/multiplicity terminal been proposed or rejected before (search EPIPHANIES, ISSUES, entries for `BindingsSet`,
   `multiset`, `bag`)?
Q4 Does the label codebook R-4 relies on exist as code today (name the type and its resolve fn), or only as D-RA-3 prose?
Q5 Is there prior art for a `Space`-shaped adapter over a fold engine (entries 2026-10-12/14)? What did it conclude that R-5
   contradicts, if anything?
### S2 iron-rule-savant
Q1 R-4 binder vs I-VSA-IDENTITIES Test 0 (register laziness): YIELDS/VIOLATES?
Q2 R-3 options (i)–(iii) vs F2 and AP1–AP9: does any option smuggle a per-row Rust loop or a second engine?
Q3 R-5 `add` as "one Lance version per batch" vs I-SUBSTRATE-MARKOV / split-tunnel pattern: YIELDS/VIOLATES?
Q4 F3 zero-copy: does `BindingsSet` construction in R-5 materialise a population (a `Vec` proportional to rows) outside a
   `materialize*`-named path? If yes, VIOLATES with the line in the plan.
Q5 R-5's bag UNION of fold answers and inner-`GroundingSpace` answers: is a `Bindings` built from a key and one built by
   Hyperon's matcher the same category (I-VSA Test 0 register vs content)? YIELDS/VIOLATES with the matcher line.
### S3 code-truth (runtime-archaeologist charter, general-purpose)
Q1 Every file:line in §2.1 — CODED / CLAIMED / ABSENT? (Open each.)
Q2 Does quack `lower` today emit `Terminal::GroupReduce{Count}` with a `GroupKey` naming an arbitrary lane (not only a
   `GROUP BY` column)? If not, R-1(b) needs a quack `Agg`/`GroupBy` shape first — name the gap with the line.
Q3 Does `Terminal::CountKeyRunsU32` require a key-ORDERED lane (`ir.rs:334-360`)? Does R-3(i) satisfy it for a projected key
   that is not the sort key?
Q4 Does `BindingsSet::count(n)` (`:975`) produce n FULL matches with no variable bindings — i.e. can R-3's adapter use it, or
   must it push n cloned `Bindings` with bindings? Cite.
Q5 Is `hyperon-probe/golden` reachable from this repo or only from the fork clone? Where exactly?
Q6 Does quack's `Semijoin` lowering gate the read by `valid(fk)` (`lib.rs:506-530`) — must R-2(b) replicate that for G4 to hold?
### S4 cascade-impact-savant
Q1 Every file that must change for W1–W3 (mandatory) vs follow-up; include `Cargo.toml`s, board files, plan tables.
Q2 Does adding `hyperon-atom`/`hyperon-space` as git deps of metta pull `hyperon`'s interpreter crate transitively? If so,
   the F6 "not the interpreter crate" line is unachievable — flag.
Q3 Does R-1b/R-5 ripple into `lance-graph-quack` (new `Filter` composition, `GroupKey` on a non-GROUP-BY lane) or
   `cypher-quack`? Name any file outside `crates/lance-graph-metta` that must change.
Q4 Which existing tests (metta `join_probe`/`join_range_probe`, quack suites) could go red from R-1/R-5 and why?
Q5 Which board rows besides D-ASM-1..4 must flip (D-MTP-2, D-RA-3 status wording)?
### S5 creative-explorer-savant
Q1 Strongest alternative to R-1b's home (`lance-graph-metta::datalog` values): inside quack `bind.rs` (extending `Op`), or the
   D-CMR set IR. One RISK if it beats metta; else CONFIRMS with why.
Q2 Second-order consequence of the R-1 tranche limit (one-variable projections only): does it make W1–W3 unable to answer
   `b4_nondeterm` at all (kill condition) — read the fixture's queries and say which project one variable.
Q3 R-1(b) says bag = keyed count. Name the reading under which a two-variable projection is ALSO a keyed count (composite key
   lane, or `GroupKey::Via` chain) without a tuple carrier — or confirm the GAP is real.
Q4 N4 stamp-as-atom: which signature does a stamp have (frame or motion, per `reference-frame-vs-motion.md`) and therefore
   which home — an atom in the space, a lane on the row, or a CE64 band? One line, no design.
Q5 The orthogonal claim hiding behind "assimilate": is the real deliverable a Datalog engine, or a `Space` whose answers are
   provenance-gated (the plan's §7 expansion claim)? If the latter, what in W1–W3 is premature?

## 7. PREMISE GATE (premise-auditor, 2026-10-16) — recorded, verified, applied
Verdict **PREMISE-SPLIT**; sub-premise "stamp travels as an atom" **PREMISE-WRONG**. Six load-bearing code claims were re-read by
the orchestrator (the quoted lines were printed and read, 2026-10-16) and hold: `SpaceMut::add` returns `()` (`hyperon-space/src/lib.rs:237`); `BindingsSet` is in
`hyperon-atom/src/matcher.rs:883-888`; `CountKeyRunsU32` is `COUNT(DISTINCT)` on a key-ordered lane (`mask-risc/src/ir.rs:369-389`);
`GroupReduce` is the keyed count family (`:420-446`); `join_probe.rs:43,60-61` refuses a second fact per subject; relalg `ir.rs:41`
"a set of rows of the table, described, not computed".
Re-asked questions and where v1.1 answers them: (1) existence half vs answer half → R-1(a)/(b); (2) bag = keyed count, where do
multi-variable keys live → R-1(b) tranche limit + S5-Q3; (3) compound term interned frame-side → R-9 (W5 text); (4) rule = program
for d′, refusal channel at `add` → R-5, R-9; (5) stamp beside the atom as a lane → R-9; (6) per-category oracle diff → R-8.
Struck from v1: relalg `Atom::Via`/`Rel::Semijoin`; `Keep + CountKeyRunsU32` as a bag carrier; "refused at `add`".
Struck from v1.1 (by the savants): `Rule` value in W1–W3; "existing label codebook"; `MultiVarProjection` blanket refusal;
`b4_nondeterm` as the multiplicity gate; "golden 32/33"; u32 label ids; one Lance version per `add`.

## 8. CHANGE LEDGER v1.1 → v2 (49 findings: S1 10, S2 10, S3 9 + Q1 table, S4 10, S5 10)
Verdict counts (from the banked reports S1–S5 in the council scratchpad, recounted by the orchestrator): CONFIRMS 17 ·
PRIOR-ART-AT 7 · GAP 11 · RISK 12 · VIOLATES 3 (S2 #8 F3; S1 #10 plan sentence; S3 #8 vacuous G5).
| # | finding | applied as |
|---|---|---|
| S2 #8, S3 #2, S3 #4 | F3: `BindingsSet` construction and the K-sized sink are population-shaped | R-3: ONE named exit `materialize_bindings`, O(Σn), `ANSWER_ROWS_MAX`; R-1(b): K pinned to the label universe (≤ 2¹⁶), never rows |
| S2 #2 | u32 id truncates in u16 tiles | R-4: u16 ids, `CodebookFull` at 2¹⁶ − 1; G1 can-fire/can-stay-silent |
| S1 #8 | the "existing label codebook" is prose | R-4: `metta::codebook::LabelCodebook` is W1 scope; `Binder::code` contract; mints at add only |
| S2 #4 | `Rule` value stored, never evaluated (AP6) | R-1b: NO `Rule` value in W1–W3; `=` atoms routed; value arrives with W4 |
| S5 #5–#7, S3 #1, quack `GroupAddr::Pair` | blanket `MultiVarProjection` over-narrows; Pair is reachable from quack | R-1(b): same-table two-var via `Pair`; only cross-table refused; G3 c rewritten |
| S5 #3/#4, S3 #8 | b4 never reaches the fold; G5 vacuous | G5: b4 = routing fixture (100 % inner-space, asserted); new committed bag fixture; `e1` = fold fixture; kill condition rewritten |
| S3 #5 | `hyperon-probe/golden` ABSENT | G6: build + commit the 26-script baseline FIRST; "32/33" struck |
| S2 #6 | one version per `add` violates split-tunnel | R-5: resident staging, publish per batch / before query |
| S2 #5, S2 #10, S5 #10 | inner space ≈ fallback; union category; provenance lost in union | R-5: disjoint-by-kind, counted, reported; union = bag concatenation; provenance note under N4 |
| S1 #10 | metta-polyfill "never stored as text" contradicted | R-5 + R-9: plan sentence amended to "never stored as ROWS"; recorded, not silently overridden |
| S1 #9 | D-MTP-0: module `&self` separate; D-MTP-1 corrected relalg→quack | R-5 keeps the module space separate; R-1(a) cites the correction |
| S1 #1, S1 #3, S3 #9 | cypher-quack is the exact precedent; not a second spelling | R-1b copies `Refusal`/`Demand`; third-AST concern resolved as a bound plan over quack's home |
| S1 #4 | procedures-crate migration debt | R-1b: recorded on the row |
| S1 #5, S1 #6 | bag/set split precedent; keyed count never tested as Hyperon carrier | R-3 cites; G5 is the first test |
| S1 #7 | `GroupBy` needs dense keys | superseded by S3 #1 (`Agg::GroupReduce{GroupAddr::Local}` lowers on any u32 lane); losing finding kept here |
| S3 #6, S3 #7, S4 #7 | G4 conjunct order / no `sql_where`; functional-only; builder lives in an example | G4 rewritten; builder duplicated, probes untouched |
| S4 #2 | pin by `rev`, interpreter dev-dep default-features off | R-7 |
| S4 #8 | lockfile tracked? no CI line | lock is gitignored (orchestrator re-read, S4 half-wrong); CI job = follow-up; G7 local |
| S4 #9, S4 #10 | D-MTP-2 == D-ASM-4; D-RA-3 wording; entries + supersession last | G9 |
| S4 #4 | quack change needed if `GroupReduce{Count}` not reachable on an arbitrary lane | closed by S3 #1 (`Agg::GroupReduce{GroupAddr::Local}`); pinned by G4b |
| S5 #9 | reserve the evidence lane | R-4: reserved `evidence` u32 lane, written 0 (v2; corrected to `3` in v3, §9) |
| S5 #8, S1 #2, S5 #1/#2, S2 #1/#3/#7/#9, S3 Q1 table, S4 #1/#3/#5/#6 | CONFIRMS / PRIOR-ART with no change | cited inline where they bear |
| S3 Q1 offsets | `join_probe` query at ~:70; `NalError` ≥ 10 variants; trie const :511; Semijoin gate doc :560-585 | §2 corrected in place |
Conflicts resolved: S1 #7 vs S3 #1 — code wins (S3 read the lowering), S1's dense-key constraint applies to `GroupBy` only.
S5 #7 vs v1.1 tranche limit — Pair wins (quack `GroupAddr::Pair` exists); the limit narrows to cross-table. S4 #4 GAP closed by S3 #1.

## 9. CHANGE LEDGER v2 → v3 (Phase 3: 3 reviewers, 0 BLOCK, 2 FIX-P1, 16 FIX-P2, 1 P3)
| reviewer | verdict | applied as |
|---|---|---|
| R1 overclaim | P1: `evidence` lane "written 0 = NO_PROVENANCE" wrong (`NO_PROVENANCE = 3`) | R-4: written `3`; 0 would read INDEPENDENT |
| R1 | P2: unsourced numbers (relalg 11, 26 scripts, SHAs, ns/pair, NalError 7, `≥ 20`, 2²⁰, 2²⁴) | each sourced (count command + date) or labelled POLICY PIN with its reason; §2.1 NalError = 10 |
| R1 | P2: F6 amended elsewhere; scratchpad paths; G4 tilde; "not a third spelling" by citation; R-7 transitive set unshown | F6 carries the amendment; §2.2 fork-relative@rev; G4 `:69-79`; R-1b "a reading, G4 is the test"; G8b `cargo tree` assertion |
| R2 dilution | P1: routing vs refusal conflated; union merges two categories | F6 two words defined; R-5 two tagged bags, concatenated only at the trait boundary, diffed per R-8 |
| R2 | P2: "head table's row" phrasing; R-3 exit in the carrier paragraph; R-2/G4 `valid(fk)` wording; §8 S4 #4 row | R-1 phrasing fixed; R-3b split out; R-2 rewritten to "enclosing `Plane(valid)` baked into `Gather{under}`"; row added |
| R3 firewall | GATE PASS; P2: fork URL; TECH_DEBT/ISSUES homes; one-writer; supersession literally last; P3 serde check | R-7 URL; G9 extended; G8b |
Conflicts between reviewers: none. Stricter verdict rule not invoked.
