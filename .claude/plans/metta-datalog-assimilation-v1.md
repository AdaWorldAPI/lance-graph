# metta-datalog-assimilation-v1 — Hyperon / PeTTa / MeTTa atoms assimilated into a Datalog-AST · NAL · R2IL · ONA-shaped substrate, with MeTTa as the oracle

**Status:** PROPOSAL (D-ASM-0 only: this plan, its board rows, and the inventory it rests on). Nothing else is authorized.
Every wave below is gated on the measurement of the one before it, and the plan goes through `/5plus3` before the first
non-probe code lands.
**Date:** 2026-10-15.
**Repos:** lance-graph (`lance-graph-relalg`, `lance-graph-metta`, `lance-graph-nal`, `lance-graph-r2il`, `lance-graph-mask-risc`,
contract), the forks `AdaWorldAPI/hyperon-experimental`, `AdaWorldAPI/PeTTa`, `AdaWorldAPI/Narsese.rs`, `AdaWorldAPI/OpenNARS-for-Applications`,
`AdaWorldAPI/datafrog`; OGAR only if a class or vocabulary byte is earned (none is expected).
**Parents:** `.claude/research/2026-10-15-nars-metta-ce64-mapping-inventory.md` (the source-read inventory, D-ASM-0's evidence),
`metta-polyfill-v1.md` (D-MTP-0..6), `relalg-v1.md` (D-RA-3/4), `nal-polyfill-v1.md` (D-NALP-1..11),
`.arxiv/2026-10_hyperon_metta_to_lance_graph_parity_map.md` (HY-Q01..15), `entries/2026-10-03-coresearch-chained-hop.md` (option d′).
**Umbrella, not a rival:** this plan sequences D-MTP, D-RA and D-NALP work toward one target and adds only what none of them names
(D-ASM-1, D-ASM-5, D-ASM-6, D-ASM-7). It mints no D-id for work another plan already owns; those rows are cited, not copied.

## 1. The target, in one picture

```text
MeTTa text ──hyperon SExprParser (fork, unchanged)──► Atom          PeTTa clauses ──► (oracle only)
                                                       │
                                     D-ASM-1 binder: atoms → addresses, once
                                                       ▼
                        Datalog AST (relalg values: Fact / Rule / Query over coordinates)   ◄── Narsese.rs terms (NAL front end)
                                                       │  D-RA-3 values, D-ASM-2 join = D-CMR-5 preimage, D-RA-4 μ via d′
                                                       ▼
                        lance-graph-relalg  ──► R2IL FoldDialect ──► mask-risc ──► ndarray::simd
                                                       │
                              D-ASM-3 bag terminal (BindingsSet multiplicity)   D-NALP-4 per-row truth law
                                                       ▼
            answers diffed three ways: Hyperon interpreter · PeTTa · LanceSpace  (D-ASM-4), NAL answers vs ONA · OpenNARS (D-NALP-7)
```

"Assimilate" means: everything that is **data** in a MeTTa program (facts, `=` rules in the Datalog fragment, `:` type atoms,
the space contents) becomes rows and algebra values addressed by coordinates; everything that is **execution** beyond Datalog
(nondeterministic rewriting, `superpose`/`collapse` bags of new terms, grounded calls, dependent-type checking, `new-state`)
stays in the Hyperon interpreter and is used as the oracle that our answers are diffed against. No atom changes meaning:
the parser is Hyperon's own, the `Space` trait is the only seam, and refusal is typed, never approximated.

## 2. What is settled before this plan starts (VERIFIED-IN-CODE, from the inventory)

- The seam is `hyperon_space::Space` (`common`, `query(&Atom) -> BindingsSet`, `visit`, `as_any`) + `SpaceMut::{add, remove, replace}`;
  `match` hands its pattern unchanged to `space.borrow().query(&pattern)` (`stdlib/core.rs:163`). `ModuleSpace` splits conjunctions
  itself, so `LanceSpace` is bound as a named space, never as `&self` (D-MTP-2).
- `Atom` is `Symbol | Expression | Variable | Grounded` (`hyperon-atom/src/lib.rs:827-849`). A Narsese statement `<S --> P>` is the
  MeTTa fact `(--> S P)`: `lance-graph-nal::guid` already lays an arity-2 fact out as semantic tiles
  `[classid_hi, classid_lo, S, copula, P, id, 0, 0]` (`lib.rs:157-170`); the same tiles carry `(f a₁…aₖ)` for k ≤ 5.
- relalg describes one table's row set (`Rel`, `Measure`) and rewrites it before it runs; it has **no join and no μ** by design,
  and names a hop as a D-CMR-5 preimage over coordinates, never a foreign key (`relalg-v1.md` §4). D-MTP-1 measured the hop
  both ways (`Gather` semijoin and the range preimage, D-MTP-1b); D-RA-4 names the fixpoint driver d′.
- `BindingsSet` is a bag (`hyperon-space/src/lib.rs:886`); `Measure::Keep` is a mask. The `b4_nondeterm` fixture counts duplicates.
- Hyperon's own engine: join_10000 panics (`trie.rs:539`, `TK_MAX_EXPRESSION_SIZE = 1 << 10`), anc_800 takes 417 s; PeTTa stays
  near its 0.32 s startup; datafrog does join 10⁶ at 65–69 ns/pair (D-MTP-0/6). These are the bars, not our primitives.
- Truth algebra is shared across OpenNARS, ONA, PeTTa and us; **argument order is not**: lance-graph's abduction/induction match
  OpenNARS Java, ONA and PeTTa use the swapped convention (`Truth.c:84-92` vs `TruthFunctions.java:250-287`). A parity harness binds
  premises by rule shape. ONA's revision clamps `c = min(0.99, max(w2c(Σw), c₁, c₂))`; ours and OpenNARS's are exact Σw.
- ONA's `Term` is `Atom atoms[COMPOUND_TERM_SIZE_MAX = 64]`, `ATOMS_MAX = 255`, "the encoding used is that of a binary heap"
  (`Term.h:34-37, 54-59`; `Config.h:140-142`): a fixed-width, u8-indexed compound term. That is the shape ledger R10 lacks.
- No foreign system carries a 59..63 or a 40..42. The epistemic ladder and the Pearl mask stay ours; nothing is imported into them.
- None of the 34 OGAR recipes implements a NAL operation (`nal-recipe-audit/recipes.tsv`); their bytes are names, not semantics.

## 3. Decisions (each with the alternative it rejects)

**A. One Datalog AST, two front ends, as relalg values (DECISION, extends D-RA-3).** `Fact(rel, args)`, `Rule(head, body)`,
`Query(body, project)` are values over coordinates: relation = `(functor classid, arity)`, argument = lane index, constant =
codebook id, variable = a join column. MeTTa (via `SExprParser`) and Narsese (via `Narsese.rs`) both produce it. Rejected:
a MeTTa-specific IR beside a NAL-specific one (two spellings of a join), and Datalog text as a query path (strings are labels).

**B. The join is the D-CMR-5 preimage, lowered to what exists (DECISION).** A shared variable between two body atoms lowers to
`Pred::EqU32Via { fk, key, v }` when one side is a constant and to `MaskOp::Gather { lane, foreign, under, dst }` (gated semijoin)
otherwise; the range form (D-MTP-1b) when the lane is address-ordered. No new byte. Rejected: a hash join inside mask-risc
(a second engine), and a `Rel::Join` that materialises an intermediate relation (the exact cost §2 measured in Hyperon).

**C. Multiplicity is a terminal, not a post-pass (DECISION, the b4 kill condition).** A `query` answer is a bag of `Bindings`.
The terminal emits projected key tuples with their multiplicity; the adapter builds `Bindings` from them. Whether this is a
composition of `Keep` + `Gather` + `CountKeyRunsU32` or needs one new terminal is D-ASM-3's measurement, with the
`isa-anti-lasagne-warden` insufficiency proof required for the latter. Rejected: dedup-then-count (loses `superpose` semantics).

**D. Recursion is a driver outside the engine (DECISION, = D-RA-4 / D-MTP-3).** Datalog-fragment `=` rules are stored in
`LanceSpace` as algebra values; `match` over a derived relation answers with the closure computed by d′, semi-naive, one fold per
iteration, bounded, refusing an unbounded closure rather than truncating. The interpreter never sees these rules in `&self`.

**E. Compound terms take ONA's shape (DECISION, new: D-ASM-5).** A compound term is a fixed 64-byte tenant of u8 atom indices in
binary-heap order, hashed once, with `Term_Equal`, `Term_ExtractSubterm`, `Term_Complexity`, `Term_Hash` as the parity surface.
Subterm extraction is index arithmetic, so a compound-term predicate is a strided u8 match (`MatchFacet16Strided` family), not a
tree walk. This lifts ledger R10 for NAL-3/4/5 statements and for MeTTa expressions deeper than one level. Rejected: a pointer
tree (allocation in the bulk loop) and widening the 16-byte facet (the key is canon).

**F. NAL rules are Datalog rules with a truth-function label and an engine-order tag (DECISION, new: D-ASM-6).** ONA's 135
`R1`/`R2` lines (`NAL.h`) and OpenNARS's syllogistic/local/compositional tables are harvested into the AST as data: premise shapes,
conclusion shape, truth function, and which argument convention the source engine uses. The per-row truth law stays D-NALP-4's
single candidate byte; everything else is the join of D-ASM-2. Rejected: porting `RuleTables.reason` or `RuleTable.c`.

**G. Three oracles, never averaged (DECISION, = D-MTP-5/6, D-NALP-7).** Hyperon fork, PeTTa, and datafrog for Datalog answers;
ONA `./NAR shell` and OpenNARS `RunNal.java` for NAL truth. Divergences are recorded per engine. An answer that only one oracle
produces is reported, not adopted.

**H. What stays in Hyperon, by name.** `superpose`/`collapse` over constructed terms, `cons-atom`/`decons-atom`, grounded ops,
`new-state`/`change-state!`, dependent types (`d3_deptypes`), and any `=` rule with a function symbol in its head. `LanceSpace`
refuses these at `add` with a typed error; they live in the module's own `GroundingSpace`.

## 4. Deliverables and gates

| D-id | wave | work | gate / falsifier | depends on |
|---|---|---|---|---|
| D-ASM-0 | W0 | this plan, the inventory, board rows | — | — |
| D-ASM-1 | W1 | **Binder**: `(functor, arity) → ClassView` registry + label codebook; `add` of `Symbol`/flat `Expression` facts → rows; `Variable` → join column; `Grounded`/nested/typed-state → typed refusal. Reuses `lance-graph-nal::guid` tiles. Probe: `e1_kb_write` facts round-trip through `add`/`query`/`remove` | every fixture atom resolves to the same address twice; a refused atom is named by kind; no string survives bind | D-MTP-2 scope |
| D-ASM-2 | W2 | **Join as preimage** in relalg: a body-level shared variable lowers to `EqU32Via` / gated `Gather` / range (D-MTP-1b); `normalize` keeps its rewrites; symbolic `∅` still short-circuits | pairs == datafrog as a bag at n = 10²..10⁶; join_10000 answers where Hyperon panics; `Program` byte-identical whichever front end produced the AST | D-RA-3 |
| D-ASM-3 | W2 | **Bag terminal**: emit projected tuples with multiplicity; adapter builds `BindingsSet` | `b4_nondeterm` multiset equal to Hyperon's; a dedup arm fails it (the falsifier must fire); new terminal only with an insufficiency proof | D-ASM-2 |
| D-ASM-4 | W3 | **`LanceSpace`** (`Space + SpaceMut`) as a named space over D-ASM-1..3; three-way harness Hyperon · PeTTa · LanceSpace on `b4`, `c1`, `e1`, `join_n`, `anc_n`; refusals counted | Hyperon golden corpus unchanged (32/33); every LanceSpace answer equal to Hyperon's as a normalised multiset or a named refusal; PeTTa divergences logged | D-MTP-2, D-MTP-5 |
| D-RA-4 / D-MTP-3 | W4 | **μ via d′** for Datalog-fragment rules stored in `LanceSpace` | `anc_n` identical; no fixpoint state in mask-risc; unbounded closure refused | D-ASM-4 |
| D-ASM-5 | W5 | **ONA-shaped compound-term tenant**: 64 × u8 heap, `Term_Hash`/`Equal`/`ExtractSubterm`/`Complexity` parity vs ONA on its `examples/nal/*.nal` terms; strided predicate over subterm positions | byte-identical hash and extraction on every fixture term; a term wider than 64 atoms refused, never truncated; `v3-envelope-auditor` LAYOUT-CLEAN (a tenant, no stride change) | D-ASM-1 |
| D-ASM-6 | W5 | **Rule catalogue as data**: ONA's `R1`/`R2` table and OpenNARS's syllogistic tables harvested into Datalog rules with truth-function label + engine argument-order tag; `Narsese.rs` as the NAL front end into the same AST | each harvested rule re-emits its source line; the order tag is proven by running one fixture through both engines and binding by shape | D-ASM-1, D-NALP-4 |
| D-NALP-5 | W6 | syllogistic population ops = D-ASM-2 join + D-NALP-4 law, now over the shared AST | equals the scalar rule pair by pair; equals ONA and OpenNARS under their own order tags | D-ASM-2, D-ASM-6 |
| D-ASM-7 | W7 | **Mining seam** reserved: `lance-graph-cog` as a driver tier (support = D-ASM-2/3 count, specialisation = lane census, surprisingness via `arm-discovery`), oracle `hyperon-miner`; its own plan, not authorized here | — | D-ASM-3 |

**Kill conditions.**
- D-ASM-3: if no composition of existing terminals reproduces `b4_nondeterm`'s multiset and the insufficiency proof for a new
  terminal fails the warden, the MeTTa adapter answers counts only and says so; it does not dedup.
- D-ASM-4: any answer that differs from Hyperon's and is not a named refusal is a lowering bug, never a re-pin of the fixture.
- D-ASM-5: if heap-order extraction cannot be a strided match (measured: a tree walk per row), the tenant is rejected and R10 stays open.
- D-ASM-6: a rule whose argument order cannot be proven by shape on both engines is kept out of the catalogue.
- Anywhere: a new opcode or vocabulary byte without an execution-level insufficiency proof; any change to what a `Symbol`,
  `Expression`, `Variable` or `Grounded` atom means; any Hyperon interpreter code ported rather than called.

## 5. Order and ownership

W1 and W2 can start together (binder; join + bag terminal). W3 needs both. W4 needs W3. W5 needs W1 and, for the law, D-NALP-4's
measurement. W6 needs W2 and W5. W7 is a separate plan. Wardens that apply: `isa-anti-lasagne-warden` (D-ASM-2/3),
`zero-copy-warden` (D-ASM-1 binder must not hold atoms), `v3-envelope-auditor` (D-ASM-5), `cypher-lowering-warden`'s rule that a
front end lowers from the bound plan, never the raw AST, `fold-carrier-scientist` (bag vs set carriers in D-ASM-3),
`premise-auditor` before any option set reaches the operator.

## 6. Open

- HY-Q07: a `query` whose answer is a **new** expression (not a row) — where is it allocated? Out of scope for W1–W6; D-ASM-7 will meet it.
- HY-Q08: `new-space`/multi-space scope vs Lance versions; D-ASM-4 proves isolation on `e1` only.
- Whether `match &self` can reach a whole conjunction without patching Hyperon (D-MTP-2's open point) — if not, the named-space rule is permanent.
- The 59..63 / 40..42 writer for assimilated facts: a MeTTa fact arrives with no evidence; `Revision` refuses `MissingProvenance` by design.
  The stamp has to travel as an atom. Shape not decided.
