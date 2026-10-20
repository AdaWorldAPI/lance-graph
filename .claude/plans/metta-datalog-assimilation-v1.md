# metta-datalog-assimilation-v1 — Hyperon / PeTTa / MeTTa atoms assimilated into a Datalog-AST · NAL · R2IL · ONA-shaped substrate, with MeTTa as the oracle

**Status:** PROPOSAL (D-ASM-0 only: this plan, its board rows, and the inventory it rests on). Nothing else is authorized.
**Council:** `/5plus3` ran 2026-10-16 on decisions A–H and the W1–W3 tranche (D-ASM-1..4); the ratified, executable spec is
`metta-datalog-assimilation-v1-council-v3.md` (premise gate PREMISE-SPLIT, 5 savants, 3 reviewers, 0 BLOCK). Where a decision
below carries a ⊘ note, the council's text supersedes it; the original wording stays for the record.
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

**A. One Datalog AST, two front ends, as relalg values (DECISION, extends D-RA-3).** ⊘ Council v3 (R-1b): the values live in
`lance-graph-metta::datalog` as the front end's bound plan and lower into quack `Query`, not in relalg (whose scope is one table's
row set) and not in quack's `bind.rs`; no `Rule` value in W1–W3 (an `=` atom is routed, the value arrives with W4 and d′). `Fact(rel, args)`, `Rule(head, body)`,
`Query(body, project)` are values over coordinates: relation = `(functor classid, arity)`, argument = lane index, constant =
codebook id, variable = a join column. MeTTa (via `SExprParser`) and Narsese (via `Narsese.rs`) both produce it. Rejected:
a MeTTa-specific IR beside a NAL-specific one (two spellings of a join), and Datalog text as a query path (strings are labels).

**B. The join is the D-CMR-5 preimage, lowered to what exists (DECISION).** ⊘ Council v3 (R-1/R-2): the premise gate split this
into an EXISTENCE half (which rows satisfy the body: quack `Filter::{Semijoin, EqU32Via}`, as D-MTP-1 shipped) and an ANSWER half
(which binding tuples, with multiplicity: a keyed count). "In relalg" is struck; relalg gains nothing in this tranche (D-RA-3 stays
behind D-RA-2). A shared variable between two body atoms lowers to
`Pred::EqU32Via { fk, key, v }` when one side is a constant and to `MaskOp::Gather { lane, foreign, under, dst }` (gated semijoin)
otherwise; the range form (D-MTP-1b) when the lane is address-ordered. No new byte. Rejected: a hash join inside mask-risc
(a second engine), and a `Rel::Join` that materialises an intermediate relation (the exact cost §2 measured in Hyperon).

**C. Multiplicity is a terminal, not a post-pass (DECISION, the b4 kill condition).** ⊘ Council v3 (R-1(b)/R-3): the bag is
`Terminal::GroupReduce{Count}` keyed on the projected label lane (`GroupAddr::Local`, or `Pair` for two same-table variables);
`CountKeyRunsU32` is `COUNT(DISTINCT)` and is the dedup falsifier, never the carrier; `b4_nondeterm` never reaches the fold (every
fact is an `=` rule) and is the ROUTING fixture — the multiplicity fixture is purpose-built. Cross-table projections are refused
in W1–W3. A `query` answer is a bag of `Bindings`.
The terminal emits projected key tuples with their multiplicity; the adapter builds `Bindings` from them. Whether this is a
composition of `Keep` + `Gather` + `CountKeyRunsU32` or needs one new terminal is D-ASM-3's measurement, with the
`isa-anti-lasagne-warden` insufficiency proof required for the latter. Rejected: dedup-then-count (loses `superpose` semantics).

**D. Recursion is a driver outside the engine (DECISION, = D-RA-4 / D-MTP-3).** ⊘ Council v3 (R-9): a Datalog `=` rule is a PROGRAM
for d′, not space content; `match` over the derived relation answers the closure only through the EVALUATED call, and `SpaceMut::add`
returns `()` so there is no refusal channel at `add` — `LanceSpace` routes at `add` and refuses at `query`. Wording to be corrected
when W4 opens. Datalog-fragment `=` rules are stored in
`LanceSpace` as algebra values; `match` over a derived relation answers with the closure computed by d′, semi-naive, one fold per
iteration, bounded, refusing an unbounded closure rather than truncating. The interpreter never sees these rules in `&self`.

**E. Compound terms take ONA's shape (DECISION, new: D-ASM-5).** ⊘ Council v3 (R-9, premise gate test 3): a compound term is
INTERNED once, frame-side, in a term table with ONA `Term_Hash/Equal/ExtractSubterm/Complexity` parity; a row carries a 16-byte
reference to it; a subterm-match lane is a derived projection of that table. The "fixed 64-byte per-row tenant" below is struck
(+64 B per row = 12.5 % of the 512-byte stride; a label filed as a row value; ONA itself calls the encoding wasteful for
low-complexity terms, `Term.h:36`). The parity surface and the strided-match idea survive; the per-row copy does not. **The surviving
design (⊘ v3.1, Codex P2 on #1576 — the struck sentence used to follow here and read as live):** the term TABLE holds each distinct
compound term once, in ONA binary-heap order, hashed once; `Term_Equal`, `Term_ExtractSubterm`, `Term_Complexity`, `Term_Hash` are
the parity surface OF THE TABLE; a fact row carries a 16-byte reference (a facet, like every other key) to its term; subterm
extraction is index arithmetic over the table row, so a compound-term predicate is a strided match over the table's projection,
not a tree walk per fact row. This lifts ledger R10 for NAL-3/4/5 statements and for MeTTa expressions deeper than one level.
Rejected: a pointer tree (allocation in the bulk loop), widening the 16-byte facet (the key is canon), and the per-row 64-byte copy.

**F. NAL rules are Datalog rules with a truth-function label and an engine-order tag (DECISION, new: D-ASM-6).** ONA's 135
`R1`/`R2` lines (`NAL.h`) and OpenNARS's syllogistic/local/compositional tables are harvested into the AST as data: premise shapes,
conclusion shape, truth function, and which argument convention the source engine uses. The per-row truth law stays D-NALP-4's
single candidate byte; everything else is the join of D-ASM-2. Rejected: porting `RuleTables.reason` or `RuleTable.c`.

**G. Three oracles, never averaged (DECISION, = D-MTP-5/6, D-NALP-7).** ⊘ Council v3 (R-8): every comparison names its CATEGORY
(bag vs bag, set vs Hyperon's bag deduplicated, count vs bag size) and projects the oracle's answer to it before the diff; the oracle
call is the evaluated call, never a raw `space.query` on a rule atom. Hyperon fork, PeTTa, and datafrog for Datalog answers;
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
| D-ASM-5 | W5 | **ONA-shaped compound-term TABLE** (⊘ v3.1: was "per-row tenant"): one interned heap-ordered term per distinct compound, `Term_Hash`/`Equal`/`ExtractSubterm`/`Complexity` parity vs ONA on its `examples/nal/*.nal` terms; a fact row carries a 16-byte reference; strided predicate over the table's subterm projection | byte-identical hash and extraction on every fixture term; a term wider than the table's heap width refused, never truncated; `v3-envelope-auditor` LAYOUT-CLEAN (a reference lane, no stride change) | D-ASM-1 |
| D-ASM-6 | W5 | **Rule catalogue as data**: ONA's `R1`/`R2` table and OpenNARS's syllogistic tables harvested into Datalog rules with truth-function label + engine argument-order tag; `Narsese.rs` as the NAL front end into the same AST | each harvested rule re-emits its source line; the order tag is proven by running one fixture through both engines and binding by shape | D-ASM-1, D-NALP-4 |
| D-NALP-5 | W6 | syllogistic population ops = D-ASM-2 join + D-NALP-4 law, now over the shared AST | equals the scalar rule pair by pair; equals ONA and OpenNARS under their own order tags | D-ASM-2, D-ASM-6 |
| D-ASM-7 | W7 | **Mining seam** reserved: `lance-graph-cog` as a driver tier (support = D-ASM-2/3 count, specialisation = lane census, surprisingness via `arm-discovery`), oracle `hyperon-miner`; its own plan, not authorized here | — | D-ASM-3 |

**Kill conditions.**
- D-ASM-3: if no composition of existing terminals reproduces `b4_nondeterm`'s multiset and the insufficiency proof for a new
  terminal fails the warden, the MeTTa adapter answers counts only and says so; it does not dedup.
- D-ASM-4: any answer that differs from Hyperon's and is not a named refusal is a lowering bug, never a re-pin of the fixture.
- D-ASM-5: if heap-order extraction over the term table cannot be a strided match (measured: a tree walk per term), the table design is rejected and R10 stays open.
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
  ⊘ DECISION (council v3, premise gate PREMISE-WRONG on "as an atom"): the stamp travels BESIDE the atom as an evidence lane at the
  same coordinate (the nal `Evidence`/class-lane shape), never inside it — an atom has no provenance field, and a stamp inside it would
  make two pieces of evidence two atoms and break revision's grouping by statement key. 53..58 and 59..63 are runtime state written
  over that lane. W1 reserves the lane (written `3` = `NO_PROVENANCE`); its layout is W5/W6 work.

## 7. Validation doctrine — oracle parity first, CE64 enrichment second (DECISION 2026-10-15)

The target is one unified substrate. It is validated in two passes, in this order, and a crate does not enter pass 2 until pass 1 is green:

1. **Each `lance-graph-*` crate is validated against its original as the oracle.** The original's answers on its own fixtures are the
   ground truth for that crate's surface. Divergences are recorded per engine and never averaged (Decision G).
2. **Then CausalEdge64 is used to enrich, to validate, and to measure expansion potential.** Enrichment = the bands the original does not
   have (40..42 Pearl mask, 50..52 plasticity, 53..58 witness, 59..63 epistemic ladder, provenance classes). Validation = the enriched
   answer must still reproduce the original's answer when the enrichment is zero (ZERO_FALLBACK, all-zero CE64 fields). Expansion =
   a falsifiable claim the original cannot make, measured, not asserted.

| crate | original (oracle, pinned) | pass-1 parity surface | pass-2 CE64 enrichment axis | expansion claim (falsifiable) |
|---|---|---|---|---|
| `lance-graph-nal` | OpenNARS Java `4d1bdb0` (`RunNal.java`), ONA `dc4efd0` (`./NAR shell`) | NAL-1/2 fixtures, `revision_points.nal`; truth to one u8 code; argument order bound by rule shape | evidence classes (`INDEPENDENT/DUPLICATE/OVERLAP/NO_PROVENANCE`), W witness, 59..63 written by `pearl::hydrate`/`revise` | a population revision that refuses dependent evidence by class where both originals silently pool it; a statement's epistemic state after an executed intervention, which neither original represents |
| `lance-graph-relalg` | `AdaWorldAPI/datafrog` `edd7cfc` (exact Datalog) | bags of tuples at n = 10²..10⁶; `μ` closures | 40..42 as the question asked of a join (SO/PO/SPO/SP) | a join whose pairs carry a Pearl mask and a topology, so a two-hop answer states whether the middle term is a known intermediate |
| `lance-graph-metta` | hyperon-experimental `main` (interpreter; read at `3f76dc4`, no pin — ⊘ v3.1 NO PINS), PeTTa `1d19a1d` | `b4_nondeterm`, `c1_grounded_basic`, `e1_kb_write`, golden 32/33, `join_n`, `anc_n`; multisets, refusals named | facts assimilated with provenance; `(stv f c)` into F/C; stamp travels as an atom | answers Hyperon panics or times out on (join_10000, anc_800) with identical multisets where it finishes; a `match` whose answer is gated by provenance, which no space in Hyperon or PeTTa can express |
| `lance-graph-ona` (when it exists) | ONA `dc4efd0` (`Term_*`, `Truth_*`, `examples/nal/*.nal`) | `Term_Hash`/`Equal`/`ExtractSubterm` byte parity on the interned heap-ordered term table (⊘ v3.1: not a per-row tenant); sensorimotor fixtures if NAL-7/8 enter scope | compound terms as a tenant addressable by strided match; events as rows with time lanes | a compound-term predicate as one fold over the population where ONA walks concepts one at a time |
| `lance-graph-cog` (when it exists) | `trueagi-io/hyperon-miner` (patterns, surprisingness), PeTTa `lib_pln.metta` second, classic `opencog/miner` read-only | mined pattern set and surprisingness on the shared fixture (HY-Q15) | support counted under provenance masks; patterns scored with `NarsTruth` via `arm-discovery` | a mined pattern whose support excludes dependent evidence; attention left to #1497 |

Rules of the doctrine:
- **The oracle is the original, never a sibling.** `lance-graph-metta` is not validated against `lance-graph-nal`, and PeTTa is not a
  substitute for Hyperon (HY-Q12). CE64 is never the oracle in pass 1.
- **Zero enrichment must be invisible.** With every CE64 band at its zero fallback, pass-2 answers equal pass-1 answers byte for byte.
  A difference is a lowering bug, not a feature.
- **An expansion claim is a fixture the original cannot run.** It is recorded with the original's actual behaviour on that input
  (refusal, panic, timeout, silent pooling) and ours, and graded `MEASURED` only with the command that produced both.
- **Enrichment never changes the original's answer on the original's inputs.** If an enriched reading disagrees with the oracle on an
  oracle fixture, the enrichment is wrong, not the oracle.
- The status vocabulary is the repo's: `MEASURED` with its command, `VERIFIED-IN-CODE` with its location, `HYPOTHESIS`, `OPEN`.

## Connected sources from #1601–#1617 (2026-10-20)

Docs and data merged in #1601–#1617 that this plan owns. They are reading notes or reference data, not implementation; each now links back here. Board: `.claude/board/entries/2026-10-20-prs-1601-1617-plan-links.md`.

- [`.arxiv/2310.18318v1_opencog_hyperon_100_page_reading_bookmark.md`](../../.arxiv/2310.18318v1_opencog_hyperon_100_page_reading_bookmark.md) (#1611): READ LATER: OpenCog Hyperon (100-page background and introduction)
