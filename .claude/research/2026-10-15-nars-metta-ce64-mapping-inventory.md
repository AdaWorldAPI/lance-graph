# Mapping inventory — Hyperon/MeTTa/PeTTa atoms and three NARS engines against CausalEdge64 and the polyfill crates

**Date:** 2026-10-15. **Status:** INVENTORY (VERIFIED-IN-CODE where a file:line is given; nothing here changes a meaning or a byte).
**Question:** what does each foreign system carry, where does the same thing already live on our side, and what does the NARS polyfill
(`lance-graph-nal`, `lance-graph-relalg`, `lance-graph-metta`, and the not-yet-existing `lance-graph-ona` / `lance-graph-cog`) still have to grow —
without changing what a MeTTa or PeTTa atom means.

Sources read (local shallow clones, this session; heads as of 2026-10-15): `AdaWorldAPI/hyperon-experimental` `3f76dc4`, `AdaWorldAPI/PeTTa` `1d19a1d`,
`AdaWorldAPI/OpenNARS-for-Applications` `dc4efd0` (ONA), `opennars/opennars` `4d1bdb0` (Pei Wang's Java, "OpenNARS for Research"), `AdaWorldAPI/Narsese.rs` `16d4d18`.
Our side: `causal-edge`, `lance-graph-contract`, `lance-graph-planner/src/nars`, `lance-graph-nal`, `lance-graph-relalg`, `lance-graph-metta`, OGAR `ogar-loco`.
Ten Haiku fact maps, each spot-checked against source, are in the session scratchpad (`nars-map/maps/01..10`). Prior art this builds on:
`.claude/plans/nal-polyfill-v1.md`, `.claude/plans/metta-polyfill-v1.md`, `.arxiv/2026-10_hyperon_metta_to_lance_graph_parity_map.md`,
`.claude/knowledge/nal-recipe-audit/`.

## 0. The two rules this inventory keeps

1. **Atoms keep their meaning.** A MeTTa `Atom` stays `Symbol | Variable | Expression | Grounded` (`hyperon-atom/src/lib.rs:827-849`); a PeTTa atom stays a Prolog
   term; an ONA `Term` stays its 64-atom heap. We map them to addresses and rows; we do not redefine them.
2. **The engine stays mask-risc.** Everything foreign lowers to procedures over existing R2IL calls (nal-polyfill §4 A). A new byte needs an
   execution-level insufficiency proof. Hyperon's interpreter, PeTTa's Prolog, ONA's cycle and OpenNARS's RuleTables are oracles, not engines we port.

## 1. CausalEdge64 — what each foreign system would put in each band

| bits | ours (`causal-edge/src/layout.rs`, `edge.rs`) | OpenNARS Java | ONA | Hyperon/MeTTa | PeTTa |
|---|---|---|---|---|---|
| 0..23 | S, P, O palette indices, u8 each | `Statement(subject, predicate)` over a `Term` tree; copula is the class (`Inheritance`, `Similarity`, `Implication`, `Equivalence`, `language/`) | `Term { Atom atoms[64] }`, copulas are atoms; `ATOMS_MAX 255` (`Config.h:142`) — **an ONA atom index already fits a u8 palette slot** | `Expression(functor a₁…aₖ)`: no fixed arity, no copula; a fact `(parent a b)` is functor + 2 args | same shape as MeTTa, stored as `Space(=, [F\|Args], Body)` Prolog clauses (`spaces.pl:38`) |
| 24..31 F, 32..39 C | u8 codes; `expectation = c(f−½)+½` (`edge.rs:345`) | `TruthValue{frequency f32, confidence f64, analytic}`; cap `1−TRUTH_EPSILON = 0.99` (`TruthValue.java:138`) | `Truth{frequency, confidence} double`; `MAX_CONFIDENCE 0.99`, `MIN_CONFIDENCE 0.01` (`Config.h:172-174`) | none; PLN/NARS truth lives in library atoms `(stv f c)` | `lib_nars.metta` `(stv $f $c)` with ONA's formulas |
| 40..42 Pearl `CausalMask` | SO / PO / SPO / SP + marginals (`pearl.rs`) | no counterpart; rules selected by `RuleTables.reason` link figure (`RuleTables.java:55, 411`) | no counterpart; rule selected by `RuleTable.c` premise shapes | none | none |
| 43..45 direction triad | per-plane sign | `Tense {Past, Present, Future}`, `Eternal = null` (`Tense.java:29-39`); temporal copulas `=/> =\|> =\> </> <\|>` (`Symbols.java:154-159`) | `occurrenceTime`, `occurrenceTimeOffset`, `OCCURRENCE_ETERNAL −1` on `Event`/`Implication` | none | none |
| 46..49 inference mantissa i4 | `InferenceType::to_mantissa` (`contract/nars.rs:64`): Deduction, Induction, Abduction, Revision, Synthesis; +6 Intervention, −6 Counterfactual | rule name chosen by `SyllogisticRules` / `LocalRules` / `CompositionalRules`; not stored on the sentence | rule id in the generated `RuleTable.c` (135 `R1`/`R2` lines in `NAL.h`); not stored on the event | none | none |
| 50..52 plasticity | per-plane hot bit | `BudgetValue{priority, durability, quality}` is the nearest control scalar, a different axis | `Concept.priority`, `Usage` | none | none |
| 53..58 W witness slot | corpus-root handle, 0 = none | `Stamp.evidentialBase: BaseEntry[]`, max 20 000 (`Parameters.java:143`) | `Stamp { long evidentialBase[STAMP_SIZE=10] }` (`Config.h:136`) | none | none |
| 59..63 `EpistemicState5` | `Topology2 × Certification3` | none; nearest is the analytic flag + question/goal punctuation | none; nearest is `Decision.execute` and `predicted_belief` | none | none |

Two consequences for the polyfill: (a) **no foreign system has a 59..63**, so the epistemic ladder is ours to write (`pearl::hydrate`/`revise`), never ported;
(b) **evidential bases are where the three NARS engines disagree most** (unbounded list / 10 longs / our `Stamp(u64)` bitset of ≤ 64 sources at
`planner/nars/belief.rs:31`, and `lance-graph-nal`'s `&[u64]` root list with `INDEPENDENT / DUPLICATE / OVERLAP / NO_PROVENANCE` classes, `lib.rs:234-244`).

## 2. Truth functions — same algebra, two argument conventions

Measured from `isa.rs:228-342`, `planner/nars/truth.rs:55-109`, `exploration.rs:110-124`, `ona/src/Truth.c`, `opennars/.../TruthFunctions.java`, `petta/lib/lib_nars.metta`.

| function | lance-graph (`isa::truth`, planner `TruthValue`) | OpenNARS Java | ONA | PeTTa `lib_nars` |
|---|---|---|---|---|
| horizon `k` in `c = w/(w+k)` | `HORIZON = 1` (`nal/lib.rs:216`), `w/(w+1)` | `HORIZON = 1` (`Parameters.java:59`) | `TRUTH_EVIDENTIAL_HORIZON_INITIAL 1.0` | `(/ $w (+ $w 1))` |
| deduction | `(f₁f₂, f₁f₂c₁c₂)` | same | same | same |
| **abduction(v1,v2)** | **`(f₁, w2c(f₂c₁c₂))`** | **`(f₁, w2c(f₂c₁c₂))`** (`TruthFunctions.java:250-260`) | **`(f₂, w2c(f₁c₁c₂))`** (`Truth.c:84-87`) | ONA's |
| **induction(v1,v2)** | `(f₂, w2c(f₁c₁c₂))` | `abduction(v2,v1)` = same as ours | `abduction(v2,v1)` = `(f₁, w2c(f₂c₁c₂))` | ONA's |
| revision | exact `Σw`, `c = w2c(Σw)`, capped at code 252 (= 0.99) only at commit (`nal.rs:291-317`); `isa` returns `None` on `Σw ≈ 0`; `exploration` adds `1e-9` and `min(0.99)` | exact `Σw`, `w2c` (`:167-181`); cap by `TruthValue` setter | pairwise, `c = min(MAX_CONFIDENCE, max(w2c(w), c₁, c₂))` (`Truth.c:73-74`), special-cases `c = 0` and `c = 1` | not in `lib_nars` head |
| expectation | `c(f−½)+½` | same | same | — |
| analogy | `(f·f_sim, c·c_sim·f_sim)` (planner only) | same | same | same |
| exemplification, comparison, conversion, negation, intersection/union/difference, eternalize, projection, goal-deduction, decompose | **absent** | present | present (`Truth.c:101-212`) | present |
| synthesis (mean of f and of c) | present (`isa` `Opcode::Synthesis`) | absent | absent | absent |

Finding: **lance-graph follows the Java argument order; ONA and PeTTa follow the swapped one.** Both engines encode the same NAL rules
(ONA `NAL.h:66-68` `R2((S-->M),(M-->P)) ⊢ (S-->P) Deduction`, `(A-->B),(A-->C) ⊢ (C-->B) Induction`, `(A-->C),(B-->C) ⊢ (B-->A) Abduction`), so the
difference is which premise is `v1`. A parity harness must bind premises by **rule shape**, never by function name. The revision oracle in
`.claude/knowledge/nal-recipe-audit/` already measured our four right / two wrong laws against OpenNARS; the same fixture run through ONA will differ
by ONA's `max(…, c₁, c₂)` clamp, which is not a bug on either side.

## 3. Term and atom structures — what binds to what

| foreign | shape (verbatim source) | ours today | binder rule |
|---|---|---|---|
| MeTTa `Symbol` | `SymbolAtom{name: UniqueString}` (`lib.rs:131`) | a label → `classid × NodeGuid` or a term id in a class (metta plan §5) | resolve once through the label codebook; never a string after bind |
| MeTTa `Variable` | `VariableAtom{name, id}` (`:230`) | never a row; a join column | shared variables across conjuncts → `EqU32Via` / `Gather` |
| MeTTa `Expression` | `ExpressionAtom{children: CowArray<Atom>, evaluated}` (`:158`) | a row of a class keyed by `(functor, arity)`, one lane per argument | refuse shapes the class does not declare |
| MeTTa `Grounded` | `Box<dyn GroundedAtom>`, `Grounded`/`CustomExecute`/`CustomMatch` traits (`:413-622`) | stays in Hyperon | typed refusal on our side |
| `Space::query(&Atom) -> BindingsSet`, `SpaceMut::{add, remove, replace}` (`hyperon-space/src/lib.rs:175, 237-275`) | `LanceSpace` (D-MTP-2, not built) | the seam; `complex_query` (`:340-368`) splits `(, …)` tuple-at-a-time, so `LanceSpace` must receive the whole conjunction as a named space |
| `BindingsSet` is a bag (`PartialEq` "conceptually unordered", `:886`) | `Measure::Keep` gives a mask, not a bag | multiplicity must come from the terminal (`b4_nondeterm` kill condition) |
| PeTTa space | Prolog clauses per space, `assertz(RawTerm)` (`spaces.pl:38`), `match/4` conjunctive (`:204-220`), `add-atom`/`remove-atom` + `-proven` variants (`:14-179`) | same `LanceSpace` seam; PeTTa is an oracle, not a second adapter | diff PeTTa and Hyperon answers, never merge them into one oracle |
| ONA `Term` | hashed `Atom[64]` heap, copulas as atoms, `ATOMS_MAX 255` | `lance-graph-nal::Statement{s:u16, copula, p:u16}`, `Copula {Inheritance=1, Similarity=2, Implication=3, Equivalence=4}` (`lib.rs:94-102`); compound terms out of scope (ledger R10) | atomic statements only today; `(A * B)` products, sequences `&/`, images are R10 |
| ONA `Event` / `Implication` / `Concept` / `Decision` | `Event{term, truth, stamp, occurrenceTime, occurrenceTimeOffset, processed, creationTime}`; `Implication{…, sourceConcept}`; `Concept{belief, belief_spike, predicted_belief, goal_spike, precondition_beliefs[OPERATIONS_MAX+1=11], priority}`; `Decision{desire, execute, operationTerm, operationID[1], arguments}` | `Belief{s,copula,p,id,f,c}` + sealed lanes `f, c, w, evidence` (`nal/lib.rs:130-144, 260-268`); planner `Belief{stmt, truth, stamp, rung, premises, contradiction}` | **no event time, no goal, no operation** on our NAL row; these are what a `lance-graph-ona` crate would have to declare as lanes, not bits |
| OpenNARS `Sentence{term, punctuation, truth, stamp}`, `Task{budget, sentence}`, `Concept{taskLinks, termLinks}` | — | `lance-graph-nal` has judgment + question only (`Question`, `TruthAtLeast`, `Choice`, `Revision`); no goal `!`, no budget | budget is control, keep it off the edge (sag plan §0) |
| `Narsese.rs` `Term` enum | `Word, Placeholder, Variable{Independent,Dependent,Query}, Set*, Intersection*, Difference*, Product, Image{Ext,Int}, Conjunction{,Sequential,Parallel}, Disjunction, Negation, Inheritance, Similarity, Implication{,Predictive,Concurrent,Retrospective}, Equivalence{…}` (`enum_narsese/term/structs.rs:20-77`), `Punctuation`, `Truth`, `Stamp`, `Budget` | `lance-graph-nal` has no parser yet (`NAL::narsese()` is nal-polyfill W-later) | **the parser is already written, in Rust, in our fork** — the front end should consume `Narsese.rs` and refuse every variant outside the four copulas by name |

## 4. Rules and control — what each engine dispatches on

| engine | rule store | selection | our counterpart |
|---|---|---|---|
| OpenNARS Java | `SyllogisticRules`, `LocalRules`, `CompositionalRules`, `StructuralRules`, `TemporalRules` | `RuleTables.reason(TaskLink, TermLink)` → `indexToFigure` (`:55, :411`) by where the shared term sits | Pearl mask 40..42 chooses *which question*; the inference mantissa names the rule after the fact; the shared-middle-term pairing is the join `EqU32Via`/`Gather` (nal-polyfill §3) |
| ONA | 135 `R1`/`R2` macro lines in `NAL.h` → generated `RuleTable.c` (`NAL_GenerateRuleTable`) | premise-shape match per cycle; `Decision` when desire expectation ≥ `DECISION_THRESHOLD_INITIAL 0.501` | same join; the decision threshold has no home on the edge (intent lives outside CE64, sag plan §0) |
| Hyperon | `=` rules are atoms in the space; interpreter is a 12-instruction dispatch on `EVAL_SYMBOL` etc. (`interpreter.rs:374-460`) | `interpret_stack` | Datalog-shaped `=` rules → relalg `μ` driven externally (D-MTP-3); everything else stays in Hyperon |
| PeTTa | MeTTa → Prolog clauses (`translator.pl`), builtins via `register_fun` (`:449-463`) | SWI-Prolog resolution | oracle only |
| OGAR `ogar-loco::nars` | the 34 recipes as vocabulary bytes `NARS_BASE = DOMAIN_FLOOR = 0x90` (`nars.rs:74`, `lib.rs:353`), tiers Datapath / Control / Gate | `tier_of`, `is_mask_op` | **none of the 34 implements a NAL operation** (`nal-recipe-audit/recipes.tsv`, 2026-10-11); the bytes are reserved names, not NAL semantics |

## 5. What the polyfill crates have, crate by crate

| crate | exists | holds (verified) | missing for the mapping above |
|---|---|---|---|
| `lance-graph-nal` | yes, 953 lines, no tests in `src/` (tests in `tests/`) | `Copula` ×4, `Statement`, `Belief`, `guid` via `FacetCascade::from_semantic_tiles` (classid 4 + S 2 + copula 2 + P 2 + id 2), sealed `Beliefs` with `f, c, w (i32 ×2¹⁶), evidence` lanes, `Aperture` (statement / subject), ops `Question`, `TruthAtLeast`, `Choice`, `Revision` — all `FN = None` (no byte earned); `NalError` ×10 refusals | the parser (use `Narsese.rs`), the syllogism join + per-row truth law (nal-polyfill §4 C, the one candidate byte), `evidence_weight`-based cap parity with ONA's `max(c₁,c₂)` (document only), goals/events |
| `lance-graph-relalg` | yes, 734 lines | `Rel {All, Empty, Plane, Atom(I32 cmp / U32 eq), And, Or, Not}`, `Measure {Count, Sum, Min, Max, Keep}`, `normalize`, `symbolic`, `run` via `FoldDialect` | **no join, no `μ`** (its own docs; D-MTP-1 pre-registration) — the MeTTa conjunction cannot lower yet |
| `lance-graph-metta` | yes, 7 lines + `examples/join_probe.rs` | nothing shipped; deps `mask-risc`, `quack`; `datafrog` fork as oracle | `LanceSpace: Space + SpaceMut`, atom→address binder, `(, …)` → `Rel` joins, bag-valued terminal |
| `lance-graph-ona` | **no** | — | not a port of ONA; the honest scope is (i) an ONA oracle harness (`./NAR shell`, already in `nal-recipe-audit/oracles.md`), (ii) the ONA argument-order adapter for rule binding, (iii) event/goal/operation **lanes** if sensorimotor NAL-7/8 is ever in scope — none of which is a byte on CE64 |
| `lance-graph-cog` | **no**, and the metta plan §6 recommends against the name | — | "cog" would mean Classic AtomSpace/PLN/miner (AGPL, different engine); #1497 owns attention; keep MeTTa under `-metta` |

## 6. Gaps, ordered by what unblocks the most

1. **relalg join (`⋈` on shared variables) and `μ`** — blocks `lance-graph-metta` D-MTP-1/3 and the NAL syllogism alike. One mechanism, two front ends.
2. **Per-row truth law into a lane** (nal-polyfill §4 C: `MapPairU8` table terminal or fixed-point multiply) — the only candidate byte; W4 measurement decides.
3. **Parser**: consume `AdaWorldAPI/Narsese.rs` (`enum_narsese` + `lexical`) in `lance-graph-nal`; refuse by variant name. Do not write a second Narsese parser.
4. **Argument-order binding** for the parity harness: bind `v1`/`v2` from the rule's premise shape; run the same fixture through OpenNARS (`RunNal.java`) and ONA, record both (`oracles.md` precedent).
5. **Truth-function census**: our set is `{deduction, induction, abduction, revision, analogy, expectation, surprise, synthesis}`; the engines also carry
   exemplification, comparison, conversion, negation, intersection/union/difference, eternalize/projection. Add only the ones a rule in scope needs; each as a scalar oracle in `isa::truth` first.
6. **Evidential base width**: `Stamp(u64)` (planner), `&[u64]` roots (nal), ONA 10, OpenNARS 20 000. Pin the policy before revision over large populations is measured.
7. **Doc drift to fix when touched**: `contract/nars.rs` `InferenceType` docs still describe CAM search strategies ("exact CAM search", "DN-tree traversal"), not NAL; `QueryStrategy` is the same vintage.

## 7. What this inventory does not claim

No parity was run in this session; every formula above is read from source, not measured end to end. The Hyperon trie index (`hyperon-space/src/index`) and ONA's
`RuleTable.c` body were not read. The 34-recipe statuses are the 2026-10-11 audit's, unchanged.
