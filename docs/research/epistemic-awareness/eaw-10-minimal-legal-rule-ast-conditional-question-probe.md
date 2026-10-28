# EAW-10 — Minimal legal rule AST: 'es kommt darauf an' under exact counterfactual folds

**2026-10-27. Source-independent, docs-only RESEARCH PLAN. No real statute was compiled, no tax/legal conclusion, tests, model training, production integration or merge authorization.** Research spine [EAW-6 #1826](https://github.com/AdaWorldAPI/lance-graph/pull/1826) / EAW-7 strict worlds / EAW-8 sparse NNUE / EAW-9 blind rediscovery. Source/domain gate [EAW-5 #1825](https://github.com/AdaWorldAPI/lance-graph/pull/1825). This is explicitly a tiny heuristic experiment, NOT a legal ontology or legal-advice engine.

## Research question
Can the same **falsifiable discovery pipeline** proven in a chess testbed identify a *minimal deciding question* for a finite rule system with source-labelled independent supports, conjunctive obligations, qualified exceptions, temporal applicability and multiple typed destinations? Output conditional explanations retaining different support paths, never a single unsupported verdict. Learning may prioritize *which missing fact to investigate*, and deterministic rule AST/RelAlg alone evaluates the declared rule semantics.

## Do not duplicate earlier work
- [#1814](https://github.com/AdaWorldAPI/lance-graph/pull/1814): source-attested logical ON×WHEN×DO typed method identity, K3 guard AST and multiple mappings to existing recipes.
- [#1819](https://github.com/AdaWorldAPI/lance-graph/pull/1819): candidate ON×DO bitmap + witnessed WHERE factored Quack lowering; no multiple-method collapse, stale Keep or invented OR/SUM.
- [#1821](https://github.com/AdaWorldAPI/lance-graph/pull/1821): explicit relationship presence vs source-covered absence vs undiscovered.
- [#1822](https://github.com/AdaWorldAPI/lance-graph/pull/1822): CE64 Epi5 categorical evidence and declared activation; no causal certification from learned scores.
- [#1823](https://github.com/AdaWorldAPI/lance-graph/pull/1823): align 3 input Boolean planes for TERNLOG; symbolic/numeric and K3 evidence remain separate.
- [#1824](https://github.com/AdaWorldAPI/lance-graph/pull/1824): multiple valid rules converge *inside one world*, incompatible sibling worlds stay separate; conditional synthesis is a candidate, not law.
- [#1825](https://github.com/AdaWorldAPI/lance-graph/pull/1825): *real* Odoo/OGIT/legal harvest is source/owner pending. Odoo `balance→debit,credit` is not a verified dual-ledger case; actual taxable amounts and lawful routing must be attested.
- LegalRuleML, LKIF, ELI, LOMO, Akoma Ntoso and OpenFisca/Catala are **cold comparison targets only** after verifying exact version and source semantics. No ontology's existence itself authorizes an executable law.

## Synthetic miniature ruleset (independently specified reference semantics)
Typed case identity C; Boolean/K3 facts A,B,D,E,G, each with source/version/knownness; decision target Q; taxable amount N; two destinations Ledger_X and Ledger_Y are **typed and distinct**, with a test-only source-defined routing relation.

- R1: if A applies, independently supports status Q.
- R2: if B applies, independently supports status Q. Two support paths yield same Q value but both derivation roots persist; removing A alone while B holds changes explanation without changing Q.
- R3: if D and E both apply, supports a second status S (joint hyperedge); neither is individually sufficient.
- R4: if G holds in a declared legal/scope window, *derogates from R1's contribution* (not R2); never generically overrides all rule results.
- R5: if Q, select rate parameter r(t,j,c) by **explicit test-local rule** (time t, jurisdiction j, category c). Computing amount includes source-defined currency, precedence, rounding and sequence (no implicit OR/SUM).
- R6: route ONE qualified determination into Ledger_X and Ledger_Y under two independently attested test-only destination permissions. This is separate from debit and credit fields of a single journal entry.
- R7: K3 UNKNOWN in A or G is an unknown applicability state, not source-proved absence or false. Need complete source coverage to claim a negative.
- R8: effectful posting of a journal/legal action is always inert PROPOSAL/REFUSE; no real authorization comes from deterministic result, NNUE score or CausalEdge64 category.

Define an independent scalar/AST oracle whose rule priority, support expression, temporal scope and routing are explicit. AST can be analyzed structurally by CFA/RelAlg/Datafrog; a monotone positive subset may fixed-point fold under an admitted law, while defeasible exception or unstratified negation requires specialized verified semantics/refusal. No extra hot executor.

## Exact question discovery ('it depends on...')
For each initially UNKNOWN fact/qualification, T1..T63 represent *independent possible completions* of a pinned partial case, not samples from a statistical population. Enumerate enough assignments to compare **Q value**, Q derivation/support roots, subordinate amount, temporal interpretation, ledger routing and effect proposals, each in separate world. Deduplicate only *logically equivalent outcome projections with equal relevant support witnesses*. Calculate smallest distinguishing predicate set / counterexample over checked worlds; report conditional decision tree and what question would separate outcomes. If complete enumeration is impossible, label result bounded/incomplete and do NOT claim universal minimality. Search ordering may use a frozen NNUE-style feature ranker; exact qualification always stays with reference AST.

The 64^n game matrix is a logical factorization mental model, not 64^n materialized worlds or an exhaustive procedure. T0 immutable, up to 63 resident hypothetical worlds with generation IDs; no sibling world writes merged. Real legal changes/time intervals can require bitemporal modelling (law effective date != historical observation date), explicitly left source-owned until real source harvest.

## Proposed test-only staging (independent approvals)
**L0:** independently specify synthetic rule AST, complete truth table and two distinct ledger identity/permission fixtures. Keep scope small enough for exhaustive scalar oracle (e.g. 5 unknown binary facts = 32 assignments, plus explicitly typed UNKNOWN cases).
**L1:** compare source-qualified logical ON/WHEN/DO × DR64³ relationships vs pure scalar oracle. Verify K3 under NOT and guards and exact typed provenance of both supports; same Boolean Q need not imply same explanation or numeric/ledger projections.
**L2:** compare RelAlg/Quack watched WHERE & TERNLOG candidates under witness equality with independent oracle; no equality from same local ordinals. Charge preparation and fresh bind; enforce masks inside source lease only.
**L3:** Thought-Git sealed 0+63 worlds: within-world valid multiple supports, across-world isolation, exact counterexample discovery and minimal distinguishing condition on fixed small truth table. Never count hypothesis branches as independent evidence.
**L4:** simple learning baseline only: seeded random question selection, deterministic information-gain or minimal split heuristic, ARM co-activation, optional NNUE feature-ranker trained on **disjoint synthetic families**. Compare number of exact probes required to locate all material decisive conditions and explanation errors. A *score* can prioritize query, never establish law.
**L5 (later separate source owner):** source-pin one real LegalRuleML/OpenFisca/Catala or Odoo tax fragment with independent policy owner, as-of versions, genuine rates/exception semantics and effect permissions; until then synthetic labels are NOT legal interpretations. Full independent real-source oracle obligatory before claiming legal utility.

## Falsifiers / negative controls (NOT RUN)
| ID | Counterexample | Required |
|---|---|---|
| LAW-01 | A and B independently support Q; disable only A | Q stays true while A root disappears, B root remains |
| LAW-02 | D AND E required; D true E false/unknown | S false/unknown by K3, never unconditional |
| LAW-03 | G derogates only R1, B independent | R2 contribution intact; no global last-wins |
| LAW-04 | Q unchanged, time/rate/rounding changed | Numeric N and intermediates diverge visibly |
| LAW-05 | Qualified determination routes to Ledger_X and Ledger_Y | Both identities and permissions preserved, not aliased by ordinal |
| LAW-06 | Q absence deduced from non-discovery or incomplete scope | UNKNOWN, not FALSE |
| LAW-07 | Mutate future interpretations, source policy or legally effective date | Historical strict output rebinds/refuses or remains pinned, no hindsight leakage |
| LAW-08 | Two distinct source versions share local guard ordinal | Refuse unqualified join, no cross-version rewrite |
| LAW-09 | Two counterfactual sibling postings unioned | Reject as a real-world outcome; protected alternatives stay independent |
| LAW-10 | NNUE high score proposes false exception/causal certainty | Preserve prediction as candidate, independent rule oracle and source-owner veto |
| LAW-11 | Minimal deciding set derived from only 63 sampled worlds vs larger unknown space | Label bounded, no universal minimality claim |
| LAW-12 | An apparent 'independent' legal authority copied from one root | No confidence amplification, roots distinct |
| LAW-13 | Real tax or dual-ledger claim sourced only to Odoo synthetic #44 | REFUSE; first independently source-harvest exact Odoo behavior |
| LAW-14 | A result grants effect permission | Refuse source writes under all speculative worlds |
| LAW-15 | Heuristic/NNUE fails deterministic information-gain baseline after costs | Park learned arm without affecting exact inference |

## Measurable result / owners / gate
For each test, print conditional decision graph, exact valuations and witnesses, truth/knownness masks, support paths, counterexamples, numeric+two-ledger projections, independent scalar parity, minimal question set within declared coverage, count of oracle evaluations, amortized costs, learned predictor generalization and abstention. Separate (1) exact math and logic, (2) complete finite-domain proof, (3) real-law applicability, (4) authority to effect. No hidden promotion.

EAW-5 and actual source owners own law harvest; EXO-46 binds source identity/focus, EXO-73 recursion, EXO-98/148 thoughts, EXO-109/133 oracle/source quality, EXO-127/135 algebra and legal admission, EXO-140 Quack gather, EXO-146 CE64, EXO-152 numerics. EXO-50 is append-only; do not imply CC cross-account ACK. Closed-unmerged operator-rejected [#1812](https://github.com/AdaWorldAPI/lance-graph/pull/1812) remains forbidden as a source of ideas, test evidence or implementations.

**Outcome hypothesis:** A correct system can answer 'it depends on A or B, except under G; the tax amount also depends on t; both ledgers remain distinct' with exact supporting derivations and a precisely scoped question, while NNUE merely helps choose the next unknown to inspect. Chess supplies a temporal-leak-safe development method; it does not validate statute semantics.
