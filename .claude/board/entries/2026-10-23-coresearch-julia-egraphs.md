# 2026-10-23 — Coresearch: Julia e-graphs vs egglog / datafrog / Ascent; Lisp vs Prolog for MeTTa

**Status:** exploration map. Nothing is adopted; the operator chooses.
**Question (operator):** "evaluate Julia's egraph capabilities vs egglog datalog datafrog ascent capabilities. I
understand it's lisp adjacent which may or may not be adjacent to the prolog capabilities we want for MeTTa."

## Premise check: PREMISE-SPLIT

"Lisp-adjacent" and "Prolog capabilities" are different properties. They were asked separately.

- **Lisp side (syntax and metaprogramming).** Code is data (`Expr`, quoting, macros run at parse time). Julia has
  this (Julia manual, metaprogramming, SECTION-READ).
- **Prolog side (evaluation model).** Logic variables, unification, nondeterministic search and tabling. Julia's
  core language has none of these (same source).

MeTTa uses both halves: S-expression syntax, and two-sided unification with all-solutions results.

## Anchors (not re-opened)

- **A1.** egglog is the proof until replaced. The proof is egglog's `ProofStore`/`ProofId`, never a string or a
  hash (operator, 2026-10-23; EXO-50 `07c`/`07d`).
- **A2.** The internal proof representation is chosen by measurement. Candidates: relational algebra over DR64,
  the R2IL polyfill, or something NNUE-adjacent.
- **A3.** egglog, eggcc, ruler, Ascent and datafrog are the closest thing to a logical trace route lowered into
  algebra.
- **A4.** Single Rust binary; no float or LLM on the hot path; zero-copy; ns per op.
- **A5.** MeTTa is a declaration IR (`2026-10-14-metta-polyfill-map.md`).
- **A6.** Originals act as oracles (`2026-10-15-metta-datalog-assimilation-plan.md` §7).

## What the scouts found (grades in brackets)

**Julia, never evaluated here before.** Nothing in `crates/`, `.claude/` or `.arxiv/` mentions Julia, Metatheory,
Symbolics or Julog (closed search, ABSENT).

**Metatheory.jl** (MIT), a pure-Julia port of egg.
- Rules are Julia macros of four kinds: directed, equational, disequality, and "dynamic", whose right-hand side
  runs Julia code.
- It has e-class analyses and cost-based extraction, with Float64 costs.
- The paper never mentions proofs, explanations or checking (arXiv 2102.07888, READ-IN-FULL).
- Neither the released version (v2.1.0) nor the 3.0 branch has a proof object (repo, SECTION-READ).
- An unmerged proof branch (`ale/3.0-proof`, last commit 2024-07-20) has an egg-style explanation union-find. It has
  no checker, and its tests are not run (`proof.jl` read in full).

**SymbolicUtils.jl and Symbolics.jl** (MIT) (repos, SECTION-READ).
- SymbolicUtils is destructive, first-match term rewriting with associative-commutative matching. It has no e-graph
  and no derivation object.
- Symbolics is a computer-algebra system on top of it, with rule-based differentiation.

**Julog.jl** (Apache-2.0, v0.1.15, dormant since 2023-02) (README, SECTION-READ).
- SLD resolution with unification. The occurs check is off by default.
- Breadth-first search by default.
- No tabling, and its cut does not behave like Prolog's.
- It returns substitutions, not proof trees.

**egglog, as our probes build it** (rev 9c8396d, VERIFIED-IN-CODE).
- `prove` returns a `ProofStore`/`ProofId`.
- egglog checks the proof before and after simplification and panics if a check fails
  (`src/proofs/proof_extraction.rs:126-139`). `check_proof` is `pub(crate)`.
- Both our probes keep only `proof_to_string` and drop the `ProofStore` (`dr-math-binding-probe/src/relcomp.rs`
  `prove_equation`; `egglog-r2il-probe/src/session.rs:75-108`).
- egglog is bottom-up: no backtracking, top-down queries only via magic sets (egglog paper §7, SECTION-READ).

**datafrog and Ascent.**
- Both are bottom-up fixpoint engines with no provenance (repos, SECTION-READ; closed grep).
- Here they are dev-dependency oracles only, and the tuple-comparison oracle role was withdrawn on 2026-10-21
  (`2026-10-21-eval-egglog-datafrog-ascent.md`).

**MeTTa's actual requirement: two-sided unification and all-solutions results.**
- Variables may appear on both sides of a match and must bind consistently. Loops are excluded (Hyperon paper §3.1.2;
  `hyperon-atom/src/matcher.rs` `match_atoms`, `has_loops()`; SECTION-READ).
- Meta-MeTTa's operational semantics unifies against every `(= t u)` and emits a multiset of results, with no
  backtracking primitive (arXiv 2305.17218, SECTION-READ).
- PeTTa compiles MeTTa to SWI-Prolog clauses; `collapse` becomes `findall/3` (README, SECTION-READ). It is already
  the second oracle (D-MTP-5).

**Embedding Julia** (jlrs, MIT) puts a full Julia runtime (LLVM JIT and GC) into the binary (README, ABSTRACT-ONLY).

## Exploration map

| # | idea | bridge | firewall | verdict | probe / reason |
|---|---|---|---|---|---|
| X1 | Metatheory.jl as the math compiler | DROP | TRAP | **SKIP** | It duplicates egglog without proofs, conflicts with A1, and Julia conflicts with A4. |
| X2 | Metatheory proof branch | DROP (concept only) | TRAP | **SKIP** | Unmerged and unchecked, so egglog is strictly ahead. Its LCA flat-proof extraction is a reference point if we ever write our own proof walker. |
| X3 | Rules whose right side runs host code | DROP | TRAP | **SKIP**, constraint kept | A host computation can enter a proof only as `Fiat`. Constraint: no arbitrary host code in rule bodies on the proof path. Open question below: how egglog justifies a primitive step. |
| X4 | Build rules as typed values, not strings | OPPORTUNITY | PASS | **PROBE P-J1** (S–M) | Rebuild the associativity rules and `prove_equation` through egglog's Rust API. Kill if the API cannot express a rule the string path uses, or if the proof differs. No Julia involved. |
| X5 | Symbolics as a refutation oracle for derivative candidates | WORTH-EXPLORING | CONFLICT: offline refutation-oracle harness, dev-only, refute-only | **PROBE P-J2** (S, non-Julia arm first) | Take the Leibniz candidate plus 2–3 deliberately wrong variants. Check numeric finite differences on random inputs in a dev-only Rust test. If that catches every wrong variant, Julia adds nothing and X5 dies on redundancy. Run the Julia arm only if it misses one. |
| X6 | Julog.jl as Prolog oracle | DROP | TRAP | **SKIP** | PeTTa on SWI-Prolog already fills the slot and is stronger. |
| X7 | Julia metaprogramming ("Lisp-adjacent") | DROP (concept only) | PASS (finding) | **SKIP** | It answers only the syntax half. MeTTa's S-expressions are already handled as a declaration IR (A5). |
| X8 | Two-sided unification with all-solutions results | WORTH-EXPLORING | PASS for the Datalog subset; CONFLICT for tier 3 (Hyperon/PeTTa oracle seam) | **PROBE P-J3** (M) | Run about 10 matches through both Hyperon and PeTTa: repeated variables, variables on both sides, a loop case, and a multi-solution case. Compare the result multisets with the parity map's SUBSET/refuse prediction. Kill if the engines disagree on a case marked SUBSET. Firewall note: the multiset must stay a mask fold, never a `Vec` of substitutions. |
| X9 | Tabling (SLG) | DROP (concept only) | PASS as concept | **PARK** | Bottom-up Datalog to relational-algebra μ (D-MTP-3) is the chosen termination route. A top-down tabled engine would be a second execution IR. A cross-check against SWI tabling, on left-recursive transitive closure, becomes useful once D-MTP-3 exists. |
| X10 | Embedding Julia (jlrs) | DROP | TRAP | **SKIP** | Conflicts with A4. Even an oracle would not need embedding; an out-of-process harness is enough. |
| X11 | Ruler / Enumo rule discovery | WORTH-EXPLORING | CONFLICT: offline rule-proposal factory, never auto-admitted | **PARK** | Offline proposer only; every candidate must be proved by egglog before admission. Its licence is not recorded yet. Not related to Julia. |
| X12 | Keep the proof DAG; checking by replay | OPPORTUNITY | PASS | **PROBE P-J4** (M) | Keep `ProofStore`/`ProofId` instead of the string, which A1 already requires. Take 20 egglog proofs and write a standalone checker that replays the proof steps and congruence without rebuilding the e-graph. Kill if it disagrees with egglog's `check_proof`, accepts a corrupted proof, or needs the e-graph rebuilt. The result feeds A2 by measurement. |
| X13 | datafrog / Ascent provenance | DROP | PASS | **SKIP** | They have no provenance, and their oracle roles are fixed. |

## Answer to the two questions

- **Q-A (Julia's e-graphs).** Julia offers nothing egglog lacks for a proof-carrying compiler, and egglog is ahead on
  proofs. Julia's real advantage is ergonomics: rules are native terms. That gap is ours, and it can be closed in Rust
  with egglog's API (X4).
- **Q-B (Lisp vs Prolog for MeTTa).** Julia's Lisp side does not supply the Prolog half. MeTTa's requirement is
  two-sided unification with all-solutions multiset results, not chronological backtracking. PeTTa on SWI-Prolog
  already covers it as the oracle, and the bottom-up Datalog route covers the subset we lower.

## Open

- **Primitive steps in egglog proofs.** Scout 5 lists `Eval` among egglog's justifications. Whether a step computed
  by a primitive is justified as `Eval`, or collapses to `Fiat`, was not tested. It matters for X3's constraint.
- **Not searched:**
  - whether Julia's lowering still defaults to femtolisp;
  - the per-call cost of jlrs;
  - MORK's licence;
  - whether Hyperon supports tabling (only a grep found nothing);
  - the MeTTa specification PDF;
  - the Ascent/BYODS paper (HTTP 403);
  - Abiteboul, Hull and Vianu (HTTP 503);
  - Julia SMT bindings.
- **Grading note.** One scout read egglog proof code at rev 4b5686b (the local clone), the others at 9c8396d (the rev
  the probes build). The two were not compared.

`coresearch | STATUS: open | OUTCOME: Julia offers no capability egglog lacks for proof-carrying compilation; MeTTa needs unification with all-solutions results (Prolog side), not Lisp | OPEN: P-J1..P-J4 unauthorized; the Eval-vs-Fiat question for primitive steps`
