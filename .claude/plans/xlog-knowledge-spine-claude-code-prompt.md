# Claude Code handoff: XLOG knowledge-spine donor harvest, research, and smallest integration gate

> Paste this entire file into a **new Claude Code session with read access to the AdaWorldAPI repositories**. This is a research-to-PR assignment; **do not merge**, do not create a new executor, and do not interpret this documentation as implementation authority.

## Mission

Reconcile and harvest the *actual* deployed [xlog.md](https://xlog.md/) documentation against **pinned XLOG source, release tags and tests**, then prepare a cross-repository, source-backed integration plan with one smallest independent behavioral falsifier. The north star is:

**MeTTa/Hyperon (language) ↔ MORK (structural matching) → XLOG (candidate common logic compiler, RIR/PIR/EIR) → OUR existing RelAlg/R2IL/Quack/mask-risc → DT64/DR Cube (execution) → existing epistemic witness/CE64/Thought-Git → AriGraph semantic+episodic view; Mistaber + Legalis + regelrecht consume and validate. Datafrog is an independent Datalog oracle, not another runtime we want to preserve in production. Soufflé, Crepe and Scallop are semantic/compiler donors, not justification for importing duplicate engines.**

This architecture has pre-existing parts: XLOG CubeExecutor, Codegraph Cube, Mistaber Kripke + makor, MeTTa polyfill plans, AriGraph recall and separate provenance/epistemic lanes. Research the residual. Reuse the exact earlier PRs and authority boundaries.

## Source setup, read-only first

Read `AdaWorldAPI/lance-graph/CLAUDE.md`, `AGENTS.md`, and the linked architectural capstones **in full before mutating this repo**. Read the following three authored files:
- `.claude/plans/xlog-knowledge-spine-v1.md`
- `.claude/plans/xlog-knowledge-spine-harvest-20261115.md`
- this prompt.

Use local sibling clones if present; fetch read-only and record exact SHA if absent. Prioritize `AdaWorldAPI/lance-graph`, `AdaWorldAPI/xlog`, `BrainyBlaze/xlog`, `AdaWorldAPI/ndarray`, `AdaWorldAPI/MORK`, `AdaWorldAPI/metta-on-mork`, `AdaWorldAPI/hyperon-experimental`, `AdaWorldAPI/PeTTa`, `AdaWorldAPI/datafrog`, `AdaWorldAPI/mistaber`, `AdaWorldAPI/mistaber-rs`, `AdaWorldAPI/legalis`, `AdaWorldAPI/regelrecht`, `AdaWorldAPI/regelrecht-corpus`, `AdaWorldAPI/AriGraph`, `AdaWorldAPI/ruff`, `AdaWorldAPI/ogar`, plus Soufflé, Crepe and Scallop source when necessary. No automatic dependency addition. Honor licenses, author ownership and reuse limits.

## Mandatory actual inspection (no architecture by README alone)

1. **XLOG site harvest:** enumerate the deployed xlog.md navigation and docs source tree; inspect pages and docs Git history. Record number discovered, accessible, inaccessible, duplicate/redirect, stale and changed pages; do not assert a count until measured. The site advertises `/llms.txt`; a direct fetch returned not-found at this intake. Recover its source index if possible. Extract factual claims on all RIR node operators, optimizer/magic/set/stratification, EIR FAEEL/G91/WFS, PIR provenance/evidence, Free Join/WCOJ/factorized deltas, solver, neural and externally visible caller API. Maintain a page→code symbol/test/release-tag map.
2. **XLOG fork vs site discrepancies:** check xlog.md homepage **2.80× Scallop** claim against `AdaWorldAPI/xlog` README's explicit cross-framework timing caveat; distinguish the separate 2.74× *within-XLOG cached-vs-uncached* result. Confirm actual harness/data/hardware and source provenance; do not recite the disputed comparison as verified. Check docs claims against newer tags and implementation. Mark STALE / UNSUPPORTED / CONFLICT / VERIFIED rather than silently rewriting claims.
3. **Donor archaeology:** for Soufflé, Crepe, Scallop, Datafrog, MORK/MeTTa/PeTTa and Mistaber, write compact capability records with exact source path:line, accepted subset, semantics, and *which existing DT64/Codegraph operation or oracle actually needs it*. Particularly evaluate compiler magic demand, type preservation, symbol identity, n-ary relations, nonmonotonic negation, Set/Bag multiplicity, probabilistic semirings, grounded behaviors, nested term construction, and provenance. Do not infer one formalism is equivalent to another from similar vocabulary.
4. **Existing xlog↔cube census:** re-read and reproduce `lance-graph#2084/#2091`, `xlog#2/#8`; distinguish 230 compiled rule coverage categories, 43 native-tested subset, W5 fixpoint, W7 F100, refusal propagation, changed XLOG commits, and host-row/diff residuals. Compare current source, don't repeat stale snapshots as current measurements.
5. **Codegraph identity and witness:** trace `CALLS/NODE/MAY/IMPL/RCONF/RPOSS`, source generations, inferred vs confirmed, unknown/unharvested effects, source spans and compiler evidence. Locate actual `EpistemicWitness64`, `EpisodicWitness64`, EW64, CE64/Epi5, existing DerivationLedger and Thought-Git owners. Never conflate a proposed label with a shipped type; do not mint new registers, ClassIds, F### words, or field semantics without formal owner decision.
6. **Mistaber and legal crosswalk:** reproduce the existing 11 Clingo goldens; inspect `holds`, inheritance, endorsements, overrides, world identity, source `makor`, 141 citations and DR64³ contraction, then compare to Legalis's `LegalResult::{Deterministic, JudicialDiscretion, Void}` and regelrecht YAML law-model + independent Gherkin BDD. Treat named jurisdictions as different legal systems, not one interchangeable source of truth. Record explicit authority/scope, uncertainty, explanation receipt and human-decision boundary.
7. **AriGraph memory:** inspect the existing `EpisodicMemory::episodic_search` and semantic-hit→episode transition, not just the Python README. Trace how an inferred result or counterfactual is stored, recalled, revised and retracted without overwriting original episode evidence or adding a second heap-owned TripletGraph/BeliefArena.
8. **Exact donor-to-IR ABI map:** for each harvested capability, specify cold/typed frontend IR, MethodRef/lens, F000 or F100 behavioral ownership, admitted Algebra op, borrowed frame/position/mask representation, observable native terminal, independent oracle, and explicit refusal for unsupported shapes. Check the W5/W7, EXO-73/79/84/85/101/108/143/152/153/154 and EXO-50 owner records; preserve Amendment E (#1910), typed refusals and forbidden Gather. Source facts are not legal/epistemic authority.

## Minimal discriminating experiment (propose, then smallest approved lab implementation)

Make a **domain-neutral synthetic** two-authority legal claim with:
- same proposition `P` asserted in world A and inherited in world B;
- one exception/override with a separate source root;
- one confirmed Codegraph source fact and one only-possible dispatch, plus one missing-coverage fact;
- two derivation paths sharing one root (must count that root once);
- one counterfactual branch that retracts the exception source but leaves the observed world untouched;
- an AriGraph episodic observation before and after revision, so historical attribution is inspectable;
- one discrete Legalis deterministic result, one `JudicialDiscretion` case, and one regelrecht Gherkin example mapping only a legally comparable *synthetic* subset;
- an XLOG common relational subset compared against parsed-AST naive/Datalog Datafrog, and against Mistaber Clingo goldens where applicable. EIR/PIR require separate oracles, not implicit equivalence to Clingo.

**Positive expectation:** world-specific holdings, source-root independence, one recovered conclusion upon retraction, preservation of original episode, and correct explicit DISCRETION/ABSTAIN boundary.  
**Negative falsifiers:** remove override-block mask, collapse world identity, substitute `possible` for `confirmed`, duplicate a supporting source, lose missing-coverage grade, bypass tenant fence, use a stale frame/plan, mutate observed state during hypothetical revision, or silently treat refused SCC as empty. Each negative must produce disagreement or typed REFUSE/ABSTAIN.  
**Minimum evidence:** normalized result masks / fold checksums, per-world holdings, root identities, direct-vs-inferred grades, revision generation, route status, exact source/commit, typed refusal, native terminal materialization count, cold/hot allocations, test command, independent oracle command, deliberate disable/red run. No invented benchmark.

If the operator permissions or actual code boundaries prevent this fixture, STOP the implementation stage and deliver only a precise blocked gate/decision packet. Do not resolve the problem by a second evaluator or invented new ABI.

## Deliverables expected from this session

**A.** `XLOG_SITE_HARVEST.tsv` covering all discovered docs with source URL and code/release anchors, completeness metrics, drift and unknowns.  
**B.** `DONOR_TO_IR_MATRIX.md`: Soufflé/Crepe/Scallop/Datalog/Datafrog/MeTTa/MORK/PeTTa/Mistaber plus XLOG and each Codegraph/DT64/Legalis/regelrecht/AriGraph consumer, per-operation admission/refusal.  
**C.** `SEMANTIC_LOSS_LEDGER.tsv`: every conversion that loses worlds, source roots, type/arity, Bag counts, effect order, causal/epistemic grade, source identity, tenant/scope, or revision.  
**D.** `EPISTEMIC_WITNESS_OWNERSHIP.md`: find actual shipped types/owners and prove whether `EpistemicWitness64` is an alias, proposed mapping or missing residual.  
**E.** `DIFFERENTIAL_GATE_PLAN.md`: ordered P0→P7 sequence with baseline, fixture inputs, exact expected results, independent oracle, kill switch, native ABI counters, refusal, EXO owner and pass/fail evidence classification.  
**F.** A **single smallest draft PR** implementing only the first explicitly authorized lab gate if owner constraints allow it; otherwise make it documentation-only and list exact missing approvals. Draft PR, never merge. Crosslink the old PRs and current documents; no duplicate execution engine.

Write a succinct executive review at the top: `REUSE NOW / LAB-ONLY / MISSING / REFUSE / OWNER DECISION`. Grade every statement `SOURCE-OBSERVED | REPRODUCED | LAB | PROPOSED | UNKNOWN`. Record untouched source commits and build environment. Use the project's append-only board conventions rather than silently editing prior verdicts.

## STOP conditions

- Any new hot path requiring tuple reconstruction, JSON soup, `Vec<row>` as population owner, source string interning during execution or copy-out/copy-back.
- A source fact promoted into a legal verdict, method execution, memory truth, causal cause, epistemic knowledge or proof without independently admissible witness.
- Treating EIR/PIR as mere RIR, Clingo stable models as FAEEL, NARS truth as Scallop probability, a MORK Set as MeTTa Bag, or a hypothetical thought revision as an observed mutation.
- Modifying upstream XLOG CUDA runtime, Legalis/regelrecht law engines, MORK, MeTTa or AriGraph before explicit scoped authorization.
- Unsupported negation, unbounded arity, generic Gather, unresolved dynamic dispatch, untracked source effects or no permission proof.
- Any benchmark or source claim without command, source SHA, semantics, and independent control.
- No rewrite of existing PR/board histories, no auto-merge, no manufactured acceptance.

Finish by answering: **What can XLOG uniquely contribute as the compiler of witnessed relations that the existing Cube cannot already do, and what must intentionally remain owned by MORK, the legal domain systems, and AriGraph?**
