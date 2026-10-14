# Claude Code handoff: D-ORACLE-0, oracle fidelity before H11

You are operating in AdaWorldAPI/lance-graph. This is a **research-first, read-first** handoff. Operator intent and gates are recorded in .claude/plans/d-oracle-fidelity-matrix-v1.md; provenance anchors in .claude/knowledge/oracle-fidelity-reference-ledger-v1.md. Read both entirely before action. Rebase your working branch on current main where appropriate, without damaging uncommitted operator work.

## Non-negotiable principles

1. **Keep original oracles independent.** Read and version-freeze OpenNARS Java (Pei Wang NAL), ONA C, OpenPsi, trueagi-io/hyperon-experimental Rust/MeTTa. Never rewrite their rules as Ada approximations and then claim oracle fidelity. Run their own test suites/CLIs where feasible; if unavailable report blocked with cause.
2. **Do not blend meaning.** NAL S->P, ONA runtime inference, OpenPsi motivational/action behavior, Hyperon unification, Pearl causal proof, Shannon entropy, Gadamer interpretative admission and grammar SPO have distinct contracts. Shared byte format != shared semantics. Explicitly test that syntax-complete SPO does not automatically count as Pearl counterfactual/intervention.
3. **Run 144 pre-registered candidate fixtures** distributed 26/20/18/16/20/16/16/12 across NAL, evidence, ONA, OpenPsi, Hyperon, grammar, Pearl, and symbolic class/method/substrate. They are PLANNED, NOT ALREADY IMPLEMENTED. First create a fixture ledger with input, original SHA, native invocation/output, expected gold, tolerance, and kill criteria. No invented test outcomes. Track excluded/non-comparable honestly.
4. **64 Auditable Thought alpha channels** hold independent model-specific hypotheses + source/event/premise lineage + operation + interpretation context + time + stable ID/epoch. 64 are simultaneous slots, not 64 total experiments or unlimited provenance compressed into u6. Verify channel reuse cannot conflate roots.
5. **Class/Schema/Methods and Datalog are existing logical abstractions.** $var must be algebraically composable before $var.think(). Do not resurrect the old pointer-chasing/session scatter-gather class executor; that implementation's removal did NOT disqualify Class Methods. Use existing Quack/R2IL/mask-risc and immutable contextual ISA; add no model-specific hardcoded opcodes.
6. **No premature performance conclusions.** D-HOLO H1/H5/H6 tested kernels/ranges, not full solutions. H8/H9 logic/math falsifiers precede H11. End-to-end comparisons must count oracle adapter, normalization, seal-time preprocessing, maintained inverses, scratch masks, update work, final payload materialization and replay/audit overhead. Shannon entropy of declared variables != NARS confidence; normalization means repeated algebraic form, not guaranteed lossless entropy compression.

## Work order

A. Inventory and source: read the above two documents, PRs #1314/#1319/#1321/#1407/#1408/#1410/#1227/#1294, current DeepNSM-v2 belief/reason/fsm/tekamolo/hydrate and CE64 isa/pearl. Reconcile historical PR scopes against current code. Inspect original oracle licenses and native build instructions. Write exact SHA ledger. Read #495/#497/#850 locally if accessible and report true contents, never infer from snippets.

B. Register RED-first falsifiers before changes:
- X17: derived A->C from roots 0+1 then source-0 comes back. Empty stamp may mean missing direct event, NEVER proven disjoint. Preserve source-1 lineage. Control: genuinely new source.
- F/C near c=1: CE64 saturated-weight revision must never silently become NaN-to-zero.
- Grammar-Pearl: syntax S/P/O coverage must not upgrade causal proof.
- Generic Sim nontransitivity; arbitrary relational verb does not transit.
- Original-model agreement/disagreement traces, native ONA task operations and Hyperon binding alternatives.
- OpenPsi/Rubikon mapping must demonstrate correspondences rather than asserting identity.
- Source versus observation event versus derivation root, modulo stamp collisions and 64-slot epoch reuse.
- Interpretation gate changes admissibility without silently modifying raw independent evidence.
Log native expected values before looking at Ada results where possible.

C. Build *test adapters only after* native output schemas are understood. Retain original outputs and explicit normalized comparison views. A refusal/unsupported/non-comparable is valid. Compare models only under documented matching assumptions. For each defect report code anchor, reproduction, expected result, false-positive controls, remedy candidates and cost. Do NOT fix or refactor code in this research PR.

D. Produce a truthful closure table: built / measured / failed / blocked / proposed for each module and each fixture group. List exact commands and artifact hashes; calculate parity per comparable subgroup, not one blended oracle score. Prioritize source dependencies and 6 falsifiers before adding breadth. Return design decisions requiring operator authorization for follow-on code PRs.

E. Only when fidelity and audit gates are met, sketch H11 2^3 end-to-end ablation; keep kernels, full-solution materialization and semantic fidelity separate.

## Expected deliverables

- Versioned/oracle source ledger with native run commands and licenses.
- 144-fixture test matrix with independent Gold/native transcripts, unrun explicitly marked.
- Auditable Thought trace schema proposal, NOT an ABI migration.
- Failure-first reproduction notes for X17, causal-ladder semantics, CE64 saturation, Sim transitivity and hypothesis loss.
- Implementation status and executable-gap list.
- Follow-up PR split by ownership (DeepNSM, CE64, adapters, orchestration, H11). No single fusion PR.
- A short operator decision memo: ADOPT / PROBE / PARK / REJECT with costs and counterexamples.

**Stop condition:** If a planned oracle cannot run, record why, supply a minimal still-useful independent mathematical oracle, and NEVER quietly substitute a new Ada implementation as the "original." Do not merge or modify production reasoning in this task.
