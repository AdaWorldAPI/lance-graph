# Work handoff: implement D-FLOW-COG after updating from main

**Mode:** ChatGPT Work / Claude Code / equivalent source-editing coding agent.  
**Scope:** Create *coding* follow-up PRs, not another design-only response. **Do not auto-merge.**

## Prep and branch

1. Inspect **latest** `AdaWorldAPI/lance-graph` main and its merged [#1536](https://github.com/AdaWorldAPI/lance-graph/pull/1536) and [#1539](https://github.com/AdaWorldAPI/lance-graph/pull/1539) corrections. Read [D-FLOW-COG](../../docs/architecture/FLOW_COG_RUBICON_ONA_CONVERGENCE.md), [ONA assimilation](../../docs/architecture/ONA_NAL_R2IL_FOLD_METHOD_ASSIMILATION.md) and [temporal agenda](../../docs/architecture/EPOCH_TEMPORAL_GRAPH_RESEARCH_AGENDA.md) in their live current versions.
2. Inspect sibling `AdaWorldAPI/OGAR`, `AdaWorldAPI/ndarray`, `AdaWorldAPI/rs-graph-llm` and `AdaWorldAPI/z8run` and pin head revisions. Establish source-level status; later merged changes supersede the documentation where proven.
3. Before rebasing, inspect `git status`, remotes and unsaved work. Preserve local worktree, no hard resets / force pushes. If branch exists, `git fetch` then `git rebase origin/main` only if clean and appropriate; otherwise create a fresh implementation branch from current `origin/main`. Never rewrite someone else's published commits without explicit consent.
4. Report which work is **SHIPPED / TEST-ONLY / DORMANT / PROPOSED** and which of four proposed crates would genuinely contribute a nonduplicated binding. Don't create crate names as placeholders.

## Design and constraints

Preserve:
- `lance-graph-contract` canonical `KanbanColumn`/`KanbanMove`/`UnifiedStep`, class readings, method and action identities.
- `lance-graph-nal` global truth/inference laws → `lance-graph-r2il` procedures → `lance-graph-mask-risc`/`ndarray` bulk execution. Use independent ONA oracle fixtures pinned to `dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a` with evidence-root overlap and counterexamples.
- OGAR remains owner of `ogar-loco`/`ogar-r2il` r2sleigh lineage, and OGAR action authorization. Do not relocate them.
- `graph-flow-lance` in rs-graph-llm and `z8run-lance` in z8run already share one `lance-abi` registry and handle protocol. Extend existing adapters before introducing `lance-graph-flow` or `lance-graph-z8`.
- `graph-flow` has optional Rig and `graph-flow-lance::llm::Completion` already exists; only add `lance-graph-rig` if a tested model/tool integration cannot be expressed there.
- Keep `lance-graph-cog` a **pure cognitive-method bundle** if created, not another AtomSpace, NARS engine, workflow runner, agent framework or actor scheduler.
- Psychologist Heckhausen/Gollwitzer Rubicon (deliberation/commit/implementation intention), O'Reilly Axon Rubicon (cost-benefit/goal maintenance/credit assignment), and ONA (evidence/f/c/inference/anticipation) remain distinct semantic sources, unified by method calls and existing Kanban/permission gates. OpenPsi motivational scoring is optional and does not turn confidence into utility or calibrated probability.
- No side-effect from NAL confidence, LLM text, desired goal or synthetic internal learning. Intent is not authorization. Use current `ActionInvocation::commit_via` / fresh RBAC and existing audited guards; investigate legacy `commit` bypass.

## Start with one falsifiable vertical PR

Build one *actual executable* NAL W3 revision or other genuinely absent pure method grounded in pinned ONA source:

- Two independent observations with stable identities, f/c and original evidence roots.
- Bind once through existing method/R2IL/mask-risc path; typed result with truth and witness, not merely a count.
- Compare ONA actual output, independent scalar oracle and the compiled path. A duplicate root or swapped law must break the test.
- Compose the resulting *inert* hypothesis with one Rubicon goal score, then show an authorized versus denied action candidate **without executing any external effect**.
- If pure method work is blocked by missing W4 lane output, implement the smallest shared reusable generic lane primitive with SIMD/scalar parity instead of hardcoding ONA execution in an adapter.
- Preserve current test coverage and Cargo path/feature/ABI resolution. Each follow-up implementation PR should be independent and reviewable.

## Then demonstrate adapters only if warranted

Run existing `graph-flow-lance` / `z8run-lance` handle interoperability tests. Add one `CognitiveMethod`-shaped graph task that carries generation-pinned handles, may interrupt and resume, and cannot materialize data without explicit terminal. Reuse the single z8run registry. Verify stale-source failure, duplicate events, denied RBAC, no data in FlowMessage or SessionStorage, and no second runner.

Only then test Axon-inspired retained goal state and outcome credit assignment through real observed receipts, with a feedback-disabled control. Do not conflate psychological stage semantics with the stable current Kanban enum or create unauthorized new discriminants.

## Critical known defects to keep visible

Current audit documented `DispatchToG` as Health-only, mailbox bound declared but unenforced, `DrainTask` unbounded, and a broken supervisor respawn backoff (corrected documentation #1539). Do not route an external workflow through these as if production-ready. Either implement bounded/authorized routing as a dedicated prerequisite PR or isolate the test behind a labelled local laboratory seam.

## End deliverables

1. Concrete implementation PR link(s) with before/after diff, new tests, command output and compilation results.
2. Semantic parity table naming ONA pinned oracle revision and quantified tolerances.
3. Dependency-ownership matrix showing why each newly created crate is necessary; omit unnecessary crates.
4. Negative-control/ablation evidence, including overlap, future knowledge, stale handles, unauthorized action and restart/backpressure when included.
5. Clear list of unfinished W3–W9 and F0–F6 capabilities. **Do not merge automatically and do not claim full ONA/Rubicon assimilation until exercised end to end.**

Start by auditing live main and opening the smallest real code PR.
