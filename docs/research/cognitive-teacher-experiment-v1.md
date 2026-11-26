# LG-COG-001 — Claude Teacher × RAG × Thinking Config × Feedback: falsifiable protocol

**PROPOSED / DOCS ONLY / PRE-REGISTRATION CANDIDATE (2026-11-14). No experiment has run. No API credits spent.**  
Pair with [mindmap](cognitive-teacher-feedback-mindmap-v1.md) and [typed method/bag seam](cognitive-method-bag-abi-v1.md). Never claim hidden LLM chain-of-thought access: request a deliberately constructed, compact, auditable *reasoning/operation trace*.

## 0. Objective and causal contrast

Can the Claude API generate more correct, executable **typed method plans** for existing Lance-graph capabilities when supplied with (A) no architecture context, (B) evidence-linked RAG, (C) RAG plus ThinkingStyle configuration, or (D) RAG + identical ThinkingStyle configuration + **prior independent oracle feedback**? Goal is to discover the smallest wiring/representation gaps, not train a new LLM, benchmark SQL against SQL, or turn RolloutDB into a planner.

Primary estimand: paired difference in *independently verified fully bound native method traces per task*, D vs A. Secondary contrasts B−A, C−B, D−C; report absolute percentage-point differences and CIs, not only relative lifts. Compare cost per verified trace. With few cases, this is exploratory rather than strong statistical evidence.

### Pre-registered hypotheses and falsifiers

| ID | Hypothesis | Falsifying/negative outcome |
| --- | --- | --- |
| H1 | RAG improves evidenced correct mappings vs no context | B ≤ A, or any gain is solely answer leakage |
| H2 | Thinking Config improves mapping vs RAG alone | C ≤ B, or changes prose style only and not S2–S5 |
| H3 | Verified historical feedback improves transfer vs same config | D ≤ C, or performance disappears on disjoint holdout fixtures |
| H4 | Generated explicit operations can be expressed as existing algebra/typed methods | No additional S2/S3/S4 admit; only invented MethodRefs/JSON programs |
| H5 | Admitted replay is deterministic under frozen version/law/basis | Changed semantics/basis/order passes silently, or teardown changes result |
| H6 | Feedback correctly identifies *missing seams* vs invention | Oracle-independent audit cannot reproduce claimed gap or negative disable fails |

Failure to improve is valid evidence. The external model is not the oracle; no generated confidence or reward authorizes a Rubicon transition or production write.

## 1. Four arms, one controlled model

- **A (bare)**: identical task + output schema + public high-level allowed op classes. No retrieval, no prior feedback, no method table and no explicit config.
- **B (RAG)**: A plus evidence-linked fixed top-K excerpts from indexed source, tests, decisions, Codegraph/Ruff ORE/SLAG and earlier plans. Every excerpt includes SHA/path/range and status. No answer keys.
- **C (RAG + Config)**: B plus frozen StyleFamily (0..11), canonical ThinkingStyle runbook (0..35), seven modulation parameters/preset hash and concise behavioural instruction; no claims of altering model internals.
- **D (RAG + Config + Feedback)**: C *unchanged*, plus top-K outcomes from independent prior development fixtures (oracle, gap code, refusal, constraints, source version, cost). No test fixture solutions and no feedback written during evaluation.

Hold constant: Claude model ID and revision where exposed, decoding settings, max tool iterations, schema, tool allowlist, max tokens, task wording, source Git SHAs, retrieved KB snapshot, tool permissions, model-facing operation descriptions. Randomize order of calls within fixture; each arm runs with isolated context/cache session. If provider controls prevent exact deterministic repeat, report variance across seeds/repetitions.

**Design note:** D tests feedback conditional on config. It does not isolate feedback without config; add a fifth arm only in a separate amendment.

## 2. Frozen inputs and corpus

Pre-registration proposal:
- 12 pilot fixtures (development-only, used to validate schema and estimate token cost);
- 24 development fixtures (build/validate retrieval and prior-feedback collection);
- **72 held-out evaluation fixtures**, 12 in each domain: F###/R2IL method law and binding; DT64/Quack/relalg/mask-RISC; Rubicon/Kanban revision; ThoughtGit branch/squash/generation; Codegraph/ORE/SLAG source-to-method mapping; cycle/BatchWriter/WAL/replay.
- Across each domain include explicit supported, blocked and unknown cases (no cherry-picking). Label source/live vs lab/fixture; unknown must remain unknown.
- 72 × 4 arms × 1 = 288 primary requests. Repeats optional and separately costed, not automatically 3×. Each of 36 runbooks is scheduled twice across the 72 fixture assignments, balanced across domains where feasible. Each family's default and non-default coverage is recorded.
- Freeze fixture text, source metadata and independent expectations **before** running held-out prompts; use SHA digests and independent reviewer signoff.

Per-fixture immutable input:
```json
{
  "fixture_id": "RUBICON-NEG-03",
  "scenario": "Independent evidence increased eligibility, counterfactual not run",
  "facts": [{"name":"phase","value":"Evaluation"},{"name":"effect","value":"IncreaseEligible"},{"name":"counterfactual","value":"NotRun"}],
  "source_commit": "<pinned SHA>",
  "world": {"dataset_ref":"<fixture-id>", "version": 7, "basis_ref":"<digest>"},
  "allowed_effects": ["observe","classify","explain","readonly_probe"],
  "max_steps": 6
}
```

No private customer data; inspect prompt data for secrets, credentials, PII and license restrictions before outbound API requests.

## 3. RAG and feedback controls

**Retrieval is evidence, not truth.** Index source paths, commits, line ranges, doc status, independent tests, claims, negative tests, PR disposition and stale flags into Lance-context ContextStore. Distinguish the source/owner data from a generated description. Freeze index snapshot and embedder version. Same retrieval query construction, K, token cap and sorting for B/C/D. A does not see this material. RAG must not include test ground truth, hidden oracle outcome or previously leaked agent guesses.

D's feedback only uses *completed, independently checked* development trajectories. Record exact retrieval IDs and similarity scores/filters. Prevent cross-contamination by splitting by semantic equivalence class or source contract, not just filenames. Mark any overlap and exclude that case from primary analysis; never silently drop bad retrievals.

Do **not** assume Claude Code has no retrieval tools of any kind; the specific deficit is *no project-wide persistent semantic RAG provided by this proposed integration by default*. Compare against an explicit controlled no-context arm.

## 4. Canonical Thinking Config

Use `StyleFamily` 12-family enum from `crates/lance-graph-contract/src/style_family.rs`, and `ThinkingStyle` 36 runbooks/6 clusters from `thinking.rs`. They are not 432 unconstrained pairs. A family delegates to a canonical representative or runbook with verified `ThinkingStyle::family()`. Seven `FieldModulation` parameters (resonance_threshold, fan_out, depth_bias, breadth_bias, noise_tolerance, speed_bias, exploration) are frozen before evaluation. Fingerprint is available in source; it is **not** enough alone as an exact reversible precision-preserving policy identity. Retain full config + hash/version for replay.

A/B do not receive explicit style directives. C/D receive the *same* family, runbook and parameters. If no executable implementation of a runbook is verified, count its effect strictly as **model prompt conditioning**, not a native execution claim. Metrics on styles are exploratory (two fixtures/style insufficient for ranking).

## 5. Explicit Claude output schema

One compact structured plan, not a presumed transcription of inaccessible private reasoning:
```json
{
  "fixture_id": "RUBICON-NEG-03",
  "observations": [{"claim":"counterfactual absent","evidence_ids":["fixture:counterfactual"]}],
  "hypotheses": [{"claim":"commit not admitted","status":"unverified"}],
  "thinking": {"family":1,"runbook":2,"config_ref":"C1"},
  "proposed_ops": [
    {"step":0,"op":"rubicon.revision","param_refs":["effect","counterfactual"],"method_ref":null,"bag_ref":"fixture:rubicon-03"}
  ],
  "prediction": {"phase":"Plan","abstain":false},
  "uncertainties": [],
  "capability_gaps": []
}
```
If A/B do not receive a style, `thinking` must be null rather than inferred as a native config. Unknown `method_ref` is null until trusted resolver fills it. Schema validator rejects unsupported operation names, dangling evidence IDs, unexpected effects, malformed DAG and invented runtime identities. Do not require long prose rationale.

## 6. Method mapping stages: retain every intermediate outcome

| Stage | Checked condition | Example refusal |
| --- | --- | --- |
| S0 | response schema and bounded explicit plan valid | invalid ref or cyclic plan |
| S1 | symbolic operation linked to codegraph/Source Contract SHA | invented method or stale source |
| S2 | expressible by admitted MethodRef/F###/relalg/R2IL law | unsupported gather or different NULL semantics |
| S3 | binding to pinned bag version, basis, owner, lens and operand position succeeds | missing resident mask / stale lease |
| S4 | existing native executor actually runs | planner-only stub, disabled opcode |
| S5 | independent oracle + negative control + replay checks pass | equal-arity word swap changes semantics |

**No upward promotion without proof.** A proposal that has the correct natural-language answer but only reaches S1 is not an executable success. For S2 compositions, require same typed semantics and evidence, not AST similarity. Explicit `UNKNOWN`, `REFUSE` and `MISSING_ORACLE` are correct outcomes where applicable. Record `method_word`, `basis_digest`, `law_digest`, `codebook_generation` and owner/lease lifetime only where authoritative types exist.

## 7. Independent oracles and negative controls

- **Rubicon:** independently table-driven transition expectations against `KanbanColumn::next_phases`, `advance_on_revision`; `IncreaseEligible + NotRun` must NOT commit; `NoIncrease` returns to Plan when applicable; `Suspend` holds. Disable test mutating route must turn red.
- **ThoughtGit:** use from-scratch `thought-git-probe::oracle` for givens+ordered thought writes. Exercise 63→64 live pressure, chained squash, child references, stale generation and FIFO vs LeastUsed as separate stress slice; disable should fail.
- **F###:** mutate equal-arity method words while names stay constant; positional semantic witness or refusal must catch change. Check exact method and opcode mapping under pinned source revision.
- **relalg/Quack/mask-RISC:** compare only admitted operations against independent scalar/reference implementations, including bag counts and NULL/refusal cases. Refuse cross-table gather not authorized by mask-RISC law. No fake zero-copy claims.
- **Replay:** tear down process/owner; re-open pinned Lance version and verified codebook/method law, reconstruct parent causal DAG; mismatched source generation or order must refuse. If complete replay is not implemented, report `REPLAY_GAP` rather than simulating.
- **RolloutDB:** append acknowledgement ≠ query visibility ≠ canonical Lance dataset version. Verify `add`, `flush`, query/trajectory results and persisted evidence checksum. Link experiment ID to durable record ID, not returned base version.

All oracles and expected outcomes must be finalized before held-out calls. Where an oracle is impossible, the case can test discovery/abstention but never S5.

## 8. ThoughtGit, Rubicon and flow

- Thought 0 = canonical Lance-backed givens/version; thoughts 1..63 = reusable speculative live overlays. Generation-bound slot handles protect live accesses; durable history needs a separate event/thought identity. Squash merges speculative writes, not a commit to main.
- `lance-graph-flow` is currently a **proposal** per D-LFA. Use a read-only prototype or caller of existing declared control verbs; no second executor, scheduler or commit owner. Pause/resume handles only IDs and version/generation; on mismatch refuse or explicit rebind and re-evaluate.
- Guarded flow verification may use GKAT as proposed in D-LFA: K3 WHERE projection `{T}` for admission; never CHECK-style `{T,U}` for commit permission.
- Heckhausen phases and storage commits are related by a declared mapping, NOT identical events. `RequestCommit` is intent. Existing supervisor and BatchWriter own publication; no automated production landing in this experiment.

## 9. RolloutDB schema and evidence

Phase 1 uses existing `RolloutRecord` plus cold-side structured schema versioned metadata and source links. Store four distinct artifacts: input/context manifest, model plan, mapping + oracle report, cost/latency usage. Record condition, model ID, prompt/retrieval/config hashes, source SHA, fixture ID, exact prior feedback IDs, tool result identity, train/test partition, gap classification, action refusals, attempt budget. Keep model prompts outside the resident execution path.

Later (separate PR) propose dedicated Arrow columns and direct batch adapter for `MethodRef, ParamRef, DRBagRef, basis, law, cycle, thought, phase, outcome`. This is NOT needed to validate initial pedagogical benefit. RolloutDB does not provide model planning, scoring or native Rubicon law.

## 10. Metrics, analysis and kill switches

Primary: S5 full verified trace rate per fixture (for cases with independent oracle), paired D−A. Secondary: S0..S4 rates, correct abstention, invalid invented identities, wrong-side effect rate, precise gap diagnosis reproducibility, retrieval evidence citation validity, tokens/$/latency per S5, native execution/cold-model time separately, replay after teardown, owner/basis refusal, style-family exploratory breakdown. Report fixture-vs-real counts, missing-oracle separately. Use per-fixture paired differences and bootstrap intervals, with raw counts; do not claim significance from a tiny sample.

Preliminary *decision criteria* (not observations): D−A ≥ +10 percentage points S5 without regression on critical refusals; 100% safety/identity negative-gate enforcement for tested fixtures; zero silent side-effect bypass; each reported new gap has a source witness and independent reproducer. If thresholds fail, mark hypothesis not supported, preserve all failures and costs.

Budget proposal if user actually has $200 available credits: $30 pilot/cost calibration, $120 held-out four-arm runs, $30 independent negative controls and selective reruns, $20 reserve. No unbounded tool loop; no implicit model upgrade; max per request and per fixture reserve; abort before cap, track API costs from actual provider usage. Subscription API balance/eligibility must be confirmed, not inferred.

## 11. Deliverable and first executable vertical slice

1. Freeze existing source SHAs and oracle fixtures and record status. Split docs/lab/real.
2. Stand up read-only capability registry and search over source-backed Codegraph + ContextStore; apply auth/scope and fixed retrieval.
3. Use Rig or direct Claude API in cold boundary with structured tools, reject unknown methods; run a single `IncreaseEligible, NotRun` Rubicon fixture.
4. Map output to existing `KanbanColumn::advance_on_revision` where applicable; evaluate independent oracle and negative toggle.
5. Record in RolloutDB, **flush** before test read-back, query again after restart, attach costs/source anchors.
6. Repeat that single fixture A/B/C/D with independently frozen feedback; then pilot and main only after gates.
7. Expose a read-only REST and optionally Railway MCP facade after auth and deterministic parity. No arbitrary SQL, shell, token store or writes.

**Stop if:** source/model identity cannot be pinned; RAG contains holdout answers; hidden-or-unverifiable outputs are called an oracle; no independent witness; semantic word swap undetected; invalid branch/owner used; model-controlled commit bypass; duplicate executor appears; WAL append conflated with canonical main; budget uncontrolled.

Related existing plans: `.claude/plans/d-lfa-lance-graph-flow-rubicon-assimilation-v1.md`, `d-rft-relalg-thought-git-frontier-v1.md`, `thought-git-api-v1.md`, `thought-git-basics-v1.md`, `d-dynbag-holographic-bags-v1.md`. They remain authoritative where this scratchpad is merely proposed.
