# LG-COG-001 — Pre-registered Claude Teacher × Lance-graph method-mapping experiment

**Status:** PROPOSAL / NOT RUN. Companion to [scratchpad mindmap](2026-11-14-cognitive-teacher-method-bag-mindmap-v1.md). Keep separate from implementation PRs. The experimental dependent variable is **independently verified method binding/execution**, not prettier Claude answers.

## Causal question and null

Teacher API tokens act as controlled external semantic probes. Does additional context (B), explicit versioned thinking config (C), and prior independently verified feedback (D) improve alignment of teacher-suggested operations with already admitted typed R2IL/Quack/relalg/mask-RISC methods over pinned DR bags? H0: no improvement in independently verified S5 method traces; lexical answers, speculation or feedback copying are not evidence.

## Arms and isolation

| Arm | Input | Only changed variable |
|---|---|---|
| A | question + identical output schema + operation type vocabulary | no RAG, no style assignment, no prior outcomes |
| B | A + bounded retrieval from ContextStore | source-bound context |
| C | B + pinned StyleFamily/ThinkingStyle + seven FieldModulation values | explicit thinking configuration |
| D | C + *disjoint* prior audited rollout feedback | experiment history |

Fix model ID, decoding, tool schema, output budget, repository SHA and temperature. Randomize arm order within each fixture. Fresh context per call. No live retrieval of answer-bearing test oracles. D uses only frozen, independently verified **development** trajectories not semantically duplicate to holdout fixtures. RAG and D both carry source/corpus hashes and exact retrieved record IDs; do not put unverified teacher claims in authority context.

## Population and sample

- 12 pilot fixtures for prompt/interface validation; 24 development fixtures for frozen feedback generation; 72 holdout fixtures across six domains: (1) F###/R2IL, (2) DT64/Quack, (3) Rubicon/Kanban, (4) ThoughtGit, (5) Codegraph/source-to-method, (6) Cycle/replay, 12 each. Stratify positive, negative and deliberately unknown cases.
- Four arms × 72 holdouts = **288 primary model calls** with one frozen assignment of ThinkingStyle per fixture, repeated across C/D.
- The 36 runbooks appear twice across 72 tests; each has exactly one canonical 12-family owner. Balance by domain before inspecting results, report family/style comparisons as exploratory (insufficient power).
- Optional independent style follow-up: 36 runbooks × 2 examples × 2 repeats = 144 calls only if budget remains.
- Pilot not counted in primary estimates. Any prompt or retriever change after preregistration forces a new protocol version.

## Example input and required explicit output

```json
{
  "fixture_id": "RUBICON-NEG-03",
  "world": {"dataset_id":"fixture-world","version":91},
  "phase":"Evaluation",
  "effect":"IncreaseEligible",
  "counterfactual":"NotRun",
  "allowed_effects":["READ","PROPOSE","REFUSE"]
}
```

External JSON from Claude is a cold transport shape, NOT internal execution ABI:

```json
{
  "claims":[{"text":"Counterfactual has not run","evidence_refs":["fixture:RUBICON-NEG-03"],"status":"GROUNDED"}],
  "method_plan":[{"op":"rubicon_transition","param_refs":["effect","counterfactual"],"method_ref":null,"bag_ref":null,"predicted":"Plan"}],
  "thinking":{"style_family":1,"thinking_style":2,"config_ref":"pinned-config"},
  "uncertainties":[],
  "abstain":false,
  "predicted_result":"Plan"
}
```

Allow null method references and abstention. Validate enum discriminants, evidence IDs, source scopes, dependency ordering, schema version, method version and authorized effects. Schema-valid ≠ semantically correct ≠ admitted/executeable. Never demand or claim access to private hidden LLM chain of thought: ask for explicit observations, hypotheses and method candidates.

## Exact proposed method/IR bridge: no parsing in executor

```text
external Claude tool JSON (COLD)
   → schema validation + capability discovery + Codegraph symbol resolution
   → compile/bind once, pinned by dataset/source/registry generations
   → typed invocation: crate::{op}::{param}::{DR<bag_ref>}
   → MethodRef × ParamRef × BorrowedVirtualBagRef
   → relalg normalize/lower where expressible
   → Quack bind / existing R2IL positional polyfill
   → admitted mask-RISC program / terminal kernel
   → OutcomeRef | TypedRefusal
```

This is an **intended typed ABI** and readable notation; not an existing string syntax, trait, parser or magic generic. Codegraph preserves source-to-method lineage and a semantic digest; method binding checks law/lens, operand shape, null/unknown mode, row-order/population, basis, lease generation, version and effects. `DR64³` is a candidate virtual bag substrate: prove actual routing separately. Do not accept gather/derived-mask-reuse/XLOG composition or spatial equivalence by declaration. Do not materialize rows as objects to drive `crate::{op}`.

## Stage-specific scorecard

| Stage | Success criterion | Independent negative |
|---|---|---|
| S0 parse | valid normalized explicit operation trace | missing/forged field refused |
| S1 resolve | source-backed Codegraph MethodRef | symbol absent / stale source |
| S2 express | declared law equivalent to current R2IL/relalg ops | same arity, changed semantic opcode |
| S3 bind | pinned, authorized BagRef + ParamRef + method | wrong basis, stale lease, null policy |
| S4 execute | admitted native execution (not mock or text simulation) | disable the actual kernel or route |
| S5 verify | independent reference oracle + replay result agree | counterexample, swapped order/semantics |

Report `EXACT / COMPOSE / ALIAS / DEFER / REFUSE / UNKNOWN / NEW_METHOD_CANDIDATE`. REFUSE/UNKNOWN are correct outcomes for preclassified invalid/unsupported fixtures, not generic failures. No claim that `S0–S3` imply `S4`. A Rubicon method unit test alone does not prove a F### × resident DR64³ binding.

## Independent oracle suite

- Rubicon: frozen legal-phase table and independently asserted `Evaluation + IncreaseEligible + NotRun = Plan`; ensure `Commit` only with completed acceptable counterfactual and other applicable gates.
- ThoughtGit: `thought-git-probe::oracle` from-scratch next-version computation, generation stale rejection after slot reuse, parent/child squash precedence, eviction with 63 live slots.
- R2IL/Quack/relalg: witnessed real method differential to an independent reference; for F000 prefer existing Ruby/SQL semantics evidence but qualify whether the fixture is actually real.
- Codegraph: frozen source/AST/symbol ground truth with two-author review; cite concrete file/line+commit, no self-verifying model verdict.
- Cycle: replay from fully torn-down state under same world/version/order/effect witnesses. Out-of-order or stale world must refuse.
- Disable controls must fail targeted gates; oracle must not use the candidate kernel. A missing oracle is `ORACLE_GAP` and never accepted as S5.

## Architecture gaps and feedback

A model is graded separately on interpretive correctness, evidence validity, discoverability, IR expressibility, ABI bindability, native execution and verifier agreement. Gap classes: `DISCOVERY_GAP`, `SOURCE_GAP`, `METHOD_IDENTITY_GAP`, `IR_EXPRESSIVITY_GAP`, `ABI_BINDING_GAP`, `EXECUTION_ADMISSION_GAP`, `REPLAY_GAP`, `ORACLE_GAP`, `PERFORMANCE_GAP`. Each gap carries failed gate, source links, minimal falsifier, and smallest candidate contract change. Reuse feedback through ContextStore only after oracle verification and with cross-fixture contamination controls.

## Durable rollout contract

For each call capture: run/fixture/arm/model and prompt hashes; retrieved context IDs and versions; codegraph/source SHA; style family + runbook + parameter fingerprint; explicit teacher output; proposed and resolved method/param/bag identities; S0–S5 verdicts; oracle ID/digest/negative tests; thought durable ID and transient (slot,generation); causal predecessor / cycle and world version; attempt cost/tokens/latency; next-gap decision. Persist append-only in RolloutDB. A successful MemWAL `add` is durable but not instantly visible or an advance in the base Lance version. Explicit `flush` before read-back, separate canonical seal receipt.

## Primary estimand, uncertainty, thresholds

- Primary binary outcome per held-out fixture: **correct, natively executed, independently verified S5 trace OR correctly refused by a fixture whose oracle demands refusal**, with the two categories also reported separately. Avoid inflating total accuracy by lots of trivial refusals. Primary analysis also explicitly reports genuine S5 execution rate on positive fixtures.
- Paired contrasts `B-A`, `C-B`, `D-C`; overall `D-A`. Report numerator/denominator, absolute percentage-point differences and paired bootstrap CIs, grouped by fixture; no unpowered "best 36 styles" conclusions.
- Hypothesis considered *provisionally supported* only if D exceeds A in valid S5 execution on eligible positives by preregistered ≥10 percentage points, with uncertainty favoring improvement and no safety-gate regression. A high prose score alone is null.
- Mandatory refusal rate 100% for critical invalid admission, stale lease, invalid method binding or prohibited Rubicon commit; any observed bypass triggers STOP and incident investigation. Performance claim requires measured native timings and allocations, not model estimates.

## Budget and failure policy

Maximum $200 in *verified* available API credits; allocate $30 pilot, $120 primary, $30 replication/negatives, $20 buffer; caps are not forecasts. 288 primary calls → $0.417/call budget mean; hard per-call maxima derived from observed pilot tokens and current model prices, no unlimited retries. Count prompt, output, tool and cache usage, provider charges and operational failures. Abort when available balance/run budget cannot fund a call's maximum. Never claim credits auto-bill from a Claude Code subscription; confirm Anthropic Console API balance and expiry. No external customer secrets in prompts; redact and scope all evidence.

## Integration stages and stop rules

1. P0 documents and preregistration only (this PR).
2. P1 repository source census + pin commits + concrete `MethodRef/ParamRef/BagRef` inventory; mark `SHIPPED/LAB/PROPOSED/ABSENT/BLOCKED/UNKNOWN`.
3. P2 Claude/Rig external adapter + ContextStore RAG + fixed schema and source provenance. Use existing `examples/mcp-claude-code` as a reference, not an implementation claim.
4. P3 `thought-git-probe` only, deterministic oracle and branch/squash stress; no actual canonical write.
5. P4 RolloutStore record → explicit flush → readback → feedback, under a synthetic pilot.
6. P5 real pinned F### × DR bag via relalg/Quack/R2IL/mask-RISC, after verified admission and independent differential; all unsupported routes refuse.
7. P6 thin REST tool registry + optional Railway MCP, scoped auth (ZITADEL subject/tenant), read-only initially, no arbitrary SQL/shell/write.
8. P7 randomized evaluation, publish full denominators/costs/disables/gaps and decide follow-up.

**STOP:** fabricated source/method identity; teacher reward authorizes Rubicon commit; stale slot accepted; arbitrary `Gather`/cross-lane reads; unverified computed-mask reuse; repeated text dispatch inside hot path; second evaluator/commit owner; WAL append mistaken for canonical land; feedback leakage; positive result without independent oracle.

No code implementation is part of this scratchpad PR.
