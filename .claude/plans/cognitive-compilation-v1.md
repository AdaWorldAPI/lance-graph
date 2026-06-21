# Cognitive Compilation for Lance-Graph — v1

> **Status:** PROPOSED (scaffold landed; logic deferred)
> **Branch:** `claude/cognitive-compilation-lance-graph-h8sgym`
> **Operator:** jan@exo.red
> **Date:** 2026-06-21

---

## SCOPE CORRECTION (operator, 2026-06-21) — read first

The operator narrowed the scope after the first draft. The **only new idea is
the Elixir-shaped template.** Everything else in the loop already exists and is
**not touched**:

- **ractor** — already implemented (control-plane fence). Do not touch.
- **surrealdb** — already integrated as `kv-lance` only. Do not touch the repo;
  it is wired *as a backend for rig* (point `rig-surrealdb` at the AdaWorldAPI
  fork — Cargo wiring only).
- **Rubicon / kanbanview** — already exists; no new orchestration crate.
- **thinking styles, JITson, i4-32D thinking-style vectors** — already exist;
  reused, not recreated.

So this PR lands, additively:
1. **lance-graph** — the Elixir-template stack: `elixir-template` (NEW: the gap),
   `template-runtime`, `template-equivalence`, `cognitive-compiler`. Four
   standalone zero-dep excluded crates.
2. **rig** — `rig-surrealdb` Cargo-wired at the AdaWorldAPI kv-lance fork.
3. **rs-graph-llm** — one isolated, cherry-pickable graph-flow Task that calls
   the template runtime. (Operator will reset rs-graph-llm from upstream and
   cherry-pick this back; a local copy of the repo is taken as a safety net.)

The `lance-template-index` / `review-gates` / `github-promoter` crates described
below are **DEFERRED** (future work, not this PR) — the basin index and review
machinery already have homes in the existing planner / agent ensemble. They are
kept in this doc as the loop's target shape, not as deliverables of this commit.

---

## 0. The one-sentence design

> Use LLMs to **discover** cognition, traces to **prove** cognition,
> templates to **compile** cognition, and Lance-Graph to **run** cognition
> without asking the LLM again.

The LLM does not remain in the runtime path once a task pattern has been
learned. It is **teacher / compiler / critic**, never the reflex. The reflex
is a deterministic, replayable, OGAR-validated template executed against
Lance-Graph's basin index.

```
LLM        = teacher / compiler / critic   (Rig — learning + escalation only)
Lance-Graph= runtime / reflex memory       (basin match, template execution)
OGAR       = semantic type system          (meaning, classes, constraints)
SurrealDB  = provenance ledger             (traces, claims, evidence, approvals)
template   = compiled thinking macro       (Elixir-shaped, deterministic)
graph-flow = orchestration (optional)      (Rubicon phase choreography)
ractor     = control-plane ownership fence (shard / write authority)
```

---

## 1. Responsibility matrix (§17 of the source plan, repo-mapped)

| Layer | Owns | Repo / crate (home) |
|---|---|---|
| **OGAR** | meaning, types, constraints | `OGAR/crates/ogar-cognitive` (new) + `ogar-ontology` |
| **SurrealDB** | facts, traces, provenance, approvals | `surrealdb/doc/cognitive-compilation/ledger.surql` (new) |
| **Lance-Graph** | embeddings, basins, thinking styles, template match | `lance-graph/crates/lance-template-index` (new) |
| **Compiler** | trace → template synthesis | `lance-graph/crates/cognitive-compiler` (new) |
| **Runtime** | deterministic template execution | `lance-graph/crates/template-runtime` (new) |
| **Equivalence** | replay comparison | `lance-graph/crates/template-equivalence` (new) |
| **Review** | specialist + brutal reviewers | `lance-graph/crates/review-gates` (new) |
| **Promotion** | branch / test / commit / PR | `lance-graph/crates/github-promoter` (new) |
| **Rig** | LLM calls (learning / escalation ONLY) | `rig/crates/rig-cognitive` (new) |
| **graph-flow** | Rubicon workflow choreography | `rs-graph-llm/crates/rubicon` (new) |
| **ractor** | ownership / control-plane fence | `ractor/docs/cognitive-compilation-ownership-fence.md` (new) |

The scaffold lands all eleven homes as **compiling, zero-/minimal-logic
surfaces** — trait definitions, the OGAR-shaped DTOs, and the §18 gate
enumerations. No template synthesis, replay, or LLM call is implemented;
every method returns an explicit `NotImplemented` until its probe lands.

---

## 2. The cognitive-compilation loop (§2)

```
New task arrives
  ↓
lance-template-index : match task signature against basins   (Deliberation)
  ↓
├─ known + compatible + low-risk → template-runtime executes  (no LLM)
│       ↓ OGAR validate → SurrealDB store → done
│
└─ unknown / low-confidence → rig-cognitive (LLM solves)      (Commitment→Execution)
        ↓ record ExecutionTrace → SurrealDB ledger
        ↓ cognitive-compiler synthesizes TemplateCandidate
        ↓ template-equivalence replays candidate vs trace
        ↓ review-gates : 5 specialist + 3 brutal reviewers
        ↓ patch + re-replay + re-review  (repair loop)
        ↓ github-promoter : gates → branch → tests → PR → merge
        ↓ lance-template-index indexes the promoted template basin
```

---

## 3. Rubicon mapping (§3) — `rs-graph-llm/crates/rubicon`

| Phase | Question | Node |
|---|---|---|
| Deliberation | Seen this shape? Which style? | `DeliberateNode` (template match) |
| Commitment | Run template or escalate? | `CommitNode` (§12 decision rule) |
| Planning | Deterministic macro vs LLM exploration | `PlanNode` |
| Execution | Known path vs unknown path | `ExecuteTaskNode` / `RecordTraceNode` |
| Evaluation | Constraints met? Promotable? | `SynthesizeTemplateNode` → `ReplayTemplateNode` → `CompareEquivalenceNode` → `SpecialistReviewNode` → `BrutalReviewNode` → `PatchTemplateNode` → `CommitTemplateNode` |

The Rubicon crate provides these as node stubs. They are *choreography*;
they own no meaning (that is OGAR) and no facts (that is SurrealDB).

---

## 4. The Elixir-shaped template (§4)

A template is **declarative, replayable, versioned, OGAR-typed**. Each step
maps to an OGAR action; the runtime never interprets free text.

```
defmacro source_ranking_v1(input) do
  pipeline do
    step :extract_sources         # → ogar action ExtractSources
    step :normalize_claims        # → ogar action NormalizeClaims
    step :score_primary_proximity # → ogar action ScorePrimaryProximity
    step :score_independence      # → ogar action ScoreIndependence
    step :score_evidence_density  # → ogar action ScoreEvidenceDensity
    step :penalize_incentive_risk # → ogar action PenalizeIncentiveRisk
    step :emit_ranked_sources     # → ogar action EmitRankedSources
  end
end
```

OGAR already has `ogar-from-elixir` (Elixir → OGAR). The compiled template's
canonical form is an OGAR `Template` class whose `steps` are OGAR action
references — `ogar-from-elixir` is the eventual parse front-end, the
`template-runtime` crate is the back-end executor.

---

## 5. The §18 non-negotiable gates (enforced by `github-promoter`)

1. **No trace → no template** — synthesis requires a recorded `ExecutionTrace`.
2. **No replay → no promotion** — `template-equivalence` must run.
3. **No source span → no claim** — every `Claim` carries a `SourceSpan`.
4. **No equivalence → no merge** — equivalence class must clear threshold.
5. **No OGAR validation → no execution** — runtime validates against OGAR.
6. **No human approval → no high-risk OSINT promotion** — §14 guardrails.
7. **No LLM in hot path once a template passes** — the reflex is deterministic.

These are encoded as the `PromotionGate` enum + `gates()` ledger in
`github-promoter`. A promotion that skips any gate is a compile-visible
omission (the gate set is exhaustive-matched).

---

## 6. OSINT ethics guardrails (§14)

Templates may analyze **public** claims / officials / institutions / media
narratives. Templates must **not** target private individuals, family,
personal secrets, doxxing, harassment, or identity inference without a
public-interest justification. Every OSINT template carries:
`public_interest_reason`, `scope_boundary`, `source_provenance_requirement`,
`harm_minimization_check`. Modeled in `ogar-cognitive` as the
`OsintGuardrail` constraint and re-checked by the brutal `Adversarial` /
`Ethics` reviewers in `review-gates`.

---

## 7. First vertical slice (§15) — build this FIRST

**Source ranking for public-narrative claims.**

Input: 10–30 public articles around one claim →
1. LLM extracts claims (Rig)
2. LLM ranks sources (Rig)
3. Store trace (SurrealDB ledger)
4. Synthesize `source_ranking_v1` (cognitive-compiler)
5. Replay (template-equivalence)
6. Compare vs LLM output
7. Review (review-gates)
8. Patch
9. Merge (github-promoter)
10. Future source-ranking runs **skip the LLM**.

The scaffold provides the type surface for exactly this slice; the slice's
logic is the first probe (D-CC-RUNTIME-1, see STATUS_BOARD).

---

## 8. Deliverables (D-ids, see STATUS_BOARD)

- **D-CC-OGAR-1** — `ogar-cognitive` canonical classes + constraints (OGAR)
- **D-CC-COMPILER-1** — `cognitive-compiler` trace→template surface
- **D-CC-RUNTIME-1** — `template-runtime` executor surface + first slice
- **D-CC-EQUIV-1** — `template-equivalence` replay/equivalence surface
- **D-CC-INDEX-1** — `lance-template-index` basin match + §12 decision rule
- **D-CC-REVIEW-1** — `review-gates` 5 specialist + 3 brutal reviewer surface
- **D-CC-PROMOTE-1** — `github-promoter` §18 gates + PR automation
- **D-CC-LEDGER-1** — SurrealDB provenance schema (`.surql`)
- **D-CC-RUBICON-1** — `graph-flow-rubicon` orchestration nodes
- **D-CC-RIG-1** — `rig-cognitive` teacher/critic adapter
- **D-CC-FENCE-1** — ractor ownership-fence design doc

Status legend: this PR lands the **surface** for all eleven (Queued →
Scaffolded). Each becomes `In progress` when its probe / logic lands.

---

## 9. Scaffold conventions

- lance-graph crates follow the repo's standalone pattern: zero-dep,
  `exclude`d from the workspace, verified via
  `cargo test --manifest-path crates/<name>/Cargo.toml`. They define the
  trait surface + OGAR-shaped DTOs natively; doc-comments name the canonical
  type home (`lance-graph-contract` / `ogar-cognitive`) for the eventual
  dedup pass.
- Every compute method returns `Err(..::NotImplemented)` — the scaffold
  compiles and `cargo test` passes (tests assert construction + the
  NotImplemented contract), with zero fabricated results (truth-architect /
  overclaim discipline).
- Sibling-repo crates follow each repo's edition/license/workspace idiom.

---

## 10. What this is NOT (yet)

No LLM is called. No template is synthesized. No replay runs. No PR is
opened. This is the **type system + trait spine** for the loop — the
"obligatory shape" that the slice-first logic will fill. Per §18: no logic
ships without its trace, replay, equivalence, review, and gate.

---

## 11. The verifier is load-bearing — `template-equivalence` is the crux

The "take the training wheels off" decision (let a deterministic template
**replace** the LLM run) is exactly the output of `template-equivalence`. It is
not a footnote — it is the gate the whole loop's integrity rests on. Two
consequences (see `EPIPHANIES.md` E-EQUIVALENCE-IS-THE-CRUX):

1. **Fail closed.** A pass is an affirmative PROOF of reproduction, never "no
   difference detected". The checker compares claims as a set both ways (no adds,
   no drops), source spans exactly, and requires the ranked-item set be preserved
   for `RankOrder`; every unevaluable dimension (incl. the deferred Semantic
   class) returns `Failure`. If the comparison can pass when it shouldn't, the
   loop self-certifies on a lie.
2. **It rides on transparent versioning (surrealdb #50).** The loop records each
   orchestration step (run by rig + rs-graph-llm onto surrealdb-on-kv-lance) as a
   versioned Lance commit, and verifies by **AS-OF replay + compare**. This needs
   the corrected version→snapshot mapping from surrealdb #50; the old broken
   `checkout_version(versionstamp)` would replay the wrong step and compare
   garbage. #50 is the substrate the verifier records and replays on.
