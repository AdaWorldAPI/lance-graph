# AI architecture research digest — 2026-08-23

Status: snapshot of the 2026-08-23 research radar. This file preserves the **digest shape** that preceded the longer pressure map in `2026-08-23-research-pressure-and-forward-momentum.md`.

It is intentionally concise and decision-oriented: **READ FIRST → ARCHITECTURE PRESSURE → DON'T CHASE**.

## Today’s shortlist

| Source | Date / evidence | Tag | Verdict |
|---|---|---|---|
| Joris M. Mooij, *Causal Reasoning with Bipartite Graphical Causal Models* | arXiv 20 Aug 2026; UAI 2026 | **NOVEL GAP** | **READ NOW** |
| Schmidt et al., *Property-driven Causal Abstractions for Markov Decision Processes* | arXiv 29 Jul / 7 Aug 2026; FMCAD 2026 + artifact | **NOVEL GAP** | **READ NOW** |
| Kim & Jang, *Chance-constrained selection of sequential intervention strategies from counterfactual estimates* | arXiv 13 Aug 2026 + code | **NOVEL GAP** | **READ NOW** |
| Wassim Jaziri, *Onto-Explain* | online 7 Aug 2026; Expert Systems with Applications | **CONFIRMS** | **SKIM** |
| Ji et al., *RippleMem* | arXiv 13 Aug 2026 | **CONFIRMS** | **SKIM** |

## READ FIRST

### Mooij, Bipartite Graphical Causal Models

- Proceedings: https://proceedings.mlr.press/v337/mooij26a.html
- arXiv: https://arxiv.org/abs/2608.19831

The sharp result is not merely “causal graphs with extra nodes.” In cyclic/equilibrium systems, two perfect interventions can impose the **same target variable and same value** yet have different consequences because they replace **different mechanisms/equations**.

Architectural pressure for lance-graph:

```text
same target X
same value x
but mechanism f1 != f2
        ↓
intervention identity must remain distinct
```

This is the clearest candidate for a silent alias in the current dense causal substrate.

**First probe:** `PROBE-BGCM-MECHANISM-IDENTITY-1`.

Try to carry mechanism identity through the existing OGAR-loco / `FnIndex` execution receipt first. **Do not mint a CausalEdge64 bit unless that representation demonstrably aliases.**

## ARCHITECTURE PRESSURE

### 1. Intervention identity may require mechanism identity

BGCM pressures the counterfactual receipt boundary, not the 64-bit edge layout by default.

### 2. Reason-equivalence may be better than state similarity

Schmidt et al. suggest abstracting states by the **causal reasons** that make a property hold/fail rather than by surface similarity.

- arXiv: https://arxiv.org/abs/2607.26787
- artifact: https://zenodo.org/records/21825827

Pressure on #991/#993:

```text
same candidate count != same reasoning state
same causal reason-set may justify abstraction
```

Use Sudoku traces as the sealed-world falsifier before applying the idea to learned behavioral/qualia basins.

### 3. Local safety does not compose into trajectory safety

Kim & Jang show that sequential strategies can share similar local/expected costs while differing materially in whole-trajectory budget-overrun risk.

- arXiv: https://arxiv.org/abs/2608.13209
- code: https://github.com/mfriendly/counterfactual-chance-selection

Pressure:

```text
edge-safe + edge-safe + edge-safe
           !=
trajectory-safe
```

Whole-path claims require whole-path evidence. Keep this as a guard for Counterfactual / MUL / future self-programming promotion.

### 4. HOW, WHY and LOOKED-HERE stay different

Onto-Explain reinforces a separation already emerging in lance-graph:

```text
HOW         = execution provenance
WHY         = ontology / semantic justification
LOOKED-HERE = alpha attention / metacognitive cartography
```

- publisher: https://www.sciencedirect.com/science/article/abs/pii/S0957417426027843

Compose #992 execution receipts with HHTL/ontology justification. Do not turn alpha into proof.

### 5. Anchor retrieval can be only the first epistemic action

RippleMem’s useful pattern is anchor retrieval followed by bounded graph expansion to recover distributed evidence.

- arXiv: https://arxiv.org/abs/2608.13334

The correct response is a benchmark against existing AriGraph, not another memory subsystem.

## Forward momentum

Ordered by information value rather than novelty glamour:

1. `PROBE-BGCM-MECHANISM-IDENTITY-1`
2. `PROBE-INTERMEDIATE-UNKNOWN-SUDOKU-1`
3. `PROBE-REASON-EQUIVALENCE-ABSTRACTION-1`
4. `PROBE-WHOLE-TRAJECTORY-RISK-1`
5. `PROBE-HOW-WHY-CONSISTENCY-1`
6. `PROBE-EPISODIC-ASSOCIATIVE-COMPLETION-1`
7. only then continue #993 Sudoku → chess → crossword universality work

The common rule is:

> **Preserve distinctions that change reasoning. Make every new primitive earn itself by first failing an aliasing falsifier.**

## DON'T CHASE

### RippleMem as a new architecture

The anchor → bounded expansion idea is worth benchmarking, but the substrate already has AriGraph / episodic / retrieval composition. If existing machinery produces the same gain, stop.

### EvoGraph-Mem as an editable-memory subsystem

Yuxi Qian & Yuxiang Ren, *EvoGraph-Mem: Failure-Aware Editable Graph Memory for Long-Term Language Agents*, arXiv 3 Aug 2026:

- https://arxiv.org/abs/2608.11248

Its finding that append-only distilled memory can become stale/harmful is useful, but lance-graph already separates versioned history, Revision, provenance and epistemic state. Treat it as confirmation / a possible later benchmark, not as a reason to import another mutable insight graph.

### Lance 10 prereleases

Lance has moved through the 10.x prerelease line, including `v10.0.0-rc.3` and `v10.1.0-beta.*`:

- https://github.com/lance-format/lance/releases

There are interesting format/index/cache changes, but nothing in today’s research radar justifies reopening the deliberate Lance 9 / LanceDB 0.33 dependency boundary solely because a new major line exists.

Revisit only for a **specific measured capability or blocker**, not version gravity.

### Generic “agent memory” / RAG papers

Do not collect papers that merely restate retrieval, graph memory, editable summaries or reflective memory without producing a new falsifier against the current substrate.

## Bottom line

Today’s digest does **not** recommend five new subsystems.

It identifies a much smaller set of places where compression must preserve epistemically important distinctions:

```text
different mechanism under same intervention target/value
different causal reason under similar surface state
different whole-path risk under similar local steps
different execution provenance under similar explanation text
different distributed evidence behind the same first anchor
```

That is the useful forward pressure for the current architecture.