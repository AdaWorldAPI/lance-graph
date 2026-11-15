# XLOG documentation and donor harvest: initial source-backed map (2026-11-15)

**Plan:** D-XLOG-KNOWLEDGE-0, gate D-XLOG-KNOWLEDGE-G0 (`xlog-knowledge-spine-v1.md` §4).

**Evidence grade:** RESEARCH INTAKE / PARTIAL. This is not a complete mirror, code audit or runnable benchmark. Public pages were read from https://xlog.md/ on 2026-11-15; repo descriptions were checked against AdaWorldAPI forks and merged PR records. Pin exact commits, inspect docs source and recrawl before issuing a completeness verdict. Site `/llms.txt` was advertised in page navigation but a direct public fetch returned 404 during this intake; find the authoritative docs tree, do not count pages by assumption.

## Confirmed documentation entrypoints and first operator crosswalk

| Live doc | Public claim / semantics to preserve | What to inspect in source and DT64 | Current verdict |
| --- | --- | --- | --- |
| [Architecture overview](https://xlog.md/architecture/overview) | One frontend, parallel RIR deterministic / PIR probabilistic / EIR epistemic; host control and CUDA-owned data planes | `xlog-logic`, `xlog-ir`, `xlog-runtime`, `xlog-prob`, `xlog-gpu`, `xlog-cuda`; keep three separately typed contracts; no imaginary common RIR for modal logic | Source-linked public design, not DT64 behavior |
| [XLOG internals](https://xlog.md/core-concepts/how-xlog-works) | Four user paradigms, three reasoning IRs plus CNF/CDCL service | Language grammar, normalized AST, IR fanout, Python caller API and cold artifacts | To pin code and releases |
| [Language reference](https://xlog.md/reference/language) | Typed facts, recursion, stratified negation, aggregation, probabilistic annotations, `know` / `possible` and mode pragma | Coverage row per predicate type/arity, metapredicate, list, guard, non-ground body and query head | Do not infer executable support from grammar |
| [Query optimizer](https://xlog.md/architecture/query-optimizer) | Lowerer greedy join ordering, generic predicate pushdown, shape-only selectivity on triangle/4-cycle, multiway promotion and dispatch gates | Inspect magic-set/polarity/source-span retention; `MultiWayJoin` promotion not equivalent to native dispatch; keep stats advisory | Requires operation-level mapping |
| [GPU execution](https://xlog.md/architecture/gpu-execution) | Host RIR interpreter chooses CUDA kernels; device-resident relations | The AdaWorldAPI polyfill keeps caller contract; no CUDA runtime transplantation; native mask counters and a cold/hot boundary | External baseline, not a CPU cube guarantee |
| [Factorized execution](https://xlog.md/architecture/factorized-execution) | WCOJ fused aggregate, Free Join, count-by-root, recursive delta, probabilistic fold; per-route dispatch counters and kill switches | Translate *algebra and admission*, not hash/trie data structures; separately certify bag multiplicity and differential deltas; forbid row expansion | Test shape by shape |
| [Factorized internals](https://xlog.md/architecture/factorized-execution-internals) | Free Join frontiers, dense bitvectors/sparse hash delta, cost veto; fallback parity | What maps to Cube relation view/Keep/Count; what needs explicit typed refusal (Gather/EXO-108) | Do not call a host enumeration 'factorized native' |
| [Epistemic internals](https://xlog.md/architecture/epistemic-internals) | EIR built directly from AST, not RIR. Default FAEEL founded least fixpoint, G91 greatest compatible exact-tuple fixpoint, WFS three-valued negatives; restricted GPU candidate workspace; typed unsupported shapes | Modal source spans, world/scope, mode identity, founded vs self-support, WFS unknown; EIR boundary and reducer evidence must survive | Semantics distinct from Mistaber Clingo ASP |
| [Probabilistic engines](https://xlog.md/probabilistic/engines) | Tuple provenance to CNF to decision-DNNF / weighted model count or seeded Monte Carlo | Correlating source roots, semiring and confidence semantics; Scallop comparison; independent PIR oracle | Never map to NARS f/c or Epi5 numerically by intuition |
| [Certification](https://xlog.md/architecture/certification) | GPU release claims require exact pinned artifacts, GPU host, command and observed telemetry; includes a pinned `mistaber-xlog` acceptance corpus `74f2895486737b4caa42229389d309994e7ad3ea` | Distinguish source acceptance corpus from current Mistaber implementation; no generic speed certification from docs | Acceptance recipe only, not reproduced here |
| [Benchmarking](https://xlog.md/guides/benchmarking) | Separate measured from target claims | Pin harness, source inputs, hardware, optimizer route, comparison confounders and matched accuracy/semantics | Needs claim audit |
| [Roadmap](https://xlog.md/architecture/roadmap) | Listed tagged capabilities include newer epistemic/provenance work | Compare release tags, changelog, source and currently deployed docs, not mutable README promises | Release identity not yet verified |

## Concrete documentation drift to investigate

- The deployed [homepage](https://xlog.md/) says **2.80× faster than Scallop** at matched accuracy alongside **2.74× cached-vs-uncached** XLOG training improvement.
- The [AdaWorldAPI/xlog README](https://github.com/AdaWorldAPI/xlog/blob/main/README.md) reports the 2.74× internal ablation but explicitly says its separate Scallop comparison supports accuracy parity and **does not establish an epoch-time speedup**, because host and CPU-quota confounding precluded matched timing.
- Therefore: publish a claim-by-claim `site_claim -> source_commit -> benchmark_fixture -> exact_comparator -> evidence/limitation -> outcome` receipt; mark the 2.80× cross-framework speed statement **CONFLICT / NOT ADMITTED** until independently reconciled. Do not copy it into another performance claim.
- Public architecture docs describe optimized CUDA implementation, while [lance-graph #1910](https://github.com/AdaWorldAPI/lance-graph/pull/1910) explicitly decides *assimilate RIR operators and polyfill the caller*, not replace every CUDA buffer/kernal path. This is an architectural translation, not an upstream performance reproduction.

## Local work already done: avoid duplicate tasks

| Existing proof | What it says | Missing |
| --- | --- | --- |
| [xlog #2](https://github.com/AdaWorldAPI/xlog/pull/2) | Excluded `xlog-cube` compiles XLOG ExecutionPlans over a resident lance-graph Cube and drives SCC recursion in limited binary shapes | Host-row adjacency and diff, n-ary, computed columns, full native-gating evidence |
| [xlog #8](https://github.com/AdaWorldAPI/xlog/pull/8) | Refused-relation propagation and derived-head seed facts fixed | Reproduce with pinned XLOG and lance-graph heads |
| [lance-graph #2091](https://github.com/AdaWorldAPI/lance-graph/pull/2091) | 230-rule corpus classified; independent naive AST comparator | Coverage versus completeness, W5 fixpoint/W7 F100, bag counts |
| [lance-graph #1873](https://github.com/AdaWorldAPI/lance-graph/pull/1873) | DR64 Kripke least fixpoint matches 11 Clingo goldens | Non-stratified negative cycles remain with Clingo |
| [lance-graph #1874](https://github.com/AdaWorldAPI/lance-graph/pull/1874) | World/origin/rule DR64³ contraction | Exact witness + arbitrary-width tenant/world semantics |
| [lance-graph #1880](https://github.com/AdaWorldAPI/lance-graph/pull/1880) | 141 makor addresses via deepnsm-v2 | Retraction/provenance root integration with decision consumers |
| [lance-graph #1921](https://github.com/AdaWorldAPI/lance-graph/pull/1921) | Graded Codegraph CALLS/MAY/IMPL over Cube | Real coverage remains constrained; source grade must survive all lowering |
| [lance-graph #1958](https://github.com/AdaWorldAPI/lance-graph/pull/1958) | MeTTa logical method schema and Soufflé/Crepe/XLOG cold planning already designed | No granted production method builder or new interpreter |
| [lance-graph #725](https://github.com/AdaWorldAPI/lance-graph/pull/725) | AriGraph semantic-hit-seeded episodic search pure method | Actual wired consumer and witness-authority boundary remain gated |

The coverage census gives measured evidence on EIR. Of the literals it observes, 208 never compile, and 202 of those stop at the RIR lowering boundary as "Unsupported epistemic construct". This is evidence that EIR must not be lowered as RIR. It does not show that EIR can be lowered some other way. Source: `crates/xlog-coverage-probe/census-656def9.tsv`, taken over the XLOG fork at `656def9`.

## Source citation and page discovery requirements for next harvest

Do not represent this first 12-page reading as the full xlog.md site. Recursively enumerate the deployed documentation nav and docs-source tree; distinguish deployment content from code-generated and reference artifacts, include redirects and changed pages. Keep a manifest with columns:

```text
page_url | page_title | navigation_path | site_read_utc | docs_source_path
docs_commit | source_claim_id | literal_claim_summary | code_owner_path:line
source_commit | first_release_tag | source_test_or_example | donor_semantics
xlog_ir_route | dt64_admission | oracle | status | contradiction | falsifier
```

When no doc source or exact line is verified, use `UNKNOWN` explicitly. Keep a machine-diffable raw claim ledger and a human-readable architecture crosswalk; do not add a runtime schema, new key or fact owner to implement the harvest. Review upstream licenses for code excerpts and corpora; do not vendor proprietary third-party text or licensed legal translations.

## Immediate research priorities

1. Reconcile site vs XLOG fork for **RIR/PIR/EIR operator availability** and the Scallop benchmark claim.
2. Review source semantics of EIR FAEEL/G91/WFS and the Mistaber Clingo oracle *independently*, recording non-equivalences.
3. Audit the `EpistemicWitness64` proposed name against existing EW64/EpisodicWitness64/CausalEdge64 Epi5; trace data ownership before bit allocation.
4. Identify exact eligible operator mappings and native refusals from coverage #2091 and current EXO-73/108/143/W5/W7 boundaries.
5. Propose **one** first behavioral cross-domain fixture with a source withdrawal, separate authority worlds, witnessed legal discretion and AriGraph episodic recall. Do not try to implement all consumers in the first code PR.
