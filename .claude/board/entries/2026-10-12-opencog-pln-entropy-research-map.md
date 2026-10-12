# 2026-10-12 — OpenCog PLN x Shannon attention: questions and sources

**Status:** RESEARCH MAP / OPEN, documentation-only; no implementation, no measurement, no stored semantics or decision changed.
**Map:** [`.arxiv/2026-10_opencog_pln_shannon_attention_research_map.md`](../../../.arxiv/2026-10_opencog_pln_shannon_attention_research_map.md)
**Anchors:** [CE64 plateau](../../knowledge/ce64-reasoning-substrate-plateau.md), [D-HL](../../plans/ce64-hypothesis-layers-v1.md), [D-NALP](../../plans/nal-polyfill-v1.md).

## Finding

The upstream OpenCog ecosystem gives **distinct** research comparisons, not one interchangeable "OpenCog" oracle:

- [trueagi-io/PLN](https://github.com/trueagi-io/PLN) supplies executable MeTTa PLN rules, an evidence-ID scheme and a documented confidence-priority/horizon queue (snapshot `4405956`). The [revision implementation](https://github.com/trueagi-io/PLN/blob/main/lib_pln.metta) computes a weighted frequency and then imposes confidence floors/clamps. Thus a shared weighted-sum primitive does not establish NARS truth-law parity.
- [opencog/attention](https://github.com/opencog/attention) implements classic ECAN. Its existence **does not** demonstrate Shannon-entropy or expected-information-gain scheduling.
- [opencog/atomspace](https://github.com/opencog/atomspace) exposes hypergraph Atoms, mutable Values and Frames/ChangeSets. Similarity to Alpha layers and `SparseDelta` is a working hypothesis, not code parity.
- [trueagi-io/pln-experimental](https://github.com/trueagi-io/pln-experimental) and [trueagi-io/hyperon-experimental](https://github.com/trueagi-io/hyperon-experimental) supply mathematical and execution context, not a replacement for pinned PLN tests.

## OPEN questions / next proof

The map has source-referenced questions **OC-Q01..OC-Q13**: define the entropy random variable and calibrated `p`; define a leakage-free expected-information-gain estimator; test PLN-vs-NARS truth-law parity; identify evidence overlap; identify exact PLN and ECAN attention equations; scope AtomSpace graph/Frame projection; enforce equal budgets; include noise and duplicates; isolate sparse masking from scheduling; pin licenses, runtime, metacognitive controls.

**First falsifier:** on identical rules, evidence and fold budget, does naïve `H(f)` chase high-entropy zero-information distractors while a documented confidence queue or Stoanen/Moore policy discovers more *oracle-supported* target edges? Raw Shannon and *expected* information gain must be separate experimental arms. Corrected spelling in the research map: `Staunen`.

**Not measured:** no upstream oracle executed; no new runtime code; no implementation gate cleared. Preserve the byte-level ALU/ISA register contract and per-ClassView semantic readings.

## Closeout

```text
PR #<tbd> | STATUS: DRAFT | OUTCOME: sourced research map, OC-Q01..13,
controlled experiment design and kill conditions | OPEN: pin PeTTa,
inspect executable queue/ECAN internals, choose predictive entropy target
```
