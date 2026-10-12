# 2026-10-12 — Hyperon/MeTTa to lance-graph: sibling semantic-parity map

**Status:** SOURCE INVENTORY / WORKING-MODEL, documentation only. No code, ABI, new opcode, stored meaning, benchmark result or implementation authorization.
**Research map:** [`.arxiv/2026-10_hyperon_metta_to_lance_graph_parity_map.md`](../../../.arxiv/2026-10_hyperon_metta_to_lance_graph_parity_map.md)
**Sibling:** [PR #1497, OpenCog/PLN x Shannon attention](https://github.com/AdaWorldAPI/lance-graph/pull/1497), independently based on `main`.

## Finding

The OpenCog family exposes *three non-interchangeable translation targets*: classic AtomSpace metagraph/storage and Frames, executable Hyperon/MeTTa atom/matcher/space/rewrite semantics, and the abstract morphism zoo from [Goertzel's metagraph folding paper](https://arxiv.org/abs/2012.01759). Merely sharing the words "graph" and "fold" with lance-graph proves no execution parity.

- Modern [Hyperon Rust/MeTTa](https://github.com/trueagi-io/hyperon-experimental) has `Symbol`, `Variable`, `Grounded`, `Expression` atom kinds, bindings, spaces and executable fixtures for nondeterminism, dependent types, grounded functions and writes.
- Classic [AtomSpace](https://github.com/opencog/atomspace) is a separate hypergraph engine. Its Frames and Values are *candidates* for comparison with immutable graph identities + sparse overlays, not drop-in Alpha semantics.
- [Hyperon Miner](https://github.com/trueagi-io/hyperon-miner) offers source-backed workload examples for nested pattern matching and support counts; scoring/scheduler policy belongs to the #1497 research PR.

## Main falsifier

Run a pinned `b4_nondeterm.metta` example: repeated matching and `(pair (bin) (bin))` return four nondeterministic results, whereas a naive bitmask-only projection can collapse them to two surviving rows. If the proposed LG lowering loses multiplicity, substitutions, unreduced forms, dependent typing or newly constructed terms, it is not MeTTa equivalent regardless of speed.

A separate mathematical witness must select one *typed metagraph* catamorphism from arXiv:2012.01759 and test its laws; `GroupSum` parity alone is insufficient.

## OPEN

Research map HY-Q01..HY-Q15 covers identity, typed/nested hyperedges, matcher binding scope, nondeterminism, rewrite, dependent types, output construction, space mutation, metagraph folds, missing primitive proof, reflective behavior, runtime choice, performance parity, licensing and sharing a corpus with #1497.

**Next:** source-pin and execute one Hyperon reference fixture, construct a minimal typed structural roundtrip, then see whether Quack/R2IL/mask-risc can produce the same result set. No runtime change before that proof.

## Closeout

```text
PR #<tbd> | STATUS: DRAFT | OUTCOME: sibling Hyperon -> LG source map,
semantic translation matrix, HY-Q01..15 and parity/kill probes | OPEN:
source oracle and fold-law tests not run, no new primitive authorized
```
