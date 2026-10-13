# D-SAG-5 pinned sources, executable oracles and refusal matrix

**Date:** 2026-10-12. **Status:** repository locations and primary papers checked, but **NO NEW EXPERIMENT WAS EXECUTED**. Source URL or headline is not a green test. Read [.claude/knowledge/FIRST-HAND-SOURCE-LAW.md](../.claude/knowledge/FIRST-HAND-SOURCE-LAW.md) before a future implementation.

## Repositories at research snapshot

| Source | Commit SHA at inspection | Why here |
|---|---|---|
| [lance-graph #1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509) | 42fa2a347cda22eaaa327bba514f2029cc9daa82 (merge to main) | current MethodTable -> SoftwareGraph -> Quack one-hop and H1 negative performance |
| [ONA](https://github.com/opennars/OpenNARS-for-Applications) | dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a | Cycle, PriorityQueue, Memory, Stamp, Truth, Decision C source |
| [modern PLN](https://github.com/trueagi-io/PLN) | 4405956947c4b53c7ff01bd565aa3b114bc970a1 | lib_pln.metta, truth formulas and scheduler |
| [PeTTa](https://github.com/trueagi-io/PeTTa) | 1d19a1da1ad8ba6327d660abd65433a0e341c23f | MeTTa runner, separate from native Hyperon |
| [Hyperon](https://github.com/trueagi-io/hyperon-experimental) | 3f76dc460da6961f57f69f6c3e550c59c74ada83 | Rust atom kinds, matcher, non-determinism, dependent types and spaces |
| [classic AtomSpace](https://github.com/opencog/atomspace) | c8d633bf272b838c2132c21b7c8ddf169a18dd22 | historic typed hypergraphs/Frames, not Hyperon |
| [classic ECAN](https://github.com/opencog/attention) | 96107652e1cf72747e69fcd2a20ffb6edba562d2 | historical attention baseline, not Shannon-EIG implementation |
| [Hyperon Miner](https://github.com/trueagi-io/hyperon-miner) | e955479bca1603597529d175cc22ac1583c67e15 | frequent/subpattern and surprisingness fixtures |
| [AdaWorldAPI/ruff](https://github.com/AdaWorldAPI/ruff) | 9407c748812454bcadfff2d3c89cf942bf66f5de | Python lint/format diagnostics only; **not** a Rust code-equivalence oracle |

Root licenses need per-file review: modern PLN and Hyperon root licenses state MIT; classic AtomSpace and attention have AGPL text with OpenCog linking exceptions. Do not copy source merely because this ledger links it.

## Compact falsification matrix

| Topic | Oracle or independent baseline | Negative / red fixture | Gate |
|---|---|---|---|
| typed SAG identity | real prepared MethodTable and tests/awareness.rs | same label different ClassId/seal, wrong compiled tier | exact graph answers and verified origin |
| self source snapshot | git SHA + compiled Program + certificate | stale Debug/FNV digest/toolchain/reading | stale reported, no implicit current evidence |
| multi-hop | independent scalar CALLS closure | A->B->C from one-hop only | legal relation carrier or typed refusal |
| local model context | direct source with full answer, rg/file text | omitted negative evidence, stale SHA, unverifiable claim | exact source/answer, measured token+time, no zero-token claim |
| NARS stamps | pinned ONA or OpenNARS where rules applicable | identical evidence across two hypothesis layers | no confidence inflation; independent cases still count |
| significance | version-k immutable event | new k+1 relationship falsely rewrites fact at k | fact stable, reinterpretation traceable |
| Shannon/BALD | analytic 0 vs 1 bit toy with defined model | equally high raw H noisy and epistemic candidates | no false claim that H(f) is calibrated |
| policy/appraisal | FIFO and frozen no-meta, same budget | noise, deadline missed, trusted stale path | improved useful results and no truth-law drift |
| respectful contradiction | two claims and independent supporting tests | refute one claim -> global source trust penalty | claim-level revision independent of person relation |
| MeTTa | upstream b4_nondeterm.metta | four combinations collapse into a 2-row mask | exact multiplicity or typed refusal |
| metagraph fold | typed constructor scalar law, pinned k | port 0 != port n; k changes during fold | laws and snapshot parity or named gap |
| optimize | compiled H1 plus scalar row oracle | NULL/overflow/bag/set mismatch; nonexistent fused terminal | no automatic applied modification |
| decision | independently checked proof, human approval event | graph self-issues verified tier or code write | deny by default, rollback when separately authorized |

## Reviewer decisions (OPEN)

- Which source-bound token/context question should be the first minimal probe?
- May D-SAG-1-4 / D-CMH-G1-4 introduce a relation carrier in a future authorized PR, or does first-wave SAG explicitly refuse every two-hop question?
- Which source/event receipts can be durable under mailbox ownership and D-SAG-1-6?
- Which truth law, inference oracle and independent evidence base does each policy use? No silent PLN/NARS conversion.
- What model distribution does BALD use, and how is its prediction validated? If unknown, call it a heuristic, not EIG.
- Which (if any) of the four named functional policies is worth measuring; what time/compute budget and kill threshold are approved?
- How are human approval and code correctness proof represented as different facts?
- What token/cost reduction threshold is set *before* the first run?
- What licenses and exact runtime commits permit independent fixture execution?

**No empirical win has been asserted by this documentation PR.**
