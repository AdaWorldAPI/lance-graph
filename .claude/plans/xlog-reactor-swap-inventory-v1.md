# XLOG Reactor Swap inventory (v1)

**Status:** INVENTORY, docs only. **Date:** 2026-11-24.
**Pinned:** xlog `main` 097c351, lance-graph `main` 855d051. **Claim:** EXO-390 Track C.
**Cross-references:** EXO-377 (RIR lowering onto DR/DT/Cube/Quack without tuple sets), EXO-330 (multi-hop folded, never stored), EXO-73 (generic recursion driver), EXO-87 (Soufflé RAM × 64 thought slots), EXO-108 (distinct vs bag count), EXO-338 (Cube width / Codec9).

**Principle:** same interfaces above, different physics below. We do not assume NAL = probabilistic inference, thought-git = EIR, or NSL = DeepProbLog. Parity is decided per operation.
**Not in scope:** a CUDA port, a new evaluator, a prerequisite for the LG-COG-001 pilot. Nothing here authorizes execution (cf. `xlog-knowledge-spine-v1.md` §0, `d-xlog-native-v1.md` Amendment E).

**Citation base.** Lines were read in local checkouts, then checked against the pins. The xlog checkout (86461fc) is an ancestor of 097c351. Between them, `crates/` differs only in two `Cargo.toml` arrow version lines (xlog#10), so every source and test line cited here is the same at 097c351. The lance-graph branch differs from 855d051 only under `.claude/`. The orchestrator re-read a sample of citations at the pins: `node_dispatch.rs:127`, `types.rs:14-23`, `wfs.rs:84`, `exact_induce.py:212`, `solver.rs:401`.

**Gap types** (the only three used; UNVERIFIED where the search was not exhaustive. A missing API is not automatically missing mathematics):
- **missing mathematics:** the native substrate cannot express the operation.
- **missing grammar:** the operation exists, but no admitted method composition describes it.
- **missing wiring:** operation and semantics exist; the XLOG adapter does not consume them.

## 1. Summary table

Paths are under `xlog/crates/` (X) or `lance-graph/crates/` (L). "Candidate" means a reuse target, not an equivalent.

| Track | XLOG caller contract (X) | Lance-graph reuse candidate (L) | Gap | Independent oracle | Mismatch / typed refusal | Cold / hot | Admission dependencies |
|---|---|---|---|---|---|---|---|
| RIR / xlog-runtime | `pyxlog/src/lib.rs` (`Program`, `CompiledProgram`); `xlog-cli/src/main.rs:46` `Run`; `xlog-runtime/src/executor/node_dispatch.rs:98,127` | `lance-graph-xlog/src/cube/body.rs:533`; `cube.rs:1083,1133`; `xlog-cube/src/lib.rs:866` is the adapter's private join | wiring (body, closure); grammar/representation (n-ary correlated relations; set union and difference over relations); bag counts UNVERIFIED (see §2.1) | `xlog-coverage-probe` naive AST oracle + Datafrog | set vs bag; one tile; refuses `Union`, `Diff`, `MultiWayJoin`, `TensorMaskedJoin` | hot | EXO-377, EXO-108, EXO-73 |
| PIR / xlog-prob | `xlog-prob/src/lib.rs:17-18,32,33` (`PirGraph`, `Provenance`, `ExactDdnnfProgram`, `Mc*`); `xlog-cli` `Prob` | none found for weighted model counting. Boolean skeleton only: `cube/body.rs` | UNVERIFIED: no weighted model counting integration was found (search not exhaustive). That is missing integration, not a proof that every numerical primitive is absent | `kc/ddnnf.rs:1` CPU evaluator; `xgcf.rs:1` CPU evaluator | probability ≠ NARS f/c; no automatic mapping | cold compile, hot WMC | PIR gate separate from RIR |
| EIR + WFS / xlog-prob | `xlog-cli` `--epistemic-plan-json`; `xlog-ir/src/eir.rs`; `xlog-logic/src/epistemic.rs:663,2235`; `xlog-prob/src/wfs.rs:501` | Mistaber Kripke `holds` fixpoint (`xlog-knowledge-spine-v1.md` §1); `thought-git-probe` | K3 exists (Quack `Logic::Kleene`, `lance-graph-quack/src/set.rs:25`); WFS, FAEEL, G91 and world-view fixpoints are not found (UNVERIFIED as missing mathematics); grammar (modal literals) | `tests/test_epistemic_g91.rs`; Clingo goldens (spine §4 G4) | FAEEL ≠ G91 ≠ WFS ≠ stable models; no EIR = ASP claim | cold | spine G6, separate EIR gate |
| xlog-solve | `xlog-solve/src/lib.rs:47-68`; used by `xlog-prob/src/compilation/validation.rs:766` | none found (not exhaustive) | UNVERIFIED: no complete SAT procedure was found. That does not show every constraint operation lacks a native realisation | CPU CLS `solver.rs:401` is heuristic only, not an oracle for UNSAT | SAT/UNSAT completeness; `Unknown` | cold | none declared |
| xlog-induce | `xlog-induce/src/lib.rs`; `pyxlog/python/pyxlog/ilp/exact_induce.py:78` | `cube/body.rs:533`; `cube.rs:1083`; `dt64/dr/pair.rs:26,50` | grammar; wiring | `nary_reference.rs` (CPU); Python `backend="python"` (see §3) | topology templates fixed; set coverage | cold (search), warm (score) | EXO-377, P1 |
| xlog-neural | `xlog-neural/src/lib.rs:31-41`; `pyxlog/src/neural.rs:12` | `lance-graph-nsl/src/lib.rs` is neighbor selection, not DeepProbLog; `lance-graph-onnx` is vocabulary | mathematics (differentiable weights, depends on PIR); grammar | none located | AD through circuit; PyTorch runtime | cold | PIR row |
| xlog-stats | `xlog-stats/src/lib.rs`; `xlog-runtime/src/executor/wcoj_dispatch.rs` | `cube.rs:322` `CubeStats`; `cube.rs:1156` `add_edge` | wiring | exact-degree probe (`xlog-wiring-to-hot-path-v1.md` W4) | avg/max degree only (`stats.rs:181`) | cold read, hot update | W4 |

## 2. Per-track findings

### 2.1 RIR / xlog-runtime
- **XLOG.** All RIR execution is one dispatcher, `execute_node_uncached` (`xlog-runtime/src/executor/node_dispatch.rs:127`), with arms for `Unit`, `Scan`, `Filter`, `Project`, `Join`, `GroupBy`, `Union`, `Distinct`, `Diff`, `Fixpoint`, `TensorMaskedJoin`, `MultiWayJoin`/`ChainJoin` (lines 129-316). The result type is `CudaBuffer`. This executor is GPU/CUDA only (`xlog-runtime/Cargo.toml` depends on `cudarc`, `xlog-cuda`).
- **Caller surface.** `xlog-cli/src/main.rs:46` lists `Run`, `Prob`, `Explain`, `Extract`, `Manifest`, `Repl`, `Watch`. `pyxlog/src/lib.rs:1252` registers `Program`, `CompiledProgram`, `LogicProgram`, and others.
- **Adapter today.** `xlog-cube` is CPU. It uses `const TILE: u32 = 0` (`lib.rs:77`) and `Cube::new` (`lib.rs:307`). Rows are `BTreeSet<Vec<u32>>` (`lib.rs:194`, set semantics, bag counts lost). The join is private (`lib.rs:866`).
- **Lance-graph.** `Cube::body_answer` (`cube/body.rs:533`) returns a witness count plus one support mask per variable, no tuple built. `materialize_witnesses` (`body.rs:569`) is the single named row builder. `Cube::closure` (`cube.rs:1083`) is mask-native semi-naive `edge⁺`. `closure_wide` is at `cube.rs:1133`, `new_tiled` at `cube.rs:472`.
- **Gap.** Wiring: the adapter calls none of `body_answer`, `closure_wide`, `base_closure` (`cube.rs:1231`), `add_edge`/`stage`/`commit`/`refold` (`cube.rs:1156,1357,1497,1308`). Grammar: `Atom { rel, vars: [u8;2] }` (`body.rs:46`) is binary only and bodies are one tile (`body.rs:28`). Not missing mathematics: `SetExpr` already has `And`/`Or`/`AndNot` (`cube.rs:70-78`). What is missing for *relational* union and difference is the representation of the operand relations as composable views (`xlog-wiring-to-hot-path-v1.md`, "Limits of the cube"). That makes it a grammar/representation gap until shown otherwise. `body_answer` counts distinct full bindings, "Boolean per base relation" (`body.rs:22`). Bag multiplicity is a different quantity (EXO-108). Whether it needs new mathematics or only a different terminal is UNVERIFIED.
- **Census.** Last committed: 230 compiled rules, 102 oracle-agrees (58 native-tested), 79 refused, 49 refused-on-data, 0 disagreements (`xlog-coverage-probe/README.md:150-151`). Refresh at 097c351 (2026-11-24, `census-097c351.tsv`): byte-identical to 007bb19 apart from the revision label. Board entry D-XLOG-COV-3.

### 2.2 PIR / xlog-prob (provenance and exact/MC inference)
- **XLOG.** `PirNode` has `Const`, `Lit`, `NegLit`, `And`, `Or`, `Decision` (`pir.rs:55-85`). Provenance extraction (`provenance.rs:475,661`), CNF (`cnf.rs`), the d-DNNF CPU evaluator (`kc/ddnnf.rs:1`), the XGCF CPU evaluator (`xgcf.rs:1`) and WFS (`wfs.rs`) show 0 CUDA references. Exact inference (`exact.rs:1`, "GPU-native Decision-DNNF") and `mc/resident.rs` are GPU.
- **Lance-graph.** No weighted model counting or semiring evaluation was found (search not exhaustive). The Boolean skeleton of provenance maps to the same masks as RIR.
- **Gap.** Mathematics: no native weighted fold over a circuit. Do not map PIR onto NAL truth values without a proof; the spine says "no automatic numeric mapping" (§2).

### 2.3 EIR and WFS (xlog-prob, xlog-logic)
- **XLOG.** `EirProgram` (`xlog-ir/src/eir.rs:169`). Recursive epistemic classification and FAEEL/G91 handling in `xlog-logic/src/epistemic.rs:1202-1267`; planning at `:663`, `:2235`. WFS returns `TruthValue::{True, False, Undefined}` (`wfs.rs:84`, `evaluate_wfs_rules` at `:501`) on CPU. The GPU epistemic path is `xlog-prob/src/epistemic_production.rs:1`.
- **Lance-graph.** K3 already exists as an admitted logic: Quack `Logic::Kleene` (`lance-graph-quack/src/set.rs:25`), plus the contract `set_algebra`. That is not WFS: WFS needs an unfounded-set fixpoint, and FAEEL / G91 need world-view semantics. None of these was found (search not exhaustive). The Mistaber `holds` least fixpoint is cited by the spine (§1) and is stratified only.
- **Gap.** UNVERIFIED whether WFS and world-view fixpoints need new mathematics or only composition over K3 plus a recursion driver (EXO-73). Grammar: modal literals (`know` / `possible`).

### 2.4 xlog-solve
- **XLOG.** `Solver::new_cpu` (`solver.rs:401`) is a heuristic continuous local search, CPU. `GpuCdclSolver::new` (`gpu_cdcl.rs:333`) takes `Arc<CudaKernelProvider>`: complete SAT/UNSAT, GPU only. Callers: `xlog-prob/src/compilation/validation.rs:766,898`; `xlog-cli/src/bin/xlog-epistemic-evidence.rs:23`.
- **Gap.** Mathematics: a native CDCL is not an algebra word. Not a candidate for early assimilation. Tier 3 of Amendment E (CubeCL residue) at most.

### 2.5 xlog-induce
See §3. In short: GPU scoring kernel (`score.rs:1-5`), CPU n-ary reference (`nary_reference.rs:1-12`), four topologies (`types.rs:14-23`).

### 2.6 xlog-neural
- **XLOG.** CPU only: depends on `xlog-core` and optional `pyo3`. Registry, batch, bridge (`lib.rs:31-41`); `bridge.rs:1` turns softmax outputs into probabilistic constructs. GPU glue lives in `xlog-prob/src/neural_fast_path.rs:1`.
- **Lance-graph.** `lance-graph-nsl` reproduces NSL `pack_nbrs` neighbor selection (`lib.rs` header). It is not DeepProbLog. No counterpart for neural predicates.
- **Gap.** Mathematics: differentiation through circuits; gated on the PIR row.

### 2.7 xlog-stats
- **XLOG.** CPU crate (depends on `xlog-core` only), despite the "GPU-resident" doc. `PrefixDegreeStats` holds average and max (`stats.rs:181-188`). Only caller on a production path: `xlog-integration/src/bin/wcoj_phase_report.rs:347`, with literal values. Runtime consumers: `wcoj_dispatch.rs`, `wcoj_cost_model.rs`.
- **Lance-graph.** `CubeStats` (`cube.rs:322`) counts programs, rows materialized and relations added. It is not a degree statistic.
- **Gap.** Wiring (W4 in `xlog-wiring-to-hot-path-v1.md`: no producer on the insert path).

## 3. xlog-induce as a teacher-independent baseline

**What it is.** `xlog-induce/src/lib.rs:1-13` scores `(left, right)` pairs for four topologies in one batched GPU pass. Output is `ScoredCandidate` with `positives_covered` and `negatives_covered` (`types.rs:56-70`). Templates (`types.rs:14-23`): chain `H(X,Y):-L(X,Z),R(Z,Y)`; star `L(X,Y),R(X,Y)`; fanout `L(X,Z),R(X,Y)`; fanin `L(X,Y),R(Z,Y)`.

**Candidate oracles.**
1. `nary_reference.rs:1-12`: a plain host backtracking interpreter of coverage. It is the more auditable of the two, and it does not use the CUDA kernel. I did not check whether it covers the four binary topologies exactly (`canonical_binary_pattern`, `nary.rs:7-10`, suggests yes).
2. Python `exact_induce.py:78` with `backend="python"`. Its parity test is `xlog/python/tests/test_ilp_exact_induce.py:1-10`. **Independence must be checked, and as read it is partial:** the Python path calls `prog.set_rule_mask`, `prog.evaluate()` and `prog.batch_fact_membership_device` (`exact_induce.py:212-248`), i.e. XLOG's own compiled ILP program on CUDA (the test is `skip_unless_pyxlog_cuda`). It is independent of the native scoring kernel but shares XLOG's evaluator. The Python path also needs `XLOG_ALLOW_PYTHON_ILP_REFERENCE=1`, and `strict_per_topology=True` for parity (`exact_induce.py:97-112`).
3. A third, lance-graph-side oracle would be `xlog-coverage-probe` naive (`README.md`), which is built from parsed AST, not from compiled plans.

**Comparison design.** Given one pinned fact set and example sets, produce three rule rankings under identical verification: (a) xlog-induce, (b) Claude-proposed rules, (c) native-composition scores. Verify each rule by coverage counts against one oracle above, with the same example sets and tie rules. Report counts, not preferences. The teacher-comparison protocol is LG-COG-001 census §8 and §14.1 (`.claude/plans/lg-cog-001-reuse-census-v1.md`, PR #2167).

**Lowering hypothesis. PROPOSAL, untested.**

| Topology | Native composition |
|---|---|
| chain | relation composition (`delta ∘ edge`, cf. `cube.rs:1083`) |
| star | intersection of the two relations' pair sets (shared-coordinate mask AND) |
| fanout | forward pair-plane fold of L (`PairDir::Forward`) projected to X, AND with R |
| fanin | backward pair-plane fold of R (`PairDir::Backward`) projected to Y, AND with L |
| n-ary | `Cube::body_answer` (`body.rs:533`) |

Coverage `positives_covered` would be a popcount of the head-example mask ANDed with the composed support. Open questions: set vs bag (coverage here counts examples, not derivations); self-referential candidates (head allowed as body, `exact_induce.py:100-105`); one-tile limit.

## 4. Ranked follow-ups (consistent with EXO-377)

- **P0.** Refresh the RIR census at xlog 097c351. Oracle: naive AST oracle + Datafrog (`xlog-coverage-probe`).
- **P1.** Adapter consumes `body_answer` for n-ary bodies. Oracle: naive AST oracle; `materialize_witnesses` only at the edge.
- **P2.** Composable virtual n-ary bags (view handles, no tuple set). Oracle: bag-count differential against `GroupBy count` rules.
- **P3.** Tiled body execution and borrowed relation routing (lift `body.rs:28` one-tile scope; uses `new_tiled`). Oracle: single-tile result on fixtures that fit.
- **P4.** Monotone SCC over native frontiers, no D-RA-4 gather. Oracle: three-way C0 parity (`xlog-cube/tests/closure_parity.rs`, per W6).
- **P5.** DRed retraction with thought-git receipts. Falsifiers: IAM diamond, cyclic relation, multi-support. Oracle: from-scratch recompute (thought-git oracle, `thought-git-probe/src/oracle.rs`). Starting point: `Cube::add_edge` (`cube.rs:1156`), `Cube::retire_edge` (`cube.rs:1177`) and the delta log they fold into (described at `cube.rs:1125-1131`).
- **P6.** Compiler-donor reconciliation. Check XLOG `xlog-logic/src/magic_sets.rs` and `stratify.rs` before any Crepe extraction. Oracle: stratum and binding-pattern equality on the 230-rule corpus.

## 5. Not searched / unverified

- Source lines were checked as pin-equivalent (see Citation base); only a sample was re-read at the pins.
- lance-graph search for weighted model counting, semiring evaluation, SAT, WFS or three-valued evaluation was not exhaustive. "None found" is not absence.
- The Haiku maps were used only as indexes. `map-xprob-b`, `map-xinduce.part*` and `map-cube.md` were not read in full. Citations come from direct reads.
- `nary_reference.rs` coverage of the four binary topologies: not verified.
- Whether `xlog-cube` is built or tested in CPU-only CI: not checked.
- Callers of `xlog-induce` and `xlog-neural` inside `pyxlog` beyond the cited lines: not enumerated. `xlog-integration` callers not enumerated.
- `xlog-stats` runtime consumers beyond the three file names: not read.
- The lowering hypothesis in §3 is untested.

## 6. P1 admission check against the corpus (2026-11-24, EXO-393)

**Question.** How many of the 128 refused rules in `census-097c351.tsv` can `Cube::body_answer` admit within its existing contract? That contract gives: the witness count, one independent support mask per variable, binary atoms, one tile, at most 8 variables, and no repeated variable within an atom.

**Answer: 0.** Every refused conjunctive rule needs one of the following:

- **A correlated head of arity ≥ 2.** For example `tri(X,Y,Z)`, `cyc(W,X,Y,Z)` and `clique5(…)`.
- **A binary projection of an n-ary derived relation.** For example `e1(X,Y) :- tri(X,Z,Y)`, `r1(X,Y) :- tri(X,Y,Z)` and `path01(A,C) :- clique5(…)`. The support masks cannot answer these (§4.2 of the mandate: independent masks lose correlation).
- **Something outside the conjunctive fragment.** `fof` and `mutual` carry `X != Z` filters. `isolated`, `no_in` and `keep` use negation. `user_perm` reads a ternary relation. `sg_result`'s SCC also contains `sg_result(W,Z) :- sg_witness(W,X,Y,Z)`, which reads a 4-ary relation and so refuses the whole SCC. The census found no rule with a 0- or 1-column head over a multi-atom binary body without filters or negation.

So the measured result **falsifies "the remaining RIR gap is mostly wiring" for the conjunctive cluster**. Wiring `body_answer` alone would admit no corpus rule. P1 is therefore not implemented. A native route that the census cannot exercise would only add untested surface.

**The 128 refused rules by class.** This table maps the census reasons to classes. It is a coarse reading of reason strings, not a per-rule proof:

| Rules | Class | Refused / on-data | Gap reading |
|---:|---|---|---|
| 39 | n-ary or computed head column | 11 / 28 | grammar/representation (P2), plus arithmetic heads (`add_result`…) |
| 25 | n-ary relation read or produced | 4 / 21 | grammar/representation (P2) |
| 23 | cascade: reads a refused head | 23 / 0 | follows its producer |
| 16 | comparison filter beyond `col == const` | 16 / 0 | candidate wiring/grammar: Quack already has `Cmp`. UNVERIFIED |
| 8 | negation, nullary or join-shaped body | 8 / 0 | candidate wiring/grammar: mask `AndNot` exists. UNVERIFIED for stratified negation |
| 5 | nested projection / arithmetic | 5 / 0 | UNVERIFIED |
| 5 | behaviour lens (`0xFFFF :: 0xF100`) | 5 / 0 | refused by design (no word executes yet) |
| 4 | n-ary join key | 4 / 0 | P2 |
| 3 | aggregate (`GroupBy`) | 3 / 0 | bag/aggregate semantics (EXO-108) |

**What follows (not implemented here):**
- **P2 (virtual correlated relations)** is the main lever for at least 68 rules: 39 + 25 + 4, before counting cascades.
- **The smaller clusters** (comparison filters, 16; negation, 8) are the next candidates for "missing wiring". Each needs its own claim, its own oracle gate, and a proof that the existing `Cmp` / `AndNot` semantics match XLOG's. Stratified negation is the key point.
