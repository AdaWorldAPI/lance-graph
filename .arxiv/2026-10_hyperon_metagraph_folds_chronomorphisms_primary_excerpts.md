# Hyperon / MeTTa -> LG: metagraph algebra, mutable-world caveat and parity witnesses

**Date:** 2026-10-12. **Status:** READ / HIGH-SIGNAL SOURCE EXCERPTS, no interpreter executed and no production implementation. **Applies to:** [#1498](https://github.com/AdaWorldAPI/lance-graph/pull/1498); related attention experiments [#1497](https://github.com/AdaWorldAPI/lance-graph/pull/1497).
**Read with:** [Hyperon→LG parity map](2026-10_hyperon_metta_to_lance_graph_parity_map.md), D-FPW, D-SQLP, D-NALP, V3/hotplug and CE64 substrate plateau. Summaries below are paraphrases of primary sources, separated from unverified LG interpretations.

## Filter

**Keep only if:** the source defines a law/semantics or describes a demonstrable counterexample, supplies page+section, and yields a small probe with both can-fire and can-stay-silent cases. **Reject:** "hypergraph = SPO", "MeTTa = graph query", "metagraph fold = SIMD SUM", "all recursion schemes are associative", and claims that the 2023 architecture roadmap is a measured capability.

## H1. Goertzel (2020), _Folding and Unfolding on Metagraphs_

**Source:** [arXiv:2012.01759](https://arxiv.org/pdf/2012.01759#page=4) §§2–6, PDF printed pp. 4–16 (especially §3 constructors, §5.1 pp. 12–13 and §6 pp. 14–16).

**Exzerpt A, sinngemäß (pp. 4–5, §2):** A typed metagraph permits **typed target ports**, an edge connecting to another edge (or its target), and a connection with a **whole** edge as destination. Binary subject–predicate–object edges cannot express this without a declared reification/projection scheme. The source also distinguishes a target-slot index of 0 (whole edge) from a positive slot index.

**LG test:** build a typed two-edge construction with (i) a connection to the target of an edge and (ii) a connection to the whole edge. Preserve target-slot identity, types and child order after `source -> canonical NodeGuid/edge population -> source`. A mapping that collapses the two cases is lossy, even if query counts are equal.

**Exzerpt B, sinngemäß (pp. 8–12, §§3–5):** The algebra distinguishes **Beside** (parallel combination), **Connect** with a routing function on input/output ports, and **Swap** of target positions. It motivates a symmetric strict monoidal structure with composition laws. In §5.1 a catamorphism is not an arbitrary aggregation: it preserves how *each declared constructor* is interpreted (`empty`, `edge`, `Beside`, `Connect_P`) and the relevant compatibility and algebraic laws.

**LG test:** choose one finite **ordered typed** metagraph and an explicit interpretation algebra `B`. Evaluate `f(empty)`, `f(edge)`, `f(Beside(x,y))`, `f(Connect_P(x,y))`, and a well-typed Swap; compare scalar structural reference with compositional R2IL call sequences. Check associativity/identity and an intentionally mismatched port type. Refuse malformed `Connect_P`; do not silently treat it as a binary relationship.

**Exzerpt C, sinngemäß (§5.1.1 p. 13):** Three concrete candidates named are target count, shortest-path length, and shortest-path edge list. Counting may be approachable using existing folds; *shortest-path edge list* demands **ordered path witnesses**, not just a scalar minimum.

**Kill condition:** if a mask-risc implementation returns the correct shortest-path length but cannot return the path edges, it has only partial parity. A sum over edges does not establish catamorphism parity.

**Exzerpt D, sinngemäß (§6 pp. 14–16):** Histomorphism makes previously accumulated intermediate results available to the fold; futumorphism retains the unfolding history; chronomorphism combines a history-aware unfold with a history-aware fold. Their *history hypertree* is an algorithmic dependency object, not synonymous with an event log. The paper itself frames parts of the extension as prospective.

**LG test:** same final numerical fold but distinct intermediate derivation histories; a history-aware operation must distinguish them when the source law says it should. Compare a trace-only log (non-readable by computation) with a true history operand accessible to subsequent transformation. If no intermediate state is available in current `FoldDialect`, record an insufficiency rather than mint an opcode or infer it exists.

## H2. Goertzel (2021), _Patterns of Cognition: Cognitive Algorithms as Galois Connections Fulfilled by Chronomorphisms_

**Source:** [arXiv:2102.10581](https://arxiv.org/pdf/2102.10581#page=3), abstract and §1 printed pp. 1–4; §5.2 printed pp. 23–24.

**Exzerpt E, sinngemäß:** It suggests a common mathematical description of inference, evolutionary learning, pattern mining and attention allocation via *discrete decision systems* and Galois-connected search/optimization. The reduction to a chronomorphism in its theorem depends on **mutual associativity** of the combination operations, not a universal property of all cognitive algorithms (§5.2). For inferential programs, reversible transformations may be necessary to obtain some of the desired laws.

**LG test:** prove or falsify rebracketing on a chosen concrete inference-composition law, including CE64 truth rounding, saturated u8 codes and shared evidential bases. `f((a⊕b)⊕c)` vs `f(a⊕(b⊕c))` across permutations. A mismatch rules out naive parallel reduction for that law; wide exact-integer accumulation before one final rounding is a separate candidate, to be checked rather than assumed.

**The most important caveat, printed p. 4:** the paper explicitly limits the rigorous fold/unfold equivalence to **a metagraph that does not change during the folding**. It says the mathematical extension to graph-changing cognitive systems has not yet been carried out.

**LG test:** run the same fold (a) against a pinned `LanceVersion` / immutable snapshot, (b) while a sparse overlay appends new claims, (c) under restart + replay. Require (a) exact reproducibility; record whether (b) is snapshot-consistent, an interleaved transition semantics, or unspecified. Being able to replay is *not* the same thing as proving the static-graph theorem applies to a mutable graph. Alpha layer ID is not evidence source ID.

**Red:** perturb mutation order or mid-fold visibility; if the answer changes under a supposedly snapshot-stable read, the contract fails. **Silence twin:** an isolated uncommitted layer never changes a pinned base read.

## H3. Goertzel et al. (2023), _OpenCog Hyperon: A Framework for AGI at the Human Level and Beyond_

**Source:** [arXiv:2310.18318](https://arxiv.org/pdf/2310.18318#page=22), §3.1.3 pp. 22–23, §4.2 p. 55, §4.3.4 pp. 61–63, §4.6.1 p. 83.

**Exzerpt F, sinngemäß:** The authors motivate efficient generic metagraph fold/unfold *as an objective*, not as an established property of the then-current interpreter. In the reasoning chapter, inference control (which rule is chosen next and in which context) is identified as a principal bottleneck. Programmable/interceptable chaining was a desired mechanism, with various features described as in progress.

**LG test:** split the semantics of (1) selecting rule+premises, (2) executing the rule, (3) recording the receipt, and (4) scheduling the next step. Only (2) is naturally a lane-law candidate. A matching per-row truth transformation without identical (1),(3),(4) is **not** end-to-end Hyperon inference equivalence. Receipt replay plus learned strategy selection can later be compared to an untrained policy; that experiment belongs to #1497.

**Exzerpt G (§4.6.1, printed p. 83):** Pattern Miner is described as using information-theoretic surprisingness. That does not identify it with either Shannon label entropy or expected information gain. Follow primary [Geisweiller & Goertzel (2019)](https://agi-conf.org/2019/wp-content/uploads/2019/07/paper_27.pdf) for the distinction.

## H4. Executable Hyperon witnesses, not theoretical equivalences

**Verified paths in the public repo, snapshot** `trueagi-io/hyperon-experimental@3f76dc460da6961f57f69f6c3e550c59c74ada83`:

- [`hyperon-atom/src/lib.rs`](https://github.com/trueagi-io/hyperon-experimental/blob/main/hyperon-atom/src/lib.rs): `Symbol`, `Variable`, `Grounded`, `Expression` distinctions; custom grounded matching/execution precludes blanket numeric-lane lowering.
- [`b4_nondeterm.metta`](https://github.com/trueagi-io/hyperon-experimental/blob/main/python/tests/scripts/b4_nondeterm.metta): `(pair (bin) (bin))` enumerates 4 combinations of A/B, with ordering unspecified for nondeterministic results.
- [`d3_deptypes.metta`](https://github.com/trueagi-io/hyperon-experimental/blob/main/python/tests/scripts/d3_deptypes.metta): dependent-typed vectors; a check must preserve the source's surprising *accepted but invalid* negative-length result in an arithmetic type example, rather than assuming dependent typing implies all invariants hold.
- [`e1_kb_write.metta`](https://github.com/trueagi-io/hyperon-experimental/blob/main/python/tests/scripts/e1_kb_write.metta): `new-space` + `add-atom` + `match`, reference for write visibility.
- [`c1_grounded_basic.metta`](https://github.com/trueagi-io/hyperon-experimental/blob/main/python/tests/scripts/c1_grounded_basic.metta): distinction between unreduced expression and explicit `Error`.

**LG test:** a fixed set of source fixtures executed by a pinned runtime compared to a probe-local front end that resolves labels to addresses through the binder, preserves binding multiplicities, and emits **typed refusal** for unsupported forms. Source read ≠ oracle execution; no parity claim until that separate run.

## Decision-oriented matrix

| Test | Existing LG asset | Genuine gap or constraint | Stop condition |
|---|---|---|---|
| typed source atom → address roundtrip | OGAR labels, ClassView, NodeGuid | ordered child-ports / metagraph whole-edge target semantics | two distinct source atoms become indistinguishable |
| constructors and finite catamorphism | explicit edge-as-population, R2IL procedures | typed Connect routing, ordered path outputs | constructor law fails, even if counts match |
| history-aware morphism | versioned snapshots, sparse append, receipts | history accessible **within** intermediate operator, not merely after | two histories with identical final counts collapse incorrectly |
| recursive fold with live updates | Lance snapshot, Alpha overlays | source theorem assumes static metagraph | snapshot replay differs or reads uncommitted claims |
| MeTTa matcher and rewrite | Quack joins/masks | multiple substitutions and NEW term construction | wrong cardinality, error/unreduced semantics |
| inference control | NAL procedures and planner receipts | rule-selection policy and provenance | per-rule parity mistaken for complete scheduler parity |

**No automatic conclusion:** Hyperon follows a fold/unfold-oriented programming-language design; LG provides fast masked population operations. Bridging them requires explicit constructor semantics, bindings, controlled mutation and history operands. A valuable result may be a **small, rigorously pinned subset**, plus a list of non-expressible cases.

**Implementation not authorized by this note.** No new CE64 meaning, tenant, opcode, VM or parallel IR without the normal source/effect/insufficiency gates.
