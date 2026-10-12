# Source excerpts: Shannon, PLN, pattern surprise, typed metagraph folding and safe program rewrites

**Date:** 2026-10-12. **Status:** first-hand paper distillation and counterexamples, not experimental results on SAG. **Companion:** [D-SAG-2](../.claude/plans/sag-reflexive-cognition-from-1509-v1.md), prior #1497/#1498 paper distillations.

## Houlsby et al., 2011: Bayesian Active Learning by Disagreement

**Primary:** https://arxiv.org/pdf/1112.5745 , Eq. (2), printed p3.
**Paper's equation:** I(theta; y | x,D) = H(y | x,D) - E_theta[H(y | x,theta)].
**Extract:** predictive entropy combines unresolved model uncertainty with outcome noise. BALD subtracts conditional outcome entropy under the model posterior. The paper studies a Bayesian GP classifier; it does **not** establish a PLN, ONA or CE64 scheduler.

**Synthetic red test:** candidate A: every model predicts Bernoulli 0.5. Candidate B: half predict 0, half predict 1 deterministically. Both aggregate predictive entropy values are 1 bit. Under the stated equal-weight model ensemble, BALD-like model information is 0 vs 1 bit. Raw Shannon H(f) cannot tell them apart. Without a calibrated model-posterior meaning of theta and p(y), never call H(NARS frequency) epistemic information gain.

## Geisweiller and Goertzel, 2019: inferential surprisingness in hypergraphs

**Primary:** https://doi.org/10.1007/978-3-030-27005-6_6 ; https://agi-conf.org/2019/wp-content/uploads/2019/07/paper_27.pdf , §§3–5; implementation candidates https://github.com/opencog/miner and https://github.com/trueagi-io/hyperon-miner .
**Extract:** compare an empirically supported pattern with a model-based/independence estimate; second-order distributions and a Jensen–Shannon-based discrepancy avoid treating every raw frequency difference as equally significant. Shared variables violate naive products of marginal pattern frequencies. This is neither pure Shannon maximization nor a finished general-purpose EIG estimator.

**SAG transfer:** #1509's label-vs-typed-identity MDL counterexample is a high-value source fixture. Two fields called hours in distinct concept/seal spaces are not one repeated method. A miner that extracts a false reusable pattern from labels fails before scoring.

## Goertzel, 2020: Folding and Unfolding on Metagraphs

**Primary:** https://arxiv.org/pdf/2012.01759 , §2 printed p5, §5.1 p13, §6 history-aware morphisms.
**Extract:** typed metagraphs permit an edge target connected to a whole edge (index zero) or to a specific typed port; those cannot be silently conflated in an SPO projection. Catamorphism must respect the empty, edge, Beside and typed Connect constructors, not merely return a matching scalar. A shortest-path length and an ordered shortest-path edge list are distinct outputs. Histomorphism/futumorphism needs intermediate history available *during* the computation.

**SAG transfer:** a typed Quack count is not a metagraph recursion-scheme proof. A legal partial lowering may count a bounded population. A whole-edge/port ambiguity, inaccessible intermediate history or inability to construct output expressions is an explicit typed refusal.

## Goertzel, 2021: Patterns of Cognition

**Primary:** https://arxiv.org/pdf/2102.10581 , printed p4 and §5.2.
**Extract:** formal equivalence between recursive decision processes and a metagraph fold/unfold assumes that the graph does not change during the fold. Mutual associativity of the relevant combination operations is also a stated requirement for parts of the approach. The paper does not prove live self-modifying metagraphs have those properties.

**SAG transfer:** stable MethodKey does not imply stable read snapshot. Pin a source revision and graph version k, append k+1 and replay. Alter evidence overlap and u8 rounding and compare both bracketings. If results differ, a blanket parallel-fold equivalence claim fails, though an explicitly order-sensitive computation can remain valid.

## Jøsang, 2007: evidence fusion and fission

**Primary:** https://arxiv.org/pdf/0712.1182 , §3 and fission sections.
**Extract:** independent observations can support cumulative fusion, while observations derived from common experience demand a dependence-aware rule. Fission is an inverse operation under assumptions on the underlying belief model. Subjective Logic and NARS are different truth formalisms.

**SAG transfer:** removing a contributor and recalculating a conclusion is a stronger epistemic ablation than merely deleting its graph node. Replaying or copying one receipt does not add evidence. If compressed confidence omits dependence and weights, refuse exact fission.

## Willsey et al., 2021: egg equality saturation

**Primary:** https://arxiv.org/abs/2004.03082 ; DOI https://doi.org/10.1145/3434304 .
**Extract:** e-graphs compactly encode many equivalent expressions and equality saturation explores rewrites before extracting a cost-ranked candidate, under explicitly chosen language, semantics and rewrite rules. It does not automatically prove side-effect, NULL, bag/set or source-claim equivalence.

**SAG transfer:** egg can be an external reference for *tiny, pure* H1-like rewrites; SAG is not an e-graph. The current H1 has a measured negative performance gap; neither a cost model nor source-syntax factoring proves the fused SIMD terminal exists. Correctness and authority must be hard constraints, not penalty scores.

## High-signal separation

- Entropy of defined outcome distribution != uncertainty of a source claim != surprisingness of model mismatch != MDL redundancy != cost of an executed query.
- Named attention/appraisal policies are independent of all those scores.
- A proposed code optimization is only a hypothesis until semantic proof, measurements on representative hardware/workloads and human authorization.
- The appropriate null result is "no advantage from this policy or lowering under fair budget", not an AGI headline.
