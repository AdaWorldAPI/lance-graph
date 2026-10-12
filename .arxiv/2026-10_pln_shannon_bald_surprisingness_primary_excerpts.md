# PLN, Shannon, BALD and surprise: primary-source excerpts with falsifiers

**Date:** 2026-10-12. **Status:** READ / secondary distillation of primary papers, **not** implemented, reproduced, or benchmarked. **Applies to:** [#1497](https://github.com/AdaWorldAPI/lance-graph/pull/1497), sibling [#1498](https://github.com/AdaWorldAPI/lance-graph/pull/1498).
**Read with:** [OpenCog/PLN/entropy research map](2026-10_opencog_pln_shannon_attention_research_map.md), D-HL-3/4/8, D-RPF-P5, D-CE64-STAUNEN-0. These excerpts distinguish the paper's stated result from a suggested LG probe. Page numbers refer to the PDF's printed pages, not necessarily the browser's zero-based PDF page index. Paraphrases, not copied source text.

## High-signal filter

1. **KEEP:** papers that distinguish *predictive entropy*, *epistemic information gain*, *difference between independent inference paths*, and *evidence uncertainty* with equations and test cases.
2. **KEEP:** independent source formula or executable upstream fixture, a measurable failure mode, and a consequence for source selection.
3. **DROP:** "entropy = curiosity", "unexpected = useful", "PLN F/C = NARS F/C", "Shannon = ECAN", claims based only on architectural resemblance.
4. **SEPARATE:** entropy of a binary answer, entropy of a *distribution of outcomes*, second-order uncertainty about probability, and entropy of equiprobable pattern instances (`log2 Count` under a special model) are different random variables.

## P1. Houlsby, Huszár, Ghahramani, Lengyel (2011), _Bayesian Active Learning for Classification and Preference Learning_

**Source:** [arXiv:1112.5745](https://arxiv.org/pdf/1112.5745#page=3), §2, Eq. (2), printed pp. 2–3; experiments §5 and discussion pp. 11–13. **Evidence:** derivation and experimental comparisons in a *Gaussian-process classifier*, not a general AGI model.

**Exzerpt, sinngemäß:** Choosing a query because the predicted label is uncertain mixes irreducible outcome noise with what the model could learn. Under their Bayesian assumptions, the information gained about latent model parameters is:

```text
I(theta; y | x, D) = H[y | x, D] - E_{theta | D} H[y | x, theta].
```

The second term removes uncertainty that remains even with the parameter setting known. Their examples show noisy samples can attract less principled acquisition methods; information gain prioritizes epistemic disagreement instead.

**LG consequence / question:** What is `theta`? Not automatically "the 64 hypothesis layer IDs": a layer is an *alternative world / working hypothesis*, not a probabilistic model with posterior weight. To call an arm **BALD**, first specify a weighted ensemble/posterior, `p(y | theta,x,D)`, calibration, predictive noise, and update rule. If these are absent, use the explicit name **committee-disagreement heuristic**, not Bayesian information gain.

**Red / green pair:** two candidates have predictive `p(y=1)=0.5`. Candidate A: every model separately predicts `p=0.5` (pure aleatoric ambiguity). Candidate B: half the models predict `0`, half `1` (epistemic disagreement). Same raw `H(y)=1 bit`, but conditional terms are 1 vs 0, yielding 0 vs 1 bit of mutual information for deterministic, equally weighted committee models. A raw Shannon scheduler cannot discriminate; a correctly defined MI score must. Add uncalibrated NARS `f` arm as a deliberately naive negative control, never the default truth semantics.

**Do not claim:** BALD is implemented by OpenCog, or that its GP empirical speedups transfer directly to graph inference.

## P2. Geisweiller and Goertzel (2019), _An Inferential Approach to Mining Surprising Patterns in Hypergraphs_

**Source:** [AGI-2019 proceedings paper](https://agi-conf.org/2019/wp-content/uploads/2019/07/paper_27.pdf#page=7), §§3–5, printed pp. 5–9; [OpenCog Miner](https://github.com/opencog/miner); [Hyperon Miner](https://github.com/trueagi-io/hyperon-miner). **Evidence:** proposed semantic formalism and an implementation described in the paper; §6 explicitly says full experimental results are omitted.

**Exzerpt, sinngemäß:** The surprisingness of a *pattern* is conceived as a disagreement between different inferential estimates concerning the same conjecture, particularly an empirical estimate from observations versus an estimate from independence assumptions. For second-order belief distributions the paper suggests **Jensen–Shannon distance** so small-sample uncertainty tempers apparent discrepancies. Their I-surprisingness example corrects naive independent-clause multiplication when the clauses share variables, by making the equality constraint explicit.

**Paper's suggested shape:**

```text
surp(P,D) = distance(empirical_distribution(P,D),
                     model_estimate_distribution(P,D)).
```

In §5 it represents a chain of inference over `emp`, `est`, and a JSD-like comparison, not simply the Shannon entropy of a truth byte. The paper leaves parts of the general estimator open and acknowledges only the independence-based estimator implemented then. Distinguish **Jensen–Shannon divergence** from its square-root metric **distance**; specify which is used.

**LG consequence:** Our evidence-driven NARS arm and model-assumption/PLN arm can furnish two *separate prediction paths* on the same pattern. The experiment must control source overlap and choice of predictive distributions; a raw comparison of `f` bytes is not second-order JSD. The conditional-equality trick has a direct candidate expression as `EqU32Via` / joined support counts over an anchor population, subject to D-FPW's real fan-in and multiplicity limitations.

**Red / green pair:** make two patterns equally frequent, but only one violates a declared independence prediction with sufficient support; require surprise ranking to favor the latter. Then shrink sample size while retaining the same apparent frequency gap; the distribution-aware variant should reflect uncertainty instead of declaring both equally established. A permutation that changes shared-variable bindings must break a naive product-of-marginals estimate but not the properly joined model.

**Do not claim:** a general-purpose JSD surprise oracle is already reproducible from the paper, or that this is equivalent to entropy maximization.

## P3. OpenCog _Interesting Pattern Mining_ (historical technical note)

**Source:** [OpenCog wiki, Interaction Information section](https://www.wiki.opencog.org/w/Interesting_Pattern_Mining) (project documentation, **not a peer-reviewed mathematical equivalence theorem**).

**Exzerpt, sinngemäß:** The worked three-pattern example sets `H(X) = log2 Count(X)` only because **each matching instance is treated as equiprobable**. For three overlaps it uses:

```text
I3(A,B,C) = H(A)+H(B)+H(C)-H(AB)-H(AC)-H(BC)+H(ABC).
```

With their counts `Count(A)=Count(B)=Count(C)=10`, `Count(AB)=Count(AC)=Count(BC)=Count(ABC)=5`, the note derives `3 log2(10)-2 log2(5)` (~5.322 bits) under **that counting measure**.

**Falsifier:** reproduce exactly from mask intersections and `Count` with the published fixture, then perturb instance weights. If weighted probabilities are unequal, `log2 Count` is no longer the entropy of the weighted instance distribution. Also separate sign conventions for multiway co-information from higher-order synergy labels: a positive/negative sign alone does not establish a causal interaction.

**Do not claim:** `H(X)=log2 Count(X)` for arbitrary graph attributes, evidence weights, CE64 states, or binary facts.

## P4. Goertzel (2021), _Patterns of Cognition_

**Source:** [arXiv:2102.10581](https://arxiv.org/pdf/2102.10581#page=19), §§3.4–3.6 (attention, clustering, mining) and §5.2 (foldability and mutual associativity), PDF pp. 17–24.

**Exzerpt, sinngemäß:** Inference, attention and pattern mining can be modeled as discrete decision processes that choose and combine substructures; the formal reduction to certain fold/unfold decompositions requires **mutual associativity** of the combination operations. Pattern mining's decisions combine pattern quality, expansion operators and expected reward, not just maximum uncertainty.

**LG consequence:** test first whether the selected *inference-composition and scheduling* operation is associative when it contains evidence overlap, saturation, round-off or history. If not, scan-order rebracketing may change the output; distinguish an order-sensitive algorithm from an associative monoid fold. Compare reward for useful target discoveries against Shannon and JSD; do not call all "entropy".

**Red:** permute hypothesis order and evidence-source overlap; one resulting output difference falsifies a blanket order-independent fold claim. Keep source-order semantics where required instead of normalizing away real behavior.

## Compact experiment and decision ledger

| Arm | Input interpretation | Main negative control | Positive criterion |
|---|---|---|---|
| binary `H(f)` | `f` treated as a scored pseudo-probability | uncalibrated `f`, irrelevant noisy 50/50 | may be a heuristic; **not** epistemic EIG without calibration |
| BALD-like MI | weighted predictive hypotheses + model noise | all hypotheses give 50/50 vs two deterministic camps | separates inherent label randomness from reducible disagreement |
| pattern JSD | empirical vs inferentially predicted second-order distributions | tiny vs large sample with same apparent discrepancy | expresses evidence uncertainty under declared prior and sample model |
| interaction information | explicit equiprobable instance-count model | unequal instance probabilities; shared-variable mismatch | reproduces reference count example only under its assumptions |
| confidence/Staunen/Moore | same graph and truth laws as entropy arms | high-entropy distractors, duplicate source, no-meta | best useful-oracle yield per budget, with full cost |

**Proposed gate:** record `{source_revision, target_random_variable, prediction_model, evidence_provenance, scheduler, source_counts, score, oracle_answer, budget, decoded_bytes}`. Equalize rule engines and compute budgets before comparing attention. Pre-register useful-discovery metrics, then perform ablation by *swapping only the scheduler*. Verify no source-oracle labels leak into the acquisition score.

**Loose ends:** formalize posterior weights for multiple hypothesis layers; implement no-meta / no-noise-learning comparator; extract executable PLN priority-queue semantics from source; fix normalization/zero-count policy for JSD and log counts; specify per-pattern independence estimator and uncertainty prior; retain a "no probability mapping" refusal mode.

**Deliberately excluded:** performance claims, a new CE64 bit meaning, a second IR/engine, and literal word-for-word paper excerpts. Statements above are short paraphrased research extracts tied to explicit pages and falsifiers.
