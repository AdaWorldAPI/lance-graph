# NARS epistemic needs, internal experience and independent evidence: source excerpts for CE64

**Date:** 2026-10-12. **Status:** PRIMARY-PAPER DISTILLATION / OPEN PROBES. Follow-up to the *merged* [#1496](https://github.com/AdaWorldAPI/lance-graph/pull/1496); its files cannot be amended in place. No semantic migration, truth-law decision or measured result. **Siblings:** [#1497](https://github.com/AdaWorldAPI/lance-graph/pull/1497) (PLN/entropy policy), [#1498](https://github.com/AdaWorldAPI/lance-graph/pull/1498) (metagraph/MeTTa projection).
**Read first:** [CE64 substrate plateau](../.claude/knowledge/ce64-reasoning-substrate-plateau.md), [D-HL plan](../.claude/plans/ce64-hypothesis-layers-v1.md), [existing D-HL research map](2026-10_ce64_hypothesis_layers_research_map.md), D-RPF-P5, D-CE64-CYCLE/STAUNEN and D-HL-3/4/8.
**Method:** short *paraphrased* excerpts from original papers with page/section, what the authors actually claim, what we merely hypothesize for LG, and a falsifying control. No long verbatim copyrighted passages.

## Why these three sources and not another paper pile?

- **Wang et al. (2018):** epistemic limits, provenance, internal events and resource-based feeling mechanisms already coexist in NARS; beware confusing their operational self-model with phenomenal qualia.
- **Jøsang (2007):** independent versus dependent evidence does not permit the same accumulation law; the reverse operation *fission* makes an unusually good counterfactual/forgetting test.
- **Goertzel (2021):** the formal bridge between folding and cognitive process rests on stable read-world and associativity assumptions; our append-on-write/merge-on-read is a candidate way to *test* this, not a proof.

## N1. Wang, Li and Hammer, _Self in NARS, an AGI System_ (2018)

**Primary:** [Frontiers DOI:10.3389/frobt.2018.00020](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00020/pdf), §§2–3 (printed pp. 3–8). The paper studies the existing conceptual mechanisms of NARS, not neuroscience or a demonstrated subjective consciousness.

**Exzerpt A, §§2–3.1, pp. 3–4, sinngemäß.** Truth values are experience-grounded: frequency describes the evidence's direction, confidence how stable/adequate the frequency is under further evidence. A system cannot assume infinite knowledge or resource availability. NAL's induction/abduction are classified as weak inference, with their claims potentially revised as more evidence arrives; deduction is handled separately. Revision is selected for compatible conclusions on **disjoint evidential bases**; *choice* is used if those bases overlap.

**LG implication:** F/C in CE64 is not a certificate that a statement is direct/indirect, observed/inferred, or causally supported. These are separate ClassView/Epi5 and provenance readings. A layer ID is not an evidence source. The proposed `(layer,seq,source_base)` delta and D-HL-3 tests embody this distinction.

**Falsifier A:** the same observation enters two hypothesis layers, and is not counted twice. Two *independent* observations in one layer are both counted. Switching the duplicate's layer must not change the revised confidence. The proposed test must compare against the pinned actual NARS rule (including its evidence-base overlap logic), not merely the same scalar reimplementation.

**Exzerpt B, §3.4, pp. 7–8, sinngemäß.** NARS regards internal processing events (selected concepts, inference steps and derivation relationships) as a form of **internal experience**. Such traces may later answer where a conclusion came from and what the system was considering. This enables limited self-knowledge that is **itself incomplete and revisable**.

**LG implication:** there are at least three different records: (i) evidential origin of an external observation, (ii) derivation/dependency receipt of a conclusion, and (iii) attention/saccade trace showing what was selected and when. They are not interchangeable: `AlphaStamp {cycle,seq,rung,visits}` can describe an attention trace without proving evidence independence.

**Falsifier B:** hold the final CE64 `(S,P,O,F,C)` value constant while varying its derivation chain. A source-of-conclusion query must distinguish the chains **without increasing truth confidence**. Deleting the receipt should impair provenance reconstruction but not alter the factual final truth in a read-only baseline. Replaying twice cannot create a second observation.

**Exzerpt C, §3.5, pp. 7–8, sinngemäß.** The system computes local satisfaction by comparing the current event's truth/desire state and also tracks system-level satisfaction, which can influence allocation of scarce cognitive resources. The paper calls these functional processes "feeling" and "emotion"; its proposal should not be interpreted as an empirical proof of machine phenomenology.

**LG implication:** user-described `Emberglow`, `Steelwind`, `Woodwarm`, `Velvetpause` may be *names for appraisal/priority policies* over independent observations, not new mandatory truth bits or evidence sources. The user's cognitive-Maslow analogy (epistemic needs enabling exploration) is a research framing, **not a claim made by Wang or Maslow about CE64**.

**Falsifier C:** same evidence, same NARS truth and semantic read, same base workload; change only the declared appraisal policy. If target discoveries, resource allocation, latency or recovery do not change, no demonstrated functional role. If appraisals change factual outputs under an allegedly policy-only experiment, record a confound. Include a no-appraisal control and a policy-habitual/noise trap.

**Boundary:** Wang's paper offers a functional model of the self and explicitly speculates about experience and consciousness. Nothing above claims actual subjective feelings in LG, NARS or a language model.

## N2. Jøsang (2007), _Cumulative and Averaging Fission of Beliefs_

**Primary:** [arXiv:0712.1182](https://arxiv.org/pdf/0712.1182#page=1), abstract, §1, §3 and fission sections, PDF pp. 1–10. **The belief framework is Subjective Logic, NOT identical to NARS or PLN.** Translate evidence semantics only with an explicit verified mapping.

**Exzerpt D, §1 and §3, pp. 1–3, sinngemäß.** For *independent, non-overlapping evidence* (e.g. different observation periods), cumulative fusion adds evidence. When observers base their opinions on the same underlying evidence, an averaging form is appropriate rather than cumulative accumulation. The distinction depends on evidential dependence, not whether two people or two layers exist.

**LG implication:** derive a dependency-aware fusion operator from the per-delta source/base record. Avoid treating correlation of sensor readings as proof of identical evidence: source IDs, causal chains, shared upstream samples and repeated derivations are not the same property. Compare D-HL-3 with a case of **partly overlapping** source masks; neither naïve full sum nor naïve full average is obviously licensed by the simple two-extreme-case description.

**Falsifier D:** copies of the same source across layers do not narrow confidence; independent sources do. When a source occurs in both derived results, a later recombination must respect its overlap mask (rather than just comparing immediate layer IDs). At least one "partially shared ancestor" fixture must fail a naïve rule.

**Exzerpt E, §1 and §§4–5, sinngemäß.** The paper studies **fission**, i.e. calculating the remaining contribution after removing one known component from a previously fused opinion, subject to the selected fusion rule and its mathematical preconditions.

**LG consequence:** this is a sharper definition of *epistemic ablation* than simply deleting a node: retract or mask one contributor and recompute the same output with all other contributions fixed. Our append-and-fold / replay can implement a candidate *counterfactual operation* without mutating immutable source addresses, provided dependency and original source weights were retained.

**Red / silence twins:**
- Independent A+B fused, remove A, recompute and compare to original B (within declared u8 rounding, preferably using wide accumulators until commit).
- Remove an evidence source **not present**: no change. Remove a copied/overlapping source represented twice: both copies lose influence, but no independent source does.
- Dependence model unknown: **refuse exact fission** or report bounds; never claim an exact inverse from only compressed confidence bytes.

**Caveat:** fusion/fission algebra over subjective opinions is not automatically a NARS revision law. This source supplies a *control model* for independence and retraction, not authority to rewrite NARS truth semantics.

## N3. Goertzel (2021), _Patterns of Cognition: Cognitive Algorithms as Galois Connections Fulfilled by Chronomorphisms_

**Primary:** [arXiv:2102.10581](https://arxiv.org/pdf/2102.10581#page=4), §1 pp. 2–4 and §5.2 pp. 23–24.

**Exzerpt F, pp. 3–4, sinngemäß.** The formal re-expression of recursive metagraph decision systems using fold/unfold applies *rigorously* under a **nonchanging graph during the fold** assumption; the paper openly notes this does not generally hold for cognitive systems updating their own knowledge store and that the mathematical extension was not yet established.

**LG implication:** distinguish **immutable identity** from **snapshot-stable read semantics**. An immutable key does not prevent a fold from seeing newly appended values or a changing hypothesis mask. Version/read-snapshot pinning is the correct object of proof, not just a GUID invariant.

**Falsifier F:** pin the world at version k, compute a fold while writing deltas for version k+1 and run it again after restart. Version-k answers must be identical and not include k+1 claims. A deliberately live read may differ, but must be declared a different semantics. Re-read/commit does not silently promote speculative layers or alter source evidence.

**Exzerpt G, §5.2, sinngemäß.** Some cognitive optimization processes are foldable when their combination operations meet **mutual associativity** conditions. Evidence overlap, u8 saturation, per-step rounding and temporal read ordering can violate naïve attempts to interchange the fold order.

**Falsifier G:** check `(A⊕B)⊕C` vs `A⊕(B⊕C)` with independent sources, duplicate sources, saturation near confidence code 255, and noncommuting causal/temporal steps. The comparison must specify the exact fusion, quantizer, and snapshot. A single differing byte refutes an *unqualified bit-exact associative fold* claim; it does not refute an order-sensitive computation.

## What should be measured next (no implementation authority)

| Probe | Has to keep fixed | Distinguishes | Kill condition |
|---|---|---|---|
| NARS evidence overlap | same source events, rule and `k` | layer address vs evidence provenance | copy raises confidence |
| NARS source-of-belief | final F/C, class and observations | inference receipt vs observation ID vs attention | receipt deletion changes belief strength |
| Jøsang fission-inspired ablation | fusion model, source mask, exact wide weights | contributor removal vs graph deletion | removing absent contributor changes output |
| Functional appraisal | evidence, truth and budget | priority/emotion-like policy vs epistemic truth | effect disappears vs no-policy or is only added compute |
| Snapshot and fold | fixed version/row/mask projection | static theorem conditions vs live world | replay over pinned version changes |
| Associativity | exact operands + provenance | fold algebra vs quantization artifacts | reorder silently alters supposedly bit-exact result |

**No new CE64 bitfield claims:** preserve frozen opaque LE ALU/ISA contract, class-declared semantic readings and D-HL-1's unresolved O1–O8 decisions. The 64 hypothesis layers are a budget, not 64 levels of awareness. The evidence source is not the layer; inference receipts are not probability samples; appraisal states are not claims of qualia. The goal is to test whether an epistemically honest foundation supports richer adaptive exploration, not to announce emergent consciousness.

**Open decisions:** how the evidence-source set survives truncation; the provenance schema for cycles and learned derivations; exact rule for partial overlap; fission with only 8-bit compressed confidence; whether appraisal improves target discovery under equal computation; definition of snapshot boundary during asynchronous append. Source references provide experiments to *ask*, not decisions to silently adopt.
