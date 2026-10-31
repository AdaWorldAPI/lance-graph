# D-STAUNEN-4 | Metaplasticity and multi-thought adjudication | draft
**Docs only; no claims of AGI or consciousness.**
## First pass
- Budget: 0 sealed givens, up to 63 sparse hypotheses; every W=thought slot 0..63 qualified by live generation/base version and never used as durable provenance. ThoughtCommitId + DurableRef separate.
- A thought tests an *explanation or learning-policy proposal*, e.g. STDP, novelty-gated STDP, NSL-regularized, XLOG candidate induction, cortical perspective / ACT-R goal modulation. Some policies may share evidence; never count shared roots independently.
- Compare held-out predictive behavior, neural trace parity, learning cost, causal ablation, conflict/contradiction, side effects and replay. Permute shuffle and no-meta negative controls. A hypothesis must not raise its own certainty through circular revision.
- Rule selection can itself be learned (metaplasticity); biological parameter updates require Atlas-allowed state transitions, symbolic revisions require source roots, confidence isn't a posterior without proper independent likelihood/evidence model.
- WG + Moore local activation -> candidate structural changes -> independent supervision -> consolidate or abstain, preserving protected Thought-Git intermediate readers and multiplicity.
- ACT-R goals, PointCare situated context, qualia operational readings are *optional experiments*, not prerequisites or phenomenology proofs. Brain SoC is later software/hardware compatibility qualification, not assumed.
## Research gates
M0 baseline/pinned data; M1 64 slot and stale/recycle; M2 root dedupe; M3 siblings cannot self-vote; M4 counterfactual intervention and read protection; M5 learned-policy ablation; M6 replay across restart; M7 held-out utility vs compute; M8 integration/admission and publication claims.
**Prereq:** D-STAUNEN-1..3; #1599/#1628, #1739, #1847. Any implementation work is owned by CC after explicit review.

## Round 2 | Cross-thought meta-learning protocol and CC integration gates

### Scientific target (hypothesis)
Does **learning which local plasticity or symbolic-revision policy to run** improve subsequent adaptation under equal model capacity, source evidence and compute, compared to the strongest single policy? This is **metaplasticity / learning-to-learn**, not a result of having 64 branches by itself. User's Staunen is unexpected novelty initiating plasticity, while wisdom is a *held-out, independently supported* consolidation result. Operational qualia, ACT-R goals and PointCare context may define signals or tasks but do not certify subjective experience.

### Inputs / outputs
- Input: one sealed WORLD/Atlas definition and frozen initial biological instance; distinct observation datasets/seed identities; Atlas neural events/weights/traces; XLOG compiled rule provenance; NSL exact structural graph; optional VSA codebook; thought slot leases with base source seal.
- THOUGHT 0 = sealed givens; THOUGHT 1..63 = independently addressable sparse speculative deltas; 64 slots is capacity, not obligatory concurrency. `W` and per-thought presence words do **not** uniquely identify evidence or permanent commit. Use source-root set and generation-qualified `ThoughtCommitId` / `DurableRef` from Thought-Git owner.
- Output: proposed policy rank, accepted/refused candidate, external outcome score and uncertainty with separate model-specific calibration, counterfactual result, protected replay receipts; only an admitted write path can land the next sealed reference version.
- Reasoning across thoughts may read a masked `Predicate64 × Thought64 × CE64` bit-sliced *projection*, ideal 32 KiB dense logical payload for 64×64 words, but no persistence requirement or false assumption that all 64 thought slots were filled.

### Four coupled but distinct learning clocks
1. **Neural**: spikes/eligibility, STDP/traces, signed neuromodulator and physical simulation time.
2. **Structural**: candidate synapse activate/deactivate under model limits; separate eventual morphology/myelination research.
3. **Epistemic**: source-rooted NAL/CE64 revision, causal question, Epi5 rung only if licensed; no shared-root self-revision loop.
4. **Metacognitive**: learn policy/attention/reasoning-cost preference over *out-of-sample* outcomes, without itself writing factual truth.

### Competing arms (same dataset, seeds, compute / call budget)
B0 Atlas fixed SNN without learning; B1 Atlas STDP; B2 Atlas novelty-gated STDP; B3 HDC decoding only, no physical feedback; B4 HDC+novelty structural candidate; B5 XLOG/NSL added with exact semantic oracle; B6 B5 + meta-policy selection over distinct thoughts; B7 shuffled novelty; B8 shuffled physical topology/Moore view; B9 duplicated-root/63-thought fake quorum; B10 no-meta fixed best policy; B11 randomized policy (capacity/control match). Run B0/B1/B10/negative controls before any training gain headline.

### Gate and kill checklist
M0 Pre-register task split, minimum effect, no-leakage source families and same computational budget; freeze exact Atlas model, XLOG compiler, NSL oracle, VSA codebook.
M1 64-thought lease: no 65th; W reused after squash/land must not recover old evidence; stale frame refused; protected readers of `Intermediate`, `Contradictions`, `Divergences`, `Multiplicity` survive or squash refuses (#1739).
M2 Root-disjoint evidence: 63 clones of one observation count once; correlated observations must be modeled explicitly, not counted as 63 priors; no artificial confidence multiplication by repeated folds.
M3 XLOG derivation is not source independence; CE64 composed indirect link stays `Open` until an independently supported certification step, and unsupported intervention refuses rather than pretending `Causes`.
M4 AB test recorded physical effects vs no-effects control and a causal intervention/cut; do not permit altering WORLD for a speculative hypothesis. Record both attempted and refused edits.
M5 Compare learned policy selection with best fixed and randomized arm; report both *held-out task score* and fold cost, including classification/generalization tradeoffs and unsuccessful runs. Synthetic difficulty and fixture separability labeled.
M6 Replay and checkpoint: same model version, source roots, RNG, scheduled events, thought receipts and numeric profile returns equivalent observations; a different partition/numeric profile is a separate qualification.
M7 Cost accounting: prepared exact masks ns/batch µs hot gates; compile/plan/oracle cold overhead separate. Constrain memory size, no full synapse × thought cube; measure actual resident bytes and source-to-local prep.
M8 Publish claims: positive gate supports restricted learning-to-learn on specified tasks, not general intelligence, complete biological emulation, anthropomorphic qualia, or proof that neuronal and logical rules are one mathematical learning law.

### Read current PRs before coding
- #1400 and #1440: Moore/NARS situation-to-recipe learning, delayed-credit ablations; last Holm-controlled delayed-credit claim **inconclusive**.
- #1844: `ScentLearner` ranks investigative leads, does not write truth or ledger, isolated measured small hot operation; positive/negative controls.
- #1847: proposed Predicate64×Thought64×CE64 bit-sliced cross-thought candidate, root-disjoint tests and false positive cases; still research-only.
- #1628, #1739: thought slot generation, protected squash, durable receipts and commit provenance.
- #1884: native XLOG result topology `IndirectKnown×Open`, not certification.
- `CLAUDE.md`: no serialized tuple builders in the fold; no ms-scale result claims; no new implementation without owner/gate.

### Scientific reference ledger
- Abraham & Bear (1996) metaplasticity: https://doi.org/10.1016/S0166-2236(96)80018-X
- Miconi, Stanley & Clune (2018) differentiable plasticity/meta-learning: https://arxiv.org/abs/1804.02464
- Gerstner et al. (2018) three-factor eligibility: https://doi.org/10.3389/fncir.2018.00053
- Frey & Morris (1997) synaptic tagging and capture: https://doi.org/10.1038/385533a0
- Denève (2008) local spike-dependent Bayesian learning *under a declared model*: https://doi.org/10.1162/neco.2008.20.1.118
- Lisman & Grace (2005) hippocampal/VTA novelty: https://doi.org/10.1016/j.neuron.2005.05.002
- Spaun/NEF semantic pointer representation: https://pmc.ncbi.nlm.nih.gov/articles/PMC4762696/
- ACT-R cognitive modeling: https://www.routledge.com/The-Atomic-Components-of-Thought-1st-Edition/Anderson-Lebiere/p/book/9780805828160
- Replay consolidation review: https://doi.org/10.1016/j.neuron.2023.03.005
- AtlasIR manuscript/code: https://github.com/RockLi/brian2-atlas-preprint ; https://github.com/RockLi/brian2-atlas

### Proposed CC work/PR sequence after docs land
(1) Read-only current-state report and falsifiers -> (2) C0–C6 typed identity/gates lab -> (3) A0–A3 Atlas/HDC biological observation lab -> (4) X0–X4 exact XLOG/NSL parity lab -> (5) A4–A7 novelty/stability ablations -> (6) X5–X8 numerical/candidate-hydration lab -> (7) M0–M8 independent evidence + metapolicy experimental report -> (8) separately approved production admission.
Every numbered item is a **proposed** new PR or lab unit, never an instruction to directly edit production or auto-merge. CC should request owner approval when a step crosses Atlas, CE64 ISA, DT64 methods, or Thought-Git authority.
