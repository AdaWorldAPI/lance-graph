# Coresearch 2026-10-23: thought-git accumulation as a large Markov chain

**Status:** exploration map. Nothing is adopted; the operator chooses.
**Question (operator):** "Consider that giant thought-git accumulation as a large Markov chain."
Sources: Bittracher & Schütte, *Physica D* 416 (2021) 132799, read in full;
Askarian, Xu & Faragó, *Algorithms* 9(3):50 (2016), arXiv 1809.06564, read in full.

## Frame (what the scouts established)

- **F1.** A version is a deterministic function of the givens, the prior
  version and the landing thought set (`oracle.rs:46-108`, verified in code).
  Thought-git is therefore a transducer, or a Markov decision process; it is a
  Markov chain only once the policy that chooses thoughts is fixed.
- **F2.** No ruling grants the Markov property to lane accumulation.
  E-SUBSTRATE-1 covers bundles and E-MARKOV-TEMPORAL-STREAM-1 covers stream
  order. Last-writer replace is neither, and the seal is an order-dependent
  fold (`seal-vs-temporal-ordering-information.md`).
- **F3.** Lanes are measured not independent: they share the manuscript, the
  tokenizer and the model (`ISSUES.md:3852`).
- **F4.** No code in `crates/` estimates transition probabilities, tests
  lumpability or bounds an aggregation error. The search space was a closed
  `git grep`; ndarray/hpc was not searched.
- **F5.** Two earlier rejections apply here.
  - #1703: a replay capsule is not proof authority, and population evidence
    only nominates.
  - TORQUE T-2026-10-23-1 (killed): a learned PaletteLut aggregation did not
    compose.

## Where the papers stop applying

- **B&S Algorithm 3.1.** It needs a restart from any state to draw a column.
  A trajectory log cannot do that. Trajectory-only estimation needs about
  p·r samples (Zhu et al., arXiv 1907.00113, Thm 3, section-read), so "J
  independent of N" does not transfer [S]. It also requires homogeneity
  (fails per F1) and independent samples (fail per F3).
- **AXF Theorem 2.** It needs an irreducible, aperiodic chain. Accumulation
  over versions is a transient DAG with no recurrence, so the theorem does
  not apply.
- **What survives** without the Markov property and without independence:
  - min/max L/U bounds over counts (AXF Def. 3);
  - deterministic congruence found by partition refinement;
  - the type split: certificate (replay + seal) versus model.

## Exploration map

| idea | verdict | why / probe |
|---|---|---|
| Squash folds only committed writes | **ADOPT-NOW, done** | Found by the falsifier: `Lane::final_state` folded Prune/Veto/Plan writes. Fixed in #1736 with P8 (red before the fix, green after; disable D6 turns only P8 red). |
| D: squash equals chain for every later read | PROBE (S) | Fixtures: (b) contradiction then rewrite; (c) intermediate divergence then agreement; (d) the same value 3× vs 1×. Predicted: the intermediate contradiction and divergence records, and the multiplicity, are lost. Decide whether that loss is acceptable before building a certificate on the squash. Silent twin: `clean_lanes`. |
| D-59..63: the squash preserves the epistemic promotion | PROBE (S), blocked | The promotion reader must be pinned first. Blame is a lane bitmask, not a count, so a multiplicity-reading promotion breaks. |
| I3: counted ε-spread L/U gate | PROBE (M) | Bridge: R2 is ε = 0 with a point mass per lane. Count lanes per (cell, value class) from the blame bitmask; refuse when U−L > ε by integer cross-multiplication. With ε = 0 it must reproduce R2 (the p2/p4 tests). It is a BOUND only when the counts are exhaustive; otherwise it is an empirical interval, a gate annotation and never proof authority. |
| I7: coarsest congruence by partition refinement | PROBE (S–M) | Integer, cold. It supplies I3's blocks. Kill if the reduction is below 2× on real traces. |
| F1: is the process Markov at all, over a chosen state space | PROBE (M), needs real traces | First the revisit rate: below 1% means untestable on that state space, so coarsen. Then an order-1 vs order-2 contingency test against a Jirak band; kill if more than 10% of states differ. A synthetic policy decides the answer in advance, so it does not count. |
| F3: lane independence after conditioning on shared inputs | PROBE (M), needs real traces | Conditional mutual information against a time-shuffled null. Precondition for any cross-lane convergence gate. |
| I12/I13: lossless op-log encoding, tags plus cleanup | PROBE (S) | Use Lance's own column encodings and version tags; no parallel store. This answers the terabyte concern without any Markov assumption. |
| I2 P = Π P̂ Λ; I5 defect; I6 interval propagation; I10 Chapman-Kolmogorov; I11 VAMP; I16 MarkovTripletState | PARK | Each needs F1 to pass (homogeneity) or the dormant VAMPE+Jirak track. All are cold-path and report-only. |
| I1 J-column SVD; I4 AXF convergence; I14 Nova IVC | SKIP | I1: no restart oracle and dependent lanes. I4: transient DAG. I14: licence unknown, far outside the ns law. |
| I8 causal state; I9 certificate vs model | ALREADY-HAVE | `final_state()` is the causal state of a deterministic lane; replay plus seal is the certificate. |

## Head selection (operator follow-up)

Recency is not the criterion. The gate-grade parts are subsumption (R1/R2)
and the exact L/U spread. The others are not:

- cross-lane convergence (R̂) presumes independent lanes, which F3 refutes;
- stability (ρ) and predictive sufficiency are cold-path and report-only.

## Not searched

- ndarray/hpc (clam_compress, cam_pq);
- GOLDSTAUB, identity-temporal-evidence-primer, holographic-temporal-access-projection;
- the IDEAS joint-replay row;
- Buchholz and Franceschinis-Muntz bounding-aggregation papers (not located);
- bisimulation metrics (abstract only).

## ⊘ Correction 2026-10-23: the probabilistic premise was a category error

DECISION (operator, 2026-10-23): the substrate is not probabilistic. SPO-FC
being probabilistic today is legacy. Every probabilistic model counts as a
**counterfactual** and must be certified by the egglog / eggcc reasoning
compiler before anything rests on it. The 64 lanes × ~10^6 folds run per
second are the **material** a thought is compiled from, until it is a proof
that later operations can call by reference. They are not a population to be
aggregated statistically.
SCOPE: this map and the IDEAS row "A certificate replaces the deltas" (its
Markov block). BASIS: the operator's statement; consistent with F1 (versions
are deterministic), #1703 (population evidence only nominates) and
`dr-math-binding-probe` (`admit_recipe` ends in `NoCertifiedCompiler`).
REVISIT WHEN: never on statistical grounds.

What this changes in the map:
- The Bittracher/Schütte and Askarian papers describe at most the **search
  over counterfactuals**, never the admission and never the certificate.
  All their rows (I1–I6, I10, I11) drop to PARK as counterfactual-layer
  descriptions. The certificate/model split (I9) holds and is sharpened: the
  certificate is a checked egglog proof, not a replay witness and not P̂.
- The premise gate failed. Phase 0 split certificate from aggregation but
  never asked whether the substrate is stochastic at all. F1 (versions are
  deterministic) already said it is not; the brief should have stopped there.
- Head selection: the head is the thought whose compiled proof is admitted.
  The children's folds are the search trace. After admission they are
  discarded, except the seal and the refuted branches the proof cites.
- What actually blocks this: no certificate exists for DR64 (`dr-math-binding-probe`,
  `NoCertifiedCompiler`). The open items there are a typed relational statement
  language, egglog laws with checked proofs, and a lowering check. The candidate
  first law is associativity of Boolean relation composition. `crates/egglog-r2il-probe`
  holds the cold compile path; `.arxiv/2026-10_eggcc_effect_safe_extraction_statewalk_r2il_map.md`
  holds the effect-safe extraction lesson.
- Still valid: the P8 fix (only committed writes squash) and probe D
  (squash = chain for every later read). Both are deterministic.

## Closeout

`coresearch | STATUS: open | OUTCOME: one defect fixed in #1736 (P8); probabilistic premise withdrawn (category error) | OPEN: DR64 certified compiler (typed statement language + checked egglog laws + lowering check); probe D; promotion reader unpinned`
