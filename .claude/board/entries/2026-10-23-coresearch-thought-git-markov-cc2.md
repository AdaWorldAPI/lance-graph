# 2026-10-23 — /coresearch: thought-git accumulation as a large Markov chain

**Status:** EXPLORATION MAP. Nothing in it is adopted; the operator chooses which ideas go forward.

**Input:** the operator, citing two papers:
- **B&S:** Bittracher & Schütte, *A probabilistic algorithm for aggregating vastly undersampled large Markov chains*, Physica D 416 (2021) 132799, CC BY. Read in full.
- **AXF:** Askarian, Xu & Faragó, *Utilizing Network Structure to Accelerate Markov Chain Monte Carlo Algorithms*, Algorithms 9(3):50 (2016), doi 10.3390/a9030050, CC BY. Supplied mid-run and read in full. The PDF came from the mdpi-res.com CDN because www.mdpi.com returns 403 here; sha256 `e219f6ea…12b0`.

**Follow-up question folded in:** besides CE64 59..63 certification, witness support and resolved contradiction, which heuristic decides that a thought becomes the head and its children are squashed? And what is MCMC?

## Question, after the premise gate

The premise auditor returned **PREMISE-SPLIT**. The thought-git ops are deterministic:
- `receipt.rs` `run` is pure, and the jitter is a hash of the recorded seed.
- The version state is a growing cell map (`oracle.rs` `next_version`).
- The lanes are coupled through one 63-slot pool (`slots.rs`).
- No transition matrix P exists over any named state.

A squash composes write-sets along a path, which is a coarser lag, not a quotient of states. Promotion on 59..63 is a question of evidence, not of correlation. The question was therefore re-asked as four:

- **Q1:** is there a stochastic process at all, and over what finite state?
- **Q2:** squash safety = commutation with sequential landing.
- **Q3:** promotion; aggregation can at most be a necessary pre-check.
- **Q4:** not storing 64×10⁶ deltas = equivalence classes of receipts.

## Answer

The Markov machinery does not apply to the thought-git cells.

- Ops are functions, not draws. A 0/1 kernel has ergodic coefficient ρ = 1, so the AXF and B&S bounds are vacuous on it.
- The stochastic layer that does exist is **induced by aggregation**. Deterministic ops observed through a lossy palette `snap` give a kernel K(t|s,b). Its row entropy is positive exactly where purity is below 1. This is the exo129 chain (#1731), and it is where lumpability (EXO-132) lives.
- A second stochastic layer could come from seeds or driver choices. It exists only if someone defines it, with a named sampling law.

What survives for thought-git is algebraic:
- Commutation sweeps (Q2).
- Exact congruence by partition refinement instead of sampled purity (Q1-ii / EXO-132).
- Receipt classes keyed without `thought`, guarded by root identity (Q4).
- A head rule built from refinement, subsumption and root blocking (follow-up).

Promotion on 59..63 stays where it is: `pearl::revise` on a new independent root (Q3).

**No tolerance ε ever goes on the D-TGA-5 gate.** A tolerance would admit stacks the exact refusal rejects.

**MCMC, for the record.** You build a chain whose stationary law is a specified target π, run it until it has mixed, and read π(A) off sample frequencies.
- AXF adds that for one quantity π(A) only the partition {A, Ā} has to be ε-lumpable. The error is then ≲ ε/(p₀+q₀), independent of N.
- The bound survives a time-varying chain whose aggregated step matrices stay inside an interval [L, U] (Hartfiel's Markov set-chains).
- Thought-git has no specified target distribution. Monte Carlo over seeds is sampling, not MCMC.

## Crosswalk and verdicts

R = the relation of the outside idea to the existing surface. FW = the firewall critic's verdict.

| id | idea | R | FW | map |
|---|---|---|---|---|
| X1 | "64×10⁶ ops = one chain over cell states; the certificate is P̂" (#1736 IDEAS block) | CONFLICTS-ANCHOR | CATEGORY-ERROR ×3 | **SKIP**, and correct the block (below) |
| X2 | squash = composition along a path (lag), not lumping | ALREADY-HAVE | FITS | ADOPT-NOW as vocabulary |
| X3 | commutation sweep over random interleavings | PARTIAL | FITS (rename) | **PROBE PR-2** |
| X4 | coarsest congruence by partition refinement (Paige–Tarjan / Valmari–Franceschinis O(m log n)) as the exact form of purity | PARTIAL | FITS (cold, blocks as masks) | **PROBE PR-3** |
| X5 | question-specific ε-lumpability {A, Ā} with the AXF bound | NEW | FITS (pre-register A) | **PROBE PR-4**, exo129 layer only |
| X6 | [L,U] interval matrices for inhomogeneous chains (Hartfiel) | NEW | CATEGORY-ERROR on thought-git | PROBE as the PR-4 inhomogeneous arm on exo129; never on receipts |
| X7 | Hartfiel product bound as a squash tolerance | NEW-cond. | VIOLATES D-TGA-5 | **SKIP** |
| X8 | ergodic coefficient / start-independence of a lane | NEW | partial CATEGORY-ERROR (kernel, not lane) | PARK |
| X9 | receipt class table, key without `thought` (action cache) | PARTIAL | FITS (never addressed through W) | **PROBE PR-5** |
| X10 | path-dependent state never merged by output identity | ALREADY-HAVE | FITS | probe first, inside PR-5 / PR-7 |
| X11 | B&S column sampling + SVD sign structure | PARK-shaped | CATEGORY-ERROR | PARK until PR-6 un-parks it |
| X12 | lanes coupled through the pool ≈ NCD ⇒ not deflatable | NEW | FITS as a kill prediction | **PROBE PR-6** |
| X13 | KL aggregation loss | NEW-cond. | CATEGORY-ERROR | PARK |
| X14 | causal abstraction: the coarse map must commute with interventions (Beckers–Halpern; causal coarsening) | ALREADY-HAVE (anchor) | FITS as a necessary pre-check | PARK; never writes 59..63 |
| X15 | weak lumpability for the fixed initial law "fork from sealed main" | NEW | trivial with one base | folded into PR-3 |
| X16 | k-lumpability of a lag-k squash | NEW | not citable (abstract only) | PARK |
| X17 | head = the lane with most ops, or the lane's stationary mass under the driver | NEW | count ≠ QSD; stationary mass is endogenous and gives no evidence | most-ops only as a tie-break; **SKIP** stationary mass |
| X18 | head rule (a) ε on the question, (b) start-independence, (c) lossless subsumption, (d) a new root blocks the squash, (e) newer becomes head only if it refines | NEW | (b) CATEGORY-ERROR; (c) strongest; (d)/(e) FIT | **PROBE PR-7** with the order (d) block → (c)/(e) decide → (a) scope |
| X19 | drop history only when causally stable (op-based CRDT) | PARTIAL rhyme | FITS (eviction ≠ stability) | PROBE as bridge B7, low priority |
| X20 | log compaction / snapshot = fold of the prefix, valid iff apply is pure | ALREADY-HAVE | FITS | nothing to add |

**Bridge vs firewall conflict, resolved.** The bridge architect kept (b) start-independence as head-rule step 4. The firewall critic called it a kernel property misapplied to a lane. The stricter verdict wins: (b) is dropped from the head rule. The ρ histogram per program (bridge B6) stays measurable as a description, never as a gate.

## Head-selection answer to the follow-up (candidates; none measured)

Applied after D-TGA-5 R1/R2 pass, as an ordered chain that can only refuse or narrow, never promote:

1. **(d) Root block.** A child whose `roots` contain a root the head lacks is evidence, not redundancy. Refuse the squash with a new kind, `IndependentRoot`.
2. **(e)/(c) Refinement and subsumption.** The head's final state covers every cell the child agreed on, with no contradiction, and the child's receipt classes replay inside the head's base. This is the real head order; "newer" wins only when it refines.
3. **(a) Question scope.** Children that differ only outside the pre-registered question cells Q may merge with respect to Q.
4. **Tie-break.** Most ops, then lowest id. This is the current D-TGA-5 rule, demoted from semantics to tie-break.

CE64 59..63 is read, never written. Choosing a head is not certification.

## Pre-registered probes (top three first)

All probes are new test files, with no production edits, run with `cargo test --release`, debug 0 and incremental off. Hot loops are reported in ns/op. Cold SVD or refinement is reported in µs and labelled "not a gate". Results are counts, folds and u64 hashes, never row sets.

**PR-7 — head selection.**
- Ground truth is authored before any run. Families, each with 64 seeded variants:
  - F1: the newer lane B refines the churning lane A. GT = B.
  - F2: the lanes differ only outside Q. GT = merge w.r.t. Q.
  - F3: the newer lane has the most ops and contradicts on Q. GT = refuse.
  - F4: equal op counts and B refines. GT = B.
- Rules compared: (m) most ops, (e) ε on Q, (r) refines-older, (s) stationary mass (reported only).
- PASS:
  - (r) ≥ 0.95 on F1, F3 and F4.
  - (e) ≥ 0.95 on F2.
  - (m) ≤ 0.5 on F1. KILL this prediction if (m) ≥ 0.95 on every family.
- Disable: drop the ⊇ check from (r).
- Silent twin: on #1736 `clean_lanes`, (r) returns "merge, no head", and the read equals P1.

**PR-2 — commutation sweep.**
- Seeds 0..1023, L ∈ {2, 4, 8}, chains of 1..16 commits on 48 cells.
- Half of the fixtures are anomaly-free by construction; half carry one injected divergence or contradiction.
- 32 interleavings each.
- PASS: zero accepted mismatches, and zero refusals on the anomaly-free half.
- Reported only: the false-refusal rate.
- Disable: remove R2, after which accepted mismatches must be > 0.

**PR-1 — locate the stochastic layer (the X1 kill).**
- 4096 receipts, each replayed 8 times.
- KILL X1 if D = 0, where D is the number of receipts with more than one distinct output fold.
- Locate arm: on the exo129 worlds, (row entropy H > 0) must equal (purity < 1). KILL the locate claim on any mismatch.
- Disable: put the replay index into the seed; D must then be 4096.

**Further probes:**
- **PR-3:** partition refinement. R* = snap-block count ⇔ purity 1.0, and purity < 0.99 ⇒ R* > snap-blocks. Disable: a single refinement pass.
- **PR-4:** AXF bound on a seeded half-palette observable A over 6 hops, plus the inhomogeneous [L,U] arm. Disable: ε = 0. Silent twin: A as a union of PR-3 blocks gives ε = 0.
- **PR-5:** receipt classes without `thought`.
  - PASS: zero classes with two distinct folds, and a dedupe ratio ≥ 10.
  - Disable: drop `base` from the key.
  - Silent twin: with `thought` in the key the ratio is exactly 1.
  - Roots differing within a class are reported as merge-blocked.
- **PR-6:** NCD prediction.
  - δ is the in-aggregate column spread on the driver chain and on exo129.
  - PASS X12: δ > 0.2 and no spectral gap > 5.
  - KILL X12, which un-parks X11: δ < 0.05 and gap > 10.
  - Disable: give each lane its own slots.

## Correction owed to the #1736 IDEAS block

The "Outside grounding: the accumulation as a large Markov chain" block on #1736 (`IDEAS.md`, session 01Vnc) maps the 64×10⁶ ops onto one stochastic chain over cell states. It names P̂ as the certificate, and it offers the 4ε bound as a squash-gate tolerance.

All three claims fail against the code:
- the ops are deterministic, so there is no P;
- the receipt is lossless and P̂ is lossy;
- a tolerance weakens the exact refusal.

The 63-lanes-as-iid-columns arithmetic also breaks I-NOISE-FLOOR-JIRAK. The block's own four falsifiers stand. This entry asks for an append-only storno of the "certificate is P̂" and "4ε tolerance" bullets, written by the block's owner or after #1736 merges. It does not edit #1736.

## Not searched

- Datomic/XTDB history retention.
- Primary text of Kemeny–Snell, Buchholz 1994, Dayar–Stewart 1997, Larsen–Skou and Givan et al. (all secondhand).
- PyEMMA/MSMBuilder licences. PRISM and Storm licences are from memory.
- The ComposeTable source and the `ps2_*` tests: the P-S2 purity ≈ 0.47 is cited from the EXO-129 entry, not re-read.
- The MCGS paper (fetch failed).
- A primary source for lumpability of inhomogeneous chains beyond AXF's set-chain bound.

## Sources (read grade)

- **B&S, AXF:** READ-IN-FULL (main thread).
- **Beckers–Halpern arXiv 1812.03789 §3.1–3.2; Zhang–Wang arXiv 1802.02920 §1, Thm 2:** SECTION-READ.
- **Abstract only:** Geiger et al. arXiv 1304.6603; Zhu et al. arXiv 1907.00113; Duan–Ke–Wang arXiv 1811.02619; Storm arXiv 2002.07080; Baquero et al. arXiv 1710.04469; Derisavi–Hermanns–Sanders 2003 and Valmari–Franceschinis 2010 (abstracts); Geiger–Temmel arXiv 1212.4375.
- **Section read via summarizer:** the deeptime/PyEMMA CK-test tutorial (deeptime LGPL-3.0, checked). Kafka log compaction docs: SECTION-READ.
- **Secondhand:** Kemeny–Snell 1960; Rubino–Sericola 1989; Buchholz 1994; Dayar–Stewart 1997; Larsen–Skou 1989; Givan–Dean–Greig 2003; Chalupka et al. 1412.2309; Bazel hermeticity; git objects; event sourcing; MCTS transpositions.
- **Internal, VERIFIED-IN-CODE (scout 1):** receipt.rs:339-418; oracle.rs:67-122; slots.rs:174-234; squash_gate.rs:89-235 (pr1736); exo129_learned_lut_probe.rs:142-190; pearl.rs:282, 405-424; perturbation-sim eigen.rs:93.

Council: premise auditor; scouts: code cartographer, internal prior art, literature, systems, concepts and ontology; co-architects: bridge, firewall and fit, falsifier. Session 01G22yT6.

## Corrigendum (2026-10-23, operator correction; append-only, nothing above is deleted)

**DECISION** (operator, 2026-10-23, relayed in EXO-50 GPT:gpt-ada-01/…-11 and in chat): the SPO/CE64/Thought-Git
substrate is not probabilistic. Probabilistic "spoFC" semantics are legacy.
Every probabilistic model is a **counterfactual** (Rung 3), and it must be certified by the cold
egglog / eggcc-inspired reasoning compiler before it counts.
- **SCOPE:** Thought-Git knowledge, truth and squash semantics, and head selection.
- **BASIS:** the premise audit above (ops are deterministic, no P), plus the operator's architecture: multi-lane exploration
  compiles into checked, reusable recipes.
- **REVISIT WHEN:** never on cost grounds.

What this withdraws from the map above:
- **The stochastic layer as a home for the Markov tools.** The "Answer" section called the exo129 snap
  kernel a legitimate stochastic layer. It is an artifact of lossy quantization: it is a legacy reading, not
  a substrate property. Markov / MCMC / lumpability (X3–X8, X11–X13, and PR-4, PR-6) remain at most
  specialized counterfactual research tools, and only where someone explicitly defines a stochastic process.
  They are not Thought-Git's organizing mathematics.
- **Head by vote, stationary mass or P̂.** X17's stationary mass was already SKIP. Also withdrawn are:
  - the chat-side proposal of a source-deduplicated majority vote;
  - "certificate is P̂" (X1).
  Agreement among lanes, at any count and with any root diversity, never confers proof authority. It
  never moves 59..63.
- **Head rule.** The head of a thought family is preferably a NEW proof-carrying synthesis. It must have
  lawfully admitted rules that preserve the children's required semantics, dependencies and replay
  evidence, and it must pass the exact D-TGA-5 gate (no ε, no majority relaxation of R1/R2).
  - If there is no proof, a candidate stays an unadmitted counterfactual. It is never squashed into a
    claimed truth.
  - A proof certifies a conditional law under stated axioms. Empirical, source and causal claims still
    need independent attestation.
  - Proof certificate, CE64 59..63 state and Thought-Git replay/commit witness are three separate objects.
- **Tactic naming error.** Chat-side text called #11 "ICR". #11 is CR (Contradiction Resolution) and
  #31 is ICR (Iterative Counterfactual Reasoning). MPC #27 / HKF #34 are candidate generators, not
  certified mergers.

What survives unchanged:
- squash = path composition, checked by commutation (X2, PR-2);
- receipt classes keyed without `thought`, guarded by roots (PR-5);
- PR-1 as a short confirmation that lanes are not a stochastic chain;
- the requested storno of the #1736 IDEAS "certificate is P̂" and "4ε tolerance" bullets.

PR-7 is superseded: the head-selection probe moves to the CC2 proof-discovery plan
(`entries/2026-10-23-cc2-proof-compiling-gap-map.md`, when it lands). Its F5 synthesis fixture and its
F6 correlated-copies fixture keep their shape, with ground truth from an independent oracle, never a vote.

## File note (2026-10-23)

This file was first committed on #1737 as `entries/2026-10-23-coresearch-thought-git-markov.md`. Session
01Vnc landed a separate coresearch map on the same question under that exact path via #1736. Both maps
are kept: theirs is unchanged at the original path, and this one moved here when main was merged into
#1737. The content above is unchanged.
