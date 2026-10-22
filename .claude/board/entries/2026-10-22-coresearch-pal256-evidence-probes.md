# 2026-10-22 — /coresearch: Palette256 evidence and plasticity probes (#1706–#1722)

**Status:** EXPLORATION MAP. No idea has been adopted; the operator chooses. Scope:
- #1706: Ascent BYODS → DR cold-compiler plan.
- #1707/#1708: DR associativity probe and restack, both merged 2026-10-22.
- #1709–#1717 (set C) and #1718–#1722 (set D): Palette256 / Fisher-z / NARS / Bayes / Hebb / BNN lab probes.

## Question (after the premise gate)

The original question was: "does tanh/Fisher-z encoding over Palette256, together with Bayesian, NARS and Hebbian updates, resist hallucination (confident wrong output)?" It attached one property to four mechanisms with different signatures. The premise audit split it into four:

- **Q-frame:** does the immutable Fisher-z i8 table keep rank, margins and composition intact under quantization and per-hop re-snapping? This is codec fidelity.
- **Q-evidence:** does the evidence state count each independent source exactly once? Where should provenance distinctness live?
- **Q-motion:** what may Hebbian/MRU strength and salience write?
- **Q-decision:** which answer/abstain rule makes confident-wrong output fail, judged by output calibration?

The map had to keep the exit option "the property lives elsewhere" open, and that is where it lands.

## Answer

The encoding prevents nothing. Fisher-z/tanh preserves rank (F2) and bounds the readout. A bounded or saturated readout carries no OOD signal (Hein et al. 1812.05720, Thm 3.1), and EXO-50 already ruled that a bounded readout is not a guarantee.

What could prevent confident-wrong output is a gate in front of the decision. It answers only when four conditions hold:

1. **Evidence is counted once.** Revision, including Hebbian revision under D10, refuses to pool beliefs whose stamps overlap. On overlap it chooses the higher-confidence belief (the ONA pattern). This is E2/M1.
2. **Confidence is certified only on a tree.** A cyclic hop path may rank but may not raise confidence: loopy BP counts returning evidence again (Circular BP 2403.12106). This is E5.
3. **The input is in support.** Use distance to the nearest centroid against the distortion radius, not readout magnitude. This is D3 (with the F3 clamp counter).
4. **The margin holds.** The top answer's lower bound beats the runner-up's upper bound. This is D1.

The judge is P-D4. Score the answer/abstain output with Brier and log score (strictly proper; Gneiting–Raftery 2007) on held-out data split by block. Never judge by margin size or skill-score ratios. I-NOISE-FLOOR-JIRAK rules out iid splits and iid bounds; that includes SGR's binomial bound, plain conformal, and Fisher 1/(n−3).

## Crosswalk and verdicts

| # | idea | relation | firewall | verdict |
|---|---|---|---|---|
| F1 | ComposeTable re-snap per hop. #1721 measured a deterministic 3-hop counterexample on a non-closed codebook against Base17 XOR; a closed codebook stays exact for 64 hops. | PARTIAL | PASS | **PROBE** P-F1 (asymmetric arm with an exact query side, ADC-style) |
| F2 | Fisher-z encode is monotone within one family, so argmax works on raw i8 | ALREADY-HAVE | PASS | SKIP (have) |
| F3 | `atanh_clamp`/i8 clamp clamp silently (`fisher_z.rs:67,198`); no counter | NEW | CONFLICT (seam: sidecar counter lane, not the codebook) | **PROBE**, bundled into P-D3 |
| F4 | Immutable reference / needle vs pairwise | ALREADY-HAVE (E-PALETTE256-IS-A-NEEDLE) | PASS | SKIP; delete #1709's `needle==1` tautology |
| F5 | Typed tanh/atanh law admission (#1717, local enum only) | PARTIAL | CONFLICT (seam: DRMC law-admission stage, cold) | PARK until DRMC has a second law |
| E1 | NARS ↔ Jeffreys-Beta; sample size kept beside CE64 | ALREADY-HAVE (D-RPF-P5) | PASS | SKIP; #1712/#1720 re-derive it |
| E2 | Revision has no stamp, so self-revision raises c (128→137; replay 200→237) | PARTIAL (D-HL-3, X8, X10 pre-registered) | CONFLICT (seam: stamp in an evidence-state lane, G5; never CE64 bits) | **PROBE** P-E2; fix home is the owner's call |
| E3 | Source-disjoint merge. A set stamp cannot tell shared-premise paths from disjoint ones (Green N[X] support). | PARTIAL | PASS | Fold into D-HL-3 |
| E4 | Typed Fisher 3σ gate | CONFLICTS-ANCHOR (I-NOISE-FLOOR-JIRAK) | TRAP as a gate | SKIP as a gate; keep only the "no n ⇒ refuse" test |
| E5 | Per-hop atanh·tanh = binary BP message; cycles double count | NEW (framing) | PASS | **PROBE** P-E5 (3-cycle vs tree) |
| E6 | BNN meaning (#1714 = binarized, local only) | PARTIAL | PASS | SKIP; re-test against ndarray `hpc/bnn.rs` if needed |
| M1 | Damped Hebb dz = η·s·p·(1−tanh\|z\|) | CONFLICTS the brief premise. D10 rules that Hebb is SPOFC NARS revision, so Hebb inherits E2. | CONFLICT (seam: write only through stamped revision; integer/LUT, no f32 tanh per edge) | **PROBE** P-M1; saturation policy is the owner's call |
| M2 | Salience weighting; rare-signal recall | ALREADY-HAVE / PARTIAL | PASS | PROBE P-M2 (cheap, add to #1718) |
| D1 | Interval-overlap abstain (local in #1711/#1719). `Resolution::from_ranked` never abstains on a weak top answer. | PARTIAL | PASS (margin) / CONFLICT (iid θ bound) | **PROBE** P-D1, θ fitted on a block hold-out |
| D2 | Conformal abstain sets | NEW | CONFLICT (exchangeability; seam: block or weighted variant, cold) | PARK until P-D4 has data |
| D3 | OOD from distance to support, not magnitude | PARTIAL (`palette.rs` radii) | PASS | **PROBE** P-D3 |
| D4 | Strictly proper scoring as the judge | PARTIAL (D-SELF-CALIBRATING-LAB-0 has ECE/Brier) | PASS (offline) | **PROBE** P-D4, the deciding one |
| D5 | Never renormalize away conflict mass (Zadeh) | PARTIAL (free_energy keeps both) | PASS | ADOPT-NOW: one regression pin |
| O1 | #1706: Ascent logical tuple → DR fold, cold | PARTIAL (SKIP as hot executor, 2026-10-21) | CONFLICT (tuple ruling; fork policy) | PROBE: A1 audit of `ascent::internal` first |
| O2 | Associativity certificate (merged). Pairwise u8 revision associativity (R6) is unmeasured. | ALREADY-HAVE + NEW | PASS | **PROBE** P-R6 (gates G3) |

## Pre-registered probes (top three first)

- **P-D4 (decision, deciding).** Data: held-out labelled outcomes split by contiguous source block. Arms: (a) always answer, (b) the four-condition gate, (c) a shuffled-label null. Scores: Brier and log score, with abstain scored as the base rate, plus selective risk at matched coverage. Kill: (b) fails to beat (a) by ≥ 0.02 Brier with a block-bootstrap 95% CI that excludes 0. Silent arm: (b) must not beat (c).
- **P-E2 (stamp).** One premise is revised with itself k ∈ {1,2,4,8} times; compare against k disjoint sources. Kill: duplicates raise c above the single-source code. Silent arm: disjoint sources still raise c. Pass: duplicate sd ratio within 1.00 ± 0.05 (today 2.04×). Lives beside `crates/causal-edge/tests/d_rpf_p5_beta.rs`.
- **P-D3 (OOD).** Inputs: in-support held-out data, far-field data at ×4, and saturating inputs. Measure the AUROC of the distance flag against |readout|. Kill: distance AUROC < 0.90, or magnitude AUROC within 0.05 of it. Silent arm: FPR ≤ 5% in support, and the clamp counter reads 0 in range and > 0 in the saturating set. Per-lookup flag must stay at ns scale, via `ndarray::simd` only.
- P-F1: drift onset over 64 hops on 10⁴ random chains, symmetric vs asymmetric arm. Silent arm: the closed codebook stays exact.
- P-E5: a Hebb/revision loop around a 3-cycle vs a tree with the same leaves. Kill: the cycle's c exceeds the tree's by ≥ 1 code.
- P-M1: repeat the same co-fire n = 1..1000. Kill: c rises beyond the single-event code. Silent arm: distinct co-firings raise c.
- P-M2: plant a rare axis-16 signal under loud axis-0 energy. Kill: recall@10 < 0.8.
- P-D1: θ fitted on a block hold-out. Kill: selective risk is not below full-coverage risk.
- P-R6: 10⁶ sampled u8 triples, rev(rev(a,b),c) vs rev(a,rev(b,c)). Kill G3: > 0.1% of triples differ by > 1 code.

## Dedup — as applied (2026-10-22)

The deletion plan below was superseded the same day by the operator-relayed
GPT proposal: set C (#1709–#1717) stays the canonical series with its
board entries, and each set-D test moves into its set-C counterpart
(#1718→#1710, #1719→#1711, #1720→#1712, #1721→#1715, #1722→#1713). The D
PRs close as superseded once their C counterpart lands. Both levels survive:
the documented experiment contract and the production-surface counter-probe.

### Original recommendation (kept for the record)

Keep these, because they test production surfaces:
- #1721 (ComposeTable counterexample)
- #1720 (CE64)
- #1722 (`EpisodicEdges64`)
- #1718 (`WeightPalette`, `l1_weighted`)
- #1711 (real encoding monotonicity)
- #1709, minus its tautology

Delete or shrink these, because each is local-only and superseded:
- #1715 (superseded by #1721)
- #1713 (superseded by #1722)
- #1710 (superseded by #1718)
- #1719 (hand-set entries; its abstain helper duplicates #1711's)
- #1714 (misnamed)
- #1716: keep only the refusal test
- #1717: move into DRMC

#1712's provenance idea is the one non-redundant item in set C. It goes to D-HL-3.

## Not searched

- PQ/codebook error accumulation under repeated re-quantization: not sourced first-hand within budget.
- Oja/BCM and Fisher (1915/1921) primary sources: secondhand only.
- Ascent `ascent::internal` and its macro expansion: not read; this is the O1 gate.
- The OOPSLA'23 BYODS paper: abstract only.
- STAMP_SIZE in ONA's Stamp.h: not read.

"Nothing found" for multi-hop ComposeTable drift on the board means no prior measurement exists. It does not mean none can exist.

## Sources (read grade)

- Geifman & El-Yaniv 1705.08500 (RF)
- Guo et al. 1706.04599 (SR)
- Hein et al. 1812.05720 (SR)
- Angelopoulos & Bates 2107.07511 (SR)
- Bouttier et al. 2403.12106 (SR)
- Murphy/Weiss/Jordan 1301.6725 (RF)
- Jøsang JAIF 2018 (SR)
- Green/Karvounarakis/Tannen PODS'07 (SR)
- Goertzel 2412.19524 (RF)
- Gneiting & Raftery JASA 2007 (SR)
- ONA `Stamp.c`/`Inference.c` (SR, MIT)
- FAISS wiki (SR, MIT)
- Lucene BM25 javadoc (SR, Apache-2.0)
- MAPIE README (SR, BSD-3)
- docs.rs/ascent (SR, MIT)

Inside sources:
- `.claude/board/entries/2026-10-10-d-rpf-p5-nars-beta-correspondence.md`
- `2026-10-08-coresearch-ce64-moore-masking-wiring.md`
- `2026-10-04-coresearch-evidence-stance-dependence.md`
- `2026-08-26-e-palette256-is-a-needle-the-colon-is-the-distribution-1.md`
- `2026-10-10-d-self-calibrating-lab-0.md`
- `2026-10-17-ce64-one-layout-moore-synapse-membrane.md` (D10)

Closeout: `STATUS: open | OUTCOME: map landed; property located in evidence accounting + decision gate, not the encoding | OPEN: operator picks probes; stamp home G5; Hebb saturation policy under D10`
