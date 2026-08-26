## 2026-08-19 — E-XC21-HARNESS-CONFIRMS-C2-AND-FINDS-DEAD-CODE-1

**Status:** FINDING (measured), X-C2-1 landed.

**The harness** (`crates/rp-seal-t0-probe`, excluded, own workspace):
injections I1–I9 (I9 at all six cascade-congruent strides) over a
synthetic cycle, ground truth = the injection record, schemes S1U
(locus-unbound content digest — the shipped `ndarray::hpc::seal` shape)
and S6 (locus+version-bound — C2's recommendation, hash half). S2–S5
plug into the same `Scheme` trait in X-C2-3. Zero wall-clock anywhere
(T0.3 discipline).

**Anti-vacuity gate PASSED against REAL code, with one upgrade and one
downgrade of C2's record:**
- Control (a) reproduced on live `ndarray::hpc::merkle_tree`: corruption
  at meta words 50/70/200 → `hamming == 0`. The blind zone is WIDER than
  C2 recorded — words 48..56 and 64..96 are unhashed too, not just
  112..256 (BRANCH_REGIONS covers {0..48, 56..64, 96..112} only).
- Control (d) reproduced on live `ndarray::hpc::seal::Plane`: identical
  content at another slot verifies `Wisdom` against the stored root —
  the unbound-digest wrong-slot FA, exactly "certain for
  identical/default content".
- Controls (b)/(c) — firefly `verify_ecc` (every branch returns
  `Some`; literally cannot reject) and the container XOR parity — are
  **unreachable by construction: the `wip` feature they live behind
  fails to compile (104 errors, measured)**. M11's severity for B3/B2's
  wip halves downgrades from "live defect" to "dead code wearing a
  strong name"; the fault CLASSES are proven on the fold algebra
  (paired-flip cancellation + permutation invariance) as harness
  self-controls.

**The measured matrix confirms C2's truth table empirically at the hash
tier:** S1U false-accepts EXACTLY the substitution class — I4 wrong-slot,
I5 stale, I6 duplicate (1/1 each, the seal travelling with the chunk) —
and nothing else; S6 has ZERO false accepts and zero false alarms across
all of I1–I9; null control 10⁶ distinct clean chunks per scheme, 0
spurious; full-size 32 MiB pass green. The S6 kill floor (I1–I3 at
multiplicity 1) holds. Numbers dated in the probe README.

