# W1 probes — hexagon-plasticity-v1 (D-HXP-2)

The instruments and results behind plan §11 and §11a, committed so the numbers can be
**read and checked** rather than taken on trust (Codex P2 on PR #1233).

## ⚠ Auditable, NOT re-runnable

**These cannot be re-run from this repository.** Every measurement depends on the ~10 GB
`r2harvest` ore corpus and the `ore-full-v2` build, which lived in an ephemeral session
scratchpad, are not committable, and have no immutable revision to cite. What is committed is
the **instrument** and the **result**; the **input** is gone. Stated plainly rather than
implying a reproducibility this repo cannot offer.

They are also a **pure-Python shape proxy** — no `numpy`, no `sklearn`, and no Rust
fingerprints. Nothing here measures `ndarray`, `bgz17` or `helix`.

## Files

| file | what it is |
|---|---|
| `w1_cue.py` / `w1-cue.json` | **Run 1.** Kept as the record of three apparatus defects, all spec errors: an inert LUT, a spread radius (1–2) smaller than the cell spacing (~3.4) so it diffused into empty space, and an order falsifier applied to an order-blind representation. |
| `w1_cue2.py` / `w1-cue2.json` | **Run 2**, apparatus corrected. The 12-row matrix, the LUT scorer, F1 on all three scorers, the transitions arm. |
| `seriation_quality.py` / `w1-seriation.json` | Resolves the "adjacency hurts" confound. Adjacent-pair Fisher-z **0.6352** vs a 20-shuffle null **0.0794 ± 0.0247** (**22.49 σ**); top-5 recovery within ±3 **41.84 %** vs **7.79 %** chance; **0/2850** degenerate pairs. The ordering is sound, so the null is about the mechanism. |
| `transitions_permute.py` / `w1-transitions-permute.json` | **The falsification.** Baseline 0.1579 · cells permuted 0.1654 · **bare integer type-IDs 0.1729** · identity+order-shuffle 0.0902. 7827 types over 1305 cells = 6:1 lossy hash. The palette cue contributes nothing beyond being a consistent label. |

## Reading them

`ARM0` in `w1_cue*.py` is the positive control: it must reproduce the BPE incumbent at
**r@10 0.1805** (`REPORT2` A2a-f records 0.180), and it did, byte-identically across both runs.
Any cue number from a run whose ARM0 moved would be a claim about the harness instead.
