## 2026-08-10 — E-JC-AND-NDARRAY-BOTH-SHIP-A-RELIABILITY-BATTERY-WITH-DIFFERENT-DEGENERATE-CONTRACTS-1

**Status:** FINDING `[G]` (source-verified 2026-08-10, both files read).

`pearson` / `spearman` / `cronbach_alpha` / `icc` exist in **BOTH**
`jc::reliability` (`crates/jc/src/reliability.rs`) and `ndarray::hpc::reliability`
(`ndarray/src/hpc/reliability.rs`) — with **different degenerate-input contracts**:

| | `jc` | `ndarray::hpc` |
|---|---|---|
| signature | `-> Option<f64>` | `-> f64` |
| degenerate (n<2 / zero variance) | `None` | **`0.0`** |
| icc | `icc(ratings, IccForm)` | `icc_a1(ratings)` |
| cronbach input | `&[Vec<f64>]` | `&[&[f64]]` |

**Why this matters and is not cosmetic:** ρ = 0.0 is *also a legitimate measured
value*. The ndarray form therefore cannot distinguish "no correlation" from
"undefined" — a zero-variance window (entirely possible in a real field: a constant
patch, a saturated code lane) silently enters an aggregate as a real 0.0 and drags
the mean down, where `jc` would have returned `None` and forced the caller to decide.
This is the same shape as the vacuous-assertion family: a value that cannot fail
loudly.

**Ruling:** `jc` is the authority (operator-named "the lance-graph JC crate"); the
ndarray copy is the SIMD-side mirror. **Every reliability number in the weather POC
is computed with `jc`.** Their agreement over identical non-degenerate inputs is
itself a probe (plan `weather-substrate-poc-v2.md`, D-WXB-4), paired with an
assertion that the degenerate case is *reported*, never folded.

Cross-ref: `.claude/plans/weather-substrate-poc-v2.md` §3; `jc` = "Jirak-Cartan:
five-pillar proof-in-code" (zero external deps; Pillar 11 `hambly_lyons` is
sigker-gated); `E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1`.

