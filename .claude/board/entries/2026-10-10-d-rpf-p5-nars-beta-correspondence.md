# D-RPF-P5 — NARS ↔ Beta correspondence on the CE64 surface (2026-10-10)

**Status:** MEASURED. Tests only (`crates/causal-edge/tests/d_rpf_p5_beta.rs`).
No production semantics changed. G1 untouched.

The live revision law has horizon k = 1 (`w = c/(255−c)`, revision adds `w`;
`d_rpf_conf_0.rs` T1). Under Bernoulli evidence with prior Beta(k/2, k/2),
`e = c(f−½)+½ = (w⁺+k/2)/(w+k)`, so k = 1 is the Jeffreys prior Beta(½, ½).

| id | finding | pinned |
|---|---|---|
| E1 | The identity holds exactly for k = 1 over every code (max 3.3e-16); k = 2 differs by ≥ 1.15e-5; the f32 `expectation()` stays within 5.8e-8 | yes |
| E2 | A unit observation (w = 1 → c = 127.5) is not a code. Sequential revision of unit Bernoulli observations stops at **c = 244** (w ≈ 22.2) after 20 observations, in every one of 32 runs. From then on old evidence is forgotten as in a fixed window, while the conjugate posterior keeps narrowing: mean \|Δe\| < 0.002 at n = 8, > 0.05 at n = 256 (max 0.16) | yes |
| E2b | The stall code depends on observation strength: c_obs = 1 stalls at 75 (w 0.42), 64 at 236, 128 at 244, 240 at 252, 250 at 254 | 12-row table |
| E3 | Revision cannot tell duplicated from independent evidence. Four copies of one observation claim a posterior sd 2.04× too small; eight copies 3.48× | yes |
| E4 | A variance rebuilt from `(f, c)` moves 0.3 % per code at c = 128 and 29 % at c = 253 | yes |
| E5 | Revision of finite codes never reaches 255: the maximum, (254, 254), is 254.499 → 254. The singular cell is entered only by writing 255 directly | yes |

Disable run: production confidence rounding changed from `round` to `ceil` →
E2, E2b and E5 red.

## What this means for the lab and for G1

- **The u8 confidence code is an evidence horizon, not a counter.** Past the
  stall, NARS-on-CE64 behaves like an exponentially weighted estimate with a
  window set by observation strength, not like a conjugate Beta update. That is
  a usable forgetting behaviour (it adapts after a change), but it is an
  implicit one: no parameter names it.
- Any lab strategy that reads `(f, c)` as a Beta posterior is valid only below
  the stall; past it, intervals must come from an explicit evidence count kept
  elsewhere (e.g. a PowerSums fold), not from the code.
- E3 is why contradiction and pooling need provenance: the operands carry none.
- G1: E5 shows that (255, 255) is unreachable by accumulation, so the G1 policy
  only governs values written directly.

```
PR (this) | STATUS: measured | OUTCOME: e is the Jeffreys posterior mean
exactly; u8 confidence stalls (unit evidence at 244 after 20 obs) so CE64
revision is a fixed-window estimate past the stall; duplicates inflate
precision; 255 unreachable by revision | OPEN: G1; whether the lab keeps a
separate evidence count; decay is not an operation on CE64 truth (not tested)
```

MIRROR
- BIAS CHECK: the brief framed P5 as "is the correspondence exact"; the
  identity was never in doubt. The stall (E2) is the result that matters, and
  it showed up only because the probe ran long sequences rather than single
  revisions.
