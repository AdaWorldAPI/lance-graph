## 2026-08-11 — E-THE-TRANSFORM-MUST-MATCH-THE-DISTRIBUTION-SHAPE-1

**Status:** FINDING `[H]` (measured on real ERA5, one timestep — re-runnable at
`probes/weather-p1/`; needs a second variable + season before `[G]`).

**Fisher-Z DEGRADES a weather-anomaly palette.** Real ERA5 `2m_temperature`,
1,038,240 gridpoints, both paths into the same 256 buckets over the same
0.4–99.6 percentile window: **linear MAE 0.0684 K, 0 empty buckets, 115.7
effective buckets** vs **Fisher-Z MAE 0.2168 K, 76 empty buckets, 28.1 effective
buckets** — 3.2× worse error, 228 of 256 buckets burned.

**Mechanism, measured on this sample:** `arctanh` is ≈identity near 0 and
explodes near ±1, so it moves resolution *toward the bounds*. The tested ERA5
`2m_temperature` anomaly (one timestep, 2021-06-15 12:00 UTC) has 77 % of mass
inside |s|<0.25 (**excess** kurtosis +3.30, mass in the MIDDLE); a
correlation-like control `tanh(N(0,1.5))` has 32.7 % beyond 0.9 (mass at the
BOUNDS). **CONJECTURE `[S]`, generalizing beyond the tested sample: the
transform must match the input distribution's shape.** The *measured* claim is
narrower — for THIS field at THIS timestep, Fisher-Z costs address economy.
Promotion to a general rule needs the second variable and season named below.

**Terms:** *effective buckets* = `exp(Shannon entropy)` of the 256-bin occupancy
histogram (how many addresses actually carry data); *drift score* = `max|occ−μ|/σ`
under a multinomial null, exactly as `crates/helix/src/quantize.rs:119-146`. Fisher-Z is right for correlation-like inputs and for
helix's own `r = √u` (equal-area placement concentrates toward the rim BY
CONSTRUCTION) — nothing about the helix crate is impugned. What is falsified is
generalizing its transform to bell-shaped geophysical fields, which is exactly
what `weather-normalized-substrate.md` §1.2 did. "One shared transform ⇒ one
comparable substrate" does not survive; the shared palette + LUT may.

Also measured: ARCO-ERA5 has **no `wind_speed` variable** (exhaustive, all 52
arrays), so §6.5's Jensen-gap measurement cannot have come from the Phase-A
store; blosc is variable-dependent — **1.794×** (`2m_temperature`), **1.771×**
(`2m_dewpoint_temperature`), **1.248×** (`10m_u_component_of_wind`), each one
chunk `[1,721,1440]` at t=547476, blosc/lz4 as stored — not one 1.27×;
the BF16 anomaly gain reproduces in DIRECTION only (74.9× under a zonal-mean
climatology proxy vs the doc's unreproduced 97×).

