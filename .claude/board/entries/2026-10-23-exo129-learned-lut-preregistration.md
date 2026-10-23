# 2026-10-23 — EXO-129: does a LUT learned from observed transitions compose? (pre-registration)

**Status:** PRE-REGISTERED (written and committed before any probe code)
**Thread:** TORQUE T-2026-10-23-1, Linear EXO-129. Crumb trail T-CLOSURE (EXO-132).
**Owner:** CC:session_01G22yT6htkcdyXsihxxXdrg

## Question

P-S2 (#1726) showed that `ComposeTable` does not survive two hops. It snaps
`entry[a] ⊕ entry[b]` back to the nearest entry, and most bound pairs land
outside the codebook. This probe asks two things:

1. Does a 256×256 table **learned from observed one-hop transitions**
   (`PaletteLut`-shaped, cell = modal next state) compose over k hops where
   `ComposeTable` does not?
2. **Mechanism M:** does a data-only statistic, the **purity** `P` of the
   learned table (the share of training observations that equal their cell's
   modal result), predict how well it composes? This is the testable form
   of T-CLOSURE's "is the carrier closed under the operation".

## Setup

- **Operation per hop, with key index `b`:** `x ← x ⊕ entry[b]` in Base17
  (`xor_bind`), as in P-S2.
- **Truth for a chain** `(x0, b1..bk)`: `assign(x0 ⊕ e_b1 ⊕ … ⊕ e_bk)`, one
  snap at the end. `x0` is a population row, not a palette entry.
- **Arms:**
  - **C:** the `ComposeTable` chain from `s0 = assign(x0)`.
  - **L:** the learned-table chain from `s0`.
- **Training:** 20 000 trajectories of 6 hops from population rows (seed A).
  Each hop records `(assign(x_t), b, assign(x_{t+1}))`. `L[a][b]` is the
  modal result. Unobserved cells fall back to `ComposeTable`, and the
  unobserved fraction is reported.
- **Test:** 2 000 fresh chains (seed B) for each k = 1..6.
- **Worlds (six settings):**
  - **W1 clustered** and **W2 uniform:** P-S2's generators and seed,
    palette from `WeightPalette::build`. These are inherited from P-S2, not
    built for this probe.
  - **W3 group-closed:** 256 palette entries built directly as an XOR group.
    `entry[g]` has `dims[i] = 4096` for each set bit `i < 8` of `g`, so
    `entry[a] ⊕ entry[b] = entry[a ^ b]` exactly. Population: 4 members per
    element, each `entry[g] + u`, with `u` non-negative per-dim noise in
    `[0, n]` (non-negative so that XOR never touches the sign bit).
    Noise levels: `n ∈ {0, 500, 1500, 3000}`, four settings. Larger noise
    lets XOR of noise reach bit 12 and break closure gradually.

## Measures

- `agreement_L(k)` and `agreement_C(k)` for k = 1..6: the share of test
  chains whose arm result equals the truth.
- `P` (purity of L on its training observations) and the unobserved-cell
  fraction.

## Decision rules (fixed now)

1. **Harness control:** in W3 with n = 0, both L and C read ≥ 0.99 at every
   k. Otherwise the harness is broken and nothing below is read.
2. **EXO-129 torque card:**
   - **PROMOTE** if L ≥ 0.90 at k = 2 in **both** W1 and W2.
   - **KILL** if L < 0.90 at k = 2 in both.
   - Mixed: report, and the row stays TORQUE with the next probe named.
3. **Mechanism M:** passes if, over the six settings, the Spearman rank
   correlation between `P` and `agreement_L(2)` is ≥ 0.9. The prediction
   `agreement_L(k) ≈ P^k` (independent per-hop errors) is reported with its
   residual, but it does not gate.
4. **Disables (both must behave, or no claim is made):**
   - **D1:** shuffle L's result bytes across cells. Agreement at k = 2 must
     drop to ≤ 0.10 in every setting.
   - **D2:** set L := C. This must reproduce C's numbers exactly.

## Applying GOLDSTAUB G-2026-10-23-1 to this probe

The world that passes by construction is W3 with n = 0. That is why it is
only a harness control. The world expected to fail is W2, which has no
structure for a table to learn. So no conclusion is drawn from a single
world's pass or fail:

- the **mechanism** verdict comes only from M across the six settings;
- the **torque** verdict comes from W1/W2, which P-S2 fixed before this
  probe existed.
