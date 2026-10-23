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

## Results (2026-10-23, after the run; nothing above was changed)

`cargo test --release --test exo129_learned_lut_probe -- --nocapture`
(bgz-tensor, debug info off). Agreement with the truth at k = 1..6:

| setting | purity P | unobserved | L k=1 | L k=2 | C k=1 | C k=2 | L k=6 | C k=6 |
|---|---|---|---|---|---|---|---|---|
| W1 clustered | 0.471 | 0.199 | 0.242 | **0.017** | 0.579 | 0.029 | 0.005 | 0.005 |
| W2 uniform | 0.479 | 0.170 | 0.112 | **0.009** | 0.279 | 0.027 | 0.005 | 0.005 |
| W3 n=0 | 1.000 | 0.159 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3 n=500 | 1.000 | 0.159 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3 n=1500 | 1.000 | 0.159 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| W3 n=3000 | 0.544 | 0.340 | 0.263 | 0.145 | 0.231 | 0.260 | 0.041 | 0.248 |

- **Rule 1 (harness control):** holds. W3 with n = 0 reads 1.000 for both
  arms at every k.
- **Rule 4 (disables):** both hold.
  - D1 (shuffled table) reads ≤ 0.005 at k = 2 in every setting.
  - D2 (L := C) reproduces C exactly.
- **Rule 2 (torque): KILL.** L reads 0.017 / 0.009 at k = 2 in W1 / W2,
  against the 0.90 bar. The learned table is worse than `ComposeTable`
  there, even at k = 1.
- **Rule 3 (mechanism M): PASS by the registered rule**, Spearman ρ = 0.935,
  **but the evidence is thin.** Four of the six settings tie at P = 1.0 /
  agreement = 1.0, so the rank correlation rests on three distinct points.
  The unregistered magnitude prediction `agreement_L(k) ≈ P^k` fails badly:
  W1 k = 2 is 0.017 against P² = 0.222.
- **Pin:** both verdicts are asserted in the test. Disable run: inverting
  purity (`1 − P`) makes the M assertion fail, so the pin is real.

### What the numbers say (reading, not a ruling)

- **Purity < 1 means the palette state is not Markov for this operation.**
  The cell does not determine the next cell: members of one cell go to
  different cells. In W1/W2 the purity is about 0.47. No 256×256 table
  can compose there, learned or derived, because one byte of state does not
  carry the information the next hop needs. In Markov-chain terms this is a
  **lumpability** failure (Kemeny–Snell): the partition is not lumpable for
  the operation. That is a candidate name for trail T-CLOSURE. It is not
  adopted here.
- **Learning made W1/W2 worse at k = 1** (0.242 vs C 0.579). Training cells
  pool starts from population rows with states reached after several hops,
  and 5/6 of the observations are reached states. The modal result then
  favours the reached-state distribution, while the test starts from
  population rows. This is a real cost of the pre-registered design, and it
  is itself a non-lumpability symptom: where in the cell a state sits
  matters.
- **W3 n=3000 separates two error sources.** C stays flat at about 0.25 from
  k = 1 to 6. The table is the exact group law, so composition adds no
  error, and the loss comes from the start and end snaps alone, which
  depend on the fixed member noise. L decays (0.263 → 0.041), because its
  cells were filled from mis-snapped observations. A closed carrier
  therefore composes with its exact law even when snapping is noisy. A
  table learned from noisy snaps does not.

### Closeout

PR (this branch) | STATUS: measured | OUTCOME: a learned 256×256 table does
not compose where the codebook is not closed (KILL); purity < 1 marks those
cases, with thin rank evidence (M PASS on three distinct points) |
OPEN: whether a purity bar (for example P ≥ 0.99) predicts composition on
real Moore/DR pair data. That needs labelled data (GOLDSTAUB
G-2026-10-23-1).
