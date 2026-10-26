# D-PAL256-ALG-1 — Algebraic Preservation Ledger for 256-entry palettes

## READ BY: anyone building or reviewing a palette/LUT lowering of softmax, attention,
##          activation tables, Fisher-z/cosine tables, int8 output codes, or a deskew
##          or gamma correction on a 256-entry codebook. family-codec-smith,
##          palette-engineer, certification-officer, truth-architect.

**Status:** WORKING-MODEL. Lab evidence only; nothing here is SHIPPED in production.
**Source of measurements:** `.claude/board/entries/2026-10-26-d-tflp-parity-audit.md`
Amendments 4, 5, 6 and 6.1; probes under `.claude/probes/fisherz-softmax-v1/` and
`.claude/probes/tanh-sweep-v1/`.

The question each entry answers: **which algebraic fact lets a computation over 8-bit
codes stay in code space, under which hypotheses, and when does it stop being true.**
Statuses: SHIPPED (in a production crate, tested), TESTED-LAB (measured in a lab probe),
PROPOSED (derived, not run), REFUTED (a probe contradicted it), UNKNOWN.

Notation: code q ∈ {0..255}; representative x(q); row of n codes q_1..q_n; temperature β > 0;
weight w_i = e^{β(x(q_i) − c)} for a constant c; S = Σ w_i; p_i = w_i / S; fused output
y = Σ p_i V_i.

---

## A. Affine softmax cancellation

- **Statement.** For `x(q) = a + s·q`, `p_i` depends only on `q_i − q_max` (and on βs);
  `a` cancels; one 256-entry difference table `T[d] = e^{−βs·d}` serves every row.
- **Derivation.** `e^{β(a + s q_i − c)}` with `c = a + s q_max`: `a` cancels inside the
  exponent; the factor `e^{−β(c − …)}` common to numerator and denominator cancels in `p_i`.
- **Hypotheses.** Affine representatives; one β per table; `s > 0` (monotone codes).
- **Exact / approximate.** Exact in real arithmetic. Approximate only through the table's
  fixed-point rounding (entry G).
- **Not applicable.** Nonuniform codebooks (use B); per-row β; a code-dependent offset.
- **Falsification.** Any row whose 40-digit reference p differs from the table arm beyond
  the G bound. Measured: none (Amendments 4/5; 261 cases).
- **Permitted lowering.** Difference table indexed by `q_max − q_i`, integer accumulation.
- **Evidence.** Amendment 4 E2-E5; Amendment 5 F2-F3 arm A.
- **Status.** TESTED-LAB.

## B. Nonuniform unary LUT

- **Statement.** For any monotone or nonmonotone codebook, `p_i = E[q_i] / Σ_j E[q_j]` with
  the **unary** table `E[q] = e^{β(x(q) − c)}` for any fixed c. A 256×256 pair table is never
  required.
- **Derivation.** Same cancellation as A, without needing affinity: c is common.
- **Hypotheses.** One β per table; c fixed at build time (e.g. the codebook maximum).
- **Exact / approximate.** Exact in real arithmetic, **including** rows whose quantized p lies
  exactly on a rounding midpoint (e.g. 512 tied maxima, p = 1/512, 256·p = ½): the identity
  gives the exact p, and the declared rounding rule then decides the code. Three things must
  be kept apart:
  1. the identity (exact, no exceptions for finite representatives and β > 0);
  2. a certified numerical approximation of it (entry G);
  3. an implementation's rounding behaviour.
  The measured failures are in 3, not 1: an f64 sum of equal weights that is not exact lands
  an ulp off the half and the code breaks the wrong way (Amendment 5 F3, arm U: 3,072 p-code
  mismatches, all on 512-tie rows). That is an implementation error, not a counterexample to
  the identity. In fixed point, a row whose maximum lies far below c keeps few significant
  bits; measured in Amendment 6.1 with the reference at the codebook maximum: 81 / 261 rows
  need escalation (13 with the row maximum), and some rows have every F24 weight round to 0.
- **⊘ Struck (Amendment 6.1):** "Not applicable when exact midpoint ties must round per the
  declared rule and the sum is not exact." That conflated the identity with an implementation.
- **Not applicable.** Per-row β; a representative that is not the admitted source value.
- **Falsification.** Tie rows (n·k tied maxima with 256·p an exact half) test the
  *implementation* of the identity; they cannot falsify the identity itself.
- **Permitted lowering.** Unary table + row-max re-reference (a pair table only as a cache
  of re-referenced rows), or a unary table with exact summation.
- **Evidence.** Amendment 5 F2 (correction of Amendment 4 E4.2/E4.3), F3.
- **Status.** TESTED-LAB.

## C. Binary log-odds and Fisher z

- **Statement.** Two-way softmax is the logistic of the difference:
  `p_1 = σ(β(x_1 − x_0))`, log-odds `= β(x_1 − x_0)`. Fisher's
  `z = atanh(c) = ½ log((1 + c)/(1 − c))` is half the log-odds of `(1 + c)/2`.
  So a cosine palette decoded through `tanh` and fed to a logistic is the composition
  `σ(β·tanh(z_1) − β·tanh(z_0))`; it does **not** collapse to a function of `z_1 − z_0`.
- **Derivation.** Direct algebra; the second statement is the definition of atanh.
- **Hypotheses.** Exactly two classes; |c| < 1.
- **Exact / approximate.** Exact identities. The f32 decode→encode roundtrip is not: measured
  non-finite on up to 58,564 of 65,536 code pairs (Amendment 4), and `FamilyGamma::encode`
  truncates (`as i8`), so `encode(decode(q)) ≠ q` for 59-77 of 256 codes per gamma.
- **Not applicable.** n > 2 (no single difference); a softmax over cosines is B, not C.
- **Falsification.** Exhaustive 256×256 code pairs against a 40-digit logistic.
- **Permitted lowering.** A 256×256 table of `σ(β(x(a) − x(b)))` codes, or a 511-entry
  difference table on affine codebooks. **Never** an implicit cosine → z → cosine roundtrip.
- **Evidence.** Amendment 4 E4, E5 (binary exhaustive, 3 gammas × 5 β).
- **Status.** Identities PROPOSED-derived (no probe needed); roundtrip failure TESTED-LAB.

## D. argmax is not softmax

- **Statement.** argmax depends only on order; for a strictly monotone codebook,
  `argmax_i x(q_i) = argmax_i q_i` and no exponential is needed. Softmax needs magnitudes.
  With certified logit intervals `[ℓ_j, u_j]`, the argmax is certified when
  `ℓ_k > max_{j≠k} u_j` (early exit).
- **Derivation.** Monotone maps preserve order; the interval condition is sufficient by
  transitivity.
- **Hypotheses.** Strict monotonicity for the code-space shortcut; ties need a declared
  tie-break rule (lowest index is TFLite's ArgMax convention; not verified here).
- **Exact / approximate.** Exact.
- **Not applicable.** Nonmonotone codebooks (sort by representative instead); top-k with
  probabilities; any consumer of p.
- **Falsification.** A monotone codebook whose code-space argmax differs from the
  representative argmax (impossible by construction; a test pins the tie rule).
- **Permitted lowering.** Integer max over codes; interval early exit over logits.
- **Evidence.** None measured.
- **Status.** PROPOSED.

## E. Gamma corrections — four different things share the word

| kind | effect on softmax | evidence |
|---|---|---|
| common additive offset (e.g. `EULER_GAMMA` added to every value) | cancels exactly | TESTED-LAB: 0.0 change at 40 digits (Amendment 5 F2) |
| code-dependent additive offset | changes the answer | TESTED-LAB: p moved 0.31 (Amendment 5 F2) |
| affine decode `z = z_min + (q + 127)/254 · z_range` (`FamilyGamma`) | `z_min` cancels; `z_range/254` multiplies β (it is a temperature) | derived |
| power-law / display gamma `x ↦ x^γ` | monotone, so argmax survives (D); softmax does not | derived |

- `EulerGammaOffset` is a **design-only** name: `ISSUES.md:3571-3612`, ndarray
  `.claude/knowledge/w3-w6-soa-aos-design.md:542-546` and
  `cognitive-distance-typing.md:23,27`. No Rust type exists. Helix uses the constant
  `EULER_GAMMA` in `sprite_replay.rs:179` as a residue term, not as a softmax correction.
- **Status.** Rows 1-2 TESTED-LAB; rows 3-4 PROPOSED-derived.

## F. Distribution-aware deskew

- **Statement.** For a fixed β, the representative `v_β(q) = (1/β) log E[e^{βX} | Q = q]`
  makes each table weight equal the bin's expected weight, so `E[S]` is exact per element.
  The conditional mean `E[X | Q = q]` minimizes squared error in x, not in weight, and is
  biased low for the denominator by Jensen.
- **Caveat.** An unbiased denominator is not an unbiased p:
  `E[W_i / Σ W] ≠ E[W_i] / Σ E[W]`.
- **Hypotheses.** The bin's conditional distribution has a finite MGF at β; β is fixed when
  the table is built (the table is β-specific); calibration and held-out data are from one
  distribution.
- **Exact / approximate.** Exact in expectation for S; approximate for p.
- **Not applicable.** Heavy tails in an open-ended top bin: no representative fixes it
  (TV 0.81-0.91 at n = 4096). That needs tail resolution in the codebook.
- **Falsification.** Held-out TV and weight-sum bias against the true-value softmax.
- **Permitted lowering.** Still one 256-entry table per (codebook, β) — free at run time;
  keyed by codebook version.
- **Evidence.** Amendment 5 F6: bounded skewed source, β = 4, n = 4096 — TV 0.0515 → 0.0215,
  weight-sum bias +5.6% → +0.13%.
- **Status.** TESTED-LAB.

## G. Fixed-point error bound (F24 table, exact integer accumulation)

- **Statement.** Let `w_i` be the exact weights scaled by 2^F and `W_i = round(w_i)` with
  `|w_i − W_i| ≤ ε_i = ½`. With `Ŝ = Σ W_i`, `E = Σ ε_i = n/2`, `ŷ = Σ W_i V_i / Ŝ`,
  `p̂_i = W_i / Ŝ`, and `Ŝ > E`:
  - `|y − ŷ| ≤ Σ ε_i |V_i − ŷ| / (Ŝ − E)`
  - `|p_i − p̂_i| ≤ (ε_i + p̂_i E) / (Ŝ − E)`
- **Derivation.** `Σ W_i (V_i − ŷ) = 0`, so `y − ŷ = Σ (w_i − W_i)(V_i − ŷ) / Σ w_i`, and
  `Σ w_i ≥ Ŝ − E`. For p: `p_i − p̂_i = [(w_i − W_i) Ŝ − W_i (Σw − Ŝ)] / (Ŝ Σ w)`.
- **Hypotheses.** Each table entry is certified to within ½ (checked by interval arithmetic
  at build time, Amendment 6); V exact; codebook identity matches the table (see negative
  control: without the identity check the bound certifies a wrong code).
- **Exact / approximate.** The bound is rigorous; whether it certifies is data-dependent.
  An exact rounding midpoint can never be certified by any bound of nonzero width.
- **Tighter per-entry errors (Amendment 6.1).** The bound holds with any certified
  `ε_i ≥ |2^F w_i − W_i|` and `E = Σ ε_i`, provided `Ŝ > E`:
  - **exact zero:** `x(q_i) = x_max` exactly gives `w_i = e^0 = 1` and `W_i = 2^F` with no
    rounding, so `ε_i = 0`. On a row where every element is at the maximum the whole bound is
    zero: the arm's own exact rational `p̂ = 1/n` is then certified, including on a midpoint,
    because a zero-width interval has the same code at both ends. This needs no table data:
    `ε_i` follows from the representative's equality with the row maximum.
  - **per entry:** `ε_i = ⌈dev_i · 2^G⌉ / 2^G` with `dev_i = max(|lo − W_i|, |hi − W_i|)` from
    the entry's certified interval (G = 8; `e_i ≤ 128` fits a u8; 256 bytes per table row).
  - a refinement that cannot be certified falls back to `ε_i = ½`.
- **Falsification.** A certified output whose code differs from the 40-digit reference
  ("miss"). Measured: 0 misses of 248,778 outputs in each of 7 arms (Amendment 6 G1;
  Amendment 6.1 H1, Python and Rust).
- **Permitted lowering.** Integer `S` (u64) and `N` (i128); round integer outputs from
  `(N, S)` by quotient and remainder; compare the bound against the nearest rounding
  boundaries by cross-multiplication with a 256-bit product (measured widths: S 37, N 101,
  bound numerator 123 bits; no overflow refusal); escalate uncertified rows (H).
- **Status.** TESTED-LAB (Python and the Rust lab reproduce each other's decisions exactly).

## H. FMA and escalation

- **Statement.** A fused multiply-add rounds once per step; it does not bound the
  accumulation error of a reduction. Escalation is per row and conditional: only rows with
  an uncertified output are recomputed, by a resolver that implements the declared rounding
  rule.
- **Resolver (Amendment 6).**
  1. One distinct representative in the row: exact rationals (`p = 1/n`, `y = mean V`).
  2. Otherwise, per output column, if every distinct-representative group has the same
     V-mean t, then `y = t` exactly (Lindemann-Weierstrass: `Σ c_j e^{r_j} = 0` with distinct
     rational `r_j` forces all `c_j = 0`). Without this check an exactly rational y on a
     midpoint never certifies — disable-verified.
  3. Otherwise y and p are irrational, never on a midpoint, and outward-rounded interval
     evaluation at 64…512 bits terminates; it returns only when both interval ends give
     one code, so the result equals the code of the exact value under any rounding rule.
     Exhaustion at 512 bits is UNRESOLVED (fail closed), never a guess.
- **Hypotheses (made explicit, Amendment 6.1).** The Lindemann–Weierstrass step holds when:
  β is finite, positive and a binary floating-point value read exactly; the representatives
  are finite binary floating-point values read exactly; every `V_i` is an exact rational;
  the weights are the real exponentials `e^{β(x_i − x_max)}`; and grouping is by the
  effective representative value, so distinct groups have distinct exponents. It does
  **not** extend to arbitrary real parameters, approximated representatives, or any
  operation that changes the exact exponential relationship. Under these hypotheses a
  normalized `p_i` is never rational when the row has two or more distinct exponents, and
  `y` is rational if and only if every group has the same V-mean. Outward rounding is
  required; the plain f64 recomputation of Amendment 5 arm C is **not** a certified resolver.
- **Row vs output classification.** A row is "escalated" when any of its outputs is
  uncertified; each output then carries its own resolver. In the two-representative
  midpoint control (group V-means 5/2) the p outputs are certified by the bound, and only y
  needs the equal-group-mean rule (code 3, correct). With that rule disabled, y is
  UNRESOLVED at 512 bits and no code is returned (Amendment 6.1 H3).
- **Exact / approximate.** Exact when it returns.
- **Permitted lowering.** Reuse ndarray `simd::mul_add_f64` (`ExactFma`) only where operands
  are real-valued; never on a build without hardware FMA in a hot loop (software path costs
  ~20×). No second FMA implementation.
- **Evidence.** Amendment 5 F1, F5; Amendment 6 G1.
- **Status.** TESTED-LAB (Python lab resolver; no Rust resolver exists).

## I. Exact-uniform row (Amendment 6.1)

- **Statement.** If every active element of a row has the same effective representative
  (`x(q_1) = … = x(q_n)`, compared as values after masking and NaN admission, not as code
  bytes), then `p_i = 1/n` and `y = (Σ V_i)/n`, exactly.
- **Derivation.** All exponents are 0, so every weight is 1.
- **Hypotheses.** Masked elements are removed first. Under a NaN-carrying code space
  (Sym + NaNAtMin) a NaN code in the active row is refused before this test. Two distinct
  codes with equal representatives count as uniform; equal codes are not needed.
- **Exact / approximate.** Exact. With exact `V_i`, the final code comes from the exact
  numerator and denominator, so exact midpoints (`n = 512`, `256·p = ½`) round by the declared
  rule with no ambiguity.
- **Not applicable.** Any row with two distinct representatives (use G, then H).
- **Falsification.** A shortcut output that differs from the reference (measured: 0 of 50,883).
- **Permitted lowering.** A value-equality test over the active row before any table read;
  no exponential, no bound, no f64.
- **Evidence.** Amendment 6.1 H1: 38 of 261 rows take it, and escalated rows fall from 13 to 3.
  The exact-zero bound (G) reaches the same 3 without the shortcut. The shortcut's value is
  that it skips the arithmetic, not that it certifies more.
- **Status.** TESTED-LAB.

## J. TANH int8 code→code table vs TFLite Micro (Amendment 6.1)

- **Statement.** For int8 TANH with the TFLite output contract (scale 1/128, zero point 0), the
  code→code table `code(c) = clamp(round_away(128 · tanh(x(c))))` is correctly rounded at every
  cell. It equals TFLite Micro's bytes except where `128·tanh(x)` lies within TFLM's internal
  fixed-point error of a rounding half.
- **Derivation.** For rational `x ≠ 0`, `tanh(x)` is transcendental, so no cell is an exact tie
  and TiesAway and TiesEven give the same table (the Codec8 plan's claim, now with a reason).
- **Hypotheses.** The input real value is `scale·(c − zp)` exactly (TFLite spec), or Codec8's f32
  decode of it. The two gave identical tables on all 569,344 swept cells.
- **Exact / approximate.** The table is exact (interval-certified at 64 bits for every cell).
  TFLM is approximate.
- **Counterexample (measured).** At `|x| ≈ 1/256`, `128·tanh(x) = 0.4999975`, 2.5e-6 below the
  half. The correct code is 0; TFLM gives ±1. Seen at every power-of-two scale where ±1/256
  is reachable (2^-8 … 2^-15, 15 cells) and at one random calibration (scale bits 0x3932159a,
  zp 91; distance 9.8e-6; 2 cells): 17 of 569,344 cells.
- **Permitted lowering.** A prepared 256-entry table. A "TFLM-parity" table is a different
  artifact and must be generated from TFLM's own arithmetic. The two must not share a receipt.
- **Status.** TESTED-LAB.
