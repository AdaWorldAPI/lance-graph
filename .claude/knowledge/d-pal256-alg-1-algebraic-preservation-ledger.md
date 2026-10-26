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

## K. Source-behavior preservation by prepared transcoding (Amendment 6.2)

- **K1, mathematical table.** For a finite input code space `C_in` (|C_in| = 256 for 8-bit codes),
  source decode `D_in,θ`, mathematical unary `f` and declared output quantization `Q_out,θ`
  (rounding, saturation):

  `L_math,θ[q] = Q_out,θ( f( D_in,θ(q) ) )`

  The table is prepared with intervals or exact rationals. The result is a code.
- **K2, source table.** For a specific source kernel with observable output `K_src,θ(q)`:

  `L_src,θ[q] = K_src,θ(q)`

  It stores the kernel's bytes, deviations included. It requires a pinned, specified kernel
  (framework, kernel tree, operator, version, backend, calibration), not a correctly rounded one.
- **K3, numerical fingerprint.**

  `Δ_θ[q] = L_src,θ[q] − L_math,θ[q]`

  Computed in a wider signed integer. Δ records where the source differs from correctly rounded
  behaviour; by itself it does not say why. The cause is established only by tracing the kernel.
- **K4, finite-domain equivalence (proof).** Let `K_src,θ` be pure and deterministic over
  `C_in` under the pinned conditions. If `L[q] = K_src,θ(q)` for all 256 q (exhaustive
  comparison), then `L` and `K_src,θ` are the same function on `C_in`, by extensionality of
  functions on a finite set.
  - Lookup `y = L[q]` then reproduces the kernel's output for every admissible input, with no
    source runtime.
  - **Restrictions:** this is equivalence for that θ, that kernel revision, that backend and
    that code space only. It is not equivalence for other calibrations, other kernel versions
    (e.g. CMSIS-NN, xtensa or full TFLite), other shapes or other hardware. Sampling is not
    exhaustive: measured, 8 sampled codes agreed on a calibration where the full 256 do not.
- **K5, composition.** Let `K_1: C_0 → C_1` and `K_2: C_1 → C_2` be pure, finite, with
  `C_1` interpreted by `K_2` exactly as `K_1` produces it (same calibration, same code space).
  Then

  `L_fused[q] = K_2(K_1(q))`

  is a single unary table over `C_0`. The proof is the same as K4: on a finite domain, the
  composition of two functions is a function.
  - The intermediate rounding is not removed. It is executed once, during preparation, and its
    effect lives in the stored codes.
  - By induction this holds for any finite chain `K_n ∘ … ∘ K_1` under the same conditions.
  - It is NOT palette-chain closure in the EXO-129/132 sense: it is the exact composition of
    one admitted finite function, not a law about composing palette semantics.
- **Not admitted (refuse, or use native algebra):** MatMul/GEMM, attention reductions, softmax
  over arbitrary-length rows (a unary table suffices for the weights, but normalization is
  row-dependent; entries B and G), LayerNorm, dynamic normalization, masks that change the
  domain, shape-dependent, stateful or value-indexed operators, dynamic calibration, unbound
  external operands. A binary operator may admit a 256×256 table; that is a separate admission.
- **Exact vs approximate.** `L_src` is exact with respect to the source (byte parity). `L_math`
  is exact with respect to its declared mathematical contract. Neither stands in for the other:
  they need distinct semantic receipts even when their bytes coincide (2,215 of 2,224 measured
  calibrations coincide, 9 do not).
- **Falsification.**
  - Exhaustive 256-code comparison against the pinned kernel.
  - An independent mathematical reference for `L_math`.
  - Controls: an omitted intermediate quantization, a missing stage, a wrong intermediate
    calibration, a wrong receipt, a corrupted byte.
- **Permitted lowering.** `out = table[code + 128]`, one prepared 256-byte table per (contract,
  θ, receipt). A sparse Δ patch is a cold-storage option only, reconstructed and digest-checked
  at load. It is never a per-element runtime stage.
- **Evidence (Amendment 6.2 K1-K5, `.claude/probes/source-behavior-v1/`).**
  - TFLM int8 TANH: SourceParity = kernel bytes on all 569,344 cells. CertifiedMath = an
    independent mp.tanh reference on all cells. 17 cells differ (Δ ∈ {−1, +1}).
  - All 17 differences trace to gemmlowp's fixed-point `exp_on_negative_values` (the kernel's exp
    value undershoots by 355.7 Q0.31 units). No float roundtrip is on the path.
  - TANH → LOGISTIC fusion: the fused table equals the two-kernel sequence on all 569,344 cells.
  - The continuous mathematical composition differs from the sequence on 69,652 cells.
- **Status.** TESTED-LAB (TFLM reference kernels, x86_64 gcc, kernel tree `8f6596d`).

## L. Source causality and row-dependent source parity (Amendment 6.3)

- **L1, causal partition of a fused fingerprint.** For a two-stage chain with source and certified
  tables per stage, the four arms SS, SC, CS, CC (every arm keeping the intermediate code) give the
  partition. A cell of SS ≠ CC is attributed to one stage only if correcting that stage alone
  reaches CC and the other stage agrees on both paths; otherwise it is conjunctive. Fingerprint
  counts per stage are not additive in general. Measured on TANH → LOGISTIC (569,344 cells):
  48,789 LOGISTIC-only, 17 conjunctive, 0 TANH-only; SC ≡ CC.
- **L2, shared source cause.** TFLM int8 TANH and LOGISTIC both evaluate gemmlowp
  `exp_on_negative_values` at −1/128 and get the same integer (2,130,771,442, low by 355.67 Q0.31
  units, of which −355.97 is degree-4 Taylor truncation). In both traced cases this truncation, not
  division or output rounding, decides the byte. Measured at one argument; not a general claim.
- **L3, row-dependent source parity (softmax).** Softmax is not admitted as a unary table (entry K).
  It admits a prepared recipe: per-calibration weight tables taken from the source's own
  primitives (exp, `Rescale<12>`, admitted flag), then per row an exact integer sum, the source
  reciprocal and the source output rounding. Domain: Q12 sum ≤ INT32_MAX and exponent ∈ [0, 31],
  i.e. Σ exp < 512; outside it, a typed refusal, never a byte claim. Measured on TFLM int8
  (840 rows, 14 calibrations): 0 differing cells on 337,878 admitted outputs, 111/111 aborting
  rows refused. The source differs from the mathematical contract on 3 cells, caused by the
  accumulated Q12 rounding; F24 matches the contract, so the two remain separate receipts.
- **Status.** TESTED-LAB (TFLM reference kernels, x86_64 gcc, kernel tree `8f6596d`). The
  reciprocal stage has no binding falsifier yet (0 deciding rows in 1.8 M searched).
- **L4, source parity is per artifact (Amendment 6.3 addendum M2).** Observational equivalence
  `P_θ(q) = K_{s,b,θ}(q)` is established per build b. Six x86 builds of the pinned TFLM reference
  kernels (gcc −O0/−O2/−O3 native/no-vectorize, clang 18, UBSan) give identical bytes on the
  tested corpus, two of them with compiler-generated vector code. That is evidence for those
  builds only. Rows that reach undefined behaviour (the softmax int32 sum past 2^31; a 32-bit
  shift by 32) are `UNDEFINED_OR_UNSUPPORTED` and refused, never pinned as portable bytes.
- **L5, a stage is removable only if it is unobservable on the admitted domain (M4, M6).** The
  softmax exp stage is removable by preparation: its argument depends only on d, so the prepared
  weights are the source values (`INTERNAL_EQUIVALENT`). The source reciprocal is not removable:
  an exhaustive superset over every admitted (S, weight) finds 601 byte-changing pairs, and 529
  are kernel-confirmed reachable rows (shortest: 4 elements) in 13 of 14 calibrations; one
  calibration has none. **Correction to L3:** "no binding falsifier yet" is superseded; the
  reciprocal stage now has 529.
- **L6, error cost is per code after quantization (M5).** Measured on LOGISTIC (1/128, 0): an
  exp error on 214 of 256 codes reaches the output byte on 2. Those 2 entries reach 48,806 fused
  cells through fan-in. An adaptive-precision decision weighs crossing probability × fan-in, not
  internal error magnitude.

- **K7, a byte's meaning belongs to the contract, not to storage or execution (Amendment 6.4,
  extends entry K).** Three things are separate for an 8-bit code q: its storage pattern
  (raw i8 byte, or Codec8 `U8` biased `q+128`), its value under the tensor contract
  (`scale·(q − zp)`, or NaN under a `Sym+NaNAtMin` code space), and the instruction that
  executes it (VNNI/AMX are unsigned×signed, so ndarray shifts one operand by 128 and subtracts
  `128·colsum`). SourceParity holds when the execution recovers the contract value exactly:
  - measured `ndarray scalar = ndarray SIMD = TFLM` on every FC cell (AMX, AVX-512 VNNI,
    AVX-VNNI, scalar; NEON and WASM simd128 dot products), and on 2,009,224 fixed-point cases
    (SRDHM, RoundingDivideByPOT, MultiplyByQuantizedMultiplier) and 219,800 requantize tails;
  - the bias correction is exact under wrapping i32 arithmetic, so the precondition is that the
    TRUE accumulator fits i32, not that each intermediate does. In release the SIMD tiers wrap
    and stay exact; debug builds panic on in-range inputs (`amx_matmul.rs:1028`).
  - The source's fingerprint relative to exact math is integer rounding: double rounding
    (SRDHM then RoundingDivideByPOT) and round-half-away at negative half ties; multiplier
    quantization contributed 0 cells. That fingerprint is source behaviour and is preserved,
    never "fixed".
  - −128 under a symmetric weight contract is a nonconforming input. TFLM and ndarray compute it
    identically; a checked decode refuses it. It is never clamped to −127, and an asymmetric
    −128 is never read as NaN — the latter is not yet enforced at the binding (Amendment 6.4 N8.1).
- **Status.** TESTED-LAB (`.claude/probes/int8-semantics-v1/`, lab commit `383e9c07c`).
- **L7, parity has an admission domain, and output equivalence needs three levels (Amendment 6.4
  follow-up N11–N13).**
  - **Admission.** A source byte belongs to the SourceParity contract only on SUPPORTED inputs: no C++
    undefined operation is reachable and no source guard fires. For TFLM int8 softmax that is exactly
    2^19 ≤ S < 2^28 (derivation over the source text). Rows outside it are UNSUPPORTED (a guard aborts) or
    UNDEFINED (signed overflow from 4,096 maxima, a shift by 32 from 8,192; both bounds proved minimal).
    Agreement of several builds on an UNDEFINED row is an artifact observation and never extends parity.
  - **Three levels.** For a substitution of one stage (here the reciprocal), an arithmetic superset
    candidate is not yet a witness. It must be a reachable pair (a SUPPORTED row exists with that sum and
    that element) and then a witness row (the real kernel reproduces the source byte, and the row is
    re-checked to have that sum). Exhaustive absence over the superset proves output equivalence only if
    the superset covers every admitted case and the tested output computation is the source's; both hold
    for softmax (N12), so calibration 9 is PROVED equivalent and 13 calibrations have 529 reachable
    witnesses, with the other 72 candidates PROVED unreachable.
  - **Boundary pruning.** The byte is a monotone step function of the stage value; with a measured bound
    E on the stage's deviation, only inputs within E of a visible threshold can change a byte. On the
    softmax reciprocal (E = 6), this reproduces the exhaustive candidate set exactly from 4,078
    evaluations, and narrower windows lose candidates (494 at window 0).
- **Status.** TESTED-LAB (`.claude/probes/source-behavior-v1/`, README section "Amendment 6.4 follow-up").
