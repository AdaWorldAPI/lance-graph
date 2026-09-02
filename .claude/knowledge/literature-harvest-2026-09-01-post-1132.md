# Literature harvest — post-#1132 reasoning substrate audit (2026-09-01)

**READ BY:** truth-architect, integration-lead, theorem-checker, contradiction-cartographer; anyone proposing a new atom, axis, loco opcode, or pillar certificate.
**Status:** HARVEST (5 Opus auditors, areas A–E with F folded into A; every cited theorem read in the source, not the abstract). Adjudicated on the main thread against **shipped code**, not prose: `epistemic_bassin.rs`, `sigma_propagation.rs`, `jc/{koestenberger,ewa_sandwich,hambly_lyons}.rs`, `planner/nars/{truth,tactics}.rs`, `ogar-loco/vocabulary.rs`, `ogar-epistemic/lib.rs`. Three claims were re-verified by local probes before entering this file (marked ✔ PROBED).
**Scope:** ratified facts of lance-graph #1126..#1132 and OGAR #295..#297 are treated as constitutional — the 256-entry palette is FULL, the six loco-core calls (TERNLOG · BELNAP_JOIN · INFO_GAIN · SIGMA_TENSION · ACCUMULATE · STANCE_ENTROPY) are universal, the 24-axis basis v3 is `0x0334`, one classid per rung, BasinCodebook for value families, red-pillar + mechanism≠rhyme rules. MC #610 is a consumer shadow and was not used as evidence.
**Question answered:** *what genuinely executable or certifying reasoning capability is still absent AFTER the core ops, v3 basis, NARS recipes, canonical atoms and loco/R2IL composition are taken into account?*

**Headline:** MINT NOW is **empty**. The harvest's value is almost entirely **certificates** — three of them correct or strengthen claims the substrate already carries (`pillar_5plus_bound` is miscited and mis-unitized; ACCUMULATE is provably associative; greedy INFO_GAIN admission has a published Ω(n/log n) lower bound against it) — plus four rung-local macros that are compositions of what exists. One v4 pressure survives its witness test (a conative pair) and is blocked on a scope ruling, not on evidence.

> **Area D (Hambly-Lyons / jc Pillar 11)** returned last (its first auditor died on an API error and was relaunched). It carries the single highest-value certificate of the harvest: Hambly-Lyons' *own* Theorems 2/3 give an explicit finite truncation depth for lattice walks, and the paper's own §1.6 figure-eight falsifies the depth-2 forward leg. Both entered the census below (rows D1–D4, placed by value).

---

## 0. Ground truth the audit was adjudicated against

| carrier | what the code actually does (read, not recalled) |
|---|---|
| `EpistemicBassin24` | 24 × `(agree_u4, disagree_u4)`; `net`, `contest = min(a,d)`, `axis_state ∈ {Silent, Agree, Disagree, Contested}`, `entropy_bits` = Shannon over the 4-state census, `stance_entropy_bits(children, axis)`, `accumulate_children` = exact u32 sum per side then clamp 15, `support/refute/contested/silent_mask` u64 |
| `info_gain_u4(before, after)` | `floor(log2(before/after))` clamped 0..15; `after == 0 && before > 0` → 15 |
| `sigma_tension_u4(growth, bound)` | `ceil(4·\|growth\|/bound)` clamped 15; `growth` = `‖log Σ_n‖²_F − ‖log Σ_0‖²_F` (a squared affine-invariant distance), `bound` = `pillar_5plus_bound(n)` |
| `pillar_5plus_bound(n)` | `√(2/n)·√(1+2·0.04·n)` — a **coefficient of variation** curve, `σ_step = 0.2` hard-coded; → 0.4 as n→∞ |
| `jc::koestenberger::prove` | a genuine Sturm inductive mean `S_{n+1} = S_n ⊕_{1/(n+1)} X_{n+1}` on 2×2 SPD, `E[d²(S_n, I)]` vs K-S Thm 1 RHS — **correct use of K-S** |
| `jc::ewa_sandwich::prove` | 10000 random paths of `MΣMᵀ` with i.i.d. rotated log-normal steps; PSD rate ≥ 0.999 and *population* CV of `‖log Σ_n‖²_F` ≤ 1.75 × the curve above; the comment itself says "Köstenberger-Stark-**style**" |
| `planner/nars/truth.rs` | deduction `⟨f₁f₂, f₁f₂c₁c₂⟩`, induction/abduction `w/(w+1)` products, analogy product — NAL-exact |
| `planner/nars/tactics.rs` | RCR #4, TR #6, ASC #7, CAS #8, CR #11 shipped; S5 throttle = confidence floor + budget k + hub exclusion; `GapKind` reach-out |
| `ogar-loco` | six core calls at `0x86..0x8B`; TERNLOG pops 3, ACCUMULATE pops 1, others pop 2 |
| `ogar-epistemic` | axes 0..23 = IS_A PART_OF TYPICALITY MISSING_LINK · SUPPORT REFUTE PARTIAL REPLICATION · PREMISE DEDUCTION FALSIFIER COUNTERFACTUAL · INFO_GAIN TENSION COHERENCE AMBIGUITY · TEMPORAL KAUSAL MODAL LOKAL · PROVENANCE REVISION QUORUM CONTRADICTION |

---

## 1. Census (sorted by expected architecture value)

| # | area | result | destination | ΔH | status |
|---|---|---|---|---|---|
| D1 | D | Hambly-Lyons **Theorem 5/6** (Annals 171, 2010, §2.4; = arXiv v2 Thm 2/3, whose `e` was corrected to `2e` in publication — caught by CodeRabbit on #1133 and verified against the Annals PDF): a lattice path of length L whose first ⌊2e·log(1+√2)·L⌋ = ⌊4.7916·L⌋ iterated integrals vanish is tree-like (d=2); general d: ⌊(2⌈log₃(d/2)⌉+3)·4.7916·L⌋. Truncated signature is a homomorphism into G^N, so `S^(N)(X)=S^(N)(Y) ⟺ X∼Y` at `N ≥ ⌈c(d)·(\|X\|+\|Y\|)⌉` for unit-step lattice walks | **CERTIFICATE_ONLY — Pillar 11 FLIPPED GREEN for lattice walks**, length-parameterized | highest | ✔ constant re-read from math/0507536v2 p.11/p.14 on the main thread; W6 leg shipped in `jc/hambly_lyons.rs` (52/52 reduced words of length ≤ 3 separated at the doubled depth, 64 tree-like at 2e-15, 64 depth-2 false merges all separated by depth 3, d=1 → 7 classes; disable arm fails) |
| D2 | D | The depth-2 forward leg tests membership in `ker(G^∞→G²)`, not tree-likeness: the §1.6 figure-eight has S¹=S²=0 like the out-and-back but S³≠0 | **REJECT** depth-2 as Index regime (necessary condition only) | high | ✔ PROBED (S¹¹²: fig-8 = 1.0, out-and-back = 0.0) |
| 1 | E | `pillar_5plus_bound` cites a theorem about the inductive mean of independent samples (K-S Thm 1) to certify a congruence orbit; `sigma_tension_u4` divides a squared distance by a dimensionless CV | **CERTIFICATE_ONLY (falsifies)** | high | ✔ PROBED (code read; aligned-M arm) |
| 2 | E | Replacement: deterministic isometry bound `\|‖log Σ_n‖_F − ‖log Σ_0‖_F\| ≤ 2·Σ‖log M_k‖_F` for any invertible symmetric M | **MACRO_CANDIDATE** (composition of `logm` + Frobenius) | high | ✔ PROBED (max ratio 0.567 on 2000 random 20-hop paths) |
| 3 | C | Greedy posterior-based admission (INFO_GAIN, VoI) is Ω(n/log n)-suboptimal (Golovin-Krause-Ray Thm 9); EC² edge-cut objective is adaptive-submodular with a `(2 ln(1/p_min)+1)` bound | CERTIFICATE_ONLY + **MACRO_CANDIDATE** `admit_ec2` | high | literature [G] |
| 4 | B | ACCUMULATE (sum-then-clamp on L₁₆) is an MV-monoid: associative, commutative, non-cancellative, top-absorbing — the module's "not claimed associative" is too weak | CERTIFICATE_ONLY (strengthen) | high | ✔ PROBED (0/4096 violations, 1360 non-cancel pairs) |
| 5 | A | Independent per-side clamp is not ratio-preserving: (30,2)→(15,2) moves f 0.938→0.882; heavy agreement drifts toward a=d=15 = false Contested | CERTIFICATE_ONLY (saturation flag) | high | arithmetic [G] |
| 6 | B | Knowledge-order MEET (⊗_k, per-component min) is not expressible by TERNLOG (boolean) or BELNAP_JOIN (max) | **MACRO_CANDIDATE** `MEET_K` | high | non-expressibility [G] |
| 7 | B | Scalar STANCE_ENTROPY is permutation-invariant over {T,F,B,N}: cannot separate Both-heavy from Neither-heavy | EXISTING_COMPOSITION (+ `contest`/`silent_mask`) + certificate | med | theorem [G] |
| 8 | C | AMBIGUITY (E[H(o|s)]) is level-set-inequivalent to INFO_GAIN (log-det vs log-det-ratio); witness in Corva 2026 Remark 4 | GROUNDING_ONLY (axis stands) | med | [G] |
| 9 | A | NAL revision ≡ ACCUMULATE; ⟨f,c⟩ ↔ (w⁺,w⁻) is a bijection for c<1 | EXISTING_PRIMITIVE | high | [G] |
| 10 | A | NAL choice/expectation `(c(f−½)+½) = (w⁺+k/2)/(w+k)` = Krichevsky-Trofimov posterior mean; antisymmetric in a↔d, so STANCE_ENTROPY cannot order beliefs | **MACRO_CANDIDATE** `EXPECT` (integer cross-multiply) | med-high | [G] |
| 11 | A | Syllogistic chaining is multiplicative and absent from the six pooling ops — but it lives in the exact NARS carrier and the shipped tactics use the exact NAL products | EXISTING_PRIMITIVE (exact carrier) | med | ✔ PROBED (`truth.rs` read) |
| 12 | C | Wald-Wolfowitz SPRT is a TERNLOG over an ACCUMULATE'd log-odds with two thresholds; the free-energy pins 0.2/0.05/0.8 have no derivation and mix a bounded and an unbounded term | MACRO_CANDIDATE `sprt_stop`; REJECT the three pins as primitives | med-high | [G]/[S] |
| 13 | C | NOVELTY (parameter info gain) = `½(1/a − 1/a₀)` over Dirichlet counts = a script over REVISION/ACCUMULATE registers | EXISTING_COMPOSITION (v4 axis rejected) | med | [G] |
| 14 | C | Pragmatic value is `−E[log p̃(o)]`, a consumer preference prior | CODEBOOK_VALUE | med | [G] |
| 15 | B | Subjective-logic cumulative fusion ≡ ACCUMULATE in evidence space; base rate `a` enters only at projection | GROUNDING_ONLY + CODEBOOK_VALUE (`prior_u4` per axis) | med | [G] |
| 16 | B | Mares 2002 paraconsistent revision: coherent ⇔ accept∩reject = ∅ ⇔ `contest = 0` | GROUNDING_ONLY | med | [G] |
| 17 | B | MIS-based inconsistency measures are not functions of per-axis (a,d); `contest` satisfies Consistency + Monotony only | REJECT as v4 axis (relational, lives in premise ancestry) | med (fence) | [G] |
| 18 | B | Every interlaced bilattice ≅ L⊙L (Avron 1996); the pair IS the canonical form; signed-net = non-injective truth projection | GROUNDING_ONLY | med | [G] |
| 19 | A | ONA anticipation = eager `FALSIFIER.disagree += δ⁻`, then `agree += δ⁺ > δ⁻`; loop is control-plane | EXISTING_COMPOSITION + certificate (`δ⁺>δ⁻` post-clamp) | med | [G] |
| 20 | A | NARS temporal projection = uniform scaling of both sides = right-shift on the pair; Allen's 13 relations are a value family | MACRO_CANDIDATE `PROJECT` + CODEBOOK_VALUE (Allen) | med | [G]/[H] |
| 21 | A | Typicality/defaults: Nixon diamond → Contested; specificity = IS_A depth → `EXPECT` → choice; Reiter extension *sets* are the one thing not expressible | EXISTING_COMPOSITION | med | [G]/[H] |
| 22 | A | Recipe 12 TCA "Granger" is a rhyme: NARS temporal induction is co-occurrence + decay, Granger is a residual-variance test | REJECT the label (relabel "temporal precedence induction", re-check rung delta) | low-med | [S] |
| 23 | A | Stamp disjointness prevents double-counting, does not certify independence (no theorem exists) | CERTIFICATE_ONLY (narrow the wording) | med | [H] |
| 24 | E | If a probabilistic bound is wanted it is the random-matrix-product CLT (Cuny-Dedecker-Merlevède-Peligrad Thm 2.1/3.1), centred at `2nλ_μ` with `√n` spread — requires i.i.d. M | CERTIFICATE_ONLY | med | [G] |
| 25 | E | No Berry-Esseen for *dependent* matrix products found; edges on a path share endpoints, so this is the live case | CERTIFICATE_ONLY, OPEN | 0 | [H] |
| 26 | E | `SIGMA_TENSION` quarters are linear in k; tail probability is ~1/k² (Cantelli/Chebyshev) — 12 of 16 codes spent on p∈[0.05,1] | CODEBOOK_VALUE (tail-class ladder), after #1 is fixed | med | [G] |
| 27 | E | "EWA" names three different operations (convex EMA, congruence, geodesic); the PSD 10000/10000 proof is a one-line identity | GROUNDING_ONLY + rename; promote `det(M)>eps` to the real gate | low | [G] |
| 28 | E | Le Gouic-Paris-Rigollet-Stromme Cor. 11: `E d² ≤ σ²/n` dimension-free on CAT(0) — sharper than K-S for any actual barycenter | GROUNDING_ONLY | 0 | [G] |
| 29 | B | Shramko-Wansing SIXTEEN₃: t-order and f-order are already the two components; permits asymmetric per-side ops | GROUNDING_ONLY | low | [G] |
| 30 | C | INFO_GAIN = Lindley EIG = I(θ;y) for uniform posteriors; u4 cap = ratio 2¹⁵ with silent one-sided saturation; Miller-Madow applies only to sampled counts | GROUNDING_ONLY (contract text) | low-med | [G] |
| 31 | C | Under additive control + fixed noise, EIG is constant across policies (Koudahl 2021) → a flat-frontier detector | CERTIFICATE_ONLY | med | [G] |
| 32 | A/F | WANT (NSM prime) ≡ Xu 2026 anti-goal `G¡ ≡ ¬(G ⇒ D̃)`: a conative (desire, aversion) pair, not an evidence lobe | **V4_AXIS_CANDIDATE — blocked on scope ruling** | med-high | [G] paradox / [H] relevance |
| 33 | F | CAN (ability/affordance) — one source, no NARS counterpart | DO NOT MINT | 0 | [S] |
| D3 | D | d=1 collapse: `S(X) = (1, Δ, Δ²/2!, …)`; a single `u8:u8` rail read as ONE scalar axis is d=1 and carries only the endpoint (Diehl-Ebrahimi-Fard-Tapia Rem. 1.4) | REJECT 1-rail carrier; `d ≥ 2` static precondition | high (fence) | [G] |
| D4 | D | Goursat scheme is O(h²) (Salvi et al. Thm 15) — the 2e-7 floor is the scheme, not signal; but Thm 8 needs C¹ and the probe paths are piecewise-linear (segment-wise Chen lemma must be written) | CERTIFICATE_ONLY | med | [G]/[H] |
| D5 | D | The depth-∞ leg against a constant path reads `dev ≈ ½‖S²‖²` to leading order — the same Lévy area the depth-2 leg reads; a figure-eight scaled by ε has `dev ∝ ε⁶` and sinks under the O(h²) floor | GROUNDING_ONLY / not independent evidence; mint a min-loop-scale precondition | med | [H] |
| D6 | D | Chevyrev-Oberhauser tensor normalization Λ (Thm 21): without it truncated signature features are not characteristic; the 1e6 "discrimination ratio" is a scale artifact | CERTIFICATE_ONLY + MACRO_CANDIDATE `Λ` | med | [G] |
| D7 | D | Factorial tail decay bounds magnitude, not discriminating power — never cite it for losslessness | GROUNDING_ONLY | low | [G] |
| D8 | D | Iterated-sums signature (ISS) quotients by time-warping, deliberately RETAINS tree-like excursions, has no group structure and no injectivity theorem | REJECT for the Index regime (separate discovery for stutter walks) | med (fence) | [G] |

---

## 2. Detailed rows (the strongest nine)

### R1 · `pillar_5plus_bound` is miscited and `sigma_tension_u4` is mis-unitized — CERTIFICATE_ONLY (falsifies)
- **CLAIM** [G] The K-S citation on `pillar_5plus_bound` is a category error; the readout that consumes it divides a squared distance by a coefficient of variation.
- **PRIMARY SOURCE** Köstenberger & Stark, arXiv:2307.06057v2 (22 Jul 2024), Theorem 1, pp. 6–8, read verbatim.
- **SPECIFIC RESULT** Thm 1 bounds `E[d²(S_n, μ)]` for the **inductive mean** `S_{n+1} = S_n ⊕_{1/(n+1)} X_{n+1}` of **independent** `L²`-valued samples. Its RHS is `O(1/n)`. `pillar_5plus_bound(n) = √(2/n)·√(1+0.08n)` is dimensionless, has no `μ`, `μ_k`, `D_n` or `Var`, and tends to `2σ_step = 0.4` (probed: 1.470, 0.812, 0.534, 0.424, 0.400 at n = 1, 4, 16, 100, 10⁴). jc's own `ewa_sandwich.rs:317` calls it "Köstenberger-Stark-*style*" and derives it from a log-normal/χ² heuristic over a population of random paths.
- **CURRENT MAP** `sigma_propagation::pillar_5plus_bound` doc ("per Pillar 5+ proof-in-code"); `sigma_tension_u4(growth, bound)` (#1129); v3 axis TENSION's grounding line in `E-THE-24-AXIS-BASIS-V3…-1`. **Pillar 5+ itself (`jc/koestenberger.rs`) is NOT affected** — it builds a real inductive mean and is a correct use of Thm 1.
- **DESTINATION** CERTIFICATE_ONLY (falsification of a citation + a units defect; no new concept).
- **CREDIT WHERE DUE (added after operator review, 2026-09-01):** jc's own Pillar 6 header is honest about this — it states the goal as *"rate consistent with Köstenberger-Stark Theorem 1, even though the aggregation operator is sandwich (not inductive mean)"* and its PASS criterion as *"variance concentration consistent with KS Theorem 1 FORM … with n_eff accounting for path-length-dependent volatility"*. That is a documented consistency heuristic, not a claimed theorem. The promotion from heuristic to "bound" happened downstream: PR #322 exported the CV curve into the contract as `pillar_5plus_bound` with the doc-comment "per Pillar 5+ proof-in-code", and #1129's `sigma_tension_u4` consumed it as a per-path bound. The correction therefore targets the contract's doc-comment and the readout's units, not jc.
- **ΔH** Removes "the sandwich walk is concentration-certified by K-S" and the free parameter `σ_step = 0.2`.
- **MECHANISM CHECK** `Σ' = MΣMᵀ` is a congruence, an isometry of `d_AI` for invertible M, not a geodesic convex combination; there is no `S_n` in the code (`grep barycenter|frechet|inductive_mean sigma_propagation.rs` → 0 hits). Units: `growth` is `d_AI(Σ_n, I)² − d_AI(Σ_0, I)²`; `bound` is a CV. The 10000/10000 PASS is a property of the i.i.d. rotated-step generator in `ewa_sandwich::prove`, not of the bound.
- **CERTIFICATE** Replace with R2. Until then, label `pillar_5plus_bound` "empirical CV reference curve for the jc i.i.d. generator, not a bound".
- **FALSIFIER** `Σ_0 = I`, `M_k = diag(e^0.2, e^-0.2)` for all k (eigen-aligned; `‖log M_k‖_F = 0.2√2`). Then `log Σ_n = diag(0.4n, −0.4n)`, growth `= 0.32n²`; at n=100 that is 3200 against a "bound" of 0.424 → `sigma_tension_u4` saturates at 15 on a perfectly regular walk. The i.i.d. arm passes. If the substrate's edge transforms ever correlate along a path (they do — consecutive edges share an endpoint) this arm is the live one.
- **MINT CONSEQUENCE** None. Doc + readout change; see R2.

### R2 · Deterministic isometry bound for the sandwich — MACRO_CANDIDATE (composition), ✔ PROBED
- **CLAIM** [G] For symmetric invertible `M_k`: `|‖log Σ_n‖_F − ‖log Σ_0‖_F| ≤ 2·Σ_{k≤n} ‖log M_k‖_F`.
- **PRIMARY SOURCE** Affine-invariant metric on SPD, `d(A,B) = ‖log(B^{-1/2}AB^{-1/2})‖_F`, GL-congruence invariance (stated at arXiv:2307.06057v2 p.6); triangle inequality.
- **SPECIFIC RESULT** `d(MΣM, I) ≤ d(MΣM, M·I·M) + d(M², I) = d(Σ, I) + 2‖log M‖_F`; iterate.
- **CURRENT MAP** replaces `pillar_5plus_bound` inside `sigma_tension_u4`; `ewa_sandwich::log_norm_growth` keeps its measured quantity (R3 in census: `‖log Σ‖²_F = d_AI(Σ,I)²` exactly, closed form `(ln λ₁)² + (ln λ₂)²` for 2×2).
- **DESTINATION** MACRO_CANDIDATE — `logm` + Frobenius already exist in `Spd2`; the RHS is a running sum the propagator has in hand. Not a primitive: expressible.
- **ΔH** Deletes `σ_step`, deletes the i.i.d./isotropy assumption, adds an inequality valid for adversarial `M` sequences.
- **MECHANISM CHECK** All growth is base-point drift; drift is the accumulated step size. No distributional assumption.
- **CERTIFICATE** ✔ PROBED locally: 2000 random 20-hop paths (random rotations, log-eigenvalues uniform ±0.3), max `lhs/rhs = 0.567`. Aligned arm from R1: `lhs/rhs → 1` (tight).
- **FALSIFIER** Any invertible symmetric `M` sequence with `lhs/rhs > 1 + 1e-12`. Singular `M` voids it — which is the same failure PSD-preservation cannot see (census #27); promote `det(M) > eps` to the gate.
- **MINT CONSEQUENCE** `SIGMA_TENSION`'s `bound` argument becomes `2·Σ‖log M_k‖_F`; the u4 then measures *how much of the deterministic budget a path spent* — a lawful, unit-consistent readout. If a probabilistic statement is later wanted, census #24 (Cuny-Dedecker-Merlevède-Peligrad Thm 2.1/3.1, `2nλ_μ ± 2s√n`) is the correct pillar, and it requires i.i.d. M (census #25 is the open dependent case).

### R3 · Greedy INFO_GAIN admission has a published Ω(n/log n) lower bound; EC² is the adaptive-submodular repair — CERTIFICATE_ONLY + MACRO_CANDIDATE
- **CLAIM** [G] Every posterior-based greedy admission policy (info gain, VoI) is Ω(n/log n)-suboptimal on a family with uniform priors; the `(1−1/e)` guarantee needs adaptive submodularity, which information gain lacks.
- **PRIMARY SOURCE** Golovin, Krause, Ray, arXiv:1010.3091v2 (NeurIPS 2010; rev. 2013-12-16), Theorems 8–9 + Appendix B; Golovin & Krause, arXiv:1003.3967v5 (JAIR 42, 2011), Theorem 5 and Prop. 2.
- **SPECIFIC RESULT** Construction: `m = 2^q` classes × 2 hypotheses, one test `t₀` reveals `v`, tests `t_k` reveal `1[φ_k(a)=v]` — individually zero information gain until `t₀` has run — plus slow sequential tests and unbounded dummies. Optimal cost `log₂n`; any posterior-based greedy pays `≥ m/2`. Mechanism: **complementarity** — a set of hops is far better than the sum of its parts. EC² (`f_EC = w(⋃ E_t)` over cross-class pairs, `w = P(h)P(h')`) is adaptive submodular, bound `(2 ln(1/p_min) + 1)·c(π*)`.
- **CURRENT MAP** S5 throttle (`c_min`, budget `k`, hub exclusion) + `admit_derived`; `info_gain_u4` ranking; the FIELD axis INFO_GAIN.
- **DESTINATION** CERTIFICATE_ONLY (the throttle inherits no approximation guarantee) + MACRO_CANDIDATE `admit_ec2` (a weighted pair-cut count over the Belnap-partitioned candidate set — registers that already exist; the budget-`k` throttle is exactly Theorem 5's cardinality constraint).
- **ΔH** Moves the frontier's admit rule from "unproven" to "in the proved-bad region" until the EC² certificate runs.
- **MECHANISM CHECK** Multi-hop abductive derivation is exactly the `t_k`-shaped case: individually worthless, jointly decisive.
- **CERTIFICATE** Adaptive-submodularity check over 10⁴ sampled `(ψ ⊆ ψ', t)` triples: 0 violations for `f_EC`, ≥1 exhibited for INFO_GAIN.
- **FALSIFIER** Instantiate the construction at `q=8` (256 classes, 512 hypotheses): optimal 9 hops, posterior-greedy ≥128 expected. Run the shipped frontier; ≤20 hops means it is not posterior-based greedy and the exposure does not apply.
- **MINT CONSEQUENCE** None. Rung-local admit script.

### R4 · ACCUMULATE is an MV-monoid — CERTIFICATE_ONLY (strengthen), ✔ PROBED
- **CLAIM** [G] `a ⊕ b = min(15, a+b)` on `L₁₆` is associative and commutative (Chang 1958 MV-algebra axioms; Mundici tutorial); it is non-cancellative with absorbing top.
- **CURRENT MAP** `accumulate_children` and the module doc's "recursively composing already-clamped child registers is NOT associative".
- **SPECIFIC RESULT** ✔ PROBED: 0 violations of `(a⊕b)⊕c = a⊕(b⊕c)` over all 4096 triples per side; 1360 `(a,b≠c)` pairs with `a⊕b = a⊕c`. Any DAG fold order gives a bit-identical register; what is lost is *mass* (a clamped 15 means "≥15"), never *order*.
- **DESTINATION** CERTIFICATE_ONLY. Rewrite the caveat as "associative, commutative, monotone, non-cancellative, top-absorbing; saturation understates mass, never converts conflict to silence".
- **ΔH** Licenses schedulers to reassociate hop trees without proof obligation. Area A's assertion that the clamp *causes* non-associativity is refuted by the same probe; what the clamp causes is R5.
- **FALSIFIER** already run; re-runnable as a `#[test]` over `0..16³`.
- **MINT CONSEQUENCE** Doc/contract only.

### R5 · Per-side clamp is not ratio-preserving: saturation manufactures Contested — CERTIFICATE_ONLY (saturation flag)
- **CLAIM** [G] arithmetic, [H] practical bite. NAL `f = w⁺/w`, `c = w/(w+k)`; independent clamping of `w⁺` and `w⁻` at 15 is not ratio-preserving.
- **PRIMARY SOURCE** Wang NAL (`c = w/(w+k)`), restated Xu arXiv:2607.20902v1 §2.2 and Hammer & Lofthouse ONA (AGI-2020).
- **SPECIFIC RESULT** true `(30,2)` → `f = 0.938`; stored `(15,2)` → `0.882`; as both sides saturate every heavily-evidenced axis drifts to `(15,15)` = Contested. The confidence ceiling `c ≤ 15/16` is AIKR-legal (NAL requires `c < 1`); the frequency drift is not. Interaction with ONA's Assumption-of-Failure (census #19): once `disagree` saturates, no `δ⁺` can outweigh it — an eagerly-pessimistic axis becomes permanently pessimistic.
- **CURRENT MAP** `axis_state` → `Contested` on `(a>0, d>0)`; `contested_mask`; ASKED_CONTESTED ternlog.
- **DESTINATION** CERTIFICATE_ONLY: a Contested read must carry `saturated = (a==15 || d==15)` or is uncertified. Do **not** widen u4→u8 (moves the ceiling, keeps the distortion).
- **FALSIFIER** 30 agree + 2 disagree via repeated `accumulate_children` on one axis → assert the classification is not reported as genuine conflict without the flag. Second arm: 40 predict/confirm cycles with `δ⁻=1, δ⁺=2`; `EXPECT` (R7) must be non-decreasing; under the current clamp it plateaus and, on one failure, drops and never recovers.
- **MINT CONSEQUENCE** None (one derived flag bit; no storage).

### R6 · Knowledge-order MEET is the one bilattice law the core set cannot express — MACRO_CANDIDATE `MEET_K`
- **CLAIM** [G] (Arieli-Avron 1996; Fitting): an interlaced bilattice needs both `⊕_k` (join, componentwise max = BELNAP_JOIN) and `⊗_k` (consensus, componentwise min). Every interlaced bilattice is `L⊙L` (Avron 1996), so the `(a,d)` pair is the canonical form and the signed net was the non-injective truth-projection (census #18).
- **CURRENT MAP** BELNAP_JOIN only; TERNLOG is boolean over 3 mask columns, so u4 `min` is not a TERNLOG; no saturating subtract to recover `a+b−max`.
- **SPECIFIC RESULT / FALSIFIER** A=(3,0), B=(0,3): BELNAP_JOIN → (3,3) Contested; ACCUMULATE → (3,3); `⊗_k` → (0,0) Silent ("what both sources agree on"). Exhaustive search over length-≤3 compositions of the six calls on the 256×256 pair table for one yielding (0,0) here and (3,3) on ((3,0),(3,0)): expected none.
- **DESTINATION** MACRO_CANDIDATE — rung-local compare-select over nibbles. Not a global opcode. Interlacing certificate (B1 falsifier: `¬, ∧_t, ∨_t` preserve `≤_k` over all 256 pairs) travels with it. Also grounds Mares 2002 (census #16): coherent ⇔ `contest = 0`.
- **MINT CONSEQUENCE** One script; zero axes; zero opcodes.

### R7 · NAL choice/expectation is a Krichevsky-Trofimov mean the entropy readouts cannot reproduce — MACRO_CANDIDATE `EXPECT`
- **CLAIM** [G] `exp(f,c) = c(f−½)+½ = (w⁺ + k/2)/(w + k)`; with k=1 the Jeffreys/KT posterior mean `(a+½)/(a+d+1)`.
- **PRIMARY SOURCE** Hammer & Lofthouse, ONA (AGI-2020), verbatim definition of `exp`.
- **CURRENT MAP** belief-arena CHOICE path (CR #11, ASC #7 overlap fallback) — ordering key not pinned to a formula. STANCE_ENTROPY is symmetric in `a↔d`: `(12,3)` and `(3,12)` have equal stance entropy, equal info gain, `exp` 0.78 vs 0.22.
- **DESTINATION** MACRO_CANDIDATE. Integer-safe: order by `(2a+k)(w'+k)` vs `(2a'+k)(w+k)` — a u8 cross-multiply, no division. `k` is an operand (coupled to the ceiling in R5), not a global constant.
- **FALSIFIER** (i) `exp(a,d) == (a+0.5)/(a+d+1)` over the u4 grid; (ii) anti-vacuity: find a pair with equal STANCE_ENTROPY and equal INFO_GAIN and different `exp`, assert no choice rule over those two ops separates it.
- **MINT CONSEQUENCE** No classid. Also the required ingredient for census #21 (specificity preemption) and #23 (choice-on-overlap fallback).

### R8 · AMBIGUITY ≠ INFO_GAIN, with a witness pair — GROUNDING_ONLY (axis stands)
- **CLAIM** [G] Corva, arXiv:2607.20306v1 (2026-07-22) Remark 4 / Example 1: in linear-Gaussian active inference with state-dependent observation noise, `ε_k(π) = ε_k(π')` for every policy pair (equal information gain, a log-det **ratio**) while ambiguity `½ ln det(2πe R)` differs (a log-**det**). Plus Koudahl-Kouw-de Vries 2021: under additive control and fixed noise EIG is *constant* across policies (census #31 → a flat-frontier assertion).
- **CURRENT MAP** FIELD::AMBIGUITY vs FIELD::INFO_GAIN.
- **DESTINATION** GROUNDING_ONLY; retires "AMBIGUITY duplicates INFO_GAIN". Two-sided certificate: a pair with equal INFO_GAIN/unequal AMBIGUITY (Remark 4) and a pair with equal AMBIGUITY/unequal INFO_GAIN (halve a clean candidate set: 1 bit gain, ambiguity unchanged).
- **FALSIFIER** Scalar model `A=B=C=Q=1, Σ₀=1, R(x)=1+x²`: `u=0` → gain 0.549 nats, ambiguity 1.419; `u=2` → gain 0.168, ambiguity 2.224; then the Remark-4 level-set arm.
- **MINT CONSEQUENCE** None. (Companion collapses: NOVELTY = `½(1/a − 1/a₀)` over Dirichlet counts = a script over REVISION/ACCUMULATE registers, census #13; pragmatic value = consumer prior `C`, census #14.)

### R9 · Hambly-Lyons Theorems 5/6 green Pillar 11 for lattice walks and kill the fixed-depth Index regime — CERTIFICATE_ONLY
- **CLAIM** [G] (**constant corrected 2026-09-01 after review**: the arXiv v2 text states Theorems 2/3 with `e·log(1+√2)`; the published Annals 171 text states Theorems 5/6 with `2e·log(1+√2)`, its proof taking `x = 2·log(1+√2)·L` — the arXiv proof's index `k` counts pairs of degrees, so the journal form is the corrected one). For a length-`L` path on the 2-d integer lattice, vanishing of the first `⌊2e·log(1+√2)·L⌋ = ⌊4.7916·L⌋` iterated integrals implies tree-likeness (Thm 5); in `R^d`, `⌊(2⌈log₃(d/2)⌉+3)·4.7916·L⌋` (Thm 6). The GL(2,C)-projected integrals carry less than the full tensor algebra, so the full truncated signature satisfies it a fortiori.
- **SPECIFIC RESULT** `S^{(N)}` is a homomorphism into the free nilpotent group `G^N`, so `S^{(N)}(X) = S^{(N)}(Y) ⟺ S^{(N)}(X ⋆ Y^{-1}) = 1`; apply Thm 2/3 to the concatenation: **`N ≥ ⌊c(d)·(|X|+|Y|)⌋`, `c(2) = 2e·ln(1+√2)`** is a complete two-sided certificate for unit-step lattice walks.
- **CURRENT MAP** `jc::hambly_lyons` (cites Thm 1/Cor 1.5 only — correct, depth-∞, insufficient); `sigker` `CodecRoute::Sigker` Index regime; the `E-V3-FACET-4-PLUS-12` rails as carrier.
- **DESTINATION** CERTIFICATE_ONLY. The auditor proposed CORE_PRIMITIVE_CANDIDATE; **overruled on the main thread**: this greens a *held* primitive (the operator's stated highest-value outcome), it mints nothing new, and the Index regime it certifies is length-parameterized, so `sigker` must carry a walk-length budget and escalate beyond it. Preconditions before flipping: (a) `d ≥ 2` — a `u8:u8` rail read as one scalar axis is d=1 and collapses to the endpoint (census D3); the canon "u8:u8 is two bytes, never widened" is load-bearing here; (b) steps must be unit basis-aligned lattice steps — Thm 2's hypothesis is literally `‖x_k − x_{k+1}‖ = 1` on `Z^{|A|}`; arbitrary quantized vectors are outside it.
- **ΔH** Eliminates "no finite-depth Hambly-Lyons exists" and "depth 2 suffices" simultaneously. The depth-2 forward leg (census D2, ✔ PROBED) is a necessary condition only; the 1e6 discrimination ratio is a scale artifact until Λ-normalized (D6).
- **FALSIFIER** Enumerate reduced/unreduced word pairs on 2 generators up to L=6; assert `S^{(⌈2.3959·2L⌉)}` (exact rational arithmetic on basis exponentials) separates every non-tree-equivalent pair and merges every tree-equivalent pair; re-run at depth 2 and count false merges (expect many; the figure-eight is one).
- **MINT CONSEQUENCE** No new concept. Pillar 11 may flip green *for the lattice-walk class* once the constant is re-read and the two preconditions are asserted in `sigker`; the fixed-depth wording is retired in place.

---

## 3. MINT NOW
**EMPTY.** No result survived both the composition test and the mechanism test with a new executable address attached.

## 4. MACROS WORTH PROBING (rung-local loco/R2IL scripts; no global opcode)
| macro | what | composition | probe first |
|---|---|---|---|
| `SANDWICH_BUDGET` | R2 replacement for `pillar_5plus_bound` | running `2·Σ‖log M_k‖_F` (2×2 closed form `√((ln λ₁)²+(ln λ₂)²)`) | aligned-M arm + `det(M)>eps` gate |
| `MEET_K` | knowledge-order consensus | per-nibble min over the pair | interlacing check over 256 pairs; non-expressibility search |
| `EXPECT` | NAL choice key | `(2a+k)(w'+k)` cross-multiply | R7 falsifiers (i),(ii) |
| `admit_ec2` | adaptive-submodular frontier admission | weighted pair-cut count over Belnap-partitioned candidates, under budget `k` | R3 construction at q=8; submodularity triples |
| `sprt_stop` | theorem-backed stopping (Wald-Wolfowitz 1948) | TERNLOG over (`cum_llr < ln B`, `> ln A`, budget) on an ACCUMULATE'd log-odds | `(α,β)=(0.05,0.05)` → `E[N] ≈ 5.9` at ±0.5 nats/hop vs fixed k=20 |
| `PROJECT` | NARS temporal projection | right-shift both nibbles by `n` (`λ = 2^-n`) before ACCUMULATE | `f` preserved within 1 LSB, `c` strictly decreasing |
| `NOVELTY` | parameter info gain | `½(1/agree − 1/total)` over REVISION/ACCUMULATE | hop α (1/10) → 0.450, hop β (9/10) → 0.0056 at equal INFO_GAIN |
| `Λ` (tensor normalization) | Chevyrev-Oberhauser Def. 12 | dilation `δ_c` with `ψ` injective | triangle scaled ×{0.1,1,10}: ratio stable only when normalized |
| `flat_frontier_assert` | Koudahl degenerate regime detector | `stddev(info_gain_u4)` over admitted frontier `> 0` | 32 candidates each 4096→2048 |

## 5. CERTIFICATES TO HARVEST (green or falsify an existing pillar/claim)
1. **Pillar 11 green path** — Hambly-Lyons Annals Thm 5/6 (§2.4) + homomorphism into `G^N`; constant `2e·log(1+√2)` (the arXiv v2 `e` is pre-publication); assert `d ≥ 2` and unit-lattice steps in `sigker`; retire fixed depth 2 (necessary-condition only). Companion: Salvi et al. Thm 15 (O(h²)) certifies the 2e-7 floor but Thm 8 needs C¹ — write the segment-wise Chen lemma; add a minimum-loop-scale precondition (figure-eight `dev ∝ ε⁶` sinks under the floor).
2. **Pillar 5+ scope** — keep `jc/koestenberger.rs` green for inductive means; strike "per Pillar 5+ / K-S" from `sigma_propagation::pillar_5plus_bound`; relabel it as the jc generator's empirical CV curve; fix `sigma_tension_u4` units via R2. If a stochastic pillar is wanted later: Cuny-Dedecker-Merlevède-Peligrad arXiv:2110.10937v2 Thm 2.1/3.1 (i.i.d. M only; the dependent case is OPEN and is the live one).
3. **ACCUMULATE contract** — associative/commutative/non-cancellative/top-absorbing (R4, probed); saturation flag on Contested reads (R5); `δ⁺ > δ⁻` post-clamp for Assumption-of-Failure (census #19).
4. **Frontier admission** — Golovin-Krause-Ray Thm 9 lower bound; EC² adaptive-submodularity as the repair (R3); Koudahl flat-EIG detector (#31).
5. **STANCE_ENTROPY never gates alone** — permutation invariance over {T,F,B,N} (census #7): pair it with mean `contest` or `silent_mask`.
6. **Stamp wording** — disjointness certifies "no evidence item counted twice", not independence (census #23); confirm the arena's choice-on-overlap fallback exists and uses `EXPECT`.
7. **Recipe 12 TCA** — drop "Granger" (rhyme); relabel "temporal precedence induction"; re-check its rung delta (#22).
8. **INFO_GAIN contract text** — `= Lindley EIG` for uniform posteriors; u4 cap = ratio 2¹⁵ with one-sided silent saturation (expose a flag); Miller-Madow only where counts are sampled (#30).
9. **SIGMA_TENSION ladder** — after (2): re-index the u4 as a Cantelli/Chebyshev tail class (`q=7 ↦ k=1.75 ↦ p ≤ 0.327` today; proposed `q=15 ↦ k=4.472 ↦ p ≤ 0.05`) — CODEBOOK_VALUE (#26).

## 6. V3 SURVIVES (proposed novelties that collapse into the 24 axes or the six calls)
- **NAL revision** → ACCUMULATE (bijection `⟨f,c⟩ ↔ (w⁺,w⁻)` for c<1; census #9).
- **Subjective-logic cumulative fusion** → ACCUMULATE; base rate → per-axis `prior_u4` in the BasinCodebook (#15).
- **Mares paraconsistent coherence** → `contest = 0` (#16).
- **Shramko-Wansing truth/falsity orders** → the two components; permits asymmetric per-side ops (#29).
- **Novelty / parameter info gain** → script over REVISION + ACCUMULATE (#13).
- **Pragmatic value** → consumer codebook prior, not an axis (#14).
- **Anticipation / negative evidence on failed prediction** → `FALSIFIER.disagree += δ⁻` then `agree += δ⁺` (#19); the loop is control-plane.
- **Typicality / defaults / specificity preemption** → IS_A depth → `EXPECT` → choice; Nixon diamond → Contested (#21). Reiter extension *sets* are the one thing not expressible, deliberately.
- **Allen interval relations** → ValueCodebook operand on TEKAMOLO, never an opcode (#20).
- **Syllogistic chaining** → exact NARS carrier (`truth.rs` products, ✔ read); the six calls are a pooling algebra and are meant to be (#11).
- **SPRT** → TERNLOG over an ACCUMULATE'd log-odds (#12).
- **AMBIGUITY** stands as its own axis (R8) — a survival in the other direction.

## 7. V4 PRESSURE (irreducible dimensions, with witness pair)
- **Conative pair (desire, aversion) — blocked on a scope ruling, not on evidence.** Two independent sources converge: NSM's prime WANT (Area F) and Xu 2026 (arXiv:2607.20902v1, Thm 4: `G¡ ≡ G ⇒ ¬D̃ ≡ ¬(G ⇒ D̃)`; "avoid G" ≠ "pursue ¬G"). **Witness pair:** the light-press case — P1/P2/G1 with identical evidence on every one of the 24 axes (all are evidence-about-propositions), where "avoid hurt" and "pursue not-hurt" are indistinguishable in v3 yet the lawful downstream action differs (press vs not press). **Mechanism fence:** modelling aversion as `disagree` on an evidence axis is exactly the paradox's assumption A3 — if ever minted it is a *separate* pair, never an evidence lobe. **Ruling needed:** is the basis epistemic-only? If yes → REJECT permanently and record why; goal handling then lives in `ActionDef`/kanban, outside the basis.
- Nothing else survived the witness test: MIS-inconsistency (relational, in premise ancestry; #17), novelty (#13), pragmatic value (#14), a third bilattice order (#29) all collapsed.

## 8. DO NOT MINT
- **"Granger"** on recipe 12 — a residual-variance test NARS does not compute (#22).
- **A Hambly-Lyons axis or fixed-depth signature primitive** — the certificate is length-parameterized; a depth-2 Index regime is falsified by the paper's own figure-eight (D2, probed).
- **Iterated-sums signature** as the discrete signature — quotients by time-warping, retains tree-like excursions, no group, no injectivity (D8). Separate discovery for stutter walks only.
- **An inconsistency-measure axis** — `I_MI` is not a function of per-axis `(a,d)` (#17).
- **A "novelty" or "epistemic value" scalar** — reducible (#13, #14).
- **u4→u8 widening** to fix saturation — moves the ceiling, keeps the ratio distortion (R5).
- **A seventh core opcode for MEET/EXPECT/SYLLOG/SPRT** — all rung-local scripts.
- **CAN (ability/affordance)** — one source, no mechanism (#33).
- **Factorial tail decay as a losslessness argument** — bounds magnitude, not discrimination (D7).
- **"EWA" as a name for the congruence** — three operations share the word (#27).

---

### Method notes
- Five Opus auditors (A+F, B, C, D, E), alphaXiv + WebFetch, theorems read in source. Area D's first run died on an API error and was relaunched with the same brief.
- Main-thread adjudication overrode two auditor destinations: D's CORE_PRIMITIVE_CANDIDATE → CERTIFICATE_ONLY (greening a held primitive is the higher-value, lower-entropy outcome); A's "the clamp causes non-associativity" → refuted by the exhaustive probe, retained as ratio distortion (R5).
- Local probes (scratchpad, not committed): L₁₆ associativity (4096 triples), sandwich isometry bound (2000 paths), aligned-M growth, figure-eight depth-2/depth-3 signature. Each is small enough to become a `#[test]` in the crate it certifies.
- Zero code changed by this harvest. Board hygiene: one EPIPHANIES entry for the two corrections (Pillar 5+ scope, Pillar 11 green path); SUPERSESSION-INDEX regenerated last.
