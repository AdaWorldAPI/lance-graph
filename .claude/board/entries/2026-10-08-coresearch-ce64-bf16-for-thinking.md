# 2026-10-08 — Coresearch: "CausalEdge64 is like BF16 for thinking"

**Status:** EXPLORATION MAP. Nothing ratified, nothing adopted. Five scouts,
three co-architects, premise gate; probes are pre-registered, not run (one
main-thread arithmetic check is marked as such).
**Operator baseline:** "ALU, ISA, causaledge64 is like BF16 for thinking."
**Harvest:** five papers read in full by the main thread and written to
`.arxiv/` (1905.12322, 2209.05433, 2603.24161, 0712.1182, 2310.10537).

## Premise gate: PREMISE-SPLIT

`premise-auditor`:
- **CE64 is not a number format.** It is a fixed-width tagged record, i.e. an
  instruction word of nine fields: identity (SPO), truth (F, C), codes (Pearl,
  Direction), opcode (Inference), flags (Plasticity), handle (Witness) and a
  state code (Epi5). Each op declares per field whether it computes, passes or
  constant-sets it (`isa.rs:377-472`).
- **"BF16 for thinking" holds for bits 24..39 only.** F and C are two u8
  fixed-point truth scalars, and `isa::truth` is their ALU. Rounding, range and
  accumulation vocabulary applies there and nowhere else. "Not applicable —
  wrong category" is the verdict for every other field.

The question was therefore asked as two:
- **Q-A (format):** the F/C truth number.
- **Q-B (instruction word):** the whole 64-bit record read in place.

**The infographic's readings are different concepts from the code:**

| bits | infographic reading | code |
|---|---|---|
| 46..49 | "activation" | the signed opcode: sign = chain direction, magnitude = NARS rule (`layout.rs:17-26`) |
| 50..52 | "entropy/novelty" | three per-plane hot/frozen flags (`plasticity.rs`) |
| 53..58 | "W1/W2/W3 belief update" | a corpus root handle that `forward` and `syllogize` zero (`isa.rs:392,440`) |
| 40..42 | "Pearl ladder" | `pearl::CausalMask`, a 2³ plane-subset lattice |

On the last row: `SO = 0b101` (Association) and `PO = 0b011` (Intervention) are incomparable subsets (`pearl.rs:37-77`), so "rung ≥ k" is not an up-set of this field. Read as a ladder ordinal, it is CONFLICTS-ANCHOR. Read as a plane mask, it is consistent.

## Verified ground (code)

- **Encode/decode.** Decode is `u8/255`. Encode is `(x.clamp(0,1)*255).round()`, which rounds half away from zero; NaN becomes 0 through the saturating cast (`edge.rs:322-340, 805-806`).
- **Revision.** `w = c/(1−c)` gives horizon k = 1, and `evidence_weight` returns `f32::MAX` only for code 255 (`isa.rs:216-261`).
  - Two operands at 255: c = inf/inf = NaN, stored as 0.
  - One operand at 255: c = 255.
- **Four quantisers for one concept:**
  - `isa` revision;
  - contract `NarsTruth`: capped at 0.99, with a +1e-9 term;
  - `NarsTables`: c in 16 bins, `deduce` c_out = f_out, not used by the live ISA;
  - `TruthU8` (arm-discovery): integer floor, never produces 255.

  There is no integer truth ALU.
- **Folds.** mask-risc `MaskedStridedGroupSum` (`ir.rs:309`) already gives an unweighted ΣF / ΣC over a masked population, in place. No weighted fold exists.
- **ndarray.** It has u8 saturating add/sub, a u8×i8 VNNI dot and BF16 RNE. Stochastic rounding is ABSENT (closed grep).

**MAIN-THREAD ARITHMETIC CHECK** (a numpy f32 model of `isa::truth::revision`, NOT the Rust code; it needs the Rust probe to confirm).

With `w = c/(255−c)`, revision has an exact integer form:

```text
c_out·255 = 255·N/D
N = c1(255−c2) + c2(255−c1)
D = 255² − c1c2
```

- `c_out` depends only on `(c1, c2)`, so the c side is 65,536 cases.
- At `c1 = c2 = 255`, N = D = 0. The dogmatic case is 0/0 even in exact arithmetic: a policy question, not a float bug.
- The f32 model agrees with the exact form on all but 3 of 65,535 pairs. All three are exact .5 ties that f32 lands just below: `(45,45) → 76.5` and `(85,165)/(165,85) → 178.5`.
- There are 0 monotonicity violations in c2.
- So the rounding mode is not cosmetic for any integer reference ALU.

## Crosswalk and verdicts

Relation codes: HAVE / PARTIAL / NEW / CONFLICTS-ANCHOR. Firewall codes: PASS / CONFLICT (fits only behind a named seam) / TRAP.

| # | idea | outside source | relation | firewall | verdict |
|---|---|---|---|---|---|
| X1 | narrow storage, wide accumulate, round once | BF16 1905.12322; TPU and VDPBF16PS docs | PARTIAL: one op already does it; chains and `NarsTables` re-quantise | CONFLICT; seam: fold-side or new entry only | **PROBE** P-X1 |
| X2 | c=255 is a policy cell | FP8 2209.05433; posits; Jøsang 0712.1182; PCRLLM 2511.08392 | PARTIAL | (a) cap or saturate at 254: CONFLICT, seam = I-LEGACY gate. (b) give 255 a stored "dogmatic" meaning: TRAP | **PROBE** P-X2, then a policy decision; (b) **SKIP** |
| X3 | rounding/saturation/c255 policy declared in the op contract | VDPBF16PS; VNNI wrap vs saturate | PARTIAL: `isa::contracts` lacks these fields | PASS (additive) | **PROBE** P-X3 (tie count; the model already found 2 distinct ties) |
| X4 | one concept, four quantisers | OCP vs FNUZ name hazard | HAVE (the hazard) | diagnosis PASS; unifying them CONFLICT (I-LEGACY, unknown NarsTables callers) | **ADOPT-NOW as an ISSUE**; **PROBE** P-X4 parity grid |
| X5 | integer-only truth ALU | Jacob 1712.05877 | NEW | CONFLICT; seam: an A4 oracle or a T1 primitive, never a substitute | **PROBE** P-X5 (folded into P-X2) |
| X6 | stochastic rounding against stagnation | 2603.24161 | NEW | TRAP as proposed: determinism, I-LEGACY, Jirak | **PARK** until P-X1 shows stagnation |
| X7 | evidence space as the currency | posit, LNS 2012.03458, SL/NARS | PARTIAL | exact Σw/Σw·f internal: CONFLICT with seam; log-w or Mitchell: TRAP (lossy) | **PROBE** P-X7 (exact fold); log-w **SKIP** |
| X8 | MX block scale | 2310.10537 | NEW | TRAP as a terminal (lossy) | **SKIP**, unless P-X8 finds the exact fold insufficient |
| X9 | predicted scaling (Transformer Engine) | TE docs | NEW | TRAP | **SKIP** |
| X10 | (F,C) as an interval `[fc, fc+1−c]`; C is the information order | Wang; Belnap/Ginsberg | PARTIAL | PASS | **ADOPT-NOW** (doc line) |
| X11 | contradiction magnitude as a transient output | Dempster–Shafer conflict; Belnap B | NEW | PASS as a sibling fn; a stored B flag is a TRAP | **PROBE** P-X11. The DS form `K = c1c2[f1(1−f2)+f2(1−f1)]` is predicted to fail (non-zero on `x+x`); the candidate is `K₂ = c1c2·\|f1−f2\|` |
| X12 | subjective-logic opinions as the belief number | Jøsang | CONFLICTS-ANCHOR (earlier rejection) | TRAP | **SKIP**; only the case analysis is harvested |
| X13 | bits 40..42 are a plane lattice | PCH / CHT 2401.02602 | HAVE (the code is right) | PASS; an ordinal "rung ≥ k" predicate is a TRAP | **ADOPT-NOW** (doc line + guard test, P-X13) |
| X14 | Epi5 order = chain6 × poset4 | lattice theory | HAVE | PASS | cite only; P-X14 pins cover size == n+1 |
| X15 | FCA: eligibility extents bounded by the concept lattice | Ganter–Wille | PARTIAL (never counted) | PASS | **PROBE** P-X15: sizes D-RPF-1's covers below the separable 24+8 bound |
| X16 | illegal encodings trap | RISC-V rationale | PARTIAL: an all-zero operand already faults in `forward` | status quo PASS; making all-zero data illegal is a TRAP | **PROBE** P-X16 (pin; check `revise(x, 0) == x`) |
| X17 | predicate separate from data; merge vs zero declared | SVE, AVX-512 writemask | HAVE (law A2) | PASS as documentation | **PROBE** P-X17 (pin learn's frozen planes 8/8) |
| X18 | opcode from the instruction, not from an operand | ISA practice | PARTIAL (PARKED earlier) | CONFLICT; seam: a new additive entry point | **PARK** |
| X19 | MSB-first early stop for strided predicates | BitWeaving/V SIGMOD13 | PARTIAL | PASS (an exact stop in ndarray::simd, plus an A4 differential) | **PROBE** P-X19 |
| X20 | bit-sliced F/C beside the row | SIMDRAM 2012.11890; FastLanes | NEW | persisted: TRAP (a second copy); per-tile transient in scratch: PASS | **PROBE** P-X20, transient only; prefer the strided pattern covers from R-MHB-1 |
| X21 | infographic readings stored in bits | operator image | CONFLICTS-ANCHOR | TRAP | **SKIP** as stored meaning; "novelty" may be the derived K₂ |

## Epiphany candidates

These have not been admitted to `EPIPHANIES.md`. Each passes the "would it matter without the defect?" test as stated by the bridge architect, and still needs its probe.

1. **CE64 is an instruction word wrapped around a 16-bit reasoning number.**
   - The BF16 analogy is exact for F/C: range over precision, wide accumulate, round once, declared edge codes.
   - Everything else is an opcode, a predicate, a handle or a tag, read through its own order structure.
2. **One truth quantiser, declared in the contract** (X1–X5, X16).
   - Encode, decode, rounding, saturation, the meaning of code 255 and the accumulation width are implicit in four places today.
   - Declare them once in `isa::contracts`. Every other quantiser then becomes a tested reading of that declaration.
3. **Revision is cumulative fusion in evidence space** (X1, X7, X20; harvest from 0712.1182).
   - In evidence space it is an exact, cancellative sum. That gives:
     - **D-RPF-4:** a 256-entry `c → w` table applied per byte, then the existing strided group-sum. No new opcode is needed.
     - **Fission:** removing a known contribution, for detached revision.
     - **The stamp gate selects the rule:** cumulative fusion applies only to independent evidence, and averaging fusion is the idempotent one. The self-revision defect is this missing gate.
4. **Contradiction is a transient output, not a state** (X10, X11, X21, and the optics "cancellation loses contradiction").
   - Revision is information-monotone but folds "both" into "balanced".
   - `K₂ = c1c2·|f1−f2|` reports the disagreement without storing it.
   - The DS form is predicted to be wrong, because it fires on `x ⊕ x`.
5. **Predicates follow the field's order, not its bit width** (X13–X15).
   - Pearl is a subset lattice; Epi5 is chain × poset; C is the information order; F is the truth order.
   - A threshold is only well-posed over a chain.
   - D-RPF-1 cover sizes are bounded by the concept lattice of the eligibility table.

## Pre-registered probes

Size is S, M or L. Every guard probe has a can-fire half and a can-stay-silent half.

**P-X2 + P-X5** (S, next to `tests/ce64_isa_golden.rs`)
- **Grid:** all 65,536 `(c1,c2)` pairs, plus `c ∈ {253,254,255}` × all f values.
- **Count:** NaN results, zero-stores, and monotonicity violations.
- **Compare three forms:** current `isa`; capped at 254 before the weight; the exact integer form.
- **Kill (b):** the cap gives 0 NaN, 0 violations, and agreement on all 65,535 non-(255,255) pairs.
- **Baseline pin:** exactly the (255,255) cell family, using `==`.

**P-X1/X6** (S, new `tests/ce64_chain_rounding.rs`)
- **Input:** same-sign weak-revision chains, N up to 1000.
- **Compare three forms:** re-encode per hop; f64 accumulate with one round at the end; hashed stochastic rounding.
- **Kill SR:** stagnation = 0 over the whole grid.
- **Kill X1:** `max|Δ| ≤ 1` code.

**P-X3** (S)
- **Measure:** count exact .5 ties over the revision grid.
- **Kill:** count = 0 makes the declaration cosmetic. The model already predicts ≥ 2.

**P-X4** (S–M)
- **Input:** a differential across the four quantisers on a common grid.
- **Kill unification:** all pairs agree within 1 code on non-255 cells.

**P-X7/X8** (S)
- **Input:** a sequential pairwise fold vs the closed form `Σw / Σw·f` over random permutations.
- **Kill:** one distinct code per population. Expected outcome: it fails, so the exact Σ terminal is needed.

**P-X11** (S)
- **Check:** `K₂(x,x) = 0` for all x, and `K₂((1,c),(0,c)) > K₂((½,c),(½,c))`.
- **Prediction:** the DS form K fails on `x + x`.

**P-X13/X14/X15** (S, next to the affordance law and `resident_ce64_predicate.rs`)
- Pearl monotone under ⊆.
- An Epi5 threshold cover equals n+1.
- Count distinct `facts_population` extents, and sum the minimal ternary covers against the 24+8 bound.
- **Kill X15:** no reduction.

**P-X16/X17** (S)
- Pin the operand-zero table: 8 cells.
- `revise(x, 0) == x`.
- Learn's frozen planes are bit-identical: 8/8.

**P-X19/X20** (M, mask-risc bench)
- **X19:** MSB-first early stop vs the strided ≤ n+1 passes. PARK if the mean is ≥ 6/8 planes or the speed-up is < 1.5×.
- **X20:** a transient bit-sliced weighted Σ vs a strided u8 gather + VNNI vs one pass per pattern, all checked against an i128 oracle.

**Order:** P-X2/X5, then P-X1/X6 and P-X4, then P-X11, then P-X7/X8, then the guards, then the benches.

## What was NOT searched

- DuckDB and Velox source (FastLanes, abstract only, stood in).
- ByteSlice beyond its abstract.
- The official Arm ARM and RISC-V spec text (both secondhand).
- Belnap/Kleene bit-field encodings (no source read).
- Gustafson's posit standard and LogNet (secondhand).
- The full bodies of `causal-plane-inventory.md` and `encoding-ecosystem.md`.
- The callers of `NarsTables::revise`.
- The users of `MaskedStridedGroupSum`.
- Every consumer that gates on Epi5.

"Nothing found" in these areas is not "nothing exists".

## Closeout

```
PR #<tbd> | STATUS: open | OUTCOME: exploration map + 5 .arxiv harvests;
premise split (CE64 = instruction word around a 16-bit truth number);
no code changed | OPEN: operator choice of probes; c=255 policy; NarsTables
caller census; Pearl-field ordinal readings to be guarded
```
