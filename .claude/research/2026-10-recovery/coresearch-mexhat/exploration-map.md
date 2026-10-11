# EXPLORATION MAP — Mexican-hat popcount serving deforestation (Phase 4, NOT landed; awaits operator)

Main path (from .arxiv/2605.01487): ring-wise C_k = popcount(M ∧ R_k(M)), R_k = OR of k-step Moore shifts via ndarray mask_shift_morton (enum direction, no wrap, Morton 4^k field; 256×256 = 4^5 words ✓). Counts are unsigned and exact; centre vs surround is decided in jc by u64 cross-multiplication. No kernel, no sign, no float, no materialised population.

| idea | verdict | why / next |
|---|---|---|
| X14 ring autocorrelation (shift∧popcount) | PROBE → likely ADOPT | ops already shipped; probe below |
| X15 cross-time W(Δ) | PROBE | direction comes only from temporal asymmetry |
| X6 cross-multiplication in jc | PROBE (MHP-3) → ADOPT | u64 is enough at 64K; u32 overflows at exactly 65536² |
| X16/X17/X18/X19 (materialization invariance, one widened fold, fold-before-mask, integer round-never) | ADOPT-NOW as review gates | laws only, no code |
| X8 census rank, single centre | PROBE (MHP-7) | runnable now; per-pixel reading conflicts with the anchor |
| X4 Range bands | PROBE (MHP-5) | bgz17 run-count kill arm |
| X11 palette-spread diffusion | PROBE gate (MHP-10) | needs a jc Jirak margin to give a PASS |
| X1 NestedBands | PARTIAL | AND_ANDNOT2 + with_boundaries only; calibrate/quantize_2z use f64/round (seal time); buckets are a second population copy |
| X3/X5 palette-LUT histogram | PARK | needs a widened population-sized fk lane, or a new strided-u8 Via kernel, plus a LUT copy; from_floor rounds; superseded by X14 for the spatial case |
| X7 smooth weights | PARK | only with integer-native, split-lobe (w+, w−) weights |
| X2 unimodality / κ | PARK | proposal; κ only as rational p/q by cross-multiplication |
| X10 single terminal | record | do not lift the one-terminal rule |
| X12 signed BSI / ±1 / float / Pos-Neg / geometric window | SKIP | contradicts the rulings |
| X13 bgz17/KNOWLEDGE.md:178 "semiring" | ADOPT-NOW (doc storno, needs approval) | it is a LUT |
| X20 Moore = predictor, not purpose | → 5+3 | weighs against the MooreNars16 Energy4 reading |

Named probes for X14 (to be written ONLY after approval):
- PROBE-MHP-R1: ring counts == scalar brute force over Morton cells (exact ==), n ∈ {4096, 65536}; anti-vacuity ≥3 non-empty rings.
- PROBE-MHP-R2: acf(M,Δ) == acf(M,−Δ) for all 8 Δ (symmetry); cross-time W on M_{t+1}=shift(M_t,Δ0) argmax == Δ0; static M invents no direction.
- PROBE-MHP-R3: allocation bytes(4096) == bytes(65536) == 0 beyond caller scratch (Counting allocator, mask-risc tests/no_alloc.rs); can-fire arm = row-id compaction.
- PROBE-MHP-R4: fold(mask) == fold(materialize(mask)).
- Disable runs: drop one ring shift / swap ∧ for ∨ / use a wrapping shift → each must go red.

Not searched: Druid/Lucene/FastLanes internals; CenSurE ECCV full paper; alphaXiv (auth missing). `.arxiv` was initially missed because I searched for `.archiv`.

## CORRECTION after the fit review of X14-X20 (supersedes the X14 row above)
- X14 → PARTIAL, not ALREADY-HAVE. mask_shift_morton writes a full dst plane (simd_masking_ops.rs:5384-5386) and cannot be tiled: a sub-span drops the boundary carry (:5352-5362). That is a materialisation under the tiled law (mask-risc lib.rs:16-22). MortonDir has 4 directions only, so a diagonal needs a second full-field scratch. mask-risc has no shift lowering at all.
- Fit path: a fused terminal popcount(M ∧ shift(M,dir)) as a NEW LOWERING over existing facade words: per word, AND the source word with the neighbour bits (interior case-shift + the one carry word), then popcount. No dst plane is written, and the scratch is O(1). This is the deforested form; it is a lowering, not a kernel.
- Ring definition must be chosen: pair count Σ_Δ C(Δ) vs cell count popcount(M ∧ ball_k \ ball_{k-1}) (Chebyshev, needs diagonals). They differ: OR collapses multiplicity.
- X15 needs a tie rule: compare W(Δ) vs W(−Δ) by unsigned ≥ / ≤; a tie gives explicit Inconclusive. Otherwise a static state reports a direction (T4).
- X14 sign check: PASS (enum direction, no wrap, u64 index arithmetic). Symmetry C(Δ)=C(−Δ) holds exactly.
- Revised probe MHP-R3: the fused lowering must allocate 0 bytes at both n; can-fire arm = the current plane-writing shift.

## Falsifier addendum (MHP-11..18) — banked
- F-a: mask_shift_morton OR-accumulates into dst; reusing scratch across directions silently mixes shifts. Caller must zero the scratch it owns (no alloc).
- F-b: direction sign (the reference maps +x to NegQ); symmetry tests cannot catch a flipped sign; MHP-13 pins it absolutely.
- F-c: ring = cells (popcount(M ∧ ball_k\ball_{k-1})) vs pairs (Σ C(Δ)); O_k ≤ P_k, strict on dense. OPERATOR RULING NEEDED.
- F-d: no shift instruction in mask-risc IR → quack/Program arm blocked; would be a new lowering.
- Runnable now: MHP-11 (symmetry), 12 (both arms), 13 (synthetic translation), 14 (static → Inconclusive), 15 (0 alloc independent of population), 16 A/C (materialization + row-major vs Morton invariance).
- Blocked: MHP-12 verdict (F-c), MHP-16 B (F-d), MHP-17 (promotion entry point + Energy4 ruling), MHP-13 envelope / MHP-18 narrowing (jc).
