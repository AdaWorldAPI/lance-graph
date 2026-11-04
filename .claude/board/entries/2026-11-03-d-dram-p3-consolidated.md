# D-DRAM-P3: consolidated findings, high-signal items, fruits and open ends (D-GLOBAL-WHERE P2 + D-DRAM-0 #1949 + Abakus register model)

**Status:** MEASURED (P2, P3-1), VERIFIED-IN-CODE (P3-3 census), WORKING-MODEL (Abakus register contract). Lab only; no production change. Owner: this session (plan `.claude/plans/d-dram-p3-integrated-v1.md`).

**Provenance:** #1949 (D-DRAM-0) was authored by a GPT session. Its gates, stop conditions and its statement of the operator's intent are that session's proposal, not verified fact or operator text. This record uses them only where this session checked them against source (the P3-3 census) or measured them (P2, P3-1). The Abakus register model in §2 comes from the operator directly, not from #1949.

## 1. Findings (measured or verified)

| # | Finding | Evidence | Grade |
|---|---|---|---|
| 1 | Semantic parity reached: every count/EXISTS answer equals the oracle and DuckDB 2.0.0 at 1M and 10M rows, threads 1 and 4. Every Lance arm equals the oracle. | `entries/2026-11-03-d-global-where-p2.md` | MEASURED-LAB |
| 2 | Count beats DuckDB 14.2× (1M, 1 thread) down to 1.96× (10M, 4 threads). EXISTS loses at 10M (0.43×, 0.16×) because `Terminal::Any` has no per-tile early exit. | Same entry | MEASURED-LAB |
| 3 | Lance 13 selective-read bytes follow page locality, not selectivity. A scattered 0.1% selection reads 33% of bytes at k=1 and 99.8% at k=16. A clustered 0.1% read is 112× faster than a full scan at 10M. | Same entry | MEASURED-LAB |
| 4 | RowAddrMask ingress is a copy that scales with ranges: 0.4–5.4 µs for 1 range; about 4,600 µs for 91,527 ranges; 123,162 µs for 3.26M ranges. Labelled `S_INGRESS_COPY` in #1949 C4. | Same entry | MEASURED-LAB |
| 5 | P3-1: `ReadPolicy` gained `page_rows`. A full read now requires dense coverage AND many ranges (`merged.len() * page_rows >= n_rows`). At 1M, clustered 10–50% now read ranges, 1.2–7.8× faster (10%, k=16: 103,492 → 13,292 µs). Scattered 10% still reads full. Disable runs: 2 tests red under coverage-only, 6 red with spread forced false. | `crates/crepe-plan-probe/src/addressed.rs` | MEASURED-LAB + TEST-PINNED |
| 6 | P3-3 census: ndarray ships exactly two carry words, `half_add_mask(a, carry) -> (u64, u64)` (`src/simd_masking_ops.rs:6523`) and `full_add_mask(a, b, c) -> (u64, u64)` (`:6545`), both re-exported at `src/simd.rs:1062`. `U24x5` (`examples/microtile/tile.rs:60`) and `Abacus<BITS>` (`examples/u9_domino/domino.rs:316`) are lab fixtures only. | `entries/2026-11-03-d-dram-1-census.md` | VERIFIED-IN-CODE |
| 7 | P3-3 census: no production focus/lease host exists. `FocusKey` (`drf-focus-probe`) and `FocusLease` (`lance-graph-nsl`) are in excluded lab crates. dt64's `Focus` is a tile revision. | Same entry | VERIFIED-IN-CODE |
| 8 | ndarray #402 cost gate (ns per add, AVX-512 VNNI): native per-lane 8.79–9.08, eager bit-plane 2.46–2.66, Abacus 2.27–2.35 on dense masks. On random masks all three are within about 0.75–1.1× (8.78–8.80 / 7.79–8.05 / 8.34–8.80). | ndarray `.claude/knowledge/binary-abacus-polyfill-probe.md` | MEASURED (ndarray, cited) |
| 9 | ndarray #389: bit-plane MULTIPLICATION is REFUSED as a replacement for the U9 byte-dot (best variant 5.2× slower, carry-save variants about 120×; no break-even for prepared operands). `carry_save3` (the full adder) is ADOPT-LAB. | ndarray `.claude/knowledge/u9-abacus-probe.md` | MEASURED (ndarray, cited) |
| 10 | ndarray microtile: bit planes win the FOLD for narrow fields (U9x14 fold 64K: 0.60–0.76 ns/tile vs 2.81–3.22 SoA) but not the add. Wide fields (U24x5) favour widened lanes. | ndarray `.claude/knowledge/microtile-probe.md` | MEASURED (ndarray, cited) |

## 2. The Abakus register model (WORKING-MODEL, not measured)

Direction: an Abakus kernel is repeated folding over bit planes at CPU speed. Each `u64` word is one significance column across 64 independent lanes. `half_add_mask` and `full_add_mask` move the carry one column up in the same lane. A fold keeps applying them without leaving registers.

Read this way, the plane stack is an ISA register contract: a fixed physical register shape, with ALU words over it, that the shader driver dispatches against and that ClassView readings interpret. This matches existing prior art:

- `.claude/plans/d-holo-h14c-thought-first-v1.md` §1: "The 16-byte lane arrangement is a generic register contract. CE64, SPOFC, NARS evidence, thought identity and the like are typed readings over it, not fixed meanings of its bytes."
- `.claude/plans/iam-substrate-sag-dir-sim-v1.md`: "don't move schema semantics into the 8-bit ALU/ISA register contract."
- `.claude/knowledge/fold-execution-laws.md` law 6: "No semantic convenience opcode. A new V4 op needs a proof that existing ops cannot express the computation faithfully and efficiently."

### What the measurements already say about it

Using findings 8–10 only:

- Carry/fold over planes is CPU-speed and beats per-lane adds about 4× on dense and single-lane masks.
- It does not beat per-lane on random masks.
- It loses badly as a multiplier.
- It wins folds of narrow fields.

### What it does not license

- No new opcode and no public `Abacus` type without a consumer (#402 deferred PR B for exactly that reason).
- No semantic meaning in register bits.
- Presence is not carried by the counters. A lane never touched and a lane that summed to zero both read 0, so a consumer keeps its own presence plane.

The A-family gates (A1, A2, A5–A7) proposed in #1949 are adopted here as the falsifiers of this model; P3-5 runs them on the two shipped words only.

## 3. High-signal items

1. **P3-5 k-of-n bench** on `half_add_mask`/`full_add_mask` vs native AND/OR/COUNT (A1, A2, A5–A7). It is the only test of the Abakus model on a real WHERE shape; no k-of-n measurement exists.
2. **P3-4 EXISTS early exit per tile.** It is the one DuckDB loss (0.43×, 0.16× at 10M). Production mask-risc, so a draft PR for its owner.
3. **P3-8 D4/S-INGRESS decision memo.** The integrated DR → Lance path is still unmeasured, and ingress cost (finding 4) is the input.
4. **P3-7 route once per frame.** Routing dominates `compose` (35,914 of 50,178 µs).

## 4. Fruits

The high-signal fruit has its own section in `.claude/board/IDEAS.md` ("k-of-n on the shipped carry words"). The others are tracked as plan items P3-2, P3-6 and P3-9.

- **Low-hanging:** P3-2 non-vacuous `project` k4/k16 cases (an oracle count of 0 cannot catch a wrong answer). `S_INGRESS_COPY` label in the Lance arm output.
- **High-signal:** P3-5 k-of-n Abakus bench (IDEAS: "k-of-n on the shipped carry words").
- **Heavy:** P3-6 ordered u8-in-u32 compare word (ndarray owner, needs codec attestation). P3-9 locator receipts (gated on EXO-46/89/151, no production lease host).

## 5. Open ends

- D-DRAM-2 locator receipts: GATED on EXO-46, EXO-89, EXO-151. No production lease host exists (finding 7).
- D-DRAM-4 end-to-end: GATED on D4 + S-INGRESS (#1788).
- Bit-plane add on random masks is not faster than per-lane (finding 8). Whether a k-of-n threshold inherits the dense-mask win or the random-mask parity is UNKNOWN until P3-5.
- Why the borrowed lane read is 2.3–2.5× slower at 50% selectivity: OPEN.
- Every Lance byte count was served from the page cache, not disk. Cold-disk behaviour is OPEN.
- ndarray microtile: SoA add is faster on AVX2 than AVX-512, not diagnosed (ndarray's own open point).

## 6. Stop conditions in force

The stop conditions proposed in #1949 §7 are adopted by this session after review (listed in the plan §4); in particular no public Abacus type, no new opcode, and no 5×24 API claimed as shipped.
