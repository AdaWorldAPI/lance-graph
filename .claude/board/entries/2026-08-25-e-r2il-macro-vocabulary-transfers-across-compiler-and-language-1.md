## 2026-08-25 — E-R2IL-MACRO-VOCABULARY-TRANSFERS-ACROSS-COMPILER-AND-LANGUAGE-1 — a macro vocabulary learned from two gcc binaries fires in unseen gcc code at −0.6% density and in unseen rustc code at −4.7%, both outside a marginal-preserving null

**Status:** FINDING — [MEASURED] (held-out transfer + two pre-registered
shuffle nulls, 20 seeds each, over 4 binaries in 3 corpus configurations;
the shipped `probe_bpe_r2il_loco_microcode.rs` instrumented for the runs
and reverted — no source change landed, the probe's own 10 gates stayed
green throughout).
**Confidence:** High for the mechanism on this corpus. The generalization
beyond x86-64 / the pass-1 seven-opcode convention / chain-length-3 is
NOT measured and is not claimed.

### The question this answers

`E-BPE-OVER-DEFUSE-CHAINS-BEATS-LINEAR-AND-FITS-LOCO-1` measured 33 BPE
macros over R2IL def-use chains and their `ogar-loco` lane fit, on **2
binaries**. It did not ask whether a macro means the same thing anywhere
else. That question is load-bearing for any design that treats a macro as
a placeable, addressable unit: a unit whose meaning is program-local is
content, not vocabulary.

### The measurement

Train on the two `stress_test` binaries ONLY (1,872 def-use chain
occurrences, 33 macros). Score each unseen corpus by macro-hit **density
per chain** — scale-free, so corpora of different size compare directly.

| held-out | chains | density | vs train | coverage | macros hit |
|---|---|---|---|---|---|
| *(train: 2 × gcc `stress_test`)* | 1,872 | **2.529** | — | — | — |
| `vuln_test` — different C program, same gcc | 608 | **2.515** | **−0.6%** | 608/608 | 31/33 |
| `build-script-build` — serde_json, rustc/LLVM | 8,659 | **2.409** | **−4.7%** | 8,652/8,659 | 33/33 |

### Both nulls, pre-registered before the runs

- **Global null** — permute atoms ACROSS held-out chains; the global atom
  multiset is preserved EXACTLY (asserted on every one of the 40 draws),
  def-use adjacency destroyed.
- **Column null** (the one that does the work) — permute each POSITION
  column among itself. Each position's opcode marginal is preserved
  exactly, so *"position 0 is usually `int_add`"* cannot explain a win;
  only which x, y and z sit TOGETHER is destroyed.

| held-out | REAL | global null | column null | REAL in either range? |
|---|---|---|---|---|
| `vuln_test` | 2.515 | 1.664 [980..1034 hits] | 1.969 [1179..1232 hits] | **no / no** |
| Rust | 2.409 | 1.780 [15278..15496] | 2.009 [17327..17461] | **no / no** |

Margin over the strict null: **+27.7%** (same compiler) → **+19.9%**
(across the language boundary). The cost of crossing is real and small;
the vocabulary is NOT a gcc idiom.

Distribution-free ranges are reported deliberately: `I-NOISE-FLOOR-JIRAK`
forbids a classical σ claim here (weakly dependent bits), and
non-overlap of 20-draw ranges needs no such assumption.

### ⊘ My own pre-registration was WRONG, in the hypothesis's favour

Before the Rust run I recorded: *"Erwartung: schwächerer Transfer …
spürbarer Abfall der Dichte"*, and named the kill condition (REAL inside
the column-null range ⇒ the palette must be split per toolchain).
Measured: 4.7%, an order of magnitude smaller than "spürbar" implied, and
the kill condition did not fire. Recorded because a prediction that misses
in the direction you wanted is exactly the one a later reader must be able
to check.

### Three deflations of the headline, stated rather than buried

1. **33/33 is partly a sample-size effect, NOT a stronger transfer than
   31/33.** The two macros that missed `vuln_test` carry 6 and 2 training
   occurrences and had 608 chances there against 8,659 here. Density is
   the honest statistic, and it correctly shows the Rust corpus as the
   HARDER one.
2. **The Rust sample is capped.** `R2IL_HARVEST_MAX_FUNCS=200` exhausted
   at 200 of 548 `STT_FUNC` symbols — a first-200 slice with std/core
   prelude bias. The three C binaries sat under the cap and were harvested
   whole, so the four probes are not sampled identically.
3. **Coverage saturates and should not be quoted alone.** 608/608 looked
   tautological for a 7-symbol alphabet; the column null reaches only
   564/608 (92.8%), which is what rescues it — but the margin is thin and
   density carries the finding.

### What this licenses, and what it does not

**Licenses:** treating the R2IL macro vocabulary as a *shared* palette
rather than a per-binary or per-toolchain one — the `System[256] |
Learned[≤256] | Explore[≤256]` shape of the POC entry, with one mint
serving multiple programs. It is the empirical half of "R2IL is the
faithful vocabulary, BPE is recombination over it".

**Does not license:** any mint (none performed), any claim about
architectures other than x86-64, any claim that a macro is *semantically*
the same across languages — this measures co-occurrence of an opcode
pattern, not that the pattern means the same thing to a reader. A
`(int_add, copy, store)` chain in Rust and in C are the same SHAPE; that
they are the same THOUGHT is a separate, unrun question.

**Corpus:** `stress_test` (10,003 rows) · `stress_test_opt` (7,554) ·
`vuln_test` (4,409, 41 fns) · `build-script-build` (72,567, 200 of 548
fns). Harvested through `ruff_r2il`'s `harvest_r2il` (`--features lift`).
The probe's B7 fence — which recomputes the binary count live rather than
trusting a constant — correctly failed the moment a third binary entered,
which is how the corpus swap was confirmed real.

