# PROBE-HHTL-INTAKE-BLINDNESS (PROBE-CLAM-VS-HELIX-RESIDUE leg 0) — execution record

Follows the merged `PROBE-WORDNET-44-ACTIVATION` (#875,
`E-WORDNET-MAKES-THE-4-ARY-ADDRESS-SEMANTIC-1`). Operator ruling fixed the
division of labour: the 4⁴ fold is the ADDRESS, **CLAM is the established
CALCULATOR**, **HHTL + helix residue** is the alternative — *"exact address,
exact spacial location / perturbation"*.

## Files
- NEW `crates/helix/examples/probe_hhtl_intake_blindness.rs` (3 gates).
- Board: `EPIPHANIES.md` prepend, this file. Same commit (board-hygiene rule).

## Core-First compliance
CONSUMES `helix::curve_ruler::CurveRuler` — nothing re-derived. helix is in-tree
at `crates/helix` (standalone crate, own workspace) and already ships the seam:
`from_hhtl(path, depth)` → place; `index(k) = (start + STRIDE·k) mod MODULUS`
regenerates the interior; `ResidueEdge`/`Signed360` are the stored residue.

## The sweep was NOT RUN — and that is the finding

The queued hypothesis (finer address ⇒ more deterministic place ⇒ less stored
residue) was gated behind two PRE-REGISTERED failure modes. Both were run first,
exactly as the brief required. One is fatal to the sweep; the other is NOT CONFIRMED (a suspicion retracted, not a hypothesis disproven).

### H1 — the carving cannot reach the ruler (failure mode (a): CONFIRMED, fatal)

Two halves. **Type-level (stated, not measured):** `from_hhtl(path: u64,
depth: u8)` has no parameter that could carry a carving, so level structure has
no channel into `from_place(p) = p % 17`. **Empirical (the gate):** the entire
difference between the two carvings, as the ruler sees it, is a **constant
rotation of 2 across all 256 cells, variance 0** — a cell-dependent shift is the
falsifier. Unsorted histograms:

```text
4-ary  (depth 4): [15,15,15,15,16,15,15,15,15,15,15,15,15,15,15,15,15]
16-ary (depth 2): [15,15,16,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
```

> **⊘ Gate corrected pre-merge (codex P1 on #876).** The first version sorted
> both histograms and compared multisets. Sorting discards which cell maps to
> which offset, and the UNSORTED histograms genuinely differ — so the gate
> passed while hiding that every cell had moved. An aggregate distribution does
> not establish per-cell invariance; the constant-shift measurement does.

**Consequence: a residue-vs-granularity sweep cannot carry signal.** Its null
would have been a false negative caused entirely by the intake, and reporting
"4⁴ doesn't help the calculator" from it would have been wrong. Reported as an
INTAKE finding, per the pre-registration. The 4⁴ ADDRESS result is untouched.

### H2 — stride 4 vs the φ stride 11 (failure mode (b): NOT CONFIRMED)

The suspicion: `gcd(4,17)=1` buys coverage, not low discrepancy; stride 4's
first four steps `0,4,8,12` are one column of the 4×4 raster, while the φ stride
over 17 is 11 (`17/φ = 10.51`; C5 pins 11/17 as the golden step). Since HHTL
terminates early, short prefixes are what get consumed.

Measured gap spread (max−min gap; lower = more uniform):

| n | stride 4 | gaps | stride 11 (φ) | gaps |
|---|---|---|---|---|
| 2 | 9 | [4,13] | **5** | [6,11] |
| 3 | 5 | [4,4,9] | **1** | [5,6,6] |
| **4** | **1** | **[4,4,4,5]** | 5 | [1,5,5,6] |
| 5 | **3** | [1,4,4,4,4] | 4 | [1,1,5,5,5] |
| 6 | **3** | [1,1,3,4,4,4] | 4 | [1,1,1,4,5,5] |

At n=4 stride 4 is near-perfect (spread 1) and beats φ (spread 5); `4·4 = 16 ≈ 17`.

> **⊘ DESIGN CLAIM WITHDRAWN (codex P1 on #876).** The first version concluded
> "the shipped constants are MATCHED to the 4-ary use case". That premise — that
> a 4-ary tier consumes 4 ruler steps — is **not wired into anything that
> ships**: `ResidueEncoder::encode` reads only `start_offset` (n=1,
> `residue.rs:157`); `index`/`arc` appear solely in `walk_spectrum` and this
> crate's tests; the shipped `NiblePath` is `FAN_OUT = 16`. The n=4 comparison is
> therefore a property of a HYPOTHETICAL consumer. What survives: the
> pre-registered suspicion is **NOT CONFIRMED** — a retraction of a concern, not
> a vindication of a design.

Operator framing that produced this check: *"4x4 usually only helps with an
irrational stride as Pythagorean comma."* Confirmed in mechanism — stride 4 over
**16** covers only 4/16 (retraces one column forever); the `+1` is the comma.

### H3 — the blindness is FIXABLE at the intake, not fundamental

Per-tier seeding (one ruler per 2-bit group, folded with its level) preserves
ancestry by construction. **EXHAUSTIVE over all 256·255/2 = 32,640 cell pairs —
no sampling, no thinning.** Eligible population = **8,064** pairs sharing ≥1
address level: **flat intake preserves ancestry in 360/8,064 = 4.4643 %** against a base rate
of **1,800/32,640 = 5.5147 %** over ALL pairs — Δ = **−1.05 pp**, i.e. the flat
seed gives **no ancestry lift at all** and is in fact mildly ancestry-AVERSE
(cells sharing an address prefix are numerically close, so they are *less* likely
to be congruent mod 17). **Per-tier: 8,064/8,064 = 100 % BY CONSTRUCTION**,
asserted as a construction check rather than as evidence.

The acceptance criterion is a **declared null model**, not a threshold: H0 = "the
flat seed carries no ancestry information", so the eligible rate must not EXCEED
the base rate. Falsifier: an eligible rate materially above base, which would
mean the flat seed *does* carry ancestry.

> **⊘ THRESHOLD REPLACED (CodeRabbit on #876).** The gate was `flat_rate < 0.25`
> — hand-tuned, and 24.9 % would have passed it while the prose claimed ≈1/17.
> Replaced with the null-model comparison above. Running it immediately FAILED
> the first formulation ("the two rates coincide"), which was itself wrong: the
> rates do not coincide, they diverge by −1.05 pp. The gate now states the true
> and weaker claim — no lift — and the direction is reported rather than assumed. The unblock is to feed the ruler the
address's HIERARCHY, not its integer value.

> **⊘ SAMPLE SIZE CORRECTED TWICE (codex P2, then CodeRabbit, both on #876).**
> (1) The counter incremented BEFORE the `k > 0` eligibility check, so the
> published "4,662 pairs sharing ≥1 address level" was really every thinned pair.
> The ≈23× ratio was arithmetic on a mislabelled denominator. (2) The *fix* then
> repeated the error one level up: 1,152 was the eligible **thinned sample**, not
> the eligible **population**. Root cause of both: a `(a+b) % 7` thinning that
> bought nothing — 32,640 pairs is trivial to enumerate. **The thinning is
> removed**, so there is no sample left to mislabel; the figures are now the full
> population, 360/8,064 = 4.5 %, independently reproduced by the reviewer.

## Verification
- `cargo run --release --manifest-path crates/helix/Cargo.toml --example
  probe_hhtl_intake_blindness` → ALL GATES GREEN.
- `cargo clippy --manifest-path crates/helix/Cargo.toml --example
  probe_hhtl_intake_blindness -- -D warnings` → clean.
- `cargo fmt --manifest-path crates/helix/Cargo.toml` → applied.

## A self-caught vacuous gate (kept in the record)
H3's first run sampled **0 pairs** and correctly FAILED: `for a in 0..CELLS as u8`
where `CELLS = 256`, and `256 as u8 == 0`, so the loop was empty. Had the gate
been written to pass on an empty set it would have reported success on zero
evidence. Iterating in `u16` fixed it. Fourth instance in this arc of a defect
living in the falsifier rather than the measurement.

## Honest boundaries
- H1/H2/H3 are properties of the RULER (integer arithmetic over 17). No residue
  was encoded, no CLAM comparison ran, no cost was measured.
- Gates (1) AGREEMENT and (2) COST from the #66 brief remain NOT RUN — they are
  blocked on H1 being addressed, not skipped.
- H2 compares two strides on gap-spread only; it does not claim stride 4 is
  optimal, only that it is better than φ at n=4..6 and therefore not the defect
  the pre-registration suspected.

## Follow-ons
- Decide whether to extend the ruler with a per-tier intake (H3 shows it works).
  That is a helix API change and needs its own review — NOT done here.
- Once the intake sees hierarchy, the original residue-vs-granularity sweep
  becomes meaningful and gates (1)+(2) can run.
