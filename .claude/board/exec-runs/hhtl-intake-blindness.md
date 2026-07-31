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
exactly as the brief required. One is fatal to the sweep; the other is REFUTED.

### H1 — the intake is STRUCTURE-BLIND (failure mode (a): CONFIRMED, fatal)

`from_place(p) = p % 17` consumes the address as a **NUMBER**, never as a
hierarchy. Feeding the same 256 cells carved 4-ary (depth 4) vs 16-ary (depth 2)
yields the SAME start-offset multiset — a pure rotation:

```
4-ary  (depth 4): [15,15,15,15,16,15,15,15,15,15,15,15,15,15,15,15,15]
16-ary (depth 2): [15,15,16,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
identical multiset = true
```

**Consequence: a residue-vs-granularity sweep cannot carry signal.** Its null
would have been a false negative caused entirely by the intake, and reporting
"4⁴ doesn't help the calculator" from it would have been wrong. Reported as an
INTAKE finding, per the pre-registration. The 4⁴ ADDRESS result is untouched.

### H2 — stride 4 vs the φ stride 11 (failure mode (b): REFUTED)

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

**At n=4 — the prefix a 4-ary tier actually consumes — stride 4 is near-perfect
(spread 1) and beats φ (spread 5).** `4·4 = 16 ≈ 17`: the comma is what makes a
4-step prefix tile the circle. The shipped constants are MATCHED to the 4-ary
use case. My suspicion ("a raster wearing a φ-spiral's docstring") is refuted;
this vindicates `STRIDE = 4` / `MODULUS = 17` rather than indicting them.

Operator framing that produced this check: *"4x4 usually only helps with an
irrational stride as Pythagorean comma."* Confirmed in mechanism — stride 4 over
**16** covers only 4/16 (retraces one column forever); the `+1` is the comma.

### H3 — the blindness is FIXABLE at the intake, not fundamental

Per-tier seeding (one ruler per 2-bit group, folded with its level) preserves
ancestry by construction. Over 4,662 sampled cell pairs sharing ≥1 address level:
**flat intake preserves ancestry in 51; per-tier in 1,152 (≈23×).** The unblock
is to feed the ruler the address's HIERARCHY, not its integer value.

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
