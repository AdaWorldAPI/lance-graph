# PROBE-HELIX-LEAF-RESIDUE-VS-JINA

> **Status:** NOT RUN — CONJECTURE. No code built. This is a probe *spec*, parked
> per the workspace insight-update cycle (Claim → Probe defined → … → run → record).
> **Priority:** P2 (research / forward; gates the entropy_ladder fork's premise).
> **Date:** 2026-06-17.
> **Cross-ref:** `bf16-hhtl-terrain.md` (HHTL cascade, probe-queue discipline);
> `ndarray::hpc::entropy_ladder` (`residue_surprise` / `fork_decision`, PR #221);
> iron rules `I-NOISE-FLOOR-JIRAK`, `I-VSA-IDENTITIES`.

---

## Claim under test (CONJECTURE)

The **experienced** qualia layer carries signal **orthogonal to** the **observed**
(Jina) embedding — i.e. there is a "felt" component (the CMYK-K / accumulated
channel) that the Jina-measured "intensity" (RGB) view does not capture. Operationally:
when an experienced-qualia state is expanded to a VSA fingerprint and addressed via
the HHTL cascade (HEEL→HIP→TWIG→**LEAF**), the **LEAF residue** — the part not
explained by the deterministic coarse-tier *place* — has variance that is **not
predictable from Jina**. If true, that residue is the legitimate trigger for an
`entropy_ladder` domain fork (the original "orthogonal helix-leaf residue → Friston
free energy → HHTL domain shift" idea). If false, the fork is firing on noise.

## Why this is the right shape (tool-fit, settled in dialogue)

- HHTL is a **high-D fingerprint cascade** (95%-skip payoff at 16K-bit). Raw
  experienced qualia (4-texture / 4-quaternion / 17D-felt) is **too low-D** for HHTL
  to buy anything. So the probe **must** first expand felt → role-bound VSA fingerprint
  (16K), *then* take the HHTL LEAF as the residue. Running HHTL on the raw 4–17D vector
  is a category error and is explicitly out of scope.
- `place` = coarse tiers (deterministic, regenerable from the address — the helix /
  coprime-stride walk); `residue` = LEAF (the stored remainder). This is the OGAR
  place/residue split, not a new mechanism.

## Method

1. **Inputs (content-neutral):** the experienced-qualia vectors for the *neutral*
   glyph domains only (soulmap / mindmap structural archetypes) + synthetic felt
   vectors. NO NSFW payload. N ≥ 128 distinct felt states.
2. **Expand:** felt vector → role-bound `Vsa16kF32` fingerprint (existing
   `grammar/role_keys` style binding; no new carrier).
3. **Address:** HHTL cascade → (`heel`, `hip`, `twig`, `leaf`). `place` =
   reconstruction from HEEL+HIP+TWIG centroids; `residue` = fingerprint − place
   (or the LEAF Slot-V residual already defined in the cascade).
4. **Bridge to Jina:** for the same items, take the Jina embedding (the existing
   `ada:glyph:{byte}` 1024D link / the r=0.9913 Hamming↔cosine calibration). Regress
   the felt fingerprint (or its place) against Jina; keep the **Jina-explained** part.
5. **Partial correlation:** compute `ρ_partial(leaf_residue, X | jina)` — the
   association between the LEAF residue and any internal structure X (e.g. the felt
   axis it was generated from), **controlling for Jina**. Also compute the residue's
   internal autocorrelation under the coprime stride (does it have structure at all,
   or is it white noise?).

## Pass / Fail / KILL

| outcome | condition | meaning |
|---|---|---|
| **PASS** | LEAF residue is **Jina-orthogonal** (`|ρ(residue, jina)|` below the Jirak floor) **AND** internally **structured** (autocorrelation / partial-ρ with the generating felt axis above the Jirak floor) | experienced layer carries real signal the observed layer misses → CMYK-K confirmed → the `entropy_ladder` fork has a real premise |
| **FAIL** | LEAF residue is **fully Jina-explained** (`ρ(residue, jina)` high) | the "experienced" residue is just re-encoded observed signal → no orthogonal felt channel; demote the CMYK-K claim |
| **KILL** | LEAF residue is **unstructured noise** (no internal autocorrelation above the Jirak floor) | the fork would be chasing noise; `fork_decision`'s residue threshold is firing on nothing → re-derive the threshold or drop the fork-on-residue path |

## Hard constraints (workspace iron rules)

- **I-NOISE-FLOOR-JIRAK:** the fingerprint bits are weakly dependent. Every
  significance claim ("residue is N σ above floor", "ρ ≠ 0") **must** cite the
  Jirak-2016 weak-dependence rate (`n^(p/2-1)`), **not** classical IID Berry-Esseen.
  A non-zero residual variance is NOT automatically signal — that is exactly what the
  KILL condition guards.
- **NaN guard (the one we already hit, PR #221):** the residue-magnitude /
  `residue_surprise` computation underflows (`MIN_POSITIVE² → 0 → 0/0`) unless the
  span is floored *after* the subtract. Reuse the entropy_ladder floored-span guard;
  finite-assert the residue before any ρ computation.
- **I-VSA-IDENTITIES:** the felt state is expanded by binding to role *identities*,
  never by superposing content registers. The Jina link is a *bridge for comparison*,
  never mixed into the fingerprint.

## Harness (when run — NOT built yet)

- ndarray `entropy_ladder` (residue magnitude, floored span) + the existing HHTL
  cascade (`bgz-tensor/src/cascade.rs`) + a Spearman/partial-ρ helper (mirror the
  one in `lance-graph-contract/src/qualia.rs` tests, Jirak-floored threshold).
- Deterministic seeding (SplitMix64) per the certification-officer pattern; real
  Jina vectors via `hydrate --download` or the baked lens; neutral glyph inputs only.

## Update protocol

On run: record PASS/FAIL/KILL here + in the `bf16-hhtl-terrain.md` probe queue; if
PASS, the `entropy_ladder` fork-on-residue path is promoted CONJECTURE→FINDING; if
KILL, the hand-tuned `fork_decision` residue threshold is flagged as
noise-floor-ungrounded and re-derived from the Jirak bound.
