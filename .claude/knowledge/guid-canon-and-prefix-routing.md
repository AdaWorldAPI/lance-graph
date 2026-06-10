# GUID Canon + Prefix Routing — the policy side (crystallization)

> **READ BY:** integration-lead, truth-architect, family-codec-smith,
> palette-engineer, any agent touching `identity.rs`, `hhtl.rs`,
> `high_heel.rs`, `quorum.rs`, the ontology registry, or codebook builds.
>
> **Date:** 2026-06-10. **Canon source:** `OGAR/CLAUDE.md` (operator-pinned;
> cited, never forked). **Counterpart:** ndarray
> `.claude/knowledge/guid-prefix-shape-routing.md` (the mechanism side).
> Conjectures are labeled; probes named (no unmarked conjectures).

## 1. The canon (cited)

```
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx     32 hex = 128 bit = the GUID
classid    HEEL   HIP    TWIG   basin·leaf(6)+identity(6)
```

- **Key-of-key-value:** node = key(128) + value(3968) = 4096 bits. The key
  routes/resolves/compares/scopes/names with zero value decode.
- **3×4 uniform** (`tier = nibble >> 2`); RFC 9562 = wrapper concern;
  standing 3×4-vs-4×3 watch lives in `OGAR/CLAUDE.md`.
- **Centroid tile [H]:** path = 6 bytes = CAM-PQ 6×256; per-tier 256×256
  LUT distance; codebooks 4⁴-hierarchical; **scoped by class prefix**
  (longest-prefix wins).
- Wrapper audit direction: `contract::identity::NodeGuid` (#480) is the
  carving of this GUID — audited against the canon **group-by-group**,
  never the reverse. Groups 1–2 + the 24-bit `local` already match; the
  Phase B question is groups 3–4 yielding all eight nibbles to HIP/TWIG.

## 2. lance-graph owns POLICY (ndarray owns mechanism)

| Concern | Where it lands | Status |
|---|---|---|
| `(entity_type ↔ NiblePath)` bijection mint | ontology registry — Phase B | [H], planned (identity plan) |
| Per-class centroid codebooks (4⁴-hierarchical, prefix-scoped) | registry shelf, next to `ClassView`/`StructuralSignature` — minted with the class, trained once (amortized) | CONJECTURE → PROBE-CODEBOOK-44 (ndarray doc §6) |
| `PrefixShapeTable` registration (classid/prefix → `ShapeId`) | lance-graph builds the table from the registry; ndarray routes by it, never knowing semantics | CONJECTURE → PROBE-ROUTE-1 |
| Quorum certificate type | `contract::quorum` — the #411 scaffold (`todo!()`) is the named landing spot; `HighHeelBGZ`'s basin-merge L1-threshold consensus is the existing mechanism to generalize | CONJECTURE → PROBE-QUORUM-1 |
| Escalation on quorum fail | HHTL tier escalation (`bgz-tensor::hhtl_cache::RouteAction::Escalate` is the shipped precedent) | [G] mechanism / [H] wiring |

## 3. The anti-theater rules, contract side

The ndarray casebook (`pp13-brutally-honest-tester-verdict.md`) defines
eigenvalue theater: cheap arithmetic wearing metric/spectral language it
does not earn (unsatisfiable PSD gates; optimism thresholds; enforced
placeholders; unrun "verified" claims; raw-XOR-u64 as "nearest").

Contract-side consequences:
1. **Cheap-path answers carry a quorum certificate** — k-of-n φ-stride
   probes agree within τ; τ from measured anchors (ρ = 0.9973 HIP / 0.965
   TWIG; Pflug-10 palette certification) under **I-NOISE-FLOOR-JIRAK**
   (Jirak 2016 rates, never classical Berry-Esseen, never optimism).
2. **Metrics are named typed fns** (`cognitive-distance-typing.md`
   no-umbrella rule): popcount Hamming, palette L1 ADC — raw-XOR-u64
   ordering is the named anti-pattern.
3. **Quorum fail escalates a tier; never silently accepts.**
4. **`ShapeId` is a register key** (I-VSA-IDENTITIES Test 0): points to a
   shape; never bundled, never content-hashed.
5. **No spectral language on the cheap path.** PSD/eigen/Σ claims route to
   ndarray's pillar suite with relative tolerances and measured thresholds.

## 4. Probes (shared numbering with the ndarray doc)

PROBE-ROUTE-1 (batch parity + ≥4× bench) · PROBE-QUORUM-1 (accept ⇒ ρ ≥
anchor; reject ⇒ escalate) · PROBE-PHI-1 (φ-stride discrepancy beats
uniform) · PROBE-PYR-1 (perturbation pyramid byte-exact) ·
PROBE-CODEBOOK-44 (4⁴ vs flat-256 within Pflug band) · PROBE-HILBERT-L4
(ndarray P0-4 — blocks any L4 cascade-addressing claim until green).

## 5. Cross-references

`OGAR/CLAUDE.md` (canon) · `OGAR/docs/INTEGRATION-MAP.md` (seams S1/S7/S9;
gates F10–F14) · ndarray `guid-prefix-shape-routing.md` (mechanism) ·
`EPIPHANIES.md` E-IDENTITY-WHITEBOX-1 (the bijection + roundtrip_eq
whitening) · iron rules I-NOISE-FLOOR-JIRAK, I-VSA-IDENTITIES,
I-LEGACY-API-FEATURE-GATED.
