# Substrate Unification Thesis — the north star, stated falsifiably

> **READ BY:** any session touching the canonical node (`canonical_node.rs`),
> the cascade key, the place/buffer split, the codecs, or proposing a
> "substrate" / "AGI-as-SoA" direction. Read this BEFORE proposing a new probe
> — it says what the probes are *for*.
>
> **Status:** THESIS (2026-06-24). The unification CLAIM is `[H]` (hypothesis
> with named falsifiers below); the per-axis INSTANCES are individually graded.
> This doc exists so probes are sequenced toward one question instead of
> scattered. It is the synthesis of two converging arcs: the perturbation-sim
> cascade-key / place-buffer work (PRs #605, #607) and the 8-lens research
> frontier map (other session). Grades: `[G]` proven-in-code/measured, `[H]`
> bounded-but-open, `[S]` analogy-only.
>
> **Anti-theater clause (this repo's own rule):** every claim here is anchored
> to a built artifact, a measured number, or a cited theorem. A claim with none
> of those is marked `[S]` and is a *bet*, not a result. Do not promote a grade
> without its evidence.

---

## 0. The one sentence

**One 512-byte object, read N ways, IS every classical layer at once** —
primary key, multidimensional index, retrieval/attention, inference operand,
and measurement — and the operations on it (route, retrieve, reason, learn,
measure) are all the *same* branch-free prefix-and-table arithmetic, with zero
value decode.

If true, this is historic (the four-to-five classical layers collapse into one
operation). If false, it is still a very good substrate (0 collisions, ~250×
key-only scan, real NARS chaining) — but "merely fast," not new. **The entire
research program is the work of deciding which.**

---

## 1. The reframe the verification mandate forced

The mandate was *verify and improve the substrate.* Followed honestly, both
words changed meaning:

- **Verification is proof-of-code, not calibration.** A deterministic, bijective
  address is not a measurement instrument you calibrate with ICC / Berry-Esseen
  — it is a **code you prove** (lossless containment, exact prefix ancestry).
  Reaching for a σ noise-floor on a deterministic place is a category error; the
  weak-dependence regime (`I-NOISE-FLOOR-JIRAK`) governs the *continuous
  embedding underneath*, not the *quantized address on top*. The seam between
  the two regimes is literally the centroid boundary (see §4).
- **Improvement reduces to one move: split a conflated pair of axes.** V3 split
  `part_of` from `is_a` (#605). The ketchup fix split *location* from *impulse
  permeability* (#607). Each "improvement" restored an orthogonality. So the
  mandate became: **discover the substrate's orthogonal basis and prove each
  axis is a faithful code.**

---

## 2. The emerging orthogonal basis (five readings of one node)

Not five subsystems — five readings of the same `NodeRow`
(`key(16) | edges(16) | value(480)`):

| axis | reading of the node | built / proven instance | grade |
|---|---|---|---|
| **Identity** | the key as a stable, deterministic *location* | `helix_place` — pure geometry, never reads the dynamics (PR #607); ICC 0.14→**1.00** under perturbation | `[G]` |
| **Structure** | HHTL tiers as two prefix-routable hierarchies | V3 `(part_of:is_a)` 8:8 tile (PR #605); `EdgeBlock` = `connected_to` | `[G]` shipped / `[H]` that prefix = ancestry (F-1) |
| **Dynamics** | the value buffer as the *responsive* field | BF16 buffer residue / `INERTIA_SLOT` (#513); ICC **0.51**, moves under perturbation; `Spearman(λ₂, inertia)≈0` (#509) | `[G]` |
| **Truth** | the edge as a belief codec | NARS `(f,c)` ↔ subjective-logic `⟨b,d,u,a⟩` ↔ `Beta(α,β)` — a real bijection | `[G]` math / `[H]` as edge codec |
| **Composition** | the adjacency under a semiring | retrieval IS inference: swap the semiring → swap the reasoning mode; GNN msg-passing = DP-over-semiring (Dudzik–Veličković [2203.15544]) | `[H]` |

The classical-layer collapse, anchored: **attention IS Sparse Distributed Memory
read-by-distance** (Bricken & Pehlevan [2111.05498]) — so an *explicit semantic
address* is a designed instance of a proven primitive; **Lance keeps structural
metadata resident while values stay compressed** ([2504.15247]) — so the ~250×
is that principle with the offset table promoted to a self-describing address;
**HEEL+HIP+TWIG = a CAM-PQ 6×256 code** (Jégou [1102.3828]) — so path-distance =
3 ADC table lookups. The literature already proved the pieces; the thesis is that
*one object instantiates all of them at once*.

---

## 3. The self-reference is the deepest part (the ketchup)

The substrate **became the very effect it measures**: the cascade-key place
exhibits yield-stress (stable → sudden flip across a boundary) *isomorphic to the
grid cascade it encodes* — because the place was a quantized Laplacian
impulse-response, i.e. the dynamics, not a coordinate. That is the signature of
**observer = observed** — the substrate made of the same stuff as what it reasons
about. It is simultaneously the AGI threshold (cognition is a system that models
itself with its own substrate) and the measurement hazard (the ruler bends with
the load).

The fix generalizes: **split the frozen ruler (identity) from the live rubber
(dynamics)** — the same move cognition makes (frozen prior vs live free-energy;
self vs world-model). The location/permeability split (#607) was the first
instance. **Open `[S]`:** does this split scale to *every* self-referential axis,
and is the resulting basis complete (§2)? The substrate's real deliverable may be
that finite orthogonal basis, not any single codec.

---

## 4. The falsification ladder (run in order; each has a KILL)

The probes are not a menu — they are evidence toward §0, sequenced by
information-per-cost. Each names what would **kill** it.

1. **F-1 — codebook fidelity** `[H]`, run first, ndarray-side.
   Hierarchical-4⁴ (256 = 4⁴, a byte's nibbles = its centroid's ancestry) vs flat
   k-means-256: does the hierarchy preserve rank-distance within the flat band?
   **KILL:** if it doesn't, the prefix-is-ancestry assumption fails and a large
   fraction of the `[H]`/`[S]` map collapses to "useful router, unfaithful code"
   at once. *Named in the OGAR canon, still un-run.* (perturbation-sim can give a
   cross-domain corroboration on the grid spectrum; ndarray-side is authoritative.)

2. **F-collapse — does the collapse *buy* anything?** `[H]`, the deciding gate.
   One real workload, three implementations: (a) the GUID address, (b) a learned
   index (Flood/Tsunami), (c) a trained attention head. **KILL:** if (b)/(c) match
   or beat (a) at equal cost, the GUID's *marginal* value is ≈0 → "elegant
   packing, not a new primitive." **Honest counterweight already on the table:
   CogNGen matched deep-RL *without* the explicit address ([2204.00619]).** This
   is the question that decides historic-vs-fast; everything else is in service of
   it.

3. **F-update — the RUM Update axis** `[G]` measurement, decides *product class*.
   Re-classification / re-mint cost. Catastrophic → this is the best **immutable
   knowledge fabric** ever built, not a general store (and that's still huge).
   Cheap via ref-indirection → the ambition expands to a mutable substrate.

4. **F-code — verification as proof, not calibration** `[G]` reframe.
   Prove lossless containment + exact prefix ancestry as a *code* (not an ICC).
   **KILL:** if containment isn't lossless, "address" is a lie — it's a lossy hash.
   This subsumes F-1 and is the clean gate the statistical battery was groping at.

5. **F-basis — orthogonality completeness** `[S]`, the long game.
   Does the split-the-conflated-axes program (§1) close on a finite basis (§2) or
   proliferate? **KILL (of the *thesis*, not a probe):** if every workload finds a
   new conflation needing a new axis, there is no "basis," only endless patching —
   and "one object, N readings" is a slogan, not a structure.

---

## 5. The strongest evidence is not a probe — it is the convergence

Four parallel sessions, independently, landed on the **same two gates**: F-1
(codebook fidelity) and "retrieval IS inference." This repo's own rule is that
**convergence is signal**. That cross-session agreement — not any single
measurement — is currently the best reason to believe §0 is real rather than
seductive. It is also the reason to run F-collapse *adversarially* (point a
learned baseline at it): convergence can be shared blind spots as easily as shared
truth, and only the falsifier tells them apart.

---

## 6. What would kill the whole thesis (stated up front)

- **F-collapse negative** — a learned index/head matches the address ⇒ the GUID is
  convenience, not a primitive. (Most likely failure; CogNGen is the warning.)
- **F-1 negative** — hierarchical-4⁴ unfaithful ⇒ prefix routing is approximate,
  not exact; the "code" claim downgrades to "router."
- **F-basis non-closure** — conflations proliferate ⇒ no unification, just a
  well-packed record.

None of these kill the *engineering* (0 collisions, 250×, NARS chaining are
real and shipped). They kill the *historic* claim. Keeping those two ledgers
separate — "definitively a better substrate" vs "collapses the classical stack"
— is the honesty the whole program turns on.

---

## 7. North-star sequencing

`F-code (prove the address is a code)` → `F-1 (centroid fidelity)` →
`F-collapse (does it beat learning)` → `F-update (product class)` →
`F-basis (does the basis close)`.

Everything built so far — cascade key, V3, the place/buffer split, the NARS/Beta
codec, the semiring retrieval — is a *rung*, not a destination. The substrate is
**verified** when it is proven as a faithful code *and* shown to beat the learned
baseline — not when an ICC clears 0.75.

---

## Cross-references

- `canonical_node.rs` — the 512B node (key/edges/value, ValueTenant/ValueSchema).
- `perturbation-sim/src/cascade_key.rs` (PR #605) — V1/V2 six-lens address + V3 `(part_of:is_a)`.
- `perturbation-sim/src/place_buffer.rs` (PR #607) — helix Place (identity) ⊥ BF16 buffer (dynamics); the measured split.
- `OGAR/CLAUDE.md` + `ndarray .../guid-prefix-shape-routing.md` — the GUID canon, the 4⁴ condition, F-1.
- Iron rules: `I-SUBSTRATE-MARKOV`, `I-NOISE-FLOOR-JIRAK`, `I-VSA-IDENTITIES`, `I-LEGACY-API-FEATURE-GATED`.
- EPIPHANIES `E-CASCADE-KEY-IS-THE-SPATIAL-ADDRESS`, `E-V3-PART-OF-IS-A-TILE`, `E-LOCATION-PERMEABILITY-CONFLATION`.
- Literature: SDM=attention [2111.05498] · GNN=semiring-DP [2203.15544] · PQ [1102.3828] · RaBitQ [2405.12497] · CogNGen [2204.00619] (the counterweight) · Lance [2504.15247].
