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

## 8. The strong form — the substrate as a full-stack compiler `[H]`/`[S]`

§0–§7 read *one* node five ways. The strong form asks: what if the **value
slab itself** is homogeneous in the same algebra as the key — and the `classid`
in the key is not just a router but a **schema pointer**? Then the 512-byte node
stops being a record and becomes a *compilation unit*: data → index → schema →
view, all from one self-describing block. This is the most ambitious reading;
it is graded `[H]` where it reuses shipped structure and `[S]` where it bets on
unbuilt tooling. **Nothing here is canon yet** — it is the north-star's far end,
written down so the rungs point somewhere.

### 8.1 The homogeneous facet `[H]` — layout-preserving, not layout-breaking

Carve the 480-byte value as **N × 16-byte facets**, each facet itself a
`(part_of:is_a)` cascade in the §2-Structure algebra:

```
facet (16 B) = facet_classid(4) | 6 × (8:8 part_of:is_a tile, 2 B each = 12)
value (480 B) = up to 30 homogeneous facets        ← ValueSchema::Homogeneous
```

The key insight is **compatibility, not replacement**: this is a new
`ValueSchema::Homogeneous` variant *alongside* the existing `ValueTenant` SoA
columns — the 16/16/480 split (`canonical_node.rs`) is **untouched**, so it is
layout-preserving and needs no `ENVELOPE_LAYOUT_VERSION` bump (contrast a *key*
re-carve, which is canon-level and separate). The value's facets are read by
the *same* prefix-and-table arithmetic as the path tiers (§2), so "read the
value" becomes the same operation as "route the key."

**The conflation trap, named up front (`[H]` gate):** not every facet is a
`part_of:is_a` mereology/taxonomy pair. Scalars (a susceptance, a price, a
timestamp) are *not* hierarchical — forcing them into an 8:8 tile is exactly
the §1 "split a conflated pair" error run in reverse. The honest form: scalar
facets carry **PQ codes** (Jégou [1102.3828]) in the same 16 bytes, and the
`facet_classid` discriminates codec-per-facet. **This is gated on F-1**
(codebook fidelity): if the centroid hierarchy isn't a faithful code, a PQ
facet is a lossy hash wearing an address's clothes, and the homogeneous slab
degrades to "compact but unfaithful." F-1 must be green before any scalar facet
ships.

### 8.2 classid dual-dispatch `[H]` — one prefix, two resolutions

The `classid(4)` already routes the codebook scope (OGAR canon: longest-prefix
binding). The strong form gives it a **second, parallel resolution** off the
same radix lookup:

- **classid → ReadMode** (the *codec* axis): how to decode this node's value —
  `place ⊕ residue` = Helix Place (identity, §2-Identity, #607) ⊕ CAM-PQ residue
  (the scalar/centroid lanes). This is the **deterministic-place + stored-
  magnitude** split the OGAR perturbation-encoding canon already pins (phase
  deterministic, magnitude stored). `[H]` — the split is shipped in
  perturbation-sim; the *per-classid* codec table is unbuilt.
- **classid → ClassView** (the *schema* axis): what this node's facets *mean* —
  the class's field roster, edge roster, and method-resolution manifest
  (OGAR `ClassView`, the `has_function`/`inherits_from`/`virtually_overrides`
  SPO harvest). `[S]` — the harvest exists in OGAR; the lance-graph-side
  ClassView read is a bet.

One key, resolved once, yields *both* "how do I read these bytes" and "what do
these bytes mean." That co-resolution is the load-bearing claim — and its
honest failure mode is **drift between the two tables** (`I-LEGACY-API-FEATURE-
GATED` in spirit: same prefix, two semantics, must never silently diverge).

### 8.3 LEGO across domains `[S]` — EdgeBlock click via shared codebook

If two programs (an ERP, a OCR pipeline) mint nodes against the *same* OGAR
concept codebook, their `EdgeBlock` slots are **directly clickable**: an
out-of-family edge from domain A's node resolves, by `canonical_concept_id`,
into domain B's node — no adapter, no serialization, because both speak the one
codebook. "Compile on OGAR classes and do LEGO with class shapes" becomes:
**compile = SPO manifest → ClassView**; **run = SoA under `UnifiedStep` /
semiring** (§2-Composition). `[S]` — this is the OGAR core-first doctrine's
end state, explicitly CONJECTURE until `PROBE-OGAR-ADAPTER-UNICHARSET` is green.

**Bounds (the doctrine's own fences, not optional):** the click only works on a
**shared-concept lattice** — domains that don't share concepts get adapter
bricks at the membrane, paying the cost explicitly (OGAR consumer-preflight).
**Structure ⊥ flow**: the EdgeBlock click composes *structure*; it does not
import domain B's *control flow* into A. And per core-first: a Core gap is
*extended deliberately*, never hacked into the adapter.

### 8.4 The view layer `[S]` — ClassView → askama, with Redmine as donor

The far rung: the `ClassView` schema drives a row/table view the way
Redmine/OpenProject's metadata→issue-list machinery does — except *generated
from the schema*, not hand-maintained over 17 years. The theft is sharp because
Redmine already solved **exactly** the ClassView-renderer problem, and three of
its classes map almost 1:1:

| Redmine class | What it does | ClassView analogue |
|---|---|---|
| `Redmine::FieldFormat` | per-type cell formatter registry (string/int/float/date/enum/list/user/link…), each knows render + edit | **codebook-kind → cell-renderer map**: partonomy tile → link/enum cell, value-quantile tile → number/gauge cell, identity → reference cell |
| `Query` / `QueryColumn` | "given a model + its fields, produce a row" | `ClassView → lenses → cells` — literally |
| `CustomField` / `CustomValue` | runtime, per-model, arbitrary-type fields | the **customattribute lens** — consumer schema per `classid` |

So it is not "port a UI" — it is lifting a **proven metadata→row architecture**.

**The one real seam — and its resolution.** Askama is **compile-time** (type-
checked); Redmine's custom fields are **runtime**. Three reconciliations:

- *codegen* — emit one Askama template per ClassView at `build.rs` time.
  Compile-checked ("compile on OGAR classes"), but needs schemas at build.
- *generic renderer* — one Askama *shell* (table/row skeleton, compile-checked)
  + data-driven cell formatters resolved from the codebook at runtime (the
  FieldFormat move). Fully dynamic.
- **hybrid (the answer)** — static type-safe shell + dynamic per-tile cells:
  Askama gives the skeleton, the codebook gives the per-tile semantics. This is
  the `jinja <> dynamic classview` arrow, and it dissolves the seam rather than
  picking a side.

All three stay inside the firewall: build-time codegen from a manifest is the
sanctioned "compile types" pattern (medcare-rs Iron Rule 7 — *not* runtime
serialization), and the hybrid's runtime cell-resolution reads the codebook (a
compile-time contract), it does not deserialize a wire payload.

**The payoff closes the loop:** a Redmine-style row/table is just a **4th
projection** of the same node — next to the 3D scene / graph / splat
(`TorsoMap`'s "three tenants of one identity" → four). One ClassView, every
view. That is §0's "one object, N readings" reaching all the way to the screen.

**Bounds:** transfer the donor's *patterns* (FieldFormat / Query / CustomField
idioms), **not** the 17-year accretion (the legacy cruft is the anti-goal).
**Structure ⊥ presentation**: ClassView builds the *default* view; bespoke views
are **override hooks**, not schema edits. Realistic coverage is ~80% generic +
explicit overrides — claiming 100% schema-driven view is the overclaim this
bound exists to catch. `[S]` — askama is shipped and standard, Redmine's
architecture is proven; the ClassView→cell codegen is entirely unbuilt.

### 8.5 What the strong form adds to the ladder

§8 does not get its own KILL — it **inherits** §4's gates and sequences behind
them:

- **8.1 (homogeneous facet)** is gated on **F-1** (scalar facets need faithful
  centroids) and **F-code** (the facet cascade must be a lossless code, not a
  lossy hash).
- **8.2 (dual-dispatch)** is gated on **F-collapse** — if a learned index/head
  matches the address (§4.2, CogNGen counterweight), then classid-as-schema is
  elegant packing, not a new primitive.
- **8.3 / 8.4 (LEGO, view)** are gated on the **OGAR core-first probe**
  (`PROBE-OGAR-ADAPTER-UNICHARSET`) — adapter parity must go byte-green before
  cross-domain click or schema-driven view is more than a slogan.

**What would kill the strong form specifically** (beyond §6): if facets turn out
to be *irreducibly heterogeneous* — every class needs a bespoke value layout and
no homogeneous 16-byte cascade fits — then 8.1 collapses and §8 reduces to "the
key is a schema pointer" (still useful, far less than claimed). That is the
8-specific entry on §6's ledger: **homogeneity non-closure** is to §8 what
F-basis non-closure is to the whole thesis.

The honest one-line summary of §8: **the engineering rungs (§2 axes, #605, #607)
are real and shipped; the full-stack-compiler reading is a coherent bet whose
every load-bearing joint already has a named, un-run probe.** It earns a place
in the thesis precisely because it is falsifiable end-to-end, not because it is
proven.

---

## Cross-references

- `canonical_node.rs` — the 512B node (key/edges/value, ValueTenant/ValueSchema).
- `perturbation-sim/src/cascade_key.rs` (PR #605) — V1/V2 six-lens address + V3 `(part_of:is_a)`.
- `perturbation-sim/src/place_buffer.rs` (PR #607) — helix Place (identity) ⊥ BF16 buffer (dynamics); the measured split.
- `OGAR/CLAUDE.md` + `ndarray .../guid-prefix-shape-routing.md` — the GUID canon, the 4⁴ condition, F-1.
- Iron rules: `I-SUBSTRATE-MARKOV`, `I-NOISE-FLOOR-JIRAK`, `I-VSA-IDENTITIES`, `I-LEGACY-API-FEATURE-GATED`.
- EPIPHANIES `E-CASCADE-KEY-IS-THE-SPATIAL-ADDRESS`, `E-V3-PART-OF-IS-A-TILE`, `E-LOCATION-PERMEABILITY-CONFLATION`.
- Literature: SDM=attention [2111.05498] · GNN=semiring-DP [2203.15544] · PQ [1102.3828] · RaBitQ [2405.12497] · CogNGen [2204.00619] (the counterweight) · Lance [2504.15247].
