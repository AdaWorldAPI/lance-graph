# S3.0 audit — why PR #973 was closed, and what replaces it

> **Status:** AUDIT (operator-directed, 2026-08-20). Written BEFORE any
> replacement code, per the reset brief. #973 is closed and is **not** to be
> reopened or continued; this note is the record of why, so the failure stays
> visible rather than being smoothed into "an earlier design iteration".

---

## 1. The exact claim in #973 that was wrong

#973 measured, correctly:

```text
domain + S + P + O  =  4 × u16  =  64 bits  =  16 nibbles  =  NiblePath::MAX_DEPTH
```

and then concluded, **incorrectly**:

> therefore the exact literal cannot itself live in HHTL, because no path depth
> is left for evidence below it — so identity must be separated from HHTL
> routing, and HHTL retained only as a lossy prefix projection.

**The measurement is fine. The inference is rejected.** It silently promoted

- a limit of **one concrete sequential representation** (`NiblePath`: a `u64`
  carved 16-ary, `FAN_OUT = 16`, `MAX_DEPTH = 16`, `child()` shifting left 4)

into

- a limit of **HHTL itself** as hierarchy / locality / exact-addressing
  substrate.

`NiblePath::MAX_DEPTH` is a property of that carving. It is not the
dimensionality of HHTL, and it says nothing about whether orthogonal facets can
be keyed by the same exact address.

The second, subtler error rode along with it: treating a **Morton / prefix**
view as inherently lossy. Truncation is lossy. Interleaving is not.

---

## 2. The measured evidence that contradicts it

Six items. The first two are in the **same crate #973 modified**; the third is
a finding **this session measured and reported itself**. Any one of them
falsifies the inference.

### 2.1 `RAIL_MAX_DEPTH = 24` — in `lance-graph-contract`, one file away

`crates/lance-graph-contract/src/rail_geometry.rs:50-55`:

```rust
pub const RAIL_PAIR_LEVELS: usize = 6;    // levels per interleaved-pair register
pub const RAIL_SLAB_LEVELS: usize = 12;   // levels per axis-slab register
/// Maximum addressable depth: a slab register plus its continuation.
pub const RAIL_MAX_DEPTH: usize = 2 * RAIL_SLAB_LEVELS;   // = 24
```

with `RailCarving::AxisSlab { reg, cont: Option<usize> }` — *"twelve contiguous
level bytes at `reg`, with an optional (possibly discontiguous) **continuation**
register of twelve more."*

**The contract crate declares the canonical maximum addressable depth as 24
levels via a continuation register.** #973 asserted a hard ceiling of 16
nibbles by reading `NiblePath::MAX_DEPTH` — a different type, in a different
module of the same crate — and never checked the rail geometry that owns the
question.

### 2.2 The canonical tenant is `FacetCascade`, and it is already coded

`crates/lance-graph-contract/src/facet.rs`: `FacetTier { lo, hi }` (2 B) +
`FacetCascade { facet_classid: u32, tiers: [FacetTier; 6] }` = **4 + 12 = 16 B**,
size-asserted. Operator ruling, 2026-08-20: *"HHTL is the canonical 6×2×8bit
tenant for SoA identity. Period."* The module's own doc gives the rule:

> The substrate is **ALWAYS 8:8** … only the CONSUMER projects meaning onto the
> bytes … The producer bakes in nothing.

Re-carvings are already an algebra — `CascadeShape::{G6D2, G4D3, G3D4}`,
"byte-for-byte the same 12-unit register". **`G4D3` is the 4×24 reading**, which
is MedCare-rs's special case (operator, same session).

### 2.3 THE SHARPEST ONE — this session already measured the continuation, and said so

On 2026-08-19, working the Zipper/DN thread (session tasks *"Zipper/DN
precedent archaeology"* and *"Phase 1: logical DN across base+continuation"*,
both completed), the operator scoped it:

> *"a hierarchy path deeper than 12 can be represented as a continuation; the
> full logical DN can be reconstructed; `parent()` is truncation by one
> position; `ancestor()` is prefix containment; depths 16–18 round-trip"*

and **this session reported back, in its own words**:

> *"`read()` does exactly the concatenation — `p[..12]` from the slab, `p[12..]`
> from the continuation slab. **So the logical DN across base+continuation
> already exists and is already assembled.**"*

**#973 then claimed that depth past the 16-nibble mark makes the exact literal
unrepresentable in HHTL.** That is not a forgotten repo precedent from months
ago — it is a **direct contradiction of a measurement this session made and
published hours earlier**. This is the single most damning item and it belongs
at the top of the record.

### 2.4 The primitive layer was declared COMPLETE — by the operator, in session

2026-08-20T00:53Z:

> Facet bytes · `hi_chain` / `lo_chain` (2 × 6-level readings) ·
> `shared_prefix_tiles` (XOR + TZCNT prefix length) · `row_match_mask` (4-bit
> equality mask) · distance/group helpers.
> **"So there is no missing HHTL operation and no reason to invent an ndarray
> helper, gather path, or extra mask algebra."**

and 00:46Z: *"The split is already implemented as a no-op as part of the **2× 6×2×8bit
cast** for one CPU cycle."* All four primitives are present in `facet.rs`
(`hi_chain` :211, `lo_chain` :220, `shared_prefix_tiles` :255, `row_match_mask`).

### 2.5 `E-WORDNET-MAKES-THE-4-ARY-ADDRESS-SEMANTIC-1` (PR #875)

FINDING, 5/5 gates green on real WordNet 3.1 (82,192 noun synsets): the `@`
hypernym relation is used **as the HHTL address** — it encodes structure rather
than discovering it (W1 +0.494 real vs −0.036 shuffled; W3 out-of-cell band
0.763 vs 0.031 random = **24.71×**). W4, verbatim:

> `NiblePath` (`FAN_OUT = 16`, the shipped router) can express exactly two
> levels in a byte; inside one top nibble it sees a single undifferentiated
> bucket. The 4-ary address splits that same population into two rungs that
> differ by 2.47 WordNet hops. **That is real structure the current router is
> blind to.**

The repo had already **measured** that `NiblePath`'s carving is one carving
among others. #973 took the shipped router's ceiling for the substrate's.
PR #876 (`PROBE-HHTL-INTAKE-BLINDNESS`) is the paired caution: a null produced
by an intake limitation must not be reported as a property of the address.

### 2.6 OSM / WebMercator and Bible Rosetta — coordinate-plus-facets

- **OSM:** OGAR `MERCATOR-HHTL-HELIX-MAP.md` §1 + OGAR `CLAUDE.md` — *"domains
  bind the axes (**OSM: literal x/y**; semantic: PQ subspace pairs)"*; a tier is
  a 256×256 tile, canon "one byte per axis per tier" = exactly the `8:8` pair.

  > **⊘ I OVER-CLAIMED THIS ROW — corrected by a read-only audit, and the
  > overclaim is the same failure mode this document exists to record.** An
  > earlier revision ended this bullet: *"Exact Cartesian coordinates are an
  > exact HHTL address, **in a shipped domain**."* **False.** Measured:
  > `MERCATOR-HHTL-HELIX-MAP.md:5` sets the legend *"`[G]` = in code, `[H]` =
  > design"* and **§1 at `:17` is graded `[H]`**; its round-trip falsifier
  > (`:94-96`) is **unrun**. `ogar-osm/src/lib.rs:212-296` declares
  > `GEO_V3_FACET` — a **byte-position table**; grep for lon/lat/mercator/
  > morton/zoom math in that crate returns **zero hits**. The cited reader is a
  > stub: `ndarray/crates/cesium/src/esri_crs.rs:285` `inverse_mercator` is
  > `unimplemented!("scaffold only")`; `osm_pbf.rs:12` *"this file is
  > **D-OSM-1** — the stub"*. `OGAR/docs/DISCOVERY-MAP.md:222` grades `D-OSM`
  > **`H` / `IDEA` / queued**.
  >
  > **What OSM actually establishes:** the binding of Cartesian axes to HHTL
  > tiers is DECLARED as a byte schema and minted as classids — not computed,
  > and nothing hydrated. It still refutes #973's inference (nothing anywhere
  > treats the address plane as unable to hold a Cartesian point), but it is a
  > **design precedent, not a shipped one**. Citing it as shipped was
  > strengthening a receipt to fit an argument — the reflex that produced #973,
  > repeated inside the document written to record it.

- **Rosetta:** `.claude/plans/rosetta-codebook-convergence-v1.md` — *"The verse
  address is a frozen external key … the exact sentence in ALL translations
  lands in [the same row]"*, while WordNet synsets supply a separate
  language-neutral semantic coordinate. One absolute coordinate roots orthogonal
  coordinate systems; language lane, clause index, sense and qualia vary as
  **facets**, never as nibbles appended below the verse.

  > **⊘ SAME CORRECTION, SAME DIRECTION.** That plan's own status line (`:3`)
  > is **PROPOSED (doc-only)**; the verse row is deliverable `D-RCC-2`
  > (`:108-114`), and no verse-identity type exists in `crates/`. The book row's
  > real state is the operator's point exactly: **the address exists, the
  > concept field does not.** A book is a HORIZONTAL STREAM of addressed
  > sentences; concepts are the VERTICAL axis and are not materialized as SoA
  > at HEEL/HIP. Reasoning is what has to hydrate them.

### 2.7 MedCare-rs: FMA anatomy IS the HHTL address, in production

Operator, 2026-08-20: *"MedCare-rs is using FMA anatomy 70k nodes 4 Mio vercels
as HHTL"* — and, separately, *"MedCare-rs using 4×24 as a special case"*
(= `CascadeShape::G4D3`, the same 12-unit register re-carved 4 groups × 3
levels).

This is the WordNet result (§2.5) again, in a second domain and **in a shipped
consumer**: a real ontology — the Foundational Model of Anatomy — is not
*routed by* HHTL, it **is** the HHTL address, at 70k nodes and ~4 M vertices.
Two independent domains, two different carvings (4-ary semantic in #875, 4×24
here), one substrate.

It also makes the #973 inference untenable on its own terms: a 70k-node anatomy
addressed at 4×24 is precisely the case that a 16-nibble ceiling would have
declared impossible, and it is running.

> **⊘ MORTON IS NOT PART OF THIS — AND REACHING FOR IT WAS THE SAME MISTAKE
> AGAIN (operator, 2026-08-20: "No SoA ever was allowed to use Morton", "No NARS
> ever residing in Morton", "Nobody fucking asked you to hallucinate Morton").**
> The Stage-3 brief mentioned Morton **only** to deny the premise *"Morton
> implies lossy"*. An earlier draft of THIS audit turned that denial into a
> component to build — a second instance of the same failure mode, inside the
> document written to correct the first. Morton exists here as a SIMD nibble
> lens (`FacetTier::morton`, GFNI) and in non-SoA spatial probes. It is not the
> SoA identity substrate, no NARS state resides in it, and S3.0 must not contain
> it.

## 3. What code from #973 is still mathematically valid

Salvageable, after re-interpretation:

| #973 artefact | verdict |
|---|---|
| four canonical `u16` components; exact component equality; no hash, no tolerance, no learned assignment | **valid** |
| reversible fixed-width packing (`as_u64` / `to_le_bytes` and inverses) | **valid** |
| `const _: () = assert!(size_of == 8)` as the structural guard that no evidence/confidence/source/version field can be added | **valid, and worth keeping** — it makes falsifiers 7–10 compile-time |
| the component-isolation matrix from a non-zero baseline | **valid** |
| the injectivity sweep and the "three sources → one literal" test | **valid** |
| the `E-A-DISABLE-THAT-DOES-NOT-BIND-IS-NOT-A-DISABLE-2` method finding | **valid, unrelated to the error** |

Invalid and removed:

| #973 artefact | verdict |
|---|---|
| the "identity is outside HHTL" framing | **rejected** |
| `E-THE-LITERAL-CANNOT-LIVE-IN-THE-PATH-IT-ROOTS-1` | **retracted** (never merged — #973 closed unmerged, so nothing on `main` needs editing) |
| `routing_prefix_is_not_identity` **as the architectural headline** | **demoted** — it is a fact about *truncation*, not a discovery about HHTL |
| the `LITERAL_PATH_NIBBLES == MAX_DEPTH` assert read as a budget *impossibility* | **re-read** as one carving's capacity, nothing more |
| the name `CausalLiteral` | **renamed** — see §4 |

---

## 4. Terminology and types — and the type that already exists

**The canonical carrier is not to be re-minted.** Operator ruling, 2026-08-20:
*"HHTL is the canonical 6×2×8bit tenant for SoA identity. Period."* That tenant
is already CODED, in this crate:

`lance_graph_contract::facet` —
`FacetTier { lo, hi }` (2 B) + `FacetCascade { facet_classid: u32, tiers: [FacetTier; 6] }`
= **4 + 12 = 16 B**, one 128-bit register, with `const _` size asserts. Its own
module doc states the governing rule:

> The substrate is **ALWAYS 8:8** (each tier is two opaque bytes `hi:lo`); only
> the CONSUMER projects meaning onto the bytes … The producer bakes in nothing
> (AGI-as-glove: the SoA is content-blind, the reader interprets).

and the re-carvings are already an algebra — `CascadeShape::{G6D2, G4D3, G3D4}`,
"byte-for-byte the same 12-unit register". **`G4D3` (4 groups × 3 levels) is the
4×24 reading**, which is the MedCare-rs special case (operator, same session),
and OGAR's grace-carving amendment lists `G2 4×u24` in the same family.

Consequences for S3.0:

- **No new packed identity type.** `ExactLiteralAddr` is a *reading* over the
  canonical facet, not a parallel 4×`u16` container. Minting one would be the
  ruling-E anti-pattern (*"a container minted to avoid completing the address
  transition"*) — the same mistake in a new costume.
- **`CausalLiteral` → exact literal, generic.** Causality is a predicate family
  above identity, never inside it: `ASSOCIATED_WITH`, `PART_OF`,
  `INTERACTS_WITH`, `CAUSES`, `MEDIATES`, `PREVENTS`, `SUPPORTS`,
  `CONTRADICTS` all address through the same substrate.
- **No Morton anywhere in this path** (§2.3 fence).
- **`routing_prefix` → a locality/cohort *truncation*,** demoted from headline;
  the facet already ships `prefix_distance` (LCP) for that job, so a new prefix
  API may not be needed at all.

## 5. Replacement invariants

```text
                  ExactLiteralAddr (D, S, P, O)
                  one immutable Cartesian point
                  = the exact HHTL literal address
                              │
        ┌──────────┬──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
   ontology   concept    CausalMeta  Epistemic   BasinSet /
   coordinate coordinate             Meta        EntropyWork
```

1. Identity is the four canonical axes. Exact, reversible, no hash, no
   tolerance, no learned assignment.
2. The address is **inside** HHTL — it is a reading over the canonical
   `6×(8:8)` facet tenant (`FacetCascade`), not a parallel container beside it.
3. Evidence, Meta, basins, qualifications and receipts are **orthogonal facets
   keyed by** the address. They are never appended as extra nibbles, and they
   never change the address.
4. Truncating the locality view is coarse **by choice**; that is a property of
   truncation, not of the substrate.
5. Causality is a predicate family above the literal, never inside identity.
6. A local V3 `u16` target is a tenant-local proxy and never the absolute
   address.

---

## 6. The process finding

The failure was **not** "we tried a reasonable architecture and later found a
better one." It was: *a new local derivation ignored already-measured substrate
facts and confidently promoted its incomplete premise into an architectural
impossibility* — in a repository that contained the counterexample in three
domains and a passing test named `morton_roundtrip_is_identity`.

Recorded as `E-A-LOCAL-DERIVATION-CANNOT-OVERRULE-A-MEASURED-COUNTEREXAMPLE-1`.
The operative rule: **when a local derivation concludes "X is impossible",
that is a claim about the whole substrate and requires a search for
counterexamples before it is written down — not merely a correct calculation.**
