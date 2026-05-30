# FAISS Homology — CAM / CAM_PQ (v0.1)

> Companion to core (v0.1), classes (v0.2), wikidata-hhtl-load (v0.1).
> "Sieht entfernt wie FAISS aus" was right from turn one — it was the FAISS *architecture* (layering), never the FAISS *algorithm* (ANN). This doc pins the homology and the ONE inversion.
> ⚠ HOMOLOGY IS A CHEAT-SHEET, NOT A DEPENDENCY. Do not pull IVF/FAISS code. Implement with Lance-SoA, HHTL-nibbles, BLAKE-CAM, provenance-reasoning-store.

## The structural homology (term for term)

| FAISS | This stack |
|---|---|
| flat vector backing arrays | **SoA** (flat columnar, ID-encoded, `(start,len)` backing) |
| IVF cells / coarse quantizer | **HHTL buckets** (16^n nibble routing, arithmetic not associative) |
| PQ residual codes | **facet codes** (product of per-facet closed-vocab indices) |
| orthogonal index (the inverted file) | **Reasoning layer** (separate indexed store, Derived tier) |
| vector-id → offset | **CAM hash → shape/row** (exact-match) |

## The ONE inversion (the whole thread, in one line)

**FAISS addresses by SIMILARITY (ANN). This stack addresses by IDENTITY (CAM).**
Same architecture — flat + bucket + index. Opposite addressing — near vs. exact.
You kept FAISS's skeleton and swapped its heart: similarity out, identity in. That's why it's *entfernt* like FAISS — the shape is there, the soul is inverted.

## CAM_PQ — product quantization, made symbolic and exact

PQ's move: split a vector into m subvectors → quantize each against a small codebook → store m codebook indices (the product code) instead of the full vector.

**CAM_PQ applies the PQ LAYOUT, not the PQ LOSS:**
- split an entity descriptor into **facet subspaces** (Abstammungs-path, capability, habitat, shape, organic, ...)
- each facet's "codebook" = its **closed OWL range** (owl:oneOf / small rdfs:range) — declared, not learned
- store one **index per facet** = the product code (a tuple of small symbolic codes)
- **CAM-hash the whole product code** for exact identity

Structurally this is **IVFPQ**: HHTL path = the coarse quantizer (IVF cell), facet codes = the PQ residual codes, CAM hash = the exact ID. But:

### Critical: exact, not lossy
PQ is lossy (quantization error). **CAM demands exactness — so facet codes are LOSSLESS.** Because the vocabularies are CLOSED and SMALL, every value gets its own code with zero quantization error. It is really **dictionary encoding with product structure**, not quantization. The "PQ feel" is the product LAYOUT (split into sub-codes, code per subspace); the exactness comes from closed-vocab codebooks. No information lost → CAM identity holds → audit/GoBD safe.

| | FAISS PQ | CAM_PQ |
|---|---|---|
| codebook | learned (k-means over floats) | declared (OWL closed range) |
| code | lossy (nearest centroid) | exact (every value has its own code) |
| addressing | ANN (similarity) | CAM (exact hash) |
| layout | product of m sub-codes | product of facet codes |
| use | retrieval by nearness | identity + dedup + codegen |

### Why CAM_PQ is cheaper than float PQ
- codebooks are **declared** (OWL/DOLCE), not trained — no learning pass, no drift
- codes are **exact** — no re-ranking pass to fix quantization error
- product code is tiny: k facets × small index ≈ one u64 → AND-testable / hashable in a cycle
- the facet u64 IS the SoA facet column → SIMD batch-AND = facet filter (cognitive-shader-driver grid run)

## Where REAL (lossy) PQ is still allowed — and only there

The invariant holds: **similarity/lossy lives ONLY in the proposer/discovery layer, never in addressing.** So genuine lossy PQ (or any ANN/embedding) is legitimate for:
- the **value stream entropy wall** (the irreducible SPO-object data that doesn't fold into a deck slot) — IF you ever want lossy compression there, it's a discovery aid, not identity
- **label/text similarity** and **shape-family discovery** (Aerial+, Jina) — proposing inheritance/relations
Never for: recipe selection, identity, the CAM key, the reasoning-store keys. Those stay exact.

**Rule restated:** facet-PQ = product layout + exact closed-vocab codes (lossless, addresses identity). float-PQ/ANN = lossy, discovery only, never addresses. Same triples, two indexes, never swapped.

## The closed picture (architecture is closed, now falsifiable)

Class (Quartett mask, inherited along HHTL as delta) + SoA (flat columnar backing) + HHTL (16^n nibble router = IVF cells, OWL/DOLCE axis template, facets as orthogonal bitmasks) + CAM (exact identity = the swapped FAISS heart) + CAM_PQ (product-structured lossless facet codes) + Reasoning (orthogonal indexed Derived store = FAISS's inverted index, over inferences). All one SPO substrate, separated by provenance, governed by: dumb-uniform below, meaning above.

**Closed, not finished.** No open edge forces a new structural question; every layer docked without overturning an invariant. The claim is now falsifiable: chess + Odoo + Wikidata-anatomy all run through the same Class+SoA+HHTL+CAM+Reasoning with no special-case. The architecture is closed in the head; whether it's closed in the bytes is told by the first dataset that runs all five layers and either comes out clean or forces a layer to lie.

---
*v0.1. The homology explains the form; your building blocks are the implementation. Cheat-sheet, not dependency.*
