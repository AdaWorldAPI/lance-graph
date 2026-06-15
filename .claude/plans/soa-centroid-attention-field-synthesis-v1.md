# SoA Centroid Attention Field — Unified Synthesis v1

> **Type:** plan (phase-2 marker / co-architecture). Unifies recognition + reasoning + grammar as reads of ONE field.
> **Status:** PLANTED 2026-06-15. Gated on `cycle-coherent-soa-snapshot-v1` (plastic field ⇒ COW writes).
> **Canon:** helix crate (golden-index residue, φ-template); deepnsm; causal-edge (pearl/nars); TEKAMOLO (#495).

---

## 0. The one idea

The **48-bit helix residue + Morton-tile stacked-pyramid perturbation-shader IS a
centroid attention field.** Place (HHTL) = centroid; residue (24-bit golden index)
= each point's perturbation off it = the **query↔key alignment**; the pyramid =
**multi-scale attention** (coarse centroid → fine). The field is *evaluated from the
φ-spiral template, never stored*. Everything below is a **read of this one field at
a different scale** — not separate engines bolted together.

## 1. The reads (each is the same field, different scale)

| Capability | Real crate / source | What it is, as a field read |
|---|---|---|
| **Perception (ONNX/LSTM)** | embedanything(candle)/GGUF host | emits a **query** into the field (golden index + posteriors); the ONLY learned-perceptual part, stays hosted |
| **Attention eval** | `helix` (golden index, curve-ruler, `DistanceLut`) | query↔centroid alignment; Morton pyramid = coarse→fine resolution |
| **Markov context building / bundling** | `deepnsm::markov_bundle`, `encoder` | temporal **superposition along the field** = the bundling read (context = bundled perturbations) |
| **Quorum + NARS reasoning** | `causal-edge::{pearl,nars,syllogism}` | centroid **coupling** = edge read; quorum = agreement of multiple field reads; NARS truth = coupling strength |
| **Grammar heuristics** | `deepnsm::{parser,pos,morphology,spo,syllogism}` | syntactic **field masks** = structured attention over the field |
| **Relative-pronoun / syntax order** | TEKAMOLO resolver (#495) | resolves adverbial/relative-pronoun binding = constrained attention path |
| **Episodic / coref** | AriGraph (`EpisodicWitness64`) *(name "aerial" — confirm)* | temporal chain read = the field over witness-time |
| **Nearest-valid-token** | `crystal_neighborhood`, `cam64`, CAKES + `turbovec` | field-alignment argmax = read-off to codebook word |

## 2. Why this is one object, not a pipeline

VSA bind/bundle/similarity **are** the field operations: bind = perturbation off
centroid, bundle = the pyramid's coarse-level superposition, similarity = field
alignment (`DistanceLut`). So DeepNSM's markov_bundle is the *symbolic readout* of
the field; NARS/quorum is the *edge coupling*; grammar/TEKAMOLO are *attention
masks*. No separate learning machine is needed — the attention field already does
binding/bundling/attention in one structure (Frady/Kleyko 1707.01429: trained-RNN
⊁ VSA for symbol sequences). What's missing is only **plasticity**.

## 3. Phase-2: make the field plastic (the "learning edges")

Not new tenants — **the field adapts**:
- centroid **drift** (place-centroids move toward corpus density);
- shader **perturbation-gain** adaptation (the pyramid's response sharpens);
- timed by `Plasticity` tenant; coupled by `CausalEdge64` strength (NARS mantissa moves).
Evaluated from the φ-template (not materialized). **Hard dep:** `cycle-coherent-soa-snapshot`
COW — plastic field mutates per cycle; without snapshot it thrashes Lance.

## 4. ONNX combination (operator's point)

The ONNX-shaped recognizer and the field **meet at the query boundary**: ONNX emits
posteriors → the field's golden-index query; field eval + grammar masks + NARS
coupling resolve to the token. So ONNX = the perceptual *encoder into* the field;
the field = everything symbolic/sequential/relational. One substrate, two scales.

## 5. Determinism split (non-negotiable)
- **Frozen mode** (centroids/gains fixed) → bit-reproducible → the Tesseract oracle + golden-file harness run here.
- **Plastic mode** (field adapts) → live use; NOT golden-diffable; gated by snapshot.
Two modes, explicitly separated, or the bit-repro guarantee is lost.

## 6. Open
- **OD-A:** confirm "aerial" = AriGraph (`EpisodicWitness64`) vs a distinct AST crate.
- **OD-B:** centroid drift rule — Hebbian on `Plasticity`, or NARS-revision on `CausalEdge64`? (probe-gate, measure first.)
- **OD-C:** operator sign-off required for any new tenant (anti-invention guardrail) — phase-2 should need NONE (field is evaluated, not stored).
