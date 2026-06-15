# OCR → Canonical SoA Integration v1

> **Type:** plan (sub-plan — the one that binds OCR to the lance-graph substrate). Deliverables D-OCR-50/51/52/53.
> **Status:** PLANTED 2026-06-15 — design only. THIS is "use the new architecture we raced for."
> **Front:** post-#496. Integration surface = `canonical_node.rs` (`NodeGuid`/`EdgeBlock`/`EdgeCodecFlavor`/`NodeRow`/`ValueTenant`/`ValueSchema`/`NodeRowPacket`) + `class_view.rs` (`ClassView`/`FieldMask`).
> **Canon anchors:** OGAR/CLAUDE.md P0 GUID; lance-graph/CLAUDE.md SoA node (`4ea6ac9`); soa-three-tier-model; DeepNSM crate (`lance-graph/crates/deepnsm`); helix/CAM-PQ (`crates/helix`, `bgz-tensor` CAM-PQ).
> **Skip-by-rule:** OCR introduces NO bespoke row geometry. It rides the existing value-tenant carve.

---

## 0. Intent

An OCR token is not a foreign payload that needs a boundary adapter — it **is** a
canonical SoA node. This plan defines the mapping so recognized text lands directly
in the substrate: addressed by HHTL, classed by OGAR, valued by a `ValueSchema`
preset over *existing* `ValueTenant`s, edged by `EdgeCodecFlavor`, repaired by
DeepNSM + CAM/PQ, and persisted via `NodeRowPacket`. Zero boundary tax — the whole
point of the splat-native / "one representation, many views" doctrine, applied to OCR.

## 1. OCR token → `NodeRow` mapping (D-OCR-50)

**Key (`NodeGuid`, 16 B):** `classid · HEEL · HIP · TWIG · family · identity`.
- `classid` = the minted OCR class prefix (see §2). `0x0000_0000` fallback until OGAR mints it.
- HHTL path (HEEL/HIP/TWIG) = document → page → block (the layout hierarchy from `ocrs::layout_analysis`).
- `family` (3 B) = line/region basin; `identity` (3 B) = token ordinal within the basin.
  → `local_key()` (trailing 6 B) addresses a token within its line after the trie walk.

**Edges (`EdgeBlock`, 16 B = 12 in-family + 4 out-of-family):**
- in-family (12): reading-order + local-layout adjacency (prev/next token, same-line
  neighbors, baseline siblings). `EdgeCodecFlavor::CoarseOnly` (1 B/slot) — pure topology.
- out-of-family (4): inherited adapters — (A) table-cell membership, (B) block/column
  parent, (C) semantic/coref link (post-DeepNSM), (D) source-region (bbox → page geometry).

## 2. OCR class + HHTL address scheme (D-OCR-50)

- Mint an OCR class family in OGAR (`ogar-ontology`): `Document → Page → Block →
  Line → Token`, with leaf token subtypes (`Word`, `Number`, `Date`, `Currency`,
  `Glyph`, `TableCell`). Until OGAR mints them, hardcode the classid prefix space
  per the reserve-don't-reclaim ladder (the classid bytes stay reserved at offset 0).
- `ClassView` for the OCR class declares `edge_codec_flavor` (`CoarseOnly`) and
  `value_schema` (the OCR preset, §3).

## 3. OCR `ValueSchema` preset over EXISTING tenants (D-OCR-51)

The 480-byte value slab already carves into `VALUE_TENANTS`. OCR **rides existing
tenants** — no new tenant for the POC:

| Tenant (existing) | OCR use |
|---|---|
| `Fingerprint` (32 B / 256-bit) | glyph/line identity print (DeepNSM `encoder` XOR-bind/bundle of the crop) |
| `TurbovecResidue` (16 B, PQ) | glyph embedding → CAKES nearest-valid-token search |
| `HelixResidue` (48 B) | orthogonal residue: per-token deviation from class centroid (confidence-as-residue) |
| `Meta` (u64) | packed confidence + NSM-repair flags + token-subtype bits |
| `EntityType` (u16) | OCR token class discriminator (Word/Number/Date/Glyph/TableCell) |
| `Plasticity` (u32) | correction history / last-repair stamp |

→ define `ValueSchema::Ocr` (or select `Cognitive` if its mask already covers the
above) as a `FieldMask` over those `ValueTenant` positions. Selection only — it
carves *within* the slab, moves nothing (canon: tenants never move/reuse).

**OD-1 (deferred):** a dedicated `ValueTenant::OcrEvidence` (bbox `[f16;4]` +
per-char confidence + top-k recodebeam candidates) is the clean home for
recognizer evidence. Adding a tenant is canon-significant, so the POC packs a
compressed form into `Meta`+`HelixResidue` and defers the dedicated tenant to a
follow-up once the evidence shape is stable (needs D-OCR-21 `lstm_choice_mode`).

## 4. Repair: DeepNSM + CAM/PQ nearest-valid-token (D-OCR-52)

The recognizer emits candidates+confidence; repair is the brainstem we already have:
- **Character/orthographic layer (new, thin, below DeepNSM):** `0/O 1/I/l 5/S rn/m`
  confusion table + number/date/currency/table-cell grammars. Repairs orthography on
  OOV garbage (codes, IDs like `69B8`) BEFORE the word layer. (This is the only
  genuinely greenfield code; the word-frequency half already exists as
  `deepnsm/word_frequency`.)
- **Word layer = `deepnsm`:** `vocabulary` → `codebook` → `parser`/`pos` → `encoder`
  → `similarity`/`cam64`/`crystal_neighborhood`. Word-level plausibility + disambiguation.
- **Nearest-valid-token = helix / CAM-PQ / CAKES:** the glyph `TurbovecResidue`
  (PQ) + `HelixResidue` feed CAKES nearest-valid-token; CHAODA flags anomalous
  tokens (likely-misrecognized). This is `bgz-tensor` CAM-PQ + `crates/helix`.

Repaired token writes back: corrected text → `Fingerprint`/`EntityType`, repair
provenance → `Meta`/`Plasticity`.

## 5. Persistence + planner (kv-lance / surreal)

- `NodeRowPacket` → `SoaEnvelope` → Lance (kv-lance backend, per `surrealdb` fork).
  OCR nodes are ordinary rows; a Lance version is a coherent page/document snapshot.
- `surreal_container` as the **OCR-job control plane** (per its role: planner / AST
  adapter / time-series / kanban): kanban of OCR jobs (queued→detect→recognize→
  repair→persisted via the Rubicon transitions already in `soa_view.rs`), time-series
  of throughput, AST API for the repair-grammar (compile-time vs JIT grammars).

## 6. Bit-reproducibility harness (D-OCR-53) — the migration payoff

The transcode oracle (D-OCR-2x) makes OCR a **deterministic regression source for
the whole SoA migration**: the same line crop → C++ Tesseract text AND Rust port
text AND the resulting `NodeRow` bytes. Because every stage is supposed to be
bit-reproducible (DeepNSM bit-reproducible, envelope version-stamped, CausalEdge64
locked), a golden-file diff over (crop → NodeRow) exercises exactly the muscles the
migration must harden: `ndarray::hpc` hydration, the envelope LE round-trip, and
SIMD numeric exactness. OCR is the best external oracle the substrate has.

## 7. Deliverables

- **D-OCR-50:** OCR class + HHTL address scheme; `ClassView` impl for OCR class.
- **D-OCR-51:** `ValueSchema` OCR preset (FieldMask over existing tenants); a token
  round-trips token→NodeRow→token with no geometry change.
- **D-OCR-52:** DeepNSM + character-confusion layer + CAM/PQ repair wired; a known
  OCR-garbage fixture (`69B8`, `rn`→`m`) is repaired by plausibility.
- **D-OCR-53:** golden-file (crop → NodeRow bytes) regression green, shared with the
  SoA migration suite.

## 8. Open decisions

- **OD-1:** dedicated `ValueTenant::OcrEvidence` vs ride `Meta`+`HelixResidue` (POC rides).
- **OD-50a:** is a "Token" one node, or is a "Line" the node and tokens are value-slab
  sub-records? (Node-per-token is simpler + edges are natural; node-per-line is denser.)
- **OD-52a:** character-confusion layer as a `deepnsm` submodule vs a sibling
  `coca-codebook` crate. (Word-frequency half already lives in `deepnsm/word_frequency`.)
