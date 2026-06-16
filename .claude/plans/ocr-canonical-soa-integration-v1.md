# OCR → Canonical SoA Integration v1

> **Type:** plan (sub-plan — the one that binds OCR to the lance-graph substrate). Deliverables D-OCR-50/51/52/53.
> **Status:** PLANTED 2026-06-15 — design only. THIS is "use the new architecture we raced for."
> **Front:** post-#498 (helix `Signed360` right-sized **48 B → 6 B**; OCR keystone `LayoutBlock::to_node_row` + `BlockKind::entity_type` + `classid_read_mode` **already SHIPPED** in `ocr.rs`; `ENVELOPE_LAYOUT_VERSION` = 2; value carve `[32,144)`, Full 112 B / Compressed 56 B). Integration surface = `canonical_node.rs` (`NodeGuid`/`EdgeBlock`/`EdgeCodecFlavor`/`NodeRow`/`ValueTenant`/`ValueSchema`/`NodeRowPacket`) + `class_view.rs` (`ClassView`/`FieldMask`) + shipped `ocr.rs`.
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
  `Glyph`, `TableCell`). **Stay at classid `0x0000_0000` (bootstrap address,
  identity-only discrimination) until OGAR actually mints the class** — do NOT
  hardcode a non-zero classid prefix, which would wake prefix-routing with no
  registry entry and fall silently to `ReadMode::DEFAULT` (matches shipped `ocr.rs`,
  which writes `NodeGuid::new(classid, 0,0,0, FAMILY_DEFAULT, identity)`).
- **HHTL = a layout-address trie for OCR nodes, NOT a similarity cascade.** The
  5 layout levels map onto 3 key tiers + family + identity as a *prefix*
  decomposition: Document/Page/Block → HEEL/HIP/TWIG (radix-walk prefix), Line →
  family (locality basin), Token → identity. This deliberately forgoes the
  *similarity-basin* reading of HEEL/HIP/TWIG/family (canon's coarse→fine
  neighbourhood tiers); the OCR `classid` marks these nodes as layout-addressed so no
  cross-document family-purity / two-basin benchmark runs against these coordinates.
- `ClassView` for the OCR class declares `edge_codec_flavor` (`CoarseOnly`) and
  `value_schema` (ride `Full` POC / `Compressed` — no new variant, §3).

## 3. OCR `ValueSchema` preset over EXISTING tenants (D-OCR-51)

The 480-byte value slab already carves into `VALUE_TENANTS`. An OCR token's
**recognized string is NOT stored in the node** (I-VSA-IDENTITIES, enforced by
shipped `ocr.rs:97-101`): the node is the *identity that points to* OCR content;
the string + pixel geometry live in an external content store keyed by `identity`.
The value tenants carry typed scalars + a compressed similarity coordinate for
*repair / disambiguation*, never a reversible text payload.

| Tenant (existing, post-#498 sizes) | OCR role |
|---|---|
| `HelixResidue` (**6 B = 48 bit `Signed360`**, NOT 48 B) | A **stored** golden-spiral *place index* (rim 3 B + sign-partition polar 1 B + golden azimuth 2 B; `helix/src/residue.rs:63-116`). The 6 B IS the kept index; the multi-scale field is the *deterministic decode* of it (`RollingFloor::quantize` / `HemispherePoint::lift`, pure `&self`) — "8K resolution at Super-8 cost." It is a place code, **not** a confidence carrier. (The old "48-byte, category-wrong, do NOT use it" line was written pre-#498 against a bits→bytes slip; the tenant is **6 B** and is exactly the keep-the-index design — use it.) |
| `TurbovecResidue` (16 B, `Pq32x4`) | The **edge-block** PQ residue (`EdgeCodecFlavor::Pq32x4`, rank-preserving / absolute-distance-lossy, ICC 0.11–0.29). NOT the glyph→word carrier — nearest-**valid**-token needs absolute distance, so the glyph→word search uses **DeepNSM's `Codebook` CamCodes** (6×256×16, 6 B; `deepnsm/src/codebook.rs`) + `vocabulary.rs` reverse, not this tenant. |
| `Meta` (u64) | A SMALL codebook anchor only (a ≤12-bit vocab rank fits). It does NOT carry confidence (→ `Energy`, shipped `ocr.rs:112-114`), repair flags (→ `Plasticity`), or the OOV recoder-code (→ external content store). `Meta` is the cognitive `MetaWord`; overloading it 5 ways is an I-LEGACY-API-FEATURE-GATED hazard (one u64, different meaning per class). Prefer a future `ValueTenant::OcrEvidence` (OD-1) for OCR-specific evidence. |
| `EntityType` (u16) | token subtype (Word/Number/Date/Glyph/TableCell) |
| `Plasticity` (u32) | correction history / last-repair stamp |

**Recognition vs reconstruction — be precise (corrects the pre-#498 framing).**
The recognized string is recovered by **identity → external content-store lookup**,
not by inverting a residue. There is **no `residue → codebook-rank` inverse** in the
code: `deepnsm/vocabulary.rs` maps `rank → &str` via a stored table, and every
`nearest_words(rank,k)` / `word_neighbors(word,k)` entry point takes a *known*
rank/word as input. So "reversible without a hash" is NOT a property of the substrate
today — the codebook code is a **repair / disambiguation signal**, not a lossless
text payload. The honest mapping:
- **text** → external content store keyed by `identity` (I-VSA-IDENTITIES).
- **`Meta` codebook anchor + `TurbovecResidue` / `HelixResidue`** → similarity
  coordinates that feed *repair* (DeepNSM plausibility + char-confusion + CAKES
  nearest-valid-token), NOT round-trip text recovery.
- the multi-scale decode uses the **real** primitives — `framebuffer::build_mipmap_pyramid`
  / the HHTL `splat3d/depth_cascade` / the helix φ-template / the CAKES ladder
  (`high_heel.rs:16-24`) — there is **no** "Morton-tile stacked-pyramid perturbation-shader"
  in either repo (Morton is explicitly rejected for Hilbert in `linalg/hilbert.rs:50`).
- **Gate:** if a measured `residue → rank` round-trip is ever wanted, it must be
  PROVEN first (probe **OCR-RT**, see `ocr-probes-v1.md`) — it is CONJECTURE today.

**True-OOV (a raw code like `69B8`):** `recodebeam` emits recoder codes; the code +
its char-confusion repair (D-OCR-52) live in the content store with the token text,
keyed by `identity`. Not bundled into the node.

**ValueSchema:** do **NOT** add a 5th `ValueSchema::Ocr` enum variant — that is a
contract-surface addition against the #496 §0 anti-invention guardrail. Shipped
`ocr.rs` already transcodes by riding the POC-`Full` default (`classid_read_mode →
Full`) and writing only the tenants it populates. Post-POC, OCR rides the existing
**`Compressed`** preset (already = Fingerprint + HelixResidue + TurbovecResidue +
EntityType) — or, if a distinct tenant set is truly needed, **mint an OCR class** in
OGAR whose `ClassView` selects existing tenants (the §0-sanctioned opt-in route).
New capability = new column/class, never a new enum variant.

## 4. Repair: DeepNSM + CAM/PQ nearest-valid-token (D-OCR-52)

The recognizer emits candidates+confidence; repair is the brainstem we already have:
- **Character/orthographic layer (new, thin, below DeepNSM):** `0/O 1/I/l 5/S rn/m`
  confusion table + number/date/currency/table-cell grammars. Repairs orthography on
  OOV garbage (codes, IDs like `69B8`) BEFORE the word layer. (This is the only
  genuinely greenfield code; the word-frequency half already exists as
  `deepnsm/word_frequency`.)
- **Word layer = `deepnsm`:** `vocabulary` → `codebook` → `parser`/`pos` → `encoder`
  → `similarity`/`cam64`/`crystal_neighborhood`. Word-level plausibility + disambiguation.
- **Nearest-valid-token = DeepNSM codebook + the L1 CAM-PQ cascade (NOT Hamming):**
  glyph → **DeepNSM `Codebook` CamCodes** (`deepnsm/codebook.rs`) → `vocabulary`
  reverse, ranked through the **L1** CAM-PQ stroke cascade (`ndarray::hpc::cam_pq`
  `cascade_query`) + CLAM DFS-sieve (`clam.rs` `knn_dfs_sieve`); CHAODA
  (clustered-hierarchical outlier detection) flags anomalous tokens. The Hamming
  σ-band `Cascade` (`cascade.rs`) is for binary fingerprints only — palette/codebook
  data is L1 (see `bgz-tensor::hdr_belichtung`). `TurbovecResidue` is the edge codec,
  not the glyph carrier (§3).

Repaired token writes back: corrected **text → external content store** (keyed by
`identity`, I-VSA-IDENTITIES — never into `Fingerprint`); token subtype → `EntityType`;
confidence → `Energy`; repair provenance / last-repair stamp → `Plasticity`.

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
text AND the resulting `NodeRow` bytes. The VSA bind/bundle path is integer
(bit-reproducible), but **DeepNSM's repair / similarity stage is f32** (`encoder.rs`
similarity → f32; `pipeline.rs` weighted blend) — so the **frozen-mode** golden diff
MUST exclude (or pin) the f32 repair stage, and the helix `floor_version` MUST be
fixed in the golden bytes (else the rolling floor rolls and the diff spuriously
fails). With those carve-outs (probe **OCR-DET**), a golden-file diff over
(crop → NodeRow) exercises exactly the muscles the migration must harden:
`ndarray::hpc` hydration, the envelope LE round-trip, and SIMD numeric exactness.
OCR is a strong external oracle for the substrate.

## 7. Deliverables

- **D-OCR-50 (PARTIALLY SHIPPED in #498):** block→`NodeRow` already lands via
  `ocr.rs` `LayoutBlock::to_node_row` + `BlockKind::entity_type`. Remaining: (a)
  token-grain nodes (OD-50a), (b) populate the HHTL layout-trie (HHT currently 0),
  (c) mint the OGAR OCR class. Re-cast as *extend the shipped `ocr.rs`*, not build.
- **D-OCR-51:** OCR rides the `Full` POC / `Compressed` preset (**NO** new
  `ValueSchema` variant — §0); a token lands token→NodeRow with identity→content-store
  text recovery (no in-row reversible text — see §3).
- **D-OCR-52:** DeepNSM + character-confusion layer + CAM/PQ repair wired; a known
  OCR-garbage fixture (`69B8`, `rn`→`m`) is repaired by plausibility.
- **D-OCR-53:** golden-file (crop → NodeRow bytes) regression green, shared with the
  SoA migration suite. **Prereq: D-OCR-50 + D-OCR-51** (class/HHTL/ValueSchema must
  define the row layout before bytes can be golden-diffed).

## 8. Open decisions

- **OD-1:** dedicated `ValueTenant::OcrEvidence` vs ride `Meta`+`HelixResidue` (POC rides).
- **OD-50a:** is a "Token" one node, or is a "Line" the node and tokens are value-slab
  sub-records? (Node-per-token is simpler + edges are natural; node-per-line is denser.)
- **OD-52a:** character-confusion layer as a `deepnsm` submodule vs a sibling
  `coca-codebook` crate. (Word-frequency half already lives in `deepnsm/word_frequency`.)
