# deepnsm-v2 Cam96 pairwise — v5 RATIFIED: the L4 identity facet, its readings, the COCA fixed points, three reference sets

**Status:** RETRACTED-IN-PART (operator, 2026-09-30, same day) — see § ⊘ RETRACTION directly below; it overrides §3.1–3.3, the word-register half of §3.2, and the gates and D-ids it names. What it does not name stands. Was: RATIFIED by the second 5+3 council (2026-09-30). Supersedes `-v4.md` and `-v3.md` §3, §6.3, §7. Council record: `AGENT_LOG.md` 2026-09-30 (2).
**Deliverables:** §10 (`D-CPW-*`), read through the retraction. **§11 (added 2026-09-30):** the execution socket beneath the lexical substrate — D-CPW-15, spec only.

## ⊘ RETRACTION (operator ruling, 2026-09-30) — the word is a cell of the spatial-perturbation LUT; nothing is sliced from an embedding

The operator's statement, in order, verbatim where quoted:
- *"We don't want fuzzy embedding of 6 different readings of SPO."*
- *"I don't want any witness."* — *"I want the sentence's deterministic words."*
- *"No basin identity — superposition of 2 needles."* — *"Not rails."* — *"6 words each represented as LUT."*
- *"A word is spatial perturbation LUT"* — which is why the reading belongs to `cognitive-shader-driver`.

**The design, and all of it:**
- One calibrated COCA `palette256` codebook. Deterministic.
- **A word is one pair `(a:b)`: two `palette256` needles superposed — its cell in the 256×256 LUT.** The LUT is the word's representation. The colon is not a split of anything; it is the superposition.
- **A facet's 12 bytes are six words of the sentence.** Six words, deterministic from the text. Not six readings of one word; not S/P/O roles with witness pairs; not rails.
- Relations between words are LUT reads between cells. This is the machine `cognitive-shader-driver` already is: the p64 `CognitiveShader` over the bgz17 `palette256` semiring (`semiring.distance(query, target)`), the perturbation field `M[addr@coarse]·P(phase)` rendered around the cell. A sentence is six cells; the shader reads their field.
- Meaning is in the ClassView. The bytes stay content-blind.

**Struck by this ruling (append-only; the text below is kept as the record of the error):**
- §3.1 entirely: the 96-d embedding cut into six 16-d subspaces; per-subspace codebooks; `[a,b]` as nearest/second-nearest of an embedding slice; the retrain; the "home" argument built on that code.
- §3.2's word-register half: the "lens per subspace", the O/S/P plane assignment of `a`/`b`/spread, the "two registers" split. The 2³ mask stays what it always was in this workspace — Pearl over a triple — and is not re-derived here.
- §3.3 entirely (per-subspace sums, shared gamma across subspaces, `Z_SELF`, `M+`).
- Gates G-D, G-A, G-P, G-PO, G-M, G-T, G-ROT, G-UNIQ, G-CASC, G-ε, G1, G3, G3v, G5b as written (they measure the struck code).
- Deliverables D-CPW-0 (harness of the struck arms), D-CPW-3, D-CPW-4 (the retrain), D-CPW-12 (cascade wiring of the struck lens): **WITHDRAWN**. D-CPW-2 re-scoped: the pair type is the word-cell `(a:b)` over the one COCA codebook, with `ReferenceSet`; nothing about subspaces. D-CPW-10 re-scoped: the driver reads a sentence as six cells; its edges carry cells, not "subspace-0 needles".
- The v3/v4 measurements (0.617 / 0.774 / 0.786 on KJV) were measurements of the struck code and are not evidence about this design.

**Kept (independent of the struck half):** F23 three references and `ReferenceSet` (§3.7, D-CPW-14, G-REF); `self_code` as the basin's own identity (F25, D-CPW-7, G-SELF); the COCA prior `<f,c>` (L-1, D-CPW-9, G-FC); register-F (L-3, D-CPW-13, G-F); lemma routing (L-4, D-CPW-11); the Fisher-z LUT move into the contract (D-CPW-1) — which gains weight, since the LUT IS the representation.

**⊘ CORRECTION 2 (operator, 2026-09-30, after the artifact check) — `[a,b]` identifies a WORD, not two poles.** The artifact check found that nothing shipped, released, or in Tigris assigns a word two needles: the COCA CAM-PQ gives six (one per subspace), bgz17/bgz-tensor `nearest`/`assign` give one, and "(nearest : second-nearest)" originated in v1 with no source. The operator then pinned the unit of representation:

> *Six lexical slots, not six embedding subspaces. Each slot's two-byte address resolves through an immutable, versioned lexical codebook; the driver supplies the selected relational reading.*

- **The 96-bit payload is `[a_0,b_0] [a_1,b_1] … [a_5,b_5]`: six word slots.** Each pair is a 16-bit lexical address — 65,536 addressable cells, enough for either vocabulary — resolving to a word's **declared reading** (lemma + PoS) in an immutable, versioned lexical codebook (`ReferenceSet`, F23). The split into two bytes carries no semantics of its own.
- **Two interpretations were being silently conflated and are now separated:** (1) `[a,b]` identifies a word (unique identity, exact, this spec); (2) `[a,b]` identifies two lexical poles whose cell is a relation (not automatically a word identity). v4/v5's "nearest and second-nearest centroid" substituted (2) for (1). **(1) is the representation.** (2) is not a unit of identity anywhere in this spec.
- **Relations are the driver's job, not the bytes'.** The cognitive shader driver supplies the selected relational reading between two resolved words, calibrated by Fisher-z through the LUT. Fisher-z governs relation values; it never makes word identity fuzzy.
- **Embeddings are an optional comparison instrument, never the runtime intermediary and never what the bytes mean.** Embedding reconstruction is NOT an acceptance criterion (it would force the representation to imitate a different architecture). Every fidelity gate against Jina reconstructions is struck as a criterion; it may remain as a comparison report.
- **The first regression (G-LEX, replaces every fidelity gate as the first gate):** exact lexical address resolution — every declared vocabulary entry of a `ReferenceSet` resolves to exactly its declared reading; lemma/PoS distinctions survive round-trip; identical ordinals from different codebooks never cross-read (a resolution against the wrong `ReferenceSet` is refused, never a different word). This is G-REF's first half made concrete; D-CPW-14's correspondence artifact is its fixture.
- D-CPW-2 re-scoped: the slot type is a 16-bit lexical address with its `ReferenceSet`, resolved through the versioned codebook — no centroids, no nearest-anything.

Closeout: `STATUS: corrected | OUTCOME: [a,b] = a word's 16-bit lexical address in a versioned codebook; six word slots per facet; relations driver-supplied, Fisher-z calibrated; embeddings optional comparison only | OPEN: which relational reading(s) the driver selects between two resolved words (ClassView), and the LUT that carries them — not decided here`.

**⊘ CORRECTION 3 (operator, 2026-09-30) — `CausalEdge64` IS the ALU register.** One `u64`; the NARS truth, Pearl 2³ mask and inference operations run on it directly (`causal-edge/src/edge.rs:1-3`: *"One u64. One register. One read."*). It is not a reference, not a lens over something else, not a record in a store, and not a container for one word's bytes. "Baton" in older docs is this same register as it passes from one cycle into the next (operator, same day). Struck from this document wherever it says otherwise: §3.2's "a `CausalEdge64` lens per subspace" and its O/S/P plane table over a word's bytes; §3.4 L-2's "the driver's emitted `CausalEdge64` is a TRIPLE edge … S/P/O = the parsed words' subspace-0 needles"; and this session's chat framings of S/P/O as "register references" or of packing one word's `[a,b]` into S and O. The six word slots are the facet's stored identity; the register is where reasoning happens.

## 0. The rulings this version records

1. *No materialized point, no cosine* → similarity is Fisher-z table reads (§3.3).
2. *The duality/triplicity runs the NARS 2³ rung ladder in `CausalEdge64`* → the readings are `CausalMask` projections of one register, an escalation cascade (§3.2).
3. *6×2×8bit is part of the 512-byte SoA; the 6 [a,b] are the identity the other tenants chime in to* → the identity code is a content-blind facet of the row (§3.1).
4. *Use the calibrated COCA codebook — frequency, PoS and lemma tables on billions of words — as the fixed points* → §3.4.
5. *Three reference sets, kept distinct; correspondence preserved explicitly; an ordinal is never reused across references* → F23, §3.7, G-REF.

## 1. Frozen decisions

| # | decision | source |
|---|---|---|
| F1 | V3 atom = `classid(4) + 12 B`, content-blind; the ClassView selects the reading; slot purity; the `(8:8)` pair is polymorphic, classview-selected | `E-V3-FACET-4-PLUS-12` (archive `:23735-23738`); `le-contract.md` §1–2 |
| F2 | L4 = `6×(8:8) palette256²`; a plane's similarity is one table read; grounding = the COCA codebook. Under the `NeedlePair` reading a rail costs |mask| reads (§3.3) — the "one read" is per plane. | `le-contract.md:59` |
| F3 | Sub-byte carvings are lane readings, never a §3 payload layout | `causal_witness.rs:13-26` |
| F4 | No computed cosine; Fisher-z is the LUT; the table replaces the READ, not the semiring COMPOSE | `hexagon-plasticity-v1.md:500-513`; archive `:22506-22509` |
| F5 | No float on reasoning paths at read/dispatch time; offline table BUILD is float (`fisher_z.rs:200-219`) and that is the sanctioned derived side | `cosine-census-CONSOLIDATED.md:33-36` |
| F6 | "Without materialization" = no rendered-field cache; a per-class metric table is an ingredient | `E-PERTURBATION-CONVERGENCE-1` (archive `:21859`) |
| F7 | `CausalMask` 2³ over SPO planes; **Pearl Level 1 = `SO`**, Level 2 = `PO`, Level 3 = `SPO`; `SP` = confounder detection | `pearl.rs:6-49` |
| F8 | The driver's rung→mask convention: rungs 0–2 → `O` (**hand-chosen; NOT Pearl's Level 1**), 3–5 → `PO`, 6–9 → `SPO`; sustained BLOCK elevates, sustained FLOW relaxes to base. Certified only for the planner's triple planes; for the word register it is a hypothesis (§3.2). Reaches no distance call today. | `cognitive_shader.rs:221-250, 304-327`; `driver.rs:240-249, 691-693` |
| F9 | Compose = mask AND | `edge.rs:692-694` |
| F10 | `CausalEdge64` v2 layout (default): `S P O f c mask dir infer(i4) plasticity`, 53–63 reclaimed; `pack` (v1) drops `temporal`; doc block `:138-150` still shows v1 | `edge.rs:138-161, 186-242` |
| F11 | Frequency-rank = routing ⟂ meaning (ρ ≈ −0.07); count-derived meaning is a coarse floor | archive `:21063-21064` |
| F12 | The SPO role mask collapses homographs (809/809); S/O→nominal, P→verbal is a labelled simplification; adjectives fall outside | archive `:21796-21809` |
| F13 | Relations are stored edges; the substrate generalizes analogically | archive `:21066` |
| F14 | Fisher-z wins rank reads, loses level reads | v3 F11 |
| F15 | One copy; borrows never cross a mailbox; elsewhere takes an ADDRESS. **`NodeGuid` is the instance key, unique at mint (`debug_assert` in the default basin).** | `canonical_node.rs:1710-1728`; `CLAUDE.md` § CANON |
| F16 | Falsifiability: anti-vacuity, can-fire AND stay-silent, inertness | `CLAUDE.md` |
| F17 | Choices on validation; eval report-only | v3 |
| F18 | The embedding key is environment-only | v3 ledger 20 |
| F19 | Lance upstream; every other fork via AdaWorldAPI | `CLAUDE.md` P0 |
| F20 | The 8 masks are canonical contract constants with per-class election; the ladder is an escalation cascade, not 8 parallel views | `causal-rung-standing-wave-v1.md:39-42, 98-135`; `soa-32…:184, 217-224` |
| F21 | The nearest-vs-second gap is a pre-NARS match-strength signal, not a replacement for `<f,c>` | `bindspace-columns-v1.md:133, 136-139` |
| F22 | L1 (`part_of:is_a`) ⟂ L4 (ρ 0.035) | archive `:21463-21469` |
| **F23** | **Three reference sets, distinct, versioned:** the 4,096-word codebook (compact English), the 20k academic COCA codebook (broader academic), the 5k lemma table (lemma, PoS, frequency evidence). A reading contract names (reference id, version, digest). Immutability makes an ordinal stable WITHIN a reference, never interchangeable across them. Correspondence is preserved explicitly — through the lemma, with ambiguity and missing entries as values — never assumed nested or ordinal-aligned. | operator 2026-09-30; measured §3.7 |
| F24 | The second facet (bytes 16..32) is a content-blind facet cascade the ClassView reads; it is forbidden to "know it is an edge block" | `CLAUDE.md` § CANON, `E-THE-SECOND-FACET-IS-NOT-AN-EDGE-BLOCK-1`; `canonical_node.rs:786` |
| F25 | `EpisodicBasin.self_code` is the basin's OWN identity at a strictly higher rung than its inputs — not a cached member, not a copy | `episodic_basin.rs:22-31, 91-92` |

## 2. Input inventory (all ranges opened; graded by S3 and re-verified by R1)

### 2.1 Register, key, facets
- `facet.rs:77-135` `FacetCascade`; `:461-488` `tier_bytes`/`cascade_byte`; `:801-890` `CascadeShape`.
- `canonical_node.rs:444-475` byte-identical `From` impls `NodeGuid ↔ FacetCascade` and `NodeGuid::facet()` — the key admits the facet reading; `:786` `pub type EdgeBlock = EdgeFacet` (the second facet); `:862` `NodeRow` = key(16) | second facet(16) | value(480); `:1026/:1044/:1078` tenants 13/14/15; `:1284-1354` `ValueSchema`; `:1411-1423` `TailVariant` (V3 = the L1 6-byte tail, `guid-v3-tail`); `:1454-1483` `ReadMode`; `:1691-1696` `classid_read_mode`.
- `hhtl.rs:386-396` HHTL routing reads KEY bytes 4..12 — untouched by this spec (the identity is not in the key).
- `facet_schema.rs:21-55` `FacetSchema` from `(facet_classid >> 24) & 0b11` — provisional; under canon-high those are the domain byte's low bits (OSINT `0x07` → 3); no `.schema()` caller exists. **Not used by this spec.**
- `awareness_facet.rs:21-65` `SpoFacet` (six `(basin, identity)` pairs). No codebook producer: `text_stream_to_soa.rs:243-255` is an EXAMPLE writing the rank split; `facet_fold.rs:59-77` relabels fields.
- `tekamolo_facet.rs:1-59`; `causal_witness.rs:13-26, 112-151, 199-219`; `witness_fabric.rs:498-501, 588-589` (`Antecedent` → OFFSET only); `wave.rs:50-53, 80-83` (no text→loci producer); `fsm.rs:130, :160`.
- `tenants.md:56-58`; `le-contract.md:50-67`.

### 2.2 Edge and ladder
- `pearl.rs`; `edge.rs:8-134, 182-242, 353-413, 692-694`; `tables.rs:1-120` (revision indexed by c-quantile bucket midpoints `:78-79, :117-121`; deduction `c_out = f_out` `:69`).
- `cognitive_shader.rs:157-250, 268-335`; `causal_mask_bits` exercised only by tests `:774-864`, `doc_graph.rs:101, :116`, `driver.rs:691-693`; `causal_distance` once, `lance-graph-planner/src/cache/nars_engine.rs:135` (fixed `MASK_*` `:630`).
- `p64-bridge/src/lib.rs:45-89` (`edge_to_layer_mask`: bit2 → CONTRADICTS), `:105`, `:121-128` (predicate plane bit2 = SUPPORTS), `:383-429` (`cascade`: one `semiring.distance(query, target)` over ONE u8 palette, `k = semiring.k`).

### 2.3 Driver
- `driver.rs:1-23, 72-111, 240-249, 263-268`; `:323-327` resonance = bgz17 single-palette cascade; `:334-341` `_revised_truth = tables.revise(…, c2 = 128)` computed and discarded; `:376-387` `FreeEnergy::compose(top_resonance, std_dev)`; `:454-470` gate; `:472-497` emission (`row%256 / 0 / (row/4)%256`, `f = c`, mask `predicates & 0x07` `:489`, `style_ord_to_inference` `:491`, `temporal` dropped by v2 `pack`). `BackingStore` exposure of a row's facets at stage [5]: NOT verified — D-CPW-10 precondition.
- Pinning tests: `tests/p64_target_identity_probe.rs:103-113`; `src/edge_v3_compare.rs:81-108` (hand mirror), `:114-129`; `w2_differential.rs:166-170`; pass-through `engine_bridge.rs:788, 990`, `grpc.rs:102`, `wire.rs:1008`.
- `grammar/free_energy.rs:26-35, 104-129` (f32 surface; pins 0.8/0.2/0.05 "calibrated empirically").

### 2.4 Codec
- `deepnsm-v2/src/space.rs:38, 163` (`Cam96 = [u8;12]` bare alias), `:181-203, 219-224, 230-251, 257-272, 287`; `recipe_substrate.rs:88-147`; `contract/distance.rs:20-69, 82-108`; `bgz-tensor/src/fisher_z.rs:28-33, 63-68, 126-133, 158-166, 199, 200-219` (per-table `FamilyGamma`; diagonal saturates to 127, which is also the code of the closest off-diagonal pair; build is float); `ndarray/src/hpc/edge_codec.rs:46-70` (no second-nearest or `T_s` builder); `episodic_basin.rs:22-31, 76-144, 155-207`; `arigraph/episodic.rs:208-227, 945-970`; `deepnsm-v2/src/basin.rs:46`.

### 2.5 COCA fixed points and the three references
- `word_rank_lookup.csv` (5,050 rows; ranks ≤ 4096 → 3,559 distinct words); `vocabulary.rs:97-185` (`load` reads `word_rank_lookup.csv` AND `word_forms.csv`; idx = rank−1 `:139`; consecutive-only dedupe `:131-135`; never opens `lemmas_5k.csv`); `pos.rs:9-36`.
- `lemmas_5k.csv` (5,050 (lemma, PoS); 4,380 lemmas; `freq, perMil, range, disp`, 8 genres); `range`/`disp` read by no `src/` code; PM columns parsed by `freq_is_cosine.rs:51`.
- `academic_20k.csv` (20,845 rows; 18,559 distinct words; `ID, band, status, word, Pos, COCA-All, COCA-Acad, ratio, disp, range`). The Tigris artifact `lance-graph/codebooks/deepnsm-v2-academic-coca-v1/` = `MANIFEST.json` (entries 18,559 of 20,480 reserved; basins 73..79 empty; 2,286 duplicate surface forms) + the CSV (sha256 `1dfd5eda…` = the committed file) + `deepnsm_v2_academic_codebook.tsv` (`word_id, basin, slot, word, pos, coca_all, coca_acad, source_rank`) — **a `PaletteVocab` carve of the list, not a trained centroid codebook.**
- `deepnsm-v2/src/lexical.rs:1-42, 285-382, 398-487`; `vocab.rs:42-87`; `homograph_collapse.rs:37-51, 84-92, 259-263`; `deepnsm-morton-comma-facet-v1.md:149-162`; `gridlake_spo_ngrams.rs:10-12`.

## 3. The resolution (ratified)

### 3.1 The identity facet: L4 `[a,b]`, in the row's SECOND facet
- Per subspace `s ∈ 0..6` (16 d of the 96-d embedding): one codebook `C_s` (256 × 16-d, `Codebook::train`, deterministic seed), a Fisher-z table `T_s` (256×256 i8) under **ONE shared `FamilyGamma` for all six tables** (pin; per-table fit loss reported) so codes are commensurable when summed, and `N_a` = the other 255 centroids by `T_s(a,·)`.
- The rail is `[a, b]` = (nearest, second-nearest), **two full palette bytes**. No `j:t` nibble (F3). Position along the arc is not stored; G-T reports whether a LANE would earn it.
- **Codebook geometry changes:** 6 subspaces × 16 d × 256 centroids, a second-nearest encoder, a `T_s` builder — none exists. D-CPW-4 (ndarray fork, F19).
- **Home — decision (R1 BLOCK):** the code is NOT the instance key. `NodeGuid` stays the unique mint (F15) and HHTL routing (`hhtl.rs:386-396`) is untouched. The identity code lives in the row's **second facet** (bytes 16..32; F24), read as `NeedlePair` by classes that elect it; such a class has no in-row edge bytes (its edges live in `MaterializedEdges` / the planes — the per-class opt-out `CLAUDE.md` § CANON already provides). ⊘ v4's `TailVariant::V3` and draft v5's "the key IS the facet" are struck. Recorded alternative (Q6): a new 16-B `Identity` value tenant (RESERVE-DON'T-RECLAIM headroom exists; costs an additive lane).
- **Reading selection:** a ClassView-selected reading `PairReading::{RankSplit /* (basin, identity), F11 routing */, NeedlePair /* this spec */}` resolved through `classid → ClassView` (home of the field: Q5). Default for an unmapped class: `RankSplit` (the shipped convention). ⊘ `FacetSchema::PalettePair` withdrawn (§2.1).
- **Reference contract (F23):** every `NeedlePair` artifact and every ClassView naming it carries `ReferenceSet { id ∈ {COCA4096, COCA5K_LEMMA, COCA20K_ACAD}, version, sha256 }`. A code is meaningful only against its reference; mixing references is refused (G-REF).
- **Refusal:** a `NeedlePair` read requires the ClassView naming it AND the artifact digest AND a matching `ReferenceSet`; a rank-split, 12-axis, or other-reference register is refused, never reinterpreted. The pair type is a NEWTYPE (no `[u8;12]` enters the path).

### 3.2 The readings: a `CausalEdge64` lens per subspace; an escalation cascade (PROPOSAL, D-CPW-12)
| plane | value | source |
|---|---|---|
| O | `a` (needle) | stored |
| S | `b` (second pole) | stored |
| P | `q(T_s(a,b))` — the pair's spread (tightness), u8 | table read, derived (F21: pre-NARS strength) |
| `f, c` | §3.4 prior, revised | `NarsTables` |
| mask | the rung's (F8, convention) | `RungElevator::causal_mask_bits` |

| mask | geometry (§3.3) | v3 name | rung (F8) |
|---|---|---|---|
| `O` | needle | **D** (48-bit point) | 0–2 (convention; Pearl's L1 is `SO`) |
| `PO` | needle + tightness — **differs from `O`** | — | 3–5 |
| `SPO` | needle + pole + tightness = `SO + s_P` (`s_P` redundant given S; G-M measures whether it helps or hurts) | **A ≡ v3's M−t** | 6–9 |
| `SO` | pole pair | **A** | election |
| `P`, `S`, `SP`, `None` | marginals / confounder detection | — | election |

- v3's **M** (needle + neighbour + `t`) and **M−j** carried `t`; with `t` out of the key they have no mask. The `t` leg survives as G-T (`SO + t-lane`, `M+ + t-lane` vs `SO`). v3's **B** (quantized Fisher-z coordinates to PCA anchors) is **REJECTED**: a coordinate, not a palette address (F2), and a level read (F14) — recorded here and in §9.
- **Status: PROPOSAL.** `causal_mask_bits` reaches no distance call (§2.2); the cascade `O → PO → SPO` under sustained BLOCK is a deliverable (D-CPW-12), not shipped behaviour. F8's rung→mask certification is for the planner's triple planes; for this register it is a hypothesis tested by G-CASC.
- Two registers, distinguished: the `SpoFacet` TRIPLE register maps mask bits to rails (S = rail 0, P = rail 1, O = rail 2; witness rails 3–5 as contrast — `causal-rung-standing-wave-v1`); this WORD register maps them to planes inside one rail. A third, the p64 predicate-plane byte, is what the driver's `predicates` carries (bit2 = SUPPORTS) and what `edge_to_layer_mask` inverts (bit2 = CONTRADICTS) — the disagreement is `ISS-CE64-EMIT-INVERSE-BIT2-DISAGREE`. The rung, Pearl and p64 masks are three vocabularies; nothing here unifies them.
- Direction: `O = a` kept (L2 drops the neighbour, not the word); G-A carries a swapped-byte control. Q1 stays open.
- Cite: `E-H268-REPLAYABLE-TILE-1`, `E-RECIPE-SUBSTRATE-WIRING-1`, `E-PROOF-IS-A-REGISTER-READING-1`, `E-PARTOF-ISA-vs-PALETTE256-1`, A2 `PearlRungFacet`.

### 3.3 Similarity: table reads over the active planes, one shared scale
```
s_O = z(a, a')                      one T_s read           (similarity; i8 ≤ 126 off-diagonal)
s_S = z(b, b')                      one T_s read
s_P = 127 − |q(T_s(a,b)) − q(T_s(a',b'))|   a similarity of spreads (u8 difference; a LEVEL read — F14; gate G-P)
sim_s(m) = Σ_{plane ∈ m} s_plane     one shared gamma ⇒ one scale across the six subspaces
sim(A,B) = Σ_s sim_s(m)
```
- No slerp, sqrt, division, reconstruction at read time; F4/F5/F6 hold at read time (the table build is float, offline — F5's derived side).
- **Diagonal by address:** `Z_SELF = 127` is reserved; off-diagonal codes clamp to `≤ 126` at build. An exact needle match therefore strictly beats its closest neighbour. `T_s(a,a)` is never read (its gamma is fitted on `i<j`).
- **`M+` arm:** `s_O + s_S + z(a,b') + z(b,a')` — v3's four-cell Gram sum with `w = 1`, in z. Four table reads. Not a perturbation-field render.
- The similarity that RANKS (this section) and the resonance the driver REVISES with (§3.4 L-2, bgz17 cascade over the triple palette) are different metrics on different registers; §3.4 makes the driver's palette a `T_0` table so the two agree on the needle plane.

### 3.4 The COCA fixed points in the driver (the lever)
The COCA columns never enter the 12 identity bytes (v3 G4; F11) nor the similarity (F5). They enter `<f, c>` and the F gate.

| lever | commitment | where |
|---|---|---|
| **L-1 prior truth** | The prior edge of a `(lemma, PoS)` identity row asserts *"this reading is the identity of its surface form"*. **`f` = the reading's share among its surface form's readings** (`form_count(this) / surface_count(surface)` from `LexicalEvidence`; F12 made numeric; `f = 1` for an unambiguous form; unknown counts stay unknown, never 0). **`c = w/(w+1)`, `w = ln(1+freq) / ln(1+freq_max) · K`** (evidence amount; `K` a labelled pin). Quantized to u8 by the loader; no float at dispatch. ⊘ `range`/`disp` leave the truth (collinear; `disp` is evenness, not evidence). | new reader of `lemmas_5k.csv` (`freq`; genre PM reused from `freq_is_cosine.rs:51`) beside `Vocabulary::load`; `LexicalEvidence` for the share; rank-alignment test |
| **L-2 edges carry identity (triple register)** | The driver's emitted `CausalEdge64` is a TRIPLE edge: `S/P/O` = the parsed subject/predicate/object words' **subspace-0 needles** `a_0` (the FSM assigns roles — F12; embedding subspaces are NOT roles, so no PoS→subspace map exists or is claimed). The p64 semiring for those planes is `T_0` (so the cascade's resonance and §3.3's needle plane agree). `<f, c>` = the L-1 prior revised with the hit's resonance through `NarsTables::revise` (wiring the discarded `_revised_truth`; bucket resolution accepted). `pack_v2`. ClassView-gated (home per Q5): rows whose class does not name `NeedlePair` keep the legacy `row%256` path — G-EDGE's disable run. Tests `p64_target_identity_probe.rs:103-113` (storno) and `edge_v3_compare.rs:81-108` (lockstep) named. Precondition: `BackingStore` exposes the row's facets at stage [5]. | `driver.rs` stage [5] |
| **L-3 register F** | `kl` = divergence of the window's rank histogram from the best of the 8 genre priors — **register detection, routing not meaning (F11)**. It is the existing f32 `FreeEnergy` surface (a derived quantity, F5), with a quantized histogram. ClassView-gated like L-2; `std_dev` stays the default; pins unchanged; G-F measures. `FreeEnergy::kl_from_prior` is a carrier method. | `driver.rs:379-387`; `free_energy.rs` |
| **L-4 lemma routing** (D-CPW-11, Blocked) | Identity rows are `(lemma, PoS)`; surface forms route via `LexicalEvidence`; OOV / named entities have no identity row and stay unknown (never rank 0). `Antecedent` resolves to an offset only and no text→loci producer exists. | `deepnsm-v2` |

Two algebras never merge (F13): structure (identity bytes, mask) vs strength (`f, c`).

### 3.5 `self_code`, Tekamolo, CausalWitness
- `EpisodicBasin.self_code` is the basin's own higher-rung identity code (F25) — it stays, 12 B, same lane, no layout change. Its READING is per class (`NeedlePair` vs legacy 12-axis), computed at basin promotion from decoded points (v3 G7; offline, float allowed). **Gate (R3 BLOCK):** the `NeedlePair` reading applies only where the class names it; a non-zero legacy `self_code` under a `NeedlePair` class is refused; a paired legacy-nonzero test in the style of `pal8_v1_nonzero_temporal_is_blocked_by_version_gate`. D-CPW-7.
- Tekamolo (13) and CausalWitness (14) read their own 12 B and point at identity rows; G-ROT asserts the `G4D3` rotation of a `NeedlePair` register is meaningless.

### 3.6 Legacy
`Cam96Space` (12 axes) stays; `Cam96AxisSpace` alias; pair NEWTYPE in the contract; deepnsm-v2 ships codebook artifacts; `CAM96P01`/`CAM96PW1` + digest + `Cam96PairCodes<'a>` unchanged (v3 §6), each header gaining `ReferenceSet`.

### 3.7 The three reference sets (F23) — measured 2026-09-30
| reference | rows | distinct keys | key |
|---|---|---|---|
| COCA4096 (`word_rank_lookup.csv`, ranks ≤ 4096) | 4,096 ranks | 3,559 words | rank (homographs share) |
| COCA5K_LEMMA (`lemmas_5k.csv`) | 5,050 | 4,380 lemmas / 5,050 (lemma, PoS) | (lemma, PoS) |
| COCA20K_ACAD (`academic_20k.csv`; Tigris carve v1) | 20,845 | 18,559 words / 20,842 (word, Pos) | `word_id` = admission order; basins 0..72 |

Correspondence: 5k ∩ 20k = 4,895 (lemma, PoS) of 5,050; 4,264 words; **116 of the 5k lemmas are absent from the 20k**; 4096 ∩ 20k = 3,462 of 3,559; **only 3 of 4,264 shared words carry the same ordinal in the 5k and the 20k** — the sets are neither nested nor aligned. The 20k drops 2,286 (word, Pos) duplicates to one id; the 4096 shares ranks across homographs; the 5k keeps them distinct.
- **Artifact (D-CPW-14):** `lexical_correspondence.tsv` — one row per `(lemma, PoS)`: `id4096 | id5k | id20k | status ∈ {Exact, Ambiguous{n}, Missing}`, with the three `ReferenceSet` digests in its header. Generated, never hand-edited.
- **Regression invariant (G-REF):** any switch of reference preserves the declared correspondence and never reuses an ordinal as if unchanged: re-derive the table from the artifacts and diff; a code carrying reference X read against Y is refused (§3.1).

## 4. Non-goals
Phrases (licensed n-grams); Tekamolo registration in §3; `PairPalette` rename; the 96-subgenre palette; tesseract-rs migration (after D-7); the three F pins; wiring `causal_mask_bits` into the planner's `causal_distance`; the bit-2 emit/inverse disagreement (ISSUES); a trained ACADEMIC centroid codebook (the Tigris artifact is a vocabulary carve — recorded, not built here).

## 5. Pre-registered gates (validation decides, eval reports; every threshold a labelled hand-tuned pin; "floor" = v3 G3's word-blocked bootstrap floor)
| gate | criterion | fires / stays silent |
|---|---|---|
| G-L4 | pair NEWTYPE; no nibble type; `compile_fail` `Cam96` ↔ pair; a `NeedlePair` read without ClassView + digest + `ReferenceSet` is refused | fires on `(j:t)`, rank-split, wrong reference |
| G-REF | `lexical_correspondence.tsv` re-derived == declared; a cross-reference read is refused; the "3 of 4,264" ordinal fact reproduced | fires on a reused ordinal |
| G-UNIQ | structural: the key is unchanged by the identity facet; report: rows per identical L4 code (collision census) | — |
| G-D | `O` ρ ≥ in-harness 48-bit − floor; shuffled `a` drops below | shuffled `a` fires; identity permutation of the table stays silent |
| G-A | `SO` − `O` > floor; swapped-byte control; `b := random N_a` fires | — |
| G-P | `s_P` predicts the `O`-only rank error per word (Spearman ≤ −0.3 pin); permuted spread → |ρ| < floor | both halves |
| G-PO | `PO` ρ − `O` ρ reported; if within floor the rung-3 widening is decoration and D-CPW-12 records it | — |
| G-M | `SO` vs `SPO` vs `M+`; tie within floor ⇒ the simpler ships (`SO`) | — |
| G-T | report: `SO + t-lane`, `M+ + t-lane` vs `SO` | — |
| G-CASC | the cascade under sustained BLOCK yields a strictly wider candidate set at each rung on a fixed query set; never narrower (superset-monotone) | narrowing fires |
| G-ROT | `G4D3` rotation of `NeedlePair` bytes |ρ| < floor; structural no dual read | a ranking rotation fires |
| G0 / G-ε | v3 G0 (ii)–(iv) reported; encoder `ε`; `N_a` excludes `a` | — |
| G1 | shipping mask eval ρ ≥ in-harness 12-axis; precondition 48-bit ≤ 12-axis else HARNESS-SUSPECT | — |
| G3 / G3v | v3: field-carries-information vs floor; Voronoi change rate | — |
| G-F | with L-3, F separates KJV windows from `academic_20k`-register windows (AUC ≥ 0.75 pin); uniform prior must NOT separate; `std_dev` control | both halves |
| G-PoS | triple edges with FSM roles vs permuted roles: used_for purity drops ≥ 0.05 (pin) | — |
| G-FC | prior `c` predicts revision magnitude with `f` partialled out (partial Spearman ≤ −0.3 pin); permuted `c` → |ρ| < floor | both halves |
| G-EDGE | `NeedlePair` rows: emitted `S/P/O == a_0` of the parsed words; pinned rows: `the` (`f = 1`), `record` (`f < 1`); legacy rows keep `row%256` (disable run) | — |
| G-SELF | a non-zero legacy `self_code` under a `NeedlePair` class is refused (paired test) | — |
| G-LEM | (D-CPW-11) in-vocabulary antecedents only | — |
| G2 / G4 / G5 / G5b / G6 / G7 / G8 | v3, unchanged (G5b: one shared gamma, per-subspace fit loss reported) | — |

## 6. Council record
Phase 1 (5): S1 prior-art (Opus, 10), S2 iron-rules (10, YIELDS-WITH-AP), S3 code-truth (Q1 table + 8), S4 cascade (8), S5 different-views (10). Phase 3 (3) on draft v5: R1 overclaim (1 BLOCK, 7 P1, 6 P2), R2 dilution-collapse (0 BLOCK, 5 P1, 5 P2), R3 firewall (1 BLOCK, 3 P1, 2 P2). Stricter verdict applied throughout. Banked: scratchpad `c96p4-council/{s1..s5,r1..r3,draft-v5,ref-correspondence}.md`.

## 7. Operator questions (open)
- Q1 `O = a` vs `S = a` (F8 convention; the point-vs-pair gap below rung 3).
- Q2 (closed by F25): `self_code` stays; encoding of its reading per class → envelope-auditor.
- Q3 the 96-d embeddings (block D-CPW-0); the ACADEMIC reference has no embedding artifact at all.
- Q4 the `t` lane, if `SO` clears G-M.
- Q5 the `PairReading` field's home (`ReadMode` vs `ClassView`).
- Q6 second facet vs a new `Identity` tenant as the home (§3.1 commits the facet; the tenant is the recorded alternative).
- Q7 whether ACADEMIC gets its own trained codebook (a fourth artifact) or shares COCA4096's centroids with a correspondence-mapped vocabulary.

## 8. Board hygiene (this commit)
`INTEGRATION_PLANS.md` PREPEND; `STATUS_BOARD.md` v5 block (§10); `AGENT_LOG.md` council entry; `entries/2026-09-30-three-reference-sets-are-not-ordinal-aligned.md` + `entries_index.py --write`; `ISSUES.md` (`ISS-CE64-EMIT-INVERSE-BIT2-DISAGREE`, `ISS-DID-PATTERN-EXCLUDED-D-C96P`); `TECH_DEBT.md` (adjective/adverb role gap; `causal_mask_bits` unwired; provisional `FacetSchema` field aliases the domain byte; six per-table gammas); v3/v4 status lines → SUPERSEDED-IN-PART / SUPERSEDED; EPIPHANIES: **no Eureka** (the ordinal finding is a measurement; the home decision is a decision); `SUPERSESSION-INDEX.md` regenerated LAST.

## 9. Change ledger

### draft v5 → ratified v5 (Phase 4, from the three)
| # | change | source |
|---|---|---|
| 1 | Home moved from the KEY to the second facet; key uniqueness and HHTL routing untouched; G-UNIQ | R1 BLOCK |
| 2 | `self_code` stays as the basin's own identity (F25); per-class reading + refusal + paired test (G-SELF) | R3 BLOCK, R2-7, R1 P2 |
| 3 | `P` back in the score as a spread similarity; `PO ≠ O`; `SPO = SO + s_P` stated; G-PO | R2-3, R1-1 |
| 4 | One shared gamma; `Z_SELF = 127` reserved, off-diagonal ≤ 126; build-is-float stated | R1-2 |
| 5 | L-1: `f` = reading share, `c` from `ln(1+freq)`; `range`/`disp` out; proposition stated; u8 at load | R1-3, R2-5, R3 |
| 6 | L-2 on the TRIPLE register (FSM roles → subspace-0 needles), semiring = `T_0`; PoS→subspace map disclaimed; ClassView-gated | R2-6, R1-4, R3 |
| 7 | §3.2 graded PROPOSAL with D-CPW-12 and G-CASC; F7/F8 Level-1 mismatch named | R1-1, R1-6, R2 §1 |
| 8 | B rejected with reason; M/M−j mapped to G-T; G0/G3v reinstated; G-M tie rule | R2-1/2, R2 §5 |
| 9 | G-FC re-specified (partial out `f`, permuted `c`); G-EDGE pinned rows | R1-5, R1 P2 |
| 10 | L-3: existing f32 surface labelled; gated; D-CPW-13 split from D-CPW-10 | R3 P1, R2 §10 |
| 11 | §8: TECH_DEBT, plan file, EPIPHANIES outcome; G2/G3 rows; alias column marked non-matching | R3 |
| 12 | F23 three references; §3.7 measurement; `ReferenceSet` in the contract; G-REF; D-CPW-14; Q7 | operator 2026-09-30 |
| — | **Not adopted:** R2's suggestion to keep `d_P` as a distance (it is a similarity of spreads now — same leg, right polarity). R1's HHTL note is moot once the key is untouched. | — |

### v4 → draft v5 (Phase 2) — see scratchpad `draft-v5.md` §9; summary: `TailVariant` struck; `FacetSchema::PalettePair` withdrawn; newtype; `M+` relabelled; cascade wording; L-4 split; thresholds labelled; inventory corrections (S3).

### v3 → v4 (Phase 0) — `-v4.md` §9.

## 10. Deliverables
The `D-C96P-*` aliases (v1–v3) do not match the D-id pattern and are cited here only as history; tooling joins on `D-CPW-*`.

| D-id | (alias) | scope | gates | status |
|---|---|---|---|---|
| D-CPW-0 | D-C96P-0 | harness: masks + `M+` + `t`-lane arms; in-harness 12-axis / 48-bit / RQ controls; split file | G-D, G-A, G-P, G-PO, G-M, G-T, G-ROT, G0, G-ε, G1, G3, G3v, G5b | Blocked (embeddings, Q3) |
| D-CPW-1 | D-C96P-1 | Fisher-z codec into the contract; shared gamma | golden-bytes parity | Queued (authorized) |
| D-CPW-2 | D-C96P-2 | pair NEWTYPE + `PairReading` + `ReferenceSet` + refusal | G-L4, G-REF, G-UNIQ, G2, G5, G6 | Queued (after Q5) |
| D-CPW-3 | D-C96P-3 | loaders, digest, view (v3 §6) + `ReferenceSet` header | G6 | Blocked (on D-0) |
| D-CPW-4 | D-C96P-4 | trainer: 6×16-d retrain, second-nearest encoder, `T_s` builder (ndarray fork) | G1 | Blocked (on D-0) |
| D-CPW-5 | D-C96P-5 | `Cam96AxisSpace` alias; legacy kept; refusal | G4, G6 | Blocked (on D-2) |
| D-CPW-6 | D-C96P-6 | `SemanticSpace` routing-only | G4 | Queued |
| D-CPW-7 | D-C96P-7 | `self_code` per-class reading + refusal + paired test | G-SELF, field-isolation matrix | Queued (envelope-auditor) |
| D-CPW-8 | D-C96P-8 | consumer migration | G7, G8 | Blocked (on D-5, D-7) |
| D-CPW-9 | — | COCA prior reader (`freq`; PM via `freq_is_cosine.rs`) + reading-share from `LexicalEvidence` + rank-alignment test | G-FC | Queued (collides with D-LXC-4) |
| D-CPW-10 | — | driver L-2 (triple edges, `T_0` semiring, `pack_v2`, revision wired, test storno) | G-EDGE, G-PoS | Blocked (on D-2, D-9; BackingStore precondition) |
| D-CPW-11 | — | L-4 lemma routing + `Antecedent` → row | G-LEM | Blocked (no text→loci producer) |
| D-CPW-12 | — | rung → mask → similarity wiring (the cascade) | G-CASC, G-PO | Blocked (on D-0, D-2) |
| D-CPW-13 | — | driver L-3 register F (`kl_from_prior`) | G-F | Blocked (on D-9) |
| D-CPW-14 | — | `lexical_correspondence.tsv` generator + `ReferenceSet` digests | G-REF | Queued |
| D-CPW-15 | — | the execution socket (§11): enforce the firewall + integer-only law on `lance-graph-mask-risc`, and pin the full-width `u64` register lane | G-SOCK-1..5 | Queued (spec only in #1303; next PR per §11.8) |

## 11. The execution socket under the lexical substrate (D-CPW-15, 2026-09-30)

**Scope:** a ruling, not an implementation. It names the narrow seam everything in §1–§10 eventually lowers into, so that the semantics above can keep evolving without an ABI revision below. No code lands with it.

### 11.1 The socket already exists — it is `lance-graph-mask-risc`'s IR, not a new "Quack ABI"
Inspection (2026-09-30) found the proposed socket shipped under other names. **`lance-graph-quack` is the DuckDB-shaped SURFACE** (D-QCK-0..10): it builds programs and never evaluates one. The socket is the IR it lowers to:
- **Program:** `Program { ops: Vec<MaskOp>, terminal: Terminal, scratch_slots }` — straight-line, one terminal.
- **Structural views (borrowed, never owned):** `Planes { n_rows, masks: &[&[u64]], lanes: &[LaneRef] }` with `LaneRef::{I32, U32, U64, Strided}`, plus `Foreign { planes, lanes }` for another table's row space.
- **Results:** `Value` + caller-owned `Out::{None, I32, I64, Mask}`; `ExecError` is shared by executor and oracle.
- **Guarantees already enforced:** a row-at-a-time `reference` oracle that never touches ndarray; the executor-vs-oracle differential; tiled == single-tile; zero allocation (`tests/no_alloc.rs`); `the_crate_names_no_isa`.

The stack in this repo therefore reads, top to bottom: DuckDB-shaped `quack` / Java `lgj-abi` (`plan_eval`) / cognitive consumers → **mask-risc IR (the socket)** → `ndarray::simd` (the only backend layer: scalar, AVX2, AVX-512, NEON, WASM). DuckDB and Java sit ABOVE the socket as lowerings; Arrow is an input VIEW, not a backend.

### 11.2 The invariant (authoritative)
> **The socket operates on borrowed masks, borrowed integer lanes and caller-owned sinks. It knows no COCA, lemma, PoS, `ReferenceSet`, Fisher-z, NARS, Pearl, EWA, JC/Jirak, HHTL, classid meaning, or `cognitive-shader-driver`. Every meaning is decided above it, by the lowering that builds the `Program`; semantic change above never requires an IR change below.**

Evidence it already holds: mask-risc's only dependency is `ndarray` (facade features); it reaches ndarray only through `ndarray::simd`; its `src/` has zero hits for the cognitive terms above (NARS, Pearl, COCA, Fisher, lemma, `ReferenceSet`, EWA, Jirak, HHTL, shader, thinking, `CausalEdge`). It does name V3 **storage geometry** (26 hits: `facet`, `classid`, `ClassView`, `NodeRow`), almost all in doc comments that explain why a lane width or stride exists. It names that geometry once in an op: `Pred::MatchFacetStrided`, a content-blind 12-byte ternary match whose name records a layout, not a meaning. Geometry is allowed below the line; meaning is not. Whether that op should carry a geometry-neutral name is **OPEN** and out of scope.

### 11.3 Primitive vocabulary after inspection — three families, not six verbs
| proposed verb | what the IR already has | verdict |
|---|---|---|
| MASK | `MaskOp::{Pred, And, Or, Xor, AndNot, Not, Ternlog}`, `Pred::Range`, gated evaluation via `under` | **primitive** (family 1) |
| SELECT | a selection IS a mask; `Terminal::Keep` returns it, `Terminal::BlendI32` is the CASE shape | **not a primitive.** An index-compacting select is the `SelectionVector` the IR forbids by law (A1; matrix R1 ELIMINATE). The one row-id exit is the named `materialize_rows`. |
| JOIN | `MaskOp::Gather` (pull a foreign mask through an index lane), `Pred::EqU32Via` (predicate through an fk), `GroupKey::Via` (group key through an fk) | **primitive** (family 2, TRANSPORT) — mask transport through an index lane; no bag-semantics join exists or is wanted |
| TRAVERSE | one step: `Gather` (pull) and `Terminal::ScatterOrU32` (push, the one-to-many hop) | **one step = TRANSPORT.** Multi-hop iteration is NOT in the IR (a scattered mask may not feed another program — its survival condition) — **OPEN**, see §11.8 |
| FOLD / REDUCE | ONE terminal per program: `Count/Any/All`, `MaskedSum/Min/MaxI32`, `MaskedStridedGroupSum`, `GroupSumI32`, `GroupSumViaI32`, `GroupReduce{GroupKey, GroupFold}`, `CountKeyRunsU32`, `ScatterCountU32` (held) | **one primitive** (family 3, REDUCE). "Fold" is also the name of the write-side per-row fold in `persist_sink`; do not reuse it here. |

So: **MASK · TRANSPORT · REDUCE**, over the structural views of §11.1. No primitive is added by this ruling. A new one earns its place only when a named workload cannot lower without it (the missing-capability STOP rule).

### 11.4 Where `CausalEdge64` sits — above the line; it crosses as a `u64` lane
- The register crosses as **`LaneRef::U64`**, borrowed through the existing `MailboxSoaView::edges_raw() -> &[u64]`. `CausalEdge64` is `#[repr(transparent)]` over `u64`, so the borrow copies nothing. No wrapper type is minted.
- **`CausalEdge64` itself is not an IR type.** Its field layout is feature-gated (`causal-edge-v2-layout`, default; v1 under `--no-default-features`). Naming it in the socket would couple the IR to that layout. Instead the consumer computes `Pred::MatchU64 { pattern, care }` from the register's own accessors above the line, and the IR compares bits it does not interpret.
- **Bit preservation is the requirement, not an option.** The IR never decodes or re-encodes a lane, so every bit survives, including the defining topology/band bits 59–63 that F-BBB-NARS-2 protects against aliasing.
- **NARS arithmetic does not enter this IR.** `membrane-tiers.md` rules two sibling T1 algebras: the population algebra (this socket) and the epistemic one (`TruthU8`, revision, …, D-BBB-NARS-1..4). A future `Truth(…)` plan operation is the epistemic sibling's, named by T2 `plan_eval`; it is not a mask-risc op.

### 11.5 The factorized adjacency at the boundary, and Arrow `ListArray`
- **Today's factorized form is an index lane over the child population:** a `U32` lane naming, per child row, its row in the other table (`Gather`, `EqU32Via`, `GroupKey::Via`, `ScatterOrU32`). A key-ordered lane is also a segmentation: `CountKeyRunsU32` counts runs with O(1) state and refuses an unordered lane (`LaneNotOrdered`).
- **Arrow `ListArray` does not map today without materialization.** A `ListArray` carries `offsets[parents + 1]` + child values; the IR has no offsets view. The child values borrow fine as a lane; the SEGMENTATION would have to be expanded into a per-child parent-index lane (O(children) — a derived-lane materialization). **OPEN:** the zero-copy route is one structural view (a segment/offsets view) plus a parent-mask → child-mask expansion (a union of `mask_set_range` writes) and its reverse. Filed as `ISS-MASK-RISC-HAS-NO-SEGMENT-VIEW`; not built here.
- **Arrow bitmaps meet the tail law.** An Arrow validity bitmap reinterpreted as `&[u64]` must have zero bits past `n_rows`, or the program is refused (`ExecError::PlaneTail`) rather than read. Arrow validity is a NULL role, which the socket eliminates (matrix R6); a bitmap enters only as a plane.

### 11.6 The dependency firewall (named crates)
- **`lance-graph-mask-risc` `[dependencies]` is exactly `{ ndarray }`**, and `src/` reaches it only as `ndarray::simd` — never `ndarray::hpc` (which carries `nars`, `causal_diff`, `styles`, `cascade`).
- **Forbidden in `[dependencies]` and `[dev-dependencies]`** (all real crates in this workspace): `lance-graph-contract`, `causal-edge`, `cognitive-shader-driver`, `thinking-engine`, `deepnsm`, `deepnsm-v2`, `lance-graph-planner`, `lance-graph-cognitive`, `lance-graph`, `p64-bridge`, `bgz17`, `bgz-tensor`, `highheelbgz`, `holograph`, `jc`, `perturbation-sim`. `lance-graph-quack` and `lance-graph-report` too: they are consumers ABOVE the socket.
- **Direction:** consumers depend on the socket, never the reverse — `quack` → mask-risc; `lgj-abi` → mask-risc (it already does); a cognitive consumer that lowers into it → mask-risc. The contract dependency is forbidden on purpose: `lance-graph-contract` carries the cognitive vocabulary (`cognitive_shader`, `nars`, `thinking`).

### 11.7 Falsifiers (G-SOCK-*) and what "identical" may honestly claim
- **G-SOCK-1 exact equivalence.** For every program in the differential corpus, executor `Value` and every `Out` buffer `==` the reference oracle, and tiled `==` single-tile. **Shipped** (`tests/differential.rs`). Backend realizations are ndarray's to prove (its masking-parity matrix: native / nightly / wasm / wasm-scalar / neon-qemu); mask-risc is backend-independent by construction, but its own suite runs in CI on `x86-64-v3` only (D-MRX-5: NEON/WASM/scalar arms unexercised for this crate), so "identical on every backend" is ndarray's measured claim plus mask-risc's by-construction one — not a mask-risc measurement on each arm.
- **G-SOCK-2 integer-only law.** `==` is honest only because the IR carries no floating point (`i32`/`u32`/`u64` lanes, `i64` sums with stated row bounds that refuse rather than wrap). A float reduction may never join the `==` suite; if one is ever admitted it gets its own declared tolerance class. Fisher-z values, NARS `(f, c)` in f32 and `PerturbationDto` energy therefore stay above the socket.
- **G-SOCK-3 firewall.** `[dependencies]` keys `== ["ndarray"]`; no `ndarray::hpc` in `src/`; zero COGNITIVE vocabulary in `src/` code AND comments (the §11.2 term list — storage-geometry words are allowed). Disable-runs: adding any forbidden crate, and adding one listed term to a `src/` line, each turn it red.
- **G-SOCK-4 full-width register lane.** `Pred::MatchU64` with `care` over high fields (for example bits 40–42 and 59–63), not only low bits: executor `==` oracle `==` an independent per-row `(x ^ pattern) & care == 0` count, two-sided anti-vacuity (`0 < selected < n`). Disable-run: truncating the lane to 32 bits must redden. **Gap measured:** today's differential exercises `MatchU64` only at `care: 0xF000` (bits 12–15), so a 32-bit truncation would pass it.
- **G-SOCK-5 no materialization.** Already enforced (`no_alloc.rs`; tiled state is `slots × TILE_WORDS` words, independent of `n_rows`; only demanded `Out` sinks are population-sized).
- **Later, with the segment view:** the same program over a lane borrowed from an Arrow buffer and over one borrowed from a `Vec` answers `==`; an Arrow bitmap with a dirty tail is refused.

### 11.8 The smallest next PR (not in #1303)
**Title:** `mask-risc: D-CPW-15 socket guards`. **One crate, no cognitive vocabulary, no new primitive.**
1. `crates/lance-graph-mask-risc/tests/firewall.rs`: `dependencies_are_exactly_ndarray` (reads its own `Cargo.toml` via `include_str!`), `src_reaches_ndarray_only_through_simd`, `src_names_no_cognitive_vocabulary`, `ir_is_integer_only` (production text of `ir.rs` + `value.rs`). Each disable-verified.
2. `crates/lance-graph-mask-risc/tests/u64_register_lane.rs`: G-SOCK-4 on a synthetic `u64` lane (seeded values, several `pattern/care` fields up to bit 63, gated and ungated, tiled and single-tile).
3. Board: D-CPW-15 → Shipped; AGENT_LOG only if workers ran.

**It demonstrates value with no cognitive vocabulary at all:** "factorized graph operations over compact registers and borrowed structural views, without materialization" is exactly what the existing suite plus these two files prove.

**Second PR (after):** the cognitive consumer binding. It is a dev-dependency test in `cognitive-shader-driver`, the only side allowed to know both. It borrows `MailboxSoaView::edges_raw()` as `LaneRef::U64`, builds `pattern/care` from `CausalEdge64` accessors, and diffs the count against a per-row decode. It also checks that the borrowed slice's pointer equals the column's (no copy).

**Deliberately NOT implemented, here or next:** the segment/offsets view and any Arrow adapter; multi-hop traversal; an epistemic (NARS) op; any float reduction; GPU; renaming `quack` or `mask-risc`; moving `CausalEdge64` fields; any change to lexical representation or Fisher-z calibration; #1304's multi-reading problem.

**Open for the operator:** whether "Quack" should become the umbrella name for the socket. In the repo today it names only the DuckDB-shaped surface above it; this ruling does not rename anything.

Closeout: `STATUS: ruled (spec only) | OUTCOME: the execution socket is lance-graph-mask-risc's existing IR; three primitive families (MASK · TRANSPORT · REDUCE); CausalEdge64 crosses as a borrowed u64 lane and stays an above-the-line type | OPEN: segment view for Arrow ListArray; multi-hop traversal; umbrella naming`.
