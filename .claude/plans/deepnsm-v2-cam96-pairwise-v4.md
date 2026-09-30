# deepnsm-v2 Cam96 pairwise — v4: the L4 identity facet, its 2³ readings, and the COCA fixed points

**Status:** PHASE-0 SPEC (council input). Supersedes `-v3.md` §3 (reading), §6.3 (home), §7 decision table. Everything in v3 not named in §9 below stands.
**Date:** 2026-09-30. **Council:** `/5plus3`, second invocation on D-C96P (deliberate re-invocation per the harness base case: v3 was ratified, then three operator rulings changed its premises).
**Panel:** default five (prior-art on Opus, the rest Sonnet); default three.

## 0. Why v4 exists — three operator rulings after v3

1. *"I'd rather go Perturbationsfeld than violating the cosine replacement centroid materialisierung."* v3 §3 rendered `x̂ = slerp(u_a,u_b,t)·n̄` (a materialized point between centroids) and scored it with a cosine (sqrt, division, a Gram matrix built from atanh values). Both are struck.
2. *"The duality/triplicity runs the NARS decomposition rung ladder 2³ in CausalEdge64."* v3 treated the readings A/B/D as ablations of M with a decision table that drops a tied reading. A reading is not an arm. The readings are the `CausalMask` projections of one register; the rung ladder selects the mask.
3. *"6×2×8bit is part of the 512-byte SoA. The 6 [a,b] should represent the identity the other tenants chime in to."* v3 §6.3 homed the code in `EpisodicBasin.self_code` (a value tenant). The code is the **key facet** under `le-contract.md` §3 **L4**; Tekamolo and CausalWitness point at it.

Plus the driver mandate: *"use the scientific value of the already calibrated COCA codebook where frequency, PoS and lemma tables are calibrated on billions of words … the fixed points to unhinge the world with a lever."*

## 1. Frozen decisions (cite; the council may only file VIOLATES with file:line)

| # | decision | source |
|---|---|---|
| F1 | The V3 atom is `classid(4) + 12-byte payload`; the 12 B is content-blind; the ClassView selects the reading; slot purity (no labels in the payload) | `E-V3-FACET-4-PLUS-12` (archive `:23735`), `le-contract.md` §1–§2 |
| F2 | **L4** = `6 × (8:8) palette256²`, CAM_PQ "digital": each byte pair indexes the 256×256 palette table; similarity = ONE table read. Grounding: *"the codebook is DeepNSM's 4096-word COCA codebook"* | `le-contract.md:59`; archive `:23738` |
| F3 | §3 is byte-axis by construction. Sub-byte carvings are LANE readings, never a §3 payload layout | `causal_witness.rs:13-26` (the G24N4 correction) |
| F4 | No cosine; Fisher-z is the normalized cheap LUT; distance is a table read, never a computed cosine | `hexagon-plasticity-v1.md:500-513` (2026-09-15 STORNO); `E-FISHERZ-CANONICAL-COSINE-REPLACEMENT-1` (archive `:22506`) |
| F5 | No float on reasoning paths; float only as derived decode | `cosine-census-CONSOLIDATED.md:33-36` |
| F6 | "Without materialization" = no global rendered-field cache; the per-class metric table is an ingredient, not the avoided cache (ledger invariant 2); a static position off the transmitted-index manifold re-materializes (invariant 1) | `E-PERTURBATION-CONVERGENCE-1` (archive `:21826-21875`) |
| F7 | `CausalMask` 2³: bit enables one SPO plane in distance; L1 Association = `SO`, L2 Intervention = `PO`, L3 Counterfactual = `SPO` | `causal-edge/src/pearl.rs:6-49` |
| F8 | Rung → Pearl level → mask: rungs 0–2 → `0b001` (O, hand-chosen convention), 3–5 → `0b011` (certified), 6–9 → `0b111` (certified); elevation only widens | `cognitive_shader.rs:221-250`; `driver.rs:685-723` |
| F9 | Compose = mask AND: only planes active in both survive | `causal-edge/src/edge.rs:692-694` |
| F10 | `CausalEdge64` layout: `S[0:7] P[8:15] O[16:23] f[24:31] c[32:39] mask[40:42] dir[43:45] infer[46:49]`; v2 reclaims 52–63; `#[repr(transparent)]` | `edge.rs:138-161` |
| F11 | Frequency-rank = ROUTING, ⟂ meaning (ρ ≈ −0.07 vs Jina); count-derived meaning is a coarse floor (8-genre ρ = +0.039 on random pairs) | `E-CAM96-DISTRIBUTION-MEASURED-1` (archive `:21063-21064`) |
| F12 | The SPO 2³ role mask is a homograph-collapse operator: 809/809 verb∩noun homographs collapse to distinct lemma centroids under P vs S/O | `E-SURFACE-FORM-COLLAPSE-1` (archive `:21796-21809`) |
| F13 | Relations are STORED edges; the substrate generalizes analogically; predicate-token arithmetic fails | archive `:21066` |
| F14 | Fisher-z wins rank reads, loses level reads (v3 F11) | v3 §1 |
| F15 | One copy: lance-graph owns the canonical bytes; a second authority is forbidden; borrows never cross a mailbox | `canonical_node.rs:1710-1728`; lgj `CLAUDE.md` one-copy law |
| F16 | Falsifiability rule: every filter an anti-vacuity test; every guard can-fire AND can-stay-silent; every threshold an inertness test | `CLAUDE.md` § falsifiability |
| F17 | Choices on validation, gates reported on eval; eval never changes a selection | v3 §5, §7 (post-review) |
| F18 | The embedding key reaches the harness by environment only; never a file, commit, brief, CI log, command line or echo | v3 ledger 20 |
| F19 | Lance family upstream; every other forked crate via the AdaWorldAPI fork | `CLAUDE.md` P0 |

## 2. Input inventory (file:line, read this session)

### 2.1 The register and its readings (contract)
- `facet.rs:77-135` — `FacetCascade = facet_classid(4) | 6×(8:8)`, 16 B, `#[repr(C, align(16))]`, const-asserted.
- `facet.rs:461-488` — `tier_bytes()` (the 12-unit ladder) and `cascade_byte(shape, group, level)`: *"the same lookup whether the ClassView reads the facet as 6×2, 4×3, or 3×4; the bytes never move."*
- `facet.rs:801-890` — `CascadeShape::{G6D2, G4D3, G3D4}`, `ROTATIONS`, `from_levels`, `index = group·D + level`.
- `facet_schema.rs:21-55` — `FacetSchema::{TierCascade=0, SpoTriplet=1, Pair48=2}`, resolved from `(facet_classid >> 24) & 0b11` ("provisional field position"). **Value 3 is free; L4 has no variant.**
- `awareness_facet.rs:21-65` — `SpoFacet`: six `Palette256Pair = (u8,u8)` rails, 3 SPO + 3 episodic-witness; "similarity between two pairs is one table read … never a float."
- `tekamolo_facet.rs:1-59` — the `G4D3` carving named Te/Ka/Mo/Lo, each lane `256:256:256`; EXPERIMENTAL, not in §3; straddles tier boundaries.
- `causal_witness.rs:28-53, 112-151, 199-219` — `CausalWitnessFacet` = 24 signed i4 loci over 12 B (a LANE reading); `Locus::{Temporal..Contradiction}` incl. `Antecedent = 7` ("relativPronomen → its antecedent"); sign = before/after; 0 = unbound.
- `canonical_node.rs:33-35` — `NodeGuid([u8;16])`; `:786` `pub type EdgeBlock = EdgeFacet`; `:862` `NodeRow`; `:1026` `Tekamolo = 13`, `:1078` `EpisodicBasin = 15`; `:1284-1354` `ValueSchema` (Full carries 13/14/15); `:1411-1423` `TailVariant::{V1,V2,V3}`; `:1454-1483` `ReadMode {tail_variant, value_schema, edge_codec}`; `:1691-1696` `classid_read_mode`.
- `tenants.md:56-58` — tenants 13 (Tekamolo, 16 B facet), 14 (CausalWitness, G24N4, EXPERIMENTAL), 15 (EpisodicBasin, `self_code` 12 B).
- `le-contract.md:50-67` — the L1–L8 catalogue; byte accounting 6×2 = 4×3 = 3×4 = 2×6.

### 2.2 The edge and the ladder
- `causal-edge/src/pearl.rs` (whole file) — `CausalMask`, `pearl_level()`, `simpsons_paradox_risk`.
- `causal-edge/src/edge.rs:8-134` — `InferenceType`, signed mantissa (`to_mantissa`/`from_mantissa`); `:182-242` `pack` (v2: temporal ignored); `:353-413` mask accessors, `matches_causal`; `:692-694` compose mask AND.
- `causal-edge/src/tables.rs:1-120` — `NarsTables`: 256×256 `PackedTruth` deduction + per-c-quantile revision; `w = c/(1−c)`, `c_rev = ws/(ws+1)` (`:80-84`).
- `cognitive_shader.rs:157-250` — `RungLevel` 0–9, `pearl_level`, `causal_mask_bits`; `:272-335` `RungElevator` (threshold 2, hand-tuned, labelled).
- `p64-bridge/src/lib.rs:45-89` — `edge_to_layer_mask` (causal bits → CAUSES/ENABLES/CONTRADICTS planes; inference → plane); `:105` `edges_to_layered_rows`; `:121-128` plane consts; `:383-429` `CognitiveShader::cascade` (per-plane sweep, `per_col_predicates`, `semiring.distance(query,target)`).

### 2.3 The driver
- `driver.rs:1-23` — the seven stages; `:72-111` `ShaderDriver` fields (`planes`, `awareness`, `nars_tables: Option<Arc<NarsTables>>`, `rung_elevator`); `:240-249` rung-widened layer mask; `:263-268` cascade; `:376-387` FreeEnergy from `(top_resonance, std_dev)`; `:454-470` gate; **`:472-497` edge emission: `s_palette = row % 256`, `p = 0`, `o_palette = (row/4) % 256`, `f = c = resonance`, `temporal = cycle_index`** — the edge carries no identity and no prior today.
- `grammar/free_energy.rs:26-35` — `FAILURE_CEILING 0.8`, `HOMEOSTASIS_FLOOR 0.2`, `EPIPHANY_MARGIN 0.05` ("calibrated empirically; adjust once Animal Farm benchmark runs report"); `:104-129` `compose(likelihood, kl) → total = (1−l)+kl`.
- `lib.rs:20-28` — the cycle fingerprint: cache key / retrieval key / replay key / cursor.

### 2.4 The codec side
- `deepnsm-v2/src/space.rs:38` `AxisCodebook = Vec<Vec<f32>>`; `:163` `pub type Cam96 = [u8;12]`; `:181-203` `Cam96Space` (12 axes); `:230-251` `encode`; `:257-272` `distance` (Σ 12 squared-L2); `:14, :35-56, :85` `SemanticSpace` over `PairPalette`.
- `recipe_substrate.rs:88-147` — `PairPalette {basin, identity}`; distance = sum of two axis squared-L2s (`ISS-PAIRPALETTE-IS-TWO-AXES-NOT-A-PAIR`, `ISSUES.md:1-13`).
- `contract/distance.rs:20-69` — `Distance` trait, `similarity_z` (inline atanh clamp ±0.9999), `fisher_z_inverse`, `mean_similarity_fisher`; `:82-108` `[u64;256]` Hamming, `[u8;6]` L1.
- `bgz-tensor/src/fisher_z.rs` — `FamilyGamma`, `FisherZTable::lookup_i8`, no diagonal guard (v3 §4).
- `ndarray/src/hpc/edge_codec.rs:46-69` — `Codebook::train(data, n, dim, k, iters, seed)` (Lloyd's, deterministic, re-seeds empty clusters).
- `episodic_basin.rs:76-144` — `BasinRow.self_code: [u8;12]`, `from_le_bytes`; producer `arigraph/episodic.rs:208-227` writes `self_code: [0;12]` ("left zero").

### 2.5 The COCA fixed points
- `deepnsm/word_frequency/lemmas_5k.csv:1` — header `rank,lemma,PoS,freq,perMil,%caps,%allC,range,disp,blog,web,TVM,spok,fic,mag,news,acad,…PM`; row 1 `the … range 482995, disp 0.98`.
- `deepnsm/src/vocabulary.rs:18-152` — `VOCAB_SIZE 4096`; row index = COCA rank (`:139`); `pos_table`, `freq_table` per rank (`:85-88`); loads ONLY `rank,word,pos,freq` (`:97`); first-PoS-wins on homographs (`:131-135`). **`range`/`disp`/genre columns are loaded nowhere in `deepnsm` or `deepnsm-v2` `src/`** (grep 2026-09-30: no hit on `\bdisp\b` in a loader).
- `deepnsm/src/pos.rs:9-36` — 13 COCA tags, 4 bits.
- `deepnsm-v2/src/lexical.rs:1-42, 285-382, 398-444` — `LexicalEvidence` keeps every `(surface, PoS, lemma)` reading with exact `Option<u64>` counts; refuses conflicts; `WORD_FORMS_HEADER = lemRank,lemma,PoS,lemFreq,wordFreq,word`; API `readings / surface_count / surface_pos_count / lemma_pos_count / lemma_count`.
- `deepnsm-v2/src/vocab.rs:42-87` — `PaletteVocab::from_frequency_ranked`, first id wins, capacity 65 536.
- `deepnsm-v2/src/fsm.rs:38-62` — 7-state `Pos` incl. `Rel` (relativizer → the ±8 antecedent pointer feeder).
- `deepnsm/examples/gridlake_spo_ngrams.rs:1-14` — COCA n-grams are LICENSED, read from a local path, never committed.
- Prior measurements: `E-FREQ-IS-COSINE-REPLACEMENT-1` (8-genre Fisher-z distance ρ = 0.762 on 8 curated NSM pairs; `freq_is_cosine.rs`); `E-SURFACE-FORM-COLLAPSE-1` (`homograph_collapse.rs`); `E-CAM96-DISTRIBUTION-MEASURED-1` (F11, F13).

## 3. The proposed resolution (committed)

### 3.1 The identity facet: L4, two full bytes per subspace
- Per subspace `s ∈ 0..6` (16 d of the 96-d embedding): ONE codebook `C_s` (256 × 16-d, `Codebook::train`, deterministic seed), norms `n_c`, a Fisher-z table `T_s` (256×256 i8 + `FamilyGamma`; diagonal never consulted), and `N_a` = the 255 other centroids ranked by `T_s(a,·)`.
- The rail is **`[a, b]`**: `a` = nearest centroid (the needle, v3 F7), `b` = second-nearest centroid **as a full palette byte**. No `j:t` nibble: F3 forbids a sub-byte key payload; this is v3's arm **A**, and it is L4-conformant by construction.
- Position along the arc is **not stored in the key.** If a measured arm needs it (§5 G-T), it lives in a LANE reading (F3), decided then, never now.
- The facet: `FacetCascade { facet_classid, tiers = [(a_0,b_0) … (a_5,b_5)] }`. The **key** of a word/lemma row (`NodeGuid`, `TailVariant::V3`) carries it; `EpisodicBasin.self_code` is a *reference* to the basin row's key, not a second code (§3.5).
- Reading selection: **`FacetSchema::PalettePair = 3`** (the one free value of the provisional 2-bit field, `facet_schema.rs:48-54`). Selected by `facet_classid`, as F1 requires. The 2-bit field is then full — recorded as RISK-C1 for the cascade savant, not decided here.

### 3.2 The eight readings: one `CausalEdge64` lens per subspace
Each subspace rail reads as an edge — a `Copy` microcopy at read time (data-flow rule 2), never stored:

| plane | value | source |
|---|---|---|
| **O** | `a` (needle) | stored byte |
| **S** | `b` (second pole) | stored byte |
| **P** | `T_s(a,b)` quantized to u8 — the colon: the spread between the poles | table read, derived |
| `f,c` | §3.4 prior, revised | `NarsTables` |
| mask | selected by rung (F8) | `RungElevator::causal_mask_bits` |

The eight `CausalMask` projections ARE the readings (F7):

| mask | reading | v3 name | Pearl |
|---|---|---|---|
| `O` | needle only — the 48-bit point | **D** | Level 1 for rungs 0–2 (F8) |
| `SO` | the pole pair, no spread | **A** (≡ v3's M−t) | Association |
| `PO` | needle + spread, pole implicit | **M−j** | Level 2 |
| `SPO` | needle + pole + spread | **M** | Level 3 |
| `P` | spread alone | **B**'s level read | marginal |
| `S`, `SP`, `None` | free marginals | — | — |

Nothing is ablated. A reading that fails its own gate (§5) is a finding about the bytes; the other readings stand. Compose is mask AND (F9): a needle-only edge composed with a full edge yields needle-only — a reading may coarsen, never invent.

### 3.3 Similarity: table reads over the active planes, Fisher-z domain, no x̂
For two rails `A=[a,b]`, `B=[a',b']` and mask `m`:
```
d_O = z(a, a')                       one T_s read
d_S = z(b, b')                       one T_s read
d_P = |q(T_s(a,b)) − q(T_s(a',b'))|  two reads, one u8 difference   (a LEVEL read — F14 applies; gate G-P)
score_s(m) = Σ_{plane ∈ m} d_plane   summed in the Fisher-z domain (F14)
score(A,B) = Σ_s score_s(m)          six subspaces
```
- No slerp, no sqrt, no division, no reconstruction. Every term is an i8/u8 table read or difference. F4/F5/F6 hold by construction.
- **SPO refinement arm (the perturbation-field render):** `d_SPO+ = d_O + d_S + z(a,b') + z(b,a')` — the cross reads, the two poles' field rendered onto each other's needle. Pre-registered as arm **M+** against `SPO` plane-sum; both are table reads.
- The encoder's merge identity (`t = θ_a/φ`, excess `ε`) survives **offline** as the encoder's diagnostic (G0 reports `ε`); decode never renders it.

### 3.4 The COCA fixed points in the driver (the lever)
The row index is the COCA rank (`vocabulary.rs:139`); rank, PoS, lemma, `freq`, `range`, `disp` and 8 genre `PM` columns are a billion-word prior. They never enter the 12 identity bytes (v3 G4; F11: frequency is routing, not meaning). They land where NARS already lives:

| lever | what changes | where |
|---|---|---|
| **L-1 prior truth** | Each identity row's prior `CausalEdge64` truth: `f = range / N_texts` (probability the word occurs in a random COCA text — a frequency of positive evidence), `c = w/(w+1)` with `w = ln(1 + freq)` (policy pin, labelled). Loaded once from `lemmas_5k.csv` columns `freq, range` (a loader that reads them does not exist — `§2.5`). | `nars_tables` prior; new loader in `deepnsm` beside `Vocabulary::load` |
| **L-2 edge emission carries identity** | `driver.rs:472-497` today packs `S = row%256, P = 0, O = (row/4)%256, f = c = resonance`. Change: `O = a` of the row's L4 rail for the plane selected by the row's PoS role (F12: S/O ← nominal, P ← verbal), `S = b`, `P = q(T_s(a,b))`; `f,c` = the L-1 prior REVISED with the hit's resonance through `NarsTables::revise`. | `driver.rs` stage [5] |
| **L-3 F's KL term is measured** | `FreeEnergy::compose(top_resonance, std_dev)` uses std-dev as the KL surrogate. Change: `kl` = divergence of the dispatch window's observed rank histogram from the COCA prior (the genre prior that minimises it, from the 8 `PM` columns). The three thresholds (0.8/0.2/0.05) stay as pins; G-F measures whether they now separate. | `driver.rs:379-387`; `free_energy.rs` gains a `kl_from_prior` helper, zero-dep |
| **L-4 lemma is the identity** | Identity rows are LEMMA rows (`lemRank`); surface forms route through `LexicalEvidence` readings → `(lemma, PoS)`; the `Antecedent` locus (`causal_witness.rs:131-132`) resolves to a lemma row, and coreference compares two key facets with §3.3. | `deepnsm-v2` routing; no contract change |

Two algebras never merge (F13, v3 SPOFC): structure (the 12 identity bytes, the mask) vs strength (`f,c`). COCA feeds only the second.

### 3.5 Home and one-copy
- The identity lives in the **key** (`NodeGuid`, `TailVariant::V3`, `FacetSchema::PalettePair`).
- `EpisodicBasin.self_code` (tenant 15) today persists 12 zero bytes. Under one-copy (F15) it becomes a **reference**: the basin row's own key carries its L4 identity; `self_code` is read through the key, and D-C96P-7 re-scopes from "shape reading of `self_code`" to "`self_code` ⇒ key reference or removal". `v3-envelope-auditor` decides the mechanism; this spec commits the direction.
- Tekamolo (tenant 13, `G4D3`) and CausalWitness (tenant 14, `24×i4`) are readings of their OWN 12 B, pointing at identity rows. They are not readings of the identity bytes. The `G4D3` rotation of an L4-coded register is structurally possible (`ROTATIONS`) and is asserted meaningless for an L4 class (G-ROT).

### 3.6 Migration and gates on legacy
- `Cam96Space` (12 axes) stays as the legacy reading; `pub type Cam96AxisSpace = Cam96Space` (v3). The L4 pair space is a new type in `lance-graph-contract` (the reading is contract law; deepnsm-v2 ships the codebook artifact). Artifact magics `CAM96P01`/`CAM96PW1` and the digest gate (v3 §6) unchanged; the code-file view `Cam96PairCodes<'a>` unchanged.
- I-LEGACY-API-FEATURE-GATED: an L4 read of a 12-axis-coded register is refused by `FacetSchema` mismatch (magic + classid), never silently reinterpreted.

## 4. Non-goals
- Phrases / n-grams as lemma-one-level-up — the data is licensed and local-only (`gridlake_spo_ngrams.rs:10-12`); a separate plan once an artifact exists.
- Registering Tekamolo in §3 — its own certification (jc pillar) per `tekamolo_facet.rs:40-42`.
- `PairPalette` rename/rebuild — `ISS-PAIRPALETTE-IS-TWO-AXES-NOT-A-PAIR` closes after D-C96P-1.
- The 96-subgenre palette — gitignored; the 8 genres are the floor (F11).
- tesseract-rs migration (D-C96P-8) — after D-7.
- Changing `FAILURE_CEILING`/`HOMEOSTASIS_FLOOR`/`EPIPHANY_MARGIN` — pins stay; G-F measures.

## 5. Pre-registered gates (validation decides, eval reports — F17)

| gate | criterion | can fire / can stay silent |
|---|---|---|
| **G-L4 structural** | The key payload is 12 palette bytes; no type in the pair path exposes a nibble; `compile_fail` doctest that `Cam96` and the pair rail do not mix; `FacetSchema::PalettePair` round-trips through `of_classid` | fires on a `(j:t)` type; silent on `[a,b]` |
| **G-D needle** | `O`-only score recovers the in-harness 48-bit control's ρ within the G3 floor | a shuffled `a` fires |
| **G-A pair** | `SO` ρ − `O` ρ > floor (the second pole carries rank information) | `b` replaced by a random `N_a` member fires |
| **G-P spread** | `P` alone is monotone with the true angle φ (Spearman on validation ≥ 0.9, policy pin) — the F14 level-read test | permuted `T_s` fires |
| **G-M full** | `SPO` ≥ every proper sub-mask on validation; `M+` vs `SPO` reported | — |
| **G-T position** | Report only: ρ of `SPO` + a 4-bit `t` LANE vs `SPO`. If the gain exceeds the floor, a lane reading is specified in a follow-up; the key is unchanged either way | — |
| **G-ROT** | The `G4D3` rotation of L4 bytes has ρ within the floor of zero against Jina (must be MEANINGLESS); a class reading L4 must not also read `G4D3` of the same bytes (structural) | a rotation that ranks fires — that would mean the bytes are not content-blind |
| **G-ε agreement** | Encoder's triangle excess `ε` distribution reported per subspace; `N_a` excludes `a` (v3 ledger 25) | — |
| **G1 fidelity** (v3, kept) | Shipping mask's eval ρ ≥ in-harness 12-axis ρ; precondition 48-bit ≤ 12-axis else HARNESS-SUSPECT | — |
| **G-F prior** | With L-3, F separates in-domain KJV text from word-shuffled text at the existing 0.2/0.8 pins (AUC ≥ 0.75, policy pin); the std-dev surrogate is the control | flat prior (uniform) must NOT separate = stays-silent test |
| **G-PoS** | L-2 plane assignment from PoS vs a PoS permutation: the 2³ projections' function-clustering purity (the 15 used_for groups, archive `:21066`) drops under permutation | — |
| **G-LEM** | Coreference agreement (`Antecedent` locus → row) at lemma identity vs surface identity on the `l9_loci_real_text` harness inputs; lemma ≥ surface | — |
| **G-FC** | Prior `c` from `ln(1+freq)` predicts revision magnitude on a held-out text (words with low prior `c` move more); a flat `c` prior must not | stays-silent half |
| **G-EDGE** | Emitted edges: `O == a` of the row's rail (not `row % 256`); `f ≠ c` in general; disable run restores `row % 256` and the equality test fails | — |
| G4 / G5 / G5b / G6 / G7 / G8 (v3) | unchanged | — |

Every guard: red with it removed, green with it back.

## 6. Per-savant question sets

**S1 prior-art (Opus).**
1. Is an L4 `FacetSchema` selector already specified anywhere (plans, OGAR mint tables, `facet_schema.rs` history)? PRIOR-ART-AT or GAP.
2. Does any shipped code already read the key facet as `(needle, second-nearest)` pairs (e.g. `SpoFacet` consumers, `style_rails_at`)? CONFIRMS/GAP.
3. Is the "prior truth from COCA `range`/`freq`" idea already an E-id or a plan row? Name it or GAP.
4. Does `E-SURFACE-FORM-COLLAPSE-1`'s `homograph_collapse.rs` already implement L-2's role→plane map reusably? PRIOR-ART-AT.
5. Any prior ruling that the rung ladder must NOT select codec readings (i.e. that F8 is dispatch-only)? VIOLATES-with-evidence or CONFIRMS.
6. Duplicate E-ids for "one register, N readings" (#729, A1, A9) that §3.2 must cite instead of restating?

**S2 iron rules.**
1. I-VSA-IDENTITIES: does §3.3 bundle content or compare identities? YIELDS/VIOLATES.
2. I-LEGACY-API-FEATURE-GATED: is every path that could read 12-axis bytes as L4 refused by a gate (§3.6)? Name the ungated path or YIELDS.
3. I-SUBSTRATE-MARKOV: does compose-as-mask-AND (F9) on these lenses touch the bundle algebra? YIELDS/NA.
4. I-NOISE-FLOOR-JIRAK: which §5 thresholds cite a bound and which are labelled pins? List any unlabelled.
5. One-copy (F15): is `self_code` as a key reference (§3.5) a borrow that crosses a mailbox? YIELDS/VIOLATES.
6. AP1–AP9: any anti-pattern in L-2's edge emission change (same fn name, different semantics under a feature)?

**S3 code truth (general-purpose, Read-tool only).**
1. Every `file:line` in §2 — CODED / CLAIMED / ABSENT. Especially `facet_schema.rs:48-54` (the 2-bit field), `driver.rs:472-497`, `vocabulary.rs:139`, `edge.rs:692-694`.
2. Is `FacetSchema` consulted by any live reader today, or only defined? (Determines whether `PalettePair = 3` is behavioural.)
3. Does `RungElevator::causal_mask_bits` reach any distance call in the driver today (`causal_distance`)? CODED or CLAIMED.
4. Are `range`/`disp` truly loaded nowhere? Confirm the negative by opening `deepnsm/src/vocabulary.rs` and `deepnsm-v2/src/lexical.rs` loaders.
5. Does `Antecedent` resolve to a row anywhere (`wave.rs` / `witness_fabric.rs`), or only to an offset?
6. Does `SpoFacet` have any producer that writes `(a,b)` from a codebook, or is it label-only?

**S4 cascade impact.**
1. Every file/test/doc/board row that changes if `FacetSchema::PalettePair = 3` lands (the 2-bit field is then full — who else wanted a slot?).
2. The blast radius of changing `driver.rs` stage [5] edge emission — which tests pin `row % 256`?
3. What breaks in `arigraph/episodic.rs` and `episodic_basin.rs` tests if `self_code` becomes a reference?
4. The mandatory vs follow-up split for the new COCA loader (`range`, `disp`, genre PM).
5. Does the `Cam96 → [u8;12]` alias (`space.rs:163`) leak into any consumer that would now compile against the pair type?
6. Board rows: which STATUS_BOARD D-ids re-scope (D-7), which are new.

**S5 different views (no redesign).**
1. Strongest alternative to `O = a, S = b` (the L1 convention F8 marks hand-chosen) — name it and its second-order consequence; RISK only.
2. Is `P = q(T_s(a,b))` a level read that F14 predicts will fail G-P? What is the honest expectation?
3. Does L-3's KL-from-prior collide with F11 (frequency ⟂ meaning) — i.e. is register detection being mistaken for meaning?
4. Is `M+` (cross reads) a perturbation-field render or a Gram cosine in disguise? Evidence either way.
5. The second-order consequence of lemma-as-identity for named entities (OOV, `is_named_entity`, `vocabulary.rs:49-57`).

## 7. Operator questions (open; not decided by the council)
- Q1 `O = a` vs `S = a` when the L1 probe runs (F8 convention).
- Q2 `self_code`: reference or removal.
- Q3 the embedding key (environment only, F18) and the 96-d embeddings (Q4 of v3, still blocking D-C96P-0).
- Q4 whether the `t` LANE (G-T) is wanted at all if the pair alone clears G-M.

## 8. Board hygiene (same commit as ratification)
`INTEGRATION_PLANS.md` PREPEND; `STATUS_BOARD.md` v4 block (D-C96P-0 re-scoped arms, D-7 re-scoped, new D-C96P-9 COCA loader, D-C96P-10 driver levers L-2/L-3); `AGENT_LOG.md` council entry; `entries/2026-09-30-*.md` for the L4-identity finding + `entries_index.py --write`; `SUPERSESSION-INDEX.md` regenerated LAST; v3 status line → SUPERSEDED-IN-PART (§3, §6.3, §7 decision table).

## 9. Change ledger v3 → v4 (Phase 0)
| # | change | source |
|---|---|---|
| 1 | §3 reading: `[a,b]` two full bytes; `j:t` nibble struck (F3) | operator ruling 3, F3 |
| 2 | Similarity: table reads over active planes, no x̂, no cosine | operator ruling 1, F4–F6 |
| 3 | A/B/D/M: eight `CausalMask` projections; decision table struck; per-reading gates | operator ruling 2, F7–F9 |
| 4 | Home: key facet via `FacetSchema::PalettePair`; `self_code` → reference | operator ruling 3, F1, F15 |
| 5 | Driver levers L-1..L-4 with gates G-F/G-PoS/G-LEM/G-FC/G-EDGE | operator mandate; F11–F13 |
| 6 | G-ROT: content-blindness of L4 bytes under rotation made falsifiable | F1, F16 |
