# DisMech × Causality-V3 — repository-grounded rebase report, v1

> **Status: REPORT, no code.** This is the §26 deliverable that gates
> implementation. Every number carries the command or `file:line` that produced
> it. A number I did not measure myself is labelled **claimed, unverified**.
> **Nothing here is a plan to be executed until the operator picks a slice.**

## 0. What this corrects

The briefing's §0 asked that its own numbers be verified before anything is
changed. They were. **Most are stale, and two are wrong in a way that changes
the design.** The corpus census on the board
(`EPIPHANIES.md` `E-DISMECH-CORPUS-CENSUS-1`, 2026-08-20) already measured
much of this a day earlier; that entry is the current truth, not the brief.

## 1. Current implementation inventory

| organ | home | state |
|---|---|---|
| DisMech transcode | `MedCare-rs/crates/medcare-dismech` (3,303 LOC, 4 bins) | SHIPPED, no bake artifact |
| DisMech evidence types | `lance-graph-contract/src/dismech_evidence.rs` (662 LOC, 9 tests) | SHIPPED, **0 callers** |
| CausalEdge64 v2 | `causal-edge/src/layout.rs` + `edge.rs` | SHIPPED, default-on |
| CausalEdgeV3 (12 B) | `causal-edge/src/edge_v3.rs` | **0 production callers** |
| EpisodicWitness64 | — | **NOT A CODE SYMBOL** |
| CausalWitnessFacet (24 × i4) | `lance-graph-contract/src/causal_witness.rs` (787 LOC) | read-wired, **0 production writers** |
| AriGraph | `lance-graph/src/graph/arigraph/` (15 mod, 8,750 LOC, 187 tests) | SHIPPED, example-only callers |
| DeepNSM-v2 | `crates/deepnsm-v2` (5,550 LOC, 108 tests) | workspace-EXCLUDED, **0 dependents** |
| Aerial+ ARM discovery | `crates/lance-graph-arm-discovery` (1,093 LOC, 42 tests) | SHIPPED, workspace-excluded |
| CLAM + CHAODA | `ndarray/src/hpc/clam*.rs` (4,900 LOC, 77 tests) | SHIPPED, one library caller |
| HHTL (`NiblePath`) | `lance-graph-contract/src/hhtl.rs` (1,040 LOC, 24 tests) | SHIPPED + called |
| OCR ingest | `tesseract-rs/crates/tesseract-ogar` | SHIPPED, 14 declared caps |
| DOM ingest | `AdaWorldAPI/spider` `spider_doc_ir` (343 LOC) | on disk, **0 dependents** |
| doc IR | `OGAR/crates/ogar-doc-ir` | SHIPPED, both retinas converge |
| ELK closure | `OGAR/crates/ogar-elk` (1,149 LOC, 18 tests) | **dev-dependency, 0 production calls** |
| HoleV3 / 4 metacog states | — | **ABSENT, zero hits repo-wide** |
| eval harness | — | **ABSENT** |

## 2. Confirmed vs stale — every §0 claim adjudicated

| §0 claim | verdict | measured |
|---|---|---|
| "~99.9996% parity" | **STALE — string absent from the repo** | `grep -rl "99.9996" MedCare-rs` → 0 hits. Real records: **99.4%** (1,848/1,860 *diseases*, self-labelled "historical … not reproducible") and **0.009%** edge drift measured in a *different* repo (`/workspace/dismech-rs`) |
| ~1,996 disease YAMLs | **STALE** | **1,968** @`557e154` |
| ~124 mechanism modules | **STALE** | **123** |
| 586 registered identities | **STALE** | **580** measured; 585 in a doc comment; 586 in a plan |
| 33,455 causal edges "in the Rust bake" | **NOT-FOUND** | there is **no bake** — no artifact, no `bakes.tsv` row; the number is a `dismech_census` stdout line on the vanished 1,996-file corpus |
| MONDO ~99.6% / HP ~99.8% / UBERON 100% | **claimed, unverified** | prose only, no method, no artifact. Board measures MONDO at **4.4% of edge endpoints** |
| 100% pathophysiology identity coverage | **contradicted by the code** | `bake.rs:15-18`: nodes without `conforms_to` "stay address-`0` — F3 tier 2 is **not yet wired**" |
| four buckets are "explicitly represented" | **CONFIRMED in data, ABSENT in Rust** | see §3 |

**Three corpora, three counts. The ordering invariant holds; the absolutes do not.**

| bucket | brief §0 | board (upstream, 2,100 files) | measured @`557e154` (1,968 files) |
|---|---:|---:|---:|
| DIRECT | 8,313 | 9,073 | **8,058** |
| INDIRECT_KNOWN_INTERMEDIATES | 3,869 | 3,978 | **3,825** |
| INDIRECT_UNKNOWN_INTERMEDIATES | 4,250 | 4,539 | **4,150** |
| UNKNOWN | 371 | 408 | **361** |
| total | 16,803 | 17,998 | **16,394** |

Measured two independent ways that agree byte-for-byte: a raw token census
(`grep -rhoE 'causal_link_type:[[:space:]]*[A-Z_]+' kb/disorders/*.yaml`) and a
structural per-edge parse. **F5.0's first act is pinning a corpus revision** —
the corpus lives at `/workspace/dismech`, outside every repo, with no checksum,
no fetch script and no `bakes.tsv` row. The 1,996-file snapshot the brief's
numbers came from no longer exists on this filesystem. That has already cost
five stale claims once.

## 3. DisMech parity — exact status

- **What is compared:** `medcare-dismech/src/parity.rs:113-121`, seven dimensions
  — node-id set, edge multiset on `(source,target,predicate)`, per-edge
  `hypothesis_groups` / `causal_link_type` / `intermediate_mechanisms`,
  `orphan_targets`, node `evidence_count`. Semantic, not byte-exact.
- **Harness:** `dismech_parity <kb/disorders> <pathographs>`; oracle present at
  `/workspace/dismech/pathographs/` (1,870 entries). Zero Python in the lane.
- **It is not a gate.** No CI reference; `parity.rs` and `graph.rs` contain
  **zero `#[test]`**. It is a binary someone must remember to run, against an
  oracle no script fetches.
- **`CausalLinkType` does not exist as a Rust type.** Zero hits across all
  `.rs`. The carrier is `graph.rs:92 pub causal_link_type: Option<String>` —
  unvalidated, an unknown string passes silently — and `bake.rs:343-347`
  writes it with `{:?}`, so the TSV contains `Some("DIRECT")`, not an ordinal.
  The typed version exists **on the other side of the boundary**:
  `contract::dismech_evidence::DismechTopology` (`:56`), fail-closed.

### ★ 3a. The finding that resizes the benchmark

**Only 2,449 of 3,825 (64.0%) `INDIRECT_KNOWN_INTERMEDIATES` edges name an
intermediate.** 1,376 (36.0%) carry the label and no mediator.

Measured independently three ways, all agreeing:
- my line-oriented pass: **2,449** oracle edges, **534** disorder files, 3,714 mediator strings;
- `intermediate_mechanisms:` key occurrences: **2,525** = 2,449 + 74 + 1 + 1 (exactly the per-bucket split), 0 inline empty lists;
- `dismech_evidence.rs:155-176` at a narrower scope (`pathophysiology[].downstream[]`): **1,347 of 3,844 (35.0%)** empty, usable oracle **2,497 tuples across 539 diseases**.

Consequences:
1. The supervision corpus is **2,449 edges over 534 diseases**, not 3,869.
2. A 20% hold-out is **≈490 edges / ≈107 diseases**, not 774.
3. `dismech-missing-links-v1.md` Gate W1.1 asserts `known_links.tsv == 3.869`
   and **cannot pass on any corpus revision**. Its own rule is *"stoppen und
   melden, nicht die Zahl anpassen"* — hence this report rather than an edit.
4. **74 `INDIRECT_UNKNOWN_INTERMEDIATES` edges DO name intermediates** — a
   source contradiction the four-label taxonomy does not anticipate. They must
   be excluded from the restraint control or they read as hallucinated closure
   by the benchmark's own definition.

### 3b. Leakage, measured — lower than feared, and a lower bound

Exact-match (lowercased) mediator strings across diseases:
**3,298 distinct · 84 (2.5%) appear in >1 disease · reuse 1.03×**
(histogram: 3,214 in one disease, 62 in two, 18 in three, 4 in ≥4).

So mediators are overwhelmingly disease-specific: a random-edge split is less
contaminated than §11 feared, and disease-held-out is feasible at 534 groups.
**This is a LOWER BOUND** — exact string match cannot see near-duplicates
(`"Impaired X"` vs `"impaired X synthesis"`), and the board already measured
that lexical-variant problem on phenotype labels. Split C
(mechanism-family-held-out) still has to be built and reported separately.

## 4. CausalEdge64 / V3 layouts — and the free space question

`causal-edge` is workspace-EXCLUDED; `default = ["causal-edge-v2-layout"]`
(`Cargo.toml:18`) and **no dependent opts out**, so v2 is active everywhere.

| bits | field |
|---|---|
| 0–23 | s_idx / p_idx / o_idx (u8 each) |
| 24–39 | NARS frequency u8 / confidence u8 |
| 40–42 | causal_mask (Pearl 2³) |
| 43–45 | direction triad |
| 46–49 | **inference mantissa, signed i4** |
| 50–52 | plasticity |
| 53–58 | **w_slot (witness corpus root, 6 b)** |
| 59–60 | **truth-band lens (2 b)** |
| 61–63 | **spare (3 b)** |

A compile assert pins full coverage (`layout.rs:94-111`). **There are zero
unallocated bits.** §16's "do not overload CE64" is arithmetic, not preference.

**Three lenses over the SAME bits 59–60** — `TrustTexture` (`:141`) and
`CausalTopology` (`:239`, `Direct/IndirectKnown/IndirectUnknown/Unknown`) are
ordinal-identical *by design*, and the doc warns historical provenance is NOT
guaranteed for old rows (`:217-227`). `ReasoningBand` reads 61–63 (`:353`).

**The 11 high bits read 0 on every production edge.** Only two production
producers exist — `cognitive-shader-driver/src/driver.rs:483` and
`lance-graph-planner/src/cache/nars_engine.rs:488` — both call
`CausalEdge64::pack`, which under v2 carries `let _ = temporal;` with the
comment *"v2: temporal is IGNORED … Writing the v1 temporal here would corrupt
the reclaim zone; silently drop it."* `with_topology` and `with_reasoning_band`
have **zero call sites anywhere**.

**`CausalEdgeV3`** (`edge_v3.rs:96`, 12 B const-asserted): MO freq/conf ·
KA mask+direction / mantissa+plasticity · LO target u16 (SPO deduped to the
node's CAM-PQ facet) · anaphora nibble · TE temporal i8 · byte 8 w_slot+truth
RAW · byte 9 spare RAW · bytes 10–12 reserved. **≈28 free bits, only 2
contiguous reserved bytes.** Zero production callers. `rehydrate` restores the
RAW mantissa via `set_inference_mantissa` because `InferenceType` is lossy on
**8 of 16** states.

**Stale doc, measured:** `causal-edge/src/lib.rs:9-15` still prints the **v1**
diagram as the crate headline, contradicting `layout.rs` and the default feature.

### 4a. EpisodicWitness64 — the brief's §16 preserves a one-sided distinction

**It is not a code symbol.** Four hits repo-wide, all one comment block.
`soa_view.rs:272`: *"`EpisodicWitness64` is NOT YET a code symbol (a queued
design — see EPIPHANIES `E-EW64-IS-PREDICTIVE-PREFETCH`; the shipped seeds are
the 6-bit W-slot `CausalEdge64` + `WitnessTable<64>`/`WitnessEntry` +
`arigraph::{episodic,witness_corpus}`)."* Note one of the three seeds is the
same organ §3 assigns to AriGraph. Whether EW64 becomes a tenant is gated by
`ew64-witness-unification-v1.md` F0: *"is this genuinely missing canonical
information, or a container minted to avoid completing the address/mask
transition?"*

## 5. Hole-related code — ABSENT, unambiguously

`HoleV3`, `KnownUnknown`, `UnknownKnown`, `UnknownUnknown`, `awareness_state`,
`unknown_kind`: **zero hits repo-wide**, `.rs` and `.md` alike. The hyphenated
prose form appears in exactly **three comments** — `probe_babel_stances.rs:432`,
`recipe_kernels.rs:2372`, `dismech_evidence.rs:64` — i.e. three subsystems name
the concept and each re-derives it locally. That is the argument for a contract
type rather than a per-crate enum.

Nearest real carriers, none of which is it:
- `nars/tactics.rs:88 ReasoningGap` + `GapKind` — SHIPPED, but a transient
  "this tactic cannot proceed" record in the NARS arena: planner-local, not
  addressable, not persistable.
- `sensorium.rs:87 HealingType::InferMissingLinks` — a graph-healing *action*.
- `dismech_evidence::DismechTopology::IndirectUnknownIntermediates` — the
  closest thing to hole *evidence*: a 2-bit source ordinal, not a hole object.
- **`SeekMediator` exists only in a doc string** (`dismech_evidence.rs:142,146`).

The reasoning behind that doc string is worth preserving verbatim in any design:
`mediator_unresolved()` fires for `IndirectUnknownIntermediates` and
deliberately **not** for `Unknown`, because `A → ? → B` establishes the mediator
ROLE while `A ? B` does not — dispatching there would *"MINT a schema slot the
source never asserted."* That is §9's `unknown_kind` axis, already argued
correctly, with no type behind it.

**`ProvenanceTier::{Curated,Extracted,ArmDiscovered,Ratified,Conjecture}`** —
listed on STATUS_BOARD as shipped deliverable D-ARM-1 — appears in **12 files,
none of them `.rs`**. It is spec-only. §20's "emit research hypotheses, not
discoveries" must mint that ladder, not consume it.

## 6. DeepNSM-v2 — the §2 ruling is not implemented

The ruling "DeepNSM-v2 resolves causal grammar; it constrains what can occupy a
missing semantic position" has **no code behind it**. Census over
`crates/deepnsm-v2/src/**`: `admissible` **0**, `constrain` **0**,
`selectional` **0**, `valence` **0** (measured). No function in the crate
accepts a hole — `Option<WordId>` appears once, as a dictionary miss.

What DOES exist: `parse_to_spo(tokens: &[Tagged]) -> Vec<Spo>` (`fsm.rs:118`),
a 3-state transducer over **pre-tagged** tokens — there is no tagger and no
parser; tagging happens only in examples via a 24-word hand list plus
`ends_with("eth")`. `Spo` is three `u16` **surface-form indices**
(`vocab.rs:70-87`) — the predicate is a surface verb id, not a typed relation.
`Copula::transits()` (`belief.rs:71`) is the one relation-type gate, and it
gates a *composition* after both premises are bound.

It is not a search engine either — no top-k, no kNN anywhere in `src/`. The
semantic surface is two pairwise similarity functions.

**Evidence base is stale.** Every published bible_wave number
(23,145 verses / 31,327 triples / 606 subjects / 63.3% beyond ±5) came from a
run truncated at the Old Testament — `corpus.rs:24` names the constant
`KJV_OLD_TESTAMENT_VERSES = 23_145` as *"the exact truncation point of the
historical bug."* Verified dating: `probes/README.md` last touched **2026-07-23**
(`ed50d8a4`); the fix landed **2026-08-04** (`6953061e`). Nothing was re-run.
`lib.rs:31` still prints the bugged count. And the "63.3% beyond ±5" is a
same-subject **recurrence-gap histogram** (`bible_wave.rs:264-279`) — a fair
claim about what a ±5 ring forfeits, not a grammar or causality measurement.

**No ontology path at all** — zero biomedical/OBO/classid terms, zero ontology
contract imports. Workspace-excluded; **zero crates depend on it**; the only
integration is a hand-run TSV between two examples. The "18k codebook" is
**DocuScope**, a Python probe filter (`probes/README.md:15`) — no such codebook
exists. Real vocabularies: **12,543** (v2, fetched) and **4,096** (v1, capped
from 5,051 committed rows).

**Reusable as-is:** the FSM shape, the copula transit gate, the derivation
arena's soundness gates, the Cam96 loader. **Must be built:** a tagger, typed
relation identity, a slot/valence vocabulary, and the admissibility computation.

## 7. AriGraph — implemented, essentially unwired

`crates/lance-graph/src/graph/arigraph/` — 15 modules, 8,750 LOC, **187 tests**,
no feature gate.

| organ | entry | callers |
|---|---|---|
| basin (Louvain + Leiden refine) | `community.rs:108 communities()` | examples only |
| episodic basins (union-find) | `episodic.rs:243 basins()` | examples only |
| PPR (HippoRAG-style) | `ppr.rs:114 personalized_pagerank()` | examples only |
| BM25 (Okapi) | `bm25.rs:61/138` | examples only |
| **RRF (Cormack 2009)** | `rrf.rs:64 reciprocal_rank_fusion()` | **ZERO — verified** |
| Markov SoA wave | `markov_soa.rs:172 project()` | examples only |
| episodic chain / theses / paths | `episodic.rs:345/384`, `triplet_graph.rs:208` | examples only |

`OsintRetriever::retrieve` (`retrieval.rs:235`) — the one composed entry point —
runs BFS + truth filter + episodic top-k and **ignores every one of them**.
`rrf.rs:22-25` says so itself: wiring it in *"stays gated on the G0
load-bearing verdict — this module only lands the algorithm."*

**Correction to a doc claim:** `doc_graph.rs:27` asserts
`impl DocGraphQuery for TripletGraph`. **That impl does not exist** — the three
hits are the prose line, a `///` doc-example, and a `#[cfg(test)]` mock.

**"Use as-is, do not rebuild" is right on availability, wrong on wiring.**
Calling `reciprocal_rank_fusion` is the cheapest real integration in this
inventory.

## 8. HHTL / CLAM / CHAODA

**CHAODA is NOT name-only — it is implemented twice, and the right one must be
called.** ndarray `ClamTree::anomaly_scores` (`clam.rs:1618`),
`flag_anomalies` (`:1674`), `ensemble_anomaly_scores` (`:1720`, Ishaq 2021,
`ENSEMBLE_GRAPH_BUDGET = 4096`). The crate states its own limit at
`clam.rs:2864-2866`: single-method LFD scoring reaches **ROC-AUC ≈ 0.62 on a
synthetic mixture — the easiest case — against its own ≥ 0.85 bar**, because
LFD measures intra-leaf geometry, not inter-leaf isolation. §19 must therefore
call `ensemble_anomaly_scores`, never `anomaly_scores`. A CHAODA-lite also
exists (`perturbation-sim/src/chaoda.rs:72`, single kNN scorer,
workspace-excluded). CLAM is **77 tests** across four files (56/7/6/8) — the
"46 tests" figure in the repo's own CLAUDE.md is stale.

**§19's actual capability is ABSENT.** No structural-hole / open-triad /
should-close-but-does-not detector exists in either repo. CHAODA scores
**point-level isolation**, which is a different capability. The nearest usable
substrate is a pair that already exists and is never compared: `Communities`
(structural) × `EpisodicBasins` (experiential). `community.rs:16-20` writes the
semantics — *"a community that crosses basins = a discovered bridge the
episodic history has not yet captured"* — and **no function takes both.**

### ★ 8a. HHTL is populated — in the artifact nobody cites

The board says HHTL is *"zero on every baked row in both production bakes."*
Measured on the pinned `.soa` bytes, that is true for the two it names and
**false as a general claim**:

| artifact | rows | HHTL ≠ 0 |
|---|---:|---:|
| `obo-core.soa` | 68,797 | **0 (0.00%)** |
| `spine.soa` | 7,641 | **0 (0.00%)** |
| `all-lanes.soa` | 770,360 | **164,031 (21.29%)** |

Per-lane in `all-lanes.soa`: **MONDO 0x0301 100% · HPO 0x0302 100% ·
UBERON 0x0303 100% · PATO 0x0304 100% · ICD-10-GM 100% · OMIM 100% ·
OPS 98.9% · Orphanet 67.6%**; LOINC/CUI/RxNorm/FMA ≈ 0%.

**The five OBO namespaces are exactly the ones DisMech grounds against.** So
the §12 ladder's HHTL level is blocked only if a level reads `obo-core.soa`,
and available if it reads `all-lanes.soa`. **Which artifact a ladder level
reads is a first-class decision.**

#### ⊘ 2026-08-21 CORRECTION — §8a measured ONE of TWO readings

Operator: *"Obo HHTL ist meines Wissens mit zipper bereits indirekt hydriert."*
Verified, and the section above is incomplete a second time. It measured the
**cascade tiers** (row bytes 4..10). `rails::HhtlMode::of_row` **prefers the
RailHead reading** and falls back to Cascade only when the rail register is
empty — so the cascade census is the *fallback* path, not the primary one.

The Zipper hierarchy lives in four value-slab registers (`rails.rs:130-147`,
value-relative, `value` starting at row byte 32): `is_a` 44..56, `part_of`
56..68, `is_a_cont` 68..80, `part_of_cont` 80..92 — a 12-byte primary plus a
12-byte continuation concatenated into ONE logical `RailPath` of
`RAIL_DEPTH = 24`. Measured:

| lane | rows | cascade | rail `is_a` | median DN depth | `part_of` | cont |
|---|---:|---:|---:|---:|---:|---:|
| MONDO | 32,095 | 32,095 | **32,094 (99.997%)** | 6 | 0 | 90 |
| HPO | 19,836 | 19,836 | **19,835 (99.995%)** | 7 | 0 | 45 |
| UBERON | 14,975 | 14,975 | **14,973 (99.99%)** | 8 | **8,525** | 129 |
| PATO | 1,887 | 1,887 | 1,886 | 5 | 0 | 0 |
| ICD-10-GM | 16,905 | 16,905 | 16,649 | 3 | 0 | 0 |
| OPS | 38,956 | 38,544 | 38,544 | 4 | 0 | 0 |
| ATC | 6,897 | 6,896 | 6,896 | 5 | 0 | 0 |
| **Orphanet** | 20,809 | 14,063 | **0** | — | 0 | 0 |
| **OMIM** | 18,712 | 18,712 | **0** | — | 0 | 0 |
| CUI / RxNorm / FMA / LOINC | 599,244 | ~94 | 0 | — | 0 | 0 |

`obo-core.soa` and `spine.soa` are zero on **both** readings (0/68,797 and
0/7,641 across all four slabs) — so the earlier finding holds for those two
artifacts and is simply not the whole hydration story.

**Three things this adds that were not visible from the cascade census:**

1. **Cascade and rail are INDEPENDENT, not redundant.** Orphanet (14,063
   cascade rows) and OMIM (18,712) carry **zero** Zipper DN. A consumer reading
   through `HhtlMode` gets the cascade arm there; a consumer expecting prefix
   containment on a `RailPath` gets depth 0 — silently. Two registers read as
   one feature is a real asymmetry, and it is per-lane.
2. **Mereology is scoped exactly where the code says.** `part_of` is UBERON
   only (8,525), matching `graph_feed.rs:730` — *"UBERON only (the zipper is
   scoped there)"*. So the mereology axis is a single-namespace capability, not
   a general one.
3. **The continuation slab is load-bearing, not hypothetical.** 264 rows exceed
   12 levels (UBERON 129, MONDO 90, HPO 45), so the two-register split carries
   real depth and `rails::read`'s concatenation is exercised by production data.

**Consequence for §12, revised:** the HHTL rung is **available at ~100% for
MONDO / HPO / UBERON / PATO via the RailHead reading**, with genuine taxonomy
depths (median 6 / 7 / 8 / 5) — and those are precisely the namespaces DisMech
grounds against. What a ladder level must now declare is not only its artifact
but its **reading**. For Orphanet/OMIM the honest answer is narrower than
"unavailable regardless" (CodeRabbit-corrected, 2026-08-21): `HhtlMode::of_row`
falls back to the Cascade arm when RailHead is empty, and both lanes carry
**full cascade coverage** (Orphanet 14,063/14,063 of its cascade-bearing rows,
OMIM 18,712/18,712) — so a consumer reading through `HhtlMode` gets the rung
there. It is unavailable ONLY for a consumer that specifically needs
`RailPath` prefix containment (rail depth 0 on both lanes); that consumer's
requirement, not the rung itself, is what excludes Orphanet/OMIM.

Operator-ruled context: `MedCare-rs docs/RAIL_OFFENE_POSTEN.md` Posten 1 is
**ENTSCHIEDEN 2026-08-12** — *"Legacy-Read-Mode `rails::HhtlMode`, Version-Gate
pro ZEILE (Register als Zeuge)"*, and *"Damit ist der Re-Bake für Posten 1
nicht mehr blockiert."* The two-reading design is deliberate. (Note MedCare's
`CLAUDE.md` still points at that Posten as *"der HHTL-Offset, der den Re-Bake
blockiert"* — stale against its own ledger.)

## 9. Ingestion readiness — three links exist, one is missing

1. **Acquisition — EXISTS.** Pixels: `tesseract-ogar::OcrExecutor::execute`
   (`src/lib.rs:686`), 14 caps count-asserted in OGAR
   (`ogar-vocab/src/ocr_actions.rs:143`). DOM: `spider_doc_ir::harvest`
   (`spider_doc_ir/src/lib.rs:105`) — on disk at `/home/user/adaworldapi/spider`
   @`046c439`, but **no repo depends on it**; `lance-graph-osint`'s
   `ingest_url` (`pipeline.rs:42`) is the only wired web path, and `spider` is
   an opt-in dep that is off by default.
2. **Perceptual IR — EXISTS, and both retinas converge.**
   `doc.v1` (`structured.rs:302`) → `ogar_from_docv1::from_doc_v1` →
   `ogar_doc_ir::DocIr`. `RegionKind` is a closed 7-variant vocabulary; `Rail`
   is *"one byte per axis … a 256×256 tile … NEVER a `u16` (canon)"* — the doc
   IR is already addressed in the V3 rail vocabulary.
3. **★ Semantic grounding — MISSING. This is the break.**
   `grep -i "MONDO|SNOMED|CURIE|TermId|ontolog"` over all of
   `tesseract-rs/crates/` → **zero hits**. `ogar-doc-ir`'s `resolve`/`project`
   are ClassView-mask projections, not identity lookups. `osint::Triplet` is
   three `String`s from verb-pattern matching. `tesseract-ogar`'s
   `ResolvedTriple` is three lemma `String`s.
4. **Ontology binding — EXISTS but is not reachable from (3).**
   `TermId::parse` + `render_classid` (`ogar-obo/src/lib.rs:233,140`),
   `GoldenBakeIndex::resolve` (`medcare-dismech/src/identity.rs:112`).
5. **Persistence — the only shipped doc landing is `document 0x080B`**, keyed
   by **sha256 of the raw bytes**, not by any semantic identity
   (`medcare-core/src/document.rs:33-40`), behind two default-off features.
   OGAR W4 `ogar-doc` does not exist; the `semantic_id` mint that would close
   the loop is explicitly deferred
   (`DOC-IR-SPIDER-CONVERGENCE-PLAN.md:177-181`).

**Wikidata (§7): fixtures only.** `wikidata_hhtl.rs:144` returns **10
hand-written classes**; its own header defers the 115M streaming load. The
`wikidata_landing` test is behind an off-by-default feature. No dump reader, no
SPARQL client, no QID→classid mint.

## 10. Ontology identity + roundtrip

**Baked and measured** (SHA-256 of every artifact verified against
`data/config/bakes.tsv`): MONDO 32,095 · HPO 19,836 · UBERON 14,975 ·
PATO 1,887 · RO 4 nodes + 17 predicates · CUI 266,579 · FMA 104,709 ·
LOINC 103,291 · RxNorm 124,665 · SNOMED as lanes (484,031 concepts /
2,053,329 edges). **GO, CL, NCIT are CURIE-recognized but have 0 baked rows**
(`identity.rs:32-33` classes them `NoNamespaceLane` *by design*).
**ChEBI is explicitly WITHDRAWN** (`classid_census.rs:72-77`).
**lance-graph itself carries zero biomedical terms** — its `Medical/*` context
ids are annotated "BioPortal stub".

**Reuse-don't-mint is real:** `obo-core.soa` 68,797 rows / 68,797 distinct
addresses / **0 duplicates**; `all-lanes.soa` 770,360 / 770,360 distinct / 0
collisions.

**But the roundtrip proof exists for exactly one lane.**
`icd10gm.rs:445 every_real_code_round_trips_bijectively` iterates the full
16,905-code catalog, asserts injectivity, and has a can-stay-silent twin. The
OBO roundtrip tests are **synthetic 3-node fixtures**, and
`medcare-dismech/src/identity.rs` — the resolver §22 depends on — has **0 tests**
and its only exerciser needs an off-repo corpus.

**ELK is a dev-dependency with zero production call sites.** `ogar-elk` is a
genuine 1,149-LOC, zero-dependency reasoner (`meet_via`, `most_specific`,
`fillers_closed`, `entails`) — every caller is an `examples/` file, and the one
manifest entry is `medcare-cohorts/Cargo.toml:109` under `[dev-dependencies]`.
A *different* reasoner IS in production: `ogar-obo::reason::{ancestors,
anatomy_of, phenotypes_of}` via `medcare-first-thought/src/obo.rs:27`. But EL
`saturate` (`reason.rs:171`) is example-only, and production ancestry is
hand-rolled in **five** places.

**§22 needs a third outcome.** The board measured that some endpoints are
*"qualified mechanism PROPOSITIONS … genuinely DisMech-local, and must not be
bullied into an ontology node."* So the identity gate is not match/mint — it is
**match / mint / legitimately-unaddressable**, and the third is itself an
epistemic state (`unknown_kind = REPRESENTATION`).

**A measured UNKNOWN_KNOWN generator:** 1,169 labels exist in more than one
namespace, 1,129 of them HP+MONDO; 26.2% of resolved phenotype labels landed in
MONDO rather than HP; 23.7% are genuinely ambiguous. *"A collapsed resolver
picks by insertion order, silently reattaching a phenotype edge to a same-named
disease node."*

## 11. The smallest possible held-out benchmark

**It needs zero new carriers.** That is the point: it is buildable today, and
it is the falsifier §11 demands before anything else is allowed to matter.

- **Corpus pin.** `/workspace/dismech` @`557e15436` → a checksummed row in
  `data/config/bakes.tsv` + a `scripts/fetch-*` entry. Without this every
  number below decays silently, as five claims already have.
- **Frozen oracle TSV.** `(disease, source, target, mediators[])` for the
  **2,449** edges that are BOTH `INDIRECT_KNOWN_INTERMEDIATES` AND name ≥1
  mediator. Parsed through `DismechTopology::from_source` (fail-closed
  `Option`), never a string compare. Excluded and counted separately: the
  1,376 label-without-mediator rows.
- **Frozen restraint TSV.** The **4,150** `INDIRECT_UNKNOWN_INTERMEDIATES`
  rows, **minus the 74** that name mediators. Plus the **361** `UNKNOWN` rows
  kept as a *third* arm — `dismech_evidence.rs:181-186` keeps them apart at a
  ~10.7× population difference *"that a merged predicate would hide."*
- **Splits, reported separately, never pooled.** (A) random edge;
  (B) disease-held-out over **534** groups; (C) mechanism-family-held-out;
  (D) gene/pathway-family; (E) rare-tail. B is the honest headline; A is the
  optimistic bound.
- **Metrics.** Recall@{1,3,5,10} + MRR on the oracle arm; **abstention rate on
  the restraint arm as a co-equal headline**, not a footnote. Report
  candidate-set reduction and per-split N.
- **LEVEL 0 = structural only**, no ontology, no DeepNSM, no LLM. If Level 0
  cannot beat a frequency-prior baseline, nothing above it is interpretable.

**Two-sided by construction:** a run that scores well on the oracle arm and
also "recovers" mediators on the restraint arm has failed, not succeeded.

## 12. Minimal PR sequence, with falsifiers

Named to the repo's existing conventions (`D-*` on STATUS_BOARD), **not** the
brief's F5.x — per §25's own instruction.

| id | scope | falsifier |
|---|---|---|
| **D-CV3-0** | Pin the corpus (checksum + fetch entry). Emit the three frozen TSVs from a typed parse. **No new types.** | Re-running on a fresh container reproduces 2,449 / 4,076 / 361 **exactly**, or the pin is not a pin |
| **D-CV3-1** | Splits A + B as committed artifacts (534 disease groups) | Group-disjointness test: no disease appears in both sides of B. Anti-vacuity: held-out share is 15–25%, not ~0 |
| **D-CV3-2** | Level-0 scorer: Recall@K + MRR + abstention. Structural only | Must produce a NON-trivial number on both arms; a scorer that abstains always, or never, fails its own can-fire/can-stay-silent pair |
| **D-CV3-3** | `HoleV3` as `ValueTenant = 16`, `awareness_state` ⟂ `unknown_kind`. **Not in CE64 — it has zero free bits.** **BLOCKED, not merely gated on the benchmark** (CodeRabbit-corrected, 2026-08-21): `ValueTenant` currently ends at `CausalWitness = 14`; `BoardAggregates = 15` is only a GATED RESERVATION on `STATUS_BOARD.md`, its own width still open. The discriminant-to-`VALUE_TENANTS` index requires contiguous descriptors, so `HoleV3 = 16` cannot land until the `BoardAggregates` mint is completed and resolved — that mint is an explicit prerequisite of this row, not implied by D-CV3-0..2 alone | Field-isolation matrix per I-LEGACY-API-FEATURE-GATED; `ENVELOPE_LAYOUT_VERSION` unchanged (292 B headroom); a round-trip that proves the two axes are independent; **AND** the contiguity assertion over `VALUE_TENANTS` passes with `BoardAggregates` resolved at 15 |
| **D-CV3-4** | The producer: `dismech_evidence` → hole rows. Gives the 662-line module its first caller | Before: 0 populated rows. After: 4,076 + 361. If it stays 0, the substrate has gained a fifth EXISTS-UNCALLED carrier |
| **D-CV3-5** | `Communities` × `EpisodicBasins` cross-validation — the only real unknown-unknown detector available | Must fire on a synthetic bridge AND stay silent on a coherent graph. Both halves non-trivial |
| **D-CV3-6** | Call `reciprocal_rank_fusion` in `OsintRetriever::retrieve` (cheapest real integration; gated on G0) | Fused ranking differs from BFS-only on ≥1 real query, or RRF is decoration |

**Ordering rule:** D-CV3-0..2 must be green before D-CV3-3 exists, **AND**
the `BoardAggregates = 15` mint must be completed and resolved first — the
discriminant sequence is contiguous, so `HoleV3 = 16` has no valid slot while
15 is still an open reservation. A carrier minted before its benchmark is a
fifth entry in the EXISTS-UNCALLED column, which is the single most repeated
shape in this inventory (CE64 high bits, CausalEdgeV3, CausalWitnessFacet,
dismech_evidence — four independent layers, all read-rich and write-empty).

**What this report does NOT authorize:** any DeepNSM-v2 grammar claim (§6
measured it absent), any HHTL ladder level that does not name its artifact
(§8a), and any use of the 4,150 restraint rows as *negative examples* — they
are a control, and treating absence as falsehood is an iron falsifier.
