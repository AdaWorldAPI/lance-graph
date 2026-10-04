# Deleted grammar heuristics — which commit had what, and where it is now

READ BY: anyone touching deepnsm / deepnsm-v2 grammar, TEKAMOLO, loci /
anaphora, the TOC→HHTL tree, WordNet/ontology registers, or ContextChain
replay — BEFORE writing a "new" one. Also read before deleting any of these
again.

Restored 2026-10-04 on operator instruction: *"restore the rest of the
deleted and repurpose or merge the ideas when you think they are superseded
but don't drop and create a document which commit had what so it can be used
for later reference."*

## How to read a row

- **commit**: where the file left the tree. Retrieve the last version with `git show <commit>^:<path>`.
- **born in**: the commits that built it. Read them for the design reasoning.
- **now**: one of
  - RESTORED: the file is back unchanged;
  - RESTORED+REPURPOSED: the file is back, with a stated change;
  - SUPERSEDED-IN-PLACE: the idea lives elsewhere, which is named;
  - HISTORY-ONLY: the code is not restored and the reason is given.
- **measured**: numbers the code itself produced, with where.

## The ledger

| path | lines | commit (date) | born in | what it held | why it was deleted | now |
|---|---|---|---|---|---|---|
| `crates/deepnsm-v2/src/toc.rs` | 426 | `68955ecb` (2026-08-22) | `abd331b9` | **The TOC as an HHTL tree of SoA node addresses.** Book:chapter:verse spawned as the full skeleton before any reasoning (operator: *"bevor wir über das Buch nachdenken wird ein Inhaltsverzeichnis als HHTL-Baum mit SoA-Knoten erstellt"*). Two nibbles per level. | "the hydrated table is rebuilt from scratch" | **RESTORED**. This is the HHTL literal chapter:verse address; the basins below are the cross-chapter nodes. |
| `crates/deepnsm-v2/src/hydrate.rs` | 370 | `68955ecb` | `bfc69bf8`, `6e385c88` | Every SPO triple joined to the verse node it was read in. The reading-order chain is kept (the Markov trajectory over the book) along with the scanpath residue. | as `toc.rs` | **RESTORED**. |
| `crates/deepnsm-v2/src/promote.rs` | 591 | `68955ecb` | `dd728f17`, `ec50f07b` | Basins written as SoA `NodeRow`s at TOC node keys: keyed by literary unit, not by bare subject (`E-HERMENEUTIK-RUNG-LADDER-1`). Also the TEKAMOLO tenant writer. | "also minted V1 keys" | **RESTORED**. The V1 claim was already false at deletion: `ec50f07b` made the promoter refuse the V1 tail (`mint_for` + read-back check, test-pinned). |
| `crates/deepnsm-v2/examples/toc_hydrate.rs` | 315 | `68955ecb` | `6e385c88` | The end-to-end run of tree → hydrate → promote → loci over the real KJV. | as above | **RESTORED**. Measured 2026-10-04: 32,357 nodes (66 books, 31,102 verses), 0 unaddressed. Superseded the same day by the PR #1321 review fix, now that toc_hydrate parses with every reading (`parse_readings`, the same stream as `bible_wave`): 57,277 triples (= `bible_wave`'s `certain` pin), 25,459 cross-verse links (44.4 %), 1,187 basins. The first run used the legacy single-reading parse: 70,393 triples, 1,227 basins. |
| `crates/deepnsm-v2/src/lexicon.rs` | 307 | `68955ecb` | `ec50f07b` | One COCA lexicon: lemma → forms → archaic → Other, monotone. It replaced three hand-copied taggers. | "taggers inlined into the two remaining examples" | **RESTORED**. The inlined copies in `bible_wave` / `genre_shapes` remain (they are pin-guarded). Merging them back onto this module is a deliberate re-pin. Measured in the file: KJV verses with no triple went from 35.6 % to 12.2 % once the forms table was read. |
| `crates/deepnsm-v2/src/loci.rs` | 357 | `68955ecb` | `f47f0528` | Derives `Locus::Antecedent` (a 24×i4 `CausalWitnessFacet` nibble) from the SPO stream. Candidates are non-pronoun subjects within REACH = 8. The ranking is `SelectionalFit` (Cam96 similarity of each candidate to the pronoun clause's predicate) against `Recency`, with a margin gate. | "resolution AND binding already shipped" (`spo_anaphora_nibble`, `l9_loci_real_text`, `probe_antecedent_binder`) | **RESTORED**, with a known defect. It is agreement-blind. On Aesop the shipped agreement resolver scores 0.727 and the agreement-blind rule scores 0.273. **Merge pending:** take the gender/number/case agreement filter (and the German case table, see the research inventory) as the candidate filter, then keep SelectionalFit as the ranker. KJV numbers from `toc_hydrate`: Recency binds 35,025 of 35,613; SelectionalFit binds 27,788 (7,237 thin-margin). |
| `crates/deepnsm-v2/src/tekamolo.rs` | 718 | `f3d42b7c` (style half), `c57cc256` (reverted that revert), `68955ecb` (whole file) | `7bae6c84` | **German** right-corner TEKAMOLO lane reading. A left-corner hypothesis is opened per adverbial and committed at the right corner on margin. It reads Luther 1545 and Elberfelder 1905 as a two-translation control, uses `de/tekamolo.tsv`, and hydrates the Kausal/Modal/Lokal lanes. | "insight_coca_read already emits all four lanes" — but that replacement is **English COCA only**, so the German capability was lost (D-LXC-18 blast radius) | **RESTORED**, with a ruling conflict. `V2StyleProvider` (ThinkingStyle → FieldModulation knobs) is the half §3o of `alpha-channel-rung-overlay-v1` refused for deepnsm-v2: reasoning lives in the planner. **Resolved 2026-10-04 (D-LXC-22):** grammar is not a thinking style, so `read_clause` now takes `ReadParams` (`LEFT_CORNER` / `RIGHT_CORNER` presets). The style mapping is kept as the adapter `ReadParams::from_style`; its destination is the planner, or a thinking dialect on `ogar-loco` (the long-term IR substrate). |
| `crates/deepnsm/src/ontology_vocab.rs` | 141 | `48405aa2` (2026-08-22) | `ae8e762e` (`5ff56bdf` is cited by `48405aa2` but not in this clone) | An OBO ontology concept register (`0x03XX`) over `ogar_codebook::concepts_in_domain`: name ↔ id, with the COCA rank space kept apart. Lookups only; it never used CAM-PQ itself. | "CAM-PQ is prohibited for ontologies", and it sat in v1 when the instruction named v2 | **RESTORED+REPURPOSED**. (1) The OBO half returns empty: `98de36eb` retracted the 14 0x03XX mirror rows as never minted, so the domain is reserved with zero rows, and a test pins that. (2) **New WordNet half:** `WordNetRail` (all senses, all hypernym edges from `wordnet31_isa_v2.tsv`, exact 7-column arity), `ancestors`, `is_a`, and `ConceptMask`, which matches by membership of the synset or any ancestor. Identity only, no distance, so the A1 ruling holds. Tested on real WordNet rows: whale/dolphin under `aquatic_mammal`, shark/manta under `cartilaginous_fish`, and a three-root water-animal mask. |
| `crates/deepnsm/examples/persona_chain_replay.rs` | 376 | `8c03aaa9` (revert of `a74c3bbd`, no reason given) | `a74c3bbd` (not in this clone; its file content survives at `8c03aaa9^`) | The first caller of `ContextChain::disambiguate_with` with REAL candidates (window NP heads), scored against the agreement resolver's gold. Pronoun case gate: Ambiguous never decides. | none recorded | **RESTORED**. Measured 2026-10-04: the margin gate and the rank metric are on incompatible scales. The widest coherence spread is 0.0003 against a threshold of 0.1, 335× too small, so every replay escalates. This is the baseline for the counterfactual "does this make sense" test. |
| `crates/deepnsm/src/markov_soa.rs` | 360 | `9a5f54c1` (2026-05-31) | — | The Markov wave over SoA SPO rows. | layer inversion: a core concern sat in a linguistics sensor | **SUPERSEDED-IN-PLACE** at `crates/lance-graph/src/graph/arigraph/markov_soa.rs` (vocabulary-agnostic, with an injected distance). |
| `crates/deepnsm/src/{content_fp,markov_bundle,trajectory}.rs` | 98 / 250 / 298 | `0ae9f906` (2026-04-24) | D5 | 10K-bit XOR content fingerprints, bit-rotation braiding, Hamming recovery margin. | wrong substrate: GF(2)/XOR where the stack uses ℝ multiply+add (I-VSA-IDENTITIES, CHANGELOG 2026-04-21) | `markov_bundle.rs` and `trajectory.rs` exist again on the ℝ substrate. **`content_fp.rs` is HISTORY-ONLY**, because its `Vsa10k` type was removed from the contract. Its idea (a deterministic per-rank content vector) lives in the ℝ switchboard. |

## Not grammar, listed so they are not mistaken for losses

| path | commit | now |
|---|---|---|
| `crates/lance-graph-planner/examples/data/coca/lexicon.tsv` | `4d00cba3` | moved to a release asset (data, not code) |
| `crates/deepnsm-v2/data/{bible_vocab.txt,cam96_*.bin}` | `73a8745f` | release `v0.1.0-cam96-data` |
| `crates/lance-graph-cognitive/src/learning/cognitive_frameworks.rs` | `6265ac7d` | moved to a standalone crate |
| `crates/jc/examples/ontology_locality_probe.rs` | `0b9860c3` | VSA vision corrected; not restored |
| `crates/lance-graph/examples/p_community_basin_agree.rs` | `5e1393a3` | S1 probe removed with the jc dev-coupling |

## Merge queue (ideas restored but not yet merged)

1. **loci.rs and the agreement resolver.** Add a gender/number/case agreement filter, then keep SelectionalFit as the ranker. Measure against `l9_loci_real_text` gold (agreement-only 0.727).
2. **tekamolo.rs and planner styles.** The grammar path no longer uses styles (D-LXC-22). Open: moving `V2StyleProvider` out of deepnsm-v2 once the planner side or an `ogar-loco` dialect can receive it.
3. **lexicon.rs and the inlined taggers** in `bible_wave` / `genre_shapes`. This is a deliberate KJV re-pin.
4. **persona_chain_replay and ContextChain.** Fix the scale mismatch (a rank thermometer against a 0.1 margin) before the replay can decide anything.
