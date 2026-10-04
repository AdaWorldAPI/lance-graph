# 2026-10-04 — deleted grammar heuristics restored; WordNet identity register (D-LXC-21)

**Status:** VERIFIED-IN-CODE and MEASURED (`cargo test --release` deepnsm-v2: 194 lib tests; deepnsm: 222; `toc_hydrate` and `persona_chain_replay` run on real data, debug-0).

Operator (2026-10-04): restore the deleted grammar heuristics, repurpose or merge the superseded ones, drop none, and write down which commit had what. Ledger: `.claude/knowledge/deleted-grammar-heuristics-ledger.md`.

## Restored

**From `68955ecb`, into deepnsm-v2:**
- `toc.rs`, `hydrate.rs` and `promote.rs`: the KJV TOC as an HHTL tree, triples addressed to verse nodes, and basins promoted at TOC keys.
- `lexicon.rs`.
- `loci.rs`.
- `tekamolo.rs`: the German right-corner lanes.
- the `toc_hydrate` example.

**Into deepnsm:**
- `ontology_vocab.rs`, from `48405aa2`.
- the `persona_chain_replay` example, from `8c03aaa9`.

## Repurposed

**`ontology_vocab.rs` now holds a WordNet identity register:**
- `WordNetRail` reads every sense and every hypernym edge.
- `ConceptMask` covers a sense when the sense or any ancestor is a mask root, so a mask can cut across the taxonomy (e.g. water animals).
- There are no distances, so A1 holds.
- Its OBO half answers empty, because `98de36eb` retracted the 0x03 rows; a test pins that.
- Disable runs:
  - is_a without the ancestor walk → 3 of 5 tests red;
  - every sense collapsed to number 1 → 2 red.

## Measured

**`toc_hydrate` on the KJV:**
- 32,357 nodes (66 books, 31,102 verses);
- 70,393 triples, 0 unaddressed;
- 38.8 % cross-verse links;
- 1,227 basins at TOC keys;
- loci: Recency binds 35,025 of 35,613 pronoun subjects, SelectionalFit binds 27,788.

**`persona_chain_replay`:** the ContextChain margin gate is 335× too coarse for the rank metric (spread 0.0003 vs threshold 0.1), so every replay escalates.

## OPEN

- `tekamolo.rs` still carries `V2StyleProvider`. §3o refused thinking styles in deepnsm-v2; the operator decides whether to keep it here or move it to the planner.
- `loci.rs` is agreement-blind. The shipped agreement resolver scores 0.727 against 0.273 for the agreement-blind rule. Merge queue in the ledger.
